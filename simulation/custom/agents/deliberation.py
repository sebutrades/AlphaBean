"""
simulation/custom/agents/deliberation.py — Agent deliberation framework.

Controls how agents discuss and reach decisions. Three modes:

  QUICK:    Single analyst → PM selects → Risk reviews
            Fast, cheap. ~3-7 API calls per scan.

  STANDARD: Analyst evaluates → Senior analyst reviews & can override →
            PM selects with full reasoning → Risk reviews
            Balanced. ~7-12 API calls per scan.

  THOROUGH: 3 independent analysts vote → Strategist synthesizes →
            PM deliberates with full context → Risk detailed review
            Maximum insight. ~12-20 API calls per scan.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable

from backend.scoring.multi_factor import ScoredSetup
from simulation.agents.base import call_anthropic, call_ollama, AgentResponse
from simulation.custom.config import CustomSimConfig, DeliberationConfig


def _is_local_model(model: str) -> bool:
    """Check if a model string refers to a local Ollama model vs Anthropic API."""
    return "claude" not in model.lower()


async def _call_model(
    prompt: str, system: str, model: str,
    max_tokens: int = 1024, temperature: float = 0.3,
) -> AgentResponse:
    """Route to Ollama or Anthropic based on model name."""
    if _is_local_model(model):
        return await call_ollama(
            prompt=prompt, system=system, model=model,
            max_tokens=max_tokens, temperature=temperature,
        )
    else:
        return await call_anthropic(
            prompt=prompt, system=system, model=model,
            max_tokens=max_tokens, temperature=temperature,
        )


@dataclass
class AgentMessage:
    """A single message in the agent deliberation log."""
    agent: str          # "analyst_1", "analyst_2", "senior_analyst", "pm", "risk", "strategist"
    action: str         # "EVALUATE", "REVIEW", "SELECT", "APPROVE", "REJECT", etc.
    symbol: str
    pattern: str
    content: dict       # full response data
    model: str
    elapsed: float
    tokens: int = 0


@dataclass
class DeliberationResult:
    """Outcome of a full deliberation cycle."""
    approved_trades: list[dict]      # [{symbol, pattern, verdict, reasoning, size_modifier}]
    messages: list[AgentMessage]     # full reasoning trail
    api_calls: int
    total_tokens: int
    total_elapsed: float


class DeliberationEngine:
    """Orchestrates multi-agent discussion based on deliberation mode."""

    def __init__(self, config: CustomSimConfig, emit: Optional[Callable] = None):
        self.config = config
        self.delib = config.deliberation
        self.models = config.models
        self.emit = emit  # optional event emitter for live streaming
        self._messages: list[AgentMessage] = []
        self._api_calls = 0
        self._total_tokens = 0

    async def deliberate(
        self,
        candidates: list[ScoredSetup],
        portfolio_context: dict,
        regime: str = "unknown",
    ) -> DeliberationResult:
        """Run the full deliberation pipeline on a set of candidates.

        Args:
            candidates: Scored setups to evaluate (pre-filtered by strategy + score)
            portfolio_context: Dict with cash, positions, heat, cumulative_pnl, etc.
            regime: Market regime string
        """
        self._messages = []
        self._api_calls = 0
        self._total_tokens = 0
        t0 = __import__("time").time()

        if self.delib.mode == "quick":
            result = await self._quick_pipeline(candidates, portfolio_context, regime)
        elif self.delib.mode == "thorough":
            result = await self._thorough_pipeline(candidates, portfolio_context, regime)
        else:
            result = await self._standard_pipeline(candidates, portfolio_context, regime)

        return DeliberationResult(
            approved_trades=result,
            messages=self._messages,
            api_calls=self._api_calls,
            total_tokens=self._total_tokens,
            total_elapsed=round(__import__("time").time() - t0, 2),
        )

    # ── Quick Pipeline ───────────────────────────────────────────────────────

    async def _quick_pipeline(
        self, candidates: list[ScoredSetup], ctx: dict, regime: str
    ) -> list[dict]:
        """Single analyst → PM → Risk. Fast and cheap."""
        # Step 1: Analyst evaluates each candidate
        verdicts = await self._analyst_evaluate_all(candidates, regime, analyst_id="analyst")

        confirmed = [v for v in verdicts if v["verdict"] == "CONFIRMED"]
        if not confirmed:
            return []

        # Step 2: PM selects
        selected = await self._pm_select(confirmed, ctx, regime)
        if not selected:
            return []

        # Step 3: Risk review
        approved = await self._risk_review(selected, ctx, regime)
        return approved

    # ── Standard Pipeline ────────────────────────────────────────────────────

    async def _standard_pipeline(
        self, candidates: list[ScoredSetup], ctx: dict, regime: str
    ) -> list[dict]:
        """Analyst → Senior review → PM → Risk. Balanced depth."""
        # Step 1: Initial analyst evaluation
        verdicts = await self._analyst_evaluate_all(candidates, regime, analyst_id="analyst")

        # Step 2: Senior analyst reviews all verdicts (can override)
        verdicts = await self._senior_review(verdicts, candidates, regime)

        confirmed = [v for v in verdicts if v["verdict"] == "CONFIRMED"]
        if not confirmed:
            return []

        # Step 3: PM selects with full reasoning context
        selected = await self._pm_select(confirmed, ctx, regime)
        if not selected:
            return []

        # Step 4: Risk review
        approved = await self._risk_review(selected, ctx, regime)
        return approved

    # ── Thorough Pipeline ────────────────────────────────────────────────────

    async def _thorough_pipeline(
        self, candidates: list[ScoredSetup], ctx: dict, regime: str
    ) -> list[dict]:
        """3 analysts vote → Strategist synthesizes → PM → Risk."""
        # Step 1: Three independent analyst evaluations (in parallel)
        all_verdicts = await asyncio.gather(
            self._analyst_evaluate_all(candidates, regime, analyst_id="analyst_1"),
            self._analyst_evaluate_all(candidates, regime, analyst_id="analyst_2"),
            self._analyst_evaluate_all(candidates, regime, analyst_id="analyst_3"),
        )

        # Step 2: Synthesize votes per candidate
        merged = self._merge_analyst_votes(all_verdicts, candidates)

        # Step 3: Strategist tiebreaker on contested candidates
        if self.delib.include_strategist:
            contested = [m for m in merged if m.get("contested")]
            if contested:
                merged = await self._strategist_review(merged, contested, regime)

        confirmed = [v for v in merged if v["verdict"] == "CONFIRMED"]
        if not confirmed:
            return []

        # Step 4: PM deliberates with full voting context
        selected = await self._pm_select(confirmed, ctx, regime)
        if not selected:
            return []

        # Step 5: Detailed risk review
        approved = await self._risk_review(selected, ctx, regime)
        return approved

    # ── Shared Agent Calls ───────────────────────────────────────────────────

    async def _analyst_evaluate_all(
        self, candidates: list[ScoredSetup], regime: str, analyst_id: str
    ) -> list[dict]:
        """Have one analyst evaluate all candidates."""
        tasks = [
            self._analyst_evaluate_one(scored, regime, analyst_id)
            for scored in candidates
        ]
        return await asyncio.gather(*tasks)

    async def _analyst_evaluate_one(
        self, scored: ScoredSetup, regime: str, analyst_id: str
    ) -> dict:
        """Single analyst evaluation of one candidate."""
        setup = scored.setup
        prompt = self._build_analyst_prompt(scored, regime)
        system = (
            "You are a senior quantitative trader evaluating a trade setup. "
            "Give a thorough, balanced assessment. Good setups should be CONFIRMED. "
            "Bad setups should be DENIED. CAUTION is for genuinely mixed cases.\n\n"
            "RESPOND IN JSON ONLY:\n"
            '{"verdict": "CONFIRMED"|"DENIED"|"CAUTION", "confidence": 0.0-1.0, '
            '"bull_case": "why this works", "bear_case": "why this fails", '
            '"reasoning": "3-5 sentence synthesis", "key_factors": ["f1","f2","f3"], '
            '"size_recommendation": 0.5-1.5}'
        )

        self._emit_thinking(analyst_id, setup.symbol, setup.pattern_name,
                           f"{analyst_id} evaluating {setup.pattern_name} on {setup.symbol}...")

        resp = await _call_model(
            prompt=prompt, system=system,
            model=self.models.analyst,
            max_tokens=1024,
            temperature=self.delib.analyst_temperature,
        )
        self._api_calls += 1
        self._total_tokens += resp.input_tokens + resp.output_tokens

        verdict_data = {
            "symbol": setup.symbol,
            "pattern": setup.pattern_name,
            "score": scored.composite_score,
            "scored": scored,
            "verdict": resp.data.get("verdict", "CAUTION") if resp.success else "CAUTION",
            "confidence": resp.data.get("confidence", 0.5) if resp.success else 0.5,
            "reasoning": resp.data.get("reasoning", resp.error or "") if resp.success else f"Error: {resp.error}",
            "bull_case": resp.data.get("bull_case", "") if resp.success else "",
            "bear_case": resp.data.get("bear_case", "") if resp.success else "",
            "key_factors": resp.data.get("key_factors", []) if resp.success else [],
            "size_recommendation": resp.data.get("size_recommendation", 1.0) if resp.success else 1.0,
            "analyst_id": analyst_id,
        }

        msg = AgentMessage(
            agent=analyst_id, action="EVALUATE",
            symbol=setup.symbol, pattern=setup.pattern_name,
            content=verdict_data, model=self.models.analyst,
            elapsed=resp.elapsed, tokens=resp.input_tokens + resp.output_tokens,
        )
        self._messages.append(msg)
        self._emit_verdict(analyst_id, verdict_data)

        return verdict_data

    async def _senior_review(
        self, verdicts: list[dict], candidates: list[ScoredSetup], regime: str
    ) -> list[dict]:
        """Senior analyst reviews all verdicts and can override."""
        verdicts_text = "\n".join(
            f"  {v['symbol']} — {v['pattern']}: {v['verdict']} (conf={v['confidence']:.0%}) "
            f"— {v['reasoning'][:150]}"
            for v in verdicts
        )

        prompt = (
            f"MARKET REGIME: {regime}\n\n"
            f"JUNIOR ANALYST VERDICTS ({len(verdicts)}):\n{verdicts_text}\n\n"
            "Review each verdict. Override any you disagree with. "
            "Focus on cases where the junior analyst may have missed something."
        )
        system = (
            "You are a SENIOR quantitative analyst reviewing a junior's work. "
            "You can CONFIRM or OVERRIDE each verdict. Only override when you have "
            "a strong reason — don't second-guess for the sake of it.\n\n"
            "RESPOND IN JSON ONLY:\n"
            '{"reviews": [{"symbol": "SYM", "original_verdict": "X", '
            '"final_verdict": "CONFIRMED"|"DENIED"|"CAUTION", '
            '"override_reason": "only if changed", "confidence": 0.0-1.0}]}'
        )

        self._emit_thinking("senior_analyst", "", "", "Senior analyst reviewing all verdicts...")

        resp = await _call_model(
            prompt=prompt, system=system,
            model=self.models.strategist,  # senior gets a better model
            max_tokens=1024,
            temperature=self.delib.analyst_temperature,
        )
        self._api_calls += 1
        self._total_tokens += resp.input_tokens + resp.output_tokens

        if resp.success and resp.data.get("reviews"):
            review_map = {r.get("symbol", ""): r for r in resp.data["reviews"]}
            for v in verdicts:
                review = review_map.get(v["symbol"])
                if review and review.get("final_verdict"):
                    old = v["verdict"]
                    v["verdict"] = review["final_verdict"]
                    v["confidence"] = review.get("confidence", v["confidence"])
                    if old != v["verdict"]:
                        v["reasoning"] = f"OVERRIDDEN ({old}→{v['verdict']}): {review.get('override_reason', '')} | Original: {v['reasoning']}"

                        self._messages.append(AgentMessage(
                            agent="senior_analyst", action="OVERRIDE",
                            symbol=v["symbol"], pattern=v["pattern"],
                            content={"from": old, "to": v["verdict"], "reason": review.get("override_reason", "")},
                            model=self.models.strategist,
                            elapsed=resp.elapsed, tokens=0,
                        ))

        return verdicts

    def _merge_analyst_votes(
        self, all_verdicts: list[list[dict]], candidates: list[ScoredSetup]
    ) -> list[dict]:
        """Merge 3 independent analyst evaluations into consensus verdicts."""
        merged = []
        for i, scored in enumerate(candidates):
            votes = []
            for analyst_verdicts in all_verdicts:
                if i < len(analyst_verdicts):
                    votes.append(analyst_verdicts[i])

            if not votes:
                continue

            # Count verdicts
            verdict_counts = {}
            for v in votes:
                vd = v.get("verdict", "CAUTION")
                verdict_counts[vd] = verdict_counts.get(vd, 0) + 1

            # Majority wins
            majority = max(verdict_counts, key=verdict_counts.get)
            is_contested = verdict_counts.get(majority, 0) < len(votes)  # not unanimous

            avg_confidence = sum(v.get("confidence", 0.5) for v in votes) / len(votes)
            all_reasoning = " | ".join(
                f"[{v.get('analyst_id', '?')}] {v.get('reasoning', '')[:100]}"
                for v in votes
            )

            merged.append({
                "symbol": scored.setup.symbol,
                "pattern": scored.setup.pattern_name,
                "score": scored.composite_score,
                "scored": scored,
                "verdict": majority,
                "confidence": round(avg_confidence, 2),
                "reasoning": f"VOTE: {dict(verdict_counts)} — {all_reasoning}",
                "contested": is_contested,
                "vote_counts": verdict_counts,
                "size_recommendation": sum(v.get("size_recommendation", 1.0) for v in votes) / len(votes),
            })

        return merged

    async def _strategist_review(
        self, all_merged: list[dict], contested: list[dict], regime: str
    ) -> list[dict]:
        """Strategist breaks ties on contested candidates."""
        contested_text = "\n".join(
            f"  {c['symbol']} — {c['pattern']}: Votes={c['vote_counts']} | {c['reasoning'][:200]}"
            for c in contested
        )

        prompt = (
            f"REGIME: {regime}\n\n"
            f"CONTESTED SETUPS (analysts disagree):\n{contested_text}\n\n"
            "Break the tie for each. Consider market context and risk."
        )
        system = (
            "You are the Chief Strategist making final calls on contested trades. "
            "RESPOND IN JSON ONLY:\n"
            '{"decisions": [{"symbol": "SYM", "final_verdict": "CONFIRMED"|"DENIED", '
            '"reasoning": "why this call"}]}'
        )

        self._emit_thinking("strategist", "", "", f"Strategist resolving {len(contested)} contested setups...")

        resp = await _call_model(
            prompt=prompt, system=system,
            model=self.models.strategist,
            max_tokens=1024,
            temperature=0.2,
        )
        self._api_calls += 1
        self._total_tokens += resp.input_tokens + resp.output_tokens

        if resp.success and resp.data.get("decisions"):
            dec_map = {d["symbol"]: d for d in resp.data["decisions"]}
            for m in all_merged:
                if m["symbol"] in dec_map:
                    dec = dec_map[m["symbol"]]
                    m["verdict"] = dec.get("final_verdict", m["verdict"])
                    m["reasoning"] = f"STRATEGIST: {dec.get('reasoning', '')} | {m['reasoning']}"

                    self._messages.append(AgentMessage(
                        agent="strategist", action="TIEBREAK",
                        symbol=m["symbol"], pattern=m["pattern"],
                        content=dec, model=self.models.strategist,
                        elapsed=resp.elapsed, tokens=0,
                    ))

        return all_merged

    async def _pm_select(
        self, confirmed: list[dict], ctx: dict, regime: str
    ) -> list[dict]:
        """Portfolio Manager selects trades from confirmed setups."""
        max_trades = self.config.max_trades_per_scan

        if len(confirmed) <= max_trades:
            # All confirmed, no PM decision needed
            for c in confirmed:
                self._emit_verdict("pm", {
                    "symbol": c["symbol"], "pattern": c["pattern"],
                    "verdict": "PM_AUTO_SELECTED", "reasoning": "Only confirmed setup(s)",
                })
            return confirmed

        setups_text = "\n".join(
            f"  {c['symbol']} — {c['pattern']} | Score: {c['score']:.0f} | "
            f"Confidence: {c['confidence']:.0%} | Size rec: {c.get('size_recommendation', 1.0):.1f}x\n"
            f"    {c['reasoning'][:200]}"
            for c in confirmed
        )

        prompt = (
            f"PORTFOLIO: Cash: ${ctx.get('cash', 0):,.0f} | "
            f"Positions: {ctx.get('positions', 0)}/{ctx.get('max_positions', 10)} | "
            f"Heat: {ctx.get('heat_pct', 0):.1f}% | Cum P&L: ${ctx.get('cumulative_pnl', 0):+,.0f}\n"
            f"Regime: {regime}\n\n"
            f"CONFIRMED SETUPS ({len(confirmed)}):\n{setups_text}\n\n"
            f"Select the BEST {max_trades} trade(s). Consider diversification and conviction."
        )
        system = (
            f"You are a Portfolio Manager. Select up to {max_trades} trade(s) from confirmed setups. "
            "RESPOND IN JSON ONLY:\n"
            '{"selected": [{"symbol": "SYM", "reasoning": "why this one", "size_modifier": 1.0}], '
            '"skipped_reasoning": "why others were skipped"}'
        )

        self._emit_thinking("pm", "", "", f"PM choosing from {len(confirmed)} confirmed setups...")

        resp = await _call_model(
            prompt=prompt, system=system,
            model=self.models.portfolio_manager,
            max_tokens=1024,
            temperature=self.delib.pm_temperature,
        )
        self._api_calls += 1
        self._total_tokens += resp.input_tokens + resp.output_tokens

        if not resp.success:
            # Fallback: take top by score
            return confirmed[:max_trades]

        selected_syms = {s.get("symbol", "") for s in resp.data.get("selected", [])}
        pm_mods = {s.get("symbol", ""): s for s in resp.data.get("selected", [])}

        result = []
        for c in confirmed:
            if c["symbol"] in selected_syms:
                pm_info = pm_mods.get(c["symbol"], {})
                c["pm_reasoning"] = pm_info.get("reasoning", "")
                c["size_modifier"] = max(0.5, min(2.0, pm_info.get("size_modifier", c.get("size_recommendation", 1.0))))
                result.append(c)
                self._emit_verdict("pm", {
                    "symbol": c["symbol"], "pattern": c["pattern"],
                    "verdict": "PM_SELECTED",
                    "reasoning": c["pm_reasoning"],
                })

        return result if result else confirmed[:max_trades]

    async def _risk_review(
        self, selected: list[dict], ctx: dict, regime: str
    ) -> list[dict]:
        """Risk Manager reviews selected trades."""
        trades_text = "\n".join(
            f"  {s['symbol']} — {s['pattern']} | Size: {s.get('size_modifier', 1.0):.1f}x | "
            f"Score: {s['score']:.0f}\n    {s.get('pm_reasoning', s.get('reasoning', ''))[:150]}"
            for s in selected
        )

        prompt = (
            f"PORTFOLIO RISK STATE:\n"
            f"  Heat: {ctx.get('heat_pct', 0):.1f}% / {self.config.sizing.max_heat_pct}%\n"
            f"  Positions: {ctx.get('positions', 0)} / {self.config.sizing.max_positions}\n"
            f"  Cum P&L: ${ctx.get('cumulative_pnl', 0):+,.0f}\n"
            f"  Cash: ${ctx.get('cash', 0):,.0f}\n"
            f"  Regime: {regime}\n\n"
            f"PROPOSED TRADES ({len(selected)}):\n{trades_text}\n\n"
            "Review each trade. APPROVE, REDUCE size, or REJECT."
        )
        system = (
            "You are the Risk Manager. Protect capital. Enforce limits.\n"
            "Rules: Max heat not exceeded, no duplicate symbols, reduce during drawdowns.\n\n"
            "RESPOND IN JSON ONLY:\n"
            '{"decisions": [{"symbol": "SYM", "action": "APPROVE"|"REDUCE"|"REJECT", '
            '"size_modifier": 1.0, "reasoning": "why"}], '
            '"portfolio_risk_note": "overall risk assessment"}'
        )

        self._emit_thinking("risk", "", "", f"Risk Manager reviewing {len(selected)} trades...")

        resp = await _call_model(
            prompt=prompt, system=system,
            model=self.models.risk_manager,
            max_tokens=1024,
            temperature=self.delib.risk_temperature,
        )
        self._api_calls += 1
        self._total_tokens += resp.input_tokens + resp.output_tokens

        if not resp.success:
            # Fallback: approve all (mechanical checks happen in engine)
            return selected

        dec_map = {d.get("symbol", ""): d for d in resp.data.get("decisions", [])}
        approved = []

        for s in selected:
            dec = dec_map.get(s["symbol"], {"action": "APPROVE"})
            action = dec.get("action", "APPROVE").upper()

            if action == "REJECT":
                self._emit_verdict("risk", {
                    "symbol": s["symbol"], "pattern": s["pattern"],
                    "verdict": "RISK_REJECTED",
                    "reasoning": dec.get("reasoning", ""),
                })
                continue

            if action == "REDUCE":
                s["size_modifier"] = max(0.25, min(
                    s.get("size_modifier", 1.0),
                    dec.get("size_modifier", 0.5),
                ))

            s["risk_reasoning"] = dec.get("reasoning", "")
            approved.append(s)

            self._emit_verdict("risk", {
                "symbol": s["symbol"], "pattern": s["pattern"],
                "verdict": f"RISK_{action}",
                "reasoning": dec.get("reasoning", ""),
            })

        return approved

    # ── Prompt Building ──────────────────────────────────────────────────────

    def _build_analyst_prompt(self, scored: ScoredSetup, regime: str) -> str:
        setup = scored.setup
        return (
            f"SETUP: {setup.pattern_name} on {setup.symbol}\n"
            f"Bias: {setup.bias.value.upper()} | Strategy: {setup.strategy_type}\n"
            f"Entry: ${setup.entry_price:.2f} | Stop: ${setup.stop_loss:.2f} | "
            f"T1: ${setup.target_1:.2f} | T2: ${setup.target_2:.2f}\n"
            f"R:R = {setup.risk_reward_ratio:.1f} | Confidence: {setup.confidence:.0%}\n"
            f"Composite Score: {scored.composite_score:.0f}/100\n"
            f"Market Regime: {regime}\n\n"
            f"SCORING BREAKDOWN:\n"
            f"  Pattern confidence: {scored.pattern_confidence_score:.0f}\n"
            f"  Feature score: {scored.feature_score:.0f}\n"
            f"  Strategy score: {scored.strategy_score:.0f}\n"
            f"  Regime alignment: {scored.regime_alignment_score:.0f}\n"
            f"  Backtest edge: {scored.backtest_edge_score:.0f}\n"
            f"  Volume confirm: {scored.volume_confirm_score:.0f}\n"
            f"  R:R quality: {scored.rr_quality_score:.0f}"
        )

    # ── Event Emission ───────────────────────────────────────────────────────

    def _emit_thinking(self, agent: str, symbol: str, pattern: str, message: str):
        if self.emit:
            self.emit({
                "type": "agent_thinking",
                "agent": agent,
                "symbol": symbol,
                "pattern": pattern,
                "message": message,
            })

    def _emit_verdict(self, agent: str, data: dict):
        if self.emit:
            self.emit({
                "type": "agent_verdict",
                "agent": agent,
                **data,
            })
