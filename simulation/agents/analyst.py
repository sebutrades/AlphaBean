"""
simulation/agents/analyst.py — Analyst agent: evaluates individual setups.

Two tiers:
  - Bulk evaluation via Qwen3 (local, free) for all setups
  - Top N setups get deeper evaluation via Haiku

Returns a verdict (CONFIRMED/DENIED/CAUTION) with reasoning for each setup.
"""
import asyncio
import json
from typing import Optional

from backend.scoring.multi_factor import ScoredSetup
from simulation.agents.base import call_ollama, call_anthropic, AgentResponse
from simulation.config import SimConfig


ANALYST_SYSTEM = """\
You are a senior quantitative trader evaluating a trade setup in a simulation.
Your job is to decide whether to take this trade.

DECISION RULES:
- Composite Score >= 55 with R:R >= 1.5: lean CONFIRMED unless there is a clear structural reason not to
- Composite Score >= 65: should almost always be CONFIRMED
- Composite Score < 40: should be DENIED
- Only use CAUTION when the score is 40-55 AND the setup is genuinely ambiguous

Be decisive. In a simulation, taking good-enough trades and managing risk is better than waiting for perfect setups.

RESPOND IN JSON ONLY:
{
  "verdict": "CONFIRMED" | "DENIED" | "CAUTION",
  "confidence": 0.0 to 1.0,
  "score_delta": -15 to 15,
  "bull_case": "why this works",
  "bear_case": "why this fails",
  "reasoning": "3-5 sentence synthesis",
  "key_factors": ["factor1", "factor2", "factor3"]
}"""


def _build_analyst_prompt(scored: ScoredSetup, regime: str = "unknown") -> str:
    """Build the evaluation prompt from a scored setup."""
    setup = scored.setup
    rr = setup.risk_reward_ratio

    lines = [
        f"SETUP: {setup.pattern_name} on {setup.symbol}",
        f"Bias: {setup.bias.value.upper()} | Strategy: {setup.strategy_type}",
        f"Entry: ${setup.entry_price:.2f} | Stop: ${setup.stop_loss:.2f} | "
        f"T1: ${setup.target_1:.2f} | T2: ${setup.target_2:.2f}",
        f"R:R = {rr:.1f} | Confidence: {setup.confidence:.0%}",
        f"Composite Score: {scored.composite_score:.0f}/100",
        f"Market Regime: {regime}",
        "",
        "SCORING BREAKDOWN:",
        f"  Pattern confidence: {scored.pattern_confidence_score:.0f}",
        f"  Feature score: {scored.feature_score:.0f}",
        f"  Strategy score: {scored.strategy_score:.0f}",
        f"  Regime alignment: {scored.regime_alignment_score:.0f}",
        f"  Backtest edge: {scored.backtest_edge_score:.0f}",
        f"  Volume confirm: {scored.volume_confirm_score:.0f}",
        f"  R:R quality: {scored.rr_quality_score:.0f}",
    ]

    if setup.description:
        lines.extend(["", f"Pattern description: {setup.description[:300]}"])

    return "\n".join(lines)


async def evaluate_setup_ollama(
    scored: ScoredSetup,
    regime: str = "unknown",
    model: str = "qwen3:8b",
    timeout: int = 120,
) -> dict:
    """Evaluate a single setup via Ollama (free, local)."""
    prompt = _build_analyst_prompt(scored, regime)
    resp = await call_ollama(
        prompt=prompt,
        model=model,
        system=ANALYST_SYSTEM,
        max_tokens=4096,
        timeout=timeout,
    )

    if resp.success:
        return {
            "symbol": scored.setup.symbol,
            "pattern": scored.setup.pattern_name,
            "verdict": resp.data.get("verdict", "CAUTION"),
            "confidence": resp.data.get("confidence", 0.5),
            "score_delta": resp.data.get("score_delta", 0),
            "reasoning": resp.data.get("reasoning", ""),
            "bull_case": resp.data.get("bull_case", ""),
            "bear_case": resp.data.get("bear_case", ""),
            "key_factors": resp.data.get("key_factors", []),
            "model": model,
            "elapsed": resp.elapsed,
        }

    return {
        "symbol": scored.setup.symbol,
        "pattern": scored.setup.pattern_name,
        "verdict": "CAUTION",
        "confidence": 0.5,
        "score_delta": 0,
        "reasoning": f"Evaluation failed: {resp.error}",
        "model": model,
        "elapsed": resp.elapsed,
    }


async def evaluate_setup_haiku(
    scored: ScoredSetup,
    regime: str = "unknown",
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """Evaluate a single setup via Haiku (deeper analysis)."""
    prompt = _build_analyst_prompt(scored, regime)
    resp = await call_anthropic(
        prompt=prompt,
        system=ANALYST_SYSTEM,
        model=model,
        max_tokens=1024,
        temperature=0.3,
    )

    if resp.success:
        return {
            "symbol": scored.setup.symbol,
            "pattern": scored.setup.pattern_name,
            "verdict": resp.data.get("verdict", "CAUTION"),
            "confidence": resp.data.get("confidence", 0.5),
            "score_delta": resp.data.get("score_delta", 0),
            "reasoning": resp.data.get("reasoning", ""),
            "bull_case": resp.data.get("bull_case", ""),
            "bear_case": resp.data.get("bear_case", ""),
            "key_factors": resp.data.get("key_factors", []),
            "model": model,
            "elapsed": resp.elapsed,
            "tokens": resp.input_tokens + resp.output_tokens,
        }

    return {
        "symbol": scored.setup.symbol,
        "pattern": scored.setup.pattern_name,
        "verdict": "CAUTION",
        "confidence": 0.5,
        "score_delta": 0,
        "reasoning": f"Evaluation failed: {resp.error}",
        "model": model,
        "elapsed": resp.elapsed,
    }


async def evaluate_batch(
    setups: list[ScoredSetup],
    config: SimConfig,
    regime: str = "unknown",
) -> list[dict]:
    """Evaluate a batch of setups: bulk via Ollama, top N via Haiku.

    Returns list of verdict dicts sorted by adjusted score.
    """
    if not setups:
        return []

    semaphore = asyncio.Semaphore(config.max_concurrent_api_calls)
    results = []

    # Bulk: evaluate all via Ollama
    async def eval_with_sem(scored, fn, **kwargs):
        async with semaphore:
            return await fn(scored, **kwargs)

    tasks = [
        eval_with_sem(
            s, evaluate_setup_ollama,
            regime=regime, model=config.analyst_bulk_model,
            timeout=config.ollama_timeout,
        )
        for s in setups
    ]
    bulk_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(bulk_results):
        if isinstance(r, Exception):
            results.append({
                "symbol": setups[i].setup.symbol,
                "pattern": setups[i].setup.pattern_name,
                "verdict": "CAUTION", "confidence": 0.5,
                "score_delta": 0, "reasoning": f"Error: {r}",
                "composite_score": setups[i].composite_score,
            })
        else:
            r["composite_score"] = setups[i].composite_score
            results.append(r)

    # Apply score_delta and sort
    for r in results:
        r["adjusted_score"] = r["composite_score"] + r.get("score_delta", 0)

    results.sort(key=lambda x: x["adjusted_score"], reverse=True)

    # Top N: re-evaluate with Haiku for deeper analysis
    top_n = config.analyst_top_n
    if top_n > 0:
        top_setups = results[:top_n]
        # Find matching ScoredSetup objects
        setup_map = {(s.setup.symbol, s.setup.pattern_name): s for s in setups}

        haiku_tasks = []
        for r in top_setups:
            key = (r["symbol"], r["pattern"])
            if key in setup_map:
                haiku_tasks.append(
                    eval_with_sem(
                        setup_map[key], evaluate_setup_haiku,
                        regime=regime, model=config.analyst_top_model,
                    )
                )

        if haiku_tasks:
            haiku_results = await asyncio.gather(*haiku_tasks, return_exceptions=True)
            for i, hr in enumerate(haiku_results):
                if isinstance(hr, Exception):
                    continue
                # Replace bulk result with Haiku result (richer analysis)
                hr["composite_score"] = results[i]["composite_score"]
                hr["adjusted_score"] = hr["composite_score"] + hr.get("score_delta", 0)
                results[i] = hr

    # Re-sort after Haiku updates
    results.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)
    return results
