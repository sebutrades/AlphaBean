"""
simulation/agents/coordinator.py — Orchestrates the daily agent flow.

Routes data between Analyst → Portfolio Manager → Risk Manager,
manages async execution, and returns final trade selections.
"""
import asyncio
from typing import Optional

from backend.scoring.multi_factor import ScoredSetup
from simulation.agents.analyst import evaluate_batch
from simulation.agents.portfolio_manager import select_trades as pm_select
from simulation.agents.risk_manager import review_trades as risk_review
from simulation.config import SimConfig
from simulation.portfolio import PortfolioState
from simulation.timeline import TimelineManager


class AgentCoordinator:
    """Orchestrates the full agent pipeline for each simulation day."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.day_logs: list[dict] = []  # reasoning trail for reports

    async def select_trades(
        self,
        setups: list[ScoredSetup],
        portfolio: PortfolioState,
        timeline: TimelineManager,
        date: str,
    ) -> list[ScoredSetup]:
        """Full agent pipeline: Analyst → PM → Risk → final selection.

        Returns list of ScoredSetup objects that passed all agent gates.
        """
        if not setups:
            return []

        # Get regime
        spy_regime = timeline.get_spy_regime()
        regime_str = spy_regime.regime.value if spy_regime else "unknown"

        # Step 1: Analyst evaluation (bulk Ollama + top N Haiku)
        verdicts = await evaluate_batch(
            setups, self.config, regime=regime_str
        )

        if not verdicts:
            return []

        # Filter out DENIED setups
        viable = [v for v in verdicts if v.get("verdict") != "DENIED"]
        if not viable:
            self._log_day(date, "All setups denied by analyst", verdicts, [], [])
            return []

        # Step 2: Portfolio Manager selects trades
        pm_selections = await pm_select(
            viable, portfolio, self.config,
            regime=regime_str,
            knowledge_summary="",  # Sprint 3 will wire knowledge here
        )

        if not pm_selections:
            self._log_day(date, "PM selected no trades", verdicts, [], [])
            return []

        # Step 3: Risk Manager reviews
        approved = await risk_review(
            pm_selections, portfolio, self.config,
            regime=regime_str,
        )

        if not approved:
            self._log_day(date, "Risk rejected all trades", verdicts, pm_selections, [])
            return []

        # Map approved back to ScoredSetup objects
        setup_map = {(s.setup.symbol, s.setup.pattern_name): s for s in setups}
        final: list[ScoredSetup] = []

        for trade in approved:
            key = (trade["symbol"], trade["pattern"])
            if key in setup_map:
                scored = setup_map[key]
                # Attach agent reasoning to the scored setup for the engine
                scored._agent_meta = {
                    "analyst_verdict": self._find_verdict(verdicts, trade["symbol"]),
                    "pm_reasoning": trade.get("reasoning", ""),
                    "risk_reasoning": trade.get("risk_reasoning", ""),
                    "size_modifier": trade.get("size_modifier", 1.0),
                }
                final.append(scored)

        self._log_day(date, f"Selected {len(final)} trades", verdicts, pm_selections, approved)
        return final

    def _find_verdict(self, verdicts: list[dict], symbol: str) -> str:
        """Find analyst verdict for a symbol."""
        for v in verdicts:
            if v.get("symbol") == symbol:
                reasoning = v.get("reasoning", "")
                verdict = v.get("verdict", "CAUTION")
                return f"{verdict}: {reasoning[:300]}"
        return ""

    def _log_day(self, date: str, summary: str,
                 verdicts: list[dict], pm_selections: list[dict],
                 approved: list[dict]):
        """Record reasoning trail for the day."""
        self.day_logs.append({
            "date": date,
            "summary": summary,
            "analyst_verdicts": len(verdicts),
            "confirmed": sum(1 for v in verdicts if v.get("verdict") == "CONFIRMED"),
            "denied": sum(1 for v in verdicts if v.get("verdict") == "DENIED"),
            "pm_selected": len(pm_selections),
            "risk_approved": len(approved),
        })
