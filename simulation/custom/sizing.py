"""
simulation/custom/sizing.py — Adaptive position sizing.

Three modes:
  - fixed:    Always risk base_risk_pct of STARTING capital
  - compound: Risk base_risk_pct of CURRENT equity (grows with profits)
  - adaptive: Compound + scale by strategy win rate (proven strategies get more)
"""
import math
from dataclasses import dataclass
from typing import Optional

from simulation.custom.config import SizingConfig


@dataclass
class SizeResult:
    """Output from position sizing calculation."""
    shares: int
    dollar_risk: float
    size_modifier: float     # final multiplier applied
    reasoning: str


class AdaptiveSizer:
    """Calculates position sizes based on equity, strategy performance, and risk rules."""

    def __init__(self, config: SizingConfig, starting_capital: float):
        self.config = config
        self.starting_capital = starting_capital
        self._strategy_stats: dict[str, dict] = {}  # pattern_name → {wins, losses, total_r}

    def record_trade(self, pattern_name: str, realized_r: float):
        """Record a closed trade for adaptive sizing adjustments."""
        if pattern_name not in self._strategy_stats:
            self._strategy_stats[pattern_name] = {"wins": 0, "losses": 0, "total_r": 0.0, "trades": 0}
        stats = self._strategy_stats[pattern_name]
        stats["trades"] += 1
        stats["total_r"] += realized_r
        if realized_r > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    def get_strategy_multiplier(self, pattern_name: str) -> float:
        """Get sizing multiplier based on strategy's track record.

        Returns 1.0 if not enough data. Scales between min and max multiplier
        based on win rate relative to 50% baseline.
        """
        stats = self._strategy_stats.get(pattern_name)
        if not stats or stats["trades"] < self.config.min_trades_for_adjustment:
            return 1.0

        win_rate = stats["wins"] / stats["trades"]
        # Map win rate [0.3, 0.7] → [min_mult, max_mult]
        # Below 30% win rate → min_mult, above 70% → max_mult
        t = max(0.0, min(1.0, (win_rate - 0.3) / 0.4))
        mult = self.config.strategy_multiplier_min + t * (
            self.config.strategy_multiplier_max - self.config.strategy_multiplier_min
        )
        return round(mult, 2)

    def get_drawdown_multiplier(self, current_equity: float) -> float:
        """Reduce size during drawdowns."""
        if not self.config.drawdown_reduction:
            return 1.0
        drawdown_pct = ((self.starting_capital - current_equity) / self.starting_capital) * 100
        if drawdown_pct > 0 and drawdown_pct >= self.config.drawdown_threshold_pct:
            return self.config.drawdown_scale
        return 1.0

    def calculate(
        self,
        entry_price: float,
        stop_price: float,
        current_equity: float,
        current_cash: float,
        current_heat_pct: float,
        pattern_name: str = "",
        agent_size_modifier: float = 1.0,
    ) -> SizeResult:
        """Calculate position size with all adjustments applied.

        Args:
            entry_price: Planned entry price
            stop_price: Stop loss price
            current_equity: Current total equity (cash + positions)
            current_cash: Available cash
            current_heat_pct: Current portfolio heat %
            pattern_name: Strategy name for adaptive sizing
            agent_size_modifier: PM/Risk agent's size recommendation (0.25-2.0)
        """
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return SizeResult(0, 0.0, 0.0, "Zero risk per share")

        # Base equity to size against
        if self.config.mode == "fixed":
            base_equity = self.starting_capital
        else:
            base_equity = current_equity  # compound and adaptive

        # Base dollar risk
        dollar_budget = base_equity * self.config.base_risk_pct / 100.0

        # Strategy multiplier (adaptive mode only)
        strategy_mult = 1.0
        if self.config.mode == "adaptive":
            strategy_mult = self.get_strategy_multiplier(pattern_name)

        # Drawdown multiplier
        drawdown_mult = self.get_drawdown_multiplier(current_equity)

        # Combined modifier
        combined_mult = strategy_mult * drawdown_mult * agent_size_modifier
        combined_mult = max(0.25, min(2.0, combined_mult))
        dollar_budget *= combined_mult

        # Cap by remaining heat budget
        max_risk = base_equity * self.config.max_heat_pct / 100.0
        current_risk = base_equity * current_heat_pct / 100.0
        dollar_budget = min(dollar_budget, max(0, max_risk - current_risk))

        # Calculate shares
        shares = math.floor(dollar_budget / risk_per_share)
        if shares <= 0:
            return SizeResult(0, 0.0, combined_mult, "Budget exhausted after adjustments")

        # Cap at max position size
        max_pos = base_equity * self.config.max_position_pct / 100.0
        if shares * entry_price > max_pos:
            shares = math.floor(max_pos / entry_price)

        # Cash check
        if shares * entry_price > current_cash:
            shares = math.floor(current_cash / entry_price)

        if shares <= 0:
            return SizeResult(0, 0.0, combined_mult, "Insufficient cash")

        actual_risk = round(shares * risk_per_share, 2)
        parts = [f"{self.config.mode} sizing"]
        if strategy_mult != 1.0:
            parts.append(f"strategy×{strategy_mult:.1f}")
        if drawdown_mult != 1.0:
            parts.append(f"drawdown×{drawdown_mult:.1f}")
        if agent_size_modifier != 1.0:
            parts.append(f"agent×{agent_size_modifier:.1f}")

        return SizeResult(
            shares=shares,
            dollar_risk=actual_risk,
            size_modifier=round(combined_mult, 2),
            reasoning=" | ".join(parts),
        )

    def get_strategy_summary(self) -> list[dict]:
        """Get all strategy performance stats for display."""
        results = []
        for name, stats in sorted(self._strategy_stats.items()):
            trades = stats["trades"]
            if trades == 0:
                continue
            results.append({
                "strategy": name,
                "trades": trades,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": round(stats["wins"] / trades * 100, 1),
                "total_r": round(stats["total_r"], 3),
                "avg_r": round(stats["total_r"] / trades, 3),
                "size_multiplier": self.get_strategy_multiplier(name),
            })
        return results
