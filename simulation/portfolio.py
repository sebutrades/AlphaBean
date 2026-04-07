"""
simulation/portfolio.py — Portfolio state: cash, positions, P&L, heat.

Tracks the full lifecycle of a simulated portfolio:
  - Opening/closing positions with proper sizing
  - Daily equity snapshots for curve generation
  - Cumulative R tracking
  - Heat management (total open risk)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from simulation.config import SimConfig


@dataclass
class SimPosition:
    """An open position in the simulated portfolio."""
    symbol: str
    pattern_name: str
    strategy_type: str
    bias: str                   # "long" or "short"
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    shares: int
    dollar_risk: float          # $ at risk when opened
    entry_date: str             # YYYY-MM-DD
    splits: tuple = (0.5, 0.3, 0.2)
    atr: float = 0.0

    # State
    t1_hit: bool = False
    t2_hit: bool = False
    partial_rs: list = field(default_factory=list)
    remaining_weight: float = 1.0
    bars_held: int = 0

    # Agent reasoning (stored for reports)
    analyst_verdict: str = ""
    pm_reasoning: str = ""
    risk_reasoning: str = ""

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def current_risk(self) -> float:
        """Current dollar risk (reduces after T1 when stop moves to breakeven)."""
        return self.shares * abs(self.entry_price - self.stop_loss) * self.remaining_weight

    def check_bar(self, bar) -> Optional[tuple[str, float]]:
        """Check if any exit conditions are met on this bar.

        Returns (outcome, realized_r) when fully resolved, None if still open.
        Mirrors PendingTrade.check_resolution logic.
        """
        risk = self.risk_per_share
        if risk <= 0:
            return ("loss", -1.0)

        is_long = self.bias == "long"

        # Stop check (always first)
        stop_hit = False
        if is_long and bar.low <= self.stop_loss:
            stop_hit = True
        elif not is_long and bar.high >= self.stop_loss:
            stop_hit = True

        if stop_hit:
            if is_long:
                stop_r = (self.stop_loss - self.entry_price) / risk
            else:
                stop_r = (self.entry_price - self.stop_loss) / risk
            self.partial_rs.append((self.remaining_weight, round(stop_r, 3)))
            self.remaining_weight = 0.0
            return self._finalize()

        # T1 check
        if not self.t1_hit:
            if (is_long and bar.high >= self.target_1) or \
               (not is_long and bar.low <= self.target_1):
                self.t1_hit = True
                t1_r = abs(self.target_1 - self.entry_price) / risk
                self.partial_rs.append((self.splits[0], round(t1_r, 3)))
                self.remaining_weight -= self.splits[0]
                self.stop_loss = self.entry_price  # move to breakeven
                if self.remaining_weight <= 0.01:
                    return self._finalize()

        # T2 check (only after T1)
        if self.t1_hit and not self.t2_hit:
            if (is_long and bar.high >= self.target_2) or \
               (not is_long and bar.low <= self.target_2):
                self.t2_hit = True
                t2_r = abs(self.target_2 - self.entry_price) / risk
                self.partial_rs.append((self.splits[1], round(t2_r, 3)))
                self.remaining_weight -= self.splits[1]
                if self.remaining_weight <= 0.01:
                    return self._finalize()

        self.bars_held += 1
        return None

    def timeout_resolve(self, current_price: float) -> tuple[str, float]:
        """Force-close remaining position at current price."""
        risk = self.risk_per_share
        if risk <= 0:
            return ("timeout", 0.0)

        if self.bias == "long":
            remaining_r = (current_price - self.entry_price) / risk
        else:
            remaining_r = (self.entry_price - current_price) / risk

        self.partial_rs.append((self.remaining_weight, round(remaining_r, 3)))
        self.remaining_weight = 0.0
        return self._finalize()

    def _finalize(self) -> tuple[str, float]:
        """Calculate final weighted-average R and outcome."""
        if not self.partial_rs:
            return ("loss", -1.0)

        total_weight = sum(w for w, _ in self.partial_rs)
        if total_weight <= 0:
            return ("loss", -1.0)

        weighted_r = sum(w * r for w, r in self.partial_rs) / total_weight
        weighted_r = max(-10.0, min(10.0, weighted_r))  # R-cap
        weighted_r -= 0.02  # transaction cost
        weighted_r = round(weighted_r, 3)

        if self.t1_hit and self.t2_hit:
            outcome = "win"
        elif self.t1_hit:
            outcome = "partial_win"
        elif weighted_r > 0:
            outcome = "timeout"
        else:
            outcome = "loss"

        return (outcome, weighted_r)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "pattern": self.pattern_name,
            "strategy_type": self.strategy_type, "bias": self.bias,
            "entry": self.entry_price, "stop": self.stop_loss,
            "t1": self.target_1, "t2": self.target_2,
            "shares": self.shares, "dollar_risk": self.dollar_risk,
            "entry_date": self.entry_date, "bars_held": self.bars_held,
            "t1_hit": self.t1_hit, "t2_hit": self.t2_hit,
            "remaining_weight": self.remaining_weight,
            "analyst_verdict": self.analyst_verdict,
            "pm_reasoning": self.pm_reasoning,
        }


@dataclass
class ClosedTrade:
    """A resolved trade with full outcome data."""
    symbol: str
    pattern_name: str
    strategy_type: str
    bias: str
    entry_price: float
    stop_price: float
    target_1: float
    target_2: float
    shares: int
    dollar_risk: float
    entry_date: str
    exit_date: str
    outcome: str            # "win", "partial_win", "loss", "timeout"
    realized_r: float
    bars_held: int
    t1_hit: bool
    t2_hit: bool
    analyst_verdict: str = ""
    pm_reasoning: str = ""
    risk_reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "pattern": self.pattern_name,
            "strategy_type": self.strategy_type, "bias": self.bias,
            "entry": self.entry_price, "stop": self.stop_price,
            "t1": self.target_1, "t2": self.target_2,
            "shares": self.shares, "dollar_risk": self.dollar_risk,
            "entry_date": self.entry_date, "exit_date": self.exit_date,
            "outcome": self.outcome, "realized_r": self.realized_r,
            "bars_held": self.bars_held,
            "t1_hit": self.t1_hit, "t2_hit": self.t2_hit,
            "analyst_verdict": self.analyst_verdict,
            "pm_reasoning": self.pm_reasoning,
        }


@dataclass
class DailySnapshot:
    """End-of-day portfolio snapshot for equity curve."""
    date: str
    equity: float
    cash: float
    open_positions: int
    day_r: float            # R gained/lost this day
    cumulative_r: float
    trades_opened: int
    trades_closed: int
    heat_pct: float


class PortfolioState:
    """Full portfolio tracking for simulation."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.cash = config.starting_capital
        self.positions: list[SimPosition] = []
        self.closed_trades: list[ClosedTrade] = []
        self.daily_snapshots: list[DailySnapshot] = []
        self.cumulative_r: float = 0.0
        self._day_r: float = 0.0
        self._day_opened: int = 0
        self._day_closed: int = 0

    def start_day(self):
        """Reset daily counters at start of each day."""
        self._day_r = 0.0
        self._day_opened = 0
        self._day_closed = 0

    @property
    def total_heat_pct(self) -> float:
        """Current portfolio heat as % of capital."""
        total_risk = sum(p.current_risk for p in self.positions)
        return (total_risk / self.config.starting_capital) * 100.0

    @property
    def can_add_trade(self) -> bool:
        """Can we open a new position?"""
        if len(self.positions) >= self.config.max_concurrent_positions:
            return False
        if self.total_heat_pct >= self.config.max_portfolio_heat_pct:
            return False
        return True

    @property
    def remaining_risk_budget(self) -> float:
        """Remaining risk budget in dollars."""
        max_risk = self.config.starting_capital * self.config.max_portfolio_heat_pct / 100.0
        current_risk = sum(p.current_risk for p in self.positions)
        return max(0.0, max_risk - current_risk)

    def size_trade(self, entry_price: float, stop_loss: float) -> tuple[int, float]:
        """Calculate shares and dollar risk for a new trade.

        Returns (shares, dollar_risk). Returns (0, 0) if can't trade.
        """
        if not self.can_add_trade:
            return 0, 0.0

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0, 0.0

        # Dollar risk budget: 1% of starting capital
        dollar_budget = self.config.starting_capital * self.config.risk_per_trade_pct / 100.0

        # Don't exceed remaining heat budget
        dollar_budget = min(dollar_budget, self.remaining_risk_budget)

        shares = math.floor(dollar_budget / risk_per_share)
        if shares <= 0:
            return 0, 0.0

        # Cap at 15% of capital in a single position
        max_position = self.config.starting_capital * 0.15
        if shares * entry_price > max_position:
            shares = math.floor(max_position / entry_price)

        # Can't spend more cash than we have
        if shares * entry_price > self.cash:
            shares = math.floor(self.cash / entry_price)
            if shares <= 0:
                return 0, 0.0

        dollar_risk = shares * risk_per_share
        return shares, round(dollar_risk, 2)

    def open_position(self, pos: SimPosition):
        """Add a new open position and deduct capital."""
        position_cost = pos.shares * pos.entry_price
        self.cash -= position_cost
        self.positions.append(pos)
        self._day_opened += 1

    def close_position(self, pos: SimPosition, exit_date: str,
                       outcome: str, realized_r: float):
        """Move a position from open to closed."""
        closed = ClosedTrade(
            symbol=pos.symbol, pattern_name=pos.pattern_name,
            strategy_type=pos.strategy_type, bias=pos.bias,
            entry_price=pos.entry_price, stop_price=pos.stop_loss,
            target_1=pos.target_1, target_2=pos.target_2,
            shares=pos.shares, dollar_risk=pos.dollar_risk,
            entry_date=pos.entry_date, exit_date=exit_date,
            outcome=outcome, realized_r=realized_r,
            bars_held=pos.bars_held,
            t1_hit=pos.t1_hit, t2_hit=pos.t2_hit,
            analyst_verdict=pos.analyst_verdict,
            pm_reasoning=pos.pm_reasoning,
            risk_reasoning=pos.risk_reasoning,
        )
        self.closed_trades.append(closed)

        # Return position capital + P&L to cash
        position_cost = pos.shares * pos.entry_price
        pnl = realized_r * pos.dollar_risk
        self.cash += position_cost + pnl
        self.cumulative_r += realized_r
        self._day_r += realized_r
        self._day_closed += 1

        # Remove from open
        self.positions = [p for p in self.positions if p is not pos]

    def end_day(self, date: str, current_prices: Optional[dict[str, float]] = None):
        """Record end-of-day snapshot.

        Args:
            current_prices: {symbol: close_price} for mark-to-market.
                           Falls back to entry_price if not provided.
        """
        position_value = 0.0
        for p in self.positions:
            price = (current_prices or {}).get(p.symbol, p.entry_price)
            position_value += p.shares * price
        equity = self.cash + position_value

        snap = DailySnapshot(
            date=date,
            equity=round(equity, 2),
            cash=round(self.cash, 2),
            open_positions=len(self.positions),
            day_r=round(self._day_r, 3),
            cumulative_r=round(self.cumulative_r, 3),
            trades_opened=self._day_opened,
            trades_closed=self._day_closed,
            heat_pct=round(self.total_heat_pct, 2),
        )
        self.daily_snapshots.append(snap)

    def get_stats(self) -> dict:
        """Compute summary statistics."""
        closed = self.closed_trades
        if not closed:
            return {"total_trades": 0, "total_r": 0, "win_rate": 0,
                    "avg_r": 0, "profit_factor": 0}

        wins = [t for t in closed if t.realized_r > 0]
        losses = [t for t in closed if t.realized_r <= 0]
        total_r = sum(t.realized_r for t in closed)
        win_rate = len(wins) / len(closed) * 100 if closed else 0

        gross_profit = sum(t.realized_r for t in wins) if wins else 0
        gross_loss = abs(sum(t.realized_r for t in losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "total_r": round(total_r, 2),
            "win_rate": round(win_rate, 1),
            "avg_r": round(total_r / len(closed), 3),
            "profit_factor": round(pf, 2),
            "best_trade_r": round(max(t.realized_r for t in closed), 2),
            "worst_trade_r": round(min(t.realized_r for t in closed), 2),
            "open_positions": len(self.positions),
            "current_heat": round(self.total_heat_pct, 2),
        }

    def to_checkpoint(self) -> dict:
        """Serialize state for checkpoint/resume."""
        return {
            "cash": self.cash,
            "cumulative_r": self.cumulative_r,
            "positions": [p.to_dict() for p in self.positions],
            "closed_trades": [t.to_dict() for t in self.closed_trades],
            "daily_snapshots": [
                {"date": s.date, "equity": s.equity, "cash": s.cash,
                 "open_positions": s.open_positions, "day_r": s.day_r,
                 "cumulative_r": s.cumulative_r, "trades_opened": s.trades_opened,
                 "trades_closed": s.trades_closed, "heat_pct": s.heat_pct}
                for s in self.daily_snapshots
            ],
        }

    def save_checkpoint(self, path: Path):
        """Save full state to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_checkpoint(), indent=2))
