"""
strategies/evaluator.py — Rolling Strategy Performance Evaluator

Tracks the last N signals per strategy/pattern, computes rolling metrics,
and ranks strategies by recent performance. This powers:
  - "Hot Strategies" panel in the UI
  - Adaptive strategy selection (favor what's working NOW)
  - Multi-factor scoring (strategy_score component)

How it works:
  1. Every time a trade signal is generated, call record_signal()
  2. When a trade resolves (hit target or stop), call record_outcome()
  3. Call get_rankings() to see which strategies are performing best
  4. The scanner uses get_strategy_score() to boost setups from hot strategies

Persistence: saves to cache/strategy_performance.json
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


CACHE_PATH = Path("cache/strategy_performance.json")
MAX_HISTORY = 60  # Track last 60 outcomes per strategy


# ==============================================================================
# DATA TYPES
# ==============================================================================

@dataclass
class TradeOutcome:
    """A resolved trade signal with outcome."""
    pattern_name: str
    strategy_type: str      # momentum, mean_reversion, breakout, scalp
    symbol: str
    bias: str               # long or short
    entry_price: float
    target_price: float
    stop_price: float
    outcome: str            # "win", "loss", "timeout"
    realized_r: float       # R-multiple achieved
    timestamp: str          # ISO format

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern_name, "strategy": self.strategy_type,
            "symbol": self.symbol, "bias": self.bias,
            "entry": self.entry_price, "target": self.target_price,
            "stop": self.stop_price, "outcome": self.outcome,
            "realized_r": self.realized_r, "ts": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TradeOutcome":
        return cls(
            pattern_name=d["pattern"], strategy_type=d["strategy"],
            symbol=d["symbol"], bias=d["bias"],
            entry_price=d["entry"], target_price=d["target"],
            stop_price=d["stop"], outcome=d["outcome"],
            realized_r=d["realized_r"], timestamp=d["ts"],
        )


@dataclass
class StrategyMetrics:
    """Rolling performance metrics for one strategy."""
    name: str               # Pattern or strategy name
    strategy_type: str      # momentum, mean_reversion, breakout, scalp
    total_signals: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float         # 0-1
    avg_r: float            # Average R-multiple per trade
    profit_factor: float    # Gross wins / gross losses
    expectancy: float       # (WR * avg_win) - ((1-WR) * avg_loss)
    best_r: float           # Best single trade
    worst_r: float          # Worst single trade
    recent_streak: int      # Positive = win streak, negative = loss streak
    hot_score: float        # 0-100 composite "how hot is this strategy"

    def to_dict(self) -> dict:
        return {
            "name": self.name, "strategy_type": self.strategy_type,
            "total_signals": self.total_signals, "wins": self.wins,
            "losses": self.losses, "win_rate": round(self.win_rate, 3),
            "avg_r": round(self.avg_r, 3), "profit_factor": round(self.profit_factor, 2),
            "expectancy": round(self.expectancy, 3),
            "best_r": round(self.best_r, 2), "worst_r": round(self.worst_r, 2),
            "recent_streak": self.recent_streak,
            "hot_score": round(self.hot_score, 1),
        }


# ==============================================================================
# EVALUATOR
# ==============================================================================

class StrategyEvaluator:
    """
    Tracks and evaluates rolling strategy performance.

    Usage:
        evaluator = StrategyEvaluator()
        evaluator.load()

        # Record outcomes from backtest or live signals
        evaluator.record_outcome(TradeOutcome(...))

        # Get rankings
        rankings = evaluator.get_rankings()
        hot_strategies = evaluator.get_hot_strategies(top_n=3)

        # Score a specific pattern
        score = evaluator.get_strategy_score("Bull Flag")

        evaluator.save()
    """

    def __init__(self):
        # outcomes[pattern_name] = list of TradeOutcome (most recent last)
        self.outcomes: dict[str, list[TradeOutcome]] = defaultdict(list)

    def record_outcome(self, outcome: TradeOutcome):
        """Record a resolved trade outcome."""
        self.outcomes[outcome.pattern_name].append(outcome)
        # Trim to MAX_HISTORY
        if len(self.outcomes[outcome.pattern_name]) > MAX_HISTORY:
            self.outcomes[outcome.pattern_name] = \
                self.outcomes[outcome.pattern_name][-MAX_HISTORY:]

    def record_batch(self, outcomes: list[TradeOutcome]):
        """Record multiple outcomes at once (from backtest)."""
        for o in outcomes:
            self.record_outcome(o)

    def compute_metrics(self, pattern_name: str) -> Optional[StrategyMetrics]:
        """Compute rolling metrics for a single pattern/strategy."""
        trades = self.outcomes.get(pattern_name, [])
        if len(trades) < 3:
            return None

        wins = [t for t in trades if t.outcome == "win"]
        losses = [t for t in trades if t.outcome == "loss"]
        timeouts = [t for t in trades if t.outcome == "timeout"]

        n = len(trades)
        n_wins = len(wins)
        n_losses = len(losses)
        wr = n_wins / n if n > 0 else 0

        all_r = [t.realized_r for t in trades]
        avg_r = float(np.mean(all_r)) if all_r else 0

        gross_win = sum(t.realized_r for t in wins) if wins else 0
        gross_loss = sum(abs(t.realized_r) for t in losses) if losses else 0
        pf = gross_win / gross_loss if gross_loss > 0 else (99.0 if gross_win > 0 else 0)

        avg_win_r = float(np.mean([t.realized_r for t in wins])) if wins else 0
        avg_loss_r = float(np.mean([abs(t.realized_r) for t in losses])) if losses else 0
        expectancy = (wr * avg_win_r) - ((1 - wr) * avg_loss_r)

        best_r = max(all_r) if all_r else 0
        worst_r = min(all_r) if all_r else 0

        # Recent streak
        streak = 0
        for t in reversed(trades):
            if t.outcome == "win":
                if streak >= 0:
                    streak += 1
                else:
                    break
            elif t.outcome == "loss":
                if streak <= 0:
                    streak -= 1
                else:
                    break
            else:
                break

        # Hot score: composite of recent performance (0-100)
        hot_score = _compute_hot_score(wr, pf, expectancy, n, streak)

        strategy_type = trades[-1].strategy_type if trades else "unknown"

        return StrategyMetrics(
            name=pattern_name, strategy_type=strategy_type,
            total_signals=n, wins=n_wins, losses=n_losses, timeouts=len(timeouts),
            win_rate=wr, avg_r=avg_r, profit_factor=min(pf, 99.0),
            expectancy=expectancy, best_r=best_r, worst_r=worst_r,
            recent_streak=streak, hot_score=hot_score,
        )

    def get_rankings(self) -> list[StrategyMetrics]:
        """Get all strategies ranked by hot_score (best first)."""
        metrics = []
        for name in self.outcomes:
            m = self.compute_metrics(name)
            if m is not None:
                metrics.append(m)
        metrics.sort(key=lambda x: x.hot_score, reverse=True)
        return metrics

    def get_hot_strategies(self, top_n: int = 5) -> list[StrategyMetrics]:
        """Get the top N performing strategies right now."""
        return self.get_rankings()[:top_n]

    def get_hot_strategy_types(self, top_n: int = 3) -> list[str]:
        """Get the strategy TYPES that are performing best (for regime alignment)."""
        rankings = self.get_rankings()
        type_scores: dict[str, list[float]] = defaultdict(list)
        for m in rankings:
            type_scores[m.strategy_type].append(m.hot_score)
        # Average hot_score per type
        type_avg = {t: float(np.mean(scores)) for t, scores in type_scores.items()}
        return sorted(type_avg, key=type_avg.get, reverse=True)[:top_n]

    def get_strategy_score(self, pattern_name: str) -> float:
        """
        Get a 0-100 score for a pattern's recent performance.
        Used by the multi-factor scorer (Phase 6).
        Returns 40.0 (below-neutral) if no live outcome data exists, so that
        patterns must earn their score rather than getting a free pass.
        """
        m = self.compute_metrics(pattern_name)
        if m is None:
            return 40.0
        # Scale confidence by sample size: fewer than 10 trades = partial credit
        confidence = min(1.0, m.total_signals / 10.0)
        return round(confidence * m.hot_score + (1 - confidence) * 40.0, 1)

    def get_pattern_summary(self, pattern_name: str) -> dict:
        """Get a summary dict for a pattern (for API/UI)."""
        m = self.compute_metrics(pattern_name)
        if m is None:
            return {"name": pattern_name, "has_data": False, "hot_score": 40.0}
        d = m.to_dict()
        d["has_data"] = True
        return d

    # --- Persistence ---

    def save(self):
        """Save all outcomes to cache file."""
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for name, trades in self.outcomes.items():
            data[name] = [t.to_dict() for t in trades[-MAX_HISTORY:]]
        CACHE_PATH.write_text(json.dumps(data, indent=2))

    def load(self):
        """Load outcomes from cache file."""
        if not CACHE_PATH.exists():
            return
        try:
            data = json.loads(CACHE_PATH.read_text())
            for name, trades in data.items():
                self.outcomes[name] = [TradeOutcome.from_dict(t) for t in trades]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Corrupt cache, start fresh

    def clear(self):
        """Clear all recorded outcomes."""
        self.outcomes.clear()

    def stats_summary(self) -> dict:
        """Quick summary of what's tracked."""
        total_trades = sum(len(v) for v in self.outcomes.values())
        return {
            "strategies_tracked": len(self.outcomes),
            "total_trades": total_trades,
            "strategies": list(self.outcomes.keys()),
        }


# ==============================================================================
# HOT SCORE COMPUTATION
# ==============================================================================

def _compute_hot_score(
    win_rate: float,
    profit_factor: float,
    expectancy: float,
    sample_size: int,
    streak: int,
) -> float:
    """
    Composite score (0-100) for how "hot" a strategy is.

    Components:
      Win Rate     (30%): 35% WR = 0, 70%+ = 100
      Profit Factor(25%): 0.5 PF = 0, 3.0+ = 100
      Expectancy   (20%): -0.5R = 0, +1.0R = 100
      Sample Size  (15%): <5 trades = low confidence, 30+ = full
      Streak       (10%): Win streak adds, loss streak subtracts
    """
    # win_rate here is 0-1 fraction (from live outcomes, NOT the 0-100 backtest cache)
    wr_score = max(0, min(100, (win_rate - 0.35) / 0.35 * 100))

    # Profit factor score
    pf_capped = min(profit_factor, 5.0)
    pf_score = max(0, min(100, (pf_capped - 0.5) / 2.5 * 100))

    # Expectancy score
    exp_score = max(0, min(100, (expectancy + 0.5) / 1.5 * 100))

    # Sample size confidence
    sample_score = min(100, sample_size / 30 * 100)

    # Streak score: +3 streak = +30, -3 streak = -30, capped
    streak_score = max(0, min(100, 50 + streak * 10))

    hot = (
        wr_score * 0.30 +
        pf_score * 0.25 +
        exp_score * 0.20 +
        sample_score * 0.15 +
        streak_score * 0.10
    )

    return round(max(0, min(100, hot)), 1)