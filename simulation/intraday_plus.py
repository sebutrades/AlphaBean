"""
simulation/intraday_plus.py — Deterministic+ intraday simulation engine.

Upgrades over base intraday.py:
  - Outcome feedback: closed trades feed back into StrategyEvaluator,
    so strategy_score (18% of composite) reflects actual performance
  - Adaptive sizing: compound equity + strategy performance multiplier
  - Multi-timeframe: daily setups scanned once/day, held across days (no EOD close)
  - Multi-trade: takes up to N qualifying setups per scan, not just the top 1
  - Strategy scaling: after 50+ trades, scale down losing strategies (don't remove)
  - Comprehensive logging: per-day stats, strategy breakdown, sizing decisions
  - $1M starting capital, capped daily risk

Walk-forward integrity: only sees bars up to current time. No lookahead.
"""
import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from backend.data.schemas import Bar, BarSeries
from backend.features.engine import compute_features
from backend.patterns.classifier import classify_all
from backend.regime.detector import detect_regime
from backend.scoring.multi_factor import score_setup, ScoredSetup
from backend.strategies.evaluator import StrategyEvaluator, TradeOutcome

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

BAR_DATA_DIR = Path("cache/bar_data")
LIVE_DATA_DIR = Path("live_data_cache/data/5min")
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_DIR = Path("simulation/output/det_plus_logs")


def _native(obj):
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


# ── Events ───────────────────────────────────────────────────────────────────

@dataclass
class SimEvent:
    type: str
    timestamp: str
    data: dict

    def to_dict(self) -> dict:
        return {"type": self.type, "timestamp": self.timestamp, "data": _native(self.data)}


# ── Position ─────────────────────────────────────────────────────────────────

@dataclass
class Position:
    id: str
    symbol: str
    pattern_name: str
    strategy_type: str
    bias: str
    entry_price: float
    stop_loss: float
    original_stop: float
    target_1: float
    target_2: float
    shares: int
    dollar_risk: float
    entry_time: str
    composite_score: float
    timeframe: str = "5min"  # "5min" or "1d"
    size_modifier: float = 1.0
    splits: tuple = (0.5, 0.3, 0.2)

    # State
    t1_hit: bool = False
    t2_hit: bool = False
    partial_rs: list = field(default_factory=list)
    remaining_weight: float = 1.0
    bars_held: int = 0
    current_price: float = 0.0
    unrealized_r: float = 0.0
    high_water: float = 0.0

    def update_price(self, price: float):
        self.current_price = price
        risk = abs(self.entry_price - self.original_stop)
        if risk > 0:
            if self.bias == "long":
                self.unrealized_r = (price - self.entry_price) / risk
            else:
                self.unrealized_r = (self.entry_price - price) / risk
        self.high_water = max(self.high_water, self.unrealized_r)
        self.bars_held += 1

    @property
    def unrealized_pnl(self) -> float:
        return self.unrealized_r * self.dollar_risk

    def check_bar(self, bar: Bar) -> Optional[tuple[str, float]]:
        risk = abs(self.entry_price - self.original_stop)
        if risk <= 0:
            return ("loss", -1.0)
        is_long = self.bias == "long"

        # Stop
        if (is_long and bar.low <= self.stop_loss) or \
           (not is_long and bar.high >= self.stop_loss):
            stop_r = ((self.stop_loss - self.entry_price) / risk) if is_long else ((self.entry_price - self.stop_loss) / risk)
            self.partial_rs.append((self.remaining_weight, round(stop_r, 3)))
            self.remaining_weight = 0.0
            return self._finalize()

        # T1
        if not self.t1_hit:
            if (is_long and bar.high >= self.target_1) or \
               (not is_long and bar.low <= self.target_1):
                self.t1_hit = True
                t1_r = abs(self.target_1 - self.entry_price) / risk
                self.partial_rs.append((self.splits[0], round(t1_r, 3)))
                self.remaining_weight -= self.splits[0]
                self.stop_loss = self.entry_price
                if self.remaining_weight <= 0.01:
                    return self._finalize()

        # T2
        if self.t1_hit and not self.t2_hit:
            if (is_long and bar.high >= self.target_2) or \
               (not is_long and bar.low <= self.target_2):
                self.t2_hit = True
                t2_r = abs(self.target_2 - self.entry_price) / risk
                self.partial_rs.append((self.splits[1], round(t2_r, 3)))
                self.remaining_weight -= self.splits[1]
                if self.remaining_weight <= 0.01:
                    return self._finalize()

        self.update_price(bar.close)
        return None

    def force_close(self, price: float) -> tuple[str, float]:
        risk = abs(self.entry_price - self.original_stop)
        if risk <= 0:
            return ("closed", 0.0)
        remaining_r = ((price - self.entry_price) / risk) if self.bias == "long" else ((self.entry_price - price) / risk)
        self.partial_rs.append((self.remaining_weight, round(remaining_r, 3)))
        self.remaining_weight = 0.0
        return self._finalize()

    def _finalize(self) -> tuple[str, float]:
        if not self.partial_rs:
            return ("loss", -1.0)
        total_weight = sum(w for w, _ in self.partial_rs)
        if total_weight <= 0:
            return ("loss", -1.0)
        weighted_r = sum(w * r for w, r in self.partial_rs) / total_weight
        weighted_r = max(-10.0, min(10.0, weighted_r)) - 0.02
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
            "id": self.id, "symbol": self.symbol, "pattern": self.pattern_name,
            "strategy_type": self.strategy_type, "bias": self.bias, "timeframe": self.timeframe,
            "entry": self.entry_price, "stop": self.stop_loss, "original_stop": self.original_stop,
            "t1": self.target_1, "t2": self.target_2,
            "shares": self.shares, "dollar_risk": self.dollar_risk,
            "entry_time": self.entry_time, "score": self.composite_score,
            "size_modifier": self.size_modifier,
            "t1_hit": self.t1_hit, "t2_hit": self.t2_hit,
            "bars_held": self.bars_held,
            "current_price": round(self.current_price, 2),
            "unrealized_r": round(self.unrealized_r, 3),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "high_water_r": round(self.high_water, 3),
            "remaining_weight": round(self.remaining_weight, 2),
        }


# ── Data Loading ─────────────────────────────────────────────────────────────

def _load_5min_for_date(date: str) -> dict[str, list[dict]]:
    symbol_bars: dict[str, list[dict]] = {}
    if BAR_DATA_DIR.exists():
        for fpath in BAR_DATA_DIR.glob("*_5min.json"):
            try:
                data = json.loads(fpath.read_text())
                bars = [b for b in data.get("bars", [])
                        if b["t"][:10] == date and MARKET_OPEN <= b["t"][11:16] < MARKET_CLOSE]
                if bars:
                    sym = data.get("symbol", fpath.stem.split("_")[0]).upper()
                    symbol_bars[sym] = bars
            except Exception:
                continue
    if not symbol_bars and LIVE_DATA_DIR.exists():
        for fpath in LIVE_DATA_DIR.glob("*.json"):
            try:
                data = json.loads(fpath.read_text())
                bars = [b for b in data.get("bars", [])
                        if b["t"][:10] == date and MARKET_OPEN <= b["t"][11:16] < MARKET_CLOSE]
                if bars:
                    symbol_bars[fpath.stem.upper()] = bars
            except Exception:
                continue
    return symbol_bars


def _load_daily_bars_up_to(date: str) -> dict[str, list[dict]]:
    """Load daily bars for all symbols up to (and including) date.

    Used for daily-timeframe pattern detection. Walk-forward safe.
    """
    symbol_bars: dict[str, list[dict]] = {}
    if BAR_DATA_DIR.exists():
        for fpath in BAR_DATA_DIR.glob("*_1d.json"):
            try:
                data = json.loads(fpath.read_text())
                bars = [b for b in data.get("bars", []) if b["t"][:10] <= date]
                if len(bars) >= 20:
                    sym = data.get("symbol", fpath.stem.split("_")[0]).upper()
                    symbol_bars[sym] = bars
            except Exception:
                continue
    return symbol_bars


def get_available_dates() -> list[str]:
    dates: set[str] = set()
    if BAR_DATA_DIR.exists():
        spy_path = BAR_DATA_DIR / "SPY_5min.json"
        if spy_path.exists():
            try:
                data = json.loads(spy_path.read_text())
                for b in data.get("bars", []):
                    if MARKET_OPEN <= b["t"][11:16] < MARKET_CLOSE:
                        dates.add(b["t"][:10])
            except Exception:
                pass
    return sorted(dates)


# ── Adaptive Sizing ──────────────────────────────────────────────────────────

class AdaptiveSizer:
    """Position sizing that compounds and adapts to strategy performance.

    Uses recency-weighted expectancy so strategies can recover from early
    bad performance.  The last 20 trades carry 2x the weight of older ones,
    letting a strategy that "finds its groove" rebuild its multiplier
    without being permanently anchored by early losses.
    """

    RECENT_WINDOW = 20   # trades in the "recent" bucket
    RECENT_WEIGHT = 2.0  # how much more recent trades matter

    def __init__(self, starting_capital: float, base_risk_pct: float = 0.5):
        self.starting_capital = starting_capital
        self.base_risk_pct = base_risk_pct
        self._strategy_stats: dict[str, dict] = {}
        # Keep full trade history per strategy for recency weighting
        self._strategy_history: dict[str, list[float]] = {}

    def record_trade(self, pattern: str, realized_r: float):
        if pattern not in self._strategy_stats:
            self._strategy_stats[pattern] = {"wins": 0, "losses": 0, "total_r": 0.0, "trades": 0}
            self._strategy_history[pattern] = []
        s = self._strategy_stats[pattern]
        s["trades"] += 1
        s["total_r"] += realized_r
        if realized_r > 0:
            s["wins"] += 1
        else:
            s["losses"] += 1
        self._strategy_history[pattern].append(realized_r)
        # Cap history to avoid unbounded memory growth — keep enough for
        # recency weighting (RECENT_WINDOW=20) plus a generous old bucket.
        if len(self._strategy_history[pattern]) > 200:
            self._strategy_history[pattern] = self._strategy_history[pattern][-200:]

    def _recency_weighted_avg_r(self, pattern: str) -> float:
        """Compute expectancy giving recent trades more influence.

        Splits history into old and recent (last RECENT_WINDOW trades).
        Recent trades get RECENT_WEIGHT, old trades get 1.0.
        This lets a recovering strategy climb back faster while still
        respecting its full track record.
        """
        history = self._strategy_history.get(pattern, [])
        if not history:
            return 0.0

        n = len(history)
        if n <= self.RECENT_WINDOW:
            # Not enough history to split — simple average
            return sum(history) / n

        old = history[:-self.RECENT_WINDOW]
        recent = history[-self.RECENT_WINDOW:]

        old_sum = sum(old)
        recent_sum = sum(recent)

        # Weighted average: recent trades count RECENT_WEIGHT× more
        total_weight = len(old) + len(recent) * self.RECENT_WEIGHT
        weighted_sum = old_sum + recent_sum * self.RECENT_WEIGHT

        return weighted_sum / total_weight

    def get_strategy_multiplier(self, pattern: str) -> float:
        """Scale sizing by strategy performance.

        - < 50 trades: 1.0 (no opinion yet)
        - 50+ trades with positive expectancy: scale UP (1.0 - 1.5)
        - 50+ trades with negative expectancy: scale DOWN (0.3 - 0.8)
        - Never remove entirely — just reduce
        - Uses recency weighting so strategies CAN recover
        """
        s = self._strategy_stats.get(pattern)
        if not s or s["trades"] < 50:
            return 1.0

        avg_r = self._recency_weighted_avg_r(pattern)

        if avg_r >= 0.15:
            return min(1.5, 1.0 + avg_r)         # strong positive: up to 1.5x
        elif avg_r >= 0:
            return 1.0                             # marginal positive: keep 1x
        elif avg_r >= -0.1:
            return 0.7                             # slightly negative: reduce to 0.7x
        elif avg_r >= -0.3:
            return 0.5                             # negative: reduce to 0.5x
        else:
            return 0.3                             # deeply negative: reduce to 0.3x

    def get_drawdown_multiplier(self, equity: float) -> float:
        dd_pct = max(0, (self.starting_capital - equity) / self.starting_capital * 100)
        if dd_pct >= 15:
            return 0.25
        elif dd_pct >= 10:
            return 0.5
        elif dd_pct >= 5:
            return 0.75
        return 1.0

    def calculate(self, entry: float, stop: float, equity: float, cash: float,
                  heat_pct: float, pattern: str = "", max_heat: float = 8.0,
                  max_daily_risk_pct: float = 3.0, daily_risk_used: float = 0.0) -> tuple[int, float, float, str]:
        """Returns (shares, dollar_risk, size_modifier, reasoning)."""
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return 0, 0.0, 0.0, "zero risk"

        # Base: % of current equity (compounds)
        dollar_budget = equity * self.base_risk_pct / 100.0

        # Strategy multiplier
        strat_mult = self.get_strategy_multiplier(pattern)
        # Drawdown multiplier
        dd_mult = self.get_drawdown_multiplier(equity)

        combined = strat_mult * dd_mult
        combined = max(0.2, min(1.5, combined))
        dollar_budget *= combined

        # Cap by remaining heat
        max_risk_total = equity * max_heat / 100.0
        current_risk = equity * heat_pct / 100.0
        dollar_budget = min(dollar_budget, max(0, max_risk_total - current_risk))

        # Cap by daily risk limit
        daily_limit = equity * max_daily_risk_pct / 100.0
        dollar_budget = min(dollar_budget, max(0, daily_limit - daily_risk_used))

        shares = math.floor(dollar_budget / risk_per_share)
        if shares <= 0:
            return 0, 0.0, combined, "budget exhausted"

        # Cap at 8% of equity per position
        max_pos = equity * 0.08
        if shares * entry > max_pos:
            shares = math.floor(max_pos / entry)

        # Cash check
        if shares * entry > cash:
            shares = math.floor(cash / entry)

        if shares <= 0:
            return 0, 0.0, combined, "no cash"

        actual_risk = round(shares * risk_per_share, 2)
        parts = [f"compound {self.base_risk_pct}%"]
        if strat_mult != 1.0:
            parts.append(f"strat×{strat_mult:.2f}")
        if dd_mult != 1.0:
            parts.append(f"dd×{dd_mult:.2f}")

        return shares, actual_risk, round(combined, 2), " | ".join(parts)

    def get_summary(self) -> list[dict]:
        results = []
        for name, s in sorted(self._strategy_stats.items(), key=lambda x: x[1]["total_r"], reverse=True):
            if s["trades"] == 0:
                continue
            alltime_avg = round(s["total_r"] / s["trades"], 3)
            recent_avg = round(self._recency_weighted_avg_r(name), 3)
            results.append({
                "strategy": name, "trades": s["trades"],
                "wins": s["wins"], "losses": s["losses"],
                "win_rate": round(s["wins"] / s["trades"] * 100, 1),
                "total_r": round(s["total_r"], 3),
                "avg_r": alltime_avg,
                "recent_avg_r": recent_avg,
                "expectancy": alltime_avg,
                "size_mult": self.get_strategy_multiplier(name),
            })
        return results


# ── Main Engine ──────────────────────────────────────────────────────────────

class IntradayPlusSimulation:
    """Deterministic+ simulation with learning, multi-TF, adaptive sizing."""

    def __init__(
        self,
        emit: Callable[[SimEvent], None],
        starting_capital: float = 1_000_000.0,
        risk_pct: float = 0.5,
        max_heat_pct: float = 8.0,
        max_daily_risk_pct: float = 3.0,
        max_positions: int = 20,
        min_score: float = 50.0,
        max_intraday_trades_per_scan: int = 3,
        max_daily_trades_per_day: int = 2,
        playback_speed: float = 1.0,
    ):
        self.emit = emit
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.risk_pct = risk_pct
        self.max_heat_pct = max_heat_pct
        self.max_daily_risk_pct = max_daily_risk_pct
        self.max_positions = max_positions
        self.min_score = min_score
        self.max_intraday_per_scan = max_intraday_trades_per_scan
        self.max_daily_per_day = max_daily_trades_per_day
        self.playback_speed = playback_speed

        self.positions: list[Position] = []
        self.closed_trades: list[dict] = []
        self.pnl_history: list[dict] = []
        self.cumulative_r: float = 0.0
        self.cumulative_pnl: float = 0.0
        self._trade_counter: int = 0
        self._last_trade_time: str = ""
        self._daily_risk_used: float = 0.0
        self._daily_trades_taken: int = 0

        # Learning
        self._evaluator = StrategyEvaluator()
        self._sizer = AdaptiveSizer(starting_capital, risk_pct)

        # Multi-day
        self._day_number: int = 0
        self._equity_curve: list[dict] = []
        self._day_logs: list[dict] = []
        self._prev_day_r: float = 0.0
        self._prev_day_pnl: float = 0.0
        self._prev_trade_count: int = 0

        # Control
        self._running = False
        self._paused = False

    @property
    def current_equity(self) -> float:
        return self.cash + sum(p.shares * p.current_price for p in self.positions)

    @property
    def total_heat_pct(self) -> float:
        total_risk = sum(p.dollar_risk * p.remaining_weight for p in self.positions)
        eq = max(self.current_equity, 1)
        return (total_risk / eq) * 100.0

    @property
    def intraday_positions(self) -> list[Position]:
        return [p for p in self.positions if p.timeframe == "5min"]

    @property
    def daily_positions(self) -> list[Position]:
        return [p for p in self.positions if p.timeframe == "1d"]

    def _can_trade_intraday(self, bar_time: str) -> bool:
        if len(self.positions) >= self.max_positions:
            return False
        if self.total_heat_pct >= self.max_heat_pct:
            return False
        if not self._last_trade_time:
            return True
        try:
            last = datetime.strptime(self._last_trade_time, "%Y-%m-%dT%H:%M:%S")
            current = datetime.strptime(bar_time, "%Y-%m-%dT%H:%M:%S")
            return (current - last).total_seconds() >= 300
        except Exception:
            return True

    # ── Run Loop ─────────────────────────────────────────────────────────────

    async def run_day(self, date: str):
        """Run a single day (public interface matching IntradaySimulation)."""
        self._running = True
        self.emit(SimEvent("multi_day_start", "", {
            "total_days": 1, "starting_capital": self.starting_capital,
            "mode": "deterministic_plus",
        }))
        await self._run_day(date)
        self._save_results([date])
        self.emit(SimEvent("multi_day_end", "", {
            "total_days": 1,
            "total_trades": len(self.closed_trades),
            "cumulative_r": round(self.cumulative_r, 3),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "final_equity": round(self.current_equity, 2),
            "wins": sum(1 for t in self.closed_trades if t["realized_r"] > 0),
            "losses": sum(1 for t in self.closed_trades if t["realized_r"] <= 0),
            "equity_curve": self._equity_curve,
            "strategy_summary": self._sizer.get_summary(),
        }))

    async def run_continuous(self, dates: list[str]):
        self._running = True
        total_days = len(dates)

        self.emit(SimEvent("multi_day_start", "", {
            "total_days": total_days, "starting_capital": self.starting_capital,
            "mode": "deterministic_plus",
        }))

        for i, date in enumerate(dates):
            if not self._running:
                break
            await self._run_day(date)
            # Reset intraday rate limits (but keep daily positions!)
            self._last_trade_time = ""
            self._daily_risk_used = 0.0
            self._daily_trades_taken = 0

        self._save_results(dates)

        self.emit(SimEvent("multi_day_end", "", {
            "total_days": self._day_number,
            "total_trades": len(self.closed_trades),
            "cumulative_r": round(self.cumulative_r, 3),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "final_equity": round(self.current_equity, 2),
            "wins": sum(1 for t in self.closed_trades if t["realized_r"] > 0),
            "losses": sum(1 for t in self.closed_trades if t["realized_r"] <= 0),
            "equity_curve": self._equity_curve,
            "strategy_summary": self._sizer.get_summary(),
        }))

    async def _run_day(self, date: str):
        self._day_number += 1

        # Load 5min bars
        symbol_bars_5m = _load_5min_for_date(date)
        if not symbol_bars_5m:
            return

        # Load daily bars (walk-forward: only up to yesterday)
        # We scan daily setups ONCE at start of day
        daily_setups: list[ScoredSetup] = []
        if self._day_number == 1 or self._daily_trades_taken == 0:
            daily_setups = await self._scan_daily_setups(date)

        all_bar_times: set[str] = set()
        for bars in symbol_bars_5m.values():
            for b in bars:
                all_bar_times.add(b["t"])

        sorted_times = sorted(all_bar_times)
        total_bars = len(sorted_times)

        self.emit(SimEvent("day_start", sorted_times[0] if sorted_times else "", {
            "date": date, "symbols": len(symbol_bars_5m), "total_bars": total_bars,
            "capital": round(self.cash, 2), "equity": round(self.current_equity, 2),
            "open_positions": len(self.positions),
            "intraday_positions": len(self.intraday_positions),
            "daily_positions": len(self.daily_positions),
            "day_number": self._day_number,
            "daily_setups_found": len(daily_setups),
        }))

        # Take daily setups at start of day
        for scored in daily_setups[:self.max_daily_per_day]:
            if len(self.positions) >= self.max_positions:
                break
            if scored.setup.symbol in {p.symbol for p in self.positions}:
                continue
            self._open_trade(scored, sorted_times[0] if sorted_times else date, timeframe="1d")

        # Walk 5min bars
        seen_bars: dict[str, list[dict]] = {sym: [] for sym in symbol_bars_5m}
        bars_this_tick: dict[str, dict] = {}

        for bar_idx, bar_time in enumerate(sorted_times):
            if not self._running:
                break
            while self._paused:
                await asyncio.sleep(0.1)

            bars_this_tick = {}
            for sym, bars in symbol_bars_5m.items():
                for b in bars:
                    if b["t"] == bar_time:
                        seen_bars[sym].append(b)
                        bars_this_tick[sym] = b
                        break

            spy_bar = bars_this_tick.get("SPY", {})
            self.emit(SimEvent("bar", bar_time, {
                "bar_index": bar_idx, "total_bars": total_bars,
                "spy_price": spy_bar.get("c", 0),
                "symbols_active": len(bars_this_tick),
            }))

            # Resolve ALL positions (intraday + daily) against 5min bars
            await self._resolve_positions(bar_time, bars_this_tick)

            # Scan for intraday setups
            if bar_idx >= 12 and self._can_trade_intraday(bar_time):
                await self._scan_and_trade_intraday(bar_time, seen_bars, bars_this_tick)

            self._emit_pnl(bar_time, bar_idx, total_bars)

            delay = 0.5 / max(0.1, self.playback_speed)
            await asyncio.sleep(delay)

        # EOD: close INTRADAY positions only (daily positions carry)
        await self._close_intraday_eod(sorted_times[-1] if sorted_times else "", bars_this_tick)

        # Day summary
        equity = round(self.current_equity, 2)
        day_pnl = self.cumulative_pnl - self._prev_day_pnl
        trades_today = len(self.closed_trades) - self._prev_trade_count

        day_log = {
            "date": date, "day_number": self._day_number,
            "equity": equity, "day_pnl": round(day_pnl, 2),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "trades_today": trades_today,
            "total_trades": len(self.closed_trades),
            "open_intraday": len(self.intraday_positions),
            "open_daily": len(self.daily_positions),
            "heat": round(self.total_heat_pct, 1),
            "strategy_summary": self._sizer.get_summary(),
        }
        self._day_logs.append(day_log)

        self._equity_curve.append({
            "date": date, "day_number": self._day_number,
            "equity": equity,
            "cumulative_r": round(self.cumulative_r, 3),
            "day_r": round(self.cumulative_r - self._prev_day_r, 3),
            "day_pnl": round(day_pnl, 2),
            "daily_pnl": round(day_pnl, 2),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "heat": round(self.total_heat_pct, 1),
            "trades_today": trades_today,
            "open_daily": len(self.daily_positions),
        })

        self._prev_day_r = self.cumulative_r
        self._prev_day_pnl = self.cumulative_pnl
        self._prev_trade_count = len(self.closed_trades)

        self.emit(SimEvent("day_end", sorted_times[-1] if sorted_times else "", {
            **day_log,
            "wins": sum(1 for t in self.closed_trades if t["realized_r"] > 0),
            "losses": sum(1 for t in self.closed_trades if t["realized_r"] <= 0),
        }))

    # ── Daily Setup Scanning ─────────────────────────────────────────────────

    async def _scan_daily_setups(self, date: str) -> list[ScoredSetup]:
        """Scan daily timeframe for swing setups. Walk-forward: bars up to yesterday."""
        # Use the day before current date to avoid lookahead
        yesterday = None
        all_dates = get_available_dates()
        idx = all_dates.index(date) if date in all_dates else -1
        if idx > 0:
            yesterday = all_dates[idx - 1]
        else:
            return []

        daily_bars = _load_daily_bars_up_to(yesterday)
        if not daily_bars:
            return []

        candidates: list[ScoredSetup] = []
        held_symbols = {p.symbol for p in self.positions}

        for sym, bars_list in daily_bars.items():
            if sym in held_symbols:
                continue
            if len(bars_list) < 30:
                continue

            bar_objects = [
                Bar(symbol=sym,
                    timestamp=datetime.strptime(b["t"], "%Y-%m-%dT%H:%M:%S"),
                    open=b["o"], high=b["h"], low=b["l"], close=b["c"],
                    volume=int(b["v"]))
                for b in bars_list
            ]
            bar_series = BarSeries(symbol=sym, timeframe="1d", bars=bar_objects)

            setups = classify_all(bar_series)
            if not setups:
                continue

            closes = np.array([b["c"] for b in bars_list], dtype=np.float64)
            highs = np.array([b["h"] for b in bars_list], dtype=np.float64)
            lows = np.array([b["l"] for b in bars_list], dtype=np.float64)
            volumes = np.array([b["v"] for b in bars_list], dtype=np.float64)
            features = compute_features(closes, highs, lows, volumes)
            regime = detect_regime(closes, highs, lows)

            for setup in setups:
                scored = score_setup(setup, features, regime, evaluator=self._evaluator)
                if scored.composite_score >= self.min_score:
                    candidates.append(scored)

        candidates.sort(key=lambda x: x.composite_score, reverse=True)

        for scored in candidates[:10]:
            self.emit(SimEvent("setup_detected", "", {
                "symbol": scored.setup.symbol,
                "pattern": scored.setup.pattern_name,
                "bias": scored.setup.bias.value,
                "score": round(scored.composite_score, 1),
                "entry": scored.setup.entry_price,
                "stop": scored.setup.stop_loss,
                "rr": round(scored.setup.risk_reward_ratio, 1),
                "strategy_type": scored.setup.strategy_type,
                "timeframe": "1d",
            }))

        return candidates

    # ── Intraday Scanning ────────────────────────────────────────────────────

    async def _scan_and_trade_intraday(self, bar_time: str, seen_bars: dict, bars_this_tick: dict):
        candidates: list[ScoredSetup] = []
        held_symbols = {p.symbol for p in self.positions}

        for sym, bars_list in seen_bars.items():
            if len(bars_list) < 20:
                continue
            if sym in held_symbols:
                continue

            bar_objects = [
                Bar(symbol=sym,
                    timestamp=datetime.strptime(b["t"], "%Y-%m-%dT%H:%M:%S"),
                    open=b["o"], high=b["h"], low=b["l"], close=b["c"],
                    volume=int(b["v"]))
                for b in bars_list
            ]
            bar_series = BarSeries(symbol=sym, timeframe="5min", bars=bar_objects)

            setups = classify_all(bar_series)
            if not setups:
                continue

            closes = np.array([b["c"] for b in bars_list], dtype=np.float64)
            highs = np.array([b["h"] for b in bars_list], dtype=np.float64)
            lows = np.array([b["l"] for b in bars_list], dtype=np.float64)
            volumes = np.array([b["v"] for b in bars_list], dtype=np.float64)
            features = compute_features(closes, highs, lows, volumes)
            regime = detect_regime(closes, highs, lows)

            for setup in setups:
                scored = score_setup(setup, features, regime, evaluator=self._evaluator)
                if scored.composite_score >= self.min_score:
                    candidates.append(scored)

        if not candidates:
            return

        candidates.sort(key=lambda x: x.composite_score, reverse=True)
        best = candidates[:5]

        for scored in best:
            self.emit(SimEvent("setup_detected", bar_time, {
                "symbol": scored.setup.symbol,
                "pattern": scored.setup.pattern_name,
                "bias": scored.setup.bias.value,
                "score": round(scored.composite_score, 1),
                "entry": scored.setup.entry_price,
                "stop": scored.setup.stop_loss,
                "t1": scored.setup.target_1 or scored.setup.target_price,
                "t2": scored.setup.target_2 or scored.setup.target_price,
                "rr": round(scored.setup.risk_reward_ratio, 1),
                "strategy_type": scored.setup.strategy_type,
                "timeframe": "5min",
            }))

        # Take up to N qualifying trades (not just the top 1)
        trades_this_scan = 0
        for scored in best:
            if trades_this_scan >= self.max_intraday_per_scan:
                break
            if not self._can_trade_intraday(bar_time):
                break
            if scored.setup.symbol in {p.symbol for p in self.positions}:
                continue
            if scored.composite_score < 55:
                continue

            self.emit(SimEvent("agent_verdict", bar_time, {
                "symbol": scored.setup.symbol,
                "pattern": scored.setup.pattern_name,
                "verdict": "CONFIRMED",
                "reasoning": f"Score {scored.composite_score:.0f} | R:R {scored.setup.risk_reward_ratio:.1f} | {scored.setup.strategy_type} | strat_mult={self._sizer.get_strategy_multiplier(scored.setup.pattern_name):.2f}",
                "action": "DET+",
            }))

            self._open_trade(scored, bar_time, timeframe="5min")
            trades_this_scan += 1

    # ── Trade Execution ──────────────────────────────────────────────────────

    def _open_trade(self, scored: ScoredSetup, bar_time: str, timeframe: str = "5min"):
        setup = scored.setup

        shares, dollar_risk, size_mod, sizing_reason = self._sizer.calculate(
            entry=setup.entry_price, stop=setup.stop_loss,
            equity=self.current_equity, cash=self.cash,
            heat_pct=self.total_heat_pct, pattern=setup.pattern_name,
            max_heat=self.max_heat_pct, max_daily_risk_pct=self.max_daily_risk_pct,
            daily_risk_used=self._daily_risk_used,
        )

        if shares <= 0:
            return

        self._trade_counter += 1
        pos_id = f"{'D' if timeframe == '1d' else 'T'}{self._trade_counter:04d}"

        slippage = abs(setup.entry_price - setup.stop_loss) * 0.01
        entry = setup.entry_price + slippage if setup.bias.value == "long" else setup.entry_price - slippage

        pos = Position(
            id=pos_id, symbol=setup.symbol, pattern_name=setup.pattern_name,
            strategy_type=setup.strategy_type, bias=setup.bias.value,
            entry_price=round(entry, 4), stop_loss=setup.stop_loss,
            original_stop=setup.stop_loss,
            target_1=setup.target_1 or setup.target_price,
            target_2=setup.target_2 or setup.target_price,
            shares=shares, dollar_risk=dollar_risk,
            entry_time=bar_time, composite_score=scored.composite_score,
            timeframe=timeframe, size_modifier=size_mod,
            splits=setup.position_splits,
            current_price=entry,
        )

        self.cash -= shares * entry
        self.positions.append(pos)
        self._last_trade_time = bar_time
        self._daily_risk_used += dollar_risk
        self._daily_trades_taken += 1

        self.emit(SimEvent("trade_open", bar_time, {
            **pos.to_dict(),
            "sizing": sizing_reason,
            "heat_after": round(self.total_heat_pct, 1),
            "cash_remaining": round(self.cash, 2),
        }))

    def _close_position(self, pos: Position, bar_time: str,
                        outcome: str, realized_r: float, exit_price: float):
        position_cost = pos.shares * pos.entry_price
        pnl = realized_r * pos.dollar_risk
        self.cash += position_cost + pnl
        self.cumulative_r += realized_r
        self.cumulative_pnl += pnl

        # Feed back into learning systems
        self._sizer.record_trade(pos.pattern_name, realized_r)
        self._evaluator.record_outcome(TradeOutcome(
            pattern_name=pos.pattern_name, strategy_type=pos.strategy_type,
            symbol=pos.symbol, bias=pos.bias,
            entry_price=pos.entry_price, target_price=pos.target_1,
            stop_price=pos.original_stop, outcome=outcome,
            realized_r=realized_r, timestamp=bar_time,
        ))

        trade = {
            "id": pos.id, "symbol": pos.symbol, "pattern": pos.pattern_name,
            "strategy_type": pos.strategy_type, "bias": pos.bias,
            "timeframe": pos.timeframe,
            "entry": pos.entry_price, "exit": exit_price,
            "stop": pos.original_stop, "t1": pos.target_1, "t2": pos.target_2,
            "shares": pos.shares, "dollar_risk": pos.dollar_risk,
            "size_modifier": pos.size_modifier,
            "entry_time": pos.entry_time, "exit_time": bar_time,
            "outcome": outcome, "realized_r": realized_r,
            "pnl": round(pnl, 2), "bars_held": pos.bars_held,
            "t1_hit": pos.t1_hit, "t2_hit": pos.t2_hit,
            "score": pos.composite_score,
        }
        self.closed_trades.append(trade)
        self.positions = [p for p in self.positions if p.id != pos.id]

        self.emit(SimEvent("trade_close", bar_time, {
            **trade,
            "cumulative_r": round(self.cumulative_r, 3),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
        }))

    async def _resolve_positions(self, bar_time: str, bars_this_tick: dict):
        to_close = []
        for pos in self.positions:
            raw = bars_this_tick.get(pos.symbol)
            if not raw:
                continue
            bar = Bar(
                symbol=pos.symbol,
                timestamp=datetime.strptime(raw["t"], "%Y-%m-%dT%H:%M:%S"),
                open=raw["o"], high=raw["h"], low=raw["l"], close=raw["c"],
                volume=int(raw["v"]),
            )
            result = pos.check_bar(bar)
            if result is not None:
                outcome, realized_r = result
                to_close.append((pos, outcome, realized_r, bar.close))
            else:
                self.emit(SimEvent("position_update", bar_time, pos.to_dict()))

        for pos, outcome, realized_r, close_price in to_close:
            self._close_position(pos, bar_time, outcome, realized_r, close_price)

    async def _close_intraday_eod(self, bar_time: str, bars_this_tick: dict):
        """Close only intraday (5min) positions. Daily positions carry."""
        for pos in [p for p in self.positions if p.timeframe == "5min"]:
            raw = bars_this_tick.get(pos.symbol, {})
            price = raw.get("c", pos.current_price) if raw else pos.current_price
            outcome, realized_r = pos.force_close(price)
            self._close_position(pos, bar_time, outcome, realized_r, price)

    def _emit_pnl(self, bar_time: str, bar_idx: int, total_bars: int):
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions)
        equity = self.current_equity

        snap = {
            "bar_index": bar_idx, "total_bars": total_bars,
            "time": bar_time, "equity": round(equity, 2),
            "cash": round(self.cash, 2),
            "cumulative_r": round(self.cumulative_r, 3),
            "unrealized_r": round(sum(p.unrealized_r for p in self.positions), 3),
            "total_r": round(self.cumulative_r + sum(p.unrealized_r for p in self.positions), 3),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl": round(self.cumulative_pnl + unrealized_pnl, 2),
            "positions": len(self.positions),
            "intraday_positions": len(self.intraday_positions),
            "daily_positions": len(self.daily_positions),
            "closed_trades": len(self.closed_trades),
            "heat_pct": round(self.total_heat_pct, 1),
            "day_number": self._day_number,
        }
        self.pnl_history.append(snap)
        # Keep only last ~2 days of bar-level PnL to avoid unbounded growth
        # (~78 bars/day × 2 = ~156 entries). Final results use equity_curve.
        if len(self.pnl_history) > 200:
            self.pnl_history = self.pnl_history[-200:]
        self.emit(SimEvent("pnl", bar_time, snap))

    # ── Results & Logging ────────────────────────────────────────────────────

    def _save_results(self, dates: list[str]):
        out_dir = Path("simulation/output")
        out_dir.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        equity = self.current_equity
        wins = sum(1 for t in self.closed_trades if t["realized_r"] > 0)
        losses = len(self.closed_trades) - wins
        total = len(self.closed_trades)

        gross_win = sum(t["realized_r"] for t in self.closed_trades if t["realized_r"] > 0)
        gross_loss = abs(sum(t["realized_r"] for t in self.closed_trades if t["realized_r"] <= 0)) or 1

        # Per-timeframe stats
        intraday_trades = [t for t in self.closed_trades if t.get("timeframe") == "5min"]
        daily_trades = [t for t in self.closed_trades if t.get("timeframe") == "1d"]

        results = {
            "config": {
                "starting_capital": self.starting_capital,
                "risk_per_trade_pct": self.risk_pct,
                "sim_days": len(dates),
                "universe_size": 500,
                "use_agents": False,
                "mode": "deterministic_plus",
                "max_heat_pct": self.max_heat_pct,
                "max_daily_risk_pct": self.max_daily_risk_pct,
                "max_positions": self.max_positions,
                "min_score": self.min_score,
            },
            "stats": {
                "total_trades": total, "wins": wins, "losses": losses,
                "total_r": round(self.cumulative_r, 3),
                "total_pnl": round(self.cumulative_pnl, 2),
                "win_rate": round((wins / total) * 100, 1) if total > 0 else 0,
                "avg_r": round(self.cumulative_r / total, 3) if total > 0 else 0,
                "avg_pnl": round(self.cumulative_pnl / total, 2) if total > 0 else 0,
                "profit_factor": round(gross_win / gross_loss, 2),
                "final_equity": round(equity, 2),
                "return_pct": round(((equity - self.starting_capital) / self.starting_capital) * 100, 2),
                "best_trade_r": round(max((t["realized_r"] for t in self.closed_trades), default=0), 3),
                "worst_trade_r": round(min((t["realized_r"] for t in self.closed_trades), default=0), 3),
                "open_positions": len(self.positions),
                "current_heat": round(self.total_heat_pct, 1),
                "intraday_trades": len(intraday_trades),
                "daily_trades": len(daily_trades),
                "open_daily_positions": len(self.daily_positions),
            },
            "strategy_performance": self._sizer.get_summary(),
            "equity_curve": self._equity_curve,
            "closed_trades": self.closed_trades,
        }

        # Main results file
        (out_dir / "det_plus_results.json").write_text(json.dumps(results, indent=2, default=str))

        # Detailed daily log
        (LOG_DIR / "daily_log.json").write_text(json.dumps(self._day_logs, indent=2, default=str))

        # Strategy performance log
        (LOG_DIR / "strategy_performance.json").write_text(
            json.dumps(self._sizer.get_summary(), indent=2, default=str))

        print(f"\n{'='*60}")
        print(f"DETERMINISTIC+ RESULTS")
        print(f"{'='*60}")
        print(f"Days: {self._day_number} | Trades: {total} ({len(intraday_trades)} intraday, {len(daily_trades)} daily)")
        print(f"Win Rate: {results['stats']['win_rate']}% | Profit Factor: {results['stats']['profit_factor']}")
        print(f"Total P&L: ${self.cumulative_pnl:+,.2f} | Return: {results['stats']['return_pct']}%")
        print(f"Final Equity: ${equity:,.2f}")
        print(f"\nStrategy Performance:")
        for s in self._sizer.get_summary()[:10]:
            print(f"  {s['strategy']:30s} | {s['trades']:3d} trades | {s['win_rate']:5.1f}% win | "
                  f"avg_r={s['avg_r']:+.3f} | recent={s['recent_avg_r']:+.3f} | mult={s['size_mult']:.2f}x")
        print(f"\nResults saved to simulation/output/det_plus_results.json")
        print(f"Logs saved to {LOG_DIR}/")

    # ── Control ──────────────────────────────────────────────────────────────

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False

    def set_speed(self, speed: float):
        self.playback_speed = max(0.1, min(100.0, speed))
