"""
simulation/intraday.py — Intraday bar-by-bar simulation engine.

Walks through a trading day using 5min bars, one at a time:
  - Detects setups as each bar arrives
  - Agents evaluate setups before trading
  - Active position management: re-evaluate on each bar
  - Rate-limited: max 1 trade per 5 minutes
  - Streams all events via callback for WebSocket broadcast

This is the "Agent Trading" visual mode — designed to be watched in real-time.
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
from backend.strategies.evaluator import StrategyEvaluator
from backend.structures.indicators import wilder_atr

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


# ── Event types for WebSocket streaming ──────────────────────────────────────

def _native(obj):
    """Convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    return obj


@dataclass
class SimEvent:
    """An event emitted during intraday simulation."""
    type: str           # bar, setup, agent_thinking, agent_verdict, trade_open,
                        # trade_close, position_update, pnl, day_start, day_end, error
    timestamp: str      # bar timestamp
    data: dict          # event-specific payload

    def to_dict(self) -> dict:
        return {"type": self.type, "timestamp": self.timestamp, "data": _native(self.data)}


# ── Position for intraday ────────────────────────────────────────────────────

@dataclass
class IntradayPosition:
    """An open position during intraday simulation."""
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
    splits: tuple = (0.5, 0.3, 0.2)

    # Agent reasoning
    agent_verdict: str = ""
    agent_reasoning: str = ""

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
        """Update current price and unrealized R."""
        self.current_price = price
        risk = abs(self.entry_price - self.original_stop)
        if risk > 0:
            if self.bias == "long":
                self.unrealized_r = (price - self.entry_price) / risk
            else:
                self.unrealized_r = (self.entry_price - price) / risk
        self.high_water = max(self.high_water, self.unrealized_r)
        self.bars_held += 1

    def check_bar(self, bar: Bar) -> Optional[tuple[str, float]]:
        """Check exits against bar. Returns (outcome, r) if closed."""
        risk = abs(self.entry_price - self.original_stop)
        if risk <= 0:
            return ("loss", -1.0)

        is_long = self.bias == "long"

        # Stop check
        if (is_long and bar.low <= self.stop_loss) or \
           (not is_long and bar.high >= self.stop_loss):
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

        # T2 check
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
        """Force-close at price (EOD, agent decision)."""
        risk = abs(self.entry_price - self.original_stop)
        if risk <= 0:
            return ("closed", 0.0)
        if self.bias == "long":
            remaining_r = (price - self.entry_price) / risk
        else:
            remaining_r = (self.entry_price - price) / risk
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
        weighted_r = max(-10.0, min(10.0, weighted_r))
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
            "id": self.id, "symbol": self.symbol, "pattern": self.pattern_name,
            "strategy_type": self.strategy_type, "bias": self.bias,
            "entry": self.entry_price, "stop": self.stop_loss,
            "original_stop": self.original_stop,
            "t1": self.target_1, "t2": self.target_2,
            "shares": self.shares, "dollar_risk": self.dollar_risk,
            "entry_time": self.entry_time, "score": self.composite_score,
            "t1_hit": self.t1_hit, "t2_hit": self.t2_hit,
            "bars_held": self.bars_held,
            "current_price": round(self.current_price, 2),
            "unrealized_r": round(self.unrealized_r, 3),
            "high_water_r": round(self.high_water, 3),
            "remaining_weight": round(self.remaining_weight, 2),
            "verdict": self.agent_verdict,
            "reasoning": self.agent_reasoning,
        }


# ── Main Intraday Engine ─────────────────────────────────────────────────────

class IntradaySimulation:
    """Bar-by-bar intraday simulation with event streaming."""

    def __init__(
        self,
        emit: Callable[[SimEvent], None],
        starting_capital: float = 100_000.0,
        risk_pct: float = 1.0,
        max_heat_pct: float = 6.0,
        max_positions: int = 10,
        min_score: float = 50.0,
        playback_speed: float = 1.0,  # 1.0 = ~0.5s per bar, 5.0 = ~0.1s
        use_agents: bool = False,
    ):
        self.emit = emit
        self.capital = starting_capital
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.risk_pct = risk_pct
        self.max_heat_pct = max_heat_pct
        self.max_positions = max_positions
        self.min_score = min_score
        self.playback_speed = playback_speed
        self.use_agents = use_agents

        self.positions: list[IntradayPosition] = []
        self.closed_trades: list[dict] = []
        self.pnl_history: list[dict] = []
        self.cumulative_r: float = 0.0
        self._trade_counter: int = 0
        self._last_trade_time: str = ""
        self._evaluator = StrategyEvaluator()
        self._running = False
        self._paused = False

    @property
    def total_heat_pct(self) -> float:
        total_risk = sum(p.dollar_risk * p.remaining_weight for p in self.positions)
        return (total_risk / self.starting_capital) * 100.0

    def _can_trade(self, bar_time: str) -> bool:
        """Rate limit: at most 1 trade per 5 minutes."""
        if not self._last_trade_time:
            return True
        if len(self.positions) >= self.max_positions:
            return False
        if self.total_heat_pct >= self.max_heat_pct:
            return False
        try:
            last = datetime.strptime(self._last_trade_time, "%Y-%m-%dT%H:%M:%S")
            current = datetime.strptime(bar_time, "%Y-%m-%dT%H:%M:%S")
            diff_minutes = (current - last).total_seconds() / 60
            return diff_minutes >= 5
        except Exception:
            return True

    def _size_trade(self, entry: float, stop: float) -> tuple[int, float]:
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return 0, 0.0
        dollar_budget = self.starting_capital * self.risk_pct / 100.0
        # Cap by remaining heat
        max_risk = self.starting_capital * self.max_heat_pct / 100.0
        current_risk = sum(p.dollar_risk * p.remaining_weight for p in self.positions)
        dollar_budget = min(dollar_budget, max(0, max_risk - current_risk))
        shares = math.floor(dollar_budget / risk_per_share)
        if shares <= 0:
            return 0, 0.0
        # Cap at 15% of capital
        max_pos = self.starting_capital * 0.15
        if shares * entry > max_pos:
            shares = math.floor(max_pos / entry)
        # Cash check
        if shares * entry > self.cash:
            shares = math.floor(self.cash / entry)
        if shares <= 0:
            return 0, 0.0
        return shares, round(shares * risk_per_share, 2)

    async def run_day(self, date: str):
        """Run simulation for one trading day.

        Loads all 5min bars for the date from cache, then walks through
        bar-by-bar, detecting setups, evaluating, and managing positions.
        """
        self._running = True
        bar_dir = Path("live_data_cache/data/5min")

        # Load all available symbols' bars for this date
        symbol_bars: dict[str, list[dict]] = {}
        all_bar_times: set[str] = set()

        for fpath in bar_dir.glob("*.json"):
            try:
                data = json.loads(fpath.read_text())
                bars = [b for b in data.get("bars", []) if b["t"][:10] == date]
                if bars:
                    sym = fpath.stem.upper()
                    symbol_bars[sym] = bars
                    for b in bars:
                        all_bar_times.add(b["t"])
            except Exception:
                continue

        if not symbol_bars:
            self.emit(SimEvent("error", "", {"message": f"No 5min data for {date}"}))
            return

        sorted_times = sorted(all_bar_times)
        total_bars = len(sorted_times)

        self.emit(SimEvent("day_start", sorted_times[0], {
            "date": date,
            "symbols": len(symbol_bars),
            "total_bars": total_bars,
            "capital": self.starting_capital,
        }))

        # Build cumulative bar series per symbol as we walk forward
        seen_bars: dict[str, list[dict]] = {sym: [] for sym in symbol_bars}

        for bar_idx, bar_time in enumerate(sorted_times):
            if not self._running:
                break

            while self._paused:
                await asyncio.sleep(0.1)

            # Collect this bar for all symbols
            bars_this_tick: dict[str, dict] = {}
            for sym, bars in symbol_bars.items():
                for b in bars:
                    if b["t"] == bar_time:
                        seen_bars[sym].append(b)
                        bars_this_tick[sym] = b
                        break

            # Emit bar event
            spy_bar = bars_this_tick.get("SPY", {})
            self.emit(SimEvent("bar", bar_time, {
                "bar_index": bar_idx,
                "total_bars": total_bars,
                "spy_price": spy_bar.get("c", 0),
                "symbols_active": len(bars_this_tick),
            }))

            # 1. Resolve existing positions
            await self._resolve_positions(bar_time, bars_this_tick)

            # 2. Scan for new setups (every 6 bars = 30 min to avoid noise)
            if bar_idx >= 12 and self._can_trade(bar_time):
                await self._scan_and_trade(bar_time, seen_bars, bars_this_tick)

            # 3. Emit PNL snapshot
            self._emit_pnl(bar_time, bar_idx, total_bars)

            # Playback delay
            delay = 0.5 / max(0.1, self.playback_speed)
            await asyncio.sleep(delay)

        # End of day — close all positions
        await self._close_all_eod(sorted_times[-1] if sorted_times else "", bars_this_tick)

        self.emit(SimEvent("day_end", sorted_times[-1] if sorted_times else "", {
            "total_trades": len(self.closed_trades),
            "cumulative_r": round(self.cumulative_r, 3),
            "final_equity": round(self.cash + sum(
                p.shares * p.current_price for p in self.positions
            ), 2),
            "wins": sum(1 for t in self.closed_trades if t["realized_r"] > 0),
            "losses": sum(1 for t in self.closed_trades if t["realized_r"] <= 0),
        }))

    async def _resolve_positions(self, bar_time: str, bars_this_tick: dict):
        """Check all positions against current bars."""
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
                # Emit position update
                self.emit(SimEvent("position_update", bar_time, pos.to_dict()))

        for pos, outcome, realized_r, close_price in to_close:
            self._close_position(pos, bar_time, outcome, realized_r, close_price)

    async def _scan_and_trade(self, bar_time: str, seen_bars: dict, bars_this_tick: dict):
        """Scan all symbols for setups, evaluate, and potentially trade."""
        # Only scan symbols with enough bars
        candidates: list[ScoredSetup] = []

        for sym, bars_list in seen_bars.items():
            if len(bars_list) < 20:
                continue
            if sym in {p.symbol for p in self.positions}:
                continue  # already have position

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

            # Compute features and score
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

        # Sort by score, emit setup events
        candidates.sort(key=lambda x: x.composite_score, reverse=True)
        best = candidates[:5]  # top 5 for evaluation

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
            }))

        # Evaluate and trade the best one
        top = best[0]
        verdict, reasoning = await self._evaluate_setup(top, bar_time)

        if verdict == "CONFIRMED" and self._can_trade(bar_time):
            self._open_trade(top, bar_time, verdict, reasoning)
        elif verdict == "DENIED":
            self.emit(SimEvent("agent_verdict", bar_time, {
                "symbol": top.setup.symbol,
                "pattern": top.setup.pattern_name,
                "verdict": "DENIED",
                "reasoning": reasoning,
                "action": "SKIP",
            }))

    async def _evaluate_setup(self, scored: ScoredSetup, bar_time: str) -> tuple[str, str]:
        """Evaluate a setup — agent or deterministic."""
        setup = scored.setup
        symbol = setup.symbol
        pattern = setup.pattern_name

        self.emit(SimEvent("agent_thinking", bar_time, {
            "symbol": symbol,
            "pattern": pattern,
            "score": round(scored.composite_score, 1),
            "message": f"Evaluating {pattern} on {symbol} (score: {scored.composite_score:.0f})...",
        }))

        if self.use_agents:
            try:
                from simulation.agents.analyst import evaluate_setup_ollama
                result = await evaluate_setup_ollama(scored, regime="unknown")
                return result.get("verdict", "CAUTION"), result.get("reasoning", "")
            except Exception as e:
                return "CAUTION", f"Agent error: {e}"

        # Deterministic: CONFIRM if score > 60, DENY if < 40
        if scored.composite_score >= 60:
            reasoning = (
                f"{pattern} on {symbol}: Strong composite score of {scored.composite_score:.0f}. "
                f"R:R = {setup.risk_reward_ratio:.1f}. "
                f"Strategy: {setup.strategy_type}. Pattern confidence: {setup.confidence:.0%}."
            )
            return "CONFIRMED", reasoning
        elif scored.composite_score < 40:
            return "DENIED", f"Score {scored.composite_score:.0f} below threshold"
        else:
            reasoning = (
                f"{pattern} on {symbol}: Marginal score of {scored.composite_score:.0f}. "
                f"R:R = {setup.risk_reward_ratio:.1f}. Needs stronger confirmation."
            )
            return "CAUTION", reasoning

    def _open_trade(self, scored: ScoredSetup, bar_time: str, verdict: str, reasoning: str):
        """Open a new position."""
        setup = scored.setup
        shares, dollar_risk = self._size_trade(setup.entry_price, setup.stop_loss)
        if shares <= 0:
            return

        self._trade_counter += 1
        pos_id = f"T{self._trade_counter:04d}"

        # Slippage
        slippage = abs(setup.entry_price - setup.stop_loss) * 0.01
        entry = setup.entry_price + slippage if setup.bias.value == "long" else setup.entry_price - slippage

        pos = IntradayPosition(
            id=pos_id, symbol=setup.symbol, pattern_name=setup.pattern_name,
            strategy_type=setup.strategy_type, bias=setup.bias.value,
            entry_price=round(entry, 4), stop_loss=setup.stop_loss,
            original_stop=setup.stop_loss,
            target_1=setup.target_1 or setup.target_price,
            target_2=setup.target_2 or setup.target_price,
            shares=shares, dollar_risk=dollar_risk,
            entry_time=bar_time, composite_score=scored.composite_score,
            splits=setup.position_splits,
            agent_verdict=verdict, agent_reasoning=reasoning,
            current_price=entry,
        )

        self.cash -= shares * entry
        self.positions.append(pos)
        self._last_trade_time = bar_time

        self.emit(SimEvent("trade_open", bar_time, {
            **pos.to_dict(),
            "agent_reasoning": reasoning,
            "heat_after": round(self.total_heat_pct, 1),
            "cash_remaining": round(self.cash, 2),
        }))

        self.emit(SimEvent("agent_verdict", bar_time, {
            "symbol": setup.symbol,
            "pattern": setup.pattern_name,
            "verdict": verdict,
            "reasoning": reasoning,
            "action": f"OPENED {pos_id}: {shares} shares @ ${entry:.2f}",
            "score": round(scored.composite_score, 1),
            "dollar_risk": dollar_risk,
        }))

    def _close_position(self, pos: IntradayPosition, bar_time: str,
                        outcome: str, realized_r: float, exit_price: float):
        """Close a position and record the trade."""
        position_cost = pos.shares * pos.entry_price
        pnl = realized_r * pos.dollar_risk
        self.cash += position_cost + pnl
        self.cumulative_r += realized_r

        trade = {
            "id": pos.id, "symbol": pos.symbol, "pattern": pos.pattern_name,
            "strategy_type": pos.strategy_type, "bias": pos.bias,
            "entry": pos.entry_price, "exit": exit_price,
            "stop": pos.original_stop, "t1": pos.target_1, "t2": pos.target_2,
            "shares": pos.shares, "dollar_risk": pos.dollar_risk,
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
            "cash_after": round(self.cash, 2),
        }))

    async def _close_all_eod(self, bar_time: str, bars_this_tick: dict):
        """Close all positions at end of day."""
        for pos in list(self.positions):
            raw = bars_this_tick.get(pos.symbol, {})
            price = raw.get("c", pos.current_price) if raw else pos.current_price
            outcome, realized_r = pos.force_close(price)
            self._close_position(pos, bar_time, outcome, realized_r, price)

    def _emit_pnl(self, bar_time: str, bar_idx: int, total_bars: int):
        """Emit PNL snapshot."""
        unrealized_r = sum(p.unrealized_r for p in self.positions)
        position_value = sum(p.shares * p.current_price for p in self.positions)
        equity = self.cash + position_value

        snap = {
            "bar_index": bar_idx,
            "total_bars": total_bars,
            "time": bar_time,
            "equity": round(equity, 2),
            "cash": round(self.cash, 2),
            "cumulative_r": round(self.cumulative_r, 3),
            "unrealized_r": round(unrealized_r, 3),
            "total_r": round(self.cumulative_r + unrealized_r, 3),
            "positions": len(self.positions),
            "closed_trades": len(self.closed_trades),
            "heat_pct": round(self.total_heat_pct, 1),
        }
        self.pnl_history.append(snap)
        self.emit(SimEvent("pnl", bar_time, snap))

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False

    def set_speed(self, speed: float):
        self.playback_speed = max(0.1, min(50.0, speed))


def get_available_dates() -> list[str]:
    """Return dates that have 5min bar data."""
    bar_dir = Path("live_data_cache/data/5min")
    if not bar_dir.exists():
        return []

    dates: set[str] = set()
    # Just check SPY for available dates
    spy_path = bar_dir / "SPY.json"
    if spy_path.exists():
        try:
            data = json.loads(spy_path.read_text())
            for b in data.get("bars", []):
                dates.add(b["t"][:10])
        except Exception:
            pass

    return sorted(dates)
