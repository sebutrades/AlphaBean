"""
simulation/custom/engine.py — Custom Agent Trading engine.

Background-persistent intraday simulation that:
  - Runs independently of WebSocket connections
  - Supports strategy filtering, adaptive sizing, agent deliberation
  - Logs everything for run comparison
  - Streams events to any number of subscribers
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

from simulation.custom.config import CustomSimConfig
from simulation.custom.sizing import AdaptiveSizer, SizeResult
from simulation.custom.strategy_filter import classify_filtered
from simulation.custom.agents.deliberation import DeliberationEngine, DeliberationResult
from simulation.custom import run_store

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# Data source paths
BAR_DATA_DIR = Path("cache/bar_data")
LIVE_DATA_DIR = Path("live_data_cache/data/5min")
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"


def _native(obj):
    """Convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


@dataclass
class Position:
    """An open position during simulation."""
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
    size_modifier: float = 1.0
    splits: tuple = (0.5, 0.3, 0.2)

    # Agent context
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
                self.stop_loss = self.entry_price
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
            "size_modifier": self.size_modifier,
            "t1_hit": self.t1_hit, "t2_hit": self.t2_hit,
            "bars_held": self.bars_held,
            "current_price": round(self.current_price, 2),
            "unrealized_r": round(self.unrealized_r, 3),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "high_water_r": round(self.high_water, 3),
            "remaining_weight": round(self.remaining_weight, 2),
            "verdict": self.agent_verdict,
        }


def _load_bar_data_for_date(date: str) -> dict[str, list[dict]]:
    """Load 5min bars for all symbols on a given date."""
    symbol_bars: dict[str, list[dict]] = {}

    if BAR_DATA_DIR.exists():
        for fpath in BAR_DATA_DIR.glob("*_5min.json"):
            try:
                data = json.loads(fpath.read_text())
                bars = []
                for b in data.get("bars", []):
                    if b["t"][:10] != date:
                        continue
                    bar_time = b["t"][11:16]
                    if MARKET_OPEN <= bar_time < MARKET_CLOSE:
                        bars.append(b)
                if bars:
                    sym = data.get("symbol", fpath.stem.split("_")[0]).upper()
                    symbol_bars[sym] = bars
            except Exception:
                continue

    if not symbol_bars and LIVE_DATA_DIR.exists():
        for fpath in LIVE_DATA_DIR.glob("*.json"):
            try:
                data = json.loads(fpath.read_text())
                bars = []
                for b in data.get("bars", []):
                    if b["t"][:10] != date:
                        continue
                    bar_time = b["t"][11:16]
                    if MARKET_OPEN <= bar_time < MARKET_CLOSE:
                        bars.append(b)
                if bars:
                    sym = fpath.stem.upper()
                    symbol_bars[sym] = bars
            except Exception:
                continue

    return symbol_bars


def get_available_dates() -> list[str]:
    """Return dates that have 5min bar data."""
    dates: set[str] = set()
    if BAR_DATA_DIR.exists():
        spy_path = BAR_DATA_DIR / "SPY_5min.json"
        if spy_path.exists():
            try:
                data = json.loads(spy_path.read_text())
                for b in data.get("bars", []):
                    bar_time = b["t"][11:16]
                    if MARKET_OPEN <= bar_time < MARKET_CLOSE:
                        dates.add(b["t"][:10])
            except Exception:
                pass
    if not dates and LIVE_DATA_DIR.exists():
        for fpath in LIVE_DATA_DIR.glob("*.json"):
            try:
                data = json.loads(fpath.read_text())
                for b in data.get("bars", []):
                    dates.add(b["t"][:10])
            except Exception:
                pass
    return sorted(dates)


class CustomSimEngine:
    """Background-persistent intraday simulation engine.

    Unlike IntradaySimulation, this engine:
      - Runs independently of WebSocket (background asyncio task)
      - Supports multiple subscribers (connect/disconnect freely)
      - Uses adaptive sizing, strategy filtering, agent deliberation
      - Logs everything for run comparison
    """

    def __init__(self, config: CustomSimConfig, run_id: str):
        self.config = config
        self.run_id = run_id

        # Portfolio state
        self.capital = config.starting_capital
        self.starting_capital = config.starting_capital
        self.cash = config.starting_capital
        self.positions: list[Position] = []
        self.closed_trades: list[dict] = []
        self.cumulative_r: float = 0.0
        self.cumulative_pnl: float = 0.0

        # Sizing
        self.sizer = AdaptiveSizer(config.sizing, config.starting_capital)

        # Agent deliberation
        self.deliberation = DeliberationEngine(config, emit=self._emit_agent_event)

        # Strategy evaluator
        self._evaluator = StrategyEvaluator()

        # Counters
        self._trade_counter: int = 0
        self._last_trade_time: str = ""
        self._day_number: int = 0
        self._trades_today: int = 0
        self._equity_curve: list[dict] = []
        self._agent_logs: list[dict] = []

        # Multi-day tracking
        self._prev_day_r: float = 0.0
        self._prev_day_pnl: float = 0.0
        self._prev_trade_count: int = 0

        # Control
        self._running = False
        self._paused = False
        self.status = "pending"  # pending, running, paused, completed, stopped, error

        # Subscriber queues for live event streaming
        self._subscribers: list[asyncio.Queue] = []
        self._event_buffer: list[dict] = []  # last N events for replay on reconnect
        self._max_buffer = 200

    # ── Subscriber Management ────────────────────────────────────────────────

    def add_subscriber(self) -> asyncio.Queue:
        """Add a new event subscriber (e.g., a WebSocket connection)."""
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        # Send buffered recent events for catchup
        for event in self._event_buffer[-50:]:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                break
        self._subscribers.append(q)
        return q

    def remove_subscriber(self, q: asyncio.Queue):
        """Remove a subscriber queue."""
        if q in self._subscribers:
            self._subscribers.remove(q)

    def _emit(self, event: dict):
        """Broadcast event to all subscribers and buffer it."""
        event = _native(event)
        self._event_buffer.append(event)
        if len(self._event_buffer) > self._max_buffer:
            self._event_buffer = self._event_buffer[-self._max_buffer:]

        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

    def _emit_agent_event(self, data: dict):
        """Callback for deliberation engine to emit agent events."""
        self._emit(data)
        self._agent_logs.append(data)
        # Prevent unbounded growth
        if len(self._agent_logs) > 200:
            self._agent_logs = self._agent_logs[-200:]

    # ── Portfolio Helpers ────────────────────────────────────────────────────

    @property
    def total_heat_pct(self) -> float:
        total_risk = sum(p.dollar_risk * p.remaining_weight for p in self.positions)
        equity = self.cash + sum(p.shares * p.current_price for p in self.positions)
        return (total_risk / max(equity, 1)) * 100.0

    @property
    def current_equity(self) -> float:
        return self.cash + sum(p.shares * p.current_price for p in self.positions)

    def _can_trade(self, bar_time: str) -> bool:
        if not self._last_trade_time:
            return True
        if len(self.positions) >= self.config.sizing.max_positions:
            return False
        if self.total_heat_pct >= self.config.sizing.max_heat_pct:
            return False
        if self._trades_today >= self.config.max_trades_per_day:
            return False
        try:
            last = datetime.strptime(self._last_trade_time, "%Y-%m-%dT%H:%M:%S")
            current = datetime.strptime(bar_time, "%Y-%m-%dT%H:%M:%S")
            diff_minutes = (current - last).total_seconds() / 60
            return diff_minutes >= self.config.rate_limit_minutes
        except Exception:
            return True

    def _get_portfolio_context(self) -> dict:
        """Build context dict for agent deliberation."""
        return {
            "cash": round(self.cash, 2),
            "equity": round(self.current_equity, 2),
            "positions": len(self.positions),
            "max_positions": self.config.sizing.max_positions,
            "heat_pct": round(self.total_heat_pct, 1),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "cumulative_r": round(self.cumulative_r, 3),
            "open_symbols": [p.symbol for p in self.positions],
        }

    # ── Main Run Loop ────────────────────────────────────────────────────────

    async def run(self):
        """Main entry point — runs the full simulation."""
        self._running = True
        self.status = "running"

        dates = self.config.dates or get_available_dates()
        if not dates:
            self.status = "error"
            self._emit({"type": "error", "message": "No dates available"})
            return

        # Save initial run record
        run_store.save_run(
            self.run_id, self.config.to_dict(), {}, [], [], [], [],
            status="running",
        )

        # Check API key availability if agents are enabled
        import os
        has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
        agents_mode = "agents" if (self.config.use_agents and has_api_key) else "deterministic"
        if self.config.use_agents and not has_api_key:
            agents_mode = "deterministic (no API key)"

        self._emit({"type": "run_start", "run_id": self.run_id, "total_days": len(dates),
                     "config": self.config.to_dict(), "agents_mode": agents_mode})

        try:
            for i, date in enumerate(dates):
                if not self._running:
                    break
                await self._run_day(date)
                self._last_trade_time = ""
                self._trades_today = 0

                # Save progress every 5 days
                if (i + 1) % 5 == 0:
                    self._save_progress(i + 1, len(dates))

        except Exception as e:
            self.status = "error"
            self._emit({"type": "error", "message": str(e)})

        # Final save
        if self._running:
            self.status = "completed"
        else:
            self.status = "stopped"

        self._save_final(dates)

        self._emit({"type": "run_end", "run_id": self.run_id, "status": self.status,
                     "stats": self._get_stats()})

    async def _run_day(self, date: str):
        """Run one trading day."""
        self._day_number += 1
        symbol_bars = _load_bar_data_for_date(date)

        if not symbol_bars:
            self._emit({"type": "day_skip", "date": date, "reason": "No data"})
            return

        all_bar_times: set[str] = set()
        for bars in symbol_bars.values():
            for b in bars:
                all_bar_times.add(b["t"])

        sorted_times = sorted(all_bar_times)
        total_bars = len(sorted_times)

        self._emit({"type": "day_start", "date": date, "day_number": self._day_number,
                     "symbols": len(symbol_bars), "total_bars": total_bars,
                     "capital": round(self.cash, 2)})

        seen_bars: dict[str, list[dict]] = {sym: [] for sym in symbol_bars}
        bars_this_tick: dict[str, dict] = {}

        for bar_idx, bar_time in enumerate(sorted_times):
            if not self._running:
                break

            while self._paused:
                await asyncio.sleep(0.1)

            # Collect bars at this timestamp
            bars_this_tick = {}
            for sym, bars in symbol_bars.items():
                for b in bars:
                    if b["t"] == bar_time:
                        seen_bars[sym].append(b)
                        bars_this_tick[sym] = b
                        break

            # Emit bar event
            spy_bar = bars_this_tick.get("SPY", {})
            self._emit({"type": "bar", "timestamp": bar_time, "bar_index": bar_idx,
                         "total_bars": total_bars, "spy_price": spy_bar.get("c", 0),
                         "symbols_active": len(bars_this_tick)})

            # 1. Resolve positions
            await self._resolve_positions(bar_time, bars_this_tick)

            # 2. Scan and trade (after warmup)
            if bar_idx >= 12 and self._can_trade(bar_time):
                await self._scan_and_trade(bar_time, seen_bars, bars_this_tick)

            # 3. PnL snapshot
            self._emit_pnl(bar_time, bar_idx, total_bars)

            # Playback delay
            delay = 0.5 / max(0.1, self.config.playback_speed)
            await asyncio.sleep(delay)

        # EOD
        if self.config.close_eod:
            await self._close_all_eod(sorted_times[-1] if sorted_times else "", bars_this_tick)

        day_pnl = self.cumulative_pnl - self._prev_day_pnl
        equity = round(self.current_equity, 2)

        self._equity_curve.append({
            "date": date, "day_number": self._day_number,
            "equity": equity,
            "cumulative_r": round(self.cumulative_r, 3),
            "day_r": round(self.cumulative_r - self._prev_day_r, 3),
            "day_pnl": round(day_pnl, 2),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "trades_today": len(self.closed_trades) - self._prev_trade_count,
        })

        self._prev_day_r = self.cumulative_r
        self._prev_day_pnl = self.cumulative_pnl
        self._prev_trade_count = len(self.closed_trades)

        self._emit({"type": "day_end", "date": date, "day_number": self._day_number,
                     "day_pnl": round(day_pnl, 2), "cumulative_pnl": round(self.cumulative_pnl, 2),
                     "equity": equity, "total_trades": len(self.closed_trades)})

    # ── Position Resolution ──────────────────────────────────────────────────

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
                self._emit({"type": "position_update", "timestamp": bar_time, **pos.to_dict()})

        for pos, outcome, realized_r, close_price in to_close:
            self._close_position(pos, bar_time, outcome, realized_r, close_price)

    # ── Scanning & Trading ───────────────────────────────────────────────────

    async def _scan_and_trade(self, bar_time: str, seen_bars: dict, bars_this_tick: dict):
        candidates: list[ScoredSetup] = []

        for sym, bars_list in seen_bars.items():
            if len(bars_list) < 20:
                continue
            if sym in {p.symbol for p in self.positions}:
                continue

            bar_objects = [
                Bar(symbol=sym,
                    timestamp=datetime.strptime(b["t"], "%Y-%m-%dT%H:%M:%S"),
                    open=b["o"], high=b["h"], low=b["l"], close=b["c"],
                    volume=int(b["v"]))
                for b in bars_list
            ]
            bar_series = BarSeries(symbol=sym, timeframe="5min", bars=bar_objects)

            # Use filtered classify if strategies are selected
            setups = classify_filtered(bar_series, self.config.allowed_strategies)
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
                if scored.composite_score >= self.config.min_composite_score:
                    candidates.append(scored)

        if not candidates:
            return

        candidates.sort(key=lambda x: x.composite_score, reverse=True)
        best = candidates[:self.config.deliberation.max_candidates]

        # Emit detected setups
        for scored in best:
            self._emit({"type": "setup_detected", "timestamp": bar_time,
                         "symbol": scored.setup.symbol, "pattern": scored.setup.pattern_name,
                         "bias": scored.setup.bias.value, "score": round(scored.composite_score, 1),
                         "entry": scored.setup.entry_price, "stop": scored.setup.stop_loss,
                         "t1": scored.setup.target_1 or scored.setup.target_price,
                         "t2": scored.setup.target_2 or scored.setup.target_price,
                         "rr": round(scored.setup.risk_reward_ratio, 1)})

        # Agent or deterministic evaluation
        if self.config.use_agents:
            await self._agent_trade(best, bar_time)
        else:
            await self._deterministic_trade(best, bar_time)

    async def _agent_trade(self, candidates: list[ScoredSetup], bar_time: str):
        """Use the deliberation engine for agent-driven trade decisions.

        Falls back to deterministic if agents fail (no API key, credit issues, etc).
        """
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            # No API key — fall back to deterministic with a warning
            if not getattr(self, "_warned_no_key", False):
                self._warned_no_key = True
                self._emit({"type": "agent_thinking", "timestamp": bar_time,
                             "agent": "system", "symbol": "", "pattern": "",
                             "message": "No ANTHROPIC_API_KEY set — falling back to deterministic mode"})
            await self._deterministic_trade(candidates, bar_time)
            return

        ctx = self._get_portfolio_context()
        regime = "unknown"

        result: DeliberationResult = await self.deliberation.deliberate(
            candidates, ctx, regime
        )

        # If agents returned nothing but we had candidates, fall back to deterministic
        if not result.approved_trades and candidates:
            # Check if it was a total failure (all errors)
            agent_errors = [m for m in result.messages if "Error" in str(m.content.get("reasoning", ""))]
            if len(agent_errors) == len(result.messages) and result.messages:
                self._emit({"type": "agent_thinking", "timestamp": bar_time,
                             "agent": "system", "symbol": "", "pattern": "",
                             "message": "Agent calls failed — falling back to deterministic for this scan"})
                await self._deterministic_trade(candidates, bar_time)
                return

        # Open approved trades
        setup_map = {(s.setup.symbol, s.setup.pattern_name): s for s in candidates}
        for trade in result.approved_trades:
            key = (trade["symbol"], trade["pattern"])
            if key not in setup_map:
                continue
            if not self._can_trade(bar_time):
                break

            scored = setup_map[key]
            agent_size_mod = trade.get("size_modifier", 1.0)
            reasoning = trade.get("reasoning", "")
            if trade.get("pm_reasoning"):
                reasoning = f"PM: {trade['pm_reasoning']} | {reasoning}"
            if trade.get("risk_reasoning"):
                reasoning = f"{reasoning} | Risk: {trade['risk_reasoning']}"

            self._open_trade(scored, bar_time, "CONFIRMED", reasoning, agent_size_mod)

    async def _deterministic_trade(self, candidates: list[ScoredSetup], bar_time: str):
        """Deterministic fallback: score threshold."""
        top = candidates[0]
        setup = top.setup

        self._emit({"type": "agent_thinking", "timestamp": bar_time,
                     "symbol": setup.symbol, "pattern": setup.pattern_name,
                     "message": f"Evaluating {setup.pattern_name} on {setup.symbol} (score: {top.composite_score:.0f})..."})

        if top.composite_score >= 60:
            reasoning = (
                f"{setup.pattern_name} on {setup.symbol}: Score {top.composite_score:.0f}. "
                f"R:R = {setup.risk_reward_ratio:.1f}. Strategy: {setup.strategy_type}."
            )
            self._emit({"type": "agent_verdict", "timestamp": bar_time,
                         "symbol": setup.symbol, "pattern": setup.pattern_name,
                         "verdict": "CONFIRMED", "reasoning": reasoning, "action": "DETERMINISTIC"})
            if self._can_trade(bar_time):
                self._open_trade(top, bar_time, "CONFIRMED", reasoning)
        elif top.composite_score < 40:
            self._emit({"type": "agent_verdict", "timestamp": bar_time,
                         "symbol": setup.symbol, "pattern": setup.pattern_name,
                         "verdict": "DENIED", "reasoning": f"Score {top.composite_score:.0f} below threshold",
                         "action": "SKIP"})

    # ── Trade Execution ──────────────────────────────────────────────────────

    def _open_trade(self, scored: ScoredSetup, bar_time: str,
                    verdict: str, reasoning: str, agent_size_mod: float = 1.0):
        setup = scored.setup

        # Adaptive sizing
        size_result: SizeResult = self.sizer.calculate(
            entry_price=setup.entry_price,
            stop_price=setup.stop_loss,
            current_equity=self.current_equity,
            current_cash=self.cash,
            current_heat_pct=self.total_heat_pct,
            pattern_name=setup.pattern_name,
            agent_size_modifier=agent_size_mod,
        )

        if size_result.shares <= 0:
            return

        self._trade_counter += 1
        pos_id = f"T{self._trade_counter:04d}"

        # Slippage
        slippage = abs(setup.entry_price - setup.stop_loss) * self.config.slippage_pct
        entry = setup.entry_price + slippage if setup.bias.value == "long" else setup.entry_price - slippage

        pos = Position(
            id=pos_id, symbol=setup.symbol, pattern_name=setup.pattern_name,
            strategy_type=setup.strategy_type, bias=setup.bias.value,
            entry_price=round(entry, 4), stop_loss=setup.stop_loss,
            original_stop=setup.stop_loss,
            target_1=setup.target_1 or setup.target_price,
            target_2=setup.target_2 or setup.target_price,
            shares=size_result.shares, dollar_risk=size_result.dollar_risk,
            entry_time=bar_time, composite_score=scored.composite_score,
            size_modifier=size_result.size_modifier,
            splits=setup.position_splits,
            agent_verdict=verdict, agent_reasoning=reasoning,
            current_price=entry,
        )

        self.cash -= size_result.shares * entry
        self.positions.append(pos)
        self._last_trade_time = bar_time
        self._trades_today += 1

        self._emit({"type": "trade_open", "timestamp": bar_time,
                     **pos.to_dict(), "sizing": size_result.reasoning,
                     "heat_after": round(self.total_heat_pct, 1),
                     "cash_remaining": round(self.cash, 2)})

    def _close_position(self, pos: Position, bar_time: str,
                        outcome: str, realized_r: float, exit_price: float):
        position_cost = pos.shares * pos.entry_price
        pnl = realized_r * pos.dollar_risk
        self.cash += position_cost + pnl
        self.cumulative_r += realized_r
        self.cumulative_pnl += pnl

        # Record for adaptive sizing
        self.sizer.record_trade(pos.pattern_name, realized_r)

        trade = {
            "id": pos.id, "symbol": pos.symbol, "pattern": pos.pattern_name,
            "strategy_type": pos.strategy_type, "bias": pos.bias,
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

        self._emit({"type": "trade_close", "timestamp": bar_time, **trade,
                     "cumulative_r": round(self.cumulative_r, 3),
                     "cumulative_pnl": round(self.cumulative_pnl, 2)})

    async def _close_all_eod(self, bar_time: str, bars_this_tick: dict):
        for pos in list(self.positions):
            raw = bars_this_tick.get(pos.symbol, {})
            price = raw.get("c", pos.current_price) if raw else pos.current_price
            outcome, realized_r = pos.force_close(price)
            self._close_position(pos, bar_time, outcome, realized_r, price)

    def _emit_pnl(self, bar_time: str, bar_idx: int, total_bars: int):
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions)
        equity = self.current_equity

        self._emit({"type": "pnl", "timestamp": bar_time,
                     "bar_index": bar_idx, "total_bars": total_bars,
                     "equity": round(equity, 2), "cash": round(self.cash, 2),
                     "cumulative_r": round(self.cumulative_r, 3),
                     "cumulative_pnl": round(self.cumulative_pnl, 2),
                     "unrealized_pnl": round(unrealized_pnl, 2),
                     "total_pnl": round(self.cumulative_pnl + unrealized_pnl, 2),
                     "positions": len(self.positions),
                     "closed_trades": len(self.closed_trades),
                     "heat_pct": round(self.total_heat_pct, 1),
                     "day_number": self._day_number})

    # ── Stats & Persistence ──────────────────────────────────────────────────

    def _get_stats(self) -> dict:
        total = len(self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t["realized_r"] > 0)
        losses = total - wins
        return {
            "total_trades": total,
            "wins": wins, "losses": losses,
            "win_rate": round((wins / total) * 100, 1) if total > 0 else 0,
            "total_r": round(self.cumulative_r, 3),
            "total_pnl": round(self.cumulative_pnl, 2),
            "avg_r": round(self.cumulative_r / total, 3) if total > 0 else 0,
            "final_equity": round(self.current_equity, 2),
            "profit_factor": round(
                sum(t["realized_r"] for t in self.closed_trades if t["realized_r"] > 0) /
                abs(sum(t["realized_r"] for t in self.closed_trades if t["realized_r"] <= 0) or 1),
                2),
            "days_simulated": self._day_number,
        }

    def _save_progress(self, day: int, total: int):
        run_store.save_run_progress(
            self.run_id, day, total,
            self._equity_curve, self.closed_trades,
            self._agent_logs[-100:],
            self.sizer.get_strategy_summary(),
            self._get_stats(),
        )

    def _save_final(self, dates: list[str]):
        run_store.save_run(
            self.run_id, self.config.to_dict(),
            self._get_stats(), self._equity_curve,
            self.closed_trades, self._agent_logs[-200:],
            self.sizer.get_strategy_summary(),
            status=self.status,
        )

    # ── Control ──────────────────────────────────────────────────────────────

    def pause(self):
        self._paused = True
        self.status = "paused"

    def resume(self):
        self._paused = False
        self.status = "running"

    def stop(self):
        self._running = False

    def set_speed(self, speed: float):
        self.config.playback_speed = max(0.1, min(100.0, speed))
