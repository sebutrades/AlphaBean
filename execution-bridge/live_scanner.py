"""
Live Scanner + Execution Bridge
================================
Runs the full AlphaBean scanning pipeline against live Karb price data,
detects setups, scores them, and manages paper positions with stop losses
and scaled targets.

Supports two modes:
  1. Standalone terminal mode (python live_scanner.py)
  2. WebSocket mode (imported by FastAPI, events streamed to frontend)

Flow:
  1. Fetch today's intraday bars (5min + 15min) via Karb for N symbols
  2. Fetch daily bars for longer-term context
  3. Run AlphaBean scanner (patterns + features + regime + scoring)
  4. Rank setups by composite score
  5. "Execute" top setups as paper positions
  6. Monitor live prices: adjust stops, take profit at T1/T2, safety exit
  7. Loop every scan_interval until market close or Ctrl+C

Usage:
  python live_scanner.py                         # Scan top 30 symbols
  python live_scanner.py --symbols AAPL,NVDA,TSLA  # Specific symbols
  python live_scanner.py --max-symbols 50        # Scan top 50
  python live_scanner.py --dry-run               # Show setups but don't "trade"
  python live_scanner.py --scan-interval 300     # Scan every 5min (default)
"""

import subprocess
import json
import sys
import os
import time
import signal
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

# Add AlphaBean root to path so we can import the backend
ALPHABEAN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ALPHABEAN_ROOT))
os.chdir(ALPHABEAN_ROOT)  # so relative paths in backend work

# ---------------------------------------------------------------------------
# AlphaBean imports
# ---------------------------------------------------------------------------

import numpy as np
from backend.data.schemas import Bar, BarSeries
from backend.patterns.classifier import classify_all
from backend.features.engine import compute_features
from backend.regime.detector import detect_regime
from backend.scoring.multi_factor import score_setup, ScoredSetup
from backend.strategies.evaluator import StrategyEvaluator
from backend.patterns.registry import TradeSetup, Bias

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_OPEN_POSITIONS = 100
DEFAULT_POSITION_SIZE = 10        # shares per trade
MIN_COMPOSITE_SCORE = 50          # minimum score to take a trade
SCAN_INTERVAL = 300               # seconds between scans (5 min)
SCANNER_TIMEFRAMES = ["5min", "15min"]
KARB_BATCH_SIZE = 10              # symbols per kti call

# Kore order-placer strategy (for live mode)
ORDER_PLACER_GUID = "d0d5cc8a-46c2-4bec-9b72-764820d1b6aa"
ORDER_PLACER_ACCOUNT = "203979"
BUY_ALGO  = "2b4fdc55-ff01-416e-a5ea-e1f1d4524c7d"
SELL_ALGO = "8fdee8fe-b772-46bd-b411-5544f7a0d917"


# ---------------------------------------------------------------------------
# Event system — structured events for both terminal + WebSocket
# ---------------------------------------------------------------------------

def _native(obj):
    """Convert numpy/dataclass types to JSON-safe Python natives."""
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    return obj


@dataclass
class LiveEvent:
    """A structured event emitted during live scanning."""
    type: str           # log, scan_start, scan_done, setup_found, trade_open,
                        # trade_close, position_update, stop_hit, target_hit,
                        # pnl_snapshot, session_summary, error
    timestamp: str
    data: dict

    def to_dict(self) -> dict:
        return {"type": self.type, "timestamp": self.timestamp, "data": _native(self.data)}

# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------

class T:
    B = "\033[1m"; D = "\033[2m"; X = "\033[0m"
    G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
    C = "\033[96m"; M = "\033[95m"; W = "\033[97m"
    BG_G = "\033[42m"; BG_R = "\033[41m"; BG_Y = "\033[43m"

    @staticmethod
    def header(text):
        w = 74
        print(f"\n{T.B}{T.C}{'=' * w}")
        print(f"  {text}")
        print(f"{'=' * w}{T.X}\n")

    @staticmethod
    def sub(text):
        print(f"\n  {T.B}{T.M}--- {text} ---{T.X}\n")

    @staticmethod
    def ok(text):   print(f"  {T.G}[OK]{T.X}    {text}")
    @staticmethod
    def warn(text): print(f"  {T.Y}[!!]{T.X}    {text}")
    @staticmethod
    def fail(text): print(f"  {T.R}[FAIL]{T.X}  {text}")
    @staticmethod
    def info(text): print(f"  {T.D}[..]{T.X}    {text}")
    @staticmethod
    def trade(text): print(f"  {T.B}{T.W}[TRADE]{T.X} {text}")
    @staticmethod
    def stop(text): print(f"  {T.BG_R}{T.W} STOP {T.X}  {text}")
    @staticmethod
    def target(text): print(f"  {T.BG_G}{T.W} T1/T2 {T.X} {text}")

    @staticmethod
    def progress(cur, tot, label=""):
        pct = cur / tot * 100 if tot else 0
        bar_len = 25
        filled = int(bar_len * cur / tot) if tot else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {cur}/{tot}  {label}    ")
        sys.stdout.flush()
        if cur >= tot:
            print()


# ---------------------------------------------------------------------------
# Karb data fetcher (reuse from test_data_fetch)
# ---------------------------------------------------------------------------

def _run_kti(args, timeout=120):
    cmd = ["kti"] + args + ["--json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None, r.stderr.strip()
        return json.loads(r.stdout), None
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except json.JSONDecodeError as e:
        return None, f"JSON parse: {e}"


def fetch_bars_karb(symbols, timeframe, date=None, days=None, minutes=None):
    """Fetch bars from Karb and convert to AlphaBean BarSeries dict."""
    sym_str = ",".join(symbols)

    if timeframe == "1d" or timeframe == "daily":
        args = ["price", "daily", sym_str]
        if days:
            args += ["--days", str(days)]
        elif date:
            args += ["--date", date]
        else:
            args += ["--days", "90"]
    else:
        args = ["price", "minute", sym_str]
        if date:
            args += ["--date", date]
        elif minutes:
            args += ["--minutes", str(minutes)]
        else:
            args += ["--minutes", "780"]  # ~2 full trading days

    data, err = _run_kti(args, timeout=180)
    if err:
        return {}, err

    result = {}
    if data and "bars" in data:
        for sym, bars_raw in data["bars"].items():
            ab_bars = []
            for b in bars_raw:
                ab_bars.append(Bar(
                    symbol=sym,
                    timestamp=datetime.fromisoformat(b["timestamp"]),
                    open=b["open"],
                    high=b["high"],
                    low=b["low"],
                    close=b["close"],
                    volume=int(b["volume"]),
                ))
            result[sym] = ab_bars
    return result, None


def filter_rth(bars_1m):
    """Filter to Regular Trading Hours only (9:30-16:00 ET).
    Karb returns timestamps in Eastern time."""
    from datetime import time as dtime
    rth_open = dtime(9, 30)
    rth_close = dtime(16, 0)
    return [b for b in bars_1m if rth_open <= b.timestamp.time() < rth_close]


def aggregate_minute_to_tf(minute_bars, tf_minutes):
    """Aggregate 1-min bars into N-minute bars."""
    if not minute_bars:
        return []
    buckets = defaultdict(list)
    for bar in minute_bars:
        ts = bar.timestamp
        floored = ts.replace(
            minute=(ts.minute // tf_minutes) * tf_minutes,
            second=0, microsecond=0
        )
        buckets[floored].append(bar)

    agg = []
    for ts in sorted(buckets.keys()):
        bb = buckets[ts]
        agg.append(Bar(
            symbol=bb[0].symbol,
            timestamp=ts,
            open=bb[0].open,
            high=max(b.high for b in bb),
            low=min(b.low for b in bb),
            close=bb[-1].close,
            volume=sum(b.volume for b in bb),
        ))
    return agg


# ---------------------------------------------------------------------------
# Position tracker
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    bias: str              # "long" or "short"
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    trail_type: str
    trail_param: float
    splits: tuple
    shares: int
    remaining_shares: int
    pattern_name: str
    composite_score: float
    timeframe: str
    entered_at: datetime
    status: str = "ACTIVE"  # ACTIVE, AT_T1, AT_T2, CLOSED
    pnl: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    t1_hit: bool = False
    t2_hit: bool = False
    original_stop: float = 0.0

    def risk_r(self):
        """1R = distance from entry to stop."""
        return abs(self.entry_price - self.original_stop)

    def current_r(self, price):
        """Current P&L in R-multiples."""
        r1 = self.risk_r()
        if r1 <= 0:
            return 0.0
        if self.bias == "long":
            return (price - self.entry_price) / r1
        else:
            return (self.entry_price - price) / r1


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class LiveScanner:
    def __init__(self, symbols, dry_run=False, scan_interval=SCAN_INTERVAL,
                 emit: Optional[Callable] = None, live_mode=False):
        self.symbols = symbols
        self.dry_run = dry_run
        self.live_mode = live_mode    # True = send real orders via Kore
        self.scan_interval = scan_interval
        self.positions = {}           # symbol -> Position
        self.closed_positions = []    # list of closed Position
        self.order_count = 0
        self.scan_count = 0
        self.evaluator = self._load_evaluator()
        self.bt_scores = self._load_backtest_scores()
        self.running = True
        self.paused = False
        self._emit_cb = emit         # external event callback (for WebSocket)
        self._equity_curve = []      # list of {timestamp, equity, pnl, trades}

    def _load_evaluator(self):
        try:
            return StrategyEvaluator()
        except Exception:
            T.warn("StrategyEvaluator not available, using neutral scores")
            return None

    def _load_backtest_scores(self):
        cache = ALPHABEAN_ROOT / "cache" / "backtest_results.json"
        if not cache.exists():
            return {}
        try:
            data = json.loads(cache.read_text())
            patterns = data.get("patterns", {})
            scores = {}
            for name, stats in patterns.items():
                if isinstance(stats, dict) and "edge_score" in stats:
                    scores[name] = float(stats["edge_score"])
            return scores
        except Exception:
            return {}

    # ----------------------------------------------------------------
    # Event emission
    # ----------------------------------------------------------------

    def emit(self, event_type: str, data: dict):
        """Emit a structured event to callback and terminal."""
        event = LiveEvent(
            type=event_type,
            timestamp=datetime.now().isoformat(),
            data=data,
        )
        if self._emit_cb:
            try:
                self._emit_cb(event)
            except Exception:
                pass

    def stop(self):
        self.running = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def _build_pnl_snapshot(self):
        """Build a snapshot of current P&L state for the equity curve."""
        total_unrealized = 0.0
        total_realized = sum(p.pnl for p in self.closed_positions)
        return {
            "timestamp": datetime.now().isoformat(),
            "open_positions": len(self.positions),
            "closed_trades": len(self.closed_positions),
            "total_orders": self.order_count,
            "realized_pnl": round(total_realized, 2),
            "positions": [
                {
                    "symbol": p.symbol, "bias": p.bias,
                    "entry_price": float(p.entry_price), "stop_loss": float(p.stop_loss),
                    "target_1": float(p.target_1), "target_2": float(p.target_2),
                    "shares": p.remaining_shares, "pattern": p.pattern_name,
                    "score": float(p.composite_score), "timeframe": p.timeframe,
                    "status": p.status, "t1_hit": p.t1_hit, "t2_hit": p.t2_hit,
                    "entered_at": p.entered_at.isoformat(),
                } for p in self.positions.values()
            ],
            "closed": [
                {
                    "symbol": p.symbol, "bias": p.bias,
                    "entry_price": float(p.entry_price), "exit_price": float(p.exit_price),
                    "pnl": round(float(p.pnl), 2), "exit_reason": p.exit_reason,
                    "pattern": p.pattern_name, "score": float(p.composite_score),
                } for p in self.closed_positions
            ],
        }

    # ----------------------------------------------------------------
    # Kore order placement (live mode)
    # ----------------------------------------------------------------

    def _place_kore_order(self, symbol, side, qty, intent="init", signal_id=""):
        """Place a real order via the Kore order-placer strategy.

        Calls: kti forward start <guid> --symbol <sym> --account 203979
                 --params '{"side":"buy","qty":10,...}'
        Returns (success: bool, message: str).
        """
        algo = BUY_ALGO if side == "buy" else SELL_ALGO
        params = json.dumps({
            "side": side,
            "qty": qty,
            "intent": intent,
            "algo": algo,
            "signal_id": signal_id,
        })
        cmd = [
            "kti", "forward", "start", ORDER_PLACER_GUID,
            "--account", ORDER_PLACER_ACCOUNT,
            "--symbol", symbol,
            "--params", params,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = (proc.stdout or "").strip()
            if proc.returncode == 0:
                self.emit("log", {
                    "level": "ok",
                    "message": f"[KORE] {side.upper()} {qty}x {symbol} sent (intent={intent})",
                })
                return True, output
            else:
                err = (proc.stderr or output)[:200]
                self.emit("log", {
                    "level": "error",
                    "message": f"[KORE] Order failed: {err}",
                })
                return False, err
        except subprocess.TimeoutExpired:
            self.emit("log", {"level": "error", "message": f"[KORE] Timeout placing {side} {symbol}"})
            return False, "TIMEOUT"
        except Exception as e:
            self.emit("log", {"level": "error", "message": f"[KORE] Error: {e}"})
            return False, str(e)

    # ----------------------------------------------------------------
    # Data fetching
    # ----------------------------------------------------------------

    def fetch_all_bars(self):
        """Fetch minute bars for all symbols, aggregate to 5min + 15min."""
        T.sub("Fetching live bar data from Karb")
        self.emit("log", {"level": "info", "message": "Fetching live bar data from Karb..."})

        all_bars = {}  # symbol -> {"5min": BarSeries, "15min": BarSeries}

        # Fetch in batches
        for i in range(0, len(self.symbols), KARB_BATCH_SIZE):
            batch = self.symbols[i:i + KARB_BATCH_SIZE]
            T.progress(i, len(self.symbols), f"Batch: {batch[0]}..{batch[-1]}")
            self.emit("fetch_progress", {
                "current": i, "total": len(self.symbols),
                "batch": batch,
            })

            # Fetch today's minute bars + some history
            raw, err = fetch_bars_karb(batch, "minute", minutes=2340)  # ~6 trading days
            if err:
                T.fail(f"Karb error: {err}")
                self.emit("error", {"message": f"Karb error: {err}"})
                continue

            for sym, bars_1m in raw.items():
                # Filter to Regular Trading Hours (9:30-16:00 ET) before aggregation
                # Karb returns extended hours bars that confuse pattern detectors
                bars_rth = filter_rth(bars_1m)
                bars_5m = aggregate_minute_to_tf(bars_rth, 5)
                bars_15m = aggregate_minute_to_tf(bars_rth, 15)

                if len(bars_5m) >= 20:
                    all_bars[sym] = {
                        "5min": BarSeries(symbol=sym, timeframe="5min", bars=bars_5m),
                        "15min": BarSeries(symbol=sym, timeframe="15min", bars=bars_15m),
                    }

        T.progress(len(self.symbols), len(self.symbols), "Done")
        T.ok(f"Got bar data for {len(all_bars)}/{len(self.symbols)} symbols")
        self.emit("log", {"level": "ok", "message": f"Got bar data for {len(all_bars)}/{len(self.symbols)} symbols"})
        return all_bars

    # ----------------------------------------------------------------
    # Scanning
    # ----------------------------------------------------------------

    def scan_all(self, all_bars):
        """Run AlphaBean scanner on all symbols."""
        T.sub("Running AlphaBean Scanner")
        all_setups = []
        errors = 0

        for i, (sym, tf_bars) in enumerate(all_bars.items()):
            T.progress(i, len(all_bars), f"Scanning {sym}")

            for tf_key in SCANNER_TIMEFRAMES:
                bars = tf_bars.get(tf_key)
                if not bars or len(bars.bars) < 20:
                    continue

                try:
                    # Features
                    closes = np.array([b.close for b in bars.bars], dtype=np.float64)
                    highs = np.array([b.high for b in bars.bars], dtype=np.float64)
                    lows = np.array([b.low for b in bars.bars], dtype=np.float64)
                    volumes = np.array([b.volume for b in bars.bars], dtype=np.float64)
                    features = compute_features(closes, highs, lows, volumes)

                    # Regime
                    regime = detect_regime(closes, highs, lows, is_spy=False)

                    # Patterns — classify_all runs on fresh bars each scan,
                    # so all returned setups are current. No time filter needed.
                    setups = classify_all(bars)
                    if not setups:
                        continue

                    # Score
                    for setup in setups:
                        bt = self.bt_scores.get(setup.pattern_name, 50.0)
                        scored = score_setup(setup, features, regime, self.evaluator, bt)
                        scored.setup.timeframe_detected = tf_key.replace("min", "m")
                        if scored.composite_score >= MIN_COMPOSITE_SCORE:
                            all_setups.append(scored)

                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        T.warn(f"{sym}/{tf_key}: {type(e).__name__}: {e}")

        T.progress(len(all_bars), len(all_bars), "Done")

        # Deduplicate: same symbol+pattern+bias -> keep highest score
        seen = {}
        for s in all_setups:
            key = (s.setup.symbol, s.setup.pattern_name, s.setup.bias.value)
            if key not in seen or s.composite_score > seen[key].composite_score:
                seen[key] = s
        deduped = sorted(seen.values(), key=lambda x: x.composite_score, reverse=True)

        T.ok(f"Found {len(deduped)} unique setups (score >= {MIN_COMPOSITE_SCORE})")
        if errors:
            T.warn(f"{errors} scan errors (suppressed after 3)")

        # Emit scan results
        self.emit("scan_done", {
            "total_setups": len(deduped),
            "scan_number": self.scan_count,
            "setups": [
                {
                    "symbol": s.setup.symbol,
                    "pattern": s.setup.pattern_name,
                    "bias": s.setup.bias.value,
                    "entry_price": float(s.setup.entry_price),
                    "stop_loss": float(s.setup.stop_loss),
                    "target_1": float(s.setup.target_1),
                    "target_2": float(s.setup.target_2),
                    "risk_reward": float(s.setup.risk_reward_ratio),
                    "composite_score": round(float(s.composite_score), 1),
                    "pattern_confidence": round(float(s.pattern_confidence_score), 1),
                    "feature_score": round(float(s.feature_score), 1),
                    "strategy_score": round(float(s.strategy_score), 1),
                    "regime_score": round(float(s.regime_alignment_score), 1),
                    "backtest_score": round(float(s.backtest_edge_score), 1),
                    "volume_score": round(float(s.volume_confirm_score), 1),
                    "rr_score": round(float(s.rr_quality_score), 1),
                    "timeframe": getattr(s.setup, 'timeframe_detected', ''),
                    "in_position": s.setup.symbol in self.positions,
                } for s in deduped[:20]
            ],
        })
        return deduped

    # ----------------------------------------------------------------
    # Display setups
    # ----------------------------------------------------------------

    def display_setups(self, setups):
        if not setups:
            T.info("No setups detected this scan")
            return

        T.sub(f"Top Setups ({len(setups)} total)")

        for i, s in enumerate(setups[:20]):
            su = s.setup
            bias_color = T.G if su.bias.value == "long" else T.R
            bias_arrow = "LONG ^" if su.bias.value == "long" else "SHORT v"
            risk = abs(su.entry_price - su.stop_loss)
            in_pos = su.symbol in self.positions

            print(
                f"  {T.B}#{i+1:2d}{T.X}  "
                f"{bias_color}{bias_arrow}{T.X}  "
                f"{T.B}{su.symbol:6s}{T.X}  "
                f"Score={T.B}{s.composite_score:5.1f}{T.X}  "
                f"{su.pattern_name:20s}  "
                f"E=${su.entry_price:.2f}  S=${su.stop_loss:.2f}  "
                f"T1=${su.target_1:.2f}  T2=${su.target_2:.2f}  "
                f"R:R={su.risk_reward_ratio:.1f}  "
                f"TF={su.timeframe_detected}  "
                f"{'[IN POS]' if in_pos else ''}"
            )
            # Scoring breakdown on second line
            sc = s
            print(
                f"         "
                f"Conf={sc.pattern_confidence_score:4.0f}  "
                f"Feat={sc.feature_score:4.0f}  "
                f"Strat={sc.strategy_score:4.0f}  "
                f"Regime={sc.regime_alignment_score:4.0f}  "
                f"BT={sc.backtest_edge_score:4.0f}  "
                f"Vol={sc.volume_confirm_score:4.0f}  "
                f"R:R={sc.rr_quality_score:4.0f}"
            )

    # ----------------------------------------------------------------
    # Position management
    # ----------------------------------------------------------------

    def enter_positions(self, setups):
        """Enter positions for top setups not already in portfolio."""
        entered = 0
        for s in setups:
            if self.order_count >= MAX_OPEN_POSITIONS:
                T.warn(f"Max orders ({MAX_OPEN_POSITIONS}) reached, stopping")
                break
            if len(self.positions) >= MAX_OPEN_POSITIONS:
                break

            su = s.setup
            if su.symbol in self.positions:
                continue  # already in this symbol

            # Determine position size
            shares = DEFAULT_POSITION_SIZE
            risk_per_share = abs(su.entry_price - su.stop_loss)
            if risk_per_share <= 0:
                continue

            # Create position
            pos = Position(
                symbol=su.symbol,
                bias=su.bias.value,
                entry_price=su.entry_price,
                stop_loss=su.stop_loss,
                original_stop=su.stop_loss,
                target_1=su.target_1 if su.target_1 > 0 else su.target_price,
                target_2=su.target_2 if su.target_2 > 0 else su.target_price,
                trail_type=su.trail_type,
                trail_param=su.trail_param,
                splits=su.position_splits,
                shares=shares,
                remaining_shares=shares,
                pattern_name=su.pattern_name,
                composite_score=s.composite_score,
                timeframe=su.timeframe_detected,
                entered_at=datetime.now(),
            )
            # In live mode, place real order via Kore before tracking locally
            if self.live_mode:
                kore_side = "buy" if su.bias.value == "long" else "sell"
                signal_id = f"AB-{su.symbol}-{su.pattern_name}-{self.scan_count}"
                ok, msg = self._place_kore_order(
                    symbol=su.symbol,
                    side=kore_side,
                    qty=shares,
                    intent="init",
                    signal_id=signal_id,
                )
                if not ok:
                    T.fail(f"Kore order failed for {su.symbol}: {msg}")
                    continue  # skip this setup, don't track it

            self.positions[su.symbol] = pos
            self.order_count += 1
            entered += 1

            side = "BUY" if su.bias.value == "long" else "SELL SHORT"
            mode_tag = "[LIVE]" if self.live_mode else "[PAPER]"
            T.trade(
                f"{mode_tag} {side} {shares} x {su.symbol} @ ${su.entry_price:.2f}  "
                f"Stop=${su.stop_loss:.2f}  T1=${pos.target_1:.2f}  T2=${pos.target_2:.2f}  "
                f"Pattern={su.pattern_name}  Score={s.composite_score:.1f}  "
                f"[Order #{self.order_count}]"
            )

            self.emit("trade_open", {
                "symbol": su.symbol, "side": side, "bias": su.bias.value,
                "shares": shares, "entry_price": float(su.entry_price),
                "stop_loss": float(su.stop_loss), "target_1": float(pos.target_1),
                "target_2": float(pos.target_2), "pattern": su.pattern_name,
                "score": round(float(s.composite_score), 1),
                "timeframe": su.timeframe_detected,
                "order_number": self.order_count,
                "live_mode": self.live_mode,
            })

        if entered:
            T.ok(f"Entered {entered} new positions ({len(self.positions)} total open)")

    def check_positions(self):
        """Monitor all open positions against live prices."""
        if not self.positions:
            return

        T.sub(f"Position Monitor ({len(self.positions)} open)")

        # Fetch current prices for all position symbols
        pos_symbols = list(self.positions.keys())
        prices = {}

        for i in range(0, len(pos_symbols), KARB_BATCH_SIZE):
            batch = pos_symbols[i:i + KARB_BATCH_SIZE]
            data, err = fetch_bars_karb(batch, "minute", minutes=2)
            if err:
                continue
            for sym, bars in data.items():
                if bars:
                    prices[sym] = bars[-1].close

        to_close = []
        for sym, pos in self.positions.items():
            price = prices.get(sym)
            if not price:
                T.info(f"{sym:6s}  NO PRICE DATA")
                continue

            r_current = pos.current_r(price)
            pnl_dollars = (price - pos.entry_price) * pos.remaining_shares
            if pos.bias == "short":
                pnl_dollars = (pos.entry_price - price) * pos.remaining_shares

            # Check stop loss
            if pos.bias == "long" and price <= pos.stop_loss:
                pos.exit_price = price
                pos.exit_reason = "STOP_LOSS"
                pos.pnl = pnl_dollars
                pos.status = "CLOSED"
                to_close.append(sym)
                T.stop(
                    f"{sym:6s}  STOPPED OUT  "
                    f"Entry=${pos.entry_price:.2f} Exit=${price:.2f}  "
                    f"P&L=${pnl_dollars:+.2f} ({r_current:+.1f}R)  "
                    f"{pos.pattern_name}"
                )
                self.emit("stop_hit", {
                    "symbol": sym, "bias": pos.bias, "entry_price": float(pos.entry_price),
                    "exit_price": float(price), "pnl": round(float(pnl_dollars), 2),
                    "r_multiple": round(float(r_current), 2), "pattern": pos.pattern_name,
                })
                continue

            if pos.bias == "short" and price >= pos.stop_loss:
                pos.exit_price = price
                pos.exit_reason = "STOP_LOSS"
                pos.pnl = pnl_dollars
                pos.status = "CLOSED"
                to_close.append(sym)
                T.stop(
                    f"{sym:6s}  STOPPED OUT  "
                    f"Entry=${pos.entry_price:.2f} Exit=${price:.2f}  "
                    f"P&L=${pnl_dollars:+.2f} ({r_current:+.1f}R)  "
                    f"{pos.pattern_name}"
                )
                self.emit("stop_hit", {
                    "symbol": sym, "bias": pos.bias, "entry_price": float(pos.entry_price),
                    "exit_price": float(price), "pnl": round(float(pnl_dollars), 2),
                    "r_multiple": round(float(r_current), 2), "pattern": pos.pattern_name,
                })
                continue

            # Check target 1
            if not pos.t1_hit:
                t1_hit = (pos.bias == "long" and price >= pos.target_1) or \
                         (pos.bias == "short" and price <= pos.target_1)
                if t1_hit:
                    pos.t1_hit = True
                    pos.status = "AT_T1"
                    # Move stop to breakeven
                    pos.stop_loss = pos.entry_price
                    # Partial exit at T1
                    exit_shares = int(pos.shares * pos.splits[0])
                    pos.remaining_shares -= exit_shares
                    self.order_count += 1
                    T.target(
                        f"{sym:6s}  T1 HIT!  "
                        f"Sold {exit_shares} @ ${price:.2f}  "
                        f"Stop -> breakeven (${pos.entry_price:.2f})  "
                        f"Remaining: {pos.remaining_shares}  "
                        f"P&L so far: {r_current:+.1f}R  [Order #{self.order_count}]"
                    )
                    self.emit("target_hit", {
                        "symbol": sym, "target": "T1", "price": float(price),
                        "shares_sold": exit_shares, "remaining": pos.remaining_shares,
                        "r_multiple": round(float(r_current), 2), "new_stop": float(pos.entry_price),
                    })

                    # Partial exit via Kore in live mode
                    if self.live_mode and exit_shares > 0:
                        exit_side = "sell" if pos.bias == "long" else "buy"
                        self._place_kore_order(
                            symbol=sym, side=exit_side, qty=exit_shares,
                            intent="decrease",
                            signal_id=f"AB-T1-{sym}",
                        )

            # Check target 2
            if pos.t1_hit and not pos.t2_hit:
                t2_hit = (pos.bias == "long" and price >= pos.target_2) or \
                         (pos.bias == "short" and price <= pos.target_2)
                if t2_hit:
                    pos.t2_hit = True
                    pos.status = "AT_T2"
                    exit_shares = int(pos.shares * pos.splits[1])
                    pos.remaining_shares -= exit_shares
                    self.order_count += 1
                    T.target(
                        f"{sym:6s}  T2 HIT!  "
                        f"Sold {exit_shares} @ ${price:.2f}  "
                        f"Remaining: {pos.remaining_shares} on trail  "
                        f"P&L: {r_current:+.1f}R  [Order #{self.order_count}]"
                    )
                    self.emit("target_hit", {
                        "symbol": sym, "target": "T2", "price": float(price),
                        "shares_sold": exit_shares, "remaining": pos.remaining_shares,
                        "r_multiple": round(float(r_current), 2),
                    })

                    # Partial exit via Kore in live mode
                    if self.live_mode and exit_shares > 0:
                        exit_side = "sell" if pos.bias == "long" else "buy"
                        self._place_kore_order(
                            symbol=sym, side=exit_side, qty=exit_shares,
                            intent="decrease",
                            signal_id=f"AB-T2-{sym}",
                        )

            # If remaining shares <= 0, close out
            if pos.remaining_shares <= 0:
                pos.exit_price = price
                pos.exit_reason = "TARGETS_HIT"
                pos.pnl = pnl_dollars
                pos.status = "CLOSED"
                to_close.append(sym)
                continue

            # Status line for open positions
            stop_dist = abs(price - pos.stop_loss) / price * 100
            t1_dist = abs(pos.target_1 - price) / price * 100 if not pos.t1_hit else 0

            color = T.G if pnl_dollars >= 0 else T.R
            print(
                f"  {color}{'>' if pnl_dollars >= 0 else '<'}{T.X}  "
                f"{sym:6s}  "
                f"{'LONG' if pos.bias == 'long' else 'SHORT':5s}  "
                f"E=${pos.entry_price:.2f}  Now=${price:.2f}  "
                f"P&L=${pnl_dollars:+.2f}({r_current:+.1f}R)  "
                f"Stop=${pos.stop_loss:.2f}({stop_dist:.1f}%)  "
                f"{'T1 HIT' if pos.t1_hit else f'T1={t1_dist:.1f}%'}  "
                f"{'T2 HIT' if pos.t2_hit else ''}  "
                f"Shares={pos.remaining_shares}  "
                f"{pos.pattern_name}"
            )

        # Close positions
        for sym in to_close:
            pos = self.positions.pop(sym)
            self.closed_positions.append(pos)
            self.order_count += 1  # exit order

            # Send exit order to Kore in live mode
            if self.live_mode:
                exit_side = "sell" if pos.bias == "long" else "buy"
                self._place_kore_order(
                    symbol=sym,
                    side=exit_side,
                    qty=pos.remaining_shares,
                    intent="exit",
                    signal_id=f"AB-EXIT-{sym}-{pos.exit_reason}",
                )
            self.emit("trade_close", {
                "symbol": pos.symbol, "bias": pos.bias,
                "entry_price": float(pos.entry_price), "exit_price": float(pos.exit_price),
                "pnl": round(float(pos.pnl), 2), "exit_reason": pos.exit_reason,
                "pattern": pos.pattern_name, "score": float(pos.composite_score),
            })

        # Emit P&L snapshot after position check
        self.emit("pnl_snapshot", self._build_pnl_snapshot())

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------

    def print_summary(self):
        T.header("SESSION SUMMARY")

        # Open positions
        if self.positions:
            T.sub(f"Open Positions ({len(self.positions)})")
            for sym, pos in self.positions.items():
                print(f"  {sym:6s}  {pos.bias:5s}  E=${pos.entry_price:.2f}  "
                      f"Stop=${pos.stop_loss:.2f}  Shares={pos.remaining_shares}  "
                      f"{pos.pattern_name}")

        # Closed positions
        if self.closed_positions:
            T.sub(f"Closed Positions ({len(self.closed_positions)})")

            total_pnl = 0
            wins = 0
            losses = 0
            for pos in self.closed_positions:
                r = pos.current_r(pos.exit_price) if pos.exit_price else 0
                color = T.G if pos.pnl >= 0 else T.R
                result = "WIN" if pos.pnl >= 0 else "LOSS"
                if pos.pnl >= 0:
                    wins += 1
                else:
                    losses += 1
                total_pnl += pos.pnl
                print(
                    f"  {color}{result:4s}{T.X}  {pos.symbol:6s}  "
                    f"E=${pos.entry_price:.2f} -> ${pos.exit_price:.2f}  "
                    f"P&L=${pos.pnl:+.2f}  "
                    f"Reason={pos.exit_reason}  {pos.pattern_name}"
                )

            print()
            total = wins + losses
            wr = wins / total * 100 if total else 0
            T.info(f"Trades:   {total}  (W={wins} L={losses}  WR={wr:.0f}%)")
            T.info(f"Total P&L: ${total_pnl:+.2f}")

        print()
        T.info(f"Total orders:    {self.order_count}")
        T.info(f"Scans run:       {self.scan_count}")
        T.info(f"Open positions:  {len(self.positions)}")
        T.info(f"Closed trades:   {len(self.closed_positions)}")

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------

    def _run_one_cycle(self):
        """Run a single scan cycle. Returns False if no data available."""
        self.scan_count += 1
        now = datetime.now()

        T.header(f"SCAN #{self.scan_count}  {now.strftime('%H:%M:%S')}")
        self.emit("scan_start", {
            "scan_number": self.scan_count,
            "symbols": len(self.symbols),
            "open_positions": len(self.positions),
        })

        # 1) Fetch bars
        all_bars = self.fetch_all_bars()

        if not all_bars:
            T.warn("No bar data available. Market may be closed.")
            self.emit("log", {"level": "warn", "message": "No bar data available. Market may be closed."})
            return False

        # 2) Run scanner
        setups = self.scan_all(all_bars)

        # 3) Display top setups
        self.display_setups(setups)

        # 4) Enter new positions (unless dry run)
        if not self.dry_run and setups:
            self.enter_positions(setups)

        # 5) Check existing positions
        if self.positions:
            self.check_positions()
        else:
            # Emit snapshot even with no positions
            self.emit("pnl_snapshot", self._build_pnl_snapshot())

        # 6) Status
        T.sub("Status")
        T.info(f"Open positions: {len(self.positions)}")
        T.info(f"Total orders:   {self.order_count}")
        T.info(f"Closed trades:  {len(self.closed_positions)}")

        if self.order_count >= MAX_OPEN_POSITIONS:
            T.warn("Max order limit reached. Monitoring only.")

        return True

    def run(self):
        """Terminal mode: blocking loop with Ctrl+C handling."""
        T.header("ALPHABEAN LIVE SCANNER")
        T.info(f"Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        T.info(f"Symbols:        {len(self.symbols)}")
        T.info(f"Scan interval:  {self.scan_interval}s")
        T.info(f"Min score:      {MIN_COMPOSITE_SCORE}")
        T.info(f"Max orders:     {MAX_OPEN_POSITIONS}")
        T.info(f"Position size:  {DEFAULT_POSITION_SIZE} shares")
        T.info(f"Dry run:        {self.dry_run}")
        T.info(f"Timeframes:     {SCANNER_TIMEFRAMES}")
        print()

        self.emit("session_start", {
            "symbols": self.symbols, "dry_run": self.dry_run,
            "live_mode": self.live_mode,
            "scan_interval": self.scan_interval, "min_score": MIN_COMPOSITE_SCORE,
            "max_orders": MAX_OPEN_POSITIONS, "position_size": DEFAULT_POSITION_SIZE,
            "timeframes": SCANNER_TIMEFRAMES,
        })

        try:
            while self.running:
                while self.paused and self.running:
                    time.sleep(0.5)

                if not self.running:
                    break

                has_data = self._run_one_cycle()

                if not has_data:
                    T.info(f"Retrying in {self.scan_interval}s...")

                T.info(f"Next scan in {self.scan_interval}s  (Ctrl+C to stop)")
                time.sleep(self.scan_interval)

        except KeyboardInterrupt:
            print(f"\n\n  {T.B}Stopped by user.{T.X}\n")

        self._finalize()

    async def run_async(self):
        """Async mode: for WebSocket integration. Yields control with asyncio.sleep."""
        self.emit("session_start", {
            "symbols": self.symbols, "dry_run": self.dry_run,
            "live_mode": self.live_mode,
            "scan_interval": self.scan_interval, "min_score": MIN_COMPOSITE_SCORE,
            "max_orders": MAX_OPEN_POSITIONS, "position_size": DEFAULT_POSITION_SIZE,
            "timeframes": SCANNER_TIMEFRAMES,
        })

        while self.running:
            while self.paused and self.running:
                await asyncio.sleep(0.5)

            if not self.running:
                break

            # Run the blocking scan in a thread so we don't block the event loop
            has_data = await asyncio.get_event_loop().run_in_executor(
                None, self._run_one_cycle
            )

            if not self.running:
                break

            await asyncio.sleep(self.scan_interval)

        self._finalize()

    def _finalize(self):
        """Print summary and save session log."""
        self.print_summary()

        summary = self._build_pnl_snapshot()
        summary["scans"] = self.scan_count
        self.emit("session_summary", summary)

        # Save session log
        log = {
            "session_end": datetime.now().isoformat(),
            "scans": self.scan_count,
            "total_orders": self.order_count,
            "open_positions": [
                {"symbol": p.symbol, "bias": p.bias, "entry": p.entry_price,
                 "stop": p.stop_loss, "shares": p.remaining_shares, "pattern": p.pattern_name}
                for p in self.positions.values()
            ],
            "closed_trades": [
                {"symbol": p.symbol, "bias": p.bias, "entry": p.entry_price,
                 "exit": p.exit_price, "pnl": p.pnl, "reason": p.exit_reason,
                 "pattern": p.pattern_name}
                for p in self.closed_positions
            ],
        }
        log_path = Path(__file__).parent / "cache" / "session_log.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2, default=str)
        T.ok(f"Session log saved to {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Parse args
    symbols = None
    max_symbols = 30
    dry_run = "--dry-run" in sys.argv
    scan_interval = SCAN_INTERVAL

    if "--symbols" in sys.argv:
        idx = sys.argv.index("--symbols")
        if idx + 1 < len(sys.argv):
            symbols = sys.argv[idx + 1].split(",")

    if "--max-symbols" in sys.argv:
        idx = sys.argv.index("--max-symbols")
        if idx + 1 < len(sys.argv):
            max_symbols = int(sys.argv[idx + 1])

    if "--scan-interval" in sys.argv:
        idx = sys.argv.index("--scan-interval")
        if idx + 1 < len(sys.argv):
            scan_interval = int(sys.argv[idx + 1])

    # Load symbol universe
    if not symbols:
        top_file = ALPHABEAN_ROOT / "cache" / "top_symbols.json"
        if top_file.exists():
            data = json.loads(top_file.read_text())
            symbols = data["symbols"][:max_symbols]
            T.info(f"Loaded {len(symbols)} symbols from top_symbols.json")
        else:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                        "TSLA", "JPM", "V", "MA", "AMD", "AVGO", "CRM",
                        "NFLX", "COST", "UNH", "LLY", "ABBV", "PEP", "KO"]
            T.info(f"Using default {len(symbols)} symbols")

    scanner = LiveScanner(symbols, dry_run=dry_run, scan_interval=scan_interval)
    scanner.run()


if __name__ == "__main__":
    main()
