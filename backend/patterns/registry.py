"""
patterns/registry.py — Pattern definitions, TradeSetup, and metadata for all 42 detectors.

v2.1 — Timeframe routing + research-validated parameters:
  - tf: list of timeframes each pattern is allowed to run on
  - cd: cooldown in minutes (research-validated per strategy type)
  - mh: max hold in minutes (research-validated per strategy type)
  - Sources: Bulkowski "Encyclopedia of Chart Patterns" 3rd ed,
             Nison "Japanese Candlestick Charting Techniques",
             O'Neil "How to Make Money in Stocks",
             SMB Capital training materials,
             Connors & Alvarez "Short Term Trading Strategies That Work"
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Bias(str, Enum):
    LONG = "long"
    SHORT = "short"


class PatternCategory(str, Enum):
    CLASSICAL = "classical"
    CANDLESTICK = "candlestick"
    SMB_SCALP = "smb_scalp"
    QUANT = "quant"


@dataclass
class TradeSetup:
    pattern_name: str
    category: PatternCategory
    symbol: str
    bias: Bias
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    confidence: float
    detected_at: datetime
    description: str
    strategy_type: str = "breakout"
    win_rate: float = 0.0
    max_attempts: int = 1
    exit_strategy: str = ""
    key_levels: dict = field(default_factory=dict)
    ideal_time: str = ""
    timeframe_detected: str = ""
    multi_tf: bool = False

    def to_dict(self) -> dict:
        return {
            "pattern_name": self.pattern_name, "category": self.category.value,
            "symbol": self.symbol, "bias": self.bias.value,
            "entry_price": self.entry_price, "stop_loss": self.stop_loss,
            "target_price": self.target_price, "risk_reward_ratio": self.risk_reward_ratio,
            "confidence": self.confidence, "detected_at": self.detected_at.isoformat(),
            "description": self.description, "strategy_type": self.strategy_type,
            "win_rate": self.win_rate, "max_attempts": self.max_attempts,
            "exit_strategy": self.exit_strategy, "key_levels": self.key_levels,
            "ideal_time": self.ideal_time, "timeframe_detected": self.timeframe_detected,
            "multi_tf": self.multi_tf, "backtest_score": 50.0,
        }


# ==============================================================================
# PATTERN_META — Complete per-pattern configuration
# ==============================================================================
#
# Fields:
#   wr:  Historical win rate (Bulkowski/Nison/O'Neil research)
#   type: Strategy archetype (breakout, momentum, mean_reversion, scalp)
#   cat:  PatternCategory for UI grouping
#   tf:   Allowed timeframes — pattern will ONLY run on these bar sizes
#   cd:   Cooldown in minutes — minimum time between same pattern re-firing
#   mh:   Max hold in minutes — trade times out if not resolved
#
# Timeframe philosophy:
#   5min:  Scalps, candlestick reversals, intraday mean reversion
#   15min: Candlestick + structural overlap zone
#   1h:    Structural patterns — cleanest geometry, lowest noise
#   1d:    Swing/position strategies
#
# Structural patterns (H&S, triangles, wedges, etc.) run on 15min+1h because:
#   - Bulkowski's stats are based on daily/weekly (we scale down to 1h as minimum)
#   - On 5min, these patterns fire on noise — too many false zigzag swings
#   - 1h bars filter out intraday noise while keeping enough data points
#   - 15min included as overlap for faster detection of smaller formations
#
# Scalps run on 5min ONLY because:
#   - ORB is defined by first 15/30 minutes of 5min candles
#   - VWAP, gap, opening range are all intraday-specific constructs
#   - SMB playbook is explicitly designed for 5min execution
#
# Candlesticks run on 5min+15min because:
#   - Nison: candlestick patterns work on any timeframe
#   - But most reliable when combined with S/R context
#   - Our _candle_context_ok() provides that context
#   - 3-hour max hold matches typical intraday reversal window

PATTERN_META = {
    # =========================================================================
    # CLASSICAL STRUCTURAL (16) — 15min + 1h
    # =========================================================================
    # These form over multiple swings. On 5min bars, zigzag noise creates false
    # patterns. 15min is the minimum where structural geometry is reliable.
    # 1h is ideal — Bulkowski's research is on daily, 1h is the intraday proxy.

    # --- Head & Shoulders (Bulkowski: 81% reach target on daily) ---
    # Measured move = neckline to head distance, projected from neckline break
    # Formation: 1-3 months daily → scaled to 2-5 days on 1h
    # Hold: measured move takes ~1-2x formation time to play out
    "Head & Shoulders":     {"wr": 0.81, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 480, "mh": 1200},  # 8hr cd, 20hr hold

    "Inverse H&S":          {"wr": 0.83, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 480, "mh": 1200},

    # --- Double/Triple Top/Bottom (Bulkowski: 73-79%) ---
    # Two touches at same level define the pattern. Need clean S/R.
    # Measured move = peak-to-trough distance from breakout
    # Faster resolution than H&S — single measured move, not complex neckline
    "Double Top":           {"wr": 0.73, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 480, "mh": 960},   # 8hr cd, 16hr hold

    "Double Bottom":        {"wr": 0.78, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 480, "mh": 960},

    "Triple Top":           {"wr": 0.75, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 480, "mh": 960},

    "Triple Bottom":        {"wr": 0.79, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 480, "mh": 960},

    # --- Triangles (Bulkowski: 54-75%, Edwards & Magee) ---
    # Converging trendlines. Need 3+ touches per side for reliability.
    # 1h bars give cleanest trendline fits.
    # Breakout target = widest point of triangle projected from breakout
    "Ascending Triangle":   {"wr": 0.75, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 360, "mh": 960},   # 6hr cd, 16hr hold

    "Descending Triangle":  {"wr": 0.72, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 360, "mh": 960},

    "Symmetrical Triangle": {"wr": 0.54, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 360, "mh": 960},

    # --- Flags & Pennants (Bulkowski: 55-67%) ---
    # Continuation patterns after strong impulse move (the "pole")
    # Flag: parallel channel pullback. Pennant: converging triangle pullback.
    # Fastest classical patterns — pole defines magnitude, flag is brief pause
    # Works on 15min for intraday momentum, 1h for multi-day swings
    "Bull Flag":            {"wr": 0.67, "type": "momentum", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 240, "mh": 480},   # 4hr cd, 8hr hold

    "Bear Flag":            {"wr": 0.65, "type": "momentum", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 240, "mh": 480},

    "Pennant":              {"wr": 0.55, "type": "momentum", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 240, "mh": 480},

    # --- Cup & Handle (O'Neil: 68% — "most important pattern") ---
    # O'Neil: minimum 7-week cup on daily. On 1h: ~3-5 day formation minimum.
    # Handle: 1-2 weeks daily → hours on 1h. Must be shallow pullback.
    # Our detector requires min 30 bars for cup, handle vol decline, breakout vol.
    # 1h only — on 15min the cup is too noisy, zigzag creates false bowls.
    "Cup & Handle":         {"wr": 0.68, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["1h"],
                             "cd": 960, "mh": 1920},  # 16hr cd, 32hr hold (2 days)

    # --- Rectangle (Bulkowski: 70%) ---
    # Trading range between flat support and resistance.
    # Breakout target = height of rectangle projected from breakout.
    # Works on 15min for intraday ranges, 1h for multi-day consolidation.
    "Rectangle":            {"wr": 0.70, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 360, "mh": 960},

    # --- Wedges (Bulkowski: 68-69%) ---
    # Rising wedge: bearish — converging with both lines rising
    # Falling wedge: bullish — converging with both lines falling
    # Target = widest point of wedge (our v2 fix)
    # Need clean trendlines — noisy on 5min, reliable on 15min+
    "Rising Wedge":         {"wr": 0.69, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 360, "mh": 960},

    "Falling Wedge":        {"wr": 0.68, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "tf": ["15min", "1h"],
                             "cd": 360, "mh": 960},

    # =========================================================================
    # CANDLESTICK (10) — 5min + 15min
    # =========================================================================
    # Nison: candlestick patterns are timeframe-agnostic, but context matters.
    # Our _candle_context_ok() requires S/R confluence + volume + prior move.
    # 5min: fast reversals at intraday S/R
    # 15min: more reliable, fewer false signals
    # Max hold 3hr: reversal either works within 3 hours or it's dead.
    # Cooldown 1hr: can legitimately fire at different S/R levels same day.

    "Bullish Engulfing":    {"wr": 0.63, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "tf": ["5min", "15min"],
                             "cd": 60, "mh": 180},

    "Bearish Engulfing":    {"wr": 0.63, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "tf": ["5min", "15min"],
                             "cd": 60, "mh": 180},

    # Morning/Evening Star (Nison: 3-bar reversal, 65% with context)
    # Slightly longer cooldown — 3-bar pattern takes longer to form
    "Morning Star":         {"wr": 0.65, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "tf": ["5min", "15min"],
                             "cd": 90, "mh": 180},

    "Evening Star":         {"wr": 0.65, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "tf": ["5min", "15min"],
                             "cd": 90, "mh": 180},

    "Hammer":               {"wr": 0.60, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "tf": ["5min", "15min"],
                             "cd": 60, "mh": 180},

    "Shooting Star":        {"wr": 0.59, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "tf": ["5min", "15min"],
                             "cd": 60, "mh": 180},

    "Doji":                 {"wr": 0.55, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "tf": ["5min", "15min"],
                             "cd": 60, "mh": 180},

    "Dragonfly Doji":       {"wr": 0.55, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "tf": ["5min", "15min"],
                             "cd": 60, "mh": 180},

    # 3 White Soldiers / 3 Black Crows — momentum continuation candle patterns
    # Longer formation (3 bars) → slightly longer cooldown
    "Three White Soldiers": {"wr": 0.62, "type": "momentum", "cat": PatternCategory.CANDLESTICK,
                             "tf": ["5min", "15min"],
                             "cd": 120, "mh": 240},

    "Three Black Crows":    {"wr": 0.62, "type": "momentum", "cat": PatternCategory.CANDLESTICK,
                             "tf": ["5min", "15min"],
                             "cd": 120, "mh": 240},

    # =========================================================================
    # SMB SCALPS (7) — 5min ONLY
    # =========================================================================
    # These strategies use intraday-specific constructs: opening range, VWAP,
    # gap open, time-of-day filters. They're meaningless on 15min/1h/1d bars.
    # Parameters from SMB Capital's "The Playbook" and prop training materials.

    # RubberBand (SMB): Extended >2 ATR off open, bounce trade back to VWAP
    # Time window: 10:00-11:00 AM. Target: VWAP or opening price.
    # One-shot setup — if the rubber band snaps back, it's done for the day.
    "RubberBand Scalp":     {"wr": 0.625, "type": "scalp", "cat": PatternCategory.SMB_SCALP,
                             "tf": ["5min"],
                             "cd": 480, "mh": 120},   # Once/day, 2hr hold

    # ORB (Toby Crabel "Day Trading with Short Term Price Patterns")
    # Opening range is defined by first 15 or 30 minutes of 5min candles.
    # One breakout per direction per day. Resolution within 2 hours.
    # Our v2 fix: breakout only on current bar (no historical re-detection).
    "ORB 15min":            {"wr": 0.55, "type": "scalp", "cat": PatternCategory.SMB_SCALP,
                             "tf": ["5min"],
                             "cd": 960, "mh": 120},   # Once/day (16hr cd), 2hr hold

    "ORB 30min":            {"wr": 0.55, "type": "scalp", "cat": PatternCategory.SMB_SCALP,
                             "tf": ["5min"],
                             "cd": 960, "mh": 120},

    # Second Chance (SMB): Breakout, pullback to level, bounce
    # Can happen at multiple levels in one day → 4hr cooldown not once/day
    # But our v2 fix requires bounce within last 5 bars (recent only)
    "Second Chance Scalp":  {"wr": 0.525, "type": "scalp", "cat": PatternCategory.SMB_SCALP,
                             "tf": ["5min"],
                             "cd": 240, "mh": 180},   # 4hr cd, 3hr hold

    # Fashionably Late (SMB): Buy after morning dip when 9EMA crosses above VWAP
    # Specifically designed for 10:00-10:45 AM window after range establishes
    "Fashionably Late":     {"wr": 0.60, "type": "scalp", "cat": PatternCategory.SMB_SCALP,
                             "tf": ["5min"],
                             "cd": 480, "mh": 180},   # Once/day, 3hr hold

    # Gap Give & Go (SMB): Gap up > 1.5%, retrace into consolidation, break out
    # First 30 minutes only. Target: gap open + 50% of gap.
    "Gap Give & Go":        {"wr": 0.55, "type": "scalp", "cat": PatternCategory.SMB_SCALP,
                             "tf": ["5min"],
                             "cd": 480, "mh": 90},    # Once/day, 1.5hr hold

    # Tidal Wave (SMB): Support tested 3+ times with volume expansion on breakdown
    # Structural pattern but intraday execution → runs on 5min + 15min
    # Uses swing lows for support, needs clean touches — 15min cleaner than 5min
    "Tidal Wave":           {"wr": 0.55, "type": "scalp", "cat": PatternCategory.SMB_SCALP,
                             "tf": ["5min", "15min"],
                             "cd": 360, "mh": 480},   # 6hr cd, 8hr hold

    # =========================================================================
    # QUANT — INTRADAY (4) — 5min ONLY
    # =========================================================================
    # These use VWAP, intraday regime detection, and time-of-day filters.
    # VWAP resets daily — meaningless on 1h/1d bars.

    # Mean Reversion (Connors & Alvarez): Buy when price is >2 std devs from mean
    # Our version: ATR-based extension + VWAP confluence + regime filter
    # Hold: 2-4 hours. If mean hasn't reverted by then, thesis is broken.
    "Mean Reversion":       {"wr": 0.62, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"],
                             "cd": 120, "mh": 240},   # 2hr cd, 4hr hold

    # Trend Pullback: In trending regime, buy pullback to EMA near VWAP
    # Requires trending_bull regime + EMA proximity + VWAP support
    "Trend Pullback":       {"wr": 0.64, "type": "momentum", "cat": PatternCategory.QUANT,
                             "tf": ["5min"],
                             "cd": 120, "mh": 240},   # 2hr cd, 4hr hold

    # Gap Fade: Fade large gaps that lack follow-through
    # First 60 minutes only. Target: 50-100% gap fill.
    # One-shot — gap either fills or it doesn't.
    "Gap Fade":             {"wr": 0.58, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"],
                             "cd": 480, "mh": 90},    # Once/day, 1.5hr hold

    # VWAP Reversion: Price >2.5 ATR from VWAP, mean revert back
    # VWAP is purely intraday. Time filter: 10:30-14:00.
    "VWAP Reversion":       {"wr": 0.60, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"],
                             "cd": 120, "mh": 240},   # 2hr cd, 4hr hold

    # =========================================================================
    # DAILY ONLY (5) — 1d
    # =========================================================================
    # Swing/position strategies. Formation over weeks, hold for days/weeks.
    # Requires 250+ daily bars for proper lookback.

    "Momentum Breakout":       {"wr": 0.58, "type": "momentum",  "cat": PatternCategory.QUANT,
                                "tf": ["1d"],
                                "cd": 7200, "mh": 14400},  # 5 day cd, 10 day hold

    "Vol Compression Breakout": {"wr": 0.60, "type": "breakout", "cat": PatternCategory.QUANT,
                                "tf": ["1d"],
                                "cd": 7200, "mh": 14400},

    "Range Expansion":         {"wr": 0.56, "type": "breakout",  "cat": PatternCategory.QUANT,
                                "tf": ["1d"],
                                "cd": 4320, "mh": 7200},

    "Volume Breakout":         {"wr": 0.58, "type": "breakout",  "cat": PatternCategory.QUANT,
                                "tf": ["1d"],
                                "cd": 7200, "mh": 14400},

    "Donchian Breakout":       {"wr": 0.56, "type": "momentum",  "cat": PatternCategory.QUANT,
                                "tf": ["1d"],
                                "cd": 7200, "mh": 14400},
}


def get_all_pattern_names() -> list[str]:
    return sorted(PATTERN_META.keys())


def get_patterns_for_timeframe(timeframe: str) -> list[str]:
    """Return pattern names that are allowed to run on this timeframe."""
    return [name for name, meta in PATTERN_META.items() if timeframe in meta.get("tf", [])]