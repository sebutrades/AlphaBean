"""
patterns/registry.py — Pattern definitions, TradeSetup, and metadata for all 42 detectors.

v2.0 — Audit changes:
  Removed: Breaking News, HitchHiker Scalp, Spencer Scalp, BackSide Scalp, Relative Strength Break
  Added: timeframe_required field for daily-only strategies
  Count: 42 (was 47)
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


# Win rates from Bulkowski, Nison, and quant research
# tf_req: required timeframe(s). None = runs on 5min/15min. "1d" = daily only.
# cd: cooldown in minutes — minimum time between same pattern re-firing
# mh: max hold in minutes — timeout if not resolved
PATTERN_META = {
    # --- Classical Structural (16) ---
    # These form over hours/days. Long cooldown (don't re-detect same formation).
    # Measured moves take time — long max hold.
    "Head & Shoulders":     {"wr": 0.80, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 480, "mh": 1200},  # 8hr cd, 2.5 day hold
    "Inverse H&S":          {"wr": 0.83, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 480, "mh": 1200},
    "Double Top":           {"wr": 0.73, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 480, "mh": 960},   # 8hr cd, 2 day hold
    "Double Bottom":        {"wr": 0.78, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 480, "mh": 960},
    "Triple Top":           {"wr": 0.75, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 480, "mh": 960},
    "Triple Bottom":        {"wr": 0.79, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 480, "mh": 960},
    "Ascending Triangle":   {"wr": 0.75, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 360, "mh": 960},   # 6hr cd, 2 day hold
    "Descending Triangle":  {"wr": 0.72, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 360, "mh": 960},
    "Symmetrical Triangle": {"wr": 0.65, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 360, "mh": 960},
    "Bull Flag":            {"wr": 0.67, "type": "momentum", "cat": PatternCategory.CLASSICAL,
                             "cd": 240, "mh": 480},   # 4hr cd, 1 day hold
    "Bear Flag":            {"wr": 0.65, "type": "momentum", "cat": PatternCategory.CLASSICAL,
                             "cd": 240, "mh": 480},
    "Pennant":              {"wr": 0.55, "type": "momentum", "cat": PatternCategory.CLASSICAL,
                             "cd": 240, "mh": 480},
    "Cup & Handle":         {"wr": 0.68, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 480, "mh": 1200},  # Slow pattern, long hold
    "Rectangle":            {"wr": 0.70, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 360, "mh": 960},
    "Rising Wedge":         {"wr": 0.69, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 360, "mh": 960},
    "Falling Wedge":        {"wr": 0.68, "type": "breakout", "cat": PatternCategory.CLASSICAL,
                             "cd": 360, "mh": 960},
    # --- Candlestick (10) ---
    # Quick signals. Can fire at different S/R levels within same day.
    # If the reversal doesn't work in 3 hours, it's dead.
    "Bullish Engulfing":    {"wr": 0.63, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "cd": 60, "mh": 180},    # 1hr cd, 3hr hold
    "Bearish Engulfing":    {"wr": 0.63, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "cd": 60, "mh": 180},
    "Morning Star":         {"wr": 0.65, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "cd": 60, "mh": 180},
    "Evening Star":         {"wr": 0.65, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "cd": 60, "mh": 180},
    "Hammer":               {"wr": 0.60, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "cd": 60, "mh": 180},
    "Shooting Star":        {"wr": 0.59, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "cd": 60, "mh": 180},
    "Doji":                 {"wr": 0.55, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "cd": 60, "mh": 180},
    "Three White Soldiers": {"wr": 0.62, "type": "momentum",       "cat": PatternCategory.CANDLESTICK,
                             "cd": 120, "mh": 240},   # Momentum — slightly longer
    "Three Black Crows":    {"wr": 0.62, "type": "momentum",       "cat": PatternCategory.CANDLESTICK,
                             "cd": 120, "mh": 240},
    "Dragonfly Doji":       {"wr": 0.55, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK,
                             "cd": 60, "mh": 180},
    # --- SMB Scalps (7) ---
    # Intraday-specific. Most are once-per-day setups with short hold times.
    "RubberBand Scalp":     {"wr": 0.625, "type": "scalp", "cat": PatternCategory.SMB_SCALP,
                             "cd": 480, "mh": 120},   # Once/day, 2hr hold max
    "ORB 15min":            {"wr": 0.55, "type": "scalp",  "cat": PatternCategory.SMB_SCALP,
                             "cd": 960, "mh": 120},   # Once/day (16hr cd), 2hr hold
    "ORB 30min":            {"wr": 0.55, "type": "scalp",  "cat": PatternCategory.SMB_SCALP,
                             "cd": 960, "mh": 120},
    "Second Chance Scalp":  {"wr": 0.525, "type": "scalp", "cat": PatternCategory.SMB_SCALP,
                             "cd": 240, "mh": 180},   # 4hr cd, 3hr hold
    "Fashionably Late":     {"wr": 0.60, "type": "scalp",  "cat": PatternCategory.SMB_SCALP,
                             "cd": 480, "mh": 180},   # Once/day, 3hr hold
    "Gap Give & Go":        {"wr": 0.55, "type": "scalp",  "cat": PatternCategory.SMB_SCALP,
                             "cd": 480, "mh": 90},    # Once/day, 1.5hr hold
    "Tidal Wave":           {"wr": 0.55, "type": "scalp",  "cat": PatternCategory.SMB_SCALP,
                             "cd": 360, "mh": 480},   # 6hr cd, 1 day hold (breakdown)
    # --- Quant Intraday (4) ---
    "Mean Reversion":          {"wr": 0.62, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                                "cd": 120, "mh": 240},  # 2hr cd, 4hr hold
    "Trend Pullback":          {"wr": 0.64, "type": "momentum",       "cat": PatternCategory.QUANT,
                                "cd": 120, "mh": 240},
    "Gap Fade":                {"wr": 0.58, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                                "cd": 480, "mh": 90},   # Once/day, 1.5hr hold
    "VWAP Reversion":          {"wr": 0.60, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                                "cd": 120, "mh": 240},
    # --- Daily-only strategies (5) ---
    # On daily bars: 1 bar = 1 day. Hold for days/weeks.
    "Momentum Breakout":       {"wr": 0.58, "type": "momentum",  "cat": PatternCategory.QUANT,
                                "tf_req": "1d", "cd": 7200, "mh": 14400},  # 5 day cd, 10 day hold
    "Vol Compression Breakout": {"wr": 0.60, "type": "breakout", "cat": PatternCategory.QUANT,
                                "tf_req": "1d", "cd": 7200, "mh": 14400},
    "Range Expansion":         {"wr": 0.56, "type": "breakout",  "cat": PatternCategory.QUANT,
                                "tf_req": "1d", "cd": 4320, "mh": 7200},   # 3 day cd, 5 day hold
    "Volume Breakout":         {"wr": 0.58, "type": "breakout",  "cat": PatternCategory.QUANT,
                                "tf_req": "1d", "cd": 7200, "mh": 14400},
    "Donchian Breakout":       {"wr": 0.56, "type": "momentum",  "cat": PatternCategory.QUANT,
                                "tf_req": "1d", "cd": 7200, "mh": 14400},
}


def get_all_pattern_names() -> list[str]:
    return sorted(PATTERN_META.keys())