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
PATTERN_META = {
    # --- Classical Structural (16) ---
    "Head & Shoulders":     {"wr": 0.80, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Inverse H&S":          {"wr": 0.83, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Double Top":           {"wr": 0.73, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Double Bottom":        {"wr": 0.78, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Triple Top":           {"wr": 0.75, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Triple Bottom":        {"wr": 0.79, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Ascending Triangle":   {"wr": 0.75, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Descending Triangle":  {"wr": 0.72, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Symmetrical Triangle": {"wr": 0.65, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Bull Flag":            {"wr": 0.67, "type": "momentum", "cat": PatternCategory.CLASSICAL},
    "Bear Flag":            {"wr": 0.65, "type": "momentum", "cat": PatternCategory.CLASSICAL},
    "Pennant":              {"wr": 0.55, "type": "momentum", "cat": PatternCategory.CLASSICAL},
    "Cup & Handle":         {"wr": 0.68, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Rectangle":            {"wr": 0.70, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Rising Wedge":         {"wr": 0.69, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Falling Wedge":        {"wr": 0.68, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    # --- Candlestick (10) ---
    "Bullish Engulfing":    {"wr": 0.63, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK},
    "Bearish Engulfing":    {"wr": 0.63, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK},
    "Morning Star":         {"wr": 0.65, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK},
    "Evening Star":         {"wr": 0.65, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK},
    "Hammer":               {"wr": 0.60, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK},
    "Shooting Star":        {"wr": 0.59, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK},
    "Doji":                 {"wr": 0.55, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK},
    "Three White Soldiers": {"wr": 0.62, "type": "momentum",       "cat": PatternCategory.CANDLESTICK},
    "Three Black Crows":    {"wr": 0.62, "type": "momentum",       "cat": PatternCategory.CANDLESTICK},
    "Dragonfly Doji":       {"wr": 0.55, "type": "mean_reversion", "cat": PatternCategory.CANDLESTICK},
    # --- SMB Scalps (7) — removed: HitchHiker, Spencer, BackSide, Breaking News ---
    "RubberBand Scalp":     {"wr": 0.625, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "ORB 15min":            {"wr": 0.55, "type": "scalp",  "cat": PatternCategory.SMB_SCALP},
    "ORB 30min":            {"wr": 0.55, "type": "scalp",  "cat": PatternCategory.SMB_SCALP},
    "Second Chance Scalp":  {"wr": 0.525, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "Fashionably Late":     {"wr": 0.60, "type": "scalp",  "cat": PatternCategory.SMB_SCALP},
    "Gap Give & Go":        {"wr": 0.55, "type": "scalp",  "cat": PatternCategory.SMB_SCALP},
    "Tidal Wave":           {"wr": 0.55, "type": "scalp",  "cat": PatternCategory.SMB_SCALP},
    # --- Quant Strategies (9) — removed: Relative Strength Break ---
    "Mean Reversion":          {"wr": 0.62, "type": "mean_reversion", "cat": PatternCategory.QUANT},
    "Trend Pullback":          {"wr": 0.64, "type": "momentum",       "cat": PatternCategory.QUANT},
    "Gap Fade":                {"wr": 0.58, "type": "mean_reversion", "cat": PatternCategory.QUANT},
    "VWAP Reversion":          {"wr": 0.60, "type": "mean_reversion", "cat": PatternCategory.QUANT},
    # --- Daily-only strategies (run on 1d bars, skip on 5min/15min) ---
    "Momentum Breakout":       {"wr": 0.58, "type": "momentum",  "cat": PatternCategory.QUANT, "tf_req": "1d"},
    "Vol Compression Breakout": {"wr": 0.60, "type": "breakout", "cat": PatternCategory.QUANT, "tf_req": "1d"},
    "Range Expansion":         {"wr": 0.56, "type": "breakout",  "cat": PatternCategory.QUANT, "tf_req": "1d"},
    "Volume Breakout":         {"wr": 0.58, "type": "breakout",  "cat": PatternCategory.QUANT, "tf_req": "1d"},
    "Donchian Breakout":       {"wr": 0.56, "type": "momentum",  "cat": PatternCategory.QUANT, "tf_req": "1d"},
}


def get_all_pattern_names() -> list[str]:
    return sorted(PATTERN_META.keys())