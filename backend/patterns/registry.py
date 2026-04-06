"""
patterns/registry.py — Pattern definitions, TradeSetup, and metadata for all quant active detectors.

SECTION 1 CHANGES:
  - TradeSetup gains: target_1, target_2, trail_type, trail_param, position_splits
  - to_dict() passes through new fields
  - All else unchanged (PATTERN_META, etc.)

v2.2 — Scaled exits + ATR-based infrastructure:
  - target_1: First take-profit level (partial exit)
  - target_2: Second take-profit level (partial exit)  
  - trail_type: How to trail the remaining position ("swing", "ema9", "atr", "vwap")
  - trail_param: Parameter for trailing (ATR multiplier, EMA period, etc.)
  - position_splits: Tuple of (pct_at_t1, pct_at_t2, pct_trail) e.g. (0.5, 0.3, 0.2)
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
    target_price: float             # Primary target (full measured move) — kept for backward compat
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

    # ── Scaled Exit Fields (NEW in v2.2) ──
    target_1: float = 0.0           # First take-profit (e.g., 1R or nearest S/R)
    target_2: float = 0.0           # Second take-profit (e.g., measured move)
    trail_type: str = "atr"         # "atr", "swing", "ema9", "vwap", "none"
    trail_param: float = 2.0        # ATR multiplier or EMA period
    position_splits: tuple = (0.5, 0.3, 0.2)  # % at T1, T2, trail

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
            # Scaled exits
            "target_1": self.target_1,
            "target_2": self.target_2,
            "trail_type": self.trail_type,
            "trail_param": self.trail_param,
            "position_splits": list(self.position_splits),
        }


# ==============================================================================
# PATTERN_META — Complete per-pattern configuration
# ==============================================================================
# Unchanged from v2.1 — all tf, cd, mh, wr, type, cat fields remain the same.

PATTERN_META = {
    # =========================================================================
    # SMB SCALPS (1) — kept for its backtested edge
    # =========================================================================
    "Tidal Wave":           {"wr": 0.55, "type": "scalp", "cat": PatternCategory.SMB_SCALP,
                             "tf": ["5min", "15min"], "cd": 360, "mh": 480},

    # =========================================================================
    # QUANT — INTRADAY 5min (19)
    # =========================================================================
    "Mean Reversion":       {"wr": 0.62, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 120, "mh": 240},
    "Trend Pullback":       {"wr": 0.64, "type": "momentum", "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 120, "mh": 240},
    "Gap Fade":             {"wr": 0.58, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 480, "mh": 90},
    "VWAP Reversion":       {"wr": 0.60, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 120, "mh": 240},
    "Gap Reversal Long":    {"wr": 0.52, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 480, "mh": 240},
    "Gap Reversal Short":   {"wr": 0.48, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 480, "mh": 240},
    "Opening Drive Long":   {"wr": 0.52, "type": "momentum",       "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 480, "mh": 240},
    "Opening Drive Short":  {"wr": 0.50, "type": "momentum",       "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 480, "mh": 240},
    "Volume Climax Long":   {"wr": 0.52, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 240, "mh": 240},
    "Volume Climax Short":  {"wr": 0.50, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 240, "mh": 240},
    "VWAP Trend Long":      {"wr": 0.55, "type": "momentum",       "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 480, "mh": 240},
    "VWAP Trend Short":     {"wr": 0.50, "type": "momentum",       "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 480, "mh": 240},
    "RSI Divergence Long":  {"wr": 0.52, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 240, "mh": 240},
    "RSI Divergence Short": {"wr": 0.50, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 240, "mh": 240},
    "Midday Reversal Long": {"wr": 0.52, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 480, "mh": 180},
    "Midday Reversal Short":{"wr": 0.50, "type": "mean_reversion", "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 480, "mh": 180},
    "Power Hour Long":      {"wr": 0.52, "type": "momentum",       "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 480, "mh": 240},
    "Power Hour Short":     {"wr": 0.50, "type": "momentum",       "cat": PatternCategory.QUANT,
                             "tf": ["5min"], "cd": 480, "mh": 240},

    # =========================================================================
    # QUANT — INTRADAY 15min/1h (6)
    # =========================================================================
    "Keltner Breakout Long":  {"wr": 0.52, "type": "breakout",      "cat": PatternCategory.QUANT,
                               "tf": ["15min", "1h"], "cd": 240, "mh": 480},
    "Keltner Breakout Short": {"wr": 0.48, "type": "breakout",      "cat": PatternCategory.QUANT,
                               "tf": ["15min", "1h"], "cd": 240, "mh": 480},
    "MACD Turn Long":         {"wr": 0.52, "type": "momentum",      "cat": PatternCategory.QUANT,
                               "tf": ["15min", "1h"], "cd": 240, "mh": 480},
    "MACD Turn Short":        {"wr": 0.48, "type": "momentum",      "cat": PatternCategory.QUANT,
                               "tf": ["15min", "1h"], "cd": 240, "mh": 480},
    "VP Divergence Long":     {"wr": 0.52, "type": "mean_reversion","cat": PatternCategory.QUANT,
                               "tf": ["15min", "1h"], "cd": 240, "mh": 480},
    "VP Divergence Short":    {"wr": 0.48, "type": "mean_reversion","cat": PatternCategory.QUANT,
                               "tf": ["15min", "1h"], "cd": 240, "mh": 480},

    # =========================================================================
    # QUANT — DAILY (27)
    # =========================================================================
    "Momentum Breakout":        {"wr": 0.58, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 14400},
    "Vol Compression Breakout": {"wr": 0.60, "type": "breakout",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 14400},
    "Range Expansion":          {"wr": 0.56, "type": "breakout",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 4320,  "mh": 7200},
    "Volume Breakout":          {"wr": 0.58, "type": "breakout",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 14400},
    "Donchian Breakout":        {"wr": 0.56, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 14400},
    "Juicer Long":              {"wr": 0.55, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 14400, "mh": 99000},
    "TS Momentum Long":         {"wr": 0.52, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 14400, "mh": 99000},
    "TS Momentum Short":        {"wr": 0.48, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 14400, "mh": 99000},
    "Multi-TF Trend Long":      {"wr": 0.52, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 14400, "mh": 99000},
    "Multi-TF Trend Short":     {"wr": 0.48, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 14400, "mh": 99000},
    "ST Reversal Long":         {"wr": 0.55, "type": "mean_reversion", "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 14400},
    "ST Reversal Short":        {"wr": 0.50, "type": "mean_reversion", "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 14400},
    "Low Vol Long":             {"wr": 0.55, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 14400, "mh": 99000},
    "Turtle Breakout Long":     {"wr": 0.52, "type": "breakout",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 14400, "mh": 99000},
    "Turtle Breakout Short":    {"wr": 0.48, "type": "breakout",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 14400, "mh": 99000},
    "BAB Long":                 {"wr": 0.55, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 14400, "mh": 99000},
    "52W High Momentum":        {"wr": 0.55, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 14400, "mh": 99000},
    "RS Persistence Long":      {"wr": 0.55, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 99000},
    "Streak Reversal Long":     {"wr": 0.52, "type": "mean_reversion", "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 14400},
    "Streak Reversal Short":    {"wr": 0.50, "type": "mean_reversion", "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 14400},
    "ATR Expansion Long":       {"wr": 0.52, "type": "breakout",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 1440,  "mh": 7200},
    "ATR Expansion Short":      {"wr": 0.48, "type": "breakout",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 1440,  "mh": 7200},
    "BB Squeeze Long":          {"wr": 0.52, "type": "breakout",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 99000},
    "BB Squeeze Short":         {"wr": 0.48, "type": "breakout",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 99000},
    "Accumulation Long":        {"wr": 0.55, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 99000},
    "Distribution Short":       {"wr": 0.50, "type": "momentum",       "cat": PatternCategory.QUANT, "tf": ["1d"], "cd": 7200,  "mh": 99000},
}


def get_all_pattern_names() -> list[str]:
    return sorted(PATTERN_META.keys())


def get_patterns_for_timeframe(timeframe: str) -> list[str]:
    """Return pattern names that are allowed to run on this timeframe."""
    return [name for name, meta in PATTERN_META.items() if timeframe in meta.get("tf", [])]