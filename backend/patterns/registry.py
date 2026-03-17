"""
patterns/registry.py — Pattern definitions and TradeSetup output format.

Every pattern detector (classical, SMB, quant) produces a TradeSetup.
This is the universal output format that the scanner, scorer, and UI consume.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Bias(str, Enum):
    LONG = "long"
    SHORT = "short"


class PatternCategory(str, Enum):
    CLASSICAL = "classical"     # H&S, triangles, flags, etc.
    SMB_SCALP = "smb_scalp"    # RubberBand, HitchHiker, etc.
    QUANT = "quant"            # Momentum breakout, vol compression


@dataclass
class TradeSetup:
    """The output of every pattern detector — a complete trade idea."""
    pattern_name: str
    category: PatternCategory
    symbol: str
    bias: Bias
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    confidence: float               # 0.0-1.0
    detected_at: datetime
    description: str
    strategy_type: str = "breakout" # momentum, mean_reversion, breakout, scalp
    win_rate: float = 0.0
    max_attempts: int = 1
    exit_strategy: str = ""
    key_levels: dict = field(default_factory=dict)
    ideal_time: str = ""
    timeframe_detected: str = ""
    multi_tf: bool = False

    def to_dict(self) -> dict:
        return {
            "pattern_name": self.pattern_name,
            "category": self.category.value,
            "symbol": self.symbol,
            "bias": self.bias.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "risk_reward_ratio": self.risk_reward_ratio,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
            "description": self.description,
            "strategy_type": self.strategy_type,
            "win_rate": self.win_rate,
            "max_attempts": self.max_attempts,
            "exit_strategy": self.exit_strategy,
            "key_levels": self.key_levels,
            "ideal_time": self.ideal_time,
            "timeframe_detected": self.timeframe_detected,
            "multi_tf": self.multi_tf,
            "backtest_score": 50.0,  # Enriched later
        }


# Pattern metadata — reported win rates from Bulkowski and quant research
PATTERN_META = {
    "Head & Shoulders": {"wr": 0.75, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Inverse H&S": {"wr": 0.75, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Double Top": {"wr": 0.70, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Double Bottom": {"wr": 0.70, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Ascending Triangle": {"wr": 0.72, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Descending Triangle": {"wr": 0.72, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Symmetrical Triangle": {"wr": 0.65, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Bull Flag": {"wr": 0.65, "type": "momentum", "cat": PatternCategory.CLASSICAL},
    "Bear Flag": {"wr": 0.62, "type": "momentum", "cat": PatternCategory.CLASSICAL},
    "Cup & Handle": {"wr": 0.65, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Rising Wedge": {"wr": 0.68, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "Falling Wedge": {"wr": 0.68, "type": "breakout", "cat": PatternCategory.CLASSICAL},
    "RubberBand Scalp": {"wr": 0.625, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "HitchHiker Scalp": {"wr": 0.575, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "ORB 15min": {"wr": 0.55, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "ORB 30min": {"wr": 0.55, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "Second Chance Scalp": {"wr": 0.525, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "BackSide Scalp": {"wr": 0.55, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "Fashionably Late": {"wr": 0.60, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "Spencer Scalp": {"wr": 0.55, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "Gap Give & Go": {"wr": 0.55, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "Tidal Wave": {"wr": 0.55, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "Breaking News": {"wr": 0.50, "type": "scalp", "cat": PatternCategory.SMB_SCALP},
    "Momentum Breakout": {"wr": 0.58, "type": "momentum", "cat": PatternCategory.QUANT},
    "Vol Compression Breakout": {"wr": 0.60, "type": "breakout", "cat": PatternCategory.QUANT},
}


def get_all_pattern_names() -> list[str]:
    """Sorted list of all pattern names (for frontend dropdown)."""
    return sorted(PATTERN_META.keys())