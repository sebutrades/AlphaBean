"""
base.py — The abstract base class all pattern detectors implement.

This is the foundation of the entire pattern detection engine.
Every pattern — whether it's a RubberBand Scalp or a Double Bottom —
produces a TradeSetup with exact entry, stop, target, and R:R.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Bias(str, Enum):
    LONG = "long"
    SHORT = "short"


class Timeframe(str, Enum):
    SCALP = "scalp"          # 1-5 min setups
    INTRADAY = "intraday"    # 5-60 min setups
    SWING = "swing"          # 1h-1d setups
    POSITION = "position"    # Multi-day


@dataclass
class TradeSetup:
    """The output of every pattern detector — a complete trade idea."""
    pattern_name: str
    symbol: str
    bias: Bias
    timeframe: Timeframe
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    confidence: float              # 0.0 to 1.0
    detected_at: datetime
    description: str               # Human-readable summary
    win_rate: float = 0.0          # Historical win rate of this pattern
    max_attempts: int = 1          # How many entry attempts allowed
    exit_strategy: str = ""        # Description of exit rules
    key_levels: dict = field(default_factory=dict)
    factors_bullish: list[str] = field(default_factory=list)
    factors_bearish: list[str] = field(default_factory=list)
    avoid_conditions: list[str] = field(default_factory=list)
    ideal_time_window: str = ""    # e.g. "10:00-10:45 AM EST"


class PatternDetector(ABC):
    """
    Abstract base class for all pattern detectors.
    
    To create a new pattern:
    1. Create a new file in patterns/smb/ or patterns/classical/
    2. Create a class that inherits from PatternDetector
    3. Implement name, bias, and detect()
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pattern name."""
        ...

    @property
    @abstractmethod
    def bias(self) -> Bias:
        """Default bias (LONG or SHORT)."""
        ...

    @abstractmethod
    def detect(self, bars) -> TradeSetup | None:
        """
        Analyze bars and return a TradeSetup if pattern is found,
        or None if no pattern detected.
        """
        ...