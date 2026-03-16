"""
structures/swings.py — Swing point detection (the foundation of everything).

Two algorithms:
  1. zigzag() — Percentage-based, produces alternating high/low pivots.
     Used by: quant fund pattern engines, TrendSpider, Autochartist.
  2. find_swing_highs/lows() — Order-based local extrema detection.
     Used by: SMB scalp patterns and local micro-structure detection.

Both are needed because zigzag is better for large structural patterns
(H&S, double tops) while order-based is better for consolidation zones
and flag channels.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np


# ==============================================================================
# DATA TYPES
# ==============================================================================

class SwingType(str, Enum):
    HIGH = "high"
    LOW = "low"


@dataclass
class SwingPoint:
    """A confirmed swing high or low in the price series."""
    index: int
    price: float
    swing_type: SwingType
    timestamp: Optional[datetime] = None
    volume: int = 0

    @property
    def is_high(self) -> bool:
        return self.swing_type == SwingType.HIGH

    @property
    def is_low(self) -> bool:
        return self.swing_type == SwingType.LOW


# ==============================================================================
# ZIGZAG ALGORITHM
# ==============================================================================

def zigzag(
    highs: np.ndarray,
    lows: np.ndarray,
    threshold_pct: float = 3.0,
    timestamps: Optional[list[datetime]] = None,
    volumes: Optional[np.ndarray] = None,
) -> list[SwingPoint]:
    """
    Zigzag swing detector — the standard quant approach.

    Identifies swing points where price has reversed by at least
    `threshold_pct` percent from the last confirmed swing.
    Produces alternating HIGH -> LOW -> HIGH -> LOW pivots.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        threshold_pct: Minimum % reversal to confirm a new swing
        timestamps: Optional bar timestamps
        volumes: Optional bar volumes

    Returns:
        List of SwingPoint in chronological order, alternating H/L.
    """
    if len(highs) < 3:
        return []

    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    threshold = threshold_pct / 100.0

    swings: list[SwingPoint] = []

    # Initialize tracking
    last_high = highs[0]
    last_high_idx = 0
    last_low = lows[0]
    last_low_idx = 0

    # Determine initial direction from first few bars
    initial_high_idx = 0
    initial_low_idx = 0
    for i in range(1, min(len(highs), 20)):
        if highs[i] > highs[initial_high_idx]:
            initial_high_idx = i
        if lows[i] < lows[initial_low_idx]:
            initial_low_idx = i

    if initial_high_idx <= initial_low_idx:
        last_type = SwingType.LOW  # Start looking for a high
    else:
        last_type = SwingType.HIGH  # Start looking for a low

    def _make_swing(idx: int, price: float, stype: SwingType) -> SwingPoint:
        return SwingPoint(
            index=idx, price=price, swing_type=stype,
            timestamp=timestamps[idx] if timestamps else None,
            volume=int(volumes[idx]) if volumes is not None else 0,
        )

    for i in range(1, len(highs)):
        if last_type == SwingType.LOW:
            # Tracking upward — looking for swing high
            if highs[i] > last_high:
                last_high = highs[i]
                last_high_idx = i
            if lows[i] <= last_high * (1 - threshold):
                swings.append(_make_swing(last_high_idx, last_high, SwingType.HIGH))
                last_type = SwingType.HIGH
                last_low = lows[i]
                last_low_idx = i
        else:
            # Tracking downward — looking for swing low
            if lows[i] < last_low:
                last_low = lows[i]
                last_low_idx = i
            if highs[i] >= last_low * (1 + threshold):
                swings.append(_make_swing(last_low_idx, last_low, SwingType.LOW))
                last_type = SwingType.LOW
                last_high = highs[i]
                last_high_idx = i

    return swings


def adaptive_zigzag_threshold(timeframe: str) -> float:
    """Threshold adapts to timeframe — smaller TF needs smaller reversal."""
    return {"5min": 0.5, "15min": 1.0, "1h": 2.0, "1d": 3.0}.get(timeframe, 2.0)


# ==============================================================================
# ORDER-BASED SWING DETECTION (vectorized)
# ==============================================================================

def find_swing_highs(highs: np.ndarray, order: int = 3) -> list[int]:
    """
    Find indices where high[i] > all neighbors within `order` bars.
    Uses vectorized NumPy comparisons.
    """
    highs = np.asarray(highs, dtype=np.float64)
    n = len(highs)
    if n < 2 * order + 1:
        return []

    mask = np.ones(n, dtype=bool)
    mask[:order] = False
    mask[-order:] = False

    for offset in range(1, order + 1):
        mask[order:-order] &= highs[order:-order] > highs[order - offset:-order - offset]
        mask[order:-order] &= highs[order:-order] > highs[order + offset:n - order + offset]

    return list(np.where(mask)[0])


def find_swing_lows(lows: np.ndarray, order: int = 3) -> list[int]:
    """
    Find indices where low[i] < all neighbors within `order` bars.
    Mirror of find_swing_highs.
    """
    lows = np.asarray(lows, dtype=np.float64)
    n = len(lows)
    if n < 2 * order + 1:
        return []

    mask = np.ones(n, dtype=bool)
    mask[:order] = False
    mask[-order:] = False

    for offset in range(1, order + 1):
        mask[order:-order] &= lows[order:-order] < lows[order - offset:-order - offset]
        mask[order:-order] &= lows[order:-order] < lows[order + offset:n - order + offset]

    return list(np.where(mask)[0])


def adaptive_order(timeframe: str) -> int:
    """Timeframe-adaptive order for order-based swing detection."""
    return {"5min": 3, "15min": 4, "1h": 3, "1d": 5}.get(timeframe, 3)


# ==============================================================================
# UTILITIES
# ==============================================================================

def get_swing_sequence(swings: list[SwingPoint], last_n: int = 5) -> list[SwingPoint]:
    """Get the last N swing points."""
    return swings[-last_n:] if len(swings) >= last_n else swings


def swing_highs_from_zigzag(swings: list[SwingPoint]) -> list[SwingPoint]:
    """Extract only swing highs from a zigzag result."""
    return [s for s in swings if s.is_high]


def swing_lows_from_zigzag(swings: list[SwingPoint]) -> list[SwingPoint]:
    """Extract only swing lows from a zigzag result."""
    return [s for s in swings if s.is_low]


def swings_between(swings: list[SwingPoint], start_idx: int, end_idx: int) -> list[SwingPoint]:
    """Get swing points within a bar index range."""
    return [s for s in swings if start_idx <= s.index <= end_idx]


def highest_swing_high(swings: list[SwingPoint]) -> Optional[SwingPoint]:
    """Return the swing high with the highest price."""
    highs = [s for s in swings if s.is_high]
    return max(highs, key=lambda s: s.price) if highs else None


def lowest_swing_low(swings: list[SwingPoint]) -> Optional[SwingPoint]:
    """Return the swing low with the lowest price."""
    lows_list = [s for s in swings if s.is_low]
    return min(lows_list, key=lambda s: s.price) if lows_list else None


def swing_range(swings: list[SwingPoint]) -> float:
    """Price range covered by a set of swings."""
    if not swings:
        return 0.0
    prices = [s.price for s in swings]
    return max(prices) - min(prices)