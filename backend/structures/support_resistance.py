"""
structures/support_resistance.py — Support/Resistance level detection.

Collects swing points, clusters nearby prices into zones, counts touches,
returns scored levels. This is how professional scanners detect S/R.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.structures.swings import SwingPoint, SwingType


@dataclass
class PriceLevel:
    """A support or resistance zone."""
    price: float
    touches: int
    level_type: str  # "support", "resistance", or "both"
    touch_indices: list[int] = field(default_factory=list)
    zone_width: float = 0.0

    @property
    def strength(self) -> float:
        """Level strength 0-100. More touches = stronger."""
        return min(100.0, self.touches * 15.0)

    @property
    def zone_high(self) -> float:
        return self.price + self.zone_width

    @property
    def zone_low(self) -> float:
        return self.price - self.zone_width


@dataclass
class BreakoutSignal:
    """Price has broken through a S/R level."""
    level: PriceLevel
    direction: str  # "above" or "below"
    break_bar_index: int
    break_price: float
    level_strength: float


# ==============================================================================
# LEVEL CLUSTERING
# ==============================================================================

def cluster_levels(
    swing_points: list[SwingPoint],
    tolerance_pct: float = 0.5,
    min_touches: int = 2,
) -> list[PriceLevel]:
    """
    Cluster swing points into price zones.

    1. Sort all swing prices
    2. Group prices within tolerance_pct of each other
    3. Each group becomes a price level with touch count
    """
    if len(swing_points) < 2:
        return []

    sorted_swings = sorted(swing_points, key=lambda s: s.price)
    levels: list[PriceLevel] = []
    used = set()

    for i, anchor in enumerate(sorted_swings):
        if i in used:
            continue

        cluster_prices = [anchor.price]
        cluster_indices = [anchor.index]
        cluster_types = [anchor.swing_type]
        used.add(i)

        tolerance = anchor.price * (tolerance_pct / 100.0)

        for j in range(i + 1, len(sorted_swings)):
            if j in used:
                continue
            if abs(sorted_swings[j].price - anchor.price) <= tolerance:
                cluster_prices.append(sorted_swings[j].price)
                cluster_indices.append(sorted_swings[j].index)
                cluster_types.append(sorted_swings[j].swing_type)
                used.add(j)
            elif sorted_swings[j].price > anchor.price + tolerance:
                break

        if len(cluster_prices) < min_touches:
            continue

        num_highs = sum(1 for t in cluster_types if t == SwingType.HIGH)
        num_lows = sum(1 for t in cluster_types if t == SwingType.LOW)

        if num_highs > 0 and num_lows > 0:
            level_type = "both"
        elif num_highs > num_lows:
            level_type = "resistance"
        else:
            level_type = "support"

        center = float(np.mean(cluster_prices))
        width = float(np.std(cluster_prices)) if len(cluster_prices) > 1 else tolerance

        levels.append(PriceLevel(
            price=round(center, 4), touches=len(cluster_prices),
            level_type=level_type, touch_indices=sorted(cluster_indices),
            zone_width=round(width, 4),
        ))

    levels.sort(key=lambda l: l.touches, reverse=True)
    return levels


def find_horizontal_levels(
    highs: np.ndarray, lows: np.ndarray,
    swing_high_indices: list[int], swing_low_indices: list[int],
    tolerance_pct: float = 0.5, min_touches: int = 2,
) -> tuple[list[PriceLevel], list[PriceLevel]]:
    """Convenience: arrays + indices -> (support_levels, resistance_levels)."""
    swing_points = []
    for idx in swing_high_indices:
        swing_points.append(SwingPoint(index=idx, price=float(highs[idx]), swing_type=SwingType.HIGH))
    for idx in swing_low_indices:
        swing_points.append(SwingPoint(index=idx, price=float(lows[idx]), swing_type=SwingType.LOW))

    all_levels = cluster_levels(swing_points, tolerance_pct, min_touches)
    support = [l for l in all_levels if l.level_type in ("support", "both")]
    resistance = [l for l in all_levels if l.level_type in ("resistance", "both")]
    return support, resistance


# ==============================================================================
# BREAKOUT DETECTION
# ==============================================================================

def detect_breakouts(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    levels: list[PriceLevel], lookback: int = 3,
) -> list[BreakoutSignal]:
    """Check if price has recently broken through any S/R level."""
    if len(closes) < 2 or not levels:
        return []

    signals = []
    n = len(closes)
    check_start = max(0, n - lookback)

    for level in levels:
        for i in range(check_start, n):
            if level.level_type in ("resistance", "both"):
                if closes[i] > level.zone_high:
                    was_below = any(closes[j] < level.price for j in range(max(0, i - 5), i))
                    if was_below:
                        signals.append(BreakoutSignal(
                            level=level, direction="above",
                            break_bar_index=i, break_price=float(closes[i]),
                            level_strength=level.strength,
                        ))
                        break

            if level.level_type in ("support", "both"):
                if closes[i] < level.zone_low:
                    was_above = any(closes[j] > level.price for j in range(max(0, i - 5), i))
                    if was_above:
                        signals.append(BreakoutSignal(
                            level=level, direction="below",
                            break_bar_index=i, break_price=float(closes[i]),
                            level_strength=level.strength,
                        ))
                        break

    return signals


# ==============================================================================
# UTILITIES
# ==============================================================================

def nearest_level(price: float, levels: list[PriceLevel], direction: str = "any") -> Optional[PriceLevel]:
    """Find nearest S/R level. direction: 'above', 'below', 'any'."""
    candidates = levels
    if direction == "above":
        candidates = [l for l in levels if l.price > price]
    elif direction == "below":
        candidates = [l for l in levels if l.price < price]
    if not candidates:
        return None
    return min(candidates, key=lambda l: abs(l.price - price))


def price_in_zone(price: float, level: PriceLevel) -> bool:
    """Check if a price is within a level's zone."""
    return level.zone_low <= price <= level.zone_high


def neckline_from_swings(swing_lows: list[SwingPoint], tolerance_pct: float = 1.5) -> Optional[float]:
    """Find horizontal neckline from swing lows (H&S patterns)."""
    if len(swing_lows) < 2:
        return None
    prices = [s.price for s in swing_lows]
    avg = float(np.mean(prices))
    max_dev = max(abs(p - avg) for p in prices)
    if avg > 0 and (max_dev / avg * 100) <= tolerance_pct:
        return avg
    return None