"""
structures/trendlines.py — Trendline detection using linear regression.

Quant firms fit regression lines through multiple swing points and measure
R-squared quality. This gives more robust lines than connecting two points.

Used by: Triangle detection, wedge detection, flag channels, breakout scoring.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.structures.swings import SwingPoint


@dataclass
class Trendline:
    """A fitted trendline through price points."""
    slope: float
    intercept: float
    r_squared: float        # Goodness of fit (0-1)
    num_points: int
    start_index: int
    end_index: int

    def price_at(self, bar_index: int) -> float:
        return self.slope * bar_index + self.intercept

    @property
    def is_rising(self) -> bool:
        return self.slope > 0

    @property
    def is_falling(self) -> bool:
        return self.slope < 0


@dataclass
class Channel:
    """A price channel defined by upper and lower trendlines."""
    upper: Trendline
    lower: Trendline

    @property
    def width_at_end(self) -> float:
        end = max(self.upper.end_index, self.lower.end_index)
        return self.upper.price_at(end) - self.lower.price_at(end)

    @property
    def width_at_start(self) -> float:
        start = max(self.upper.start_index, self.lower.start_index)
        return self.upper.price_at(start) - self.lower.price_at(start)

    @property
    def is_converging(self) -> bool:
        return self.width_at_end < self.width_at_start

    @property
    def midline_slope(self) -> float:
        return (self.upper.slope + self.lower.slope) / 2.0

    def bars_to_convergence(self) -> Optional[float]:
        """Estimate bars until the two lines meet. None if parallel/diverging."""
        slope_diff = self.upper.slope - self.lower.slope
        if slope_diff >= 0:
            return None
        x = (self.lower.intercept - self.upper.intercept) / (self.upper.slope - self.lower.slope)
        bars_from_now = x - self.upper.end_index
        return bars_from_now if bars_from_now > 0 else None


# ==============================================================================
# TRENDLINE FITTING
# ==============================================================================

def fit_trendline(points: list[SwingPoint]) -> Optional[Trendline]:
    """
    Fit a linear regression line through swing points.
    Returns None if fewer than 2 points.
    """
    if len(points) < 2:
        return None

    x = np.array([p.index for p in points], dtype=np.float64)
    y = np.array([p.price for p in points], dtype=np.float64)

    coeffs = np.polyfit(x, y, deg=1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return Trendline(
        slope=slope, intercept=intercept,
        r_squared=max(0.0, float(r_squared)),
        num_points=len(points),
        start_index=int(x[0]), end_index=int(x[-1]),
    )


def fit_trendline_from_arrays(
    indices: np.ndarray, prices: np.ndarray,
) -> Optional[Trendline]:
    """Fit trendline from raw arrays (no SwingPoint objects needed)."""
    if len(indices) < 2:
        return None

    x = np.asarray(indices, dtype=np.float64)
    y = np.asarray(prices, dtype=np.float64)

    coeffs = np.polyfit(x, y, deg=1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return Trendline(
        slope=slope, intercept=intercept,
        r_squared=max(0.0, float(r_squared)),
        num_points=len(x),
        start_index=int(x[0]), end_index=int(x[-1]),
    )


# ==============================================================================
# CHANNEL DETECTION
# ==============================================================================

def detect_channel(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    min_points: int = 2,
    min_r_squared: float = 0.6,
) -> Optional[Channel]:
    """Fit a price channel. Both lines need decent R-squared to qualify."""
    if len(swing_highs) < min_points or len(swing_lows) < min_points:
        return None

    upper = fit_trendline(swing_highs)
    lower = fit_trendline(swing_lows)

    if upper is None or lower is None:
        return None
    if upper.r_squared < min_r_squared or lower.r_squared < min_r_squared:
        return None

    end_idx = max(upper.end_index, lower.end_index)
    if upper.price_at(end_idx) <= lower.price_at(end_idx):
        return None

    return Channel(upper=upper, lower=lower)


# ==============================================================================
# HELPERS
# ==============================================================================

def is_converging(upper_slope: float, lower_slope: float) -> bool:
    """Two trendlines getting closer over time."""
    return upper_slope < lower_slope


def is_flat_line(trendline: Trendline, tolerance_pct: float = 0.1) -> bool:
    """Check if a trendline is approximately horizontal."""
    if trendline.num_points < 2:
        return False
    avg_price = abs(trendline.intercept + trendline.price_at(trendline.end_index)) / 2.0
    if avg_price == 0:
        return True
    slope_pct = abs(trendline.slope / avg_price) * 100.0
    return slope_pct < tolerance_pct


def slopes_same_sign(slope1: float, slope2: float) -> bool:
    """Both slopes going same direction. Used for wedge detection."""
    return (slope1 > 0 and slope2 > 0) or (slope1 < 0 and slope2 < 0)


def compression_ratio(channel: Channel) -> float:
    """End width / start width. < 1.0 = converging (compression)."""
    start_w = channel.width_at_start
    if start_w <= 0:
        return 1.0
    return channel.width_at_end / start_w