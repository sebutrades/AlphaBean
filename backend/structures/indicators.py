"""
structures/indicators.py — Core technical indicators (single source of truth).

Every module that needs ATR, True Range, SMA, or EMA imports from here.
This ensures ONE implementation of each formula across the entire system.

All formulas match industry conventions:
  - True Range: Wilder (1978), max(H-L, |H-Pc|, |L-Pc|)
  - ATR: Wilder's smoothing, ATR_t = ((ATR_{t-1} * (n-1)) + TR_t) / n
  - SMA: Simple arithmetic mean over window
  - EMA: Seed = SMA, then EMA_t = price * k + EMA_{t-1} * (1-k), k = 2/(n+1)

References:
  - Wilder, J.W. (1978). New Concepts in Technical Trading Systems.
  - This is what TradingView, Bloomberg, and every professional platform uses.
"""
import numpy as np
from typing import Optional


# ==============================================================================
# TRUE RANGE (vectorized)
# ==============================================================================

def true_range(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """
    Compute True Range for each bar (vectorized).

    TR_t = max(
        High_t - Low_t,
        |High_t - Close_{t-1}|,
        |Low_t - Close_{t-1}|
    )

    First bar uses H-L only (no previous close available).

    Returns:
        Array of True Range values, same length as input.
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)

    n = len(highs)
    if n == 0:
        return np.array([], dtype=np.float64)

    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]  # First bar: H-L only

    if n > 1:
        hl = highs[1:] - lows[1:]
        hc = np.abs(highs[1:] - closes[:-1])
        lc = np.abs(lows[1:] - closes[:-1])
        tr[1:] = np.maximum(hl, np.maximum(hc, lc))

    return tr


# ==============================================================================
# WILDER'S ATR
# ==============================================================================

def wilder_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Average True Range using Wilder's smoothing (the industry standard).

    Algorithm:
      1. Compute True Range for all bars.
      2. First ATR value (at index period-1) = SMA of first `period` TRs.
      3. Subsequent: ATR_t = ((ATR_{t-1} * (period-1)) + TR_t) / period

    This is NOT a simple moving average of TR. Wilder's method is a specific
    type of exponential smoothing where alpha = 1/period. It gives more weight
    to recent values and is smoother than SMA.

    Returns:
        Array of ATR values. First (period-1) values are NaN.
        Length matches input arrays.
    """
    tr = true_range(highs, lows, closes)
    n = len(tr)

    atr = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return atr

    # Seed: SMA of first `period` true ranges
    atr[period - 1] = np.mean(tr[:period])

    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ==============================================================================
# ATR RATIO (for regime detection)
# ==============================================================================

def atr_ratio(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr_period: int = 14,
    baseline_lookback: int = 60,
) -> float:
    """
    Ratio of current Wilder's ATR to the historical baseline.

    current = ATR(14) at the last bar
    baseline = median of ATR(14) values from [-(baseline_lookback) : -(atr_period)]

    The baseline EXCLUDES the most recent `atr_period` bars to avoid
    contamination from a recent spike.

    Using median (not mean) makes it robust to outliers in the baseline.

    Returns:
        Ratio > 1.0 = volatility expanding
        Ratio < 1.0 = volatility compressing
        Ratio > 1.8 = high-volatility regime trigger
    """
    atr_series = wilder_atr(highs, lows, closes, atr_period)
    valid = atr_series[~np.isnan(atr_series)]

    if len(valid) < atr_period + 5:
        return 1.0

    current_atr = float(valid[-1])

    # Baseline: exclude the last atr_period values
    baseline_end = len(valid) - atr_period
    baseline_start = max(0, baseline_end - baseline_lookback)

    if baseline_end <= baseline_start:
        return 1.0

    baseline_values = valid[baseline_start:baseline_end]
    baseline_atr = float(np.median(baseline_values))

    if baseline_atr <= 0:
        return 1.0

    return current_atr / baseline_atr


# ==============================================================================
# SIMPLE MOVING AVERAGE (vectorized)
# ==============================================================================

def sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Simple Moving Average.

    Returns array same length as input. First (period-1) values are NaN.
    Uses numpy convolution for vectorized computation.
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)

    if n < period:
        return np.full(n, np.nan, dtype=np.float64)

    result = np.full(n, np.nan, dtype=np.float64)
    # Cumulative sum trick for O(n) SMA
    cumsum = np.cumsum(prices)
    cumsum = np.insert(cumsum, 0, 0)
    result[period - 1:] = (cumsum[period:] - cumsum[:-period]) / period

    return result


# ==============================================================================
# EXPONENTIAL MOVING AVERAGE
# ==============================================================================

def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average.

    Seed = SMA of first `period` values.
    EMA_t = price_t * k + EMA_{t-1} * (1 - k), where k = 2 / (period + 1).

    Returns array same length as input. First (period-1) values are NaN.
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)

    if n < period:
        return np.full(n, np.nan, dtype=np.float64)

    k = 2.0 / (period + 1)
    result = np.full(n, np.nan, dtype=np.float64)

    # Seed with SMA
    result[period - 1] = np.mean(prices[:period])

    # EMA forward
    for i in range(period, n):
        result[i] = prices[i] * k + result[i - 1] * (1 - k)

    return result


def ema_last(prices: np.ndarray, period: int) -> float:
    """Compute only the LAST EMA value (efficient for single-point scoring)."""
    if len(prices) < period:
        return float(prices[-1])
    k = 2.0 / (period + 1)
    val = float(np.mean(prices[:period]))
    for i in range(period, len(prices)):
        val = prices[i] * k + val * (1 - k)
    return val