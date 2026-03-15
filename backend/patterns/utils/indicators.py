"""
indicators.py — Core technical indicators used across all pattern detectors.

EMA = Exponential Moving Average (reacts faster to recent prices)
VWAP = Volume Weighted Average Price (institutional benchmark)
ATR = Average True Range (measures volatility in dollar terms)
RVOL = Relative Volume (how active is this stock vs. normal)
"""
import numpy as np


def ema(prices: list[float], period: int = 9) -> list[float]:
    """
    Calculate Exponential Moving Average.
    
    EMA gives more weight to recent prices. The 9 EMA is used in many
    of the SMB scalps as the trailing stop reference.
    
    Formula: EMA_today = price_today * k + EMA_yesterday * (1 - k)
    where k = 2 / (period + 1)
    """
    if len(prices) < period:
        return [None] * len(prices)
    
    k = 2 / (period + 1)  # Smoothing factor
    ema_values = [None] * (period - 1)
    
    # First EMA value = simple average of first `period` prices
    ema_values.append(sum(prices[:period]) / period)
    
    # Calculate remaining EMA values
    for i in range(period, len(prices)):
        new_ema = prices[i] * k + ema_values[-1] * (1 - k)
        ema_values.append(new_ema)
    
    return ema_values


def vwap(bars) -> list[float]:
    """
    Calculate cumulative intraday VWAP.
    
    VWAP = Cumulative(Typical Price × Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3
    
    VWAP resets at market open each day. It's the benchmark
    institutional traders use — if price is above VWAP, buyers
    are in control; below, sellers are.
    """
    vwap_values = []
    cum_vol = 0
    cum_tp_vol = 0
    
    for bar in bars:
        typical_price = (bar.high + bar.low + bar.close) / 3
        cum_vol += bar.volume
        cum_tp_vol += typical_price * bar.volume
        
        if cum_vol > 0:
            vwap_values.append(cum_tp_vol / cum_vol)
        else:
            vwap_values.append(typical_price)
    
    return vwap_values


def atr(highs, lows, closes, period: int = 14) -> list[float]:
    """
    Average True Range — measures volatility in dollar terms.
    
    True Range = max of:
      1) Current High - Current Low
      2) abs(Current High - Previous Close)
      3) abs(Current Low - Previous Close)
    
    ATR = moving average of True Range over `period` bars.
    
    Used in RubberBand Scalp: "Price down > 3 ATRs from open"
    """
    if len(highs) < 2:
        return [0.0]
    
    true_ranges = [highs[0] - lows[0]]  # First bar: just H-L
    
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)
    
    # Simple moving average of true ranges
    atr_values = [None] * (period - 1)
    for i in range(period - 1, len(true_ranges)):
        atr_values.append(np.mean(true_ranges[i - period + 1:i + 1]))
    
    return atr_values


def relative_volume(current_volume: int, avg_volume: int) -> float:
    """
    RVOL = Current Volume / Average Volume.
    
    RVOL > 2 = stock is active ("in play")
    RVOL > 5 = very in play (RubberBand Scalp threshold)
    """
    if avg_volume == 0:
        return 0.0
    return current_volume / avg_volume


def find_swing_highs(highs: list[float], order: int = 5) -> list[int]:
    """
    Find swing high indices.
    A swing high at index i means highs[i] is higher than
    the `order` bars on either side.
    """
    results = []
    for i in range(order, len(highs) - order):
        is_swing = True
        for j in range(1, order + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing = False
                break
        if is_swing:
            results.append(i)
    return results


def find_swing_lows(lows: list[float], order: int = 5) -> list[int]:
    """Find swing low indices — mirrors find_swing_highs."""
    results = []
    for i in range(order, len(lows) - order):
        is_swing = True
        for j in range(1, order + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing = False
                break
        if is_swing:
            results.append(i)
    return results


def double_bar_break_up(bars, index: int) -> bool:
    """
    Check if bar at `index` is a green candle that clears
    the highs of 2+ preceding candles (used in RubberBand entry).
    """
    if index < 2:
        return False
    bar = bars[index]
    if bar.close <= bar.open:  # Must be green (close > open)
        return False
    preceding_highs = [bars[i].high for i in range(max(0, index - 5), index)]
    cleared = sum(1 for h in preceding_highs if bar.high > h)
    return cleared >= 2


def double_bar_break_down(bars, index: int) -> bool:
    """Inverse of double_bar_break_up — for short setups."""
    if index < 2:
        return False
    bar = bars[index]
    if bar.close >= bar.open:  # Must be red
        return False
    preceding_lows = [bars[i].low for i in range(max(0, index - 5), index)]
    cleared = sum(1 for l in preceding_lows if bar.low < l)
    return cleared >= 2