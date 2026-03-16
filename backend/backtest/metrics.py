"""
metrics.py — Quantitative metrics that enhance pattern signals.

Adds research-backed factors on top of raw pattern detection:
  - Relative Strength (RS): How is the stock performing vs. SPY?
  - Market Regime: Is SPY above or below its 200-day SMA? (bull vs bear)
  - Momentum Score: Rate of change over multiple lookback periods

These are lightweight computations on data we already have cached.
"""
from datetime import datetime
from typing import Optional

from backend.backtest.data_fetcher import bars_from_cache


def get_market_regime() -> dict:
    """
    Determine if the broad market is in a BULL or BEAR regime.
    
    Method: SPY close vs. its 200-day SMA.
    This is the simplest and most empirically validated regime indicator.
    Many patterns have significantly different win rates in bull vs bear.
    """
    bars = bars_from_cache("SPY", "1d")
    if bars is None or len(bars.bars) < 200:
        return {"regime": "unknown", "spy_close": 0, "sma_200": 0, "spread_pct": 0}

    closes = [b.close for b in bars.bars]
    current = closes[-1]
    sma_200 = sum(closes[-200:]) / 200
    spread_pct = (current - sma_200) / sma_200 * 100

    regime = "bull" if current > sma_200 else "bear"

    return {
        "regime": regime,
        "spy_close": round(current, 2),
        "sma_200": round(sma_200, 2),
        "spread_pct": round(spread_pct, 2),
        "description": (
            f"SPY at ${current:.2f} is {abs(spread_pct):.1f}% "
            f"{'above' if regime == 'bull' else 'below'} its 200-day SMA (${sma_200:.2f})"
        ),
    }


def get_relative_strength(symbol: str, periods: list[int] = None) -> Optional[dict]:
    """
    Calculate Relative Strength of a stock vs. SPY.
    
    RS = (stock return over period) - (SPY return over same period)
    
    Positive RS = stock outperforming the market.
    
    Empirical research (Jegadeesh & Titman, 1993) shows momentum/RS
    is one of the strongest and most persistent market factors.
    
    Returns RS for multiple lookback periods (20, 60, 120, 252 trading days).
    """
    if periods is None:
        periods = [20, 60, 120, 252]  # ~1mo, 3mo, 6mo, 1yr

    stock_bars = bars_from_cache(symbol, "1d")
    spy_bars = bars_from_cache("SPY", "1d")

    if stock_bars is None or spy_bars is None:
        return None

    stock_closes = [b.close for b in stock_bars.bars]
    spy_closes = [b.close for b in spy_bars.bars]

    rs_data = {}
    for period in periods:
        if len(stock_closes) < period + 1 or len(spy_closes) < period + 1:
            continue

        stock_return = (stock_closes[-1] - stock_closes[-period - 1]) / stock_closes[-period - 1]
        spy_return = (spy_closes[-1] - spy_closes[-period - 1]) / spy_closes[-period - 1]
        rs = stock_return - spy_return

        rs_data[f"{period}d"] = {
            "stock_return": round(stock_return * 100, 2),
            "spy_return": round(spy_return * 100, 2),
            "relative_strength": round(rs * 100, 2),
        }

    # Composite RS score (weighted average)
    weights = {20: 0.1, 60: 0.2, 120: 0.3, 252: 0.4}
    weighted_sum = 0
    weight_total = 0
    for period in periods:
        key = f"{period}d"
        if key in rs_data:
            weighted_sum += rs_data[key]["relative_strength"] * weights.get(period, 0.25)
            weight_total += weights.get(period, 0.25)

    composite = weighted_sum / weight_total if weight_total > 0 else 0

    return {
        "symbol": symbol,
        "periods": rs_data,
        "composite_rs": round(composite, 2),
        "assessment": (
            "Strong outperformer" if composite > 10 else
            "Moderate outperformer" if composite > 3 else
            "Market performer" if composite > -3 else
            "Underperformer" if composite > -10 else
            "Significant laggard"
        ),
    }


def get_momentum_score(symbol: str) -> Optional[dict]:
    """
    Multi-timeframe momentum score (0-100).
    
    Checks:
    - Price vs. 9 EMA (short-term)
    - Price vs. 21 EMA (medium-term)  
    - Price vs. 50 SMA (intermediate)
    - 20-day rate of change
    
    Each factor contributes 25 points max → total 0-100.
    """
    bars = bars_from_cache(symbol, "1d")
    if bars is None or len(bars.bars) < 60:
        return None

    closes = [b.close for b in bars.bars]
    current = closes[-1]
    score = 0
    factors = []

    # Factor 1: Price vs 9 EMA
    ema9 = _calc_ema(closes, 9)
    if ema9 is not None:
        if current > ema9:
            score += 25
            factors.append("Above 9 EMA")
        else:
            factors.append("Below 9 EMA")

    # Factor 2: Price vs 21 EMA
    ema21 = _calc_ema(closes, 21)
    if ema21 is not None:
        if current > ema21:
            score += 25
            factors.append("Above 21 EMA")
        else:
            factors.append("Below 21 EMA")

    # Factor 3: Price vs 50 SMA
    if len(closes) >= 50:
        sma50 = sum(closes[-50:]) / 50
        if current > sma50:
            score += 25
            factors.append("Above 50 SMA")
        else:
            factors.append("Below 50 SMA")

    # Factor 4: 20-day ROC (positive = bullish momentum)
    if len(closes) >= 21:
        roc = (current - closes[-21]) / closes[-21] * 100
        if roc > 0:
            # Scale: 0% = 0pts, 5%+ = 25pts
            roc_points = min(25, roc / 5.0 * 25)
            score += roc_points
            factors.append(f"+{roc:.1f}% in 20d")
        else:
            factors.append(f"{roc:.1f}% in 20d")

    return {
        "symbol": symbol,
        "momentum_score": round(score, 1),
        "factors": factors,
        "assessment": (
            "Very strong" if score >= 80 else
            "Strong" if score >= 60 else
            "Neutral" if score >= 40 else
            "Weak" if score >= 20 else
            "Very weak"
        ),
    }


def _calc_ema(prices: list[float], period: int) -> Optional[float]:
    """Calculate the current EMA value (just the latest point)."""
    if len(prices) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period
    for i in range(period, len(prices)):
        ema = prices[i] * k + ema * (1 - k)
    return ema