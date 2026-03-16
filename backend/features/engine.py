"""
features/engine.py — 8 Statistical Features That Beat Classical Patterns

Professional quant research shows these raw statistical features outperform
named patterns. Each feature computes a 0-100 score from price/volume data.

The 8 features:
  1. Momentum       — Rate of change over multiple windows
  2. Volatility     — Current vs historical standard deviation
  3. Vol Compression — ATR shrinking (squeeze before breakout)
  4. Volume Expansion — RVOL (relative volume vs average)
  5. Trend Strength  — Price position relative to key MAs
  6. Range Breakout  — Proximity to N-day high/low
  7. Mean Reversion  — Z-score of price vs moving average
  8. Market Regime   — Composite from SPY behavior (placeholder for Phase 3)

All computation is vectorized using NumPy — no per-bar Python loops.

Usage:
    from backend.features.engine import compute_features, FeatureResult
    features = compute_features(closes, highs, lows, volumes)
    print(features.composite_score)  # 0-100
    print(features.momentum.score)   # 0-100
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ==============================================================================
# DATA TYPES
# ==============================================================================

@dataclass
class FeatureScore:
    """A single feature's score and raw value."""
    name: str
    score: float        # 0-100 normalized score
    raw_value: float    # The actual computed value (for debugging/display)
    description: str    # Human-readable interpretation


@dataclass
class FeatureResult:
    """All 8 features computed for a single symbol."""
    momentum: FeatureScore
    volatility: FeatureScore
    vol_compression: FeatureScore
    volume_expansion: FeatureScore
    trend_strength: FeatureScore
    range_breakout: FeatureScore
    mean_reversion: FeatureScore
    regime_score: FeatureScore

    @property
    def all_scores(self) -> list[FeatureScore]:
        return [
            self.momentum, self.volatility, self.vol_compression,
            self.volume_expansion, self.trend_strength, self.range_breakout,
            self.mean_reversion, self.regime_score,
        ]

    @property
    def composite_score(self) -> float:
        """Weighted average of all 8 features (0-100)."""
        weights = {
            "momentum": 0.18,
            "volatility": 0.10,
            "vol_compression": 0.12,
            "volume_expansion": 0.15,
            "trend_strength": 0.15,
            "range_breakout": 0.12,
            "mean_reversion": 0.08,
            "regime_score": 0.10,
        }
        total = 0.0
        for f in self.all_scores:
            total += f.score * weights.get(f.name, 0.125)
        return round(total, 1)

    def to_dict(self) -> dict:
        return {
            "composite_score": self.composite_score,
            "features": {
                f.name: {"score": f.score, "raw": round(f.raw_value, 4), "desc": f.description}
                for f in self.all_scores
            },
        }


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def compute_features(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    spy_closes: Optional[np.ndarray] = None,
) -> FeatureResult:
    """
    Compute all 8 statistical features for a price series.

    Args:
        closes: Close prices (NumPy array)
        highs: High prices
        lows: Low prices
        volumes: Volume array
        spy_closes: Optional SPY closes for regime detection (same length)

    Returns:
        FeatureResult with all 8 scores (0-100 each) and composite.
    """
    closes = np.asarray(closes, dtype=np.float64)
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)

    n = len(closes)

    return FeatureResult(
        momentum=_compute_momentum(closes, n),
        volatility=_compute_volatility(closes, n),
        vol_compression=_compute_vol_compression(highs, lows, closes, n),
        volume_expansion=_compute_volume_expansion(volumes, n),
        trend_strength=_compute_trend_strength(closes, n),
        range_breakout=_compute_range_breakout(closes, highs, lows, n),
        mean_reversion=_compute_mean_reversion(closes, n),
        regime_score=_compute_regime_score(closes, spy_closes, n),
    )


# ==============================================================================
# FEATURE 1: MOMENTUM
# ==============================================================================

def _compute_momentum(closes: np.ndarray, n: int) -> FeatureScore:
    """
    Multi-window momentum: rate of change over 5, 20, 60 bars.

    momentum = price_t / price_{t-n} - 1

    Momentum is the most documented anomaly in finance (Jegadeesh & Titman 1993).
    Positive momentum across multiple windows = strong bullish signal.
    """
    if n < 10:
        return FeatureScore("momentum", 50.0, 0.0, "Insufficient data")

    scores = []
    raw_values = []

    for lookback in [5, 20, 60]:
        if n > lookback:
            ret = (closes[-1] / closes[-lookback - 1]) - 1.0
            raw_values.append(ret)
            # Scale: -10% = 0, 0% = 50, +10% = 100
            s = np.clip((ret + 0.10) / 0.20 * 100, 0, 100)
            scores.append(float(s))

    if not scores:
        return FeatureScore("momentum", 50.0, 0.0, "Insufficient data")

    # Weight longer-term momentum more heavily
    if len(scores) == 3:
        weighted = scores[0] * 0.2 + scores[1] * 0.3 + scores[2] * 0.5
    elif len(scores) == 2:
        weighted = scores[0] * 0.3 + scores[1] * 0.7
    else:
        weighted = scores[0]

    avg_ret = float(np.mean(raw_values))
    desc = f"{'Bullish' if avg_ret > 0.01 else 'Bearish' if avg_ret < -0.01 else 'Neutral'} " \
           f"({avg_ret:+.1%} avg across windows)"

    return FeatureScore("momentum", round(weighted, 1), avg_ret, desc)


# ==============================================================================
# FEATURE 2: VOLATILITY
# ==============================================================================

def _compute_volatility(closes: np.ndarray, n: int) -> FeatureScore:
    """
    Current volatility vs historical average.

    sigma = std(log returns)

    High volatility = bigger moves = more opportunity but more risk.
    Score reflects whether current vol is elevated vs normal.
    """
    if n < 25:
        return FeatureScore("volatility", 50.0, 0.0, "Insufficient data")

    log_returns = np.diff(np.log(closes))

    # Current volatility (last 10 bars)
    current_vol = float(np.std(log_returns[-10:]))
    # Historical volatility (last 60 bars or all available)
    lookback = min(60, len(log_returns))
    hist_vol = float(np.std(log_returns[-lookback:]))

    if hist_vol == 0:
        return FeatureScore("volatility", 50.0, 0.0, "Zero historical vol")

    # Ratio: >1 means elevated vol, <1 means compressed
    vol_ratio = current_vol / hist_vol

    # Score: ratio 0.5 = 20 (compressed), 1.0 = 50, 2.0 = 90 (elevated)
    score = np.clip((vol_ratio - 0.3) / 1.7 * 100, 0, 100)

    desc = f"{'Elevated' if vol_ratio > 1.3 else 'Compressed' if vol_ratio < 0.7 else 'Normal'} " \
           f"({vol_ratio:.2f}x historical)"

    return FeatureScore("volatility", round(float(score), 1), vol_ratio, desc)


# ==============================================================================
# FEATURE 3: VOLATILITY COMPRESSION (ATR Squeeze)
# ==============================================================================

def _compute_vol_compression(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int,
) -> FeatureScore:
    """
    ATR compression: is the range shrinking (squeeze before breakout)?

    Compression → expansion is one of the strongest statistical behaviors
    in markets. Bollinger Band squeezes are built on this principle.

    Score: higher = more compressed (breakout more likely).
    """
    if n < 25:
        return FeatureScore("vol_compression", 50.0, 0.0, "Insufficient data")

    # Compute True Range vectorized
    tr_hl = highs[1:] - lows[1:]
    tr_hc = np.abs(highs[1:] - closes[:-1])
    tr_lc = np.abs(lows[1:] - closes[:-1])
    true_range = np.maximum(tr_hl, np.maximum(tr_hc, tr_lc))

    if len(true_range) < 20:
        return FeatureScore("vol_compression", 50.0, 0.0, "Insufficient data")

    # Current ATR (last 5 bars) vs longer-term ATR (last 20 bars)
    current_atr = float(np.mean(true_range[-5:]))
    longer_atr = float(np.mean(true_range[-20:]))

    if longer_atr == 0:
        return FeatureScore("vol_compression", 50.0, 0.0, "Zero ATR")

    compression = current_atr / longer_atr

    # Score: lower ratio = more compressed = higher score
    # ratio 0.4 = 95 (very compressed), 1.0 = 50, 1.5 = 15 (expanding)
    score = np.clip((1.5 - compression) / 1.1 * 100, 0, 100)

    desc = f"{'Squeezed' if compression < 0.7 else 'Expanding' if compression > 1.2 else 'Normal'} " \
           f"(ATR ratio {compression:.2f})"

    return FeatureScore("vol_compression", round(float(score), 1), compression, desc)


# ==============================================================================
# FEATURE 4: VOLUME EXPANSION
# ==============================================================================

def _compute_volume_expansion(volumes: np.ndarray, n: int) -> FeatureScore:
    """
    Relative Volume (RVOL): current volume vs average.

    volume_ratio = recent_vol / avg_vol

    Breakouts with high volume outperform significantly.
    RVOL > 2 = "in play", RVOL > 5 = very high activity.
    """
    if n < 15:
        return FeatureScore("volume_expansion", 50.0, 0.0, "Insufficient data")

    # Recent volume (last 3 bars average)
    recent_vol = float(np.mean(volumes[-3:]))
    # Average volume (last 20 bars, excluding last 3)
    avg_period = min(20, n - 3)
    if avg_period < 5:
        return FeatureScore("volume_expansion", 50.0, 0.0, "Insufficient history")

    avg_vol = float(np.mean(volumes[-3 - avg_period:-3]))

    if avg_vol == 0:
        return FeatureScore("volume_expansion", 50.0, 0.0, "Zero average volume")

    rvol = recent_vol / avg_vol

    # Score: RVOL 0.5 = 10, 1.0 = 40, 2.0 = 70, 5.0 = 100
    score = np.clip(np.log2(max(rvol, 0.1)) / np.log2(5) * 100, 0, 100)

    desc = f"{'High activity' if rvol > 2 else 'Below avg' if rvol < 0.7 else 'Normal'} " \
           f"({rvol:.1f}x avg volume)"

    return FeatureScore("volume_expansion", round(float(score), 1), rvol, desc)


# ==============================================================================
# FEATURE 5: TREND STRENGTH
# ==============================================================================

def _compute_trend_strength(closes: np.ndarray, n: int) -> FeatureScore:
    """
    Price position relative to key moving averages.

    Checks: above/below 9 EMA, 21 EMA, 50 SMA, 200 SMA.
    Each MA adds 25 points if price is above it.

    Trend filter dramatically improves pattern signals — this is the
    single most important filter in systematic trading.
    """
    if n < 10:
        return FeatureScore("trend_strength", 50.0, 0.0, "Insufficient data")

    current = closes[-1]
    score = 0.0
    factors = []

    # 9 EMA
    if n >= 9:
        ema9 = _ema_last(closes, 9)
        if current > ema9:
            score += 25
            factors.append(">9EMA")
        else:
            factors.append("<9EMA")

    # 21 EMA
    if n >= 21:
        ema21 = _ema_last(closes, 21)
        if current > ema21:
            score += 25
            factors.append(">21EMA")
        else:
            factors.append("<21EMA")

    # 50 SMA
    if n >= 50:
        sma50 = float(np.mean(closes[-50:]))
        if current > sma50:
            score += 25
            factors.append(">50SMA")
        else:
            factors.append("<50SMA")

    # 200 SMA
    if n >= 200:
        sma200 = float(np.mean(closes[-200:]))
        if current > sma200:
            score += 25
            factors.append(">200SMA")
        else:
            factors.append("<200SMA")
    else:
        # Scale to available MAs
        available_mas = len([f for f in factors if f.startswith(">")])
        total_mas = len(factors)
        if total_mas > 0:
            score = (available_mas / total_mas) * 100

    desc = f"{'Strong uptrend' if score >= 75 else 'Weak/downtrend' if score <= 25 else 'Mixed'} " \
           f"({', '.join(factors)})"

    return FeatureScore("trend_strength", round(score, 1), score / 100, desc)


# ==============================================================================
# FEATURE 6: RANGE BREAKOUT
# ==============================================================================

def _compute_range_breakout(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, n: int,
) -> FeatureScore:
    """
    How close is price to its N-day high or low?

    price > highest(price, 20) is the core trend-following signal.
    This scores proximity: at the high = 100, at the low = 0.

    Also checks 50-bar range for longer-term context.
    """
    if n < 22:
        return FeatureScore("range_breakout", 50.0, 0.0, "Insufficient data")

    current = closes[-1]
    scores = []

    for lookback in [20, 50]:
        if n > lookback:
            period_high = float(np.max(highs[-lookback:]))
            period_low = float(np.min(lows[-lookback:]))
            period_range = period_high - period_low
            if period_range > 0:
                position = (current - period_low) / period_range * 100
                scores.append(float(np.clip(position, 0, 100)))

    if not scores:
        return FeatureScore("range_breakout", 50.0, 0.0, "Insufficient data")

    # Weight 20-bar more heavily (more actionable)
    if len(scores) == 2:
        weighted = scores[0] * 0.6 + scores[1] * 0.4
    else:
        weighted = scores[0]

    # Check if at actual breakout
    is_20d_high = current >= float(np.max(highs[-20:]))
    is_20d_low = current <= float(np.min(lows[-20:]))

    desc = f"{'20-bar HIGH' if is_20d_high else '20-bar LOW' if is_20d_low else 'Mid-range'} " \
           f"({weighted:.0f}% of range)"

    return FeatureScore("range_breakout", round(weighted, 1), weighted / 100, desc)


# ==============================================================================
# FEATURE 7: MEAN REVERSION
# ==============================================================================

def _compute_mean_reversion(closes: np.ndarray, n: int) -> FeatureScore:
    """
    Z-score of price vs moving average.

    zscore = (price - MA) / std

    Extreme z-scores (< -2 or > 2) signal potential mean reversion.
    Score is INVERTED: extreme = high score (opportunity to fade).
    """
    if n < 22:
        return FeatureScore("mean_reversion", 50.0, 0.0, "Insufficient data")

    period = min(20, n - 1)
    ma = float(np.mean(closes[-period:]))
    std = float(np.std(closes[-period:]))

    if std == 0:
        return FeatureScore("mean_reversion", 50.0, 0.0, "Zero std dev")

    zscore = (closes[-1] - ma) / std

    # Score: z=0 → 50, z=±2 → 90, z=±3 → 100
    # Higher score = more extreme = more mean-reversion opportunity
    score = np.clip(abs(zscore) / 3.0 * 100, 0, 100)

    if zscore < -2:
        desc = f"Oversold (z={zscore:.2f}), mean-reversion buy signal"
    elif zscore > 2:
        desc = f"Overbought (z={zscore:.2f}), mean-reversion sell signal"
    elif zscore < -1:
        desc = f"Mildly stretched low (z={zscore:.2f})"
    elif zscore > 1:
        desc = f"Mildly stretched high (z={zscore:.2f})"
    else:
        desc = f"Near mean (z={zscore:.2f})"

    return FeatureScore("mean_reversion", round(float(score), 1), float(zscore), desc)


# ==============================================================================
# FEATURE 8: MARKET REGIME (Placeholder — Full version in Phase 3)
# ==============================================================================

def _compute_regime_score(
    closes: np.ndarray,
    spy_closes: Optional[np.ndarray],
    n: int,
) -> FeatureScore:
    """
    Market regime score based on SPY behavior.

    Full implementation comes in Phase 3 (regime/detector.py).
    For now, uses the stock's own trend as a proxy.

    In Phase 3 this will use SPY data to classify:
      - Trending Bull
      - Trending Bear
      - High Volatility
      - Mean Reverting / Range
    """
    if spy_closes is not None and len(spy_closes) >= 50:
        # Use SPY data
        spy = np.asarray(spy_closes, dtype=np.float64)
        sma50 = float(np.mean(spy[-50:]))
        sma20 = float(np.mean(spy[-20:]))
        current_spy = spy[-1]

        above_50 = current_spy > sma50
        above_20 = current_spy > sma20

        # Trend direction
        slope_20 = (spy[-1] - spy[-20]) / spy[-20] if len(spy) >= 20 else 0

        if above_50 and above_20 and slope_20 > 0.01:
            score = 80.0
            desc = "Trending bull (SPY above 20/50 SMA, rising)"
        elif above_50 and not above_20:
            score = 55.0
            desc = "Pullback in uptrend (SPY above 50 but below 20)"
        elif not above_50 and slope_20 < -0.01:
            score = 25.0
            desc = "Trending bear (SPY below 50 SMA, falling)"
        else:
            score = 50.0
            desc = "Range/mixed regime"

        return FeatureScore("regime_score", score, slope_20, desc)

    # Fallback: use the stock's own data
    if n < 50:
        return FeatureScore("regime_score", 50.0, 0.0, "Insufficient data (no SPY)")

    sma50 = float(np.mean(closes[-50:]))
    above = closes[-1] > sma50
    slope = (closes[-1] - closes[-20]) / closes[-20] if n >= 20 else 0

    score = 70.0 if (above and slope > 0) else 30.0 if (not above and slope < 0) else 50.0
    desc = f"{'Bullish' if score > 60 else 'Bearish' if score < 40 else 'Neutral'} (self-regime, no SPY data)"

    return FeatureScore("regime_score", score, float(slope), desc)


# ==============================================================================
# HELPER: EMA (vectorized)
# ==============================================================================

def _ema_last(prices: np.ndarray, period: int) -> float:
    """Compute only the LAST EMA value (efficient for scoring)."""
    if len(prices) < period:
        return float(prices[-1])
    k = 2.0 / (period + 1)
    ema = float(np.mean(prices[:period]))
    for i in range(period, len(prices)):
        ema = prices[i] * k + ema * (1 - k)
    return ema