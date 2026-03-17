"""
features/engine.py — 8 Statistical Features That Beat Classical Patterns

All ATR/EMA calculations now use the shared indicators module
(backend/structures/indicators.py) to ensure consistent math everywhere.

The 8 features:
  1. Momentum       — Rate of change over multiple windows
  2. Volatility     — Current vs historical standard deviation of returns
  3. Vol Compression — Wilder's ATR ratio (current vs baseline)
  4. Volume Expansion — RVOL (relative volume vs average)
  5. Trend Strength  — Price position relative to key MAs
  6. Range Breakout  — Proximity to N-day high/low
  7. Mean Reversion  — Z-score of price vs moving average
  8. Market Regime   — Composite from SPY behavior
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.structures.indicators import wilder_atr, atr_ratio, ema_last


# ==============================================================================
# DATA TYPES
# ==============================================================================

@dataclass
class FeatureScore:
    """A single feature's score and raw value."""
    name: str
    score: float        # 0-100
    raw_value: float    # Actual computed value
    description: str


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
        weights = {
            "momentum": 0.18, "volatility": 0.10, "vol_compression": 0.12,
            "volume_expansion": 0.15, "trend_strength": 0.15,
            "range_breakout": 0.12, "mean_reversion": 0.08, "regime_score": 0.10,
        }
        total = sum(f.score * weights.get(f.name, 0.125) for f in self.all_scores)
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
    """
    if n < 10:
        return FeatureScore("momentum", 50.0, 0.0, "Insufficient data")

    scores = []
    raw_values = []

    for lookback in [5, 20, 60]:
        if n > lookback:
            ret = (closes[-1] / closes[-lookback - 1]) - 1.0
            raw_values.append(ret)
            s = np.clip((ret + 0.10) / 0.20 * 100, 0, 100)
            scores.append(float(s))

    if not scores:
        return FeatureScore("momentum", 50.0, 0.0, "Insufficient data")

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
    Current volatility vs historical. sigma = std(log returns).
    """
    if n < 25:
        return FeatureScore("volatility", 50.0, 0.0, "Insufficient data")

    log_returns = np.diff(np.log(closes))
    current_vol = float(np.std(log_returns[-10:]))
    lookback = min(60, len(log_returns))
    hist_vol = float(np.std(log_returns[-lookback:]))

    if hist_vol == 0:
        return FeatureScore("volatility", 50.0, 0.0, "Zero historical vol")

    vol_ratio = current_vol / hist_vol
    score = np.clip((vol_ratio - 0.3) / 1.7 * 100, 0, 100)

    desc = f"{'Elevated' if vol_ratio > 1.3 else 'Compressed' if vol_ratio < 0.7 else 'Normal'} " \
           f"({vol_ratio:.2f}x historical)"

    return FeatureScore("volatility", round(float(score), 1), vol_ratio, desc)


# ==============================================================================
# FEATURE 3: VOLATILITY COMPRESSION (Wilder's ATR)
# ==============================================================================

def _compute_vol_compression(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int,
) -> FeatureScore:
    """
    ATR compression using Wilder's ATR ratio from shared indicators.

    Uses atr_ratio() which computes:
      current Wilder ATR(14) / median baseline ATR(14)

    Score: lower ratio = more compressed = higher score (breakout likely).
    """
    if n < 30:
        return FeatureScore("vol_compression", 50.0, 0.0, "Insufficient data")

    ratio = atr_ratio(highs, lows, closes, atr_period=14, baseline_lookback=60)

    # Score: ratio 0.4 = 95, 1.0 = 50, 1.5 = 15 (expanding)
    score = np.clip((1.5 - ratio) / 1.1 * 100, 0, 100)

    desc = f"{'Squeezed' if ratio < 0.7 else 'Expanding' if ratio > 1.2 else 'Normal'} " \
           f"(ATR ratio {ratio:.2f})"

    return FeatureScore("vol_compression", round(float(score), 1), ratio, desc)


# ==============================================================================
# FEATURE 4: VOLUME EXPANSION
# ==============================================================================

def _compute_volume_expansion(volumes: np.ndarray, n: int) -> FeatureScore:
    """RVOL: recent volume vs average. Breakouts with high volume outperform."""
    if n < 15:
        return FeatureScore("volume_expansion", 50.0, 0.0, "Insufficient data")

    recent_vol = float(np.mean(volumes[-3:]))
    avg_period = min(20, n - 3)
    if avg_period < 5:
        return FeatureScore("volume_expansion", 50.0, 0.0, "Insufficient history")

    avg_vol = float(np.mean(volumes[-3 - avg_period:-3]))
    if avg_vol == 0:
        return FeatureScore("volume_expansion", 50.0, 0.0, "Zero average volume")

    rvol = recent_vol / avg_vol
    score = np.clip(np.log2(max(rvol, 0.1)) / np.log2(5) * 100, 0, 100)

    desc = f"{'High activity' if rvol > 2 else 'Below avg' if rvol < 0.7 else 'Normal'} " \
           f"({rvol:.1f}x avg volume)"

    return FeatureScore("volume_expansion", round(float(score), 1), rvol, desc)


# ==============================================================================
# FEATURE 5: TREND STRENGTH
# ==============================================================================

def _compute_trend_strength(closes: np.ndarray, n: int) -> FeatureScore:
    """Price vs 9 EMA, 21 EMA, 50 SMA, 200 SMA. Each adds 25 points."""
    if n < 10:
        return FeatureScore("trend_strength", 50.0, 0.0, "Insufficient data")

    current = closes[-1]
    score = 0.0
    factors = []

    if n >= 9:
        ema9 = ema_last(closes, 9)
        if current > ema9: score += 25; factors.append(">9EMA")
        else: factors.append("<9EMA")

    if n >= 21:
        ema21 = ema_last(closes, 21)
        if current > ema21: score += 25; factors.append(">21EMA")
        else: factors.append("<21EMA")

    if n >= 50:
        sma50 = float(np.mean(closes[-50:]))
        if current > sma50: score += 25; factors.append(">50SMA")
        else: factors.append("<50SMA")

    if n >= 200:
        sma200 = float(np.mean(closes[-200:]))
        if current > sma200: score += 25; factors.append(">200SMA")
        else: factors.append("<200SMA")
    else:
        above = len([f for f in factors if f.startswith(">")])
        total = len(factors)
        if total > 0:
            score = (above / total) * 100

    desc = f"{'Strong uptrend' if score >= 75 else 'Weak/downtrend' if score <= 25 else 'Mixed'} " \
           f"({', '.join(factors)})"

    return FeatureScore("trend_strength", round(score, 1), score / 100, desc)


# ==============================================================================
# FEATURE 6: RANGE BREAKOUT
# ==============================================================================

def _compute_range_breakout(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, n: int,
) -> FeatureScore:
    """Proximity to N-day high/low. At high = 100, at low = 0."""
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

    weighted = scores[0] * 0.6 + scores[1] * 0.4 if len(scores) == 2 else scores[0]

    is_20d_high = current >= float(np.max(highs[-20:]))
    is_20d_low = current <= float(np.min(lows[-20:]))

    desc = f"{'20-bar HIGH' if is_20d_high else '20-bar LOW' if is_20d_low else 'Mid-range'} " \
           f"({weighted:.0f}% of range)"

    return FeatureScore("range_breakout", round(weighted, 1), weighted / 100, desc)


# ==============================================================================
# FEATURE 7: MEAN REVERSION
# ==============================================================================

def _compute_mean_reversion(closes: np.ndarray, n: int) -> FeatureScore:
    """Z-score of price vs 20-MA. Extreme z = high score (opportunity to fade)."""
    if n < 22:
        return FeatureScore("mean_reversion", 50.0, 0.0, "Insufficient data")

    period = min(20, n - 1)
    ma = float(np.mean(closes[-period:]))
    std = float(np.std(closes[-period:]))

    if std == 0:
        return FeatureScore("mean_reversion", 50.0, 0.0, "Zero std dev")

    zscore = (closes[-1] - ma) / std
    score = np.clip(abs(zscore) / 3.0 * 100, 0, 100)

    if zscore < -2: desc = f"Oversold (z={zscore:.2f}), mean-reversion buy signal"
    elif zscore > 2: desc = f"Overbought (z={zscore:.2f}), mean-reversion sell signal"
    elif zscore < -1: desc = f"Mildly stretched low (z={zscore:.2f})"
    elif zscore > 1: desc = f"Mildly stretched high (z={zscore:.2f})"
    else: desc = f"Near mean (z={zscore:.2f})"

    return FeatureScore("mean_reversion", round(float(score), 1), float(zscore), desc)


# ==============================================================================
# FEATURE 8: REGIME SCORE (placeholder for Phase 3 full version)
# ==============================================================================

def _compute_regime_score(
    closes: np.ndarray, spy_closes: Optional[np.ndarray], n: int,
) -> FeatureScore:
    """Market regime score. Full version in Phase 3; uses SPY if available."""
    if spy_closes is not None and len(spy_closes) >= 50:
        spy = np.asarray(spy_closes, dtype=np.float64)
        sma50 = float(np.mean(spy[-50:]))
        sma20 = float(np.mean(spy[-20:]))
        current_spy = spy[-1]

        above_50 = current_spy > sma50
        above_20 = current_spy > sma20
        slope_20 = (spy[-1] - spy[-20]) / spy[-20] if len(spy) >= 20 else 0

        if above_50 and above_20 and slope_20 > 0.01:
            return FeatureScore("regime_score", 80.0, slope_20,
                                "Trending bull (SPY above 20/50 SMA, rising)")
        elif above_50 and not above_20:
            return FeatureScore("regime_score", 55.0, slope_20,
                                "Pullback in uptrend (SPY above 50 but below 20)")
        elif not above_50 and slope_20 < -0.01:
            return FeatureScore("regime_score", 25.0, slope_20,
                                "Trending bear (SPY below 50 SMA, falling)")
        else:
            return FeatureScore("regime_score", 50.0, slope_20, "Range/mixed regime")

    if n < 50:
        return FeatureScore("regime_score", 50.0, 0.0, "Insufficient data (no SPY)")

    sma50 = float(np.mean(closes[-50:]))
    above = closes[-1] > sma50
    slope = (closes[-1] - closes[-20]) / closes[-20] if n >= 20 else 0

    score = 70.0 if (above and slope > 0) else 30.0 if (not above and slope < 0) else 50.0
    desc = f"{'Bullish' if score > 60 else 'Bearish' if score < 40 else 'Neutral'} (self-regime, no SPY data)"

    return FeatureScore("regime_score", score, float(slope), desc)