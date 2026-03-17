"""
regime/detector.py — Market Regime Classifier

Classifies the current market into one of 4 states:
  1. TRENDING_BULL  — Above 200 SMA, rising slope, normal volatility
  2. TRENDING_BEAR  — Below 200 SMA, falling slope
  3. HIGH_VOLATILITY — Wilder's ATR spiking relative to baseline
  4. MEAN_REVERTING  — Low ATR, flat MA slope, range-bound

Uses Wilder's ATR (industry standard) for volatility measurement.
See structures/indicators.py for formula reference.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np

from backend.structures.indicators import wilder_atr, atr_ratio, sma


# ==============================================================================
# TYPES
# ==============================================================================

class MarketRegime(str, Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    HIGH_VOLATILITY = "high_volatility"
    MEAN_REVERTING = "mean_reverting"


class StrategyType(str, Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALP = "scalp"


REGIME_ALIGNMENT = {
    MarketRegime.TRENDING_BULL: {
        StrategyType.MOMENTUM: 90,
        StrategyType.MEAN_REVERSION: 20,
        StrategyType.BREAKOUT: 80,
        StrategyType.SCALP: 60,
    },
    MarketRegime.TRENDING_BEAR: {
        StrategyType.MOMENTUM: 70,
        StrategyType.MEAN_REVERSION: 30,
        StrategyType.BREAKOUT: 65,
        StrategyType.SCALP: 50,
    },
    MarketRegime.HIGH_VOLATILITY: {
        StrategyType.MOMENTUM: 40,
        StrategyType.MEAN_REVERSION: 70,
        StrategyType.BREAKOUT: 50,
        StrategyType.SCALP: 80,
    },
    MarketRegime.MEAN_REVERTING: {
        StrategyType.MOMENTUM: 25,
        StrategyType.MEAN_REVERSION: 90,
        StrategyType.BREAKOUT: 30,
        StrategyType.SCALP: 70,
    },
}


# ==============================================================================
# RESULT
# ==============================================================================

@dataclass
class RegimeResult:
    regime: MarketRegime
    confidence: float
    spy_price: float
    sma_200: float
    sma_50: float
    atr_ratio: float
    slope_20d: float
    description: str

    def alignment_for(self, strategy_type: StrategyType) -> float:
        return REGIME_ALIGNMENT.get(self.regime, {}).get(strategy_type, 50.0)

    @property
    def emoji(self) -> str:
        return {"trending_bull": "🟢", "trending_bear": "🔴",
                "high_volatility": "🟡", "mean_reverting": "⚪"}.get(self.regime.value, "⚪")

    @property
    def label(self) -> str:
        return {"trending_bull": "Trending Bull", "trending_bear": "Trending Bear",
                "high_volatility": "High Volatility", "mean_reverting": "Mean Reverting"}.get(self.regime.value, "Unknown")

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value, "label": self.label, "emoji": self.emoji,
            "confidence": self.confidence,
            "spy_price": round(self.spy_price, 2), "sma_200": round(self.sma_200, 2),
            "sma_50": round(self.sma_50, 2), "atr_ratio": round(self.atr_ratio, 2),
            "slope_20d": round(self.slope_20d, 4), "description": self.description,
            "alignment": {st.value: self.alignment_for(st) for st in StrategyType},
        }


# ==============================================================================
# DETECTION
# ==============================================================================

def detect_regime(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    is_spy: bool = False,
) -> RegimeResult:
    closes = np.asarray(closes, dtype=np.float64)
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    n = len(closes)

    if n < 30:
        return RegimeResult(
            regime=MarketRegime.MEAN_REVERTING, confidence=20.0,
            spy_price=0, sma_200=0, sma_50=0, atr_ratio=1.0, slope_20d=0,
            description="Insufficient data for regime detection",
        )

    current = float(closes[-1])

    # --- SMAs ---
    sma_50_val = float(np.mean(closes[-50:])) if n >= 50 else float(np.mean(closes))
    sma_200_val = float(np.mean(closes[-200:])) if n >= 200 else float(np.mean(closes))

    above_50 = current > sma_50_val
    above_200 = current > sma_200_val

    # --- 20-bar slope (% change) ---
    slope_20d = float((closes[-1] - closes[-20]) / closes[-20]) if n >= 20 else 0.0

    # --- ATR ratio using Wilder's ATR (the correct way) ---
    ratio = atr_ratio(highs, lows, closes, atr_period=14, baseline_lookback=60)

    # --- Classify ---
    regime, confidence, desc = _classify(
        current, sma_50_val, sma_200_val, above_50, above_200,
        slope_20d, ratio, is_spy,
    )

    return RegimeResult(
        regime=regime, confidence=confidence,
        spy_price=current if is_spy else 0.0,
        sma_200=sma_200_val, sma_50=sma_50_val,
        atr_ratio=ratio, slope_20d=slope_20d,
        description=desc,
    )


def _classify(
    price, sma_50, sma_200, above_50, above_200,
    slope_20d, atr_ratio_val, is_spy,
):
    label = "Market" if is_spy else "Stock"

    # Rule 1: High volatility override (ATR ratio > 1.8)
    if atr_ratio_val > 1.8:
        conf = min(95, 50 + (atr_ratio_val - 1.8) * 50)
        return (
            MarketRegime.HIGH_VOLATILITY, conf,
            f"{label} in high-volatility regime (ATR {atr_ratio_val:.1f}x normal). "
            f"Expect large swings. Favor scalps and mean reversion.",
        )

    # Rule 2: Trending bull
    if above_200 and slope_20d > 0.005:
        conf = 60.0
        if above_50: conf += 15
        if slope_20d > 0.02: conf += 10
        if atr_ratio_val < 1.2: conf += 5
        spread = (price - sma_200) / sma_200 * 100
        return (
            MarketRegime.TRENDING_BULL, min(95, conf),
            f"{label} trending bullish — {spread:.1f}% above 200 SMA, "
            f"20d slope {slope_20d:+.1%}. Favor momentum and breakouts.",
        )

    # Rule 3: Trending bear
    if not above_200 and slope_20d < -0.005:
        conf = 60.0
        if not above_50: conf += 15
        if slope_20d < -0.02: conf += 10
        if atr_ratio_val > 1.3: conf += 5
        spread = (price - sma_200) / sma_200 * 100
        return (
            MarketRegime.TRENDING_BEAR, min(95, conf),
            f"{label} trending bearish — {abs(spread):.1f}% below 200 SMA, "
            f"20d slope {slope_20d:+.1%}. Favor short momentum, avoid longs.",
        )

    # Rule 4: Mean reverting
    conf = 55.0
    if atr_ratio_val < 0.7: conf += 15
    if abs(slope_20d) < 0.005: conf += 10
    return (
        MarketRegime.MEAN_REVERTING, min(90, conf),
        f"{label} in range/mean-reverting regime — slope {slope_20d:+.1%}, "
        f"ATR {atr_ratio_val:.1f}x. Favor mean reversion, fade extremes.",
    )


# ==============================================================================
# CONVENIENCE
# ==============================================================================

def get_regime_alignment(regime: MarketRegime, strategy_type: StrategyType) -> float:
    return REGIME_ALIGNMENT.get(regime, {}).get(strategy_type, 50.0)

def best_strategy_types(regime: MarketRegime) -> list[tuple[StrategyType, float]]:
    alignments = REGIME_ALIGNMENT.get(regime, {})
    return sorted(alignments.items(), key=lambda x: x[1], reverse=True)