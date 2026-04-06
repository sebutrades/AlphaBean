"""
scoring/multi_factor.py — Multi-Factor Trade Setup Scoring  (v2)

Composite score formula (0-100):

  composite =
      pattern_confidence × 0.20   ← Phase 4 classifier quality
    + feature_score      × 0.22   ← 8 statistical features (pattern-type-aware)
    + strategy_score     × 0.18   ← rolling live hot score
    + regime_alignment   × 0.15   ← strategy type vs current regime
    + backtest_edge      × 0.10   ← edge score from backtest cache
    + volume_confirm     × 0.10   ← RVOL at signal
    + rr_quality         × 0.05   ← R:R ratio quality gate

Changes from v1:
  • Defaults raised: strategy_score 40→50, backtest_edge 30→50 when no data.
    The old below-neutral defaults suppressed all scores on a cold system.
  • feature_score is now pattern-type-aware: breakout setups up-weight momentum
    and range_breakout; mean-reversion setups up-weight mean_reversion and
    vol_compression; scalp setups up-weight volume_expansion.
  • Volume double-count fixed: volume_expansion removed from the feature
    composite sub-weights; volume lives only in the dedicated volume_confirm slot.
  • R:R quality added as a 5 % gate (1.0 R:R = 0, 2.0 = 50, 3.0+ = 100).
  • Weights sum to exactly 1.0 (0.20+0.22+0.18+0.15+0.10+0.10+0.05).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.patterns.registry import TradeSetup, PATTERN_META
from backend.features.engine import compute_features, FeatureResult
from backend.regime.detector import (
    RegimeResult, MarketRegime, StrategyType, detect_regime, get_regime_alignment,
)
from backend.strategies.evaluator import StrategyEvaluator


# ── Weights ────────────────────────────────────────────────────────────────────
# Must sum to 1.0

WEIGHTS = {
    "pattern_confidence": 0.20,
    "feature_score":      0.22,
    "strategy_score":     0.18,
    "regime_alignment":   0.15,
    "backtest_edge":      0.10,
    "volume_confirm":     0.10,
    "rr_quality":         0.05,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── Pattern-type-aware feature sub-weights ─────────────────────────────────────
# Each dict must sum to ~1.0 (will be normalised anyway).
# Volume_expansion is intentionally excluded here — it lives in volume_confirm.

_FEATURE_WEIGHTS_BREAKOUT = {
    "momentum":       0.22,   # price must be moving
    "volatility":     0.10,
    "vol_compression": 0.15,  # squeeze before breakout
    "trend_strength": 0.22,   # must be in trend
    "range_breakout": 0.20,   # near N-day high/low
    "mean_reversion": 0.04,   # counterproductive for breakouts
    "regime_score":   0.07,
}

_FEATURE_WEIGHTS_MEAN_REVERSION = {
    "momentum":       0.05,   # extended moves are good for reversals
    "volatility":     0.15,   # elevated vol = opportunity
    "vol_compression": 0.20,  # tight pre-reversal
    "trend_strength": 0.08,   # not critical; reversal works against trend
    "range_breakout": 0.07,   # at extremes of range = good
    "mean_reversion": 0.35,   # MOST important for this type
    "regime_score":   0.10,
}

_FEATURE_WEIGHTS_MOMENTUM = {
    "momentum":       0.30,   # most important
    "volatility":     0.10,
    "vol_compression": 0.10,
    "trend_strength": 0.28,   # second most important
    "range_breakout": 0.12,
    "mean_reversion": 0.03,
    "regime_score":   0.07,
}

_FEATURE_WEIGHTS_SCALP = {
    "momentum":       0.15,
    "volatility":     0.18,   # elevated vol = wider intraday range
    "vol_compression": 0.12,
    "trend_strength": 0.12,
    "range_breakout": 0.18,
    "mean_reversion": 0.10,
    "regime_score":   0.15,
}

# Fallback: equal weights (no vol_expansion, normalised)
_FEATURE_WEIGHTS_DEFAULT = {
    "momentum": 0.17, "volatility": 0.12, "vol_compression": 0.15,
    "trend_strength": 0.19, "range_breakout": 0.15,
    "mean_reversion": 0.10, "regime_score": 0.12,
}

_FEATURE_WEIGHT_BY_TYPE = {
    "breakout":       _FEATURE_WEIGHTS_BREAKOUT,
    "momentum":       _FEATURE_WEIGHTS_MOMENTUM,
    "mean_reversion": _FEATURE_WEIGHTS_MEAN_REVERSION,
    "scalp":          _FEATURE_WEIGHTS_SCALP,
}


def _pattern_aware_feature_score(features: FeatureResult, strategy_type: str) -> float:
    """
    Compute the feature score using weights appropriate for the setup's
    strategy type — breakout, momentum, mean_reversion, or scalp.
    Volume expansion is excluded from this calculation (deduplicated into
    the separate volume_confirm factor).
    """
    weights = _FEATURE_WEIGHT_BY_TYPE.get(strategy_type, _FEATURE_WEIGHTS_DEFAULT)

    name_map = {
        "momentum":        features.momentum,
        "volatility":      features.volatility,
        "vol_compression": features.vol_compression,
        "trend_strength":  features.trend_strength,
        "range_breakout":  features.range_breakout,
        "mean_reversion":  features.mean_reversion,
        "regime_score":    features.regime_score,
    }

    total_w = 0.0
    weighted_score = 0.0
    for name, fs in name_map.items():
        w = weights.get(name, 0.0)
        weighted_score += fs.score * w
        total_w += w

    if total_w == 0:
        return 50.0
    return round(weighted_score / total_w, 1)


def _rr_quality_score(setup: TradeSetup) -> float:
    """
    Score the risk:reward ratio (0-100).
      < 1.0  → 0   (unacceptable)
      1.0    → 10
      1.5    → 35
      2.0    → 60
      2.5    → 80
      3.0+   → 100
    Linear interpolation between anchors.
    """
    entry  = setup.entry_price
    stop   = setup.stop_loss
    t1     = getattr(setup, "target_1", None) or getattr(setup, "target_price", None) or 0

    risk   = abs(entry - stop)
    reward = abs(t1 - entry) if t1 else 0

    if risk <= 0 or reward <= 0:
        return 40.0   # No info — neutral rather than zero

    rr = reward / risk

    if rr < 1.0:
        return 0.0
    elif rr < 1.5:
        return 10.0 + (rr - 1.0) / 0.5 * 25.0    # 10 → 35
    elif rr < 2.0:
        return 35.0 + (rr - 1.5) / 0.5 * 25.0    # 35 → 60
    elif rr < 2.5:
        return 60.0 + (rr - 2.0) / 0.5 * 20.0    # 60 → 80
    elif rr < 3.0:
        return 80.0 + (rr - 2.5) / 0.5 * 20.0    # 80 → 100
    else:
        return 100.0


# ── ScoredSetup ────────────────────────────────────────────────────────────────

@dataclass
class ScoredSetup:
    """A TradeSetup enriched with multi-factor scoring breakdown."""
    setup:                    TradeSetup
    composite_score:          float      # 0-100

    # Individual factor scores (each 0-100)
    pattern_confidence_score: float
    feature_score:            float
    strategy_score:           float
    regime_alignment_score:   float
    backtest_edge_score:      float
    volume_confirm_score:     float
    rr_quality_score:         float

    # Context
    regime:          Optional[str]       = None
    hot_strategies:  Optional[list[str]] = None

    def to_dict(self) -> dict:
        d = self.setup.to_dict()
        d["composite_score"] = round(self.composite_score, 1)
        d["scoring"] = {
            "pattern_confidence": round(self.pattern_confidence_score, 1),
            "feature_score":      round(self.feature_score, 1),
            "strategy_score":     round(self.strategy_score, 1),
            "regime_alignment":   round(self.regime_alignment_score, 1),
            "backtest_edge":      round(self.backtest_edge_score, 1),
            "volume_confirm":     round(self.volume_confirm_score, 1),
            "rr_quality":         round(self.rr_quality_score, 1),
            "weights":            WEIGHTS,
        }
        d["regime"] = self.regime
        return d


# ── Main scorer ────────────────────────────────────────────────────────────────

def score_setup(
    setup: TradeSetup,
    features: FeatureResult,
    regime: RegimeResult,
    evaluator: Optional[StrategyEvaluator] = None,
    backtest_score: float = 50.0,    # v2: neutral 50, not below-neutral 30
) -> ScoredSetup:
    """
    Score a single TradeSetup using all 7 factors.

    backtest_score: pass the pattern's known edge from the backtest cache
                    (0-100).  The default is 50 (neutral / no information)
                    so that an unproven pattern is not penalised just for
                    being new.  Patterns with a known negative edge will
                    have their backtest cache entry < 50 and be suppressed
                    naturally.
    """
    strategy_type = setup.strategy_type   # "breakout", "momentum", etc.

    # Factor 1: Pattern Confidence (0-100)
    pattern_conf = setup.confidence * 100.0

    # Factor 2: Feature Score — pattern-type-aware, volume deduplicated
    feat_score = _pattern_aware_feature_score(features, strategy_type)

    # Factor 3: Strategy Score — neutral 50 when no live data
    if evaluator is not None:
        strat_score = evaluator.get_strategy_score(setup.pattern_name)
    else:
        strat_score = 50.0

    # Factor 4: Regime Alignment
    st_enum    = _map_strategy_type(strategy_type)
    regime_align = get_regime_alignment(regime.regime, st_enum)

    # Factor 5: Backtest Edge
    bt_edge = backtest_score

    # Factor 6: Volume Confirmation (dedicated, no longer double-counted)
    vol_score = features.volume_expansion.score

    # Factor 7: R:R Quality
    rr_score = _rr_quality_score(setup)

    # Composite
    composite = (
        pattern_conf  * WEIGHTS["pattern_confidence"]
        + feat_score  * WEIGHTS["feature_score"]
        + strat_score * WEIGHTS["strategy_score"]
        + regime_align* WEIGHTS["regime_alignment"]
        + bt_edge     * WEIGHTS["backtest_edge"]
        + vol_score   * WEIGHTS["volume_confirm"]
        + rr_score    * WEIGHTS["rr_quality"]
    )
    composite = round(max(0.0, min(100.0, composite)), 1)

    return ScoredSetup(
        setup=setup,
        composite_score=composite,
        pattern_confidence_score=pattern_conf,
        feature_score=feat_score,
        strategy_score=strat_score,
        regime_alignment_score=regime_align,
        backtest_edge_score=bt_edge,
        volume_confirm_score=vol_score,
        rr_quality_score=rr_score,
        regime=regime.regime.value,
    )


def score_setups_batch(
    setups: list[TradeSetup],
    features: FeatureResult,
    regime: RegimeResult,
    evaluator: Optional[StrategyEvaluator] = None,
    backtest_scores: Optional[dict[str, float]] = None,
) -> list[ScoredSetup]:
    """
    Score multiple setups, return sorted by composite score descending.
    backtest_scores: {pattern_name: edge_score}.  Defaults to neutral 50.
    """
    if backtest_scores is None:
        backtest_scores = {}

    scored = []
    for setup in setups:
        bt = backtest_scores.get(setup.pattern_name, 50.0)   # v2: 50 not 30
        s  = score_setup(setup, features, regime, evaluator, bt)
        scored.append(s)

    scored.sort(key=lambda x: x.composite_score, reverse=True)
    return scored


# ── Helpers ────────────────────────────────────────────────────────────────────

def _map_strategy_type(type_str: str) -> StrategyType:
    mapping = {
        "momentum":       StrategyType.MOMENTUM,
        "mean_reversion": StrategyType.MEAN_REVERSION,
        "breakout":       StrategyType.BREAKOUT,
        "scalp":          StrategyType.SCALP,
    }
    return mapping.get(type_str, StrategyType.BREAKOUT)


def explain_score(scored: ScoredSetup) -> str:
    """Human-readable breakdown for the UI tooltip / detail panel."""
    lines = [
        f"Composite Score: {scored.composite_score:.0f}/100",
        "",
        f"  Pattern Confidence: {scored.pattern_confidence_score:.0f}/100  (×{WEIGHTS['pattern_confidence']:.0%})",
        f"  Feature Score:      {scored.feature_score:.0f}/100  (×{WEIGHTS['feature_score']:.0%})  [type-aware]",
        f"  Strategy Hot Score: {scored.strategy_score:.0f}/100  (×{WEIGHTS['strategy_score']:.0%})",
        f"  Regime Alignment:   {scored.regime_alignment_score:.0f}/100  (×{WEIGHTS['regime_alignment']:.0%})",
        f"  Backtest Edge:      {scored.backtest_edge_score:.0f}/100  (×{WEIGHTS['backtest_edge']:.0%})",
        f"  Volume Confirm:     {scored.volume_confirm_score:.0f}/100  (×{WEIGHTS['volume_confirm']:.0%})",
        f"  R:R Quality:        {scored.rr_quality_score:.0f}/100  (×{WEIGHTS['rr_quality']:.0%})",
    ]

    factors = [
        ("Pattern Confidence", scored.pattern_confidence_score),
        ("Feature Score",      scored.feature_score),
        ("Strategy Hot Score", scored.strategy_score),
        ("Regime Alignment",   scored.regime_alignment_score),
        ("Backtest Edge",      scored.backtest_edge_score),
        ("Volume Confirm",     scored.volume_confirm_score),
        ("R:R Quality",        scored.rr_quality_score),
    ]
    strengths  = [n for n, v in factors if v >= 70]
    weaknesses = [n for n, v in factors if v <= 30]

    if strengths:
        lines.append(f"\n  Strengths:  {', '.join(strengths)}")
    if weaknesses:
        lines.append(f"  Weaknesses: {', '.join(weaknesses)}")

    return "\n".join(lines)
