"""
scoring/multi_factor.py — Multi-Factor Trade Setup Scoring

Combines everything from Phases 1-5 into a single 0-100 composite score:

  Score = (
      pattern_confidence × 0.20      ← Phase 4 classifier
    + feature_score      × 0.25      ← Phase 2 feature engine
    + strategy_score     × 0.20      ← Phase 5 evaluator (hot score)
    + regime_alignment   × 0.15      ← Phase 3 regime detector
    + backtest_edge      × 0.10      ← from backtest cache
    + volume_confirm     × 0.10      ← RVOL at signal
  )

This is the single number shown for each trade setup in the UI.
Higher = more factors align = higher probability trade.
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


# ==============================================================================
# WEIGHTS (matching the approved proposal)
# ==============================================================================

WEIGHTS = {
    "pattern_confidence": 0.20,
    "feature_score":      0.25,
    "strategy_score":     0.20,
    "regime_alignment":   0.15,
    "backtest_edge":      0.10,
    "volume_confirm":     0.10,
}


# ==============================================================================
# SCORED SETUP
# ==============================================================================

@dataclass
class ScoredSetup:
    """A TradeSetup enriched with multi-factor scoring breakdown."""
    setup: TradeSetup
    composite_score: float          # 0-100 final score

    # Individual factor scores (each 0-100)
    pattern_confidence_score: float
    feature_score: float
    strategy_score: float
    regime_alignment_score: float
    backtest_edge_score: float
    volume_confirm_score: float

    # Context
    regime: Optional[str] = None
    hot_strategies: Optional[list[str]] = None

    def to_dict(self) -> dict:
        d = self.setup.to_dict()
        d["composite_score"] = round(self.composite_score, 1)
        d["scoring"] = {
            "pattern_confidence": round(self.pattern_confidence_score, 1),
            "feature_score": round(self.feature_score, 1),
            "strategy_score": round(self.strategy_score, 1),
            "regime_alignment": round(self.regime_alignment_score, 1),
            "backtest_edge": round(self.backtest_edge_score, 1),
            "volume_confirm": round(self.volume_confirm_score, 1),
            "weights": WEIGHTS,
        }
        d["regime"] = self.regime
        return d


# ==============================================================================
# MAIN SCORER
# ==============================================================================

def score_setup(
    setup: TradeSetup,
    features: FeatureResult,
    regime: RegimeResult,
    evaluator: Optional[StrategyEvaluator] = None,
    backtest_score: float = 30.0,
) -> ScoredSetup:
    """
    Score a single TradeSetup using all 6 factors.

    Args:
        setup: The detected trade setup from classifier
        features: Pre-computed features for this symbol
        regime: Current market regime
        evaluator: Strategy evaluator with rolling performance data
        backtest_score: Historical backtest edge (0-100). Default 30 (below-neutral)
            — requires evidence before a pattern earns a high composite score.

    Returns:
        ScoredSetup with composite score and full breakdown.
    """
    # --- Factor 1: Pattern Confidence (0-100) ---
    # setup.confidence is 0.0-1.0, scale to 0-100
    pattern_conf = setup.confidence * 100.0

    # --- Factor 2: Feature Score (0-100) ---
    # Composite of all 8 statistical features
    feat_score = features.composite_score

    # --- Factor 3: Strategy Score (0-100) ---
    # How well is this pattern performing recently?
    if evaluator is not None:
        strat_score = evaluator.get_strategy_score(setup.pattern_name)
    else:
        strat_score = 50.0  # Neutral if no evaluator

    # --- Factor 4: Regime Alignment (0-100) ---
    # How well does this setup's strategy type fit the current regime?
    strategy_type = _map_strategy_type(setup.strategy_type)
    regime_align = get_regime_alignment(regime.regime, strategy_type)

    # --- Factor 5: Backtest Edge (0-100) ---
    bt_edge = backtest_score

    # --- Factor 6: Volume Confirmation (0-100) ---
    # From the feature engine's volume expansion score
    vol_score = features.volume_expansion.score

    # --- Composite ---
    composite = (
        pattern_conf * WEIGHTS["pattern_confidence"]
        + feat_score * WEIGHTS["feature_score"]
        + strat_score * WEIGHTS["strategy_score"]
        + regime_align * WEIGHTS["regime_alignment"]
        + bt_edge * WEIGHTS["backtest_edge"]
        + vol_score * WEIGHTS["volume_confirm"]
    )
    composite = round(max(0, min(100, composite)), 1)

    return ScoredSetup(
        setup=setup,
        composite_score=composite,
        pattern_confidence_score=pattern_conf,
        feature_score=feat_score,
        strategy_score=strat_score,
        regime_alignment_score=regime_align,
        backtest_edge_score=bt_edge,
        volume_confirm_score=vol_score,
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
    Score multiple setups at once. Returns sorted by composite score.

    Args:
        setups: List of TradeSetups from classify_all()
        features: Pre-computed features for the symbol
        regime: Current market regime
        evaluator: Strategy evaluator (optional)
        backtest_scores: Dict of {pattern_name: score} from backtest cache

    Returns:
        List of ScoredSetup sorted by composite_score descending.
    """
    if backtest_scores is None:
        backtest_scores = {}

    scored = []
    for setup in setups:
        # Default 30: no-evidence patterns are below-neutral until backtest validates
        bt = backtest_scores.get(setup.pattern_name, 30.0)
        s = score_setup(setup, features, regime, evaluator, bt)
        scored.append(s)

    scored.sort(key=lambda x: x.composite_score, reverse=True)
    return scored


# ==============================================================================
# HELPERS
# ==============================================================================

def _map_strategy_type(type_str: str) -> StrategyType:
    """Map pattern's strategy_type string to the StrategyType enum."""
    mapping = {
        "momentum": StrategyType.MOMENTUM,
        "mean_reversion": StrategyType.MEAN_REVERSION,
        "breakout": StrategyType.BREAKOUT,
        "scalp": StrategyType.SCALP,
    }
    return mapping.get(type_str, StrategyType.BREAKOUT)


def explain_score(scored: ScoredSetup) -> str:
    """
    Human-readable explanation of why a setup scored the way it did.
    Useful for the UI tooltip or detail panel.
    """
    lines = [
        f"Composite Score: {scored.composite_score:.0f}/100",
        f"",
        f"  Pattern Confidence: {scored.pattern_confidence_score:.0f}/100 (×{WEIGHTS['pattern_confidence']:.0%})",
        f"  Feature Score:      {scored.feature_score:.0f}/100 (×{WEIGHTS['feature_score']:.0%})",
        f"  Strategy Hot Score: {scored.strategy_score:.0f}/100 (×{WEIGHTS['strategy_score']:.0%})",
        f"  Regime Alignment:   {scored.regime_alignment_score:.0f}/100 (×{WEIGHTS['regime_alignment']:.0%})",
        f"  Backtest Edge:      {scored.backtest_edge_score:.0f}/100 (×{WEIGHTS['backtest_edge']:.0%})",
        f"  Volume Confirm:     {scored.volume_confirm_score:.0f}/100 (×{WEIGHTS['volume_confirm']:.0%})",
    ]

    # Highlight strengths and weaknesses
    factors = [
        ("Pattern Confidence", scored.pattern_confidence_score),
        ("Feature Score", scored.feature_score),
        ("Strategy Hot Score", scored.strategy_score),
        ("Regime Alignment", scored.regime_alignment_score),
        ("Backtest Edge", scored.backtest_edge_score),
        ("Volume Confirm", scored.volume_confirm_score),
    ]
    strengths = [name for name, val in factors if val >= 70]
    weaknesses = [name for name, val in factors if val <= 30]

    if strengths:
        lines.append(f"\n  Strengths: {', '.join(strengths)}")
    if weaknesses:
        lines.append(f"  Weaknesses: {', '.join(weaknesses)}")

    return "\n".join(lines)