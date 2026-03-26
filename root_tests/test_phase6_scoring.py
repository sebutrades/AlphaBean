"""
test_phase6_scoring.py — Validates Phase 6: Multi-Factor Scoring

Run: python test_phase6_scoring.py

Tests:
  Part A: Imports + weights
  Part B: Score a single setup (hand-calculated verification)
  Part C: Batch scoring + ranking
  Part D: Regime alignment impact
  Part E: Strategy evaluator impact
  Part F: Score explanation
  Part G: Edge cases + full integration
"""
import sys
import numpy as np
from datetime import datetime

print("=" * 70)
print("  AlphaBean v3.0 — Phase 6: Multi-Factor Scoring Test")
print("=" * 70)

# ── Imports ─────────────────────────────────────────────────
print("\n[IMPORTS]")
try:
    from backend.scoring.multi_factor import (
        score_setup, score_setups_batch, ScoredSetup, explain_score, WEIGHTS,
    )
    from backend.patterns.registry import TradeSetup, Bias, PatternCategory, PATTERN_META
    from backend.features.engine import compute_features, FeatureResult
    from backend.regime.detector import (
        detect_regime, MarketRegime, StrategyType, get_regime_alignment,
    )
    from backend.strategies.evaluator import StrategyEvaluator, TradeOutcome
    print("  PASS — All imports successful")
except ImportError as e:
    print(f"  FAIL — {e}")
    sys.exit(1)

PASS = 0; FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  PASS — {name}" + (f" ({detail})" if detail else ""))
    else: FAIL += 1; print(f"  FAIL — {name}" + (f" ({detail})" if detail else ""))


# ── Build test fixtures ──────────────────────────────────────

# A mock TradeSetup
def make_setup(name="Bull Flag", bias=Bias.LONG, conf=0.65, stype="momentum"):
    return TradeSetup(
        pattern_name=name, category=PatternCategory.CLASSICAL,
        symbol="AAPL", bias=bias,
        entry_price=150.0, stop_loss=147.0, target_price=156.0,
        risk_reward_ratio=2.0, confidence=conf,
        detected_at=datetime(2024, 6, 15, 10, 30),
        description=f"Test {name}", strategy_type=stype,
        win_rate=0.65, timeframe_detected="15min",
    )

# Features from synthetic uptrend
np.random.seed(42)
n = 200
up_c = np.linspace(100, 150, n) + np.random.normal(0, 0.5, n)
up_h = up_c + np.abs(np.random.normal(0, 0.3, n)) + 0.5
up_l = up_c - np.abs(np.random.normal(0, 0.3, n)) - 0.5
up_v = np.random.randint(8000, 15000, n).astype(np.float64)
features = compute_features(up_c, up_h, up_l, up_v)

# Regime from same data
regime_bull = detect_regime(up_c, up_h, up_l, is_spy=False)

# Evaluator with some data
evaluator = StrategyEvaluator()
for i in range(20):
    evaluator.record_outcome(TradeOutcome(
        pattern_name="Bull Flag", strategy_type="momentum", symbol="AAPL",
        bias="long", entry_price=150, target_price=155, stop_price=148,
        outcome="win" if i % 10 < 7 else "loss",
        realized_r=2.0 if i % 10 < 7 else -1.0,
        timestamp=f"2024-06-{i+1:02d}T10:00:00",
    ))


# =====================================================================
# PART A: Weights
# =====================================================================
print("\n" + "=" * 70)
print("  PART A: Weights Configuration")
print("=" * 70)

check("Weights sum to 1.0", abs(sum(WEIGHTS.values()) - 1.0) < 0.001,
      f"sum={sum(WEIGHTS.values())}")
check("6 weight factors", len(WEIGHTS) == 6)
print(f"\n  Weights:")
for k, v in WEIGHTS.items():
    print(f"    {k:<22} {v:.0%}")


# =====================================================================
# PART B: Single Setup Scoring (hand-calculated)
# =====================================================================
print("\n" + "=" * 70)
print("  PART B: Single Setup Scoring (hand-calculated verification)")
print("=" * 70)

setup = make_setup("Bull Flag", Bias.LONG, conf=0.65, stype="momentum")
scored = score_setup(setup, features, regime_bull, evaluator, backtest_score=70.0)

check("ScoredSetup created", isinstance(scored, ScoredSetup))
check("Composite 0-100", 0 <= scored.composite_score <= 100,
      f"got {scored.composite_score}")

# Verify each factor
print(f"\n  Factor Breakdown:")
print(f"    Pattern Confidence: {scored.pattern_confidence_score:.1f}/100  (setup.confidence={setup.confidence})")
print(f"    Feature Score:      {scored.feature_score:.1f}/100  (features.composite)")
print(f"    Strategy Score:     {scored.strategy_score:.1f}/100  (evaluator hot score)")
print(f"    Regime Alignment:   {scored.regime_alignment_score:.1f}/100  (regime={scored.regime})")
print(f"    Backtest Edge:      {scored.backtest_edge_score:.1f}/100  (from cache)")
print(f"    Volume Confirm:     {scored.volume_confirm_score:.1f}/100  (RVOL)")
print(f"    ─────────────────────────────")
print(f"    COMPOSITE:          {scored.composite_score:.1f}/100")

# Hand-calculate expected composite
expected = (
    scored.pattern_confidence_score * 0.20
    + scored.feature_score * 0.25
    + scored.strategy_score * 0.20
    + scored.regime_alignment_score * 0.15
    + scored.backtest_edge_score * 0.10
    + scored.volume_confirm_score * 0.10
)
check("Composite matches hand calculation",
      abs(scored.composite_score - expected) < 0.5,
      f"expected {expected:.1f}, got {scored.composite_score:.1f}")

# Pattern confidence should be 0.65 * 100 = 65
check("Pattern confidence = 65.0", abs(scored.pattern_confidence_score - 65.0) < 0.1)

# Backtest edge should be what we passed in
check("Backtest edge = 70.0", abs(scored.backtest_edge_score - 70.0) < 0.1)

# Strategy score should come from evaluator (Bull Flag = ~87 hot score)
check("Strategy score from evaluator > 50", scored.strategy_score > 50,
      f"got {scored.strategy_score:.1f}")

# Regime alignment: momentum in bull should be high (90)
check("Regime alignment for momentum+bull = 90",
      scored.regime_alignment_score == 90.0,
      f"got {scored.regime_alignment_score}")


# =====================================================================
# PART C: Batch Scoring + Ranking
# =====================================================================
print("\n" + "=" * 70)
print("  PART C: Batch Scoring + Ranking")
print("=" * 70)

setups = [
    make_setup("Bull Flag", Bias.LONG, conf=0.75, stype="momentum"),
    make_setup("Mean Reversion", Bias.LONG, conf=0.55, stype="mean_reversion"),
    make_setup("Breaking News", Bias.LONG, conf=0.45, stype="scalp"),
]

scored_batch = score_setups_batch(
    setups, features, regime_bull, evaluator,
    backtest_scores={"Bull Flag": 80.0, "Mean Reversion": 60.0, "Breaking News": 40.0},
)

check("Batch returns 3 scored setups", len(scored_batch) == 3)
check("Sorted by composite score",
      scored_batch[0].composite_score >= scored_batch[1].composite_score >= scored_batch[2].composite_score)

print(f"\n  Batch Rankings:")
for i, s in enumerate(scored_batch):
    print(f"    #{i+1} {s.setup.pattern_name:<20} composite={s.composite_score:.1f}  "
          f"pattern={s.pattern_confidence_score:.0f} feat={s.feature_score:.0f} "
          f"strat={s.strategy_score:.0f} regime={s.regime_alignment_score:.0f}")

# Bull Flag should rank highest (high conf + momentum in bull regime)
check("Bull Flag ranks #1", scored_batch[0].setup.pattern_name == "Bull Flag")


# =====================================================================
# PART D: Regime Alignment Impact
# =====================================================================
print("\n" + "=" * 70)
print("  PART D: Regime Alignment Impact")
print("=" * 70)

# Score same setup in different regimes
setup_mr = make_setup("Mean Reversion", Bias.LONG, conf=0.60, stype="mean_reversion")

# In bull regime: mean_reversion alignment = 20
scored_mr_bull = score_setup(setup_mr, features, regime_bull, evaluator)
print(f"  Mean Reversion in BULL regime: alignment={scored_mr_bull.regime_alignment_score}")

# Create bear regime data
np.random.seed(99)
dn_c = np.linspace(150, 100, 200) + np.random.normal(0, 0.5, 200)
dn_h = dn_c + 0.5; dn_l = dn_c - 0.5
regime_bear = detect_regime(dn_c, dn_h, dn_l, is_spy=False)
feat_bear = compute_features(dn_c, dn_h, dn_l, up_v)

# Create range regime data
np.random.seed(55)
rng_c = 120 + np.random.normal(0, 1.0, 250)
rng_h = rng_c + 0.3; rng_l = rng_c - 0.3
regime_range = detect_regime(rng_c, rng_h, rng_l, is_spy=False)
feat_range = compute_features(rng_c, rng_h, rng_l, up_v[:250])

scored_mr_range = score_setup(setup_mr, feat_range, regime_range, evaluator)
print(f"  Mean Reversion in RANGE regime: alignment={scored_mr_range.regime_alignment_score}")

# Mean reversion should score MUCH better in range than bull
check("MR alignment higher in range vs bull",
      scored_mr_range.regime_alignment_score > scored_mr_bull.regime_alignment_score,
      f"range={scored_mr_range.regime_alignment_score} vs bull={scored_mr_bull.regime_alignment_score}")

# Check all regime × strategy type combinations produce valid scores
print(f"\n  Regime × Strategy alignment matrix:")
for regime in MarketRegime:
    for st in StrategyType:
        val = get_regime_alignment(regime, st)
        check(f"  {regime.value[:8]}×{st.value[:8]}",
              0 <= val <= 100, f"={val}")


# =====================================================================
# PART E: Strategy Evaluator Impact
# =====================================================================
print("\n" + "=" * 70)
print("  PART E: Strategy Evaluator Impact")
print("=" * 70)

# Score with evaluator vs without
scored_with = score_setup(setup, features, regime_bull, evaluator, 70.0)
scored_without = score_setup(setup, features, regime_bull, None, 70.0)

check("With evaluator: strategy_score > 50", scored_with.strategy_score > 50,
      f"got {scored_with.strategy_score:.1f}")
check("Without evaluator: strategy_score = 50", scored_without.strategy_score == 50.0)
check("With evaluator gives different composite",
      scored_with.composite_score != scored_without.composite_score,
      f"with={scored_with.composite_score:.1f}, without={scored_without.composite_score:.1f}")


# =====================================================================
# PART F: Score Explanation
# =====================================================================
print("\n" + "=" * 70)
print("  PART F: Score Explanation")
print("=" * 70)

explanation = explain_score(scored)
check("Explanation is string", isinstance(explanation, str))
check("Contains 'Composite Score'", "Composite Score" in explanation)
check("Contains factor lines", "Pattern Confidence" in explanation and "Regime Alignment" in explanation)
print(f"\n{explanation}")


# =====================================================================
# PART G: Edge Cases + to_dict
# =====================================================================
print("\n" + "=" * 70)
print("  PART G: Edge Cases + Serialization")
print("=" * 70)

# to_dict includes scoring breakdown
d = scored.to_dict()
check("to_dict has composite_score", "composite_score" in d)
check("to_dict has scoring breakdown", "scoring" in d and len(d["scoring"]) >= 6)
check("to_dict has regime", "regime" in d)

# Low confidence setup
low_setup = make_setup("Doji", Bias.LONG, conf=0.10, stype="mean_reversion")
low_scored = score_setup(low_setup, features, regime_bull, evaluator, 30.0)
check("Low confidence → low composite", low_scored.composite_score < 50,
      f"got {low_scored.composite_score:.1f}")

# Perfect setup
perf_setup = make_setup("Bull Flag", Bias.LONG, conf=0.95, stype="momentum")
perf_scored = score_setup(perf_setup, features, regime_bull, evaluator, 95.0)
check("High confidence → high composite", perf_scored.composite_score > 60,
      f"got {perf_scored.composite_score:.1f}")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
total = PASS + FAIL
status = "ALL PASS" if FAIL == 0 else f"{FAIL} FAILED"
print(f"  PHASE 6 RESULTS: {PASS}/{total} passed — {status}")
print("=" * 70)

print(f"""
  Files created:
    backend/scoring/__init__.py
    backend/scoring/multi_factor.py

  Scoring formula:
    Composite = pattern_conf×20% + features×25% + strategy×20%
              + regime×15% + backtest×10% + volume×10%

  Integration verified:
    ✓ Phase 2 features → feature_score
    ✓ Phase 3 regime → regime_alignment
    ✓ Phase 4 patterns → pattern_confidence
    ✓ Phase 5 evaluator → strategy_score

  Next: Phase 7 — Scanner Rewrite + API + Frontend
""")