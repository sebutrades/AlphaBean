"""
test_phase5_strategies.py — Validates Phase 5: Rolling Strategy Evaluator

Run: python test_phase5_strategies.py

Tests:
  Part A: Imports + data types
  Part B: Recording outcomes + metrics computation
  Part C: Hot score ranking
  Part D: Strategy type aggregation
  Part E: Persistence (save/load)
  Part F: Integration with Phase 4 patterns
  Part G: Edge cases
"""
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("  AlphaBean v3.0 — Phase 5: Rolling Strategy Evaluator Test")
print("=" * 70)

# ── Imports ─────────────────────────────────────────────────
print("\n[IMPORTS]")
try:
    from backend.strategies.evaluator import (
        StrategyEvaluator, StrategyMetrics, TradeOutcome,
        _compute_hot_score, MAX_HISTORY,
    )
    from backend.patterns.registry import PATTERN_META
    print("  PASS — All imports successful")
except ImportError as e:
    print(f"  FAIL — {e}")
    sys.exit(1)

PASS = 0; FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  PASS — {name}" + (f" ({detail})" if detail else ""))
    else: FAIL += 1; print(f"  FAIL — {name}" + (f" ({detail})" if detail else ""))


# =====================================================================
# PART A: Data Types
# =====================================================================
print("\n" + "=" * 70)
print("  PART A: Data Types")
print("=" * 70)

outcome = TradeOutcome(
    pattern_name="Bull Flag", strategy_type="momentum", symbol="AAPL",
    bias="long", entry_price=150.0, target_price=155.0, stop_price=148.0,
    outcome="win", realized_r=2.5, timestamp=datetime.now().isoformat(),
)
check("TradeOutcome creates", outcome.pattern_name == "Bull Flag")
d = outcome.to_dict()
check("to_dict works", "pattern" in d and "realized_r" in d)
restored = TradeOutcome.from_dict(d)
check("from_dict roundtrip", restored.pattern_name == "Bull Flag" and restored.realized_r == 2.5)


# =====================================================================
# PART B: Recording + Metrics
# =====================================================================
print("\n" + "=" * 70)
print("  PART B: Recording Outcomes + Metrics Computation")
print("=" * 70)

ev = StrategyEvaluator()

# Record 20 Bull Flag trades: 14 wins (70% WR), 6 losses
np.random.seed(42)
for i in range(20):
    is_win = i % 10 < 7  # 70% win rate
    ev.record_outcome(TradeOutcome(
        pattern_name="Bull Flag", strategy_type="momentum", symbol="AAPL",
        bias="long", entry_price=150.0, target_price=155.0, stop_price=148.0,
        outcome="win" if is_win else "loss",
        realized_r=2.0 if is_win else -1.0,
        timestamp=f"2024-06-{i+1:02d}T10:00:00",
    ))

# Record 15 Mean Reversion trades: 9 wins (60% WR)
for i in range(15):
    is_win = i % 5 < 3  # 60% win rate
    ev.record_outcome(TradeOutcome(
        pattern_name="Mean Reversion", strategy_type="mean_reversion", symbol="SPY",
        bias="long", entry_price=440.0, target_price=445.0, stop_price=437.0,
        outcome="win" if is_win else "loss",
        realized_r=1.5 if is_win else -1.0,
        timestamp=f"2024-06-{i+1:02d}T11:00:00",
    ))

# Record 10 Breaking News trades: 3 wins (30% WR) — cold strategy
for i in range(10):
    is_win = i < 3
    ev.record_outcome(TradeOutcome(
        pattern_name="Breaking News", strategy_type="scalp", symbol="TSLA",
        bias="long", entry_price=200.0, target_price=205.0, stop_price=197.0,
        outcome="win" if is_win else "loss",
        realized_r=1.5 if is_win else -1.0,
        timestamp=f"2024-06-{i+1:02d}T12:00:00",
    ))

stats = ev.stats_summary()
check("3 strategies tracked", stats["strategies_tracked"] == 3)
check("45 total trades", stats["total_trades"] == 45)

# Bull Flag metrics
bf = ev.compute_metrics("Bull Flag")
check("Bull Flag metrics exist", bf is not None)
check("BF win rate = 70%", abs(bf.win_rate - 0.70) < 0.01, f"got {bf.win_rate:.2%}")
check("BF total signals = 20", bf.total_signals == 20)
check("BF wins = 14", bf.wins == 14)
check("BF avg_r > 0", bf.avg_r > 0, f"avg_r={bf.avg_r:.3f}")

# Profit factor: gross_wins / gross_losses = (14*2.0) / (6*1.0) = 28/6 = 4.67
expected_pf = (14 * 2.0) / (6 * 1.0)
check("BF profit factor ≈ 4.67", abs(bf.profit_factor - expected_pf) < 0.1,
      f"expected {expected_pf:.2f}, got {bf.profit_factor:.2f}")

# Expectancy: (0.7 * 2.0) - (0.3 * 1.0) = 1.4 - 0.3 = 1.1
check("BF expectancy ≈ 1.1", abs(bf.expectancy - 1.1) < 0.1,
      f"expected 1.1, got {bf.expectancy:.3f}")

print(f"\n  Bull Flag full metrics:")
for k, v in bf.to_dict().items():
    print(f"    {k}: {v}")


# =====================================================================
# PART C: Rankings
# =====================================================================
print("\n" + "=" * 70)
print("  PART C: Hot Score Rankings")
print("=" * 70)

rankings = ev.get_rankings()
check("Rankings returns 3 strategies", len(rankings) == 3)
check("Sorted by hot_score", rankings[0].hot_score >= rankings[1].hot_score >= rankings[2].hot_score)

print(f"\n  Strategy Rankings:")
for i, m in enumerate(rankings):
    bar = "█" * int(m.hot_score / 5) + "░" * (20 - int(m.hot_score / 5))
    print(f"  #{i+1} {m.name:<20} {bar} {m.hot_score:.0f}  "
          f"WR={m.win_rate:.0%} PF={m.profit_factor:.1f} Exp={m.expectancy:+.2f}R")

check("Bull Flag is #1 (highest hot_score)", rankings[0].name == "Bull Flag",
      f"#1 is {rankings[0].name}")
check("Breaking News is last (worst performer)", rankings[-1].name == "Breaking News",
      f"last is {rankings[-1].name}")

# Hot strategies
hot = ev.get_hot_strategies(top_n=2)
check("get_hot_strategies returns top 2", len(hot) == 2)
check("Top hot strategy is Bull Flag", hot[0].name == "Bull Flag")

# Strategy score
bf_score = ev.get_strategy_score("Bull Flag")
bn_score = ev.get_strategy_score("Breaking News")
unk_score = ev.get_strategy_score("Nonexistent Pattern")
check("Bull Flag score > Breaking News score", bf_score > bn_score,
      f"BF={bf_score:.0f}, BN={bn_score:.0f}")
check("Unknown pattern returns 50 (neutral)", unk_score == 50.0)


# =====================================================================
# PART D: Strategy Type Aggregation
# =====================================================================
print("\n" + "=" * 70)
print("  PART D: Strategy Type Aggregation")
print("=" * 70)

hot_types = ev.get_hot_strategy_types(top_n=3)
check("Returns strategy types", len(hot_types) > 0 and isinstance(hot_types[0], str))
check("momentum is hot type (Bull Flag dominates)", "momentum" in hot_types,
      f"hot types: {hot_types}")

print(f"  Hot strategy types (ranked): {hot_types}")


# =====================================================================
# PART E: Persistence
# =====================================================================
print("\n" + "=" * 70)
print("  PART E: Persistence (save/load)")
print("=" * 70)

# Save
ev.save()
cache_path = Path("cache/strategy_performance.json")
check("Cache file created", cache_path.exists())

# Verify JSON structure
data = json.loads(cache_path.read_text())
check("JSON has Bull Flag", "Bull Flag" in data)
check("Bull Flag has 20 trades in cache", len(data["Bull Flag"]) == 20)

# Load into fresh evaluator
ev2 = StrategyEvaluator()
ev2.load()
check("Loaded 3 strategies", len(ev2.outcomes) == 3)

bf2 = ev2.compute_metrics("Bull Flag")
check("Loaded metrics match", bf2 is not None and bf2.wins == bf.wins,
      f"wins={bf2.wins if bf2 else 'None'}")

# Roundtrip: rankings should be identical
r2 = ev2.get_rankings()
check("Rankings preserved after reload",
      r2[0].name == rankings[0].name and abs(r2[0].hot_score - rankings[0].hot_score) < 0.1)


# =====================================================================
# PART F: Integration with Pattern Registry
# =====================================================================
print("\n" + "=" * 70)
print("  PART F: Integration with Pattern Registry")
print("=" * 70)

# Every pattern in registry can be scored
scored = 0
for name in PATTERN_META:
    s = ev.get_strategy_score(name)
    if s != 50.0:
        scored += 1

print(f"  {scored}/{len(PATTERN_META)} patterns have performance data")
check("Patterns with data return real scores", scored >= 3)

# get_pattern_summary
summary = ev.get_pattern_summary("Bull Flag")
check("Pattern summary has data", summary.get("has_data") == True)
check("Summary has hot_score", "hot_score" in summary)

no_data = ev.get_pattern_summary("Cup & Handle")
check("No-data pattern returns has_data=False", no_data.get("has_data") == False)


# =====================================================================
# PART G: Edge Cases
# =====================================================================
print("\n" + "=" * 70)
print("  PART G: Edge Cases")
print("=" * 70)

# Max history enforcement
ev3 = StrategyEvaluator()
for i in range(100):
    ev3.record_outcome(TradeOutcome(
        pattern_name="Overflow", strategy_type="test", symbol="X",
        bias="long", entry_price=100, target_price=105, stop_price=98,
        outcome="win", realized_r=1.0, timestamp=f"2024-01-{(i%28)+1:02d}",
    ))
check(f"Max history enforced ({MAX_HISTORY})",
      len(ev3.outcomes["Overflow"]) == MAX_HISTORY,
      f"got {len(ev3.outcomes['Overflow'])}")

# Empty evaluator
ev4 = StrategyEvaluator()
check("Empty rankings = []", ev4.get_rankings() == [])
check("Empty hot strategies = []", ev4.get_hot_strategies() == [])
check("Empty score = 50", ev4.get_strategy_score("Anything") == 50.0)

# All losses
ev5 = StrategyEvaluator()
for i in range(10):
    ev5.record_outcome(TradeOutcome(
        pattern_name="BadStrat", strategy_type="test", symbol="X",
        bias="long", entry_price=100, target_price=105, stop_price=98,
        outcome="loss", realized_r=-1.0, timestamp=f"2024-01-{i+1:02d}",
    ))
bad = ev5.compute_metrics("BadStrat")
check("All-loss strategy: WR = 0%", bad.win_rate == 0.0)
check("All-loss strategy: hot_score < 20", bad.hot_score < 20, f"got {bad.hot_score}")
check("All-loss strategy: negative streak", bad.recent_streak < 0,
      f"streak={bad.recent_streak}")

# All wins
ev6 = StrategyEvaluator()
for i in range(10):
    ev6.record_outcome(TradeOutcome(
        pattern_name="PerfectStrat", strategy_type="test", symbol="X",
        bias="long", entry_price=100, target_price=110, stop_price=98,
        outcome="win", realized_r=5.0, timestamp=f"2024-01-{i+1:02d}",
    ))
perf = ev6.compute_metrics("PerfectStrat")
check("All-win strategy: WR = 100%", perf.win_rate == 1.0)
check("All-win strategy: hot_score > 80", perf.hot_score > 80, f"got {perf.hot_score}")
check("All-win strategy: positive streak", perf.recent_streak > 0)

# Hot score formula verification
hs = _compute_hot_score(win_rate=0.70, profit_factor=4.67, expectancy=1.1,
                        sample_size=20, streak=3)
check("Hot score for BF-like metrics > 70", hs > 70, f"got {hs}")

hs_bad = _compute_hot_score(win_rate=0.30, profit_factor=0.5, expectancy=-0.5,
                            sample_size=10, streak=-3)
check("Hot score for bad metrics < 30", hs_bad < 30, f"got {hs_bad}")


# Clean up test cache
try:
    cache_path.unlink(missing_ok=True)
except:
    pass


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
total = PASS + FAIL
status = "ALL PASS" if FAIL == 0 else f"{FAIL} FAILED"
print(f"  PHASE 5 RESULTS: {PASS}/{total} passed — {status}")
print("=" * 70)

print(f"""
  Files created:
    backend/strategies/__init__.py
    backend/strategies/evaluator.py

  What it does:
    - Tracks last {MAX_HISTORY} outcomes per strategy/pattern
    - Computes rolling: win rate, profit factor, expectancy, streak
    - Ranks strategies by "hot score" (0-100 composite)
    - get_hot_strategies() → top N performers for UI
    - get_strategy_score() → feeds into multi-factor scorer (Phase 6)
    - Persists to cache/strategy_performance.json

  Next: Phase 6 — Multi-Factor Scoring System
    backend/scoring/multi_factor.py
""")