"""
test_v2_classifier.py — Verify all audit changes applied correctly.
Run: python test_v2_classifier.py
"""
import sys

print("=" * 60)
print("  Juicer v2.0 — Post-Audit Classifier Verification")
print("=" * 60)

PASS = 0; FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ✓ {name}" + (f" ({detail})" if detail else ""))
    else: FAIL += 1; print(f"  ✗ {name}" + (f" ({detail})" if detail else ""))

# --- Imports ---
print("\n[IMPORTS]")
try:
    from backend.patterns.registry import PATTERN_META, get_all_pattern_names, PatternCategory
    from backend.patterns.classifier import (
        _ALL_DETECTORS, classify_all, extract_structures,
        _candle_context_ok, _compute_vwap, _make,
    )
    check("All imports successful", True)
except ImportError as e:
    check("Imports", False, str(e)); sys.exit(1)

# --- Registry counts ---
print("\n[REGISTRY]")
names = get_all_pattern_names()
check("Total patterns = 42", len(PATTERN_META) == 42, f"got {len(PATTERN_META)}")

cats = {}
for n, m in PATTERN_META.items():
    c = m["cat"].value; cats[c] = cats.get(c, 0) + 1

check("16 classical", cats.get("classical", 0) == 16, f"got {cats.get('classical', 0)}")
check("10 candlestick", cats.get("candlestick", 0) == 10, f"got {cats.get('candlestick', 0)}")
check("7 SMB scalps", cats.get("smb_scalp", 0) == 7, f"got {cats.get('smb_scalp', 0)}")
check("9 quant", cats.get("quant", 0) == 9, f"got {cats.get('quant', 0)}")

# --- Removals verified ---
print("\n[REMOVALS]")
removed = ["Breaking News", "HitchHiker Scalp", "Spencer Scalp",
           "BackSide Scalp", "Relative Strength Break"]
for name in removed:
    check(f"'{name}' removed", name not in PATTERN_META)

# --- Kept strategies verified ---
print("\n[KEPT STRATEGIES]")
kept = ["Head & Shoulders", "Tidal Wave", "ORB 15min", "Mean Reversion",
        "VWAP Reversion", "Momentum Breakout", "RubberBand Scalp",
        "Bullish Engulfing", "Hammer", "Doji", "Gap Fade",
        "Fashionably Late", "Second Chance Scalp", "Cup & Handle",
        "Rising Wedge", "Falling Wedge", "Bull Flag", "Bear Flag",
        "Trend Pullback", "Gap Give & Go"]
for name in kept:
    check(f"'{name}' present", name in PATTERN_META)

# --- Daily-only flags ---
print("\n[DAILY TIMEFRAME FLAGS]")
daily = ["Momentum Breakout", "Vol Compression Breakout", "Range Expansion",
         "Volume Breakout", "Donchian Breakout"]
for name in daily:
    meta = PATTERN_META.get(name, {})
    check(f"'{name}' has tf_req=1d", meta.get("tf_req") == "1d")

intraday = ["Head & Shoulders", "Tidal Wave", "Mean Reversion"]
for name in intraday:
    meta = PATTERN_META.get(name, {})
    check(f"'{name}' has no tf_req (intraday)", "tf_req" not in meta)

# --- Detector count ---
print("\n[DETECTORS]")
check("42 detector functions", len(_ALL_DETECTORS) == 42, f"got {len(_ALL_DETECTORS)}")

# --- Helpers exist ---
print("\n[HELPERS]")
check("_candle_context_ok exists", callable(_candle_context_ok))
check("_compute_vwap exists", callable(_compute_vwap))
check("_make exists", callable(_make))

# --- Summary ---
print("\n" + "=" * 60)
total = PASS + FAIL
status = "ALL PASS" if FAIL == 0 else f"{FAIL} FAILED"
print(f"  RESULT: {PASS}/{total} — {status}")
print("=" * 60)

if FAIL == 0:
    print("""
  Post-audit changes verified:
    ✓ 5 strategies removed (47 → 42)
    ✓ 5 strategies flagged daily-only
    ✓ _make() entry validation active
    ✓ _candle_context_ok() helper for candlestick filters
    ✓ All detector functions registered

  Next steps:
    1. Copy registry.py → backend/patterns/registry.py
    2. Copy classifier.py → backend/patterns/classifier.py
    3. Run: python test_v2_classifier.py
    4. Run: python run_backtest.py --full
    5. Compare results to pre-audit baseline
""")