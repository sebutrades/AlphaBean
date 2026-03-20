"""
test_phase7_scanner.py — Phase 7a: Scanner Integration Smoke Test

This is the REAL test — it hits the Massive.com API with 10 symbols
and runs the full pipeline: fetch → structures → features → patterns
→ regime → scoring → ranked output.

Run: python test_phase7_scanner.py

Expected: ~2-5 minutes depending on API speed.
"""
import sys
import time
import json
from datetime import datetime
from pathlib import Path

print("=" * 70)
print("  AlphaBean v3.0 — Phase 7a: Scanner Integration Smoke Test")
print("  10 symbols × 2 timeframes (5m+15m) × 47 patterns")
print("=" * 70)

# ── Imports ─────────────────────────────────────────────────
print("\n[1/6] Importing modules...")
t0 = time.time()
try:
    from backend.scanner.engine import scan_symbol, scan_multiple
    from backend.data.massive_client import fetch_bars, SCANNER_TIMEFRAMES
    from backend.patterns.registry import get_all_pattern_names, PATTERN_META
    from backend.features.engine import compute_features
    from backend.regime.detector import detect_regime
    from backend.strategies.evaluator import StrategyEvaluator
    from backend.scoring.multi_factor import score_setup, WEIGHTS
    import numpy as np
    print(f"       All imports OK ({time.time()-t0:.1f}s)")
    print(f"       {len(get_all_pattern_names())} patterns registered")
    print(f"       Scanner timeframes: {SCANNER_TIMEFRAMES}")
except ImportError as e:
    print(f"       FAIL — {e}")
    sys.exit(1)

PASS = 0; FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ✓ {name}" + (f" ({detail})" if detail else ""))
    else: FAIL += 1; print(f"  ✗ {name}" + (f" ({detail})" if detail else ""))

# ── Test symbols ─────────────────────────────────────────────
SYMBOLS = ["AAPL", "NVDA", "TSLA", "MSFT", "META", "AMZN", "GOOGL", "AMD", "SPY", "QQQ"]

# ── Load evaluator ───────────────────────────────────────────
print("\n[2/6] Loading strategy evaluator...")
evaluator = StrategyEvaluator()
evaluator.load()
stats = evaluator.stats_summary()
print(f"       {stats['strategies_tracked']} strategies tracked, {stats['total_trades']} total trades")


# =====================================================================
# TEST 1: Single Symbol Fetch
# =====================================================================
print("\n[3/6] Testing single symbol data fetch...")
t1 = time.time()

try:
    bars_5m = fetch_bars("AAPL", "5min", days_back=3)
    bars_15m = fetch_bars("AAPL", "15min", days_back=3)
    print(f"       AAPL 5min:  {len(bars_5m.bars)} bars")
    print(f"       AAPL 15min: {len(bars_15m.bars)} bars")
    check("5min fetch returns bars", len(bars_5m.bars) > 50)
    check("15min fetch returns bars", len(bars_15m.bars) > 20)
    check("Bars have correct fields",
          bars_5m.bars[-1].close > 0 and bars_5m.bars[-1].volume > 0)
    print(f"       Latest: ${bars_5m.bars[-1].close:.2f} at {bars_5m.bars[-1].timestamp}")
    print(f"       Fetch time: {time.time()-t1:.1f}s")
except Exception as e:
    print(f"       FETCH ERROR: {e}")
    check("Data fetch", False, str(e))


# =====================================================================
# TEST 2: Single Symbol Full Scan
# =====================================================================
print("\n[4/6] Running full scan on AAPL (5m+15m, 47 patterns)...")
print(f"       {'-'*50}")
t2 = time.time()

try:
    setups = scan_symbol("AAPL", mode="active", evaluator=evaluator)
    elapsed = time.time() - t2
    print(f"       {'-'*50}")
    print(f"       Scan complete: {len(setups)} scored setups in {elapsed:.1f}s")

    check("scan_symbol returns list", isinstance(setups, list))

    if setups:
        best = setups[0]
        check("Setups have composite_score", "composite_score" in best)
        check("Setups have pattern_name", "pattern_name" in best)
        check("Setups have scoring breakdown", "scoring" in best)
        check("Setups sorted by score",
              all(setups[i].get("composite_score",0) >= setups[i+1].get("composite_score",0)
                  for i in range(len(setups)-1)) if len(setups) > 1 else True)

        print(f"\n       Top AAPL setups:")
        for i, s in enumerate(setups[:5]):
            tf = s.get("timeframe_detected", "?")
            cs = s.get("composite_score", 0)
            name = s.get("pattern_name", "?")
            bias = s.get("bias", "?")
            entry = s.get("entry_price", 0)
            rr = s.get("risk_reward_ratio", 0)
            print(f"         #{i+1} {name:<25} {bias:<6} ${entry:<8.2f} "
                  f"R:R={rr:.1f}  score={cs:.0f}  [{tf}]")
    else:
        print("       No setups found (normal if market just opened or flat day)")
        check("Scan completed without error", True)

except Exception as e:
    print(f"       SCAN ERROR: {e}")
    import traceback; traceback.print_exc()
    check("scan_symbol runs", False, str(e))


# =====================================================================
# TEST 3: Multi-Symbol Scan (10 symbols)
# =====================================================================
print(f"\n[5/6] Scanning {len(SYMBOLS)} symbols...")
print(f"       Symbols: {', '.join(SYMBOLS)}")
print(f"       Mode: active | Timeframes: 5min + 15min | Patterns: 47")
print(f"       {'='*55}")
t3 = time.time()

try:
    all_setups = scan_multiple(SYMBOLS, mode="active", evaluator=evaluator)
    elapsed = time.time() - t3
    print(f"\n       {'='*55}")
    print(f"       MULTI-SCAN COMPLETE: {len(all_setups)} total setups in {elapsed:.1f}s")
    print(f"       Avg: {elapsed/len(SYMBOLS):.1f}s per symbol")

    check("Multi-scan returns results", isinstance(all_setups, list))
    check("Results globally sorted",
          all(all_setups[i].get("composite_score",0) >= all_setups[i+1].get("composite_score",0)
              for i in range(len(all_setups)-1)) if len(all_setups) > 1 else True)

    # Count by symbol
    by_sym = {}
    for s in all_setups:
        sym = s.get("symbol", "?")
        by_sym[sym] = by_sym.get(sym, 0) + 1

    print(f"\n       Setups per symbol:")
    for sym in SYMBOLS:
        cnt = by_sym.get(sym, 0)
        bar = "█" * cnt + "░" * max(0, 10-cnt)
        print(f"         {sym:<6} {bar} {cnt}")

    # Count by category
    by_cat = {}
    for s in all_setups:
        cat = s.get("category", "?")
        by_cat[cat] = by_cat.get(cat, 0) + 1

    print(f"\n       Setups by category:")
    for cat, cnt in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"         {cat:<15} {cnt}")

    # Count multi-TF
    mtf = sum(1 for s in all_setups if s.get("multi_tf", False))
    print(f"\n       Multi-timeframe confirmations: {mtf}")

    # Show top 10 overall
    print(f"\n       TOP 10 SETUPS ACROSS ALL SYMBOLS:")
    print(f"       {'─'*65}")
    for i, s in enumerate(all_setups[:10]):
        sym = s.get("symbol", "?")
        name = s.get("pattern_name", "?")
        bias = s.get("bias", "?")
        cs = s.get("composite_score", 0)
        rr = s.get("risk_reward_ratio", 0)
        entry = s.get("entry_price", 0)
        tf = s.get("timeframe_detected", "?")
        mtf_tag = " ★" if s.get("multi_tf") else ""
        print(f"         #{i+1:<2} {sym:<6} {name:<25} {bias:<6} "
              f"${entry:<8.2f} R:R={rr:.1f}  score={cs:.0f} [{tf}]{mtf_tag}")

except Exception as e:
    print(f"\n       MULTI-SCAN ERROR: {e}")
    import traceback; traceback.print_exc()
    check("Multi-scan runs", False, str(e))


# =====================================================================
# TEST 4: Regime Check
# =====================================================================
print(f"\n[6/6] Checking market regime (SPY daily)...")
t4 = time.time()
try:
    spy_bars = fetch_bars("SPY", "1d", days_back=250)
    closes = np.array([b.close for b in spy_bars.bars], dtype=np.float64)
    highs = np.array([b.high for b in spy_bars.bars], dtype=np.float64)
    lows = np.array([b.low for b in spy_bars.bars], dtype=np.float64)
    regime = detect_regime(closes, highs, lows, is_spy=True)
    print(f"       SPY: {len(spy_bars.bars)} daily bars")
    print(f"       Current regime: {regime.regime.value}")
    print(f"       ATR ratio: {regime.atr_ratio:.3f}")
    print(f"       Trend: {regime.trend_direction}")
    check("Regime detection works", regime.regime is not None)
    print(f"       Regime check: {time.time()-t4:.1f}s")
except Exception as e:
    print(f"       REGIME ERROR: {e}")
    check("Regime detection", False, str(e))


# =====================================================================
# SUMMARY
# =====================================================================
total_time = time.time() - t0
print("\n" + "=" * 70)
total = PASS + FAIL
status = "ALL PASS" if FAIL == 0 else f"{FAIL} FAILED"
print(f"  PHASE 7a RESULTS: {PASS}/{total} passed — {status}")
print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
print("=" * 70)

print(f"""
  Phase 7a delivers:
    backend/scanner/__init__.py     — Scanner package
    backend/scanner/engine.py       — Orchestrator (the brain)
    backend/data/massive_client.py  — Updated data fetcher (5m+15m)
    backend/main.py                 — FastAPI v3.0 with all endpoints

  To start the server:
    uvicorn backend.main:app --reload --port 8000

  API endpoints:
    GET /api/health           → status + pattern count
    GET /api/patterns         → all 47 patterns with metadata
    GET /api/scan?symbol=AAPL → full scan (5m+15m, scored)
    GET /api/scan-multiple?symbols=AAPL,NVDA,TSLA
    GET /api/chart/AAPL       → candlestick data for frontend
    GET /api/regime           → current market regime
    GET /api/hot-strategies   → top performing strategies

  Next: Phase 7b — Frontend rewrite
""")