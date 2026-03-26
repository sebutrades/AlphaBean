"""
test_phase1_phase2.py — Verify all post-backtest fixes before re-running.

Checks:
  Phase 1:
    Fix 1: _make() min R:R raised to 1.5
    Fix 2: Classical stop buffers at 0.10 ATR (not 0.20)
    Fix 3: Classical T1 at 75% measured move (not 50%)
    Fix 4: Pennant cut (returns None or removed from dispatch)
    Fix 5: Candlestick patterns cut from dispatch
  
  Phase 2:
    Fix 6:  Trend Pullback — 0.3 ATR proximity, pullback depth, red/green checks
    Fix 7:  Flags — pole 15 bars, 2.0 ATR, 50% retrace, 0.10 stop
    Fix 8:  Second Chance — 0.3 ATR tolerance, 0.10 stop
    Fix 9:  RubberBand — bounce volume filter, 0.10 stop
    Fix 10: Donchian — squeeze filter
    Fix 11: Fashionably Late — slope ≥ 0.1 ATR, 0.10 stop, T1 at 75%

Run: python test_phase1_phase2.py
"""
import sys
import inspect
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("  Phase 1 + Phase 2 Fix Verification")
print("=" * 70)

PASS = 0; FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✓ {name}" + (f" ({detail})" if detail else ""))
    else:
        FAIL += 1; print(f"  ✗ {name}" + (f" ({detail})" if detail else ""))


# ══════════════════════════════════════════════════════════════
# PHASE 1, FIX 1: _make() min R:R = 1.5
# ══════════════════════════════════════════════════════════════
print("\n[Fix 1] _make() min R:R = 1.5...")

try:
    from backend.patterns.classifier import _make
    src = inspect.getsource(_make)
    
    # Should contain "rr < 1.5" not "rr < 0.5"
    has_15 = "rr < 1.5" in src or "rr< 1.5" in src or "rr <1.5" in src
    has_05 = "rr < 0.5" in src or "rr< 0.5" in src or "rr <0.5" in src
    
    check("_make() rejects R:R < 1.5", has_15, "Found 'rr < 1.5'")
    check("_make() does NOT have old 0.5 threshold", not has_05,
          "STILL HAS 'rr < 0.5' — needs update" if has_05 else "Clean")
except Exception as e:
    check("_make() import", False, str(e))


# ══════════════════════════════════════════════════════════════
# PHASE 1, FIX 2: Classical stop buffers at 0.10 ATR
# ══════════════════════════════════════════════════════════════
print("\n[Fix 2] Classical stop buffers (0.10 not 0.20)...")

try:
    from backend.patterns.classifier import (
        _detect_head_and_shoulders, _detect_inverse_hs,
        _detect_double_top, _detect_double_bottom,
        _detect_triple_top, _detect_triple_bottom,
        _detect_ascending_triangle, _detect_descending_triangle,
        _detect_symmetrical_triangle,
        _detect_cup_and_handle, _detect_rectangle,
        _detect_rising_wedge, _detect_falling_wedge,
    )
    
    classical_funcs = {
        "H&S": _detect_head_and_shoulders,
        "Inv H&S": _detect_inverse_hs,
        "Double Top": _detect_double_top,
        "Double Bottom": _detect_double_bottom,
        "Triple Top": _detect_triple_top,
        "Triple Bottom": _detect_triple_bottom,
        "Asc Triangle": _detect_ascending_triangle,
        "Desc Triangle": _detect_descending_triangle,
        "Sym Triangle": _detect_symmetrical_triangle,
        "Cup & Handle": _detect_cup_and_handle,
        "Rectangle": _detect_rectangle,
        "Rising Wedge": _detect_rising_wedge,
        "Falling Wedge": _detect_falling_wedge,
    }
    
    for name, func in classical_funcs.items():
        src = inspect.getsource(func)
        # Count occurrences of 0.20 in stop-related atr_offset calls
        has_020_stop = "atr_offset(atr, 0.20)" in src or "atr_offset(atr, 0.2)" in src
        check(f"{name}: no 0.20 ATR stop buffer", not has_020_stop,
              "STILL HAS 0.20 — change stop lines to 0.10" if has_020_stop else "Clean")

except Exception as e:
    check("Classical pattern imports", False, str(e))


# ══════════════════════════════════════════════════════════════
# PHASE 1, FIX 3: T1 at 75% measured move
# ══════════════════════════════════════════════════════════════
print("\n[Fix 3] T1 at 75% measured move...")

try:
    patterns_with_measured_move = {
        "H&S": _detect_head_and_shoulders,
        "Inv H&S": _detect_inverse_hs,
        "Double Top": _detect_double_top,
        "Double Bottom": _detect_double_bottom,
        "Triple Top": _detect_triple_top,
        "Triple Bottom": _detect_triple_bottom,
        "Asc Triangle": _detect_ascending_triangle,
        "Desc Triangle": _detect_descending_triangle,
        "Cup & Handle": _detect_cup_and_handle,
        "Rectangle": _detect_rectangle,
        "Rising Wedge": _detect_rising_wedge,
        "Falling Wedge": _detect_falling_wedge,
    }
    
    for name, func in patterns_with_measured_move.items():
        src = inspect.getsource(func)
        has_075 = "* 0.75" in src or "*0.75" in src
        has_050_t1 = "target_half" in src  # Old variable name suggests 50%
        
        check(f"{name}: uses 0.75 for T1", has_075,
              "Has 0.75" if has_075 else "MISSING — still at 50%?")

except Exception as e:
    check("T1 check", False, str(e))


# ══════════════════════════════════════════════════════════════
# PHASE 1, FIX 4: Pennant cut
# ══════════════════════════════════════════════════════════════
print("\n[Fix 4] Pennant removed...")

try:
    from backend.patterns.classifier import classify_all, extract_structures
    from backend.data.schemas import Bar, BarSeries
    
    # Check if _detect_pennant returns None immediately or doesn't exist in dispatch
    try:
        from backend.patterns.classifier import _detect_pennant
        src = inspect.getsource(_detect_pennant)
        # Should have "return None" very early (first few lines after def)
        lines = src.strip().split('\n')
        early_return = any("return None" in line and i < 5 for i, line in enumerate(lines))
        check("Pennant: returns None immediately", early_return,
              "Early return found" if early_return else "May still be active — check classify_all dispatch")
    except ImportError:
        check("Pennant: removed from imports", True, "Function not importable = cut")
        
except Exception as e:
    check("Pennant check", False, str(e))


# ══════════════════════════════════════════════════════════════
# PHASE 1, FIX 5: Candlestick patterns cut
# ══════════════════════════════════════════════════════════════
print("\n[Fix 5] Candlestick standalone patterns cut...")

try:
    src_classify = inspect.getsource(classify_all)
    
    candle_names = [
        "bullish_engulfing", "bearish_engulfing",
        "morning_star", "evening_star",
        "hammer", "shooting_star",
        "doji", "dragonfly_doji",
        "three_white_soldiers", "three_black_crows",
    ]
    
    active_candles = []
    for name in candle_names:
        # Check if the function is called in classify_all (not commented out)
        # Look for the function name without a leading #
        in_dispatch = False
        for line in src_classify.split('\n'):
            stripped = line.strip()
            if name in stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
                in_dispatch = True
                break
        if in_dispatch:
            active_candles.append(name)
    
    if active_candles:
        check(f"Candlestick patterns cut from dispatch", False,
              f"Still active: {', '.join(active_candles)}")
    else:
        check("All 10 candlestick patterns cut from dispatch", True)

except Exception as e:
    check("Candlestick cut check", False, str(e))


# ══════════════════════════════════════════════════════════════
# PHASE 2, FIX 6: Trend Pullback
# ══════════════════════════════════════════════════════════════
print("\n[Fix 6] Trend Pullback...")

try:
    from backend.patterns.classifier import _detect_trend_pullback
    src = inspect.getsource(_detect_trend_pullback)
    
    check("EMA proximity: 0.3 ATR", "atr * 0.3" in src,
          "Found" if "atr * 0.3" in src else "MISSING")
    check("Pullback depth check (atr * 1.0)", "atr * 1.0" in src,
          "Found" if "atr * 1.0" in src else "MISSING")
    check("Red bar check (recent_reds)", "recent_reds" in src,
          "Found" if "recent_reds" in src else "MISSING")
    check("Green bar check for SHORT (recent_greens)", "recent_greens" in src,
          "Found" if "recent_greens" in src else "MISSING — SHORT side needs this")
    check("Stop at 0.10 ATR", "atr, 0.10)" in src or "atr, 0.1)" in src,
          "Found" if ("atr, 0.10)" in src or "atr, 0.1)" in src) else "MISSING")
    check("No 0.20 ATR stop", "atr, 0.20)" not in src and "atr, 0.2)" not in src,
          "Clean" if ("atr, 0.20)" not in src and "atr, 0.2)" not in src) else "STILL HAS 0.20")
    check("SHORT pullback depth (for bear side)",
          "pullback_depth" in src.split("trending_bear")[1] if "trending_bear" in src else False,
          "Found in bear block" if "trending_bear" in src else "MISSING")

except Exception as e:
    check("Trend Pullback check", False, str(e))


# ══════════════════════════════════════════════════════════════
# PHASE 2, FIX 7: Flags
# ══════════════════════════════════════════════════════════════
print("\n[Fix 7] Bull Flag / Bear Flag...")

try:
    from backend.patterns.classifier import _detect_bull_flag, _detect_bear_flag
    
    for name, func in [("Bull Flag", _detect_bull_flag), ("Bear Flag", _detect_bear_flag)]:
        src = inspect.getsource(func)
        
        check(f"{name}: pole min 2.0 ATR", "atr * 2.0" in src or "atr * 2" in src,
              "Found" if ("atr * 2.0" in src or "atr * 2" in src) else "STILL AT 1.5")
        check(f"{name}: pole max 15 bars", "> 15" in src,
              "Found" if "> 15" in src else "STILL AT 10")
        check(f"{name}: retrace 50%", "> 0.50" in src or "> 0.5" in src,
              "Found" if ("> 0.50" in src or "> 0.5" in src) else "STILL AT 0.40")
        check(f"{name}: stop 0.10 ATR", "atr, 0.10)" in src or "atr, 0.1)" in src,
              "Found" if ("atr, 0.10)" in src or "atr, 0.1)" in src) else "STILL AT 0.15")
        check(f"{name}: T1 at 75%", "* 0.75" in src,
              "Found" if "* 0.75" in src else "MISSING")

except Exception as e:
    check("Flag check", False, str(e))


# ══════════════════════════════════════════════════════════════
# PHASE 2, FIX 8: Second Chance Scalp
# ══════════════════════════════════════════════════════════════
print("\n[Fix 8] Second Chance Scalp...")

try:
    from backend.patterns.classifier import _detect_second_chance_scalp
    src = inspect.getsource(_detect_second_chance_scalp)
    
    check("Tolerance: 0.3 ATR (not 0.5)", "atr * 0.3" in src,
          "Found" if "atr * 0.3" in src else "MISSING — may still be 0.5")
    check("Stop at 0.10 ATR", "atr, 0.10)" in src or "atr, 0.1)" in src,
          "Found" if ("atr, 0.10)" in src or "atr, 0.1)" in src) else "MISSING")

except Exception as e:
    check("Second Chance check", False, str(e))


# ══════════════════════════════════════════════════════════════
# PHASE 2, FIX 9: RubberBand Scalp
# ══════════════════════════════════════════════════════════════
print("\n[Fix 9] RubberBand Scalp...")

try:
    from backend.patterns.classifier import _detect_rubberband_scalp
    src = inspect.getsource(_detect_rubberband_scalp)
    
    # VWAP check at bounce should be removed
    has_vwap_bounce = "close < vwap" in src and "bounce" in src.split("close < vwap")[0][-100:]
    check("VWAP-at-bounce check removed", "close < vwap" not in src,
          "Removed" if "close < vwap" not in src else "STILL PRESENT — remove it")
    
    # Bounce volume filter should exist
    check("Bounce volume filter (1.3x)", "1.3" in src and "prior_avg_vol" in src,
          "Found" if ("1.3" in src and "prior_avg_vol" in src) else "MISSING")
    
    check("Stop at 0.10 ATR", "atr, 0.10)" in src or "atr, 0.1)" in src,
          "Found" if ("atr, 0.10)" in src or "atr, 0.1)" in src) else "STILL AT 0.20")

except Exception as e:
    check("RubberBand check", False, str(e))


# ══════════════════════════════════════════════════════════════
# PHASE 2, FIX 10: Donchian Breakout
# ══════════════════════════════════════════════════════════════
print("\n[Fix 10] Donchian Breakout...")

try:
    from backend.patterns.classifier import _detect_donchian_breakout
    src = inspect.getsource(_detect_donchian_breakout)
    
    check("Squeeze filter (0.85 compression)", "0.85" in src,
          "Found" if "0.85" in src else "MISSING")
    check("Recent ranges vs longer ranges", "recent_ranges" in src or "avg_recent" in src,
          "Found" if ("recent_ranges" in src or "avg_recent" in src) else "MISSING")

except Exception as e:
    check("Donchian check", False, str(e))


# ══════════════════════════════════════════════════════════════
# PHASE 2, FIX 11: Fashionably Late
# ══════════════════════════════════════════════════════════════
print("\n[Fix 11] Fashionably Late...")

try:
    from backend.patterns.classifier import _detect_fashionably_late
    src = inspect.getsource(_detect_fashionably_late)
    
    check("EMA slope ≥ 0.1 ATR", "atr * 0.1" in src,
          "Found" if "atr * 0.1" in src else "MISSING — still just > 0?")
    check("Stop at 0.10 ATR", "atr, 0.10)" in src or "atr, 0.1)" in src,
          "Found" if ("atr, 0.10)" in src or "atr, 0.1)" in src) else "STILL AT 0.20")
    check("T1 at 75% (mm * 0.75)", "mm * 0.75" in src,
          "Found" if "mm * 0.75" in src else "STILL AT 50%")

except Exception as e:
    check("Fashionably Late check", False, str(e))


# ══════════════════════════════════════════════════════════════
# BONUS: Verify backtest engine is still wired
# ══════════════════════════════════════════════════════════════
print("\n[Bonus] Backtest engine still wired...")

try:
    from run_backtest import PendingTrade, _compute_atr_at, _compute_pattern_stats
    
    # Quick smoke test
    pt = PendingTrade("Test", "breakout", "X", "long",
                      entry=100, target=110, stop=95,
                      bar_idx=0, max_hold=100, atr=2.0)
    check("PendingTrade creates OK", pt.entry == 100.10)
    check("check_resolution exists", hasattr(pt, 'check_resolution'))
    check("timeout_resolve exists", hasattr(pt, 'timeout_resolve'))
    
    # Verify partial_win in stats
    from backend.strategies.evaluator import TradeOutcome
    trades = [
        TradeOutcome("T", "b", "X", "long", 100, 110, 95, "win", 2.0, "2024-01-01"),
        TradeOutcome("T", "b", "X", "long", 100, 110, 95, "partial_win", 0.5, "2024-01-02"),
    ]
    stats = _compute_pattern_stats("T", trades)
    check("Stats count partial_win", stats["partial_wins"] == 1)

except Exception as e:
    check("Backtest engine check", False, str(e))


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
total = PASS + FAIL
status = "ALL PASS ✓ — Ready for backtest" if FAIL == 0 else f"{FAIL} FAILED — Fix before running"
print(f"  VERIFICATION: {PASS}/{total} — {status}")
print("=" * 70)

if FAIL == 0:
    print("""
  All fixes verified. To re-run the backtest:
  
    1. Delete old cache:
       del cache\\backtest_checkpoint.json
       del cache\\backtest_results.json
       del cache\\strategy_performance.json
    
    2. Run:
       python run_backtest.py --from-cache --days 90 --daily --resume
""")
else:
    print(f"\n  Fix the {FAIL} failure(s) above, then re-run this test.\n")
    sys.exit(1)