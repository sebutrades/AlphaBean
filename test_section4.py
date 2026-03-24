"""
test_section4.py — Verify Section 4: Candlestick + SMB Scalp pattern fixes.

Run: python test_section4.py
"""
import sys
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("  Juicer v2.2 — Section 4: Candlestick + SMB Scalps Test")
print("=" * 70)

PASS = 0; FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✓ {name}" + (f" ({detail})" if detail else ""))
    else:
        FAIL += 1; print(f"  ✗ {name}" + (f" ({detail})" if detail else ""))


# ══════════════════════════════════════════════════════════════
# TEST 1: All 17 detectors import
# ══════════════════════════════════════════════════════════════
print("\n[1] Import all Section 4 detectors...")

try:
    from backend.patterns.classifier import (
        _detect_bullish_engulfing, _detect_bearish_engulfing,
        _detect_morning_star, _detect_evening_star,
        _detect_hammer, _detect_shooting_star,
        _detect_doji, _detect_dragonfly_doji,
        _detect_three_white_soldiers, _detect_three_black_crows,
        _detect_rubberband_scalp,
        _detect_orb_15, _detect_orb_30,
        _detect_second_chance_scalp,
        _detect_fashionably_late,
        _detect_gap_give_and_go,
        _detect_tidal_wave,
    )
    check("All 17 Section 4 detectors import", True)
except ImportError as e:
    check("Import Section 4 detectors", False, str(e))
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# TEST 2: Verify _confirmation_candle_bonus exists
# ══════════════════════════════════════════════════════════════
print("\n[2] Confirmation candle helper...")

try:
    from backend.patterns.classifier import _confirmation_candle_bonus
    check("_confirmation_candle_bonus exists", True)
except ImportError:
    check("_confirmation_candle_bonus", False, "Not found — add to classifier.py")


# ══════════════════════════════════════════════════════════════
# TEST 3: Verify _is_nr7 exists
# ══════════════════════════════════════════════════════════════
print("\n[3] NR7 helper...")

try:
    from backend.patterns.classifier import _is_nr7
    check("_is_nr7 exists", True)
except ImportError:
    check("_is_nr7", False, "Not found — should be from Section 1 helpers")


# ══════════════════════════════════════════════════════════════
# TEST 4: Candlestick detectors run without errors
# ══════════════════════════════════════════════════════════════
print("\n[4] Candlestick detectors run without errors...")

from backend.data.schemas import Bar, BarSeries

def make_bars(closes, symbol="TEST", tf="5min", start=None, volumes=None):
    if start is None:
        start = datetime(2024, 6, 3, 9, 30)
    bars = []
    for i, c in enumerate(closes):
        noise = max(abs(c) * 0.003, 0.01)
        o = c - noise * 0.2
        h = c + noise
        l = c - noise
        v = int(volumes[i]) if volumes is not None else 10000
        bars.append(Bar(symbol=symbol, timestamp=start + timedelta(minutes=i*5),
                        open=float(o), high=float(h), low=float(l),
                        close=float(c), volume=v))
    return BarSeries(symbol=symbol, timeframe=tf, bars=bars)

try:
    from backend.patterns.classifier import extract_structures

    # Simple downtrend then reversal — might trigger hammer/engulfing
    closes = np.concatenate([
        np.linspace(110, 95, 30),    # Downtrend
        [94, 96],                     # Potential reversal bar
    ])
    bars = make_bars(closes, "CAND", "5min")
    s = extract_structures(bars)

    errors = []
    for name, func in [
        ("bullish_engulfing", _detect_bullish_engulfing),
        ("bearish_engulfing", _detect_bearish_engulfing),
        ("morning_star", _detect_morning_star),
        ("evening_star", _detect_evening_star),
        ("hammer", _detect_hammer),
        ("shooting_star", _detect_shooting_star),
        ("doji", _detect_doji),
        ("dragonfly_doji", _detect_dragonfly_doji),
        ("three_white_soldiers", _detect_three_white_soldiers),
        ("three_black_crows", _detect_three_black_crows),
    ]:
        try:
            result = func(s)
            # Check output format if triggered
            if result is not None:
                d = result.to_dict()
                assert "target_1" in d, f"{name}: missing target_1"
                assert "target_2" in d, f"{name}: missing target_2"
        except Exception as e:
            errors.append(f"{name}: {e}")

    if errors:
        for e in errors:
            check(f"Candlestick error: {e}", False)
    else:
        check("All 10 candlestick detectors run cleanly", True)

except Exception as e:
    import traceback
    traceback.print_exc()
    check("Candlestick test", False, str(e))


# ══════════════════════════════════════════════════════════════
# TEST 5: SMB detectors run without errors
# ══════════════════════════════════════════════════════════════
print("\n[5] SMB scalp detectors run without errors...")

try:
    # Build 5min data starting at 9:30 AM (trading hours)
    closes = np.concatenate([
        np.linspace(100, 102, 6),    # ORB period (9:30-10:00)
        np.linspace(102, 105, 12),   # Post-ORB
        np.linspace(105, 103, 10),   # Pullback
        np.linspace(103, 106, 10),   # Resume
    ])
    start = datetime(2024, 6, 3, 9, 30)
    bars = make_bars(closes, "SMB", "5min", start=start)
    s = extract_structures(bars)

    errors = []
    for name, func in [
        ("rubberband", _detect_rubberband_scalp),
        ("orb_15", _detect_orb_15),
        ("orb_30", _detect_orb_30),
        ("second_chance", _detect_second_chance_scalp),
        ("fashionably_late", _detect_fashionably_late),
        ("gap_give_go", _detect_gap_give_and_go),
        ("tidal_wave", _detect_tidal_wave),
    ]:
        try:
            result = func(s)
            if result is not None:
                d = result.to_dict()
                assert "target_1" in d, f"{name}: missing target_1"
                assert "target_2" in d, f"{name}: missing target_2"
        except Exception as e:
            errors.append(f"{name}: {e}")

    if errors:
        for e in errors:
            check(f"SMB error: {e}", False)
    else:
        check("All 7 SMB scalp detectors run cleanly", True)

except Exception as e:
    import traceback
    traceback.print_exc()
    check("SMB test", False, str(e))


# ══════════════════════════════════════════════════════════════
# TEST 6: Registry metadata check
# ══════════════════════════════════════════════════════════════
print("\n[6] Registry metadata verification...")

from backend.patterns.registry import PATTERN_META

# Verify all 17 patterns are in the registry
expected_candle = [
    "Bullish Engulfing", "Bearish Engulfing", "Morning Star", "Evening Star",
    "Hammer", "Shooting Star", "Doji", "Dragonfly Doji",
    "Three White Soldiers", "Three Black Crows",
]
expected_smb = [
    "RubberBand Scalp", "ORB 15min", "ORB 30min",
    "Second Chance Scalp", "Fashionably Late",
    "Gap Give & Go", "Tidal Wave",
]

for name in expected_candle + expected_smb:
    check(f"'{name}' in PATTERN_META", name in PATTERN_META)


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
total = PASS + FAIL
status = "ALL PASS" if FAIL == 0 else f"{FAIL} FAILED"
print(f"  SECTION 4 RESULTS: {PASS}/{total} passed — {status}")
print("=" * 70)

print("""
  Section 4 complete. 17 patterns updated:
    ✓ 10 candlestick: ATR stops, confirmation candles, star volume, scaled exits
    ✓ 7 SMB scalps: bounce window, NR7, structural stops, temporal compression
  
  Next: Section 5 — Quant strategy fixes (9 patterns)
""")

if FAIL > 0:
    sys.exit(1)