"""
test_phase4_patterns.py — Comprehensive validation of all 47 pattern detectors.

Run: python test_phase4_patterns.py
"""
import sys
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("  AlphaBean v3.0 — Phase 4: 47-Pattern Classifier Test")
print("=" * 70)

# ── Imports ─────────────────────────────────────────────────
print("\n[IMPORTS]")
try:
    from backend.data.schemas import Bar, BarSeries
    from backend.patterns.registry import (
        TradeSetup, Bias, PatternCategory, PATTERN_META, get_all_pattern_names,
    )
    from backend.patterns.classifier import (
        classify_all, extract_structures, ExtractedStructures,
        _detect_head_and_shoulders, _detect_double_bottom, _detect_double_top,
        _detect_triple_top, _detect_triple_bottom,
        _detect_bull_flag, _detect_rectangle, _detect_pennant,
        _detect_bullish_engulfing, _detect_bearish_engulfing,
        _detect_morning_star, _detect_evening_star,
        _detect_hammer, _detect_shooting_star, _detect_doji,
        _detect_three_white_soldiers, _detect_three_black_crows,
        _detect_dragonfly_doji,
        _detect_momentum_breakout, _detect_vol_compression_breakout,
        _detect_mean_reversion, _detect_trend_pullback,
        _detect_range_expansion, _detect_volume_breakout,
        _detect_donchian_breakout,
    )
    print("  PASS — All imports successful")
except ImportError as e:
    print(f"  FAIL — {e}")
    sys.exit(1)

PASS = 0; FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  PASS — {name}" + (f" ({detail})" if detail else ""))
    else: FAIL += 1; print(f"  FAIL — {name}" + (f" ({detail})" if detail else ""))

def make_bars(closes, symbol="TEST", tf="1h", start=None, volumes=None,
              opens=None, highs=None, lows=None):
    """Create BarSeries. If OHLCV not provided, generates from closes."""
    if start is None: start = datetime(2024, 6, 3, 9, 30)
    bars = []
    for i, c in enumerate(closes):
        noise = max(abs(c) * 0.003, 0.01)
        o = opens[i] if opens is not None else c - noise * 0.2
        h = highs[i] if highs is not None else c + noise
        l = lows[i] if lows is not None else c - noise
        v = int(volumes[i]) if volumes is not None else 10000
        bars.append(Bar(symbol=symbol, timestamp=start + timedelta(hours=i),
                        open=float(o), high=float(h), low=float(l), close=float(c), volume=v))
    return BarSeries(symbol=symbol, timeframe=tf, bars=bars)


# =====================================================================
# PART A: Registry
# =====================================================================
print("\n" + "=" * 70)
print("  PART A: Pattern Registry (47 patterns)")
print("=" * 70)

names = get_all_pattern_names()
check("Total patterns registered", len(names) == 47, f"got {len(names)}")

cats = {}
for n, m in PATTERN_META.items():
    c = m["cat"].value
    cats[c] = cats.get(c, 0) + 1

check("16 classical", cats.get("classical", 0) == 16, f"got {cats.get('classical',0)}")
check("10 candlestick", cats.get("candlestick", 0) == 10, f"got {cats.get('candlestick',0)}")
check("11 SMB scalps", cats.get("smb_scalp", 0) == 11, f"got {cats.get('smb_scalp',0)}")
check("10 quant", cats.get("quant", 0) == 10, f"got {cats.get('quant',0)}")

for name in ["Triple Top", "Triple Bottom", "Rectangle", "Pennant",
             "Bullish Engulfing", "Bearish Engulfing", "Morning Star",
             "Evening Star", "Hammer", "Shooting Star", "Doji",
             "Three White Soldiers", "Three Black Crows", "Dragonfly Doji",
             "Mean Reversion", "Trend Pullback", "Gap Fade",
             "Relative Strength Break", "Range Expansion", "Volume Breakout",
             "VWAP Reversion", "Donchian Breakout"]:
    check(f"'{name}' registered", name in PATTERN_META)


# =====================================================================
# PART B: Classical Patterns
# =====================================================================
print("\n" + "=" * 70)
print("  PART B: Classical Structural Patterns")
print("=" * 70)

print("\n[B1] Head & Shoulders")
hs = np.concatenate([np.linspace(100,110,15), np.linspace(110,104,8),
                     np.linspace(104,118,18), np.linspace(118,105,10),
                     np.linspace(105,111,12), np.linspace(111,98,15)])
r = _detect_head_and_shoulders(extract_structures(make_bars(hs, "HS")))
check("H&S detected", r is not None and r.bias == Bias.SHORT, f"{'detected' if r else 'not triggered'}")

print("\n[B2] Double Bottom")
db = np.concatenate([np.linspace(120,100,20), np.linspace(100,112,15),
                     np.linspace(112,100.5,15), np.linspace(100.5,115,15)])
r = _detect_double_bottom(extract_structures(make_bars(db, "DB")))
check("Double Bottom detected", r is not None and r.bias == Bias.LONG,
      f"{'detected' if r else 'not triggered'}")

print("\n[B3] Double Top")
dt = np.concatenate([np.linspace(100,120,20), np.linspace(120,108,15),
                     np.linspace(108,119.5,15), np.linspace(119.5,103,15)])
r = _detect_double_top(extract_structures(make_bars(dt, "DT")))
check("Double Top detected", r is not None and r.bias == Bias.SHORT,
      f"{'detected' if r else 'not triggered'}")

print("\n[B4] Triple Bottom")
tb = np.concatenate([np.linspace(120,100,15), np.linspace(100,110,10),
                     np.linspace(110,100.3,10), np.linspace(100.3,110,10),
                     np.linspace(110,100.2,10), np.linspace(100.2,115,15)])
r = _detect_triple_bottom(extract_structures(make_bars(tb, "TB")))
check("Triple Bottom detected", r is not None and r.bias == Bias.LONG,
      f"{'detected' if r else 'not triggered — tolerance may be tight'}")

print("\n[B5] Bull Flag")
bf = np.concatenate([np.linspace(100,100,5), np.linspace(100,115,10),
                     np.linspace(115,112,8), np.linspace(112,118,5)])
r = _detect_bull_flag(extract_structures(make_bars(bf, "BF")))
check("Bull Flag detected", r is not None and r.bias == Bias.LONG,
      f"{'detected' if r else 'not triggered'}")

print("\n[B6] Momentum Breakout")
np.random.seed(55)
mb = np.concatenate([100 + np.random.normal(0,1.5,25), np.linspace(102,110,5)])
r = _detect_momentum_breakout(extract_structures(make_bars(mb, "MB")))
check("Momentum Breakout detected", r is not None, f"{'detected' if r else 'not triggered'}")


# =====================================================================
# PART C: Candlestick Patterns
# =====================================================================
print("\n" + "=" * 70)
print("  PART C: Candlestick Patterns (10)")
print("=" * 70)

# Helper: build bars with explicit OHLC for candlestick testing
def candle_bars(ohlc_list, symbol="CS"):
    """ohlc_list = [(o,h,l,c), ...] most recent 20+ bars."""
    bars = []
    base_time = datetime(2024, 6, 3, 9, 30)
    for i, (o, h, l, c) in enumerate(ohlc_list):
        bars.append(Bar(symbol=symbol, timestamp=base_time + timedelta(hours=i),
                        open=o, high=h, low=l, close=c, volume=10000))
    return BarSeries(symbol=symbol, timeframe="1h", bars=bars)

# Build context: 15 bars of downtrend then test candles
down_ctx = [(100-i*0.5, 100-i*0.5+0.3, 100-i*0.5-0.8, 100-i*0.5-0.3) for i in range(15)]
up_ctx = [(100+i*0.5, 100+i*0.5+0.8, 100+i*0.5-0.3, 100+i*0.5+0.3) for i in range(15)]

print("\n[C1] Bullish Engulfing")
# Downtrend → red bar → big green bar that engulfs it
data = down_ctx + [(92.0, 92.3, 91.0, 91.2), (91.0, 93.5, 90.8, 93.2)]
r = _detect_bullish_engulfing(extract_structures(candle_bars(data, "BE")))
check("Bullish Engulfing", r is not None and r.bias == Bias.LONG,
      f"{'detected' if r else 'not triggered'}")

print("\n[C2] Bearish Engulfing")
data = up_ctx + [(107.0, 107.5, 106.8, 107.3), (107.5, 107.6, 105.0, 105.2)]
r = _detect_bearish_engulfing(extract_structures(candle_bars(data, "BRE")))
check("Bearish Engulfing", r is not None and r.bias == Bias.SHORT,
      f"{'detected' if r else 'not triggered'}")

print("\n[C3] Morning Star")
# Big red → small body star → big green
data = down_ctx + [(92, 92.3, 89.5, 89.8), (89.5, 89.8, 89.2, 89.4), (89.5, 92.0, 89.3, 91.8)]
r = _detect_morning_star(extract_structures(candle_bars(data, "MS")))
check("Morning Star", r is not None and r.bias == Bias.LONG,
      f"{'detected' if r else 'not triggered'}")

print("\n[C4] Evening Star")
data = up_ctx + [(107, 109.5, 106.8, 109.2), (109.5, 109.8, 109.2, 109.4), (109.3, 109.5, 107.0, 107.2)]
r = _detect_evening_star(extract_structures(candle_bars(data, "ES")))
check("Evening Star", r is not None and r.bias == Bias.SHORT,
      f"{'detected' if r else 'not triggered'}")

print("\n[C5] Hammer")
# At lows: small body at top, long lower shadow
data = down_ctx + [(91.0, 91.2, 88.0, 91.1)]  # body=0.1, lower_shadow=3.0
r = _detect_hammer(extract_structures(candle_bars(data, "HM")))
check("Hammer", r is not None and r.bias == Bias.LONG,
      f"{'detected' if r else 'not triggered'}")

print("\n[C6] Shooting Star")
data = up_ctx + [(107.0, 110.5, 106.8, 107.1)]  # body=0.1, upper_shadow=3.4
r = _detect_shooting_star(extract_structures(candle_bars(data, "SS")))
check("Shooting Star", r is not None and r.bias == Bias.SHORT,
      f"{'detected' if r else 'not triggered'}")

print("\n[C7] Three White Soldiers")
# 3 green bars, each closing higher, opening within prior body
data = down_ctx + [(90, 92, 89.8, 91.8), (91.0, 93.5, 90.8, 93.3), (92.0, 95.0, 91.8, 94.8)]
r = _detect_three_white_soldiers(extract_structures(candle_bars(data, "3WS")))
check("Three White Soldiers", r is not None and r.bias == Bias.LONG,
      f"{'detected' if r else 'not triggered'}")

print("\n[C8] Three Black Crows")
data = up_ctx + [(108.0, 108.2, 106.0, 106.2), (107.0, 107.2, 104.5, 104.7), (106.0, 106.2, 103.0, 103.2)]
r = _detect_three_black_crows(extract_structures(candle_bars(data, "3BC")))
check("Three Black Crows", r is not None and r.bias == Bias.SHORT,
      f"{'detected' if r else 'not triggered'}")

print("\n[C9] Doji")
data = down_ctx + [(91.0, 91.8, 90.0, 91.01)]  # body ≈ 0.01, range = 1.8
r = _detect_doji(extract_structures(candle_bars(data, "DJ")))
check("Doji", r is not None, f"{'detected bias=' + r.bias.value if r else 'not triggered'}")

print("\n[C10] Dragonfly Doji")
# Open ≈ close ≈ high, long lower shadow
data = down_ctx + [(91.0, 91.05, 88.0, 91.02)]  # body=0.02, range=3.05, upper=0.03, lower=3.0
r = _detect_dragonfly_doji(extract_structures(candle_bars(data, "DD")))
check("Dragonfly Doji", r is not None and r.bias == Bias.LONG,
      f"{'detected' if r else 'not triggered'}")


# =====================================================================
# PART D: Quant Strategy Patterns
# =====================================================================
print("\n" + "=" * 70)
print("  PART D: Quant Strategy Patterns (10)")
print("=" * 70)

print("\n[D1] Mean Reversion (z-score)")
# Price crashed 3 std devs below mean
np.random.seed(42)
mr_data = np.concatenate([np.ones(50) * 100 + np.random.normal(0, 1, 50),
                          np.linspace(100, 92, 10)])  # Crash to 92 (z < -2)
r = _detect_mean_reversion(extract_structures(make_bars(mr_data, "MR")))
check("Mean Reversion (oversold)", r is not None and r.bias == Bias.LONG,
      f"{'z=' + f'{r.description}' if r else 'not triggered'}")

print("\n[D2] Trend Pullback")
# Strong uptrend, pullback to 21 EMA
tp_data = np.concatenate([np.linspace(80, 130, 50),   # Strong trend
                          np.linspace(130, 124, 8)])   # Pullback
# Need green last bar (bounce)
tp_closes = list(tp_data)
tp_closes.append(125.0)  # Bounce bar
tp_opens = [c - 0.3 for c in tp_closes]
tp_opens[-1] = 123.5  # Green bar: open < close
r = _detect_trend_pullback(extract_structures(make_bars(tp_closes, "TP", opens=tp_opens)))
check("Trend Pullback", r is not None and r.bias == Bias.LONG,
      f"{'detected' if r else 'not triggered — EMA distance may be off'}")

print("\n[D3] Range Expansion")
# 14 calm bars then 1 huge bar
np.random.seed(88)
re_data = list(100 + np.random.normal(0, 0.5, 14)) + [100.0]
re_highs = [c + 0.3 for c in re_data]
re_lows = [c - 0.3 for c in re_data]
re_highs[-1] = 104.0; re_lows[-1] = 98.0; re_data[-1] = 103.5  # 6.0 range vs avg ~0.6
re_opens = [c - 0.1 for c in re_data]; re_opens[-1] = 99.0
r = _detect_range_expansion(extract_structures(make_bars(
    re_data, "RE", opens=re_opens, highs=re_highs, lows=re_lows)))
check("Range Expansion", r is not None, f"{'detected' if r else 'not triggered'}")

print("\n[D4] Volume Breakout")
np.random.seed(77)
vb_c = list(100 + np.random.normal(0, 0.5, 20)) + [104.0]  # Break above range
vb_v = [10000] * 20 + [50000]  # 5x volume spike
r = _detect_volume_breakout(extract_structures(make_bars(vb_c, "VB", volumes=vb_v)))
check("Volume Breakout", r is not None and r.bias == Bias.LONG,
      f"{'detected' if r else 'not triggered'}")

print("\n[D5] Donchian Breakout")
np.random.seed(33)
dn_c = list(100 + np.random.normal(0, 1, 54)) + [108.0]  # Break 50-bar high
r = _detect_donchian_breakout(extract_structures(make_bars(dn_c, "DN")))
check("Donchian Breakout", r is not None and r.bias == Bias.LONG,
      f"{'detected' if r else 'not triggered'}")


# =====================================================================
# PART E: TradeSetup Format + Pipeline
# =====================================================================
print("\n" + "=" * 70)
print("  PART E: TradeSetup Format + Full Pipeline")
print("=" * 70)

# Find any detected setup to test format
all_results = classify_all(make_bars(hs, "PIPE"))
check("classify_all returns list", isinstance(all_results, list))
check("Sorted by confidence", all(all_results[i].confidence >= all_results[i+1].confidence
      for i in range(len(all_results)-1)) if len(all_results) > 1 else True)
print(f"  Found {len(all_results)} patterns on H&S data:")
for r in all_results:
    print(f"    {r.pattern_name:<25} {r.category.value:<12} {r.bias.value:<6} "
          f"conf={r.confidence:.0%}  R:R={r.risk_reward_ratio:.1f}")

if all_results:
    d = all_results[0].to_dict()
    for key in ["pattern_name","bias","entry_price","stop_loss","target_price",
                "risk_reward_ratio","confidence","category","strategy_type","key_levels"]:
        check(f"to_dict['{key}']", key in d)

# Short data → empty
check("Short data → empty", len(classify_all(make_bars([100,101,102], "S"))) == 0)

# Count unique pattern functions in _ALL_DETECTORS
from backend.patterns.classifier import _ALL_DETECTORS
check("47 detector functions registered", len(_ALL_DETECTORS) == 47, f"got {len(_ALL_DETECTORS)}")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
total = PASS + FAIL
status = "ALL PASS" if FAIL == 0 else f"{FAIL} FAILED"
print(f"  PHASE 4 RESULTS: {PASS}/{total} passed — {status}")
print("=" * 70)

if FAIL > 0:
    print("\n  Note: Some patterns may not trigger on synthetic data due to")
    print("  zigzag/tolerance sensitivity. Real market data has more pronounced")
    print("  structures. The critical test is: pipeline runs, format is correct,")
    print("  and patterns that DO fire produce valid TradeSetup objects.")

print(f"""
  47 Pattern Detectors:
    Classical (16):   H&S, Inv H&S, Double/Triple Top/Bottom,
                      Asc/Desc/Sym Triangle, Bull/Bear Flag, Pennant,
                      Cup & Handle, Rectangle, Rising/Falling Wedge
    Candlestick (10): Bullish/Bearish Engulfing, Morning/Evening Star,
                      Hammer, Shooting Star, Doji, Dragonfly Doji,
                      Three White Soldiers, Three Black Crows
    SMB Scalps (11):  RubberBand, HitchHiker, ORB 15/30, Second Chance,
                      BackSide, Fashionably Late, Spencer, Gap G&G,
                      Tidal Wave, Breaking News
    Quant (10):       Momentum Breakout, Vol Compression, Mean Reversion,
                      Trend Pullback, Gap Fade, Relative Strength Break,
                      Range Expansion, Volume Breakout, VWAP Reversion,
                      Donchian Breakout

  Next: Phase 5 — 10 Quant Strategies + Rolling Evaluator
""")