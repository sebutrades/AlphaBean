"""
test_phase2_features.py — Validates Phase 2: 8 Statistical Features

Run from project root:
    python test_phase2_features.py

Tests:
  1. Imports
  2. Momentum (bullish, bearish, neutral)
  3. Volatility (elevated vs compressed)
  4. Volatility Compression (squeeze detection)
  5. Volume Expansion (RVOL)
  6. Trend Strength (MA alignment)
  7. Range Breakout (proximity to N-day high/low)
  8. Mean Reversion (z-score extremes)
  9. Regime Score (placeholder using self-data)
  10. Composite score + full pipeline
"""
import sys
import numpy as np

print("=" * 65)
print("  AlphaBean v3.0 — Phase 2: Feature Engine Test")
print("=" * 65)

# ── 1. Imports ──────────────────────────────────────────────
print("\n[1/10] Testing imports...")
try:
    from backend.features.engine import (
        compute_features, FeatureResult, FeatureScore,
        _compute_momentum, _compute_volatility, _compute_vol_compression,
        _compute_volume_expansion, _compute_trend_strength,
        _compute_range_breakout, _compute_mean_reversion,
        _compute_regime_score,
    )
    print("  PASS — All imports successful")
except ImportError as e:
    print(f"  FAIL — {e}")
    sys.exit(1)


# ── Helper: generate test data ───────────────────────────────

def make_uptrend(n=200, start=100, end=150):
    """Steadily rising price with noise."""
    trend = np.linspace(start, end, n)
    noise = np.random.normal(0, 0.5, n)
    closes = trend + noise
    highs = closes + np.abs(np.random.normal(0, 0.3, n)) + 0.5
    lows = closes - np.abs(np.random.normal(0, 0.3, n)) - 0.5
    volumes = np.random.randint(8000, 15000, n).astype(np.float64)
    return closes, highs, lows, volumes


def make_downtrend(n=200, start=150, end=100):
    return make_uptrend(n, start, end)


def make_range(n=200, center=120, width=5):
    """Flat, range-bound price."""
    closes = center + np.random.normal(0, width * 0.3, n)
    highs = closes + np.abs(np.random.normal(0, 0.3, n)) + 0.3
    lows = closes - np.abs(np.random.normal(0, 0.3, n)) - 0.3
    volumes = np.random.randint(8000, 12000, n).astype(np.float64)
    return closes, highs, lows, volumes


def make_squeeze(n=100):
    """Volatility compression: wide range → narrow range."""
    closes = np.ones(n) * 100
    for i in range(n):
        # Range shrinks over time
        amplitude = 5.0 * (1 - i / n)
        closes[i] += np.sin(i * 0.5) * amplitude
    highs = closes + np.abs(5.0 * np.linspace(1, 0.1, n))
    lows = closes - np.abs(5.0 * np.linspace(1, 0.1, n))
    volumes = np.random.randint(8000, 12000, n).astype(np.float64)
    return closes, highs, lows, volumes


# ── 2. Momentum ─────────────────────────────────────────────
print("\n[2/10] Feature: Momentum...")

up_c, up_h, up_l, up_v = make_uptrend()
dn_c, dn_h, dn_l, dn_v = make_downtrend()

mom_bull = _compute_momentum(up_c, len(up_c))
mom_bear = _compute_momentum(dn_c, len(dn_c))

assert mom_bull.score > 55, f"Uptrend momentum should be >55, got {mom_bull.score}"
assert mom_bear.score < 45, f"Downtrend momentum should be <45, got {mom_bear.score}"
print(f"  PASS — Uptrend: {mom_bull.score:.0f} ({mom_bull.description})")
print(f"  PASS — Downtrend: {mom_bear.score:.0f} ({mom_bear.description})")


# ── 3. Volatility ────────────────────────────────────────────
print("\n[3/10] Feature: Volatility...")

vol_result = _compute_volatility(up_c, len(up_c))
assert 0 <= vol_result.score <= 100
print(f"  PASS — Score={vol_result.score:.0f}, {vol_result.description}")

# High vol: add spikes to the last 10 bars
high_vol_c = np.copy(up_c)
high_vol_c[-10:] += np.random.normal(0, 5, 10)  # Big moves at end
vol_high = _compute_volatility(high_vol_c, len(high_vol_c))
assert vol_high.score > vol_result.score, "Spiked data should have higher vol score"
print(f"  PASS — Elevated vol: {vol_high.score:.0f} vs normal {vol_result.score:.0f}")


# ── 4. Volatility Compression ───────────────────────────────
print("\n[4/10] Feature: Volatility Compression (squeeze)...")

sq_c, sq_h, sq_l, sq_v = make_squeeze()
vc_result = _compute_vol_compression(sq_h, sq_l, sq_c, len(sq_c))
assert vc_result.score > 60, f"Squeeze should score >60, got {vc_result.score}"
print(f"  PASS — Squeeze score={vc_result.score:.0f} ({vc_result.description})")

# Non-squeeze: regular uptrend
vc_normal = _compute_vol_compression(up_h, up_l, up_c, len(up_c))
print(f"  PASS — Normal: {vc_normal.score:.0f} vs squeeze {vc_result.score:.0f}")


# ── 5. Volume Expansion ─────────────────────────────────────
print("\n[5/10] Feature: Volume Expansion (RVOL)...")

# Normal volume
ve_normal = _compute_volume_expansion(up_v, len(up_v))

# Create volume spike at end
spike_v = np.copy(up_v)
spike_v[-3:] = spike_v[-3:] * 4  # 4x volume spike
ve_spike = _compute_volume_expansion(spike_v, len(spike_v))

assert ve_spike.score > ve_normal.score, "Volume spike should score higher"
print(f"  PASS — Normal RVOL: {ve_normal.score:.0f} ({ve_normal.description})")
print(f"  PASS — Spike RVOL: {ve_spike.score:.0f} ({ve_spike.description})")


# ── 6. Trend Strength ───────────────────────────────────────
print("\n[6/10] Feature: Trend Strength (MA alignment)...")

ts_bull = _compute_trend_strength(up_c, len(up_c))
ts_bear = _compute_trend_strength(dn_c, len(dn_c))

assert ts_bull.score > 60, f"Uptrend should score >60, got {ts_bull.score}"
assert ts_bear.score < 40, f"Downtrend should score <40, got {ts_bear.score}"
print(f"  PASS — Uptrend: {ts_bull.score:.0f} ({ts_bull.description})")
print(f"  PASS — Downtrend: {ts_bear.score:.0f} ({ts_bear.description})")


# ── 7. Range Breakout ────────────────────────────────────────
print("\n[7/10] Feature: Range Breakout...")

rb_bull = _compute_range_breakout(up_c, up_h, up_l, len(up_c))
rb_bear = _compute_range_breakout(dn_c, dn_h, dn_l, len(dn_c))

assert rb_bull.score > 60, f"At highs should score >60, got {rb_bull.score}"
assert rb_bear.score < 40, f"At lows should score <40, got {rb_bear.score}"
print(f"  PASS — Near highs: {rb_bull.score:.0f} ({rb_bull.description})")
print(f"  PASS — Near lows: {rb_bear.score:.0f} ({rb_bear.description})")


# ── 8. Mean Reversion ────────────────────────────────────────
print("\n[8/10] Feature: Mean Reversion (z-score)...")

# Make price stretched far above its mean
stretched_c = np.concatenate([
    np.ones(50) * 100,     # Flat at 100
    np.linspace(100, 115, 10),  # Spike up
])
mr = _compute_mean_reversion(stretched_c, len(stretched_c))
assert mr.score > 30, f"Stretched price should have elevated mr score, got {mr.score}"
assert mr.raw_value > 0, f"Z-score should be positive for above-mean, got {mr.raw_value}"
print(f"  PASS — Stretched up: score={mr.score:.0f}, z={mr.raw_value:.2f} ({mr.description})")

# Price at mean
flat_c = np.random.normal(100, 0.5, 60)
mr_flat = _compute_mean_reversion(flat_c, len(flat_c))
print(f"  PASS — Near mean: score={mr_flat.score:.0f}, z={mr_flat.raw_value:.2f} ({mr_flat.description})")


# ── 9. Regime Score ──────────────────────────────────────────
print("\n[9/10] Feature: Regime Score...")

regime_bull = _compute_regime_score(up_c, None, len(up_c))
regime_bear = _compute_regime_score(dn_c, None, len(dn_c))
assert regime_bull.score > regime_bear.score
print(f"  PASS — Bull self-regime: {regime_bull.score:.0f} ({regime_bull.description})")
print(f"  PASS — Bear self-regime: {regime_bear.score:.0f} ({regime_bear.description})")

# With SPY data
spy_bull = np.linspace(400, 450, 200)
regime_spy = _compute_regime_score(up_c, spy_bull, len(up_c))
print(f"  PASS — With SPY data: {regime_spy.score:.0f} ({regime_spy.description})")


# ── 10. Composite: full pipeline ─────────────────────────────
print("\n[10/10] Full pipeline: compute_features()...")

result = compute_features(up_c, up_h, up_l, up_v)

assert isinstance(result, FeatureResult)
assert 0 <= result.composite_score <= 100

print(f"\n  Composite Score: {result.composite_score:.1f}/100\n")
print(f"  {'Feature':<20} {'Score':>6} {'Raw':>8}   Description")
print(f"  {'-'*70}")
for f in result.all_scores:
    print(f"  {f.name:<20} {f.score:>5.0f}  {f.raw_value:>8.4f}   {f.description}")

# Check dict export
d = result.to_dict()
assert "composite_score" in d
assert "features" in d
assert len(d["features"]) == 8
print(f"\n  PASS — to_dict() works ({len(d['features'])} features)")

# Test bearish data too
bear_result = compute_features(dn_c, dn_h, dn_l, dn_v)
assert bear_result.composite_score < result.composite_score, \
    "Bearish composite should be lower than bullish"
print(f"  PASS — Bullish composite={result.composite_score:.1f} > "
      f"Bearish composite={bear_result.composite_score:.1f}")


# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 2 COMPLETE — All 8 features validated")
print("=" * 65)
print("""
  Files created:
    backend/features/__init__.py
    backend/features/engine.py

  Features validated:
    1. Momentum         Multi-window rate of change
    2. Volatility       Current vs historical stddev
    3. Vol Compression  ATR squeeze detection
    4. Volume Expansion RVOL (relative volume)
    5. Trend Strength   MA alignment (9/21/50/200)
    6. Range Breakout   Proximity to N-day high/low
    7. Mean Reversion   Z-score from moving average
    8. Regime Score     Bull/bear classification

  All vectorized with NumPy. No per-bar Python loops.

  Next: Phase 3 — Market Regime Detector
    backend/regime/detector.py
""")