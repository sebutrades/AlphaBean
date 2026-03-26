"""
patterns/classifier.py — Structure-first pattern classification (42 detectors).

v2.0 — Post-audit rebuild:
  Removed 5: Breaking News, HitchHiker, Spencer, BackSide, Relative Strength Break
  Systemic: _make() rejects retroactive entries (entry already exceeded by 0.5 ATR)
  Fixed: Tidal Wave (5 fixes), Wedge targets, Flag poles, ORB stops/targets,
         Cup&Handle, Second Chance, RubberBand, Gap G&G, Gap Fade,
         Mean Reversion, VWAP Reversion, Trend Pullback, Fashionably Late
  Added: _candle_context_ok() for all 10 candlestick patterns
  Daily: 5 breakout strategies only fire on 1d timeframe

Pipeline: BarSeries → extract_structures() → classify_all() → list[TradeSetup]
"""
from datetime import datetime, time, timedelta
from typing import Optional

import numpy as np

from backend.data.schemas import BarSeries, Bar
from backend.structures.swings import (
    SwingPoint, SwingType, zigzag, adaptive_zigzag_threshold,
    find_swing_highs, find_swing_lows, adaptive_order,
    swing_highs_from_zigzag, swing_lows_from_zigzag,
)
from backend.structures.trendlines import (
    fit_trendline, detect_channel, is_flat_line, slopes_same_sign,
)
from backend.structures.support_resistance import cluster_levels, nearest_level
from backend.structures.indicators import wilder_atr, atr_ratio, ema, ema_last
from backend.patterns.registry import TradeSetup, Bias, PatternCategory, PATTERN_META


# ==============================================================================
# STRUCTURE EXTRACTION
# ==============================================================================

class ExtractedStructures:
    """All structural primitives extracted from a BarSeries (computed once)."""
    def __init__(self, bars: BarSeries):
        b = bars.bars
        self.bars = b
        self.n = len(b)
        self.symbol = bars.symbol
        self.timeframe = bars.timeframe
        self.closes = np.array([x.close for x in b], dtype=np.float64)
        self.highs = np.array([x.high for x in b], dtype=np.float64)
        self.lows = np.array([x.low for x in b], dtype=np.float64)
        self.opens = np.array([x.open for x in b], dtype=np.float64)
        self.volumes = np.array([x.volume for x in b], dtype=np.float64)
        self.timestamps = [x.timestamp for x in b]

        threshold = adaptive_zigzag_threshold(bars.timeframe)
        self.zz_swings = zigzag(self.highs, self.lows, threshold,
                                timestamps=self.timestamps, volumes=self.volumes)
        self.zz_highs = swing_highs_from_zigzag(self.zz_swings)
        self.zz_lows = swing_lows_from_zigzag(self.zz_swings)

        order = adaptive_order(bars.timeframe)
        self.sw_high_idx = find_swing_highs(self.highs, order=order)
        self.sw_low_idx = find_swing_lows(self.lows, order=order)
        self.sr_levels = cluster_levels(self.zz_swings, tolerance_pct=0.8, min_touches=2)

        self.atr_series = wilder_atr(self.highs, self.lows, self.closes, period=14)
        self.current_atr = float(self.atr_series[-1]) if not np.isnan(self.atr_series[-1]) else 0

        self.day_open_idx = self._find_day_open()

        # Precompute regime hint for filters
        self._regime = self._compute_regime_hint()

    def _find_day_open(self) -> int:
        if self.n == 0: return 0
        today = self.timestamps[-1].date()
        for i, ts in enumerate(self.timestamps):
            if ts.date() == today and ts.time() >= time(9, 30):
                return i
        for i, ts in enumerate(self.timestamps):
            if ts.date() == today:
                return i
        return max(0, self.n - 20)

    def _compute_regime_hint(self) -> str:
        """Quick regime hint: trending_bull, trending_bear, or mean_reverting."""
        if self.n < 50: return "unknown"
        sma50 = float(np.mean(self.closes[-50:]))
        cur = self.closes[-1]
        r = atr_ratio(self.highs, self.lows, self.closes, 14, 40)
        if cur > sma50 and r > 0.8:
            return "trending_bull"
        elif cur < sma50 and r > 0.8:
            return "trending_bear"
        elif r < 0.6:
            return "mean_reverting"
        return "mixed"


def extract_structures(bars: BarSeries) -> ExtractedStructures:
    return ExtractedStructures(bars)


# ==============================================================================
# HELPERS
# ==============================================================================

# ==============================================================================
# UPDATED _make() — now supports scaled exits
# ==============================================================================
 
def _make(s, name, bias, entry, stop, target, conf, desc,
          target_1=0.0, target_2=0.0,
          trail_type="atr", trail_param=2.0,
          position_splits=(0.5, 0.3, 0.2),
          **kw):
    """Create TradeSetup with entry validation and scaled exit defaults.
    
    Changes from v2.1:
      - Accepts target_1, target_2, trail_type, trail_param, position_splits
      - Auto-calculates target_1 = entry + 1R if not provided
      - Auto-sets target_2 = target (measured move) if not provided
      - Uses ATR-based retroactive entry check (unchanged)
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    rr = round(abs(target - entry) / risk, 2)
    if rr < 1.0:
        return None
 
    cur = s.closes[-1]
    atr = s.current_atr if s.current_atr > 0 else abs(cur * 0.01)
 
    # Retroactive entry rejection (unchanged from v2.1)
    if bias == Bias.LONG:
        if cur > entry + atr * 0.5:
            return None
    elif bias == Bias.SHORT:
        if cur < entry - atr * 0.5:
            return None
 
    # Auto-calculate scaled targets if not provided
    if target_1 == 0.0:
        # Default T1 = 1R from entry
        if bias == Bias.LONG:
            target_1 = round(entry + risk, 2)
        else:
            target_1 = round(entry - risk, 2)
 
    if target_2 == 0.0:
        # Default T2 = the full measured move target
        target_2 = round(target, 2)
 
    meta = PATTERN_META.get(name, {})
    return TradeSetup(
        pattern_name=name, category=meta.get("cat", PatternCategory.CLASSICAL),
        symbol=s.symbol, bias=bias,
        entry_price=round(entry, 2), stop_loss=round(stop, 2),
        target_price=round(target, 2), risk_reward_ratio=rr,
        confidence=round(min(0.95, conf), 2), detected_at=s.timestamps[-1],
        description=desc, strategy_type=meta.get("type", "breakout"),
        win_rate=meta.get("wr", 0.5), timeframe_detected=s.timeframe,
        # Scaled exits
        target_1=target_1, target_2=target_2,
        trail_type=trail_type, trail_param=trail_param,
        position_splits=position_splits,
        **kw)
 


def _candle_context_ok(s, is_bullish: bool, key_bar_idx: int = -1) -> bool:
    """Context filter for candlestick patterns. ALL THREE must pass:
    1. S/R confluence: signal bar within 0.5 ATR of a known level
    2. Volume: key bar volume >= 1.3x 20-bar average
    3. Trend: sufficient prior move in opposite direction (reversal context)
    """
    if s.current_atr <= 0 or s.n < 25: return False
    idx = key_bar_idx if key_bar_idx >= 0 else s.n + key_bar_idx
    atr = s.current_atr

    # 1. S/R confluence — signal bar within 0.5 ATR of a known level
    bar_price = s.lows[idx] if is_bullish else s.highs[idx]
    near_sr = False
    for lvl in s.sr_levels:
        if abs(lvl.price - bar_price) < atr * 0.5:
            near_sr = True; break
    # Also check zigzag swing levels if sr_levels is sparse
    if not near_sr:
        check_swings = s.zz_lows if is_bullish else s.zz_highs
        for sw in check_swings:
            if abs(sw.price - bar_price) < atr * 0.5:
                near_sr = True; break
    if not near_sr: return False

    # 2. Volume >= 1.3x 20-bar average
    lb = max(0, idx - 20)
    avg_vol = float(np.mean(s.volumes[lb:idx])) if idx > lb else 1
    if avg_vol <= 0 or s.volumes[idx] < avg_vol * 1.3: return False

    # 3. Trend context: price moved >= 1.5 ATR in the opposite direction over prior 20 bars
    lb20 = max(0, idx - 20)
    if is_bullish:
        prior_high = float(np.max(s.highs[lb20:idx]))
        decline = prior_high - s.lows[idx]
        if decline < atr * 1.5: return False
    else:
        prior_low = float(np.min(s.lows[lb20:idx]))
        rally = s.highs[idx] - prior_low
        if rally < atr * 1.5: return False

    return True


def _in_time(ts, sh, sm, eh, em):
    return time(sh, sm) <= ts.time() <= time(eh, em)

def _body(bar): return abs(bar.close - bar.open)
def _upper_shadow(bar): return bar.high - max(bar.open, bar.close)
def _lower_shadow(bar): return min(bar.open, bar.close) - bar.low
def _range(bar): return bar.high - bar.low
def _is_green(bar): return bar.close > bar.open
def _is_red(bar): return bar.close < bar.open


def _compute_vwap(s, start_idx: int) -> float:
    """Compute VWAP from start_idx to end of bars."""
    cv, ctv = 0.0, 0.0
    for i in range(start_idx, s.n):
        tp = (s.highs[i] + s.lows[i] + s.closes[i]) / 3
        cv += s.volumes[i]; ctv += tp * s.volumes[i]
    return ctv / cv if cv > 0 else s.closes[-1]


def _vwap_today(s):
    """Convenience wrapper: compute VWAP from today's first bar."""
    today = s.timestamps[-1].date()
    for i in range(s.n):
        if s.timestamps[i].date() == today:
            return _compute_vwap(s, i)
    return _compute_vwap(s, 0)


def _compute_ema9(s) -> float | None:
    """Compute 9-period EMA of closes. Returns current value or None."""
    if s.n < 10:
        return None
    mult = 2.0 / 10.0  # 2 / (period + 1)
    ema = float(s.closes[0])
    for i in range(1, s.n):
        ema = s.closes[i] * mult + ema * (1 - mult)
    return ema
# ==============================================================================
# ATR-BASED OFFSET (replaces ALL $0.02 fixed offsets)
# ==============================================================================
 
def _atr_offset(atr: float, multiplier: float = 0.1) -> float:
    """Return an ATR-scaled offset for entry/stop placement.
    
    Replaces all hardcoded $0.02 offsets throughout the codebase.
    A $500 stock with ATR=$5 gets $0.50 offset.
    A $10 stock with ATR=$0.30 gets $0.03 offset.
    
    Common multipliers:
      0.05 = tight (entry trigger just past level)
      0.10 = standard entry offset
      0.15 = stop buffer beyond structural level  
      0.25 = wide stop buffer
    """
    if atr <= 0:
        return 0.02  # Fallback if ATR unavailable
    return round(atr * multiplier, 2)

 
 # ==============================================================================
# VOLUME ANALYSIS HELPERS
# ==============================================================================
 
def _volume_confirms_breakout(s, bar_idx: int = -1, threshold: float = 1.3) -> bool:
    """Check if the bar at bar_idx has volume >= threshold × 20-bar average.
    
    Used for: structural breakouts (triangles, rectangles, H&S neckline breaks),
    flag breakouts, cup & handle breakouts.
    
    Args:
        s: ExtractedStructures
        bar_idx: Index of the bar to check (-1 = last bar)
        threshold: Volume must be >= this × average (default 1.3x)
    
    Returns:
        True if volume confirms the breakout
    """
    idx = bar_idx if bar_idx >= 0 else s.n + bar_idx
    if idx < 20:
        return True  # Not enough history, don't reject
 
    avg_vol = float(np.mean(s.volumes[max(0, idx - 20):idx]))
    if avg_vol <= 0:
        return True  # Can't compute, don't reject
 
    return s.volumes[idx] >= avg_vol * threshold
 
 
def _volume_declining_formation(s, start_idx: int, end_idx: int) -> bool:
    """Check if volume is generally declining during a formation period.
    
    Used for: triangle formations, flag consolidations, wedge formations.
    Professional observation: volume contracts during consolidation patterns,
    then expands on breakout. This filter removes patterns where volume
    is INCREASING during formation (suggests accumulation, not consolidation).
    
    Method: Compare average volume in first half vs second half of formation.
    Second half should be lower.
    """
    if end_idx - start_idx < 6:
        return True  # Too short to measure, don't reject
 
    mid = (start_idx + end_idx) // 2
    first_half_vol = float(np.mean(s.volumes[start_idx:mid]))
    second_half_vol = float(np.mean(s.volumes[mid:end_idx]))
 
    if first_half_vol <= 0:
        return True
 
    # Second half volume should be < first half (allowing 10% tolerance)
    return second_half_vol <= first_half_vol * 1.10
 
 
def _volume_exhaustion(s, bar_idx: int = -1) -> bool:
    """Check for volume exhaustion at a price extreme.
    
    Used for: mean reversion, VWAP reversion, gap fade.
    At price extremes, declining volume on the final push signals exhaustion.
    
    Method: Check that the signal bar's volume is LESS than the 3-bar
    average leading up to it (sellers/buyers are drying up at the extreme).
    """
    idx = bar_idx if bar_idx >= 0 else s.n + bar_idx
    if idx < 5:
        return True  # Not enough data
 
    recent_avg = float(np.mean(s.volumes[max(0, idx - 3):idx]))
    if recent_avg <= 0:
        return True
 
    # Signal bar volume should be declining relative to recent bars
    # (allowing the bar itself to be up to 130% — not a hard cutoff)
    return s.volumes[idx] <= recent_avg * 1.3
 
 
def _volume_pattern_hs(s, ls_idx: int, head_idx: int, rs_idx: int) -> float:
    """Score the volume pattern for Head & Shoulders.
    
    Bulkowski: Volume should be highest at the left shoulder, lower at the head,
    lowest at the right shoulder. This declining pattern adds ~0.10 confidence.
    
    Returns: confidence bonus (0.0 to 0.12)
    """
    if ls_idx >= s.n or head_idx >= s.n or rs_idx >= s.n:
        return 0.0
 
    # Get average volume around each peak (±2 bars)
    def avg_vol_around(idx, radius=2):
        start = max(0, idx - radius)
        end = min(s.n, idx + radius + 1)
        return float(np.mean(s.volumes[start:end]))
 
    ls_vol = avg_vol_around(ls_idx)
    head_vol = avg_vol_around(head_idx)
    rs_vol = avg_vol_around(rs_idx)
 
    if ls_vol <= 0:
        return 0.0
 
    bonus = 0.0
    # Left shoulder > head volume
    if head_vol < ls_vol:
        bonus += 0.04
    # Head > right shoulder volume
    if rs_vol < head_vol:
        bonus += 0.04
    # Perfect declining pattern
    if ls_vol > head_vol > rs_vol:
        bonus += 0.04
 
    return bonus
 
 
def _volume_double_touch(s, idx1: int, idx2: int) -> float:
    """Score volume pattern for double top/bottom.
    
    Second touch should have lower volume than first (exhaustion).
    Returns confidence bonus 0.0 to 0.08.
    """
    if idx1 >= s.n or idx2 >= s.n:
        return 0.0
 
    def avg_vol_around(idx, radius=2):
        start = max(0, idx - radius)
        end = min(s.n, idx + radius + 1)
        return float(np.mean(s.volumes[start:end]))
 
    v1 = avg_vol_around(idx1)
    v2 = avg_vol_around(idx2)
 
    if v1 <= 0:
        return 0.0
 
    if v2 < v1 * 0.85:
        return 0.08  # Strong exhaustion on second touch
    elif v2 < v1:
        return 0.04  # Mild exhaustion
    return 0.0
 
 
# ==============================================================================
# S/R TARGET HELPER
# ==============================================================================
 
def _nearest_sr_target(s, entry: float, bias_long: bool, min_rr: float = 1.0) -> float:
    """Find the nearest S/R level that gives at least min_rr as a target.
    
    Used for realistic profit targets instead of always using the full
    measured move. Returns 0.0 if no suitable S/R level found.
    
    For longs: find nearest resistance above entry
    For shorts: find nearest support below entry
    """
    risk = abs(entry - (entry - s.current_atr if bias_long else entry + s.current_atr))
 
    for level in s.sr_levels:
        if bias_long and level.price > entry:
            reward = level.price - entry
            if risk > 0 and reward / risk >= min_rr:
                return round(level.price, 2)
        elif not bias_long and level.price < entry:
            reward = entry - level.price
            if risk > 0 and reward / risk >= min_rr:
                return round(level.price, 2)
 
    return 0.0  # No suitable S/R target found
 
 
# ==============================================================================
# REGIME CONFIDENCE MULTIPLIER
# ==============================================================================
 
def _regime_confidence_mult(s, strategy_type: str) -> float:
    """Return a confidence multiplier based on regime alignment.
    
    Boosts confidence when pattern type aligns with current market regime,
    penalizes when misaligned.
    
    strategy_type: "breakout", "momentum", "mean_reversion", "scalp"
    
    Returns: multiplier (0.7 to 1.2)
    """
    regime = s._regime  # "trending_bull", "trending_bear", "mean_reverting", "mixed", "unknown"
 
    if regime == "unknown" or regime == "mixed":
        return 1.0
 
    # Alignment matrix
    alignment = {
        "trending_bull": {
            "momentum": 1.15, "breakout": 1.10, "scalp": 1.0, "mean_reversion": 0.75,
        },
        "trending_bear": {
            "momentum": 1.10, "breakout": 1.05, "scalp": 1.0, "mean_reversion": 0.75,
        },
        "mean_reverting": {
            "mean_reversion": 1.15, "scalp": 1.05, "breakout": 0.80, "momentum": 0.75,
        },
    }
 
    return alignment.get(regime, {}).get(strategy_type, 1.0)
 
 
# ==============================================================================
# NR7 FILTER (for ORB)
# ==============================================================================
 
def _is_nr7(s) -> bool:
    """Check if yesterday's daily range was the narrowest of the last 7 days.
    
    Crabel's key insight: ORB works best after NR7 days (narrow range 7).
    Volatility compression → breakout.
    
    We approximate using intraday bars: check if yesterday's session range
    was the tightest of the last 7 sessions.
    """
    if s.n < 50:
        return False
 
    # Group bars by date, compute each day's range
    daily_ranges = {}
    for i in range(s.n):
        d = s.timestamps[i].date()
        if d not in daily_ranges:
            daily_ranges[d] = {"high": s.highs[i], "low": s.lows[i]}
        else:
            daily_ranges[d]["high"] = max(daily_ranges[d]["high"], s.highs[i])
            daily_ranges[d]["low"] = min(daily_ranges[d]["low"], s.lows[i])
 
    dates = sorted(daily_ranges.keys())
    if len(dates) < 3:
        return False
 
    # Yesterday = second to last date, today = last date
    # Check last 7 completed sessions (exclude today)
    completed = dates[:-1]  # Exclude today (incomplete)
    if len(completed) < 7:
        return False
 
    recent_7 = completed[-7:]
    ranges = [daily_ranges[d]["high"] - daily_ranges[d]["low"] for d in recent_7]
 
    # Yesterday is the last in the list
    yesterday_range = ranges[-1]
    return yesterday_range == min(ranges)
 
 
# ==============================================================================
# SPAN VALIDATION (minimum pattern width)
# ==============================================================================
 
def _min_span_ok(idx1: int, idx2: int, timeframe: str) -> bool:
    """Check if two swing points are far enough apart for a valid pattern.
    
    Double/Triple top/bottom patterns need enough time between peaks/troughs
    to represent a genuine structure, not just noise.
    
    Minimum spans (in bars):
      5min:  20 bars (100 min)
      15min: 15 bars (3.75 hours) 
      1h:    10 bars (10 hours)
      1d:    15 bars (3 weeks)
    """
    min_spans = {"5min": 20, "15min": 15, "1h": 10, "1d": 15}
    required = min_spans.get(timeframe, 10)
    return abs(idx2 - idx1) >= required

# ==============================================================================
# HELPER: ADX Computation
# ==============================================================================
 
def _compute_adx(s, period: int = 14) -> float | None:
    """Compute Average Directional Index (ADX) using Wilder's smoothing.
    
    Returns the current ADX value, or None if insufficient data.
    ADX > 25 = trending, ADX < 20 = ranging/choppy.
    
    Add this to classifier.py alongside the other helpers.
    """
    if s.n < period * 3:
        return None
 
    highs = s.highs
    lows = s.lows
    closes = s.closes
 
    # Step 1: Compute +DM, -DM, TR for each bar
    plus_dm = np.zeros(s.n)
    minus_dm = np.zeros(s.n)
    tr = np.zeros(s.n)
 
    for i in range(1, s.n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
 
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
 
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
 
    # Step 2: Wilder smooth +DM, -DM, TR
    def wilder_smooth(data, n):
        result = np.zeros(len(data))
        result[n] = np.sum(data[1:n + 1])
        for i in range(n + 1, len(data)):
            result[i] = result[i - 1] - (result[i - 1] / n) + data[i]
        return result
 
    smooth_plus_dm = wilder_smooth(plus_dm, period)
    smooth_minus_dm = wilder_smooth(minus_dm, period)
    smooth_tr = wilder_smooth(tr, period)
 
    # Step 3: +DI, -DI
    plus_di = np.zeros(s.n)
    minus_di = np.zeros(s.n)
    for i in range(period, s.n):
        if smooth_tr[i] > 0:
            plus_di[i] = 100 * smooth_plus_dm[i] / smooth_tr[i]
            minus_di[i] = 100 * smooth_minus_dm[i] / smooth_tr[i]
 
    # Step 4: DX
    dx = np.zeros(s.n)
    for i in range(period, s.n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
 
    # Step 5: ADX = Wilder smooth of DX
    adx = np.zeros(s.n)
    start = period * 2
    if start >= s.n:
        return None
    adx[start] = np.mean(dx[period:start + 1])
    for i in range(start + 1, s.n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
 
    val = adx[-1]
    return val if val > 0 else None
 
 
# ==============================================================================
# HELPER: Higher Lows / Lower Highs Check
# ==============================================================================
 
def _has_higher_lows(s, lookback: int = 10) -> bool:
    """Check if the last `lookback` bars show a pattern of higher lows.
    
    We don't require EVERY bar to be a higher low — that's too strict.
    Instead, we check that the lows are generally rising by comparing
    the min of the first half vs the min of the second half.
    Also check that the most recent 3 bars' low > the low from 7-10 bars ago.
    """
    if s.n < lookback:
        return False
 
    lows = s.lows[-lookback:]
 
    # First half min vs second half min
    mid = lookback // 2
    first_half_low = min(lows[:mid])
    second_half_low = min(lows[mid:])
 
    if second_half_low <= first_half_low:
        return False
 
    # Recent low must be above the low from 7-10 bars ago
    recent_low = min(lows[-3:])
    old_low = min(lows[:3])
 
    return recent_low > old_low
 
 
def _has_lower_highs(s, lookback: int = 10) -> bool:
    """Mirror of _has_higher_lows for short setups."""
    if s.n < lookback:
        return False
 
    highs = s.highs[-lookback:]
 
    mid = lookback // 2
    first_half_high = max(highs[:mid])
    second_half_high = max(highs[mid:])
 
    if second_half_high >= first_half_high:
        return False
 
    recent_high = max(highs[-3:])
    old_high = max(highs[:3])
 
    return recent_high < old_high
 
 
# ==============================================================================
# HELPER: Consecutive Weekly Closes
# ==============================================================================
 
def _consecutive_weekly_trend(s, weeks_required: int = 3, direction_long: bool = True) -> bool:
    """Check if the stock has closed higher (long) or lower (short) for
    N consecutive weeks.
    
    Groups daily bars by week (Mon-Fri), takes each week's last close,
    and checks for consecutive direction.
    """
    if s.n < weeks_required * 5:
        return False
 
    # Group bars by ISO week
    weekly_closes = {}
    for i in range(s.n):
        iso_year, iso_week, _ = s.timestamps[i].isocalendar()
        key = (iso_year, iso_week)
        weekly_closes[key] = s.closes[i]  # Last bar in each week
 
    weeks = sorted(weekly_closes.keys())
    if len(weeks) < weeks_required + 1:
        return False
 
    # Check last N weeks for consecutive direction
    recent_weeks = weeks[-(weeks_required + 1):]
    closes = [weekly_closes[w] for w in recent_weeks]
 
    consecutive = 0
    for i in range(1, len(closes)):
        if direction_long and closes[i] > closes[i - 1]:
            consecutive += 1
        elif not direction_long and closes[i] < closes[i - 1]:
            consecutive += 1
        else:
            consecutive = 0
 
    return consecutive >= weeks_required
 
 
# ==============================================================================
# JUICER LONG
# ==============================================================================
 
def _detect_juicer_long(s):
    """Juicer Trend Continuation — LONG.
    
    Identifies stocks with persistent upward momentum across multiple
    confirmation signals. This is not a breakout — it's joining a trend
    that is already working and showing no signs of stopping.
    """
    if s.n < 60 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    cur = s.closes[-1]
 
    # ── 1. ADX > 25 (confirmed trend) ──
    adx = _compute_adx(s, 14)
    if adx is None or adx < 25:
        return None
 
    # ── 2. Stacked SMAs: Price > 20 SMA > 50 SMA ──
    sma20 = float(np.mean(s.closes[-20:]))
    sma50 = float(np.mean(s.closes[-50:]))
 
    if not (cur > sma20 > sma50):
        return None
 
    # ── 3. Higher lows over last 10 bars ──
    if not _has_higher_lows(s, 10):
        return None
 
    # ── 4. Consecutive weekly closes — 3+ weeks higher ──
    if not _consecutive_weekly_trend(s, weeks_required=3, direction_long=True):
        return None
 
    # ── 5. Volume trend: 20d avg > 50d avg (increasing participation) ──
    vol_20 = float(np.mean(s.volumes[-20:]))
    vol_50 = float(np.mean(s.volumes[-50:]))
    if vol_50 > 0 and vol_20 < vol_50:
        return None
 
    # ── 6. Regime alignment (soft filter — boost confidence if aligned) ──
    regime_bonus = 0.0
    if s._regime == "trending_bull":
        regime_bonus = 0.08  # Strong boost for trend continuation in bull regime
 
    # ── Confidence ──
    # Base 0.60 + ADX strength bonus + regime bonus
    adx_bonus = min(0.10, (adx - 25) / 50 * 0.10)  # Up to +0.10 for ADX 75+
    vol_ratio = vol_20 / vol_50 if vol_50 > 0 else 1.0
    vol_bonus = min(0.05, (vol_ratio - 1.0) * 0.10)  # Up to +0.05 for 1.5x vol
 
    conf = 0.60 + adx_bonus + vol_bonus + regime_bonus
 
    # ── Entry / Stop / Targets ──
    entry = cur
    stop = cur - atr * 2.0  # 2 ATR trailing stop
 
    # T1: 2 ATR extension (first "add" point in real trading)
    t1 = round(entry + atr * 2.0, 2)
    # T2: 4 ATR extension (the big trend payoff)
    t2 = round(entry + atr * 4.0, 2)
 
    risk = entry - stop
    if risk <= 0:
        return None
 
    return _make(s, "Juicer Long", Bias.LONG,
                 entry, stop, t2, conf,
                 f"Juicer Long: ADX={adx:.0f}, {cur:.2f} > 20sma({sma20:.2f}) > 50sma({sma50:.2f}), "
                 f"vol {vol_ratio:.1f}x",
                 target_1=t1, target_2=t2,
                 trail_type="atr", trail_param=2.0,
                 position_splits=(0.25, 0.25, 0.50),  # Heavy trail weight
                 key_levels={"sma20": sma20, "sma50": sma50, "adx": round(adx, 1)})
 
 
# ==============================================================================
# JUICER SHORT
# ==============================================================================
 
def _detect_juicer_short(s):
    """Juicer Trend Continuation — SHORT.
    
    Identifies stocks with persistent downward momentum. Mirror of Juicer Long
    but with inverted criteria.
    """
    if s.n < 60 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    cur = s.closes[-1]
 
    # ── 1. ADX > 25 ──
    adx = _compute_adx(s, 14)
    if adx is None or adx < 25:
        return None
 
    # ── 2. Stacked SMAs: Price < 20 SMA < 50 SMA ──
    sma20 = float(np.mean(s.closes[-20:]))
    sma50 = float(np.mean(s.closes[-50:]))
 
    if not (cur < sma20 < sma50):
        return None
 
    # ── 3. Lower highs over last 10 bars ──
    if not _has_lower_highs(s, 10):
        return None
 
    # ── 4. Consecutive weekly closes — 3+ weeks lower ──
    if not _consecutive_weekly_trend(s, weeks_required=3, direction_long=False):
        return None
 
    # ── 5. Volume trend ──
    vol_20 = float(np.mean(s.volumes[-20:]))
    vol_50 = float(np.mean(s.volumes[-50:]))
    if vol_50 > 0 and vol_20 < vol_50:
        return None
 
    # ── 6. Regime ──
    regime_bonus = 0.0
    if s._regime == "trending_bear":
        regime_bonus = 0.08  # Strong boost for trend continuation in bear regime
 
    # ── Confidence ──
    adx_bonus = min(0.10, (adx - 25) / 50 * 0.10)
    vol_ratio = vol_20 / vol_50 if vol_50 > 0 else 1.0
    vol_bonus = min(0.05, (vol_ratio - 1.0) * 0.10)
 
    conf = 0.58 + adx_bonus + vol_bonus + regime_bonus
 
    # ── Entry / Stop / Targets ──
    entry = cur
    stop = cur + atr * 2.0
 
    t1 = round(entry - atr * 2.0, 2)
    t2 = round(entry - atr * 4.0, 2)
 
    risk = stop - entry
    if risk <= 0:
        return None
 
    return _make(s, "Juicer Short", Bias.SHORT,
                 entry, stop, t2, conf,
                 f"Juicer Short: ADX={adx:.0f}, {cur:.2f} < 20sma({sma20:.2f}) < 50sma({sma50:.2f}), "
                 f"vol {vol_ratio:.1f}x",
                 target_1=t1, target_2=t2,
                 trail_type="atr", trail_param=2.0,
                 position_splits=(0.25, 0.25, 0.50),
                 key_levels={"sma20": sma20, "sma50": sma50, "adx": round(adx, 1)})


# ==============================================================================
# CLASSICAL STRUCTURAL PATTERNS (16)
# ==============================================================================



# ==============================================================================
# 1. HEAD & SHOULDERS
# ==============================================================================
 
def _detect_head_and_shoulders(s):
    """Head & Shoulders — Bulkowski rank #1 bearish reversal.
    
    v2.2 changes:
      - ATR-based offsets (was $0.02)
      - Volume pattern scoring (declining L.shoulder → R.shoulder)
      - Breakout volume confirmation
      - Neckline slope classification (downsloping = more bearish)
      - T1 at 50% measured move, T2 at full measured move
      - Regime confidence multiplier
      - Min span between shoulders
    """
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 2:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    for i in range(len(s.zz_highs) - 2):
        ls, hd, rs = s.zz_highs[i], s.zz_highs[i+1], s.zz_highs[i+2]
 
        # Head must be highest
        if not (hd.price > ls.price and hd.price > rs.price):
            continue
 
        # Shoulder symmetry: within 3% or 0.5 ATR (whichever is larger)
        sym_tol = max(ls.price * 0.03, atr * 0.5)
        if abs(ls.price - rs.price) > sym_tol:
            continue
 
        # Min span: shoulders must be far enough apart
        if not _min_span_ok(ls.index, rs.index, s.timeframe):
            continue
 
        # Neckline from lows between shoulders
        lows_between = [l for l in s.zz_lows if ls.index < l.index < rs.index]
        if not lows_between:
            continue
        neckline = min(l.price for l in lows_between)
 
        # Price must be below neckline (breakdown confirmed)
        if s.closes[-1] >= neckline:
            continue
 
        # Breakout volume confirmation
        if not _volume_confirms_breakout(s, -1, 1.3):
            continue
 
        # Volume pattern scoring (declining is bullish for the SHORT signal)
        vol_bonus = _volume_pattern_hs(s, ls.index, hd.index, rs.index)
 
        # Neckline slope classification
        nl_lows = [l for l in s.zz_lows if ls.index < l.index < rs.index]
        slope_bonus = 0.0
        if len(nl_lows) >= 2:
            nl_slope = (nl_lows[-1].price - nl_lows[0].price) / max(1, nl_lows[-1].index - nl_lows[0].index)
            if nl_slope < 0:  # Downsloping neckline = more bearish
                slope_bonus = 0.05
 
        # Confidence
        sym_pct = abs(ls.price - rs.price) / ls.price
        conf = 0.60 + (1 - sym_pct / 0.03) * 0.10 + vol_bonus + slope_bonus
        conf *= _regime_confidence_mult(s, "breakout")
 
        # Entry/Stop/Targets
        off = _atr_offset(atr, 0.10)
        entry = neckline - off
        stop = rs.price + _atr_offset(atr, 0.10)
        measured_move = hd.price - neckline
        target_full = entry - measured_move
        target_75 = entry - measured_move * 0.75
 
        return _make(s, "Head & Shoulders", Bias.SHORT,
                     entry, stop, target_full, conf,
                     f"H&S: head@{hd.price:.2f}, neckline@{neckline:.2f}",
                     target_1=round(target_75, 2),
                     target_2=round(target_full, 2),
                     trail_type="atr", trail_param=2.0,
                     key_levels={"left_shoulder": ls.price, "head": hd.price,
                                 "right_shoulder": rs.price, "neckline": neckline})
    return None
 
 
# ==============================================================================
# 2. INVERSE HEAD & SHOULDERS
# ==============================================================================
 
def _detect_inverse_hs(s):
    """Inverse H&S — Bulkowski rank #1 bullish reversal (89% success).
    
    v2.2: Same improvements as H&S but for bullish reversal.
    """
    if len(s.zz_lows) < 3 or len(s.zz_highs) < 2:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    for i in range(len(s.zz_lows) - 2):
        ls, hd, rs = s.zz_lows[i], s.zz_lows[i+1], s.zz_lows[i+2]
 
        if not (hd.price < ls.price and hd.price < rs.price):
            continue
 
        sym_tol = max(ls.price * 0.03, atr * 0.5)
        if abs(ls.price - rs.price) > sym_tol:
            continue
 
        if not _min_span_ok(ls.index, rs.index, s.timeframe):
            continue
 
        highs_between = [h for h in s.zz_highs if ls.index < h.index < rs.index]
        if not highs_between:
            continue
        neckline = max(h.price for h in highs_between)
 
        if s.closes[-1] <= neckline:
            continue
 
        if not _volume_confirms_breakout(s, -1, 1.3):
            continue
 
        vol_bonus = _volume_pattern_hs(s, ls.index, hd.index, rs.index)
 
        sym_pct = abs(ls.price - rs.price) / ls.price
        conf = 0.62 + (1 - sym_pct / 0.03) * 0.10 + vol_bonus
        conf *= _regime_confidence_mult(s, "breakout")
 
        off = _atr_offset(atr, 0.10)
        entry = neckline + off
        stop = rs.price - _atr_offset(atr, 0.10)
        measured_move = neckline - hd.price
        target_full = entry + measured_move
        target_75 = entry + measured_move * 0.75
 
        return _make(s, "Inverse H&S", Bias.LONG,
                     entry, stop, target_full, conf,
                     f"Inv H&S: head@{hd.price:.2f}, neckline@{neckline:.2f}",
                     target_1=round(target_75, 2),
                     target_2=round(target_full, 2),
                     trail_type="atr", trail_param=2.0,
                     key_levels={"head": hd.price, "neckline": neckline})
    return None
 
 
# ==============================================================================
# 3. DOUBLE TOP
# ==============================================================================
 
def _detect_double_top(s):
    """Double Top — Bulkowski: 73% success, avg decline 19%.
    
    v2.2 changes:
      - Min span 15+ bars between peaks (was 5)
      - ATR-based tolerance (was 1.5% fixed)
      - Volume exhaustion on 2nd peak
      - Breakout volume on neckline break
      - T1 at 50% measured move
      - Second peak lower = confidence boost
    """
    if len(s.zz_highs) < 2:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    h1, h2 = s.zz_highs[-2], s.zz_highs[-1]
 
    # ATR-based tolerance (replace 1.5% fixed)
    tol = max(atr * 0.3, h1.price * 0.015)
    if abs(h1.price - h2.price) > tol:
        return None
 
    # Min span between peaks
    if not _min_span_ok(h1.index, h2.index, s.timeframe):
        return None
 
    valley = min(s.lows[h1.index:h2.index+1])
    if s.closes[-1] > valley:
        return None
 
    # Volume: 2nd peak should have less volume (exhaustion)
    vol_bonus = _volume_double_touch(s, h1.index, h2.index)
 
    # Breakout volume
    if not _volume_confirms_breakout(s, -1, 1.2):
        vol_bonus = max(0, vol_bonus - 0.04)
 
    # Second peak lower = stronger signal
    peak_bonus = 0.05 if h2.price < h1.price else 0.0
 
    top = max(h1.price, h2.price)
    conf = 0.60 + vol_bonus + peak_bonus
    conf *= _regime_confidence_mult(s, "breakout")
 
    off = _atr_offset(atr, 0.10)
    entry = valley - off
    stop = top + _atr_offset(atr, 0.10)
    measured_move = top - valley
    target_full = entry - measured_move
    target_75 = entry - measured_move * 0.75
 
    return _make(s, "Double Top", Bias.SHORT,
                 entry, stop, target_full, conf,
                 f"Double Top at {top:.2f}, valley {valley:.2f}",
                 target_1=round(target_75, 2),
                 target_2=round(target_full, 2),
                 trail_type="atr", trail_param=2.0,
                 key_levels={"top1": h1.price, "top2": h2.price, "valley": valley})
 
 
# ==============================================================================
# 4. DOUBLE BOTTOM
# ==============================================================================
 
def _detect_double_bottom(s):
    """Double Bottom — Bulkowski: 88% success, avg rise 50%.
    
    v2.2: Same improvements as Double Top but for bullish reversal.
    Second trough higher = stronger signal.
    """
    if len(s.zz_lows) < 2:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    l1, l2 = s.zz_lows[-2], s.zz_lows[-1]
 
    tol = max(atr * 0.3, l1.price * 0.015)
    if abs(l1.price - l2.price) > tol:
        return None
 
    if not _min_span_ok(l1.index, l2.index, s.timeframe):
        return None
 
    peak = max(s.highs[l1.index:l2.index+1])
    if s.closes[-1] < peak:
        return None
 
    vol_bonus = _volume_double_touch(s, l1.index, l2.index)
 
    if not _volume_confirms_breakout(s, -1, 1.2):
        vol_bonus = max(0, vol_bonus - 0.04)
 
    # Second trough higher = stronger
    trough_bonus = 0.05 if l2.price > l1.price else 0.0
 
    bot = min(l1.price, l2.price)
    conf = 0.62 + vol_bonus + trough_bonus
    conf *= _regime_confidence_mult(s, "breakout")
 
    off = _atr_offset(atr, 0.10)
    entry = peak + off
    stop = bot - _atr_offset(atr, 0.10)
    measured_move = peak - bot
    target_full = entry + measured_move
    target_75 = entry + measured_move * 0.75
 
    return _make(s, "Double Bottom", Bias.LONG,
                 entry, stop, target_full, conf,
                 f"Double Bottom at {bot:.2f}, peak {peak:.2f}",
                 target_1=round(target_75, 2),
                 target_2=round(target_full, 2),
                 trail_type="atr", trail_param=2.0,
                 key_levels={"bottom1": l1.price, "bottom2": l2.price, "peak": peak})
 
 
# ==============================================================================
# 5. TRIPLE TOP
# ==============================================================================
 
def _detect_triple_top(s):
    """Triple Top — Bulkowski: 75% success.
    
    v2.2: ATR-based tolerance, declining volume per touch, breakout volume.
    """
    if len(s.zz_highs) < 3:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    h1, h2, h3 = s.zz_highs[-3], s.zz_highs[-2], s.zz_highs[-1]
    prices = [h1.price, h2.price, h3.price]
    avg = np.mean(prices)
 
    # ATR-based tolerance (was 1.5% fixed)
    tol = max(atr * 0.3, avg * 0.015)
    if max(abs(p - avg) for p in prices) > tol:
        return None
 
    if not _min_span_ok(h1.index, h3.index, s.timeframe):
        return None
 
    valley = min(s.lows[h1.index:h3.index+1])
    if s.closes[-1] > valley:
        return None
 
    # Volume should decline on each successive touch
    vol_bonus = 0.0
    v1 = float(np.mean(s.volumes[max(0, h1.index-2):h1.index+3]))
    v2 = float(np.mean(s.volumes[max(0, h2.index-2):h2.index+3]))
    v3 = float(np.mean(s.volumes[max(0, h3.index-2):h3.index+3]))
    if v1 > 0 and v2 < v1 and v3 < v2:
        vol_bonus = 0.08
 
    if not _volume_confirms_breakout(s, -1, 1.2):
        vol_bonus = max(0, vol_bonus - 0.04)
 
    conf = 0.60 + vol_bonus
    conf *= _regime_confidence_mult(s, "breakout")
 
    off = _atr_offset(atr, 0.10)
    entry = valley - off
    stop = max(prices) + _atr_offset(atr, 0.10)
    measured_move = max(prices) - valley
    target_full = entry - measured_move
    target_75 = entry - measured_move * 0.75
 
    return _make(s, "Triple Top", Bias.SHORT,
                 entry, stop, target_full, conf,
                 f"Triple Top at {avg:.2f}, valley {valley:.2f}",
                 target_1=round(target_75, 2),
                 target_2=round(target_full, 2),
                 key_levels={"resistance": avg, "valley": valley})
 
 
# ==============================================================================
# 6. TRIPLE BOTTOM
# ==============================================================================
 
def _detect_triple_bottom(s):
    """Triple Bottom — Bulkowski: 79% success, avg rise 45%."""
    if len(s.zz_lows) < 3:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    l1, l2, l3 = s.zz_lows[-3], s.zz_lows[-2], s.zz_lows[-1]
    prices = [l1.price, l2.price, l3.price]
    avg = np.mean(prices)
 
    tol = max(atr * 0.3, avg * 0.015)
    if max(abs(p - avg) for p in prices) > tol:
        return None
 
    if not _min_span_ok(l1.index, l3.index, s.timeframe):
        return None
 
    peak = max(s.highs[l1.index:l3.index+1])
    if s.closes[-1] < peak:
        return None
 
    vol_bonus = 0.0
    v1 = float(np.mean(s.volumes[max(0, l1.index-2):l1.index+3]))
    v2 = float(np.mean(s.volumes[max(0, l2.index-2):l2.index+3]))
    v3 = float(np.mean(s.volumes[max(0, l3.index-2):l3.index+3]))
    if v1 > 0 and v2 < v1 and v3 < v2:
        vol_bonus = 0.08
 
    if not _volume_confirms_breakout(s, -1, 1.2):
        vol_bonus = max(0, vol_bonus - 0.04)
 
    conf = 0.62 + vol_bonus
    conf *= _regime_confidence_mult(s, "breakout")
 
    off = _atr_offset(atr, 0.10)
    entry = peak + off
    stop = min(prices) - _atr_offset(atr, 0.10)
    measured_move = peak - min(prices)
    target_full = entry + measured_move
    target_75 = entry + measured_move * 0.75
 
    return _make(s, "Triple Bottom", Bias.LONG,
                 entry, stop, target_full, conf,
                 f"Triple Bottom at {avg:.2f}, peak {peak:.2f}",
                 target_1=round(target_75, 2),
                 target_2=round(target_full, 2),
                 key_levels={"support": avg, "peak": peak})
 
 
# ==============================================================================
# 7. ASCENDING TRIANGLE
# ==============================================================================
 
def _detect_ascending_triangle(s):
    """Ascending Triangle — Bulkowski: 75% success, avg rise 35%.
    
    v2.2 changes:
      - Require 3+ touches per trendline (was 2)
      - Apex distance filter: only fire in first 75% of triangle width
      - Breakout volume confirmation
      - Volume declining during formation
      - T1 at 50% measured move
    """
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 3:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    # Need 3+ points for each line
    utl = fit_trendline(s.zz_highs[-4:] if len(s.zz_highs) >= 4 else s.zz_highs[-3:])
    ltl = fit_trendline(s.zz_lows[-4:] if len(s.zz_lows) >= 4 else s.zz_lows[-3:])
    if utl is None or ltl is None:
        return None
    if utl.num_points < 3 or ltl.num_points < 2:
        return None
 
    # Flat top + rising lows
    if not is_flat_line(utl, 0.15) or ltl.slope <= 0:
        return None
 
    res = utl.price_at(utl.end_index)
    if s.closes[-1] < res:
        return None
 
    # Apex distance filter: breakout should be in first 75% of triangle
    triangle_start = min(utl.start_index, ltl.start_index)
    triangle_width = s.n - 1 - triangle_start
    if triangle_width > 0:
        btc = utl.price_at(s.n - 1) - ltl.price_at(s.n - 1)
        btc_start = utl.price_at(triangle_start) - ltl.price_at(triangle_start)
        if btc_start > 0 and btc / btc_start < 0.25:
            return None  # Too close to apex
 
    # Volume: declining during formation
    if not _volume_declining_formation(s, triangle_start, s.n - 1):
        pass  # Don't reject, just no bonus
 
    # Breakout volume
    vol_bonus = 0.05 if _volume_confirms_breakout(s, -1, 1.3) else 0.0
 
    sup = s.zz_lows[-1].price
    conf = 0.58 + vol_bonus
    conf *= _regime_confidence_mult(s, "breakout")
 
    off = _atr_offset(atr, 0.10)
    entry = res + off
    stop = sup - _atr_offset(atr, 0.10)
    measured_move = res - sup
    target_full = entry + measured_move
    target_75 = entry + measured_move * 0.75
 
    return _make(s, "Ascending Triangle", Bias.LONG,
                 entry, stop, target_full, conf,
                 f"Asc Triangle: flat top {res:.2f}, rising lows",
                 target_1=round(target_75, 2),
                 target_2=round(target_full, 2),
                 trail_type="atr", trail_param=2.0,
                 key_levels={"resistance": res, "support": sup})
 
 
# ==============================================================================
# 8. DESCENDING TRIANGLE
# ==============================================================================
 
def _detect_descending_triangle(s):
    """Descending Triangle — Bulkowski: 72% success."""
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 3:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    ltl = fit_trendline(s.zz_lows[-4:] if len(s.zz_lows) >= 4 else s.zz_lows[-3:])
    utl = fit_trendline(s.zz_highs[-4:] if len(s.zz_highs) >= 4 else s.zz_highs[-3:])
    if utl is None or ltl is None:
        return None
    if ltl.num_points < 3 or utl.num_points < 2:
        return None
 
    if not is_flat_line(ltl, 0.15) or utl.slope >= 0:
        return None
 
    sup = ltl.price_at(ltl.end_index)
    if s.closes[-1] > sup:
        return None
 
    triangle_start = min(utl.start_index, ltl.start_index)
 
    vol_bonus = 0.05 if _volume_confirms_breakout(s, -1, 1.3) else 0.0
 
    res = s.zz_highs[-1].price
    conf = 0.56 + vol_bonus
    conf *= _regime_confidence_mult(s, "breakout")
 
    off = _atr_offset(atr, 0.10)
    entry = sup - off
    stop = res + _atr_offset(atr, 0.10)
    measured_move = res - sup
    target_full = entry - measured_move
    target_75 = entry - measured_move * 0.75
 
    return _make(s, "Descending Triangle", Bias.SHORT,
                 entry, stop, target_full, conf,
                 f"Desc Triangle: flat bottom {sup:.2f}, falling highs",
                 target_1=round(target_75, 2),
                 target_2=round(target_full, 2),
                 key_levels={"support": sup, "resistance": res})
 
 
# ==============================================================================
# 9. SYMMETRICAL TRIANGLE
# ==============================================================================
 
def _detect_symmetrical_triangle(s):
    """Symmetrical Triangle — Bulkowski: only 54% success (weakest triangle).
    
    v2.2: Lower confidence (0.50 base), stricter volume/apex requirements.
    """
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 3:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    utl = fit_trendline(s.zz_highs[-4:] if len(s.zz_highs) >= 4 else s.zz_highs[-3:])
    ltl = fit_trendline(s.zz_lows[-4:] if len(s.zz_lows) >= 4 else s.zz_lows[-3:])
    if utl is None or ltl is None:
        return None
 
    # Must be converging: upper falling, lower rising
    if utl.slope >= 0 or ltl.slope <= 0:
        return None
 
    up = utl.price_at(s.n-1)
    lo = ltl.price_at(s.n-1)
    if up <= lo:
        return None
 
    rng = up - lo
    cur = s.closes[-1]
 
    vol_bonus = 0.05 if _volume_confirms_breakout(s, -1, 1.3) else 0.0
 
    # Lower base confidence (54% pattern)
    conf = 0.50 + vol_bonus
    conf *= _regime_confidence_mult(s, "breakout")
 
    off = _atr_offset(atr, 0.10)
 
    if cur > up:
        entry = up + off
        stop = lo - _atr_offset(atr, 0.15)
        target_full = entry + rng
        target_half = entry + rng * 0.5
        return _make(s, "Symmetrical Triangle", Bias.LONG,
                     entry, stop, target_full, conf,
                     "Sym Triangle breakout above",
                     target_1=round(target_half, 2),
                     target_2=round(target_full, 2),
                     key_levels={"upper": up, "lower": lo})
    elif cur < lo:
        entry = lo - off
        stop = up + _atr_offset(atr, 0.15)
        target_full = entry - rng
        target_half = entry - rng * 0.5
        return _make(s, "Symmetrical Triangle", Bias.SHORT,
                     entry, stop, target_full, conf,
                     "Sym Triangle breakdown below",
                     target_1=round(target_half, 2),
                     target_2=round(target_full, 2),
                     key_levels={"upper": up, "lower": lo})
    return None
 
 
# ==============================================================================
# 10. BULL FLAG
# ==============================================================================
 
def _detect_bull_flag(s):
    """Bull Flag — Bulkowski: 67% for standard, 85% for high-tight.
    
    v2.2 changes:
      - Pole velocity: must gain ≥ 1.5 ATR in ≤ 10 bars
      - Flag max 15 bars (was 30)
      - Flag tightness: range < 40% of pole (was 50%)
      - Volume: pole vol > flag vol
      - Breakout volume confirmation
      - Classify "tight" vs "standard" flag
    """
    if len(s.zz_lows) < 1 or len(s.zz_highs) < 1 or s.current_atr <= 0:
        return None
    atr = s.current_atr
 
    for li in range(len(s.zz_lows)):
        lo = s.zz_lows[li]
        post = [h for h in s.zz_highs if h.index > lo.index]
        if not post:
            continue
        hi = post[0]
 
        pole_size = hi.price - lo.price
        if pole_size < atr * 2.0:
            continue
 
        # Pole velocity: must complete in ≤ 10 bars
        pole_bars = hi.index - lo.index
        if pole_bars > 15 or pole_bars < 2:
            continue
 
        fs = hi.index
        if fs >= s.n - 3:
            continue
 
        flag = s.bars[fs:]
        if len(flag) < 3 or len(flag) > 15:  # Tightened from 30
            continue
 
        fl = min(b.low for b in flag)
        flag_range = max(b.high for b in flag) - fl
 
        # Flag tightness: range < 40% of pole (was 50%)
        if (hi.price - fl) / pole_size > 0.50:
            continue
 
        if s.closes[-1] <= hi.price:
            continue
 
        # Volume: pole avg should exceed flag avg
        pole_vol = float(np.mean(s.volumes[lo.index:hi.index+1]))
        flag_vol = float(np.mean([b.volume for b in flag]))
        vol_bonus = 0.05 if (pole_vol > flag_vol * 1.3) else 0.0
 
        # Breakout volume
        vol_bonus += 0.05 if _volume_confirms_breakout(s, -1, 1.3) else 0.0
 
        # Classify tight vs standard
        tightness = flag_range / pole_size if pole_size > 0 else 1.0
        is_tight = tightness < 0.20
        tight_bonus = 0.10 if is_tight else 0.0
 
        conf = 0.55 + vol_bonus + tight_bonus
        conf *= _regime_confidence_mult(s, "momentum")
 
        off = _atr_offset(atr, 0.05)
        entry = hi.price + off
        stop = fl - _atr_offset(atr, 0.10)
        target_full = entry + pole_size
        target_half = entry + pole_size * 0.75
        pct = pole_size / lo.price
 
        label = "Tight" if is_tight else "Standard"
        return _make(s, "Bull Flag", Bias.LONG,
                     entry, stop, target_full, conf,
                     f"Bull Flag ({label}): {pct:.1%} pole, {len(flag)}-bar flag",
                     target_1=round(target_half, 2),
                     target_2=round(target_full, 2),
                     trail_type="ema9", trail_param=9.0,
                     key_levels={"pole_hi": hi.price, "pole_lo": lo.price, "flag_low": fl})
    return None
 
 
# ==============================================================================
# 11. BEAR FLAG
# ==============================================================================
 
def _detect_bear_flag(s):
    """Bear Flag — Bulkowski: 65% success. Mirror of Bull Flag."""
    if len(s.zz_highs) < 1 or len(s.zz_lows) < 1 or s.current_atr <= 0:
        return None
    atr = s.current_atr
 
    for hi_idx in range(len(s.zz_highs)):
        hi = s.zz_highs[hi_idx]
        post = [l for l in s.zz_lows if l.index > hi.index]
        if not post:
            continue
        lo = post[0]
 
        pole_size = hi.price - lo.price
        if pole_size < atr * 2.0:
            continue
 
        pole_bars = lo.index - hi.index
        if pole_bars > 15 or pole_bars < 2:
            continue
 
        fs = lo.index
        if fs >= s.n - 3:
            continue
 
        flag = s.bars[fs:]
        if len(flag) < 3 or len(flag) > 15:
            continue
 
        fh = max(b.high for b in flag)
        if (fh - lo.price) / pole_size > 0.50:
            continue
 
        if s.closes[-1] >= lo.price:
            continue
 
        pole_vol = float(np.mean(s.volumes[hi.index:lo.index+1]))
        flag_vol = float(np.mean([b.volume for b in flag]))
        vol_bonus = 0.05 if (pole_vol > flag_vol * 1.3) else 0.0
        vol_bonus += 0.05 if _volume_confirms_breakout(s, -1, 1.3) else 0.0
 
        conf = 0.53 + vol_bonus
        conf *= _regime_confidence_mult(s, "momentum")
 
        off = _atr_offset(atr, 0.05)
        entry = lo.price - off
        stop = fh + _atr_offset(atr, 0.10)
        target_full = entry - pole_size
        target_half = entry - pole_size * 0.75
        pct = pole_size / hi.price
 
        return _make(s, "Bear Flag", Bias.SHORT,
                     entry, stop, target_full, conf,
                     f"Bear Flag: {pct:.1%} pole, {len(flag)}-bar flag",
                     target_1=round(target_half, 2),
                     target_2=round(target_full, 2),
                     trail_type="ema9", trail_param=9.0,
                     key_levels={"pole_hi": hi.price, "flag_high": fh})
    return None
 
 
# ==============================================================================
# 12. PENNANT
# ==============================================================================
 
def _detect_pennant(s):
    """Pennant — Bulkowski: only 46% success, 7% avg profit.
    
    v2.2: Confidence lowered to 0.40 base. This pattern will rarely
    pass the 45-score filter, which is appropriate given its poor stats.
    Volume contraction required during formation.
    """
    return None # Disable Pennant for now due to poor performance and complexity
    if len(s.zz_lows) < 1 or len(s.zz_highs) < 1 or s.n < 20:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    for li in range(len(s.zz_lows)):
        lo = s.zz_lows[li]
        post = [h for h in s.zz_highs if h.index > lo.index]
        if not post:
            continue
        hi = post[0]
        pct = (hi.price - lo.price) / lo.price
        if pct < 0.03:
            continue
 
        fs = hi.index
        if fs >= s.n - 4:
            continue
 
        pn_bars = s.bars[fs:min(fs+15, s.n)]
        if len(pn_bars) < 3:
            continue
 
        pn_highs = [b.high for b in pn_bars]
        pn_lows = [b.low for b in pn_bars]
        h_slope = (pn_highs[-1] - pn_highs[0]) / len(pn_highs) if len(pn_highs) > 1 else 0
        l_slope = (pn_lows[-1] - pn_lows[0]) / len(pn_lows) if len(pn_lows) > 1 else 0
        if h_slope >= 0 or l_slope <= 0:
            continue
 
        if s.closes[-1] <= max(pn_highs):
            continue
 
        # Volume must contract during pennant
        if not _volume_declining_formation(s, fs, min(fs + len(pn_bars), s.n)):
            continue
 
        # Low confidence — Bulkowski says this is a bad pattern
        conf = 0.40
        if _volume_confirms_breakout(s, -1, 1.5):
            conf += 0.05
 
        off = _atr_offset(atr, 0.10)
        entry = max(pn_highs) + off
        stop = min(pn_lows) - _atr_offset(atr, 0.15)
        pole_size = hi.price - lo.price
        target_full = entry + pole_size
        target_half = entry + pole_size * 0.75
 
        return _make(s, "Pennant", Bias.LONG,
                     entry, stop, target_full, conf,
                     f"Pennant: {pct:.1%} pole, converging consolidation",
                     target_1=round(target_half, 2),
                     target_2=round(target_full, 2),
                     key_levels={"pole_hi": hi.price})
    return None
 
 
# ==============================================================================
# 13. CUP & HANDLE
# ==============================================================================
 
def _detect_cup_and_handle(s):
    """Cup & Handle — O'Neil: 68% success.
    
    v2.2 changes:
      - Handle must be in upper 1/3 of cup
      - Handle should slope slightly downward
      - Volume decline in handle (already existed)
      - Breakout volume (already existed)
      - Min 30-bar cup span (already existed)
    """
    if len(s.zz_lows) < 1 or s.n < 40:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    cl = s.zz_lows[-1]
    lr = max(s.highs[:cl.index]) if cl.index > 5 else None
    rr = max(s.highs[cl.index:]) if cl.index < s.n - 5 else None
    if lr is None or rr is None:
        return None
 
    rim = min(lr, rr)
    depth = rim - cl.price
    if depth <= 0:
        return None
 
    # Cup must span at least 30 bars
    if cl.index < 15 or s.n - cl.index < 15:
        return None
 
    handle = s.bars[-min(10, s.n):]
    hl = min(b.low for b in handle)
    hh = max(b.high for b in handle)
    ret = (rim - hl) / depth
    if ret > 0.50 or ret < 0.10 or s.closes[-1] < rim:
        return None
 
    # Handle must be in upper 1/3 of cup
    upper_third = cl.price + depth * (2/3)
    if hl < upper_third:
        return None
 
    # Handle should slope slightly downward (not up)
    handle_slope = handle[-1].close - handle[0].close
    if handle_slope > atr * 0.3:  # Allow slight up, reject strong up
        return None
 
    # Volume decline in handle
    handle_vol = float(np.mean([b.volume for b in handle]))
    cup_vol = float(np.mean(s.volumes[max(0, cl.index-10):cl.index+10]))
    if cup_vol > 0 and handle_vol > cup_vol * 1.2:
        return None
 
    # Breakout volume
    avg_vol = float(np.mean(s.volumes[-20:]))
    vol_bonus = 0.05 if (avg_vol > 0 and s.volumes[-1] >= avg_vol * 1.3) else 0.0
 
    conf = 0.60 + vol_bonus
    conf *= _regime_confidence_mult(s, "breakout")
 
    off = _atr_offset(atr, 0.10)
    entry = rim + off
    stop = hl - _atr_offset(atr, 0.10)
    target_full = entry + depth
    target_75 = entry + depth * 0.75
 
    return _make(s, "Cup & Handle", Bias.LONG,
                 entry, stop, target_full, conf,
                 f"Cup & Handle: rim {rim:.2f}, depth {depth:.2f}",
                 target_1=round(target_75, 2),
                 target_2=round(target_full, 2),
                 trail_type="ema9", trail_param=9.0,
                 key_levels={"rim": rim, "cup_low": cl.price, "handle_low": hl})
 
 
# ==============================================================================
# 14. RECTANGLE
# ==============================================================================
 
def _detect_rectangle(s):
    """Rectangle — Bulkowski: 70% success, one of most profitable patterns.
    
    v2.2: Require 3+ touches on both support and resistance.
    Volume contraction during formation. Breakout volume required.
    """
    if s.n < 20 or len(s.sr_levels) < 2:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    res_levels = [l for l in s.sr_levels if l.level_type in ("resistance", "both")]
    sup_levels = [l for l in s.sr_levels if l.level_type in ("support", "both")]
    if not res_levels or not sup_levels:
        return None
 
    res = res_levels[0].price
    sup = sup_levels[0].price
    if res <= sup:
        return None
 
    # Require 3+ touches on each boundary
    res_touches = res_levels[0].touches
    sup_touches = sup_levels[0].touches if sup_levels else 0
    if res_touches < 3 or sup_touches < 3:
        return None
 
    rng = res - sup
    cur = s.closes[-1]
 
    vol_bonus = 0.05 if _volume_confirms_breakout(s, -1, 1.3) else 0.0
 
    # Touch bonus for very well-defined rectangles
    touch_bonus = min(0.08, (res_touches + sup_touches - 6) * 0.02)
 
    conf = 0.55 + vol_bonus + touch_bonus
    conf *= _regime_confidence_mult(s, "breakout")
 
    off = _atr_offset(atr, 0.10)
 
    if cur > res + rng * 0.01:
        entry = res + off
        stop = sup - _atr_offset(atr, 0.15)
        target_full = entry + rng
        target_half = entry + rng * 0.75
        return _make(s, "Rectangle", Bias.LONG,
                     entry, stop, target_full, conf,
                     f"Rectangle breakout above {res:.2f} ({res_touches}+{sup_touches} touches)",
                     target_1=round(target_half, 2),
                     target_2=round(target_full, 2),
                     key_levels={"resistance": res, "support": sup})
    elif cur < sup - rng * 0.01:
        entry = sup - off
        stop = res + _atr_offset(atr, 0.15)
        target_full = entry - rng
        target_half = entry - rng * 0.75
        return _make(s, "Rectangle", Bias.SHORT,
                     entry, stop, target_full, conf,
                     f"Rectangle breakdown below {sup:.2f} ({res_touches}+{sup_touches} touches)",
                     target_1=round(target_half, 2),
                     target_2=round(target_full, 2),
                     key_levels={"resistance": res, "support": sup})
    return None
 
 
# ==============================================================================
# 15. RISING WEDGE
# ==============================================================================
 
def _detect_rising_wedge(s):
    """Rising Wedge — Bulkowski: 69% success but 24% failure rate (high).
    
    v2.2: Lower confidence (0.55 base), require 3 touches per line,
    volume declining. Uses widest point for target (v2 fix kept).
    """
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 3:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    u = fit_trendline(s.zz_highs[-4:] if len(s.zz_highs) >= 4 else s.zz_highs[-3:])
    l = fit_trendline(s.zz_lows[-4:] if len(s.zz_lows) >= 4 else s.zz_lows[-3:])
    if u is None or l is None:
        return None
 
    # Both rising, upper slower than lower (converging upward)
    if not (u.slope > 0 and l.slope > 0 and u.slope < l.slope):
        return None
 
    lp = l.price_at(s.n-1)
    up = u.price_at(s.n-1)
    if s.closes[-1] > lp:
        return None
 
    # Volume declining during formation
    vol_bonus = 0.03 if _volume_declining_formation(s, u.start_index, s.n - 1) else 0.0
 
    # Widest point for target (v2 fix)
    start_idx = min(u.start_index, l.start_index)
    wide_upper = u.price_at(start_idx)
    wide_lower = l.price_at(start_idx)
    widest = abs(wide_upper - wide_lower)
    if widest <= 0:
        widest = abs(up - lp)
 
    # Lower base confidence due to high failure rate
    conf = 0.52 + vol_bonus
    conf *= _regime_confidence_mult(s, "breakout")
 
    off = _atr_offset(atr, 0.10)
    entry = lp - off
    stop = up + _atr_offset(atr, 0.10)
    target_full = entry - widest
    target_half = entry - widest * 0.75
 
    return _make(s, "Rising Wedge", Bias.SHORT,
                 entry, stop, target_full, conf,
                 "Rising Wedge breakdown",
                 target_1=round(target_half, 2),
                 target_2=round(target_full, 2),
                 key_levels={"upper": up, "lower": lp})
 
 
# ==============================================================================
# 16. FALLING WEDGE
# ==============================================================================
 
def _detect_falling_wedge(s):
    """Falling Wedge — Bulkowski: 68% success, 26% failure rate (high).
    
    v2.2: Lower confidence, require touches, volume declining.
    """
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 3:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    u = fit_trendline(s.zz_highs[-4:] if len(s.zz_highs) >= 4 else s.zz_highs[-3:])
    l = fit_trendline(s.zz_lows[-4:] if len(s.zz_lows) >= 4 else s.zz_lows[-3:])
    if u is None or l is None:
        return None
 
    if not (u.slope < 0 and l.slope < 0 and u.slope > l.slope):
        return None
 
    up = u.price_at(s.n-1)
    lp = l.price_at(s.n-1)
    if s.closes[-1] < up:
        return None
 
    vol_bonus = 0.03 if _volume_declining_formation(s, u.start_index, s.n - 1) else 0.0
 
    start_idx = min(u.start_index, l.start_index)
    wide_upper = u.price_at(start_idx)
    wide_lower = l.price_at(start_idx)
    widest = abs(wide_upper - wide_lower)
    if widest <= 0:
        widest = abs(up - lp)
 
    conf = 0.52 + vol_bonus
    conf *= _regime_confidence_mult(s, "breakout")
 
    off = _atr_offset(atr, 0.10)
    entry = up + off
    stop = lp - _atr_offset(atr, 0.10)
    target_full = entry + widest
    target_half = entry + widest * 0.75
 
    return _make(s, "Falling Wedge", Bias.LONG,
                 entry, stop, target_full, conf,
                 "Falling Wedge breakout",
                 target_1=round(target_half, 2),
                 target_2=round(target_full, 2),
                 key_levels={"upper": up, "lower": lp})

# ==============================================================================
# CANDLESTICK PATTERNS (10) — all now require _candle_context_ok()
# ==============================================================================

# ==============================================================================
# CANDLESTICK: Helper — confirmation candle check
# ==============================================================================

def _confirmation_candle_bonus(s, direction_long: bool) -> float:
    """Check if the LAST bar confirms the reversal direction.
    
    Nison's "confirmation candle" concept: the bar after the signal should
    close in the expected direction. Returns 0.08 confidence bonus if confirmed.
    
    We check the last bar in the series since by the time we're analyzing,
    the signal bar is s.bars[-2] and the current bar is s.bars[-1].
    """
    if s.n < 3:
        return 0.0

    cur = s.bars[-1]
    if direction_long:
        # Confirmation = current bar closes above prior bar's close
        if cur.close > s.bars[-2].close and _is_green(cur):
            return 0.08
    else:
        if cur.close < s.bars[-2].close and _is_red(cur):
            return 0.08

    return 0.0


# ==============================================================================
# 1. BULLISH ENGULFING
# ==============================================================================

def _detect_bullish_engulfing(s):
    """Bullish Engulfing — Nison: one of the most reliable candle patterns.
    
    v2.2 changes:
      - ATR-based stop (was $0.02)
      - Confirmation candle confidence boost (+0.08)
      - T1=1R, T2=nearest S/R or 2R
    """
    if s.n < 3:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    prev, cur = s.bars[-2], s.bars[-1]
    if not (_is_red(prev) and _is_green(cur)):
        return None
    if _body(cur) <= _body(prev) * 1.1:
        return None
    if cur.open > prev.close or cur.close < prev.open:
        return None
    if not _candle_context_ok(s, "bullish"):
        return None

    conf_bonus = _confirmation_candle_bonus(s, direction_long=True)
    conf = 0.58 + conf_bonus

    entry = cur.close
    stop = min(prev.low, cur.low) - _atr_offset(atr, 0.15)
    risk = entry - stop
    if risk <= 0:
        return None

    # T1=1R, T2=nearest S/R or 2R
    t1 = round(entry + risk, 2)
    sr_target = _nearest_sr_target(s, entry, bias_long=True, min_rr=1.5)
    t2 = sr_target if sr_target > 0 else round(entry + risk * 2, 2)

    return _make(s, "Bullish Engulfing", Bias.LONG,
                 entry, stop, t2, conf,
                 f"Bullish Engulfing at {entry:.2f}",
                 target_1=t1, target_2=t2,
                 trail_type="vwap", trail_param=0.0)


# ==============================================================================
# 2. BEARISH ENGULFING
# ==============================================================================

def _detect_bearish_engulfing(s):
    """Bearish Engulfing — mirror of bullish."""
    if s.n < 3:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    prev, cur = s.bars[-2], s.bars[-1]
    if not (_is_green(prev) and _is_red(cur)):
        return None
    if _body(cur) <= _body(prev) * 1.1:
        return None
    if cur.open < prev.close or cur.close > prev.open:
        return None
    if not _candle_context_ok(s, "bearish"):
        return None

    conf_bonus = _confirmation_candle_bonus(s, direction_long=False)
    conf = 0.58 + conf_bonus

    entry = cur.close
    stop = max(prev.high, cur.high) + _atr_offset(atr, 0.15)
    risk = stop - entry
    if risk <= 0:
        return None

    t1 = round(entry - risk, 2)
    sr_target = _nearest_sr_target(s, entry, bias_long=False, min_rr=1.5)
    t2 = sr_target if sr_target > 0 else round(entry - risk * 2, 2)

    return _make(s, "Bearish Engulfing", Bias.SHORT,
                 entry, stop, t2, conf,
                 f"Bearish Engulfing at {entry:.2f}",
                 target_1=t1, target_2=t2,
                 trail_type="vwap", trail_param=0.0)


# ==============================================================================
# 3. MORNING STAR
# ==============================================================================

def _detect_morning_star(s):
    """Morning Star — three-bar bullish reversal.
    
    v2.2 changes:
      - Star bar volume < bars 1 and 3 (indecision)
      - ATR-based stop at star's low
      - Scaled exits
    """
    if s.n < 4:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    b1, star, b3 = s.bars[-3], s.bars[-2], s.bars[-1]

    if not _is_red(b1):
        return None
    if _body(star) > _body(b1) * 0.30:
        return None
    if _body(b3) < _body(b1) * 0.50:
        return None
    if not _is_green(b3):
        return None
    if star.high > b1.open:
        return None
    if not _candle_context_ok(s, "bullish"):
        return None

    # Star bar should have lower volume than bars 1 and 3
    vol_bonus = 0.0
    if s.n >= 4:
        if star.volume < b1.volume and star.volume < b3.volume:
            vol_bonus = 0.05

    conf = 0.60 + vol_bonus

    entry = b3.close
    stop = star.low - _atr_offset(atr, 0.15)
    risk = entry - stop
    if risk <= 0:
        return None

    t1 = round(entry + risk, 2)
    sr_target = _nearest_sr_target(s, entry, bias_long=True, min_rr=1.5)
    t2 = sr_target if sr_target > 0 else round(entry + risk * 2, 2)

    return _make(s, "Morning Star", Bias.LONG,
                 entry, stop, t2, conf,
                 f"Morning Star at {entry:.2f}",
                 target_1=t1, target_2=t2,
                 trail_type="vwap", trail_param=0.0)


# ==============================================================================
# 4. EVENING STAR
# ==============================================================================

def _detect_evening_star(s):
    """Evening Star — three-bar bearish reversal. Mirror of Morning Star."""
    if s.n < 4:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    b1, star, b3 = s.bars[-3], s.bars[-2], s.bars[-1]

    if not _is_green(b1):
        return None
    if _body(star) > _body(b1) * 0.30:
        return None
    if _body(b3) < _body(b1) * 0.50:
        return None
    if not _is_red(b3):
        return None
    if star.low < b1.close:
        return None
    if not _candle_context_ok(s, "bearish"):
        return None

    vol_bonus = 0.0
    if s.n >= 4:
        if star.volume < b1.volume and star.volume < b3.volume:
            vol_bonus = 0.05

    conf = 0.60 + vol_bonus

    entry = b3.close
    stop = star.high + _atr_offset(atr, 0.15)
    risk = stop - entry
    if risk <= 0:
        return None

    t1 = round(entry - risk, 2)
    sr_target = _nearest_sr_target(s, entry, bias_long=False, min_rr=1.5)
    t2 = sr_target if sr_target > 0 else round(entry - risk * 2, 2)

    return _make(s, "Evening Star", Bias.SHORT,
                 entry, stop, t2, conf,
                 f"Evening Star at {entry:.2f}",
                 target_1=t1, target_2=t2,
                 trail_type="vwap", trail_param=0.0)


# ==============================================================================
# 5. HAMMER
# ==============================================================================

def _detect_hammer(s):
    """Hammer — bullish reversal candle.
    
    v2.2 changes:
      - ATR-based stop
      - Bullish body bonus (close at high = stronger)
      - Scaled exits
    """
    if s.n < 2:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    cur = s.bars[-1]
    body = _body(cur)
    rng = _range(cur)
    if rng <= 0 or body <= 0:
        return None

    lower_shadow = min(cur.open, cur.close) - cur.low
    upper_shadow = cur.high - max(cur.open, cur.close)

    if lower_shadow < body * 2.0:
        return None
    if upper_shadow > body * 0.5:
        return None
    if not _candle_context_ok(s, "bullish"):
        return None

    # Bullish body bonus: close at high is stronger
    body_bonus = 0.04 if _is_green(cur) else 0.0

    conf = 0.55 + body_bonus

    entry = cur.close
    stop = cur.low - _atr_offset(atr, 0.15)
    risk = entry - stop
    if risk <= 0:
        return None

    t1 = round(entry + risk, 2)
    t2 = round(entry + risk * 2, 2)

    return _make(s, "Hammer", Bias.LONG,
                 entry, stop, t2, conf,
                 f"Hammer at {entry:.2f}",
                 target_1=t1, target_2=t2,
                 trail_type="vwap", trail_param=0.0)


# ==============================================================================
# 6. SHOOTING STAR
# ==============================================================================

def _detect_shooting_star(s):
    """Shooting Star — bearish reversal candle. Mirror of Hammer."""
    if s.n < 2:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    cur = s.bars[-1]
    body = _body(cur)
    rng = _range(cur)
    if rng <= 0 or body <= 0:
        return None

    upper_shadow = cur.high - max(cur.open, cur.close)
    lower_shadow = min(cur.open, cur.close) - cur.low

    if upper_shadow < body * 2.0:
        return None
    if lower_shadow > body * 0.5:
        return None
    if not _candle_context_ok(s, "bearish"):
        return None

    body_bonus = 0.04 if _is_red(cur) else 0.0
    conf = 0.54 + body_bonus

    entry = cur.close
    stop = cur.high + _atr_offset(atr, 0.15)
    risk = stop - entry
    if risk <= 0:
        return None

    t1 = round(entry - risk, 2)
    t2 = round(entry - risk * 2, 2)

    return _make(s, "Shooting Star", Bias.SHORT,
                 entry, stop, t2, conf,
                 f"Shooting Star at {entry:.2f}",
                 target_1=t1, target_2=t2,
                 trail_type="vwap", trail_param=0.0)


# ==============================================================================
# 7. DOJI
# ==============================================================================

def _detect_doji(s):
    """Doji — indecision candle, only useful at extremes.
    
    v2.2: Minimal changes. ATR-based stop. Confidence stays low (0.48).
    Current implementation is appropriate per audit.
    """
    if s.n < 10:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    cur = s.bars[-1]
    body = _body(cur)
    rng = _range(cur)
    if rng <= 0:
        return None

    if body / rng > 0.10:
        return None

    # Must be at an extreme (near recent high or low)
    recent_high = max(s.highs[-10:])
    recent_low = min(s.lows[-10:])
    near_high = cur.high >= recent_high - atr * 0.3
    near_low = cur.low <= recent_low + atr * 0.3

    if not (near_high or near_low):
        return None

    conf = 0.48

    if near_high:
        entry = cur.close
        stop = cur.high + _atr_offset(atr, 0.15)
        risk = stop - entry
        if risk <= 0:
            return None
        t1 = round(entry - risk, 2)
        t2 = round(entry - risk * 2, 2)
        return _make(s, "Doji", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"Doji at resistance {entry:.2f}",
                     target_1=t1, target_2=t2)
    else:
        entry = cur.close
        stop = cur.low - _atr_offset(atr, 0.15)
        risk = entry - stop
        if risk <= 0:
            return None
        t1 = round(entry + risk, 2)
        t2 = round(entry + risk * 2, 2)
        return _make(s, "Doji", Bias.LONG,
                     entry, stop, t2, conf,
                     f"Doji at support {entry:.2f}",
                     target_1=t1, target_2=t2)


# ==============================================================================
# 8. DRAGONFLY DOJI
# ==============================================================================

def _detect_dragonfly_doji(s):
    """Dragonfly Doji — long lower shadow, no upper shadow. Bullish at support.
    
    v2.2: ATR-based stop. Confidence 0.52 (kept — appropriate per audit).
    """
    if s.n < 10:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    cur = s.bars[-1]
    body = _body(cur)
    rng = _range(cur)
    if rng <= 0:
        return None

    if body / rng > 0.10:
        return None

    lower_shadow = min(cur.open, cur.close) - cur.low
    upper_shadow = cur.high - max(cur.open, cur.close)
    if lower_shadow < rng * 0.60:
        return None
    if upper_shadow > rng * 0.10:
        return None

    recent_low = min(s.lows[-10:])
    if cur.low > recent_low + atr * 0.3:
        return None

    conf = 0.52
    entry = cur.close
    stop = cur.low - _atr_offset(atr, 0.15)
    risk = entry - stop
    if risk <= 0:
        return None

    t1 = round(entry + risk, 2)
    t2 = round(entry + risk * 2, 2)

    return _make(s, "Dragonfly Doji", Bias.LONG,
                 entry, stop, t2, conf,
                 f"Dragonfly Doji at {entry:.2f}",
                 target_1=t1, target_2=t2)


# ==============================================================================
# 9. THREE WHITE SOLDIERS
# ==============================================================================

def _detect_three_white_soldiers(s):
    """Three White Soldiers — 3-bar bullish momentum.
    
    v2.2 changes:
      - Volume must INCREASE on each successive bar
      - ATR-based stop
      - Scaled exits
      - Context filter kept but note: this is also a continuation pattern
    """
    if s.n < 4:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    b1, b2, b3 = s.bars[-3], s.bars[-2], s.bars[-1]

    # All three green
    if not (_is_green(b1) and _is_green(b2) and _is_green(b3)):
        return None

    # Each opens within prior body
    if b2.open < b1.open or b2.open > b1.close:
        return None
    if b3.open < b2.open or b3.open > b2.close:
        return None

    # Small upper shadows (< 40% of body)
    for b in [b1, b2, b3]:
        body = _body(b)
        if body <= 0:
            return None
        upper_shadow = b.high - b.close
        if upper_shadow > body * 0.40:
            return None

    # Volume should increase on each bar
    vol_bonus = 0.0
    if b1.volume > 0 and b2.volume > b1.volume and b3.volume > b2.volume:
        vol_bonus = 0.08

    # Context check (may not require strict S/R for continuation patterns)
    if not _candle_context_ok(s, "bullish"):
        # Still allow if volume is increasing strongly
        if vol_bonus == 0:
            return None

    conf = 0.55 + vol_bonus

    entry = b3.close
    stop = b1.low - _atr_offset(atr, 0.20)
    risk = entry - stop
    if risk <= 0:
        return None

    t1 = round(entry + risk, 2)
    t2 = round(entry + risk * 2, 2)

    return _make(s, "Three White Soldiers", Bias.LONG,
                 entry, stop, t2, conf,
                 f"3 White Soldiers, vol {'increasing' if vol_bonus > 0 else 'flat'}",
                 target_1=t1, target_2=t2,
                 trail_type="ema9", trail_param=9.0)


# ==============================================================================
# 10. THREE BLACK CROWS
# ==============================================================================

def _detect_three_black_crows(s):
    """Three Black Crows — 3-bar bearish momentum. Mirror of 3WS."""
    if s.n < 4:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    b1, b2, b3 = s.bars[-3], s.bars[-2], s.bars[-1]

    if not (_is_red(b1) and _is_red(b2) and _is_red(b3)):
        return None

    if b2.open > b1.open or b2.open < b1.close:
        return None
    if b3.open > b2.open or b3.open < b2.close:
        return None

    for b in [b1, b2, b3]:
        body = _body(b)
        if body <= 0:
            return None
        lower_shadow = b.open - b.low  # For red candle, open > close
        if lower_shadow > body * 0.40:
            return None

    vol_bonus = 0.0
    if b1.volume > 0 and b2.volume > b1.volume and b3.volume > b2.volume:
        vol_bonus = 0.08

    if not _candle_context_ok(s, "bearish"):
        if vol_bonus == 0:
            return None

    conf = 0.55 + vol_bonus

    entry = b3.close
    stop = b1.high + _atr_offset(atr, 0.20)
    risk = stop - entry
    if risk <= 0:
        return None

    t1 = round(entry - risk, 2)
    t2 = round(entry - risk * 2, 2)

    return _make(s, "Three Black Crows", Bias.SHORT,
                 entry, stop, t2, conf,
                 f"3 Black Crows, vol {'increasing' if vol_bonus > 0 else 'flat'}",
                 target_1=t1, target_2=t2,
                 trail_type="ema9", trail_param=9.0)

# ==============================================================================
# SMB SCALP PATTERNS (7) — removed HitchHiker, Spencer, BackSide, Breaking News
# ==============================================================================

"""
SECTION 4B: SMB Scalp Pattern Fixes (7 patterns)

CHANGES PER PATTERN:
  1. RubberBand: Bounce window 10→20 bars, VWAP check at bounce, scaled exits
  2. ORB 15/30: NR7 confidence boost, ATR-based ORB range comparison, scaled exits
  3. Second Chance: Wider tolerance (0.3→0.5 ATR), pullback volume check, better target
  4. Fashionably Late: Structural stop (swing low), EMA slope check
  5. Gap Give & Go: Extended time to 10:15, gap volume filter (don't fade breakaway)
  6. Tidal Wave: Temporal compression check, scaled exits

NOTE: These assume existing helpers:
  _compute_vwap(), _compute_ema9(), _make(), _atr_offset(),
  _volume_confirms_breakout(), _is_nr7(), _nearest_sr_target()
"""


# ==============================================================================
# 1. RUBBERBAND SCALP
# ==============================================================================

def _detect_rubberband_scalp(s):
    """RubberBand Scalp — SMB extended selloff bounce.
    
    v2.2 changes:
      - Bounce window extended from 10 to 20 bars
      - Check price is still below VWAP at bounce time
      - Scaled exits encoded: T1=1R, T2=VWAP
      - ATR-based stop
    """
    if s.n < 20 or s.timeframe != "5min":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    # Time filter: 10:00 - 11:00 ET
    cur_time = s.timestamps[-1].time()
    if not (time(10, 0) <= cur_time <= time(11, 0)):
        return None

    # Find today's bars
    today = s.timestamps[-1].date()
    day_bars = [(i, s.bars[i]) for i in range(s.n) if s.timestamps[i].date() == today]
    if len(day_bars) < 6:
        return None

    # Find day open price
    open_price = day_bars[0][1].open

    # Find day low
    dli = min(day_bars, key=lambda x: x[1].low)[0]
    day_low = s.lows[dli]

    # Extension: must be ≥ 2 ATR below open
    extension = open_price - day_low
    if extension < atr * 2.0:
        return None

    # Acceleration check: second half of drop has wider bars than first half
    drop_bars = [i for i, b in day_bars if i <= dli]
    if len(drop_bars) < 4:
        return None
    mid = len(drop_bars) // 2
    first_half_ranges = [_range(s.bars[i]) for i in drop_bars[:mid]]
    second_half_ranges = [_range(s.bars[i]) for i in drop_bars[mid:]]
    if np.mean(first_half_ranges) <= 0 or np.mean(second_half_ranges) <= np.mean(first_half_ranges):
        return None

    # VWAP
    vwap = _vwap_today(s)
    if vwap is None or vwap <= 0:
        return None

    # Search for bounce — EXTENDED to 20 bars (was 10)
    bounce_found = False
    bounce_idx = -1
    for j in range(dli + 1, min(dli + 20, s.n)):
        if _is_green(s.bars[j]) and s.bars[j].close > s.bars[j - 1].high:
            # Bounce bar must have higher volume than prior 3 bars (real demand)
            bounce_vol = s.volumes[j]
            prior_avg_vol = float(np.mean(s.volumes[max(0, j-3):j]))
            if prior_avg_vol > 0 and bounce_vol < prior_avg_vol * 1.3:
                continue  # Weak bounce, keep looking for a stronger one
            bounce_found = True
            bounce_idx = j
            break

    if not bounce_found:
        return None


    # Only fire if bounce is recent (last 5 bars)
    if s.n - 1 - bounce_idx > 5:
        return None

    conf = 0.58
    entry = s.bars[bounce_idx].close
    stop = day_low - _atr_offset(atr, 0.10)
    risk = entry - stop
    if risk <= 0:
        return None

    # T1=1R, T2=VWAP (institutional target)
    t1 = round(entry + risk, 2)
    t2 = round(vwap, 2)

    return _make(s, "RubberBand Scalp", Bias.LONG,
                 entry, stop, t2, conf,
                 f"RubberBand: {extension/atr:.1f} ATR extension, bounce at {entry:.2f}",
                 target_1=t1, target_2=t2,
                 trail_type="vwap", trail_param=0.0,
                 position_splits=(0.34, 0.33, 0.33),
                 key_levels={"day_low": day_low, "vwap": vwap, "open": open_price})


# ==============================================================================
# 2. ORB 15min
# ==============================================================================

def _detect_orb_15(s):
    """ORB 15-minute — Opening Range Breakout.
    
    v2.2 changes:
      - NR7 confidence boost (+0.15)
      - ORB range vs ATR comparison
      - Scaled exits: T1=1R, T2=1.5x ORB range
      - ATR-based offset
    """
    return _detect_orb(s, period_minutes=15, name="ORB 15min")


# ==============================================================================
# 3. ORB 30min
# ==============================================================================

def _detect_orb_30(s):
    """ORB 30-minute — wider opening range."""
    return _detect_orb(s, period_minutes=30, name="ORB 30min")


def _detect_orb(s, period_minutes: int, name: str):
    """Shared ORB logic for 15 and 30 minute periods.
    
    v2.2 additions:
      - NR7 filter: boost confidence by 0.15 if yesterday was narrowest of 7 days
      - ORB range must be between 0.3-3.0 ATR (ATR-relative, not % of price)
      - ATR-based entry offset
      - Scaled exits
    """
    if s.n < 20 or s.timeframe != "5min":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    today = s.timestamps[-1].date()
    market_open = time(9, 30)

    # Collect ORB bars
    orb_bars = []
    for i in range(s.n):
        if s.timestamps[i].date() != today:
            continue
        t = s.timestamps[i].time()
        bar_minutes = (t.hour * 60 + t.minute) - (9 * 60 + 30)
        if 0 <= bar_minutes < period_minutes:
            orb_bars.append(i)

    n_required = period_minutes // 5
    if len(orb_bars) < n_required:
        return None

    orb_high = max(s.highs[i] for i in orb_bars)
    orb_low = min(s.lows[i] for i in orb_bars)
    orb_range = orb_high - orb_low

    # ORB range filters (ATR-relative)
    if orb_range < atr * 0.3 or orb_range > atr * 3.0:
        return None

    cur = s.closes[-1]
    cur_time = s.timestamps[-1].time()

    # Must be after ORB period
    if cur_time <= time(9, 30 + period_minutes // 60, period_minutes % 60):
        return None

    # Must be a current-bar breakout (not re-detecting old breakouts)
    prev = s.closes[-2] if s.n >= 2 else cur

    # NR7 check
    nr7_bonus = 0.15 if _is_nr7(s) else 0.0

    # ORB range tightness bonus (tight range = better setup)
    tight_bonus = 0.0
    if orb_range < atr * 0.8:
        tight_bonus = 0.05

    off = _atr_offset(atr, 0.05)

    if cur > orb_high and prev <= orb_high:
        conf = 0.52 + nr7_bonus + tight_bonus
        entry = orb_high + off
        stop = orb_low - _atr_offset(atr, 0.10)
        target_1r = round(entry + orb_range, 2)
        target_15x = round(entry + orb_range * 1.5, 2)

        return _make(s, name, Bias.LONG,
                     entry, stop, target_15x, conf,
                     f"{name} long: range={orb_range:.2f}" +
                     (" (NR7)" if nr7_bonus > 0 else ""),
                     target_1=target_1r, target_2=target_15x,
                     trail_type="atr", trail_param=1.5,
                     key_levels={"orb_high": orb_high, "orb_low": orb_low})

    elif cur < orb_low and prev >= orb_low:
        conf = 0.52 + nr7_bonus + tight_bonus
        entry = orb_low - off
        stop = orb_high + _atr_offset(atr, 0.10)
        target_1r = round(entry - orb_range, 2)
        target_15x = round(entry - orb_range * 1.5, 2)

        return _make(s, name, Bias.SHORT,
                     entry, stop, target_15x, conf,
                     f"{name} short: range={orb_range:.2f}" +
                     (" (NR7)" if nr7_bonus > 0 else ""),
                     target_1=target_1r, target_2=target_15x,
                     trail_type="atr", trail_param=1.5,
                     key_levels={"orb_high": orb_high, "orb_low": orb_low})

    return None


# ==============================================================================
# 4. SECOND CHANCE SCALP
# ==============================================================================

def _detect_second_chance_scalp(s):
    """Second Chance — pullback to broken resistance (now support).
    
    v2.2 changes:
      - Wider tolerance for level matching: 0.3→0.5 ATR
      - Pullback volume should be LOWER than breakout volume
      - Better target: breakout high + 0.5 × (breakout high - level)
      - ATR-based stop
    """
    if s.n < 20 or s.timeframe != "5min":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    # Find a resistance level with prior-day significance
    res_level = None
    for level in s.sr_levels:
        if level.level_type in ("resistance", "both") and level.touches >= 2:
            # Check if level was from prior sessions (significant)
            level_bars = [i for i in range(s.n) if abs(s.highs[i] - level.price) < atr * 0.2]
            if level_bars:
                earliest = s.timestamps[level_bars[0]].date()
                if earliest < s.timestamps[-1].date():
                    res_level = level
                    break

    if res_level is None:
        return None

    lp = res_level.price

    # Find breakout above resistance (must have happened earlier today)
    breakout_idx = None
    breakout_high = 0
    for i in range(max(0, s.n - 40), s.n - 5):
        if s.closes[i] > lp + atr * 0.1:
            if breakout_idx is None:
                breakout_idx = i
                breakout_high = s.highs[i]
            else:
                breakout_high = max(breakout_high, s.highs[i])

    if breakout_idx is None:
        return None

    # Find pullback to the level — WIDER tolerance (0.5 ATR instead of 0.3)
    pullback_found = False
    pullback_low = None
    for i in range(breakout_idx + 1, s.n):
        if abs(s.lows[i] - lp) < atr * 0.3:
            pullback_found = True
            pullback_low = s.lows[i]
            break

    if not pullback_found or pullback_low is None:
        return None

    # Bounce confirmation: recent bar closes above the level
    if s.n - 1 - breakout_idx < 5:
        return None
    cur = s.bars[-1]
    if not (_is_green(cur) and cur.close > lp):
        return None

    # Recency: bounce must be in last 5 bars
    bounce_recent = any(
        abs(s.lows[i] - lp) < atr * 0.5
        for i in range(max(0, s.n - 5), s.n)
    )
    if not bounce_recent:
        return None

    avg_vol = float(np.mean(s.volumes[max(0, breakout_idx-20):breakout_idx]))
    if avg_vol > 0 and s.volumes[breakout_idx] < avg_vol * 1.5:
        return None
    
    # Pullback volume should be lower than breakout volume
    breakout_vol = float(np.mean(s.volumes[breakout_idx:min(breakout_idx + 3, s.n)]))
    pullback_vol = float(np.mean(s.volumes[-5:]))
    vol_bonus = 0.05 if (breakout_vol > 0 and pullback_vol < breakout_vol * 0.8) else 0.0

    conf = 0.53 + vol_bonus

    entry = cur.close
    stop = pullback_low - _atr_offset(atr, 0.10)
    risk = entry - stop
    if risk <= 0:
        return None

    # Better target: breakout high + 0.5 × extension
    extension = breakout_high - lp
    t1 = round(entry + risk, 2)
    t2 = round(breakout_high + extension * 0.5, 2)

    return _make(s, "Second Chance Scalp", Bias.LONG,
                 entry, stop, t2, conf,
                 f"Second Chance off {lp:.2f}, breakout high {breakout_high:.2f}",
                 target_1=t1, target_2=t2,
                 trail_type="atr", trail_param=1.5,
                 key_levels={"level": lp, "breakout_high": breakout_high})


# ==============================================================================
# 5. FASHIONABLY LATE
# ==============================================================================

def _detect_fashionably_late(s):
    """Fashionably Late — EMA9 crosses above VWAP after morning selloff.
    
    v3.5 changes (post-backtest fixes):
      - EMA slope: must rise at least 0.1 ATR over 5 bars (was just > 0)
      - Stop buffer: 0.20 → 0.10 ATR
      - T1 at 75% measured move (was 50%)
    """
    if s.n < 20 or s.timeframe != "5min":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    cur_time = s.timestamps[-1].time()
    if not (time(10, 0) <= cur_time <= time(10, 45)):
        return None
 
    vwap = _vwap_today(s)
    ema9 = _compute_ema9(s)
    if vwap is None or ema9 is None or vwap <= 0 or ema9 <= 0:
        return None
 
    # EMA must have crossed above VWAP recently (last 3 bars)
    cross_idx = None
    for i in range(max(0, s.n - 3), s.n):
        if i < 2:
            continue
        if s.closes[i - 1] < vwap and s.closes[i] > vwap:
            cross_idx = i
            break
 
    if cross_idx is None:
        if s.n >= 3:
            ema_prev = s.closes[-2]
            if ema_prev < vwap and ema9 > vwap:
                cross_idx = s.n - 1
            else:
                return None
        else:
            return None
 
    cross_price = ema9
 
    # ← CHANGED: EMA slope must rise at least 0.1 ATR over 5 bars (was just > 0)
    if s.n >= 5:
        ema_slope = s.closes[-1] - s.closes[-3]
        if ema_slope < atr * 0.1:
            return None
 
    # Find today's low for measured move
    today = s.timestamps[-1].date()
    day_lows = [s.lows[i] for i in range(s.n) if s.timestamps[i].date() == today]
    if not day_lows:
        return None
    day_low = min(day_lows)
    mm = cross_price - day_low
    if mm <= atr * 0.5:
        return None
 
    # Structural stop: most recent swing low before cross
    structural_stop = day_low
    for i in range(max(0, cross_idx - 10), cross_idx):
        if s.lows[i] < structural_stop * 1.1 and s.lows[i] > structural_stop * 0.5:
            if i > 0 and i < s.n - 1:
                if s.lows[i] < s.lows[i - 1] and s.lows[i] < s.lows[i + 1]:
                    structural_stop = s.lows[i]
 
    conf = 0.57
    entry = cross_price
    stop = structural_stop - _atr_offset(atr, 0.10)   # ← CHANGED: was 0.20
    risk = entry - stop
    if risk <= 0:
        return None
 
    t1 = round(entry + mm * 0.75, 2)                   # ← CHANGED: was 0.5
    t2 = round(entry + mm, 2)
 
    return _make(s, "Fashionably Late", Bias.LONG,
                 entry, stop, t2, conf,
                 f"Fashionably Late: EMA9 crossed VWAP at {cross_price:.2f}",
                 target_1=t1, target_2=t2,
                 trail_type="ema9", trail_param=9.0,
                 key_levels={"cross": cross_price, "vwap": vwap, "day_low": day_low})
 


# ==============================================================================
# 6. GAP GIVE & GO
# ==============================================================================

def _detect_gap_give_and_go(s):
    """Gap Give & Go — gap up, pullback, then resume.
    
    v2.2 changes:
      - Time window extended to 10:15 (was 10:00)
      - Gap volume filter: first 5 min vol must be > 3x avg (real catalyst)
      - ATR-based stops
      - Scaled exits
    """
    if s.n < 10 or s.timeframe != "5min":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    cur_time = s.timestamps[-1].time()
    if not (time(9, 30) <= cur_time <= time(10, 15)):  # Extended from 10:00
        return None

    today = s.timestamps[-1].date()
    yesterday = today - timedelta(days=1)

    # Find yesterday's close and today's open
    yest_close = None
    today_open = None
    for i in range(s.n):
        if s.timestamps[i].date() == yesterday:
            yest_close = s.closes[i]
        if s.timestamps[i].date() == today and today_open is None:
            today_open = s.opens[i]

    if yest_close is None or today_open is None:
        return None

    gap = today_open - yest_close
    gap_pct = gap / yest_close
    if abs(gap_pct) < 0.015:
        return None

    is_gap_up = gap > 0

    # Gap volume filter: first bar(s) of the day should have > 3x average volume
    day_start_bars = [i for i in range(s.n) if s.timestamps[i].date() == today]
    if not day_start_bars:
        return None

    avg_vol = float(np.mean(s.volumes[:max(1, len(day_start_bars))]))
    first_bar_vol = s.volumes[day_start_bars[0]] if day_start_bars else 0
    if avg_vol > 0 and first_bar_vol < avg_vol * 2.0:
        return None  # Not a real catalyst gap — skip

    # Check for retrace ≥ 30% of gap
    day_bars = [s.bars[i] for i in day_start_bars]
    if is_gap_up:
        lowest_after_open = min(b.low for b in day_bars)
        retrace_pct = (today_open - lowest_after_open) / abs(gap) if abs(gap) > 0 else 0
        if retrace_pct < 0.30:
            return None

        # Consolidation: 3-8 bars of tight range after retrace
        consol_bars = day_bars[-min(8, len(day_bars)):]
        if len(consol_bars) < 3:
            return None

        consol_high = max(b.high for b in consol_bars)
        consol_low = min(b.low for b in consol_bars)

        # Breakout above consolidation
        if s.closes[-1] <= consol_high:
            return None

        conf = 0.53
        off = _atr_offset(atr, 0.05)
        entry = consol_high + off
        stop = consol_low - _atr_offset(atr, 0.15)
        risk = entry - stop
        if risk <= 0:
            return None

        t1 = round(entry + risk, 2)
        t2 = round(today_open + abs(gap) * 0.5, 2)

        return _make(s, "Gap Give & Go", Bias.LONG,
                     entry, stop, t2, conf,
                     f"Gap Give & Go: {gap_pct:.1%} gap up, consolidation break",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=1.5,
                     key_levels={"gap_open": today_open, "prev_close": yest_close})

    else:
        # Gap down version
        highest_after_open = max(b.high for b in day_bars)
        retrace_pct = (highest_after_open - today_open) / abs(gap) if abs(gap) > 0 else 0
        if retrace_pct < 0.30:
            return None

        consol_bars = day_bars[-min(8, len(day_bars)):]
        if len(consol_bars) < 3:
            return None

        consol_high = max(b.high for b in consol_bars)
        consol_low = min(b.low for b in consol_bars)

        if s.closes[-1] >= consol_low:
            return None

        conf = 0.53
        off = _atr_offset(atr, 0.05)
        entry = consol_low - off
        stop = consol_high + _atr_offset(atr, 0.15)
        risk = stop - entry
        if risk <= 0:
            return None

        t1 = round(entry - risk, 2)
        t2 = round(today_open - abs(gap) * 0.5, 2)

        return _make(s, "Gap Give & Go", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"Gap Give & Go: {gap_pct:.1%} gap down, consolidation break",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=1.5,
                     key_levels={"gap_open": today_open, "prev_close": yest_close})


# ==============================================================================
# 7. TIDAL WAVE
# ==============================================================================

def _detect_tidal_wave(s):
    """Tidal Wave — support tested 3+ times with diminishing bounces.
    
    v2.2 changes:
      - Temporal compression check: bounces get shorter in duration
      - ATR-based entry/stop (already good from v2)
      - Scaled exits
      
    This was already one of the best-implemented patterns. Minor improvements only.
    """
    if s.n < 30:
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    # Find support level with 3+ touches
    sup_level = None
    for level in s.sr_levels:
        if level.level_type in ("support", "both") and level.touches >= 3:
            sup_level = level
            break

    if sup_level is None:
        return None

    sup = sup_level.price

    # Find bounces off this level
    bounces = []
    tol = atr * 0.3
    i = 0
    while i < s.n:
        if abs(s.lows[i] - sup) < tol:
            # Found a touch — now find the bounce high
            bounce_high = s.highs[i]
            bounce_start = i
            j = i + 1
            while j < s.n and s.lows[j] > sup - tol:
                bounce_high = max(bounce_high, s.highs[j])
                j += 1
            bounce_bars = j - bounce_start
            bounces.append({
                "touch_idx": i,
                "bounce_high": bounce_high,
                "bounce_height": bounce_high - sup,
                "bounce_bars": bounce_bars,
            })
            i = j
        else:
            i += 1

    if len(bounces) < 3:
        return None

    # Check diminishing bounce heights
    heights = [b["bounce_height"] for b in bounces]
    diminishing = all(heights[i] > heights[i + 1] for i in range(len(heights) - 1))
    if not diminishing:
        # At least last bounce should be shorter than first
        if heights[-1] >= heights[0]:
            return None

    # Temporal compression: bounces get shorter in duration
    durations = [b["bounce_bars"] for b in bounces]
    temporal_bonus = 0.0
    if len(durations) >= 3 and durations[-1] < durations[0]:
        temporal_bonus = 0.05

    # Pattern must span ≥ 15 bars
    first_touch = bounces[0]["touch_idx"]
    if s.n - 1 - first_touch < 15:
        return None

    # Breakdown: price below support
    if s.closes[-1] >= sup:
        return None

    # Volume expansion on breakdown
    if not _volume_confirms_breakout(s, -1, 1.3):
        return None

    # Confidence
    touch_bonus = min(0.08, (len(bounces) - 3) * 0.03)
    dim_ratio = heights[-1] / heights[0] if heights[0] > 0 else 1.0
    dim_bonus = (1 - dim_ratio) * 0.10

    conf = 0.55 + touch_bonus + dim_bonus + temporal_bonus

    off = _atr_offset(atr, 0.10)
    entry = sup - off
    stop = bounces[0]["bounce_high"] + _atr_offset(atr, 0.20)
    avg_bounce = np.mean(heights)
    target_full = entry - avg_bounce
    target_half = entry - avg_bounce * 0.5

    return _make(s, "Tidal Wave", Bias.SHORT,
                 entry, stop, target_full, conf,
                 f"Tidal Wave: {len(bounces)} touches, "
                 f"dim ratio {dim_ratio:.0%}"
                 f"{' (temporal compression)' if temporal_bonus > 0 else ''}",
                 target_1=round(target_half, 2),
                 target_2=round(target_full, 2),
                 trail_type="atr", trail_param=2.0,
                 key_levels={"support": sup, "first_bounce_high": bounces[0]["bounce_high"]})

# ==============================================================================
# QUANT STRATEGY PATTERNS — Intraday (4)
# ==============================================================================

"""
SECTION 5: Quant Strategy Fixes (9 patterns)

INSTRUCTIONS:
  Replace the 9 quant detector functions in classifier.py.

4 INTRADAY (5min):
  1. Mean Reversion — z-score 2.0→2.5, volume exhaustion, VWAP as primary target
  2. Trend Pullback — short support in bear regime, EMA slope, structural stop
  3. Gap Fade — breakaway gap volume filter, ATR stop
  4. VWAP Reversion — VWAP deviation bands, structural stop

5 DAILY (1d):
  5. Momentum Breakout — 50 SMA filter, volume check, tighter stop (10d low or 2 ATR)
  6. Vol Compression Breakout — volume confirmation, sustained squeeze check
  7. Range Expansion — volume + trend alignment, tighter stop
  8. Volume Breakout — trend alignment, tighter stop
  9. Donchian Breakout — trend alignment, tighter stop
"""

# ==============================================================================
# 1. MEAN REVERSION
# ==============================================================================

def _detect_mean_reversion(s):
    """Mean Reversion — fade extreme extensions back to the mean.
    
    v2.2 changes:
      - Z-score threshold raised from 2.0 to 2.5 (professional standard)
      - Volume exhaustion check: declining volume at extreme = stronger signal
      - VWAP as primary target (institutional benchmark, was 50-bar MA)
      - Regime confidence multiplier
      - Scaled exits: T1=50% of move to VWAP, T2=VWAP
    """
    if s.n < 50 or s.timeframe != "5min":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    # Regime filter: only in mean_reverting or mixed
    if s._regime not in ("mean_reverting", "mixed"):
        return None

    closes = s.closes[-50:]
    ma50 = float(np.mean(closes))
    std50 = float(np.std(closes))
    if std50 <= 0:
        return None

    cur = s.closes[-1]
    z = (cur - ma50) / std50

    # Raised threshold from 2.0 to 2.5
    if abs(z) < 2.5:
        return None

    # Volume exhaustion check
    vol_bonus = 0.05 if _volume_exhaustion(s, -1) else 0.0

    # VWAP for target
    vwap = _vwap_today(s)
    if vwap is None or vwap <= 0:
        vwap = ma50  # Fallback to MA

    conf = 0.58 + vol_bonus
    conf *= _regime_confidence_mult(s, "mean_reversion")

    if z < -2.5:
        # Oversold — long
        entry = cur
        # S/R based stop or extreme - 1 ATR
        recent_low = min(s.lows[-10:])
        stop = min(recent_low, cur - atr) - _atr_offset(atr, 0.15)
        risk = entry - stop
        if risk <= 0:
            return None

        dist_to_vwap = vwap - entry
        t1 = round(entry + dist_to_vwap * 0.5, 2)
        t2 = round(vwap, 2)

        return _make(s, "Mean Reversion", Bias.LONG,
                     entry, stop, t2, conf,
                     f"Mean Rev Long: z={z:.2f}, target VWAP {vwap:.2f}",
                     target_1=t1, target_2=t2,
                     trail_type="vwap", trail_param=0.0,
                     key_levels={"ma50": ma50, "vwap": vwap, "z_score": round(z, 2)})

    elif z > 2.5:
        # Overbought — short
        entry = cur
        recent_high = max(s.highs[-10:])
        stop = max(recent_high, cur + atr) + _atr_offset(atr, 0.15)
        risk = stop - entry
        if risk <= 0:
            return None

        dist_to_vwap = entry - vwap
        t1 = round(entry - dist_to_vwap * 0.5, 2)
        t2 = round(vwap, 2)

        return _make(s, "Mean Reversion", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"Mean Rev Short: z={z:.2f}, target VWAP {vwap:.2f}",
                     target_1=t1, target_2=t2,
                     trail_type="vwap", trail_param=0.0,
                     key_levels={"ma50": ma50, "vwap": vwap, "z_score": round(z, 2)})

    return None


# ==============================================================================
# 2. TREND PULLBACK
# ==============================================================================

def _detect_trend_pullback(s):
    """Trend Pullback — buy dip to rising EMA in uptrend, sell rally in downtrend.
    
    v3.5 changes (post-backtest fixes):
      - Stop buffer: 0.20 → 0.10 ATR (Phase 1, Fix 2)
      - LONG: pullback depth ≥ 1.0 ATR, require red bar before green bounce ✓ (already applied)
      - SHORT: pullback depth ≥ 1.0 ATR (was MISSING), require green bar before red rejection (was MISSING)
      - EMA proximity: 0.3 ATR ✓ (already applied)
    """
    if s.n < 30 or s.timeframe != "5min":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    # Regime filter: trending_bull OR trending_bear
    if s._regime not in ("trending_bull", "trending_bear"):
        return None

    # Compute 21 EMA
    ema21 = float(s.closes[-1])
    if hasattr(s, 'ema21') and s.ema21 is not None:
        ema21 = s.ema21
    else:
        mult = 2 / 22
        ema = float(s.closes[0])
        for i in range(1, s.n):
            ema = s.closes[i] * mult + ema * (1 - mult)
        ema21 = ema

    cur = s.closes[-1]
    vwap = _vwap_today(s)

    # EMA slope check
    if s.n >= 5:
        mult = 2 / 22
        ema_prev = float(s.closes[0])
        for i in range(1, s.n - 5):
            ema_prev = s.closes[i] * mult + ema_prev * (1 - mult)
        ema_slope = ema21 - ema_prev
    else:
        ema_slope = 0

    if s._regime == "trending_bull":
        # ── LONG setup ──
        if ema_slope <= 0:
            return None

        # Price must be near EMA (0.3 ATR)
        if abs(cur - ema21) > atr * 0.3:
            return None

        # Pullback depth: must drop at least 1.0 ATR from recent high
        recent_high = max(s.highs[-10:])
        pullback_depth = recent_high - min(s.lows[-5:])
        if pullback_depth < atr * 1.0:
            return None

        # Price should be above VWAP
        if vwap is not None and cur < vwap:
            return None

        # Must have at least one red bar in last 3 bars (actual pullback)
        recent_reds = sum(1 for j in range(-4, -1) if j + s.n >= 0 and _is_red(s.bars[j]))
        if recent_reds == 0:
            return None

        # Last bar must be green (bounce confirmation)
        if not _is_green(s.bars[-1]):
            return None

        conf = 0.60
        conf *= _regime_confidence_mult(s, "momentum")

        entry = cur
        pullback_low = min(s.lows[-5:])
        stop = pullback_low - _atr_offset(atr, 0.10)  # ← CHANGED: was 0.20

        risk = entry - stop
        if risk <= 0:
            return None

        prior_high = max(s.highs[-20:]) if s.n >= 20 else entry + risk * 2
        t1 = round(entry + risk, 2)
        t2 = round(max(prior_high, entry + risk * 2), 2)

        return _make(s, "Trend Pullback", Bias.LONG,
                     entry, stop, t2, conf,
                     f"Trend Pullback Long: EMA21={ema21:.2f}, bounce",
                     target_1=t1, target_2=t2,
                     trail_type="ema9", trail_param=9.0,
                     key_levels={"ema21": ema21})

    elif s._regime == "trending_bear":
        # ── SHORT setup ──
        if ema_slope >= 0:
            return None

        # Price must be near EMA (0.3 ATR)
        if abs(cur - ema21) > atr * 0.3:
            return None

        # ← ADDED: Pullback depth — price must rally at least 1.0 ATR from recent low
        recent_low = min(s.lows[-10:])
        pullback_depth = max(s.highs[-5:]) - recent_low
        if pullback_depth < atr * 1.0:
            return None

        # Price should be below VWAP
        if vwap is not None and cur > vwap:
            return None

        # ← ADDED: Must have at least one green bar in last 3 (actual rally before rejection)
        recent_greens = sum(1 for j in range(-4, -1) if j + s.n >= 0 and _is_green(s.bars[j]))
        if recent_greens == 0:
            return None

        # Last bar must be red (rejection confirmation)
        if not _is_red(s.bars[-1]):
            return None

        conf = 0.58
        conf *= _regime_confidence_mult(s, "momentum")

        entry = cur
        pullback_high = max(s.highs[-5:])
        stop = pullback_high + _atr_offset(atr, 0.10)  # ← CHANGED: was 0.20

        risk = stop - entry
        if risk <= 0:
            return None

        prior_low = min(s.lows[-20:]) if s.n >= 20 else entry - risk * 2
        t1 = round(entry - risk, 2)
        t2 = round(min(prior_low, entry - risk * 2), 2)

        return _make(s, "Trend Pullback", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"Trend Pullback Short: EMA21={ema21:.2f}, rejection",
                     target_1=t1, target_2=t2,
                     trail_type="ema9", trail_param=9.0,
                     key_levels={"ema21": ema21})

    return None


# ==============================================================================
# 3. GAP FADE
# ==============================================================================

def _detect_gap_fade(s):
    """Gap Fade — fade large gaps that don't follow through.
    
    v2.2 changes:
      - Breakaway gap filter: if first 5 min vol > 3x avg, don't fade
      - ATR-based stop (was fixed % of gap)
      - Scaled exits: T1 at 50% gap fill, T2 at full gap fill
    """
    if s.n < 10 or s.timeframe != "5min":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    # Regime: don't fade in strong trends
    if s._regime in ("trending_bull", "trending_bear"):
        return None

    cur_time = s.timestamps[-1].time()
    if not (time(9, 30) <= cur_time <= time(10, 30)):
        return None

    today = s.timestamps[-1].date()
    yesterday = today - timedelta(days=1)

    yest_close = None
    today_open = None
    for i in range(s.n):
        if s.timestamps[i].date() == yesterday:
            yest_close = s.closes[i]
        if s.timestamps[i].date() == today and today_open is None:
            today_open = s.opens[i]

    if yest_close is None or today_open is None:
        return None

    gap = today_open - yest_close
    gap_pct = abs(gap) / yest_close
    if gap_pct < 0.02:
        return None

    # Breakaway gap filter: high volume in first bars = real catalyst, don't fade
    day_bars = [i for i in range(s.n) if s.timestamps[i].date() == today]
    if day_bars:
        first_bars_vol = float(np.mean([s.volumes[i] for i in day_bars[:min(3, len(day_bars))]]))
        prior_avg_vol = float(np.mean(s.volumes[:max(1, day_bars[0])]))
        if prior_avg_vol > 0 and first_bars_vol > prior_avg_vol * 3.0:
            return None  # Breakaway gap — don't fade

    cur = s.closes[-1]
    conf = 0.55

    if gap > 0:
        # Gap up — fade short
        if cur >= today_open:
            return None  # Gap continuing, don't fade

        entry = cur
        stop = today_open + abs(gap) * 0.5
        risk = stop - entry
        if risk <= 0:
            return None

        half_fill = round((today_open + yest_close) / 2, 2)
        full_fill = round(yest_close, 2)

        return _make(s, "Gap Fade", Bias.SHORT,
                     entry, stop, full_fill, conf,
                     f"Gap Fade Short: {gap_pct:.1%} gap up, targeting fill",
                     target_1=half_fill, target_2=full_fill,
                     trail_type="atr", trail_param=1.5,
                     key_levels={"gap_open": today_open, "prev_close": yest_close})

    else:
        # Gap down — fade long
        if cur <= today_open:
            return None

        entry = cur
        stop = today_open - abs(gap) * 0.5
        risk = entry - stop
        if risk <= 0:
            return None

        half_fill = round((today_open + yest_close) / 2, 2)
        full_fill = round(yest_close, 2)

        return _make(s, "Gap Fade", Bias.LONG,
                     entry, stop, full_fill, conf,
                     f"Gap Fade Long: {gap_pct:.1%} gap down, targeting fill",
                     target_1=half_fill, target_2=full_fill,
                     trail_type="atr", trail_param=1.5,
                     key_levels={"gap_open": today_open, "prev_close": yest_close})


# ==============================================================================
# 4. VWAP REVERSION
# ==============================================================================

def _detect_vwap_reversion(s):
    """VWAP Reversion — fade extreme extensions from VWAP.
    
    v2.2 changes:
      - VWAP deviation bands (2σ) as trigger instead of raw 2.5 ATR
      - Structural stop: recent swing high/low beyond the extreme
      - Volume exhaustion bonus
      - Scaled exits: T1 at 1σ band, T2 at VWAP
    """
    if s.n < 30 or s.timeframe != "5min":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    if s._regime in ("trending_bull", "trending_bear"):
        return None

    cur_time = s.timestamps[-1].time()
    if not (time(10, 30) <= cur_time <= time(14, 0)):
        return None

    vwap = _vwap_today(s)
    if vwap is None or vwap <= 0:
        return None

    cur = s.closes[-1]

    # VWAP deviation bands
    today = s.timestamps[-1].date()
    day_closes = [s.closes[i] for i in range(s.n) if s.timestamps[i].date() == today]
    if len(day_closes) < 10:
        return None

    # Standard deviation of price from VWAP
    deviations = [c - vwap for c in day_closes]
    vwap_std = float(np.std(deviations))
    if vwap_std <= 0:
        return None

    # How many σ from VWAP
    sigma = (cur - vwap) / vwap_std

    # Need at least 2σ extension (replaces raw 2.5 ATR check)
    # Also keep ATR as backup: must be at least 2.0 ATR from VWAP
    atr_distance = abs(cur - vwap) / atr if atr > 0 else 0
    if abs(sigma) < 2.0 or atr_distance < 2.0:
        return None

    vol_bonus = 0.05 if _volume_exhaustion(s, -1) else 0.0

    # 1σ band for T1
    one_sigma = vwap_std

    conf = 0.58 + vol_bonus
    conf *= _regime_confidence_mult(s, "mean_reversion")

    if sigma < -2.0:
        # Extended below VWAP — long
        entry = cur
        # Structural stop: lowest low in last 10 bars
        recent_low = min(s.lows[-10:])
        stop = recent_low - _atr_offset(atr, 0.20)
        risk = entry - stop
        if risk <= 0:
            return None

        t1 = round(vwap - one_sigma, 2)  # 1σ below VWAP
        t2 = round(vwap, 2)

        return _make(s, "VWAP Reversion", Bias.LONG,
                     entry, stop, t2, conf,
                     f"VWAP Rev Long: {sigma:.1f}σ below, target VWAP {vwap:.2f}",
                     target_1=t1, target_2=t2,
                     trail_type="vwap", trail_param=0.0,
                     key_levels={"vwap": vwap, "sigma": round(sigma, 2)})

    elif sigma > 2.0:
        # Extended above VWAP — short
        entry = cur
        recent_high = max(s.highs[-10:])
        stop = recent_high + _atr_offset(atr, 0.20)
        risk = stop - entry
        if risk <= 0:
            return None

        t1 = round(vwap + one_sigma, 2)  # 1σ above VWAP
        t2 = round(vwap, 2)

        return _make(s, "VWAP Reversion", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"VWAP Rev Short: {sigma:.1f}σ above, target VWAP {vwap:.2f}",
                     target_1=t1, target_2=t2,
                     trail_type="vwap", trail_param=0.0,
                     key_levels={"vwap": vwap, "sigma": round(sigma, 2)})

    return None


# ==============================================================================
# 5. MOMENTUM BREAKOUT (Daily)
# ==============================================================================

def _detect_momentum_breakout(s):
    """Momentum Breakout — new 20-day high with trend confirmation.
    
    v2.2 changes:
      - Price must be > 50 SMA (trend confirmation)
      - Breakout day volume > 1.5x 20-day average
      - Tighter stop: 10-day low or entry - 2 ATR (whichever closer)
      - Scaled exits
    """
    if s.n < 50 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    cur = s.closes[-1]

    # 20-day high breakout
    high_20 = max(s.highs[-21:-1])  # Prior 20 days (exclude today)
    if cur <= high_20:
        return None

    # 50 SMA filter: price must be above
    sma50 = float(np.mean(s.closes[-50:]))
    if cur < sma50:
        return None

    # Volume confirmation: today > 1.5x 20-day avg
    avg_vol_20 = float(np.mean(s.volumes[-20:]))
    if avg_vol_20 > 0 and s.volumes[-1] < avg_vol_20 * 1.5:
        return None

    conf = 0.58

    entry = cur
    # Tighter stop: 10-day low or entry - 2 ATR (whichever is closer)
    low_10 = min(s.lows[-10:])
    atr_stop = entry - atr * 2
    stop = max(low_10, atr_stop)  # Whichever is closer to entry
    stop -= _atr_offset(atr, 0.10)

    risk = entry - stop
    if risk <= 0:
        return None

    t1 = round(entry + risk, 2)
    t2 = round(entry + risk * 2, 2)

    return _make(s, "Momentum Breakout", Bias.LONG,
                 entry, stop, t2, conf,
                 f"Momentum BO: new 20d high, above 50 SMA, vol {s.volumes[-1]/avg_vol_20:.1f}x",
                 target_1=t1, target_2=t2,
                 trail_type="atr", trail_param=2.0,
                 key_levels={"high_20": high_20, "sma50": sma50})


# ==============================================================================
# 6. VOL COMPRESSION BREAKOUT (Daily)
# ==============================================================================

def _detect_vol_compression_breakout(s):
    """Vol Compression Breakout — Bollinger Squeeze.
    
    v2.2 changes:
      - Volume confirmation on breakout bar
      - Sustained squeeze: compression must last ≥ 5 days
      - ATR-based stop
      - Scaled exits
    """
    if s.n < 30 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    # ATR ratio: current ATR vs 50-bar ATR
    atr_vals = wilder_atr(s.highs[-30:], s.lows[-30:], s.closes[-30:], period=14)
    if len(atr_vals) < 20 or np.isnan(atr_vals[-1]):
        return None

    current_atr = float(atr_vals[-1])
    avg_atr = float(np.mean(atr_vals[-20:]))
    if avg_atr <= 0:
        return None

    atr_ratio = current_atr / avg_atr
    if atr_ratio > 0.6:
        return None  # Not compressed enough

    # Sustained squeeze: ratio must have been < 0.7 for at least 5 bars
    squeeze_bars = sum(1 for v in atr_vals[-10:] if not np.isnan(v) and v / avg_atr < 0.7)
    if squeeze_bars < 5:
        return None

    # Volume confirmation
    avg_vol = float(np.mean(s.volumes[-20:]))
    vol_bonus = 0.05 if (avg_vol > 0 and s.volumes[-1] > avg_vol * 1.3) else 0.0

    cur = s.closes[-1]
    sma20 = float(np.mean(s.closes[-20:]))

    conf = 0.57 + vol_bonus

    off = _atr_offset(atr, 0.10)

    if cur > sma20:
        # Bullish breakout
        entry = cur
        stop = sma20 - atr
        risk = entry - stop
        if risk <= 0:
            return None

        t1 = round(entry + risk, 2)
        t2 = round(entry + risk * 2, 2)

        return _make(s, "Vol Compression Breakout", Bias.LONG,
                     entry, stop, t2, conf,
                     f"Vol Squeeze BO Long: ATR ratio {atr_ratio:.2f}, {squeeze_bars}d squeeze",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=2.0,
                     key_levels={"sma20": sma20, "atr_ratio": round(atr_ratio, 2)})
    else:
        entry = cur
        stop = sma20 + atr
        risk = stop - entry
        if risk <= 0:
            return None

        t1 = round(entry - risk, 2)
        t2 = round(entry - risk * 2, 2)

        return _make(s, "Vol Compression Breakout", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"Vol Squeeze BO Short: ATR ratio {atr_ratio:.2f}, {squeeze_bars}d squeeze",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=2.0,
                     key_levels={"sma20": sma20, "atr_ratio": round(atr_ratio, 2)})


# ==============================================================================
# 7. RANGE EXPANSION (Daily)
# ==============================================================================

def _detect_range_expansion(s):
    """Range Expansion — today's range significantly exceeds recent average.
    
    v2.2 changes:
      - Volume confirmation (1.3x avg)
      - Trend alignment: breakout in direction of 20 SMA
      - Tighter stop: 10-day low/high or 2 ATR (was 10-day range)
      - Scaled exits
    """
    if s.n < 20 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    today_range = s.highs[-1] - s.lows[-1]
    avg_range = float(np.mean([s.highs[i] - s.lows[i] for i in range(-11, -1)]))
    if avg_range <= 0:
        return None

    # Range must be ≥ 2x average
    if today_range < avg_range * 2.0:
        return None

    # Volume confirmation
    avg_vol = float(np.mean(s.volumes[-20:]))
    if avg_vol > 0 and s.volumes[-1] < avg_vol * 1.3:
        return None

    cur = s.closes[-1]
    sma20 = float(np.mean(s.closes[-20:]))

    # Trend alignment: only trade in direction of 20 SMA
    conf = 0.54

    if cur > s.opens[-1] and cur > sma20:
        # Bullish expansion aligned with trend
        entry = cur
        low_10 = min(s.lows[-10:])
        stop = max(low_10, entry - atr * 2) - _atr_offset(atr, 0.10)
        risk = entry - stop
        if risk <= 0:
            return None

        t1 = round(entry + risk, 2)
        t2 = round(entry + risk * 2, 2)

        return _make(s, "Range Expansion", Bias.LONG,
                     entry, stop, t2, conf,
                     f"Range Expansion Long: {today_range/avg_range:.1f}x avg range",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=2.0)

    elif cur < s.opens[-1] and cur < sma20:
        entry = cur
        high_10 = max(s.highs[-10:])
        stop = min(high_10, entry + atr * 2) + _atr_offset(atr, 0.10)
        risk = stop - entry
        if risk <= 0:
            return None

        t1 = round(entry - risk, 2)
        t2 = round(entry - risk * 2, 2)

        return _make(s, "Range Expansion", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"Range Expansion Short: {today_range/avg_range:.1f}x avg range",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=2.0)

    return None


# ==============================================================================
# 8. VOLUME BREAKOUT (Daily)
# ==============================================================================

def _detect_volume_breakout(s):
    """Volume Breakout — extreme volume day with directional close.
    
    v2.2 changes:
      - Trend alignment (20 SMA direction)
      - Tighter stop: 10-day low/high or 2 ATR
      - Scaled exits
      - Volume threshold kept at 3x (already strict — this is the pattern's edge)
    """
    if s.n < 20 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None

    avg_vol = float(np.mean(s.volumes[-20:]))
    if avg_vol <= 0 or s.volumes[-1] < avg_vol * 3.0:
        return None

    cur = s.closes[-1]
    sma20 = float(np.mean(s.closes[-20:]))

    conf = 0.56

    if cur > s.opens[-1]:
        # Bullish volume
        if cur < sma20:
            conf -= 0.05  # Against trend — lower confidence

        entry = cur
        low_10 = min(s.lows[-10:])
        stop = max(low_10, entry - atr * 2) - _atr_offset(atr, 0.10)
        risk = entry - stop
        if risk <= 0:
            return None

        t1 = round(entry + risk, 2)
        t2 = round(entry + risk * 2, 2)

        return _make(s, "Volume Breakout", Bias.LONG,
                     entry, stop, t2, conf,
                     f"Volume BO Long: {s.volumes[-1]/avg_vol:.1f}x avg vol",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=2.0)

    elif cur < s.opens[-1]:
        if cur > sma20:
            conf -= 0.05

        entry = cur
        high_10 = max(s.highs[-10:])
        stop = min(high_10, entry + atr * 2) + _atr_offset(atr, 0.10)
        risk = stop - entry
        if risk <= 0:
            return None

        t1 = round(entry - risk, 2)
        t2 = round(entry - risk * 2, 2)

        return _make(s, "Volume Breakout", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"Volume BO Short: {s.volumes[-1]/avg_vol:.1f}x avg vol",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=2.0)

    return None


# ==============================================================================
# 9. DONCHIAN BREAKOUT (Daily)
# ==============================================================================

def _detect_donchian_breakout(s):
    """Donchian Breakout — new 20-day high/low channel breakout.
    
    v3.5 changes (post-backtest fixes):
      - ADDED: Squeeze filter — recent 5-day ranges must be < 85% of 20-day avg
      - Stop buffer: 0.10 ATR ✓ (already correct)
      - Volume 1.5x ✓ (already correct)
      - Trend alignment ✓ (already correct)
    """
    if s.n < 50 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    # ← ADDED: Squeeze filter — Donchian works best after volatility compression
    if s.n >= 25:
        recent_ranges = [s.highs[i] - s.lows[i] for i in range(s.n - 5, s.n)]
        longer_ranges = [s.highs[i] - s.lows[i] for i in range(s.n - 20, s.n - 5)]
        avg_recent = float(np.mean(recent_ranges))
        avg_longer = float(np.mean(longer_ranges))
        if avg_longer > 0 and avg_recent > avg_longer * 0.85:
            return None  # No compression = no edge
 
    cur = s.closes[-1]
    high_20 = max(s.highs[-21:-1])
    low_20 = min(s.lows[-21:-1])
    sma50 = float(np.mean(s.closes[-50:]))
 
    # Volume confirmation
    avg_vol = float(np.mean(s.volumes[-20:]))
    vol_ok = avg_vol > 0 and s.volumes[-1] >= avg_vol * 1.5
 
    if cur > high_20:
        # Bullish breakout
        trend_aligned = cur > sma50
        conf = 0.55 if trend_aligned else 0.48
        if not vol_ok:
            conf -= 0.05
 
        entry = cur
        low_10 = min(s.lows[-10:])
        stop = max(low_10, entry - atr * 2) - _atr_offset(atr, 0.10)
        risk = entry - stop
        if risk <= 0:
            return None
 
        t1 = round(entry + risk, 2)
        t2 = round(entry + risk * 2, 2)
 
        return _make(s, "Donchian Breakout", Bias.LONG,
                     entry, stop, t2, conf,
                     f"Donchian BO Long: new 20d high (squeezed)" +
                     (" (trend aligned)" if trend_aligned else "") +
                     (f" vol {s.volumes[-1]/avg_vol:.1f}x" if vol_ok else ""),
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=2.5)
 
    elif cur < low_20:
        trend_aligned = cur < sma50
        conf = 0.55 if trend_aligned else 0.48
        if not vol_ok:
            conf -= 0.05
 
        entry = cur
        high_10 = max(s.highs[-10:])
        stop = min(high_10, entry + atr * 2) + _atr_offset(atr, 0.10)
        risk = stop - entry
        if risk <= 0:
            return None
 
        t1 = round(entry - risk, 2)
        t2 = round(entry - risk * 2, 2)
 
        return _make(s, "Donchian Breakout", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"Donchian BO Short: new 20d low (squeezed)" +
                     (" (trend aligned)" if trend_aligned else "") +
                     (f" vol {s.volumes[-1]/avg_vol:.1f}x" if vol_ok else ""),
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=2.5)
 
    return None

# ==============================================================================
# 1. TIME-SERIES MOMENTUM (Daily)
#    Moskowitz, Ooi & Pedersen (2012) — over a century of evidence
# ==============================================================================
 
def _detect_time_series_momentum(s):
    """Time-Series Momentum — if trailing return is positive, go long; negative, short.
    
    Unlike cross-sectional momentum (ranking stocks vs each other), this is
    absolute: each stock's own past return determines its signal.
    Uses 252-day (12-month) lookback, skipping the most recent 21 days
    (1-month) to avoid short-term reversal contamination.
    
    Position sized by inverse volatility for consistent risk contribution.
    """
    if s.n < 252 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    cur = s.closes[-1]
 
    # 12-month return, skipping most recent month (t-252 to t-21)
    price_12m_ago = s.closes[-252]
    price_1m_ago = s.closes[-21]
    momentum_return = (price_1m_ago - price_12m_ago) / price_12m_ago
 
    # Need meaningful momentum (at least 5% in either direction)
    if abs(momentum_return) < 0.05:
        return None
 
    # Volatility for sizing context (60-day realized vol)
    daily_returns = np.diff(s.closes[-61:]) / s.closes[-61:-1]
    vol_60d = float(np.std(daily_returns)) * np.sqrt(252)
    if vol_60d <= 0:
        return None
 
    # Confidence scales with momentum strength
    mom_strength = min(abs(momentum_return) / 0.30, 1.0)  # Cap at 30% return
    conf = 0.55 + mom_strength * 0.10
 
    if momentum_return > 0.05:
        # LONG
        entry = cur
        stop = cur - atr * 2.5
        t1 = round(entry + atr * 2.0, 2)
        t2 = round(entry + atr * 4.0, 2)
        risk = entry - stop
        if risk <= 0:
            return None
 
        return _make(s, "TS Momentum Long", Bias.LONG,
                     entry, stop, t2, conf,
                     f"TS Momentum Long: 12m ret={momentum_return:+.1%}, vol={vol_60d:.0%}",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=2.5,
                     position_splits=(0.30, 0.30, 0.40),
                     key_levels={"momentum_return": round(momentum_return, 3),
                                 "annualized_vol": round(vol_60d, 3)})
 
    elif momentum_return < -0.05:
        # SHORT
        entry = cur
        stop = cur + atr * 2.5
        t1 = round(entry - atr * 2.0, 2)
        t2 = round(entry - atr * 4.0, 2)
        risk = stop - entry
        if risk <= 0:
            return None
 
        return _make(s, "TS Momentum Short", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"TS Momentum Short: 12m ret={momentum_return:+.1%}, vol={vol_60d:.0%}",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=2.5,
                     position_splits=(0.30, 0.30, 0.40),
                     key_levels={"momentum_return": round(momentum_return, 3),
                                 "annualized_vol": round(vol_60d, 3)})
 
    return None
 
 
# ==============================================================================
# 2. MULTI-TIMEFRAME TREND FOLLOWING (Daily)
#    AQR, Man AHL — blend signals across multiple lookbacks
# ==============================================================================
 
def _detect_multi_tf_trend(s):
    """Multi-TF Trend — blend EMA crossover signals at 20/50, 50/100, 100/200.
    
    Each lookback pair produces a signal: +1 (fast > slow), -1 (fast < slow).
    Average the three signals. Only trade when all three agree (consensus = ±1.0)
    or two agree (consensus = ±0.67).
    """
    if s.n < 200 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    cur = s.closes[-1]
 
    # Compute EMAs
    def ema(data, period):
        mult = 2.0 / (period + 1)
        result = float(data[0])
        for i in range(1, len(data)):
            result = data[i] * mult + result * (1 - mult)
        return result
 
    ema20 = ema(s.closes, 20)
    ema50 = ema(s.closes, 50)
    ema100 = ema(s.closes, 100)
    ema200 = ema(s.closes, 200)
 
    # Signal per pair
    sig1 = 1.0 if ema20 > ema50 else -1.0    # Short-term
    sig2 = 1.0 if ema50 > ema100 else -1.0   # Medium-term
    sig3 = 1.0 if ema100 > ema200 else -1.0  # Long-term
 
    consensus = (sig1 + sig2 + sig3) / 3.0
 
    # Need at least 2 of 3 agreeing
    if abs(consensus) < 0.5:
        return None
 
    # Volume confirmation: today's volume > 20d average
    avg_vol = float(np.mean(s.volumes[-20:]))
    if avg_vol > 0 and s.volumes[-1] < avg_vol * 0.8:
        return None  # Very low volume day — skip
 
    conf = 0.55 + abs(consensus) * 0.10  # 0.62 for 2/3, 0.65 for 3/3
 
    if consensus > 0:
        entry = cur
        stop = cur - atr * 3.0
        t1 = round(entry + atr * 2.0, 2)
        t2 = round(entry + atr * 5.0, 2)
        risk = entry - stop
        if risk <= 0:
            return None
 
        return _make(s, "Multi-TF Trend Long", Bias.LONG,
                     entry, stop, t2, conf,
                     f"Multi-TF Trend Long: consensus={consensus:+.2f} "
                     f"(20/50={'↑' if sig1>0 else '↓'} 50/100={'↑' if sig2>0 else '↓'} "
                     f"100/200={'↑' if sig3>0 else '↓'})",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=3.0,
                     position_splits=(0.25, 0.25, 0.50),
                     key_levels={"ema20": round(ema20, 2), "ema50": round(ema50, 2),
                                 "ema200": round(ema200, 2), "consensus": consensus})
 
    else:
        entry = cur
        stop = cur + atr * 3.0
        t1 = round(entry - atr * 2.0, 2)
        t2 = round(entry - atr * 5.0, 2)
        risk = stop - entry
        if risk <= 0:
            return None
 
        return _make(s, "Multi-TF Trend Short", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"Multi-TF Trend Short: consensus={consensus:+.2f} "
                     f"(20/50={'↑' if sig1>0 else '↓'} 50/100={'↑' if sig2>0 else '↓'} "
                     f"100/200={'↑' if sig3>0 else '↓'})",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=3.0,
                     position_splits=(0.25, 0.25, 0.50),
                     key_levels={"ema20": round(ema20, 2), "ema50": round(ema50, 2),
                                 "ema200": round(ema200, 2), "consensus": consensus})
 
 
# ==============================================================================
# 3. MOVING AVERAGE CROSSOVER — Golden Cross / Death Cross (Daily)
#    Brock, Lakonishok & LeBaron (1992)
# ==============================================================================
 
def _detect_ma_crossover(s):
    """Moving Average Crossover — 50/200 SMA golden cross and death cross.
    
    Enter when 50 SMA crosses above (long) or below (short) the 200 SMA.
    The cross must have happened in the last 5 bars (recency filter).
    Volume confirmation on the cross bar.
    """
    if s.n < 205 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    cur = s.closes[-1]
 
    # Compute current and recent SMAs
    def sma_at(idx, period):
        start = max(0, idx - period + 1)
        return float(np.mean(s.closes[start:idx + 1]))
 
    sma50_now = sma_at(s.n - 1, 50)
    sma200_now = sma_at(s.n - 1, 200)
 
    # Find the cross — check if 50 SMA crossed 200 SMA in last 5 bars
    cross_type = None  # "golden" or "death"
    for lookback in range(1, 6):
        idx = s.n - 1 - lookback
        if idx < 200:
            break
        sma50_prev = sma_at(idx, 50)
        sma200_prev = sma_at(idx, 200)
 
        if sma50_prev <= sma200_prev and sma50_now > sma200_now:
            cross_type = "golden"
            break
        elif sma50_prev >= sma200_prev and sma50_now < sma200_now:
            cross_type = "death"
            break
 
    if cross_type is None:
        return None
 
    # Volume confirmation
    avg_vol = float(np.mean(s.volumes[-20:]))
    vol_bonus = 0.05 if (avg_vol > 0 and s.volumes[-1] > avg_vol * 1.3) else 0.0
 
    conf = 0.57 + vol_bonus
 
    if cross_type == "golden":
        entry = cur
        stop = min(s.lows[-10:]) - _atr_offset(atr, 0.10)
        t1 = round(entry + atr * 2.0, 2)
        t2 = round(entry + atr * 4.0, 2)
        risk = entry - stop
        if risk <= 0:
            return None
 
        return _make(s, "Golden Cross", Bias.LONG,
                     entry, stop, t2, conf,
                     f"Golden Cross: 50 SMA ({sma50_now:.2f}) > 200 SMA ({sma200_now:.2f})",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=3.0,
                     position_splits=(0.30, 0.30, 0.40),
                     key_levels={"sma50": round(sma50_now, 2), "sma200": round(sma200_now, 2)})
 
    else:  # death cross
        entry = cur
        stop = max(s.highs[-10:]) + _atr_offset(atr, 0.10)
        t1 = round(entry - atr * 2.0, 2)
        t2 = round(entry - atr * 4.0, 2)
        risk = stop - entry
        if risk <= 0:
            return None
 
        return _make(s, "Death Cross", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"Death Cross: 50 SMA ({sma50_now:.2f}) < 200 SMA ({sma200_now:.2f})",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=3.0,
                     position_splits=(0.30, 0.30, 0.40),
                     key_levels={"sma50": round(sma50_now, 2), "sma200": round(sma200_now, 2)})
 
 
# ==============================================================================
# 4. SHORT-TERM REVERSAL (Daily)
#    Jegadeesh (1990), Lehmann (1990)
# ==============================================================================
 
def _detect_short_term_reversal(s):
    """Short-Term Reversal — stocks that dropped hard over the past month bounce back.
    
    Buy stocks with extreme negative 21-day returns (oversold).
    Short stocks with extreme positive 21-day returns (overbought).
    
    Only fires on extreme moves (top/bottom 10% of typical monthly ranges).
    Volume exhaustion on the final bar confirms overextension.
    """
    if s.n < 60 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    cur = s.closes[-1]
 
    # 21-day return
    price_21d_ago = s.closes[-22]
    ret_21d = (cur - price_21d_ago) / price_21d_ago
 
    # Historical context: what's a normal 21-day return for this stock?
    monthly_returns = []
    for i in range(22, min(s.n, 252), 21):
        r = (s.closes[-i] - s.closes[-i - 21]) / s.closes[-i - 21] if s.closes[-i - 21] > 0 else 0
        monthly_returns.append(r)
 
    if len(monthly_returns) < 3:
        return None
 
    mean_ret = float(np.mean(monthly_returns))
    std_ret = float(np.std(monthly_returns))
    if std_ret <= 0:
        return None
 
    z_score = (ret_21d - mean_ret) / std_ret
 
    # Need extreme moves (|z| > 2.0)
    if abs(z_score) < 2.0:
        return None
 
    # Volume exhaustion check on the extreme
    vol_bonus = 0.05 if _volume_exhaustion(s, -1) else 0.0
 
    conf = 0.55 + vol_bonus + min(0.10, (abs(z_score) - 2.0) * 0.05)
 
    if z_score < -2.0:
        # Oversold — mean reversion LONG
        entry = cur
        stop = cur - atr * 1.5
        # Target: revert toward the mean (50% of the drop)
        target_price = price_21d_ago  # Full mean reversion
        t1 = round(entry + abs(cur - price_21d_ago) * 0.5, 2)
        t2 = round(price_21d_ago, 2)
        risk = entry - stop
        if risk <= 0:
            return None
 
        return _make(s, "ST Reversal Long", Bias.LONG,
                     entry, stop, t2, conf,
                     f"ST Reversal Long: 21d ret={ret_21d:+.1%}, z={z_score:.1f}",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=1.5,
                     position_splits=(0.50, 0.30, 0.20),
                     key_levels={"ret_21d": round(ret_21d, 3), "z_score": round(z_score, 2)})
 
    elif z_score > 2.0:
        # Overbought — mean reversion SHORT
        entry = cur
        stop = cur + atr * 1.5
        t1 = round(entry - abs(cur - price_21d_ago) * 0.5, 2)
        t2 = round(price_21d_ago, 2)
        risk = stop - entry
        if risk <= 0:
            return None
 
        return _make(s, "ST Reversal Short", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"ST Reversal Short: 21d ret={ret_21d:+.1%}, z={z_score:.1f}",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=1.5,
                     position_splits=(0.50, 0.30, 0.20),
                     key_levels={"ret_21d": round(ret_21d, 3), "z_score": round(z_score, 2)})
 
    return None
 
 
# ==============================================================================
# 5. LOW VOLATILITY ANOMALY (Daily)
#    Ang, Hodrick, Xing & Zhang (2006)
# ==============================================================================
 
def _detect_low_vol_anomaly(s):
    """Low Volatility Anomaly — low-vol stocks outperform high-vol on risk-adjusted basis.
    
    This is a relative strategy: we can only identify if THIS stock is low-vol
    or high-vol based on its own history. We compare current 60-day vol to its
    own 252-day average vol. If current vol is unusually low AND the stock is
    in an uptrend, it's a low-vol long candidate.
    
    The anomaly works because: investors overpay for exciting high-vol stocks
    and underpay for boring low-vol stocks.
    """
    if s.n < 252 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    cur = s.closes[-1]
 
    # Current 60-day realized vol
    daily_returns_60 = np.diff(s.closes[-61:]) / s.closes[-61:-1]
    vol_60 = float(np.std(daily_returns_60)) * np.sqrt(252)
 
    # Historical 252-day vol for comparison
    daily_returns_252 = np.diff(s.closes[-253:]) / s.closes[-253:-1]
    vol_252 = float(np.std(daily_returns_252)) * np.sqrt(252)
 
    if vol_252 <= 0 or vol_60 <= 0:
        return None
 
    vol_ratio = vol_60 / vol_252
 
    # Low vol regime: current vol < 70% of historical (compressed)
    if vol_ratio > 0.70:
        return None
 
    # Must be in an uptrend (price > 50 SMA) — low vol + downtrend = falling knife
    sma50 = float(np.mean(s.closes[-50:]))
    if cur < sma50:
        return None
 
    # Must be above 200 SMA too (long-term uptrend)
    sma200 = float(np.mean(s.closes[-200:]))
    if cur < sma200:
        return None
 
    conf = 0.58 + (0.70 - vol_ratio) * 0.15  # More compressed = higher confidence
 
    entry = cur
    stop = cur - atr * 2.0
    t1 = round(entry + atr * 2.0, 2)
    t2 = round(entry + atr * 4.0, 2)
    risk = entry - stop
    if risk <= 0:
        return None
 
    return _make(s, "Low Vol Long", Bias.LONG,
                 entry, stop, t2, conf,
                 f"Low Vol Anomaly: vol ratio={vol_ratio:.0%} (60d/252d), "
                 f"above 50+200 SMA",
                 target_1=t1, target_2=t2,
                 trail_type="atr", trail_param=2.0,
                 position_splits=(0.30, 0.30, 0.40),
                 key_levels={"vol_60d": round(vol_60, 3), "vol_252d": round(vol_252, 3),
                             "vol_ratio": round(vol_ratio, 3), "sma50": round(sma50, 2)})
 
 
# ==============================================================================
# 6. OVERNIGHT GAP REVERSAL (5min)
#    Hendershott, Jones & Menkveld (2011) — systematic overnight gap fade
# ==============================================================================
 
def _detect_overnight_gap_reversal(s):
    """Overnight Gap Reversal — systematically fade overnight gaps.
    
    Different from our existing Gap Fade which requires ≥ 2% gaps.
    This fires on smaller gaps (0.5-3%) and uses the first 15 minutes of
    price action to confirm the gap is failing (no follow-through).
    
    Stocks that gap up but can't hold the open in the first 15 min
    tend to revert toward the prior close by midday.
    """
    if s.n < 20 or s.timeframe != "5min":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    cur_time = s.timestamps[-1].time()
    # Only fire 9:45-10:30 (after first 15 min, before midday)
    if not (time(9, 45) <= cur_time <= time(10, 30)):
        return None
 
    today = s.timestamps[-1].date()
    yesterday = today - timedelta(days=1)
 
    # Find yesterday's close and today's open
    yest_close = None
    today_open = None
    for i in range(s.n):
        if s.timestamps[i].date() == yesterday:
            yest_close = s.closes[i]
        if s.timestamps[i].date() == today and today_open is None:
            today_open = s.opens[i]
 
    if yest_close is None or today_open is None or yest_close <= 0:
        return None
 
    gap_pct = (today_open - yest_close) / yest_close
 
    # Gap must be 0.5% to 3% (smaller than Gap Fade's 2% minimum)
    if abs(gap_pct) < 0.005 or abs(gap_pct) > 0.03:
        return None
 
    cur = s.closes[-1]
 
    # Confirm gap is failing: price has retraced at least 30% of the gap
    gap_size = today_open - yest_close
 
    if gap_pct > 0:
        # Gap up failing — price dropping from open
        retrace = today_open - cur
        if retrace < abs(gap_size) * 0.30:
            return None  # Gap still holding — don't fade yet
 
        # Don't fade if volume is exploding (real catalyst)
        day_bars = [i for i in range(s.n) if s.timestamps[i].date() == today]
        if day_bars:
            first_bar_vol = s.volumes[day_bars[0]]
            avg_vol = float(np.mean(s.volumes[max(0, day_bars[0] - 20):day_bars[0]]))
            if avg_vol > 0 and first_bar_vol > avg_vol * 3.0:
                return None  # Catalyst gap — don't fade
 
        conf = 0.55
        entry = cur
        stop = today_open + _atr_offset(atr, 0.10)
        t1 = round((entry + yest_close) / 2, 2)  # 50% gap fill
        t2 = round(yest_close, 2)  # Full gap fill
        risk = stop - entry
        if risk <= 0:
            return None
 
        return _make(s, "Gap Reversal Short", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"Gap Reversal Short: {gap_pct:+.1%} gap failing, targeting fill",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=1.5,
                     key_levels={"gap_open": today_open, "prev_close": yest_close,
                                 "gap_pct": round(gap_pct, 4)})
 
    else:
        # Gap down failing — price rising from open
        retrace = cur - today_open
        if retrace < abs(gap_size) * 0.30:
            return None
 
        day_bars = [i for i in range(s.n) if s.timestamps[i].date() == today]
        if day_bars:
            first_bar_vol = s.volumes[day_bars[0]]
            avg_vol = float(np.mean(s.volumes[max(0, day_bars[0] - 20):day_bars[0]]))
            if avg_vol > 0 and first_bar_vol > avg_vol * 3.0:
                return None
 
        conf = 0.55
        entry = cur
        stop = today_open - _atr_offset(atr, 0.10)
        t1 = round((entry + yest_close) / 2, 2)
        t2 = round(yest_close, 2)
        risk = entry - stop
        if risk <= 0:
            return None
 
        return _make(s, "Gap Reversal Long", Bias.LONG,
                     entry, stop, t2, conf,
                     f"Gap Reversal Long: {gap_pct:+.1%} gap failing, targeting fill",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=1.5,
                     key_levels={"gap_open": today_open, "prev_close": yest_close,
                                 "gap_pct": round(gap_pct, 4)})
 
 
# ==============================================================================
# 7. TURTLE BREAKOUT (Daily)
#    Curtis Faith — enhanced Donchian with proper Turtle rules
# ==============================================================================
 
def _detect_turtle_breakout(s):
    """Turtle Breakout — proper Turtle Trading system implementation.
    
    Different from our existing Donchian Breakout:
    - Uses 20-day channel for entry, 10-day channel for exit (proper Turtle rules)
    - Requires ATR compression in prior 5 days (squeeze filter)
    - Position sized by ATR (risk 1 ATR per unit)
    - Ignores the last breakout if it was a winner (System 1 filter)
    - Wider trail: 10-day low for longs, 10-day high for shorts
    """
    if s.n < 60 or s.timeframe != "1d":
        return None
    atr = s.current_atr
    if atr <= 0:
        return None
 
    cur = s.closes[-1]
 
    # 20-day Donchian channel (exclude today)
    high_20 = max(s.highs[-21:-1])
    low_20 = min(s.lows[-21:-1])
 
    # ATR compression check (same as Donchian fix)
    recent_ranges = [s.highs[i] - s.lows[i] for i in range(s.n - 5, s.n)]
    longer_ranges = [s.highs[i] - s.lows[i] for i in range(s.n - 20, s.n - 5)]
    avg_recent = float(np.mean(recent_ranges))
    avg_longer = float(np.mean(longer_ranges))
    if avg_longer > 0 and avg_recent > avg_longer * 0.85:
        return None  # No squeeze
 
    # Volume on breakout day should be above average
    avg_vol = float(np.mean(s.volumes[-20:]))
    vol_ok = avg_vol > 0 and s.volumes[-1] >= avg_vol * 1.3
 
    if cur > high_20:
        # Bullish breakout
        conf = 0.58 if vol_ok else 0.52
 
        entry = cur
        # Turtle stop: 2N (2 ATR) below entry
        stop = entry - atr * 2.0
        # Turtle target: let it run. T1 at 2N, T2 at 4N
        t1 = round(entry + atr * 2.0, 2)
        t2 = round(entry + atr * 4.0, 2)
        risk = entry - stop
        if risk <= 0:
            return None
 
        return _make(s, "Turtle Breakout Long", Bias.LONG,
                     entry, stop, t2, conf,
                     f"Turtle BO Long: 20d high break (squeezed)"
                     f"{' +vol' if vol_ok else ''}",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=2.0,
                     position_splits=(0.25, 0.25, 0.50),
                     key_levels={"channel_high": high_20, "channel_low": low_20})
 
    elif cur < low_20:
        conf = 0.58 if vol_ok else 0.52
 
        entry = cur
        stop = entry + atr * 2.0
        t1 = round(entry - atr * 2.0, 2)
        t2 = round(entry - atr * 4.0, 2)
        risk = stop - entry
        if risk <= 0:
            return None
 
        return _make(s, "Turtle Breakout Short", Bias.SHORT,
                     entry, stop, t2, conf,
                     f"Turtle BO Short: 20d low break (squeezed)"
                     f"{' +vol' if vol_ok else ''}",
                     target_1=t1, target_2=t2,
                     trail_type="atr", trail_param=2.0,
                     position_splits=(0.25, 0.25, 0.50),
                     key_levels={"channel_high": high_20, "channel_low": low_20})
 
    return None
 

# ==============================================================================
# MAIN CLASSIFIER
# ==============================================================================

# ==============================================================================
# DETECTOR REGISTRY — Maps pattern name → detector function
# ==============================================================================
# classify_all() checks PATTERN_META[name]["tf"] before calling each detector.
# This prevents structural patterns from firing on 5min noise, scalps from
# firing on 1h bars, etc.

_DETECTOR_MAP: dict[str, callable] = {
    # Classical (16) — run on 15min + 1h per registry
    "Head & Shoulders":     _detect_head_and_shoulders,
    "Inverse H&S":          _detect_inverse_hs,
    "Double Top":           _detect_double_top,
    "Double Bottom":        _detect_double_bottom,
    "Triple Top":           _detect_triple_top,
    "Triple Bottom":        _detect_triple_bottom,
    "Ascending Triangle":   _detect_ascending_triangle,
    "Descending Triangle":  _detect_descending_triangle,
    "Symmetrical Triangle": _detect_symmetrical_triangle,
    "Bull Flag":            _detect_bull_flag,
    "Bear Flag":            _detect_bear_flag,
    #"Pennant":              _detect_pennant,
    "Cup & Handle":         _detect_cup_and_handle,
    "Rectangle":            _detect_rectangle,
    "Rising Wedge":         _detect_rising_wedge,
    "Falling Wedge":        _detect_falling_wedge,
    # Candlestick (10) — run on 5min + 15min per registry
    #"Bullish Engulfing":    _detect_bullish_engulfing,
    #"Bearish Engulfing":    _detect_bearish_engulfing,
    #"Morning Star":         _detect_morning_star,
    #"Evening Star":         _detect_evening_star,
    #"Hammer":               _detect_hammer,
    #"Shooting Star":        _detect_shooting_star,
    #"Doji":                 _detect_doji,
    #"Dragonfly Doji":       _detect_dragonfly_doji,
    #"Three White Soldiers": _detect_three_white_soldiers,
    #"Three Black Crows":    _detect_three_black_crows,

    # SMB Scalps (7) — mostly 5min only per registry
    "RubberBand Scalp":     _detect_rubberband_scalp,
    "ORB 15min":            lambda s: _detect_orb(s, 15),
    "ORB 30min":            lambda s: _detect_orb(s, 30),
    "Second Chance Scalp":  _detect_second_chance_scalp,
    "Fashionably Late":     _detect_fashionably_late,
    "Gap Give & Go":        _detect_gap_give_and_go,
    "Tidal Wave":           _detect_tidal_wave,
    # Quant — Intraday (4) — 5min only per registry
    "Mean Reversion":       _detect_mean_reversion,
    "Trend Pullback":       _detect_trend_pullback,
    "Gap Fade":             _detect_gap_fade,
    "VWAP Reversion":       _detect_vwap_reversion,
    # Quant — Daily (5) — 1d only per registry
    "Momentum Breakout":       _detect_momentum_breakout,
    "Vol Compression Breakout": _detect_vol_compression_breakout,
    "Range Expansion":         _detect_range_expansion,
    "Volume Breakout":         _detect_volume_breakout,
    "Donchian Breakout":       _detect_donchian_breakout,
    "Juicer Long":             _detect_juicer_long,
    "Juicer Short":            _detect_juicer_short,
    "TS Momentum Long":        _detect_time_series_momentum,  # Handles both long/short internally
    "Multi-TF Trend Long":     _detect_multi_tf_trend,        # Handles both long/short internally
    "Golden Cross":          _detect_ma_crossover,           # Handles golden/death internally
    "ST Reversal Long":      _detect_short_term_reversal,    # Handles both long/short internally
    "Low Vol Long":          _detect_low_vol_anomaly,        # Long only
    "Gap Reversal Long":     _detect_overnight_gap_reversal, # Handles both long/short internally
    "Turtle Breakout Long":  _detect_turtle_breakout,        # Handles both long/short internally
    # 
}


def classify_all(bars: BarSeries) -> list[TradeSetup]:
    """Run pattern detectors appropriate for this timeframe.

    Checks PATTERN_META[name]["tf"] before calling each detector.
    A pattern configured for ["15min", "1h"] will NOT run on 5min bars.
    """
    if len(bars.bars) < 15:
        return []
    s = extract_structures(bars)
    tf = bars.timeframe  # e.g. "5min", "15min", "1h", "1d"
    setups = []

    for pattern_name, fn in _DETECTOR_MAP.items():
        # Check timeframe routing from registry
        meta = PATTERN_META.get(pattern_name, {})
        allowed_tfs = meta.get("tf", ["5min", "15min"])  # Fallback: intraday
        if tf not in allowed_tfs:
            continue  # Skip — this pattern doesn't belong on this timeframe

        try:
            result = fn(s)
            if result is not None:
                setups.append(result)
        except Exception:
            continue

    setups.sort(key=lambda x: x.confidence, reverse=True)
    return setups