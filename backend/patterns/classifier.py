"""
patterns/classifier.py — Structure-first pattern classification.

Pipeline:
  BarSeries → extract structures → geometric matching → TradeSetup

This replaces the old edgefinder_patterns.py. Instead of 21 separate
detector classes with duplicated logic, we:
  1. Extract structures ONCE (zigzag, trendlines, S/R, indicators)
  2. Run geometric pattern rules against those structures
  3. Each pattern is a function, not a class

Patterns implemented:
  Classical (12): H&S, Inv H&S, Double Top/Bottom, Asc/Desc/Sym Triangle,
                  Bull/Bear Flag, Cup & Handle, Rising/Falling Wedge
  SMB Scalps (11): RubberBand, HitchHiker, ORB 15/30, Second Chance,
                   BackSide, Fashionably Late, Spencer, Gap Give & Go,
                   Tidal Wave, Breaking News
  Quant (2): Momentum Breakout, Vol Compression Breakout
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
    compression_ratio,
)
from backend.structures.support_resistance import (
    cluster_levels, detect_breakouts, nearest_level, neckline_from_swings,
)
from backend.structures.indicators import (
    true_range, wilder_atr, atr_ratio, sma, ema, ema_last,
)
from backend.patterns.registry import (
    TradeSetup, Bias, PatternCategory, PATTERN_META,
)


# ==============================================================================
# STRUCTURE EXTRACTION (run once per scan)
# ==============================================================================

class ExtractedStructures:
    """All structural primitives extracted from a BarSeries."""
    def __init__(self, bars: BarSeries):
        b = bars.bars
        self.bars = b
        self.n = len(b)
        self.symbol = bars.symbol
        self.timeframe = bars.timeframe

        # NumPy arrays
        self.closes = np.array([x.close for x in b], dtype=np.float64)
        self.highs = np.array([x.high for x in b], dtype=np.float64)
        self.lows = np.array([x.low for x in b], dtype=np.float64)
        self.opens = np.array([x.open for x in b], dtype=np.float64)
        self.volumes = np.array([x.volume for x in b], dtype=np.float64)
        self.timestamps = [x.timestamp for x in b]

        # Zigzag swings
        threshold = adaptive_zigzag_threshold(bars.timeframe)
        self.zz_swings = zigzag(
            self.highs, self.lows, threshold,
            timestamps=self.timestamps, volumes=self.volumes,
        )
        self.zz_highs = swing_highs_from_zigzag(self.zz_swings)
        self.zz_lows = swing_lows_from_zigzag(self.zz_swings)

        # Order-based swings
        order = adaptive_order(bars.timeframe)
        self.sw_high_idx = find_swing_highs(self.highs, order=order)
        self.sw_low_idx = find_swing_lows(self.lows, order=order)

        # S/R levels
        self.sr_levels = cluster_levels(self.zz_swings, tolerance_pct=0.8, min_touches=2)

        # ATR
        self.atr_series = wilder_atr(self.highs, self.lows, self.closes, period=14)
        self.current_atr = float(self.atr_series[-1]) if not np.isnan(self.atr_series[-1]) else 0

        # Intraday helpers
        self.day_open_idx = self._find_day_open()

    def _find_day_open(self) -> int:
        """Find the index of today's market open bar (9:30 ET)."""
        if self.n == 0:
            return 0
        today = self.timestamps[-1].date()
        for i, ts in enumerate(self.timestamps):
            if ts.date() == today and ts.time() >= time(9, 30):
                return i
        # Fallback: most recent day's first bar
        for i, ts in enumerate(self.timestamps):
            if ts.date() == today:
                return i
        return max(0, self.n - 20)


def extract_structures(bars: BarSeries) -> ExtractedStructures:
    """Extract all structural primitives from bar data."""
    return ExtractedStructures(bars)


# ==============================================================================
# MAIN CLASSIFIER
# ==============================================================================

def classify_all(bars: BarSeries) -> list[TradeSetup]:
    """
    Run ALL pattern detectors on a BarSeries.
    Returns list of detected TradeSetups sorted by confidence.
    """
    if len(bars.bars) < 15:
        return []

    s = extract_structures(bars)
    setups: list[TradeSetup] = []

    # Classical patterns (from zigzag structures)
    for fn in [
        _detect_head_and_shoulders, _detect_inverse_hs,
        _detect_double_top, _detect_double_bottom,
        _detect_ascending_triangle, _detect_descending_triangle,
        _detect_symmetrical_triangle,
        _detect_bull_flag, _detect_bear_flag,
        _detect_cup_and_handle,
        _detect_rising_wedge, _detect_falling_wedge,
    ]:
        try:
            result = fn(s)
            if result is not None:
                setups.append(result)
        except Exception:
            continue

    # SMB Scalp patterns (with time-of-day logic)
    for fn in [
        _detect_rubberband, _detect_hitchhiker,
        lambda st: _detect_orb(st, minutes=15),
        lambda st: _detect_orb(st, minutes=30),
        _detect_second_chance, _detect_backside,
        _detect_fashionably_late, _detect_spencer,
        _detect_gap_give_and_go, _detect_tidal_wave,
        _detect_breaking_news,
    ]:
        try:
            result = fn(s)
            if result is not None:
                setups.append(result)
        except Exception:
            continue

    # Quant patterns
    for fn in [_detect_momentum_breakout, _detect_vol_compression_breakout]:
        try:
            result = fn(s)
            if result is not None:
                setups.append(result)
        except Exception:
            continue

    setups.sort(key=lambda x: x.confidence, reverse=True)
    return setups


# ==============================================================================
# HELPER
# ==============================================================================

def _make_setup(s: ExtractedStructures, name: str, bias: Bias,
                entry: float, stop: float, target: float,
                confidence: float, desc: str, **kw) -> TradeSetup:
    """Convenience constructor with auto risk/reward and metadata."""
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    reward = abs(target - entry)
    rr = round(reward / risk, 2)
    meta = PATTERN_META.get(name, {})
    return TradeSetup(
        pattern_name=name, category=meta.get("cat", PatternCategory.CLASSICAL),
        symbol=s.symbol, bias=bias,
        entry_price=round(entry, 2), stop_loss=round(stop, 2),
        target_price=round(target, 2), risk_reward_ratio=rr,
        confidence=round(min(0.95, confidence), 2),
        detected_at=s.timestamps[-1], description=desc,
        strategy_type=meta.get("type", "breakout"),
        win_rate=meta.get("wr", 0.5),
        timeframe_detected=s.timeframe,
        **kw,
    )


def _is_in_time(ts: datetime, sh: int, sm: int, eh: int, em: int) -> bool:
    """Check if timestamp is within a time window (ET assumed)."""
    t = ts.time()
    return time(sh, sm) <= t <= time(eh, em)


# ==============================================================================
# CLASSICAL PATTERNS (structure-first)
# ==============================================================================

def _detect_head_and_shoulders(s: ExtractedStructures) -> Optional[TradeSetup]:
    """H&S: 3 swing highs — middle highest, shoulders within tolerance, neckline break."""
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 2:
        return None
    for i in range(len(s.zz_highs) - 2):
        ls, head, rs = s.zz_highs[i], s.zz_highs[i + 1], s.zz_highs[i + 2]
        if not (head.price > ls.price and head.price > rs.price):
            continue
        if abs(ls.price - rs.price) / ls.price > 0.03:
            continue
        # Neckline from lows between shoulders
        between_lows = [l for l in s.zz_lows if ls.index < l.index < rs.index]
        if not between_lows:
            continue
        neckline = min(l.price for l in between_lows)
        if s.closes[-1] >= neckline:
            continue
        entry = round(neckline - 0.02, 2)
        stop = round(rs.price + 0.02, 2)
        target = round(entry - (head.price - neckline), 2)
        conf = 0.60
        sym = 1 - abs(ls.price - rs.price) / ls.price
        conf += sym * 0.15
        return _make_setup(s, "Head & Shoulders", Bias.SHORT, entry, stop, target, conf,
                           f"H&S: head@{head.price:.2f}, neckline@{neckline:.2f}",
                           key_levels={"left_shoulder": ls.price, "head": head.price,
                                       "right_shoulder": rs.price, "neckline": neckline})
    return None


def _detect_inverse_hs(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Inverse H&S: 3 swing lows — middle lowest, neckline break above."""
    if len(s.zz_lows) < 3 or len(s.zz_highs) < 2:
        return None
    for i in range(len(s.zz_lows) - 2):
        ls, head, rs = s.zz_lows[i], s.zz_lows[i + 1], s.zz_lows[i + 2]
        if not (head.price < ls.price and head.price < rs.price):
            continue
        if abs(ls.price - rs.price) / ls.price > 0.03:
            continue
        between_highs = [h for h in s.zz_highs if ls.index < h.index < rs.index]
        if not between_highs:
            continue
        neckline = max(h.price for h in between_highs)
        if s.closes[-1] <= neckline:
            continue
        entry = round(neckline + 0.02, 2)
        stop = round(rs.price - 0.02, 2)
        target = round(entry + (neckline - head.price), 2)
        conf = 0.60
        sym = 1 - abs(ls.price - rs.price) / ls.price
        conf += sym * 0.15
        return _make_setup(s, "Inverse H&S", Bias.LONG, entry, stop, target, conf,
                           f"Inv H&S: head@{head.price:.2f}, neckline@{neckline:.2f}",
                           key_levels={"head": head.price, "neckline": neckline})
    return None


def _detect_double_top(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Double Top: 2 swing highs within tolerance, valley break below."""
    if len(s.zz_highs) < 2:
        return None
    h1, h2 = s.zz_highs[-2], s.zz_highs[-1]
    if abs(h1.price - h2.price) / h1.price > 0.015:
        return None
    if h2.index - h1.index < 5:
        return None
    valley = min(s.lows[h1.index:h2.index + 1])
    if s.closes[-1] > valley:
        return None
    entry = round(valley - 0.02, 2)
    stop = round(max(h1.price, h2.price) + 0.02, 2)
    target = round(entry - (max(h1.price, h2.price) - valley), 2)
    return _make_setup(s, "Double Top", Bias.SHORT, entry, stop, target, 0.63,
                       f"Double Top at {max(h1.price, h2.price):.2f}",
                       key_levels={"top1": h1.price, "top2": h2.price, "valley": valley})


def _detect_double_bottom(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Double Bottom: 2 swing lows within tolerance, peak break above."""
    if len(s.zz_lows) < 2:
        return None
    l1, l2 = s.zz_lows[-2], s.zz_lows[-1]
    if abs(l1.price - l2.price) / l1.price > 0.015:
        return None
    if l2.index - l1.index < 5:
        return None
    peak = max(s.highs[l1.index:l2.index + 1])
    if s.closes[-1] < peak:
        return None
    entry = round(peak + 0.02, 2)
    stop = round(min(l1.price, l2.price) - 0.02, 2)
    target = round(entry + (peak - min(l1.price, l2.price)), 2)
    return _make_setup(s, "Double Bottom", Bias.LONG, entry, stop, target, 0.65,
                       f"Double Bottom at {min(l1.price, l2.price):.2f}",
                       key_levels={"bottom1": l1.price, "bottom2": l2.price, "peak": peak})


def _detect_ascending_triangle(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Ascending Triangle: flat resistance + rising support, breakout above."""
    if len(s.zz_highs) < 2 or len(s.zz_lows) < 2:
        return None
    upper_tl = fit_trendline(s.zz_highs[-3:] if len(s.zz_highs) >= 3 else s.zz_highs[-2:])
    lower_tl = fit_trendline(s.zz_lows[-3:] if len(s.zz_lows) >= 3 else s.zz_lows[-2:])
    if upper_tl is None or lower_tl is None:
        return None
    if not is_flat_line(upper_tl, tolerance_pct=0.15):
        return None
    if lower_tl.slope <= 0:
        return None
    resistance = upper_tl.price_at(upper_tl.end_index)
    if s.closes[-1] < resistance:
        return None
    support = s.zz_lows[-1].price
    entry = round(resistance + 0.02, 2)
    stop = round(support - 0.02, 2)
    target = round(entry + (resistance - support), 2)
    return _make_setup(s, "Ascending Triangle", Bias.LONG, entry, stop, target, 0.62,
                       f"Asc Triangle: flat top {resistance:.2f}, rising lows",
                       key_levels={"resistance": resistance, "support": support})


def _detect_descending_triangle(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Descending Triangle: flat support + falling resistance, breakdown below."""
    if len(s.zz_highs) < 2 or len(s.zz_lows) < 2:
        return None
    lower_tl = fit_trendline(s.zz_lows[-3:] if len(s.zz_lows) >= 3 else s.zz_lows[-2:])
    upper_tl = fit_trendline(s.zz_highs[-3:] if len(s.zz_highs) >= 3 else s.zz_highs[-2:])
    if upper_tl is None or lower_tl is None:
        return None
    if not is_flat_line(lower_tl, tolerance_pct=0.15):
        return None
    if upper_tl.slope >= 0:
        return None
    support = lower_tl.price_at(lower_tl.end_index)
    if s.closes[-1] > support:
        return None
    resistance = s.zz_highs[-1].price
    entry = round(support - 0.02, 2)
    stop = round(resistance + 0.02, 2)
    target = round(entry - (resistance - support), 2)
    return _make_setup(s, "Descending Triangle", Bias.SHORT, entry, stop, target, 0.60,
                       f"Desc Triangle: flat bottom {support:.2f}, falling highs",
                       key_levels={"support": support, "resistance": resistance})


def _detect_symmetrical_triangle(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Symmetrical Triangle: converging trendlines, breakout either direction."""
    if len(s.zz_highs) < 2 or len(s.zz_lows) < 2:
        return None
    upper_tl = fit_trendline(s.zz_highs[-3:] if len(s.zz_highs) >= 3 else s.zz_highs[-2:])
    lower_tl = fit_trendline(s.zz_lows[-3:] if len(s.zz_lows) >= 3 else s.zz_lows[-2:])
    if upper_tl is None or lower_tl is None:
        return None
    if upper_tl.slope >= 0 or lower_tl.slope <= 0:
        return None  # Need falling highs and rising lows
    upper_price = upper_tl.price_at(s.n - 1)
    lower_price = lower_tl.price_at(s.n - 1)
    if upper_price <= lower_price:
        return None
    current = s.closes[-1]
    tri_range = upper_price - lower_price
    if current > upper_price:
        entry = round(upper_price + 0.02, 2)
        stop = round(lower_price - 0.02, 2)
        target = round(entry + tri_range, 2)
        bias = Bias.LONG
        desc = "Sym Triangle breakout above"
    elif current < lower_price:
        entry = round(lower_price - 0.02, 2)
        stop = round(upper_price + 0.02, 2)
        target = round(entry - tri_range, 2)
        bias = Bias.SHORT
        desc = "Sym Triangle breakdown below"
    else:
        return None
    return _make_setup(s, "Symmetrical Triangle", bias, entry, stop, target, 0.58, desc,
                       key_levels={"upper": upper_price, "lower": lower_price})


def _detect_bull_flag(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Bull Flag: impulse up + downward-sloping consolidation + breakout."""
    if len(s.zz_lows) < 1 or len(s.zz_highs) < 1:
        return None
    # Find pole: a swing low followed by swing high with >2% gain
    for li in range(len(s.zz_lows)):
        pole_lo = s.zz_lows[li]
        post_highs = [h for h in s.zz_highs if h.index > pole_lo.index]
        if not post_highs:
            continue
        pole_hi = post_highs[0]
        pole_pct = (pole_hi.price - pole_lo.price) / pole_lo.price
        if pole_pct < 0.02:
            continue
        # Flag: bars after pole high
        flag_start = pole_hi.index
        if flag_start >= s.n - 3:
            continue
        flag_bars = s.bars[flag_start:]
        if len(flag_bars) < 3 or len(flag_bars) > 30:
            continue
        flag_low = min(b.low for b in flag_bars)
        retrace = (pole_hi.price - flag_low) / (pole_hi.price - pole_lo.price)
        if retrace > 0.50:
            continue
        if s.closes[-1] <= pole_hi.price:
            continue  # Need breakout above flag high
        entry = round(pole_hi.price * 1.001, 2)
        stop = round(flag_low * 0.998, 2)
        target = round(entry + (pole_hi.price - pole_lo.price), 2)
        return _make_setup(s, "Bull Flag", Bias.LONG, entry, stop, target, 0.60,
                           f"Bull Flag: {pole_pct:.1%} pole, {len(flag_bars)}-bar flag",
                           key_levels={"pole_hi": pole_hi.price, "pole_lo": pole_lo.price,
                                       "flag_low": flag_low})
    return None


def _detect_bear_flag(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Bear Flag: impulse down + upward-sloping consolidation + breakdown."""
    if len(s.zz_highs) < 1 or len(s.zz_lows) < 1:
        return None
    for hi in range(len(s.zz_highs)):
        pole_hi = s.zz_highs[hi]
        post_lows = [l for l in s.zz_lows if l.index > pole_hi.index]
        if not post_lows:
            continue
        pole_lo = post_lows[0]
        pole_pct = (pole_hi.price - pole_lo.price) / pole_hi.price
        if pole_pct < 0.02:
            continue
        flag_start = pole_lo.index
        if flag_start >= s.n - 3:
            continue
        flag_bars = s.bars[flag_start:]
        if len(flag_bars) < 3 or len(flag_bars) > 30:
            continue
        flag_high = max(b.high for b in flag_bars)
        retrace = (flag_high - pole_lo.price) / (pole_hi.price - pole_lo.price)
        if retrace > 0.50:
            continue
        if s.closes[-1] >= pole_lo.price:
            continue
        entry = round(pole_lo.price * 0.999, 2)
        stop = round(flag_high * 1.002, 2)
        target = round(entry - (pole_hi.price - pole_lo.price), 2)
        return _make_setup(s, "Bear Flag", Bias.SHORT, entry, stop, target, 0.58,
                           f"Bear Flag: {pole_pct:.1%} pole",
                           key_levels={"pole_hi": pole_hi.price, "pole_lo": pole_lo.price,
                                       "flag_high": flag_high})
    return None


def _detect_cup_and_handle(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Cup & Handle: U-shaped low + shallow handle retrace + breakout."""
    if len(s.zz_lows) < 1 or s.n < 40:
        return None
    cup_low = s.zz_lows[-1]
    left_rim = max(s.highs[:cup_low.index]) if cup_low.index > 5 else None
    right_rim = max(s.highs[cup_low.index:]) if cup_low.index < s.n - 5 else None
    if left_rim is None or right_rim is None:
        return None
    rim = min(left_rim, right_rim)
    cup_depth = rim - cup_low.price
    if cup_depth <= 0:
        return None
    handle = s.bars[-min(10, s.n):]
    handle_low = min(b.low for b in handle)
    retrace = (rim - handle_low) / cup_depth
    if retrace > 0.50 or retrace < 0.10:
        return None
    if s.closes[-1] < rim:
        return None
    entry = round(rim + 0.02, 2)
    stop = round(handle_low - 0.02, 2)
    target = round(entry + cup_depth, 2)
    return _make_setup(s, "Cup & Handle", Bias.LONG, entry, stop, target, 0.63,
                       f"Cup & Handle: rim {rim:.2f}, depth {cup_depth:.2f}",
                       key_levels={"rim": rim, "cup_low": cup_low.price, "handle_low": handle_low})


def _detect_rising_wedge(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Rising Wedge: both trendlines sloping up, converging. Bearish breakdown."""
    if len(s.zz_highs) < 2 or len(s.zz_lows) < 2:
        return None
    upper = fit_trendline(s.zz_highs[-3:] if len(s.zz_highs) >= 3 else s.zz_highs[-2:])
    lower = fit_trendline(s.zz_lows[-3:] if len(s.zz_lows) >= 3 else s.zz_lows[-2:])
    if upper is None or lower is None:
        return None
    if not (upper.slope > 0 and lower.slope > 0):
        return None
    if upper.slope >= lower.slope:
        return None  # Must be converging (upper slope < lower slope)
    lower_price = lower.price_at(s.n - 1)
    if s.closes[-1] > lower_price:
        return None
    upper_price = upper.price_at(s.n - 1)
    entry = round(lower_price - 0.02, 2)
    stop = round(upper_price + 0.02, 2)
    wedge_height = upper_price - lower_price
    target = round(entry - wedge_height, 2)
    return _make_setup(s, "Rising Wedge", Bias.SHORT, entry, stop, target, 0.62,
                       f"Rising Wedge breakdown",
                       key_levels={"upper": upper_price, "lower": lower_price})


def _detect_falling_wedge(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Falling Wedge: both trendlines sloping down, converging. Bullish breakout."""
    if len(s.zz_highs) < 2 or len(s.zz_lows) < 2:
        return None
    upper = fit_trendline(s.zz_highs[-3:] if len(s.zz_highs) >= 3 else s.zz_highs[-2:])
    lower = fit_trendline(s.zz_lows[-3:] if len(s.zz_lows) >= 3 else s.zz_lows[-2:])
    if upper is None or lower is None:
        return None
    if not (upper.slope < 0 and lower.slope < 0):
        return None
    if upper.slope <= lower.slope:
        return None  # Must converge (upper slope > lower slope, both negative)
    upper_price = upper.price_at(s.n - 1)
    if s.closes[-1] < upper_price:
        return None
    lower_price = lower.price_at(s.n - 1)
    entry = round(upper_price + 0.02, 2)
    stop = round(lower_price - 0.02, 2)
    target = round(entry + (upper_price - lower_price), 2)
    return _make_setup(s, "Falling Wedge", Bias.LONG, entry, stop, target, 0.62,
                       f"Falling Wedge breakout",
                       key_levels={"upper": upper_price, "lower": lower_price})


# ==============================================================================
# SMB SCALP PATTERNS (time-of-day logic preserved)
# ==============================================================================

def _detect_rubberband(s: ExtractedStructures) -> Optional[TradeSetup]:
    """RubberBand: Extended move + acceleration + snapback. Intraday."""
    if s.n < 20 or s.current_atr <= 0:
        return None
    oi = s.day_open_idx
    if oi >= s.n - 5:
        return None
    open_price = s.bars[oi].open
    day_bars = s.bars[oi:]
    day_low = min(b.low for b in day_bars)
    day_low_idx = oi + min(range(len(day_bars)), key=lambda i: day_bars[i].low)
    extension = open_price - day_low
    atrs = extension / s.current_atr if s.current_atr > 0 else 0
    if atrs < 1.0 or day_low_idx - oi < 5:
        return None
    # Check acceleration
    mid = (oi + day_low_idx) // 2
    first_ranges = [s.bars[i].high - s.bars[i].low for i in range(oi, mid)] if mid > oi else [0]
    second_ranges = [s.bars[i].high - s.bars[i].low for i in range(mid, day_low_idx)] if day_low_idx > mid else [0]
    if np.mean(second_ranges) < np.mean(first_ranges) * 1.2:
        return None
    # Check snapback
    for i in range(day_low_idx + 1, min(day_low_idx + 10, s.n)):
        if s.bars[i].close > s.bars[i].open:
            cleared = sum(1 for j in range(max(day_low_idx, i - 5), i) if s.bars[i].high > s.bars[j].high)
            if cleared >= 2:
                entry = round(s.bars[i].high + 0.02, 2)
                stop = round(day_low - 0.02, 2)
                risk = entry - stop
                if risk <= 0: continue
                target = round(entry + 2 * risk, 2)
                conf = 0.45
                if atrs >= 3: conf += 0.15
                if _is_in_time(s.bars[i].timestamp, 10, 0, 10, 45): conf += 0.10
                return _make_setup(s, "RubberBand Scalp", Bias.LONG, entry, stop, target, conf,
                                   f"RubberBand: {atrs:.1f} ATR extension, snapback",
                                   max_attempts=2, exit_strategy="1/3 at 1:1, 1/3 at 2:1, 1/3 into VWAP",
                                   ideal_time="10:00-10:45 AM ET",
                                   key_levels={"day_low": day_low})
    return None


def _detect_hitchhiker(s: ExtractedStructures) -> Optional[TradeSetup]:
    """HitchHiker: Drive off open + consolidation in upper 1/3 + breakout."""
    if s.n < 15:
        return None
    oi = s.day_open_idx
    if s.n - oi < 10:
        return None
    # Check for drive off open
    drive_end = None
    for i in range(oi + 3, min(oi + 10, s.n)):
        if s.bars[i].close > s.bars[oi].open * 1.005:
            drive_end = i
            break
    if drive_end is None:
        return None
    day_high = max(b.high for b in s.bars[oi:])
    day_low = min(b.low for b in s.bars[oi:])
    day_range = day_high - day_low
    if day_range <= 0:
        return None
    upper_third = day_low + day_range * (2 / 3)
    # Look for consolidation in upper third
    for c_start in range(drive_end + 1, min(drive_end + 16, s.n - 5)):
        c_end = min(c_start + 8, s.n - 1)
        c_bars = s.bars[c_start:c_end]
        c_high = max(b.high for b in c_bars)
        c_low = min(b.low for b in c_bars)
        if c_low < upper_third or (c_high - c_low) > day_range * 0.40:
            continue
        if c_end + 1 >= s.n:
            continue
        if s.bars[c_end + 1].high > c_high:
            entry = round(c_high + 0.02, 2)
            stop = round(c_low - 0.02, 2)
            risk = entry - stop
            if risk <= 0: continue
            target = round(entry + risk * 1.9, 2)
            conf = 0.50
            if _is_in_time(s.bars[c_end + 1].timestamp, 9, 30, 9, 59): conf += 0.10
            return _make_setup(s, "HitchHiker Scalp", Bias.LONG, entry, stop, target, conf,
                               f"HitchHiker: consol in upper 1/3, breakout",
                               max_attempts=1, exit_strategy="1/2 first wave, 1/2 second wave",
                               ideal_time="9:30-9:59 AM ET",
                               key_levels={"consol_high": c_high, "consol_low": c_low})
    return None


def _detect_orb(s: ExtractedStructures, minutes: int = 15) -> Optional[TradeSetup]:
    """ORB: Time-based opening range, breakout/breakdown."""
    if s.n < 10:
        return None
    oi = s.day_open_idx
    if oi >= s.n - 5:
        return None
    open_time = s.timestamps[oi]
    cutoff = open_time + timedelta(minutes=minutes)
    orb_end = oi
    for i in range(oi, s.n):
        if s.timestamps[i] <= cutoff:
            orb_end = i
        else:
            break
    if orb_end <= oi:
        return None
    orb_bars = s.bars[oi:orb_end + 1]
    orb_high = max(b.high for b in orb_bars)
    orb_low = min(b.low for b in orb_bars)
    orb_range = orb_high - orb_low
    if orb_range <= 0:
        return None
    name = f"ORB {minutes}min"
    for i in range(orb_end + 1, min(orb_end + 30, s.n)):
        if s.bars[i].close > orb_high:
            entry = round(orb_high + 0.02, 2)
            stop = round(s.bars[i].low - 0.02, 2)
            risk = entry - stop
            if risk <= 0 or risk > orb_range: continue
            target = round(entry + 2 * orb_range, 2)
            conf = 0.50
            orb_avg_vol = np.mean([b.volume for b in orb_bars])
            if s.bars[i].volume > orb_avg_vol * 1.5: conf += 0.15
            return _make_setup(s, name, Bias.LONG, entry, stop, target, conf,
                               f"ORB {minutes}min breakout above {orb_high:.2f}",
                               max_attempts=2, ideal_time=f"After {minutes}min ORB",
                               key_levels={"orb_high": orb_high, "orb_low": orb_low})
        if s.bars[i].close < orb_low:
            entry = round(orb_low - 0.02, 2)
            stop = round(s.bars[i].high + 0.02, 2)
            risk = stop - entry
            if risk <= 0 or risk > orb_range: continue
            target = round(entry - 2 * orb_range, 2)
            return _make_setup(s, name, Bias.SHORT, entry, stop, target, 0.55,
                               f"ORB {minutes}min breakdown below {orb_low:.2f}",
                               max_attempts=2, ideal_time=f"After {minutes}min ORB",
                               key_levels={"orb_high": orb_high, "orb_low": orb_low})
    return None


def _detect_second_chance(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Second Chance: Resistance break + pullback to support + bounce."""
    if s.n < 30 or len(s.sw_high_idx) < 2:
        return None
    # Find resistance cluster
    for i in range(len(s.sw_high_idx) - 1):
        level = s.highs[s.sw_high_idx[i]]
        tol = level * 0.003
        touches = [s.sw_high_idx[i]]
        for j in range(i + 1, len(s.sw_high_idx)):
            if abs(s.highs[s.sw_high_idx[j]] - level) < tol:
                touches.append(s.sw_high_idx[j])
        if len(touches) < 2:
            continue
        resistance = float(np.mean([s.highs[idx] for idx in touches]))
        # Look for breakout then pullback
        last = touches[-1]
        for k in range(last + 1, min(last + 20, s.n)):
            if s.closes[k] > resistance * 1.002:
                # Found breakout, now look for pullback
                for p in range(k + 1, min(k + 15, s.n)):
                    if s.lows[p] <= resistance * 1.003:
                        if p + 1 < s.n and s.closes[p + 1] > s.bars[p].high:
                            entry = round(s.closes[p + 1], 2)
                            stop = round(s.bars[p].low - 0.02, 2)
                            risk = entry - stop
                            if risk <= 0: continue
                            target = round(max(s.highs[k:p + 1]) , 2)
                            rr = (target - entry) / risk
                            if rr < 1.0: continue
                            conf = 0.45
                            if s.bars[k].volume > s.bars[p].volume * 1.3: conf += 0.15
                            return _make_setup(s, "Second Chance Scalp", Bias.LONG,
                                               entry, stop, target, conf,
                                               f"Second Chance: retest {resistance:.2f} held",
                                               max_attempts=2,
                                               ideal_time="9:59-10:44, 10:45-1:29 ET",
                                               key_levels={"resistance": resistance})
                break
    return None


def _detect_backside(s: ExtractedStructures) -> Optional[TradeSetup]:
    """BackSide: Below VWAP, HH+HL forming, EMA rising, targeting VWAP."""
    if s.n < 25:
        return None
    oi = s.day_open_idx
    if s.n - oi < 15:
        return None
    # Simple VWAP proxy: cumulative TP*V / cumV for today
    cum_vol = 0.0
    cum_tpv = 0.0
    vwap_val = s.closes[-1]
    for i in range(oi, s.n):
        tp = (s.highs[i] + s.lows[i] + s.closes[i]) / 3
        cum_vol += s.volumes[i]
        cum_tpv += tp * s.volumes[i]
        if cum_vol > 0:
            vwap_val = cum_tpv / cum_vol
    ema9 = ema_last(s.closes, 9)
    if s.closes[-1] >= vwap_val or s.closes[-1] < ema9:
        return None
    if s.n >= 12 and ema_last(s.closes, 9) <= ema_last(s.closes[:-3], 9):
        return None  # EMA not rising
    entry = round(s.highs[-1] + 0.02, 2)
    recent_lows = s.lows[-8:]
    hl = float(np.sort(recent_lows)[-2]) if len(recent_lows) >= 2 else s.lows[-1]
    stop = round(hl - 0.02, 2)
    target = round(vwap_val, 2)
    risk = entry - stop
    if risk <= 0 or target <= entry:
        return None
    conf = 0.48
    if _is_in_time(s.timestamps[-1], 10, 0, 10, 45): conf += 0.10
    return _make_setup(s, "BackSide Scalp", Bias.LONG, entry, stop, target, conf,
                       f"BackSide: HH+HL above 9 EMA, targeting VWAP {vwap_val:.2f}",
                       exit_strategy="Full exit at VWAP",
                       ideal_time="10:00-10:45, 10:46-1:30 ET",
                       key_levels={"vwap": round(vwap_val, 2)})


def _detect_fashionably_late(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Fashionably Late: 9 EMA crosses VWAP after extended move, measured move target."""
    if s.n < 30:
        return None
    oi = s.day_open_idx
    if s.n - oi < 20:
        return None
    # Compute VWAP and 9 EMA
    ema9_series = ema(s.closes, 9)
    cum_vol = 0.0; cum_tpv = 0.0
    vwap_series = np.zeros(s.n)
    for i in range(oi, s.n):
        tp = (s.highs[i] + s.lows[i] + s.closes[i]) / 3
        cum_vol += s.volumes[i]
        cum_tpv += tp * s.volumes[i]
        vwap_series[i] = cum_tpv / cum_vol if cum_vol > 0 else tp
    day_low = float(np.min(s.lows[oi:]))
    # Find EMA cross above VWAP
    for i in range(oi + 10, s.n):
        if np.isnan(ema9_series[i]) or np.isnan(ema9_series[i - 1]):
            continue
        if vwap_series[i] == 0 or vwap_series[i - 1] == 0:
            continue
        if ema9_series[i - 1] < vwap_series[i - 1] and ema9_series[i] >= vwap_series[i]:
            cross_price = vwap_series[i]
            measured = cross_price - day_low
            if measured <= 0:
                continue
            entry = round(cross_price, 2)
            stop = round(cross_price - measured / 3, 2)
            target = round(cross_price + measured, 2)
            conf = 0.55
            if _is_in_time(s.timestamps[i], 10, 0, 10, 45): conf += 0.10
            return _make_setup(s, "Fashionably Late", Bias.LONG, entry, stop, target, conf,
                               f"Fashionably Late: 9 EMA crossed VWAP, measured ${measured:.2f}",
                               ideal_time="10:00-10:45, 10:46-1:30 ET",
                               key_levels={"cross": round(cross_price, 2), "day_low": day_low})
    return None


def _detect_spencer(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Spencer: Tight consolidation near HOD + breakout."""
    if s.n < 30:
        return None
    oi = s.day_open_idx
    if s.n - oi < 25:
        return None
    day_high = max(b.high for b in s.bars[oi:])
    day_low = min(b.low for b in s.bars[oi:])
    day_range = day_high - day_low
    if day_range <= 0:
        return None
    upper_third = day_low + day_range * (2 / 3)
    for start in range(oi + 5, s.n - 10):
        window = s.bars[start:start + 10]
        c_high = max(b.high for b in window)
        c_low = min(b.low for b in window)
        c_range = c_high - c_low
        if c_low < upper_third or c_range > day_range * 0.35:
            continue
        break_idx = start + 10
        if break_idx >= s.n:
            continue
        if s.bars[break_idx].high > c_high:
            entry = round(c_high + 0.02, 2)
            stop = round(c_low - 0.02, 2)
            risk = entry - stop
            if risk <= 0: continue
            target = round(entry + 2 * c_range, 2)
            return _make_setup(s, "Spencer Scalp", Bias.LONG, entry, stop, target, 0.52,
                               f"Spencer: tight consol near HOD, breakout",
                               exit_strategy="1/4 at 1:1, 1/2 at 2:1, 1/4 at 3:1",
                               key_levels={"consol_high": c_high, "consol_low": c_low})
    return None


def _detect_gap_give_and_go(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Gap Give & Go: Gap up + initial drop + consolidation + break to fill gap."""
    if s.n < 15:
        return None
    oi = s.day_open_idx
    if oi == 0 or s.n - oi < 10:
        return None
    prev_close = s.bars[oi - 1].close
    open_price = s.bars[oi].open
    gap_pct = (open_price - prev_close) / prev_close * 100
    if gap_pct < 0.5:
        return None
    # Initial drop
    drop_end = None
    for i in range(oi + 1, min(oi + 6, s.n)):
        if s.bars[i].low < open_price * 0.995:
            drop_end = i
    if drop_end is None:
        return None
    # Consolidation
    for c_len in range(3, 8):
        c_end = drop_end + c_len
        if c_end >= s.n - 1:
            continue
        c_bars = s.bars[drop_end:c_end]
        c_high = max(b.high for b in c_bars)
        c_low = min(b.low for b in c_bars)
        if (c_high - c_low) > (open_price - c_low) * 0.50:
            continue
        if s.bars[c_end].high > c_high:
            entry = round(c_high + 0.02, 2)
            stop = round(c_low - 0.02, 2)
            risk = entry - stop
            if risk <= 0: continue
            target = round(open_price, 2)
            if (target - entry) / risk < 1.0: continue
            conf = 0.50
            if _is_in_time(s.timestamps[c_end], 9, 30, 9, 45): conf += 0.10
            return _make_setup(s, "Gap Give & Go", Bias.LONG, entry, stop, target, conf,
                               f"Gap G&G: {gap_pct:.1f}% gap, {c_len}-bar consol",
                               max_attempts=2, ideal_time="9:30-9:45 AM ET",
                               key_levels={"gap_open": open_price, "consol_low": c_low})
    return None


def _detect_tidal_wave(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Tidal Wave: Multiple touches of support with diminishing bounces + breakdown."""
    if s.n < 30 or len(s.sw_low_idx) < 2:
        return None
    order = adaptive_order(s.timeframe)
    for i in range(len(s.sw_low_idx)):
        support = s.lows[s.sw_low_idx[i]]
        tol = support * 0.003
        touches = [s.sw_low_idx[i]]
        for j in range(i + 1, len(s.sw_low_idx)):
            if abs(s.lows[s.sw_low_idx[j]] - support) < tol:
                touches.append(s.sw_low_idx[j])
        if len(touches) < 2:
            continue
        # Check diminishing bounces
        bounce_highs = []
        for t in range(len(touches) - 1):
            between = s.highs[touches[t]:touches[t + 1]]
            if len(between) > 0:
                bounce_highs.append(float(np.max(between)))
        if len(bounce_highs) < 2:
            continue
        if not all(bounce_highs[k] > bounce_highs[k + 1] for k in range(len(bounce_highs) - 1)):
            continue
        # Check breakdown
        if s.closes[-1] >= support:
            continue
        entry = round(support - 0.02, 2)
        stop = round(bounce_highs[-1] + 0.02, 2)
        risk = stop - entry
        if risk <= 0: continue
        target = round(entry - 2 * risk, 2)
        return _make_setup(s, "Tidal Wave", Bias.SHORT, entry, stop, target, 0.55,
                           f"Tidal Wave: {len(touches)} touches, diminishing bounces",
                           exit_strategy="1/2 at 2x, hold rest",
                           key_levels={"support": support, "last_bounce": bounce_highs[-1]})
    return None


def _detect_breaking_news(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Breaking News: Volume spike (3x+) with range expansion (2x+)."""
    if s.n < 12:
        return None
    avg_vol = float(np.mean(s.volumes[-12:-2]))
    avg_range = float(np.mean(s.highs[-12:-2] - s.lows[-12:-2]))
    cur_vol = s.volumes[-1]
    cur_range = s.highs[-1] - s.lows[-1]
    if avg_vol == 0 or cur_vol < avg_vol * 3:
        return None
    if avg_range == 0 or cur_range < avg_range * 2:
        return None
    is_bullish = s.closes[-1] > s.opens[-1]
    if is_bullish:
        entry = round(s.closes[-1], 2)
        stop = round(s.lows[-1] - 0.02, 2)
        risk = entry - stop
        if risk <= 0: return None
        target = round(entry + risk * 2, 2)
        bias = Bias.LONG
    else:
        entry = round(s.closes[-1], 2)
        stop = round(s.highs[-1] + 0.02, 2)
        risk = stop - entry
        if risk <= 0: return None
        target = round(entry - risk * 2, 2)
        bias = Bias.SHORT
    vol_x = cur_vol / avg_vol
    conf = 0.45
    if vol_x > 5: conf += 0.15
    if cur_range > avg_range * 3: conf += 0.10
    return _make_setup(s, "Breaking News", bias, entry, stop, target, conf,
                       f"Breaking News: {vol_x:.0f}x vol, {cur_range / avg_range:.1f}x range",
                       max_attempts=2, key_levels={"vol_multiple": round(vol_x, 1)})


# ==============================================================================
# QUANT PATTERNS
# ==============================================================================

def _detect_momentum_breakout(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Momentum Breakout: Price > 20-bar high. Core trend-following signal."""
    if s.n < 25:
        return None
    high_20 = float(np.max(s.highs[-21:-1]))  # Exclude current bar
    if s.closes[-1] <= high_20:
        return None
    low_20 = float(np.min(s.lows[-20:]))
    entry = round(s.closes[-1], 2)
    stop = round(low_20, 2)
    risk = entry - stop
    if risk <= 0:
        return None
    target = round(entry + (high_20 - low_20), 2)
    return _make_setup(s, "Momentum Breakout", Bias.LONG, entry, stop, target, 0.55,
                       f"20-bar high breakout at {s.closes[-1]:.2f}",
                       key_levels={"20d_high": high_20, "20d_low": low_20})


def _detect_vol_compression_breakout(s: ExtractedStructures) -> Optional[TradeSetup]:
    """Vol Compression Breakout: Bollinger squeeze → expansion."""
    if s.n < 30:
        return None
    ratio = atr_ratio(s.highs, s.lows, s.closes, atr_period=14, baseline_lookback=40)
    if ratio > 0.6:
        return None  # Not compressed enough
    # Breakout direction from last bar
    sma20 = float(np.mean(s.closes[-20:]))
    if s.closes[-1] > sma20:
        entry = round(s.closes[-1], 2)
        stop = round(sma20 - s.current_atr, 2)
        risk = entry - stop
        if risk <= 0: return None
        target = round(entry + 2 * risk, 2)
        bias = Bias.LONG
        desc = "Vol squeeze breakout above 20 SMA"
    else:
        entry = round(s.closes[-1], 2)
        stop = round(sma20 + s.current_atr, 2)
        risk = stop - entry
        if risk <= 0: return None
        target = round(entry - 2 * risk, 2)
        bias = Bias.SHORT
        desc = "Vol squeeze breakdown below 20 SMA"
    return _make_setup(s, "Vol Compression Breakout", bias, entry, stop, target, 0.58,
                       f"{desc} (ATR ratio {ratio:.2f})",
                       key_levels={"sma20": round(sma20, 2), "atr_ratio": round(ratio, 2)})