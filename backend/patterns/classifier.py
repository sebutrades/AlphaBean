"""
patterns/classifier.py — Structure-first pattern classification (47 detectors).

Pipeline: BarSeries → extract_structures() → classify_all() → list[TradeSetup]

Categories:
  Classical Structural (16): H&S, Inv H&S, Double/Triple Top/Bottom,
    Asc/Desc/Sym Triangle, Bull/Bear Flag, Pennant, Cup&Handle,
    Rectangle, Rising/Falling Wedge
  Candlestick (10): Bullish/Bearish Engulfing, Morning/Evening Star,
    Hammer, Shooting Star, Doji, Dragonfly Doji, 3 White Soldiers, 3 Black Crows
  SMB Scalps (11): RubberBand, HitchHiker, ORB 15/30, Second Chance,
    BackSide, Fashionably Late, Spencer, Gap G&G, Tidal Wave, Breaking News
  Quant Strategies (10): Momentum Breakout, Vol Compression, Mean Reversion,
    Trend Pullback, Gap Fade, Relative Strength, Range Expansion,
    Volume Breakout, VWAP Reversion, Donchian Breakout

Candlestick math follows Steve Nison's conventions:
  body = |close - open|
  upper_shadow = high - max(open, close)
  lower_shadow = min(open, close) - low
  range = high - low
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


def extract_structures(bars: BarSeries) -> ExtractedStructures:
    return ExtractedStructures(bars)


# ==============================================================================
# HELPERS
# ==============================================================================

def _make(s, name, bias, entry, stop, target, conf, desc, **kw):
    risk = abs(entry - stop)
    if risk <= 0: return None
    rr = round(abs(target - entry) / risk, 2)
    meta = PATTERN_META.get(name, {})
    return TradeSetup(
        pattern_name=name, category=meta.get("cat", PatternCategory.CLASSICAL),
        symbol=s.symbol, bias=bias,
        entry_price=round(entry, 2), stop_loss=round(stop, 2),
        target_price=round(target, 2), risk_reward_ratio=rr,
        confidence=round(min(0.95, conf), 2), detected_at=s.timestamps[-1],
        description=desc, strategy_type=meta.get("type", "breakout"),
        win_rate=meta.get("wr", 0.5), timeframe_detected=s.timeframe, **kw)

def _in_time(ts, sh, sm, eh, em):
    return time(sh, sm) <= ts.time() <= time(eh, em)

def _body(bar): return abs(bar.close - bar.open)
def _upper_shadow(bar): return bar.high - max(bar.open, bar.close)
def _lower_shadow(bar): return min(bar.open, bar.close) - bar.low
def _range(bar): return bar.high - bar.low
def _is_green(bar): return bar.close > bar.open
def _is_red(bar): return bar.close < bar.open


# ==============================================================================
# CLASSICAL STRUCTURAL PATTERNS (16)
# ==============================================================================

def _detect_head_and_shoulders(s):
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 2: return None
    for i in range(len(s.zz_highs) - 2):
        ls, hd, rs = s.zz_highs[i], s.zz_highs[i+1], s.zz_highs[i+2]
        if not (hd.price > ls.price and hd.price > rs.price): continue
        if abs(ls.price - rs.price) / ls.price > 0.03: continue
        lows_between = [l for l in s.zz_lows if ls.index < l.index < rs.index]
        if not lows_between: continue
        neckline = min(l.price for l in lows_between)
        if s.closes[-1] >= neckline: continue
        conf = 0.60 + (1 - abs(ls.price - rs.price)/ls.price) * 0.15
        return _make(s, "Head & Shoulders", Bias.SHORT,
                     neckline - 0.02, rs.price + 0.02, neckline - 0.02 - (hd.price - neckline), conf,
                     f"H&S: head@{hd.price:.2f}, neckline@{neckline:.2f}",
                     key_levels={"left_shoulder": ls.price, "head": hd.price,
                                 "right_shoulder": rs.price, "neckline": neckline})
    return None

def _detect_inverse_hs(s):
    if len(s.zz_lows) < 3 or len(s.zz_highs) < 2: return None
    for i in range(len(s.zz_lows) - 2):
        ls, hd, rs = s.zz_lows[i], s.zz_lows[i+1], s.zz_lows[i+2]
        if not (hd.price < ls.price and hd.price < rs.price): continue
        if abs(ls.price - rs.price) / ls.price > 0.03: continue
        highs_between = [h for h in s.zz_highs if ls.index < h.index < rs.index]
        if not highs_between: continue
        neckline = max(h.price for h in highs_between)
        if s.closes[-1] <= neckline: continue
        conf = 0.60 + (1 - abs(ls.price - rs.price)/ls.price) * 0.15
        return _make(s, "Inverse H&S", Bias.LONG,
                     neckline + 0.02, rs.price - 0.02, neckline + 0.02 + (neckline - hd.price), conf,
                     f"Inv H&S: head@{hd.price:.2f}, neckline@{neckline:.2f}",
                     key_levels={"head": hd.price, "neckline": neckline})
    return None

def _detect_double_top(s):
    if len(s.zz_highs) < 2: return None
    h1, h2 = s.zz_highs[-2], s.zz_highs[-1]
    if abs(h1.price - h2.price)/h1.price > 0.015 or h2.index - h1.index < 5: return None
    valley = min(s.lows[h1.index:h2.index+1])
    if s.closes[-1] > valley: return None
    top = max(h1.price, h2.price)
    return _make(s, "Double Top", Bias.SHORT, valley-0.02, top+0.02,
                 valley-0.02-(top-valley), 0.63, f"Double Top at {top:.2f}",
                 key_levels={"top1": h1.price, "top2": h2.price, "valley": valley})

def _detect_double_bottom(s):
    if len(s.zz_lows) < 2: return None
    l1, l2 = s.zz_lows[-2], s.zz_lows[-1]
    if abs(l1.price - l2.price)/l1.price > 0.015 or l2.index - l1.index < 5: return None
    peak = max(s.highs[l1.index:l2.index+1])
    if s.closes[-1] < peak: return None
    bot = min(l1.price, l2.price)
    return _make(s, "Double Bottom", Bias.LONG, peak+0.02, bot-0.02,
                 peak+0.02+(peak-bot), 0.65, f"Double Bottom at {bot:.2f}",
                 key_levels={"bottom1": l1.price, "bottom2": l2.price, "peak": peak})

def _detect_triple_top(s):
    if len(s.zz_highs) < 3: return None
    h1, h2, h3 = s.zz_highs[-3], s.zz_highs[-2], s.zz_highs[-1]
    prices = [h1.price, h2.price, h3.price]
    avg = np.mean(prices)
    if max(abs(p - avg)/avg for p in prices) > 0.015: return None
    valley = min(s.lows[h1.index:h3.index+1])
    if s.closes[-1] > valley: return None
    return _make(s, "Triple Top", Bias.SHORT, valley-0.02, max(prices)+0.02,
                 valley-0.02-(max(prices)-valley), 0.66, f"Triple Top at {avg:.2f}",
                 key_levels={"resistance": avg, "valley": valley})

def _detect_triple_bottom(s):
    if len(s.zz_lows) < 3: return None
    l1, l2, l3 = s.zz_lows[-3], s.zz_lows[-2], s.zz_lows[-1]
    prices = [l1.price, l2.price, l3.price]
    avg = np.mean(prices)
    if max(abs(p - avg)/avg for p in prices) > 0.015: return None
    peak = max(s.highs[l1.index:l3.index+1])
    if s.closes[-1] < peak: return None
    return _make(s, "Triple Bottom", Bias.LONG, peak+0.02, min(prices)-0.02,
                 peak+0.02+(peak-min(prices)), 0.68, f"Triple Bottom at {avg:.2f}",
                 key_levels={"support": avg, "peak": peak})

def _detect_ascending_triangle(s):
    if len(s.zz_highs) < 2 or len(s.zz_lows) < 2: return None
    utl = fit_trendline(s.zz_highs[-3:] if len(s.zz_highs)>=3 else s.zz_highs[-2:])
    ltl = fit_trendline(s.zz_lows[-3:] if len(s.zz_lows)>=3 else s.zz_lows[-2:])
    if utl is None or ltl is None: return None
    if not is_flat_line(utl, 0.15) or ltl.slope <= 0: return None
    res = utl.price_at(utl.end_index)
    if s.closes[-1] < res: return None
    sup = s.zz_lows[-1].price
    return _make(s, "Ascending Triangle", Bias.LONG, res+0.02, sup-0.02,
                 res+0.02+(res-sup), 0.62, f"Asc Triangle: flat top {res:.2f}",
                 key_levels={"resistance": res, "support": sup})

def _detect_descending_triangle(s):
    if len(s.zz_highs) < 2 or len(s.zz_lows) < 2: return None
    ltl = fit_trendline(s.zz_lows[-3:] if len(s.zz_lows)>=3 else s.zz_lows[-2:])
    utl = fit_trendline(s.zz_highs[-3:] if len(s.zz_highs)>=3 else s.zz_highs[-2:])
    if utl is None or ltl is None: return None
    if not is_flat_line(ltl, 0.15) or utl.slope >= 0: return None
    sup = ltl.price_at(ltl.end_index)
    if s.closes[-1] > sup: return None
    res = s.zz_highs[-1].price
    return _make(s, "Descending Triangle", Bias.SHORT, sup-0.02, res+0.02,
                 sup-0.02-(res-sup), 0.60, f"Desc Triangle: flat bottom {sup:.2f}",
                 key_levels={"support": sup, "resistance": res})

def _detect_symmetrical_triangle(s):
    if len(s.zz_highs) < 2 or len(s.zz_lows) < 2: return None
    utl = fit_trendline(s.zz_highs[-3:] if len(s.zz_highs)>=3 else s.zz_highs[-2:])
    ltl = fit_trendline(s.zz_lows[-3:] if len(s.zz_lows)>=3 else s.zz_lows[-2:])
    if utl is None or ltl is None: return None
    if utl.slope >= 0 or ltl.slope <= 0: return None
    up = utl.price_at(s.n-1); lo = ltl.price_at(s.n-1)
    if up <= lo: return None
    rng = up - lo; cur = s.closes[-1]
    if cur > up:
        return _make(s, "Symmetrical Triangle", Bias.LONG, up+0.02, lo-0.02, up+0.02+rng, 0.58,
                     "Sym Triangle breakout above", key_levels={"upper": up, "lower": lo})
    elif cur < lo:
        return _make(s, "Symmetrical Triangle", Bias.SHORT, lo-0.02, up+0.02, lo-0.02-rng, 0.58,
                     "Sym Triangle breakdown below", key_levels={"upper": up, "lower": lo})
    return None

def _detect_bull_flag(s):
    if len(s.zz_lows) < 1 or len(s.zz_highs) < 1: return None
    for li in range(len(s.zz_lows)):
        lo = s.zz_lows[li]
        post = [h for h in s.zz_highs if h.index > lo.index]
        if not post: continue
        hi = post[0]
        pct = (hi.price - lo.price) / lo.price
        if pct < 0.02: continue
        fs = hi.index
        if fs >= s.n - 3: continue
        flag = s.bars[fs:]
        if len(flag) < 3 or len(flag) > 30: continue
        fl = min(b.low for b in flag)
        if (hi.price - fl) / (hi.price - lo.price) > 0.50: continue
        if s.closes[-1] <= hi.price: continue
        return _make(s, "Bull Flag", Bias.LONG, hi.price*1.001, fl*0.998,
                     hi.price*1.001+(hi.price-lo.price), 0.60,
                     f"Bull Flag: {pct:.1%} pole, {len(flag)}-bar flag",
                     key_levels={"pole_hi": hi.price, "pole_lo": lo.price, "flag_low": fl})
    return None

def _detect_bear_flag(s):
    if len(s.zz_highs) < 1 or len(s.zz_lows) < 1: return None
    for hi_idx in range(len(s.zz_highs)):
        hi = s.zz_highs[hi_idx]
        post = [l for l in s.zz_lows if l.index > hi.index]
        if not post: continue
        lo = post[0]
        pct = (hi.price - lo.price) / hi.price
        if pct < 0.02: continue
        fs = lo.index
        if fs >= s.n - 3: continue
        flag = s.bars[fs:]
        if len(flag) < 3 or len(flag) > 30: continue
        fh = max(b.high for b in flag)
        if (fh - lo.price) / (hi.price - lo.price) > 0.50: continue
        if s.closes[-1] >= lo.price: continue
        return _make(s, "Bear Flag", Bias.SHORT, lo.price*0.999, fh*1.002,
                     lo.price*0.999-(hi.price-lo.price), 0.58,
                     f"Bear Flag: {pct:.1%} pole", key_levels={"pole_hi": hi.price, "flag_high": fh})
    return None

def _detect_pennant(s):
    """Pennant: impulse + short symmetrical triangle (3-15 bars)."""
    if len(s.zz_lows) < 1 or len(s.zz_highs) < 1 or s.n < 20: return None
    # Find impulse (pole)
    for li in range(len(s.zz_lows)):
        lo = s.zz_lows[li]
        post = [h for h in s.zz_highs if h.index > lo.index]
        if not post: continue
        hi = post[0]
        pct = (hi.price - lo.price) / lo.price
        if pct < 0.03: continue
        # Pennant: 3-15 bars after pole, converging highs and lows
        fs = hi.index
        if fs >= s.n - 4: continue
        pn_bars = s.bars[fs:min(fs+15, s.n)]
        if len(pn_bars) < 3: continue
        pn_highs = [b.high for b in pn_bars]
        pn_lows = [b.low for b in pn_bars]
        # Check converging: highs falling, lows rising
        h_slope = (pn_highs[-1] - pn_highs[0]) / len(pn_highs) if len(pn_highs) > 1 else 0
        l_slope = (pn_lows[-1] - pn_lows[0]) / len(pn_lows) if len(pn_lows) > 1 else 0
        if h_slope >= 0 or l_slope <= 0: continue
        if s.closes[-1] <= max(pn_highs): continue  # Need breakout
        return _make(s, "Pennant", Bias.LONG, max(pn_highs)+0.02, min(pn_lows)-0.02,
                     max(pn_highs)+0.02+(hi.price-lo.price), 0.53,
                     f"Pennant: {pct:.1%} pole, converging consolidation",
                     key_levels={"pole_hi": hi.price})
    return None

def _detect_cup_and_handle(s):
    if len(s.zz_lows) < 1 or s.n < 40: return None
    cl = s.zz_lows[-1]
    lr = max(s.highs[:cl.index]) if cl.index > 5 else None
    rr = max(s.highs[cl.index:]) if cl.index < s.n-5 else None
    if lr is None or rr is None: return None
    rim = min(lr, rr); depth = rim - cl.price
    if depth <= 0: return None
    handle = s.bars[-min(10,s.n):]
    hl = min(b.low for b in handle)
    ret = (rim - hl) / depth
    if ret > 0.50 or ret < 0.10 or s.closes[-1] < rim: return None
    return _make(s, "Cup & Handle", Bias.LONG, rim+0.02, hl-0.02, rim+0.02+depth, 0.63,
                 f"Cup & Handle: rim {rim:.2f}", key_levels={"rim": rim, "cup_low": cl.price})

def _detect_rectangle(s):
    """Rectangle: horizontal S/R levels with breakout."""
    if s.n < 20 or len(s.sr_levels) < 2: return None
    # Find strongest resistance and support
    res_levels = [l for l in s.sr_levels if l.level_type in ("resistance", "both")]
    sup_levels = [l for l in s.sr_levels if l.level_type in ("support", "both")]
    if not res_levels or not sup_levels: return None
    res = res_levels[0].price
    sup = sup_levels[0].price
    if res <= sup: return None
    rng = res - sup
    cur = s.closes[-1]
    if cur > res + rng * 0.01:  # Breakout above
        return _make(s, "Rectangle", Bias.LONG, res+0.02, sup-0.02, res+0.02+rng, 0.58,
                     f"Rectangle breakout above {res:.2f}",
                     key_levels={"resistance": res, "support": sup})
    elif cur < sup - rng * 0.01:  # Breakdown below
        return _make(s, "Rectangle", Bias.SHORT, sup-0.02, res+0.02, sup-0.02-rng, 0.58,
                     f"Rectangle breakdown below {sup:.2f}",
                     key_levels={"resistance": res, "support": sup})
    return None

def _detect_rising_wedge(s):
    if len(s.zz_highs) < 2 or len(s.zz_lows) < 2: return None
    u = fit_trendline(s.zz_highs[-3:] if len(s.zz_highs)>=3 else s.zz_highs[-2:])
    l = fit_trendline(s.zz_lows[-3:] if len(s.zz_lows)>=3 else s.zz_lows[-2:])
    if u is None or l is None: return None
    if not (u.slope > 0 and l.slope > 0 and u.slope < l.slope): return None
    lp = l.price_at(s.n-1); up = u.price_at(s.n-1)
    if s.closes[-1] > lp: return None
    return _make(s, "Rising Wedge", Bias.SHORT, lp-0.02, up+0.02, lp-0.02-(up-lp), 0.62,
                 "Rising Wedge breakdown", key_levels={"upper": up, "lower": lp})

def _detect_falling_wedge(s):
    if len(s.zz_highs) < 2 or len(s.zz_lows) < 2: return None
    u = fit_trendline(s.zz_highs[-3:] if len(s.zz_highs)>=3 else s.zz_highs[-2:])
    l = fit_trendline(s.zz_lows[-3:] if len(s.zz_lows)>=3 else s.zz_lows[-2:])
    if u is None or l is None: return None
    if not (u.slope < 0 and l.slope < 0 and u.slope > l.slope): return None
    up = u.price_at(s.n-1); lp = l.price_at(s.n-1)
    if s.closes[-1] < up: return None
    return _make(s, "Falling Wedge", Bias.LONG, up+0.02, lp-0.02, up+0.02+(up-lp), 0.62,
                 "Falling Wedge breakout", key_levels={"upper": up, "lower": lp})


# ==============================================================================
# CANDLESTICK PATTERNS (10)
# Math: Steve Nison's Japanese Candlestick Charting Techniques (1991)
# ==============================================================================

def _detect_bullish_engulfing(s):
    """Current green body fully engulfs prior red body. Bullish reversal."""
    if s.n < 5: return None
    prev, cur = s.bars[-2], s.bars[-1]
    if not (_is_red(prev) and _is_green(cur)): return None
    if not (cur.open <= prev.close and cur.close >= prev.open): return None
    if _body(cur) <= _body(prev) * 1.1: return None  # Must be meaningfully larger
    # Confirm near recent lows (reversal context)
    recent_low = float(np.min(s.lows[-10:]))
    if prev.low > recent_low + (s.highs[-10:].max() - recent_low) * 0.3: return None
    entry = round(cur.close, 2)
    stop = round(min(prev.low, cur.low) - 0.02, 2)
    risk = entry - stop
    if risk <= 0: return None
    return _make(s, "Bullish Engulfing", Bias.LONG, entry, stop, entry + risk*2, 0.58,
                 "Bullish Engulfing: green body engulfs prior red at lows")

def _detect_bearish_engulfing(s):
    """Current red body fully engulfs prior green body. Bearish reversal."""
    if s.n < 5: return None
    prev, cur = s.bars[-2], s.bars[-1]
    if not (_is_green(prev) and _is_red(cur)): return None
    if not (cur.open >= prev.close and cur.close <= prev.open): return None
    if _body(cur) <= _body(prev) * 1.1: return None
    recent_high = float(np.max(s.highs[-10:]))
    if prev.high < recent_high - (recent_high - s.lows[-10:].min()) * 0.3: return None
    entry = round(cur.close, 2)
    stop = round(max(prev.high, cur.high) + 0.02, 2)
    risk = stop - entry
    if risk <= 0: return None
    return _make(s, "Bearish Engulfing", Bias.SHORT, entry, stop, entry - risk*2, 0.58,
                 "Bearish Engulfing: red body engulfs prior green at highs")

def _detect_morning_star(s):
    """3-bar reversal: big red → small body (star) → big green. Bullish."""
    if s.n < 5: return None
    b1, b2, b3 = s.bars[-3], s.bars[-2], s.bars[-1]
    if not _is_red(b1) or not _is_green(b3): return None
    if _body(b2) >= _body(b1) * 0.3: return None  # Star must be small
    if _body(b3) < _body(b1) * 0.5: return None   # Recovery must be strong
    # Star should gap down or be near b1's close
    if b2.high > b1.open: return None  # Star shouldn't exceed b1 open
    entry = round(b3.close, 2)
    stop = round(min(b1.low, b2.low, b3.low) - 0.02, 2)
    risk = entry - stop
    if risk <= 0: return None
    return _make(s, "Morning Star", Bias.LONG, entry, stop, entry + risk*2, 0.60,
                 "Morning Star: 3-bar bullish reversal")

def _detect_evening_star(s):
    """3-bar reversal: big green → small body → big red. Bearish."""
    if s.n < 5: return None
    b1, b2, b3 = s.bars[-3], s.bars[-2], s.bars[-1]
    if not _is_green(b1) or not _is_red(b3): return None
    if _body(b2) >= _body(b1) * 0.3: return None
    if _body(b3) < _body(b1) * 0.5: return None
    if b2.low < b1.close: return None  # Star shouldn't drop below b1 close area
    entry = round(b3.close, 2)
    stop = round(max(b1.high, b2.high, b3.high) + 0.02, 2)
    risk = stop - entry
    if risk <= 0: return None
    return _make(s, "Evening Star", Bias.SHORT, entry, stop, entry - risk*2, 0.60,
                 "Evening Star: 3-bar bearish reversal")

def _detect_hammer(s):
    """Hammer: small body at top, lower shadow >= 2x body, at lows. Bullish."""
    if s.n < 10: return None
    bar = s.bars[-1]
    body = _body(bar); ls = _lower_shadow(bar); us = _upper_shadow(bar); rng = _range(bar)
    if rng == 0 or body == 0: return None
    if ls < body * 2: return None       # Lower shadow must be >= 2x body
    if us > body * 0.5: return None     # Upper shadow must be small
    # Must be near recent lows
    recent_low = float(np.min(s.lows[-10:]))
    if bar.low > recent_low * 1.01: return None
    entry = round(bar.close, 2)
    stop = round(bar.low - 0.02, 2)
    risk = entry - stop
    if risk <= 0: return None
    return _make(s, "Hammer", Bias.LONG, entry, stop, entry + risk*2, 0.55,
                 f"Hammer at lows (shadow/body={ls/body:.1f}x)")

def _detect_shooting_star(s):
    """Shooting Star: small body at bottom, upper shadow >= 2x body, at highs."""
    if s.n < 10: return None
    bar = s.bars[-1]
    body = _body(bar); us = _upper_shadow(bar); ls = _lower_shadow(bar); rng = _range(bar)
    if rng == 0 or body == 0: return None
    if us < body * 2: return None
    if ls > body * 0.5: return None
    recent_high = float(np.max(s.highs[-10:]))
    if bar.high < recent_high * 0.99: return None
    entry = round(bar.close, 2)
    stop = round(bar.high + 0.02, 2)
    risk = stop - entry
    if risk <= 0: return None
    return _make(s, "Shooting Star", Bias.SHORT, entry, stop, entry - risk*2, 0.55,
                 f"Shooting Star at highs (shadow/body={us/body:.1f}x)")

def _detect_doji(s):
    """Doji: body < 10% of range. Indecision / potential reversal."""
    if s.n < 10: return None
    bar = s.bars[-1]
    body = _body(bar); rng = _range(bar)
    if rng == 0: return None
    if body / rng > 0.10: return None  # Body must be < 10% of range
    # Direction based on context: near highs = bearish doji, near lows = bullish
    recent_high = float(np.max(s.highs[-10:]))
    recent_low = float(np.min(s.lows[-10:]))
    mid = (recent_high + recent_low) / 2
    if bar.close > mid:  # Near highs → bearish
        stop = round(bar.high + 0.02, 2); entry = round(bar.close, 2)
        risk = stop - entry
        if risk <= 0: return None
        return _make(s, "Doji", Bias.SHORT, entry, stop, entry - risk*1.5, 0.48,
                     "Doji at highs: potential bearish reversal")
    else:  # Near lows → bullish
        stop = round(bar.low - 0.02, 2); entry = round(bar.close, 2)
        risk = entry - stop
        if risk <= 0: return None
        return _make(s, "Doji", Bias.LONG, entry, stop, entry + risk*1.5, 0.48,
                     "Doji at lows: potential bullish reversal")

def _detect_dragonfly_doji(s):
    """Dragonfly Doji: open ≈ close ≈ high, long lower shadow. Bullish."""
    if s.n < 10: return None
    bar = s.bars[-1]
    body = _body(bar); rng = _range(bar); ls = _lower_shadow(bar); us = _upper_shadow(bar)
    if rng == 0: return None
    if body / rng > 0.10: return None       # Doji body
    if us > rng * 0.10: return None         # Open/close near high
    if ls < rng * 0.65: return None         # Long lower shadow
    entry = round(bar.close, 2)
    stop = round(bar.low - 0.02, 2)
    risk = entry - stop
    if risk <= 0: return None
    return _make(s, "Dragonfly Doji", Bias.LONG, entry, stop, entry + risk*2, 0.52,
                 "Dragonfly Doji: rejection of lower prices")

def _detect_three_white_soldiers(s):
    """Three White Soldiers: 3 consecutive green bars, each closing higher."""
    if s.n < 5: return None
    b1, b2, b3 = s.bars[-3], s.bars[-2], s.bars[-1]
    if not (_is_green(b1) and _is_green(b2) and _is_green(b3)): return None
    if not (b2.close > b1.close and b3.close > b2.close): return None
    # Each opens within prior body
    if b2.open < b1.open or b2.open > b1.close: return None
    if b3.open < b2.open or b3.open > b2.close: return None
    # Minimal upper shadows (strong closes)
    for b in [b1, b2, b3]:
        if _upper_shadow(b) > _body(b) * 0.5: return None
    entry = round(b3.close, 2)
    stop = round(b1.low - 0.02, 2)
    risk = entry - stop
    if risk <= 0: return None
    return _make(s, "Three White Soldiers", Bias.LONG, entry, stop, entry + risk*1.5, 0.58,
                 "Three White Soldiers: strong bullish momentum")

def _detect_three_black_crows(s):
    """Three Black Crows: 3 consecutive red bars, each closing lower."""
    if s.n < 5: return None
    b1, b2, b3 = s.bars[-3], s.bars[-2], s.bars[-1]
    if not (_is_red(b1) and _is_red(b2) and _is_red(b3)): return None
    if not (b2.close < b1.close and b3.close < b2.close): return None
    if b2.open > b1.open or b2.open < b1.close: return None
    if b3.open > b2.open or b3.open < b2.close: return None
    for b in [b1, b2, b3]:
        if _lower_shadow(b) > _body(b) * 0.5: return None
    entry = round(b3.close, 2)
    stop = round(b1.high + 0.02, 2)
    risk = stop - entry
    if risk <= 0: return None
    return _make(s, "Three Black Crows", Bias.SHORT, entry, stop, entry - risk*1.5, 0.58,
                 "Three Black Crows: strong bearish momentum")


# ==============================================================================
# SMB SCALP PATTERNS (11) — time-of-day logic preserved
# ==============================================================================

def _detect_rubberband(s):
    if s.n < 20 or s.current_atr <= 0: return None
    oi = s.day_open_idx
    if oi >= s.n - 5: return None
    op = s.bars[oi].open; db = s.bars[oi:]
    dl = min(b.low for b in db)
    dli = oi + min(range(len(db)), key=lambda i: db[i].low)
    atrs = (op - dl) / s.current_atr
    if atrs < 1.0 or dli - oi < 5: return None
    mid = (oi + dli) // 2
    r1 = [s.bars[i].high - s.bars[i].low for i in range(oi, mid)] if mid > oi else [0]
    r2 = [s.bars[i].high - s.bars[i].low for i in range(mid, dli)] if dli > mid else [0]
    if np.mean(r2) < np.mean(r1) * 1.2: return None
    for i in range(dli+1, min(dli+10, s.n)):
        if s.bars[i].close > s.bars[i].open:
            if sum(1 for j in range(max(dli,i-5),i) if s.bars[i].high > s.bars[j].high) >= 2:
                e = round(s.bars[i].high+0.02,2); st = round(dl-0.02,2)
                r = e-st
                if r <= 0: continue
                c = 0.45 + (0.15 if atrs>=3 else 0) + (0.10 if _in_time(s.bars[i].timestamp,10,0,10,45) else 0)
                return _make(s,"RubberBand Scalp",Bias.LONG,e,st,round(e+2*r,2),c,
                             f"RubberBand: {atrs:.1f} ATR ext",max_attempts=2,
                             exit_strategy="1/3 at 1:1, 1/3 at 2:1, 1/3 VWAP",
                             ideal_time="10:00-10:45 AM ET",key_levels={"day_low":dl})
    return None

def _detect_hitchhiker(s):
    if s.n < 15: return None
    oi = s.day_open_idx
    if s.n - oi < 10: return None
    de = None
    for i in range(oi+3, min(oi+10,s.n)):
        if s.bars[i].close > s.bars[oi].open*1.005: de = i; break
    if de is None: return None
    dh = max(b.high for b in s.bars[oi:]); dl = min(b.low for b in s.bars[oi:])
    dr = dh - dl
    if dr <= 0: return None
    ut = dl + dr * (2/3)
    for cs in range(de+1, min(de+16,s.n-5)):
        ce = min(cs+8, s.n-1)
        cb = s.bars[cs:ce]; ch = max(b.high for b in cb); cl = min(b.low for b in cb)
        if cl < ut or (ch-cl) > dr*0.40: continue
        if ce+1 >= s.n or s.bars[ce+1].high <= ch: continue
        e = round(ch+0.02,2); st = round(cl-0.02,2); r = e-st
        if r <= 0: continue
        c = 0.50 + (0.10 if _in_time(s.bars[ce+1].timestamp,9,30,9,59) else 0)
        return _make(s,"HitchHiker Scalp",Bias.LONG,e,st,round(e+r*1.9,2),c,
                     "HitchHiker: consol upper 1/3",max_attempts=1,ideal_time="9:30-9:59 AM ET")
    return None

def _detect_orb(s, minutes=15):
    if s.n < 10: return None
    oi = s.day_open_idx
    if oi >= s.n-5: return None
    ot = s.timestamps[oi]; cut = ot + timedelta(minutes=minutes)
    oe = oi
    for i in range(oi, s.n):
        if s.timestamps[i] <= cut: oe = i
        else: break
    if oe <= oi: return None
    ob = s.bars[oi:oe+1]; oh = max(b.high for b in ob); ol = min(b.low for b in ob); orng = oh-ol
    if orng <= 0: return None
    nm = f"ORB {minutes}min"
    for i in range(oe+1, min(oe+30,s.n)):
        if s.bars[i].close > oh:
            e=round(oh+0.02,2); st=round(s.bars[i].low-0.02,2); r=e-st
            if r<=0 or r>orng: continue
            c = 0.50 + (0.15 if s.bars[i].volume > np.mean([b.volume for b in ob])*1.5 else 0)
            return _make(s,nm,Bias.LONG,e,st,round(e+2*orng,2),c,
                         f"ORB {minutes}min above {oh:.2f}",max_attempts=2,
                         key_levels={"orb_high":oh,"orb_low":ol})
        if s.bars[i].close < ol:
            e=round(ol-0.02,2); st=round(s.bars[i].high+0.02,2); r=st-e
            if r<=0 or r>orng: continue
            return _make(s,nm,Bias.SHORT,e,st,round(e-2*orng,2),0.55,
                         f"ORB {minutes}min below {ol:.2f}",max_attempts=2,
                         key_levels={"orb_high":oh,"orb_low":ol})
    return None

def _detect_second_chance(s):
    if s.n < 30 or len(s.sw_high_idx) < 2: return None
    for i in range(len(s.sw_high_idx)-1):
        lev = s.highs[s.sw_high_idx[i]]; tol = lev*0.003
        touches = [s.sw_high_idx[i]]
        for j in range(i+1, len(s.sw_high_idx)):
            if abs(s.highs[s.sw_high_idx[j]]-lev) < tol: touches.append(s.sw_high_idx[j])
        if len(touches) < 2: continue
        res = float(np.mean([s.highs[idx] for idx in touches]))
        last = touches[-1]
        for k in range(last+1, min(last+20,s.n)):
            if s.closes[k] > res*1.002:
                for p in range(k+1, min(k+15,s.n)):
                    if s.lows[p] <= res*1.003 and p+1 < s.n and s.closes[p+1] > s.bars[p].high:
                        e=round(s.closes[p+1],2); st=round(s.bars[p].low-0.02,2); r=e-st
                        if r<=0: continue
                        t=round(max(s.highs[k:p+1]),2)
                        if (t-e)/r < 1: continue
                        c = 0.45 + (0.15 if s.bars[k].volume > s.bars[p].volume*1.3 else 0)
                        return _make(s,"Second Chance Scalp",Bias.LONG,e,st,t,c,
                                     f"Second Chance: retest {res:.2f}",max_attempts=2)
                break
    return None

def _detect_backside(s):
    if s.n < 25: return None
    oi = s.day_open_idx
    if s.n - oi < 15: return None
    cv, ctv = 0.0, 0.0
    for i in range(oi,s.n):
        tp = (s.highs[i]+s.lows[i]+s.closes[i])/3; cv += s.volumes[i]; ctv += tp*s.volumes[i]
    vwap = ctv/cv if cv > 0 else s.closes[-1]
    e9 = ema_last(s.closes, 9)
    if s.closes[-1] >= vwap or s.closes[-1] < e9: return None
    if s.n >= 12 and ema_last(s.closes,9) <= ema_last(s.closes[:-3],9): return None
    e=round(s.highs[-1]+0.02,2); rl = s.lows[-8:]
    hl = float(np.sort(rl)[-2]) if len(rl)>=2 else s.lows[-1]
    st=round(hl-0.02,2); t=round(vwap,2); r=e-st
    if r<=0 or t<=e: return None
    c = 0.48 + (0.10 if _in_time(s.timestamps[-1],10,0,10,45) else 0)
    return _make(s,"BackSide Scalp",Bias.LONG,e,st,t,c,f"BackSide: VWAP {vwap:.2f}",
                 ideal_time="10:00-10:45 ET",key_levels={"vwap":round(vwap,2)})

def _detect_fashionably_late(s):
    if s.n < 30: return None
    oi = s.day_open_idx
    if s.n - oi < 20: return None
    e9s = ema(s.closes, 9); cv, ctv = 0.0, 0.0
    vs = np.zeros(s.n)
    for i in range(oi,s.n):
        tp=(s.highs[i]+s.lows[i]+s.closes[i])/3; cv+=s.volumes[i]; ctv+=tp*s.volumes[i]
        vs[i] = ctv/cv if cv > 0 else tp
    dl = float(np.min(s.lows[oi:]))
    for i in range(oi+10, s.n):
        if np.isnan(e9s[i]) or np.isnan(e9s[i-1]) or vs[i]==0 or vs[i-1]==0: continue
        if e9s[i-1] < vs[i-1] and e9s[i] >= vs[i]:
            cp = vs[i]; mm = cp - dl
            if mm <= 0: continue
            c = 0.55 + (0.10 if _in_time(s.timestamps[i],10,0,10,45) else 0)
            return _make(s,"Fashionably Late",Bias.LONG,round(cp,2),round(cp-mm/3,2),
                         round(cp+mm,2),c,f"Fashionably Late: measured ${mm:.2f}",
                         ideal_time="10:00-10:45 ET",key_levels={"cross":round(cp,2)})
    return None

def _detect_spencer(s):
    if s.n < 30: return None
    oi = s.day_open_idx
    if s.n-oi < 25: return None
    dh=max(b.high for b in s.bars[oi:]); dl=min(b.low for b in s.bars[oi:]); dr=dh-dl
    if dr<=0: return None
    ut = dl + dr*(2/3)
    for st in range(oi+5, s.n-10):
        w=s.bars[st:st+10]; ch=max(b.high for b in w); cl=min(b.low for b in w)
        if cl<ut or (ch-cl)>dr*0.35: continue
        bi=st+10
        if bi>=s.n or s.bars[bi].high<=ch: continue
        e=round(ch+0.02,2); stp=round(cl-0.02,2); r=e-stp
        if r<=0: continue
        return _make(s,"Spencer Scalp",Bias.LONG,e,stp,round(e+2*(ch-cl),2),0.52,
                     "Spencer: tight consol near HOD",key_levels={"consol_high":ch})
    return None

def _detect_gap_give_and_go(s):
    if s.n < 15: return None
    oi = s.day_open_idx
    if oi==0 or s.n-oi < 10: return None
    pc = s.bars[oi-1].close; op = s.bars[oi].open
    gp = (op-pc)/pc*100
    if gp < 0.5: return None
    de = None
    for i in range(oi+1, min(oi+6,s.n)):
        if s.bars[i].low < op*0.995: de=i
    if de is None: return None
    for cl in range(3,8):
        ce = de+cl
        if ce >= s.n-1: continue
        cb = s.bars[de:ce]; ch=max(b.high for b in cb); clo=min(b.low for b in cb)
        if (ch-clo) > (op-clo)*0.50: continue
        if s.bars[ce].high > ch:
            e=round(ch+0.02,2); st=round(clo-0.02,2); r=e-st
            if r<=0: continue
            t=round(op,2)
            if (t-e)/r < 1: continue
            c = 0.50 + (0.10 if _in_time(s.timestamps[ce],9,30,9,45) else 0)
            return _make(s,"Gap Give & Go",Bias.LONG,e,st,t,c,
                         f"Gap G&G: {gp:.1f}%",max_attempts=2,ideal_time="9:30-9:45 AM ET")
    return None

def _detect_tidal_wave(s):
    if s.n < 30 or len(s.sw_low_idx) < 2: return None
    for i in range(len(s.sw_low_idx)):
        sup = s.lows[s.sw_low_idx[i]]; tol = sup*0.003
        touches = [s.sw_low_idx[i]]
        for j in range(i+1, len(s.sw_low_idx)):
            if abs(s.lows[s.sw_low_idx[j]]-sup) < tol: touches.append(s.sw_low_idx[j])
        if len(touches) < 2: continue
        bh = []
        for t in range(len(touches)-1):
            btw = s.highs[touches[t]:touches[t+1]]
            if len(btw)>0: bh.append(float(np.max(btw)))
        if len(bh)<2 or not all(bh[k]>bh[k+1] for k in range(len(bh)-1)): continue
        if s.closes[-1] >= sup: continue
        e=round(sup-0.02,2); st=round(bh[-1]+0.02,2); r=st-e
        if r<=0: continue
        return _make(s,"Tidal Wave",Bias.SHORT,e,st,round(e-2*r,2),0.55,
                     f"Tidal Wave: {len(touches)} touches, diminishing bounces",
                     key_levels={"support":sup})
    return None

def _detect_breaking_news(s):
    if s.n < 12: return None
    av = float(np.mean(s.volumes[-12:-2])); ar = float(np.mean(s.highs[-12:-2]-s.lows[-12:-2]))
    cv = s.volumes[-1]; cr = s.highs[-1]-s.lows[-1]
    if av==0 or cv<av*3 or ar==0 or cr<ar*2: return None
    vx = cv/av
    if _is_green(s.bars[-1]):
        e=round(s.closes[-1],2); st=round(s.lows[-1]-0.02,2); r=e-st
        if r<=0: return None
        c = 0.45+(0.15 if vx>5 else 0)+(0.10 if cr>ar*3 else 0)
        return _make(s,"Breaking News",Bias.LONG,e,st,round(e+r*2,2),c,
                     f"Breaking News: {vx:.0f}x vol",max_attempts=2)
    else:
        e=round(s.closes[-1],2); st=round(s.highs[-1]+0.02,2); r=st-e
        if r<=0: return None
        c = 0.45+(0.15 if vx>5 else 0)+(0.10 if cr>ar*3 else 0)
        return _make(s,"Breaking News",Bias.SHORT,e,st,round(e-r*2,2),c,
                     f"Breaking News: {vx:.0f}x vol",max_attempts=2)


# ==============================================================================
# QUANT STRATEGY PATTERNS (10)
# ==============================================================================

def _detect_momentum_breakout(s):
    """Price > 20-bar high. Core trend-following signal."""
    if s.n < 25: return None
    h20 = float(np.max(s.highs[-21:-1])); l20 = float(np.min(s.lows[-20:]))
    if s.closes[-1] <= h20: return None
    e=round(s.closes[-1],2); st=round(l20,2); r=e-st
    if r<=0: return None
    return _make(s,"Momentum Breakout",Bias.LONG,e,st,round(e+(h20-l20),2),0.55,
                 f"20-bar high breakout",key_levels={"20d_high":h20,"20d_low":l20})

def _detect_vol_compression_breakout(s):
    """Bollinger squeeze → expansion."""
    if s.n < 30: return None
    r = atr_ratio(s.highs, s.lows, s.closes, atr_period=14, baseline_lookback=40)
    if r > 0.6: return None
    sma20 = float(np.mean(s.closes[-20:]))
    if s.closes[-1] > sma20:
        e=round(s.closes[-1],2); st=round(sma20-s.current_atr,2); risk=e-st
        if risk<=0: return None
        return _make(s,"Vol Compression Breakout",Bias.LONG,e,st,round(e+2*risk,2),0.58,
                     f"Vol squeeze breakout (ATR ratio {r:.2f})",key_levels={"sma20":round(sma20,2)})
    else:
        e=round(s.closes[-1],2); st=round(sma20+s.current_atr,2); risk=st-e
        if risk<=0: return None
        return _make(s,"Vol Compression Breakout",Bias.SHORT,e,st,round(e-2*risk,2),0.58,
                     f"Vol squeeze breakdown (ATR ratio {r:.2f})")

def _detect_mean_reversion(s):
    """Z-score extreme: buy at z < -2, sell at z > 2."""
    if s.n < 25: return None
    ma = float(np.mean(s.closes[-20:])); std = float(np.std(s.closes[-20:]))
    if std == 0: return None
    z = (s.closes[-1] - ma) / std
    if abs(z) < 2.0: return None
    if z < -2:
        e=round(s.closes[-1],2); st=round(s.closes[-1]-s.current_atr*1.5,2); r=e-st
        if r<=0: return None
        return _make(s,"Mean Reversion",Bias.LONG,e,st,round(ma,2),0.55,
                     f"Mean Reversion: z={z:.2f}, target MA {ma:.2f}")
    else:
        e=round(s.closes[-1],2); st=round(s.closes[-1]+s.current_atr*1.5,2); r=st-e
        if r<=0: return None
        return _make(s,"Mean Reversion",Bias.SHORT,e,st,round(ma,2),0.55,
                     f"Mean Reversion: z={z:.2f}, target MA {ma:.2f}")

def _detect_trend_pullback(s):
    """Above 50 SMA + price pulled back near 21 EMA. Trend continuation."""
    if s.n < 55: return None
    sma50 = float(np.mean(s.closes[-50:])); e21 = ema_last(s.closes, 21)
    cur = s.closes[-1]
    if cur < sma50: return None  # Must be in uptrend
    # Price should be near 21 EMA (within 1 ATR)
    if s.current_atr == 0: return None
    dist = abs(cur - e21) / s.current_atr
    if dist > 1.5 or cur > e21 * 1.005: return None  # Must be pulling back TO ema, not above
    # Check recent bar is bouncing (green after touching EMA)
    if not _is_green(s.bars[-1]): return None
    e=round(cur,2); st=round(e21 - s.current_atr,2); r=e-st
    if r<=0: return None
    return _make(s,"Trend Pullback",Bias.LONG,e,st,round(e+2*r,2),0.58,
                 f"Trend Pullback: above 50 SMA, bounce off 21 EMA",
                 key_levels={"sma50":round(sma50,2),"ema21":round(e21,2)})

def _detect_gap_fade(s):
    """Large gap (>2%) that's likely to retrace. Fade direction."""
    if s.n < 5: return None
    oi = s.day_open_idx
    if oi == 0: return None
    pc = s.bars[oi-1].close; op = s.bars[oi].open
    gap_pct = (op - pc) / pc * 100
    if abs(gap_pct) < 2.0: return None
    if gap_pct > 2.0:  # Gap up → short fade
        e=round(s.closes[-1],2); st=round(s.highs[oi]+0.02,2); r=st-e
        if r<=0: return None
        t=round(pc,2)  # Target: fill gap
        if (e-t)/r < 0.5: return None
        return _make(s,"Gap Fade",Bias.SHORT,e,st,t,0.52,
                     f"Gap Fade: {gap_pct:.1f}% gap up, target fill",
                     key_levels={"prev_close":pc,"gap_open":op})
    else:  # Gap down → long fade
        e=round(s.closes[-1],2); st=round(s.lows[oi]-0.02,2); r=e-st
        if r<=0: return None
        t=round(pc,2)
        if (t-e)/r < 0.5: return None
        return _make(s,"Gap Fade",Bias.LONG,e,st,t,0.52,
                     f"Gap Fade: {gap_pct:.1f}% gap down, target fill",
                     key_levels={"prev_close":pc,"gap_open":op})

def _detect_relative_strength_break(s):
    """Stock making 50-bar high while within 5% of 20-bar high. Strength signal."""
    if s.n < 55: return None
    h50 = float(np.max(s.highs[-51:-1])); h20 = float(np.max(s.highs[-21:-1]))
    cur = s.closes[-1]
    if cur <= h50: return None
    # Additional: volume confirmation
    avg_vol = float(np.mean(s.volumes[-20:]))
    if s.volumes[-1] < avg_vol * 1.2: return None
    l20 = float(np.min(s.lows[-20:]))
    e=round(cur,2); st=round(l20,2); r=e-st
    if r<=0: return None
    return _make(s,"Relative Strength Break",Bias.LONG,e,st,round(e+r*1.5,2),0.55,
                 f"50-bar high breakout with volume",key_levels={"50d_high":h50})

def _detect_range_expansion(s):
    """Today's range > 2x average range. Directional move starting."""
    if s.n < 15: return None
    avg_r = float(np.mean(s.highs[-15:-1] - s.lows[-15:-1]))
    if avg_r == 0: return None
    cur_r = s.highs[-1] - s.lows[-1]
    if cur_r < avg_r * 2: return None
    rx = cur_r / avg_r
    if _is_green(s.bars[-1]):
        e=round(s.closes[-1],2); st=round(s.lows[-1]-0.02,2); r=e-st
        if r<=0: return None
        return _make(s,"Range Expansion",Bias.LONG,e,st,round(e+r*1.5,2),0.52,
                     f"Range Expansion: {rx:.1f}x avg range (bullish)")
    else:
        e=round(s.closes[-1],2); st=round(s.highs[-1]+0.02,2); r=st-e
        if r<=0: return None
        return _make(s,"Range Expansion",Bias.SHORT,e,st,round(e-r*1.5,2),0.52,
                     f"Range Expansion: {rx:.1f}x avg range (bearish)")

def _detect_volume_breakout(s):
    """Volume > 3x average + directional close beyond recent range."""
    if s.n < 20: return None
    avg_v = float(np.mean(s.volumes[-20:-1]))
    if avg_v == 0 or s.volumes[-1] < avg_v * 3: return None
    h10 = float(np.max(s.highs[-11:-1])); l10 = float(np.min(s.lows[-11:-1]))
    cur = s.closes[-1]; vx = s.volumes[-1] / avg_v
    if cur > h10:
        e=round(cur,2); st=round(l10,2); r=e-st
        if r<=0: return None
        return _make(s,"Volume Breakout",Bias.LONG,e,st,round(e+r,2),0.55,
                     f"Volume Breakout: {vx:.0f}x vol, above 10-bar high")
    elif cur < l10:
        e=round(cur,2); st=round(h10,2); r=st-e
        if r<=0: return None
        return _make(s,"Volume Breakout",Bias.SHORT,e,st,round(e-r,2),0.55,
                     f"Volume Breakout: {vx:.0f}x vol, below 10-bar low")
    return None

def _detect_vwap_reversion(s):
    """Price far from intraday VWAP → fade back toward VWAP."""
    if s.n < 20: return None
    oi = s.day_open_idx
    if s.n - oi < 10: return None
    cv, ctv = 0.0, 0.0
    for i in range(oi, s.n):
        tp = (s.highs[i]+s.lows[i]+s.closes[i])/3; cv += s.volumes[i]; ctv += tp*s.volumes[i]
    vwap = ctv/cv if cv > 0 else s.closes[-1]
    if s.current_atr == 0: return None
    dist = (s.closes[-1] - vwap) / s.current_atr
    if abs(dist) < 1.5: return None  # Must be >1.5 ATR from VWAP
    if dist > 1.5:  # Too far above → short back to VWAP
        e=round(s.closes[-1],2); st=round(s.closes[-1]+s.current_atr,2); r=st-e
        if r<=0: return None
        return _make(s,"VWAP Reversion",Bias.SHORT,e,st,round(vwap,2),0.55,
                     f"VWAP Reversion: {dist:.1f} ATR above VWAP",key_levels={"vwap":round(vwap,2)})
    else:  # Too far below → long back to VWAP
        e=round(s.closes[-1],2); st=round(s.closes[-1]-s.current_atr,2); r=e-st
        if r<=0: return None
        return _make(s,"VWAP Reversion",Bias.LONG,e,st,round(vwap,2),0.55,
                     f"VWAP Reversion: {abs(dist):.1f} ATR below VWAP",key_levels={"vwap":round(vwap,2)})

def _detect_donchian_breakout(s):
    """Donchian Channel: close > highest high of last 50 bars. Trend-following."""
    if s.n < 55: return None
    h50 = float(np.max(s.highs[-51:-1])); l50 = float(np.min(s.lows[-51:-1]))
    cur = s.closes[-1]
    if cur > h50:
        mid = (h50+l50)/2; e=round(cur,2); st=round(mid,2); r=e-st
        if r<=0: return None
        return _make(s,"Donchian Breakout",Bias.LONG,e,st,round(e+r,2),0.53,
                     f"Donchian 50-bar high breakout",key_levels={"50_high":h50,"50_low":l50})
    elif cur < l50:
        mid = (h50+l50)/2; e=round(cur,2); st=round(mid,2); r=st-e
        if r<=0: return None
        return _make(s,"Donchian Breakout",Bias.SHORT,e,st,round(e-r,2),0.53,
                     f"Donchian 50-bar low breakdown",key_levels={"50_high":h50,"50_low":l50})
    return None


# ==============================================================================
# MAIN CLASSIFIER
# ==============================================================================

_ALL_DETECTORS = [
    # Classical (16)
    _detect_head_and_shoulders, _detect_inverse_hs,
    _detect_double_top, _detect_double_bottom,
    _detect_triple_top, _detect_triple_bottom,
    _detect_ascending_triangle, _detect_descending_triangle, _detect_symmetrical_triangle,
    _detect_bull_flag, _detect_bear_flag, _detect_pennant,
    _detect_cup_and_handle, _detect_rectangle,
    _detect_rising_wedge, _detect_falling_wedge,
    # Candlestick (10)
    _detect_bullish_engulfing, _detect_bearish_engulfing,
    _detect_morning_star, _detect_evening_star,
    _detect_hammer, _detect_shooting_star,
    _detect_doji, _detect_dragonfly_doji,
    _detect_three_white_soldiers, _detect_three_black_crows,
    # SMB Scalps (11)
    _detect_rubberband, _detect_hitchhiker,
    lambda s: _detect_orb(s, 15), lambda s: _detect_orb(s, 30),
    _detect_second_chance, _detect_backside,
    _detect_fashionably_late, _detect_spencer,
    _detect_gap_give_and_go, _detect_tidal_wave, _detect_breaking_news,
    # Quant (10)
    _detect_momentum_breakout, _detect_vol_compression_breakout,
    _detect_mean_reversion, _detect_trend_pullback, _detect_gap_fade,
    _detect_relative_strength_break, _detect_range_expansion,
    _detect_volume_breakout, _detect_vwap_reversion, _detect_donchian_breakout,
]


def classify_all(bars: BarSeries) -> list[TradeSetup]:
    """Run ALL 47 pattern detectors. Returns setups sorted by confidence."""
    if len(bars.bars) < 15:
        return []
    s = extract_structures(bars)
    setups = []
    for fn in _ALL_DETECTORS:
        try:
            result = fn(s)
            if result is not None:
                setups.append(result)
        except Exception:
            continue
    setups.sort(key=lambda x: x.confidence, reverse=True)
    return setups