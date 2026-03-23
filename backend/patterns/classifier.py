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

def _make(s, name, bias, entry, stop, target, conf, desc, **kw):
    """Create TradeSetup with entry validation.
    Rejects setups where entry has already been exceeded (retroactive entry)."""
    risk = abs(entry - stop)
    if risk <= 0: return None
    rr = round(abs(target - entry) / risk, 2)
    if rr < 0.5: return None

    cur = s.closes[-1]
    atr = s.current_atr if s.current_atr > 0 else abs(cur * 0.01)

    if bias == Bias.LONG:
        if cur > entry + atr * 0.5: return None
    elif bias == Bias.SHORT:
        if cur < entry - atr * 0.5: return None

    meta = PATTERN_META.get(name, {})
    return TradeSetup(
        pattern_name=name, category=meta.get("cat", PatternCategory.CLASSICAL),
        symbol=s.symbol, bias=bias,
        entry_price=round(entry, 2), stop_loss=round(stop, 2),
        target_price=round(target, 2), risk_reward_ratio=rr,
        confidence=round(min(0.95, conf), 2), detected_at=s.timestamps[-1],
        description=desc, strategy_type=meta.get("type", "breakout"),
        win_rate=meta.get("wr", 0.5), timeframe_detected=s.timeframe, **kw)


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
    """Bull Flag — FIX: require pole >= 1.5 ATR (was 2% fixed)."""
    if len(s.zz_lows) < 1 or len(s.zz_highs) < 1 or s.current_atr <= 0: return None
    for li in range(len(s.zz_lows)):
        lo = s.zz_lows[li]
        post = [h for h in s.zz_highs if h.index > lo.index]
        if not post: continue
        hi = post[0]
        pole_size = hi.price - lo.price
        if pole_size < s.current_atr * 1.5: continue  # FIX: ATR-based pole minimum
        fs = hi.index
        if fs >= s.n - 3: continue
        flag = s.bars[fs:]
        if len(flag) < 3 or len(flag) > 30: continue
        fl = min(b.low for b in flag)
        if (hi.price - fl) / pole_size > 0.50: continue
        if s.closes[-1] <= hi.price: continue
        pct = pole_size / lo.price
        return _make(s, "Bull Flag", Bias.LONG, hi.price*1.001, fl*0.998,
                     hi.price*1.001+pole_size, 0.60,
                     f"Bull Flag: {pct:.1%} pole, {len(flag)}-bar flag",
                     key_levels={"pole_hi": hi.price, "pole_lo": lo.price, "flag_low": fl})
    return None

def _detect_bear_flag(s):
    """Bear Flag — FIX: require pole >= 1.5 ATR."""
    if len(s.zz_highs) < 1 or len(s.zz_lows) < 1 or s.current_atr <= 0: return None
    for hi_idx in range(len(s.zz_highs)):
        hi = s.zz_highs[hi_idx]
        post = [l for l in s.zz_lows if l.index > hi.index]
        if not post: continue
        lo = post[0]
        pole_size = hi.price - lo.price
        if pole_size < s.current_atr * 1.5: continue  # FIX
        fs = lo.index
        if fs >= s.n - 3: continue
        flag = s.bars[fs:]
        if len(flag) < 3 or len(flag) > 30: continue
        fh = max(b.high for b in flag)
        if (fh - lo.price) / pole_size > 0.50: continue
        if s.closes[-1] >= lo.price: continue
        pct = pole_size / hi.price
        return _make(s, "Bear Flag", Bias.SHORT, lo.price*0.999, fh*1.002,
                     lo.price*0.999-pole_size, 0.58,
                     f"Bear Flag: {pct:.1%} pole", key_levels={"pole_hi": hi.price, "flag_high": fh})
    return None

def _detect_pennant(s):
    if len(s.zz_lows) < 1 or len(s.zz_highs) < 1 or s.n < 20: return None
    for li in range(len(s.zz_lows)):
        lo = s.zz_lows[li]
        post = [h for h in s.zz_highs if h.index > lo.index]
        if not post: continue
        hi = post[0]
        pct = (hi.price - lo.price) / lo.price
        if pct < 0.03: continue
        fs = hi.index
        if fs >= s.n - 4: continue
        pn_bars = s.bars[fs:min(fs+15, s.n)]
        if len(pn_bars) < 3: continue
        pn_highs = [b.high for b in pn_bars]
        pn_lows = [b.low for b in pn_bars]
        h_slope = (pn_highs[-1] - pn_highs[0]) / len(pn_highs) if len(pn_highs) > 1 else 0
        l_slope = (pn_lows[-1] - pn_lows[0]) / len(pn_lows) if len(pn_lows) > 1 else 0
        if h_slope >= 0 or l_slope <= 0: continue
        if s.closes[-1] <= max(pn_highs): continue
        return _make(s, "Pennant", Bias.LONG, max(pn_highs)+0.02, min(pn_lows)-0.02,
                     max(pn_highs)+0.02+(hi.price-lo.price), 0.53,
                     f"Pennant: {pct:.1%} pole, converging consolidation",
                     key_levels={"pole_hi": hi.price})
    return None

def _detect_cup_and_handle(s):
    """Cup & Handle — FIX: min 30-bar cup span, handle vol decline, breakout vol."""
    if len(s.zz_lows) < 1 or s.n < 40: return None
    cl = s.zz_lows[-1]
    lr = max(s.highs[:cl.index]) if cl.index > 5 else None
    rr = max(s.highs[cl.index:]) if cl.index < s.n-5 else None
    if lr is None or rr is None: return None
    rim = min(lr, rr); depth = rim - cl.price
    if depth <= 0: return None
    # FIX: Cup must span at least 30 bars
    if cl.index < 15 or s.n - cl.index < 15: return None
    handle = s.bars[-min(10, s.n):]
    hl = min(b.low for b in handle)
    ret = (rim - hl) / depth
    if ret > 0.50 or ret < 0.10 or s.closes[-1] < rim: return None
    # FIX: Volume should decline in handle
    handle_vol = float(np.mean([b.volume for b in handle]))
    cup_vol = float(np.mean(s.volumes[max(0, cl.index-10):cl.index+10]))
    if cup_vol > 0 and handle_vol > cup_vol * 1.2: return None  # Handle vol should be lower
    # FIX: Breakout bar needs volume
    avg_vol = float(np.mean(s.volumes[-20:]))
    if avg_vol > 0 and s.volumes[-1] < avg_vol * 1.3: return None
    return _make(s, "Cup & Handle", Bias.LONG, rim+0.02, hl-0.02, rim+0.02+depth, 0.63,
                 f"Cup & Handle: rim {rim:.2f}", key_levels={"rim": rim, "cup_low": cl.price})

def _detect_rectangle(s):
    if s.n < 20 or len(s.sr_levels) < 2: return None
    res_levels = [l for l in s.sr_levels if l.level_type in ("resistance", "both")]
    sup_levels = [l for l in s.sr_levels if l.level_type in ("support", "both")]
    if not res_levels or not sup_levels: return None
    res = res_levels[0].price; sup = sup_levels[0].price
    if res <= sup: return None
    rng = res - sup; cur = s.closes[-1]
    if cur > res + rng * 0.01:
        return _make(s, "Rectangle", Bias.LONG, res+0.02, sup-0.02, res+0.02+rng, 0.58,
                     f"Rectangle breakout above {res:.2f}",
                     key_levels={"resistance": res, "support": sup})
    elif cur < sup - rng * 0.01:
        return _make(s, "Rectangle", Bias.SHORT, sup-0.02, res+0.02, sup-0.02-rng, 0.58,
                     f"Rectangle breakdown below {sup:.2f}",
                     key_levels={"resistance": res, "support": sup})
    return None

def _detect_rising_wedge(s):
    """Rising Wedge — FIX: target uses widest point of wedge, not narrowest."""
    if len(s.zz_highs) < 2 or len(s.zz_lows) < 2: return None
    u = fit_trendline(s.zz_highs[-3:] if len(s.zz_highs)>=3 else s.zz_highs[-2:])
    l = fit_trendline(s.zz_lows[-3:] if len(s.zz_lows)>=3 else s.zz_lows[-2:])
    if u is None or l is None: return None
    if not (u.slope > 0 and l.slope > 0 and u.slope < l.slope): return None
    lp = l.price_at(s.n-1); up = u.price_at(s.n-1)
    if s.closes[-1] > lp: return None
    # FIX: Use widest point (start of wedge) for measured move
    start_idx = min(u.start_index, l.start_index)
    wide_upper = u.price_at(start_idx)
    wide_lower = l.price_at(start_idx)
    widest = abs(wide_upper - wide_lower)
    if widest <= 0: widest = abs(up - lp)
    return _make(s, "Rising Wedge", Bias.SHORT, lp-0.02, up+0.02, lp-0.02-widest, 0.62,
                 "Rising Wedge breakdown", key_levels={"upper": up, "lower": lp})

def _detect_falling_wedge(s):
    """Falling Wedge — FIX: target uses widest point of wedge."""
    if len(s.zz_highs) < 2 or len(s.zz_lows) < 2: return None
    u = fit_trendline(s.zz_highs[-3:] if len(s.zz_highs)>=3 else s.zz_highs[-2:])
    l = fit_trendline(s.zz_lows[-3:] if len(s.zz_lows)>=3 else s.zz_lows[-2:])
    if u is None or l is None: return None
    if not (u.slope < 0 and l.slope < 0 and u.slope > l.slope): return None
    up = u.price_at(s.n-1); lp = l.price_at(s.n-1)
    if s.closes[-1] < up: return None
    start_idx = min(u.start_index, l.start_index)
    wide_upper = u.price_at(start_idx)
    wide_lower = l.price_at(start_idx)
    widest = abs(wide_upper - wide_lower)
    if widest <= 0: widest = abs(up - lp)
    return _make(s, "Falling Wedge", Bias.LONG, up+0.02, lp-0.02, up+0.02+widest, 0.62,
                 "Falling Wedge breakout", key_levels={"upper": up, "lower": lp})


# ==============================================================================
# CANDLESTICK PATTERNS (10) — all now require _candle_context_ok()
# ==============================================================================

def _detect_bullish_engulfing(s):
    if s.n < 10: return None
    prev, cur = s.bars[-2], s.bars[-1]
    if not (_is_red(prev) and _is_green(cur)): return None
    if not (cur.open <= prev.close and cur.close >= prev.open): return None
    if _body(cur) <= _body(prev) * 1.1: return None
    if not _candle_context_ok(s, is_bullish=True): return None  # CONTEXT FILTER
    entry = round(cur.close, 2)
    stop = round(min(prev.low, cur.low) - 0.02, 2)
    risk = entry - stop
    if risk <= 0: return None
    return _make(s, "Bullish Engulfing", Bias.LONG, entry, stop, entry + risk*2, 0.58,
                 "Bullish Engulfing at S/R with volume")

def _detect_bearish_engulfing(s):
    if s.n < 10: return None
    prev, cur = s.bars[-2], s.bars[-1]
    if not (_is_green(prev) and _is_red(cur)): return None
    if not (cur.open >= prev.close and cur.close <= prev.open): return None
    if _body(cur) <= _body(prev) * 1.1: return None
    if not _candle_context_ok(s, is_bullish=False): return None
    entry = round(cur.close, 2)
    stop = round(max(prev.high, cur.high) + 0.02, 2)
    risk = stop - entry
    if risk <= 0: return None
    return _make(s, "Bearish Engulfing", Bias.SHORT, entry, stop, entry - risk*2, 0.58,
                 "Bearish Engulfing at S/R with volume")

def _detect_morning_star(s):
    if s.n < 10: return None
    b1, b2, b3 = s.bars[-3], s.bars[-2], s.bars[-1]
    if not _is_red(b1) or not _is_green(b3): return None
    if _body(b2) >= _body(b1) * 0.3: return None
    if _body(b3) < _body(b1) * 0.5: return None
    if b2.high > b1.open: return None
    if not _candle_context_ok(s, is_bullish=True, key_bar_idx=-2): return None
    entry = round(b3.close, 2)
    stop = round(min(b1.low, b2.low, b3.low) - 0.02, 2)
    risk = entry - stop
    if risk <= 0: return None
    return _make(s, "Morning Star", Bias.LONG, entry, stop, entry + risk*2, 0.60,
                 "Morning Star at S/R with volume")

def _detect_evening_star(s):
    if s.n < 10: return None
    b1, b2, b3 = s.bars[-3], s.bars[-2], s.bars[-1]
    if not _is_green(b1) or not _is_red(b3): return None
    if _body(b2) >= _body(b1) * 0.3: return None
    if _body(b3) < _body(b1) * 0.5: return None
    if b2.low < b1.close: return None
    if not _candle_context_ok(s, is_bullish=False, key_bar_idx=-2): return None
    entry = round(b3.close, 2)
    stop = round(max(b1.high, b2.high, b3.high) + 0.02, 2)
    risk = stop - entry
    if risk <= 0: return None
    return _make(s, "Evening Star", Bias.SHORT, entry, stop, entry - risk*2, 0.60,
                 "Evening Star at S/R with volume")

def _detect_hammer(s):
    if s.n < 10: return None
    bar = s.bars[-1]
    body = _body(bar); ls = _lower_shadow(bar); us = _upper_shadow(bar); rng = _range(bar)
    if rng == 0 or body == 0: return None
    if ls < body * 2 or us > body * 0.5: return None
    if not _candle_context_ok(s, is_bullish=True): return None
    entry = round(bar.close, 2)
    stop = round(bar.low - 0.02, 2)
    risk = entry - stop
    if risk <= 0: return None
    return _make(s, "Hammer", Bias.LONG, entry, stop, entry + risk*2, 0.55,
                 f"Hammer at S/R (shadow/body={ls/body:.1f}x)")

def _detect_shooting_star(s):
    if s.n < 10: return None
    bar = s.bars[-1]
    body = _body(bar); us = _upper_shadow(bar); ls = _lower_shadow(bar); rng = _range(bar)
    if rng == 0 or body == 0: return None
    if us < body * 2 or ls > body * 0.5: return None
    if not _candle_context_ok(s, is_bullish=False): return None
    entry = round(bar.close, 2)
    stop = round(bar.high + 0.02, 2)
    risk = stop - entry
    if risk <= 0: return None
    return _make(s, "Shooting Star", Bias.SHORT, entry, stop, entry - risk*2, 0.55,
                 f"Shooting Star at S/R (shadow/body={us/body:.1f}x)")

def _detect_doji(s):
    if s.n < 10: return None
    bar = s.bars[-1]
    body = _body(bar); rng = _range(bar)
    if rng == 0: return None
    if body / rng > 0.10: return None
    recent_high = float(np.max(s.highs[-10:]))
    recent_low = float(np.min(s.lows[-10:]))
    mid = (recent_high + recent_low) / 2
    if bar.close > mid:
        if not _candle_context_ok(s, is_bullish=False): return None
        stop = round(bar.high + 0.02, 2); entry = round(bar.close, 2)
        risk = stop - entry
        if risk <= 0: return None
        return _make(s, "Doji", Bias.SHORT, entry, stop, entry - risk*1.5, 0.48,
                     "Doji at resistance with volume")
    else:
        if not _candle_context_ok(s, is_bullish=True): return None
        stop = round(bar.low - 0.02, 2); entry = round(bar.close, 2)
        risk = entry - stop
        if risk <= 0: return None
        return _make(s, "Doji", Bias.LONG, entry, stop, entry + risk*1.5, 0.48,
                     "Doji at support with volume")

def _detect_dragonfly_doji(s):
    if s.n < 10: return None
    bar = s.bars[-1]
    body = _body(bar); rng = _range(bar); ls = _lower_shadow(bar); us = _upper_shadow(bar)
    if rng == 0: return None
    if body / rng > 0.10 or us > rng * 0.10 or ls < rng * 0.65: return None
    if not _candle_context_ok(s, is_bullish=True): return None
    entry = round(bar.close, 2)
    stop = round(bar.low - 0.02, 2)
    risk = entry - stop
    if risk <= 0: return None
    return _make(s, "Dragonfly Doji", Bias.LONG, entry, stop, entry + risk*2, 0.52,
                 "Dragonfly Doji at S/R with volume")

def _detect_three_white_soldiers(s):
    if s.n < 10: return None
    b1, b2, b3 = s.bars[-3], s.bars[-2], s.bars[-1]
    if not (_is_green(b1) and _is_green(b2) and _is_green(b3)): return None
    if not (b2.close > b1.close and b3.close > b2.close): return None
    if b2.open < b1.open or b2.open > b1.close: return None
    if b3.open < b2.open or b3.open > b2.close: return None
    for b in [b1, b2, b3]:
        if _upper_shadow(b) > _body(b) * 0.5: return None
    # Context: S/R + volume (trend context: allow momentum continuation)
    if not _candle_context_ok(s, is_bullish=True, key_bar_idx=-3): return None
    entry = round(b3.close, 2)
    stop = round(b1.low - 0.02, 2)
    risk = entry - stop
    if risk <= 0: return None
    return _make(s, "Three White Soldiers", Bias.LONG, entry, stop, entry + risk*1.5, 0.58,
                 "Three White Soldiers at S/R with volume")

def _detect_three_black_crows(s):
    if s.n < 10: return None
    b1, b2, b3 = s.bars[-3], s.bars[-2], s.bars[-1]
    if not (_is_red(b1) and _is_red(b2) and _is_red(b3)): return None
    if not (b2.close < b1.close and b3.close < b2.close): return None
    if b2.open > b1.open or b2.open < b1.close: return None
    if b3.open > b2.open or b3.open < b2.close: return None
    for b in [b1, b2, b3]:
        if _lower_shadow(b) > _body(b) * 0.5: return None
    if not _candle_context_ok(s, is_bullish=False, key_bar_idx=-3): return None
    entry = round(b3.close, 2)
    stop = round(b1.high + 0.02, 2)
    risk = stop - entry
    if risk <= 0: return None
    return _make(s, "Three Black Crows", Bias.SHORT, entry, stop, entry - risk*1.5, 0.58,
                 "Three Black Crows at S/R with volume")


# ==============================================================================
# SMB SCALP PATTERNS (7) — removed HitchHiker, Spencer, BackSide, Breaking News
# ==============================================================================

def _detect_rubberband(s):
    """RubberBand — FIX: 2.0 ATR ext (was 1.0), VWAP target, vol confirm, time filter."""
    if s.n < 20 or s.current_atr <= 0: return None
    oi = s.day_open_idx
    if oi >= s.n - 5: return None
    op = s.bars[oi].open; db = s.bars[oi:]
    dl = min(b.low for b in db)
    dli = oi + min(range(len(db)), key=lambda i: db[i].low)
    atrs = (op - dl) / s.current_atr
    if atrs < 2.0 or dli - oi < 5: return None  # FIX: 2.0 ATR minimum (was 1.0)
    # Check for accelerating selling
    mid = (oi + dli) // 2
    r1 = [s.bars[i].high - s.bars[i].low for i in range(oi, mid)] if mid > oi else [0]
    r2 = [s.bars[i].high - s.bars[i].low for i in range(mid, dli)] if dli > mid else [0]
    if np.mean(r2) < np.mean(r1) * 1.2: return None
    # FIX: VWAP for target
    vwap = _compute_vwap(s, oi)
    for i in range(dli+1, min(dli+10, s.n)):
        # FIX: Hard time filter — rubber band window is 10:00-11:00
        if not _in_time(s.bars[i].timestamp, 10, 0, 11, 0): continue
        if s.bars[i].close > s.bars[i].open:
            # FIX: Volume on bounce bar
            avg_vol = float(np.mean(s.volumes[max(0, i-20):i]))
            if avg_vol > 0 and s.volumes[i] < avg_vol * 1.3: continue
            if sum(1 for j in range(max(dli, i-5), i) if s.bars[i].high > s.bars[j].high) >= 2:
                e = round(s.bars[i].high+0.02, 2); st = round(dl-0.02, 2)
                r = e - st
                if r <= 0: continue
                # FIX: Target is VWAP or open, whichever is closer
                t = round(min(vwap, op), 2)
                if t <= e: t = round(e + r * 1.5, 2)  # Fallback if VWAP below entry
                c = 0.50 + (0.10 if atrs >= 3 else 0)
                return _make(s, "RubberBand Scalp", Bias.LONG, e, st, t, c,
                             f"RubberBand: {atrs:.1f} ATR ext, target VWAP", max_attempts=2,
                             exit_strategy="1/3 at 1:1, 1/3 at 2:1, 1/3 VWAP",
                             ideal_time="10:00-11:00 AM ET", key_levels={"day_low": dl, "vwap": round(vwap, 2)})
    return None

def _detect_orb(s, minutes=15):
    """ORB — FIX: stop at ORB opposite, target 1.5x range, range size filters."""
    if s.n < 10: return None
    oi = s.day_open_idx
    if oi >= s.n-5: return None
    ot = s.timestamps[oi]; cut = ot + timedelta(minutes=minutes)
    oe = oi
    for i in range(oi, s.n):
        if s.timestamps[i] <= cut: oe = i
        else: break
    if oe <= oi: return None
    ob = s.bars[oi:oe+1]; oh = max(b.high for b in ob); ol = min(b.low for b in ob)
    orng = oh - ol
    if orng <= 0: return None
    # FIX: ORB range filters
    price = (oh + ol) / 2
    if price > 0:
        range_pct = orng / price * 100
        if range_pct > 3.0: return None  # Too chaotic
    if s.current_atr > 0 and orng < s.current_atr * 0.3: return None  # Too narrow
    nm = f"ORB {minutes}min"
    # FIX: Only detect breakout if it happened on the CURRENT bar (last bar).
    # Previous versions searched all 30 bars after ORB, causing re-detection
    # on every scan once the breakout persisted.
    cur = s.bars[-1]
    prev_close = s.bars[-2].close if s.n >= 2 else cur.close
    # Breakout long: current bar closes above ORB high AND previous didn't
    if cur.close > oh and prev_close <= oh:
        e = round(oh+0.02, 2); st = round(ol-0.02, 2)
        r = e - st
        if r <= 0: return None
        t = round(e + orng * 1.5, 2)
        avg_ob_vol = float(np.mean([b.volume for b in ob]))
        c = 0.50 + (0.15 if cur.volume > avg_ob_vol * 1.5 else 0)
        return _make(s, nm, Bias.LONG, e, st, t, c,
                     f"ORB {minutes}min above {oh:.2f}", max_attempts=2,
                     key_levels={"orb_high": oh, "orb_low": ol})
    # Breakout short: current bar closes below ORB low AND previous didn't
    if cur.close < ol and prev_close >= ol:
        e = round(ol-0.02, 2); st = round(oh+0.02, 2)
        r = st - e
        if r <= 0: return None
        t = round(e - orng * 1.5, 2)
        return _make(s, nm, Bias.SHORT, e, st, t, 0.55,
                     f"ORB {minutes}min below {ol:.2f}", max_attempts=2,
                     key_levels={"orb_high": oh, "orb_low": ol})
    return None

def _detect_second_chance(s):
    """Second Chance — FIX: recency requirement + ATR tolerance + prior-day level + volume.
    The bounce (entry trigger) must happen within the last 5 bars to prevent
    re-detection of the same historical sequence on every scan.
    """
    if s.n < 30 or len(s.sw_high_idx) < 2 or s.current_atr <= 0: return None
    atr = s.current_atr
    oi = s.day_open_idx
    for i in range(len(s.sw_high_idx)-1):
        lev = s.highs[s.sw_high_idx[i]]
        tol = atr * 0.3
        touches = [s.sw_high_idx[i]]
        for j in range(i+1, len(s.sw_high_idx)):
            if abs(s.highs[s.sw_high_idx[j]]-lev) < tol:
                touches.append(s.sw_high_idx[j])
        if len(touches) < 2: continue
        res = float(np.mean([s.highs[idx] for idx in touches]))
        # At least one touch from a prior day
        has_prior_day = any(s.timestamps[idx].date() < s.timestamps[-1].date()
                            for idx in touches if idx < len(s.timestamps))
        if not has_prior_day and oi > 0: continue
        last = touches[-1]
        for k in range(last+1, min(last+20, s.n)):
            if s.closes[k] > res*1.002:
                avg_vol = float(np.mean(s.volumes[max(0, k-20):k]))
                if avg_vol > 0 and s.volumes[k] < avg_vol * 1.3: continue
                for p in range(k+1, min(k+15, s.n)):
                    if s.lows[p] <= res*1.003 and p+1 < s.n and s.closes[p+1] > s.bars[p].high:
                        # FIX: The bounce bar (p+1) must be within last 5 bars
                        if p + 1 < s.n - 5: continue  # Stale signal, skip
                        e = round(s.closes[p+1], 2); st = round(s.bars[p].low-0.02, 2)
                        r = e - st
                        if r <= 0: continue
                        t = round(max(s.highs[k:p+1]), 2)
                        if (t-e)/r < 1: continue
                        c = 0.50 + (0.10 if s.bars[k].volume > s.bars[p].volume*1.3 else 0)
                        return _make(s, "Second Chance Scalp", Bias.LONG, e, st, t, c,
                                     f"Second Chance: retest {res:.2f}", max_attempts=2)
                break
    return None

def _detect_fashionably_late(s):
    """Fashionably Late — FIX: cross must be RECENT (last 3 bars), hard time filter."""
    if s.n < 30: return None
    oi = s.day_open_idx
    if s.n - oi < 20: return None
    e9s = ema(s.closes, 9)
    vs = np.zeros(s.n); cv, ctv = 0.0, 0.0
    for i in range(oi, s.n):
        tp = (s.highs[i]+s.lows[i]+s.closes[i])/3; cv += s.volumes[i]; ctv += tp*s.volumes[i]
        vs[i] = ctv/cv if cv > 0 else tp
    dl = float(np.min(s.lows[oi:]))
    # FIX: Only check the last 3 bars for a FRESH cross. Searching all history
    # caused the same cross to be re-detected on every subsequent scan.
    search_start = max(oi + 10, s.n - 3)
    for i in range(search_start, s.n):
        if np.isnan(e9s[i]) or np.isnan(e9s[i-1]) or vs[i]==0 or vs[i-1]==0: continue
        if not _in_time(s.timestamps[i], 10, 0, 14, 0): continue
        if e9s[i-1] < vs[i-1] and e9s[i] >= vs[i]:
            cp = vs[i]; mm = cp - dl
            if mm <= 0: continue
            if i - oi < 6: continue
            c = 0.55 + (0.10 if _in_time(s.timestamps[i], 10, 0, 10, 45) else 0)
            return _make(s, "Fashionably Late", Bias.LONG, round(cp, 2), round(cp-mm/3, 2),
                         round(cp+mm, 2), c, f"Fashionably Late: measured ${mm:.2f}",
                         ideal_time="10:00-10:45 ET", key_levels={"cross": round(cp, 2)})
    return None

def _detect_gap_give_and_go(s):
    """Gap Give & Go — FIX: 1.5% gap min, 30% retrace, better target, time filter."""
    if s.n < 15: return None
    oi = s.day_open_idx
    if oi == 0 or s.n - oi < 10: return None
    pc = s.bars[oi-1].close; op = s.bars[oi].open
    gp = (op - pc) / pc * 100
    if gp < 1.5: return None  # FIX: 1.5% minimum gap (was 0.5%)
    gap_size = op - pc
    de = None
    for i in range(oi+1, min(oi+6, s.n)):
        retrace = (op - s.bars[i].low) / gap_size if gap_size > 0 else 0
        if retrace >= 0.30:  # FIX: Must retrace at least 30% of gap
            de = i; break
    if de is None: return None
    # FIX: Must happen in first 30 minutes
    for cl in range(3, 8):
        ce = de + cl
        if ce >= s.n - 1: continue
        if not _in_time(s.timestamps[ce], 9, 30, 10, 0): continue  # FIX: time filter
        cb = s.bars[de:ce]; ch = max(b.high for b in cb); clo = min(b.low for b in cb)
        if (ch - clo) > gap_size * 0.50: continue
        if s.bars[ce].high > ch:
            e = round(ch+0.02, 2); st = round(clo-0.02, 2); r = e - st
            if r <= 0: continue
            # FIX: Target is gap open + 50% of gap (was just gap open)
            t = round(op + gap_size * 0.5, 2)
            if (t - e) / r < 1: continue
            c = 0.55
            return _make(s, "Gap Give & Go", Bias.LONG, e, st, t, c,
                         f"Gap G&G: {gp:.1f}%", max_attempts=2,
                         ideal_time="9:30-10:00 AM ET")
    return None

def _detect_tidal_wave(s):
    """Tidal Wave — REWRITTEN: all 5 fixes applied.
    1. Stop at first (highest) bounce high
    2. Require 3+ support touches
    3. ATR-based tolerance
    4. Pattern spans >= 15 bars
    5. Volume expansion on breakdown
    """
    if s.n < 40 or len(s.sw_low_idx) < 3 or s.current_atr <= 0: return None
    atr = s.current_atr
    for i in range(len(s.sw_low_idx)):
        sup = s.lows[s.sw_low_idx[i]]
        tol = atr * 0.3
        touches = [s.sw_low_idx[i]]
        for j in range(i + 1, len(s.sw_low_idx)):
            if abs(s.lows[s.sw_low_idx[j]] - sup) < tol:
                touches.append(s.sw_low_idx[j])
        if len(touches) < 3: continue
        if touches[-1] - touches[0] < 15: continue
        bounce_highs = []
        for t in range(len(touches) - 1):
            btw = s.highs[touches[t]:touches[t + 1]]
            if len(btw) > 0: bounce_highs.append(float(np.max(btw)))
        if len(bounce_highs) < 2: continue
        if not all(bounce_highs[k] > bounce_highs[k + 1] for k in range(len(bounce_highs) - 1)):
            continue
        if s.closes[-1] >= sup: continue
        breakdown_bar = None
        for bi in range(touches[-1], s.n):
            if s.closes[bi] < sup - tol * 0.5:
                breakdown_bar = bi; break
        if breakdown_bar is None: continue
        avg_vol = float(np.mean(s.volumes[max(0, breakdown_bar - 20):breakdown_bar]))
        if avg_vol > 0 and s.volumes[breakdown_bar] < avg_vol * 1.5: continue
        first_bounce_high = bounce_highs[0]
        entry = round(sup - atr * 0.05, 2)
        stop = round(first_bounce_high + atr * 0.1, 2)
        risk = stop - entry
        if risk <= 0 or risk < atr * 0.3: continue
        avg_bounce = float(np.mean(bounce_highs)) - sup
        target = round(entry - max(avg_bounce, risk * 1.5), 2)
        reward = entry - target
        if reward / risk < 1.0: continue
        conf = 0.50 + min(0.15, (len(touches) - 3) * 0.05)
        diminish_ratio = bounce_highs[-1] / bounce_highs[0]
        conf += (1 - diminish_ratio) * 0.15
        if avg_vol > 0 and s.volumes[breakdown_bar] > avg_vol * 2.5: conf += 0.05
        return _make(s, "Tidal Wave", Bias.SHORT, entry, stop, target, conf,
                     f"Tidal Wave: {len(touches)} touches, "
                     f"bounces diminish {bounce_highs[0]:.2f}→{bounce_highs[-1]:.2f}",
                     key_levels={"support": round(sup, 2),
                                 "first_bounce": round(first_bounce_high, 2)})
    return None


# ==============================================================================
# QUANT STRATEGY PATTERNS — Intraday (4)
# ==============================================================================

def _detect_mean_reversion(s):
    """Mean Reversion — FIX: regime filter, 50-bar lookback, VWAP confluence, S/R stop."""
    if s.n < 55 or s.current_atr <= 0: return None
    # FIX: Regime filter — only in mean-reverting or mixed regimes
    if s._regime in ("trending_bull", "trending_bear"): return None
    # FIX: 50-bar lookback (was 20)
    ma = float(np.mean(s.closes[-50:])); std = float(np.std(s.closes[-50:]))
    if std == 0: return None
    z = (s.closes[-1] - ma) / std
    if abs(z) < 2.0: return None
    atr = s.current_atr
    # FIX: VWAP confluence check
    oi = s.day_open_idx
    vwap = _compute_vwap(s, oi)
    if z < -2:
        # Oversold + below VWAP = stronger signal
        if s.closes[-1] > vwap: return None  # FIX: Must be below VWAP for long mean reversion
        # FIX: S/R based stop — find nearest support below
        stop_level = s.closes[-1] - atr * 1.5
        for lvl in s.sr_levels:
            if lvl.price < s.closes[-1] - atr * 0.3:
                stop_level = lvl.price - atr * 0.1; break
        e = round(s.closes[-1], 2); st = round(stop_level, 2); r = e - st
        if r <= 0: return None
        return _make(s, "Mean Reversion", Bias.LONG, e, st, round(ma, 2), 0.55,
                     f"Mean Reversion: z={z:.2f}, VWAP confluence, target MA {ma:.2f}")
    else:
        if s.closes[-1] < vwap: return None  # Must be above VWAP for short
        stop_level = s.closes[-1] + atr * 1.5
        for lvl in s.sr_levels:
            if lvl.price > s.closes[-1] + atr * 0.3:
                stop_level = lvl.price + atr * 0.1; break
        e = round(s.closes[-1], 2); st = round(stop_level, 2); r = st - e
        if r <= 0: return None
        return _make(s, "Mean Reversion", Bias.SHORT, e, st, round(ma, 2), 0.55,
                     f"Mean Reversion: z={z:.2f}, VWAP confluence, target MA {ma:.2f}")

def _detect_trend_pullback(s):
    """Trend Pullback — FIX: regime filter, VWAP requirement, tighter EMA proximity."""
    if s.n < 55 or s.current_atr <= 0: return None
    # FIX: Regime filter — only in trending markets
    if s._regime not in ("trending_bull",): return None  # Only bull for longs
    sma50 = float(np.mean(s.closes[-50:])); e21 = ema_last(s.closes, 21)
    cur = s.closes[-1]
    if cur < sma50: return None
    # FIX: Tighter EMA proximity — within 0.5 ATR (was 1.5)
    dist = abs(cur - e21) / s.current_atr
    if dist > 0.5 or cur > e21 * 1.005: return None
    if not _is_green(s.bars[-1]): return None
    # FIX: Must be above VWAP
    vwap = _compute_vwap(s, s.day_open_idx)
    if cur < vwap: return None
    e = round(cur, 2); st = round(e21 - s.current_atr, 2); r = e - st
    if r <= 0: return None
    return _make(s, "Trend Pullback", Bias.LONG, e, st, round(e+2*r, 2), 0.58,
                 f"Trend Pullback: above 50 SMA + VWAP, bounce off 21 EMA",
                 key_levels={"sma50": round(sma50, 2), "ema21": round(e21, 2), "vwap": round(vwap, 2)})

def _detect_gap_fade(s):
    """Gap Fade — FIX: better stop, time filter, regime filter."""
    if s.n < 5: return None
    oi = s.day_open_idx
    if oi == 0: return None
    pc = s.bars[oi-1].close; op = s.bars[oi].open
    gap_pct = (op - pc) / pc * 100
    if abs(gap_pct) < 2.0: return None
    # FIX: Regime filter — don't fade gaps in strong trends
    if gap_pct > 2.0 and s._regime == "trending_bull": return None
    if gap_pct < -2.0 and s._regime == "trending_bear": return None
    # FIX: Time filter — only fade in first 60 minutes
    if not _in_time(s.timestamps[-1], 9, 30, 10, 30): return None
    gap_size = abs(op - pc)
    if gap_pct > 2.0:
        # FIX: Stop at gap open + 50% of gap (not day's high)
        e = round(s.closes[-1], 2); st = round(op + gap_size * 0.5, 2)
        r = st - e
        if r <= 0: return None
        t = round(pc, 2)
        if (e - t) / r < 0.5: return None
        return _make(s, "Gap Fade", Bias.SHORT, e, st, t, 0.52,
                     f"Gap Fade: {gap_pct:.1f}% gap up, target fill",
                     key_levels={"prev_close": pc, "gap_open": op})
    else:
        e = round(s.closes[-1], 2); st = round(op - gap_size * 0.5, 2)
        r = e - st
        if r <= 0: return None
        t = round(pc, 2)
        if (t - e) / r < 0.5: return None
        return _make(s, "Gap Fade", Bias.LONG, e, st, t, 0.52,
                     f"Gap Fade: {gap_pct:.1f}% gap down, target fill",
                     key_levels={"prev_close": pc, "gap_open": op})

def _detect_vwap_reversion(s):
    """VWAP Reversion — FIX: 2.5 ATR threshold, regime, volume, time filter."""
    if s.n < 20 or s.current_atr <= 0: return None
    oi = s.day_open_idx
    if s.n - oi < 10: return None
    # FIX: Regime filter
    if s._regime in ("trending_bull", "trending_bear"): return None
    # FIX: Time filter — VWAP reversion best mid-day
    if not _in_time(s.timestamps[-1], 10, 30, 14, 0): return None
    vwap = _compute_vwap(s, oi)
    dist = (s.closes[-1] - vwap) / s.current_atr
    if abs(dist) < 2.5: return None  # FIX: 2.5 ATR threshold (was 1.5)
    # FIX: Volume at extreme — exhaustion signal
    avg_vol = float(np.mean(s.volumes[-20:]))
    if avg_vol > 0 and s.volumes[-1] < avg_vol * 1.2: return None
    if dist > 2.5:
        e = round(s.closes[-1], 2); st = round(s.closes[-1]+s.current_atr, 2)
        r = st - e
        if r <= 0: return None
        return _make(s, "VWAP Reversion", Bias.SHORT, e, st, round(vwap, 2), 0.55,
                     f"VWAP Reversion: {dist:.1f} ATR above VWAP",
                     key_levels={"vwap": round(vwap, 2)})
    else:
        e = round(s.closes[-1], 2); st = round(s.closes[-1]-s.current_atr, 2)
        r = e - st
        if r <= 0: return None
        return _make(s, "VWAP Reversion", Bias.LONG, e, st, round(vwap, 2), 0.55,
                     f"VWAP Reversion: {abs(dist):.1f} ATR below VWAP",
                     key_levels={"vwap": round(vwap, 2)})


# ==============================================================================
# QUANT STRATEGY PATTERNS — Daily only (5)
# These only fire when timeframe is "1d". On 5min/15min they return None.
# ==============================================================================

def _detect_momentum_breakout(s):
    """Price > 20-bar high. On daily bars: 20-day breakout."""
    if s.timeframe != "1d": return None  # Daily only
    if s.n < 25: return None
    h20 = float(np.max(s.highs[-21:-1])); l20 = float(np.min(s.lows[-20:]))
    if s.closes[-1] <= h20: return None
    e = round(s.closes[-1], 2); st = round(l20, 2); r = e - st
    if r <= 0: return None
    return _make(s, "Momentum Breakout", Bias.LONG, e, st, round(e+(h20-l20), 2), 0.55,
                 f"20-day high breakout", key_levels={"20d_high": h20, "20d_low": l20})

def _detect_vol_compression_breakout(s):
    """Bollinger squeeze → expansion. Daily only."""
    if s.timeframe != "1d": return None
    if s.n < 30: return None
    r = atr_ratio(s.highs, s.lows, s.closes, atr_period=14, baseline_lookback=40)
    if r > 0.6: return None
    sma20 = float(np.mean(s.closes[-20:]))
    if s.closes[-1] > sma20:
        e = round(s.closes[-1], 2); st = round(sma20-s.current_atr, 2); risk = e - st
        if risk <= 0: return None
        return _make(s, "Vol Compression Breakout", Bias.LONG, e, st, round(e+2*risk, 2), 0.58,
                     f"Vol squeeze breakout (ATR ratio {r:.2f})", key_levels={"sma20": round(sma20, 2)})
    else:
        e = round(s.closes[-1], 2); st = round(sma20+s.current_atr, 2); risk = st - e
        if risk <= 0: return None
        return _make(s, "Vol Compression Breakout", Bias.SHORT, e, st, round(e-2*risk, 2), 0.58,
                     f"Vol squeeze breakdown (ATR ratio {r:.2f})")

def _detect_range_expansion(s):
    """Today's range > 2x average range. Daily only."""
    if s.timeframe != "1d": return None
    if s.n < 15: return None
    avg_r = float(np.mean(s.highs[-15:-1] - s.lows[-15:-1]))
    if avg_r == 0: return None
    cur_r = s.highs[-1] - s.lows[-1]
    if cur_r < avg_r * 2: return None
    rx = cur_r / avg_r
    if _is_green(s.bars[-1]):
        e = round(s.closes[-1], 2); st = round(s.lows[-1]-0.02, 2); r = e - st
        if r <= 0: return None
        return _make(s, "Range Expansion", Bias.LONG, e, st, round(e+r*1.5, 2), 0.52,
                     f"Range Expansion: {rx:.1f}x avg range (bullish)")
    else:
        e = round(s.closes[-1], 2); st = round(s.highs[-1]+0.02, 2); r = st - e
        if r <= 0: return None
        return _make(s, "Range Expansion", Bias.SHORT, e, st, round(e-r*1.5, 2), 0.52,
                     f"Range Expansion: {rx:.1f}x avg range (bearish)")

def _detect_volume_breakout(s):
    """Volume > 3x average + directional close beyond recent range. Daily only."""
    if s.timeframe != "1d": return None
    if s.n < 20: return None
    avg_v = float(np.mean(s.volumes[-20:-1]))
    if avg_v == 0 or s.volumes[-1] < avg_v * 3: return None
    h10 = float(np.max(s.highs[-11:-1])); l10 = float(np.min(s.lows[-11:-1]))
    cur = s.closes[-1]; vx = s.volumes[-1] / avg_v
    if cur > h10:
        e = round(cur, 2); st = round(l10, 2); r = e - st
        if r <= 0: return None
        return _make(s, "Volume Breakout", Bias.LONG, e, st, round(e+r, 2), 0.55,
                     f"Volume Breakout: {vx:.0f}x vol, above 10-day high")
    elif cur < l10:
        e = round(cur, 2); st = round(h10, 2); r = st - e
        if r <= 0: return None
        return _make(s, "Volume Breakout", Bias.SHORT, e, st, round(e-r, 2), 0.55,
                     f"Volume Breakout: {vx:.0f}x vol, below 10-day low")
    return None

def _detect_donchian_breakout(s):
    """Donchian Channel: close > highest high of last 50 bars. Daily only."""
    if s.timeframe != "1d": return None
    if s.n < 55: return None
    h50 = float(np.max(s.highs[-51:-1])); l50 = float(np.min(s.lows[-51:-1]))
    cur = s.closes[-1]
    if cur > h50:
        mid = (h50+l50)/2; e = round(cur, 2); st = round(mid, 2); r = e - st
        if r <= 0: return None
        return _make(s, "Donchian Breakout", Bias.LONG, e, st, round(e+r, 2), 0.53,
                     f"Donchian 50-day high breakout", key_levels={"50_high": h50, "50_low": l50})
    elif cur < l50:
        mid = (h50+l50)/2; e = round(cur, 2); st = round(mid, 2); r = st - e
        if r <= 0: return None
        return _make(s, "Donchian Breakout", Bias.SHORT, e, st, round(e-r, 2), 0.53,
                     f"Donchian 50-day low breakdown", key_levels={"50_high": h50, "50_low": l50})
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
    # SMB Scalps (7)
    _detect_rubberband,
    lambda s: _detect_orb(s, 15), lambda s: _detect_orb(s, 30),
    _detect_second_chance,
    _detect_fashionably_late,
    _detect_gap_give_and_go, _detect_tidal_wave,
    # Quant — Intraday (4)
    _detect_mean_reversion, _detect_trend_pullback,
    _detect_gap_fade, _detect_vwap_reversion,
    # Quant — Daily only (5) — return None on 5min/15min
    _detect_momentum_breakout, _detect_vol_compression_breakout,
    _detect_range_expansion, _detect_volume_breakout, _detect_donchian_breakout,
]


def classify_all(bars: BarSeries) -> list[TradeSetup]:
    """Run ALL 42 pattern detectors. Returns setups sorted by confidence."""
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