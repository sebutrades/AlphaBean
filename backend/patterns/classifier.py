"""
patterns/classifier.py — Pattern classification with tunable parameters.
v3.0 — All strategies wired with get_param() for Optuna optimization.
Candlestick patterns removed. 55 active strategies across 4 timeframes.
"""
from datetime import datetime, time, timedelta
from typing import Optional
from backend.optimization.param_inject import get_param
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


# ═══════════════════════════════════════════════════════════════
# STRUCTURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

class ExtractedStructures:
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
        if self.n < 50: return "unknown"
        sma50 = float(np.mean(self.closes[-50:]))
        cur = self.closes[-1]
        r = atr_ratio(self.highs, self.lows, self.closes, 14, 40)
        if cur > sma50 and r > 0.8: return "trending_bull"
        elif cur < sma50 and r > 0.8: return "trending_bear"
        elif r < 0.6: return "mean_reverting"
        return "mixed"


def extract_structures(bars: BarSeries) -> ExtractedStructures:
    return ExtractedStructures(bars)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _make(s, name, bias, entry, stop, target, conf, desc,
          target_1=0.0, target_2=0.0, trail_type="atr", trail_param=2.0,
          position_splits=(0.5, 0.3, 0.2), **kw):
    risk = abs(entry - stop)
    if risk <= 0: return None
    rr = round(abs(target - entry) / risk, 2)
    if rr < 1.0: return None
    cur = s.closes[-1]
    atr = s.current_atr if s.current_atr > 0 else abs(cur * 0.01)
    if bias == Bias.LONG and cur > entry + atr * 0.5: return None
    if bias == Bias.SHORT and cur < entry - atr * 0.5: return None
    if target_1 == 0.0:
        target_1 = round(entry + risk, 2) if bias == Bias.LONG else round(entry - risk, 2)
    if target_2 == 0.0:
        target_2 = round(target, 2)
    meta = PATTERN_META.get(name, {})
    return TradeSetup(
        pattern_name=name, category=meta.get("cat", PatternCategory.CLASSICAL),
        symbol=s.symbol, bias=bias, entry_price=round(entry, 2),
        stop_loss=round(stop, 2), target_price=round(target, 2),
        risk_reward_ratio=rr, confidence=round(min(0.95, conf), 2),
        detected_at=s.timestamps[-1], description=desc,
        strategy_type=meta.get("type", "breakout"), win_rate=meta.get("wr", 0.5),
        timeframe_detected=s.timeframe, target_1=target_1, target_2=target_2,
        trail_type=trail_type, trail_param=trail_param,
        position_splits=position_splits, **kw)


def _atr_offset(atr, mult=0.1):
    return round(atr * mult, 2) if atr > 0 else 0.02

def _in_time(ts, sh, sm, eh, em):
    return time(sh, sm) <= ts.time() <= time(eh, em)

def _body(bar): return abs(bar.close - bar.open)
def _range(bar): return bar.high - bar.low
def _is_green(bar): return bar.close > bar.open
def _is_red(bar): return bar.close < bar.open

def _compute_vwap(s, start_idx):
    cv, ctv = 0.0, 0.0
    for i in range(start_idx, s.n):
        tp = (s.highs[i] + s.lows[i] + s.closes[i]) / 3
        cv += s.volumes[i]; ctv += tp * s.volumes[i]
    return ctv / cv if cv > 0 else s.closes[-1]

def _vwap_today(s):
    today = s.timestamps[-1].date()
    for i in range(s.n):
        if s.timestamps[i].date() == today:
            return _compute_vwap(s, i)
    return _compute_vwap(s, 0)

def _compute_ema9(s):
    if s.n < 10: return None
    mult = 2.0 / 10.0
    val = float(s.closes[0])
    for i in range(1, s.n):
        val = s.closes[i] * mult + val * (1 - mult)
    return val

def _volume_confirms_breakout(s, bar_idx=-1, threshold=1.3):
    idx = bar_idx if bar_idx >= 0 else s.n + bar_idx
    if idx < 20: return True
    avg_vol = float(np.mean(s.volumes[max(0, idx - 20):idx]))
    if avg_vol <= 0: return True
    return s.volumes[idx] >= avg_vol * threshold

def _volume_declining_formation(s, start_idx, end_idx):
    if end_idx - start_idx < 6: return True
    mid = (start_idx + end_idx) // 2
    first_half_vol = float(np.mean(s.volumes[start_idx:mid]))
    second_half_vol = float(np.mean(s.volumes[mid:end_idx]))
    if first_half_vol <= 0: return True
    return second_half_vol <= first_half_vol * 1.10

def _volume_exhaustion(s, bar_idx=-1):
    idx = bar_idx if bar_idx >= 0 else s.n + bar_idx
    if idx < 5: return True
    recent_avg = float(np.mean(s.volumes[max(0, idx - 3):idx]))
    if recent_avg <= 0: return True
    return s.volumes[idx] <= recent_avg * 1.3

def _volume_pattern_hs(s, ls_idx, head_idx, rs_idx):
    if ls_idx >= s.n or head_idx >= s.n or rs_idx >= s.n: return 0.0
    def avg_vol_around(idx, radius=2):
        return float(np.mean(s.volumes[max(0, idx - radius):min(s.n, idx + radius + 1)]))
    ls_vol, head_vol, rs_vol = avg_vol_around(ls_idx), avg_vol_around(head_idx), avg_vol_around(rs_idx)
    if ls_vol <= 0: return 0.0
    bonus = 0.0
    if head_vol < ls_vol: bonus += 0.04
    if rs_vol < head_vol: bonus += 0.04
    if ls_vol > head_vol > rs_vol: bonus += 0.04
    return bonus

def _volume_double_touch(s, idx1, idx2):
    if idx1 >= s.n or idx2 >= s.n: return 0.0
    def avg_vol_around(idx, radius=2):
        return float(np.mean(s.volumes[max(0, idx - radius):min(s.n, idx + radius + 1)]))
    v1, v2 = avg_vol_around(idx1), avg_vol_around(idx2)
    if v1 <= 0: return 0.0
    if v2 < v1 * 0.85: return 0.08
    elif v2 < v1: return 0.04
    return 0.0

def _nearest_sr_target(s, entry, bias_long, min_rr=1.0):
    risk = abs(entry - (entry - s.current_atr if bias_long else entry + s.current_atr))
    for level in s.sr_levels:
        if bias_long and level.price > entry:
            reward = level.price - entry
            if risk > 0 and reward / risk >= min_rr: return round(level.price, 2)
        elif not bias_long and level.price < entry:
            reward = entry - level.price
            if risk > 0 and reward / risk >= min_rr: return round(level.price, 2)
    return 0.0

def _regime_confidence_mult(s, strategy_type):
    regime = s._regime
    if regime in ("unknown", "mixed"): return 1.0
    alignment = {
        "trending_bull": {"momentum": 1.15, "breakout": 1.10, "scalp": 1.0, "mean_reversion": 0.75},
        "trending_bear": {"momentum": 1.10, "breakout": 1.05, "scalp": 1.0, "mean_reversion": 0.75},
        "mean_reverting": {"mean_reversion": 1.15, "scalp": 1.05, "breakout": 0.80, "momentum": 0.75},
    }
    return alignment.get(regime, {}).get(strategy_type, 1.0)

def _is_nr7(s):
    if s.n < 50: return False
    daily_ranges = {}
    for i in range(s.n):
        d = s.timestamps[i].date()
        if d not in daily_ranges:
            daily_ranges[d] = {"high": s.highs[i], "low": s.lows[i]}
        else:
            daily_ranges[d]["high"] = max(daily_ranges[d]["high"], s.highs[i])
            daily_ranges[d]["low"] = min(daily_ranges[d]["low"], s.lows[i])
    dates = sorted(daily_ranges.keys())
    if len(dates) < 3: return False
    completed = dates[:-1]
    if len(completed) < 7: return False
    recent_7 = completed[-7:]
    ranges = [daily_ranges[d]["high"] - daily_ranges[d]["low"] for d in recent_7]
    return ranges[-1] == min(ranges)

def _min_span_ok(idx1, idx2, timeframe):
    min_spans = {"5min": 20, "15min": 15, "1h": 10, "1d": 15}
    return abs(idx2 - idx1) >= min_spans.get(timeframe, 10)

def _compute_adx(s, period=14):
    if s.n < period * 3: return None
    plus_dm = np.zeros(s.n); minus_dm = np.zeros(s.n); tr = np.zeros(s.n)
    for i in range(1, s.n):
        up = s.highs[i] - s.highs[i-1]; down = s.lows[i-1] - s.lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(s.highs[i]-s.lows[i], abs(s.highs[i]-s.closes[i-1]), abs(s.lows[i]-s.closes[i-1]))
    def wilder_smooth(data, n):
        result = np.zeros(len(data)); result[n] = np.sum(data[1:n+1])
        for i in range(n+1, len(data)): result[i] = result[i-1] - (result[i-1]/n) + data[i]
        return result
    spdm = wilder_smooth(plus_dm, period); smdm = wilder_smooth(minus_dm, period)
    str_ = wilder_smooth(tr, period)
    pdi = np.zeros(s.n); mdi = np.zeros(s.n)
    for i in range(period, s.n):
        if str_[i] > 0: pdi[i] = 100*spdm[i]/str_[i]; mdi[i] = 100*smdm[i]/str_[i]
    dx = np.zeros(s.n)
    for i in range(period, s.n):
        ds = pdi[i]+mdi[i]
        if ds > 0: dx[i] = 100*abs(pdi[i]-mdi[i])/ds
    adx = np.zeros(s.n); start = period*2
    if start >= s.n: return None
    adx[start] = np.mean(dx[period:start+1])
    for i in range(start+1, s.n): adx[i] = (adx[i-1]*(period-1)+dx[i])/period
    return adx[-1] if adx[-1] > 0 else None

def _has_higher_lows(s, lookback=10):
    if s.n < lookback: return False
    lows = s.lows[-lookback:]; mid = lookback//2
    return min(lows[mid:]) > min(lows[:mid]) and min(lows[-3:]) > min(lows[:3])

def _has_lower_highs(s, lookback=10):
    if s.n < lookback: return False
    highs = s.highs[-lookback:]; mid = lookback//2
    return max(highs[mid:]) < max(highs[:mid]) and max(highs[-3:]) < max(highs[:3])

def _consecutive_weekly_trend(s, weeks_required=3, direction_long=True):
    if s.n < weeks_required * 5: return False
    weekly_closes = {}
    for i in range(s.n):
        key = s.timestamps[i].isocalendar()[:2]
        weekly_closes[key] = s.closes[i]
    weeks = sorted(weekly_closes.keys())
    if len(weeks) < weeks_required + 1: return False
    closes = [weekly_closes[w] for w in weeks[-(weeks_required+1):]]
    consecutive = 0
    for i in range(1, len(closes)):
        if direction_long and closes[i] > closes[i-1]: consecutive += 1
        elif not direction_long and closes[i] < closes[i-1]: consecutive += 1
        else: consecutive = 0
    return consecutive >= weeks_required


# ═══════════════════════════════════════════════════════════════
# JUICER LONG / SHORT
# ═══════════════════════════════════════════════════════════════

def _detect_juicer_long(s):
    """Juicer Trend Continuation — LONG. ADX + stacked SMAs + higher lows + weekly streak."""
    _p = "Juicer Long"
    if s.n < 60 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    cur = s.closes[-1]
    adx = _compute_adx(s, 14)
    if adx is None or adx < get_param(_p, "adx_threshold", 25): return None
    _sf = get_param(_p, "sma_fast", 20); _ss = get_param(_p, "sma_slow", 50)
    sma_f = float(np.mean(s.closes[-_sf:])); sma_s = float(np.mean(s.closes[-_ss:]))
    if not (cur > sma_f > sma_s): return None
    if not _has_higher_lows(s, 10): return None
    if not _consecutive_weekly_trend(s, get_param(_p, "weekly_streak", 3), True): return None
    vol_20 = float(np.mean(s.volumes[-20:])); vol_50 = float(np.mean(s.volumes[-50:]))
    if vol_50 > 0 and vol_20 < vol_50: return None
    regime_bonus = 0.08 if s._regime == "trending_bull" else 0.0
    adx_bonus = min(0.10, (adx - 25) / 50 * 0.10)
    vol_ratio = vol_20 / vol_50 if vol_50 > 0 else 1.0
    conf = 0.60 + adx_bonus + min(0.05, (vol_ratio - 1.0) * 0.10) + regime_bonus
    _trail = get_param(_p, "trail_atr_mult", 2.0)
    entry = cur; stop = cur - atr * get_param(_p, "stop_atr_mult", 2.0)
    t1 = round(entry + atr * get_param(_p, "t1_atr_mult", 2.0), 2)
    t2 = round(entry + atr * get_param(_p, "t2_atr_mult", 4.0), 2)
    if entry - stop <= 0: return None
    return _make(s, _p, Bias.LONG, entry, stop, t2, conf,
                 f"Juicer Long: ADX={adx:.0f}, {cur:.2f}>{sma_f:.2f}>{sma_s:.2f}, vol {vol_ratio:.1f}x",
                 target_1=t1, target_2=t2, trail_type="atr", trail_param=_trail,
                 position_splits=(0.25, 0.25, 0.50),
                 key_levels={"sma20": sma_f, "sma50": sma_s, "adx": round(adx, 1)})


def _detect_juicer_short(s):
    """Juicer Trend Continuation — SHORT. Mirror of long."""
    _p = "Juicer Short"
    if s.n < 60 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    cur = s.closes[-1]
    adx = _compute_adx(s, 14)
    if adx is None or adx < get_param(_p, "adx_threshold", 25): return None
    _sf = get_param(_p, "sma_fast", 20); _ss = get_param(_p, "sma_slow", 50)
    sma_f = float(np.mean(s.closes[-_sf:])); sma_s = float(np.mean(s.closes[-_ss:]))
    if not (cur < sma_f < sma_s): return None
    if not _has_lower_highs(s, 10): return None
    if not _consecutive_weekly_trend(s, get_param(_p, "weekly_streak", 3), False): return None
    vol_20 = float(np.mean(s.volumes[-20:])); vol_50 = float(np.mean(s.volumes[-50:]))
    if vol_50 > 0 and vol_20 < vol_50: return None
    regime_bonus = 0.08 if s._regime == "trending_bear" else 0.0
    adx_bonus = min(0.10, (adx - 25) / 50 * 0.10)
    vol_ratio = vol_20 / vol_50 if vol_50 > 0 else 1.0
    conf = 0.58 + adx_bonus + min(0.05, (vol_ratio - 1.0) * 0.10) + regime_bonus
    entry = cur; stop = cur + atr * get_param(_p, "stop_atr_mult", 2.0)
    t1 = round(entry - atr * get_param(_p, "t1_atr_mult", 2.0), 2)
    t2 = round(entry - atr * get_param(_p, "t2_atr_mult", 4.0), 2)
    if stop - entry <= 0: return None
    return _make(s, _p, Bias.SHORT, entry, stop, t2, conf,
                 f"Juicer Short: ADX={adx:.0f}, vol {vol_ratio:.1f}x",
                 target_1=t1, target_2=t2, trail_type="atr", trail_param=get_param(_p, "trail_atr_mult", 2.0),
                 position_splits=(0.25, 0.25, 0.50),
                 key_levels={"sma20": sma_f, "sma50": sma_s, "adx": round(adx, 1)})


# ═══════════════════════════════════════════════════════════════
# CLASSICAL PATTERNS (16 structural — no candlesticks)
# ═══════════════════════════════════════════════════════════════

def _detect_head_and_shoulders(s):
    _p = "Head & Shoulders"
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 2: return None
    atr = s.current_atr
    if atr <= 0: return None
    for i in range(len(s.zz_highs) - 2):
        ls, hd, rs = s.zz_highs[i], s.zz_highs[i+1], s.zz_highs[i+2]
        if not (hd.price > ls.price and hd.price > rs.price): continue
        sym_tol = max(ls.price * 0.03, atr * 0.5)
        if abs(ls.price - rs.price) > sym_tol: continue
        if not _min_span_ok(ls.index, rs.index, s.timeframe): continue
        lows_between = [l for l in s.zz_lows if ls.index < l.index < rs.index]
        if not lows_between: continue
        neckline = min(l.price for l in lows_between)
        if s.closes[-1] >= neckline: continue
        if not _volume_confirms_breakout(s, -1, get_param(_p, "vol_mult", 1.3)): continue
        vol_bonus = _volume_pattern_hs(s, ls.index, hd.index, rs.index)
        sym_pct = abs(ls.price - rs.price) / ls.price
        conf = 0.60 + (1 - sym_pct / 0.03) * 0.10 + vol_bonus
        conf *= _regime_confidence_mult(s, "breakout")
        buf = get_param(_p, "stop_atr_buffer", 0.10)
        entry = neckline - _atr_offset(atr, 0.10)
        stop = rs.price + _atr_offset(atr, buf)
        mm = hd.price - neckline
        t1_pct = get_param(_p, "t1_mm_pct", 0.75)
        t_full = entry - mm; t_partial = entry - mm * t1_pct
        return _make(s, _p, Bias.SHORT, entry, stop, t_full, conf,
                     f"H&S: head@{hd.price:.2f}, neckline@{neckline:.2f}",
                     target_1=round(t_partial, 2), target_2=round(t_full, 2),
                     key_levels={"head": hd.price, "neckline": neckline})
    return None


def _detect_inverse_hs(s):
    _p = "Inverse H&S"
    if len(s.zz_lows) < 3 or len(s.zz_highs) < 2: return None
    atr = s.current_atr
    if atr <= 0: return None
    for i in range(len(s.zz_lows) - 2):
        ls, hd, rs = s.zz_lows[i], s.zz_lows[i+1], s.zz_lows[i+2]
        if not (hd.price < ls.price and hd.price < rs.price): continue
        sym_tol = max(ls.price * 0.03, atr * 0.5)
        if abs(ls.price - rs.price) > sym_tol: continue
        if not _min_span_ok(ls.index, rs.index, s.timeframe): continue
        highs_between = [h for h in s.zz_highs if ls.index < h.index < rs.index]
        if not highs_between: continue
        neckline = max(h.price for h in highs_between)
        if s.closes[-1] <= neckline: continue
        if not _volume_confirms_breakout(s, -1, get_param(_p, "vol_mult", 1.3)): continue
        vol_bonus = _volume_pattern_hs(s, ls.index, hd.index, rs.index)
        sym_pct = abs(ls.price - rs.price) / ls.price
        conf = 0.62 + (1 - sym_pct / 0.03) * 0.10 + vol_bonus
        conf *= _regime_confidence_mult(s, "breakout")
        buf = get_param(_p, "stop_atr_buffer", 0.10)
        entry = neckline + _atr_offset(atr, 0.10)
        stop = rs.price - _atr_offset(atr, buf)
        mm = neckline - hd.price
        t1_pct = get_param(_p, "t1_mm_pct", 0.75)
        return _make(s, _p, Bias.LONG, entry, stop, entry + mm, conf,
                     f"Inv H&S: head@{hd.price:.2f}, neckline@{neckline:.2f}",
                     target_1=round(entry + mm * t1_pct, 2), target_2=round(entry + mm, 2),
                     key_levels={"head": hd.price, "neckline": neckline})
    return None


def _detect_double_top(s):
    _p = "Double Top"
    if len(s.zz_highs) < 2: return None
    atr = s.current_atr
    if atr <= 0: return None
    h1, h2 = s.zz_highs[-2], s.zz_highs[-1]
    tol = max(atr * 0.3, h1.price * 0.015)
    if abs(h1.price - h2.price) > tol: return None
    if not _min_span_ok(h1.index, h2.index, s.timeframe): return None
    valley = min(s.lows[h1.index:h2.index+1])
    if s.closes[-1] > valley: return None
    vol_bonus = _volume_double_touch(s, h1.index, h2.index)
    if not _volume_confirms_breakout(s, -1, 1.2): vol_bonus = max(0, vol_bonus - 0.04)
    peak_bonus = 0.05 if h2.price < h1.price else 0.0
    top = max(h1.price, h2.price)
    conf = (0.60 + vol_bonus + peak_bonus) * _regime_confidence_mult(s, "breakout")
    buf = get_param(_p, "stop_atr_buffer", 0.10)
    entry = valley - _atr_offset(atr, 0.10); stop = top + _atr_offset(atr, buf)
    mm = top - valley; t1_pct = get_param(_p, "t1_mm_pct", 0.75)
    return _make(s, _p, Bias.SHORT, entry, stop, entry - mm, conf,
                 f"Double Top at {top:.2f}", target_1=round(entry - mm * t1_pct, 2),
                 target_2=round(entry - mm, 2), key_levels={"top": top, "valley": valley})


def _detect_double_bottom(s):
    _p = "Double Bottom"
    if len(s.zz_lows) < 2: return None
    atr = s.current_atr
    if atr <= 0: return None
    l1, l2 = s.zz_lows[-2], s.zz_lows[-1]
    tol = max(atr * 0.3, l1.price * 0.015)
    if abs(l1.price - l2.price) > tol: return None
    if not _min_span_ok(l1.index, l2.index, s.timeframe): return None
    peak = max(s.highs[l1.index:l2.index+1])
    if s.closes[-1] < peak: return None
    vol_bonus = _volume_double_touch(s, l1.index, l2.index)
    if not _volume_confirms_breakout(s, -1, 1.2): vol_bonus = max(0, vol_bonus - 0.04)
    trough_bonus = 0.05 if l2.price > l1.price else 0.0
    bot = min(l1.price, l2.price)
    conf = (0.62 + vol_bonus + trough_bonus) * _regime_confidence_mult(s, "breakout")
    buf = get_param(_p, "stop_atr_buffer", 0.10)
    entry = peak + _atr_offset(atr, 0.10); stop = bot - _atr_offset(atr, buf)
    mm = peak - bot; t1_pct = get_param(_p, "t1_mm_pct", 0.75)
    return _make(s, _p, Bias.LONG, entry, stop, entry + mm, conf,
                 f"Double Bottom at {bot:.2f}", target_1=round(entry + mm * t1_pct, 2),
                 target_2=round(entry + mm, 2), key_levels={"bottom": bot, "peak": peak})


def _detect_triple_top(s):
    _p = "Triple Top"
    if len(s.zz_highs) < 3: return None
    atr = s.current_atr
    if atr <= 0: return None
    h1, h2, h3 = s.zz_highs[-3], s.zz_highs[-2], s.zz_highs[-1]
    prices = [h1.price, h2.price, h3.price]; avg = np.mean(prices)
    tol = max(atr * 0.3, avg * 0.015)
    if max(abs(p - avg) for p in prices) > tol: return None
    if not _min_span_ok(h1.index, h3.index, s.timeframe): return None
    valley = min(s.lows[h1.index:h3.index+1])
    if s.closes[-1] > valley: return None
    conf = 0.60 * _regime_confidence_mult(s, "breakout")
    buf = get_param(_p, "stop_atr_buffer", 0.10)
    entry = valley - _atr_offset(atr, 0.10); stop = max(prices) + _atr_offset(atr, buf)
    mm = max(prices) - valley; t1_pct = get_param(_p, "t1_mm_pct", 0.75)
    return _make(s, _p, Bias.SHORT, entry, stop, entry - mm, conf,
                 f"Triple Top at {avg:.2f}", target_1=round(entry - mm * t1_pct, 2),
                 target_2=round(entry - mm, 2))


def _detect_triple_bottom(s):
    _p = "Triple Bottom"
    if len(s.zz_lows) < 3: return None
    atr = s.current_atr
    if atr <= 0: return None
    l1, l2, l3 = s.zz_lows[-3], s.zz_lows[-2], s.zz_lows[-1]
    prices = [l1.price, l2.price, l3.price]; avg = np.mean(prices)
    tol = max(atr * 0.3, avg * 0.015)
    if max(abs(p - avg) for p in prices) > tol: return None
    if not _min_span_ok(l1.index, l3.index, s.timeframe): return None
    peak = max(s.highs[l1.index:l3.index+1])
    if s.closes[-1] < peak: return None
    conf = 0.62 * _regime_confidence_mult(s, "breakout")
    buf = get_param(_p, "stop_atr_buffer", 0.10)
    entry = peak + _atr_offset(atr, 0.10); stop = min(prices) - _atr_offset(atr, buf)
    mm = peak - min(prices); t1_pct = get_param(_p, "t1_mm_pct", 0.75)
    return _make(s, _p, Bias.LONG, entry, stop, entry + mm, conf,
                 f"Triple Bottom at {avg:.2f}", target_1=round(entry + mm * t1_pct, 2),
                 target_2=round(entry + mm, 2))


def _detect_ascending_triangle(s):
    _p = "Ascending Triangle"
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 3: return None
    atr = s.current_atr
    if atr <= 0: return None
    utl = fit_trendline(s.zz_highs[-4:] if len(s.zz_highs) >= 4 else s.zz_highs[-3:])
    ltl = fit_trendline(s.zz_lows[-4:] if len(s.zz_lows) >= 4 else s.zz_lows[-3:])
    if utl is None or ltl is None: return None
    if utl.num_points < 3 or ltl.num_points < 2: return None
    if not is_flat_line(utl, 0.15) or ltl.slope <= 0: return None
    res = utl.price_at(utl.end_index)
    if s.closes[-1] < res: return None
    vol_bonus = 0.05 if _volume_confirms_breakout(s, -1, 1.3) else 0.0
    sup = s.zz_lows[-1].price
    conf = (0.58 + vol_bonus) * _regime_confidence_mult(s, "breakout")
    buf = get_param(_p, "stop_atr_buffer", 0.10)
    entry = res + _atr_offset(atr, 0.10); stop = sup - _atr_offset(atr, buf)
    mm = res - sup; t1_pct = get_param(_p, "t1_mm_pct", 0.75)
    return _make(s, _p, Bias.LONG, entry, stop, entry + mm, conf,
                 f"Asc Triangle: flat top {res:.2f}",
                 target_1=round(entry + mm * t1_pct, 2), target_2=round(entry + mm, 2))


def _detect_descending_triangle(s):
    _p = "Descending Triangle"
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 3: return None
    atr = s.current_atr
    if atr <= 0: return None
    ltl = fit_trendline(s.zz_lows[-4:] if len(s.zz_lows) >= 4 else s.zz_lows[-3:])
    utl = fit_trendline(s.zz_highs[-4:] if len(s.zz_highs) >= 4 else s.zz_highs[-3:])
    if utl is None or ltl is None: return None
    if ltl.num_points < 3 or utl.num_points < 2: return None
    if not is_flat_line(ltl, 0.15) or utl.slope >= 0: return None
    sup = ltl.price_at(ltl.end_index)
    if s.closes[-1] > sup: return None
    vol_bonus = 0.05 if _volume_confirms_breakout(s, -1, 1.3) else 0.0
    res = s.zz_highs[-1].price
    conf = (0.56 + vol_bonus) * _regime_confidence_mult(s, "breakout")
    buf = get_param(_p, "stop_atr_buffer", 0.10)
    entry = sup - _atr_offset(atr, 0.10); stop = res + _atr_offset(atr, buf)
    mm = res - sup; t1_pct = get_param(_p, "t1_mm_pct", 0.75)
    return _make(s, _p, Bias.SHORT, entry, stop, entry - mm, conf,
                 f"Desc Triangle: flat bottom {sup:.2f}",
                 target_1=round(entry - mm * t1_pct, 2), target_2=round(entry - mm, 2))


def _detect_symmetrical_triangle(s):
    _p = "Symmetrical Triangle"
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 3: return None
    atr = s.current_atr
    if atr <= 0: return None
    utl = fit_trendline(s.zz_highs[-4:] if len(s.zz_highs) >= 4 else s.zz_highs[-3:])
    ltl = fit_trendline(s.zz_lows[-4:] if len(s.zz_lows) >= 4 else s.zz_lows[-3:])
    if utl is None or ltl is None: return None
    if utl.slope >= 0 or ltl.slope <= 0: return None
    up = utl.price_at(s.n-1); lo = ltl.price_at(s.n-1)
    if up <= lo: return None
    rng = up - lo; cur = s.closes[-1]
    vol_bonus = 0.05 if _volume_confirms_breakout(s, -1, 1.3) else 0.0
    conf = (0.50 + vol_bonus) * _regime_confidence_mult(s, "breakout")
    buf = get_param(_p, "stop_atr_buffer", 0.15)
    t1_pct = get_param(_p, "t1_mm_pct", 0.50)
    if cur > up:
        entry = up + _atr_offset(atr, 0.10); stop = lo - _atr_offset(atr, buf)
        return _make(s, _p, Bias.LONG, entry, stop, entry + rng, conf,
                     "Sym Triangle breakout", target_1=round(entry + rng * t1_pct, 2), target_2=round(entry + rng, 2))
    elif cur < lo:
        entry = lo - _atr_offset(atr, 0.10); stop = up + _atr_offset(atr, buf)
        return _make(s, _p, Bias.SHORT, entry, stop, entry - rng, conf,
                     "Sym Triangle breakdown", target_1=round(entry - rng * t1_pct, 2), target_2=round(entry - rng, 2))
    return None


def _detect_bull_flag(s):
    _p = "Bull Flag"
    if len(s.zz_lows) < 1 or len(s.zz_highs) < 1 or s.current_atr <= 0: return None
    atr = s.current_atr
    for li in range(len(s.zz_lows)):
        lo = s.zz_lows[li]
        post = [h for h in s.zz_highs if h.index > lo.index]
        if not post: continue
        hi = post[0]
        pole_size = hi.price - lo.price
        if pole_size < atr * 2.0: continue
        pole_bars = hi.index - lo.index
        if pole_bars > 15 or pole_bars < 2: continue
        fs = hi.index
        if fs >= s.n - 3: continue
        flag = s.bars[fs:]
        if len(flag) < 3 or len(flag) > 15: continue
        fl = min(b.low for b in flag)
        if (hi.price - fl) / pole_size > 0.50: continue
        if s.closes[-1] <= hi.price: continue
        pole_vol = float(np.mean(s.volumes[lo.index:hi.index+1]))
        flag_vol = float(np.mean([b.volume for b in flag]))
        vol_bonus = 0.05 if (pole_vol > flag_vol * 1.3) else 0.0
        vol_bonus += 0.05 if _volume_confirms_breakout(s, -1, 1.3) else 0.0
        flag_range = max(b.high for b in flag) - fl
        tightness = flag_range / pole_size if pole_size > 0 else 1.0
        tight_bonus = 0.10 if tightness < 0.20 else 0.0
        conf = (0.55 + vol_bonus + tight_bonus) * _regime_confidence_mult(s, "momentum")
        entry = hi.price + _atr_offset(atr, 0.05); stop = fl - _atr_offset(atr, get_param(_p, "stop_atr_buffer", 0.10))
        t1_pct = get_param(_p, "t1_mm_pct", 0.75)
        return _make(s, _p, Bias.LONG, entry, stop, entry + pole_size, conf,
                     f"Bull Flag: {pole_size/lo.price:.1%} pole",
                     target_1=round(entry + pole_size * t1_pct, 2), target_2=round(entry + pole_size, 2),
                     trail_type="ema9", trail_param=9.0)
    return None


def _detect_bear_flag(s):
    _p = "Bear Flag"
    if len(s.zz_highs) < 1 or len(s.zz_lows) < 1 or s.current_atr <= 0: return None
    atr = s.current_atr
    for hi_idx in range(len(s.zz_highs)):
        hi = s.zz_highs[hi_idx]
        post = [l for l in s.zz_lows if l.index > hi.index]
        if not post: continue
        lo = post[0]
        pole_size = hi.price - lo.price
        if pole_size < atr * 2.0: continue
        pole_bars = lo.index - hi.index
        if pole_bars > 15 or pole_bars < 2: continue
        fs = lo.index
        if fs >= s.n - 3: continue
        flag = s.bars[fs:]
        if len(flag) < 3 or len(flag) > 15: continue
        fh = max(b.high for b in flag)
        if (fh - lo.price) / pole_size > 0.50: continue
        if s.closes[-1] >= lo.price: continue
        pole_vol = float(np.mean(s.volumes[hi.index:lo.index+1]))
        flag_vol = float(np.mean([b.volume for b in flag]))
        vol_bonus = 0.05 if (pole_vol > flag_vol * 1.3) else 0.0
        vol_bonus += 0.05 if _volume_confirms_breakout(s, -1, 1.3) else 0.0
        conf = (0.53 + vol_bonus) * _regime_confidence_mult(s, "momentum")
        entry = lo.price - _atr_offset(atr, 0.05); stop = fh + _atr_offset(atr, get_param(_p, "stop_atr_buffer", 0.10))
        t1_pct = get_param(_p, "t1_mm_pct", 0.75)
        return _make(s, _p, Bias.SHORT, entry, stop, entry - pole_size, conf,
                     f"Bear Flag: {pole_size/hi.price:.1%} pole",
                     target_1=round(entry - pole_size * t1_pct, 2), target_2=round(entry - pole_size, 2),
                     trail_type="ema9", trail_param=9.0)
    return None


def _detect_cup_and_handle(s):
    _p = "Cup & Handle"
    if len(s.zz_lows) < 1 or s.n < 40: return None
    atr = s.current_atr
    if atr <= 0: return None
    cl = s.zz_lows[-1]
    lr = max(s.highs[:cl.index]) if cl.index > 5 else None
    rr = max(s.highs[cl.index:]) if cl.index < s.n - 5 else None
    if lr is None or rr is None: return None
    rim = min(lr, rr); depth = rim - cl.price
    if depth <= 0 or cl.index < 15 or s.n - cl.index < 15: return None
    handle = s.bars[-min(10, s.n):]
    hl = min(b.low for b in handle); hh = max(b.high for b in handle)
    ret = (rim - hl) / depth
    if ret > 0.50 or ret < 0.10 or s.closes[-1] < rim: return None
    upper_third = cl.price + depth * (2/3)
    if hl < upper_third: return None
    handle_slope = handle[-1].close - handle[0].close
    if handle_slope > atr * 0.3: return None
    avg_vol = float(np.mean(s.volumes[-20:]))
    vol_bonus = 0.05 if (avg_vol > 0 and s.volumes[-1] >= avg_vol * 1.3) else 0.0
    conf = (0.60 + vol_bonus) * _regime_confidence_mult(s, "breakout")
    buf = get_param(_p, "stop_atr_buffer", 0.10)
    entry = rim + _atr_offset(atr, 0.10); stop = hl - _atr_offset(atr, buf)
    t1_pct = get_param(_p, "t1_mm_pct", 0.75)
    return _make(s, _p, Bias.LONG, entry, stop, entry + depth, conf,
                 f"Cup & Handle: rim {rim:.2f}, depth {depth:.2f}",
                 target_1=round(entry + depth * t1_pct, 2), target_2=round(entry + depth, 2),
                 trail_type="ema9", trail_param=9.0)


def _detect_rectangle(s):
    _p = "Rectangle"
    if s.n < 20 or len(s.sr_levels) < 2: return None
    atr = s.current_atr
    if atr <= 0: return None
    res_levels = [l for l in s.sr_levels if l.level_type in ("resistance", "both")]
    sup_levels = [l for l in s.sr_levels if l.level_type in ("support", "both")]
    if not res_levels or not sup_levels: return None
    res = res_levels[0].price; sup = sup_levels[0].price
    if res <= sup: return None
    if res_levels[0].touches < 3 or sup_levels[0].touches < 3: return None
    rng = res - sup; cur = s.closes[-1]
    vol_bonus = 0.05 if _volume_confirms_breakout(s, -1, 1.3) else 0.0
    conf = (0.55 + vol_bonus) * _regime_confidence_mult(s, "breakout")
    buf = get_param(_p, "stop_atr_buffer", 0.15)
    t1_pct = get_param(_p, "t1_mm_pct", 0.75)
    if cur > res + rng * 0.01:
        entry = res + _atr_offset(atr, 0.10); stop = sup - _atr_offset(atr, buf)
        return _make(s, _p, Bias.LONG, entry, stop, entry + rng, conf,
                     f"Rectangle breakout above {res:.2f}",
                     target_1=round(entry + rng * t1_pct, 2), target_2=round(entry + rng, 2))
    elif cur < sup - rng * 0.01:
        entry = sup - _atr_offset(atr, 0.10); stop = res + _atr_offset(atr, buf)
        return _make(s, _p, Bias.SHORT, entry, stop, entry - rng, conf,
                     f"Rectangle breakdown below {sup:.2f}",
                     target_1=round(entry - rng * t1_pct, 2), target_2=round(entry - rng, 2))
    return None


def _detect_rising_wedge(s):
    _p = "Rising Wedge"
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 3: return None
    atr = s.current_atr
    if atr <= 0: return None
    u = fit_trendline(s.zz_highs[-4:] if len(s.zz_highs) >= 4 else s.zz_highs[-3:])
    l = fit_trendline(s.zz_lows[-4:] if len(s.zz_lows) >= 4 else s.zz_lows[-3:])
    if u is None or l is None: return None
    if not (u.slope > 0 and l.slope > 0 and u.slope < l.slope): return None
    lp = l.price_at(s.n-1); up = u.price_at(s.n-1)
    if s.closes[-1] > lp: return None
    start_idx = min(u.start_index, l.start_index)
    widest = abs(u.price_at(start_idx) - l.price_at(start_idx))
    if widest <= 0: widest = abs(up - lp)
    conf = 0.52 * _regime_confidence_mult(s, "breakout")
    buf = get_param(_p, "stop_atr_buffer", 0.10)
    entry = lp - _atr_offset(atr, 0.10); stop = up + _atr_offset(atr, buf)
    t1_pct = get_param(_p, "t1_mm_pct", 0.75)
    return _make(s, _p, Bias.SHORT, entry, stop, entry - widest, conf,
                 "Rising Wedge breakdown",
                 target_1=round(entry - widest * t1_pct, 2), target_2=round(entry - widest, 2))


def _detect_falling_wedge(s):
    _p = "Falling Wedge"
    if len(s.zz_highs) < 3 or len(s.zz_lows) < 3: return None
    atr = s.current_atr
    if atr <= 0: return None
    u = fit_trendline(s.zz_highs[-4:] if len(s.zz_highs) >= 4 else s.zz_highs[-3:])
    l = fit_trendline(s.zz_lows[-4:] if len(s.zz_lows) >= 4 else s.zz_lows[-3:])
    if u is None or l is None: return None
    if not (u.slope < 0 and l.slope < 0 and u.slope > l.slope): return None
    up = u.price_at(s.n-1); lp = l.price_at(s.n-1)
    if s.closes[-1] < up: return None
    start_idx = min(u.start_index, l.start_index)
    widest = abs(u.price_at(start_idx) - l.price_at(start_idx))
    if widest <= 0: widest = abs(up - lp)
    conf = 0.52 * _regime_confidence_mult(s, "breakout")
    buf = get_param(_p, "stop_atr_buffer", 0.10)
    entry = up + _atr_offset(atr, 0.10); stop = lp - _atr_offset(atr, buf)
    t1_pct = get_param(_p, "t1_mm_pct", 0.75)
    return _make(s, _p, Bias.LONG, entry, stop, entry + widest, conf,
                 "Falling Wedge breakout",
                 target_1=round(entry + widest * t1_pct, 2), target_2=round(entry + widest, 2))


# ═══════════════════════════════════════════════════════════════
# SMB SCALP PATTERNS (7)
# ═══════════════════════════════════════════════════════════════

def _detect_rubberband_scalp(s):
    _p = "RubberBand Scalp"
    if s.n < 20 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    cur_time = s.timestamps[-1].time()
    if not (time(10, 0) <= cur_time <= time(11, 0)): return None
    today = s.timestamps[-1].date()
    day_bars = [(i, s.bars[i]) for i in range(s.n) if s.timestamps[i].date() == today]
    if len(day_bars) < 6: return None
    open_price = day_bars[0][1].open
    dli = min(day_bars, key=lambda x: x[1].low)[0]; day_low = s.lows[dli]
    ext_atr = get_param(_p, "extension_atr", 2.0)
    if open_price - day_low < atr * ext_atr: return None
    drop_bars = [i for i, b in day_bars if i <= dli]
    if len(drop_bars) < 4: return None
    mid = len(drop_bars) // 2
    if np.mean([_range(s.bars[i]) for i in drop_bars[mid:]]) <= np.mean([_range(s.bars[i]) for i in drop_bars[:mid]]):
        return None
    vwap = _vwap_today(s)
    if vwap is None or vwap <= 0: return None
    bounce_vol_mult = get_param(_p, "bounce_vol_mult", 1.3)
    bounce_found = False; bounce_idx = -1
    for j in range(dli + 1, min(dli + 20, s.n)):
        if _is_green(s.bars[j]) and s.bars[j].close > s.bars[j - 1].high:
            prior_avg_vol = float(np.mean(s.volumes[max(0, j-3):j]))
            if prior_avg_vol > 0 and s.volumes[j] < prior_avg_vol * bounce_vol_mult: continue
            bounce_found = True; bounce_idx = j; break
    if not bounce_found or s.n - 1 - bounce_idx > 5: return None
    conf = 0.58; entry = s.bars[bounce_idx].close
    stop = day_low - _atr_offset(atr, get_param(_p, "stop_atr_mult", 0.10))
    if entry - stop <= 0: return None
    return _make(s, _p, Bias.LONG, entry, stop, round(vwap, 2), conf,
                 f"RubberBand: {(open_price-day_low)/atr:.1f} ATR extension",
                 target_1=round(entry + (entry - stop), 2), target_2=round(vwap, 2),
                 trail_type="vwap", trail_param=0.0, position_splits=(0.34, 0.33, 0.33))


def _detect_orb(s, period_minutes, name="ORB"):
    if s.n < 20 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    today = s.timestamps[-1].date()
    orb_bars = []
    for i in range(s.n):
        if s.timestamps[i].date() != today: continue
        bar_min = (s.timestamps[i].time().hour * 60 + s.timestamps[i].time().minute) - 570
        if 0 <= bar_min < period_minutes: orb_bars.append(i)
    if len(orb_bars) < period_minutes // 5: return None
    orb_high = max(s.highs[i] for i in orb_bars); orb_low = min(s.lows[i] for i in orb_bars)
    orb_range = orb_high - orb_low
    if orb_range < atr * 0.3 or orb_range > atr * 3.0: return None
    cur = s.closes[-1]; prev = s.closes[-2] if s.n >= 2 else cur
    nr7_bonus = 0.15 if _is_nr7(s) else 0.0
    tight_bonus = 0.05 if orb_range < atr * 0.8 else 0.0
    if cur > orb_high and prev <= orb_high:
        conf = 0.52 + nr7_bonus + tight_bonus
        entry = orb_high + _atr_offset(atr, 0.05); stop = orb_low - _atr_offset(atr, 0.10)
        return _make(s, name, Bias.LONG, entry, stop, round(entry + orb_range * 1.5, 2), conf,
                     f"{name} long: range={orb_range:.2f}",
                     target_1=round(entry + orb_range, 2), target_2=round(entry + orb_range * 1.5, 2),
                     trail_type="atr", trail_param=1.5)
    elif cur < orb_low and prev >= orb_low:
        conf = 0.52 + nr7_bonus + tight_bonus
        entry = orb_low - _atr_offset(atr, 0.05); stop = orb_high + _atr_offset(atr, 0.10)
        return _make(s, name, Bias.SHORT, entry, stop, round(entry - orb_range * 1.5, 2), conf,
                     f"{name} short: range={orb_range:.2f}",
                     target_1=round(entry - orb_range, 2), target_2=round(entry - orb_range * 1.5, 2),
                     trail_type="atr", trail_param=1.5)
    return None


def _detect_second_chance_scalp(s):
    _p = "Second Chance Scalp"
    if s.n < 20 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    res_level = None
    for level in s.sr_levels:
        if level.level_type in ("resistance", "both") and level.touches >= 2:
            level_bars = [i for i in range(s.n) if abs(s.highs[i] - level.price) < atr * 0.2]
            if level_bars and s.timestamps[level_bars[0]].date() < s.timestamps[-1].date():
                res_level = level; break
    if res_level is None: return None
    lp = res_level.price
    breakout_idx = None; breakout_high = 0
    for i in range(max(0, s.n - 40), s.n - 5):
        if s.closes[i] > lp + atr * 0.1:
            if breakout_idx is None: breakout_idx = i; breakout_high = s.highs[i]
            else: breakout_high = max(breakout_high, s.highs[i])
    if breakout_idx is None: return None
    tol = get_param(_p, "tolerance_atr", 0.3)
    pullback_found = False; pullback_low = None
    for i in range(breakout_idx + 1, s.n):
        if abs(s.lows[i] - lp) < atr * tol:
            pullback_found = True; pullback_low = s.lows[i]; break
    if not pullback_found or pullback_low is None: return None
    cur = s.bars[-1]
    if not (_is_green(cur) and cur.close > lp): return None
    if s.n - 1 - breakout_idx < 5: return None
    avg_vol = float(np.mean(s.volumes[max(0, breakout_idx-20):breakout_idx]))
    if avg_vol > 0 and s.volumes[breakout_idx] < avg_vol * get_param(_p, "vol_mult", 1.5): return None
    conf = 0.53; entry = cur.close
    stop = pullback_low - _atr_offset(atr, get_param(_p, "stop_atr_mult", 0.10))
    if entry - stop <= 0: return None
    ext = breakout_high - lp
    return _make(s, _p, Bias.LONG, entry, stop, round(breakout_high + ext * 0.5, 2), conf,
                 f"Second Chance off {lp:.2f}",
                 target_1=round(entry + (entry - stop), 2), target_2=round(breakout_high + ext * 0.5, 2),
                 trail_type="atr", trail_param=1.5)


def _detect_fashionably_late(s):
    _p = "Fashionably Late"
    if s.n < 20 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if not (time(10, 0) <= s.timestamps[-1].time() <= time(10, 45)): return None
    vwap = _vwap_today(s); ema9 = _compute_ema9(s)
    if vwap is None or ema9 is None or vwap <= 0 or ema9 <= 0: return None
    cross_idx = None
    for i in range(max(0, s.n - 3), s.n):
        if i < 2: continue
        if s.closes[i - 1] < vwap and s.closes[i] > vwap: cross_idx = i; break
    if cross_idx is None:
        if s.n >= 3 and s.closes[-2] < vwap and ema9 > vwap: cross_idx = s.n - 1
        else: return None
    ema_slope_min = get_param(_p, "ema_slope_min_atr", 0.1)
    if s.n >= 5 and (s.closes[-1] - s.closes[-3]) < atr * ema_slope_min: return None
    today = s.timestamps[-1].date()
    day_lows = [s.lows[i] for i in range(s.n) if s.timestamps[i].date() == today]
    if not day_lows: return None
    day_low = min(day_lows); mm = ema9 - day_low
    if mm <= atr * 0.5: return None
    conf = 0.57; entry = ema9
    stop = day_low - _atr_offset(atr, get_param(_p, "stop_atr_mult", 0.10))
    if entry - stop <= 0: return None
    t1_pct = get_param(_p, "t1_mm_pct", 0.75)
    return _make(s, _p, Bias.LONG, entry, stop, round(entry + mm, 2), conf,
                 f"Fashionably Late: EMA9 crossed VWAP at {ema9:.2f}",
                 target_1=round(entry + mm * t1_pct, 2), target_2=round(entry + mm, 2),
                 trail_type="ema9", trail_param=9.0)


def _detect_gap_give_and_go(s):
    if s.n < 10 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if not (time(9, 30) <= s.timestamps[-1].time() <= time(10, 15)): return None
    today = s.timestamps[-1].date(); yesterday = today - timedelta(days=1)
    yc = None; to = None
    for i in range(s.n):
        if s.timestamps[i].date() == yesterday: yc = s.closes[i]
        if s.timestamps[i].date() == today and to is None: to = s.opens[i]
    if yc is None or to is None: return None
    gap = to - yc; gap_pct = gap / yc
    if abs(gap_pct) < 0.015: return None
    day_bars_idx = [i for i in range(s.n) if s.timestamps[i].date() == today]
    if not day_bars_idx: return None
    avg_vol = float(np.mean(s.volumes[:max(1, len(day_bars_idx))]))
    if avg_vol > 0 and s.volumes[day_bars_idx[0]] < avg_vol * 2.0: return None
    day_bars = [s.bars[i] for i in day_bars_idx]
    if gap > 0:
        lowest = min(b.low for b in day_bars)
        if (to - lowest) / abs(gap) < 0.30: return None
        consol = day_bars[-min(8, len(day_bars)):]
        if len(consol) < 3: return None
        ch = max(b.high for b in consol); cl = min(b.low for b in consol)
        if s.closes[-1] <= ch: return None
        entry = ch + _atr_offset(atr, 0.05); stop = cl - _atr_offset(atr, 0.15)
        if entry - stop <= 0: return None
        return _make(s, "Gap Give & Go", Bias.LONG, entry, stop, round(to + abs(gap) * 0.5, 2), 0.53,
                     f"Gap G&G: {gap_pct:.1%} gap up",
                     target_1=round(entry + (entry - stop), 2), target_2=round(to + abs(gap) * 0.5, 2))
    else:
        highest = max(b.high for b in day_bars)
        if (highest - to) / abs(gap) < 0.30: return None
        consol = day_bars[-min(8, len(day_bars)):]
        if len(consol) < 3: return None
        ch = max(b.high for b in consol); cl = min(b.low for b in consol)
        if s.closes[-1] >= cl: return None
        entry = cl - _atr_offset(atr, 0.05); stop = ch + _atr_offset(atr, 0.15)
        if stop - entry <= 0: return None
        return _make(s, "Gap Give & Go", Bias.SHORT, entry, stop, round(to - abs(gap) * 0.5, 2), 0.53,
                     f"Gap G&G: {gap_pct:.1%} gap down",
                     target_1=round(entry - (stop - entry), 2), target_2=round(to - abs(gap) * 0.5, 2))


def _detect_tidal_wave(s):
    _p = "Tidal Wave"
    if s.n < 30: return None
    atr = s.current_atr
    if atr <= 0: return None
    sup_level = None
    for level in s.sr_levels:
        if level.level_type in ("support", "both") and level.touches >= get_param(_p, "min_touches", 3):
            sup_level = level; break
    if sup_level is None: return None
    sup = sup_level.price; tol = atr * 0.3
    bounces = []; i = 0
    while i < s.n:
        if abs(s.lows[i] - sup) < tol:
            bh = s.highs[i]; bs = i; j = i + 1
            while j < s.n and s.lows[j] > sup - tol: bh = max(bh, s.highs[j]); j += 1
            bounces.append({"touch_idx": i, "bounce_high": bh, "bounce_height": bh - sup, "bounce_bars": j - bs})
            i = j
        else: i += 1
    if len(bounces) < 3: return None
    heights = [b["bounce_height"] for b in bounces]
    if heights[-1] >= heights[0]: return None
    first_touch = bounces[0]["touch_idx"]
    if s.n - 1 - first_touch < get_param(_p, "min_span", 15): return None
    if s.closes[-1] >= sup: return None
    if not _volume_confirms_breakout(s, -1, get_param(_p, "vol_mult", 1.3)): return None
    touch_bonus = min(0.08, (len(bounces) - 3) * 0.03)
    dim_ratio = heights[-1] / heights[0] if heights[0] > 0 else 1.0
    conf = 0.55 + touch_bonus + (1 - dim_ratio) * 0.10
    entry = sup - _atr_offset(atr, 0.10); stop = bounces[0]["bounce_high"] + _atr_offset(atr, 0.20)
    avg_bounce = np.mean(heights)
    return _make(s, _p, Bias.SHORT, entry, stop, round(entry - avg_bounce, 2), conf,
                 f"Tidal Wave: {len(bounces)} touches, dim {dim_ratio:.0%}",
                 target_1=round(entry - avg_bounce * 0.5, 2), target_2=round(entry - avg_bounce, 2),
                 trail_type="atr", trail_param=2.0)


# ═══════════════════════════════════════════════════════════════
# QUANT — INTRADAY 5min
# ═══════════════════════════════════════════════════════════════

def _detect_mean_reversion(s):
    _p = "Mean Reversion"
    if s.n < 50 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if s._regime not in ("mean_reverting", "mixed"): return None
    closes = s.closes[-50:]; ma50 = float(np.mean(closes)); std50 = float(np.std(closes))
    if std50 <= 0: return None
    cur = s.closes[-1]; z = (cur - ma50) / std50
    z_thresh = get_param(_p, "z_threshold", 2.5)
    if abs(z) < z_thresh: return None
    vol_bonus = 0.05 if _volume_exhaustion(s, -1) else 0.0
    vwap = _vwap_today(s)
    if vwap is None or vwap <= 0: vwap = ma50
    conf = (0.58 + vol_bonus) * _regime_confidence_mult(s, "mean_reversion")
    _stop_m = get_param(_p, "stop_atr_mult", 1.0)
    if z < -z_thresh:
        entry = cur; stop = min(min(s.lows[-10:]), cur - atr) - _atr_offset(atr, _stop_m)
        if entry - stop <= 0: return None
        return _make(s, _p, Bias.LONG, entry, stop, round(vwap, 2), conf,
                     f"Mean Rev Long: z={z:.2f}",
                     target_1=round(entry + (vwap - entry) * 0.5, 2), target_2=round(vwap, 2),
                     trail_type="vwap", trail_param=0.0)
    elif z > z_thresh:
        entry = cur; stop = max(max(s.highs[-10:]), cur + atr) + _atr_offset(atr, _stop_m)
        if stop - entry <= 0: return None
        return _make(s, _p, Bias.SHORT, entry, stop, round(vwap, 2), conf,
                     f"Mean Rev Short: z={z:.2f}",
                     target_1=round(entry - (entry - vwap) * 0.5, 2), target_2=round(vwap, 2),
                     trail_type="vwap", trail_param=0.0)
    return None


def _detect_trend_pullback(s):
    _p = "Trend Pullback"
    if s.n < 30 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if s._regime not in ("trending_bull", "trending_bear"): return None
    mult = 2 / 22; ema21 = float(s.closes[0])
    for i in range(1, s.n): ema21 = s.closes[i] * mult + ema21 * (1 - mult)
    cur = s.closes[-1]; vwap = _vwap_today(s)
    if s.n >= 5:
        ep = float(s.closes[0])
        for i in range(1, s.n - 5): ep = s.closes[i] * mult + ep * (1 - mult)
        ema_slope = ema21 - ep
    else: ema_slope = 0
    _stop_m = get_param(_p, "stop_atr_mult", 0.10)
    if s._regime == "trending_bull":
        if ema_slope <= 0 or abs(cur - ema21) > atr * 0.3: return None
        if max(s.highs[-10:]) - min(s.lows[-5:]) < atr * 1.0: return None
        if vwap and cur < vwap: return None
        if sum(1 for j in range(-4, -1) if j + s.n >= 0 and _is_red(s.bars[j])) == 0: return None
        if not _is_green(s.bars[-1]): return None
        conf = 0.60 * _regime_confidence_mult(s, "momentum")
        entry = cur; stop = min(s.lows[-5:]) - _atr_offset(atr, _stop_m)
        if entry - stop <= 0: return None
        return _make(s, _p, Bias.LONG, entry, stop, round(max(max(s.highs[-20:]), entry + (entry-stop) * 2), 2), conf,
                     f"Trend Pullback Long: EMA21={ema21:.2f}",
                     target_1=round(entry + (entry - stop), 2), target_2=round(max(max(s.highs[-20:]), entry + (entry-stop) * 2), 2),
                     trail_type="ema9", trail_param=9.0)
    elif s._regime == "trending_bear":
        if ema_slope >= 0 or abs(cur - ema21) > atr * 0.3: return None
        if max(s.highs[-5:]) - min(s.lows[-10:]) < atr * 1.0: return None
        if vwap and cur > vwap: return None
        if sum(1 for j in range(-4, -1) if j + s.n >= 0 and _is_green(s.bars[j])) == 0: return None
        if not _is_red(s.bars[-1]): return None
        conf = 0.58 * _regime_confidence_mult(s, "momentum")
        entry = cur; stop = max(s.highs[-5:]) + _atr_offset(atr, _stop_m)
        if stop - entry <= 0: return None
        return _make(s, _p, Bias.SHORT, entry, stop, round(min(min(s.lows[-20:]), entry - (stop-entry) * 2), 2), conf,
                     f"Trend Pullback Short: EMA21={ema21:.2f}",
                     target_1=round(entry - (stop - entry), 2), target_2=round(min(min(s.lows[-20:]), entry - (stop-entry) * 2), 2),
                     trail_type="ema9", trail_param=9.0)
    return None


def _detect_gap_fade(s):
    if s.n < 10 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if s._regime in ("trending_bull", "trending_bear"): return None
    if not (time(9, 30) <= s.timestamps[-1].time() <= time(10, 30)): return None
    today = s.timestamps[-1].date(); yesterday = today - timedelta(days=1)
    yc = None; to = None
    for i in range(s.n):
        if s.timestamps[i].date() == yesterday: yc = s.closes[i]
        if s.timestamps[i].date() == today and to is None: to = s.opens[i]
    if yc is None or to is None: return None
    gap = to - yc; gap_pct = abs(gap) / yc
    if gap_pct < 0.02: return None
    day_bars = [i for i in range(s.n) if s.timestamps[i].date() == today]
    if day_bars:
        fbv = float(np.mean([s.volumes[i] for i in day_bars[:min(3, len(day_bars))]]))
        pav = float(np.mean(s.volumes[:max(1, day_bars[0])]))
        if pav > 0 and fbv > pav * 3.0: return None
    cur = s.closes[-1]
    if gap > 0:
        if cur >= to: return None
        entry = cur; stop = to + abs(gap) * 0.5
        if stop - entry <= 0: return None
        return _make(s, "Gap Fade", Bias.SHORT, entry, stop, round(yc, 2), 0.55,
                     f"Gap Fade Short: {gap_pct:.1%}",
                     target_1=round((to + yc) / 2, 2), target_2=round(yc, 2))
    else:
        if cur <= to: return None
        entry = cur; stop = to - abs(gap) * 0.5
        if entry - stop <= 0: return None
        return _make(s, "Gap Fade", Bias.LONG, entry, stop, round(yc, 2), 0.55,
                     f"Gap Fade Long: {gap_pct:.1%}",
                     target_1=round((to + yc) / 2, 2), target_2=round(yc, 2))


def _detect_vwap_reversion(s):
    _p = "VWAP Reversion"
    if s.n < 30 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if s._regime in ("trending_bull", "trending_bear"): return None
    if not (time(10, 30) <= s.timestamps[-1].time() <= time(14, 0)): return None
    vwap = _vwap_today(s)
    if vwap is None or vwap <= 0: return None
    cur = s.closes[-1]; today = s.timestamps[-1].date()
    day_closes = [s.closes[i] for i in range(s.n) if s.timestamps[i].date() == today]
    if len(day_closes) < 10: return None
    vwap_std = float(np.std([c - vwap for c in day_closes]))
    if vwap_std <= 0: return None
    sigma = (cur - vwap) / vwap_std; atr_dist = abs(cur - vwap) / atr
    atr_thresh = get_param(_p, "atr_threshold", 2.0)
    if abs(sigma) < 2.0 or atr_dist < atr_thresh: return None
    vol_bonus = 0.05 if _volume_exhaustion(s, -1) else 0.0
    conf = (0.58 + vol_bonus) * _regime_confidence_mult(s, "mean_reversion")
    _stop_m = get_param(_p, "stop_atr_mult", 0.20)
    if sigma < -2.0:
        entry = cur; stop = min(s.lows[-10:]) - _atr_offset(atr, _stop_m)
        if entry - stop <= 0: return None
        return _make(s, _p, Bias.LONG, entry, stop, round(vwap, 2), conf,
                     f"VWAP Rev Long: {sigma:.1f}σ",
                     target_1=round(vwap - vwap_std, 2), target_2=round(vwap, 2), trail_type="vwap")
    elif sigma > 2.0:
        entry = cur; stop = max(s.highs[-10:]) + _atr_offset(atr, _stop_m)
        if stop - entry <= 0: return None
        return _make(s, _p, Bias.SHORT, entry, stop, round(vwap, 2), conf,
                     f"VWAP Rev Short: {sigma:.1f}σ",
                     target_1=round(vwap + vwap_std, 2), target_2=round(vwap, 2), trail_type="vwap")
    return None


def _detect_overnight_gap_reversal(s):
    if s.n < 20 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if not (time(9, 45) <= s.timestamps[-1].time() <= time(10, 30)): return None
    today = s.timestamps[-1].date(); yesterday = today - timedelta(days=1)
    yc = None; to = None
    for i in range(s.n):
        if s.timestamps[i].date() == yesterday: yc = s.closes[i]
        if s.timestamps[i].date() == today and to is None: to = s.opens[i]
    if yc is None or to is None or yc <= 0: return None
    gap_pct = (to - yc) / yc
    if abs(gap_pct) < 0.005 or abs(gap_pct) > 0.03: return None
    cur = s.closes[-1]; gap_size = to - yc
    if gap_pct > 0:
        if to - cur < abs(gap_size) * 0.30: return None
        day_bars = [i for i in range(s.n) if s.timestamps[i].date() == today]
        if day_bars:
            if float(np.mean(s.volumes[max(0, day_bars[0]-20):day_bars[0]])) > 0 and s.volumes[day_bars[0]] > float(np.mean(s.volumes[max(0, day_bars[0]-20):day_bars[0]])) * 3.0:
                return None
        entry = cur; stop = to + _atr_offset(atr, 0.10)
        if stop - entry <= 0: return None
        return _make(s, "Gap Reversal Short", Bias.SHORT, entry, stop, round(yc, 2), 0.55,
                     f"Gap Reversal Short: {gap_pct:+.1%}",
                     target_1=round((entry + yc) / 2, 2), target_2=round(yc, 2))
    else:
        if cur - to < abs(gap_size) * 0.30: return None
        entry = cur; stop = to - _atr_offset(atr, 0.10)
        if entry - stop <= 0: return None
        return _make(s, "Gap Reversal Long", Bias.LONG, entry, stop, round(yc, 2), 0.55,
                     f"Gap Reversal Long: {gap_pct:+.1%}",
                     target_1=round((entry + yc) / 2, 2), target_2=round(yc, 2))


def _detect_opening_drive(s):
    _p = "Opening Drive Long"
    if s.n < 20 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if not (time(10, 0) <= s.timestamps[-1].time() <= time(10, 15)): return None
    today = s.timestamps[-1].date()
    today_bars = [i for i in range(s.n) if s.timestamps[i].date() == today]
    if len(today_bars) < 6: return None
    first_6 = today_bars[:6]; op = s.opens[first_6[0]]; cl = s.closes[first_6[-1]]
    drive = cl - op; min_drive = get_param(_p, "min_drive_atr", 0.5)
    if abs(drive) < atr * min_drive: return None
    vol_mult = get_param(_p, "vol_mult", 1.3)
    f30_vol = sum(s.volumes[i] for i in first_6)
    avg_bar_vol = float(np.mean(s.volumes[max(0, first_6[0]-30):first_6[0]])) if first_6[0] > 30 else 0
    if avg_bar_vol > 0 and f30_vol < avg_bar_vol * 6 * vol_mult: return None
    cur = s.closes[-1]; conf = 0.56; _stop_m = get_param(_p, "stop_atr_mult", 0.10)
    if drive > 0:
        entry = cur; stop = min(s.lows[i] for i in first_6) - _atr_offset(atr, _stop_m)
        if entry - stop <= 0: return None
        return _make(s, "Opening Drive Long", Bias.LONG, entry, stop, round(entry + atr * 2, 2), conf,
                     f"Opening Drive Long: +${drive:.2f}",
                     target_1=round(entry + atr, 2), target_2=round(entry + atr * 2, 2))
    else:
        entry = cur; stop = max(s.highs[i] for i in first_6) + _atr_offset(atr, _stop_m)
        if stop - entry <= 0: return None
        return _make(s, "Opening Drive Short", Bias.SHORT, entry, stop, round(entry - atr * 2, 2), conf,
                     f"Opening Drive Short: -${abs(drive):.2f}",
                     target_1=round(entry - atr, 2), target_2=round(entry - atr * 2, 2))


def _detect_power_hour(s):
    _p = "Power Hour Long"
    if s.n < 30 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if not (time(15, 0) <= s.timestamps[-1].time() <= time(15, 30)): return None
    cur = s.closes[-1]; vwap = _vwap_today(s)
    if vwap is None or vwap <= 0: return None
    pm_bars = [i for i in range(s.n) if s.timestamps[i].time() >= time(13, 0)]
    if len(pm_bars) < 10: return None
    pct_above = sum(1 for i in pm_bars if s.closes[i] > vwap) / len(pm_bars)
    vol_mult = get_param(_p, "vol_mult", 1.2)
    recent_vol = float(np.mean(s.volumes[-6:]))
    day_avg = float(np.mean(s.volumes[-78:])) if s.n >= 78 else float(np.mean(s.volumes))
    if day_avg > 0 and recent_vol < day_avg * vol_mult: return None
    vwap_thresh_long = get_param(_p, "vwap_pct_threshold", 0.70)
    vwap_thresh_short = get_param("Power Hour Short", "vwap_pct_threshold", 0.30)
    if pct_above > vwap_thresh_long:
        entry = cur; stop = vwap - _atr_offset(atr, 0.10)
        if entry - stop <= 0: return None
        return _make(s, "Power Hour Long", Bias.LONG, entry, stop, round(entry + atr, 2), 0.55,
                     f"Power Hour Long: {pct_above:.0%} PM above VWAP",
                     target_1=round(entry + atr * 0.5, 2), target_2=round(entry + atr, 2))
    elif pct_above < vwap_thresh_short:
        entry = cur; stop = vwap + _atr_offset(atr, 0.10)
        if stop - entry <= 0: return None
        return _make(s, "Power Hour Short", Bias.SHORT, entry, stop, round(entry - atr, 2), 0.55,
                     f"Power Hour Short: {pct_above:.0%} PM above VWAP",
                     target_1=round(entry - atr * 0.5, 2), target_2=round(entry - atr, 2))
    return None


def _detect_volume_climax(s):
    _p = "Volume Climax Long"
    if s.n < 30 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if not (time(9, 45) <= s.timestamps[-1].time() <= time(15, 30)): return None
    avg_vol = float(np.mean(s.volumes[-30:]))
    if avg_vol <= 0: return None
    vol_ratio = s.volumes[-1] / avg_vol
    vol_thresh = get_param(_p, "vol_threshold", 4.0)
    if vol_ratio < vol_thresh: return None
    cur = s.closes[-1]
    high_20 = max(s.highs[-20:]); low_20 = min(s.lows[-20:])
    at_high = s.highs[-1] >= high_20 * 0.998; at_low = s.lows[-1] <= low_20 * 1.002
    if not at_high and not at_low: return None
    conf = 0.56 + min(0.10, (vol_ratio - 3.0) * 0.02)
    _stop_m = get_param(_p, "stop_atr_mult", 0.10)
    if at_high and s.closes[-1] < s.opens[-1]:
        entry = cur; stop = s.highs[-1] + _atr_offset(atr, _stop_m)
        if stop - entry <= 0: return None
        return _make(s, "Volume Climax Short", Bias.SHORT, entry, stop, round(entry - atr * 2, 2), conf,
                     f"Vol Climax Short: {vol_ratio:.1f}x",
                     target_1=round(entry - atr, 2), target_2=round(entry - atr * 2, 2))
    elif at_low and s.closes[-1] > s.opens[-1]:
        entry = cur; stop = s.lows[-1] - _atr_offset(atr, _stop_m)
        if entry - stop <= 0: return None
        return _make(s, "Volume Climax Long", Bias.LONG, entry, stop, round(entry + atr * 2, 2), conf,
                     f"Vol Climax Long: {vol_ratio:.1f}x",
                     target_1=round(entry + atr, 2), target_2=round(entry + atr * 2, 2))
    return None


def _detect_vwap_trend(s):
    _p = "VWAP Trend Long"
    if s.n < 40 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if not (time(10, 30) <= s.timestamps[-1].time() <= time(14, 30)): return None
    vwap = _vwap_today(s)
    if vwap is None or vwap <= 0: return None
    cur = s.closes[-1]; today = s.timestamps[-1].date()
    today_idx = [i for i in range(s.n) if s.timestamps[i].date() == today]
    if len(today_idx) < 24: return None
    pct_above = sum(1 for i in today_idx if s.closes[i] > vwap) / len(today_idx)
    prox = get_param(_p, "proximity_atr", 0.4)
    min_tests = get_param(_p, "min_vwap_tests", 2)
    vwap_tests = sum(1 for i in today_idx if abs(s.lows[i] - vwap) < atr * 0.3 and s.closes[i] > vwap) + \
                 sum(1 for i in today_idx if abs(s.highs[i] - vwap) < atr * 0.3 and s.closes[i] < vwap)
    if vwap_tests < min_tests or abs(cur - vwap) > atr * prox: return None
    pct_thresh_long = get_param(_p, "pct_above_threshold", 0.75)
    pct_thresh_short = get_param("VWAP Trend Short", "pct_above_threshold", 0.25)
    if pct_above > pct_thresh_long and cur > vwap:
        entry = cur; stop = vwap - _atr_offset(atr, 0.15)
        if entry - stop <= 0: return None
        return _make(s, "VWAP Trend Long", Bias.LONG, entry, stop, round(entry + atr * 2, 2), 0.57,
                     f"VWAP Trend Long: {pct_above:.0%} above",
                     target_1=round(entry + atr, 2), target_2=round(entry + atr * 2, 2), trail_type="atr", trail_param=1.0)
    elif pct_above < pct_thresh_short and cur < vwap:
        entry = cur; stop = vwap + _atr_offset(atr, 0.15)
        if stop - entry <= 0: return None
        return _make(s, "VWAP Trend Short", Bias.SHORT, entry, stop, round(entry - atr * 2, 2), 0.57,
                     f"VWAP Trend Short: {1-pct_above:.0%} below",
                     target_1=round(entry - atr, 2), target_2=round(entry - atr * 2, 2), trail_type="atr", trail_param=1.0)
    return None


def _detect_rsi_divergence(s):
    _p = "RSI Divergence Long"
    if s.n < 30 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if not (time(9, 45) <= s.timestamps[-1].time() <= time(15, 30)): return None
    closes = np.array(s.closes[-20:], dtype=np.float64); deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0); losses = np.where(deltas < 0, -deltas, 0)
    _period = get_param(_p, "rsi_period", 14)
    ag = float(np.mean(gains[:_period])); al = float(np.mean(losses[:_period]))
    if al == 0: return None
    rsi_prev = 100 - (100 / (1 + ag / al))
    for i in range(_period, len(deltas)):
        ag = (ag * (_period-1) + gains[i]) / _period; al = (al * (_period-1) + losses[i]) / _period
    if al == 0: return None
    rsi_now = 100 - (100 / (1 + ag / al)); cur = s.closes[-1]
    avg_vol = float(np.mean(s.volumes[-10:]))
    if avg_vol > 0 and s.volumes[-1] > avg_vol * 1.2: return None
    price_hh = s.highs[-1] > max(s.highs[-10:-1])
    rsi_thresh_short = get_param("RSI Divergence Short", "rsi_threshold", 60)
    rsi_lh = rsi_now < rsi_prev and rsi_now > rsi_thresh_short
    price_ll = s.lows[-1] < min(s.lows[-10:-1])
    rsi_thresh_long = get_param(_p, "rsi_threshold", 40)
    rsi_hl = rsi_now > rsi_prev and rsi_now < rsi_thresh_long
    _stop_m = get_param(_p, "stop_atr_mult", 0.10)
    if price_hh and rsi_lh:
        entry = cur; stop = s.highs[-1] + _atr_offset(atr, _stop_m)
        if stop - entry <= 0: return None
        return _make(s, "RSI Divergence Short", Bias.SHORT, entry, stop, round(entry - atr * 2, 2), 0.56,
                     f"RSI Div Short: RSI={rsi_now:.0f}",
                     target_1=round(entry - atr, 2), target_2=round(entry - atr * 2, 2))
    elif price_ll and rsi_hl:
        entry = cur; stop = s.lows[-1] - _atr_offset(atr, _stop_m)
        if entry - stop <= 0: return None
        return _make(s, "RSI Divergence Long", Bias.LONG, entry, stop, round(entry + atr * 2, 2), 0.56,
                     f"RSI Div Long: RSI={rsi_now:.0f}",
                     target_1=round(entry + atr, 2), target_2=round(entry + atr * 2, 2))
    return None


def _detect_midday_reversal(s):
    _p = "Midday Reversal Long"
    if s.n < 30 or s.timeframe != "5min": return None
    atr = s.current_atr
    if atr <= 0: return None
    if not (time(11, 30) <= s.timestamps[-1].time() <= time(12, 30)): return None
    today = s.timestamps[-1].date()
    morning = [i for i in range(s.n) if s.timestamps[i].date() == today and s.timestamps[i].time() < time(11, 30)]
    if len(morning) < 10: return None
    op = s.opens[morning[0]]; mc = s.closes[morning[-1]]; mm = mc - op; cur = s.closes[-1]
    min_atr = get_param(_p, "min_morning_atr", 1.0)
    if abs(mm) < atr * min_atr: return None
    retrace_pct = get_param(_p, "retrace_pct", 0.30)
    if mm > 0:
        if (mc - cur) < mm * retrace_pct: return None
        entry = cur; stop = max(s.highs[i] for i in morning) + _atr_offset(atr, 0.05)
        if stop - entry <= 0: return None
        return _make(s, "Midday Reversal Short", Bias.SHORT, entry, stop, round(entry - atr * 2, 2), 0.55,
                     f"Midday Rev Short: morning +${mm:.2f}",
                     target_1=round(entry - atr, 2), target_2=round(entry - atr * 2, 2))
    else:
        if (cur - mc) < abs(mm) * retrace_pct: return None
        entry = cur; stop = min(s.lows[i] for i in morning) - _atr_offset(atr, 0.05)
        if entry - stop <= 0: return None
        return _make(s, "Midday Reversal Long", Bias.LONG, entry, stop, round(entry + atr * 2, 2), 0.55,
                     f"Midday Rev Long: morning -${abs(mm):.2f}",
                     target_1=round(entry + atr, 2), target_2=round(entry + atr * 2, 2))


# ═══════════════════════════════════════════════════════════════
# QUANT — 15min / 1h
# ═══════════════════════════════════════════════════════════════

def _detect_keltner_breakout(s):
    _p = "Keltner Breakout Long"
    if s.n < 30 or s.timeframe not in ("15min", "1h"): return None
    atr = s.current_atr
    if atr <= 0: return None
    _ep = get_param(_p, "ema_period", 20); _am = get_param(_p, "atr_mult", 2.0)
    ema20 = float(s.closes[0]); mult = 2.0 / (_ep + 1)
    for i in range(1, s.n): ema20 = s.closes[i] * mult + ema20 * (1 - mult)
    upper = ema20 + _am * atr; lower = ema20 - _am * atr; cur = s.closes[-1]
    if not any(lower < s.closes[i] < upper for i in range(s.n - 5, s.n - 1)): return None
    sq_thresh = get_param(_p, "squeeze_threshold", 0.80)
    if s.n >= 25:
        rr = [s.highs[i] - s.lows[i] for i in range(s.n - 5, s.n)]
        lr = [s.highs[i] - s.lows[i] for i in range(s.n - 20, s.n - 5)]
        if float(np.mean(lr)) > 0 and float(np.mean(rr)) > float(np.mean(lr)) * sq_thresh: return None
    vol_mult = get_param(_p, "vol_mult", 1.5)
    avg_vol = float(np.mean(s.volumes[-20:]))
    if avg_vol > 0 and s.volumes[-1] < avg_vol * vol_mult: return None
    conf = 0.58
    if cur > upper:
        entry = cur; stop = ema20 - _atr_offset(atr, 0.10)
        if entry - stop <= 0: return None
        return _make(s, "Keltner Breakout Long", Bias.LONG, entry, stop, round(entry + atr * 3, 2), conf,
                     f"Keltner BO Long: above EMA+{_am}ATR",
                     target_1=round(entry + atr * 1.5, 2), target_2=round(entry + atr * 3, 2))
    elif cur < lower:
        entry = cur; stop = ema20 + _atr_offset(atr, 0.10)
        if stop - entry <= 0: return None
        return _make(s, "Keltner Breakout Short", Bias.SHORT, entry, stop, round(entry - atr * 3, 2), conf,
                     f"Keltner BO Short: below EMA-{_am}ATR",
                     target_1=round(entry - atr * 1.5, 2), target_2=round(entry - atr * 3, 2))
    return None


def _detect_macd_reversal(s):
    _p = "MACD Turn Long"
    if s.n < 35 or s.timeframe not in ("15min", "1h"): return None
    atr = s.current_atr
    if atr <= 0: return None
    _fast = get_param(_p, "fast_period", 12); _slow = get_param(_p, "slow_period", 26)
    def ema_val(data, period):
        k = 2.0 / (period + 1); r = float(data[0])
        for i in range(1, len(data)): r = data[i] * k + r * (1 - k)
        return r
    hist = []
    for j in range(5):
        idx = s.n - 5 + j; hist.append(ema_val(s.closes[:idx+1], _fast) - ema_val(s.closes[:idx+1], _slow))
    if len(hist) < 5: return None
    p2, p1, c = hist[-3], hist[-2], hist[-1]; cur = s.closes[-1]
    _stop_m = get_param(_p, "stop_atr_mult", 0.10)
    if p2 > p1 and c > p1 and c > 0 and p1 < 0:
        entry = cur; stop = min(s.lows[-5:]) - _atr_offset(atr, _stop_m)
        if entry - stop <= 0: return None
        return _make(s, "MACD Turn Long", Bias.LONG, entry, stop, round(entry + atr * 3, 2), 0.55,
                     "MACD Turn Long: zero cross",
                     target_1=round(entry + atr * 1.5, 2), target_2=round(entry + atr * 3, 2))
    elif p2 < p1 and c < p1 and c < 0 and p1 > 0:
        entry = cur; stop = max(s.highs[-5:]) + _atr_offset(atr, _stop_m)
        if stop - entry <= 0: return None
        return _make(s, "MACD Turn Short", Bias.SHORT, entry, stop, round(entry - atr * 3, 2), 0.55,
                     "MACD Turn Short: zero cross",
                     target_1=round(entry - atr * 1.5, 2), target_2=round(entry - atr * 3, 2))
    return None


def _detect_vol_price_divergence(s):
    _p = "VP Divergence Long"
    if s.n < 20 or s.timeframe not in ("15min", "1h"): return None
    atr = s.current_atr
    if atr <= 0: return None
    cur = s.closes[-1]; p10 = s.closes[-11]
    price_trend = (cur - p10) / p10 if p10 > 0 else 0
    vfh = float(np.mean(s.volumes[-10:-5])); vsh = float(np.mean(s.volumes[-5:]))
    if vfh <= 0: return None
    vd_thresh = get_param(_p, "vol_decline_pct", 0.60)
    pt_thresh = get_param(_p, "price_trend_min", 0.02)
    if vsh >= vfh * vd_thresh or abs(price_trend) < pt_thresh: return None
    _stop_m = get_param(_p, "stop_atr_mult", 0.10)
    if price_trend > pt_thresh:
        if cur < max(s.highs[-10:]) * 0.995: return None
        entry = cur; stop = max(s.highs[-5:]) + _atr_offset(atr, _stop_m)
        if stop - entry <= 0: return None
        return _make(s, "VP Divergence Short", Bias.SHORT, entry, stop, round(entry - atr * 2, 2), 0.55,
                     f"VP Div Short: price +{price_trend:.1%}, vol declining",
                     target_1=round(entry - atr, 2), target_2=round(entry - atr * 2, 2))
    elif price_trend < -pt_thresh:
        if cur > min(s.lows[-10:]) * 1.005: return None
        entry = cur; stop = min(s.lows[-5:]) - _atr_offset(atr, _stop_m)
        if entry - stop <= 0: return None
        return _make(s, "VP Divergence Long", Bias.LONG, entry, stop, round(entry + atr * 2, 2), 0.55,
                     f"VP Div Long: price {price_trend:.1%}, vol declining",
                     target_1=round(entry + atr, 2), target_2=round(entry + atr * 2, 2))
    return None


# ═══════════════════════════════════════════════════════════════
# QUANT — DAILY
# ═══════════════════════════════════════════════════════════════

def _detect_momentum_breakout(s):
    _p = "Momentum Breakout"
    if s.n < 50 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    cur = s.closes[-1]
    _lb = get_param(_p, "lookback_period", 20)
    high_lb = max(s.highs[-_lb-1:-1])
    if cur <= high_lb: return None
    sma50 = float(np.mean(s.closes[-50:]))
    if cur < sma50: return None
    avg_vol = float(np.mean(s.volumes[-20:]))
    vm = get_param(_p, "vol_mult", 1.5)
    if avg_vol > 0 and s.volumes[-1] < avg_vol * vm: return None
    entry = cur; _sm = get_param(_p, "stop_atr_mult", 2.0)
    stop = max(min(s.lows[-10:]), entry - atr * _sm) - _atr_offset(atr, 0.10)
    if entry - stop <= 0: return None
    return _make(s, _p, Bias.LONG, entry, stop, round(entry + (entry - stop) * 2, 2), 0.58,
                 f"Momentum BO: new {_lb}d high",
                 target_1=round(entry + (entry - stop), 2), target_2=round(entry + (entry - stop) * 2, 2))


def _detect_vol_compression_breakout(s):
    if s.n < 30 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    atr_vals = wilder_atr(s.highs[-30:], s.lows[-30:], s.closes[-30:], period=14)
    if len(atr_vals) < 20 or np.isnan(atr_vals[-1]): return None
    ca = float(atr_vals[-1]); aa = float(np.mean(atr_vals[-20:]))
    if aa <= 0 or ca / aa > 0.6: return None
    if sum(1 for v in atr_vals[-10:] if not np.isnan(v) and v / aa < 0.7) < 5: return None
    avg_vol = float(np.mean(s.volumes[-20:]))
    vb = 0.05 if (avg_vol > 0 and s.volumes[-1] > avg_vol * 1.3) else 0.0
    cur = s.closes[-1]; sma20 = float(np.mean(s.closes[-20:])); conf = 0.57 + vb
    if cur > sma20:
        entry = cur; stop = sma20 - atr
        if entry - stop <= 0: return None
        return _make(s, "Vol Compression Breakout", Bias.LONG, entry, stop, round(entry + (entry-stop)*2, 2), conf,
                     f"Vol Squeeze BO Long",
                     target_1=round(entry + (entry-stop), 2), target_2=round(entry + (entry-stop)*2, 2))
    else:
        entry = cur; stop = sma20 + atr
        if stop - entry <= 0: return None
        return _make(s, "Vol Compression Breakout", Bias.SHORT, entry, stop, round(entry - (stop-entry)*2, 2), conf,
                     f"Vol Squeeze BO Short",
                     target_1=round(entry - (stop-entry), 2), target_2=round(entry - (stop-entry)*2, 2))


def _detect_range_expansion(s):
    _p = "Range Expansion"
    if s.n < 20 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    today_range = s.highs[-1] - s.lows[-1]
    avg_range = float(np.mean([s.highs[i] - s.lows[i] for i in range(-11, -1)]))
    if avg_range <= 0 or today_range < avg_range * 2.0: return None
    avg_vol = float(np.mean(s.volumes[-20:]))
    if avg_vol > 0 and s.volumes[-1] < avg_vol * 1.3: return None
    cur = s.closes[-1]; sma20 = float(np.mean(s.closes[-20:]))
    if cur > s.opens[-1] and cur > sma20:
        entry = cur; stop = max(min(s.lows[-10:]), entry - atr * 2) - _atr_offset(atr, 0.10)
        if entry - stop <= 0: return None
        return _make(s, _p, Bias.LONG, entry, stop, round(entry + (entry-stop)*2, 2), 0.54,
                     f"Range Expansion Long: {today_range/avg_range:.1f}x",
                     target_1=round(entry + (entry-stop), 2), target_2=round(entry + (entry-stop)*2, 2))
    elif cur < s.opens[-1] and cur < sma20:
        entry = cur; stop = min(max(s.highs[-10:]), entry + atr * 2) + _atr_offset(atr, 0.10)
        if stop - entry <= 0: return None
        return _make(s, _p, Bias.SHORT, entry, stop, round(entry - (stop-entry)*2, 2), 0.54,
                     f"Range Expansion Short: {today_range/avg_range:.1f}x",
                     target_1=round(entry - (stop-entry), 2), target_2=round(entry - (stop-entry)*2, 2))
    return None


def _detect_volume_breakout(s):
    _p = "Volume Breakout"
    if s.n < 20 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    avg_vol = float(np.mean(s.volumes[-20:]))
    vt = get_param(_p, "vol_threshold", 3.0)
    if avg_vol <= 0 or s.volumes[-1] < avg_vol * vt: return None
    cur = s.closes[-1]; sma20 = float(np.mean(s.closes[-20:]))
    _sm = get_param(_p, "stop_atr_mult", 2.0)
    if cur > s.opens[-1]:
        conf = 0.56 if cur > sma20 else 0.51
        entry = cur; stop = max(min(s.lows[-10:]), entry - atr * _sm) - _atr_offset(atr, 0.10)
        if entry - stop <= 0: return None
        return _make(s, _p, Bias.LONG, entry, stop, round(entry + (entry-stop)*2, 2), conf,
                     f"Volume BO Long: {s.volumes[-1]/avg_vol:.1f}x",
                     target_1=round(entry + (entry-stop), 2), target_2=round(entry + (entry-stop)*2, 2))
    elif cur < s.opens[-1]:
        conf = 0.56 if cur < sma20 else 0.51
        entry = cur; stop = min(max(s.highs[-10:]), entry + atr * _sm) + _atr_offset(atr, 0.10)
        if stop - entry <= 0: return None
        return _make(s, _p, Bias.SHORT, entry, stop, round(entry - (stop-entry)*2, 2), conf,
                     f"Volume BO Short: {s.volumes[-1]/avg_vol:.1f}x",
                     target_1=round(entry - (stop-entry), 2), target_2=round(entry - (stop-entry)*2, 2))
    return None


def _detect_donchian_breakout(s):
    _p = "Donchian Breakout"
    if s.n < 50 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    sq_thresh = get_param(_p, "squeeze_threshold", 0.85)
    if s.n >= 25:
        rr = [s.highs[i]-s.lows[i] for i in range(s.n-5, s.n)]
        lr = [s.highs[i]-s.lows[i] for i in range(s.n-20, s.n-5)]
        if float(np.mean(lr)) > 0 and float(np.mean(rr)) > float(np.mean(lr)) * sq_thresh: return None
    _cp = get_param(_p, "channel_period", 20)
    cur = s.closes[-1]; h20 = max(s.highs[-_cp-1:-1]); l20 = min(s.lows[-_cp-1:-1])
    sma50 = float(np.mean(s.closes[-50:]))
    avg_vol = float(np.mean(s.volumes[-20:]))
    vm = get_param(_p, "vol_mult", 1.5)
    vol_ok = avg_vol > 0 and s.volumes[-1] >= avg_vol * vm
    _sm = get_param(_p, "stop_atr_mult", 2.0)
    if cur > h20:
        conf = 0.55 if cur > sma50 else 0.48
        if not vol_ok: conf -= 0.05
        entry = cur; stop = max(min(s.lows[-10:]), entry - atr * _sm) - _atr_offset(atr, 0.10)
        if entry - stop <= 0: return None
        return _make(s, _p, Bias.LONG, entry, stop, round(entry + (entry-stop)*2, 2), conf,
                     f"Donchian BO Long", target_1=round(entry + (entry-stop), 2), target_2=round(entry + (entry-stop)*2, 2))
    elif cur < l20:
        conf = 0.55 if cur < sma50 else 0.48
        if not vol_ok: conf -= 0.05
        entry = cur; stop = min(max(s.highs[-10:]), entry + atr * _sm) + _atr_offset(atr, 0.10)
        if stop - entry <= 0: return None
        return _make(s, _p, Bias.SHORT, entry, stop, round(entry - (stop-entry)*2, 2), conf,
                     f"Donchian BO Short", target_1=round(entry - (stop-entry), 2), target_2=round(entry - (stop-entry)*2, 2))
    return None


def _detect_time_series_momentum(s):
    _p = "TS Momentum Long"
    if s.n < 126 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    cur = s.closes[-1]
    _lb = get_param(_p, "lookback_days", 126); _skip = get_param(_p, "skip_days", 21)
    if s.n < _lb: return None
    mom_ret = (s.closes[-_skip] - s.closes[-_lb]) / s.closes[-_lb]
    _thresh = get_param(_p, "mom_threshold", 0.05)
    if abs(mom_ret) < _thresh: return None
    rets = np.diff(s.closes[-61:]) / s.closes[-61:-1]
    vol_60 = float(np.std(rets)) * np.sqrt(252)
    if vol_60 <= 0: return None
    conf = 0.55 + min(abs(mom_ret) / 0.30, 1.0) * 0.10
    _sm = get_param(_p, "stop_atr_mult", 2.5)
    if mom_ret > _thresh:
        entry = cur; stop = cur - atr * _sm
        if entry - stop <= 0: return None
        return _make(s, "TS Momentum Long", Bias.LONG, entry, stop, round(entry + atr * 4, 2), conf,
                     f"TS Mom Long: ret={mom_ret:+.1%}",
                     target_1=round(entry + atr * 2, 2), target_2=round(entry + atr * 4, 2),
                     trail_type="atr", trail_param=2.5, position_splits=(0.30, 0.30, 0.40))
    elif mom_ret < -_thresh:
        entry = cur; stop = cur + atr * _sm
        if stop - entry <= 0: return None
        return _make(s, "TS Momentum Short", Bias.SHORT, entry, stop, round(entry - atr * 4, 2), conf,
                     f"TS Mom Short: ret={mom_ret:+.1%}",
                     target_1=round(entry - atr * 2, 2), target_2=round(entry - atr * 4, 2),
                     trail_type="atr", trail_param=2.5, position_splits=(0.30, 0.30, 0.40))
    return None


def _detect_multi_tf_trend(s):
    _p = "Multi-TF Trend Long"
    if s.n < 200 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    def _ema(data, period):
        m = 2.0 / (period + 1); r = float(data[0])
        for i in range(1, len(data)): r = data[i] * m + r * (1 - m)
        return r
    _ef = get_param(_p, "ema_fast", 20); _em = get_param(_p, "ema_mid", 50)
    _es = get_param(_p, "ema_slow", 100); _ea = get_param(_p, "ema_anchor", 200)
    e1 = _ema(s.closes, _ef); e2 = _ema(s.closes, _em)
    e3 = _ema(s.closes, _es); e4 = _ema(s.closes, _ea)
    s1 = 1.0 if e1 > e2 else -1.0; s2 = 1.0 if e2 > e3 else -1.0; s3 = 1.0 if e3 > e4 else -1.0
    consensus = (s1 + s2 + s3) / 3.0
    if abs(consensus) < 0.5: return None
    avg_vol = float(np.mean(s.volumes[-20:]))
    if avg_vol > 0 and s.volumes[-1] < avg_vol * 0.8: return None
    conf = 0.55 + abs(consensus) * 0.10; cur = s.closes[-1]
    _sm = get_param(_p, "stop_atr_mult", 3.0)
    if consensus > 0:
        entry = cur; stop = cur - atr * _sm
        if entry - stop <= 0: return None
        return _make(s, "Multi-TF Trend Long", Bias.LONG, entry, stop, round(entry + atr * 5, 2), conf,
                     f"Multi-TF Long: consensus={consensus:+.2f}",
                     target_1=round(entry + atr * 2, 2), target_2=round(entry + atr * 5, 2),
                     trail_type="atr", trail_param=3.0, position_splits=(0.25, 0.25, 0.50))
    else:
        entry = cur; stop = cur + atr * _sm
        if stop - entry <= 0: return None
        return _make(s, "Multi-TF Trend Short", Bias.SHORT, entry, stop, round(entry - atr * 5, 2), conf,
                     f"Multi-TF Short: consensus={consensus:+.2f}",
                     target_1=round(entry - atr * 2, 2), target_2=round(entry - atr * 5, 2),
                     trail_type="atr", trail_param=3.0, position_splits=(0.25, 0.25, 0.50))


def _detect_short_term_reversal(s):
    _p = "ST Reversal Long"
    if s.n < 60 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    cur = s.closes[-1]; p21 = s.closes[-22]; ret = (cur - p21) / p21
    monthly_returns = []
    for i in range(22, min(s.n, 252), 21):
        r = (s.closes[-i] - s.closes[-i-21]) / s.closes[-i-21] if s.closes[-i-21] > 0 else 0
        monthly_returns.append(r)
    if len(monthly_returns) < 3: return None
    std_ret = float(np.std(monthly_returns))
    if std_ret <= 0: return None
    z = (ret - float(np.mean(monthly_returns))) / std_ret
    _zt = get_param(_p, "z_threshold", 2.0)
    if abs(z) < _zt: return None
    vol_bonus = 0.05 if _volume_exhaustion(s, -1) else 0.0
    conf = 0.55 + vol_bonus + min(0.10, (abs(z) - _zt) * 0.05)
    _sm = get_param(_p, "stop_atr_mult", 1.5)
    if z < -_zt:
        entry = cur; stop = cur - atr * _sm
        if entry - stop <= 0: return None
        return _make(s, "ST Reversal Long", Bias.LONG, entry, stop, round(p21, 2), conf,
                     f"ST Rev Long: z={z:.1f}",
                     target_1=round(entry + abs(cur-p21)*0.5, 2), target_2=round(p21, 2),
                     position_splits=(0.50, 0.30, 0.20))
    elif z > _zt:
        entry = cur; stop = cur + atr * _sm
        if stop - entry <= 0: return None
        return _make(s, "ST Reversal Short", Bias.SHORT, entry, stop, round(p21, 2), conf,
                     f"ST Rev Short: z={z:.1f}",
                     target_1=round(entry - abs(cur-p21)*0.5, 2), target_2=round(p21, 2),
                     position_splits=(0.50, 0.30, 0.20))
    return None


def _detect_low_vol_anomaly(s):
    _p = "Low Vol Long"
    if s.n < 252 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    cur = s.closes[-1]
    v60 = float(np.std(np.diff(s.closes[-61:]) / s.closes[-61:-1])) * np.sqrt(252)
    v252 = float(np.std(np.diff(s.closes[-253:]) / s.closes[-253:-1])) * np.sqrt(252)
    if v252 <= 0 or v60 <= 0: return None
    max_vol = get_param(_p, "max_vol", 0.75)
    if v60 / v252 > max_vol: return None
    sma50 = float(np.mean(s.closes[-50:])); sma200 = float(np.mean(s.closes[-200:]))
    if cur < sma50 or cur < sma200: return None
    conf = 0.58 + (max_vol - v60/v252) * 0.15
    _sm = get_param(_p, "stop_atr_mult", 2.0)
    entry = cur; stop = cur - atr * _sm
    if entry - stop <= 0: return None
    return _make(s, _p, Bias.LONG, entry, stop, round(entry + atr * 4, 2), conf,
                 f"Low Vol: ratio={v60/v252:.0%}",
                 target_1=round(entry + atr * 2, 2), target_2=round(entry + atr * 4, 2),
                 position_splits=(0.30, 0.30, 0.40))


def _detect_turtle_breakout(s):
    _p = "Turtle Breakout Long"
    if s.n < 60 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    cur = s.closes[-1]
    _cp = get_param(_p, "channel_period", 20)
    h20 = max(s.highs[-_cp-1:-1]); l20 = min(s.lows[-_cp-1:-1])
    sq_thresh = get_param(_p, "squeeze_threshold", 0.85)
    rr = [s.highs[i]-s.lows[i] for i in range(s.n-5, s.n)]
    lr = [s.highs[i]-s.lows[i] for i in range(s.n-20, s.n-5)]
    if float(np.mean(lr)) > 0 and float(np.mean(rr)) > float(np.mean(lr)) * sq_thresh: return None
    avg_vol = float(np.mean(s.volumes[-20:]))
    vm = get_param(_p, "vol_mult", 1.3)
    vol_ok = avg_vol > 0 and s.volumes[-1] >= avg_vol * vm
    _sm = get_param(_p, "stop_atr_mult", 2.0)
    if cur > h20:
        conf = 0.58 if vol_ok else 0.52
        entry = cur; stop = entry - atr * _sm
        if entry - stop <= 0: return None
        return _make(s, "Turtle Breakout Long", Bias.LONG, entry, stop, round(entry + atr * 4, 2), conf,
                     f"Turtle BO Long",
                     target_1=round(entry + atr * 2, 2), target_2=round(entry + atr * 4, 2),
                     trail_type="atr", trail_param=2.0, position_splits=(0.25, 0.25, 0.50))
    elif cur < l20:
        conf = 0.58 if vol_ok else 0.52
        entry = cur; stop = entry + atr * _sm
        if stop - entry <= 0: return None
        return _make(s, "Turtle Breakout Short", Bias.SHORT, entry, stop, round(entry - atr * 4, 2), conf,
                     f"Turtle BO Short",
                     target_1=round(entry - atr * 2, 2), target_2=round(entry - atr * 4, 2),
                     trail_type="atr", trail_param=2.0, position_splits=(0.25, 0.25, 0.50))
    return None


def _detect_bab(s):
    _p = "BAB Long"
    if s.n < 120 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    cur = s.closes[-1]
    v60 = float(np.std(np.diff(s.closes[-61:]) / s.closes[-61:-1])) * np.sqrt(252)
    max_vol = get_param(_p, "max_vol", 0.25)
    if v60 > max_vol: return None
    sma50 = float(np.mean(s.closes[-50:]))
    sma200 = float(np.mean(s.closes[-200:])) if s.n >= 200 else sma50
    if cur < sma50 or cur < sma200: return None
    if s.n >= 63 and (cur - s.closes[-63]) / s.closes[-63] < 0: return None
    conf = 0.58 + (max_vol - v60) * 0.20
    _sm = get_param(_p, "stop_atr_mult", 2.0)
    entry = cur; stop = cur - atr * _sm
    if entry - stop <= 0: return None
    return _make(s, _p, Bias.LONG, entry, stop, round(entry + atr * 4, 2), conf,
                 f"BAB Long: vol={v60:.0%}",
                 target_1=round(entry + atr * 2, 2), target_2=round(entry + atr * 4, 2),
                 position_splits=(0.30, 0.30, 0.40))


def _detect_52w_high_momentum(s):
    _p = "52W High Momentum"
    if s.n < 252 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    cur = s.closes[-1]; h52 = max(s.highs[-252:])
    if h52 <= 0: return None
    pfh = (h52 - cur) / h52
    max_pct = get_param(_p, "pct_from_high", 0.05)
    if pfh > max_pct or pfh < 0: return None
    avg_vol = float(np.mean(s.volumes[-20:]))
    if avg_vol > 0 and s.volumes[-1] < avg_vol * 0.8: return None
    conf = 0.58 + (max_pct - pfh) * 2.0
    _sm = get_param(_p, "stop_atr_mult", 2.5)
    entry = cur; stop = cur - atr * _sm
    if entry - stop <= 0: return None
    return _make(s, _p, Bias.LONG, entry, stop, round(h52 + atr * 2, 2), conf,
                 f"52W High: {pfh:.1%} from high",
                 target_1=round(h52, 2), target_2=round(h52 + atr * 2, 2),
                 position_splits=(0.30, 0.30, 0.40))


def _detect_relative_strength(s):
    _p = "RS Persistence Long"
    if s.n < 60 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    cur = s.closes[-1]; sma50 = float(np.mean(s.closes[-50:]))
    if sma50 <= 0: return None
    rs = cur / sma50; rs_thresh = get_param(_p, "rs_threshold", 1.05)
    if rs < rs_thresh: return None
    sma20 = float(np.mean(s.closes[-20:]))
    if cur < sma20: return None
    v20 = float(np.mean(s.volumes[-20:])); v50 = float(np.mean(s.volumes[-50:]))
    conf = 0.56 + (0.05 if v50 > 0 and v20 > v50 else 0) + (0.05 if rs > 1.10 else 0)
    entry = cur; stop = sma20 - _atr_offset(atr, 0.10)
    if entry - stop <= 0: return None
    return _make(s, _p, Bias.LONG, entry, stop, round(entry + atr * 4, 2), conf,
                 f"RS Persistence: RS={rs:.2f}",
                 target_1=round(entry + atr * 2, 2), target_2=round(entry + atr * 4, 2),
                 position_splits=(0.30, 0.30, 0.40))


def _detect_consecutive_reversal(s):
    _p = "Streak Reversal Long"
    if s.n < 10 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    up_count = 0; dn_count = 0
    for i in range(s.n - 1, max(s.n - 10, 0), -1):
        if s.closes[i] > s.closes[i-1]:
            if dn_count > 0: break
            up_count += 1
        elif s.closes[i] < s.closes[i-1]:
            if up_count > 0: break
            dn_count += 1
        else: break
    cur = s.closes[-1]; _ms = get_param(_p, "min_streak", 5)
    _sm = get_param(_p, "stop_atr_mult", 0.10)
    if up_count >= _ms:
        conf = 0.55 + min(0.10, (up_count - _ms) * 0.03)
        entry = cur; stop = max(s.highs[-up_count:]) + _atr_offset(atr, _sm)
        if stop - entry <= 0: return None
        return _make(s, "Streak Reversal Short", Bias.SHORT, entry, stop, round(entry - atr * 2, 2), conf,
                     f"Streak Rev Short: {up_count} up days",
                     target_1=round(entry - atr, 2), target_2=round(entry - atr * 2, 2))
    elif dn_count >= _ms:
        conf = 0.55 + min(0.10, (dn_count - _ms) * 0.03)
        entry = cur; stop = min(s.lows[-dn_count:]) - _atr_offset(atr, _sm)
        if entry - stop <= 0: return None
        return _make(s, "Streak Reversal Long", Bias.LONG, entry, stop, round(entry + atr * 2, 2), conf,
                     f"Streak Rev Long: {dn_count} down days",
                     target_1=round(entry + atr, 2), target_2=round(entry + atr * 2, 2))
    return None


def _detect_atr_expansion(s):
    _p = "ATR Expansion Long"
    if s.n < 30 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    tr = s.highs[-1] - s.lows[-1]
    ar = float(np.mean([s.highs[i]-s.lows[i] for i in range(s.n-20, s.n-1)]))
    if ar <= 0: return None
    exp_thresh = get_param(_p, "expansion_threshold", 1.8)
    exp = tr / ar
    if exp < exp_thresh: return None
    vm = get_param(_p, "vol_mult", 1.3)
    avg_vol = float(np.mean(s.volumes[-20:]))
    if avg_vol > 0 and s.volumes[-1] < avg_vol * vm: return None
    cur = s.closes[-1]; conf = 0.56 + min(0.10, (exp - exp_thresh) * 0.05)
    dp = (cur - s.lows[-1]) / tr if tr > 0 else 0.5
    _sm = get_param(_p, "stop_atr_mult", 0.10)
    cp_long = get_param(_p, "close_position", 0.65)
    cp_short = get_param("ATR Expansion Short", "close_position", 0.35)
    if dp > cp_long:
        entry = cur; stop = s.lows[-1] - _atr_offset(atr, _sm)
        if entry - stop <= 0: return None
        return _make(s, "ATR Expansion Long", Bias.LONG, entry, stop, round(entry + atr * 3, 2), conf,
                     f"ATR Exp Long: {exp:.1f}x",
                     target_1=round(entry + atr * 1.5, 2), target_2=round(entry + atr * 3, 2))
    elif dp < cp_short:
        entry = cur; stop = s.highs[-1] + _atr_offset(atr, _sm)
        if stop - entry <= 0: return None
        return _make(s, "ATR Expansion Short", Bias.SHORT, entry, stop, round(entry - atr * 3, 2), conf,
                     f"ATR Exp Short: {exp:.1f}x",
                     target_1=round(entry - atr * 1.5, 2), target_2=round(entry - atr * 3, 2))
    return None


def _detect_bollinger_squeeze(s):
    _p = "BB Squeeze Long"
    if s.n < 130 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    _bp = get_param(_p, "bb_period", 20); _bs = get_param(_p, "bb_std", 2.0)
    sma = float(np.mean(s.closes[-_bp:])); std = float(np.std(s.closes[-_bp:]))
    if sma <= 0 or std <= 0: return None
    upper = sma + _bs * std; lower = sma - _bs * std; bw = (upper - lower) / sma
    bws = []
    for j in range(20, min(s.n, 126)):
        idx = s.n - j; m = float(np.mean(s.closes[idx-_bp:idx])); sd = float(np.std(s.closes[idx-_bp:idx]))
        if m > 0 and sd > 0: bws.append((m + _bs*sd - (m - _bs*sd)) / m)
    if len(bws) < 20: return None
    bw_pct_thresh = get_param(_p, "bw_percentile", 0.10)
    bw_pct = sum(1 for b in bws if b > bw) / len(bws)
    if bw_pct > bw_pct_thresh: return None
    cur = s.closes[-1]; _sm = get_param(_p, "stop_atr_mult", 0.10)
    if cur > upper:
        entry = cur; stop = sma - _atr_offset(atr, _sm)
        if entry - stop <= 0: return None
        return _make(s, "BB Squeeze Long", Bias.LONG, entry, stop, round(entry + atr * 4, 2), 0.58,
                     f"BB Squeeze Long: bw at {bw_pct:.0%} pctl",
                     target_1=round(entry + atr * 2, 2), target_2=round(entry + atr * 4, 2))
    elif cur < lower:
        entry = cur; stop = sma + _atr_offset(atr, _sm)
        if stop - entry <= 0: return None
        return _make(s, "BB Squeeze Short", Bias.SHORT, entry, stop, round(entry - atr * 4, 2), 0.56,
                     f"BB Squeeze Short: bw at {bw_pct:.0%} pctl",
                     target_1=round(entry - atr * 2, 2), target_2=round(entry - atr * 4, 2))
    return None


def _detect_accumulation_day(s):
    _p = "Accumulation Long"
    if s.n < 30 or s.timeframe != "1d": return None
    atr = s.current_atr
    if atr <= 0: return None
    avg_vol = float(np.mean(s.volumes[-50:])) if s.n >= 50 else float(np.mean(s.volumes[-20:]))
    if avg_vol <= 0: return None
    _lw = get_param(_p, "lookback_window", 10); _adm = get_param(_p, "acc_days_min", 3)
    acc = 0; dist = 0
    for i in range(max(0, s.n - _lw), s.n):
        if i == 0: continue
        up = s.closes[i] > s.closes[i-1]; va = s.volumes[i] > avg_vol
        if up and va: acc += 1
        elif not up and va: dist += 1
    cur = s.closes[-1]; _sm = get_param(_p, "stop_atr_mult", 0.10)
    if acc >= _adm and dist <= 1:
        sma50 = float(np.mean(s.closes[-50:])) if s.n >= 50 else cur
        if cur < sma50: return None
        conf = 0.56 + min(0.08, (acc - _adm) * 0.03)
        entry = cur; stop = min(s.lows[-5:]) - _atr_offset(atr, _sm)
        if entry - stop <= 0: return None
        return _make(s, "Accumulation Long", Bias.LONG, entry, stop, round(entry + atr * 4, 2), conf,
                     f"Accumulation: {acc} acc days",
                     target_1=round(entry + atr * 2, 2), target_2=round(entry + atr * 4, 2))
    _ddm = get_param("Distribution Short", "dist_days_min", 3)
    if dist >= _ddm and acc <= 1:
        sma50 = float(np.mean(s.closes[-50:])) if s.n >= 50 else cur
        if cur > sma50: return None
        conf = 0.55 + min(0.08, (dist - _ddm) * 0.03)
        entry = cur; stop = max(s.highs[-5:]) + _atr_offset(atr, _sm)
        if stop - entry <= 0: return None
        return _make(s, "Distribution Short", Bias.SHORT, entry, stop, round(entry - atr * 4, 2), conf,
                     f"Distribution: {dist} dist days",
                     target_1=round(entry - atr * 2, 2), target_2=round(entry - atr * 4, 2))
    return None


# ═══════════════════════════════════════════════════════════════
# DETECTOR MAP — ALL strategies enabled for optimization
# ═══════════════════════════════════════════════════════════════

_DETECTOR_MAP: dict[str, callable] = {
    # Classical structural (14 re-enabled)
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
    "Cup & Handle":         _detect_cup_and_handle,
    "Rectangle":            _detect_rectangle,
    "Rising Wedge":         _detect_rising_wedge,
    "Falling Wedge":        _detect_falling_wedge,
    # SMB Scalps (7)
    "RubberBand Scalp":     _detect_rubberband_scalp,
    "ORB 15min":            lambda s: _detect_orb(s, 15, "ORB 15min"),
    "ORB 30min":            lambda s: _detect_orb(s, 30, "ORB 30min"),
    "Second Chance Scalp":  _detect_second_chance_scalp,
    "Fashionably Late":     _detect_fashionably_late,
    "Gap Give & Go":        _detect_gap_give_and_go,
    "Tidal Wave":           _detect_tidal_wave,
    # Quant Intraday 5min (10)
    "Mean Reversion":       _detect_mean_reversion,
    "Trend Pullback":       _detect_trend_pullback,
    "Gap Fade":             _detect_gap_fade,
    "VWAP Reversion":       _detect_vwap_reversion,
    "Gap Reversal Long":    _detect_overnight_gap_reversal,
    "Opening Drive Long":   _detect_opening_drive,
    "Power Hour Long":      _detect_power_hour,
    "Volume Climax Long":   _detect_volume_climax,
    "VWAP Trend Long":      _detect_vwap_trend,
    "RSI Divergence Long":  _detect_rsi_divergence,
    "Midday Reversal Long": _detect_midday_reversal,
    # Quant 15min/1h (3)
    "Keltner Breakout Long":  _detect_keltner_breakout,
    "MACD Turn Long":         _detect_macd_reversal,
    "VP Divergence Long":     _detect_vol_price_divergence,
    # Quant Daily (18)
    "Juicer Long":             _detect_juicer_long,
    "Juicer Short":            _detect_juicer_short,
    "Momentum Breakout":       _detect_momentum_breakout,
    "Vol Compression Breakout": _detect_vol_compression_breakout,
    "Range Expansion":         _detect_range_expansion,
    "Volume Breakout":         _detect_volume_breakout,
    "Donchian Breakout":       _detect_donchian_breakout,
    "TS Momentum Long":        _detect_time_series_momentum,
    "Multi-TF Trend Long":     _detect_multi_tf_trend,
    "ST Reversal Long":        _detect_short_term_reversal,
    "Low Vol Long":            _detect_low_vol_anomaly,
    "Turtle Breakout Long":    _detect_turtle_breakout,
    "BAB Long":                _detect_bab,
    "52W High Momentum":       _detect_52w_high_momentum,
    "RS Persistence Long":     _detect_relative_strength,
    "Streak Reversal Long":    _detect_consecutive_reversal,
    "ATR Expansion Long":      _detect_atr_expansion,
    "BB Squeeze Long":         _detect_bollinger_squeeze,
    "Accumulation Long":       _detect_accumulation_day,
}


def classify_all(bars: BarSeries) -> list[TradeSetup]:
    if len(bars.bars) < 15: return []
    s = extract_structures(bars)
    tf = bars.timeframe
    setups = []
    for pattern_name, fn in _DETECTOR_MAP.items():
        meta = PATTERN_META.get(pattern_name, {})
        allowed_tfs = meta.get("tf", ["5min", "15min"])
        if tf not in allowed_tfs: continue
        try:
            result = fn(s)
            if result is not None: setups.append(result)
        except Exception: continue
    setups.sort(key=lambda x: x.confidence, reverse=True)
    return setups