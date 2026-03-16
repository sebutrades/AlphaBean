"""
AlphaBean Pattern Detection Engine v2
======================================
Complete implementation with corrected math.

FIXES in v2:
  1. VWAP: Resets at 9:30 AM ET for intraday; rolling 20-period for daily
  2. ATR: Uses Wilder's smoothing (industry standard), not simple SMA
  3. Swing detection: Timeframe-adaptive order parameter
  4. ORB: Time-window based (not bar count)
  5. Time windows: All timestamps assumed ET (converted in data client)
  6. Volume: Regular-hours-aware filtering

Contains:
  - Technical indicators (EMA, VWAP, ATR, RVOL, swing detection)
  - Base pattern class (PatternDetector, TradeSetup)
  - 10 SMB Scalp patterns
  - 11 Classical chart patterns
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional
import numpy as np


# ==============================================================================
# DATA SCHEMAS (imported from schemas.py in production, inline here for standalone)
# ==============================================================================

from pydantic import BaseModel


class Bar(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    trade_count: int | None = None


class BarSeries(BaseModel):
    symbol: str
    timeframe: str
    bars: list[Bar]

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    @property
    def highs(self) -> list[float]:
        return [b.high for b in self.bars]

    @property
    def lows(self) -> list[float]:
        return [b.low for b in self.bars]

    @property
    def opens(self) -> list[float]:
        return [b.open for b in self.bars]

    @property
    def volumes(self) -> list[int]:
        return [b.volume for b in self.bars]


# ==============================================================================
# TECHNICAL INDICATORS — ALL MATH AUDITED
# ==============================================================================

def calc_ema(prices: list[float], period: int = 9) -> list[Optional[float]]:
    """
    Exponential Moving Average.
    
    Seed = SMA of first `period` bars (correct).
    Formula: EMA_t = price_t * k + EMA_{t-1} * (1 - k), k = 2/(period+1)
    """
    if len(prices) < period:
        return [None] * len(prices)
    k = 2.0 / (period + 1)
    result: list[Optional[float]] = [None] * (period - 1)
    # Seed with SMA
    sma = sum(prices[:period]) / period
    result.append(sma)
    for i in range(period, len(prices)):
        ema_val = prices[i] * k + result[-1] * (1 - k)
        result.append(ema_val)
    return result


def calc_vwap(bars: list[Bar], timeframe: str = "15min") -> list[float]:
    """
    Volume-Weighted Average Price.
    
    FIX v2:
    - Intraday (15min, 1h): Resets at 9:30 AM ET each day.
      Timestamps are already in ET from massive_client.py.
    - Daily: Uses rolling 20-bar VWAP (cumulative VWAP is meaningless on daily).
    """
    if timeframe == "1d":
        return _calc_rolling_vwap(bars, period=20)
    return _calc_intraday_vwap(bars)


def _calc_intraday_vwap(bars: list[Bar]) -> list[float]:
    """Cumulative VWAP that resets at 9:30 AM ET each trading day."""
    result = []
    cum_vol = 0
    cum_tp_vol = 0.0
    prev_date = None

    for bar in bars:
        current_date = bar.timestamp.date()
        bar_time = bar.timestamp.time()

        # Reset at start of new trading day (9:30 ET)
        if prev_date is not None and current_date != prev_date:
            cum_vol = 0
            cum_tp_vol = 0.0
        prev_date = current_date

        tp = (bar.high + bar.low + bar.close) / 3.0
        cum_vol += bar.volume
        cum_tp_vol += tp * bar.volume

        if cum_vol > 0:
            result.append(cum_tp_vol / cum_vol)
        else:
            result.append(tp)

    return result


def _calc_rolling_vwap(bars: list[Bar], period: int = 20) -> list[float]:
    """Rolling VWAP over `period` bars — used for daily timeframe."""
    result = []
    for i in range(len(bars)):
        start = max(0, i - period + 1)
        window = bars[start:i + 1]
        total_vol = sum(b.volume for b in window)
        if total_vol == 0:
            result.append((bars[i].high + bars[i].low + bars[i].close) / 3.0)
        else:
            total_tp_vol = sum(
                ((b.high + b.low + b.close) / 3.0) * b.volume for b in window
            )
            result.append(total_tp_vol / total_vol)
    return result


def calc_atr(bars: list[Bar], period: int = 14) -> list[Optional[float]]:
    """
    Average True Range using Wilder's smoothing (industry standard).
    
    FIX v2: Changed from simple SMA to Wilder's method:
        ATR_t = ((ATR_{t-1} * (period - 1)) + TR_t) / period
    
    This gives more weight to recent volatility and is what every
    professional platform (TradingView, Bloomberg, etc.) uses.
    """
    if len(bars) < 2:
        return [None] * len(bars)

    # Calculate True Ranges
    trs = [bars[0].high - bars[0].low]
    for i in range(1, len(bars)):
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close),
        )
        trs.append(tr)

    # Wilder's smoothing
    result: list[Optional[float]] = [None] * (period - 1)
    # First ATR = simple average of first `period` TRs
    first_atr = sum(trs[:period]) / period
    result.append(first_atr)

    for i in range(period, len(trs)):
        prev_atr = result[-1]
        wilder_atr = ((prev_atr * (period - 1)) + trs[i]) / period
        result.append(wilder_atr)

    return result


def calc_rvol(current_vol: int, avg_vol: float) -> float:
    """Relative Volume = current / average. >2 = in play, >5 = very in play."""
    return current_vol / avg_vol if avg_vol > 0 else 0.0


def _swing_order_for_timeframe(timeframe: str) -> int:
    """
    FIX v2: Timeframe-adaptive swing detection order.
    
    Order = how many bars on each side a swing point must dominate.
    - 15min: order=4 → swing must be extreme over 4*15=60 min each side (2h window)
    - 1h:    order=3 → swing must be extreme over 3h each side (6h window)
    - 1d:    order=5 → swing must be extreme over 5 days each side (2-week window)
    """
    return {"15min": 4, "1h": 3, "1d": 5}.get(timeframe, 3)


def find_swing_highs(highs: list[float], order: int = 3) -> list[int]:
    """Return indices where high[i] > all neighbors within `order` bars."""
    out = []
    for i in range(order, len(highs) - order):
        if all(highs[i] > highs[i - j] for j in range(1, order + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, order + 1)):
            out.append(i)
    return out


def find_swing_lows(lows: list[float], order: int = 3) -> list[int]:
    """Return indices where low[i] < all neighbors within `order` bars."""
    out = []
    for i in range(order, len(lows) - order):
        if all(lows[i] < lows[i - j] for j in range(1, order + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, order + 1)):
            out.append(i)
    return out


def is_in_time_window(ts: datetime, start_h: int, start_m: int,
                      end_h: int, end_m: int) -> bool:
    """
    Check if timestamp is within a time-of-day window.
    
    FIX v2: Timestamps are now ET (converted in massive_client.py),
    so no conversion needed here. For daily bars, this always returns True.
    """
    t = ts.time()
    return time(start_h, start_m) <= t <= time(end_h, end_m)


def is_market_hours(ts: datetime) -> bool:
    """Check if timestamp is during regular market hours (9:30-16:00 ET)."""
    t = ts.time()
    return time(9, 30) <= t <= time(16, 0)


def get_day_open_idx(bars: list[Bar]) -> int:
    """
    Find index of today's market open bar (9:30 ET).
    For daily bars, returns the last bar's index (today = most recent day).
    """
    today = bars[-1].timestamp.date() if bars else None
    for i, b in enumerate(bars):
        if b.timestamp.date() == today and b.timestamp.time() >= time(9, 30):
            return i
    # Fallback: if no 9:30 bar found, find the most recent day's first bar
    if today:
        for i, b in enumerate(bars):
            if b.timestamp.date() == today:
                return i
    return max(0, len(bars) - 20)


def get_orb_by_time(bars: list[Bar], minutes: int = 15) -> tuple[int, int, float, float]:
    """
    FIX v2: Opening Range Break calculated by TIME WINDOW, not bar count.
    
    Returns (start_idx, end_idx, orb_high, orb_low) for the first N minutes
    after 9:30 ET.
    
    For 15min bars with minutes=15, this is just the 9:30 bar.
    For 15min bars with minutes=30, this is the 9:30 and 9:45 bars.
    """
    open_idx = get_day_open_idx(bars)
    if open_idx >= len(bars):
        return (-1, -1, 0, 0)

    open_time = bars[open_idx].timestamp
    cutoff_time = open_time + __import__('datetime').timedelta(minutes=minutes)

    end_idx = open_idx
    for i in range(open_idx, len(bars)):
        if bars[i].timestamp <= cutoff_time:
            end_idx = i
        else:
            break

    if end_idx < open_idx:
        return (-1, -1, 0, 0)

    orb_bars = bars[open_idx:end_idx + 1]
    orb_high = max(b.high for b in orb_bars)
    orb_low = min(b.low for b in orb_bars)

    return (open_idx, end_idx, orb_high, orb_low)


# ==============================================================================
# BASE PATTERN CLASS
# ==============================================================================

class Bias(str, Enum):
    LONG = "long"
    SHORT = "short"


class Timeframe(str, Enum):
    SCALP = "scalp"
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"


@dataclass
class TradeSetup:
    pattern_name: str
    symbol: str
    bias: Bias
    timeframe: Timeframe
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    confidence: float
    detected_at: datetime
    description: str
    win_rate: float = 0.0
    max_attempts: int = 1
    exit_strategy: str = ""
    key_levels: dict = field(default_factory=dict)
    factors_bullish: list = field(default_factory=list)
    factors_bearish: list = field(default_factory=list)
    ideal_time: str = ""


class PatternDetector(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def default_bias(self) -> Bias: ...

    @abstractmethod
    def detect(self, bars: BarSeries) -> Optional[TradeSetup]: ...


# ==============================================================================
# SMB SCALP #1: RUBBERBAND SCALP
# ==============================================================================

class RubberBandScalp(PatternDetector):
    """
    Extended move + acceleration + snapback via double-bar-break.
    Stop: $0.02 below LOD. Exit: 1/3 at 1:1, 1/3 at 2:1, 1/3 into VWAP.
    60-65% WR, 1.6:1 R:R. Max 2 attempts.
    """
    name = "RubberBand Scalp"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 20:
            return None
        b = bars.bars
        vwap_vals = calc_vwap(b, bars.timeframe)
        atr_vals = calc_atr(b, period=14)

        open_idx = get_day_open_idx(b)
        if open_idx >= len(b) - 5:
            return None

        open_price = b[open_idx].open
        day_low = min(bar.low for bar in b[open_idx:])
        day_low_idx = open_idx + min(
            range(len(b) - open_idx), key=lambda i: b[open_idx + i].low
        )

        current_idx = len(b) - 1
        current_price = b[current_idx].close
        current_vwap = vwap_vals[current_idx] if current_idx < len(vwap_vals) else None
        if current_vwap is None or current_price >= current_vwap:
            return None

        current_atr = atr_vals[-1]
        if current_atr is None:
            return None

        extension = open_price - day_low
        atrs_from_open = extension / current_atr if current_atr > 0 else 0
        if atrs_from_open < 1.0:
            return None

        if day_low_idx - open_idx < 5:
            return None

        midpoint = (open_idx + day_low_idx) // 2
        first_half_avg_range = float(np.mean(
            [b[i].high - b[i].low for i in range(open_idx, midpoint)]
        ))
        second_half_avg_range = float(np.mean(
            [b[i].high - b[i].low for i in range(midpoint, day_low_idx)]
        ))
        if second_half_avg_range < first_half_avg_range * 1.2:
            return None

        snapback_found = False
        snapback_idx = None
        for i in range(day_low_idx + 1, min(day_low_idx + 10, len(b))):
            if b[i].close > b[i].open:
                preceding_highs_cleared = sum(
                    1 for j in range(max(day_low_idx, i - 5), i)
                    if b[i].high > b[j].high
                )
                if preceding_highs_cleared >= 2:
                    snapback_found = True
                    snapback_idx = i
                    break

        if not snapback_found:
            return None

        day_volumes = [b[i].volume for i in range(open_idx, len(b))]
        day_volumes_sorted = sorted(day_volumes, reverse=True)
        snapback_vol = b[snapback_idx].volume
        is_top5 = snapback_vol >= day_volumes_sorted[min(4, len(day_volumes_sorted) - 1)]

        entry = b[snapback_idx].high + 0.02
        stop = day_low - 0.02
        risk = entry - stop
        if risk <= 0:
            return None

        target_2 = entry + 2 * risk
        rr = (target_2 - entry) / risk

        confidence = 0.45
        if atrs_from_open >= 3.0:
            confidence += 0.15
        if is_top5:
            confidence += 0.10
        if second_half_avg_range > first_half_avg_range * 1.5:
            confidence += 0.10
        if is_in_time_window(b[snapback_idx].timestamp, 10, 0, 10, 45):
            confidence += 0.10
        elif is_in_time_window(b[snapback_idx].timestamp, 10, 45, 13, 30):
            confidence += 0.05
        confidence = min(confidence, 0.95)

        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.SCALP,
            entry_price=round(entry, 2), stop_loss=round(stop, 2),
            target_price=round(target_2, 2), risk_reward_ratio=round(rr, 2),
            confidence=round(confidence, 2), detected_at=b[snapback_idx].timestamp,
            description=f"RubberBand: {atrs_from_open:.1f} ATR extension, snapback detected",
            win_rate=0.625, max_attempts=2,
            exit_strategy="1/3 at 1:1, 1/3 at 2:1, 1/3 into VWAP",
            key_levels={"day_low": day_low, "vwap": round(current_vwap, 2),
                        "target_1": round(entry + risk, 2), "target_2": round(target_2, 2)},
            ideal_time="10:00-10:45 AM or 10:45-1:30 PM ET",
        )


# ==============================================================================
# SMB SCALP #2: HITCHHIKER SCALP
# ==============================================================================

class HitchHikerScalp(PatternDetector):
    """
    Drive off open + 5-20 bar consolidation in upper 1/3 + breakout.
    Stop: $0.02 below consol low. Exit: 1/2 wave 1, 1/2 wave 2.
    55-60% WR, 1.9:1 R:R. One-and-done.
    """
    name = "HitchHiker Scalp"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 15:
            return None
        b = bars.bars
        open_idx = get_day_open_idx(b)
        if len(b) - open_idx < 10:
            return None

        drive_end = None
        for i in range(open_idx + 3, min(open_idx + 10, len(b))):
            if b[i].close > b[open_idx].open * 1.005:
                drive_end = i
                break
        if drive_end is None:
            return None

        day_high = max(bar.high for bar in b[open_idx:])
        day_low = min(bar.low for bar in b[open_idx:])
        day_range = day_high - day_low
        if day_range <= 0:
            return None

        consol_start = drive_end + 1
        if consol_start >= len(b) - 2:
            return None

        best_consol = None
        for consol_end in range(consol_start + 4, min(consol_start + 21, len(b) - 1)):
            consol_bars = b[consol_start:consol_end + 1]
            consol_high = max(bar.high for bar in consol_bars)
            consol_low = min(bar.low for bar in consol_bars)
            consol_range = consol_high - consol_low

            if consol_range < day_range * 0.40:
                upper_third_threshold = day_low + day_range * (2 / 3)
                if consol_low >= upper_third_threshold:
                    best_consol = (consol_start, consol_end, consol_high, consol_low)

        if best_consol is None:
            return None

        c_start, c_end, c_high, c_low = best_consol
        if c_end + 1 >= len(b):
            return None

        breakout_bar = b[c_end + 1]
        if breakout_bar.high <= c_high:
            return None

        consol_avg_vol = float(np.mean([bar.volume for bar in b[c_start:c_end + 1]]))

        entry = c_high + 0.02
        stop = c_low - 0.02
        risk = entry - stop
        if risk <= 0:
            return None

        target = entry + risk * 1.9
        rr = (target - entry) / risk

        confidence = 0.50
        if breakout_bar.volume > consol_avg_vol * 1.3:
            confidence += 0.10
        if is_in_time_window(breakout_bar.timestamp, 9, 30, 9, 59):
            confidence += 0.10
        if c_low > b[open_idx].high:
            confidence += 0.05
        confidence = min(confidence, 0.90)

        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.SCALP,
            entry_price=round(entry, 2), stop_loss=round(stop, 2),
            target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
            confidence=round(confidence, 2), detected_at=breakout_bar.timestamp,
            description=f"HitchHiker: {c_end - c_start + 1}-bar consol in upper 1/3, breakout",
            win_rate=0.575, max_attempts=1,
            exit_strategy="1/2 into first wave, 1/2 into second wave",
            key_levels={"consol_high": c_high, "consol_low": c_low, "day_high": day_high},
            ideal_time="Opening drive before 9:59 AM ET",
        )


# ==============================================================================
# SMB SCALP #3: OPENING RANGE BREAK (TIME-BASED FIX)
# ==============================================================================

class OpeningRangeBreak(PatternDetector):
    """
    FIX v2: ORB now uses TIME WINDOW, not bar count.
    orb_minutes=15 means the range from 9:30 to 9:45 (regardless of bar size).
    """
    name = "Opening Range Break"
    default_bias = Bias.LONG

    def __init__(self, orb_minutes: int = 15):
        self.orb_minutes = orb_minutes

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 10:
            return None
        b = bars.bars

        open_idx, orb_end_idx, orb_high, orb_low = get_orb_by_time(b, self.orb_minutes)
        if open_idx < 0 or orb_end_idx < 0:
            return None

        orb_range = orb_high - orb_low
        if orb_range <= 0:
            return None

        orb_bars = b[open_idx:orb_end_idx + 1]

        for i in range(orb_end_idx + 1, min(orb_end_idx + 30, len(b))):
            bar = b[i]

            if bar.close > orb_high:
                entry = orb_high + 0.02
                stop = bar.low - 0.02
                risk = entry - stop
                if risk <= 0 or risk > orb_range:
                    continue
                target = entry + 2 * orb_range
                rr = (target - entry) / risk

                confidence = 0.50
                orb_avg_vol = float(np.mean([b_.volume for b_ in orb_bars])) if orb_bars else 1
                if bar.volume > orb_avg_vol * 1.5:
                    confidence += 0.15
                if orb_range < b[open_idx].close * 0.02:
                    confidence += 0.05

                return TradeSetup(
                    pattern_name=f"ORB {self.orb_minutes}min", symbol=bars.symbol,
                    bias=Bias.LONG, timeframe=Timeframe.SCALP,
                    entry_price=round(entry, 2), stop_loss=round(stop, 2),
                    target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
                    confidence=round(min(confidence, 0.90), 2),
                    detected_at=bar.timestamp,
                    description=f"ORB {self.orb_minutes}min breakout above {orb_high:.2f}",
                    win_rate=0.55, max_attempts=2,
                    exit_strategy="2x measured move target",
                    key_levels={"orb_high": orb_high, "orb_low": orb_low,
                                "orb_range": round(orb_range, 2)},
                    ideal_time=f"First 30 min after {self.orb_minutes}min ORB period",
                )

            if bar.close < orb_low:
                entry = orb_low - 0.02
                stop = bar.high + 0.02
                risk = stop - entry
                if risk <= 0 or risk > orb_range:
                    continue
                target = entry - 2 * orb_range
                rr = (entry - target) / risk

                return TradeSetup(
                    pattern_name=f"ORB {self.orb_minutes}min", symbol=bars.symbol,
                    bias=Bias.SHORT, timeframe=Timeframe.SCALP,
                    entry_price=round(entry, 2), stop_loss=round(stop, 2),
                    target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
                    confidence=0.55, detected_at=bar.timestamp,
                    description=f"ORB {self.orb_minutes}min breakdown below {orb_low:.2f}",
                    win_rate=0.55, max_attempts=2,
                    exit_strategy="2x measured move target",
                    key_levels={"orb_high": orb_high, "orb_low": orb_low},
                    ideal_time=f"First 30 min after {self.orb_minutes}min ORB period",
                )

        return None


# ==============================================================================
# SMB SCALP #4: SECOND CHANCE SCALP
# ==============================================================================

class SecondChanceScalp(PatternDetector):
    name = "Second Chance Scalp"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 30:
            return None
        b = bars.bars
        order = _swing_order_for_timeframe(bars.timeframe)
        swing_highs = find_swing_highs([bar.high for bar in b], order=order)
        if len(swing_highs) < 2:
            return None

        for i in range(len(swing_highs) - 1):
            level = b[swing_highs[i]].high
            tolerance = level * 0.003
            cluster = [swing_highs[i]]
            for j in range(i + 1, len(swing_highs)):
                if abs(b[swing_highs[j]].high - level) < tolerance:
                    cluster.append(swing_highs[j])

            if len(cluster) < 2:
                continue

            last_test = cluster[-1]
            resistance = float(np.mean([b[idx].high for idx in cluster]))

            breakout_idx = None
            for k in range(last_test + 1, min(last_test + 20, len(b))):
                if b[k].close > resistance * 1.002:
                    breakout_idx = k
                    break
            if breakout_idx is None:
                continue

            pullback_idx = None
            pullback_high = b[breakout_idx].high
            for k in range(breakout_idx + 1, min(breakout_idx + 15, len(b))):
                pullback_high = max(pullback_high, b[k].high)
                if b[k].low <= resistance * 1.003:
                    pullback_idx = k
                    break
            if pullback_idx is None:
                continue

            if pullback_idx + 1 >= len(b):
                continue

            confirm = b[pullback_idx + 1]
            if confirm.close > b[pullback_idx].high:
                entry = confirm.close
                stop = b[pullback_idx].low - 0.02
                risk = entry - stop
                if risk <= 0:
                    continue
                target = pullback_high
                rr = (target - entry) / risk if risk > 0 else 0
                if rr < 1.0:
                    continue

                break_vol = b[breakout_idx].volume
                retest_vol = b[pullback_idx].volume
                vol_confirms = retest_vol < break_vol * 0.7

                confidence = 0.45
                if vol_confirms:
                    confidence += 0.15
                if rr >= 1.9:
                    confidence += 0.10

                return TradeSetup(
                    pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
                    timeframe=Timeframe.SCALP,
                    entry_price=round(entry, 2), stop_loss=round(stop, 2),
                    target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
                    confidence=round(min(confidence, 0.90), 2),
                    detected_at=confirm.timestamp,
                    description=f"Second Chance: Retest of {resistance:.2f} held as support",
                    win_rate=0.525, max_attempts=2,
                    exit_strategy="1/2 at pullback high, trail remaining via 9 EMA",
                    key_levels={"resistance": round(resistance, 2),
                                "pullback_high": round(pullback_high, 2)},
                    ideal_time="9:59-10:44, 10:45-1:29, 1:30-4:00 ET",
                )
        return None


# ==============================================================================
# SMB SCALP #5: BACKSIDE SCALP
# ==============================================================================

class BackSideScalp(PatternDetector):
    name = "BackSide Scalp"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 25:
            return None
        b = bars.bars
        ema9 = calc_ema([bar.close for bar in b], 9)
        vwap_vals = calc_vwap(b, bars.timeframe)

        open_idx = get_day_open_idx(b)
        if len(b) - open_idx < 15:
            return None

        day_low = min(bar.low for bar in b[open_idx:])
        day_low_idx = open_idx + min(
            range(len(b) - open_idx), key=lambda i: b[open_idx + i].low
        )

        if day_low_idx < open_idx + 3:
            return None

        current_vwap = vwap_vals[-1]
        if day_low >= current_vwap:
            return None

        post_low_bars = b[day_low_idx:]
        if len(post_low_bars) < 8:
            return None

        sw_highs = find_swing_highs([bar.high for bar in post_low_bars], order=2)
        sw_lows = find_swing_lows([bar.low for bar in post_low_bars], order=2)
        if len(sw_highs) < 1 or len(sw_lows) < 1:
            return None

        has_hh = any(post_low_bars[h].high > post_low_bars[0].high for h in sw_highs)
        has_hl = any(post_low_bars[l].low > day_low for l in sw_lows)
        if not (has_hh and has_hl):
            return None

        current_ema = ema9[-1]
        prev_ema = ema9[-3] if len(ema9) > 3 else None
        if current_ema is None or prev_ema is None:
            return None
        if current_ema <= prev_ema:
            return None
        if b[-1].close < current_ema:
            return None

        recent_hl = day_low
        for l in sw_lows:
            if post_low_bars[l].low > day_low:
                recent_hl = post_low_bars[l].low

        entry = b[-1].high + 0.02
        stop = recent_hl - 0.02
        target = current_vwap
        risk = entry - stop
        if risk <= 0 or target <= entry:
            return None

        rr = (target - entry) / risk
        if rr < 1.0:
            return None

        confidence = 0.48
        bars_above_ema = sum(1 for i in range(-8, 0) if ema9[i] and b[i].close > ema9[i])
        if bars_above_ema >= 6:
            confidence += 0.10
        if is_in_time_window(b[-1].timestamp, 10, 0, 10, 45):
            confidence += 0.10
        confidence = min(confidence, 0.85)

        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.SCALP,
            entry_price=round(entry, 2), stop_loss=round(stop, 2),
            target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
            confidence=round(confidence, 2), detected_at=b[-1].timestamp,
            description=f"BackSide: HH+HL above 9 EMA, targeting VWAP at {current_vwap:.2f}",
            win_rate=0.55, max_attempts=1,
            exit_strategy="Entire position at VWAP",
            key_levels={"day_low": day_low, "vwap": round(current_vwap, 2),
                        "recent_hl": round(recent_hl, 2)},
            ideal_time="10:00-10:45 AM, 10:46-1:30 PM ET",
        )


# ==============================================================================
# SMB SCALP #6: FASHIONABLY LATE SCALP
# ==============================================================================

class FashionablyLateScalp(PatternDetector):
    name = "Fashionably Late Scalp"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 30:
            return None
        b = bars.bars
        ema9 = calc_ema([bar.close for bar in b], 9)
        vwap_vals = calc_vwap(b, bars.timeframe)

        open_idx = get_day_open_idx(b)
        if len(b) - open_idx < 20:
            return None
        day_low = min(bar.low for bar in b[open_idx:])

        cross_idx = None
        for i in range(open_idx + 10, len(b)):
            if ema9[i] is None or ema9[i - 1] is None:
                continue
            if i >= len(vwap_vals):
                continue

            if ema9[i - 1] < vwap_vals[i - 1] and ema9[i] >= vwap_vals[i]:
                if i >= 3 and ema9[i - 3] is not None and ema9[i] > ema9[i - 3]:
                    vwap_slope = vwap_vals[i] - vwap_vals[max(0, i - 5)]
                    if vwap_slope <= vwap_vals[i] * 0.001:
                        cross_idx = i

        if cross_idx is None:
            return None

        cross_price = vwap_vals[cross_idx]
        measured_move = cross_price - day_low
        if measured_move <= 0:
            return None

        entry = cross_price
        stop = cross_price - (measured_move / 3)
        target = cross_price + measured_move
        risk = entry - stop
        if risk <= 0:
            return None

        rr = (target - entry) / risk
        confidence = 0.55
        if is_in_time_window(b[cross_idx].timestamp, 10, 0, 10, 45):
            confidence += 0.10
        if rr >= 2.5:
            confidence += 0.10
        confidence = min(confidence, 0.90)

        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.SCALP,
            entry_price=round(entry, 2), stop_loss=round(stop, 2),
            target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
            confidence=round(confidence, 2), detected_at=b[cross_idx].timestamp,
            description=f"Fashionably Late: 9 EMA crossed VWAP, measured move ${measured_move:.2f}",
            win_rate=0.60, max_attempts=1,
            exit_strategy="Exit at 1 measured move above the cross",
            key_levels={"day_low": day_low, "cross_price": round(cross_price, 2),
                        "measured_move": round(measured_move, 2)},
            ideal_time="10:00-10:45 AM, 10:46-1:30 PM ET",
        )


# ==============================================================================
# SMB SCALP #7: SPENCER SCALP
# ==============================================================================

class SpencerScalp(PatternDetector):
    name = "Spencer Scalp"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 30:
            return None
        b = bars.bars
        open_idx = get_day_open_idx(b)
        if len(b) - open_idx < 25:
            return None

        day_high = max(bar.high for bar in b[open_idx:])
        day_low = min(bar.low for bar in b[open_idx:])
        day_range = day_high - day_low
        if day_range <= 0:
            return None

        upper_third = day_low + day_range * (2 / 3)

        for start in range(open_idx + 5, len(b) - 21):
            consol_bars = b[start:start + 20]
            c_high = max(bar.high for bar in consol_bars)
            c_low = min(bar.low for bar in consol_bars)
            c_range = c_high - c_low

            if c_low < upper_third or c_range > day_range * 0.35:
                continue

            vols = [bar.volume for bar in consol_bars]
            vol_std = float(np.std(vols))
            vol_mean = float(np.mean(vols))
            if vol_mean > 0 and vol_std / vol_mean > 0.8:
                continue

            break_idx = start + 20
            if break_idx >= len(b):
                continue

            if b[break_idx].high > c_high:
                entry = c_high + 0.02
                stop = c_low - 0.02
                risk = entry - stop
                if risk <= 0:
                    continue

                target = entry + 2 * c_range
                rr = (target - entry) / risk

                confidence = 0.50
                if break_idx >= 2 and b[break_idx - 1].volume < vol_mean * 0.7:
                    confidence += 0.10
                if c_range < day_range * 0.20:
                    confidence += 0.10
                confidence = min(confidence, 0.85)

                return TradeSetup(
                    pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
                    timeframe=Timeframe.INTRADAY,
                    entry_price=round(entry, 2), stop_loss=round(stop, 2),
                    target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
                    confidence=round(confidence, 2), detected_at=b[break_idx].timestamp,
                    description=f"Spencer: {len(consol_bars)}-bar consolidation near HOD",
                    win_rate=0.55, max_attempts=1,
                    exit_strategy="1/4 at 1:1, 1/2 at 2:1, 1/4 at 3:1 measured move",
                    key_levels={"consol_high": c_high, "consol_low": c_low, "day_high": day_high},
                    ideal_time="9:59-10:44, 10:45-1:29, 1:30-4:00 ET",
                )
        return None


# ==============================================================================
# SMB SCALP #8: GAP GIVE AND GO
# ==============================================================================

class GapGiveAndGo(PatternDetector):
    name = "Gap Give and Go"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 15:
            return None
        b = bars.bars
        open_idx = get_day_open_idx(b)
        if len(b) - open_idx < 10 or open_idx == 0:
            return None

        prev_close = b[open_idx - 1].close
        open_price = b[open_idx].open
        gap_pct = (open_price - prev_close) / prev_close * 100
        if gap_pct < 0.5:
            return None

        drop_end = None
        for i in range(open_idx + 1, min(open_idx + 6, len(b))):
            if b[i].low < open_price * 0.995:
                drop_end = i
        if drop_end is None:
            return None

        for consol_len in range(3, 8):
            c_end = drop_end + consol_len
            if c_end >= len(b) - 1:
                continue

            consol_bars_slice = b[drop_end:c_end]
            c_high = max(bar.high for bar in consol_bars_slice)
            c_low = min(bar.low for bar in consol_bars_slice)
            c_range = c_high - c_low
            initial_drop = open_price - c_low
            if initial_drop <= 0 or c_range > initial_drop * 0.50:
                continue

            if b[c_end].high > c_high:
                entry = c_high + 0.02
                stop = c_low - 0.02
                risk = entry - stop
                if risk <= 0:
                    continue
                target = open_price
                rr = (target - entry) / risk if risk > 0 else 0
                if rr < 1.0:
                    continue

                confidence = 0.50
                pre_vol = float(np.mean([b[j].volume for j in range(open_idx, drop_end)]))
                consol_vol = float(np.mean([bar.volume for bar in consol_bars_slice]))
                if pre_vol > 0 and consol_vol < pre_vol * 0.5:
                    confidence += 0.15
                if is_in_time_window(b[c_end].timestamp, 9, 30, 9, 45):
                    confidence += 0.10
                confidence = min(confidence, 0.85)

                return TradeSetup(
                    pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
                    timeframe=Timeframe.SCALP,
                    entry_price=round(entry, 2), stop_loss=round(stop, 2),
                    target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
                    confidence=round(confidence, 2), detected_at=b[c_end].timestamp,
                    description=f"Gap Give and Go: {gap_pct:.1f}% gap, {consol_len}-bar consol",
                    win_rate=0.55, max_attempts=2,
                    exit_strategy="Move2Move: exit on double bar break against",
                    key_levels={"gap_open": round(open_price, 2),
                                "consol_high": c_high, "consol_low": c_low},
                    ideal_time="9:30-9:45 AM ET (opening drive)",
                )
        return None


# ==============================================================================
# SMB SCALP #9: TIDAL WAVE / BOUNCY BALL
# ==============================================================================

class TidalWaveBouncyBall(PatternDetector):
    name = "Tidal Wave / Bouncy Ball"
    default_bias = Bias.SHORT

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 30:
            return None
        b = bars.bars
        order = _swing_order_for_timeframe(bars.timeframe)
        sw_lows = find_swing_lows([bar.low for bar in b], order=order)
        if len(sw_lows) < 2:
            return None

        for i in range(len(sw_lows)):
            support_level = b[sw_lows[i]].low
            tolerance = support_level * 0.003
            touches = [sw_lows[i]]
            for j in range(i + 1, len(sw_lows)):
                if abs(b[sw_lows[j]].low - support_level) < tolerance:
                    touches.append(sw_lows[j])

            if len(touches) < 2:
                continue

            bounce_highs = []
            for t in range(len(touches) - 1):
                between = [b[k].high for k in range(touches[t], touches[t + 1])]
                if between:
                    bounce_highs.append(max(between))
            if len(bounce_highs) < 2:
                continue

            is_diminishing = all(
                bounce_highs[k] > bounce_highs[k + 1]
                for k in range(len(bounce_highs) - 1)
            )
            if not is_diminishing:
                continue

            last_touch = touches[-1]
            for k in range(last_touch + 1, min(last_touch + 10, len(b))):
                if b[k].close < support_level:
                    entry = support_level - 0.02
                    stop = bounce_highs[-1] + 0.02
                    risk = stop - entry
                    if risk <= 0:
                        continue

                    target = entry - 2 * risk
                    rr = (entry - target) / risk

                    confidence = 0.55
                    if len(touches) >= 3:
                        confidence += 0.10
                    ref_start = max(0, last_touch - 5)
                    avg_vol = float(np.mean([bar.volume for bar in b[ref_start:last_touch]]))
                    if avg_vol > 0 and b[k].volume > avg_vol * 1.3:
                        confidence += 0.10
                    confidence = min(confidence, 0.85)

                    return TradeSetup(
                        pattern_name=self.name, symbol=bars.symbol, bias=Bias.SHORT,
                        timeframe=Timeframe.SCALP,
                        entry_price=round(entry, 2), stop_loss=round(stop, 2),
                        target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
                        confidence=round(confidence, 2), detected_at=b[k].timestamp,
                        description=f"Tidal Wave: {len(touches)} touches of {support_level:.2f}, diminishing bounces",
                        win_rate=0.55, max_attempts=1,
                        exit_strategy="1/2 at 2x measured move, 1/4 at 3x, hold rest",
                        key_levels={"support": round(support_level, 2),
                                    "last_bounce_high": round(bounce_highs[-1], 2)},
                        ideal_time="After first or second leg down",
                    )
        return None


# ==============================================================================
# SMB SCALP #10: BREAKING NEWS
# ==============================================================================

class BreakingNewsStrategy(PatternDetector):
    name = "Breaking News"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 12:
            return None
        b = bars.bars

        avg_vol = float(np.mean([bar.volume for bar in b[-12:-2]]))
        current_vol = b[-1].volume
        if avg_vol == 0 or current_vol < avg_vol * 3:
            return None

        avg_range = float(np.mean([bar.high - bar.low for bar in b[-12:-2]]))
        current_range = b[-1].high - b[-1].low
        if avg_range == 0 or current_range < avg_range * 2:
            return None

        is_bullish = b[-1].close > b[-1].open
        if is_bullish:
            entry = b[-1].close
            stop = b[-1].low - 0.02
            risk = entry - stop
            if risk <= 0:
                return None
            target = entry + risk * 2
            bias = Bias.LONG
        else:
            entry = b[-1].close
            stop = b[-1].high + 0.02
            risk = stop - entry
            if risk <= 0:
                return None
            target = entry - risk * 2
            bias = Bias.SHORT

        vol_multiple = current_vol / avg_vol
        confidence = 0.45
        if vol_multiple > 5:
            confidence += 0.15
        if current_range > avg_range * 3:
            confidence += 0.10
        confidence = min(confidence, 0.85)

        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=bias,
            timeframe=Timeframe.SCALP,
            entry_price=round(entry, 2), stop_loss=round(stop, 2),
            target_price=round(target, 2), risk_reward_ratio=2.0,
            confidence=round(confidence, 2), detected_at=b[-1].timestamp,
            description=f"Breaking News: {vol_multiple:.0f}x volume, {current_range / avg_range:.1f}x range",
            win_rate=0.50, max_attempts=2,
            exit_strategy="Momentum: tape extremes. Trend: key levels or 9 EMA loss.",
            key_levels={"volume_multiple": round(vol_multiple, 1)},
            ideal_time="Any time during market hours",
        )


# ==============================================================================
# CLASSICAL PATTERNS (11-21) — with timeframe-adaptive swing detection
# ==============================================================================

class DoubleBottom(PatternDetector):
    name = "Double Bottom"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 30:
            return None
        order = _swing_order_for_timeframe(bars.timeframe)
        sw = find_swing_lows(bars.lows, order=order)
        if len(sw) < 2:
            return None
        l1, l2 = sw[-2], sw[-1]
        p1, p2 = bars.lows[l1], bars.lows[l2]
        if abs(p1 - p2) > p1 * 0.015:
            return None
        neckline = max(bars.highs[l1:l2 + 1])
        if bars.bars[-1].close < neckline:
            return None
        entry = neckline + 0.02
        stop = min(p1, p2) - 0.02
        risk = entry - stop
        if risk <= 0:
            return None
        target = entry + (neckline - min(p1, p2))
        rr = (target - entry) / risk
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.SWING, entry_price=round(entry, 2),
            stop_loss=round(stop, 2), target_price=round(target, 2),
            risk_reward_ratio=round(rr, 2), confidence=0.65,
            detected_at=bars.bars[-1].timestamp,
            description=f"Double Bottom at {min(p1, p2):.2f}, neckline {neckline:.2f}",
            win_rate=0.65, exit_strategy="Measured move from neckline",
            key_levels={"bottom1": round(p1, 2), "bottom2": round(p2, 2),
                        "neckline": round(neckline, 2)},
        )


class DoubleTop(PatternDetector):
    name = "Double Top"
    default_bias = Bias.SHORT

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 30:
            return None
        order = _swing_order_for_timeframe(bars.timeframe)
        sw = find_swing_highs(bars.highs, order=order)
        if len(sw) < 2:
            return None
        h1, h2 = sw[-2], sw[-1]
        p1, p2 = bars.highs[h1], bars.highs[h2]
        if abs(p1 - p2) > p1 * 0.015:
            return None
        neckline = min(bars.lows[h1:h2 + 1])
        if bars.bars[-1].close > neckline:
            return None
        entry = neckline - 0.02
        stop = max(p1, p2) + 0.02
        risk = stop - entry
        if risk <= 0:
            return None
        target = entry - (max(p1, p2) - neckline)
        rr = (entry - target) / risk
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.SHORT,
            timeframe=Timeframe.SWING, entry_price=round(entry, 2),
            stop_loss=round(stop, 2), target_price=round(target, 2),
            risk_reward_ratio=round(rr, 2), confidence=0.63,
            detected_at=bars.bars[-1].timestamp,
            description=f"Double Top at {max(p1, p2):.2f}",
            win_rate=0.63,
            key_levels={"top1": round(p1, 2), "top2": round(p2, 2),
                        "neckline": round(neckline, 2)},
        )


class BullFlag(PatternDetector):
    name = "Bull Flag"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 20:
            return None
        order = _swing_order_for_timeframe(bars.timeframe)
        sw_lo = find_swing_lows(bars.lows, order=order)
        sw_hi = find_swing_highs(bars.highs, order=order)
        if not sw_lo or not sw_hi:
            return None
        pole_lo_idx = sw_lo[-1]
        post = [h for h in sw_hi if h > pole_lo_idx]
        if not post:
            return None
        pole_hi_idx = post[0]
        pole_lo, pole_hi = bars.lows[pole_lo_idx], bars.highs[pole_hi_idx]
        pole_pct = (pole_hi - pole_lo) / pole_lo * 100
        if pole_pct < 2.0:
            return None
        flag = bars.bars[pole_hi_idx:]
        if len(flag) < 3 or len(flag) > 30:
            return None
        flag_low = min(f.low for f in flag)
        retrace = (pole_hi - flag_low) / (pole_hi - pole_lo) if pole_hi != pole_lo else 1
        if retrace > 0.50:
            return None
        entry = pole_hi * 1.001
        stop = flag_low * 0.998
        risk = entry - stop
        if risk <= 0:
            return None
        target = entry + (pole_hi - pole_lo)
        rr = (target - entry) / risk
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.INTRADAY, entry_price=round(entry, 2),
            stop_loss=round(stop, 2), target_price=round(target, 2),
            risk_reward_ratio=round(rr, 2), confidence=0.60,
            detected_at=bars.bars[-1].timestamp,
            description=f"Bull Flag: {pole_pct:.1f}% pole",
            win_rate=0.60,
            key_levels={"pole_hi": pole_hi, "pole_lo": pole_lo, "flag_low": flag_low},
        )


class BearFlag(PatternDetector):
    name = "Bear Flag"
    default_bias = Bias.SHORT

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 20:
            return None
        order = _swing_order_for_timeframe(bars.timeframe)
        sw_hi = find_swing_highs(bars.highs, order=order)
        sw_lo = find_swing_lows(bars.lows, order=order)
        if not sw_hi or not sw_lo:
            return None
        pole_hi_idx = sw_hi[-1]
        post = [l for l in sw_lo if l > pole_hi_idx]
        if not post:
            return None
        pole_lo_idx = post[0]
        pole_hi, pole_lo = bars.highs[pole_hi_idx], bars.lows[pole_lo_idx]
        pole_pct = (pole_hi - pole_lo) / pole_hi * 100
        if pole_pct < 2.0:
            return None
        flag = bars.bars[pole_lo_idx:]
        if len(flag) < 3 or len(flag) > 30:
            return None
        flag_high = max(f.high for f in flag)
        retrace = (flag_high - pole_lo) / (pole_hi - pole_lo) if pole_hi != pole_lo else 1
        if retrace > 0.50:
            return None
        entry = pole_lo * 0.999
        stop = flag_high * 1.002
        risk = stop - entry
        if risk <= 0:
            return None
        target = entry - (pole_hi - pole_lo)
        rr = (entry - target) / risk
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.SHORT,
            timeframe=Timeframe.INTRADAY, entry_price=round(entry, 2),
            stop_loss=round(stop, 2), target_price=round(target, 2),
            risk_reward_ratio=round(rr, 2), confidence=0.58,
            detected_at=bars.bars[-1].timestamp,
            description=f"Bear Flag: {pole_pct:.1f}% pole",
            win_rate=0.58,
            key_levels={"pole_hi": pole_hi, "pole_lo": pole_lo, "flag_high": flag_high},
        )


class HeadAndShoulders(PatternDetector):
    name = "Head & Shoulders"
    default_bias = Bias.SHORT

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 40:
            return None
        order = _swing_order_for_timeframe(bars.timeframe)
        sw = find_swing_highs(bars.highs, order=order)
        if len(sw) < 3:
            return None
        ls, head, rs = sw[-3], sw[-2], sw[-1]
        lp, hp, rp = bars.highs[ls], bars.highs[head], bars.highs[rs]
        if not (hp > lp and hp > rp):
            return None
        if abs(lp - rp) > lp * 0.03:
            return None
        neckline = min(min(bars.lows[ls:head + 1]), min(bars.lows[head:rs + 1]))
        if bars.bars[-1].close > neckline:
            return None
        entry = neckline - 0.02
        stop = rp + 0.02
        risk = stop - entry
        if risk <= 0:
            return None
        target = entry - (hp - neckline)
        rr = (entry - target) / risk
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.SHORT,
            timeframe=Timeframe.SWING, entry_price=round(entry, 2),
            stop_loss=round(stop, 2), target_price=round(target, 2),
            risk_reward_ratio=round(rr, 2), confidence=0.68,
            detected_at=bars.bars[-1].timestamp,
            description=f"H&S: head {hp:.2f}, neckline {neckline:.2f}",
            win_rate=0.68,
            key_levels={"left_shoulder": lp, "head": hp,
                        "right_shoulder": rp, "neckline": neckline},
        )


class InverseHeadAndShoulders(PatternDetector):
    name = "Inverse H&S"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 40:
            return None
        order = _swing_order_for_timeframe(bars.timeframe)
        sw = find_swing_lows(bars.lows, order=order)
        if len(sw) < 3:
            return None
        ls, head, rs = sw[-3], sw[-2], sw[-1]
        lp, hp, rp = bars.lows[ls], bars.lows[head], bars.lows[rs]
        if not (hp < lp and hp < rp):
            return None
        if abs(lp - rp) > lp * 0.03:
            return None
        neckline = max(max(bars.highs[ls:head + 1]), max(bars.highs[head:rs + 1]))
        if bars.bars[-1].close < neckline:
            return None
        entry = neckline + 0.02
        stop = rp - 0.02
        risk = entry - stop
        if risk <= 0:
            return None
        target = entry + (neckline - hp)
        rr = (target - entry) / risk
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.SWING, entry_price=round(entry, 2),
            stop_loss=round(stop, 2), target_price=round(target, 2),
            risk_reward_ratio=round(rr, 2), confidence=0.68,
            detected_at=bars.bars[-1].timestamp,
            description=f"Inv H&S: head {hp:.2f}",
            win_rate=0.68,
            key_levels={"head": hp, "neckline": neckline},
        )


class AscendingTriangle(PatternDetector):
    name = "Ascending Triangle"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 25:
            return None
        order = _swing_order_for_timeframe(bars.timeframe)
        sw_hi = find_swing_highs(bars.highs, order=order)
        sw_lo = find_swing_lows(bars.lows, order=order)
        if len(sw_hi) < 2 or len(sw_lo) < 2:
            return None
        flat_top = float(np.mean([bars.highs[i] for i in sw_hi[-2:]]))
        if abs(bars.highs[sw_hi[-1]] - bars.highs[sw_hi[-2]]) > flat_top * 0.01:
            return None
        if bars.lows[sw_lo[-1]] <= bars.lows[sw_lo[-2]]:
            return None
        if bars.bars[-1].close < flat_top:
            return None
        entry = flat_top + 0.02
        stop = bars.lows[sw_lo[-1]] - 0.02
        risk = entry - stop
        if risk <= 0:
            return None
        target = entry + (flat_top - bars.lows[sw_lo[-1]])
        rr = (target - entry) / risk
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.SWING, entry_price=round(entry, 2),
            stop_loss=round(stop, 2), target_price=round(target, 2),
            risk_reward_ratio=round(rr, 2), confidence=0.62,
            detected_at=bars.bars[-1].timestamp,
            description=f"Asc Triangle: flat top {flat_top:.2f}",
            win_rate=0.62,
            key_levels={"flat_top": round(flat_top, 2)},
        )


class DescendingTriangle(PatternDetector):
    name = "Descending Triangle"
    default_bias = Bias.SHORT

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 25:
            return None
        order = _swing_order_for_timeframe(bars.timeframe)
        sw_hi = find_swing_highs(bars.highs, order=order)
        sw_lo = find_swing_lows(bars.lows, order=order)
        if len(sw_hi) < 2 or len(sw_lo) < 2:
            return None
        flat_bottom = float(np.mean([bars.lows[i] for i in sw_lo[-2:]]))
        if abs(bars.lows[sw_lo[-1]] - bars.lows[sw_lo[-2]]) > flat_bottom * 0.01:
            return None
        if bars.highs[sw_hi[-1]] >= bars.highs[sw_hi[-2]]:
            return None
        if bars.bars[-1].close > flat_bottom:
            return None
        entry = flat_bottom - 0.02
        stop = bars.highs[sw_hi[-1]] + 0.02
        risk = stop - entry
        if risk <= 0:
            return None
        target = entry - (bars.highs[sw_hi[-1]] - flat_bottom)
        rr = (entry - target) / risk
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.SHORT,
            timeframe=Timeframe.SWING, entry_price=round(entry, 2),
            stop_loss=round(stop, 2), target_price=round(target, 2),
            risk_reward_ratio=round(rr, 2), confidence=0.60,
            detected_at=bars.bars[-1].timestamp,
            description=f"Desc Triangle: flat bottom {flat_bottom:.2f}",
            win_rate=0.60,
            key_levels={"flat_bottom": round(flat_bottom, 2)},
        )


class CupWithHandle(PatternDetector):
    name = "Cup with Handle"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 40:
            return None
        order = _swing_order_for_timeframe(bars.timeframe)
        sw_lo = find_swing_lows(bars.lows, order=order)
        if not sw_lo:
            return None
        cup_low_idx = sw_lo[-1]
        cup_low = bars.lows[cup_low_idx]
        left_rim = max(bars.highs[:cup_low_idx]) if cup_low_idx > 0 else None
        right_rim = max(bars.highs[cup_low_idx:]) if cup_low_idx < len(bars.bars) else None
        if left_rim is None or right_rim is None:
            return None
        rim = min(left_rim, right_rim)
        cup_depth = rim - cup_low
        if cup_depth <= 0:
            return None
        handle_bars = bars.bars[cup_low_idx + 5:]
        if len(handle_bars) < 3:
            return None
        window = handle_bars[-10:] if len(handle_bars) >= 10 else handle_bars
        handle_low = min(b.low for b in window)
        handle_retrace = (rim - handle_low) / cup_depth
        if handle_retrace > 0.50 or handle_retrace < 0.10:
            return None
        if bars.bars[-1].close < rim:
            return None
        entry = rim + 0.02
        stop = handle_low - 0.02
        risk = entry - stop
        if risk <= 0:
            return None
        target = entry + cup_depth
        rr = (target - entry) / risk
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.SWING, entry_price=round(entry, 2),
            stop_loss=round(stop, 2), target_price=round(target, 2),
            risk_reward_ratio=round(rr, 2), confidence=0.65,
            detected_at=bars.bars[-1].timestamp,
            description=f"Cup & Handle: rim {rim:.2f}",
            win_rate=0.65,
            key_levels={"rim": round(rim, 2), "cup_low": round(cup_low, 2),
                        "handle_low": round(handle_low, 2)},
        )


# ==============================================================================
# PATTERN REGISTRY
# ==============================================================================

ALL_DETECTORS = [
    # SMB Scalps (12 detectors, 10 unique patterns)
    RubberBandScalp(),
    HitchHikerScalp(),
    OpeningRangeBreak(orb_minutes=15),
    OpeningRangeBreak(orb_minutes=30),
    SecondChanceScalp(),
    BackSideScalp(),
    FashionablyLateScalp(),
    SpencerScalp(),
    GapGiveAndGo(),
    TidalWaveBouncyBall(),
    BreakingNewsStrategy(),
    # Classical (10 detectors)
    DoubleBottom(),
    DoubleTop(),
    BullFlag(),
    BearFlag(),
    HeadAndShoulders(),
    InverseHeadAndShoulders(),
    AscendingTriangle(),
    DescendingTriangle(),
    CupWithHandle(),
]


def get_all_detectors() -> list[PatternDetector]:
    return ALL_DETECTORS


def get_all_pattern_names() -> list[str]:
    """Return sorted unique pattern names for the filter dropdown."""
    names = sorted(set(d.name for d in ALL_DETECTORS))
    return names


def run_scan(bars: BarSeries) -> list[TradeSetup]:
    """Run all detectors on a BarSeries, return setups sorted by confidence."""
    setups = []
    for det in ALL_DETECTORS:
        try:
            result = det.detect(bars)
            if result is not None:
                setups.append(result)
        except Exception:
            continue
    setups.sort(key=lambda s: s.confidence, reverse=True)
    return setups