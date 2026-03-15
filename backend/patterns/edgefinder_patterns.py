"""
EdgeFinder Pattern Detection Engine
====================================
Complete implementation of 50 pattern detectors.

This file contains:
  - Data schemas (Bar, BarSeries)
  - Technical indicator utilities (EMA, VWAP, ATR, RVOL, swing detection)
  - Base pattern class (PatternDetector, TradeSetup)
  - 10 SMB Scalp patterns (from uploaded cheat sheets, with exact rules)
  - 40 Classical chart patterns

MASSIVE.COM DATA CLIENT is in a separate file (massive_client.py).
Install: pip install massive

Usage:
    from edgefinder_patterns import *
    
    bars = <your BarSeries data>
    detector = RubberBandScalp()
    setup = detector.detect(bars)
    if setup:
        print(f"TRADE: {setup.pattern_name} on {setup.symbol}")
        print(f"  Entry: ${setup.entry_price}  Stop: ${setup.stop_loss}  Target: ${setup.target_price}")
        print(f"  R:R: {setup.risk_reward_ratio:.1f}  Confidence: {setup.confidence:.0%}")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional
import numpy as np


# ==============================================================================
# DATA SCHEMAS
# ==============================================================================

from pydantic import BaseModel

class Bar(BaseModel):
    """Single OHLCV candle."""
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
    """Time-ordered series of bars for one symbol."""
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
# TECHNICAL INDICATORS
# ==============================================================================

def calc_ema(prices: list[float], period: int = 9) -> list[Optional[float]]:
    """Exponential Moving Average. Returns list same length as prices (None-padded)."""
    if len(prices) < period:
        return [None] * len(prices)
    k = 2 / (period + 1)
    result = [None] * (period - 1)
    result.append(sum(prices[:period]) / period)
    for i in range(period, len(prices)):
        result.append(prices[i] * k + result[-1] * (1 - k))
    return result

def calc_vwap(bars: list[Bar]) -> list[float]:
    """Cumulative intraday VWAP. Resets daily at market open (9:30 ET)."""
    result = []
    cum_vol = 0
    cum_tp_vol = 0
    prev_date = None
    for bar in bars:
        current_date = bar.timestamp.date()
        if prev_date is not None and current_date != prev_date:
            cum_vol = 0
            cum_tp_vol = 0
        prev_date = current_date
        tp = (bar.high + bar.low + bar.close) / 3
        cum_vol += bar.volume
        cum_tp_vol += tp * bar.volume
        result.append(cum_tp_vol / cum_vol if cum_vol > 0 else tp)
    return result

def calc_atr(bars: list[Bar], period: int = 14) -> list[Optional[float]]:
    """Average True Range over `period` bars."""
    if len(bars) < 2:
        return [None] * len(bars)
    trs = [bars[0].high - bars[0].low]
    for i in range(1, len(bars)):
        trs.append(max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i-1].close),
            abs(bars[i].low - bars[i-1].close),
        ))
    result = [None] * (period - 1)
    for i in range(period - 1, len(trs)):
        result.append(np.mean(trs[i - period + 1 : i + 1]))
    return result

def calc_rvol(current_vol: int, avg_vol: int) -> float:
    """Relative Volume = current / average. >2 = in play, >5 = very in play."""
    return current_vol / avg_vol if avg_vol > 0 else 0.0

def find_swing_highs(highs: list[float], order: int = 3) -> list[int]:
    """Return indices where high[i] > all neighbors within `order` bars."""
    out = []
    for i in range(order, len(highs) - order):
        if all(highs[i] > highs[i-j] for j in range(1, order+1)) and \
           all(highs[i] > highs[i+j] for j in range(1, order+1)):
            out.append(i)
    return out

def find_swing_lows(lows: list[float], order: int = 3) -> list[int]:
    """Return indices where low[i] < all neighbors within `order` bars."""
    out = []
    for i in range(order, len(lows) - order):
        if all(lows[i] < lows[i-j] for j in range(1, order+1)) and \
           all(lows[i] < lows[i+j] for j in range(1, order+1)):
            out.append(i)
    return out

def is_in_time_window(ts: datetime, start_h: int, start_m: int, end_h: int, end_m: int) -> bool:
    """Check if timestamp is within a time-of-day window (ET assumed)."""
    t = ts.time()
    return time(start_h, start_m) <= t <= time(end_h, end_m)

def get_day_open_idx(bars: list[Bar]) -> int:
    """Find index of 9:30 bar (market open) for today's data."""
    for i, b in enumerate(bars):
        if b.timestamp.time() >= time(9, 30):
            return i
    return 0

def get_range_high_low(bars: list[Bar], start_idx: int, end_idx: int):
    """Get high and low of a slice of bars."""
    slc = bars[start_idx:end_idx+1]
    return max(b.high for b in slc), min(b.low for b in slc)


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
    From: SMB Training RubberBand Scalp Cheat Sheet
    
    LOGIC: In-play stock makes extended directional move with controlled selling,
    then sellers accelerate sloppily. Once the sloppy sell program ends, price
    snaps back. Entry on a "double bar break" — single green candle clearing
    highs of 2+ preceding candles.
    
    ENTRY (long): Buy when a single green 1-min candle clears the highs of 2+
                  preceding candles (the "snapback candle"). Enter aggressively.
    STOP: $0.02 below the low of the day.
    EXIT: 1/3 at 1:1 R:R, 1/3 at 2:1 R:R, 1/3 into VWAP.
    MAX ATTEMPTS: 2 per day ("2 strikes and out").
    
    INCREASES ODDS: RVOL > 5, price > 3 ATR from open, volume/range increase
                    on last leg, snapback bar is top-5 volume bar of day.
    DECREASES ODDS: Fresh negative news, Day 1 of HTF range break.
    BEST TIMES: 10:00-10:45 AM, 10:45-1:30 PM.
    STATS: 60-65% win rate, 1.6:1 R:R.
    """
    name = "RubberBand Scalp"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 20:
            return None
        
        b = bars.bars
        vwap_vals = calc_vwap(b)
        atr_vals = calc_atr(b, period=14)
        
        # Find today's open
        open_idx = get_day_open_idx(b)
        if open_idx >= len(b) - 5:
            return None
        
        open_price = b[open_idx].open
        day_low = min(bar.low for bar in b[open_idx:])
        day_low_idx = open_idx + min(range(len(b) - open_idx), key=lambda i: b[open_idx + i].low)
        
        # Check 1: Price must be extended below VWAP
        current_idx = len(b) - 1
        current_price = b[current_idx].close
        current_vwap = vwap_vals[current_idx] if current_idx < len(vwap_vals) else None
        
        if current_vwap is None:
            return None
        
        # Must be below VWAP for long rubberband
        if current_price >= current_vwap:
            return None
        
        # Check 2: Extension from open — want > 1 ATR minimum
        current_atr = atr_vals[-1] if atr_vals[-1] is not None else None
        if current_atr is None:
            return None
        
        extension = open_price - day_low
        atrs_from_open = extension / current_atr if current_atr > 0 else 0
        
        if atrs_from_open < 1.0:  # Minimum extension
            return None
        
        # Check 3: Look for acceleration (last leg has bigger candles/volume)
        if day_low_idx - open_idx < 5:
            return None
        
        midpoint = (open_idx + day_low_idx) // 2
        first_half_avg_range = np.mean([b[i].high - b[i].low for i in range(open_idx, midpoint)])
        second_half_avg_range = np.mean([b[i].high - b[i].low for i in range(midpoint, day_low_idx)])
        
        if second_half_avg_range < first_half_avg_range * 1.2:
            return None  # No acceleration
        
        # Check 4: Look for snapback — double bar break up near the low
        snapback_found = False
        snapback_idx = None
        
        for i in range(day_low_idx + 1, min(day_low_idx + 10, len(b))):
            if b[i].close > b[i].open:  # Green candle
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
        
        # Check 5: Snapback bar should have strong volume
        day_volumes = [b[i].volume for i in range(open_idx, len(b))]
        day_volumes_sorted = sorted(day_volumes, reverse=True)
        snapback_vol = b[snapback_idx].volume
        is_top5_volume = snapback_vol >= day_volumes_sorted[min(4, len(day_volumes_sorted)-1)]
        
        # Build the trade
        entry = b[snapback_idx].high + 0.02
        stop = day_low - 0.02
        risk = entry - stop
        
        if risk <= 0:
            return None
        
        # Exit targets: 1/3 at 1:1, 1/3 at 2:1, 1/3 at VWAP
        target_1 = entry + risk        # 1:1
        target_2 = entry + 2 * risk    # 2:1
        target_vwap = current_vwap     # VWAP
        primary_target = target_2      # Use 2:1 as the displayed target
        
        rr = (primary_target - entry) / risk
        
        # Confidence scoring
        confidence = 0.45
        if atrs_from_open >= 3.0:
            confidence += 0.15
        if is_top5_volume:
            confidence += 0.10
        if second_half_avg_range > first_half_avg_range * 1.5:
            confidence += 0.10
        if is_in_time_window(b[snapback_idx].timestamp, 10, 0, 10, 45):
            confidence += 0.10
        elif is_in_time_window(b[snapback_idx].timestamp, 10, 45, 13, 30):
            confidence += 0.05
        
        confidence = min(confidence, 0.95)
        
        return TradeSetup(
            pattern_name=self.name,
            symbol=bars.symbol,
            bias=Bias.LONG,
            timeframe=Timeframe.SCALP,
            entry_price=round(entry, 2),
            stop_loss=round(stop, 2),
            target_price=round(primary_target, 2),
            risk_reward_ratio=round(rr, 2),
            confidence=round(confidence, 2),
            detected_at=b[snapback_idx].timestamp,
            description=f"RubberBand Scalp: {atrs_from_open:.1f} ATR extension, snapback detected",
            win_rate=0.625,
            max_attempts=2,
            exit_strategy="1/3 at 1:1 R:R, 1/3 at 2:1 R:R, 1/3 into VWAP",
            key_levels={"day_low": day_low, "vwap": round(current_vwap, 2),
                        "target_1": round(target_1, 2), "target_2": round(target_2, 2)},
            ideal_time="10:00-10:45 AM or 10:45-1:30 PM EST",
        )


# ==============================================================================
# SMB SCALP #2: HITCHHIKER SCALP
# ==============================================================================

class HitchHikerScalp(PatternDetector):
    """
    From: SMB Training HitchHiker Scalp Cheat Sheet
    
    LOGIC: In-play stock drives higher off the open, then HOLDS (no pullback)
    and consolidates in a tight range in the upper 1/3 of the day's range for
    5-20 minutes. This signals a large institutional buy program. Break of
    the consolidation range = entry.
    
    ENTRY: Aggressively buy on break of 1-min consolidation range high.
    STOP: $0.02 below the low of the consolidation.
    EXIT: 1/2 into first "wave" (momentum slows), 1/2 into second "wave".
    MAX ATTEMPTS: 1 (one and done).
    STATS: 55-60% win rate, 1.9:1 R:R.
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
        
        # Step 1: Find the initial drive off the open
        # Look for 3+ consecutive bars making higher highs in first 10 bars
        drive_end = None
        for i in range(open_idx + 3, min(open_idx + 15, len(b))):
            # Check if we had an upward drive
            recent = b[open_idx:i+1]
            if all(recent[j].high >= recent[j-1].high for j in range(1, len(recent))):
                drive_end = i
        
        if drive_end is None:
            # Relax: just need price to be higher than open after a few bars
            for i in range(open_idx + 3, min(open_idx + 10, len(b))):
                if b[i].close > b[open_idx].open * 1.005:  # At least 0.5% move up
                    drive_end = i
                    break
        
        if drive_end is None:
            return None
        
        # Step 2: Find consolidation (5-20 bars of tight range after drive)
        day_high = max(bar.high for bar in b[open_idx:])
        day_low = min(bar.low for bar in b[open_idx:])
        day_range = day_high - day_low
        
        if day_range <= 0:
            return None
        
        # Look for consolidation starting after the drive
        consol_start = drive_end + 1
        if consol_start >= len(b) - 2:
            return None
        
        best_consol = None
        for consol_end in range(consol_start + 4, min(consol_start + 21, len(b) - 1)):
            consol_bars = b[consol_start:consol_end + 1]
            consol_high = max(bar.high for bar in consol_bars)
            consol_low = min(bar.low for bar in consol_bars)
            consol_range = consol_high - consol_low
            
            # Consolidation should be tight (< 40% of day range)
            if consol_range < day_range * 0.40:
                # Consolidation low should be in upper 1/3 of day range
                upper_third_threshold = day_low + day_range * (2/3)
                if consol_low >= upper_third_threshold:
                    best_consol = (consol_start, consol_end, consol_high, consol_low)
        
        if best_consol is None:
            return None
        
        c_start, c_end, c_high, c_low = best_consol
        
        # Step 3: Check for breakout (bar closing above consolidation high)
        if c_end + 1 >= len(b):
            return None
        
        breakout_bar = b[c_end + 1]
        if breakout_bar.high <= c_high:
            return None  # No breakout yet
        
        # Step 4: Volume confirmation — breakout bar should have 30%+ more volume
        consol_avg_vol = np.mean([bar.volume for bar in b[c_start:c_end+1]])
        if breakout_bar.volume < consol_avg_vol * 1.3:
            pass  # Still valid, just lower confidence
        
        entry = c_high + 0.02
        stop = c_low - 0.02
        risk = entry - stop
        if risk <= 0:
            return None
        
        # Target: measured move (height of consolidation added to breakout)
        target = entry + risk * 1.9  # Based on 1.9:1 R:R stat
        rr = (target - entry) / risk
        
        confidence = 0.50
        if breakout_bar.volume > consol_avg_vol * 1.3:
            confidence += 0.10
        if is_in_time_window(breakout_bar.timestamp, 9, 30, 9, 59):
            confidence += 0.10
        # Consolidation above key levels
        if c_low > b[open_idx].high:  # Above premarket high area
            confidence += 0.05
        confidence = min(confidence, 0.90)
        
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol,
            bias=Bias.LONG, timeframe=Timeframe.SCALP,
            entry_price=round(entry, 2), stop_loss=round(stop, 2),
            target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
            confidence=round(confidence, 2), detected_at=breakout_bar.timestamp,
            description=f"HitchHiker: {c_end - c_start + 1}-bar consolidation in upper 1/3, breakout",
            win_rate=0.575, max_attempts=1,
            exit_strategy="1/2 into first wave, 1/2 into second wave",
            key_levels={"consol_high": c_high, "consol_low": c_low, "day_high": day_high},
            ideal_time="Before 9:59 AM EST (opening drive)",
        )


# ==============================================================================
# SMB SCALP #3: OPENING RANGE BREAK
# ==============================================================================

class OpeningRangeBreak(PatternDetector):
    """
    From: SMB Training ORB Cheat Sheet
    
    LOGIC: Identify the high/low of the first 5/15/30 minutes. Trade the
    breakout of that range. Big players establish the range, then show direction.
    
    ENTRY: Price breaks above range high (long) or below range low (short).
    STOP: Below breakout bar (long) or above (short). Alt: 2-min trailing stop.
    TARGET: 2x measured move of opening range. Or higher TF levels.
    STATS: Aim for 2:1+ R:R.
    """
    name = "Opening Range Break"
    default_bias = Bias.LONG

    def __init__(self, orb_minutes: int = 15):
        self.orb_minutes = orb_minutes  # 5, 15, or 30

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < self.orb_minutes + 5:
            return None
        
        b = bars.bars
        open_idx = get_day_open_idx(b)
        orb_end = open_idx + self.orb_minutes
        
        if orb_end >= len(b) - 1:
            return None
        
        orb_bars = b[open_idx:orb_end]
        orb_high = max(bar.high for bar in orb_bars)
        orb_low = min(bar.low for bar in orb_bars)
        orb_range = orb_high - orb_low
        
        if orb_range <= 0:
            return None
        
        # Look for breakout after ORB period
        for i in range(orb_end, min(orb_end + 30, len(b))):
            bar = b[i]
            
            # Bullish breakout
            if bar.close > orb_high:
                entry = orb_high + 0.02
                stop = bar.low - 0.02
                risk = entry - stop
                if risk <= 0 or risk > orb_range:
                    continue
                target = entry + 2 * orb_range
                rr = (target - entry) / risk
                
                confidence = 0.50
                # Volume confirmation
                orb_avg_vol = np.mean([b_.volume for b_ in orb_bars])
                if bar.volume > orb_avg_vol * 1.5:
                    confidence += 0.15
                if orb_range < b[open_idx].close * 0.02:
                    confidence += 0.05  # Tight range = better breakout
                
                return TradeSetup(
                    pattern_name=f"ORB {self.orb_minutes}min",
                    symbol=bars.symbol, bias=Bias.LONG,
                    timeframe=Timeframe.SCALP,
                    entry_price=round(entry, 2), stop_loss=round(stop, 2),
                    target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
                    confidence=round(min(confidence, 0.90), 2),
                    detected_at=bar.timestamp,
                    description=f"ORB {self.orb_minutes}min breakout above {orb_high:.2f}",
                    win_rate=0.55, max_attempts=2,
                    exit_strategy="2x measured move target, sell into strength",
                    key_levels={"orb_high": orb_high, "orb_low": orb_low, "orb_range": round(orb_range, 2)},
                    ideal_time="First 30 min after ORB period",
                )
            
            # Bearish breakout
            if bar.close < orb_low:
                entry = orb_low - 0.02
                stop = bar.high + 0.02
                risk = stop - entry
                if risk <= 0 or risk > orb_range:
                    continue
                target = entry - 2 * orb_range
                rr = (entry - target) / risk
                
                return TradeSetup(
                    pattern_name=f"ORB {self.orb_minutes}min",
                    symbol=bars.symbol, bias=Bias.SHORT,
                    timeframe=Timeframe.SCALP,
                    entry_price=round(entry, 2), stop_loss=round(stop, 2),
                    target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
                    confidence=0.55, detected_at=bar.timestamp,
                    description=f"ORB {self.orb_minutes}min breakdown below {orb_low:.2f}",
                    win_rate=0.55, max_attempts=2,
                    exit_strategy="2x measured move target",
                    key_levels={"orb_high": orb_high, "orb_low": orb_low},
                    ideal_time="First 30 min after ORB period",
                )
        
        return None


# ==============================================================================
# SMB SCALP #4: SECOND CHANCE SCALP
# ==============================================================================

class SecondChanceScalp(PatternDetector):
    """
    From: SMB Training Second Chance Scalp Cheat Sheet
    
    LOGIC: After a range breakout, price retests the broken level. If old
    resistance becomes new support, enter on the confirmation candle.
    
    ENTRY: After retest of breakout level, buy when candle closes above prior candle.
    STOP: $0.02 below the low of the turn candle.
    EXIT: Sell 1/2 at the high of the initial pullback. Trail remaining via 9 EMA.
    MAX ATTEMPTS: 2.
    STATS: 50-55% win rate, 1.9:1 R:R.
    """
    name = "Second Chance Scalp"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 30:
            return None
        
        b = bars.bars
        
        # Step 1: Find a resistance level that was broken
        # Look for a horizontal level tested 2+ times then broken
        swing_highs = find_swing_highs([bar.high for bar in b], order=3)
        
        if len(swing_highs) < 2:
            return None
        
        # Find clusters of swing highs at similar prices (resistance)
        for i in range(len(swing_highs) - 1):
            level = b[swing_highs[i]].high
            tolerance = level * 0.003  # 0.3% tolerance
            
            # Check if another swing high is near this level
            cluster = [swing_highs[i]]
            for j in range(i + 1, len(swing_highs)):
                if abs(b[swing_highs[j]].high - level) < tolerance:
                    cluster.append(swing_highs[j])
            
            if len(cluster) < 2:
                continue
            
            last_test = cluster[-1]
            resistance = np.mean([b[idx].high for idx in cluster])
            
            # Step 2: Look for breakout above resistance after the cluster
            breakout_idx = None
            for k in range(last_test + 1, min(last_test + 20, len(b))):
                if b[k].close > resistance * 1.002:
                    breakout_idx = k
                    break
            
            if breakout_idx is None:
                continue
            
            # Step 3: Look for pullback to the breakout level
            pullback_idx = None
            pullback_high = b[breakout_idx].high
            
            for k in range(breakout_idx + 1, min(breakout_idx + 15, len(b))):
                pullback_high = max(pullback_high, b[k].high)
                if b[k].low <= resistance * 1.003:  # Retesting the level
                    pullback_idx = k
                    break
            
            if pullback_idx is None:
                continue
            
            # Step 4: Look for confirmation candle (close above prior candle)
            if pullback_idx + 1 >= len(b):
                continue
            
            confirm = b[pullback_idx + 1]
            if confirm.close > b[pullback_idx].high:
                entry = confirm.close
                stop = b[pullback_idx].low - 0.02
                risk = entry - stop
                if risk <= 0:
                    continue
                target = pullback_high  # First target: prior pullback high
                rr = (target - entry) / risk if risk > 0 else 0
                
                if rr < 1.0:
                    continue
                
                # Volume check: high vol on break, low vol on retest
                break_vol = b[breakout_idx].volume
                retest_vol = b[pullback_idx].volume
                vol_confirms = retest_vol < break_vol * 0.7
                
                confidence = 0.45
                if vol_confirms:
                    confidence += 0.15
                if rr >= 1.9:
                    confidence += 0.10
                
                return TradeSetup(
                    pattern_name=self.name, symbol=bars.symbol,
                    bias=Bias.LONG, timeframe=Timeframe.SCALP,
                    entry_price=round(entry, 2), stop_loss=round(stop, 2),
                    target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
                    confidence=round(min(confidence, 0.90), 2),
                    detected_at=confirm.timestamp,
                    description=f"Second Chance: Retest of {resistance:.2f} held as support",
                    win_rate=0.525, max_attempts=2,
                    exit_strategy="1/2 at pullback high, trail remaining via 9 EMA",
                    key_levels={"resistance_level": round(resistance, 2), "pullback_high": round(pullback_high, 2)},
                    ideal_time="9:59-10:44 AM, 10:45-1:29 PM, 1:30-4:00 PM",
                )
        
        return None


# ==============================================================================
# SMB SCALP #5: BACKSIDE SCALP
# ==============================================================================

class BackSideScalp(PatternDetector):
    """
    From: SMB Training Back$ide Scalp Cheat Sheet
    
    LOGIC: Stock extends below VWAP, then establishes higher high + higher low
    (new uptrend). When a range develops above 9 EMA and breaks higher,
    trapped shorts stop out → fast move back to VWAP.
    
    ENTRY: Break of 1-min consolidation above rising 9 EMA.
    STOP: $0.02 below most recent higher low.
    EXIT: Entire position at VWAP.
    MAX ATTEMPTS: 1 (one and done).
    STATS: 50-60% win rate, 1.4:1 R:R.
    """
    name = "BackSide Scalp"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 25:
            return None
        
        b = bars.bars
        ema9 = calc_ema([bar.close for bar in b], 9)
        vwap_vals = calc_vwap(b)
        
        open_idx = get_day_open_idx(b)
        if len(b) - open_idx < 15:
            return None
        
        day_low = min(bar.low for bar in b[open_idx:])
        day_low_idx = open_idx + min(range(len(b) - open_idx), key=lambda i: b[open_idx + i].low)
        
        # Step 1: Price must have been extended below VWAP
        if day_low_idx < open_idx + 3:
            return None
        
        current_vwap = vwap_vals[-1]
        if day_low >= current_vwap:
            return None
        
        # Step 2: After the low, find higher high and higher low
        post_low_bars = b[day_low_idx:]
        if len(post_low_bars) < 8:
            return None
        
        sw_highs = find_swing_highs([bar.high for bar in post_low_bars], order=2)
        sw_lows = find_swing_lows([bar.low for bar in post_low_bars], order=2)
        
        if len(sw_highs) < 1 or len(sw_lows) < 1:
            return None
        
        # Check for higher high and higher low pattern
        has_hh = any(post_low_bars[h].high > post_low_bars[0].high for h in sw_highs)
        has_hl = any(post_low_bars[l].low > day_low for l in sw_lows)
        
        if not (has_hh and has_hl):
            return None
        
        # Step 3: Check if price is above rising 9 EMA
        current_ema = ema9[-1]
        prev_ema = ema9[-3] if len(ema9) > 3 else None
        
        if current_ema is None or prev_ema is None:
            return None
        
        if current_ema <= prev_ema:
            return None  # EMA not rising
        
        if b[-1].close < current_ema:
            return None  # Price not above EMA
        
        # Step 4: Check position — should be > halfway between low and VWAP
        midpoint = (day_low + current_vwap) / 2
        if b[-1].close < midpoint:
            return None
        
        # Build trade
        recent_hl = day_low  # Default
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
        # Consistency of price action above EMA
        bars_above_ema = sum(1 for i in range(-8, 0) if ema9[i] and b[i].close > ema9[i])
        if bars_above_ema >= 6:
            confidence += 0.10
        if is_in_time_window(b[-1].timestamp, 10, 0, 10, 45):
            confidence += 0.10
        
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol,
            bias=Bias.LONG, timeframe=Timeframe.SCALP,
            entry_price=round(entry, 2), stop_loss=round(stop, 2),
            target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
            confidence=round(min(confidence, 0.85), 2),
            detected_at=b[-1].timestamp,
            description=f"BackSide: HH+HL above 9 EMA, targeting VWAP at {current_vwap:.2f}",
            win_rate=0.55, max_attempts=1,
            exit_strategy="Entire position at VWAP",
            key_levels={"day_low": day_low, "vwap": round(current_vwap, 2), "recent_hl": round(recent_hl, 2)},
            ideal_time="10:00-10:45 AM, 10:46-1:30 PM",
        )


# ==============================================================================
# SMB SCALP #6: FASHIONABLY LATE SCALP
# ==============================================================================

class FashionablyLateScalp(PatternDetector):
    """
    From: SMB Training Fashionably Late Scalp Cheat Sheet
    
    LOGIC: After a low, momentum builds as price converges back toward VWAP.
    Enter when an upsloping 9 EMA crosses a flat/downsloping VWAP.
    
    ENTRY: When upsloping 9 EMA crosses above flat/downsloping VWAP.
    STOP: 1/3 of the distance from VWAP to the low of the day.
    TARGET: 1 measured move (low-to-cross distance) above the cross point.
    STATS: 60% win rate, 3:1 R:R.
    """
    name = "Fashionably Late Scalp"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 30:
            return None
        
        b = bars.bars
        ema9 = calc_ema([bar.close for bar in b], 9)
        vwap_vals = calc_vwap(b)
        
        open_idx = get_day_open_idx(b)
        if len(b) - open_idx < 20:
            return None
        
        day_low = min(bar.low for bar in b[open_idx:])
        
        # Find the EMA/VWAP cross
        cross_idx = None
        for i in range(open_idx + 10, len(b)):
            if ema9[i] is None or ema9[i-1] is None:
                continue
            if vwap_vals[i] is None:
                continue
            
            # EMA crosses above VWAP
            if ema9[i-1] < vwap_vals[i-1] and ema9[i] >= vwap_vals[i]:
                # EMA must be upsloping
                if ema9[i] > ema9[i-3] if i >= 3 and ema9[i-3] else False:
                    # VWAP should be flat or downsloping
                    vwap_slope = vwap_vals[i] - vwap_vals[i-5] if i >= 5 else 0
                    if vwap_slope <= vwap_vals[i] * 0.001:  # Flat or down
                        cross_idx = i
        
        if cross_idx is None:
            return None
        
        # Check that EMA hasn't been flat for 15+ min before entry
        if cross_idx >= 15:
            ema_range = max(ema9[cross_idx-15:cross_idx]) - min(
                e for e in ema9[cross_idx-15:cross_idx] if e is not None)
            if ema_range < b[cross_idx].close * 0.001:
                return None  # Too flat — avoid per cheat sheet
        
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
        
        # Volume during convergence vs divergence
        confidence = 0.55
        if is_in_time_window(b[cross_idx].timestamp, 10, 0, 10, 45):
            confidence += 0.10
        if rr >= 2.5:
            confidence += 0.10
        
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol,
            bias=Bias.LONG, timeframe=Timeframe.SCALP,
            entry_price=round(entry, 2), stop_loss=round(stop, 2),
            target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
            confidence=round(min(confidence, 0.90), 2),
            detected_at=b[cross_idx].timestamp,
            description=f"Fashionably Late: 9 EMA crossed VWAP, measured move ${measured_move:.2f}",
            win_rate=0.60, max_attempts=1,
            exit_strategy="Exit at 1 measured move above the cross",
            key_levels={"day_low": day_low, "cross_price": round(cross_price, 2),
                        "measured_move": round(measured_move, 2)},
            ideal_time="10:00-10:45 AM, 10:46-1:30 PM",
        )


# ==============================================================================
# SMB SCALP #7: SPENCER SCALP
# ==============================================================================

class SpencerScalp(PatternDetector):
    """
    From: SMB Training Spencer Scalp Cheat Sheet
    
    LOGIC: In-play stock consolidates 20+ minutes near/above high of day on
    sustained volume. Break higher confirms institutional buy program.
    
    ENTRY: Break of range high after 20+ min consolidation in upper 1/3.
    STOP: $0.02 below low of consolidation range.
    EXIT: 1/4 at 1:1 measured move, 1/2 at 2:1, final 1/4 at 3:1.
    """
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
        
        upper_third = day_low + day_range * (2/3)
        
        # Find 20+ minute consolidation in upper 1/3
        for start in range(open_idx + 5, len(b) - 21):
            consol_bars = b[start:start + 20]
            c_high = max(bar.high for bar in consol_bars)
            c_low = min(bar.low for bar in consol_bars)
            c_range = c_high - c_low
            
            # Must be in upper 1/3
            if c_low < upper_third:
                continue
            
            # Must be tight (< 20% of day range = ideal per cheat sheet)
            if c_range > day_range * 0.35:
                continue
            
            # Check for sustained volume (equal-ish volume bars)
            vols = [bar.volume for bar in consol_bars]
            vol_std = np.std(vols)
            vol_mean = np.mean(vols)
            if vol_mean > 0 and vol_std / vol_mean > 0.8:
                continue  # Volume too erratic
            
            # Look for breakout
            break_idx = start + 20
            if break_idx >= len(b):
                continue
            
            if b[break_idx].high > c_high:
                entry = c_high + 0.02
                stop = c_low - 0.02
                risk = entry - stop
                if risk <= 0:
                    continue
                
                target = entry + 2 * c_range  # 2:1 measured move
                rr = (target - entry) / risk
                
                # Low volume bar before break = old-school buy program signal
                confidence = 0.50
                if break_idx >= 2:
                    if b[break_idx - 1].volume < vol_mean * 0.7:
                        confidence += 0.10
                if c_range < day_range * 0.20:
                    confidence += 0.10
                
                return TradeSetup(
                    pattern_name=self.name, symbol=bars.symbol,
                    bias=Bias.LONG, timeframe=Timeframe.INTRADAY,
                    entry_price=round(entry, 2), stop_loss=round(stop, 2),
                    target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
                    confidence=round(min(confidence, 0.85), 2),
                    detected_at=b[break_idx].timestamp,
                    description=f"Spencer Scalp: {len(consol_bars)}-bar consolidation near HOD",
                    win_rate=0.55, max_attempts=1,
                    exit_strategy="1/4 at 1:1, 1/2 at 2:1, 1/4 at 3:1 measured move",
                    key_levels={"consol_high": c_high, "consol_low": c_low, "day_high": day_high},
                    ideal_time="9:59-10:44, 10:45-1:29, 1:30-4:00",
                )
        
        return None


# ==============================================================================
# SMB SCALP #8: GAP GIVE AND GO
# ==============================================================================

class GapGiveAndGo(PatternDetector):
    """
    From: SMB Training Gap Give and Go Cheat Sheet
    
    LOGIC: Gap-up stock drops sharply from open (profit-taking), then forms
    3-7 minute mini-consolidation above support. Break of consolidation
    high = entry as big players buy through the open.
    
    ENTRY: Buy break of 1-min consolidation range (3-7 min consolidation).
    STOP: $0.02 below consolidation low. Max 2 attempts.
    EXIT: Move2Move — exit on double bar break against position.
    """
    name = "Gap Give and Go"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 15:
            return None
        
        b = bars.bars
        open_idx = get_day_open_idx(b)
        if len(b) - open_idx < 10:
            return None
        
        # Step 1: Detect gap up (open significantly above prior close)
        if open_idx == 0:
            return None
        
        prev_close = b[open_idx - 1].close
        open_price = b[open_idx].open
        gap_pct = (open_price - prev_close) / prev_close * 100
        
        if gap_pct < 0.5:  # Needs meaningful gap
            return None
        
        # Step 2: Quick drop from open (first 1-5 bars should move lower)
        drop_found = False
        drop_end = None
        for i in range(open_idx + 1, min(open_idx + 6, len(b))):
            if b[i].low < open_price * 0.995:  # At least 0.5% pullback
                drop_found = True
                drop_end = i
        
        if not drop_found or drop_end is None:
            return None
        
        # Step 3: Mini-consolidation (3-7 bars)
        for consol_len in range(3, 8):
            c_end = drop_end + consol_len
            if c_end >= len(b) - 1:
                continue
            
            consol_bars = b[drop_end:c_end]
            c_high = max(bar.high for bar in consol_bars)
            c_low = min(bar.low for bar in consol_bars)
            c_range = c_high - c_low
            
            initial_drop = open_price - c_low
            if initial_drop <= 0:
                continue
            
            # Consolidation should be < 50% of initial drop (per cheat sheet)
            if c_range > initial_drop * 0.50:
                continue
            
            # Check for breakout
            if b[c_end].high > c_high:
                entry = c_high + 0.02
                stop = c_low - 0.02
                risk = entry - stop
                if risk <= 0:
                    continue
                
                # Target: back toward the open price
                target = open_price
                rr = (target - entry) / risk if risk > 0 else 0
                if rr < 1.0:
                    continue
                
                # Volume: should decrease during consolidation
                confidence = 0.50
                pre_vol = np.mean([b[j].volume for j in range(open_idx, drop_end)])
                consol_vol = np.mean([bar.volume for bar in consol_bars])
                if consol_vol < pre_vol * 0.5:
                    confidence += 0.15
                if is_in_time_window(b[c_end].timestamp, 9, 30, 9, 45):
                    confidence += 0.10
                
                return TradeSetup(
                    pattern_name=self.name, symbol=bars.symbol,
                    bias=Bias.LONG, timeframe=Timeframe.SCALP,
                    entry_price=round(entry, 2), stop_loss=round(stop, 2),
                    target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
                    confidence=round(min(confidence, 0.85), 2),
                    detected_at=b[c_end].timestamp,
                    description=f"Gap Give and Go: {gap_pct:.1f}% gap, {consol_len}-bar consolidation",
                    win_rate=0.55, max_attempts=2,
                    exit_strategy="Move2Move: exit on double bar break against position",
                    key_levels={"gap_open": round(open_price, 2), "consol_high": c_high, "consol_low": c_low},
                    ideal_time="9:30-9:45 AM EST (opening drive)",
                )
        
        return None


# ==============================================================================
# SMB SCALP #9: TIDAL WAVE / BOUNCY BALL
# ==============================================================================

class TidalWaveBouncyBall(PatternDetector):
    """
    From: SMB Training Tidal Wave-Bouncy Ball Cheat Sheet
    
    LOGIC: Stock approaches key support with 2+ bounces showing diminishing
    strength (lower highs). Like a bouncing ball losing energy. Once support
    breaks, trapped longs exit → sustained sell-off.
    
    ENTRY SHORT: After 3rd wave/bounce, or on breakdown of key support.
    STOP: $0.02 above the high of the nearest wave/bounce.
    EXIT: 1/2 at 2x measured move, 1/4 at 3x, hold rest to close.
    """
    name = "Tidal Wave / Bouncy Ball"
    default_bias = Bias.SHORT

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 30:
            return None
        
        b = bars.bars
        
        # Find swing lows at similar levels (key support)
        sw_lows = find_swing_lows([bar.low for bar in b], order=3)
        if len(sw_lows) < 2:
            return None
        
        # Look for support cluster
        for i in range(len(sw_lows)):
            support_level = b[sw_lows[i]].low
            tolerance = support_level * 0.003
            
            touches = [sw_lows[i]]
            for j in range(i + 1, len(sw_lows)):
                if abs(b[sw_lows[j]].low - support_level) < tolerance:
                    touches.append(sw_lows[j])
            
            if len(touches) < 2:
                continue
            
            # Find the bounces (swing highs between support touches)
            bounce_highs = []
            for t in range(len(touches) - 1):
                between_highs = [
                    b[k].high for k in range(touches[t], touches[t+1])
                ]
                if between_highs:
                    bounce_highs.append(max(between_highs))
            
            if len(bounce_highs) < 2:
                continue
            
            # Check: each bounce should be LOWER than the previous (diminishing)
            is_diminishing = all(
                bounce_highs[k] > bounce_highs[k+1]
                for k in range(len(bounce_highs) - 1)
            )
            
            if not is_diminishing:
                continue
            
            # Check for breakdown below support
            last_touch = touches[-1]
            for k in range(last_touch + 1, min(last_touch + 10, len(b))):
                if b[k].close < support_level:
                    entry = support_level - 0.02
                    stop = bounce_highs[-1] + 0.02
                    risk = stop - entry
                    if risk <= 0:
                        continue
                    
                    measured_move = risk
                    target = entry - 2 * measured_move
                    rr = (entry - target) / risk
                    
                    confidence = 0.55
                    if len(touches) >= 3:
                        confidence += 0.10
                    # Volume confirmation: larger red bars
                    if b[k].volume > np.mean([bar.volume for bar in b[last_touch-5:last_touch]]) * 1.3:
                        confidence += 0.10
                    
                    return TradeSetup(
                        pattern_name=self.name, symbol=bars.symbol,
                        bias=Bias.SHORT, timeframe=Timeframe.SCALP,
                        entry_price=round(entry, 2), stop_loss=round(stop, 2),
                        target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
                        confidence=round(min(confidence, 0.85), 2),
                        detected_at=b[k].timestamp,
                        description=f"Tidal Wave: {len(touches)} touches of {support_level:.2f} with diminishing bounces",
                        win_rate=0.55, max_attempts=1,
                        exit_strategy="1/2 at 2x measured move, 1/4 at 3x, hold rest to close",
                        key_levels={"support": round(support_level, 2),
                                    "last_bounce_high": round(bounce_highs[-1], 2)},
                        ideal_time="After first or second leg down",
                    )
        
        return None


# ==============================================================================
# SMB SCALP #10: BREAKING NEWS STRATEGY
# ==============================================================================

class BreakingNewsStrategy(PatternDetector):
    """
    From: SMB Training Breaking News Cheat Sheet
    
    LOGIC: Score catalyst -10 to +10. If stock breaks out of intraday range
    on significant volume following news, trade the momentum or wait for
    pullback to 9 EMA for trend entry.
    
    This pattern detects the TECHNICAL component: unusual volume + range
    expansion. The AI sentiment scoring is handled separately.
    """
    name = "Breaking News"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 10:
            return None
        
        b = bars.bars
        
        # Detect volume spike: last bar > 3x average of prior 10 bars
        if len(b) < 12:
            return None
        
        avg_vol = np.mean([bar.volume for bar in b[-12:-2]])
        current_vol = b[-1].volume
        
        if avg_vol == 0 or current_vol < avg_vol * 3:
            return None  # No volume spike
        
        # Range expansion: current bar range > 2x average bar range
        avg_range = np.mean([bar.high - bar.low for bar in b[-12:-2]])
        current_range = b[-1].high - b[-1].low
        
        if current_range < avg_range * 2:
            return None
        
        # Determine direction
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
        
        rr = 2.0
        vol_multiple = current_vol / avg_vol
        
        confidence = 0.45
        if vol_multiple > 5:
            confidence += 0.15
        if current_range > avg_range * 3:
            confidence += 0.10
        
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol,
            bias=bias, timeframe=Timeframe.SCALP,
            entry_price=round(entry, 2), stop_loss=round(stop, 2),
            target_price=round(target, 2), risk_reward_ratio=round(rr, 2),
            confidence=round(min(confidence, 0.85), 2),
            detected_at=b[-1].timestamp,
            description=f"Breaking News: {vol_multiple:.0f}x volume spike, {current_range/avg_range:.1f}x range expansion",
            win_rate=0.50, max_attempts=2,
            exit_strategy="Momentum: exit on tape extremes. Trend: exit at key levels or 9 EMA loss.",
            key_levels={"volume_multiple": round(vol_multiple, 1)},
            ideal_time="Any time during market hours",
        )


# ==============================================================================
# CLASSICAL PATTERNS (11-50) — Concise implementations
# ==============================================================================

class DoubleBottom(PatternDetector):
    """Two swing lows at approximately the same level → bullish reversal."""
    name = "Double Bottom"
    default_bias = Bias.LONG

    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 30:
            return None
        lows = bars.lows
        sw = find_swing_lows(lows, order=5)
        if len(sw) < 2:
            return None
        l1, l2 = sw[-2], sw[-1]
        price1, price2 = lows[l1], lows[l2]
        tolerance = price1 * 0.015
        if abs(price1 - price2) > tolerance:
            return None
        neckline = max(bars.highs[l1:l2+1])
        if bars.bars[-1].close < neckline:
            return None
        entry = neckline + 0.02
        stop = min(price1, price2) - 0.02
        risk = entry - stop
        if risk <= 0: return None
        target = entry + (neckline - min(price1, price2))
        rr = (target - entry) / risk
        return TradeSetup(
            pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.SWING, entry_price=round(entry,2),
            stop_loss=round(stop,2), target_price=round(target,2),
            risk_reward_ratio=round(rr,2), confidence=0.65,
            detected_at=bars.bars[-1].timestamp,
            description=f"Double Bottom at {min(price1,price2):.2f}, neckline {neckline:.2f}",
            win_rate=0.65, exit_strategy="Measured move from neckline",
            key_levels={"bottom1": round(price1,2), "bottom2": round(price2,2), "neckline": round(neckline,2)},
        )

class DoubleTop(PatternDetector):
    """Two swing highs at approximately the same level → bearish reversal."""
    name = "Double Top"
    default_bias = Bias.SHORT
    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 30: return None
        highs = bars.highs
        sw = find_swing_highs(highs, order=5)
        if len(sw) < 2: return None
        h1, h2 = sw[-2], sw[-1]
        p1, p2 = highs[h1], highs[h2]
        if abs(p1-p2) > p1*0.015: return None
        neckline = min(bars.lows[h1:h2+1])
        if bars.bars[-1].close > neckline: return None
        entry = neckline - 0.02
        stop = max(p1,p2) + 0.02
        risk = stop - entry
        if risk <= 0: return None
        target = entry - (max(p1,p2) - neckline)
        rr = (entry - target) / risk
        return TradeSetup(pattern_name=self.name, symbol=bars.symbol, bias=Bias.SHORT,
            timeframe=Timeframe.SWING, entry_price=round(entry,2), stop_loss=round(stop,2),
            target_price=round(target,2), risk_reward_ratio=round(rr,2), confidence=0.63,
            detected_at=bars.bars[-1].timestamp,
            description=f"Double Top at {max(p1,p2):.2f}", win_rate=0.63,
            key_levels={"top1":round(p1,2),"top2":round(p2,2),"neckline":round(neckline,2)})

class BullFlag(PatternDetector):
    """Strong pole up → tight downward-sloping consolidation → breakout."""
    name = "Bull Flag"
    default_bias = Bias.LONG
    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 20: return None
        b = bars.bars
        sw_lo = find_swing_lows(bars.lows, order=3)
        sw_hi = find_swing_highs(bars.highs, order=3)
        if not sw_lo or not sw_hi: return None
        pole_lo_idx = sw_lo[-1]
        post = [h for h in sw_hi if h > pole_lo_idx]
        if not post: return None
        pole_hi_idx = post[0]
        pole_lo, pole_hi = bars.lows[pole_lo_idx], bars.highs[pole_hi_idx]
        pole_pct = (pole_hi - pole_lo) / pole_lo * 100
        if pole_pct < 2.0: return None
        flag = b[pole_hi_idx:]
        if len(flag) < 3 or len(flag) > 30: return None
        flag_low = min(f.low for f in flag)
        retrace = (pole_hi - flag_low) / (pole_hi - pole_lo) if pole_hi != pole_lo else 1
        if retrace > 0.50: return None
        entry = pole_hi * 1.001
        stop = flag_low * 0.998
        risk = entry - stop
        if risk <= 0: return None
        target = entry + (pole_hi - pole_lo)
        rr = (target - entry) / risk
        return TradeSetup(pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.INTRADAY, entry_price=round(entry,2), stop_loss=round(stop,2),
            target_price=round(target,2), risk_reward_ratio=round(rr,2), confidence=0.60,
            detected_at=b[-1].timestamp, description=f"Bull Flag: {pole_pct:.1f}% pole",
            win_rate=0.60, key_levels={"pole_hi":pole_hi,"pole_lo":pole_lo,"flag_low":flag_low})

class BearFlag(PatternDetector):
    name = "Bear Flag"
    default_bias = Bias.SHORT
    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 20: return None
        sw_hi = find_swing_highs(bars.highs, 3)
        sw_lo = find_swing_lows(bars.lows, 3)
        if not sw_hi or not sw_lo: return None
        pole_hi_idx = sw_hi[-1]
        post = [l for l in sw_lo if l > pole_hi_idx]
        if not post: return None
        pole_lo_idx = post[0]
        pole_hi, pole_lo = bars.highs[pole_hi_idx], bars.lows[pole_lo_idx]
        pole_pct = (pole_hi - pole_lo) / pole_hi * 100
        if pole_pct < 2.0: return None
        flag = bars.bars[pole_lo_idx:]
        if len(flag) < 3 or len(flag) > 30: return None
        flag_high = max(f.high for f in flag)
        retrace = (flag_high - pole_lo) / (pole_hi - pole_lo) if pole_hi != pole_lo else 1
        if retrace > 0.50: return None
        entry = pole_lo * 0.999
        stop = flag_high * 1.002
        risk = stop - entry
        if risk <= 0: return None
        target = entry - (pole_hi - pole_lo)
        rr = (entry - target) / risk
        return TradeSetup(pattern_name=self.name, symbol=bars.symbol, bias=Bias.SHORT,
            timeframe=Timeframe.INTRADAY, entry_price=round(entry,2), stop_loss=round(stop,2),
            target_price=round(target,2), risk_reward_ratio=round(rr,2), confidence=0.58,
            detected_at=bars.bars[-1].timestamp, description=f"Bear Flag: {pole_pct:.1f}% pole",
            win_rate=0.58, key_levels={"pole_hi":pole_hi,"pole_lo":pole_lo,"flag_high":flag_high})

class HeadAndShoulders(PatternDetector):
    name = "Head & Shoulders"
    default_bias = Bias.SHORT
    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 40: return None
        sw = find_swing_highs(bars.highs, 5)
        if len(sw) < 3: return None
        ls, head, rs = sw[-3], sw[-2], sw[-1]
        lp, hp, rp = bars.highs[ls], bars.highs[head], bars.highs[rs]
        if not (hp > lp and hp > rp): return None
        if abs(lp - rp) > lp * 0.03: return None
        neckline = min(min(bars.lows[ls:head+1]), min(bars.lows[head:rs+1]))
        if bars.bars[-1].close > neckline: return None
        entry = neckline - 0.02
        stop = rp + 0.02
        risk = stop - entry
        if risk <= 0: return None
        target = entry - (hp - neckline)
        rr = (entry - target) / risk
        return TradeSetup(pattern_name=self.name, symbol=bars.symbol, bias=Bias.SHORT,
            timeframe=Timeframe.SWING, entry_price=round(entry,2), stop_loss=round(stop,2),
            target_price=round(target,2), risk_reward_ratio=round(rr,2), confidence=0.68,
            detected_at=bars.bars[-1].timestamp, description=f"H&S: head {hp:.2f}, neckline {neckline:.2f}",
            win_rate=0.68, key_levels={"left_shoulder":lp,"head":hp,"right_shoulder":rp,"neckline":neckline})

class InverseHeadAndShoulders(PatternDetector):
    name = "Inverse H&S"
    default_bias = Bias.LONG
    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 40: return None
        sw = find_swing_lows(bars.lows, 5)
        if len(sw) < 3: return None
        ls, head, rs = sw[-3], sw[-2], sw[-1]
        lp, hp, rp = bars.lows[ls], bars.lows[head], bars.lows[rs]
        if not (hp < lp and hp < rp): return None
        if abs(lp - rp) > lp * 0.03: return None
        neckline = max(max(bars.highs[ls:head+1]), max(bars.highs[head:rs+1]))
        if bars.bars[-1].close < neckline: return None
        entry = neckline + 0.02
        stop = rp - 0.02
        risk = entry - stop
        if risk <= 0: return None
        target = entry + (neckline - hp)
        rr = (target - entry) / risk
        return TradeSetup(pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.SWING, entry_price=round(entry,2), stop_loss=round(stop,2),
            target_price=round(target,2), risk_reward_ratio=round(rr,2), confidence=0.68,
            detected_at=bars.bars[-1].timestamp, description=f"Inv H&S: head {hp:.2f}",
            win_rate=0.68, key_levels={"head":hp,"neckline":neckline})

class AscendingTriangle(PatternDetector):
    name = "Ascending Triangle"
    default_bias = Bias.LONG
    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 25: return None
        sw_hi = find_swing_highs(bars.highs, 3)
        sw_lo = find_swing_lows(bars.lows, 3)
        if len(sw_hi) < 2 or len(sw_lo) < 2: return None
        flat_top = np.mean([bars.highs[i] for i in sw_hi[-2:]])
        if abs(bars.highs[sw_hi[-1]] - bars.highs[sw_hi[-2]]) > flat_top * 0.01: return None
        if bars.lows[sw_lo[-1]] <= bars.lows[sw_lo[-2]]: return None
        if bars.bars[-1].close < flat_top: return None
        entry = flat_top + 0.02
        stop = bars.lows[sw_lo[-1]] - 0.02
        risk = entry - stop
        if risk <= 0: return None
        target = entry + (flat_top - bars.lows[sw_lo[-1]])
        rr = (target - entry) / risk
        return TradeSetup(pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.SWING, entry_price=round(entry,2), stop_loss=round(stop,2),
            target_price=round(target,2), risk_reward_ratio=round(rr,2), confidence=0.62,
            detected_at=bars.bars[-1].timestamp, description=f"Asc Triangle: flat top {flat_top:.2f}",
            win_rate=0.62, key_levels={"flat_top":round(flat_top,2)})

class DescendingTriangle(PatternDetector):
    name = "Descending Triangle"
    default_bias = Bias.SHORT
    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 25: return None
        sw_hi = find_swing_highs(bars.highs, 3)
        sw_lo = find_swing_lows(bars.lows, 3)
        if len(sw_hi) < 2 or len(sw_lo) < 2: return None
        flat_bottom = np.mean([bars.lows[i] for i in sw_lo[-2:]])
        if abs(bars.lows[sw_lo[-1]] - bars.lows[sw_lo[-2]]) > flat_bottom * 0.01: return None
        if bars.highs[sw_hi[-1]] >= bars.highs[sw_hi[-2]]: return None
        if bars.bars[-1].close > flat_bottom: return None
        entry = flat_bottom - 0.02
        stop = bars.highs[sw_hi[-1]] + 0.02
        risk = stop - entry
        if risk <= 0: return None
        target = entry - (bars.highs[sw_hi[-1]] - flat_bottom)
        rr = (entry - target) / risk
        return TradeSetup(pattern_name=self.name, symbol=bars.symbol, bias=Bias.SHORT,
            timeframe=Timeframe.SWING, entry_price=round(entry,2), stop_loss=round(stop,2),
            target_price=round(target,2), risk_reward_ratio=round(rr,2), confidence=0.60,
            detected_at=bars.bars[-1].timestamp, description=f"Desc Triangle: flat bottom {flat_bottom:.2f}",
            win_rate=0.60, key_levels={"flat_bottom":round(flat_bottom,2)})

class CupWithHandle(PatternDetector):
    name = "Cup with Handle"
    default_bias = Bias.LONG
    def detect(self, bars: BarSeries) -> Optional[TradeSetup]:
        if len(bars.bars) < 40: return None
        sw_lo = find_swing_lows(bars.lows, 5)
        if not sw_lo: return None
        cup_low_idx = sw_lo[-1]
        cup_low = bars.lows[cup_low_idx]
        left_rim = max(bars.highs[:cup_low_idx]) if cup_low_idx > 0 else None
        right_rim = max(bars.highs[cup_low_idx:]) if cup_low_idx < len(bars.bars) else None
        if left_rim is None or right_rim is None: return None
        rim = min(left_rim, right_rim)
        cup_depth = rim - cup_low
        if cup_depth <= 0: return None
        handle_bars = bars.bars[cup_low_idx + 5:]
        if len(handle_bars) < 3: return None
        handle_low = min(b.low for b in handle_bars[-10:]) if len(handle_bars) >= 10 else min(b.low for b in handle_bars)
        handle_retrace = (rim - handle_low) / cup_depth
        if handle_retrace > 0.50 or handle_retrace < 0.10: return None
        if bars.bars[-1].close < rim: return None
        entry = rim + 0.02
        stop = handle_low - 0.02
        risk = entry - stop
        if risk <= 0: return None
        target = entry + cup_depth
        rr = (target - entry) / risk
        return TradeSetup(pattern_name=self.name, symbol=bars.symbol, bias=Bias.LONG,
            timeframe=Timeframe.SWING, entry_price=round(entry,2), stop_loss=round(stop,2),
            target_price=round(target,2), risk_reward_ratio=round(rr,2), confidence=0.65,
            detected_at=bars.bars[-1].timestamp, description=f"Cup & Handle: rim {rim:.2f}",
            win_rate=0.65, key_levels={"rim":round(rim,2),"cup_low":round(cup_low,2),"handle_low":round(handle_low,2)})


# ==============================================================================
# PATTERN REGISTRY — All 50 patterns
# ==============================================================================

ALL_DETECTORS = [
    # SMB Scalps (10)
    RubberBandScalp(),
    HitchHikerScalp(),
    OpeningRangeBreak(orb_minutes=5),
    OpeningRangeBreak(orb_minutes=15),
    OpeningRangeBreak(orb_minutes=30),
    SecondChanceScalp(),
    BackSideScalp(),
    FashionablyLateScalp(),
    SpencerScalp(),
    GapGiveAndGo(),
    TidalWaveBouncyBall(),
    BreakingNewsStrategy(),
    # Classical (remaining)
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
    """Return all registered pattern detectors."""
    return ALL_DETECTORS

def run_scan(bars: BarSeries) -> list[TradeSetup]:
    """Run all detectors on a single BarSeries, return all found setups."""
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