"""
features/correlation.py — Live SPY Relative Strength/Weakness

Compares a stock's INTRADAY performance vs SPY to determine:
  - Relative Strength (stock up, market down — or stock outperforming)
  - Relative Weakness (stock down, market up — or stock underperforming)
  - In Line (moving similarly)

Method:
  1. Fetch today's 15min bars for both stock and SPY
  2. Compare session returns: (current price - open) / open
  3. Compare bar-by-bar direction agreement
  4. Grade the correlation

This is SIMPLE and CORRECT. No fancy windowing that can misalign.
"""
from dataclasses import dataclass
from datetime import datetime, time as dt_time
import numpy as np


@dataclass
class CorrelationResult:
    symbol: str
    stock_return_pct: float     # Today's return %
    spy_return_pct: float       # SPY's return %
    spread_pct: float           # Stock return - SPY return
    direction_agreement: float  # 0-100% of bars same direction
    correlation_score: float    # 0-100 (lower = more independent)
    grade: str                  # A-F
    label: str                  # "Relative Strength", etc.
    color: str                  # hex for UI
    sample_bars: int

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "stock_return_pct": round(self.stock_return_pct, 2),
            "spy_return_pct": round(self.spy_return_pct, 2),
            "spread_pct": round(self.spread_pct, 2),
            "direction_agreement": round(self.direction_agreement, 1),
            "correlation_score": round(self.correlation_score, 1),
            "grade": self.grade,
            "label": self.label,
            "color": self.color,
            "sample_bars": self.sample_bars,
        }


def compute_correlation_live(symbol: str) -> CorrelationResult:
    """Compute live relative strength/weakness vs SPY."""
    from backend.data.massive_client import fetch_bars

    try:
        stock_data = fetch_bars(symbol.upper(), "15min", days_back=3)
        spy_data = fetch_bars("SPY", "15min", days_back=3)

        stock_bars = stock_data.bars
        spy_bars = spy_data.bars

        if len(stock_bars) < 5 or len(spy_bars) < 5:
            return _empty(symbol)

        # --- Get TODAY's bars only ---
        today = stock_bars[-1].timestamp.date()
        stock_today = [b for b in stock_bars if b.timestamp.date() == today]
        spy_today = [b for b in spy_bars if b.timestamp.date() == today]

        # If today has too few bars (pre-market), use yesterday
        if len(stock_today) < 3 or len(spy_today) < 3:
            yesterday = stock_bars[-1].timestamp.date()
            for b in reversed(stock_bars):
                if b.timestamp.date() != today:
                    yesterday = b.timestamp.date()
                    break
            stock_today = [b for b in stock_bars if b.timestamp.date() == yesterday]
            spy_today = [b for b in spy_bars if b.timestamp.date() == yesterday]

        if len(stock_today) < 3 or len(spy_today) < 3:
            return _empty(symbol)

        # --- Session returns ---
        stock_open = stock_today[0].open
        stock_current = stock_today[-1].close
        spy_open = spy_today[0].open
        spy_current = spy_today[-1].close

        stock_ret = ((stock_current - stock_open) / stock_open) * 100  # percentage
        spy_ret = ((spy_current - spy_open) / spy_open) * 100
        spread = stock_ret - spy_ret

        # --- Direction agreement (bar-by-bar) ---
        # Use min length and align from the end (most recent bars)
        n = min(len(stock_today), len(spy_today))
        st = stock_today[-n:]
        sp = spy_today[-n:]

        same_dir = 0
        total = 0
        for i in range(n):
            s_green = st[i].close > st[i].open
            p_green = sp[i].close > sp[i].open
            s_red = st[i].close < st[i].open
            p_red = sp[i].close < sp[i].open
            if (s_green or s_red) and (p_green or p_red):
                total += 1
                if s_green == p_green:
                    same_dir += 1

        agreement = (same_dir / total * 100) if total > 0 else 0

        # --- Correlation score (0-100, lower = more independent) ---
        score = agreement  # Simple: just use direction agreement

        # --- Grade ---
        if score <= 30: grade = "A"
        elif score <= 45: grade = "B"
        elif score <= 60: grade = "C"
        elif score <= 75: grade = "D"
        else: grade = "F"

        # --- Label based on ACTUAL returns ---
        stock_up = stock_ret > 0.05   # Up more than 0.05%
        stock_dn = stock_ret < -0.05
        spy_up = spy_ret > 0.05
        spy_dn = spy_ret < -0.05

        if stock_up and spy_dn:
            label = "Relative Strength"
            color = "#22c55e"
        elif stock_dn and spy_up:
            label = "Relative Weakness"
            color = "#ef4444"
        elif stock_up and spy_up:
            if stock_ret > spy_ret + 0.3:
                label = "Outperforming"
                color = "#22c55e"
            elif stock_ret < spy_ret - 0.3:
                label = "Lagging"
                color = "#f59e0b"
            else:
                label = "In Line"
                color = "#64748b"
        elif stock_dn and spy_dn:
            if abs(stock_ret) < abs(spy_ret) * 0.6:
                label = "Holding Up"
                color = "#22c55e"
            elif abs(stock_ret) > abs(spy_ret) * 1.5:
                label = "Underperforming"
                color = "#ef4444"
            else:
                label = "In Line"
                color = "#64748b"
        else:
            label = "Neutral"
            color = "#64748b"

        return CorrelationResult(
            symbol=symbol.upper(),
            stock_return_pct=stock_ret,
            spy_return_pct=spy_ret,
            spread_pct=spread,
            direction_agreement=agreement,
            correlation_score=score,
            grade=grade,
            label=label,
            color=color,
            sample_bars=n,
        )

    except Exception as e:
        return _empty(symbol, str(e))


def _empty(symbol: str, reason: str = "") -> CorrelationResult:
    return CorrelationResult(
        symbol=symbol.upper(), stock_return_pct=0, spy_return_pct=0,
        spread_pct=0, direction_agreement=50, correlation_score=50,
        grade="?", label="No data", color="#64748b", sample_bars=0,
    )