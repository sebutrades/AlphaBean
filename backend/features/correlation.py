"""
features/correlation.py — Live SPY Correlation Scorer

Computes LIVE correlation (no caching) between a stock and SPY
using 15min candles over last 3 days.

Grades:
  A = Very independent / Relative strength or weakness to market
  B = Somewhat independent
  C = Moderate correlation
  D = Correlated
  F = Locked to SPY

Also detects relative strength/weakness:
  If stock is UP while SPY is DOWN → "Relative Strength"
  If stock is DOWN while SPY is UP → "Relative Weakness"
  If both same direction but stock outperforms → "Outperforming"
  If both same direction but stock underperforms → "Underperforming"
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class CorrelationResult:
    symbol: str
    pearson_r: float            # -1 to 1
    direction_agreement: float  # 0-1
    correlation_score: float    # 0-100 (lower = more independent)
    grade: str                  # A-F
    label: str                  # "Relative Strength", "Relative Weakness", etc.
    color: str                  # hex color for UI
    sample_bars: int

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "pearson_r": round(self.pearson_r, 3),
            "direction_agreement": round(self.direction_agreement * 100, 1),
            "correlation_score": round(self.correlation_score, 1),
            "grade": self.grade,
            "label": self.label,
            "color": self.color,
            "sample_bars": self.sample_bars,
        }


# SPY bars cached in memory (refreshed per request cycle, not across)
_spy_cache: dict = {"closes": None, "opens": None, "returns": None}


def compute_correlation_live(symbol: str) -> CorrelationResult:
    """
    Fetch LIVE 15min bars for both stock and SPY (last 3 days),
    compute correlation. No disk caching — always fresh.
    """
    from backend.data.massive_client import fetch_bars

    try:
        # Fetch SPY (reuse in-memory if recent enough)
        spy_data = fetch_bars("SPY", "15min", days_back=3)
        stock_data = fetch_bars(symbol.upper(), "15min", days_back=3)

        spy_closes = np.array([b.close for b in spy_data.bars], dtype=np.float64)
        spy_opens = np.array([b.open for b in spy_data.bars], dtype=np.float64)
        stock_closes = np.array([b.close for b in stock_data.bars], dtype=np.float64)
        stock_opens = np.array([b.open for b in stock_data.bars], dtype=np.float64)

        return _compute(symbol.upper(), stock_closes, stock_opens, spy_closes, spy_opens)

    except Exception as e:
        return CorrelationResult(
            symbol=symbol.upper(), pearson_r=0, direction_agreement=0.5,
            correlation_score=50, grade="?", label="Unknown", color="#64748b",
            sample_bars=0,
        )


def _compute(
    symbol: str,
    stock_closes: np.ndarray, stock_opens: np.ndarray,
    spy_closes: np.ndarray, spy_opens: np.ndarray,
) -> CorrelationResult:
    """Core correlation computation."""
    n = min(len(stock_closes), len(spy_closes))
    if n < 10:
        return CorrelationResult(symbol=symbol, pearson_r=0, direction_agreement=0.5,
                                 correlation_score=50, grade="?", label="Insufficient data",
                                 color="#64748b", sample_bars=n)

    # Align to same length
    sc = stock_closes[-n:]
    so = stock_opens[-n:]
    yc = spy_closes[-n:]
    yo = spy_opens[-n:]

    # --- Pearson correlation of bar-over-bar returns ---
    stock_ret = np.diff(sc) / sc[:-1]
    spy_ret = np.diff(yc) / yc[:-1]

    # Clean NaN/Inf
    mask = np.isfinite(stock_ret) & np.isfinite(spy_ret)
    sr = stock_ret[mask]
    yr = spy_ret[mask]

    if len(sr) < 10 or np.std(sr) == 0 or np.std(yr) == 0:
        pearson_r = 0.0
    else:
        pearson_r = float(np.corrcoef(sr, yr)[0, 1])
        if np.isnan(pearson_r):
            pearson_r = 0.0

    # --- Direction agreement ---
    stock_dir = np.sign(sc - so)  # 1=green, -1=red
    spy_dir = np.sign(yc - yo)
    clear = (stock_dir != 0) & (spy_dir != 0)
    agreement = float(np.mean(stock_dir[clear] == spy_dir[clear])) if clear.sum() > 0 else 0.5

    # --- Correlation score (0-100, lower = more independent) ---
    abs_r = abs(pearson_r)
    raw = (abs_r * 0.6 + agreement * 0.4) * 100
    score = max(0, min(100, raw))

    # --- Grade ---
    if score <= 25: grade = "A"
    elif score <= 40: grade = "B"
    elif score <= 55: grade = "C"
    elif score <= 70: grade = "D"
    else: grade = "F"

    # --- Relative Strength / Weakness label ---
    # Compare recent performance (last 20 bars)
    recent = min(20, len(sr))
    stock_perf = float(np.sum(sr[-recent:]))  # Cumulative return
    spy_perf = float(np.sum(yr[-recent:]))

    stock_up = stock_perf > 0.001
    stock_dn = stock_perf < -0.001
    spy_up = spy_perf > 0.001
    spy_dn = spy_perf < -0.001

    if stock_up and spy_dn:
        label = "Relative Strength"
        color = "#10b981"  # green
    elif stock_dn and spy_up:
        label = "Relative Weakness"
        color = "#ef4444"  # red
    elif stock_up and spy_up:
        if stock_perf > spy_perf * 1.5:
            label = "Outperforming"
            color = "#10b981"
        elif stock_perf < spy_perf * 0.5:
            label = "Underperforming"
            color = "#f59e0b"
        else:
            label = "In Line"
            color = "#64748b"
    elif stock_dn and spy_dn:
        if abs(stock_perf) < abs(spy_perf) * 0.5:
            label = "Holding Up"
            color = "#10b981"
        elif abs(stock_perf) > abs(spy_perf) * 1.5:
            label = "Underperforming"
            color = "#ef4444"
        else:
            label = "In Line"
            color = "#64748b"
    else:
        label = "Neutral"
        color = "#64748b"

    # Override color for highly independent (grade A/B)
    if grade in ("A", "B"):
        if label in ("Relative Strength", "Outperforming", "Holding Up"):
            color = "#10b981"
        elif label in ("Relative Weakness", "Underperforming"):
            color = "#ef4444"
        else:
            color = "#3b82f6"  # blue for independent

    return CorrelationResult(
        symbol=symbol, pearson_r=pearson_r, direction_agreement=agreement,
        correlation_score=score, grade=grade, label=label, color=color,
        sample_bars=n,
    )