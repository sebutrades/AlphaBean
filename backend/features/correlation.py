"""
features/correlation.py — SPY Correlation Scorer

Computes rolling correlation between a stock and SPY using 15min candles.
Stocks with LOW correlation are more interesting for independent setups.

Method:
  - Fetch 15min bars for both the stock and SPY
  - Compute candle direction (green=1, red=-1) for each bar
  - Rolling Pearson correlation over last N candles
  - Also compute return-based correlation for magnitude

Output:
  correlation_score: 0-100 where:
    0-30  = Low correlation (independent mover) ★ best for setups
    30-60 = Moderate correlation
    60-100 = High correlation (moves with SPY)

  direction_agreement: % of candles where stock and SPY move same direction
"""
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

CACHE_DIR = Path("cache/correlation")
CACHE_TTL = 900  # 15 minutes


@dataclass
class CorrelationResult:
    symbol: str
    pearson_r: float           # -1 to 1 Pearson correlation of returns
    direction_agreement: float  # 0-1 % of candles moving same direction as SPY
    correlation_score: float   # 0-100 (lower = more independent)
    grade: str                 # A (independent) to F (locked to SPY)
    sample_bars: int           # How many bars used
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "pearson_r": round(self.pearson_r, 3),
            "direction_agreement": round(self.direction_agreement, 3),
            "correlation_score": round(self.correlation_score, 1),
            "grade": self.grade,
            "sample_bars": self.sample_bars,
        }


# Pre-cached SPY bars (fetched once per session)
_spy_cache: dict = {"bars": None, "fetched_at": 0}


def compute_correlation(
    stock_closes: np.ndarray,
    stock_opens: np.ndarray,
    spy_closes: np.ndarray,
    spy_opens: np.ndarray,
    symbol: str = "",
    window: int = 50,
) -> CorrelationResult:
    """
    Compute correlation between a stock and SPY.
    
    Args:
        stock_closes/opens: Stock 15min OHLC arrays
        spy_closes/opens: SPY 15min OHLC arrays (same length)
        window: Rolling window for correlation
    """
    n = min(len(stock_closes), len(spy_closes))
    if n < 20:
        return CorrelationResult(
            symbol=symbol, pearson_r=0, direction_agreement=0.5,
            correlation_score=50, grade="?", sample_bars=n,
        )
    
    # Align to same length (trim from front)
    sc = stock_closes[-n:]
    so = stock_opens[-n:]
    yc = spy_closes[-n:]
    yo = spy_opens[-n:]
    
    # --- Return-based correlation ---
    stock_returns = np.diff(sc) / sc[:-1]
    spy_returns = np.diff(yc) / yc[:-1]
    
    # Use last `window` bars for rolling correlation
    w = min(window, len(stock_returns))
    sr = stock_returns[-w:]
    yr = spy_returns[-w:]
    
    # Handle edge cases
    if np.std(sr) == 0 or np.std(yr) == 0:
        pearson_r = 0.0
    else:
        pearson_r = float(np.corrcoef(sr, yr)[0, 1])
        if np.isnan(pearson_r):
            pearson_r = 0.0
    
    # --- Direction agreement ---
    # Green = close > open, Red = close < open
    stock_dir = np.sign(sc - so)  # 1=green, -1=red, 0=doji
    spy_dir = np.sign(yc - yo)
    
    # Only count bars where both have clear direction
    mask = (stock_dir != 0) & (spy_dir != 0)
    if mask.sum() > 0:
        agreement = float(np.mean(stock_dir[mask] == spy_dir[mask]))
    else:
        agreement = 0.5
    
    # --- Correlation Score (0-100, lower = more independent) ---
    # Combine Pearson R (magnitude) and direction agreement
    abs_r = abs(pearson_r)
    
    # Weight: 60% Pearson, 40% direction agreement
    raw_score = (abs_r * 0.6 + agreement * 0.4) * 100
    correlation_score = max(0, min(100, raw_score))
    
    # Grade
    if correlation_score <= 25:
        grade = "A"  # Very independent
    elif correlation_score <= 40:
        grade = "B"  # Somewhat independent
    elif correlation_score <= 55:
        grade = "C"  # Moderate
    elif correlation_score <= 70:
        grade = "D"  # Correlated
    else:
        grade = "F"  # Locked to SPY
    
    return CorrelationResult(
        symbol=symbol,
        pearson_r=pearson_r,
        direction_agreement=agreement,
        correlation_score=correlation_score,
        grade=grade,
        sample_bars=n,
    )


def fetch_spy_15min_cached() -> tuple[np.ndarray, np.ndarray]:
    """Fetch SPY 15min bars, cached for 15 minutes."""
    from backend.data.massive_client import fetch_bars
    
    if _spy_cache["bars"] is not None and time.time() - _spy_cache["fetched_at"] < CACHE_TTL:
        return _spy_cache["bars"]
    
    spy_data = fetch_bars("SPY", "15min", days_back=10)
    closes = np.array([b.close for b in spy_data.bars], dtype=np.float64)
    opens = np.array([b.open for b in spy_data.bars], dtype=np.float64)
    _spy_cache["bars"] = (closes, opens)
    _spy_cache["fetched_at"] = time.time()
    return closes, opens


def compute_correlation_for_symbol(symbol: str) -> CorrelationResult:
    """
    High-level: fetch both stock and SPY 15min bars, compute correlation.
    Uses caching for SPY bars.
    """
    from backend.data.massive_client import fetch_bars
    
    # Check cache
    cached = _load_cache(symbol)
    if cached is not None:
        return cached
    
    try:
        spy_closes, spy_opens = fetch_spy_15min_cached()
        stock_data = fetch_bars(symbol, "15min", days_back=10)
        stock_closes = np.array([b.close for b in stock_data.bars], dtype=np.float64)
        stock_opens = np.array([b.open for b in stock_data.bars], dtype=np.float64)
        
        result = compute_correlation(stock_closes, stock_opens, spy_closes, spy_opens, symbol)
        _save_cache(symbol, result)
        return result
        
    except Exception as e:
        return CorrelationResult(
            symbol=symbol, pearson_r=0, direction_agreement=0.5,
            correlation_score=50, grade="?", sample_bars=0,
        )


def _load_cache(symbol: str) -> Optional[CorrelationResult]:
    path = CACHE_DIR / f"{symbol}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if time.time() - data.get("cached_at", 0) > CACHE_TTL:
            return None
        return CorrelationResult(**{k: v for k, v in data.items() if k != "cached_at"})
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _save_cache(symbol: str, result: CorrelationResult):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = result.to_dict()
    data["cached_at"] = time.time()
    (CACHE_DIR / f"{symbol}.json").write_text(json.dumps(data))