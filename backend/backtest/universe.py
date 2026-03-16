"""
universe.py — Builds and caches the stock universe for backtesting.

Fetches all stock tickers from Massive.com, sorts by dollar volume,
takes the top 300. Cached per quarter — only re-fetches if the
current quarter's file doesn't exist.

Usage:
    from backend.backtest.universe import get_universe
    symbols = get_universe()  # Returns list of 300 ticker strings
"""
import json
import os
from datetime import datetime
from pathlib import Path

from backend.data.massive_client import get_client

CACHE_DIR = Path("cache")
UNIVERSE_SIZE = 300

# Common ETFs and non-equity tickers to exclude
EXCLUDE_TICKERS = {
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VXX", "UVXY", "SQQQ", "TQQQ",
    "SDS", "SH", "PSQ", "SPXU", "SPXS", "TZA", "SDOW", "SOXS", "LABU", "LABD",
    "ARKK", "ARKG", "XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLP", "XLB",
    "GLD", "SLV", "USO", "UNG", "TLT", "HYG", "LQD", "BND", "AGG", "EMB",
}


def _current_quarter() -> str:
    """Return current quarter string like '2026-Q1'."""
    now = datetime.now()
    q = (now.month - 1) // 3 + 1
    return f"{now.year}-Q{q}"


def _cache_path() -> Path:
    """Return path to the current quarter's universe file."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"universe_{_current_quarter()}.json"


def get_universe(force_refresh: bool = False) -> list[str]:
    """
    Get the top 300 stocks by dollar volume.
    
    Uses cached data if available for the current quarter.
    Set force_refresh=True to re-fetch even if cache exists.
    """
    path = _cache_path()

    if not force_refresh and path.exists():
        data = json.loads(path.read_text())
        print(f"[Universe] Loaded {len(data['symbols'])} symbols from cache ({data['quarter']})")
        return data["symbols"]

    print(f"[Universe] Fetching top {UNIVERSE_SIZE} stocks by dollar volume...")
    symbols = _fetch_top_by_volume()

    # Save to cache
    cache_data = {
        "quarter": _current_quarter(),
        "generated": datetime.now().isoformat(),
        "count": len(symbols),
        "symbols": symbols,
    }
    path.write_text(json.dumps(cache_data, indent=2))
    print(f"[Universe] Saved {len(symbols)} symbols to {path}")

    return symbols


def _fetch_top_by_volume() -> list[str]:
    """
    Fetch all stock snapshots and return top N by dollar volume.
    
    Dollar volume = last_price × day_volume
    This is better than raw volume because a $1 stock trading 100M shares
    is less "in play" than a $200 stock trading 10M shares.
    """
    client = get_client()

    tickers = []
    try:
        # get_snapshot_all returns snapshot data for all tickers
        for snapshot in client.list_snapshot_all("stocks"):
            try:
                ticker = snapshot.ticker
                if not ticker or ticker in EXCLUDE_TICKERS:
                    continue
                # Skip tickers with special characters (warrants, units, etc.)
                if not ticker.isalpha() or len(ticker) > 5:
                    continue

                day = snapshot.day
                prev = snapshot.prev_day
                if day is None or prev is None:
                    continue

                price = day.close if day.close and day.close > 0 else (prev.close if prev.close else 0)
                volume = day.volume if day.volume else 0

                if price < 5.0:  # Skip penny stocks
                    continue
                if volume < 100000:  # Skip very low volume
                    continue

                dollar_vol = price * volume
                tickers.append((ticker, dollar_vol, price, volume))

            except (AttributeError, TypeError):
                continue

    except Exception as e:
        print(f"[Universe] Error fetching snapshots: {e}")
        print("[Universe] Falling back to hardcoded top 100...")
        return _fallback_universe()

    # Sort by dollar volume descending
    tickers.sort(key=lambda x: x[1], reverse=True)

    # Take top N
    top = tickers[:UNIVERSE_SIZE]
    symbols = [t[0] for t in top]

    print(f"[Universe] Top 5 by dollar volume:")
    for sym, dvol, price, vol in top[:5]:
        print(f"  {sym}: ${dvol / 1e9:.1f}B (${price:.2f} × {vol:,.0f})")

    return symbols


def _fallback_universe() -> list[str]:
    """
    Hardcoded fallback if the API call fails.
    These are consistently the most liquid US stocks.
    """
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK.B",
        "AVGO", "JPM", "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "PG",
        "JNJ", "ABBV", "WMT", "NFLX", "CRM", "BAC", "AMD", "CVX", "ORCL",
        "KO", "MRK", "PEP", "ACN", "TMO", "MCD", "CSCO", "LIN", "ADBE",
        "ABT", "WFC", "DHR", "TXN", "QCOM", "PM", "NEE", "INTU", "DIS",
        "AMGN", "RTX", "UNP", "HON", "LOW", "AMAT", "CAT", "GS", "ISRG",
        "BLK", "DE", "PFE", "NOW", "MS", "AXP", "BKNG", "IBM", "GILD",
        "VRTX", "MDLZ", "SYK", "LRCX", "ADI", "MMC", "SCHW", "C", "REGN",
        "CB", "T", "MU", "KLAC", "ZTS", "SO", "PANW", "SNPS", "DUK",
        "CME", "EQIX", "PGR", "BSX", "AON", "CDNS", "FI", "CL", "ITW",
        "ICE", "MCO", "SHW", "NOC", "CMG", "MO", "BMY", "PLD", "USB",
    ]