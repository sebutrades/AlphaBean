"""
data_fetcher.py — Downloads and caches bar data for backtesting.

For each symbol in the universe, fetches 1 year of bars on each
timeframe (15min, 1h, 1d) and saves to cache/bars/.

Rate limited to stay within Massive.com Starter plan limits.
Skips symbols where cache already exists and is fresh.

Usage:
    from backend.backtest.data_fetcher import fetch_all_data
    stats = fetch_all_data(symbols=["AAPL", "NVDA"], timeframes=["1d"])
"""
import json
import time as time_module
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from backend.data.massive_client import fetch_bars

CACHE_DIR = Path("cache/bars")
RATE_LIMIT_PER_MIN = 75  # Conservative for Starter plan (limit ~100)
MAX_CACHE_AGE_DAYS = 7   # Re-fetch if cache is older than this

TIMEFRAMES = ["15min", "1h", "1d"]
DAYS_BACK = 365  # 1 year of history for backtesting


def fetch_all_data(
    symbols: list[str],
    timeframes: list[str] = None,
    days_back: int = DAYS_BACK,
    force_refresh: bool = False,
) -> dict:
    """
    Fetch bar data for all symbols × timeframes. Cache results locally.
    
    Returns stats dict with counts of fetched, cached, failed.
    """
    if timeframes is None:
        timeframes = TIMEFRAMES

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    total = len(symbols) * len(timeframes)
    fetched = 0
    cached = 0
    failed = 0
    call_count = 0
    start_time = time_module.time()

    print(f"\n{'=' * 60}")
    print(f"Data Fetch: {len(symbols)} symbols × {len(timeframes)} timeframes = {total} jobs")
    print(f"{'=' * 60}\n")

    for si, symbol in enumerate(symbols):
        for tf in timeframes:
            cache_path = _cache_path(symbol, tf)

            # Check cache
            if not force_refresh and _is_cache_fresh(cache_path):
                cached += 1
                continue

            # Rate limiting
            call_count += 1
            if call_count > 1 and call_count % RATE_LIMIT_PER_MIN == 0:
                elapsed = time_module.time() - start_time
                if elapsed < 60:
                    sleep_time = 60 - elapsed + 1
                    print(f"  [RATE LIMIT] Sleeping {sleep_time:.0f}s...")
                    time_module.sleep(sleep_time)
                start_time = time_module.time()

            # Fetch
            try:
                progress = fetched + cached + failed + 1
                print(f"  [{progress}/{total}] Fetching {symbol} {tf}...", end=" ")

                bars = fetch_bars(symbol, timeframe=tf, days_back=days_back)

                if len(bars.bars) < 5:
                    print(f"SKIP (only {len(bars.bars)} bars)")
                    failed += 1
                    continue

                # Save to cache
                bar_dicts = [
                    {
                        "t": b.timestamp.isoformat(),
                        "o": b.open, "h": b.high, "l": b.low, "c": b.close,
                        "v": b.volume,
                        "vw": b.vwap,
                        "tc": b.trade_count,
                    }
                    for b in bars.bars
                ]

                cache_data = {
                    "symbol": symbol,
                    "timeframe": tf,
                    "fetched_at": datetime.now().isoformat(),
                    "bar_count": len(bar_dicts),
                    "bars": bar_dicts,
                }

                cache_path.write_text(json.dumps(cache_data))
                print(f"OK ({len(bar_dicts)} bars)")
                fetched += 1

            except Exception as e:
                print(f"FAIL ({e})")
                failed += 1

        # Progress summary every 10 symbols
        if (si + 1) % 10 == 0:
            done = fetched + cached + failed
            print(f"\n  --- Progress: {done}/{total} ({fetched} fetched, {cached} cached, {failed} failed) ---\n")

    print(f"\n{'=' * 60}")
    print(f"Data Fetch Complete")
    print(f"  Fetched: {fetched} | Cached: {cached} | Failed: {failed}")
    print(f"{'=' * 60}\n")

    return {"fetched": fetched, "cached": cached, "failed": failed, "total": total}


def load_cached_bars(symbol: str, timeframe: str) -> Optional[list[dict]]:
    """Load bars from cache file. Returns list of bar dicts or None."""
    path = _cache_path(symbol, timeframe)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data["bars"]
    except (json.JSONDecodeError, KeyError):
        return None


def bars_from_cache(symbol: str, timeframe: str):
    """
    Load cached bars and convert to BarSeries for pattern detection.
    Returns BarSeries (backend.data.schemas) or None.
    """
    from backend.data.schemas import Bar, BarSeries

    raw = load_cached_bars(symbol, timeframe)
    if raw is None:
        return None

    bars = []
    for b in raw:
        bars.append(Bar(
            symbol=symbol,
            timestamp=datetime.fromisoformat(b["t"]),
            open=b["o"], high=b["h"], low=b["l"], close=b["c"],
            volume=b["v"],
            vwap=b.get("vw"),
            trade_count=b.get("tc"),
        ))

    return BarSeries(symbol=symbol, timeframe=timeframe, bars=bars)


def get_cache_stats() -> dict:
    """Return stats about what's in the cache."""
    if not CACHE_DIR.exists():
        return {"total_files": 0, "symbols": 0, "timeframes": {}}

    files = list(CACHE_DIR.glob("*.json"))
    symbols = set()
    tf_counts = {}

    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            symbols.add(parts[0])
            tf = "_".join(parts[1:])
            tf_counts[tf] = tf_counts.get(tf, 0) + 1

    return {
        "total_files": len(files),
        "symbols": len(symbols),
        "timeframes": tf_counts,
    }


def _cache_path(symbol: str, timeframe: str) -> Path:
    return CACHE_DIR / f"{symbol}_{timeframe}.json"


def _is_cache_fresh(path: Path) -> bool:
    """Check if cache file exists and is less than MAX_CACHE_AGE_DAYS old."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        fetched_at = datetime.fromisoformat(data["fetched_at"])
        age = datetime.now() - fetched_at
        return age.days < MAX_CACHE_AGE_DAYS
    except (json.JSONDecodeError, KeyError, ValueError):
        return False