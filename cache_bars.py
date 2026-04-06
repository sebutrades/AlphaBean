"""
cache_bars.py — Pre-fetch and cache all bar data for optimization.

Fetches bars for N symbols across all timeframes, saves to disk.
The optimizer reads from cache instead of hitting Polygon every trial.

Usage:
  python cache_bars.py --symbols 250 --days 180              # Cache 250 symbols, 2 quarters
  python cache_bars.py --symbols 250 --days 365 --daily-only # Just daily bars (fastest)

Output: cache/bar_data/{symbol}_{timeframe}.json

After caching, the optimizer runs ~10x faster because it reads
from local files instead of making API calls.
"""
import json
import time
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

from backend.data.massive_client import fetch_bars
from backend.data.schemas import BarSeries

CACHE_DIR = Path("cache/bar_data")
SYMBOLS_CACHE = Path("cache/top_symbols.json")
TIMEFRAMES = ["5min", "15min", "1h", "1d"]


def cache_symbol(symbol: str, timeframe: str, days_back: int) -> dict:
    """Fetch and cache bars for one symbol/timeframe combo.
    
    Returns: {"bars": int, "cached": bool, "error": str|None}
    """
    cache_file = CACHE_DIR / f"{symbol}_{timeframe}.json"
    
    # Skip if already cached (and recent enough)
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            cached_days = data.get("days_back", 0)
            cached_at = data.get("cached_at", "")
            # Re-cache if requesting more days than what's cached
            if cached_days >= days_back:
                return {"bars": data.get("n_bars", 0), "cached": True, "error": None, "skipped": True}
        except Exception:
            pass

    # Fetch from Polygon
    fetch_days = max(days_back, 365) if timeframe == "1d" else days_back
    
    try:
        bars_data = fetch_bars(symbol, timeframe, fetch_days)
        bars = bars_data.bars
        
        if not bars:
            return {"bars": 0, "cached": False, "error": "No bars returned"}
        
        # Serialize bars to JSON
        bar_list = []
        for b in bars:
            bar_list.append({
                "t": b.timestamp.isoformat(),
                "o": round(b.open, 4),
                "h": round(b.high, 4),
                "l": round(b.low, 4),
                "c": round(b.close, 4),
                "v": int(b.volume),
            })
        
        cache_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "days_back": days_back,
            "n_bars": len(bar_list),
            "cached_at": datetime.now().isoformat(),
            "bars": bar_list,
        }
        
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file + rename prevents corrupt cache on crash
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache_data))
        tmp.replace(cache_file)
        
        return {"bars": len(bar_list), "cached": True, "error": None, "skipped": False}
    
    except Exception as e:
        return {"bars": 0, "cached": False, "error": str(e)}


def load_cached_bars(symbol: str, timeframe: str) -> BarSeries | None:
    """Load cached bars from disk. Returns BarSeries or None."""
    from backend.data.schemas import Bar
    
    cache_file = CACHE_DIR / f"{symbol}_{timeframe}.json"
    if not cache_file.exists():
        return None
    
    try:
        data = json.loads(cache_file.read_text())
        bars = []
        for b in data["bars"]:
            bars.append(Bar(
                symbol=symbol,
                timestamp=datetime.fromisoformat(b["t"]),
                open=b["o"], high=b["h"], low=b["l"], close=b["c"],
                volume=b["v"],
            ))
        return BarSeries(symbol=symbol, timeframe=timeframe, bars=bars)
    except Exception:
        return None


def is_cached(symbol: str, timeframe: str) -> bool:
    """Check if bars are cached for a symbol/timeframe."""
    return (CACHE_DIR / f"{symbol}_{timeframe}.json").exists()


def cache_all(
    n_symbols: int = 250,
    days_back: int = 180,
    timeframes: list[str] = None,
):
    """Cache bars for top N symbols across all timeframes."""
    if timeframes is None:
        timeframes = TIMEFRAMES
    
    # Load symbols
    if not SYMBOLS_CACHE.exists():
        print("  ✗ No cached symbols. Run: python fetch_symbols.py")
        sys.exit(1)
    
    symbols = json.loads(SYMBOLS_CACHE.read_text()).get("symbols", [])[:n_symbols]
    
    total_jobs = len(symbols) * len(timeframes)
    
    print(f"\n{'═' * 60}")
    print(f"  Bar Data Cache Builder")
    print(f"{'═' * 60}")
    print(f"  Symbols:    {len(symbols)}")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Days:       {days_back}")
    print(f"  Total jobs: {total_jobs}")
    print(f"  Cache dir:  {CACHE_DIR}")
    print(f"{'═' * 60}")
    
    t_start = time.time()
    done = 0; errors = 0; skipped = 0; total_bars = 0
    
    for si, symbol in enumerate(symbols):
        sym_bars = 0; sym_errors = []
        
        for tf in timeframes:
            result = cache_symbol(symbol, tf, days_back)
            done += 1
            
            if result.get("skipped"):
                skipped += 1
                sym_bars += result["bars"]
            elif result["error"]:
                errors += 1
                sym_errors.append(f"{tf}: {result['error']}")
            else:
                sym_bars += result["bars"]
                total_bars += result["bars"]
            
            # Rate limit: small delay between API calls
            if not result.get("skipped"):
                time.sleep(0.15)
        
        # Progress
        elapsed = time.time() - t_start
        rate = (si + 1) / elapsed if elapsed > 0 else 1
        eta = (len(symbols) - si - 1) / rate
        
        err_str = f" ✗ {', '.join(sym_errors)}" if sym_errors else ""
        print(f"  [{si+1:>4}/{len(symbols)}] {symbol:<6} "
              f"{sym_bars:>6} bars  "
              f"eta:{eta/60:.0f}m{err_str}")
    
    elapsed = time.time() - t_start
    
    # Summary
    cache_size = sum(f.stat().st_size for f in CACHE_DIR.glob("*.json")) / (1024 * 1024)
    
    print(f"\n{'═' * 60}")
    print(f"  CACHE COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Cached:   {done - errors - skipped} new files")
    print(f"  Skipped:  {skipped} (already cached)")
    print(f"  Errors:   {errors}")
    print(f"  Bars:     {total_bars:,}")
    print(f"  Size:     {cache_size:.0f} MB")
    print(f"  Time:     {elapsed/60:.1f} min")
    print(f"{'═' * 60}")
    print(f"\n  Now run optimization (reads from cache):")
    print(f"    python run_weekend_optimization.py --symbols {n_symbols} --cores 6")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache bar data from Polygon")
    parser.add_argument("--symbols", type=int, default=250)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--daily-only", action="store_true", help="Only cache 1d bars")
    parser.add_argument("--timeframes", type=str, default=None, help="Comma-separated: 5min,15min,1h,1d")
    args = parser.parse_args()
    
    if args.timeframes:
        tfs = [t.strip() for t in args.timeframes.split(",")]
    elif args.daily_only:
        tfs = ["1d"]
    else:
        tfs = TIMEFRAMES
    
    cache_all(args.symbols, args.days, tfs)