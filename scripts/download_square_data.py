"""
scripts/download_square_data.py — Download perfectly square bar data.

Downloads 1000 top-liquid US stocks across 4 timeframes (5min, 15min, 1h, 1d)
from June 1, 2025 to April 1, 2026. Deletes previous cache/bar_data entirely
and replaces with clean, uniform data.

Usage:
    python scripts/download_square_data.py
    python scripts/download_square_data.py --symbols 500 --workers 10
    python scripts/download_square_data.py --verify-only   # just verify existing data

Estimated: ~4,000 API calls, ~15-25 min, ~3-4 GB disk.
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

from massive import RESTClient

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    ET = pytz.timezone("America/New_York")


# ── Config ──────────────────────────────────────────────────────────────────

BAR_DATA_DIR = Path("cache/bar_data")

DATE_FROM = "2025-06-01"
DATE_TO   = "2026-04-01"

TIMEFRAMES = {
    "5min":  (5, "minute"),
    "15min": (15, "minute"),
    "1h":    (1, "hour"),
    "1d":    (1, "day"),
}

# Market hours for filtering intraday bars (ET)
MARKET_OPEN_H, MARKET_OPEN_M = 9, 30
MARKET_CLOSE_H, MARKET_CLOSE_M = 16, 0

# Rate limiting
CALLS_PER_SECOND = 4  # conservative — most plans allow 5/s
_rate_lock = Lock()
_last_call_time = 0.0

# Progress
_progress_lock = Lock()
_completed = 0
_failed = 0
_total = 0


def _rate_limit():
    """Enforce rate limiting across threads."""
    global _last_call_time
    with _rate_lock:
        now = time.time()
        min_interval = 1.0 / CALLS_PER_SECOND
        wait = min_interval - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.time()


def _progress(symbol: str, tf: str, bars: int, success: bool):
    """Thread-safe progress update."""
    global _completed, _failed
    with _progress_lock:
        if success:
            _completed += 1
        else:
            _failed += 1
        done = _completed + _failed
        pct = (done / _total * 100) if _total > 0 else 0
        status = f"{bars:>6} bars" if success else "FAILED"
        print(f"  [{done:>4}/{_total}] {pct:5.1f}%  {symbol:<6} {tf:<5}  {status}", flush=True)


# ── Symbol Selection ────────────────────────────────────────────────────────

def get_top_symbols(n: int = 1000) -> list[str]:
    """Get top N liquid US stocks/ETFs by volume (price >= $5, vol >= 500K)."""
    print(f"Fetching market snapshot to find top {n} liquid symbols...")
    client = RESTClient(api_key=os.getenv("MASSIVE_API_KEY"))

    tickers = []
    for snap in client.get_snapshot_all("stocks"):
        t = snap.ticker
        # Skip warrants, units, rights, and tickers with weird chars
        if any(c in t for c in [".", "/", "-"]):
            continue
        if len(t) > 5:
            continue
        if snap.day and snap.day.volume and snap.day.close:
            if snap.day.close >= 5 and snap.day.volume >= 500_000:
                tickers.append((t, snap.day.volume))

    tickers.sort(key=lambda x: x[1], reverse=True)
    symbols = [t[0] for t in tickers[:n]]

    # Ensure SPY is always included (needed as benchmark)
    if "SPY" not in symbols:
        symbols[-1] = "SPY"

    print(f"  Selected {len(symbols)} symbols (min volume: {tickers[min(n-1, len(tickers)-1)][1]:,.0f})")
    return sorted(symbols)


# ── Download ────────────────────────────────────────────────────────────────

def download_symbol_tf(symbol: str, tf: str, multiplier: int, timespan: str,
                       retries: int = 3) -> dict | None:
    """Download bars for one symbol+timeframe. Returns cache dict or None."""
    for attempt in range(retries):
        try:
            _rate_limit()
            client = RESTClient(api_key=os.getenv("MASSIVE_API_KEY"))

            raw_bars = []
            for agg in client.list_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=DATE_FROM,
                to=DATE_TO,
                limit=50000,
                sort="asc",
            ):
                raw_bars.append(agg)

            # Convert to ET and format
            bars = []
            for agg in raw_bars:
                utc_dt = datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc)
                et_dt = utc_dt.astimezone(ET)
                et_naive = et_dt.replace(tzinfo=None)

                # Filter intraday bars to market hours only
                if tf != "1d":
                    h, m = et_naive.hour, et_naive.minute
                    bar_minutes = h * 60 + m
                    open_minutes = MARKET_OPEN_H * 60 + MARKET_OPEN_M
                    close_minutes = MARKET_CLOSE_H * 60 + MARKET_CLOSE_M
                    if bar_minutes < open_minutes or bar_minutes >= close_minutes:
                        continue

                bars.append({
                    "t": et_naive.strftime("%Y-%m-%dT%H:%M:%S"),
                    "o": round(agg.open, 4),
                    "h": round(agg.high, 4),
                    "l": round(agg.low, 4),
                    "c": round(agg.close, 4),
                    "v": int(agg.volume),
                })

            if not bars:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                return None

            return {
                "symbol": symbol,
                "timeframe": tf,
                "date_from": DATE_FROM,
                "date_to": DATE_TO,
                "n_bars": len(bars),
                "cached_at": datetime.now().isoformat(),
                "bars": bars,
            }

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"    ERROR {symbol} {tf}: {e}")
            return None

    return None


def download_one(symbol: str, tf: str):
    """Download and save one symbol+timeframe."""
    multiplier, timespan = TIMEFRAMES[tf]
    result = download_symbol_tf(symbol, tf, multiplier, timespan)

    if result and result["bars"]:
        out_path = BAR_DATA_DIR / f"{symbol}_{tf}.json"
        out_path.write_text(json.dumps(result, separators=(",", ":")))
        _progress(symbol, tf, len(result["bars"]), True)
        return (symbol, tf, len(result["bars"]))
    else:
        _progress(symbol, tf, 0, False)
        return (symbol, tf, 0)


# ── Verification ────────────────────────────────────────────────────────────

def verify_data(symbols: list[str] | None = None) -> bool:
    """Verify data integrity and squareness."""
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    if not BAR_DATA_DIR.exists():
        print("ERROR: cache/bar_data does not exist!")
        return False

    # Discover what we have
    all_files = list(BAR_DATA_DIR.glob("*.json"))
    print(f"Total files: {len(all_files)}")

    # Group by timeframe
    tf_data: dict[str, dict[str, dict]] = {tf: {} for tf in TIMEFRAMES}

    for f in all_files:
        parts = f.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        sym, tf = parts
        if tf not in TIMEFRAMES:
            continue

        try:
            data = json.loads(f.read_text())
            bars = data.get("bars", [])
            dates = sorted(set(b["t"][:10] for b in bars))
            tf_data[tf][sym] = {
                "bars": len(bars),
                "dates": len(dates),
                "first": dates[0] if dates else "",
                "last": dates[-1] if dates else "",
            }
        except Exception as e:
            print(f"  CORRUPT: {f.name}: {e}")

    # Report per timeframe
    all_ok = True
    reference_symbols = None
    reference_dates_1d = None

    for tf in ["1d", "5min", "15min", "1h"]:
        syms = tf_data[tf]
        if not syms:
            print(f"\n{tf}: NO DATA")
            all_ok = False
            continue

        bar_counts = [v["bars"] for v in syms.values()]
        date_counts = [v["dates"] for v in syms.values()]
        firsts = set(v["first"] for v in syms.values())
        lasts = set(v["last"] for v in syms.values())

        print(f"\n{tf}:")
        print(f"  Symbols:     {len(syms)}")
        print(f"  Bars/symbol: min={min(bar_counts)}, max={max(bar_counts)}, median={sorted(bar_counts)[len(bar_counts)//2]}")
        print(f"  Days/symbol: min={min(date_counts)}, max={max(date_counts)}, median={sorted(date_counts)[len(date_counts)//2]}")
        print(f"  Start dates: {firsts}")
        print(f"  End dates:   {lasts}")

        # Check squareness: all symbols should have same date range
        if len(firsts) > 1 or len(lasts) > 1:
            print(f"  WARNING: Not perfectly square — multiple start/end dates")
            # Show distribution
            from collections import Counter
            start_dist = Counter(v["first"] for v in syms.values())
            end_dist = Counter(v["last"] for v in syms.values())
            for d, c in sorted(start_dist.items()):
                if c < len(syms):
                    print(f"    Start {d}: {c} symbols ({c/len(syms)*100:.1f}%)")
            for d, c in sorted(end_dist.items()):
                if c < len(syms):
                    print(f"    End {d}: {c} symbols ({c/len(syms)*100:.1f}%)")

        # Check consistency across timeframes
        if reference_symbols is None:
            reference_symbols = set(syms.keys())
        else:
            missing = reference_symbols - set(syms.keys())
            extra = set(syms.keys()) - reference_symbols
            if missing:
                print(f"  MISSING vs reference: {len(missing)} symbols ({list(missing)[:5]}...)")
                all_ok = False
            if extra:
                print(f"  EXTRA vs reference: {len(extra)} symbols")

        # Tolerance check: allow symbols that simply didn't exist yet (IPO)
        # but flag symbols with suspiciously few bars
        if tf == "5min":
            expected_min = 100 * 78  # at least 100 days × 78 bars
            thin = [(s, v["bars"]) for s, v in syms.items() if v["bars"] < expected_min]
            if thin:
                print(f"  THIN (<{expected_min} bars): {len(thin)} symbols")
                for s, b in sorted(thin, key=lambda x: x[1])[:10]:
                    print(f"    {s}: {b} bars ({syms[s]['dates']} days, {syms[s]['first']} to {syms[s]['last']})")

        if tf == "1d":
            reference_dates_1d = date_counts

    # Overall verdict
    print(f"\n{'='*70}")
    if symbols:
        expected = len(symbols)
        for tf in TIMEFRAMES:
            actual = len(tf_data[tf])
            if actual < expected:
                print(f"INCOMPLETE: {tf} has {actual}/{expected} symbols")
                all_ok = False

    if all_ok:
        print("VERDICT: DATA LOOKS GOOD")
    else:
        print("VERDICT: ISSUES FOUND (see above)")

    return all_ok


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global _total

    parser = argparse.ArgumentParser(description="Download square bar data")
    parser.add_argument("--symbols", type=int, default=1000, help="Number of symbols (default: 1000)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download threads (default: 8)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")
    parser.add_argument("--no-delete", action="store_true", help="Don't delete existing data first")
    parser.add_argument("--resume", action="store_true", help="Skip symbols that already have all 4 TFs")
    args = parser.parse_args()

    if not os.getenv("MASSIVE_API_KEY"):
        print("ERROR: MASSIVE_API_KEY not set in .env")
        sys.exit(1)

    if args.verify_only:
        ok = verify_data()
        sys.exit(0 if ok else 1)

    # 1. Get symbols
    symbols = get_top_symbols(args.symbols)
    print(f"\nTarget: {len(symbols)} symbols × {len(TIMEFRAMES)} timeframes = {len(symbols) * len(TIMEFRAMES)} downloads")
    print(f"Date range: {DATE_FROM} to {DATE_TO}")

    # 2. Save symbol list for reference
    BAR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (BAR_DATA_DIR / "_symbol_list.json").write_text(json.dumps({
        "symbols": symbols,
        "count": len(symbols),
        "generated_at": datetime.now().isoformat(),
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
    }, indent=2))

    # 3. Delete ALL old bar data (unless --no-delete or --resume)
    if not args.no_delete and not args.resume:
        # Primary cache
        if BAR_DATA_DIR.exists():
            old_count = len(list(BAR_DATA_DIR.glob("*.json")))
            print(f"\nDeleting {old_count} existing files in cache/bar_data/...")
            for f in BAR_DATA_DIR.glob("*.json"):
                if f.name != "_symbol_list.json":
                    f.unlink()
            print("  Done.")

        # Also clear live_data_cache/data/ (fallback source — stale data contaminates sims)
        live_cache = Path("live_data_cache/data")
        if live_cache.exists():
            live_count = 0
            for tf_dir in live_cache.iterdir():
                if tf_dir.is_dir():
                    for f in tf_dir.glob("*.json"):
                        f.unlink()
                        live_count += 1
            if live_count:
                print(f"  Cleared {live_count} files from live_data_cache/data/ (stale fallback)")

    # 4. Build download tasks
    tasks = []
    for sym in symbols:
        for tf in TIMEFRAMES:
            if args.resume:
                out_path = BAR_DATA_DIR / f"{sym}_{tf}.json"
                if out_path.exists():
                    continue
            tasks.append((sym, tf))

    _total = len(tasks)
    print(f"\nDownloading {_total} files with {args.workers} workers...")
    print(f"Estimated time: {_total / CALLS_PER_SECOND / 60:.0f} minutes\n")

    t0 = time.time()

    # 5. Download in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, sym, tf): (sym, tf) for sym, tf in tasks}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                sym, tf = futures[future]
                print(f"  EXCEPTION {sym} {tf}: {e}")
                results.append((sym, tf, 0))

    elapsed = time.time() - t0
    successful = sum(1 for _, _, n in results if n > 0)
    failed = sum(1 for _, _, n in results if n == 0)

    print(f"\n{'='*70}")
    print(f"DOWNLOAD COMPLETE")
    print(f"  Time:    {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Success: {successful}/{_total}")
    print(f"  Failed:  {failed}/{_total}")

    # 6. Retry failures once
    failures = [(sym, tf) for sym, tf, n in results if n == 0]
    if failures:
        print(f"\nRetrying {len(failures)} failures...")
        _total = len(failures)
        with _progress_lock:
            global _completed, _failed
            _completed = 0
            _failed = 0

        retry_results = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(download_one, sym, tf): (sym, tf) for sym, tf in failures}
            for future in as_completed(futures):
                try:
                    retry_results.append(future.result())
                except Exception:
                    pass

        still_failed = [(sym, tf) for sym, tf, n in retry_results if n == 0]
        if still_failed:
            print(f"\n  Still failed after retry ({len(still_failed)}):")
            for sym, tf in still_failed[:20]:
                print(f"    {sym} {tf}")

    # 7. Verify
    verify_data(symbols)


if __name__ == "__main__":
    main()
