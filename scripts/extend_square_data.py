"""
scripts/extend_square_data.py — Incrementally extend existing bar data.

For every file in cache/bar_data/, fetches new bars from the day after
the file's last bar through today, appends them, and saves back.

Much faster than a full re-download — only pulls the gap.

Usage:
    python scripts/extend_square_data.py
    python scripts/extend_square_data.py --to 2026-04-15
    python scripts/extend_square_data.py --workers 10
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

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


BAR_DATA_DIR = Path("cache/bar_data")

TIMEFRAMES = {
    "5min":  (5, "minute"),
    "15min": (15, "minute"),
    "1h":    (1, "hour"),
    "1d":    (1, "day"),
}

MARKET_OPEN_H, MARKET_OPEN_M = 9, 30
MARKET_CLOSE_H, MARKET_CLOSE_M = 16, 0

CALLS_PER_SECOND = 4
_rate_lock = Lock()
_last_call_time = 0.0

_progress_lock = Lock()
_completed = 0
_added_bars = 0
_skipped = 0
_failed = 0
_total = 0


def _rate_limit():
    global _last_call_time
    with _rate_lock:
        now = time.time()
        min_interval = 1.0 / CALLS_PER_SECOND
        wait = min_interval - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.time()


def _progress(symbol: str, tf: str, added: int, status: str):
    global _completed, _added_bars, _skipped, _failed
    with _progress_lock:
        _completed += 1
        if status == "ok":
            _added_bars += added
        elif status == "skip":
            _skipped += 1
        elif status == "fail":
            _failed += 1
        pct = (_completed / _total * 100) if _total > 0 else 0
        tag = f"+{added:>4} bars" if status == "ok" else status.upper()
        print(f"  [{_completed:>4}/{_total}] {pct:5.1f}%  {symbol:<6} {tf:<5}  {tag}", flush=True)


def extend_file(file_path: Path, date_to: str, retries: int = 3) -> tuple[str, str, int, str]:
    """Extend one cache file with new bars through date_to."""
    stem = file_path.stem  # e.g., "SPY_5min"
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return (stem, "?", 0, "skip")
    symbol, tf = parts
    if tf not in TIMEFRAMES:
        return (symbol, tf, 0, "skip")

    try:
        data = json.loads(file_path.read_text())
    except Exception:
        return (symbol, tf, 0, "fail")

    bars = data.get("bars", [])
    if not bars:
        return (symbol, tf, 0, "skip")

    last_ts = bars[-1]["t"]  # "2026-04-01T15:55:00"
    last_date = last_ts[:10]

    # Figure out fetch start — day after last bar
    start_dt = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
    start_date = start_dt.strftime("%Y-%m-%d")

    if start_date > date_to:
        return (symbol, tf, 0, "skip")  # already up to date

    multiplier, timespan = TIMEFRAMES[tf]

    for attempt in range(retries):
        try:
            _rate_limit()
            client = RESTClient(api_key=os.getenv("MASSIVE_API_KEY"))

            raw_bars = []
            for agg in client.list_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=date_to,
                limit=50000,
                sort="asc",
            ):
                raw_bars.append(agg)

            # Convert to ET, filter, format
            new_bars = []
            for agg in raw_bars:
                utc_dt = datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc)
                et_dt = utc_dt.astimezone(ET)
                et_naive = et_dt.replace(tzinfo=None)

                if tf != "1d":
                    bar_minutes = et_naive.hour * 60 + et_naive.minute
                    open_minutes = MARKET_OPEN_H * 60 + MARKET_OPEN_M
                    close_minutes = MARKET_CLOSE_H * 60 + MARKET_CLOSE_M
                    if bar_minutes < open_minutes or bar_minutes >= close_minutes:
                        continue

                ts = et_naive.strftime("%Y-%m-%dT%H:%M:%S")

                # Dedupe against existing tail
                if ts <= last_ts:
                    continue

                new_bars.append({
                    "t": ts,
                    "o": round(agg.open, 4),
                    "h": round(agg.high, 4),
                    "l": round(agg.low, 4),
                    "c": round(agg.close, 4),
                    "v": int(agg.volume),
                })

            if not new_bars:
                return (symbol, tf, 0, "skip")

            # Append and save
            bars.extend(new_bars)
            data["bars"] = bars
            data["n_bars"] = len(bars)
            data["date_to"] = date_to
            data["cached_at"] = datetime.now().isoformat()
            file_path.write_text(json.dumps(data, separators=(",", ":")))

            return (symbol, tf, len(new_bars), "ok")

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return (symbol, tf, 0, "fail")

    return (symbol, tf, 0, "fail")


def extend_one(file_path: Path, date_to: str):
    sym, tf, added, status = extend_file(file_path, date_to)
    _progress(sym, tf, added, status)
    return (sym, tf, added, status)


def verify_extension(date_to: str):
    """Verify all files extended successfully."""
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    tf_stats: dict[str, dict] = {tf: {"files": 0, "at_target": 0, "last_dates": {}} for tf in TIMEFRAMES}

    for f in BAR_DATA_DIR.glob("*.json"):
        if f.name == "_symbol_list.json":
            continue
        parts = f.stem.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in TIMEFRAMES:
            continue
        tf = parts[1]
        try:
            data = json.loads(f.read_text())
            bars = data.get("bars", [])
            if not bars:
                continue
            last = bars[-1]["t"][:10]
            tf_stats[tf]["files"] += 1
            tf_stats[tf]["last_dates"][last] = tf_stats[tf]["last_dates"].get(last, 0) + 1
            if last >= date_to:
                tf_stats[tf]["at_target"] += 1
        except Exception:
            pass

    all_ok = True
    for tf in ["1d", "5min", "15min", "1h"]:
        s = tf_stats[tf]
        print(f"\n{tf}:")
        print(f"  Files:       {s['files']}")
        print(f"  At target:   {s['at_target']}/{s['files']}")
        print(f"  End dates:")
        for d, c in sorted(s["last_dates"].items(), reverse=True)[:5]:
            print(f"    {d}: {c}")
        if s["at_target"] < s["files"] * 0.95:
            all_ok = False

    print(f"\n{'=' * 70}")
    if all_ok:
        print(f"VERDICT: DATA EXTENDED SUCCESSFULLY")
    else:
        print(f"VERDICT: Some files may not have reached target date (expected if markets closed)")


def main():
    global _total

    parser = argparse.ArgumentParser(description="Extend existing bar data to a newer date")
    parser.add_argument("--to", type=str, default=None,
                        help="Target end date YYYY-MM-DD (default: today)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel threads")
    args = parser.parse_args()

    if not os.getenv("MASSIVE_API_KEY"):
        print("ERROR: MASSIVE_API_KEY not set in .env")
        sys.exit(1)

    date_to = args.to or datetime.now().strftime("%Y-%m-%d")
    print(f"Extending bar data through {date_to}")

    files = [f for f in BAR_DATA_DIR.glob("*.json") if f.name != "_symbol_list.json"]
    _total = len(files)
    print(f"Files to process: {_total}")
    print(f"Workers: {args.workers}\n")

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extend_one, f, date_to) for f in files]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  EXCEPTION: {e}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"EXTENSION COMPLETE")
    print(f"  Time:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Processed:  {_completed}/{_total}")
    print(f"  Bars added: {_added_bars:,}")
    print(f"  Skipped:    {_skipped} (already up to date or no new bars)")
    print(f"  Failed:     {_failed}")

    verify_extension(date_to)


if __name__ == "__main__":
    main()
