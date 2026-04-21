"""
Test A: KTI Data Fetch & Cache Test
====================================
Fetches historical + live price data via Karb (kti price CLI),
aggregates 5-min bars from 1-min bars, and caches everything locally.

Tests:
  A) Historical daily bars   -- 10 symbols, 3 months
  B) Historical minute bars  -- 10 symbols, 1 full day
  C) 5-minute bar aggregation from 1-min bars
  D) Live polling mode       -- fetches latest minute bars every 60s

Usage:
  python test_data_fetch.py                # Run tests A, B, C
  python test_data_fetch.py --live         # Run test D (live polling)
  python test_data_fetch.py --live --interval 30   # Poll every 30s
"""

import subprocess
import json
import os
import sys
import time
import math
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "MA"]

CACHE_DIR = Path(__file__).parent / "cache"

HISTORY_MONTHS = 3

# How many symbols to batch per kti call (avoid rate limits)
BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------

class Term:
    """Simple terminal formatting without external deps."""

    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    CLEAR_LINE = "\r\033[K"

    @staticmethod
    def header(text):
        width = 70
        print()
        print(f"{Term.BOLD}{Term.CYAN}{'=' * width}")
        print(f"  {text}")
        print(f"{'=' * width}{Term.RESET}")
        print()

    @staticmethod
    def subheader(text):
        print(f"\n{Term.BOLD}{Term.MAGENTA}--- {text} ---{Term.RESET}\n")

    @staticmethod
    def ok(text):
        print(f"  {Term.GREEN}[OK]{Term.RESET}  {text}")

    @staticmethod
    def warn(text):
        print(f"  {Term.YELLOW}[!!]{Term.RESET}  {text}")

    @staticmethod
    def fail(text):
        print(f"  {Term.RED}[FAIL]{Term.RESET}  {text}")

    @staticmethod
    def info(text):
        print(f"  {Term.DIM}[..]{Term.RESET}  {text}")

    @staticmethod
    def progress(current, total, label=""):
        pct = current / total * 100 if total else 0
        bar_len = 30
        filled = int(bar_len * current / total) if total else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(
            f"{Term.CLEAR_LINE}  [{bar}] {pct:5.1f}%  {current}/{total}  {label}"
        )
        sys.stdout.flush()
        if current >= total:
            print()  # newline when done

    @staticmethod
    def table(headers, rows, col_widths=None):
        if not col_widths:
            col_widths = []
            for i, h in enumerate(headers):
                max_w = len(str(h))
                for row in rows:
                    if i < len(row):
                        max_w = max(max_w, len(str(row[i])))
                col_widths.append(min(max_w + 2, 20))

        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print(f"  {Term.BOLD}{fmt.format(*headers)}{Term.RESET}")
        print(f"  {'-' * sum(col_widths + [2 * (len(col_widths) - 1)])}")
        for row in rows:
            print(f"  {fmt.format(*[str(c) for c in row])}")


# ---------------------------------------------------------------------------
# KTI CLI wrappers
# ---------------------------------------------------------------------------

def run_kti(args, timeout=120):
    """Run a kti command and return parsed JSON or None on failure."""
    cmd = ["kti"] + args + ["--json"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return None, result.stderr.strip()
        return json.loads(result.stdout), None
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


def fetch_daily_bars(symbols, start_date, end_date):
    """Fetch daily bars for a list of symbols over a date range."""
    sym_str = ",".join(symbols)
    data, err = run_kti([
        "price", "daily", sym_str,
        "--start-date", start_date,
        "--end-date", end_date,
    ], timeout=180)
    return data, err


def fetch_minute_bars(symbols, date=None, minutes=None):
    """Fetch minute bars for a list of symbols."""
    sym_str = ",".join(symbols)
    args = ["price", "minute", sym_str]
    if date:
        args += ["--date", date]
    elif minutes:
        args += ["--minutes", str(minutes)]
    data, err = run_kti(args, timeout=180)
    return data, err


# ---------------------------------------------------------------------------
# Bar aggregation
# ---------------------------------------------------------------------------

def aggregate_to_5min(minute_bars):
    """
    Aggregate 1-minute bars into 5-minute bars.
    Groups by the 5-minute bucket each bar falls into.
    Returns list of aggregated bar dicts.
    """
    buckets = defaultdict(list)
    for bar in minute_bars:
        ts = datetime.fromisoformat(bar["timestamp"])
        # Floor to nearest 5-minute boundary
        floored_minute = (ts.minute // 5) * 5
        bucket_ts = ts.replace(minute=floored_minute, second=0, microsecond=0)
        buckets[bucket_ts].append(bar)

    agg_bars = []
    for bucket_ts in sorted(buckets.keys()):
        bars_in_bucket = buckets[bucket_ts]
        agg_bars.append({
            "timestamp": bucket_ts.isoformat(),
            "open": bars_in_bucket[0]["open"],
            "high": max(b["high"] for b in bars_in_bucket),
            "low": min(b["low"] for b in bars_in_bucket),
            "close": bars_in_bucket[-1]["close"],
            "volume": sum(b["volume"] for b in bars_in_bucket),
            "bar_count": len(bars_in_bucket),
        })
    return agg_bars


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def save_cache(data, filename):
    """Save data to JSON cache file."""
    filepath = CACHE_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return filepath


def load_cache(filename):
    """Load data from JSON cache file, or return None."""
    filepath = CACHE_DIR / filename
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Test A: Historical daily bars
# ---------------------------------------------------------------------------

def test_historical_daily():
    Term.header("TEST A: Historical Daily Bars (3 months, 10 symbols)")

    today = datetime.now()
    start = today - timedelta(days=HISTORY_MONTHS * 30)
    start_str = start.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    Term.info(f"Symbols:    {', '.join(SYMBOLS)}")
    Term.info(f"Date range: {start_str} -> {end_str}")
    Term.info(f"Fetching daily bars via Karb (kti price daily)...")
    print()

    t0 = time.time()

    # Fetch in batches
    all_bars = {}
    for i in range(0, len(SYMBOLS), BATCH_SIZE):
        batch = SYMBOLS[i:i + BATCH_SIZE]
        Term.progress(i, len(SYMBOLS), f"Batch: {','.join(batch)}")
        data, err = fetch_daily_bars(batch, start_str, end_str)
        if err:
            Term.fail(f"Fetch error for {','.join(batch)}: {err}")
            continue
        if data and "bars" in data:
            all_bars.update(data["bars"])

    Term.progress(len(SYMBOLS), len(SYMBOLS), "Done")
    elapsed = time.time() - t0

    # Summarize results
    Term.subheader("Results")

    rows = []
    total_bars = 0
    for sym in SYMBOLS:
        bars = all_bars.get(sym, [])
        count = len(bars)
        total_bars += count
        if count > 0:
            first_date = bars[0]["timestamp"][:10]
            last_date = bars[-1]["timestamp"][:10]
            last_close = bars[-1]["close"]
            rows.append([sym, count, first_date, last_date, f"${last_close:.2f}"])
            Term.ok(f"{sym:6s}  {count:4d} bars  {first_date} -> {last_date}  last=${last_close:.2f}")
        else:
            rows.append([sym, 0, "-", "-", "-"])
            Term.fail(f"{sym:6s}  NO DATA")

    print()
    Term.info(f"Total bars:  {total_bars}")
    Term.info(f"Fetch time:  {elapsed:.1f}s")
    Term.info(f"Throughput:  {total_bars / elapsed:.0f} bars/sec")

    # Cache it
    cache_file = "daily_bars.json"
    path = save_cache(all_bars, cache_file)
    Term.ok(f"Cached to {path}")

    return all_bars


# ---------------------------------------------------------------------------
# Test B: Historical minute bars (1 full day)
# ---------------------------------------------------------------------------

def test_historical_minute():
    Term.header("TEST B: Historical Minute Bars (full trading day, 10 symbols)")

    # Use most recent completed trading day
    today = datetime.now()
    if today.weekday() == 0:  # Monday -> use Friday
        target = today - timedelta(days=3)
    elif today.weekday() == 6:  # Sunday -> use Friday
        target = today - timedelta(days=2)
    elif today.weekday() == 5:  # Saturday -> use Friday
        target = today - timedelta(days=1)
    else:
        target = today - timedelta(days=1)  # use yesterday

    target_str = target.strftime("%Y-%m-%d")

    Term.info(f"Symbols:    {', '.join(SYMBOLS)}")
    Term.info(f"Date:       {target_str}")
    Term.info(f"Fetching 1-minute bars via Karb (kti price minute)...")
    print()

    t0 = time.time()

    all_bars = {}
    for i in range(0, len(SYMBOLS), BATCH_SIZE):
        batch = SYMBOLS[i:i + BATCH_SIZE]
        Term.progress(i, len(SYMBOLS), f"Batch: {','.join(batch)}")
        data, err = fetch_minute_bars(batch, date=target_str)
        if err:
            Term.fail(f"Fetch error for {','.join(batch)}: {err}")
            continue
        if data and "bars" in data:
            all_bars.update(data["bars"])

    Term.progress(len(SYMBOLS), len(SYMBOLS), "Done")
    elapsed = time.time() - t0

    # Summarize
    Term.subheader("Results")

    total_bars = 0
    for sym in SYMBOLS:
        bars = all_bars.get(sym, [])
        count = len(bars)
        total_bars += count
        if count > 0:
            first_ts = bars[0]["timestamp"]
            last_ts = bars[-1]["timestamp"]
            open_px = bars[0]["open"]
            close_px = bars[-1]["close"]
            high_px = max(b["high"] for b in bars)
            low_px = min(b["low"] for b in bars)
            total_vol = sum(b["volume"] for b in bars)
            Term.ok(
                f"{sym:6s}  {count:4d} bars  "
                f"O=${open_px:.2f} H=${high_px:.2f} L=${low_px:.2f} C=${close_px:.2f}  "
                f"Vol={total_vol:,.0f}"
            )
        else:
            Term.fail(f"{sym:6s}  NO DATA")

    print()
    Term.info(f"Total 1-min bars:  {total_bars}")
    Term.info(f"Expected ~390/sym: {'PASS' if total_bars > len(SYMBOLS) * 300 else 'LOW COUNT'}")
    Term.info(f"Fetch time:        {elapsed:.1f}s")
    Term.info(f"Throughput:        {total_bars / elapsed:.0f} bars/sec")

    # Cache
    cache_file = f"minute_bars_{target_str}.json"
    path = save_cache(all_bars, cache_file)
    Term.ok(f"Cached to {path}")

    return all_bars, target_str


# ---------------------------------------------------------------------------
# Test C: 5-minute bar aggregation
# ---------------------------------------------------------------------------

def test_5min_aggregation(minute_bars_data):
    Term.header("TEST C: 5-Minute Bar Aggregation")

    Term.info("Aggregating 1-min bars into 5-min bars...")
    print()

    all_5min = {}
    for sym in SYMBOLS:
        bars_1m = minute_bars_data.get(sym, [])
        if not bars_1m:
            Term.warn(f"{sym:6s}  No 1-min bars to aggregate")
            continue

        bars_5m = aggregate_to_5min(bars_1m)
        all_5min[sym] = bars_5m

        expected_5m = math.ceil(len(bars_1m) / 5)
        Term.ok(
            f"{sym:6s}  {len(bars_1m):4d} x 1min -> {len(bars_5m):3d} x 5min  "
            f"(expected ~{expected_5m})"
        )

        # Show first and last 5-min bar as proof
        if bars_5m:
            first = bars_5m[0]
            last = bars_5m[-1]
            Term.info(
                f"         First 5m: {first['timestamp']}  "
                f"O={first['open']:.2f} H={first['high']:.2f} "
                f"L={first['low']:.2f} C={first['close']:.2f} "
                f"V={first['volume']:,.0f} ({first['bar_count']} bars)"
            )
            Term.info(
                f"         Last  5m: {last['timestamp']}  "
                f"O={last['open']:.2f} H={last['high']:.2f} "
                f"L={last['low']:.2f} C={last['close']:.2f} "
                f"V={last['volume']:,.0f} ({last['bar_count']} bars)"
            )

    total_5m = sum(len(v) for v in all_5min.values())
    total_1m = sum(len(minute_bars_data.get(s, [])) for s in SYMBOLS)
    print()
    Term.info(f"Total 1-min bars:  {total_1m}")
    Term.info(f"Total 5-min bars:  {total_5m}")
    Term.info(f"Compression ratio: {total_1m / total_5m:.1f}x" if total_5m else "N/A")

    cache_file = "5min_bars.json"
    path = save_cache(all_5min, cache_file)
    Term.ok(f"Cached to {path}")

    return all_5min


# ---------------------------------------------------------------------------
# Test D: Live polling mode
# ---------------------------------------------------------------------------

def test_live_polling(interval=60):
    Term.header(f"TEST D: Live Polling Mode (every {interval}s)")

    Term.info(f"Symbols:  {', '.join(SYMBOLS)}")
    Term.info(f"Interval: {interval}s")
    Term.info(f"Press Ctrl+C to stop")
    print()

    # Rolling cache: symbol -> list of bars
    cache = defaultdict(list)
    seen_timestamps = defaultdict(set)  # symbol -> set of timestamps already cached
    poll_count = 0

    try:
        while True:
            poll_count += 1
            poll_start = time.time()
            now = datetime.now().strftime("%H:%M:%S")

            print(f"{Term.BOLD}{Term.CYAN}  [Poll #{poll_count}  {now}]{Term.RESET}")

            # Fetch last N minutes to catch any we might have missed
            fetch_window = max(interval // 60 + 2, 5)
            data, err = fetch_minute_bars(SYMBOLS, minutes=fetch_window)

            if err:
                Term.fail(f"Fetch error: {err}")
                Term.info(f"Retrying in {interval}s...")
                time.sleep(interval)
                continue

            if not data or "bars" not in data:
                Term.warn("No bar data returned (market may be closed)")
                Term.info(f"Retrying in {interval}s...")
                time.sleep(interval)
                continue

            new_bars_total = 0
            for sym in SYMBOLS:
                bars = data["bars"].get(sym, [])
                new_count = 0
                for bar in bars:
                    ts = bar["timestamp"]
                    if ts not in seen_timestamps[sym]:
                        seen_timestamps[sym].add(ts)
                        cache[sym].append(bar)
                        new_count += 1
                new_bars_total += new_count

                if new_count > 0:
                    latest = bars[-1]
                    Term.ok(
                        f"{sym:6s}  +{new_count} new  "
                        f"total={len(cache[sym]):5d}  "
                        f"last=${latest['close']:.2f} @ {latest['timestamp'][11:]}"
                    )
                else:
                    Term.info(
                        f"{sym:6s}  +0 new   "
                        f"total={len(cache[sym]):5d}  (no new bars)"
                    )

            poll_elapsed = time.time() - poll_start

            print()
            Term.info(f"New bars this poll: {new_bars_total}")
            Term.info(f"Poll took: {poll_elapsed:.1f}s")
            Term.info(f"Cache totals: {sum(len(v) for v in cache.values())} bars across {len(cache)} symbols")

            # Save cache snapshot
            cache_data = {sym: list(bars) for sym, bars in cache.items()}
            save_cache(cache_data, "live_cache_snapshot.json")
            Term.ok(f"Cache snapshot saved")

            # Aggregate latest 5-min bars and show
            Term.subheader("Latest 5-min aggregated bars")
            for sym in SYMBOLS:
                if len(cache[sym]) >= 5:
                    recent = cache[sym][-10:]  # last 10 minutes
                    bars_5m = aggregate_to_5min(recent)
                    if bars_5m:
                        last_5m = bars_5m[-1]
                        Term.info(
                            f"{sym:6s}  5m: {last_5m['timestamp'][11:]}  "
                            f"O={last_5m['open']:.2f} H={last_5m['high']:.2f} "
                            f"L={last_5m['low']:.2f} C={last_5m['close']:.2f} "
                            f"V={last_5m['volume']:,.0f}"
                        )

            print(f"\n  {Term.DIM}Sleeping {interval}s until next poll...{Term.RESET}\n")
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n{Term.BOLD}  Stopped by user.{Term.RESET}")
        total = sum(len(v) for v in cache.values())
        Term.info(f"Final cache: {total} bars across {len(cache)} symbols in {poll_count} polls")
        cache_data = {sym: list(bars) for sym, bars in cache.items()}
        path = save_cache(cache_data, "live_cache_final.json")
        Term.ok(f"Final cache saved to {path}")


# ---------------------------------------------------------------------------
# Cache summary
# ---------------------------------------------------------------------------

def print_cache_summary():
    Term.header("CACHE SUMMARY")

    if not CACHE_DIR.exists():
        Term.warn("No cache directory found")
        return

    total_size = 0
    for f in sorted(CACHE_DIR.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.1f} KB"
            else:
                size_str = f"{size} B"
            Term.info(f"{f.name:40s}  {size_str:>10s}")

    print()
    if total_size > 1_000_000:
        Term.info(f"Total cache size: {total_size / 1_000_000:.1f} MB")
    else:
        Term.info(f"Total cache size: {total_size / 1_000:.1f} KB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Term.header("KTI DATA FETCH & CACHE TEST")
    Term.info(f"Started at:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    Term.info(f"Symbols:     {len(SYMBOLS)} ({', '.join(SYMBOLS)})")
    Term.info(f"Cache dir:   {CACHE_DIR.resolve()}")
    Term.info(f"Karb server: localhost:4269")

    # Parse args
    live_mode = "--live" in sys.argv
    interval = 60
    if "--interval" in sys.argv:
        idx = sys.argv.index("--interval")
        if idx + 1 < len(sys.argv):
            interval = int(sys.argv[idx + 1])

    if live_mode:
        test_live_polling(interval=interval)
    else:
        # Test A: Historical daily
        daily_bars = test_historical_daily()

        # Test B: Historical minute
        minute_bars, minute_date = test_historical_minute()

        # Test C: 5-min aggregation
        five_min_bars = test_5min_aggregation(minute_bars)

        # Summary
        print_cache_summary()

        Term.header("ALL TESTS COMPLETE")
        Term.ok(f"Daily bars:  {sum(len(v) for v in daily_bars.values())} bars for {len(daily_bars)} symbols")
        Term.ok(f"1-min bars:  {sum(len(v) for v in minute_bars.values())} bars for {len(minute_bars)} symbols")
        Term.ok(f"5-min bars:  {sum(len(v) for v in five_min_bars.values())} bars for {len(five_min_bars)} symbols")
        print()
        Term.info("Next steps:")
        Term.info("  1. Run with --live to test real-time polling")
        Term.info("  2. Upload execution-bridge to Kore for order event tests")
        Term.info(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()


if __name__ == "__main__":
    main()
