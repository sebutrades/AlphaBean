"""
live_data_cache/bar_updater.py — Incremental bar fetcher

Fetches new bars for tracked symbols and stores them via bar_store.
Only fetches bars that arrived since the last stored timestamp — the API
call is cheap (just a handful of new candles after the first backfill).

Update cadence (called by live_feed.py):
    Hot list  (in-play + active trades): 5-min bars every 5 min
                                         15-min bars every 15 min
    Full list (top 500):                 5-min bars every 30 min
                                         15-min + 1-h bars every 60 min
    Daily bars:                          once at 4:30 PM ET (full universe)

First run per symbol: full backfill (BACKFILL_DAYS from bar_store.py).
Subsequent runs:      incremental — fetch from last stored timestamp only.

Thread pool (max_workers=20) keeps the whole hot-list update under ~3 s.
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

from live_data_cache.bar_store import (
    get_last_timestamp, append_bars, needs_backfill,
    BACKFILL_DAYS, get_store_stats,
)
from live_data_cache.watchlist import get_all_tracked, get_hot_list, get_inplay_symbols


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_new_bars(symbol: str, tf: str, since: Optional[datetime]) -> list:
    """
    Retrieve bars from the Massive API.
    If `since` is None (no existing data), do a full BACKFILL_DAYS fetch.
    Otherwise, fetch only the days from `since` to now, then filter.
    Returns list[Bar].
    """
    from backend.data.massive_client import fetch_bars

    if since is None or needs_backfill(symbol, tf):
        # First-time backfill
        days = BACKFILL_DAYS.get(tf, 14)
        result = fetch_bars(symbol, tf, days_back=days)
        return result.bars if result else []

    # Incremental: compute calendar days since the last stored bar
    delta_days = (datetime.now().date() - since.date()).days + 1
    delta_days  = max(delta_days, 1)

    result = fetch_bars(symbol, tf, days_back=delta_days)
    if not result:
        return []

    # Filter out bars already stored (≤ since) — keeps the append idempotent
    return [b for b in result.bars if b.timestamp > since]


def _update_one(symbol: str, tf: str) -> dict:
    """Update a single symbol for one timeframe. Returns a status dict."""
    t0 = time.time()
    try:
        since    = get_last_timestamp(symbol, tf)
        new_bars = _fetch_new_bars(symbol, tf, since)

        if not new_bars:
            return {
                "symbol": symbol, "tf": tf,
                "added": 0, "status": "no_new_bars",
                "elapsed": round(time.time() - t0, 2),
            }

        added = append_bars(symbol, tf, new_bars)
        return {
            "symbol": symbol, "tf": tf,
            "added": added, "status": "ok",
            "elapsed": round(time.time() - t0, 2),
        }
    except Exception as e:
        return {
            "symbol": symbol, "tf": tf,
            "added": 0, "status": "error",
            "error": str(e)[:120],
            "elapsed": round(time.time() - t0, 2),
        }


# ── Public API ────────────────────────────────────────────────────────────────

def update_timeframe(
    tf: str,
    symbols: Optional[list[str]] = None,
    max_workers: int = 20,
) -> dict:
    """
    Update all (or specified) symbols for one timeframe.

    Returns a summary dict with counts of symbols updated, bars added,
    errors, and elapsed time — suitable for logging and the feed status.
    """
    t_start  = time.time()
    sym_list = symbols if symbols is not None else get_all_tracked()

    if not sym_list:
        return {"tf": tf, "symbols_updated": 0, "total_added": 0, "errors": 0,
                "elapsed_s": 0.0}

    total_added = 0
    errors      = 0

    with ThreadPoolExecutor(max_workers=min(max_workers, len(sym_list))) as ex:
        futures = {ex.submit(_update_one, sym, tf): sym for sym in sym_list}
        for fut in as_completed(futures):
            r = fut.result()
            if r["status"] == "error":
                errors += 1
            else:
                total_added += r.get("added", 0)

    return {
        "tf":              tf,
        "symbols_updated": len(sym_list),
        "total_added":     total_added,
        "errors":          errors,
        "elapsed_s":       round(time.time() - t_start, 2),
    }


def update_hot_list(tf: str, max_workers: int = 20) -> dict:
    """Convenience: update only the hot list (in-play + active trades)."""
    return update_timeframe(tf, symbols=get_hot_list(), max_workers=max_workers)


def backfill_symbol(symbol: str, timeframes: Optional[list[str]] = None) -> dict:
    """
    Force a full backfill for a symbol across all (or specified) timeframes.
    Useful on first startup or when a symbol is added mid-session.
    """
    if timeframes is None:
        timeframes = ["5min", "15min", "1h", "1d"]

    results = {}
    for tf in timeframes:
        from backend.data.massive_client import fetch_bars
        days = BACKFILL_DAYS.get(tf, 14)
        try:
            bars_series = fetch_bars(symbol, tf, days_back=days)
            bars = bars_series.bars if bars_series else []
            added = append_bars(symbol, tf, bars)
            results[tf] = {"added": added, "status": "ok"}
        except Exception as e:
            results[tf] = {"status": "error", "error": str(e)[:80]}

    return {"symbol": symbol, "timeframes": results}


def ensure_hot_list_ready(timeframes: Optional[list[str]] = None) -> dict:
    """
    Called on startup: backfill any hot-list symbol that has no data yet.
    Runs sequentially to avoid hammering the API on cold start.
    Returns a summary of what was backfilled.
    """
    if timeframes is None:
        timeframes = ["5min", "15min", "1h", "1d"]

    hot     = get_hot_list()
    filled  = 0
    skipped = 0

    for sym in hot:
        missing_tfs = [tf for tf in timeframes if needs_backfill(sym, tf)]
        if missing_tfs:
            backfill_symbol(sym, missing_tfs)
            filled += 1
        else:
            skipped += 1

    return {"symbols": len(hot), "backfilled": filled, "already_cached": skipped}


def get_health() -> dict:
    """Bar store health summary for monitoring endpoints."""
    return get_store_stats()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json as _json

    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        print(_json.dumps(get_health(), indent=2))

    elif cmd == "backfill":
        # python -m live_data_cache.bar_updater backfill NVDA AAPL TSLA
        symbols = sys.argv[2:] or None
        if symbols:
            for sym in symbols:
                result = backfill_symbol(sym)
                print(f"{sym}: {result}")
        else:
            result = ensure_hot_list_ready()
            print(f"Hot-list backfill complete: {result}")

    elif cmd == "update":
        # python -m live_data_cache.bar_updater update 5min
        tf = sys.argv[2] if len(sys.argv) > 2 else "5min"
        result = update_timeframe(tf)
        print(f"Updated {tf}: {result}")

    elif cmd == "hot":
        # python -m live_data_cache.bar_updater hot
        for tf in ["5min", "15min"]:
            result = update_hot_list(tf)
            print(f"Hot-list {tf}: +{result['total_added']} bars in {result['elapsed_s']}s")

    else:
        print("Usage: python -m live_data_cache.bar_updater [status|backfill [SYM ...]|update TF|hot]")
