"""
fetch_symbols.py — Fetch top N US equities ranked by dollar volume.

Uses Massive.com (Polygon) API:
  1. get_snapshot_all("stocks") → all tickers with prev_day price/volume
  2. Filter: common stocks, price >= $5, volume >= 500K
  3. Rank by dollar volume (close * volume)
  4. Cache to cache/top_symbols.json

Usage:
  python fetch_symbols.py                # Top 500 (default)
  python fetch_symbols.py --count 200    # Top 200
  python fetch_symbols.py --refresh      # Force refresh
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from massive import RESTClient
import os
from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path("cache/top_symbols.json")
MIN_PRICE = 5.0
MIN_VOLUME = 500_000


def get_client() -> RESTClient:
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise ValueError("MASSIVE_API_KEY not found in .env file")
    return RESTClient(api_key=api_key)


def is_valid_ticker(ticker: str) -> bool:
    """Keep only clean common-stock tickers (1-5 uppercase letters)."""
    if not ticker or len(ticker) > 5:
        return False
    if not ticker.isalpha():
        return False
    if len(ticker) > 1 and ticker[-1] in ("W", "U", "R"):
        return False
    return True


def fetch_top_symbols(count: int = 500, verbose: bool = True) -> list[dict]:
    client = get_client()

    if verbose:
        print(f"\n  Fetching all US stock snapshots...")

    snapshots = client.get_snapshot_all("stocks")

    if verbose:
        print(f"  Got {len(snapshots)} ticker snapshots")

    ranked = []
    skipped = {"no_prev": 0, "low_price": 0, "low_vol": 0, "bad_ticker": 0}

    for snap in snapshots:
        ticker = snap.ticker
        if not is_valid_ticker(ticker):
            skipped["bad_ticker"] += 1
            continue

        prev = snap.prev_day
        if prev is None:
            skipped["no_prev"] += 1
            continue

        close = prev.close or 0
        volume = prev.volume or 0

        if close < MIN_PRICE:
            skipped["low_price"] += 1
            continue
        if volume < MIN_VOLUME:
            skipped["low_vol"] += 1
            continue

        dollar_vol = close * volume
        ranked.append({
            "symbol": ticker,
            "close": round(close, 2),
            "volume": int(volume),
            "dollar_volume": round(dollar_vol, 0),
            "vwap": round(prev.vwap, 2) if prev.vwap else None,
            "change_pct": round(snap.todays_change_percent, 2) if snap.todays_change_percent else None,
        })

    ranked.sort(key=lambda x: x["dollar_volume"], reverse=True)
    top = ranked[:count]

    if verbose:
        print(f"  Filtered: {len(ranked)} valid stocks "
              f"(skipped: {skipped['bad_ticker']} bad ticker, "
              f"{skipped['no_prev']} no data, "
              f"{skipped['low_price']} < ${MIN_PRICE}, "
              f"{skipped['low_vol']} < {MIN_VOLUME:,} vol)")

        print(f"\n  Top {min(count, len(top))} by dollar volume:")
        print(f"  {'─' * 64}")
        print(f"  {'#':>4}  {'Symbol':<8}  {'Close':>10}  {'Volume':>14}  {'$ Volume':>16}")
        print(f"  {'─' * 64}")
        for i, r in enumerate(top[:25]):
            print(f"  {i+1:>4}  {r['symbol']:<8}  ${r['close']:>9,.2f}  "
                  f"{r['volume']:>14,}  ${r['dollar_volume']:>14,.0f}")
        if len(top) > 25:
            print(f"  {'...':>4}")
            for r in top[-3:]:
                idx = top.index(r) + 1
                print(f"  {idx:>4}  {r['symbol']:<8}  ${r['close']:>9,.2f}  "
                      f"{r['volume']:>14,}  ${r['dollar_volume']:>14,.0f}")
        print(f"  {'─' * 64}")

    return top


def save_symbols(top: list[dict]):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    symbols = [r["symbol"] for r in top]
    data = {
        "symbols": symbols,
        "count": len(symbols),
        "fetched_at": datetime.now().isoformat(),
        "min_price": MIN_PRICE,
        "min_volume": MIN_VOLUME,
        "details": top,
    }
    CACHE_PATH.write_text(json.dumps(data, indent=2))
    print(f"\n  ✓ Saved {len(symbols)} symbols to {CACHE_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Fetch top US stocks by dollar volume")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.refresh and CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text())
            cached_count = cached.get("count", 0)
            cached_at = cached.get("fetched_at", "unknown")
            if cached_count >= args.count:
                print(f"\n  Using cached symbols ({cached_count} from {cached_at})")
                print(f"  Use --refresh to update")
                return
        except Exception:
            pass

    if not args.quiet:
        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  Juicer — Fetch Top {args.count:<4} Symbols         ║")
        print(f"  ╠══════════════════════════════════════════╣")
        print(f"  ║  Source:     Massive.com snapshots        ║")
        print(f"  ║  Min price:  ${MIN_PRICE:<29}║")
        print(f"  ║  Min vol:    {MIN_VOLUME:>10,}                  ║")
        print(f"  ║  Ranked by:  Dollar volume                ║")
        print(f"  ╚══════════════════════════════════════════╝")

    t0 = time.time()
    top = fetch_top_symbols(count=args.count, verbose=not args.quiet)
    save_symbols(top)

    if not args.quiet:
        print(f"  Time: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()