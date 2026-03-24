"""Probe the snapshot and grouped daily endpoints to see return format."""
from massive import RESTClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
load_dotenv()

client = RESTClient(os.getenv("MASSIVE_API_KEY"))

# ── Test 1: get_snapshot_all ──
print("=" * 60)
print("TEST 1: get_snapshot_all('stocks')")
print("=" * 60)
try:
    result = client.get_snapshot_all("stocks")
    print(f"  Type: {type(result)}")
    
    # If it's iterable, grab first few
    if hasattr(result, '__iter__'):
        items = []
        for i, item in enumerate(result):
            items.append(item)
            if i >= 2:
                break
        print(f"  Got {len(items)} items (stopped at 3)")
        if items:
            first = items[0]
            print(f"  First item type: {type(first)}")
            print(f"  First item attrs: {[a for a in dir(first) if not a.startswith('_')]}")
            print(f"  First item: {first}")
    else:
        print(f"  Result: {str(result)[:500]}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Test 2: get_grouped_daily_aggs ──
print(f"\n{'=' * 60}")
print("TEST 2: get_grouped_daily_aggs")
print("=" * 60)
try:
    # Try with a recent date
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"  Date: {yesterday}")
    result = client.get_grouped_daily_aggs(yesterday)
    print(f"  Type: {type(result)}")
    
    if hasattr(result, '__iter__'):
        items = []
        for i, item in enumerate(result):
            items.append(item)
            if i >= 2:
                break
        print(f"  Got {len(items)} items (stopped at 3)")
        if items:
            first = items[0]
            print(f"  First item type: {type(first)}")
            print(f"  First item attrs: {[a for a in dir(first) if not a.startswith('_')]}")
            print(f"  First item: {first}")
    else:
        print(f"  Result: {str(result)[:500]}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Test 3: list_tickers (backup) ──
print(f"\n{'=' * 60}")
print("TEST 3: list_tickers (first 3)")
print("=" * 60)
try:
    items = []
    for i, t in enumerate(client.list_tickers(market="stocks", active=True, limit=3)):
        items.append(t)
        if i >= 2:
            break
    print(f"  Got {len(items)} items")
    if items:
        first = items[0]
        print(f"  First item type: {type(first)}")
        print(f"  First item attrs: {[a for a in dir(first) if not a.startswith('_')]}")
        print(f"  First item: {first}")
except Exception as e:
    print(f"  ERROR: {e}")