"""Probe Polygon news, options, and financials API endpoints."""
from massive import RESTClient
import os
from dotenv import load_dotenv
load_dotenv()

client = RESTClient(os.getenv("MASSIVE_API_KEY"))

print("=" * 60)
print("TEST 1: list_ticker_news('AAPL', limit=2)")
print("=" * 60)
try:
    items = []
    for i, item in enumerate(client.list_ticker_news(ticker="AAPL", limit=2)):
        items.append(item)
        if i >= 1:
            break
    print(f"  Got {len(items)} news items")
    if items:
        first = items[0]
        print(f"  Type: {type(first)}")
        print(f"  Attrs: {[a for a in dir(first) if not a.startswith('_')]}")
        print(f"  Item: {first}")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 60)
print("TEST 2: list_snapshot_options_chain('AAPL') - first 2")
print("=" * 60)
try:
    items = []
    for i, item in enumerate(client.list_snapshot_options_chain("AAPL")):
        items.append(item)
        if i >= 1:
            break
    print(f"  Got {len(items)} options contracts")
    if items:
        first = items[0]
        print(f"  Type: {type(first)}")
        print(f"  Attrs: {[a for a in dir(first) if not a.startswith('_')]}")
        print(f"  First: {str(first)[:600]}")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 60)
print("TEST 3: list_financials_income_statements('AAPL', limit=1)")
print("=" * 60)
try:
    items = []
    for i, item in enumerate(client.list_financials_income_statements(ticker="AAPL", limit=1)):
        items.append(item)
        if i >= 0:
            break
    print(f"  Got {len(items)} statements")
    if items:
        first = items[0]
        print(f"  Type: {type(first)}")
        print(f"  Attrs: {[a for a in dir(first) if not a.startswith('_')]}")
        print(f"  First: {str(first)[:600]}")
except Exception as e:
    print(f"  ERROR: {e}")