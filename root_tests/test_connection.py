# test_connection.py (in project root)
from backend.data.massive_client import fetch_bars

bars = fetch_bars("AAPL", timeframe="1d", days_back=5)
print(f"Got {len(bars.bars)} bars for {bars.symbol}")
for b in bars.bars[-3:]:
    print(f"  {b.timestamp.date()} | O:{b.open} H:{b.high} L:{b.low} C:{b.close} V:{b.volume:,}")