"""
massive_client.py — Fetches price data from Massive.com

v3 changes:
  - Scanner uses 5min + 15min (auto, both at once)
  - All 4 timeframes available for backtest
  - fetch_chart_bars() for Lightweight Charts frontend
  - Timestamps converted UTC → ET
"""
from massive import RESTClient
from datetime import datetime, timedelta, timezone
from backend.data.schemas import Bar, BarSeries
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    ET = pytz.timezone("America/New_York")

ALL_TIMEFRAMES = {
    "5min":  (5, "minute"),
    "15min": (15, "minute"),
    "1h":    (1, "hour"),
    "1d":    (1, "day"),
}

SCANNER_TIMEFRAMES = ["5min", "15min"]


def get_client() -> RESTClient:
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise ValueError("MASSIVE_API_KEY not found in .env file")
    return RESTClient(api_key=api_key)


def fetch_bars(symbol: str, timeframe: str = "15min", days_back: int = 10) -> BarSeries:
    if timeframe not in ALL_TIMEFRAMES:
        raise ValueError(f"Invalid timeframe: '{timeframe}'. Use: {list(ALL_TIMEFRAMES.keys())}")

    client = get_client()
    multiplier, timespan = ALL_TIMEFRAMES[timeframe]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    raw_bars = []
    for agg in client.list_aggs(
        ticker=symbol.upper(),
        multiplier=multiplier,
        timespan=timespan,
        from_=start_date.strftime("%Y-%m-%d"),
        to=end_date.strftime("%Y-%m-%d"),
        limit=50000,
    ):
        raw_bars.append(agg)

    bars = []
    for agg in raw_bars:
        utc_dt = datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc)
        et_dt = utc_dt.astimezone(ET)
        et_naive = et_dt.replace(tzinfo=None)

        bars.append(Bar(
            symbol=symbol.upper(),
            timestamp=et_naive,
            open=agg.open, high=agg.high, low=agg.low, close=agg.close,
            volume=int(agg.volume),
            vwap=getattr(agg, 'vwap', None),
            trade_count=getattr(agg, 'transactions', None),
        ))

    return BarSeries(symbol=symbol.upper(), timeframe=timeframe, bars=bars)


def fetch_chart_bars(symbol: str, timeframe: str = "5min", days_back: int = 5) -> list[dict]:
    """Fetch bars formatted for TradingView Lightweight Charts."""
    bars_series = fetch_bars(symbol, timeframe, days_back)
    return [
        {
            "time": int(b.timestamp.timestamp()),
            "open": b.open, "high": b.high, "low": b.low, "close": b.close,
            "volume": b.volume,
        }
        for b in bars_series.bars
    ]