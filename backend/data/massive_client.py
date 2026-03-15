"""
massive_client.py — Fetches price data from Massive.com (formerly Polygon.io)

AlphaBean only uses 3 timeframes: 15min, 1h, 1d.
"""
from massive import RESTClient
from datetime import datetime, timedelta
from backend.data.schemas import Bar, BarSeries
import os
from dotenv import load_dotenv

load_dotenv()


def get_client() -> RESTClient:
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise ValueError("MASSIVE_API_KEY not found in .env file")
    return RESTClient(api_key=api_key)


# Only these 3 timeframes are supported in AlphaBean
TIMEFRAME_MAP = {
    "15min": (15, "minute"),
    "1h":    (1, "hour"),
    "1d":    (1, "day"),
}

VALID_TIMEFRAMES = list(TIMEFRAME_MAP.keys())


def fetch_bars(
    symbol: str,
    timeframe: str = "1d",
    days_back: int = 30,
) -> BarSeries:
    """
    Fetch OHLCV bars from Massive.com for a given symbol.
    Only supports: 15min, 1h, 1d.
    """
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(
            f"Invalid timeframe '{timeframe}'. AlphaBean only supports: {VALID_TIMEFRAMES}"
        )

    client = get_client()
    multiplier, timespan = TIMEFRAME_MAP[timeframe]

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
        bars.append(Bar(
            symbol=symbol.upper(),
            timestamp=datetime.utcfromtimestamp(agg.timestamp / 1000),
            open=agg.open,
            high=agg.high,
            low=agg.low,
            close=agg.close,
            volume=int(agg.volume),
            vwap=getattr(agg, 'vwap', None),
            trade_count=getattr(agg, 'transactions', None),
        ))

    return BarSeries(symbol=symbol.upper(), timeframe=timeframe, bars=bars)


def fetch_snapshot(symbol: str) -> dict:
    """Fetch current snapshot for a symbol."""
    client = get_client()
    snapshot = client.get_snapshot_ticker("stocks", symbol.upper())
    return {
        "symbol": symbol.upper(),
        "last_price": snapshot.ticker.last_trade.price,
        "today_change_pct": snapshot.ticker.todays_change_percent,
        "volume": snapshot.ticker.day.volume,
        "prev_close": snapshot.ticker.prev_day.close,
    }