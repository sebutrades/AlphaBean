"""
massive_client.py — Fetches price data from Massive.com (formerly Polygon.io)

The `massive` package was formerly called `polygon-api-client`.
As of Oct 30 2025, Polygon.io rebranded to Massive.com.
The pip package name is now `massive`, and it defaults to api.massive.com.
"""
from massive import RESTClient
from datetime import datetime, timedelta
from backend.data.schemas import Bar, BarSeries
import os
from dotenv import load_dotenv

load_dotenv()  # Reads .env file and loads variables into environment


def get_client() -> RESTClient:
    """Create a Massive.com REST client using your API key."""
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise ValueError("MASSIVE_API_KEY not found in .env file")
    return RESTClient(api_key=api_key)


def fetch_bars(
    symbol: str,
    timeframe: str = "1min",
    days_back: int = 30,
) -> BarSeries:
    """
    Fetch OHLCV bars from Massive.com for a given symbol.
    
    Args:
        symbol: Stock ticker, e.g. "AAPL", "NVDA"
        timeframe: Bar size. Options: "1min", "5min", "15min", "1h", "1d"
        days_back: How many days of history to fetch
    
    Returns:
        BarSeries with all bars normalized into our standard format.
    
    How the Massive API works:
        client.list_aggs() calls:
        GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
        
        It returns objects with attributes:
            .open, .high, .low, .close  — prices
            .volume                      — share volume
            .vwap                        — volume-weighted average price
            .timestamp                   — Unix milliseconds
            .transactions                — number of trades
        
        list_aggs() automatically paginates — if there are more results
        than fit in one response, it fetches all pages for you.
    """
    client = get_client()
    
    # Parse timeframe into multiplier + timespan
    # Massive API wants them separate: multiplier=1, timespan="minute"
    timeframe_map = {
        "1min":  (1, "minute"),
        "5min":  (5, "minute"),
        "15min": (15, "minute"),
        "30min": (30, "minute"),
        "1h":    (1, "hour"),
        "1d":    (1, "day"),
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"Unknown timeframe: {timeframe}. Use one of: {list(timeframe_map.keys())}")
    
    multiplier, timespan = timeframe_map[timeframe]
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Fetch from Massive.com API
    # list_aggs returns an iterator of Agg objects
    raw_bars = []
    for agg in client.list_aggs(
        ticker=symbol.upper(),
        multiplier=multiplier,
        timespan=timespan,
        from_=start_date.strftime("%Y-%m-%d"),
        to=end_date.strftime("%Y-%m-%d"),
        limit=50000,  # Max per page — the client handles pagination
    ):
        raw_bars.append(agg)
    
    # Convert Massive Agg objects → our Bar model
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
    
    return BarSeries(
        symbol=symbol.upper(),
        timeframe=timeframe,
        bars=bars,
    )


def fetch_snapshot(symbol: str) -> dict:
    """
    Fetch current snapshot for a symbol (last price, today's change, volume).
    Useful for filtering 'in play' stocks.
    """
    client = get_client()
    snapshot = client.get_snapshot_ticker("stocks", symbol.upper())
    return {
        "symbol": symbol.upper(),
        "last_price": snapshot.ticker.last_trade.price,
        "today_change_pct": snapshot.ticker.todays_change_percent,
        "volume": snapshot.ticker.day.volume,
        "prev_close": snapshot.ticker.prev_day.close,
    }