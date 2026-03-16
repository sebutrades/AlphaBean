"""
massive_client.py — Fetches price data from Massive.com (formerly Polygon.io)

CHANGES in v2:
  - Restricted to 15min, 1h, 1d timeframes only
  - Timestamps converted to US/Eastern timezone
  - Added news fetching for catalyst reports
"""
from massive import RESTClient
from datetime import datetime, timedelta, timezone
from backend.data.schemas import Bar, BarSeries
import os
from dotenv import load_dotenv

load_dotenv()

# US Eastern timezone offset helpers
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    ET = pytz.timezone("America/New_York")


VALID_TIMEFRAMES = {
    "15min": (15, "minute"),
    "1h":    (1, "hour"),
    "1d":    (1, "day"),
}


def get_client() -> RESTClient:
    """Create a Massive.com REST client using your API key."""
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise ValueError("MASSIVE_API_KEY not found in .env file")
    return RESTClient(api_key=api_key)


def fetch_bars(
    symbol: str,
    timeframe: str = "1d",
    days_back: int = 30,
) -> BarSeries:
    """
    Fetch OHLCV bars from Massive.com for a given symbol.
    
    Only 3 timeframes are supported: "15min", "1h", "1d"
    
    Timestamps are converted to US/Eastern for proper market-hour logic.
    """
    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(
            f"Invalid timeframe: '{timeframe}'. "
            f"AlphaBean supports: {list(VALID_TIMEFRAMES.keys())}"
        )

    client = get_client()
    multiplier, timespan = VALID_TIMEFRAMES[timeframe]

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
        # Massive returns Unix milliseconds in UTC — convert to ET
        utc_dt = datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc)
        et_dt = utc_dt.astimezone(ET)
        # Store as naive datetime in ET (our patterns assume ET)
        et_naive = et_dt.replace(tzinfo=None)

        bars.append(Bar(
            symbol=symbol.upper(),
            timestamp=et_naive,
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


def fetch_news(symbol: str, limit: int = 200) -> list[dict]:
    """
    Fetch news headlines from Massive.com for catalyst confirmation.
    Returns list of dicts with title, published, source, url, tickers.
    """
    client = get_client()
    articles = []
    try:
        for article in client.list_ticker_news(symbol.upper(), limit=limit):
            articles.append({
                "title": article.title,
                "published": getattr(article, 'published_utc', ''),
                "source": getattr(article.publisher, 'name', 'Unknown') if hasattr(article, 'publisher') else 'Unknown',
                "url": getattr(article, 'article_url', ''),
                "tickers": getattr(article, 'tickers', []),
                "description": getattr(article, 'description', ''),
            })
    except Exception as e:
        print(f"[WARN] News fetch failed for {symbol}: {e}")
    return articles


def fetch_snapshot(symbol: str) -> dict:
    """Fetch current snapshot for a symbol (last price, today's change, volume)."""
    client = get_client()
    snapshot = client.get_snapshot_ticker("stocks", symbol.upper())
    return {
        "symbol": symbol.upper(),
        "last_price": snapshot.ticker.last_trade.price,
        "today_change_pct": snapshot.ticker.todays_change_percent,
        "volume": snapshot.ticker.day.volume,
        "prev_close": snapshot.ticker.prev_day.close,
    }