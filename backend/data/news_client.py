"""
backend/data/news_client.py — Polygon/Massive News Client

Fetches headlines with built-in sentiment analysis from Polygon's news API.
Much richer than Finnhub — includes per-ticker sentiment reasoning,
descriptions, keywords, and publisher info.

Usage:
    from backend.data.news_client import fetch_polygon_news, format_news_context

    items = fetch_polygon_news("AAPL", limit=10)
    context_str = format_news_context(items)
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from backend.data.massive_client import get_client

CACHE_DIR = Path("cache/news_polygon")
CACHE_TTL = 600  # 10 minutes


@dataclass
class PolygonInsight:
    """Per-ticker sentiment from Polygon's AI analysis."""
    ticker: str
    sentiment: str          # "positive", "negative", "neutral"
    reasoning: str          # Why this sentiment was assigned


@dataclass
class PolygonNewsItem:
    """A single news article from Polygon."""
    title: str
    description: str
    published_utc: str
    source: str             # Publisher name
    article_url: str
    tickers: list[str]      # All tickers mentioned
    keywords: list[str]
    insights: list[PolygonInsight]   # Per-ticker sentiment
    author: str = ""

    def get_sentiment_for(self, symbol: str) -> Optional[PolygonInsight]:
        """Get the sentiment insight for a specific ticker."""
        for ins in self.insights:
            if ins.ticker.upper() == symbol.upper():
                return ins
        return None

    def to_dict(self) -> dict:
        return {
            "title": self.title, "description": self.description,
            "published_utc": self.published_utc, "source": self.source,
            "article_url": self.article_url, "tickers": self.tickers,
            "keywords": self.keywords, "author": self.author,
            "insights": [{"ticker": i.ticker, "sentiment": i.sentiment,
                          "reasoning": i.reasoning} for i in self.insights],
        }


def fetch_polygon_news(symbol: str, limit: int = 10) -> list[PolygonNewsItem]:
    """Fetch recent news for a ticker from Polygon.

    Returns up to `limit` articles, newest first.
    Cached for 10 minutes per symbol.
    """
    cached = _load_cache(symbol)
    if cached is not None:
        return cached[:limit]

    try:
        client = get_client()
        items = []
        for i, article in enumerate(client.list_ticker_news(ticker=symbol.upper(), limit=limit)):
            if i >= limit:
                break

            # Parse insights
            insights = []
            raw_insights = getattr(article, 'insights', None) or []
            for ins in raw_insights:
                insights.append(PolygonInsight(
                    ticker=getattr(ins, 'ticker', ''),
                    sentiment=getattr(ins, 'sentiment', 'neutral'),
                    reasoning=getattr(ins, 'sentiment_reasoning', ''),
                ))

            # Parse publisher
            pub = getattr(article, 'publisher', None)
            source_name = getattr(pub, 'name', 'Unknown') if pub else 'Unknown'

            items.append(PolygonNewsItem(
                title=getattr(article, 'title', '') or '',
                description=getattr(article, 'description', '') or '',
                published_utc=getattr(article, 'published_utc', '') or '',
                source=source_name,
                article_url=getattr(article, 'article_url', '') or '',
                tickers=getattr(article, 'tickers', []) or [],
                keywords=getattr(article, 'keywords', []) or [],
                insights=insights,
                author=getattr(article, 'author', '') or '',
            ))
        
        if symbol:
            items = [item for item in items if symbol.upper() in item.tickers[:3]]

        _save_cache(symbol, items)
        return items

    except Exception as e:
        print(f"  [Polygon News] Error for {symbol}: {e}")
        return []


def fetch_market_news(limit: int = 15) -> list[PolygonNewsItem]:
    """Fetch general market news using the existing Google News pipeline
    (more relevant than Polygon's unfiltered feed).
    """
    try:
        from backend.news.pipeline import fetch_market_news as _gn_market
        gn_items = _gn_market(max_items=limit)
        # Convert to PolygonNewsItem format for compatibility
        return [PolygonNewsItem(
            title=item.headline,
            description="",
            published_utc=item.published,
            source=item.source,
            article_url=item.url,
            tickers=[],
            keywords=[],
            insights=[],
        ) for item in gn_items]
    except Exception:
        return []


def format_news_context(items: list[PolygonNewsItem], symbol: str = "",
                        max_items: int = 10) -> str:
    """Format news into a context string for the AI agent.

    For each article:
      - Title + source
      - 1-line description (truncated)
      - Per-ticker sentiment if available
    """
    if not items:
        return "No recent news found."

    lines = []
    for item in items[:max_items]:
        # Title and source
        age = _age_string(item.published_utc)
        line = f"- {item.title} ({item.source}, {age})"

        # Add description (truncated to 150 chars)
        if item.description:
            desc = item.description[:150]
            if len(item.description) > 150:
                desc += "..."
            line += f"\n  Summary: {desc}"

        # Per-ticker sentiment if we have a target symbol
        if symbol:
            insight = item.get_sentiment_for(symbol)
            if insight:
                line += f"\n  Sentiment for {symbol}: {insight.sentiment.upper()}"
                if insight.reasoning:
                    reason = insight.reasoning[:120]
                    if len(insight.reasoning) > 120:
                        reason += "..."
                    line += f" — {reason}"

        lines.append(line)

    return "\n".join(lines)


def aggregate_sentiment(items: list[PolygonNewsItem], symbol: str) -> dict:
    """Aggregate sentiment across all articles for a symbol.

    Returns: {"positive": N, "negative": N, "neutral": N, "net": float(-1 to +1)}
    """
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for item in items:
        insight = item.get_sentiment_for(symbol)
        if insight and insight.sentiment in counts:
            counts[insight.sentiment] += 1

    total = sum(counts.values())
    if total == 0:
        return {**counts, "net": 0.0, "total": 0}

    net = (counts["positive"] - counts["negative"]) / total
    return {**counts, "net": round(net, 2), "total": total}


# ── Cache helpers ──

def _age_string(published_utc: str) -> str:
    """Convert timestamp to human-readable age like '2h ago'."""
    try:
        pub = datetime.fromisoformat(published_utc.replace("Z", "+00:00"))
        now = datetime.now(pub.tzinfo)
        delta = now - pub
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)}m ago"
        elif hours < 24:
            return f"{int(hours)}h ago"
        else:
            return f"{int(hours / 24)}d ago"
    except Exception:
        return "recent"


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = key.replace(" ", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}.json"


def _load_cache(key: str) -> Optional[list[PolygonNewsItem]]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if time.time() - data.get("cached_at", 0) > CACHE_TTL:
            return None
        items = []
        for d in data.get("items", []):
            insights = [PolygonInsight(**i) for i in d.get("insights", [])]
            items.append(PolygonNewsItem(
                title=d["title"], description=d["description"],
                published_utc=d["published_utc"], source=d["source"],
                article_url=d["article_url"], tickers=d["tickers"],
                keywords=d["keywords"], insights=insights,
                author=d.get("author", ""),
            ))
        return items
    except Exception:
        return None


def _save_cache(key: str, items: list[PolygonNewsItem]):
    path = _cache_path(key)
    data = {"cached_at": time.time(), "items": [i.to_dict() for i in items]}
    path.write_text(json.dumps(data, indent=2))