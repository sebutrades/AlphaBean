"""
news/pipeline.py — Multi-source news aggregator

Sources:
  1. Finnhub (free tier, 60 calls/min) — real-time company news
  2. Google News RSS — broad coverage, no auth needed

Provides:
  - fetch_news(symbol) → list of headlines for a ticker
  - fetch_market_news() → general market/macro headlines
  - Caching: won't re-fetch within 10 minutes per symbol

Required:
  pip install feedparser requests
  FINNHUB_API_KEY in .env (free at https://finnhub.io)
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import feedparser
import requests
from dotenv import load_dotenv

load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
CACHE_DIR = Path("cache/news")
CACHE_TTL = 600  # 10 minutes


# ==============================================================================
# DATA TYPES
# ==============================================================================

@dataclass
class NewsItem:
    headline: str
    source: str
    url: str
    published: str          # ISO format
    symbol: str             # Ticker this relates to
    provider: str           # "finnhub", "google_news"
    sentiment_hint: str = ""  # Pre-computed hint: "positive", "negative", "neutral"

    def to_dict(self) -> dict:
        return {
            "headline": self.headline, "source": self.source,
            "url": self.url, "published": self.published,
            "symbol": self.symbol, "provider": self.provider,
            "sentiment_hint": self.sentiment_hint,
        }


# ==============================================================================
# FINNHUB
# ==============================================================================

def _fetch_finnhub(symbol: str, days_back: int = 3) -> list[NewsItem]:
    """Fetch company news from Finnhub free tier."""
    if not FINNHUB_KEY:
        return []

    end = datetime.now()
    start = end - timedelta(days=days_back)

    try:
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": symbol.upper(),
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "token": FINNHUB_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []

        articles = resp.json()
        if not isinstance(articles, list):
            return []

        items = []
        for a in articles[:20]:  # Cap at 20 per source
            headline = a.get("headline", "").strip()
            if not headline:
                continue

            pub_ts = a.get("datetime", 0)
            try:
                pub_str = datetime.fromtimestamp(int(pub_ts)).isoformat() if pub_ts and int(pub_ts) > 0 else ""
            except (ValueError, OSError):
                pub_str = ""

            items.append(NewsItem(
                headline=headline,
                source=a.get("source", "Finnhub"),
                url=a.get("url", ""),
                published=pub_str,
                symbol=symbol.upper(),
                provider="finnhub",
                sentiment_hint=_quick_sentiment(headline),
            ))

        return items

    except Exception as e:
        print(f"  [Finnhub] Error for {symbol}: {e}")
        return []


def _fetch_finnhub_general() -> list[NewsItem]:
    """Fetch general market news from Finnhub."""
    if not FINNHUB_KEY:
        return []

    try:
        url = "https://finnhub.io/api/v1/news"
        params = {"category": "general", "token": FINNHUB_KEY}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []

        articles = resp.json()
        if not isinstance(articles, list):
            return []

        items = []
        for a in articles[:25]:
            headline = a.get("headline", "").strip()
            if not headline:
                continue
            pub_ts = a.get("datetime", 0)
            try:
                pub_str = datetime.fromtimestamp(int(pub_ts)).isoformat() if pub_ts and int(pub_ts) > 0 else ""
            except (ValueError, OSError):
                pub_str = ""
            items.append(NewsItem(
                headline=headline, source=a.get("source", "Finnhub"),
                url=a.get("url", ""), published=pub_str,
                symbol="MARKET", provider="finnhub",
                sentiment_hint=_quick_sentiment(headline),
            ))
        return items

    except Exception:
        return []


# ==============================================================================
# GOOGLE NEWS RSS
# ==============================================================================

def _fetch_google_news(query: str, symbol: str = "") -> list[NewsItem]:
    """Fetch news from Google News RSS (no auth needed)."""
    try:
        encoded = quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)

        items = []
        for entry in feed.entries[:15]:
            headline = _clean_html(entry.get("title", ""))
            if not headline:
                continue

            # Extract source from title (Google News format: "Headline - Source")
            source = "Google News"
            if " - " in headline:
                parts = headline.rsplit(" - ", 1)
                if len(parts) == 2:
                    headline = parts[0].strip()
                    source = parts[1].strip()

            pub = entry.get("published", "")
            try:
                pub_dt = datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else None
                pub_str = pub_dt.isoformat() if pub_dt else pub
            except Exception:
                pub_str = pub

            items.append(NewsItem(
                headline=headline, source=source,
                url=entry.get("link", ""), published=pub_str,
                symbol=symbol.upper() or "MARKET", provider="google_news",
                sentiment_hint=_quick_sentiment(headline),
            ))

        return items

    except Exception as e:
        print(f"  [Google News] Error for '{query}': {e}")
        return []


# ==============================================================================
# PUBLIC API
# ==============================================================================

def fetch_news(symbol: str, max_items: int = 30) -> list[NewsItem]:
    """
    Fetch news for a specific ticker from all sources.
    Cached for 10 minutes.
    """
    cached = _load_cache(f"ticker_{symbol}")
    if cached is not None:
        return cached

    all_items = []

    # Finnhub company news
    all_items.extend(_fetch_finnhub(symbol))

    # Google News: search for "SYMBOL stock"
    all_items.extend(_fetch_google_news(f"{symbol} stock news", symbol))

    # Deduplicate by headline similarity
    unique = _deduplicate(all_items)

    # Sort by published date (newest first)
    unique.sort(key=lambda x: x.published, reverse=True)

    # Cap
    unique = unique[:max_items]

    _save_cache(f"ticker_{symbol}", unique)
    return unique


def fetch_market_news(max_items: int = 40) -> list[NewsItem]:
    """
    Fetch general market/macro news from all sources.
    Cached for 10 minutes.
    """
    cached = _load_cache("market_general")
    if cached is not None:
        return cached

    all_items = []

    # Finnhub general news
    all_items.extend(_fetch_finnhub_general())

    # Google News: market terms
    all_items.extend(_fetch_google_news("stock market today"))
    all_items.extend(_fetch_google_news("Wall Street premarket movers"))
    all_items.extend(_fetch_google_news("Fed interest rate economy"))

    unique = _deduplicate(all_items)
    unique.sort(key=lambda x: x.published, reverse=True)
    unique = unique[:max_items]

    _save_cache("market_general", unique)
    return unique


def fetch_news_batch(symbols: list[str]) -> dict[str, list[NewsItem]]:
    """Fetch news for multiple symbols."""
    result = {}
    for sym in symbols:
        result[sym] = fetch_news(sym, max_items=10)
        time.sleep(0.2)  # Rate limit courtesy
    return result


def format_headlines_for_llm(items: list[NewsItem], max_items: int = 15) -> str:
    """Format news items as a concise string for LLM context."""
    if not items:
        return "No recent news found."

    lines = []
    for item in items[:max_items]:
        hint = f" [{item.sentiment_hint}]" if item.sentiment_hint else ""
        source = f" ({item.source})" if item.source else ""
        lines.append(f"• {item.headline}{source}{hint}")

    return "\n".join(lines)


# ==============================================================================
# HELPERS
# ==============================================================================

def _quick_sentiment(headline: str) -> str:
    """Fast rule-based sentiment hint (not AI — just keywords)."""
    h = headline.lower()
    pos = ["surge", "soar", "rally", "jump", "gain", "beat", "upgrade",
           "bullish", "record", "breakout", "boom", "high", "strong", "growth",
           "buy", "outperform", "raise", "positive"]
    neg = ["crash", "plunge", "drop", "fall", "miss", "downgrade", "sell",
           "bearish", "slump", "decline", "cut", "weak", "loss", "layoff",
           "warning", "risk", "fear", "concern", "negative", "lawsuit"]

    pos_count = sum(1 for w in pos if w in h)
    neg_count = sum(1 for w in neg if w in h)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    return "neutral"


def _clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    clean = re.sub(r"<[^>]+>", "", text)
    return unescape(clean).strip()


def _deduplicate(items: list[NewsItem]) -> list[NewsItem]:
    """Remove duplicate headlines (fuzzy: first 60 chars lowercase)."""
    seen = set()
    unique = []
    for item in items:
        key = item.headline[:60].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", key)
    return CACHE_DIR / f"{safe_key}.json"


def _load_cache(key: str) -> Optional[list[NewsItem]]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > CACHE_TTL:
            return None
        return [NewsItem(**item) for item in data.get("items", [])]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _save_cache(key: str, items: list[NewsItem]):
    path = _cache_path(key)
    data = {
        "cached_at": time.time(),
        "items": [item.to_dict() for item in items],
    }
    path.write_text(json.dumps(data, indent=2))