"""
ai/inplay_detector.py — In-Play Stocks - Trending Tickers

Scrapes https://finance.yahoo.com/markets/stocks/trending/
Multiple extraction methods with robust fallbacks.

Cached for 30 minutes.
"""
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

CACHE_PATH = Path("cache/in_play.json")
CACHE_TTL = 1800  # 30 minutes

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}


@dataclass
class InPlayStock:
    symbol: str
    reason: str
    catalyst: str
    expected_direction: str
    priority: int
    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "reason": self.reason, "catalyst": self.catalyst,
                "expected_direction": self.expected_direction, "priority": self.priority}


@dataclass
class InPlayResult:
    stocks: list[InPlayStock]
    market_summary: str
    generated_at: str
    cached: bool = False
    source: str = "yahoo_trending"
    def to_dict(self) -> dict:
        return {"stocks": [s.to_dict() for s in self.stocks], "market_summary": self.market_summary,
                "generated_at": self.generated_at, "cached": self.cached,
                "count": len(self.stocks), "source": self.source}


def get_in_play() -> InPlayResult:
    cached = _load_cache()
    if cached is not None:
        cached.cached = True
        return cached
    result = _fetch_trending()
    _save_cache(result)
    return result


def refresh_in_play() -> InPlayResult:
    _clear_cache()
    return get_in_play()


def _fetch_trending() -> InPlayResult:
    """Try multiple methods to get Yahoo trending tickers."""

    # Method 1: Direct page scrape
    tickers = _scrape_yahoo_page()
    if tickers and len(tickers) >= 10:
        print(f"  [In-Play] Got {len(tickers)} from page scrape")
        return _build_result(tickers, "yahoo_scrape")

    # Method 2: Yahoo Finance API endpoints
    for url in [
        "https://query1.finance.yahoo.com/v1/finance/trending/US",
        "https://query2.finance.yahoo.com/v1/finance/trending/US",
    ]:
        tickers = _try_api(url)
        if tickers and len(tickers) >= 5:
            print(f"  [In-Play] Got {len(tickers)} from API")
            return _build_result(tickers, "yahoo_api")

    # Method 3: yfinance screener
    tickers = _try_yfinance_screener()
    if tickers and len(tickers) >= 5:
        print(f"  [In-Play] Got {len(tickers)} from yfinance")
        return _build_result(tickers, "yfinance")

    # Method 4: Fallback with high-volume names
    print("  [In-Play] All methods failed, using fallback watchlist")
    return _fallback_result()


def _scrape_yahoo_page() -> list[str]:
    """Scrape Yahoo Finance trending page for ticker symbols."""
    try:
        # Fetch the page
        resp = requests.get(
            "https://finance.yahoo.com/markets/stocks/trending/",
            headers=HEADERS,
            params={"start": "0", "count": "50"},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"  [In-Play] Yahoo page returned {resp.status_code}")
            return []

        html = resp.text
        tickers = []

        # Strategy 1: data-symbol attributes (most reliable)
        found = re.findall(r'data-symbol="([A-Z]{1,5})"', html)
        if found:
            tickers = list(dict.fromkeys(found))  # Dedupe, preserve order

        # Strategy 2: /quote/TICKER links
        if len(tickers) < 10:
            found2 = re.findall(r'/quote/([A-Z]{1,5})(?:[/?"\s])', html)
            for t in found2:
                if t not in tickers:
                    tickers.append(t)

        # Strategy 3: fin-streamer elements
        if len(tickers) < 10:
            found3 = re.findall(r'symbol="([A-Z]{1,5})"', html)
            for t in found3:
                if t not in tickers:
                    tickers.append(t)

        # Strategy 4: Look in embedded JSON/script data
        if len(tickers) < 10:
            found4 = re.findall(r'"symbol"\s*:\s*"([A-Z]{1,5})"', html)
            for t in found4:
                if t not in tickers:
                    tickers.append(t)

        return _filter_tickers(tickers)

    except Exception as e:
        print(f"  [In-Play] Yahoo scrape error: {e}")
        return []


def _try_api(url: str) -> list[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = data.get("finance", {}).get("result", [])
        if not results:
            return []
        quotes = results[0].get("quotes", [])
        tickers = [q.get("symbol", "") for q in quotes if q.get("symbol")]
        if not tickers:
            # Try alternate format
            tickers = results[0].get("symbols", [])
        return _filter_tickers(tickers)
    except Exception:
        return []


def _try_yfinance_screener() -> list[str]:
    try:
        import yfinance as yf
        # Get most active stocks
        screen = yf.Screener()
        screen.set_predefined_body("most_actives")
        data = screen.response
        if data and "quotes" in data:
            return _filter_tickers([q["symbol"] for q in data["quotes"] if "symbol" in q])
    except Exception:
        pass
    return []


def _filter_tickers(tickers: list[str]) -> list[str]:
    seen = set()
    filtered = []
    skip = {"USD", "EUR", "GBP", "JPY", "BTC", "ETH", "GC", "CL", "SI", "ES", "NQ", "YM", "RTY"}
    for t in tickers:
        t = t.strip().upper()
        if not t or t in seen or len(t) > 5 or not t.isalpha() or t in skip:
            continue
        seen.add(t)
        filtered.append(t)
    return filtered[:50]


def _build_result(tickers: list[str], source: str) -> InPlayResult:
    stocks = [
        InPlayStock(symbol=sym, reason="Trending today",
                    catalyst="trending", expected_direction="volatile", priority=i+1)
        for i, sym in enumerate(tickers[:50])
    ]
    return InPlayResult(
        stocks=stocks,
        market_summary=f"Top {len(stocks)} trending stocks",
        generated_at=datetime.now().isoformat(), source=source,
    )


# High-volume names that are nearly always active
FALLBACK = [
    "NVDA","TSLA","AAPL","AMD","PLTR","AMZN","META","MSFT","GOOGL","COIN",
    "SOFI","MARA","NIO","RIVN","SMCI","ARM","AVGO","MU","INTC","NFLX",
    "CRM","UBER","BA","JPM","GS","MSTR","RIOT","DKNG","SQ","PYPL",
    "SHOP","SNOW","CRWD","PANW","NET","DDOG","RBLX","HOOD","AFRM","DASH",
    "LLY","UNH","XOM","CVX","CAT","GE","SPY","QQQ","IWM","DIA",
]

def _fallback_result() -> InPlayResult:
    stocks = [InPlayStock(symbol=s, reason="High volume", catalyst="volume",
                          expected_direction="volatile", priority=i+1)
              for i, s in enumerate(FALLBACK)]
    return InPlayResult(stocks=stocks, market_summary="Using high-volume watchlist",
                        generated_at=datetime.now().isoformat(), source="fallback")


def _load_cache() -> Optional[InPlayResult]:
    if not CACHE_PATH.exists(): return None
    try:
        data = json.loads(CACHE_PATH.read_text())
        if time.time() - data.get("cached_at", 0) > CACHE_TTL: return None
        return InPlayResult(stocks=[InPlayStock(**s) for s in data.get("stocks", [])],
                            market_summary=data.get("market_summary", ""),
                            generated_at=data.get("generated_at", ""),
                            source=data.get("source", "cache"))
    except Exception: return None

def _save_cache(result: InPlayResult):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({
        "cached_at": time.time(), "stocks": [s.to_dict() for s in result.stocks],
        "market_summary": result.market_summary, "generated_at": result.generated_at,
        "source": result.source,
    }, indent=2))

def _clear_cache():
    if CACHE_PATH.exists(): CACHE_PATH.unlink()