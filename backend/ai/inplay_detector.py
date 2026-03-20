"""
ai/inplay_detector.py — In-Play Stocks from Yahoo Finance Trending

Scrapes https://finance.yahoo.com/markets/stocks/trending/
to get the actual trending tickers — no AI needed, just real data.

Multiple extraction methods (tries in order):
  1. Yahoo trending API endpoint (fastest)
  2. HTML scrape with embedded JSON extraction
  3. yfinance library
  4. Hardcoded high-volume fallback

Cached for 30 minutes.

Required: pip install requests beautifulsoup4 yfinance
"""
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path("cache/in_play.json")
CACHE_TTL = 1800  # 30 minutes

YAHOO_TRENDING_URL = "https://finance.yahoo.com/markets/stocks/trending/"
YAHOO_API_TRENDING = "https://query1.finance.yahoo.com/v1/finance/trending/US"
YAHOO_API_MOVERS = "https://query2.finance.yahoo.com/v6/finance/trending/US"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ==============================================================================
# DATA TYPES
# ==============================================================================

@dataclass
class InPlayStock:
    symbol: str
    reason: str
    catalyst: str
    expected_direction: str  # bullish, bearish, volatile, neutral
    priority: int

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "reason": self.reason,
            "catalyst": self.catalyst, "expected_direction": self.expected_direction,
            "priority": self.priority,
        }


@dataclass
class InPlayResult:
    stocks: list[InPlayStock]
    market_summary: str
    generated_at: str
    cached: bool = False
    source: str = "yahoo_trending"

    def to_dict(self) -> dict:
        return {
            "stocks": [s.to_dict() for s in self.stocks],
            "market_summary": self.market_summary,
            "generated_at": self.generated_at,
            "cached": self.cached,
            "count": len(self.stocks),
            "source": self.source,
        }


# ==============================================================================
# PUBLIC API
# ==============================================================================

def get_in_play() -> InPlayResult:
    """Get today's trending stocks from Yahoo Finance. Cached 30 min."""
    cached = _load_cache()
    if cached is not None:
        cached.cached = True
        return cached

    result = _fetch_trending()
    _save_cache(result)
    return result


def refresh_in_play() -> InPlayResult:
    """Force refresh, bypass cache."""
    _clear_cache()
    return get_in_play()


# ==============================================================================
# TRENDING FETCH — tries multiple methods
# ==============================================================================

def _fetch_trending() -> InPlayResult:
    """Try multiple methods to get Yahoo trending tickers."""

    # Method 1: Yahoo trending API
    tickers = _try_yahoo_api()
    if tickers:
        print(f"  [In-Play] Got {len(tickers)} tickers from Yahoo API")
        return _build_result(tickers, "yahoo_api")

    # Method 2: Scrape Yahoo trending page HTML
    tickers = _try_yahoo_html_scrape()
    if tickers:
        print(f"  [In-Play] Got {len(tickers)} tickers from Yahoo HTML")
        return _build_result(tickers, "yahoo_html")

    # Method 3: yfinance library
    tickers = _try_yfinance()
    if tickers:
        print(f"  [In-Play] Got {len(tickers)} tickers from yfinance")
        return _build_result(tickers, "yfinance")

    # Method 4: Fallback
    print("  [In-Play] All methods failed, using fallback")
    return _fallback_result()


def _try_yahoo_api() -> list[str]:
    """Try Yahoo Finance trending API endpoint."""
    try:
        for url in [YAHOO_API_TRENDING, YAHOO_API_MOVERS]:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # v1 format
                results = data.get("finance", {}).get("result", [])
                if results:
                    quotes = results[0].get("quotes", [])
                    tickers = [q.get("symbol", "") for q in quotes if q.get("symbol")]
                    if tickers:
                        return _filter_tickers(tickers)
                # v6 format
                results = data.get("finance", {}).get("result", [])
                if results:
                    symbols = results[0].get("symbols", [])
                    if symbols:
                        return _filter_tickers(symbols)
    except Exception as e:
        print(f"  [In-Play] Yahoo API error: {e}")
    return []


def _try_yahoo_html_scrape() -> list[str]:
    """Scrape Yahoo trending page HTML for embedded ticker data."""
    try:
        resp = requests.get(
            YAHOO_TRENDING_URL,
            headers=HEADERS,
            timeout=15,
            params={"start": "0", "count": "50"},
        )
        if resp.status_code != 200:
            print(f"  [In-Play] Yahoo HTML returned {resp.status_code}")
            return []

        html = resp.text

        # Method A: Look for ticker symbols in data attributes
        # Yahoo uses data-symbol or data-testid patterns
        symbols = re.findall(r'data-symbol="([A-Z]{1,5})"', html)
        if symbols:
            return _filter_tickers(list(dict.fromkeys(symbols)))  # Dedupe, preserve order

        # Method B: Look for JSON embedded in <script> tags
        # Yahoo embeds page data in script type="application/json"
        json_blocks = re.findall(r'<script[^>]*>(\{.*?"symbols?".*?\})</script>', html, re.DOTALL)
        for block in json_blocks:
            try:
                data = json.loads(block)
                syms = _extract_symbols_from_json(data)
                if syms:
                    return _filter_tickers(syms)
            except json.JSONDecodeError:
                continue

        # Method C: Look for fin-streamer tags (Yahoo's custom elements)
        symbols = re.findall(r'data-field="regularMarketPrice"[^>]*data-symbol="([A-Z]{1,5})"', html)
        if symbols:
            return _filter_tickers(list(dict.fromkeys(symbols)))

        # Method D: Look for href="/quote/TICKER" patterns
        symbols = re.findall(r'href="/quote/([A-Z]{1,5})(?:\?|")', html)
        if symbols:
            return _filter_tickers(list(dict.fromkeys(symbols)))

        # Method E: JSON-LD or structured data
        ld_blocks = re.findall(r'"symbol"\s*:\s*"([A-Z]{1,5})"', html)
        if ld_blocks:
            return _filter_tickers(list(dict.fromkeys(ld_blocks)))

        print("  [In-Play] Could not extract tickers from Yahoo HTML")
        return []

    except Exception as e:
        print(f"  [In-Play] Yahoo HTML scrape error: {e}")
        return []


def _try_yfinance() -> list[str]:
    """Try using yfinance library to get trending tickers."""
    try:
        import yfinance as yf

        # yfinance doesn't have a direct trending endpoint,
        # but we can get most active / gainers as a proxy
        # Try the screener approach
        try:
            # This may work depending on yfinance version
            screen = yf.Screener()
            screen.set_predefined_body("most_actives")
            data = screen.response
            if data:
                quotes = data.get("quotes", [])
                tickers = [q.get("symbol", "") for q in quotes if q.get("symbol")]
                if tickers:
                    return _filter_tickers(tickers)
        except Exception:
            pass

        return []

    except ImportError:
        return []
    except Exception as e:
        print(f"  [In-Play] yfinance error: {e}")
        return []


# ==============================================================================
# HELPERS
# ==============================================================================

def _filter_tickers(tickers: list[str]) -> list[str]:
    """Filter to valid US equity tickers only."""
    filtered = []
    seen = set()
    for t in tickers:
        t = t.strip().upper()
        if not t or t in seen:
            continue
        # Skip: too long, has dots/dashes (preferred shares, warrants)
        if len(t) > 5 or not t.isalpha():
            continue
        # Skip common non-equities
        skip = {"USD", "EUR", "GBP", "JPY", "BTC", "ETH", "GC", "CL", "SI", "ES", "NQ"}
        if t in skip:
            continue
        seen.add(t)
        filtered.append(t)
    return filtered[:50]  # Cap at 50


def _extract_symbols_from_json(data, depth=0) -> list[str]:
    """Recursively extract symbol strings from nested JSON."""
    if depth > 5:
        return []
    symbols = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k in ("symbol", "ticker") and isinstance(v, str) and 1 <= len(v) <= 5 and v.isalpha():
                symbols.append(v.upper())
            else:
                symbols.extend(_extract_symbols_from_json(v, depth + 1))
    elif isinstance(data, list):
        for item in data[:100]:  # Cap to avoid huge lists
            symbols.extend(_extract_symbols_from_json(item, depth + 1))
    return symbols


def _build_result(tickers: list[str], source: str) -> InPlayResult:
    """Build InPlayResult from a list of tickers."""
    stocks = []
    for i, sym in enumerate(tickers[:50]):
        stocks.append(InPlayStock(
            symbol=sym,
            reason="Trending on Yahoo Finance",
            catalyst="trending",
            expected_direction="volatile",
            priority=i + 1,
        ))

    return InPlayResult(
        stocks=stocks,
        market_summary=f"Top {len(stocks)} trending stocks from Yahoo Finance",
        generated_at=datetime.now().isoformat(),
        source=source,
    )


# Fallback: popular high-volume names that are almost always active
FALLBACK_TICKERS = [
    "NVDA", "TSLA", "AAPL", "AMD", "PLTR", "AMZN", "META", "MSFT",
    "GOOGL", "COIN", "SOFI", "MARA", "NIO", "RIVN", "SMCI",
    "ARM", "AVGO", "MU", "INTC", "NFLX", "CRM", "UBER",
    "BA", "JPM", "GS", "SPY", "QQQ", "MSTR", "RIOT",
    "DKNG", "SQ", "PYPL", "SHOP", "SNOW", "CRWD",
]

def _fallback_result() -> InPlayResult:
    stocks = [
        InPlayStock(symbol=sym, reason="High-volume watchlist",
                    catalyst="volume", expected_direction="volatile", priority=i+1)
        for i, sym in enumerate(FALLBACK_TICKERS)
    ]
    return InPlayResult(
        stocks=stocks,
        market_summary="Using high-volume watchlist (Yahoo fetch unavailable)",
        generated_at=datetime.now().isoformat(),
        source="fallback",
    )


# ==============================================================================
# CACHE
# ==============================================================================

def _load_cache() -> Optional[InPlayResult]:
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text())
        if time.time() - data.get("cached_at", 0) > CACHE_TTL:
            return None
        stocks = [InPlayStock(**s) for s in data.get("stocks", [])]
        return InPlayResult(
            stocks=stocks, market_summary=data.get("market_summary", ""),
            generated_at=data.get("generated_at", ""),
            source=data.get("source", "cache"),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _save_cache(result: InPlayResult):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "cached_at": time.time(),
        "stocks": [s.to_dict() for s in result.stocks],
        "market_summary": result.market_summary,
        "generated_at": result.generated_at,
        "source": result.source,
    }
    CACHE_PATH.write_text(json.dumps(data, indent=2))


def _clear_cache():
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()