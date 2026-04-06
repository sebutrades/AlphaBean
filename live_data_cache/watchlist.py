"""
live_data_cache/watchlist.py — Symbol watchlist management

Manages which symbols to track. Two tiers:
  Hot list  — current in-play 50 stocks (updated every 30 min)
              + symbols in the active trade tracker
              These get 5-min bar updates every cycle.

  Full list — top 500 symbols by volume
              These get 5-min bar updates every 30 min,
              15-min and 1-h updates every 60 min.
"""
import json
from pathlib import Path

SYMBOLS_CACHE  = Path("cache/top_symbols.json")
INPLAY_CACHE   = Path("cache/in_play.json")
TRADES_CACHE   = Path("cache/active_trades.json")

_FALLBACK = [
    "AAPL","NVDA","TSLA","MSFT","META","AMZN","GOOGL","AMD","SPY","QQQ",
    "PLTR","COIN","SOFI","NIO","RIVN","F","GM","INTC","MU","SMCI",
    "ARM","AVGO","QCOM","TSM","ASML","NFLX","DIS","ORCL","CRM","NOW",
    "SNOW","UBER","LYFT","ABNB","BKNG","HOOD","MARA","RIOT","CLSK","MSTR",
    "JPM","GS","MS","BAC","C","WFC","XLF","GLD","SLV","USO",
]


def get_top_symbols(limit: int = 500) -> list[str]:
    """Top N symbols by volume from cache."""
    if SYMBOLS_CACHE.exists():
        try:
            return json.loads(SYMBOLS_CACHE.read_text()).get("symbols", [])[:limit]
        except Exception:
            pass
    return _FALLBACK


def get_inplay_symbols() -> list[str]:
    """Current in-play 50 from the in-play detector cache."""
    if INPLAY_CACHE.exists():
        try:
            data = json.loads(INPLAY_CACHE.read_text())
            stocks = data.get("stocks", [])
            return [s["symbol"] for s in stocks if s.get("symbol")][:50]
        except Exception:
            pass
    return []


def get_active_trade_symbols() -> list[str]:
    """Symbols that have open tracked trades."""
    if TRADES_CACHE.exists():
        try:
            data  = json.loads(TRADES_CACHE.read_text())
            trades = data.get("trades", [])
            active_statuses = {"PENDING","ACTIVE","AT_T1","AT_T2","TRAILING"}
            return list({
                t["symbol"] for t in trades
                if t.get("status") in active_statuses
            })
        except Exception:
            pass
    return []


def get_hot_list() -> list[str]:
    """
    In-play + active trades — these get the highest update frequency.
    In-play stocks come first (priority order preserved).
    """
    inplay = get_inplay_symbols()
    active = get_active_trade_symbols()
    seen   = set(inplay)
    result = list(inplay)
    for s in active:
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result


def get_all_tracked() -> list[str]:
    """
    Full tracked universe: hot list first, then rest of top 500.
    """
    hot  = get_hot_list()
    top  = get_top_symbols()
    seen = set(hot)
    result = list(hot)
    for s in top:
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result
