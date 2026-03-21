"""
main.py — AlphaBean API Server v3.3

Start: uvicorn backend.main:app --reload --port 8000
"""
import json
import time
from collections import defaultdict
from datetime import datetime, time as dt_time
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.scanner.engine import scan_symbol, scan_multiple
from backend.patterns.registry import get_all_pattern_names, PATTERN_META
from backend.data.massive_client import fetch_chart_bars, fetch_bars
from backend.regime.detector import detect_regime
from backend.strategies.evaluator import StrategyEvaluator
from backend.news.pipeline import (
    fetch_news, fetch_market_news, format_headlines_for_llm, fetch_news_batch,
)
from backend.ai.ollama_agent import evaluate_setups_batch, check_ollama_status
from backend.ai.inplay_detector import get_in_play, refresh_in_play
from backend.features.correlation import compute_correlation_for_symbol

import numpy as np

app = FastAPI(title="AlphaBean API", version="3.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_evaluator = StrategyEvaluator()
_evaluator.load()

BACKTEST_RESULTS = Path("cache/backtest_results.json")


def _is_market_hours() -> bool:
    """Check if current time is within US market hours (9:30-16:00 ET)."""
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:  # Weekend
        return False
    return dt_time(9, 30) <= now.time() <= dt_time(16, 0)


# ═══════════════════════════════════════════════════════════
# CORE
# ═══════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "ok", "app": "AlphaBean", "version": "3.3.0",
        "patterns": len(get_all_pattern_names()),
        "strategies_tracked": _evaluator.stats_summary()["strategies_tracked"],
        "ollama": check_ollama_status(),
        "market_open": _is_market_hours(),
    }


@app.get("/api/patterns")
async def list_patterns():
    patterns = []
    for name in get_all_pattern_names():
        meta = PATTERN_META.get(name, {})
        patterns.append({
            "name": name,
            "category": meta.get("cat", "").value if hasattr(meta.get("cat", ""), "value") else "",
            "strategy_type": meta.get("type", ""),
            "win_rate": meta.get("wr", 0),
        })
    return {"count": len(patterns), "patterns": patterns}


# ═══════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════

@app.get("/api/scan")
async def scan(
    symbol: str = Query(...),
    mode: str = Query("today"),
    ai: bool = Query(True),
):
    if mode not in ("today", "active"):
        return {"error": "mode must be 'today' or 'active'"}

    market_open = _is_market_hours()

    setups = scan_symbol(symbol.upper(), mode=mode, evaluator=_evaluator)

    # Filter to market hours setups only
    if mode == "today":
        setups = _filter_market_hours_setups(setups)

    if ai and setups:
        try:
            news = fetch_news(symbol.upper(), max_items=15)
            headlines = format_headlines_for_llm(news)
            regime = _get_regime_str()
            setups = evaluate_setups_batch(setups, {symbol.upper(): headlines}, regime, top_n=5)
        except Exception as e:
            print(f"  AI eval error: {e}")

    # Add correlation score
    try:
        corr = compute_correlation_for_symbol(symbol.upper())
        for s in setups:
            s["spy_correlation"] = corr.to_dict()
    except Exception:
        pass

    return {
        "symbol": symbol.upper(), "mode": mode,
        "count": len(setups), "setups": setups,
        "market_open": market_open,
    }


@app.get("/api/scan-multiple")
async def scan_multi(
    symbols: str = Query(...),
    mode: str = Query("today"),
    ai: bool = Query(True),
):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    setups = scan_multiple(symbol_list, mode=mode, evaluator=_evaluator)

    if mode == "today":
        setups = _filter_market_hours_setups(setups)

    if ai and setups:
        try:
            news_batch = fetch_news_batch(symbol_list)
            news_str = {sym: format_headlines_for_llm(items) for sym, items in news_batch.items()}
            regime = _get_regime_str()
            setups = evaluate_setups_batch(setups, news_str, regime, top_n=5)
        except Exception as e:
            print(f"  AI eval error: {e}")

    return {"symbols": symbol_list, "mode": mode, "count": len(setups), "setups": setups}


# ═══════════════════════════════════════════════════════════
# IN-PLAY & TOP OPPORTUNITIES
# ═══════════════════════════════════════════════════════════

@app.get("/api/in-play")
async def in_play():
    result = get_in_play()
    return result.to_dict()


@app.get("/api/in-play/refresh")
async def in_play_refresh():
    result = refresh_in_play()
    return result.to_dict()


@app.get("/api/top-opportunities")
async def top_opportunities(
    ai: bool = Query(True),
    per_symbol: int = Query(2),
    max_symbols: int = Query(15),
):
    t = time.time()
    market_open = _is_market_hours()

    in_play_result = get_in_play()
    symbols = [s.symbol for s in in_play_result.stocks if s.symbol][:max_symbols]

    if not symbols:
        return {
            "market_summary": in_play_result.market_summary,
            "in_play": in_play_result.to_dict(),
            "setups": [], "count": 0, "market_open": market_open,
        }

    all_setups = scan_multiple(symbols, mode="active", evaluator=_evaluator)

    # Cap per symbol
    by_sym: dict[str, list[dict]] = defaultdict(list)
    for s in all_setups:
        by_sym[s.get("symbol", "")].append(s)

    capped = []
    for sym, ss in by_sym.items():
        ss.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        capped.extend(ss[:per_symbol])
    capped.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

    # Filter market hours
    capped = _filter_market_hours_setups(capped)

    # AI evaluate
    if ai and capped:
        try:
            active_syms = list(set(s.get("symbol", "") for s in capped))
            news_batch = fetch_news_batch(active_syms)
            news_str = {sym: format_headlines_for_llm(items) for sym, items in news_batch.items()}
            regime = _get_regime_str()
            capped = evaluate_setups_batch(capped, news_str, regime, top_n=per_symbol)
        except Exception as e:
            print(f"  AI eval error: {e}")

    # Add correlation scores
    corr_cache: dict[str, dict] = {}
    for s in capped:
        sym = s.get("symbol", "")
        if sym not in corr_cache:
            try:
                corr_cache[sym] = compute_correlation_for_symbol(sym).to_dict()
            except Exception:
                corr_cache[sym] = {}
        s["spy_correlation"] = corr_cache.get(sym, {})

    # Enrich with in-play info
    ip_map = {s.symbol: s.to_dict() for s in in_play_result.stocks}
    for s in capped:
        sym = s.get("symbol", "")
        if sym in ip_map:
            s["in_play_info"] = ip_map[sym]

    elapsed = time.time() - t
    return {
        "market_summary": in_play_result.market_summary,
        "in_play": in_play_result.to_dict(),
        "setups": capped, "count": len(capped),
        "elapsed_seconds": round(elapsed, 1),
        "market_open": market_open,
    }


# ═══════════════════════════════════════════════════════════
# LIVE TRADE TRACKER
# ═══════════════════════════════════════════════════════════

@app.get("/api/track-price")
async def track_price(symbol: str = Query(...)):
    """Get current price for live trade tracking. Called every 5min by frontend."""
    try:
        bars = fetch_bars(symbol.upper(), "5min", days_back=1)
        if not bars.bars:
            return {"error": "No data"}
        last = bars.bars[-1]
        return {
            "symbol": symbol.upper(),
            "price": last.close,
            "high": last.high,
            "low": last.low,
            "volume": last.volume,
            "timestamp": last.timestamp.isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/track-prices")
async def track_prices(symbols: str = Query(...)):
    """Batch price fetch for multiple tracked trades."""
    results = {}
    for sym in symbols.split(","):
        sym = sym.strip().upper()
        if not sym:
            continue
        try:
            bars = fetch_bars(sym, "5min", days_back=1)
            if bars.bars:
                last = bars.bars[-1]
                results[sym] = {
                    "price": last.close, "high": last.high, "low": last.low,
                    "volume": last.volume, "timestamp": last.timestamp.isoformat(),
                }
        except Exception:
            results[sym] = {"error": "fetch failed"}
    return {"prices": results, "market_open": _is_market_hours()}


# ═══════════════════════════════════════════════════════════
# CORRELATION
# ═══════════════════════════════════════════════════════════

@app.get("/api/correlation/{symbol}")
async def get_correlation(symbol: str):
    """SPY correlation score for a symbol."""
    try:
        result = compute_correlation_for_symbol(symbol.upper())
        return result.to_dict()
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# BACKTEST RESULTS (detailed for viewer modal)
# ═══════════════════════════════════════════════════════════

@app.get("/api/backtest/results")
async def backtest_results():
    if not BACKTEST_RESULTS.exists():
        return {"has_results": False, "message": "Run: python run_backtest.py"}
    try:
        data = json.loads(BACKTEST_RESULTS.read_text())
        return {"has_results": True, **data}
    except (json.JSONDecodeError, KeyError):
        return {"has_results": False, "message": "Cache corrupt"}


@app.get("/api/backtest/patterns")
async def backtest_patterns_sorted(sort: str = Query("edge_score")):
    """Patterns sorted by a given field — for the backtest viewer."""
    if not BACKTEST_RESULTS.exists():
        return {"patterns": []}
    try:
        data = json.loads(BACKTEST_RESULTS.read_text())
        patterns = data.get("patterns", {})
        items = [{"name": k, **v} for k, v in patterns.items() if v.get("total_signals", 0) >= 3]
        valid_sorts = ["edge_score", "win_rate", "profit_factor", "expectancy", "total_signals"]
        if sort not in valid_sorts:
            sort = "edge_score"
        items.sort(key=lambda x: x.get(sort, 0), reverse=True)
        return {
            "patterns": items,
            "count": len(items),
            "sort": sort,
            "summary": data.get("summary", {}),
            "config": data.get("config", {}),
        }
    except (json.JSONDecodeError, KeyError):
        return {"patterns": []}


# ═══════════════════════════════════════════════════════════
# NEWS / CHART / REGIME / STRATEGIES
# ═══════════════════════════════════════════════════════════

@app.get("/api/news/{symbol}")
async def get_news(symbol: str):
    items = fetch_news(symbol.upper(), max_items=20)
    return {"symbol": symbol.upper(), "count": len(items),
            "headlines": [item.to_dict() for item in items]}

@app.get("/api/news-market")
async def get_market_news():
    items = fetch_market_news(max_items=30)
    return {"count": len(items), "headlines": [item.to_dict() for item in items]}

@app.get("/api/chart/{symbol}")
async def chart_data(symbol: str, timeframe: str = Query("5min"), days_back: int = Query(5)):
    if timeframe not in ("5min", "15min", "1h", "1d"):
        return {"error": f"Invalid timeframe: {timeframe}"}
    try:
        bars = fetch_chart_bars(symbol.upper(), timeframe, days_back)
        return {"symbol": symbol.upper(), "timeframe": timeframe, "bars": bars}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/regime")
async def get_regime():
    try:
        spy_bars = fetch_bars("SPY", timeframe="1d", days_back=250)
        c = np.array([b.close for b in spy_bars.bars], dtype=np.float64)
        h = np.array([b.high for b in spy_bars.bars], dtype=np.float64)
        l = np.array([b.low for b in spy_bars.bars], dtype=np.float64)
        regime = detect_regime(c, h, l, is_spy=True)
        return regime.to_dict()
    except Exception as e:
        return {"error": str(e), "regime": "unknown"}

@app.get("/api/hot-strategies")
async def hot_strategies(top_n: int = Query(5)):
    hot = _evaluator.get_hot_strategies(top_n=top_n)
    return {
        "count": len(hot),
        "strategies": [m.to_dict() for m in hot],
        "hot_types": _evaluator.get_hot_strategy_types(top_n=3),
    }

@app.get("/api/strategy/{pattern_name}")
async def strategy_detail(pattern_name: str):
    return _evaluator.get_pattern_summary(pattern_name)

@app.get("/api/ollama/status")
async def ollama_status():
    return check_ollama_status()

@app.post("/api/reload-evaluator")
async def reload_evaluator():
    _evaluator.load()
    return {"status": "reloaded", **_evaluator.stats_summary()}


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _get_regime_str() -> str:
    try:
        spy_bars = fetch_bars("SPY", timeframe="1d", days_back=60)
        c = np.array([b.close for b in spy_bars.bars], dtype=np.float64)
        h = np.array([b.high for b in spy_bars.bars], dtype=np.float64)
        l = np.array([b.low for b in spy_bars.bars], dtype=np.float64)
        regime = detect_regime(c, h, l, is_spy=True)
        return regime.regime.value
    except Exception:
        return "unknown"


def _filter_market_hours_setups(setups: list[dict]) -> list[dict]:
    """Filter setups to only those detected during market hours (9:30-16:00 ET)."""
    filtered = []
    for s in setups:
        try:
            det = s.get("detected_at", "")
            if not det:
                filtered.append(s)
                continue
            dt = datetime.fromisoformat(det)
            h, m = dt.hour, dt.minute
            # 9:30 to 16:00
            if (h > 9 or (h == 9 and m >= 30)) and h < 16:
                filtered.append(s)
            elif h == 16 and m == 0:
                filtered.append(s)
        except (ValueError, TypeError):
            filtered.append(s)  # Can't parse? Include it
    return filtered