"""
main.py — AlphaBean API Server v3.2

Start: uvicorn backend.main:app --reload --port 8000

All AI is local (Ollama). In-play from Yahoo Finance trending.
"""
import json
import time
from collections import defaultdict
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

import numpy as np

app = FastAPI(title="AlphaBean API", version="3.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_evaluator = StrategyEvaluator()
_evaluator.load()

BACKTEST_RESULTS = Path("cache/backtest_results.json")


# ═══════════════════════════════════════════════════════════
# CORE
# ═══════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    ollama = check_ollama_status()
    return {
        "status": "ok", "app": "AlphaBean", "version": "3.2.0",
        "patterns": len(get_all_pattern_names()),
        "strategies_tracked": _evaluator.stats_summary()["strategies_tracked"],
        "ollama": ollama,
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
# SCANNER (individual — no limit)
# ═══════════════════════════════════════════════════════════

@app.get("/api/scan")
async def scan(
    symbol: str = Query(...),
    mode: str = Query("today"),
    ai: bool = Query(True),
):
    if mode not in ("today", "active"):
        return {"error": "mode must be 'today' or 'active'"}

    print(f"\n{'=' * 55}")
    print(f"  SCAN: {symbol.upper()} | mode={mode} | AI={'on' if ai else 'off'}")
    print(f"{'=' * 55}")

    setups = scan_symbol(symbol.upper(), mode=mode, evaluator=_evaluator)

    if ai and setups:
        try:
            news = fetch_news(symbol.upper(), max_items=15)
            headlines = format_headlines_for_llm(news)
            regime = _get_regime_str()
            setups = evaluate_setups_batch(setups, {symbol.upper(): headlines}, regime, top_n=5)
        except Exception as e:
            print(f"  AI evaluation error: {e}")

    print(f"  RESULT: {len(setups)} setups")
    return {"symbol": symbol.upper(), "mode": mode, "count": len(setups), "setups": setups}


@app.get("/api/scan-multiple")
async def scan_multi(
    symbols: str = Query(...),
    mode: str = Query("today"),
    ai: bool = Query(True),
):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    setups = scan_multiple(symbol_list, mode=mode, evaluator=_evaluator)

    if ai and setups:
        try:
            news_batch = fetch_news_batch(symbol_list)
            news_str = {sym: format_headlines_for_llm(items) for sym, items in news_batch.items()}
            regime = _get_regime_str()
            setups = evaluate_setups_batch(setups, news_str, regime, top_n=5)
        except Exception as e:
            print(f"  AI evaluation error: {e}")

    return {"symbols": symbol_list, "mode": mode, "count": len(setups), "setups": setups}


# ═══════════════════════════════════════════════════════════
# IN-PLAY & TOP OPPORTUNITIES
# ═══════════════════════════════════════════════════════════

@app.get("/api/in-play")
async def in_play():
    """Get trending stocks from Yahoo Finance. Cached 30 min."""
    t = time.time()
    print(f"\n  IN-PLAY: Fetching Yahoo Finance trending...")
    result = get_in_play()
    elapsed = time.time() - t
    print(f"  Got {len(result.stocks)} trending tickers ({elapsed:.1f}s) "
          f"[{'cached' if result.cached else result.source}]")
    return result.to_dict()


@app.get("/api/in-play/refresh")
async def in_play_refresh():
    result = refresh_in_play()
    return result.to_dict()


@app.get("/api/top-opportunities")
async def top_opportunities(
    ai: bool = Query(True),
    per_symbol: int = Query(2, description="Max setups per symbol"),
    max_symbols: int = Query(15, description="Max symbols to scan"),
):
    """
    Front page:
    1. Yahoo Finance trending tickers
    2. Scan each through 47-pattern engine
    3. Keep only top N per symbol (default: 2)
    4. Ollama evaluates the survivors
    5. Return ranked
    """
    t = time.time()
    print(f"\n{'═' * 55}")
    print(f"  TOP OPPORTUNITIES — {per_symbol} per symbol, max {max_symbols} symbols")
    print(f"{'═' * 55}")

    # Step 1: Get trending tickers from Yahoo
    in_play_result = get_in_play()
    symbols = [s.symbol for s in in_play_result.stocks if s.symbol][:max_symbols]

    if not symbols:
        return {
            "market_summary": in_play_result.market_summary,
            "in_play": in_play_result.to_dict(),
            "setups": [], "count": 0,
        }

    print(f"  Trending: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

    # Step 2: Scan all symbols
    all_setups = scan_multiple(symbols, mode="active", evaluator=_evaluator)
    print(f"  Scanner found {len(all_setups)} total setups")

    # Step 3: Keep only top N per symbol by composite_score
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for s in all_setups:
        by_symbol[s.get("symbol", "")].append(s)

    capped_setups = []
    for sym, sym_setups in by_symbol.items():
        sym_setups.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        capped_setups.extend(sym_setups[:per_symbol])

    # Re-sort globally
    capped_setups.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    print(f"  After {per_symbol}-per-symbol cap: {len(capped_setups)} setups")

    # Step 4: AI evaluate
    if ai and capped_setups:
        try:
            # Only fetch news for symbols that have setups
            active_syms = list(set(s.get("symbol", "") for s in capped_setups))
            news_batch = fetch_news_batch(active_syms)
            news_str = {sym: format_headlines_for_llm(items) for sym, items in news_batch.items()}
            regime = _get_regime_str()
            # Evaluate all (they're already capped)
            capped_setups = evaluate_setups_batch(capped_setups, news_str, regime, top_n=per_symbol)
        except Exception as e:
            print(f"  AI evaluation error: {e}")

    elapsed = time.time() - t
    print(f"  Pipeline done: {len(capped_setups)} opportunities ({elapsed:.1f}s)")

    # Enrich with in-play info
    in_play_map = {s.symbol: s.to_dict() for s in in_play_result.stocks}
    for setup in capped_setups:
        sym = setup.get("symbol", "")
        if sym in in_play_map:
            setup["in_play_info"] = in_play_map[sym]

    return {
        "market_summary": in_play_result.market_summary,
        "in_play": in_play_result.to_dict(),
        "setups": capped_setups,
        "count": len(capped_setups),
        "elapsed_seconds": round(elapsed, 1),
    }


# ═══════════════════════════════════════════════════════════
# NEWS
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


# ═══════════════════════════════════════════════════════════
# CHART / REGIME / STRATEGIES
# ═══════════════════════════════════════════════════════════

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
        closes = np.array([b.close for b in spy_bars.bars], dtype=np.float64)
        highs = np.array([b.high for b in spy_bars.bars], dtype=np.float64)
        lows = np.array([b.low for b in spy_bars.bars], dtype=np.float64)
        regime = detect_regime(closes, highs, lows, is_spy=True)
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


@app.get("/api/backtest/results")
async def backtest_results():
    if not BACKTEST_RESULTS.exists():
        return {"has_results": False, "message": "Run: python run_backtest.py"}
    try:
        data = json.loads(BACKTEST_RESULTS.read_text())
        return {"has_results": True, **data}
    except (json.JSONDecodeError, KeyError):
        return {"has_results": False, "message": "Cache corrupt"}


@app.get("/api/ollama/status")
async def ollama_status():
    return check_ollama_status()


@app.post("/api/reload-evaluator")
async def reload_evaluator():
    _evaluator.load()
    return {"status": "reloaded", **_evaluator.stats_summary()}


def _get_regime_str() -> str:
    try:
        spy_bars = fetch_bars("SPY", timeframe="1d", days_back=60)
        closes = np.array([b.close for b in spy_bars.bars], dtype=np.float64)
        highs = np.array([b.high for b in spy_bars.bars], dtype=np.float64)
        lows = np.array([b.low for b in spy_bars.bars], dtype=np.float64)
        regime = detect_regime(closes, highs, lows, is_spy=True)
        return regime.regime.value
    except Exception:
        return "unknown"