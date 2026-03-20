"""
main.py — AlphaBean API Server v3.0

Start: uvicorn backend.main:app --reload --port 8000

Endpoints:
  GET  /api/health              → status
  GET  /api/patterns            → all 47 pattern names
  GET  /api/scan                → scan symbol (5m+15m, scored)
  GET  /api/scan-multiple       → scan multiple symbols
  GET  /api/chart/{symbol}      → candlestick data for frontend
  GET  /api/regime              → current market regime
  GET  /api/hot-strategies      → top performing strategies
  GET  /api/backtest/results    → cached backtest results
"""
import json
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.scanner.engine import scan_symbol, scan_multiple
from backend.patterns.registry import get_all_pattern_names, PATTERN_META
from backend.data.massive_client import fetch_chart_bars, fetch_bars
from backend.regime.detector import detect_regime, MarketRegime, StrategyType
from backend.strategies.evaluator import StrategyEvaluator
from backend.features.engine import compute_features

import numpy as np

app = FastAPI(title="AlphaBean API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global evaluator instance (loaded once at startup)
_evaluator = StrategyEvaluator()
_evaluator.load()

BACKTEST_RESULTS = Path("cache/backtest_results.json")


# ── Core ─────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "app": "AlphaBean", "version": "3.0.0",
            "patterns": len(get_all_pattern_names()),
            "strategies_tracked": _evaluator.stats_summary()["strategies_tracked"]}


@app.get("/api/patterns")
async def list_patterns():
    """All 47 pattern names with metadata."""
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


# ── Scanner ──────────────────────────────────────────────────

@app.get("/api/scan")
async def scan(
    symbol: str = Query(..., description="Ticker, e.g. AAPL"),
    mode: str = Query("today", description="'today' or 'active'"),
):
    """
    Scan a symbol on BOTH 5min + 15min simultaneously.
    Returns multi-factor scored setups sorted by composite score.
    """
    if mode not in ("today", "active"):
        return {"error": "mode must be 'today' or 'active'"}

    print(f"\n{'=' * 55}")
    print(f"  SCAN: {symbol.upper()} | mode={mode} | 5min + 15min | 47 patterns")
    print(f"{'=' * 55}")

    setups = scan_symbol(symbol.upper(), mode=mode, evaluator=_evaluator)

    print(f"  RESULT: {len(setups)} scored setups")
    for s in setups[:5]:
        print(f"    {s['pattern_name']:<25} composite={s.get('composite_score', 0):.0f}  "
              f"conf={s['confidence']:.0%}")

    return {
        "symbol": symbol.upper(), "mode": mode,
        "count": len(setups), "setups": setups,
    }


@app.get("/api/scan-multiple")
async def scan_multi(
    symbols: str = Query(..., description="Comma-separated: AAPL,NVDA,TSLA"),
    mode: str = Query("today"),
):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    setups = scan_multiple(symbol_list, mode=mode, evaluator=_evaluator)
    return {"symbols": symbol_list, "mode": mode, "count": len(setups), "setups": setups}


# ── Chart ────────────────────────────────────────────────────

@app.get("/api/chart/{symbol}")
async def chart_data(
    symbol: str,
    timeframe: str = Query("5min"),
    days_back: int = Query(5),
):
    """Candlestick data for Lightweight Charts."""
    if timeframe not in ("5min", "15min", "1h", "1d"):
        return {"error": f"Invalid timeframe: {timeframe}"}
    try:
        bars = fetch_chart_bars(symbol.upper(), timeframe, days_back)
        return {"symbol": symbol.upper(), "timeframe": timeframe, "bars": bars}
    except Exception as e:
        return {"error": str(e)}


# ── Regime ───────────────────────────────────────────────────

@app.get("/api/regime")
async def get_regime():
    """Current market regime based on SPY data."""
    try:
        spy_bars = fetch_bars("SPY", timeframe="1d", days_back=250)
        closes = np.array([b.close for b in spy_bars.bars], dtype=np.float64)
        highs = np.array([b.high for b in spy_bars.bars], dtype=np.float64)
        lows = np.array([b.low for b in spy_bars.bars], dtype=np.float64)
        regime = detect_regime(closes, highs, lows, is_spy=True)
        return regime.to_dict()
    except Exception as e:
        return {"error": str(e), "regime": "unknown"}


# ── Hot Strategies ───────────────────────────────────────────

@app.get("/api/hot-strategies")
async def hot_strategies(top_n: int = Query(5)):
    """Top performing strategies based on rolling evaluation."""
    hot = _evaluator.get_hot_strategies(top_n=top_n)
    return {
        "count": len(hot),
        "strategies": [m.to_dict() for m in hot],
        "hot_types": _evaluator.get_hot_strategy_types(top_n=3),
    }


@app.get("/api/strategy/{pattern_name}")
async def strategy_detail(pattern_name: str):
    """Detailed performance for a specific pattern/strategy."""
    return _evaluator.get_pattern_summary(pattern_name)


# ── Backtest Results ─────────────────────────────────────────

@app.get("/api/backtest/results")
async def backtest_results():
    """Cached backtest results from run_backtest.py."""
    if not BACKTEST_RESULTS.exists():
        return {"has_results": False, "message": "Run: python run_backtest.py"}
    try:
        data = json.loads(BACKTEST_RESULTS.read_text())
        return {"has_results": True, **data}
    except (json.JSONDecodeError, KeyError):
        return {"has_results": False, "message": "Cache corrupt"}


# ── Reload ───────────────────────────────────────────────────

@app.post("/api/reload-evaluator")
async def reload_evaluator():
    """Reload strategy evaluator from cache (after running backtest)."""
    _evaluator.load()
    return {"status": "reloaded", **_evaluator.stats_summary()}