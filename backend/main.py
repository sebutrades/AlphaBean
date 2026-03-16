"""
main.py — AlphaBean API Server v2.1 (Phase 2: Backtesting)

Start: uvicorn backend.main:app --reload --port 8000
"""
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from backend.scanner.engine import scan_symbol, scan_multiple
from backend.patterns.edgefinder_patterns import get_all_pattern_names
from backend.backtest.engine import (
    run_full_backtest, get_backtest_results, get_pattern_stats, get_pattern_score,
)
from backend.backtest.data_fetcher import fetch_all_data, get_cache_stats
from backend.backtest.universe import get_universe
from backend.backtest.metrics import get_market_regime, get_relative_strength, get_momentum_score

app = FastAPI(title="AlphaBean API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core Endpoints (from Phase 1) ───────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "app": "AlphaBean", "version": "2.1.0"}


@app.get("/api/patterns")
async def list_patterns():
    return {"patterns": get_all_pattern_names()}


@app.get("/api/scan")
async def scan(
    symbol: str = Query(...),
    timeframe: str = Query("1d"),
    days_back: int = Query(30),
):
    if timeframe not in ("15min", "1h", "1d"):
        return {"error": f"Invalid timeframe '{timeframe}'. Use: 15min, 1h, 1d"}

    print(f"\n{'=' * 50}")
    print(f"SCAN: {symbol.upper()} | {timeframe} | {days_back}d")
    print(f"{'=' * 50}")

    setups = scan_symbol(symbol.upper(), timeframe, days_back)

    # Enrich each setup with backtest score if available
    for s in setups:
        s["backtest_score"] = get_pattern_score(s["pattern_name"], timeframe)

    print(f"RESULT: {len(setups)} setups\n")
    return {
        "symbol": symbol.upper(), "timeframe": timeframe,
        "days_back": days_back, "count": len(setups), "setups": setups,
    }


@app.get("/api/scan-multiple")
async def scan_multi(
    symbols: str = Query(...),
    timeframes: str = Query("1d"),
    days_back: int = Query(30),
):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    tf_list = [t.strip() for t in timeframes.split(",") if t.strip()]
    for tf in tf_list:
        if tf not in ("15min", "1h", "1d"):
            return {"error": f"Invalid timeframe '{tf}'."}

    setups = scan_multiple(symbol_list, tf_list, days_back)
    return {
        "symbols": symbol_list, "timeframes": tf_list,
        "days_back": days_back, "count": len(setups), "setups": setups,
    }


# ── Backtest Endpoints (Phase 2) ────────────────────────────

@app.get("/api/backtest/status")
async def backtest_status():
    """Check if backtest results exist and what data is cached."""
    results = get_backtest_results()
    cache_stats = get_cache_stats()
    return {
        "has_results": results is not None,
        "generated": results.get("generated") if results else None,
        "quarter": results.get("quarter") if results else None,
        "total_signals": results.get("total_signals") if results else 0,
        "patterns_tested": len(results.get("patterns", {})) if results else 0,
        "cache": cache_stats,
    }


@app.post("/api/backtest/fetch-data")
async def backtest_fetch_data(
    background_tasks: BackgroundTasks,
    symbols_count: int = Query(50, description="How many top symbols (max 300)"),
):
    """
    Step 1 of backtesting: Fetch and cache bar data.
    Runs in background so the API doesn't time out.
    """
    count = min(symbols_count, 300)
    symbols = get_universe()[:count]

    def do_fetch():
        fetch_all_data(symbols=symbols, timeframes=["15min", "1h", "1d"])

    background_tasks.add_task(do_fetch)

    return {
        "status": "started",
        "message": f"Fetching data for {count} symbols × 3 timeframes in background",
        "symbols_count": count,
    }


@app.post("/api/backtest/run")
async def backtest_run(
    background_tasks: BackgroundTasks,
    symbols_count: int = Query(50, description="How many symbols to backtest"),
    force: bool = Query(False, description="Force re-run even if cached"),
):
    """
    Step 2 of backtesting: Run pattern detection on all cached data.
    This is the heavy computation — runs in background.
    """
    symbols = get_universe()[:min(symbols_count, 300)]

    def do_backtest():
        run_full_backtest(symbols=symbols, force=force)

    background_tasks.add_task(do_backtest)

    return {
        "status": "started",
        "message": f"Running backtest on {len(symbols)} symbols",
        "check_status_at": "/api/backtest/status",
    }


@app.get("/api/backtest/results")
async def backtest_results():
    """Get the cached backtest results summary."""
    results = get_backtest_results()
    if results is None:
        return {
            "error": "No backtest results found. Run /api/backtest/fetch-data first, then /api/backtest/run.",
            "has_results": False,
        }
    return results


@app.get("/api/backtest/pattern/{pattern_name}")
async def backtest_pattern(pattern_name: str):
    """Get detailed backtest stats for a specific pattern."""
    stats = get_pattern_stats(pattern_name)
    if stats is None:
        return {"error": f"No backtest data for pattern '{pattern_name}'", "pattern": pattern_name}
    return {"pattern": pattern_name, "stats": stats}


@app.get("/api/backtest/score/{pattern_name}/{timeframe}")
async def backtest_score(pattern_name: str, timeframe: str):
    """Get the 0-100 backtest quality score for a pattern+timeframe."""
    score = get_pattern_score(pattern_name, timeframe)
    return {"pattern": pattern_name, "timeframe": timeframe, "score": score}


# ── Market Metrics Endpoints (Phase 2) ──────────────────────

@app.get("/api/metrics/regime")
async def market_regime():
    """Get current market regime (bull/bear based on SPY vs 200 SMA)."""
    return get_market_regime()


@app.get("/api/metrics/rs/{symbol}")
async def relative_strength(symbol: str):
    """Get relative strength of a stock vs SPY."""
    rs = get_relative_strength(symbol.upper())
    if rs is None:
        return {"error": f"No cached data for {symbol.upper()}. Run backtest data fetch first."}
    return rs


@app.get("/api/metrics/momentum/{symbol}")
async def momentum(symbol: str):
    """Get momentum score (0-100) for a stock."""
    m = get_momentum_score(symbol.upper())
    if m is None:
        return {"error": f"No cached data for {symbol.upper()}."}
    return m