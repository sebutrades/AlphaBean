"""
main.py — FastAPI backend for AlphaBean.

Endpoints:
  GET /api/health          → Health check
  GET /api/scan            → Scan a single symbol
  GET /api/scan-multiple   → Scan multiple symbols
  GET /api/in-play         → Get today's in-play symbols (Claude, cached daily)
  GET /api/top-opps        → Scan all in-play symbols, return best setups

Start with:
  uvicorn backend.main:app --reload --port 8000
"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.scanner.engine import scan_symbol, scan_multiple
from backend.ai.in_play import get_in_play_symbols
from backend.data.massive_client import VALID_TIMEFRAMES

app = FastAPI(title="AlphaBean API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_timeframe(tf: str) -> str:
    if tf not in VALID_TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe '{tf}'. AlphaBean only supports: {VALID_TIMEFRAMES}"
        )
    return tf


@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "AlphaBean API is running", "version": "2.0.0"}


@app.get("/api/scan")
async def scan(
    symbol: str = Query(..., description="Stock ticker, e.g. AAPL, NVDA"),
    timeframe: str = Query("1d", description="15min, 1h, or 1d"),
    days_back: int = Query(30, description="Days of history to analyze"),
):
    validate_timeframe(timeframe)

    print(f"\n{'='*50}")
    print(f"SCAN: {symbol} | {timeframe} | {days_back} days")
    print(f"{'='*50}")

    setups = scan_symbol(symbol=symbol.upper(), timeframe=timeframe, days_back=days_back)
    print(f"RESULT: {len(setups)} setups found\n")

    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "days_back": days_back,
        "count": len(setups),
        "setups": setups,
    }


@app.get("/api/scan-multiple")
async def scan_multi(
    symbols: str = Query(..., description="Comma-separated tickers"),
    timeframes: str = Query("1d", description="Comma-separated: 15min,1h,1d"),
    days_back: int = Query(30, description="Days of history"),
):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    tf_list = [t.strip() for t in timeframes.split(",") if t.strip()]

    for tf in tf_list:
        validate_timeframe(tf)

    setups = scan_multiple(symbols=symbol_list, timeframes=tf_list, days_back=days_back)

    return {
        "symbols": symbol_list,
        "timeframes": tf_list,
        "days_back": days_back,
        "count": len(setups),
        "setups": setups,
    }


@app.get("/api/in-play")
async def in_play(force_refresh: bool = Query(False)):
    """Get today's top 10 in-play symbols (cached daily, 1 Claude API call/day)."""
    result = get_in_play_symbols(force_refresh=force_refresh)
    return result


@app.get("/api/top-opps")
async def top_opps(
    days_back: int = Query(30, description="Days of history"),
):
    """
    Scan ALL in-play symbols across ALL 3 timeframes.
    Returns top setups ranked by confidence.
    """
    in_play = get_in_play_symbols()
    symbols = in_play["symbols"]

    print(f"\n{'='*50}")
    print(f"TOP OPPS: Scanning {len(symbols)} symbols × 3 timeframes")
    print(f"{'='*50}")

    setups = scan_multiple(
        symbols=symbols,
        timeframes=VALID_TIMEFRAMES,
        days_back=days_back,
    )

    print(f"TOP OPPS RESULT: {len(setups)} total setups\n")

    return {
        "in_play_source": in_play["source"],
        "in_play_date": in_play["date"],
        "symbols_scanned": symbols,
        "timeframes": VALID_TIMEFRAMES,
        "count": len(setups),
        "setups": setups,
    }