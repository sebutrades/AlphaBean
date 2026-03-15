"""
main.py — FastAPI backend for EdgeFinder.

Endpoints:
  GET /api/health          → Check if server is running
  GET /api/scan            → Scan a symbol for patterns
  GET /api/scan-multiple   → Scan multiple symbols at once

Start with:
  uvicorn backend.main:app --reload --port 8000
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.scanner.engine import scan_symbol, scan_multiple

app = FastAPI(title="EdgeFinder API", version="1.0.0")

# Allow the React frontend (running on port 5173) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # In case you use a different port
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    """Simple health check — hit this first to make sure the server is running."""
    return {"status": "ok", "message": "EdgeFinder API is running"}


@app.get("/api/scan")
async def scan(
    symbol: str = Query(..., description="Stock ticker, e.g. AAPL, NVDA"),
    timeframe: str = Query("1d", description="Bar size: 1min, 5min, 15min, 1h, 1d"),
    days_back: int = Query(30, description="How many days of history to analyze"),
):
    """
    Scan a single symbol for all patterns.

    Example: /api/scan?symbol=AAPL&timeframe=1d&days_back=30
    
    Returns a list of trade setups found, sorted by confidence.
    """
    print(f"\n{'='*50}")
    print(f"SCAN REQUEST: {symbol} | {timeframe} | {days_back} days")
    print(f"{'='*50}")

    setups = scan_symbol(
        symbol=symbol.upper(),
        timeframe=timeframe,
        days_back=days_back,
    )

    print(f"RESULT: {len(setups)} setups found for {symbol}\n")

    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "days_back": days_back,
        "count": len(setups),
        "setups": setups,
    }


@app.get("/api/scan-multiple")
async def scan_multi(
    symbols: str = Query(..., description="Comma-separated tickers: AAPL,NVDA,TSLA"),
    timeframes: str = Query("1d", description="Comma-separated: 1min,5min,1d"),
    days_back: int = Query(30, description="Days of history"),
):
    """
    Scan multiple symbols at once.

    Example: /api/scan-multiple?symbols=AAPL,NVDA,TSLA&timeframes=1d&days_back=30
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    tf_list = [t.strip() for t in timeframes.split(",")]

    print(f"\n{'='*50}")
    print(f"MULTI-SCAN: {symbol_list} | {tf_list} | {days_back} days")
    print(f"{'='*50}")

    setups = scan_multiple(
        symbols=symbol_list,
        timeframes=tf_list,
        days_back=days_back,
    )

    print(f"RESULT: {len(setups)} total setups found\n")

    return {
        "symbols": symbol_list,
        "timeframes": tf_list,
        "days_back": days_back,
        "count": len(setups),
        "setups": setups,
    }