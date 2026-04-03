"""
backend/tracker/routes.py — Trade Tracker API Routes

Wire into your FastAPI app:
  from backend.tracker.routes import router as tracker_router
  app.include_router(tracker_router, prefix="/api/tracker")

Endpoints:
  GET  /api/tracker/trades          — All active trades
  GET  /api/tracker/trades/closed   — Closed trades history
  GET  /api/tracker/trades/{id}     — Single trade detail
  GET  /api/tracker/summary         — Portfolio summary
  POST /api/tracker/scan            — Trigger scan for new setups
  POST /api/tracker/refresh         — Refresh prices for all trades
  POST /api/tracker/trades          — Manually add a trade
  POST /api/tracker/trades/{id}/close  — Close a trade
  DELETE /api/tracker/trades/{id}   — Remove a trade
  GET  /api/tracker/trades/{id}/chart — Get chart data for a trade
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np

from backend.tracker.trade_tracker import TradeTracker

router = APIRouter(tags=["tracker"])

# Singleton tracker instance
_tracker: Optional[TradeTracker] = None

def get_tracker() -> TradeTracker:
    global _tracker
    if _tracker is None:
        _tracker = TradeTracker()
    return _tracker


# ── Models ──

class ManualTradeRequest(BaseModel):
    symbol: str
    pattern_name: str = "Manual"
    bias: str = "long"
    timeframe: str = "1d"
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float = 0.0
    confidence: float = 0.5
    description: str = ""
    notes: str = ""

class CloseTradeRequest(BaseModel):
    reason: str = "manual"

class ScanRequest(BaseModel):
    top_n: int = 50
    symbols: list[str] = None


# ── Routes ──

@router.get("/trades")
def get_active_trades():
    tracker = get_tracker()
    trades = tracker.get_active_trades()
    return {
        "count": len(trades),
        "trades": trades,
    }


@router.get("/trades/closed")
def get_closed_trades():
    tracker = get_tracker()
    trades = tracker.get_closed_trades()
    return {
        "count": len(trades),
        "trades": trades,
    }


@router.get("/trades/all")
def get_all_trades():
    tracker = get_tracker()
    return {
        "count": len(tracker.trades),
        "trades": tracker.get_all_trades(),
    }


@router.get("/trades/{trade_id}")
def get_trade(trade_id: str):
    tracker = get_tracker()
    trade = tracker.get_trade_by_id(trade_id)
    if not trade:
        raise HTTPException(404, "Trade not found")
    return trade


@router.get("/summary")
def get_summary():
    tracker = get_tracker()
    return tracker.get_summary()


@router.post("/scan")
def trigger_scan(req: ScanRequest = ScanRequest()):
    tracker = get_tracker()
    added = tracker.scan_and_add(top_n=req.top_n, symbols=req.symbols)
    return {
        "added": added,
        "total_active": sum(1 for t in tracker.trades if t.is_active),
    }


@router.post("/refresh")
def refresh_prices():
    tracker = get_tracker()
    tracker.refresh_prices()
    return {
        "refreshed": sum(1 for t in tracker.trades if t.is_active),
        "summary": tracker.get_summary(),
    }


@router.post("/trades")
def add_manual_trade(req: ManualTradeRequest):
    tracker = get_tracker()
    trade = tracker.add_manual(req.dict())
    return trade.to_dict()


@router.post("/trades/{trade_id}/close")
def close_trade(trade_id: str, req: CloseTradeRequest = CloseTradeRequest()):
    tracker = get_tracker()
    if tracker.close_trade(trade_id, req.reason):
        return {"status": "closed", "trade_id": trade_id}
    raise HTTPException(404, "Trade not found or already closed")


@router.delete("/trades/{trade_id}")
def remove_trade(trade_id: str):
    tracker = get_tracker()
    tracker.remove_trade(trade_id)
    return {"status": "removed", "trade_id": trade_id}


@router.get("/trades/{trade_id}/chart")
def get_trade_chart_data(trade_id: str, bars: int = 60):
    """Get OHLCV bar data for a trade's chart.

    Returns bars from before the setup was detected through now,
    plus the key levels (entry, stop, T1, T2) for overlay.
    """
    tracker = get_tracker()
    trade = tracker.get_trade_by_id(trade_id)
    if not trade:
        raise HTTPException(404, "Trade not found")

    from backend.data.massive_client import fetch_bars as _fetch

    tf = trade["timeframe"]
    symbol = trade["symbol"]

    try:
        bar_data = _fetch(symbol, tf, bars)
        ohlcv = []
        for b in bar_data.bars:
            ohlcv.append({
                "t": b.timestamp.isoformat(),
                "o": round(b.open, 2),
                "h": round(b.high, 2),
                "l": round(b.low, 2),
                "c": round(b.close, 2),
                "v": int(b.volume),
            })

        # Simple ATR from last 14 bars
        current_atr = 0.0
        if len(bar_data.bars) >= 14:
            current_atr = round(float(np.mean([b.high - b.low for b in bar_data.bars[-14:]])), 2)

        return {
            "symbol": symbol,
            "timeframe": tf,
            "bars": ohlcv,
            "levels": {
                "entry": trade["entry_price"],
                "stop": trade["stop_loss"],
                "target_1": trade["target_1"],
                "target_2": trade["target_2"],
                "trailing_stop": trade.get("trailing_stop", 0),
            },
            "trade": trade,
            "current_atr": round(current_atr, 2),
        }
    except Exception as e:
        raise HTTPException(500, f"Error fetching chart data: {e}")


@router.post("/archive")
def archive_closed():
    tracker = get_tracker()
    n = tracker.archive_closed()
    return {"archived": n}