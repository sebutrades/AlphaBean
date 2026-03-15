# backend/scanner/engine.py
from backend.patterns.base import TradeSetup
from backend.data.massive_client import fetch_bars

def run_scan(symbols: list[str], timeframes: list[str], detectors: list) -> list[TradeSetup]:
    """Scan all symbols × all timeframes × all detectors."""
    setups = []
    for symbol in symbols:
        for tf in timeframes:
            try:
                bars = fetch_bars(symbol, timeframe=tf, days_back=30)
            except Exception:
                continue
            for detector in detectors:
                result = detector.detect(bars)
                if result is not None:
                    setups.append(result)
    # Sort by confidence descending
    setups.sort(key=lambda s: s.confidence, reverse=True)
    return setups