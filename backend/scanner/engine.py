"""
engine.py — The scanner that ties everything together.

It fetches price data from Massive.com, runs all pattern detectors
against it, and returns a ranked list of trade setups.
"""
from backend.data.massive_client import fetch_bars
from backend.patterns.edgefinder_patterns import get_all_detectors, TradeSetup
import traceback


def scan_symbol(
    symbol: str,
    timeframe: str = "1min",
    days_back: int = 30,
) -> list[dict]:
    """
    Scan a single symbol for all patterns on a single timeframe.
    
    Returns a list of dicts (JSON-serializable) with trade setups found.
    """
    # Step 1: Fetch price data
    try:
        bars = fetch_bars(symbol, timeframe=timeframe, days_back=days_back)
    except Exception as e:
        print(f"  [ERROR] Failed to fetch {symbol} {timeframe}: {e}")
        return []

    if len(bars.bars) < 10:
        print(f"  [SKIP] {symbol} {timeframe}: only {len(bars.bars)} bars (need 10+)")
        return []

    print(f"  [OK] {symbol} {timeframe}: {len(bars.bars)} bars loaded")

    # Step 2: Run every detector
    detectors = get_all_detectors()
    setups = []

    for detector in detectors:
        try:
            result = detector.detect(bars)
            if result is not None:
                print(f"  [FOUND] {detector.name} on {symbol}!")
                setups.append(_setup_to_dict(result))
        except Exception as e:
            # Don't crash the whole scan if one detector fails
            print(f"  [WARN] {detector.name} error on {symbol}: {e}")
            continue

    # Step 3: Sort by confidence (highest first)
    setups.sort(key=lambda s: s["confidence"], reverse=True)
    return setups


def scan_multiple(
    symbols: list[str],
    timeframes: list[str] = None,
    days_back: int = 30,
) -> list[dict]:
    """
    Scan multiple symbols across multiple timeframes.
    """
    if timeframes is None:
        timeframes = ["1d"]

    all_setups = []
    total = len(symbols) * len(timeframes)
    done = 0

    for symbol in symbols:
        for tf in timeframes:
            done += 1
            print(f"[{done}/{total}] Scanning {symbol} on {tf}...")
            results = scan_symbol(symbol, timeframe=tf, days_back=days_back)
            all_setups.extend(results)

    all_setups.sort(key=lambda s: s["confidence"], reverse=True)
    return all_setups


def _setup_to_dict(setup: TradeSetup) -> dict:
    """Convert a TradeSetup dataclass to a JSON-serializable dict."""
    return {
        "pattern_name": setup.pattern_name,
        "symbol": setup.symbol,
        "bias": setup.bias.value if hasattr(setup.bias, 'value') else str(setup.bias),
        "timeframe": setup.timeframe.value if hasattr(setup.timeframe, 'value') else str(setup.timeframe),
        "entry_price": setup.entry_price,
        "stop_loss": setup.stop_loss,
        "target_price": setup.target_price,
        "risk_reward_ratio": setup.risk_reward_ratio,
        "confidence": setup.confidence,
        "detected_at": setup.detected_at.isoformat(),
        "description": setup.description,
        "win_rate": setup.win_rate,
        "max_attempts": setup.max_attempts,
        "exit_strategy": setup.exit_strategy,
        "key_levels": setup.key_levels,
        "ideal_time": getattr(setup, 'ideal_time', ''),
    }