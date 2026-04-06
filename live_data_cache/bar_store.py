"""
live_data_cache/bar_store.py — Rolling-window persistent bar storage

Stores OHLCV bars per symbol × timeframe as JSON in:
    live_data_cache/data/{timeframe}/{SYMBOL}.json

Rolling windows (max bars kept):
    5min   →  10 trading days  ≈  780 bars
    15min  →  10 trading days  ≈  260 bars
    1h     →  30 trading days  ≈  210 bars
    1d     →  unlimited (append-only, capped at 2 520 ≈ 10 years)

Timestamps are stored as ISO-format ET naive datetimes so the file is
readable and timezone-unambiguous. Thread-safe via per-(symbol, tf) locks
and atomic rename writes.
"""
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.data.schemas import Bar, BarSeries

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path("live_data_cache/data")

MAX_BARS: dict[str, int] = {
    "5min":  780,    # 10 trading days × 78 bars/day
    "15min": 260,    # 10 trading days × 26 bars/day
    "1h":    210,    # 30 trading days × ~7 bars/day
    "1d":    2520,   # ~10 years — effectively unlimited for daily
}

# Full backfill depth when the store has no data for a symbol
BACKFILL_DAYS: dict[str, int] = {
    "5min":  14,    # 14 calendar days → ~10 trading days
    "15min": 14,
    "1h":    45,    # 45 calendar days → ~30 trading days
    "1d":    400,   # 400 calendar days → ~270 trading days (~1.5 years)
}

_TS_FMT = "%Y-%m-%dT%H:%M:%S"   # no microseconds for compact, stable keys

# Per-(symbol, timeframe) write locks
_lock_registry: dict[str, threading.Lock] = {}
_registry_lock  = threading.Lock()


def _lock(symbol: str, tf: str) -> threading.Lock:
    key = f"{symbol.upper()}_{tf}"
    with _registry_lock:
        if key not in _lock_registry:
            _lock_registry[key] = threading.Lock()
        return _lock_registry[key]


def _file(symbol: str, tf: str) -> Path:
    p = ROOT / tf / f"{symbol.upper()}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar_to_dict(bar: Bar) -> dict:
    d = {
        "t": bar.timestamp.strftime(_TS_FMT),
        "o": bar.open,
        "h": bar.high,
        "l": bar.low,
        "c": bar.close,
        "v": bar.volume,
    }
    if bar.vwap is not None:
        d["vw"] = bar.vwap
    if bar.trade_count is not None:
        d["n"] = bar.trade_count
    return d


def _dict_to_bar(d: dict, symbol: str) -> Bar:
    return Bar(
        symbol=symbol.upper(),
        timestamp=datetime.strptime(d["t"], _TS_FMT),
        open=d["o"], high=d["h"], low=d["l"], close=d["c"],
        volume=int(d["v"]),
        vwap=d.get("vw"),
        trade_count=d.get("n"),
    )


def _read_raw(symbol: str, tf: str) -> list[dict]:
    """Read raw bar dicts from disk. Returns [] on any error."""
    p = _file(symbol, tf)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text()).get("bars", [])
    except Exception:
        return []


def _write_raw(symbol: str, tf: str, bars: list[dict]) -> None:
    """Atomically write bar dicts to disk."""
    p = _file(symbol, tf)
    payload = {
        "symbol": symbol.upper(),
        "timeframe": tf,
        "last_updated": datetime.now().strftime(_TS_FMT),
        "bar_count": len(bars),
        "bars": bars,
    }
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(p)


# ── Public API ────────────────────────────────────────────────────────────────

def get_bars(symbol: str, tf: str) -> Optional[BarSeries]:
    """
    Load stored bars as a BarSeries ready for the scanner/classifier.
    Returns None if no data exists.
    """
    raw = _read_raw(symbol, tf)
    if not raw:
        return None
    try:
        bars = [_dict_to_bar(d, symbol) for d in raw]
        return BarSeries(symbol=symbol.upper(), timeframe=tf, bars=bars)
    except Exception as e:
        print(f"  [BarStore] parse error {symbol} {tf}: {e}")
        return None


def get_last_timestamp(symbol: str, tf: str) -> Optional[datetime]:
    """Timestamp of the most recent stored bar, or None."""
    raw = _read_raw(symbol, tf)
    if not raw:
        return None
    try:
        return datetime.strptime(raw[-1]["t"], _TS_FMT)
    except Exception:
        return None


def needs_backfill(symbol: str, tf: str) -> bool:
    """True when no bars are stored yet."""
    p = _file(symbol, tf)
    return not p.exists() or p.stat().st_size < 60


def append_bars(symbol: str, tf: str, new_bars: list[Bar]) -> int:
    """
    Merge new_bars into the store (dedup by timestamp), trim to rolling window.
    Returns the count of bars actually inserted (0 = all were duplicates).
    """
    if not new_bars:
        return 0

    symbol = symbol.upper()
    with _lock(symbol, tf):
        existing = _read_raw(symbol, tf)
        known    = {d["t"] for d in existing}

        added = 0
        for bar in new_bars:
            ts = bar.timestamp.strftime(_TS_FMT)
            if ts not in known:
                existing.append(_bar_to_dict(bar))
                known.add(ts)
                added += 1

        if added == 0:
            return 0

        # Sort chronologically
        existing.sort(key=lambda d: d["t"])

        # Rolling window trim
        cap = MAX_BARS.get(tf, 2520)
        if len(existing) > cap:
            existing = existing[-cap:]

        _write_raw(symbol, tf, existing)
        return added


def get_bar_count(symbol: str, tf: str) -> int:
    """Return the number of bars currently stored."""
    raw = _read_raw(symbol, tf)
    return len(raw)


def get_store_stats() -> dict:
    """
    Summary of everything in the bar store — useful for health checks
    and the /api/cache/status endpoint.
    """
    if not ROOT.exists():
        return {"total_files": 0, "total_bars": 0, "by_timeframe": {}}

    total_files = 0
    total_bars  = 0
    by_tf: dict = {}

    for tf_dir in sorted(ROOT.iterdir()):
        if not tf_dir.is_dir():
            continue
        tf    = tf_dir.name
        files = 0
        bars  = 0
        oldest = ""
        newest = ""
        for f in tf_dir.glob("*.json"):
            try:
                data   = json.loads(f.read_text())
                b_list = data.get("bars", [])
                bars  += len(b_list)
                files += 1
                if b_list:
                    if not oldest or b_list[0]["t"] < oldest:
                        oldest = b_list[0]["t"]
                    if not newest or b_list[-1]["t"] > newest:
                        newest = b_list[-1]["t"]
            except Exception:
                pass
        by_tf[tf] = {"symbols": files, "total_bars": bars,
                     "oldest_bar": oldest, "newest_bar": newest}
        total_files += files
        total_bars  += bars

    return {
        "total_files": total_files,
        "total_bars":  total_bars,
        "by_timeframe": by_tf,
        "data_root": str(ROOT),
    }
