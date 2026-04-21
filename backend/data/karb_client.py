"""
karb_client.py -- Fetches price data from Karb (localhost:4269) via kti CLI.

Drop-in replacement for massive_client.py.  Exposes the same public API:
    fetch_bars(symbol, timeframe, days_back) -> BarSeries
    fetch_bars_since(symbol, timeframe, since) -> list[Bar]
    fetch_chart_bars(symbol, timeframe, days_back) -> list[dict]
    SCANNER_TIMEFRAMES, ALL_TIMEFRAMES
"""
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, time as dtime

from backend.data.schemas import Bar, BarSeries

# ---- constants ---------------------------------------------------------------

ALL_TIMEFRAMES = {
    "5min":  5,
    "15min": 15,
    "1h":    60,
    "1d":    "daily",
}

SCANNER_TIMEFRAMES = ["5min", "15min", "1h"]

# RTH window (Karb returns extended-hours bars)
_RTH_OPEN  = dtime(9, 30)
_RTH_CLOSE = dtime(16, 0)

# ---- kti runner --------------------------------------------------------------

def _run_kti(args, timeout=120):
    """Run a kti command and return parsed JSON or (None, error)."""
    cmd = ["kti"] + args + ["--json"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()[:200]
            return None, f"kti exit {proc.returncode}: {err}"
        return json.loads(proc.stdout), None
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except json.JSONDecodeError as e:
        return None, f"JSON parse: {e}"
    except FileNotFoundError:
        return None, "kti not found on PATH"


def _filter_rth(bars):
    """Keep only Regular Trading Hours bars (9:30 - 16:00 ET)."""
    return [b for b in bars if _RTH_OPEN <= b.timestamp.time() < _RTH_CLOSE]


def _aggregate(minute_bars, tf_minutes):
    """Aggregate 1-min bars into N-minute bars."""
    if not minute_bars:
        return []
    buckets = defaultdict(list)
    for bar in minute_bars:
        ts = bar.timestamp
        floored = ts.replace(
            minute=(ts.minute // tf_minutes) * tf_minutes,
            second=0, microsecond=0,
        )
        buckets[floored].append(bar)

    agg = []
    for ts in sorted(buckets):
        bb = buckets[ts]
        agg.append(Bar(
            symbol=bb[0].symbol,
            timestamp=ts,
            open=bb[0].open,
            high=max(b.high for b in bb),
            low=min(b.low for b in bb),
            close=bb[-1].close,
            volume=sum(b.volume for b in bb),
        ))
    return agg


# ---- public API (matches massive_client signatures) --------------------------

def fetch_bars(symbol: str, timeframe: str = "15min", days_back: int = 10) -> BarSeries:
    """
    Fetch bars for *one* symbol at the given timeframe.

    For intraday timeframes (5min, 15min, 1h) we pull 1-min bars from Karb,
    filter to RTH, then aggregate locally.  For daily we call kti price daily.
    """
    if timeframe not in ALL_TIMEFRAMES:
        raise ValueError(f"Invalid timeframe: '{timeframe}'. Use: {list(ALL_TIMEFRAMES.keys())}")

    sym = symbol.upper()
    tf_val = ALL_TIMEFRAMES[timeframe]

    if tf_val == "daily":
        data, err = _run_kti(
            ["price", "daily", sym, "--days", str(days_back)],
            timeout=60,
        )
        if err:
            raise RuntimeError(f"Karb daily fetch failed for {sym}: {err}")

        bars = _parse_bars(data, sym)
        return BarSeries(symbol=sym, timeframe=timeframe, bars=bars)

    # Intraday: fetch 1-min bars, filter RTH, aggregate
    # Trading day ~= 390 minutes.  days_back trading days ~ days_back * 1.5 calendar days
    total_minutes = days_back * 780  # generous: covers weekends/holidays
    data, err = _run_kti(
        ["price", "minute", sym, "--minutes", str(total_minutes)],
        timeout=120,
    )
    if err:
        raise RuntimeError(f"Karb minute fetch failed for {sym}: {err}")

    raw_bars = _parse_bars(data, sym)
    rth_bars = _filter_rth(raw_bars)

    if tf_val == 1:
        # 1-min bars (not currently used but future-proof)
        return BarSeries(symbol=sym, timeframe=timeframe, bars=rth_bars)

    agg_bars = _aggregate(rth_bars, tf_val)
    return BarSeries(symbol=sym, timeframe=timeframe, bars=agg_bars)


def fetch_bars_since(symbol: str, timeframe: str, since: datetime) -> list:
    """Incremental fetch: return only bars newer than `since`."""
    delta = (datetime.now().date() - since.date()).days
    days_back = max(delta + 1, 1)
    result = fetch_bars(symbol, timeframe, days_back=days_back)
    if not result:
        return []
    return [b for b in result.bars if b.timestamp > since]


def fetch_chart_bars(symbol: str, timeframe: str = "5min", days_back: int = 5) -> list[dict]:
    """Fetch bars formatted for TradingView Lightweight Charts."""
    bars_series = fetch_bars(symbol, timeframe, days_back)
    return [
        {
            "time": int(b.timestamp.timestamp()),
            "open": b.open, "high": b.high, "low": b.low, "close": b.close,
            "volume": b.volume,
        }
        for b in bars_series.bars
    ]


def get_client():
    """Stub for code that calls get_client() -- no longer needed with Karb."""
    raise NotImplementedError(
        "get_client() is a Massive.com API concept. "
        "Use fetch_bars() / fetch_chart_bars() instead (Karb-based)."
    )


# ---- internal helpers --------------------------------------------------------

def _parse_bars(data, symbol):
    """Parse kti price JSON output into list[Bar]."""
    bars = []
    if not data:
        return bars

    # kti price returns {"bars": {"SYMBOL": [...]}}
    bars_dict = data.get("bars", {})
    raw_list = bars_dict.get(symbol, bars_dict.get(symbol.upper(), []))

    for b in raw_list:
        ts_str = b.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            continue
        bars.append(Bar(
            symbol=symbol.upper(),
            timestamp=ts,
            open=float(b["open"]),
            high=float(b["high"]),
            low=float(b["low"]),
            close=float(b["close"]),
            volume=int(b.get("volume", 0)),
            vwap=None,
            trade_count=None,
        ))
    return bars
