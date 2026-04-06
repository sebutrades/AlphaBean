"""
live_data_cache/intraday_setup_tracker.py — Intraday Setup Lifecycle Manager

Tracks 5-min and 15-min setups from first detection through resolution.

Lifecycle:
    PENDING   → detected, waiting for price to reach entry
    ACTIVE    → entry hit, trade is live
    AT_T1     → first target hit, partial exit, stop moved to breakeven
    STOPPED   → stop (or breakeven stop) hit — loss or scratch
    AT_T2     → second target hit, remainder exits (counted as a win)
    EXPIRED   → stop hit before entry (setup invalidated)

Overnight holds:
    Any intraday setup still PENDING/ACTIVE/AT_T1 when EOD is called gets
    flagged  overnight_hold_risk = True  so the trader can decide whether
    to hold or cut.  The setup stays open and continues to be tracked the
    next morning.

Storage:
    live_data_cache/open_setups.json            — all currently open setups
    live_data_cache/closed_YYYY-MM-DD.json      — daily archive of closed setups
    live_data_cache/daily_perf.json             — rolling performance log

Detection strategy:
    After each 5-min or 15-min bar update for a symbol:
    1. Load full bar history from bar_store (has multi-day context)
    2. Run all pattern classifiers (classify_all)
    3. Filter to setups where entry is within 2 % of current price
    4. Deduplicate: skip if (symbol, pattern, bias, price_bucket) already open
    5. Update every existing open setup with new bar data
    6. Archive setups that hit stop or target

    This means the system always operates on the freshest available bars
    (the bar store is updated before the classifier is run), so a setup
    that needs yesterday's close + today's first hour for context is
    detected correctly.
"""
import json
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Optional

STORE_ROOT = Path("live_data_cache")
OPEN_FILE  = STORE_ROOT / "open_setups.json"
DAILY_PERF = STORE_ROOT / "daily_perf.json"

INTRADAY_TFS = {"5min", "15min"}

# Setups whose entry is more than this far from current price are stale
ENTRY_PROXIMITY_PCT = 0.025   # 2.5 %

_lock = threading.Lock()


# ── Persistence helpers ───────────────────────────────────────────────────────

def _load_open() -> list[dict]:
    if OPEN_FILE.exists():
        try:
            return json.loads(OPEN_FILE.read_text()).get("setups", [])
        except Exception:
            return []
    return []


def _save_open(setups: list[dict]) -> None:
    STORE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = OPEN_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "updated_at": datetime.now().isoformat(),
        "count": len(setups),
        "setups": setups,
    }, indent=2))
    tmp.replace(OPEN_FILE)


def _archive_closed(closed: list[dict]) -> None:
    today = date.today().isoformat()
    path  = STORE_ROOT / f"closed_{today}.json"
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text()).get("setups", [])
        except Exception:
            pass
    existing.extend(closed)
    path.write_text(json.dumps({
        "date": today,
        "count": len(existing),
        "setups": existing,
    }, indent=2))


def _load_daily_perf() -> dict:
    if DAILY_PERF.exists():
        try:
            return json.loads(DAILY_PERF.read_text())
        except Exception:
            pass
    return {"days": {}}


def _save_daily_perf(perf: dict) -> None:
    DAILY_PERF.write_text(json.dumps(perf, indent=2))


# ── Setup conversion ──────────────────────────────────────────────────────────

def _price_bucket(price: float) -> float:
    """Coarse bucket for dedup — tolerates small entry drifts across cycles."""
    if price > 200:
        return round(price / 5) * 5        # bucket width = $5
    elif price > 50:
        return round(price / 2) * 2        # bucket width = $2
    elif price > 10:
        return round(price, 0)             # $1
    else:
        return round(price, 1)             # $0.10


def _setup_to_dict(setup, symbol: str, tf: str) -> dict:
    bias = setup.bias.value if hasattr(setup.bias, "value") else str(setup.bias)
    now  = datetime.now().isoformat()

    t1 = round(float(getattr(setup, "target_1", 0) or 0), 4)
    t2 = round(float(getattr(setup, "target_2", 0) or getattr(setup, "target_price", 0) or 0), 4)

    return {
        "id":             f"{symbol}_{setup.pattern_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "symbol":         symbol,
        "pattern_name":   setup.pattern_name,
        "timeframe":      tf,
        "bias":           bias,
        "entry_price":    round(float(setup.entry_price), 4),
        "stop_loss":      round(float(setup.stop_loss), 4),
        "target_1":       t1,
        "target_2":       t2,
        "confidence":     round(float(setup.confidence), 3),
        "description":    setup.description,
        "detected_at":    (setup.detected_at.isoformat()
                           if hasattr(setup.detected_at, "isoformat")
                           else str(setup.detected_at)),
        "status":         "PENDING",
        "t1_hit":         False,
        "t2_hit":         False,
        "current_price":  round(float(setup.entry_price), 4),
        "unrealized_r":   0.0,
        "realized_r":     0.0,
        "overnight_hold_risk": False,
        "last_bar_ts":    "",     # filled after first update pass
        "added_at":       now,
    }


# ── Core update logic ─────────────────────────────────────────────────────────

def _update_setup_with_bar(setup: dict, bar) -> bool:
    """
    Apply a single new bar to an open setup.
    Mutates `setup` in place.  Returns True if the setup is now closed.
    The bar's worst extreme (low for longs, high for shorts) is used for
    stop detection — catches intraday wicks that later close back above.
    """
    is_long = setup.get("bias", "long") == "long"
    entry   = setup["entry_price"]
    stop    = setup["stop_loss"]
    t1      = setup["target_1"]
    t2      = setup["target_2"]
    status  = setup["status"]

    close   = bar.close
    high    = bar.high
    low     = bar.low
    ts      = bar.timestamp.isoformat()

    setup["last_bar_ts"] = ts

    if status in ("STOPPED", "CLOSED", "EXPIRED", "AT_T2"):
        return True

    # ── PENDING ──────────────────────────────────────────────────────────────
    if status == "PENDING":
        if (is_long and close >= entry) or (not is_long and close <= entry):
            setup["status"] = "ACTIVE"
            setup["entered_at"] = ts
            status = "ACTIVE"
        elif (is_long and low <= stop) or (not is_long and high >= stop):
            # Stop hit before entry — invalidate
            setup["status"]   = "EXPIRED"
            setup["closed_at"]  = ts
            setup["result"]     = "expired_before_entry"
            setup["realized_r"] = 0.0
            return True

    # ── ACTIVE ────────────────────────────────────────────────────────────────
    if status == "ACTIVE":
        worst = low if is_long else high
        risk  = abs(entry - stop)

        if (is_long and worst <= stop) or (not is_long and worst >= stop):
            setup["status"]   = "STOPPED"
            setup["realized_r"] = round((stop - entry) / risk, 3) if is_long and risk > 0 else \
                                   round((entry - stop) / risk, 3) if risk > 0 else -1.0
            setup["closed_at"]  = ts
            setup["result"]     = "stopped"
            return True

        if t1 > 0:
            if (is_long and high >= t1) or (not is_long and low <= t1):
                setup["status"]   = "AT_T1"
                setup["t1_hit"]   = True
                setup["t1_at"]    = ts
                setup["stop_loss"]  = entry          # move stop to breakeven
                status = "AT_T1"
                stop   = entry

    # ── AT_T1 ────────────────────────────────────────────────────────────────
    if status == "AT_T1":
        stop = setup["stop_loss"]                    # breakeven stop
        risk = abs(entry - setup.get("original_stop", stop))
        if risk == 0:
            risk = abs(entry - setup.get("stop_loss", entry) or 1)

        worst = low if is_long else high
        if (is_long and worst <= stop) or (not is_long and worst >= stop):
            setup["status"]   = "STOPPED"
            setup["realized_r"] = 0.0
            setup["closed_at"]  = ts
            setup["result"]     = "breakeven"
            return True

        if t2 > 0:
            if (is_long and high >= t2) or (not is_long and low <= t2):
                orig_risk = abs(entry - (setup.get("original_stop") or setup.get("stop_loss") or entry - 1))
                setup["status"]   = "AT_T2"
                setup["t2_hit"]   = True
                setup["t2_at"]    = ts
                setup["realized_r"] = round(abs(t2 - entry) / orig_risk, 3) if orig_risk > 0 else 2.0
                setup["closed_at"]  = ts
                setup["result"]     = "target_2"
                return True

    # ── Live P&L update (still open) ─────────────────────────────────────────
    setup["current_price"] = close
    orig_stop = setup.get("original_stop") or setup.get("stop_loss") or stop
    risk = abs(entry - orig_stop)
    if risk > 0:
        setup["unrealized_r"] = round(
            (close - entry) / risk if is_long else (entry - close) / risk, 3
        )
    return False


def process_new_bars(symbol: str, tf: str, new_bars: list) -> list[dict]:
    """
    Main entry point — called by the live feed after a bar update.

    1. Loads full bar history from bar_store.
    2. Runs pattern classifiers on the full history.
    3. Filters to actionable setups (entry ≤ 2.5 % from current price).
    4. Adds genuinely new setups (dedup by symbol × pattern × bias × price bucket).
    5. Advances every existing open setup for this symbol/tf through new_bars.
    6. Archives setups that resolved (stopped, target hit, expired).
    7. Persists state and returns the list of newly detected setups.
    """
    if tf not in INTRADAY_TFS or not new_bars:
        return []

    from live_data_cache.bar_store import get_bars
    from backend.patterns.classifier import classify_all

    bars = get_bars(symbol, tf)
    if bars is None or len(bars.bars) < 20:
        return []

    current_close = bars.bars[-1].close

    # ── Run classifiers ───────────────────────────────────────────────────────
    try:
        detected = classify_all(bars)
    except Exception as e:
        print(f"  [IntradayTracker] classify error {symbol} {tf}: {e}")
        detected = []

    # Filter to setups with entry near current price (actionable right now)
    actionable = [
        s for s in detected
        if current_close > 0 and
           abs(s.entry_price - current_close) / current_close <= ENTRY_PROXIMITY_PCT
    ]

    with _lock:
        all_open = _load_open()

        # Split into this-symbol-tf vs everything else
        sym_open   = [s for s in all_open if s["symbol"] == symbol and s["timeframe"] == tf]
        other_open = [s for s in all_open if not (s["symbol"] == symbol and s["timeframe"] == tf)]

        # ── 1. Advance existing setups ────────────────────────────────────────
        newly_closed: list[dict] = []
        still_open:   list[dict] = []

        for setup in sym_open:
            last_ts = setup.get("last_bar_ts", "")
            bars_to_process = [
                b for b in new_bars
                if b.timestamp.isoformat() > last_ts
            ] if last_ts else new_bars

            closed = False
            for bar in bars_to_process:
                closed = _update_setup_with_bar(setup, bar)
                if closed:
                    break

            if closed:
                newly_closed.append(setup)
            else:
                still_open.append(setup)

        # ── 2. Detect new setups ──────────────────────────────────────────────
        existing_keys = {
            (s["symbol"], s["pattern_name"], s["bias"], _price_bucket(s["entry_price"]))
            for s in still_open + other_open
        }

        new_detected: list[dict] = []
        for setup in actionable:
            bias = setup.bias.value if hasattr(setup.bias, "value") else str(setup.bias)
            key  = (symbol, setup.pattern_name, bias, _price_bucket(setup.entry_price))
            if key not in existing_keys:
                d = _setup_to_dict(setup, symbol, tf)
                # Store original stop for R-multiple calculations after T1 moves stop
                d["original_stop"] = d["stop_loss"]
                # Run through current new bars immediately so entry hit can be detected
                for bar in new_bars:
                    if _update_setup_with_bar(d, bar):
                        newly_closed.append(d)
                        d = None
                        break
                if d is not None:
                    still_open.append(d)
                    existing_keys.add(key)
                    new_detected.append(d)

        # ── 3. Persist ────────────────────────────────────────────────────────
        _save_open(other_open + still_open)
        if newly_closed:
            _archive_closed(newly_closed)

        return new_detected


# ── Queries ───────────────────────────────────────────────────────────────────

def get_open_setups(
    symbol: Optional[str] = None,
    tf: Optional[str] = None,
) -> list[dict]:
    """Return all (or filtered) open intraday setups."""
    setups = _load_open()
    if symbol:
        setups = [s for s in setups if s.get("symbol") == symbol]
    if tf:
        setups = [s for s in setups if s.get("timeframe") == tf]
    return setups


def get_closed_setups(target_date: Optional[str] = None) -> list[dict]:
    """Return closed setups for a specific date (ISO) or today."""
    if target_date is None:
        target_date = date.today().isoformat()
    path = STORE_ROOT / f"closed_{target_date}.json"
    if path.exists():
        try:
            return json.loads(path.read_text()).get("setups", [])
        except Exception:
            pass
    return []


# ── EOD management ────────────────────────────────────────────────────────────

def flag_overnight_holds() -> int:
    """
    Called at 4:00 PM ET.  Flags every intraday setup that is still
    PENDING / ACTIVE / AT_T1 with overnight_hold_risk = True.
    The setups remain open and continue to be tracked next morning.
    Returns the number of setups flagged.
    """
    with _lock:
        setups  = _load_open()
        flagged = 0
        for s in setups:
            if s.get("timeframe") in INTRADAY_TFS and \
               s.get("status") in ("PENDING", "ACTIVE", "AT_T1"):
                s["overnight_hold_risk"] = True
                flagged += 1
        _save_open(setups)
    return flagged


def generate_daily_summary() -> dict:
    """
    End-of-day performance summary for today's intraday setups.
    Includes both closed setups and still-open (overnight holds).
    Persists the summary to daily_perf.json.
    """
    today      = date.today().isoformat()
    closed     = get_closed_setups(today)
    open_today = [
        s for s in _load_open()
        if s.get("added_at", "")[:10] == today and s.get("timeframe") in INTRADAY_TFS
    ]
    all_today = closed + open_today

    winners    = [s for s in closed if (s.get("realized_r") or 0) > 0]
    losers     = [s for s in closed if (s.get("realized_r") or 0) < 0]
    breakevens = [s for s in closed if (s.get("realized_r") or 0) == 0 and s.get("status") == "STOPPED"]
    overnight  = [s for s in open_today if s.get("overnight_hold_risk")]
    t1_hits    = [s for s in closed if s.get("t1_hit")]
    t2_hits    = [s for s in closed if s.get("t2_hit")]

    total_r = sum((s.get("realized_r") or 0) for s in closed)

    # Pattern breakdown
    by_pattern: dict = {}
    for s in all_today:
        name = s.get("pattern_name", "Unknown")
        g = by_pattern.setdefault(name, {"count": 0, "wins": 0, "total_r": 0.0})
        g["count"] += 1
        r = s.get("realized_r") or 0
        if r > 0:
            g["wins"] += 1
        g["total_r"] = round(g["total_r"] + r, 2)
    for g in by_pattern.values():
        g["win_rate"] = round(g["wins"] / g["count"] * 100, 1) if g["count"] else 0.0

    summary = {
        "date":              today,
        "total_setups":      len(all_today),
        "closed":            len(closed),
        "still_open":        len(open_today),
        "overnight_holds":   len(overnight),
        "winners":           len(winners),
        "losers":            len(losers),
        "breakevens":        len(breakevens),
        "t1_hits":           len(t1_hits),
        "t2_hits":           len(t2_hits),
        "total_realized_r":  round(total_r, 2),
        "win_rate_pct":      round(len(winners) / len(closed) * 100, 1) if closed else 0.0,
        "avg_winner_r":      round(sum((s.get("realized_r") or 0) for s in winners) / len(winners), 2) if winners else 0.0,
        "avg_loser_r":       round(sum((s.get("realized_r") or 0) for s in losers) / len(losers), 2) if losers else 0.0,
        "by_pattern":        by_pattern,
        "generated_at":      datetime.now().isoformat(),
    }

    perf = _load_daily_perf()
    perf["days"][today] = summary
    _save_daily_perf(perf)

    return summary
