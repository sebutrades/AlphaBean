"""
analytics/performance.py — Performance Dashboard Analytics

Read-only analysis of historical trade data to produce:
  - Equity curve (cumulative R-multiples over time)
  - Daily / Weekly / Monthly P&L grids
  - Pattern attribution (which patterns earn/lose R)
  - Drawdown tracking (max and current drawdown in R)
  - Win rate by market regime
  - Time-of-day analysis (best/worst hours for intraday)
  - Win/loss streak tracking

Data sources:
  - cache/archived_trades.json  (closed swing trades)
  - cache/active_trades.json    (open trades with unrealized_r)
  - live_data_cache/closed_YYYY-MM-DD.json  (intraday closed setups)
  - live_data_cache/daily_perf.json          (daily performance log)
"""
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]  # AlphaBean project root
ARCHIVED_TRADES = _ROOT / "cache" / "archived_trades.json"
ACTIVE_TRADES = _ROOT / "cache" / "active_trades.json"
LIVE_CACHE_DIR = _ROOT / "live_data_cache"
DAILY_PERF = LIVE_CACHE_DIR / "daily_perf.json"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    """Read and parse a JSON file. Returns None if missing or malformed."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_archived_trades() -> list[dict]:
    """Load closed swing trades from archived_trades.json."""
    data = _read_json(ARCHIVED_TRADES)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("trades", [])
    return []


def _load_active_trades() -> list[dict]:
    """Load the active (open) trades list."""
    data = _read_json(ACTIVE_TRADES)
    if isinstance(data, dict):
        return data.get("trades", [])
    if isinstance(data, list):
        return data
    return []


def _load_intraday_closed() -> list[dict]:
    """Aggregate all live_data_cache/closed_YYYY-MM-DD.json files."""
    setups: list[dict] = []
    if not LIVE_CACHE_DIR.exists():
        return setups
    for path in sorted(LIVE_CACHE_DIR.glob("closed_*.json")):
        data = _read_json(path)
        if isinstance(data, dict):
            setups.extend(data.get("setups", []))
        elif isinstance(data, list):
            setups.extend(data)
    return setups


def _load_closed_from_active() -> list[dict]:
    """Extract closed trades still sitting in active_trades.json.

    The tracker keeps closed trades (STOPPED/CLOSED/EXPIRED) in the active
    file until archive_closed() is explicitly called.  We read them here so
    the dashboard sees them regardless.
    """
    data = _read_json(ACTIVE_TRADES)
    if not data:
        return []
    trades = data.get("trades", []) if isinstance(data, dict) else data
    return [t for t in trades if t.get("status") in ("STOPPED", "CLOSED", "EXPIRED")]


def _all_closed_trades() -> list[dict]:
    """Merge archived swing trades, intraday closed setups, and closed
    trades still in active_trades.json.

    Deduplicates by trade id and keeps only trades with a realized_r value
    and a valid closed_at timestamp.  Sorted by closed_at ascending.
    """
    seen_ids: set[str] = set()
    merged: list[dict] = []

    sources = (
        _load_archived_trades()
        + _load_intraday_closed()
        + _load_closed_from_active()
    )

    for trade in sources:
        tid = trade.get("id", "")
        if tid in seen_ids:
            continue
        seen_ids.add(tid)

        # Must have a closed_at and a realized_r to be useful for analytics
        if trade.get("closed_at") is None:
            continue
        if trade.get("realized_r") is None:
            continue

        merged.append(trade)

    # Sort chronologically
    merged.sort(key=lambda t: t.get("closed_at", ""))
    return merged


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO datetime string. Returns None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _trade_date(trade: dict) -> str | None:
    """Return the closed_at date as YYYY-MM-DD string."""
    dt = _parse_dt(trade.get("closed_at"))
    return dt.strftime("%Y-%m-%d") if dt else None


def _round(val: float, decimals: int = 3) -> float:
    return round(val, decimals)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_performance_summary() -> dict:
    """Full dashboard payload: equity curve, drawdown, stats, attribution.

    Returns a dict with all major analytics sections suitable for rendering
    a performance dashboard in the frontend.
    """
    trades = _all_closed_trades()
    total_r = sum(t.get("realized_r", 0) for t in trades)
    wins = [t for t in trades if t.get("realized_r", 0) > 0]
    losses = [t for t in trades if t.get("realized_r", 0) < 0]
    flat = [t for t in trades if t.get("realized_r", 0) == 0]

    win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
    avg_win = (sum(t["realized_r"] for t in wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(t["realized_r"] for t in losses) / len(losses)) if losses else 0.0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    gross_win = sum(t["realized_r"] for t in wins)
    gross_loss = abs(sum(t["realized_r"] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (
        99.0 if gross_win > 0 else 0.0
    )

    # Active trade summary
    active = _load_active_trades()
    open_r = sum(t.get("unrealized_r", 0) for t in active)

    return {
        "total_trades": len(trades),
        "total_r": _round(total_r),
        "wins": len(wins),
        "losses": len(losses),
        "flat": len(flat),
        "win_rate": _round(win_rate, 1),
        "avg_win_r": _round(avg_win),
        "avg_loss_r": _round(avg_loss),
        "expectancy": _round(expectancy, 4),
        "profit_factor": _round(profit_factor, 2),
        "open_trades": len(active),
        "open_r": _round(open_r),
        "equity_curve": get_equity_curve(),
        "drawdown": get_drawdown_series(),
        "pattern_attribution": get_pattern_attribution(),
        "streaks": get_streaks(),
        "time_of_day": get_time_of_day_stats(),
    }


def get_equity_curve() -> list[dict]:
    """Ordered list of daily equity points.

    Each entry: {date, cumulative_r, daily_r, trade_count}.
    R-multiples are the universal performance measure — no dollar values.
    """
    trades = _all_closed_trades()
    if not trades:
        return []

    # Group by closed date
    daily: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        d = _trade_date(t)
        if d:
            daily[d].append(t)

    curve: list[dict] = []
    cumulative_r = 0.0

    for date_str in sorted(daily.keys()):
        day_trades = daily[date_str]
        day_r = sum(t.get("realized_r", 0) for t in day_trades)
        cumulative_r += day_r
        curve.append({
            "date": date_str,
            "cumulative_r": _round(cumulative_r),
            "daily_r": _round(day_r),
            "trade_count": len(day_trades),
        })

    return curve


def get_pattern_attribution() -> list[dict]:
    """Per-pattern breakdown sorted by total_r descending.

    Each entry: {pattern_name, total_r, trade_count, win_rate, avg_r,
                 best_r, worst_r}.
    """
    trades = _all_closed_trades()
    if not trades:
        return []

    buckets: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        name = t.get("pattern_name", "Unknown")
        buckets[name].append(t.get("realized_r", 0))

    results: list[dict] = []
    for pattern, rs in buckets.items():
        wins = sum(1 for r in rs if r > 0)
        total = len(rs)
        results.append({
            "pattern_name": pattern,
            "total_r": _round(sum(rs)),
            "trade_count": total,
            "win_rate": _round(wins / total * 100, 1) if total else 0.0,
            "avg_r": _round(sum(rs) / total) if total else 0.0,
            "best_r": _round(max(rs)),
            "worst_r": _round(min(rs)),
        })

    results.sort(key=lambda x: x["total_r"], reverse=True)
    return results


def get_drawdown_series() -> dict:
    """Max drawdown, current drawdown, and the drawdown-over-time series.

    Drawdown is measured in R-multiples: (peak_cumulative - current_cumulative).
    Returns:
      {max_drawdown_r, current_drawdown_r, max_drawdown_date,
       series: [{date, drawdown_r}]}
    """
    curve = get_equity_curve()
    if not curve:
        return {
            "max_drawdown_r": 0.0,
            "current_drawdown_r": 0.0,
            "max_drawdown_date": None,
            "series": [],
        }

    peak = 0.0
    max_dd = 0.0
    max_dd_date = curve[0]["date"]
    series: list[dict] = []

    for point in curve:
        cum = point["cumulative_r"]
        if cum > peak:
            peak = cum
        dd = peak - cum  # always >= 0
        if dd > max_dd:
            max_dd = dd
            max_dd_date = point["date"]
        series.append({"date": point["date"], "drawdown_r": _round(dd)})

    current_dd = series[-1]["drawdown_r"] if series else 0.0

    return {
        "max_drawdown_r": _round(max_dd),
        "current_drawdown_r": _round(current_dd),
        "max_drawdown_date": max_dd_date,
        "series": series,
    }


def get_time_of_day_stats() -> list[dict]:
    """Performance by hour of day (0-23).

    Uses detected_at (preferred) or entered_at to determine the hour.
    Each entry: {hour, trade_count, win_rate, avg_r, total_r}.
    """
    trades = _all_closed_trades()
    if not trades:
        return []

    hourly: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        dt = _parse_dt(t.get("detected_at")) or _parse_dt(t.get("entered_at"))
        if dt is None:
            continue
        hourly[dt.hour].append(t.get("realized_r", 0))

    results: list[dict] = []
    for hour in sorted(hourly.keys()):
        rs = hourly[hour]
        wins = sum(1 for r in rs if r > 0)
        total = len(rs)
        results.append({
            "hour": hour,
            "trade_count": total,
            "win_rate": _round(wins / total * 100, 1) if total else 0.0,
            "avg_r": _round(sum(rs) / total) if total else 0.0,
            "total_r": _round(sum(rs)),
        })

    return results


def get_regime_stats() -> dict:
    """Win rate by market regime.

    If trades carry a 'regime' field (e.g., trending, choppy, volatile),
    this groups performance by that label.  Falls back to an empty breakdown
    if no regime data is present.

    Returns: {regimes: [{regime, trade_count, win_rate, avg_r, total_r}],
              has_regime_data: bool}
    """
    trades = _all_closed_trades()
    if not trades:
        return {"regimes": [], "has_regime_data": False}

    buckets: dict[str, list[float]] = defaultdict(list)
    has_data = False

    for t in trades:
        regime = t.get("regime") or t.get("market_regime") or "unknown"
        if regime != "unknown":
            has_data = True
        buckets[regime].append(t.get("realized_r", 0))

    results: list[dict] = []
    for regime, rs in sorted(buckets.items()):
        wins = sum(1 for r in rs if r > 0)
        total = len(rs)
        results.append({
            "regime": regime,
            "trade_count": total,
            "win_rate": _round(wins / total * 100, 1) if total else 0.0,
            "avg_r": _round(sum(rs) / total) if total else 0.0,
            "total_r": _round(sum(rs)),
        })

    return {"regimes": results, "has_regime_data": has_data}


def get_daily_pnl(days: int = 30) -> list[dict]:
    """Daily P&L for the last N calendar days.

    Each entry: {date, r_total, wins, losses, trade_count}.
    Days with no trades are included with zeroes for continuity.
    """
    trades = _all_closed_trades()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    daily: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        d = _trade_date(t)
        if d and d >= cutoff:
            daily[d].append(t.get("realized_r", 0))

    # Fill in zero-days for a continuous grid
    start = datetime.now() - timedelta(days=days)
    results: list[dict] = []
    for offset in range(days + 1):
        d = (start + timedelta(days=offset)).strftime("%Y-%m-%d")
        rs = daily.get(d, [])
        results.append({
            "date": d,
            "r_total": _round(sum(rs)),
            "wins": sum(1 for r in rs if r > 0),
            "losses": sum(1 for r in rs if r < 0),
            "trade_count": len(rs),
        })

    return results


def get_weekly_pnl(weeks: int = 12) -> list[dict]:
    """Weekly P&L for the last N weeks.

    Each entry: {week_start (Monday), r_total, wins, losses, trade_count}.
    """
    trades = _all_closed_trades()
    cutoff_dt = datetime.now() - timedelta(weeks=weeks)
    cutoff = cutoff_dt.strftime("%Y-%m-%d")

    weekly: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        d = _trade_date(t)
        if not d or d < cutoff:
            continue
        dt = _parse_dt(t.get("closed_at"))
        if dt is None:
            continue
        # ISO Monday-start week
        monday = dt - timedelta(days=dt.weekday())
        week_key = monday.strftime("%Y-%m-%d")
        weekly[week_key].append(t.get("realized_r", 0))

    # Fill in missing weeks
    results: list[dict] = []
    current_monday = cutoff_dt - timedelta(days=cutoff_dt.weekday())
    now = datetime.now()
    while current_monday <= now:
        wk = current_monday.strftime("%Y-%m-%d")
        rs = weekly.get(wk, [])
        results.append({
            "week_start": wk,
            "r_total": _round(sum(rs)),
            "wins": sum(1 for r in rs if r > 0),
            "losses": sum(1 for r in rs if r < 0),
            "trade_count": len(rs),
        })
        current_monday += timedelta(weeks=1)

    return results


def get_monthly_pnl(months: int = 6) -> list[dict]:
    """Monthly P&L for the last N months.

    Each entry: {month (YYYY-MM), r_total, wins, losses, trade_count}.
    """
    trades = _all_closed_trades()
    cutoff_dt = datetime.now() - timedelta(days=months * 31)
    cutoff = cutoff_dt.strftime("%Y-%m")

    monthly: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        dt = _parse_dt(t.get("closed_at"))
        if dt is None:
            continue
        month_key = dt.strftime("%Y-%m")
        if month_key < cutoff:
            continue
        monthly[month_key].append(t.get("realized_r", 0))

    results: list[dict] = []
    for month_key in sorted(monthly.keys()):
        rs = monthly[month_key]
        results.append({
            "month": month_key,
            "r_total": _round(sum(rs)),
            "wins": sum(1 for r in rs if r > 0),
            "losses": sum(1 for r in rs if r < 0),
            "trade_count": len(rs),
        })

    return results


def get_streaks() -> dict:
    """Current win/loss streak and best/worst streaks ever.

    Returns:
      {current_streak: int (positive=wins, negative=losses),
       best_win_streak: int,
       worst_loss_streak: int,
       current_type: "win" | "loss" | "none"}
    """
    trades = _all_closed_trades()
    if not trades:
        return {
            "current_streak": 0,
            "current_type": "none",
            "best_win_streak": 0,
            "worst_loss_streak": 0,
        }

    best_win = 0
    worst_loss = 0
    current = 0

    for t in trades:
        r = t.get("realized_r", 0)
        if r > 0:
            current = current + 1 if current > 0 else 1
        elif r < 0:
            current = current - 1 if current < 0 else -1
        else:
            # Flat trade (expired/breakeven) — resets streak
            current = 0

        if current > best_win:
            best_win = current
        if current < worst_loss:
            worst_loss = current

    streak_type = "win" if current > 0 else ("loss" if current < 0 else "none")

    return {
        "current_streak": current,
        "current_type": streak_type,
        "best_win_streak": best_win,
        "worst_loss_streak": abs(worst_loss),
    }
