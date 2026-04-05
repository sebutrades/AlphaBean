"""
backend/jobs/live_feed.py — Industry-Grade 5-Minute Live Feed

Four independent jobs run every 5 minutes during market hours:
  1. prices    — Refresh active tracker trade prices
  2. inplay    — Refresh in-play stock list (movers, volume leaders)
  3. regime    — Refresh SPY market regime
  4. scan      — Scan top 15 in-play symbols for setups, cache opportunities

Results are saved to cache/live_feed.json with per-job timestamps.
Each job is fully isolated — failures are logged, never propagate.

Architecture:
  LiveFeed owns the scheduler. main.py calls feed.start() / feed.stop().
  API endpoints call feed.get_status() and feed.get_opportunities().
  No global state — all data flows through the cache file.
"""
import json
import os
import time
import threading
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

FEED_CACHE = Path("cache/live_feed.json")
FEED_CACHE.parent.mkdir(parents=True, exist_ok=True)

_MARKET_OPEN_TZ = "America/New_York"
_CRON = dict(day_of_week="mon-fri", hour="9-15", minute="*/5", timezone=_MARKET_OPEN_TZ)
_lock = threading.Lock()

# Set JUICER_WEEKEND_TEST=1 to run the full live feed on weekends as if market is open.
# Useful for testing the entire pipeline without waiting for Mon–Fri.
WEEKEND_TEST = os.getenv("JUICER_WEEKEND_TEST", "0") == "1"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now_et() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M ET")
    except Exception:
        return datetime.now().strftime("%H:%M")


def _mkt_open() -> bool:
    if WEEKEND_TEST:
        return True
    try:
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
        except ImportError:
            import pytz
            et = pytz.timezone("America/New_York")
        now = datetime.now(et)
        return now.weekday() < 5 and dt_time(9, 30) <= now.time() <= dt_time(16, 0)
    except Exception:
        return False


def _load_feed() -> dict:
    if FEED_CACHE.exists():
        try:
            return json.loads(FEED_CACHE.read_text())
        except Exception:
            pass
    return {
        "started_at": None,
        "update_count": 0,
        "jobs": {},
        "market_open": False,
        "opportunities": [],
        "in_play_count": 0,
        "regime": "unknown",
    }


def _save_feed(data: dict) -> None:
    with _lock:
        tmp = FEED_CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(FEED_CACHE)


def _job_result(status: str, detail: str = "", elapsed: float = 0.0) -> dict:
    return {
        "status": status,
        "detail": detail,
        "elapsed_s": round(elapsed, 2),
        "ran_at": datetime.now().isoformat(),
    }


# ── Jobs ─────────────────────────────────────────────────────────────────────

def _job_prices(feed_ref: dict) -> None:
    """Refresh active tracker trade prices."""
    t = time.time()
    try:
        from backend.tracker.routes import get_tracker
        get_tracker().refresh_prices()
        elapsed = time.time() - t
        feed_ref["jobs"]["prices"] = _job_result("ok", "tracker prices refreshed", elapsed)
        print(f"  [Feed] prices  ok  {elapsed:.1f}s  {_now_et()}")
    except Exception as e:
        feed_ref["jobs"]["prices"] = _job_result("error", str(e)[:120], time.time() - t)
        print(f"  [Feed] prices  ERR  {e}")


def _job_inplay(feed_ref: dict) -> None:
    """Refresh in-play stock list."""
    t = time.time()
    try:
        from backend.ai.inplay_detector import refresh_in_play
        result = refresh_in_play()
        count = len(result.stocks) if hasattr(result, "stocks") else 0
        elapsed = time.time() - t
        feed_ref["jobs"]["inplay"] = _job_result("ok", f"{count} stocks", elapsed)
        feed_ref["in_play_count"] = count
        print(f"  [Feed] inplay  ok  {count} stocks  {elapsed:.1f}s  {_now_et()}")
    except Exception as e:
        feed_ref["jobs"]["inplay"] = _job_result("error", str(e)[:120], time.time() - t)
        print(f"  [Feed] inplay  ERR  {e}")


def _job_regime(feed_ref: dict) -> None:
    """Refresh SPY market regime."""
    t = time.time()
    try:
        from backend.data.massive_client import fetch_bars
        from backend.regime.detector import detect_regime
        import numpy as np
        d = fetch_bars("SPY", "1d", 60)
        c = np.array([b.close for b in d.bars])
        h = np.array([b.high  for b in d.bars])
        l = np.array([b.low   for b in d.bars])
        regime = detect_regime(c, h, l, True).regime.value
        elapsed = time.time() - t
        feed_ref["jobs"]["regime"] = _job_result("ok", regime, elapsed)
        feed_ref["regime"] = regime
        print(f"  [Feed] regime  ok  {regime}  {elapsed:.1f}s  {_now_et()}")
    except Exception as e:
        feed_ref["jobs"]["regime"] = _job_result("error", str(e)[:120], time.time() - t)
        print(f"  [Feed] regime  ERR  {e}")


def _job_scan(feed_ref: dict) -> None:
    """Scan top 15 in-play symbols, cache opportunities with AI verdicts."""
    t = time.time()
    try:
        from backend.ai.inplay_detector import get_in_play
        from backend.scanner.engine import scan_multiple
        from backend.strategies.evaluator import StrategyEvaluator
        from backend.news.pipeline import fetch_news_batch, format_headlines_for_llm
        from backend.ai.ollama_agent import evaluate_setups_batch

        ip = get_in_play()
        symbols = [s.symbol for s in ip.stocks if s.symbol][:15]
        if not symbols:
            feed_ref["jobs"]["scan"] = _job_result("skip", "no in-play symbols", time.time() - t)
            return

        evaluator = StrategyEvaluator()
        evaluator.load()

        setups = scan_multiple(symbols, mode="active", evaluator=evaluator)
        # Keep top 2 per symbol, minimum score 45
        by_sym: dict = {}
        for s in setups:
            sym = s.get("symbol", "")
            by_sym.setdefault(sym, []).append(s)
        capped = []
        for sym, ss in by_sym.items():
            ss.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
            capped.extend([s for s in ss[:2] if s.get("composite_score", 0) >= 45])
        capped.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

        # AI evaluate top half
        if capped:
            top_half = capped[:max(len(capped) // 2, 1)]
            regime = feed_ref.get("regime", "unknown")
            try:
                syms_ai = list({s.get("symbol", "") for s in top_half})
                nb = fetch_news_batch(syms_ai)
                ns = {sym: format_headlines_for_llm(items) for sym, items in nb.items()}
                top_half = evaluate_setups_batch(top_half, ns, regime, top_n=2)
            except Exception as ai_err:
                print(f"  [Feed] scan  AI error: {ai_err}")
            capped = top_half + capped[max(len(capped) // 2, 1):]

        elapsed = time.time() - t
        feed_ref["opportunities"] = capped
        feed_ref["jobs"]["scan"] = _job_result("ok", f"{len(capped)} setups from {len(symbols)} symbols", elapsed)
        print(f"  [Feed] scan    ok  {len(capped)} setups  {elapsed:.1f}s  {_now_et()}")
    except Exception as e:
        feed_ref["jobs"]["scan"] = _job_result("error", str(e)[:120], time.time() - t)
        print(f"  [Feed] scan    ERR  {e}")


# ── LiveFeed ──────────────────────────────────────────────────────────────────

class LiveFeed:
    """
    Encapsulates all scheduled live-feed jobs.
    All 4 jobs share a single feed dict that gets atomically written to disk
    after every cycle. Each job mutates its own slice of that dict independently.
    """

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(
            timezone=_MARKET_OPEN_TZ,
            job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 60},
        )
        self._feed: dict = _load_feed()

    def _run_cycle(self) -> None:
        """Run all 4 jobs in sequence, persist feed after each."""
        self._feed["market_open"] = _mkt_open()
        self._feed["update_count"] = self._feed.get("update_count", 0) + 1

        _job_prices(self._feed)
        _save_feed(self._feed)

        _job_inplay(self._feed)
        _save_feed(self._feed)

        _job_regime(self._feed)
        _save_feed(self._feed)

        _job_scan(self._feed)
        self._feed["last_updated"] = datetime.now().isoformat()
        _save_feed(self._feed)

    def start(self) -> None:
        self._feed["started_at"] = datetime.now().isoformat()
        if WEEKEND_TEST:
            # Fire immediately and then every 5 min — no day/hour restriction
            trigger = IntervalTrigger(minutes=5)
            label = "Weekend Test — every 5 min, no time gate"
        else:
            trigger = CronTrigger(**_CRON)
            label = "5-min cycle market hours | daily scan 4:30 PM ET"

        self._scheduler.add_job(
            self._run_cycle,
            trigger,
            id="live_feed",
            name="Live Feed Cycle",
        )
        if not WEEKEND_TEST:
            self._scheduler.add_job(
                self._daily_scan,
                CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone=_MARKET_OPEN_TZ),
                id="daily_scan",
                name="Daily Scan",
            )
        self._scheduler.start()
        # Fire one cycle immediately so the feed is populated on startup
        threading.Thread(target=self._run_cycle, daemon=True).start()
        _save_feed(self._feed)
        mode = "WEEKEND TEST MODE" if WEEKEND_TEST else "production"
        print(f"  [Feed] Started ({mode}) — {label}")

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        print("  [Feed] Stopped")

    def _daily_scan(self) -> None:
        """Post-close: scan top 50 symbols and add best setups to tracker."""
        t = time.time()
        try:
            from backend.tracker.routes import get_tracker
            added = get_tracker().scan_and_add(top_n=50)
            elapsed = time.time() - t
            self._feed["jobs"]["daily_scan"] = _job_result("ok", f"{added} setups added", elapsed)
            _save_feed(self._feed)
            print(f"  [Feed] daily_scan  ok  {added} setups  {elapsed:.1f}s")
        except Exception as e:
            self._feed["jobs"]["daily_scan"] = _job_result("error", str(e)[:120], time.time() - t)
            _save_feed(self._feed)
            print(f"  [Feed] daily_scan  ERR  {e}")

    def get_status(self) -> dict:
        """Return current feed status for /api/feed/status."""
        data = _load_feed()
        jobs = data.get("jobs", {})
        # Compute staleness for each job
        stale = {}
        for jname, jdata in jobs.items():
            ran_at = jdata.get("ran_at")
            if ran_at:
                age_s = (datetime.now() - datetime.fromisoformat(ran_at)).total_seconds()
                stale[jname] = round(age_s)
        return {
            "started_at":    data.get("started_at"),
            "last_updated":  data.get("last_updated"),
            "update_count":  data.get("update_count", 0),
            "market_open":   data.get("market_open", False),
            "regime":        data.get("regime", "unknown"),
            "in_play_count": data.get("in_play_count", 0),
            "opportunity_count": len(data.get("opportunities", [])),
            "jobs":          jobs,
            "staleness_s":   stale,
            "scheduler_running": self._scheduler.running,
        }

    def get_opportunities(self) -> list:
        """Return cached opportunities from last scan cycle."""
        return _load_feed().get("opportunities", [])

    def trigger_now(self) -> str:
        """Manually trigger a feed cycle (for /api/feed/refresh endpoint)."""
        threading.Thread(target=self._run_cycle, daemon=True).start()
        return "cycle triggered"
