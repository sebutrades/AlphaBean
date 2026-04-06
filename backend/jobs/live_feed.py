"""
backend/jobs/live_feed.py — Industry-Grade 5-Minute Live Feed  (v2)

Every 5-minute cycle (market hours):
  1. bar_update_hot   — Fetch new 5-min bars for hot list (in-play + active trades)
  2. prices           — Refresh active tracker trade statuses (now uses bar store)
  3. inplay           — Refresh in-play stock list
  4. regime           — Refresh SPY market regime (uses bar store)
  5. scan             — Scan top 50 in-play symbols using bar-store data (fast)
  6. intraday_track   — Run intraday setup tracker on updated bars

Every 15-minute cycle (every 3rd 5-min cycle):
  - bar_update_15min_hot  — Fetch new 15-min bars for hot list

Every 30-minute cycle (every 6th 5-min cycle):
  - bar_update_5min_full  — Update 5-min bars for full 500-symbol universe

Every 60-minute cycle (every 12th 5-min cycle):
  - bar_update_15min_full — Update 15-min bars for full universe
  - bar_update_1h         — Update 1-h bars for full universe

At 4:30 PM ET (daily):
  - bar_update_1d         — Update daily bars for full universe
  - overnight_flag        — Flag open intraday setups as overnight-hold risk
  - daily_summary         — Generate & save today's intraday performance report
  - daily_scan            — Walk-forward scan for daily-timeframe tracker setups

Results are saved to cache/live_feed.json with per-job timestamps.
Each job is fully isolated — failures are logged, never propagate.
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
WEEKEND_TEST = os.getenv("JUICER_WEEKEND_TEST", "0") == "1"


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        "intraday_open_count": 0,
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


# ── Bar update jobs ───────────────────────────────────────────────────────────

def _job_update_bars(
    tf: str,
    feed_ref: dict,
    symbols: Optional[list[str]] = None,
    label: str = "",
) -> None:
    """Fetch new bars for the given timeframe and symbol list."""
    t = time.time()
    job_key = f"bar_update_{label or tf}"
    try:
        from live_data_cache.bar_updater import update_timeframe
        result  = update_timeframe(tf, symbols=symbols)
        elapsed = time.time() - t
        detail  = (
            f"{result['symbols_updated']} symbols, "
            f"+{result['total_added']} bars, "
            f"{result['errors']} errors"
        )
        feed_ref["jobs"][job_key] = _job_result("ok", detail, elapsed)
        print(f"  [Feed] {job_key:<22}  ok  {detail}  {elapsed:.1f}s  {_now_et()}")
    except Exception as e:
        feed_ref["jobs"][job_key] = _job_result("error", str(e)[:120], time.time() - t)
        print(f"  [Feed] {job_key}  ERR  {e}")


# ── Existing jobs (updated to use bar store) ──────────────────────────────────

def _job_prices(feed_ref: dict) -> None:
    """Refresh active tracker trade prices — now reads from bar store first."""
    t = time.time()
    try:
        from backend.tracker.routes import get_tracker
        get_tracker().refresh_prices()
        elapsed = time.time() - t
        feed_ref["jobs"]["prices"] = _job_result("ok", "tracker prices refreshed", elapsed)
        print(f"  [Feed] {'prices':<22}  ok  {elapsed:.1f}s  {_now_et()}")
    except Exception as e:
        feed_ref["jobs"]["prices"] = _job_result("error", str(e)[:120], time.time() - t)
        print(f"  [Feed] prices  ERR  {e}")


def _job_inplay(feed_ref: dict) -> None:
    """Refresh in-play stock list."""
    t = time.time()
    try:
        from backend.ai.inplay_detector import refresh_in_play
        result = refresh_in_play()
        count  = len(result.stocks) if hasattr(result, "stocks") else 0
        elapsed = time.time() - t
        feed_ref["jobs"]["inplay"]  = _job_result("ok", f"{count} stocks", elapsed)
        feed_ref["in_play_count"]   = count
        print(f"  [Feed] {'inplay':<22}  ok  {count} stocks  {elapsed:.1f}s  {_now_et()}")
    except Exception as e:
        feed_ref["jobs"]["inplay"] = _job_result("error", str(e)[:120], time.time() - t)
        print(f"  [Feed] inplay  ERR  {e}")


def _job_regime(feed_ref: dict) -> None:
    """Refresh SPY market regime — prefers bar store for SPY daily bars."""
    t = time.time()
    try:
        from backend.regime.detector import detect_regime
        import numpy as np

        # Try bar store first (fast, no API call)
        spy_bars = None
        try:
            from live_data_cache.bar_store import get_bars
            spy_series = get_bars("SPY", "1d")
            if spy_series and len(spy_series.bars) >= 20:
                spy_bars = spy_series
        except Exception:
            pass

        if spy_bars is None:
            from backend.data.massive_client import fetch_bars
            spy_bars = fetch_bars("SPY", "1d", 60)

        c = np.array([b.close for b in spy_bars.bars])
        h = np.array([b.high  for b in spy_bars.bars])
        l = np.array([b.low   for b in spy_bars.bars])
        regime  = detect_regime(c, h, l, True).regime.value
        elapsed = time.time() - t
        feed_ref["jobs"]["regime"] = _job_result("ok", regime, elapsed)
        feed_ref["regime"] = regime
        print(f"  [Feed] {'regime':<22}  ok  {regime}  {elapsed:.1f}s  {_now_et()}")
    except Exception as e:
        feed_ref["jobs"]["regime"] = _job_result("error", str(e)[:120], time.time() - t)
        print(f"  [Feed] regime  ERR  {e}")


def _job_scan(feed_ref: dict) -> None:
    """
    Scan top 50 in-play symbols using bar-store data (no API calls during scan).
    Because bars are pre-loaded, this is ~4× faster than the old approach.
    """
    t = time.time()
    try:
        from backend.ai.inplay_detector import get_in_play
        from backend.scanner.engine import scan_multiple
        from backend.strategies.evaluator import StrategyEvaluator
        from backend.news.pipeline import fetch_news_batch, format_headlines_for_llm
        from backend.ai.ollama_agent import evaluate_setups_batch

        ip      = get_in_play()
        # Expanded from 15 → 50 because bar store makes scans fast
        symbols = [s.symbol for s in ip.stocks if s.symbol][:50]
        if not symbols:
            feed_ref["jobs"]["scan"] = _job_result("skip", "no in-play symbols", time.time() - t)
            return

        evaluator = StrategyEvaluator()
        evaluator.load()

        setups = scan_multiple(symbols, mode="active", evaluator=evaluator)

        # Keep top 2 per symbol — score threshold raised to 50 to match the
        # recalibrated neutral baseline (v2 defaults give ~55 average vs ~46 before).
        by_sym: dict = {}
        for s in setups:
            by_sym.setdefault(s.get("symbol", ""), []).append(s)
        capped = []
        for sym, ss in by_sym.items():
            ss.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
            capped.extend([s for s in ss[:2] if s.get("composite_score", 0) >= 45])
        capped.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

        # AI-evaluate ALL setups that cleared the threshold.
        # Previously: top-half only — meant genuinely good setups ranked below
        # the median never received a verdict, even if the AI would confirm them.
        # Now: every setup in `capped` gets evaluated; the AI delta re-ranks them.
        # The per-day cache (symbol × pattern) keeps this cheap after the first hit.
        if capped:
            regime = feed_ref.get("regime", "unknown")
            try:
                syms_ai = list({s.get("symbol", "") for s in capped})
                nb      = fetch_news_batch(syms_ai)
                ns      = {sym: format_headlines_for_llm(items) for sym, items in nb.items()}
                capped  = evaluate_setups_batch(capped, ns, regime, top_n=3)
            except Exception as ai_err:
                print(f"  [Feed] scan  AI error: {ai_err}")

        elapsed = time.time() - t
        feed_ref["opportunities"] = capped
        feed_ref["jobs"]["scan"]  = _job_result(
            "ok", f"{len(capped)} setups from {len(symbols)} symbols", elapsed
        )
        print(f"  [Feed] {'scan':<22}  ok  {len(capped)} setups  {elapsed:.1f}s  {_now_et()}")
    except Exception as e:
        feed_ref["jobs"]["scan"] = _job_result("error", str(e)[:120], time.time() - t)
        print(f"  [Feed] scan  ERR  {e}")


def _job_intraday_track(feed_ref: dict) -> None:
    """
    Run the intraday setup tracker over the most recently updated bars.

    For each in-play symbol on 5-min and 15-min timeframes:
      - Detects newly formed setups (entry within 2.5 % of current price)
      - Advances all open setups through the latest bars (stop / target detection)
      - Archives resolved setups
    """
    t = time.time()
    try:
        from live_data_cache.intraday_setup_tracker import process_new_bars, get_open_setups
        from live_data_cache.bar_store import get_bars
        from live_data_cache.watchlist import get_hot_list

        hot_symbols = get_hot_list()[:50]
        new_total   = 0

        for symbol in hot_symbols:
            for tf in ("5min", "15min"):
                bars = get_bars(symbol, tf)
                if bars and bars.bars:
                    # Pass the last 5 bars as the "new context" window.
                    # process_new_bars uses last_bar_ts to avoid re-processing old bars.
                    context_bars = bars.bars[-5:]
                    detected = process_new_bars(symbol, tf, context_bars)
                    new_total += len(detected)

        open_count = len(get_open_setups())
        elapsed    = time.time() - t
        feed_ref["jobs"]["intraday_track"]   = _job_result(
            "ok", f"{new_total} new setups, {open_count} open", elapsed
        )
        feed_ref["intraday_open_count"] = open_count
        print(f"  [Feed] {'intraday_track':<22}  ok  "
              f"{new_total} new, {open_count} open  {elapsed:.1f}s  {_now_et()}")
    except Exception as e:
        feed_ref["jobs"]["intraday_track"] = _job_result("error", str(e)[:120], time.time() - t)
        print(f"  [Feed] intraday_track  ERR  {e}")


# ── LiveFeed ──────────────────────────────────────────────────────────────────

class LiveFeed:
    """
    Encapsulates all scheduled live-feed jobs.
    All jobs share a single feed dict that is atomically written to disk
    after every cycle.  Each job mutates its own slice independently.

    Cycle counter drives multi-frequency bar updates:
      Every cycle  (5 min):  5-min hot-list update
      Every 3rd    (15 min): 15-min hot-list update
      Every 6th    (30 min): 5-min full-universe update
      Every 12th   (60 min): 15-min + 1-h full-universe update
    """

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(
            timezone=_MARKET_OPEN_TZ,
            job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 60},
        )
        self._feed: dict  = _load_feed()
        self._cycle_count = 0

    def _run_cycle(self) -> None:
        """Run the full cycle in the correct order, persist after each job."""
        self._cycle_count += 1
        n = self._cycle_count

        self._feed["market_open"]  = _mkt_open()
        self._feed["update_count"] = self._feed.get("update_count", 0) + 1

        # ── Step 1: Update bars FIRST so the scan uses fresh data ─────────────
        # Hot list 5-min bars — every cycle
        from live_data_cache.watchlist import get_hot_list
        hot = get_hot_list()
        _job_update_bars("5min", self._feed, symbols=hot, label="5min_hot")
        _save_feed(self._feed)

        # Hot list 15-min bars — every 3 cycles (15 min)
        if n % 3 == 0:
            _job_update_bars("15min", self._feed, symbols=hot, label="15min_hot")
            _save_feed(self._feed)

        # Full universe 5-min bars — every 6 cycles (30 min)
        if n % 6 == 0:
            threading.Thread(
                target=_job_update_bars,
                args=("5min", self._feed),
                kwargs={"label": "5min_full"},
                daemon=True,
            ).start()

        # Full universe 15-min + 1-h bars — every 12 cycles (60 min)
        if n % 12 == 0:
            threading.Thread(
                target=_job_update_bars,
                args=("15min", self._feed),
                kwargs={"label": "15min_full"},
                daemon=True,
            ).start()
            threading.Thread(
                target=_job_update_bars,
                args=("1h", self._feed),
                kwargs={"label": "1h_full"},
                daemon=True,
            ).start()

        # ── Step 2: Core jobs ─────────────────────────────────────────────────
        _job_prices(self._feed)
        _save_feed(self._feed)

        _job_inplay(self._feed)
        _save_feed(self._feed)

        _job_regime(self._feed)
        _save_feed(self._feed)

        _job_scan(self._feed)
        _save_feed(self._feed)

        # ── Step 3: Intraday setup lifecycle ──────────────────────────────────
        _job_intraday_track(self._feed)
        self._feed["last_updated"] = datetime.now().isoformat()
        _save_feed(self._feed)

    def start(self) -> None:
        self._feed["started_at"] = datetime.now().isoformat()

        if WEEKEND_TEST:
            trigger = IntervalTrigger(minutes=5)
            label   = "Weekend Test — every 5 min, no time gate"
        else:
            trigger = CronTrigger(**_CRON)
            label   = "5-min cycle market hours | daily jobs 4:30 PM ET"

        self._scheduler.add_job(
            self._run_cycle, trigger,
            id="live_feed", name="Live Feed Cycle",
        )

        if not WEEKEND_TEST:
            self._scheduler.add_job(
                self._daily_close,
                CronTrigger(
                    day_of_week="mon-fri", hour=16, minute=30,
                    timezone=_MARKET_OPEN_TZ,
                ),
                id="daily_close", name="Daily Close Jobs",
            )

        self._scheduler.start()

        # Fire one cycle immediately so the feed is populated on startup.
        # Also kick off a one-time backfill of the hot list in the background.
        threading.Thread(target=self._startup_sequence, daemon=True).start()
        _save_feed(self._feed)

        mode = "WEEKEND TEST MODE" if WEEKEND_TEST else "production"
        print(f"  [Feed] Started ({mode}) — {label}")

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        print("  [Feed] Stopped")

    def _startup_sequence(self) -> None:
        """
        Called once on startup (background thread).
        1. Ensure hot-list symbols have bar data (backfill if missing).
        2. Fire the first scan cycle.
        """
        try:
            print("  [Feed] Startup: ensuring hot-list bars are ready...")
            from live_data_cache.bar_updater import ensure_hot_list_ready
            result = ensure_hot_list_ready(timeframes=["5min", "15min", "1h", "1d"])
            print(f"  [Feed] Startup backfill done — "
                  f"{result['backfilled']} symbols filled, "
                  f"{result['already_cached']} already cached")
        except Exception as e:
            print(f"  [Feed] Startup backfill error: {e}")

        # First scan cycle
        self._run_cycle()

    def _daily_close(self) -> None:
        """
        Post-close jobs (4:30 PM ET):
          1. Update daily bars for full universe
          2. Flag overnight holds
          3. Generate today's intraday performance summary
          4. Walk-forward scan for daily-TF tracker setups
        """
        t = time.time()

        # 1. Daily bar update
        _job_update_bars("1d", self._feed, label="1d_full")
        _save_feed(self._feed)

        # 2. Flag overnight holds
        try:
            from live_data_cache.intraday_setup_tracker import flag_overnight_holds
            n_flagged = flag_overnight_holds()
            self._feed["jobs"]["overnight_flag"] = _job_result(
                "ok", f"{n_flagged} setups flagged", time.time() - t
            )
            print(f"  [Feed] overnight_flag  ok  {n_flagged} flagged")
        except Exception as e:
            self._feed["jobs"]["overnight_flag"] = _job_result("error", str(e)[:120])
            print(f"  [Feed] overnight_flag  ERR  {e}")
        _save_feed(self._feed)

        # 3. Daily performance summary
        try:
            from live_data_cache.intraday_setup_tracker import generate_daily_summary
            summary = generate_daily_summary()
            self._feed["jobs"]["daily_summary"] = _job_result(
                "ok",
                f"{summary['total_setups']} setups  "
                f"W:{summary['winners']} L:{summary['losers']}  "
                f"{summary['total_realized_r']:+.2f}R",
                time.time() - t,
            )
            self._feed["daily_summary"] = summary
            print(f"  [Feed] daily_summary  ok  "
                  f"{summary['total_setups']} setups  "
                  f"{summary['total_realized_r']:+.2f}R")
        except Exception as e:
            self._feed["jobs"]["daily_summary"] = _job_result("error", str(e)[:120])
            print(f"  [Feed] daily_summary  ERR  {e}")
        _save_feed(self._feed)

        # 4. EOD intraday close — shut 5min/15min trades, flag gap-risk holds
        try:
            from backend.tracker.routes import get_tracker
            eod = get_tracker().end_of_day_intraday_close()
            self._feed["jobs"]["intraday_eod"] = _job_result(
                "ok",
                f"{eod['closed']} closed, {len(eod['gap_risk'])} gap-risk",
                time.time() - t,
            )
            print(f"  [Feed] intraday_eod  ok  "
                  f"{eod['closed']} closed, gap-risk: {eod['gap_risk']}")
        except Exception as e:
            self._feed["jobs"]["intraday_eod"] = _job_result("error", str(e)[:120])
            print(f"  [Feed] intraday_eod  ERR  {e}")
        _save_feed(self._feed)

        # 5. Daily tracker scan (existing behaviour)
        try:
            from backend.tracker.routes import get_tracker
            added = get_tracker().scan_and_add(top_n=50)
            elapsed = time.time() - t
            self._feed["jobs"]["daily_scan"] = _job_result(
                "ok", f"{added} setups added", elapsed
            )
            _save_feed(self._feed)
            print(f"  [Feed] daily_scan  ok  {added} setups  {elapsed:.1f}s")
        except Exception as e:
            self._feed["jobs"]["daily_scan"] = _job_result("error", str(e)[:120], time.time() - t)
            _save_feed(self._feed)
            print(f"  [Feed] daily_scan  ERR  {e}")

    def get_status(self) -> dict:
        """Return current feed status for /api/feed/status."""
        data  = _load_feed()
        jobs  = data.get("jobs", {})
        stale = {}
        for jname, jdata in jobs.items():
            ran_at = jdata.get("ran_at")
            if ran_at:
                try:
                    age_s = (datetime.now() - datetime.fromisoformat(ran_at)).total_seconds()
                    stale[jname] = round(age_s)
                except Exception:
                    pass
        return {
            "started_at":          data.get("started_at"),
            "last_updated":        data.get("last_updated"),
            "update_count":        data.get("update_count", 0),
            "market_open":         data.get("market_open", False),
            "regime":              data.get("regime", "unknown"),
            "in_play_count":       data.get("in_play_count", 0),
            "opportunity_count":   len(data.get("opportunities", [])),
            "intraday_open_count": data.get("intraday_open_count", 0),
            "jobs":                jobs,
            "staleness_s":         stale,
            "scheduler_running":   self._scheduler.running,
        }

    def get_opportunities(self) -> list:
        """Return cached opportunities from last scan cycle."""
        return _load_feed().get("opportunities", [])

    def trigger_now(self) -> str:
        """Manually trigger a feed cycle (for /api/feed/refresh endpoint)."""
        threading.Thread(target=self._run_cycle, daemon=True).start()
        return "cycle triggered"
