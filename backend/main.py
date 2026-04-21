"""
main.py — Juicer API Server v1.0

Start: uvicorn backend.main:app --reload --port 8000
"""
from dotenv import load_dotenv
load_dotenv()  # Load .env file (ANTHROPIC_API_KEY, etc.)

import json, os, time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, time as dt_time
from pathlib import Path
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from backend.scanner.engine import scan_symbol, scan_multiple
from backend.patterns.registry import get_all_pattern_names, PATTERN_META
from backend.data.massive_client import fetch_chart_bars, fetch_bars
from backend.regime.detector import detect_regime
from backend.strategies.evaluator import StrategyEvaluator
from backend.news.pipeline import fetch_news, fetch_market_news, format_headlines_for_llm, fetch_news_batch
from backend.ai.ollama_agent import evaluate_setups_batch, check_ollama_status, test_agent
from backend.ai.inplay_detector import get_in_play, refresh_in_play
from backend.features.correlation import compute_correlation_live
from backend.analytics.symbol_stats import get_symbol_analytics
import numpy as np


# ═══ LIVE FEED ═══

from backend.jobs.live_feed import LiveFeed

_feed = LiveFeed()

@asynccontextmanager
async def lifespan(app: FastAPI):
    _feed.start()
    yield
    _feed.stop()

app = FastAPI(title="Juicer API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
from backend.tracker.routes import router as tracker_router
app.include_router(tracker_router, prefix="/api/tracker")

_evaluator = StrategyEvaluator()
_evaluator.load()
BT = Path("cache/backtest_results.json")

_WEEKEND_TEST = os.getenv("JUICER_WEEKEND_TEST", "0") == "1"

def _mkt_open() -> bool:
    if _WEEKEND_TEST:
        return True
    try:
        from zoneinfo import ZoneInfo; et = ZoneInfo("America/New_York")
    except ImportError:
        import pytz; et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5: return False
    return dt_time(9,30) <= now.time() <= dt_time(16,0)

_regime_cache: dict = {"value": "unknown", "ts": 0.0}

def _regime_str() -> str:
    """Cached SPY regime — recomputes at most once every 5 minutes."""
    if time.time() - _regime_cache["ts"] < 300:
        return _regime_cache["value"]
    try:
        d = fetch_bars("SPY","1d",60)
        c = np.array([b.close for b in d.bars])
        h = np.array([b.high  for b in d.bars])
        l = np.array([b.low   for b in d.bars])
        val = detect_regime(c, h, l, True).regime.value
        _regime_cache["value"] = val
        _regime_cache["ts"]    = time.time()
        return val
    except Exception as e:
        print(f"  [Regime] Error: {e}")
        return _regime_cache["value"]

def _mkt_hours_filter(setups):
    out = []
    for s in setups:
        try:
            dt = datetime.fromisoformat(s.get("detected_at",""))
            if (dt.hour > 9 or (dt.hour == 9 and dt.minute >= 30)) and dt.hour < 16: out.append(s)
            elif dt.hour == 16 and dt.minute == 0: out.append(s)
        except Exception:
            out.append(s)
    return out


# ═══ CORE ═══

@app.get("/api/health")
async def health():
    return {"status":"ok","app":"Juicer","version":"1.0.0","patterns":len(get_all_pattern_names()),
            "strategies_tracked":_evaluator.stats_summary()["strategies_tracked"],
            "ollama":check_ollama_status(),"market_open":_mkt_open()}

@app.get("/api/patterns")
async def list_patterns():
    p = [{"name":n,"category":PATTERN_META.get(n,{}).get("cat","").value if hasattr(PATTERN_META.get(n,{}).get("cat",""),"value") else "","strategy_type":PATTERN_META.get(n,{}).get("type",""),"win_rate":PATTERN_META.get(n,{}).get("wr",0)} for n in get_all_pattern_names()]
    return {"count":len(p),"patterns":p}


# ═══ SCANNER ═══

@app.get("/api/scan")
async def scan(symbol:str=Query(...),mode:str=Query("today"),ai:bool=Query(True)):
    """
    Scan a single symbol.  Merges:
      1. Intraday tracker open setups for this symbol (accumulated all day via rolling window)
      2. Fresh current-state scan (catches setups forming right now)
    """
    if mode not in ("today","active"): return {"error":"mode must be 'today' or 'active'"}
    sym = symbol.upper()

    # ── Source 1: intraday tracker accumulated setups for this symbol ──────────
    try:
        from live_data_cache.intraday_setup_tracker import get_open_setups
        from backend.scanner.engine import scan_symbol as _scan_sym
        from backend.scoring.multi_factor import score_setup, ScoredSetup
        from backend.features.engine import compute_features
        from backend.regime.detector import detect_regime
        from live_data_cache.bar_store import get_bars as _get_cached_bars
        import numpy as np

        tracked = get_open_setups(symbol=sym)
        # Convert tracker dicts to scored-setup dicts (they lack scoring breakdown)
        # Re-score them using current features so scoring is fresh
        tracker_setups = []
        if tracked:
            for tf in ("5min", "15min"):
                b = _get_cached_bars(sym, tf)
                if b and len(b.bars) >= 20:
                    c = np.array([x.close for x in b.bars])
                    h = np.array([x.high  for x in b.bars])
                    l = np.array([x.low   for x in b.bars])
                    v = np.array([x.volume for x in b.bars])
                    feats  = compute_features(c, h, l, v)
                    regime = detect_regime(c, h, l, is_spy=False)
                    for s in tracked:
                        if s.get("timeframe") == tf:
                            # Embed tracker status into a setup-like dict
                            sd = dict(s)
                            sd.setdefault("composite_score", s.get("confidence", 0.5) * 80)
                            sd["tracker_status"] = s.get("status","")
                            tracker_setups.append(sd)
    except Exception as e:
        print(f"  [Scan] tracker merge error: {e}")
        tracker_setups = []

    # ── Source 2: fresh current-state scan ────────────────────────────────────
    fresh_setups = scan_symbol(sym, mode=mode, evaluator=_evaluator)
    if mode == "today" and _mkt_open(): fresh_setups = _mkt_hours_filter(fresh_setups)
    fresh_setups = [s for s in fresh_setups if s.get("composite_score", 0) >= 40]

    # Merge: tracker setups first (they have history), then fresh (current state)
    seen = {(s.get("pattern_name",""), s.get("bias","")) for s in tracker_setups}
    for s in fresh_setups:
        if (s.get("pattern_name",""), s.get("bias","")) not in seen:
            tracker_setups.append(s)
            seen.add((s.get("pattern_name",""), s.get("bias","")))

    setups = sorted(tracker_setups, key=lambda x: x.get("composite_score",0), reverse=True)

    # AI evaluation on fresh setups only (tracker setups already evaluated historically)
    if ai and fresh_setups:
        try:
            news = fetch_news(sym, 15); hl = format_headlines_for_llm(news)
            fresh_setups = evaluate_setups_batch(fresh_setups, {sym: hl}, _regime_str(), top_n=5)
        except Exception as e: print(f"  AI error: {e}")

    # Live correlation
    try:
        corr = compute_correlation_live(sym).to_dict()
        for s in setups: s["spy_correlation"] = corr
    except Exception as e:
        print(f"  [Correlation] {sym}: {e}")

    return {"symbol":sym,"mode":mode,"count":len(setups),"setups":setups,"market_open":_mkt_open()}

@app.get("/api/scan-multiple")
async def scan_multi(symbols:str=Query(...),mode:str=Query("today"),ai:bool=Query(True)):
    sl = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    setups = scan_multiple(sl,mode=mode,evaluator=_evaluator)
    if mode == "today" and _mkt_open(): setups = _mkt_hours_filter(setups)
    if ai and setups:
        try:
            nb = fetch_news_batch(sl); ns = {sym:format_headlines_for_llm(items) for sym,items in nb.items()}
            setups = evaluate_setups_batch(setups,ns,_regime_str(),top_n=5)
        except Exception as e: print(f"  AI error: {e}")
    setups = [s for s in setups if s.get("composite_score", 0) >= 40]
    return {"symbols":sl,"mode":mode,"count":len(setups),"setups":setups}


# ═══ IN-PLAY & TOP OPPORTUNITIES ═══

@app.get("/api/in-play")
async def in_play():
    return get_in_play().to_dict()

@app.get("/api/in-play/refresh")
async def in_play_refresh():
    return refresh_in_play().to_dict()

@app.get("/api/top-opportunities")
async def top_opportunities(ai:bool=Query(True),per_symbol:int=Query(2),max_symbols:int=Query(50)):
    """
    Returns the live feed's pre-computed opportunities (updated every 5 min by the
    background job) merged with any current-scan hits.  The live feed already runs
    scan_multiple across all 50 in-play symbols, scores them, and caches the result —
    so this endpoint is near-instant and always shows the freshest data.
    """
    t = time.time()
    ip = get_in_play()

    # ── Primary source: live feed cache (updated every 5 min) ─────────────────
    cached_opps = _feed.get_opportunities()

    # ── Fallback: run a fresh scan if cache is empty or very stale ────────────
    if not cached_opps:
        symbols = [s.symbol for s in ip.stocks if s.symbol][:max_symbols]
        if symbols:
            all_setups = scan_multiple(symbols, mode="active", evaluator=_evaluator)
            by_sym: dict[str, list] = defaultdict(list)
            for s in all_setups: by_sym[s.get("symbol","")].append(s)
            for sym, ss in by_sym.items():
                ss.sort(key=lambda x: x.get("composite_score",0), reverse=True)
                cached_opps.extend(ss[:per_symbol])
            cached_opps.sort(key=lambda x: x.get("composite_score",0), reverse=True)
            cached_opps = [s for s in cached_opps if s.get("composite_score", 0) >= 45]

    capped = cached_opps

    # Live correlation per symbol
    corr_map: dict[str, dict] = {}
    for s in capped:
        sym = s.get("symbol","")
        if sym and sym not in corr_map:
            try: corr_map[sym] = compute_correlation_live(sym).to_dict()
            except Exception: corr_map[sym] = {}
        s["spy_correlation"] = corr_map.get(sym, {})

    # Enrich with in-play info
    ip_map = {s.symbol:s.to_dict() for s in ip.stocks}
    for s in capped:
        sym = s.get("symbol","")
        if sym in ip_map: s["in_play_info"] = ip_map[sym]

    elapsed = time.time() - t
    return {"market_summary":ip.market_summary,"in_play":ip.to_dict(),"setups":capped,
            "count":len(capped),"elapsed_seconds":round(elapsed,1),"market_open":_mkt_open()}


# ═══ TRACKING ═══

@app.get("/api/track-prices")
async def track_prices(symbols:str=Query(...)):
    r = {}
    for sym in symbols.split(","):
        sym=sym.strip().upper()
        if not sym: continue
        try:
            bars=fetch_bars(sym,"5min",1)
            if bars.bars:
                l=bars.bars[-1]; r[sym]={"price":l.close,"high":l.high,"low":l.low,"volume":l.volume,"timestamp":l.timestamp.isoformat()}
        except Exception as e: r[sym]={"error":str(e)}
    return {"prices":r,"market_open":_mkt_open()}


# ═══ CORRELATION ═══

@app.get("/api/correlation/{symbol}")
async def get_correlation(symbol:str):
    try: return compute_correlation_live(symbol.upper()).to_dict()
    except Exception as e: return {"error":str(e)}


# ═══ BACKTEST ═══

@app.get("/api/backtest/results")
async def backtest_results():
    if not BT.exists(): return {"has_results":False,"message":"Run: python run_backtest.py"}
    try: return {"has_results":True,**json.loads(BT.read_text())}
    except: return {"has_results":False,"message":"Cache corrupt"}

@app.get("/api/backtest/patterns")
async def backtest_patterns_sorted(sort:str=Query("edge_score")):
    if not BT.exists(): return {"patterns":[]}
    try:
        data=json.loads(BT.read_text()); patterns=data.get("patterns",{})
        items=[{"name":k,**v} for k,v in patterns.items() if v.get("total_signals",0)>=3]
        valid=["edge_score","win_rate","profit_factor","expectancy","total_signals"]
        if sort not in valid: sort="edge_score"
        items.sort(key=lambda x:x.get(sort,0),reverse=True)
        return {"patterns":items,"count":len(items),"sort":sort,"summary":data.get("summary",{}),"config":data.get("config",{})}
    except: return {"patterns":[]}


# ═══ NEWS / CHART / REGIME / STRATEGIES ═══

@app.get("/api/news/{symbol}")
async def get_news(symbol:str):
    items=fetch_news(symbol.upper(),20); return {"symbol":symbol.upper(),"count":len(items),"headlines":[i.to_dict() for i in items]}

@app.get("/api/news-market")
async def get_market_news():
    items=fetch_market_news(30); return {"count":len(items),"headlines":[i.to_dict() for i in items]}

@app.get("/api/chart/{symbol}")
async def chart_data(symbol:str,timeframe:str=Query("5min"),days_back:int=Query(5)):
    if timeframe not in ("5min","15min","1h","1d"): return {"error":f"Invalid: {timeframe}"}
    try: return {"symbol":symbol.upper(),"timeframe":timeframe,"bars":fetch_chart_bars(symbol.upper(),timeframe,days_back)}
    except Exception as e: return {"error":str(e)}

@app.get("/api/regime")
async def get_regime():
    try:
        d=fetch_bars("SPY","1d",250); c=np.array([b.close for b in d.bars]); h=np.array([b.high for b in d.bars]); l=np.array([b.low for b in d.bars])
        result = detect_regime(c,h,l,True).to_dict()
        # Update the shared regime cache so _regime_str() stays in sync
        _regime_cache["value"] = result.get("regime", "unknown")
        _regime_cache["ts"] = time.time()
        return result
    except Exception as e: return {"error":str(e),"regime":"unknown"}

@app.get("/api/hot-strategies")
async def hot_strategies(top_n:int=Query(5)):
    hot=_evaluator.get_hot_strategies(top_n); return {"count":len(hot),"strategies":[m.to_dict() for m in hot],"hot_types":_evaluator.get_hot_strategy_types(3)}

@app.get("/api/strategy/{pattern_name}")
async def strategy_detail(pattern_name:str): return _evaluator.get_pattern_summary(pattern_name)

@app.get("/api/ollama/status")
async def ollama_status(): return check_ollama_status()

@app.get("/api/ai/test/{symbol}")
async def ai_test(symbol: str):
    """Run the full AI pipeline on a synthetic setup for a symbol — verifies model quality."""
    regime = _regime_str()
    return test_agent(symbol.upper(), regime)

@app.post("/api/reload-evaluator")
async def reload_evaluator(): _evaluator.load(); return {"status":"reloaded",**_evaluator.stats_summary()}

# ═══ SYMBOL SPECIFIC ANALYTICS ═══
@app.get("/api/analytics/{symbol}")
async def symbol_analytics(symbol: str, pattern: str = Query("")):
    """Per-symbol pattern analytics. Shows how patterns perform on this specific stock."""
    return get_symbol_analytics(symbol.upper(), pattern)


# ═══ FEED ═══

@app.get("/api/feed/status")
async def feed_status():
    """Live feed health — job timestamps, staleness, regime, counts."""
    return _feed.get_status()

@app.get("/api/feed/opportunities")
async def feed_opportunities():
    """Cached opportunities from the last scan cycle."""
    return {"opportunities": _feed.get_opportunities()}

@app.get("/api/feed/refresh")
async def feed_refresh():
    """Manually trigger an immediate feed cycle."""
    msg = _feed.trigger_now()
    return {"status": msg}


# ── Live Data Cache endpoints ─────────────────────────────────────────────────

@app.get("/api/cache/status")
async def cache_status():
    """Bar store health — symbols cached, bar counts, oldest/newest per timeframe."""
    try:
        from live_data_cache.bar_store import get_store_stats
        return get_store_stats()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/intraday/open")
async def intraday_open(symbol: str = None, tf: str = None):
    """
    Currently open intraday setups (5-min and 15-min).
    Optional filters: ?symbol=NVDA  and/or  ?tf=5min
    """
    try:
        from live_data_cache.intraday_setup_tracker import get_open_setups
        setups = get_open_setups(symbol=symbol, tf=tf)
        setups.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        return {"count": len(setups), "setups": setups}
    except Exception as e:
        return {"error": str(e), "count": 0, "setups": []}


@app.get("/api/intraday/closed")
async def intraday_closed(date: str = None):
    """
    Closed intraday setups for a given date (ISO: 2026-04-06) or today.
    """
    try:
        from live_data_cache.intraday_setup_tracker import get_closed_setups
        setups = get_closed_setups(date)
        setups.sort(key=lambda s: s.get("closed_at", ""), reverse=True)
        return {"date": date, "count": len(setups), "setups": setups}
    except Exception as e:
        return {"error": str(e), "count": 0, "setups": []}


@app.get("/api/intraday/summary")
async def intraday_summary():
    """Today's intraday performance summary (wins, losses, R-multiples)."""
    try:
        from live_data_cache.intraday_setup_tracker import generate_daily_summary
        return generate_daily_summary()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/intraday/daily-perf")
async def intraday_daily_perf():
    """Rolling daily performance log (all days stored in daily_perf.json)."""
    from pathlib import Path
    import json
    p = Path("live_data_cache/daily_perf.json")
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception as e:
            return {"error": str(e)}
    return {"days": {}}


@app.get("/api/cache/bars/{symbol}")
async def cache_bars(symbol: str, tf: str = "5min"):
    """
    Return the stored bars for a symbol/timeframe from the live_data_cache.
    Useful for debugging or feeding custom chart views.
    """
    try:
        from live_data_cache.bar_store import get_bars, get_bar_count
        bs = get_bars(symbol.upper(), tf)
        if bs is None:
            return {"symbol": symbol, "tf": tf, "count": 0, "bars": []}
        bars_out = [
            {
                "time": int(b.timestamp.timestamp()),
                "open": b.open, "high": b.high, "low": b.low, "close": b.close,
                "volume": b.volume,
            }
            for b in bs.bars
        ]
        return {"symbol": symbol.upper(), "tf": tf, "count": len(bars_out), "bars": bars_out}
    except Exception as e:
        return {"error": str(e)}


# ═══ ON-DEMAND SYMBOL ANALYSIS ═══════════════════════════════════════════════

_TEMP_CACHE = Path("cache/temp_symbols")


def _cleanup_temp_cache():
    today = datetime.now().strftime("%Y%m%d")
    if _TEMP_CACHE.exists():
        for f in _TEMP_CACHE.glob("*.json"):
            if today not in f.name:
                try:
                    f.unlink()
                except Exception:
                    pass


@app.get("/api/analyze/{symbol}")
async def analyze_symbol(
    symbol: str,
    timeframe: str = Query("1d"),
    days: int = Query(365),
    force: bool = Query(False),
):
    """Fetch + classify any symbol on demand. Results cached for the trading day."""
    sym = symbol.upper().strip()
    today = datetime.now().strftime("%Y%m%d")
    _TEMP_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = _TEMP_CACHE / f"{sym}_{timeframe}_{today}.json"

    if not force and cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    try:
        from backend.patterns.classifier import classify_all
        from backend.data.massive_client import fetch_bars as _fb

        bars_obj = _fb(sym, timeframe, days)
        if not bars_obj or not bars_obj.bars or len(bars_obj.bars) < 20:
            return {"error": f"No data for {sym}", "symbol": sym}

        b = bars_obj.bars
        closes = [x.close for x in b]
        last = b[-1]
        atr = float(np.mean([x.high - x.low for x in b[-14:]])) if len(b) >= 14 else 0.0

        def sma(n):
            return round(sum(closes[-n:]) / n, 2) if len(closes) >= n else None

        sma20, sma50, sma200 = sma(20), sma(50), sma(200)

        lookback = b[-252:] if len(b) >= 252 else b
        hi52 = round(max(x.high for x in lookback), 2)
        lo52 = round(min(x.low  for x in lookback), 2)

        def rsi14():
            if len(closes) < 15:
                return None
            deltas = [closes[i] - closes[i-1] for i in range(-14, 0)]
            ag = sum(d for d in deltas if d > 0) / 14
            al = sum(-d for d in deltas if d < 0) / 14
            return round(100 - 100 / (1 + ag / al), 1) if al else 100.0

        def trend_line():
            p, parts = last.close, []
            if sma200: parts.append(f"{'above' if p > sma200 else 'below'} 200d SMA (${sma200})")
            if sma50:  parts.append(f"{'above' if p > sma50  else 'below'} 50d SMA (${sma50})")
            if sma20:  parts.append(f"{'above' if p > sma20  else 'below'} 20d SMA (${sma20})")
            return ", ".join(parts) if parts else "insufficient history"

        setups_raw = classify_all(bars_obj)
        setups = []
        for s in setups_raw:
            bias_str = s.bias.value if hasattr(s.bias, "value") else str(s.bias)
            ep, sl, tp = s.entry_price, s.stop_loss, s.target_price
            t1 = getattr(s, "target_1", tp)
            t2 = getattr(s, "target_2", tp)
            risk = abs(ep - sl)
            setups.append({
                "symbol": sym, "pattern_name": s.pattern_name, "bias": bias_str,
                "entry_price": round(ep, 2), "stop_loss": round(sl, 2),
                "target_price": round(tp, 2), "target_1": round(t1, 2), "target_2": round(t2, 2),
                "confidence": round(s.confidence, 3), "description": s.description,
                "timeframe_detected": timeframe,
                "category": PATTERN_META.get(s.pattern_name, {}).get("category", "momentum"),
                "risk_reward_ratio": round(abs(tp - ep) / risk, 2) if risk > 0 else 0,
                "composite_score": round(s.confidence * 100, 1),
                "detected_at": datetime.now().isoformat(),
            })

        chart_bars = [
            {"time": int(x.timestamp.timestamp()), "open": round(x.open, 2),
             "high": round(x.high, 2), "low": round(x.low, 2), "close": round(x.close, 2),
             "volume": int(x.volume)}
            for x in b[-120:]
        ]

        prev = b[-2].close if len(b) >= 2 else last.close
        result = {
            "symbol": sym, "timeframe": timeframe,
            "last_price": round(last.close, 2),
            "change_pct": round((last.close - prev) / prev * 100, 2) if prev else 0,
            "atr": round(atr, 2), "atr_pct": round(atr / last.close * 100, 2) if last.close else 0,
            "high_52w": hi52, "low_52w": lo52,
            "pct_from_high": round((last.close - hi52) / hi52 * 100, 2) if hi52 else 0,
            "pct_from_low":  round((last.close - lo52) / lo52 * 100, 2) if lo52 else 0,
            "sma20": sma20, "sma50": sma50, "sma200": sma200,
            "rsi14": rsi14(), "trend_line": trend_line(),
            "bars_count": len(b), "setups": setups, "setups_count": len(setups),
            "chart_bars": chart_bars, "cached_at": datetime.now().isoformat(),
        }

        cache_file.write_text(json.dumps(result))
        _cleanup_temp_cache()
        return result

    except Exception as e:
        return {"error": str(e), "symbol": sym}


# ═══ PERFORMANCE DASHBOARD ═══════════════════════════════════════════════════

@app.get("/api/performance/summary")
async def performance_summary():
    """Full performance dashboard: equity curve, drawdown, attribution, streaks."""
    try:
        from backend.analytics.performance import get_performance_summary
        return get_performance_summary()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/performance/equity")
async def performance_equity():
    try:
        from backend.analytics.performance import get_equity_curve
        return {"equity_curve": get_equity_curve()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/performance/patterns")
async def performance_patterns():
    try:
        from backend.analytics.performance import get_pattern_attribution
        return {"pattern_attribution": get_pattern_attribution()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/performance/daily")
async def performance_daily(days: int = Query(30)):
    try:
        from backend.analytics.performance import get_daily_pnl
        return {"daily_pnl": get_daily_pnl(days)}
    except Exception as e:
        return {"error": str(e)}


# ═══ ALERTS / WEBHOOKS ═══════════════════════════════════════════════════════

@app.get("/api/alerts/config")
async def alerts_config_get():
    try:
        from backend.alerts.webhook import get_alert_config
        return get_alert_config()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/alerts/config")
async def alerts_config_set(request: Request):
    try:
        from backend.alerts.webhook import save_alert_config
        body = await request.json()
        save_alert_config(body)
        return {"status": "saved"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/alerts/test")
async def alerts_test():
    try:
        from backend.alerts.webhook import test_alerts
        return test_alerts()
    except Exception as e:
        return {"error": str(e), "success": False}


# ═══ POSITION SIZING ═══════════════════════════════════════════════════════

@app.get("/api/sizing/config")
async def sizing_config_get():
    try:
        from backend.sizing.engine import get_sizing_config
        return get_sizing_config()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/sizing/config")
async def sizing_config_set(request: Request):
    try:
        from backend.sizing.engine import save_sizing_config
        body = await request.json()
        save_sizing_config(body)
        return {"status": "saved"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/sizing/calculate")
async def sizing_calculate(entry: float = Query(...), stop: float = Query(...), bias: str = Query("long"), modifier: float = Query(1.0)):
    try:
        from backend.sizing.engine import calculate_position
        return calculate_position(entry, stop, bias, modifier)
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/sizing/summary")
async def sizing_summary():
    try:
        from backend.sizing.engine import get_position_summary
        return get_position_summary()
    except Exception as e:
        return {"error": str(e)}


# ═══ AGENT TRADING SIMULATION ═══

@app.get("/api/agent-trading/dates")
async def agent_trading_dates():
    """Return available dates for intraday simulation."""
    try:
        from simulation.intraday import get_available_dates
        return {"dates": get_available_dates()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/agent-trading/results")
async def agent_trading_results():
    """Return long-term simulation results (equity curve, trades, stats)."""
    results_file = Path("simulation/output/simulation_results.json")
    if not results_file.exists():
        return {"error": "No simulation results found. Run: python -m simulation.run"}
    try:
        data = json.loads(results_file.read_text())
        # Compute daily P&L from equity curve
        curve = data.get("equity_curve", [])
        for i, pt in enumerate(curve):
            if i == 0:
                pt["daily_pnl"] = pt["equity"] - data["config"]["starting_capital"]
            else:
                pt["daily_pnl"] = pt["equity"] - curve[i - 1]["equity"]
        # Compute per-trade cumulative P&L for trade timeline
        trades = data.get("closed_trades", [])
        cum_pnl = 0
        cum_r = 0
        for tr in trades:
            risk = tr.get("dollar_risk", 0)
            pnl = tr.get("realized_r", 0) * risk
            cum_pnl += pnl
            cum_r += tr.get("realized_r", 0)
            tr["cum_pnl"] = round(cum_pnl, 2)
            tr["cum_r"] = round(cum_r, 3)
            tr["pnl"] = round(pnl, 2)
        return data
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/agent-trading/results-plus")
async def agent_trading_results_plus():
    """Return Deterministic+ simulation results."""
    results_file = Path("simulation/output/det_plus_results.json")
    if not results_file.exists():
        return {"error": "No Deterministic+ results found."}
    try:
        data = json.loads(results_file.read_text())
        # Compute per-trade cumulative P&L
        trades = data.get("closed_trades", [])
        cum_pnl = 0
        cum_r = 0
        for tr in trades:
            risk = tr.get("dollar_risk", 0)
            pnl = tr.get("pnl", tr.get("realized_r", 0) * risk)
            cum_pnl += pnl
            cum_r += tr.get("realized_r", 0)
            tr["cum_pnl"] = round(cum_pnl, 2)
            tr["cum_r"] = round(cum_r, 3)
            tr.setdefault("pnl", round(pnl, 2))
        return data
    except Exception as e:
        return {"error": str(e)}


# Active simulation instance (one at a time)
_active_sim = {"instance": None, "task": None}


@app.websocket("/ws/agent-trading")
async def agent_trading_ws(websocket: WebSocket):
    """WebSocket endpoint for live agent trading simulation.

    Client sends JSON commands:
      {"action": "start", "date": "2026-04-01", "speed": 2.0, "use_agents": false, "engine": "standard"}
      {"action": "start_continuous", "speed": 10.0, "engine": "det_plus"}
      {"action": "pause"}
      {"action": "resume"}
      {"action": "stop"}
      {"action": "set_speed", "speed": 5.0}

    engine: "standard" (default) or "det_plus" (Deterministic+ with adaptive sizing, multi-TF, learning)
    Server streams SimEvent JSON objects continuously.
    """
    await websocket.accept()

    import asyncio
    from simulation.intraday import IntradaySimulation, SimEvent, get_available_dates

    sim = None  # IntradaySimulation or IntradayPlusSimulation
    sim_task: asyncio.Task | None = None
    send_queue: asyncio.Queue = asyncio.Queue()

    async def sender():
        """Drain the send queue to the WebSocket."""
        try:
            while True:
                event_dict = await send_queue.get()
                try:
                    await websocket.send_json(event_dict)
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    def emit(event):
        """Queue event for sending (sync-safe)."""
        try:
            send_queue.put_nowait(event.to_dict())
        except Exception:
            pass

    sender_task = asyncio.create_task(sender())

    async def _stop_sim():
        nonlocal sim, sim_task
        if sim is not None:
            sim.stop()
        if sim_task is not None and not sim_task.done():
            sim_task.cancel()
            try:
                await sim_task
            except (asyncio.CancelledError, Exception):
                pass

    def _create_sim(data: dict):
        """Create the right simulation engine based on 'engine' param."""
        engine = data.get("engine", "standard")
        speed = data.get("speed", 2.0)
        capital = data.get("capital", 100_000)
        min_score = data.get("min_score", 50.0)

        if engine == "det_plus":
            from simulation.intraday_plus import IntradayPlusSimulation
            return IntradayPlusSimulation(
                emit=emit,
                starting_capital=capital,
                playback_speed=speed,
                min_score=min_score,
            )
        else:
            use_agents = data.get("use_agents", False)
            return IntradaySimulation(
                emit=emit,
                starting_capital=capital,
                playback_speed=speed,
                use_agents=use_agents,
                min_score=min_score,
            )

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "")

            if action == "start":
                await _stop_sim()
                date = data.get("date", "")
                sim = _create_sim(data)
                sim_task = asyncio.create_task(sim.run_day(date))

            elif action == "start_continuous":
                await _stop_sim()
                dates = data.get("dates") or get_available_dates()
                sim = _create_sim(data)
                sim_task = asyncio.create_task(sim.run_continuous(dates))

            elif action == "pause" and sim:
                sim.pause()
            elif action == "resume" and sim:
                sim.resume()
            elif action == "stop" and sim:
                sim.stop()
            elif action == "set_speed" and sim:
                sim.set_speed(data.get("speed", 2.0))

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if sim:
            sim.stop()
        sender_task.cancel()


# ═══ CUSTOM AGENT TRADING ═══
# Background-persistent runs, strategy filtering, adaptive sizing,
# agent deliberation, run logging. Navigate away without stopping.

from simulation.custom.manager import RunManager
from simulation.custom.config import (
    CustomSimConfig, AgentModelConfig, SizingConfig, DeliberationConfig,
)

_run_manager = RunManager.instance()


@app.get("/api/custom-trading/status")
async def get_custom_status():
    """Return system status including API key availability."""
    import os
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
    return {
        "api_key_set": has_key,
        "active_runs": len(_run_manager.get_active_runs()),
        "saved_runs": len(_run_manager.get_saved_runs()),
        "available_dates": len(_run_manager.get_available_dates()),
    }


@app.get("/api/custom-trading/strategies")
async def get_strategies():
    """Return all available strategies grouped by category."""
    return {
        "strategies": _run_manager.get_strategies(),
        "groups": _run_manager.get_strategy_groups(),
    }


@app.get("/api/custom-trading/dates")
async def get_custom_dates():
    """Return all available dates for simulation."""
    dates = _run_manager.get_available_dates()
    return {"dates": dates, "count": len(dates)}


@app.get("/api/custom-trading/runs")
async def get_custom_runs():
    """List all saved + active runs."""
    saved = _run_manager.get_saved_runs()
    active = _run_manager.get_active_runs()
    return {"saved": saved, "active": active}


@app.get("/api/custom-trading/runs/{run_id}")
async def get_custom_run(run_id: str):
    """Get a specific run (live status or saved results)."""
    status = _run_manager.get_run_status(run_id)
    if status and status.get("live"):
        return status
    saved = _run_manager.get_saved_run(run_id)
    if saved:
        return saved
    return {"error": "Run not found"}


@app.delete("/api/custom-trading/runs/{run_id}")
async def delete_custom_run(run_id: str):
    """Delete a saved run."""
    ok = _run_manager.delete_saved_run(run_id)
    return {"deleted": ok}


@app.post("/api/custom-trading/compare")
async def compare_custom_runs(request: Request):
    """Compare multiple runs side-by-side."""
    body = await request.json()
    run_ids = body.get("run_ids", [])
    return _run_manager.compare_saved_runs(run_ids)


@app.post("/api/custom-trading/start")
async def start_custom_run(request: Request):
    """Start a new custom trading simulation run.

    The run continues in the background — no WebSocket needed to keep it alive.
    Connect via /ws/custom-trading/{run_id} to stream live events.
    """
    body = await request.json()

    # Build config from request
    config = CustomSimConfig.from_dict(body)

    run_id = await _run_manager.start_run(config)
    return {"run_id": run_id, "status": "started"}


@app.post("/api/custom-trading/stop/{run_id}")
async def stop_custom_run(run_id: str):
    ok = _run_manager.stop_run(run_id)
    return {"stopped": ok}


@app.post("/api/custom-trading/pause/{run_id}")
async def pause_custom_run(run_id: str):
    ok = _run_manager.pause_run(run_id)
    return {"paused": ok}


@app.post("/api/custom-trading/resume/{run_id}")
async def resume_custom_run(run_id: str):
    ok = _run_manager.resume_run(run_id)
    return {"resumed": ok}


@app.post("/api/custom-trading/speed/{run_id}")
async def set_custom_speed(run_id: str, request: Request):
    body = await request.json()
    ok = _run_manager.set_speed(run_id, body.get("speed", 10.0))
    return {"ok": ok}


@app.websocket("/ws/custom-trading/{run_id}")
async def custom_trading_ws(websocket: WebSocket, run_id: str):
    """WebSocket for streaming live events from a custom run.

    This is a VIEW-ONLY connection. The run continues even if this
    WebSocket disconnects. Reconnecting replays recent events.

    Client can send: {"action": "pause"}, {"action": "resume"},
                     {"action": "stop"}, {"action": "set_speed", "speed": N}
    """
    await websocket.accept()
    import asyncio

    queue = _run_manager.subscribe(run_id)
    if queue is None:
        await websocket.send_json({"type": "error", "message": f"Run {run_id} not found"})
        await websocket.close()
        return

    async def sender():
        try:
            while True:
                event = await queue.get()
                try:
                    await websocket.send_json(event)
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    async def receiver():
        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action", "")
                if action == "pause":
                    _run_manager.pause_run(run_id)
                elif action == "resume":
                    _run_manager.resume_run(run_id)
                elif action == "stop":
                    _run_manager.stop_run(run_id)
                elif action == "set_speed":
                    _run_manager.set_speed(run_id, data.get("speed", 10.0))
        except (WebSocketDisconnect, Exception):
            pass

    sender_task = asyncio.create_task(sender())
    receiver_task = asyncio.create_task(receiver())

    try:
        # Wait for either task to complete (usually receiver on disconnect)
        done, pending = await asyncio.wait(
            [sender_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except Exception:
        pass
    finally:
        # Unsubscribe WITHOUT stopping the run
        _run_manager.unsubscribe(run_id, queue)
        sender_task.cancel()
        receiver_task.cancel()


# ═══ LIVE TRADING SCANNER ═══
# Real-time scanning via execution-bridge live_scanner.py
# WebSocket streams structured events to frontend.

@app.websocket("/ws/live-trading")
async def live_trading_ws(websocket: WebSocket):
    """WebSocket for live market scanning + paper trading.

    Client sends:
      {"action": "start", "symbols": ["AAPL","NVDA"], "dry_run": false, "scan_interval": 300}
      {"action": "start", "max_symbols": 30}   # uses top_symbols.json
      {"action": "pause"}
      {"action": "resume"}
      {"action": "stop"}
    Server streams LiveEvent JSON objects.
    """
    await websocket.accept()

    import asyncio
    import sys as _sys
    from pathlib import Path as _Path

    # Add execution-bridge to path
    eb_path = _Path(__file__).resolve().parent.parent / "execution-bridge"
    if str(eb_path) not in _sys.path:
        _sys.path.insert(0, str(eb_path))

    scanner = None
    scanner_task: asyncio.Task | None = None
    send_queue: asyncio.Queue = asyncio.Queue()

    async def sender():
        try:
            while True:
                event_dict = await send_queue.get()
                try:
                    await websocket.send_json(event_dict)
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    def emit(event):
        try:
            send_queue.put_nowait(event.to_dict())
        except Exception:
            pass

    sender_task = asyncio.create_task(sender())

    async def _stop_scanner():
        nonlocal scanner, scanner_task
        if scanner is not None:
            scanner.stop()
        if scanner_task is not None and not scanner_task.done():
            scanner_task.cancel()
            try:
                await scanner_task
            except (asyncio.CancelledError, Exception):
                pass

    def _resolve_symbols(data: dict) -> list[str]:
        symbols = data.get("symbols")
        if symbols and isinstance(symbols, list):
            return symbols
        max_syms = data.get("max_symbols", 30)
        top_file = _Path(__file__).resolve().parent.parent / "cache" / "top_symbols.json"
        if top_file.exists():
            try:
                return json.loads(top_file.read_text())["symbols"][:max_syms]
            except Exception:
                pass
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                "TSLA", "JPM", "V", "MA", "AMD", "AVGO", "CRM",
                "NFLX", "COST", "UNH", "LLY", "ABBV", "PEP", "KO"]

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "")

            if action == "start":
                await _stop_scanner()
                from live_scanner import LiveScanner

                symbols = _resolve_symbols(data)
                dry_run = data.get("dry_run", True)
                interval = data.get("scan_interval", 300)
                live_mode = data.get("live_mode", False)

                scanner = LiveScanner(
                    symbols=symbols,
                    dry_run=dry_run,
                    scan_interval=interval,
                    emit=emit,
                    live_mode=live_mode,
                )
                scanner_task = asyncio.create_task(scanner.run_async())

            elif action == "pause" and scanner:
                scanner.pause()
            elif action == "resume" and scanner:
                scanner.resume()
            elif action == "stop" and scanner:
                scanner.stop()

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if scanner:
            scanner.stop()
        sender_task.cancel()


@app.get("/api/live-trading/session")
async def live_trading_session():
    """Return the most recent session log if one exists."""
    log_path = Path(__file__).resolve().parent.parent / "execution-bridge" / "cache" / "session_log.json"
    if log_path.exists():
        return json.loads(log_path.read_text())
    return {"error": "No session log found"}

