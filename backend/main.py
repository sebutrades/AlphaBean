"""
main.py — Juicer API Server v1.0

Start: uvicorn backend.main:app --reload --port 8000
"""
import json, os, time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, time as dt_time
from pathlib import Path
from fastapi import FastAPI, Query
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
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://localhost:3000","http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
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

