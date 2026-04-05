"""
main.py — Juicer API Server v1.0

Start: uvicorn backend.main:app --reload --port 8000
"""
import json, time
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

# ═══ SCHEDULER ═══

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

_scheduler = BackgroundScheduler(timezone="America/New_York")
_last_scan: dict = {"ran_at": None, "added": 0}
_last_refresh: dict = {"ran_at": None}

def _job_refresh():
    """Refresh live prices every 30 min on market days 9:30–16:00 ET."""
    try:
        from backend.tracker.routes import get_tracker
        get_tracker().refresh_prices()
        _last_refresh["ran_at"] = datetime.now().isoformat()
        print(f"  [Scheduler] Prices refreshed at {datetime.now().strftime('%H:%M ET')}")
    except Exception as e:
        print(f"  [Scheduler] Refresh error: {e}")

def _job_scan():
    """Daily setup scan at 4:30 PM ET after market close."""
    try:
        from backend.tracker.routes import get_tracker
        added = get_tracker().scan_and_add(top_n=50)
        _last_scan["ran_at"] = datetime.now().isoformat()
        _last_scan["added"] = added
        print(f"  [Scheduler] Daily scan complete — {added} new setups added")
    except Exception as e:
        print(f"  [Scheduler] Scan error: {e}")

# Refresh every 30 min, Mon–Fri, 9:30 AM – 4:00 PM ET
_scheduler.add_job(_job_refresh, CronTrigger(
    day_of_week="mon-fri", hour="9-15", minute="*/5", timezone="America/New_York"
), id="refresh", name="Price Refresh")
# Daily scan 30 min after close
_scheduler.add_job(_job_scan, CronTrigger(
    day_of_week="mon-fri", hour=16, minute=30, timezone="America/New_York"
), id="daily_scan", name="Daily Scan")

@asynccontextmanager
async def lifespan(app: FastAPI):
    _scheduler.start()
    print("  [Scheduler] Started — daily scan 4:30 PM ET | refresh every 5 min market hours")
    yield
    _scheduler.shutdown(wait=False)
    print("  [Scheduler] Stopped")

app = FastAPI(title="Juicer API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://localhost:3000","http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
from backend.tracker.routes import router as tracker_router
app.include_router(tracker_router, prefix="/api/tracker")

_evaluator = StrategyEvaluator()
_evaluator.load()
BT = Path("cache/backtest_results.json")

def _mkt_open() -> bool:
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
    if mode not in ("today","active"): return {"error":"mode must be 'today' or 'active'"}
    setups = scan_symbol(symbol.upper(),mode=mode,evaluator=_evaluator)
    # Only filter market hours when market is actually open
    if mode == "today" and _mkt_open(): setups = _mkt_hours_filter(setups)
    # AI evaluation
    if ai and setups:
        try:
            news = fetch_news(symbol.upper(),15); hl = format_headlines_for_llm(news)
            setups = evaluate_setups_batch(setups,{symbol.upper():hl},_regime_str(),top_n=5)
        except Exception as e: print(f"  AI error: {e}")
    # Live correlation
    try:
        corr = compute_correlation_live(symbol.upper()).to_dict()
        for s in setups: s["spy_correlation"] = corr
    except Exception as e:
        print(f"  [Correlation] {symbol}: {e}")
    # Remove anything below 45 score
    setups = [s for s in setups if s.get("composite_score", 0) >= 40]
    return {"symbol":symbol.upper(),"mode":mode,"count":len(setups),"setups":setups,"market_open":_mkt_open()}

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
async def top_opportunities(ai:bool=Query(True),per_symbol:int=Query(2),max_symbols:int=Query(15)):
    t = time.time()
    ip = get_in_play()
    symbols = [s.symbol for s in ip.stocks if s.symbol][:max_symbols]
    if not symbols:
        return {"market_summary":ip.market_summary,"in_play":ip.to_dict(),"setups":[],"count":0,"market_open":_mkt_open()}

    all_setups = scan_multiple(symbols,mode="active",evaluator=_evaluator)

    # Cap to top N per symbol
    by_sym: dict[str, list] = defaultdict(list)
    for s in all_setups: by_sym[s.get("symbol","")].append(s)
    capped = []
    for sym, ss in by_sym.items():
        ss.sort(key=lambda x: x.get("composite_score",0), reverse=True)
        capped.extend(ss[:per_symbol])
    capped.sort(key=lambda x: x.get("composite_score",0), reverse=True)
    # Remove anything below 45
    capped = [s for s in capped if s.get("composite_score", 0) >= 45]
    # Only filter market hours when market is open
    if _mkt_open(): capped = _mkt_hours_filter(capped)
    # Only AI-evaluate top 50% by composite score — cuts Ollama calls in half
    cutoff = max(len(capped) // 2, 1)
    ai_batch = capped[:cutoff]
    skip_batch = capped[cutoff:]
    print(f"  Top 50% filter: {len(ai_batch)} for AI, {len(skip_batch)} skipped")

    # AI evaluate only the top half
    if ai and ai_batch:
        try:
            syms = list(set(s.get("symbol","") for s in ai_batch))
            nb = fetch_news_batch(syms); ns = {sym:format_headlines_for_llm(items) for sym,items in nb.items()}
            ai_batch = evaluate_setups_batch(ai_batch,ns,_regime_str(),top_n=per_symbol)
        except Exception as e: print(f"  AI error: {e}")

    # Only return AI-evaluated setups (top 50%)
    capped = ai_batch

    # Live correlation per symbol
    corr_map: dict[str, dict] = {}
    for s in capped:
        sym = s.get("symbol","")
        if sym and sym not in corr_map:
            try: corr_map[sym] = compute_correlation_live(sym).to_dict()
            except Exception as e:
                print(f"  [Correlation] {sym}: {e}")
                corr_map[sym] = {}
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


# ═══ SCHEDULER STATUS ═══
@app.get("/api/tracker/schedule")
async def tracker_schedule():
    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        })
    return {
        "running": _scheduler.running,
        "jobs": jobs,
        "last_scan": _last_scan,
        "last_refresh": _last_refresh,
    }
 
