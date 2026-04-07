"""
tools/agent_swarm.py  —  Multi-Agent Trading Analysis Swarm

Spawns hundreds of Claude agents, each with a different trading-expert persona,
feeds them your actual 180-day trading data, and synthesizes their insights
into a ranked, actionable report.

ARCHITECTURE:
  1. DATA EXTRACTION  — pulls trades, patterns, equity curve, drawdowns,
                        AI verdicts, regime data from AlphaBean's data stores
  2. EXPERT ROSTER    — 15 specialist personas, each given different data slices
  3. ORCHESTRATOR     — async parallel API calls with concurrency limits
  4. SYNTHESIS        — a meta-agent reads all insights and produces a final report

USAGE:
  # Set your API key first
  export ANTHROPIC_API_KEY=sk-ant-...

  # Full run (default ~200 agents, takes ~5-10 min, ~$3-5 in API cost)
  python tools/agent_swarm.py

  # Quick test (15 agents, one per persona)
  python tools/agent_swarm.py --quick

  # Custom agent count per persona
  python tools/agent_swarm.py --agents-per-persona 20

  # Skip synthesis (just collect raw insights)
  python tools/agent_swarm.py --no-synthesis

OUTPUT:
  reports/swarm_report_YYYY-MM-DD_HHMM.json   — full raw data
  reports/swarm_report_YYYY-MM-DD_HHMM.md     — formatted actionable report
"""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic SDK not installed. Run: pip install anthropic")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_all_data() -> dict:
    """Pull every relevant data source into a single dict for agent consumption."""
    data = {}

    # ── Closed trades (from all sources) ──────────────────────────────────────
    all_trades = []
    seen_ids = set()

    # Active trades file (includes closed trades that haven't been archived)
    active_raw = _read_json(ROOT / "cache" / "active_trades.json")
    if active_raw and isinstance(active_raw, dict):
        for t in active_raw.get("trades", []):
            tid = t.get("id", "")
            if tid not in seen_ids:
                seen_ids.add(tid)
                all_trades.append(t)

    # Archived trades
    archive_raw = _read_json(ROOT / "cache" / "archived_trades.json")
    if archive_raw and isinstance(archive_raw, dict):
        for t in archive_raw.get("trades", []):
            tid = t.get("id", "")
            if tid not in seen_ids:
                seen_ids.add(tid)
                all_trades.append(t)

    # Intraday closed setups
    live_cache = ROOT / "live_data_cache"
    if live_cache.exists():
        for path in sorted(live_cache.glob("closed_*.json")):
            raw = _read_json(path)
            setups = []
            if isinstance(raw, dict):
                setups = raw.get("setups", [])
            elif isinstance(raw, list):
                setups = raw
            for t in setups:
                tid = t.get("id", "")
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    all_trades.append(t)

    # Separate into closed and open — a trade is "closed" if it has a realized_r
    # and closed_at timestamp, regardless of status label (intraday winners may
    # have status="AT_T2" since they hit T2 and were closed with profit)
    closed = [t for t in all_trades
              if t.get("realized_r") is not None
              and t.get("closed_at")]
    active = [t for t in all_trades
              if not t.get("closed_at")
              and t.get("status") in ("PENDING", "ACTIVE", "AT_T1", "AT_T2", "TRAILING")]

    closed.sort(key=lambda t: t.get("closed_at", ""))

    data["closed_trades"] = closed
    data["active_trades"] = active
    data["total_closed"] = len(closed)
    data["total_active"] = len(active)

    # ── Summary stats ─────────────────────────────────────────────────────────
    if closed:
        rs = [t.get("realized_r", 0) for t in closed]
        wins = [r for r in rs if r > 0]
        losses = [r for r in rs if r < 0]
        data["summary"] = {
            "total_closed": len(closed),
            "total_r": round(sum(rs), 3),
            "wins": len(wins),
            "losses": len(losses),
            "flat": len(rs) - len(wins) - len(losses),
            "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "avg_win_r": round(sum(wins) / len(wins), 3) if wins else 0,
            "avg_loss_r": round(sum(losses) / len(losses), 3) if losses else 0,
            "best_trade_r": round(max(rs), 3),
            "worst_trade_r": round(min(rs), 3),
            "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else 99.0,
        }
    else:
        data["summary"] = {"total_closed": 0, "note": "No closed trades found"}

    # ── Pattern breakdown ─────────────────────────────────────────────────────
    by_pattern = defaultdict(list)
    for t in closed:
        by_pattern[t.get("pattern_name", "Unknown")].append(t.get("realized_r", 0))
    pattern_stats = {}
    for name, rs in by_pattern.items():
        w = [r for r in rs if r > 0]
        l = [r for r in rs if r < 0]
        pattern_stats[name] = {
            "trades": len(rs),
            "total_r": round(sum(rs), 3),
            "win_rate": round(len(w) / len(rs) * 100, 1) if rs else 0,
            "avg_r": round(sum(rs) / len(rs), 3) if rs else 0,
            "avg_win": round(sum(w) / len(w), 3) if w else 0,
            "avg_loss": round(sum(l) / len(l), 3) if l else 0,
            "best": round(max(rs), 3),
            "worst": round(min(rs), 3),
            "profit_factor": round(sum(w) / abs(sum(l)), 2) if l else 99.0,
        }
    data["pattern_stats"] = dict(sorted(pattern_stats.items(),
                                        key=lambda x: x[1]["total_r"], reverse=True))

    # ── Symbol breakdown ──────────────────────────────────────────────────────
    by_symbol = defaultdict(list)
    for t in closed:
        by_symbol[t.get("symbol", "?")].append(t.get("realized_r", 0))
    symbol_stats = {}
    for sym, rs in by_symbol.items():
        w = [r for r in rs if r > 0]
        l = [r for r in rs if r < 0]
        symbol_stats[sym] = {
            "trades": len(rs),
            "total_r": round(sum(rs), 3),
            "win_rate": round(len(w) / len(rs) * 100, 1) if rs else 0,
            "avg_r": round(sum(rs) / len(rs), 3) if rs else 0,
        }
    data["symbol_stats"] = dict(sorted(symbol_stats.items(),
                                       key=lambda x: x[1]["total_r"], reverse=True))

    # ── Daily P&L ─────────────────────────────────────────────────────────────
    daily = defaultdict(list)
    for t in closed:
        try:
            d = t["closed_at"][:10]
            daily[d].append(t.get("realized_r", 0))
        except (KeyError, TypeError):
            pass
    data["daily_pnl"] = {d: {"r": round(sum(rs), 3), "trades": len(rs),
                              "wins": sum(1 for r in rs if r > 0),
                              "losses": sum(1 for r in rs if r < 0)}
                          for d, rs in sorted(daily.items())}

    # ── Equity curve ──────────────────────────────────────────────────────────
    cum = 0.0
    equity = []
    for d in sorted(daily.keys()):
        day_r = sum(daily[d])
        cum += day_r
        equity.append({"date": d, "cumulative_r": round(cum, 3),
                        "daily_r": round(day_r, 3)})
    data["equity_curve"] = equity

    # ── Drawdown series ───────────────────────────────────────────────────────
    peak = 0.0
    max_dd = 0.0
    for pt in equity:
        c = pt["cumulative_r"]
        if c > peak:
            peak = c
        dd = peak - c
        if dd > max_dd:
            max_dd = dd
    data["max_drawdown_r"] = round(max_dd, 3)

    # ── Backtest results ──────────────────────────────────────────────────────
    bt = _read_json(ROOT / "cache" / "backtest_results.json")
    if bt:
        data["backtest"] = {
            "config": bt.get("config", {}),
            "summary": bt.get("summary", {}),
            "patterns": {},
        }
        for name, stats in bt.get("patterns", {}).items():
            if isinstance(stats, dict):
                s = stats.get("1d", stats) if "1d" in stats else stats
                if s.get("total_signals", 0) > 0:
                    data["backtest"]["patterns"][name] = {
                        "signals": s.get("total_signals", 0),
                        "win_rate": s.get("win_rate", 0),
                        "expectancy": s.get("expectancy", 0),
                        "profit_factor": s.get("profit_factor", 0),
                        "edge_score": s.get("edge_score", 0),
                        "avg_win_r": s.get("avg_win_r", 0),
                        "avg_loss_r": s.get("avg_loss_r", 0),
                        "t1_hit_rate": s.get("t1_hit_rate"),
                        "t2_hit_rate": s.get("t2_hit_rate"),
                    }

    # ── AI verdict audit data ─────────────────────────────────────────────────
    verdicts_dir = ROOT / "cache" / "ai_verdicts"
    if verdicts_dir.exists():
        verdict_data = []
        for f in verdicts_dir.glob("*.json"):
            v = _read_json(f)
            if v:
                # Extract symbol and pattern from filename
                parts = f.stem.rsplit("_", 1)
                verdict_data.append({
                    "file": f.stem,
                    "verdict": v.get("verdict"),
                    "confidence": v.get("confidence"),
                    "score_delta": v.get("score_delta"),
                    "reasoning": v.get("reasoning", "")[:300],
                    "news_sentiment": v.get("news_sentiment"),
                    "risk_flags": v.get("risk_flags", []),
                })
        data["ai_verdicts"] = verdict_data

    # ── Trade detail samples (for deep analysis) ─────────────────────────────
    # Top 20 winners and top 20 losers with full detail
    sorted_by_r = sorted(closed, key=lambda t: t.get("realized_r", 0))
    worst_20 = sorted_by_r[:20]
    best_20 = sorted_by_r[-20:]

    def _trade_detail(t: dict) -> dict:
        return {
            "symbol": t.get("symbol"),
            "pattern": t.get("pattern_name"),
            "bias": t.get("bias"),
            "entry": t.get("entry_price"),
            "stop": t.get("stop_loss"),
            "target_1": t.get("target_1"),
            "target_2": t.get("target_2"),
            "realized_r": t.get("realized_r"),
            "status": t.get("status"),
            "entered_at": t.get("entered_at"),
            "closed_at": t.get("closed_at"),
            "bars_held": t.get("bars_held"),
            "peak_r": t.get("peak_r"),
            "trough_r": t.get("trough_r"),
            "t1_hit": t.get("t1_hit"),
            "t2_hit": t.get("t2_hit"),
            "initial_risk": t.get("initial_risk"),
            "confidence": t.get("confidence"),
            "source": t.get("source"),
        }

    data["best_trades"] = [_trade_detail(t) for t in best_20]
    data["worst_trades"] = [_trade_detail(t) for t in worst_20]

    # ── Win/loss streaks ──────────────────────────────────────────────────────
    streaks = []
    current = {"type": None, "length": 0, "r": 0}
    for t in closed:
        r = t.get("realized_r", 0)
        stype = "win" if r > 0 else ("loss" if r < 0 else "flat")
        if stype == current["type"]:
            current["length"] += 1
            current["r"] += r
        else:
            if current["type"]:
                streaks.append(dict(current))
            current = {"type": stype, "length": 1, "r": r}
    if current["type"]:
        streaks.append(dict(current))
    data["streaks"] = streaks

    # ── Bias analysis ─────────────────────────────────────────────────────────
    by_bias = defaultdict(list)
    for t in closed:
        by_bias[t.get("bias", "unknown")].append(t.get("realized_r", 0))
    data["bias_stats"] = {}
    for bias, rs in by_bias.items():
        w = [r for r in rs if r > 0]
        data["bias_stats"][bias] = {
            "trades": len(rs),
            "total_r": round(sum(rs), 3),
            "win_rate": round(len(w) / len(rs) * 100, 1) if rs else 0,
        }

    # ── Time-of-day analysis (from entered_at) ───────────────────────────────
    by_hour = defaultdict(list)
    for t in closed:
        try:
            h = datetime.fromisoformat(t["entered_at"]).hour
            by_hour[h].append(t.get("realized_r", 0))
        except Exception:
            pass
    data["hour_stats"] = {
        str(h): {"trades": len(rs), "total_r": round(sum(rs), 3),
                 "win_rate": round(sum(1 for r in rs if r > 0) / len(rs) * 100, 1)}
        for h, rs in sorted(by_hour.items())
    }

    # ── Hold time analysis ────────────────────────────────────────────────────
    hold_times = []
    for t in closed:
        try:
            enter = datetime.fromisoformat(t["entered_at"])
            close = datetime.fromisoformat(t["closed_at"])
            hours = (close - enter).total_seconds() / 3600
            hold_times.append({
                "hours": round(hours, 1),
                "realized_r": t.get("realized_r", 0),
                "pattern": t.get("pattern_name"),
            })
        except Exception:
            pass
    if hold_times:
        winners_ht = [h["hours"] for h in hold_times if h["realized_r"] > 0]
        losers_ht = [h["hours"] for h in hold_times if h["realized_r"] < 0]
        data["hold_time"] = {
            "avg_winner_hours": round(sum(winners_ht) / len(winners_ht), 1) if winners_ht else 0,
            "avg_loser_hours": round(sum(losers_ht) / len(losers_ht), 1) if losers_ht else 0,
            "median_hold_hours": round(sorted([h["hours"] for h in hold_times])[len(hold_times)//2], 1),
        }

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXPERT PERSONAS
# ═══════════════════════════════════════════════════════════════════════════════

EXPERT_PERSONAS = [
    {
        "id": "pattern_edge",
        "name": "Pattern Edge Analyst",
        "focus": "Which patterns actually make money and which are bleeding capital",
        "system": (
            "You are a quantitative pattern researcher. You analyze which trading patterns "
            "generate real edge (positive expectancy) and which are fool's gold. Focus on: "
            "sample size, win rate, profit factor, expectancy per trade, consistency across "
            "symbols. Be brutally honest about patterns that should be dropped. "
            "Recommend specific changes — which patterns to keep, scale up, or remove entirely."
        ),
        "data_keys": ["pattern_stats", "backtest", "summary"],
    },
    {
        "id": "risk_manager",
        "name": "Risk Manager",
        "focus": "Position sizing, drawdowns, risk management effectiveness",
        "system": (
            "You are a portfolio risk manager at a prop desk. Analyze the drawdown profile, "
            "position sizing, stop placement, and overall risk management. Look at: max drawdown, "
            "drawdown duration, risk per trade distribution, stop-loss effectiveness (are stops "
            "too tight? too wide?), and whether the portfolio heat is managed. "
            "Flag specific scenarios where risk was mismanaged and recommend concrete changes."
        ),
        "data_keys": ["summary", "equity_curve", "max_drawdown_r", "worst_trades", "bias_stats", "hold_time"],
    },
    {
        "id": "loss_forensics",
        "name": "Loss Forensics Expert",
        "focus": "Autopsy of losing trades — what went wrong and why",
        "system": (
            "You are a trade forensics analyst who dissects losing trades to find recurring causes "
            "of failure. Look for: common patterns among losers, whether entries were chasing, "
            "whether stops were logically placed, whether the bias was wrong, "
            "hold times that suggest premature stops or overstaying, and whether certain "
            "symbols or times of day consistently produce losses. "
            "Provide specific, actionable filters that would have prevented the worst losses."
        ),
        "data_keys": ["worst_trades", "pattern_stats", "symbol_stats", "hour_stats", "hold_time", "bias_stats"],
    },
    {
        "id": "winner_analysis",
        "name": "Winner Analysis Expert",
        "focus": "What do winning trades have in common — amplify what works",
        "system": (
            "You are a performance analyst focused on replicating success. Analyze the best "
            "trades and identify what they have in common: pattern type, bias, time of day, "
            "hold time, entry quality, R-multiple achieved, whether T1/T2 were hit. "
            "Recommend how to get more trades like the winners — which patterns to prioritize, "
            "what filters to add, and whether position sizing should be increased for high-edge setups."
        ),
        "data_keys": ["best_trades", "pattern_stats", "hour_stats", "hold_time", "bias_stats"],
    },
    {
        "id": "execution_quality",
        "name": "Execution Quality Analyst",
        "focus": "Entry timing, stop placement, target achievement",
        "system": (
            "You are an execution analyst. Examine whether entries are well-timed "
            "(do trades go into immediate drawdown?), whether stops are structurally sound "
            "(too tight = noise stops, too wide = excessive risk), and whether targets are "
            "realistic (T1/T2 hit rates). Look at peak_r vs realized_r to see if profits are "
            "being left on the table or if the trailing mechanism is effective. "
            "Recommend specific changes to entry, stop, and target logic."
        ),
        "data_keys": ["best_trades", "worst_trades", "pattern_stats", "backtest", "hold_time"],
    },
    {
        "id": "time_analyst",
        "name": "Time-of-Day Specialist",
        "focus": "When is the system profitable vs when does it lose money",
        "system": (
            "You are a market microstructure expert analyzing time-of-day effects. "
            "Determine which hours generate the most edge and which are negative. "
            "Consider: opening drive (9:30-10:30), midday chop (11:00-14:00), "
            "power hour (15:00-16:00), and whether the system should avoid certain periods. "
            "Recommend specific time filters."
        ),
        "data_keys": ["hour_stats", "pattern_stats", "summary"],
    },
    {
        "id": "symbol_specialist",
        "name": "Symbol/Sector Specialist",
        "focus": "Which symbols work, which are money pits, sector patterns",
        "system": (
            "You are a sector rotation analyst. Examine per-symbol performance to identify "
            "which stocks the system trades well and which it consistently loses on. "
            "Look for sector patterns — does it work better on tech? energy? defensive names? "
            "Are there symbols that should be blacklisted? Recommend a symbol selection strategy."
        ),
        "data_keys": ["symbol_stats", "pattern_stats", "summary"],
    },
    {
        "id": "streak_psychology",
        "name": "Streak & Psychology Analyst",
        "focus": "Win/loss streaks, revenge trading, tilt detection",
        "system": (
            "You are a trading psychology expert who reads performance data for behavioral patterns. "
            "Analyze streak data: are losing streaks followed by even worse trades (revenge trading)? "
            "Do win streaks lead to overconfidence and bigger losses? Does P&L degrade as more trades "
            "are taken per day (fatigue)? Look at daily trade counts and whether quality degrades "
            "when quantity increases. Recommend behavioral guardrails."
        ),
        "data_keys": ["streaks", "daily_pnl", "summary", "worst_trades"],
    },
    {
        "id": "ai_verdict_auditor",
        "name": "AI Verdict Auditor",
        "focus": "Does the AI evaluation actually improve outcomes",
        "system": (
            "You are an ML model evaluation expert auditing the AI agent's trading verdicts. "
            "Analyze: verdict distribution (too many CAUTION?), does CONFIRMED outperform DENIED? "
            "Is confidence calibrated (higher confidence = better outcomes)? Are the AI's risk flags "
            "actually predictive? Does the score_delta correlate with actual trade outcomes? "
            "Recommend how to improve the AI evaluation pipeline."
        ),
        "data_keys": ["ai_verdicts", "summary", "pattern_stats"],
    },
    {
        "id": "backtest_live_gap",
        "name": "Backtest vs Live Analyst",
        "focus": "Do backtested edges hold up in live trading",
        "system": (
            "You are a quantitative strategist examining backtest vs live performance gap. "
            "Compare: backtest expectancy vs live expectancy per pattern, backtest win rate vs "
            "live win rate, and whether high-edge-score patterns actually deliver in live conditions. "
            "Identify patterns where the live results significantly underperform backtest "
            "(possible overfitting or regime change). Recommend which patterns are robust vs fragile."
        ),
        "data_keys": ["backtest", "pattern_stats", "summary"],
    },
    {
        "id": "portfolio_construction",
        "name": "Portfolio Construction Expert",
        "focus": "Diversification, correlation, capital allocation across patterns",
        "system": (
            "You are a portfolio manager designing optimal capital allocation across trading strategies. "
            "Analyze: Are trades concentrated in one pattern or well-diversified? Are multiple "
            "correlated positions held simultaneously? Is the long/short mix balanced? "
            "Should capital be reallocated from losing patterns to winning ones? "
            "Recommend a portfolio construction strategy: position limits per pattern, "
            "per symbol, per sector, and overall portfolio heat management."
        ),
        "data_keys": ["pattern_stats", "symbol_stats", "bias_stats", "daily_pnl", "summary"],
    },
    {
        "id": "stop_loss_engineer",
        "name": "Stop Loss Engineer",
        "focus": "Are stops optimally placed — noise vs protection tradeoff",
        "system": (
            "You are a stop loss optimization specialist. Analyze: What percentage of trades "
            "hit the exact stop vs close at a worse price (gap risk)? Do trades that hit T1 "
            "then stop at breakeven represent good risk management or missed opportunities? "
            "Is the initial_risk consistent or varying wildly? Compare patterns where stops "
            "are tight vs wide — which delivers better outcomes? "
            "Recommend specific stop logic improvements."
        ),
        "data_keys": ["best_trades", "worst_trades", "pattern_stats", "backtest", "hold_time"],
    },
    {
        "id": "target_optimizer",
        "name": "Target & Exit Optimizer",
        "focus": "Are exits leaving money on the table or taking profits too early",
        "system": (
            "You are an exit strategy specialist. Analyze: T1 and T2 hit rates — are targets "
            "realistic or aspirational? peak_r vs realized_r gap — how much are we leaving on "
            "the table? Should the position split ratio (50/30/20) be adjusted? "
            "Are trailing stops working — do they capture trend continuation or chop out? "
            "Recommend specific target and exit improvements."
        ),
        "data_keys": ["best_trades", "worst_trades", "pattern_stats", "backtest"],
    },
    {
        "id": "edge_decay",
        "name": "Edge Decay Detective",
        "focus": "Is the system's edge growing or shrinking over time",
        "system": (
            "You are an alpha decay analyst. Examine the equity curve and daily P&L over time. "
            "Is the system's edge stable, growing, or decaying? Are recent months worse than "
            "earlier months? Which patterns have had their edges erode? Is the total expectancy "
            "improving or deteriorating? Look for regime shifts that changed what works. "
            "Recommend whether the system needs a fundamental overhaul or just tuning."
        ),
        "data_keys": ["equity_curve", "daily_pnl", "pattern_stats", "summary", "max_drawdown_r"],
    },
    {
        "id": "system_architect",
        "name": "Trading System Architect",
        "focus": "Overall system design — is the architecture sound",
        "system": (
            "You are a senior trading system architect reviewing the entire system. "
            "Look at: the 30-pattern roster — are there too many overlapping patterns? "
            "The scoring system — does composite score correlate with outcomes? "
            "The trade lifecycle — are entries, exits, and position management well-designed? "
            "The data pipeline — is data quality sufficient? "
            "Provide a holistic architectural review with the top 5 structural changes that "
            "would have the biggest impact on profitability."
        ),
        "data_keys": ["summary", "pattern_stats", "backtest", "equity_curve", "bias_stats", "hold_time"],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA SLICING — give each agent run a focused data slice
# ═══════════════════════════════════════════════════════════════════════════════

def build_data_slices(full_data: dict, persona: dict, num_slices: int) -> list[dict]:
    """Build multiple data slices for a persona so each agent run sees different data."""
    keys = persona["data_keys"]
    base = {k: full_data.get(k) for k in keys if full_data.get(k) is not None}

    if num_slices <= 1:
        return [base]

    slices = [base]  # First slice always gets the full view

    # Create focused sub-slices for subsequent runs
    patterns = list(full_data.get("pattern_stats", {}).keys())
    symbols = list(full_data.get("symbol_stats", {}).keys())
    daily_dates = list(full_data.get("daily_pnl", {}).keys())

    for i in range(1, num_slices):
        variation = dict(base)
        # Rotate focus: by pattern group, by symbol group, by time period
        mode = i % 4

        if mode == 0 and patterns:
            # Focus on a subset of patterns
            chunk_size = max(3, len(patterns) // (num_slices - 1))
            start = (i * chunk_size) % len(patterns)
            focus_patterns = patterns[start:start + chunk_size]
            variation["focus_patterns"] = focus_patterns
            variation["instruction"] = f"Focus your analysis specifically on these patterns: {', '.join(focus_patterns)}"

        elif mode == 1 and symbols:
            # Focus on top/bottom symbols
            chunk_size = max(5, len(symbols) // (num_slices - 1))
            start = (i * chunk_size) % len(symbols)
            focus_symbols = symbols[start:start + chunk_size]
            variation["focus_symbols"] = focus_symbols
            variation["instruction"] = f"Focus your analysis on these symbols: {', '.join(focus_symbols[:10])}"

        elif mode == 2 and daily_dates:
            # Focus on a specific time window
            chunk_size = max(7, len(daily_dates) // (num_slices - 1))
            start = (i * chunk_size) % len(daily_dates)
            focus_dates = daily_dates[start:start + chunk_size]
            focus_pnl = {d: full_data["daily_pnl"][d] for d in focus_dates if d in full_data.get("daily_pnl", {})}
            variation["daily_pnl"] = focus_pnl
            variation["instruction"] = f"Focus on the period {focus_dates[0]} to {focus_dates[-1]}"

        elif mode == 3:
            # Contrarian view — challenge assumptions
            variation["instruction"] = (
                "Play devil's advocate. Challenge the prevailing narrative. "
                "If the data looks good, find hidden weaknesses. If it looks bad, "
                "find overlooked strengths. Be contrarian and skeptical."
            )

        slices.append(variation)

    return slices


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AGENT ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

ANALYSIS_MODEL = "claude-sonnet-4-6"      # Fast, cheap for individual analyses
SYNTHESIS_MODEL = "claude-opus-4-6"        # Best reasoning for final synthesis
MAX_CONCURRENT = 30                        # Parallel API calls
MAX_TOKENS = 2000                          # Per agent response


async def run_single_agent(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    persona: dict,
    data_slice: dict,
    run_id: int,
) -> dict:
    """Run a single agent with its persona and data slice."""
    async with semaphore:
        system_prompt = persona["system"]
        extra_instruction = data_slice.pop("instruction", "")
        if extra_instruction:
            system_prompt += f"\n\nSPECIFIC FOCUS FOR THIS ANALYSIS:\n{extra_instruction}"

        # Truncate data to fit context — prioritize the most relevant keys
        data_str = json.dumps(data_slice, indent=2, default=str)
        if len(data_str) > 80000:
            data_str = data_str[:80000] + "\n... [truncated]"

        user_prompt = (
            f"Here is the complete trading data for the past 180 days from our AlphaBean "
            f"trading system. Analyze it thoroughly from your expert perspective.\n\n"
            f"DATA:\n```json\n{data_str}\n```\n\n"
            f"Provide your analysis as a JSON object with these fields:\n"
            f"- \"insights\": list of 3-7 specific, actionable findings (each a string)\n"
            f"- \"critical_issues\": list of the 1-3 most urgent problems to fix\n"
            f"- \"opportunities\": list of 1-3 biggest opportunities to improve performance\n"
            f"- \"specific_recommendations\": list of 2-5 concrete actions to take "
            f"(include specific numbers, thresholds, or parameters when possible)\n"
            f"- \"grade\": overall system grade in this area (A/B/C/D/F)\n"
            f"- \"reasoning\": 2-3 paragraphs explaining your analysis\n\n"
            f"Output ONLY the JSON block."
        )

        t0 = time.time()
        try:
            response = await client.messages.create(
                model=ANALYSIS_MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = response.content[0].text
            elapsed = time.time() - t0

            # Parse JSON from response
            result = _extract_json(raw_text)
            if result is None:
                result = {"raw_text": raw_text[:1500], "parse_error": True}

            return {
                "persona_id": persona["id"],
                "persona_name": persona["name"],
                "run_id": run_id,
                "result": result,
                "elapsed": round(elapsed, 1),
                "tokens_in": response.usage.input_tokens,
                "tokens_out": response.usage.output_tokens,
                "success": True,
            }

        except Exception as e:
            return {
                "persona_id": persona["id"],
                "persona_name": persona["name"],
                "run_id": run_id,
                "result": {"error": str(e)},
                "elapsed": round(time.time() - t0, 1),
                "tokens_in": 0,
                "tokens_out": 0,
                "success": False,
            }


def _extract_json(text: str) -> dict | None:
    """Extract JSON from model response, handling markdown fences."""
    import re
    text = text.strip()

    # Try fenced json block
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    return None


async def run_synthesis(
    client: anthropic.AsyncAnthropic,
    all_results: list[dict],
    full_data: dict,
) -> dict:
    """Final synthesis pass — one Opus agent reads all insights and produces ranked report."""

    # Aggregate insights by persona
    by_persona = defaultdict(list)
    for r in all_results:
        if r["success"] and isinstance(r["result"], dict) and not r["result"].get("parse_error"):
            by_persona[r["persona_name"]].append(r["result"])

    # Build synthesis input
    synthesis_input = []
    for persona_name, results in by_persona.items():
        # Merge insights across runs
        all_insights = []
        all_critical = []
        all_opportunities = []
        all_recommendations = []
        grades = []
        reasonings = []

        for r in results:
            all_insights.extend(r.get("insights", []))
            all_critical.extend(r.get("critical_issues", []))
            all_opportunities.extend(r.get("opportunities", []))
            all_recommendations.extend(r.get("specific_recommendations", []))
            if r.get("grade"):
                grades.append(r["grade"])
            if r.get("reasoning"):
                reasonings.append(r["reasoning"][:500])

        synthesis_input.append({
            "expert": persona_name,
            "grade": grades[0] if grades else "N/A",
            "insights": all_insights[:15],
            "critical_issues": all_critical[:5],
            "opportunities": all_opportunities[:5],
            "recommendations": all_recommendations[:10],
            "sample_reasoning": reasonings[0] if reasonings else "",
        })

    synth_data = json.dumps(synthesis_input, indent=2)

    summary_data = json.dumps({
        "overall_stats": full_data.get("summary", {}),
        "max_drawdown": full_data.get("max_drawdown_r"),
        "patterns_count": len(full_data.get("pattern_stats", {})),
        "total_symbols": len(full_data.get("symbol_stats", {})),
    }, indent=2)

    system_prompt = (
        "You are the Chief Investment Officer synthesizing reports from 15 specialist analysts "
        "who independently reviewed a quantitative trading system's 180-day track record. "
        "Your job is to distill their findings into the definitive improvement roadmap. "
        "Weight insights by frequency (if multiple experts flag the same issue, it's real), "
        "severity (capital-at-risk issues first), and actionability (concrete > vague). "
        "Be direct and specific. This report will drive actual trading decisions."
    )

    user_prompt = (
        f"SYSTEM PERFORMANCE SUMMARY:\n```json\n{summary_data}\n```\n\n"
        f"ANALYST REPORTS ({len(synthesis_input)} experts, "
        f"{sum(len(p['insights']) for p in synthesis_input)} total insights):\n"
        f"```json\n{synth_data}\n```\n\n"
        f"Produce the final synthesis report as a JSON object:\n"
        f"- \"executive_summary\": 3-5 sentence overall assessment\n"
        f"- \"system_grade\": A through F\n"
        f"- \"top_5_actions\": ordered list of the 5 highest-impact changes to make "
        f"(each: {{\"action\": str, \"impact\": str, \"urgency\": \"immediate|this_week|this_month\", "
        f"\"details\": str}})\n"
        f"- \"patterns_to_keep\": list of pattern names that are working\n"
        f"- \"patterns_to_drop\": list of pattern names that should be removed\n"
        f"- \"patterns_to_investigate\": patterns that need more data or parameter tuning\n"
        f"- \"risk_management_changes\": list of specific risk management improvements\n"
        f"- \"execution_improvements\": list of entry/exit/stop improvements\n"
        f"- \"structural_issues\": fundamental system design problems\n"
        f"- \"hidden_strengths\": things the system does well that should be amplified\n"
        f"- \"contrarian_take\": one counterintuitive finding from the analysis\n"
        f"- \"90_day_plan\": {{\"month_1\": str, \"month_2\": str, \"month_3\": str}}\n\n"
        f"Output ONLY the JSON block."
    )

    try:
        response = await client.messages.create(
            model=SYNTHESIS_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text
        result = _extract_json(raw)
        return {
            "synthesis": result or {"raw_text": raw[:3000]},
            "tokens_in": response.usage.input_tokens,
            "tokens_out": response.usage.output_tokens,
        }
    except Exception as e:
        return {"synthesis": {"error": str(e)}, "tokens_in": 0, "tokens_out": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. REPORT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_markdown_report(synthesis: dict, all_results: list[dict], meta: dict) -> str:
    """Convert synthesis JSON into a readable markdown report."""
    s = synthesis.get("synthesis", {})
    lines = []

    lines.append("# AlphaBean Trading System — Agent Swarm Analysis Report")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append(f"*Agents: {meta['total_agents']} | Personas: {meta['personas']} | "
                 f"Cost: ~${meta['estimated_cost']:.2f} | Time: {meta['elapsed_min']:.1f}min*")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append(f"**System Grade: {s.get('system_grade', 'N/A')}**")
    lines.append("")
    lines.append(s.get("executive_summary", "No summary available."))
    lines.append("")

    # Top 5 Actions
    lines.append("## Top 5 Highest-Impact Actions")
    for i, action in enumerate(s.get("top_5_actions", []), 1):
        if isinstance(action, dict):
            urgency_emoji = {"immediate": "🔴", "this_week": "🟡", "this_month": "🟢"}.get(
                action.get("urgency", ""), "⚪")
            lines.append(f"### {i}. {action.get('action', 'N/A')} {urgency_emoji}")
            lines.append(f"**Impact:** {action.get('impact', 'N/A')}")
            lines.append(f"**Urgency:** {action.get('urgency', 'N/A')}")
            lines.append(f"**Details:** {action.get('details', 'N/A')}")
            lines.append("")
        else:
            lines.append(f"{i}. {action}")

    # Pattern Recommendations
    lines.append("## Pattern Recommendations")
    keep = s.get("patterns_to_keep", [])
    drop = s.get("patterns_to_drop", [])
    investigate = s.get("patterns_to_investigate", [])
    if keep:
        lines.append(f"**KEEP** ({len(keep)}): " + ", ".join(keep))
    if drop:
        lines.append(f"**DROP** ({len(drop)}): " + ", ".join(drop))
    if investigate:
        lines.append(f"**INVESTIGATE** ({len(investigate)}): " + ", ".join(investigate))
    lines.append("")

    # Risk Management
    rm = s.get("risk_management_changes", [])
    if rm:
        lines.append("## Risk Management Changes")
        for item in rm:
            lines.append(f"- {item}")
        lines.append("")

    # Execution Improvements
    ei = s.get("execution_improvements", [])
    if ei:
        lines.append("## Execution Improvements")
        for item in ei:
            lines.append(f"- {item}")
        lines.append("")

    # Structural Issues
    si = s.get("structural_issues", [])
    if si:
        lines.append("## Structural Issues")
        for item in si:
            lines.append(f"- {item}")
        lines.append("")

    # Hidden Strengths
    hs = s.get("hidden_strengths", [])
    if hs:
        lines.append("## Hidden Strengths (Amplify These)")
        for item in hs:
            lines.append(f"- {item}")
        lines.append("")

    # Contrarian Take
    ct = s.get("contrarian_take", "")
    if ct:
        lines.append("## Contrarian Take")
        lines.append(f"> {ct}")
        lines.append("")

    # 90-Day Plan
    plan = s.get("90_day_plan", {})
    if plan:
        lines.append("## 90-Day Improvement Plan")
        for month, desc in plan.items():
            lines.append(f"**{month.replace('_', ' ').title()}:** {desc}")
        lines.append("")

    # Per-Expert Grades
    lines.append("## Expert Panel Grades")
    lines.append("| Expert | Grade | Key Finding |")
    lines.append("|--------|-------|-------------|")
    by_persona = defaultdict(list)
    for r in all_results:
        if r["success"] and isinstance(r["result"], dict):
            by_persona[r["persona_name"]].append(r["result"])
    for persona_name, results in by_persona.items():
        grades = [r.get("grade", "?") for r in results if r.get("grade")]
        grade = grades[0] if grades else "?"
        insights = []
        for r in results:
            insights.extend(r.get("critical_issues", r.get("insights", []))[:1])
        finding = insights[0][:80] if insights else "No critical findings"
        lines.append(f"| {persona_name} | {grade} | {finding} |")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

async def main(args: argparse.Namespace):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable first.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    t0 = time.time()

    # ── Extract data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("ALPHABEAN AGENT SWARM — Multi-Expert Trading Analysis")
    print("=" * 60)
    print("\n[1/4] Extracting trading data...")
    data = extract_all_data()
    print(f"  Closed trades: {data.get('total_closed', 0)}")
    print(f"  Active trades: {data.get('total_active', 0)}")
    print(f"  Patterns: {len(data.get('pattern_stats', {}))}")
    print(f"  Symbols: {len(data.get('symbol_stats', {}))}")
    print(f"  Trading days: {len(data.get('daily_pnl', {}))}")
    print(f"  Total R: {data.get('summary', {}).get('total_r', 0):+.1f}R")

    # ── Build agent runs ──────────────────────────────────────────────────────
    agents_per = args.agents_per_persona
    personas = EXPERT_PERSONAS
    total_agents = len(personas) * agents_per

    print(f"\n[2/4] Launching {total_agents} agents ({len(personas)} personas × {agents_per} runs each)...")
    print(f"  Model: {ANALYSIS_MODEL}")
    print(f"  Concurrency: {MAX_CONCURRENT}")

    # Build all tasks
    tasks = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    for persona in personas:
        slices = build_data_slices(data, persona, agents_per)
        for run_id, data_slice in enumerate(slices):
            tasks.append(run_single_agent(client, semaphore, persona, data_slice, run_id))

    # Run all agents
    results = []
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        status = "OK" if result["success"] else "FAIL"
        if completed % 10 == 0 or completed == total_agents:
            print(f"  [{completed}/{total_agents}] {result['persona_name']} #{result['run_id']} — "
                  f"{status} ({result['elapsed']:.1f}s)")

    successes = sum(1 for r in results if r["success"])
    failures = total_agents - successes
    total_tokens_in = sum(r["tokens_in"] for r in results)
    total_tokens_out = sum(r["tokens_out"] for r in results)
    print(f"\n  Results: {successes} success, {failures} failed")
    print(f"  Tokens: {total_tokens_in:,} in + {total_tokens_out:,} out")

    # ── Synthesis ─────────────────────────────────────────────────────────────
    synthesis = {"synthesis": {}}
    if not args.no_synthesis:
        print(f"\n[3/4] Running synthesis (Opus)...")
        synthesis = await run_synthesis(client, results, data)
        total_tokens_in += synthesis["tokens_in"]
        total_tokens_out += synthesis["tokens_out"]
        print(f"  Synthesis tokens: {synthesis['tokens_in']:,} in + {synthesis['tokens_out']:,} out")
    else:
        print(f"\n[3/4] Skipping synthesis (--no-synthesis)")

    # ── Save reports ──────────────────────────────────────────────────────────
    elapsed_min = (time.time() - t0) / 60
    # Rough cost estimate: sonnet input $3/M, output $15/M; opus input $15/M, output $75/M
    est_cost = (total_tokens_in * 3 / 1_000_000) + (total_tokens_out * 15 / 1_000_000)

    meta = {
        "total_agents": total_agents,
        "personas": len(personas),
        "agents_per_persona": agents_per,
        "successes": successes,
        "failures": failures,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "estimated_cost": round(est_cost, 2),
        "elapsed_min": round(elapsed_min, 1),
        "models": {"analysis": ANALYSIS_MODEL, "synthesis": SYNTHESIS_MODEL},
        "run_date": datetime.now().isoformat(),
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    # Full JSON report
    json_path = reports_dir / f"swarm_report_{timestamp}.json"
    json_path.write_text(json.dumps({
        "meta": meta,
        "synthesis": synthesis.get("synthesis", {}),
        "agent_results": results,
    }, indent=2, default=str))

    # Markdown report
    md_path = reports_dir / f"swarm_report_{timestamp}.md"
    md_content = format_markdown_report(synthesis, results, meta)
    md_path.write_text(md_content)

    print(f"\n[4/4] Reports saved:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    print(f"\n{'=' * 60}")
    print(f"  Total time: {elapsed_min:.1f} minutes")
    print(f"  Estimated cost: ~${est_cost:.2f}")
    print(f"  Grade: {synthesis.get('synthesis', {}).get('system_grade', '(run synthesis to see)')}")
    print(f"{'=' * 60}")


def cli():
    parser = argparse.ArgumentParser(description="AlphaBean Agent Swarm Analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 agent per persona (15 total)")
    parser.add_argument("--agents-per-persona", type=int, default=12,
                        help="Number of agent runs per persona (default: 12, total = 15 × N)")
    parser.add_argument("--no-synthesis", action="store_true",
                        help="Skip the final Opus synthesis pass")
    parser.add_argument("--concurrency", type=int, default=30,
                        help="Max parallel API calls (default: 30)")
    args = parser.parse_args()

    if args.quick:
        args.agents_per_persona = 1

    global MAX_CONCURRENT  # noqa
    MAX_CONCURRENT = args.concurrency

    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
