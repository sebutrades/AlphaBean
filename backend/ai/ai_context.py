"""
backend/ai/ai_context.py — AI Context Assembly Pipeline

Builds a structured briefing string sent to the Qwen3 trade analysis agent.
Each section is independently computed and gracefully degrades on failure.

SECTIONS (in order):
  1. [SETUP MATH]           Entry, stop, targets in $ and ATR multiples
  2. [PATTERN HISTORY]      Aggregate backtest stats for this pattern (all symbols)
  3. [SYMBOL PERFORMANCE]   How THIS pattern performs on THIS specific stock
  4. [PRICE ACTION]         Candlestick narrative over the relevant lookback window
  5. [TECHNICAL]            Price vs SMAs, volatility regime, S/R, RVOL
  6. [SCORING]              Multi-factor score breakdown with weights
  7. [STRATEGY HOT/COLD]    Rolling live performance of this pattern (StrategyEvaluator)
  8. [MARKET]               SPY regime interpretation + recent macro headlines
  9. [NEWS]                 Per-ticker headlines with Polygon AI sentiment
  10. [RISK ASSESSMENT]     Flagged risks (earnings, IV, correlation, volume, ATR)
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

BACKTEST_CACHE = Path("cache/backtest_results.json")
SYM_STATS_DIR = Path("cache/backtest_by_symbol")

# Bars to include in price action narrative per timeframe
_PA_WINDOW = {"5min": 24, "15min": 16, "1h": 16, "1d": 20}


# ==============================================================================
# 1. MATHEMATICAL SETUP CONTEXT
# ==============================================================================

def get_setup_math_context(setup: dict) -> str:
    """
    Express the trade setup in precise mathematical terms.
    Tells the AI exactly what it's risking and what it stands to gain,
    in both dollar amounts and ATR multiples.
    """
    lines = ["[SETUP MATH]"]

    sym    = setup.get("symbol", "?")
    bias   = setup.get("bias", "long").upper()
    entry  = float(setup.get("entry_price", 0) or 0)
    stop   = float(setup.get("stop_loss", 0) or 0)
    t1     = float(setup.get("target_1", 0) or setup.get("target_price", 0) or 0)
    t2     = float(setup.get("target_2", 0) or t1)
    atr    = float(setup.get("current_atr", 0) or 0)
    desc   = setup.get("description", "")

    if entry <= 0:
        return "[SETUP MATH]\nInsufficient price data.\n"

    lines.append(f"Symbol: {sym} | Direction: {bias}")
    lines.append(f"Entry:  ${entry:.2f}")
    lines.append(f"Stop:   ${stop:.2f}  ({abs(entry - stop):.2f} below entry, "
                 f"{abs(entry - stop)/entry*100:.1f}% risk)")

    risk_dollar = abs(entry - stop)

    if t1 > 0:
        reward_t1 = abs(t1 - entry)
        rr1 = reward_t1 / risk_dollar if risk_dollar > 0 else 0
        lines.append(f"T1:     ${t1:.2f}  ({reward_t1:.2f} from entry, {rr1:.2f}R)")
    if t2 > 0 and t2 != t1:
        reward_t2 = abs(t2 - entry)
        rr2 = reward_t2 / risk_dollar if risk_dollar > 0 else 0
        lines.append(f"T2:     ${t2:.2f}  ({reward_t2:.2f} from entry, {rr2:.2f}R)")

    if atr > 0:
        stop_atr = risk_dollar / atr
        t1_atr   = abs(t1 - entry) / atr if t1 > 0 else 0
        t2_atr   = abs(t2 - entry) / atr if t2 > 0 else 0
        lines.append(
            f"In ATR multiples — Stop: {stop_atr:.2f}× ATR, "
            f"T1: {t1_atr:.2f}× ATR, T2: {t2_atr:.2f}× ATR  (ATR=${atr:.2f})"
        )
        if stop_atr > 2.0:
            lines.append("⚠ Stop is unusually wide (>2 ATR) — reduces position size needed for 1R risk")
        elif stop_atr < 0.3:
            lines.append("⚠ Stop is very tight (<0.3 ATR) — high probability of stop-out noise")

    if desc:
        lines.append(f"Signal detail: {desc}")

    # Pattern splits
    splits = setup.get("position_splits", [])
    if splits and len(splits) >= 2:
        lines.append(
            f"Exit plan: {splits[0]:.0%} at T1, "
            f"{splits[1]:.0%} at T2, "
            f"{splits[2]:.0%} trail (if 3 splits provided)"
            if len(splits) >= 3 else
            f"Exit plan: {splits[0]:.0%} at T1, {splits[1]:.0%} remainder"
        )

    return "\n".join(lines) + "\n"


# ==============================================================================
# 2. PATTERN HISTORY CONTEXT
# ==============================================================================

def get_pattern_history_context(pattern_name: str) -> str:
    """Load backtest stats for this pattern and provide concrete guidance."""
    stats = _load_pattern_stats(pattern_name)
    lines = [f"[PATTERN HISTORY — {pattern_name}]"]

    if not stats or stats.get("total_signals", 0) == 0:
        lines.append("No backtest data available for this pattern.")
        lines.append("Treat as unproven — require very strong confluence from other factors.")
        return "\n".join(lines) + "\n"

    n     = stats.get("total_signals", 0)
    wr    = stats.get("win_rate", 0)
    pf    = stats.get("profit_factor", 0)
    exp   = stats.get("expectancy", 0)
    aw    = stats.get("avg_win_r", 0)
    al    = stats.get("avg_loss_r", 0)
    edge  = stats.get("edge_score", 0)
    t1_r  = stats.get("t1_hit_rate")
    t2_r  = stats.get("t2_hit_rate")
    bars  = stats.get("avg_bars_to_resolution", 0)

    lines.append(f"Sample: {n} signals | Win rate: {wr:.1f}% | Profit factor: {pf:.2f}")
    lines.append(f"Expectancy: {exp:+.3f}R/trade | Avg win: +{aw:.2f}R | Avg loss: -{al:.2f}R")
    lines.append(f"Edge score: {edge:.0f}/100 | Avg bars to resolution: {bars:.1f}")
    if t1_r is not None and t2_r is not None:
        lines.append(f"T1 hit rate: {t1_r:.1f}% | T2 hit rate: {t2_r:.1f}%")

    # Kelly sizing guidance — wr stored as 0-100, convert to fraction
    if wr > 0 and pf > 0 and n >= 10:
        win_pct = wr / 100.0
        loss_pct = 1 - win_pct
        win_r = aw
        loss_r = al if al > 0 else 1.0
        kelly = (win_pct / loss_r) - (loss_pct / win_r) if win_r > 0 else 0
        kelly = max(0, min(kelly, 0.25))   # Cap at 25% of capital
        lines.append(f"Kelly fraction: {kelly:.1%} of capital per trade")

    # Assessment
    if exp > 0.20:
        lines.append("Assessment: STRONG positive edge — historically highly profitable.")
    elif exp > 0.05:
        lines.append("Assessment: Solid positive edge — take when other factors confirm.")
    elif exp > 0:
        lines.append("Assessment: Marginal edge — needs above-average setup quality to be worth taking.")
    elif exp > -0.05:
        lines.append("Assessment: Near breakeven — the individual setup quality is the deciding factor.")
    else:
        lines.append("Assessment: NEGATIVE edge historically — only take with exceptional confluence.")

    return "\n".join(lines) + "\n"


# ==============================================================================
# 3. TECHNICAL STRUCTURE CONTEXT
# ==============================================================================

def get_technical_context(s) -> str:
    """
    Build technical structure summary from extracted structures object.
    s: ExtractedStructures from classifier.py extract_structures()
    """
    if s is None or s.n < 10:
        return "[TECHNICAL]\nInsufficient bar data for technical analysis.\n"

    lines = [f"[TECHNICAL — {s.symbol}, {s.timeframe}]"]

    cur = float(s.closes[-1])
    atr = float(s.current_atr) if s.current_atr else 0

    lines.append(f"Current price: ${cur:.2f}  |  ATR(14): ${atr:.2f} ({atr/cur*100:.2f}% of price)")

    # Moving averages + position
    for period, label in [(20, "SMA20"), (50, "SMA50"), (200, "SMA200")]:
        if s.n >= period + 1:
            sma = float(np.mean(s.closes[-period:]))
            pct = (cur - sma) / sma * 100
            direction = "above" if cur > sma else "below"
            lines.append(f"{label}: ${sma:.2f}  (price {direction} by {abs(pct):.1f}%)")

    # Recent returns
    for lookback, label in [(5, "5-bar"), (20, "20-bar"), (60, "60-bar")]:
        if s.n >= lookback + 1:
            ret = (cur - s.closes[-lookback - 1]) / s.closes[-lookback - 1] * 100
            lines.append(f"{label} return: {ret:+.1f}%")

    # Volatility regime
    if s.n >= 60:
        recent_rets = np.diff(np.array(s.closes[-61:], dtype=float)) / np.array(s.closes[-61:-1], dtype=float)
        vol_recent = float(np.std(recent_rets)) * np.sqrt(252)
        if s.n >= 252:
            long_rets = np.diff(np.array(s.closes[-253:], dtype=float)) / np.array(s.closes[-253:-1], dtype=float)
            vol_long = float(np.std(long_rets)) * np.sqrt(252)
            ratio = vol_recent / vol_long if vol_long > 0 else 1.0
            if ratio < 0.70:
                regime = f"COMPRESSED ({ratio:.0%} of 1yr avg) — favors breakout setups"
            elif ratio > 1.40:
                regime = f"EXPANDED ({ratio:.0%} of 1yr avg) — wider stops needed; reversals possible"
            else:
                regime = f"Normal ({ratio:.0%} of 1yr avg)"
            lines.append(f"Volatility regime: {regime}")
        else:
            lines.append(f"60-bar annualized vol: {vol_recent:.0%}")

    # Volume
    if s.n >= 20 and len(s.volumes) >= 20:
        avg_vol = float(np.mean(np.array(s.volumes[-20:], dtype=float)))
        cur_vol = float(s.volumes[-1])
        rvol = cur_vol / avg_vol if avg_vol > 0 else 1.0
        vol_label = "ELEVATED" if rvol > 1.5 else ("light" if rvol < 0.7 else "normal")
        lines.append(f"Volume: {cur_vol:,.0f}  (RVOL {rvol:.1f}x vs 20-bar avg — {vol_label})")

    # Support and resistance levels
    sr_levels = getattr(s, 'sr_levels', []) or []
    above = sorted([l for l in sr_levels if l.price > cur], key=lambda x: x.price)[:2]
    below = sorted([l for l in sr_levels if l.price < cur], key=lambda x: x.price, reverse=True)[:2]
    if above:
        res_str = " | ".join(f"${l.price:.2f}({l.touches}t)" for l in above)
        lines.append(f"Resistance above: {res_str}")
    if below:
        sup_str = " | ".join(f"${l.price:.2f}({l.touches}t)" for l in below)
        lines.append(f"Support below:    {sup_str}")

    return "\n".join(lines) + "\n"


# ==============================================================================
# 4. SCORING CONTEXT
# ==============================================================================

def get_scoring_context(setup: dict) -> str:
    """Format the multi-factor score breakdown with weight context."""
    scoring = setup.get("scoring", {})
    composite = setup.get("composite_score", 0)
    pre_ai = setup.get("composite_score_pre_ai", composite)

    lines = [f"[SCORING — Composite: {composite:.0f}/100]"]
    if pre_ai != composite:
        lines[0] += f"  (pre-AI: {pre_ai:.0f})"

    factor_map = [
        ("pattern_confidence", "Pattern confidence",  "20%"),
        ("feature_score",      "Feature score",       "25%"),
        ("strategy_score",     "Strategy hot score",  "20%"),
        ("regime_alignment",   "Regime alignment",    "15%"),
        ("backtest_edge",      "Backtest edge",       "10%"),
        ("volume_confirm",     "Volume confirmation", "10%"),
    ]
    for key, label, weight in factor_map:
        val = scoring.get(key)
        if val is not None:
            filled = max(0, min(10, int(val / 10)))
            bar = "█" * filled + "░" * (10 - filled)
            lines.append(f"  {label:<24} {bar}  {val:.0f}/100  (wt {weight})")

    return "\n".join(lines) + "\n"


# ==============================================================================
# 5. MARKET CONTEXT
# ==============================================================================

def get_market_context(regime_str: str = "unknown") -> str:
    """Market regime + interpretation + recent macro headlines."""
    lines = ["[MARKET CONTEXT]"]
    lines.append(f"SPY regime: {regime_str.upper()}")

    regime_guide = {
        "trending_bull":  "Uptrend intact. Favor breakout longs and momentum continuation. Mean reversion longs less reliable. Short setups are against the trend — reduce size.",
        "trending_bear":  "Downtrend. Favor short momentum and break-of-support setups. Long breakouts have lower follow-through. Quality/defensive names outperform.",
        "mean_reverting": "Range-bound/choppy. Breakouts frequently fail. Favor mean-reversion entries near extremes. Tight stops on all trades.",
        "high_volatility":"Elevated VIX. Gaps and reversals dominate. Widen stops beyond normal ATR targets. Reduce position size 25-50%.",
        "mixed":          "No clear regime. Mixed signals. Use individual setup quality as primary filter. Prefer high-composite-score setups only.",
    }
    lines.append(regime_guide.get(regime_str.lower(), f"{regime_str} — use individual setup quality as primary filter."))

    # Recent market headlines
    try:
        from backend.news.pipeline import fetch_market_news
        mkt = fetch_market_news(10)
        if mkt:
            lines.append("\nRecent macro headlines:")
            for item in mkt[:5]:
                lines.append(f"  • {item.headline}  [{item.source}]")
    except Exception:
        pass

    return "\n".join(lines) + "\n"


# ==============================================================================
# 6. NEWS CONTEXT
# ==============================================================================

def get_headline_context(symbol: str, max_headlines: int = 8) -> str:
    """
    Fetch headlines using Polygon news client (richer — has AI sentiment per ticker).
    Falls back to Finnhub/RSS if Polygon fails.
    """
    lines = [f"[NEWS — {symbol}]"]

    # Try Polygon first (has per-ticker AI sentiment reasoning)
    polygon_ok = False
    try:
        from backend.data.news_client import fetch_polygon_news, format_news_context, aggregate_sentiment
        items = fetch_polygon_news(symbol, limit=max_headlines)
        if items:
            agg = aggregate_sentiment(items, symbol)
            lines.append(
                f"Sentiment summary: net={agg['net']:+.2f}  "
                f"({agg['positive']}+ / {agg['negative']}- / {agg['neutral']}=  "
                f"from {agg['total']} articles)"
            )
            lines.append(format_news_context(items, symbol=symbol, max_items=max_headlines))
            polygon_ok = True
    except Exception:
        pass

    # Fall back to Finnhub/RSS
    if not polygon_ok:
        try:
            from backend.news.pipeline import fetch_news, format_headlines_for_llm
            items = fetch_news(symbol, max_headlines)
            if items:
                lines.append(format_headlines_for_llm(items))
            else:
                lines.append("No recent news found.")
        except Exception:
            lines.append("News unavailable.")

    return "\n".join(lines) + "\n"


# ==============================================================================
# 7. RISK ASSESSMENT CONTEXT
# ==============================================================================

def get_risk_assessment_context(setup: dict, structures=None) -> str:
    """
    Identify and flag concrete risks that should affect position sizing or entry.
    These become inputs to the AI's risk_flags list.
    """
    lines = ["[RISK ASSESSMENT]"]
    flags = []

    entry  = float(setup.get("entry_price", 0) or 0)
    stop   = float(setup.get("stop_loss", 0) or 0)
    atr    = float(setup.get("current_atr", 0) or 0)
    symbol = setup.get("symbol", "")
    desc   = setup.get("description", "").lower()
    bias   = setup.get("bias", "long").lower()
    conf   = float(setup.get("confidence", 0) or 0)
    scoring = setup.get("scoring", {})

    risk_dollar = abs(entry - stop)

    # 1. ATR-relative stop width
    if atr > 0 and risk_dollar > 0:
        stop_atr_mult = risk_dollar / atr
        if stop_atr_mult > 2.5:
            flags.append(f"Wide stop ({stop_atr_mult:.1f}× ATR) — position must be sized down proportionally")
        elif stop_atr_mult < 0.25:
            flags.append(f"Tight stop ({stop_atr_mult:.2f}× ATR) — high noise stop-out risk in normal price action")

    # 2. Low volume confirmation
    vol_score = scoring.get("volume_confirm", 60)
    if vol_score < 30:
        flags.append("Very low volume (<30/100) — no institutional participation; high risk of false breakout")
    elif vol_score < 45:
        flags.append(f"Below-average volume ({vol_score:.0f}/100) — reduced conviction; size down")

    # 3. Regime misalignment
    regime_score = scoring.get("regime_alignment", 60)
    if regime_score < 35:
        flags.append("Low regime alignment — this strategy type historically underperforms current regime")

    # 4. Low backtest edge
    bt_score = scoring.get("backtest_edge", 50)
    if bt_score < 30:
        flags.append("Weak or unproven backtest edge — pattern lacks sufficient historical validation")

    # 5. Strategy score (hot score)
    strat_score = scoring.get("strategy_score", 50)
    if strat_score < 35:
        flags.append("Strategy cold — this pattern has been underperforming in recent live signals")

    # 6. Earnings/event risk detection (from description)
    event_keywords = ["earnings", "fda", "clinical", "approval", "merger", "acquisition", "split"]
    for kw in event_keywords:
        if kw in desc:
            flags.append(f"Potential event risk: '{kw}' detected in setup description — verify earnings dates")
            break

    # 7. Correlation risk (from setup if available)
    spy_corr = setup.get("spy_correlation", {})
    if isinstance(spy_corr, dict):
        corr_val = spy_corr.get("correlation", spy_corr.get("r", None))
        if corr_val is not None:
            corr_val = float(corr_val)
            if corr_val > 0.85 and bias == "long":
                flags.append(f"High SPY correlation ({corr_val:.2f}) — adverse market move will amplify loss")
            elif corr_val < -0.60 and bias == "short":
                flags.append(f"Negative SPY correlation ({corr_val:.2f}) — market rally could squeeze short")

    # 8. Technical context risks
    if structures is not None:
        try:
            cur = float(structures.closes[-1])
            # Proximity to round numbers (psychological resistance)
            nearest_round = round(cur / 50) * 50
            if abs(cur - nearest_round) / cur < 0.01:
                flags.append(f"Price near major round number (${nearest_round:.0f}) — potential resistance/support")

            # Extended from SMA
            if structures.n >= 50:
                sma50 = float(np.mean(structures.closes[-50:]))
                ext_pct = abs(cur - sma50) / sma50 * 100
                if ext_pct > 20 and bias == "long":
                    flags.append(f"Price {ext_pct:.0f}% extended from 50 SMA — mean-reversion risk elevated")
        except Exception:
            pass

    if not flags:
        lines.append("No major risk flags identified.")
    else:
        for f in flags:
            lines.append(f"⚠ {f}")

    lines.append(f"\nRisk flag count: {len(flags)}/8")
    return "\n".join(lines) + "\n"


# ==============================================================================
# TECHNICAL CONTEXT FROM BARS (fallback when no ExtractedStructures available)
# ==============================================================================

def _technical_from_bars(symbol: str, bars) -> str:
    """Build technical context from a BarSeries when no ExtractedStructures available."""
    if bars is None or len(bars.bars) < 20:
        return "[TECHNICAL]\nInsufficient bar data for technical analysis.\n"

    b = bars.bars
    n = len(b)
    closes  = np.array([x.close  for x in b], dtype=np.float64)
    highs   = np.array([x.high   for x in b], dtype=np.float64)
    lows    = np.array([x.low    for x in b], dtype=np.float64)
    volumes = np.array([x.volume for x in b], dtype=np.float64)
    cur = float(closes[-1])

    # ATR(14)
    atr = 0.0
    if n >= 15:
        tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
        atr = float(np.mean(tr[-14:]))

    lines = [f"[TECHNICAL — {symbol}]"]
    atr_pct = atr / cur * 100 if cur > 0 else 0
    lines.append(f"Current price: ${cur:.2f}  |  ATR(14): ${atr:.2f} ({atr_pct:.2f}% of price)")

    # Moving averages
    for period, label in [(20, "SMA20"), (50, "SMA50"), (200, "SMA200")]:
        if n >= period + 1:
            sma = float(np.mean(closes[-period:]))
            pct = (cur - sma) / sma * 100
            direction = "above" if cur > sma else "below"
            lines.append(f"{label}: ${sma:.2f}  (price {direction} by {abs(pct):.1f}%)")

    # Recent returns
    for lookback, label in [(5, "5-bar"), (20, "20-bar"), (60, "60-bar")]:
        if n >= lookback + 1:
            ret = (cur - closes[-lookback - 1]) / closes[-lookback - 1] * 100
            lines.append(f"{label} return: {ret:+.1f}%")

    # Volatility regime
    if n >= 60:
        recent_rets = np.diff(closes[-61:]) / closes[-61:-1]
        vol_60 = float(np.std(recent_rets)) * np.sqrt(252)
        if n >= 252:
            yr_rets = np.diff(closes[-253:]) / closes[-253:-1]
            vol_yr  = float(np.std(yr_rets)) * np.sqrt(252)
            ratio   = vol_60 / vol_yr if vol_yr > 0 else 1.0
            if ratio < 0.70:
                lines.append(f"Volatility: COMPRESSED ({ratio:.0%} of 1yr avg) — favors breakout setups")
            elif ratio > 1.40:
                lines.append(f"Volatility: EXPANDED ({ratio:.0%} of 1yr avg) — wider stops needed")
            else:
                lines.append(f"Volatility: Normal ({ratio:.0%} of 1yr avg)")

    # Volume RVOL
    if n >= 20:
        avg_vol = float(np.mean(volumes[-20:]))
        cur_vol = float(volumes[-1])
        rvol    = cur_vol / avg_vol if avg_vol > 0 else 1.0
        vol_lbl = "ELEVATED" if rvol > 1.5 else ("light" if rvol < 0.7 else "normal")
        lines.append(f"Volume: {cur_vol:,.0f}  (RVOL {rvol:.1f}x — {vol_lbl})")

    return "\n".join(lines) + "\n"


# ==============================================================================
# 8. SYMBOL-SPECIFIC PATTERN PERFORMANCE
# ==============================================================================

def get_symbol_pattern_context(symbol: str, pattern_name: str) -> str:
    """
    Show how THIS specific pattern has performed on THIS stock historically.
    Sourced from per-symbol backtest data saved by run_backtest.py.
    More actionable than aggregate stats — a pattern may work well overall
    but consistently fail on a specific stock (sector mismatch, price behaviour, etc.)
    """
    lines = [f"[SYMBOL PERFORMANCE — {symbol}]"]

    path = SYM_STATS_DIR / f"{symbol}.json"
    if not path.exists():
        lines.append("No per-symbol backtest data yet — aggregate stats apply.")
        return "\n".join(lines) + "\n"

    try:
        data = json.loads(path.read_text())
    except Exception:
        lines.append("Could not load per-symbol data.")
        return "\n".join(lines) + "\n"

    patterns = data.get("patterns", {})
    if not patterns:
        lines.append("No signals recorded for this stock in last backtest.")
        return "\n".join(lines) + "\n"

    # Highlight the current pattern on this stock
    this = patterns.get(pattern_name)
    if this and this.get("total_signals", 0) >= 2:
        n = this["total_signals"]; wr = this["win_rate"]; exp = this.get("expectancy", 0)
        pf = this.get("profit_factor", 0); edge = this.get("edge_score", 0)
        lines.append(f"{pattern_name} on {symbol}: {n} signals | WR {wr:.1f}% | PF {pf:.2f} | Exp {exp:+.3f}R | Edge {edge:.0f}/100")
        if exp > 0.15:
            lines.append("  → Strong positive edge on this specific stock — historically reliable here")
        elif exp > 0:
            lines.append("  → Marginal positive edge — proceed normally")
        elif exp > -0.05:
            lines.append("  → Near breakeven on this stock — individual setup quality is the deciding factor")
        else:
            lines.append("  → NEGATIVE edge on this stock — pattern struggles here despite aggregate stats; reduce size")
    elif this:
        lines.append(f"{pattern_name} on {symbol}: {this.get('total_signals', 0)} signals (insufficient sample)")
    else:
        lines.append(f"{pattern_name} has not fired on {symbol} in the backtest period.")

    # Top 3 best patterns on this stock
    sorted_pats = sorted(
        [(n, s) for n, s in patterns.items() if s.get("total_signals", 0) >= 3],
        key=lambda x: x[1].get("edge_score", 0), reverse=True
    )[:3]
    if sorted_pats:
        lines.append(f"\nTop patterns on {symbol} historically:")
        for rank, (pname, s) in enumerate(sorted_pats, 1):
            marker = " ★" if pname == pattern_name else ""
            lines.append(
                f"  #{rank} {pname}{marker}: {s['total_signals']} signals, "
                f"WR {s['win_rate']:.0f}%, Exp {s.get('expectancy', 0):+.3f}R, "
                f"Edge {s.get('edge_score', 0):.0f}/100"
            )

    return "\n".join(lines) + "\n"


# ==============================================================================
# 9. PRICE ACTION NARRATIVE
# ==============================================================================

def get_price_action_narrative(symbol: str, timeframe: str, bars) -> str:
    """
    Describe recent price action in concrete narrative terms.
    Window size is calibrated to the setup's timeframe:
      5min → 24 bars (2 hours)  |  15min → 16 bars (4 hours)
      1h   → 16 bars (2 days)   |  1d    → 20 bars (1 month)
    """
    n_bars = _PA_WINDOW.get(timeframe, 20)
    if bars is None or len(bars.bars) < n_bars + 3:
        return f"[PRICE ACTION — {symbol}]\nInsufficient bar data.\n"

    b = bars.bars[-n_bars:]
    closes  = np.array([x.close  for x in b], dtype=np.float64)
    highs   = np.array([x.high   for x in b], dtype=np.float64)
    lows    = np.array([x.low    for x in b], dtype=np.float64)
    opens   = np.array([x.open   for x in b], dtype=np.float64)
    volumes = np.array([x.volume for x in b], dtype=np.float64)

    lines = [f"[PRICE ACTION — {symbol}, last {n_bars} {timeframe} bars]"]

    # Overall movement
    total_change = (closes[-1] - closes[0]) / closes[0] * 100
    lines.append(f"Period move: {total_change:+.1f}%  (${closes[0]:.2f} → ${closes[-1]:.2f})")

    # Trend structure — count HH/HL vs LH/LL in last 5 candles
    look = min(5, n_bars - 1)
    hh = sum(1 for i in range(1, look + 1) if highs[-i] > highs[-i - 1])
    lh = sum(1 for i in range(1, look + 1) if highs[-i] < highs[-i - 1])
    hl = sum(1 for i in range(1, look + 1) if lows[-i]  > lows[-i - 1])
    ll = sum(1 for i in range(1, look + 1) if lows[-i]  < lows[-i - 1])

    if hh >= 3 and hl >= 2:
        structure = "Higher highs + higher lows → uptrend structure"
    elif lh >= 3 and ll >= 2:
        structure = "Lower highs + lower lows → downtrend structure"
    elif hh >= 2 and ll >= 2:
        structure = "Mixed swings → consolidation or reversal in progress"
    else:
        structure = "Indeterminate structure"
    lines.append(f"Structure: {structure}")

    # Last 3 candles
    lines.append("Last 3 candles:")
    for i in [-3, -2, -1]:
        bar = bars.bars[i]
        body    = float(closes[i] - opens[i])
        body_pct = abs(body / opens[i] * 100) if opens[i] > 0 else 0
        rng     = float(highs[i] - lows[i])
        close_pos = (closes[i] - lows[i]) / rng if rng > 0 else 0.5
        direction = "bullish" if body > 0 else "bearish" if body < 0 else "doji"
        position  = "upper" if close_pos > 0.67 else ("lower" if close_pos < 0.33 else "middle")
        upper_wick = float(highs[i] - max(opens[i], closes[i]))
        lower_wick = float(min(opens[i], closes[i]) - lows[i])
        wick_note = ""
        if rng > 0:
            if upper_wick / rng > 0.40:
                wick_note = " (long upper wick — rejection of highs)"
            elif lower_wick / rng > 0.40:
                wick_note = " (long lower wick — rejection of lows / hammer)"
        ts = bar.timestamp.strftime("%m/%d %H:%M") if hasattr(bar.timestamp, "strftime") else str(bar.timestamp)
        lines.append(
            f"  {ts}: {direction} {body_pct:.1f}% body, "
            f"closed {position} third{wick_note}"
        )

    # Volume trend
    avg_vol = float(np.mean(volumes))
    recent_vol = float(np.mean(volumes[-3:]))
    vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
    if vol_ratio > 1.3:
        lines.append(f"Volume: expanding (+{(vol_ratio - 1) * 100:.0f}% vs period avg) — momentum confirmation")
    elif vol_ratio < 0.7:
        lines.append(f"Volume: contracting ({(1 - vol_ratio) * 100:.0f}% below avg) — possible squeeze / fading interest")
    else:
        lines.append(f"Volume: stable (RVOL {vol_ratio:.1f}x vs period avg)")

    # Gap detection for daily
    if timeframe == "1d" and len(bars.bars) >= 2:
        prev_close = float(bars.bars[-2].close)
        today_open = float(bars.bars[-1].open)
        gap_pct = (today_open - prev_close) / prev_close * 100
        if abs(gap_pct) > 0.5:
            gap_dir = "up" if gap_pct > 0 else "down"
            lines.append(f"Gap {gap_dir}: {gap_pct:+.2f}% at open vs prior close")

    return "\n".join(lines) + "\n"


# ==============================================================================
# 10. STRATEGY HOT / COLD STATUS
# ==============================================================================

def get_strategy_hot_context(pattern_name: str) -> str:
    """
    Show rolling live performance of this pattern from the StrategyEvaluator.
    This reflects actual live outcomes tracked post-backtest, not historical data.
    Tells the AI whether the pattern is 'hot' (overperforming) or 'cold' (underperforming).
    """
    lines = [f"[STRATEGY HOT/COLD — {pattern_name}]"]
    try:
        from backend.strategies.evaluator import StrategyEvaluator
        ev = StrategyEvaluator()
        ev.load()

        summary = ev.get_pattern_summary(pattern_name)
        if not summary.get("has_data", False):
            lines.append("No live tracking data yet — pattern not yet evaluated in live conditions.")
            return "\n".join(lines) + "\n"

        total    = summary.get("total_signals", 0)
        wr       = summary.get("win_rate", 0)
        hot      = summary.get("hot_score", 40)
        pf       = summary.get("profit_factor", 0)
        exp      = summary.get("expectancy", 0)
        trend    = summary.get("trend", "stable")

        lines.append(f"Live signals tracked: {total}")
        lines.append(f"Win rate: {wr:.1%} | Profit factor: {pf:.2f} | Expectancy: {exp:+.3f}R")
        lines.append(f"Hot score: {hot:.0f}/100 | Trend: {trend}")

        if hot >= 70:
            lines.append("Status: CURRENTLY HOT — outperforming historical expectation")
        elif hot >= 50:
            lines.append("Status: Normal — performing in line with expectation")
        elif hot >= 30:
            lines.append("Status: Cold — running below historical expectation recently")
        else:
            lines.append("Status: VERY COLD — significant recent underperformance; reduce size")

        # Currently hot types for context
        hot_types = ev.get_hot_strategy_types(3)
        if hot_types:
            lines.append(f"Currently hot strategy types: {', '.join(hot_types)}")

    except Exception:
        lines.append("Live tracking data unavailable.")

    return "\n".join(lines) + "\n"


# ==============================================================================
# 11. STOCK STRUCTURAL BEHAVIOR (new)
# ==============================================================================

def get_structural_behavior_context(symbol: str, bars) -> str:
    """
    Summarize the stock's overall structural behavior across multiple timeframes.
    Gives the AI understanding of: trend phase, momentum character, volatility
    profile, relative strength, and how the stock has been behaving overall.
    """
    if bars is None or len(bars.bars) < 30:
        return ""

    import numpy as np
    b = bars.bars
    n = len(b)
    closes  = np.array([x.close  for x in b], dtype=np.float64)
    highs   = np.array([x.high   for x in b], dtype=np.float64)
    lows    = np.array([x.low    for x in b], dtype=np.float64)
    volumes = np.array([x.volume for x in b], dtype=np.float64)
    cur = float(closes[-1])

    lines = [f"[STOCK BEHAVIOR — {symbol}]"]

    # Trend phase classification
    smas = {}
    for p in [20, 50, 200]:
        if n >= p:
            smas[p] = float(np.mean(closes[-p:]))

    if smas:
        above = [p for p, s in smas.items() if cur > s]
        below = [p for p, s in smas.items() if cur <= s]
        if len(above) == len(smas):
            lines.append(f"Trend phase: BULLISH — price above all SMAs ({', '.join(str(p)+'d' for p in sorted(smas))})")
        elif len(below) == len(smas):
            lines.append(f"Trend phase: BEARISH — price below all SMAs ({', '.join(str(p)+'d' for p in sorted(smas))})")
        else:
            above_str = "/".join(str(p)+"d" for p in sorted(above)) if above else "none"
            below_str = "/".join(str(p)+"d" for p in sorted(below)) if below else "none"
            lines.append(f"Trend phase: MIXED — above {above_str}, below {below_str} SMAs")

    # Multi-period returns (momentum character)
    periods = [(5, "1wk"), (20, "1mo"), (60, "3mo"), (120, "6mo"), (252, "1yr")]
    returns = []
    for p, label in periods:
        if n >= p + 1:
            r = (cur - closes[-p-1]) / closes[-p-1] * 100
            returns.append(f"{label}: {r:+.1f}%")
    if returns:
        lines.append("Returns: " + " | ".join(returns))

    # 52-week range position
    lookback = b[-252:] if n >= 252 else b
    hi52 = float(max(x.high for x in lookback))
    lo52 = float(min(x.low  for x in lookback))
    rng = hi52 - lo52
    pct_in_range = (cur - lo52) / rng * 100 if rng > 0 else 50
    if pct_in_range >= 80:
        lines.append(f"52W position: NEAR HIGH — {pct_in_range:.0f}% of annual range (hi=${hi52:.2f} / lo=${lo52:.2f})")
    elif pct_in_range <= 20:
        lines.append(f"52W position: NEAR LOW — {pct_in_range:.0f}% of annual range (hi=${hi52:.2f} / lo=${lo52:.2f})")
    else:
        lines.append(f"52W position: mid-range — {pct_in_range:.0f}% (hi=${hi52:.2f} / lo=${lo52:.2f})")

    # Momentum character: recent vs longer-term performance
    if n >= 21:
        ret_1mo = (cur - closes[-21]) / closes[-21] * 100
        ret_3mo = (cur - closes[min(-61, -n+1)]) / closes[min(-61, -n+1)] * 100 if n >= 61 else None
        if ret_3mo is not None:
            if ret_1mo > 0 and ret_3mo > 0 and ret_1mo > ret_3mo * 0.5:
                lines.append("Momentum: ACCELERATING — recent month outperforming 3-month pace")
            elif ret_1mo < 0 and ret_3mo > 0:
                lines.append("Momentum: DECELERATING — pulling back after longer-term uptrend")
            elif ret_1mo < 0 and ret_3mo < 0:
                lines.append("Momentum: DECLINING — weakness across both 1-month and 3-month windows")

    # Volatility vs history
    if n >= 60:
        recent_rets = np.diff(closes[-21:]) / closes[-21:-1]
        hist_rets   = np.diff(closes[-61:-21]) / closes[-61:-22]
        vol_rec  = float(np.std(recent_rets)) * np.sqrt(252) * 100
        vol_hist = float(np.std(hist_rets))   * np.sqrt(252) * 100
        ratio = vol_rec / vol_hist if vol_hist > 0 else 1.0
        lines.append(f"Volatility: recent {vol_rec:.0f}% ann. | 60-bar hist {vol_hist:.0f}% ann. | ratio {ratio:.2f}x {'(EXPANDING)' if ratio > 1.3 else '(COMPRESSING)' if ratio < 0.7 else '(stable)'}")

    # Volume trend (accumulation vs distribution)
    if n >= 20:
        up_vol   = float(np.mean([volumes[i] for i in range(-20, 0) if closes[i] > closes[i-1]]) or 0)
        down_vol = float(np.mean([volumes[i] for i in range(-20, 0) if closes[i] <= closes[i-1]]) or 0)
        if up_vol > 0 and down_vol > 0:
            vol_bias = up_vol / down_vol
            if vol_bias > 1.3:
                lines.append(f"Volume bias: ACCUMULATION — up-day avg vol {vol_bias:.1f}x > down-day avg (20-bar)")
            elif vol_bias < 0.7:
                lines.append(f"Volume bias: DISTRIBUTION — down-day avg vol {1/vol_bias:.1f}x > up-day avg (20-bar)")
            else:
                lines.append(f"Volume bias: Neutral (up/down vol ratio {vol_bias:.2f})")

    # RSI-14 context
    if n >= 15:
        deltas = np.diff(closes[-15:])
        ag = float(np.mean([d for d in deltas if d > 0]) or 0)
        al = float(np.mean([-d for d in deltas if d < 0]) or 0)
        rsi = 100 - 100 / (1 + ag/al) if al > 0 else 100.0
        context = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "neutral"
        lines.append(f"RSI(14): {rsi:.1f} — {context}")

    return "\n".join(lines) + "\n"


# ==============================================================================
# FULL CONTEXT BUILDER
# ==============================================================================

def build_full_context(
    symbol: str,
    setup_dict: dict,
    regime_str: str = "unknown",
    structures=None,
    bars=None,
    timeframe: str = "1d",
    max_headlines: int = 8,
) -> str:
    """
    Assemble the complete AI briefing from all 10 context sections.
    Each section degrades gracefully on failure — never raises.
    """
    sections = []
    pattern_name = setup_dict.get("pattern_name", "")

    def _safe(fn, *args, **kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"[{fn.__name__} failed: {e}]\n"

    # 1. Mathematical setup
    sections.append(_safe(get_setup_math_context, setup_dict))
    # 2. Aggregate pattern history
    sections.append(_safe(get_pattern_history_context, pattern_name))
    # 3. Symbol-specific performance for this pattern
    sections.append(_safe(get_symbol_pattern_context, symbol, pattern_name))
    # 4. Price action narrative
    if bars is not None:
        sections.append(_safe(get_price_action_narrative, symbol, timeframe, bars))
    # 4b. Stock structural behavior (full trend/momentum/vol profile)
    if bars is not None:
        ctx = _safe(get_structural_behavior_context, symbol, bars)
        if ctx.strip():
            sections.append(ctx)
    # 5. Technical structure
    if structures is not None:
        sections.append(_safe(get_technical_context, structures))
    elif bars is not None:
        sections.append(_safe(_technical_from_bars, symbol, bars))
    # 6. Scoring breakdown
    sections.append(_safe(get_scoring_context, setup_dict))
    # 7. Rolling live hot/cold status
    sections.append(_safe(get_strategy_hot_context, pattern_name))
    # 8. Market regime + macro
    sections.append(_safe(get_market_context, regime_str))
    # 9. News + sentiment
    sections.append(_safe(get_headline_context, symbol, max_headlines))
    # 10. Risk flags
    sections.append(_safe(get_risk_assessment_context, setup_dict, structures))

    return "\n".join(sections)


# ==============================================================================
# HELPERS
# ==============================================================================

def _load_pattern_stats(pattern_name: str) -> dict:
    if not BACKTEST_CACHE.exists():
        return {}
    try:
        data = json.loads(BACKTEST_CACHE.read_text())
        patterns = data.get("patterns", {})
        stats = patterns.get(pattern_name, {})
        # Handle nested TF format — prefer 1d stats for daily patterns
        if "1d" in stats and isinstance(stats["1d"], dict):
            return stats["1d"]
        return stats
    except Exception:
        return {}
