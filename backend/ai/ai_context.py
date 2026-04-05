"""
backend/ai/ai_context.py — AI Context Assembly Pipeline

Builds a structured briefing string sent to the Qwen3 trade analysis agent.
Each section is independently computed and gracefully degrades on failure.

SECTIONS (in order):
  1. [SETUP MATH]      Entry, stop, targets in $ and ATR multiples
  2. [PATTERN HISTORY] Historical backtest stats for this specific pattern
  3. [TECHNICAL]       Price structure, volatility, S/R levels, momentum
  4. [SCORING]         Multi-factor score breakdown with weights
  5. [MARKET]          SPY regime interpretation + recent macro headlines
  6. [NEWS]            Per-ticker headlines with Polygon AI sentiment
  7. [RISK ASSESSMENT] Flagged risks (earnings, IV, correlation, volume, ATR)
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

BACKTEST_CACHE = Path("cache/backtest_results.json")


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
            bar = "█" * int(val / 10) + "░" * (10 - int(val / 10))
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
# FULL CONTEXT BUILDER
# ==============================================================================

def build_full_context(
    symbol: str,
    setup_dict: dict,
    regime_str: str = "unknown",
    structures=None,
    bars=None,
    max_headlines: int = 8,
) -> str:
    """
    Assemble the complete AI briefing from all context sections.
    Each section degrades gracefully on failure — never raises.
    """
    sections = []

    def _safe(fn, *args, **kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"[{fn.__name__} failed: {e}]\n"

    sections.append(_safe(get_setup_math_context, setup_dict))
    sections.append(_safe(get_pattern_history_context, setup_dict.get("pattern_name", "")))
    if structures is not None:
        sections.append(_safe(get_technical_context, structures))
    elif bars is not None:
        sections.append(_safe(_technical_from_bars, symbol, bars))
    sections.append(_safe(get_scoring_context, setup_dict))
    sections.append(_safe(get_market_context, regime_str))
    sections.append(_safe(get_headline_context, symbol, max_headlines))
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
