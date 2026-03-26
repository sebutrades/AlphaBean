"""
backend/ai/ai_context.py — AI Context Assembly Pipeline

Collects information from multiple sources and concatenates them into
a single rich context string for the AI trade evaluator.

Each function returns a plain string. They get concatenated together
to form the full briefing that gets sent to the LLM.

CONTEXT SECTIONS:
  1. get_headline_context()      — Recent news headlines + sentiment
  2. get_technical_context()     — Price structure, ATR, SMAs, VWAP
  3. get_pattern_history_context() — How this pattern performs historically
  4. get_market_context()        — SPY regime, correlation, market news
  5. get_setup_context()         — The specific trade setup details
  6. get_scoring_context()       — Multi-factor score breakdown

Usage:
    from backend.ai.ai_context import build_full_context

    context = build_full_context(
        symbol="AAPL",
        setup=scored_setup,
        regime=regime_result,
        bars=bar_series,
    )
    # context is a single string ready to send to the LLM
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.data.news_client import (
    fetch_polygon_news, format_news_context, aggregate_sentiment,
)
from backend.patterns.registry import PATTERN_META


BACKTEST_CACHE = Path("cache/backtest_results.json")


# ==============================================================================
# 1. HEADLINE CONTEXT
# ==============================================================================

def get_headline_context(symbol: str, max_headlines: int = 10) -> str:
    """Fetch and format recent headlines for a symbol.

    Returns a string with headlines, sources, ages, and per-ticker sentiment.
    """
    items = fetch_polygon_news(symbol, limit=max_headlines)

    if not items:
        return f"[HEADLINES]\nNo recent news found for {symbol}.\n"

    formatted = format_news_context(items, symbol=symbol, max_items=max_headlines)
    sentiment = aggregate_sentiment(items, symbol)

    header = f"[HEADLINES — {symbol}] ({sentiment['total']} articles, "
    header += f"net sentiment: {sentiment['net']:+.2f}, "
    header += f"{sentiment['positive']}+ / {sentiment['negative']}- / {sentiment['neutral']}=)"

    return f"{header}\n{formatted}\n"


# ==============================================================================
# 2. TECHNICAL CONTEXT
# ==============================================================================

def get_technical_context(s) -> str:
    """Build technical structure summary from extracted structures.

    Args:
        s: The extracted structures object from extract_structures(bars)
           Has: closes, highs, lows, volumes, current_atr, timestamps,
                zz_highs, zz_lows, support_levels, resistance_levels, etc.

    Returns string describing price structure.
    """
    if s.n < 10:
        return "[TECHNICAL]\nInsufficient data for technical analysis.\n"

    lines = [f"[TECHNICAL — {s.symbol}, {s.timeframe}]"]

    cur = s.closes[-1]
    atr = s.current_atr

    # Price and ATR
    lines.append(f"Current price: ${cur:.2f}")
    lines.append(f"ATR(14): ${atr:.2f} ({atr/cur*100:.1f}% of price)")

    # Moving averages
    if s.n >= 20:
        sma20 = float(np.mean(s.closes[-20:]))
        pct_from_20 = (cur - sma20) / sma20 * 100
        lines.append(f"20 SMA: ${sma20:.2f} (price is {pct_from_20:+.1f}% from it)")

    if s.n >= 50:
        sma50 = float(np.mean(s.closes[-50:]))
        pct_from_50 = (cur - sma50) / sma50 * 100
        lines.append(f"50 SMA: ${sma50:.2f} (price is {pct_from_50:+.1f}% from it)")

    if s.n >= 200:
        sma200 = float(np.mean(s.closes[-200:]))
        pct_from_200 = (cur - sma200) / sma200 * 100
        lines.append(f"200 SMA: ${sma200:.2f} (price is {pct_from_200:+.1f}% from it)")

    # Recent performance
    if s.n >= 5:
        ret_5d = (cur - s.closes[-6]) / s.closes[-6] * 100
        lines.append(f"5-bar return: {ret_5d:+.1f}%")
    if s.n >= 20:
        ret_20d = (cur - s.closes[-21]) / s.closes[-21] * 100
        lines.append(f"20-bar return: {ret_20d:+.1f}%")

    # Volume context
    if s.n >= 20:
        avg_vol = float(np.mean(s.volumes[-20:]))
        cur_vol = s.volumes[-1]
        rvol = cur_vol / avg_vol if avg_vol > 0 else 1.0
        lines.append(f"Volume: {cur_vol:,.0f} (RVOL: {rvol:.1f}x vs 20-bar avg)")

    # Volatility regime
    if s.n >= 60:
        daily_rets = np.diff(s.closes[-61:]) / np.array(s.closes[-61:-1])
        vol_60 = float(np.std(daily_rets))
        if s.n >= 252:
            daily_rets_long = np.diff(s.closes[-253:]) / np.array(s.closes[-253:-1])
            vol_252 = float(np.std(daily_rets_long))
            if vol_252 > 0:
                vol_ratio = vol_60 / vol_252
                if vol_ratio < 0.7:
                    lines.append(f"Volatility: COMPRESSED ({vol_ratio:.0%} of historical)")
                elif vol_ratio > 1.3:
                    lines.append(f"Volatility: EXPANDED ({vol_ratio:.0%} of historical)")
                else:
                    lines.append(f"Volatility: Normal ({vol_ratio:.0%} of historical)")

    # Support/Resistance levels
    if hasattr(s, 'support_levels') and s.support_levels:
        top_support = sorted(s.support_levels, key=lambda l: l.price, reverse=True)[:3]
        sr_str = ", ".join(f"${l.price:.2f}({l.touches}t)" for l in top_support)
        lines.append(f"Key support: {sr_str}")

    if hasattr(s, 'resistance_levels') and s.resistance_levels:
        top_res = sorted(s.resistance_levels, key=lambda l: l.price)[:3]
        sr_str = ", ".join(f"${l.price:.2f}({l.touches}t)" for l in top_res)
        lines.append(f"Key resistance: {sr_str}")

    return "\n".join(lines) + "\n"


# ==============================================================================
# 3. PATTERN HISTORY CONTEXT
# ==============================================================================

def get_pattern_history_context(pattern_name: str) -> str:
    """Load backtest stats for this pattern and format as context.

    Tells the AI how this pattern has performed historically.
    """
    stats = _load_pattern_stats(pattern_name)

    if not stats or stats.get("total_signals", 0) == 0:
        return f"[PATTERN HISTORY — {pattern_name}]\nNo backtest data available.\n"

    lines = [f"[PATTERN HISTORY — {pattern_name}]"]
    lines.append(f"Sample size: {stats.get('total_signals', 0)} signals across 500 symbols, 90 days")
    lines.append(f"Win rate: {stats.get('win_rate', 0):.1f}%")
    lines.append(f"Profit factor: {stats.get('profit_factor', 0):.2f}")
    lines.append(f"Expectancy: {stats.get('expectancy', 0):+.3f}R per trade")
    lines.append(f"Avg win: {stats.get('avg_win_r', 0):.2f}R | Avg loss: {stats.get('avg_loss_r', 0):.2f}R")
    lines.append(f"Edge score: {stats.get('edge_score', 0):.0f}/100")

    # T1/T2 hit rates if available
    t1 = stats.get('t1_hit_rate')
    t2 = stats.get('t2_hit_rate')
    if t1 is not None:
        lines.append(f"T1 hit rate: {t1:.1f}% | T2 hit rate: {t2:.1f}%")

    # Interpret
    exp = stats.get('expectancy', 0)
    if exp > 0.1:
        lines.append("Assessment: STRONG positive edge. This pattern has been profitable.")
    elif exp > 0:
        lines.append("Assessment: Marginal positive edge. Proceed with caution.")
    elif exp > -0.1:
        lines.append("Assessment: Near breakeven. Quality of individual setup matters more than the pattern average.")
    else:
        lines.append("Assessment: NEGATIVE edge historically. Only take with very strong confluence.")

    return "\n".join(lines) + "\n"


# ==============================================================================
# 4. MARKET CONTEXT
# ==============================================================================

def get_market_context(regime_str: str = "unknown") -> str:
    """Build market-level context: regime + recent market headlines.

    Args:
        regime_str: Current market regime string from regime detector
    """
    lines = [f"[MARKET CONTEXT]"]
    lines.append(f"Current regime: {regime_str.upper()}")

    # Regime interpretation
    regime_map = {
        "trending_bull": "Markets trending higher. Momentum and breakout strategies favored. Mean reversion less reliable.",
        "trending_bear": "Markets trending lower. Short momentum favored. Breakout longs risky. Quality/safety factors outperform.",
        "mean_reverting": "Markets range-bound/choppy. Mean reversion strategies favored. Breakouts prone to failure.",
        "high_volatility": "Elevated volatility. Wider stops needed. Reduced position sizing. Gap and reversal plays can work.",
        "mixed": "No clear regime. Mixed signals. Be selective. Look for strongest individual setup quality.",
    }
    interp = regime_map.get(regime_str, "Regime unknown. Use individual setup quality as primary filter.")
    lines.append(f"Interpretation: {interp}")

    # Try to get market headlines
    try:
        from backend.data.news_client import fetch_market_news, format_news_context
        market_items = fetch_market_news(limit=5)
        if market_items:
            lines.append("\nRecent market headlines:")
            lines.append(format_news_context(market_items, max_items=5))
    except Exception:
        pass

    return "\n".join(lines) + "\n"


# ==============================================================================
# 5. SETUP CONTEXT
# ==============================================================================

def get_setup_context(setup_dict: dict) -> str:
    """Format the specific trade setup details.

    Args:
        setup_dict: TradeSetup.to_dict() or ScoredSetup.to_dict()
    """
    lines = [f"[TRADE SETUP]"]

    name = setup_dict.get("pattern_name", "Unknown")
    sym = setup_dict.get("symbol", "")
    bias = setup_dict.get("bias", "").upper()
    entry = setup_dict.get("entry_price", 0)
    stop = setup_dict.get("stop_loss", 0)
    target = setup_dict.get("target_price", 0)
    t1 = setup_dict.get("target_1", 0)
    t2 = setup_dict.get("target_2", 0)
    rr = setup_dict.get("risk_reward_ratio", 0)
    conf = setup_dict.get("confidence", 0)
    desc = setup_dict.get("description", "")

    lines.append(f"Pattern: {name}")
    lines.append(f"Symbol: {sym} | Bias: {bias}")
    lines.append(f"Entry: ${entry:.2f}")
    lines.append(f"Stop: ${stop:.2f}")
    if t1 and t2:
        lines.append(f"Target 1 (partial): ${t1:.2f}")
        lines.append(f"Target 2 (full): ${t2:.2f}")
    else:
        lines.append(f"Target: ${target:.2f}")
    lines.append(f"Risk:Reward: {rr:.1f}")
    lines.append(f"Pattern confidence: {conf:.0%}")

    if desc:
        lines.append(f"Setup detail: {desc}")

    # Key levels
    key_levels = setup_dict.get("key_levels", {})
    if key_levels:
        kl_str = ", ".join(f"{k}=${v:.2f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in key_levels.items())
        lines.append(f"Key levels: {kl_str}")

    return "\n".join(lines) + "\n"


# ==============================================================================
# 6. SCORING CONTEXT
# ==============================================================================

def get_scoring_context(setup_dict: dict) -> str:
    """Format the multi-factor score breakdown."""
    scoring = setup_dict.get("scoring", {})
    composite = setup_dict.get("composite_score", 0)

    if not scoring:
        return f"[SCORING]\nComposite score: {composite:.0f}/100\n"

    lines = [f"[SCORING — Composite: {composite:.0f}/100]"]
    lines.append(f"  Pattern confidence: {scoring.get('pattern_confidence', 'N/A')}/100 (weight: 20%)")
    lines.append(f"  Feature score:      {scoring.get('feature_score', 'N/A')}/100 (weight: 25%)")
    lines.append(f"  Strategy hot score: {scoring.get('strategy_score', 'N/A')}/100 (weight: 20%)")
    lines.append(f"  Regime alignment:   {scoring.get('regime_alignment', 'N/A')}/100 (weight: 15%)")
    lines.append(f"  Backtest edge:      {scoring.get('backtest_edge', 'N/A')}/100 (weight: 10%)")
    lines.append(f"  Volume confirm:     {scoring.get('volume_confirm', 'N/A')}/100 (weight: 10%)")

    return "\n".join(lines) + "\n"


# ==============================================================================
# FULL CONTEXT BUILDER
# ==============================================================================

def build_full_context(
    symbol: str,
    setup_dict: dict,
    regime_str: str = "unknown",
    structures: object = None,
    max_headlines: int = 10,
) -> str:
    """Assemble the complete AI context from all sources.

    Args:
        symbol: Ticker symbol
        setup_dict: TradeSetup.to_dict() or ScoredSetup.to_dict()
        regime_str: Current market regime string
        structures: Extracted structures object (from extract_structures)
        max_headlines: Max news headlines to include

    Returns:
        A single concatenated string with all context sections.
    """
    sections = []

    # 1. The trade setup itself (always first — this is what we're evaluating)
    sections.append(get_setup_context(setup_dict))

    # 2. Scoring breakdown
    sections.append(get_scoring_context(setup_dict))

    # 3. Pattern history (how does this pattern perform?)
    pattern_name = setup_dict.get("pattern_name", "")
    if pattern_name:
        sections.append(get_pattern_history_context(pattern_name))

    # 4. Technical structure
    if structures is not None:
        sections.append(get_technical_context(structures))

    # 5. Market context (regime + market news)
    sections.append(get_market_context(regime_str))

    # 6. Headlines for this specific stock (last — most variable)
    sections.append(get_headline_context(symbol, max_headlines=max_headlines))

    return "\n".join(sections)


# ==============================================================================
# HELPERS
# ==============================================================================

def _load_pattern_stats(pattern_name: str) -> dict:
    """Load backtest stats for a specific pattern from cache."""
    if not BACKTEST_CACHE.exists():
        return {}
    try:
        data = json.loads(BACKTEST_CACHE.read_text())
        return data.get("patterns", {}).get(pattern_name, {})
    except Exception:
        return {}