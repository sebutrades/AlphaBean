"""
ai/ollama_agent.py — Qwen3 Trade Analysis Agent

HARDWARE RECOMMENDATION (RTX 3070 8GB VRAM):
  qwen3:8b  — ~5 GB VRAM in Q4_K_M, ~40 tok/s — recommended
  qwen3:14b — ~9 GB with CPU offload for some layers — slower but sharper

  ollama pull qwen3:8b

PIPELINE:
  1. Assemble full context briefing (ai_context.py)
       - Mathematical setup (entry, stop, targets in ATR multiples + dollar distances)
       - Pattern backtest history (WR, PF, expectancy, edge score)
       - Technical structure (price vs SMAs, volatility regime, S/R levels)
       - Market regime + interpretation
       - Recent headlines with Polygon AI sentiment (fallback: Finnhub/RSS)
       - Multi-factor scoring breakdown
       - Risk assessment (IV, earnings proximity, correlation, ATR vs avg)
  2. Send to qwen3:8b with thinking mode enabled
  3. Parse JSON response block from model output
  4. Extract verdict, confidence, score_delta, risk_flags, catalysts, reasoning
  5. Apply score_delta to composite_score in the batch evaluator
  6. Cache result per (symbol, pattern, date) — one LLM call per setup per day

SCORE DELTA LOGIC:
  CONFIRMED high conf (80+): +12 to +15
  CONFIRMED med  conf (60+): +6  to +12
  CAUTION                  : -5  to +5
  DENIED  med  conf (60+)  : -8  to -12
  DENIED  high conf (80+)  : -12 to -15
"""
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
# Optimal for RTX 3070 8GB — full GPU inference, ~40 tok/s
# Change to "qwen3:14b" for higher reasoning quality at the cost of speed
MODEL = "qwen3:8b"
TIMEOUT = 90       # seconds — qwen3 thinking mode needs ~30-60s at 8B
CACHE_DIR = Path("cache/ai_verdicts")
BACKTEST_CACHE = Path("cache/backtest_results.json")


# ==============================================================================
# SYSTEM PROMPT
# ==============================================================================

SYSTEM_PROMPT = """\
You are a senior quantitative trader at a prop firm evaluating trade setups.
Your job is to give a thorough, balanced assessment — not to be cautious by default.
Good setups should be CONFIRMED. Bad setups should be DENIED. CAUTION is for genuinely mixed cases only.

EVALUATION PROCESS — follow in order:
Step 1 — BULL CASE: Identify 2-3 concrete reasons this trade succeeds.
  Consider: positive symbol stats, confirming price action, strong pattern edge, hot strategy,
  confirming news catalyst, clean R:R, regime alignment, volume expansion.
Step 2 — BEAR CASE: Identify 2-3 concrete reasons this trade could fail.
  Consider: adverse symbol-specific history, conflicting price structure, weak or absent catalyst,
  wrong regime for this pattern type, upcoming risk events, thin volume, wide stop.
Step 3 — WEIGH THE EVIDENCE and decide:
  Bull dominates → CONFIRMED (this should be ~40-50% of setups that reach AI evaluation)
  Bear dominates → DENIED (this should be ~20-30%)
  Genuinely balanced, could go either way → CAUTION (only ~20-30%, NOT the default)

IMPORTANT: A setup that reached AI evaluation already passed quantitative scoring filters.
If the pattern has positive edge, structure confirms, and no major risks exist → CONFIRMED.
Do NOT default to CAUTION out of conservatism. Reserve CAUTION for genuinely conflicted cases
where bull and bear factors are roughly equal.

CONTEXT WEIGHTING (sections in the briefing, ordered by importance):
1. PATTERN EDGE + SYMBOL PERFORMANCE — Does this pattern work? Does it work on THIS stock?
   Positive expectancy (>0.05R) with sufficient sample (≥10) is a strong CONFIRMED signal.
2. PRICE ACTION + TECHNICAL STRUCTURE — Does the chart confirm the bias? HH/HL for longs,
   LH/LL for shorts. Volume expanding on signal = strong confirmation.
3. NEWS + CATALYSTS — Is there a fundamental driver supporting the direction? Recent headlines
   with positive sentiment for the bias = adds conviction. Contradicting news = red flag.
4. REGIME FIT — Does the market environment suit this pattern type?
5. STRATEGY HOT/COLD — Is this pattern currently performing well in live conditions?
6. RISK/REWARD — R:R ≥1.5 required; ≥2.0 preferred.
7. RISK FLAGS — earnings proximity, extended ATR, low volume, correlation risk.

SCORE DELTA:
  Strong CONFIRMED (+10 to +15): most factors align — edge, structure, news, regime
  Moderate CONFIRMED (+5 to +10): majority support, clear edge, minor concerns
  Weak CONFIRMED (+1 to +5): edge present but some risk flags — reduced size warranted
  CAUTION (-3 to +3): genuinely mixed — factors conflict in roughly equal measure
  Moderate DENIED (-5 to -10): clear reasons against — bear case dominates
  Strong DENIED (-10 to -15): multiple disqualifying factors — do not take this trade

SIZE MODIFIER:
  1.5 — exceptional: all factors align plus hot strategy + confirming news
  1.0 — standard: CONFIRMED with normal confluence
  0.75 — reduced: CONFIRMED but 1-2 risk flags present
  0.5  — half size: weak CONFIRMED or genuinely mixed CAUTION
  0.25 — quarter: DENIED or high-risk with compelling pattern only\
"""

RESPONSE_FORMAT = """\

Output ONLY this JSON block — no text before or after, no explanation outside it:
```json
{
  "verdict": "CONFIRMED|CAUTION|DENIED",
  "confidence": <integer 0-100>,
  "score_delta": <integer -15 to 15>,
  "size_modifier": <one of: 0.25, 0.5, 0.75, 1.0, 1.5>,
  "news_sentiment": "bullish|bearish|neutral|mixed",
  "bull_case": "<2-3 sentences: specific reasons supporting the trade — cite exact numbers from the briefing (win rate, expectancy, R:R, RVOL, SMA position, sentiment score). Name the strongest factor.>",
  "bear_case": "<2-3 sentences: specific reasons against the trade — cite exact numbers. Name the most concerning factor.>",
  "reasoning": "<3-5 sentences: your synthesis weighing bull vs bear. Explain WHY the verdict tilts the way it does. Reference the specific data that tips the balance. Include what price action, news, or regime factor was most decisive.>",
  "news_analysis": "<1-2 sentences: what do the headlines say about this stock right now? Is there a catalyst, or is it quiet? How does news sentiment align with the trade bias?>",
  "risk_flags": ["<specific risk 1>", "<specific risk 2>"],
  "catalysts": ["<specific catalyst driving this trade>"],
  "key_factors": ["<strongest bull factor>", "<strongest bear factor>", "<deciding factor that tips the verdict>"]
}
```\
"""


# ==============================================================================
# DATA TYPES
# ==============================================================================

@dataclass
class AgentVerdict:
    verdict: str            # CONFIRMED / CAUTION / DENIED
    confidence: int         # 0-100
    score_delta: int        # Applied to composite_score (-15 to +15)
    size_modifier: float    # Position size multiplier: 0.25 / 0.5 / 0.75 / 1.0 / 1.5
    news_sentiment: str     # bullish / bearish / neutral / mixed
    reasoning: str
    risk_flags: list[str]
    catalysts: list[str]
    key_factors: list[str]
    processing_time: float
    model: str = MODEL
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "score_delta": self.score_delta,
            "size_modifier": self.size_modifier,
            "news_sentiment": self.news_sentiment,
            "reasoning": self.reasoning,
            "risk_flags": self.risk_flags,
            "catalysts": self.catalysts,
            "key_factors": self.key_factors,
            "processing_time": round(self.processing_time, 2),
            "model": self.model,
            "cached": self.cached,
        }


# ==============================================================================
# PUBLIC API
# ==============================================================================

def evaluate_setup(
    setup: dict,
    news_headlines: str = "",
    regime: str = "unknown",
) -> AgentVerdict:
    """
    Evaluate a single trade setup.
    Builds the full ai_context briefing, calls Qwen3, parses JSON response.
    """
    t0 = time.time()
    symbol = setup.get("symbol", "")
    pattern = setup.get("pattern_name", "Unknown")

    # Serve from cache if already evaluated today
    cached = _load_cache(symbol, pattern)
    if cached:
        cached.processing_time = 0.0
        cached.cached = True
        return cached

    # Fetch bars — cache-first, live API fallback (fresh data needed for technical context)
    bars = None
    tf_label = setup.get("timeframe_detected", "1d")
    # Normalize scanner labels ("5m" → "5min", "15m" → "15min") to cache key format
    _TF_NORM = {"5m": "5min", "15m": "15min", "5m & 15m": "15min"}
    tf_cache = _TF_NORM.get(tf_label, tf_label)
    try:
        from cache_bars import load_cached_bars
        bars = load_cached_bars(symbol, tf_cache)
    except Exception:
        pass
    if bars is None:
        try:
            from backend.data.massive_client import fetch_bars
            days = 365 if tf_cache == "1d" else 60
            bars = fetch_bars(symbol, timeframe=tf_cache, days_back=days)
        except Exception:
            pass

    try:
        from backend.ai.ai_context import build_full_context
        briefing = build_full_context(
            symbol=symbol,
            setup_dict=setup,
            regime_str=regime,
            bars=bars,
            timeframe=tf_cache,
        )
    except Exception as e:
        # Degrade gracefully: use basic briefing from setup dict
        briefing = _fallback_briefing(setup, news_headlines, regime)

    prompt = f"{SYSTEM_PROMPT}\n\n{briefing}\n{RESPONSE_FORMAT}"

    try:
        raw = _call_ollama(prompt)
        verdict = _parse_response(raw)
        verdict.processing_time = time.time() - t0
        _save_cache(symbol, pattern, verdict)
        return verdict

    except Exception as e:
        return AgentVerdict(
            verdict="CAUTION", confidence=40, score_delta=0, size_modifier=0.5,
            news_sentiment="neutral", reasoning=f"Agent unavailable: {str(e)[:80]}",
            risk_flags=["agent_error"], catalysts=[], key_factors=["agent_error"],
            processing_time=time.time() - t0,
        )


def evaluate_setups_batch(
    setups: list[dict],
    news_by_symbol: dict[str, str],
    regime: str = "unknown",
    top_n: int = 5,
) -> list[dict]:
    """
    Evaluate top N setups per symbol, apply score_delta to composite_score.
    Setups outside the top N per symbol receive a PENDING verdict with no delta.
    Returns list sorted by adjusted composite_score.
    """
    by_symbol: dict[str, list[dict]] = {}
    for s in setups:
        by_symbol.setdefault(s.get("symbol", ""), []).append(s)

    evaluated = []
    for sym, sym_setups in by_symbol.items():
        ranked = sorted(sym_setups, key=lambda x: x.get("composite_score", 0), reverse=True)
        headlines = news_by_symbol.get(sym, "")

        for setup in ranked[:top_n]:
            verdict = evaluate_setup(setup, headlines, regime)
            vd = verdict.to_dict()

            # Programmatic caps — override model delta when hard risk conditions apply
            delta = verdict.score_delta
            scoring = setup.get("scoring", {})
            vol_score = scoring.get("volume_confirm", 100)
            if vol_score < 30:
                # Very thin volume: cap positive delta at +2
                delta = min(delta, 2)
            elif vol_score < 45:
                # Below-average volume: cap positive delta at +5
                delta = min(delta, 5)
            vd["score_delta"] = delta

            setup["ai_verdict"] = vd
            raw_score = setup.get("composite_score", 0)
            adjusted = max(0.0, min(100.0, raw_score + delta))
            setup["composite_score"] = round(adjusted, 1)
            setup["composite_score_pre_ai"] = round(raw_score, 1)
            evaluated.append(setup)

        for setup in ranked[top_n:]:
            setup["ai_verdict"] = _pending_verdict()
            evaluated.append(setup)

    evaluated.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    return evaluated


def check_ollama_status() -> dict:
    """Check if Ollama is running and the configured model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return {"status": "error", "message": f"HTTP {resp.status_code}"}
        models = resp.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        has_model = any(MODEL.split(":")[0] in n for n in model_names)
        exact = any(MODEL in n for n in model_names)
        return {
            "status": "ok",
            "model": MODEL,
            "model_available": has_model,
            "exact_match": exact,
            "all_models": model_names,
            "message": (
                "Ready" if exact else
                f"Pull model: ollama pull {MODEL}" if not has_model else
                f"Model family found but exact tag missing — try: ollama pull {MODEL}"
            ),
        }
    except requests.ConnectionError:
        return {"status": "offline", "message": "Ollama not running — start with: ollama serve"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==============================================================================
# TEST HARNESS
# ==============================================================================

def test_agent(symbol: str, regime: str = "unknown") -> dict:
    """
    Run the full pipeline for a symbol with a synthetic setup.
    Used to verify model quality and reasoning via /api/ai/test/{symbol}.
    Returns the full verdict + the raw model output for inspection.
    """
    t0 = time.time()

    # Build a synthetic but realistic setup for testing
    synthetic = {
        "symbol": symbol,
        "pattern_name": "Momentum Breakout",
        "bias": "long",
        "entry_price": 150.00,
        "stop_loss":   148.50,
        "target_1":    154.00,
        "target_2":    157.00,
        "target_price": 157.00,
        "risk_reward_ratio": 4.67,
        "confidence": 0.58,
        "composite_score": 62.0,
        "description": f"Momentum BO: new 30d high",
        "timeframe_detected": "1d",
        "scoring": {
            "pattern_confidence": 58.0,
            "feature_score": 65.0,
            "strategy_score": 55.0,
            "regime_alignment": 70.0,
            "backtest_edge": 43.0,
            "volume_confirm": 72.0,
        },
    }

    try:
        from backend.ai.ai_context import build_full_context
        briefing = build_full_context(
            symbol=symbol,
            setup_dict=synthetic,
            regime_str=regime,
            timeframe="1d",
        )
    except Exception as e:
        briefing = f"[Context build failed: {e}]"

    prompt = f"{SYSTEM_PROMPT}\n\n{briefing}\n{RESPONSE_FORMAT}"

    try:
        raw = _call_ollama(prompt)
        verdict = _parse_response(raw)
        return {
            "symbol": symbol,
            "setup": synthetic,
            "verdict": verdict.to_dict(),
            "raw_response": raw[:2000],   # Truncated for readability
            "briefing_length": len(briefing),
            "elapsed": round(time.time() - t0, 1),
        }
    except Exception as e:
        return {"error": str(e), "elapsed": round(time.time() - t0, 1)}


# ==============================================================================
# OLLAMA COMMUNICATION
# ==============================================================================

def _call_ollama(prompt: str) -> str:
    """Send prompt to Ollama and return model response text."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,   # Near-deterministic for trading decisions
            "num_predict": 8192,   # Qwen3 thinking mode: 1500-3000 think + 800 JSON
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama {resp.status_code}: {resp.text[:200]}")
    return resp.json().get("response", "")


# ==============================================================================
# RESPONSE PARSING
# ==============================================================================

def _parse_response(text: str) -> AgentVerdict:
    """
    Extract the JSON block from the model response.
    Qwen3 thinking mode may produce <think>...</think> before the output.
    We strip thinking, then extract the json block.
    """
    # Strip thinking block if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Extract JSON block — try fenced first, then depth-tracked scan
    d = None
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    if d is None:
        # Depth-counting extractor: handles nested JSON correctly
        start = text.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            candidate = json.loads(text[start:i+1])
                            if "verdict" in candidate:
                                d = candidate
                        except json.JSONDecodeError:
                            pass
                        break
            start = text.find("{", start + 1)
            if d is not None:
                break

    if d is None:
        return _default_verdict("Could not parse model response")

    verdict = str(d.get("verdict", "CAUTION")).upper()
    if verdict not in ("CONFIRMED", "CAUTION", "DENIED"):
        verdict = "CAUTION"

    confidence = _clamp(d.get("confidence", 50), 0, 100)
    score_delta = _clamp(d.get("score_delta", 0), -15, 15)

    # Enforce consistency: don't let score_delta contradict verdict
    if verdict == "CONFIRMED" and score_delta < 0:
        score_delta = max(1, -score_delta // 2)
    elif verdict == "DENIED" and score_delta > 0:
        score_delta = min(-1, -score_delta // 2)
    if verdict == "CAUTION" and abs(score_delta) > 5:
        score_delta = 5 if score_delta > 0 else -5

    # size_modifier: validate against allowed values
    _VALID_SIZES = {0.25, 0.5, 0.75, 1.0, 1.5}
    raw_size = d.get("size_modifier", 1.0)
    try:
        raw_size = float(raw_size)
        size_modifier = min(_VALID_SIZES, key=lambda x: abs(x - raw_size))
    except (TypeError, ValueError):
        size_modifier = 1.0
    # Enforce size consistency with verdict
    if verdict == "DENIED" and size_modifier > 0.5:
        size_modifier = 0.25
    elif verdict == "CONFIRMED" and size_modifier < 0.5:
        size_modifier = 0.75

    sentiment = str(d.get("news_sentiment", "neutral")).lower()
    if sentiment not in ("bullish", "bearish", "neutral", "mixed"):
        sentiment = "neutral"

    # Build comprehensive reasoning from all available fields
    bull_case = str(d.get("bull_case", "")).strip()
    bear_case = str(d.get("bear_case", "")).strip()
    reasoning = str(d.get("reasoning", "No reasoning provided")).strip()
    news_analysis = str(d.get("news_analysis", "")).strip()

    # Combine into a rich analysis string
    parts = []
    if bull_case:
        parts.append(f"BULL: {bull_case}")
    if bear_case:
        parts.append(f"BEAR: {bear_case}")
    if reasoning:
        parts.append(reasoning)
    if news_analysis:
        parts.append(f"NEWS: {news_analysis}")
    full_reasoning = " | ".join(parts) if parts else reasoning

    return AgentVerdict(
        verdict=verdict,
        confidence=confidence,
        score_delta=score_delta,
        size_modifier=size_modifier,
        news_sentiment=sentiment,
        reasoning=full_reasoning[:1500],
        risk_flags=[str(f) for f in d.get("risk_flags", [])][:6],
        catalysts=[str(c) for c in d.get("catalysts", [])][:5],
        key_factors=[str(f) for f in d.get("key_factors", [])][:5],
        processing_time=0.0,
    )


# ==============================================================================
# CACHE
# ==============================================================================

def _cache_path(symbol: str, pattern: str) -> Path:
    """Cache path with hourly granularity.

    Old key: (symbol, pattern, date)   → same verdict all day
    New key: (symbol, pattern, date, hour) → re-evaluates each hour
    so intraday setups get fresh context as price evolves.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = pattern.replace(" ", "_").replace("/", "-")
    from datetime import datetime
    now = datetime.now()
    return CACHE_DIR / f"{symbol}_{safe}_{now.strftime('%Y-%m-%d_%H')}.json"


def _load_cache(symbol: str, pattern: str) -> Optional[AgentVerdict]:
    path = _cache_path(symbol, pattern)
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
        return AgentVerdict(
            verdict=d["verdict"], confidence=d["confidence"],
            score_delta=d["score_delta"], size_modifier=d.get("size_modifier", 1.0),
            news_sentiment=d["news_sentiment"],
            reasoning=d["reasoning"], risk_flags=d.get("risk_flags", []),
            catalysts=d.get("catalysts", []), key_factors=d.get("key_factors", []),
            processing_time=0.0, model=d.get("model", MODEL), cached=True,
        )
    except Exception:
        return None


def _save_cache(symbol: str, pattern: str, verdict: AgentVerdict) -> None:
    try:
        _cache_path(symbol, pattern).write_text(json.dumps(verdict.to_dict()))
    except Exception:
        pass


# ==============================================================================
# HELPERS
# ==============================================================================

def _default_verdict(reason: str) -> AgentVerdict:
    return AgentVerdict(
        verdict="CAUTION", confidence=40, score_delta=0, size_modifier=0.5,
        news_sentiment="neutral", reasoning=reason,
        risk_flags=[], catalysts=[], key_factors=[],
        processing_time=0.0,
    )


def _pending_verdict() -> dict:
    return {
        "verdict": "PENDING", "confidence": 0, "score_delta": 0, "size_modifier": 1.0,
        "news_sentiment": "neutral",
        "reasoning": "Not evaluated — below per-symbol top-N cutoff",
        "risk_flags": [], "catalysts": [], "key_factors": [],
        "processing_time": 0, "model": MODEL, "cached": False,
    }


def _clamp(val, lo, hi) -> int:
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return 0


def _fallback_briefing(setup: dict, headlines: str, regime: str) -> str:
    """Minimal briefing when ai_context fails."""
    entry = setup.get("entry_price", 0)
    stop = setup.get("stop_loss", 0)
    t1 = setup.get("target_1", 0) or setup.get("target_price", 0)
    risk = abs(entry - stop)
    rr = abs(t1 - entry) / risk if risk > 0 else 0
    return (
        f"[TRADE SETUP]\n"
        f"Symbol: {setup.get('symbol')} | Pattern: {setup.get('pattern_name')} | Bias: {setup.get('bias','').upper()}\n"
        f"Entry: ${entry:.2f} | Stop: ${stop:.2f} | T1: ${t1:.2f} | R:R {rr:.1f}\n"
        f"Confidence: {setup.get('confidence',0):.0%} | Composite: {setup.get('composite_score',0):.0f}/100\n\n"
        f"[MARKET REGIME]\n{regime}\n\n"
        f"[NEWS]\n{headlines or 'No headlines available.'}\n"
    )
