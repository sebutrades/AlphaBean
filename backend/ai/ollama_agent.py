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
You are a senior quantitative trader at a prop firm evaluating trade setups for live deployment.
You have rigorous standards — capital preservation is paramount.

DECISION FRAMEWORK:
1. PATTERN EDGE: Positive expectancy (>0.05R) and adequate sample (≥10 signals) are required
   to trust the pattern's historical record. Insufficient data demands stronger confluence elsewhere.
2. NEWS ALIGNMENT: Identify the dominant catalyst. A breakout long on bullish earnings beat is
   strong. A breakout long into an analyst downgrade is suspect. Headlines confirming the bias add
   conviction; contradicting headlines should reduce size or block entry.
3. TECHNICAL STRUCTURE: Is the entry at a logical price relative to S/R? Is ATR normal or
   extended? Volume expansion on the signal bar confirms institutional participation.
4. REGIME FIT: Momentum patterns underperform in choppy regimes. Mean-reversion fails in trends.
   High-volatility regimes require wider stops than the pattern was backtested with.
5. RISK/REWARD: After correcting for realistic entry (next bar's open), is the R:R ≥1.5?
   For patterns with <55% historical win rate, R:R must exceed 2.0 to have positive expectancy.
6. RISK FLAGS: Identify concrete adverse factors — upcoming earnings, extended ATR, thin volume,
   adverse sector momentum, high SPY correlation in a declining market, proximity to major resistance.

SCORE DELTA GUIDELINES:
  Strong CONFIRMED: +10 to +15 (all factors align, high conviction)
  Moderate CONFIRMED: +5 to +10 (more factors support than oppose)
  Weak CONFIRMED: +1 to +5 (slight edge, take with reduced size)
  CAUTION positive: 0 to +3 (conflicted — let other factors decide)
  CAUTION negative: -3 to 0 (conflicted with slight lean against)
  Moderate DENIED: -5 to -10 (clear reasons against)
  Strong DENIED: -10 to -15 (multiple disqualifying factors)\
"""

RESPONSE_FORMAT = """\

After your analysis, output ONLY the following JSON block (no text before or after):
```json
{
  "verdict": "CONFIRMED|CAUTION|DENIED",
  "confidence": <integer 0-100>,
  "score_delta": <integer -15 to 15>,
  "news_sentiment": "bullish|bearish|neutral|mixed",
  "reasoning": "<2-3 sentences — cite specific numbers and factors>",
  "risk_flags": ["<flag1>", "<flag2>"],
  "catalysts": ["<catalyst1>", "<catalyst2>"],
  "key_factors": ["<factor1>", "<factor2>", "<factor3>"]
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

    # Fetch bars to enable [TECHNICAL] section in briefing
    bars = None
    try:
        from backend.data.massive_client import fetch_bars
        tf = setup.get("timeframe_detected", "1d")
        days = 365 if tf == "1d" else 30
        bars = fetch_bars(symbol, timeframe=tf, days_back=days)
    except Exception:
        pass

    try:
        from backend.ai.ai_context import build_full_context
        briefing = build_full_context(
            symbol=symbol,
            setup_dict=setup,
            regime_str=regime,
            bars=bars,
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
            verdict="CAUTION", confidence=40, score_delta=0,
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
            "num_predict": 2048,   # Enough for thinking + JSON response
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

    # Extract JSON block
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not m:
        # Try bare JSON object
        m = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", text, flags=re.DOTALL)
    if not m:
        return _default_verdict("Could not parse model response")

    try:
        d = json.loads(m.group(1) if "```json" in text else m.group(0))
    except json.JSONDecodeError:
        return _default_verdict("JSON decode error")

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

    # CAUTION verdict caps delta magnitude
    if verdict == "CAUTION" and abs(score_delta) > 5:
        score_delta = 5 if score_delta > 0 else -5

    sentiment = str(d.get("news_sentiment", "neutral")).lower()
    if sentiment not in ("bullish", "bearish", "neutral", "mixed"):
        sentiment = "neutral"

    return AgentVerdict(
        verdict=verdict,
        confidence=confidence,
        score_delta=score_delta,
        news_sentiment=sentiment,
        reasoning=str(d.get("reasoning", "No reasoning provided"))[:400],
        risk_flags=[str(f) for f in d.get("risk_flags", [])][:5],
        catalysts=[str(c) for c in d.get("catalysts", [])][:5],
        key_factors=[str(f) for f in d.get("key_factors", [])][:3],
        processing_time=0.0,
    )


# ==============================================================================
# CACHE
# ==============================================================================

def _cache_path(symbol: str, pattern: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = pattern.replace(" ", "_").replace("/", "-")
    return CACHE_DIR / f"{symbol}_{safe}_{date.today()}.json"


def _load_cache(symbol: str, pattern: str) -> Optional[AgentVerdict]:
    path = _cache_path(symbol, pattern)
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
        return AgentVerdict(
            verdict=d["verdict"], confidence=d["confidence"],
            score_delta=d["score_delta"], news_sentiment=d["news_sentiment"],
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
        verdict="CAUTION", confidence=40, score_delta=0,
        news_sentiment="neutral", reasoning=reason,
        risk_flags=[], catalysts=[], key_factors=[],
        processing_time=0.0,
    )


def _pending_verdict() -> dict:
    return {
        "verdict": "PENDING", "confidence": 0, "score_delta": 0,
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
