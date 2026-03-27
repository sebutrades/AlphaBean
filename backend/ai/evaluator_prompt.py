"""
backend/ai/evaluator_prompt.py — Trade Evaluator Pre-Prompt + Qwen3 Integration

UPGRADE PATH:
  This file provides the new system prompt and a drop-in replacement
  for the evaluate_setup() function in ollama_agent.py.

  Option A: Replace the entire ollama_agent.py with this
  Option B: Just update MODEL, _build_prompt, and the evaluate function

MODEL: qwen3:8b (Q4_K_M, ~5.5GB VRAM, ~40 tok/s on RTX 3070 8GB)
  - Built-in chain-of-thought reasoning (thinks before answering)
  - Best reasoning model at 8B parameters
  - Install: ollama pull qwen3:8b

CHANGES FROM EXISTING ollama_agent.py:
  1. Model: llama3.1:8b → qwen3:8b
  2. Prompt: Generic trader prompt → rich context-aware prompt using ai_context.py
  3. Response format: Same structured format (VERDICT/CONFIDENCE/etc) but
     with thinking mode enabled for better reasoning
  4. Context: Now receives full briefing from ai_context pipeline
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from backend.ai.ai_context import build_full_context


# ==============================================================================
# CONFIG
# ==============================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"        # ← CHANGED from llama3.1:8b
TIMEOUT = 60               # ← INCREASED: qwen3 thinking mode needs more time


# ==============================================================================
# SYSTEM PROMPT
# ==============================================================================

SYSTEM_PROMPT = """You are a senior quantitative trader at a proprietary trading firm. Your job is to evaluate trade setups and decide whether to risk capital on them.

You have access to:
- The specific trade setup (pattern, entry, stop, target, R:R)
- Historical backtest statistics for this pattern across 500 stocks
- The stock's current technical structure (price vs SMAs, ATR, volume, support/resistance)
- Recent news headlines with AI-generated sentiment analysis
- The current market regime (trending/choppy/volatile)
- A multi-factor composite score from the scanning engine

YOUR DECISION FRAMEWORK:
1. PATTERN QUALITY: Does the backtest history support this pattern? Positive expectancy patterns deserve benefit of the doubt. Negative expectancy patterns need exceptional confluence.
2. NEWS ALIGNMENT: Do headlines support or contradict the trade bias? A bullish setup with negative news is suspicious. A bearish setup confirmed by bad news is stronger.
3. TECHNICAL STRUCTURE: Is the stock at a meaningful level? Is volume confirming? Is volatility compressed (good for breakouts) or extended (good for reversals)?
4. REGIME FIT: Does this type of trade work in the current market environment? Breakouts work in trends. Mean reversion works in ranges. Don't fight the regime.
5. RISK/REWARD: Is the R:R acceptable given the win rate? A 50% WR pattern needs at least 1.5 R:R. A 70% WR pattern can work at 1:1.

RESPOND IN EXACTLY THIS FORMAT (no other text before or after):
VERDICT: [CONFIRMED/CAUTION/DENIED]
CONFIDENCE: [0-100]
NEWS_SENTIMENT: [bullish/bearish/neutral/mixed]
REASONING: [2-3 sentence explanation of your decision. Be specific about which factors drove it.]
FACTORS: [factor1, factor2, factor3]

VERDICT GUIDELINES:
- CONFIRMED: Strong confluence. Pattern has edge, news aligns, regime supports it, R:R is good. You would put money on this.
- CAUTION: Mixed signals. Some factors support, others don't. Trade could work but has meaningful risk factors. Reduce size.
- DENIED: Poor setup. Pattern has no edge, news contradicts, regime is wrong, or R:R is bad. Skip this trade."""


# ==============================================================================
# EVALUATE WITH FULL CONTEXT
# ==============================================================================

def evaluate_setup_v2(
    setup_dict: dict,
    symbol: str,
    regime_str: str = "unknown",
    structures: object = None,
    max_headlines: int = 10,
) -> dict:
    """Evaluate a trade setup using the full AI context pipeline.

    This is the upgraded version that:
      1. Builds rich context from ai_context.py
      2. Sends to qwen3:8b with the trader system prompt
      3. Parses the structured response

    Args:
        setup_dict: TradeSetup.to_dict() or ScoredSetup.to_dict()
        symbol: Ticker symbol
        regime_str: Current market regime
        structures: Extracted structures (optional, for technical context)
        max_headlines: Max headlines to include

    Returns:
        AgentVerdict dict with verdict, confidence, reasoning, etc.
    """
    t_start = time.time()

    # Build the full context
    context = build_full_context(
        symbol=symbol,
        setup_dict=setup_dict,
        regime_str=regime_str,
        structures=structures,
        max_headlines=max_headlines,
    )

    # Build the prompt
    prompt = f"{SYSTEM_PROMPT}\n\n--- FULL BRIEFING ---\n\n{context}\n\n--- YOUR EVALUATION --- /no_think"

    try:
        response = _call_ollama(prompt)
        verdict = _parse_response(response)
        verdict["processing_time"] = round(time.time() - t_start, 2)
        verdict["model"] = MODEL
        verdict["context_length"] = len(context)
        return verdict

    except Exception as e:
        return {
            "verdict": "CAUTION",
            "confidence": 50,
            "reasoning": f"Agent error: {str(e)[:100]}",
            "news_sentiment": "neutral",
            "key_factors": ["agent_error"],
            "processing_time": round(time.time() - t_start, 2),
            "model": MODEL,
        }


def evaluate_batch_v2(
    scored_setups: list[dict],
    regime_str: str = "unknown",
    structures_by_symbol: dict = None,
    top_n: int = 5,
) -> list[dict]:
    """Evaluate the top N setups per symbol using the full context pipeline.

    Args:
        scored_setups: List of ScoredSetup.to_dict()
        regime_str: Current market regime
        structures_by_symbol: Dict of {symbol: structures} for technical context
        top_n: Max setups to evaluate per symbol

    Returns:
        Same list with "ai_verdict" key added to each setup.
    """
    if structures_by_symbol is None:
        structures_by_symbol = {}

    # Group by symbol
    by_symbol = {}
    for s in scored_setups:
        sym = s.get("symbol", "")
        by_symbol.setdefault(sym, []).append(s)

    evaluated = []
    for sym, sym_setups in by_symbol.items():
        sorted_setups = sorted(sym_setups,
                               key=lambda x: x.get("composite_score", 0),
                               reverse=True)

        structures = structures_by_symbol.get(sym)

        for setup in sorted_setups[:top_n]:
            verdict = evaluate_setup_v2(
                setup_dict=setup,
                symbol=sym,
                regime_str=regime_str,
                structures=structures,
            )
            setup["ai_verdict"] = verdict
            evaluated.append(setup)

        # Pass through remaining without evaluation
        for setup in sorted_setups[top_n:]:
            setup["ai_verdict"] = {
                "verdict": "PENDING", "confidence": 0,
                "reasoning": "Not evaluated (below top cutoff)",
                "news_sentiment": "neutral", "key_factors": [],
                "processing_time": 0, "model": MODEL,
            }
            evaluated.append(setup)

    evaluated.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    return evaluated


# ==============================================================================
# OLLAMA COMMUNICATION
# ==============================================================================

def _call_ollama(prompt: str) -> str:
    """Call Ollama with qwen3:8b."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,    # Low temp for consistent decisions
            "num_predict": 2000,    # More room for thinking + response
            "top_p": 0.9,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama returned {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    return data.get("response", "")


def _parse_response(text: str) -> dict:
    """Parse the structured response from the LLM.

    Qwen3's thinking mode may include <think>...</think> blocks
    before the actual response. We strip those.
    """
    # Strip thinking blocks if present
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()

    verdict = "CAUTION"
    confidence = 50
    sentiment = "neutral"
    reasoning = ""
    factors = []

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            if v in ("CONFIRMED", "CAUTION", "DENIED"):
                verdict = v
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = int(line.split(":", 1)[1].strip().rstrip("%"))
                confidence = max(0, min(100, confidence))
            except ValueError:
                pass
        elif line.startswith("NEWS_SENTIMENT:"):
            s = line.split(":", 1)[1].strip().lower()
            if s in ("bullish", "bearish", "neutral", "mixed"):
                sentiment = s
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
        elif line.startswith("FACTORS:"):
            raw = line.split(":", 1)[1].strip()
            factors = [f.strip().strip("[]") for f in raw.split(",")][:3]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning or "No reasoning provided",
        "news_sentiment": sentiment,
        "key_factors": factors,
    }


def check_model_status() -> dict:
    """Check if Ollama is running and qwen3:8b is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            has_model = any(MODEL in n for n in model_names)
            return {
                "status": "ok",
                "model": MODEL,
                "available": has_model,
                "all_models": model_names,
                "message": "Ready" if has_model else f"Pull model: ollama pull {MODEL}",
            }
        return {"status": "error", "message": f"HTTP {resp.status_code}"}
    except requests.ConnectionError:
        return {"status": "offline", "message": "Ollama not running. Start: ollama serve"}
    except Exception as e:
        return {"status": "error", "message": str(e)}