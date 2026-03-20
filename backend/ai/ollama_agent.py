"""
ai/ollama_agent.py — Ollama Trade Confirmation Agent

Uses Llama 3.1 8B (local, free) to evaluate trade setups.

For each setup, the agent receives:
  - Pattern name, entry/stop/target, R:R, confidence
  - Backtest stats for this pattern (WR, PF, expectancy, sample size)
  - Recent news headlines for the ticker
  - Market regime
  - Feature scores

Returns: CONFIRMED / CAUTION / DENIED + reasoning

Requires: Ollama running on localhost:11434 with llama3.1:8b
  ollama pull llama3.1:8b
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
TIMEOUT = 30  # seconds per call

BACKTEST_CACHE = Path("cache/backtest_results.json")


# ==============================================================================
# DATA TYPES
# ==============================================================================

@dataclass
class AgentVerdict:
    """The agent's decision on a trade setup."""
    verdict: str            # "CONFIRMED", "CAUTION", "DENIED"
    confidence: int         # 0-100
    reasoning: str          # Short explanation
    news_sentiment: str     # "bullish", "bearish", "neutral", "mixed"
    key_factors: list[str]  # Top 3 factors driving the decision
    processing_time: float  # seconds

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "news_sentiment": self.news_sentiment,
            "key_factors": self.key_factors,
            "processing_time": round(self.processing_time, 2),
        }


# ==============================================================================
# AGENT
# ==============================================================================

def evaluate_setup(
    setup: dict,
    news_headlines: str,
    regime: str = "unknown",
    backtest_stats: Optional[dict] = None,
) -> AgentVerdict:
    """
    Evaluate a single trade setup using Ollama.

    Args:
        setup: TradeSetup dict from scanner
        news_headlines: Formatted headlines string
        regime: Current market regime string
        backtest_stats: Pattern backtest stats dict (from cache)

    Returns:
        AgentVerdict with decision and reasoning
    """
    t_start = time.time()

    # Build context
    bt = backtest_stats or _load_pattern_stats(setup.get("pattern_name", ""))
    prompt = _build_prompt(setup, news_headlines, regime, bt)

    try:
        response = _call_ollama(prompt)
        verdict = _parse_response(response)
        verdict.processing_time = time.time() - t_start
        return verdict

    except Exception as e:
        return AgentVerdict(
            verdict="CAUTION",
            confidence=50,
            reasoning=f"Agent error: {str(e)[:100]}",
            news_sentiment="neutral",
            key_factors=["agent_error"],
            processing_time=time.time() - t_start,
        )


def evaluate_setups_batch(
    setups: list[dict],
    news_by_symbol: dict[str, str],
    regime: str = "unknown",
    top_n: int = 5,
) -> list[dict]:
    """
    Evaluate top N setups per symbol. Returns setups enriched with verdicts.
    """
    # Group by symbol, take top N per symbol by composite_score
    by_symbol: dict[str, list[dict]] = {}
    for s in setups:
        sym = s.get("symbol", "")
        by_symbol.setdefault(sym, []).append(s)

    evaluated = []
    for sym, sym_setups in by_symbol.items():
        # Sort by composite score, take top N
        sorted_setups = sorted(sym_setups, key=lambda x: x.get("composite_score", 0), reverse=True)
        headlines = news_by_symbol.get(sym, "No recent news available.")

        for setup in sorted_setups[:top_n]:
            verdict = evaluate_setup(setup, headlines, regime)
            setup["ai_verdict"] = verdict.to_dict()
            evaluated.append(setup)

        # Pass through remaining without evaluation
        for setup in sorted_setups[top_n:]:
            setup["ai_verdict"] = {
                "verdict": "PENDING", "confidence": 0,
                "reasoning": "Not evaluated (below top 5 cutoff)",
                "news_sentiment": "neutral", "key_factors": [],
                "processing_time": 0,
            }
            evaluated.append(setup)

    # Re-sort by composite score
    evaluated.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    return evaluated


def check_ollama_status() -> dict:
    """Check if Ollama is running and model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            has_model = any(MODEL in n for n in model_names)
            return {
                "status": "ok",
                "models": model_names,
                "has_llama": has_model,
                "message": "Ready" if has_model else f"Pull model: ollama pull {MODEL}",
            }
        return {"status": "error", "message": f"HTTP {resp.status_code}"}
    except requests.ConnectionError:
        return {"status": "offline", "message": "Ollama not running. Start: ollama serve"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==============================================================================
# PROMPT ENGINEERING
# ==============================================================================

def _build_prompt(setup: dict, news: str, regime: str, bt: dict) -> str:
    """Build the evaluation prompt for the LLM."""
    name = setup.get("pattern_name", "Unknown")
    sym = setup.get("symbol", "")
    bias = setup.get("bias", "")
    entry = setup.get("entry_price", 0)
    stop = setup.get("stop_loss", 0)
    target = setup.get("target_price", 0)
    rr = setup.get("risk_reward_ratio", 0)
    conf = setup.get("confidence", 0)
    composite = setup.get("composite_score", 0)
    category = setup.get("category", "")
    tf = setup.get("timeframe_detected", "")

    # Backtest stats
    bt_wr = bt.get("win_rate", "N/A")
    bt_pf = bt.get("profit_factor", "N/A")
    bt_exp = bt.get("expectancy", "N/A")
    bt_n = bt.get("total_signals", 0)
    bt_edge = bt.get("edge_score", "N/A")

    # Scoring breakdown
    scoring = setup.get("scoring", {})

    return f"""You are a senior prop trader evaluating a trade setup. Give a decisive verdict.

TRADE SETUP:
  Symbol: {sym}
  Pattern: {name} ({category})
  Bias: {bias.upper()}
  Timeframe: {tf}
  Entry: ${entry:.2f} | Stop: ${stop:.2f} | Target: ${target:.2f}
  Risk:Reward: {rr:.1f}
  Pattern Confidence: {conf:.0%}
  Composite Score: {composite:.0f}/100

BACKTEST STATISTICS (this pattern across ~300 symbols, last 3 months):
  Win Rate: {bt_wr}%
  Profit Factor: {bt_pf}
  Expectancy: {bt_exp}R per trade
  Sample Size: {bt_n} trades
  Edge Score: {bt_edge}/100

SCORING BREAKDOWN:
  Pattern Confidence: {scoring.get('pattern_confidence', 'N/A')}/100
  Feature Score: {scoring.get('feature_score', 'N/A')}/100
  Strategy Hot Score: {scoring.get('strategy_score', 'N/A')}/100
  Regime Alignment: {scoring.get('regime_alignment', 'N/A')}/100
  Backtest Edge: {scoring.get('backtest_edge', 'N/A')}/100
  Volume Confirmation: {scoring.get('volume_confirm', 'N/A')}/100

MARKET REGIME: {regime}

RECENT NEWS FOR {sym}:
{news}

INSTRUCTIONS:
Evaluate this trade. Consider:
1. Does the news support or contradict the trade bias?
2. Are the backtest stats strong enough to trust this pattern?
3. Does the market regime support this type of trade?
4. Is the risk:reward acceptable?

Respond in EXACTLY this format (no other text):
VERDICT: [CONFIRMED/CAUTION/DENIED]
CONFIDENCE: [0-100]
NEWS_SENTIMENT: [bullish/bearish/neutral/mixed]
REASONING: [1-2 sentence explanation]
FACTORS: [factor1, factor2, factor3]"""


# ==============================================================================
# OLLAMA COMMUNICATION
# ==============================================================================

def _call_ollama(prompt: str) -> str:
    """Call Ollama API and return the response text."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,    # Low temp for consistent decisions
            "num_predict": 200,    # Short response
            "top_p": 0.9,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama returned {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    return data.get("response", "")


def _parse_response(text: str) -> AgentVerdict:
    """Parse the structured response from the LLM."""
    lines = text.strip().split("\n")

    verdict = "CAUTION"
    confidence = 50
    sentiment = "neutral"
    reasoning = ""
    factors = []

    for line in lines:
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

    return AgentVerdict(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning or "No reasoning provided",
        news_sentiment=sentiment,
        key_factors=factors,
        processing_time=0,
    )


# ==============================================================================
# HELPERS
# ==============================================================================

def _load_pattern_stats(pattern_name: str) -> dict:
    """Load backtest stats for a specific pattern from cache."""
    if not BACKTEST_CACHE.exists():
        return {}
    try:
        data = json.loads(BACKTEST_CACHE.read_text())
        patterns = data.get("patterns", {})
        return patterns.get(pattern_name, {})
    except (json.JSONDecodeError, KeyError):
        return {}