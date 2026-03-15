"""
in_play.py — Uses 1 Claude API call per day to discover "in-play" symbols.

Results are cached to backend/ai/in_play_cache.json with today's date.
If cache exists for today, the API is NOT called again.
"""
import json
import os
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path(__file__).parent / "in_play_cache.json"

# Default fallback if Claude API fails or key is missing
FALLBACK_SYMBOLS = ["AAPL", "NVDA", "TSLA", "META", "AMZN", "MSFT", "AMD", "GOOGL", "SPY", "QQQ"]


def get_in_play_symbols(force_refresh: bool = False) -> dict:
    """
    Get today's in-play symbols. Uses cache if available for today.
    
    Returns:
        {
            "date": "2026-03-15",
            "symbols": ["NVDA", "TSLA", ...],
            "reasoning": "NVDA: earnings beat...",
            "source": "claude" | "cache" | "fallback"
        }
    """
    today = date.today().isoformat()

    # Check cache first
    if not force_refresh and CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text())
            if cached.get("date") == today:
                cached["source"] = "cache"
                print(f"[IN-PLAY] Using cached symbols for {today}")
                return cached
        except (json.JSONDecodeError, KeyError):
            pass  # Cache corrupt, re-fetch

    # Try Claude API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[IN-PLAY] No ANTHROPIC_API_KEY found, using fallback symbols")
        return {"date": today, "symbols": FALLBACK_SYMBOLS, "reasoning": "Fallback — no API key", "source": "fallback"}

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""Today is {today}. You are a professional day trader's research assistant.

Identify the top 10 US stock symbols that are most likely "in play" today — meaning they have:
- Unusual volume or recent catalysts (earnings, FDA, M&A, guidance change)
- Strong technical setups visible on daily/hourly charts
- High institutional interest or analyst upgrades/downgrades
- Sector momentum or macro-driven moves

Focus on liquid stocks (>$10 price, >1M avg daily volume).

Respond in this EXACT JSON format and nothing else:
{{
    "symbols": ["TICK1", "TICK2", "TICK3", "TICK4", "TICK5", "TICK6", "TICK7", "TICK8", "TICK9", "TICK10"],
    "reasoning": "TICK1: brief reason. TICK2: brief reason. ..."
}}"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.content[0].text.strip()
        # Handle potential markdown wrapping
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        symbols = [s.upper().strip() for s in parsed["symbols"][:10]]
        reasoning = parsed.get("reasoning", "")

        result = {
            "date": today,
            "symbols": symbols,
            "reasoning": reasoning,
            "source": "claude",
        }

        # Save cache
        CACHE_PATH.write_text(json.dumps(result, indent=2))
        print(f"[IN-PLAY] Claude returned {len(symbols)} symbols, cached for {today}")
        return result

    except Exception as e:
        print(f"[IN-PLAY] Claude API error: {e}, using fallback")
        return {"date": today, "symbols": FALLBACK_SYMBOLS, "reasoning": f"Fallback — API error: {e}", "source": "fallback"}