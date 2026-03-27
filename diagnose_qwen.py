"""Quick diagnostic: what is Qwen3 actually returning?"""
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

prompt = """You are a senior trader evaluating a trade setup.

TRADE: Buy NVDA at $172.50, stop $168.00, target $185.00. R:R = 2.8.
Pattern: Juicer Long (57% win rate, +0.78R expectancy, 539 sample trades).
Market regime: Trending bull.
News sentiment: Positive (3 bullish / 0 bearish / 2 neutral articles).

Respond in EXACTLY this format, nothing else before or after:
VERDICT: [CONFIRMED/CAUTION/DENIED]
CONFIDENCE: [0-100]
NEWS_SENTIMENT: [bullish/bearish/neutral/mixed]
REASONING: [1-2 sentence explanation]
FACTORS: [factor1, factor2, factor3]"""

print("Sending to qwen3:8b...")
resp = requests.post(OLLAMA_URL, json={
    "model": "qwen3:8b",
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.3, "num_predict": 400, "top_p": 0.9},
}, timeout=60)

data = resp.json()
raw = data.get("response", "")

print(f"\n{'=' * 60}")
print("RAW RESPONSE ({} chars):".format(len(raw)))
print('=' * 60)
print(raw)
print('=' * 60)

# Check if think tags are present
if '<think>' in raw:
    print("\n[!] Response contains <think> tags")
    import re
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    print(f"\nAFTER STRIPPING THINK TAGS ({len(cleaned)} chars):")
    print(cleaned)
    if not cleaned:
        print("\n[!] ENTIRE response was inside <think> tags — nothing left after stripping!")
        print("    FIX: Add /no_think to the prompt or adjust num_predict")