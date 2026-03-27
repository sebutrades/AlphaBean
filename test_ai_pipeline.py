"""
test_ai_pipeline.py — End-to-end test of the AI analysis pipeline.

Tests on 10 real stocks:
  1. Polygon News API: fetches headlines with sentiment
  2. AI Context Assembly: builds full briefing strings
  3. Ollama Evaluation: sends context to qwen3:8b for trade verdicts
  4. Full Scanner Pipeline: scan → score → context → AI verdict

Prerequisites:
  - Ollama running: ollama serve
  - Model pulled: ollama pull qwen3:8b
  - Polygon API key in .env

Run: python test_ai_pipeline.py
"""
import sys
import time
import json
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════

TEST_SYMBOLS = ["NVDA", "AAPL", "TSLA", "META", "AMZN", "AMD", "MSFT", "GOOGL", "SPY", "PLTR"]

PASS = 0; FAIL = 0; SKIP = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✓ {name}" + (f" ({detail})" if detail else ""))
    else:
        FAIL += 1; print(f"  ✗ {name}" + (f" ({detail})" if detail else ""))

def skip(name, detail=""):
    global SKIP
    SKIP += 1; print(f"  ⊘ {name}" + (f" ({detail})" if detail else ""))

def header(title):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")

def subheader(title):
    print(f"\n  {'─' * 60}")
    print(f"  {title}")
    print(f"  {'─' * 60}")


# ══════════════════════════════════════════════════════════════
# TEST 1: POLYGON NEWS CLIENT
# ══════════════════════════════════════════════════════════════

header("TEST 1: Polygon News Client")

try:
    from backend.data.news_client import (
        fetch_polygon_news, format_news_context, aggregate_sentiment,
        fetch_market_news,
    )
    check("news_client imports", True)
except ImportError as e:
    check("news_client imports", False, str(e))
    print("\n  → Place news_client.py in backend/data/")
    sys.exit(1)

# Fetch headlines for each test symbol
news_by_symbol = {}
print(f"\n  Fetching headlines for {len(TEST_SYMBOLS)} symbols...")

for sym in TEST_SYMBOLS:
    try:
        items = fetch_polygon_news(sym, limit=5)
        news_by_symbol[sym] = items
        n = len(items)

        if n > 0:
            sentiment = aggregate_sentiment(items, sym)
            first = items[0]
            print(f"\n  {sym}: {n} articles, sentiment net={sentiment['net']:+.2f} "
                  f"({sentiment['positive']}+ / {sentiment['negative']}- / {sentiment['neutral']}=)")
            print(f"    Latest: {first.title[:80]}...")
            print(f"    Source: {first.source} | Published: {first.published_utc[:16]}")

            # Show per-ticker sentiment if available
            insight = first.get_sentiment_for(sym)
            if insight:
                print(f"    Sentiment: {insight.sentiment.upper()}")
                print(f"    Reasoning: {insight.reasoning[:100]}...")
        else:
            print(f"\n  {sym}: No articles found")

        check(f"{sym} news fetch", n > 0, f"{n} articles")
        time.sleep(0.3)  # Rate limit courtesy

    except Exception as e:
        check(f"{sym} news fetch", False, str(e))

# Market news
subheader("Market-level news")
try:
    market_news = fetch_market_news(limit=5)
    print(f"  Got {len(market_news)} market headlines")
    for item in market_news[:3]:
        print(f"    • {item.title[:80]}... ({item.source})")
    check("Market news fetch", len(market_news) > 0)
except Exception as e:
    check("Market news fetch", False, str(e))


# ══════════════════════════════════════════════════════════════
# TEST 2: AI CONTEXT ASSEMBLY
# ══════════════════════════════════════════════════════════════

header("TEST 2: AI Context Assembly Pipeline")

try:
    from backend.ai.ai_context import (
        get_headline_context, get_technical_context,
        get_pattern_history_context, get_market_context,
        get_setup_context, get_scoring_context,
        build_full_context,
    )
    check("ai_context imports", True)
except ImportError as e:
    check("ai_context imports", False, str(e))
    print("\n  → Place ai_context.py in backend/ai/")
    sys.exit(1)

# Test each context function individually
subheader("Individual context functions")

# 2a: Headline context
print()
ctx = get_headline_context("NVDA", max_headlines=5)
print(f"  Headline context for NVDA ({len(ctx)} chars):")
for line in ctx.split('\n')[:6]:
    print(f"    {line[:90]}")
check("get_headline_context", len(ctx) > 50)

# 2b: Pattern history context
print()
ctx = get_pattern_history_context("Juicer Long")
print(f"  Pattern history for Juicer Long ({len(ctx)} chars):")
for line in ctx.split('\n'):
    print(f"    {line}")
check("get_pattern_history_context", "Juicer" in ctx)

# 2c: Market context
print()
ctx = get_market_context("trending_bull")
print(f"  Market context ({len(ctx)} chars):")
for line in ctx.split('\n')[:5]:
    print(f"    {line[:90]}")
check("get_market_context", "trending_bull" in ctx.lower() or "TRENDING_BULL" in ctx)

# 2d: Setup context (mock setup)
print()
mock_setup = {
    "pattern_name": "Juicer Long",
    "symbol": "NVDA",
    "bias": "long",
    "entry_price": 172.50,
    "stop_loss": 168.00,
    "target_price": 185.00,
    "target_1": 178.00,
    "target_2": 185.00,
    "risk_reward_ratio": 2.8,
    "confidence": 0.72,
    "description": "Juicer Long: ADX=45, above 20+50 SMA, vol 1.3x",
    "key_levels": {"sma20": 170.5, "sma50": 165.2, "adx": 45.0},
    "composite_score": 68,
    "scoring": {
        "pattern_confidence": 72,
        "feature_score": 65,
        "strategy_score": 55,
        "regime_alignment": 80,
        "backtest_edge": 75,
        "volume_confirm": 60,
    },
}
ctx = get_setup_context(mock_setup)
print(f"  Setup context ({len(ctx)} chars):")
for line in ctx.split('\n'):
    print(f"    {line}")
check("get_setup_context", "Juicer" in ctx and "172.50" in ctx)

# 2e: Full context build
subheader("Full context assembly (NVDA)")
print()
full_ctx = build_full_context(
    symbol="NVDA",
    setup_dict=mock_setup,
    regime_str="trending_bull",
    structures=None,  # No bars loaded for this test
    max_headlines=5,
)
print(f"  Full context length: {len(full_ctx)} characters")
print(f"  Sections found:")
for section in ["[TRADE SETUP]", "[SCORING", "[PATTERN HISTORY", "[MARKET CONTEXT]", "[HEADLINES"]:
    found = section in full_ctx
    print(f"    {'✓' if found else '✗'} {section}")
check("build_full_context assembles all sections", "[TRADE SETUP]" in full_ctx and "[HEADLINES" in full_ctx)

# Print the first 2000 chars of the full context so user can see it
print(f"\n  ── Full Context Preview (first 2000 chars) ──")
for line in full_ctx[:2000].split('\n'):
    print(f"  │ {line}")
if len(full_ctx) > 2000:
    print(f"  │ ... ({len(full_ctx) - 2000} more chars)")


# ══════════════════════════════════════════════════════════════
# TEST 3: OLLAMA STATUS + MODEL CHECK
# ══════════════════════════════════════════════════════════════

header("TEST 3: Ollama Status + Model Check")

try:
    from backend.ai.evaluator_prompt import check_model_status, evaluate_setup_v2
    check("evaluator_prompt imports", True)
except ImportError as e:
    check("evaluator_prompt imports", False, str(e))
    print("\n  → Place evaluator_prompt.py in backend/ai/")
    sys.exit(1)

status = check_model_status()
print(f"\n  Ollama status: {status.get('status')}")
print(f"  Model: {status.get('model', 'N/A')}")
print(f"  Available: {status.get('available', False)}")
print(f"  All models: {status.get('all_models', [])}")
check("Ollama is running", status.get("status") == "ok")

has_model = status.get("available", False)
if has_model:
    check("qwen3:8b model available", True)
else:
    skip("qwen3:8b model not available", "Run: ollama pull qwen3:8b")


# ══════════════════════════════════════════════════════════════
# TEST 4: LIVE AI EVALUATION (if Ollama + model available)
# ══════════════════════════════════════════════════════════════

header("TEST 4: Live AI Trade Evaluation")

if not has_model:
    print("\n  Skipping — qwen3:8b not available. Pull it: ollama pull qwen3:8b")
else:
    # Evaluate 3 stocks with mock setups
    test_setups = [
        {
            "symbol": "NVDA",
            "pattern_name": "Juicer Long",
            "bias": "long",
            "entry_price": 172.50,
            "stop_loss": 168.00,
            "target_price": 185.00,
            "target_1": 178.00,
            "target_2": 185.00,
            "risk_reward_ratio": 2.8,
            "confidence": 0.72,
            "description": "Juicer Long: ADX=45, stacked SMAs, vol expanding",
            "key_levels": {"sma20": 170.5, "sma50": 165.2, "adx": 45.0},
            "composite_score": 68,
            "scoring": {
                "pattern_confidence": 72, "feature_score": 65,
                "strategy_score": 55, "regime_alignment": 80,
                "backtest_edge": 75, "volume_confirm": 60,
            },
        },
        {
            "symbol": "TSLA",
            "pattern_name": "Mean Reversion",
            "bias": "long",
            "entry_price": 365.00,
            "stop_loss": 358.00,
            "target_price": 380.00,
            "target_1": 372.00,
            "target_2": 380.00,
            "risk_reward_ratio": 2.1,
            "confidence": 0.60,
            "description": "Mean Reversion Long: z-score -2.5, VWAP confluence",
            "key_levels": {"z_score": -2.5, "vwap": 370.0},
            "composite_score": 55,
            "scoring": {
                "pattern_confidence": 60, "feature_score": 50,
                "strategy_score": 45, "regime_alignment": 60,
                "backtest_edge": 55, "volume_confirm": 70,
            },
        },
        {
            "symbol": "AAPL",
            "pattern_name": "BB Squeeze Long",
            "bias": "long",
            "entry_price": 248.00,
            "stop_loss": 244.00,
            "target_price": 256.00,
            "target_1": 252.00,
            "target_2": 256.00,
            "risk_reward_ratio": 2.0,
            "confidence": 0.65,
            "description": "BB Squeeze Long: bandwidth at 5th percentile, breaking upper band",
            "key_levels": {"bw_pct": 0.05},
            "composite_score": 62,
            "scoring": {
                "pattern_confidence": 65, "feature_score": 60,
                "strategy_score": 50, "regime_alignment": 70,
                "backtest_edge": 60, "volume_confirm": 65,
            },
        },
    ]

    for setup in test_setups:
        sym = setup["symbol"]
        subheader(f"AI Evaluation: {sym} — {setup['pattern_name']}")
        print()

        t0 = time.time()
        verdict = evaluate_setup_v2(
            setup_dict=setup,
            symbol=sym,
            regime_str="trending_bull",
            structures=None,
            max_headlines=5,
        )
        elapsed = time.time() - t0

        print(f"  Symbol:     {sym}")
        print(f"  Pattern:    {setup['pattern_name']}")
        print(f"  Bias:       {setup['bias'].upper()}")
        print(f"  Entry:      ${setup['entry_price']:.2f}")
        print(f"  R:R:        {setup['risk_reward_ratio']}")
        print()
        print(f"  ┌─ AI VERDICT ─────────────────────────────────┐")
        print(f"  │ Verdict:    {verdict.get('verdict', 'N/A'):>12}                    │")
        print(f"  │ Confidence: {verdict.get('confidence', 0):>12}%                   │")
        print(f"  │ Sentiment:  {verdict.get('news_sentiment', 'N/A'):>12}                    │")
        print(f"  │ Time:       {verdict.get('processing_time', 0):>11.1f}s                    │")
        print(f"  └──────────────────────────────────────────────┘")
        print(f"  Reasoning: {verdict.get('reasoning', 'N/A')}")
        print(f"  Factors:   {', '.join(verdict.get('key_factors', []))}")
        print(f"  Context:   {verdict.get('context_length', 0)} chars sent to LLM")
        print()

        v = verdict.get("verdict", "")
        check(f"{sym} AI evaluation returned verdict", v in ("CONFIRMED", "CAUTION", "DENIED"),
              f"verdict={v}")
        check(f"{sym} processing time reasonable", elapsed < 120, f"{elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════
# TEST 5: EXISTING OLLAMA AGENT COMPATIBILITY
# ══════════════════════════════════════════════════════════════

header("TEST 5: Existing ollama_agent.py Compatibility")

try:
    from backend.ai.ollama_agent import check_ollama_status, AgentVerdict
    old_status = check_ollama_status()
    print(f"\n  Old agent status: {old_status.get('status')}")
    print(f"  Old agent model: llama3.1:8b")
    print(f"  Has llama3.1: {old_status.get('has_llama', False)}")
    check("Old ollama_agent still works", old_status.get("status") in ("ok", "offline"))
except Exception as e:
    check("Old agent compat", False, str(e))


# ══════════════════════════════════════════════════════════════
# TEST 6: LIVE SCAN + AI (full pipeline on real data)
# ══════════════════════════════════════════════════════════════

header("TEST 6: Live Scan → AI Pipeline (NVDA)")

try:
    from backend.data.massive_client import fetch_bars
    from backend.patterns.classifier import classify_all, extract_structures

    print("\n  Fetching NVDA 1d bars (365 days)...")
    bars = fetch_bars("NVDA", "1d", 365)
    print(f"  Got {len(bars.bars)} daily bars")

    print("  Running classify_all...")
    setups = classify_all(bars)
    print(f"  Found {len(setups)} setups")

    if setups:
        best = setups[0]
        print(f"\n  Top setup: {best.pattern_name} ({best.bias.value}) "
              f"conf={best.confidence:.2f}")

        # Build full context with real structures
        s = extract_structures(bars)
        setup_dict = best.to_dict()
        setup_dict["composite_score"] = 60
        setup_dict["scoring"] = {}

        subheader("Full AI Context (real data)")
        full_ctx = build_full_context(
            symbol="NVDA",
            setup_dict=setup_dict,
            regime_str="trending_bull",
            structures=s,
            max_headlines=5,
        )
        print(f"\n  Context length: {len(full_ctx)} chars")
        print(f"\n  ── Context Sections ──")
        for section in ["[TRADE SETUP]", "[SCORING", "[PATTERN HISTORY",
                        "[TECHNICAL", "[MARKET CONTEXT]", "[HEADLINES"]:
            if section in full_ctx:
                # Find and print first 3 lines of each section
                idx = full_ctx.index(section)
                section_text = full_ctx[idx:idx+300].split('\n')[:4]
                print(f"\n  {section}")
                for line in section_text:
                    print(f"    {line[:85]}")

        # AI evaluation if available
        if has_model:
            subheader("AI Evaluation on real NVDA scan")
            verdict = evaluate_setup_v2(
                setup_dict=setup_dict,
                symbol="NVDA",
                regime_str="trending_bull",
                structures=s,
                max_headlines=5,
            )
            print(f"\n  VERDICT:    {verdict.get('verdict')}")
            print(f"  CONFIDENCE: {verdict.get('confidence')}%")
            print(f"  SENTIMENT:  {verdict.get('news_sentiment')}")
            print(f"  REASONING:  {verdict.get('reasoning')}")
            print(f"  FACTORS:    {', '.join(verdict.get('key_factors', []))}")
            print(f"  TIME:       {verdict.get('processing_time')}s")
            check("Live scan → AI verdict", verdict.get("verdict") in ("CONFIRMED", "CAUTION", "DENIED"))
        else:
            skip("Live AI eval", "qwen3:8b not available")
    else:
        print("  No setups found on NVDA daily — this is normal if no patterns are active")
        check("classify_all ran without error", True)

except Exception as e:
    check("Live scan pipeline", False, str(e))
    import traceback
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

header("SUMMARY")

total = PASS + FAIL + SKIP
print(f"\n  Passed:  {PASS}")
print(f"  Failed:  {FAIL}")
print(f"  Skipped: {SKIP}")
print(f"  Total:   {total}")

if FAIL == 0:
    print(f"""
  ✓ AI Pipeline fully operational:
    ✓ Polygon news fetching headlines with per-ticker sentiment
    ✓ AI context assembly building rich briefings
    ✓ {'Ollama qwen3:8b evaluating trade setups' if has_model else 'Ollama available (pull qwen3:8b for eval)'}
    ✓ Full scan → context → AI pipeline working end-to-end

  The pipeline is ready. Next steps:
    1. Lower the scanner score threshold (45 → 20) to see setups on frontend
    2. Run the backtest with cuts + fixes applied
    3. Start building sentiment-based strategies using this news data
""")
else:
    print(f"\n  Fix the {FAIL} failure(s) above before proceeding.\n")
    sys.exit(1)