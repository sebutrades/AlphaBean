"""
test_ai_25.py — AI Pipeline Validation: 25 Real Setups

Generates 25 real trade setups from cached bar data, runs them through the full
AI context + agent pipeline, validates every output field, and prints a summary.

Usage:
  python test_ai_25.py              # Full run (requires Ollama)
  python test_ai_25.py --context-only  # Validate context build only (no Ollama needed)
  python test_ai_25.py --symbols AAPL,NVDA,TSLA  # Specific symbols

Output columns:
  SYM  PATTERN              TF    SCORE  VERDICT    CONF  DELTA  SIZE  TIME   STATUS
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

N_SYMBOLS  = 25
TIMEFRAMES = ["15min", "5min", "1h"]   # prefer 15min for speed
MIN_SCORE  = 30                         # minimum composite to include
SYMBOLS_CACHE = Path("cache/top_symbols.json")

_VALID_VERDICTS  = {"CONFIRMED", "CAUTION", "DENIED"}
_VALID_SENTIMENT = {"bullish", "bearish", "neutral", "mixed"}
_VALID_SIZES     = {0.25, 0.5, 0.75, 1.0, 1.5}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_symbols(override: list[str] | None) -> list[str]:
    if override:
        return [s.upper() for s in override]
    if not SYMBOLS_CACHE.exists():
        print("  ERR: cache/top_symbols.json not found. Run fetch_symbols.py first.")
        sys.exit(1)
    return json.loads(SYMBOLS_CACHE.read_text()).get("symbols", [])[:N_SYMBOLS]


def _get_setup_for_symbol(symbol: str) -> dict | None:
    """Load cached bars, classify patterns, return highest-confidence setup dict."""
    from cache_bars import load_cached_bars
    from backend.data.schemas import BarSeries
    from backend.patterns.classifier import classify_all
    from backend.scoring.multi_factor import score_setup, ScoredSetup
    from backend.features.engine import compute_features
    from backend.regime.detector import detect_regime
    from backend.strategies.evaluator import StrategyEvaluator
    import numpy as np

    evaluator = StrategyEvaluator()
    evaluator.load()

    for tf in TIMEFRAMES:
        bars = load_cached_bars(symbol, tf)
        if bars is None or len(bars.bars) < 30:
            continue

        setups = classify_all(bars)
        if not setups:
            continue

        closes  = np.array([b.close  for b in bars.bars], dtype=np.float64)
        highs   = np.array([b.high   for b in bars.bars], dtype=np.float64)
        lows    = np.array([b.low    for b in bars.bars], dtype=np.float64)
        volumes = np.array([b.volume for b in bars.bars], dtype=np.float64)

        features = compute_features(closes, highs, lows, volumes)
        regime   = detect_regime(closes, highs, lows, is_spy=False)

        best: ScoredSetup | None = None
        for s in setups:
            scored = score_setup(s, features, regime, evaluator)
            scored.setup.timeframe_detected = tf
            if best is None or scored.composite_score > best.composite_score:
                best = scored

        if best and best.composite_score >= MIN_SCORE:
            d = best.to_dict()
            d["symbol"] = symbol
            d["timeframe_detected"] = tf
            return d

    return None


def _validate_output(verdict: dict) -> list[str]:
    """Return list of validation errors, empty if clean."""
    errors = []
    v = verdict.get("verdict", "")
    if v not in _VALID_VERDICTS:
        errors.append(f"verdict={v!r} not in {_VALID_VERDICTS}")
    c = verdict.get("confidence")
    if not isinstance(c, int) or not (0 <= c <= 100):
        errors.append(f"confidence={c!r} must be int 0-100")
    d = verdict.get("score_delta")
    if not isinstance(d, int) or not (-15 <= d <= 15):
        errors.append(f"score_delta={d!r} must be int -15..15")
    sz = verdict.get("size_modifier")
    if sz not in _VALID_SIZES:
        errors.append(f"size_modifier={sz!r} not in {_VALID_SIZES}")
    s = verdict.get("news_sentiment", "")
    if s not in _VALID_SENTIMENT:
        errors.append(f"news_sentiment={s!r} invalid")
    r = verdict.get("reasoning", "")
    if not r or len(r) < 10:
        errors.append("reasoning too short")
    # Check consistency
    if v == "CONFIRMED" and isinstance(d, int) and d < 0:
        errors.append(f"CONFIRMED but score_delta={d} is negative")
    if v == "DENIED" and isinstance(d, int) and d > 0:
        errors.append(f"DENIED but score_delta={d} is positive")
    if v == "DENIED" and sz and sz > 0.5:
        errors.append(f"DENIED but size_modifier={sz} > 0.5")
    return errors


def _run_one(symbol: str, context_only: bool) -> dict:
    """Full pipeline for one symbol. Returns result dict."""
    t0 = time.time()
    result = {
        "symbol": symbol, "pattern": "—", "tf": "—", "score": 0,
        "verdict": "—", "confidence": 0, "delta": 0, "size": 1.0,
        "elapsed": 0.0, "status": "no_setup", "errors": [],
    }

    try:
        setup = _get_setup_for_symbol(symbol)
        if setup is None:
            result["status"] = "no_setup"
            result["elapsed"] = round(time.time() - t0, 1)
            return result

        result["pattern"] = setup.get("pattern_name", "?")[:22]
        result["tf"]      = setup.get("timeframe_detected", "?")
        result["score"]   = round(setup.get("composite_score", 0), 1)

        # Build AI context (always)
        from backend.ai.ai_context import build_full_context
        from cache_bars import load_cached_bars
        bars = load_cached_bars(symbol, setup.get("timeframe_detected", "15min"))
        context = build_full_context(
            symbol=symbol,
            setup_dict=setup,
            regime_str="unknown",
            bars=bars,
            timeframe=setup.get("timeframe_detected", "15min"),
        )
        result["context_len"] = len(context)

        if context_only:
            result["status"]  = "context_ok"
            result["elapsed"] = round(time.time() - t0, 1)
            return result

        # Run AI agent
        from backend.ai.ollama_agent import evaluate_setup
        verdict_obj = evaluate_setup(setup, regime="unknown")
        vd = verdict_obj.to_dict()

        result["verdict"]    = vd["verdict"]
        result["confidence"] = vd["confidence"]
        result["delta"]      = vd["score_delta"]
        result["size"]       = vd["size_modifier"]
        result["reasoning"]  = vd.get("reasoning", "")[:80]
        result["cached"]     = vd.get("cached", False)
        result["errors"]     = _validate_output(vd)
        result["status"]     = "ok" if not result["errors"] else "invalid"
        result["elapsed"]    = round(time.time() - t0, 1)

    except Exception as e:
        result["status"]  = f"error: {str(e)[:60]}"
        result["elapsed"] = round(time.time() - t0, 1)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI pipeline validation — 25 real setups")
    parser.add_argument("--context-only", action="store_true",
                        help="Validate context build only, skip Ollama call")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols to test")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel workers (default 3)")
    args = parser.parse_args()

    symbols = _load_symbols(
        [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    )

    mode = "CONTEXT ONLY" if args.context_only else "FULL PIPELINE"
    print(f"\n{'═' * 90}")
    print(f"  Juicer AI Validation — {mode}  |  {len(symbols)} symbols  |  {args.workers} workers")
    print(f"{'═' * 90}")
    print(f"  {'SYM':<6} {'PATTERN':<24} {'TF':<6} {'SCORE':>5}  {'VERDICT':<10} "
          f"{'CONF':>4}  {'Δ':>4}  {'SZ':>4}  {'SEC':>5}  STATUS")
    print(f"  {'─' * 86}")

    t_total = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_run_one, sym, args.context_only): sym for sym in symbols}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)

            verdict_str = r["verdict"]
            color_pfx = ""
            if verdict_str == "CONFIRMED": color_pfx = "✓"
            elif verdict_str == "DENIED":  color_pfx = "✗"
            elif verdict_str == "CAUTION": color_pfx = "~"

            err_str = f" [{', '.join(r['errors'][:2])}]" if r.get("errors") else ""
            cached  = " [cached]" if r.get("cached") else ""
            ctx_len = f" ctx:{r.get('context_len', 0)}" if args.context_only else ""

            print(
                f"  {r['symbol']:<6} {r['pattern']:<24} {r['tf']:<6} {r['score']:>5.1f}  "
                f"{color_pfx}{r['verdict']:<9} {r['confidence']:>4}  "
                f"{r['delta']:>+4}  {r['size']:>4.2f}  {r['elapsed']:>5.1f}s"
                f"  {r['status']}{err_str}{cached}{ctx_len}"
            )

    # Summary
    elapsed = round(time.time() - t_total, 1)
    ok       = sum(1 for r in results if r["status"] in ("ok", "context_ok"))
    no_setup = sum(1 for r in results if r["status"] == "no_setup")
    invalid  = sum(1 for r in results if r["status"] == "invalid")
    errors   = sum(1 for r in results if r["status"].startswith("error"))
    confirmed = sum(1 for r in results if r["verdict"] == "CONFIRMED")
    caution   = sum(1 for r in results if r["verdict"] == "CAUTION")
    denied    = sum(1 for r in results if r["verdict"] == "DENIED")

    # Verdict distribution
    by_pattern = defaultdict(list)
    for r in results:
        if r["verdict"] in _VALID_VERDICTS:
            by_pattern[r["pattern"]].append(r["verdict"])

    print(f"\n{'─' * 90}")
    print(f"  SUMMARY  {elapsed}s total")
    print(f"  Pass: {ok}  |  No setup: {no_setup}  |  Invalid output: {invalid}  |  Errors: {errors}")
    if not args.context_only:
        print(f"  Verdicts: CONFIRMED={confirmed}  CAUTION={caution}  DENIED={denied}")
        avg_delta = (sum(r["delta"] for r in results if isinstance(r["delta"], int)) /
                     max(1, sum(1 for r in results if isinstance(r["delta"], int))))
        print(f"  Avg score_delta: {avg_delta:+.1f}")

    # Flag any validation errors
    all_errors = [(r["symbol"], r["errors"]) for r in results if r.get("errors")]
    if all_errors:
        print(f"\n  VALIDATION ERRORS ({len(all_errors)} setups):")
        for sym, errs in all_errors:
            print(f"    {sym}: {errs}")
    else:
        print(f"\n  All outputs valid ✓")

    print(f"{'═' * 90}\n")
    return 0 if (invalid + errors) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
