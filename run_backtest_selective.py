"""
run_backtest_selective.py — Run backtest for specific patterns only, merge into cache.

Use this to re-test only the patterns whose results are stale/missing without
re-running all 500 symbols × 54 patterns from scratch.

Results are MERGED into cache/backtest_results.json — existing patterns are kept,
only the specified ones are replaced.

Usage:
  # Re-test the 2 VP Divergence patterns (contaminated by ghost fix)
  python run_backtest_selective.py --patterns "VP Divergence Long,VP Divergence Short"

  # Re-test Tier 1 daily patterns
  python run_backtest_selective.py --patterns "Juicer Long,RS Persistence Long,Accumulation Long"

  # Use fewer symbols for a quick check
  python run_backtest_selective.py --patterns "VP Divergence Long" --count 100 --days 90

  # List what patterns would run (dry run)
  python run_backtest_selective.py --patterns "VP Divergence Long" --dry-run
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ── reuse everything from the main backtest ──────────────────────────────────
from run_backtest import (
    _compute_pattern_stats, _save_evaluator_incremental,
    RESULTS_PATH, SYMBOLS_CACHE, TradeOutcome, TIMEFRAMES,
)
from backend.data.massive_client import fetch_bars
from backend.data.schemas import BarSeries, Bar
from backend.patterns.classifier import classify_all, _DETECTOR_MAP
from backend.patterns.registry import PATTERN_META, PatternCategory, get_patterns_for_timeframe
from backend.structures.indicators import wilder_atr
from backend.strategies.evaluator import StrategyEvaluator

# Copy of backtest_symbol_tf that accepts a pattern filter
BAR_MINUTES   = {"5min": 5, "15min": 15, "1h": 60, "1d": 390}
SCAN_INTERVAL = {"5min": 12, "15min": 6, "1h": 2, "1d": 1}
MIN_BARS      = {"5min": 60, "15min": 40, "1h": 20, "1d": 60}
DEFAULT_COOLDOWN_MIN = 120
DEFAULT_MAX_HOLD_MIN = 600

# ── PendingTrade / TradeOutcome — imported from run_backtest via module ──────
from run_backtest import PendingTrade, wilder_atr


def backtest_symbol_tf_filtered(
    symbol: str,
    timeframe: str,
    days_back: int,
    patterns_filter: set[str],
    verbose: bool = True,
) -> list[TradeOutcome]:
    """Same as backtest_symbol_tf but only runs patterns in patterns_filter."""
    try:
        bars_data = fetch_bars(symbol, timeframe, days_back)
        if not bars_data or not bars_data.bars or len(bars_data.bars) < MIN_BARS[timeframe]:
            return []
    except Exception:
        return []

    bars = bars_data.bars
    bar_min = BAR_MINUTES[timeframe]

    # Only run the TF's patterns that are in the filter
    tf_patterns = [p for p in get_patterns_for_timeframe(timeframe) if p in patterns_filter]
    if not tf_patterns:
        return []

    outcomes: list[TradeOutcome] = []
    active_trades: list[PendingTrade] = []
    cooldowns: dict[str, int] = {}
    signals_found = 0

    scan_interval = SCAN_INTERVAL.get(timeframe, 4)
    min_bars_needed = MIN_BARS.get(timeframe, 40)

    for i, bar in enumerate(bars):
        if i < min_bars_needed:
            continue

        # Resolve active trades
        remaining_trades = []
        for pt in active_trades:
            resolved, outcome = pt.check_resolution(bar)
            if resolved and outcome is not None:
                outcomes.append(outcome)
            else:
                remaining_trades.append(pt)
        active_trades = remaining_trades

        if i % scan_interval != 0:
            continue

        window = bars[max(0, i - 250): i + 1]
        s = BarSeries(symbol=symbol, timeframe=timeframe, bars=window)
        atr = wilder_atr(np.array([b.high for b in window]),
                         np.array([b.low  for b in window]),
                         np.array([b.close for b in window]), 14)
        current_atr = float(atr[-1]) if len(atr) > 0 and not np.isnan(atr[-1]) else 0.0

        for pattern_name in tf_patterns:
            meta = PATTERN_META.get(pattern_name, {})
            cd_bars = int(meta.get("cd", DEFAULT_COOLDOWN_MIN) / bar_min)
            mh_bars = int(meta.get("mh", DEFAULT_MAX_HOLD_MIN) / bar_min)

            if cooldowns.get(pattern_name, 0) > i:
                continue

            try:
                result = classify_all(s, pattern_name)
            except Exception:
                continue

            if result is None:
                continue

            if any(t.pattern_name == pattern_name for t in active_trades):
                continue

            signals_found += 1
            cooldowns[pattern_name] = i + cd_bars

            setup_t1 = result.get("target_1", 0.0) if isinstance(result, dict) else getattr(result, "target_1", 0.0)
            setup_t2 = result.get("target_2", 0.0) if isinstance(result, dict) else getattr(result, "target_2", 0.0)
            setup_splits = result.get("position_splits", (0.5, 0.3, 0.2)) if isinstance(result, dict) else getattr(result, "position_splits", (0.5, 0.3, 0.2))

            entry = result.get("entry_price", bar.close) if isinstance(result, dict) else getattr(result, "entry_price", bar.close)
            stop  = result.get("stop_loss",   0.0)      if isinstance(result, dict) else getattr(result, "stop_loss",   0.0)
            tgt   = result.get("target_price", 0.0)     if isinstance(result, dict) else getattr(result, "target_price", 0.0)
            bias  = result.get("bias", "long")           if isinstance(result, dict) else getattr(result, "bias", "long")
            if hasattr(bias, "value"):
                bias = bias.value

            active_trades.append(PendingTrade(
                pattern_name=pattern_name,
                symbol=symbol,
                entry_price=entry,
                stop_price=stop,
                target_price=tgt,
                target_1=setup_t1,
                target_2=setup_t2,
                position_splits=setup_splits,
                bias=bias,
                max_hold_bars=mh_bars,
                entry_bar_idx=i,
                entry_atr=current_atr,
            ))

    # Force-resolve remaining open trades at last bar
    if bars:
        last_bar = bars[-1]
        for pt in active_trades:
            outcome = pt.timeout_resolve(last_bar)
            if outcome is not None:
                outcomes.append(outcome)

    wins = sum(1 for o in outcomes if o.outcome in ("win", "partial_win"))
    losses = sum(1 for o in outcomes if o.outcome == "loss")
    timeouts = sum(1 for o in outcomes if o.outcome == "timeout")
    total = wins + losses + timeouts
    if verbose and total > 0:
        wr = wins / total * 100
        print(f"        {timeframe:>5}: {signals_found} signals → "
              f"W:{wins} L:{losses} T:{timeouts} ({wr:.0f}% WR)")

    return outcomes


def run_selective_backtest(
    patterns_filter: set[str],
    symbols: list[str],
    days_back: int = 90,
    include_daily: bool = True,
    verbose: bool = True,
) -> dict:
    """Run backtest for only the specified patterns, merge into existing results."""

    timeframes = ["5min", "15min", "1h"] + (["1d"] if include_daily else [])

    # Validate all requested patterns exist in the detector map
    unknown = patterns_filter - set(_DETECTOR_MAP.keys())
    if unknown:
        print(f"  ✗ Unknown patterns: {unknown}")
        sys.exit(1)

    # Show routing
    print(f"\n  Selective backtest — {len(patterns_filter)} pattern(s):")
    for tf in timeframes:
        in_tf = [p for p in get_patterns_for_timeframe(tf) if p in patterns_filter]
        if in_tf:
            print(f"    {tf:>5}: {', '.join(in_tf)}")

    print(f"\n  Symbols: {len(symbols)} | Days: {days_back}")
    print(f"{'═' * 70}")

    t_start = time.time()
    all_outcomes: list[TradeOutcome] = []
    errors: list[str] = []

    for si, symbol in enumerate(symbols):
        sym_outcomes = []
        pct = (si + 1) / len(symbols) * 100
        elapsed = time.time() - t_start
        rate = (si + 1) / elapsed if elapsed > 0 else 1
        eta = (len(symbols) - si - 1) / rate
        print(f"\n  [{si+1:>3}/{len(symbols)}] {symbol} ({pct:.0f}%) eta:{eta/60:.1f}m")

        for tf in timeframes:
            outcomes = backtest_symbol_tf_filtered(
                symbol, tf, days_back, patterns_filter, verbose=verbose
            )
            sym_outcomes.extend(outcomes)

        sym_wins  = sum(1 for o in sym_outcomes if o.outcome in ("win", "partial_win"))
        sym_total = len(sym_outcomes)
        sym_wr    = sym_wins / sym_total * 100 if sym_total > 0 else 0
        if sym_total == 0:
            errors.append(symbol)
        print(f"      TOTAL: {sym_total} signals, {sym_wr:.0f}% WR")

        all_outcomes.extend(sym_outcomes)

    # ── Compute stats for the freshly-run patterns ───────────────────────────
    new_pattern_stats: dict[str, dict] = {}
    for pattern_name in patterns_filter:
        trades = [o for o in all_outcomes if o.pattern_name == pattern_name]
        new_pattern_stats[pattern_name] = _compute_pattern_stats(pattern_name, trades)

    # ── Merge into existing results file ─────────────────────────────────────
    if RESULTS_PATH.exists():
        try:
            existing = json.loads(RESULTS_PATH.read_text())
        except Exception:
            existing = {"patterns": {}, "summary": {}, "config": {}, "symbol_stats": {}}
    else:
        existing = {"patterns": {}, "summary": {}, "config": {}, "symbol_stats": {}}

    existing_patterns = existing.get("patterns", {})
    replaced = [p for p in patterns_filter if p in existing_patterns]
    added    = [p for p in patterns_filter if p not in existing_patterns]

    for name, stats in new_pattern_stats.items():
        existing_patterns[name] = stats
    existing["patterns"] = existing_patterns
    existing["selective_update"] = {
        "updated_at": datetime.now().isoformat(),
        "patterns_replaced": replaced,
        "patterns_added":    added,
        "symbols_used":      len(symbols),
        "days_back":         days_back,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(existing, indent=2))

    # ── Update evaluator ─────────────────────────────────────────────────────
    _save_evaluator_incremental(all_outcomes)

    total_time = time.time() - t_start
    print(f"\n{'═' * 70}")
    print(f"  SELECTIVE BACKTEST COMPLETE")
    print(f"  Patterns updated: {len(patterns_filter)}")
    print(f"  Replaced: {replaced}")
    print(f"  Added:    {added}")
    print(f"  Signals:  {len(all_outcomes)}")
    print(f"  Time:     {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Saved:    {RESULTS_PATH}")
    print(f"{'═' * 70}")

    # ── Print stats for updated patterns ─────────────────────────────────────
    print(f"\n  {'Pattern':<30} {'N':>5} {'WR':>6} {'Exp':>8} {'Edge':>5}")
    print(f"  {'─' * 58}")
    for name in sorted(patterns_filter):
        s = new_pattern_stats[name]
        n = s["total_signals"]
        if n == 0:
            print(f"  {name:<30} {'NO DATA':>5}")
            continue
        print(f"  {name:<30} {n:>5} {s['win_rate']:>5.1f}% {s['expectancy']:>+7.3f}R {s['edge_score']:>5.0f}")

    return existing


def _load_cached_symbols(count: int) -> list[str]:
    if not SYMBOLS_CACHE.exists():
        print(f"  ✗ No symbol cache. Run: python fetch_symbols.py")
        sys.exit(1)
    all_syms = json.loads(SYMBOLS_CACHE.read_text()).get("symbols", [])
    return all_syms[:count]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Selective Pattern Backtest")
    parser.add_argument(
        "--patterns", type=str, required=True,
        help='Comma-separated pattern names e.g. "VP Divergence Long,VP Divergence Short"'
    )
    parser.add_argument("--count",    type=int,  default=200,  help="Symbol count (default 200)")
    parser.add_argument("--days",     type=int,  default=90,   help="Days lookback (default 90)")
    parser.add_argument("--no-daily", action="store_true",     help="Skip 1d timeframe")
    parser.add_argument("--dry-run",  action="store_true",     help="Show what would run, then exit")
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    patterns = {p.strip() for p in args.patterns.split(",") if p.strip()}

    if args.dry_run:
        print(f"\n  DRY RUN — would test {len(patterns)} pattern(s):")
        for p in sorted(patterns):
            meta = PATTERN_META.get(p, {})
            tfs  = meta.get("tf", ["?"])
            in_map = p in _DETECTOR_MAP
            print(f"    {'✓' if in_map else '✗'} {p}  [{', '.join(tfs)}]")
        unknown = patterns - set(_DETECTOR_MAP.keys())
        if unknown:
            print(f"\n  ✗ UNKNOWN (not in detector map): {unknown}")
        sys.exit(0)

    symbols = _load_cached_symbols(args.count)
    run_selective_backtest(
        patterns_filter=patterns,
        symbols=symbols,
        days_back=args.days,
        include_daily=not args.no_daily,
        verbose=not args.quiet,
    )
