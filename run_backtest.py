"""
run_backtest.py — Walk-Forward Backtest Engine v3.1

Runs all 47 pattern detectors on historical data with strict walk-forward
methodology: detect on bar N using ONLY bars 0..N, then check bars N+1
onward to see if target or stop was hit first.

Usage:
  python run_backtest.py                                  # Quick: 10 symbols, 30 days
  python run_backtest.py --symbols AAPL,NVDA,TSLA         # Specific symbols
  python run_backtest.py --from-cache --days 90            # Use cached top symbols
  python run_backtest.py --from-cache --days 90 --resume   # Resume interrupted run

Output:
  cache/backtest_results.json        — per-pattern stats (feeds scanner + AI)
  cache/strategy_performance.json    — rolling evaluator data
  cache/backtest_checkpoint.json     — checkpoint for resume
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from backend.data.massive_client import fetch_bars
from backend.data.schemas import BarSeries, Bar
from backend.patterns.classifier import classify_all
from backend.patterns.registry import PATTERN_META, PatternCategory
from backend.strategies.evaluator import StrategyEvaluator, TradeOutcome


# ==============================================================================
# CONFIG
# ==============================================================================

DEFAULT_SYMBOLS = [
    "AAPL", "NVDA", "TSLA", "MSFT", "META",
    "AMZN", "GOOGL", "AMD", "SPY", "QQQ",
]

TIMEFRAMES = ["5min", "15min"]

# Scan intervals scale with lookback to keep runtime manageable
# Format: {timeframe: bars_between_scans}
SCAN_INTERVAL_SHORT = {"5min": 6, "15min": 4}    # 30 days: every 30min / 1hr
SCAN_INTERVAL_LONG = {"5min": 12, "15min": 6}    # 60+ days: every 1hr / 1.5hr

MAX_HOLD_BARS = {"5min": 120, "15min": 60}
MIN_BARS = 60
COOLDOWN_BARS = {"5min": 24, "15min": 12}

# Checkpointing
CHECKPOINT_PATH = Path("cache/backtest_checkpoint.json")
CHECKPOINT_EVERY = 10  # Save checkpoint every N symbols

RESULTS_PATH = Path("cache/backtest_results.json")
EVALUATOR_PATH = Path("cache/strategy_performance.json")
SYMBOLS_CACHE = Path("cache/top_symbols.json")


# ==============================================================================
# OUTCOME TRACKER
# ==============================================================================

class PendingTrade:
    def __init__(self, pattern_name, strategy_type, symbol, bias, entry, target, stop, bar_idx):
        self.pattern_name = pattern_name
        self.strategy_type = strategy_type
        self.symbol = symbol
        self.bias = bias
        self.entry = entry
        self.target = target
        self.stop = stop
        self.bar_idx = bar_idx
        self.risk = abs(entry - stop)

    def check_resolution(self, bar: Bar) -> tuple[str, float] | None:
        if self.risk <= 0:
            return ("loss", -1.0)
        if self.bias == "long":
            if bar.low <= self.stop:
                return ("loss", -1.0)
            if bar.high >= self.target:
                return ("win", round((self.target - self.entry) / self.risk, 2))
        else:
            if bar.high >= self.stop:
                return ("loss", -1.0)
            if bar.low <= self.target:
                return ("win", round((self.entry - self.target) / self.risk, 2))
        return None

    def timeout_r(self, current_price: float) -> float:
        if self.risk <= 0: return 0.0
        if self.bias == "long":
            return round((current_price - self.entry) / self.risk, 2)
        return round((self.entry - current_price) / self.risk, 2)


# ==============================================================================
# SINGLE SYMBOL BACKTEST
# ==============================================================================

def backtest_symbol_tf(
    symbol: str, timeframe: str, days_back: int, verbose: bool = True,
) -> list[TradeOutcome]:
    try:
        bars_data = fetch_bars(symbol, timeframe, days_back)
    except Exception as e:
        if verbose: print(f"      ✗ FETCH ERROR: {e}")
        return []

    bars = bars_data.bars
    n = len(bars)

    if n < MIN_BARS + 20:
        if verbose: print(f"      ✗ Only {n} bars (need {MIN_BARS + 20}+)")
        return []

    if verbose:
        print(f"      {n} bars ({bars[0].timestamp.strftime('%m/%d')} → "
              f"{bars[-1].timestamp.strftime('%m/%d %H:%M')})")

    # Use wider scan intervals for long lookbacks
    scan_interval = (SCAN_INTERVAL_LONG if days_back >= 60 else SCAN_INTERVAL_SHORT).get(timeframe, 6)
    max_hold = MAX_HOLD_BARS.get(timeframe, 100)
    cooldown = COOLDOWN_BARS.get(timeframe, 20)

    outcomes: list[TradeOutcome] = []
    pending: list[PendingTrade] = []
    last_fired: dict[str, int] = {}

    wins = 0; losses = 0; timeouts = 0; signals_found = 0

    for i in range(MIN_BARS, n):
        # Check pending trades
        still_pending = []
        for trade in pending:
            bars_held = i - trade.bar_idx
            result = trade.check_resolution(bars[i])
            if result is not None:
                outcome_str, realized_r = result
                outcomes.append(TradeOutcome(
                    pattern_name=trade.pattern_name, strategy_type=trade.strategy_type,
                    symbol=symbol, bias=trade.bias,
                    entry_price=trade.entry, target_price=trade.target, stop_price=trade.stop,
                    outcome=outcome_str, realized_r=realized_r,
                    timestamp=bars[i].timestamp.isoformat(),
                ))
                if outcome_str == "win": wins += 1
                else: losses += 1
            elif bars_held >= max_hold:
                timeout_r = trade.timeout_r(bars[i].close)
                outcomes.append(TradeOutcome(
                    pattern_name=trade.pattern_name, strategy_type=trade.strategy_type,
                    symbol=symbol, bias=trade.bias,
                    entry_price=trade.entry, target_price=trade.target, stop_price=trade.stop,
                    outcome="timeout", realized_r=timeout_r,
                    timestamp=bars[i].timestamp.isoformat(),
                ))
                timeouts += 1
            else:
                still_pending.append(trade)
        pending = still_pending

        # Scan for new patterns
        if (i - MIN_BARS) % scan_interval != 0:
            continue

        window = BarSeries(symbol=symbol, timeframe=timeframe, bars=bars[:i+1])
        try:
            setups = classify_all(window)
        except Exception:
            continue

        for setup in setups:
            lf = last_fired.get(setup.pattern_name, -999)
            if i - lf < cooldown: continue
            if any(p.pattern_name == setup.pattern_name for p in pending): continue

            signals_found += 1
            last_fired[setup.pattern_name] = i
            pending.append(PendingTrade(
                pattern_name=setup.pattern_name, strategy_type=setup.strategy_type,
                symbol=symbol, bias=setup.bias.value,
                entry=setup.entry_price, target=setup.target_price, stop=setup.stop_loss,
                bar_idx=i,
            ))

    # Resolve remaining
    for trade in pending:
        timeout_r = trade.timeout_r(bars[-1].close)
        outcomes.append(TradeOutcome(
            pattern_name=trade.pattern_name, strategy_type=trade.strategy_type,
            symbol=symbol, bias=trade.bias,
            entry_price=trade.entry, target_price=trade.target, stop_price=trade.stop,
            outcome="timeout", realized_r=timeout_r,
            timestamp=bars[-1].timestamp.isoformat(),
        ))
        timeouts += 1

    total = wins + losses + timeouts
    wr = wins / total * 100 if total > 0 else 0
    if verbose:
        print(f"      {signals_found} signals → {total} resolved "
              f"(W:{wins} L:{losses} T:{timeouts}) WR:{wr:.0f}%")

    return outcomes


# ==============================================================================
# FULL BACKTEST WITH CHECKPOINTING
# ==============================================================================

def run_full_backtest(
    symbols: list[str], days_back: int = 30, verbose: bool = True,
    resume: bool = False,
) -> dict:
    t_start = time.time()

    # Load checkpoint if resuming
    completed_symbols: set[str] = set()
    all_outcomes: list[TradeOutcome] = []

    if resume and CHECKPOINT_PATH.exists():
        try:
            cp = json.loads(CHECKPOINT_PATH.read_text())
            completed_symbols = set(cp.get("completed", []))
            # Reload outcomes from evaluator
            eval_temp = StrategyEvaluator()
            eval_temp.load()
            print(f"\n  RESUMING: {len(completed_symbols)} symbols already done")
            print(f"  Remaining: {len(symbols) - len(completed_symbols)} symbols")
        except Exception:
            print("  Could not load checkpoint, starting fresh")

    remaining = [s for s in symbols if s not in completed_symbols]

    print(f"\n{'═' * 72}")
    print(f"  AlphaBean v3.1 — Walk-Forward Backtest Engine")
    print(f"  {len(symbols)} total symbols ({len(remaining)} remaining) × {len(TIMEFRAMES)} TFs × 47 patterns")
    print(f"  Lookback: {days_back} days | Scan interval: {'wide (1hr)' if days_back >= 60 else 'standard (30min)'}")
    print(f"  Checkpoint: every {CHECKPOINT_EVERY} symbols | Resume: {'yes' if resume else 'no'}")
    print(f"{'═' * 72}")

    # Time estimate
    per_sym = 20 if days_back <= 30 else 45 if days_back <= 60 else 70  # seconds
    est_total = len(remaining) * per_sym
    print(f"\n  Estimated time: ~{est_total // 3600}h {(est_total % 3600) // 60}m")
    print(f"  Checkpoint saves every {CHECKPOINT_EVERY} symbols so you can resume if interrupted\n")

    symbol_stats: dict[str, dict] = {}
    errors: list[str] = []
    batch_outcomes: list[TradeOutcome] = []

    for si, symbol in enumerate(remaining):
        sym_start = time.time()
        sym_outcomes = []

        # Progress bar
        pct = (si + 1 + len(completed_symbols)) / len(symbols) * 100
        elapsed = time.time() - t_start
        rate = (si + 1) / elapsed if elapsed > 0 else 1
        eta = (len(remaining) - si - 1) / rate if rate > 0 else 0

        # Compact progress line
        print(f"  [{si+1+len(completed_symbols):>3}/{len(symbols)}] {symbol:<6} "
              f"({pct:.0f}%) elapsed:{elapsed/60:.1f}m eta:{eta/60:.1f}m", end="")

        for tf in TIMEFRAMES:
            outcomes = backtest_symbol_tf(symbol, tf, days_back, verbose=False)
            sym_outcomes.extend(outcomes)

        sym_time = time.time() - sym_start
        sym_wins = sum(1 for o in sym_outcomes if o.outcome == "win")
        sym_total = len(sym_outcomes)
        sym_wr = sym_wins / sym_total * 100 if sym_total > 0 else 0

        if sym_total == 0:
            print(f" → 0 signals ({sym_time:.1f}s)")
            errors.append(symbol)
        else:
            print(f" → {sym_total} signals, {sym_wr:.0f}% WR ({sym_time:.1f}s)")

        symbol_stats[symbol] = {
            "total_signals": sym_total, "wins": sym_wins,
            "win_rate": round(sym_wr, 1), "time_seconds": round(sym_time, 1),
        }

        batch_outcomes.extend(sym_outcomes)
        all_outcomes.extend(sym_outcomes)
        completed_symbols.add(symbol)

        # Checkpoint
        if (si + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(completed_symbols, symbol_stats)
            # Also incrementally save evaluator
            _save_evaluator_incremental(batch_outcomes)
            batch_outcomes = []

            done = si + 1 + len(completed_symbols) - len(remaining)
            print(f"\n  ✓ CHECKPOINT: {len(completed_symbols)}/{len(symbols)} symbols saved\n")

    # Save remaining batch
    if batch_outcomes:
        _save_evaluator_incremental(batch_outcomes)

    # ─── Compute pattern stats ────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  COMPUTING PATTERN STATISTICS ({len(all_outcomes)} total outcomes)")
    print(f"{'═' * 72}")

    pattern_stats = {}
    for pattern_name in sorted(set(o.pattern_name for o in all_outcomes)):
        trades = [o for o in all_outcomes if o.pattern_name == pattern_name]
        stats = _compute_pattern_stats(pattern_name, trades)
        pattern_stats[pattern_name] = stats

    # Print results table
    print(f"\n{'─' * 95}")
    print(f"  {'Pattern':<28} {'Cat':<10} {'N':>5} {'WR':>6} {'PF':>6} "
          f"{'Exp':>8} {'AvgW':>6} {'AvgL':>6} {'Edge':>5} {'Grade':<5}")
    print(f"{'─' * 95}")

    sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1]["edge_score"], reverse=True)

    for name, s in sorted_patterns:
        meta = PATTERN_META.get(name, {})
        cat = meta.get("cat", PatternCategory.CLASSICAL).value[:8]
        n = s["total_signals"]
        if n < 3: continue

        wr = s["win_rate"]; pf = s["profit_factor"]; exp = s["expectancy"]
        aw = s["avg_win_r"]; al = s["avg_loss_r"]; edge = s["edge_score"]

        # Grade
        if edge >= 70 and n >= 20: grade = "A"
        elif edge >= 55 and n >= 10: grade = "B"
        elif edge >= 40 and n >= 5: grade = "C"
        elif edge >= 25: grade = "D"
        else: grade = "F"

        marker = "★" if grade in ("A", "B") else "☆" if grade == "C" else " "

        print(f"  {marker} {name:<26} {cat:<10} {n:>5} {wr:>5.1f}% {pf:>5.1f} "
              f"{exp:>+7.3f}R {aw:>5.1f}R {al:>5.1f}R {edge:>5.0f}  {grade}")

    # ─── Grade summary ────────────────────────────────────────────
    grades = defaultdict(list)
    for name, s in sorted_patterns:
        n = s["total_signals"]; edge = s["edge_score"]
        if n < 3: continue
        if edge >= 70 and n >= 20: grades["A"].append(name)
        elif edge >= 55 and n >= 10: grades["B"].append(name)
        elif edge >= 40 and n >= 5: grades["C"].append(name)
        elif edge >= 25: grades["D"].append(name)
        else: grades["F"].append(name)

    print(f"\n  Grade Summary:")
    for grade in ["A", "B", "C", "D", "F"]:
        names = grades.get(grade, [])
        if names:
            shown = ", ".join(names[:6])
            extra = f" +{len(names)-6} more" if len(names) > 6 else ""
            print(f"    {grade}: {shown}{extra}")

    # ─── Save final evaluator ─────────────────────────────────────
    evaluator = StrategyEvaluator()
    evaluator.record_batch(all_outcomes)
    evaluator.save()

    rankings = evaluator.get_rankings()
    print(f"\n  Evaluator: {len(all_outcomes)} outcomes loaded")
    print(f"  Top 10 by hot score:")
    for i, m in enumerate(rankings[:10]):
        bar = "█" * int(m.hot_score / 5)
        print(f"    #{i+1:>2} {m.name:<26} {bar} {m.hot_score:.0f}  "
              f"WR={m.win_rate:.0%} PF={m.profit_factor:.1f} Exp={m.expectancy:+.2f}R")

    # ─── Save results ─────────────────────────────────────────────
    total_time = time.time() - t_start
    total_wins = sum(1 for o in all_outcomes if o.outcome == "win")

    results = {
        "version": "3.1",
        "run_date": datetime.now().isoformat(),
        "config": {
            "symbols": list(completed_symbols),
            "symbol_count": len(completed_symbols),
            "days_back": days_back,
            "timeframes": TIMEFRAMES,
        },
        "summary": {
            "total_symbols": len(completed_symbols),
            "total_signals": len(all_outcomes),
            "total_wins": total_wins,
            "total_losses": sum(1 for o in all_outcomes if o.outcome == "loss"),
            "total_timeouts": sum(1 for o in all_outcomes if o.outcome == "timeout"),
            "overall_win_rate": round(total_wins / len(all_outcomes) * 100, 1) if all_outcomes else 0,
            "total_time_seconds": round(total_time, 1),
            "errors": errors,
        },
        "symbol_stats": symbol_stats,
        "patterns": pattern_stats,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"\n{'═' * 72}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'═' * 72}")
    print(f"  Symbols:      {len(completed_symbols)}")
    print(f"  Signals:      {len(all_outcomes)}")
    print(f"  Win Rate:     {total_wins / len(all_outcomes) * 100:.1f}%" if all_outcomes else "  Win Rate:     N/A")
    print(f"  Time:         {total_time:.0f}s ({total_time/60:.1f}m / {total_time/3600:.1f}h)")
    print(f"  Errors:       {len(errors)} symbols failed to fetch")
    print(f"  Saved to:     {RESULTS_PATH}")
    print(f"  Evaluator:    {EVALUATOR_PATH}")
    print(f"{'═' * 72}\n")

    return results


# ==============================================================================
# HELPERS
# ==============================================================================

def _compute_pattern_stats(pattern_name: str, trades: list[TradeOutcome]) -> dict:
    n = len(trades)
    if n == 0:
        return {"total_signals": 0, "wins": 0, "losses": 0, "timeouts": 0,
                "win_rate": 0, "profit_factor": 0, "expectancy": 0,
                "avg_win_r": 0, "avg_loss_r": 0, "best_r": 0, "worst_r": 0,
                "edge_score": 0, "sample_size": 0}

    wins = [t for t in trades if t.outcome == "win"]
    losses = [t for t in trades if t.outcome == "loss"]

    n_wins = len(wins)
    wr = n_wins / n if n > 0 else 0

    all_r = [t.realized_r for t in trades]
    win_rs = [t.realized_r for t in wins]
    loss_rs = [abs(t.realized_r) for t in losses]

    avg_win_r = float(np.mean(win_rs)) if win_rs else 0
    avg_loss_r = float(np.mean(loss_rs)) if loss_rs else 0

    gross_win = sum(win_rs) if win_rs else 0
    gross_loss = sum(loss_rs) if loss_rs else 0
    pf = gross_win / gross_loss if gross_loss > 0 else (99.0 if gross_win > 0 else 0)

    expectancy = (wr * avg_win_r) - ((1 - wr) * avg_loss_r)

    # Edge score
    wr_score = max(0, min(100, (wr - 0.35) / 0.35 * 100))
    pf_score = max(0, min(100, (min(pf, 5.0) - 0.5) / 2.5 * 100))
    exp_score = max(0, min(100, (expectancy + 0.5) / 1.5 * 100))
    sample_score = min(100, n / 50 * 100)  # Scaled for larger sample
    r_std = float(np.std(all_r)) if len(all_r) > 1 else 2.0
    consistency_score = max(0, min(100, (3.0 - r_std) / 3.0 * 100))

    edge = (wr_score * 0.30 + pf_score * 0.25 + exp_score * 0.20 +
            sample_score * 0.15 + consistency_score * 0.10)
    edge = round(max(0, min(100, edge)), 1)

    return {
        "total_signals": n, "wins": n_wins, "losses": len(losses),
        "timeouts": len([t for t in trades if t.outcome == "timeout"]),
        "win_rate": round(wr * 100, 1), "profit_factor": round(min(pf, 99.0), 2),
        "expectancy": round(expectancy, 3),
        "avg_win_r": round(avg_win_r, 2), "avg_loss_r": round(avg_loss_r, 2),
        "best_r": round(max(all_r), 2), "worst_r": round(min(all_r), 2),
        "edge_score": edge, "sample_size": n,
    }


def _save_checkpoint(completed: set[str], symbol_stats: dict):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cp = {
        "completed": list(completed),
        "symbol_stats": symbol_stats,
        "saved_at": datetime.now().isoformat(),
    }
    CHECKPOINT_PATH.write_text(json.dumps(cp, indent=2))


def _save_evaluator_incremental(outcomes: list[TradeOutcome]):
    """Incrementally add outcomes to the evaluator cache."""
    evaluator = StrategyEvaluator()
    evaluator.load()
    evaluator.record_batch(outcomes)
    evaluator.save()


def _load_cached_symbols() -> list[str]:
    if not SYMBOLS_CACHE.exists():
        print(f"  ✗ No cached symbols. Run: python fetch_symbols.py")
        sys.exit(1)
    data = json.loads(SYMBOLS_CACHE.read_text())
    return data.get("symbols", [])


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="AlphaBean v3.1 Walk-Forward Backtest")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated: AAPL,NVDA,TSLA")
    parser.add_argument("--count", type=int, default=10,
                        help="N symbols from default list (default: 10)")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of history (default: 30)")
    parser.add_argument("--from-cache", action="store_true",
                        help="Use symbols from cache/top_symbols.json")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--quiet", action="store_true",
                        help="Compact output")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.from_cache:
        symbols = _load_cached_symbols()
        print(f"  Loaded {len(symbols)} symbols from cache")
    else:
        symbols = DEFAULT_SYMBOLS[:args.count]

    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  AlphaBean v3.1 — Walk-Forward Backtest             ║")
    print(f"  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Symbols:     {len(symbols):<40}║")
    print(f"  ║  Days back:   {args.days:<40}║")
    print(f"  ║  Timeframes:  {', '.join(TIMEFRAMES):<40}║")
    print(f"  ║  Patterns:    47 (16+10+11+10)                      ║")
    print(f"  ║  Method:      Strict walk-forward                   ║")
    if args.resume:
        print(f"  ║  Mode:        RESUME from checkpoint                ║")
    print(f"  ╚══════════════════════════════════════════════════════╝")

    run_full_backtest(symbols, days_back=args.days, verbose=not args.quiet,
                      resume=args.resume)


if __name__ == "__main__":
    main()