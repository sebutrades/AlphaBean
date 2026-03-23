"""
run_backtest.py — Walk-Forward Backtest Engine v3.3

v3.3 changes:
  - 4 timeframes: 5min, 15min, 1h, 1d (was just 5min+15min)
  - Patterns only run on their designated TFs (from registry "tf" field)
  - Per-pattern cooldown (cd) and max_hold (mh) from registry
  - Composite cooldown key (pattern + bias + price bucket)
  - 1h structural patterns: cleaner geometry, fewer false signals

Usage:
  python run_backtest.py                                  # Quick: 10 symbols, 30 days
  python run_backtest.py --symbols AAPL,NVDA,TSLA         # Specific symbols
  python run_backtest.py --count 50 --days 90             # 50 symbols, 90 days
  python run_backtest.py --from-cache --days 90 --resume  # Resume interrupted run
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
from backend.patterns.registry import PATTERN_META, PatternCategory, get_patterns_for_timeframe


# ==============================================================================
# CONFIG
# ==============================================================================

DEFAULT_SYMBOLS = [
    "AAPL", "NVDA", "TSLA", "MSFT", "META",
    "AMZN", "GOOGL", "AMD", "SPY", "QQQ",
]

# All 4 timeframes — each pattern only runs on its designated TFs
TIMEFRAMES = ["5min", "15min", "1h"]
# Note: "1d" excluded from default backtest — requires 250+ days of daily data
# and separate fetch. Add with --daily flag if desired.

# Bar duration in minutes for converting cd/mh to bars
BAR_MINUTES = {"5min": 5, "15min": 15, "1h": 60, "1d": 390}

# How often to scan (in bars) per timeframe
SCAN_INTERVAL = {
    "5min":  6,    # every 30 min
    "15min": 4,    # every 1 hr
    "1h":    2,    # every 2 hrs
    "1d":    1,    # every day
}

# Wider scan intervals for long lookbacks (60+ days)
SCAN_INTERVAL_LONG = {
    "5min":  12,   # every 1 hr
    "15min": 6,    # every 1.5 hr
    "1h":    2,    # every 2 hrs (unchanged — already wide)
    "1d":    1,
}

# Minimum bars needed before first scan
MIN_BARS = {"5min": 60, "15min": 40, "1h": 20, "1d": 60}

# Defaults for patterns missing cd/mh in registry
DEFAULT_COOLDOWN_MIN = 120
DEFAULT_MAX_HOLD_MIN = 600

# Checkpointing
CHECKPOINT_PATH = Path("cache/backtest_checkpoint.json")
CHECKPOINT_EVERY = 10
RESULTS_PATH = Path("cache/backtest_results.json")
EVALUATOR_PATH = Path("cache/strategy_performance.json")
SYMBOLS_CACHE = Path("cache/top_symbols.json")

# Import evaluator
from backend.strategies.evaluator import StrategyEvaluator, TradeOutcome


# ==============================================================================
# HELPERS: Per-pattern parameters from registry
# ==============================================================================

def _cooldown_bars(pattern_name: str, timeframe: str) -> int:
    meta = PATTERN_META.get(pattern_name, {})
    cd_min = meta.get("cd", DEFAULT_COOLDOWN_MIN)
    bar_min = BAR_MINUTES.get(timeframe, 5)
    return max(1, int(cd_min / bar_min))

def _max_hold_bars(pattern_name: str, timeframe: str) -> int:
    meta = PATTERN_META.get(pattern_name, {})
    mh_min = meta.get("mh", DEFAULT_MAX_HOLD_MIN)
    bar_min = BAR_MINUTES.get(timeframe, 5)
    return max(1, int(mh_min / bar_min))


# ==============================================================================
# PENDING TRADE
# ==============================================================================

class PendingTrade:
    def __init__(self, pattern_name, strategy_type, symbol, bias, entry, target, stop, bar_idx, max_hold):
        self.pattern_name = pattern_name
        self.strategy_type = strategy_type
        self.symbol = symbol
        self.bias = bias
        self.entry = entry
        self.target = target
        self.stop = stop
        self.bar_idx = bar_idx
        self.max_hold = max_hold
        self.risk = abs(entry - stop)

    def check_resolution(self, bar: Bar) -> tuple[str, float] | None:
        if self.risk <= 0:
            return ("loss", -1.0)
        if self.bias == "long":
            if bar.low <= self.stop: return ("loss", -1.0)
            if bar.high >= self.target:
                return ("win", round((self.target - self.entry) / self.risk, 2))
        else:
            if bar.high >= self.stop: return ("loss", -1.0)
            if bar.low <= self.target:
                return ("win", round((self.entry - self.target) / self.risk, 2))
        return None

    def timeout_r(self, current_price: float) -> float:
        if self.risk <= 0: return 0.0
        if self.bias == "long":
            return round((current_price - self.entry) / self.risk, 2)
        return round((self.entry - current_price) / self.risk, 2)


# ==============================================================================
# SINGLE SYMBOL × SINGLE TIMEFRAME BACKTEST
# ==============================================================================

def backtest_symbol_tf(
    symbol: str, timeframe: str, days_back: int, verbose: bool = True,
) -> list[TradeOutcome]:
    """Walk-forward backtest for one symbol on one timeframe.

    classify_all() handles TF routing — it only runs patterns whose
    registry "tf" field includes this timeframe.
    """
    try:
        bars_data = fetch_bars(symbol, timeframe, days_back)
    except Exception as e:
        if verbose: print(f"        ✗ {timeframe} FETCH ERROR: {e}")
        return []

    bars = bars_data.bars
    n = len(bars)
    min_bars = MIN_BARS.get(timeframe, 60)

    if n < min_bars + 10:
        if verbose: print(f"        ✗ {timeframe}: {n} bars (need {min_bars + 10}+)")
        return []

    scan_interval = (SCAN_INTERVAL_LONG if days_back >= 60 else SCAN_INTERVAL).get(timeframe, 4)

    outcomes: list[TradeOutcome] = []
    pending: list[PendingTrade] = []
    last_fired: dict[str, int] = {}
    wins = 0; losses = 0; timeouts = 0; signals_found = 0

    for i in range(min_bars, n):
        # Check pending trades (each has per-pattern max_hold)
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
            elif bars_held >= trade.max_hold:
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

        # Scan at intervals
        if (i - min_bars) % scan_interval != 0:
            continue

        window = BarSeries(symbol=symbol, timeframe=timeframe, bars=bars[:i+1])
        try:
            setups = classify_all(window)  # TF routing happens inside
        except Exception:
            continue

        for setup in setups:
            # Composite cooldown: pattern + bias + price level
            price_bucket = round(setup.entry_price, 0)
            cooldown_key = f"{setup.pattern_name}_{setup.bias.value}_{price_bucket}"

            cd_bars = _cooldown_bars(setup.pattern_name, timeframe)
            lf = last_fired.get(cooldown_key, -9999)
            if i - lf < cd_bars: continue
            if any(p.pattern_name == setup.pattern_name for p in pending): continue

            mh_bars = _max_hold_bars(setup.pattern_name, timeframe)
            signals_found += 1
            last_fired[cooldown_key] = i
            pending.append(PendingTrade(
                pattern_name=setup.pattern_name, strategy_type=setup.strategy_type,
                symbol=symbol, bias=setup.bias.value,
                entry=setup.entry_price, target=setup.target_price, stop=setup.stop_loss,
                bar_idx=i, max_hold=mh_bars,
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
    if verbose and total > 0:
        wr = wins / total * 100
        print(f"        {timeframe:>5}: {signals_found} signals → "
              f"W:{wins} L:{losses} T:{timeouts} ({wr:.0f}% WR)")

    return outcomes


# ==============================================================================
# FULL BACKTEST
# ==============================================================================

def run_full_backtest(
    symbols: list[str], days_back: int = 30, verbose: bool = True,
    resume: bool = False, include_daily: bool = False,
) -> dict:
    t_start = time.time()
    timeframes = TIMEFRAMES + (["1d"] if include_daily else [])

    # Show what runs where
    if verbose:
        print(f"\n  Pattern routing:")
        for tf in timeframes:
            patterns = get_patterns_for_timeframe(tf)
            print(f"    {tf:>5}: {len(patterns)} patterns")

    completed_symbols: set[str] = set()
    all_outcomes: list[TradeOutcome] = []

    if resume and CHECKPOINT_PATH.exists():
        try:
            cp = json.loads(CHECKPOINT_PATH.read_text())
            completed_symbols = set(cp.get("completed", []))
            print(f"\n  RESUMING: {len(completed_symbols)} done, "
                  f"{len(symbols) - len(completed_symbols)} remaining")
        except Exception:
            pass

    remaining = [s for s in symbols if s not in completed_symbols]

    print(f"\n{'═' * 72}")
    print(f"  Juicer v2.1 — Walk-Forward Backtest Engine")
    print(f"  {len(symbols)} symbols × {len(timeframes)} TFs × 42 patterns (TF-routed)")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Lookback: {days_back} days | Per-pattern cooldown + max_hold")
    print(f"{'═' * 72}")

    per_sym = len(timeframes) * (8 if days_back <= 30 else 15)
    est_total = len(remaining) * per_sym
    print(f"\n  Estimated: ~{est_total // 60}m ({est_total // 3600}h {(est_total % 3600) // 60}m)")

    symbol_stats: dict[str, dict] = {}
    errors: list[str] = []
    batch_outcomes: list[TradeOutcome] = []

    for si, symbol in enumerate(remaining):
        sym_start = time.time()
        sym_outcomes = []

        pct = (si + 1 + len(completed_symbols)) / len(symbols) * 100
        elapsed = time.time() - t_start
        rate = (si + 1) / elapsed if elapsed > 0 else 1
        eta = (len(remaining) - si - 1) / rate if rate > 0 else 0

        print(f"\n  [{si+1+len(completed_symbols):>3}/{len(symbols)}] {symbol} "
              f"({pct:.0f}%) eta:{eta/60:.1f}m")

        for tf in timeframes:
            # How many patterns will run on this TF?
            n_patterns = len(get_patterns_for_timeframe(tf))
            if n_patterns == 0:
                continue

            outcomes = backtest_symbol_tf(symbol, tf, days_back, verbose=verbose)
            sym_outcomes.extend(outcomes)

        sym_time = time.time() - sym_start
        sym_wins = sum(1 for o in sym_outcomes if o.outcome == "win")
        sym_total = len(sym_outcomes)
        sym_wr = sym_wins / sym_total * 100 if sym_total > 0 else 0

        if sym_total == 0:
            errors.append(symbol)
        print(f"      TOTAL: {sym_total} signals, {sym_wr:.0f}% WR ({sym_time:.1f}s)")

        symbol_stats[symbol] = {
            "total_signals": sym_total, "wins": sym_wins,
            "win_rate": round(sym_wr, 1), "time_seconds": round(sym_time, 1),
        }

        batch_outcomes.extend(sym_outcomes)
        all_outcomes.extend(sym_outcomes)
        completed_symbols.add(symbol)

        if (si + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(completed_symbols, symbol_stats)
            _save_evaluator_incremental(batch_outcomes)
            batch_outcomes = []
            print(f"\n  ✓ CHECKPOINT: {len(completed_symbols)}/{len(symbols)} saved")

    if batch_outcomes:
        _save_evaluator_incremental(batch_outcomes)

    # ─── Pattern stats ────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  PATTERN STATISTICS ({len(all_outcomes)} outcomes)")
    print(f"{'═' * 72}")

    pattern_stats = {}
    for pattern_name in sorted(set(o.pattern_name for o in all_outcomes)):
        trades = [o for o in all_outcomes if o.pattern_name == pattern_name]
        stats = _compute_pattern_stats(pattern_name, trades)
        pattern_stats[pattern_name] = stats

    print(f"\n{'─' * 100}")
    print(f"  {'Pattern':<28} {'Cat':<10} {'TFs':<10} {'N':>5} {'WR':>6} {'PF':>6} "
          f"{'Exp':>8} {'AvgW':>6} {'AvgL':>6} {'Edge':>5} {'Gr':<3}")
    print(f"{'─' * 100}")

    sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1]["edge_score"], reverse=True)

    for name, s in sorted_patterns:
        meta = PATTERN_META.get(name, {})
        cat = meta.get("cat", PatternCategory.CLASSICAL).value[:8]
        tfs = ",".join(meta.get("tf", ["?"]))[:9]
        n = s["total_signals"]
        if n < 3: continue

        wr = s["win_rate"]; pf = s["profit_factor"]; exp = s["expectancy"]
        aw = s["avg_win_r"]; al = s["avg_loss_r"]; edge = s["edge_score"]

        if edge >= 70 and n >= 20: grade = "A"
        elif edge >= 55 and n >= 10: grade = "B"
        elif edge >= 40 and n >= 5: grade = "C"
        elif edge >= 25: grade = "D"
        else: grade = "F"

        marker = "★" if grade in ("A", "B") else "☆" if grade == "C" else " "
        print(f"  {marker} {name:<26} {cat:<10} {tfs:<10} {n:>5} {wr:>5.1f}% {pf:>5.1f} "
              f"{exp:>+7.3f}R {aw:>5.1f}R {al:>5.1f}R {edge:>5.0f}  {grade}")

    # Grade summary
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

    # TF coverage summary
    tf_counts = defaultdict(int)
    for o in all_outcomes:
        meta = PATTERN_META.get(o.pattern_name, {})
        for tf in meta.get("tf", []):
            tf_counts[tf] += 1
    print(f"\n  Signals by designated TF:")
    for tf in timeframes:
        print(f"    {tf:>5}: {tf_counts.get(tf, 0)} signals")

    # Evaluator
    evaluator = StrategyEvaluator()
    evaluator.record_batch(all_outcomes)
    evaluator.save()

    rankings = evaluator.get_rankings()
    print(f"\n  Top 10 by hot score:")
    for i, m in enumerate(rankings[:10]):
        bar = "█" * int(m.hot_score / 5)
        print(f"    #{i+1:>2} {m.name:<26} {bar} {m.hot_score:.0f}  "
              f"WR={m.win_rate:.0%} PF={m.profit_factor:.1f} Exp={m.expectancy:+.2f}R")

    # Save
    total_time = time.time() - t_start
    total_wins = sum(1 for o in all_outcomes if o.outcome == "win")

    results = {
        "version": "3.3",
        "run_date": datetime.now().isoformat(),
        "config": {
            "symbols": list(completed_symbols),
            "symbol_count": len(completed_symbols),
            "days_back": days_back,
            "timeframes": timeframes,
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

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"\n{'═' * 72}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'═' * 72}")
    print(f"  Symbols:      {len(completed_symbols)}")
    print(f"  Timeframes:   {', '.join(timeframes)}")
    print(f"  Signals:      {len(all_outcomes)}")
    print(f"  Win Rate:     {total_wins / len(all_outcomes) * 100:.1f}%" if all_outcomes else "  Win Rate:     N/A")
    print(f"  Time:         {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Saved:        {RESULTS_PATH}")
    print(f"{'═' * 72}\n")

    return results


# ==============================================================================
# STATS
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
    wr = n_wins / n

    all_r = [t.realized_r for t in trades]
    win_rs = [t.realized_r for t in wins]
    loss_rs = [abs(t.realized_r) for t in losses]

    avg_win_r = float(np.mean(win_rs)) if win_rs else 0
    avg_loss_r = float(np.mean(loss_rs)) if loss_rs else 0

    gross_win = sum(win_rs) if win_rs else 0
    gross_loss = sum(loss_rs) if loss_rs else 0
    pf = gross_win / gross_loss if gross_loss > 0 else (99.0 if gross_win > 0 else 0)

    expectancy = (wr * avg_win_r) - ((1 - wr) * avg_loss_r)

    wr_score = max(0, min(100, (wr - 0.35) / 0.35 * 100))
    pf_score = max(0, min(100, (min(pf, 5.0) - 0.5) / 2.5 * 100))
    exp_score = max(0, min(100, (expectancy + 0.5) / 1.5 * 100))
    sample_score = min(100, n / 50 * 100)
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
    CHECKPOINT_PATH.write_text(json.dumps({
        "completed": list(completed), "symbol_stats": symbol_stats,
        "saved_at": datetime.now().isoformat(),
    }, indent=2))

def _save_evaluator_incremental(outcomes: list[TradeOutcome]):
    evaluator = StrategyEvaluator()
    evaluator.load()
    evaluator.record_batch(outcomes)
    evaluator.save()

def _load_cached_symbols() -> list[str]:
    if not SYMBOLS_CACHE.exists():
        print(f"  ✗ No cached symbols. Run: python fetch_symbols.py")
        sys.exit(1)
    return json.loads(SYMBOLS_CACHE.read_text()).get("symbols", [])


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Juicer v2.1 Walk-Forward Backtest")
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--daily", action="store_true",
                        help="Include 1d timeframe (requires 250+ days lookback)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.from_cache:
        symbols = _load_cached_symbols()
        print(f"  Loaded {len(symbols)} symbols from cache")
    else:
        symbols = DEFAULT_SYMBOLS[:args.count]

    tfs = TIMEFRAMES + (["1d"] if args.daily else [])

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  Juicer v2.1 — Walk-Forward Backtest                   ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Symbols:     {len(symbols):<42}║")
    print(f"  ║  Days back:   {args.days:<42}║")
    print(f"  ║  Timeframes:  {', '.join(tfs):<42}║")
    print(f"  ║  Patterns:    42 (TF-routed per registry)              ║")
    print(f"  ║  Method:      Strict walk-forward, per-pattern params  ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

    run_full_backtest(symbols, days_back=args.days, verbose=not args.quiet,
                      resume=args.resume, include_daily=args.daily)


if __name__ == "__main__":
    main()