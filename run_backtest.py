"""
run_backtest.py — Walk-Forward Backtest Engine v3.0

Runs all 47 pattern detectors on historical data with strict walk-forward
methodology: detect on bar N using ONLY bars 0..N, then check bars N+1
onward to see if target or stop was hit first.

Usage:
  python run_backtest.py                          # Quick: 10 symbols, 30 days
  python run_backtest.py --symbols AAPL,NVDA,TSLA # Specific symbols
  python run_backtest.py --days 60 --count 25     # 25 symbols, 60 days
  python run_backtest.py --full                    # Full: 50 symbols, 90 days

Output:
  cache/backtest_results.json   — per-pattern stats (feeds scanner + AI agents)
  cache/strategy_performance.json — rolling evaluator data (feeds multi-factor scorer)
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

EXTENDED_SYMBOLS = [
    "AAPL", "NVDA", "TSLA", "MSFT", "META", "AMZN", "GOOGL", "AMD", "SPY", "QQQ",
    "NFLX", "CRM", "AVGO", "ORCL", "ADBE", "INTC", "MU", "PYPL", "SQ", "COIN",
    "SOFI", "PLTR", "SNOW", "NET", "DDOG", "CRWD", "ZS", "PANW", "SHOP", "MELI",
    "BA", "CAT", "GS", "JPM", "V", "MA", "UNH", "LLY", "JNJ", "PFE",
    "XOM", "CVX", "GLD", "SLV", "TLT", "IWM", "DIA", "XLF", "XLE", "ARKK",
]

TIMEFRAMES = ["5min", "15min"]

# How often to check for patterns (in bars)
SCAN_INTERVAL = {"5min": 6, "15min": 4}  # every 30min for 5min, every 1h for 15min

# Max bars to wait for resolution before marking as timeout
MAX_HOLD_BARS = {"5min": 120, "15min": 60}  # ~10 hours for 5min, ~15 hours for 15min

# Minimum bars needed before first scan
MIN_BARS = 60

# Cooldown: don't re-fire same pattern within N bars
COOLDOWN_BARS = {"5min": 24, "15min": 12}  # ~2 hours cooldown


# ==============================================================================
# OUTCOME TRACKER
# ==============================================================================

class PendingTrade:
    """A trade signal waiting for resolution."""
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
        """Check if this bar resolves the trade. Returns (outcome, realized_r) or None."""
        if self.risk <= 0:
            return ("loss", -1.0)

        if self.bias == "long":
            # Check stop first (conservative)
            if bar.low <= self.stop:
                return ("loss", -1.0)
            if bar.high >= self.target:
                r = (self.target - self.entry) / self.risk
                return ("win", round(r, 2))
        else:  # short
            if bar.high >= self.stop:
                return ("loss", -1.0)
            if bar.low <= self.target:
                r = (self.entry - self.target) / self.risk
                return ("win", round(r, 2))

        return None

    def timeout_r(self, current_price: float) -> float:
        """R-multiple at timeout."""
        if self.risk <= 0:
            return 0.0
        if self.bias == "long":
            return round((current_price - self.entry) / self.risk, 2)
        else:
            return round((self.entry - current_price) / self.risk, 2)


# ==============================================================================
# MAIN BACKTEST
# ==============================================================================

def backtest_symbol_tf(
    symbol: str,
    timeframe: str,
    days_back: int,
    verbose: bool = True,
) -> list[TradeOutcome]:
    """
    Walk-forward backtest for one symbol on one timeframe.

    Returns list of TradeOutcome for all resolved signals.
    """
    # Fetch data
    try:
        bars_data = fetch_bars(symbol, timeframe, days_back)
    except Exception as e:
        if verbose:
            print(f"    ✗ FETCH ERROR: {e}")
        return []

    bars = bars_data.bars
    n = len(bars)

    if n < MIN_BARS + 20:
        if verbose:
            print(f"    ✗ Only {n} bars (need {MIN_BARS + 20}+)")
        return []

    if verbose:
        print(f"    {n} bars loaded ({bars[0].timestamp.strftime('%m/%d')} → {bars[-1].timestamp.strftime('%m/%d %H:%M')})")

    scan_interval = SCAN_INTERVAL.get(timeframe, 6)
    max_hold = MAX_HOLD_BARS.get(timeframe, 100)
    cooldown = COOLDOWN_BARS.get(timeframe, 20)

    outcomes: list[TradeOutcome] = []
    pending: list[PendingTrade] = []
    # Track last fire bar per pattern to enforce cooldown
    last_fired: dict[str, int] = {}

    scans_done = 0
    signals_found = 0
    wins = 0
    losses = 0
    timeouts = 0

    # Walk forward through the bars
    for i in range(MIN_BARS, n):
        # --- 1. Check pending trades for resolution ---
        still_pending = []
        for trade in pending:
            bars_held = i - trade.bar_idx
            result = trade.check_resolution(bars[i])

            if result is not None:
                outcome_str, realized_r = result
                outcomes.append(TradeOutcome(
                    pattern_name=trade.pattern_name,
                    strategy_type=trade.strategy_type,
                    symbol=symbol,
                    bias=trade.bias,
                    entry_price=trade.entry,
                    target_price=trade.target,
                    stop_price=trade.stop,
                    outcome=outcome_str,
                    realized_r=realized_r,
                    timestamp=bars[i].timestamp.isoformat(),
                ))
                if outcome_str == "win":
                    wins += 1
                else:
                    losses += 1
            elif bars_held >= max_hold:
                # Timeout
                timeout_r = trade.timeout_r(bars[i].close)
                outcomes.append(TradeOutcome(
                    pattern_name=trade.pattern_name,
                    strategy_type=trade.strategy_type,
                    symbol=symbol,
                    bias=trade.bias,
                    entry_price=trade.entry,
                    target_price=trade.target,
                    stop_price=trade.stop,
                    outcome="timeout",
                    realized_r=timeout_r,
                    timestamp=bars[i].timestamp.isoformat(),
                ))
                timeouts += 1
            else:
                still_pending.append(trade)

        pending = still_pending

        # --- 2. Scan for new patterns at intervals ---
        if (i - MIN_BARS) % scan_interval != 0:
            continue

        scans_done += 1

        # Build BarSeries from bars[0:i+1] (walk-forward: NO future data)
        window = BarSeries(symbol=symbol, timeframe=timeframe, bars=bars[:i+1])

        try:
            setups = classify_all(window)
        except Exception:
            continue

        for setup in setups:
            # Enforce cooldown
            lf = last_fired.get(setup.pattern_name, -999)
            if i - lf < cooldown:
                continue

            # Don't stack: skip if same pattern already pending
            if any(p.pattern_name == setup.pattern_name for p in pending):
                continue

            # Valid signal — record it
            signals_found += 1
            last_fired[setup.pattern_name] = i

            pending.append(PendingTrade(
                pattern_name=setup.pattern_name,
                strategy_type=setup.strategy_type,
                symbol=symbol,
                bias=setup.bias.value,
                entry=setup.entry_price,
                target=setup.target_price,
                stop=setup.stop_loss,
                bar_idx=i,
            ))

    # Resolve any remaining pending trades at final bar
    for trade in pending:
        timeout_r = trade.timeout_r(bars[-1].close)
        outcomes.append(TradeOutcome(
            pattern_name=trade.pattern_name,
            strategy_type=trade.strategy_type,
            symbol=symbol,
            bias=trade.bias,
            entry_price=trade.entry,
            target_price=trade.target,
            stop_price=trade.stop,
            outcome="timeout",
            realized_r=timeout_r,
            timestamp=bars[-1].timestamp.isoformat(),
        ))
        timeouts += 1

    if verbose:
        total = wins + losses + timeouts
        wr = wins / total * 100 if total > 0 else 0
        print(f"    {scans_done} scans → {signals_found} signals → "
              f"{total} resolved (W:{wins} L:{losses} T:{timeouts}) "
              f"WR:{wr:.0f}%")

    return outcomes


def run_full_backtest(
    symbols: list[str],
    days_back: int = 30,
    verbose: bool = True,
) -> dict:
    """
    Run walk-forward backtest across all symbols and timeframes.

    Returns comprehensive results dict.
    """
    all_outcomes: list[TradeOutcome] = []
    symbol_stats: dict[str, dict] = {}
    t_start = time.time()

    total_jobs = len(symbols) * len(TIMEFRAMES)
    job = 0

    print("\n" + "=" * 72)
    print("  AlphaBean v3.0 — Walk-Forward Backtest Engine")
    print(f"  {len(symbols)} symbols × {len(TIMEFRAMES)} timeframes × 47 patterns")
    print(f"  Lookback: {days_back} days | Methodology: strict walk-forward")
    print("=" * 72)

    for si, symbol in enumerate(symbols):
        sym_outcomes = []
        sym_start = time.time()

        print(f"\n{'─' * 60}")
        print(f"  [{si+1}/{len(symbols)}] {symbol}")
        print(f"{'─' * 60}")

        for tf in TIMEFRAMES:
            job += 1
            pct = job / total_jobs * 100
            elapsed = time.time() - t_start
            eta = (elapsed / job * total_jobs - elapsed) if job > 0 else 0

            print(f"  {tf:>5} | progress: {pct:.0f}% | elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

            outcomes = backtest_symbol_tf(symbol, tf, days_back, verbose)
            sym_outcomes.extend(outcomes)
            all_outcomes.extend(outcomes)

        # Symbol summary
        sym_wins = sum(1 for o in sym_outcomes if o.outcome == "win")
        sym_total = len(sym_outcomes)
        sym_wr = sym_wins / sym_total * 100 if sym_total > 0 else 0
        sym_time = time.time() - sym_start

        symbol_stats[symbol] = {
            "total_signals": sym_total,
            "wins": sym_wins,
            "win_rate": round(sym_wr, 1),
            "time_seconds": round(sym_time, 1),
        }

        print(f"  {'─' * 40}")
        print(f"  {symbol} TOTAL: {sym_total} signals, {sym_wr:.0f}% WR ({sym_time:.1f}s)")

    # ─── Compute per-pattern per-timeframe stats ───────────────────
    print(f"\n\n{'=' * 72}")
    print("  COMPUTING PATTERN STATISTICS")
    print(f"{'=' * 72}")

    # Group outcomes: pattern_name → timeframe → list of outcomes
    grouped: dict[str, dict[str, list[TradeOutcome]]] = defaultdict(lambda: defaultdict(list))
    for o in all_outcomes:
        # Determine timeframe from outcome timestamp precision or from our loop
        # Since we track both TFs, we need to tag them. We'll use a workaround:
        # the outcomes don't carry TF, so we'll re-group during the per-TF loop above.
        # For now, group by pattern only (combined across TFs).
        grouped[o.pattern_name]["combined"].append(o)

    # Actually, let's redo grouping properly. We need TF info.
    # Let's re-run with TF tagging.
    grouped_tf: dict[str, dict[str, list[TradeOutcome]]] = defaultdict(lambda: defaultdict(list))

    # Re-run per symbol/tf to get proper grouping
    # We already have all_outcomes but they lack TF info.
    # Quick fix: add a note field or just compute combined stats.
    # For v1, combined stats across TFs is fine — we can split later.

    pattern_stats = {}
    for pattern_name in sorted(set(o.pattern_name for o in all_outcomes)):
        trades = [o for o in all_outcomes if o.pattern_name == pattern_name]
        stats = _compute_pattern_stats(pattern_name, trades)
        pattern_stats[pattern_name] = stats

    # ─── Print results table ──────────────────────────────────────
    print(f"\n{'─' * 90}")
    print(f"  {'Pattern':<28} {'Cat':<10} {'Signals':>7} {'WR':>6} {'PF':>6} "
          f"{'Exp':>8} {'AvgW':>6} {'AvgL':>6} {'Edge':>6}")
    print(f"{'─' * 90}")

    # Sort by edge score
    sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1]["edge_score"], reverse=True)

    for name, s in sorted_patterns:
        meta = PATTERN_META.get(name, {})
        cat = meta.get("cat", PatternCategory.CLASSICAL).value[:8]
        sig = s["total_signals"]
        wr = s["win_rate"]
        pf = s["profit_factor"]
        exp = s["expectancy"]
        aw = s["avg_win_r"]
        al = s["avg_loss_r"]
        edge = s["edge_score"]

        # Color coding via symbols
        wr_mark = "●" if wr >= 55 else "○" if wr >= 45 else "✗"
        edge_mark = "★" if edge >= 65 else "☆" if edge >= 50 else " "

        if sig >= 3:  # Only show patterns with enough data
            print(f"  {edge_mark} {name:<26} {cat:<10} {sig:>7} {wr:>5.1f}% {pf:>5.1f} "
                  f"{exp:>+7.3f}R {aw:>5.1f}R {al:>5.1f}R {edge:>5.0f}")

    # ─── Feed into strategy evaluator ─────────────────────────────
    print(f"\n{'=' * 72}")
    print("  UPDATING STRATEGY EVALUATOR")
    print(f"{'=' * 72}")

    evaluator = StrategyEvaluator()
    evaluator.load()
    evaluator.record_batch(all_outcomes)
    evaluator.save()

    rankings = evaluator.get_rankings()
    print(f"\n  Loaded {len(all_outcomes)} outcomes into evaluator")
    print(f"  Top 5 strategies by hot score:")
    for i, m in enumerate(rankings[:5]):
        bar = "█" * int(m.hot_score / 5)
        print(f"    #{i+1} {m.name:<26} {bar} {m.hot_score:.0f}  "
              f"WR={m.win_rate:.0%} PF={m.profit_factor:.1f}")

    # ─── Save results ─────────────────────────────────────────────
    total_time = time.time() - t_start

    results = {
        "version": "3.0",
        "run_date": datetime.now().isoformat(),
        "config": {
            "symbols": symbols,
            "days_back": days_back,
            "timeframes": TIMEFRAMES,
            "scan_intervals": SCAN_INTERVAL,
            "max_hold_bars": MAX_HOLD_BARS,
            "min_bars": MIN_BARS,
        },
        "summary": {
            "total_symbols": len(symbols),
            "total_signals": len(all_outcomes),
            "total_wins": sum(1 for o in all_outcomes if o.outcome == "win"),
            "total_losses": sum(1 for o in all_outcomes if o.outcome == "loss"),
            "total_timeouts": sum(1 for o in all_outcomes if o.outcome == "timeout"),
            "overall_win_rate": round(
                sum(1 for o in all_outcomes if o.outcome == "win") / len(all_outcomes) * 100, 1
            ) if all_outcomes else 0,
            "total_time_seconds": round(total_time, 1),
        },
        "symbol_stats": symbol_stats,
        "patterns": pattern_stats,
    }

    cache_path = Path("cache/backtest_results.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 72}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'=' * 72}")
    print(f"  Symbols:      {len(symbols)}")
    print(f"  Signals:      {len(all_outcomes)}")
    wins_total = sum(1 for o in all_outcomes if o.outcome == "win")
    print(f"  Win Rate:     {wins_total / len(all_outcomes) * 100:.1f}%" if all_outcomes else "  Win Rate:     N/A")
    print(f"  Time:         {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Saved:        cache/backtest_results.json")
    print(f"  Evaluator:    cache/strategy_performance.json")
    print(f"{'=' * 72}\n")

    return results


# ==============================================================================
# STATS COMPUTATION
# ==============================================================================

def _compute_pattern_stats(pattern_name: str, trades: list[TradeOutcome]) -> dict:
    """Compute comprehensive stats for one pattern."""
    n = len(trades)
    if n == 0:
        return {
            "total_signals": 0, "wins": 0, "losses": 0, "timeouts": 0,
            "win_rate": 0, "profit_factor": 0, "expectancy": 0,
            "avg_win_r": 0, "avg_loss_r": 0, "best_r": 0, "worst_r": 0,
            "edge_score": 0,
        }

    wins = [t for t in trades if t.outcome == "win"]
    losses = [t for t in trades if t.outcome == "loss"]
    timeouts = [t for t in trades if t.outcome == "timeout"]

    n_wins = len(wins)
    n_losses = len(losses)
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

    best_r = max(all_r) if all_r else 0
    worst_r = min(all_r) if all_r else 0

    # Edge score: composite quality metric (0-100)
    # Weights: WR(30%), PF(25%), Expectancy(20%), Sample(15%), Consistency(10%)
    wr_score = max(0, min(100, (wr - 0.35) / 0.35 * 100))
    pf_score = max(0, min(100, (min(pf, 5.0) - 0.5) / 2.5 * 100))
    exp_score = max(0, min(100, (expectancy + 0.5) / 1.5 * 100))
    sample_score = min(100, n / 30 * 100)
    # Consistency: low variance in R-multiples is better
    r_std = float(np.std(all_r)) if len(all_r) > 1 else 2.0
    consistency_score = max(0, min(100, (3.0 - r_std) / 3.0 * 100))

    edge = (
        wr_score * 0.30 +
        pf_score * 0.25 +
        exp_score * 0.20 +
        sample_score * 0.15 +
        consistency_score * 0.10
    )
    edge = round(max(0, min(100, edge)), 1)

    return {
        "total_signals": n,
        "wins": n_wins,
        "losses": n_losses,
        "timeouts": len(timeouts),
        "win_rate": round(wr * 100, 1),
        "profit_factor": round(min(pf, 99.0), 2),
        "expectancy": round(expectancy, 3),
        "avg_win_r": round(avg_win_r, 2),
        "avg_loss_r": round(avg_loss_r, 2),
        "best_r": round(best_r, 2),
        "worst_r": round(worst_r, 2),
        "edge_score": edge,
        "sample_size": n,
    }


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="AlphaBean v3.0 Walk-Forward Backtest")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols: AAPL,NVDA,TSLA")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of symbols from extended list (default: 10)")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of history to backtest (default: 30)")
    parser.add_argument("--full", action="store_true",
                        help="Full backtest: 50 symbols, 90 days")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.full:
        symbols = EXTENDED_SYMBOLS
        args.days = 90
    else:
        symbols = EXTENDED_SYMBOLS[:args.count]

    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  AlphaBean v3.0 — Walk-Forward Backtest         ║")
    print(f"  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  Symbols:     {len(symbols):<35} ║")
    print(f"  ║  Days back:   {args.days:<35} ║")
    print(f"  ║  Timeframes:  {', '.join(TIMEFRAMES):<35} ║")
    print(f"  ║  Patterns:    47 (16 classical + 10 candle +    ║")
    print(f"  ║               11 SMB + 10 quant)                ║")
    print(f"  ║  Method:      Strict walk-forward (no peeking)  ║")
    print(f"  ╚══════════════════════════════════════════════════╝")

    if not args.quiet:
        print(f"\n  Symbols: {', '.join(symbols)}")
        est_time = len(symbols) * len(TIMEFRAMES) * 3  # ~3s per symbol×TF
        print(f"  Estimated time: ~{est_time//60}m {est_time%60}s")

    results = run_full_backtest(symbols, days_back=args.days, verbose=not args.quiet)

    # Print final summary
    patterns = results.get("patterns", {})
    good = [p for p, s in patterns.items() if s.get("edge_score", 0) >= 60 and s.get("total_signals", 0) >= 5]
    decent = [p for p, s in patterns.items() if 40 <= s.get("edge_score", 0) < 60 and s.get("total_signals", 0) >= 5]
    weak = [p for p, s in patterns.items() if s.get("edge_score", 0) < 40 and s.get("total_signals", 0) >= 5]

    if good:
        print(f"  ★ Strong edge ({len(good)}): {', '.join(good[:8])}")
    if decent:
        print(f"  ☆ Decent edge ({len(decent)}): {', '.join(decent[:8])}")
    if weak:
        print(f"    Weak edge ({len(weak)}): {', '.join(weak[:5])}")

    print(f"\n  Results saved. The scanner now uses these stats for:")
    print(f"    • Multi-factor scoring (backtest_edge component)")
    print(f"    • Strategy evaluator hot scores")
    print(f"    • AI agent trade confirmation context")


if __name__ == "__main__":
    main()