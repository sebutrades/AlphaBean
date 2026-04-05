"""
backtest/engine.py — The core backtesting engine.

For every symbol's cached bar data:
  1. Runs all pattern detectors (same classifier used by the live scanner)
  2. For each signal, simulates entry at the NEXT bar's open (realistic fill)
  3. Checks if price hit T1/T2 or STOP, tracking scaled exits
  4. Tracks MFE and MAE in R-multiples
  5. Aggregates results per pattern per timeframe

Results are cached quarterly in cache/backtest_results_YYYY-Q#.json.

Usage:
    from backend.backtest.engine import run_full_backtest, get_backtest_results

    results = run_full_backtest()   # heavy — runs once per quarter
    results = get_backtest_results()  # read cached results
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from backend.backtest.data_fetcher import bars_from_cache, load_cached_bars
from backend.backtest.universe import get_universe
from backend.data.schemas import BarSeries, Bar
from backend.patterns.registry import TradeSetup, Bias
from backend.patterns.classifier import classify_all

CACHE_DIR = Path("cache")
TIMEFRAMES = ["15min", "1h", "1d"]

# How many bars forward to check for target/stop hit
FORWARD_BARS = {
    "15min": 40,   # 40 × 15min = 10 hours (intraday + next day)
    "1h": 30,      # 30 hours = ~4 trading days
    "1d": 40,      # 40 trading days = ~2 months
}

# Realistic execution costs applied to entry price.
# Entry slippage: long trades pay MORE than next-open; short trades receive LESS.
# Commission: round-trip cost expressed as fraction of entry price.
# These are conservative estimates for liquid US equities at retail/prop rates.
SLIPPAGE = {
    "15min": 0.0010,   # 0.10% — wider spread on intraday signals
    "1h":    0.0008,   # 0.08%
    "1d":    0.0005,   # 0.05% — tighter spread on daily EOD entries
}
COMMISSION_RT = 0.0002   # 0.02% round-trip ($0.20 per $1000 notional)


def _current_quarter() -> str:
    now = datetime.now()
    q = (now.month - 1) // 3 + 1
    return f"{now.year}-Q{q}"


def _results_path() -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"backtest_results_{_current_quarter()}.json"


def get_backtest_results() -> Optional[dict]:
    """Load cached backtest results for the current quarter."""
    path = _results_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, KeyError):
        return None


def run_full_backtest(
    symbols: list[str] = None,
    timeframes: list[str] = None,
    force: bool = False,
) -> dict:
    """
    Run the full backtest across all symbols × timeframes.
    
    This is the heavy operation — runs once per quarter.
    Results are cached to cache/backtest_results_YYYY-Q#.json.
    """
    # Check cache first
    if not force:
        existing = get_backtest_results()
        if existing:
            print(f"[Backtest] Using cached results from {existing.get('generated', '?')}")
            return existing

    if symbols is None:
        symbols = get_universe()
    if timeframes is None:
        timeframes = TIMEFRAMES

    # Structure: results[pattern_name][timeframe] = list of trade outcomes
    raw_results = defaultdict(lambda: defaultdict(list))
    # Dedup: only count the first signal per (symbol, pattern, window) to avoid
    # stacking identical signals on consecutive bars.
    seen: set[tuple] = set()

    total = len(symbols) * len(timeframes)
    done = 0
    signals_found = 0

    print(f"\n{'=' * 60}")
    print(f"BACKTEST: {len(symbols)} symbols × {len(timeframes)} timeframes")
    print(f"{'=' * 60}\n")

    for symbol in symbols:
        for tf in timeframes:
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{total}] Processing... ({signals_found} signals so far)")

            # Load cached bar data
            bar_series = bars_from_cache(symbol, tf)
            if bar_series is None or len(bar_series.bars) < 60:
                continue

            bars = bar_series.bars
            forward_limit = FORWARD_BARS.get(tf, 30)

            # Slide a window bar-by-bar — step=1 ensures no signals are missed.
            # Prior code used step=forward_limit//4 which silently skipped ~75%
            # of all potential signal bars.
            for window_end in range(60, len(bars) - forward_limit):
                window_bars = bars[:window_end + 1]
                window_series = BarSeries(
                    symbol=symbol, timeframe=tf, bars=window_bars
                )

                try:
                    setups = classify_all(window_series)
                except Exception:
                    continue

                for setup in setups:
                    try:
                        # Dedup: if the same pattern fired on this symbol in the
                        # previous N bars, skip — it's the same setup persisting.
                        dedup_key = (symbol, setup.pattern_name, tf, window_end // 5)
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)

                        outcome = _evaluate_trade(
                            setup, bars, window_end, forward_limit, tf
                        )
                        if outcome is not None:
                            raw_results[setup.pattern_name][tf].append(outcome)
                            signals_found += 1
                    except Exception:
                        continue

    print(f"\n[Backtest] Total signals found: {signals_found}")
    print("[Backtest] Aggregating results...")

    # Aggregate into summary stats
    results = _aggregate_results(raw_results)

    # Add metadata
    results["generated"] = datetime.now().isoformat()
    results["quarter"] = _current_quarter()
    results["universe_size"] = len(symbols)
    results["total_signals"] = signals_found

    # Cache
    path = _results_path()
    path.write_text(json.dumps(results, indent=2))
    print(f"[Backtest] Saved to {path}")

    return results


def _evaluate_trade(
    setup: TradeSetup,
    all_bars: list,
    signal_idx: int,
    forward_limit: int,
    timeframe: str = "1d",
) -> Optional[dict]:
    """
    Simulate a trade triggered by setup at signal_idx.

    Entry: next bar's open + slippage (realistic fill cost).
    Exit model: scaled exits matching live trade management
      · 50% at T1, stop moves to breakeven
      · Remaining 50% at T2 (or trailed to close at forward limit)
      · Full loss if stopped before T1 (-1R)
    Costs: slippage on entry + round-trip commission deducted from realized_r.
    MFE/MAE tracked from entry in R-multiples.
    """
    if signal_idx + 1 >= len(all_bars):
        return None

    # Realistic fill: next bar's open with execution slippage
    slip = SLIPPAGE.get(timeframe, 0.0005)
    raw_open = all_bars[signal_idx + 1].open
    is_long = setup.bias == Bias.LONG
    # Longs pay above open; shorts receive below open
    entry = raw_open * (1 + slip) if is_long else raw_open * (1 - slip)
    stop = setup.stop_loss
    t1 = getattr(setup, 'target_1', 0) or setup.target_price
    t2 = getattr(setup, 'target_2', 0) or t1

    # Sanity checks — skip degenerate setups
    if entry <= 0 or stop <= 0 or t1 <= 0:
        return None
    if is_long and (entry <= stop or t1 <= entry):
        return None
    if not is_long and (entry >= stop or t1 >= entry):
        return None

    risk = abs(entry - stop)
    if risk <= 0:
        return None

    end_idx = min(signal_idx + 1 + forward_limit, len(all_bars))
    if end_idx <= signal_idx + 2:
        return None

    max_favorable = 0.0
    max_adverse = 0.0
    t1_hit = False
    t2_hit = False
    hit_stop = False
    bars_to_resolution = 0
    current_stop = stop  # will move to breakeven once T1 is hit

    # Check entry bar AND all forward bars (entry is at open of signal_idx+1)
    for i in range(signal_idx + 1, end_idx):
        bar = all_bars[i]
        bars_elapsed = i - signal_idx

        if is_long:
            max_favorable = max(max_favorable, bar.high - entry)
            max_adverse = max(max_adverse, entry - bar.low)

            # T1 check (use high — intrabar can reach T1)
            if not t1_hit and bar.high >= t1:
                t1_hit = True
                current_stop = entry  # breakeven after T1

            # T2 check
            if t1_hit and t2 > t1 and bar.high >= t2:
                t2_hit = True
                bars_to_resolution = bars_elapsed
                break

            # Stop check (use low for worst-case intrabar breach)
            if bar.low <= current_stop:
                hit_stop = True
                bars_to_resolution = bars_elapsed
                break
        else:
            max_favorable = max(max_favorable, entry - bar.low)
            max_adverse = max(max_adverse, bar.high - entry)

            if not t1_hit and bar.low <= t1:
                t1_hit = True
                current_stop = entry  # breakeven

            if t1_hit and t2 < t1 and bar.low <= t2:
                t2_hit = True
                bars_to_resolution = bars_elapsed
                break

            if bar.high >= current_stop:
                hit_stop = True
                bars_to_resolution = bars_elapsed
                break

    # Compute realized R using the same split as live trade management (50/50)
    t1_r = abs(t1 - entry) / risk
    t2_r = abs(t2 - entry) / risk

    if t2_hit:
        # Best case: both halves won
        realized_r = 0.5 * t1_r + 0.5 * t2_r
        win = True
    elif t1_hit and hit_stop:
        # T1 hit → stop moved to breakeven → second half stopped flat
        realized_r = 0.5 * t1_r  # first half wins, second exits at 0R
        win = True
    elif t1_hit:
        # T1 hit, T2 not reached, time expired — close remainder at final price
        final = all_bars[end_idx - 1].close
        trail_r = (final - entry) / risk if is_long else (entry - final) / risk
        realized_r = 0.5 * t1_r + 0.5 * trail_r
        bars_to_resolution = end_idx - signal_idx
        win = realized_r > 0
    elif hit_stop:
        realized_r = -1.0
        win = False
    else:
        # Time expired — close full position at final bar
        final = all_bars[end_idx - 1].close
        realized_r = (final - entry) / risk if is_long else (entry - final) / risk
        bars_to_resolution = end_idx - signal_idx
        win = realized_r > 0

    mfe_r = max_favorable / risk
    mae_r = max_adverse / risk

    # Deduct round-trip commission expressed in R-multiples.
    # commission_R = cost / risk_per_share. This is small (typically 0.01–0.05R)
    # but compounds materially across hundreds of trades.
    commission_r = (entry * COMMISSION_RT) / risk if risk > 0 else 0
    realized_r -= commission_r
    win = realized_r > 0

    return {
        "win": win,
        "realized_r": round(realized_r, 3),
        "mfe_r": round(mfe_r, 2),
        "mae_r": round(mae_r, 2),
        "bars_to_resolution": bars_to_resolution,
        "t1_hit": t1_hit,
        "t2_hit": t2_hit,
        "symbol": setup.symbol,
    }


def _aggregate_results(raw_results: dict) -> dict:
    """
    Aggregate individual trade outcomes into summary statistics
    per pattern per timeframe.
    """
    aggregated = {"patterns": {}}

    for pattern_name, tf_data in raw_results.items():
        pattern_stats = {}

        for tf, trades in tf_data.items():
            if len(trades) < 10:  # Need minimum sample size
                continue

            wins = [t for t in trades if t["win"]]
            losses = [t for t in trades if not t["win"]]

            total = len(trades)
            win_count = len(wins)
            loss_count = len(losses)
            win_rate = win_count / total if total > 0 else 0

            # Average R-multiples
            avg_win_r = float(np.mean([t["realized_r"] for t in wins])) if wins else 0
            avg_loss_r = float(np.mean([abs(t["realized_r"]) for t in losses])) if losses else 0

            # Profit factor = gross wins / gross losses
            gross_wins = sum(t["realized_r"] for t in wins) if wins else 0
            gross_losses = sum(abs(t["realized_r"]) for t in losses) if losses else 0
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else (
                float('inf') if gross_wins > 0 else 0
            )

            # Expectancy (average R per trade)
            all_r = [t["realized_r"] for t in trades]
            expectancy = float(np.mean(all_r)) if all_r else 0

            # MFE / MAE
            avg_mfe = float(np.mean([t["mfe_r"] for t in trades]))
            avg_mae = float(np.mean([t["mae_r"] for t in trades]))

            # Time to resolution
            avg_bars = float(np.mean([t["bars_to_resolution"] for t in trades]))

            # Best and worst symbols
            symbol_wins = defaultdict(int)
            symbol_total = defaultdict(int)
            for t in trades:
                symbol_total[t["symbol"]] += 1
                if t["win"]:
                    symbol_wins[t["symbol"]] += 1

            # Symbols with at least 3 signals
            symbol_wr = {
                sym: symbol_wins[sym] / cnt
                for sym, cnt in symbol_total.items() if cnt >= 3
            }
            best_symbols = sorted(symbol_wr, key=symbol_wr.get, reverse=True)[:5]
            worst_symbols = sorted(symbol_wr, key=symbol_wr.get)[:5]

            t1_hits = sum(1 for t in trades if t.get("t1_hit"))
            t2_hits = sum(1 for t in trades if t.get("t2_hit"))

            pattern_stats[tf] = {
                "total_signals": total,
                "wins": win_count,
                "losses": loss_count,
                "win_rate": round(win_rate * 100, 1),
                "avg_win_r": round(avg_win_r, 2),
                "avg_loss_r": round(avg_loss_r, 2),
                "profit_factor": round(min(profit_factor, 99.9), 2),
                "expectancy": round(expectancy, 3),
                "avg_mfe_r": round(avg_mfe, 2),
                "avg_mae_r": round(avg_mae, 2),
                "avg_bars_to_resolution": round(avg_bars, 1),
                "t1_hit_rate": round(t1_hits / total * 100, 1) if total > 0 else 0,
                "t2_hit_rate": round(t2_hits / total * 100, 1) if total > 0 else 0,
                "partial_wins": 0,
                "best_symbols": best_symbols,
                "worst_symbols": worst_symbols,
            }

        if pattern_stats:
            aggregated["patterns"][pattern_name] = pattern_stats

    return aggregated


def get_pattern_stats(pattern_name: str) -> Optional[dict]:
    """Get backtest stats for a specific pattern across all timeframes."""
    results = get_backtest_results()
    if results is None:
        return None
    return results.get("patterns", {}).get(pattern_name)


def get_pattern_score(pattern_name: str, timeframe: str) -> float:
    """
    Get a 0-100 backtest quality score for a pattern on a timeframe.
    
    Score = weighted combination of:
      - Win rate (40% weight)
      - Profit factor (30% weight) 
      - Expectancy (20% weight)
      - Sample size confidence (10% weight)
    """
    stats = get_pattern_stats(pattern_name)
    if stats is None or timeframe not in stats:
        return 50.0  # Default score when no backtest data

    s = stats[timeframe]
    
    # Win rate score (0-100): 40% at 40 WR, 100% at 70+ WR (stored as 0-100 percentage)
    wr_score = min(100, max(0, (s["win_rate"] - 35.0) / 35.0 * 100))

    # Profit factor score (0-100): 0 at 0.5 PF, 100 at 3.0+ PF
    pf = min(s["profit_factor"], 5.0)
    pf_score = min(100, max(0, (pf - 0.5) / 2.5 * 100))

    # Expectancy score (0-100): 0 at -0.5R, 100 at 1.0+R
    exp_score = min(100, max(0, (s["expectancy"] + 0.5) / 1.5 * 100))

    # Sample size confidence (0-100): low confidence < 20 signals
    n = s["total_signals"]
    sample_score = min(100, n / 50 * 100)  # Full score at 50+ signals

    score = (
        wr_score * 0.40 +
        pf_score * 0.30 +
        exp_score * 0.20 +
        sample_score * 0.10
    )

    return round(score, 1)