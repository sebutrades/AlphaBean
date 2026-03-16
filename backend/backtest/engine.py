"""
backtest/engine.py — The core backtesting engine.

For every symbol's cached bar data:
  1. Runs all 21 pattern detectors
  2. For each signal, checks if price hit TARGET or STOP first
  3. Tracks MFE (max favorable excursion) and MAE (max adverse excursion)
  4. Aggregates results per pattern per timeframe

Results are cached quarterly in cache/backtest_results_YYYY-Q#.json.

Usage:
    from backend.backtest.engine import run_full_backtest, get_backtest_results
    
    # Run the backtest (takes ~5-15 min depending on cache)
    results = run_full_backtest()
    
    # Get cached results
    results = get_backtest_results()
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from backend.backtest.data_fetcher import bars_from_cache, load_cached_bars
from backend.backtest.universe import get_universe
from backend.patterns.edgefinder_patterns import (
    get_all_detectors, BarSeries, Bar, TradeSetup, Bias
)

CACHE_DIR = Path("cache")
TIMEFRAMES = ["15min", "1h", "1d"]

# How many bars forward to check for target/stop hit
FORWARD_BARS = {
    "15min": 40,   # 40 × 15min = 10 hours (intraday + next day)
    "1h": 30,      # 30 hours = ~4 trading days
    "1d": 40,      # 40 trading days = ~2 months
}


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

    detectors = get_all_detectors()
    
    # Structure: results[pattern_name][timeframe] = list of trade outcomes
    raw_results = defaultdict(lambda: defaultdict(list))

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
            if bar_series is None or len(bar_series.bars) < 30:
                continue

            bars = bar_series.bars
            forward_limit = FORWARD_BARS.get(tf, 30)

            # We need to run detectors on rolling windows, not just the full series.
            # Strategy: slide a window through the bars, run detectors at each position.
            # To be efficient, we check every N bars (not every single bar).
            step = max(1, forward_limit // 4)  # Check every ~25% of forward window

            for window_end in range(60, len(bars) - forward_limit, step):
                # Create a sub-series up to window_end
                window_bars = bars[:window_end + 1]
                window_series = BarSeries(
                    symbol=symbol, timeframe=tf, bars=window_bars
                )

                for detector in detectors:
                    try:
                        setup = detector.detect(window_series)
                        if setup is None:
                            continue

                        # We have a signal! Now check forward bars for outcome.
                        outcome = _evaluate_trade(
                            setup, bars, window_end, forward_limit
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
    all_bars: list[Bar],
    signal_idx: int,
    forward_limit: int,
) -> Optional[dict]:
    """
    Given a trade signal at signal_idx, look forward to see if
    price hit TARGET or STOP first.
    
    Returns a dict with outcome details, or None if data is insufficient.
    """
    entry = setup.entry_price
    stop = setup.stop_loss
    target = setup.target_price
    is_long = setup.bias == Bias.LONG

    if entry <= 0 or stop <= 0 or target <= 0:
        return None

    risk = abs(entry - stop)
    if risk <= 0:
        return None

    end_idx = min(signal_idx + forward_limit, len(all_bars))
    if end_idx <= signal_idx + 1:
        return None

    # Track MFE and MAE
    max_favorable = 0.0
    max_adverse = 0.0
    hit_target = False
    hit_stop = False
    bars_to_resolution = 0

    for i in range(signal_idx + 1, end_idx):
        bar = all_bars[i]
        bars_elapsed = i - signal_idx

        if is_long:
            favorable = bar.high - entry
            adverse = entry - bar.low
            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

            if bar.high >= target:
                hit_target = True
                bars_to_resolution = bars_elapsed
                break
            if bar.low <= stop:
                hit_stop = True
                bars_to_resolution = bars_elapsed
                break
        else:
            favorable = entry - bar.low
            adverse = bar.high - entry
            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

            if bar.low <= target:
                hit_target = True
                bars_to_resolution = bars_elapsed
                break
            if bar.high >= stop:
                hit_stop = True
                bars_to_resolution = bars_elapsed
                break

    # If neither hit, classify based on final P&L
    if not hit_target and not hit_stop:
        final_price = all_bars[end_idx - 1].close
        if is_long:
            pnl = final_price - entry
        else:
            pnl = entry - final_price
        hit_target = pnl > 0
        hit_stop = pnl <= 0
        bars_to_resolution = end_idx - signal_idx

    # Normalize MFE/MAE to R-multiples
    mfe_r = max_favorable / risk if risk > 0 else 0
    mae_r = max_adverse / risk if risk > 0 else 0

    # Realized R-multiple
    if hit_target:
        realized_r = abs(target - entry) / risk
    elif hit_stop:
        realized_r = -1.0  # Lost 1R
    else:
        realized_r = 0.0

    return {
        "win": hit_target,
        "realized_r": round(realized_r, 2),
        "mfe_r": round(mfe_r, 2),
        "mae_r": round(mae_r, 2),
        "bars_to_resolution": bars_to_resolution,
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
            if len(trades) < 3:  # Need minimum sample size
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

            pattern_stats[tf] = {
                "total_signals": total,
                "wins": win_count,
                "losses": loss_count,
                "win_rate": round(win_rate, 3),
                "avg_win_r": round(avg_win_r, 2),
                "avg_loss_r": round(avg_loss_r, 2),
                "profit_factor": round(min(profit_factor, 99.9), 2),
                "expectancy": round(expectancy, 3),
                "avg_mfe_r": round(avg_mfe, 2),
                "avg_mae_r": round(avg_mae, 2),
                "avg_bars_to_resolution": round(avg_bars, 1),
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
    
    # Win rate score (0-100): 40% at 0.4 WR, 100% at 0.7+ WR
    wr_score = min(100, max(0, (s["win_rate"] - 0.35) / 0.35 * 100))

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