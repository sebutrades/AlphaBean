"""
backend/optimization/param_optimizer.py — Strategy Parameter Optimizer v3

Uses Optuna (Bayesian) + 6-core parallel execution to tune all strategies.

PARALLELISM
  - 30 patterns distributed round-robin across N_CORES=6 workers
  - Each worker runs its ~5 patterns sequentially
  - All data is 100% cache-only — no API calls during optimization

TRAIN / VALIDATION SPLIT
  - Symbol-based (NOT time-based): 500 symbols split 250 train / 250 val
  - Even/odd interleaving by volume-rank: avoids correlation between halves
  - Same time window for both sets
  - Proves cross-sectional generalization: params that work on 250 unseen
    stocks in identical market conditions are genuinely robust
  - Better than 90/90 time-split which leaves daily strategies with <10
    signals per symbol in the training window

OBJECTIVE
  - Maximize expectancy (EXP/R) on train symbols
  - After Optuna completes, run one evaluation pass on val symbols
  - Flag overfit when val_exp < 0 or val_exp < train_exp * 0.4

USAGE:
  python -m backend.optimization.param_optimizer --strategy "Juicer Long" --trials 30
  python -m backend.optimization.param_optimizer --trials 50           # All, 6 cores
  python -m backend.optimization.param_optimizer --list                # Show all
  python -m backend.optimization.param_optimizer --cores 4             # Override cores

DEPENDENCIES:
  pip install optuna
"""
import argparse
import json
import random
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

# ==============================================================================
# PARALLELISM CONFIG
# ==============================================================================

N_CORES = 6

# Relative slowness by timeframe — used to sort strategies before submission
# so each core gets a mix of fast (1d) and slow (5min) work rather than all
# slow strategies piling up on some cores while others idle.
TF_WEIGHT = {"5min": 4, "15min": 3, "1h": 2, "1d": 1}

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ==============================================================================
# UNIVERSAL PARAMS — Applied to EVERY strategy
# ==============================================================================

UNIVERSAL_PARAMS = {
    "stop_atr_mult":  {"type": "float", "low": 0.5,  "high": 4.0,  "step": 0.25},
    "t1_atr_mult":    {"type": "float", "low": 0.5,  "high": 4.0,  "step": 0.25},
    "t2_atr_mult":    {"type": "float", "low": 1.0,  "high": 8.0,  "step": 0.5},
    "trail_atr_mult": {"type": "float", "low": 1.0,  "high": 4.0,  "step": 0.5},
    "max_hold_bars":  {"type": "int",   "low": 5,    "high": 200,  "step": 5},
    "cooldown_bars":  {"type": "int",   "low": 1,    "high": 51,   "step": 2},
    "split_t1":       {"type": "float", "low": 0.20, "high": 0.70, "step": 0.10},
    "split_t2":       {"type": "float", "low": 0.10, "high": 0.50, "step": 0.10},
}


# ==============================================================================
# STRATEGY-SPECIFIC PARAMS — Detection logic tuning
# ==============================================================================

STRATEGY_SPECIFIC_PARAMS = {
    "Juicer Long": {
        "adx_threshold":     {"type": "int",   "low": 15, "high": 40, "step": 5},
        "weekly_streak":     {"type": "int",   "low": 2,  "high": 5,  "step": 1},
        "sma_fast":          {"type": "int",   "low": 10, "high": 30, "step": 5},
        "sma_slow":          {"type": "int",   "low": 30, "high": 60, "step": 10},
    },
    "RS Persistence Long": {
        "rs_threshold":      {"type": "float", "low": 1.02, "high": 1.15, "step": 0.01},
    },
    "TS Momentum Long": {
        "lookback_days":     {"type": "int",   "low": 60,  "high": 252, "step": 21},
        "skip_days":         {"type": "int",   "low": 5,   "high": 42,  "step": 7},
        "mom_threshold":     {"type": "float", "low": 0.02, "high": 0.10, "step": 0.01},
    },
    "TS Momentum Short": {
        "lookback_days":     {"type": "int",   "low": 60,  "high": 252, "step": 21},
        "skip_days":         {"type": "int",   "low": 5,   "high": 42,  "step": 7},
        "mom_threshold":     {"type": "float", "low": 0.02, "high": 0.10, "step": 0.01},
    },
    "Momentum Breakout": {
        "lookback_period":   {"type": "int",   "low": 10, "high": 30, "step": 5},
        "vol_mult":          {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
    },
    "BAB Long": {
        "max_vol":           {"type": "float", "low": 0.15, "high": 0.40, "step": 0.05},
    },
    "BB Squeeze Long": {
        "bb_period":         {"type": "int",   "low": 10, "high": 30, "step": 5},
        "bb_std":            {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25},
        "bw_percentile":     {"type": "float", "low": 0.03, "high": 0.25, "step": 0.02},
    },
    "BB Squeeze Short": {
        "bb_period":         {"type": "int",   "low": 10, "high": 30, "step": 5},
        "bb_std":            {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25},
        "bw_percentile":     {"type": "float", "low": 0.03, "high": 0.25, "step": 0.02},
    },
    "Accumulation Long": {
        "acc_days_min":      {"type": "int",   "low": 2,  "high": 5,  "step": 1},
        "lookback_window":   {"type": "int",   "low": 7,  "high": 15, "step": 1},
    },
    "Turtle Breakout Long": {
        "channel_period":    {"type": "int",   "low": 10, "high": 30, "step": 5},
        "squeeze_threshold": {"type": "float", "low": 0.65, "high": 0.95, "step": 0.05},
        "vol_mult":          {"type": "float", "low": 1.0, "high": 2.0, "step": 0.25},
    },
    "Mean Reversion": {
        "z_threshold":       {"type": "float", "low": 1.5, "high": 3.5, "step": 0.25},
        "lookback":          {"type": "int",   "low": 30,  "high": 80,  "step": 10},
    },
    "Tidal Wave": {
        "min_touches":       {"type": "int",   "low": 2,  "high": 5,  "step": 1},
        "min_span":          {"type": "int",   "low": 8,  "high": 25, "step": 3},
        "vol_mult":          {"type": "float", "low": 1.0, "high": 2.0, "step": 0.25},
    },
    "VWAP Reversion": {
        "atr_threshold":     {"type": "float", "low": 1.5, "high": 4.0, "step": 0.5},
    },
    "Streak Reversal Long": {
        "min_streak":        {"type": "int",   "low": 3,  "high": 8,  "step": 1},
    },
    "Streak Reversal Short": {
        "min_streak":        {"type": "int",   "low": 3,  "high": 8,  "step": 1},
    },
    "ATR Expansion Long": {
        "expansion_threshold": {"type": "float", "low": 1.3, "high": 2.5, "step": 0.2},
        "close_position":      {"type": "float", "low": 0.55, "high": 0.80, "step": 0.05},
        "vol_mult":            {"type": "float", "low": 1.0, "high": 2.0, "step": 0.25},
    },
    "ATR Expansion Short": {
        "expansion_threshold": {"type": "float", "low": 1.3, "high": 2.5, "step": 0.2},
        "close_position":      {"type": "float", "low": 0.20, "high": 0.45, "step": 0.05},
        "vol_mult":            {"type": "float", "low": 1.0, "high": 2.0, "step": 0.25},
    },
    "Multi-TF Trend Long": {
        "ema_fast":          {"type": "int",   "low": 5,   "high": 25,  "step": 5},
        "ema_mid":           {"type": "int",   "low": 20,  "high": 60,  "step": 10},
        "ema_slow":          {"type": "int",   "low": 50,  "high": 120, "step": 10},
        "ema_anchor":        {"type": "int",   "low": 100, "high": 250, "step": 50},
    },
    "Multi-TF Trend Short": {
        "ema_fast":          {"type": "int",   "low": 5,   "high": 25,  "step": 5},
        "ema_mid":           {"type": "int",   "low": 20,  "high": 60,  "step": 10},
        "ema_slow":          {"type": "int",   "low": 50,  "high": 120, "step": 10},
        "ema_anchor":        {"type": "int",   "low": 100, "high": 250, "step": 50},
    },
    "Gap Reversal Long": {
        "min_gap_pct":       {"type": "float", "low": 0.003, "high": 0.015, "step": 0.002},
        "max_gap_pct":       {"type": "float", "low": 0.02, "high": 0.05, "step": 0.01},
        "retrace_pct":       {"type": "float", "low": 0.15, "high": 0.50, "step": 0.05},
    },
    "Gap Reversal Short": {
        "min_gap_pct":       {"type": "float", "low": 0.003, "high": 0.015, "step": 0.002},
        "max_gap_pct":       {"type": "float", "low": 0.02, "high": 0.05, "step": 0.01},
        "retrace_pct":       {"type": "float", "low": 0.15, "high": 0.50, "step": 0.05},
    },
    "Opening Drive Long": {
        "min_drive_atr":     {"type": "float", "low": 0.3, "high": 1.5, "step": 0.1},
        "vol_mult":          {"type": "float", "low": 1.0, "high": 2.0, "step": 0.25},
    },
    "Opening Drive Short": {
        "min_drive_atr":     {"type": "float", "low": 0.3, "high": 1.5, "step": 0.1},
        "vol_mult":          {"type": "float", "low": 1.0, "high": 2.0, "step": 0.25},
    },
    "Volume Climax Long": {
        "vol_threshold":     {"type": "float", "low": 2.0, "high": 6.0, "step": 0.5},
    },
    "Volume Climax Short": {
        "vol_threshold":     {"type": "float", "low": 2.0, "high": 6.0, "step": 0.5},
    },
    "Midday Reversal Long": {
        "min_morning_atr":   {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
        "retrace_pct":       {"type": "float", "low": 0.15, "high": 0.50, "step": 0.05},
    },
    "Midday Reversal Short": {
        "min_morning_atr":   {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
        "retrace_pct":       {"type": "float", "low": 0.15, "high": 0.50, "step": 0.05},
    },
    "RSI Divergence Long": {
        "rsi_period":        {"type": "int",   "low": 8,  "high": 20, "step": 2},
        "rsi_threshold":     {"type": "float", "low": 25, "high": 45, "step": 5},
    },
    "RSI Divergence Short": {
        "rsi_period":        {"type": "int",   "low": 8,  "high": 20, "step": 2},
        "rsi_threshold":     {"type": "float", "low": 55, "high": 75, "step": 5},
    },
    "Power Hour Long": {
        "vwap_pct_threshold": {"type": "float", "low": 0.55, "high": 0.85, "step": 0.05},
        "vol_mult":           {"type": "float", "low": 1.0, "high": 1.8, "step": 0.1},
    },
    "Power Hour Short": {
        "vwap_pct_threshold": {"type": "float", "low": 0.15, "high": 0.45, "step": 0.05},
        "vol_mult":           {"type": "float", "low": 1.0, "high": 1.8, "step": 0.1},
    },
    "VWAP Trend Long": {
        "pct_above_threshold": {"type": "float", "low": 0.60, "high": 0.90, "step": 0.05},
        "min_vwap_tests":      {"type": "int",   "low": 1,  "high": 4,  "step": 1},
        "proximity_atr":       {"type": "float", "low": 0.2, "high": 0.8, "step": 0.1},
    },
    "VWAP Trend Short": {
        "pct_above_threshold": {"type": "float", "low": 0.10, "high": 0.40, "step": 0.05},
        "min_vwap_tests":      {"type": "int",   "low": 1,  "high": 4,  "step": 1},
        "proximity_atr":       {"type": "float", "low": 0.2, "high": 0.8, "step": 0.1},
    },
    "Keltner Breakout Long": {
        "ema_period":        {"type": "int",   "low": 10, "high": 30, "step": 5},
        "atr_mult":          {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25},
        "squeeze_threshold": {"type": "float", "low": 0.65, "high": 0.95, "step": 0.05},
        "vol_mult":          {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
    },
    "Keltner Breakout Short": {
        "ema_period":        {"type": "int",   "low": 10, "high": 30, "step": 5},
        "atr_mult":          {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25},
        "squeeze_threshold": {"type": "float", "low": 0.65, "high": 0.95, "step": 0.05},
        "vol_mult":          {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
    },
    "MACD Turn Long": {
        "fast_period":       {"type": "int",   "low": 6,  "high": 16, "step": 2},
        "slow_period":       {"type": "int",   "low": 18, "high": 34, "step": 2},
    },
    "MACD Turn Short": {
        "fast_period":       {"type": "int",   "low": 6,  "high": 16, "step": 2},
        "slow_period":       {"type": "int",   "low": 18, "high": 34, "step": 2},
    },
    "VP Divergence Long": {
        "vol_decline_pct":   {"type": "float", "low": 0.40, "high": 0.80, "step": 0.05},
        "price_trend_min":   {"type": "float", "low": 0.005, "high": 0.04, "step": 0.005},
    },
    "VP Divergence Short": {
        "vol_decline_pct":   {"type": "float", "low": 0.40, "high": 0.80, "step": 0.05},
        "price_trend_min":   {"type": "float", "low": 0.005, "high": 0.04, "step": 0.005},
    },
    "Donchian Breakout": {
        "channel_period":    {"type": "int",   "low": 10, "high": 30, "step": 5},
        "squeeze_threshold": {"type": "float", "low": 0.65, "high": 0.95, "step": 0.05},
        "vol_mult":          {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
    },
    "Volume Breakout": {
        "vol_threshold":     {"type": "float", "low": 1.5, "high": 5.0, "step": 0.5},
    },
    "ST Reversal Long": {
        "z_threshold":       {"type": "float", "low": 1.0, "high": 3.5, "step": 0.25},
    },
    "ST Reversal Short": {
        "z_threshold":       {"type": "float", "low": 1.0, "high": 3.5, "step": 0.25},
    },
    "Distribution Short": {
        "dist_days_min":     {"type": "int",   "low": 2,  "high": 6,  "step": 1},
        "lookback_window":   {"type": "int",   "low": 7,  "high": 21, "step": 2},
    },
    "Turtle Breakout Short": {
        "channel_period":    {"type": "int",   "low": 10, "high": 30, "step": 5},
        "squeeze_threshold": {"type": "float", "low": 0.65, "high": 0.95, "step": 0.05},
    },
    "52W High Momentum": {
        "pct_from_high":     {"type": "float", "low": 0.02, "high": 0.15, "step": 0.01},
    },
}


# ==============================================================================
# Get full param space for a strategy (universal + specific)
# ==============================================================================

def get_all_params(strategy_name: str) -> dict:
    """Combine universal + strategy-specific params."""
    params = dict(UNIVERSAL_PARAMS)
    specific = STRATEGY_SPECIFIC_PARAMS.get(strategy_name, {})
    params.update(specific)
    return params


# ==============================================================================
# OPTIMIZER ENGINE
# ==============================================================================

SYMBOLS_CACHE = Path("cache/top_symbols.json")
BAR_DATA_DIR  = Path("cache/bar_data")
RESULTS_DIR   = Path("cache/optimization")
GLOBAL_PARAMS = Path("cache/optimized_params.json")
BAR_MINUTES   = {"5min": 5, "15min": 15, "1h": 60, "1d": 390}


def load_symbols(count: int = 500, timeframe: str = "1d", min_bars: int = 0) -> list[str]:
    """
    Load symbols that have sufficient cached bar data.

    Filters out symbols with partial history (late IPOs, delistings, etc.) by
    checking actual bar count in cache. Only symbols meeting min_bars are returned.

    Defaults:
      timeframe="1d", min_bars=0 → returns all symbols with any 1d data
      Call with min_bars=200 to ensure ~1 year of daily history
    """
    if SYMBOLS_CACHE.exists():
        all_syms = json.loads(SYMBOLS_CACHE.read_text()).get("symbols", [])
    else:
        all_syms = ["AAPL", "NVDA", "TSLA", "MSFT", "META", "AMZN", "GOOGL", "AMD", "SPY", "QQQ"]

    if min_bars <= 0:
        return all_syms[:count]

    # Filter by actual cached bar count
    valid = []
    for sym in all_syms:
        fp = BAR_DATA_DIR / f"{sym}_{timeframe}.json"
        if not fp.exists():
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            bars = data.get("bars", data) if isinstance(data, dict) else data
            if len(bars) >= min_bars:
                valid.append(sym)
        except Exception:
            continue
        if len(valid) >= count:
            break
    return valid


def split_symbols(
    symbols: list[str],
    train_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split symbols into train and validation sets using even/odd interleaving.

    Symbol list is sorted by volume-rank (position in the list), so slicing
    first/last half would put the highest-volume stocks in train and lowest
    in val — correlated sets. Even/odd interleaving breaks that correlation:
    every consecutive pair (rank 0 vs 1, rank 2 vs 3, ...) is split across
    train and val, so both sets have similar volume distributions.

    Example: [AAPL, MSFT, NVDA, TSLA, META, AMZN]
      train (even): [AAPL, NVDA, META]   indices 0, 2, 4
      val   (odd):  [MSFT, TSLA, AMZN]  indices 1, 3, 5
    """
    train = [s for i, s in enumerate(symbols) if i % 2 == 0]
    val   = [s for i, s in enumerate(symbols) if i % 2 == 1]
    return train, val


def run_single_strategy_backtest(
    strategy_name: str,
    symbols: list[str],
    params: dict,
) -> dict:
    """
    Mini-backtest for one strategy on given symbols using ONLY cached bar data.
    No API calls are ever made — symbols without cached data are silently skipped.
    Uses all available cached history (full 180-day window).
    """
    """Run a mini-backtest with injected params. This is Optuna's objective."""
    from backend.data.schemas import BarSeries
    from backend.patterns.classifier import classify_all
    from backend.patterns.registry import PATTERN_META
    from backend.optimization.param_inject import set_params, clear_overrides
    from run_backtest import PendingTrade, _compute_atr_at
    from cache_bars import load_cached_bars

    # ── INJECT PARAMS ──
    set_params(strategy_name, params)

    meta = PATTERN_META.get(strategy_name, {})
    allowed_tfs = meta.get("tf", ["1d"])

    # Override max_hold and cooldown from trial params
    bar_min = BAR_MINUTES.get(allowed_tfs[0], 5)
    max_hold_bars = params.get("max_hold_bars", max(1, int(meta.get("mh", 600) / bar_min)))
    cooldown_bars = params.get("cooldown_bars", max(1, int(meta.get("cd", 120) / bar_min)))

    # Override position splits
    s_t1 = params.get("split_t1", 0.50)
    s_t2 = params.get("split_t2", 0.30)
    s_trail = round(max(0.0, 1.0 - s_t1 - s_t2), 2)
    if s_trail < 0.05:
        s_t2 = round(1.0 - s_t1 - 0.05, 2)
        s_trail = 0.05
    splits = (s_t1, s_t2, s_trail)

    outcomes = []

    for symbol in symbols:
        for tf in allowed_tfs:
            try:
                # Cache-only: skip if not cached — no API calls
                bars_data = load_cached_bars(symbol, tf)
                if bars_data is None:
                    continue
                bars = bars_data.bars
                n = len(bars)

                min_bars = {"5min": 60, "15min": 40, "1h": 20, "1d": 60}.get(tf, 60)
                if n < min_bars + 10:
                    continue

                scan_interval = {"5min": 12, "15min": 6, "1h": 2, "1d": 1}.get(tf, 4)

                pending = []
                last_fired = {}

                for i in range(min_bars, n):
                    # Resolve pending
                    still_pending = []
                    for trade in pending:
                        bars_held = i - trade.bar_idx
                        result = trade.check_resolution(bars[i])
                        if result is not None:
                            outcomes.append({"outcome": result[0], "realized_r": result[1]})
                        elif bars_held >= max_hold_bars:
                            result = trade.timeout_resolve(bars[i].close)
                            outcomes.append({"outcome": result[0], "realized_r": result[1]})
                        else:
                            still_pending.append(trade)
                    pending = still_pending

                    if (i - min_bars) % scan_interval != 0:
                        continue

                    window = BarSeries(symbol=symbol, timeframe=tf, bars=bars[:i+1])
                    try:
                        setups = classify_all(window)
                    except Exception:
                        continue

                    for setup in setups:
                        if setup.pattern_name != strategy_name:
                            continue

                        price_bucket = round(setup.entry_price, 0)
                        ck = f"{setup.pattern_name}_{setup.bias.value}_{price_bucket}"
                        if i - last_fired.get(ck, -9999) < cooldown_bars:
                            continue
                        if any(p.pattern_name == setup.pattern_name for p in pending):
                            continue

                        last_fired[ck] = i
                        atr = _compute_atr_at(bars, i)

                        # Override targets with trial params
                        risk_dir = 1 if setup.bias.value == "long" else -1
                        entry = setup.entry_price
                        risk_amt = abs(entry - setup.stop_loss) if abs(entry - setup.stop_loss) > 0 else atr
                        t1_override = round(entry + risk_dir * atr * params.get("t1_atr_mult", 2.0), 2)
                        t2_override = round(entry + risk_dir * atr * params.get("t2_atr_mult", 4.0), 2)
                        stop_override = round(entry - risk_dir * atr * params.get("stop_atr_mult", 2.0), 2)

                        pending.append(PendingTrade(
                            pattern_name=setup.pattern_name,
                            strategy_type=setup.strategy_type,
                            symbol=symbol, bias=setup.bias.value,
                            entry=entry, target=t2_override,
                            stop=stop_override, bar_idx=i,
                            max_hold=max_hold_bars,
                            target_1=t1_override,
                            target_2=t2_override,
                            position_splits=splits,
                            atr=atr,
                        ))

                # Resolve remaining
                for trade in pending:
                    result = trade.timeout_resolve(bars[-1].close)
                    outcomes.append({"outcome": result[0], "realized_r": result[1]})

            except Exception:
                continue

    # ── CLEAR OVERRIDES ──
    clear_overrides(strategy_name)

    # ── COMPUTE STATS ──
    if len(outcomes) < 5:
        return {"expectancy": -999, "win_rate": 0, "signals": len(outcomes), "pf": 0}

    wins = [o for o in outcomes if o["outcome"] in ("win", "partial_win")]
    losses = [o for o in outcomes if o["outcome"] == "loss"]

    wr = len(wins) / len(outcomes) if outcomes else 0
    all_r = [o["realized_r"] for o in outcomes]
    expectancy = float(np.mean(all_r))

    gross_win = sum(o["realized_r"] for o in wins) if wins else 0
    gross_loss = sum(abs(o["realized_r"]) for o in losses) if losses else 0.001
    pf = gross_win / gross_loss if gross_loss > 0 else 0

    return {
        "expectancy": round(expectancy, 4),
        "win_rate": round(wr, 4),
        "signals": len(outcomes),
        "pf": round(min(pf, 99), 3),
    }


def optimize_strategy(
    strategy_name: str,
    n_trials: int = 50,
    train_symbols: list[str] = None,
    val_symbols: list[str] = None,
    worker_id: int = 0,
) -> dict:
    """
    Optimize one strategy on train_symbols, then validate on val_symbols.

    train_symbols: symbols used for Optuna parameter search
    val_symbols:   held-out symbols evaluated once with the best params found
    worker_id:     used to offset Optuna seed so parallel workers explore differently
    """
    if not HAS_OPTUNA:
        return {"error": "Install optuna: pip install optuna"}

    # Fallback if called standalone (not via optimize_all)
    if train_symbols is None or val_symbols is None:
        all_syms = load_symbols(500)
        train_symbols, val_symbols = split_symbols(all_syms)

    all_params = get_all_params(strategy_name)

    total_combos = 1
    for spec in all_params.values():
        if spec["type"] == "cat":
            total_combos *= len(spec["choices"])
        else:
            total_combos *= max(1, int((spec["high"] - spec["low"]) / spec["step"]) + 1)

    prefix = f"[W{worker_id}] {strategy_name}"
    print(f"\n  ┌── {prefix}")
    print(f"  │   Params: {len(all_params)} ({len(UNIVERSAL_PARAMS)} universal + "
          f"{len(STRATEGY_SPECIFIC_PARAMS.get(strategy_name, {}))} specific) | "
          f"~{total_combos:,} combos | {n_trials} trials")
    print(f"  │   Train: {len(train_symbols)} syms | Val: {len(val_symbols)} syms | "
          f"Cache-only")

    best_exp = -999

    def objective(trial):
        nonlocal best_exp
        params = {}
        for pname, spec in all_params.items():
            if spec["type"] == "int":
                params[pname] = trial.suggest_int(pname, spec["low"], spec["high"], step=spec["step"])
            elif spec["type"] == "float":
                params[pname] = trial.suggest_float(pname, spec["low"], spec["high"], step=spec["step"])
            elif spec["type"] == "cat":
                params[pname] = trial.suggest_categorical(pname, spec["choices"])

        result = run_single_strategy_backtest(strategy_name, train_symbols, params)

        # Hard gates — degenerate solutions rejected outright
        if result["signals"] < 25:
            return -10.0
        if result["win_rate"] < 0.25:
            return -10.0

        # Score = expectancy weighted by statistical confidence.
        # sqrt(N/100) scales from ~0.5x at 25 signals to 1.0x at 100+,
        # so a strategy with more signals must genuinely outperform a
        # lucky sparse one, not just tie it.
        n_factor = min(1.0, (result["signals"] / 100) ** 0.5)
        score = result["expectancy"] * n_factor

        if result["expectancy"] > best_exp:
            best_exp = result["expectancy"]
            print(f"  │   #{trial.number:>3} ★ Exp={result['expectancy']:+.3f}R "
                  f"WR={result['win_rate']:.0%} PF={result['pf']:.2f} N={result['signals']}")

        return score

    # Each worker uses a different seed so parallel searches explore different regions
    sampler = optuna.samplers.TPESampler(seed=42 + worker_id * 137)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - t0

    best = study.best_trial
    best_params = best.params
    train_exp = round(best.value, 4)

    # ── VALIDATION PASS ─────────────────────────────────────────────────────
    # Run best params once on the held-out 250 symbols — no optimization here
    val_result = run_single_strategy_backtest(strategy_name, val_symbols, best_params)
    val_exp = round(val_result["expectancy"], 4) if val_result["signals"] >= 15 else None

    # Overfit flag: val drops below zero or is less than 50% of train
    overfit = False
    if val_exp is not None:
        if val_exp < 0 or (train_exp > 0 and val_exp < train_exp * 0.5):
            overfit = True

    best_universal = {k: v for k, v in best_params.items() if k in UNIVERSAL_PARAMS}
    best_specific = {k: v for k, v in best_params.items() if k not in UNIVERSAL_PARAMS}

    # Status line
    val_str = f"{val_exp:+.3f}R" if val_exp is not None else "n/a (too few)"
    flag = " ⚠ OVERFIT" if overfit else (" ✓" if not overfit else "")
    print(f"  └── {strategy_name}: train={train_exp:+.3f}R  val={val_str}{flag}  ({elapsed:.0f}s)")

    result = {
        "strategy": strategy_name,
        "best_params": best_params,
        "best_universal": best_universal,
        "best_specific": best_specific,
        "train_expectancy": train_exp,
        "val_expectancy": val_exp,
        "val_signals": val_result["signals"],
        "overfit_flag": overfit,
        "best_expectancy": train_exp,  # kept for run_weekend_optimization.py compat
        "n_trials": n_trials,
        "n_train_symbols": len(train_symbols),
        "n_val_symbols": len(val_symbols),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = strategy_name.replace(" ", "_").replace("/", "_")
    (RESULTS_DIR / f"{safe}.json").write_text(json.dumps(result, indent=2))

    return result


# ==============================================================================
# PARALLEL WORKER — must be top-level (picklable) for ProcessPoolExecutor
# ==============================================================================

def _worker_optimize_single(args: tuple) -> tuple[str, dict]:
    """
    Runs optimization for a single strategy.
    Called by ProcessPoolExecutor — must be module-level (picklable).
    Returns (name, result) so the main process can save checkpoints after each.
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    name, n_trials, train_syms, val_syms, seed = args
    try:
        result = optimize_strategy(name, n_trials, train_syms, val_syms, seed)
    except Exception as e:
        result = {"error": str(e), "strategy": name}
    return name, result


def _save_opt_checkpoint(path: Path, results: dict) -> None:
    """Atomic checkpoint write — safe against crashes mid-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "completed": list(results.keys()),
        "results": results,
        "saved_at": datetime.now().isoformat(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def optimize_all(
    n_trials: int = 60,
    n_total_symbols: int = 500,
    strategies: list[str] = None,
    n_cores: int = N_CORES,
    checkpoint_path: Path = None,
) -> dict:
    """
    Optimize all strategies in parallel across n_cores workers.

    Each strategy is submitted as an individual job. With max_workers=6 and
    30 strategies, Python keeps 6 in flight at all times — no manual chunking.

    Strategies are sorted slowest-first (5min before 1d) so that when a fast
    daily strategy finishes early, the next queued job is another slow one,
    keeping all cores busy instead of stalling at the end.

    Checkpoint: if checkpoint_path is given, results are saved after every
    completed strategy. Re-run with the same checkpoint_path to resume.

    Only optimizes strategies that are EXPLICITLY OPTED IN via either:
      1. Being passed in the `strategies` argument
      2. Having an entry in STRATEGY_SPECIFIC_PARAMS (when called directly)

    New patterns added to classifier/registry are NOT automatically optimized.
    """
    if strategies is None:
        strategies = sorted(STRATEGY_SPECIFIC_PARAMS.keys())

    from backend.patterns.registry import PATTERN_META

    # ── Guard: skip strategies not registered in PATTERN_META ───────────────
    # Strategies in STRATEGY_SPECIFIC_PARAMS but not in PATTERN_META are
    # planned/future patterns — the classifier won't detect them, so every
    # trial produces 0 signals. Skip them and warn.
    unregistered = [s for s in strategies if s not in PATTERN_META]
    if unregistered:
        print(f"\n  WARNING: {len(unregistered)} strategies not in PATTERN_META (no detector) — skipping:")
        for s in unregistered:
            print(f"    - {s}")
    strategies = [s for s in strategies if s in PATTERN_META]

    if not strategies:
        print("  No optimizable strategies found in PATTERN_META.")
        return {}

    # ── Resume from checkpoint ───────────────────────────────────────────────
    already_done: dict = {}
    if checkpoint_path and checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text())
            already_done = ckpt.get("results", {})
            done_names = set(ckpt.get("completed", []))
            before = len(strategies)
            strategies = [s for s in strategies if s not in done_names]
            print(f"  Checkpoint loaded: {len(done_names)} done, "
                  f"{len(strategies)}/{before} remaining")
        except Exception as e:
            print(f"  Checkpoint read failed ({e}) — starting fresh")

    if not strategies:
        print("  All strategies already completed. Use --fresh to re-run.")
        return already_done

    # ── Symbol pools, one per timeframe ─────────────────────────────────────
    sym_pools = {
        "1d":    load_symbols(n_total_symbols, timeframe="1d",    min_bars=200),
        "1h":    load_symbols(n_total_symbols, timeframe="1h",    min_bars=500),
        "15min": load_symbols(n_total_symbols, timeframe="15min", min_bars=2000),
        "5min":  load_symbols(n_total_symbols, timeframe="5min",  min_bars=5000),
    }
    split_pools = {tf: split_symbols(syms) for tf, syms in sym_pools.items()}

    def train_val_for(name: str) -> tuple[list[str], list[str]]:
        primary_tf = PATTERN_META.get(name, {}).get("tf", ["5min"])[0]
        return split_pools.get(primary_tf, split_pools["5min"])

    # ── Sort slowest-first (LPT scheduling) ──────────────────────────────────
    # 5min strategies take ~10x longer than 1d. Submitting slowest first means
    # a freed core always picks up a slow job rather than stalling at the end
    # with one slow strategy left while 5 cores idle.
    def tf_weight(name: str) -> int:
        primary_tf = PATTERN_META.get(name, {}).get("tf", ["5min"])[0]
        return TF_WEIGHT.get(primary_tf, 4)

    strategies = sorted(strategies, key=tf_weight, reverse=True)

    n_remaining = len(strategies)
    n_workers = min(n_cores, n_remaining)

    print(f"\n{'=' * 70}")
    print(f"  Strategy Parameter Optimizer v3 -- Parallel")
    print(f"  Strategies:      {n_remaining} remaining  ({len(already_done)} already done)")
    print(f"  Cores:           {n_workers}")
    print(f"  Trials/strategy: {n_trials}")
    for tf, syms in sym_pools.items():
        tr, va = split_pools[tf]
        print(f"  {tf:<6} symbols:  {len(syms)} usable -> {len(tr)} train / {len(va)} val")
    print(f"  Data source:     cache-only (no API calls)")
    print(f"  Checkpoint:      {'auto-save per strategy' if checkpoint_path else 'disabled'}")
    print(f"  Submission order (slowest first):")
    for name in strategies:
        tf = PATTERN_META.get(name, {}).get("tf", ["?"])[0]
        print(f"    {name:<32} [{tf}]")
    print(f"{'=' * 70}\n")

    t_start = time.time()
    all_results: dict = dict(already_done)
    completed_count = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for i, name in enumerate(strategies):
            train_syms, val_syms = train_val_for(name)
            seed = 42 + i * 137
            fut = executor.submit(
                _worker_optimize_single,
                (name, n_trials, train_syms, val_syms, seed),
            )
            futures[fut] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                _, result = future.result()
            except Exception as e:
                result = {"error": str(e), "strategy": name}

            all_results[name] = result
            completed_count += 1

            # Save checkpoint after every strategy — crash-safe
            if checkpoint_path:
                _save_opt_checkpoint(checkpoint_path, all_results)

            tf = PATTERN_META.get(name, {}).get("tf", ["?"])[0]
            if "error" in result:
                print(f"  [{completed_count}/{n_remaining}] {name:<32} ERROR: {result['error'][:40]}")
            else:
                t_exp = result.get("train_expectancy", result.get("expectancy", 0))
                v_exp = result.get("val_expectancy")
                v_str = f"{v_exp:+.3f}R" if v_exp is not None else "n/a"
                overfit = " OVERFIT" if result.get("overfit_flag") else ""
                print(f"  [{completed_count}/{n_remaining}] {name:<32} "
                      f"train:{t_exp:+.3f}R  val:{v_str:>8}  [{tf}]{overfit}")

    elapsed = time.time() - t_start

    # Save combined results
    GLOBAL_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_PARAMS.write_text(json.dumps(all_results, indent=2))

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  OPTIMIZATION COMPLETE — {elapsed/60:.0f}m ({elapsed/3600:.1f}h)")
    print(f"{'═' * 70}")
    print(f"\n  {'Strategy':<32} {'TF':<6} {'Train EXP':>10} {'Val EXP':>10}  Status")
    print(f"  {'─' * 66}")

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get("train_expectancy", -999),
        reverse=True,
    )

    n_positive_val = 0
    for name, res in sorted_results:
        tf = PATTERN_META.get(name, {}).get("tf", ["?"])[0]
        if "error" in res:
            print(f"  {name:<32} {tf:<6} {'ERROR':>10} {'':>10}  {res['error'][:20]}")
            continue
        t_exp = res.get("train_expectancy", 0)
        v_exp = res.get("val_expectancy")
        overfit = res.get("overfit_flag", False)
        v_str = f"{v_exp:+.3f}R" if v_exp is not None else "  n/a  "
        status = "OVERFIT" if overfit else ("ok" if v_exp and v_exp >= 0 else "neg val")
        if v_exp is not None and v_exp >= 0:
            n_positive_val += 1
        print(f"  {name:<32} {tf:<6} {t_exp:>+9.3f}R {v_str:>10}  {status}")

    print(f"\n  Strategies with positive val EXP: {n_positive_val}/{len(sorted_results)}")
    print(f"  Results: {GLOBAL_PARAMS}")
    print(f"{'═' * 70}")

    return all_results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Parameter Optimizer v3")
    parser.add_argument("--strategy", type=str, default=None, help="Single strategy name")
    parser.add_argument("--trials", type=int, default=60, help="Optuna trials per strategy")
    parser.add_argument("--symbols", type=int, default=500, help="Total symbol pool size (split 50/50)")
    parser.add_argument("--cores", type=int, default=N_CORES, help="Parallel worker processes")
    parser.add_argument("--list", action="store_true", help="List all strategies and params")
    args = parser.parse_args()

    if args.list:
        from backend.patterns.registry import PATTERN_META
        all_strats = sorted(PATTERN_META.keys())
        print(f"\n  Strategies ({len(all_strats)}):")
        for name in all_strats:
            specific = STRATEGY_SPECIFIC_PARAMS.get(name, {})
            total = len(UNIVERSAL_PARAMS) + len(specific)
            print(f"    {name:<30} [{total} params: {len(UNIVERSAL_PARAMS)} universal + {len(specific)} specific]")
        sys.exit(0)

    if not HAS_OPTUNA:
        print("  Install Optuna: pip install optuna")
        sys.exit(1)

    if args.strategy:
        from backend.patterns.registry import PATTERN_META
        primary_tf = PATTERN_META.get(args.strategy, {}).get("tf", ["5min"])[0]
        min_bars_map = {"1d": 200, "1h": 500, "15min": 2000, "5min": 5000}
        mb = min_bars_map.get(primary_tf, 5000)
        all_syms = load_symbols(args.symbols, timeframe=primary_tf, min_bars=mb)
        train_syms, val_syms = split_symbols(all_syms)
        print(f"  {args.strategy} [{primary_tf}]: "
              f"{len(train_syms)} train / {len(val_syms)} val symbols ({len(all_syms)} usable)")
        optimize_strategy(args.strategy, args.trials, train_syms, val_syms)
    else:
        optimize_all(args.trials, args.symbols, strategies=None, n_cores=args.cores)