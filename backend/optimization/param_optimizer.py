"""
backend/optimization/param_optimizer.py — Strategy Parameter Optimizer v2

Uses Optuna (Bayesian optimization) to find optimal parameters for each
strategy including: detection thresholds, stop/target multiples, hold time,
cooldown, position splits, and minimum R:R gate.

EVERY strategy gets these UNIVERSAL params tuned automatically:
  - stop_atr_mult:  How wide the stop is (ATR multiplier)
  - t1_atr_mult:    First target distance
  - t2_atr_mult:    Second target distance
  - max_hold_min:   How long to hold before timeout (minutes)
  - cooldown_min:   Minimum time between signals (minutes)
  - split_t1:       Position % exited at T1
  - split_t2:       Position % exited at T2
  - split_trail:    Position % that trails (1 - t1 - t2)

Plus STRATEGY-SPECIFIC params (ADX threshold, EMA periods, z-scores, etc.)

USAGE:
  pip install optuna
  python -m backend.optimization.param_optimizer --strategy "Juicer Long" --trials 30
  python -m backend.optimization.param_optimizer --trials 50           # All strategies
  python -m backend.optimization.param_optimizer --list                # Show all

DEPENDENCIES:
  pip install optuna
"""
import argparse
import json
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

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
    "cooldown_bars":  {"type": "int",   "low": 1,    "high": 50,   "step": 2},
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
    "Second Chance Scalp": {
        "tolerance_atr":     {"type": "float", "low": 0.15, "high": 0.60, "step": 0.05},
        "vol_mult":          {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
    },
    "Fashionably Late": {
        "ema_slope_min_atr": {"type": "float", "low": 0.03, "high": 0.25, "step": 0.02},
        "t1_mm_pct":         {"type": "float", "low": 0.40, "high": 1.00, "step": 0.10},
    },
    "RubberBand Scalp": {
        "extension_atr":     {"type": "float", "low": 1.5, "high": 3.5, "step": 0.25},
        "bounce_vol_mult":   {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
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
    "Head & Shoulders": {
        "stop_atr_buffer":   {"type": "float", "low": 0.03, "high": 0.25, "step": 0.02},
        "t1_mm_pct":         {"type": "float", "low": 0.40, "high": 1.00, "step": 0.10},
        "vol_mult":          {"type": "float", "low": 1.0, "high": 2.0, "step": 0.1},
    },
    "Rising Wedge": {
        "stop_atr_buffer":   {"type": "float", "low": 0.03, "high": 0.25, "step": 0.02},
        "t1_mm_pct":         {"type": "float", "low": 0.40, "high": 1.00, "step": 0.10},
    },
    "Descending Triangle": {
        "stop_atr_buffer":   {"type": "float", "low": 0.03, "high": 0.25, "step": 0.02},
        "t1_mm_pct":         {"type": "float", "low": 0.40, "high": 1.00, "step": 0.10},
        "vol_mult":          {"type": "float", "low": 1.0, "high": 2.0, "step": 0.1},
    },
    "Double Bottom": {
        "stop_atr_buffer":   {"type": "float", "low": 0.03, "high": 0.25, "step": 0.02},
        "t1_mm_pct":         {"type": "float", "low": 0.40, "high": 1.00, "step": 0.10},
    },
    "Cup & Handle": {
        "stop_atr_buffer":   {"type": "float", "low": 0.03, "high": 0.25, "step": 0.02},
        "t1_mm_pct":         {"type": "float", "low": 0.40, "high": 1.00, "step": 0.10},
    },
    "Falling Wedge": {
        "stop_atr_buffer":   {"type": "float", "low": 0.03, "high": 0.25, "step": 0.02},
        "t1_mm_pct":         {"type": "float", "low": 0.40, "high": 1.00, "step": 0.10},
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
        "lookback_window":   {"type": "int",   "low": 7,  "high": 20, "step": 2},
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
RESULTS_DIR = Path("cache/optimization")
GLOBAL_PARAMS = Path("cache/optimized_params.json")
BAR_MINUTES = {"5min": 5, "15min": 15, "1h": 60, "1d": 390}


def load_symbols(count: int = 100) -> list[str]:
    if SYMBOLS_CACHE.exists():
        data = json.loads(SYMBOLS_CACHE.read_text())
        return data.get("symbols", [])[:count]
    return ["AAPL", "NVDA", "TSLA", "MSFT", "META", "AMZN", "GOOGL", "AMD", "SPY", "QQQ"]


def run_single_strategy_backtest(
    strategy_name: str,
    symbols: list[str],
    params: dict,
    days_back: int = 90,
) -> dict:
    """Run a mini-backtest with injected params. This is Optuna's objective."""
    from backend.data.massive_client import fetch_bars
    from backend.data.schemas import BarSeries
    from backend.patterns.classifier import classify_all
    from backend.patterns.registry import PATTERN_META
    from backend.optimization.param_inject import set_params, clear_overrides
    from run_backtest import PendingTrade, _compute_atr_at

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
                fetch_days = max(days_back, 365) if tf == "1d" else days_back
                bars_data = fetch_bars(symbol, tf, fetch_days)
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
    n_symbols: int = 50,
    days_back: int = 90,
) -> dict:
    if not HAS_OPTUNA:
        return {"error": "Install optuna: pip install optuna"}

    all_params = get_all_params(strategy_name)
    symbols = load_symbols(n_symbols)

    total_combos = 1
    for spec in all_params.values():
        if spec["type"] == "cat":
            total_combos *= len(spec["choices"])
        else:
            total_combos *= max(1, int((spec["high"] - spec["low"]) / spec["step"]) + 1)

    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  Optimizing: {strategy_name}")
    print(f"  Universal params: {len(UNIVERSAL_PARAMS)}")
    print(f"  Specific params:  {len(STRATEGY_SPECIFIC_PARAMS.get(strategy_name, {}))}")
    print(f"  Total params:     {len(all_params)}")
    print(f"  Search space:     ~{total_combos:,} combinations")
    print(f"  Optuna trials:    {n_trials} (Bayesian, not grid)")
    print(f"  Symbols: {len(symbols)} | Days: {days_back}")
    print(f"  ═══════════════════════════════════════════════════")

    best_exp = -999

    def objective(trial):
        nonlocal best_exp

        params = {}
        for name, spec in all_params.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec["step"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], step=spec["step"])
            elif spec["type"] == "cat":
                params[name] = trial.suggest_categorical(name, spec["choices"])

        result = run_single_strategy_backtest(
            strategy_name, symbols, params, days_back,
        )

        # Penalize too few signals
        if result["signals"] < 10:
            return -10.0

        score = result["expectancy"]

        # Small bonus for reasonable signal count
        if result["signals"] > 30:
            score += 0.005

        is_best = result["expectancy"] > best_exp
        if is_best:
            best_exp = result["expectancy"]

        marker = "★ BEST" if is_best else ""
        print(f"    #{trial.number:>3}: Exp={result['expectancy']:+.3f}R "
              f"WR={result['win_rate']:.0%} PF={result['pf']:.2f} "
              f"N={result['signals']:>4} {marker}")

        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - t0

    best = study.best_trial

    # Categorize params
    best_universal = {k: v for k, v in best.params.items() if k in UNIVERSAL_PARAMS}
    best_specific = {k: v for k, v in best.params.items() if k not in UNIVERSAL_PARAMS}

    result = {
        "strategy": strategy_name,
        "best_params": best.params,
        "best_universal": best_universal,
        "best_specific": best_specific,
        "best_expectancy": round(best.value, 4),
        "n_trials": n_trials,
        "n_symbols": n_symbols,
        "days_back": days_back,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = strategy_name.replace(" ", "_").replace("/", "_")
    (RESULTS_DIR / f"{safe}.json").write_text(json.dumps(result, indent=2))

    print(f"\n  ── Results: {strategy_name} ──")
    print(f"  Best expectancy: {best.value:+.4f}R")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Universal params:")
    for k, v in sorted(best_universal.items()):
        print(f"    {k:<20} = {v}")
    if best_specific:
        print(f"  Strategy-specific params:")
        for k, v in sorted(best_specific.items()):
            print(f"    {k:<20} = {v}")

    return result


def optimize_all(
    n_trials: int = 50,
    n_symbols: int = 50,
    days_back: int = 90,
    strategies: list[str] = None,
) -> dict:
    if strategies is None:
        strategies = sorted(set(list(STRATEGY_SPECIFIC_PARAMS.keys())))

    print(f"\n{'═' * 70}")
    print(f"  Strategy Parameter Optimizer v2")
    print(f"  {len(strategies)} strategies × {n_trials} trials each")
    print(f"  {len(UNIVERSAL_PARAMS)} universal + strategy-specific params")
    print(f"  {n_symbols} symbols × {days_back} days")
    print(f"{'═' * 70}")

    all_results = {}
    t_start = time.time()

    for i, name in enumerate(strategies):
        print(f"\n  [{i+1}/{len(strategies)}] {name}")
        try:
            result = optimize_strategy(name, n_trials, n_symbols, days_back)
            all_results[name] = result
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results[name] = {"error": str(e)}

    GLOBAL_PARAMS.write_text(json.dumps(all_results, indent=2))

    elapsed = time.time() - t_start
    print(f"\n{'═' * 70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"  Strategies: {len(strategies)}")
    print(f"  Time: {elapsed/60:.0f}m ({elapsed/3600:.1f}h)")
    print(f"  Results: {GLOBAL_PARAMS}")
    print(f"{'═' * 70}")

    # Summary table
    print(f"\n  {'Strategy':<30} {'Before':>8} {'After':>8} {'Change':>8}")
    print(f"  {'─' * 56}")
    for name, res in sorted(all_results.items(), key=lambda x: x[1].get("best_expectancy", -999), reverse=True):
        if "error" in res:
            continue
        exp = res.get("best_expectancy", 0)
        print(f"  {name:<30} {'?':>8} {exp:>+7.3f}R {'':>8}")

    return all_results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Parameter Optimizer v2")
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--symbols", type=int, default=50)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"\n  Optimizable strategies ({len(STRATEGY_SPECIFIC_PARAMS)}):")
        for name in sorted(STRATEGY_SPECIFIC_PARAMS.keys()):
            specific = STRATEGY_SPECIFIC_PARAMS[name]
            total = len(UNIVERSAL_PARAMS) + len(specific)
            print(f"    {name:<30} [{total} params: {len(UNIVERSAL_PARAMS)} universal + {len(specific)} specific]")
        sys.exit(0)

    if not HAS_OPTUNA:
        print("  Install Optuna: pip install optuna")
        sys.exit(1)

    if args.strategy:
        optimize_strategy(args.strategy, args.trials, args.symbols, args.days)
    else:
        optimize_all(args.trials, args.symbols, args.days)