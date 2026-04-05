"""
run_weekend_optimization.py — Automated weekend optimization runner.

Runs optimization on all strategies in priority order with:
  - Checkpointing after each strategy (crash-safe)
  - Priority tiers (best strategies get more trials)
  - Time estimates and progress reporting
  - Auto-resume if interrupted

Usage:
  python run_weekend_optimization.py                    # Full run
  python run_weekend_optimization.py --resume           # Resume after crash
  python run_weekend_optimization.py --tier 1           # Only Tier 1 strategies
  python run_weekend_optimization.py --quick            # 20 trials per (fast test)

Expected runtime (50 symbols, 90 days):
  Tier 1 (13 strategies × 80 trials):  ~9 hours
  Tier 2 (12 strategies × 50 trials):  ~5 hours
  Tier 3 (27 strategies × 30 trials):  ~7 hours
  TOTAL:                                ~21 hours
"""
import json
import time
import sys
import argparse
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# STRATEGY TIERS — priority order for optimization
# ═══════════════════════════════════════════════════════════════

# Tier 1: Confirmed positive edge — more trials to make them elite
TIER_1 = [
    # Grade B — proven edge
    "Accumulation Long",       # +0.756R, Grade B
    "Juicer Long",             # +2.486R, Grade B (fat tail)
    "RS Persistence Long",     # +2.089R, Grade B (fat tail)
    "Gap Reversal Long",       # +0.100R, Grade B
    "Tidal Wave",              # +0.128R, Grade B
    # Positive expectancy with large sample
    "Momentum Breakout",       # +0.724R, 269 signals
    "Turtle Breakout Long",    # +0.639R, 442 signals
    "VP Divergence Long",      # +0.429R, 6629 signals
    "VP Divergence Short",     # +0.195R, 8564 signals
    "BAB Long",                # +0.406R, 1842 signals
    "Streak Reversal Long",    # +0.308R, 1582 signals
    "TS Momentum Long",        # +0.274R, 1985 signals
    "ST Reversal Long",        # +0.136R, 392 signals
]

# Tier 2: Near breakeven — optimization might flip them positive
TIER_2 = [
    "BB Squeeze Long",         # +0.100R, 331 signals
    "Mean Reversion",          # +0.093R, 2914 signals
    "Gap Fade",                # untested — 5min quant
    "Trend Pullback",          # untested — 5min quant
    "Opening Drive Long",      # +0.029R, 224 signals
    "VWAP Reversion",          # +0.029R, 2679 signals
    "Volume Climax Long",      # -0.018R, 847 signals (near zero)
    "ATR Expansion Long",      # -0.017R, 1278 signals (near zero)
    "Midday Reversal Long",    # -0.138R, 374 signals
    "VWAP Trend Long",         # -0.013R, 1308 signals (near zero)
    "RSI Divergence Long",     # -0.043R, 3168 signals
    "Donchian Breakout",       # -0.088R, 1241 signals
]

# Tier 3: Previously negative or unproven classical — establish baseline
TIER_3 = [
    # Classical structural (need baseline before optimizing)
    "Head & Shoulders",
    "Inverse H&S",
    "Double Top",
    "Double Bottom",
    "Triple Top",
    "Triple Bottom",
    "Ascending Triangle",
    "Descending Triangle",
    "Symmetrical Triangle",
    "Bull Flag",
    "Bear Flag",
    "Cup & Handle",
    "Rectangle",
    "Rising Wedge",
    "Falling Wedge",
    # SMB scalps
    "RubberBand Scalp",
    "ORB 15min",
    "ORB 30min",
    "Gap Give & Go",
    "Second Chance Scalp",
    "Fashionably Late",
    # Daily quant with negative results
    "Multi-TF Trend Long",     # -0.131R, 741 signals
    "Range Expansion",         # -0.146R, 1788 signals
    "Volume Breakout",         # -0.064R, 614 signals
    "Keltner Breakout Long",   # -0.152R, 1279 signals
    "MACD Turn Long",          # -0.267R, 1445 signals
]

TIER_TRIALS = {1: 80, 2: 50, 3: 30}
TIER_MAP = {name: 1 for name in TIER_1}
TIER_MAP.update({name: 2 for name in TIER_2})
TIER_MAP.update({name: 3 for name in TIER_3})

# ═══════════════════════════════════════════════════════════════
# CHECKPOINTING
# ═══════════════════════════════════════════════════════════════

CHECKPOINT = Path("cache/optimization_checkpoint.json")
RESULTS_DIR = Path("cache/optimization")


def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        try:
            return json.loads(CHECKPOINT.read_text())
        except Exception:
            pass
    return {"completed": [], "results": {}, "started_at": datetime.now().isoformat()}


def save_checkpoint(state: dict):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    state["saved_at"] = datetime.now().isoformat()
    CHECKPOINT.write_text(json.dumps(state, indent=2))


# ═══════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════

def run_optimization(
    tiers: list[int] = None,
    n_symbols: int = 50,
    days_back: int = 90,
    trial_override: int = None,
    resume: bool = False,
):
    try:
        from backend.optimization.param_optimizer import (
            optimize_strategy, STRATEGY_SPECIFIC_PARAMS, UNIVERSAL_PARAMS
        )
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        print("  Place param_optimizer.py in backend/optimization/")
        sys.exit(1)

    if tiers is None:
        tiers = [1, 2, 3]

    # Build strategy list in priority order
    strategies = []
    for tier in sorted(tiers):
        tier_list = {1: TIER_1, 2: TIER_2, 3: TIER_3}.get(tier, [])
        for name in tier_list:
            if name in STRATEGY_SPECIFIC_PARAMS or True:  # Universal params always available
                strategies.append((name, tier))

    # Load checkpoint
    state = load_checkpoint() if resume else {
        "completed": [], "results": {}, "started_at": datetime.now().isoformat()
    }
    completed = set(state.get("completed", []))

    remaining = [(name, tier) for name, tier in strategies if name not in completed]

    # Time estimate
    total_trials = sum(
        trial_override or TIER_TRIALS.get(tier, 50)
        for _, tier in remaining
    )
    # Rough estimate: ~3 minutes per trial at 50 symbols
    est_minutes = total_trials * 3
    est_hours = est_minutes / 60

    print(f"\n{'═' * 70}")
    print(f"  WEEKEND OPTIMIZATION RUNNER")
    print(f"{'═' * 70}")
    print(f"  Strategies:    {len(strategies)} total ({len(remaining)} remaining)")
    print(f"  Completed:     {len(completed)}")
    print(f"  Symbols:       {n_symbols}")
    print(f"  Days:          {days_back}")
    print(f"  Tiers:         {tiers}")
    print(f"  Total trials:  {total_trials}")
    print(f"  Est. time:     {est_hours:.1f} hours ({est_minutes:.0f} min)")
    print(f"  Universal params: {len(UNIVERSAL_PARAMS)} (stop, targets, hold, splits)")
    print(f"{'═' * 70}")

    if not remaining:
        print("\n  All strategies already optimized! Delete checkpoint to re-run:")
        print(f"    del {CHECKPOINT}")
        return

    t_global = time.time()
    errors = []

    for idx, (name, tier) in enumerate(remaining):
        trials = trial_override or TIER_TRIALS.get(tier, 50)

        elapsed_global = time.time() - t_global
        if idx > 0:
            rate = elapsed_global / idx  # seconds per strategy
            eta = rate * (len(remaining) - idx)
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}m"
        else:
            eta_str = "calculating..."

        print(f"\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │ [{idx+1+len(completed)}/{len(strategies)}] {name:<30} Tier {tier} │")
        print(f"  │ Trials: {trials:<5}  Symbols: {n_symbols:<5}  ETA: {eta_str:<10} │")
        print(f"  └─────────────────────────────────────────────────┘")

        try:
            t0 = time.time()
            result = optimize_strategy(name, trials, n_symbols, days_back)
            elapsed = time.time() - t0

            if "error" in result:
                print(f"    ✗ ERROR: {result['error']}")
                errors.append(name)
                state["results"][name] = {"error": result["error"], "tier": tier}
            else:
                exp = result.get("best_expectancy", 0)
                marker = "★" if exp > 0.1 else "✓" if exp > 0 else "—"
                print(f"    {marker} Best exp: {exp:+.4f}R  ({elapsed:.0f}s)")
                state["results"][name] = {
                    "best_expectancy": exp,
                    "best_params": result.get("best_params", {}),
                    "tier": tier,
                    "trials": trials,
                    "elapsed": round(elapsed, 1),
                }

        except KeyboardInterrupt:
            print(f"\n  ⚠ INTERRUPTED — saving checkpoint...")
            save_checkpoint(state)
            print(f"  Checkpoint saved. Resume with: python run_weekend_optimization.py --resume")
            sys.exit(0)

        except Exception as e:
            print(f"    ✗ CRASH: {e}")
            errors.append(name)
            state["results"][name] = {"error": str(e), "tier": tier}

        # Save checkpoint after each strategy
        state["completed"] = list(completed | {name})
        completed.add(name)
        save_checkpoint(state)

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════

    total_time = time.time() - t_global

    print(f"\n{'═' * 70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'═' * 70}")
    print(f"  Time: {total_time/3600:.1f} hours ({total_time/60:.0f} min)")
    print(f"  Strategies: {len(completed)}")
    print(f"  Errors: {len(errors)}")

    # Results table
    print(f"\n  {'Strategy':<30} {'Tier':>4} {'Exp':>8} {'Status'}")
    print(f"  {'─' * 60}")

    sorted_results = sorted(
        state["results"].items(),
        key=lambda x: x[1].get("best_expectancy", -999),
        reverse=True,
    )

    positive = 0
    for name, res in sorted_results:
        if "error" in res:
            print(f"  {name:<30} {res.get('tier', '?'):>4} {'ERROR':>8} {res['error'][:30]}")
        else:
            exp = res.get("best_expectancy", 0)
            marker = "★" if exp > 0.1 else "✓" if exp > 0 else "✗"
            print(f"  {name:<30} {res.get('tier', '?'):>4} {exp:>+7.3f}R  {marker}")
            if exp > 0:
                positive += 1

    print(f"\n  Positive edge: {positive}/{len(sorted_results)}")

    # Save final combined params
    combined_params = {}
    for name, res in state["results"].items():
        if "best_params" in res:
            combined_params[name] = res["best_params"]

    params_path = Path("cache/optimized_params.json")
    params_path.write_text(json.dumps(combined_params, indent=2))
    print(f"  Params saved: {params_path}")

    # Generate apply script
    apply_path = Path("cache/apply_optimized_params.py")
    apply_path.write_text(
        '"""Auto-generated: load optimized params at startup."""\n'
        'import json\n'
        'from pathlib import Path\n'
        'from backend.optimization.param_inject import set_params\n\n'
        'def load_optimized_params():\n'
        '    p = Path("cache/optimized_params.json")\n'
        '    if not p.exists(): return 0\n'
        '    data = json.loads(p.read_text())\n'
        '    for name, params in data.items():\n'
        '        set_params(name, params)\n'
        '    return len(data)\n\n'
        'if __name__ == "__main__":\n'
        '    n = load_optimized_params()\n'
        '    print(f"Loaded optimized params for {n} strategies")\n'
    )
    print(f"  Apply script: {apply_path}")

    if errors:
        print(f"\n  Failed strategies: {', '.join(errors)}")

    print(f"\n  NEXT STEPS:")
    print(f"    1. python cache/apply_optimized_params.py  (verify params load)")
    print(f"    2. del cache\\backtest_checkpoint.json")
    print(f"    3. del cache\\backtest_results.json")
    print(f"    4. python run_backtest.py --from-cache --days 160 --daily  (validate)")
    print(f"    5. Compare before/after pattern stats")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekend Optimization Runner")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--tier", type=int, default=None, help="Only run specific tier (1, 2, or 3)")
    parser.add_argument("--quick", action="store_true", help="20 trials per strategy (fast test)")
    parser.add_argument("--symbols", type=int, default=50, help="Symbols per optimization")
    parser.add_argument("--days", type=int, default=90, help="Days lookback")
    args = parser.parse_args()

    tiers = [args.tier] if args.tier else [1, 2, 3]
    trial_override = 20 if args.quick else None

    run_optimization(
        tiers=tiers,
        n_symbols=args.symbols,
        days_back=args.days,
        trial_override=trial_override,
        resume=args.resume,
    )