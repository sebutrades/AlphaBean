"""
run_weekend_optimization.py — Strategy optimization runner.

Optimizes every strategy listed in STRATEGY_SPECIFIC_PARAMS with 60 Optuna
trials each, across 6 parallel cores. Strategies are sorted slowest-first
(5min before 1d) so all cores stay busy throughout the run.

Checkpointing is automatic — every completed strategy is saved immediately.
If the run crashes or is interrupted, just re-run the same command to resume.
Use --fresh to ignore the checkpoint and start over.

Usage:
  python run_weekend_optimization.py                       # Full run / auto-resume
  python run_weekend_optimization.py --fresh               # Start from scratch
  python run_weekend_optimization.py --trials 10           # Smoke test (fast)
  python run_weekend_optimization.py --symbols 500 --cores 6

Prerequisites:
  pip install optuna
  python fetch_symbols.py
  python cache_bars.py --symbols 500 --days 180
"""
import json
import sys
import argparse
from pathlib import Path

CHECKPOINT = Path("cache/optimization_checkpoint.json")


def run_optimization(
    n_symbols: int = 500,
    n_trials: int = 60,
    n_cores: int = 6,
    fresh: bool = False,
):
    try:
        from backend.optimization.param_optimizer import (
            optimize_all,
            STRATEGY_SPECIFIC_PARAMS,
        )
    except ImportError as e:
        print(f"  Import error: {e}")
        sys.exit(1)

    if fresh and CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("  Checkpoint cleared — starting fresh.")

    checkpoint = None if fresh else CHECKPOINT
    strategies = sorted(STRATEGY_SPECIFIC_PARAMS.keys())

    print(f"\n{'=' * 60}")
    print(f"  Optimization Runner")
    print(f"{'=' * 60}")
    print(f"  Strategies: {len(strategies)}")
    print(f"  Trials:     {n_trials} per strategy")
    print(f"  Cores:      {n_cores}")
    print(f"  Symbols:    {n_symbols} (split 50/50 even/odd)")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"{'=' * 60}")

    try:
        all_results = optimize_all(
            n_trials=n_trials,
            n_total_symbols=n_symbols,
            strategies=strategies,
            n_cores=n_cores,
            checkpoint_path=checkpoint,
        )
    except KeyboardInterrupt:
        print("\n  INTERRUPTED — progress saved. Re-run to resume.")
        sys.exit(0)

    # Extract best params (only non-overfit, or flagged-but-saved if all overfit)
    combined: dict = {}
    for name, res in all_results.items():
        if "best_params" in res:
            combined[name] = res["best_params"]

    params_path = Path("cache/optimized_params.json")
    params_path.write_text(json.dumps(combined, indent=2))
    print(f"\n  Params saved: {params_path}  ({len(combined)} strategies)")
    print(f"\n  NEXT STEPS:")
    print(f"    1. python cache/apply_optimized_params.py")
    print(f"    2. del cache\\backtest_results.json")
    print(f"    3. python run_backtest.py --from-cache --daily")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Optimization Runner")
    parser.add_argument("--fresh",   action="store_true", help="Ignore checkpoint, start from scratch")
    parser.add_argument("--trials",  type=int, default=60,  help="Optuna trials per strategy")
    parser.add_argument("--symbols", type=int, default=500, help="Total symbol pool size")
    parser.add_argument("--cores",   type=int, default=6,   help="Parallel worker processes")
    args = parser.parse_args()

    run_optimization(
        n_symbols=args.symbols,
        n_trials=args.trials,
        n_cores=args.cores,
        fresh=args.fresh,
    )
