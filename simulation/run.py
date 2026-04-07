"""
simulation/run.py — CLI entry point for the autonomous trading simulation.

Usage:
    python -m simulation.run                          # Full 180-day agent run
    python -m simulation.run --days 10 --no-agents    # Quick deterministic test
    python -m simulation.run --days 30 --verbose      # 30-day run with progress
    python -m simulation.run --capital 50000           # Custom starting capital
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.config import SimConfig
from simulation.engine import SimulationEngine


def main():
    parser = argparse.ArgumentParser(
        description="AlphaBean Autonomous Trading Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m simulation.run --days 10 --no-agents   Deterministic test
  python -m simulation.run --days 180              Full agent-driven run
  python -m simulation.run --days 30 --capital 50000 --universe 30
""",
    )
    parser.add_argument("--days", type=int, default=180,
                        help="Number of trading days to simulate (default: 180)")
    parser.add_argument("--no-agents", action="store_true",
                        help="Run in deterministic mode (no AI agents)")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Starting capital (default: $100,000)")
    parser.add_argument("--universe", type=int, default=0,
                        help="Symbols per day (0 = all cached, default: 0)")
    parser.add_argument("--risk", type=float, default=1.0,
                        help="Risk per trade %% (default: 1.0)")
    parser.add_argument("--max-trades", type=int, default=5,
                        help="Max new trades per day (default: 5)")
    parser.add_argument("--lookback", type=int, default=60,
                        help="Lookback bars before first sim day (default: 60)")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save checkpoint every N days (default: 5)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-day output")
    parser.add_argument("--min-score", type=float, default=45.0,
                        help="Minimum composite score to consider (default: 45)")

    args = parser.parse_args()

    config = SimConfig(
        starting_capital=args.capital,
        risk_per_trade_pct=args.risk,
        sim_days=args.days,
        lookback_bars=args.lookback,
        universe_size=args.universe,
        max_trades_per_day=args.max_trades,
        min_composite_score=args.min_score,
        use_agents=not args.no_agents,
        checkpoint_interval=args.checkpoint_every,
        verbose=not args.quiet,
    )

    engine = SimulationEngine(config)
    engine.run(max_days=args.days)

    # Print cost summary if agents were used
    if config.use_agents:
        from simulation.agents.base import cost_tracker
        print(f"\n  API Cost: {cost_tracker.summary()}")


if __name__ == "__main__":
    main()
