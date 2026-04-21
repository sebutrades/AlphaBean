"""
simulation/engine.py — SimulationEngine: main day-by-day loop.

Walks through historical bar data one day at a time:
  1. Select universe (top 50 by volume)
  2. Detect patterns on each symbol
  3. Score setups with multi-factor model
  4. Either use agents (PM + Risk) or deterministic top-N selection
  5. Open positions, resolve existing ones bar-by-bar
  6. Record daily snapshots

In deterministic mode (--no-agents): takes the top N setups by composite
score each day. In agent mode: delegates to the coordinator for full
agent-driven decision making.
"""
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

# Suppress harmless numpy warnings from empty arrays in feature computation
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

from backend.data.schemas import BarSeries
from backend.features.engine import compute_features
from backend.patterns.classifier import classify_all
from backend.regime.detector import detect_regime
from backend.scoring.multi_factor import score_setup, ScoredSetup
from backend.strategies.evaluator import StrategyEvaluator

from simulation.config import SimConfig
from simulation.portfolio import PortfolioState, SimPosition
from simulation.timeline import TimelineManager
from simulation.universe import select_universe


class SimulationEngine:
    """Runs the full day-by-day trading simulation."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.config.ensure_dirs()

        # Load ALL symbols from bar store to build timeline
        self._all_symbols = self._discover_symbols()
        self.timeline = TimelineManager(config, self._all_symbols)
        self.portfolio = PortfolioState(config)
        self.evaluator = StrategyEvaluator()

        # Agent coordinator (lazy import — only when agents enabled)
        self._coordinator = None

        # Stats
        self.start_time = 0.0
        self.days_run = 0
        self.total_setups_found = 0

    def _discover_symbols(self) -> list[str]:
        """Find all symbols with daily bar data on disk."""
        tf_dir = self.config.bar_data_dir / self.config.timeframe
        if not tf_dir.exists():
            return []
        return [f.stem.upper() for f in tf_dir.glob("*.json")]

    def run(self, max_days: int = 0):
        """Run the full simulation.

        Args:
            max_days: Override config.sim_days (0 = use config).
        """
        days_to_run = max_days or self.config.sim_days
        days_to_run = min(days_to_run, self.timeline.simulatable_days)

        if days_to_run <= 0:
            print("[Sim] No days available to simulate.")
            return

        self.start_time = time.time()
        self.timeline.reset()

        # Skip lookback period
        for _ in range(self.config.lookback_bars):
            if not self.timeline.advance():
                print("[Sim] Not enough data for lookback period.")
                return

        print(f"\n{'='*70}")
        print(f"  SIMULATION START")
        print(f"  Capital: ${self.config.starting_capital:,.0f}")
        print(f"  Days: {days_to_run} | Universe: {self.config.universe_size}/day")
        print(f"  Mode: {'AGENT-DRIVEN' if self.config.use_agents else 'DETERMINISTIC'}")
        print(f"  Data range: {self.timeline.current_date} onwards")
        print(f"{'='*70}\n")

        for day_num in range(days_to_run):
            if not self.timeline.advance():
                print(f"[Sim] Ran out of data after {day_num} days.")
                break

            self._run_day(day_num + 1, days_to_run)
            self.days_run = day_num + 1

            # Checkpoint
            if (day_num + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(day_num + 1)

        self._print_final_report()
        self._save_results()

    def _run_day(self, day_num: int, total_days: int):
        """Execute one simulation day."""
        date = self.timeline.current_date
        self.portfolio.start_day()

        # 1. Resolve existing positions with today's bar
        self._resolve_positions(date)

        # 2. Select universe for today
        universe = select_universe(self.timeline, self.config)

        # 3. Detect patterns and score setups
        setups = self._scan_universe(universe)
        self.total_setups_found += len(setups)

        # 4. Select trades (deterministic or agent-driven)
        if self.config.use_agents and setups:
            selected = self._agent_select(setups, date)
        else:
            selected = self._deterministic_select(setups)

        # 5. Open new positions
        for scored in selected:
            self._open_trade(scored, date)

        # 6. End-of-day snapshot (with mark-to-market prices)
        current_prices = {}
        for pos in self.portfolio.positions:
            bar = self.timeline.get_current_bar(pos.symbol)
            if bar:
                current_prices[pos.symbol] = bar.close
        self.portfolio.end_day(date, current_prices)

        # 7. Print day summary
        if self.config.verbose:
            self._print_day_summary(day_num, total_days, date, universe, setups, selected)

    def _resolve_positions(self, date: str):
        """Check all open positions against today's bar."""
        resolved = []
        for pos in self.portfolio.positions:
            bar = self.timeline.get_current_bar(pos.symbol)
            if bar is None:
                continue

            result = pos.check_bar(bar)
            if result is not None:
                outcome, realized_r = result
                resolved.append((pos, outcome, realized_r))
            elif pos.bars_held >= self.config.default_max_hold_days:
                # Timeout — force close at current price
                outcome, realized_r = pos.timeout_resolve(bar.close)
                resolved.append((pos, outcome, realized_r))

        for pos, outcome, realized_r in resolved:
            self.portfolio.close_position(pos, date, outcome, realized_r)

    def _scan_universe(self, universe: list[str]) -> list[ScoredSetup]:
        """Run pattern detection + scoring on all universe symbols."""
        all_scored: list[ScoredSetup] = []

        # Get SPY regime for scoring context
        spy_regime = self.timeline.get_spy_regime()

        for symbol in universe:
            bar_series = self.timeline.get_bars(symbol)
            if bar_series is None or len(bar_series.bars) < 30:
                continue

            # Detect patterns
            setups = classify_all(bar_series)
            if not setups:
                continue

            # Compute features for scoring
            closes = np.array([b.close for b in bar_series.bars], dtype=np.float64)
            highs = np.array([b.high for b in bar_series.bars], dtype=np.float64)
            lows = np.array([b.low for b in bar_series.bars], dtype=np.float64)
            volumes = np.array([b.volume for b in bar_series.bars], dtype=np.float64)

            features = compute_features(closes, highs, lows, volumes)

            # Detect regime for this symbol
            regime = detect_regime(closes, highs, lows)

            # Score each setup
            for setup in setups:
                scored = score_setup(
                    setup, features, regime,
                    evaluator=self.evaluator,
                    backtest_score=50.0,  # neutral default
                )
                if scored.composite_score >= self.config.min_composite_score:
                    all_scored.append(scored)

        # Sort by composite score
        all_scored.sort(key=lambda x: x.composite_score, reverse=True)
        if self.config.max_setups_per_day > 0:
            return all_scored[:self.config.max_setups_per_day]
        return all_scored

    def _deterministic_select(self, setups: list[ScoredSetup]) -> list[ScoredSetup]:
        """Select top N setups by score (no agent involvement)."""
        if not setups:
            return []

        # Don't open more than portfolio allows
        available_slots = self.config.max_concurrent_positions - len(self.portfolio.positions)
        max_new = min(self.config.max_trades_per_day, available_slots)
        if max_new <= 0:
            return []

        # Avoid duplicate symbols (both existing positions and within new selections)
        seen_symbols = {p.symbol for p in self.portfolio.positions}
        filtered = []
        for s in setups:
            if s.setup.symbol not in seen_symbols:
                filtered.append(s)
                seen_symbols.add(s.setup.symbol)
                if len(filtered) >= max_new:
                    break

        return filtered

    def _agent_select(self, setups: list[ScoredSetup], date: str) -> list[ScoredSetup]:
        """Use AI agents to select trades. Falls back to deterministic if agents unavailable."""
        try:
            if self._coordinator is None:
                from simulation.agents.coordinator import AgentCoordinator
                self._coordinator = AgentCoordinator(self.config)

            import asyncio
            selected = asyncio.run(
                self._coordinator.select_trades(
                    setups=setups,
                    portfolio=self.portfolio,
                    timeline=self.timeline,
                    date=date,
                )
            )
            return selected

        except Exception as e:
            if self.config.verbose:
                print(f"  [Agent] Error: {e} — falling back to deterministic")
            return self._deterministic_select(setups)

    def _open_trade(self, scored: ScoredSetup, date: str):
        """Open a new simulated position from a scored setup."""
        setup = scored.setup

        # Get agent metadata if available
        agent_meta = getattr(scored, "_agent_meta", {})
        size_modifier = agent_meta.get("size_modifier", 1.0)

        # Calculate position size (with agent size modifier)
        shares, dollar_risk = self.portfolio.size_trade(
            setup.entry_price, setup.stop_loss
        )
        if shares <= 0:
            return

        # Apply agent size modifier
        import math
        shares = max(1, math.floor(shares * size_modifier))
        dollar_risk = round(shares * abs(setup.entry_price - setup.stop_loss), 2)

        # Compute ATR for slippage
        bar_series = self.timeline.get_bars(setup.symbol)
        atr = 0.0
        if bar_series and len(bar_series.bars) >= 14:
            from backend.structures.indicators import wilder_atr
            highs = np.array([b.high for b in bar_series.bars[-20:]], dtype=np.float64)
            lows = np.array([b.low for b in bar_series.bars[-20:]], dtype=np.float64)
            closes = np.array([b.close for b in bar_series.bars[-20:]], dtype=np.float64)
            atr_vals = wilder_atr(highs, lows, closes, period=14)
            if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]):
                atr = float(atr_vals[-1])

        # Apply slippage to entry
        slippage = atr * self.config.slippage_atr_pct
        if setup.bias.value == "long":
            entry = setup.entry_price + slippage
        else:
            entry = setup.entry_price - slippage

        pos = SimPosition(
            symbol=setup.symbol,
            pattern_name=setup.pattern_name,
            strategy_type=setup.strategy_type,
            bias=setup.bias.value,
            entry_price=round(entry, 4),
            stop_loss=setup.stop_loss,
            target_1=setup.target_1 if setup.target_1 > 0 else setup.target_price,
            target_2=setup.target_2 if setup.target_2 > 0 else setup.target_price,
            shares=shares,
            dollar_risk=dollar_risk,
            entry_date=date,
            splits=setup.position_splits,
            atr=atr,
            analyst_verdict=agent_meta.get("analyst_verdict", ""),
            pm_reasoning=agent_meta.get("pm_reasoning", ""),
            risk_reasoning=agent_meta.get("risk_reasoning", ""),
        )

        self.portfolio.open_position(pos)

    def _print_day_summary(self, day_num: int, total_days: int,
                           date: str, universe: list[str],
                           setups: list[ScoredSetup],
                           selected: list[ScoredSetup]):
        """Print a one-line progress update."""
        snap = self.portfolio.daily_snapshots[-1] if self.portfolio.daily_snapshots else None
        stats = self.portfolio.get_stats()

        elapsed = time.time() - self.start_time
        pct = day_num / total_days * 100

        day_r_str = f"{snap.day_r:+.2f}R" if snap else "---"
        cum_r_str = f"{snap.cumulative_r:+.2f}R" if snap else "---"
        open_str = f"{len(self.portfolio.positions)}"
        heat_str = f"{snap.heat_pct:.1f}%" if snap else "0%"

        trades_str = ""
        if selected:
            syms = [s.setup.symbol for s in selected]
            trades_str = f" | New: {','.join(syms)}"

        closed_today = snap.trades_closed if snap else 0
        close_str = f" | Closed: {closed_today}" if closed_today else ""

        print(
            f"  Day {day_num:3d}/{total_days} [{pct:5.1f}%] {date} | "
            f"Scanned {len(universe)} -> {len(setups)} setups -> {len(selected)} trades | "
            f"Day: {day_r_str} | Cum: {cum_r_str} | "
            f"Open: {open_str} | Heat: {heat_str}"
            f"{trades_str}{close_str}"
            f"  [{elapsed:.0f}s]"
        )

    def _print_final_report(self):
        """Print summary statistics at end of simulation."""
        stats = self.portfolio.get_stats()
        elapsed = time.time() - self.start_time

        print(f"\n{'='*70}")
        print(f"  SIMULATION COMPLETE — {self.days_run} days in {elapsed:.1f}s")
        print(f"{'='*70}")
        print(f"  Total Trades:      {stats.get('total_trades', 0)}")
        print(f"  Wins / Losses:     {stats.get('wins', 0)} / {stats.get('losses', 0)}")
        print(f"  Win Rate:          {stats.get('win_rate', 0):.1f}%")
        print(f"  Total R:           {stats.get('total_r', 0):+.2f}")
        print(f"  Avg R/Trade:       {stats.get('avg_r', 0):+.3f}")
        print(f"  Profit Factor:     {stats.get('profit_factor', 0):.2f}")
        print(f"  Best Trade:        {stats.get('best_trade_r', 0):+.2f}R")
        print(f"  Worst Trade:       {stats.get('worst_trade_r', 0):+.2f}R")
        print(f"  Open Positions:    {stats.get('open_positions', 0)}")

        # Equity curve stats
        if self.portfolio.daily_snapshots:
            equities = [s.equity for s in self.portfolio.daily_snapshots]
            peak = max(equities)
            current = equities[-1]
            dd = (peak - current) / peak * 100 if peak > 0 else 0
            print(f"  Peak Equity:       ${peak:,.2f}")
            print(f"  Final Equity:      ${current:,.2f}")
            print(f"  Max Drawdown:      {dd:.1f}%")

        print(f"  Setups Scanned:    {self.total_setups_found}")
        print(f"{'='*70}\n")

    def _save_checkpoint(self, day_num: int):
        """Save portfolio state for resume."""
        path = self.config.checkpoint_dir / f"checkpoint_day{day_num:03d}.json"
        self.portfolio.save_checkpoint(path)
        if self.config.verbose:
            print(f"  [Checkpoint] Saved at day {day_num}")

    def _save_results(self):
        """Save final results to disk."""
        results = {
            "config": {
                "starting_capital": self.config.starting_capital,
                "risk_per_trade_pct": self.config.risk_per_trade_pct,
                "sim_days": self.days_run,
                "universe_size": self.config.universe_size,
                "use_agents": self.config.use_agents,
            },
            "stats": self.portfolio.get_stats(),
            "equity_curve": [
                {"date": s.date, "equity": s.equity, "cumulative_r": s.cumulative_r,
                 "day_r": s.day_r, "heat": s.heat_pct}
                for s in self.portfolio.daily_snapshots
            ],
            "closed_trades": [t.to_dict() for t in self.portfolio.closed_trades],
            "open_positions": [p.to_dict() for p in self.portfolio.positions],
        }

        out_path = self.config.output_dir / "simulation_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [Results] Saved to {out_path}")

        # Also save trade log as CSV-like for easy analysis
        trades_path = self.config.output_dir / "trade_log.json"
        trades_path.write_text(json.dumps(
            [t.to_dict() for t in self.portfolio.closed_trades], indent=2
        ))
