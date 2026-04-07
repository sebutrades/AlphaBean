"""
simulation/config.py — All simulation parameters in one place.

Covers portfolio rules, agent model selection, scan/detection settings,
trade execution, and output paths.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SimConfig:
    """Central configuration for a simulation run."""

    # ── Portfolio ─────────────────────────────────────────────────────────
    starting_capital: float = 100_000.0
    risk_per_trade_pct: float = 1.0       # % of capital risked per trade
    max_portfolio_heat_pct: float = 6.0   # max total open risk
    max_concurrent_positions: int = 10
    max_correlated_positions: int = 3     # same sector / closely correlated

    # ── Simulation scope ──────────────────────────────────────────────────
    sim_days: int = 180                   # trading days to simulate
    lookback_bars: int = 60               # bars visible before first sim day
    universe_size: int = 0                # 0 = scan ALL cached symbols
    timeframe: str = "1d"                 # primary simulation timeframe
    bar_data_dir: Path = Path("live_data_cache/data")

    # ── Detection & scoring ──────────────────────────────────────────────
    min_composite_score: float = 45.0     # minimum score to consider
    max_setups_per_day: int = 0           # 0 = no cap (find everything)
    max_trades_per_day: int = 5           # PM can select at most this many

    # ── Trade execution ──────────────────────────────────────────────────
    slippage_atr_pct: float = 0.05        # 5% of ATR adverse slippage
    transaction_cost_r: float = 0.02      # ~$5-10 per trade on $500 risk
    max_r_per_trade: float = 10.0         # R-cap to prevent outlier noise
    default_max_hold_days: int = 10       # timeout if no exit

    # ── Agent models ─────────────────────────────────────────────────────
    analyst_bulk_model: str = "qwen3:8b"          # local Ollama — free
    analyst_top_model: str = "claude-haiku-4-5-20251001"
    pm_model: str = "claude-haiku-4-5-20251001"
    risk_model: str = "claude-haiku-4-5-20251001"
    strategist_model: str = "claude-sonnet-4-6-20250514"
    researcher_model: str = "claude-sonnet-4-6-20250514"

    analyst_top_n: int = 5                # top N setups get Haiku eval
    use_agents: bool = True               # False = deterministic mode

    # ── Agent concurrency ────────────────────────────────────────────────
    max_concurrent_api_calls: int = 5
    ollama_timeout: int = 120             # seconds
    anthropic_timeout: int = 60           # seconds

    # ── Knowledge ────────────────────────────────────────────────────────
    knowledge_dir: Path = Path("simulation/knowledge/data")
    researcher_interval: int = 5          # run researcher every N days

    # ── Output ───────────────────────────────────────────────────────────
    output_dir: Path = Path("simulation/output")
    checkpoint_dir: Path = Path("simulation/output/checkpoints")
    reports_dir: Path = Path("simulation/output/reports")
    checkpoint_interval: int = 5          # save state every N days
    verbose: bool = True

    def ensure_dirs(self):
        """Create all output directories."""
        for d in (self.output_dir, self.checkpoint_dir,
                  self.reports_dir, self.knowledge_dir):
            d.mkdir(parents=True, exist_ok=True)
