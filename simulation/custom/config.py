"""
simulation/custom/config.py — Configuration for Custom Agent Trading.

Everything the user can tweak from the UI: strategies, models, sizing,
deliberation depth, risk parameters, and output settings.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AgentModelConfig:
    """Model selection per agent role."""
    analyst: str = "qwen3:8b"               # local Ollama — free
    portfolio_manager: str = "qwen3:8b"      # local Ollama — free
    risk_manager: str = "qwen3:8b"           # local Ollama — free
    strategist: str = "qwen3:8b"             # upgrade to claude-sonnet when credits available

    def to_dict(self) -> dict:
        return {
            "analyst": self.analyst,
            "portfolio_manager": self.portfolio_manager,
            "risk_manager": self.risk_manager,
            "strategist": self.strategist,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SizingConfig:
    """Position sizing configuration."""
    mode: str = "compound"  # "fixed", "compound", "adaptive"

    base_risk_pct: float = 1.0  # base % of equity risked per trade

    # Compound mode: risk is % of CURRENT equity, not starting
    # Adaptive mode: additionally scales by strategy win rate
    strategy_multiplier_min: float = 0.5   # worst strategy gets this
    strategy_multiplier_max: float = 2.0   # best strategy gets this
    min_trades_for_adjustment: int = 10    # need N closed trades before adjusting

    # Drawdown protection
    drawdown_reduction: bool = True
    drawdown_threshold_pct: float = 5.0    # reduce size after X% drawdown
    drawdown_scale: float = 0.5            # multiply size by this during drawdown

    # Hard caps
    max_position_pct: float = 15.0         # max % of equity in one position
    max_heat_pct: float = 6.0              # max total open risk
    max_positions: int = 10

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "SizingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DeliberationConfig:
    """How agents discuss and decide."""
    mode: str = "standard"  # "quick", "standard", "thorough"

    # Quick: 1 analyst call → PM selects → Risk reviews (~3-7 API calls)
    # Standard: 2 analyst calls (initial + senior review) → PM → Risk (~7-12 calls)
    # Thorough: 3 analysts vote → PM deliberates → Risk detailed review (~12-20 calls)

    max_candidates: int = 5            # max setups to evaluate per scan
    max_api_calls_per_candidate: int = 10  # hard cap on calls per candidate
    analyst_count: int = 1             # 1 for quick/standard, 3 for thorough
    include_senior_review: bool = True  # second analyst pass (standard+)
    include_strategist: bool = False    # strategist tiebreaker (thorough only)

    # Temperature controls
    analyst_temperature: float = 0.3
    pm_temperature: float = 0.2
    risk_temperature: float = 0.1

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "DeliberationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_mode(cls, mode: str) -> "DeliberationConfig":
        """Create config from a preset mode name."""
        if mode == "quick":
            return cls(
                mode="quick",
                analyst_count=1,
                include_senior_review=False,
                include_strategist=False,
                max_api_calls_per_candidate=3,
            )
        elif mode == "thorough":
            return cls(
                mode="thorough",
                analyst_count=3,
                include_senior_review=True,
                include_strategist=True,
                max_api_calls_per_candidate=10,
            )
        else:  # standard
            return cls(
                mode="standard",
                analyst_count=1,
                include_senior_review=True,
                include_strategist=False,
                max_api_calls_per_candidate=6,
            )


@dataclass
class CustomSimConfig:
    """Full configuration for a custom agent trading run."""

    # ── Identity ─────────────────────────────────────────────────────────
    name: str = ""                        # user-given run name
    description: str = ""                 # optional notes

    # ── Capital & Risk ───────────────────────────────────────────────────
    starting_capital: float = 100_000.0
    sizing: SizingConfig = field(default_factory=SizingConfig)

    # ── Strategy Filtering ───────────────────────────────────────────────
    allowed_strategies: list[str] = field(default_factory=list)  # empty = all
    min_composite_score: float = 50.0
    max_trades_per_scan: int = 1          # max trades opened per 5min scan
    max_trades_per_day: int = 5

    # ── Agent Configuration ──────────────────────────────────────────────
    models: AgentModelConfig = field(default_factory=AgentModelConfig)
    deliberation: DeliberationConfig = field(default_factory=DeliberationConfig)
    use_agents: bool = True

    # ── Simulation Scope ─────────────────────────────────────────────────
    dates: list[str] = field(default_factory=list)  # empty = all available
    timeframe: str = "5min"
    close_eod: bool = True                # close all positions end of day
    playback_speed: float = 10.0

    # ── Trade Mechanics ──────────────────────────────────────────────────
    slippage_pct: float = 0.01            # % of risk as adverse slippage
    transaction_cost_r: float = 0.02      # R deducted per trade
    rate_limit_minutes: int = 5           # min minutes between trades

    # ── Output ───────────────────────────────────────────────────────────
    output_dir: Path = Path("simulation/output/custom_runs")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "starting_capital": self.starting_capital,
            "sizing": self.sizing.to_dict(),
            "allowed_strategies": self.allowed_strategies,
            "min_composite_score": self.min_composite_score,
            "max_trades_per_scan": self.max_trades_per_scan,
            "max_trades_per_day": self.max_trades_per_day,
            "models": self.models.to_dict(),
            "deliberation": self.deliberation.to_dict(),
            "use_agents": self.use_agents,
            "dates": self.dates,
            "timeframe": self.timeframe,
            "close_eod": self.close_eod,
            "playback_speed": self.playback_speed,
            "slippage_pct": self.slippage_pct,
            "transaction_cost_r": self.transaction_cost_r,
            "rate_limit_minutes": self.rate_limit_minutes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CustomSimConfig":
        sizing = SizingConfig.from_dict(d.pop("sizing", {})) if "sizing" in d else SizingConfig()
        models = AgentModelConfig.from_dict(d.pop("models", {})) if "models" in d else AgentModelConfig()
        delib = DeliberationConfig.from_dict(d.pop("deliberation", {})) if "deliberation" in d else DeliberationConfig()
        # Filter to known fields
        known = {k for k in cls.__dataclass_fields__} - {"sizing", "models", "deliberation", "output_dir"}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(sizing=sizing, models=models, deliberation=delib, **filtered)
