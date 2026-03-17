"""regime/ — Market regime classification."""
from backend.regime.detector import (
    MarketRegime, StrategyType, RegimeResult,
    detect_regime, get_regime_alignment, best_strategy_types,
    REGIME_ALIGNMENT,
)