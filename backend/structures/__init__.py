"""structures/ — Structural primitives for price analysis."""
from backend.structures.swings import (
    SwingPoint, SwingType, zigzag, adaptive_zigzag_threshold,
    find_swing_highs, find_swing_lows, adaptive_order,
    swing_highs_from_zigzag, swing_lows_from_zigzag,
)
from backend.structures.trendlines import (
    Trendline, Channel, fit_trendline, fit_trendline_from_arrays,
    detect_channel, is_converging, is_flat_line, slopes_same_sign, compression_ratio,
)
from backend.structures.support_resistance import (
    PriceLevel, BreakoutSignal, cluster_levels, find_horizontal_levels,
    detect_breakouts, nearest_level, neckline_from_swings,
)
from backend.structures.indicators import (
    true_range, wilder_atr, atr_ratio, sma, ema, ema_last,
)