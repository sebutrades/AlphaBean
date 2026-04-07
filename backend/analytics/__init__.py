"""analytics/ — Per-symbol pattern performance analytics and trade performance dashboard."""
from backend.analytics.symbol_stats import get_symbol_analytics
from backend.analytics.performance import (
    get_performance_summary,
    get_equity_curve,
    get_pattern_attribution,
    get_drawdown_series,
    get_time_of_day_stats,
    get_regime_stats,
    get_daily_pnl,
    get_weekly_pnl,
    get_monthly_pnl,
    get_streaks,
)