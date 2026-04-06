"""
live_data_cache — Continuous bar storage and intraday setup tracking

Public surface:

  from live_data_cache import (
      # Bar store
      get_bars, append_bars, get_last_timestamp, needs_backfill, get_store_stats,
      # Bar updater
      update_timeframe, update_hot_list, backfill_symbol, ensure_hot_list_ready,
      # Intraday setup tracker
      process_new_bars, get_open_setups, get_closed_setups,
      flag_overnight_holds, generate_daily_summary,
      # Watchlist
      get_hot_list, get_all_tracked, get_inplay_symbols,
  )
"""

from live_data_cache.bar_store import (
    get_bars,
    append_bars,
    get_last_timestamp,
    needs_backfill,
    get_store_stats,
)

from live_data_cache.bar_updater import (
    update_timeframe,
    update_hot_list,
    backfill_symbol,
    ensure_hot_list_ready,
    get_health,
)

from live_data_cache.intraday_setup_tracker import (
    process_new_bars,
    get_open_setups,
    get_closed_setups,
    flag_overnight_holds,
    generate_daily_summary,
)

from live_data_cache.watchlist import (
    get_hot_list,
    get_all_tracked,
    get_inplay_symbols,
)

__all__ = [
    "get_bars", "append_bars", "get_last_timestamp", "needs_backfill", "get_store_stats",
    "update_timeframe", "update_hot_list", "backfill_symbol", "ensure_hot_list_ready", "get_health",
    "process_new_bars", "get_open_setups", "get_closed_setups",
    "flag_overnight_holds", "generate_daily_summary",
    "get_hot_list", "get_all_tracked", "get_inplay_symbols",
]
