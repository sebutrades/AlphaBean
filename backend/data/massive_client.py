"""
massive_client.py -- DEPRECATED: now re-exports from karb_client.py

All bar data is fetched from Karb (localhost:4269) instead of Massive.com.
This file exists only for backward compatibility with existing imports.
"""
from backend.data.karb_client import (      # noqa: F401
    fetch_bars,
    fetch_bars_since,
    fetch_chart_bars,
    get_client,
    ALL_TIMEFRAMES,
    SCANNER_TIMEFRAMES,
)
