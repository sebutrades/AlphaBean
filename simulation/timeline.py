"""
simulation/timeline.py — TimelineManager: walks calendar days, windows bars.

Loads all daily bars from the bar store once at init, then provides a
sliding window view: on simulation day D, only bars[0..D] are visible.
No future data leaks.

Also provides SPY regime detection per day for scoring context.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from backend.data.schemas import Bar, BarSeries
from backend.regime.detector import detect_regime, RegimeResult
from simulation.config import SimConfig


class TimelineManager:
    """Manages the day-by-day reveal of historical bar data.

    On init, loads all daily bars for every symbol from disk.
    Each call to `advance()` moves to the next trading day,
    revealing one more bar of data per symbol.
    """

    def __init__(self, config: SimConfig, symbols: list[str]):
        self.config = config
        self._all_bars: dict[str, list[dict]] = {}   # symbol → raw bar dicts
        self._trading_dates: list[str] = []           # all unique dates, sorted
        self._current_day_idx: int = -1               # index into _trading_dates
        self._spy_bars: list[dict] = []

        self._load_all(symbols)

    def _load_all(self, symbols: list[str]):
        """Load all daily bar files from disk."""
        tf_dir = self.config.bar_data_dir / self.config.timeframe
        all_dates: set[str] = set()

        for symbol in symbols:
            fpath = tf_dir / f"{symbol.upper()}.json"
            if not fpath.exists():
                continue
            try:
                data = json.loads(fpath.read_text())
                bars = data.get("bars", [])
                if bars:
                    self._all_bars[symbol.upper()] = bars
                    for b in bars:
                        all_dates.add(b["t"][:10])  # YYYY-MM-DD
            except Exception:
                continue

        # SPY is always loaded for regime detection
        spy_path = tf_dir / "SPY.json"
        if spy_path.exists():
            try:
                self._spy_bars = json.loads(spy_path.read_text()).get("bars", [])
            except Exception:
                pass

        self._trading_dates = sorted(all_dates)

    @property
    def total_trading_days(self) -> int:
        """Total available trading days in the dataset."""
        return len(self._trading_dates)

    @property
    def simulatable_days(self) -> int:
        """Days available after lookback period."""
        return max(0, len(self._trading_dates) - self.config.lookback_bars)

    @property
    def current_date(self) -> Optional[str]:
        """Current simulation date (YYYY-MM-DD)."""
        if 0 <= self._current_day_idx < len(self._trading_dates):
            return self._trading_dates[self._current_day_idx]
        return None

    @property
    def current_day_number(self) -> int:
        """0-based day number within the simulation (after lookback)."""
        return max(0, self._current_day_idx - self.config.lookback_bars + 1)

    @property
    def all_symbols(self) -> list[str]:
        """All symbols with bar data loaded."""
        return list(self._all_bars.keys())

    def reset(self):
        """Reset to before the first simulation day."""
        self._current_day_idx = self.config.lookback_bars - 1

    def advance(self) -> bool:
        """Move to the next trading day. Returns False if no more days."""
        next_idx = self._current_day_idx + 1
        if next_idx >= len(self._trading_dates):
            return False
        self._current_day_idx = next_idx
        return True

    def get_bars(self, symbol: str, max_bars: Optional[int] = None) -> Optional[BarSeries]:
        """Get bars for a symbol up to (and including) the current day.

        Only returns bars that have been 'revealed' — no future data.
        """
        raw = self._all_bars.get(symbol.upper())
        if not raw:
            return None

        current_date = self.current_date
        if not current_date:
            return None

        # Filter bars up to current date (inclusive)
        visible = [b for b in raw if b["t"][:10] <= current_date]
        if not visible:
            return None

        if max_bars and len(visible) > max_bars:
            visible = visible[-max_bars:]

        bars = []
        for d in visible:
            bars.append(Bar(
                symbol=symbol.upper(),
                timestamp=datetime.strptime(d["t"], "%Y-%m-%dT%H:%M:%S"),
                open=d["o"], high=d["h"], low=d["l"], close=d["c"],
                volume=int(d["v"]),
                vwap=d.get("vw"),
                trade_count=d.get("n"),
            ))

        return BarSeries(symbol=symbol.upper(), timeframe=self.config.timeframe, bars=bars)

    def get_current_bar(self, symbol: str) -> Optional[Bar]:
        """Get just the current day's bar for a symbol (for trade resolution)."""
        raw = self._all_bars.get(symbol.upper())
        if not raw:
            return None

        current_date = self.current_date
        if not current_date:
            return None

        for b in raw:
            if b["t"][:10] == current_date:
                return Bar(
                    symbol=symbol.upper(),
                    timestamp=datetime.strptime(b["t"], "%Y-%m-%dT%H:%M:%S"),
                    open=b["o"], high=b["h"], low=b["l"], close=b["c"],
                    volume=int(b["v"]),
                    vwap=b.get("vw"),
                    trade_count=b.get("n"),
                )
        return None

    def get_spy_regime(self) -> Optional[RegimeResult]:
        """Detect market regime from SPY bars up to current date."""
        current_date = self.current_date
        if not current_date or not self._spy_bars:
            return None

        visible = [b for b in self._spy_bars if b["t"][:10] <= current_date]
        if len(visible) < 30:
            return None

        closes = np.array([b["c"] for b in visible], dtype=np.float64)
        highs = np.array([b["h"] for b in visible], dtype=np.float64)
        lows = np.array([b["l"] for b in visible], dtype=np.float64)

        return detect_regime(closes, highs, lows, is_spy=True)

    def get_price_on_date(self, symbol: str, date_str: str) -> Optional[float]:
        """Get closing price for a symbol on a specific date."""
        raw = self._all_bars.get(symbol.upper())
        if not raw:
            return None
        for b in raw:
            if b["t"][:10] == date_str:
                return b["c"]
        return None

    def get_dollar_volume(self, symbol: str, lookback: int = 20) -> float:
        """Average daily dollar volume over the lookback period ending at current date."""
        raw = self._all_bars.get(symbol.upper())
        if not raw:
            return 0.0

        current_date = self.current_date
        if not current_date:
            return 0.0

        visible = [b for b in raw if b["t"][:10] <= current_date]
        recent = visible[-lookback:] if len(visible) > lookback else visible
        if not recent:
            return 0.0

        dvols = [b["c"] * b["v"] for b in recent]
        return sum(dvols) / len(dvols)
