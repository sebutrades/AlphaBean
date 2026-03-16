"""
schemas.py — Universal data models for AlphaBean.
Updated in v3 with NumPy array accessors for vectorized computation.
"""
from pydantic import BaseModel
from datetime import datetime
import numpy as np


class Bar(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    trade_count: int | None = None


class BarSeries(BaseModel):
    symbol: str
    timeframe: str
    bars: list[Bar]

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    @property
    def highs(self) -> list[float]:
        return [b.high for b in self.bars]

    @property
    def lows(self) -> list[float]:
        return [b.low for b in self.bars]

    @property
    def opens(self) -> list[float]:
        return [b.open for b in self.bars]

    @property
    def volumes(self) -> list[int]:
        return [b.volume for b in self.bars]

    @property
    def vwaps(self) -> list[float | None]:
        return [b.vwap for b in self.bars]

    @property
    def timestamps(self) -> list[datetime]:
        return [b.timestamp for b in self.bars]

    # --- NumPy array accessors (vectorized computation) ---

    def closes_array(self) -> np.ndarray:
        return np.array(self.closes, dtype=np.float64)

    def highs_array(self) -> np.ndarray:
        return np.array(self.highs, dtype=np.float64)

    def lows_array(self) -> np.ndarray:
        return np.array(self.lows, dtype=np.float64)

    def opens_array(self) -> np.ndarray:
        return np.array(self.opens, dtype=np.float64)

    def volumes_array(self) -> np.ndarray:
        return np.array(self.volumes, dtype=np.float64)