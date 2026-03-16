"""
schemas.py — Universal data models for AlphaBean.
"""
from pydantic import BaseModel
from datetime import datetime


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