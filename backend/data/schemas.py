"""
schemas.py — The universal data models for EdgeFinder.

Every bar of price data in the entire system flows through these models.
When Massive.com sends us data, we normalize it into a Bar.
When a pattern detector needs prices, it receives a BarSeries.
"""
from pydantic import BaseModel
from datetime import datetime


class Bar(BaseModel):
    """
    A single OHLCV bar — one candle on a chart.
    
    O = Open price (where the candle started)
    H = High price (highest point during this period)
    L = Low price (lowest point during this period)  
    C = Close price (where the candle ended)
    V = Volume (how many shares traded)
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None       # Volume-Weighted Average Price
    trade_count: int | None = None   # Number of individual trades


class BarSeries(BaseModel):
    """
    A time-ordered series of bars for one symbol.
    This is what pattern detectors receive as input.
    """
    symbol: str
    timeframe: str   # "1min", "5min", "15min", "1h", "1d"
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