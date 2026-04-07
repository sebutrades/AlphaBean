"""
simulation/universe.py — Historical symbol universe selector.

For each simulation day, selects the top N symbols by dollar volume
from the available bar data, using only bars visible up to that date.
This mimics a "trending tickers" screen that would be available live.
"""
from simulation.config import SimConfig
from simulation.timeline import TimelineManager


def select_universe(timeline: TimelineManager, config: SimConfig) -> list[str]:
    """Select top symbols by average dollar volume for the current sim day.

    Uses the 20-day trailing average dollar volume (only from visible bars).
    Excludes ETFs from the main universe (SPY/QQQ kept for regime only).
    """
    ETF_EXCLUDE = {
        "SPY", "QQQ", "IWM", "DIA", "ACWI", "VTI", "VOO", "XLF", "XLE",
        "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLC", "XLRE", "XLB",
        "GLD", "SLV", "TLT", "HYG", "LQD", "EEM", "EFA", "VEA", "VWO",
        "ARKK", "ARKG", "ARKW", "ARKF", "SQQQ", "TQQQ", "SPXU", "UPRO",
        "UVXY", "VXX", "SOXL", "SOXS", "LABU", "LABD", "FAS", "FAZ",
    }

    symbols = []
    for symbol in timeline.all_symbols:
        if symbol in ETF_EXCLUDE:
            continue
        # Only include symbols that have a bar on the current date
        bar = timeline.get_current_bar(symbol)
        if bar is not None:
            symbols.append(symbol)

    # If universe_size is 0, return ALL symbols (scan everything)
    if config.universe_size <= 0:
        return symbols

    # Otherwise rank by dollar volume and take top N
    scored = []
    for symbol in symbols:
        dvol = timeline.get_dollar_volume(symbol, lookback=20)
        scored.append((symbol, dvol))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:config.universe_size]]
