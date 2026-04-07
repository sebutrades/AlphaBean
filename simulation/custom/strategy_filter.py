"""
simulation/custom/strategy_filter.py — Strategy filtering.

Provides the full list of available strategies (51 patterns) and
filters classify_all() results to only allowed strategies.
"""
from backend.patterns.registry import PATTERN_META
from backend.patterns.classifier import classify_all
from backend.data.schemas import BarSeries


def get_all_strategies() -> list[dict]:
    """Return all available strategy patterns with metadata.

    Used by the frontend to populate strategy selection checkboxes.
    """
    strategies = []
    for name, meta in sorted(PATTERN_META.items()):
        # Determine category from pattern name
        if "Long" in name:
            direction = "long"
        elif "Short" in name:
            direction = "short"
        else:
            direction = "both"

        category = _categorize_pattern(name)

        strategies.append({
            "name": name,
            "direction": direction,
            "category": category,
            "strategy_type": meta.get("strategy_type", "unknown"),
            "timeframes": meta.get("timeframes", []),
        })
    return strategies


def _categorize_pattern(name: str) -> str:
    """Assign a human-readable category to a pattern."""
    name_lower = name.lower()
    if any(k in name_lower for k in ["breakout", "expansion", "squeeze", "donchian", "turtle", "keltner"]):
        return "breakout"
    if any(k in name_lower for k in ["reversal", "reversion", "divergence", "fade"]):
        return "mean_reversion"
    if any(k in name_lower for k in ["momentum", "trend", "drive", "persistence", "flag", "wave"]):
        return "momentum"
    if any(k in name_lower for k in ["volume", "climax", "vp ", "vwap"]):
        return "volume"
    if any(k in name_lower for k in ["gap", "opening", "power hour", "midday"]):
        return "intraday"
    if any(k in name_lower for k in ["accumulation", "distribution"]):
        return "institutional"
    return "other"


def classify_filtered(bar_series: BarSeries, allowed: list[str]) -> list:
    """Run classify_all() then filter to only allowed strategy names.

    If allowed is empty, returns all setups (no filter).
    """
    setups = classify_all(bar_series)
    if not allowed:
        return setups
    allowed_set = set(allowed)
    return [s for s in setups if s.pattern_name in allowed_set]


def get_strategy_groups() -> dict[str, list[str]]:
    """Return strategies grouped by category for UI display."""
    groups: dict[str, list[str]] = {}
    for s in get_all_strategies():
        cat = s["category"]
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(s["name"])
    return groups
