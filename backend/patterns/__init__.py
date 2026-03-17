"""patterns/ — Structure-first pattern classification."""
from backend.patterns.registry import (
    TradeSetup, Bias, PatternCategory, PATTERN_META, get_all_pattern_names,
)
from backend.patterns.classifier import classify_all, extract_structures