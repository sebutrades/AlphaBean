"""
scanner/engine.py — The orchestrator. Connects all 6 phases into one scan.

Pipeline per symbol:
  1. Fetch 5min + 15min bars from Massive.com
  2. Extract structural primitives (swings, trendlines, S/R)
  3. Compute 8 statistical features (vectorized)
  4. Run 47 pattern classifiers
  5. Detect market regime
  6. Score everything with multi-factor system
  7. Merge multi-TF detections
  8. Return ranked ScoredSetups

Modes:
  "today"  — Only setups from today's session
  "active" — Setups from last 7 days still valid
"""
import json
from collections import defaultdict
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Optional

import numpy as np

from backend.data.massive_client import fetch_bars, SCANNER_TIMEFRAMES
from backend.data.schemas import BarSeries
from backend.patterns.classifier import classify_all
from backend.patterns.registry import TradeSetup
from backend.features.engine import compute_features, FeatureResult
from backend.regime.detector import detect_regime, RegimeResult
from backend.scoring.multi_factor import score_setup, score_setups_batch, ScoredSetup
from backend.strategies.evaluator import StrategyEvaluator

BACKTEST_CACHE = Path("cache/backtest_results.json")


def scan_symbol(
    symbol: str,
    mode: str = "today",
    evaluator: Optional[StrategyEvaluator] = None,
    spy_closes: Optional[np.ndarray] = None,
) -> list[dict]:
    """
    Full scan of a symbol on BOTH 5min and 15min simultaneously.

    Returns list of scored setup dicts, sorted by composite score.
    """
    days_back = 3 if mode == "today" else 10
    all_scored: list[ScoredSetup] = []

    # Track multi-TF merges: (pattern_name, bias, price_bucket) → list of scored
    tf_hits = defaultdict(list)

    # Load backtest scores if available
    bt_scores = _load_backtest_scores()

    for tf in SCANNER_TIMEFRAMES:
        try:
            bars = fetch_bars(symbol, timeframe=tf, days_back=days_back)
        except Exception as e:
            print(f"  [{symbol}] {tf} FETCH ERROR: {e}")
            continue

        if len(bars.bars) < 20:
            print(f"  [{symbol}] {tf} SKIP ({len(bars.bars)} bars)")
            continue

        print(f"  [{symbol}] {tf} OK — {len(bars.bars)} bars")

        # --- Phase 2: Features ---
        closes = np.array([b.close for b in bars.bars], dtype=np.float64)
        highs = np.array([b.high for b in bars.bars], dtype=np.float64)
        lows = np.array([b.low for b in bars.bars], dtype=np.float64)
        volumes = np.array([b.volume for b in bars.bars], dtype=np.float64)
        features = compute_features(closes, highs, lows, volumes, spy_closes)

        # --- Phase 3: Regime ---
        regime = detect_regime(closes, highs, lows, is_spy=False)

        # --- Phase 4: Patterns ---
        setups = classify_all(bars)

        if not setups:
            print(f"  [{symbol}] {tf} — 0 patterns")
            continue

        # Apply mode filter
        if mode == "today":
            setups = [s for s in setups if _is_today(s.detected_at)]

        print(f"  [{symbol}] {tf} — {len(setups)} patterns after filter")

        # --- Phase 6: Score ---
        for setup in setups:
            # Default 30: no-evidence patterns are penalized until backtest validates them
            bt = bt_scores.get(setup.pattern_name, 30.0)
            scored = score_setup(setup, features, regime, evaluator, bt)
            scored.setup.timeframe_detected = _tf_label(tf)

            # Track for multi-TF merge
            price_bucket = round(setup.entry_price, -1 if setup.entry_price > 50 else 0)
            merge_key = (setup.pattern_name, setup.bias.value, price_bucket)
            tf_hits[merge_key].append((tf, scored))

    # --- Merge multi-TF detections ---
    for key, hits in tf_hits.items():
        tfs = sorted(set(tf for tf, _ in hits))
        if len(tfs) > 1:
            # Same pattern on both TFs — use best scoring one, mark as multi-TF
            best = max(hits, key=lambda x: x[1].composite_score)[1]
            best.setup.timeframe_detected = "5m & 15m"
            best.setup.multi_tf = True
            # Boost composite by 5 points for multi-TF confirmation
            best.composite_score = min(100, best.composite_score + 5.0)
            best.setup.confidence = min(0.95, best.setup.confidence + 0.10)
            best.setup.description = f"[5m & 15m] {best.setup.description}"
            all_scored.append(best)
        else:
            best = max(hits, key=lambda x: x[1].composite_score)[1]
            best.setup.timeframe_detected = _tf_label(tfs[0])
            all_scored.append(best)

    # Sort by composite score
    all_scored.sort(key=lambda s: s.composite_score, reverse=True)

    return [s.to_dict() for s in all_scored]


def scan_multiple(
    symbols: list[str],
    mode: str = "today",
    evaluator: Optional[StrategyEvaluator] = None,
) -> list[dict]:
    """Scan multiple symbols, return merged results."""
    all_results = []
    total = len(symbols)
    for i, sym in enumerate(symbols):
        print(f"\n[{i+1}/{total}] Scanning {sym}...")
        results = scan_symbol(sym, mode=mode, evaluator=evaluator)
        all_results.extend(results)

    all_results.sort(key=lambda s: s.get("composite_score", 0), reverse=True)
    return all_results


# ==============================================================================
# HELPERS
# ==============================================================================

def _is_today(ts: datetime) -> bool:
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    if ts.date() == today:
        return True
    if ts.date() == yesterday and ts.time() >= time(15, 30):
        return True
    return False


def _tf_label(tf: str) -> str:
    return {"5min": "5m", "15min": "15m", "1h": "1h", "1d": "1d"}.get(tf, tf)


def _load_backtest_scores() -> dict[str, float]:
    """Load pattern edge scores from backtest cache."""
    if not BACKTEST_CACHE.exists():
        return {}
    try:
        data = json.loads(BACKTEST_CACHE.read_text())
        patterns = data.get("patterns", {})
        scores = {}
        for name, stats in patterns.items():
            if isinstance(stats, dict):
                # Flat format: {"edge_score": 72.3, ...}
                if "edge_score" in stats:
                    scores[name] = float(stats["edge_score"])
                else:
                    # Nested TF format: {"5min": {"edge_score": ...}, ...}
                    tf_scores = [
                        s.get("edge_score", 50)
                        for s in stats.values()
                        if isinstance(s, dict) and "edge_score" in s
                    ]
                    if tf_scores:
                        scores[name] = float(np.mean(tf_scores))
        return scores
    except (json.JSONDecodeError, KeyError):
        return {}