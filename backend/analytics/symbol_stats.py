"""
analytics/symbol_stats.py — Per-Symbol Pattern Analytics

When user clicks "📊 Stats" on a chart, this provides:
  1. The specific pattern's performance ON THIS STOCK
  2. Top 5 best-performing patterns on this stock

Data sources (in order):
  1. cache/backtest_by_symbol/{SYMBOL}.json — pre-computed during backtest
  2. On-demand computation if not cached (runs ~5-10 sec)

To pre-compute: next time you run run_backtest.py, it will auto-save per-symbol.
"""
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path("cache/backtest_by_symbol")
BT_RESULTS = Path("cache/backtest_results.json")


@dataclass
class PatternStat:
    name: str
    signals: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win_r: float
    avg_loss_r: float
    edge_score: float

    def to_dict(self) -> dict:
        return {
            "name": self.name, "signals": self.signals,
            "wins": self.wins, "losses": self.losses,
            "win_rate": round(self.win_rate, 1),
            "profit_factor": round(self.profit_factor, 2),
            "expectancy": round(self.expectancy, 4),
            "avg_win_r": round(self.avg_win_r, 2),
            "avg_loss_r": round(self.avg_loss_r, 2),
            "edge_score": round(self.edge_score, 1),
        }


def get_symbol_analytics(symbol: str, pattern_name: str = "") -> dict:
    """
    Get pattern analytics for a specific symbol.

    Returns:
      - highlighted: stats for the requested pattern on this symbol (if any)
      - top_patterns: top 5 best-performing patterns on this symbol
      - total_signals: total signals across all patterns on this symbol
      - source: "cached" or "computed"
    """
    symbol = symbol.upper()

    # Try cached per-symbol data first
    cached = _load_cached(symbol)
    if cached:
        return _format_response(cached, pattern_name, "cached")

    # Try extracting from main backtest results
    extracted = _extract_from_backtest(symbol)
    if extracted:
        return _format_response(extracted, pattern_name, "extracted")

    # On-demand computation
    computed = _compute_on_demand(symbol)
    if computed:
        _save_cached(symbol, computed)
        return _format_response(computed, pattern_name, "computed")

    return {
        "symbol": symbol, "highlighted": None,
        "top_patterns": [], "total_signals": 0,
        "source": "none", "message": "No backtest data available for this symbol",
    }


def _load_cached(symbol: str) -> dict | None:
    path = CACHE_DIR / f"{symbol}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, KeyError):
        return None


def _save_cached(symbol: str, data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data["cached_at"] = time.time()
    (CACHE_DIR / f"{symbol}.json").write_text(json.dumps(data, indent=2))


def _extract_from_backtest(symbol: str) -> dict | None:
    """
    The main backtest_results.json has aggregate stats across all symbols.
    It doesn't have per-symbol breakdowns (yet).
    Return None — the next backtest run will save per-symbol data.
    """
    # Future: if backtest_results.json gets per-symbol data, extract here
    return None


def _compute_on_demand(symbol: str) -> dict | None:
    """
    Run a quick backtest for just this one symbol.
    Uses the same walk-forward logic as run_backtest.py.
    """
    try:
        from backend.data.massive_client import fetch_bars
        from backend.patterns.classifier import classify_all

        print(f"  [Analytics] Computing on-demand for {symbol}...")
        t_start = time.time()

        results: dict[str, dict] = defaultdict(lambda: {
            "wins": 0, "losses": 0, "win_rs": [], "loss_rs": [],
        })

        for tf in ["5min", "15min"]:
            try:
                bars = fetch_bars(symbol, tf, days_back=90)
            except Exception:
                continue

            if len(bars.bars) < 50:
                continue

            # Walk forward: detect at bar N, check outcome on N+1..
            for i in range(50, len(bars.bars) - 10):
                # Create a slice up to bar i
                from backend.data.schemas import BarSeries
                slice_bars = BarSeries(
                    symbol=symbol, timeframe=tf,
                    bars=bars.bars[max(0, i-200):i+1]
                )

                setups = classify_all(slice_bars)
                if not setups:
                    continue

                # Check each setup's outcome
                for setup in setups:
                    # Only count if detected at the last bar of the slice
                    # Use date+hour+minute comparison to avoid microsecond mismatches
                    if (setup.detected_at.replace(second=0, microsecond=0) !=
                            bars.bars[i].timestamp.replace(second=0, microsecond=0)):
                        continue

                    entry = setup.entry_price
                    stop = setup.stop_loss
                    target = setup.target_price
                    is_long = setup.bias.value == "long"
                    risk = abs(entry - stop)
                    if risk <= 0:
                        continue

                    # Walk forward from bar i+1
                    hit_target = False
                    hit_stop = False
                    for j in range(i + 1, min(i + 100, len(bars.bars))):
                        bar = bars.bars[j]
                        if is_long:
                            if bar.low <= stop:
                                hit_stop = True
                                break
                            if bar.high >= target:
                                hit_target = True
                                break
                        else:
                            if bar.high >= stop:
                                hit_stop = True
                                break
                            if bar.low <= target:
                                hit_target = True
                                break

                    pname = setup.pattern_name
                    if hit_target:
                        reward = abs(target - entry) / risk
                        results[pname]["wins"] += 1
                        results[pname]["win_rs"].append(reward)
                    elif hit_stop:
                        results[pname]["losses"] += 1
                        results[pname]["loss_rs"].append(-1.0)

        # Build stats
        patterns = {}
        for name, data in results.items():
            total = data["wins"] + data["losses"]
            if total < 1:
                continue
            wr = data["wins"] / total * 100
            avg_w = sum(data["win_rs"]) / max(len(data["win_rs"]), 1)
            avg_l = abs(sum(data["loss_rs"]) / max(len(data["loss_rs"]), 1))
            gross_win  = avg_w * data["wins"]
            gross_loss = avg_l * data["losses"]
            pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
            exp = (wr / 100 * avg_w) - ((100 - wr) / 100 * avg_l)
            edge = min(100, wr * 0.4 + min(pf, 5) * 10 + min(exp, 1) * 20 + min(total, 20))

            patterns[name] = {
                "signals": total, "wins": data["wins"], "losses": data["losses"],
                "win_rate": wr, "profit_factor": round(pf, 2),
                "expectancy": round(exp, 4),
                "avg_win_r": round(avg_w, 2), "avg_loss_r": round(avg_l, 2),
                "edge_score": round(edge, 1),
            }

        elapsed = time.time() - t_start
        print(f"  [Analytics] {symbol}: {len(patterns)} patterns, {sum(p['signals'] for p in patterns.values())} signals ({elapsed:.1f}s)")

        if not patterns:
            return None

        return {"symbol": symbol, "patterns": patterns}

    except Exception as e:
        print(f"  [Analytics] Error computing {symbol}: {e}")
        return None


def _format_response(data: dict, pattern_name: str, source: str) -> dict:
    """Format the analytics response."""
    patterns = data.get("patterns", {})

    # Build sorted list
    all_stats = []
    for name, stats in patterns.items():
        all_stats.append(PatternStat(
            name=name, signals=stats.get("signals", 0),
            wins=stats.get("wins", 0), losses=stats.get("losses", 0),
            win_rate=stats.get("win_rate", 0), profit_factor=stats.get("profit_factor", 0),
            expectancy=stats.get("expectancy", 0),
            avg_win_r=stats.get("avg_win_r", 0), avg_loss_r=stats.get("avg_loss_r", 0),
            edge_score=stats.get("edge_score", 0),
        ))

    all_stats.sort(key=lambda x: x.edge_score, reverse=True)

    # Highlighted pattern
    highlighted = None
    if pattern_name:
        for s in all_stats:
            if s.name == pattern_name:
                highlighted = s.to_dict()
                break

    # Top 5
    top5 = [s.to_dict() for s in all_stats[:5]]

    return {
        "symbol": data.get("symbol", ""),
        "highlighted": highlighted,
        "top_patterns": top5,
        "total_signals": sum(s.signals for s in all_stats),
        "total_patterns": len(all_stats),
        "source": source,
    }