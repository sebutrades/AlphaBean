"""
backend/tracker/trade_tracker.py — Active Trade Tracker

Manages a persistent store of detected trade setups with live status tracking.
Trades are auto-detected by the scanner or manually added via the UI.

STATUS LIFECYCLE:
  PENDING   → Setup detected, price hasn't reached entry yet
  ACTIVE    → Entry hit, trade is live
  AT_T1     → First target hit, partial exit taken, stop moved to breakeven
  AT_T2     → Second target hit, trailing remainder
  TRAILING  → Past T2, riding the trail
  STOPPED   → Stop hit (loss or breakeven after T1)
  CLOSED    → Manually closed or fully exited
  EXPIRED   → Max hold time exceeded without resolution

USAGE:
  from backend.tracker.trade_tracker import TradeTracker
  
  tracker = TradeTracker()
  tracker.scan_and_add(top_n=50)        # Auto-scan top setups
  tracker.refresh_prices()               # Update all with live prices
  trades = tracker.get_active_trades()   # Get for frontend
  tracker.add_manual(setup_dict)         # Manual add from UI
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

TRADES_FILE = Path("cache/active_trades.json")
ARCHIVE_FILE = Path("cache/archived_trades.json")
SYMBOLS_CACHE = Path("cache/top_symbols.json")

# How many days back to look for setups on initial scan
LOOKBACK_DAYS = 30

# Max active trades (auto-added — manual has no limit)
MAX_AUTO_TRADES = 50


class TrackedTrade:
    """A single tracked trade with live status."""

    def __init__(self, data: dict):
        # Setup info (immutable after creation)
        self.id = data.get("id", "")
        self.symbol = data.get("symbol", "")
        self.pattern_name = data.get("pattern_name", "")
        self.bias = data.get("bias", "long")
        self.timeframe = data.get("timeframe", "1d")
        self.entry_price = data.get("entry_price", 0.0)
        self.stop_loss = data.get("stop_loss", 0.0)
        self.target_1 = data.get("target_1", 0.0)
        self.target_2 = data.get("target_2", 0.0)
        self.trail_type = data.get("trail_type", "atr")
        self.trail_param = data.get("trail_param", 2.0)
        self.confidence = data.get("confidence", 0.0)
        self.description = data.get("description", "")
        self.detected_at = data.get("detected_at", "")
        self.position_splits = data.get("position_splits", [0.5, 0.3, 0.2])
        self.source = data.get("source", "auto")  # "auto" or "manual"
        self.key_levels = data.get("key_levels", {})

        # Live state (updated on refresh)
        self.status = data.get("status", "PENDING")
        self.current_price = data.get("current_price", 0.0)
        self.current_atr = data.get("current_atr", 0.0)
        self.unrealized_r = data.get("unrealized_r", 0.0)
        self.unrealized_pnl_pct = data.get("unrealized_pnl_pct", 0.0)
        self.peak_r = data.get("peak_r", 0.0)
        self.trough_r = data.get("trough_r", 0.0)
        self.t1_hit = data.get("t1_hit", False)
        self.t2_hit = data.get("t2_hit", False)
        self.trailing_stop = data.get("trailing_stop", 0.0)
        self.bars_held = data.get("bars_held", 0)
        self.last_updated = data.get("last_updated", "")
        self.entered_at = data.get("entered_at", "")
        self.closed_at = data.get("closed_at", "")
        self.realized_r = data.get("realized_r", 0.0)
        self.notes = data.get("notes", "")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "pattern_name": self.pattern_name,
            "bias": self.bias,
            "timeframe": self.timeframe,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "trail_type": self.trail_type,
            "trail_param": self.trail_param,
            "confidence": self.confidence,
            "description": self.description,
            "detected_at": self.detected_at,
            "position_splits": self.position_splits,
            "source": self.source,
            "key_levels": self.key_levels,
            "status": self.status,
            "current_price": self.current_price,
            "current_atr": self.current_atr,
            "unrealized_r": self.unrealized_r,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "peak_r": self.peak_r,
            "trough_r": self.trough_r,
            "t1_hit": self.t1_hit,
            "t2_hit": self.t2_hit,
            "trailing_stop": self.trailing_stop,
            "bars_held": self.bars_held,
            "last_updated": self.last_updated,
            "entered_at": self.entered_at,
            "closed_at": self.closed_at,
            "realized_r": self.realized_r,
            "notes": self.notes,
        }

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def is_long(self) -> bool:
        return self.bias == "long"

    @property
    def is_active(self) -> bool:
        return self.status in ("PENDING", "ACTIVE", "AT_T1", "AT_T2", "TRAILING")

    @property
    def is_closed(self) -> bool:
        return self.status in ("STOPPED", "CLOSED", "EXPIRED")

    def update_with_price(self, price: float, atr: float = 0.0):
        """Update trade status based on current price."""
        self.current_price = price
        self.current_atr = atr
        self.last_updated = datetime.now().isoformat()

        if self.risk <= 0:
            return

        # Compute current R
        if self.is_long:
            self.unrealized_r = round((price - self.entry_price) / self.risk, 3)
        else:
            self.unrealized_r = round((self.entry_price - price) / self.risk, 3)

        self.unrealized_pnl_pct = round(
            (price - self.entry_price) / self.entry_price * 100
            * (1 if self.is_long else -1), 2
        )

        # Track peak and trough
        self.peak_r = max(self.peak_r, self.unrealized_r)
        self.trough_r = min(self.trough_r, self.unrealized_r)

        # Status transitions
        if self.status == "PENDING":
            # Check if entry price reached
            if self.is_long and price >= self.entry_price:
                self.status = "ACTIVE"
                self.entered_at = self.last_updated
            elif not self.is_long and price <= self.entry_price:
                self.status = "ACTIVE"
                self.entered_at = self.last_updated
            # Check if stop hit before entry (invalidated)
            elif self.is_long and price <= self.stop_loss:
                self.status = "EXPIRED"
                self.closed_at = self.last_updated
            elif not self.is_long and price >= self.stop_loss:
                self.status = "EXPIRED"
                self.closed_at = self.last_updated

        elif self.status == "ACTIVE":
            # Check stop
            if self.is_long and price <= self.stop_loss:
                self.status = "STOPPED"
                self.realized_r = round((self.stop_loss - self.entry_price) / self.risk, 3)
                self.closed_at = self.last_updated
            elif not self.is_long and price >= self.stop_loss:
                self.status = "STOPPED"
                self.realized_r = round((self.entry_price - self.stop_loss) / self.risk, 3)
                self.closed_at = self.last_updated
            # Check T1
            elif self.is_long and price >= self.target_1:
                self.status = "AT_T1"
                self.t1_hit = True
                self.stop_loss = self.entry_price  # Move stop to breakeven
            elif not self.is_long and price <= self.target_1:
                self.status = "AT_T1"
                self.t1_hit = True
                self.stop_loss = self.entry_price

        elif self.status == "AT_T1":
            # Check breakeven stop
            if self.is_long and price <= self.stop_loss:
                self.status = "STOPPED"
                self.realized_r = 0.0  # Breakeven
                self.closed_at = self.last_updated
            elif not self.is_long and price >= self.stop_loss:
                self.status = "STOPPED"
                self.realized_r = 0.0
                self.closed_at = self.last_updated
            # Check T2
            elif self.is_long and price >= self.target_2:
                self.status = "AT_T2"
                self.t2_hit = True
                # Set trailing stop
                if atr > 0:
                    self.trailing_stop = price - atr * self.trail_param
                else:
                    self.trailing_stop = self.entry_price
            elif not self.is_long and price <= self.target_2:
                self.status = "AT_T2"
                self.t2_hit = True
                if atr > 0:
                    self.trailing_stop = price + atr * self.trail_param
                else:
                    self.trailing_stop = self.entry_price

        elif self.status in ("AT_T2", "TRAILING"):
            self.status = "TRAILING"
            # Update trailing stop
            if atr > 0:
                if self.is_long:
                    new_trail = price - atr * self.trail_param
                    self.trailing_stop = max(self.trailing_stop, new_trail)
                    if price <= self.trailing_stop:
                        self.status = "CLOSED"
                        self.realized_r = round((self.trailing_stop - self.entry_price) / self.risk, 3)
                        self.closed_at = self.last_updated
                else:
                    new_trail = price + atr * self.trail_param
                    self.trailing_stop = min(self.trailing_stop, new_trail) if self.trailing_stop > 0 else new_trail
                    if price >= self.trailing_stop:
                        self.status = "CLOSED"
                        self.realized_r = round((self.entry_price - self.trailing_stop) / self.risk, 3)
                        self.closed_at = self.last_updated


class TradeTracker:
    """Manages the active trade store."""

    def __init__(self):
        self.trades: list[TrackedTrade] = []
        self.load()

    def load(self):
        """Load trades from disk."""
        if TRADES_FILE.exists():
            try:
                data = json.loads(TRADES_FILE.read_text())
                self.trades = [TrackedTrade(t) for t in data.get("trades", [])]
            except Exception:
                self.trades = []
        else:
            self.trades = []

    def save(self):
        """Save trades to disk."""
        TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": datetime.now().isoformat(),
            "total": len(self.trades),
            "active": sum(1 for t in self.trades if t.is_active),
            "closed": sum(1 for t in self.trades if t.is_closed),
            "trades": [t.to_dict() for t in self.trades],
        }
        TRADES_FILE.write_text(json.dumps(data, indent=2))

    def _generate_id(self, symbol: str, pattern: str) -> str:
        import random
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rnd = random.randint(1000, 9999)
        return f"{symbol}_{pattern.replace(' ', '_')}_{ts}_{rnd}"

    # ── SCANNING ──

    def scan_and_add(self, top_n: int = 50, symbols: list[str] = None,
                     min_confidence: float = 0.50, lookback_days: int = 20):
        """Scan top symbols looking back N trading days for historical setups.

        For each symbol:
        1. Fetch 365d of daily bars
        2. Walk backward through the last `lookback_days` bars
        3. At each bar, truncate data and run classify_all
        4. Record when the setup FIRST fired (oldest detection wins)
        5. Walk price forward to today to compute current status
        """
        from backend.data.massive_client import fetch_bars
        from backend.patterns.classifier import classify_all, extract_structures
        from copy import deepcopy
        import numpy as np

        POSITIVE_EDGE = {
            "Juicer Long", "RS Persistence Long", "Momentum Breakout",
            "Turtle Breakout Long", "BB Squeeze Long", "TS Momentum Long",
            "BAB Long", "Accumulation Long",
        }

        if symbols is None:
            if SYMBOLS_CACHE.exists():
                symbols = json.loads(SYMBOLS_CACHE.read_text()).get("symbols", [])[:100]
            else:
                symbols = ["AAPL", "NVDA", "TSLA", "MSFT", "META", "AMZN",
                           "GOOGL", "AMD", "SPY", "QQQ"]

        existing = {(t.symbol, t.pattern_name) for t in self.trades if t.is_active}
        all_setups = []
        print(f"  Historical scan: {len(symbols)} symbols × {lookback_days} days back")

        for i, symbol in enumerate(symbols):
            if i % 10 == 0:
                print(f"    [{i}/{len(symbols)}] {symbol}... ({len(all_setups)} setups so far)")

            try:
                bars = fetch_bars(symbol, "1d", 365)
                if not bars or not bars.bars or len(bars.bars) < 60:
                    continue
            except Exception:
                continue

            full_bars_list = bars.bars
            n_bars = len(full_bars_list)
            today_close = full_bars_list[-1].close

            # Simple ATR approximation from last 14 bars
            if n_bars >= 14:
                today_atr = float(np.mean([b.high - b.low for b in full_bars_list[-14:]]))
            else:
                today_atr = 0.0

            # Track which patterns already detected for this symbol (oldest wins)
            seen_patterns = {}

            # Walk backward: check each of the last N days
            max_offset = min(lookback_days, n_bars - 60)
            for day_offset in range(max_offset, 0, -1):
                end_idx = n_bars - day_offset
                if end_idx < 60:
                    continue

                # Create truncated BarSeries (Pydantic — properties auto-compute from bars)
                truncated = type(bars)(
                    symbol=bars.symbol,
                    timeframe=bars.timeframe,
                    bars=full_bars_list[:end_idx],
                )

                try:
                    setups = classify_all(truncated)
                    for setup in setups:
                        if setup.pattern_name not in POSITIVE_EDGE:
                            continue
                        if setup.confidence < min_confidence:
                            continue
                        if (symbol, setup.pattern_name) in existing:
                            continue
                        if setup.pattern_name not in seen_patterns:
                            seen_patterns[setup.pattern_name] = {
                                "setup": setup,
                                "detected_bar_idx": end_idx - 1,
                                "detected_date": full_bars_list[end_idx - 1].timestamp,
                                "entry_price": setup.entry_price,
                                "today_close": today_close,
                                "today_atr": today_atr,
                            }
                except Exception:
                    pass

            # Convert seen_patterns to setup entries
            for pattern_name, info in seen_patterns.items():
                setup = info["setup"]
                all_setups.append({
                    "symbol": symbol,
                    "setup": setup,
                    "timeframe": "1d",
                    "detected_date": info["detected_date"],
                    "today_close": info["today_close"],
                    "today_atr": info["today_atr"],
                })

        # Sort by confidence, take top N
        all_setups.sort(key=lambda x: x["setup"].confidence, reverse=True)
        added = 0

        for item in all_setups[:top_n]:
            setup = item["setup"]
            det_date = item["detected_date"]
            det_str = det_date.isoformat() if hasattr(det_date, 'isoformat') else str(det_date)

            trade_data = {
                "id": self._generate_id(setup.symbol, setup.pattern_name),
                "symbol": setup.symbol,
                "pattern_name": setup.pattern_name,
                "bias": setup.bias.value if hasattr(setup.bias, 'value') else str(setup.bias),
                "timeframe": item["timeframe"],
                "entry_price": setup.entry_price,
                "stop_loss": setup.stop_loss,
                "target_1": getattr(setup, 'target_1', setup.target_price),
                "target_2": getattr(setup, 'target_2', setup.target_price),
                "trail_type": getattr(setup, 'trail_type', 'atr'),
                "trail_param": getattr(setup, 'trail_param', 2.0),
                "confidence": setup.confidence,
                "description": setup.description,
                "detected_at": det_str,
                "position_splits": list(getattr(setup, 'position_splits', (0.5, 0.3, 0.2))),
                "source": "auto",
                "key_levels": getattr(setup, 'key_levels', {}),
                "status": "ACTIVE",
                "entered_at": det_str,
                "current_price": item["today_close"],
                "current_atr": item["today_atr"],
            }

            trade = TrackedTrade(trade_data)
            # Walk forward to compute current status
            trade.update_with_price(item["today_close"], item["today_atr"])

            self.trades.append(trade)
            existing.add((setup.symbol, setup.pattern_name))
            added += 1

        self.save()
        print(f"\n  Added {added} historical trades ({len(self.trades)} total)")
        print(f"  Detection range: {lookback_days} trading days back")
        return added

    def add_manual(self, setup_dict: dict) -> TrackedTrade:
        """Manually add a trade from the UI."""
        setup_dict["id"] = self._generate_id(
            setup_dict.get("symbol", "UNK"),
            setup_dict.get("pattern_name", "Manual")
        )
        setup_dict["source"] = "manual"
        setup_dict["status"] = "PENDING"
        trade = TrackedTrade(setup_dict)
        self.trades.append(trade)
        self.save()
        return trade

    # ── PRICE REFRESH ──

    def refresh_prices(self):
        """Fetch current prices and update all active trade statuses."""
        from backend.data.massive_client import fetch_bars

        active = [t for t in self.trades if t.is_active]
        if not active:
            print("  No active trades to refresh")
            return

        symbols = set(t.symbol for t in active)
        print(f"  Refreshing prices for {len(active)} trades across {len(symbols)} symbols...")

        price_cache = {}
        atr_cache = {}

        for symbol in symbols:
            try:
                bars = fetch_bars(symbol, "1d", 30)
                if bars.bars:
                    price_cache[symbol] = bars.bars[-1].close
                    if len(bars.bars) >= 14:
                        atr_cache[symbol] = float(np.mean([b.high - b.low for b in bars.bars[-14:]]))
            except Exception:
                continue

        # Update each trade
        for trade in active:
            price = price_cache.get(trade.symbol)
            atr = atr_cache.get(trade.symbol, 0.0)
            if price:
                trade.update_with_price(price, atr)

        # Archive closed trades
        newly_closed = [t for t in self.trades if t.is_closed and t.closed_at == t.last_updated]
        if newly_closed:
            print(f"  {len(newly_closed)} trades closed this refresh")

        self.save()
        print(f"  Updated {len(active)} trades")

    # ── QUERIES ──

    def get_active_trades(self) -> list[dict]:
        """Get all active trades for the frontend."""
        return [t.to_dict() for t in self.trades if t.is_active]

    def get_closed_trades(self) -> list[dict]:
        """Get closed trades for history view."""
        return [t.to_dict() for t in self.trades if t.is_closed]

    def get_all_trades(self) -> list[dict]:
        """Get everything."""
        return [t.to_dict() for t in self.trades]

    def get_trade_by_id(self, trade_id: str) -> Optional[dict]:
        """Get a specific trade."""
        for t in self.trades:
            if t.id == trade_id:
                return t.to_dict()
        return None

    def get_summary(self) -> dict:
        """Get portfolio summary stats."""
        active = [t for t in self.trades if t.is_active]
        closed = [t for t in self.trades if t.is_closed]

        active_long = [t for t in active if t.is_long]
        active_short = [t for t in active if not t.is_long]

        total_unrealized_r = sum(t.unrealized_r for t in active)
        total_realized_r = sum(t.realized_r for t in closed if t.realized_r != 0)

        winners = [t for t in closed if t.realized_r > 0]
        losers = [t for t in closed if t.realized_r < 0]

        return {
            "active_count": len(active),
            "active_long": len(active_long),
            "active_short": len(active_short),
            "closed_count": len(closed),
            "total_unrealized_r": round(total_unrealized_r, 2),
            "total_realized_r": round(total_realized_r, 2),
            "win_rate": round(len(winners) / len(closed) * 100, 1) if closed else 0,
            "avg_winner_r": round(np.mean([t.realized_r for t in winners]), 2) if winners else 0,
            "avg_loser_r": round(np.mean([t.realized_r for t in losers]), 2) if losers else 0,
            "best_trade": max((t.unrealized_r for t in active), default=0),
            "worst_trade": min((t.unrealized_r for t in active), default=0),
            "by_pattern": self._group_by_pattern(active),
            "last_updated": max((t.last_updated for t in self.trades), default=""),
        }

    def _group_by_pattern(self, trades: list) -> dict:
        groups = {}
        for t in trades:
            if t.pattern_name not in groups:
                groups[t.pattern_name] = {"count": 0, "avg_r": 0, "symbols": []}
            groups[t.pattern_name]["count"] += 1
            groups[t.pattern_name]["symbols"].append(t.symbol)
        for name, g in groups.items():
            pattern_trades = [t for t in trades if t.pattern_name == name]
            g["avg_r"] = round(np.mean([t.unrealized_r for t in pattern_trades]), 3)
        return groups

    # ── MANAGEMENT ──

    def close_trade(self, trade_id: str, reason: str = "manual"):
        """Manually close a trade."""
        for t in self.trades:
            if t.id == trade_id and t.is_active:
                t.status = "CLOSED"
                t.closed_at = datetime.now().isoformat()
                t.notes = f"Manually closed: {reason}"
                if t.current_price > 0 and t.risk > 0:
                    if t.is_long:
                        t.realized_r = round((t.current_price - t.entry_price) / t.risk, 3)
                    else:
                        t.realized_r = round((t.entry_price - t.current_price) / t.risk, 3)
                self.save()
                return True
        return False

    def remove_trade(self, trade_id: str):
        """Remove a trade entirely (doesn't archive)."""
        self.trades = [t for t in self.trades if t.id != trade_id]
        self.save()

    def archive_closed(self):
        """Move closed trades to archive file."""
        closed = [t for t in self.trades if t.is_closed]
        if not closed:
            return 0

        # Load existing archive
        archive = []
        if ARCHIVE_FILE.exists():
            try:
                archive = json.loads(ARCHIVE_FILE.read_text()).get("trades", [])
            except Exception:
                pass

        archive.extend([t.to_dict() for t in closed])
        ARCHIVE_FILE.write_text(json.dumps({
            "archived_at": datetime.now().isoformat(),
            "total": len(archive),
            "trades": archive,
        }, indent=2))

        # Remove from active
        self.trades = [t for t in self.trades if t.is_active]
        self.save()
        return len(closed)

    def clear_all(self):
        """Clear all trades (reset)."""
        self.trades = []
        self.save()


# ═══════════════════════════════════════════════════════════════
# CLI — quick testing
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    tracker = TradeTracker()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "scan":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            tracker.scan_and_add(top_n=n)

        elif cmd == "refresh":
            tracker.refresh_prices()

        elif cmd == "status":
            summary = tracker.get_summary()
            print(f"\n  Trade Tracker Status:")
            print(f"  Active:  {summary['active_count']} ({summary['active_long']} long, {summary['active_short']} short)")
            print(f"  Closed:  {summary['closed_count']}")
            print(f"  Unrealized: {summary['total_unrealized_r']:+.2f}R")
            print(f"  Realized:   {summary['total_realized_r']:+.2f}R")

            if summary['by_pattern']:
                print(f"\n  By Pattern:")
                for name, g in sorted(summary['by_pattern'].items(), key=lambda x: -x[1]['count']):
                    print(f"    {name:<25} {g['count']} trades  avg {g['avg_r']:+.3f}R")

        elif cmd == "list":
            trades = tracker.get_active_trades()
            print(f"\n  Active Trades ({len(trades)}):")
            print(f"  {'Symbol':<8} {'Pattern':<25} {'Bias':<6} {'Entry':>8} {'Current':>8} {'R':>7} {'Status'}")
            print(f"  {'─' * 80}")
            for t in sorted(trades, key=lambda x: -x['unrealized_r']):
                print(f"  {t['symbol']:<8} {t['pattern_name']:<25} {t['bias']:<6} "
                      f"${t['entry_price']:>7.2f} ${t['current_price']:>7.2f} "
                      f"{t['unrealized_r']:>+6.2f}R  {t['status']}")

        elif cmd == "archive":
            n = tracker.archive_closed()
            print(f"  Archived {n} closed trades")

        elif cmd == "clear":
            tracker.clear_all()
            print("  Cleared all trades")

        else:
            print(f"  Unknown command: {cmd}")
            print(f"  Usage: python -m backend.tracker.trade_tracker [scan|refresh|status|list|archive|clear]")
    else:
        print("  Usage: python -m backend.tracker.trade_tracker [scan|refresh|status|list|archive|clear]")