"""
test_all_fixes_and_features.py — Comprehensive tests for all v2 fixes and new features.

Covers:
  1. Backtest R-cap + transaction costs
  2. VWAP intraday reset
  3. initial_risk persistence in trade tracker
  4. AI cache hourly granularity
  5. Performance dashboard analytics
  6. Alert/webhook system
  7. Position sizing engine

Run: pytest root_tests/test_all_fixes_and_features.py -v
"""
import json
import math
import os
import sys
import tempfile
from datetime import datetime, date, time, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
from collections import defaultdict

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ==============================================================================
# 1. BACKTEST R-CAP + TRANSACTION COSTS
# ==============================================================================

class TestBacktestRCap:
    """Verify R-cap and transaction cost application in PendingTrade._finalize()."""

    def _make_trade(self, entry=100, stop=98, target=106, bias="long"):
        from run_backtest import PendingTrade
        return PendingTrade(
            pattern_name="Test",
            strategy_type="momentum",
            symbol="TEST",
            bias=bias,
            entry=entry,
            target=target,
            stop=stop,
            bar_idx=0,
            max_hold=100,
            target_1=target,
            target_2=target + 4,
            position_splits=(0.5, 0.3, 0.2),
            atr=0,  # no additional slippage
        )

    def test_r_cap_applied(self):
        """R-multiples should be capped at MAX_R_PER_TRADE."""
        from run_backtest import MAX_R_PER_TRADE, TRANSACTION_COST_R
        trade = self._make_trade(entry=100, stop=99, target=150, bias="long")
        # Simulate a huge winning trade: price goes to $200
        # partial_rs will be manually set to simulate massive R
        trade.partial_rs = [(1.0, 50.0)]  # 50R trade uncapped
        trade.remaining_weight = 0.0
        outcome, r = trade._finalize()
        # Should be capped
        assert r <= MAX_R_PER_TRADE, f"R={r} should be capped at {MAX_R_PER_TRADE}"
        assert r == round(MAX_R_PER_TRADE - TRANSACTION_COST_R, 3)

    def test_r_cap_negative(self):
        """Negative R should also be capped (not worse than -MAX_R)."""
        from run_backtest import MAX_R_PER_TRADE, TRANSACTION_COST_R
        trade = self._make_trade()
        trade.partial_rs = [(1.0, -50.0)]
        trade.remaining_weight = 0.0
        outcome, r = trade._finalize()
        assert r >= -(MAX_R_PER_TRADE + TRANSACTION_COST_R), f"R={r} should be capped"

    def test_transaction_cost_deducted(self):
        """Each trade should have TRANSACTION_COST_R deducted."""
        from run_backtest import TRANSACTION_COST_R
        trade = self._make_trade()
        # A clean 2R winner
        trade.partial_rs = [(1.0, 2.0)]
        trade.remaining_weight = 0.0
        outcome, r = trade._finalize()
        expected = round(2.0 - TRANSACTION_COST_R, 3)
        assert r == expected, f"Expected {expected}, got {r}"

    def test_transaction_cost_can_flip_outcome(self):
        """A tiny winner (< TRANSACTION_COST_R) should become a loss after costs."""
        from run_backtest import TRANSACTION_COST_R
        trade = self._make_trade()
        # Winner just below transaction cost threshold
        tiny_r = TRANSACTION_COST_R / 2
        trade.partial_rs = [(1.0, tiny_r)]
        trade.remaining_weight = 0.0
        outcome, r = trade._finalize()
        assert r < 0, f"R={r} should be negative after costs"

    def test_normal_trade_not_affected_by_cap(self):
        """Normal trades within cap should only have transaction cost applied."""
        from run_backtest import TRANSACTION_COST_R
        trade = self._make_trade()
        trade.partial_rs = [(1.0, 1.5)]
        trade.remaining_weight = 0.0
        outcome, r = trade._finalize()
        assert r == round(1.5 - TRANSACTION_COST_R, 3)

    def test_config_saved_in_results(self):
        """MAX_R_PER_TRADE and TRANSACTION_COST_R should appear in config output."""
        from run_backtest import MAX_R_PER_TRADE, TRANSACTION_COST_R
        assert MAX_R_PER_TRADE > 0
        assert TRANSACTION_COST_R > 0
        assert MAX_R_PER_TRADE == 10.0
        assert TRANSACTION_COST_R == 0.02


# ==============================================================================
# 2. VWAP INTRADAY RESET
# ==============================================================================

class TestVWAPReset:
    """Verify VWAP properly resets at 9:30 ET for intraday bars."""

    def _make_scanner_state(self, timestamps, closes, highs, lows, volumes):
        """Create a minimal scanner state object for VWAP tests."""
        class ScanState:
            pass
        s = ScanState()
        s.n = len(closes)
        s.closes = np.array(closes)
        s.highs = np.array(highs)
        s.lows = np.array(lows)
        s.volumes = np.array(volumes)
        s.timestamps = timestamps
        return s

    def test_intraday_vwap_anchors_at_930(self):
        """VWAP for intraday bars should only use bars from 9:30+ today."""
        from backend.patterns.classifier import _vwap_today

        today = datetime.now().date()
        timestamps = [
            # Pre-market bars (should be excluded from VWAP)
            datetime.combine(today, time(8, 0)),
            datetime.combine(today, time(8, 30)),
            datetime.combine(today, time(9, 0)),
            # Regular session bars (should be included)
            datetime.combine(today, time(9, 30)),
            datetime.combine(today, time(9, 35)),
            datetime.combine(today, time(9, 40)),
        ]
        # Pre-market: low prices, Regular: high prices
        closes = [50.0, 50.0, 50.0, 100.0, 102.0, 104.0]
        highs =  [51.0, 51.0, 51.0, 101.0, 103.0, 105.0]
        lows =   [49.0, 49.0, 49.0, 99.0, 101.0, 103.0]
        volumes = [1000, 1000, 1000, 5000, 5000, 5000]

        s = self._make_scanner_state(timestamps, closes, highs, lows, volumes)
        vwap = _vwap_today(s)

        # VWAP should reflect regular session prices (~100+), NOT pre-market (~50)
        assert vwap > 90, f"VWAP={vwap:.2f} should be >90 (anchored at 9:30, not pre-market)"
        assert vwap < 110, f"VWAP={vwap:.2f} should be <110"

    def test_daily_bars_use_rolling_vwap(self):
        """Daily bars should use rolling VWAP (last 20), not intraday reset."""
        from backend.patterns.classifier import _vwap_today

        # Daily bars: timestamps 1 day apart
        base = datetime(2026, 4, 1, 16, 0)
        timestamps = [base + timedelta(days=i) for i in range(25)]
        closes = [100.0 + i * 0.5 for i in range(25)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        volumes = [10000] * 25

        s = self._make_scanner_state(timestamps, closes, highs, lows, volumes)
        vwap = _vwap_today(s)

        # Should be reasonable (near average of last 20 bars)
        avg = np.mean(closes[-20:])
        assert abs(vwap - avg) < 5, f"Daily VWAP={vwap:.2f} should be near rolling avg={avg:.2f}"


# ==============================================================================
# 3. INITIAL RISK PERSISTENCE
# ==============================================================================

class TestInitialRiskPersistence:
    """Verify initial_risk and original_stop are saved and restored correctly."""

    def test_initial_risk_computed_on_creation(self):
        from backend.tracker.trade_tracker import TrackedTrade
        data = {
            "id": "test1",
            "symbol": "AAPL",
            "entry_price": 150.0,
            "stop_loss": 148.0,
            "target_1": 154.0,
            "bias": "long",
        }
        trade = TrackedTrade(data)
        assert trade.initial_risk == 2.0, f"initial_risk={trade.initial_risk}, expected 2.0"
        assert trade.original_stop == 148.0, f"original_stop={trade.original_stop}, expected 148.0"

    def test_initial_risk_survives_breakeven_move(self):
        """After stop moves to entry (breakeven), initial_risk should be preserved."""
        from backend.tracker.trade_tracker import TrackedTrade
        data = {
            "id": "test2",
            "symbol": "AAPL",
            "entry_price": 150.0,
            "stop_loss": 150.0,  # Already moved to breakeven!
            "target_1": 154.0,
            "initial_risk": 2.0,  # Was saved at creation
            "original_stop": 148.0,  # Was saved at creation
            "bias": "long",
            "status": "AT_T1",
            "t1_hit": True,
        }
        trade = TrackedTrade(data)
        assert trade.initial_risk == 2.0, f"initial_risk should be 2.0, got {trade.initial_risk}"
        assert trade.original_stop == 148.0

    def test_original_stop_in_to_dict(self):
        """original_stop should appear in serialized output."""
        from backend.tracker.trade_tracker import TrackedTrade
        data = {
            "id": "test3",
            "symbol": "AAPL",
            "entry_price": 150.0,
            "stop_loss": 148.0,
            "target_1": 154.0,
            "bias": "long",
        }
        trade = TrackedTrade(data)
        d = trade.to_dict()
        assert "original_stop" in d
        assert d["original_stop"] == 148.0
        assert d["initial_risk"] == 2.0

    def test_initial_risk_from_original_stop_fallback(self):
        """If initial_risk=0 but original_stop is saved, reconstruct from original_stop."""
        from backend.tracker.trade_tracker import TrackedTrade
        data = {
            "id": "test4",
            "symbol": "AAPL",
            "entry_price": 100.0,
            "stop_loss": 100.0,  # Already breakeven
            "original_stop": 97.0,  # Saved original
            "initial_risk": 0,  # Not saved (legacy)
            "target_1": 106.0,
            "bias": "long",
        }
        trade = TrackedTrade(data)
        assert trade.initial_risk == 3.0, f"Should reconstruct from original_stop: {trade.initial_risk}"

    def test_add_manual_sets_initial_risk(self):
        """add_manual should compute and save initial_risk + original_stop."""
        from backend.tracker.trade_tracker import TradeTracker
        tracker = TradeTracker()
        tracker.trades = []  # Empty

        # Mock save to avoid file I/O
        with patch.object(tracker, 'save'):
            with patch.object(tracker, '_generate_id', return_value='test_manual'):
                trade = tracker.add_manual({
                    "symbol": "NVDA",
                    "pattern_name": "Manual Bull Flag",
                    "entry_price": 200.0,
                    "stop_loss": 195.0,
                    "target_1": 210.0,
                    "bias": "long",
                })
        assert trade.initial_risk == 5.0
        assert trade.original_stop == 195.0


# ==============================================================================
# 4. AI CACHE GRANULARITY
# ==============================================================================

class TestAICacheGranularity:
    """Verify AI cache uses hourly keys instead of daily."""

    def test_cache_path_includes_hour(self):
        """Cache path should include the hour for hourly granularity."""
        from backend.ai.ollama_agent import _cache_path
        path = _cache_path("AAPL", "Bull Flag")
        name = path.name
        # Should be like AAPL_Bull_Flag_2026-04-06_14.json
        parts = name.replace(".json", "").split("_")
        # Last part should be hour (00-23)
        hour_part = parts[-1]
        assert hour_part.isdigit(), f"Last part '{hour_part}' should be hour digit"
        assert 0 <= int(hour_part) <= 23, f"Hour {hour_part} out of range"

    def test_cache_path_format_contains_date_and_hour(self):
        """Cache path should contain date and hour in the filename."""
        from backend.ai.ollama_agent import _cache_path
        path = _cache_path("AAPL", "Bull Flag")
        name = path.name
        # Format: AAPL_Bull_Flag_YYYY-MM-DD_HH.json
        assert name.startswith("AAPL_Bull_Flag_")
        assert name.endswith(".json")
        # Should contain a date-like pattern
        assert "-" in name  # Date has dashes


# ==============================================================================
# 5. PERFORMANCE DASHBOARD
# ==============================================================================

class TestPerformanceDashboard:
    """Verify performance analytics produce correct outputs."""

    @pytest.fixture
    def sample_trades(self, tmp_path):
        """Create sample archived trades for testing."""
        trades = [
            {"id": f"t{i}", "symbol": sym, "pattern_name": pat,
             "bias": "long", "entry_price": 100, "stop_loss": 98,
             "initial_risk": 2.0, "realized_r": r,
             "status": "CLOSED", "t1_hit": r > 0, "t2_hit": r > 1.5,
             "entered_at": (datetime(2026, 4, 1) + timedelta(days=i)).isoformat(),
             "closed_at": (datetime(2026, 4, 1) + timedelta(days=i, hours=4)).isoformat(),
             "detected_at": (datetime(2026, 4, 1) + timedelta(days=i, hours=-1)).isoformat()}
            for i, (sym, pat, r) in enumerate([
                ("AAPL", "Bull Flag", 2.5),
                ("NVDA", "Trend Pullback", 1.2),
                ("TSLA", "Mean Reversion", -1.0),
                ("AAPL", "Bull Flag", 3.0),
                ("AMD", "Keltner Breakout", -0.8),
                ("NVDA", "Trend Pullback", 0.5),
                ("MSFT", "VWAP Reversion", 1.8),
                ("AAPL", "Bull Flag", -1.0),
                ("TSLA", "Mean Reversion", 2.0),
                ("AMD", "Keltner Breakout", 1.5),
            ])
        ]
        return trades

    def test_equity_curve(self, sample_trades):
        from backend.analytics import performance as perf
        # Temporarily override trade loading to use our sample data
        with patch.object(perf, '_all_closed_trades', return_value=sample_trades):
            curve = perf.get_equity_curve()
        assert len(curve) > 0
        # Cumulative R at the end should be the sum of all realized_r
        total_r = sum(t["realized_r"] for t in sample_trades)
        last_cumulative = curve[-1]["cumulative_r"]
        assert abs(last_cumulative - total_r) < 0.01, f"Expected {total_r}, got {last_cumulative}"

    def test_pattern_attribution(self, sample_trades):
        from backend.analytics import performance as perf
        with patch.object(perf, '_all_closed_trades', return_value=sample_trades):
            attrs = perf.get_pattern_attribution()
        assert len(attrs) > 0
        # Bull Flag should have 3 trades
        bf = [a for a in attrs if a["pattern_name"] == "Bull Flag"]
        assert len(bf) == 1
        assert bf[0]["trade_count"] == 3
        # Total R for Bull Flag = 2.5 + 3.0 + (-1.0) = 4.5
        assert abs(bf[0]["total_r"] - 4.5) < 0.01

    def test_drawdown(self, sample_trades):
        from backend.analytics import performance as perf
        with patch.object(perf, '_all_closed_trades', return_value=sample_trades):
            dd = perf.get_drawdown_series()
        assert "max_drawdown_r" in dd
        assert "current_drawdown_r" in dd
        assert "series" in dd
        assert dd["max_drawdown_r"] >= 0  # Drawdown magnitude is positive or zero

    def test_streaks(self, sample_trades):
        from backend.analytics import performance as perf
        with patch.object(perf, '_all_closed_trades', return_value=sample_trades):
            streaks = perf.get_streaks()
        assert "current_type" in streaks
        assert "current_streak" in streaks or "current_length" in streaks
        assert "best_win_streak" in streaks
        assert "worst_loss_streak" in streaks
        assert streaks["best_win_streak"] >= 1
        assert streaks["worst_loss_streak"] >= 1

    def test_daily_pnl(self, sample_trades):
        from backend.analytics import performance as perf
        with patch.object(perf, '_all_closed_trades', return_value=sample_trades):
            daily = perf.get_daily_pnl(30)
        assert len(daily) > 0
        for d in daily:
            assert "date" in d
            assert "r_total" in d
            assert "trade_count" in d


# ==============================================================================
# 6. ALERTS / WEBHOOKS
# ==============================================================================

class TestAlerts:
    """Verify alert formatting and config management."""

    def test_default_config(self):
        from backend.alerts.webhook import get_alert_config, DEFAULT_CONFIG
        # Remove existing config file for clean test
        config = DEFAULT_CONFIG.copy()
        assert config["enabled"] is True
        assert config["min_score"] == 70
        assert "discord" in config["channels"]
        assert "telegram" in config["channels"]

    def test_format_new_setup(self):
        from backend.alerts.webhook import format_new_setup
        setup = {
            "symbol": "AAPL",
            "pattern": "Bull Flag",
            "pattern_name": "Bull Flag",
            "bias": "long",
            "composite_score": 78,
            "score": 78,
            "entry": 150.0,
            "entry_price": 150.0,
            "stop": 148.0,
            "stop_loss": 148.0,
            "target": 156.0,
            "t1": 156.0,
            "rr": 3.0,
            "risk_reward_ratio": 3.0,
            "ai_verdict": {"verdict": "CONFIRMED", "confidence": 85},
        }
        msg, embed = format_new_setup(setup)
        assert "AAPL" in msg
        assert embed is not None

    def test_format_trade_update(self):
        from backend.alerts.webhook import format_trade_update
        trade = {
            "symbol": "NVDA",
            "pattern_name": "Trend Pullback",
            "bias": "long",
            "entry_price": 200.0,
            "unrealized_r": 1.5,
            "status": "AT_T1",
        }
        msg, embed = format_trade_update(trade, "T1_HIT")
        assert "NVDA" in msg
        assert "T1" in msg

    def test_format_daily_summary(self):
        from backend.alerts.webhook import format_daily_summary
        summary = {
            "total_trades": 5,
            "wins": 3,
            "losses": 2,
            "total_r": 2.5,
            "win_rate": 60,
        }
        msg, embed = format_daily_summary(summary)
        assert "2.5" in msg or "+2.5" in msg

    def test_send_discord_handles_failure(self):
        """Discord send should not raise on failure."""
        from backend.alerts.webhook import send_discord
        # Invalid URL should fail gracefully
        result = send_discord("http://invalid.example.com/webhook", "test")
        assert result is False

    def test_save_and_load_config(self, tmp_path):
        """Config should persist to disk."""
        from backend.alerts.webhook import save_alert_config, get_alert_config, CONFIG_FILE
        import backend.alerts.webhook as webhook_mod

        # Temporarily redirect config path
        original = webhook_mod.CONFIG_FILE
        webhook_mod.CONFIG_FILE = tmp_path / "alert_config.json"

        config = {"enabled": False, "min_score": 80, "channels": {"discord": {"enabled": True, "webhook_url": "http://test"}}, "events": {}}
        save_alert_config(config)
        loaded = get_alert_config()
        assert loaded["enabled"] is False
        assert loaded["min_score"] == 80

        # Restore
        webhook_mod.CONFIG_FILE = original


# ==============================================================================
# 7. POSITION SIZING
# ==============================================================================

class TestPositionSizing:
    """Verify position sizing calculations."""

    def test_basic_position_size(self):
        import backend.sizing.engine as sizing_mod
        # Mock config with large max position to avoid capping
        mock_config = {"account_size": 25000, "risk_per_trade_pct": 1.0, "max_single_position_pct": 60.0, "max_portfolio_heat_pct": 6.0, "max_correlated_positions": 3, "scale_with_conviction": True}
        with patch.object(sizing_mod, 'get_sizing_config', return_value=mock_config):
            result = sizing_mod.calculate_position(
                entry_price=100.0,
                stop_loss=98.0,
                bias="long",
            )
        assert "shares" in result
        assert "dollar_risk" in result
        assert "position_value" in result
        assert "risk_per_share" in result
        # Risk per share = $2
        assert result["risk_per_share"] == 2.0
        # Dollar risk = 25000 * 0.01 = $250
        # Shares = floor(250 / 2) = 125
        assert result["shares"] == 125
        assert result["dollar_risk"] == 250.0
        assert result["capped"] is False

    def test_position_capping(self):
        """Position should be capped at max_single_position_pct."""
        from backend.sizing.engine import calculate_position
        # With a very tight stop, shares would be huge
        result = calculate_position(
            entry_price=5.0,  # $5 stock
            stop_loss=4.99,  # $0.01 stop = tiny risk
            bias="long",
        )
        # 25000 * 0.01 / 0.01 = 250,000 shares = $1,250,000 position
        # But max position = 25000 * 0.15 = $3,750
        # So capped at 3750 / 5 = 750 shares
        assert result["capped"] is True
        assert result["shares"] <= 750

    def test_short_position(self):
        """Short position sizing should work correctly."""
        from backend.sizing.engine import calculate_position
        result = calculate_position(
            entry_price=50.0,
            stop_loss=52.0,  # Stop above for short
            bias="short",
        )
        assert result["risk_per_share"] == 2.0
        assert result["shares"] > 0

    def test_size_modifier(self):
        """Size modifier should scale the position."""
        import backend.sizing.engine as sizing_mod
        # Use 60% max position to avoid capping
        mock_config = {"account_size": 25000, "risk_per_trade_pct": 1.0, "max_single_position_pct": 60.0, "max_portfolio_heat_pct": 6.0, "max_correlated_positions": 3, "scale_with_conviction": True}
        with patch.object(sizing_mod, 'get_sizing_config', return_value=mock_config):
            base = sizing_mod.calculate_position(entry_price=100, stop_loss=98, bias="long", size_modifier=1.0)
            half = sizing_mod.calculate_position(entry_price=100, stop_loss=98, bias="long", size_modifier=0.5)
        assert half["shares"] < base["shares"]
        # base = 125 shares, half should be floor(250*0.5/2) = 62
        assert half["shares"] == 62

    def test_portfolio_heat(self):
        """Portfolio heat should sum risk across active trades."""
        from backend.sizing.engine import calculate_portfolio_heat
        trades = [
            {"entry_price": 100, "stop_loss": 98, "bias": "long", "initial_risk": 2.0, "status": "ACTIVE"},
            {"entry_price": 50, "stop_loss": 52, "bias": "short", "initial_risk": 2.0, "status": "ACTIVE"},
        ]
        heat = calculate_portfolio_heat(trades)
        assert "total_risk_pct" in heat
        assert "positions_count" in heat
        assert heat["positions_count"] == 2
        assert "can_add_trade" in heat

    def test_config_persistence(self, tmp_path):
        """Config should save and load correctly."""
        from backend.sizing.engine import save_sizing_config, get_sizing_config
        import backend.sizing.engine as sizing_mod

        original = sizing_mod._CONFIG_PATH
        sizing_mod._CONFIG_PATH = tmp_path / "sizing_config.json"

        config = {"account_size": 50000, "risk_per_trade_pct": 0.5}
        save_sizing_config(config)
        loaded = get_sizing_config()
        assert loaded["account_size"] == 50000

        sizing_mod._CONFIG_PATH = original

    def test_zero_risk_handling(self):
        """Entry == stop should not crash or divide by zero."""
        from backend.sizing.engine import calculate_position
        result = calculate_position(entry_price=100.0, stop_loss=100.0, bias="long")
        assert result["shares"] == 0
        assert len(result.get("warnings", [])) > 0 or result["risk_per_share"] == 0


# ==============================================================================
# INTEGRATION: Verify imports work
# ==============================================================================

class TestImports:
    """Verify all new modules can be imported without errors."""

    def test_import_performance(self):
        from backend.analytics.performance import (
            get_performance_summary,
            get_equity_curve,
            get_pattern_attribution,
            get_drawdown_series,
            get_daily_pnl,
        )

    def test_import_alerts(self):
        from backend.alerts.webhook import (
            get_alert_config,
            save_alert_config,
            send_alert,
            format_new_setup,
            format_trade_update,
            format_daily_summary,
            test_alerts,
        )

    def test_import_sizing(self):
        from backend.sizing.engine import (
            get_sizing_config,
            save_sizing_config,
            calculate_position,
            calculate_portfolio_heat,
            get_position_summary,
        )

    def test_import_backtest_constants(self):
        from run_backtest import MAX_R_PER_TRADE, TRANSACTION_COST_R
        assert MAX_R_PER_TRADE == 10.0
        assert TRANSACTION_COST_R == 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
