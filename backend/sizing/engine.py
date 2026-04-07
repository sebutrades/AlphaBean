"""
Position sizing engine for AlphaBean.

Calculates proper position sizes based on account risk parameters,
enforces per-trade and portfolio-level risk limits, and tracks
aggregate portfolio heat across all active positions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
_CONFIG_PATH = _CACHE_DIR / "sizing_config.json"
_ACTIVE_TRADES_PATH = _CACHE_DIR / "active_trades.json"

# ── Defaults ──────────────────────────────────────────────────────────────

_DEFAULT_CONFIG: dict[str, Any] = {
    "account_size": 25_000.0,
    "risk_per_trade_pct": 1.0,
    "max_portfolio_heat_pct": 6.0,
    "max_single_position_pct": 15.0,
    "max_correlated_positions": 3,
    "scale_with_conviction": True,
}

# ── Configuration ─────────────────────────────────────────────────────────


def get_sizing_config() -> dict[str, Any]:
    """Load sizing config from *cache/sizing_config.json*.

    Returns the persisted configuration if the file exists, otherwise
    returns a copy of the built-in defaults.
    """
    if _CONFIG_PATH.exists():
        try:
            with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return _DEFAULT_CONFIG.copy()


def save_sizing_config(config: dict[str, Any]) -> None:
    """Persist *config* to *cache/sizing_config.json*."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)


# ── Position Sizing ──────────────────────────────────────────────────────


def calculate_position(
    entry_price: float,
    stop_loss: float,
    bias: str = "long",
    size_modifier: float = 1.0,
    symbol: str = "",
) -> dict[str, Any]:
    """Calculate position size for a single trade.

    Parameters
    ----------
    entry_price:
        Planned entry price.
    stop_loss:
        Stop-loss price.
    bias:
        ``"long"`` or ``"short"``.
    size_modifier:
        Multiplier applied to the base risk budget (e.g. 0.5 for half
        size, 1.5 for conviction).
    symbol:
        Ticker symbol (used only for warning messages).

    Returns
    -------
    dict with keys: shares, dollar_risk, position_value, risk_per_share,
    risk_pct_of_account, size_modifier_applied, max_position_value,
    capped, warnings.
    """
    config = get_sizing_config()
    account_size: float = config["account_size"]
    risk_per_trade_pct: float = config["risk_per_trade_pct"]
    max_single_position_pct: float = config["max_single_position_pct"]

    warnings: list[str] = []

    # --- risk per share ------------------------------------------------
    risk_per_share = abs(entry_price - stop_loss)

    if risk_per_share == 0:
        return {
            "shares": 0,
            "dollar_risk": 0.0,
            "position_value": 0.0,
            "risk_per_share": 0.0,
            "risk_pct_of_account": 0.0,
            "size_modifier_applied": size_modifier,
            "max_position_value": account_size * max_single_position_pct / 100.0,
            "capped": False,
            "warnings": ["risk_per_share is 0 — entry and stop are identical"],
        }

    # --- base dollar risk budget ---------------------------------------
    dollar_risk_allowed = account_size * (risk_per_trade_pct / 100.0) * size_modifier

    # --- raw share count (always round down) ---------------------------
    shares = math.floor(dollar_risk_allowed / risk_per_share)

    # --- cap to max single-position value ------------------------------
    max_position_value = account_size * max_single_position_pct / 100.0
    position_value = shares * entry_price
    capped = False

    if position_value > max_position_value:
        shares = math.floor(max_position_value / entry_price)
        position_value = shares * entry_price
        capped = True
        warnings.append(
            f"Position capped to {max_single_position_pct}% of account "
            f"(max ${max_position_value:,.2f})"
        )

    # --- actual dollar risk after possible cap -------------------------
    dollar_risk = shares * risk_per_share
    risk_pct_of_account = (dollar_risk / account_size) * 100.0 if account_size else 0.0

    # --- sanity warnings -----------------------------------------------
    if shares > 10_000:
        tag = f" ({symbol})" if symbol else ""
        warnings.append(
            f"Large share count{tag}: {shares:,} shares — "
            "verify this is not a penny-stock sizing error"
        )

    if account_size and position_value > account_size * 0.50:
        warnings.append(
            f"Position value ${position_value:,.2f} exceeds 50% of account"
        )

    return {
        "shares": shares,
        "dollar_risk": round(dollar_risk, 2),
        "position_value": round(position_value, 2),
        "risk_per_share": round(risk_per_share, 4),
        "risk_pct_of_account": round(risk_pct_of_account, 4),
        "size_modifier_applied": size_modifier,
        "max_position_value": round(max_position_value, 2),
        "capped": capped,
        "warnings": warnings,
    }


# ── Portfolio Heat ────────────────────────────────────────────────────────


def calculate_portfolio_heat(active_trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate aggregate portfolio risk across all active positions.

    Parameters
    ----------
    active_trades:
        List of trade dicts.  Expected keys per trade:

        * ``entry_price`` (float)
        * ``stop_loss`` (float)
        * ``bias`` (str) — ``"long"`` or ``"short"``
        * ``shares`` (int, optional) — if missing, computed from
          ``initial_risk`` and per-share risk
        * ``initial_risk`` (float, optional) — dollar risk when opened
        * ``current_price`` (float, optional)
        * ``unrealized_r`` (float, optional)

    Returns
    -------
    dict with keys: total_risk_dollars, total_risk_pct, positions_count,
    long_exposure, short_exposure, net_exposure, gross_exposure,
    can_add_trade, remaining_risk_budget, warnings.
    """
    config = get_sizing_config()
    account_size: float = config["account_size"]
    max_heat_pct: float = config["max_portfolio_heat_pct"]

    total_risk_dollars = 0.0
    long_exposure = 0.0
    short_exposure = 0.0
    warnings: list[str] = []

    for trade in active_trades:
        entry = float(trade.get("entry_price", 0))
        stop = float(trade.get("stop_loss", 0))
        bias = str(trade.get("bias", "long")).lower()
        risk_per_share = abs(entry - stop)

        # Determine share count
        shares = trade.get("shares")
        if shares is None:
            initial_risk = trade.get("initial_risk")
            if initial_risk is not None and risk_per_share > 0:
                shares = math.floor(float(initial_risk) / risk_per_share)
            else:
                shares = 0
        else:
            shares = int(shares)

        # Accumulate risk
        trade_risk = shares * risk_per_share
        total_risk_dollars += trade_risk

        # Exposure by direction
        position_value = shares * entry
        if bias == "short":
            short_exposure += position_value
        else:
            long_exposure += position_value

    total_risk_pct = (total_risk_dollars / account_size * 100.0) if account_size else 0.0
    net_exposure = long_exposure - short_exposure
    gross_exposure = long_exposure + short_exposure

    max_heat_dollars = account_size * max_heat_pct / 100.0
    remaining = max(max_heat_dollars - total_risk_dollars, 0.0)
    can_add = total_risk_pct < max_heat_pct

    if not can_add:
        warnings.append(
            f"Portfolio heat {total_risk_pct:.2f}% has reached the "
            f"{max_heat_pct:.1f}% limit — no new trades allowed"
        )

    if account_size and gross_exposure > account_size:
        warnings.append(
            f"Gross exposure ${gross_exposure:,.2f} exceeds account size "
            f"${account_size:,.2f}"
        )

    return {
        "total_risk_dollars": round(total_risk_dollars, 2),
        "total_risk_pct": round(total_risk_pct, 4),
        "positions_count": len(active_trades),
        "long_exposure": round(long_exposure, 2),
        "short_exposure": round(short_exposure, 2),
        "net_exposure": round(net_exposure, 2),
        "gross_exposure": round(gross_exposure, 2),
        "can_add_trade": can_add,
        "remaining_risk_budget": round(remaining, 2),
        "warnings": warnings,
    }


# ── Summary Helper ────────────────────────────────────────────────────────


def get_position_summary() -> dict[str, Any]:
    """Load active trades and return sizing config combined with portfolio heat.

    Reads ``cache/active_trades.json`` (expects a JSON array of trade
    dicts).  If the file is missing or unreadable, an empty trade list is
    assumed.

    Returns
    -------
    dict combining the sizing config (under ``"config"``) and the full
    portfolio-heat breakdown (under ``"portfolio_heat"``).
    """
    active_trades: list[dict[str, Any]] = []
    if _ACTIVE_TRADES_PATH.exists():
        try:
            with _ACTIVE_TRADES_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    active_trades = data
        except (json.JSONDecodeError, OSError):
            pass

    config = get_sizing_config()
    heat = calculate_portfolio_heat(active_trades)

    return {
        "config": config,
        "portfolio_heat": heat,
    }
