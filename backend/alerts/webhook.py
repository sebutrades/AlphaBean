"""
backend/alerts/webhook.py — Alert & Notification System

Sends notifications via Discord and/or Telegram webhooks when important
trading events happen (new setups, trade status changes, daily summaries).

Config is stored in cache/alert_config.json.  All sends are fire-and-forget
with a 5-second timeout — alerting failures are logged but never raised so
they cannot crash the main application.

USAGE:
  from backend.alerts.webhook import alert_new_setup, alert_trade_event

  alert_new_setup(setup_dict)
  alert_trade_event(trade_dict, "T1_HIT")
"""
import json
import logging
from datetime import datetime
from pathlib import Path
import requests

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_FILE = Path("cache/alert_config.json")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CONFIG: dict = {
    "enabled": True,
    "min_score": 70,
    "channels": {
        "discord": {
            "enabled": False,
            "webhook_url": "",
        },
        "telegram": {
            "enabled": False,
            "bot_token": "",
            "chat_id": "",
        },
    },
    "events": {
        "new_setup": True,
        "t1_hit": True,
        "stopped": True,
        "daily_summary": True,
        "gap_risk": True,
    },
}

# Discord embed colours (decimal)
COLOR_GREEN = 0x2ECC71   # long / win
COLOR_RED   = 0xE74C3C   # short / loss
COLOR_BLUE  = 0x3498DB   # info / neutral

SEND_TIMEOUT = 5  # seconds


# ── Configuration ─────────────────────────────────────────────────────────────

def get_alert_config() -> dict:
    """Load alert config.  Returns defaults if file doesn't exist."""
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIG.copy()
    try:
        data = json.loads(CONFIG_FILE.read_text())
        # Merge with defaults so new keys are always present
        merged = DEFAULT_CONFIG.copy()
        merged.update(data)
        merged["channels"] = {**DEFAULT_CONFIG["channels"], **data.get("channels", {})}
        merged["events"] = {**DEFAULT_CONFIG["events"], **data.get("events", {})}
        return merged
    except Exception as exc:
        log.warning("Failed to read alert config, using defaults: %s", exc)
        return DEFAULT_CONFIG.copy()


def save_alert_config(config: dict) -> None:
    """Save alert config to cache/alert_config.json."""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(config, indent=2))
    except Exception as exc:
        log.error("Failed to save alert config: %s", exc)


# ── Alert Sending ─────────────────────────────────────────────────────────────

def send_alert(event_type: str, data: dict) -> dict:
    """
    Route an alert to all configured channels.

    Returns ``{channel_name: True/False}`` indicating delivery status per channel.
    """
    config = get_alert_config()

    if not config.get("enabled"):
        return {}

    # Check whether this event type is enabled
    if not config.get("events", {}).get(event_type, False):
        return {}

    message = data.get("message", "")
    embed = data.get("embed")
    results: dict[str, bool] = {}

    channels = config.get("channels", {})

    # Discord
    dc = channels.get("discord", {})
    if dc.get("enabled") and dc.get("webhook_url"):
        results["discord"] = send_discord(dc["webhook_url"], message, embed)

    # Telegram
    tg = channels.get("telegram", {})
    if tg.get("enabled") and tg.get("bot_token") and tg.get("chat_id"):
        results["telegram"] = send_telegram(tg["bot_token"], tg["chat_id"], message)

    return results


def send_discord(webhook_url: str, message: str, embed: dict = None) -> bool:
    """Send a message to Discord via webhook.  Returns True on success."""
    try:
        payload: dict = {"content": message}
        if embed:
            payload["embeds"] = [embed]
        resp = requests.post(
            webhook_url,
            json=payload,
            timeout=SEND_TIMEOUT,
        )
        if resp.status_code in (200, 204):
            return True
        log.warning("Discord webhook returned %s: %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        log.error("Discord send failed: %s", exc)
        return False


def send_telegram(bot_token: str, chat_id: str, message: str) -> bool:
    """Send a message to Telegram via Bot API.  Returns True on success."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=SEND_TIMEOUT,
        )
        if resp.ok:
            return True
        log.warning("Telegram API returned %s: %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        log.error("Telegram send failed: %s", exc)
        return False


# ── Alert Formatting ─────────────────────────────────────────────────────────

def _bias_color(bias: str) -> int:
    """Return embed colour based on trade bias."""
    if bias and bias.upper() == "LONG":
        return COLOR_GREEN
    if bias and bias.upper() == "SHORT":
        return COLOR_RED
    return COLOR_BLUE


def _fmt_price(val) -> str:
    """Format a price value, handling None gracefully."""
    if val is None:
        return "—"
    try:
        return f"${float(val):.2f}"
    except (TypeError, ValueError):
        return str(val)


def format_new_setup(setup: dict) -> tuple[str, dict]:
    """Format a new high-score setup alert.  Returns (message, discord_embed)."""
    symbol  = setup.get("symbol", "???")
    pattern = setup.get("pattern", "unknown")
    bias    = setup.get("bias", "")
    score   = setup.get("composite_score", setup.get("score", 0))
    entry   = setup.get("entry")
    stop    = setup.get("stop")
    t1      = setup.get("t1", setup.get("target"))
    rr      = setup.get("rr", setup.get("risk_reward"))
    verdict = setup.get("ai_verdict", "")

    arrow = "\u2B06" if bias.upper() == "LONG" else "\u2B07" if bias.upper() == "SHORT" else "\u2194"

    # Plain-text (used for Telegram)
    lines = [
        f"{arrow} *New Setup: {symbol}*",
        f"Pattern: {pattern}  |  Bias: {bias}  |  Score: {score:.0f}",
        f"Entry: {_fmt_price(entry)}  Stop: {_fmt_price(stop)}  T1: {_fmt_price(t1)}",
    ]
    if rr is not None:
        lines.append(f"R:R  {float(rr):.1f}")
    if verdict:
        lines.append(f"AI: _{verdict}_")
    message = "\n".join(lines)

    # Discord embed
    fields = [
        {"name": "Pattern", "value": pattern, "inline": True},
        {"name": "Bias",    "value": bias,    "inline": True},
        {"name": "Score",   "value": f"{score:.0f}",  "inline": True},
        {"name": "Entry",   "value": _fmt_price(entry), "inline": True},
        {"name": "Stop",    "value": _fmt_price(stop),  "inline": True},
        {"name": "T1",      "value": _fmt_price(t1),    "inline": True},
    ]
    if rr is not None:
        fields.append({"name": "R:R", "value": f"{float(rr):.1f}", "inline": True})
    if verdict:
        fields.append({"name": "AI Verdict", "value": verdict, "inline": False})

    embed = {
        "title": f"{arrow} {symbol} — New Setup",
        "color": _bias_color(bias),
        "fields": fields,
        "footer": {"text": f"AlphaBean  |  {datetime.now().strftime('%H:%M:%S')}"},
    }

    return message, embed


def format_trade_update(trade: dict, event: str) -> tuple[str, dict]:
    """Format trade status change (T1_HIT, STOPPED, etc).  Returns (message, discord_embed)."""
    symbol  = trade.get("symbol", "???")
    bias    = trade.get("bias", "")
    entry   = trade.get("entry")
    stop    = trade.get("stop")
    pnl_r   = trade.get("pnl_r")
    pattern = trade.get("pattern", "")

    event_upper = event.upper()

    # Choose emoji and colour
    if event_upper in ("T1_HIT", "T2_HIT"):
        icon = "\u2705"
        color = COLOR_GREEN
    elif event_upper == "STOPPED":
        color = COLOR_RED
        icon = "\u274C"
    elif event_upper == "GAP_RISK":
        color = COLOR_BLUE
        icon = "\u26A0\uFE0F"
    else:
        color = COLOR_BLUE
        icon = "\u2139\uFE0F"

    lines = [
        f"{icon} *{symbol}* — {event_upper}",
        f"Pattern: {pattern}  |  Bias: {bias}",
        f"Entry: {_fmt_price(entry)}  Stop: {_fmt_price(stop)}",
    ]
    if pnl_r is not None:
        lines.append(f"P&L: {float(pnl_r):+.1f}R")
    message = "\n".join(lines)

    fields = [
        {"name": "Event",   "value": event_upper, "inline": True},
        {"name": "Pattern", "value": pattern,      "inline": True},
        {"name": "Bias",    "value": bias,          "inline": True},
        {"name": "Entry",   "value": _fmt_price(entry), "inline": True},
        {"name": "Stop",    "value": _fmt_price(stop),  "inline": True},
    ]
    if pnl_r is not None:
        fields.append({"name": "P&L", "value": f"{float(pnl_r):+.1f}R", "inline": True})

    embed = {
        "title": f"{icon} {symbol} — {event_upper}",
        "color": color,
        "fields": fields,
        "footer": {"text": f"AlphaBean  |  {datetime.now().strftime('%H:%M:%S')}"},
    }

    return message, embed


def format_daily_summary(summary: dict) -> tuple[str, dict]:
    """Format end-of-day summary.  Returns (message, discord_embed)."""
    date_str  = summary.get("date", datetime.now().strftime("%Y-%m-%d"))
    setups    = summary.get("setups_detected", 0)
    trades    = summary.get("trades_taken", 0)
    wins      = summary.get("wins", 0)
    losses    = summary.get("losses", 0)
    total_r   = summary.get("total_r", 0.0)
    win_rate  = summary.get("win_rate", 0.0)
    best      = summary.get("best_trade", "—")
    worst     = summary.get("worst_trade", "—")

    color = COLOR_GREEN if total_r >= 0 else COLOR_RED

    lines = [
        f"\U0001F4CA *Daily Summary — {date_str}*",
        f"Setups: {setups}  |  Trades: {trades}",
        f"W/L: {wins}/{losses}  |  Win Rate: {win_rate:.0f}%",
        f"Total P&L: {total_r:+.1f}R",
        f"Best: {best}  |  Worst: {worst}",
    ]
    message = "\n".join(lines)

    fields = [
        {"name": "Setups",   "value": str(setups), "inline": True},
        {"name": "Trades",   "value": str(trades), "inline": True},
        {"name": "Win Rate", "value": f"{win_rate:.0f}%", "inline": True},
        {"name": "Wins",     "value": str(wins),   "inline": True},
        {"name": "Losses",   "value": str(losses), "inline": True},
        {"name": "Total R",  "value": f"{total_r:+.1f}R", "inline": True},
        {"name": "Best",     "value": str(best),   "inline": False},
        {"name": "Worst",    "value": str(worst),  "inline": False},
    ]

    embed = {
        "title": f"\U0001F4CA Daily Summary — {date_str}",
        "color": color,
        "fields": fields,
        "footer": {"text": "AlphaBean"},
    }

    return message, embed


# ── Event Handlers ────────────────────────────────────────────────────────────

def alert_new_setup(setup: dict) -> None:
    """Called when a new high-confidence setup is detected (composite_score >= 70)."""
    try:
        config = get_alert_config()
        min_score = config.get("min_score", 70)
        score = setup.get("composite_score", setup.get("score", 0))
        if score < min_score:
            return

        message, embed = format_new_setup(setup)
        send_alert("new_setup", {"message": message, "embed": embed})
    except Exception as exc:
        log.error("alert_new_setup failed: %s", exc)


def alert_trade_event(trade: dict, event: str) -> None:
    """Called when a tracked trade hits T1, T2, gets stopped, or has gap risk."""
    try:
        event_key = event.lower()
        message, embed = format_trade_update(trade, event)
        send_alert(event_key, {"message": message, "embed": embed})
    except Exception as exc:
        log.error("alert_trade_event failed: %s", exc)


def alert_daily_summary(summary: dict) -> None:
    """Called at end of day with performance summary."""
    try:
        message, embed = format_daily_summary(summary)
        send_alert("daily_summary", {"message": message, "embed": embed})
    except Exception as exc:
        log.error("alert_daily_summary failed: %s", exc)


def test_alerts() -> dict:
    """Send a test message to all configured channels.  Returns results per channel."""
    config = get_alert_config()
    results: dict[str, bool] = {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    test_msg = f"\u2705 AlphaBean alert test — {now}"
    test_embed = {
        "title": "\u2705 Alert Test",
        "description": "If you see this, alerts are working!",
        "color": COLOR_GREEN,
        "footer": {"text": f"AlphaBean  |  {now}"},
    }

    channels = config.get("channels", {})

    dc = channels.get("discord", {})
    if dc.get("enabled") and dc.get("webhook_url"):
        results["discord"] = send_discord(dc["webhook_url"], test_msg, test_embed)
    else:
        results["discord"] = False

    tg = channels.get("telegram", {})
    if tg.get("enabled") and tg.get("bot_token") and tg.get("chat_id"):
        results["telegram"] = send_telegram(tg["bot_token"], tg["chat_id"], test_msg)
    else:
        results["telegram"] = False

    return results
