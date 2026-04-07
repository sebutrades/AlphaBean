"""
simulation/agents/risk_manager.py — Risk Manager agent.

Reviews PM's trade selections against portfolio constraints.
Can APPROVE, REDUCE (size), or REJECT each trade.
Enforces heat limits, correlation rules, and drawdown protection.
"""
import json

from simulation.agents.base import call_anthropic, AgentResponse
from simulation.config import SimConfig
from simulation.portfolio import PortfolioState


RISK_SYSTEM = """\
You are the Risk Manager for an autonomous trading simulation.
Your job is to protect capital and enforce risk rules on each proposed trade.

RULES YOU ENFORCE:
1. Portfolio heat must not exceed {max_heat}%
2. Max {max_positions} concurrent positions
3. No more than {max_correlated} positions in the same sector or closely correlated stocks
4. Reduce size during drawdowns (if cumulative R < -3, max 0.75x; if < -5, max 0.5x)
5. Reject trades in symbols we already hold a position in
6. Flag any trade where risk/reward < 1.5

For each trade, decide: APPROVE / REDUCE / REJECT

RESPOND IN JSON ONLY:
{{
  "decisions": [
    {{
      "symbol": "AAPL",
      "pattern": "Bull Flag",
      "action": "APPROVE",
      "size_modifier": 1.0,
      "reasoning": "Within heat limits, good R:R, no correlation conflict"
    }}
  ],
  "portfolio_risk_note": "1-2 sentence risk assessment"
}}"""


def _build_risk_prompt(
    pm_selections: list[dict],
    portfolio: PortfolioState,
    regime: str = "unknown",
) -> str:
    """Build the risk review prompt."""
    lines = ["PORTFOLIO STATE:"]
    lines.append(f"  Heat: {portfolio.total_heat_pct:.1f}% / {portfolio.config.max_portfolio_heat_pct}%")
    lines.append(f"  Positions: {len(portfolio.positions)} / {portfolio.config.max_concurrent_positions}")
    lines.append(f"  Cumulative R: {portfolio.cumulative_r:+.2f}")
    lines.append(f"  Regime: {regime}")
    lines.append(f"  Remaining risk budget: ${portfolio.remaining_risk_budget:,.0f}")

    if portfolio.positions:
        lines.append(f"\n  Current positions: {', '.join(p.symbol for p in portfolio.positions)}")

    lines.append(f"\nPROPOSED TRADES ({len(pm_selections)}):")
    for s in pm_selections:
        lines.append(
            f"  {s['symbol']} — {s['pattern']} | Size: {s.get('size_modifier', 1.0):.1f}x"
        )
        if s.get("reasoning"):
            lines.append(f"    PM reasoning: {s['reasoning'][:200]}")

    return "\n".join(lines)


async def review_trades(
    pm_selections: list[dict],
    portfolio: PortfolioState,
    config: SimConfig,
    regime: str = "unknown",
) -> list[dict]:
    """Review PM selections through risk lens.

    Returns list of approved trades with possibly modified size_modifier.
    """
    if not pm_selections:
        return []

    system = RISK_SYSTEM.format(
        max_heat=config.max_portfolio_heat_pct,
        max_positions=config.max_concurrent_positions,
        max_correlated=config.max_correlated_positions,
    )
    prompt = _build_risk_prompt(pm_selections, portfolio, regime)

    resp = await call_anthropic(
        prompt=prompt,
        system=system,
        model=config.risk_model,
        max_tokens=1024,
        temperature=0.2,
    )

    if not resp.success:
        # Fallback: apply mechanical risk rules
        return _mechanical_risk_check(pm_selections, portfolio, config)

    decisions = resp.data.get("decisions", [])

    # Match decisions back to selections
    approved = []
    decision_map = {d.get("symbol", ""): d for d in decisions}

    for sel in pm_selections:
        dec = decision_map.get(sel["symbol"], {})
        action = dec.get("action", "APPROVE").upper()

        if action == "REJECT":
            continue
        elif action == "REDUCE":
            sel["size_modifier"] = max(0.25, min(
                sel.get("size_modifier", 1.0),
                dec.get("size_modifier", 0.5),
            ))

        sel["risk_reasoning"] = dec.get("reasoning", "")
        approved.append(sel)

    return approved


def _mechanical_risk_check(
    selections: list[dict],
    portfolio: PortfolioState,
    config: SimConfig,
) -> list[dict]:
    """Fallback risk rules when agent call fails."""
    existing_symbols = {p.symbol for p in portfolio.positions}
    approved = []

    for sel in selections:
        # No duplicate positions
        if sel["symbol"] in existing_symbols:
            continue

        # Heat check
        if not portfolio.can_add_trade:
            break

        # Drawdown scaling
        if portfolio.cumulative_r < -5:
            sel["size_modifier"] = min(sel.get("size_modifier", 1.0), 0.5)
        elif portfolio.cumulative_r < -3:
            sel["size_modifier"] = min(sel.get("size_modifier", 1.0), 0.75)

        sel["risk_reasoning"] = "mechanical risk check (agent fallback)"
        approved.append(sel)
        existing_symbols.add(sel["symbol"])

    return approved
