"""
simulation/agents/portfolio_manager.py — Portfolio Manager agent.

Receives all analyst verdicts + portfolio state + knowledge context.
Selects which trades to take, allocates position sizes, and provides
reasoning for each decision.
"""
import json

from simulation.agents.base import call_anthropic, AgentResponse
from simulation.config import SimConfig
from simulation.portfolio import PortfolioState


PM_SYSTEM = """\
You are the Portfolio Manager for an autonomous trading simulation.
You receive analyst evaluations and must decide which trades to execute today.

YOUR ROLE:
- Select 0-{max_trades} trades from the evaluated setups
- Consider portfolio heat, correlation, and position limits
- Diversify across patterns and sectors
- Size each trade according to conviction (modifier 0.5x to 1.5x of base risk)
- Prefer CONFIRMED setups with high adjusted scores
- DENIED setups should almost never be selected
- Consider the current portfolio state and avoid overconcentration

RESPOND IN JSON ONLY:
{{
  "selected_trades": [
    {{
      "symbol": "AAPL",
      "pattern": "Bull Flag",
      "size_modifier": 1.0,
      "reasoning": "Strong breakout setup with regime alignment, diversifies from tech-heavy portfolio"
    }}
  ],
  "skipped_reasoning": "Brief explanation of why other setups were not selected",
  "portfolio_assessment": "1-2 sentence assessment of current portfolio state"
}}"""


def _build_pm_prompt(
    verdicts: list[dict],
    portfolio: PortfolioState,
    regime: str = "unknown",
    knowledge_summary: str = "",
) -> str:
    """Build the portfolio manager's decision prompt."""
    lines = ["CURRENT PORTFOLIO STATE:"]
    stats = portfolio.get_stats()
    lines.append(f"  Capital: ${portfolio.cash:,.0f}")
    lines.append(f"  Open positions: {len(portfolio.positions)}/{portfolio.config.max_concurrent_positions}")
    lines.append(f"  Portfolio heat: {portfolio.total_heat_pct:.1f}% (max {portfolio.config.max_portfolio_heat_pct}%)")
    lines.append(f"  Cumulative R: {portfolio.cumulative_r:+.2f}")
    lines.append(f"  Market regime: {regime}")

    if portfolio.positions:
        lines.append(f"\n  Open symbols: {', '.join(p.symbol for p in portfolio.positions)}")

    if stats.get("total_trades", 0) > 0:
        lines.append(f"\n  Track record: {stats['total_trades']} trades, "
                      f"{stats['win_rate']:.0f}% win rate, {stats['avg_r']:+.3f} avg R")

    lines.append(f"\nTODAY'S EVALUATED SETUPS ({len(verdicts)}):")
    for v in verdicts[:15]:  # Cap at 15 to keep prompt manageable
        lines.append(
            f"  [{v.get('verdict', 'CAUTION')}] {v['symbol']} — {v['pattern']} | "
            f"Score: {v.get('adjusted_score', 0):.0f} | "
            f"Confidence: {v.get('confidence', 0.5):.0%}"
        )
        reasoning = v.get("reasoning", "")
        if reasoning:
            lines.append(f"    Reasoning: {reasoning[:200]}")

    if knowledge_summary:
        lines.append(f"\nLEARNED KNOWLEDGE:\n{knowledge_summary[:1500]}")

    return "\n".join(lines)


async def select_trades(
    verdicts: list[dict],
    portfolio: PortfolioState,
    config: SimConfig,
    regime: str = "unknown",
    knowledge_summary: str = "",
) -> list[dict]:
    """Use PM agent to select trades from evaluated setups.

    Returns list of selected trade dicts with symbol, pattern, size_modifier, reasoning.
    """
    if not verdicts:
        return []

    system = PM_SYSTEM.format(max_trades=config.max_trades_per_day)
    prompt = _build_pm_prompt(verdicts, portfolio, regime, knowledge_summary)

    resp = await call_anthropic(
        prompt=prompt,
        system=system,
        model=config.pm_model,
        max_tokens=1024,
        temperature=0.3,
    )

    if not resp.success:
        # Fallback: select top CONFIRMED setups deterministically
        confirmed = [v for v in verdicts if v.get("verdict") == "CONFIRMED"]
        if not confirmed:
            confirmed = [v for v in verdicts if v.get("verdict") != "DENIED"]
        return [
            {"symbol": v["symbol"], "pattern": v["pattern"],
             "size_modifier": 1.0, "reasoning": "PM fallback — deterministic"}
            for v in confirmed[:config.max_trades_per_day]
        ]

    selected = resp.data.get("selected_trades", [])

    # Validate selections exist in verdicts
    valid_symbols = {(v["symbol"], v["pattern"]) for v in verdicts}
    validated = []
    for s in selected:
        key = (s.get("symbol", ""), s.get("pattern", ""))
        if key in valid_symbols:
            s["size_modifier"] = max(0.5, min(1.5, s.get("size_modifier", 1.0)))
            validated.append(s)

    return validated[:config.max_trades_per_day]
