"""
simulation — Autonomous Trading Simulation & Continuous Learning System

Replays historical bar data day-by-day with AI agents making all trading
decisions: what to trade, how much, when to exit.  Agents learn from
outcomes and compound knowledge over time.

Usage:
    python -m simulation.run --days 180
    python -m simulation.run --days 10 --no-agents   # deterministic mode
"""
