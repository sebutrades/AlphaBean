"""
simulation/custom/run_store.py — Persistent run logging and comparison.

Every run gets a unique ID and is saved to disk with:
  - Full config snapshot
  - All closed trades
  - Equity curve
  - Agent reasoning logs
  - Final statistics

Runs are never overwritten — they accumulate for comparison.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


RUNS_DIR = Path("simulation/output/custom_runs")


def _ensure_dir():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def create_run_id() -> str:
    """Generate a unique run ID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"run_{ts}_{short_uuid}"


def save_run(
    run_id: str,
    config: dict,
    stats: dict,
    equity_curve: list[dict],
    closed_trades: list[dict],
    agent_logs: list[dict],
    strategy_summary: list[dict],
    status: str = "completed",
) -> Path:
    """Save a complete run to disk.

    Returns the path to the saved file.
    """
    _ensure_dir()

    run_data = {
        "run_id": run_id,
        "status": status,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "stats": stats,
        "equity_curve": equity_curve,
        "closed_trades": closed_trades,
        "agent_logs": agent_logs,
        "strategy_summary": strategy_summary,
    }

    out_path = RUNS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps(run_data, indent=2, default=str))
    return out_path


def update_run_status(run_id: str, status: str, stats: Optional[dict] = None):
    """Update the status of an existing run (e.g., running → completed)."""
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data["status"] = status
    data["updated_at"] = datetime.now().isoformat()
    if stats:
        data["stats"] = stats
    path.write_text(json.dumps(data, indent=2, default=str))


def save_run_progress(
    run_id: str,
    day_number: int,
    total_days: int,
    equity_curve: list[dict],
    closed_trades: list[dict],
    agent_logs: list[dict],
    strategy_summary: list[dict],
    stats: dict,
):
    """Save intermediate progress (called periodically during long runs)."""
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data["status"] = "running"
    data["progress"] = {"day": day_number, "total": total_days}
    data["stats"] = stats
    data["equity_curve"] = equity_curve
    data["closed_trades"] = closed_trades
    data["agent_logs"] = agent_logs[-100:]  # keep last 100 to avoid huge files
    data["strategy_summary"] = strategy_summary
    data["updated_at"] = datetime.now().isoformat()
    path.write_text(json.dumps(data, indent=2, default=str))


def load_run(run_id: str) -> Optional[dict]:
    """Load a specific run by ID."""
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_runs() -> list[dict]:
    """List all saved runs with summary info (no full trade data)."""
    _ensure_dir()
    runs = []
    for path in sorted(RUNS_DIR.glob("run_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            runs.append({
                "run_id": data.get("run_id", path.stem),
                "name": data.get("config", {}).get("name", ""),
                "status": data.get("status", "unknown"),
                "created_at": data.get("created_at", ""),
                "progress": data.get("progress"),
                "config_summary": {
                    "starting_capital": data.get("config", {}).get("starting_capital", 0),
                    "use_agents": data.get("config", {}).get("use_agents", False),
                    "deliberation_mode": data.get("config", {}).get("deliberation", {}).get("mode", ""),
                    "strategies": len(data.get("config", {}).get("allowed_strategies", [])) or "all",
                    "sizing_mode": data.get("config", {}).get("sizing", {}).get("mode", ""),
                },
                "stats": data.get("stats", {}),
            })
        except Exception:
            continue
    return runs


def delete_run(run_id: str) -> bool:
    """Delete a run from disk."""
    path = RUNS_DIR / f"{run_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def compare_runs(run_ids: list[str]) -> dict:
    """Compare multiple runs side-by-side."""
    runs = []
    for rid in run_ids:
        data = load_run(rid)
        if data:
            runs.append({
                "run_id": rid,
                "name": data.get("config", {}).get("name", rid),
                "config": data.get("config", {}),
                "stats": data.get("stats", {}),
                "equity_curve": data.get("equity_curve", []),
                "strategy_summary": data.get("strategy_summary", []),
            })
    return {"runs": runs, "compared_at": datetime.now().isoformat()}
