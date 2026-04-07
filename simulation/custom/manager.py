"""
simulation/custom/manager.py — Background run manager.

Singleton that manages simulation runs independently of WebSocket connections.
Navigate away, come back, reconnect — runs keep going.

Usage:
    manager = RunManager.instance()
    run_id = await manager.start_run(config)
    queue = manager.subscribe(run_id)  # get live events
    manager.unsubscribe(run_id, queue)  # disconnect without stopping
    manager.stop_run(run_id)            # actually stop
"""
import asyncio
from typing import Optional

from simulation.custom.config import CustomSimConfig
from simulation.custom.engine import CustomSimEngine, get_available_dates
from simulation.custom.run_store import create_run_id, list_runs, load_run, delete_run, compare_runs
from simulation.custom.strategy_filter import get_all_strategies, get_strategy_groups


class RunManager:
    """Manages simulation runs as background tasks."""

    _instance: Optional["RunManager"] = None

    def __init__(self):
        self._runs: dict[str, CustomSimEngine] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    @classmethod
    def instance(cls) -> "RunManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start_run(self, config: CustomSimConfig) -> str:
        """Start a new simulation run in the background.

        Returns the run_id. The run continues even if no WebSocket is connected.
        """
        run_id = create_run_id()

        # Fill in dates if not provided
        if not config.dates:
            config.dates = get_available_dates()

        engine = CustomSimEngine(config, run_id)
        self._runs[run_id] = engine

        # Launch as background task
        task = asyncio.create_task(self._run_wrapper(run_id, engine))
        self._tasks[run_id] = task

        return run_id

    async def _run_wrapper(self, run_id: str, engine: CustomSimEngine):
        """Wrapper to handle task completion and cleanup."""
        try:
            await engine.run()
        except asyncio.CancelledError:
            engine.status = "stopped"
        except Exception as e:
            engine.status = "error"
            engine._emit({"type": "error", "message": str(e)})
        finally:
            # Keep in _runs for status queries, but remove task
            self._tasks.pop(run_id, None)

    def stop_run(self, run_id: str) -> bool:
        """Stop a running simulation."""
        engine = self._runs.get(run_id)
        if not engine:
            return False
        engine.stop()
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        return True

    def pause_run(self, run_id: str) -> bool:
        engine = self._runs.get(run_id)
        if not engine:
            return False
        engine.pause()
        return True

    def resume_run(self, run_id: str) -> bool:
        engine = self._runs.get(run_id)
        if not engine:
            return False
        engine.resume()
        return True

    def set_speed(self, run_id: str, speed: float) -> bool:
        engine = self._runs.get(run_id)
        if not engine:
            return False
        engine.set_speed(speed)
        return True

    def subscribe(self, run_id: str) -> Optional[asyncio.Queue]:
        """Subscribe to live events from a run.

        Returns an asyncio.Queue that receives events.
        Returns None if run doesn't exist.
        """
        engine = self._runs.get(run_id)
        if not engine:
            return None
        return engine.add_subscriber()

    def unsubscribe(self, run_id: str, queue: asyncio.Queue):
        """Remove a subscriber without stopping the run."""
        engine = self._runs.get(run_id)
        if engine:
            engine.remove_subscriber(queue)

    def get_run_status(self, run_id: str) -> Optional[dict]:
        """Get current status of a run."""
        engine = self._runs.get(run_id)
        if not engine:
            # Check disk
            saved = load_run(run_id)
            if saved:
                return {
                    "run_id": run_id,
                    "status": saved.get("status", "completed"),
                    "stats": saved.get("stats", {}),
                    "config": saved.get("config", {}),
                    "live": False,
                }
            return None

        return {
            "run_id": run_id,
            "status": engine.status,
            "stats": engine._get_stats(),
            "config": engine.config.to_dict(),
            "day_number": engine._day_number,
            "positions": len(engine.positions),
            "live": True,
        }

    def get_active_runs(self) -> list[dict]:
        """List all currently active (in-memory) runs."""
        return [
            {
                "run_id": rid,
                "status": engine.status,
                "day_number": engine._day_number,
                "stats": engine._get_stats(),
                "config_name": engine.config.name,
            }
            for rid, engine in self._runs.items()
            if engine.status in ("running", "paused", "pending")
        ]

    def cleanup_finished(self):
        """Remove completed/stopped runs from memory (they're saved to disk)."""
        to_remove = [
            rid for rid, engine in self._runs.items()
            if engine.status in ("completed", "stopped", "error")
        ]
        for rid in to_remove:
            self._runs.pop(rid, None)
            self._tasks.pop(rid, None)

    # ── Static Data ──────────────────────────────────────────────────────────

    @staticmethod
    def get_available_dates() -> list[str]:
        return get_available_dates()

    @staticmethod
    def get_strategies() -> list[dict]:
        return get_all_strategies()

    @staticmethod
    def get_strategy_groups() -> dict[str, list[str]]:
        return get_strategy_groups()

    @staticmethod
    def get_saved_runs() -> list[dict]:
        return list_runs()

    @staticmethod
    def get_saved_run(run_id: str) -> Optional[dict]:
        return load_run(run_id)

    @staticmethod
    def delete_saved_run(run_id: str) -> bool:
        return delete_run(run_id)

    @staticmethod
    def compare_saved_runs(run_ids: list[str]) -> dict:
        return compare_runs(run_ids)
