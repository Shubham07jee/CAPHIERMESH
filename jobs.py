"""Background experiment runner so Streamlit shows real progress states.

Jobs run in a worker thread; progress is polled by the UI. Completed runs are
persisted through :mod:`store` so a new app session reloads the artifacts.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

import store
from engine import RunConfig, run_experiment

_LOCK = threading.RLock()
_JOBS: dict[str, dict[str, Any]] = {}
_THREADS: dict[str, threading.Thread] = {}


def start_job(config: dict[str, Any]) -> str:
    """Queue a background experiment; returns a job id."""
    job_id = uuid.uuid4().hex[:8]
    with _LOCK:
        _JOBS[job_id] = {
            "id": job_id, "status": "queued", "progress": 0.0, "message": "Queued",
            "error": None, "result": None, "ledger": None, "saved_at": None, "config": config,
        }

    def worker() -> None:
        ledger: list[dict[str, Any]] = []
        try:
            cfg = RunConfig(**config)

            def update(progress: float, message: str) -> None:
                with _LOCK:
                    if job_id in _JOBS:
                        _JOBS[job_id]["progress"] = float(progress)
                        _JOBS[job_id]["message"] = str(message)

            with _LOCK:
                if job_id in _JOBS:
                    _JOBS[job_id]["status"] = "running"
                    _JOBS[job_id]["message"] = "Starting experiment"
            result = run_experiment(cfg, progress=update, ledger=ledger)
            _, _, saved_at = store.save_run(result, ledger)
            with _LOCK:
                if job_id in _JOBS:
                    _JOBS[job_id].update(
                        status="complete", progress=1.0, message="Complete",
                        result=result, ledger=ledger, saved_at=saved_at,
                    )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            with _LOCK:
                if job_id in _JOBS:
                    _JOBS[job_id].update(status="failed", message="Failed", error=str(exc))
        finally:
            _THREADS.pop(job_id, None)

    thread = threading.Thread(target=worker, name=f"ciphermesh-{job_id}", daemon=True)
    with _LOCK:
        _THREADS[job_id] = thread
    thread.start()
    return job_id


def latest_job() -> dict[str, Any] | None:
    """Return the most recently created job snapshot, if any."""
    with _LOCK:
        if not _JOBS:
            return None
        return dict(_JOBS[sorted(_JOBS.keys())[-1]])


def latest_completed() -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None, str | None]:
    """Return (result, ledger, saved_at) from the latest completed or persisted run."""
    with _LOCK:
        jobs = [dict(job) for job in _JOBS.values() if job["status"] == "complete"]
    for job in sorted(jobs, key=lambda item: item["id"]):
        if job["result"] is not None:
            return job["result"], job["ledger"], job["saved_at"]
    run, info = store.load_latest()
    if info["status"] == "ok" and run is not None:
        return run["result"], run["ledger"], info["saved_at"]
    return None, None, None


def reset() -> None:
    """Clear in-memory job state (used by tests)."""
    with _LOCK:
        _JOBS.clear()
        _THREADS.clear()