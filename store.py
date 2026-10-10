"""Atomic persistence and safe reloading of experiment artifacts.

Artifacts live under a gitignored ``artifacts/`` directory:

    artifacts/latest.json              -> current complete run (model + ledger + metrics)
    artifacts/experiments/run_<ts>.json -> immutable timestamped copies

Writes are atomic (temp file + os.replace). Loading tolerates corrupt or
incompatible files and reports the reason instead of fabricating a result.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(os.environ.get("CIPHERMESH_ARTIFACT_DIR", Path(__file__).parent / "artifacts"))
MODEL_KIND = "ciphermesh_model_v1"
RESULT_REQUIRED_KEYS = ("config", "metrics", "per_bank", "privacy", "pooled_model", "feature_schema")


def _atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=target.stem + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_run(result: dict[str, Any], ledger: list[dict[str, Any]], artifacts_dir: Path | None = None) -> tuple[Path, Path, str]:
    """Atomically persist a completed run. Returns (latest_path, run_path, saved_at)."""
    if not result.get("metrics") or not result.get("pooled_model"):
        raise ValueError("Refusing to save a run without real metrics and a trained model.")
    if not ledger:
        raise ValueError("Refusing to save a run without an audit ledger.")
    if not result.get("feature_schema"):
        raise ValueError("Refusing to save a run without a feature schema.")
    artifacts_dir = artifacts_dir or ARTIFACT_DIR
    saved_at = _run_timestamp()
    payload = {
        "kind": MODEL_KIND,
        "saved_at": saved_at,
        "result": result,
        "ledger": ledger,
    }
    runs_dir = artifacts_dir / "experiments"
    latest = artifacts_dir / "latest.json"
    run_file = runs_dir / f"run_{saved_at}.json"
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    _atomic_write_text(run_file, serialized)
    _atomic_write_text(latest, serialized)
    return latest, run_file, saved_at


def _saved_at_from_payload(payload: dict[str, Any]) -> str:
    return payload.get("saved_at", "unknown")


def load_latest(artifacts_dir: Path | None = None) -> tuple[dict[str, Any] | None, dict[str, str]]:
    """Load ``artifacts/latest.json``. Returns (run_or_none, info)."""
    artifacts_dir = artifacts_dir or ARTIFACT_DIR
    latest = artifacts_dir / "latest.json"
    if not latest.is_file():
        return None, {"status": "no_artifacts", "message": "No saved run found — run an experiment first."}
    return _load_run_file(latest, "latest")

def _load_run_file(path: Path, label: str) -> tuple[dict[str, Any] | None, dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return None, {"status": "corrupt", "message": f"{label} artifact is corrupt or unreadable: {exc}."}
    if payload.get("kind") != MODEL_KIND:
        return None, {"status": "incompatible", "message": f"{label} artifact has an unknown model kind."}
    result = payload.get("result", {})
    missing = [key for key in RESULT_REQUIRED_KEYS if key not in result]
    if missing:
        return None, {"status": "incompatible", "message": f"{label} artifact is missing required sections: {missing}."}
    if not isinstance(payload.get("ledger"), list) or not payload["ledger"]:
        return None, {"status": "incompatible", "message": f"{label} artifact is missing its audit ledger."}
    return payload, {"status": "ok", "saved_at": _saved_at_from_payload(payload), "source": label}


def list_runs(artifacts_dir: Path | None = None) -> list[str]:
    artifacts_dir = artifacts_dir or ARTIFACT_DIR
    runs_dir = artifacts_dir / "experiments"
    if not runs_dir.is_dir():
        return []
    return sorted((path.name for path in runs_dir.glob("run_*.json")), reverse=True)