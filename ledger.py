"""Hash-linked audit ledger for the CIPHERMESH experiment workflow.

Every event stores the sha256 of its canonical JSON body (including the
previous hash), so tampering with any recorded event breaks the chain. The
ledger is a LOCAL simulation log; it is not evidence of a real network capture.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

GENESIS = "GENESIS"


def _canonical(body: dict[str, Any]) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def add_event(ledger: list[dict[str, Any]], event_type: str, actor: str, details: dict[str, Any]) -> dict[str, Any]:
    """Append a hash-linked event to ``ledger`` (in place) and return it."""
    previous_hash = ledger[-1]["hash"] if ledger else GENESIS
    body: dict[str, Any] = {
        "event_id": len(ledger) + 1,
        "timestamp_utc": _now(),
        "event_type": event_type,
        "actor": actor,
        "details": details,
        "previous_hash": previous_hash,
    }
    body["hash"] = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
    ledger.append(body)
    return ledger[-1]


def verify_ledger(ledger: list[dict[str, Any]]) -> bool:
    """Return True only if every hash link, id sequence and body hash is intact."""
    previous = GENESIS
    for index, event in enumerate(ledger, start=1):
        if event.get("event_id") != index or event.get("previous_hash") != previous:
            return False
        body = {key: value for key, value in event.items() if key != "hash"}
        expected = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
        if event.get("hash") != expected:
            return False
        previous = event["hash"]
    return True


def demo_tamper(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of the ledger with one event altered (for the UI demo)."""
    altered = [dict(event) for event in ledger]
    if altered:
        altered[0] = dict(altered[0])
        altered[0]["actor"] = "Modified event"
    return altered