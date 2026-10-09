"""Sample-count-weighted FedAvg and client-level Gaussian privacy controls.

These helpers are a transparent local simulation. Clipping plus Gaussian noise
does NOT by itself certify a formal differential-privacy guarantee; the
``gaussian_epsilon`` number is a conservative unsampled RDP composition bound
that we report only as an estimate for the configured mechanism.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def clip_client_update(update: np.ndarray, clip_norm: float) -> np.ndarray:
    """L2-clip a flattened client update to the configured sensitivity bound."""
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    if not np.all(np.isfinite(update)):
        raise ValueError("Cannot clip a non-finite update.")
    norm = float(np.linalg.norm(update))
    return update * min(1.0, clip_norm / max(norm, 1e-12))


def validate_updates(updates: list[dict[str, Any]], expected_shape: tuple[int, ...], schema: list[str]) -> None:
    """Reject empty, incompatible, non-finite or mis-schematized client updates."""
    if not updates:
        raise ValueError("No client updates received for aggregation.")
    for item in updates:
        w = item.get("delta_w")
        if w is None or not isinstance(w, np.ndarray):
            raise ValueError(f"Update from bank {item.get('bank')} is missing delta_w.")
        if w.shape != expected_shape:
            raise ValueError(
                f"Client schema mismatch: expected {expected_shape}, got {w.shape} "
                f"from bank {item.get('bank')}."
            )
        if not (np.all(np.isfinite(w)) and np.isfinite(float(item.get("delta_b", float("nan"))))):
            raise ValueError(f"Update from bank {item.get('bank')} is non-finite.")
        n = item.get("n", 0)
        if int(n) <= 0:
            raise ValueError(f"Update from bank {item.get('bank')} has a non-positive weight.")
        if list(item.get("feature_schema", [])) != schema:
            raise ValueError(f"Feature schema mismatch from bank {item.get('bank')}.")


def fedavg(updates: list[dict[str, Any]], expected_shape: tuple[int, ...], schema: list[str]) -> tuple[np.ndarray, float]:
    """Sample-count-weighted FedAvg over validated parameter updates."""
    validate_updates(updates, expected_shape, schema)
    weights = np.asarray([float(item["n"]) for item in updates], dtype=float)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Total training samples for aggregation must be positive.")
    proportions = weights / total
    delta_w = np.zeros(expected_shape, dtype=float)
    delta_b = 0.0
    for item, weight in zip(updates, proportions):
        delta_w += weight * item["delta_w"]
        delta_b += weight * float(item["delta_b"])
    if not np.all(np.isfinite(delta_w)) or not np.isfinite(delta_b):
        raise ValueError("Aggregation produced non-finite parameters.")
    return delta_w, float(delta_b)


def aggregate_with_gaussian_noise(
    updates: list[dict[str, Any]],
    clip_norm: float,
    noise_multiplier: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Average L2-clipped deltas (sample-count weighted) and add Gaussian noise.

    Noise scale uses the weighted-average sensitivity: with weights ``w_i``
    summing to one, the L2 sensitivity of the weighted mean of clipped vectors
    is ``clip_norm * sqrt(sum(w_i^2))``.
    """
    if not updates:
        raise ValueError("At least one client update is required")
    if noise_multiplier < 0:
        raise ValueError("noise_multiplier cannot be negative")
    weights = np.asarray([float(item["n"]) for item in updates], dtype=float)
    proportions = weights / weights.sum()
    clipped: list[np.ndarray] = []
    delta_b = 0.0
    for item, weight in zip(updates, proportions):
        clipped_w = clip_client_update(item["delta_w"], clip_norm)
        clipped_b = float(item["delta_b"]) * min(1.0, clip_norm / max(np.abs(float(item["delta_b"])) + 1e-12, 1e-12))
        clipped.append(weight * clipped_w)
        delta_b += weight * clipped_b
    average = np.stack(clipped).sum(axis=0)
    if noise_multiplier:
        rng = np.random.default_rng(seed)
        scale = noise_multiplier * clip_norm * float(np.sqrt(np.sum(proportions ** 2)))
        average = average + rng.normal(0.0, scale, size=average.shape)
        delta_b = float(delta_b + rng.normal(0.0, scale))
    return average, float(delta_b)


def gaussian_epsilon(rounds: int, noise_multiplier: float, delta: float) -> float:
    """Conservative basic RDP composition bound for unsampled Gaussian releases."""
    if rounds < 1 or noise_multiplier <= 0 or not 0 < delta < 1:
        raise ValueError("Require rounds >= 1, noise_multiplier > 0, and 0 < delta < 1")
    orders = np.array([2, 3, 4, 8, 16, 32, 64], dtype=float)
    epsilon = rounds * orders / (2 * noise_multiplier**2) + np.log(1 / delta) / (orders - 1)
    return float(np.min(epsilon))