"""Simulated bank clients and per-client model training.

Each client owns ONLY its own local partition and returns, through the
aggregation interface, model parameters plus approved metadata — never raw
transaction rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_local_model(
    x: np.ndarray,
    y: np.ndarray,
    initial_w: np.ndarray | None = None,
    initial_b: float = 0.0,
    epochs: int = 5,
    lr: float = 0.1,
    l2: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Small transparent logistic-regression trainer (full-batch gradient descent).

    ``x`` and ``y`` must be finite and non-empty. Returns the fitted weights
    and intercept. Everything else (feature engineering, scaling) is done
    before this function is called.
    """
    n, features = x.shape
    if n == 0:
        raise ValueError("Cannot train on an empty training set.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Training inputs contain non-finite values.")
    w = np.zeros(features, dtype=float) if initial_w is None else np.asarray(initial_w, dtype=float).copy()
    if w.shape != (features,):
        raise ValueError(f"Expected initial weights of shape ({features},), got {w.shape}.")
    if not all(np.isfinite(w)):
        raise ValueError("Initial weights contain non-finite values.")
    b = float(initial_b)
    for _ in range(epochs):
        p = sigmoid(x @ w + b)
        error = p - y
        grad_w = (x.T @ error) / n + l2 * w
        grad_b = float(np.mean(error))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, float(b)


@dataclass
class BankClient:
    """A simulated bank holding its own local training data only."""

    bank_id: int
    x_train: np.ndarray
    y_train: np.ndarray
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.x_train.shape[0] == 0:
            raise ValueError(f"Bank {self.bank_id} has an empty training set.")
        if self.x_train.shape[0] != len(self.y_train):
            raise ValueError(f"Bank {self.bank_id} training rows do not match its labels.")
        if not np.all(np.isfinite(self.x_train)) or not np.all(np.isfinite(self.y_train)):
            raise ValueError(f"Bank {self.bank_id} training data contains non-finite values.")

    @property
    def n(self) -> int:
        return int(self.x_train.shape[0])

    def submit_update(
        self,
        global_w: np.ndarray,
        global_b: float,
        epochs: int,
        lr: float,
        l2: float = 0.01,
    ) -> dict[str, Any]:
        """Train locally from the shared model and return ONLY the update + metadata."""
        local_w, local_b = train_local_model(
            self.x_train, self.y_train, global_w, global_b, epochs, lr, l2
        )
        delta_w = local_w - np.asarray(global_w)
        delta_b = local_b - global_b
        if not np.all(np.isfinite(delta_w)) or not np.isfinite(delta_b):
            raise ValueError(f"Bank {self.bank_id} produced a non-finite model update.")
        return {
            "bank": self.bank_id,
            "n": self.n,
            "feature_schema": list(self.feature_names),
            "delta_w": delta_w,
            "delta_b": float(delta_b),
            "raw_transaction_rows": 0,
        }

    def fit_local(self, epochs: int, lr: float, l2: float = 0.01) -> tuple[np.ndarray, float]:
        """Train a private local-only model from scratch (upper-left baseline)."""
        return train_local_model(self.x_train, self.y_train, None, 0.0, epochs, lr, l2)