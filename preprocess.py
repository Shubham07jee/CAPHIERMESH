"""Preprocessing: fixed-order feature engineering shared by every bank.

Design rules (from the integration brief):
- Exclude ``BankID``, ``isFraud`` (target) and ``isFlaggedFraud`` (leakage).
- Exclude customer/account identifiers ``nameOrig`` / ``nameDest``.
- Balance columns are excluded by default because their target-correlated
  accounting arithmetic makes the task near-trivial (measured leakage).
- Every bank uses the SAME feature order and the SAME transformations.
- Any learned scaler is fitted on training data only.

The feature set is ``[log1p(step), log1p(amount)] + one-hot(type)`` by default,
optionally extended with the four balance fields for the leakage sensitivity run:
``[log1p(step), log1p(amount), log1p(x4 balances)] + one-hot(type)``.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

BALANCES: tuple[str, ...] = ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
TRANSACTION_TYPES: tuple[str, ...] = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")

TARGET_COLUMN = "isFraud"
EXCLUDED_COLUMNS: tuple[str, ...] = (
    "nameOrig", "nameDest", "isFlaggedFraud", "BankID", TARGET_COLUMN,
)


def numeric_columns(include_balances: bool = False) -> list[str]:
    return ["step", "amount", *(list(BALANCES) if include_balances else ())]


def feature_columns(include_balances: bool = False) -> list[str]:
    """Exact, stable ordering of all model features."""
    return [*numeric_columns(include_balances), *list(TRANSACTION_TYPES)]


def make_features(frame: object, include_balances: bool = False) -> np.ndarray:
    """Build a float feature matrix. Order is identical for every bank and match.

    Numeric columns are log1p-clipped to nonnegatives; transaction type is
    one-hot encoded with the fixed category order.
    """
    columns = numeric_columns(include_balances)
    numeric = np.log1p(np.clip(frame[columns].to_numpy(np.float64), 0, None))
    transaction_type = frame["type"].astype("string").fillna("").to_numpy()
    categorical = np.column_stack([transaction_type == value for value in TRANSACTION_TYPES]).astype(np.float64)
    return np.column_stack((numeric, categorical))


def fit_scaler(training_x: np.ndarray) -> StandardScaler:
    """Fit the standardizer on pooled TRAINING data only (see README)."""
    return StandardScaler().fit(training_x)


def validate_frame_columns(frame: object, include_balances: bool = False) -> None:
    """Raise if required columns are missing so a bank never trains on a bad schema."""
    required = {"step", "type", "amount", TARGET_COLUMN}
    if include_balances:
        required.update(BALANCES)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")