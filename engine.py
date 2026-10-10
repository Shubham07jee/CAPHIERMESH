"""End-to-end fraud experiment: local baselines, FedAvg, pooled baseline.

Every model uses the SAME transparent full-batch gradient-descent optimizer
and epoch budget so the comparison is about data access, not solver tuning:

- local-only: each bank trains a private model on its own partition.
- federated: rounds of client-local steps from the shared model; only
  clipped/noisy deltas are aggregated (sample-count weighted).
- pooled: one model trained on the union pool (centralized baseline).

The fraud-review threshold is tuned on a VALIDATION split of the bounded
training sample (never on the final held-out test split). All reported
test metrics use the untouched official test split at natural prevalence.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from banks import BankClient, sigmoid, train_local_model
from data import bounded_train_sample, load_frames, partition_banks, train_val_split
from preprocess import (
    TRANSACTION_TYPES, feature_columns, fit_scaler, make_features,
    numeric_columns, validate_frame_columns,
)
from privacy import aggregate_with_gaussian_noise, fedavg, gaussian_epsilon

LABEL = "isFraud"
NEEDED_COLUMNS = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "isFraud", "BankID",
]


@dataclass
class RunConfig:
    """Settings for one comparable local, federated, and pooled run."""

    sample_per_bank: int = 4000
    rounds: int = 20
    local_epochs: int = 5
    learning_rate: float = 0.1
    l2: float = 0.01
    include_balances: bool = False
    use_protection: bool = True
    clip_norm: float = 1.0
    dp_noise_multiplier: float = 0.5
    dp_delta: float = 1e-5
    val_frac: float = 0.3
    seed: int = 42

    def validate(self) -> None:
        if not (1 <= self.sample_per_bank <= 20_000):
            raise ValueError("sample_per_bank must be between 1 and 20000")
        if self.rounds < 1 or self.local_epochs < 1 or self.learning_rate <= 0 or self.l2 < 0:
            raise ValueError("rounds, local_epochs, learning_rate must be positive; l2 nonnegative")
        if not 0 < self.val_frac < 1:
            raise ValueError("val_frac must be between 0 and 1")
        if self.use_protection and not 0 < self.dp_delta < 1:
            raise ValueError("dp_delta must be between 0 and 1 when protection is enabled")
        if self.clip_norm <= 0:
            raise ValueError("clip_norm must be positive")


def _threshold_grid() -> np.ndarray:
    return np.linspace(0.05, 0.95, 19)


def _tune_threshold(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    """Pick the threshold that maximizes F1 on VALIDATION data (not test)."""
    if len(np.unique(y_val)) < 2:
        raise ValueError("Validation set needs both classes to tune a threshold.")
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in _threshold_grid():
        score = f1_score(y_val, (scores_val >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    if best_f1 < 0:
        raise ValueError("Threshold tuning could not find any valid threshold.")
    return best_threshold


def _error_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "test_rows": int(len(y)),
        "fraud_rows": int(np.sum(y)),
        "flagged_rows": int(np.sum(pred)),
    }


def _classification_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    return _error_metrics(y, scores, threshold)


def run_experiment(
    config: RunConfig | None = None,
    progress: Callable[[float, str], None] | None = None,
    ledger: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the configured local / federated / pooled comparison and return results."""
    import ledger as ledger_module

    config = config or RunConfig()
    config.validate()
    started = time.perf_counter()

    def say(frac: float, message: str) -> None:
        if progress:
            progress(frac, message)

    say(0.02, "Loading cached official dataset shards")
    train, test = load_frames(offline=True, columns=NEEDED_COLUMNS)
    if ledger is not None:
        ledger_module.add_event(ledger, "DATASET_LOADED", "Coordinator", {
            "split": "train/test", "train_rows": int(len(train)), "test_rows": int(len(test)),
            "source": "flwrlabs/fed-fraud-paysim-banks"
        })

    say(0.08, "Sampling a bounded, fraud-stratified training set per bank")
    sampled = bounded_train_sample(train, config.sample_per_bank, config.seed)
    available_banks = sorted(int(b) for b in sampled["BankID"].unique())

    say(0.14, "Splitting each bank partition into train + validation")
    bank_train, bank_val = {}, {}
    for bank_id in available_banks:
        subset = sampled[sampled["BankID"] == bank_id]
        tr, va = train_val_split(subset, val_frac=config.val_frac, seed=config.seed + bank_id)
        stable_train, stable_val = tr.copy(), va.copy()
        bank_train[bank_id] = stable_train
        bank_val[bank_id] = stable_val

    validate_frame_columns(sampled, config.include_balances)
    schema = feature_columns(config.include_balances)

    say(0.20, "Building features (fixed order) and fitting the scaler on training only")
    train_frame = pd.concat([bank_train[b] for b in available_banks], ignore_index=True)
    x_train = make_features(train_frame, config.include_balances)
    x_val = make_features(pd.concat([bank_val[b] for b in available_banks], ignore_index=True), config.include_balances)
    x_test = make_features(test, config.include_balances)
    scaler = fit_scaler(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    y_train = train_frame[LABEL].to_numpy(dtype=np.int64)
    y_val = pd.concat([bank_val[b] for b in available_banks], ignore_index=True)[LABEL].to_numpy(dtype=np.int64)
    y_test = test[LABEL].to_numpy(dtype=np.int64)
    bank_ids_test = test["BankID"].to_numpy(dtype=np.int64)
    val_banks = np.concatenate([np.full(len(bank_val[b]), b, dtype=np.int64) for b in available_banks])

    n_features = x_train.shape[1]
    if n_features != len(schema):
        raise ValueError("Feature matrix shape does not match the declared feature schema.")

    if ledger is not None:
        ledger_module.add_event(ledger, "EXPERIMENT_STARTED", "Coordinator", {
            "banks": available_banks, "rounds": config.rounds, "sample_per_bank": config.sample_per_bank,
            "include_balances": config.include_balances,
        })

    # ---------------- pooled baseline (same optimizer, same epoch budget) ----------------
    say(0.30, "Fitting the pooled centralized baseline")
    pooled_w, pooled_b = train_local_model(
        x_train, y_train, None, 0.0, epochs=config.local_epochs, lr=config.learning_rate, l2=config.l2
    )
    pooled_val_scores = sigmoid(x_val @ pooled_w + pooled_b)
    pooled_threshold = _tune_threshold(y_val, pooled_val_scores)
    pooled_test_scores = sigmoid(x_test @ pooled_w + pooled_b)
    pooled_metrics = _classification_metrics(y_test, pooled_test_scores, pooled_threshold)
    if ledger is not None:
        ledger_module.add_event(ledger, "POOLED_MODEL_TRAINED", "Coordinator", {
            "training_rows": int(len(y_train)), "epochs": config.local_epochs,
        })

    # ---------------- local-only baselines ----------------
    say(0.45, "Training a private local-only model per bank")
    local_models: dict[int, tuple[np.ndarray, float]] = {}
    local_val_scores_parts = []
    for bank_id in available_banks:
        x_tr = x_train[bank_ids_train(train_frame, bank_id)]
        y_tr = y_train[bank_ids_train(train_frame, bank_id)]
        w, b = train_local_model(x_tr, y_tr, None, 0.0, epochs=config.local_epochs, lr=config.learning_rate, l2=config.l2)
        local_models[bank_id] = (w, b)
        mask = val_banks == bank_id
        local_val_scores_parts.append(sigmoid(x_val[mask] @ w + b))
        if ledger is not None:
            ledger_module.add_event(ledger, "BANK_LOCAL_MODEL_TRAINED", f"Bank {bank_id}", {
                "training_examples": int(len(y_tr)), "raw_transaction_rows_sent": 0,
            })
    # Align per-bank validation scores with y_val using the concatenated index order.
    local_val_ids = np.concatenate([np.where(val_banks == b)[0] for b in available_banks])
    local_val_scores = np.concatenate(local_val_scores_parts)
    local_threshold = _tune_threshold(y_val[local_val_ids], local_val_scores)
    if not np.isfinite(local_threshold):
        raise ValueError("Local threshold tuning produced an invalid threshold.")

    # ---------------- simulated federated averaging ----------------
    say(0.55, "Simulating FedAvg rounds (clients submit deltas, coordinator aggregates)")
    clients = {
        bank_id: BankClient(bank_id, x_train[bank_ids_train(train_frame, bank_id)],
                            y_train[bank_ids_train(train_frame, bank_id)], schema)
        for bank_id in available_banks
    }
    global_w = np.zeros(n_features, dtype=float)
    global_b = 0.0
    update_count = 0
    raw_rows_sent = 0
    rng = np.random.default_rng(config.seed)
    for round_num in range(1, config.rounds + 1):
        updates = []
        for bank_id in available_banks:
            client_update = clients[bank_id].submit_update(
                global_w, global_b, config.local_epochs, config.learning_rate, config.l2
            )
            norm_delta = float(np.sqrt(np.sum(client_update["delta_w"] ** 2) + client_update["delta_b"] ** 2))
            clipped = bool(config.use_protection and norm_delta > config.clip_norm)
            updates.append(client_update)
            update_count += 1
            raw_rows_sent += client_update["raw_transaction_rows"]
            if ledger is not None:
                ledger_module.add_event(ledger, "MODEL_UPDATE_SUBMITTED", f"Bank {bank_id}", {
                    "round": round_num, "training_examples": client_update["n"],
                    "feature_schema_version": list(schema),
                    "raw_transaction_rows_sent": client_update["raw_transaction_rows"],
                    "update_norm": round(norm_delta, 6), "clipped": clipped,
                })
        if config.use_protection:
            delta_w, delta_b = aggregate_with_gaussian_noise(
                updates, config.clip_norm, config.dp_noise_multiplier, seed=int(rng.integers(0, 2**31 - 1))
            )
            noise_note = f"Gaussian scale per client {config.dp_noise_multiplier:.2f} x clip {config.clip_norm}"
        else:
            delta_w, delta_b = fedavg(updates, (n_features,), schema)
            noise_note = "no clipping, no noise"
        global_w = global_w + delta_w
        global_b = global_b + delta_b
        if ledger is not None:
            ledger_module.add_event(ledger, "FEDAVG_COMPLETED", "Coordinator", {
                "round": round_num, "updates_aggregated": len(updates),
                "weighting": "sample-count-weighted", "noise": noise_note,
                "raw_transaction_rows_received": 0,
            })
        say(0.55 + 0.25 * round_num / config.rounds, f"FedAvg round {round_num}/{config.rounds}")

    federated_val_scores = sigmoid(x_val @ global_w + global_b)
    federated_threshold = _tune_threshold(y_val, federated_val_scores)
    federated_test_scores = sigmoid(x_test @ global_w + global_b)
    federated_metrics = _classification_metrics(y_test, federated_test_scores, federated_threshold)
    if ledger is not None:
        ledger_module.add_event(ledger, "FEDERATED_MODEL_FINALIZED", "Coordinator", {
            "rounds": config.rounds, "raw_transaction_rows_received_total": raw_rows_sent,
        })

    # ---------------- per-bank evaluation on the SAME held-out test split ----------------
    say(0.95, "Evaluating every model on each bank's held-out test rows")
    local_test_scores = np.empty(len(test), dtype=float)
    per_bank: dict[str, Any] = {}
    for bank_id in available_banks:
        mask = bank_ids_test == bank_id
        local_w, local_b = local_models[bank_id]
        local_test_scores[mask] = sigmoid(x_test[mask] @ local_w + local_b)
        per_bank[str(bank_id)] = {
            "local": _classification_metrics(y_test[mask], local_test_scores[mask], local_threshold),
            "federated": _classification_metrics(y_test[mask], federated_test_scores[mask], federated_threshold),
            "pooled": _classification_metrics(y_test[mask], pooled_test_scores[mask], pooled_threshold),
            "test_rows": int(mask.sum()),
            "fraud_rows": int(y_test[mask].sum()),
        }
        if ledger is not None:
            ledger_module.add_event(ledger, "BANK_EVALUATION_COMPLETED", f"Bank {bank_id}", {
                "test_transactions": int(mask.sum()), "raw_transaction_rows_sent": 0
            })

    # ---------------- privacy metadata ----------------
    epsilon = None
    if config.use_protection and config.dp_noise_multiplier > 0:
        epsilon = gaussian_epsilon(config.rounds, config.dp_noise_multiplier, config.dp_delta)
    privacy = {
        "mechanism": "client-level L2 clipping + Gaussian noise on aggregated model deltas",
        "clip_norm": config.clip_norm,
        "noise_multiplier": config.dp_noise_multiplier,
        "weighting": "sample-count-weighted averages",
        "delta": config.dp_delta,
        "epsilon_upper_bound": epsilon,
        "accounting_note": (
            "Conservative unsampled Gaussian RDP composition bound; no secure aggregation, "
            "no sampling amplification. Not a formal production privacy guarantee."
        ) if epsilon else None,
    }

    result: dict[str, Any] = {
        "config": asdict(config),
        "dataset": {
            "dataset_id": "flwrlabs/fed-fraud-paysim-banks",
            "train_rows_total": int(len(train)),
            "test_rows_total": int(len(test)),
            "test_fraud_rows_total": int(y_test.sum()),
            "banks_in_experiment": available_banks,
        },
        "feature_schema": {
            "feature_columns": schema,
            "numeric_columns": numeric_columns(config.include_balances),
            "type_categories": list(TRANSACTION_TYPES),
            "excluded": {
                "target": LABEL,
                "leakage_rule": "isFlaggedFraud",
                "identifiers": ["nameOrig", "nameDest", "BankID"],
                "balances_by_default": list(__import__("preprocess", fromlist=["BALANCES"]).BALANCES),
                "note": "Balance fields are task leakage; included only for the sensitivity run.",
            },
        },
        "thresholds": {"pooled": pooled_threshold, "local": local_threshold, "federated": federated_threshold},
        "metrics": {
            "pooled": pooled_metrics,
            "local": _classification_metrics(y_test, local_test_scores, local_threshold),
            "federated": federated_metrics,
        },
        "per_bank": per_bank,
        "client_rows": {str(bank_id): int(np.sum(bank_ids_train(train_frame, bank_id))) for bank_id in available_banks},
        "evaluation": {
            "threshold_tuning": f"F1 maximized on a {config.val_frac:.0%} validation split of the bounded training sample",
            "test_is_separate_from_training": True,
            "test_rows_total": int(len(test)),
            "test_prevalence": float(y_test.mean()),
            "same_held_out_for_all_models": True,
        },
        "privacy": privacy,
        "pooled_model": {
            "kind": "logistic_regression",
            "numeric_columns": numeric_columns(config.include_balances),
            "type_categories": list(TRANSACTION_TYPES),
            "feature_columns": schema,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": pooled_w.tolist(),
            "intercept": float(pooled_b),
        },
        "raw_transaction_rows_sent_to_coordinator": raw_rows_sent,
        "model_updates_submitted": update_count,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if ledger is not None:
        ledger_module.add_event(ledger, "EXPERIMENT_COMPLETED", "Coordinator", {
            "raw_transaction_rows_sent": raw_rows_sent, "model_updates_submitted": update_count,
            "elapsed_seconds": round(result["elapsed_seconds"], 3),
        })
    say(1.0, "Experiment complete")
    return json.loads(json.dumps(result))


def bank_ids_train(train_frame: pd.DataFrame, bank_id: int) -> np.ndarray:
    """Boolean mask for rows of one bank inside the concatenated training frame."""
    return train_frame["BankID"].to_numpy() == bank_id


def score_transaction(values: dict[str, Any], model: dict[str, Any]) -> float:
    """Score a transaction with a persisted model; validates every input."""
    if not model:
        raise ValueError("The selected run has no persisted trained model.")
    schema = model.get("feature_columns")
    numeric_columns = model.get("numeric_columns")
    type_categories = model.get("type_categories")
    if not schema or not numeric_columns or not type_categories:
        raise ValueError("The persisted model is missing its feature schema.")
    if values.get("type") not in type_categories:
        raise ValueError("Unsupported transaction type.")
    try:
        raw_values = [float(values[column]) for column in numeric_columns]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Required transaction values must be numeric and complete.") from exc
    if not np.all(np.isfinite(raw_values)) or any(value < 0 for value in raw_values):
        raise ValueError("Transaction values must be finite and nonnegative.")
    raw = np.log1p(np.asarray(raw_values))
    categorical = np.array([float(values["type"] == category) for category in type_categories])
    features = np.concatenate((raw, categorical))
    normalized = (features - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logit = float(normalized @ np.asarray(model["coef"]) + model["intercept"])
    return float(sigmoid(np.asarray([logit]))[0])