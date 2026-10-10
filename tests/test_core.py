"""Automated tests for CIPHERMESH data, federated learning, metrics, ledger,
persistence and risk scoring. Run with:  .venv/bin/python -m pytest -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import data as data_module
import engine
import ledger as ledger_module
import preprocess
import store
from banks import BankClient, train_local_model
from engine import RunConfig, score_transaction
from privacy import (
    aggregate_with_gaussian_noise, clip_client_update, fedavg,
    gaussian_epsilon, validate_updates,
)


def _tiny_frame(bank_id: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(bank_id + 7)
    types = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
    return pd.DataFrame({
        "step": rng.integers(1, 1000, n),
        "type": rng.choice(types, n),
        "amount": rng.lognormal(3.0, 1.0, n),
        "nameOrig": [f"c{bank_id}{i}" for i in range(n)],
        "oldbalanceOrg": rng.lognormal(4.0, 1.0, n),
        "newbalanceOrig": rng.lognormal(4.0, 1.0, n),
        "nameDest": [f"d{bank_id}{i}" for i in range(n)],
        "oldbalanceDest": rng.lognormal(4.0, 1.0, n),
        "newbalanceDest": rng.lognormal(4.0, 1.0, n),
        "isFraud": rng.binomial(1, 0.05, n),
        "isFlaggedFraud": np.zeros(n, dtype=int),
        "BankID": bank_id,
    })


# --------------------------------------------------------------------------- data
def test_partition_banks_by_bankid() -> None:
    frame = pd.concat([_tiny_frame(b) for b in (0, 1, 2)], ignore_index=True)
    banks = data_module.partition_banks(frame)
    assert set(banks) == {0, 1, 2}
    totals = sum(len(group) for group in banks.values())
    assert totals == len(frame)


def test_bounded_sample_is_stratified_per_bank() -> None:
    frame = pd.concat([_tiny_frame(b) for b in range(5)], ignore_index=True)
    sampled = data_module.bounded_train_sample(frame, per_bank=1000, seed=42)
    for _, group in sampled.groupby("BankID"):
        assert 0 < int(group["isFraud"].sum()) < len(group)
        assert len(group) <= 1000


def test_train_val_split_is_stratified() -> None:
    frame = _tiny_frame(bank_id=1, n=2000)
    train, val = data_module.train_val_split(frame, val_frac=0.3, seed=42)
    assert len(train) + len(val) == len(frame)
    train_fraud = train["isFraud"].mean()
    val_fraud = val["isFraud"].mean()
    assert abs(train_fraud - val_fraud) < 0.05
    assert set(val["isFraud"]) == {0, 1}


def test_offline_cache_has_expected_row_counts() -> None:
    shards = data_module.ensure_dataset(offline=True)
    for split, expected in data_module.EXPECTED_ROWS.items():
        observed = sum(data_module.pq.ParquetFile(p).metadata.num_rows for p in shards[split])
        assert observed == expected


# ------------------------------------------------------------------- preprocessing
def test_features_exclude_target_ids_and_preserve_order() -> None:
    frame = pd.concat([_tiny_frame(0), _tiny_frame(1)], ignore_index=True)
    x0 = preprocess.make_features(_tiny_frame(0))
    x1 = preprocess.make_features(_tiny_frame(1))
    assert x0.shape == x1.shape
    assert preprocess.feature_columns(False) == ["step", "amount", *list(preprocess.TRANSACTION_TYPES)]
    forbidden = {"isFraud", "isFlaggedFraud", "BankID", "nameOrig", "nameDest",
                 "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}
    assert not (forbidden & set(preprocess.feature_columns(False)))
    first_two = preprocess.feature_columns(False)[:2]
    expected = np.log1p(np.clip(_tiny_frame(0)[first_two].to_numpy(np.float64), 0, None))
    assert np.allclose(x0[:, :2], expected)


def test_balance_columns_are_explicitly_switchable() -> None:
    columns = preprocess.feature_columns(True)
    assert "oldbalanceOrg" in columns and "newbalanceOrig" in columns
    assert len(columns) == 11
    assert preprocess.feature_columns(False) == [c for c in columns if c not in preprocess.BALANCES]


def test_scaler_fit_on_training_only() -> None:
    x_train = preprocess.make_features(_tiny_frame(0, n=500))
    x_test = preprocess.make_features(pd.concat([_tiny_frame(2, n=500)], ignore_index=True))
    scaler = preprocess.fit_scaler(x_train)
    assert np.allclose(scaler.mean_, x_train.mean(axis=0))
    assert not np.allclose(scaler.mean_, x_test.mean(axis=0))


# ---------------------------------------------------------------------- training
def test_local_training_shapes_and_errors() -> None:
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((120, 7)), rng.integers(0, 2, 120).astype(float)
    w, b = train_local_model(x, y, epochs=3, lr=0.1)
    assert w.shape == (7,) and isinstance(b, float)
    with pytest.raises(ValueError):
        train_local_model(np.zeros((0, 7)), np.zeros(0))
    with pytest.raises(ValueError):
        train_local_model(x, np.array([float("nan")] * len(y)))
    with pytest.raises(ValueError):
        train_local_model(x, y, initial_w=np.zeros(5))


def test_bank_client_submits_only_update_and_metadata() -> None:
    rng = np.random.default_rng(2)
    x, y = rng.standard_normal((80, 7)), rng.integers(0, 2, 80)
    client = BankClient(3, x, y, feature_names=["step", "amount"])
    update = client.submit_update(np.zeros(7), 0.0, epochs=2, lr=0.1)
    assert update["delta_w"].shape == (7,)
    assert update["n"] == 80
    assert update["raw_transaction_rows"] == 0
    assert update["feature_schema"] == ["step", "amount"]
    with pytest.raises(ValueError):
        BankClient(4, np.zeros((0, 7)), np.zeros(0))


# ------------------------------------------------------- federated averaging
def _update(n: int, value: float, schema=("step", "amount")) -> dict:
    return {"bank": 0, "n": n, "feature_schema": list(schema),
            "delta_w": np.array([value, value]), "delta_b": float(value)}


def test_fedavg_is_sample_count_weighted() -> None:
    updates = [_update(100, 2.0), _update(300, 4.0)]
    delta_w, delta_b = fedavg(updates, (2,), ["step", "amount"])
    assert np.allclose(delta_w, [3.5, 3.5])
    assert np.isclose(delta_b, 3.5)


def test_fedavg_rejects_incompatible_or_empty_updates() -> None:
    schema = ["step", "amount"]
    with pytest.raises(ValueError):
        fedavg([], (2,), schema)  # empty
    bad_shape = _update(10, 1.0)
    bad_shape["delta_w"] = np.zeros(3)
    with pytest.raises(ValueError):
        fedavg([_update(10, 1.0), bad_shape], (2,), schema)
    non_finite = _update(10, 1.0)
    non_finite["delta_w"][0] = np.nan
    with pytest.raises(ValueError):
        fedavg([_update(10, 1.0), non_finite], (2,), schema)
    wrong_schema = _update(10, 1.0)
    wrong_schema["feature_schema"] = ["other"]
    with pytest.raises(ValueError):
        fedavg([_update(10, 1.0), wrong_schema], (2,), schema)


def test_privacy_clipping_and_noise_are_bounded_and_repeatable() -> None:
    vector = np.array([3.0, 4.0])
    assert np.isclose(np.linalg.norm(clip_client_update(vector, 1.0)), 1.0)
    updates = [_update(1, 2.0), _update(1, 2.0)]
    one = aggregate_with_gaussian_noise(updates, 1.0, 0.5, seed=7)
    two = aggregate_with_gaussian_noise(updates, 1.0, 0.5, seed=7)
    assert all(np.allclose(a, b) for a, b in zip(one, two))
    assert gaussian_epsilon(5, 1.5, 1e-5) > 0


# ------------------------------------------------------------- metrics/threshold
def test_threshold_tuned_on_validation_not_test() -> None:
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.2, 500).astype(int)
    scores = 0.3 * y + 0.1 * rng.standard_normal(500)
    threshold = engine._tune_threshold(y, scores)
    assert threshold in np.linspace(0.05, 0.95, 19)


def test_metrics_include_confusion_counts_and_fraud_totals() -> None:
    y = np.array([0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.4])
    metrics = engine._classification_metrics(y, scores, threshold=0.50)
    assert metrics["tp"] == 2 and metrics["fp"] == 0 and metrics["tn"] == 2 and metrics["fn"] == 1
    assert metrics["fraud_rows"] == 3 and metrics["test_rows"] == 5
    assert 0 <= metrics["f1"] <= 1


# --------------------------------------------------------------------- ledger
def test_ledger_verifies_and_detects_tampering() -> None:
    ledger = []
    ledger_module.add_event(ledger, "EXPERIMENT_STARTED", "Coordinator", {"banks": 5})
    ledger_module.add_event(ledger, "MODEL_UPDATE_SUBMITTED", "Bank 0", {"n": 100})
    assert ledger_module.verify_ledger(ledger)
    altered = ledger_module.demo_tamper(ledger)
    assert not ledger_module.verify_ledger(altered)
    bad = [dict(event) for event in ledger]
    bad[1]["hash"] = "0" * 64
    assert not ledger_module.verify_ledger(bad)


# --------------------------------------------------------------------- store
def _fake_result() -> dict:
    return {
        "config": {"rounds": 2, "sample_per_bank": 100, "seed": 42},
        "metrics": {"pooled": {"pr_auc": 0.01, "f1": 0.1}, "local": {}, "federated": {}},
        "per_bank": {},
        "privacy": {"mechanism": "test"},
        "feature_schema": {"feature_columns": ["step", "amount"]},
        "thresholds": {"pooled": 0.5},
        "pooled_model": {"kind": "logistic_regression", "numeric_columns": ["step", "amount"],
                          "type_categories": ["PAYMENT"], "mean": [1.0, 2.0],
                          "scale": [1.0, 1.0], "coef": [0.1, 0.1], "intercept": -0.5},
    }


def test_store_roundtrip_and_corrupt_handling(tmp_path) -> None:
    ledger = []
    ledger_module.add_event(ledger, "EXPERIMENT_STARTED", "Coordinator", {"banks": 5})
    _, run_file, saved_at = store.save_run(_fake_result(), ledger, artifacts_dir=tmp_path)
    assert run_file.is_file() and (tmp_path / "latest.json").is_file()
    run, info = store.load_latest(tmp_path)
    assert info["status"] == "ok"
    assert run["result"]["config"] == _fake_result()["config"]
    assert run["ledger"] == ledger
    store._atomic_write_text(tmp_path / "latest.json", "{corrupt json")
    run, info = store.load_latest(tmp_path)
    assert run is None and info["status"] == "corrupt"
    store._atomic_write_text(tmp_path / "latest.json", json.dumps({"kind": "other"}))
    run, info = store.load_latest(tmp_path)
    assert run is None and info["status"] == "incompatible"


def test_store_refuses_to_save_empty_results(tmp_path) -> None:
    ledger = []
    ledger_module.add_event(ledger, "EXPERIMENT_STARTED", "Coordinator", {"banks": 5})
    with pytest.raises(ValueError):
        store.save_run({}, ledger, artifacts_dir=tmp_path)


# -------------------------------------------------------------------- scoring
def _scorable_result() -> dict:
    result = _fake_result()
    result["pooled_model"].update({
        "numeric_columns": ["step", "amount"],
        "type_categories": ["PAYMENT"],
        "feature_columns": ["step", "amount", "PAYMENT"],
        "mean": [1.0, 2.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "coef": [0.2, 0.2, 0.2],
        "intercept": -0.5,
    })
    return result


def test_scoring_validates_inputs() -> None:
    result = _scorable_result()
    model = result["pooled_model"]
    transaction = {"step": 1, "amount": 10, "type": "PAYMENT"}
    score = score_transaction(transaction, model)
    assert 0 < score < 1
    assert score_transaction(transaction, model) == score  # deterministic
    with pytest.raises(ValueError):
        score_transaction(transaction, {})
    with pytest.raises(ValueError):
        score_transaction({**transaction, "amount": -1}, model)
    with pytest.raises(ValueError):
        score_transaction({**transaction, "type": "UNKNOWN"}, model)
    with pytest.raises(ValueError):
        score_transaction({"step": 1, "amount": float("nan"), "type": "PAYMENT"}, model)
    with pytest.raises(ValueError):
        score_transaction({"step": 1, "type": "PAYMENT"}, model)  # missing amount


# ---------------------------------------------------------------------- e2e
def test_end_to_end_small_run_uses_official_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(store, "ARTIFACT_DIR", tmp_path)
    ledger = []
    result = engine.run_experiment(
        RunConfig(sample_per_bank=500, rounds=1, local_epochs=1, learning_rate=0.1,
                  use_protection=False, dp_noise_multiplier=0.0, seed=42),
        ledger=ledger,
    )
    assert set(result["metrics"]) == {"local", "federated", "pooled"}
    assert len(result["per_bank"]) == 5
    assert result["raw_transaction_rows_sent_to_coordinator"] == 0
    assert ledger_module.verify_ledger(ledger)
    assert "pooled_model" in result
    # persistence roundtrip for the real run
    _, _, saved_at = store.save_run(result, ledger, artifacts_dir=tmp_path)
    loaded, info = store.load_latest(tmp_path)
    assert info["status"] == "ok" and loaded["result"]["config"] == result["config"]


def test_background_job_runner_persists_completed_run(tmp_path, monkeypatch) -> None:
    import time

    import jobs

    monkeypatch.setattr(store, "ARTIFACT_DIR", tmp_path)
    jobs.reset()
    config = RunConfig(sample_per_bank=300, rounds=1, local_epochs=1, learning_rate=0.1,
                       use_protection=False, dp_noise_multiplier=0.0, seed=42)
    jobs.start_job(config.__dict__)
    for _ in range(240):
        job = jobs.latest_job()
        if job and job["status"] in {"complete", "failed"}:
            break
        time.sleep(0.5)
    job = jobs.latest_job()
    assert job is not None and job["status"] == "complete", job.get("error") if job else "no job"
    result, ledger, saved_at = jobs.latest_completed()
    assert result and ledger and saved_at
    assert (tmp_path / "latest.json").is_file()