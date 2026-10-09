"""Dataset download, offline cache, bounded loading, and per-bank partitioning.

Uses the organisers' official dataset:

    https://huggingface.co/datasets/flwrlabs/fed-fraud-paysim-banks

Observed schema (12 columns):

    step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest,
    oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud, BankID

- ``isFraud``      : prediction target (kept out of model features).
- ``isFlaggedFraud``: output of an existing fraud rule; excluded to avoid leakage.
- ``BankID``       : partition key; excluded from model features.
- ``nameOrig``/``nameDest``: customer/account identifiers; excluded from features.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests

DATASET_ID = "flwrlabs/fed-fraud-paysim-banks"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/data"
FILES: dict[str, tuple[str, ...]] = {
    "train": ("train-00000-of-00002.parquet", "train-00001-of-00002.parquet"),
    "test": ("test-00000-of-00001.parquet",),
}
FILE_SIZES: dict[str, int] = {
    "train-00000-of-00002.parquet": 123_440_035,
    "train-00001-of-00002.parquet": 124_378_832,
    "test-00000-of-00001.parquet": 27_994_293,
}
# Row totals verified from the cached Parquet files (not assumed from the card).
EXPECTED_ROWS = {"train": 5_726_358, "test": 636_262}

CACHE_DIR = Path(os.environ.get("CIPHERMESH_CACHE_DIR", Path(__file__).parent / ".cache" / "fedguard"))


def ensure_dataset(cache_dir: Path = CACHE_DIR, offline: bool = False) -> dict[str, list[Path]]:
    """Ensure every official Parquet shard is cached. Offline mode never uses the network."""
    paths: dict[str, list[Path]] = {}
    for split, filenames in FILES.items():
        paths[split] = []
        for filename in filenames:
            target = cache_dir / filename
            if not target.is_file():
                if offline:
                    raise FileNotFoundError(
                        f"Offline cache is missing {target}. On first run the app "
                        "downloads the official shards (~276 MB total) from Hugging Face."
                    )
                cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                for attempt in range(10):
                    offset = temporary.stat().st_size if temporary.exists() else 0
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    try:
                        with requests.get(
                            f"{BASE_URL}/{filename}", headers=headers, stream=True,
                            timeout=(30, 120),
                        ) as response:
                            response.raise_for_status()
                            append = offset > 0 and response.status_code == 206
                            if offset and not append:
                                offset = 0
                            with temporary.open("ab" if append else "wb") as output:
                                for chunk in response.iter_content(chunk_size=1024 * 1024):
                                    if chunk:
                                        output.write(chunk)
                            if temporary.stat().st_size == FILE_SIZES[filename]:
                                break
                            if temporary.stat().st_size > FILE_SIZES[filename]:
                                temporary.unlink()
                                raise IOError(f"Downloaded an invalid size for {filename}")
                    except requests.RequestException:
                        if attempt == 9:
                            raise
                    time.sleep(min(2**attempt, 30))
                else:
                    raise ConnectionError(f"Could not finish downloading {filename}")
                if temporary.stat().st_size != FILE_SIZES[filename]:
                    raise ConnectionError(
                        f"Incomplete download for {filename}; partial file retained for resume"
                    )
                temporary.replace(target)
            paths[split].append(target)
    return paths


def load_split(
    split: str,
    offline: bool = True,
    columns: list[str] | None = None,
) -> Any:
    """Load one cached split as a pandas DataFrame (optionally only chosen columns)."""
    import pandas as pd

    shards = ensure_dataset(offline=offline)
    frames = [pd.read_parquet(path, columns=columns) for path in shards[split]]
    return pd.concat(frames, ignore_index=True)


def load_frames(offline: bool = True, columns: list[str] | None = None) -> tuple[Any, Any]:
    """Load cached train and test shards as (train, test) DataFrames."""
    train = load_split("train", offline=offline, columns=columns)
    test = load_split("test", offline=offline, columns=columns)
    return train, test


def bounded_train_sample(frame: Any, per_bank: int, seed: int) -> Any:
    """Fraud-stratified, bounded per-bank training sample for reproducible runtimes."""
    if per_bank <= 0:
        raise ValueError("per_bank must be positive")
    import numpy as np
    import pandas as pd

    parts = []
    for _, bank_rows in frame.groupby("BankID", sort=True):
        positives = bank_rows[bank_rows["isFraud"] == 1]
        negatives = bank_rows[bank_rows["isFraud"] == 0]
        target_positives = min(len(positives), max(1, per_bank // 5))
        target_negatives = min(len(negatives), max(1, per_bank - target_positives))
        parts.append(positives.sample(n=target_positives, random_state=seed))
        parts.append(negatives.sample(n=target_negatives, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def train_val_split(frame: Any, val_frac: float = 0.3, seed: int = 42) -> tuple[Any, Any]:
    """Stratified train/validation split on the target, applied before touching test."""
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be strictly between 0 and 1")
    from sklearn.model_selection import train_test_split

    labels = frame["isFraud"].to_numpy()
    if len(labels) < 2 or len(set(labels.tolist())) < 2:
        raise ValueError("Each bank partition needs at least one fraud and one non-fraud row.")
    train, val = train_test_split(
        frame, test_size=val_frac, random_state=seed, stratify=labels
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def partition_banks(frame: Any) -> dict[int, Any]:
    """Split a frame into per-BankID partitions keyed by integer bank id."""
    import numpy as np

    banks: dict[int, Any] = {}
    for bank_id, group in frame.groupby("BankID", sort=True):
        banks[int(bank_id)] = group.reset_index(drop=True)
    return banks


def inspect_dataset(paths: dict[str, list[Path]]) -> dict[str, Any]:
    """Stream through the cached Parquet data and compute verifiable facts."""
    facts: dict[str, Any] = {"dataset_id": DATASET_ID, "splits": {}, "banks": {}}
    bank_counts: Counter[int] = Counter()
    bank_fraud: Counter[int] = Counter()
    bank_amount: defaultdict[int, float] = defaultdict(float)
    bank_types: defaultdict[int, Counter[str]] = defaultdict(Counter)
    split_counts: Counter[str] = Counter()
    split_fraud: Counter[str] = Counter()
    split_schemas: dict[str, list[dict[str, str]]] = {}
    global_types: Counter[str] = Counter()
    for split, shards in paths.items():
        split_schemas[split] = []
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            if not split_schemas[split]:
                split_schemas[split] = [
                    {"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow
                ]
            for batch in parquet.iter_batches(
                columns=["BankID", "isFraud", "amount", "type"], batch_size=100_000
            ):
                table = batch.to_pydict()
                for bank, fraud, amount, transaction_type in zip(
                    table["BankID"], table["isFraud"], table["amount"], table["type"]
                ):
                    bank_id = int(bank)
                    label = int(fraud)
                    bank_counts[bank_id] += 1
                    bank_fraud[bank_id] += label
                    bank_amount[bank_id] += float(amount)
                    bank_types[bank_id][str(transaction_type)] += 1
                    split_counts[split] += 1
                    split_fraud[split] += label
                    global_types[str(transaction_type)] += 1
    facts["splits"] = {
        split: {"rows": split_counts[split], "fraud_rows": split_fraud[split]}
        for split in paths
    }
    facts["schema"] = split_schemas
    facts["transaction_types"] = dict(sorted(global_types.items()))
    facts["banks"] = {
        str(bank): {
            "rows": bank_counts[bank],
            "fraud_rows": bank_fraud[bank],
            "fraud_rate": bank_fraud[bank] / bank_counts[bank],
            "mean_amount": bank_amount[bank] / bank_counts[bank],
            "transaction_types": dict(sorted(bank_types[bank].items())),
        }
        for bank in sorted(bank_counts)
    }
    facts["total_rows"] = sum(split_counts.values())
    facts["total_fraud_rows"] = sum(split_fraud.values())
    facts["row_counts_match"] = {
        split: facts["splits"][split]["rows"] == EXPECTED_ROWS[split] for split in EXPECTED_ROWS
    }
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Only load existing cached shards")
    args = parser.parse_args()
    result = inspect_dataset(ensure_dataset(offline=args.offline))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()