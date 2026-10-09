# CIPHERMESH — Federated Fraud Signal Sharing Across Banks

CIPHERMESH is a hackathon prototype that connects its existing Streamlit UI to a
real, reproducible fraud-detection backend. Simulated bank clients train local
models on their own partition of the official PaySim bank dataset, submit only
model updates to a coordinator, and the shared model is evaluated on the
official held-out test split.

> Research prototype on **synthetic** PaySim data. It is not a production fraud
> system, not a secure multi-party deployment, and not a formal
> differential-privacy guarantee. Do not use it with real customer data or for
> financial decisions.

## Dataset

- Organisers' dataset: <https://huggingface.co/datasets/flwrlabs/fed-fraud-paysim-banks>
- Observed schema (12 columns): `step, type, amount, nameOrig, oldbalanceOrg,
  newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud,
  isFlaggedFraud, BankID`.
- Observed rows: **5,726,358 train** and **636,262 test**; **8,213 fraud** in total.
- Partitioned by the real `BankID` field (banks 0–4).
- `isFraud` is the target, `isFlaggedFraud` is excluded to avoid rule leakage,
  and `BankID` plus `nameOrig`/`nameDest` are excluded from model features.
- Balance columns are excluded by default because their accounting arithmetic
  makes the task near-trivial (a documented leakage sensitivity run is available).

The repository ships with the cached Parquet shards under `.cache/fedguard/`
(gitignored, ~276 MB) so the app and tests run offline. On a clean checkout the
loader downloads them from Hugging Face on first start.

## Install and run (Linux)

```bash
cd CIPHERMESH_Professional
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Open <http://localhost:8501>. If the cache is missing, the first start downloads
the official shards. To inspect the cache schema/statistics without the UI:

```bash
.venv/bin/python data.py --offline
```

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids unrelated ROS/system pytest plugins on
this machine. All 20 tests pass (data partitioning and target separation,
training shapes, weighted FedAvg, empty/incompatible client updates, metrics and
threshold behaviour, ledger integrity/tamper detection, artifact save/reload,
risk-scoring validation, and an end-to-end run on the official cache).

## Module map

| File | Responsibility |
| --- | --- |
| `data.py` | Download/offline cache, bounded fraud-stratified sampling, per-`BankID` partitioning, streaming dataset facts. |
| `preprocess.py` | Fixed-order feature engineering; target/leakage/identifier exclusion; scaler fitted on training only. |
| `banks.py` | Simulated bank clients: local training and model updates that carry **no** raw rows. |
| `privacy.py` | L2 clipping, sample-count-weighted averaging, Gaussian noise, conservative epsilon estimate. |
| `ledger.py` | Hash-linked audit ledger with integrity verification and a tamper demo. |
| `engine.py` | Local-only, FedAvg and pooled baselines; threshold tuning on validation; held-out evaluation. |
| `store.py` | Atomic JSON artifact persistence and safe reload of runs, models and ledgers. |
| `jobs.py` | Background experiment runner so the UI shows genuine progress states. |
| `app.py` | The CIPHERMESH Streamlit UI, wired to the pipeline above. |

## How a run works

1. Load the cached train/test shards and take a bounded, fraud-stratified
   training sample per `BankID`.
2. Split each bank's sample into train + validation (stratified on the target).
   The official test split is never touched during training or threshold tuning.
3. Build features in a fixed order and fit the standardizer on pooled training
   rows only. The scoring model reuses the saved mean/scale, so scaling is
   reproducible across sessions.
4. Train three model families with the **same optimizer and epoch budget**:
   pooled (all banks' sample), local-only (one bank), and federated (FedAvg).
5. FedAvg: each round, clients start from the shared model, take local gradient
   steps, and return only their delta plus metadata (bank id, row count, feature
   schema). The coordinator clips, optionally adds Gaussian noise, and averages
   the deltas weighted by sample count.
6. Tune one review threshold per model on the validation split, then report
   precision, recall, F1, PR-AUC, ROC-AUC, confusion counts and fraud counts on
   the full official test split at natural prevalence.
7. Persist the run (metrics, thresholds, feature schema, preprocessing config,
   model parameters, ledger) to `artifacts/latest.json`; the next session reloads it.

## Measured results (this machine, Python 3.14.4)

Default run: 5 banks, 4,000 rows/bank, 20 FedAvg rounds, 5 local epochs,
lr 0.1, clipping norm 1.0. Completed in ~13 seconds. Evaluated on all 636,262
official test rows (821 fraud, 0.129% prevalence). Thresholds tuned on
validation.

**With protection ON (clip 1.0, Gaussian noise multiplier 0.5):**

| Model | PR-AUC | Recall | Precision | F1 | Threshold |
| --- | --- | --- | --- | --- | --- |
| Pooled (centralized) | 0.01147 | 0.694 | 0.0107 | 0.0211 | 0.50 |
| Local-only | 0.01154 | 0.689 | 0.0106 | 0.0209 | 0.50 |
| Federated (noisy) | 0.00819 | 0.544 | 0.0061 | 0.0120 | 0.50 |

**With protection OFF (no clipping/noise):**

| Model | PR-AUC | Recall | Precision | F1 | Threshold |
| --- | --- | --- | --- | --- | --- |
| Pooled (centralized) | 0.01147 | 0.694 | 0.0107 | 0.0211 | 0.50 |
| Local-only | 0.01154 | 0.689 | 0.0106 | 0.0209 | 0.50 |
| Federated | 0.01207 | 0.777 | 0.0083 | 0.0164 | 0.35 |

**Interpretation (honest):** without noise the federated model slightly improves
PR-AUC and recall over local-only and the pooled baseline, but the absolute
PR-AUC is very low because fraud is extremely rare and the available features
are weak. This is not evidence of real-world improvement. The default active
Gaussian noise degrades the federated model — a real privacy/utility trade-off
we report rather than hide. The privacy estimate at the default (20 rounds,
noise multiplier 0.5, delta 1e-5) is a conservative epsilon upper bound of
~91.5; this is **not** a formal DP guarantee.

## Privacy and audit boundaries

- Clipping and Gaussian noise are implemented and their effect is measurable,
  but the documented epsilon is only a conservative composition estimate.
- No secure aggregation, TLS/mTLS, authentication, access control or network
  transport is implemented. The "coordinator" is a function in one process.
- The hash-linked ledger is a **local simulation log** with working tamper
  detection; it is not a capture from a real network.
- Raw transaction rows are never passed to aggregation in this code path; the
  coordinator receives model deltas and metadata only. Zero raw rows does not
  mean zero privacy risk.

## Known limitations and remaining work

1. The standardizer is fitted on the pooled bounded training sample (a
   documented, small departure from strict data locality).
2. Balance columns are excluded by default; the leakage sensitivity comparison
   is available but not queued automatically.
3. Training is single-process and synchronous per round; no real distribution.
4. Low absolute metrics reflect a hard, rare-event task; feature engineering or a
   gradient-boosted model would be the next honest step.
5. No authentication, so the app is intended for local demonstration only.
