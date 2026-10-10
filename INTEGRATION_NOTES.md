# CIPHERMESH Integration Notes

Senior-engineer handover for the federated fraud backend connecting the
CIPHERMESH Streamlit UI to real data, training, aggregation and evaluation.

## 1. Original state (audit)

Two candidate implementations existed:

- `CIPHERMESH_Professional/app.py` — polished UI (navy/teal branding, hero,
  cards, tabs, plotly charts) but **synthetic-only** mono-file backend: it
  generated fake transactions (`make_synthetic_bank_data`), trained tiny logistic
  models, did unweighted FedAvg and a hash ledger. Best **UI**.
- `CIPHERMESH/` (FedGuard) — modular real backend: official dataset download and
  cache, SQLite persistence, auth, background jobs, clipping/noise. Best
  **ML/data logic**, plain UI.

The professional repo had no commits. The sibling was not a git repo.

## 2. Files changed and why

New backend modules (Git history is split into granular feature commits):

| File | Why |
| --- | --- |
| `data.py` | Real dataset load/cache, `BankID` partitioning, bounded stratified sampling, streaming facts. |
| `preprocess.py` | Fixed feature order, target/leakage/identifier exclusion, train-only scaler. |
| `banks.py` | Simulated clients that submit only deltas + metadata. |
| `privacy.py` | Weighted FedAvg, clipping, weighted Gaussian noise, conservative epsilon. |
| `ledger.py` | Hash-linked audit ledger + tamper detection. |
| `engine.py` | Local/FedAvg/pooled runs, validation threshold tuning, held-out evaluation. |
| `store.py` | Atomic artifact persistence + safe reload. |
| `jobs.py` | Background runner for honest progress states. |
| `app.py` | The existing UI, rewired to the pipeline (branding preserved). |
| `tests/test_core.py` | 20 automated tests. |
| `README.md`, `.gitignore`, `.streamlit/config.toml`, `requirements.txt` | Docs, ignores, run config, pinned deps. |

## 3. Measured results

Default run (5 banks, 4,000 rows/bank, 20 rounds, 5 epochs) on the full
636,262-row official test split (821 fraud, 0.129%). Protection OFF: pooled
PR-AUC 0.01147 / local 0.01154 / federated 0.01207. Protection ON (noise 0.5):
federated PR-AUC 0.00819. Absolute values are low; fraud is rare and features are
weak. No claim of real-world improvement is made.

## 4. Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q` → 20 passed.
- `streamlit.testing.v1.AppTest` on `app.py` → 0 exceptions, 6 tabs, persisted
  run reloaded.
- Ledger integrity PASS; 0 raw rows sent to the coordinator.

## 5. Honest limitations

Standardizer is fitted on the pooled bounded training sample; no secure
aggregation, TLS, authentication or formal DP guarantee; single-process
simulation; balance columns excluded by default (documented leakage).

## 6. Sharing with the sibling FedGuard project

This repo is mirrored to a local bare remote. The sibling `CIPHERMESH/` folder has
been given a `professional` remote and can pull the backend:

```bash
git -C ~/Documents/CIPHERMESH fetch professional
git -C ~/Documents/CIPHERMESH diff --stat professional/main -- ':!*.sqlite3'
```
