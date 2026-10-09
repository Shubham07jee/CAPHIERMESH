import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="CIPHERMESH | Federated Fraud Intelligence", page_icon="🔐", layout="wide")

st.markdown('\n<style>\n:root {\n  --cm-navy: #0b1220;\n  --cm-panel: #111c2f;\n  --cm-teal: #19c7b5;\n  --cm-muted: #93a4ba;\n}\n.stApp { background: #f4f7fb; color: #152238; }\n[data-testid="stHeader"] { background: rgba(244,247,251,0.92); }\n[data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid #22314a; }\n[data-testid="stSidebar"] * { color: #e6edf7 !important; }\n[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aab8cc !important; }\n[data-testid="stSidebar"] button { border-radius: 10px !important; }\n.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1440px; }\nh1, h2, h3 { letter-spacing: -0.035em; color: #122039; }\nh1 { font-weight: 800 !important; }\n[data-testid="stMetric"] {\n  background: white; border: 1px solid #e1e8f1; border-radius: 16px;\n  padding: 16px 18px; box-shadow: 0 3px 14px rgba(18,32,57,.035);\n}\n[data-testid="stMetricLabel"] { color: #64748b !important; font-size: .83rem !important; }\n[data-testid="stMetricValue"] { color: #122039 !important; font-weight: 750 !important; }\n.stButton > button, .stDownloadButton > button {\n  border-radius: 10px; border: 1px solid #d7e0eb; font-weight: 650;\n  min-height: 2.6rem;\n}\n.stButton > button[kind="primary"] { background: #087f8c; border-color: #087f8c; color: white; }\n.stButton > button[kind="primary"]:hover { background: #066b76; border-color: #066b76; }\n[data-testid="stTabs"] button { font-weight: 650; }\n[data-testid="stDataFrame"] { border: 1px solid #e1e8f1; border-radius: 12px; overflow: hidden; }\ndiv[data-testid="stAlert"] { border-radius: 12px; }\n.cm-hero {\n  background: linear-gradient(115deg, #0b1220 0%, #142641 68%, #0d555e 100%);\n  border-radius: 22px; padding: 28px 30px; color: white; margin-bottom: 20px;\n  box-shadow: 0 12px 32px rgba(11,18,32,.13);\n}\n.cm-hero h1 { color: white !important; font-size: 2.25rem; margin: 0 0 7px 0; }\n.cm-hero p { color: #c9d6e8; margin: 0; font-size: 1rem; }\n.cm-eyebrow { color: #62e2d3; font-size: .76rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 10px; }\n.cm-chip { display: inline-block; padding: 5px 10px; border-radius: 999px; background: rgba(25,199,181,.13); color: #72e9dc; border: 1px solid rgba(25,199,181,.25); font-size: .76rem; margin-top: 16px; }\n.cm-section { border-bottom: 1px solid #e0e7f0; padding-bottom: 10px; margin: 12px 0 16px 0; }\n.cm-card {\n  background: white; border: 1px solid #e1e8f1; border-radius: 16px;\n  padding: 18px 20px; min-height: 132px; box-shadow: 0 3px 14px rgba(18,32,57,.035);\n}\n.cm-card h3 { margin: 0 0 8px 0; font-size: 1rem; color: #122039; }\n.cm-card p { margin: 0; color: #61718a; font-size: .9rem; line-height: 1.55; }\n.cm-step { color: #087f8c; font-weight: 800; font-size: .75rem; letter-spacing: .09em; margin-bottom: 8px; }\n.small-muted { color: #64748b; font-size: .86rem; }\n</style>\n', unsafe_allow_html=True)

FEATURES = ["amount_log", "balance_change", "is_transfer", "is_cashout", "unusual_amount"]
LABEL = "is_fraud"


def make_synthetic_bank_data(n=1800, seed=42, bank_id=0):
    """Create explicitly synthetic transactions for a reproducible demo."""
    rng = np.random.default_rng(seed + bank_id * 101)
    amount = rng.lognormal(mean=3.0 + bank_id * 0.05, sigma=1.0, size=n)
    txn_type = rng.choice([0, 1, 2, 3], size=n, p=[0.50, 0.22, 0.18, 0.10])
    balance_change = amount * rng.normal(0.8, 0.35, size=n)
    unusual_amount = (amount > np.quantile(amount, 0.88)).astype(float)
    is_transfer = (txn_type == 1).astype(float)
    is_cashout = (txn_type == 2).astype(float)

    # Synthetic label mechanism, not a real-world fraud model.
    logit = (
        -5.0
        + 0.85 * is_transfer
        + 1.05 * is_cashout
        + 0.55 * unusual_amount
        + 0.18 * (balance_change / (np.median(amount) + 1e-6))
        + rng.normal(0, 0.55, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob)
    X = pd.DataFrame({
        "amount_log": np.log1p(amount),
        "balance_change": balance_change / (np.median(amount) + 1e-6),
        "is_transfer": is_transfer,
        "is_cashout": is_cashout,
        "unusual_amount": unusual_amount,
    })
    return X.astype(float), y.astype(int)


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def train_local_model(X, y, initial_w, initial_b, epochs=80, lr=0.08, l2=0.002):
    """Small transparent logistic-regression trainer using gradient descent."""
    w = initial_w.copy()
    b = float(initial_b)
    n = len(y)
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        error = p - y
        grad_w = (X.T @ error) / n + l2 * w
        grad_b = float(np.mean(error))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def fedavg(updates):
    """Weighted average of compatible model parameters only; no transaction rows."""
    total = sum(item["n"] for item in updates)
    if total <= 0:
        raise ValueError("No training samples to aggregate.")
    shapes = {item["w"].shape for item in updates}
    if len(shapes) != 1:
        raise ValueError("Incompatible model parameter shapes.")
    w = sum((item["n"] / total) * item["w"] for item in updates)
    b = sum((item["n"] / total) * item["b"] for item in updates)
    return w, float(b)


def metrics_for(y, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    return {
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "PR-AUC (Average Precision)": average_precision_score(y, probabilities) if len(np.unique(y)) > 1 else float("nan"),
        "ROC-AUC": roc_auc_score(y, probabilities) if len(np.unique(y)) > 1 else float("nan"),
        "Fraud examples": int(np.sum(y)),
        "Test transactions": int(len(y)),
    }


def add_ledger_event(ledger, event_type, actor, details):
    previous_hash = ledger[-1]["hash"] if ledger else "GENESIS"
    body = {
        "event_id": len(ledger) + 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event_type": event_type,
        "actor": actor,
        "details": details,
        "previous_hash": previous_hash,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    ledger.append(body)


def verify_ledger(ledger):
    prev = "GENESIS"
    for idx, event in enumerate(ledger, start=1):
        if event.get("event_id") != idx or event.get("previous_hash") != prev:
            return False
        body = {k: v for k, v in event.items() if k != "hash"}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if event.get("hash") != expected:
            return False
        prev = event["hash"]
    return True


def prepare_banks(n_banks=5, rows_per_bank=1800, seed=42):
    banks = {}
    for i in range(n_banks):
        X, y = make_synthetic_bank_data(rows_per_bank, seed, i)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed + i,
            stratify=y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
        )
        banks[f"Bank {chr(65+i)}"] = {
            "X_train": X_train.to_numpy(dtype=float),
            "y_train": y_train,
            "X_test": X_test.to_numpy(dtype=float),
            "y_test": y_test,
        }
    return banks


def run_experiment(banks, rounds, epochs, lr, threshold, clip_norm, noise_scale, use_protection):
    rng = np.random.default_rng(123)
    global_w = np.zeros(len(FEATURES), dtype=float)
    global_b = 0.0
    ledger = []
    transfer_rows = 0
    update_count = 0
    clipping_records = []
    bank_local = {}
    for name, data in banks.items():
        w, b = train_local_model(data["X_train"], data["y_train"], np.zeros(len(FEATURES)), 0.0, epochs, lr)
        bank_local[name] = (w, b)

    add_ledger_event(ledger, "EXPERIMENT_STARTED", "Coordinator", {
        "banks": len(banks), "rounds": rounds, "dataset": "synthetic demo data"
    })

    round_records = []
    for round_num in range(1, rounds + 1):
        updates = []
        for name, data in banks.items():
            local_w, local_b = train_local_model(
                data["X_train"], data["y_train"], global_w, global_b, epochs, lr
            )
            delta_w = local_w - global_w
            delta_b = local_b - global_b

            norm = float(np.sqrt(np.sum(delta_w ** 2) + delta_b ** 2))
            clipped = False
            if use_protection and norm > clip_norm:
                factor = clip_norm / (norm + 1e-12)
                delta_w = delta_w * factor
                delta_b = delta_b * factor
                clipped = True

            if use_protection and noise_scale > 0:
                delta_w = delta_w + rng.normal(0, noise_scale, size=delta_w.shape)
                delta_b = float(delta_b + rng.normal(0, noise_scale))

            protected_w = global_w + delta_w
            protected_b = global_b + delta_b
            updates.append({"w": protected_w, "b": protected_b, "n": len(data["y_train"]), "bank": name})
            update_count += 1
            clipping_records.append({
                "Round": round_num, "Bank": name, "Original update norm": norm,
                "Clipped": clipped, "Noise scale": noise_scale if use_protection else 0.0
            })
            add_ledger_event(ledger, "MODEL_UPDATE_SUBMITTED", name, {
                "round": round_num, "training_examples": len(data["y_train"]),
                "raw_transaction_rows_sent": 0, "update_norm": round(norm, 6),
                "clipped": clipped, "noise_scale": noise_scale if use_protection else 0.0
            })

        global_w, global_b = fedavg(updates)
        add_ledger_event(ledger, "FEDAVG_COMPLETED", "Coordinator", {
            "round": round_num, "updates_aggregated": len(updates),
            "raw_transaction_rows_received": 0
        })
        round_records.append({"Round": round_num, "Updates aggregated": len(updates)})

    results = []
    for name, data in banks.items():
        local_w, local_b = bank_local[name]
        local_prob = sigmoid(data["X_test"] @ local_w + local_b)
        global_prob = sigmoid(data["X_test"] @ global_w + global_b)
        lm = metrics_for(data["y_test"], local_prob, threshold)
        fm = metrics_for(data["y_test"], global_prob, threshold)
        for model_name, m in [("Local-only", lm), ("Federated", fm)]:
            row = {"Bank": name, "Model": model_name, **m}
            results.append(row)
        add_ledger_event(ledger, "BANK_EVALUATION_COMPLETED", name, {
            "test_transactions": len(data["y_test"]), "raw_transaction_rows_sent": 0
        })

    add_ledger_event(ledger, "EXPERIMENT_COMPLETED", "Coordinator", {
        "raw_transaction_rows_sent": transfer_rows, "model_updates_submitted": update_count
    })
    return {
        "results": pd.DataFrame(results),
        "ledger": ledger,
        "clipping": pd.DataFrame(clipping_records),
        "rounds": pd.DataFrame(round_records),
        "global_w": global_w,
        "global_b": global_b,
        "update_count": update_count,
        "raw_rows_sent": transfer_rows,
        "banks": banks,
        "threshold": threshold,
    }


if "experiment" not in st.session_state:
    st.session_state.experiment = None
if "tamper_demo" not in st.session_state:
    st.session_state.tamper_demo = False

st.markdown("""
<div class="cm-hero">
  <div class="cm-eyebrow">Federated fraud intelligence · CSB-04</div>
  <h1>🔐 CIPHERMESH</h1>
  <p>Collaborative fraud learning. Raw transaction rows stay within each simulated bank.</p>
  <span class="cm-chip">PROTOTYPE ENVIRONMENT · SYNTHETIC DATA</span>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="cm-section"><h3>Experiment workspace</h3><div class="small-muted">Train local models, aggregate updates, and evaluate performance bank by bank.</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔐 CIPHERMESH")
    st.markdown("**Federated Fraud Intelligence**")
    st.markdown('<div class="small-muted">CSB-04 · Hackathon prototype</div>', unsafe_allow_html=True)
    st.divider()
    st.header("Experiment settings")
    n_banks = st.selectbox("Simulated banks", [3, 4, 5], index=2)
    rows_per_bank = st.slider("Synthetic transactions per bank", 500, 3000, 1800, step=100)
    rounds = st.slider("Federated rounds", 1, 8, 3)
    epochs = st.slider("Local training epochs per round", 20, 150, 80, step=10)
    lr = st.slider("Learning rate", 0.01, 0.20, 0.08, step=0.01)
    threshold = st.slider("Fraud review threshold", 0.10, 0.90, 0.50, step=0.05)
    use_protection = st.checkbox("Enable update clipping / optional noise", value=True)
    clip_norm = st.slider("Update clipping norm", 0.05, 2.00, 0.50, step=0.05)
    noise_scale = st.slider("Gaussian noise scale (0 = off)", 0.0, 0.20, 0.0, step=0.01)
    run_button = st.button("🚀 Run federated experiment", type="primary", use_container_width=True)
    st.caption("Clipping/noise are experimental controls—not a formal differential-privacy guarantee.")
    st.caption("Demo only: no real bank connection or production security controls.")

if run_button:
    with st.spinner("Training simulated banks and aggregating model updates..."):
        banks = prepare_banks(n_banks, rows_per_bank, seed=42)
        st.session_state.experiment = run_experiment(
            banks, rounds, epochs, lr, threshold, clip_norm, noise_scale, use_protection
        )
    st.session_state.tamper_demo = False

exp = st.session_state.experiment

if exp is None:
    st.info("Choose settings in the sidebar and click **Run federated experiment** to train the models and calculate real metrics.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows sent to coordinator", "—")
    c2.metric("Federated rounds", "—")
    c3.metric("Simulated banks", str(n_banks))
    st.markdown("### What this prototype demonstrates")
    st.markdown("1. Each simulated bank trains on its own local partition.\n2. Only model parameters are passed to the FedAvg coordinator.\n3. A shared model is evaluated separately on each bank's held-out test set.\n4. A hash-linked ledger records the workflow.")
    st.warning("This demo generates synthetic transactions. It is not connected to real banks and does not prove real-world fraud detection performance.")
else:
    results = exp["results"]
    federated = results[results["Model"] == "Federated"]
    local = results[results["Model"] == "Local-only"]
    avg_f1_fed = float(federated["F1"].mean())
    avg_f1_local = float(local["F1"].mean())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Simulated banks", len(exp["banks"]))
    m2.metric("Federated rounds", len(exp["rounds"]))
    m3.metric("Raw rows sent to coordinator", exp["raw_rows_sent"])
    m4.metric("Model updates submitted", exp["update_count"])
    st.caption("The raw-row count is measured for this in-process simulation's coordinator interface. It is not a claim of production security.")

    tabs = st.tabs(["Overview & metrics", "Bank Divergence Radar", "Privacy & Trust Ledger", "Risk scoring demo", "Implementation notes"])

    with tabs[0]:
        st.markdown("### Experiment summary")
        col1, col2 = st.columns(2)
        col1.metric("Mean F1 — local-only", f"{avg_f1_local:.3f}")
        col2.metric("Mean F1 — federated", f"{avg_f1_fed:.3f}", delta=f"{avg_f1_fed-avg_f1_local:+.3f}")
        st.caption("Measured on generated synthetic data only. These results are not evidence of performance on real banking transactions.")
        bank_summary = results.pivot(index="Bank", columns="Model", values=["Precision", "Recall", "F1", "PR-AUC (Average Precision)"])
        st.dataframe(bank_summary.round(3), use_container_width=True)
        st.markdown("### Workflow")
        st.markdown("**Local data → local model updates → FedAvg coordinator → shared model → bank-wise evaluation**")
        chart = px.bar(results, x="Bank", y="F1", color="Model", barmode="group", title="F1-score by simulated bank")
        st.plotly_chart(chart, use_container_width=True)

    with tabs[1]:
        st.markdown("### Bank Divergence Radar")
        metric = st.selectbox("Metric to compare", ["PR-AUC (Average Precision)", "Recall", "Precision", "F1", "ROC-AUC"])
        radar_data = results[["Bank", "Model", metric]].copy()
        st.plotly_chart(px.bar(radar_data, x="Bank", y=metric, color="Model", barmode="group", title=f"{metric}: local-only vs federated"), use_container_width=True)
        pivot = results.pivot(index="Bank", columns="Model", values=metric)
        if "Federated" in pivot and "Local-only" in pivot:
            delta_df = (pivot["Federated"] - pivot["Local-only"]).rename("Federated minus local-only").reset_index()
            st.dataframe(delta_df.round(4), use_container_width=True)
        st.caption("Federated learning is not guaranteed to improve every bank. Interpret small synthetic experiments cautiously.")

    with tabs[2]:
        st.markdown("### Privacy Transparency Panel")
        p1, p2, p3 = st.columns(3)
        p1.metric("Raw transaction rows sent", exp["raw_rows_sent"])
        p2.metric("Model updates submitted", exp["update_count"])
        p3.metric("Ledger integrity", "PASS" if verify_ledger(exp["ledger"]) else "FAIL")
        st.write("**What the coordinator receives:** compatible model parameters and limited metadata. In this prototype, raw transaction rows are not passed into `fedavg()`.")
        st.warning("Model updates can still reveal information. Zero raw rows sent does not mean zero privacy risk. This local simulation does not implement secure aggregation, network transport security, authentication, or formal differential privacy.")
        st.markdown("### Update protection measurements")
        st.dataframe(exp["clipping"].round(4), use_container_width=True)
        st.markdown("### Hash-linked Trust Ledger")
        ledger_df = pd.DataFrame(exp["ledger"])
        st.dataframe(ledger_df[["event_id", "timestamp_utc", "event_type", "actor", "details", "previous_hash", "hash"]], use_container_width=True)
        st.button("Demonstrate tamper detection", on_click=lambda: st.session_state.update(tamper_demo=True))
        if st.session_state.tamper_demo:
            altered = [dict(event) for event in exp["ledger"]]
            if altered:
                altered[0]["actor"] = "Modified event"
            st.error("Modified ledger verification: " + ("PASS (unexpected)" if verify_ledger(altered) else "FAIL — tampering detected"))

        st.download_button(
            "Download experiment metrics (CSV)",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="ciphermesh_metrics.csv",
            mime="text/csv"
        )
        st.download_button(
            "Download audit ledger (JSON)",
            data=json.dumps(exp["ledger"], indent=2).encode("utf-8"),
            file_name="ciphermesh_audit_ledger.json",
            mime="application/json"
        )

    with tabs[3]:
        st.markdown("### Risk scoring demo")
        st.caption("Illustrative input only. This is a score from the synthetic demo model, not a bank decision.")
        amount_log = st.slider("Log-scaled transaction amount", 0.0, 12.0, 5.0, step=0.1)
        balance_change = st.slider("Normalised balance change", -2.0, 8.0, 1.0, step=0.1)
        is_transfer = st.selectbox("Transfer transaction?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        is_cashout = st.selectbox("Cash-out transaction?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        unusual_amount = st.selectbox("Unusually large amount?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        sample = np.array([[amount_log, balance_change, is_transfer, is_cashout, unusual_amount]], dtype=float)
        risk = float(sigmoid(sample @ exp["global_w"] + exp["global_b"])[0])
        st.metric("Demo fraud-risk score", f"{risk:.1%}")
        if risk >= exp["threshold"]:
            st.error("Flag for review — score is at or above the selected threshold.")
        else:
            st.success("Below the selected review threshold.")
        st.caption("A risk score is not proof of fraud. Real deployment requires validated features, calibration, monitoring, and human investigation.")

    with tabs[4]:
        st.markdown("### What is implemented")
        st.markdown("- Five simulated bank partitions (configurable to 3–5).\n- Local logistic-regression-style training using gradient descent.\n- Sample-count-weighted FedAvg over compatible model parameters.\n- Local-only vs federated evaluation by bank.\n- Precision, recall, F1, PR-AUC, ROC-AUC and fraud counts.\n- Optional clipping and Gaussian noise experiment.\n- Hash-linked ledger and tamper-detection demonstration.\n- Downloadable metrics and audit events.")
        st.markdown("### What is not implemented")
        st.markdown("- Real bank connections or distributed servers.\n- Real customer transaction data.\n- Secure aggregation, TLS, authentication, access controls, or production monitoring.\n- Formal differential privacy guarantees or privacy accounting.\n- Proof that federated learning improves real-world fraud detection.")
        st.markdown("### Judge-ready explanation")
        st.info("Each simulated bank trains on its own local transaction partition. The coordinator receives model parameters rather than raw transaction rows and combines compatible parameters using sample-count-weighted Federated Averaging. We then compare the local-only and shared models on each bank's held-out test set. This prototype uses synthetic data, so the results demonstrate the workflow rather than real-world banking performance.")
