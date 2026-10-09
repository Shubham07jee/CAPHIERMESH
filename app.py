import json

import pandas as pd
import plotly.express as px
import streamlit as st

import jobs
import ledger as ledger_module
from data import ensure_dataset, inspect_dataset, load_frames
from engine import RunConfig, score_transaction
from store import load_latest

st.set_page_config(page_title="CIPHERMESH | Federated Fraud Intelligence", page_icon="🔐", layout="wide")

st.markdown("""
<style>
:root {
  --cm-navy: #0b1220;
  --cm-panel: #111c2f;
  --cm-teal: #19c7b5;
  --cm-muted: #93a4ba;
}
.stApp { background: #f4f7fb; color: #152238; }
[data-testid="stHeader"] { background: rgba(244,247,251,0.92); }
[data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid #22314a; }
[data-testid="stSidebar"] * { color: #e6edf7 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aab8cc !important; }
[data-testid="stSidebar"] button { border-radius: 10px !important; }
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1440px; }
h1, h2, h3 { letter-spacing: -0.035em; color: #122039; }
h1 { font-weight: 800 !important; }
[data-testid="stMetric"] {
  background: white; border: 1px solid #e1e8f1; border-radius: 16px;
  padding: 16px 18px; box-shadow: 0 3px 14px rgba(18,32,57,.035);
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: .83rem !important; }
[data-testid="stMetricValue"] { color: #122039 !important; font-weight: 750 !important; }
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; border: 1px solid #d7e0eb; font-weight: 650;
  min-height: 2.6rem;
}
.stButton > button[kind="primary"] { background: #087f8c; border-color: #087f8c; color: white; }
.stButton > button[kind="primary"]:hover { background: #066b76; border-color: #066b76; }
[data-testid="stTabs"] button { font-weight: 650; }
[data-testid="stDataFrame"] { border: 1px solid #e1e8f1; border-radius: 12px; overflow: hidden; }
div[data-testid="stAlert"] { border-radius: 12px; }
.cm-hero {
  background: linear-gradient(115deg, #0b1220 0%, #142641 68%, #0d555e 100%);
  border-radius: 22px; padding: 28px 30px; color: white; margin-bottom: 20px;
  box-shadow: 0 12px 32px rgba(11,18,32,.13);
}
.cm-hero h1 { color: white !important; font-size: 2.25rem; margin: 0 0 7px 0; }
.cm-hero p { color: #c9d6e8; margin: 0; font-size: 1rem; }
.cm-eyebrow { color: #62e2d3; font-size: .76rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 10px; }
.cm-chip { display: inline-block; padding: 5px 10px; border-radius: 999px; background: rgba(25,199,181,.13); color: #72e9dc; border: 1px solid rgba(25,199,181,.25); font-size: .76rem; margin-top: 16px; }
.cm-section { border-bottom: 1px solid #e0e7f0; padding-bottom: 10px; margin: 12px 0 16px 0; }
.cm-card {
  background: white; border: 1px solid #e1e8f1; border-radius: 16px;
  padding: 18px 20px; min-height: 132px; box-shadow: 0 3px 14px rgba(18,32,57,.035);
}
.cm-card h3 { margin: 0 0 8px 0; font-size: 1rem; color: #122039; }
.cm-card p { margin: 0; color: #61718a; font-size: .9rem; line-height: 1.55; }
.cm-step { color: #087f8c; font-weight: 800; font-size: .75rem; letter-spacing: .09em; margin-bottom: 8px; }
.small-muted { color: #64748b; font-size: .86rem; }
</style>
""", unsafe_allow_html=True)

if "tamper_demo" not in st.session_state:
    st.session_state.tamper_demo = False
if "shown_job" not in st.session_state:
    st.session_state.shown_job = None


@st.cache_resource(show_spinner="Checking the official dataset cache…")
def backend_startup():
    try:
        paths = ensure_dataset(offline=True)
    except FileNotFoundError:
        paths = ensure_dataset(offline=False)
    return inspect_dataset(paths)


try:
    DATA_FACTS = backend_startup()
    BACKEND_READY = True
except Exception as exc:  # noqa: BLE001 - surfaced to the user below
    DATA_FACTS = {}
    BACKEND_READY = False
    BACKEND_ERROR = f"{type(exc).__name__}: {exc}"


st.markdown("""
<div class="cm-hero">
  <div class="cm-eyebrow">Federated fraud intelligence · CSB-04</div>
  <h1>🔐 CIPHERMESH</h1>
  <p>Collaborative fraud learning. Raw transaction rows stay within each simulated bank.</p>
  <span class="cm-chip">OFFICIAL PAYSIM-BANKS DATASET · SYNTHETIC TRANSACTIONS</span>
</div>
""", unsafe_allow_html=True)


def run_view() -> None:
    result, ledger, saved_at = jobs.latest_completed()
    st.session_state["result"] = result
    st.session_state["ledger"] = ledger
    st.session_state["saved_at"] = saved_at
    return result, ledger, saved_at


def results_long(result) -> pd.DataFrame:
    rows = []
    for bank, models in result["per_bank"].items():
        for model_name in ("local", "federated", "pooled"):
            m = models[model_name]
            rows.append({
                "Bank": f"Bank {bank}", "Model": model_name.title(),
                "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
                "PR-AUC": m["pr_auc"], "Threshold": m["threshold"],
                "Fraud flagged": m["flagged_rows"], "Fraud actual": m["fraud_rows"],
                "Test rows": m["test_rows"],
            })
    return pd.DataFrame(rows)


if not BACKEND_READY:
    st.error(f"The official dataset could not be loaded: {BACKEND_ERROR}")
    st.info("First run downloads the official Parquet shards (~276 MB) from "
            "https://huggingface.co/datasets/flwrlabs/fed-fraud-paysim-banks "
            "into `.cache/fedguard/`. Check network access or re-run `python data.py`.")
    st.stop()

with st.sidebar:
    st.markdown("## 🔐 CIPHERMESH")
    st.markdown("**Federated Fraud Intelligence**")
    st.markdown('<div class="small-muted">CSB-04 · Hackathon prototype</div>', unsafe_allow_html=True)
    st.caption(f"Dataset: {DATA_FACTS['total_rows']:,} official transactions · "
               f"{DATA_FACTS['total_fraud_rows']:,} fraud")
    st.divider()
    st.header("Experiment settings")
    st.markdown('<div class="small-muted">Every option below feeds the actual training, '
                'aggregation and evaluation pipeline.</div>', unsafe_allow_html=True)
    available_banks = sorted(int(b) for b in DATA_FACTS["banks"])
    n_banks = st.selectbox("Participating banks (from dataset)", [1, 2, 3, 4, 5], index=4,
                           help="Subset of the dataset BankID partitions (0…4).")
    rows_per_bank = st.slider("Bounded training rows per bank", 500, 12000, 4000, step=500)
    rounds = st.slider("Federated rounds", 1, 30, 20)
    epochs = st.slider("Local gradient epochs per round", 1, 20, 5)
    lr = st.slider("Learning rate", 0.02, 0.30, 0.10, step=0.02)
    use_protection = st.checkbox("Enable update clipping / optional noise", value=True)
    clip_norm = st.slider("Update clipping norm", 0.05, 2.00, 1.00, step=0.05, disabled=not use_protection)
    noise_scale = st.slider("Gaussian noise multiplier (0 = off)", 0.0, 2.0, 0.5, step=0.1, disabled=not use_protection)
    include_balances = st.checkbox("Include balance columns (leakage sensitivity)", value=False)
    run_button = st.button("🚀 Run federated experiment", type="primary", width="stretch")
    if run_button:
        config = RunConfig(
            sample_per_bank=int(rows_per_bank), rounds=int(rounds), local_epochs=int(epochs),
            learning_rate=float(lr), include_balances=bool(include_balances),
            use_protection=bool(use_protection), clip_norm=float(clip_norm),
            dp_noise_multiplier=float(noise_scale), seed=42,
        )
        jobs.start_job(config.__dict__)
        st.success(f"Experiment queued: {jobs.latest_job()['id']}")
    st.caption("Clipping/noise are experimental controls—not a formal differential-privacy guarantee.")
    st.caption("Demo only: no real bank connection or production security controls.")


@st.fragment(run_every="1.5s")
def live_progress() -> None:
    job = jobs.latest_job()
    if job and job["status"] in ("queued", "running"):
        placeholder = st.empty()
        with placeholder.container():
            st.progress(min(1.0, max(0.0, job["progress"])), text=job["message"])
            st.caption(f"Background job {job['id']} — the app stays responsive.")
        if job["status"] == "complete":
            if st.session_state.get("shown_job") != job["id"]:
                st.session_state.shown_job = job["id"]
                st.rerun(scope="app")
    elif job and job["status"] == "failed":
        st.error(f"Experiment failed: {job['error']}")
    elif job and job["status"] == "complete" and st.session_state.get("shown_job") != job["id"]:
        st.session_state.shown_job = job["id"]
        st.rerun(scope="app")


live_progress()

result, ledger, saved_at = jobs.latest_completed()
st.session_state["result"] = result
st.session_state["ledger"] = ledger
st.session_state["saved_at"] = saved_at

st.markdown('<div class="cm-section"><h3>Experiment workspace</h3>'
            '<div class="small-muted">Train local models, aggregate model updates, and evaluate '
            'performance on the official held-out split — bank by bank.</div></div>',
            unsafe_allow_html=True)

if result is None:
    st.info("Choose settings in the sidebar and click **Run federated experiment** to train against "
            "the official dataset and calculate real metrics.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Official transactions", f"{DATA_FACTS['total_rows']:,}")
    c2.metric("Raw rows sent to coordinator", "0")
    c3.metric("Status", "Awaiting run")
    st.markdown("### What this prototype demonstrates")
    st.markdown("1. Each simulated bank trains on its official `BankID` partition locally.\n"
                "2. Only model parameters and approved metadata are passed to the FedAvg coordinator.\n"
                "3. A shared model is evaluated separately on each bank's held-out test set.\n"
                "4. A hash-linked ledger records every step, and artifacts persist to disk.")
    st.warning("This app uses the official PaySim (synthetic) dataset. It is not connected to real "
               "banks and does not prove real-world fraud-detection performance.")
else:
    st.caption(f"Reloaded from saved artifacts · saved {saved_at} · verified ledger: "
               f"{'PASS' if ledger and ledger_module.verify_ledger(ledger) else 'FAIL'}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Participating banks", str(len(result["per_bank"])))
    m2.metric("FedAvg rounds", str(result["config"]["rounds"]))
    m3.metric("Raw rows sent to coordinator", f"{result['raw_transaction_rows_sent_to_coordinator']:,}")
    m4.metric("Model updates submitted", f"{result['model_updates_submitted']:,}")

    tabs = st.tabs(["Overview & metrics", "Data Explorer", "Bank Divergence",
                    "Privacy & Trust Ledger", "Risk scoring demo", "Implementation notes"])

    with tabs[0]:
        st.markdown("### Experiment summary")
        metrics = result["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Pooled baseline PR-AUC", f"{metrics['pooled']['pr_auc']:.4f}")
        c2.metric("Local-only PR-AUC", f"{metrics['local']['pr_auc']:.4f}")
        c3.metric("Federated PR-AUC", f"{metrics['federated']['pr_auc']:.4f}")
        st.caption("PR-AUC (average precision) on the full official test split at natural fraud "
                   "prevalence. Absolute values are small because real fraud is rare (~0.13%).")
        st.markdown("#### Thresholds (tuned on validation, never on test)")
        thr = result["thresholds"]
        st.dataframe(pd.DataFrame([
            {"Model": name.title(), "Review threshold": value,
             "Test PR-AUC": result["metrics"][name]["pr_auc"],
             "Test recall": result["metrics"][name]["recall"],
             "Test F1": result["metrics"][name]["f1"]}
            for name, value in thr.items()
        ]), hide_index=True, width="stretch")
        st.markdown("#### Confusion-matrix counts at the tuned threshold (full test split)")
        st.dataframe(pd.DataFrame([
            {"Model": name.title(), "TP": mval["tp"], "FP": mval["fp"],
             "TN": mval["tn"], "FN": mval["fn"], "Fraud actual": mval["fraud_rows"],
             "Fraud flagged": mval["flagged_rows"]}
            for name, mval in metrics.items()
        ]), hide_index=True, width="stretch")
        st.markdown("#### Per-bank comparison")
        st.dataframe(results_long(result), hide_index=True, width="stretch")
        st.markdown("### Workflow")
        st.markdown("**Local data → local model updates → weighted FedAvg coordinator → shared "
                    "model → bank-wise held-out evaluation**")
        chart = px.bar(results_long(result), x="Bank", y="F1", color="Model", barmode="group",
                       title="F1-score by simulated bank (official test split)")
        st.plotly_chart(chart, width="stretch")

    with tabs[1]:
        st.markdown("### Official dataset structure")
        st.caption(f"Dataset: {DATA_FACTS['dataset_id']} · "
                   f"train {DATA_FACTS['splits']['train']['rows']:,} rows · "
                   f"test {DATA_FACTS['splits']['test']['rows']:,} rows")
        st.markdown("#### Observed schema")
        st.dataframe(pd.DataFrame(DATA_FACTS["schema"]["train"]), hide_index=True, width="stretch")
        st.markdown("#### Per-bank distribution")
        banks = pd.DataFrame([
            {"BankID": b, "Transactions": v["rows"], "Fraud labels": v["fraud_rows"],
             "Fraud rate": v["fraud_rate"], "Mean amount": v["mean_amount"]}
            for b, v in DATA_FACTS["banks"].items()
        ])
        st.dataframe(banks, hide_index=True, width="stretch")
        st.bar_chart(banks.set_index("BankID")[["Fraud rate"]], color="#087f8c")
        st.markdown("#### Feature hygiene used for training")
        st.write("Excluded from features: **isFraud** (target), **isFlaggedFraud** (rule leakage), "
                 "**BankID** (partition key) and customer identifiers **nameOrig/nameDest**. "
                 "Balance columns are excluded by default (measured task leakage).")
        if st.button("Load a reproducible 2,000-row view (from offline cache)"):
            with st.spinner("Reading rows from the offline cache"):
                train, _ = load_frames(offline=True,
                                       columns=["step", "type", "amount", "isFraud", "BankID"])
                view = train.sample(n=min(2000, len(train)), random_state=42)
            st.dataframe(view, hide_index=True, width="stretch")

    with tabs[1 + 1]:  # Bank Divergence
        long = results_long(result)
        metric = st.selectbox("Metric to compare", ["PR-AUC", "Recall", "Precision", "F1"])
        st.plotly_chart(px.bar(long, x="Bank", y=metric, color="Model", barmode="group",
                               title=f"{metric}: local-only vs federated vs pooled"),
                        width="stretch")
        focus_bank = st.selectbox("Focus on one bank", sorted(long["Bank"].unique()))
        piv = (long[long["Bank"] == focus_bank]
               .set_index("Model")[["Precision", "Recall", "F1", "PR-AUC", "Fraud flagged", "Fraud actual"]])
        st.dataframe(piv.round(4), width="stretch")
        st.caption("Federated learning is not guaranteed to improve every bank. Interpret "
                   "synthetics with care.")

    with tabs[1 + 2]:  # Privacy & Trust Ledger
        st.markdown("### Privacy Transparency Panel")
        p1, p2, p3 = st.columns(3)
        p1.metric("Raw transaction rows sent", f"{result['raw_transaction_rows_sent_to_coordinator']:,}")
        p2.metric("Model updates submitted", f"{result['model_updates_submitted']:,}")
        p3.metric("Ledger integrity", "PASS" if ledger and ledger_module.verify_ledger(ledger) else "FAIL")
        st.write("**What the coordinator receives:** sample-count-weighted model deltas plus approved "
                 "metadata (bank id, row count, feature schema). Raw transaction rows are never passed "
                 "to aggregation.")
        st.json(result["privacy"])
        st.warning("Model updates can still reveal information. Zero raw rows does not mean zero "
                   "privacy risk. This local simulation implements neither secure aggregation, "
                   "network transport security, nor a formal differential-privacy guarantee; "
                   "the reported epsilon is a conservative composition estimate.")
        st.markdown("### Hash-linked Trust Ledger")
        ledger_df = pd.DataFrame(ledger)
        st.dataframe(ledger_df[["event_id", "timestamp_utc", "event_type", "actor", "details",
                                "previous_hash", "hash"]], width="stretch")
        st.button("Demonstrate tamper detection",
                  on_click=lambda: st.session_state.update(tamper_demo=True))
        if st.session_state.tamper_demo:
            altered = ledger_module.demo_tamper(ledger)
            st.error("Modified ledger verification: " +
                     ("PASS (unexpected)" if ledger_module.verify_ledger(altered)
                      else "FAIL — tampering detected"))
        st.download_button("Download experiment metrics (CSV)",
                           data=results_long(result).to_csv(index=False).encode("utf-8"),
                           file_name="ciphermesh_metrics.csv", mime="text/csv")
        st.download_button("Download audit ledger (JSON)",
                           data=json.dumps(ledger, indent=2).encode("utf-8"),
                           file_name="ciphermesh_audit_ledger.json", mime="application/json")
        st.download_button("Download model artifact (JSON)",
                           data=json.dumps(result["pooled_model"], indent=2).encode("utf-8"),
                           file_name="ciphermesh_model.json", mime="application/json")

    with tabs[1 + 3]:  # Risk scoring demo
        st.markdown("### Risk scoring (uses the persisted trained model)")
        model = result["pooled_model"]
        st.caption("Scores come from the pooled model actually trained above; nothing is fabricated. "
                   "A risk score is not proof of fraud.")
        with st.form("score_form"):
            fields = st.columns(3)
            numeric_inputs = {}
            for index, numeric_column in enumerate(model["numeric_columns"]):
                if index < 3:
                    with fields[index]:
                        label = {"step": "Step", "amount": "Amount"}.get(numeric_column, numeric_column)
                        value = st.number_input(label, min_value=0.0,
                                                value=1000.0 if numeric_column == "amount" else 0.0)
                        numeric_inputs[numeric_column] = value
            balance_cols = [column for column in model["numeric_columns"] if column.startswith("old") or column.startswith("new")]
            submitted_type = st.selectbox("Transaction type", list(model["type_categories"]))
            for column in balance_cols:
                numeric_inputs[column] = st.number_input(f"{column} balance", min_value=0.0, value=0.0)
            submitted = st.form_submit_button("Score transaction", type="primary")
        if submitted:
            values = {**numeric_inputs, "type": submitted_type}
            score = score_transaction(values, model)
            threshold = result["thresholds"]["pooled"]
            risk = "Elevated — flag for review" if score >= threshold else "Lower"
            st.metric("Model fraud score", f"{score:.4%}")
            st.write(f"Risk band: **{risk}** (pooled review threshold {threshold:.2f})")
            if score >= threshold:
                st.error("Flag for review.")
            else:
                st.success("Below the review threshold.")

    with tabs[1 + 4]:  # Implementation notes
        st.markdown("### What is implemented")
        st.markdown("- Official dataset loading with offline cache (`data.py`).\n"
                    "- Fixed-order feature engineering with target/leakage/identifier exclusion (`preprocess.py`).\n"
                    "- Simulated bank clients that submit only model updates + metadata (`banks.py`).\n"
                    "- Sample-count-weighted FedAvg with schema/shape/finiteness checks (`privacy.py`).\n"
                    "- Local-only, federated and pooled baselines with the same optimizer and epoch budget (`engine.py`).\n"
                    "- Threshold tuning on validation; prec/recall/F1/PR-AUC/confusion counts on held-out test.\n"
                    "- Optional clipping + Gaussian noise and a conservative epsilon estimate (not formal DP).\n"
                    "- Hash-linked audit ledger with tamper detection (`ledger.py`).\n"
                    "- Atomic artifact persistence and reload across sessions (`store.py`, `jobs.py`).")
        st.markdown("### What is not implemented")
        st.markdown("- Real bank connections or distributed servers.\n"
                    "- Secure aggregation, TLS, authentication, access controls or production monitoring.\n"
                    "- Formal differential-privacy guarantees or privacy accounting for sampling.\n"
                    "- Proof that federated learning improves fraud detection in the real world.")
        st.markdown("### Judge-ready explanation")
        st.info("Each simulated bank trains on its own official PaySim partition. The coordinator "
                "receives sample-count-weighted model deltas plus approved metadata — never raw "
                "transaction rows — and combines them with federated averaging. Every model family "
                "uses the same optimizer, epoch budget, held-out evaluation policy and threshold "
                "tuning rule, using the official synthetic dataset. Results demonstrate the "
                "workflow, not real banking performance.")