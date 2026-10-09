# CIPHERMESH — Professional Demo UI

## Run locally (Kali Linux / Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```
Open the localhost URL printed in the terminal, normally `http://localhost:8501`.

## Demo flow for judges
1. Click **Run federated experiment**.
2. Show the local-only vs federated metrics by bank.
3. Open **Privacy & Trust Ledger** and explain that only model parameters are passed to the coordinator in this local simulation.
4. Click **Demonstrate tamper detection**.
5. Open the risk-scoring demo and explain that the score is for review, not a fraud verdict.

## Honest scope
This is a prototype using generated synthetic data. It simulates banks within one Python process. It does not connect to any real bank, and it is not production-ready. Do not describe its results as evidence of real-world fraud detection improvement.
