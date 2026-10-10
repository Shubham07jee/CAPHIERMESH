# CIPHERMESH: Path to Real-Bank Deployment

## Reality check
A student prototype cannot be connected to real banks simply by changing the website design. Banks require formal authorization, data-sharing and processing agreements, architecture/security reviews, and controlled testing. Do not connect to real customer data or bank systems without written approval.

## Proposed production architecture

```text
Bank A secure environment                 Bank B secure environment
┌──────────────────────────┐               ┌──────────────────────────┐
│ Approved transaction data│               │ Approved transaction data│
│ Local feature pipeline   │               │ Local feature pipeline   │
│ Local fraud model        │               │ Local fraud model        │
│ Update clipping / checks │               │ Update clipping / checks │
└────────────┬─────────────┘               └────────────┬─────────────┘
             │ authenticated encrypted update channel  │
             └──────────────────┬──────────────────────┘
                                ▼
                 ┌───────────────────────────┐
                 │ Controlled coordinator   │
                 │ Authenticated participants│
                 │ Update validation        │
                 │ Secure aggregation       │
                 │ Audit + monitoring       │
                 └─────────────┬─────────────┘
                               │ global model
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
          Bank A validation            Bank B validation
          Human-reviewed alerts       Human-reviewed alerts
```

## Required work before any real pilot

1. **Governance and authorization**
   - Identify a bank sponsor and approved use case.
   - Complete legal, privacy, risk, and information-security reviews.
   - Define permitted data, retention, purpose, participant responsibilities, and incident response.
   - Agree on model ownership, audit access, and withdrawal procedures.

2. **Data and model validation**
   - Use bank-approved data dictionaries and label definitions.
   - Validate feature availability at decision time and eliminate label leakage.
   - Establish a baseline using each bank's existing approved model.
   - Evaluate PR-AUC, recall, precision, false-positive rates, calibration, and performance by bank.
   - Test data drift, non-IID data, class imbalance, and poisoning/faulty updates.

3. **Secure communications and identity**
   - Deploy authenticated service-to-service communication with TLS/mTLS.
   - Use bank-managed identities, secrets management, least-privilege access, and key rotation.
   - Do not put credentials or customer data in source control or logs.

4. **Privacy protections**
   - Perform a privacy threat model for model updates.
   - Consider secure aggregation and a formally specified differential-privacy mechanism with privacy accounting.
   - Set clipping/noise parameters based on a documented threat model and measured utility impact.
   - Have independent reviewers assess leakage risk.

5. **Coordinator hardening**
   - Validate update schema, sizes, numeric values, participant identity, round membership, and replay protection.
   - Add rate limits, monitoring, alerts, backup, recovery, and immutable or independently protected audit storage.
   - Define defenses against malicious participants and model poisoning.

6. **Controlled pilot**
   - Begin with synthetic data, then bank-approved de-identified or controlled data where permitted.
   - Run in shadow mode; do not automatically block or decline payments.
   - Require human investigation for alerts.
   - Define go/no-go criteria, rollback, monitoring, and incident escalation.

## Recommended engineering sequence
- Phase 1: polished local prototype with reproducible tests.
- Phase 2: service API and separate simulated bank clients in isolated processes/containers.
- Phase 3: authenticated encrypted transport, participant identity, secure deployment and observability.
- Phase 4: privacy review, secure aggregation and adversarial testing.
- Phase 5: bank-authorized shadow pilot and independent evaluation.

## Claims to avoid
- “Banks can deploy this immediately.”
- “Raw data never leaves the bank” unless the architecture and telemetry verify it end to end.
- “100% privacy” or “zero privacy risk.”
- “Fraud detection improves” unless measured against a valid baseline on appropriate test data.
- “Compliant with all banking regulations” without formal review and evidence.
