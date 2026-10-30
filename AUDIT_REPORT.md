# FleetGuard AI — Phase 1 Project Audit Report

**Project:** FleetGuard AI — Intelligent Predictive Maintenance for Connected Fleets
**Hackathon:** Teliport Season 4 — Tata Elxsi (Case Study 1: Self-Healing Machines)
**Audit date:** 2026-10-30
**Repository:** `C:\Users\HP\fleetguard-ai` (branch `main`, clean working tree)
**Live deployment:** https://fleetguard-ebbp.onrender.com/ (not yet re-tested after this audit)

---

## 1. Executive Summary

The existing FleetGuard AI codebase is a **functional Streamlit prototype** with a real predictive maintenance stack: a deterministic fleet simulator, rule-based anomaly detection, health scoring, rule-based maintenance triage with automatic work-order generation, and a NASA C-MAPSS RUL prediction pipeline (Random Forest + linear baseline). All 66 existing pytest tests pass (1 skipped).

It is **not yet 100/100 hackathon-ready**. The audit found missing or incomplete capability in every phase area: no RUL clipping or NASA asymmetric scoring, model artifacts are not persisted to disk (only metadata in DB), no per-engine or per-RUL-range error analysis, no alembic migrations, no duplicate work-order prevention, simulation is not deterministic under a single fixed seed, the UI theme is light-primary rather than the specified dark navy/charcoal industrial aesthetic, several required documentation files are missing, and MLflow is broken in the local environment (missing `fastapi`).

No fake metrics or hardcoded predictions were found — all metrics originate from actual model evaluation, and simulated data is explicitly labelled. These are strengths to preserve.

---

## 2. Inventory by Layer

| Layer | Files | Status |
|-------|-------|--------|
| Entry point / UI shell | `app/main.py` | Working; 7-page navigation, sidebar demo controls, session state |
| Styling | inline CSS in `app/main.py` | Working but light-themed; needs dark navy/charcoal industrial aesthetic |
| Database models | `app/models/database.py` | Working; 12 normalized tables with FKs, indexes, enums |
| DB connection | `app/models/connection.py` | Working; PostgreSQL + NullPool; requires `DATABASE_URL` |
| Anomaly detection | `app/ml/anomaly_detection.py` | Working; threshold, statistical, Isolation Forest, trend |
| Health scoring + triage | `app/ml/health_scoring.py` | Working; heuristic health score, 12 deterministic rules, auto work orders |
| RUL prediction | `app/ml/rul_prediction.py` | Working; C-MAPSS ingestion, feature engineering, RF/linear train/eval |
| Simulation | `app/simulation/engine.py` | Working; 10 seed vehicles, 5 fault types, per-vehicle deterministic seeds |
| Fleet service | `app/services/fleet_service.py` | Working; orchestrates sim + triage + work orders |
| Pages | `app/pages/{overview,fleet,vehicle_detail,predictions,alerts,maintenance,data_management}.py` | All 7 render; some N+1 query patterns; no MLOps-specific page |
| Seed data | `scripts/init_db.py` | Working; 10 vehicles, 5 technicians, initial readings + health history |
| Config | `requirements.txt`, `.env.example`, `Dockerfile`, `render.yaml`, `docker-compose.yml`, `entrypoint.sh` | Working; `psycopg v3` missing locally (only psycopg2-binary); `mlflow` broken (missing fastapi) |
| Tests | `tests/{test_anomaly_detection,test_database,test_health_scoring,test_rul_prediction,test_simulation}.py`, `conftest.py` | 66 passed, 1 skipped; covers models, detectors, triage, simulator |

---

## 3. Feature Checklist

### Existing and working (verified by tests or runtime)

- Fleet simulator: start/pause/reset, 10 assets (6 trucks, 2 buses, 2 more), sensor generation (temp, RPM, oil pressure, coolant, vibration, battery, fuel)
- Fault injection: overheating, abnormal vibration, low oil pressure, battery deterioration, gradual degradation
- Anomaly detection: threshold breach, statistical outlier (rolling z-score), Isolation Forest, trend anomaly
- Health score (0–100, heuristic, explicitly labelled as such in README) with green/yellow/red status
- Rule-based maintenance triage (12 rules, no LLM), severity, recommended action, evidence
- Automatic work-order creation from critical/high alerts
- Work order lifecycle: Open → Assigned → In Progress → Completed (note: spec wants Resolved + Closed)
- Technician assignment, maintenance history
- NASA C-MAPSS FD001 ingestion, RUL label generation, rolling-window + statistical features
- Train/validation split by engine unit (no data leakage), scaler fit on train only
- Model comparison: Random Forest vs Linear Baseline, with baseline metrics
- Actual vs predicted RUL plot, residual plot, feature importance, MAE/RMSE/R²
- MLflow experiment tracking instrumentation (graceful degradation when unavailable)
- Dataset upload, preview, validation; model registry with activate/archive
- Alert centre with severity/status/time filters, acknowledge/resolve actions
- Database persistence (PostgreSQL in production), Docker/Render deployment config
- "SIMULATED DATA" and "DEMO MODE" labels present; NASA model not applied to truck sensors

### Existing but broken / degraded

| Item | Problem |
|------|---------|
| `mlflow` import | Fails locally (`ModuleNotFoundError: fastapi`); `rul_prediction.py` degrades gracefully but `predictions.py` calls `import mlflow` directly inside a try/except, so predictions tab surfaces an error UI instead of silent degradation |
| `datetime.utcnow()` | Deprecated in Python 3.15 (still works, emits DeprecationWarning) |
| `WorkOrderStatus.COMPLETED` | Spec (Phase 6/10) requires statuses: Open, Assigned, In Progress, **Resolved**, **Closed**. "Completed" is not in the spec list |
| Health score circle CSS | Uses `var(--score)` which is never set — the conic-gradient ring does not render correctly |
| Work order link in alerts | `/?page=maintenance&wo={id}` is not a valid Streamlit navigation mechanism |
| Duplicate work orders | `alert_id` column has a `unique=True` constraint, but `auto_generate_work_orders` has no idempotency guard beyond the alert→WO 1:1 link; repeated processing of the same alert event could race |

### Missing (not implemented)

| Item | Phase | Detail |
|------|-------|--------|
| RUL clipping (piecewise target) | 2 | No configurable RUL clipping threshold |
| NASA asymmetric scoring function | 3 | No $score = \exp(-\ldots)$ asymmetric error metric |
| Model artifact persistence to disk | 3 | Model/scaler/feature config saved as DB metadata only; no joblib/pickle artifact, so inference cannot reload a trained model without retraining |
| Prediction error by engine | 3 | No per-engine error breakdown on the test set |
| Prediction error across RUL ranges | 3 | No error-binned analysis (e.g., low/medium/high RUL) |
| Training time tracking | 3 | No training duration recorded |
| Dataset versioning | 3 | No dataset version/hash tracking |
| Data validation on upload | 2 | Column count and missing-value checks exist in UI preview, but no structured validation (types, duplicates, invalid sensor values) persisted or enforced |
| Sorting by unit + cycle on ingest | 2 | Not explicitly enforced in the processor |
| Preprocessing config saved for inference | 2 | Feature column list stored in DB but scaler not persisted; inference path re-fits or assumes |
| FD002–FD004 support | 2 | UI lists them, processor is FD001-column compatible but operating-settings handling not exercised |
| Model promotion / rollback | 9 | Activate/archive exists in UI but no formal promotion workflow with validation against production model |
| Data drift / prediction drift monitoring | 9 | Not implemented |
| Retraining workflow | 9 | Train button exists; no scheduled/triggered retraining with production comparison |
| Alembic migrations | 10 | `create_all()` only; no versioned migrations |
| Dark navy/charcoal industrial theme | 8 | CSS is light-primary with a dark media query |
| Responsive mobile layout | 8 | Not addressed |
| Fixed-seed deterministic simulation | 7 | Per-vehicle seeds exist but global `random.seed()` is used; no single fixed-seed mode |
| Fault severity adjustment | 7 | Fault injection is on/off; no severity slider |
| Per-vehicle fault selection in UI | 7 | Exists in data_management page, not in the demo control panel |
| Human approval in triage | 6 | Work orders auto-create; no explicit human approval step |
| Configurable rules (external config) | 6 | Rules hardcoded in `_initialize_rules()` |
| Docs: ARCHITECTURE.md, ML_METHODOLOGY.md, DATA_DICTIONARY.md, DEMO_GUIDE.md, DEPLOYMENT.md, TEST_REPORT.md, AUDIT_REPORT.md | 14 | Only README.md exists |

### Implemented during this upgrade (planned)

See Section 8 (Prioritized Improvement Plan).

### Verified through tests

- 66 tests pass across anomaly detection, database models, health scoring/triage, RUL processor, and simulation engine
- Edge cases covered: insufficient history, zero std dev, untrained ML model, missing readings, duplicate vehicle_id constraint, model relationships

---

## 4. Detailed Findings by Phase

### Phase 1 — Audit
This report. Repository inspected end-to-end; app imports cleanly; tests pass; no PostgreSQL available locally for a full end-to-end runtime test (SQLite fallback possible for local verification).

### Phase 2 — ML Pipeline
- **Present:** CSV/Text upload, C-MAPSS column schema (26 cols), RUL label generation (`max_cycle - cycle`), rolling-window features (mean/std/min/max/trend), sensor statistics, cycle_norm, setting interactions, engine-level train/val split, scaler fit on train only.
- **Missing:** missing-value handling strategy, duplicate-record identification, invalid sensor value checks, explicit sort by unit+cycle, configurable RUL clipping, preprocessing config persistence for inference, FD002–FD004 exercised.

### Phase 3 — Model Development & Evaluation
- **Present:** Linear baseline, Random Forest regressor, MAE/RMSE/R², baseline comparison, actual-vs-predicted plot, residual plot, feature importance, model versioning in DB, dataset upload records.
- **Missing:** NASA asymmetric scoring, per-engine error, RUL-range error binned analysis, training time, dataset version, persisted model artifacts (joblib), explicit error display if training fails (partially present — try/except shows error).

### Phase 4 — Anomaly Detection Engine
- **Present:** Isolation Forest, statistical thresholds, rolling mean/std, threshold-based detection, severity, anomaly score, affected sensors, description, detection timestamp. Threshold alerts vs ML anomalies are distinguishable via `anomaly_type`.
- **Gap:** No explicit "sensor drift" or multivariate expected-range model beyond Isolation Forest; every unusual reading is not labelled as imminent failure (good).

### Phase 5 — Explainable Health and RUL Engine
- **Present:** Health score + factors, anomaly status, sensor deviations, risk category (status), recommendations, evidence; green/yellow/red; heuristic explicitly labelled; Mode A / Mode B separation documented in UI warning.
- **Gap:** No RUL output for simulated fleet (correctly shows none); health score not described as heuristic on every display surface.

### Phase 6 — Rule-Based Maintenance Triage Agent
- **Present:** 12 deterministic rules (no LLM), rule ID, sensor values, thresholds, severity, reason, recommended action, timestamp; work orders with ID, asset, fault, severity, evidence, action, technician, timestamps, status; statuses Open/Assigned/In Progress/Completed.
- **Gaps:** statuses should be Resolved + Closed per spec; no human-approval gate; rules not externally configurable; no explicit duplicate-work-order prevention beyond the unique constraint.

### Phase 7 — Live Fleet Simulator
- **Present:** 10 assets with ID/model/hours/temp/RPM/vibration/oil pressure/coolant/battery/mileage/health; 7 scenarios (normal, degradation, overheating, vibration, low oil, multi-fault via sequential injection, sensor drift not explicit, recovery via clear faults); start/pause/reset; vehicle selection via fleet table; fault injection buttons.
- **Gaps:** no fault severity adjustment; no explicit "select vehicle" + "trigger fault" flow in demo panel; not deterministic under one fixed seed; "SIMULATED DATA" label present.

### Phase 8 — Premium Industrial Dashboard
- **Present:** 8 pages worth of content across 7 pages; KPI cards, charts, severity badges, work-order cards.
- **Gaps:** light-themed (not dark navy/charcoal), excessive cards/gradients in places, not mobile-responsive, no dedicated MLOps page (model registry tab lives inside Data & Models), no dedicated anomaly center page (alerts page covers it).

### Phase 9 — Real MLOps
- **Present:** MLflow experiment ID/run ID, params, metrics, timestamps, model artifact logging (when mlflow works), model registry (activate/archive), metrics comparison.
- **Gaps:** mlflow broken locally; no model promotion/rollback with production comparison; no drift monitoring; no retraining workflow.

### Phase 10 — Database & Backend Quality
- **Present:** normalized tables, PK/FK, indexes, timestamps, PostgreSQL, env-var config, `init_db.py`.
- **Gaps:** no alembic migrations, `datetime.utcnow()` deprecated, no duplicate-work-order idempotency, no graceful DB-failure handling in UI pages.

### Phase 11 — Hackathon Demonstration Mode
- **Present:** demo control panel with start/stop/reset, fault injection buttons, work-order generate button, live indicator, "SIMULATED DATA" labels.
- **Gaps:** no structured DEMO A–D scenarios, no reset-to-initial-demo-state button, no 3-minute judge flow guidance.

### Phase 12 — Testing & Reliability
- **Present:** 66 tests passing.
- **Gaps:** no API tests (no API layer), no duplicate event handling test at integration level, no DB-unavailable error-handling test, no model-unavailable inference test, no insufficient-history test for RUL inference.

### Phase 13 — Performance & Deployment
- **Present:** Dockerfile, render.yaml, docker-compose, entrypoint, bind 0.0.0.0, PORT env, PostgreSQL, health check.
- **Gaps:** no .env file locally; mlflow broken in image build (fastapi missing → `import mlflow` may fail at runtime in `predictions.py`); large models not a concern (sklearn RF small); no caching.

### Phase 14 — Documentation
- **Present:** README.md (comprehensive).
- **Missing:** ARCHITECTURE.md, ML_METHODOLOGY.md, DATA_DICTIONARY.md, DEMO_GUIDE.md, DEPLOYMENT.md, TEST_REPORT.md, AUDIT_REPORT.md (this report).

### Phase 15 — Final Evaluation
Deferred to completion of the upgrade (see Section 9).

---

## 5. Prioritized Improvement Plan

**Priority 1 — Critical correctness (fix before cosmetic):**
1. Fix `WorkOrderStatus` to include `RESOLVED` and `CLOSED` (spec compliance); migrate UI references from `COMPLETED`.
2. Add duplicate work-order prevention (idempotency) in `auto_generate_work_orders`.
3. Fix mlflow runtime path in `predictions.py` so a broken mlflow doesn't hard-error the tab.
4. Replace `datetime.utcnow()` with timezone-aware `datetime.now(timezone.utc)`.
5. Add alembic migrations (initial migration = current schema).

**Priority 2 — ML pipeline (Phase 2/3):**
6. Add dataset validation: column/dtype check, missing values, duplicates, invalid sensor values, sort by unit+cycle; store validation results.
7. Add configurable RUL clipping (piecewise target).
8. Add NASA asymmetric scoring function.
9. Persist model artifacts (joblib: model + scaler + feature columns) to disk/DB; reload for inference without retraining.
10. Add per-engine error, RUL-range binned error, training time, dataset version.

**Priority 3 — Simulator & triage (Phase 6/7):**
11. Fixed-seed deterministic simulation mode; fault severity slider; per-vehicle fault trigger from demo panel.
12. Human-approval step for work-order creation; externally configurable triage rules.

**Priority 4 — Dashboard & MLOps (Phase 8/9):**
13. Dark navy/charcoal industrial CSS theme; responsive layout.
14. Model promotion/rollback workflow with production comparison; drift monitoring placeholder; separate MLOps page.

**Priority 5 — Testing & docs (Phase 12/14):**
15. Add tests for validation, duplicate events, DB-unavailable, model-unavailable, insufficient history.
16. Write ARCHITECTURE.md, ML_METHODOLOGY.md, DATA_DICTIONARY.md, DEMO_GUIDE.md, DEPLOYMENT.md, TEST_REPORT.md.
17. Demo mode: DEMO A–D scenarios + reset-to-initial-state button.

**Priority 6 — Deploy & verify (Phase 13/15):**
18. Test locally end-to-end; test deployed URL; final judge scorecard.

---

## 6. Risks

- **No PostgreSQL locally:** end-to-end runtime testing must use SQLite fallback or a Docker postgres container. Deployment to Render uses PostgreSQL; the connection code supports it via `DATABASE_URL`.
- **mlflow broken locally:** any code path that does a bare `import mlflow` outside a try/except will crash. `rul_prediction.py` guards it; `predictions.py` does not fully.
- **pandas 3.1.0 / sklearn 1.9.1 / numpy 2.5.4:** newer than the requirements file pins (`pandas>=2.1.0`, `scikit-learn>=1.3.0`, `numpy>=1.26.0`); code was already patched for `applymap→map`. Verify no other deprecations surface at runtime.
- **Do not claim 100/100 or perfect accuracy:** all metrics must come from real evaluation; simulated truck data must never be presented as NASA-validated.

---

## 7. Definition of Done for This Upgrade

- Judge can trigger a fault → observe actual detection → understand why (rule ID + evidence) → see a valid RUL prediction where supported (NASA benchmark) → watch an actionable work order auto-generate.
- All tests pass; no fabricated metrics.
- Simulated data labelled everywhere; NASA model never applied to truck sensors.
- Deployment config updated and deployment tested at the live URL.
- All 8 documentation files present.
- Final judge scorecard with evidence, limitations, and deductions per category.
