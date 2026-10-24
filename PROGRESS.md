# 📈 PROGRESS LOG — ISM Customer Retention & Value Optimization DSS

> This log is updated after **every step** following the mandatory four-action cycle:
> **Code & Verify → Update Documentation → Git Commit & Push → Report & Pause.**

## 🗺️ Roadmap Overview

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project Scaffolding & Environment Setup | [x] Completed |
| 2 | Data Acquisition — `src/data_generator.py` (IBM Telco Customer Churn, 7,043 rows) | [x] Completed |
| 3 | ETL Pipeline & SQLite DSS Warehouse — `src/etl_pipeline.py` → `database/ism_dss.db` (3 dimensions + 2 fact tables + views) | [x] Completed |
| 4 | Data Mining Engine — `src/mining_engine.py` (K-Means Value Tier + XGBoost Churn + SHAP) | [ ] Pending |
| 5 | Decision Engine — `src/decision_engine.py` (Decision Matrix, ROI, Revenue at Risk) | [ ] Pending |
| 6 | Streamlit DSS Dashboard — `app.py` (4-Tier ISM Presentation Layer) & Final Documentation | [ ] Pending |

---

## ✅ Step 1 — Project Scaffolding & Environment Setup

- **Status:** [x] Completed
- **Timestamp:** 2026-10-24 08:31 (UTC+7)
- **Commit:** `feat(step-1): project scaffolding, requirements and environment setup and update docs`
- **Objective:** Establish the directory structure directly in the repository root (`ISM-project/`), declare dependencies, configure an isolated virtual environment and initialize project documentation.

### Files / Directories Created or Modified
| Path | Description |
|------|-------------|
| `data/raw/.gitkeep` | Directory for the original raw dataset |
| `data/processed/.gitkeep` | Directory for ETL outputs |
| `database/.gitkeep` | Directory for the SQLite database `ism_dss.db` |
| `models/.gitkeep` | Directory for trained model artifacts (`.joblib`) |
| `src/__init__.py` | Declares `src/` as a Python package |
| `.gitignore` | Excludes `__pycache__/`, `.venv/`, IDE files, logs and secrets |
| `requirements.txt` | numpy, pandas, requests, scikit-learn, xgboost, shap, joblib, matplotlib, plotly, streamlit |
| `PROGRESS.md` | Project progress log (this file) |
| `README.md` | Project overview, 4-Tier architecture, project structure and setup instructions |

### Verification Results
- Python interpreter: **3.9.2**
- Virtual environment `.venv/` created; all packages in `requirements.txt` installed successfully.
- Import test passed: numpy 1.26.4 · pandas 2.3.3 · scikit-learn 1.5.2 · xgboost 2.1.4 · shap 0.46.0 · plotly 7.1.0 · streamlit 1.50.0

---

## ✅ Step 2 — Data Acquisition (`src/data_generator.py`)

- **Status:** [x] Completed
- **Timestamp:** 2026-10-24 08:39 (UTC+7)
- **Commit:** `feat(step-2): add telco dataset ingestion and convert all docs to English`
- **Objective:** Acquire the original IBM Telco Customer Churn dataset into the raw data layer with schema validation and a resilient offline fallback.

### Files Created or Modified
| Path | Description |
|------|-------------|
| `src/data_generator.py` | Data acquisition module: multi-mirror download, schema validation (21 columns, 7,043 rows), synthetic fallback generator, data-quality profiling; CLI flags `--force` and `--synthetic` |
| `data/raw/telco_churn_raw.csv` | Original raw dataset (stored unmodified; `TotalCharges` kept as text to preserve blank values) |
| `README.md` | Fully rewritten in academic English; added Dataset, Database Schema (planned) and Retention Decision Matrix (planned) sections |
| `PROGRESS.md` | Fully rewritten in academic English; added Step 2 log |

### Implementation Highlights
- **Primary source:** official IBM GitHub repository; two community mirrors of the Kaggle file are configured as secondary sources.
- **Schema validation:** the downloaded file is accepted only if its column list and row count exactly match the reference schema.
- **Synthetic fallback:** a seeded (`seed=42`) generator reproduces the 21-column schema, categorical domains, 11 zero-tenure customers with blank `TotalCharges`, and a logistic churn mechanism driven by contract type, tenure, internet service, payment method and add-on services.

### Verification Results (command: `.venv\Scripts\python.exe src/data_generator.py --force`)
| Metric | Value |
|--------|-------|
| Data source | IBM Telco Customer Churn (original, downloaded from the IBM repository) |
| Rows | **7,043** |
| Columns | **21** |
| Unique `customerID` | 7,043 |
| Churn = Yes | **1,869 (26.54%)** |
| Churn = No | 5,174 (73.46%) |
| Blank/whitespace `TotalCharges` | **11** |
| Customers with `tenure = 0` | 11 (identical to the blank `TotalCharges` records) |
| Duplicate rows | 0 |
| Synthetic fallback self-test | 7,043 × 21, churn rate 25.78%, 11 blank `TotalCharges`, schema validation passed |

### Notes for Step 3 (ETL)
- The 11 blank `TotalCharges` values must be converted to numeric (imputed as `0.0` because `tenure = 0`).
- `SeniorCitizen` is already binary (0/1); the remaining binary columns use `Yes`/`No` encoding.
- Moderate class imbalance (≈ 26.5% churn) must be addressed during model training in Step 4 (e.g. `scale_pos_weight`, stratified splitting).

---

## ✅ Step 3 — ETL Pipeline & SQLite DSS Warehouse (`src/etl_pipeline.py`)

- **Status:** [x] Completed
- **Timestamp:** 2026-10-24 08:59 (UTC+7)
- **Commit:** `feat(step-3): build ETL pipeline, feature engineering, and initialize SQLite DSS warehouse`
- **Objective:** Cleanse and enrich the raw dataset, then load it into a normalized SQLite decision-support warehouse whose fact-table structures are pre-defined for Step 4 (Mining) and Step 5 (Decision Engine).

### Files Created or Modified
| Path | Description |
|------|-------------|
| `src/etl_pipeline.py` | Extract → Transform (cleaning + feature engineering) → Load (DDL + constrained inserts) → Validate & Export; writes an audit record per run |
| `database/ism_dss.db` | SQLite DSS warehouse (≈ 1.8 MB) |
| `data/processed/telco_churn_processed.csv` | Cleaned, feature-engineered dataset exported from `customer_master` (7,043 × 26, ≈ 1.0 MB) |
| `README.md` | Replaced the planned schema with the implemented DSS warehouse ERD, object catalogue, cleaning rules, engineered-feature dictionary and the Step 3 run command |
| `PROGRESS.md` | Added Step 3 log |

### Warehouse Objects
| Object | Type | Rows | Notes |
|--------|------|------|-------|
| `customers` | Dimension table | 7,043 | PK `customerID`; demographics, `tenure_group`, `churn`, `churn_binary` |
| `services` | Dimension table | 7,043 | PK/FK `customerID`; services and `total_services_count` |
| `billing` | Dimension table | 7,043 | PK/FK `customerID`; payment, charges, `avg_monthly_spend`, `is_auto_payment` |
| `customer_master` | View | 7,043 | Integrated cleaned record (26 columns) |
| `fact_mining_scores` | Fact table (DDL) | 0 | `customerID`, `value_tier`, `cluster_id`, `churn_probability`, `risk_level`, `top_shap_driver_1`, `top_shap_driver_2`, `scored_at`; index on (`value_tier`, `risk_level`) |
| `fact_retention_actions` | Fact table (DDL) | 0 | `customerID`, `annualized_revenue`, `revenue_at_risk`, `decision_quadrant`, `recommended_action`, `intervention_cost`, `expected_saved_revenue`, `expected_net_benefit`, `expected_roi`, `decided_at`; index on `decision_quadrant` |
| `vw_customer_360` | View | 7,043 | `customer_master` LEFT JOIN both fact tables (dashboard data source) |
| `etl_audit_log` | Audit table | 1 per run | Run timestamp, source, rows extracted/loaded, imputations, duplicates, status |

### Engineered Features
| Feature | Definition | Observed Result |
|---------|------------|-----------------|
| `tenure_group` | `0-12m`, `12-24m`, `24-48m`, `48m+` | 2,186 / 1,024 / 1,594 / 2,239 customers; churn rate 47.4% / 28.7% / 20.4% / 9.5% |
| `total_services_count` | Count of 6 value-added services subscribed | Mean 2.04, range 0–6 |
| `avg_monthly_spend` | `total_charges / tenure` (`monthly_charges` if `tenure = 0`) | Mean 64.76, range 13.78–121.40 |
| `is_auto_payment` | Automatic bank transfer or credit card | 43.5% of customers |
| `churn_binary` | 1 if `churn = 'Yes'` else 0 | Mean 26.54% (consistent with raw data) |

### ETL Statistics & Verification Results (command: `.venv\Scripts\python.exe src/etl_pipeline.py`)
| Check | Result |
|-------|--------|
| Rows extracted / loaded | 7,043 / 7,043 |
| Blank `TotalCharges` imputed (→ 0.0) | 11 (all with `tenure = 0`) |
| Duplicate `customerID` removed | 0 |
| Missing values in `customer_master` and processed CSV | **0** |
| Row count `customers` = `services` = `billing` = `customer_master` = `vw_customer_360` | 7,043 |
| `PRAGMA foreign_key_check` violations | 0 |
| Foreign-key enforcement test (insert unknown `customerID` into `fact_mining_scores`) | Rejected ✔ |
| `CHECK` constraint test (`churn_probability = 1.5`) | Rejected ✔ |
| Idempotency (pipeline executed twice) | Identical results; audit log records both runs ✔ |

### Notes for Step 4 (Mining Engine)
- Read model inputs from `customer_master`; multi-category attributes require one-hot encoding inside the modelling pipeline.
- Write outputs to `fact_mining_scores` using `value_tier` / `risk_level` domains `'High'`, `'Medium'`, `'Low'` (enforced by `CHECK` constraints).

---
