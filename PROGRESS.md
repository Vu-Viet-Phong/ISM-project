# 📈 PROGRESS LOG — ISM Customer Retention & Value Optimization DSS

> This log is updated after **every step** following the mandatory four-action cycle:
> **Code & Verify → Update Documentation → Git Commit & Push → Report & Pause.**

## 🗺️ Roadmap Overview

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project Scaffolding & Environment Setup | [x] Completed |
| 2 | Data Acquisition — `src/data_generator.py` (IBM Telco Customer Churn, 7,043 rows) | [x] Completed |
| 3 | ETL Pipeline & SQLite Database — `src/etl_pipeline.py` → `database/ism_dss.db` (3 normalized tables) | [ ] Pending |
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
