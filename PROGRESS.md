# 📈 PROGRESS LOG — ISM Customer Retention & Value Optimization DSS

> This log is updated after **every step** following the mandatory four-action cycle:
> **Code & Verify → Update Documentation → Git Commit & Push → Report & Pause.**

## 🗺️ Roadmap Overview

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project Scaffolding & Environment Setup | [x] Completed |
| 2 | Data Acquisition — `src/data_generator.py` (IBM Telco Customer Churn, 7,043 rows) | [x] Completed |
| 3 | ETL Pipeline & SQLite DSS Warehouse — `src/etl_pipeline.py` → `database/ism_dss.db` (3 dimensions + 2 fact tables + views) | [x] Completed |
| 4 | Data Mining Engine — `src/mining_engine.py` (K-Means Value Tier + XGBoost Churn + SHAP) | [x] Completed |
| 5 | Decision Engine — `src/decision_engine.py` (Decision Matrix, ROI, Revenue at Risk) | [x] Completed |
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

## ✅ Step 4 — Data Mining Engine (`src/mining_engine.py`)

- **Status:** [x] Completed
- **Timestamp:** 2026-10-24 09:21 (UTC+7)
- **Commit:** `feat(step-4): implement K-Means value tiering, XGBoost churn model, and SHAP explainability`
- **Objective:** Train the customer value segmentation pipeline (K-Means), the churn prediction pipeline (XGBoost) and the explainability module (SHAP TreeExplainer); persist model artifacts to `models/` and write all 7,043 customer score records into `fact_mining_scores`.

### Files Created or Modified
| Path | Description |
|------|-------------|
| `src/mining_engine.py` | End-to-end data mining engine (K-Means value tiering, XGBoost churn classification, SHAP local/global explainability, SQLite persistence, single-customer inference helper `score_customer`) |
| `models/kmeans_value_pipeline.joblib` | Fitted `StandardScaler + KMeans(n_clusters=3, random_state=42)` pipeline and cluster-to-tier mapping |
| `models/xgb_churn_pipeline.joblib` | Fitted `ColumnTransformer (OneHotEncoder + StandardScaler) + XGBClassifier` pipeline and risk thresholds |
| `models/shap_metadata.joblib` | Aggregated SHAP values (`7,043 × 23`), global feature importance, expected value and feature mappings |
| `models/model_metrics.json` | Hold-out evaluation metrics, Silhouette scores, cluster profiles, SHAP global importance and cross-tabulations |
| `database/ism_dss.db` | Populated `fact_mining_scores` with 7,043 customer score records |
| `README.md` | Added Section 5 (Data Mining Engine experimental results, cluster profiles, XGBoost metrics, SHAP top features, Value Tier × Risk Level matrix) |
| `PROGRESS.md` | Added Step 4 log |

### Module 1 — Customer Value Tiering (K-Means Clustering)
- **Features:** `tenure`, `monthly_charges`, `total_charges`, `total_services_count` (standardized via `StandardScaler`).
- **Silhouette scores ($k = 2 \dots 6$):** $k=2$: `0.459`, $k=3$: **`0.402`**, $k=4$: `0.396`, $k=5$: `0.376`, $k=6$: `0.365`.
- **Cluster-to-Tier mapping:** `{0: 'High', 2: 'Medium', 1: 'Low'}` (ranked by mean `total_charges` and `monthly_charges`).

| Value Tier | Cluster ID | Customers | Mean Tenure (m) | Mean Monthly Charges (\$) | Mean Total Charges (\$) | Mean VAS Count | Observed Churn Rate | Monthly Revenue Share |
|------------|------------|-----------|-----------------|---------------------------|-------------------------|----------------|---------------------|-----------------------|
| **High** | 0 | 2,211 | 58.395 | 89.343 | 5,207.532 | 3.980 | 14.74% | 43.31% |
| **Medium** | 2 | 2,708 | 15.319 | 74.961 | 1,143.356 | 1.891 | 45.61% | 44.51% |
| **Low** | 1 | 2,124 | 27.022 | 26.169 | 680.840 | 0.203 | 14.50% | 12.19% |

### Module 2 — Churn Prediction Model (XGBoost Classifier)
- **Pipeline:** `ColumnTransformer` (`OneHotEncoder(handle_unknown="ignore")` on 12 categorical columns + `StandardScaler` on 11 numeric columns → 48 encoded features) + `XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0, scale_pos_weight=2.7686, random_state=42)`.
- **Hold-out evaluation (stratified 80/20 split: 5,634 train / 1,409 test):**

| Metric | Value |
|--------|-------|
| **ROC-AUC (Test)** | **0.8423** (Train: 0.9038) |
| **PR-AUC (Test)** | **0.6651** |
| **Recall / Sensitivity** | **0.7701** |
| **Precision** | **0.5284** |
| **F1-Score** | **0.6268** |
| **Accuracy** | **0.7566** |
| **Brier Score** | **0.1612** |
| **Confusion Matrix (Test)** | `TN = 778`, `FP = 257`, `FN = 86`, `TP = 288` |

### Module 3 — Explainable AI (SHAP TreeExplainer)
- Computed exact TreeSHAP values (`7,043 × 48`, log-odds units) via `shap.TreeExplainer` (`shap 0.46.0`) and aggregated one-hot contributions back to the 23 business attributes.
- **Top 5 Global SHAP Features (mean $|\text{SHAP}|$):**
  1. `contract`: **0.7832**
  2. `tenure`: **0.4331**
  3. `online_security`: **0.2600**
  4. `internet_service`: **0.2146**
  5. `monthly_charges`: **0.2069**
- Extracted per-customer `top_shap_driver_1` (populated for 7,033 customers with positive drivers) and `top_shap_driver_2` (e.g., `tenure=1 (+0.88)`, `contract=Month-to-month (+0.61)`).

### Cross-Tabulation: `value_tier` × `risk_level` (7,043 customers)
| Value Tier \ Risk Level | High ($p \ge 0.60$) | Medium ($0.30 \le p < 0.60$) | Low ($p < 0.30$) | All |
|-------------------------|---------------------|------------------------------|------------------|-----|
| **High** | 266 | 550 | 1,395 | 2,211 |
| **Medium** | 1,658 | 626 | 424 | 2,708 |
| **Low** | 338 | 367 | 1,419 | 2,124 |
| **All** | 2,262 | 1,543 | 3,238 | 7,043 |

### Database & View Verification
- `fact_mining_scores`: **7,043 rows** inserted atomically (`DELETE` + `INSERT` within a transaction with `PRAGMA foreign_keys = ON`).
- `vw_customer_360` join check: `rows = 7,043`, `value_tier = 7,043` non-null, `churn_probability = 7,043` non-null, `top_shap_driver_1 = 7,033` non-null.

### Notes for Step 5 (Decision Engine)
- Read scored records from `vw_customer_360` (or join `customer_master` with `fact_mining_scores`).
- Compute `annualized_revenue`, `revenue_at_risk`, map each (`value_tier`, `risk_level`) pair to its `decision_quadrant` and `recommended_action`, evaluate `intervention_cost`, `expected_saved_revenue`, `expected_net_benefit`, `expected_roi`, and populate `fact_retention_actions`.

---

## ✅ Step 5 — Decision Support & ROI Optimization Engine (`src/decision_engine.py`)

- **Status:** [x] Completed
- **Timestamp:** 2026-10-24 09:26 (UTC+7)
- **Commit:** `feat(step-5): implement decision matrix, ROI optimization engine, and populate retention actions`
- **Objective:** Translate customer value tiers, churn probabilities and local SHAP drivers into a 9-cell prescriptive retention strategy, evaluate customer and portfolio ROI, and populate `fact_retention_actions` in `database/ism_dss.db`.

### Files Created or Modified
| Path | Description |
|------|-------------|
| `src/decision_engine.py` | Parameterized decision simulation (`run_decision_simulation`), SHAP-personalized action builder (`build_recommended_action`), portfolio aggregation (`summarize_decisions`) and SQLite persistence |
| `models/decision_summary.json` | Portfolio KPIs, 6-quadrant breakdown and 9-cell (`value_tier` × `risk_level`) financial summary |
| `database/ism_dss.db` | Populated `fact_retention_actions` with 7,043 customer decision records |
| `README.md` | Replaced planned Section 6 with implemented 9-cell retention playbook, economic equations and baseline ROI tables |
| `PROGRESS.md` | Added Step 5 log |

### Baseline Simulation Parameters (`DEFAULT_PARAMS`)
- `annual_horizon_months = 12`, `discount_months = 3`
- `call_cost = $5.00` (1-on-1 Account Manager / VIP care call), `email_sms_cost = $0.50` (automated outreach)
- `high_value_discount_pct = 0.20` (20% 3-month loyalty discount), `medium_value_discount_pct = 0.10` (10% 3-month voucher)
- `success_rate_vip_call = 0.45` (45% for High-Value + High-Risk; 36% for High-Value + Medium-Risk)
- `success_rate_auto_campaign = 0.25` (25% for Medium-Value + High-Risk; 17.5% for Medium-Value + Medium-Risk)

### Portfolio Financial KPIs (7,043 Customers)
| Portfolio Metric | Value |
|------------------|-------|
| **Total Annualized Revenue** | **\$5,473,399.20** |
| **Total Annual Revenue at Risk** | **\$2,445,852.11** (44.69% of annual revenue) |
| **Targeted Customers ($\text{Intervention Cost} > \$0$)** | **3,100** (44.02% of customer base) |
| **Total Intervention Budget** | **\$29,292.27** |
| **Expected Saved Revenue** | **\$567,861.45** |
| **Expected Net Benefit** | **\$538,569.18** |
| **Portfolio Expected ROI** | **1,838.61%** |

### Breakdown by Strategic Decision Quadrant
| Strategic Decision Quadrant | Customers | Mean Churn Prob | Annualized Revenue (\$) | Revenue at Risk (\$) | Intervention Cost (\$) | Expected Saved Revenue (\$) | Expected Net Benefit (\$) | Quadrant ROI (%) |
|---|---|---|---|---|---|---|---|---|
| `Critical VIP Retention` | 266 | 0.69 | 320,080.80 | 222,565.44 | 8,531.83 | 100,154.52 | 91,622.69 | **1,073.89%** |
| `Proactive VIP Care` | 550 | 0.45 | 636,784.80 | 284,663.12 | 8,481.04 | 102,478.81 | 93,997.77 | **1,108.33%** |
| `Automated Retention Campaign` | 1,658 | 0.80 | 1,601,436.60 | 1,293,815.33 | 10,838.14 | 323,453.94 | 312,615.80 | **2,884.40%** |
| `Targeted Nudge` | 626 | 0.46 | 515,786.40 | 238,710.17 | 1,441.26 | 41,774.18 | 40,332.92 | **2,798.45%** |
| `No Financial Intervention` | 705 | 0.58 | 269,094.00 | 160,128.58 | 0.00 | 0.00 | 0.00 | 0.00% |
| `Champions / Nurture (No Intervention)` | 3,238 | 0.10 | 2,130,216.60 | 245,969.47 | 0.00 | 0.00 | 0.00 | 0.00% |

### 9-Cell (`value_tier` × `risk_level`) Matrix Verification
| Value Tier | Risk Level | Decision Quadrant | Customers | Mean Monthly (\$) | Revenue at Risk (\$) | Cost (\$) | Saved Revenue (\$) | Net Benefit (\$) | Cell ROI (%) |
|---|---|---|---|---|---|---|---|---|---|
| **High** | **High** | `Critical VIP Retention` | 266 | 100.28 | 222,565.44 | 8,531.83 | 100,154.52 | 91,622.69 | 1,073.89% |
| **High** | **Medium** | `Proactive VIP Care` | 550 | 96.48 | 284,663.12 | 8,481.04 | 102,478.81 | 93,997.77 | 1,108.33% |
| **High** | **Low** | `Champions / Nurture (No Intervention)` | 1,395 | 84.44 | 156,456.03 | 0.00 | 0.00 | 0.00 | 0.00% |
| **Medium** | **High** | `Automated Retention Campaign` | 1,658 | 80.49 | 1,293,815.33 | 10,838.14 | 323,453.94 | 312,615.80 | 2,884.40% |
| **Medium** | **Medium** | `Targeted Nudge` | 626 | 68.66 | 238,710.17 | 1,441.26 | 41,774.18 | 40,332.92 | 2,798.45% |
| **Medium** | **Low** | `Champions / Nurture (No Intervention)` | 424 | 62.64 | 56,359.74 | 0.00 | 0.00 | 0.00 | 0.00% |
| **Low** | **High** | `No Financial Intervention` | 338 | 34.35 | 103,033.95 | 0.00 | 0.00 | 0.00 | 0.00% |
| **Low** | **Medium** | `No Financial Intervention` | 367 | 29.46 | 57,094.63 | 0.00 | 0.00 | 0.00 | 0.00% |
| **Low** | **Low** | `Champions / Nurture (No Intervention)` | 1,419 | 23.37 | 33,153.70 | 0.00 | 0.00 | 0.00 | 0.00% |

### Database & View Verification
- `fact_retention_actions`: **7,043 rows** inserted atomically (`DELETE` + `INSERT` within a transaction with `PRAGMA foreign_keys = ON`).
- `vw_customer_360` completeness check: **7,043 rows × 41 columns**, **0 NULLs** across all core, mining and retention decision attributes.

---
