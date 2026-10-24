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
| 6 | Streamlit DSS Dashboard — `app.py` (4-Tier ISM Presentation Layer) & Academic Report Outline (`ISM_REPORT_OUTLINE.md`) | [x] Completed |
| 7 | Multi-Model Benchmarking, Profit Curve & Constrained Optimizer — `src/optimization_engine.py` | [x] Completed |
| 8 | Closed-Loop CRM Tracking, Executive Report Exporter & Dashboard v2.0 Upgrade — `app.py` | [x] Completed |

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

## ✅ Step 6 — Streamlit Decision Support Dashboard (`app.py`) & Final Academic Documentation (`ISM_REPORT_OUTLINE.md`)

- **Status:** [x] Completed (100% — 6/6 Steps Finished)
- **Timestamp:** 2026-10-24 09:34 (UTC+7)
- **Commit:** `feat(step-6): build 4-tab Streamlit DSS dashboard and finalize ISM academic report documentation`
- **Objective:** Build the 4-Tab Streamlit Decision Support System dashboard (`app.py`) connected to `database/ism_dss.db` (`vw_customer_360`) and `models/`, author the comprehensive academic report reference (`ISM_REPORT_OUTLINE.md`), and finalize all project documentation.

### Files Created or Modified
| Path | Description |
|------|-------------|
| `app.py` | 4-Tab interactive Streamlit Decision Support Dashboard (Plotly visualizations, real-time What-If ROI & Budget Cap simulator, operational campaign queue with CSV download & Customer 360° SHAP inspector, live new-customer predictor & advisor) |
| `ISM_REPORT_OUTLINE.md` | Full academic report outline and empirical reference for the ISM course project (Executive Summary, Business Problem, 4-Tier Architecture, Hybrid 3NF/Star Warehouse Schema, K-Means/XGBoost/SHAP Methodology & Results, 9-Cell Decision Matrix & Financial ROI Analysis, Managerial Implications & Limitations) |
| `README.md` | Added Section 7 (Interactive Streamlit DSS Dashboard overview), updated project structure, added `streamlit run app.py` launch instructions, and marked all 6 steps completed |
| `PROGRESS.md` | Added Step 6 completion log and marked project 100% completed |

### 4-Tab Dashboard Architecture (`app.py`)
1. **Tab 1 — Executive KPI & Portfolio Overview:**
   - Displays 7 top-level KPI metric cards (`7,043` customers, `26.54%` churn rate, `\$5.47M` annualized revenue, `\$2.45M` revenue at risk, `\$567.86K` expected saved revenue, `\$538.57K` net benefit, `1,838.61%` portfolio ROI).
   - Interactive 9-cell Value Tier × Risk Level heatmap (toggleable between customer count/share and revenue at risk).
   - Grouped bar chart comparing Annual Revenue at Risk, Expected Saved Revenue, and Expected Net Benefit across the 6 strategic decision quadrants.
   - K-Means cluster summary table (`High`, `Medium`, `Low` tiers) + Silhouette score diagnostic plot ($k = 2 \dots 6$).
   - Top 12 Global SHAP Churn Drivers horizontal bar chart + XGBoost hold-out evaluation table & confusion matrix heatmap.
2. **Tab 2 — Interactive What-If ROI & Budget Simulator:**
   - Real-time sliders for High/Medium risk probability cutoffs, 1-on-1 Account Manager call cost (`call_cost`), automated email/SMS cost (`email_sms_cost`), 3-month promotional discount rates (`high_value_discount_pct`, `medium_value_discount_pct`), expected retention success rates (`success_rate_vip_call`, `success_rate_auto_campaign`), and an optional **Maximum Campaign Budget Cap** (`apply_budget_cap`, prioritizing highest-ROI accounts first).
   - Invokes `run_decision_simulation` and `summarize_decisions` dynamically, displaying **Simulated vs. Baseline** delta KPIs, grouped comparison bar charts, and updated quadrant breakdown tables.
3. **Tab 3 — Operational Action Queue & Customer 360 Inspector:**
   - Multi-select filters by `decision_quadrant` (defaulting to `Critical VIP Retention` — `266` accounts and `Automated Retention Campaign` — `1,658` accounts), `value_tier`, `risk_level`, and minimum Revenue at Risk.
   - Displays prioritized customer table sorted by `revenue_at_risk` descending and provides a one-click **Download Actionable Campaign CSV** button (`st.download_button`).
   - **Customer 360° Drill-Down Inspector:** Displays full demographic, service, billing, mining, and prescriptive decision details for any selected `customerID` alongside an individualized Top-10 local SHAP contribution bar chart.
4. **Tab 4 — Real-Time New Customer Predictor & Decision Advisor:**
   - Interactive form accepting 19 raw subscriber attributes; automatically derives the engineered features (`tenure_group`, `total_services_count`, `avg_monthly_spend`, `is_auto_payment`), scores the record through `kmeans_value_pipeline.joblib`, `xgb_churn_pipeline.joblib`, and `shap.TreeExplainer`, and evaluates the prescriptive retention action and expected ROI via `run_decision_simulation`.

### Verification Results
- `py_compile` syntax verification on `app.py`: **Passed**.
- Headless execution via `streamlit.testing.v1.AppTest.from_file('app.py')`:
  - Initial full-app render across all 4 tabs: **0 exceptions**, `16` KPI metric cards rendered, `4` data tables rendered.
  - Interactive form submission in Tab 4 (`Predict & Generate Retention Strategy`): **0 exceptions**, live K-Means + XGBoost + TreeSHAP + ROI inference executed cleanly (`22` total metric cards rendered).
- End-to-end project status: **6 / 6 Steps Completed (100%)**.


---

## ✅ Step 7 — Multi-Model Benchmarking, Profit Curve & Constrained Resource Optimizer (`src/optimization_engine.py`)

- **Status:** [x] Completed
- **Timestamp:** 2026-10-24 09:40 (UTC+7)
- **Commit:** `feat(step-7): add multi-model benchmarking, profit curve, and constrained knapsack budget optimizer`
- **Objective:** Upgrade the analytical tier to Enterprise DSS v2.0 by benchmarking three supervised classifiers (Logistic Regression, Random Forest, XGBoost), optimizing the probability cutoff via a financial profit curve (`0.15..0.85`), and implementing a dual-constrained Knapsack / Greedy ROI resource allocator with a Pareto Efficient Frontier.

### Files Created or Modified
| Path | Description |
|------|-------------|
| `src/optimization_engine.py` | Multi-model benchmark (`run_model_benchmark`), threshold profit curve sweep (`run_threshold_profit_curve`), and dual-constrained resource optimizer + Pareto frontier (`optimize_under_constraints`) |
| `models/model_benchmark.json` | Hold-out comparison metrics and ROC / PR curve coordinates for Logistic Regression, Random Forest, and XGBoost |
| `models/threshold_profit_curve.json` | 15-point threshold profit sweep (`0.15..0.85`), optimal financial cutoff (`0.45`), constrained allocation reference summary (`$10,000` budget, `150` VIP calls), and 61-point Pareto Efficient Frontier |
| `README.md` | Added Section 7 (Enterprise DSS v2.0 Extensions) and updated project structure / roadmap |
| `ISM_REPORT_OUTLINE.md` | Added Section 6.3 covering multi-model benchmarking, profit curve optimization, and Pareto resource allocation |
| `PROGRESS.md` | Added Step 7 log |

### 1. Multi-Model Hold-Out Benchmark ($n_{\text{train}} = 5{,}634$, $n_{\text{test}} = 1{,}409$)
| Model | ROC-AUC | PR-AUC | Recall | Precision | F1-Score | Accuracy | Brier Score | Train ROC-AUC | Training Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.8440 | **0.6671** | **0.7914** | 0.5103 | 0.6205 | 0.7431 | 0.1656 | 0.8501 | **0.066** |
| **Random Forest** | **0.8456** | 0.6664 | 0.7647 | **0.5448** | **0.6363** | **0.7679** | **0.1564** | 0.8950 | 0.787 |
| **XGBoost (Champion)** | 0.8423 | 0.6651 | 0.7701 | 0.5284 | 0.6268 | 0.7566 | 0.1612 | 0.9038 | 0.789 |

### 2. Threshold Profit Curve Optimization (`0.15` to `0.85`, Step `0.05`)
- **Optimal Financial Threshold ($\tau_H^* = 0.45$, $\tau_M^* = 0.29$):**
  - High-Risk Accounts: `3,027` | Targeted Accounts: `3,137` | VIP Calls Required: `835`
  - Direct Campaign Cost: `\$34,998.56` | Effective Total Cost (incl. false-positive cannibalization & overload): `\$77,992.28`
  - Expected Saved Revenue: **`\$603,502.53`** | Maximum Expected Net Benefit: **`\$525,510.25`** | Effective ROI: **`673.80%`**
- **Baseline Cutoff ($\tau_H = 0.60$, $\tau_M = 0.39$):**
  - High-Risk Accounts: `2,262` | Targeted Accounts: `2,740` | Effective Net Benefit: `\$499,933.88` (Direct Net Benefit: `\$538,569.18`, Direct ROI: `1,838.61%`).

### 3. Constrained Resource Allocation (`max_budget = $10,000`, `max_vip_calls = 150`)
- **Funded Accounts:** `1,756 / 3,100` (`1,344` accounts deferred due to budget/call constraints).
- **Budget Spent:** `\$9,999.99` (**34.14%** of unconstrained `\$29,292.27` budget; `99.9999%` budget utilization).
- **Expected Saved Revenue:** `\$311,924.27` | **Expected Net Benefit:** **`\$301,924.28`** (**56.06%** of unconstrained net benefit captured!).
- **Constrained Portfolio ROI:** **`3,019.25%`**.


---

## ✅ Step 8 — Closed-Loop CRM Tracking, Executive Report Exporter & Dashboard v2.0 Upgrade (`app.py`)

- **Status:** [x] Completed (100% — All 8/8 Enterprise v2.0 Steps Finished)
- **Timestamp:** 2026-10-24 13:29 (UTC+7)
- **Commit:** `feat(step-8): complete enterprise v2.0 with closed-loop CRM, constrained optimizer, and model governance tab`
- **Objective:** Complete the Enterprise DSS v2.0 upgrade by extending `fact_retention_actions` and `vw_customer_360` with Closed-Loop CRM tracking attributes (`action_status`, `manager_notes`, `updated_at`), upgrading `app.py` on the Modern Enterprise Slate Dark Theme to 5 comprehensive tabs (integrating the Pareto Efficient Frontier, Threshold Profit Curve, live CRM status persistence, Executive Briefing `.zip`/`.md`/`.html`/`.csv` exporter, and Tab 5 Model Benchmarking & System Governance), and finalizing all documentation.

### Files Created or Modified
| Path | Description |
|------|-------------|
| `src/etl_pipeline.py` | Updated DDL of `fact_retention_actions` and `vw_customer_360` to include `action_status` (`CHECK IN ('Pending','Contacted','Retained','Churned')`), `manager_notes`, and `updated_at` |
| `src/decision_engine.py` | Added idempotent schema migration helper `ensure_crm_columns`, CRM-preserving `persist_retention_actions`, and direct SQLite updater `update_customer_crm_status` |
| `database/ism_dss.db` | Migrated `fact_retention_actions` and `vw_customer_360` (now `7,043 rows × 44 columns`, `0` unexpected NULLs) |
| `.streamlit/config.toml` | Configured Modern Enterprise Slate Dark Theme (`#0B0F19` background, `#111827` secondary cards, `#38BDF8` accent, `#F9FAFB` crisp text) |
| `app.py` | Upgraded to 5-Tab Enterprise DSS v2.0 Dashboard with high-contrast KPI cards, Pareto Efficient Frontier, Threshold Profit Curve, Closed-Loop CRM write-back, Executive Briefing Bundle (`.zip`/`.md`/`.csv`) exporter, and Tab 5 Model Benchmarking & System Governance |
| `README.md` | Updated warehouse schema, 5-Tab dashboard specification, and 8-step completion table |
| `ISM_REPORT_OUTLINE.md` | Updated Section 7 to document the 5-Tab Enterprise v2.0 dashboard and Closed-Loop CRM governance |
| `PROGRESS.md` | Added Step 8 completion log |

### Verification Results
- Warehouse verification (`src/decision_engine.py`): `vw_customer_360` verified at **7,043 rows × 44 columns** with **0 NULLs** across all core, mining, decision, and CRM status columns.
- Headless Streamlit verification (`streamlit.testing.v1.AppTest.from_file('app.py')`):
  - Initial render across all **5 Tabs**: **0 exceptions**, `5` tabs and `7` data tables rendered cleanly.
  - **Tab 3 Closed-Loop CRM Form Submit (`💾 Save CRM Status to DB`):** **0 exceptions**, verified direct SQLite `UPDATE` on `fact_retention_actions` and automatic cache refresh.
  - **Tab 4 Real-Time Predictor Form Submit (`🔮 Predict & Generate Retention Strategy`):** **0 exceptions**, verified end-to-end live K-Means + XGBoost + TreeSHAP + ROI inference.


---

## ✅ Step 9 — Enterprise DSS v3.0: Causal Uplift Modeling & GenAI Retention Copilot (`src/uplift_engine.py`, `src/copilot_engine.py`, `app.py`)

- **Status:** [x] Completed (100% — All 9/9 Enterprise v3.0 Steps Finished)
- **Timestamp:** 2026-10-24 13:50 (UTC+7)
- **Commit:** `feat(step-9): implement causal uplift modeling and GenAI retention copilot in dashboard v3.0`
- **Objective:** Upgrade the system to **Enterprise DSS v3.0** by introducing (1) **Causal Uplift Modeling & "Sleeping Dog" Suppression** (`src/uplift_engine.py`) to classify all 7,043 subscribers into *Persuadables*, *Sleeping Dogs*, *Lost Causes*, and *Sure Things*, flagging `is_do_not_disturb = 1` on negative-uplift accounts, and (2) **GenAI 1-to-1 Retention Copilot** (`src/copilot_engine.py`) to synthesize personalized Account Manager Telesales Call Scripts and Customer Retention Emails grounded in each subscriber's local TreeSHAP drivers and service gaps.

### Files Created or Modified
| Path | Description |
|------|-------------|
| `src/uplift_engine.py` | Causal Treatment Effect ($\tau_i = P(\text{Stay}\mid T=1) - P(\text{Stay}\mid T=0)$) evaluator (`compute_causal_uplift`), causal budget exclusion engine (`apply_uplift_exclusion`), SQLite persistence (`persist_causal_uplift`), and `models/uplift_summary.json` exporter |
| `src/copilot_engine.py` | Zero-latency Contextual GenAI Retention Copilot (`generate_retention_pitch`) producing 3-stage Account Manager Telesales Scripts (`call_script`) and Personalized Customer Emails (`executive_email`) tailored to `top_shap_driver_1`, `top_shap_driver_2`, and missing VAS add-ons |
| `src/etl_pipeline.py` & `src/decision_engine.py` | Extended `fact_retention_actions` and `vw_customer_360` schema (`47` columns total) with `uplift_score REAL`, `causal_segment TEXT CHECK IN ('Persuadable','Sleeping Dog','Lost Cause','Sure Thing')`, and `is_do_not_disturb INTEGER CHECK IN (0,1)` |
| `models/uplift_summary.json` | Portfolio-wide causal segmentation breakdown and financial comparison between Correlational Targeting and Causal Uplift Exclusion |
| `app.py` | Upgraded to **Enterprise DSS v3.0** (Tab 1 Sleeping Dogs Suppressed KPI & Guardrail banner; Tab 2 `"Enforce Causal Uplift Exclusion"` toggle; Tab 3 Causal Segment filter, `⛔ DO NOT DISTURB` alert, and `"🤖 Generate AI Retention Script & Email"` dual-tab Copilot; Tab 4 live Causal Segment + auto-generated GenAI pitch) |
| `README.md`, `ISM_REPORT_OUTLINE.md`, `PROGRESS.md` | Updated in 100% Academic English with Causal Uplift and GenAI Copilot methodology and empirical results |

### Empirical Causal Uplift Results (`models/uplift_summary.json`)
| Causal Segment | Subscribers | Portfolio Share | Mean Uplift ($\bar{\tau}$) | Mean Churn Prob | Annualized Revenue (\$) | Revenue at Risk (\$) | Managerial Protocol |
|---|---|---|---|---|---|---|---|
| **`Persuadable`** | **3,066** | 43.53% | `+0.3125` | 0.5681 | 2,718,435.00 | 1,565,305.91 | Priority target for VIP Concierge & Automated Retention bundles |
| **`Sleeping Dog`** | **892** | 12.66% | `-0.1755` | 0.3694 | 525,313.80 | **195,483.60** | **`is_do_not_disturb = 1`** (`\$0` outbound spend; suppress calls to prevent backfire churn) |
| **`Lost Cause`** | **453** | 6.43% | `+0.0247` | 0.8820 | 428,925.00 | 378,722.42 | Exclude from paid budget (`\$0` spend; passive self-service offboarding) |
| **`Sure Thing`** | **2,632** | 37.37% | `+0.0569` | 0.0897 | 1,800,725.40 | 306,340.18 | Naturally loyal (`\$0` retention spend; standard service excellence) |
| **Total Portfolio** | **7,043** | **100.00%** | — | **0.3809** | **5,473,399.20** | **2,445,852.11** | **Causal Net Benefit: `\$600,280.55` (`+11.46%` Lift) · Causal ROI: `2,326.40%` (`+487.8 pts`)** |

### Verification Results
- `src/uplift_engine.py`: Migrated and updated all **7,043 rows** in `fact_retention_actions` (`vw_customer_360` verified at **7,043 rows × 47 columns**, `0` NULLs).
- `src/copilot_engine.py`: Smoke-tested both `Persuadable` VIP accounts and `Sleeping Dog` (`is_do_not_disturb = 1`) accounts with 0 errors.
- Headless Streamlit `AppTest` on `app.py` v3.0: **0 exceptions** across initial load, Tab 3 Copilot script/email generation, and Tab 4 real-time predictor + Copilot generation (`9` total tabs rendered cleanly).

