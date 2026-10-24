# 📘 ISM PROJECT REPORT OUTLINE & EMPIRICAL REFERENCE
## Building an Intelligent Customer Retention & Value Optimization System using Data Mining Techniques
**(K-Means Clustering + XGBoost Classification + SHAP Explainability + Prescriptive Decision Support Dashboard)**

> **Course:** Information Systems Management (ISM)  
> **Institution:** School of Computer Science and Engineering — International University, VNU-HCMC (HCMIU)  
> **System Repository:** `ISM-project` (4-Tier Decision Support System)

---

## 1. Executive Summary

Customer acquisition in the telecommunications industry costs five to seven times more than retaining an existing subscriber. However, blanket retention campaigns—where every at-risk subscriber receives an identical financial discount—erode profit margins by subsidizing low-value accounts while under-investing in high-value subscribers.

This project designs, implements, and validates an end-to-end **4-Tier Decision Support System (DSS)** on the benchmark **IBM Telco Customer Churn** dataset (**7,043 subscribers**, **26.54% observed churn rate**, **\$5,473,399.20** in total annualized revenue). The system integrates four analytical pillars:

1. **Data Warehouse & ETL Layer (`src/etl_pipeline.py`):** Cleanses raw operational records, resolves the 11 zero-tenure blank `TotalCharges` anomalies, engineers five behavioral indicators, and loads a normalized SQLite decision warehouse (`database/ism_dss.db`) featuring three 3NF dimension tables (`customers`, `services`, `billing`), two fact tables (`fact_mining_scores`, `fact_retention_actions`), and an integrated `vw_customer_360` analytical view.
2. **Unsupervised Customer Value Segmentation (`src/mining_engine.py` — Module 1):** Applies `StandardScaler` and **K-Means ($k=3$, Silhouette = `0.402`)** across `tenure`, `monthly_charges`, `total_charges`, and `total_services_count` to stratify the subscriber base into **High Value** (`2,211` accounts, 43.31% of monthly revenue), **Medium Value** (`2,708` accounts, 44.51% of revenue, 45.61% churn rate), and **Low Value** (`2,124` accounts, 12.19% of revenue).
3. **Supervised Churn Prediction & Explainable AI (`src/mining_engine.py` — Modules 2 & 3):** Trains a leakage-free `ColumnTransformer + XGBClassifier` pipeline (`scale_pos_weight = 2.7686`) on a stratified 80/20 split, achieving a hold-out **ROC-AUC of `0.8423`**, **PR-AUC of `0.6651`**, **Recall of `77.01%`**, and **F1-Score of `0.6268`**. Exact **TreeSHAP** attributions (`7,043 × 48` encoded features aggregated back to 23 business attributes) identify `contract` (`0.7832`), `tenure` (`0.4331`), `online_security` (`0.2600`), `internet_service` (`0.2146`), and `monthly_charges` (`0.2069`) as the top global churn drivers, while extracting the top two risk-increasing drivers per customer.
4. **Prescriptive Retention Decision Matrix & ROI Engine (`src/decision_engine.py` & `app.py`):** Positions all 7,043 subscribers across a **9-cell Value Tier × Churn Risk matrix** and personalizes retention offers using each subscriber's top SHAP drivers. Out of **\$2,445,852.11** in annual revenue at risk, the baseline policy selectively targets **3,100 High/Medium-Value at-risk accounts** with a total intervention budget of **\$29,292.27**, generating **\$567,861.45** in expected saved revenue, **\$538,569.18** in expected net benefit, and a **Portfolio ROI of `1,838.61%`**.

---

## 2. Business Problem & Managerial Motivation

### 2.1 Operational Challenge
Telecommunication operators operate in saturated markets characterized by low switching costs for month-to-month subscribers and intense price competition on fiber-optic bundles. In our dataset of **7,043 subscribers**:
- **1,869 subscribers (`26.54%`)** have churned.
- Early-lifecycle subscribers (`tenure_group = 0-12m`, $n = 2{,}186$) exhibit an acute churn rate of **47.4%**, compared to **9.5%** among long-tenured subscribers (`48m+`, $n = 2{,}239$).
- Total annualized recurring revenue is **\$5,473,399.20**, of which **\$2,445,852.11 (`44.69%`)** is exposed to expected annual churn risk.

### 2.2 Limitations of Traditional Retention Approaches
1. **Reactive rather than proactive:** Traditional CRM workflows trigger retention offers only when a subscriber calls to cancel service.
2. **Value-blind targeting:** Treating a \$26/month single-line subscriber identically to a \$100/month multi-service fiber subscriber leads to negative ROI on low-tier interventions.
3. **Black-box risk scores:** Providing a retention agent with a churn probability ($p = 0.83$) without explaining *why* the customer is leaving prevents tailored offer selection.

### 2.3 Research Questions & System Objectives
| Research Question | Analytical Method | Managerial Deliverable |
|---|---|---|
| **RQ1:** How should subscribers be stratified by economic importance? | Standardized K-Means Clustering ($k=3$) | Customer Value Tiers (`High`, `Medium`, `Low`) |
| **RQ2:** Which subscribers are at highest risk of churning in the next billing cycle? | Cost-sensitive XGBoost Classification | Calibrated churn probability $p_i \in [0, 1]$ and Risk Level |
| **RQ3:** Which specific contract or service attributes drive each individual's churn risk? | Exact TreeSHAP local & global attributions | Per-customer `top_shap_driver_1` & `top_shap_driver_2` |
| **RQ4:** How should a constrained retention budget be allocated to maximize net financial benefit? | 9-Cell Prescriptive Decision Matrix & What-If Simulator | Personalized action queue, expected net benefit, and ROI |

---

## 3. Four-Tier Information Systems Architecture

Following Information Systems Management (ISM) design principles, the system decouples data storage, data processing, analytical modeling, and executive presentation into four distinct tiers:

```mermaid
flowchart TD
    subgraph T1["Tier 1 — Data Acquisition & Warehouse Layer"]
        A["src/data_generator.py\nMulti-mirror download + Synthetic Fallback"]
        B["data/raw/telco_churn_raw.csv\n(7,043 rows × 21 columns)"]
        C[("database/ism_dss.db\nSQLite Warehouse: 3 Dimensions + 2 Facts + Views")]
    end
    subgraph T2["Tier 2 — ETL & Feature Engineering Layer"]
        D["src/etl_pipeline.py\nType Cleansing, Imputation, 5 Engineered Features"]
        E["data/processed/telco_churn_processed.csv\n(7,043 rows × 26 columns)"]
    end
    subgraph T3["Tier 3 — Data Mining & Prescriptive Decision Layer"]
        F["src/mining_engine.py\nK-Means (k=3) + XGBoost + TreeSHAP"]
        G["src/decision_engine.py\n9-Cell Matrix + Parameterized ROI Engine"]
        H["models/*.joblib & *.json\nSerialized Pipelines & Audit Metrics"]
    end
    subgraph T4["Tier 4 — Executive Presentation Layer"]
        I["app.py — 4-Tab Streamlit DSS Dashboard\nKPIs · What-If Simulator · Action Queue · Live Predictor"]
    end
    A --> B --> D
    D --> E
    D --> C
    C --> F --> G --> I
    F --> H --> I
```

---

## 4. Data Warehouse & Hybrid 3NF / Star Schema Design (`database/ism_dss.db`)

### 4.1 Data Cleansing & Feature Engineering (`src/etl_pipeline.py`)
- **Missing-Value Resolution:** Exactly **11 records** contain whitespace in `TotalCharges`. Inspection confirms all 11 records have `tenure = 0` (newly onboarded subscribers who have not yet completed their first billing cycle). Imputing `total_charges = 0.0` preserves economic consistency without discarding records.
- **Engineered Features:**
  1. `tenure_group`: Categorizes subscriber lifecycle into `0-12m` (47.4% churn), `12-24m` (28.7% churn), `24-48m` (20.4% churn), and `48m+` (9.5% churn).
  2. `total_services_count`: Counts active subscriptions across the six value-added services (`online_security`, `online_backup`, `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies`; mean = `2.04`, range `0–6`), serving as a proxy for ecosystem lock-in.
  3. `avg_monthly_spend`: Ratio `total_charges / tenure` (or `monthly_charges` when `tenure = 0`; mean = `\$64.76`).
  4. `is_auto_payment`: Binary indicator (`1` for automatic bank transfer or credit card; `43.5%` of subscribers).
  5. `churn_binary`: Numeric target (`1` = Churned, `0` = Retained).

### 4.2 Relational Warehouse Schema
The SQLite warehouse combines **Third Normal Form (3NF) dimension tables** (eliminating update anomalies across demographics, service subscriptions, and billing) with **Analytical Fact Tables** for mining scores and retention actions:

- **Dimension Tables (7,043 rows each):**
  - `customers`: `customerID` (PK), `gender`, `senior_citizen`, `partner`, `dependents`, `tenure`, `tenure_group`, `contract`, `churn`, `churn_binary`.
  - `services`: `customerID` (PK, FK), `phone_service`, `multiple_lines`, `internet_service`, 6 VAS attributes, `total_services_count`.
  - `billing`: `customerID` (PK, FK), `paperless_billing`, `payment_method`, `is_auto_payment`, `monthly_charges`, `total_charges`, `avg_monthly_spend`.
- **Fact Tables (7,043 rows each):**
  - `fact_mining_scores`: `customerID` (PK, FK), `value_tier` (`CHECK IN ('High','Medium','Low')`), `cluster_id`, `churn_probability` (`CHECK BETWEEN 0 AND 1`), `risk_level` (`CHECK IN ('High','Medium','Low')`), `top_shap_driver_1`, `top_shap_driver_2`, `scored_at`.
  - `fact_retention_actions`: `customerID` (PK, FK), `annualized_revenue`, `revenue_at_risk`, `decision_quadrant`, `recommended_action`, `intervention_cost`, `expected_saved_revenue`, `expected_net_benefit`, `expected_roi`, `decided_at`.
- **Analytical Views & Lineage:**
  - `customer_master`: Joins `customers`, `services`, and `billing` (`7,043 × 26`).
  - `vw_customer_360`: Master Customer 360° view (`7,043 × 41`) joining `customer_master` with both fact tables.
  - `etl_audit_log`: Tracks run timestamps, extracted/loaded row counts, and imputation counts across executions.

---

## 5. Data Mining Methodology & Empirical Results (`src/mining_engine.py`)

### 5.1 Module 1 — Customer Value Segmentation (K-Means Clustering)
To prevent high-magnitude monetary variables (`total_charges` up to \$8,684.80) from dominating Euclidean distance calculations, all four value indicators (`tenure`, `monthly_charges`, `total_charges`, `total_services_count`) are standardized via `StandardScaler` inside a scikit-learn `Pipeline`.

- **Cluster Validity Analysis (Silhouette Coefficient):**
  - $k=2$: `0.4586` · **$k=3$: `0.4025`** · $k=4$: `0.3955` · $k=5$: `0.3757` · $k=6$: `0.3650`.
  - $k=3$ provides strong cluster separation while mapping directly onto the three-tier executive resource allocation framework (`High`, `Medium`, `Low`).

| Value Tier | Cluster ID | Customers ($n$, %) | Mean Tenure (Months) | Mean Monthly Charges (\$) | Mean Total Charges (\$) | Mean VAS Count | Observed Churn Rate | Monthly Revenue Share |
|---|---|---|---|---|---|---|---|---|
| **High Value** | 0 | 2,211 (31.39%) | 58.39 | \$89.34 | \$5,207.53 | 3.98 | 14.74% | 43.31% |
| **Medium Value** | 2 | 2,708 (38.45%) | 15.32 | \$74.96 | \$1,143.36 | 1.89 | **45.61%** | **44.51%** |
| **Low Value** | 1 | 2,124 (30.16%) | 27.02 | \$26.17 | \$680.84 | 0.20 | 14.50% | 12.19% |

**Key Managerial Insight:** Cluster 2 (**Medium Value**) represents the single largest vulnerability: these subscribers pay high monthly bills (\$74.96/month, generating **44.51%** of monthly revenue) but are still early in their tenure (15.3 months) and suffer an alarming **45.61% churn rate**.

### 5.2 Module 2 — Supervised Churn Prediction (XGBoost Classifier)
- **Feature Engineering & Leakage Prevention:** A `ColumnTransformer` applies `OneHotEncoder(handle_unknown="ignore")` to 12 categorical attributes and `StandardScaler` to 11 numerical attributes (producing 48 encoded features), encapsulated with `XGBClassifier` (`n_estimators=400`, `max_depth=4`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3`, `reg_lambda=1.0`).
- **Class Imbalance Handling:** Stratified 80/20 split (`5,634` train / `1,409` test, `random_state=42`) with cost-sensitive weighting `scale_pos_weight = 2.7686` ($N_{\text{neg}} / N_{\text{pos}}$). Setting `scale_pos_weight > 1` intentionally prioritizes **Recall (77.01%)** over Precision (52.84%) because the financial cost of missing a churning high-value subscriber (False Negative, ~\$1,072/year) is far greater than the cost of sending a retention intervention to a loyal subscriber (False Positive, \$0.50–\$33.00).

| Evaluation Metric | Hold-Out Test Set ($n = 1{,}409$) | Managerial Interpretation |
|---|---|---|
| **ROC-AUC** | **0.8423** (Train: 0.9038) | Strong discriminative ranking across all probability cutoffs |
| **PR-AUC** | **0.6651** | Robust precision-recall trade-off under 26.54% positive class prevalence |
| **Recall (Sensitivity)** | **0.7701** (`288 / 374` churners caught) | Captures more than three-quarters of actual churners before defection |
| **Precision** | **0.5284** (`288 / 545` flagged) | Over half of flagged accounts actually churn without intervention |
| **F1-Score** | **0.6268** | Harmonic balance between precision and sensitivity |
| **Accuracy / Brier Score** | **75.66%** / **0.1612** | Well-calibrated probabilistic outputs |

### 5.3 Module 3 — Explainable AI (SHAP TreeExplainer)
To bridge predictive accuracy and prescriptive action, exact **TreeSHAP** values (in log-odds units) are computed for all 7,043 subscribers. Because one-hot encoding splits multi-category attributes across multiple columns, SHAP contributions are additively aggregated back to the **23 original business attributes** (by the linearity property of Shapley values).

| Rank | Business Attribute | Mean \|SHAP\| (Log-Odds) | Managerial Explanation |
|---|---|---|---|
| 1 | `contract` | **0.7832** | Month-to-month contracts are the single strongest driver of subscriber defection |
| 2 | `tenure` | **0.4331** | Subscribers in their first 12 months face severe onboarding churn risk |
| 3 | `online_security` | **0.2600** | Subscribers lacking OnlineSecurity have lower switching barriers |
| 4 | `internet_service` | **0.2146** | Fiber-optic subscribers exhibit elevated churn due to price/QoS sensitivity |
| 5 | `monthly_charges` | **0.2069** | Higher recurring monthly charges amplify bill-shock sensitivity |
| 6 | `tech_support` | **0.1873** | Absence of TechSupport increases frustration during technical incidents |
| 7 | `total_charges` | **0.1805** | Cumulative spend reflects long-term relationship depth |
| 8 | `payment_method` | **0.1803** | Electronic check users churn at significantly higher rates than auto-pay users |

---

## 6. Prescriptive Retention Decision Matrix & Financial ROI Analysis (`src/decision_engine.py`)

### 6.1 Nine-Cell Value × Risk Stratification
Combining the K-Means `value_tier` (`High`, `Medium`, `Low`) with the XGBoost `risk_level` (`High`: $p \ge 0.60$; `Medium`: $0.30 \le p < 0.60$; `Low`: $p < 0.30$) yields nine operational cells grouped into **six strategic decision quadrants**:

1. **Critical VIP Retention (`High Value × High Risk`, $n = 266$):** Immediate 1-on-1 Account Manager outreach (`\$5.00` call cost) + 20% 3-month loyalty discount or free 6-month VAS bundle tailored to the customer's top SHAP driver (`expected_success_rate = 45%`).
2. **Proactive VIP Care (`High Value × Medium Risk`, $n = 550$):** Proactive VIP care call (`\$5.00`) + complimentary VAS upgrade & 10% 3-month loyalty credit (`expected_success_rate = 36%`).
3. **Automated Retention Campaign (`Medium Value × High Risk`, $n = 1{,}658$):** Automated email/SMS campaign (`\$0.50`) + 10% 3-month discount voucher or 3-month VAS trial (`expected_success_rate = 25%`).
4. **Targeted Nudge (`Medium Value × Medium Risk`, $n = 626$):** Automated contract-renewal nudge (`\$0.50`) + 5% 3-month promotional voucher (`expected_success_rate = 17.5%`).
5. **No Financial Intervention (`Low Value × High/Medium Risk`, $n = 705$):** Zero-cost automated satisfaction survey and self-service FAQ email (`\$0.00` financial spend to prevent negative ROI on low-margin accounts).
6. **Champions / Nurture (`All Low-Risk Tiers`, $n = 3{,}238$):** Standard service excellence and organic cross-sell/upsell (`\$0.00` retention spend).

### 6.2 Empirical Financial Performance Across All 9 Cells
| Value Tier | Risk Level | Strategic Decision Quadrant | Customers | Mean Monthly (\$) | Mean Churn Prob | Annualized Revenue (\$) | Revenue at Risk (\$) | Intervention Cost (\$) | Expected Saved Revenue (\$) | Expected Net Benefit (\$) | Cell ROI (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **High** | **High** | `Critical VIP Retention` | 266 | 100.28 | 0.69 | 320,080.80 | 222,565.44 | 8,531.83 | 100,154.52 | 91,622.69 | **1,073.89%** |
| **High** | **Medium** | `Proactive VIP Care` | 550 | 96.48 | 0.45 | 636,784.80 | 284,663.12 | 8,481.04 | 102,478.81 | 93,997.77 | **1,108.33%** |
| **High** | **Low** | `Champions / Nurture (No Intervention)` | 1,395 | 84.44 | 0.11 | 1,413,580.80 | 156,456.03 | 0.00 | 0.00 | 0.00 | 0.00% |
| **Medium** | **High** | `Automated Retention Campaign` | 1,658 | 80.49 | 0.80 | 1,601,436.60 | 1,293,815.33 | 10,838.14 | 323,453.94 | 312,615.80 | **2,884.40%** |
| **Medium** | **Medium** | `Targeted Nudge` | 626 | 68.66 | 0.46 | 515,786.40 | 238,710.17 | 1,441.26 | 41,774.18 | 40,332.92 | **2,798.45%** |
| **Medium** | **Low** | `Champions / Nurture (No Intervention)` | 424 | 62.64 | 0.17 | 318,723.00 | 56,359.74 | 0.00 | 0.00 | 0.00 | 0.00% |
| **Low** | **High** | `No Financial Intervention` | 338 | 34.35 | 0.73 | 139,338.60 | 103,033.95 | 0.00 | 0.00 | 0.00 | 0.00% |
| **Low** | **Medium** | `No Financial Intervention` | 367 | 29.46 | 0.44 | 129,755.40 | 57,094.63 | 0.00 | 0.00 | 0.00 | 0.00% |
| **Low** | **Low** | `Champions / Nurture (No Intervention)` | 1,419 | 23.37 | 0.08 | 397,912.80 | 33,153.70 | 0.00 | 0.00 | 0.00 | 0.00% |
| **Total** | — | **Full Portfolio** | **7,043** | **64.76** | **0.38** | **5,473,399.20** | **2,445,852.11** | **29,292.27** | **567,861.45** | **538,569.18** | **1,838.61%** |

### 6.3 Enterprise DSS v2.0 Extensions: Multi-Model Benchmarking, Threshold Profit Curve & Constrained Knapsack Optimizer (`src/optimization_engine.py`)

#### A. Multi-Model Benchmarking ($n_{\text{train}} = 5{,}634$, $n_{\text{test}} = 1{,}409$)
To justify selecting **XGBoost** as the deployed production model, it is benchmarked against a linear classifier (**Logistic Regression**) and a bagging tree ensemble (**Random Forest**) over the exact same `ColumnTransformer` feature space:

| Model | ROC-AUC (Test) | PR-AUC (Test) | Recall | Precision | F1-Score | Accuracy | Brier Score | Training Time (s) | Architectural Assessment |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.8440 | **0.6671** | **0.7914** | 0.5103 | 0.6205 | 0.7431 | 0.1656 | **0.066** | Strong linear baseline; highest recall (`79.14%`) but highest false-positive rate |
| **Random Forest** | **0.8456** | 0.6664 | 0.7647 | **0.5448** | **0.6363** | **0.7679** | **0.1564** | 0.787 | Highest F1 and lowest Brier score, but slower exact Shapley decomposition |
| **XGBoost (Champion)** | 0.8423 | 0.6651 | 0.7701 | 0.5284 | 0.6268 | 0.7566 | 0.1612 | 0.789 | Balanced sensitivity (`77.01%`), native log-odds calibration, and fast exact `TreeSHAP` |

#### B. Threshold Profit Curve Optimization (`0.15` to `0.85`)
Sweeping `high_risk_threshold` ($\tau_H \in [0.15, 0.85]$, step `0.05`) under an operational cost formulation—incorporating False Positive promotional discount cannibalization among loyal subscribers and VIP call capacity surcharges—identifies a concave profit curve peaking at **`high_risk_threshold = 0.45`**:
- **Optimal Financial Cutoff ($\tau_H^* = 0.45$, $\tau_M^* = 0.29$):** Targets `3,137` subscribers (`835` VIP calls), maximizing **Expected Net Benefit at `\$525,510.25`** (`\$603,502.53` saved revenue minus `\$77,992.28` effective cost; `673.80%` effective ROI).
- **Conservative Baseline ($\tau_H = 0.60$, $\tau_M = 0.39$):** Targets `2,740` subscribers (`632` VIP calls), generating `\$499,933.88` effective net benefit (`\$538,569.18` direct net benefit) at a higher capital efficiency (`934.22%` effective ROI / `1,838.61%` direct ROI).

#### C. Constrained Resource Allocation (`optimize_under_constraints` & Pareto Efficient Frontier)
Under strict operational limits—**Maximum Budget Cap = `\$10,000.00`** and **Maximum VIP Agent Calls = `150`**—ranking eligible interventions by marginal efficiency (`expected_net_benefit / intervention_cost`) yields a dramatic Pareto concentration:
- **34.14% of the unconstrained budget (`\$9,999.99 / \$29,292.27`)** funds **1,756 high-efficiency accounts** and captures **56.06% of the total unconstrained net benefit (`\$301,924.28 / \$538,569.18`)**, achieving a **Constrained Portfolio ROI of `3,019.25%`**.

### 6.4 Enterprise DSS v3.0 Extensions: Causal Uplift Modeling & GenAI Retention Copilot (`src/uplift_engine.py`, `src/copilot_engine.py`)

#### A. Causal Uplift Matrix & "Sleeping Dog" Suppression (`src/uplift_engine.py`)
A fundamental limitation of correlational churn prediction is that high churn risk does not guarantee positive treatment responsiveness. To prevent wasted marketing spend and outreach-induced defection, **Enterprise DSS v3.0** models the **Individual Treatment Effect (Uplift Score)** $\tau_i = P(\text{Stay} \mid T=1, X_i) - P(\text{Stay} \mid T=0, X_i)$ and partitions all 7,043 subscribers into four causal quadrants persisted in `fact_retention_actions` (`uplift_score`, `causal_segment`, `is_do_not_disturb`):

| Causal Segment | Subscribers | Share (%) | Mean Uplift ($\bar{\tau}$) | Mean Churn Prob | Annualized Revenue (\$) | Revenue at Risk (\$) | Prescriptive Causal Policy |
|---|---|---|---|---|---|---|---|
| **`Persuadable`** | **3,066** | 43.53% | `+0.3125` | 0.5681 | 2,718,435.00 | 1,565,305.91 | Primary target for human VIP calls & automated loyalty bundles |
| **`Sleeping Dog`** | **892** | 12.66% | `-0.1755` | 0.3694 | 525,313.80 | **195,483.60** | **`is_do_not_disturb = 1`** (`\$0` spend; suppress outbound calls to prevent backfire churn) |
| **`Lost Cause`** | **453** | 6.43% | `+0.0247` | 0.8820 | 428,925.00 | 378,722.42 | Suppress paid incentives (`\$0` spend; passive self-service offboarding) |
| **`Sure Thing`** | **2,632** | 37.37% | `+0.0569` | 0.0897 | 1,800,725.40 | 306,340.18 | Naturally loyal (`\$0` retention spend; standard service excellence) |

Enforcing **Causal Uplift Exclusion** eliminates wasted spend on `Sleeping Dog` and `Lost Cause` accounts (`-\$3,489.27` budget reduction), protects **\$195,483.60 in Sleeping Dog Revenue at Risk**, increases **Expected Net Benefit from `\$538,569.18` to `\$600,280.55` (`+\$61,711.37` lift)**, and raises **Portfolio ROI from `1,838.61%` to `2,326.40%` (`+487.8` percentage points)**.

#### B. GenAI 1-to-1 Retention Copilot (`src/copilot_engine.py`)
To bridge the gap between analytical SHAP log-odds attributions and frontline contact-center execution, `generate_retention_pitch(customer_row)` dynamically synthesizes:
1. **3-Stage Account Manager Telesales Call Script (`call_script`):** Combines a personalized tenure/ARPU opening hook, root-cause empathy and objection handling mapped directly to `top_shap_driver_1` and `top_shap_driver_2`, and a tailored retention incentive close (or an automatic **⛔ DO NOT DISTURB Inbound-Only Care Protocol** for `Sleeping Dog` accounts).
2. **Personalized Customer Retention Email (`executive_email`):** Generates an executive-grade Markdown email with a dynamic `subject_line` and 1-click loyalty upgrade offer tailored to missing value-added services (`OnlineSecurity`, `TechSupport`, `OnlineBackup`, `DeviceProtection`).

---

## 7. Interactive Decision Support Dashboard v3.0 (`app.py`)

Styled with a **Modern Enterprise Slate Dark Theme** (`.streamlit/config.toml` and custom high-contrast FinTech CSS cards), the presentation tier (`app.py`) provides **five specialized workspaces** for executive, operational, and governance users:
1. **Tab 1 — Executive KPI & Portfolio Overview:** Displays 8 high-contrast dark slate KPI cards (including **Sleeping Dogs Suppressed: `892` accounts / `\$195,484` Revenue at Risk Protected**) and a Causal Guardrail status banner, an interactive 9-cell Value × Risk heatmap, a grouped bar chart comparing Revenue at Risk, Saved Revenue, and Net Benefit across decision quadrants, K-Means cluster profiles with Silhouette diagnostics, Top 12 Global SHAP drivers, and the XGBoost hold-out confusion matrix.
2. **Tab 2 — Interactive What-If, Causal Uplift & Knapsack Optimizer:** Features the **`"Enforce Causal Uplift Exclusion (Filter Out Sleeping Dogs)"`** toggle alongside sliders for High/Medium risk thresholds, 1-on-1 call costs, email/SMS costs, discount percentages, campaign conversion rates, and dual operational resource constraints (**Max Campaign Budget Cap ($)** and **Max VIP Account Manager Calls**). Renders both the **Pareto Efficient Frontier** and the **Threshold Profit Curve** (`0.15..0.85`, highlighting the optimal cutoff $\tau_H^* = 0.45$).
3. **Tab 3 — Operational Action Queue, Customer 360° Copilot & Closed-Loop CRM:** Filters the prioritized customer queue by decision quadrant, **causal uplift segment**, value tier, risk level, CRM action status, and minimum revenue at risk; flags `Sleeping Dog` records with a red **`⛔ DO NOT DISTURB`** alert; integrates the **`"🤖 Generate AI Retention Script & Email"`** dual-tab Copilot (`Telesales Call Script` + `Personalized Customer Email`); provides one-click **Executive Briefing Bundle export (`.zip`)**; and provides a live **Closed-Loop CRM Action Logger** persisted directly to `fact_retention_actions` in SQLite.
4. **Tab 4 — Real-Time New Customer Predictor & GenAI Pitch Advisor:** Allows an agent to input a new or hypothetical subscriber's profile and immediately computes their K-Means Value Tier, XGBoost Churn Probability, **Causal Uplift Segment & $\tau$ score**, Local SHAP attributions, Recommended Retention Action, Expected ROI, and **auto-generated 1-to-1 AI Telesales Script & Customer Email draft**.
5. **Tab 5 — Model Benchmarking & System Governance (ISM Audit View):** Displays the multi-model benchmark comparison table and overlaid ROC / Precision-Recall curves (**Logistic Regression** vs. **Random Forest** vs. **XGBoost Champion**), relational warehouse integrity checks (`0` foreign-key/null errors), the live `etl_audit_log` lineage table, and the Closed-Loop CRM status breakdown.

---

## 8. Managerial Implications, Limitations & Future Enhancements

### 8.1 Strategic Recommendations for Telecom Management
1. **Prioritize the "Medium-Value, High-Risk" Fiber Segment (`1,658` subscribers, \$1.29M Revenue at Risk):** These recently acquired, high-monthly-spend subscribers account for over half of total revenue at risk. Automated contract-migration vouchers and 3-month `OnlineSecurity`/`TechSupport` trials deliver the highest return (**2,884.40% ROI**).
2. **Protect High-Value VIPs with Human Concierge Outreach (`266` High-Risk + `550` Medium-Risk VIPs):** Deploying dedicated Account Managers to the 816 at-risk High-Value subscribers requires only **\$17,012.87** in budget while protecting **\$507,228.56** in annual revenue at risk and saving an expected **\$202,633.33**.
3. **Enforce Causal "Sleeping Dog" Suppression (`892` subscribers, \$195.48K Revenue at Risk Protected):** Suppressing outbound sales calls on month-to-month subscribers with negative treatment uplift ($\tau < 0$) prevents outreach-triggered churn and lifts portfolio ROI to **2,326.40%**.
4. **Institutionalize Ecosystem Lock-In during Onboarding:** Because SHAP identifies `contract=Month-to-month`, short `tenure`, and absence of `online_security`/`tech_support` as the top churn drivers, bundling complimentary 3-month security/support add-ons during the first 12 months will structurally reduce early-lifecycle churn.

### 8.2 Methodological Limitations & Future Work
- **Observational Calibration of Uplift:** While the Causal Uplift layer (`src/uplift_engine.py`) models heterogeneous treatment effects and Sleeping Dog backfire mechanics, production deployment should continuously recalibrate $\tau_i$ using randomized **multi-armed bandit / A/B hold-out control groups** tracked via the Closed-Loop CRM `action_status` table.
- **Dynamic Temporal Signals:** Incorporating monthly call-detail records (CDRs), network latency/outage tickets, and billing complaint logs would further improve PR-AUC and early warning lead time.
