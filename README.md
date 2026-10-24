# ISM-project

## 🧠 Intelligent Customer Retention & Value Optimization System
**Using Data Mining Techniques — K-Means + XGBoost + SHAP + Decision Support Dashboard**

> Course project for **Information Systems Management (ISM)** — International University, VNU-HCMC (HCMIU).

---

## 1. Business Problem

Telecommunication providers lose a substantial share of their subscribers every month (customer churn), while retention budgets are inherently limited. Allocating retention incentives uniformly is costly and inefficient: low-value customers may receive expensive offers, whereas high-value customers at risk may be overlooked.

This Decision Support System (DSS) transforms raw customer data into actionable managerial decisions by answering three questions:

| # | Managerial Question | Data Mining Technique | Output |
|---|---------------------|-----------------------|--------|
| 1 | *Which customers are most valuable?* | **K-Means Clustering** | Customer Value Tier (High / Medium / Low) |
| 2 | *Which customers are likely to leave?* | **XGBoost Classification** | Churn probability per customer |
| 3 | *Why are they leaving and what should we do?* | **SHAP Explainability** + **Decision Matrix** | Churn drivers, recommended action, Revenue at Risk, campaign ROI |

## 2. 4-Tier ISM Architecture

```mermaid
flowchart TD
    subgraph T1["Tier 1 - Data Layer"]
        A["data/raw: IBM Telco Customer Churn CSV (7,043 rows x 21 columns)"]
        B[("database/ism_dss.db - SQLite, 3 normalized tables")]
    end
    subgraph T2["Tier 2 - ETL and Data Processing Layer"]
        C["src/etl_pipeline.py: Cleaning, Feature Engineering, Loading"]
    end
    subgraph T3["Tier 3 - Analytics and Decision Logic Layer"]
        D["src/mining_engine.py: K-Means Value Tier, XGBoost Churn, SHAP"]
        E["src/decision_engine.py: Decision Matrix, Revenue at Risk, ROI"]
    end
    subgraph T4["Tier 4 - Presentation Layer"]
        F["app.py: Streamlit DSS Dashboard"]
    end
    G["src/data_generator.py: Data Acquisition"] --> A
    A --> C --> B
    B --> D --> E --> F
    D --> M["models/*.joblib"]
```

| Tier | Responsibility | Components |
|------|----------------|------------|
| 1. Data Layer | Raw data acquisition and persistent relational storage | `data/raw/`, `database/ism_dss.db`, `src/data_generator.py` |
| 2. ETL Layer | Data cleaning, transformation, feature engineering, loading | `src/etl_pipeline.py`, `data/processed/` |
| 3. Analytics & Decision Layer | Segmentation, prediction, explanation, prescriptive rules | `src/mining_engine.py`, `src/decision_engine.py`, `models/` |
| 4. Presentation Layer | Interactive decision support for managers | `app.py` (Streamlit) |

## 3. Dataset

- **Source:** [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (also distributed on Kaggle as `WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- **Size:** 7,043 customers × 21 attributes (demographics, subscribed services, account and billing information, churn label).
- **Target variable:** `Churn` (Yes = 1,869 customers, **26.54%**; No = 5,174 customers, 73.46%).
- **Known data-quality issue:** 11 records with a blank `TotalCharges` value, all corresponding to newly acquired customers with `tenure = 0`.
- **Resilience:** if all download mirrors are unreachable, `src/data_generator.py` falls back to a seeded synthetic generator that reproduces the original schema, row count, categorical domains, churn rate (~26%) and the blank-`TotalCharges` anomaly.

## 4. Database Schema (Planned — Step 3)

The ETL pipeline will load the cleaned data into an SQLite database in Third Normal Form (3NF), split into three tables linked by `customer_id`:

```mermaid
erDiagram
    CUSTOMERS ||--|| SERVICES : subscribes
    CUSTOMERS ||--|| BILLING : "is billed"
    CUSTOMERS {
        TEXT customer_id PK
        TEXT gender
        INTEGER senior_citizen
        TEXT partner
        TEXT dependents
        INTEGER tenure
        TEXT contract
        INTEGER churn
    }
    SERVICES {
        TEXT customer_id PK, FK
        TEXT phone_service
        TEXT multiple_lines
        TEXT internet_service
        TEXT online_security
        TEXT online_backup
        TEXT device_protection
        TEXT tech_support
        TEXT streaming_tv
        TEXT streaming_movies
        INTEGER num_services
    }
    BILLING {
        TEXT customer_id PK, FK
        TEXT paperless_billing
        TEXT payment_method
        REAL monthly_charges
        REAL total_charges
        REAL avg_monthly_spend
        REAL clv_estimate
    }
```

## 5. Retention Decision Matrix (Planned — Step 5)

Each customer is positioned on a **Value Tier × Churn Risk** grid. The resulting cell determines the prescribed retention action and its priority for budget allocation.

| Value Tier \ Churn Risk | **High Risk** (p ≥ 0.60) | **Medium Risk** (0.30 ≤ p < 0.60) | **Low Risk** (p < 0.30) |
|---|---|---|---|
| **High Value** | 🔴 **Priority Rescue** — dedicated account manager, premium loyalty discount, contract upgrade incentive | 🟠 **Proactive Engagement** — personalized offers, service bundle upgrade | 🟢 **Reward & Upsell** — loyalty rewards, premium cross-selling |
| **Medium Value** | 🟠 **Targeted Retention** — moderate discount, long-term contract migration | 🟡 **Nurture** — satisfaction survey, add-on service trials | 🟢 **Maintain** — standard engagement, upsell opportunities |
| **Low Value** | 🟡 **Low-Cost Retention** — automated e-mail/SMS campaigns, self-service support | ⚪ **Monitor** — automated monitoring, no paid incentives | ⚪ **No Action** — standard service |

Key decision metrics (implemented in `src/decision_engine.py`):

$$\text{Revenue at Risk}_i = p_i \times \text{MonthlyCharges}_i \times H$$

$$\text{ROI} = \frac{\sum_i p_i \cdot s \cdot \text{CLV}_i - \sum_i c_i}{\sum_i c_i}$$

where $p_i$ is the predicted churn probability, $H$ is the planning horizon in months, $s$ is the expected campaign success rate, $\text{CLV}_i$ is the customer lifetime value and $c_i$ is the intervention cost.

## 6. Project Structure

```text
./ (ISM-project)
├── data/
│   ├── raw/                  # Original IBM Telco Customer Churn dataset
│   └── processed/            # Cleaned and feature-engineered data (ETL output)
├── database/
│   └── ism_dss.db            # SQLite database (3 normalized tables)
├── models/                   # Trained model artifacts (.joblib)
├── src/
│   ├── data_generator.py     # Dataset acquisition (synthetic fallback, 7,043 rows)
│   ├── etl_pipeline.py       # Cleaning, feature engineering and loading into SQLite
│   ├── mining_engine.py      # K-Means (Value Tier) + XGBoost (Churn) + SHAP
│   └── decision_engine.py    # Decision matrix, Revenue at Risk and ROI
├── app.py                    # Streamlit DSS dashboard
├── requirements.txt          # Python dependencies
├── PROGRESS.md               # Project progress log
└── README.md                 # Project documentation
```

## 7. Setup Instructions

**Prerequisites:** Python 3.9 or later, Git.

```powershell
# Windows (PowerShell)
git clone https://github.com/Vu-Viet-Phong/ISM-project.git
cd ISM-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
git clone https://github.com/Vu-Viet-Phong/ISM-project.git
cd ISM-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Running the pipeline:**

```powershell
# Step 2 - Data acquisition (use --force to re-download, --synthetic to force the fallback generator)
.\.venv\Scripts\python.exe src/data_generator.py
```

*(Further pipeline commands will be added as subsequent steps are completed.)*

## 8. Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project Scaffolding & Environment Setup | ✅ Completed |
| 2 | Data Acquisition (`src/data_generator.py`) | ✅ Completed |
| 3 | ETL Pipeline & SQLite Database (`src/etl_pipeline.py`) | ⏳ Pending |
| 4 | Data Mining Engine (`src/mining_engine.py`) | ⏳ Pending |
| 5 | Decision Engine (`src/decision_engine.py`) | ⏳ Pending |
| 6 | Streamlit DSS Dashboard (`app.py`) & Final Documentation | ⏳ Pending |

See [PROGRESS.md](PROGRESS.md) for the detailed progress log.

## 9. Technology Stack

Python 3.9 · pandas · NumPy · scikit-learn · XGBoost · SHAP · SQLite · Plotly · Streamlit
