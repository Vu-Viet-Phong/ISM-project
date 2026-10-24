"""
ETL Pipeline Module (Tier 2 - ETL and Data Processing Layer).

This module transforms the raw IBM Telco Customer Churn dataset into an
analysis-ready, normalized SQLite Decision Support System (DSS) warehouse.

Pipeline stages:
    1. EXTRACT   - Read ``data/raw/telco_churn_raw.csv`` (all values preserved as text).
    2. TRANSFORM - Clean data types, impute the 11 blank ``TotalCharges`` values,
                   standardize column names, encode binary attributes and derive
                   managerial / predictive engineered features.
    3. LOAD      - Create the DSS schema in ``database/ism_dss.db``:
                     * Dimension tables (3NF): ``customers``, ``services``, ``billing``
                     * Integrated view:        ``customer_master`` (7,043 cleaned rows)
                     * Fact tables (DDL only):  ``fact_mining_scores`` (Step 4),
                                                ``fact_retention_actions`` (Step 5)
                     * Analytical view:         ``vw_customer_360``
                     * Audit table:             ``etl_audit_log``
    4. VALIDATE  - Verify row counts, missing values, key uniqueness and
                   referential integrity; export ``data/processed/telco_churn_processed.csv``.

Usage:
    .venv\\Scripts\\python.exe src/etl_pipeline.py
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "telco_churn_raw.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "telco_churn_processed.csv"
DB_DIR = PROJECT_ROOT / "database"
DB_FILE = DB_DIR / "ism_dss.db"

EXPECTED_ROWS = 7043

# Mapping from the original CamelCase columns to warehouse snake_case names.
# ``customerID`` is intentionally retained as the business key across all tables.
COLUMN_MAP = {
    "customerID": "customerID",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

# Pure Yes/No attributes that are encoded as 0/1 integers.
BINARY_COLUMNS = ["partner", "dependents", "phone_service", "paperless_billing"]

# Value-added services (VAS) counted in ``total_services_count``.
VALUE_ADDED_SERVICES = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]

TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12m", "12-24m", "24-48m", "48m+"]

AUTO_PAYMENT_METHODS = ["Bank transfer (automatic)", "Credit card (automatic)"]


# ---------------------------------------------------------------------------
# 1. EXTRACT
# ---------------------------------------------------------------------------
def extract(path: Path = RAW_FILE) -> pd.DataFrame:
    """
    Read the raw CSV file without any implicit type inference on ``TotalCharges``.

    Args:
        path: Location of the raw dataset.

    Returns:
        The raw DataFrame (7,043 rows x 21 columns).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run 'src/data_generator.py' first."
        )
    df = pd.read_csv(path, dtype={"TotalCharges": str}, keep_default_na=False)
    print(f"[EXTRACT]   Loaded {len(df):,} rows x {df.shape[1]} columns from {path.relative_to(PROJECT_ROOT)}")
    return df


# ---------------------------------------------------------------------------
# 2. TRANSFORM
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Perform data cleansing and type standardization.

    Cleaning rules:
        * Strip leading/trailing whitespace from every text attribute.
        * Convert ``TotalCharges`` to numeric; blank values (customers with
          ``tenure = 0`` who have not yet received a bill) are imputed with 0.0,
          which is the economically correct cumulative charge.
        * Remove duplicate records on the business key ``customerID``.

    Args:
        df: Raw DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, dictionary of cleaning statistics).
    """
    df = df.rename(columns=COLUMN_MAP).copy()
    stats = {}

    text_cols = df.select_dtypes(include="object").columns
    df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())

    blank_mask = df["total_charges"].eq("")
    stats["blank_total_charges"] = int(blank_mask.sum())
    stats["blank_with_zero_tenure"] = int((blank_mask & df["tenure"].eq(0)).sum())

    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    stats["non_numeric_total_charges"] = int(df["total_charges"].isna().sum())
    df["total_charges"] = df["total_charges"].fillna(0.0)

    before = len(df)
    df = df.drop_duplicates(subset="customerID", keep="first")
    stats["duplicates_removed"] = before - len(df)

    df["senior_citizen"] = df["senior_citizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["monthly_charges"] = df["monthly_charges"].astype(float)

    print(f"[TRANSFORM] Imputed {stats['blank_total_charges']} blank TotalCharges -> 0.0 "
          f"({stats['blank_with_zero_tenure']} with tenure = 0); "
          f"duplicates removed: {stats['duplicates_removed']}")
    return df, stats


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary attributes and derive engineered features.

    Engineered features:
        * ``tenure_group``         - Customer lifecycle stage: 0-12m, 12-24m, 24-48m, 48m+.
        * ``total_services_count`` - Number of subscribed value-added services (0-6).
        * ``avg_monthly_spend``    - Historical average monthly spend
                                     (total_charges / tenure; equals monthly_charges when tenure = 0).
        * ``is_auto_payment``      - 1 if the customer pays via automatic bank transfer or credit card.
        * ``churn_binary``         - Target variable: 1 if churn == 'Yes', otherwise 0.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame enriched with encoded and engineered columns.
    """
    df = df.copy()

    for col in BINARY_COLUMNS:
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype(int)

    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    df["total_services_count"] = (df[VALUE_ADDED_SERVICES] == "Yes").sum(axis=1).astype(int)
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["total_charges"] / df["tenure"].replace(0, 1), df["monthly_charges"]
    ).round(2)
    df["is_auto_payment"] = df["payment_method"].isin(AUTO_PAYMENT_METHODS).astype(int)
    df["churn_binary"] = df["churn"].map({"Yes": 1, "No": 0}).astype(int)

    print("[TRANSFORM] Engineered features: tenure_group, total_services_count, "
          "avg_monthly_spend, is_auto_payment, churn_binary")
    return df


# ---------------------------------------------------------------------------
# 3. LOAD - DSS warehouse schema (DDL)
# ---------------------------------------------------------------------------
SCHEMA_DDL = """
-- Drop in dependency order (facts and views first) to allow idempotent re-runs.
DROP VIEW  IF EXISTS vw_customer_360;
DROP VIEW  IF EXISTS customer_master;
DROP TABLE IF EXISTS fact_retention_actions;
DROP TABLE IF EXISTS fact_mining_scores;
DROP TABLE IF EXISTS billing;
DROP TABLE IF EXISTS services;
DROP TABLE IF EXISTS customers;

-- ---------------------------------------------------------------------------
-- Dimension 1: customer demographics, lifecycle and churn outcome
-- ---------------------------------------------------------------------------
CREATE TABLE customers (
    customerID      TEXT    PRIMARY KEY,
    gender          TEXT    NOT NULL CHECK (gender IN ('Male', 'Female')),
    senior_citizen  INTEGER NOT NULL CHECK (senior_citizen IN (0, 1)),
    partner         INTEGER NOT NULL CHECK (partner IN (0, 1)),
    dependents      INTEGER NOT NULL CHECK (dependents IN (0, 1)),
    tenure          INTEGER NOT NULL CHECK (tenure >= 0),
    tenure_group    TEXT    NOT NULL CHECK (tenure_group IN ('0-12m', '12-24m', '24-48m', '48m+')),
    contract        TEXT    NOT NULL CHECK (contract IN ('Month-to-month', 'One year', 'Two year')),
    churn           TEXT    NOT NULL CHECK (churn IN ('Yes', 'No')),
    churn_binary    INTEGER NOT NULL CHECK (churn_binary IN (0, 1))
);

-- ---------------------------------------------------------------------------
-- Dimension 2: subscribed telecommunication services (1:1 with customers)
-- ---------------------------------------------------------------------------
CREATE TABLE services (
    customerID            TEXT    PRIMARY KEY REFERENCES customers(customerID) ON DELETE CASCADE,
    phone_service         INTEGER NOT NULL CHECK (phone_service IN (0, 1)),
    multiple_lines        TEXT    NOT NULL,
    internet_service      TEXT    NOT NULL CHECK (internet_service IN ('DSL', 'Fiber optic', 'No')),
    online_security       TEXT    NOT NULL,
    online_backup         TEXT    NOT NULL,
    device_protection     TEXT    NOT NULL,
    tech_support          TEXT    NOT NULL,
    streaming_tv          TEXT    NOT NULL,
    streaming_movies      TEXT    NOT NULL,
    total_services_count  INTEGER NOT NULL CHECK (total_services_count BETWEEN 0 AND 6)
);

-- ---------------------------------------------------------------------------
-- Dimension 3: billing and payment profile (1:1 with customers)
-- ---------------------------------------------------------------------------
CREATE TABLE billing (
    customerID         TEXT    PRIMARY KEY REFERENCES customers(customerID) ON DELETE CASCADE,
    paperless_billing  INTEGER NOT NULL CHECK (paperless_billing IN (0, 1)),
    payment_method     TEXT    NOT NULL,
    is_auto_payment    INTEGER NOT NULL CHECK (is_auto_payment IN (0, 1)),
    monthly_charges    REAL    NOT NULL CHECK (monthly_charges >= 0),
    total_charges      REAL    NOT NULL CHECK (total_charges >= 0),
    avg_monthly_spend  REAL    NOT NULL CHECK (avg_monthly_spend >= 0)
);

-- ---------------------------------------------------------------------------
-- Fact 1 (populated in Step 4): data mining outputs per customer
-- ---------------------------------------------------------------------------
CREATE TABLE fact_mining_scores (
    customerID         TEXT    PRIMARY KEY REFERENCES customers(customerID) ON DELETE CASCADE,
    value_tier         TEXT    NOT NULL CHECK (value_tier IN ('High', 'Medium', 'Low')),
    cluster_id         INTEGER NOT NULL,
    churn_probability  REAL    NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    risk_level         TEXT    NOT NULL CHECK (risk_level IN ('High', 'Medium', 'Low')),
    top_shap_driver_1  TEXT,
    top_shap_driver_2  TEXT,
    scored_at          TEXT    NOT NULL
);
CREATE INDEX idx_scores_tier_risk ON fact_mining_scores(value_tier, risk_level);

-- ---------------------------------------------------------------------------
-- Fact 2 (populated in Step 5): prescriptive retention decisions per customer
-- ---------------------------------------------------------------------------
CREATE TABLE fact_retention_actions (
    customerID              TEXT    PRIMARY KEY REFERENCES customers(customerID) ON DELETE CASCADE,
    annualized_revenue      REAL    NOT NULL CHECK (annualized_revenue >= 0),
    revenue_at_risk         REAL    NOT NULL CHECK (revenue_at_risk >= 0),
    decision_quadrant       TEXT    NOT NULL,
    recommended_action      TEXT    NOT NULL,
    intervention_cost       REAL    NOT NULL CHECK (intervention_cost >= 0),
    expected_saved_revenue  REAL    NOT NULL,
    expected_net_benefit    REAL    NOT NULL,
    expected_roi            REAL,
    decided_at              TEXT    NOT NULL,
    action_status           TEXT    NOT NULL DEFAULT 'Pending' CHECK (action_status IN ('Pending', 'Contacted', 'Retained', 'Churned')),
    manager_notes           TEXT    NOT NULL DEFAULT '',
    updated_at              TEXT
);
CREATE INDEX idx_actions_quadrant ON fact_retention_actions(decision_quadrant);

-- ---------------------------------------------------------------------------
-- Integrated view: one cleaned, analysis-ready row per customer
-- ---------------------------------------------------------------------------
CREATE VIEW customer_master AS
SELECT
    c.customerID, c.gender, c.senior_citizen, c.partner, c.dependents,
    c.tenure, c.tenure_group, c.contract,
    s.phone_service, s.multiple_lines, s.internet_service,
    s.online_security, s.online_backup, s.device_protection, s.tech_support,
    s.streaming_tv, s.streaming_movies, s.total_services_count,
    b.paperless_billing, b.payment_method, b.is_auto_payment,
    b.monthly_charges, b.total_charges, b.avg_monthly_spend,
    c.churn, c.churn_binary
FROM customers c
JOIN services  s ON s.customerID = c.customerID
JOIN billing   b ON b.customerID = c.customerID;

-- ---------------------------------------------------------------------------
-- Analytical view: Customer 360 (master + mining scores + retention decisions)
-- ---------------------------------------------------------------------------
CREATE VIEW vw_customer_360 AS
SELECT
    m.*,
    f.value_tier, f.cluster_id, f.churn_probability, f.risk_level,
    f.top_shap_driver_1, f.top_shap_driver_2, f.scored_at,
    r.annualized_revenue, r.revenue_at_risk, r.decision_quadrant, r.recommended_action,
    r.intervention_cost, r.expected_saved_revenue, r.expected_net_benefit, r.expected_roi,
    r.action_status, r.manager_notes, r.updated_at
FROM customer_master m
LEFT JOIN fact_mining_scores     f ON f.customerID = m.customerID
LEFT JOIN fact_retention_actions r ON r.customerID = m.customerID;

-- ---------------------------------------------------------------------------
-- ETL audit trail (persisted across runs)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS etl_audit_log (
    run_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at                TEXT    NOT NULL,
    source_file           TEXT    NOT NULL,
    rows_extracted        INTEGER NOT NULL,
    rows_loaded           INTEGER NOT NULL,
    blank_total_charges   INTEGER NOT NULL,
    duplicates_removed    INTEGER NOT NULL,
    status                TEXT    NOT NULL
);
"""

CUSTOMERS_COLS = ["customerID", "gender", "senior_citizen", "partner", "dependents",
                  "tenure", "tenure_group", "contract", "churn", "churn_binary"]
SERVICES_COLS = ["customerID", "phone_service", "multiple_lines", "internet_service",
                 *VALUE_ADDED_SERVICES, "total_services_count"]
BILLING_COLS = ["customerID", "paperless_billing", "payment_method", "is_auto_payment",
                "monthly_charges", "total_charges", "avg_monthly_spend"]


def load(df: pd.DataFrame, db_path: Path = DB_FILE) -> None:
    """
    Create the DSS schema and load the three normalized dimension tables.

    Data are appended into pre-defined tables (rather than letting pandas create
    them) so that primary keys, foreign keys and CHECK constraints are enforced.

    Args:
        df: Transformed DataFrame.
        db_path: Target SQLite database file.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_DDL)
        df[CUSTOMERS_COLS].to_sql("customers", conn, if_exists="append", index=False)
        df[SERVICES_COLS].to_sql("services", conn, if_exists="append", index=False)
        df[BILLING_COLS].to_sql("billing", conn, if_exists="append", index=False)
        conn.commit()
    print(f"[LOAD]      Schema created and dimension tables loaded into {db_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# 4. VALIDATE & EXPORT
# ---------------------------------------------------------------------------
def validate_and_export(db_path: Path = DB_FILE) -> pd.DataFrame:
    """
    Run data-quality checks on the warehouse and export the integrated dataset.

    Checks performed:
        * Row count of every dimension table and of ``customer_master`` equals 7,043.
        * ``customer_master`` contains zero missing values.
        * ``PRAGMA foreign_key_check`` reports no referential-integrity violations.

    Args:
        db_path: SQLite database file.

    Returns:
        The ``customer_master`` DataFrame that is exported to the processed layer.

    Raises:
        AssertionError: If any data-quality check fails.
    """
    with sqlite3.connect(db_path) as conn:
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ["customers", "services", "billing", "customer_master",
                            "fact_mining_scores", "fact_retention_actions"]}
        fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
        master = pd.read_sql("SELECT * FROM customer_master", conn)

    missing = int(master.isna().sum().sum())
    for table in ["customers", "services", "billing", "customer_master"]:
        assert counts[table] == EXPECTED_ROWS, f"{table}: expected {EXPECTED_ROWS}, got {counts[table]}"
    assert missing == 0, f"customer_master contains {missing} missing values"
    assert not fk_violations, f"Foreign-key violations detected: {fk_violations}"
    assert master["customerID"].is_unique, "customerID is not unique"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(PROCESSED_FILE, index=False)
    print(f"[VALIDATE]  All checks passed; exported {PROCESSED_FILE.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 64)
    print(" ETL SUMMARY - ISM DSS Warehouse (database/ism_dss.db)")
    print("=" * 64)
    for table, n in counts.items():
        print(f" {table:<28}: {n:>6,} rows")
    print(f" Missing values (customer_master): {missing}")
    print(f" Foreign-key violations          : {len(fk_violations)}")
    print(f" Processed file shape            : {master.shape[0]:,} rows x {master.shape[1]} columns")
    print(f" Churn rate (churn_binary mean)  : {master['churn_binary'].mean() * 100:.2f}%")
    print("-" * 64)
    print(" tenure_group distribution (customers | churn rate):")
    grp = master.groupby("tenure_group")["churn_binary"].agg(["count", "mean"]).reindex(TENURE_LABELS)
    for label, row in grp.iterrows():
        print(f"   {label:<8}: {int(row['count']):>5,} | {row['mean'] * 100:5.1f}%")
    print(" total_services_count: mean = {:.2f}, min = {}, max = {}".format(
        master["total_services_count"].mean(), master["total_services_count"].min(),
        master["total_services_count"].max()))
    print(" avg_monthly_spend   : mean = {:.2f}, min = {:.2f}, max = {:.2f}".format(
        master["avg_monthly_spend"].mean(), master["avg_monthly_spend"].min(),
        master["avg_monthly_spend"].max()))
    print(" is_auto_payment     : {:.1f}% of customers".format(master["is_auto_payment"].mean() * 100))
    print("=" * 64)
    return master


def log_run(stats: dict, rows_extracted: int, rows_loaded: int, status: str) -> None:
    """Append an entry to the ``etl_audit_log`` table."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT INTO etl_audit_log (run_at, source_file, rows_extracted, rows_loaded, "
            "blank_total_charges, duplicates_removed, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), str(RAW_FILE.relative_to(PROJECT_ROOT)),
             rows_extracted, rows_loaded, stats.get("blank_total_charges", 0),
             stats.get("duplicates_removed", 0), status),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_pipeline() -> pd.DataFrame:
    """Execute the full Extract-Transform-Load-Validate pipeline."""
    raw = extract()
    cleaned, stats = clean(raw)
    transformed = engineer_features(cleaned)
    load(transformed)
    master = validate_and_export()
    log_run(stats, rows_extracted=len(raw), rows_loaded=len(master), status="SUCCESS")
    return master


def main() -> int:
    try:
        run_pipeline()
    except Exception as exc:
        print(f"[ERROR] ETL pipeline failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
