"""
Data Acquisition Module (Tier 1 - Data Layer).

This module acquires the IBM Telco Customer Churn dataset (7,043 customers,
21 attributes) and stores it, unmodified, at ``data/raw/telco_churn_raw.csv``.

Acquisition strategy:
    1. Attempt to download the original dataset from a list of public mirrors.
    2. Validate the downloaded file against the expected schema and row count.
    3. If every mirror fails (e.g. no Internet connection), fall back to a
       statistically faithful synthetic generator that reproduces the exact
       schema, row count, categorical domains, approximate churn rate and the
       known data-quality issue (blank ``TotalCharges`` for zero-tenure customers).

Usage:
    .venv\\Scripts\\python.exe src/data_generator.py [--force] [--synthetic]
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "telco_churn_raw.csv"

DATASET_URLS = [
    # Official IBM repository
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    # Community mirrors of the Kaggle file "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/carlosfab/dsnp2/master/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

EXPECTED_ROWS = 7043
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
REQUEST_TIMEOUT = 30  # seconds
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Download & validation
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame matches the expected Telco Churn schema."""
    return list(df.columns) == EXPECTED_COLUMNS and len(df) == EXPECTED_ROWS


def download_dataset() -> pd.DataFrame:
    """
    Attempt to download the original dataset from each configured mirror.

    ``TotalCharges`` is read as a string (``dtype=str``) so that the raw layer
    preserves the original blank values for the ETL stage to handle explicitly.

    Returns:
        The validated raw DataFrame, or ``None`` if every mirror fails.
    """
    for url in DATASET_URLS:
        try:
            print(f"[INFO] Downloading from: {url}")
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text), dtype={"TotalCharges": str},
                             keep_default_na=False)
            if validate_schema(df):
                print(f"[OK]   Download succeeded and schema validated ({len(df):,} rows).")
                return df
            print(f"[WARN] Schema mismatch (rows={len(df)}, cols={df.shape[1]}). Trying next mirror.")
        except Exception as exc:  # network errors, HTTP errors, parsing errors
            print(f"[WARN] Failed: {type(exc).__name__}: {exc}")
    return None


# ---------------------------------------------------------------------------
# Synthetic fallback generator
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(n_rows: int = EXPECTED_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors the IBM Telco Customer Churn schema.

    The generator encodes the main behavioural relationships documented for the
    original data: month-to-month contracts, fibre-optic service, electronic-check
    payment and short tenure increase churn propensity, while long contracts and
    add-on security/support services reduce it.

    Args:
        n_rows: Number of customer records to generate.
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame with the 21 original columns, all values formatted as in the source file.
    """
    rng = np.random.default_rng(seed)

    def choice(options, probs, size=n_rows):
        return rng.choice(options, size=size, p=probs)

    # --- Demographics ---
    gender = choice(["Male", "Female"], [0.505, 0.495])
    senior = choice([0, 1], [0.838, 0.162])
    partner = choice(["Yes", "No"], [0.483, 0.517])
    dependents = np.where(partner == "Yes",
                          choice(["Yes", "No"], [0.52, 0.48]),
                          choice(["Yes", "No"], [0.10, 0.90]))

    # --- Contract & tenure ---
    contract = choice(["Month-to-month", "One year", "Two year"], [0.550, 0.209, 0.241])
    tenure = np.empty(n_rows, dtype=int)
    for ctype, low, high in [("Month-to-month", 0, 72), ("One year", 0, 72), ("Two year", 0, 72)]:
        mask = contract == ctype
        if ctype == "Month-to-month":
            t = rng.exponential(scale=18, size=mask.sum())
        elif ctype == "One year":
            t = rng.normal(loc=42, scale=18, size=mask.sum())
        else:
            t = rng.normal(loc=57, scale=15, size=mask.sum())
        tenure[mask] = np.clip(np.round(t), low, high).astype(int)
    # Reproduce the 11 zero-tenure (newly joined) customers of the original data
    tenure[tenure == 0] = 1
    zero_idx = rng.choice(np.where(contract == "Two year")[0], size=11, replace=False)
    tenure[zero_idx] = 0

    # --- Services ---
    phone = choice(["Yes", "No"], [0.903, 0.097])
    multiple_lines = np.where(phone == "No", "No phone service", choice(["Yes", "No"], [0.467, 0.533]))
    internet = choice(["Fiber optic", "DSL", "No"], [0.440, 0.344, 0.216])

    def addon(p_yes):
        return np.where(internet == "No", "No internet service", choice(["Yes", "No"], [p_yes, 1 - p_yes]))

    online_security, online_backup = addon(0.37), addon(0.44)
    device_protection, tech_support = addon(0.44), addon(0.37)
    streaming_tv, streaming_movies = addon(0.49), addon(0.50)

    paperless = choice(["Yes", "No"], [0.592, 0.408])
    payment = choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"], [0.336, 0.229, 0.219, 0.216])

    # --- Charges (price built from subscribed services) ---
    monthly = np.where(phone == "Yes", 20.0, 0.0)
    monthly += np.where(multiple_lines == "Yes", 5.0, 0.0)
    monthly += np.select([internet == "Fiber optic", internet == "DSL"], [50.0, 25.0], 0.0)
    for svc in [online_security, online_backup, device_protection, tech_support,
                streaming_tv, streaming_movies]:
        monthly += np.where(svc == "Yes", rng.uniform(5, 10, n_rows), 0.0)
    monthly = np.clip(monthly + rng.normal(0, 2, n_rows), 18.25, 118.75).round(2)
    total = (monthly * tenure * rng.uniform(0.95, 1.05, n_rows)).round(2)

    # --- Churn propensity (logistic model) ---
    logit = (-1.4
             + 1.3 * (contract == "Month-to-month") - 1.2 * (contract == "Two year")
             - 0.035 * tenure
             + 0.8 * (internet == "Fiber optic") - 0.6 * (internet == "No")
             + 0.6 * (payment == "Electronic check")
             + 0.3 * (paperless == "Yes") + 0.3 * senior
             - 0.4 * (online_security == "Yes") - 0.4 * (tech_support == "Yes")
             - 0.2 * (partner == "Yes")
             + 0.01 * (monthly - 65))
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n_rows) < churn_prob, "Yes", "No")

    # --- Customer IDs in the original "NNNN-XXXXX" format ---
    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    ids = set()
    while len(ids) < n_rows:
        ids.add(f"{rng.integers(0, 10000):04d}-{''.join(rng.choice(letters, 5))}")
    customer_ids = list(ids)

    # Original file stores TotalCharges as text with a single blank space when tenure == 0
    total_str = np.where(tenure == 0, " ", total.astype(str))

    return pd.DataFrame({
        "customerID": customer_ids, "gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multiple_lines, "InternetService": internet,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total_str, "Churn": churn,
    })[EXPECTED_COLUMNS]


# ---------------------------------------------------------------------------
# Data profiling
# ---------------------------------------------------------------------------
def print_data_profile(df: pd.DataFrame, source: str) -> None:
    """Print a concise data-quality profile of the acquired raw dataset."""
    blank_total = df["TotalCharges"].astype(str).str.strip().eq("").sum()
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    print("\n" + "=" * 60)
    print(" RAW DATA PROFILE - IBM Telco Customer Churn")
    print("=" * 60)
    print(f" Source                       : {source}")
    print(f" Output file                  : {RAW_FILE.relative_to(PROJECT_ROOT)}")
    print(f" Rows                         : {df.shape[0]:,}")
    print(f" Columns                      : {df.shape[1]}")
    print(f" Unique customerID            : {df['customerID'].nunique():,}")
    print(f" Churn = Yes                  : {(df['Churn'] == 'Yes').sum():,} ({churn_rate:.2f}%)")
    print(f" Churn = No                   : {(df['Churn'] == 'No').sum():,} ({100 - churn_rate:.2f}%)")
    print(f" Blank/whitespace TotalCharges: {blank_total}")
    print(f" Customers with tenure == 0   : {(df['tenure'] == 0).sum()}")
    print(f" Duplicate rows               : {df.duplicated().sum()}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def acquire(force: bool = False, synthetic: bool = False) -> pd.DataFrame:
    """
    Acquire the raw dataset and persist it to ``data/raw/telco_churn_raw.csv``.

    Args:
        force: Re-acquire the data even if the raw file already exists.
        synthetic: Skip downloading and use the synthetic generator directly.

    Returns:
        The raw DataFrame.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists() and not force and not synthetic:
        print(f"[INFO] Raw file already exists: {RAW_FILE.relative_to(PROJECT_ROOT)} (use --force to refresh)")
        df = pd.read_csv(RAW_FILE, dtype={"TotalCharges": str}, keep_default_na=False)
        print_data_profile(df, "Cached local file")
        return df

    df = None if synthetic else download_dataset()
    source = "IBM Telco Customer Churn (original, downloaded)"
    if df is None:
        print("[WARN] Using synthetic fallback generator (7,043 rows, original schema).")
        df = generate_synthetic_dataset()
        source = "Synthetic fallback generator (seed=42)"

    df.to_csv(RAW_FILE, index=False)
    print(f"[OK]   Saved raw dataset to {RAW_FILE.relative_to(PROJECT_ROOT)}")
    print_data_profile(df, source)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire the IBM Telco Customer Churn dataset.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the raw file exists.")
    parser.add_argument("--synthetic", action="store_true", help="Force the synthetic fallback generator.")
    args = parser.parse_args()

    df = acquire(force=args.force, synthetic=args.synthetic)
    if not validate_schema(df):
        print("[ERROR] Acquired dataset does not match the expected schema.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
