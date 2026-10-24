"""
Data Mining Engine (Tier 3 - Analytics and Decision Logic Layer).

This module implements the three analytical components of the DSS and persists
their outputs to the ``fact_mining_scores`` table of ``database/ism_dss.db``.

    Module 1 - Customer Value Tiering (unsupervised learning)
        StandardScaler + K-Means (k = 3) on tenure, monthly_charges, total_charges
        and total_services_count. Clusters are ranked by their mean economic value
        and mapped to the value tiers 'High', 'Medium' and 'Low'.

    Module 2 - Churn Prediction (supervised learning)
        ColumnTransformer (OneHotEncoder + StandardScaler) + XGBoost classifier in a
        single scikit-learn Pipeline, trained on a stratified 80/20 split with
        ``scale_pos_weight`` to handle class imbalance. The deployed pipeline is
        refitted on all 7,043 customers and scores every customer with a churn
        probability and a risk level ('High' >= 0.60 > 'Medium' >= 0.30 > 'Low').

    Module 3 - Explainable AI (SHAP TreeExplainer)
        Exact TreeSHAP values are computed for every customer. One-hot encoded
        contributions are aggregated back to their original business attribute, and
        the two attributes that increase churn risk the most are stored in a
        manager-readable form, e.g. ``"contract=Month-to-month (+0.82)"``.
        SHAP values are expressed in log-odds units.

Artifacts written to ``models/``:
    kmeans_value_pipeline.joblib, xgb_churn_pipeline.joblib,
    shap_metadata.joblib, model_metrics.json

Usage:
    .venv\\Scripts\\python.exe src/mining_engine.py
"""

import json
import sqlite3
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_FILE = PROJECT_ROOT / "database" / "ism_dss.db"
MODELS_DIR = PROJECT_ROOT / "models"
KMEANS_FILE = MODELS_DIR / "kmeans_value_pipeline.joblib"
XGB_FILE = MODELS_DIR / "xgb_churn_pipeline.joblib"
SHAP_FILE = MODELS_DIR / "shap_metadata.joblib"
METRICS_FILE = MODELS_DIR / "model_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CLUSTERS = 3
SILHOUETTE_K_RANGE = range(2, 7)

# Module 1 - value segmentation features
VALUE_FEATURES = ["tenure", "monthly_charges", "total_charges", "total_services_count"]
VALUE_TIER_LABELS = ["High", "Medium", "Low"]  # ordered from highest to lowest value

# Module 2 - churn model features (identifiers and target columns are excluded)
TARGET = "churn_binary"
CATEGORICAL_FEATURES = [
    "gender", "tenure_group", "contract", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "payment_method",
]
NUMERIC_FEATURES = [
    "senior_citizen", "partner", "dependents", "phone_service", "paperless_billing",
    "is_auto_payment", "tenure", "total_services_count",
    "monthly_charges", "total_charges", "avg_monthly_spend",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Risk thresholds (must match the CHECK constraint domain of fact_mining_scores)
HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.30
CLASSIFICATION_THRESHOLD = 0.50

XGB_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
    eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
)

MONEY_FEATURES = {"monthly_charges", "total_charges", "avg_monthly_spend"}


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def load_customer_master(db_path: Path = DB_FILE) -> pd.DataFrame:
    """Load the integrated, cleaned customer records from the ``customer_master`` view."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM customer_master ORDER BY customerID", conn)
    print(f"[DATA]   Loaded {len(df):,} customers x {df.shape[1]} columns from customer_master")
    return df


# ---------------------------------------------------------------------------
# Module 1 - Customer Value Tiering (K-Means)
# ---------------------------------------------------------------------------
def train_value_tiering(df: pd.DataFrame):
    """
    Segment customers into three economic value tiers with K-Means.

    The Silhouette coefficient is reported for k = 2..6 as a cluster-validity
    diagnostic; k = 3 is retained because it maps directly onto the three-level
    managerial value hierarchy used by the decision matrix.

    Clusters are ranked by mean ``total_charges`` (realized customer value), with
    mean ``monthly_charges`` (current revenue run-rate) as the tie-breaker.

    Args:
        df: Customer master DataFrame.

    Returns:
        Tuple of (fitted pipeline, cluster->tier mapping, cluster ids, tier labels,
        silhouette scores dict, cluster profile DataFrame).
    """
    X = df[VALUE_FEATURES].astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    silhouettes = {}
    for k in SILHOUETTE_K_RANGE:
        labels_k = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_scaled)
        silhouettes[k] = float(silhouette_score(X_scaled, labels_k, random_state=RANDOM_STATE))

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)),
    ])
    cluster_ids = pipeline.fit_predict(X)

    profile = (df.assign(cluster_id=cluster_ids)
                 .groupby("cluster_id")
                 .agg(customers=("customerID", "count"),
                      tenure=("tenure", "mean"),
                      monthly_charges=("monthly_charges", "mean"),
                      total_charges=("total_charges", "mean"),
                      total_services_count=("total_services_count", "mean"),
                      churn_rate=("churn_binary", "mean")))
    ranked = profile.sort_values(["total_charges", "monthly_charges"], ascending=False).index
    tier_map = {int(cid): VALUE_TIER_LABELS[rank] for rank, cid in enumerate(ranked)}
    profile["value_tier"] = profile.index.map(tier_map)
    profile["revenue_share"] = (df.assign(cluster_id=cluster_ids)
                                  .groupby("cluster_id")["monthly_charges"].sum()
                                / df["monthly_charges"].sum())
    profile = profile.loc[ranked]

    value_tiers = pd.Series(cluster_ids).map(tier_map).values
    print(f"[K-MEANS] Silhouette scores: " +
          ", ".join(f"k={k}: {s:.3f}" for k, s in silhouettes.items()))
    print(f"[K-MEANS] Trained k={N_CLUSTERS}; cluster -> tier mapping: {tier_map}")
    return pipeline, tier_map, cluster_ids, value_tiers, silhouettes, profile


# ---------------------------------------------------------------------------
# Module 2 - Churn Prediction (XGBoost)
# ---------------------------------------------------------------------------
def build_churn_pipeline(scale_pos_weight: float) -> Pipeline:
    """
    Build the end-to-end churn prediction pipeline.

    Encapsulating preprocessing inside the pipeline guarantees that encoders and
    scalers are fitted only on training data (no data leakage) and allows the
    Streamlit dashboard to score a single raw customer record directly.
    """
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def assign_risk_level(prob: np.ndarray) -> np.ndarray:
    """Map churn probabilities to the 'High' / 'Medium' / 'Low' risk levels."""
    return np.select([prob >= HIGH_RISK_THRESHOLD, prob >= MEDIUM_RISK_THRESHOLD],
                     ["High", "Medium"], default="Low")


def train_churn_model(df: pd.DataFrame):
    """
    Train, evaluate and refit the XGBoost churn pipeline.

    Evaluation protocol: stratified 80/20 hold-out split; metrics are computed on
    the unseen test set. The deployed model is then refitted on all customers.

    Args:
        df: Customer master DataFrame.

    Returns:
        Tuple of (deployed pipeline, metrics dict).
    """
    X, y = df[MODEL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    eval_pipeline = build_churn_pipeline(spw).fit(X_train, y_train)

    prob_test = eval_pipeline.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= CLASSIFICATION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
    prob_train = eval_pipeline.predict_proba(X_train)[:, 1]

    metrics = {
        "train_size": int(len(X_train)), "test_size": int(len(X_test)),
        "train_churn_rate": float(y_train.mean()), "test_churn_rate": float(y_test.mean()),
        "scale_pos_weight": round(spw, 4),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "roc_auc": float(roc_auc_score(y_test, prob_test)),
        "pr_auc": float(average_precision_score(y_test, prob_test)),
        "precision": float(precision_score(y_test, pred_test)),
        "recall": float(recall_score(y_test, pred_test)),
        "f1": float(f1_score(y_test, pred_test)),
        "accuracy": float((pred_test == y_test).mean()),
        "brier_score": float(brier_score_loss(y_test, prob_test)),
        "train_roc_auc": float(roc_auc_score(y_train, prob_train)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "xgb_params": {k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
    }
    print(f"[XGBOOST] Test ROC-AUC={metrics['roc_auc']:.4f} | PR-AUC={metrics['pr_auc']:.4f} | "
          f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f}")
    print(f"[XGBOOST] Confusion matrix (test): TN={tn} FP={fp} FN={fn} TP={tp}")

    spw_full = float((y == 0).sum() / (y == 1).sum())
    final_pipeline = build_churn_pipeline(spw_full).fit(X, y)
    print(f"[XGBOOST] Deployed pipeline refitted on all {len(X):,} customers (scale_pos_weight={spw_full:.3f})")
    return final_pipeline, metrics


# ---------------------------------------------------------------------------
# Module 3 - Explainable AI (SHAP)
# ---------------------------------------------------------------------------
def encoded_to_original_mapping(pipeline: Pipeline) -> list:
    """Return, for each encoded model column, the name of its original business attribute."""
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    mapping = []
    for col, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
        mapping.extend([col] * len(cats))
    mapping.extend(NUMERIC_FEATURES)
    return mapping


def compute_shap_values(pipeline: Pipeline, X: pd.DataFrame):
    """
    Compute exact TreeSHAP values (log-odds) for every customer.

    Uses ``shap.TreeExplainer``; falls back to XGBoost's native TreeSHAP
    implementation (``pred_contribs=True``), which is mathematically identical,
    if the installed SHAP version is incompatible with the XGBoost model format.

    Returns:
        Tuple of (SHAP matrix [n_customers x n_encoded_features], expected value, method name).
    """
    X_enc = pipeline.named_steps["preprocessor"].transform(X)
    model = pipeline.named_steps["model"]
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(X_enc))
        base = float(np.ravel(explainer.expected_value)[0])
        method = f"shap.TreeExplainer (shap {shap.__version__})"
    except Exception as exc:  # pragma: no cover - compatibility fallback
        import xgboost as xgb
        print(f"[SHAP]    TreeExplainer unavailable ({type(exc).__name__}); using XGBoost native TreeSHAP")
        contribs = model.get_booster().predict(xgb.DMatrix(X_enc), pred_contribs=True)
        values, base = contribs[:, :-1], float(contribs[0, -1])
        method = "xgboost native TreeSHAP (pred_contribs)"
    return values, base, method


def format_value(feature: str, value) -> str:
    """Render an original attribute value in a manager-readable form."""
    if feature in MONEY_FEATURES:
        return f"{float(value):.2f}"
    if isinstance(value, (int, np.integer, float, np.floating)):
        return str(int(value)) if float(value).is_integer() else f"{float(value):.2f}"
    return str(value)


def extract_top_drivers(shap_by_feature: pd.DataFrame, X: pd.DataFrame, top_n: int = 2) -> list:
    """
    Extract, per customer, the attributes with the largest positive SHAP contribution.

    Only risk-increasing (positive) contributions are reported; if a customer has
    fewer than ``top_n`` positive drivers, the remaining slots are ``None``.

    Returns:
        List of length ``top_n`` containing arrays of formatted driver strings.
    """
    values = shap_by_feature.values
    features = shap_by_feature.columns.to_numpy()
    order = np.argsort(-values, axis=1)[:, :top_n]
    drivers = [[None] * len(X) for _ in range(top_n)]
    X_records = X.reset_index(drop=True)
    for i in range(len(X)):
        for rank in range(top_n):
            j = order[i, rank]
            contribution = values[i, j]
            if contribution > 0:
                feat = features[j]
                drivers[rank][i] = (f"{feat}={format_value(feat, X_records.at[i, feat])} "
                                    f"(+{contribution:.2f})")
    return drivers


def explain_churn(pipeline: Pipeline, X: pd.DataFrame):
    """
    Run Module 3: compute SHAP values, aggregate to business attributes and
    derive per-customer top drivers and global feature importance.

    Returns:
        Tuple of (driver_1 list, driver_2 list, aggregated SHAP DataFrame,
        global importance Series, metadata dict).
    """
    values, base, method = compute_shap_values(pipeline, X)
    mapping = encoded_to_original_mapping(pipeline)
    encoded_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())

    shap_by_feature = (pd.DataFrame(values, columns=encoded_names)
                         .T.groupby(np.array(mapping)).sum().T)[MODEL_FEATURES]
    global_importance = shap_by_feature.abs().mean().sort_values(ascending=False)
    driver_1, driver_2 = extract_top_drivers(shap_by_feature, X, top_n=2)

    metadata = {
        "method": method,
        "units": "log-odds",
        "expected_value": base,
        "encoded_feature_names": encoded_names,
        "encoded_to_original": mapping,
        "model_features": MODEL_FEATURES,
        "global_importance": global_importance.to_dict(),
        "shap_by_feature": shap_by_feature.astype(np.float32),
        "customer_ids": None,  # filled by the caller
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"[SHAP]    Computed {values.shape[0]:,} x {values.shape[1]} SHAP values via {method}")
    return driver_1, driver_2, shap_by_feature, global_importance, metadata


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def persist_scores(scores: pd.DataFrame, db_path: Path = DB_FILE) -> None:
    """
    Write mining outputs to ``fact_mining_scores`` atomically and idempotently.

    Existing rows are deleted inside the same transaction before the bulk insert,
    so re-running the engine never violates the primary key.
    """
    cols = ["customerID", "value_tier", "cluster_id", "churn_probability", "risk_level",
            "top_shap_driver_1", "top_shap_driver_2", "scored_at"]
    rows = [tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in r)
            for r in scores[cols].itertuples(index=False, name=None)]
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        with conn:  # single transaction: commit on success, rollback on error
            conn.execute("DELETE FROM fact_mining_scores;")
            conn.executemany(f"INSERT INTO fact_mining_scores ({', '.join(cols)}) "
                             f"VALUES ({', '.join('?' * len(cols))})", rows)
    finally:
        conn.close()
    print(f"[PERSIST] Wrote {len(rows):,} rows to fact_mining_scores")


def save_artifacts(kmeans_pipeline, tier_map, xgb_pipeline, shap_meta, metrics) -> None:
    """Serialize model pipelines, SHAP metadata and evaluation metrics to ``models/``."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": kmeans_pipeline, "features": VALUE_FEATURES, "tier_map": tier_map},
                KMEANS_FILE)
    joblib.dump({"pipeline": xgb_pipeline, "features": MODEL_FEATURES,
                 "categorical_features": CATEGORICAL_FEATURES, "numeric_features": NUMERIC_FEATURES,
                 "risk_thresholds": {"high": HIGH_RISK_THRESHOLD, "medium": MEDIUM_RISK_THRESHOLD}},
                XGB_FILE)
    joblib.dump(shap_meta, SHAP_FILE)
    METRICS_FILE.write_text(json.dumps(
        metrics, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"[ARTIFACT] Saved {KMEANS_FILE.name}, {XGB_FILE.name}, {SHAP_FILE.name}, {METRICS_FILE.name}")


# ---------------------------------------------------------------------------
# Inference helpers (used by the Streamlit dashboard in Step 6)
# ---------------------------------------------------------------------------
def load_artifacts() -> dict:
    """Load all persisted model artifacts."""
    return {"kmeans": joblib.load(KMEANS_FILE), "xgb": joblib.load(XGB_FILE),
            "shap": joblib.load(SHAP_FILE)}


def score_customer(record: dict, artifacts: dict = None) -> dict:
    """
    Score a single customer record (raw attribute values as in ``customer_master``).

    Returns:
        Dictionary with value tier, cluster id, churn probability, risk level and
        the two strongest risk-increasing drivers.
    """
    artifacts = artifacts or load_artifacts()
    X = pd.DataFrame([record])
    km, xgb_art = artifacts["kmeans"], artifacts["xgb"]
    cluster = int(km["pipeline"].predict(X[km["features"]].astype(float))[0])
    prob = float(xgb_art["pipeline"].predict_proba(X[xgb_art["features"]])[0, 1])
    d1, d2, *_ = explain_churn(xgb_art["pipeline"], X[xgb_art["features"]])
    return {"cluster_id": cluster, "value_tier": km["tier_map"][cluster],
            "churn_probability": prob, "risk_level": str(assign_risk_level(np.array([prob]))[0]),
            "top_shap_driver_1": d1[0], "top_shap_driver_2": d2[0]}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(profile, silhouettes, metrics, global_importance, scores, db_path=DB_FILE):
    """Print the experimental results and verify the join through ``vw_customer_360``."""
    pd.set_option("display.width", 140)
    print("\n" + "=" * 72)
    print(" MINING ENGINE SUMMARY")
    print("=" * 72)
    print(f" Silhouette (k=2..6): {', '.join(f'{k}:{s:.3f}' for k, s in silhouettes.items())}")
    print("\n [Module 1] K-Means value tiers (k=3):")
    print(profile.round(3).to_string())
    print("\n [Module 2] XGBoost hold-out metrics:")
    for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "brier_score", "train_roc_auc"]:
        print(f"   {key:<14}: {metrics[key]:.4f}")
    print(f"   confusion     : {metrics['confusion_matrix']}")
    print("\n [Module 3] Top 5 global SHAP features (mean |SHAP|, log-odds):")
    for feat, val in global_importance.head(5).items():
        print(f"   {feat:<22}: {val:.4f}")
    print("\n Risk level distribution:")
    print(scores["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).to_string())
    print("\n Cross-tabulation value_tier x risk_level:")
    print(pd.crosstab(scores["value_tier"], scores["risk_level"], margins=True)
            .reindex(index=["High", "Medium", "Low", "All"], columns=["High", "Medium", "Low", "All"])
            .to_string())

    with sqlite3.connect(db_path) as conn:
        check = conn.execute(
            "SELECT COUNT(*), SUM(value_tier IS NOT NULL), SUM(churn_probability IS NOT NULL), "
            "SUM(top_shap_driver_1 IS NOT NULL) FROM vw_customer_360").fetchone()
        sample = pd.read_sql(
            "SELECT customerID, contract, tenure, monthly_charges, value_tier, churn_probability, "
            "risk_level, top_shap_driver_1, top_shap_driver_2 FROM vw_customer_360 "
            "ORDER BY churn_probability DESC LIMIT 5", conn)
    print(f"\n vw_customer_360 join check: rows={check[0]:,}, value_tier={check[1]:,}, "
          f"churn_probability={check[2]:,}, top_shap_driver_1={check[3]:,}")
    print(" Top 5 highest-risk customers (from vw_customer_360):")
    print(sample.to_string(index=False))
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_engine() -> pd.DataFrame:
    """Execute Modules 1-3, persist artifacts and scores, and print the summary."""
    df = load_customer_master()

    kmeans_pipeline, tier_map, cluster_ids, value_tiers, silhouettes, profile = train_value_tiering(df)
    xgb_pipeline, metrics = train_churn_model(df)

    X_all = df[MODEL_FEATURES]
    churn_prob = xgb_pipeline.predict_proba(X_all)[:, 1]
    driver_1, driver_2, shap_by_feature, global_importance, shap_meta = explain_churn(xgb_pipeline, X_all)
    shap_meta["customer_ids"] = df["customerID"].tolist()

    scores = pd.DataFrame({
        "customerID": df["customerID"],
        "value_tier": value_tiers,
        "cluster_id": cluster_ids.astype(int),
        "churn_probability": np.round(churn_prob.astype(float), 6),
        "risk_level": assign_risk_level(churn_prob),
        "top_shap_driver_1": driver_1,
        "top_shap_driver_2": driver_2,
        "scored_at": datetime.now().isoformat(timespec="seconds"),
    })

    metrics.update({
        "silhouette_scores": silhouettes,
        "value_tier_profile": profile.reset_index().to_dict(orient="records"),
        "global_shap_importance": {k: float(v) for k, v in global_importance.items()},
        "risk_distribution": scores["risk_level"].value_counts().to_dict(),
        "tier_risk_crosstab": pd.crosstab(scores["value_tier"], scores["risk_level"]).to_dict(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    })

    save_artifacts(kmeans_pipeline, tier_map, xgb_pipeline, shap_meta, metrics)
    persist_scores(scores)
    print_summary(profile, silhouettes, metrics, global_importance, scores)
    return scores


def main() -> int:
    try:
        run_engine()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Mining engine failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
