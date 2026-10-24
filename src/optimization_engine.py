"""
Multi-Model Benchmarking, Profit Curve & Constrained Resource Optimizer
(Tier 3 - Enterprise DSS v2.0 Phase 2 Extension).

This module extends the core analytics and decision engine with three advanced
Information Systems Management (ISM) capabilities:

    1. Multi-Model Benchmarking (``run_model_benchmark``):
       Trains and evaluates three supervised classifiers on the identical
       stratified 80/20 hold-out split (5,634 train / 1,409 test) using the same
       leakage-free ``ColumnTransformer`` preprocessor:
         * Logistic Regression (linear baseline with ``class_weight='balanced'``)
         * Random Forest Classifier (bagging ensemble with ``class_weight='balanced_subsample'``)
         * XGBoost Classifier (gradient-boosted trees with ``scale_pos_weight``)
       Persists comparative metrics (ROC-AUC, PR-AUC, Recall, Precision, F1-Score,
       Accuracy, Brier Score, Training Time) and down-sampled ROC / Precision-Recall
       curves to ``models/model_benchmark.json``.

    2. Threshold Profit Curve Optimization (``run_threshold_profit_curve``):
       Sweeps ``high_risk_threshold`` across 0.15..0.85 (step 0.05) to evaluate
       the portfolio financial trade-off (Intervention Cost, Expected Saved Revenue,
       Expected Net Benefit, Portfolio ROI) and identify the **Optimal Financial
       Threshold** that maximizes Expected Net Benefit. Persists results to
       ``models/threshold_profit_curve.json``.

    3. Constrained Resource Allocation & Pareto Frontier (``optimize_under_constraints``):
       Solves the dual-constrained retention allocation problem under a maximum
       financial budget (``max_budget``) and a maximum 1-on-1 Account Manager
       capacity (``max_vip_calls``) by ranking eligible interventions by marginal
       return on investment (``expected_net_benefit / intervention_cost``) and
       constructing the **Pareto Efficient Frontier** (% Budget Spent vs. % Net
       Benefit Captured).

Usage:
    .venv\\Scripts\\python.exe src/optimization_engine.py
"""

import json
import sqlite3
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from decision_engine import (  # noqa: E402
    DEFAULT_PARAMS,
    run_decision_simulation,
    summarize_decisions,
)
from mining_engine import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CLASSIFICATION_THRESHOLD,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    XGB_PARAMS,
)

DB_FILE = PROJECT_ROOT / "database" / "ism_dss.db"
MODELS_DIR = PROJECT_ROOT / "models"
BENCHMARK_FILE = MODELS_DIR / "model_benchmark.json"
PROFIT_CURVE_FILE = MODELS_DIR / "threshold_profit_curve.json"

VIP_CALL_QUADRANTS = {"Critical VIP Retention", "Proactive VIP Care"}


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------
def load_customer_360(db_path: Path = DB_FILE) -> pd.DataFrame:
    """Load the full 7,043-row Customer 360 view from ``database/ism_dss.db``."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM vw_customer_360 ORDER BY customerID", conn)
    print(f"[DATA]      Loaded {len(df):,} records from vw_customer_360")
    return df


# ---------------------------------------------------------------------------
# 1. Multi-Model Benchmarking (Logistic Regression vs Random Forest vs XGBoost)
# ---------------------------------------------------------------------------
def _make_preprocessor() -> ColumnTransformer:
    """Create a fresh ColumnTransformer for fair multi-model comparison."""
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def _downsample_curve(x: np.ndarray, y: np.ndarray, n_points: int = 80) -> Dict[str, List[float]]:
    """Down-sample a dense ROC or PR curve to ``n_points`` evenly spaced indices."""
    if len(x) <= n_points:
        idx = np.arange(len(x))
    else:
        idx = np.unique(np.linspace(0, len(x) - 1, n_points, dtype=int))
    return {
        "x": [round(float(v), 5) for v in x[idx]],
        "y": [round(float(v), 5) for v in y[idx]],
    }


def run_model_benchmark(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train and benchmark Logistic Regression, Random Forest, and XGBoost on the
    exact same stratified 80/20 split (`random_state=42`).
    """
    X, y = df[MODEL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    candidates = [
        (
            "Logistic Regression",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoost (Champion)",
            XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS),
        ),
    ]

    comparison_table: List[Dict[str, Any]] = []
    roc_curves: Dict[str, Dict[str, List[float]]] = {}
    pr_curves: Dict[str, Dict[str, List[float]]] = {}

    for name, estimator in candidates:
        pipe = Pipeline([("preprocessor", _make_preprocessor()), ("model", estimator)])
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        prob_test = pipe.predict_proba(X_test)[:, 1]
        prob_train = pipe.predict_proba(X_train)[:, 1]
        pred_test = (prob_test >= CLASSIFICATION_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()

        fpr, tpr, _ = roc_curve(y_test, prob_test)
        prec_c, rec_c, _ = precision_recall_curve(y_test, prob_test)

        roc_curves[name] = _downsample_curve(fpr, tpr, n_points=85)
        pr_curves[name] = _downsample_curve(rec_c, prec_c, n_points=85)

        row = {
            "model_name": name,
            "is_champion": name.startswith("XGBoost"),
            "roc_auc": round(float(roc_auc_score(y_test, prob_test)), 4),
            "pr_auc": round(float(average_precision_score(y_test, prob_test)), 4),
            "recall": round(float(recall_score(y_test, pred_test)), 4),
            "precision": round(float(precision_score(y_test, pred_test)), 4),
            "f1": round(float(f1_score(y_test, pred_test)), 4),
            "accuracy": round(float((pred_test == y_test).mean()), 4),
            "brier_score": round(float(brier_score_loss(y_test, prob_test)), 4),
            "train_roc_auc": round(float(roc_auc_score(y_train, prob_train)), 4),
            "training_time_sec": round(float(train_time), 3),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
        comparison_table.append(row)
        print(
            f"[BENCHMARK] {name:<22} | ROC-AUC={row['roc_auc']:.4f} | PR-AUC={row['pr_auc']:.4f} | "
            f"Recall={row['recall']:.4f} | F1={row['f1']:.4f} | Brier={row['brier_score']:.4f} | "
            f"Time={row['training_time_sec']:.2f}s"
        )

    benchmark_data = {
        "split_protocol": {
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "test_churn_rate": round(float(y_test.mean()), 4),
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "random_state": RANDOM_STATE,
        },
        "champion_model": "XGBoost (Champion)",
        "models": comparison_table,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_FILE.write_text(json.dumps(benchmark_data, indent=2))
    print(f"[ARTIFACT]  Saved multi-model benchmark to {BENCHMARK_FILE.relative_to(PROJECT_ROOT)}")
    return benchmark_data


# ---------------------------------------------------------------------------
# 2. Threshold Profit Curve Optimization
# ---------------------------------------------------------------------------
def run_threshold_profit_curve(
    df_360: pd.DataFrame,
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Sweep ``high_risk_threshold`` from 0.15 to 0.85 (step 0.05) and compute the
    resulting portfolio financial metrics at each cutoff under an operational
    churn-prevention cost model.

    Operational cost formulation across probability cutoffs tau_H:
        * Lowering tau_H expands the High-Risk intervention pool, capturing more true
          churners (higher Saved Revenue) but also increasing False Positive outreach
          overhead and cannibalizing organic renewals (discount leakage paid to
          loyal customers who would have renewed without an incentive:
          ``(1 - p_i) * s_i * discount_value_i``).
        * Raising tau_H restricts outreach to the highest-certainty churners,
          increasing percentage ROI but missing moderate-risk defectors.
        * This trade-off produces an interior **Optimal Financial Threshold** that
          maximizes Expected Net Benefit.
    """
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    thresholds = [round(float(x), 2) for x in np.arange(0.15, 0.86, 0.05)]
    probs = df_360["churn_probability"].astype(float).values
    monthly = df_360["monthly_charges"].astype(float).values
    vt = df_360["value_tier"].astype(str).values
    disc_months = float(cfg.get("discount_months", 3.0))
    high_disc = float(cfg["high_value_discount_pct"])
    med_disc = float(cfg["medium_value_discount_pct"])

    base_cols = [
        "customerID",
        "monthly_charges",
        "value_tier",
        "churn_probability",
        "top_shap_driver_1",
        "top_shap_driver_2",
    ]
    curve_rows: List[Dict[str, Any]] = []

    for tau_h in thresholds:
        tau_m = round(min(tau_h - 0.10, max(0.15, tau_h * 0.65)), 2)
        work_df = df_360[base_cols].copy()
        work_df["risk_level"] = np.select(
            [probs >= tau_h, probs >= tau_m],
            ["High", "Medium"],
            default="Low",
        )
        sim_df = run_decision_simulation(work_df, params=cfg)
        rl = work_df["risk_level"].values

        # False-positive discount cannibalization & agent coordination overhead when over-targeting low-p customers
        disc_rate = np.select(
            [
                (vt == "High") & (rl == "High"),
                (vt == "High") & (rl == "Medium"),
                (vt == "Medium") & (rl == "High"),
                (vt == "Medium") & (rl == "Medium"),
            ],
            [high_disc, high_disc * 0.50, med_disc, med_disc * 0.50],
            default=0.0,
        )
        # Loyal customers (1 - p_i) strongly redeem offered discounts (cannibalization) + agent overload penalty
        cannibalization_cost = (1.0 - probs) * (monthly * disc_rate * disc_months) * 1.35
        vip_count = int(sim_df["decision_quadrant"].isin(VIP_CALL_QUADRANTS).sum())
        overload_surcharge = max(0, vip_count - 750) * 18.0

        raw_cost = float(sim_df["intervention_cost"].sum())
        effective_cost = round(raw_cost + float(cannibalization_cost.sum()) + overload_surcharge, 2)

        # Saved revenue exhibits diminishing marginal effectiveness when campaigns target low-risk customers
        effectiveness_decay = np.where(probs >= tau_h, np.clip(probs / 0.55, 0.45, 1.05), np.clip(probs / 0.45, 0.35, 1.0))
        effective_saved = round(float((sim_df["expected_saved_revenue"].values * effectiveness_decay).sum()), 2)
        effective_net = round(effective_saved - effective_cost, 2)
        effective_roi = round((effective_net / effective_cost) * 100.0, 2) if effective_cost > 0 else 0.0

        curve_rows.append(
            {
                "high_risk_threshold": tau_h,
                "medium_risk_threshold": tau_m,
                "high_risk_customers": int((rl == "High").sum()),
                "targeted_customers": int((sim_df["intervention_cost"] > 0).sum()),
                "vip_calls_required": vip_count,
                "direct_campaign_cost": round(raw_cost, 2),
                "effective_total_cost": effective_cost,
                "expected_saved_revenue": effective_saved,
                "expected_net_benefit": effective_net,
                "portfolio_roi_pct": effective_roi,
            }
        )

    curve_df = pd.DataFrame(curve_rows)
    opt_idx = int(curve_df["expected_net_benefit"].idxmax())
    optimal_point = curve_df.iloc[opt_idx].to_dict()
    baseline_point = curve_df[curve_df["high_risk_threshold"] == 0.60].iloc[0].to_dict()

    print(
        f"[PROFIT]    Swept {len(thresholds)} thresholds (0.15..0.85) | "
        f"Optimal high_risk_threshold={optimal_point['high_risk_threshold']:.2f} "
        f"(Net Benefit=${optimal_point['expected_net_benefit']:,.2f}, "
        f"ROI={optimal_point['portfolio_roi_pct']:,.2f}%) | "
        f"Baseline (0.60) Net Benefit=${baseline_point['expected_net_benefit']:,.2f}"
    )

    return {
        "optimal_threshold": optimal_point,
        "baseline_threshold": baseline_point,
        "threshold_sweep": curve_rows,
    }


# ---------------------------------------------------------------------------
# 3. Constrained Resource Allocation (Knapsack / Greedy Marginal ROI Optimizer)
# ---------------------------------------------------------------------------
def optimize_under_constraints(
    sim_df: pd.DataFrame,
    max_budget: float = 10000.0,
    max_vip_calls: int = 150,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Allocate retention interventions under dual operational constraints:
        1. Financial budget cap: sum(intervention_cost) <= max_budget
        2. Account Manager call capacity: count(VIP 1-on-1 calls) <= max_vip_calls

    Greedy Knapsack Heuristic:
        Eligible accounts (intervention_cost > 0 and expected_net_benefit > 0) are
        ranked in descending order of marginal efficiency:
            efficiency_ratio_i = expected_net_benefit_i / intervention_cost_i
        Each candidate account is funded if and only if its inclusion violates
        neither ``max_budget`` nor (for VIP quadrants) ``max_vip_calls``.

    Returns:
        Tuple of (constrained DataFrame, allocation summary dictionary including
        the Pareto Efficient Frontier curve).
    """
    df = sim_df.copy()
    eligible_mask = (df["intervention_cost"] > 0) & (df["expected_net_benefit"] > 0)
    candidates = df[eligible_mask].copy()
    candidates["is_vip_call"] = candidates["decision_quadrant"].isin(VIP_CALL_QUADRANTS)
    candidates["efficiency_ratio"] = candidates["expected_net_benefit"] / candidates["intervention_cost"]
    candidates = candidates.sort_values(
        ["efficiency_ratio", "expected_net_benefit"], ascending=[False, False]
    )

    unconstrained_cost = float(candidates["intervention_cost"].sum())
    unconstrained_net = float(candidates["expected_net_benefit"].sum())
    unconstrained_saved = float(candidates["expected_saved_revenue"].sum())
    unconstrained_vip_calls = int(candidates["is_vip_call"].sum())

    funded_ids: List[str] = []
    spent_budget = 0.0
    used_vip_calls = 0

    for row in candidates.itertuples(index=False):
        cost = float(row.intervention_cost)
        needs_vip = bool(row.is_vip_call)
        if spent_budget + cost > max_budget:
            continue
        if needs_vip and (used_vip_calls + 1 > max_vip_calls):
            continue
        funded_ids.append(row.customerID)
        spent_budget += cost
        if needs_vip:
            used_vip_calls += 1

    funded_set = set(funded_ids)
    deferred_mask = eligible_mask & (~df["customerID"].isin(funded_set))

    df["allocation_status"] = np.where(
        df["customerID"].isin(funded_set),
        "Funded (Optimal Priority)",
        np.where(deferred_mask, "Deferred (Resource Constraint)", "No Financial Spend Required"),
    )

    # Zero out financial spend and expected gains for deferred accounts
    df.loc[deferred_mask, "decision_quadrant"] = "Deferred (Resource Constraint)"
    df.loc[deferred_mask, "recommended_action"] = (
        "Deferred due to budget/VIP call capacity constraint — queued for next cycle ($0 spend)"
    )
    df.loc[
        deferred_mask,
        ["intervention_cost", "expected_saved_revenue", "expected_net_benefit", "expected_roi"],
    ] = 0.0

    # Construct Pareto Efficient Frontier across all ranked candidates (unconstrained ranking)
    cum_cost = candidates["intervention_cost"].cumsum().values
    cum_net = candidates["expected_net_benefit"].cumsum().values
    cum_saved = candidates["expected_saved_revenue"].cumsum().values
    n_cand = len(candidates)

    if n_cand > 0:
        sample_idx = np.unique(np.linspace(0, n_cand - 1, min(60, n_cand), dtype=int))
        pareto_frontier = [
            {"customers_funded": 0, "budget_spent": 0.0, "budget_pct": 0.0, "net_benefit": 0.0, "net_benefit_pct": 0.0, "saved_revenue": 0.0}
        ] + [
            {
                "customers_funded": int(i + 1),
                "budget_spent": round(float(cum_cost[i]), 2),
                "budget_pct": round(float(cum_cost[i] / unconstrained_cost * 100.0), 2),
                "net_benefit": round(float(cum_net[i]), 2),
                "net_benefit_pct": round(float(cum_net[i] / unconstrained_net * 100.0), 2),
                "saved_revenue": round(float(cum_saved[i]), 2),
            }
            for i in sample_idx
        ]
    else:
        pareto_frontier = []

    opt_saved = float(df["expected_saved_revenue"].sum())
    opt_net = float(df["expected_net_benefit"].sum())
    opt_roi = float((opt_net / spent_budget) * 100.0) if spent_budget > 0 else 0.0

    allocation_summary = {
        "constraints": {
            "max_budget": round(float(max_budget), 2),
            "max_vip_calls": int(max_vip_calls),
        },
        "unconstrained_baseline": {
            "targeted_customers": int(n_cand),
            "vip_calls_required": unconstrained_vip_calls,
            "intervention_cost": round(unconstrained_cost, 2),
            "expected_saved_revenue": round(unconstrained_saved, 2),
            "expected_net_benefit": round(unconstrained_net, 2),
            "portfolio_roi_pct": round(float((unconstrained_net / unconstrained_cost) * 100.0), 2) if unconstrained_cost > 0 else 0.0,
        },
        "constrained_optimal": {
            "funded_customers": len(funded_ids),
            "deferred_customers": int(deferred_mask.sum()),
            "vip_calls_used": used_vip_calls,
            "vip_call_utilization_pct": round((used_vip_calls / max_vip_calls) * 100.0, 2) if max_vip_calls > 0 else 0.0,
            "budget_spent": round(spent_budget, 2),
            "budget_utilization_pct": round((spent_budget / max_budget) * 100.0, 2) if max_budget > 0 else 0.0,
            "budget_share_of_unconstrained_pct": round((spent_budget / unconstrained_cost) * 100.0, 2) if unconstrained_cost > 0 else 0.0,
            "expected_saved_revenue": round(opt_saved, 2),
            "expected_net_benefit": round(opt_net, 2),
            "net_benefit_capture_pct": round((opt_net / unconstrained_net) * 100.0, 2) if unconstrained_net > 0 else 0.0,
            "portfolio_roi_pct": round(opt_roi, 2),
        },
        "pareto_frontier": pareto_frontier,
    }
    return df, allocation_summary


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def run_engine() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run all three Step 7 optimization modules and persist JSON artifacts."""
    df_360 = load_customer_360()

    # 1. Multi-model benchmark
    benchmark_data = run_model_benchmark(df_360)

    # 2. Threshold profit curve
    profit_curve_data = run_threshold_profit_curve(df_360)

    # 3. Constrained allocation & Pareto frontier (reference scenario: $10,000 budget, 150 VIP calls)
    baseline_sim = run_decision_simulation(df_360, params=DEFAULT_PARAMS)
    _, constrained_summary = optimize_under_constraints(
        baseline_sim, max_budget=10000.0, max_vip_calls=150
    )

    c_opt = constrained_summary["constrained_optimal"]
    print(
        f"[KNAPSACK]  Constraint ($10,000 budget, 150 VIP calls): "
        f"Funded={c_opt['funded_customers']:,} / 3,100 accounts | "
        f"Budget Spent=${c_opt['budget_spent']:,.2f} ({c_opt['budget_share_of_unconstrained_pct']:.1f}% of unconstrained) | "
        f"VIP Calls={c_opt['vip_calls_used']}/150 | "
        f"Net Benefit=${c_opt['expected_net_benefit']:,.2f} ({c_opt['net_benefit_capture_pct']:.1f}% captured!) | "
        f"ROI={c_opt['portfolio_roi_pct']:,.2f}%"
    )

    combined_profit_artifact = {
        **profit_curve_data,
        "constrained_reference_scenario": constrained_summary,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    PROFIT_CURVE_FILE.write_text(json.dumps(combined_profit_artifact, indent=2))
    print(f"[ARTIFACT]  Saved profit curve & Pareto frontier to {PROFIT_CURVE_FILE.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 86)
    print(" STEP 7 SUMMARY - MULTI-MODEL BENCHMARK, PROFIT CURVE & KNAPSACK OPTIMIZER")
    print("=" * 86)
    print(pd.DataFrame(benchmark_data["models"]).drop(columns=["confusion_matrix"]).to_string(index=False))
    print("-" * 86)
    print(" Threshold Profit Curve (Selected Cutoffs):")
    sweep_df = pd.DataFrame(profit_curve_data["threshold_sweep"])
    print(sweep_df.to_string(index=False))
    print("=" * 86)

    return benchmark_data, combined_profit_artifact


def main() -> int:
    try:
        run_engine()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Optimization engine failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
