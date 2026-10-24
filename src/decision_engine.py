"""
Decision Support & ROI Optimization Engine (Tier 3 - Prescriptive Decision Logic).

This module translates data mining outputs (Customer Value Tier, Churn Probability,
Risk Level, and Local SHAP Drivers) into prescriptive managerial retention actions
and financial ROI projections.

Core capabilities:
    1. Parameterized Decision Simulation (``run_decision_simulation``):
       Accepts a customer DataFrame and a configurable dictionary of economic
       parameters so that the Streamlit dashboard (Step 6) can perform real-time
       What-If sensitivity analysis.
    2. 9-Cell Value x Risk Decision Matrix:
       Maps every (value_tier, risk_level) combination to an actionable strategic
       quadrant (``decision_quadrant``) and a SHAP-personalized retention offer
       (``recommended_action``).
    3. Financial & ROI Optimization:
       Computes ``annualized_revenue``, ``revenue_at_risk``, ``intervention_cost``,
       ``expected_saved_revenue``, ``expected_net_benefit``, and ``expected_roi``
       per customer and across the entire portfolio.
    4. Warehouse Persistence:
       Atomically populates ``fact_retention_actions`` in ``database/ism_dss.db``
       and exports portfolio metrics to ``models/decision_summary.json``.

Usage:
    .venv\\Scripts\\python.exe src/decision_engine.py
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration & Default Baseline Parameters
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_FILE = PROJECT_ROOT / "database" / "ism_dss.db"
MODELS_DIR = PROJECT_ROOT / "models"
DECISION_SUMMARY_FILE = MODELS_DIR / "decision_summary.json"

DEFAULT_PARAMS: Dict[str, float] = {
    "annual_horizon_months": 12,
    "discount_months": 3,
    "call_cost": 5.0,
    "email_sms_cost": 0.50,
    "high_value_discount_pct": 0.20,
    "medium_value_discount_pct": 0.10,
    "success_rate_vip_call": 0.45,
    "success_rate_auto_campaign": 0.25,
}

# Mapping of the 9 (value_tier, risk_level) cells to the 6 strategic quadrants
QUADRANT_MAP: Dict[Tuple[str, str], str] = {
    ("High", "High"): "Critical VIP Retention",
    ("High", "Medium"): "Proactive VIP Care",
    ("Medium", "High"): "Automated Retention Campaign",
    ("Medium", "Medium"): "Targeted Nudge",
    ("Low", "High"): "No Financial Intervention",
    ("Low", "Medium"): "No Financial Intervention",
    ("High", "Low"): "Champions / Nurture (No Intervention)",
    ("Medium", "Low"): "Champions / Nurture (No Intervention)",
    ("Low", "Low"): "Champions / Nurture (No Intervention)",
}

QUADRANT_ORDER = [
    "Critical VIP Retention",
    "Proactive VIP Care",
    "Automated Retention Campaign",
    "Targeted Nudge",
    "No Financial Intervention",
    "Champions / Nurture (No Intervention)",
]


# ---------------------------------------------------------------------------
# Data Access
# ---------------------------------------------------------------------------
def load_scored_customers(db_path: Path = DB_FILE) -> pd.DataFrame:
    """
    Load customer records joined with mining scores from `` database/ism_dss.db``.

    Reads ``customer_master`` joined with ``fact_mining_scores`` to ensure clean
    inputs before ``fact_retention_actions`` is populated or refreshed.
    """
    query = """
    SELECT
        m.*,
        f.value_tier, f.cluster_id, f.churn_probability, f.risk_level,
        f.top_shap_driver_1, f.top_shap_driver_2, f.scored_at
    FROM customer_master m
    JOIN fact_mining_scores f ON f.customerID = m.customerID
    ORDER BY m.customerID
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(query, conn)
    print(f"[DATA]     Loaded {len(df):,} scored customers from customer_master + fact_mining_scores")
    return df


# ---------------------------------------------------------------------------
# SHAP-Driven Action Personalization
# ---------------------------------------------------------------------------
def _parse_driver_feature(driver_str: Optional[str]) -> str:
    """Extract the business attribute name from a formatted SHAP driver string."""
    if not driver_str or not isinstance(driver_str, str):
        return ""
    return driver_str.split("=")[0].strip()


def build_recommended_action(
    value_tier: str,
    risk_level: str,
    driver_1: Optional[str],
    driver_2: Optional[str],
    high_disc_pct: float,
    med_disc_pct: float,
) -> str:
    """
    Construct a manager-ready, SHAP-personalized retention action for one customer.

    Combines the strategic playbook of the (value_tier, risk_level) cell with a
    specific tactical offer tailored to the customer's primary SHAP churn driver.
    """
    feat1 = _parse_driver_feature(driver_1)
    feat2 = _parse_driver_feature(driver_2)
    drivers = {feat1, feat2}

    high_pct_label = int(round(high_disc_pct * 100))
    med_pct_label = int(round(med_disc_pct * 100))

    # Determine tactical offer based on top SHAP churn drivers
    if "contract" in drivers:
        vip_tactic = f"{high_pct_label}% 3-mo loyalty discount on 1-year contract upgrade"
        med_tactic = f"{med_pct_label}% 3-mo discount voucher for 1-year renewal"
    elif "online_security" in drivers or "tech_support" in drivers:
        vip_tactic = f"free 6-mo OnlineSecurity & TechSupport bundle + {high_pct_label}% 3-mo discount"
        med_tactic = f"free 3-mo OnlineSecurity/TechSupport trial + {med_pct_label}% voucher"
    elif "internet_service" in drivers or "monthly_charges" in drivers:
        vip_tactic = f"Fiber QoS priority audit + {high_pct_label}% 3-mo bill credit"
        med_tactic = f"{med_pct_label}% 3-mo promotional rate lock & plan optimization"
    elif "payment_method" in drivers or "is_auto_payment" in drivers:
        vip_tactic = f"auto-pay enrollment concierge + {high_pct_label}% 3-mo loyalty credit"
        med_tactic = f"$10 auto-pay enrollment bonus + {med_pct_label}% 3-mo voucher"
    else:
        vip_tactic = f"onboarding concierge check-in + {high_pct_label}% 3-mo loyalty discount"
        med_tactic = f"{med_pct_label}% 3-mo retention voucher & onboarding guide"

    if value_tier == "High" and risk_level == "High":
        return f"1-on-1 Account Manager call + {vip_tactic}"
    if value_tier == "High" and risk_level == "Medium":
        return f"Proactive VIP check-in call + complimentary VAS bundle (OnlineSecurity/TechSupport)"
    if value_tier == "Medium" and risk_level == "High":
        return f"Automated email/SMS campaign + {med_tactic}"
    if value_tier == "Medium" and risk_level == "Medium":
        return f"Automated contract-renewal nudge + {med_tactic}"
    if value_tier == "Low" and risk_level in ("High", "Medium"):
        return "Automated satisfaction survey & self-service FAQ email ($0 financial incentive)"
    return "Standard service quality + organic cross-sell/upsell nurture ($0 intervention)"


# ---------------------------------------------------------------------------
# Parameterized Decision & ROI Simulation Engine
# ---------------------------------------------------------------------------
def run_decision_simulation(
    df: pd.DataFrame,
    params: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Execute the prescriptive retention decision matrix and ROI model.

    This function is pure and parameterized so that the Streamlit dashboard can
    invoke it dynamically during What-If sensitivity simulations.

    Economic formulation per customer i:
        * annualized_revenue_i = monthly_charges_i * annual_horizon_months
        * revenue_at_risk_i    = churn_probability_i * annualized_revenue_i
        * expected_saved_rev_i = revenue_at_risk_i * s_i
        * intervention_cost_i  = channel_cost_i + s_i * (discount_pct_i * monthly_charges_i * discount_months)
          (Contact cost is incurred upfront for every targeted customer, whereas
          the promotional discount/credit is redeemed only when the customer
          accepts the retention offer with probability s_i.)
        * expected_net_benefit = expected_saved_rev_i - intervention_cost_i
        * expected_roi (%)     = ((expected_saved_rev_i - intervention_cost_i) / intervention_cost_i) * 100
                                 (0.0 when intervention_cost_i == 0)

    Args:
        df: DataFrame containing at least ``customerID``, ``monthly_charges``,
            ``value_tier``, ``churn_probability``, ``risk_level``,
            ``top_shap_driver_1``, and ``top_shap_driver_2``.
        params: Optional dictionary overriding keys in ``DEFAULT_PARAMS``.

    Returns:
        DataFrame with all columns required for ``fact_retention_actions`` plus
        diagnostic fields (``expected_success_rate``).
    """
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    horizon = float(cfg["annual_horizon_months"])
    disc_months = float(cfg.get("discount_months", 3.0))
    call_cost = float(cfg["call_cost"])
    email_cost = float(cfg["email_sms_cost"])
    high_disc = float(cfg["high_value_discount_pct"])
    med_disc = float(cfg["medium_value_discount_pct"])
    sr_vip = float(cfg["success_rate_vip_call"])
    sr_auto = float(cfg["success_rate_auto_campaign"])

    vt = df["value_tier"].astype(str).values
    rl = df["risk_level"].astype(str).values
    monthly = df["monthly_charges"].astype(float).values
    prob = df["churn_probability"].astype(float).values
    d1 = df["top_shap_driver_1"].values if "top_shap_driver_1" in df.columns else [None] * len(df)
    d2 = df["top_shap_driver_2"].values if "top_shap_driver_2" in df.columns else [None] * len(df)

    # 1. Revenue metrics
    annualized_revenue = np.round(monthly * horizon, 2)
    revenue_at_risk = np.round(prob * annualized_revenue, 2)

    # 2. Quadrant & personalized action assignment
    quadrants = [QUADRANT_MAP[(v, r)] for v, r in zip(vt, rl)]
    actions = [
        build_recommended_action(v, r, drv1, drv2, high_disc, med_disc)
        for v, r, drv1, drv2 in zip(vt, rl, d1, d2)
    ]

    # 3. Intervention cost & success rate per cell
    cond_hh = (vt == "High") & (rl == "High")
    cond_hm = (vt == "High") & (rl == "Medium")
    cond_mh = (vt == "Medium") & (rl == "High")
    cond_mm = (vt == "Medium") & (rl == "Medium")

    # Success rate s_i
    success_rate = np.select(
        [cond_hh, cond_hm, cond_mh, cond_mm],
        [sr_vip, sr_vip * 0.80, sr_auto, sr_auto * 0.70],
        default=0.0,
    )

    # Expected intervention cost c_i (upfront contact cost + expected voucher/credit redemption)
    cost_hh = call_cost + sr_vip * (monthly * high_disc * disc_months)
    cost_hm = call_cost + (sr_vip * 0.80) * (monthly * (high_disc * 0.50) * disc_months)
    cost_mh = email_cost + sr_auto * (monthly * med_disc * disc_months)
    cost_mm = email_cost + (sr_auto * 0.70) * (monthly * (med_disc * 0.50) * disc_months)

    intervention_cost = np.round(
        np.select([cond_hh, cond_hm, cond_mh, cond_mm], [cost_hh, cost_hm, cost_mh, cost_mm], default=0.0),
        2,
    )

    # 4. Expected saved revenue, net benefit, and ROI (%)
    expected_saved_revenue = np.round(revenue_at_risk * success_rate, 2)
    expected_net_benefit = np.round(expected_saved_revenue - intervention_cost, 2)
    safe_cost = np.where(intervention_cost > 0, intervention_cost, 1.0)
    expected_roi = np.where(
        intervention_cost > 0,
        np.round(((expected_saved_revenue - intervention_cost) / safe_cost) * 100.0, 2),
        0.0,
    )

    decided_at = datetime.now().isoformat(timespec="seconds")

    result = pd.DataFrame({
        "customerID": df["customerID"].values,
        "value_tier": vt,
        "risk_level": rl,
        "monthly_charges": monthly,
        "churn_probability": prob,
        "annualized_revenue": annualized_revenue,
        "revenue_at_risk": revenue_at_risk,
        "decision_quadrant": quadrants,
        "recommended_action": actions,
        "expected_success_rate": np.round(success_rate, 4),
        "intervention_cost": intervention_cost,
        "expected_saved_revenue": expected_saved_revenue,
        "expected_net_benefit": expected_net_benefit,
        "expected_roi": expected_roi,
        "decided_at": decided_at,
    })
    return result


def summarize_decisions(sim_df: pd.DataFrame, params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Aggregate customer-level decision outputs into portfolio KPIs and breakdowns.
    """
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    total_rev = float(sim_df["annualized_revenue"].sum())
    total_rar = float(sim_df["revenue_at_risk"].sum())
    total_cost = float(sim_df["intervention_cost"].sum())
    total_saved = float(sim_df["expected_saved_revenue"].sum())
    total_net = float(sim_df["expected_net_benefit"].sum())
    portfolio_roi = float(((total_saved - total_cost) / total_cost) * 100.0) if total_cost > 0 else 0.0
    targeted_mask = sim_df["intervention_cost"] > 0

    # Breakdown by decision_quadrant
    q_grp = (
        sim_df.groupby("decision_quadrant", as_index=False)
        .agg(
            customers=("customerID", "count"),
            mean_churn_prob=("churn_probability", "mean"),
            annualized_revenue=("annualized_revenue", "sum"),
            revenue_at_risk=("revenue_at_risk", "sum"),
            intervention_cost=("intervention_cost", "sum"),
            expected_saved_revenue=("expected_saved_revenue", "sum"),
            expected_net_benefit=("expected_net_benefit", "sum"),
        )
    )
    q_safe_cost = np.where(q_grp["intervention_cost"] > 0, q_grp["intervention_cost"], 1.0)
    q_grp["quadrant_roi_pct"] = np.where(
        q_grp["intervention_cost"] > 0,
        ((q_grp["expected_saved_revenue"] - q_grp["intervention_cost"]) / q_safe_cost) * 100.0,
        0.0,
    )
    order_map = {name: idx for idx, name in enumerate(QUADRANT_ORDER)}
    q_grp["sort_key"] = q_grp["decision_quadrant"].map(lambda x: order_map.get(x, 99))
    q_grp = q_grp.sort_values("sort_key").drop(columns=["sort_key"])

    # Breakdown by all 9 (value_tier x risk_level) cells
    cell_grp = (
        sim_df.groupby(["value_tier", "risk_level", "decision_quadrant"], as_index=False)
        .agg(
            customers=("customerID", "count"),
            mean_monthly_charges=("monthly_charges", "mean"),
            mean_churn_prob=("churn_probability", "mean"),
            annualized_revenue=("annualized_revenue", "sum"),
            revenue_at_risk=("revenue_at_risk", "sum"),
            intervention_cost=("intervention_cost", "sum"),
            expected_saved_revenue=("expected_saved_revenue", "sum"),
            expected_net_benefit=("expected_net_benefit", "sum"),
        )
    )
    c_safe_cost = np.where(cell_grp["intervention_cost"] > 0, cell_grp["intervention_cost"], 1.0)
    cell_grp["cell_roi_pct"] = np.where(
        cell_grp["intervention_cost"] > 0,
        ((cell_grp["expected_saved_revenue"] - cell_grp["intervention_cost"]) / c_safe_cost) * 100.0,
        0.0,
    )
    tier_order = {"High": 0, "Medium": 1, "Low": 2}
    cell_grp["t_ord"] = cell_grp["value_tier"].map(tier_order)
    cell_grp["r_ord"] = cell_grp["risk_level"].map(tier_order)
    cell_grp = cell_grp.sort_values(["t_ord", "r_ord"]).drop(columns=["t_ord", "r_ord"])

    return {
        "parameters": cfg,
        "portfolio_kpis": {
            "total_customers": int(len(sim_df)),
            "targeted_customers": int(targeted_mask.sum()),
            "total_annualized_revenue": round(total_rev, 2),
            "total_revenue_at_risk": round(total_rar, 2),
            "revenue_at_risk_pct_of_annual": round((total_rar / total_rev) * 100.0, 2),
            "total_intervention_cost": round(total_cost, 2),
            "total_expected_saved_revenue": round(total_saved, 2),
            "total_expected_net_benefit": round(total_net, 2),
            "portfolio_roi_pct": round(portfolio_roi, 2),
        },
        "quadrant_breakdown": q_grp.round(2).to_dict(orient="records"),
        "matrix_9_cells": cell_grp.round(2).to_dict(orient="records"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Warehouse Persistence
# ---------------------------------------------------------------------------
FACT_RETENTION_COLS = [
    "customerID",
    "annualized_revenue",
    "revenue_at_risk",
    "decision_quadrant",
    "recommended_action",
    "intervention_cost",
    "expected_saved_revenue",
    "expected_net_benefit",
    "expected_roi",
    "uplift_score",
    "causal_segment",
    "is_do_not_disturb",
    "decided_at",
    "action_status",
    "manager_notes",
    "updated_at",
]

ACTION_STATUS_DOMAIN = ("Pending", "Contacted", "Retained", "Churned")
CAUSAL_SEGMENT_DOMAIN = ("Persuadable", "Sleeping Dog", "Lost Cause", "Sure Thing")


def ensure_crm_columns(conn: sqlite3.Connection) -> None:
    """
    Ensure ``fact_retention_actions`` and ``vw_customer_360`` include the
    Causal Uplift attributes (`uplift_score`, `causal_segment`, `is_do_not_disturb`)
    and Closed-Loop CRM tracking attributes (`action_status`, `manager_notes`, `updated_at`).
    """
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(fact_retention_actions);").fetchall()}
    altered = False
    if "uplift_score" not in existing_cols:
        conn.execute("ALTER TABLE fact_retention_actions ADD COLUMN uplift_score REAL NOT NULL DEFAULT 0.0;")
        altered = True
    if "causal_segment" not in existing_cols:
        conn.execute(
            "ALTER TABLE fact_retention_actions ADD COLUMN causal_segment TEXT NOT NULL DEFAULT 'Sure Thing' "
            "CHECK (causal_segment IN ('Persuadable', 'Sleeping Dog', 'Lost Cause', 'Sure Thing'));"
        )
        altered = True
    if "is_do_not_disturb" not in existing_cols:
        conn.execute(
            "ALTER TABLE fact_retention_actions ADD COLUMN is_do_not_disturb INTEGER NOT NULL DEFAULT 0 "
            "CHECK (is_do_not_disturb IN (0, 1));"
        )
        altered = True
    if "action_status" not in existing_cols:
        conn.execute(
            "ALTER TABLE fact_retention_actions ADD COLUMN action_status TEXT NOT NULL DEFAULT 'Pending' "
            "CHECK (action_status IN ('Pending', 'Contacted', 'Retained', 'Churned'));"
        )
        altered = True
    if "manager_notes" not in existing_cols:
        conn.execute("ALTER TABLE fact_retention_actions ADD COLUMN manager_notes TEXT NOT NULL DEFAULT '';")
        altered = True
    if "updated_at" not in existing_cols:
        conn.execute("ALTER TABLE fact_retention_actions ADD COLUMN updated_at TEXT;")
        altered = True

    view_cols = {row[1] for row in conn.execute("PRAGMA table_info(vw_customer_360);").fetchall()}
    if altered or ("action_status" not in view_cols) or ("causal_segment" not in view_cols):
        conn.executescript(
            """
            DROP VIEW IF EXISTS vw_customer_360;
            CREATE VIEW vw_customer_360 AS
            SELECT
                m.*,
                f.value_tier, f.cluster_id, f.churn_probability, f.risk_level,
                f.top_shap_driver_1, f.top_shap_driver_2, f.scored_at,
                r.annualized_revenue, r.revenue_at_risk, r.decision_quadrant, r.recommended_action,
                r.intervention_cost, r.expected_saved_revenue, r.expected_net_benefit, r.expected_roi,
                r.uplift_score, r.causal_segment, r.is_do_not_disturb,
                r.action_status, r.manager_notes, r.updated_at
            FROM customer_master m
            LEFT JOIN fact_mining_scores     f ON f.customerID = m.customerID
            LEFT JOIN fact_retention_actions r ON r.customerID = m.customerID;
            """
        )


def persist_retention_actions(sim_df: pd.DataFrame, db_path: Path = DB_FILE) -> None:
    """
    Atomically write all 7,043 retention decision rows to ``fact_retention_actions``.

    Preserves any existing non-default ``uplift_score``, ``causal_segment``,
    ``is_do_not_disturb``, ``action_status``, ``manager_notes``, and ``updated_at``
    values across re-runs so that causal uplift and closed-loop CRM tracking are retained.
    """
    df = sim_df.copy()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        ensure_crm_columns(conn)
        existing_crm = pd.read_sql(
            "SELECT customerID, uplift_score, causal_segment, is_do_not_disturb, "
            "action_status, manager_notes, updated_at FROM fact_retention_actions",
            conn,
        )
        if not existing_crm.empty:
            crm_map = existing_crm.set_index("customerID").to_dict(orient="index")
        else:
            crm_map = {}

        now_ts = datetime.now().isoformat(timespec="seconds")
        if "uplift_score" not in df.columns:
            df["uplift_score"] = [
                float(crm_map.get(cid, {}).get("uplift_score") or 0.0) for cid in df["customerID"]
            ]
        if "causal_segment" not in df.columns:
            df["causal_segment"] = [
                crm_map.get(cid, {}).get("causal_segment") or "Sure Thing" for cid in df["customerID"]
            ]
        if "is_do_not_disturb" not in df.columns:
            df["is_do_not_disturb"] = [
                int(crm_map.get(cid, {}).get("is_do_not_disturb") or 0) for cid in df["customerID"]
            ]
        df["action_status"] = [
            crm_map.get(cid, {}).get("action_status") or "Pending" for cid in df["customerID"]
        ]
        df["manager_notes"] = [
            crm_map.get(cid, {}).get("manager_notes") or "" for cid in df["customerID"]
        ]
        df["updated_at"] = [
            crm_map.get(cid, {}).get("updated_at") or now_ts for cid in df["customerID"]
        ]

        rows = list(df[FACT_RETENTION_COLS].itertuples(index=False, name=None))
        with conn:
            conn.execute("DELETE FROM fact_retention_actions;")
            conn.executemany(
                f"INSERT INTO fact_retention_actions ({', '.join(FACT_RETENTION_COLS)}) "
                f"VALUES ({', '.join('?' * len(FACT_RETENTION_COLS))})",
                rows,
            )
    finally:
        conn.close()
    print(f"[PERSIST]   Wrote {len(rows):,} rows to fact_retention_actions in {db_path.relative_to(PROJECT_ROOT)}")


def update_customer_crm_status(
    customer_id: str,
    new_status: str,
    manager_notes: str = "",
    db_path: Path = DB_FILE,
) -> Dict[str, Any]:
    """
    Update the closed-loop CRM action status and manager notes for a customer in SQLite.
    """
    if new_status not in ACTION_STATUS_DOMAIN:
        raise ValueError(f"Invalid action_status '{new_status}'. Expected one of {ACTION_STATUS_DOMAIN}.")
    ts = datetime.now().isoformat(timespec="seconds")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        ensure_crm_columns(conn)
        with conn:
            cur = conn.execute(
                "UPDATE fact_retention_actions "
                "SET action_status = ?, manager_notes = ?, updated_at = ? "
                "WHERE customerID = ?",
                (new_status, manager_notes.strip(), ts, customer_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"customerID '{customer_id}' not found in fact_retention_actions.")
    finally:
        conn.close()
    return {
        "customerID": customer_id,
        "action_status": new_status,
        "manager_notes": manager_notes.strip(),
        "updated_at": ts,
    }


def save_decision_summary(summary: Dict[str, Any], out_path: Path = DECISION_SUMMARY_FILE) -> None:
    """Save the portfolio decision summary JSON to ``models/decision_summary.json``."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[ARTIFACT]  Saved portfolio decision summary to {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Verification & Reporting
# ---------------------------------------------------------------------------
def verify_and_report(summary: Dict[str, Any], db_path: Path = DB_FILE) -> None:
    """Verify ``vw_customer_360`` completeness and print a managerial summary."""
    with sqlite3.connect(db_path) as conn:
        view_df = pd.read_sql("SELECT * FROM vw_customer_360", conn)

    # Check non-null counts across vw_customer_360
    null_counts = view_df.isna().sum()
    non_driver_nulls = null_counts.drop(
        labels=["top_shap_driver_1", "top_shap_driver_2", "updated_at"], errors="ignore"
    )
    assert int(non_driver_nulls.sum()) == 0, f"Unexpected NULLs in vw_customer_360: {non_driver_nulls[non_driver_nulls > 0]}"
    assert len(view_df) == 7043, f"Expected 7,043 rows in vw_customer_360, got {len(view_df)}"

    kpis = summary["portfolio_kpis"]
    q_df = pd.DataFrame(summary["quadrant_breakdown"])
    c_df = pd.DataFrame(summary["matrix_9_cells"])

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 84)
    print(" DECISION ENGINE & ROI OPTIMIZATION SUMMARY")
    print("=" * 84)
    print(f" Total Customers                : {kpis['total_customers']:,}")
    print(f" Targeted Customers (Cost > $0) : {kpis['targeted_customers']:,}")
    print(f" Total Annualized Revenue       : ${kpis['total_annualized_revenue']:>12,.2f}")
    print(f" Total Revenue at Risk (Annual) : ${kpis['total_revenue_at_risk']:>12,.2f} ({kpis['revenue_at_risk_pct_of_annual']:.2f}%)")
    print(f" Total Intervention Budget      : ${kpis['total_intervention_cost']:>12,.2f}")
    print(f" Expected Saved Revenue         : ${kpis['total_expected_saved_revenue']:>12,.2f}")
    print(f" Expected Net Benefit           : ${kpis['total_expected_net_benefit']:>12,.2f}")
    print(f" Portfolio Expected ROI         : {kpis['portfolio_roi_pct']:>11.2f}%")
    print("-" * 84)
    print("\n [1] Breakdown by Strategic Decision Quadrant:")
    print(q_df.to_string(index=False))
    print("\n [2] 9-Cell Value Tier x Risk Level Matrix Breakdown:")
    print(c_df.to_string(index=False))
    print("-" * 84)
    print(f" vw_customer_360 verification   : {len(view_df):,} rows x {view_df.shape[1]} columns | "
          f"0 NULLs in all core/decision columns")
    print(" Sample Top-3 Priority Rescue Records (from vw_customer_360):")
    sample = (
        view_df[view_df["decision_quadrant"] == "Critical VIP Retention"]
        .sort_values("revenue_at_risk", ascending=False)
        [["customerID", "value_tier", "risk_level", "churn_probability",
          "annualized_revenue", "revenue_at_risk", "intervention_cost",
          "expected_saved_revenue", "expected_roi", "recommended_action"]]
        .head(3)
    )
    print(sample.to_string(index=False))
    print("=" * 84)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def run_engine(params: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the full Decision Support & ROI Optimization Engine."""
    df = load_scored_customers()
    sim_df = run_decision_simulation(df, params=params)
    summary = summarize_decisions(sim_df, params=params)
    persist_retention_actions(sim_df)
    save_decision_summary(summary)
    verify_and_report(summary)
    return sim_df, summary


def main() -> int:
    try:
        run_engine()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Decision engine failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
