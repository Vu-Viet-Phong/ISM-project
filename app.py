"""
Streamlit Decision Support Dashboard (Tier 4 - Presentation Layer).

Intelligent Customer Retention & Value Optimization System using Data Mining
Techniques (K-Means + XGBoost + SHAP + Prescriptive ROI Decision Engine).

Dashboard architecture (4 Tabs):
    Tab 1 - Executive KPI & Portfolio Overview
    Tab 2 - Interactive What-If ROI & Budget Simulator
    Tab 3 - Operational Action Queue & Customer 360 Inspector
    Tab 4 - Real-Time New Customer Predictor & Decision Advisor

Usage:
    .venv\\Scripts\\streamlit.exe run app.py
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure local modules in src/ are importable
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from decision_engine import (  # noqa: E402
    DEFAULT_PARAMS,
    QUADRANT_ORDER,
    run_decision_simulation,
    summarize_decisions,
)
from etl_pipeline import (  # noqa: E402
    AUTO_PAYMENT_METHODS,
    TENURE_BINS,
    TENURE_LABELS,
    VALUE_ADDED_SERVICES,
)
from mining_engine import explain_churn, score_customer  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & Page Configuration
# ---------------------------------------------------------------------------
DB_FILE = PROJECT_ROOT / "database" / "ism_dss.db"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_FILE = MODELS_DIR / "model_metrics.json"
DECISION_SUMMARY_FILE = MODELS_DIR / "decision_summary.json"
KMEANS_FILE = MODELS_DIR / "kmeans_value_pipeline.joblib"
XGB_FILE = MODELS_DIR / "xgb_churn_pipeline.joblib"
SHAP_FILE = MODELS_DIR / "shap_metadata.joblib"

st.set_page_config(
    page_title="ISM Customer Retention & Value Optimization DSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for executive cards and badges
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Data & Artifact Loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_customer_360() -> pd.DataFrame:
    """Load the full 7,043-row Customer 360 analytical view from SQLite."""
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql("SELECT * FROM vw_customer_360 ORDER BY customerID", conn)
    return df


@st.cache_data(show_spinner=False)
def load_json_summaries() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load model evaluation metrics and baseline decision summary JSON files."""
    metrics = json.loads(METRICS_FILE.read_text())
    decision_summary = json.loads(DECISION_SUMMARY_FILE.read_text())
    return metrics, decision_summary


@st.cache_resource(show_spinner=False)
def load_ml_artifacts() -> Dict[str, Any]:
    """Load serialized K-Means, XGBoost, and SHAP metadata artifacts."""
    return {
        "kmeans": joblib.load(KMEANS_FILE),
        "xgb": joblib.load(XGB_FILE),
        "shap": joblib.load(SHAP_FILE),
    }


def apply_budget_cap(sim_df: pd.DataFrame, budget_cap: float) -> pd.DataFrame:
    """
    Enforce an optional campaign budget constraint by prioritizing customers
    with the highest expected ROI (and expected net benefit as tie-breaker).
    """
    df = sim_df.copy()
    targeted = df[df["intervention_cost"] > 0].sort_values(
        ["expected_roi", "expected_net_benefit"], ascending=[False, False]
    )
    cum_cost = targeted["intervention_cost"].cumsum()
    funded_ids = set(targeted.loc[cum_cost <= budget_cap, "customerID"])

    unfunded_mask = (df["intervention_cost"] > 0) & (~df["customerID"].isin(funded_ids))
    df.loc[unfunded_mask, "decision_quadrant"] = "Deferred (Budget Cap Exceeded)"
    df.loc[unfunded_mask, "recommended_action"] = (
        "Deferred due to campaign budget cap — queue for next billing cycle ($0 spend)"
    )
    df.loc[unfunded_mask, ["intervention_cost", "expected_saved_revenue", "expected_net_benefit", "expected_roi"]] = 0.0
    return df


# ---------------------------------------------------------------------------
# Sidebar — System Architecture & Global Summary
# ---------------------------------------------------------------------------
df_360 = load_customer_360()
model_metrics, baseline_summary = load_json_summaries()
artifacts = load_ml_artifacts()
base_kpis = baseline_summary["portfolio_kpis"]

with st.sidebar:
    st.markdown("## 🧠 ISM Retention DSS")
    st.caption(
        "**4-Tier Decision Support System**\n"
        "K-Means Value Tiering · XGBoost Churn Prediction · SHAP Explainability · ROI Optimization"
    )
    st.divider()
    st.markdown("### 📦 Warehouse Status")
    st.markdown(
        f"- **Database:** `database/ism_dss.db`\n"
        f"- **Active View:** `vw_customer_360`\n"
        f"- **Total Subscribers:** `{len(df_360):,}`\n"
        f"- **Observed Churn Rate:** `{df_360['churn_binary'].mean() * 100:.2f}%`\n"
        f"- **XGBoost Test ROC-AUC:** `{model_metrics['roc_auc']:.4f}`\n"
        f"- **XGBoost Test Recall:** `{model_metrics['recall'] * 100:.1f}%`"
    )
    st.divider()
    st.markdown("### 💼 Baseline Portfolio Snapshot")
    st.markdown(
        f"- **Annualized Revenue:** `${base_kpis['total_annualized_revenue']:,.0f}`\n"
        f"- **Revenue at Risk:** `${base_kpis['total_revenue_at_risk']:,.0f}`\n"
        f"- **Targeted Accounts:** `{base_kpis['targeted_customers']:,}`\n"
        f"- **Campaign Budget:** `${base_kpis['total_intervention_cost']:,.0f}`\n"
        f"- **Expected Net Benefit:** `${base_kpis['total_expected_net_benefit']:,.0f}`\n"
        f"- **Portfolio ROI:** `{base_kpis['portfolio_roi_pct']:,.1f}%`"
    )
    st.divider()
    st.caption("International University — VNU-HCMC\nInformation Systems Management Project")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Intelligent Customer Retention & Value Optimization DSS")
st.markdown(
    "**Enterprise Decision Support System** integrating **Unsupervised Value Segmentation (K-Means)**, "
    "**Predictive Churn Modeling (XGBoost)**, **Explainable AI (SHAP)**, and **Prescriptive ROI Optimization**."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Tab 1: Executive KPI & Portfolio Overview",
        "🎛️ Tab 2: Interactive What-If ROI & Budget Simulator",
        "🎯 Tab 3: Operational Action Queue & Customer 360",
        "🔮 Tab 4: Real-Time Customer Predictor & Advisor",
    ]
)

# ===========================================================================
# TAB 1 — EXECUTIVE KPI & PORTFOLIO OVERVIEW
# ===========================================================================
with tab1:
    st.subheader("Executive Portfolio Key Performance Indicators (Baseline Policy)")
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Total Customers", f"{base_kpis['total_customers']:,}", "100% Active Base")
    k2.metric("Observed Churn Rate", f"{df_360['churn_binary'].mean() * 100:.2f}%", "1,869 Churned")
    k3.metric("Annualized Revenue", f"${base_kpis['total_annualized_revenue'] / 1e6:.2f}M", "$64.76 Avg/Mo")
    k4.metric(
        "Annual Revenue at Risk",
        f"${base_kpis['total_revenue_at_risk'] / 1e6:.2f}M",
        f"{base_kpis['revenue_at_risk_pct_of_annual']:.1f}% of Annual Rev",
        delta_color="inverse",
    )
    k5.metric(
        "Expected Saved Revenue",
        f"${base_kpis['total_expected_saved_revenue'] / 1e3:.2f}K",
        f"Cost: ${base_kpis['total_intervention_cost']:,.0f}",
    )
    k6.metric(
        "Expected Net Benefit",
        f"${base_kpis['total_expected_net_benefit'] / 1e3:.2f}K",
        f"{base_kpis['targeted_customers']:,} Targeted",
    )
    k7.metric("Portfolio Expected ROI", f"{base_kpis['portfolio_roi_pct']:,.2f}%", "Net / Budget")

    st.divider()

    # Row 1 of charts: 9-Cell Matrix Heatmap + Revenue at Risk vs Net Benefit by Quadrant
    col_left, col_right = st.columns([1.05, 1.15])

    with col_left:
        st.markdown("#### 1. 9-Cell Value Tier × Churn Risk Matrix")
        matrix_metric = st.radio(
            "Select Matrix Metric:",
            ["Customer Count & Share", "Annual Revenue at Risk ($)"],
            horizontal=True,
            key="tab1_matrix_metric",
        )
        cells_df = pd.DataFrame(baseline_summary["matrix_9_cells"])
        tier_order = ["High", "Medium", "Low"]
        risk_order = ["High", "Medium", "Low"]

        if matrix_metric == "Customer Count & Share":
            piv_val = (
                cells_df.pivot(index="value_tier", columns="risk_level", values="customers")
                .reindex(index=tier_order, columns=risk_order)
            )
            piv_rar = (
                cells_df.pivot(index="value_tier", columns="risk_level", values="revenue_at_risk")
                .reindex(index=tier_order, columns=risk_order)
            )
            text_grid = [
                [
                    f"<b>{int(piv_val.loc[t, r]):,}</b> ({piv_val.loc[t, r] / len(df_360) * 100:.1f}%)<br>"
                    f"RaR: ${piv_rar.loc[t, r] / 1e3:,.0f}K"
                    for r in risk_order
                ]
                for t in tier_order
            ]
            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=piv_val.values,
                    x=["High Risk (p ≥ 0.60)", "Medium Risk (0.30–0.60)", "Low Risk (p < 0.30)"],
                    y=["High Value Tier", "Medium Value Tier", "Low Value Tier"],
                    text=text_grid,
                    texttemplate="%{text}",
                    colorscale="YlOrRd",
                )
            )
        else:
            piv_val = (
                cells_df.pivot(index="value_tier", columns="risk_level", values="revenue_at_risk")
                .reindex(index=tier_order, columns=risk_order)
            )
            piv_cnt = (
                cells_df.pivot(index="value_tier", columns="risk_level", values="customers")
                .reindex(index=tier_order, columns=risk_order)
            )
            piv_roi = (
                cells_df.pivot(index="value_tier", columns="risk_level", values="cell_roi_pct")
                .reindex(index=tier_order, columns=risk_order)
            )
            text_grid = [
                [
                    f"<b>${piv_val.loc[t, r] / 1e3:,.1f}K</b><br>"
                    f"{int(piv_cnt.loc[t, r]):,} KH | ROI: {piv_roi.loc[t, r]:,.0f}%"
                    for r in risk_order
                ]
                for t in tier_order
            ]
            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=piv_val.values,
                    x=["High Risk (p ≥ 0.60)", "Medium Risk (0.30–0.60)", "Low Risk (p < 0.30)"],
                    y=["High Value Tier", "Medium Value Tier", "Low Value Tier"],
                    text=text_grid,
                    texttemplate="%{text}",
                    colorscale="Reds",
                )
            )
        fig_heat.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Predicted Churn Risk Level (XGBoost)",
            yaxis_title="Customer Value Tier (K-Means)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_right:
        st.markdown("#### 2. Revenue at Risk vs. Expected Net Benefit by Decision Quadrant")
        q_df = pd.DataFrame(baseline_summary["quadrant_breakdown"])
        fig_quad = go.Figure()
        fig_quad.add_trace(
            go.Bar(
                name="Annual Revenue at Risk ($)",
                x=q_df["decision_quadrant"],
                y=q_df["revenue_at_risk"],
                marker_color="#ef4444",
                text=[f"${v / 1e3:,.0f}K" for v in q_df["revenue_at_risk"]],
                textposition="outside",
            )
        )
        fig_quad.add_trace(
            go.Bar(
                name="Expected Saved Revenue ($)",
                x=q_df["decision_quadrant"],
                y=q_df["expected_saved_revenue"],
                marker_color="#10b981",
                text=[f"${v / 1e3:,.0f}K" for v in q_df["expected_saved_revenue"]],
                textposition="outside",
            )
        )
        fig_quad.add_trace(
            go.Bar(
                name="Expected Net Benefit ($)",
                x=q_df["decision_quadrant"],
                y=q_df["expected_net_benefit"],
                marker_color="#3b82f6",
                text=[f"${v / 1e3:,.0f}K" for v in q_df["expected_net_benefit"]],
                textposition="outside",
            )
        )
        fig_quad.update_layout(
            barmode="group",
            height=430,
            margin=dict(l=10, r=10, t=30, b=60),
            yaxis_title="USD ($)",
            xaxis_title="Prescriptive Decision Quadrant",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_quad, use_container_width=True)

    st.divider()

    # Row 2 of charts: K-Means Value Tier Profile + Global SHAP Importance + XGBoost Evaluation
    c_km, c_shap, c_xgb = st.columns([1.05, 1.15, 0.95])

    with c_km:
        st.markdown("#### 3. K-Means Customer Value Tier Profiles ($k=3$)")
        km_prof = pd.DataFrame(model_metrics["value_tier_profile"])
        km_display = pd.DataFrame(
            {
                "Value Tier": km_prof["value_tier"],
                "Customers": km_prof["customers"].map(lambda x: f"{int(x):,}"),
                "Tenure (m)": km_prof["tenure"].round(1),
                "Monthly ($)": km_prof["monthly_charges"].map(lambda x: f"${x:.2f}"),
                "Total ($)": km_prof["total_charges"].map(lambda x: f"${x:,.0f}"),
                "VAS Count": km_prof["total_services_count"].round(2),
                "Churn Rate": (km_prof["churn_rate"] * 100).map(lambda x: f"{x:.1f}%"),
                "Rev Share": (km_prof["revenue_share"] * 100).map(lambda x: f"{x:.1f}%"),
            }
        )
        st.dataframe(km_display, use_container_width=True, hide_index=True)

        sil_df = pd.DataFrame(
            {
                "k": [int(k) for k in model_metrics["silhouette_scores"].keys()],
                "Silhouette Score": [float(v) for v in model_metrics["silhouette_scores"].values()],
            }
        )
        fig_sil = px.line(
            sil_df,
            x="k",
            y="Silhouette Score",
            markers=True,
            title="Silhouette Score Diagnostic (k = 2..6, Selected k = 3: 0.402)",
        )
        fig_sil.update_layout(height=230, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_sil, use_container_width=True)

    with c_shap:
        st.markdown("#### 4. Global SHAP Churn Drivers (Top 12 Business Attributes)")
        shap_imp = pd.Series(model_metrics["global_shap_importance"]).sort_values(ascending=True).tail(12)
        fig_shap = go.Figure(
            go.Bar(
                x=shap_imp.values,
                y=shap_imp.index,
                orientation="h",
                marker_color="#6366f1",
                text=[f"{v:.3f}" for v in shap_imp.values],
                textposition="outside",
            )
        )
        fig_shap.update_layout(
            height=385,
            margin=dict(l=10, r=35, t=20, b=10),
            xaxis_title="Mean |SHAP Value| (Impact on Log-Odds of Churn)",
            yaxis_title="Business Attribute",
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with c_xgb:
        st.markdown("#### 5. XGBoost Hold-Out Test Performance ($n = 1{,}409$)")
        m_df = pd.DataFrame(
            {
                "Metric": ["ROC-AUC", "PR-AUC", "Recall (Sensitivity)", "F1-Score", "Accuracy", "Precision"],
                "Score": [
                    f"{model_metrics['roc_auc']:.4f}",
                    f"{model_metrics['pr_auc']:.4f}",
                    f"{model_metrics['recall']:.4f}",
                    f"{model_metrics['f1']:.4f}",
                    f"{model_metrics['accuracy']:.4f}",
                    f"{model_metrics['precision']:.4f}",
                ],
            }
        )
        st.dataframe(m_df, use_container_width=True, hide_index=True)

        cm = model_metrics["confusion_matrix"]
        cm_z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        cm_text = [
            [f"TN: {cm['tn']}<br>(True Loyal)", f"FP: {cm['fp']}<br>(False Alarm)"],
            [f"FN: {cm['fn']}<br>(Missed Churn)", f"TP: {cm['tp']}<br>(Caught Churn)"],
        ]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm_z,
                x=["Predicted: Loyal (0)", "Predicted: Churn (1)"],
                y=["Actual: Loyal (0)", "Actual: Churn (1)"],
                text=cm_text,
                texttemplate="<b>%{text}</b>",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_cm.update_layout(
            title="Hold-Out Confusion Matrix (Threshold = 0.50)",
            height=220,
            margin=dict(l=10, r=10, t=35, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)


# ===========================================================================
# TAB 2 — INTERACTIVE WHAT-IF ROI & BUDGET SIMULATOR
# ===========================================================================
with tab2:
    st.subheader("Interactive What-If Retention Policy & Budget Simulator")
    st.markdown(
        "Adjust risk classification thresholds, intervention channel costs, promotional discount rates, "
        "expected campaign conversion rates, and an optional **Maximum Campaign Budget Cap** to evaluate "
        "real-time financial trade-offs against the baseline strategy."
    )

    with st.expander("⚙️ Configure Simulation Parameters (Sliders)", expanded=True):
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)

        with s_col1:
            st.markdown("**1. Churn Risk Thresholds**")
            sim_high_thr = st.slider(
                "High Risk Threshold (p ≥)",
                min_value=0.45,
                max_value=0.85,
                value=0.60,
                step=0.05,
                help="Baseline = 0.60",
            )
            sim_med_thr = st.slider(
                "Medium Risk Threshold (p ≥)",
                min_value=0.15,
                max_value=0.44,
                value=0.30,
                step=0.05,
                help="Baseline = 0.30",
            )

        with s_col2:
            st.markdown("**2. Unit Channel Costs ($)**")
            sim_call_cost = st.slider(
                "1-on-1 Account Manager Call ($)",
                min_value=1.0,
                max_value=25.0,
                value=float(DEFAULT_PARAMS["call_cost"]),
                step=0.5,
            )
            sim_email_cost = st.slider(
                "Automated Email / SMS Cost ($)",
                min_value=0.10,
                max_value=5.00,
                value=float(DEFAULT_PARAMS["email_sms_cost"]),
                step=0.10,
            )

        with s_col3:
            st.markdown("**3. Promotional Discounts (3-Mo)**")
            sim_high_disc_pct = (
                st.slider(
                    "High-Value Loyalty Discount (%)",
                    min_value=5,
                    max_value=40,
                    value=int(DEFAULT_PARAMS["high_value_discount_pct"] * 100),
                    step=5,
                )
                / 100.0
            )
            sim_med_disc_pct = (
                st.slider(
                    "Medium-Value Voucher Discount (%)",
                    min_value=5,
                    max_value=30,
                    value=int(DEFAULT_PARAMS["medium_value_discount_pct"] * 100),
                    step=5,
                )
                / 100.0
            )

        with s_col4:
            st.markdown("**4. Success Rates & Budget Cap**")
            sim_sr_vip = (
                st.slider(
                    "VIP 1-on-1 Retention Success (%)",
                    min_value=10,
                    max_value=80,
                    value=int(DEFAULT_PARAMS["success_rate_vip_call"] * 100),
                    step=5,
                )
                / 100.0
            )
            sim_sr_auto = (
                st.slider(
                    "Auto Campaign Retention Success (%)",
                    min_value=5,
                    max_value=60,
                    value=int(DEFAULT_PARAMS["success_rate_auto_campaign"] * 100),
                    step=5,
                )
                / 100.0
            )

        b_col1, b_col2 = st.columns([1, 2])
        with b_col1:
            enable_cap = st.checkbox("Enable Maximum Campaign Budget Cap", value=False)
        with b_col2:
            budget_cap_val = st.slider(
                "Maximum Campaign Budget Cap ($) — prioritizes highest-ROI customers first",
                min_value=2000.0,
                max_value=60000.0,
                value=20000.0,
                step=1000.0,
                disabled=not enable_cap,
            )

    # Re-assign risk_level dynamically based on simulated thresholds
    sim_input_df = df_360[
        [
            "customerID",
            "monthly_charges",
            "value_tier",
            "churn_probability",
            "top_shap_driver_1",
            "top_shap_driver_2",
        ]
    ].copy()
    probs = sim_input_df["churn_probability"].values
    sim_input_df["risk_level"] = np.select(
        [probs >= sim_high_thr, probs >= sim_med_thr],
        ["High", "Medium"],
        default="Low",
    )

    sim_params = {
        **DEFAULT_PARAMS,
        "call_cost": sim_call_cost,
        "email_sms_cost": sim_email_cost,
        "high_value_discount_pct": sim_high_disc_pct,
        "medium_value_discount_pct": sim_med_disc_pct,
        "success_rate_vip_call": sim_sr_vip,
        "success_rate_auto_campaign": sim_sr_auto,
    }

    sim_results_df = run_decision_simulation(sim_input_df, params=sim_params)
    if enable_cap:
        sim_results_df = apply_budget_cap(sim_results_df, budget_cap=budget_cap_val)
    sim_summary = summarize_decisions(sim_results_df, params=sim_params)
    sim_kpis = sim_summary["portfolio_kpis"]

    # Delta KPI row (Simulated vs Baseline)
    st.markdown("#### Simulated vs. Baseline Financial Impact")
    d1, d2, d3, d4, d5 = st.columns(5)

    delta_targeted = sim_kpis["targeted_customers"] - base_kpis["targeted_customers"]
    delta_cost = sim_kpis["total_intervention_cost"] - base_kpis["total_intervention_cost"]
    delta_saved = sim_kpis["total_expected_saved_revenue"] - base_kpis["total_expected_saved_revenue"]
    delta_net = sim_kpis["total_expected_net_benefit"] - base_kpis["total_expected_net_benefit"]
    delta_roi = sim_kpis["portfolio_roi_pct"] - base_kpis["portfolio_roi_pct"]

    d1.metric(
        "Targeted Customers",
        f"{sim_kpis['targeted_customers']:,}",
        f"{delta_targeted:+,} vs. Baseline",
    )
    d2.metric(
        "Simulated Campaign Cost",
        f"${sim_kpis['total_intervention_cost']:,.2f}",
        f"{delta_cost:+,.2f} vs. Baseline",
        delta_color="inverse",
    )
    d3.metric(
        "Simulated Saved Revenue",
        f"${sim_kpis['total_expected_saved_revenue']:,.2f}",
        f"{delta_saved:+,.2f} vs. Baseline",
    )
    d4.metric(
        "Simulated Net Benefit",
        f"${sim_kpis['total_expected_net_benefit']:,.2f}",
        f"{delta_net:+,.2f} vs. Baseline",
    )
    d5.metric(
        "Simulated Portfolio ROI",
        f"{sim_kpis['portfolio_roi_pct']:,.2f}%",
        f"{delta_roi:+,.2f}% vs. Baseline",
    )

    sim_c1, sim_c2 = st.columns([1.05, 1.15])
    with sim_c1:
        st.markdown("#### Portfolio Comparison: Baseline vs. Simulated")
        comp_df = pd.DataFrame(
            {
                "Metric": ["Intervention Cost ($)", "Expected Saved Revenue ($)", "Expected Net Benefit ($)"],
                "Baseline Policy": [
                    base_kpis["total_intervention_cost"],
                    base_kpis["total_expected_saved_revenue"],
                    base_kpis["total_expected_net_benefit"],
                ],
                "Simulated Policy": [
                    sim_kpis["total_intervention_cost"],
                    sim_kpis["total_expected_saved_revenue"],
                    sim_kpis["total_expected_net_benefit"],
                ],
            }
        )
        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Bar(
                name="Baseline Policy",
                x=comp_df["Metric"],
                y=comp_df["Baseline Policy"],
                marker_color="#64748b",
                text=[f"${v:,.0f}" for v in comp_df["Baseline Policy"]],
                textposition="outside",
            )
        )
        fig_comp.add_trace(
            go.Bar(
                name="Simulated Policy",
                x=comp_df["Metric"],
                y=comp_df["Simulated Policy"],
                marker_color="#0ea5e9",
                text=[f"${v:,.0f}" for v in comp_df["Simulated Policy"]],
                textposition="outside",
            )
        )
        fig_comp.update_layout(
            barmode="group",
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="USD ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with sim_c2:
        st.markdown("#### Simulated Breakdown by Strategic Decision Quadrant")
        sim_q_df = pd.DataFrame(sim_summary["quadrant_breakdown"])
        st.dataframe(
            sim_q_df.rename(
                columns={
                    "decision_quadrant": "Decision Quadrant",
                    "customers": "Customers",
                    "mean_churn_prob": "Mean Churn Prob",
                    "revenue_at_risk": "Revenue at Risk ($)",
                    "intervention_cost": "Cost ($)",
                    "expected_saved_revenue": "Saved Rev ($)",
                    "expected_net_benefit": "Net Benefit ($)",
                    "quadrant_roi_pct": "ROI (%)",
                }
            )[
                [
                    "Decision Quadrant",
                    "Customers",
                    "Mean Churn Prob",
                    "Revenue at Risk ($)",
                    "Cost ($)",
                    "Saved Rev ($)",
                    "Net Benefit ($)",
                    "ROI (%)",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


# ===========================================================================
# TAB 3 — OPERATIONAL ACTION QUEUE & CUSTOMER 360 INSPECTOR
# ===========================================================================
with tab3:
    st.subheader("Operational Retention Action Queue & Campaign Export")
    st.markdown(
        "Filter prioritized customer segments, inspect individualized SHAP churn drivers and prescribed "
        "actions, export campaign lists for Telesales/Marketing execution, or drill down into any "
        "single customer's 360° profile."
    )

    f_col1, f_col2, f_col3, f_col4 = st.columns([1.6, 1.0, 1.0, 1.1])
    with f_col1:
        selected_quadrants = st.multiselect(
            "Filter by Strategic Decision Quadrant:",
            options=QUADRANT_ORDER,
            default=["Critical VIP Retention", "Automated Retention Campaign"],
        )
    with f_col2:
        selected_tiers = st.multiselect(
            "Filter by Value Tier:",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
    with f_col3:
        selected_risks = st.multiselect(
            "Filter by Churn Risk Level:",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
    with f_col4:
        min_rar = st.number_input(
            "Minimum Annual Revenue at Risk ($):",
            min_value=0.0,
            max_value=1500.0,
            value=0.0,
            step=50.0,
        )

    filtered_df = df_360[
        df_360["decision_quadrant"].isin(selected_quadrants)
        & df_360["value_tier"].isin(selected_tiers)
        & df_360["risk_level"].isin(selected_risks)
        & (df_360["revenue_at_risk"] >= min_rar)
    ].sort_values("revenue_at_risk", ascending=False)

    q_m1, q_m2, q_m3, q_m4 = st.columns(4)
    q_m1.metric("Filtered Accounts in Queue", f"{len(filtered_df):,}")
    q_m2.metric("Queue Revenue at Risk", f"${filtered_df['revenue_at_risk'].sum():,.2f}")
    q_m3.metric("Queue Intervention Cost", f"${filtered_df['intervention_cost'].sum():,.2f}")
    q_m4.metric("Queue Expected Net Benefit", f"${filtered_df['expected_net_benefit'].sum():,.2f}")

    queue_cols = [
        "customerID",
        "value_tier",
        "risk_level",
        "churn_probability",
        "contract",
        "tenure",
        "monthly_charges",
        "annualized_revenue",
        "revenue_at_risk",
        "top_shap_driver_1",
        "top_shap_driver_2",
        "decision_quadrant",
        "recommended_action",
        "intervention_cost",
        "expected_saved_revenue",
        "expected_net_benefit",
        "expected_roi",
    ]

    st.dataframe(
        filtered_df[queue_cols].head(500),
        use_container_width=True,
        hide_index=True,
        height=340,
    )

    csv_bytes = filtered_df[queue_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Actionable Campaign CSV ({len(filtered_df):,} Customers)",
        data=csv_bytes,
        file_name="ism_retention_campaign_queue.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("🔍 Customer 360° Profile & Local SHAP Explainability Inspector")

    candidate_ids = filtered_df["customerID"].tolist() if len(filtered_df) > 0 else df_360["customerID"].tolist()
    selected_cid = st.selectbox(
        "Select a Customer ID from the current queue (sorted by Revenue at Risk):",
        options=candidate_ids[:1000],
        index=0,
    )

    cust_row = df_360[df_360["customerID"] == selected_cid].iloc[0]

    p_col1, p_col2, p_col3 = st.columns([1.05, 1.05, 1.3])
    with p_col1:
        st.markdown("##### 👤 Demographics & Services")
        st.markdown(
            f"- **Customer ID:** `{cust_row['customerID']}`\n"
            f"- **Gender / Senior:** `{cust_row['gender']}` / `{'Yes' if cust_row['senior_citizen'] else 'No'}`\n"
            f"- **Partner / Dependents:** `{'Yes' if cust_row['partner'] else 'No'}` / `{'Yes' if cust_row['dependents'] else 'No'}`\n"
            f"- **Tenure:** `{cust_row['tenure']} months ({cust_row['tenure_group']})`\n"
            f"- **Contract:** `{cust_row['contract']}`\n"
            f"- **Internet / Phone:** `{cust_row['internet_service']}` / `{'Yes' if cust_row['phone_service'] else 'No'}`\n"
            f"- **VAS Count:** `{cust_row['total_services_count']} / 6` "
            f"(Sec: `{cust_row['online_security']}`, Tech: `{cust_row['tech_support']}`)\n"
            f"- **Payment Method:** `{cust_row['payment_method']}`"
        )

    with p_col2:
        st.markdown("##### 🎯 Mining Scores & Financial Profile")
        st.markdown(
            f"- **Customer Value Tier:** **{cust_row['value_tier']}** (Cluster `{cust_row['cluster_id']}`)\n"
            f"- **Churn Probability:** **{cust_row['churn_probability'] * 100:.2f}%** (**{cust_row['risk_level']} Risk**)\n"
            f"- **Monthly / Total Charges:** `${cust_row['monthly_charges']:.2f}` / `${cust_row['total_charges']:,.2f}`\n"
            f"- **Annualized Revenue:** `${cust_row['annualized_revenue']:,.2f}`\n"
            f"- **Annual Revenue at Risk:** `${cust_row['revenue_at_risk']:,.2f}`\n"
            f"- **Top SHAP Driver 1:** `{cust_row['top_shap_driver_1']}`\n"
            f"- **Top SHAP Driver 2:** `{cust_row['top_shap_driver_2']}`\n"
            f"- **Decision Quadrant:** `{cust_row['decision_quadrant']}`"
        )
        st.info(
            f"**Prescribed Action:** {cust_row['recommended_action']}\n\n"
            f"**Cost:** `${cust_row['intervention_cost']:.2f}` · "
            f"**Expected Saved:** `${cust_row['expected_saved_revenue']:.2f}` · "
            f"**Net Benefit:** `${cust_row['expected_net_benefit']:.2f}` · "
            f"**Expected ROI:** `{cust_row['expected_roi']:,.1f}%`"
        )

    with p_col3:
        st.markdown("##### 📊 Local SHAP Feature Contributions (Log-Odds)")
        shap_meta = artifacts["shap"]
        cid_idx = shap_meta["customer_ids"].index(selected_cid)
        cust_shap_series = shap_meta["shap_by_feature"].iloc[cid_idx].sort_values(key=lambda s: s.abs(), ascending=True).tail(10)
        colors = ["#ef4444" if v > 0 else "#10b981" for v in cust_shap_series.values]
        fig_local_shap = go.Figure(
            go.Bar(
                x=cust_shap_series.values,
                y=cust_shap_series.index,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f}" for v in cust_shap_series.values],
                textposition="outside",
            )
        )
        fig_local_shap.update_layout(
            height=300,
            margin=dict(l=10, r=35, t=10, b=10),
            xaxis_title="SHAP Contribution (+ Increases Churn Risk | - Reduces Risk)",
            yaxis_title="Attribute",
        )
        st.plotly_chart(fig_local_shap, use_container_width=True)


# ===========================================================================
# TAB 4 — REAL-TIME NEW CUSTOMER PREDICTOR & DECISION ADVISOR
# ===========================================================================
with tab4:
    st.subheader("Real-Time New Customer Predictor & Prescriptive Decision Advisor")
    st.markdown(
        "Enter a hypothetical or newly onboarded subscriber's profile to execute live inference "
        "through the trained **K-Means Value Pipeline**, **XGBoost Churn Pipeline**, **SHAP TreeExplainer**, "
        "and **Prescriptive ROI Engine**."
    )

    with st.form("realtime_predictor_form"):
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            st.markdown("**Demographics & Lifecycle**")
            in_gender = st.selectbox("Gender", ["Female", "Male"])
            in_senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            in_partner = st.selectbox("Partner", ["No", "Yes"])
            in_dependents = st.selectbox("Dependents", ["No", "Yes"])
            in_tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=2)

        with f2:
            st.markdown("**Core Connectivity & Contract**")
            in_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
            in_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0)
            in_phone = st.selectbox("Phone Service", ["Yes", "No"], index=0)
            in_multilines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0)

        with f3:
            st.markdown("**Value-Added Services (VAS)**")
            in_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0)
            in_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=0)
            in_dev = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0)
            in_tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0)
            in_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=0)
            in_mov = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], index=0)

        with f4:
            st.markdown("**Billing & Payment**")
            in_paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
            in_payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=0,
            )
            in_monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=95.50, step=1.0)
            default_total = round(float(in_monthly * max(in_tenure, 1)), 2) if in_tenure > 0 else 0.0
            in_total = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=default_total,
                step=10.0,
            )

        submitted = st.form_submit_button("🔮 Predict & Generate Retention Strategy", type="primary")

    if submitted:
        # Build engineered record matching customer_master schema
        vas_list = [in_sec, in_backup, in_dev, in_tech, in_tv, in_mov]
        vas_count = sum(1 for v in vas_list if v == "Yes")
        t_group = str(pd.cut([in_tenure], bins=TENURE_BINS, labels=TENURE_LABELS)[0])
        avg_spend = round(in_total / in_tenure, 2) if in_tenure > 0 else round(in_monthly, 2)
        auto_pay = 1 if in_payment in AUTO_PAYMENT_METHODS else 0

        record = {
            "customerID": "SIM-NEW-001",
            "gender": in_gender,
            "senior_citizen": int(in_senior),
            "partner": 1 if in_partner == "Yes" else 0,
            "dependents": 1 if in_dependents == "Yes" else 0,
            "tenure": int(in_tenure),
            "tenure_group": t_group,
            "contract": in_contract,
            "phone_service": 1 if in_phone == "Yes" else 0,
            "multiple_lines": "No phone service" if in_phone == "No" else in_multilines,
            "internet_service": in_internet,
            "online_security": "No internet service" if in_internet == "No" else in_sec,
            "online_backup": "No internet service" if in_internet == "No" else in_backup,
            "device_protection": "No internet service" if in_internet == "No" else in_dev,
            "tech_support": "No internet service" if in_internet == "No" else in_tech,
            "streaming_tv": "No internet service" if in_internet == "No" else in_tv,
            "streaming_movies": "No internet service" if in_internet == "No" else in_mov,
            "total_services_count": int(vas_count),
            "paperless_billing": 1 if in_paperless == "Yes" else 0,
            "payment_method": in_payment,
            "is_auto_payment": int(auto_pay),
            "monthly_charges": float(in_monthly),
            "total_charges": float(in_total),
            "avg_monthly_spend": float(avg_spend),
        }

        scored = score_customer(record, artifacts=artifacts)
        single_df = pd.DataFrame([{**record, **scored}])
        decision_df = run_decision_simulation(single_df, params=DEFAULT_PARAMS)
        dec_row = decision_df.iloc[0]

        # Also compute full local SHAP vector for plotting
        _, _, single_shap_df, _, _ = explain_churn(
            artifacts["xgb"]["pipeline"],
            pd.DataFrame([record])[artifacts["xgb"]["features"]],
        )

        st.success("✅ Real-Time Mining & Prescriptive Decision Evaluation Completed")
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("Predicted Value Tier", f"{dec_row['value_tier']} Value", f"Cluster {scored['cluster_id']}")
        r2.metric(
            "Churn Probability",
            f"{dec_row['churn_probability'] * 100:.2f}%",
            f"{dec_row['risk_level']} Risk",
            delta_color="inverse",
        )
        r3.metric("Annual Revenue at Risk", f"${dec_row['revenue_at_risk']:,.2f}", f"Annual: ${dec_row['annualized_revenue']:,.0f}")
        r4.metric("Intervention Cost", f"${dec_row['intervention_cost']:,.2f}", f"Success Rate: {dec_row['expected_success_rate'] * 100:.0f}%")
        r5.metric("Expected Net Benefit", f"${dec_row['expected_net_benefit']:,.2f}", f"Saved: ${dec_row['expected_saved_revenue']:,.2f}")
        r6.metric("Expected Campaign ROI", f"{dec_row['expected_roi']:,.1f}%")

        adv_left, adv_right = st.columns([1.1, 1.2])
        with adv_left:
            st.markdown("#### 🎯 Prescriptive Retention Playbook Recommendation")
            st.markdown(
                f"- **Strategic Decision Quadrant:** `{dec_row['decision_quadrant']}`\n"
                f"- **Primary SHAP Churn Driver 1:** `{scored['top_shap_driver_1']}`\n"
                f"- **Secondary SHAP Churn Driver 2:** `{scored['top_shap_driver_2']}`\n"
                f"- **Engineered Lifecycle Stage:** `{t_group}` (`{vas_count}` active VAS add-ons)"
            )
            st.warning(f"**Recommended Managerial Action:**\n\n{dec_row['recommended_action']}")

        with adv_right:
            st.markdown("#### 📊 Customer-Specific SHAP Attribution Waterfall (Top 10 Features)")
            s_series = single_shap_df.iloc[0].sort_values(key=lambda s: s.abs(), ascending=True).tail(10)
            s_colors = ["#ef4444" if v > 0 else "#10b981" for v in s_series.values]
            fig_single_shap = go.Figure(
                go.Bar(
                    x=s_series.values,
                    y=s_series.index,
                    orientation="h",
                    marker_color=s_colors,
                    text=[f"{v:+.2f}" for v in s_series.values],
                    textposition="outside",
                )
            )
            fig_single_shap.update_layout(
                height=310,
                margin=dict(l=10, r=35, t=10, b=10),
                xaxis_title="SHAP Log-Odds Contribution (+ Risk | - Protective)",
                yaxis_title="Business Attribute",
            )
            st.plotly_chart(fig_single_shap, use_container_width=True)
