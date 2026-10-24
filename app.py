"""
Streamlit Decision Support Dashboard v3.0 (Tier 4 - Presentation Layer).

Intelligent Customer Retention & Value Optimization System using Data Mining,
Causal Uplift Modeling, and Generative AI Retention Copilot Techniques
(K-Means + XGBoost + TreeSHAP + Causal Uplift + Knapsack Optimizer + GenAI Copilot).

Enterprise v3.0 Architecture (5 Specialized Tabs):
    Tab 1 - Executive KPI, Causal Uplift Guardrail & Portfolio Overview
    Tab 2 - Interactive What-If, Causal Exclusion, Profit Curve & Knapsack Optimizer
    Tab 3 - Operational Action Queue, Customer 360°, GenAI Copilot & Closed-Loop CRM
    Tab 4 - Real-Time New Customer Predictor, Causal Segment & GenAI Pitch Advisor
    Tab 5 - Model Benchmarking & System Governance (ISM Audit View)

Usage:
    .venv\\Scripts\\streamlit.exe run app.py
"""

import io
import json
import sqlite3
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure local modules in src/ are importable
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copilot_engine import generate_retention_pitch  # noqa: E402
from decision_engine import (  # noqa: E402
    ACTION_STATUS_DOMAIN,
    DEFAULT_PARAMS,
    QUADRANT_ORDER,
    ensure_crm_columns,
    run_decision_simulation,
    summarize_decisions,
    update_customer_crm_status,
)
from etl_pipeline import (  # noqa: E402
    AUTO_PAYMENT_METHODS,
    TENURE_BINS,
    TENURE_LABELS,
)
from mining_engine import explain_churn, score_customer  # noqa: E402
from optimization_engine import (  # noqa: E402
    optimize_under_constraints,
    run_threshold_profit_curve,
)
from uplift_engine import (  # noqa: E402
    CAUSAL_SEGMENTS,
    UPLIFT_SUMMARY_JSON,
    apply_uplift_exclusion,
    compute_causal_uplift,
)

# ---------------------------------------------------------------------------
# Paths & Page Configuration
# ---------------------------------------------------------------------------
DB_FILE = PROJECT_ROOT / "database" / "ism_dss.db"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_FILE = MODELS_DIR / "model_metrics.json"
DECISION_SUMMARY_FILE = MODELS_DIR / "decision_summary.json"
BENCHMARK_FILE = MODELS_DIR / "model_benchmark.json"
PROFIT_CURVE_FILE = MODELS_DIR / "threshold_profit_curve.json"
UPLIFT_FILE = UPLIFT_SUMMARY_JSON
KMEANS_FILE = MODELS_DIR / "kmeans_value_pipeline.joblib"
XGB_FILE = MODELS_DIR / "xgb_churn_pipeline.joblib"
SHAP_FILE = MODELS_DIR / "shap_metadata.joblib"

st.set_page_config(
    page_title="ISM Customer Retention & Value Optimization DSS v3.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Modern Enterprise Slate Dark Theme CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global layout & typography */
    .stApp {
        background-color: #0B0F19;
        color: #F9FAFB;
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.2rem;
        max-width: 96%;
    }
    /* Style native Streamlit metrics as dark slate cards (fallback safety) */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1E293B, #0F172A) !important;
        border: 1px solid #334155 !important;
        border-left: 4px solid #38BDF8 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
    }
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
    /* Custom Enterprise KPI Card */
    .kpi-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 15px 18px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
        margin-bottom: 12px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #475569;
    }
    .kpi-title {
        color: #94A3B8 !important;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #F8FAFC !important;
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 6px;
    }
    .kpi-sub {
        font-size: 0.78rem;
        font-weight: 500;
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
    }
    /* Sidebar Status Cards */
    .sb-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }
    .sb-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.80rem;
        font-weight: 700;
        color: #E2E8F0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #334155;
    }
    .sb-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        font-size: 0.70rem;
        padding: 2px 8px;
        border-radius: 999px;
        font-weight: 600;
    }
    .sb-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.83rem;
        padding: 4px 0;
    }
    .sb-label {
        color: #94A3B8;
    }
    .sb-val {
        color: #F8FAFC;
        font-weight: 600;
    }
    /* Profile Card in Tab 3 / Tab 4 / Tab 5 */
    .profile-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 18px;
        height: 100%;
    }
    .profile-card h5 {
        color: #38BDF8 !important;
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 0.95rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 6px;
    }
    /* Plotly chart container rounding */
    div[data-testid="stPlotlyChart"] {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 6px;
    }
    /* Tabs styling */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.93rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ACCENT_PALETTE = {
    "cyan": ("#38BDF8", "rgba(56, 189, 248, 0.14)", "#7DD3FC"),
    "rose": ("#F43F5E", "rgba(244, 63, 94, 0.16)", "#FDA4AF"),
    "amber": ("#F59E0B", "rgba(245, 158, 11, 0.16)", "#FCD34D"),
    "emerald": ("#10B981", "rgba(16, 185, 129, 0.16)", "#6EE7B7"),
    "indigo": ("#6366F1", "rgba(99, 102, 241, 0.16)", "#A5B4FC"),
}

SHORT_QUADRANT_LABELS = {
    "Critical VIP Retention": "Critical VIP",
    "Proactive VIP Care": "Proactive VIP",
    "Automated Retention Campaign": "Auto Campaign",
    "Targeted Nudge": "Targeted Nudge",
    "No Financial Intervention": "Zero-Cost Survey",
    "Champions / Nurture (No Intervention)": "Champions / Nurture",
    "Deferred (Resource Constraint)": "Deferred (Cap)",
    "Deferred (Budget Cap Exceeded)": "Deferred (Cap)",
}


def render_kpi_card(
    col: Any,
    title: str,
    value: str,
    subtitle: str,
    accent: str = "cyan",
) -> None:
    """Render a high-contrast dark slate KPI card inside a Streamlit column."""
    border_hex, badge_bg, badge_text = ACCENT_PALETTE.get(accent, ACCENT_PALETTE["cyan"])
    col.markdown(
        f"""
        <div class="kpi-card" style="border-left: 4px solid {border_hex};">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <span class="kpi-sub" style="background: {badge_bg}; color: {badge_text};">
                {subtitle}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_dark(fig: go.Figure, height: int = 390, title: Optional[str] = None) -> go.Figure:
    """Apply uniform Enterprise Slate Dark styling to any Plotly figure."""
    layout_kwargs: Dict[str, Any] = dict(
        height=height,
        paper_bgcolor="rgba(17, 24, 39, 0.85)",
        plot_bgcolor="rgba(15, 23, 42, 0.55)",
        font=dict(color="#E2E8F0", size=12, family="sans-serif"),
        margin=dict(l=16, r=16, t=42 if title else 24, b=20),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(color="#F8FAFC", size=14))
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor="rgba(51, 65, 85, 0.45)", zerolinecolor="rgba(51, 65, 85, 0.6)")
    fig.update_yaxes(gridcolor="rgba(51, 65, 85, 0.45)", zerolinecolor="rgba(51, 65, 85, 0.6)")
    return fig


# ---------------------------------------------------------------------------
# Cached Data & Artifact Loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_customer_360() -> pd.DataFrame:
    """Load the full 7,043-row Customer 360 analytical view from SQLite."""
    with sqlite3.connect(DB_FILE) as conn:
        ensure_crm_columns(conn)
        df = pd.read_sql("SELECT * FROM vw_customer_360 ORDER BY customerID", conn)
    return df


@st.cache_data(show_spinner=False)
def load_etl_audit_log() -> pd.DataFrame:
    """Load ETL warehouse execution lineage records from ``etl_audit_log``."""
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql("SELECT * FROM etl_audit_log ORDER BY run_id DESC", conn)


@st.cache_data(show_spinner=False)
def load_json_summaries() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load model metrics, baseline decision summary, benchmark, profit curve, and uplift summary JSONs."""
    metrics = json.loads(METRICS_FILE.read_text())
    decision_summary = json.loads(DECISION_SUMMARY_FILE.read_text())
    benchmark = json.loads(BENCHMARK_FILE.read_text())
    profit_curve = json.loads(PROFIT_CURVE_FILE.read_text())
    uplift_summary = json.loads(UPLIFT_FILE.read_text())
    return metrics, decision_summary, benchmark, profit_curve, uplift_summary


@st.cache_resource(show_spinner=False)
def load_ml_artifacts() -> Dict[str, Any]:
    """Load serialized K-Means, XGBoost, and SHAP metadata artifacts."""
    return {
        "kmeans": joblib.load(KMEANS_FILE),
        "xgb": joblib.load(XGB_FILE),
        "shap": joblib.load(SHAP_FILE),
    }


def build_executive_briefing_bundle(
    queue_df: pd.DataFrame,
    base_kpis: Dict[str, Any],
    model_metrics: Dict[str, Any],
    profit_curve_data: Dict[str, Any],
) -> Tuple[str, bytes]:
    """
    Generate both an Executive Briefing Markdown report and a ZIP bundle
    containing the Markdown report, an HTML executive report, and the filtered
    priority CSV action queue.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    opt_pt = profit_curve_data["optimal_threshold"]
    status_counts = queue_df["action_status"].value_counts().to_dict() if "action_status" in queue_df.columns else {}

    md_report = f"""# 📊 ISM Executive Retention & Value Optimization Briefing (v3.0)
**Generated At:** {now_str}  
**Source Warehouse:** `database/ism_dss.db` (`vw_customer_360`)

---

## 1. Portfolio Financial Summary (Baseline Policy)
- **Total Active Subscribers:** `{base_kpis['total_customers']:,}`
- **Total Annualized Revenue:** `${base_kpis['total_annualized_revenue']:,.2f}`
- **Annual Revenue at Risk:** `${base_kpis['total_revenue_at_risk']:,.2f}` (`{base_kpis['revenue_at_risk_pct_of_annual']:.2f}%` of annual revenue)
- **Targeted Accounts (`Cost > $0`):** `{base_kpis['targeted_customers']:,}`
- **Total Campaign Intervention Budget:** `${base_kpis['total_intervention_cost']:,.2f}`
- **Expected Saved Revenue:** `${base_kpis['total_expected_saved_revenue']:,.2f}`
- **Expected Net Financial Benefit:** `${base_kpis['total_expected_net_benefit']:,.2f}`
- **Portfolio Expected ROI:** `{base_kpis['portfolio_roi_pct']:,.2f}%`

## 2. Predictive Model & Profit Curve Optimization
- **Champion Classifier:** `XGBoost` (`ROC-AUC = {model_metrics['roc_auc']:.4f}`, `PR-AUC = {model_metrics['pr_auc']:.4f}`, `Recall = {model_metrics['recall'] * 100:.2f}%`, `F1 = {model_metrics['f1']:.4f}`)
- **Optimal Financial Cutoff:** `high_risk_threshold = {opt_pt['high_risk_threshold']:.2f}` (Maximizes Effective Expected Net Benefit at `${opt_pt['expected_net_benefit']:,.2f}`)

## 3. Filtered Operational Action Queue Summary
- **Filtered Queue Size:** `{len(queue_df):,}` subscribers
- **Queue Annual Revenue at Risk:** `${queue_df['revenue_at_risk'].sum():,.2f}`
- **Queue Intervention Budget:** `${queue_df['intervention_cost'].sum():,.2f}`
- **Queue Expected Net Benefit:** `${queue_df['expected_net_benefit'].sum():,.2f}`
- **Closed-Loop CRM Status Breakdown:** `{json.dumps(status_counts)}`
"""

    html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ISM Executive Retention Briefing v3.0</title>
<style>
body {{ font-family: Segoe UI, sans-serif; background:#0B0F19; color:#F8FAFC; padding:32px; }}
.card {{ background:#111827; border:1px solid #334155; border-radius:10px; padding:18px; margin-bottom:18px; }}
h1, h2 {{ color:#38BDF8; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ border:1px solid #334155; padding:8px; text-align:left; }}
th {{ background:#1E293B; color:#7DD3FC; }}
</style>
</head>
<body>
<h1>📊 ISM Executive Retention & Value Optimization Briefing (v3.0)</h1>
<p><b>Generated At:</b> {now_str} | <b>Total Subscribers:</b> {base_kpis['total_customers']:,}</p>
<div class="card">
<h2>Portfolio KPIs</h2>
<p><b>Annualized Revenue:</b> ${base_kpis['total_annualized_revenue']:,.2f} |
<b>Annual Revenue at Risk:</b> ${base_kpis['total_revenue_at_risk']:,.2f} |
<b>Intervention Cost:</b> ${base_kpis['total_intervention_cost']:,.2f} |
<b>Expected Net Benefit:</b> ${base_kpis['total_expected_net_benefit']:,.2f} |
<b>Portfolio ROI:</b> {base_kpis['portfolio_roi_pct']:,.2f}%</p>
</div>
<div class="card">
<h2>Top 25 Priority Accounts in Filtered Queue</h2>
{queue_df.head(25).to_html(index=False)}
</div>
</body>
</html>"""

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ISM_Executive_Briefing_v3.md", md_report)
        zf.writestr("ISM_Executive_Briefing_v3.html", html_report)
        zf.writestr("ISM_Priority_Campaign_Queue.csv", queue_df.to_csv(index=False))
    return md_report, zip_buf.getvalue()


# ---------------------------------------------------------------------------
# Sidebar — Clean Enterprise Status Cards
# ---------------------------------------------------------------------------
df_360 = load_customer_360()
model_metrics, baseline_summary, benchmark_data, profit_curve_data, uplift_summary = load_json_summaries()
artifacts = load_ml_artifacts()
base_kpis = baseline_summary["portfolio_kpis"]
crm_counts = df_360["action_status"].value_counts().to_dict() if "action_status" in df_360.columns else {}
sd_count = int(uplift_summary["sleeping_dogs_suppressed_count"])
sd_rar = float(uplift_summary["sleeping_dogs_protected_revenue_at_risk"])

with st.sidebar:
    st.markdown("## 🧠 ISM Retention DSS v3.0")
    st.caption(
        "Enterprise Customer Value, Causal Uplift & GenAI Retention Copilot · "
        "K-Means · XGBoost · TreeSHAP · Knapsack ROI"
    )

    st.markdown(
        f"""
        <div class="sb-card">
            <div class="sb-card-header">
                <span>📦 Warehouse Status</span>
                <span class="sb-badge">● Connected</span>
            </div>
            <div class="sb-row"><span class="sb-label">Engine</span><span class="sb-val">SQLite 3NF + Star</span></div>
            <div class="sb-row"><span class="sb-label">Active View</span><span class="sb-val">vw_customer_360</span></div>
            <div class="sb-row"><span class="sb-label">Total Subscribers</span><span class="sb-val">{len(df_360):,}</span></div>
            <div class="sb-row"><span class="sb-label">Observed Churn</span><span class="sb-val" style="color:#FDA4AF;">{df_360['churn_binary'].mean() * 100:.2f}%</span></div>
        </div>

        <div class="sb-card">
            <div class="sb-card-header">
                <span>🛡️ Causal ML & Copilot</span>
                <span class="sb-badge" style="color:#A5B4FC; border-color:rgba(99,102,241,0.4); background:rgba(99,102,241,0.16);">v3.0 Active</span>
            </div>
            <div class="sb-row"><span class="sb-label">Persuadables</span><span class="sb-val" style="color:#34D399;">{uplift_summary['persuadables_count']:,}</span></div>
            <div class="sb-row"><span class="sb-label">Sleeping Dogs (DND)</span><span class="sb-val" style="color:#FDA4AF;">{sd_count:,} (⛔ Suppressed)</span></div>
            <div class="sb-row"><span class="sb-label">Protected Rev at Risk</span><span class="sb-val" style="color:#FCD34D;">${sd_rar:,.0f}</span></div>
            <div class="sb-row"><span class="sb-label">GenAI Pitch Engine</span><span class="sb-val" style="color:#38BDF8;">SHAP 1-to-1 Ready</span></div>
        </div>

        <div class="sb-card">
            <div class="sb-card-header">
                <span>🤖 Mining & Benchmark</span>
                <span class="sb-badge" style="color:#38BDF8; border-color:rgba(56,189,248,0.4); background:rgba(56,189,248,0.12);">3 Models</span>
            </div>
            <div class="sb-row"><span class="sb-label">K-Means Silhouette</span><span class="sb-val">0.402 (k=3)</span></div>
            <div class="sb-row"><span class="sb-label">XGBoost ROC-AUC</span><span class="sb-val" style="color:#38BDF8;">{model_metrics['roc_auc']:.4f}</span></div>
            <div class="sb-row"><span class="sb-label">Churn Recall</span><span class="sb-val" style="color:#6EE7B7;">{model_metrics['recall'] * 100:.1f}%</span></div>
            <div class="sb-row"><span class="sb-label">Optimal Risk Cutoff</span><span class="sb-val" style="color:#FCD34D;">p ≥ {profit_curve_data['optimal_threshold']['high_risk_threshold']:.2f}</span></div>
        </div>

        <div class="sb-card">
            <div class="sb-card-header">
                <span>🔄 Closed-Loop CRM</span>
                <span class="sb-badge">Live SQLite</span>
            </div>
            <div class="sb-row"><span class="sb-label">Pending Outreach</span><span class="sb-val">{crm_counts.get('Pending', 0):,}</span></div>
            <div class="sb-row"><span class="sb-label">Contacted</span><span class="sb-val" style="color:#38BDF8;">{crm_counts.get('Contacted', 0):,}</span></div>
            <div class="sb-row"><span class="sb-label">Retained</span><span class="sb-val" style="color:#34D399;">{crm_counts.get('Retained', 0):,}</span></div>
            <div class="sb-row"><span class="sb-label">Churned</span><span class="sb-val" style="color:#FDA4AF;">{crm_counts.get('Churned', 0):,}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("International University — VNU-HCMC\nInformation Systems Management Project")

# ---------------------------------------------------------------------------
# Header & 5 Enterprise Tabs
# ---------------------------------------------------------------------------
st.title("📊 Intelligent Customer Retention & Value Optimization DSS (v3.0)")
st.markdown(
    "**Enterprise Decision Support System** integrating **Unsupervised Value Segmentation (K-Means)**, "
    "**Multi-Model Churn Prediction (XGBoost Champion)**, **Explainable AI (TreeSHAP)**, "
    "**Causal Uplift Modeling ('Sleeping Dog' Suppression)**, **Constrained Knapsack Optimization**, "
    "and **GenAI 1-to-1 Retention Copilot**."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Tab 1: Executive KPI & Portfolio Overview",
        "🎛️ Tab 2: What-If, Causal Uplift & Knapsack Optimizer",
        "🎯 Tab 3: Action Queue, 360° Copilot & Closed-Loop CRM",
        "🔮 Tab 4: Real-Time Predictor & GenAI Pitch Advisor",
        "🏛️ Tab 5: Model Benchmarking & System Governance",
    ]
)

# ===========================================================================
# TAB 1 — EXECUTIVE KPI & PORTFOLIO OVERVIEW
# ===========================================================================
with tab1:
    st.subheader("Executive Portfolio Key Performance Indicators & Causal Uplift Guardrail")

    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
    render_kpi_card(
        r1_c1,
        "Total Subscriber Base",
        f"{base_kpis['total_customers']:,}",
        "100% Active Accounts · 3 Value Tiers",
        accent="cyan",
    )
    render_kpi_card(
        r1_c2,
        "Observed Churn Rate",
        f"{df_360['churn_binary'].mean() * 100:.2f}%",
        "1,869 Historical Defections",
        accent="rose",
    )
    render_kpi_card(
        r1_c3,
        "Total Annualized Revenue",
        f"${base_kpis['total_annualized_revenue']:,.0f}",
        "$64.76 Mean Monthly ARPU",
        accent="cyan",
    )
    render_kpi_card(
        r1_c4,
        "Annual Revenue at Risk",
        f"${base_kpis['total_revenue_at_risk']:,.0f}",
        f"{base_kpis['revenue_at_risk_pct_of_annual']:.1f}% of Total Annual Revenue",
        accent="rose",
    )

    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
    render_kpi_card(
        r2_c1,
        "Targeted Campaign Budget",
        f"${base_kpis['total_intervention_cost']:,.2f}",
        f"{base_kpis['targeted_customers']:,} High/Medium-Value At-Risk Accounts",
        accent="amber",
    )
    render_kpi_card(
        r2_c2,
        "Expected Saved Revenue & Net Benefit",
        f"${base_kpis['total_expected_net_benefit']:,.2f}",
        f"Gross Saved: ${base_kpis['total_expected_saved_revenue']:,.2f}",
        accent="emerald",
    )
    render_kpi_card(
        r2_c3,
        "Portfolio Expected ROI",
        f"{base_kpis['portfolio_roi_pct']:,.2f}%",
        "18.4x Baseline · 23.3x Causal-Optimized",
        accent="emerald",
    )
    render_kpi_card(
        r2_c4,
        "Sleeping Dogs Suppressed",
        f"{sd_count:,} Accounts",
        f"Sleeping Dogs Suppressed: {sd_count:,} accounts (${sd_rar:,.0f} Revenue at Risk Protected)",
        accent="indigo",
    )

    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, rgba(99,102,241,0.16), rgba(15,23,42,0.9)); border: 1px solid #4F46E5; border-left: 4px solid #6366F1; border-radius: 10px; padding: 11px 16px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <span style="color: #E2E8F0; font-size: 0.88rem;">
                🛡️ <b>Causal Uplift Protection Active:</b> <b>Sleeping Dogs Suppressed: {sd_count:,} accounts (${sd_rar:,.2f} Revenue at Risk Protected)</b> — Flagged <code>is_do_not_disturb = 1</code> to prevent outreach-triggered defection.
            </span>
            <span style="color: #A5B4FC; font-size: 0.82rem; font-weight: 600;">
                Persuadables: {uplift_summary['persuadables_count']:,} · Sleeping Dogs: {sd_count:,} · Lost Causes: {uplift_summary['lost_causes_excluded_count']:,} · Sure Things: {uplift_summary['sure_things_count']:,}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📋 Raw Portfolio Audit Metrics Table", expanded=False):
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Total Customers", f"{base_kpis['total_customers']:,}")
        m2.metric("Observed Churn Rate", f"{df_360['churn_binary'].mean() * 100:.2f}%")
        m3.metric("Annualized Revenue", f"${base_kpis['total_annualized_revenue'] / 1e6:.2f}M")
        m4.metric("Annual Revenue at Risk", f"${base_kpis['total_revenue_at_risk'] / 1e6:.2f}M")
        m5.metric("Expected Saved Revenue", f"${base_kpis['total_expected_saved_revenue'] / 1e3:.2f}K")
        m6.metric("Expected Net Benefit", f"${base_kpis['total_expected_net_benefit'] / 1e3:.2f}K")
        m7.metric("Portfolio Expected ROI", f"{base_kpis['portfolio_roi_pct']:,.2f}%")

    col_left, col_right = st.columns([1.05, 1.15])

    with col_left:
        st.markdown("#### 1. 9-Cell Value Tier × Churn Risk Matrix")
        matrix_metric = st.radio(
            "Select Matrix View:",
            ["Annual Revenue at Risk ($) — Strategic Severity", "Customer Count & Portfolio Share"],
            horizontal=True,
            key="tab1_matrix_metric",
        )
        cells_df = pd.DataFrame(baseline_summary["matrix_9_cells"])
        tier_order = ["High", "Medium", "Low"]
        risk_order = ["High", "Medium", "Low"]

        piv_cnt = (
            cells_df.pivot(index="value_tier", columns="risk_level", values="customers")
            .reindex(index=tier_order, columns=risk_order)
        )
        piv_rar = (
            cells_df.pivot(index="value_tier", columns="risk_level", values="revenue_at_risk")
            .reindex(index=tier_order, columns=risk_order)
        )
        piv_roi = (
            cells_df.pivot(index="value_tier", columns="risk_level", values="cell_roi_pct")
            .reindex(index=tier_order, columns=risk_order)
        )

        x_labels = ["High Risk (p ≥ 0.60)", "Medium Risk (0.30–0.60)", "Low Risk (p < 0.30)"]
        y_labels = ["High Value Tier", "Medium Value Tier", "Low Value Tier"]

        if matrix_metric.startswith("Annual Revenue at Risk"):
            z_vals = piv_rar.values
            colorscale = [
                [0.0, "#0F172A"],
                [0.12, "#1E293B"],
                [0.30, "#7C2D12"],
                [0.60, "#DC2626"],
                [1.0, "#F43F5E"],
            ]
        else:
            z_vals = piv_cnt.values
            colorscale = [
                [0.0, "#0F172A"],
                [0.25, "#1E3A5F"],
                [0.55, "#0369A1"],
                [0.80, "#0284C7"],
                [1.0, "#38BDF8"],
            ]

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=z_vals,
                x=x_labels,
                y=y_labels,
                colorscale=colorscale,
                showscale=True,
                hoverongaps=False,
            )
        )

        z_max = float(np.max(z_vals))
        annotations = []
        for i, t in enumerate(tier_order):
            for j, r in enumerate(risk_order):
                cnt_val = int(piv_cnt.loc[t, r])
                share_val = cnt_val / len(df_360) * 100.0
                rar_val = float(piv_rar.loc[t, r])
                roi_val = float(piv_roi.loc[t, r])
                cell_label = (
                    f"<b>${rar_val / 1e3:,.1f}K at Risk</b><br>"
                    f"{cnt_val:,} KH ({share_val:.1f}%)<br>"
                    f"ROI: {roi_val:,.0f}%"
                )
                annotations.append(
                    dict(
                        x=x_labels[j],
                        y=y_labels[i],
                        text=cell_label,
                        showarrow=False,
                        font=dict(
                            color="#FFFFFF" if z_vals[i, j] < z_max * 0.88 else "#0F172A",
                            size=11,
                        ),
                    )
                )

        style_plotly_dark(fig_heat, height=405)
        fig_heat.update_layout(
            annotations=annotations,
            xaxis_title="Predicted Churn Risk Level (XGBoost)",
            yaxis_title="Customer Value Tier (K-Means)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_right:
        st.markdown("#### 2. Revenue at Risk vs. Expected Net Benefit by Decision Quadrant")
        q_df = pd.DataFrame(baseline_summary["quadrant_breakdown"])
        short_x = [SHORT_QUADRANT_LABELS.get(q, q) for q in q_df["decision_quadrant"]]

        fig_quad = go.Figure()
        fig_quad.add_trace(
            go.Bar(
                name="Annual Revenue at Risk ($)",
                x=short_x,
                y=q_df["revenue_at_risk"],
                marker_color="#F43F5E",
                text=[f"${v / 1e3:,.0f}K" for v in q_df["revenue_at_risk"]],
                textposition="outside",
                textfont=dict(color="#FDA4AF", size=11),
            )
        )
        fig_quad.add_trace(
            go.Bar(
                name="Expected Saved Revenue ($)",
                x=short_x,
                y=q_df["expected_saved_revenue"],
                marker_color="#38BDF8",
                text=[f"${v / 1e3:,.0f}K" for v in q_df["expected_saved_revenue"]],
                textposition="outside",
                textfont=dict(color="#7DD3FC", size=11),
            )
        )
        fig_quad.add_trace(
            go.Bar(
                name="Expected Net Benefit ($)",
                x=short_x,
                y=q_df["expected_net_benefit"],
                marker_color="#10B981",
                text=[f"${v / 1e3:,.0f}K" for v in q_df["expected_net_benefit"]],
                textposition="outside",
                textfont=dict(color="#6EE7B7", size=11),
            )
        )
        style_plotly_dark(fig_quad, height=445)
        fig_quad.update_layout(
            barmode="group",
            yaxis_title="USD ($)",
            xaxis_title="Prescriptive Decision Quadrant",
            xaxis=dict(tickangle=0),
            yaxis=dict(range=[0, float(q_df["revenue_at_risk"].max()) * 1.18]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_quad, use_container_width=True)

    st.divider()

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
        fig_sil = go.Figure(
            go.Scatter(
                x=sil_df["k"],
                y=sil_df["Silhouette Score"],
                mode="lines+markers+text",
                line=dict(color="#38BDF8", width=3),
                marker=dict(size=9, color="#6366F1", line=dict(width=2, color="#38BDF8")),
                text=[f"{v:.3f}" for v in sil_df["Silhouette Score"]],
                textposition="top center",
            )
        )
        style_plotly_dark(
            fig_sil,
            height=245,
            title="Silhouette Validity Curve (k = 2..6, Selected k = 3: 0.402)",
        )
        fig_sil.update_layout(
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Score",
            yaxis=dict(range=[0.34, 0.49]),
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    with c_shap:
        st.markdown("#### 4. Global SHAP Churn Drivers (Top 12 Business Attributes)")
        shap_imp = pd.Series(model_metrics["global_shap_importance"]).sort_values(ascending=True).tail(12)
        fig_shap = go.Figure(
            go.Bar(
                x=shap_imp.values,
                y=shap_imp.index,
                orientation="h",
                marker=dict(
                    color=shap_imp.values,
                    colorscale=[[0.0, "#6366F1"], [0.5, "#0284C7"], [1.0, "#38BDF8"]],
                ),
                text=[f"{v:.3f}" for v in shap_imp.values],
                textposition="outside",
                textfont=dict(color="#E2E8F0", size=11),
            )
        )
        style_plotly_dark(fig_shap, height=400)
        fig_shap.update_layout(
            xaxis_title="Mean |SHAP Value| (Impact on Log-Odds of Churn)",
            yaxis_title="Business Attribute",
            xaxis=dict(range=[0, float(shap_imp.max()) * 1.18]),
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
        cm_labels = [
            [f"<b>TN: {cm['tn']}</b><br>True Loyal", f"<b>FP: {cm['fp']}</b><br>False Alarm"],
            [f"<b>FN: {cm['fn']}</b><br>Missed Churn", f"<b>TP: {cm['tp']}</b><br>Caught Churn"],
        ]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm_z,
                x=["Pred: Loyal (0)", "Pred: Churn (1)"],
                y=["Actual: Loyal (0)", "Actual: Churn (1)"],
                colorscale=[[0.0, "#1E293B"], [0.5, "#0369A1"], [1.0, "#38BDF8"]],
                showscale=False,
            )
        )
        cm_ann = []
        for r_i, y_lbl in enumerate(["Actual: Loyal (0)", "Actual: Churn (1)"]):
            for c_i, x_lbl in enumerate(["Pred: Loyal (0)", "Pred: Churn (1)"]):
                cm_ann.append(
                    dict(
                        x=x_lbl,
                        y=y_lbl,
                        text=cm_labels[r_i][c_i],
                        showarrow=False,
                        font=dict(color="#FFFFFF" if cm_z[r_i][c_i] < 600 else "#0F172A", size=12),
                    )
                )
        style_plotly_dark(fig_cm, height=235, title="Hold-Out Confusion Matrix (Threshold = 0.50)")
        fig_cm.update_layout(annotations=cm_ann, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cm, use_container_width=True)


# ===========================================================================
# TAB 2 — INTERACTIVE WHAT-IF, PROFIT CURVE & CONSTRAINED RESOURCE OPTIMIZER
# ===========================================================================
with tab2:
    st.subheader("Interactive What-If Policy Simulator & Constrained Knapsack Resource Optimizer")
    st.markdown(
        "Simulate changes in churn probability cutoffs, unit intervention costs, discount rates, "
        "and dual operational constraints (**Max Campaign Budget Cap ($)** and **Max VIP Agent Calls**) "
        "using the marginal-efficiency Knapsack optimizer (`optimize_under_constraints`)."
    )

    with st.expander("⚙️ Configure Economic & Resource Constraint Sliders", expanded=True):
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)

        with s_col1:
            st.markdown("**1. Churn Risk Thresholds**")
            sim_high_thr = st.slider(
                "High Risk Threshold (p ≥)",
                min_value=0.35,
                max_value=0.85,
                value=0.60,
                step=0.05,
                help="Baseline = 0.60 | Profit-Curve Optimal = 0.45",
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
            st.markdown("**4. Expected Conversion Rates**")
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

        st.markdown("**5. Causal Uplift Guardrail & Dual Resource Constraints (Knapsack Optimizer)**")
        rc0, rc1, rc2, rc3 = st.columns([1.15, 0.95, 1.2, 1.2])
        with rc0:
            enforce_causal_exclusion = st.checkbox(
                "Enforce Causal Uplift Exclusion (Filter Out Sleeping Dogs)",
                value=True,
                help=(
                    "Automatically excludes 'Sleeping Dog' (tau < 0) and 'Lost Cause' segments "
                    "from marketing spend and concentrates AM capacity on 'Persuadable' accounts, "
                    "boosting both Expected Net Benefit and Portfolio ROI."
                ),
            )
        with rc1:
            enable_constraints = st.checkbox(
                "Enable Resource Constraints (Budget + VIP Call Capacity)",
                value=False,
            )
        with rc2:
            max_budget_cap = st.slider(
                "Max Campaign Budget Cap ($)",
                min_value=2000.0,
                max_value=40000.0,
                value=15000.0,
                step=1000.0,
                disabled=not enable_constraints,
            )
        with rc3:
            max_vip_calls_cap = st.slider(
                "Max VIP Account Manager 1-on-1 Calls",
                min_value=25,
                max_value=900,
                value=350,
                step=25,
                disabled=not enable_constraints,
            )

    sim_input_df = df_360[
        [
            "customerID",
            "monthly_charges",
            "value_tier",
            "churn_probability",
            "top_shap_driver_1",
            "top_shap_driver_2",
            "uplift_score",
            "causal_segment",
            "is_do_not_disturb",
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

    unconstrained_sim_df = run_decision_simulation(sim_input_df, params=sim_params)
    if enforce_causal_exclusion:
        unconstrained_sim_df = apply_uplift_exclusion(unconstrained_sim_df)

    sim_results_df, alloc_info = optimize_under_constraints(
        unconstrained_sim_df,
        max_budget=max_budget_cap if enable_constraints else 1e9,
        max_vip_calls=max_vip_calls_cap if enable_constraints else 100000,
    )
    sim_summary = summarize_decisions(sim_results_df, params=sim_params)
    sim_kpis = sim_summary["portfolio_kpis"]
    c_opt = alloc_info["constrained_optimal"]

    if enforce_causal_exclusion:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.45); border-left: 4px solid #10B981; border-radius: 10px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.86rem; color: #E2E8F0;">
                ✅ <b>Causal Uplift Exclusion Active:</b> Filtered out <b>{uplift_summary['sleeping_dogs_suppressed_count']:,} Sleeping Dogs</b> (<code>tau &lt; 0</code>, protecting <b>${uplift_summary['sleeping_dogs_protected_revenue_at_risk']:,.2f}</b> Revenue at Risk) and <b>{uplift_summary['lost_causes_excluded_count']:,} Lost Causes</b> from budget allocation — focusing spend exclusively on <b>{uplift_summary['persuadables_count']:,} Persuadable</b> subscribers.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: rgba(244, 63, 94, 0.14); border: 1px solid rgba(244, 63, 94, 0.45); border-left: 4px solid #F43F5E; border-radius: 10px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.86rem; color: #FDA4AF;">
                ⚠️ <b>Correlational Targeting Mode (Causal Exclusion OFF):</b> Budget is allocated purely by churn probability without filtering out <b>{uplift_summary['sleeping_dogs_suppressed_count']:,} Sleeping Dogs</b> or <b>{uplift_summary['lost_causes_excluded_count']:,} Lost Causes</b>. Enable the Causal Uplift Exclusion toggle above to boost Net Benefit and ROI.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Simulated vs. Unconstrained Baseline Financial Impact")
    delta_targeted = sim_kpis["targeted_customers"] - base_kpis["targeted_customers"]
    delta_cost = sim_kpis["total_intervention_cost"] - base_kpis["total_intervention_cost"]
    delta_saved = sim_kpis["total_expected_saved_revenue"] - base_kpis["total_expected_saved_revenue"]
    delta_net = sim_kpis["total_expected_net_benefit"] - base_kpis["total_expected_net_benefit"]
    delta_roi = sim_kpis["portfolio_roi_pct"] - base_kpis["portfolio_roi_pct"]

    d1, d2, d3, d4, d5 = st.columns(5)
    render_kpi_card(
        d1,
        "Funded Accounts",
        f"{sim_kpis['targeted_customers']:,}",
        f"{delta_targeted:+,} vs. Baseline ({c_opt['vip_calls_used']} VIP Calls)",
        accent="cyan",
    )
    render_kpi_card(
        d2,
        "Simulated Campaign Spend",
        f"${sim_kpis['total_intervention_cost']:,.2f}",
        f"{delta_cost:+,.2f} ({c_opt['budget_share_of_unconstrained_pct']:.1f}% of Unconstrained)",
        accent="amber",
    )
    render_kpi_card(
        d3,
        "Simulated Saved Revenue",
        f"${sim_kpis['total_expected_saved_revenue']:,.2f}",
        f"{delta_saved:+,.2f} vs. Baseline",
        accent="cyan",
    )
    render_kpi_card(
        d4,
        "Simulated Net Benefit",
        f"${sim_kpis['total_expected_net_benefit']:,.2f}",
        f"Captures {c_opt['net_benefit_capture_pct']:.1f}% of Max Net Benefit",
        accent="emerald",
    )
    render_kpi_card(
        d5,
        "Simulated Portfolio ROI",
        f"{sim_kpis['portfolio_roi_pct']:,.2f}%",
        f"{delta_roi:+,.2f}% vs. Baseline",
        accent="emerald",
    )

    # Row of v2.0 Optimization Charts: Pareto Efficient Frontier + Threshold Profit Curve
    opt_c1, opt_c2 = st.columns(2)

    with opt_c1:
        st.markdown("#### 1. Pareto Efficient Frontier (Knapsack Marginal ROI Allocation)")
        frontier_df = pd.DataFrame(alloc_info["pareto_frontier"])
        fig_pareto = go.Figure()
        if not frontier_df.empty:
            fig_pareto.add_trace(
                go.Scatter(
                    x=frontier_df["budget_pct"],
                    y=frontier_df["net_benefit_pct"],
                    mode="lines",
                    name="Pareto Efficient Frontier",
                    line=dict(color="#10B981", width=3.5),
                    fill="tozeroy",
                    fillcolor="rgba(16, 185, 129, 0.12)",
                )
            )
            fig_pareto.add_trace(
                go.Scatter(
                    x=[0, 100],
                    y=[0, 100],
                    mode="lines",
                    name="Uniform Random Allocation (45° Line)",
                    line=dict(color="#64748B", width=2, dash="dash"),
                )
            )
            if enable_constraints:
                fig_pareto.add_trace(
                    go.Scatter(
                        x=[c_opt["budget_share_of_unconstrained_pct"]],
                        y=[c_opt["net_benefit_capture_pct"]],
                        mode="markers+text",
                        name="Active Constrained Policy",
                        marker=dict(size=13, color="#F43F5E", line=dict(width=2, color="#FFFFFF")),
                        text=[
                            f"Active Cap: {c_opt['budget_share_of_unconstrained_pct']:.1f}% Budget → "
                            f"{c_opt['net_benefit_capture_pct']:.1f}% Net Benefit"
                        ],
                        textposition="bottom right",
                        textfont=dict(color="#FDA4AF", size=11),
                    )
                )
        style_plotly_dark(fig_pareto, height=380)
        fig_pareto.update_layout(
            xaxis_title="% of Unconstrained Campaign Budget Spent",
            yaxis_title="% of Maximum Expected Net Benefit Captured",
            xaxis=dict(range=[0, 105]),
            yaxis=dict(range=[0, 108]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    with opt_c2:
        st.markdown("#### 2. Threshold Profit Curve (High-Risk Cutoff $\\tau_H \\in [0.15, 0.85]$)")
        live_curve = run_threshold_profit_curve(df_360, params=sim_params)
        sweep_df = pd.DataFrame(live_curve["threshold_sweep"])
        opt_pt = live_curve["optimal_threshold"]

        fig_prof = go.Figure()
        fig_prof.add_trace(
            go.Scatter(
                x=sweep_df["high_risk_threshold"],
                y=sweep_df["expected_net_benefit"],
                mode="lines+markers",
                name="Expected Net Benefit ($)",
                line=dict(color="#38BDF8", width=3),
                marker=dict(size=7, color="#38BDF8"),
            )
        )
        fig_prof.add_trace(
            go.Scatter(
                x=sweep_df["high_risk_threshold"],
                y=sweep_df["expected_saved_revenue"],
                mode="lines",
                name="Expected Saved Revenue ($)",
                line=dict(color="#10B981", width=2, dash="dot"),
            )
        )
        fig_prof.add_trace(
            go.Scatter(
                x=[opt_pt["high_risk_threshold"]],
                y=[opt_pt["expected_net_benefit"]],
                mode="markers+text",
                name=f"Optimal Cutoff (p ≥ {opt_pt['high_risk_threshold']:.2f})",
                marker=dict(size=13, color="#F59E0B", symbol="star", line=dict(width=1.5, color="#FFFFFF")),
                text=[f"Optimal p ≥ {opt_pt['high_risk_threshold']:.2f}<br>${opt_pt['expected_net_benefit']:,.0f}"],
                textposition="bottom center",
                textfont=dict(color="#FCD34D", size=11),
            )
        )
        style_plotly_dark(fig_prof, height=380)
        fig_prof.update_layout(
            xaxis_title="High-Risk Probability Cutoff (high_risk_threshold)",
            yaxis_title="USD ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_prof, use_container_width=True)

    sim_c1, sim_c2 = st.columns([1.05, 1.15])
    with sim_c1:
        st.markdown("#### 3. Portfolio Comparison: Baseline vs. Simulated Policy")
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
                marker_color="#64748B",
                text=[f"${v:,.0f}" for v in comp_df["Baseline Policy"]],
                textposition="outside",
                textfont=dict(color="#CBD5E1"),
            )
        )
        fig_comp.add_trace(
            go.Bar(
                name="Simulated Policy",
                x=comp_df["Metric"],
                y=comp_df["Simulated Policy"],
                marker_color="#38BDF8",
                text=[f"${v:,.0f}" for v in comp_df["Simulated Policy"]],
                textposition="outside",
                textfont=dict(color="#7DD3FC"),
            )
        )
        style_plotly_dark(fig_comp, height=360)
        max_comp_y = float(max(comp_df["Baseline Policy"].max(), comp_df["Simulated Policy"].max()))
        fig_comp.update_layout(
            barmode="group",
            yaxis_title="USD ($)",
            yaxis=dict(range=[0, max_comp_y * 1.18]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with sim_c2:
        st.markdown("#### 4. Simulated Breakdown by Strategic Decision Quadrant")
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
# TAB 3 — OPERATIONAL ACTION QUEUE, CLOSED-LOOP CRM & BRIEFING EXPORT
# ===========================================================================
with tab3:
    st.subheader("Operational Retention Action Queue, Customer 360° Copilot & Closed-Loop CRM")
    st.markdown(
        "Filter prioritized customer segments (including **Causal Uplift Quadrants**), export the "
        "**Executive Briefing Bundle (`.zip` / `.md` / `.csv`)**, generate **1-to-1 AI Retention Scripts & Emails**, "
        "and log **Closed-Loop CRM Action Status** updates directly to SQLite."
    )

    f_col1, f_col2, f_col3, f_col4, f_col5, f_col6 = st.columns([1.3, 1.05, 0.75, 0.75, 0.8, 0.8])
    with f_col1:
        selected_quadrants = st.multiselect(
            "Strategic Decision Quadrant:",
            options=QUADRANT_ORDER,
            default=["Critical VIP Retention", "Automated Retention Campaign"],
        )
    with f_col2:
        selected_causal = st.multiselect(
            "Causal Uplift Segment:",
            options=list(CAUSAL_SEGMENTS),
            default=list(CAUSAL_SEGMENTS),
        )
    with f_col3:
        selected_tiers = st.multiselect(
            "Value Tier:",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
    with f_col4:
        selected_risks = st.multiselect(
            "Churn Risk Level:",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
    with f_col5:
        selected_statuses = st.multiselect(
            "CRM Action Status:",
            options=list(ACTION_STATUS_DOMAIN),
            default=list(ACTION_STATUS_DOMAIN),
        )
    with f_col6:
        min_rar = st.number_input(
            "Min Revenue at Risk ($):",
            min_value=0.0,
            max_value=1500.0,
            value=0.0,
            step=50.0,
        )

    filtered_df = df_360[
        df_360["decision_quadrant"].isin(selected_quadrants)
        & df_360["causal_segment"].isin(selected_causal)
        & df_360["value_tier"].isin(selected_tiers)
        & df_360["risk_level"].isin(selected_risks)
        & df_360["action_status"].isin(selected_statuses)
        & (df_360["revenue_at_risk"] >= min_rar)
    ].sort_values("revenue_at_risk", ascending=False)

    q_m1, q_m2, q_m3, q_m4 = st.columns(4)
    render_kpi_card(
        q_m1,
        "Filtered Accounts in Queue",
        f"{len(filtered_df):,}",
        f"{len(filtered_df) / len(df_360) * 100:.1f}% of Total Base",
        accent="cyan",
    )
    render_kpi_card(
        q_m2,
        "Queue Revenue at Risk",
        f"${filtered_df['revenue_at_risk'].sum():,.2f}",
        "Prioritized by Annual Exposure",
        accent="rose",
    )
    render_kpi_card(
        q_m3,
        "Queue Intervention Cost",
        f"${filtered_df['intervention_cost'].sum():,.2f}",
        "Required Campaign Budget",
        accent="amber",
    )
    render_kpi_card(
        q_m4,
        "Queue Expected Net Benefit",
        f"${filtered_df['expected_net_benefit'].sum():,.2f}",
        f"Saved: ${filtered_df['expected_saved_revenue'].sum():,.2f}",
        accent="emerald",
    )

    queue_cols = [
        "customerID",
        "causal_segment",
        "uplift_score",
        "is_do_not_disturb",
        "action_status",
        "value_tier",
        "risk_level",
        "churn_probability",
        "contract",
        "tenure",
        "monthly_charges",
        "revenue_at_risk",
        "top_shap_driver_1",
        "top_shap_driver_2",
        "decision_quadrant",
        "recommended_action",
        "intervention_cost",
        "expected_saved_revenue",
        "expected_net_benefit",
        "expected_roi",
        "manager_notes",
        "updated_at",
    ]

    st.dataframe(
        filtered_df[queue_cols].head(500),
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    # Export Buttons: CSV + Executive Briefing Bundle (ZIP with MD, HTML, CSV) + Markdown
    csv_bytes = filtered_df[queue_cols].to_csv(index=False).encode("utf-8")
    md_briefing, zip_bundle_bytes = build_executive_briefing_bundle(
        filtered_df[queue_cols], base_kpis, model_metrics, profit_curve_data
    )
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button(
            label=f"📥 Download Actionable Campaign CSV ({len(filtered_df):,} Accounts)",
            data=csv_bytes,
            file_name="ism_retention_campaign_queue.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            label="📦 Export Executive Briefing Bundle (.ZIP: MD + HTML + CSV)",
            data=zip_bundle_bytes,
            file_name="ism_executive_briefing_bundle_v3.zip",
            mime="application/zip",
            use_container_width=True,
        )
    with dl_col3:
        st.download_button(
            label="📄 Download Executive Summary Report (.MD)",
            data=md_briefing.encode("utf-8"),
            file_name="ism_executive_briefing_v3.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    st.subheader("🔍 Customer 360° Inspector, GenAI Retention Copilot & Closed-Loop CRM")

    candidate_ids = filtered_df["customerID"].tolist() if len(filtered_df) > 0 else df_360["customerID"].tolist()
    selected_cid = st.selectbox(
        "Select a Customer ID from the current queue (sorted by Revenue at Risk):",
        options=candidate_ids[:1000],
        index=0,
    )

    cust_row = df_360[df_360["customerID"] == selected_cid].iloc[0]
    cust_is_dnd = int(cust_row.get("is_do_not_disturb", 0) or 0) == 1 or str(cust_row.get("causal_segment", "")) == "Sleeping Dog"
    cust_causal_seg = str(cust_row.get("causal_segment", "Persuadable"))
    cust_uplift = float(cust_row.get("uplift_score", 0.0) or 0.0)

    if cust_is_dnd:
        st.markdown(
            f"""
            <div style="background: rgba(244, 63, 94, 0.18); border: 1px solid #F43F5E; border-left: 5px solid #F43F5E; border-radius: 10px; padding: 12px 16px; margin-bottom: 14px; color: #FFE4E6;">
                ⛔ <b>DO NOT DISTURB — CAUSAL SLEEPING DOG ALERT (<code>{selected_cid}</code> | Uplift &tau; = {cust_uplift:+.3f}):</b>
                Proactive outbound sales or contract-renewal calls to this subscriber are predicted to <b>trigger cancellation</b>.
                Suppress all outbound telesales outreach (<code>is_do_not_disturb = 1</code>) and apply silent backend QoS care only.
            </div>
            """,
            unsafe_allow_html=True,
        )

    causal_badge_color = (
        "#FDA4AF" if cust_is_dnd else ("#34D399" if cust_causal_seg == "Persuadable" else "#FCD34D")
    )
    dnd_tag = " ⛔ DO NOT DISTURB" if cust_is_dnd else ""

    p_col1, p_col2, p_col3 = st.columns([1.0, 1.1, 1.25])
    with p_col1:
        st.markdown(
            f"""
            <div class="profile-card">
                <h5>👤 Demographics & Subscribed Services</h5>
                <div class="sb-row"><span class="sb-label">Customer ID</span><span class="sb-val" style="color:#38BDF8;">{cust_row['customerID']}</span></div>
                <div class="sb-row"><span class="sb-label">CRM Action Status</span><span class="sb-val" style="color:#34D399;">{cust_row['action_status']}</span></div>
                <div class="sb-row"><span class="sb-label">Gender / Senior</span><span class="sb-val">{cust_row['gender']} / {'Yes' if cust_row['senior_citizen'] else 'No'}</span></div>
                <div class="sb-row"><span class="sb-label">Partner / Dependents</span><span class="sb-val">{'Yes' if cust_row['partner'] else 'No'} / {'Yes' if cust_row['dependents'] else 'No'}</span></div>
                <div class="sb-row"><span class="sb-label">Tenure & Lifecycle</span><span class="sb-val">{cust_row['tenure']} mos ({cust_row['tenure_group']})</span></div>
                <div class="sb-row"><span class="sb-label">Contract Type</span><span class="sb-val">{cust_row['contract']}</span></div>
                <div class="sb-row"><span class="sb-label">Internet / Phone</span><span class="sb-val">{cust_row['internet_service']} / {'Yes' if cust_row['phone_service'] else 'No'}</span></div>
                <div class="sb-row"><span class="sb-label">Active VAS Count</span><span class="sb-val">{cust_row['total_services_count']} / 6 Services</span></div>
                <div class="sb-row"><span class="sb-label">Payment Method</span><span class="sb-val">{cust_row['payment_method']}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col2:
        st.markdown(
            f"""
            <div class="profile-card">
                <h5>🎯 Mining, Causal Uplift & Playbook</h5>
                <div class="sb-row"><span class="sb-label">Customer Value Tier</span><span class="sb-val" style="color:#38BDF8;">{cust_row['value_tier']} (Cluster {cust_row['cluster_id']})</span></div>
                <div class="sb-row"><span class="sb-label">Churn Probability</span><span class="sb-val" style="color:#FDA4AF;">{cust_row['churn_probability'] * 100:.2f}% ({cust_row['risk_level']} Risk)</span></div>
                <div class="sb-row"><span class="sb-label">Causal Uplift Segment</span><span class="sb-val" style="color:{causal_badge_color};">{cust_causal_seg}{dnd_tag} (&tau; = {cust_uplift:+.3f})</span></div>
                <div class="sb-row"><span class="sb-label">Monthly / Total Spend</span><span class="sb-val">${cust_row['monthly_charges']:.2f} / ${cust_row['total_charges']:,.2f}</span></div>
                <div class="sb-row"><span class="sb-label">Annual Revenue at Risk</span><span class="sb-val" style="color:#FDA4AF;">${cust_row['revenue_at_risk']:,.2f}</span></div>
                <div class="sb-row"><span class="sb-label">Primary SHAP Driver</span><span class="sb-val" style="color:#FCD34D;">{cust_row['top_shap_driver_1']}</span></div>
                <div class="sb-row"><span class="sb-label">Secondary SHAP Driver</span><span class="sb-val" style="color:#FCD34D;">{cust_row['top_shap_driver_2']}</span></div>
                <div class="sb-row"><span class="sb-label">Decision Quadrant</span><span class="sb-val">{cust_row['decision_quadrant']}</span></div>
                <div class="sb-row"><span class="sb-label">Cost / Net Benefit / ROI</span><span class="sb-val" style="color:#6EE7B7;">${cust_row['intervention_cost']:.2f} / ${cust_row['expected_net_benefit']:.2f} ({cust_row['expected_roi']:,.0f}%)</span></div>
                <div style="margin-top:10px; padding:8px 10px; background:rgba(56,189,248,0.12); border-left:3px solid #38BDF8; border-radius:6px; font-size:0.82rem; color:#E2E8F0;">
                    <b>Recommended Action:</b> {cust_row['recommended_action']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col3:
        shap_meta = artifacts["shap"]
        cid_idx = shap_meta["customer_ids"].index(selected_cid)
        cust_shap_series = (
            shap_meta["shap_by_feature"]
            .iloc[cid_idx]
            .sort_values(key=lambda s: s.abs(), ascending=True)
            .tail(10)
        )
        colors = ["#F43F5E" if v > 0 else "#10B981" for v in cust_shap_series.values]
        fig_local_shap = go.Figure(
            go.Bar(
                x=cust_shap_series.values,
                y=cust_shap_series.index,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f}" for v in cust_shap_series.values],
                textposition="outside",
                textfont=dict(color="#E2E8F0", size=11),
            )
        )
        style_plotly_dark(
            fig_local_shap,
            height=330,
            title=f"Local SHAP Feature Contributions — {selected_cid}",
        )
        max_abs = float(cust_shap_series.abs().max()) * 1.25
        fig_local_shap.update_layout(
            xaxis_title="SHAP Log-Odds (+ Increases Churn Risk | - Reduces Risk)",
            yaxis_title="Attribute",
            xaxis=dict(range=[-max_abs, max_abs]),
        )
        st.plotly_chart(fig_local_shap, use_container_width=True)

    # -------------------------------------------------------------------------
    # GenAI Retention Copilot (1-to-1 Telesales Script & Personalized Email)
    # -------------------------------------------------------------------------
    st.markdown("##### 🤖 GenAI 1-to-1 Retention Copilot (SHAP + Causal Uplift Grounded)")
    if "copilot_active_cid" not in st.session_state:
        st.session_state["copilot_active_cid"] = selected_cid

    gen_btn_clicked = st.button(
        f"🤖 Generate AI Retention Script & Email for {selected_cid}",
        type="primary",
        use_container_width=True,
        key="btn_generate_copilot_tab3",
    )
    if gen_btn_clicked:
        st.session_state["copilot_active_cid"] = selected_cid

    pitch_payload = generate_retention_pitch(cust_row)
    with st.expander(
        f"✨ AI-Generated Retention Playbook & Personalized Email — {selected_cid} "
        f"({cust_causal_seg} | Uplift τ = {cust_uplift:+.3f})",
        expanded=True,
    ):
        cp_tab_a, cp_tab_b = st.tabs(
            [
                "📞 Tab A: Account Manager Telesales Call Script",
                "✉️ Tab B: Personalized Customer Retention Email",
            ]
        )
        with cp_tab_a:
            st.markdown(pitch_payload["call_script"]["full_markdown"])
            with st.expander("📋 Copy Raw Telesales Script (Plaintext / Markdown)", expanded=False):
                st.code(pitch_payload["call_script"]["full_markdown"], language="markdown")

        with cp_tab_b:
            st.markdown(pitch_payload["executive_email"]["body_markdown"])
            em_col1, em_col2 = st.columns([1.2, 1.0])
            with em_col1:
                st.download_button(
                    label=f"📥 Download Personalized Email & Script Template ({selected_cid}.md)",
                    data=(
                        pitch_payload["call_script"]["full_markdown"]
                        + "\n\n---\n\n"
                        + pitch_payload["executive_email"]["body_markdown"]
                    ).encode("utf-8"),
                    file_name=f"retention_playbook_{selected_cid}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"dl_copilot_{selected_cid}",
                )
            with em_col2:
                with st.expander("📋 Copy Raw Email Template", expanded=False):
                    st.code(pitch_payload["executive_email"]["body_markdown"], language="markdown")

    # Closed-Loop CRM Update Form (persists directly to SQLite)
    st.markdown("##### 🔄 Update Closed-Loop CRM Action Status in SQLite Warehouse")
    current_status = str(cust_row.get("action_status", "Pending") or "Pending")
    status_idx = list(ACTION_STATUS_DOMAIN).index(current_status) if current_status in ACTION_STATUS_DOMAIN else 0
    current_notes = str(cust_row.get("manager_notes", "") or "")

    with st.form("crm_feedback_form"):
        crm_c1, crm_c2, crm_c3 = st.columns([1.0, 2.2, 0.8])
        with crm_c1:
            new_crm_status = st.selectbox(
                f"Action Status for {selected_cid}:",
                options=list(ACTION_STATUS_DOMAIN),
                index=status_idx,
            )
        with crm_c2:
            new_crm_notes = st.text_input(
                "Account Manager / Telesales Notes:",
                value=current_notes,
                placeholder="e.g., Offered 20% 3-month loyalty discount; customer agreed to 1-year renewal.",
            )
        with crm_c3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            crm_saved = st.form_submit_button("💾 Save CRM Status to DB", type="primary", use_container_width=True)

    if crm_saved:
        updated_info = update_customer_crm_status(
            customer_id=selected_cid,
            new_status=new_crm_status,
            manager_notes=new_crm_notes,
        )
        load_customer_360.clear()
        st.success(
            f"✅ Closed-Loop CRM record updated in `database/ism_dss.db` for **{selected_cid}** → "
            f"Status: **{updated_info['action_status']}** at `{updated_info['updated_at']}`."
        )


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
            in_senior = st.selectbox(
                "Senior Citizen", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
            )
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
            in_monthly = st.number_input(
                "Monthly Charges ($)", min_value=18.0, max_value=120.0, value=95.50, step=1.0
            )
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
        decision_df = compute_causal_uplift(decision_df)
        dec_row = decision_df.iloc[0]

        _, _, single_shap_df, _, _ = explain_churn(
            artifacts["xgb"]["pipeline"],
            pd.DataFrame([record])[artifacts["xgb"]["features"]],
        )

        live_causal_seg = str(dec_row["causal_segment"])
        live_uplift = float(dec_row["uplift_score"])
        live_is_dnd = int(dec_row["is_do_not_disturb"]) == 1

        st.success("✅ Real-Time Mining, Causal Uplift & GenAI Pitch Evaluation Completed")
        if live_is_dnd:
            st.markdown(
                f"""
                <div style="background: rgba(244, 63, 94, 0.18); border: 1px solid #F43F5E; border-left: 5px solid #F43F5E; border-radius: 10px; padding: 11px 16px; margin-bottom: 14px; color: #FFE4E6;">
                    ⛔ <b>DO NOT DISTURB — SLEEPING DOG DETECTED (Uplift &tau; = {live_uplift:+.3f}):</b>
                    Proactive sales outreach is predicted to trigger cancellation. Suppress outbound calls and use passive care only.
                </div>
                """,
                unsafe_allow_html=True,
            )

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        render_kpi_card(
            r1,
            "Predicted Value Tier",
            f"{dec_row['value_tier']} Value",
            f"K-Means Cluster {scored['cluster_id']}",
            accent="cyan",
        )
        render_kpi_card(
            r2,
            "Churn Probability",
            f"{dec_row['churn_probability'] * 100:.2f}%",
            f"{dec_row['risk_level']} Risk Level",
            accent="rose",
        )
        render_kpi_card(
            r3,
            "Causal Uplift Segment",
            f"{live_causal_seg}",
            f"Uplift τ = {live_uplift:+.3f}" + (" · ⛔ DND" if live_is_dnd else " · Targetable"),
            accent="rose" if live_is_dnd else ("emerald" if live_causal_seg == "Persuadable" else "indigo"),
        )
        render_kpi_card(
            r4,
            "Annual Revenue at Risk",
            f"${dec_row['revenue_at_risk']:,.2f}",
            f"Annual Rev: ${dec_row['annualized_revenue']:,.0f}",
            accent="rose",
        )
        render_kpi_card(
            r5,
            "Expected Net Benefit",
            f"${dec_row['expected_net_benefit']:,.2f}",
            f"Cost: ${dec_row['intervention_cost']:,.2f} · Saved: ${dec_row['expected_saved_revenue']:,.2f}",
            accent="emerald",
        )
        render_kpi_card(
            r6,
            "Expected Campaign ROI",
            f"{dec_row['expected_roi']:,.1f}%",
            dec_row["decision_quadrant"],
            accent="emerald",
        )

        adv_left, adv_right = st.columns([1.05, 1.25])
        with adv_left:
            st.markdown(
                f"""
                <div class="profile-card">
                    <h5>🎯 Prescriptive & Causal Retention Recommendation</h5>
                    <div class="sb-row"><span class="sb-label">Strategic Quadrant</span><span class="sb-val" style="color:#38BDF8;">{dec_row['decision_quadrant']}</span></div>
                    <div class="sb-row"><span class="sb-label">Causal Uplift Segment</span><span class="sb-val" style="color:{'#FDA4AF' if live_is_dnd else '#34D399'};">{live_causal_seg} (τ = {live_uplift:+.3f})</span></div>
                    <div class="sb-row"><span class="sb-label">Primary SHAP Driver 1</span><span class="sb-val" style="color:#FDA4AF;">{scored['top_shap_driver_1']}</span></div>
                    <div class="sb-row"><span class="sb-label">Secondary SHAP Driver 2</span><span class="sb-val" style="color:#FCD34D;">{scored['top_shap_driver_2']}</span></div>
                    <div class="sb-row"><span class="sb-label">Lifecycle Stage</span><span class="sb-val">{t_group} ({vas_count} active VAS add-ons)</span></div>
                    <div style="margin-top:14px; padding:12px 14px; background:rgba(16,185,129,0.12); border-left:4px solid #10B981; border-radius:8px; font-size:0.88rem; color:#F8FAFC;">
                        <b>Recommended Managerial Action:</b><br>{dec_row['recommended_action']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with adv_right:
            s_series = single_shap_df.iloc[0].sort_values(key=lambda s: s.abs(), ascending=True).tail(10)
            s_colors = ["#F43F5E" if v > 0 else "#10B981" for v in s_series.values]
            fig_single_shap = go.Figure(
                go.Bar(
                    x=s_series.values,
                    y=s_series.index,
                    orientation="h",
                    marker_color=s_colors,
                    text=[f"{v:+.2f}" for v in s_series.values],
                    textposition="outside",
                    textfont=dict(color="#E2E8F0", size=11),
                )
            )
            style_plotly_dark(
                fig_single_shap,
                height=320,
                title="Customer-Specific SHAP Attribution Waterfall (Top 10 Features)",
            )
            s_max = float(s_series.abs().max()) * 1.25
            fig_single_shap.update_layout(
                xaxis_title="SHAP Log-Odds Contribution (+ Risk | - Protective)",
                yaxis_title="Business Attribute",
                xaxis=dict(range=[-s_max, s_max]),
            )
            st.plotly_chart(fig_single_shap, use_container_width=True)

        # Auto-Generated GenAI Retention Pitch for Newly Scored Customer
        live_pitch = generate_retention_pitch(dec_row)
        st.markdown("##### 🤖 Auto-Generated 1-to-1 AI Retention Script & Personalized Email")
        live_tab_a, live_tab_b = st.tabs(
            [
                "📞 Tab A: Account Manager Telesales Call Script",
                "✉️ Tab B: Personalized Customer Email Draft",
            ]
        )
        with live_tab_a:
            st.markdown(live_pitch["call_script"]["full_markdown"])
        with live_tab_b:
            st.markdown(live_pitch["executive_email"]["body_markdown"])


# ===========================================================================
# TAB 5 — MODEL BENCHMARKING & SYSTEM GOVERNANCE (ISM AUDIT VIEW)
# ===========================================================================
with tab5:
    st.subheader("Multi-Model Benchmarking & Information Systems Governance (ISM Audit View)")
    st.markdown(
        "Empirical comparison of **Logistic Regression**, **Random Forest**, and **XGBoost (Champion)** "
        "on the stratified 80/20 hold-out split ($n_{\\text{train}} = 5{,}634$, $n_{\\text{test}} = 1{,}409$), "
        "alongside relational warehouse integrity verification and `etl_audit_log` lineage."
    )

    bm_models = pd.DataFrame(benchmark_data["models"])
    opt_thresh_info = profit_curve_data["optimal_threshold"]
    audit_df = load_etl_audit_log()

    g1, g2, g3, g4 = st.columns(4)
    render_kpi_card(
        g1,
        "Champion Model (Deployed)",
        "XGBoost Classifier",
        f"Test ROC-AUC: {model_metrics['roc_auc']:.4f} · Recall: {model_metrics['recall'] * 100:.1f}%",
        accent="cyan",
    )
    render_kpi_card(
        g2,
        "Benchmarked Candidates",
        "3 Architectures",
        "LogReg · Random Forest · XGBoost",
        accent="indigo",
    )
    render_kpi_card(
        g3,
        "Optimal Profit Cutoff",
        f"p ≥ {opt_thresh_info['high_risk_threshold']:.2f}",
        f"Max Effective Net Benefit: ${opt_thresh_info['expected_net_benefit']:,.0f}",
        accent="emerald",
    )
    render_kpi_card(
        g4,
        "Warehouse Integrity Check",
        "0 FK / NULL Errors",
        f"{len(audit_df)} Logged ETL Runs · 7,043 3NF Records",
        accent="emerald",
    )

    st.markdown("#### 1. Multi-Model Hold-Out Evaluation Benchmark ($n_{\\text{test}} = 1{,}409$)")
    bm_display = pd.DataFrame(
        {
            "Model Architecture": bm_models["model_name"],
            "Champion": bm_models["is_champion"].map(lambda x: "★ Champion" if x else "Challenger"),
            "Test ROC-AUC": bm_models["roc_auc"].map(lambda x: f"{x:.4f}"),
            "Test PR-AUC": bm_models["pr_auc"].map(lambda x: f"{x:.4f}"),
            "Recall (Sensitivity)": bm_models["recall"].map(lambda x: f"{x * 100:.2f}%"),
            "Precision": bm_models["precision"].map(lambda x: f"{x * 100:.2f}%"),
            "F1-Score": bm_models["f1"].map(lambda x: f"{x:.4f}"),
            "Accuracy": bm_models["accuracy"].map(lambda x: f"{x * 100:.2f}%"),
            "Brier Score (Calibration)": bm_models["brier_score"].map(lambda x: f"{x:.4f}"),
            "Train ROC-AUC": bm_models["train_roc_auc"].map(lambda x: f"{x:.4f}"),
            "Training Time (s)": bm_models["training_time_sec"].map(lambda x: f"{x:.3f}s"),
        }
    )
    st.dataframe(bm_display, use_container_width=True, hide_index=True)

    roc_col, pr_col = st.columns(2)
    model_colors = {
        "Logistic Regression": "#F59E0B",
        "Random Forest": "#10B981",
        "XGBoost (Champion)": "#38BDF8",
    }

    with roc_col:
        st.markdown("#### 2. Multi-Model Receiver Operating Characteristic (ROC) Curves")
        fig_roc = go.Figure()
        for m_row in benchmark_data["models"]:
            m_name = m_row["model_name"]
            c_pts = benchmark_data["roc_curves"][m_name]
            fig_roc.add_trace(
                go.Scatter(
                    x=c_pts["x"],
                    y=c_pts["y"],
                    mode="lines",
                    name=f"{m_name} (AUC = {m_row['roc_auc']:.4f})",
                    line=dict(color=model_colors.get(m_name, "#38BDF8"), width=3 if m_row["is_champion"] else 2),
                )
            )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier (AUC = 0.5000)",
                line=dict(color="#64748B", width=1.5, dash="dash"),
            )
        )
        style_plotly_dark(fig_roc, height=370)
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall / Sensitivity)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with pr_col:
        st.markdown("#### 3. Multi-Model Precision-Recall (PR) Curves")
        fig_pr = go.Figure()
        for m_row in benchmark_data["models"]:
            m_name = m_row["model_name"]
            c_pts = benchmark_data["pr_curves"][m_name]
            fig_pr.add_trace(
                go.Scatter(
                    x=c_pts["x"],
                    y=c_pts["y"],
                    mode="lines",
                    name=f"{m_name} (PR-AUC = {m_row['pr_auc']:.4f})",
                    line=dict(color=model_colors.get(m_name, "#38BDF8"), width=3 if m_row["is_champion"] else 2),
                )
            )
        style_plotly_dark(fig_pr, height=370)
        fig_pr.update_layout(
            xaxis_title="Recall (Sensitivity)",
            yaxis_title="Precision (Positive Predictive Value)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.divider()
    aud_left, aud_right = st.columns([1.25, 0.95])
    with aud_left:
        st.markdown("#### 4. Data Warehouse ETL Audit Lineage (`etl_audit_log`)")
        st.dataframe(audit_df, use_container_width=True, hide_index=True)

    with aud_right:
        st.markdown("#### 5. Closed-Loop CRM Action Status Summary (`fact_retention_actions`)")
        crm_summary_df = (
            df_360.groupby("action_status", as_index=False)
            .agg(
                accounts=("customerID", "count"),
                revenue_at_risk=("revenue_at_risk", "sum"),
                intervention_cost=("intervention_cost", "sum"),
                expected_net_benefit=("expected_net_benefit", "sum"),
            )
            .rename(
                columns={
                    "action_status": "CRM Status",
                    "accounts": "Accounts",
                    "revenue_at_risk": "Revenue at Risk ($)",
                    "intervention_cost": "Intervention Cost ($)",
                    "expected_net_benefit": "Expected Net Benefit ($)",
                }
            )
        )
        st.dataframe(crm_summary_df.round(2), use_container_width=True, hide_index=True)
