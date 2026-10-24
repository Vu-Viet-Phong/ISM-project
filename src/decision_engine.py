"""
Module Ra Quyết định Khuyến nghị & Tối ưu hóa ROI — `src/decision_engine.py`.

Vai trò trong Kiến trúc 4 Tầng (4-Tier ISM Architecture):
    Thuộc Tầng 3 (Tier 3 — Prescriptive Decision Logic & Economic Modeling Layer).
    Module này chuyển hóa các kết quả khai phá dữ liệu từ Tầng 3A (Phân hạng Giá trị Khách hàng
    `value_tier`, Xác suất Rời bỏ `churn_probability`, Mức độ Rủi ro `risk_level`, và Nhân tố
    Giải thích SHAP cục bộ `top_shap_driver_1`, `top_shap_driver_2`) thành các hành động
    quản trị cụ thể (Prescriptive Actions) cùng dự phóng lợi tức đầu tư tài chính (ROI).

    4 Năng lực Cốt lõi:
        1. Mô phỏng Quyết định Tham số hóa (`run_decision_simulation`):
           Cho phép truyền vào từ điển tham số kinh tế tùy chỉnh để phục vụ phân tích độ nhạy
           What-If thời gian thực trên giao diện quản trị Streamlit.
        2. Ma trận Quyết định 9 Ô (9-Cell Value x Risk Decision Matrix):
           Ánh xạ từng tổ hợp `(value_tier, risk_level)` vào 6 góc phần tư chiến lược
           (`decision_quadrant`) và đề xuất gói ưu đãi cá nhân hóa theo SHAP (`recommended_action`).
        3. Mô hình Kinh tế & Tối ưu hóa ROI:
           Tính toán Doanh thu quy đổi năm (`annualized_revenue`), Doanh thu có nguy cơ mất
           (`revenue_at_risk`), Chi phí can thiệp kỳ vọng (`intervention_cost`), Doanh thu
           cứu được kỳ vọng (`expected_saved_revenue`), Lợi ích ròng (`expected_net_benefit`)
           và Tỷ suất hoàn vốn (`expected_roi`) cho từng khách hàng và toàn danh mục.
        4. Lưu trữ Kho dữ liệu & Theo dõi CRM Khép kín (Closed-Loop CRM Persistence):
           Ghi nguyên tử (atomic) 7.043 bản ghi vào bảng `fact_retention_actions` trong
           `database/ism_dss.db`, đồng thời bảo toàn các thuộc tính Causal Uplift và trạng thái CRM.

Nhóm tác giả (Group 6 — International University, VNU-HCM):
    1. Vũ Viết Phong — ITDSIU19048
    2. Trương Đăng Khoa — ITDSIU19027
    3. Trần Khánh Toàn — ITITIU20322
    4. Nguyễn Lê Mai Hương — ITITWE23040

Các hàm chính:
    - load_scored_customers: Tải dữ liệu khách hàng đã chấm điểm từ `customer_master` và `fact_mining_scores`.
    - build_recommended_action: Xây dựng hành động khuyến nghị cá nhân hóa theo nguyên nhân SHAP.
    - run_decision_simulation: Chạy mô phỏng tài chính, chi phí can thiệp và ROI trên toàn tập khách hàng.
    - summarize_decisions: Tổng hợp các chỉ số KPI toàn danh mục và phân rã theo 9 ô / 6 góc phần tư.
    - ensure_crm_columns / persist_retention_actions / update_customer_crm_status: Quản trị dữ liệu SQLite.
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
# Cấu hình Đường dẫn & Bộ Tham số Kinh tế Mặc định (Default Baseline Parameters)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_FILE = PROJECT_ROOT / "database" / "ism_dss.db"
MODELS_DIR = PROJECT_ROOT / "models"
DECISION_SUMMARY_FILE = MODELS_DIR / "decision_summary.json"

# Các tham số kinh tế cơ sở dùng để tính toán chi phí chiến dịch và doanh thu giữ chân kỳ vọng
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

# Ánh xạ 9 ô (value_tier, risk_level) sang 6 góc phần tư chiến lược quản trị (Strategic Quadrants)
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

# Thứ tự hiển thị chuẩn của 6 góc phần tư chiến lược trên báo cáo và biểu đồ
QUADRANT_ORDER = [
    "Critical VIP Retention",
    "Proactive VIP Care",
    "Automated Retention Campaign",
    "Targeted Nudge",
    "No Financial Intervention",
    "Champions / Nurture (No Intervention)",
]


# ---------------------------------------------------------------------------
# Tầng Truy cập Dữ liệu (Data Access Layer)
# ---------------------------------------------------------------------------
def load_scored_customers(db_path: Path = DB_FILE) -> pd.DataFrame:
    """
    Tải danh sách khách hàng kèm điểm số khai phá dữ liệu từ kho `database/ism_dss.db`.

    Thực hiện kết nối (JOIN) trực tiếp giữa view `customer_master` và bảng sự kiện
    `fact_mining_scores` để đảm bảo dữ liệu đầu vào luôn sạch trước khi tính toán
    và làm mới bảng `fact_retention_actions`.

    Tham số (Args):
        db_path (Path): Đường dẫn tới tệp cơ sở dữ liệu SQLite.

    Trả về (Returns):
        pd.DataFrame: Bảng dữ liệu 7.043 khách hàng cùng các cột điểm số K-Means, XGBoost và SHAP.
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
# Cá nhân hóa Hành động Khuyến nghị dựa trên Giải thích SHAP (SHAP Personalization)
# ---------------------------------------------------------------------------
def _parse_driver_feature(driver_str: Optional[str]) -> str:
    """
    Trích xuất tên thuộc tính nghiệp vụ gốc từ chuỗi nhân tố SHAP đã định dạng.

    Tham số (Args):
        driver_str (Optional[str]): Chuỗi nhân tố SHAP (ví dụ: `'contract=Month-to-month (+0.82)'`).

    Trả về (Returns):
        str: Tên thuộc tính gốc (ví dụ: `'contract'`) hoặc chuỗi rỗng nếu đầu vào không hợp lệ.
    """
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
    Xây dựng khuyến nghị hành động giữ chân cá nhân hóa theo SHAP cho từng khách hàng.

    Kết hợp khung chiến lược của ô `(value_tier, risk_level)` với gói ưu đãi chiến thuật
    đánh trúng vào nguyên nhân gây rời bỏ lớn nhất được chỉ ra bởi TreeSHAP.

    Tham số (Args):
        value_tier (str): Phân hạng giá trị kinh tế của khách hàng ('High', 'Medium', 'Low').
        risk_level (str): Mức độ rủi ro rời bỏ ('High', 'Medium', 'Low').
        driver_1 (Optional[str]): Nhân tố SHAP làm tăng rủi ro mạnh thứ nhất.
        driver_2 (Optional[str]): Nhân tố SHAP làm tăng rủi ro mạnh thứ hai.
        high_disc_pct (float): Tỷ lệ chiết khấu dành cho khách hàng giá trị cao (ví dụ: 0.20 tương ứng 20%).
        med_disc_pct (float): Tỷ lệ chiết khấu dành cho khách hàng giá trị trung bình (ví dụ: 0.10 tương ứng 10%).

    Trả về (Returns):
        str: Chuỗi mô tả hành động can thiệp khuyến nghị cụ thể dành cho Account Manager.
    """
    feat1 = _parse_driver_feature(driver_1)
    feat2 = _parse_driver_feature(driver_2)
    drivers = {feat1, feat2}

    high_pct_label = int(round(high_disc_pct * 100))
    med_pct_label = int(round(med_disc_pct * 100))

    # Lựa chọn gói ưu đãi chiến thuật dựa trên nhân tố rủi ro SHAP nổi trội của khách hàng
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

    # Kết hợp kênh tiếp cận theo phân khúc giá trị x rủi ro với gói ưu đãi chiến thuật
    if value_tier == "High" and risk_level == "High":
        return f"1-on-1 Account Manager call + {vip_tactic}"
    if value_tier == "High" and risk_level == "Medium":
        return "Proactive VIP check-in call + complimentary VAS bundle (OnlineSecurity/TechSupport)"
    if value_tier == "Medium" and risk_level == "High":
        return f"Automated email/SMS campaign + {med_tactic}"
    if value_tier == "Medium" and risk_level == "Medium":
        return f"Automated contract-renewal nudge + {med_tactic}"
    if value_tier == "Low" and risk_level in ("High", "Medium"):
        return "Automated satisfaction survey & self-service FAQ email ($0 financial incentive)"
    return "Standard service quality + organic cross-sell/upsell nurture ($0 intervention)"


# ---------------------------------------------------------------------------
# Động cơ Mô phỏng Quyết định & Tối ưu hóa ROI (Parameterized ROI Simulation Engine)
# ---------------------------------------------------------------------------
def run_decision_simulation(
    df: pd.DataFrame,
    params: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Thực thi ma trận quyết định giữ chân khách hàng và mô hình kinh tế dự phóng ROI.

    Hàm được thiết kế thuần túy (pure function) và tham số hóa hoàn toàn để giao diện
    Streamlit có thể gọi lại tức thì mỗi khi người quản lý kéo thanh trượt What-If.

    Công thức kinh tế học cho từng khách hàng thứ `i`:
        1. Doanh thu quy đổi theo năm (Annualized Revenue):
           `annualized_revenue_i = monthly_charges_i * annual_horizon_months`
        2. Doanh thu có nguy cơ mất (Expected Revenue at Risk - RaR):
           `revenue_at_risk_i = churn_probability_i * annualized_revenue_i`
        3. Doanh thu kỳ vọng cứu được (Expected Saved Revenue):
           `expected_saved_revenue_i = revenue_at_risk_i * s_i`
           (trong đó `s_i` là xác suất giữ chân thành công của kênh can thiệp tương ứng).
        4. Chi phí can thiệp kỳ vọng (Expected Intervention Cost):
           `intervention_cost_i = channel_cost_i + s_i * (discount_pct_i * monthly_charges_i * discount_months)`
           (Chi phí liên hệ `channel_cost_i` như gọi điện/gửi email phát sinh ngay từ đầu cho mọi
           khách hàng được nhắm mục tiêu; trong khi khoản chiết khấu cước phí chỉ thực sự phát sinh
           khi khách hàng đồng ý ở lại với xác suất thành công `s_i`).
        5. Lợi ích tài chính ròng kỳ vọng (Expected Net Benefit):
           `expected_net_benefit_i = expected_saved_revenue_i - intervention_cost_i`
        6. Tỷ suất hoàn vốn kỳ vọng (Expected ROI %):
           `expected_roi_i = ((expected_saved_revenue_i - intervention_cost_i) / intervention_cost_i) * 100`
           (bằng 0.0 nếu `intervention_cost_i == 0`).

    Tham số (Args):
        df (pd.DataFrame): Bảng dữ liệu chứa tối thiểu các cột `customerID`, `monthly_charges`,
            `value_tier`, `churn_probability`, `risk_level`, `top_shap_driver_1`, `top_shap_driver_2`.
        params (Optional[Dict[str, float]]): Từ điển ghi đè các tham số kinh tế trong `DEFAULT_PARAMS`.

    Trả về (Returns):
        pd.DataFrame: Bảng kết quả chứa đầy đủ các cột phục vụ bảng `fact_retention_actions`
            cùng cột chẩn đoán `expected_success_rate`.
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

    # 1. Tính Doanh thu quy đổi năm (Annualized Revenue) và Doanh thu có nguy cơ mất (Revenue at Risk)
    annualized_revenue = np.round(monthly * horizon, 2)
    revenue_at_risk = np.round(prob * annualized_revenue, 2)

    # 2. Phân bổ khách hàng vào góc phần tư chiến lược và gán kịch bản hành động theo SHAP
    quadrants = [QUADRANT_MAP[(v, r)] for v, r in zip(vt, rl)]
    actions = [
        build_recommended_action(v, r, drv1, drv2, high_disc, med_disc)
        for v, r, drv1, drv2 in zip(vt, rl, d1, d2)
    ]

    # 3. Xác định mặt nạ logic cho 4 ô mục tiêu có phân bổ ngân sách can thiệp (Cost > $0)
    cond_hh = (vt == "High") & (rl == "High")
    cond_hm = (vt == "High") & (rl == "Medium")
    cond_mh = (vt == "Medium") & (rl == "High")
    cond_mm = (vt == "Medium") & (rl == "Medium")

    # Tỷ lệ giữ chân thành công kỳ vọng s_i cho từng ô trong ma trận
    success_rate = np.select(
        [cond_hh, cond_hm, cond_mh, cond_mm],
        [sr_vip, sr_vip * 0.80, sr_auto, sr_auto * 0.70],
        default=0.0,
    )

    # Chi phí can thiệp kỳ vọng c_i = Chi phí kênh liên hệ ban đầu + Kỳ vọng giá trị chiết khấu được quy đổi
    cost_hh = call_cost + sr_vip * (monthly * high_disc * disc_months)
    cost_hm = call_cost + (sr_vip * 0.80) * (monthly * (high_disc * 0.50) * disc_months)
    cost_mh = email_cost + sr_auto * (monthly * med_disc * disc_months)
    cost_mm = email_cost + (sr_auto * 0.70) * (monthly * (med_disc * 0.50) * disc_months)

    intervention_cost = np.round(
        np.select([cond_hh, cond_hm, cond_mh, cond_mm], [cost_hh, cost_hm, cost_mh, cost_mm], default=0.0),
        2,
    )

    # 4. Tính Doanh thu cứu được kỳ vọng, Lợi ích ròng (Net Benefit) và Tỷ suất ROI (%)
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
    Tổng hợp kết quả mô phỏng cấp độ khách hàng thành hệ thống chỉ số KPI toàn danh mục
    và bảng phân rã theo 6 góc phần tư chiến lược cũng như 9 ô ma trận `(value_tier x risk_level)`.

    Tham số (Args):
        sim_df (pd.DataFrame): Bảng kết quả đầu ra từ hàm `run_decision_simulation`.
        params (Optional[Dict[str, float]]): Bộ tham số kinh tế đã dùng để chạy mô phỏng.

    Trả về (Returns):
        Dict[str, Any]: Từ điển chứa `parameters`, `portfolio_kpis`, `quadrant_breakdown`,
            `matrix_9_cells`, và dấu thời gian `generated_at`.
    """
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    total_rev = float(sim_df["annualized_revenue"].sum())
    total_rar = float(sim_df["revenue_at_risk"].sum())
    total_cost = float(sim_df["intervention_cost"].sum())
    total_saved = float(sim_df["expected_saved_revenue"].sum())
    total_net = float(sim_df["expected_net_benefit"].sum())
    portfolio_roi = float(((total_saved - total_cost) / total_cost) * 100.0) if total_cost > 0 else 0.0
    targeted_mask = sim_df["intervention_cost"] > 0

    # Phân rã chi tiết theo 6 góc phần tư quyết định (Decision Quadrants)
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

    # Phân rã chi tiết theo toàn bộ 9 ô (value_tier x risk_level)
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
# Lưu trữ Kho dữ liệu & Quản trị Trạng thái CRM (Warehouse & Closed-Loop CRM Persistence)
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
    Đảm bảo bảng `fact_retention_actions` và view `vw_customer_360` luôn có đầy đủ
    các cột mở rộng của Causal Uplift (`uplift_score`, `causal_segment`, `is_do_not_disturb`)
    và các cột theo dõi CRM khép kín (`action_status`, `manager_notes`, `updated_at`).

    Tham số (Args):
        conn (sqlite3.Connection): Kết nối SQLite đang mở.

    Trả về (Returns):
        None
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
    Ghi nguyên tử (atomic) toàn bộ 7.043 dòng quyết định giữ chân vào bảng `fact_retention_actions`.

    Tự động bảo toàn các giá trị `uplift_score`, `causal_segment`, `is_do_not_disturb`,
    `action_status`, `manager_notes`, và `updated_at` đã có từ trước để không làm mất
    lịch sử tương tác CRM hoặc kết quả phân tích nhân quả khi chạy lại pipeline.

    Tham số (Args):
        sim_df (pd.DataFrame): Bảng kết quả mô phỏng quyết định từ `run_decision_simulation`.
        db_path (Path): Đường dẫn tới tệp cơ sở dữ liệu SQLite.

    Trả về (Returns):
        None
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
    Cập nhật trạng thái thực thi CRM khép kín (`action_status`) và ghi chú của quản lý
    (`manager_notes`) cho một khách hàng cụ thể trực tiếp vào SQLite.

    Tham số (Args):
        customer_id (str): Mã định danh khách hàng cần cập nhật.
        new_status (str): Trạng thái mới thuộc miền `('Pending', 'Contacted', 'Retained', 'Churned')`.
        manager_notes (str): Nội dung ghi chú của Account Manager.
        db_path (Path): Đường dẫn tới tệp cơ sở dữ liệu SQLite.

    Trả về (Returns):
        Dict[str, Any]: Từ điển xác nhận thông tin bản ghi vừa cập nhật kèm dấu thời gian `updated_at`.
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
    """
    Lưu từ điển tổng hợp quyết định toàn danh mục ra tệp JSON `models/decision_summary.json`.

    Tham số (Args):
        summary (Dict[str, Any]): Từ điển kết quả từ hàm `summarize_decisions`.
        out_path (Path): Đường dẫn tệp JSON đích.

    Trả về (Returns):
        None
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[ARTIFACT]  Saved portfolio decision summary to {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Kiểm định Toàn vẹn & Báo cáo Quản trị (Verification & Managerial Reporting)
# ---------------------------------------------------------------------------
def verify_and_report(summary: Dict[str, Any], db_path: Path = DB_FILE) -> None:
    """
    Kiểm tra tính toàn vẹn của view `vw_customer_360` và in báo cáo tổng hợp ra màn hình console.

    Tham số (Args):
        summary (Dict[str, Any]): Từ điển tổng hợp các chỉ số KPI và phân rã ma trận quyết định.
        db_path (Path): Đường dẫn tới tệp cơ sở dữ liệu SQLite.

    Trả về (Returns):
        None
    """
    with sqlite3.connect(db_path) as conn:
        view_df = pd.read_sql("SELECT * FROM vw_customer_360", conn)

    # Kiểm tra đảm bảo không có giá trị NULL bất thường trên toàn bộ các cột cốt lõi
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
# Điểm Khởi chạy (Entry Point)
# ---------------------------------------------------------------------------
def run_engine(params: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Chạy trọn vẹn quy trình Mô phỏng Quyết định, Lưu trữ SQLite và Xuất báo cáo JSON.

    Tham số (Args):
        params (Optional[Dict[str, float]]): Từ điển tham số kinh tế tùy chỉnh (nếu có).

    Trả về (Returns):
        Tuple[pd.DataFrame, Dict[str, Any]]: Bộ gồm DataFrame kết quả mô phỏng và từ điển tổng hợp KPI.
    """
    df = load_scored_customers()
    sim_df = run_decision_simulation(df, params=params)
    summary = summarize_decisions(sim_df, params=params)
    persist_retention_actions(sim_df)
    save_decision_summary(summary)
    verify_and_report(summary)
    return sim_df, summary


def main() -> int:
    """
    Hàm main xử lý ngoại lệ và trả về mã thoát hệ thống (exit code) cho dòng lệnh CLI.

    Tham số (Args):
        Không có.

    Trả về (Returns):
        int: 0 nếu thực thi thành công, 1 nếu xảy ra lỗi.
    """
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
