"""
Module Trợ lý AI Sinh Kịch bản Giữ chân Cá nhân hóa 1-1 — `src/copilot_engine.py`.

Vai trò trong Kiến trúc 4 Tầng (4-Tier ISM Architecture):
    Thuộc Tầng 3 (Tier 3 — Analytics & Prescriptive Decision Logic Layer).
    Module này tổng hợp thông tin hồ sơ 360 độ của khách hàng (`tenure`, `monthly_charges`,
    `value_tier`, `churn_probability`, `causal_segment`, `uplift_score`) kết hợp với các
    nhân tố giải thích rủi ro cục bộ từ TreeSHAP (`top_shap_driver_1`, `top_shap_driver_2`)
    và khoảng trống dịch vụ gia tăng (VAS gaps) để tự động biên soạn:
        1. `call_script`: Kịch bản gọi điện Telesales 3 bước dành cho Account Manager
           (Mở đầu cá nhân hóa, Đồng cảm & Xử lý từ chối theo SHAP, Đề xuất gói ưu đãi chốt đơn)
           kèm theo hàng rào cảnh báo nhân quả (Causal Guardrail) nếu khách hàng thuộc nhóm
           `Sleeping Dog` (`is_do_not_disturb == 1`).
        2. `executive_email`: Email chăm sóc khách hàng định dạng Markdown hoàn chỉnh
           (gồm tiêu đề `subject_line` và nội dung `body_markdown`) được thiết kế riêng
           dựa trên các dịch vụ khách hàng chưa đăng ký (`OnlineSecurity`, `TechSupport`,
           `OnlineBackup`, `DeviceProtection`) và loại hợp đồng hiện tại.

Nhóm tác giả (Group 6 — International University, VNU-HCM):
    1. Vũ Viết Phong — ITDSIU19048
    2. Trương Đăng Khoa — ITDSIU19027
    3. Trần Khánh Toàn — ITITIU20322
    4. Nguyễn Lê Mai Hương — ITITWE23040

Các hàm chính:
    - _parse_shap_driver: Bóc tách tên thuộc tính và trọng số log-odds từ chuỗi SHAP.
    - _build_shap_objection_playbook: Ánh xạ từng nguyên nhân SHAP sang luận điểm đồng cảm và xử lý từ chối.
    - generate_retention_pitch: Sinh trọn bộ kịch bản Telesales 3 bước và Email giữ chân 1-1.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Tuple, Union

import pandas as pd


def _parse_shap_driver(driver_str: str) -> Tuple[str, float]:
    """
    Bóc tách chuỗi mô tả nhân tố SHAP (ví dụ: `'contract=Month-to-month (+1.12)'`
    hoặc `'contract (+1.12)'`) thành cặp `(tên_thuộc_tính, giá_trị_shap)`.

    Tham số (Args):
        driver_str (str): Chuỗi định dạng SHAP driver lấy từ bảng `fact_mining_scores`.

    Trả về (Returns):
        Tuple[str, float]: Tên thuộc tính gốc dạng chữ thường và giá trị đóng góp SHAP (log-odds).
    """
    if not driver_str:
        return ("contract", 0.0)
    # Bóc tách phần tên thuộc tính (trước dấu '=' hoặc dấu '(') và giá trị số trong ngoặc đơn
    match = re.match(r"^\s*([A-Za-z0-9_]+)(?:=[^()]*)?\s*\(([+-]?[0-9.]+)\)\s*$", str(driver_str))
    if match:
        return (match.group(1).strip().lower(), float(match.group(2)))
    clean_name = str(driver_str).split("(")[0].split("=")[0].strip().lower()
    return (clean_name or "contract", 0.0)


def _get_val(row: Mapping[str, Any], snake: str, pascal: str, default: Any = "") -> Any:
    """
    Truy xuất an toàn giá trị thuộc tính từ từ điển hoặc Series theo cả `snake_case` và `PascalCase`.

    Tham số (Args):
        row (Mapping[str, Any]): Bản ghi dữ liệu của một khách hàng.
        snake (str): Tên khóa dạng snake_case.
        pascal (str): Tên khóa dạng PascalCase.
        default (Any): Giá trị trả về mặc định nếu khóa không tồn tại hoặc mang giá trị NaN.

    Trả về (Returns):
        Any: Giá trị thuộc tính của khách hàng hoặc giá trị mặc định.
    """
    if snake in row and pd.notna(row[snake]):
        return row[snake]
    if pascal in row and pd.notna(row[pascal]):
        return row[pascal]
    return default


def _build_shap_objection_playbook(
    feature_name: str,
    shap_val: float,
    row: Mapping[str, Any],
) -> Dict[str, str]:
    """
    Ánh xạ một nhân tố rủi ro TreeSHAP sang bộ kịch bản chẩn đoán nguyên nhân gốc rễ,
    lời thoại đồng cảm (Empathy Statement) và phương án xử lý từ chối (Objection Rebuttal).

    Tham số (Args):
        feature_name (str): Tên thuộc tính gốc (ví dụ: 'contract', 'internet_service', 'tenure').
        shap_val (float): Giá trị đóng góp SHAP cục bộ của thuộc tính đó (đơn vị log-odds).
        row (Mapping[str, Any]): Hồ sơ chi tiết của khách hàng để điền các con số thực tế vào lời thoại.

    Trả về (Returns):
        Dict[str, str]: Từ điển gồm 4 trường `label`, `diagnosis`, `empathy_line`, và `rebuttal`.
    """
    contract = str(_get_val(row, "contract", "Contract", "Month-to-month"))
    internet = str(_get_val(row, "internet_service", "InternetService", "Fiber optic"))
    payment = str(_get_val(row, "payment_method", "PaymentMethod", "Electronic check"))
    monthly = float(_get_val(row, "monthly_charges", "MonthlyCharges", 70.0) or 70.0)
    tenure = int(float(_get_val(row, "tenure", "tenure", 12) or 12))

    direction = "increases churn risk" if shap_val >= 0 else "stabilizes retention"
    sign_str = f"{shap_val:+.2f}"

    # Danh mục kịch bản xử lý từ chối chuyên biệt cho từng thuộc tính viễn thông cốt lõi
    playbooks = {
        "contract": {
            "label": f"Contract Structure ({contract}, SHAP {sign_str})",
            "diagnosis": (
                f"Account operates on a '{contract}' term ({direction}). Month-to-month "
                "exposure creates zero switching friction when competitors run seasonal promos."
            ),
            "empathy_line": (
                "\"I completely understand wanting the flexibility of a month-to-month plan—"
                "nobody likes feeling locked in without knowing they are getting the best rate.\""
            ),
            "rebuttal": (
                "Offer our 'Price-Lock Guarantee' 12-month agreement: waive early upgrade fees "
                "while locking in an immediate loyalty rate reduction so their bill cannot increase."
            ),
        },
        "internet_service": {
            "label": f"Internet Tier ({internet}, SHAP {sign_str})",
            "diagnosis": (
                f"Subscribed to '{internet}' service at ${monthly:,.2f}/mo ({direction}). "
                "High-tier Fiber subscribers are heavily targeted by rival ISP fiber rollouts."
            ),
            "empathy_line": (
                f"\"As a {internet} subscriber, reliable low-latency bandwidth is critical for your "
                "household, and you deserve enterprise-grade stability for every dollar you pay.\""
            ),
            "rebuttal": (
                "Provide a complimentary Remote Line Quality Diagnostic + Wi-Fi 6E Mesh Extender "
                "upgrade and bundle a 6-month Fiber Speed Boost at zero additional monthly cost."
            ),
        },
        "tenure": {
            "label": f"Customer Tenure ({tenure} months, SHAP {sign_str})",
            "diagnosis": (
                f"Account tenure is {tenure} months ({direction}). Early-lifecycle subscribers "
                "experience highest onboarding friction during their first 12 billing cycles."
            ),
            "empathy_line": (
                f"\"Over the past {max(tenure, 1)} months with us, we want to make sure your setup "
                "has been seamless and that you're getting full value from every feature included.\""
            ),
            "rebuttal": (
                "Trigger a Milestone Onboarding Check-In credit ($25 one-time bill credit) and "
                "assign a priority support routing tag to their account for the next 90 days."
            ),
        },
        "monthly_charges": {
            "label": f"Monthly Billing Rate (${monthly:,.2f}/mo, SHAP {sign_str})",
            "diagnosis": (
                f"Monthly recurring charge of ${monthly:,.2f} ({direction}) exceeds the segment "
                "comfort threshold relative to their active value-added service bundle."
            ),
            "empathy_line": (
                f"\"Looking at your current ${monthly:,.2f} monthly statement, I noticed you aren't "
                "currently enrolled in our current promotional bundle pricing.\""
            ),
            "rebuttal": (
                f"Apply a targetedRight-Sizing Bundle Audit that reduces monthly out-of-pocket cost "
                f"by 15% (~${monthly * 0.15:,.2f}/mo savings) while preserving current bandwidth."
            ),
        },
        "total_charges": {
            "label": f"Cumulative Spend Profile (SHAP {sign_str})",
            "diagnosis": (
                f"Cumulative lifetime spend pattern ({direction}) signals sensitivity to long-term "
                "rate creep vs. new-customer acquisition pricing."
            ),
            "empathy_line": (
                "\"Loyal subscribers should never pay more than brand-new signups for the exact "
                "same tier of connectivity.\""
            ),
            "rebuttal": (
                "Match our current New-Subscriber Promotional Rate for the next 12 billing cycles "
                "and apply an immediate Loyalty Recognition Credit."
            ),
        },
        "payment_method": {
            "label": f"Billing Channel ({payment}, SHAP {sign_str})",
            "diagnosis": (
                f"Payment via '{payment}' ({direction}) introduces monthly manual payment friction "
                "and recurring bill-shock salience."
            ),
            "empathy_line": (
                "\"Managing manual monthly statements can be a hassle, and we actually reward "
                "accounts that switch to automated billing with a recurring monthly discount.\""
            ),
            "rebuttal": (
                "Offer a $5.00/month ($60/year) AutoPay & Paperless Enrollment Incentive when "
                "transitioning from manual check to automatic bank or card settlement."
            ),
        },
        "online_security": {
            "label": f"Online Security Gap (SHAP {sign_str})",
            "diagnosis": (
                f"Lack of bundled Online Security ({direction}) leaves the household unanchored "
                "to our digital protection ecosystem."
            ),
            "empathy_line": (
                "\"We noticed your connection doesn't currently have our ThreatShield Online "
                "Security suite activated, even though your tier qualifies for a complimentary trial.\""
            ),
            "rebuttal": (
                "Activate 12 months of Online Security & Identity Protection at $0/mo (regularly $9.99/mo) "
                "to increase ecosystem stickiness without raising their bill."
            ),
        },
        "tech_support": {
            "label": f"Priority Tech Support Gap (SHAP {sign_str})",
            "diagnosis": (
                f"Absence of dedicated Tech Support ({direction}) increases frustration whenever "
                "router, device, or Wi-Fi issues occur."
            ),
            "empathy_line": (
                "\"Waiting in a standard support queue when your Wi-Fi drops is frustrating—you "
                "should have direct access to our Tier-2 engineers whenever you need help.\""
            ),
            "rebuttal": (
                "Provision complimentary 24/7 VIP Priority Tech Support for 12 months so any home "
                "network or connected-device issue is resolved in under 3 minutes."
            ),
        },
    }

    if feature_name in playbooks:
        return playbooks[feature_name]

    # Kịch bản dự phòng tổng quát nếu thuộc tính SHAP nằm ngoài danh mục cố định
    pretty_name = feature_name.replace("_", " ").title()
    return {
        "label": f"{pretty_name} (SHAP {sign_str})",
        "diagnosis": f"Feature '{pretty_name}' ({direction}) is a primary local contributor to churn risk.",
        "empathy_line": (
            f"\"I'm reviewing your account profile today specifically around your {pretty_name.lower()} "
            "settings to make sure your plan is optimized for your usage.\""
        ),
        "rebuttal": (
            f"Offer a customized plan optimization addressing {pretty_name.lower()} paired with a "
            "loyalty rate lock and complimentary value-added service upgrade."
        ),
    }


def generate_retention_pitch(
    customer_row: Union[Mapping[str, Any], pd.Series],
) -> Dict[str, Any]:
    """
    Tổng hợp bộ kịch bản giữ chân khách hàng cá nhân hóa 1-1 (`call_script` và `executive_email`)
    dựa trên hồ sơ 360 độ, phân khúc nhân quả Causal Uplift và hai nhân tố rủi ro TreeSHAP hàng đầu.

    Cơ chế tổng hợp thông minh:
        1. Phân tích 2 nhân tố SHAP lớn nhất (`top_shap_driver_1`, `top_shap_driver_2`) để xây dựng
           luận điểm đồng cảm và giải pháp khắc phục đúng điểm đau (pain point) của khách hàng.
        2. Tự động rà soát các dịch vụ gia tăng (VAS) mà khách hàng chưa đăng ký (`OnlineSecurity`,
           `TechSupport`, `OnlineBackup`, `DeviceProtection`) để đóng gói thành quà tặng miễn phí
           nhằm tăng độ gắn kết hệ sinh thái (switching cost) mà không làm tăng hóa đơn hàng tháng.
        3. Kiểm tra cờ nhân quả `is_do_not_disturb` / `Sleeping Dog`: Nếu khách hàng thuộc nhóm
           nhạy cảm ngược, hệ thống tự động chuyển sang chế độ cảnh báo đỏ (chặn gọi điện outbound,
           chỉ nâng cấp chất lượng đường truyền âm thầm ở tầng kỹ thuật).

    Tham số (Args):
        customer_row (Union[Mapping[str, Any], pd.Series]): Bản ghi thông tin khách hàng từ
            `vw_customer_360` hoặc từ bộ giả lập thời gian thực ở Tab 4.

    Trả về (Returns):
        Dict[str, Any]: Từ điển có cấu trúc gồm:
            - `customer_id` (str): Mã định danh khách hàng.
            - `causal_segment` (str): Phân khúc nhân quả ('Persuadable', 'Sleeping Dog', ...).
            - `uplift_score` (float): Điểm Uplift nhân quả.
            - `is_do_not_disturb` (int): Cờ chặn làm phiền (1 hoặc 0).
            - `generation_engine` (str): Tên phiên bản động cơ sinh kịch bản.
            - `call_script` (dict): Gồm `guardrail_banner`, `stage_1_hook`,
              `stage_2_empathy_objection`, `stage_3_incentive_offer`, và `full_markdown`.
            - `executive_email` (dict): Gồm `subject_line`, `body_markdown`, và `cta_summary`.
    """
    row: Mapping[str, Any] = (
        customer_row.to_dict() if isinstance(customer_row, pd.Series) else dict(customer_row)
    )

    # Trích xuất các chỉ số hồ sơ, tài chính và rủi ro của khách hàng
    cid = str(_get_val(row, "customerID", "customerID", "CUST-LIVE-001"))
    value_tier = str(_get_val(row, "value_tier", "value_tier", "High"))
    tenure = int(float(_get_val(row, "tenure", "tenure", 12) or 12))
    monthly = float(_get_val(row, "monthly_charges", "MonthlyCharges", 75.0) or 75.0)
    annual_rev = float(_get_val(row, "annualized_revenue", "annualized_revenue", monthly * 12.0) or (monthly * 12.0))
    p_churn = float(_get_val(row, "churn_probability", "churn_probability", 0.50) or 0.50)
    quadrant = str(_get_val(row, "decision_quadrant", "decision_quadrant", "Quadrant 1 — Protect (High-Value / High-Risk)"))
    rec_action = str(_get_val(row, "recommended_action", "recommended_action", "VIP Concierge Outreach + 20% Annual Contract Discount"))
    int_cost = float(_get_val(row, "intervention_cost", "intervention_cost", 120.0) or 0.0)
    net_benefit = float(_get_val(row, "expected_net_benefit", "expected_net_benefit", 0.0) or 0.0)

    # Trích xuất thông tin phân khúc nhân quả (Causal Uplift)
    causal_seg = str(_get_val(row, "causal_segment", "causal_segment", "Persuadable"))
    uplift_score = float(_get_val(row, "uplift_score", "uplift_score", 0.24) or 0.0)
    is_dnd = int(_get_val(row, "is_do_not_disturb", "is_do_not_disturb", 1 if causal_seg == "Sleeping Dog" else 0) or 0)

    # Trích xuất cấu hình dịch vụ đang sử dụng
    contract = str(_get_val(row, "contract", "Contract", "Month-to-month"))
    internet = str(_get_val(row, "internet_service", "InternetService", "Fiber optic"))
    payment = str(_get_val(row, "payment_method", "PaymentMethod", "Electronic check"))
    online_sec = str(_get_val(row, "online_security", "OnlineSecurity", "No"))
    tech_support = str(_get_val(row, "tech_support", "TechSupport", "No"))
    online_bkp = str(_get_val(row, "online_backup", "OnlineBackup", "No"))
    dev_prot = str(_get_val(row, "device_protection", "DeviceProtection", "No"))

    # Phân giải 2 nhân tố SHAP hàng đầu và xây dựng kịch bản xử lý từ chối tương ứng
    shap_1_raw = str(_get_val(row, "top_shap_driver_1", "top_shap_driver_1", "contract (+1.10)"))
    shap_2_raw = str(_get_val(row, "top_shap_driver_2", "top_shap_driver_2", "internet_service (+0.65)"))
    f1_name, f1_val = _parse_shap_driver(shap_1_raw)
    f2_name, f2_val = _parse_shap_driver(shap_2_raw)

    pb1 = _build_shap_objection_playbook(f1_name, f1_val, row)
    pb2 = _build_shap_objection_playbook(f2_name, f2_val, row)

    # Phát hiện các dịch vụ giá trị gia tăng (VAS) còn thiếu để đưa vào gói quà tặng giữ chân
    missing_vas = []
    if online_sec == "No" and internet != "No":
        missing_vas.append("ThreatShield Online Security ($9.99/mo value)")
    if tech_support == "No" and internet != "No":
        missing_vas.append("24/7 Priority VIP Tech Support ($12.99/mo value)")
    if online_bkp == "No" and internet != "No":
        missing_vas.append("Cloud Vault 500GB Automated Backup ($6.99/mo value)")
    if dev_prot == "No" and internet != "No":
        missing_vas.append("Complete Home Device Protection ($7.99/mo value)")
    if not missing_vas:
        missing_vas.append("Complimentary Speed Tier Boost & Wi-Fi 6E Mesh Upgrade")

    featured_perks = " + ".join(missing_vas[:2])

    # Xác định cơ chế chiết khấu và thông điệp cảnh báo nhân quả theo Value Tier & Causal Segment
    if is_dnd == 1 or causal_seg == "Sleeping Dog":
        discount_pct = 0
        monthly_savings = 0.0
        offer_headline = "PASSIVE SILENT UPGRADE ONLY — DO NOT INITIATE OUTBOUND RENEWAL CALL"
        guardrail_banner = (
            "⛔ CAUSAL UPLIFT WARNING (SLEEPING DOG | tau = "
            f"{uplift_score:+.3f}): Outbound marketing or contract-renewal contact is predicted "
            "to TRIGGER churn. Suppress outbound telesales calls. Use the script below ONLY if "
            "the customer initiates an inbound support inquiry."
        )
    elif causal_seg == "Lost Cause":
        discount_pct = 10
        monthly_savings = round(monthly * 0.10, 2)
        offer_headline = "Zero-Spend Automated Digital Win-Back Workflow ($0 Outbound Budget)"
        guardrail_banner = (
            f"⚠️ CAUSAL LOST CAUSE (tau = {uplift_score:+.3f} | Churn Risk = {p_churn*100:.1f}%): "
            "High probability of attrition with low treatment responsiveness. Preserve human AM "
            "capacity and deploy zero-marginal-cost automated digital self-service offer only."
        )
    elif value_tier == "High":
        discount_pct = 20
        monthly_savings = round(monthly * 0.20, 2)
        offer_headline = (
            f"VIP Executive Rate Lock: 20% Loyalty Discount (-${monthly_savings:,.2f}/mo) + "
            f"12 Months Complimentary {featured_perks}"
        )
        guardrail_banner = (
            f"✅ HIGH-PRIORITY PERSUADABLE TARGET (tau = {uplift_score:+.3f} | Expected Net Benefit = "
            f"${net_benefit:,.2f}): Strong positive treatment uplift. Authorized Intervention Budget: ${int_cost:,.2f}."
        )
    elif value_tier == "Medium":
        discount_pct = 15
        monthly_savings = round(monthly * 0.15, 2)
        offer_headline = (
            f"Preferred Member Bundle: 15% Monthly Rate Reduction (-${monthly_savings:,.2f}/mo) + "
            f"6 Months Complimentary {featured_perks}"
        )
        guardrail_banner = (
            f"✅ PERSUADABLE GROWTH TARGET (tau = {uplift_score:+.3f} | Expected Net Benefit = "
            f"${net_benefit:,.2f}): Positive causal responsiveness to digital bundle incentives."
        )
    else:
        discount_pct = 10
        monthly_savings = round(monthly * 0.10, 2)
        offer_headline = (
            f"Automated Stay-Connected Credit: $5/mo AutoPay Bonus + {featured_perks}"
        )
        guardrail_banner = (
            f"ℹ️ AUTOMATED RETENTION WORKFLOW (Causal Segment: {causal_seg} | tau = {uplift_score:+.3f}): "
            "Deploy low-cost digital retention touchpoint."
        )

    # -------------------------------------------------------------------------
    # Tổng hợp Kịch bản Gọi điện 3 Bước (Stage 1: Hook, Stage 2: Empathy, Stage 3: Offer)
    # -------------------------------------------------------------------------
    tenure_phrase = (
        f"over the past {tenure} months" if tenure >= 2 else "since joining our network"
    )
    stage_1_hook = (
        f"\"Hi [Customer Name], this is [Account Manager] from the Executive Account Care team. "
        f"I'm reaching out—not with a sales call—but because I saw you've been a {value_tier}-Tier "
        f"{internet} member with us {tenure_phrase} on our ${monthly:,.2f}/month plan "
        f"(${annual_rev:,.2f}/yr annualized value), and your account flagged as eligible for an "
        f"unclaimed loyalty rate review.\""
    )
    if is_dnd == 1 or causal_seg == "Sleeping Dog":
        stage_1_hook = (
            "[INBOUND-ONLY REACTIVE GREETING — DO NOT CALL OUTBOUND]\n"
            f"\"Thank you for calling Customer Care! While I resolve your request today, I want "
            f"to thank you for being with us {tenure_phrase}—I've quietly applied a complimentary "
            f"network routing optimization to your {internet} line with zero changes to your "
            f"current ${monthly:,.2f}/month flexible plan.\""
        )

    stage_2_empathy = (
        f"• Primary SHAP Risk Driver #1 — {pb1['label']}:\n"
        f"  - Root Cause: {pb1['diagnosis']}\n"
        f"  - Empathy Statement: {pb1['empathy_line']}\n"
        f"  - Objection Rebuttal: {pb1['rebuttal']}\n\n"
        f"• Secondary SHAP Risk Driver #2 — {pb2['label']}:\n"
        f"  - Root Cause: {pb2['diagnosis']}\n"
        f"  - Empathy Statement: {pb2['empathy_line']}\n"
        f"  - Objection Rebuttal: {pb2['rebuttal']}"
    )

    new_monthly = max(monthly - monthly_savings, 19.99)
    stage_3_offer = (
        f"• Recommended Policy Action: {rec_action}\n"
        f"• Tailored Offer Package: {offer_headline}\n"
        f"• Financial Value to Customer: Reduces monthly bill from ${monthly:,.2f}/mo to "
        f"${new_monthly:,.2f}/mo (saving ${monthly_savings * 12.0:,.2f}/year) while unlocking "
        f"{featured_perks} at $0 extra charge.\n"
        f"• Closing Ask: \"With your permission, I can apply this ${monthly_savings:,.2f}/month "
        f"loyalty reduction and activate {missing_vas[0]} effective on your very next statement—shall "
        f"I lock that in for you right now?\""
    )
    if is_dnd == 1 or causal_seg == "Sleeping Dog":
        stage_3_offer = (
            "• Recommended Policy Action: ⛔ DO NOT DISTURB — Sleeping Dog Suppression ($0 Spend)\n"
            "• Causal Protocol: Do NOT pitch a 1-year or 2-year contract renewal and do NOT draw "
            "attention to monthly billing rates. Any unsolicited contract pitch increases cancellation "
            f"probability (Estimated Uplift tau = {uplift_score:+.3f}).\n"
            "• Silent Retention Action: Automatically provision backend QoS traffic prioritization and "
            "suppress all promotional SMS/email blasts for 180 days."
        )

    call_script_markdown = (
        f"### 🎯 Account Manager Telesales Playbook — `{cid}`\n"
        f"> **{guardrail_banner}**\n\n"
        f"#### 1. Personalized Opening Hook (`tenure={tenure}m`, `monthly=${monthly:,.2f}`, `tier={value_tier}`)\n"
        f"{stage_1_hook}\n\n"
        f"#### 2. TreeSHAP Empathy & Objection Handling (`{shap_1_raw}` & `{shap_2_raw}`)\n"
        f"{stage_2_empathy}\n\n"
        f"#### 3. Tailored Retention Incentive & Close (`{quadrant}`)\n"
        f"{stage_3_offer}\n"
    )

    # -------------------------------------------------------------------------
    # Tổng hợp Email Chăm sóc Khách hàng Cá nhân hóa (Executive Customer Email)
    # -------------------------------------------------------------------------
    if is_dnd == 1 or causal_seg == "Sleeping Dog":
        subject_line = (
            f"Service Enhancement Notice: Complimentary Network Optimization on Account #{cid}"
        )
        email_body = (
            f"**Subject:** {subject_line}\n\n"
            f"Dear Valued Member (`{cid}`),\n\n"
            f"Thank you for choosing us for your **{internet}** connectivity over the past **{max(tenure, 1)} months**. "
            f"We are writing with a brief operational update—**no action or plan change is required on your part.**\n\n"
            f"As part of our regional network upgrade, we have automatically enabled **Priority Packet Routing** "
            f"and **Complimentary {missing_vas[0]}** on your line:\n"
            f"- **Monthly Billing Impact:** `$0.00` (Your current `${monthly:,.2f}/mo` `{contract}` rate remains untouched)\n"
            f"- **Contract Commitment Required:** `None` (Your full plan flexibility is preserved)\n"
            f"- **Service Activation:** Automatic on your next billing cycle\n\n"
            f"We appreciate having you with us. If you ever need technical assistance, our engineering desk is "
            f"available 24/7 via your member portal.\n\n"
            f"Warm regards,\n"
            f"**Executive Network Care Team**  \n"
            f"*ISM Customer Value & Experience Desk*"
        )
    else:
        subject_line = (
            f"Exclusive {value_tier}-Member Rate Review: Save ${monthly_savings * 12.0:,.0f}/Year + "
            f"Complimentary {missing_vas[0].split('(')[0].strip()} (#{cid})"
        )
        autopay_note = (
            f"- **Bonus AutoPay Credit:** Switch from `{payment}` to Automated Billing for an extra **$5.00/mo ($60/yr) credit**\n"
            if "automatic" not in payment.lower()
            else f"- **AutoPay Recognition:** Continues your existing `{payment}` convenience benefit\n"
        )
        email_body = (
            f"**Subject:** {subject_line}\n\n"
            f"Dear Valued `{value_tier}`-Tier Member (`{cid}`),\n\n"
            f"Thank you for being a valued **{internet}** subscriber with us for **{max(tenure, 1)} months**. "
            f"Following a recent executive review of your account, we identified an opportunity to lower your "
            f"monthly statement while upgrading your household protection bundle.\n\n"
            f"### 🎁 Your Personalized Loyalty Upgrade Package\n"
            f"- **Immediate Rate Reduction:** **{discount_pct}% Loyalty Credit**, lowering your monthly bill from "
            f"**${monthly:,.2f}/mo** to **${new_monthly:,.2f}/mo** (**${monthly_savings * 12.0:,.2f} annual savings**)\n"
            f"- **Complimentary Value-Added Bundle ($0/mo for 12 Months):** **{featured_perks}**\n"
            f"- **Price-Lock Protection:** Guaranteed rate stability with zero surprise fee increases\n"
            f"{autopay_note}\n"
            f"### ⚡ Why We Prepared This for Your Account\n"
            f"We noticed your account is currently on a **{contract}** structure without **{missing_vas[0].split('(')[0].strip()}** "
            f"activated. We believe loyal members should always receive our best rate and our highest tier of "
            f"technical support without paying extra.\n\n"
            f"👉 **[Click Here to Activate Your ${monthly_savings:,.2f}/Month Savings & Complimentary Bundle in 1 Click](#)**  \n"
            f"*(Or reply directly to this email and your dedicated Account Manager will apply the credit immediately.)*\n\n"
            f"Warm regards,\n"
            f"**Executive Customer Retention & Concierge Desk**  \n"
            f"*ISM Telecommunications Decision Support Division*"
        )

    return {
        "customer_id": cid,
        "causal_segment": causal_seg,
        "uplift_score": round(uplift_score, 4),
        "is_do_not_disturb": is_dnd,
        "generation_engine": "ISM Contextual GenAI Copilot v3.0 (SHAP + Causal Uplift Grounded)",
        "call_script": {
            "guardrail_banner": guardrail_banner,
            "stage_1_hook": stage_1_hook,
            "stage_2_empathy_objection": stage_2_empathy,
            "stage_3_incentive_offer": stage_3_offer,
            "full_markdown": call_script_markdown,
        },
        "executive_email": {
            "subject_line": subject_line,
            "body_markdown": email_body,
            "cta_summary": offer_headline,
        },
    }


if __name__ == "__main__":
    # Chạy kiểm thử nhanh (Smoke-test) cho cả khách hàng VIP thuộc nhóm Persuadable và khách hàng nhóm Sleeping Dog
    sample_persuadable = {
        "customerID": "9999-PERS",
        "value_tier": "High",
        "tenure": 9,
        "monthly_charges": 98.50,
        "annualized_revenue": 1182.00,
        "churn_probability": 0.74,
        "decision_quadrant": "Quadrant 1 — Protect (High-Value / High-Risk)",
        "recommended_action": "VIP Concierge Outreach + 20% Annual Contract Discount",
        "intervention_cost": 120.0,
        "expected_net_benefit": 185.0,
        "causal_segment": "Persuadable",
        "uplift_score": 0.342,
        "is_do_not_disturb": 0,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "online_security": "No",
        "tech_support": "No",
        "top_shap_driver_1": "contract (+1.14)",
        "top_shap_driver_2": "internet_service (+0.68)",
    }
    res1 = generate_retention_pitch(sample_persuadable)
    print(f"[COPILOT] Generated Persuadable pitch for {res1['customer_id']} -> Subject: {res1['executive_email']['subject_line']}")

    sample_sd = dict(sample_persuadable, customerID="8888-SDOG", causal_segment="Sleeping Dog", uplift_score=-0.165, is_do_not_disturb=1)
    res2 = generate_retention_pitch(sample_sd)
    print(f"[COPILOT] Generated Sleeping Dog pitch for {res2['customer_id']} -> Subject: {res2['executive_email']['subject_line']}")
