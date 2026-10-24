"""
Module Đường ống Xử lý Dữ liệu & Kho dữ liệu DSS (Tier 2 - ETL & Data Processing Layer).

Vai trò trong Kiến trúc 4-Tier ISM:
    Thực hiện quy trình trích xuất (Extract), làm sạch & kiến tạo đặc trưng (Transform),
    và nạp (Load) dữ liệu vào Kho dữ liệu quan hệ SQLite (``database/ism_dss.db``) được
    thiết kế theo mô hình lai Chuẩn hóa 3NF kết hợp Lược đồ Hình sao (Hybrid 3NF + Star Schema).

Tác giả (Group 6 - HCMIU):
    - Vũ Viết Phong (ITDSIU19048)
    - Trương Đăng Khoa (ITDSIU19027)
    - Trần Khánh Toàn (ITITIU20322)
    - Nguyễn Lê Mai Hương (ITITWE23040)

Các chức năng chính:
    1. EXTRACT   - Đọc dữ liệu thô từ ``data/raw/telco_churn_raw.csv``.
    2. TRANSFORM - Làm sạch khoảng trắng, xử lý 11 dòng ``TotalCharges`` bị trống (gán ``0.0``
                   do ``tenure = 0``), mã hóa biến nhị phân và tạo 5 đặc trưng quản trị mới
                   (``tenure_group``, ``total_services_count``, ``avg_monthly_spend``,
                   ``is_auto_payment``, ``churn_binary``).
    3. LOAD      - Khởi tạo Schema CSDL gồm 3 bảng chiều 3NF (``customers``, ``services``, ``billing``),
                   2 bảng sự kiện (``fact_mining_scores``, ``fact_retention_actions``),
                   2 khung nhìn tích hợp (``customer_master``, ``vw_customer_360``) và bảng kiểm toán
                   nhật ký vận hành (``etl_audit_log``).
    4. VALIDATE  - Kiểm tra khóa ngoại, giá trị rỗng (NULL), số lượng bản ghi và xuất CSV đã làm sạch.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cấu hình Đường dẫn & Quy tắc Ánh xạ Thuộc tính
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "telco_churn_raw.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "telco_churn_processed.csv"
DB_DIR = PROJECT_ROOT / "database"
DB_FILE = DB_DIR / "ism_dss.db"

EXPECTED_ROWS = 7043

# Ánh xạ tên cột từ định dạng CamelCase gốc sang chuẩn snake_case trong CSDL
# Giữ nguyên khóa nghiệp vụ ``customerID`` để liên kết xuyên suốt các bảng
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

# Danh sách các cột nhị phân thuần túy (Yes/No) cần chuyển về số nguyên 1/0
BINARY_COLUMNS = ["partner", "dependents", "phone_service", "paperless_billing"]

# Danh sách 6 dịch vụ giá trị gia tăng (Value-Added Services - VAS) dùng để tính độ gắn kết hệ sinh thái
VALUE_ADDED_SERVICES = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

# Phân nhóm vòng đời khách hàng theo số tháng gắn bó (Tenure Lifecycle Bins)
TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12m", "12-24m", "24-48m", "48m+"]

# Danh sách phương thức thanh toán tự động (giúp giảm ma sát thanh toán hàng tháng)
AUTO_PAYMENT_METHODS = ["Bank transfer (automatic)", "Credit card (automatic)"]


# ---------------------------------------------------------------------------
# 1. EXTRACT (Trích xuất dữ liệu thô)
# ---------------------------------------------------------------------------
def extract(path: Path = RAW_FILE) -> pd.DataFrame:
    """
    Đọc file CSV dữ liệu thô mà không tự động ép kiểu trên cột ``TotalCharges``.

    Tham số (Args):
        path (Path): Đường dẫn tới file CSV dữ liệu thô (mặc định: ``data/raw/telco_churn_raw.csv``).

    Trả về (Returns):
        pd.DataFrame: DataFrame dữ liệu thô gồm 7.043 dòng x 21 cột.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run 'src/data_generator.py' first."
        )
    df = pd.read_csv(path, dtype={"TotalCharges": str}, keep_default_na=False)
    print(f"[EXTRACT]   Loaded {len(df):,} rows x {df.shape[1]} columns from {path.relative_to(PROJECT_ROOT)}")
    return df


# ---------------------------------------------------------------------------
# 2. TRANSFORM (Làm sạch & Kiến tạo Đặc trưng Quản trị)
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Thực hiện làm sạch dữ liệu, chuẩn hóa kiểu dữ liệu và xử lý giá trị khuyết thiếu.

    Quy tắc làm sạch:
        - Cắt bỏ khoảng trắng thừa ở đầu và cuối mọi chuỗi văn bản.
        - Chuyển đổi ``TotalCharges`` sang kiểu số thực; 11 dòng bị trống thuộc về khách hàng
          mới đăng ký (``tenure = 0``) chưa phát sinh kỳ hóa đơn đầu tiên nên được gán ``0.0``.
        - Loại bỏ các dòng trùng lặp dựa trên khóa chính ``customerID``.

    Tham số (Args):
        df (pd.DataFrame): DataFrame dữ liệu thô từ bước Extract.

    Trả về (Returns):
        Tuple[pd.DataFrame, Dict[str, Any]]: Bộ gồm (DataFrame đã làm sạch, Từ điển thống kê kiểm toán ETL).
    """
    df = df.rename(columns=COLUMN_MAP).copy()
    stats: Dict[str, Any] = {}

    # Chuẩn hóa chuỗi ký tự trên toàn bộ các cột kiểu văn bản
    text_cols = df.select_dtypes(include="object").columns
    df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())

    # Kiểm tra và xử lý 11 dòng TotalCharges bị trống khi tenure == 0
    blank_mask = df["total_charges"].eq("")
    stats["blank_total_charges"] = int(blank_mask.sum())
    stats["blank_with_zero_tenure"] = int((blank_mask & df["tenure"].eq(0)).sum())

    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    stats["non_numeric_total_charges"] = int(df["total_charges"].isna().sum())
    df["total_charges"] = df["total_charges"].fillna(0.0)

    # Khử trùng lặp theo khóa chính customerID
    before = len(df)
    df = df.drop_duplicates(subset="customerID", keep="first")
    stats["duplicates_removed"] = before - len(df)

    # Ép kiểu tường minh cho các trường số học
    df["senior_citizen"] = df["senior_citizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["monthly_charges"] = df["monthly_charges"].astype(float)

    print(
        f"[TRANSFORM] Imputed {stats['blank_total_charges']} blank TotalCharges -> 0.0 "
        f"({stats['blank_with_zero_tenure']} with tenure = 0); "
        f"duplicates removed: {stats['duplicates_removed']}"
    )
    return df, stats


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mã hóa thuộc tính nhị phân và kiến tạo 5 đặc trưng nghiệp vụ (Engineered Features) phục vụ Data Mining.

    Công thức kiến tạo đặc trưng:
        - ``tenure_group``: Phân tầng vòng đời khách hàng (``0-12m``, ``12-24m``, ``24-48m``, ``48m+``).
        - ``total_services_count``: Tổng số dịch vụ GTGT (VAS) đang sử dụng (từ 0 đến 6 dịch vụ).
        - ``avg_monthly_spend``: Chi tiêu trung bình hàng tháng thực tế = ``total_charges / tenure``
          (nếu ``tenure == 0`` thì lấy bằng ``monthly_charges``).
        - ``is_auto_payment``: Cờ thanh toán tự động (``1`` nếu chuyển khoản/thẻ tín dụng tự động, ``0`` nếu thủ công).
        - ``churn_binary``: Biến mục tiêu nhị phân phục vụ huấn luyện mô hình học có giám sát (``1`` = Rời mạng).

    Tham số (Args):
        df (pd.DataFrame): DataFrame đã qua bước làm sạch ``clean()``.

    Trả về (Returns):
        pd.DataFrame: DataFrame hoàn chỉnh gồm 26 cột sẵn sàng nạp vào Kho dữ liệu.
    """
    df = df.copy()

    for col in BINARY_COLUMNS:
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype(int)

    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    df["total_services_count"] = (df[VALUE_ADDED_SERVICES] == "Yes").sum(axis=1).astype(int)
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["total_charges"] / df["tenure"].replace(0, 1),
        df["monthly_charges"],
    ).round(2)
    df["is_auto_payment"] = df["payment_method"].isin(AUTO_PAYMENT_METHODS).astype(int)
    df["churn_binary"] = df["churn"].map({"Yes": 1, "No": 0}).astype(int)

    print(
        "[TRANSFORM] Engineered features: tenure_group, total_services_count, "
        "avg_monthly_spend, is_auto_payment, churn_binary"
    )
    return df


# ---------------------------------------------------------------------------
# 3. LOAD - Định nghĩa Lược đồ Kho dữ liệu DSS (DDL) & Nạp dữ liệu
# ---------------------------------------------------------------------------
SCHEMA_DDL = """
-- Xóa các View và Bảng theo thứ tự phụ thuộc ngược để đảm bảo tính Idempotent khi chạy lại
DROP VIEW  IF EXISTS vw_customer_360;
DROP VIEW  IF EXISTS customer_master;
DROP TABLE IF EXISTS fact_retention_actions;
DROP TABLE IF EXISTS fact_mining_scores;
DROP TABLE IF EXISTS billing;
DROP TABLE IF EXISTS services;
DROP TABLE IF EXISTS customers;

-- ---------------------------------------------------------------------------
-- Bảng chiều 1 (Dimension 1): Thông tin nhân khẩu học, hợp đồng và trạng thái rời mạng
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
-- Bảng chiều 2 (Dimension 2): Danh mục dịch vụ viễn thông & GTGT (Quan hệ 1:1 với customers)
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
-- Bảng chiều 3 (Dimension 3): Hồ sơ thanh toán & cước phí (Quan hệ 1:1 với customers)
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
-- Bảng sự kiện 1 (Fact 1 - Nạp ở Step 4): Điểm số phân cụm K-Means, xác suất XGBoost & SHAP
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
-- Bảng sự kiện 2 (Fact 2 - Nạp ở Steps 5, 8 & 9): Quyết định giữ chân, ROI, Causal Uplift & CRM
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
    updated_at              TEXT,
    uplift_score            REAL    NOT NULL DEFAULT 0.0,
    causal_segment          TEXT    NOT NULL DEFAULT 'Sure Thing' CHECK (causal_segment IN ('Persuadable', 'Sleeping Dog', 'Lost Cause', 'Sure Thing')),
    is_do_not_disturb       INTEGER NOT NULL DEFAULT 0 CHECK (is_do_not_disturb IN (0, 1))
);
CREATE INDEX idx_actions_quadrant ON fact_retention_actions(decision_quadrant);

-- ---------------------------------------------------------------------------
-- View tích hợp 1: Hồ sơ khách hàng chuẩn hóa (kết nối 3 bảng chiều 3NF)
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
-- View phân tích 2: Góc nhìn Khách hàng 360 độ (47 cột) phục vụ trực tiếp cho Streamlit DSS
-- ---------------------------------------------------------------------------
CREATE VIEW vw_customer_360 AS
SELECT
    m.*,
    f.value_tier, f.cluster_id, f.churn_probability, f.risk_level,
    f.top_shap_driver_1, f.top_shap_driver_2, f.scored_at,
    r.annualized_revenue, r.revenue_at_risk, r.decision_quadrant, r.recommended_action,
    r.intervention_cost, r.expected_saved_revenue, r.expected_net_benefit, r.expected_roi,
    r.action_status, r.manager_notes, r.updated_at,
    r.uplift_score, r.causal_segment, r.is_do_not_disturb
FROM customer_master m
LEFT JOIN fact_mining_scores     f ON f.customerID = m.customerID
LEFT JOIN fact_retention_actions r ON r.customerID = m.customerID;

-- ---------------------------------------------------------------------------
-- Bảng kiểm toán nhật ký ETL (Lưu vết lịch sử các lần chạy đường ống dữ liệu)
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

CUSTOMERS_COLS = [
    "customerID",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "tenure_group",
    "contract",
    "churn",
    "churn_binary",
]
SERVICES_COLS = [
    "customerID",
    "phone_service",
    "multiple_lines",
    "internet_service",
    *VALUE_ADDED_SERVICES,
    "total_services_count",
]
BILLING_COLS = [
    "customerID",
    "paperless_billing",
    "payment_method",
    "is_auto_payment",
    "monthly_charges",
    "total_charges",
    "avg_monthly_spend",
]


def load(df: pd.DataFrame, db_path: Path = DB_FILE) -> None:
    """
    Khởi tạo Schema Kho dữ liệu DSS và nạp dữ liệu vào 3 bảng chiều chuẩn hóa 3NF.

    Sử dụng context manager ``with sqlite3.connect(db_path)`` kết hợp ``PRAGMA foreign_keys = ON``
    để đảm bảo tính toàn vẹn tham chiếu và tự động giải phóng kết nối CSDL an toàn.

    Tham số (Args):
        df (pd.DataFrame): DataFrame đã qua bước biến đổi và tạo đặc trưng.
        db_path (Path): Đường dẫn tới file cơ sở dữ liệu SQLite (mặc định: ``database/ism_dss.db``).

    Trả về (Returns):
        None.
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
# 4. VALIDATE & EXPORT (Kiểm định Chất lượng Kho dữ liệu & Xuất file CSV)
# ---------------------------------------------------------------------------
def validate_and_export(db_path: Path = DB_FILE) -> pd.DataFrame:
    """
    Thực thi bộ kiểm thử chất lượng dữ liệu trên Kho dữ liệu SQLite và xuất file CSV sạch.

    Các tiêu chí kiểm định bắt buộc:
        - Số dòng của ``customers``, ``services``, ``billing`` và ``customer_master`` đúng bằng 7.043.
        - Không tồn tại bất kỳ giá trị khuyết thiếu (NULL / NaN) nào trong ``customer_master``.
        - ``PRAGMA foreign_key_check`` xác nhận không có lỗi vi phạm khóa ngoại.
        - Khóa chính ``customerID`` là duy nhất 100%.

    Tham số (Args):
        db_path (Path): Đường dẫn tới file SQLite cần kiểm định.

    Trả về (Returns):
        pd.DataFrame: Bảng dữ liệu tích hợp ``customer_master`` đã được xuất ra ``data/processed/``.
    """
    with sqlite3.connect(db_path) as conn:
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in [
                "customers",
                "services",
                "billing",
                "customer_master",
                "fact_mining_scores",
                "fact_retention_actions",
            ]
        }
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
    print(
        " total_services_count: mean = {:.2f}, min = {}, max = {}".format(
            master["total_services_count"].mean(),
            master["total_services_count"].min(),
            master["total_services_count"].max(),
        )
    )
    print(
        " avg_monthly_spend   : mean = {:.2f}, min = {:.2f}, max = {:.2f}".format(
            master["avg_monthly_spend"].mean(),
            master["avg_monthly_spend"].min(),
            master["avg_monthly_spend"].max(),
        )
    )
    print(" is_auto_payment     : {:.1f}% of customers".format(master["is_auto_payment"].mean() * 100))
    print("=" * 64)
    return master


def log_run(
    stats: Dict[str, Any],
    rows_extracted: int,
    rows_loaded: int,
    status: str,
) -> None:
    """
    Ghi nhận một bản ghi kiểm toán mới vào bảng ``etl_audit_log`` sau mỗi lần chạy ETL.

    Tham số (Args):
        stats (Dict[str, Any]): Từ điển thống kê làm sạch dữ liệu từ hàm ``clean()``.
        rows_extracted (int): Tổng số dòng đọc từ tầng Raw.
        rows_loaded (int): Tổng số dòng nạp thành công vào tầng Warehouse.
        status (str): Trạng thái thực thi (ví dụ: ``'SUCCESS'`` hoặc ``'FAILED'``).

    Trả về (Returns):
        None.
    """
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT INTO etl_audit_log (run_at, source_file, rows_extracted, rows_loaded, "
            "blank_total_charges, duplicates_removed, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                str(RAW_FILE.relative_to(PROJECT_ROOT)),
                rows_extracted,
                rows_loaded,
                stats.get("blank_total_charges", 0),
                stats.get("duplicates_removed", 0),
                status,
            ),
        )


# ---------------------------------------------------------------------------
# Hàm Điều phối Đường ống ETL (Entry Point)
# ---------------------------------------------------------------------------
def run_pipeline() -> pd.DataFrame:
    """
    Thực thi trọn vẹn quy trình 4 bước: Extract -> Transform -> Load -> Validate.

    Tham số (Args):
        Không có.

    Trả về (Returns):
        pd.DataFrame: Bảng dữ liệu tích hợp ``customer_master`` sau khi hoàn tất ETL.
    """
    raw = extract()
    cleaned, stats = clean(raw)
    transformed = engineer_features(cleaned)
    load(transformed)
    master = validate_and_export()
    log_run(stats, rows_extracted=len(raw), rows_loaded=len(master), status="SUCCESS")
    return master


def main() -> int:
    """
    Điểm khởi chạy CLI cho module ETL Pipeline.

    Tham số (Args):
        Không có.

    Trả về (Returns):
        int: Mã trạng thái kết thúc (``0`` nếu thành công, ``1`` nếu phát sinh ngoại lệ).
    """
    try:
        run_pipeline()
    except Exception as exc:
        print(f"[ERROR] ETL pipeline failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
