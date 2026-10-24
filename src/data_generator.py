"""
Module Thu thập & Khởi tạo Dữ liệu Gốc (Tier 1 - Data Layer).

Vai trò trong Kiến trúc 4-Tier ISM:
    Đóng vai trò là điểm đầu vào (Ingestion Layer) của toàn bộ hệ thống Hỗ trợ Ra quyết định (DSS),
    đảm bảo cung cấp tập dữ liệu chuẩn IBM Telco Customer Churn (7.043 khách hàng x 21 thuộc tính)
    và lưu trữ nguyên bản tại ``data/raw/telco_churn_raw.csv``.

Tác giả (Group 6 - HCMIU):
    - Vũ Viết Phong (ITDSIU19048)
    - Trương Đăng Khoa (ITDSIU19027)
    - Trần Khánh Toàn (ITITIU20322)
    - Nguyễn Lê Mai Hương (ITITWE23040)

Các chức năng chính:
    1. Tải tự động tập dữ liệu gốc từ danh sách các kho lưu trữ công khai (IBM / Kaggle mirrors).
    2. Kiểm định tính toàn vẹn của Schema (đúng 7.043 dòng và 21 cột chuẩn).
    3. Cơ chế dự phòng ngoại tuyến (Synthetic Fallback Generator): Tự động sinh tập dữ liệu
       mô phỏng có cùng phân phối thống kê, tỷ lệ rời mạng (~26,5%) và tái hiện chính xác
       11 dòng ``TotalCharges`` bị trống khi ``tenure = 0`` nếu mất kết nối Internet.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Cấu hình Đường dẫn & Hằng số Kiểm định Schema
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "telco_churn_raw.csv"

# Danh sách các URL dự phòng chứa tập dữ liệu chuẩn IBM Telco Customer Churn
DATASET_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/carlosfab/dsnp2/master/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

EXPECTED_ROWS = 7043
EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]
REQUEST_TIMEOUT = 30
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Nhóm hàm Tải dữ liệu & Kiểm định Schema
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> bool:
    """
    Kiểm tra xem DataFrame đầu vào có khớp tuyệt đối với Schema chuẩn của bài toán hay không.

    Tham số (Args):
        df (pd.DataFrame): Bảng dữ liệu thô vừa tải về hoặc đọc từ đĩa.

    Trả về (Returns):
        bool: ``True`` nếu bảng có đúng 21 cột theo thứ tự chuẩn và đủ 7.043 bản ghi.
    """
    return list(df.columns) == EXPECTED_COLUMNS and len(df) == EXPECTED_ROWS


def download_dataset() -> Optional[pd.DataFrame]:
    """
    Thử tải tập dữ liệu IBM Telco Customer Churn từ danh sách các mirror công khai.

    Lưu ý kỹ thuật: Cột ``TotalCharges`` được đọc dưới dạng chuỗi (``dtype=str``)
    và tắt cơ chế tự động gán ``NaN`` (``keep_default_na=False``) để bảo toàn nguyên trạng
    11 giá trị khoảng trắng cho tầng ETL (Tier 2) xử lý tường minh.

    Tham số (Args):
        Không có.

    Trả về (Returns):
        Optional[pd.DataFrame]: DataFrame hợp lệ nếu tải thành công, hoặc ``None`` nếu mọi mirror đều lỗi.
    """
    for url in DATASET_URLS:
        try:
            print(f"[INFO] Downloading from: {url}")
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            df = pd.read_csv(
                io.StringIO(response.text),
                dtype={"TotalCharges": str},
                keep_default_na=False,
            )
            if validate_schema(df):
                print(f"[OK]   Download succeeded and schema validated ({len(df):,} rows).")
                return df
            print(f"[WARN] Schema mismatch (rows={len(df)}, cols={df.shape[1]}). Trying next mirror.")
        except Exception as exc:
            print(f"[WARN] Failed: {type(exc).__name__}: {exc}")
    return None


# ---------------------------------------------------------------------------
# Bộ sinh dữ liệu dự phòng (Synthetic Fallback Generator)
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(
    n_rows: int = EXPECTED_ROWS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Sinh dữ liệu mô phỏng tái hiện chính xác cấu trúc và quy luật hành vi của tập dữ liệu IBM Telco.

    Cơ chế mô phỏng kinh tế & hành vi:
        - Hợp đồng ngắn hạn (Month-to-month), cáp quang (Fiber optic), thanh toán bằng séc điện tử
          (Electronic check) và thời gian gắn bó ngắn (low tenure) làm tăng xác suất rời mạng (Churn).
        - Hợp đồng dài hạn (One/Two year) và các dịch vụ bảo mật/hỗ trợ kỹ thuật (OnlineSecurity,
          TechSupport) tạo hiệu ứng neo giữ (lock-in), làm giảm xác suất rời mạng.
        - Tái tạo đúng 11 khách hàng mới gia nhập (``tenure = 0``) có ``TotalCharges = " "``.

    Tham số (Args):
        n_rows (int): Số lượng bản ghi khách hàng cần sinh (mặc định: 7.043).
        seed (int): Hạt giống ngẫu nhiên cố định để đảm bảo tính tái lập 100% (mặc định: 42).

    Trả về (Returns):
        pd.DataFrame: Bảng dữ liệu thô gồm 7.043 dòng x 21 cột chuẩn.
    """
    rng = np.random.default_rng(seed)

    def choice(options, probs, size=n_rows):
        return rng.choice(options, size=size, p=probs)

    # 1. Sinh thuộc tính nhân khẩu học (Demographics)
    gender = choice(["Male", "Female"], [0.505, 0.495])
    senior = choice([0, 1], [0.838, 0.162])
    partner = choice(["Yes", "No"], [0.483, 0.517])
    dependents = np.where(
        partner == "Yes",
        choice(["Yes", "No"], [0.52, 0.48]),
        choice(["Yes", "No"], [0.10, 0.90]),
    )

    # 2. Sinh loại hợp đồng và thời gian gắn bó (Contract & Tenure)
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

    # Tái hiện chính xác 11 dòng khách hàng mới (tenure = 0) của bộ dữ liệu gốc
    tenure[tenure == 0] = 1
    zero_idx = rng.choice(np.where(contract == "Two year")[0], size=11, replace=False)
    tenure[zero_idx] = 0

    # 3. Sinh danh mục dịch vụ viễn thông & dịch vụ giá trị gia tăng (VAS)
    phone = choice(["Yes", "No"], [0.903, 0.097])
    multiple_lines = np.where(
        phone == "No",
        "No phone service",
        choice(["Yes", "No"], [0.467, 0.533]),
    )
    internet = choice(["Fiber optic", "DSL", "No"], [0.440, 0.344, 0.216])

    def addon(p_yes):
        return np.where(
            internet == "No",
            "No internet service",
            choice(["Yes", "No"], [p_yes, 1 - p_yes]),
        )

    online_security, online_backup = addon(0.37), addon(0.44)
    device_protection, tech_support = addon(0.44), addon(0.37)
    streaming_tv, streaming_movies = addon(0.49), addon(0.50)

    paperless = choice(["Yes", "No"], [0.592, 0.408])
    payment = choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        [0.336, 0.229, 0.219, 0.216],
    )

    # 4. Tính cước phí hàng tháng (MonthlyCharges) và tổng cước tích lũy (TotalCharges)
    monthly = np.where(phone == "Yes", 20.0, 0.0)
    monthly += np.where(multiple_lines == "Yes", 5.0, 0.0)
    monthly += np.select([internet == "Fiber optic", internet == "DSL"], [50.0, 25.0], 0.0)
    for svc in [
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
    ]:
        monthly += np.where(svc == "Yes", rng.uniform(5, 10, n_rows), 0.0)
    monthly = np.clip(monthly + rng.normal(0, 2, n_rows), 18.25, 118.75).round(2)
    total = (monthly * tenure * rng.uniform(0.95, 1.05, n_rows)).round(2)

    # 5. Mô hình hóa xác suất rời mạng bằng hàm Logistic (Log-Odds)
    logit = (
        -1.4
        + 1.3 * (contract == "Month-to-month")
        - 1.2 * (contract == "Two year")
        - 0.035 * tenure
        + 0.8 * (internet == "Fiber optic")
        - 0.6 * (internet == "No")
        + 0.6 * (payment == "Electronic check")
        + 0.3 * (paperless == "Yes")
        + 0.3 * senior
        - 0.4 * (online_security == "Yes")
        - 0.4 * (tech_support == "Yes")
        - 0.2 * (partner == "Yes")
        + 0.01 * (monthly - 65)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = np.where(rng.random(n_rows) < churn_prob, "Yes", "No")

    # 6. Khởi tạo mã định danh khách hàng theo đúng định dạng "NNNN-XXXXX"
    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    ids = set()
    while len(ids) < n_rows:
        ids.add(f"{rng.integers(0, 10000):04d}-{''.join(rng.choice(letters, 5))}")
    customer_ids = list(ids)

    # Ghi nhận khoảng trắng " " cho TotalCharges đối với khách hàng có tenure == 0
    total_str = np.where(tenure == 0, " ", total.astype(str))

    return pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_str,
            "Churn": churn,
        }
    )[EXPECTED_COLUMNS]


# ---------------------------------------------------------------------------
# Thống kê Kiểm toán Dữ liệu Thô (Data Profiling)
# ---------------------------------------------------------------------------
def print_data_profile(df: pd.DataFrame, source: str) -> None:
    """
    In bảng báo cáo kiểm định chất lượng dữ liệu thô (Data Quality Profile) ra màn hình console.

    Tham số (Args):
        df (pd.DataFrame): Bảng dữ liệu thô cần thống kê.
        source (str): Nguồn gốc dữ liệu (File cục bộ, Mirror trực tuyến, hoặc Synthetic Fallback).

    Trả về (Returns):
        None.
    """
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
# Hàm Điều phối Chính (Entry Point)
# ---------------------------------------------------------------------------
def acquire(force: bool = False, synthetic: bool = False) -> pd.DataFrame:
    """
    Thu thập dữ liệu gốc và lưu trữ bền vững tại ``data/raw/telco_churn_raw.csv``.

    Tham số (Args):
        force (bool): Nếu ``True``, bắt buộc tải lại dữ liệu ngay cả khi file cục bộ đã tồn tại.
        synthetic (bool): Nếu ``True``, bỏ qua bước tải mạng và dùng bộ sinh dữ liệu mô phỏng.

    Trả về (Returns):
        pd.DataFrame: Bảng dữ liệu thô đã được kiểm định Schema.
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
    """
    Hàm thực thi dòng lệnh (CLI) cho tiến trình thu thập dữ liệu.

    Tham số (Args):
        Không có (đọc từ ``sys.argv`` qua ``argparse``).

    Trả về (Returns):
        int: Mã thoát hệ thống (``0`` nếu thành công, ``1`` nếu lỗi Schema).
    """
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
