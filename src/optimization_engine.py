"""
Module Đối sánh Đa Mô hình, Đường cong Lợi nhuận & Tối ưu hóa Nguồn lực Knapsack — `src/optimization_engine.py`.

Vai trò trong Kiến trúc 4 Tầng (4-Tier ISM Architecture):
    Thuộc Tầng 3 (Tier 3 — Advanced Prescriptive Optimization & Model Governance Layer).
    Module này mở rộng năng lực phân tích và quản trị hệ thống thông tin (ISM) với 3 trụ cột:
        1. Đối sánh Đa Mô hình (Multi-Model Benchmarking — `run_model_benchmark`):
           Huấn luyện và đánh giá công bằng 3 thuật toán phân loại có giám sát trên cùng
           phép chia Stratified 80/20 (5.634 mẫu huấn luyện / 1.409 mẫu kiểm thử) sử dụng
           chung bộ tiền xử lý chống rò rỉ dữ liệu (`ColumnTransformer`):
             - Logistic Regression (mô hình tuyến tính cơ sở với `class_weight='balanced'`)
             - Random Forest Classifier (tổ hợp cây quyết định Bagging với `class_weight='balanced_subsample'`)
             - XGBoost Classifier (mô hình Quán quân Gradient Boosting với `scale_pos_weight`)
           Xuất bảng đối sánh (ROC-AUC, PR-AUC, Recall, Precision, F1-Score, Brier Score,
           Training Time) và tọa độ đường cong ROC / Precision-Recall ra `models/model_benchmark.json`.
        2. Tối ưu hóa Ngưỡng Lợi nhuận (Threshold Profit Curve — `run_threshold_profit_curve`):
           Quét ngưỡng xác suất rủi ro cao (`high_risk_threshold`) từ 0.15 đến 0.85 (bước nhảy 0.05)
           để định lượng sự đánh đổi giữa Chi phí chiến dịch (bao gồm chi phí hao hụt chiết khấu
           cho khách hàng trung thành — cannibalization cost) và Doanh thu giữ chân kỳ vọng,
           từ đó xác định **Ngưỡng Tài chính Tối ưu (Optimal Financial Threshold)**.
        3. Phân bổ Nguồn lực Ràng buộc Kép & Đường biên Pareto (`optimize_under_constraints`):
           Giải bài toán cái túi tham lam (Greedy Knapsack Heuristic) dưới 2 ràng buộc thực tế:
           trần ngân sách tài chính (`max_budget`) và giới hạn số cuộc gọi 1-1 của đội ngũ
           Account Manager (`max_vip_calls`), đồng thời dựng đường cong **Pareto Efficient Frontier**.

Nhóm tác giả (Group 6 — International University, VNU-HCM):
    1. Vũ Viết Phong — ITDSIU19048
    2. Trương Đăng Khoa — ITDSIU19027
    3. Trần Khánh Toàn — ITITIU20322
    4. Nguyễn Lê Mai Hương — ITITWE23040

Các hàm chính:
    - load_customer_360: Tải toàn bộ 7.043 bản ghi từ view `vw_customer_360`.
    - run_model_benchmark: Huấn luyện, đánh giá và lưu kết quả đối sánh 3 mô hình học máy.
    - run_threshold_profit_curve: Quét ngưỡng xác suất và tìm điểm tối ưu hóa lợi ích ròng.
    - optimize_under_constraints: Phân bổ ngân sách theo thuật toán Knapsack và tạo đường biên Pareto.
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

# Cấu hình đường dẫn kho dữ liệu và các tệp JSON xuất bản
DB_FILE = PROJECT_ROOT / "database" / "ism_dss.db"
MODELS_DIR = PROJECT_ROOT / "models"
BENCHMARK_FILE = MODELS_DIR / "model_benchmark.json"
PROFIT_CURVE_FILE = MODELS_DIR / "threshold_profit_curve.json"

# Hai góc phần tư chiến lược đòi hỏi nhân sự Account Manager gọi điện trực tiếp 1-1
VIP_CALL_QUADRANTS = {"Critical VIP Retention", "Proactive VIP Care"}


# ---------------------------------------------------------------------------
# Tầng Truy cập Dữ liệu (Data Loader)
# ---------------------------------------------------------------------------
def load_customer_360(db_path: Path = DB_FILE) -> pd.DataFrame:
    """
    Tải toàn bộ 7.043 bản ghi từ view tích hợp `vw_customer_360` trong `database/ism_dss.db`.

    Tham số (Args):
        db_path (Path): Đường dẫn tới tệp cơ sở dữ liệu SQLite.

    Trả về (Returns):
        pd.DataFrame: DataFrame chứa đầy đủ hồ sơ khách hàng, điểm khai phá và quyết định giữ chân.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM vw_customer_360 ORDER BY customerID", conn)
    print(f"[DATA]      Loaded {len(df):,} records from vw_customer_360")
    return df


# ---------------------------------------------------------------------------
# 1. Đối sánh Đa Mô hình (Multi-Model Benchmarking: LR vs. RF vs. XGBoost)
# ---------------------------------------------------------------------------
def _make_preprocessor() -> ColumnTransformer:
    """
    Khởi tạo bộ biến đổi đặc trưng `ColumnTransformer` mới cho từng mô hình.

    Đảm bảo các biến phân loại được mã hóa One-Hot và các biến số được chuẩn hóa
    StandardScaler độc lập bên trong Pipeline để chống rò rỉ dữ liệu (data leakage).

    Tham số (Args):
        Không có.

    Trả về (Returns):
        ColumnTransformer: Bộ tiền xử lý chuẩn hóa đặc trưng đầu vào.
    """
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def _downsample_curve(x: np.ndarray, y: np.ndarray, n_points: int = 80) -> Dict[str, List[float]]:
    """
    Giảm mật độ điểm (down-sample) trên đường cong ROC hoặc Precision-Recall xuống `n_points`
    điểm cách đều nhau để tối ưu dung lượng tệp JSON và tăng tốc độ vẽ đồ thị Plotly.

    Tham số (Args):
        x (np.ndarray): Mảng tọa độ trục hoành (ví dụ: FPR hoặc Recall).
        y (np.ndarray): Mảng tọa độ trục tung (ví dụ: TPR hoặc Precision).
        n_points (int): Số lượng điểm tối đa cần giữ lại.

    Trả về (Returns):
        Dict[str, List[float]]: Từ điển chứa hai danh sách số thực `x` và `y` đã làm tròn 5 chữ số.
    """
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
    Huấn luyện và đối sánh 3 mô hình (Logistic Regression, Random Forest, XGBoost)
    trên cùng một tập dữ liệu chia tách phân tầng Stratified 80/20 (`random_state=42`).

    Tham số (Args):
        df (pd.DataFrame): Bảng dữ liệu khách hàng chứa các cột đặc trưng `MODEL_FEATURES` và nhãn `TARGET`.

    Trả về (Returns):
        Dict[str, Any]: Từ điển chứa thông tin giao thức chia tập dữ liệu, bảng so sánh chỉ số
            đánh giá của 3 mô hình và tọa độ các đường cong ROC / PR.
    """
    X, y = df[MODEL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # Tính tỷ lệ mất cân bằng lớp trên tập huấn luyện để truyền vào XGBoost
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
# 2. Tối ưu hóa Ngưỡng Lợi nhuận (Threshold Profit Curve Optimization)
# ---------------------------------------------------------------------------
def run_threshold_profit_curve(
    df_360: pd.DataFrame,
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Quét ngưỡng phân loại rủi ro cao (`high_risk_threshold`) từ 0.15 đến 0.85 (bước nhảy 0.05)
    và tính toán các chỉ số tài chính toàn danh mục tại từng ngưỡng cắt xác suất.

    Bản chất kinh tế của bài toán tối ưu ngưỡng cắt xác suất (`tau_H`):
        - Nếu hạ `tau_H` quá thấp: Tập khách hàng nhận ưu đãi phình to, giúp bắt được nhiều
          người sắp rời mạng hơn nhưng đồng thời làm tăng mạnh lượng Dương tính giả (False Positives).
          Các khách hàng trung thành tự nhiên (`1 - p_i`) sẵn sàng nhận mã giảm giá, gây ra
          **chi phí ăn mòn doanh thu tự nhiên (Discount Cannibalization Cost)** và làm quá tải
          đội ngũ tổng đài viên VIP (Overload Surcharge).
        - Nếu nâng `tau_H` quá cao: Chỉ can thiệp vào nhóm có nguy cơ cực đoan, giúp tỷ lệ
          phần trăm ROI cao nhưng bỏ sót lượng lớn khách hàng rủi ro trung bình-cao, làm giảm
          tổng lợi ích ròng tuyệt đối.
        - Sự đánh đổi phi tuyến này tạo ra một **Điểm Cực đại Nội tại (Optimal Financial Threshold)**
          giúp tối đa hóa Lợi ích Ròng Kỳ vọng (Expected Net Benefit).

    Tham số (Args):
        df_360 (pd.DataFrame): Bảng dữ liệu 7.043 khách hàng từ `vw_customer_360`.
        params (Optional[Dict[str, float]]): Từ điển ghi đè các tham số kinh tế cơ sở.

    Trả về (Returns):
        Dict[str, Any]: Từ điển chứa điểm tối ưu `optimal_threshold`, điểm cơ sở `baseline_threshold`
            (tại 0.60) và danh sách chi tiết toàn bộ 15 ngưỡng quét `threshold_sweep`.
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

        # Tính tỷ lệ chiết khấu áp dụng cho từng phân khúc để ước lượng hao hụt do Dương tính giả
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
        # Khách hàng vốn trung thành (1 - p_i) sẽ tận dụng ngay ưu đãi giảm giá (cannibalization)
        # cộng thêm phụ phí quá tải vận hành khi tổng số cuộc gọi VIP vượt ngưỡng năng lực 750 cuộc
        cannibalization_cost = (1.0 - probs) * (monthly * disc_rate * disc_months) * 1.35
        vip_count = int(sim_df["decision_quadrant"].isin(VIP_CALL_QUADRANTS).sum())
        overload_surcharge = max(0, vip_count - 750) * 18.0

        raw_cost = float(sim_df["intervention_cost"].sum())
        effective_cost = round(raw_cost + float(cannibalization_cost.sum()) + overload_surcharge, 2)

        # Hiệu suất cứu doanh thu suy giảm biên khi chiến dịch dàn trải sang nhóm rủi ro thấp
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
# 3. Phân bổ Nguồn lực Ràng buộc Kép (Knapsack / Greedy Marginal ROI Optimizer)
# ---------------------------------------------------------------------------
def optimize_under_constraints(
    sim_df: pd.DataFrame,
    max_budget: float = 10000.0,
    max_vip_calls: int = 150,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Giải bài toán phân bổ ngân sách giữ chân khách hàng dưới 2 ràng buộc vận hành thực tế:
        1. Trần ngân sách tài chính: `sum(intervention_cost) <= max_budget`
        2. Giới hạn năng lực gọi điện 1-1 của Account Manager: `count(VIP calls) <= max_vip_calls`

    Thuật toán Tham lam theo Tỷ suất Sinh lời Biên (Greedy Knapsack Heuristic):
        - Bước 1: Lọc toàn bộ các tài khoản đủ điều kiện nhận can thiệp (`intervention_cost > 0`
          và `expected_net_benefit > 0`).
        - Bước 2: Tính chỉ số hiệu quả biên trên mỗi đô-la chi phí:
          `efficiency_ratio_i = expected_net_benefit_i / intervention_cost_i`
        - Bước 3: Sắp xếp ứng viên theo thứ tự giảm dần của `efficiency_ratio_i` (nếu bằng nhau
          thì ưu tiên `expected_net_benefit_i` lớn hơn).
        - Bước 4: Duyệt tuần tự qua danh sách đã xếp hạng; cấp vốn (`Funded (Optimal Priority)`)
          cho khách hàng nếu và chỉ nếu việc thêm khách hàng đó không vượt quá `max_budget`
          và (đối với khách hàng thuộc nhóm VIP Call) không vượt quá `max_vip_calls`.
        - Bước 5: Xây dựng **Đường biên Hiệu quả Pareto (Pareto Efficient Frontier)** biểu diễn
          mối quan hệ phi tuyến giữa `% Ngân sách Đã chi` và `% Lợi ích Ròng Thu được`.

    Tham số (Args):
        sim_df (pd.DataFrame): Bảng mô phỏng quyết định từ `run_decision_simulation`.
        max_budget (float): Ngân sách chiến dịch tối đa cho phép (USD).
        max_vip_calls (int): Số lượng cuộc gọi 1-1 tối đa mà đội ngũ AM có thể thực hiện.

    Trả về (Returns):
        Tuple[pd.DataFrame, Dict[str, Any]]: Bộ gồm DataFrame đã cập nhật trạng thái phân bổ
            (`allocation_status`) và từ điển tổng hợp kết quả tối ưu hóa kèm tọa độ đường biên Pareto.
    """
    df = sim_df.copy()
    eligible_mask = (df["intervention_cost"] > 0) & (df["expected_net_benefit"] > 0)
    candidates = df[eligible_mask].copy()
    candidates["is_vip_call"] = candidates["decision_quadrant"].isin(VIP_CALL_QUADRANTS)

    # Xếp hạng ứng viên theo tỷ suất lợi ích ròng trên một đơn vị chi phí (Marginal ROI Efficiency)
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

    # Duyệt tham lam (Greedy Knapsack) qua danh sách ứng viên đã sắp xếp theo hiệu quả biên
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

    # Đưa chi phí và lợi ích kỳ vọng của các tài khoản bị trì hoãn (Deferred) về 0 trong kỳ ngân sách này
    df.loc[deferred_mask, "decision_quadrant"] = "Deferred (Resource Constraint)"
    df.loc[deferred_mask, "recommended_action"] = (
        "Deferred due to budget/VIP call capacity constraint — queued for next cycle ($0 spend)"
    )
    df.loc[
        deferred_mask,
        ["intervention_cost", "expected_saved_revenue", "expected_net_benefit", "expected_roi"],
    ] = 0.0

    # Dựng Đường biên Hiệu quả Pareto (Pareto Efficient Frontier) trên tập ứng viên đã xếp hạng tối ưu
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
# Điểm Khởi chạy (Entry Point)
# ---------------------------------------------------------------------------
def run_engine() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Thực thi toàn bộ 3 mô-đun tối ưu hóa nâng cao (Step 7) và lưu các tệp JSON phục vụ Dashboard.

    Tham số (Args):
        Không có.

    Trả về (Returns):
        Tuple[Dict[str, Any], Dict[str, Any]]: Bộ gồm dữ liệu đối sánh đa mô hình (`benchmark_data`)
            và dữ liệu đường cong lợi nhuận kèm kịch bản Knapsack tham chiếu (`combined_profit_artifact`).
    """
    df_360 = load_customer_360()

    # 1. Đối sánh 3 mô hình học máy (Logistic Regression, Random Forest, XGBoost)
    benchmark_data = run_model_benchmark(df_360)

    # 2. Quét ngưỡng xác suất để dựng Đường cong Lợi nhuận (Threshold Profit Curve)
    profit_curve_data = run_threshold_profit_curve(df_360)

    # 3. Chạy bài toán Knapsack dưới kịch bản tham chiếu ($10,000 ngân sách, 150 cuộc gọi VIP)
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
    """
    Hàm main điều phối thực thi script từ dòng lệnh và bắt lỗi an toàn.

    Tham số (Args):
        Không có.

    Trả về (Returns):
        int: 0 nếu thành công, 1 nếu phát sinh ngoại lệ.
    """
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
