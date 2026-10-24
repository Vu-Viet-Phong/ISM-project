"""
Module Khai phá Dữ liệu & Học máy Giải thích được (XAI) — `src/mining_engine.py`.

Vai trò trong Kiến trúc 4 Tầng (4-Tier ISM Architecture):
    Thuộc Tầng 3 (Tier 3 — Core Data Mining & Explainable AI Layer).
    Module này hiện thực hóa 3 mô-đun phân tích cốt lõi của hệ thống DSS và lưu trữ
    kết quả chấm điểm vào bảng sự kiện `fact_mining_scores` trong `database/ism_dss.db`:
        - Module 1 — Phân hạng Giá trị Khách hàng (Unsupervised Learning — K-Means):
          Chuẩn hóa `StandardScaler` kết hợp phân cụm `K-Means (k = 3)` trên 4 chiều kinh tế:
          `tenure`, `monthly_charges`, `total_charges` và `total_services_count`. Các cụm được
          xếp hạng tự động theo giá trị kinh tế trung bình và ánh xạ sang 3 phân hạng
          `'High'`, `'Medium'`, và `'Low'`.
        - Module 2 — Dự báo Rời bỏ Dịch vụ (Supervised Learning — XGBoost):
          Đóng gói `ColumnTransformer` (`OneHotEncoder` + `StandardScaler`) và bộ phân loại
          `XGBClassifier` trong một `Pipeline` duy nhất, huấn luyện trên phép chia phân tầng
          Stratified 80/20 với trọng số `scale_pos_weight` để xử lý mất cân bằng lớp.
          Sau khi đánh giá độc lập trên tập kiểm thử (Test set), pipeline được huấn luyện lại
          trên toàn bộ 7.043 khách hàng để dự báo xác suất rời bỏ (`churn_probability`) và
          gán nhãn mức độ rủi ro (`'High' >= 0.60 > 'Medium' >= 0.30 > 'Low'`).
        - Module 3 — Trí tuệ Nhân tạo Giải thích được (Explainable AI — TreeSHAP):
          Tính toán giá trị đóng góp TreeSHAP chính xác (đơn vị log-odds) cho từng khách hàng.
          Cộng gộp các cột mã hóa One-Hot về lại thuộc tính nghiệp vụ gốc, trích xuất 2 nhân tố
          đẩy rủi ro rời bỏ tăng mạnh nhất của mỗi cá nhân (ví dụ: `"contract=Month-to-month (+0.82)"`).

Nhóm tác giả (Group 6 — International University, VNU-HCM):
    1. Vũ Viết Phong — ITDSIU19048 (Leader)
    2. Trương Đăng Khoa — ITDSIU19027
    3. Trần Khánh Toàn — ITITIU20322
    4. Nguyễn Lê Mai Hương — ITITWE23040

Các hàm chính:
    - train_value_tiering: Phân cụm K-Means và kiểm định hệ số Silhouette từ k=2..6.
    - train_churn_model: Huấn luyện, kiểm thử hold-out 80/20 và refit mô hình XGBoost.
    - explain_churn / compute_shap_values: Tính giá trị TreeSHAP và trích xuất top 2 nhân tố rủi ro.
    - persist_scores / save_artifacts: Lưu điểm số vào SQLite và đóng gói mô hình `.joblib`.
    - score_customer: Chấm điểm và giải thích SHAP thời gian thực cho một khách hàng mới (Tab 4).
"""

import json
import sqlite3
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Cấu hình Đường dẫn & Siêu tham số Mô hình (Configuration & Hyperparameters)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_FILE = PROJECT_ROOT / "database" / "ism_dss.db"
MODELS_DIR = PROJECT_ROOT / "models"
KMEANS_FILE = MODELS_DIR / "kmeans_value_pipeline.joblib"
XGB_FILE = MODELS_DIR / "xgb_churn_pipeline.joblib"
SHAP_FILE = MODELS_DIR / "shap_metadata.joblib"
METRICS_FILE = MODELS_DIR / "model_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CLUSTERS = 3
SILHOUETTE_K_RANGE = range(2, 7)

# Đặc trưng đầu vào của Module 1 — Phân cụm giá trị kinh tế khách hàng (K-Means)
VALUE_FEATURES = ["tenure", "monthly_charges", "total_charges", "total_services_count"]
VALUE_TIER_LABELS = ["High", "Medium", "Low"]  # Sắp xếp từ giá trị cao nhất xuống thấp nhất

# Đặc trưng đầu vào của Module 2 — Dự báo xác suất rời bỏ dịch vụ (XGBoost)
TARGET = "churn_binary"
CATEGORICAL_FEATURES = [
    "gender", "tenure_group", "contract", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "payment_method",
]
NUMERIC_FEATURES = [
    "senior_citizen", "partner", "dependents", "phone_service", "paperless_billing",
    "is_auto_payment", "tenure", "total_services_count",
    "monthly_charges", "total_charges", "avg_monthly_spend",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Ngưỡng phân tầng mức độ rủi ro (khớp với ràng buộc CHECK trong bảng fact_mining_scores)
HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.30
CLASSIFICATION_THRESHOLD = 0.50

XGB_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
    eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
)

MONEY_FEATURES = {"monthly_charges", "total_charges", "avg_monthly_spend"}


# ---------------------------------------------------------------------------
# Tầng Truy cập Dữ liệu (Data Access)
# ---------------------------------------------------------------------------
def load_customer_master(db_path: Path = DB_FILE) -> pd.DataFrame:
    """
    Tải toàn bộ bản ghi khách hàng đã làm sạch và chuẩn hóa từ view `customer_master`.

    Tham số (Args):
        db_path (Path): Đường dẫn tới tệp cơ sở dữ liệu SQLite (`database/ism_dss.db`).

    Trả về (Returns):
        pd.DataFrame: Bảng dữ liệu 7.043 khách hàng với đầy đủ các trường nhân khẩu học, dịch vụ và cước phí.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM customer_master ORDER BY customerID", conn)
    print(f"[DATA]   Loaded {len(df):,} customers x {df.shape[1]} columns from customer_master")
    return df


# ---------------------------------------------------------------------------
# Module 1 — Phân hạng Giá trị Khách hàng bằng K-Means (Customer Value Tiering)
# ---------------------------------------------------------------------------
def train_value_tiering(df: pd.DataFrame):
    """
    Phân khúc khách hàng thành 3 tầng giá trị kinh tế (`High`, `Medium`, `Low`) bằng thuật toán K-Means.

    Tính toán hệ số Silhouette cho `k = 2..6` để kiểm định chất lượng phân cụm; lựa chọn
    `k = 3` vì tương thích trực tiếp với cấu trúc phân tầng quản trị 3 cấp của Ma trận Quyết định 9 Ô.
    Các cụm được sắp xếp giảm dần theo `total_charges` trung bình (giá trị lũy kế thực tế),
    nếu bằng nhau sẽ xét đến `monthly_charges` trung bình (doanh thu định kỳ hàng tháng).

    Tham số (Args):
        df (pd.DataFrame): Bảng dữ liệu khách hàng từ `customer_master`.

    Trả về (Returns):
        Tuple: Bộ 6 phần tử gồm `(pipeline đã huấn luyện, từ điển ánh xạ cluster->tier,
            mảng cluster_ids, mảng nhãn value_tiers, từ điển điểm silhouette, DataFrame hồ sơ cụm)`.
    """
    X = df[VALUE_FEATURES].astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    # Quét k = 2..6 để ghi nhận chỉ số kiểm định Silhouette
    silhouettes = {}
    for k in SILHOUETTE_K_RANGE:
        labels_k = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_scaled)
        silhouettes[k] = float(silhouette_score(X_scaled, labels_k, random_state=RANDOM_STATE))

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)),
    ])
    cluster_ids = pipeline.fit_predict(X)

    # Tổng hợp hồ sơ kinh tế của từng cụm và xếp hạng từ Cao xuống Thấp
    profile = (df.assign(cluster_id=cluster_ids)
                 .groupby("cluster_id")
                 .agg(customers=("customerID", "count"),
                      tenure=("tenure", "mean"),
                      monthly_charges=("monthly_charges", "mean"),
                      total_charges=("total_charges", "mean"),
                      total_services_count=("total_services_count", "mean"),
                      churn_rate=("churn_binary", "mean")))
    ranked = profile.sort_values(["total_charges", "monthly_charges"], ascending=False).index
    tier_map = {int(cid): VALUE_TIER_LABELS[rank] for rank, cid in enumerate(ranked)}
    profile["value_tier"] = profile.index.map(tier_map)
    profile["revenue_share"] = (df.assign(cluster_id=cluster_ids)
                                  .groupby("cluster_id")["monthly_charges"].sum()
                                / df["monthly_charges"].sum())
    profile = profile.loc[ranked]

    value_tiers = pd.Series(cluster_ids).map(tier_map).values
    print(f"[K-MEANS] Silhouette scores: " +
          ", ".join(f"k={k}: {s:.3f}" for k, s in silhouettes.items()))
    print(f"[K-MEANS] Trained k={N_CLUSTERS}; cluster -> tier mapping: {tier_map}")
    return pipeline, tier_map, cluster_ids, value_tiers, silhouettes, profile


# ---------------------------------------------------------------------------
# Module 2 — Dự báo Rời bỏ Dịch vụ bằng XGBoost (Churn Prediction)
# ---------------------------------------------------------------------------
def build_churn_pipeline(scale_pos_weight: float) -> Pipeline:
    """
    Xây dựng Pipeline dự báo rời bỏ khép kín từ bước tiền xử lý đến mô hình XGBoost.

    Việc đóng gói `ColumnTransformer` bên trong `Pipeline` đảm bảo các bộ mã hóa One-Hot
    và chuẩn hóa StandardScaler chỉ được khớp (fit) trên tập huấn luyện, ngăn chặn triệt để
    hiện tượng rò rỉ dữ liệu (data leakage) và cho phép chấm điểm trực tiếp bản ghi thô trên Dashboard.

    Tham số (Args):
        scale_pos_weight (float): Hệ số cân bằng trọng số lớp dương (tỷ lệ mẫu âm / mẫu dương).

    Trả về (Returns):
        Pipeline: Đối tượng scikit-learn Pipeline hoàn chỉnh.
    """
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def assign_risk_level(prob: np.ndarray) -> np.ndarray:
    """
    Ánh xạ mảng xác suất rời bỏ sang 3 nhãn mức độ rủi ro `'High'`, `'Medium'`, và `'Low'`.

    Tham số (Args):
        prob (np.ndarray): Mảng xác suất rời bỏ dự báo từ mô hình XGBoost (miền [0.0, 1.0]).

    Trả về (Returns):
        np.ndarray: Mảng chuỗi nhãn mức độ rủi ro tương ứng.
    """
    return np.select([prob >= HIGH_RISK_THRESHOLD, prob >= MEDIUM_RISK_THRESHOLD],
                     ["High", "Medium"], default="Low")


def train_churn_model(df: pd.DataFrame):
    """
    Huấn luyện, đánh giá trên tập kiểm thử độc lập (Hold-out 80/20) và khớp lại (refit)
    pipeline XGBoost trên toàn bộ danh mục khách hàng.

    Tham số (Args):
        df (pd.DataFrame): Bảng dữ liệu khách hàng từ `customer_master`.

    Trả về (Returns):
        Tuple[Pipeline, dict]: Bộ gồm `(pipeline triển khai đã refit trên 7.043 mẫu, từ điển chỉ số đánh giá)`.
    """
    X, y = df[MODEL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # Huấn luyện trên 80% dữ liệu và đo lường khách quan trên 20% tập kiểm thử chưa từng thấy
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    eval_pipeline = build_churn_pipeline(spw).fit(X_train, y_train)

    prob_test = eval_pipeline.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= CLASSIFICATION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
    prob_train = eval_pipeline.predict_proba(X_train)[:, 1]

    metrics = {
        "train_size": int(len(X_train)), "test_size": int(len(X_test)),
        "train_churn_rate": float(y_train.mean()), "test_churn_rate": float(y_test.mean()),
        "scale_pos_weight": round(spw, 4),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "roc_auc": float(roc_auc_score(y_test, prob_test)),
        "pr_auc": float(average_precision_score(y_test, prob_test)),
        "precision": float(precision_score(y_test, pred_test)),
        "recall": float(recall_score(y_test, pred_test)),
        "f1": float(f1_score(y_test, pred_test)),
        "accuracy": float((pred_test == y_test).mean()),
        "brier_score": float(brier_score_loss(y_test, prob_test)),
        "train_roc_auc": float(roc_auc_score(y_train, prob_train)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "xgb_params": {k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
    }
    print(f"[XGBOOST] Test ROC-AUC={metrics['roc_auc']:.4f} | PR-AUC={metrics['pr_auc']:.4f} | "
          f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f}")
    print(f"[XGBOOST] Confusion matrix (test): TN={tn} FP={fp} FN={fn} TP={tp}")

    # Khớp lại mô hình sản xuất (production refit) trên toàn bộ 7.043 khách hàng
    spw_full = float((y == 0).sum() / (y == 1).sum())
    final_pipeline = build_churn_pipeline(spw_full).fit(X, y)
    print(f"[XGBOOST] Deployed pipeline refitted on all {len(X):,} customers (scale_pos_weight={spw_full:.3f})")
    return final_pipeline, metrics


# ---------------------------------------------------------------------------
# Module 3 — Trí tuệ Nhân tạo Giải thích được (Explainable AI — TreeSHAP)
# ---------------------------------------------------------------------------
def encoded_to_original_mapping(pipeline: Pipeline) -> list:
    """
    Tạo danh sách ánh xạ từng cột đặc trưng sau mã hóa One-Hot về lại tên thuộc tính nghiệp vụ gốc.

    Tham số (Args):
        pipeline (Pipeline): Pipeline XGBoost đã được huấn luyện.

    Trả về (Returns):
        list: Danh sách tên thuộc tính gốc tương ứng với từng cột đầu ra của `ColumnTransformer`.
    """
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    mapping = []
    for col, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
        mapping.extend([col] * len(cats))
    mapping.extend(NUMERIC_FEATURES)
    return mapping


def compute_shap_values(pipeline: Pipeline, X: pd.DataFrame):
    """
    Tính toán ma trận giá trị đóng góp TreeSHAP chính xác (đơn vị log-odds) cho mọi khách hàng.

    Ưu tiên sử dụng `shap.TreeExplainer`; tự động chuyển sang cơ chế tính TreeSHAP nội tại
    của XGBoost (`pred_contribs=True`, hoàn toàn tương đương về mặt toán học) nếu xảy ra
    khác biệt phiên bản thư viện.

    Tham số (Args):
        pipeline (Pipeline): Pipeline XGBoost đã huấn luyện.
        X (pd.DataFrame): Bảng dữ liệu đặc trưng đầu vào (`MODEL_FEATURES`).

    Trả về (Returns):
        Tuple[np.ndarray, float, str]: Bộ gồm `(ma trận SHAP [n_customers x n_encoded_features],
            giá trị kỳ vọng nền expected_value, tên phương pháp tính)`.
    """
    X_enc = pipeline.named_steps["preprocessor"].transform(X)
    model = pipeline.named_steps["model"]
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(X_enc))
        base = float(np.ravel(explainer.expected_value)[0])
        method = f"shap.TreeExplainer (shap {shap.__version__})"
    except Exception as exc:  # pragma: no cover - cơ chế dự phòng tương thích phiên bản
        import xgboost as xgb
        print(f"[SHAP]    TreeExplainer unavailable ({type(exc).__name__}); using XGBoost native TreeSHAP")
        contribs = model.get_booster().predict(xgb.DMatrix(X_enc), pred_contribs=True)
        values, base = contribs[:, :-1], float(contribs[0, -1])
        method = "xgboost native TreeSHAP (pred_contribs)"
    return values, base, method


def format_value(feature: str, value) -> str:
    """
    Định dạng giá trị thuộc tính gốc của khách hàng thành chuỗi dễ đọc cho nhà quản trị.

    Tham số (Args):
        feature (str): Tên thuộc tính gốc.
        value (Any): Giá trị thô của thuộc tính.

    Trả về (Returns):
        str: Chuỗi biểu diễn giá trị đã làm tròn hoặc định dạng tiền tệ.
    """
    if feature in MONEY_FEATURES:
        return f"{float(value):.2f}"
    if isinstance(value, (int, np.integer, float, np.floating)):
        return str(int(value)) if float(value).is_integer() else f"{float(value):.2f}"
    return str(value)


def extract_top_drivers(shap_by_feature: pd.DataFrame, X: pd.DataFrame, top_n: int = 2) -> list:
    """
    Trích xuất `top_n` thuộc tính có đóng góp SHAP dương lớn nhất (làm tăng nguy cơ rời bỏ)
    đối với từng khách hàng.

    Tham số (Args):
        shap_by_feature (pd.DataFrame): Bảng giá trị SHAP đã cộng gộp theo 23 thuộc tính gốc.
        X (pd.DataFrame): Bảng giá trị thuộc tính gốc của khách hàng.
        top_n (int): Số lượng nhân tố rủi ro hàng đầu cần trích xuất (mặc định là 2).

    Trả về (Returns):
        list: Danh sách gồm `top_n` mảng chứa chuỗi định dạng nhân tố rủi ro (ví dụ: `'contract=Month-to-month (+0.82)'`).
    """
    values = shap_by_feature.values
    features = shap_by_feature.columns.to_numpy()
    order = np.argsort(-values, axis=1)[:, :top_n]
    drivers = [[None] * len(X) for _ in range(top_n)]
    X_records = X.reset_index(drop=True)
    for i in range(len(X)):
        for rank in range(top_n):
            j = order[i, rank]
            contribution = values[i, j]
            if contribution > 0:
                feat = features[j]
                drivers[rank][i] = (f"{feat}={format_value(feat, X_records.at[i, feat])} "
                                    f"(+{contribution:.2f})")
    return drivers


def explain_churn(pipeline: Pipeline, X: pd.DataFrame):
    """
    Thực thi Module 3: Tính giá trị TreeSHAP, cộng gộp theo thuộc tính nghiệp vụ gốc,
    xác định tầm quan trọng đặc trưng toàn cục (Global Importance) và trích xuất nhân tố cục bộ.

    Tham số (Args):
        pipeline (Pipeline): Pipeline XGBoost đã huấn luyện.
        X (pd.DataFrame): Bảng dữ liệu đặc trưng của khách hàng cần giải thích.

    Trả về (Returns):
        Tuple: Bộ 5 phần tử `(danh sách driver_1, danh sách driver_2, DataFrame SHAP theo thuộc tính,
            Series tầm quan trọng toàn cục, từ điển metadata SHAP)`.
    """
    values, base, method = compute_shap_values(pipeline, X)
    mapping = encoded_to_original_mapping(pipeline)
    encoded_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())

    # Cộng gộp giá trị SHAP của các cột One-Hot cùng thuộc một thuộc tính gốc (tính chất cộng tính của SHAP)
    shap_by_feature = (pd.DataFrame(values, columns=encoded_names)
                         .T.groupby(np.array(mapping)).sum().T)[MODEL_FEATURES]
    global_importance = shap_by_feature.abs().mean().sort_values(ascending=False)
    driver_1, driver_2 = extract_top_drivers(shap_by_feature, X, top_n=2)

    metadata = {
        "method": method,
        "units": "log-odds",
        "expected_value": base,
        "encoded_feature_names": encoded_names,
        "encoded_to_original": mapping,
        "model_features": MODEL_FEATURES,
        "global_importance": global_importance.to_dict(),
        "shap_by_feature": shap_by_feature.astype(np.float32),
        "customer_ids": None,  # Sẽ được gán danh sách customerID bởi hàm gọi
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"[SHAP]    Computed {values.shape[0]:,} x {values.shape[1]} SHAP values via {method}")
    return driver_1, driver_2, shap_by_feature, global_importance, metadata


# ---------------------------------------------------------------------------
# Lưu trữ Điểm số & Đóng gói Mô hình (Persistence & Model Serialization)
# ---------------------------------------------------------------------------
def persist_scores(scores: pd.DataFrame, db_path: Path = DB_FILE) -> None:
    """
    Ghi kết quả chấm điểm khai phá dữ liệu vào bảng `fact_mining_scores` một cách nguyên tử (atomic).

    Xóa dữ liệu cũ và chèn hàng loạt (bulk insert) 7.043 bản ghi mới ngay trong cùng một
    giao dịch SQLite để đảm bảo tính lũy đẳng (idempotent) khi chạy lại pipeline.

    Tham số (Args):
        scores (pd.DataFrame): Bảng kết quả chấm điểm của 7.043 khách hàng.
        db_path (Path): Đường dẫn tới tệp cơ sở dữ liệu SQLite.

    Trả về (Returns):
        None
    """
    cols = ["customerID", "value_tier", "cluster_id", "churn_probability", "risk_level",
            "top_shap_driver_1", "top_shap_driver_2", "scored_at"]
    rows = [tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in r)
            for r in scores[cols].itertuples(index=False, name=None)]
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        with conn:  # Giao dịch đơn (single transaction): tự động commit nếu thành công, rollback nếu lỗi
            conn.execute("DELETE FROM fact_mining_scores;")
            conn.executemany(f"INSERT INTO fact_mining_scores ({', '.join(cols)}) "
                             f"VALUES ({', '.join('?' * len(cols))})", rows)
    finally:
        conn.close()
    print(f"[PERSIST] Wrote {len(rows):,} rows to fact_mining_scores")


def save_artifacts(kmeans_pipeline, tier_map, xgb_pipeline, shap_meta, metrics) -> None:
    """
    Tuần tự hóa (serialize) các pipeline mô hình, dữ liệu SHAP và chỉ số đánh giá vào thư mục `models/`.

    Tham số (Args):
        kmeans_pipeline (Pipeline): Pipeline phân cụm K-Means đã huấn luyện.
        tier_map (dict): Từ điển ánh xạ mã cụm sang nhãn phân hạng giá trị ('High', 'Medium', 'Low').
        xgb_pipeline (Pipeline): Pipeline dự báo rời bỏ XGBoost đã huấn luyện.
        shap_meta (dict): Từ điển chứa ma trận giá trị SHAP và thông tin giải thích.
        metrics (dict): Từ điển chứa toàn bộ chỉ số kiểm định mô hình.

    Trả về (Returns):
        None
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": kmeans_pipeline, "features": VALUE_FEATURES, "tier_map": tier_map},
                KMEANS_FILE)
    joblib.dump({"pipeline": xgb_pipeline, "features": MODEL_FEATURES,
                 "categorical_features": CATEGORICAL_FEATURES, "numeric_features": NUMERIC_FEATURES,
                 "risk_thresholds": {"high": HIGH_RISK_THRESHOLD, "medium": MEDIUM_RISK_THRESHOLD}},
                XGB_FILE)
    joblib.dump(shap_meta, SHAP_FILE)
    METRICS_FILE.write_text(json.dumps(
        metrics, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"[ARTIFACT] Saved {KMEANS_FILE.name}, {XGB_FILE.name}, {SHAP_FILE.name}, {METRICS_FILE.name}")


# ---------------------------------------------------------------------------
# Hàm Suy luận Thời gian thực (Inference Helpers dùng cho Tab 4 trên Dashboard)
# ---------------------------------------------------------------------------
def load_artifacts() -> dict:
    """
    Tải toàn bộ các tệp mô hình đã đóng gói (`kmeans`, `xgb`, `shap`) từ thư mục `models/`.

    Tham số (Args):
        Không có.

    Trả về (Returns):
        dict: Từ điển chứa 3 đối tượng artifact `'kmeans'`, `'xgb'`, và `'shap'`.
    """
    return {"kmeans": joblib.load(KMEANS_FILE), "xgb": joblib.load(XGB_FILE),
            "shap": joblib.load(SHAP_FILE)}


def score_customer(record: dict, artifacts: dict = None) -> dict:
    """
    Chấm điểm và giải thích SHAP thời gian thực cho một hồ sơ khách hàng đơn lẻ.

    Tham số (Args):
        record (dict): Từ điển chứa các thuộc tính thô của khách hàng (tương tự cấu trúc `customer_master`).
        artifacts (dict, optional): Bộ mô hình đã tải sẵn trong bộ nhớ đệm để tăng tốc độ phản hồi.

    Trả về (Returns):
        dict: Từ điển chứa `cluster_id`, `value_tier`, `churn_probability`, `risk_level`,
            `top_shap_driver_1`, và `top_shap_driver_2`.
    """
    artifacts = artifacts or load_artifacts()
    X = pd.DataFrame([record])
    km, xgb_art = artifacts["kmeans"], artifacts["xgb"]
    cluster = int(km["pipeline"].predict(X[km["features"]].astype(float))[0])
    prob = float(xgb_art["pipeline"].predict_proba(X[xgb_art["features"]])[0, 1])
    d1, d2, *_ = explain_churn(xgb_art["pipeline"], X[xgb_art["features"]])
    return {"cluster_id": cluster, "value_tier": km["tier_map"][cluster],
            "churn_probability": prob, "risk_level": str(assign_risk_level(np.array([prob]))[0]),
            "top_shap_driver_1": d1[0], "top_shap_driver_2": d2[0]}


# ---------------------------------------------------------------------------
# Báo cáo Thực nghiệm & Kiểm định Kết nối (Reporting & Verification)
# ---------------------------------------------------------------------------
def print_summary(profile, silhouettes, metrics, global_importance, scores, db_path=DB_FILE):
    """
    In báo cáo kết quả thực nghiệm của cả 3 mô-đun và kiểm tra kết nối qua view `vw_customer_360`.

    Tham số (Args):
        profile (pd.DataFrame): Bảng thống kê đặc trưng kinh tế của 3 cụm K-Means.
        silhouettes (dict): Từ điển hệ số Silhouette ứng với k = 2..6.
        metrics (dict): Từ điển chỉ số đánh giá mô hình XGBoost trên tập kiểm thử.
        global_importance (pd.Series): Chuỗi tầm quan trọng SHAP toàn cục của các thuộc tính.
        scores (pd.DataFrame): Bảng điểm số của toàn bộ 7.043 khách hàng.
        db_path (Path): Đường dẫn tệp cơ sở dữ liệu SQLite.

    Trả về (Returns):
        None
    """
    pd.set_option("display.width", 140)
    print("\n" + "=" * 72)
    print(" MINING ENGINE SUMMARY")
    print("=" * 72)
    print(f" Silhouette (k=2..6): {', '.join(f'{k}:{s:.3f}' for k, s in silhouettes.items())}")
    print("\n [Module 1] K-Means value tiers (k=3):")
    print(profile.round(3).to_string())
    print("\n [Module 2] XGBoost hold-out metrics:")
    for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "brier_score", "train_roc_auc"]:
        print(f"   {key:<14}: {metrics[key]:.4f}")
    print(f"   confusion     : {metrics['confusion_matrix']}")
    print("\n [Module 3] Top 5 global SHAP features (mean |SHAP|, log-odds):")
    for feat, val in global_importance.head(5).items():
        print(f"   {feat:<22}: {val:.4f}")
    print("\n Risk level distribution:")
    print(scores["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).to_string())
    print("\n Cross-tabulation value_tier x risk_level:")
    print(pd.crosstab(scores["value_tier"], scores["risk_level"], margins=True)
            .reindex(index=["High", "Medium", "Low", "All"], columns=["High", "Medium", "Low", "All"])
            .to_string())

    with sqlite3.connect(db_path) as conn:
        check = conn.execute(
            "SELECT COUNT(*), SUM(value_tier IS NOT NULL), SUM(churn_probability IS NOT NULL), "
            "SUM(top_shap_driver_1 IS NOT NULL) FROM vw_customer_360").fetchone()
        sample = pd.read_sql(
            "SELECT customerID, contract, tenure, monthly_charges, value_tier, churn_probability, "
            "risk_level, top_shap_driver_1, top_shap_driver_2 FROM vw_customer_360 "
            "ORDER BY churn_probability DESC LIMIT 5", conn)
    print(f"\n vw_customer_360 join check: rows={check[0]:,}, value_tier={check[1]:,}, "
          f"churn_probability={check[2]:,}, top_shap_driver_1={check[3]:,}")
    print(" Top 5 highest-risk customers (from vw_customer_360):")
    print(sample.to_string(index=False))
    print("=" * 72)


# ---------------------------------------------------------------------------
# Điểm Khởi chạy (Entry Point)
# ---------------------------------------------------------------------------
def run_engine() -> pd.DataFrame:
    """
    Điều phối thực thi tuần tự Module 1 (K-Means), Module 2 (XGBoost), Module 3 (TreeSHAP),
    lưu trữ điểm số vào SQLite và xuất các tệp mô hình ra thư mục `models/`.

    Tham số (Args):
        Không có.

    Trả về (Returns):
        pd.DataFrame: Bảng điểm số khai phá dữ liệu của 7.043 khách hàng.
    """
    df = load_customer_master()

    kmeans_pipeline, tier_map, cluster_ids, value_tiers, silhouettes, profile = train_value_tiering(df)
    xgb_pipeline, metrics = train_churn_model(df)

    X_all = df[MODEL_FEATURES]
    churn_prob = xgb_pipeline.predict_proba(X_all)[:, 1]
    driver_1, driver_2, shap_by_feature, global_importance, shap_meta = explain_churn(xgb_pipeline, X_all)
    shap_meta["customer_ids"] = df["customerID"].tolist()

    scores = pd.DataFrame({
        "customerID": df["customerID"],
        "value_tier": value_tiers,
        "cluster_id": cluster_ids.astype(int),
        "churn_probability": np.round(churn_prob.astype(float), 6),
        "risk_level": assign_risk_level(churn_prob),
        "top_shap_driver_1": driver_1,
        "top_shap_driver_2": driver_2,
        "scored_at": datetime.now().isoformat(timespec="seconds"),
    })

    metrics.update({
        "silhouette_scores": silhouettes,
        "value_tier_profile": profile.reset_index().to_dict(orient="records"),
        "global_shap_importance": {k: float(v) for k, v in global_importance.items()},
        "risk_distribution": scores["risk_level"].value_counts().to_dict(),
        "tier_risk_crosstab": pd.crosstab(scores["value_tier"], scores["risk_level"]).to_dict(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    })

    save_artifacts(kmeans_pipeline, tier_map, xgb_pipeline, shap_meta, metrics)
    persist_scores(scores)
    print_summary(profile, silhouettes, metrics, global_importance, scores)
    return scores


def main() -> int:
    """
    Hàm main xử lý ngoại lệ và trả về mã thoát cho hệ điều hành.

    Tham số (Args):
        Không có.

    Trả về (Returns):
        int: 0 nếu chạy thành công, 1 nếu gặp lỗi.
    """
    try:
        run_engine()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Mining engine failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
