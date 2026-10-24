# 📈 PROGRESS LOG — ISM Customer Retention & Value Optimization DSS

> Nhật ký tiến độ dự án. Được cập nhật sau **mỗi bước** (Code & Verify → Update Docs → Git Commit & Push → Report).

## 🗺️ Roadmap tổng quan

| Step | Nội dung | Trạng thái |
|------|----------|-----------|
| 1 | Project Scaffolding & Environment Setup | [x] Hoàn thành |
| 2 | Data Acquisition — `src/data_generator.py` (Telco Customer Churn, 7,043 dòng) | [ ] |
| 3 | ETL Pipeline & SQLite DB — `src/etl_pipeline.py` → `database/ism_dss.db` (3 bảng chuẩn hóa) | [ ] |
| 4 | Data Mining Engine — `src/mining_engine.py` (K-Means Value Tier + XGBoost Churn + SHAP) | [ ] |
| 5 | Decision Engine — `src/decision_engine.py` (Ma trận quyết định, ROI, Revenue at Risk) | [ ] |
| 6 | Streamlit DSS Dashboard — `app.py` (4-Tier ISM Presentation Layer) & Final Docs | [ ] |

---

## ✅ Step 1 — Project Scaffolding & Environment Setup

- **Trạng thái:** [x] Hoàn thành
- **Thời gian:** 2026-10-24 08:31 (UTC+7)
- **Mục tiêu:** Dựng khung thư mục trực tiếp tại root `ISM-project/`, khai báo thư viện, thiết lập môi trường ảo và tài liệu theo dõi tiến độ.

### Files / thư mục đã tạo hoặc chỉnh sửa
| Đường dẫn | Mô tả |
|-----------|-------|
| `data/raw/.gitkeep` | Thư mục chứa dữ liệu gốc |
| `data/processed/.gitkeep` | Thư mục chứa dữ liệu sau ETL |
| `database/.gitkeep` | Thư mục chứa SQLite DB `ism_dss.db` |
| `models/.gitkeep` | Thư mục lưu model artifacts (`.joblib`) |
| `src/__init__.py` | Biến `src/` thành Python package |
| `.gitignore` | Bỏ qua `__pycache__/`, `.venv/`, IDE files, logs, secrets |
| `requirements.txt` | numpy, pandas, requests, scikit-learn, xgboost, shap, joblib, matplotlib, plotly, streamlit |
| `PROGRESS.md` | File nhật ký tiến độ (file này) |
| `README.md` | Cập nhật tổng quan dự án, kiến trúc 4-Tier, cấu trúc thư mục, hướng dẫn cài đặt |

### Kết quả kiểm chứng
- Python: **3.9.2**
- Tạo môi trường ảo `.venv/` và cài đặt thành công toàn bộ `requirements.txt`.
- Import test thành công: numpy 1.26.4 · pandas 2.3.3 · scikit-learn 1.5.2 · xgboost 2.1.4 · shap 0.46.0 · plotly 7.1.0 · streamlit 1.50.0

---
