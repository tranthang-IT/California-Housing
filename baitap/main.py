"""
Điểm thực thi chính (Main Entry Point) của dự án California Housing
Sử dụng 100% các thuật toán và pipeline tự xây dựng từ đầu (From Scratch).
"""

import os
import sys

# Cấu hình UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đưa thư mục gốc của project vào sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import numpy as np
import pandas as pd

from src.data.loader import load_data
from src.data.split import stratified_train_test_split
from src.features.feature_engineering import CombinedAttributesAdderScratch
from src.algorithms.scalers import SimpleImputerScratch, StandardScalerScratch, OneHotEncoderScratch
from src.algorithms.linear_regression import LinearRegressionScratch
from src.algorithms.decision_tree import DecisionTreeRegressorScratch
from src.algorithms.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from src.pipeline.pipeline import CustomPipeline, CustomColumnTransformer


def find_dataset():
    candidates = [
        os.path.join(BASE_DIR, "..", "datasets", "housing.csv"),
        os.path.join(BASE_DIR, "datasets", "housing.csv"),
        os.path.join(BASE_DIR, "housing.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError("Không tìm thấy file housing.csv tại các thư mục lân cận.")


def build_scratch_preprocessor(X_sample):
    num_attr = list(X_sample.select_dtypes(include="number").columns)
    cat_attr = list(X_sample.select_dtypes(include=["object", "str", "category"]).columns)

    rooms_ix = num_attr.index("total_rooms")
    bedrooms_ix = num_attr.index("total_bedrooms")
    population_ix = num_attr.index("population")
    households_ix = num_attr.index("households")

    num_pipeline = CustomPipeline([
        ("imputer", SimpleImputerScratch(strategy="median")),
        ("attribs_adder", CombinedAttributesAdderScratch(
            add_bedrooms_per_room=True,
            rooms_ix=rooms_ix,
            bedrooms_ix=bedrooms_ix,
            population_ix=population_ix,
            households_ix=households_ix
        )),
        ("std_scaler", StandardScalerScratch())
    ])

    cat_pipeline = CustomPipeline([
        ("imputer", SimpleImputerScratch(strategy="most_frequent")),
        ("onehot", OneHotEncoderScratch(handle_unknown="ignore"))
    ])

    full_preprocessor = CustomColumnTransformer([
        ("num", num_pipeline, num_attr),
        ("cat", cat_pipeline, cat_attr)
    ])

    return full_preprocessor


def run_pipeline():
    print("=" * 70)
    print("  CALIFORNIA HOUSING PREDICTION - FROM SCRATCH MACHINE LEARNING")
    print("=" * 70)

    # 1. Đọc dữ liệu
    dataset_path = find_dataset()
    print(f"\n[1/5] Đang tải dữ liệu từ: {dataset_path}")
    data = load_data(dataset_path)
    print(f"      -> Tổng số dòng: {len(data):,} | Số cột: {len(data.columns)}")

    # 2. Phân tầng Train / Test
    print("\n[2/5] Đang phân chia tập Train / Test (Stratified Sampling theo thu nhập)...")
    train_set, test_set = stratified_train_test_split(data, test_size=0.2, random_state=42)
    print(f"      -> Tập Train: {len(train_set):,} mẫu | Tập Test: {len(test_set):,} mẫu")

    X_train = train_set.drop("median_house_value", axis=1)
    y_train = train_set["median_house_value"].to_numpy(dtype=float)

    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].to_numpy(dtype=float)

    # 3. Tiền xử lý dữ liệu
    print("\n[3/5] Khởi tạo và áp dụng Pipeline tiền xử lý hoàn toàn tự viết...")
    preprocessor = build_scratch_preprocessor(X_train)
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)
    print(f"      -> Kích thước ma trận sau tiền xử lý: {X_train_prepared.shape}")

    # 4. Huấn luyện các mô hình From Scratch
    print("\n[4/5] Huấn luyện các mô hình tự cài đặt bằng NumPy thuần:")

    # Mô hình A: Linear Regression (Normal Equation)
    print("      - Đang huấn luyện: Linear Regression (Normal Equation / Nghiệm chính xác)...")
    lr_normal = LinearRegressionScratch(method="normal_equation")
    lr_normal.fit(X_train_prepared, y_train)
    y_pred_lr = lr_normal.predict(X_test_prepared)
    rmse_lr = root_mean_squared_error(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    print(f"        + Linear Regression (Normal Eq) -> RMSE: {rmse_lr:,.0f} USD | MAE: {mae_lr:,.0f} USD | R2: {r2_lr:.4f}")

    # Mô hình B: Decision Tree Regressor (Variance Reduction)
    print("      - Đang huấn luyện: Decision Tree Regressor (Độ giảm phương sai, max_depth=7)...")
    tree = DecisionTreeRegressorScratch(max_depth=7, min_samples_split=20, min_samples_leaf=10, n_thresholds=25)
    tree.fit(X_train_prepared, y_train)
    y_pred_tree = tree.predict(X_test_prepared)
    rmse_tree = root_mean_squared_error(y_test, y_pred_tree)
    mae_tree = mean_absolute_error(y_test, y_pred_tree)
    r2_tree = r2_score(y_test, y_pred_tree)
    print(f"        + Decision Tree Regressor       -> RMSE: {rmse_tree:,.0f} USD | MAE: {mae_tree:,.0f} USD | R2: {r2_tree:.4f}")

    # 5. Đóng gói mô hình tốt nhất thành Pipeline End-to-End
    best_model = tree if rmse_tree < rmse_lr else lr_normal
    best_name = "Decision Tree Regressor" if best_model == tree else "Linear Regression"
    print(f"\n[5/5] Mô hình chiến thắng: {best_name}")

    full_prediction_pipeline = CustomPipeline([
        ("preprocessor", preprocessor),
        ("model", best_model)
    ])

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "california_scratch_model.pkl")

    joblib.dump(full_prediction_pipeline, model_save_path)
    print(f"      -> Đã xuất xưởng toàn bộ Pipeline End-to-End tại: {model_save_path}")

    # Kiểm tra dự đoán trực tiếp từ Pipeline vừa đóng gói
    sample_raw = X_test.iloc[:5]
    sample_preds = full_prediction_pipeline.predict(sample_raw)
    print("\n--- Thử nghiệm dự đoán 5 mẫu đầu tiên từ tập Test (Dữ liệu thô) ---")
    for idx, (real, pred) in enumerate(zip(y_test[:5], sample_preds)):
        print(f"  Mẫu {idx + 1}: Thực tế = {real:,.0f} USD | AI dự đoán = {pred:,.0f} USD (Chênh lệch: {abs(real - pred):,.0f} USD)")

    print("\n-> Hoàn thành toàn bộ quy trình From Scratch thành công!")


if __name__ == "__main__":
    run_pipeline()
