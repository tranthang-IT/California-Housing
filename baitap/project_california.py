"""
California Housing Price Prediction - End-to-End Machine Learning
Dự án dự đoán giá nhà tại bang California dựa trên các thông số khu dân cư.
Quy trình:
  1. Load & Kiểm tra dữ liệu (EDA)
  2. Phân tầng và chia tập Train / Test (Stratified Sampling)
  3. Tiền xử lý dữ liệu và tạo đặc trưng mới (Feature Engineering & Pipeline)
  4. Huấn luyện, đánh giá chéo (Cross Validation) và Tinh chỉnh siêu tham số (GridSearchCV)
  5. Đánh giá trên tập Test và đóng gói Pipeline hoàn chỉnh để xuất xưởng (Export .pkl)
"""

import os
import sys

# Cấu hình UTF-8 cho console Windows để in tiếng Việt không bị lỗi font (UnicodeEncodeError)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đăng ký alias module vào sys.modules: Khi chạy trực tiếp file như một script (__main__),
# việc này giúp joblib/pickle lưu trữ đúng tên module 'project_california' thay vì '__main__',
# đảm bảo file .pkl có thể được import và load ở bất kỳ file/ứng dụng nào khác (Streamlit, Flask, v.v.).
if __name__ == "__main__":
    sys.modules["project_california"] = sys.modules["__main__"]

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


def load_data(path_csv):
    """
    Đọc dữ liệu từ file CSV.
    Sử dụng try-except để bắt lỗi và đưa thông báo rõ ràng nếu sai đường dẫn.
    """
    try:
        data = pd.read_csv(path_csv)
        return data
    except Exception as e:
        print(f"Lỗi khi đọc file dữ liệu tại '{path_csv}': {e}")
        raise e


def display_data(data):
    """
    Trực quan hóa và khám phá dữ liệu (EDA - Exploratory Data Analysis):
    - data.info(): Kiểm tra kiểu dữ liệu và số lượng giá trị thiếu (Missing values).
    - data.describe(): Tóm tắt các thống kê mô tả (mean, std, min, max, phân vị).
    - Scatter plot bản đồ địa lý kết hợp màu sắc giá nhà và kích thước theo mật độ dân số.
    - Ma trận tương quan (Correlation Matrix) đối với thuộc tính mục tiêu 'median_house_value'.
    """
    print("--- Thông tin cấu trúc dữ liệu ---")
    print(data.info())
    print("\n--- Thống kê mô tả các thuộc tính số ---")
    print(data.describe())

    # Vẽ biểu đồ phân bố địa lý
    plt.figure(figsize=(10, 7))
    x = data["longitude"].values
    y = data["latitude"].values
    scatter_plt = plt.scatter(
        x, y,
        s=data["population"] / 100,  # Kích thước chấm đại diện cho mật độ dân số
        c=data["median_house_value"],  # Màu sắc đại diện cho giá nhà
        cmap="jet",
        alpha=0.4
    )
    plt.colorbar(scatter_plt, label="Giá nhà trung bình (USD)")
    plt.gca().set_facecolor('black')
    plt.xlabel("Kinh độ (Longitude)")
    plt.ylabel("Vĩ độ (Latitude)")
    plt.title("Bản đồ phân bố giá nhà và mật độ dân cư California")
    plt.tight_layout()
    plt.show()

    # Phân tích độ tương quan tuyến tính (Pearson correlation) với giá nhà
    corr_matrix = data.corr(numeric_only=True).sort_values(by="median_house_value", ascending=False)
    print("\n--- Ma trận tương quan với giá nhà (median_house_value) ---")
    print(corr_matrix["median_house_value"])


def split_data(data):
    """
    Chia tập Train / Test theo phương pháp Phân tầng (Stratified Sampling):
    
    TẠI SAO PHẢI PHÂN TẦNG (STRATIFIED SPLIT)?
    - Phân tích tương quan cho thấy 'median_income' (thu nhập trung bình) là biến dự đoán quan trọng nhất.
    - Nếu chia ngẫu nhiên (pure random sampling), tập test có thể bị thiếu hụt các nhóm thu nhập cực cao
      hoặc cực thấp, gây ra sai lệch (sampling bias).
    - Phân tầng giúp tập Test phản ánh đúng tỷ lệ phân bố thu nhập thực tế của toàn bộ dân cư.
    """
    # Tạo bản sao để tránh làm thay đổi trực tiếp (mutate) DataFrame ban đầu
    data_copy = data.copy()

    # Chia 'median_income' thành 5 nhóm: 1: (0-1.5), 2: (1.5-3.0), 3: (3.0-4.5), 4: (4.5-6.0), 5: (>6.0)
    data_copy["income_category"] = pd.cut(
        data_copy["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    # Sử dụng train_test_split với tham số stratify để giữ nguyên tỷ lệ nhóm income_category
    train_set, test_set = train_test_split(
        data_copy,
        test_size=0.2,
        stratify=data_copy["income_category"],
        random_state=42
    )

    # Loại bỏ thuộc tính tạm thời 'income_category' để khôi phục cấu trúc ban đầu của dữ liệu
    for set_ in (train_set, test_set):
        set_.drop("income_category", axis=1, inplace=True)

    return train_set, test_set


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Custom Transformer theo chuẩn Scikit-Learn để tạo thêm các đặc trưng kết hợp (Feature Engineering):
    
    TẠI SAO CẦN TẠO THÊM ĐẶC TRƯNG NÀY?
    - Tổng số phòng (total_rooms) của cả khu phố không hữu ích bằng số phòng TRÊN MỖI HỘ (rooms_per_household).
    - Tương tự, tổng dân số (population) không hữu ích bằng số người TRÊN MỖI HỘ (population_per_household).
    - Tỷ lệ phòng ngủ trên tổng số phòng (bedrooms_per_room) phản ánh tính chất căn hộ (sang trọng hay bình dân).

    """
    __module__ = "project_california"

    def __init__(self, add_bedrooms_per_room=True, rooms_ix=3, bedrooms_ix=4, population_ix=5, households_ix=6):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = rooms_ix
        self.bedrooms_ix = bedrooms_ix
        self.population_ix = population_ix
        self.households_ix = households_ix

    def fit(self, X, y=None):
        # Transformer này chỉ biến đổi dữ liệu số học thuần túy, không cần học tham số từ dữ liệu,
        # nên hàm fit chỉ cần trả về chính nó (return self) để tương thích Pipeline.
        return self

    def transform(self, X):
        # Tính toán các đặc trưng tỷ lệ mới từ mảng NumPy
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            # Ghép các cột mới vào bên phải của mảng dữ liệu X
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def preprocessing_data(train_set):
    """
    Xây dựng Pipeline tiền xử lý dữ liệu chuẩn hóa và tự động:
    
    QUY TRÌNH:
    1. Tách nhãn dự đoán (median_house_value) khỏi các thuộc tính đầu vào (X_train).
    2. Tách riêng các thuộc tính dạng số (num_attr) và dạng phân loại chuỗi (category_attr).
    3. Pipeline cho các cột số (num_pipeline):
       - SimpleImputer(strategy='median'): Điền giá trị bị khuyết thiếu (NaN) bằng giá trị trung vị
         (dùng median tốt hơn mean vì dữ liệu giá nhà hay bị lệch đuôi / outliers).
       - CombinedAttributesAdder: Tự động thêm các đặc trưng tỷ lệ vừa tạo ở trên.
       - StandardScaler: Đưa các biến về cùng thang đo chuẩn hóa (mean = 0, std = 1),
         giúp các thuật toán Gradient/Linear hội tụ tốt và không bị biến có giá trị lớn lấn át.
    4. Pipeline cho các cột phân loại (category_pipeline):
       - SimpleImputer(strategy='most_frequent'): Điền thiếu bằng giá trị xuất hiện nhiều nhất.
       - OneHotEncoder: Chuyển dữ liệu chữ ('NEAR OCEAN', 'INLAND'...) thành dạng số nhị phân (One-Hot).
    5. ColumnTransformer: Hợp nhất 2 pipeline chạy song song và tự động ráp các cột lại với nhau.
    """
    X_train = train_set.drop("median_house_value", axis=1)
    y_train = train_set["median_house_value"].copy()

    # Tự động nhận diện danh sách cột số và cột phân loại
    num_attr = list(X_train.select_dtypes(include="number"))
    category_attr = list(X_train.select_dtypes(include=["object", "str", "category"]))

    # Xác định chỉ số (index) cột số động dựa vào vị trí trong danh sách num_attr
    rooms_ix = num_attr.index("total_rooms")
    bedrooms_ix = num_attr.index("total_bedrooms")
    population_ix = num_attr.index("population")
    households_ix = num_attr.index("households")

    # Pipeline xử lý dữ liệu số
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder(
            add_bedrooms_per_room=True,
            rooms_ix=rooms_ix,
            bedrooms_ix=bedrooms_ix,
            population_ix=population_ix,
            households_ix=households_ix
        )),
        ("std_scaler", StandardScaler())
    ])

    # Pipeline xử lý dữ liệu phân loại
    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Hợp nhất 2 nhánh xử lý
    full_pipeline = ColumnTransformer([
        ("num_pipeline", num_pipeline, num_attr),
        ("category_pipeline", category_pipeline, category_attr)
    ])

    # Fit (học các tham số như median, mean, std) và transform trên tập huấn luyện
    X_train_prepared = full_pipeline.fit_transform(X_train)
    return X_train_prepared, y_train, full_pipeline


def train_and_tune_model(X_train_prepared, y_train):
    """
    Đánh giá sơ bộ các mô hình (Model Selection) và Tinh chỉnh siêu tham số (Hyperparameter Tuning):
    
    1. K-Fold Cross-Validation (cv=10):
       - Chia tập train làm 10 phần, dùng 9 phần để train và 1 phần kiểm thử (lặp lại 10 lần).
       - Tránh hiện tượng học vẹt (overfitting) và cho cái nhìn khách quan về khả năng tổng quát hóa.
       - Scikit-Learn dùng 'neg_mean_squared_error' (điểm càng lớn càng tốt, nên giá trị âm).
         Ta đổi dấu (-) rồi lấy căn bậc hai (np.sqrt) để ra RMSE đơn vị USD thực tế.
    2. GridSearchCV:
       - Tự động chạy thử tất cả các tổ hợp siêu tham số (n_estimators, max_features)
         trên mô hình tốt nhất (Random Forest) để tìm ra phiên bản tối ưu nhất.
    """
    linear_model = LinearRegression()
    tree_reg = DecisionTreeRegressor(random_state=42)
    forest_reg = RandomForestRegressor(random_state=42)

    list_model = [linear_model, tree_reg, forest_reg]
    print("\n--- Đánh giá sơ bộ các mô hình qua 10-Fold Cross-Validation ---")
    for model in list_model:
        scores = cross_val_score(
            model, X_train_prepared, y_train,
            cv=10, scoring="neg_mean_squared_error", n_jobs=-1
        )
        rmse_scores = np.sqrt(-scores)
        print(f"Mô hình: {model.__class__.__name__:<25} | Sai số RMSE trung bình: {rmse_scores.mean():,.0f} USD (std: {rmse_scores.std():,.0f})")

    # Không gian tìm kiếm siêu tham số cho Random Forest
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]}
    ]

    print("\n--- Đang tinh chỉnh siêu tham số (GridSearchCV cho Random Forest)... ---")
    grid_search = GridSearchCV(
        estimator=forest_reg,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
        return_train_score=True
    )
    grid_search.fit(X_train_prepared, y_train)

    print(f"Cấu hình tham số tối ưu nhất: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_and_save_model(best_model, test_set, full_pipeline, output_path="california_housing_model.pkl"):
    """
    Đánh giá mô hình trên tập Test và Xuất file đóng gói hoàn chỉnh:
    
    ĐIỂM CẢI TIẾN QUAN TRỌNG NHẤT:
    - Ở bản cũ, ta chỉ lưu 'best_model'. Điều đó khiến khi deploy lên Web/App, mô hình không thể
      dự đoán được vì thiếu 'full_pipeline' để làm sạch và scale dữ liệu đầu vào.
    - Ở bản này, ta đóng gói cả 'full_pipeline' và 'best_model' thành một 'full_prediction_pipeline' duy nhất.
    - Nhờ vậy, khi đem file .pkl đi deploy, hàm predict có thể nhận trực tiếp DataFrame thô
      từ người dùng nhập vào mà không cần viết lại code tiền xử lý!
    """
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()

    # Đóng gói toàn bộ Pipeline tiền xử lý và Mô hình vào một Pipeline End-to-End duy nhất
    full_prediction_pipeline = Pipeline([
        ("preparation", full_pipeline),
        ("model", best_model)
    ])

    # Dự đoán trực tiếp trên tập X_test (dữ liệu thô chưa biến đổi)
    predictions = full_prediction_pipeline.predict(X_test)

    # Đánh giá sai số RMSE cuối cùng trên tập Test
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print("\n--- Đánh giá mô hình trên tập Test nguyên sơ ---")
    print(f"Sai số RMSE cuối cùng: {rmse:,.0f} USD")

    # Lưu toàn bộ Pipeline End-to-End ra file .pkl
    joblib.dump(full_prediction_pipeline, output_path)
    print(f"-> Đã đóng gói thành công trọn bộ (Pipeline + Model) vào file: '{output_path}'")


if __name__ == "__main__":
    # Đường dẫn tương đối đến file dữ liệu housing.csv
    dataset_path = r"../datasets/housing.csv"
    data = load_data(dataset_path)

    # Khám phá dữ liệu (Bỏ comment dòng dưới nếu bạn muốn xem biểu đồ bản đồ trực quan)
    # display_data(data)

    print("Bước 1: Đang chia tập Train / Test bằng phương pháp Stratified Sampling...")
    train_set, test_set = split_data(data)

    print("Bước 2: Đang xây dựng Pipeline và tiền xử lý dữ liệu...")
    X_train_prepared, y_train, full_pipeline = preprocessing_data(train_set)

    print("Bước 3: Đang huấn luyện và tinh chỉnh mô hình tốt nhất (Vui lòng đợi vài giây)...")
    best_model = train_and_tune_model(X_train_prepared, y_train)

    print("Bước 4: Đang chấm điểm trên tập Test và đóng gói xuất xưởng mô hình...")
    evaluate_and_save_model(best_model, test_set, full_pipeline)