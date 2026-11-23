# California Housing Price Prediction (From Scratch & End-to-End Machine Learning)

Dự án Học máy (Machine Learning) toàn diện dự đoán **Giá nhà trung bình** tại California, Mỹ. Dự án hỗ trợ cả hai phương pháp tiếp cận:
1. **Kiến trúc Tự cài đặt thuật toán từ đầu (From Scratch with NumPy):** Tự tay hiện thực hóa các thuật toán tối ưu hóa toán học, tiền xử lý và cấu trúc Pipeline mà không phụ thuộc vào các class mô hình có sẵn của Scikit-Learn.
2. **Kiến trúc Thư viện chuẩn (Scikit-Learn Standard):** Dùng để đối soát, kiểm thử đối đầu (Benchmarking) và xác thực tính đúng đắn về mặt giải tích.

---

## 1. Cấu trúc Dự án Sau khi Tái cấu trúc (Modular Architecture)

Mã nguồn được tổ chức theo chuẩn kiến trúc sạch (Clean ML Architecture), tách biệt rõ ràng giữa Dữ liệu, Thuật toán, Kỹ thuật đặc trưng và Pipeline thực thi:

```text
california_housing/
├── src/                                # Gói mã nguồn cốt lõi (Core Package)
│   ├── algorithms/                     # Thuật toán và độ đo tự viết từ đầu (From Scratch)
│   │   ├── __init__.py
│   │   ├── linear_regression.py        # Hồi quy tuyến tính (Normal Equation & Gradient Descent)
│   │   ├── decision_tree.py            # Cây quyết định hồi quy (Variance Reduction)
│   │   ├── scalers.py                  # StandardScalerScratch, SimpleImputerScratch, OneHotEncoderScratch
│   │   └── metrics.py                  # RMSE, MAE, R2-score, MSE
│   │
│   ├── data/                           # Module quản lý dữ liệu
│   │   ├── __init__.py
│   │   ├── loader.py                   # Đọc dữ liệu, kiểm tra tệp
│   │   └── split.py                    # Phân tầng Stratified Sampling theo thu nhập
│   │
│   ├── features/                       # Kỹ thuật tạo đặc trưng (Feature Engineering)
│   │   ├── __init__.py
│   │   └── feature_engineering.py     # CombinedAttributesAdderScratch (rooms/hh, pop/hh...)
│   │
│   └── pipeline/                       # Pipeline tự viết (Custom Pipeline Engine)
│       ├── __init__.py
│       └── pipeline.py                 # CustomPipeline & CustomColumnTransformer
│
├── tests/
│   └── test_algorithms.py              # Unit Tests kiểm tra độ khớp thuật toán vs Scikit-Learn
│
├── models/
│   └── california_scratch_model.pkl    # Checkpoint Pipeline End-to-End tự xuất xưởng
│
├── main.py                             # Điểm thực thi chính (Chạy toàn bộ quy trình From Scratch)
├── benchmark.py                        # Script đo lường và so sánh hiệu năng (Scratch vs Sklearn)
├── project_california.py               # Script nguyên bản (sử dụng Scikit-Learn để đối chiếu)
├── GHI_CHU_KIEN_TRUC_PIPELINE.md       # Tài liệu giải thích chi tiết kiến trúc Pipeline
├── requirements.txt                    # Danh sách thư viện phụ thuộc
└── README.md
```

---

## 2. Bản chất Toán học của các Thuật toán Tự viết (From Scratch)

### 2.1. Hồi quy Tuyến tính (Linear Regression)
* **Phương trình chuẩn giải tích (Normal Equation):**
  $$\theta = (X_b^T X_b)^{-1} X_b^T y$$
  Sử dụng giả nghịch đảo Moore-Penrose (`np.linalg.pinv`) nhằm đảm bảo tính ổn định số học và chống suy biến ma trận khi có đa cộng tuyến.
* **Hạ độ dốc theo mẻ (Batch Gradient Descent):**
  $$\theta^{(t+1)} = \theta^{(t)} - \alpha \frac{2}{m} X_b^T (X_b \theta^{(t)} - y)$$

### 2.2. Cây Quyết định Hồi quy (Decision Tree Regressor)
* **Tiêu chí phân nhánh:** Tối đa hóa mức giảm phương sai (Variance Reduction):
  $$\Delta \text{Var} = \text{Var}(y) - \left( \frac{N_L}{N} \text{Var}(y_L) + \frac{N_R}{N} \text{Var}(y_R) \right)$$
* **Tối ưu hóa hiệu năng:** Áp dụng phương pháp phân vị quantile sampling cho các ngưỡng ứng viên giúp tăng tốc độ huấn luyện trên 16.512 mẫu chỉ mất vài giây trên CPU.

### 2.3. Tiền xử lý & Chuẩn hóa (Scalers & Imputer)
* `StandardScalerScratch`: Chuẩn hóa $z = \frac{x - \mu}{\sigma}$ kèm bảo vệ chống chia cho 0.
* `SimpleImputerScratch`: Điền khuyết thiếu bằng trung vị (`median`), trung bình (`mean`) hoặc `most_frequent`.
* `OneHotEncoderScratch`: Chuyển danh mục thành các vector nhị phân 0/1.

---

## 3. Kết quả Kiểm thử & So sánh Hiệu năng (Benchmarking)

Khi chạy script kiểm thử đối đầu `python benchmark.py` trên cùng tập dữ liệu Train/Test:

| Mô hình | Thời gian Scratch | Thời gian Sklearn | RMSE Scratch | RMSE Sklearn | Độ lệch RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **5.34 ms** | 13.18 ms | **66,913 USD** | 66,913 USD | **0.00 USD** *(Khớp tuyệt đối 100%)* |
| **Decision Tree (depth=7)** | 4.15 s | 479.10 ms | **61,029 USD** | 61,430 USD | **400.49 USD** *(Độ chính xác tương đương)* |

> **Nhận xét:** 
> - Hồi quy tuyến tính tự viết cho kết quả dự đoán trùng khớp tuyệt đối tới từng chữ số thập phân so với thư viện Scikit-Learn.
> - Cây quyết định tự viết đạt sai số RMSE ~61k USD, hội tụ tương đương với thuật toán Scikit-Learn.

---

## 4. Hướng dẫn Khởi chạy

### 4.1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 4.2. Chạy Unit Test kiểm tra tính chính xác thuật toán
```bash
python tests/test_algorithms.py
```
*(Tất cả 4/4 bài test kiểm tra Scaler, Imputer, Metrics và Normal Equation đều đạt chuẩn `OK`)*.

### 4.3. Chạy toàn bộ quy trình From Scratch & Đóng gói mô hình
```bash
python main.py
```
Quy trình sẽ tự động:
1. Đọc và phân tầng dữ liệu Train / Test (80% / 20%).
2. Chạy tiền xử lý qua `CustomPipeline` và `CustomColumnTransformer`.
3. Huấn luyện Linear Regression và Decision Tree thuần NumPy.
4. Chọn mô hình chiến thắng và xuất xưởng tệp `models/california_scratch_model.pkl`.
5. Thực hiện dự đoán mẫu trên dữ liệu thô chưa qua biến đổi.

### 4.4. Chạy bảng so sánh hiệu năng (Benchmark)
```bash
python benchmark.py
```
