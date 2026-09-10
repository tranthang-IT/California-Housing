# California Housing Price Prediction (End-to-End Machine Learning)

Đây là một dự án Học máy (Machine Learning) toàn diện, từ khâu làm sạch dữ liệu thô đến khi xuất xưởng một mô hình Trí tuệ nhân tạo có khả năng dự đoán giá nhà tại bang California, Mỹ.

## Giới thiệu Dự án
Mục tiêu của dự án là xây dựng một mô hình AI đọc các thông tin về một khu dân cư (tọa độ, dân số, thu nhập trung bình, số phòng, vị trí gần biển...) để dự đoán **Giá nhà trung bình** tại khu vực đó.

Dự án áp dụng quy trình chuẩn của một Data Scientist:
1. **Trực quan hóa dữ liệu (EDA):** Vẽ bản đồ phân bố nhiệt độ giá nhà kết hợp mật độ dân cư.
2. **Stratified Sampling:** Chia tập Train/Test chống lệch chuẩn (dựa trên tỷ lệ Thu nhập).
3. **Data Pipelines & Feature Engineering:** Tự động hóa toàn bộ quá trình:
   - Viết Custom Class (`CombinedAttributesAdder`) để tự động tạo ra các đặc trưng quan trọng mới như số phòng/hộ gia đình, số dân/hộ gia đình.
   - Vá dữ liệu thiếu bằng `SimpleImputer`.
   - Chuẩn hóa bằng `StandardScaler` và Biến đổi biến phân loại bằng `OneHotEncoder`.
4. **Model Selection & GridSearch:** Chấm điểm chéo các mô hình và Tinh chỉnh siêu tham số để tìm ra cấu hình AI hoàn hảo nhất.

## Kết quả Tổ chức Giải đấu (Model Selection)
Trước khi tìm ra nhà vô địch, các mô hình đã được cho thi đấu vòng loại bằng phương pháp `cross_val_score` (cv=10). Dưới đây là sai số dự đoán trung bình (RMSE):

- **LinearRegression:** 69,104 USD
- **DecisionTreeRegressor:** 71,630 USD
- **RandomForestRegressor:** 50,436 USD (Nhà vô địch)

Sau khi ép xung Random Forest bằng `GridSearchCV` (cv=5) trên không gian dữ liệu đã được áp dụng Feature Engineering và cho đi thi thật trên tập Test nguyên sơ, AI đạt được:
- **Sai số RMSE Cuối cùng: 47,873 USD**

## Hạn chế và Hướng phát triển tương lai

### Hạn chế hiện tại
- **Giới hạn thuật toán:** Dự án mới chỉ thử nghiệm với Random Forest. Các thuật toán Boosting tiên tiến hơn như XGBoost hay LightGBM chưa được đưa vào thi đấu vòng loại.
- **Không gian tinh chỉnh (Hyperparameter Tuning) hẹp:** Lưới tìm kiếm (GridSearch) mới chỉ được thiết lập với một số lượng nhỏ các tham số (tối đa 30 cây) nhằm tiết kiệm tài nguyên máy tính. Vì không gian tìm kiếm nhỏ nên AI chưa phát huy được tối đa sức mạnh của các cột đặc trưng (Feature Engineering) mới được thêm vào.

### Hướng phát triển tương lai
- **Nâng cấp thuật toán:** Triển khai thử nghiệm thêm Gradient Boosting Regressor hoặc XGBoost kết hợp với mở rộng không gian tìm kiếm `GridSearchCV` để tối ưu hóa và giảm sai số RMSE xuống dưới mức 45,000 USD.
- **Triển khai ứng dụng (Deployment):** Đưa file `california_housing_model.pkl` lên một ứng dụng Web tương tác bằng thư viện Streamlit hoặc Flask, cho phép người dùng tự do nhập thông số khu dân cư và xem AI dự đoán giá nhà ngay trên trình duyệt.

## Cài đặt & Hướng dẫn Chạy Code

### 1. Cấu trúc thư mục
Để code có thể chạy được, bạn cần đảm bảo cấu trúc thư mục sau (thư mục code và thư mục dữ liệu phải đặt cạnh nhau):
```text
/ (Thư mục gốc)
├── datasets/
│   └── housing.csv          # File dữ liệu chứa thông tin giá nhà
└── baitap/
    ├── project_california.py # File chứa toàn bộ mã nguồn AI
    └── README.md
```

### 2. Yêu cầu Thư viện (Requirements)
Cài đặt các thư viện lõi của dự án bằng file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy
Mở Terminal và chạy file Python: 
```bash
cd baitap
python project_california.py
```
**Lưu ý:**
- Code được cấu hình mặc định đọc file dữ liệu bằng đường dẫn tương đối: `load_data(r"../datasets/housing.csv")`. 
- Nếu muốn xem bản đồ nhiệt phân bố giá nhà, bạn có thể bỏ comment dòng `# display_data(data)` trong file script.
- Chạy xong, thư mục sẽ xuất hiện thêm file `california_housing_model.pkl`. File này đã được đóng gói toàn bộ quy trình tiền xử lý (`full_pipeline`) kèm mô hình tối ưu nhất (`best_model`), sẵn sàng nhận trực tiếp DataFrame thô từ người dùng khi tích hợp vào Web/App (Streamlit, Flask, FastAPI).
