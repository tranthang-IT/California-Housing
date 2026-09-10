# Sổ Tay Kiến Trúc: ColumnTransformer, Pipeline & Model trong Machine Learning

Tài liệu này đúc kết toàn bộ bản chất, cơ chế hoạt động ngầm và những điểm mấu chốt dễ gây nhầm lẫn nhất về luồng xử lý dữ liệu giữa **`ColumnTransformer`**, **`Pipeline`** và **`Model`** (Estimator) trong thư viện Scikit-Learn.

---

## 1. Bản chất của 3 Đối Tượng Cốt Lõi

Trong Scikit-Learn, các đối tượng được chia thành các vai trò rất rõ ràng:

| Đối tượng | Nhóm (Role) | Các hàm cốt lõi | Nhiệm vụ chính |
| :--- | :--- | :--- | :--- |
| **`SimpleImputer`**, **`StandardScaler`**, **`OneHotEncoder`** | **Transformer** *(Bộ biến đổi đơn lẻ)* | `fit()`<br>`transform()`<br>`fit_transform()` | Học thống kê của 1 nhóm cột và biến đổi dữ liệu. **Không có `predict()`**. |
| **`full_pipeline`**<br>*(ColumnTransformer)* | **Composite Transformer** *(Bộ biến đổi đa luồng)* | `fit()`<br>`transform()`<br>`fit_transform()` | Chia luồng cột số / cột chữ, chạy các pipeline con song song rồi ghép ngang lại. **Không có `predict()`**. |
| **`best_model`**<br>*(RandomForest, LinearReg)* | **Estimator / Predictor** *(Mô hình dự đoán)* | `fit()`<br>`predict()` | Học quy luật ánh xạ từ đặc trưng $X$ sang nhãn $y$. **Không có `transform()`**. |
| **`full_prediction_pipeline`**<br>*(Pipeline trọn gói)* | **Coordinator** *(Tổng chỉ huy điều phối)* | `fit()`<br>`predict()` | Nối Transformer ở đầu và Model ở cuối thành một khối duy nhất. |

---

## 2. Bản chất của 4 Phương Thức: `fit`, `transform`, `fit_transform`, `predict`

### A. `fit(X)` — CHỈ LÀ HỌC, TUYỆT ĐỐI KHÔNG BIẾN ĐỔI DỮ LIỆU
* **Hành động:** Chỉ quan sát dữ liệu $X$ để tính toán và lưu các tham số thống kê vào bộ nhớ (thuộc tính có đuôi gạch dưới `_`).
  * `SimpleImputer.fit()`: Học giá trị trung vị (`statistics_ = median`).
  * `StandardScaler.fit()`: Học giá trị trung bình (`mean_`) và độ lệch chuẩn (`scale_`).
  * `OneHotEncoder.fit()`: Quét và ghi nhớ các nhóm chữ (`categories_`).
  * `CombinedAttributesAdder.fit()`: Không cần học gì, chỉ `return self`.
* **Kết quả trả về:** Trả về chính đối tượng đó (`self`) đã được nạp tri thức. **Không hề sinh ra cột mới hay điền vào ô trống nào cả!**

### B. `transform(X)` — ÁP DỤNG TRI THỨC ĐỂ BIẾN ĐỔI VÀ SINH CỘT MỚI
* **Hành động:** Dùng các con số đã học ở bước `fit` để thực sự xử lý mảng dữ liệu $X$:
  * `SimpleImputer.transform()`: Cầm `median` đắp vào các ô bị khuyết `NaN`.
  * `CombinedAttributesAdder.transform()`: **Chính tại đây mới chia các cột và sinh ra 3 cột mới ghép vào!**
  * `StandardScaler.transform()`: Lấy dữ liệu trừ `mean_` chia `scale_`.
  * `OneHotEncoder.transform()`: **Chính tại đây mới đẻ ra các cột nhị phân 0 và 1!**
* **Kết quả trả về:** Mảng dữ liệu mới đã qua biến đổi.

### C. `fit_transform(X)` — PHÍM TẮT TRÊN TẬP TRAIN
* Bản chất là: Chạy `fit(X)` trước để học tham số, sau đó chạy ngay `transform(X)` để trả về dữ liệu đã biến đổi.

### D. `predict(X)` — ĐƯA RA DỰ ĐOÁN
* Chỉ có ở Mô hình (hoặc Pipeline có Mô hình ở bước cuối). Nhận ma trận số sạch sẽ và tính ra nhãn dự đoán (ví dụ: Giá nhà USD).

---

## 3. Cơ Chế Điều Hướng Của `ColumnTransformer`

```python
full_pipeline = ColumnTransformer([
    ("num_pipeline", num_pipeline, num_attr),          # Danh sách tên cột số
    ("category_pipeline", category_pipeline, category_attr) # Danh sách tên cột chữ
])
```

### Câu hỏi: Khi đưa $X_{test}$ vào, làm sao nó biết cột nào số, cột nào chữ mà không cần đoán lại kiểu dữ liệu?
1. **Chốt danh sách từ đầu:** Khi khởi tạo, `num_attr` và `category_attr` là danh sách các chuỗi tên cột cụ thể đã được lưu cứng vào `ColumnTransformer`.
2. **Không tự động đoán lại:** Khi chạy `transform(X_test)`, `ColumnTransformer` **không** dùng `select_dtypes` để đoán lại kiểu. Nó lọc trực tiếp theo tên:
   * Trích xuất `X_test[num_attr]` đẩy vào `num_pipeline`.
   * Trích xuất `X_test[category_attr]` đẩy vào `category_pipeline`.
3. **Ghép nối (Stacking):** Sau khi 2 nhánh chạy xong, nó dùng `np.hstack` ghép mảng kết quả của 2 nhánh lại thành một ma trận 2D thống nhất.
👉 **Ý nghĩa:** Đảm bảo dù dữ liệu mới có bị lỗi định dạng nhẹ (ví dụ một số bị nhầm thành chuỗi) thì cấu trúc các cột đưa vào mô hình vẫn luôn luôn đồng nhất 100% với lúc train.

---

## 4. "Cái Hay" Đỉnh Cao Của `Pipeline` (Design Pattern)

Khi đóng gói toàn bộ quy trình:
```python
full_prediction_pipeline = Pipeline([
    ("preparation", full_pipeline),  # Transformer
    ("model", best_model)            # Estimator
])
```

Pipeline sở hữu cơ chế **chuyển mạch thông minh** giữa 2 chế độ:

```text
1. KHI GỌI: full_prediction_pipeline.fit(X_train, y_train)
   ├── full_pipeline.fit_transform(X_train) ──> Vừa học tham số, vừa biến đổi ra X_train_clean
   └── best_model.fit(X_train_clean, y_train) ──> Học quy luật dự đoán giá nhà

────────────────────────────────────────────────────────────────────────────────────────────

2. KHI GỌI: full_prediction_pipeline.predict(X_test)
   ├── full_pipeline.transform(X_test) ────────> CHỈ dịch và làm sạch (DÙNG LẠI median/std cũ)
   └── best_model.predict(X_test_clean) ───────> Dự đoán ra giá nhà
```

> 🌟 **Điểm mấu chốt:** 
> Khi bạn gọi `.predict()`, Scikit-Learn đã **khóa cứng** hành vi của các bước trung gian: nó **bắt buộc chỉ được gọi `.transform()`**, tuyệt đối không bao giờ gọi `.fit()`.

---

## 5. Chuyện Gì Xảy Ra Nếu Gọi Nhầm `.fit(X_test)`?

Nếu đổi `predictions = full_prediction_pipeline.predict(X_test)` thành `.fit(X_test, y_test)`:

1. **Ghi đè tham số tiền xử lý:** Toàn bộ `median`, `mean`, `std` học từ 16.500 dòng Train sẽ bị xóa sổ và ghi đè bằng giá trị của 4.100 dòng Test.
2. **Xóa sạch Model đã train:** Mô hình Random Forest tối ưu được tìm bởi GridSearch sẽ bị vứt bỏ, mô hình bị train lại từ đầu chỉ trên tập Test.
3. **Gây Crash chương trình:** Hàm `.fit()` chỉ trả về đối tượng `self` (Pipeline), không trả về mảng số giá nhà. Đến dòng tính `mean_squared_error(y_test, predictions)` code sẽ sập ngay lập tức.
4. **Vi phạm Data Leakage (Rò rỉ dữ liệu):** Bạn đã phát đề thi và đáp án cho học sinh học thuộc trước khi nộp bài. Điểm số lúc này là "ảo", khi gặp căn nhà thực tế ngoài đời sẽ dự đoán hoàn toàn sai!

---

## 6. Mối Quan Hệ Giữa Tiền Xử Lý (Pipeline) và Mô Hình (Model)

* **Về mặt toán học:** Chúng **tách biệt**. `median` hay `std` được tính bằng công thức thống kê thuần túy, hoàn toàn không phụ thuộc vào việc bạn dùng thuật toán Linear Regression hay Random Forest.
* **Về mặt vận hành (Inference):** Chúng **gắn liền như hình với bóng**. Mô hình chỉ hiểu được dữ liệu khi dữ liệu đó đi qua đúng bộ lọc đã tạo ra nó (cùng `mean`, cùng `std`, cùng thứ tự cột One-Hot).

### Lợi ích khi đóng gói thành 1 file `.pkl` duy nhất:
Thay vì lưu rời rạc 5 file (`imputer.pkl`, `scaler.pkl`, `encoder.pkl`, `model.pkl`...) và phải viết lại 50 dòng code tiền xử lý thủ công trên Web/App:
```python
# Chỉ cần 2 dòng code khi Deploy sản phẩm (Streamlit / Flask / FastAPI):
model = joblib.load("california_housing_model.pkl")
predicted_price = model.predict(raw_user_dataframe)
```
Toàn bộ khâu kiểm tra missing data, tính feature mới, scale số, One-Hot đều diễn ra tự động và an toàn tuyệt đối.
