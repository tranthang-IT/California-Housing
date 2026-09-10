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

---

## 7. Phân Biệt Sâu: `ColumnTransformer` vs `Pipeline` (Dòng Chảy Ngang vs Dòng Chảy Dọc)

Thoạt nhìn, cả hai đều nhận danh sách các tuple `("tên", ...)` và đều có `fit`, `transform`. Nhưng **hướng xử lý dữ liệu của chúng vuông góc nhau**:

> 🔑 **Nguyên tắc cốt lõi:**
> * **`Pipeline` = DÒNG CHẢY DỌC (Nối tiếp - Tuần tự):** Trạm trước làm xong, lấy toàn bộ kết quả ném sang trạm sau.
> * **`ColumnTransformer` = DÒNG CHẢY NGANG (Phân luồng - Song song):** Bổ dọc bảng dữ liệu thành các nhóm cột, cho đi vào các làn đường độc lập rồi ghép ngang lại.

### A. So sánh trực quan về cấu trúc luồng

```text
1. PIPELINE (Xử lý DỌC - Dây chuyền sản xuất nối tiếp)
   Dữ liệu vào
       │
       ▼
   ┌───────────────────────┐
   │ Bước 1: SimpleImputer │  (Điền NaN cho toàn bộ bảng)
   └───────────┬───────────┘
               │ (Kết quả bước 1 chuyển tiếp xuống bước 2)
               ▼
   ┌───────────────────────┐
   │ Bước 2: StandardScaler│  (Scale toàn bộ bảng)
   └───────────┬───────────┘
               │ (Kết quả bước 2 chuyển tiếp xuống bước 3)
               ▼
   ┌───────────────────────┐
   │ Bước 3: Model (nếu có)│  (Dự đoán kết quả)
   └───────────────────────┘

───────────────────────────────────────────────────────────────────────────────────

2. COLUMNTRANSFORMER (Xử lý NGANG - Cổng phân luồng giao thông)
                  Bảng dữ liệu ban đầu
                          │
          ┌───────────────┴───────────────┐
          │ (Tách các cột số)             │ (Tách các cột chữ)
          ▼                               ▼
   ┌──────────────┐                ┌──────────────┐
   │ num_pipeline │ (Chạy độc lập) │ cat_pipeline │ (Chạy độc lập)
   └──────┬───────┘                └──────┬───────┘
          │                               │
          └───────────────┬───────────────┘
                          │ (np.hstack ghép ngang lại)
                          ▼
                  Mảng ma trận thống nhất
```

### B. Bảng đối chiếu chi tiết

| Tiêu chí | `Pipeline` | `ColumnTransformer` |
| :--- | :--- | :--- |
| **Hướng dòng chảy** | **Dọc (Nối tiếp)**: Bước sau chờ kết quả bước trước. | **Ngang (Song song)**: Các nhóm cột được xử lý độc lập. |
| **Phạm vi tác động** | Tác động lên **toàn bộ dữ liệu** nó nhận được. | Chỉ tác động lên **danh sách các cột được chỉ định**. |
| **Cú pháp tuple** | `("tên", Transformer_hoặc_Model)` *(Gồm 2 phần tử)* | `("tên", Transformer, [danh_sách_cột])` *(Gồm 3 phần tử)* |
| **Chứa Model được không?** | **CÓ**. Bước cuối cùng có thể là Estimator/Model. | **KHÔNG**. Tất cả các nhánh bắt buộc phải là Transformer. |
| **Có hàm `.predict()`?** | **CÓ** (nếu bước cuối là Model). | **KHÔNG BAO GIỜ**. |
| **Kết quả đầu ra** | Dữ liệu sau khi đi hết trạm cuối cùng. | Mảng ma trận được ghép ngang (`np.hstack`) từ các nhánh. |

### C. Cách kết hợp hoàn hảo trong dự án California Housing

Trong dự án thực tế, người ta luôn kết hợp lồng ghép chúng lại với nhau theo cấu trúc:

```text
full_prediction_pipeline  (Pipeline LỚN - Xử lý DỌC toàn diện)
│
├── Bước 1: full_pipeline (ColumnTransformer - Phân nhánh NGANG)
│   ├── Nhánh 'num_pipeline' (Pipeline con - DỌC): Imputer -> Adder -> Scaler
│   └── Nhánh 'category_pipeline' (Pipeline con - DỌC): Imputer -> OneHot
│
└── Bước 2: best_model (Random Forest - Chốt chặn DỌC cuối cùng)
```
* **Các Pipeline con:** Chịu trách nhiệm xử lý tuần tự (dọc) cho từng loại dữ liệu chuyên biệt.
* **ColumnTransformer:** Đóng vai trò nhạc trưởng phân phối cột nào vào pipeline nào (ngang).
* **Pipeline lớn:** Ghép toàn bộ dây chuyền xử lý dữ liệu với Mô hình AI thành một thực thể duy nhất sẵn sàng sản xuất.

---

## 8. Pipeline Có Thể Thay Thế Model Để Dự Đoán Không? (Cơ Chế "Ủy Quyền" - Delegation)

### Câu hỏi: `Pipeline` có thể thay thế `model` để gọi `.predict()` không?
👉 **Trả lời:** **ĐÚNG VỀ MẶT SỬ DỤNG**, nhưng **VỀ BẢN CHẤT BÊN TRONG LÀ CƠ CHẾ ỦY QUYỀN (DELEGATION)**.

### A. Góc nhìn bên ngoài (Người lập trình & Triển khai)
Bạn hoàn toàn có thể coi `Pipeline` là một **"Siêu Mô Hình" (Meta-Estimator)**:
* Cả `model` và `Pipeline` hoàn chỉnh đều sở hữu đầy đủ: `.fit()`, `.predict()`, `.score()`.
* Thay vì phải viết 2 dòng code thủ công:
  ```python
  # Cách làm rườm rà:
  X_test_prepared = full_pipeline.transform(X_test)
  predictions = model.predict(X_test_prepared)
  ```
* Bạn **thay thế hoàn toàn** bằng 1 dòng lệnh duy nhất qua Pipeline:
  ```python
  # Cách chuyên nghiệp:
  predictions = full_prediction_pipeline.predict(X_test)
  ```

### B. Bản chất bên trong (Hậu trường)
Bản thân `Pipeline` **không có bất kỳ thuật toán AI nào** (không có cây quyết định, không có trọng số). Nó chỉ đóng vai trò là một **"Người đại diện" (Proxy / Wrapper)**:

> 🏥 **Hình ảnh ẩn dụ dễ nhớ:**
> * **`model`** = **Bác sĩ chuyên khoa**: Người duy nhất có chuyên môn để chẩn đoán bệnh (`predict`).
> * **`Pipeline`** = **Dịch vụ Bệnh viện trọn gói**:
>   1. Đón tiếp bệnh nhân và đưa đi làm xét nghiệm máu, chụp X-quang (`full_pipeline.transform()`).
>   2. Mang toàn bộ kết quả xét nghiệm đã có đưa vào phòng cho Bác sĩ xem (`model.predict()`).
>   3. Nhận phiếu chẩn đoán từ Bác sĩ và trả lại cho bệnh nhân.
> 
> 👉 Bệnh nhân chỉ cần đến **Bệnh viện (Pipeline)**, không cần tự mình chạy đi tìm phòng xét nghiệm rồi mới mang tới bác sĩ!

### C. Điều kiện bắt buộc để Pipeline có hàm `.predict()`
Pipeline chỉ có hàm `.predict()` khi và chỉ khi:
> ⚠️ **Bước cuối cùng của Pipeline BẮT BUỘC phải là một Model (Estimator)**!
* `full_prediction_pipeline`: Bước cuối là `RandomForestRegressor` -> **CÓ `.predict()`**.
* `num_pipeline`: Bước cuối là `StandardScaler` (Transformer) -> **KHÔNG CÓ `.predict()`** (gọi sẽ văng lỗi `AttributeError`).

### D. Sức mạnh thực chiến trong Production
1. **Deploy cực gọn:** Người dùng nhập form web -> ném thẳng DataFrame thô vào `pipeline.predict(df)` -> Có ngay kết quả.
2. **Tune đồng thời cả Tiền xử lý và Model trong `GridSearchCV`:**
   ```python
   # Tìm xem điền thiếu bằng median hay mean thì Random Forest cho kết quả tốt hơn:
   param_grid = [{
       "preparation__num_pipeline__imputer__strategy": ["median", "mean"],
       "model__n_estimators": [10, 30, 50]
   }]
   grid_search = GridSearchCV(full_prediction_pipeline, param_grid, cv=5)
   ```


