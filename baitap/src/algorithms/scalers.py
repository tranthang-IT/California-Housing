"""
Các class tiền xử lý và chuẩn hóa dữ liệu tự viết từ đầu bằng NumPy:
1. SimpleImputerScratch: Điền giá trị khuyết thiếu (median, mean, most_frequent).
2. StandardScalerScratch: Chuẩn hóa Z-score (mean=0, std=1).
3. OneHotEncoderScratch: Mã hóa biến phân loại thành vector nhị phân.
"""

import numpy as np
import pandas as pd


class SimpleImputerScratch:
    """
    Điền giá trị thiếu (Missing Values Imputation) theo chuẩn fit / transform:
    Hỗ trợ strategy:
    - 'median': Điền bằng giá trị trung vị (tốt cho phân phối lệch/outlier).
    - 'mean': Điền bằng giá trị trung bình.
    - 'most_frequent': Điền bằng giá trị xuất hiện nhiều nhất (mode).
    """

    def __init__(self, strategy="median"):
        if strategy not in ["median", "mean", "most_frequent"]:
            raise ValueError(f"Strategy '{strategy}' không được hỗ trợ. Chọn: 'median', 'mean', 'most_frequent'.")
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)

        n_features = X_arr.shape[1]
        self.statistics_ = []

        for i in range(n_features):
            col = X_arr[:, i]
            # Lọc bỏ các giá trị nan / None
            mask = pd.notna(col)
            valid_vals = col[mask]

            if len(valid_vals) == 0:
                stat = 0.0
            elif self.strategy == "median":
                stat = float(np.median(valid_vals.astype(float)))
            elif self.strategy == "mean":
                stat = float(np.mean(valid_vals.astype(float)))
            elif self.strategy == "most_frequent":
                vals, counts = np.unique(valid_vals, return_counts=True)
                stat = vals[np.argmax(counts)]

            self.statistics_.append(stat)

        self.statistics_ = np.array(self.statistics_, dtype=object)
        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise RuntimeError("Cần gọi fit() trước khi transform().")

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(copy=True)
        else:
            X_arr = np.array(X, copy=True)

        for i in range(X_arr.shape[1]):
            mask = pd.isna(X_arr[:, i])
            if np.any(mask):
                X_arr[mask, i] = self.statistics_[i]

        return X_arr

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class StandardScalerScratch:
    """
    Chuẩn hóa đặc trưng Z-score:
    z = (x - mean) / std
    Đưa phân phối về kỳ vọng xấp xỉ 0 và phương sai bằng 1.
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        if self.with_mean:
            self.mean_ = np.mean(X_arr, axis=0)
        else:
            self.mean_ = np.zeros(X_arr.shape[1])

        if self.with_std:
            std = np.std(X_arr, axis=0)
            # Tránh chia cho 0 khi cột có phương sai bằng 0
            std[std == 0.0] = 1.0
            self.scale_ = std
        else:
            self.scale_ = np.ones(X_arr.shape[1])

        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Cần gọi fit() trước khi transform().")
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        """Khôi phục lại giá trị về thang đo gốc."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Cần gọi fit() trước khi inverse_transform().")
        X_arr = np.asarray(X, dtype=float)
        return (X_arr * self.scale_) + self.mean_


class OneHotEncoderScratch:
    """
    Mã hóa biến phân loại One-Hot Encoding tự cài đặt.
    Mỗi danh mục duy nhất trong cột sẽ trở thành 1 cột nhị phân (0 hoặc 1).
    """

    def __init__(self, handle_unknown="ignore"):
        self.handle_unknown = handle_unknown
        self.categories_ = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)

        self.categories_ = []
        for i in range(X_arr.shape[1]):
            cats = np.unique(X_arr[:, i].astype(str))
            self.categories_.append(cats)
        return self

    def transform(self, X):
        if not self.categories_:
            raise RuntimeError("Cần gọi fit() trước khi transform().")

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)

        encoded_cols = []
        for i in range(X_arr.shape[1]):
            col = X_arr[:, i].astype(str)
            cats = self.categories_[i]
            for cat in cats:
                # 1.0 nếu bằng cat, ngược lại 0.0
                binary_col = (col == cat).astype(float)
                encoded_cols.append(binary_col)

        return np.column_stack(encoded_cols)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
