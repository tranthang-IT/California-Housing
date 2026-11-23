"""
Module Pipeline và ColumnTransformer tự xây dựng từ đầu (Custom Pipeline Architecture).
Cho phép kết nối tuần tự hoặc song song các bước tiền xử lý và mô hình dự đoán.
"""

import numpy as np
import pandas as pd


class CustomPipeline:
    """
    Chuỗi xử lý dữ liệu tuần tự (Pipeline tự viết):
    Nhận danh sách các bước [('tên_bước', đối_tượng_bước), ...]
    Tự động chuyển tiếp output của bước trước làm input cho bước sau.
    """

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_curr = X
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                X_curr = step.fit_transform(X_curr, y)
            else:
                X_curr = step.fit(X_curr, y).transform(X_curr)

        # Bước cuối cùng
        last_name, last_step = self.steps[-1]
        last_step.fit(X_curr, y)
        return self

    def transform(self, X):
        X_curr = X
        for name, step in self.steps:
            if hasattr(step, "transform"):
                X_curr = step.transform(X_curr)
        return X_curr

    def fit_transform(self, X, y=None):
        X_curr = X
        for name, step in self.steps:
            if hasattr(step, "fit_transform"):
                X_curr = step.fit_transform(X_curr, y)
            elif hasattr(step, "transform"):
                X_curr = step.fit(X_curr, y).transform(X_curr)
            else:
                X_curr = step.fit(X_curr, y)
        return X_curr

    def predict(self, X):
        X_curr = X
        for name, step in self.steps[:-1]:
            X_curr = step.transform(X_curr)

        last_name, last_step = self.steps[-1]
        return last_step.predict(X_curr)


class CustomColumnTransformer:
    """
    Bộ hợp nhất xử lý song song theo cột (ColumnTransformer tự viết):
    Nhận danh sách [('tên_nhánh', pipeline_hoặc_transformer, danh_sách_cột), ...]
    Thực hiện xử lý độc lập từng nhóm cột và ghép nối (concatenate) lại.
    """

    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X, y=None):
        for name, transformer, columns in self.transformers:
            if isinstance(X, pd.DataFrame):
                sub_X = X[columns]
            else:
                sub_X = X[:, columns]
            transformer.fit(sub_X, y)
        return self

    def transform(self, X):
        transformed_parts = []
        for name, transformer, columns in self.transformers:
            if isinstance(X, pd.DataFrame):
                sub_X = X[columns]
            else:
                sub_X = X[:, columns]

            part = transformer.transform(sub_X)
            transformed_parts.append(np.asarray(part))

        return np.hstack(transformed_parts)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
