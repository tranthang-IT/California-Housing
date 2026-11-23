"""
Hồi quy tuyến tính (Linear Regression) tự viết từ đầu bằng NumPy.
Hỗ trợ 2 phương pháp tối ưu:
1. Normal Equation (Phương trình chuẩn giải tích): Nghiệm chính xác tuyệt đối.
2. Gradient Descent (Hạ độ dốc): Tối ưu tham số qua từng vòng lặp (Epochs).
"""

import numpy as np


class LinearRegressionScratch:
    """
    Mô hình Hồi quy tuyến tính tự cài đặt thuần NumPy:
    y_hat = theta_0 + theta_1 * x_1 + ... + theta_n * x_n = X_b @ theta
    """

    def __init__(self, method="normal_equation", learning_rate=0.05, n_iterations=1000, fit_intercept=True):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.theta_ = None
        self.intercept_ = None
        self.coef_ = None
        self.cost_history_ = []

    def fit(self, X, y):
        """
        Huấn luyện mô hình để tìm vector trọng số theta.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        m, n = X_arr.shape

        # Thêm cột x0 = 1 cho bias / intercept nếu fit_intercept=True
        if self.fit_intercept:
            X_b = np.c_[np.ones((m, 1)), X_arr]
        else:
            X_b = X_arr

        if self.method == "normal_equation":
            # Công thức Normal Equation: theta = (X_b.T @ X_b)^(-1) @ X_b.T @ y
            # Sử dụng pinv (Moore-Penrose pseudo-inverse) để chống suy biến ma trận và tăng độ ổn định số học
            self.theta_ = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_arr

        elif self.method == "gradient_descent":
            # Khởi tạo theta ngẫu nhiên hoặc bằng 0
            np.random.seed(42)
            n_features_b = X_b.shape[1]
            self.theta_ = np.zeros(n_features_b)
            self.cost_history_ = []

            for _ in range(self.n_iterations):
                gradients = (2.0 / m) * X_b.T @ (X_b @ self.theta_ - y_arr)
                self.theta_ -= self.learning_rate * gradients
                # Tính MSE cost
                cost = np.mean((X_b @ self.theta_ - y_arr) ** 2)
                self.cost_history_.append(cost)
        else:
            raise ValueError(f"Method '{self.method}' không hợp lệ. Chọn 'normal_equation' hoặc 'gradient_descent'.")

        # Tách bias (intercept) và weights (coef) để tương thích Scikit-Learn
        if self.fit_intercept:
            self.intercept_ = float(self.theta_[0])
            self.coef_ = self.theta_[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.theta_

        return self

    def predict(self, X):
        """
        Dự đoán giá trị đầu ra y_hat cho tập dữ liệu X.
        """
        if self.theta_ is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        X_arr = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X_b = np.c_[np.ones((X_arr.shape[0], 1)), X_arr]
        else:
            X_b = X_arr

        return X_b @ self.theta_
