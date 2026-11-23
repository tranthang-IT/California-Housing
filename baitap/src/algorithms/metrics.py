"""
Thư viện các hàm đánh giá sai số (Evaluation Metrics) tự viết từ đầu bằng NumPy.
Bao gồm: MSE, RMSE, MAE, R2-score.
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Tính Mean Squared Error (Sai số bình phương trung bình):
    MSE = (1 / n) * sum((y_true - y_pred) ** 2)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("Độ dài y_true và y_pred phải bằng nhau.")
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Tính Root Mean Squared Error (Căn bậc hai của MSE):
    RMSE = sqrt(MSE)
    Đơn vị cùng đơn vị với biến mục tiêu (ở đây là USD).
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """
    Tính Mean Absolute Error (Sai số tuyệt đối trung bình):
    MAE = (1 / n) * sum(|y_true - y_pred|)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Tính Hệ số xác định R^2 (R-squared score):
    R^2 = 1 - (SS_res / SS_tot)
    Trong đó:
      SS_res = sum((y_true - y_pred) ** 2)
      SS_tot = sum((y_true - mean(y_true)) ** 2)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)
