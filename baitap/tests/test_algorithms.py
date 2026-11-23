"""
Unit Tests kiểm tra tính chính xác của các thuật toán tự viết (From Scratch)
bằng cách đối chiếu trực tiếp với kết quả từ Scikit-Learn.
"""

import sys
import os
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as sk_mse, r2_score as sk_r2

from src.algorithms.scalers import StandardScalerScratch, SimpleImputerScratch
from src.algorithms.linear_regression import LinearRegressionScratch
from src.algorithms.metrics import mean_squared_error, r2_score


class TestScratchAlgorithms(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5) * 10 + 5
        self.y = self.X @ np.array([2.5, -1.2, 0.8, 3.0, -0.5]) + 15.0 + np.random.randn(100) * 0.1

    def test_metrics(self):
        y_pred = self.y + np.random.randn(100) * 0.5
        my_mse = mean_squared_error(self.y, y_pred)
        sk_val_mse = sk_mse(self.y, y_pred)
        self.assertAlmostEqual(my_mse, sk_val_mse, places=6)

        my_r2 = r2_score(self.y, y_pred)
        sk_val_r2 = sk_r2(self.y, y_pred)
        self.assertAlmostEqual(my_r2, sk_val_r2, places=6)

    def test_standard_scaler(self):
        scaler_scratch = StandardScalerScratch()
        X_scaled_scratch = scaler_scratch.fit_transform(self.X)

        scaler_sk = StandardScaler()
        X_scaled_sk = scaler_sk.fit_transform(self.X)

        np.testing.assert_allclose(X_scaled_scratch, X_scaled_sk, rtol=1e-5, atol=1e-5)

    def test_simple_imputer(self):
        X_missing = self.X.copy()
        X_missing[0, 1] = np.nan
        X_missing[5, 3] = np.nan

        imputer_scratch = SimpleImputerScratch(strategy="median")
        X_imp_scratch = imputer_scratch.fit_transform(X_missing)

        imputer_sk = SimpleImputer(strategy="median")
        X_imp_sk = imputer_sk.fit_transform(X_missing)

        np.testing.assert_allclose(X_imp_scratch.astype(float), X_imp_sk.astype(float), rtol=1e-5, atol=1e-5)

    def test_linear_regression(self):
        lr_scratch = LinearRegressionScratch(method="normal_equation")
        lr_scratch.fit(self.X, self.y)
        preds_scratch = lr_scratch.predict(self.X)

        lr_sk = LinearRegression()
        lr_sk.fit(self.X, self.y)
        preds_sk = lr_sk.predict(self.X)

        np.testing.assert_allclose(preds_scratch, preds_sk, rtol=1e-4, atol=1e-4)
        self.assertAlmostEqual(lr_scratch.intercept_, lr_sk.intercept_, places=4)
        np.testing.assert_allclose(lr_scratch.coef_, lr_sk.coef_, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
