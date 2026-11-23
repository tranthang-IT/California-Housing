"""
Module phân chia tập dữ liệu Train / Test theo phương pháp Phân tầng (Stratified Sampling).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_test_split(data, test_size=0.2, random_state=42):
    """
    Phân tầng dựa trên đặc trưng 'median_income' nhằm chống thiên lệch lấy mẫu (sampling bias).
    """
    data_copy = data.copy()

    # Phân chia thành 5 nhóm thu nhập
    data_copy["income_category"] = pd.cut(
        data_copy["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    train_set, test_set = train_test_split(
        data_copy,
        test_size=test_size,
        stratify=data_copy["income_category"],
        random_state=random_state
    )

    for s in (train_set, test_set):
        s.drop("income_category", axis=1, inplace=True)

    return train_set, test_set
