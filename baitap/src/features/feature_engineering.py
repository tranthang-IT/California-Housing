"""
Module kỹ thuật đặc trưng (Feature Engineering) tạo các chỉ số tỷ lệ quan trọng.
"""

import numpy as np
import pandas as pd


class CombinedAttributesAdderScratch:
    """
    Tự động sinh ra các đặc trưng quan trọng từ dữ liệu bất động sản:
    1. rooms_per_household: Số phòng trên mỗi hộ gia đình
    2. population_per_household: Số nhân khẩu trên mỗi hộ gia đình
    3. bedrooms_per_room: Tỷ lệ phòng ngủ trên tổng số phòng
    """

    def __init__(self, add_bedrooms_per_room=True, rooms_ix=3, bedrooms_ix=4, population_ix=5, households_ix=6):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = rooms_ix
        self.bedrooms_ix = bedrooms_ix
        self.population_ix = population_ix
        self.households_ix = households_ix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)

        rooms_per_household = X_arr[:, self.rooms_ix] / X_arr[:, self.households_ix]
        population_per_household = X_arr[:, self.population_ix] / X_arr[:, self.households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X_arr[:, self.bedrooms_ix] / X_arr[:, self.rooms_ix]
            return np.c_[X_arr, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X_arr, rooms_per_household, population_per_household]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
