"""
Module tải và kiểm tra dữ liệu đầu vào.
"""

import os
import pandas as pd


def load_data(path_csv):
    """
    Đọc dữ liệu từ file CSV, kiểm tra sự tồn tại của tệp và trả về pandas DataFrame.
    """
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại đường dẫn: '{path_csv}'")
    try:
        data = pd.read_csv(path_csv)
        return data
    except Exception as e:
        raise RuntimeError(f"Lỗi khi đọc file CSV '{path_csv}': {e}")
