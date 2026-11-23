"""
Cây quyết định hồi quy (Decision Tree Regressor) tự viết từ đầu bằng NumPy.
Sử dụng tiêu chuẩn giảm phương sai (Variance Reduction / MSE reduction) để phân nhánh.
Tối ưu tốc độ huấn luyện bằng cách lấy mẫu phân vị ngưỡng (quantile thresholding).
"""

import numpy as np


class Node:
    """Đại diện cho một nút (Node) trong Cây quyết định."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # Chỉ số thuộc tính dùng để phân tách
        self.threshold = threshold      # Giá trị ngưỡng phân tách
        self.left = left                # Nhánh con trái (<= threshold)
        self.right = right              # Nhánh con phải (> threshold)
        self.value = value              # Giá trị dự đoán (nếu là nút lá)

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressorScratch:
    """
    Cây quyết định hồi quy (Decision Tree Regressor) tự cài đặt:
    - Thuật toán: Phân nhánh nhị phân đệ quy (Recursive Binary Splitting).
    - Tiêu chí đánh giá phân tách: Tối đa hóa mức giảm phương sai (Variance Reduction).
    - Tối ưu hóa hiệu năng: Đánh giá nhanh trên các phân vị mẫu (quantile thresholds).
    """

    def __init__(self, max_depth=6, min_samples_split=10, min_samples_leaf=5, n_thresholds=20, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.root = self._build_tree(X_arr, y_arr, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # Điều kiện dừng đệ quy tạo nút lá:
        # 1. Đạt độ sâu tối đa
        # 2. Số mẫu nhỏ hơn min_samples_split
        # 3. Tất cả giá trị mục tiêu y đồng nhất (phương sai = 0)
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) == 0.0:
            return Node(value=float(np.mean(y)))

        best_feat, best_thresh, best_gain = self._best_split(X, y, n_features)

        # Nếu không tìm thấy phân tách hợp lệ thỏa mãn điều kiện
        if best_gain <= 0.0 or best_feat is None:
            return Node(value=float(np.mean(y)))

        # Phân chia dữ liệu theo ngưỡng tối ưu
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh, left=left_node, right=right_node)

    def _best_split(self, X, y, n_features):
        best_gain = -1.0
        best_feat, best_thresh = None, None
        current_variance = np.var(y)
        n_samples = len(y)

        for feat_idx in range(n_features):
            col_values = X[:, feat_idx]
            unique_vals = np.unique(col_values)
            if len(unique_vals) <= 1:
                continue

            # Lấy các ngưỡng ứng viên qua quantile để tối ưu tốc độ xử lý trên tập dữ liệu lớn
            if len(unique_vals) > self.n_thresholds:
                percentiles = np.linspace(5, 95, self.n_thresholds)
                thresholds = np.percentile(col_values, percentiles)
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = col_values <= thresh
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[~left_mask]

                # Mức giảm phương sai: Gain = Var(y) - [ (N_L/N)*Var(y_L) + (N_R/N)*Var(y_R) ]
                gain = current_variance - (n_left / n_samples * np.var(y_left) + n_right / n_samples * np.var(y_right))

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def predict(self, X):
        if self.root is None:
            raise RuntimeError("Cây chưa được huấn luyện. Hãy gọi fit() trước.")
        X_arr = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root) for row in X_arr])

    def _predict_row(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)
