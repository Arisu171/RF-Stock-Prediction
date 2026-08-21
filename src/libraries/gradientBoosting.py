# =====================================================================
# Gradient Boosting — tăng cường độ dốc (Friedman, 2001)
# =====================================================================
# Cài đặt thuần Python, không phụ thuộc thư viện ngoài.
#
# Đối lập với Random Forest về cơ chế:
#   - Random Forest    : nuôi các cây SÂU, ĐỘC LẬP rồi lấy trung bình
#                        → giảm PHƯƠNG SAI.
#   - Gradient Boosting: nuôi các cây NÔNG, TUẦN TỰ, cây sau sửa lỗi của
#                        tổ hợp trước → giảm ĐỘ CHỆCH.
#
# Mô hình cộng dồn:
#       F_0(x)  = giá trị khởi tạo tối ưu của hàm mất mát
#       F_m(x)  = F_{m-1}(x) + ν · h_m(x)
# trong đó h_m là cây hồi quy khớp với gradient âm của hàm mất mát và
# ν là learning_rate (shrinkage).
#
# Thứ tự khai báo giữ đúng quan hệ kế thừa:
#
#   ① BaseGradientBoosting       — vòng lặp cộng dồn, không gắn hàm mất mát
#   ② GradientBoostingRegressor  — mất mát bình phương: gradient = phần dư
#   ③ GradientBoostingClassifier — log-loss: thêm bước Newton tại lá
#
# Mọi khác biệt giữa ② và ③ gói gọn trong bốn điểm mở rộng của ①:
# giá trị khởi tạo F₀, gradient âm, hàm mất mát và cách hiệu chỉnh lá.
# =====================================================================

import math
import random

from . import rfMath
from .decisionTree import DecisionTreeRegressor


# ---------------------------------------------------------------------
# ① Khung boosting — vòng lặp cộng dồn F_m = F_{m-1} + ν·h_m
# ---------------------------------------------------------------------
class BaseGradientBoosting:
    """
    Khung chung cho boosting hồi quy và boosting phân loại nhị phân.

    Parameters:
        n_estimators      : số vòng lặp boosting (số cây)
        learning_rate     : hệ số co ν — càng nhỏ càng cần nhiều cây
                            nhưng tổng quát hóa tốt hơn
        max_depth         : độ sâu mỗi cây; boosting dùng cây NÔNG (2–5)
        min_samples_split : số mẫu tối thiểu để chia một nút
        min_samples_leaf  : số mẫu tối thiểu tại mỗi lá
        max_features      : số đặc trưng xét mỗi nút (None = tất cả)
        subsample         : tỷ lệ mẫu dùng cho mỗi cây (< 1.0 → stochastic
                            gradient boosting, thêm tính ngẫu nhiên)
        random_state      : hạt giống ngẫu nhiên
        verbose           : in tiến độ sau mỗi verbose cây
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, max_features=None,
                 subsample=1.0, random_state=None, verbose=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.subsample = subsample
        self.random_state = random_state
        self.verbose = verbose

        self.trees = []
        self.initial_prediction = 0.0
        self.num_features = 0
        self.training_loss_history = []

    # ── Huấn luyện ──────────────────────────────────────────────────

    def fit(self, samples, targets):
        """
        Huấn luyện mô hình cộng dồn.

        Parameters:
            samples : list of lists — mỗi phần tử là một mẫu
            targets : list giá trị mục tiêu

        Returns:
            chính đối tượng mô hình
        """
        if len(samples) != len(targets):
            raise ValueError(
                f"Số mẫu ({len(samples)}) phải bằng số giá trị mục tiêu "
                f"({len(targets)})."
            )
        if not samples:
            raise ValueError("Không thể huấn luyện trên tập rỗng.")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample phải nằm trong khoảng (0, 1].")

        self.num_features = len(samples[0])
        self._prepare_target_space(targets)
        encoded_targets = self._encode_targets(targets)

        self.initial_prediction = self._calculate_initial_prediction(encoded_targets)
        current_scores = [
            self.initial_prediction for _ in range(len(samples))
        ]

        seed_generator = random.Random(self.random_state)
        num_subsample = max(1, int(self.subsample * len(samples)))

        self.trees = []
        self.training_loss_history = []

        for tree_position in range(self.n_estimators):
            negative_gradients = self._calculate_negative_gradients(
                encoded_targets, current_scores
            )

            if self.subsample < 1.0:
                sampler = random.Random(seed_generator.randrange(2 ** 31))
                indices = sampler.sample(range(len(samples)), num_subsample)
            else:
                indices = list(range(len(samples)))

            tree = DecisionTreeRegressor(
                criterion='variance',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed_generator.randrange(2 ** 31),
            )
            tree.fit(
                rfMath.select_by_indices(samples, indices),
                rfMath.select_by_indices(negative_gradients, indices),
            )

            self._refine_leaf_values(
                tree, samples, encoded_targets, current_scores, negative_gradients
            )

            step_values = tree.predict(samples)
            current_scores = [
                score + self.learning_rate * step
                for score, step in zip(current_scores, step_values)
            ]

            self.trees.append(tree)
            self.training_loss_history.append(
                self._calculate_loss(encoded_targets, current_scores)
            )

            if self.verbose and (tree_position + 1) % self.verbose == 0:
                print(f"  Vòng {tree_position + 1}/{self.n_estimators} — "
                      f"loss = {self.training_loss_history[-1]:.6f}")

        return self

    # ── Điểm số cộng dồn ────────────────────────────────────────────

    def calculate_scores(self, samples, num_trees=None):
        """
        Điểm số thô F(x) = F_0 + ν·Σ h_m(x) trước khi biến đổi về không
        gian đầu ra.

        Parameters:
            samples   : list of lists
            num_trees : chỉ dùng num_trees cây đầu tiên (None = tất cả)

        Returns:
            list điểm số
        """
        self._assert_fitted()
        limit = len(self.trees) if num_trees is None else num_trees
        limit = max(0, min(limit, len(self.trees)))

        scores = [self.initial_prediction for _ in range(len(samples))]
        for tree in self.trees[:limit]:
            step_values = tree.predict(samples)
            scores = [
                score + self.learning_rate * step
                for score, step in zip(scores, step_values)
            ]
        return scores

    # ── Tầm quan trọng đặc trưng ────────────────────────────────────

    def calculate_feature_importances(self, normalize=True):
        """
        Trung bình mức giảm hỗn tạp của các cây trong chuỗi boosting.

        Returns:
            list độ dài num_features
        """
        self._assert_fitted()
        totals = [0.0 for _ in range(self.num_features)]
        for tree in self.trees:
            for index, value in enumerate(
                tree.calculate_feature_importances(normalize=False)
            ):
                totals[index] += value

        averaged = [value / len(self.trees) for value in totals]
        if normalize:
            return rfMath.normalize_to_distribution(averaged)
        return averaged

    # ── Các điểm mở rộng cho lớp con ────────────────────────────────

    def _prepare_target_space(self, targets):
        """Chuẩn bị thông tin phụ thuộc kiểu bài toán (mặc định: không có)."""

    def _encode_targets(self, targets):
        """Mã hóa mục tiêu về dạng số mà hàm mất mát làm việc trực tiếp."""
        return list(targets)

    def _calculate_initial_prediction(self, encoded_targets):
        raise NotImplementedError

    def _calculate_negative_gradients(self, encoded_targets, current_scores):
        raise NotImplementedError

    def _calculate_loss(self, encoded_targets, current_scores):
        raise NotImplementedError

    def _refine_leaf_values(self, tree, samples, encoded_targets,
                            current_scores, negative_gradients):
        """
        Hiệu chỉnh giá trị lá sau khi dựng cây (mặc định: giữ nguyên).

        Với hàm mất mát không phải bình phương sai số, giá trị trung bình
        của gradient tại lá không phải bước đi tối ưu; lớp con ghi đè
        phương thức này để thực hiện một bước Newton.
        """

    def _assert_fitted(self):
        if not self.trees:
            raise RuntimeError("Mô hình chưa được huấn luyện — hãy gọi fit() trước.")


# ---------------------------------------------------------------------
# ② Boosting hồi quy — mất mát bình phương, gradient âm chính là phần dư
# ---------------------------------------------------------------------
class GradientBoostingRegressor(BaseGradientBoosting):
    """
    Gradient Boosting với hàm mất mát bình phương sai số.

    L(y, F) = (1/2)·(y - F)²        →  gradient âm = y - F = phần dư

    Nghĩa là mỗi cây đơn giản học phần dư của tổ hợp trước đó.
    """

    def _calculate_initial_prediction(self, encoded_targets):
        return rfMath.calculate_mean(encoded_targets)

    def _calculate_negative_gradients(self, encoded_targets, current_scores):
        return [
            target - score
            for target, score in zip(encoded_targets, current_scores)
        ]

    def _calculate_loss(self, encoded_targets, current_scores):
        return sum(
            (target - score) ** 2
            for target, score in zip(encoded_targets, current_scores)
        ) / (2 * len(encoded_targets))

    def predict(self, samples, num_trees=None):
        """Dự đoán giá trị thực."""
        return self.calculate_scores(samples, num_trees)


# ---------------------------------------------------------------------
# ③ Boosting phân loại — log-loss, thêm bước Newton hiệu chỉnh giá trị lá
# ---------------------------------------------------------------------
class GradientBoostingClassifier(BaseGradientBoosting):
    """
    Gradient Boosting với hàm mất mát log-loss cho phân loại nhị phân.

        p(x)      = σ(F(x))
        L(y, F)   = -[ y·log p + (1-y)·log(1-p) ]
        -∂L/∂F    = y - p            (gradient âm)

    Sau khi dựng cây trên gradient âm, giá trị mỗi lá được thay bằng một
    bước Newton — bước tối ưu bậc hai của log-loss trên vùng lá đó:

        γ = Σ (y - p) / Σ p·(1 - p)

    Chỉ hỗ trợ đúng hai lớp; nhãn được mã hóa về {0, 1} theo thứ tự sắp
    xếp của không gian nhãn.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, max_features=None,
                 subsample=1.0, random_state=None, verbose=0,
                 max_leaf_step=8.0):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            subsample=subsample,
            random_state=random_state,
            verbose=verbose,
        )
        self.max_leaf_step = max_leaf_step
        self.label_space = []

    def _prepare_target_space(self, targets):
        self.label_space = sorted(set(targets))
        if len(self.label_space) != 2:
            raise ValueError(
                f"Chỉ hỗ trợ phân loại nhị phân, nhận được "
                f"{len(self.label_space)} lớp: {self.label_space}."
            )

    def _encode_targets(self, targets):
        """Nhãn nhỏ hơn → 0, nhãn lớn hơn → 1."""
        positive_label = self.label_space[1]
        return [1.0 if target == positive_label else 0.0 for target in targets]

    def _calculate_initial_prediction(self, encoded_targets):
        """Log-odds của tỷ lệ lớp dương — nghiệm tối ưu của hằng số F_0."""
        return rfMath.logit(rfMath.calculate_mean(encoded_targets))

    def _calculate_negative_gradients(self, encoded_targets, current_scores):
        return [
            target - rfMath.sigmoid(score)
            for target, score in zip(encoded_targets, current_scores)
        ]

    def _calculate_loss(self, encoded_targets, current_scores):
        """Log-loss trung bình: -[ y·log p + (1-y)·log(1-p) ]"""
        epsilon = 1e-15
        total = 0.0
        for target, score in zip(encoded_targets, current_scores):
            probability = rfMath.clip_value(
                rfMath.sigmoid(score), epsilon, 1.0 - epsilon
            )
            total -= (target * math.log(probability)
                      + (1.0 - target) * math.log(1.0 - probability))
        return total / len(encoded_targets)

    def _refine_leaf_values(self, tree, samples, encoded_targets,
                            current_scores, negative_gradients):
        """
        Thay giá trị trung bình gradient tại mỗi lá bằng một bước Newton.
        """
        leaves = tree.apply(samples)
        numerators = {}
        denominators = {}

        for position, leaf in enumerate(leaves):
            key = id(leaf)
            probability = rfMath.sigmoid(current_scores[position])
            numerators[key] = numerators.get(key, 0.0) + negative_gradients[position]
            denominators[key] = denominators.get(key, 0.0) + \
                probability * (1.0 - probability)

        epsilon = 1e-12
        for leaf in leaves:
            key = id(leaf)
            denominator = denominators[key]
            if denominator < epsilon:
                leaf.value = 0.0
            else:
                leaf.value = rfMath.clip_value(
                    numerators[key] / denominator,
                    -self.max_leaf_step, self.max_leaf_step
                )

    def predict_probabilities(self, samples, num_trees=None):
        """
        Xác suất của cả hai lớp.

        Returns:
            list of lists — [P(lớp âm), P(lớp dương)] theo thứ tự label_space
        """
        positive_probabilities = self.predict_scores(samples, num_trees)
        return [
            [1.0 - probability, probability]
            for probability in positive_probabilities
        ]

    def predict_scores(self, samples, num_trees=None):
        """Xác suất thuộc lớp dương (nhãn lớn hơn trong label_space)."""
        return [
            rfMath.sigmoid(score)
            for score in self.calculate_scores(samples, num_trees)
        ]

    def predict(self, samples, threshold=0.5, num_trees=None):
        """
        Dự đoán nhãn theo ngưỡng xác suất.

        Parameters:
            threshold : ngưỡng quyết định trên xác suất lớp dương
        """
        return [
            self.label_space[1] if probability >= threshold else self.label_space[0]
            for probability in self.predict_scores(samples, num_trees)
        ]
