# =====================================================================
# Random Forest — rừng ngẫu nhiên (Breiman, 2001)
# =====================================================================
# Cài đặt thuần Python, không phụ thuộc thư viện ngoài.
#
# Random Forest = Cây quyết định + Bagging + Ngẫu nhiên hóa đặc trưng.
# Hai nguồn ngẫu nhiên làm giảm tương quan ρ giữa các cây, nhờ đó
# phương sai của kết quả tổng hợp giảm theo công thức:
#
#       Var(trung bình B cây) = ρ·σ² + (1 - ρ)·σ² / B
#
# Số hạng thứ hai triệt tiêu khi B tăng; số hạng đầu chỉ giảm được bằng
# cách làm các cây bớt giống nhau — đó chính là vai trò của max_features.
#
# Thứ tự khai báo giữ đúng quan hệ kế thừa:
#
#   ① BaseRandomForest       — nuôi cây, giữ tập OOB, đo tầm quan trọng
#   ② RandomForestClassifier — gộp MỀM bằng trung bình xác suất
#   ③ RandomForestRegressor  — gộp bằng trung bình cộng giá trị
#
# Toàn bộ phần khó (bootstrap, OOB, permutation importance) nằm ở ①;
# ② và ③ chỉ quy định cách tạo cây con và cách gộp dự đoán.
# =====================================================================

import random

from . import rfMath
from .decisionTree import DecisionTreeClassifier, DecisionTreeRegressor


# ---------------------------------------------------------------------
# ① Khung rừng — bagging + OOB + tầm quan trọng, dùng chung hai kiểu bài toán
# ---------------------------------------------------------------------
class BaseRandomForest:
    """
    Khung chung cho rừng phân loại và rừng hồi quy.

    Thuật toán huấn luyện:
        Lặp b = 1 … B:
            1. Rút một mẫu bootstrap gồm n mẫu có hoàn lại.
            2. Nuôi một cây trên mẫu đó; tại mỗi nút chỉ xét ngẫu nhiên
               m trong p đặc trưng.
            3. Ghi lại tập out-of-bag của cây để ước lượng lỗi.

    Parameters:
        n_estimators          : số cây trong rừng (B)
        criterion             : tiêu chí phân tách của từng cây
        max_depth             : độ sâu tối đa mỗi cây
        min_samples_split     : số mẫu tối thiểu để chia một nút
        min_samples_leaf      : số mẫu tối thiểu tại mỗi lá
        min_impurity_decrease : mức giảm hỗn tạp tối thiểu để chấp nhận phép chia
        max_features          : số đặc trưng xét mỗi nút (m)
        max_thresholds        : số ngưỡng tối đa xét trên mỗi đặc trưng
        bootstrap             : True = lấy mẫu bootstrap (bắt buộc nếu muốn OOB)
        random_state          : hạt giống ngẫu nhiên gốc
        verbose               : in tiến độ huấn luyện sau mỗi verbose cây
    """

    def __init__(self, n_estimators=100, criterion=None, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, max_features='sqrt',
                 max_thresholds=None, bootstrap=True, random_state=None,
                 verbose=0):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose

        self.trees = []
        self.out_of_bag_indices = []
        self.training_samples = []
        self.training_targets = []
        self.num_features = 0

    # ── Huấn luyện ──────────────────────────────────────────────────

    def fit(self, samples, targets):
        """
        Nuôi toàn bộ rừng.

        Parameters:
            samples : list of lists — mỗi phần tử là một mẫu
            targets : list giá trị mục tiêu

        Returns:
            chính đối tượng rừng
        """
        if len(samples) != len(targets):
            raise ValueError(
                f"Số mẫu ({len(samples)}) phải bằng số giá trị mục tiêu "
                f"({len(targets)})."
            )
        if not samples:
            raise ValueError("Không thể huấn luyện trên tập rỗng.")
        if self.n_estimators < 1:
            raise ValueError("n_estimators phải ≥ 1.")

        self.training_samples = samples
        self.training_targets = targets
        self.num_features = len(samples[0])
        self._prepare_target_space(targets)

        num_samples = len(samples)
        seed_generator = random.Random(self.random_state)

        self.trees = []
        self.out_of_bag_indices = []

        for tree_position in range(self.n_estimators):
            tree_seed = seed_generator.randrange(2 ** 31)
            sampler = random.Random(tree_seed)

            if self.bootstrap:
                in_bag, out_of_bag = rfMath.generate_bootstrap_indices(
                    num_samples, sampler
                )
            else:
                in_bag = list(range(num_samples))
                out_of_bag = []

            tree = self._create_tree(tree_seed)
            tree.fit(
                rfMath.select_by_indices(samples, in_bag),
                rfMath.select_by_indices(targets, in_bag),
            )

            self.trees.append(tree)
            self.out_of_bag_indices.append(out_of_bag)

            if self.verbose and (tree_position + 1) % self.verbose == 0:
                print(f"  Đã nuôi {tree_position + 1}/{self.n_estimators} cây")

        return self

    def _create_tree(self, tree_seed):
        """Khởi tạo một cây con — lớp con phải cài đặt."""
        raise NotImplementedError

    def _prepare_target_space(self, targets):
        """Chuẩn bị thông tin phụ thuộc kiểu bài toán (mặc định: không có)."""

    # ── Dự đoán ─────────────────────────────────────────────────────

    def predict(self, samples):
        """
        Dự đoán cho một danh sách mẫu bằng cách tổng hợp toàn bộ cây.

        Returns:
            list giá trị dự đoán
        """
        self._assert_fitted()
        tree_predictions = [tree.predict(samples) for tree in self.trees]
        return [
            self._aggregate([predictions[position]
                             for predictions in tree_predictions])
            for position in range(len(samples))
        ]

    def _aggregate(self, predictions):
        """Tổng hợp dự đoán của các cây cho một mẫu — lớp con cài đặt."""
        raise NotImplementedError

    # ── Ước lượng lỗi Out-Of-Bag ────────────────────────────────────

    def calculate_out_of_bag_predictions(self, num_trees=None):
        """
        Dự đoán out-of-bag cho từng mẫu huấn luyện: mỗi mẫu chỉ được dự
        đoán bởi những cây KHÔNG dùng nó khi huấn luyện.

        Nhờ vậy có ngay một ước lượng lỗi không thiên lệch mà không cần
        tách riêng tập kiểm định — khoảng 36.8% số mẫu nằm ngoài túi ở
        mỗi cây.

        Parameters:
            num_trees : chỉ dùng num_trees cây đầu tiên (None = tất cả)

        Returns:
            covered_indices     : list chỉ số mẫu có ít nhất một cây OOB
            out_of_bag_outputs  : list dự đoán tương ứng
        """
        self._assert_fitted()
        if not self.bootstrap:
            raise RuntimeError(
                "Ước lượng OOB yêu cầu bootstrap=True khi khởi tạo rừng."
            )

        limit = self.n_estimators if num_trees is None else num_trees
        limit = max(1, min(limit, self.n_estimators))

        collected = {index: [] for index in range(len(self.training_samples))}
        for tree_position in range(limit):
            tree = self.trees[tree_position]
            indices = self.out_of_bag_indices[tree_position]
            if not indices:
                continue
            held_out = rfMath.select_by_indices(self.training_samples, indices)
            for index, prediction in zip(indices, self._predict_for_aggregation(tree, held_out)):
                collected[index].append(prediction)

        covered_indices = []
        out_of_bag_outputs = []
        for index in range(len(self.training_samples)):
            if collected[index]:
                covered_indices.append(index)
                out_of_bag_outputs.append(self._aggregate(collected[index]))
        return covered_indices, out_of_bag_outputs

    def _predict_for_aggregation(self, tree, samples):
        """Dự đoán của một cây ở dạng phù hợp để đưa vào _aggregate()."""
        return tree.predict(samples)

    def calculate_out_of_bag_error(self, error_function=None, num_trees=None):
        """
        Lỗi out-of-bag của rừng.

        Parameters:
            error_function : hàm (y_true, y_pred) -> float.
                             None → dùng hàm lỗi mặc định của lớp con.
            num_trees      : chỉ dùng num_trees cây đầu tiên

        Returns:
            float — None nếu không mẫu nào có dự đoán OOB
        """
        covered_indices, outputs = self.calculate_out_of_bag_predictions(num_trees)
        if not covered_indices:
            return None

        actual = rfMath.select_by_indices(self.training_targets, covered_indices)
        if error_function is None:
            error_function = self._default_error_function
        return error_function(actual, outputs)

    def calculate_out_of_bag_error_curve(self, tree_counts=None,
                                         error_function=None):
        """
        Đường cong lỗi OOB theo số cây — công cụ chọn n_estimators mà
        không tiêu tốn dữ liệu validate. Dừng tăng số cây tại điểm đường
        cong đi ngang.

        Parameters:
            tree_counts    : list số cây cần đo (None → lấy 10 mốc đều nhau)
            error_function : hàm (y_true, y_pred) -> float

        Returns:
            tree_counts : list số cây
            errors      : list lỗi OOB tương ứng
        """
        self._assert_fitted()
        if tree_counts is None:
            num_points = min(10, self.n_estimators)
            step = max(1, self.n_estimators // num_points)
            tree_counts = list(range(step, self.n_estimators + 1, step))
            if tree_counts[-1] != self.n_estimators:
                tree_counts.append(self.n_estimators)

        errors = [
            self.calculate_out_of_bag_error(error_function, num_trees=count)
            for count in tree_counts
        ]
        return tree_counts, errors

    def _default_error_function(self, actual, predicted):
        """Hàm lỗi mặc định dùng cho OOB — lớp con cài đặt."""
        raise NotImplementedError

    # ── Tầm quan trọng đặc trưng ────────────────────────────────────

    def calculate_feature_importances(self, normalize=True):
        """
        Mean Decrease in Impurity (MDI) — trung bình tầm quan trọng MDI
        của toàn bộ cây trong rừng.

        Lưu ý về thiên lệch: MDI ưu ái đặc trưng liên tục có nhiều giá
        trị phân biệt. Khi cần kết luận chắc chắn, hãy đối chiếu với
        calculate_permutation_importances().

        Parameters:
            normalize : True → chuẩn hóa tổng bằng 1

        Returns:
            list độ dài num_features
        """
        self._assert_fitted()
        totals = [0.0 for _ in range(self.num_features)]
        for tree in self.trees:
            tree_importances = tree.calculate_feature_importances(normalize=False)
            for index, value in enumerate(tree_importances):
                totals[index] += value

        averaged = [value / len(self.trees) for value in totals]
        if normalize:
            return rfMath.normalize_to_distribution(averaged)
        return averaged

    def calculate_permutation_importances(self, samples, targets,
                                          error_function=None, num_repeats=5,
                                          random_state=None):
        """
        Permutation Importance — mức tăng lỗi khi xáo trộn ngẫu nhiên
        một đặc trưng, đo trên tập dữ liệu chưa dùng để huấn luyện.

        Importance(j) = error(sau khi xáo trộn cột j) - error(gốc)

        Không bị thiên lệch theo số giá trị phân biệt như MDI, nhưng tốn
        chi phí tính toán hơn và có thể chia nhỏ công trạng giữa các đặc
        trưng tương quan mạnh.

        Parameters:
            samples        : list of lists — tập đánh giá (nên là validate)
            targets        : list giá trị mục tiêu tương ứng
            error_function : hàm (y_true, y_pred) -> float
            num_repeats    : số lần xáo trộn cho mỗi đặc trưng
            random_state   : hạt giống ngẫu nhiên

        Returns:
            list độ dài num_features — mức tăng lỗi trung bình
        """
        self._assert_fitted()
        if error_function is None:
            error_function = self._default_error_function

        baseline_error = error_function(targets, self.predict(samples))
        random_generator = random.Random(random_state)
        importances = []

        for feature_index in range(self.num_features):
            increases = []
            for _ in range(num_repeats):
                shuffled_column = [sample[feature_index] for sample in samples]
                random_generator.shuffle(shuffled_column)

                permuted = []
                for position, sample in enumerate(samples):
                    row = list(sample)
                    row[feature_index] = shuffled_column[position]
                    permuted.append(row)

                permuted_error = error_function(targets, self.predict(permuted))
                increases.append(permuted_error - baseline_error)

            importances.append(rfMath.calculate_mean(increases))

        return importances

    # ── Khảo sát rừng ───────────────────────────────────────────────

    def describe(self):
        """
        Tóm tắt cấu trúc rừng sau khi huấn luyện.

        Returns:
            dict { 'num_trees', 'num_features', 'max_features_resolved',
                   'average_depth', 'average_leaves', 'average_nodes',
                   'average_out_of_bag_ratio' }
        """
        self._assert_fitted()
        depths = [tree.get_depth() for tree in self.trees]
        leaves = [tree.count_leaves() for tree in self.trees]
        nodes = [tree.count_nodes() for tree in self.trees]

        num_samples = len(self.training_samples)
        if self.bootstrap and num_samples:
            ratios = [
                len(indices) / num_samples for indices in self.out_of_bag_indices
            ]
            average_out_of_bag_ratio = rfMath.calculate_mean(ratios)
        else:
            average_out_of_bag_ratio = 0.0

        return {
            'num_trees':                self.n_estimators,
            'num_features':             self.num_features,
            'max_features_resolved':    rfMath.resolve_max_features(
                                            self.num_features, self.max_features),
            'average_depth':            rfMath.calculate_mean(depths),
            'average_leaves':           rfMath.calculate_mean(leaves),
            'average_nodes':            rfMath.calculate_mean(nodes),
            'average_out_of_bag_ratio': average_out_of_bag_ratio,
        }

    def _assert_fitted(self):
        if not self.trees:
            raise RuntimeError("Rừng chưa được huấn luyện — hãy gọi fit() trước.")


# ---------------------------------------------------------------------
# ② Rừng phân loại — gộp MỀM: trung bình xác suất rồi lấy nhãn cao nhất
# ---------------------------------------------------------------------
class RandomForestClassifier(BaseRandomForest):
    """
    Rừng ngẫu nhiên cho bài toán phân loại.

    Tổng hợp bằng soft voting: trung bình các vector xác suất của từng
    cây rồi lấy nhãn có xác suất cao nhất. Cách này cho điểm số liên tục
    (dùng để vẽ ROC và điều chỉnh ngưỡng quyết định) và thường ổn định
    hơn bầu chọn cứng.

    Parameters: xem BaseRandomForest. Riêng:
        criterion    : 'gini' (mặc định) hoặc 'entropy'
        max_features : 'sqrt' — giá trị khuyến nghị cho phân loại
    """

    def __init__(self, n_estimators=100, criterion='gini', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, max_features='sqrt',
                 max_thresholds=None, bootstrap=True, random_state=None,
                 verbose=0):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            max_thresholds=max_thresholds,
            bootstrap=bootstrap,
            random_state=random_state,
            verbose=verbose,
        )
        self.label_space = []

    def _prepare_target_space(self, targets):
        self.label_space = sorted(set(targets))

    def _create_tree(self, tree_seed):
        return DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            max_thresholds=self.max_thresholds,
            random_state=tree_seed,
        )

    def _predict_for_aggregation(self, tree, samples):
        """Dùng vector xác suất để tổng hợp mềm, kể cả khi tính OOB."""
        return self._align_probabilities(tree, samples)

    def _align_probabilities(self, tree, samples):
        """
        Đưa vector xác suất của một cây về đúng thứ tự label_space của
        rừng — cần thiết vì mẫu bootstrap của một cây có thể thiếu hẳn
        một lớp.
        """
        tree_probabilities = tree.predict_probabilities(samples)
        if tree.label_space == self.label_space:
            return tree_probabilities

        position_of_label = {
            label: position for position, label in enumerate(tree.label_space)
        }
        aligned = []
        for vector in tree_probabilities:
            aligned.append([
                vector[position_of_label[label]] if label in position_of_label
                else 0.0
                for label in self.label_space
            ])
        return aligned

    def _aggregate(self, probability_vectors):
        averaged = rfMath.average_probability_vectors(probability_vectors)
        best_position = max(
            range(len(averaged)), key=lambda position: averaged[position]
        )
        return self.label_space[best_position]

    def predict(self, samples):
        """
        Dự đoán nhãn bằng soft voting trên toàn bộ cây.
        """
        self._assert_fitted()
        probabilities = self.predict_probabilities(samples)
        return [
            self.label_space[
                max(range(len(vector)), key=lambda position: vector[position])
            ]
            for vector in probabilities
        ]

    def predict_probabilities(self, samples):
        """
        Trung bình xác suất của toàn bộ cây.

        Returns:
            list of lists — vector xác suất theo thứ tự label_space
        """
        self._assert_fitted()
        per_tree = [self._align_probabilities(tree, samples) for tree in self.trees]
        return [
            rfMath.average_probability_vectors(
                [vectors[position] for vectors in per_tree]
            )
            for position in range(len(samples))
        ]

    def predict_scores(self, samples, positive_label=None):
        """
        Điểm số liên tục cho lớp dương — đầu vào của ROC-AUC và của việc
        dò ngưỡng quyết định.

        Parameters:
            samples        : list of lists
            positive_label : nhãn được coi là lớp dương.
                             None → lấy nhãn lớn nhất trong label_space.

        Returns:
            list xác suất thuộc lớp dương
        """
        self._assert_fitted()
        if positive_label is None:
            positive_label = self.label_space[-1]
        if positive_label not in self.label_space:
            raise ValueError(
                f"Nhãn '{positive_label}' không có trong không gian nhãn "
                f"{self.label_space}."
            )

        position = self.label_space.index(positive_label)
        return [vector[position] for vector in self.predict_probabilities(samples)]

    def _default_error_function(self, actual, predicted):
        """Tỷ lệ phân loại sai."""
        if not actual:
            return 0.0
        mismatches = sum(
            1 for true_label, predicted_label in zip(actual, predicted)
            if true_label != predicted_label
        )
        return mismatches / len(actual)


# ---------------------------------------------------------------------
# ③ Rừng hồi quy — gộp bằng trung bình cộng; hệ quả: KHÔNG ngoại suy được
# ---------------------------------------------------------------------
class RandomForestRegressor(BaseRandomForest):
    """
    Rừng ngẫu nhiên cho bài toán hồi quy.

    Tổng hợp bằng trung bình cộng dự đoán của các cây.

    Giới hạn cần nhớ: rừng hồi quy KHÔNG NGOẠI SUY được — mọi dự đoán
    luôn nằm trong khoảng giá trị mục tiêu đã thấy khi huấn luyện. Với
    dữ liệu có xu thế dài hạn, nên đặt mục tiêu ở dạng tỷ lệ thay vì
    giá trị tuyệt đối.

    Parameters: xem BaseRandomForest. Riêng:
        criterion    : 'variance' (tương đương squared_error)
        max_features : 'third' — giá trị khuyến nghị cho hồi quy
    """

    def __init__(self, n_estimators=100, criterion='variance', max_depth=None,
                 min_samples_split=2, min_samples_leaf=5,
                 min_impurity_decrease=0.0, max_features='third',
                 max_thresholds=None, bootstrap=True, random_state=None,
                 verbose=0):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            max_thresholds=max_thresholds,
            bootstrap=bootstrap,
            random_state=random_state,
            verbose=verbose,
        )

    def _create_tree(self, tree_seed):
        return DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            max_thresholds=self.max_thresholds,
            random_state=tree_seed,
        )

    def _aggregate(self, predictions):
        return rfMath.calculate_mean(predictions)

    def _default_error_function(self, actual, predicted):
        """Sai số bình phương trung bình."""
        if not actual:
            return 0.0
        return sum(
            (true_value - predicted_value) ** 2
            for true_value, predicted_value in zip(actual, predicted)
        ) / len(actual)
