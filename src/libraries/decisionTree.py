# =====================================================================
# Decision Tree — cây quyết định CART (Classification And Regression Tree)
# =====================================================================
# Cài đặt thuần Python, không phụ thuộc thư viện ngoài. Đây là khối xây
# dựng cơ bản của rừng ngẫu nhiên: mỗi cây học một luật phân tách nhị
# phân đệ quy trên không gian đặc trưng.
#
# Quy ước dữ liệu (xem thêm rfMath.py):
#   samples : list of lists — mỗi phần tử là một mẫu (dòng)
#   targets : list — nhãn (phân loại) hoặc giá trị thực (hồi quy)
#
# Thứ tự khai báo đi từ chi tiết nhỏ nhất ra tới mô hình hoàn chỉnh:
#
#   ① TreeNode               — một nút; cây chỉ là các nút nối với nhau
#   ② BaseDecisionTree       — thuật toán chia đệ quy, dùng chung
#   ③ DecisionTreeClassifier — điền vào ② phần "lá dự đoán nhãn nào"
#   ④ DecisionTreeRegressor  — điền vào ② phần "lá dự đoán giá trị nào"
#
# ③ và ④ chỉ khác nhau ở ba điểm: độ đo hỗn tạp, bộ tích luỹ quét
# ngưỡng và giá trị gán cho lá. Toàn bộ phần còn lại nằm ở ②.
# =====================================================================

import random

from . import rfMath


# ---------------------------------------------------------------------
# ① Nút — đơn vị nhỏ nhất; mọi thứ bên dưới chỉ là cách sinh ra nó
# ---------------------------------------------------------------------
class TreeNode:
    """
    Một nút trong cây quyết định.

    Nút trong  : có feature_index và threshold, rẽ nhánh sang left/right.
    Nút lá     : left = right = None, mang giá trị dự đoán trong value.

    Attributes:
        depth             : độ sâu của nút (gốc = 0)
        num_samples       : số mẫu huấn luyện rơi vào nút
        impurity          : độ hỗn tạp của nút trước khi chia
        value             : giá trị dự đoán nếu nút là lá
        probabilities     : phân phối xác suất nhãn (chỉ dùng khi phân loại)
        feature_index     : chỉ số đặc trưng dùng để chia
        threshold         : ngưỡng chia — mẫu có giá trị <= ngưỡng đi sang trái
        impurity_decrease : mức giảm độ hỗn tạp mà phép chia mang lại
        left, right       : hai nút con
    """

    def __init__(self, depth, num_samples, impurity):
        self.depth = depth
        self.num_samples = num_samples
        self.impurity = impurity

        self.value = None
        self.probabilities = None

        self.feature_index = None
        self.threshold = None
        self.impurity_decrease = 0.0
        self.left = None
        self.right = None

    @property
    def is_leaf(self):
        """True nếu nút không có nút con."""
        return self.left is None and self.right is None


# ---------------------------------------------------------------------
# ② Khung dựng cây — thuật toán chia đệ quy, không phụ thuộc kiểu bài toán
# ---------------------------------------------------------------------
class BaseDecisionTree:
    """
    Khung chung cho cây phân loại và cây hồi quy.

    Thuật toán xây cây (đệ quy, tham lam):
        1. Tính độ hỗn tạp của nút hiện tại.
        2. Nếu chạm điều kiện dừng → tạo lá.
        3. Chọn ngẫu nhiên m đặc trưng (nếu max_features < p).
        4. Duyệt mọi cặp (đặc trưng, ngưỡng) để tìm phép chia làm giảm
           độ hỗn tạp nhiều nhất.
        5. Chia mẫu thành hai nhánh và lặp lại cho từng nhánh.

    Parameters:
        criterion             : tiêu chí phân tách — xem rfMath.IMPURITY_FUNCTIONS
        max_depth             : độ sâu tối đa (None = không giới hạn)
        min_samples_split     : số mẫu tối thiểu để được phép chia một nút
        min_samples_leaf      : số mẫu tối thiểu phải còn lại ở mỗi nhánh con
        min_impurity_decrease : mức giảm hỗn tạp tối thiểu để chấp nhận phép chia
        max_features          : số đặc trưng xét mỗi nút — xem rfMath.resolve_max_features
        max_thresholds        : số ngưỡng tối đa xét trên mỗi đặc trưng (None = tất cả)
        random_state          : hạt giống ngẫu nhiên, cố định để tái lập kết quả
    """

    def __init__(self, criterion, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0,
                 max_features=None, max_thresholds=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.random_state = random_state

        self.root = None
        self.num_features = 0
        self.num_training_samples = 0
        self.impurity_function = rfMath.resolve_impurity_function(criterion)

    # ── Giao diện huấn luyện và dự đoán ─────────────────────────────

    def fit(self, samples, targets):
        """
        Huấn luyện cây trên tập mẫu.

        Parameters:
            samples : list of lists — mỗi phần tử là một mẫu
            targets : list giá trị mục tiêu, cùng độ dài với samples

        Returns:
            chính đối tượng cây (cho phép nối chuỗi lời gọi)
        """
        if len(samples) != len(targets):
            raise ValueError(
                f"Số mẫu ({len(samples)}) phải bằng số giá trị mục tiêu "
                f"({len(targets)})."
            )
        if not samples:
            raise ValueError("Không thể huấn luyện trên tập rỗng.")

        self.num_features = len(samples[0])
        self.num_training_samples = len(samples)
        self._prepare_target_space(targets)

        random_generator = random.Random(self.random_state)
        self.root = self._build_node(samples, targets, depth=0,
                                     random_generator=random_generator)
        return self

    def predict(self, samples):
        """
        Dự đoán cho một danh sách mẫu.

        Returns:
            list giá trị dự đoán, cùng thứ tự với samples
        """
        self._assert_fitted()
        return [self._traverse(sample).value for sample in samples]

    def predict_single(self, sample):
        """Dự đoán cho đúng một mẫu."""
        self._assert_fitted()
        return self._traverse(sample).value

    def apply(self, samples):
        """
        Trả về nút lá mà mỗi mẫu rơi vào.

        Dùng cho các thuật toán cần can thiệp trực tiếp vào giá trị của
        lá sau khi cây đã được dựng — ví dụ bước cập nhật Newton của
        gradient boosting.

        Returns:
            list các đối tượng TreeNode (nút lá)
        """
        self._assert_fitted()
        return [self._traverse(sample) for sample in samples]

    # ── Xây cây ─────────────────────────────────────────────────────

    def _build_node(self, samples, targets, depth, random_generator):
        """
        Dựng đệ quy một nút và toàn bộ cây con bên dưới nó.
        """
        impurity = self.impurity_function(targets)
        node = TreeNode(depth=depth, num_samples=len(samples), impurity=impurity)

        if self._should_stop(samples, targets, depth, impurity):
            self._assign_leaf_value(node, targets)
            return node

        best_split = self._find_best_split(
            samples, targets, impurity, random_generator
        )
        if best_split is None:
            self._assign_leaf_value(node, targets)
            return node

        feature_index, threshold, impurity_decrease = best_split
        if impurity_decrease < self.min_impurity_decrease:
            self._assign_leaf_value(node, targets)
            return node

        left_indices, right_indices = rfMath.partition_indices_by_threshold(
            samples, feature_index, threshold
        )
        if not left_indices or not right_indices:
            # Chốt chặn cuối: sai số dấu phẩy động có thể đẩy toàn bộ mẫu về
            # một phía. Khi đó phép chia vô nghĩa, tạo lá thay vì đệ quy tiếp.
            self._assign_leaf_value(node, targets)
            return node

        node.feature_index = feature_index
        node.threshold = threshold
        node.impurity_decrease = impurity_decrease
        node.left = self._build_node(
            rfMath.select_by_indices(samples, left_indices),
            rfMath.select_by_indices(targets, left_indices),
            depth + 1, random_generator
        )
        node.right = self._build_node(
            rfMath.select_by_indices(samples, right_indices),
            rfMath.select_by_indices(targets, right_indices),
            depth + 1, random_generator
        )
        return node

    def _should_stop(self, samples, targets, depth, impurity):
        """
        Kiểm tra các điều kiện dừng trước khi tìm phép chia.
        """
        if impurity <= 0.0:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if len(samples) < self.min_samples_split:
            return True
        if len(samples) < 2 * self.min_samples_leaf:
            return True
        return False

    def _find_best_split(self, samples, targets, parent_impurity,
                         random_generator):
        """
        Tìm phép chia làm giảm độ hỗn tạp nhiều nhất trong số các đặc
        trưng được chọn ngẫu nhiên.

        Với mỗi đặc trưng, thuật toán sắp xếp mẫu theo giá trị rồi QUÉT
        MỘT LƯỢT: tại mỗi biên giữa hai giá trị khác nhau, một mẫu được
        chuyển từ nhánh phải sang nhánh trái và thống kê được cập nhật
        trong O(1). Chi phí trên mỗi đặc trưng là O(n log n) thay vì
        O(n²) của cách duyệt thô từng ngưỡng.

        Parameters:
            samples          : list of lists — các mẫu tại nút
            targets          : list giá trị mục tiêu tại nút
            parent_impurity  : độ hỗn tạp của nút trước khi chia
            random_generator : đối tượng random.Random

        Returns:
            (feature_index, threshold, impurity_decrease) hoặc None nếu
            không tồn tại phép chia hợp lệ.
        """
        num_selected = rfMath.resolve_max_features(
            self.num_features, self.max_features
        )
        feature_indices = rfMath.select_random_feature_indices(
            self.num_features, num_selected, random_generator
        )

        num_samples = len(samples)
        best_feature_index = None
        best_threshold = None
        best_decrease = 0.0

        for feature_index in feature_indices:
            order = sorted(
                range(num_samples),
                key=lambda position: samples[position][feature_index]
            )
            sorted_values = [samples[position][feature_index] for position in order]
            if sorted_values[0] == sorted_values[-1]:
                continue

            sorted_targets = [targets[position] for position in order]
            accumulator = self._create_split_accumulator(sorted_targets)
            stride = self._resolve_boundary_stride(num_samples)

            for position in range(num_samples - 1):
                accumulator.move_to_left(sorted_targets[position])

                if sorted_values[position] == sorted_values[position + 1]:
                    continue
                if position + 1 < self.min_samples_leaf:
                    continue
                if num_samples - position - 1 < self.min_samples_leaf:
                    continue
                if stride > 1 and position % stride:
                    continue

                decrease = parent_impurity - accumulator.calculate_weighted_impurity()
                if decrease > best_decrease:
                    best_decrease = decrease
                    best_feature_index = feature_index
                    best_threshold = _midpoint_threshold(
                        sorted_values[position], sorted_values[position + 1]
                    )

        if best_feature_index is None:
            return None
        return best_feature_index, best_threshold, best_decrease

    def _resolve_boundary_stride(self, num_samples):
        """
        Bước nhảy giữa các biên được xét, suy ra từ max_thresholds.

        Cho phép đánh đổi độ chính xác của ngưỡng lấy tốc độ khi nút có
        rất nhiều mẫu. stride = 1 nghĩa là xét mọi biên.
        """
        if self.max_thresholds is None or self.max_thresholds <= 0:
            return 1
        if num_samples <= self.max_thresholds:
            return 1
        return max(1, num_samples // self.max_thresholds)

    def _create_split_accumulator(self, sorted_targets):
        """Tạo bộ tích lũy quét ngưỡng — lớp con phải cài đặt."""
        raise NotImplementedError

    def _traverse(self, sample):
        """Đi từ gốc xuống lá theo các luật phân tách."""
        node = self.root
        while not node.is_leaf:
            if sample[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    # ── Các điểm mở rộng cho lớp con ────────────────────────────────

    def _prepare_target_space(self, targets):
        """Chuẩn bị thông tin phụ thuộc kiểu bài toán (mặc định: không có)."""

    def _assign_leaf_value(self, node, targets):
        """Gán giá trị dự đoán cho một nút lá — lớp con phải cài đặt."""
        raise NotImplementedError

    # ── Khảo sát cấu trúc cây ───────────────────────────────────────

    def get_depth(self):
        """Độ sâu lớn nhất của cây (cây chỉ có gốc → 0)."""
        self._assert_fitted()
        return self._measure_depth(self.root)

    def _measure_depth(self, node):
        if node.is_leaf:
            return node.depth
        return max(self._measure_depth(node.left), self._measure_depth(node.right))

    def count_nodes(self):
        """Tổng số nút của cây."""
        self._assert_fitted()
        return self._count_nodes(self.root)

    def _count_nodes(self, node):
        if node.is_leaf:
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def count_leaves(self):
        """Số nút lá của cây."""
        self._assert_fitted()
        return self._count_leaves(self.root)

    def _count_leaves(self, node):
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    # ── Tầm quan trọng đặc trưng theo MDI ───────────────────────────

    def calculate_feature_importances(self, normalize=True):
        """
        Mean Decrease in Impurity — tổng mức giảm hỗn tạp mà mỗi đặc
        trưng mang lại, có trọng số theo số mẫu đi qua nút.

        Importance(j) = Σ_{nút t chia theo j} (n_t / n) · ΔI(t)

        Parameters:
            normalize : True → chuẩn hóa tổng bằng 1

        Returns:
            list độ dài num_features
        """
        self._assert_fitted()
        importances = [0.0 for _ in range(self.num_features)]
        self._accumulate_importances(self.root, importances)

        if normalize:
            return rfMath.normalize_to_distribution(importances)
        return importances

    def _accumulate_importances(self, node, importances):
        if node.is_leaf:
            return
        weight = node.num_samples / self.num_training_samples
        importances[node.feature_index] += weight * node.impurity_decrease
        self._accumulate_importances(node.left, importances)
        self._accumulate_importances(node.right, importances)

    # ── Xuất cây dạng văn bản ───────────────────────────────────────

    def export_text(self, feature_names=None, max_depth=None, decimals=4):
        """
        Xuất cấu trúc cây thành chuỗi nhiều dòng để đọc và kiểm tra.

        Parameters:
            feature_names : list tên đặc trưng (None → dùng "x[j]")
            max_depth     : chỉ in tới độ sâu này (None = in toàn bộ)
            decimals      : số chữ số thập phân của ngưỡng và giá trị

        Returns:
            str
        """
        self._assert_fitted()
        lines = []
        self._export_node(self.root, feature_names, max_depth, decimals,
                          '', '', lines)
        return '\n'.join(lines)

    def _export_node(self, node, feature_names, max_depth, decimals,
                     prefix, connector, lines):
        reached_limit = max_depth is not None and node.depth >= max_depth

        if node.is_leaf:
            description = (
                f"→ dự đoán = {self._format_value(node.value, decimals)}  "
                f"(n = {node.num_samples})"
            )
        elif reached_limit:
            description = f"… cắt bớt {self._count_nodes(node) - 1} nút"
        else:
            if feature_names is not None:
                name = feature_names[node.feature_index]
            else:
                name = f"x[{node.feature_index}]"
            description = (
                f"[{name} <= {round(node.threshold, decimals)}]  "
                f"(n = {node.num_samples})"
            )

        lines.append(f"{prefix}{connector}{description}")
        if node.is_leaf or reached_limit:
            return

        if connector:
            child_prefix = prefix + ('│   ' if connector.startswith('├') else '    ')
        else:
            child_prefix = prefix

        self._export_node(node.left, feature_names, max_depth, decimals,
                          child_prefix, '├─ đúng: ', lines)
        self._export_node(node.right, feature_names, max_depth, decimals,
                          child_prefix, '└─ sai : ', lines)

    def _format_value(self, value, decimals):
        if isinstance(value, float):
            return f"{round(value, decimals)}"
        return f"{value}"

    def _assert_fitted(self):
        if self.root is None:
            raise RuntimeError("Cây chưa được huấn luyện — hãy gọi fit() trước.")


# ---------------------------------------------------------------------
# Phép phụ dùng chung cho toàn module
# ---------------------------------------------------------------------
def _midpoint_threshold(lower_value, upper_value):
    """
    Ngưỡng nằm giữa hai giá trị phân biệt liền kề.

    Điểm giữa giúp ranh giới quyết định không dính vào một mẫu cụ thể.
    Nhưng khi hai giá trị quá sát nhau, điểm giữa có thể bị LÀM TRÒN
    thành đúng giá trị lớn hơn — lúc đó phép so sánh `<= ngưỡng` sẽ dồn
    cả hai nhóm về nhánh trái và làm nhánh phải rỗng. Trường hợp đó lùi
    về dùng chính giá trị nhỏ hơn làm ngưỡng.
    """
    midpoint = (lower_value + upper_value) / 2.0
    if midpoint >= upper_value:
        return lower_value
    return midpoint

# ---------------------------------------------------------------------
# ③ Cây phân loại — điền vào ② phần "lá dự đoán gì" cho nhãn rời rạc
# ---------------------------------------------------------------------
class DecisionTreeClassifier(BaseDecisionTree):
    """
    Cây quyết định cho bài toán phân loại.

    Nút lá dự đoán nhãn chiếm đa số và lưu kèm phân phối xác suất theo
    tần suất, phục vụ soft voting và vẽ đường ROC ở tầng rừng.

    Parameters: xem BaseDecisionTree. Riêng criterion mặc định là 'gini'.
    """

    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0,
                 max_features=None, max_thresholds=None, random_state=None):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            max_thresholds=max_thresholds,
            random_state=random_state,
        )
        self.label_space = []

    def _prepare_target_space(self, targets):
        """Ghi nhận không gian nhãn, sắp xếp để vector xác suất ổn định."""
        self.label_space = sorted(set(targets))

    def _create_split_accumulator(self, sorted_targets):
        return rfMath.LabelSplitAccumulator(sorted_targets, self.criterion)

    def _assign_leaf_value(self, node, targets):
        node.value = rfMath.majority_vote(targets)
        node.probabilities = rfMath.calculate_label_probabilities(
            targets, self.label_space
        )

    def predict_probabilities(self, samples):
        """
        Dự đoán phân phối xác suất nhãn cho từng mẫu.

        Returns:
            list of lists — mỗi phần tử là vector xác suất theo thứ tự
            của thuộc tính label_space
        """
        self._assert_fitted()
        return [list(self._traverse(sample).probabilities) for sample in samples]


# ---------------------------------------------------------------------
# ④ Cây hồi quy — cùng khung ②, nhưng lá trả về trung bình giá trị
# ---------------------------------------------------------------------
class DecisionTreeRegressor(BaseDecisionTree):
    """
    Cây quyết định cho bài toán hồi quy.

    Nút lá dự đoán trung bình cộng của các giá trị mục tiêu rơi vào lá —
    giá trị cực tiểu hóa tổng bình phương sai số trong nút.

    Hệ quả quan trọng: cây hồi quy KHÔNG NGOẠI SUY được. Mọi dự đoán
    luôn nằm trong khoảng giá trị mục tiêu đã thấy khi huấn luyện.

    Parameters: xem BaseDecisionTree. Riêng criterion mặc định là
    'variance' (tương đương squared_error).
    """

    def __init__(self, criterion='variance', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0,
                 max_features=None, max_thresholds=None, random_state=None):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            max_thresholds=max_thresholds,
            random_state=random_state,
        )

    def _create_split_accumulator(self, sorted_targets):
        return rfMath.ValueSplitAccumulator(sorted_targets)

    def _assign_leaf_value(self, node, targets):
        node.value = rfMath.calculate_mean(targets)
