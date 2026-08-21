# =====================================================================
# RF Math — nền tảng toán học cho cây quyết định và rừng ngẫu nhiên
# =====================================================================
# Module thuần toán học: độ đo hỗn tạp (impurity), độ lợi của phép chia,
# lấy mẫu bootstrap, ngẫu nhiên hoá đặc trưng và tổng hợp dự đoán.
# Không phụ thuộc vào bài toán, dữ liệu hay thư viện bên ngoài.
#
# Quy ước cấu trúc dữ liệu dùng chung cho toàn tầng thuật toán:
#   samples : list of lists — mỗi phần tử là MỘT MẪU (một dòng)
#   targets : list — giá trị mục tiêu, cùng độ dài với samples
#   columns : list of lists — mỗi phần tử là MỘT ĐẶC TRƯNG (một cột)
#
# Thứ tự khai báo bám đúng chuỗi dựng một cây rồi gộp thành rừng:
#
#   ①–④  Đổi hình dạng dữ liệu     — cột ↔ mẫu, lấy phần tử theo chỉ số
#   ⑤–⑧  Thống kê cơ bản           — nền cho mọi độ đo bên dưới
#   ⑨–⑫  Độ hỗn tạp của MỘT nút    — cần ⑤–⑧
#   ⑬–⑯  Độ lợi của MỘT phép chia  — cần ⑨–⑫
#   ⑰–⑳  Tìm ngưỡng chia           — sinh ứng viên rồi quét một lượt
#   ㉑–㉒  Bootstrap                 — trụ cột ngẫu nhiên thứ NHẤT
#   ㉓–㉔  Ngẫu nhiên hoá đặc trưng  — trụ cột ngẫu nhiên thứ HAI
#   ㉕–㉘  Tổng hợp dự đoán của rừng — bước cuối, cần ⑤ và ⑥
#   ㉙–㉛  Hàm số phụ trợ cho boosting
# =====================================================================

import math


# ---------------------------------------------------------------------
# ① Chuyển vị — phép đổi hình dạng gốc, ② và ③ chỉ là tên gọi ngữ nghĩa
# ---------------------------------------------------------------------
def transpose_matrix(matrix):
    """
    Chuyển vị ma trận dạng list of lists.

    Dùng để đổi qua lại giữa biểu diễn theo cột (đầu ra của tầng nạp dữ
    liệu) và biểu diễn theo dòng (đầu vào của cây quyết định).

    Parameters:
        matrix : list of lists — các dòng có độ dài bằng nhau

    Returns:
        list of lists đã chuyển vị
    """
    if not matrix:
        return []

    num_inner = len(matrix[0])
    for row in matrix:
        if len(row) != num_inner:
            raise ValueError("Các dòng của ma trận phải có độ dài bằng nhau.")

    return [[row[index] for row in matrix] for index in range(num_inner)]


# ---------------------------------------------------------------------
# ② Cột → mẫu — hướng dùng nhiều nhất: nạp dữ liệu xong là gọi hàm này
# ---------------------------------------------------------------------
def columns_to_samples(columns):
    """
    Đổi biểu diễn theo cột thành biểu diễn theo mẫu (dòng).

    Parameters:
        columns : list of lists — mỗi phần tử là một cột đặc trưng

    Returns:
        samples : list of lists — mỗi phần tử là một mẫu
    """
    return transpose_matrix(columns)


# ---------------------------------------------------------------------
# ③ Mẫu → cột — hướng ngược lại, cần khi muốn thao tác trên từng đặc trưng
# ---------------------------------------------------------------------
def samples_to_columns(samples):
    """
    Đổi biểu diễn theo mẫu (dòng) thành biểu diễn theo cột.

    Parameters:
        samples : list of lists — mỗi phần tử là một mẫu

    Returns:
        columns : list of lists — mỗi phần tử là một cột đặc trưng
    """
    return transpose_matrix(samples)


# ---------------------------------------------------------------------
# ④ Lấy theo chỉ số — mọi phép chia nhánh và bootstrap đều đi qua đây
# ---------------------------------------------------------------------
def select_by_indices(sequence, indices):
    """
    Lấy ra các phần tử của một dãy theo danh sách chỉ số.

    Parameters:
        sequence : list nguồn
        indices  : list chỉ số cần lấy

    Returns:
        list các phần tử tương ứng, giữ đúng thứ tự của indices
    """
    return [sequence[index] for index in indices]


# ---------------------------------------------------------------------
# ⑤ Đếm tần suất nhãn — nguyên liệu của mọi độ đo hỗn tạp phân loại
# ---------------------------------------------------------------------
def count_label_frequencies(labels):
    """
    Đếm số lần xuất hiện của từng nhãn.

    Parameters:
        labels : list nhãn (giá trị rời rạc, có thể hash)

    Returns:
        dict { nhãn: số lần xuất hiện }
    """
    frequencies = {}
    for label in labels:
        frequencies[label] = frequencies.get(label, 0) + 1
    return frequencies


# ---------------------------------------------------------------------
# ⑥ Trung bình — giá trị dự đoán của lá hồi quy, và là nền của ⑦
# ---------------------------------------------------------------------
def calculate_mean(values):
    """
    Trung bình cộng.
    mean = (1/n) * Σ x_i
    """
    n = len(values)
    if n == 0:
        raise ValueError("Không thể tính trung bình trên tập rỗng.")
    return sum(values) / n


# ---------------------------------------------------------------------
# ⑦ Phương sai — vừa là thống kê, vừa là ĐỘ HỖN TẠP của nút hồi quy
# ---------------------------------------------------------------------
def calculate_variance(values):
    """
    Phương sai tổng thể (chia cho n, không phải n-1).
    Var = (1/n) * Σ (x_i - x̄)²

    Đây chính là độ hỗn tạp dùng cho nút của cây hồi quy: giảm phương
    sai tương đương giảm MSE nội bộ nút.
    """
    n = len(values)
    if n == 0:
        return 0.0
    mean_value = sum(values) / n
    return sum((value - mean_value) ** 2 for value in values) / n


# ---------------------------------------------------------------------
# ⑧ Độ lệch chuẩn — đưa ⑦ về cùng đơn vị với biến gốc, tiện diễn giải
# ---------------------------------------------------------------------
def calculate_standard_deviation(values):
    """
    Độ lệch chuẩn tổng thể: σ = √Var
    """
    return calculate_variance(values) ** 0.5


# ---------------------------------------------------------------------
# ⑨ Gini — độ hỗn tạp mặc định của cây phân loại, rẻ hơn Entropy
# ---------------------------------------------------------------------
def calculate_gini_impurity(labels):
    """
    Chỉ số Gini — xác suất phân loại sai khi gán nhãn ngẫu nhiên theo
    phân phối của nút.

    Gini(t) = 1 - Σ p_k²      với p_k = tỷ lệ mẫu thuộc lớp k

    Giá trị nằm trong [0, 1 - 1/K]:
      - 0    : nút thuần khiết (chỉ một lớp)
      - lớn  : các lớp phân bố càng đều

    Returns:
        float — 0.0 nếu nút rỗng
    """
    n = len(labels)
    if n == 0:
        return 0.0

    frequencies = count_label_frequencies(labels)
    return 1.0 - sum((count / n) ** 2 for count in frequencies.values())


# ---------------------------------------------------------------------
# ⑩ Entropy — cùng vai trò với ⑨ nhưng phạt nặng hơn khi lớp phân tán
# ---------------------------------------------------------------------
def calculate_entropy(labels):
    """
    Entropy Shannon — lượng thông tin trung bình cần để mô tả nhãn.

    H(t) = - Σ p_k * log₂(p_k)

    Giá trị nằm trong [0, log₂K]:
      - 0      : nút thuần khiết
      - log₂K  : K lớp phân bố hoàn toàn đều

    Returns:
        float — 0.0 nếu nút rỗng
    """
    n = len(labels)
    if n == 0:
        return 0.0

    frequencies = count_label_frequencies(labels)
    entropy = 0.0
    for count in frequencies.values():
        proportion = count / n
        if proportion > 0:
            entropy -= proportion * math.log2(proportion)
    return entropy


# ---------------------------------------------------------------------
# ⑪ Tỷ lệ sai — độ hỗn tạp thô nhất, ít nhạy nên chỉ dùng khi cắt tỉa
# ---------------------------------------------------------------------
def calculate_misclassification_rate(labels):
    """
    Tỷ lệ phân loại sai của nút nếu dự đoán bằng lớp chiếm đa số.

    Error(t) = 1 - max(p_k)

    Ít nhạy hơn Gini và Entropy nên thường chỉ dùng khi cắt tỉa cây.
    """
    n = len(labels)
    if n == 0:
        return 0.0

    frequencies = count_label_frequencies(labels)
    return 1.0 - max(frequencies.values()) / n


# ---------------------------------------------------------------------
# ⑫ Bảng tra — gom ⑦, ⑨, ⑩, ⑪ về một cửa để cây chọn bằng tên chuỗi
# ---------------------------------------------------------------------
IMPURITY_FUNCTIONS = {
    'gini':              calculate_gini_impurity,
    'entropy':           calculate_entropy,
    'misclassification': calculate_misclassification_rate,
    'variance':          calculate_variance,
    'squared_error':     calculate_variance,
}


def resolve_impurity_function(criterion):
    """
    Tra hàm độ hỗn tạp theo tên tiêu chí phân tách.

    Parameters:
        criterion : 'gini' | 'entropy' | 'misclassification'
                    | 'variance' | 'squared_error'

    Returns:
        callable(labels) -> float
    """
    if criterion not in IMPURITY_FUNCTIONS:
        raise ValueError(
            f"Tiêu chí '{criterion}' không hợp lệ. "
            f"Các lựa chọn: {sorted(IMPURITY_FUNCTIONS)}"
        )
    return IMPURITY_FUNCTIONS[criterion]


# ---------------------------------------------------------------------
# ⑬ Hỗn tạp sau khi chia — gộp hai nút con lại bằng trọng số số mẫu
# ---------------------------------------------------------------------
def calculate_weighted_impurity(left_targets, right_targets, impurity_function):
    """
    Độ hỗn tạp bình quân có trọng số của hai nút con sau khi chia.

    I_split = (n_L/n) * I(left) + (n_R/n) * I(right)

    Parameters:
        left_targets      : list giá trị mục tiêu của nhánh trái
        right_targets     : list giá trị mục tiêu của nhánh phải
        impurity_function : hàm tính độ hỗn tạp cho một nút

    Returns:
        float
    """
    num_left = len(left_targets)
    num_right = len(right_targets)
    total = num_left + num_right
    if total == 0:
        return 0.0

    return (num_left / total) * impurity_function(left_targets) + \
           (num_right / total) * impurity_function(right_targets)


# ---------------------------------------------------------------------
# ⑭ Mức giảm hỗn tạp — hiệu của nút cha (⑨–⑫) và nút con (⑬); ĐÍCH tối ưu
# ---------------------------------------------------------------------
def calculate_impurity_decrease(parent_targets, left_targets, right_targets,
                                impurity_function):
    """
    Mức giảm độ hỗn tạp của một phép phân tách — đại lượng mà cây tìm
    cách cực đại hoá tại mỗi nút.

    ΔI = I(parent) - [ (n_L/n)·I(left) + (n_R/n)·I(right) ]

    Với criterion='gini'/'entropy' đây là Information Gain;
    với criterion='variance' đây là Variance Reduction.

    Returns:
        float — luôn ≥ 0 với các độ đo lồi (Gini, Entropy, Variance)
    """
    parent_impurity = impurity_function(parent_targets)
    child_impurity = calculate_weighted_impurity(
        left_targets, right_targets, impurity_function
    )
    return parent_impurity - child_impurity


# ---------------------------------------------------------------------
# ⑮ Thông tin nội tại của phép chia — mẫu số chuẩn hoá cho ⑯
# ---------------------------------------------------------------------
def calculate_split_information(left_targets, right_targets):
    """
    Thông tin nội tại của phép chia (Split Information) — dùng để chuẩn
    hoá Information Gain thành Gain Ratio, hạn chế thiên lệch về phía
    đặc trưng có nhiều giá trị.

    SplitInfo = - Σ (n_i/n) * log₂(n_i/n)
    """
    num_left = len(left_targets)
    num_right = len(right_targets)
    total = num_left + num_right
    if total == 0:
        return 0.0

    split_information = 0.0
    for count in (num_left, num_right):
        if count > 0:
            proportion = count / total
            split_information -= proportion * math.log2(proportion)
    return split_information


# ---------------------------------------------------------------------
# ⑯ Gain Ratio — ⑭ chia cho ⑮, dùng khi muốn phạt phép chia quá lệch
# ---------------------------------------------------------------------
def calculate_gain_ratio(parent_targets, left_targets, right_targets,
                         impurity_function):
    """
    Tỷ số độ lợi (Gain Ratio) — Information Gain chuẩn hoá theo
    SplitInfo.

    GainRatio = ΔI / SplitInfo

    Returns:
        float — 0.0 khi SplitInfo = 0 (một nhánh rỗng)
    """
    split_information = calculate_split_information(left_targets, right_targets)
    if split_information == 0:
        return 0.0

    impurity_decrease = calculate_impurity_decrease(
        parent_targets, left_targets, right_targets, impurity_function
    )
    return impurity_decrease / split_information


# ---------------------------------------------------------------------
# ⑰ Ngưỡng ứng viên — điểm giữa các giá trị phân biệt, không dính vào mẫu
# ---------------------------------------------------------------------
def find_candidate_thresholds(values, max_thresholds=None):
    """
    Sinh danh sách ngưỡng ứng viên cho một đặc trưng liên tục: điểm
    giữa của các cặp giá trị khác nhau liên tiếp sau khi sắp xếp.

    Lấy điểm giữa thay vì lấy chính giá trị quan sát giúp ranh giới
    quyết định không dính vào một mẫu cụ thể, tổng quát hoá tốt hơn.

    Parameters:
        values         : list giá trị của một đặc trưng
        max_thresholds : giới hạn số ngưỡng xét (lấy mẫu đều trên danh
                         sách đã sắp xếp). None = xét tất cả.

    Returns:
        list ngưỡng tăng dần — rỗng nếu đặc trưng là hằng số
    """
    unique_values = sorted(set(values))
    if len(unique_values) < 2:
        return []

    thresholds = [
        (unique_values[index] + unique_values[index + 1]) / 2.0
        for index in range(len(unique_values) - 1)
    ]

    if max_thresholds is not None and len(thresholds) > max_thresholds > 0:
        step = len(thresholds) / max_thresholds
        thresholds = [
            thresholds[int(index * step)] for index in range(max_thresholds)
        ]

    return thresholds


# ---------------------------------------------------------------------
# ⑱ Chia chỉ số theo ngưỡng — biến một ngưỡng của ⑰ thành hai nhánh
# ---------------------------------------------------------------------
def partition_indices_by_threshold(samples, feature_index, threshold):
    """
    Chia chỉ số mẫu thành hai nhánh theo một ngưỡng.

    Quy ước: giá trị <= threshold đi về nhánh TRÁI.

    Parameters:
        samples       : list of lists — các mẫu
        feature_index : chỉ số đặc trưng dùng để chia
        threshold     : ngưỡng chia

    Returns:
        left_indices, right_indices : hai list chỉ số
    """
    left_indices = []
    right_indices = []
    for index, sample in enumerate(samples):
        if sample[feature_index] <= threshold:
            left_indices.append(index)
        else:
            right_indices.append(index)
    return left_indices, right_indices


# ---------------------------------------------------------------------
# ⑲ Bộ tích luỹ nhãn — thay ⑬ khi quét ngưỡng: cập nhật O(1) mỗi bước
# ---------------------------------------------------------------------
class LabelSplitAccumulator:
    """
    Bộ tích luỹ tần suất nhãn — dùng cho cây phân loại.

    Duyệt thô mọi cặp (đặc trưng, ngưỡng) tốn O(n) cho MỖI ngưỡng, tổng
    cộng O(n²) trên một đặc trưng. Lớp này cho phép quét MỘT LƯỢT trên
    dãy đã sắp xếp: mỗi bước chỉ chuyển một mẫu từ nhánh phải sang nhánh
    trái, đưa chi phí về O(n log n) — chi phí sắp xếp.

    Quy ước: khởi tạo với TOÀN BỘ mẫu nằm ở nhánh phải.

    Parameters:
        targets   : list nhãn đã sắp xếp theo giá trị đặc trưng
        criterion : 'gini' | 'entropy' | 'misclassification'
    """

    def __init__(self, targets, criterion='gini'):
        if criterion not in ('gini', 'entropy', 'misclassification'):
            raise ValueError(
                f"Tiêu chí '{criterion}' không dùng được cho bài toán phân loại."
            )
        self.criterion = criterion
        self.total = len(targets)
        self.left_counts = {}
        self.right_counts = count_label_frequencies(targets)
        self.num_left = 0
        self.num_right = self.total

    def move_to_left(self, label):
        """Chuyển một mẫu từ nhánh phải sang nhánh trái."""
        self.left_counts[label] = self.left_counts.get(label, 0) + 1
        self.right_counts[label] -= 1
        if self.right_counts[label] == 0:
            del self.right_counts[label]
        self.num_left += 1
        self.num_right -= 1

    def _calculate_impurity(self, counts, total):
        """Độ hỗn tạp của một nhánh, tính thẳng từ bảng tần suất."""
        if total == 0:
            return 0.0
        if self.criterion == 'gini':
            return 1.0 - sum((count / total) ** 2 for count in counts.values())
        if self.criterion == 'entropy':
            entropy = 0.0
            for count in counts.values():
                proportion = count / total
                if proportion > 0:
                    entropy -= proportion * math.log2(proportion)
            return entropy
        return 1.0 - max(counts.values()) / total

    def calculate_weighted_impurity(self):
        """Độ hỗn tạp bình quân có trọng số của hai nhánh hiện tại."""
        if self.total == 0:
            return 0.0
        left = self._calculate_impurity(self.left_counts, self.num_left)
        right = self._calculate_impurity(self.right_counts, self.num_right)
        return (self.num_left / self.total) * left + \
               (self.num_right / self.total) * right


# ---------------------------------------------------------------------
# ⑳ Bộ tích luỹ giá trị — bản đối ứng của ⑲ cho cây hồi quy
# ---------------------------------------------------------------------
class ValueSplitAccumulator:
    """
    Bộ tích luỹ tổng và tổng bình phương — dùng cho cây hồi quy.

    Phương sai được tính theo dạng khai triển để cập nhật trong O(1):

        Var = E[x²] - (E[x])² = (Σx²)/n - ((Σx)/n)²

    Quy ước: khởi tạo với TOÀN BỘ mẫu nằm ở nhánh phải.

    Parameters:
        targets : list giá trị đã sắp xếp theo giá trị đặc trưng
    """

    def __init__(self, targets):
        self.total = len(targets)
        self.num_left = 0
        self.sum_left = 0.0
        self.sum_square_left = 0.0
        self.num_right = self.total
        self.sum_right = float(sum(targets))
        self.sum_square_right = float(sum(value * value for value in targets))

    def move_to_left(self, value):
        """Chuyển một mẫu từ nhánh phải sang nhánh trái."""
        self.num_left += 1
        self.sum_left += value
        self.sum_square_left += value * value
        self.num_right -= 1
        self.sum_right -= value
        self.sum_square_right -= value * value

    @staticmethod
    def _calculate_variance(count, total_sum, total_square_sum):
        """Phương sai của một nhánh theo dạng khai triển E[x²] - (E[x])²."""
        if count == 0:
            return 0.0
        mean_value = total_sum / count
        variance = total_square_sum / count - mean_value * mean_value
        return variance if variance > 0.0 else 0.0

    def calculate_weighted_impurity(self):
        """Phương sai bình quân có trọng số của hai nhánh hiện tại."""
        if self.total == 0:
            return 0.0
        left = self._calculate_variance(
            self.num_left, self.sum_left, self.sum_square_left
        )
        right = self._calculate_variance(
            self.num_right, self.sum_right, self.sum_square_right
        )
        return (self.num_left / self.total) * left + \
               (self.num_right / self.total) * right


# ---------------------------------------------------------------------
# ㉑ Bootstrap — trụ cột ngẫu nhiên THỨ NHẤT: mỗi cây một tập mẫu khác nhau
# ---------------------------------------------------------------------
def generate_bootstrap_indices(num_samples, random_generator):
    """
    Sinh một mẫu bootstrap: rút ngẫu nhiên CÓ HOÀN LẠI đúng n chỉ số.

    Xác suất một mẫu KHÔNG được chọn lần nào:
        (1 - 1/n)^n  →  1/e ≈ 0.368  khi n lớn

    Nghĩa là khoảng 36.8% số mẫu nằm ngoài túi (out-of-bag) và có thể
    dùng làm tập kiểm định miễn phí cho cây đó.

    Parameters:
        num_samples      : số mẫu của tập huấn luyện
        random_generator : đối tượng random.Random đã cố định hạt giống

    Returns:
        in_bag_indices     : list chỉ số được chọn (có lặp), độ dài n
        out_of_bag_indices : list chỉ số không được chọn lần nào
    """
    if num_samples <= 0:
        raise ValueError("Số mẫu phải là số nguyên dương.")

    in_bag_indices = [
        random_generator.randrange(num_samples) for _ in range(num_samples)
    ]
    selected = set(in_bag_indices)
    out_of_bag_indices = [
        index for index in range(num_samples) if index not in selected
    ]
    return in_bag_indices, out_of_bag_indices


# ---------------------------------------------------------------------
# ㉒ Tỷ lệ OOB lý thuyết — mốc để kiểm chứng cài đặt ㉑ chạy đúng
# ---------------------------------------------------------------------
def calculate_expected_out_of_bag_ratio(num_samples):
    """
    Tỷ lệ kỳ vọng của mẫu out-of-bag: (1 - 1/n)^n

    Dùng để kiểm chứng cài đặt bootstrap — giá trị phải tiến về
    1/e ≈ 0.3679 khi n tăng.
    """
    if num_samples <= 0:
        raise ValueError("Số mẫu phải là số nguyên dương.")
    return (1.0 - 1.0 / num_samples) ** num_samples


# ---------------------------------------------------------------------
# ㉓ Quy đổi max_features — biến quy ước chuỗi thành số nguyên m cụ thể
# ---------------------------------------------------------------------
def resolve_max_features(num_features, max_features):
    """
    Quy đổi tham số max_features về một số nguyên m — số đặc trưng được
    xét tại mỗi nút.

    m là siêu tham số quan trọng nhất của rừng ngẫu nhiên: m nhỏ làm
    giảm tương quan giữa các cây (tốt cho việc tổng hợp) nhưng cũng làm
    giảm sức mạnh của từng cây.

    Parameters:
        num_features : tổng số đặc trưng p
        max_features : 'sqrt'  → ⌊√p⌋   (mặc định cho phân loại)
                       'log2'  → ⌊log₂p⌋
                       'third' → ⌊p/3⌋  (mặc định cho hồi quy)
                       'all' / None → p
                       int   → dùng trực tiếp
                       float → tỷ lệ trong khoảng (0, 1]

    Returns:
        int trong đoạn [1, num_features]
    """
    if num_features <= 0:
        raise ValueError("Số đặc trưng phải là số nguyên dương.")

    if max_features is None or max_features == 'all':
        resolved = num_features
    elif max_features == 'sqrt':
        resolved = int(math.sqrt(num_features))
    elif max_features == 'log2':
        resolved = int(math.log2(num_features))
    elif max_features == 'third':
        resolved = int(num_features / 3)
    elif isinstance(max_features, bool):
        raise TypeError("max_features không nhận giá trị kiểu bool.")
    elif isinstance(max_features, int):
        resolved = max_features
    elif isinstance(max_features, float):
        if not 0.0 < max_features <= 1.0:
            raise ValueError("max_features dạng float phải nằm trong (0, 1].")
        resolved = int(max_features * num_features)
    else:
        raise ValueError(
            f"max_features='{max_features}' không hợp lệ. "
            f"Dùng 'sqrt', 'log2', 'third', 'all', số nguyên hoặc tỷ lệ."
        )

    return max(1, min(resolved, num_features))


# ---------------------------------------------------------------------
# ㉔ Chọn đặc trưng — trụ cột ngẫu nhiên THỨ HAI, áp dụng m từ ㉓
# ---------------------------------------------------------------------
def select_random_feature_indices(num_features, num_selected, random_generator):
    """
    Chọn ngẫu nhiên KHÔNG hoàn lại một tập con đặc trưng để xét tại một
    nút — trụ cột thứ hai của rừng ngẫu nhiên, bên cạnh bootstrap.

    Parameters:
        num_features     : tổng số đặc trưng
        num_selected     : số đặc trưng cần chọn (đã qua resolve_max_features)
        random_generator : đối tượng random.Random

    Returns:
        list chỉ số đặc trưng đã sắp xếp tăng dần
    """
    num_selected = max(1, min(num_selected, num_features))
    return sorted(
        random_generator.sample(range(num_features), num_selected)
    )


# ---------------------------------------------------------------------
# ㉕ Bầu chọn đa số — quy tắc gộp CỨNG của rừng phân loại, dựa trên ⑤
# ---------------------------------------------------------------------
def majority_vote(labels):
    """
    Bầu chọn đa số — quy tắc tổng hợp của rừng cho bài toán phân loại.

    Khi có nhiều nhãn cùng số phiếu cao nhất, chọn nhãn nhỏ nhất theo
    thứ tự sắp xếp để kết quả có tính tất định (tái lập được).

    Parameters:
        labels : list nhãn do từng cây dự đoán

    Returns:
        nhãn thắng cuộc
    """
    if not labels:
        raise ValueError("Không thể bầu chọn trên danh sách rỗng.")

    frequencies = count_label_frequencies(labels)
    highest_count = max(frequencies.values())
    winners = [
        label for label, count in frequencies.items() if count == highest_count
    ]
    return sorted(winners)[0]


# ---------------------------------------------------------------------
# ㉖ Xác suất nhãn tại lá — nguyên liệu cho cách gộp MỀM ở ㉗
# ---------------------------------------------------------------------
def calculate_label_probabilities(labels, label_space):
    """
    Phân phối xác suất theo tần suất nhãn trên một không gian nhãn cho
    trước.

    P(k) = số lần xuất hiện của nhãn k / tổng số nhãn

    Parameters:
        labels      : list nhãn quan sát được
        label_space : list toàn bộ nhãn có thể có, quyết định thứ tự
                      của vector xác suất trả về

    Returns:
        list xác suất, cùng thứ tự với label_space, tổng bằng 1
    """
    total = len(labels)
    if total == 0:
        return [0.0 for _ in label_space]

    frequencies = count_label_frequencies(labels)
    return [frequencies.get(label, 0) / total for label in label_space]


# ---------------------------------------------------------------------
# ㉗ Gộp mềm — trung bình các vector ㉖, cho điểm liên tục để vẽ ROC
# ---------------------------------------------------------------------
def average_probability_vectors(probability_vectors):
    """
    Trung bình cộng nhiều vector xác suất — cách tổng hợp mềm (soft
    voting) của rừng, cho kết quả mượt hơn bầu chọn cứng và cung cấp
    điểm số liên tục để vẽ đường ROC.

    Parameters:
        probability_vectors : list các vector xác suất cùng độ dài

    Returns:
        list xác suất trung bình
    """
    if not probability_vectors:
        raise ValueError("Không thể lấy trung bình trên danh sách rỗng.")

    num_classes = len(probability_vectors[0])
    for vector in probability_vectors:
        if len(vector) != num_classes:
            raise ValueError("Các vector xác suất phải có cùng độ dài.")

    num_vectors = len(probability_vectors)
    return [
        sum(vector[index] for vector in probability_vectors) / num_vectors
        for index in range(num_classes)
    ]


# ---------------------------------------------------------------------
# ㉘ Chuẩn hoá về phân phối — dùng cho vector tầm quan trọng đặc trưng
# ---------------------------------------------------------------------
def normalize_to_distribution(values):
    """
    Chuẩn hoá một vector không âm thành phân phối xác suất (tổng bằng 1).

    Nếu tổng bằng 0, trả về phân phối đều.
    """
    total = sum(values)
    if total <= 0:
        num_values = len(values)
        if num_values == 0:
            return []
        return [1.0 / num_values for _ in values]
    return [value / total for value in values]


# ---------------------------------------------------------------------
# ㉙ Sigmoid — đưa điểm số thô của boosting về xác suất trong (0, 1)
# ---------------------------------------------------------------------
def sigmoid(value):
    """
    Hàm logistic: σ(z) = 1 / (1 + e^(-z))

    Cài đặt ổn định số học cho cả z âm rất lớn lẫn z dương rất lớn,
    tránh tràn số khi tính e^(-z).
    """
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


# ---------------------------------------------------------------------
# ㉚ Logit — hàm ngược của ㉙, dùng để đặt điểm khởi tạo F₀ của boosting
# ---------------------------------------------------------------------
def logit(probability, epsilon=1e-15):
    """
    Hàm ngược của sigmoid: log(p / (1 - p))

    Parameters:
        probability : xác suất trong khoảng (0, 1)
        epsilon     : biên kẹp để tránh log(0)
    """
    clipped = min(max(probability, epsilon), 1.0 - epsilon)
    return math.log(clipped / (1.0 - clipped))


# ---------------------------------------------------------------------
# ㉛ Kẹp giá trị — chặn bước Newton quá lớn khi mẫu số gần 0
# ---------------------------------------------------------------------
def clip_value(value, lower_bound, upper_bound):
    """
    Kẹp một giá trị vào đoạn [lower_bound, upper_bound].
    """
    return max(lower_bound, min(value, upper_bound))
