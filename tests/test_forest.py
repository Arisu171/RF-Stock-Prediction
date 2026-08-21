# =====================================================================
# Test Forest — kiểm thử tầng lõi thuật toán
# =====================================================================
# Kiểm tra ba nhóm tính chất:
#
#   1. ĐÚNG TOÁN HỌC   — Gini, Entropy, bootstrap khớp công thức lý thuyết
#   2. ĐÚNG HÀNH VI    — cây học được quy luật, rừng ổn định hơn cây đơn
#   3. TÁI LẬP ĐƯỢC    — cùng hạt giống phải cho cùng kết quả
#
# Nhóm 3 dễ bị xem nhẹ nhưng là điều kiện để mọi con số trong báo cáo có
# thể kiểm chứng lại được.
#
# Thứ tự khai báo:
#
#   ①  Dữ liệu mẫu
#   ②  Độ đo hỗn tạp và độ lợi
#   ③  Bootstrap và ngẫu nhiên hoá đặc trưng
#   ④  Cây quyết định
#   ⑤  Rừng ngẫu nhiên
#   ⑥  Gradient boosting
#   ⑦  Tách tập và gán nhãn
# =====================================================================

import math
import random

import pytest

from libraries import rfMath
from libraries.decisionTree import DecisionTreeClassifier, DecisionTreeRegressor
from libraries.gradientBoosting import (GradientBoostingClassifier,
                                        GradientBoostingRegressor)
from libraries.randomForest import RandomForestClassifier, RandomForestRegressor
from pipeline import labeling, splitter


# ---------------------------------------------------------------------
# ① Dữ liệu mẫu — bài toán XOR có nhiễu, cùng vài đặc trưng rác
# ---------------------------------------------------------------------
@pytest.fixture
def xor_dataset():
    """
    XOR là phép thử tốt vì KHÔNG mô hình tuyến tính nào giải được: chỉ
    riêng dấu của từng đặc trưng không nói lên gì, phải kết hợp cả hai.
    Ba đặc trưng rác thêm vào để kiểm tra mô hình có phân biệt được
    tín hiệu với nhiễu hay không.
    """
    generator = random.Random(0)
    samples, labels = [], []
    for _ in range(400):
        first = generator.gauss(0, 1)
        second = generator.gauss(0, 1)
        noise = [generator.gauss(0, 1) for _ in range(3)]
        label = 1 if (first > 0) != (second > 0) else 0
        if generator.random() < 0.05:
            label = 1 - label
        samples.append([first, second] + noise)
        labels.append(label)
    return samples, labels


@pytest.fixture
def regression_dataset():
    """Quan hệ phi tuyến y = x₁² - 2·x₂ kèm nhiễu nhẹ."""
    generator = random.Random(1)
    samples, targets = [], []
    for _ in range(300):
        first = generator.uniform(-3, 3)
        second = generator.uniform(-3, 3)
        samples.append([first, second, generator.gauss(0, 1)])
        targets.append(first ** 2 - 2 * second + generator.gauss(0, 0.3))
    return samples, targets


def accuracy(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


# ---------------------------------------------------------------------
# ② Độ đo hỗn tạp — đối chiếu với giá trị lý thuyết
# ---------------------------------------------------------------------
def test_gini_of_pure_node_is_zero():
    assert rfMath.calculate_gini_impurity([1, 1, 1, 1]) == pytest.approx(0.0)


def test_gini_of_balanced_binary_node_is_half():
    """Hai lớp chia đôi: Gini = 1 - (0.5² + 0.5²) = 0.5 — giá trị lớn nhất."""
    assert rfMath.calculate_gini_impurity([0, 0, 1, 1]) == pytest.approx(0.5)


def test_entropy_of_balanced_binary_node_is_one_bit():
    """Hai lớp chia đôi cần đúng 1 bit thông tin để mô tả."""
    assert rfMath.calculate_entropy([0, 0, 1, 1]) == pytest.approx(1.0)


def test_entropy_of_four_balanced_classes_is_two_bits():
    assert rfMath.calculate_entropy([0, 1, 2, 3]) == pytest.approx(2.0)


def test_impurity_decrease_is_maximal_for_perfect_split():
    """Phép chia tách sạch hai lớp làm giảm toàn bộ độ hỗn tạp của nút cha."""
    parent = [0, 0, 1, 1]
    decrease = rfMath.calculate_impurity_decrease(
        parent, [0, 0], [1, 1], rfMath.calculate_gini_impurity)

    assert decrease == pytest.approx(0.5)


def test_impurity_decrease_is_zero_for_useless_split():
    """Phép chia giữ nguyên tỷ lệ lớp ở hai nhánh không mang lại thông tin."""
    decrease = rfMath.calculate_impurity_decrease(
        [0, 0, 1, 1], [0, 1], [0, 1], rfMath.calculate_gini_impurity)

    assert decrease == pytest.approx(0.0)


def test_split_accumulator_matches_direct_calculation():
    """
    Bộ tích luỹ quét một lượt phải cho đúng kết quả như tính trực tiếp —
    đây là tối ưu về tốc độ, không được đổi kết quả.
    """
    targets = [0, 1, 0, 1, 1, 0, 1, 1]
    accumulator = rfMath.LabelSplitAccumulator(targets, 'gini')

    for position in range(len(targets) - 1):
        accumulator.move_to_left(targets[position])
        direct = rfMath.calculate_weighted_impurity(
            targets[:position + 1], targets[position + 1:],
            rfMath.calculate_gini_impurity)
        assert accumulator.calculate_weighted_impurity() == pytest.approx(direct)


def test_value_split_accumulator_matches_direct_calculation():
    targets = [1.0, 3.0, 2.0, 7.0, 5.0, 4.0]
    accumulator = rfMath.ValueSplitAccumulator(targets)

    for position in range(len(targets) - 1):
        accumulator.move_to_left(targets[position])
        direct = rfMath.calculate_weighted_impurity(
            targets[:position + 1], targets[position + 1:],
            rfMath.calculate_variance)
        assert accumulator.calculate_weighted_impurity() == pytest.approx(direct)


# ---------------------------------------------------------------------
# ③ Bootstrap — trụ cột ngẫu nhiên thứ nhất, kiểm chứng bằng lý thuyết
# ---------------------------------------------------------------------
def test_bootstrap_draws_exactly_n_samples():
    in_bag, out_of_bag = rfMath.generate_bootstrap_indices(100, random.Random(0))

    assert len(in_bag) == 100
    assert set(in_bag).isdisjoint(out_of_bag)
    assert set(in_bag) | set(out_of_bag) == set(range(100))


def test_out_of_bag_ratio_converges_to_one_over_e():
    """
    Tỷ lệ mẫu ngoài túi phải tiến về 1/e ≈ 0.3679 khi n lớn. Đây là phép
    kiểm chứng lý thuyết trực tiếp nhất cho cài đặt bootstrap.
    """
    generator = random.Random(42)
    ratios = []
    for _ in range(30):
        _, out_of_bag = rfMath.generate_bootstrap_indices(500, generator)
        ratios.append(len(out_of_bag) / 500)

    average_ratio = sum(ratios) / len(ratios)
    assert average_ratio == pytest.approx(1 / math.e, abs=0.02)


def test_expected_out_of_bag_ratio_formula():
    assert rfMath.calculate_expected_out_of_bag_ratio(10000) == pytest.approx(
        1 / math.e, abs=1e-4)


def test_resolve_max_features_conventions():
    assert rfMath.resolve_max_features(25, 'sqrt') == 5
    assert rfMath.resolve_max_features(16, 'log2') == 4
    assert rfMath.resolve_max_features(30, 'third') == 10
    assert rfMath.resolve_max_features(30, 'all') == 30
    assert rfMath.resolve_max_features(30, None) == 30
    assert rfMath.resolve_max_features(30, 0.5) == 15
    assert rfMath.resolve_max_features(30, 100) == 30    # kẹp về tổng số
    assert rfMath.resolve_max_features(30, 0) == 1       # kẹp về tối thiểu


def test_resolve_max_features_rejects_invalid_input():
    with pytest.raises(ValueError):
        rfMath.resolve_max_features(10, 'không-tồn-tại')
    with pytest.raises(ValueError):
        rfMath.resolve_max_features(10, 1.5)


def test_majority_vote_is_deterministic_on_ties():
    """Hoà phiếu phải cho kết quả tất định để mô hình tái lập được."""
    assert rfMath.majority_vote([0, 1]) == rfMath.majority_vote([1, 0])


# ---------------------------------------------------------------------
# ④ Cây quyết định — học được quy luật và tôn trọng ràng buộc
# ---------------------------------------------------------------------
def test_tree_learns_a_separable_rule():
    """Quy luật tách được hoàn toàn thì cây phải học đúng 100%."""
    samples = [[value, 0.0] for value in range(20)]
    labels = [1 if value >= 10 else 0 for value in range(20)]

    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(samples, labels)

    assert tree.predict(samples) == labels


def test_tree_respects_max_depth(xor_dataset):
    samples, labels = xor_dataset
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(samples, labels)

    assert tree.get_depth() <= 3


def test_tree_respects_min_samples_leaf(xor_dataset):
    """Mọi lá phải có ít nhất min_samples_leaf mẫu."""
    samples, labels = xor_dataset
    tree = DecisionTreeClassifier(min_samples_leaf=25, random_state=0).fit(
        samples, labels)

    leaves = []

    def collect(node):
        if node.is_leaf:
            leaves.append(node.num_samples)
        else:
            collect(node.left)
            collect(node.right)

    collect(tree.root)
    assert min(leaves) >= 25


def test_tree_probabilities_form_a_distribution(xor_dataset):
    samples, labels = xor_dataset
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(samples, labels)

    for vector in tree.predict_probabilities(samples[:20]):
        assert sum(vector) == pytest.approx(1.0)
        assert all(0.0 <= value <= 1.0 for value in vector)


def test_regression_tree_cannot_extrapolate(regression_dataset):
    """
    Tính chất cấu trúc của mô hình dựa trên cây: dự đoán luôn nằm TRONG
    khoảng giá trị mục tiêu đã thấy khi huấn luyện.
    """
    samples, targets = regression_dataset
    tree = DecisionTreeRegressor(max_depth=6, random_state=0).fit(samples, targets)

    far_outside = [[100.0, 100.0, 0.0], [-100.0, -100.0, 0.0]]
    for prediction in tree.predict(far_outside):
        assert min(targets) <= prediction <= max(targets)


def test_tree_export_text_is_readable(xor_dataset):
    samples, labels = xor_dataset
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(samples, labels)

    text = tree.export_text(feature_names=['a', 'b', 'n1', 'n2', 'n3'])

    assert 'dự đoán' in text
    assert any(name in text for name in ('a', 'b'))


def test_tree_raises_before_fit():
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier().predict([[1.0]])


def test_tree_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit([[1.0], [2.0]], [0])


# ---------------------------------------------------------------------
# ⑤ Rừng ngẫu nhiên — tính chất tổng hợp và ước lượng OOB
# ---------------------------------------------------------------------
def test_forest_learns_xor(xor_dataset):
    """Rừng phải học được quan hệ phi tuyến mà mô hình tuyến tính bó tay."""
    samples, labels = xor_dataset
    forest = RandomForestClassifier(
        n_estimators=30, max_features='sqrt', min_samples_leaf=5,
        random_state=42).fit(samples[:300], labels[:300])

    assert accuracy(labels[300:], forest.predict(samples[300:])) > 0.80


def test_forest_is_reproducible(xor_dataset):
    """Cùng hạt giống phải cho kết quả giống hệt — điều kiện để tái lập."""
    samples, labels = xor_dataset
    settings = dict(n_estimators=15, max_features='sqrt', random_state=7)

    first = RandomForestClassifier(**settings).fit(samples, labels)
    second = RandomForestClassifier(**settings).fit(samples, labels)

    assert first.predict(samples[:50]) == second.predict(samples[:50])


def test_different_seeds_give_different_forests(xor_dataset):
    """
    Đổi hạt giống phải cho rừng khác — nếu không, ngẫu nhiên hoá hỏng.

    So sánh ở mức CẤU TRÚC chứ không so dự đoán: với bài toán dễ, hai
    rừng khác nhau vẫn có thể dự đoán giống hệt nhau trên tập huấn
    luyện, nên dự đoán trùng nhau không chứng minh được điều gì.
    """
    samples, labels = xor_dataset

    first = RandomForestClassifier(n_estimators=15, random_state=1).fit(samples, labels)
    second = RandomForestClassifier(n_estimators=15, random_state=2).fit(samples, labels)

    assert first.out_of_bag_indices != second.out_of_bag_indices
    assert [(tree.root.feature_index, tree.root.threshold) for tree in first.trees] \
        != [(tree.root.feature_index, tree.root.threshold) for tree in second.trees]


def test_forest_out_of_bag_ratio_matches_theory(xor_dataset):
    samples, labels = xor_dataset
    forest = RandomForestClassifier(n_estimators=30, random_state=0).fit(samples, labels)

    assert forest.describe()['average_out_of_bag_ratio'] == pytest.approx(
        1 / math.e, abs=0.03)


def test_forest_out_of_bag_error_is_reasonable(xor_dataset):
    """Lỗi OOB phải nằm trong [0,1] và tốt hơn hẳn mức đoán bừa 0.5."""
    samples, labels = xor_dataset
    forest = RandomForestClassifier(
        n_estimators=40, min_samples_leaf=5, random_state=0).fit(samples, labels)

    error = forest.calculate_out_of_bag_error()
    assert 0.0 <= error <= 1.0
    assert error < 0.35


def test_out_of_bag_requires_bootstrap(xor_dataset):
    """Tắt bootstrap thì không còn mẫu ngoài túi để ước lượng."""
    samples, labels = xor_dataset
    forest = RandomForestClassifier(
        n_estimators=5, bootstrap=False, random_state=0).fit(samples, labels)

    with pytest.raises(RuntimeError):
        forest.calculate_out_of_bag_error()


def test_out_of_bag_error_curve_is_monotone_in_length(xor_dataset):
    samples, labels = xor_dataset
    forest = RandomForestClassifier(n_estimators=20, random_state=0).fit(samples, labels)

    tree_counts, errors = forest.calculate_out_of_bag_error_curve()

    assert len(tree_counts) == len(errors)
    assert tree_counts == sorted(tree_counts)
    assert tree_counts[-1] == 20


def test_feature_importances_identify_signal_over_noise(xor_dataset):
    """
    Hai đặc trưng đầu mang tín hiệu, ba đặc trưng sau là nhiễu thuần.
    Tổng tầm quan trọng của nhóm tín hiệu phải vượt trội.
    """
    samples, labels = xor_dataset
    forest = RandomForestClassifier(
        n_estimators=40, min_samples_leaf=5, random_state=0).fit(samples, labels)

    importances = forest.calculate_feature_importances()

    assert sum(importances) == pytest.approx(1.0)
    assert sum(importances[:2]) > sum(importances[2:])


def test_permutation_importances_identify_signal_over_noise(xor_dataset):
    samples, labels = xor_dataset
    forest = RandomForestClassifier(
        n_estimators=30, min_samples_leaf=5, random_state=0).fit(
            samples[:300], labels[:300])

    importances = forest.calculate_permutation_importances(
        samples[300:], labels[300:], num_repeats=3, random_state=0)

    assert min(importances[:2]) > max(importances[2:])


def test_forest_probabilities_form_a_distribution(xor_dataset):
    samples, labels = xor_dataset
    forest = RandomForestClassifier(n_estimators=20, random_state=0).fit(samples, labels)

    for vector in forest.predict_probabilities(samples[:20]):
        assert sum(vector) == pytest.approx(1.0)


def test_forest_regressor_beats_constant_baseline(regression_dataset):
    samples, targets = regression_dataset
    forest = RandomForestRegressor(
        n_estimators=30, min_samples_leaf=5, random_state=0).fit(
            samples[:250], targets[:250])

    predictions = forest.predict(samples[250:])
    actual = targets[250:]
    mean_value = sum(targets[:250]) / 250

    model_error = sum((a - b) ** 2 for a, b in zip(actual, predictions))
    baseline_error = sum((a - mean_value) ** 2 for a in actual)

    assert model_error < baseline_error


def test_forest_raises_before_fit():
    with pytest.raises(RuntimeError):
        RandomForestClassifier().predict([[1.0]])


# ---------------------------------------------------------------------
# ⑥ Gradient boosting — mất mát phải giảm đơn điệu theo vòng lặp
# ---------------------------------------------------------------------
def test_boosting_regressor_reduces_training_loss(regression_dataset):
    samples, targets = regression_dataset
    model = GradientBoostingRegressor(
        n_estimators=30, learning_rate=0.1, max_depth=3, random_state=0).fit(
            samples, targets)

    history = model.training_loss_history
    assert history[-1] < history[0]
    assert len(history) == 30


def test_boosting_classifier_reduces_log_loss(xor_dataset):
    samples, labels = xor_dataset
    model = GradientBoostingClassifier(
        n_estimators=40, learning_rate=0.1, max_depth=3, random_state=0).fit(
            samples, labels)

    history = model.training_loss_history
    assert history[-1] < history[0]
    assert accuracy(labels, model.predict(samples)) > 0.75


def test_boosting_classifier_probabilities_in_range(xor_dataset):
    samples, labels = xor_dataset
    model = GradientBoostingClassifier(
        n_estimators=20, max_depth=2, random_state=0).fit(samples, labels)

    for probability in model.predict_scores(samples[:50]):
        assert 0.0 <= probability <= 1.0


def test_boosting_classifier_rejects_more_than_two_classes():
    samples = [[float(value)] for value in range(30)]
    labels = [value % 3 for value in range(30)]

    with pytest.raises(ValueError):
        GradientBoostingClassifier(n_estimators=5).fit(samples, labels)


# ---------------------------------------------------------------------
# ⑦ Tách tập và gán nhãn — hai nơi dễ gây rò rỉ dữ liệu nhất
# ---------------------------------------------------------------------
def test_split_preserves_time_order():
    """Ba tập phải liên tiếp theo thời gian và không chồng lấn."""
    train, validation, test = splitter.split_indices_sequentially(1000, 0.70, 0.15)

    assert max(train) < min(validation) < max(validation) < min(test)
    assert set(train).isdisjoint(validation)
    assert set(validation).isdisjoint(test)


def test_split_gap_creates_real_separation():
    """gap phải tạo khoảng trống thật giữa hai tập."""
    train, validation, _ = splitter.split_indices_sequentially(1000, 0.70, 0.15, gap=10)

    assert min(validation) - max(train) == 11    # 10 mẫu bỏ trống ở giữa


def test_split_rejects_invalid_ratios():
    with pytest.raises(ValueError):
        splitter.split_indices_sequentially(100, 0.90, 0.15)
    with pytest.raises(ValueError):
        splitter.split_indices_sequentially(100, 1.5, 0.15)


def test_expanding_window_folds_only_move_forward():
    """Mọi mẫu kiểm định phải nằm SAU mọi mẫu huấn luyện của cùng vòng."""
    folds = splitter.generate_expanding_window_folds(1000, num_folds=4, gap=5)

    assert len(folds) == 4
    previous_train_size = 0
    for train_indices, validation_indices in folds:
        assert max(train_indices) < min(validation_indices)
        assert len(train_indices) > previous_train_size    # cửa sổ MỞ RỘNG
        previous_train_size = len(train_indices)


def test_rolling_window_folds_keep_train_size_constant():
    folds = splitter.generate_rolling_window_folds(
        1000, num_folds=4, train_size=400, gap=5)

    for train_indices, validation_indices in folds:
        assert len(train_indices) == 400
        assert max(train_indices) < min(validation_indices)


def test_direction_labels_shift_backwards_not_forwards():
    """
    Nhãn tại t phải lấy từ giá trị tại t+horizon. Kiểm tra trực tiếp trên
    một chuỗi mà mỗi bước đều biết trước chiều.
    """
    series = [10.0, 11.0, 10.0, 12.0, 11.0]
    labels = labeling.create_direction_labels(series, horizon=1)

    assert labels == [1, 0, 1, 0, None]


def test_labels_leave_tail_without_target():
    """horizon phần tử cuối không thể có nhãn."""
    labels = labeling.create_direction_labels([1.0] * 10, horizon=3, flat_label=0)

    assert labels[-3:] == [None, None, None]


def test_flat_observations_are_droppable():
    """flat_label=None khiến quan sát đứng yên bị loại ở bước khớp."""
    series = [10.0, 10.0, 11.0, 11.0, 12.0]
    labels = labeling.create_direction_labels(series, horizon=1, flat_label=None)

    assert labels == [None, 1, None, 1, None]


def test_count_flat_observations_matches_labels():
    series = [10.0, 10.0, 11.0, 11.0, 12.0]
    summary = labeling.count_flat_observations(series, horizon=1)

    assert summary == {'increase': 2, 'decrease': 0, 'flat': 2,
                       'flat_ratio': 0.5, 'comparable': 4}


def test_align_features_and_targets_drops_unusable_rows():
    samples = [[1.0], [2.0], [None], [4.0]]
    targets = [1, None, 0, 0]

    clean_samples, clean_targets, kept = labeling.align_features_and_targets(
        samples, targets)

    assert clean_samples == [[1.0], [4.0]]
    assert clean_targets == [1, 0]
    assert kept == [0, 3]


def test_class_balance_reports_majority():
    balance = labeling.calculate_class_balance([0, 0, 0, 1, 1])

    assert balance['majority_label'] == 0
    assert balance['majority_ratio'] == pytest.approx(0.6)
    assert balance['num_samples'] == 5
