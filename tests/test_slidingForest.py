# =====================================================================
# Test Sliding Forest — kiểm thử rừng biết thay máu
# =====================================================================
# Hai tính chất quan trọng nhất:
#
#   1. LÚC KHỞI TẠO, rừng trượt PHẢI dự đoán y hệt rừng gốc. Nếu không,
#      phép so sánh "tĩnh ↔ thích nghi" mất ý nghĩa vì hai đường đã
#      khác nhau ngay từ đầu, chưa cần thay máu.
#   2. RỪNG GỐC KHÔNG ĐƯỢC ĐỔI khi bị bọc. Cả hai dùng chung đối tượng
#      cây, nên nếu danh sách cây bị chia sẻ thì việc thay máu sẽ âm
#      thầm làm hỏng luôn mô hình đối chứng.
#
# Thứ tự khai báo:
#
#   ①  Dữ liệu và rừng mẫu
#   ②  Khởi tạo — giống hệt rừng gốc, không làm hỏng rừng gốc
#   ③  Nhịp thay máu
#   ④  Kích thước rừng: thay máu ↔ chỉ thêm
#   ⑤  Bộ đệm quan sát
#   ⑥  Tái lập và báo lỗi tham số
# =====================================================================

import random

import pytest

from libraries.randomForest import RandomForestClassifier
from libraries.slidingForest import (SlidingRandomForestClassifier,
                                     SlidingRandomForestRegressor)


# ---------------------------------------------------------------------
# ① Dữ liệu và rừng mẫu — XOR có nhiễu, đủ để cây phải chia nhiều tầng
# ---------------------------------------------------------------------
@pytest.fixture
def dataset():
    generator = random.Random(0)
    samples, labels = [], []
    for _ in range(300):
        first = generator.gauss(0, 1)
        second = generator.gauss(0, 1)
        label = 1 if (first > 0) != (second > 0) else 0
        samples.append([first, second, generator.gauss(0, 1)])
        labels.append(label)
    return samples, labels


@pytest.fixture
def forest(dataset):
    samples, labels = dataset
    return RandomForestClassifier(
        n_estimators=20, max_depth=4, min_samples_leaf=5,
        random_state=42).fit(samples, labels)


def feed(model, samples, labels, count):
    """Nạp `count` quan sát đầu tiên vào mô hình thích nghi."""
    updates = 0
    for position in range(count):
        if model.observe(samples[position], labels[position]):
            updates += 1
    return updates


# ---------------------------------------------------------------------
# ② Khởi tạo — điều kiện để phép so sánh "tĩnh ↔ thích nghi" có nghĩa
# ---------------------------------------------------------------------
def test_wrapped_forest_predicts_identically(forest, dataset):
    """Trước khi thay máu lần nào, hai rừng phải cho kết quả trùng khít."""
    samples, _ = dataset
    sliding = SlidingRandomForestClassifier.from_forest(forest)

    assert sliding.predict(samples) == forest.predict(samples)
    assert sliding.predict_scores(samples) == pytest.approx(
        forest.predict_scores(samples))


def test_wrapping_does_not_mutate_source_forest(forest, dataset):
    """
    Rừng gốc phải giữ nguyên trạng thái tĩnh sau khi bị bọc VÀ sau khi
    bản bọc đã thay máu — nếu không, mô hình đối chứng bị hỏng âm thầm
    và mọi so sánh về sau đều sai.
    """
    samples, labels = dataset
    before = forest.predict(samples)
    original_trees = list(forest.trees)

    sliding = SlidingRandomForestClassifier.from_forest(
        forest, trees_per_update=5, window_size=50, update_every=10)
    feed(sliding, samples, labels, 60)

    assert sliding.describe_adaptation()['num_updates'] > 0
    assert forest.trees == original_trees
    assert forest.predict(samples) == before


def test_label_space_is_carried_over(forest):
    sliding = SlidingRandomForestClassifier.from_forest(forest)
    assert sliding.label_space == forest.label_space


def test_requires_trained_forest():
    """Chưa huấn luyện thì không có cây nào để thay máu."""
    with pytest.raises(RuntimeError):
        SlidingRandomForestClassifier().configure_sliding()


def test_methods_require_configuration(forest):
    sliding = SlidingRandomForestClassifier.from_forest(forest)
    del sliding.update_every                      # giả lập chưa cấu hình

    with pytest.raises(RuntimeError):
        sliding.observe([0.0, 0.0, 0.0], 1)


# ---------------------------------------------------------------------
# ③ Nhịp thay máu — đúng update_every quan sát mới thay một lần
# ---------------------------------------------------------------------
def test_update_fires_on_schedule(forest, dataset):
    samples, labels = dataset
    sliding = SlidingRandomForestClassifier.from_forest(
        forest, trees_per_update=3, window_size=60, update_every=20)

    updates = feed(sliding, samples, labels, 100)

    assert updates == 5
    assert sliding.describe_adaptation()['num_updates'] == 5


def test_no_update_before_threshold(forest, dataset):
    samples, labels = dataset
    sliding = SlidingRandomForestClassifier.from_forest(
        forest, trees_per_update=3, window_size=60, update_every=50)

    assert feed(sliding, samples, labels, 49) == 0
    assert sliding.describe_adaptation()['num_updates'] == 0
    assert sliding.describe_adaptation()['pending_until_update'] == 1


def test_update_skipped_when_window_has_one_class(forest):
    """
    Cửa sổ chỉ có một lớp thì cây mới không phân biệt được gì. Thà giữ
    nguyên rừng cũ còn hơn nhét vào những cây chỉ biết đoán một phía.
    """
    sliding = SlidingRandomForestClassifier.from_forest(
        forest, trees_per_update=3, window_size=40, update_every=10)

    generator = random.Random(1)
    for _ in range(30):
        sliding.observe([generator.gauss(0, 1) for _ in range(3)], 1)

    assert sliding.describe_adaptation()['num_updates'] == 0
    assert len(sliding.trees) == 20


# ---------------------------------------------------------------------
# ④ Kích thước rừng — thay máu giữ nguyên, chỉ-thêm thì lớn dần
# ---------------------------------------------------------------------
def test_replacing_keeps_forest_size_constant(forest, dataset):
    samples, labels = dataset
    sliding = SlidingRandomForestClassifier.from_forest(
        forest, trees_per_update=4, window_size=60, update_every=20)

    feed(sliding, samples, labels, 200)

    assert len(sliding.trees) == 20
    assert sliding.describe_adaptation()['forest_size'] == 20


def test_adding_without_retiring_grows_the_forest(forest, dataset):
    samples, labels = dataset
    sliding = SlidingRandomForestClassifier.from_forest(
        forest, trees_per_update=4, window_size=60, update_every=20,
        retire_old=False)

    feed(sliding, samples, labels, 200)
    updates = sliding.describe_adaptation()['num_updates']

    assert len(sliding.trees) == 20 + updates * 4


def test_trees_actually_rotate(forest, dataset):
    """
    Sau khi thay máu, cây cũ nhất phải biến mất khỏi rừng — nếu không
    thì "thay máu" chỉ là thêm cây, không đúng như tên gọi.
    """
    samples, labels = dataset
    oldest = forest.trees[0]

    sliding = SlidingRandomForestClassifier.from_forest(
        forest, trees_per_update=5, window_size=60, update_every=20)
    feed(sliding, samples, labels, 40)

    assert oldest not in sliding.trees
    assert oldest in forest.trees          # rừng gốc vẫn còn nguyên


def test_predictions_change_after_adaptation(forest, dataset):
    """Thay máu phải thực sự làm đổi hành vi, nếu không nó vô nghĩa."""
    samples, labels = dataset
    sliding = SlidingRandomForestClassifier.from_forest(
        forest, trees_per_update=10, window_size=80, update_every=20)

    feed(sliding, samples, labels, 120)

    assert sliding.predict_scores(samples[:50]) != pytest.approx(
        forest.predict_scores(samples[:50]))


# ---------------------------------------------------------------------
# ⑤ Bộ đệm quan sát — giữ đúng window_size quan sát gần nhất
# ---------------------------------------------------------------------
def test_buffer_is_capped_at_window_size(forest, dataset):
    samples, labels = dataset
    sliding = SlidingRandomForestClassifier.from_forest(
        forest, trees_per_update=2, window_size=30, update_every=1000)

    feed(sliding, samples, labels, 100)

    assert len(sliding.buffer_samples) == 30
    assert sliding.buffer_samples[-1] == samples[99]


def test_seed_buffer_preloads_training_data(forest, dataset):
    """
    Nạp sẵn dữ liệu gốc để cây mới không bị thiệt về lượng mẫu — cách
    tách biến "tính gần đây" khỏi biến "thiếu dữ liệu" khi làm thí nghiệm.
    """
    samples, labels = dataset
    sliding = SlidingRandomForestClassifier.from_forest(
        forest, trees_per_update=2, window_size=500, update_every=1000)
    sliding.seed_buffer(samples, labels)

    assert len(sliding.buffer_samples) == len(samples)

    sliding.observe(samples[0], labels[0])
    assert len(sliding.buffer_samples) == len(samples) + 1


def test_seed_buffer_respects_window_size(forest, dataset):
    samples, labels = dataset
    sliding = SlidingRandomForestClassifier.from_forest(
        forest, trees_per_update=2, window_size=40, update_every=1000)
    sliding.seed_buffer(samples, labels)

    assert len(sliding.buffer_samples) == 40
    assert sliding.buffer_samples[-1] == samples[-1]     # giữ phần MỚI nhất


def test_observe_copies_the_sample(forest):
    """Sửa danh sách gốc sau khi nạp không được làm đổi bộ đệm."""
    sliding = SlidingRandomForestClassifier.from_forest(
        forest, trees_per_update=2, window_size=10, update_every=1000)

    sample = [1.0, 2.0, 3.0]
    sliding.observe(sample, 1)
    sample[0] = 999.0

    assert sliding.buffer_samples[0] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------
# ⑥ Tái lập và báo lỗi tham số
# ---------------------------------------------------------------------
def test_adaptation_is_reproducible(forest, dataset):
    samples, labels = dataset
    settings = dict(trees_per_update=4, window_size=60, update_every=20)

    first = SlidingRandomForestClassifier.from_forest(forest, **settings)
    second = SlidingRandomForestClassifier.from_forest(forest, **settings)
    feed(first, samples, labels, 120)
    feed(second, samples, labels, 120)

    assert first.predict(samples[:50]) == second.predict(samples[:50])


def test_invalid_settings_raise(forest):
    for override in ({'trees_per_update': 0}, {'window_size': 1},
                     {'update_every': 0}):
        with pytest.raises(ValueError):
            SlidingRandomForestClassifier.from_forest(forest, **override)


def test_regressor_variant_works(dataset):
    from libraries.randomForest import RandomForestRegressor

    samples, _ = dataset
    targets = [row[0] ** 2 - 2 * row[1] for row in samples]
    model = RandomForestRegressor(
        n_estimators=10, max_depth=4, random_state=7).fit(samples, targets)

    sliding = SlidingRandomForestRegressor.from_forest(
        model, trees_per_update=3, window_size=60, update_every=20)
    assert sliding.predict(samples[:20]) == pytest.approx(
        model.predict(samples[:20]))

    for position in range(60):
        sliding.observe(samples[position], targets[position])

    assert sliding.describe_adaptation()['num_updates'] == 3
    assert len(sliding.trees) == 10
