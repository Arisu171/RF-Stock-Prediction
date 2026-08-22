# =====================================================================
# Test Model Store — kiểm thử việc lưu và nạp lại mô hình
# =====================================================================
# Một mô hình lưu ra rồi nạp lại phải cho dự đoán GIỐNG HỆT bản gốc.
# Nếu không, mọi con số đánh giá lúc huấn luyện đều không áp dụng được
# cho mô hình đang chạy thật — và sai lệch này rất khó phát hiện vì
# chương trình vẫn chạy trơn tru.
#
# Nhóm test quan trọng nhất ở đây là ② và ⑥: đối chiếu từng giá trị dự
# đoán, và chặn trường hợp công thức đặc trưng bị lệch.
#
# Thứ tự khai báo:
#
#   ①  Dữ liệu và mô hình mẫu
#   ②  Vòng lưu-nạp giữ nguyên dự đoán
#   ③  Vòng lưu-nạp giữ nguyên cấu trúc cây
#   ④  Điều cố ý KHÔNG giữ lại sau khi nạp
#   ⑤  Ghi và đọc file JSON
#   ⑥  Kiểm tra tương thích công thức đặc trưng
#   ⑦  Đếm tham số
# =====================================================================

import json
import os
import random

import pytest

from libraries import modelStore
from libraries.decisionTree import DecisionTreeClassifier
from libraries.randomForest import RandomForestClassifier, RandomForestRegressor
from pipeline import modelBundle


# ---------------------------------------------------------------------
# ① Dữ liệu và mô hình mẫu — dùng chung cho cả tệp
# ---------------------------------------------------------------------
@pytest.fixture
def dataset():
    """Bài toán XOR có nhiễu, đủ để cây phải chia nhiều tầng."""
    generator = random.Random(0)
    samples, labels = [], []
    for _ in range(200):
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
        n_estimators=12, max_depth=4, min_samples_leaf=5,
        random_state=42).fit(samples, labels)


@pytest.fixture
def minimal_config():
    """Cấu hình tối thiểu đủ để đóng gói — không gắn với đề tài nào."""
    return {
        'description': 'Cấu hình dùng cho kiểm thử.',
        'dataset': {
            'key_column': 'Date',
            'numeric_columns': ['Close'],
            'series': {'close': 'Close'},
        },
        'preprocess': {'max_forward_fill': 2},
        'features': [
            {'name': 'ratio_3', 'indicator': 'ratio_to_moving_average',
             'inputs': ['close'], 'params': {'window': 3}},
            {'name': 'ratio_5', 'indicator': 'ratio_to_moving_average',
             'inputs': ['close'], 'params': {'window': 5}},
            {'name': 'roc_2', 'indicator': 'rate_of_change',
             'inputs': ['close'], 'params': {'window': 2}},
        ],
        'labeling': {'task': 'classification', 'source_series': 'close',
                     'horizon': 1, 'positive_label': 1, 'negative_label': 0},
    }


# ---------------------------------------------------------------------
# ② Dự đoán không đổi — test quan trọng nhất của cả tệp
# ---------------------------------------------------------------------
def test_round_trip_preserves_predictions(forest, dataset):
    """Nạp lại phải cho ĐÚNG TỪNG nhãn dự đoán như bản gốc."""
    samples, _ = dataset
    restored = modelStore.deserialize_forest(modelStore.serialize_forest(forest))

    assert restored.predict(samples) == forest.predict(samples)


def test_round_trip_preserves_probabilities(forest, dataset):
    """Xác suất cũng phải trùng khít, không chỉ nhãn."""
    samples, _ = dataset
    restored = modelStore.deserialize_forest(modelStore.serialize_forest(forest))

    original = forest.predict_probabilities(samples)
    recovered = restored.predict_probabilities(samples)

    for first, second in zip(original, recovered):
        assert first == pytest.approx(second)


def test_round_trip_preserves_scores(forest, dataset):
    """Điểm số liên tục phải trùng — nếu lệch thì ROC-AUC sẽ khác."""
    samples, _ = dataset
    restored = modelStore.deserialize_forest(modelStore.serialize_forest(forest))

    assert restored.predict_scores(samples) == pytest.approx(
        forest.predict_scores(samples))


def test_round_trip_preserves_feature_importances(forest):
    """Tầm quan trọng đặc trưng suy từ cấu trúc cây nên cũng phải trùng."""
    restored = modelStore.deserialize_forest(modelStore.serialize_forest(forest))

    assert restored.calculate_feature_importances() == pytest.approx(
        forest.calculate_feature_importances())


def test_regressor_round_trip(dataset):
    """Rừng hồi quy cũng phải giữ nguyên giá trị dự đoán."""
    samples, _ = dataset
    targets = [row[0] ** 2 - 2 * row[1] for row in samples]

    model = RandomForestRegressor(
        n_estimators=10, max_depth=4, random_state=7).fit(samples, targets)
    restored = modelStore.deserialize_forest(modelStore.serialize_forest(model))

    assert restored.predict(samples) == pytest.approx(model.predict(samples))


# ---------------------------------------------------------------------
# ③ Cấu trúc cây không đổi — bảo đảm không mất mát khi tuần tự hoá
# ---------------------------------------------------------------------
def test_round_trip_preserves_tree_shape(forest):
    restored = modelStore.deserialize_forest(modelStore.serialize_forest(forest))

    for original, recovered in zip(forest.trees, restored.trees):
        assert recovered.get_depth() == original.get_depth()
        assert recovered.count_nodes() == original.count_nodes()
        assert recovered.count_leaves() == original.count_leaves()


def test_round_trip_preserves_split_rules(forest):
    """Từng luật chia (đặc trưng, ngưỡng) phải giữ nguyên tuyệt đối."""
    restored = modelStore.deserialize_forest(modelStore.serialize_forest(forest))

    def collect(node, rules):
        if not node.is_leaf:
            rules.append((node.feature_index, node.threshold))
            collect(node.left, rules)
            collect(node.right, rules)
        return rules

    for original, recovered in zip(forest.trees, restored.trees):
        assert collect(recovered.root, []) == collect(original.root, [])


def test_single_tree_round_trip(dataset):
    samples, labels = dataset
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(samples, labels)
    restored = modelStore.deserialize_tree(modelStore.serialize_tree(tree))

    assert restored.predict(samples) == tree.predict(samples)
    assert restored.label_space == tree.label_space


# ---------------------------------------------------------------------
# ④ Cố ý không giữ lại — lỗi rõ ràng tốt hơn con số sai
# ---------------------------------------------------------------------
def test_out_of_bag_unavailable_after_reload(forest):
    """
    Tập mẫu huấn luyện không được lưu, nên lỗi OOB phải báo lỗi thay vì
    trả về một con số tính từ dữ liệu rỗng.
    """
    restored = modelStore.deserialize_forest(modelStore.serialize_forest(forest))

    covered, outputs = restored.calculate_out_of_bag_predictions()
    assert covered == [] and outputs == []
    assert restored.calculate_out_of_bag_error() is None


def test_unfitted_model_cannot_be_saved():
    with pytest.raises(RuntimeError):
        modelStore.serialize_forest(RandomForestClassifier())


def test_wrong_format_version_is_rejected(forest):
    payload = modelStore.serialize_forest(forest)
    payload['format_version'] = 999

    with pytest.raises(ValueError):
        modelStore.deserialize_forest(payload)


# ---------------------------------------------------------------------
# ⑤ Vào/ra file — gói mô hình phải là MỘT file JSON đọc được
# ---------------------------------------------------------------------
def test_bundle_survives_json_file(tmp_path, forest, minimal_config):
    bundle = modelBundle.build_bundle(
        forest, minimal_config, ['ratio_3', 'ratio_5', 'roc_2'])
    path = os.path.join(str(tmp_path), 'model.json')
    modelBundle.save_bundle(bundle, path)

    assert os.path.exists(path)
    loaded = modelBundle.load_bundle(path)

    assert loaded['feature_names'] == bundle['feature_names']
    assert loaded['recipe']['features'] == minimal_config['features']
    assert loaded['estimator'].num_features == forest.num_features


def test_bundle_is_valid_json(tmp_path, forest, minimal_config):
    """File phải đọc được bằng json chuẩn, không phụ thuộc mã dự án."""
    bundle = modelBundle.build_bundle(
        forest, minimal_config, ['ratio_3', 'ratio_5', 'roc_2'])
    path = os.path.join(str(tmp_path), 'model.json')
    modelBundle.save_bundle(bundle, path)

    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)

    assert payload['bundle_version'] == modelBundle.BUNDLE_VERSION
    assert 'recipe' in payload and 'model' in payload


def test_bundle_rejects_mismatched_feature_count(forest, minimal_config):
    """Số tên đặc trưng phải khớp số cột mô hình đã học."""
    with pytest.raises(ValueError):
        modelBundle.build_bundle(forest, minimal_config, ['chỉ_một_tên'])


def test_missing_bundle_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        modelBundle.load_bundle(os.path.join(str(tmp_path), 'không_có.json'))


# ---------------------------------------------------------------------
# ⑥ Tương thích công thức — chặn dạng sai nguy hiểm nhất
# ---------------------------------------------------------------------
def test_compatible_features_pass(forest, minimal_config):
    names = ['ratio_3', 'ratio_5', 'roc_2']
    bundle = modelBundle.build_bundle(forest, minimal_config, names)

    modelBundle.verify_feature_compatibility(bundle, names)      # không ném lỗi


def test_missing_feature_is_rejected(forest, minimal_config):
    bundle = modelBundle.build_bundle(
        forest, minimal_config, ['ratio_3', 'ratio_5', 'roc_2'])

    with pytest.raises(ValueError, match='thiếu'):
        modelBundle.verify_feature_compatibility(bundle, ['ratio_3', 'ratio_5'])


def test_reordered_features_are_rejected(forest, minimal_config):
    """
    Hoán vị hai cột là dạng sai NGUY HIỂM NHẤT: mô hình vẫn chạy, vẫn
    cho ra số, nhưng đọc nhầm chỉ báo này thành chỉ báo kia.
    """
    bundle = modelBundle.build_bundle(
        forest, minimal_config, ['ratio_3', 'ratio_5', 'roc_2'])

    with pytest.raises(ValueError, match='KHÁC THỨ TỰ'):
        modelBundle.verify_feature_compatibility(
            bundle, ['roc_2', 'ratio_3', 'ratio_5'])


def test_prediction_keeps_rows_without_labels(minimal_config):
    """
    Đường dự đoán phải GIỮ LẠI quan sát mới nhất — thứ chưa có nhãn vì
    tương lai chưa xảy ra, và cũng chính là thứ ta cần dự báo.
    """
    table = {
        'Date': [f'2024-01-{day:02d}' for day in range(1, 21)],
        'Close': [100.0 + (index % 5) + index * 0.3 for index in range(20)],
    }
    samples, keys, names = modelBundle.prepare_features_for_prediction(
        minimal_config['dataset'] | {
            'preprocess': minimal_config['preprocess'],
            'features': minimal_config['features'],
        }, table)

    assert names == ['ratio_3', 'ratio_5', 'roc_2']
    assert len(samples) == len(keys)
    assert keys[-1].isoformat() == '2024-01-20'   # mốc cuối cùng còn nguyên


# ---------------------------------------------------------------------
# ⑦ Đếm tham số — thước đo kích thước thật của mô hình
# ---------------------------------------------------------------------
def test_parameter_count_matches_tree_structure(forest):
    size = modelStore.count_parameters(forest)

    expected_internal = sum(
        tree.count_nodes() - tree.count_leaves() for tree in forest.trees)
    expected_leaves = sum(tree.count_leaves() for tree in forest.trees)

    assert size['num_trees'] == len(forest.trees)
    assert size['internal_nodes'] == expected_internal
    assert size['leaf_nodes'] == expected_leaves
    # nút trong 2 số + lá 1 số giá trị + 2 số xác suất (bài toán hai lớp)
    assert size['parameters'] == expected_internal * 2 + expected_leaves * 3
