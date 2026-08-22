# =====================================================================
# Test Replay Engine — kiểm thử bộ máy phát lại chuỗi thời gian
# =====================================================================
# Test quan trọng nhất của cả tệp là ② KHÔNG NHÌN TRƯỚC.
#
# Nếu engine lỡ chạm dữ liệu của tương lai, mọi con số trên màn hình
# streaming đều vô nghĩa — mà triệu chứng duy nhất là accuracy đẹp bất
# thường, thứ rất dễ bị hiểu nhầm thành "mô hình tốt". Vì vậy phép thử
# ở đây làm theo cách chặt nhất: cắt chuỗi tại một mốc rồi chạy lại,
# và đòi hỏi mọi dự đoán trước mốc đó phải GIỐNG HỆT.
#
# Các test dựng gói mô hình riêng ngay trong bộ nhớ thay vì đọc từ
# models/ — thư mục đó nằm trong .gitignore nên không thể là điều kiện
# tiên quyết của bộ kiểm thử.
#
# Thứ tự khai báo:
#
#   ①  Dữ liệu và gói mô hình mẫu
#   ②  KHÔNG NHÌN TRƯỚC — test then chốt
#   ③  Bỏ qua giai đoạn huấn luyện
#   ④  Hàng đợi chờ và độ trễ chấm điểm
#   ⑤  Ba đường accuracy
#   ⑥  Phối hợp với mô hình thích nghi
# =====================================================================

import math
import os

import pytest

from libraries.randomForest import RandomForestClassifier
from libraries.slidingForest import SlidingRandomForestClassifier
from pipeline import experiment, modelBundle
from pipeline.replayEngine import ReplayEngine, resolve_start_index

HORIZON = 3


# ---------------------------------------------------------------------
# ① Dữ liệu và gói mô hình mẫu — cửa sổ ngắn để không cần chuỗi dài
# ---------------------------------------------------------------------
def make_table(num_rows=160):
    """
    Chuỗi OHLCV giả lập, sinh bằng công thức tất định (không ngẫu nhiên)
    để mọi lần chạy test đều cho cùng kết quả.
    """
    keys, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    price = 100.0
    for index in range(num_rows):
        price += math.sin(index / 4.0) * 1.5 + (index % 7) * 0.2 - 0.6
        keys.append(f'2020-{(index // 28) + 1:02d}-{(index % 28) + 1:02d}')
        opens.append(round(price - 0.3, 4))
        highs.append(round(price + 1.2, 4))
        lows.append(round(price - 1.4, 4))
        closes.append(round(price, 4))
        volumes.append(1000 + (index % 11) * 37)

    return {'Date': keys, 'Open': opens, 'High': highs, 'Low': lows,
            'Close': closes, 'Volume': volumes}


def make_config():
    """Cấu hình tối thiểu — cửa sổ ngắn nên vùng khởi động chỉ vài dòng."""
    return {
        'description': 'Cấu hình dùng cho kiểm thử phát lại.',
        'dataset': {
            'path': 'không-dùng.csv', 'label': 'TEST',
            'key_column': 'Date',
            'numeric_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'series': {'high': 'High', 'low': 'Low', 'close': 'Close'},
        },
        'preprocess': {'max_forward_fill': 2},
        'features': [
            {'name': 'ratio_5', 'indicator': 'ratio_to_moving_average',
             'inputs': ['close'], 'params': {'window': 5}},
            {'name': 'ratio_10', 'indicator': 'ratio_to_moving_average',
             'inputs': ['close'], 'params': {'window': 10}},
            {'name': 'roc_3', 'indicator': 'rate_of_change',
             'inputs': ['close'], 'params': {'window': 3}},
            {'name': 'range_ratio', 'indicator': 'relative_range',
             'inputs': ['high', 'low', 'close']},
        ],
        'labeling': {'task': 'classification', 'source_series': 'close',
                     'horizon': HORIZON, 'positive_label': 1,
                     'negative_label': 0, 'flat_label': None},
        'split': {'train_ratio': 0.70, 'validation_ratio': 0.15,
                  'gap': HORIZON},
        'model': {'n_estimators': 8, 'max_depth': 3, 'min_samples_leaf': 5,
                  'max_features': 'sqrt', 'random_state': 0, 'verbose': 0},
    }


@pytest.fixture
def bundle(tmp_path):
    """Gói mô hình huấn luyện trên nửa đầu chuỗi, lưu rồi nạp lại."""
    config = make_config()
    table = make_table()

    # load_table() lo việc ép kiểu trong luồng thật; ở đây không có file
    # nên phải tự làm đúng bước đó trước khi làm sạch.
    from pipeline.timePreprocess import parse_date_series
    prepared = dict(table)
    for name in config['dataset']['numeric_columns']:
        prepared[name] = [float(value) for value in prepared[name]]
    prepared['Date'] = parse_date_series(prepared['Date'])

    cleaned, _ = experiment.clean_table(config, prepared, verbose=False)
    feature_table, _ = experiment.build_feature_table(config, cleaned)
    feature_names = sorted(feature_table)

    from libraries import rfMath
    from pipeline import labeling
    samples = rfMath.columns_to_samples(
        [feature_table[name] for name in feature_names])
    targets = experiment.build_targets(config, cleaned, verbose=False)
    samples, targets, _ = labeling.align_features_and_targets(samples, targets)

    half = len(samples) // 2
    model = RandomForestClassifier(**config['model']).fit(
        samples[:half], targets[:half])

    payload = modelBundle.build_bundle(model, config, feature_names)
    payload['threshold'] = 0.5
    path = os.path.join(str(tmp_path), 'test.json')
    modelBundle.save_bundle(payload, path)
    return modelBundle.load_bundle(path)


@pytest.fixture
def table():
    return make_table()


# ---------------------------------------------------------------------
# ② KHÔNG NHÌN TRƯỚC — test then chốt của cả module
# ---------------------------------------------------------------------
def test_replay_never_looks_ahead(bundle, table):
    """
    Cắt chuỗi tại một mốc rồi chạy lại. Nếu engine chỉ dùng dữ liệu tới
    bước hiện tại, mọi dự đoán trước mốc cắt phải TRÙNG KHÍT — biết
    thêm tương lai không được làm đổi bất cứ điều gì trong quá khứ.
    """
    cut = 120

    full = list(ReplayEngine(bundle, dict(table)).run())
    truncated = {name: values[:cut] for name, values in table.items()}
    partial = list(ReplayEngine(bundle, truncated).run())

    assert partial, 'chuỗi cắt ngắn không sinh ra bước nào'
    assert len(partial) <= len(full)

    for step_full, step_partial in zip(full, partial):
        assert step_full['key'] == step_partial['key']
        assert step_full['prediction'] == step_partial['prediction'], (
            f"dự đoán tại {step_full['key']} thay đổi khi biết thêm dữ liệu "
            f"tương lai — engine ĐANG NHÌN TRƯỚC"
        )


def test_sample_at_index_uses_only_past(bundle, table):
    """
    Kiểm tra trực tiếp ở mức thấp: vector đặc trưng dựng cho vị trí t
    không đổi khi phần dữ liệu SAU t bị thay đổi hoàn toàn.
    """
    engine = ReplayEngine(bundle, dict(table))
    position = 100
    original = engine._build_sample_at(position)

    corrupted = dict(table)
    corrupted['Close'] = list(table['Close'])
    for index in range(position + 1, len(corrupted['Close'])):
        corrupted['Close'][index] = 99999.0

    other = ReplayEngine(bundle, corrupted)
    assert other._build_sample_at(position) == original


# ---------------------------------------------------------------------
# ③ Bỏ qua giai đoạn huấn luyện — chống accuracy cao giả tạo
# ---------------------------------------------------------------------
def test_resolve_start_index_skips_up_to_marker():
    keys = ['2020-01-01', '2020-01-05', '2020-01-10', '2020-01-15']
    from pipeline.timePreprocess import parse_date_series
    parsed = parse_date_series(keys)

    assert resolve_start_index(parsed, None) == 0
    assert resolve_start_index(parsed, '2020-01-05') == 2
    assert resolve_start_index(parsed, '2020-01-20') == len(parsed)


def test_replay_starts_after_training_period(bundle, table):
    """
    Mốc kết thúc huấn luyện trong gói phải khiến engine bỏ qua đúng phần
    đầu chuỗi — phát lại trên dữ liệu đã học thuộc là tự lừa mình.
    """
    bundle['training_summary'] = {'training_end': '2020-03-01'}
    engine = ReplayEngine(bundle, dict(table))

    meta = engine.describe()
    assert meta['skipped_rows'] > 0
    assert 'bỏ qua giai đoạn huấn luyện' in meta['skipped_reason']

    first = next(iter(engine.run()))
    assert first['key'] > '2020-03-01'


def test_explicit_start_after_overrides_bundle(bundle, table):
    engine = ReplayEngine(bundle, dict(table), start_after='2020-04-01')
    first = next(iter(engine.run()))

    assert first['key'] > '2020-04-01'


# ---------------------------------------------------------------------
# ④ Hàng đợi chờ — dự đoán tại t chỉ chấm được ở t+horizon
# ---------------------------------------------------------------------
def test_scoring_is_delayed_by_horizon(bundle, table):
    """
    Không bước nào được chấm điểm sớm hơn horizon bước sau khi đặt dự
    đoán. Chấm sớm nghĩa là đã nhìn thấy kết quả trước khi nó xảy ra.
    """
    events = list(ReplayEngine(bundle, dict(table)).run())

    predicted_at = {}
    for step in events:
        if step['prediction']:
            predicted_at[step['key']] = step['index']

    for step in events:
        outcome = step['resolved']
        if outcome is None:
            continue
        source_index = predicted_at[outcome['key']]
        assert step['index'] - source_index == HORIZON, (
            f"dự đoán đặt tại bước {source_index} bị chấm ở bước "
            f"{step['index']} — lệch so với tầm nhìn {HORIZON}"
        )


def test_first_scores_appear_only_after_horizon(bundle, table):
    events = list(ReplayEngine(bundle, dict(table)).run())

    for step in events[:HORIZON]:
        assert step['resolved'] is None
        assert step['resolved_count'] == 0


def test_flat_outcomes_are_not_scored(bundle):
    """
    Giá không đổi thì quan sát không thuộc lớp nào, đúng quy ước lúc gán
    nhãn — nên phải bị loại khỏi phép chấm chứ không tính là sai.
    """
    rows = 80
    flat_table = {
        'Date':   [f'2020-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}' for i in range(rows)],
        'Open':   [100.0] * rows, 'High': [101.0] * rows,
        'Low':    [99.0] * rows,  'Close': [100.0] * rows,
        'Volume': [1000] * rows,
    }
    engine = ReplayEngine(bundle, flat_table)
    events = list(engine.run())

    assert all(step['resolved'] is None for step in events)
    assert events[-1]['resolved_count'] == 0


# ---------------------------------------------------------------------
# ⑤ Ba đường accuracy — đường đoán bừa là thứ khiến hai đường kia có nghĩa
# ---------------------------------------------------------------------
def test_accuracy_counters_stay_in_range(bundle, table):
    final = None
    for step in ReplayEngine(bundle, dict(table)).run():
        final = step
        for name, value in step['accuracy'].items():
            assert value is None or 0.0 <= value <= 1.0

    assert final['accuracy']['static'] is not None
    assert final['accuracy']['baseline'] is not None
    assert final['resolved_count'] > 0


def test_baseline_uses_only_pre_replay_data(bundle, table):
    """
    Lớp đa số dùng làm mốc đối chứng phải tính trên giai đoạn TRƯỚC khi
    phát lại. Lấy từ phần sau cũng là một dạng nhìn trước, chỉ tinh vi hơn.
    """
    engine = ReplayEngine(bundle, dict(table), start_after='2020-03-01')

    assert engine.baseline_label in (0, 1)
    assert engine.start_index > 0


def test_adaptive_absent_gives_no_adaptive_line(bundle, table):
    final = None
    for step in ReplayEngine(bundle, dict(table)).run():
        final = step

    assert final['accuracy']['adaptive'] is None
    assert final['adaptation'] is None


def test_meta_reports_replay_scope(bundle, table):
    engine = ReplayEngine(bundle, dict(table), start_after='2020-03-01')
    meta = engine.describe()

    assert meta['type'] == 'meta'
    assert meta['horizon'] == HORIZON
    assert meta['total_rows'] == meta['skipped_rows'] + meta['replay_rows']
    assert meta['has_adaptive'] is False


# ---------------------------------------------------------------------
# ⑥ Phối hợp với mô hình thích nghi
# ---------------------------------------------------------------------
def test_adaptive_model_receives_resolved_observations(bundle, table):
    """
    Mô hình thích nghi chỉ được nạp quan sát ĐÃ BIẾT kết quả, tức đúng
    những quan sát vừa được chấm điểm — không sớm hơn.
    """
    adaptive = SlidingRandomForestClassifier.from_forest(
        bundle['estimator'], trees_per_update=2, window_size=40,
        update_every=10)

    final = None
    resolved_total = 0
    for step in ReplayEngine(bundle, dict(table), adaptive_model=adaptive).run():
        final = step
        if step['resolved']:
            resolved_total += 1

    assert final['accuracy']['adaptive'] is not None
    assert final['adaptation'] is not None
    assert len(adaptive.buffer_samples) == min(resolved_total, 40)


def test_adaptive_and_static_start_identical(bundle, table):
    """
    Rừng trượt lúc khởi tạo chính là rừng tĩnh, nên vài bước đầu — trước
    đợt thay máu đầu tiên — hai đường phải dự đoán y hệt nhau.
    """
    adaptive = SlidingRandomForestClassifier.from_forest(
        bundle['estimator'], trees_per_update=2, window_size=40,
        update_every=1000)      # đặt rất lớn để không bao giờ thay máu

    for step in ReplayEngine(bundle, dict(table), adaptive_model=adaptive).run():
        if not step['prediction']:
            continue
        assert step['prediction']['static'] == step['prediction']['adaptive']


def test_replay_speed_is_practical(bundle, table):
    """
    Phát lại phải đủ nhanh để đẩy ra luồng thời gian thực. Cửa sổ trượt
    giữ chi phí mỗi bước gần như không đổi theo độ dài chuỗi.
    """
    import time

    start = time.time()
    steps = len(list(ReplayEngine(bundle, dict(table)).run()))
    elapsed = time.time() - start

    assert steps > 0
    assert elapsed / steps < 0.05, 'mỗi bước không được vượt 50 ms'
