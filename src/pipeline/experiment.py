# =====================================================================
# Experiment — điều phối một lượt thí nghiệm từ BẢN CẤU HÌNH
# =====================================================================
# Module thuần Python cho phần điều phối; chỉ tầng vẽ đồ thị ở nơi khác
# mới cần thư viện ngoài.
#
# Module này ghép các bước rời rạc của tầng pipeline thành một mạch
# hoàn chỉnh: nạp → làm sạch → dựng đặc trưng → gán nhãn → tách tập.
# Toàn bộ giá trị cụ thể (đường dẫn, tên cột, độ dài cửa sổ, siêu tham
# số) đến từ tham số `config` do người gọi truyền vào — không một hằng
# số nào của đề tài nằm trong mã nguồn.
#
# Cấu trúc bản cấu hình: xem config/classification.json.
#
# Thứ tự khai báo bám đúng trình tự chạy một thí nghiệm:
#
#   ①  Nạp và ép kiểu       — đọc file, đưa các cột về dạng số và ngày
#   ②  Làm sạch chuỗi       — sắp xếp, khử trùng, điền tiến, bỏ dòng hỏng
#   ③  Dựng đặc trưng       — thi hành đặc tả trong cấu hình
#   ④  Gán nhãn và khớp     — sinh mục tiêu rồi cắt phần không có nhãn
#   ⑤  Gộp ①→④ thành một    — hàm mà hai script chạy gọi trực tiếp
#   ⑥  Tách tập theo thời gian
#   ⑦  Kiểm định walk-forward — chạy lại ⑥ nhiều lần theo cửa sổ tiến dần
#   ⑧  Đọc đường dẫn theo gốc dự án
# =====================================================================

import os

from . import featureBuilder
from . import labeling
from . import splitter
from . import timePreprocess
from utilities import dataLoader


# ---------------------------------------------------------------------
# ① Nạp và ép kiểu — mọi thứ phía sau đều giả định cột đã đúng kiểu
# ---------------------------------------------------------------------
def load_table(config, project_root='', verbose=True):
    """
    Đọc file dữ liệu và ép các cột về đúng kiểu.

    Parameters:
        config       : dict cấu hình, dùng nhánh 'dataset'
        project_root : thư mục gốc để giải đường dẫn tương đối
        verbose      : in thông tin file đã đọc

    Returns:
        dict bảng { tên_cột: list giá trị }
    """
    dataset = config['dataset']
    path = resolve_path(dataset['path'], project_root)

    extension = os.path.splitext(path)[1].lower()
    if extension == '.csv':
        _, table = dataLoader.load_csv_data(path, verbose=verbose)
    else:
        _, table = dataLoader.load_excel_data(path, verbose=verbose)

    for name in dataset['numeric_columns']:
        table[name] = [
            None if value is None or value == '' else float(value)
            for value in table[name]
        ]
    table[dataset['key_column']] = timePreprocess.parse_date_series(
        table[dataset['key_column']]
    )
    return table


# ---------------------------------------------------------------------
# ② Làm sạch — chạy đúng chuỗi ①→⑥ của timePreprocess theo cấu hình
# ---------------------------------------------------------------------
def clean_table(config, table, verbose=True):
    """
    Sắp xếp theo thời gian, khử khoá trùng, điền tiến và bỏ dòng hỏng.

    Returns:
        table   : dict bảng đã làm sạch
        report  : dict { 'num_rows_before', 'num_rows_after',
                         'missing_before', 'largest_gap_days',
                         'extreme_changes' }
    """
    dataset = config['dataset']
    settings = config.get('preprocess', {})
    key_column = dataset['key_column']

    num_rows_before = len(table[key_column])
    missing_before = timePreprocess.count_missing_values(table)

    table = timePreprocess.sort_table_by_column(table, key_column)
    table = timePreprocess.remove_duplicate_keys(table, key_column)

    for name in dataset['numeric_columns']:
        table[name] = timePreprocess.forward_fill_series(
            table[name], settings.get('max_forward_fill', 2)
        )
    table, _ = timePreprocess.drop_rows_with_missing(table)

    gaps = timePreprocess.summarize_key_gaps(table[key_column])
    reference_role = settings.get('extreme_change_series')
    extreme_changes = []
    if reference_role:
        reference_column = dataset['series'][reference_role]
        extreme_changes = timePreprocess.detect_extreme_changes(
            table[reference_column], settings.get('extreme_change_threshold', 0.40)
        )

    report = {
        'num_rows_before':  num_rows_before,
        'num_rows_after':   len(table[key_column]),
        'missing_before':   missing_before,
        'largest_gap_days': gaps['largest_gap_days'],
        'extreme_changes':  extreme_changes,
    }
    if verbose:
        print(f"Làm sạch: {report['num_rows_before']} → "
              f"{report['num_rows_after']} dòng | "
              f"gap lớn nhất {report['largest_gap_days']} ngày | "
              f"biến động bất thường: {len(extreme_changes)}")
    return table, report


# ---------------------------------------------------------------------
# ③ Dựng đặc trưng — dịch tên vai trò trong cấu hình thành dãy số thật
# ---------------------------------------------------------------------
def build_feature_table(config, table, verbose=False):
    """
    Thi hành đặc tả đặc trưng trong cấu hình.

    Bước dịch quan trọng: cấu hình nói tới các VAI TRÒ ('close', 'high',
    …), còn bảng dữ liệu dùng TÊN CỘT thật. Ánh xạ giữa hai thứ nằm ở
    config['dataset']['series'], nhờ vậy đổi nguồn dữ liệu có tên cột
    khác chỉ cần sửa cấu hình.

    Returns:
        feature_table : dict các đặc trưng sẽ đưa vào mô hình
        all_series    : dict toàn bộ dãy kể cả trung gian
    """
    source_series = {
        role: table[column]
        for role, column in config['dataset']['series'].items()
    }
    return featureBuilder.build_features(
        config['features'], source_series, verbose=verbose
    )


# ---------------------------------------------------------------------
# ④ Gán nhãn và khớp — sinh mục tiêu rồi cắt bỏ mọi dòng không dùng được
# ---------------------------------------------------------------------
def build_targets(config, table, verbose=True):
    """
    Sinh biến mục tiêu theo cấu hình.

    Hỗ trợ ba dạng, chọn bằng config['labeling']:
        - task='classification'                → nhãn chiều biến động
        - task='regression', target='future_return' → tỷ suất tương lai
        - task='regression', target='future_value'  → giá trị tương lai

    Returns:
        list mục tiêu, cùng độ dài với các cột của bảng
    """
    settings = config['labeling']
    column = config['dataset']['series'][settings['source_series']]
    series = table[column]
    horizon = settings.get('horizon', 1)

    if settings['task'] == 'classification':
        flat_summary = labeling.count_flat_observations(series, horizon)
        if verbose:
            print(f"Quan sát đứng yên: {flat_summary['flat']} "
                  f"({flat_summary['flat_ratio']:.2%})")
        return labeling.create_direction_labels(
            series,
            horizon=horizon,
            positive_label=settings.get('positive_label', 1),
            negative_label=settings.get('negative_label', 0),
            flat_label=settings.get('flat_label'),
        )

    if settings['task'] == 'regression':
        if settings.get('target', 'future_return') == 'future_value':
            return labeling.create_future_values(series, horizon)
        return labeling.create_future_returns(series, horizon)

    raise ValueError(
        f"task='{settings['task']}' không hợp lệ — dùng "
        f"'classification' hoặc 'regression'."
    )


# ---------------------------------------------------------------------
# ⑤ Chuẩn bị trọn bộ — gộp ①→④, đây là hàm hai script chạy gọi trực tiếp
# ---------------------------------------------------------------------
def prepare_dataset(config, project_root='', verbose=True):
    """
    Chạy trọn mạch nạp → làm sạch → đặc trưng → nhãn → khớp.

    Returns:
        dict {
            'samples':       list of lists,
            'targets':       list,
            'keys':          list mốc thời gian tương ứng từng mẫu,
            'feature_names': list tên đặc trưng theo đúng thứ tự cột,
            'table':         bảng đã làm sạch,
            'all_series':    toàn bộ dãy kể cả trung gian,
            'clean_report':  báo cáo bước làm sạch,
        }
    """
    from libraries import rfMath

    table = load_table(config, project_root, verbose)
    table, clean_report = clean_table(config, table, verbose)

    feature_table, all_series = build_feature_table(config, table)
    feature_names = sorted(feature_table)
    samples = rfMath.columns_to_samples(
        [feature_table[name] for name in feature_names]
    )

    targets = build_targets(config, table, verbose)
    samples, targets, kept_indices = labeling.align_features_and_targets(
        samples, targets
    )
    keys = [table[config['dataset']['key_column']][index] for index in kept_indices]

    if verbose:
        print(f"Đặc trưng: {len(feature_names)} | "
              f"mẫu dùng được: {len(samples)} "
              f"(bỏ {clean_report['num_rows_after'] - len(samples)} dòng)")

    return {
        'samples':       samples,
        'targets':       targets,
        'keys':          keys,
        'feature_names': feature_names,
        'table':         table,
        'all_series':    all_series,
        'clean_report':  clean_report,
    }


# ---------------------------------------------------------------------
# ⑥ Tách tập — áp cấu hình 'split' lên bộ dữ liệu đã chuẩn bị ở ⑤
# ---------------------------------------------------------------------
def split_prepared_dataset(config, dataset, verbose=True):
    """
    Tách bộ dữ liệu thành train/validation/test theo thứ tự thời gian.

    Returns:
        dict như splitter.split_dataset(), bổ sung khoá 'summary'
    """
    settings = config['split']
    parts = splitter.split_dataset(
        dataset['samples'], dataset['targets'],
        train_ratio=settings.get('train_ratio', 0.70),
        validation_ratio=settings.get('validation_ratio', 0.15),
        gap=settings.get('gap', 0),
    )
    parts['summary'] = splitter.describe_split(
        *parts['indices'], key_series=dataset['keys']
    )

    if verbose:
        for name, info in parts['summary'].items():
            period = info.get('key_range')
            period_text = f"{period[0]} → {period[1]}" if period else ''
            print(f"  {name:<11} n = {info['size']:>5}  "
                  f"({info['ratio']:.1%})  {period_text}")
    return parts


# ---------------------------------------------------------------------
# ⑦ Walk-forward — lặp lại ⑥ theo cửa sổ tiến dần để đo ĐỘ ỔN ĐỊNH
# ---------------------------------------------------------------------
def run_walk_forward(config, samples, targets, model_factory, score_function,
                     verbose=True):
    """
    Kiểm định tiến dần: mỗi vòng huấn luyện lại từ đầu trên phần quá khứ
    và chấm điểm trên đoạn kế tiếp chưa từng thấy.

    Một con số trung bình đẹp chưa đủ — phải xem cả độ lệch chuẩn. Mô
    hình dao động mạnh giữa các vòng là mô hình không dùng được, dù
    trung bình có cao.

    Parameters:
        config         : dict cấu hình, dùng nhánh 'evaluation' và 'split'
        samples        : list of lists — phần dữ liệu dành cho kiểm định
        targets        : list mục tiêu tương ứng
        model_factory  : hàm không tham số, trả về một mô hình CHƯA huấn luyện
        score_function : hàm (y_true, y_pred) -> float
        verbose        : in kết quả từng vòng

    Returns:
        dict { 'fold_scores', 'fold_sizes', 'mean', 'standard_deviation' }
    """
    settings = config.get('evaluation', {})
    folds = splitter.generate_expanding_window_folds(
        len(samples),
        num_folds=settings.get('walk_forward_folds', 5),
        gap=config['split'].get('gap', 0),
    )

    fold_scores = []
    fold_sizes = []
    for position, (train_indices, validation_indices) in enumerate(folds, start=1):
        model = model_factory()
        model.fit(
            [samples[index] for index in train_indices],
            [targets[index] for index in train_indices],
        )
        actual = [targets[index] for index in validation_indices]
        predicted = model.predict([samples[index] for index in validation_indices])

        score = score_function(actual, predicted)
        fold_scores.append(score)
        fold_sizes.append((len(train_indices), len(validation_indices)))

        if verbose:
            print(f"  vòng {position}: train = {len(train_indices):>5}  "
                  f"validate = {len(validation_indices):>4}  điểm = {score:.4f}")

    mean_score = sum(fold_scores) / len(fold_scores)
    variance = sum(
        (score - mean_score) ** 2 for score in fold_scores
    ) / len(fold_scores)
    return {
        'fold_scores':        fold_scores,
        'fold_sizes':         fold_sizes,
        'mean':               mean_score,
        'standard_deviation': variance ** 0.5,
    }


# ---------------------------------------------------------------------
# ⑧ Đường dẫn — quy mọi đường dẫn tương đối về gốc dự án
# ---------------------------------------------------------------------
def resolve_path(path, project_root=''):
    """
    Ghép đường dẫn tương đối vào thư mục gốc dự án.

    Đường dẫn tuyệt đối được giữ nguyên, nhờ vậy cấu hình chạy được cả
    khi gọi từ thư mục khác.
    """
    if os.path.isabs(path) or not project_root:
        return path
    return os.path.join(project_root, path)
