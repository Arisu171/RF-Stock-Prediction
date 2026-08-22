# =====================================================================
# Model Bundle — gói mô hình hoàn chỉnh, dùng lại được sau khi lưu
# =====================================================================
# Module thuần Python, không phụ thuộc thư viện ngoài.
#
# ĐIỀU QUAN TRỌNG NHẤT CỦA MODULE NÀY:
#
# Tham số của rừng KHÔNG PHẢI toàn bộ mô hình. Đặc trưng của bài toán
# được SINH RA từ một bản đặc tả; nếu chỉ lưu cây mà không lưu bản đặc
# tả đó, thì khi nạp lại với cấu hình khác, mô hình vẫn chạy và vẫn cho
# ra số — nhưng cột thứ ba lúc huấn luyện là một chỉ báo, lúc dự đoán
# lại là chỉ báo khác. Sai lặng lẽ, không có thông báo lỗi nào.
#
# Vì vậy một gói mô hình gồm BA phần dính liền nhau:
#
#   1. Bản đặc tả đặc trưng + ánh xạ vai trò → tên cột  (công thức)
#   2. Tham số của rừng đã huấn luyện                    (tham số)
#   3. Siêu dữ liệu: dữ liệu nguồn, khoảng thời gian, chỉ số đạt được
#
# KHÁC BIỆT GIỮA ĐƯỜNG HUẤN LUYỆN VÀ ĐƯỜNG DỰ ĐOÁN — cũng dễ bỏ sót:
# experiment.prepare_dataset() loại bỏ những dòng KHÔNG CÓ NHÃN, tức
# đúng các quan sát mới nhất. Đó là việc cần thiết khi huấn luyện, nhưng
# lúc dự đoán thì chính những dòng đó mới là thứ ta cần. Hàm ③ dưới đây
# đi một đường riêng: dựng đặc trưng, bỏ dòng thiếu đặc trưng, nhưng
# GIỮ LẠI toàn bộ phần đuôi chưa có nhãn.
#
# Thứ tự khai báo bám đúng vòng đời của một mô hình:
#
#   ①  Đóng gói           — gộp công thức, tham số và siêu dữ liệu
#   ②  Ghi và đọc file    — vào/ra JSON
#   ③  Dựng đặc trưng để DỰ ĐOÁN — đường đi khác lúc huấn luyện
#   ④  Kiểm tra tương thích — chặn sai lệch công thức trước khi dự đoán
#   ⑤  Dự đoán từ gói     — gộp ③ và ④ thành một lời gọi
#   ⑥  Tóm tắt gói        — bản tin để in ra kiểm tra
# =====================================================================

import json
import os

from . import featureBuilder
from . import timePreprocess
from libraries import modelStore

BUNDLE_VERSION = 1


# ---------------------------------------------------------------------
# ① Đóng gói — buộc công thức đặc trưng đi cùng tham số mô hình
# ---------------------------------------------------------------------
def build_bundle(model, config, feature_names, training_summary=None,
                 metrics=None):
    """
    Gộp mô hình đã huấn luyện với mọi thứ cần để dùng lại nó.

    Parameters:
        model            : rừng đã huấn luyện
        config           : dict cấu hình đã dùng để huấn luyện
        feature_names    : list tên đặc trưng THEO ĐÚNG THỨ TỰ CỘT của
                           ma trận mẫu — đây là phần dễ sai nhất nếu
                           tách rời khỏi mô hình
        training_summary : dict mô tả dữ liệu huấn luyện (tuỳ chọn)
        metrics          : dict chỉ số đạt được lúc huấn luyện (tuỳ chọn)

    Returns:
        dict gói mô hình, ghi thẳng ra JSON được
    """
    if len(feature_names) != model.num_features:
        raise ValueError(
            f"Số tên đặc trưng ({len(feature_names)}) không khớp số đặc "
            f"trưng của mô hình ({model.num_features})."
        )

    return {
        'bundle_version': BUNDLE_VERSION,
        'description':    config.get('description', ''),
        'recipe': {
            'series':     dict(config['dataset']['series']),
            'numeric_columns': list(config['dataset']['numeric_columns']),
            'key_column': config['dataset']['key_column'],
            'preprocess': dict(config.get('preprocess', {})),
            'features':   list(config['features']),
            'labeling':   dict(config['labeling']),
        },
        'feature_names':    list(feature_names),
        'model':            modelStore.serialize_forest(model),
        'size':             modelStore.count_parameters(model),
        'training_summary': training_summary or {},
        'metrics':          metrics or {},
    }


# ---------------------------------------------------------------------
# ② Ghi và đọc — một gói mô hình là MỘT file, không phải nhiều mảnh
# ---------------------------------------------------------------------
def save_bundle(bundle, path):
    """
    Ghi gói mô hình ra file JSON.

    Returns:
        path : đường dẫn file đã ghi
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(bundle, handle, ensure_ascii=False, separators=(',', ':'))
    return path


def load_bundle(path):
    """
    Đọc gói mô hình từ file JSON và dựng lại đối tượng rừng.

    Returns:
        bundle : dict gói mô hình, bổ sung khoá 'estimator' là đối tượng
                 rừng đã sẵn sàng dự đoán
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy gói mô hình: {path}")

    with open(path, encoding='utf-8') as handle:
        bundle = json.load(handle)

    version = bundle.get('bundle_version')
    if version != BUNDLE_VERSION:
        raise ValueError(
            f"Gói mô hình phiên bản {version} không đọc được bằng phiên "
            f"bản {BUNDLE_VERSION}."
        )

    bundle['estimator'] = modelStore.deserialize_forest(bundle['model'])
    return bundle


# ---------------------------------------------------------------------
# ③ Dựng đặc trưng để DỰ ĐOÁN — giữ lại phần đuôi chưa có nhãn
# ---------------------------------------------------------------------
def prepare_features_for_prediction(recipe, table):
    """
    Làm sạch dữ liệu và dựng đặc trưng theo công thức trong gói mô hình.

    Khác với đường huấn luyện ở đúng một điểm, nhưng là điểm quyết định:
    hàm này KHÔNG sinh nhãn và KHÔNG loại bỏ phần đuôi. Quan sát mới
    nhất — thứ chưa thể có nhãn vì tương lai chưa xảy ra — chính là thứ
    ta muốn dự đoán.

    Parameters:
        recipe : dict công thức lấy từ gói mô hình (khoá 'recipe')
        table  : dict bảng dữ liệu thô, đã đọc bằng dataLoader

    Returns:
        samples       : list of lists — mẫu đặc trưng theo đúng thứ tự cột
        keys          : list mốc thời gian tương ứng từng mẫu
        feature_names : list tên đặc trưng theo thứ tự cột
    """
    key_column = recipe['key_column']

    for name in recipe['numeric_columns']:
        table[name] = [
            None if value is None or value == '' else float(value)
            for value in table[name]
        ]
    table[key_column] = timePreprocess.parse_date_series(table[key_column])

    table = timePreprocess.sort_table_by_column(table, key_column)
    table = timePreprocess.remove_duplicate_keys(table, key_column)
    for name in recipe['numeric_columns']:
        table[name] = timePreprocess.forward_fill_series(
            table[name], recipe['preprocess'].get('max_forward_fill', 2)
        )
    table, _ = timePreprocess.drop_rows_with_missing(table)

    source_series = {
        role: table[column] for role, column in recipe['series'].items()
    }
    feature_table, _ = featureBuilder.build_features(
        recipe['features'], source_series
    )
    feature_names = sorted(feature_table)

    keys = table[key_column]
    samples = []
    kept_keys = []
    for position in range(len(keys)):
        row = [feature_table[name][position] for name in feature_names]
        if any(value is None for value in row):
            continue
        samples.append(row)
        kept_keys.append(keys[position])

    return samples, kept_keys, feature_names


# ---------------------------------------------------------------------
# ④ Kiểm tra tương thích — chặn sai lặng lẽ trước khi nó xảy ra
# ---------------------------------------------------------------------
def verify_feature_compatibility(bundle, feature_names):
    """
    Đối chiếu danh sách đặc trưng vừa dựng với danh sách lúc huấn luyện.

    So sánh cả TÊN và THỨ TỰ. Chỉ cần hoán vị hai cột là mô hình sẽ đọc
    nhầm chỉ báo này thành chỉ báo kia mà không có dấu hiệu nào — nên
    phép kiểm tra này phải nghiêm ngặt, không được nới lỏng.

    Raises:
        ValueError kèm mô tả chính xác chỗ lệch
    """
    expected = bundle['feature_names']
    if feature_names == expected:
        return

    missing = [name for name in expected if name not in feature_names]
    extra = [name for name in feature_names if name not in expected]

    details = []
    if missing:
        details.append(f"thiếu {missing}")
    if extra:
        details.append(f"thừa {extra}")
    if not details:
        details.append('cùng tập tên nhưng KHÁC THỨ TỰ')

    raise ValueError(
        f"Đặc trưng không khớp với lúc huấn luyện: {'; '.join(details)}. "
        f"Mô hình chờ {len(expected)} cột, dữ liệu cho ra "
        f"{len(feature_names)} cột."
    )


# ---------------------------------------------------------------------
# ⑤ Dự đoán từ gói — gộp ③ và ④, đây là hàm được gọi nhiều nhất
# ---------------------------------------------------------------------
def predict_with_bundle(bundle, table, num_rows=None, threshold=None):
    """
    Dự đoán bằng gói mô hình đã nạp.

    Parameters:
        bundle    : dict gói mô hình đã qua load_bundle()
        table     : dict bảng dữ liệu thô cùng định dạng lúc huấn luyện
        num_rows  : chỉ trả về num_rows mốc thời gian gần nhất
                    (None = toàn bộ)
        threshold : ngưỡng quyết định cho bài toán phân loại
                    (None = dùng ngưỡng lưu trong gói, mặc định 0.5)

    Returns:
        list of dict — mỗi phần tử gồm 'key', 'prediction' và, với bài
        toán phân loại, thêm 'score' là xác suất của lớp dương
    """
    recipe = bundle['recipe']
    samples, keys, feature_names = prepare_features_for_prediction(recipe, table)
    verify_feature_compatibility(bundle, feature_names)

    if not samples:
        raise ValueError(
            "Không dựng được mẫu nào — dữ liệu quá ngắn so với cửa sổ "
            "dài nhất trong công thức đặc trưng."
        )

    if num_rows is not None:
        samples = samples[-num_rows:]
        keys = keys[-num_rows:]

    estimator = bundle['estimator']
    results = []

    if bundle['model']['task'] == 'classifier':
        positive_label = recipe['labeling'].get('positive_label',
                                                estimator.label_space[-1])
        negative_label = recipe['labeling'].get('negative_label',
                                                estimator.label_space[0])
        if threshold is None:
            threshold = bundle.get('threshold', 0.5)

        scores = estimator.predict_scores(samples, positive_label=positive_label)
        for key, score in zip(keys, scores):
            results.append({
                'key':        key,
                'score':      score,
                'prediction': positive_label if score >= threshold
                              else negative_label,
            })
    else:
        for key, value in zip(keys, estimator.predict(samples)):
            results.append({'key': key, 'prediction': value})

    return results


# ---------------------------------------------------------------------
# ⑥ Tóm tắt gói — để người dùng biết mình đang cầm mô hình nào
# ---------------------------------------------------------------------
def describe_bundle(bundle):
    """
    Tóm tắt nội dung gói mô hình thành chuỗi nhiều dòng.

    Returns:
        str
    """
    size = bundle['size']
    summary = bundle.get('training_summary', {})
    metrics = bundle.get('metrics', {})

    lines = [
        f"Mô tả        : {bundle.get('description', '(không có)')}",
        f"Bài toán     : {bundle['model']['task']}",
        f"Đặc trưng    : {len(bundle['feature_names'])} cột",
        f"Kích thước   : {size['num_trees']} cây, "
        f"{size['parameters']:,} tham số",
    ]

    if summary:
        lines.append(
            f"Huấn luyện   : {summary.get('num_samples', '?')} mẫu"
            + (f" | {summary.get('period', '')}" if summary.get('period') else '')
        )
        if summary.get('parameters_per_sample') is not None:
            lines.append(
                f"Tham số/mẫu  : {summary['parameters_per_sample']:.1f}"
                f"{'  (vượt 2 — mô hình đủ chỗ ghi nhớ nhiễu)' if summary['parameters_per_sample'] > 2 else ''}"
            )
    if metrics:
        lines.append('Chỉ số lúc huấn luyện:')
        for name, value in metrics.items():
            formatted = f'{value:.4f}' if isinstance(value, float) else str(value)
            lines.append(f"  {name:<22}: {formatted}")

    return '\n'.join(lines)
