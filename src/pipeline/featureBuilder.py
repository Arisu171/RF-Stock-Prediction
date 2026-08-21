# =====================================================================
# Feature Builder — dựng bảng đặc trưng từ một BẢN ĐẶC TẢ khai báo
# =====================================================================
# Module thuần Python, không phụ thuộc thư viện ngoài.
#
# VÌ SAO CẦN TẦNG NÀY. Danh sách chỉ báo, độ dài cửa sổ và tên cột là
# những thứ thay đổi theo từng đề tài. Nếu viết thẳng vào mã nguồn thì
# mỗi lần đổi đề tài lại phải sửa mã. Ở đây chúng được mô tả bằng DỮ
# LIỆU (một danh sách dict, thường đọc từ file cấu hình hoặc khai báo
# trong notebook), còn module này chỉ biết cách THI HÀNH bản đặc tả đó.
#
# Nhờ vậy không một tên cột hay hằng số cụ thể nào của đề tài xuất hiện
# trong mã nguồn.
#
# CẤU TRÚC MỘT MỤC ĐẶC TẢ:
#
#   {
#     "name":      "close_over_sma20",     # tên đặc trưng sinh ra
#     "indicator": "ratio_to_moving_average",
#     "inputs":    ["close"],              # tên dãy nguồn
#     "params":    {"window": 20},         # tham số của chỉ báo
#     "keep":      true                    # có đưa vào mô hình không
#   }
#
# Với chỉ báo trả về NHIỀU dãy, dùng "outputs" để đặt tên từng dãy:
#
#   {
#     "name": "macd", "indicator": "macd", "inputs": ["close"],
#     "params": {"fast_window": 12, "slow_window": 26, "signal_window": 9},
#     "outputs": {"line": "macd_line", "histogram": "macd_histogram"},
#     "keep": false
#   }
#
# Tên trong "inputs" được tra lần lượt: trước hết trong các dãy gốc,
# sau đó trong các dãy đã dựng ở những mục TRƯỚC ĐÓ. Nhờ vậy đặc tả có
# thể xếp tầng — một chỉ báo lấy đầu ra của chỉ báo khác làm đầu vào.
#
# Thứ tự khai báo:
#
#   ①  Sổ đăng ký chỉ báo   — bảng tra tên → hàm, khai báo một lần
#   ②  Thi hành một mục     — đơn vị nhỏ nhất của quá trình dựng
#   ③  Thi hành cả đặc tả   — lặp ② theo thứ tự, tích luỹ kết quả
#   ④  Liệt kê chỉ báo      — tra cứu khi viết đặc tả
# =====================================================================

from . import technicalIndicators
from . import timePreprocess


# ---------------------------------------------------------------------
# ① Sổ đăng ký — nơi DUY NHẤT nối tên trong đặc tả với hàm thực thi
# ---------------------------------------------------------------------
# Mỗi mục mô tả: hàm cần gọi, số dãy đầu vào, và tên các dãy đầu ra
# theo đúng thứ tự hàm trả về (None = hàm chỉ trả về một dãy).
INDICATOR_REGISTRY = {
    # ── Khối nền ────────────────────────────────────────────────────
    'simple_moving_average': {
        'function': technicalIndicators.simple_moving_average,
        'num_inputs': 1, 'outputs': None,
    },
    'exponential_moving_average': {
        'function': technicalIndicators.exponential_moving_average,
        'num_inputs': 1, 'outputs': None,
    },
    'rolling_standard_deviation': {
        'function': technicalIndicators.rolling_standard_deviation,
        'num_inputs': 1, 'outputs': None,
    },
    'rolling_extremes': {
        'function': technicalIndicators.rolling_extremes,
        'num_inputs': 1, 'outputs': ['highest', 'lowest'],
    },

    # ── Xu hướng ────────────────────────────────────────────────────
    'macd': {
        'function': technicalIndicators.moving_average_convergence_divergence,
        'num_inputs': 1, 'outputs': ['line', 'signal', 'histogram'],
    },
    'ratio_to_moving_average': {
        'function': technicalIndicators.ratio_to_moving_average,
        'num_inputs': 1, 'outputs': None,
    },
    'crossover_indicator': {
        'function': technicalIndicators.crossover_indicator,
        'num_inputs': 2, 'outputs': None,
    },

    # ── Động lượng ──────────────────────────────────────────────────
    'relative_strength_index': {
        'function': technicalIndicators.relative_strength_index,
        'num_inputs': 1, 'outputs': None,
    },
    'rate_of_change': {
        'function': technicalIndicators.rate_of_change,
        'num_inputs': 1, 'outputs': None,
    },
    'stochastic_oscillator': {
        'function': technicalIndicators.stochastic_oscillator,
        'num_inputs': 3, 'outputs': ['percent_k', 'percent_d'],
    },
    'lag_series': {
        'function': technicalIndicators.lag_series,
        'num_inputs': 1, 'outputs': None,
    },

    # ── Biến động ───────────────────────────────────────────────────
    'bollinger_bands': {
        'function': technicalIndicators.bollinger_bands,
        'num_inputs': 1,
        'outputs': ['middle', 'upper', 'lower', 'percent_b', 'bandwidth'],
    },
    'true_range': {
        'function': technicalIndicators.true_range,
        'num_inputs': 3, 'outputs': None,
    },
    'average_true_range': {
        'function': technicalIndicators.average_true_range,
        'num_inputs': 3, 'outputs': None,
    },
    'relative_range': {
        'function': technicalIndicators.relative_range,
        'num_inputs': 3, 'outputs': None,
    },

    # ── Khối lượng ──────────────────────────────────────────────────
    'on_balance_volume': {
        'function': technicalIndicators.on_balance_volume,
        'num_inputs': 2, 'outputs': None,
    },
    'volume_ratio': {
        'function': technicalIndicators.volume_ratio,
        'num_inputs': 1, 'outputs': None,
    },

    # ── Biến đổi chung ──────────────────────────────────────────────
    'ratio_series': {
        'function': technicalIndicators.ratio_series,
        'num_inputs': 2, 'outputs': None,
    },
    'difference_series': {
        'function': technicalIndicators.difference_series,
        'num_inputs': 2, 'outputs': None,
    },
    'simple_returns': {
        'function': timePreprocess.calculate_simple_returns,
        'num_inputs': 1, 'outputs': None,
    },
    'log_returns': {
        'function': timePreprocess.calculate_log_returns,
        'num_inputs': 1, 'outputs': None,
    },
}


# ---------------------------------------------------------------------
# ② Thi hành một mục — tra sổ ①, gọi hàm, đặt tên cho các dãy sinh ra
# ---------------------------------------------------------------------
def apply_specification(specification, available_series):
    """
    Thi hành đúng MỘT mục đặc tả.

    Parameters:
        specification    : dict một mục đặc tả (xem đầu module)
        available_series : dict { tên_dãy: list giá trị } — các dãy đã có

    Returns:
        dict { tên_dãy_mới: list giá trị }

    Raises:
        KeyError   nếu chỉ báo hoặc dãy đầu vào không tồn tại
        ValueError nếu số dãy đầu vào không khớp với chỉ báo
    """
    indicator_name = specification.get('indicator')
    if indicator_name not in INDICATOR_REGISTRY:
        raise KeyError(
            f"Chỉ báo '{indicator_name}' chưa được đăng ký. "
            f"Các chỉ báo hiện có: {sorted(INDICATOR_REGISTRY)}"
        )

    entry = INDICATOR_REGISTRY[indicator_name]
    input_names = specification.get('inputs', [])
    if len(input_names) != entry['num_inputs']:
        raise ValueError(
            f"Chỉ báo '{indicator_name}' cần {entry['num_inputs']} dãy đầu vào, "
            f"đặc tả cung cấp {len(input_names)}."
        )

    arguments = []
    for name in input_names:
        if name not in available_series:
            raise KeyError(
                f"Dãy đầu vào '{name}' chưa tồn tại tại thời điểm dựng "
                f"'{specification.get('name')}'. Các dãy đang có: "
                f"{sorted(available_series)}"
            )
        arguments.append(available_series[name])

    produced = entry['function'](*arguments, **specification.get('params', {}))

    if entry['outputs'] is None:
        return {specification['name']: produced}

    output_names = specification.get('outputs')
    if not output_names:
        raise ValueError(
            f"Chỉ báo '{indicator_name}' trả về nhiều dãy "
            f"({entry['outputs']}), đặc tả phải khai báo mục 'outputs'."
        )

    result = {}
    for position, slot in enumerate(entry['outputs']):
        if slot in output_names:
            result[output_names[slot]] = produced[position]
    return result


# ---------------------------------------------------------------------
# ③ Thi hành cả đặc tả — lặp ② theo thứ tự, mục sau dùng được kết quả mục trước
# ---------------------------------------------------------------------
def build_features(specifications, source_series, verbose=False):
    """
    Dựng toàn bộ bảng đặc trưng theo đặc tả.

    Thứ tự các mục trong danh sách CÓ Ý NGHĨA: một mục chỉ tham chiếu
    được tới các dãy gốc và các dãy do những mục đứng trước sinh ra.

    Parameters:
        specifications : list các mục đặc tả
        source_series  : dict { tên_dãy_gốc: list giá trị } — các cột dữ
                         liệu đã làm sạch, do người gọi đặt tên
        verbose        : in tiến trình dựng từng mục

    Returns:
        feature_table : dict chỉ gồm các dãy có keep=True (mặc định True)
        all_series    : dict toàn bộ dãy, kể cả trung gian — hữu ích khi
                        cần vẽ lại chỉ báo hoặc kiểm tra từng bước
    """
    all_series = dict(source_series)
    kept_names = []

    for specification in specifications:
        if 'name' not in specification:
            raise ValueError("Mỗi mục đặc tả phải có khoá 'name'.")

        produced = apply_specification(specification, all_series)
        all_series.update(produced)

        if specification.get('keep', True):
            kept_names.extend(produced)

        if verbose:
            print(f"  {specification['name']:<24} "
                  f"({specification['indicator']}) → {sorted(produced)}")

    feature_table = {name: all_series[name] for name in kept_names}
    if not feature_table:
        raise ValueError(
            "Đặc tả không sinh ra đặc trưng nào — kiểm tra lại cờ 'keep'."
        )
    return feature_table, all_series


# ---------------------------------------------------------------------
# ④ Liệt kê chỉ báo — tra cứu nhanh khi viết đặc tả trong notebook
# ---------------------------------------------------------------------
def describe_registry():
    """
    Liệt kê các chỉ báo đã đăng ký cùng số dãy đầu vào và tên đầu ra.

    Returns:
        list of dict { 'indicator', 'num_inputs', 'outputs' }, sắp xếp
        theo tên chỉ báo
    """
    return [
        {
            'indicator':  name,
            'num_inputs': entry['num_inputs'],
            'outputs':    entry['outputs'],
        }
        for name, entry in sorted(INDICATOR_REGISTRY.items())
    ]
