# =====================================================================
# Time Preprocess — tiền xử lý chuỗi dữ liệu theo thời gian
# =====================================================================
# Module thuần Python, không phụ thuộc thư viện ngoài.
#
# Mọi hàm chỉ nhận DÃY SỐ và TÊN CỘT do người gọi truyền vào, không tự
# quy định cột nào mang ý nghĩa gì. Nhờ vậy dùng lại được cho mọi chuỗi
# thời gian có cấu trúc bảng, không riêng một đề tài.
#
# Quy ước cấu trúc dữ liệu:
#   table  : dict { tên_cột: list giá trị } — đúng dạng dataLoader trả về
#   series : list giá trị của MỘT cột, ô thiếu mang giá trị None
#
# Thứ tự khai báo bám đúng trình tự làm sạch một chuỗi thời gian:
#
#   ①  Ép kiểu cột khoá thời gian        — điều kiện để so sánh, sắp xếp
#   ②  Sắp xếp toàn bảng theo cột khoá   — cần ①
#   ③  Khử khoá trùng lặp                — chỉ đúng sau khi đã sắp xếp ②
#   ④  Đếm ô thiếu                       — đo trước, xử lý sau
#   ⑤  Điền tiến có giới hạn             — vá lỗ nhỏ được phát hiện ở ④
#   ⑥  Bỏ dòng còn thiếu                 — dọn nốt phần ⑤ không vá được
#   ⑦  Tỷ suất biến động đơn giản        — chuyển mức giá trị sang mức thay đổi
#   ⑧  Tỷ suất biến động log             — biến thể cộng dồn được của ⑦
#   ⑨  Bắt biến động bất thường          — dùng ⑦ để lọc dữ liệu lỗi
#   ⑩  Kiểm tra tính liên tục của khoá   — phát hiện đứt quãng trong ①
#   ⑪  Cắt phần đầu chuỗi                — bỏ vùng chưa đủ lịch sử
#   ⑫  Tóm tắt thống kê một cột          — bản tin cuối cùng để in ra
# =====================================================================

import datetime
import math


# ---------------------------------------------------------------------
# ① Ép kiểu cột khoá — không so sánh hay sắp xếp được nếu còn là chuỗi
# ---------------------------------------------------------------------
def parse_date_series(series, date_formats=None):
    """
    Ép một cột về kiểu ngày.

    Chấp nhận sẵn các giá trị đã là datetime/date; với chuỗi thì thử lần
    lượt các định dạng cho tới khi khớp.

    Parameters:
        series       : list giá trị cần ép kiểu
        date_formats : list định dạng strptime để thử.
                       None → thử bộ định dạng phổ biến.

    Returns:
        list đối tượng datetime.date, ô không phân tích được trả về None
    """
    if date_formats is None:
        date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d-%m-%Y', '%Y-%m-%d %H:%M:%S',
        ]

    parsed = []
    for value in series:
        if value is None:
            parsed.append(None)
        elif isinstance(value, datetime.datetime):
            parsed.append(value.date())
        elif isinstance(value, datetime.date):
            parsed.append(value)
        else:
            text = str(value).strip()
            result = None
            for date_format in date_formats:
                try:
                    result = datetime.datetime.strptime(text, date_format).date()
                    break
                except ValueError:
                    continue
            parsed.append(result)
    return parsed


# ---------------------------------------------------------------------
# ② Sắp xếp toàn bảng — mọi cột phải di chuyển CÙNG NHAU để không lệch dòng
# ---------------------------------------------------------------------
def sort_table_by_column(table, key_column, ascending=True):
    """
    Sắp xếp toàn bộ bảng theo một cột khoá.

    Đây là bước bắt buộc với dữ liệu thời gian: thứ tự dòng mang ý
    nghĩa, và mọi phép tính cửa sổ trượt phía sau đều giả định chuỗi đã
    tăng dần theo thời gian.

    Parameters:
        table      : dict { tên_cột: list giá trị }
        key_column : tên cột dùng làm khoá sắp xếp
        ascending  : True = tăng dần

    Returns:
        dict bảng mới đã sắp xếp (không sửa bảng gốc)
    """
    _assert_column_exists(table, key_column)
    keys = table[key_column]

    order = sorted(
        (index for index in range(len(keys)) if keys[index] is not None),
        key=lambda index: keys[index],
        reverse=not ascending,
    )
    return {name: [values[index] for index in order] for name, values in table.items()}


# ---------------------------------------------------------------------
# ③ Khử khoá trùng — sau ② các dòng trùng nằm cạnh nhau nên chỉ cần quét một lượt
# ---------------------------------------------------------------------
def remove_duplicate_keys(table, key_column, keep='last'):
    """
    Loại các dòng có khoá trùng nhau, giữ lại một dòng duy nhất.

    Parameters:
        table      : dict bảng ĐÃ sắp xếp theo key_column
        key_column : tên cột khoá
        keep       : 'first' hoặc 'last' — giữ dòng nào trong nhóm trùng

    Returns:
        dict bảng mới
    """
    _assert_column_exists(table, key_column)
    if keep not in ('first', 'last'):
        raise ValueError("keep chỉ nhận 'first' hoặc 'last'.")

    keys = table[key_column]
    kept_index_of_key = {}
    for index, key in enumerate(keys):
        if key not in kept_index_of_key or keep == 'last':
            kept_index_of_key[key] = index

    order = sorted(kept_index_of_key.values())
    return {name: [values[index] for index in order] for name, values in table.items()}


# ---------------------------------------------------------------------
# ④ Đếm ô thiếu — đo mức độ hư hại trước khi chọn cách vá ở ⑤ hay ⑥
# ---------------------------------------------------------------------
def count_missing_values(table, columns=None):
    """
    Đếm số ô thiếu (None) của từng cột.

    Parameters:
        table   : dict bảng
        columns : list tên cột cần đếm (None = tất cả)

    Returns:
        dict { tên_cột: số ô thiếu }
    """
    columns = list(table) if columns is None else columns
    for name in columns:
        _assert_column_exists(table, name)
    return {
        name: sum(1 for value in table[name] if value is None)
        for name in columns
    }


# ---------------------------------------------------------------------
# ⑤ Điền tiến có giới hạn — vá lỗ nhỏ, KHÔNG bịa dữ liệu cho lỗ lớn
# ---------------------------------------------------------------------
def forward_fill_series(series, max_consecutive=2):
    """
    Điền tiến: ô thiếu nhận giá trị hợp lệ gần nhất phía TRƯỚC.

    Giới hạn max_consecutive là điểm mấu chốt: vá một hai ô lẻ thì chấp
    nhận được, nhưng kéo dài một giá trị cũ qua chục dòng liên tiếp là
    tự bịa ra dữ liệu và sẽ làm sai lệch mọi chỉ báo tính sau đó.

    Chỉ nhìn về quá khứ nên không gây rò rỉ dữ liệu tương lai.

    Parameters:
        series          : list giá trị, ô thiếu là None
        max_consecutive : số ô thiếu liên tiếp tối đa được phép điền

    Returns:
        list mới cùng độ dài, phần vượt giới hạn vẫn để None
    """
    filled = []
    last_valid = None
    run_length = 0

    for value in series:
        if value is not None:
            filled.append(value)
            last_valid = value
            run_length = 0
        elif last_valid is not None and run_length < max_consecutive:
            filled.append(last_valid)
            run_length += 1
        else:
            filled.append(None)
            run_length += 1
    return filled


# ---------------------------------------------------------------------
# ⑥ Bỏ dòng còn thiếu — cắt theo DÒNG để mọi cột giữ nguyên khớp chỉ số
# ---------------------------------------------------------------------
def drop_rows_with_missing(table, columns=None):
    """
    Loại bỏ theo DÒNG những mẫu còn ô thiếu.

    Cắt theo dòng chứ không theo cột là điều bắt buộc: nếu mỗi cột tự
    loại phần thiếu của riêng nó thì các cột sẽ lệch chỉ số và mọi mẫu
    phía sau đều sai.

    Hàm này cũng chính là bước dọn vùng khởi động của các chỉ báo cửa sổ
    trượt — những chỉ báo đó trả về None ở đầu chuỗi.

    Parameters:
        table   : dict bảng
        columns : list tên cột phải đủ giá trị (None = mọi cột)

    Returns:
        table_clean  : dict bảng mới
        kept_indices : list chỉ số dòng được giữ lại (so với bảng gốc)
    """
    columns = list(table) if columns is None else columns
    for name in columns:
        _assert_column_exists(table, name)

    num_rows = min(len(values) for values in table.values()) if table else 0
    kept_indices = [
        index for index in range(num_rows)
        if all(table[name][index] is not None for name in columns)
    ]
    table_clean = {
        name: [values[index] for index in kept_indices]
        for name, values in table.items()
    }
    return table_clean, kept_indices


# ---------------------------------------------------------------------
# ⑦ Tỷ suất đơn giản — rời thang GIÁ TRỊ TUYỆT ĐỐI sang thang THAY ĐỔI
# ---------------------------------------------------------------------
def calculate_simple_returns(series):
    """
    Tỷ suất biến động giữa hai quan sát liên tiếp.

        r_t = (x_t - x_{t-1}) / x_{t-1}

    Bước chuyển thang đo này rất quan trọng với mô hình dựa trên cây:
    cây không ngoại suy được ra ngoài khoảng giá trị đã thấy, nên đặc
    trưng dạng tỷ lệ tổng quát hoá tốt hơn hẳn giá trị tuyệt đối.

    Returns:
        list cùng độ dài, phần tử đầu tiên là None
    """
    returns = [None]
    for index in range(1, len(series)):
        previous = series[index - 1]
        current = series[index]
        if previous in (None, 0) or current is None:
            returns.append(None)
        else:
            returns.append((current - previous) / previous)
    return returns


# ---------------------------------------------------------------------
# ⑧ Tỷ suất log — biến thể của ⑦ có tính cộng dồn theo thời gian
# ---------------------------------------------------------------------
def calculate_log_returns(series):
    """
    Tỷ suất biến động dạng logarit.

        r_t = ln(x_t / x_{t-1})

    Ưu điểm so với ⑦: cộng dồn được (tổng các tỷ suất log qua nhiều
    bước bằng đúng tỷ suất log của cả đoạn) và phân phối cân đối hơn.

    Returns:
        list cùng độ dài, phần tử đầu tiên là None
    """
    returns = [None]
    for index in range(1, len(series)):
        previous = series[index - 1]
        current = series[index]
        if previous is None or current is None or previous <= 0 or current <= 0:
            returns.append(None)
        else:
            returns.append(math.log(current / previous))
    return returns


# ---------------------------------------------------------------------
# ⑨ Bắt biến động bất thường — dùng ⑦ để khoanh vùng dữ liệu nghi lỗi
# ---------------------------------------------------------------------
def detect_extreme_changes(series, threshold=0.40):
    """
    Tìm các vị trí có biến động vượt ngưỡng so với quan sát liền trước.

    Biến động quá lớn thường không phải tín hiệu thật mà là dấu vết của
    lỗi dữ liệu hoặc của một sự kiện làm gãy chuỗi. Hàm chỉ BÁO vị trí,
    việc quyết định giữ hay bỏ thuộc về người phân tích.

    Parameters:
        series    : list giá trị
        threshold : ngưỡng biến động tuyệt đối (0.40 = 40%)

    Returns:
        list of tuple (chỉ số, tỷ suất biến động)
    """
    returns = calculate_simple_returns(series)
    return [
        (index, value)
        for index, value in enumerate(returns)
        if value is not None and abs(value) > threshold
    ]


# ---------------------------------------------------------------------
# ⑩ Kiểm tra tính liên tục — chuỗi thời gian đứt quãng làm sai cửa sổ trượt
# ---------------------------------------------------------------------
def summarize_key_gaps(date_series):
    """
    Thống kê khoảng cách (số ngày) giữa các mốc thời gian liên tiếp.

    Với dữ liệu chỉ ghi nhận ngày làm việc, khoảng cách 1 và 3 ngày là
    bình thường; khoảng cách lớn bất thường cho thấy chuỗi bị thiếu đoạn
    và các chỉ báo cửa sổ trượt sẽ trộn lẫn hai giai đoạn cách xa nhau.

    Parameters:
        date_series : list đối tượng datetime.date đã sắp xếp tăng dần

    Returns:
        dict { 'gap_counts', 'largest_gap_days', 'largest_gap_position' }
    """
    gap_counts = {}
    largest_gap = 0
    largest_position = None

    for index in range(1, len(date_series)):
        previous = date_series[index - 1]
        current = date_series[index]
        if previous is None or current is None:
            continue
        gap = (current - previous).days
        gap_counts[gap] = gap_counts.get(gap, 0) + 1
        if gap > largest_gap:
            largest_gap = gap
            largest_position = index

    return {
        'gap_counts':           dict(sorted(gap_counts.items())),
        'largest_gap_days':     largest_gap,
        'largest_gap_position': largest_position,
    }


# ---------------------------------------------------------------------
# ⑪ Cắt đầu chuỗi — bỏ vùng mà cửa sổ trượt chưa đủ dữ liệu lịch sử
# ---------------------------------------------------------------------
def trim_leading_rows(table, num_rows):
    """
    Bỏ num_rows dòng đầu tiên của toàn bộ bảng.

    Parameters:
        table    : dict bảng
        num_rows : số dòng cần cắt

    Returns:
        dict bảng mới
    """
    if num_rows < 0:
        raise ValueError("Số dòng cần cắt không được âm.")
    return {name: values[num_rows:] for name, values in table.items()}


# ---------------------------------------------------------------------
# ⑫ Tóm tắt một cột — bản tin cuối, dùng để in ra và kiểm tra bằng mắt
# ---------------------------------------------------------------------
def describe_series(series):
    """
    Thống kê mô tả cho một cột số, bỏ qua ô thiếu.

    Returns:
        dict { 'count', 'missing', 'minimum', 'maximum', 'mean',
               'standard_deviation' } — các giá trị số là None nếu cột
        không còn giá trị hợp lệ nào
    """
    values = [value for value in series if value is not None]
    missing = len(series) - len(values)

    if not values:
        return {
            'count':              0,
            'missing':            missing,
            'minimum':            None,
            'maximum':            None,
            'mean':               None,
            'standard_deviation': None,
        }

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return {
        'count':              len(values),
        'missing':            missing,
        'minimum':            min(values),
        'maximum':            max(values),
        'mean':               mean_value,
        'standard_deviation': variance ** 0.5,
    }


# ---------------------------------------------------------------------
# ⑬ Ma trận tương quan — phát hiện các cột trùng lặp thông tin với nhau
# ---------------------------------------------------------------------
def calculate_correlation_matrix(table, columns=None):
    """
    Hệ số tương quan Pearson giữa từng cặp cột.

        r(x, y) = Σ(x-x̄)(y-ȳ) / √( Σ(x-x̄)² · Σ(y-ȳ)² )

    Cặp cột có |r| rất cao gần như mang cùng một thông tin. Với mô hình
    dựa trên cây, điều đó ít ảnh hưởng tới độ chính xác nhưng làm LOÃNG
    chỉ số tầm quan trọng: công trạng bị chia đều cho các cột tương
    đương, khiến cả nhóm cùng tụt hạng và dễ bị đánh giá thấp oan.

    Chỉ tính trên các dòng mà CẢ HAI cột đều có giá trị.

    Parameters:
        table   : dict bảng
        columns : list tên cột cần tính (None = tất cả)

    Returns:
        column_names : list tên cột theo đúng thứ tự dòng/cột của ma trận
        matrix       : list of lists hệ số tương quan
    """
    column_names = list(table) if columns is None else list(columns)
    for name in column_names:
        _assert_column_exists(table, name)

    size = len(column_names)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]

    for row_index in range(size):
        for column_index in range(row_index, size):
            value = _calculate_correlation(
                table[column_names[row_index]],
                table[column_names[column_index]],
            )
            matrix[row_index][column_index] = value
            matrix[column_index][row_index] = value
    return column_names, matrix


def _calculate_correlation(first_series, second_series):
    """Hệ số Pearson của hai dãy, bỏ qua các dòng thiếu giá trị."""
    pairs = [
        (first, second)
        for first, second in zip(first_series, second_series)
        if first is not None and second is not None
    ]
    if len(pairs) < 2:
        return 0.0

    count = len(pairs)
    first_mean = sum(pair[0] for pair in pairs) / count
    second_mean = sum(pair[1] for pair in pairs) / count

    covariance = sum(
        (first - first_mean) * (second - second_mean) for first, second in pairs
    )
    first_spread = sum((first - first_mean) ** 2 for first, _ in pairs)
    second_spread = sum((second - second_mean) ** 2 for _, second in pairs)

    denominator = math.sqrt(first_spread * second_spread)
    return covariance / denominator if denominator else 0.0


def find_highly_correlated_pairs(table, columns=None, threshold=0.95):
    """
    Liệt kê các cặp cột có tương quan tuyệt đối vượt ngưỡng.

    Returns:
        list of tuple (tên cột 1, tên cột 2, hệ số tương quan), sắp xếp
        giảm dần theo trị tuyệt đối
    """
    column_names, matrix = calculate_correlation_matrix(table, columns)
    pairs = []
    for row_index in range(len(column_names)):
        for column_index in range(row_index + 1, len(column_names)):
            value = matrix[row_index][column_index]
            if abs(value) > threshold:
                pairs.append(
                    (column_names[row_index], column_names[column_index], value)
                )
    return sorted(pairs, key=lambda item: abs(item[2]), reverse=True)


# ---------------------------------------------------------------------
# Kiểm tra dùng chung cho toàn module
# ---------------------------------------------------------------------
def _assert_column_exists(table, column_name):
    """Báo lỗi rõ ràng khi gọi nhầm tên cột."""
    if column_name not in table:
        raise KeyError(
            f"Không tìm thấy cột '{column_name}'. "
            f"Các cột hiện có: {list(table)}"
        )
