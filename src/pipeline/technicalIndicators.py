# =====================================================================
# Technical Indicators — bộ chỉ báo phân tích kỹ thuật
# =====================================================================
# Module thuần Python, không phụ thuộc thư viện ngoài.
#
# Mọi hàm chỉ nhận DÃY SỐ và tham số cửa sổ; không hàm nào tự đọc file,
# tự biết tên cột hay tự quyết định dùng chuỗi nào. Việc gắn dãy số cụ
# thể vào từng chỉ báo là việc của tầng notebook.
#
# HAI QUY ƯỚC XUYÊN SUỐT — cần nhớ trước khi đọc tiếp:
#
#   1. NHÂN QUẢ. Giá trị tại vị trí t chỉ được tính từ dữ liệu ở các vị
#      trí ≤ t. Không hàm nào nhìn về phía trước. Đây là tuyến phòng thủ
#      đầu tiên chống rò rỉ dữ liệu.
#   2. GIỮ NGUYÊN ĐỘ DÀI. Mọi hàm trả về dãy CÙNG ĐỘ DÀI với đầu vào;
#      vùng khởi động chưa đủ lịch sử được trả về None. Nhờ vậy các dãy
#      luôn khớp chỉ số với nhau và chỉ cần một lần dọn duy nhất bằng
#      timePreprocess.drop_rows_with_missing().
#
# Thứ tự khai báo đi từ khối xây dựng chung tới chỉ báo phức hợp:
#
#   ①–④   Khối nền      — trung bình trượt, độ lệch chuẩn trượt, cực trị
#   ⑤–⑦   Xu hướng      — MACD và vị thế giá so với đường trung bình
#   ⑧–⑪  Động lượng    — RSI, ROC, Stochastic, chuỗi trễ
#   ⑫–⑮  Biến động     — Bollinger, True Range, ATR, biên độ tương đối
#   ⑯–⑰  Khối lượng    — OBV và tỷ lệ khối lượng
#   ⑱–⑳  Biến đổi chung — tỷ số, giao cắt, ghép ma trận đặc trưng
# =====================================================================


# ---------------------------------------------------------------------
# ① SMA — khối nền đơn giản nhất, hầu hết chỉ báo bên dưới đều dùng lại
# ---------------------------------------------------------------------
def simple_moving_average(series, window):
    """
    Trung bình trượt đơn giản (Simple Moving Average).

        SMA_t(n) = (1/n) · Σ_{i=0}^{n-1} x_{t-i}

    Mọi quan sát trong cửa sổ có trọng số như nhau.

    Parameters:
        series : list giá trị
        window : độ dài cửa sổ n

    Returns:
        list cùng độ dài; n-1 phần tử đầu là None
    """
    _assert_window(window)
    result = [None] * len(series)
    if len(series) < window:
        return result

    if _has_missing(series):
        for index in range(window - 1, len(series)):
            window_values = series[index - window + 1:index + 1]
            if None not in window_values:
                result[index] = sum(window_values) / window
        return result

    window_sum = sum(series[:window])
    result[window - 1] = window_sum / window
    for index in range(window, len(series)):
        window_sum += series[index] - series[index - window]
        result[index] = window_sum / window
    return result


# ---------------------------------------------------------------------
# ② EMA — như ① nhưng ưu tiên dữ liệu mới, phản ứng nhanh hơn với biến động
# ---------------------------------------------------------------------
def exponential_moving_average(series, window):
    """
    Trung bình trượt luỹ thừa (Exponential Moving Average).

        α       = 2 / (n + 1)
        EMA_t   = α·x_t + (1-α)·EMA_{t-1}

    Trọng số giảm dần theo cấp số nhân về quá khứ nên EMA bám giá mới
    sát hơn SMA — đổi lại nhiễu hơn.

    Giá trị khởi tạo lấy bằng SMA của n quan sát đầu tiên.

    Returns:
        list cùng độ dài; n-1 phần tử đầu là None
    """
    _assert_window(window)
    result = [None] * len(series)
    if len(series) < window:
        return result

    offset = _valid_tail_offset(series)
    if offset is None:
        return result
    if offset:
        for position, value in enumerate(
            exponential_moving_average(series[offset:], window)
        ):
            result[offset + position] = value
        return result

    smoothing = 2.0 / (window + 1)
    previous = sum(series[:window]) / window
    result[window - 1] = previous

    for index in range(window, len(series)):
        previous = smoothing * series[index] + (1 - smoothing) * previous
        result[index] = previous
    return result


# ---------------------------------------------------------------------
# ③ Độ lệch chuẩn trượt — thước đo biến động, nguyên liệu của ⑫
# ---------------------------------------------------------------------
def rolling_standard_deviation(series, window):
    """
    Độ lệch chuẩn tổng thể trên cửa sổ trượt.

        σ_t(n) = √( (1/n) · Σ (x_{t-i} - x̄_t)² )

    Returns:
        list cùng độ dài; n-1 phần tử đầu là None
    """
    _assert_window(window)
    result = [None] * len(series)

    for index in range(window - 1, len(series)):
        window_values = series[index - window + 1:index + 1]
        if None in window_values:
            continue
        mean_value = sum(window_values) / window
        variance = sum(
            (value - mean_value) ** 2 for value in window_values
        ) / window
        result[index] = variance ** 0.5
    return result


# ---------------------------------------------------------------------
# ④ Cực trị trượt — cao nhất / thấp nhất trong cửa sổ, nguyên liệu của ⑩
# ---------------------------------------------------------------------
def rolling_extremes(series, window):
    """
    Giá trị lớn nhất và nhỏ nhất trên cửa sổ trượt.

    Returns:
        highest_values : list giá trị lớn nhất trong cửa sổ
        lowest_values  : list giá trị nhỏ nhất trong cửa sổ
        (cả hai cùng độ dài với series, n-1 phần tử đầu là None)
    """
    _assert_window(window)
    highest_values = [None] * len(series)
    lowest_values = [None] * len(series)

    for index in range(window - 1, len(series)):
        window_values = series[index - window + 1:index + 1]
        if None in window_values:
            continue
        highest_values[index] = max(window_values)
        lowest_values[index] = min(window_values)
    return highest_values, lowest_values


# ---------------------------------------------------------------------
# ⑤ MACD — hiệu hai đường ② với hai cửa sổ khác nhau: đo đà của xu hướng
# ---------------------------------------------------------------------
def moving_average_convergence_divergence(series, fast_window=12,
                                          slow_window=26, signal_window=9):
    """
    MACD — đo khoảng cách giữa một đường trung bình nhanh và một đường
    chậm, tức đo ĐÀ của xu hướng chứ không phải bản thân xu hướng.

        MACD_t     = EMA(n_fast)_t - EMA(n_slow)_t
        Signal_t   = EMA(n_signal) của chuỗi MACD
        Histogram_t = MACD_t - Signal_t

    Histogram đổi dấu là tín hiệu đà đảo chiều — thường được dùng làm
    đặc trưng vì nó đã chuẩn hoá sẵn quanh 0.

    Returns:
        macd_line, signal_line, histogram — ba list cùng độ dài với series
    """
    if fast_window >= slow_window:
        raise ValueError("fast_window phải nhỏ hơn slow_window.")

    fast_line = exponential_moving_average(series, fast_window)
    slow_line = exponential_moving_average(series, slow_window)

    macd_line = [
        None if fast is None or slow is None else fast - slow
        for fast, slow in zip(fast_line, slow_line)
    ]

    valid_from = _first_valid_position(macd_line)
    signal_line = [None] * len(series)
    histogram = [None] * len(series)

    if valid_from is not None:
        compact_signal = exponential_moving_average(
            macd_line[valid_from:], signal_window
        )
        for offset, value in enumerate(compact_signal):
            position = valid_from + offset
            signal_line[position] = value
            if value is not None and macd_line[position] is not None:
                histogram[position] = macd_line[position] - value

    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------
# ⑥ Vị thế so với trung bình — chuẩn hoá ① thành đặc trưng KHÔNG đơn vị
# ---------------------------------------------------------------------
def ratio_to_moving_average(series, window, use_exponential=False):
    """
    Tỷ lệ giữa giá trị hiện tại và đường trung bình trượt của chính nó.

        ratio_t = x_t / MA_t(n)

    Đây là dạng đặc trưng nên ưu tiên với mô hình dựa trên cây: giá trị
    dao động quanh 1.0 bất kể thang đo tuyệt đối, nên luật học được vẫn
    còn hiệu lực khi chuỗi trôi sang vùng giá trị mới — điều mà đặc
    trưng tuyệt đối không làm được vì cây không ngoại suy.

    Parameters:
        use_exponential : True → dùng EMA thay cho SMA

    Returns:
        list cùng độ dài với series
    """
    baseline = (exponential_moving_average(series, window) if use_exponential
                else simple_moving_average(series, window))
    return [
        None if average in (None, 0) else value / average
        for value, average in zip(series, baseline)
    ]


# ---------------------------------------------------------------------
# ⑦ Giao cắt — nén quan hệ giữa hai đường về một biến nhị phân 0/1
# ---------------------------------------------------------------------
def crossover_indicator(fast_series, slow_series):
    """
    Biến nhị phân: 1 khi đường nhanh nằm TRÊN đường chậm, ngược lại 0.

    Cây quyết định tự tìm được ngưỡng nên không nhất thiết cần biến nhị
    phân; giá trị của chỉ báo này nằm ở chỗ nó diễn đạt trực tiếp một
    quy tắc giao dịch quen thuộc, giúp đọc feature importance dễ hơn.

    Returns:
        list cùng độ dài, None ở vị trí một trong hai đường chưa có giá trị
    """
    return [
        None if fast is None or slow is None else (1 if fast > slow else 0)
        for fast, slow in zip(fast_series, slow_series)
    ]


# ---------------------------------------------------------------------
# ⑧ RSI — chuẩn hoá tương quan tăng/giảm về thang cố định 0–100
# ---------------------------------------------------------------------
def relative_strength_index(series, window=14):
    """
    Chỉ số sức mạnh tương đối (Relative Strength Index).

        RS_t  = trung bình mức TĂNG / trung bình mức GIẢM (n phiên)
        RSI_t = 100 - 100 / (1 + RS_t)

    Trung bình được làm mượt theo phương pháp Wilder:

        avg_t = ( avg_{t-1}·(n-1) + giá_trị_hiện_tại ) / n

    Thang đo cố định 0–100 với hai mốc quy ước 70 (quá mua) và 30 (quá
    bán) khiến RSI so sánh được giữa các chuỗi có thang giá khác nhau.

    Returns:
        list cùng độ dài; n phần tử đầu là None
    """
    _assert_window(window)
    result = [None] * len(series)
    if len(series) <= window:
        return result

    offset = _valid_tail_offset(series)
    if offset is None:
        return result
    if offset:
        for position, value in enumerate(
            relative_strength_index(series[offset:], window)
        ):
            result[offset + position] = value
        return result

    gains = []
    losses = []
    for index in range(1, len(series)):
        change = series[index] - series[index - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    average_gain = sum(gains[:window]) / window
    average_loss = sum(losses[:window]) / window
    result[window] = _relative_strength_to_index(average_gain, average_loss)

    for index in range(window, len(gains)):
        average_gain = (average_gain * (window - 1) + gains[index]) / window
        average_loss = (average_loss * (window - 1) + losses[index]) / window
        result[index + 1] = _relative_strength_to_index(average_gain, average_loss)

    return result


# ---------------------------------------------------------------------
# ⑨ ROC — động lượng thô: so hiện tại với chính mình n bước trước
# ---------------------------------------------------------------------
def rate_of_change(series, window=10):
    """
    Tốc độ thay đổi (Rate of Change), tính theo phần trăm.

        ROC_t(n) = (x_t - x_{t-n}) / x_{t-n} · 100

    Khác với ⑧, ROC không bị chặn trên nên giữ được thông tin về ĐỘ LỚN
    của đà tăng/giảm.

    Returns:
        list cùng độ dài; n phần tử đầu là None
    """
    _assert_window(window)
    result = [None] * len(series)
    for index in range(window, len(series)):
        past_value = series[index - window]
        if past_value and series[index] is not None:
            result[index] = (series[index] - past_value) / past_value * 100.0
    return result


# ---------------------------------------------------------------------
# ⑩ Stochastic — vị trí tương đối trong biên độ cửa sổ, dựa trên ④
# ---------------------------------------------------------------------
def stochastic_oscillator(high_series, low_series, close_series,
                          window=14, smooth_window=3):
    """
    Dao động ngẫu nhiên (Stochastic Oscillator).

        %K_t = (C_t - Lowest_t(n)) / (Highest_t(n) - Lowest_t(n)) · 100
        %D_t = SMA(%K, m)

    %K trả lời câu hỏi: giá đóng cửa đang nằm ở đâu trong biên độ dao
    động của n phiên gần nhất — gần đỉnh (→100) hay gần đáy (→0).

    Parameters:
        high_series   : list giá trị cao nhất từng phiên
        low_series    : list giá trị thấp nhất từng phiên
        close_series  : list giá trị đóng cửa
        window        : cửa sổ n để lấy cực trị
        smooth_window : cửa sổ m để làm mượt %K thành %D

    Returns:
        percent_k, percent_d — hai list cùng độ dài với close_series
    """
    _assert_equal_length(high_series, low_series, close_series)
    _assert_window(window)

    highest_values, _ = rolling_extremes(high_series, window)
    _, lowest_values = rolling_extremes(low_series, window)

    percent_k = [None] * len(close_series)
    for index in range(len(close_series)):
        highest = highest_values[index]
        lowest = lowest_values[index]
        if highest is None or lowest is None or highest == lowest:
            continue
        percent_k[index] = (
            (close_series[index] - lowest) / (highest - lowest) * 100.0
        )

    valid_from = _first_valid_position(percent_k)
    percent_d = [None] * len(close_series)
    if valid_from is not None:
        compact = simple_moving_average(percent_k[valid_from:], smooth_window)
        for offset, value in enumerate(compact):
            percent_d[valid_from + offset] = value

    return percent_k, percent_d


# ---------------------------------------------------------------------
# ⑪ Chuỗi trễ — đưa quá khứ vào cùng một dòng dữ liệu với hiện tại
# ---------------------------------------------------------------------
def lag_series(series, lag):
    """
    Dịch một dãy lùi về sau `lag` vị trí.

        lagged_t = x_{t-lag}

    Đây là cách duy nhất để mô hình phi chuỗi như cây quyết định "nhìn
    thấy" quá khứ: thông tin cũ phải được đưa thành đặc trưng của dòng
    hiện tại. Luôn dịch LÙI, không bao giờ dịch tiến — dịch tiến chính
    là rò rỉ dữ liệu tương lai.

    Returns:
        list cùng độ dài; lag phần tử đầu là None
    """
    if lag < 1:
        raise ValueError("Độ trễ phải là số nguyên dương.")
    return [None] * lag + list(series[:-lag])


# ---------------------------------------------------------------------
# ⑫ Bollinger Bands — dựng dải quanh ① với bề rộng tỷ lệ theo ③
# ---------------------------------------------------------------------
def bollinger_bands(series, window=20, num_standard_deviations=2.0):
    """
    Dải Bollinger — biên trên/dưới đặt cách đường trung bình một bội số
    của độ lệch chuẩn.

        Middle_t = SMA_t(n)
        Upper_t  = Middle_t + k·σ_t(n)
        Lower_t  = Middle_t - k·σ_t(n)
        %B_t     = (x_t - Lower_t) / (Upper_t - Lower_t)
        Width_t  = (Upper_t - Lower_t) / Middle_t

    Nên dùng %B và Width làm đặc trưng thay vì ba đường biên: hai đại
    lượng này không có đơn vị nên giữ được ý nghĩa khi thang giá đổi.

    Returns:
        middle_band, upper_band, lower_band, percent_b, bandwidth
        — năm list cùng độ dài với series
    """
    middle_band = simple_moving_average(series, window)
    deviations = rolling_standard_deviation(series, window)

    upper_band = [None] * len(series)
    lower_band = [None] * len(series)
    percent_b = [None] * len(series)
    bandwidth = [None] * len(series)

    for index in range(len(series)):
        middle = middle_band[index]
        deviation = deviations[index]
        if middle is None or deviation is None or series[index] is None:
            continue

        upper = middle + num_standard_deviations * deviation
        lower = middle - num_standard_deviations * deviation
        upper_band[index] = upper
        lower_band[index] = lower

        if upper != lower:
            percent_b[index] = (series[index] - lower) / (upper - lower)
        if middle:
            bandwidth[index] = (upper - lower) / middle

    return middle_band, upper_band, lower_band, percent_b, bandwidth


# ---------------------------------------------------------------------
# ⑬ True Range — biên độ thực của MỘT phiên, nguyên liệu của ⑭
# ---------------------------------------------------------------------
def true_range(high_series, low_series, close_series):
    """
    Biên độ thực (True Range) của từng phiên.

        TR_t = max( H_t - L_t,
                    |H_t - C_{t-1}|,
                    |L_t - C_{t-1}| )

    Hai vế sau tính cả khoảng nhảy giữa hai phiên, thứ mà hiệu H-L đơn
    thuần bỏ sót.

    Returns:
        list cùng độ dài; phần tử đầu tiên dùng H-L vì chưa có C_{t-1}
    """
    _assert_equal_length(high_series, low_series, close_series)

    result = [None] * len(close_series)
    if not close_series:
        return result

    result[0] = high_series[0] - low_series[0]
    for index in range(1, len(close_series)):
        previous_close = close_series[index - 1]
        if (previous_close is None or high_series[index] is None
                or low_series[index] is None):
            continue
        result[index] = max(
            high_series[index] - low_series[index],
            abs(high_series[index] - previous_close),
            abs(low_series[index] - previous_close),
        )
    return result


# ---------------------------------------------------------------------
# ⑭ ATR — làm mượt ⑬ theo Wilder, cho một thước đo biến động ổn định
# ---------------------------------------------------------------------
def average_true_range(high_series, low_series, close_series, window=14):
    """
    Biên độ thực trung bình (Average True Range).

        ATR_t = ( ATR_{t-1}·(n-1) + TR_t ) / n

    ATR đo BIÊN ĐỘ dao động chứ không đo hướng, nên thường dùng để
    chuẩn hoá các đại lượng khác hoặc đặt ngưỡng phân loại theo mức
    biến động thực tế thay vì một hằng số cố định.

    Returns:
        list cùng độ dài; n-1 phần tử đầu là None
    """
    _assert_window(window)
    ranges = true_range(high_series, low_series, close_series)

    result = [None] * len(close_series)
    if len(close_series) < window:
        return result

    previous = sum(ranges[:window]) / window
    result[window - 1] = previous
    for index in range(window, len(close_series)):
        previous = (previous * (window - 1) + ranges[index]) / window
        result[index] = previous
    return result


# ---------------------------------------------------------------------
# ⑮ Biên độ tương đối — bản rẻ tiền của ⑭, không cần làm mượt
# ---------------------------------------------------------------------
def relative_range(high_series, low_series, reference_series):
    """
    Biên độ trong phiên, chuẩn hoá theo một mốc tham chiếu.

        range_t = (H_t - L_t) / ref_t

    Returns:
        list cùng độ dài
    """
    _assert_equal_length(high_series, low_series, reference_series)
    return [
        None if not reference else (high - low) / reference
        for high, low, reference in zip(high_series, low_series, reference_series)
    ]


# ---------------------------------------------------------------------
# ⑯ OBV — cộng dồn khối lượng theo DẤU biến động, nối khối lượng với giá
# ---------------------------------------------------------------------
def on_balance_volume(value_series, volume_series):
    """
    Khối lượng cân bằng (On-Balance Volume).

        OBV_t = OBV_{t-1} + V_t   nếu x_t > x_{t-1}
              = OBV_{t-1} - V_t   nếu x_t < x_{t-1}
              = OBV_{t-1}         nếu bằng nhau

    Ý tưởng: khối lượng đi kèm chiều tăng là lực mua, đi kèm chiều giảm
    là lực bán. OBV là chuỗi cộng dồn nên có xu thế mạnh — khi dùng làm
    đặc trưng cho mô hình dựa trên cây, nên đưa về dạng tỷ lệ hoặc lấy
    tỷ suất biến động của chính OBV.

    Returns:
        list cùng độ dài; phần tử đầu tiên bằng 0
    """
    _assert_equal_length(value_series, volume_series)

    result = [None] * len(value_series)
    if not value_series:
        return result

    result[0] = 0.0
    for index in range(1, len(value_series)):
        previous = result[index - 1]
        if (previous is None or value_series[index] is None
                or value_series[index - 1] is None
                or volume_series[index] is None):
            continue
        if value_series[index] > value_series[index - 1]:
            result[index] = previous + volume_series[index]
        elif value_series[index] < value_series[index - 1]:
            result[index] = previous - volume_series[index]
        else:
            result[index] = previous
    return result


# ---------------------------------------------------------------------
# ⑰ Tỷ lệ khối lượng — chính là ⑥ áp cho chuỗi khối lượng, bắt đột biến
# ---------------------------------------------------------------------
def volume_ratio(volume_series, window=20):
    """
    Tỷ lệ giữa khối lượng hiện tại và khối lượng trung bình n phiên.

        ratio_t = V_t / SMA(V)_t(n)

    Giá trị lớn hơn 1 nhiều nghĩa là phiên có khối lượng đột biến — dấu
    hiệu thường đi kèm những chuyển động đáng chú ý.

    Returns:
        list cùng độ dài
    """
    return ratio_to_moving_average(volume_series, window)


# ---------------------------------------------------------------------
# ⑱ Tỷ số hai dãy — phép biến đổi chung để tạo đặc trưng không đơn vị
# ---------------------------------------------------------------------
def ratio_series(numerator_series, denominator_series):
    """
    Tỷ số theo từng vị trí giữa hai dãy.

    Returns:
        list cùng độ dài; None ở vị trí mẫu số bằng 0 hoặc thiếu dữ liệu
    """
    _assert_equal_length(numerator_series, denominator_series)
    return [
        None if numerator is None or not denominator else numerator / denominator
        for numerator, denominator in zip(numerator_series, denominator_series)
    ]


# ---------------------------------------------------------------------
# ⑲ Hiệu hai dãy — phép biến đổi chung còn lại, dùng cho các cặp đối xứng
# ---------------------------------------------------------------------
def difference_series(first_series, second_series):
    """
    Hiệu theo từng vị trí giữa hai dãy.

    Returns:
        list cùng độ dài; None ở vị trí thiếu dữ liệu
    """
    _assert_equal_length(first_series, second_series)
    return [
        None if first is None or second is None else first - second
        for first, second in zip(first_series, second_series)
    ]


# ---------------------------------------------------------------------
# ⑳ Ghép ma trận đặc trưng — gom các dãy đã tính thành bảng để đưa vào mô hình
# ---------------------------------------------------------------------
def build_feature_table(named_series):
    """
    Gom nhiều dãy đặc trưng thành một bảng, kiểm tra đồng bộ độ dài.

    Bước này chỉ ghép và kiểm tra; việc dọn vùng khởi động (các ô None ở
    đầu chuỗi) do timePreprocess.drop_rows_with_missing() đảm nhiệm, để
    mọi phép cắt dòng đều diễn ra ở đúng một chỗ duy nhất.

    Parameters:
        named_series : dict { tên_đặc_trưng: list giá trị }

    Returns:
        dict bảng đặc trưng (bản sao nông)

    Raises:
        ValueError nếu các dãy không cùng độ dài
    """
    if not named_series:
        raise ValueError("Cần ít nhất một dãy đặc trưng.")

    lengths = {len(values) for values in named_series.values()}
    if len(lengths) > 1:
        details = {name: len(values) for name, values in named_series.items()}
        raise ValueError(
            f"Các dãy đặc trưng phải cùng độ dài, nhận được: {details}"
        )
    return {name: list(values) for name, values in named_series.items()}


# ---------------------------------------------------------------------
# Kiểm tra và phép phụ dùng chung cho toàn module
# ---------------------------------------------------------------------
def _assert_window(window):
    """Cửa sổ trượt phải là số nguyên dương."""
    if not isinstance(window, int) or window < 1:
        raise ValueError("Độ dài cửa sổ phải là số nguyên dương.")


def _assert_equal_length(*series_list):
    """Các dãy đưa vào cùng một chỉ báo phải khớp chỉ số với nhau."""
    lengths = {len(series) for series in series_list}
    if len(lengths) > 1:
        raise ValueError(
            f"Các dãy phải có cùng độ dài, nhận được: {sorted(lengths)}"
        )


def _has_missing(series):
    """True nếu dãy còn ô thiếu — dùng để chọn nhánh tính nhanh hay an toàn."""
    return any(value is None for value in series)


def _valid_tail_offset(series):
    """
    Vị trí bắt đầu đoạn đuôi không còn ô thiếu.

    Trả về 0 nếu dãy vốn đã đủ giá trị (dùng nhánh tính đệ quy thẳng),
    hoặc None nếu dãy còn ô thiếu ở giữa — trường hợp đó các hàm đệ quy
    không xử lý được và sẽ để nguyên None.
    """
    first = _first_valid_position(series)
    if first is None:
        return None
    return first if not _has_missing(series[first:]) else None


def _first_valid_position(series):
    """Vị trí đầu tiên có giá trị khác None, hoặc None nếu dãy rỗng giá trị."""
    for index, value in enumerate(series):
        if value is not None:
            return index
    return None


def _relative_strength_to_index(average_gain, average_loss):
    """Quy đổi cặp (trung bình tăng, trung bình giảm) về thang RSI 0–100."""
    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0
    relative_strength = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)
