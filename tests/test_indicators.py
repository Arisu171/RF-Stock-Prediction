# =====================================================================
# Test Indicators — kiểm thử bộ chỉ báo kỹ thuật
# =====================================================================
# Trọng tâm không phải "hàm có chạy không" mà là ba tính chất quyết định
# tính đúng đắn của toàn dự án:
#
#   1. NHÂN QUẢ    — giá trị tại t chỉ phụ thuộc dữ liệu ở các vị trí ≤ t
#   2. GIỮ ĐỘ DÀI  — mọi hàm trả về dãy cùng độ dài với đầu vào
#   3. ĐÚNG CÔNG THỨC — đối chiếu với giá trị tính tay
#
# Tính chất 1 là tuyến phòng thủ đầu tiên chống rò rỉ dữ liệu; nếu nó bị
# vi phạm thì mọi con số đánh giá phía sau đều vô nghĩa.
#
# Thứ tự khai báo:
#
#   ①  Dữ liệu mẫu dùng chung
#   ②  Kiểm thử tính nhân quả   — quan trọng nhất
#   ③  Kiểm thử giữ độ dài
#   ④  Kiểm thử đúng công thức
#   ⑤  Kiểm thử chịu ô thiếu
#   ⑥  Kiểm thử báo lỗi tham số sai
# =====================================================================

import pytest

from pipeline import technicalIndicators


# ---------------------------------------------------------------------
# ① Dữ liệu mẫu — một chuỗi giả lập đủ dài để mọi cửa sổ đều chạy được
# ---------------------------------------------------------------------
@pytest.fixture
def price_series():
    """Chuỗi giá giả lập có xu thế và dao động, không dùng số ngẫu nhiên."""
    return [
        100.0 + 10.0 * (index % 7) - 5.0 * (index % 3) + 0.2 * index
        for index in range(120)
    ]


@pytest.fixture
def bar_series(price_series):
    """Bộ ba cao / thấp / đóng cửa suy ra từ chuỗi giá."""
    high = [value + 2.0 + (index % 4) for index, value in enumerate(price_series)]
    low = [value - 2.0 - (index % 5) for index, value in enumerate(price_series)]
    return high, low, price_series


# ---------------------------------------------------------------------
# ② Nhân quả — phép thử quan trọng nhất của cả bộ kiểm thử
# ---------------------------------------------------------------------
SINGLE_INPUT_INDICATORS = [
    ('simple_moving_average',      {'window': 20}),
    ('exponential_moving_average', {'window': 20}),
    ('rolling_standard_deviation', {'window': 15}),
    ('relative_strength_index',    {'window': 14}),
    ('rate_of_change',             {'window': 10}),
    ('ratio_to_moving_average',    {'window': 20}),
    ('volume_ratio',               {'window': 20}),
    ('lag_series',                 {'lag': 3}),
]


@pytest.mark.parametrize('indicator_name,params', SINGLE_INPUT_INDICATORS)
def test_indicator_is_causal(price_series, indicator_name, params):
    """
    Cắt chuỗi tại một vị trí rồi tính lại chỉ báo trên đoạn đầu. Nếu hàm
    chỉ nhìn về quá khứ, giá trị tại mọi vị trí trước điểm cắt phải giống
    hệt giá trị tính trên chuỗi đầy đủ.
    """
    cut_position = 80
    function = getattr(technicalIndicators, indicator_name)

    on_full_series = function(price_series, **params)[:cut_position]
    on_partial_series = function(price_series[:cut_position], **params)

    for position, (full, partial) in enumerate(zip(on_full_series, on_partial_series)):
        if full is None or partial is None:
            assert full is None and partial is None, (
                f"{indicator_name}: lệch vùng khởi động tại vị trí {position}"
            )
        else:
            assert abs(full - partial) < 1e-9, (
                f"{indicator_name}: giá trị tại vị trí {position} thay đổi khi "
                f"biết thêm dữ liệu tương lai — chỉ báo KHÔNG nhân quả"
            )


def test_macd_is_causal(price_series):
    """MACD gồm ba dãy nên kiểm riêng."""
    cut_position = 90
    full = technicalIndicators.moving_average_convergence_divergence(price_series)
    partial = technicalIndicators.moving_average_convergence_divergence(
        price_series[:cut_position]
    )

    for full_line, partial_line in zip(full, partial):
        for position in range(cut_position):
            first, second = full_line[position], partial_line[position]
            if first is None or second is None:
                assert first is None and second is None
            else:
                assert abs(first - second) < 1e-9


def test_bollinger_bands_is_causal(price_series):
    """Dải Bollinger gồm năm dãy nên kiểm riêng."""
    cut_position = 70
    full = technicalIndicators.bollinger_bands(price_series, 20, 2.0)
    partial = technicalIndicators.bollinger_bands(price_series[:cut_position], 20, 2.0)

    for full_line, partial_line in zip(full, partial):
        for position in range(cut_position):
            first, second = full_line[position], partial_line[position]
            if first is None or second is None:
                assert first is None and second is None
            else:
                assert abs(first - second) < 1e-9


def test_lag_series_never_looks_forward(price_series):
    """Chuỗi trễ phải lấy đúng giá trị của quá khứ, không phải tương lai."""
    lagged = technicalIndicators.lag_series(price_series, 3)

    assert lagged[:3] == [None, None, None]
    for position in range(3, len(price_series)):
        assert lagged[position] == price_series[position - 3]


# ---------------------------------------------------------------------
# ③ Giữ độ dài — điều kiện để mọi dãy khớp chỉ số với nhau
# ---------------------------------------------------------------------
@pytest.mark.parametrize('indicator_name,params', SINGLE_INPUT_INDICATORS)
def test_indicator_preserves_length(price_series, indicator_name, params):
    """Dãy trả về phải cùng độ dài với dãy đầu vào."""
    function = getattr(technicalIndicators, indicator_name)
    assert len(function(price_series, **params)) == len(price_series)


def test_multi_output_indicators_preserve_length(bar_series):
    """Các chỉ báo trả về nhiều dãy cũng phải giữ nguyên độ dài."""
    high, low, close = bar_series

    for line in technicalIndicators.moving_average_convergence_divergence(close):
        assert len(line) == len(close)
    for line in technicalIndicators.bollinger_bands(close, 20, 2.0):
        assert len(line) == len(close)
    for line in technicalIndicators.stochastic_oscillator(high, low, close, 14, 3):
        assert len(line) == len(close)
    assert len(technicalIndicators.average_true_range(high, low, close, 14)) == len(close)


# ---------------------------------------------------------------------
# ④ Đúng công thức — đối chiếu với giá trị tính tay
# ---------------------------------------------------------------------
def test_simple_moving_average_matches_hand_calculation():
    """SMA(3) của [1,2,3,4,5] tại vị trí 2 là (1+2+3)/3 = 2."""
    result = technicalIndicators.simple_moving_average([1.0, 2.0, 3.0, 4.0, 5.0], 3)

    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_exponential_moving_average_seeds_with_simple_average():
    """Giá trị đầu tiên của EMA bằng SMA của n quan sát đầu."""
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = technicalIndicators.exponential_moving_average(series, 3)

    assert result[2] == pytest.approx(2.0)
    smoothing = 2.0 / (3 + 1)
    assert result[3] == pytest.approx(smoothing * 4.0 + (1 - smoothing) * 2.0)


def test_relative_strength_index_bounds():
    """RSI luôn nằm trong [0, 100]; chuỗi tăng đơn điệu cho RSI = 100."""
    increasing = [float(value) for value in range(1, 40)]
    result = technicalIndicators.relative_strength_index(increasing, 14)

    values = [value for value in result if value is not None]
    assert values, 'RSI không sinh được giá trị nào'
    assert all(0.0 <= value <= 100.0 for value in values)
    assert values[-1] == pytest.approx(100.0)


def test_rate_of_change_matches_definition():
    """ROC(2) tại vị trí 2 của [10,11,12] là (12-10)/10*100 = 20%."""
    result = technicalIndicators.rate_of_change([10.0, 11.0, 12.0, 15.0], 2)

    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(20.0)
    assert result[3] == pytest.approx((15.0 - 11.0) / 11.0 * 100.0)


def test_bollinger_percent_b_at_bands():
    """%B bằng 1 tại biên trên và bằng 0 tại biên dưới."""
    series = [10.0 + (index % 5) for index in range(60)]
    middle, upper, lower, percent_b, bandwidth = technicalIndicators.bollinger_bands(
        series, 20, 2.0)

    for position in range(len(series)):
        if percent_b[position] is None:
            continue
        expected = (series[position] - lower[position]) / (
            upper[position] - lower[position])
        assert percent_b[position] == pytest.approx(expected)
        assert bandwidth[position] == pytest.approx(
            (upper[position] - lower[position]) / middle[position])


def test_true_range_accounts_for_overnight_gap():
    """
    True Range phải bắt được khoảng nhảy giữa hai phiên, thứ mà hiệu
    cao-thấp trong phiên bỏ sót.
    """
    high = [10.0, 30.0]
    low = [9.0, 29.0]
    close = [9.5, 29.5]

    result = technicalIndicators.true_range(high, low, close)

    assert result[0] == pytest.approx(1.0)
    # |30 - 9.5| = 20.5 lớn hơn hẳn biên độ trong phiên là 1.0
    assert result[1] == pytest.approx(20.5)


def test_on_balance_volume_follows_price_direction():
    """OBV cộng khối lượng khi giá tăng và trừ khi giá giảm."""
    values = [10.0, 11.0, 10.5, 10.5, 12.0]
    volumes = [100.0, 200.0, 300.0, 400.0, 500.0]

    result = technicalIndicators.on_balance_volume(values, volumes)

    assert result == pytest.approx([0.0, 200.0, -100.0, -100.0, 400.0])


def test_crossover_indicator_is_binary():
    """Chỉ báo giao cắt chỉ nhận giá trị 0, 1 hoặc None."""
    fast = [1.0, 3.0, 2.0, None]
    slow = [2.0, 1.0, 2.0, 1.0]

    assert technicalIndicators.crossover_indicator(fast, slow) == [0, 1, 0, None]


# ---------------------------------------------------------------------
# ⑤ Chịu ô thiếu — đặc tả có thể xếp tầng nên đầu vào hay có None
# ---------------------------------------------------------------------
def test_indicators_tolerate_leading_missing_values():
    """Ô thiếu ở ĐẦU chuỗi chỉ làm lùi vùng khởi động, không gây lỗi."""
    series = [None] + [float(value) for value in range(1, 40)]

    for name, params in (('simple_moving_average', {'window': 5}),
                         ('exponential_moving_average', {'window': 5}),
                         ('rolling_standard_deviation', {'window': 5}),
                         ('relative_strength_index', {'window': 14})):
        result = getattr(technicalIndicators, name)(series, **params)
        assert len(result) == len(series)
        assert result[0] is None
        assert any(value is not None for value in result), (
            f'{name} không sinh được giá trị nào sau ô thiếu ở đầu'
        )


def test_window_indicators_skip_windows_containing_missing_values():
    """Cửa sổ chứa ô thiếu phải trả về None thay vì gây lỗi."""
    series = [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0]
    result = technicalIndicators.simple_moving_average(series, 3)

    assert result[2] is None and result[3] is None and result[4] is None
    assert result[5] == pytest.approx(5.0)


# ---------------------------------------------------------------------
# ⑥ Báo lỗi tham số sai — lỗi phải nổ sớm và có thông điệp rõ ràng
# ---------------------------------------------------------------------
def test_invalid_window_raises(price_series):
    """Cửa sổ không phải số nguyên dương phải báo lỗi ngay."""
    for invalid_window in (0, -5, 2.5):
        with pytest.raises(ValueError):
            technicalIndicators.simple_moving_average(price_series, invalid_window)


def test_invalid_lag_raises(price_series):
    """Độ trễ phải là số nguyên dương."""
    with pytest.raises(ValueError):
        technicalIndicators.lag_series(price_series, 0)


def test_macd_rejects_reversed_windows(price_series):
    """Cửa sổ nhanh phải nhỏ hơn cửa sổ chậm."""
    with pytest.raises(ValueError):
        technicalIndicators.moving_average_convergence_divergence(
            price_series, fast_window=26, slow_window=12)


def test_mismatched_lengths_raise():
    """Các dãy đưa vào cùng một chỉ báo phải khớp độ dài."""
    with pytest.raises(ValueError):
        technicalIndicators.ratio_series([1.0, 2.0, 3.0], [1.0, 2.0])


def test_build_feature_table_rejects_mismatched_lengths():
    """Ghép bảng đặc trưng phải phát hiện dãy lệch độ dài."""
    with pytest.raises(ValueError):
        technicalIndicators.build_feature_table({
            'first': [1.0, 2.0, 3.0],
            'second': [1.0, 2.0],
        })
