# =====================================================================
# Labeling — sinh biến mục tiêu từ một chuỗi thời gian
# =====================================================================
# Module thuần Python, không phụ thuộc thư viện ngoài.
#
# ĐÂY LÀ NƠI DỄ GÂY RÒ RỈ DỮ LIỆU NHẤT CỦA TOÀN DỰ ÁN.
#
# Nguyên tắc bất di bất dịch: DỊCH NHÃN VỀ SAU, KHÔNG DỊCH ĐẶC TRƯNG VỀ
# TRƯỚC. Đặc trưng tại vị trí t giữ nguyên; nhãn tại t được lấy từ giá
# trị ở t+horizon. Cách làm ngược lại — dịch cột đặc trưng lùi lại — sẽ
# nhét thông tin tương lai vào đặc trưng và cho ra kết quả tốt giả tạo.
#
# Hệ quả: horizon phần tử CUỐI dãy không có nhãn và mang giá trị None.
# Chúng phải bị loại trước khi huấn luyện (xem ⑤).
#
# Thứ tự khai báo bám đúng trình tự tạo mục tiêu:
#
#   ①  Giá trị tương lai      — dạng mục tiêu thô nhất, dùng cho hồi quy
#   ②  Tỷ suất tương lai      — chuẩn hoá ① về thang không đơn vị
#   ③  Nhãn hai lớp           — lấy DẤU của ②, dùng cho phân loại
#   ③b Đếm quan sát đứng yên  — con số quyết định cách cấu hình ③
#   ④  Nhãn ba lớp            — như ③ nhưng có thêm vùng trung tính
#   ⑤  Khớp đặc trưng ↔ nhãn  — cắt bỏ phần đuôi không có nhãn
#   ⑥  Cân bằng lớp           — kiểm tra trước khi huấn luyện
# =====================================================================


# ---------------------------------------------------------------------
# ① Giá trị tương lai — mục tiêu hồi quy thô, mọi nhãn khác đều suy từ đây
# ---------------------------------------------------------------------
def create_future_values(series, horizon=1):
    """
    Lấy giá trị của chính chuỗi tại thời điểm t + horizon.

        y_t = x_{t+horizon}

    LƯU Ý về mô hình dựa trên cây: mục tiêu ở dạng giá trị tuyệt đối
    khiến mô hình không thể dự đoán ra ngoài khoảng đã thấy khi huấn
    luyện. Với chuỗi có xu thế dài hạn, ② thường là lựa chọn tốt hơn.

    Parameters:
        series  : list giá trị
        horizon : số bước nhìn về tương lai

    Returns:
        list cùng độ dài; horizon phần tử cuối là None
    """
    _assert_horizon(horizon)
    return list(series[horizon:]) + [None] * horizon


# ---------------------------------------------------------------------
# ② Tỷ suất tương lai — chuẩn hoá ①, giữ được ý nghĩa khi thang giá đổi
# ---------------------------------------------------------------------
def create_future_returns(series, horizon=1):
    """
    Tỷ suất biến động từ thời điểm t tới t + horizon.

        y_t = (x_{t+horizon} - x_t) / x_t

    Returns:
        list cùng độ dài; horizon phần tử cuối là None
    """
    _assert_horizon(horizon)
    future_values = create_future_values(series, horizon)
    return [
        None if future is None or not current else (future - current) / current
        for current, future in zip(series, future_values)
    ]


# ---------------------------------------------------------------------
# ③ Nhãn hai lớp — chỉ giữ lại DẤU của ②, vứt bỏ thông tin về độ lớn
# ---------------------------------------------------------------------
def create_direction_labels(series, horizon=1, positive_label=1,
                            negative_label=0, flat_label=None):
    """
    Nhãn nhị phân cho chiều biến động của bước kế tiếp.

        y_t = positive_label   nếu x_{t+horizon} >  x_t
              negative_label   nếu x_{t+horizon} <  x_t
              flat_label       nếu bằng nhau

    Bài toán trở nên dễ hơn hồi quy vì chỉ cần đoán đúng dấu, nhưng cũng
    mất thông tin: một bước tăng rất nhỏ và một bước tăng rất mạnh nhận
    cùng một nhãn.

    CÁCH XỬ LÝ TRƯỜNG HỢP ĐỨNG YÊN CÓ ẢNH HƯỞNG LỚN. Với chuỗi được làm
    tròn thô, số quan sát không đổi có thể lên tới hàng chục phần trăm.
    Dồn hết chúng vào một phía tạo ra mất cân bằng lớp GIẢ, khiến mô
    hình học cách luôn đoán lớp đó và mọi chỉ số trở nên vô nghĩa. Vì
    vậy mặc định là loại chúng khỏi tập huấn luyện (flat_label=None), và
    nên kiểm tra tỷ lệ này trước khi chọn cách khác.

    Parameters:
        series         : list giá trị
        horizon        : số bước nhìn về tương lai
        positive_label : nhãn cho chiều tăng
        negative_label : nhãn cho chiều giảm
        flat_label     : nhãn cho trường hợp không đổi.
                         None → để None, mẫu sẽ bị loại ở bước ⑤.

    Returns:
        list cùng độ dài; horizon phần tử cuối là None
    """
    _assert_horizon(horizon)
    future_values = create_future_values(series, horizon)

    labels = []
    for current, future in zip(series, future_values):
        if future is None or current is None:
            labels.append(None)
        elif future > current:
            labels.append(positive_label)
        elif future < current:
            labels.append(negative_label)
        else:
            labels.append(flat_label)
    return labels


# ---------------------------------------------------------------------
# ③b Đếm quan sát đứng yên — con số quyết định cách chọn flat_label ở ③
# ---------------------------------------------------------------------
def count_flat_observations(series, horizon=1):
    """
    Đếm số quan sát mà giá trị không đổi sau `horizon` bước.

    Nên gọi TRƯỚC khi gán nhãn: tỷ lệ đứng yên cao là dấu hiệu chuỗi bị
    làm tròn thô hoặc thanh khoản thấp, và quyết định trực tiếp cách
    chọn tham số flat_label ở ③.

    Returns:
        dict { 'increase', 'decrease', 'flat', 'flat_ratio', 'comparable' }
    """
    _assert_horizon(horizon)
    future_values = create_future_values(series, horizon)

    increase = decrease = flat = 0
    for current, future in zip(series, future_values):
        if future is None or current is None:
            continue
        if future > current:
            increase += 1
        elif future < current:
            decrease += 1
        else:
            flat += 1

    comparable = increase + decrease + flat
    return {
        'increase':   increase,
        'decrease':   decrease,
        'flat':       flat,
        'flat_ratio': flat / comparable if comparable else 0.0,
        'comparable': comparable,
    }


# ---------------------------------------------------------------------
# ④ Nhãn ba lớp — thêm vùng trung tính để ③ không phải phán bừa lúc đi ngang
# ---------------------------------------------------------------------
def create_ternary_labels(series, threshold_series, horizon=1,
                          positive_label=1, neutral_label=0,
                          negative_label=-1):
    """
    Nhãn ba lớp: tăng / đi ngang / giảm, với vùng trung tính rộng theo
    một ngưỡng động.

        y_t = positive_label   nếu x_{t+h} - x_t >  ngưỡng_t
              negative_label   nếu x_{t+h} - x_t < -ngưỡng_t
              neutral_label    nếu nằm giữa

    Ngưỡng ĐỘNG (ví dụ suy từ biên độ dao động thực tế) hợp lý hơn một
    hằng số cố định: cùng một mức thay đổi có thể là đáng kể trong giai
    đoạn yên ắng nhưng chỉ là nhiễu trong giai đoạn biến động mạnh.

    Parameters:
        series           : list giá trị
        threshold_series : list ngưỡng tại từng thời điểm, cùng độ dài
        horizon          : số bước nhìn về tương lai

    Returns:
        list cùng độ dài; horizon phần tử cuối là None
    """
    _assert_horizon(horizon)
    if len(series) != len(threshold_series):
        raise ValueError(
            f"Chuỗi giá trị ({len(series)}) và chuỗi ngưỡng "
            f"({len(threshold_series)}) phải cùng độ dài."
        )

    future_values = create_future_values(series, horizon)

    labels = []
    for current, future, threshold in zip(series, future_values, threshold_series):
        if future is None or current is None or threshold is None:
            labels.append(None)
            continue

        change = future - current
        if change > threshold:
            labels.append(positive_label)
        elif change < -threshold:
            labels.append(negative_label)
        else:
            labels.append(neutral_label)
    return labels


# ---------------------------------------------------------------------
# ⑤ Khớp đặc trưng ↔ nhãn — bước dọn bắt buộc trước khi đưa vào mô hình
# ---------------------------------------------------------------------
def align_features_and_targets(samples, targets):
    """
    Loại bỏ những dòng không có nhãn (phần đuôi do dịch nhãn về sau, và
    bất kỳ ô None nào còn sót).

    Parameters:
        samples : list of lists — các mẫu đặc trưng
        targets : list giá trị mục tiêu, cùng độ dài với samples

    Returns:
        samples_clean : list of lists đã lọc
        targets_clean : list nhãn đã lọc
        kept_indices  : list chỉ số dòng được giữ lại (so với đầu vào)
    """
    if len(samples) != len(targets):
        raise ValueError(
            f"Số mẫu ({len(samples)}) phải bằng số nhãn ({len(targets)})."
        )

    kept_indices = [
        index for index in range(len(targets))
        if targets[index] is not None
        and all(value is not None for value in samples[index])
    ]
    samples_clean = [samples[index] for index in kept_indices]
    targets_clean = [targets[index] for index in kept_indices]
    return samples_clean, targets_clean, kept_indices


# ---------------------------------------------------------------------
# ⑥ Cân bằng lớp — con số phải xem TRƯỚC khi tin vào bất kỳ Accuracy nào
# ---------------------------------------------------------------------
def calculate_class_balance(labels):
    """
    Thống kê phân bố lớp.

    Đây là mốc so sánh tối thiểu: một mô hình luôn đoán lớp chiếm đa số
    đã đạt Accuracy đúng bằng majority_ratio. Mọi kết quả không vượt qua
    con số này đều vô nghĩa.

    Parameters:
        labels : list nhãn (đã loại None)

    Returns:
        dict { 'counts', 'proportions', 'majority_label',
               'majority_ratio', 'num_samples' }
    """
    valid_labels = [label for label in labels if label is not None]
    if not valid_labels:
        raise ValueError("Không có nhãn hợp lệ nào để thống kê.")

    counts = {}
    for label in valid_labels:
        counts[label] = counts.get(label, 0) + 1

    total = len(valid_labels)
    majority_label = max(counts, key=lambda label: counts[label])
    return {
        'counts':         dict(sorted(counts.items(), key=lambda pair: str(pair[0]))),
        'proportions':    {label: count / total for label, count in counts.items()},
        'majority_label': majority_label,
        'majority_ratio': counts[majority_label] / total,
        'num_samples':    total,
    }


# ---------------------------------------------------------------------
# Kiểm tra dùng chung cho toàn module
# ---------------------------------------------------------------------
def _assert_horizon(horizon):
    """Tầm nhìn tương lai phải là số nguyên dương."""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon phải là số nguyên dương.")
