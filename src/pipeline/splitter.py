# =====================================================================
# Splitter — tách tập dữ liệu theo THỨ TỰ THỜI GIAN
# =====================================================================
# Module thuần Python, không phụ thuộc thư viện ngoài.
#
# VÌ SAO KHÔNG DÙNG ĐƯỢC CÁCH TÁCH NGẪU NHIÊN THÔNG THƯỜNG:
#
# Tách ngẫu nhiên (train_test_split có xáo trộn, KFold) giả định các mẫu
# độc lập và cùng phân phối. Chuỗi thời gian vi phạm cả hai:
#
#   1. Xáo trộn đặt mẫu của tương lai vào tập huấn luyện, còn mẫu của
#      quá khứ vào tập kiểm tra — mô hình được học đáp án trước.
#   2. Các mẫu liền kề chia chung phần lớn cửa sổ trượt nên gần như
#      trùng nhau; tách ngẫu nhiên khiến hai tập bị "dính" thông tin.
#
# Toàn bộ module này chỉ cắt theo khối liên tục và luôn tiến về phía
# trước. Không hàm nào xáo trộn dữ liệu.
#
# Tham số gap có mặt ở mọi hàm để chèn một khoảng trống giữa hai tập,
# triệt tiêu phần chồng lấn cửa sổ trượt ở đường biên.
#
# Thứ tự khai báo bám đúng trình tự sử dụng:
#
#   ①  Tách ba khối theo tỷ lệ   — cách chia cơ bản nhất
#   ②  Áp khoảng chỉ số lên dữ liệu — biến ① thành mẫu và nhãn thật
#   ③  Gộp ① và ② thành một bước  — hàm dùng nhiều nhất trong notebook
#   ④  Vòng kiểm định cửa sổ mở rộng — đánh giá nhiều lần, quỹ thời gian lớn dần
#   ⑤  Vòng kiểm định cửa sổ trượt   — như ④ nhưng quỹ thời gian cố định
#   ⑥  Tóm tắt một phép tách      — bản tin để in ra kiểm tra
# =====================================================================


# ---------------------------------------------------------------------
# ① Tách ba khối — chỉ làm việc trên CHỈ SỐ, chưa đụng tới dữ liệu
# ---------------------------------------------------------------------
def split_indices_sequentially(num_samples, train_ratio=0.70,
                               validation_ratio=0.15, gap=0):
    """
    Chia dãy chỉ số 0…n-1 thành ba khối liên tiếp theo thứ tự thời gian.

        [--------- TRAIN ---------][gap][- VALID -][gap][- TEST -]

    Phần còn lại sau train và validation thuộc về test, nên không cần
    truyền test_ratio.

    Parameters:
        num_samples      : tổng số mẫu
        train_ratio      : tỷ lệ dành cho huấn luyện
        validation_ratio : tỷ lệ dành cho kiểm định
        gap              : số mẫu bỏ trống giữa hai khối liền nhau

    Returns:
        train_indices, validation_indices, test_indices — ba list chỉ số
    """
    if num_samples <= 0:
        raise ValueError("Số mẫu phải là số nguyên dương.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio phải nằm trong khoảng (0, 1).")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio phải nằm trong khoảng [0, 1).")
    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio phải nhỏ hơn 1 để còn chỗ cho tập test."
        )
    if gap < 0:
        raise ValueError("gap không được âm.")

    train_end = int(num_samples * train_ratio)
    validation_start = train_end + gap
    validation_end = validation_start + int(num_samples * validation_ratio)
    test_start = validation_end + gap

    if test_start >= num_samples:
        raise ValueError(
            f"Không đủ mẫu để tách: cần hơn {test_start} mẫu, "
            f"hiện chỉ có {num_samples}. Hãy giảm gap hoặc giảm tỷ lệ."
        )

    return (
        list(range(0, train_end)),
        list(range(validation_start, validation_end)),
        list(range(test_start, num_samples)),
    )


# ---------------------------------------------------------------------
# ② Áp chỉ số lên dữ liệu — bước dịch từ khoảng chỉ số sang mẫu thực tế
# ---------------------------------------------------------------------
def select_by_indices(samples, targets, indices):
    """
    Lấy ra tập con mẫu và nhãn theo danh sách chỉ số.

    Parameters:
        samples : list of lists — các mẫu
        targets : list giá trị mục tiêu
        indices : list chỉ số cần lấy

    Returns:
        samples_subset, targets_subset
    """
    if len(samples) != len(targets):
        raise ValueError(
            f"Số mẫu ({len(samples)}) phải bằng số nhãn ({len(targets)})."
        )
    return (
        [samples[index] for index in indices],
        [targets[index] for index in indices],
    )


# ---------------------------------------------------------------------
# ③ Tách trọn bộ — gộp ① và ②, đây là hàm notebook gọi trực tiếp
# ---------------------------------------------------------------------
def split_dataset(samples, targets, train_ratio=0.70, validation_ratio=0.15,
                  gap=0):
    """
    Tách bộ dữ liệu thành ba tập theo thứ tự thời gian.

    Vai trò của từng tập, cần phân biệt rạch ròi:
        - TRAIN    : huấn luyện tham số của mô hình.
        - VALIDATE : chọn siêu tham số và ngưỡng quyết định. KHÔNG dùng
                     để huấn luyện.
        - TEST     : chỉ chạy MỘT LẦN duy nhất ở cuối. Mỗi lần quay lại
                     chỉnh mô hình theo kết quả test là một lần tập test
                     mất dần tính khách quan.

    Returns:
        dict {
            'train':      (samples, targets),
            'validation': (samples, targets),
            'test':       (samples, targets),
            'indices':    (train_indices, validation_indices, test_indices),
        }
    """
    train_indices, validation_indices, test_indices = split_indices_sequentially(
        len(samples), train_ratio, validation_ratio, gap
    )
    return {
        'train':      select_by_indices(samples, targets, train_indices),
        'validation': select_by_indices(samples, targets, validation_indices),
        'test':       select_by_indices(samples, targets, test_indices),
        'indices':    (train_indices, validation_indices, test_indices),
    }


# ---------------------------------------------------------------------
# ④ Cửa sổ MỞ RỘNG — mô phỏng đúng thực tế: càng về sau càng nhiều lịch sử
# ---------------------------------------------------------------------
def generate_expanding_window_folds(num_samples, num_folds=5,
                                    minimum_train_size=None, gap=0):
    """
    Sinh các vòng kiểm định tiến dần với cửa sổ huấn luyện MỞ RỘNG.

        Vòng 1: train [=====]        → validate [--]
        Vòng 2: train [=======]      → validate    [--]
        Vòng 3: train [=========]    → validate       [--]

    Tập huấn luyện lớn dần, mỗi vòng kiểm định trên đoạn kế tiếp chưa
    từng thấy. Đây là cách đánh giá sát với cách mô hình được dùng thật:
    tại mọi thời điểm ta có toàn bộ quá khứ và phải dự đoán tương lai.

    Kết quả cần đọc theo TRUNG BÌNH ± ĐỘ LỆCH CHUẨN. Độ lệch chuẩn lớn
    nghĩa là mô hình không ổn định khi điều kiện dữ liệu thay đổi — một
    con số trung bình đẹp có thể che giấu điều đó.

    Parameters:
        num_samples        : tổng số mẫu
        num_folds          : số vòng kiểm định
        minimum_train_size : số mẫu huấn luyện tối thiểu ở vòng đầu.
                             None → dành một nửa dữ liệu cho vòng đầu.
        gap                : số mẫu bỏ trống giữa train và validate

    Returns:
        list of tuple (train_indices, validation_indices)
    """
    if num_folds < 1:
        raise ValueError("num_folds phải ≥ 1.")
    if gap < 0:
        raise ValueError("gap không được âm.")

    if minimum_train_size is None:
        minimum_train_size = num_samples // 2
    if minimum_train_size < 1:
        raise ValueError("minimum_train_size phải ≥ 1.")

    remaining = num_samples - minimum_train_size - gap
    fold_size = remaining // num_folds
    if fold_size < 1:
        raise ValueError(
            f"Không đủ mẫu cho {num_folds} vòng: sau khi dành "
            f"{minimum_train_size} mẫu huấn luyện và gap {gap}, "
            f"chỉ còn {remaining} mẫu."
        )

    folds = []
    for fold_position in range(num_folds):
        train_end = minimum_train_size + fold_position * fold_size
        validation_start = train_end + gap
        validation_end = validation_start + fold_size
        if fold_position == num_folds - 1:
            validation_end = num_samples
        if validation_start >= num_samples:
            break

        folds.append((
            list(range(0, train_end)),
            list(range(validation_start, min(validation_end, num_samples))),
        ))
    return folds


# ---------------------------------------------------------------------
# ⑤ Cửa sổ TRƯỢT — như ④ nhưng quên quá khứ xa, hợp khi quan hệ đổi theo thời gian
# ---------------------------------------------------------------------
def generate_rolling_window_folds(num_samples, num_folds=5, train_size=None,
                                  gap=0):
    """
    Sinh các vòng kiểm định tiến dần với cửa sổ huấn luyện CỐ ĐỊNH.

        Vòng 1: train [=====]        → validate [--]
        Vòng 2:    train [=====]     → validate    [--]
        Vòng 3:       train [=====]  → validate       [--]

    Khác ④ ở chỗ cửa sổ trượt đi thay vì mở rộng, tức mô hình chủ động
    quên quá khứ xa. Hợp lý khi quan hệ giữa đặc trưng và mục tiêu thay
    đổi theo thời gian, khiến dữ liệu quá cũ trở nên gây nhiễu.

    Parameters:
        num_samples : tổng số mẫu
        num_folds   : số vòng kiểm định
        train_size  : số mẫu huấn luyện cố định mỗi vòng.
                      None → một nửa tổng số mẫu.
        gap         : số mẫu bỏ trống giữa train và validate

    Returns:
        list of tuple (train_indices, validation_indices)
    """
    if num_folds < 1:
        raise ValueError("num_folds phải ≥ 1.")
    if gap < 0:
        raise ValueError("gap không được âm.")

    if train_size is None:
        train_size = num_samples // 2
    if train_size < 1:
        raise ValueError("train_size phải ≥ 1.")

    remaining = num_samples - train_size - gap
    fold_size = remaining // num_folds
    if fold_size < 1:
        raise ValueError(
            f"Không đủ mẫu cho {num_folds} vòng: sau khi dành {train_size} "
            f"mẫu huấn luyện và gap {gap}, chỉ còn {remaining} mẫu."
        )

    folds = []
    for fold_position in range(num_folds):
        train_start = fold_position * fold_size
        train_end = train_start + train_size
        validation_start = train_end + gap
        validation_end = validation_start + fold_size
        if validation_start >= num_samples:
            break

        folds.append((
            list(range(train_start, train_end)),
            list(range(validation_start, min(validation_end, num_samples))),
        ))
    return folds


# ---------------------------------------------------------------------
# ⑥ Tóm tắt phép tách — in ra để mắt người kiểm tra lại thay vì tin tưởng mù
# ---------------------------------------------------------------------
def describe_split(train_indices, validation_indices, test_indices,
                   key_series=None):
    """
    Tóm tắt một phép tách: kích thước, khoảng chỉ số và khoảng thời gian
    thực tế của từng tập.

    Parameters:
        train_indices      : list chỉ số tập huấn luyện
        validation_indices : list chỉ số tập kiểm định
        test_indices       : list chỉ số tập kiểm tra
        key_series         : list mốc thời gian của toàn bộ dữ liệu
                             (tùy chọn) — có thì báo cáo thêm mốc đầu/cuối

    Returns:
        dict { tên_tập: {'size', 'ratio', 'index_range'
                         [, 'key_range'] } }
    """
    total = len(train_indices) + len(validation_indices) + len(test_indices)
    if total == 0:
        raise ValueError("Phép tách không có mẫu nào.")

    summary = {}
    for name, indices in (('train', train_indices),
                          ('validation', validation_indices),
                          ('test', test_indices)):
        entry = {
            'size':        len(indices),
            'ratio':       len(indices) / total,
            'index_range': (indices[0], indices[-1]) if indices else None,
        }
        if key_series is not None and indices:
            entry['key_range'] = (key_series[indices[0]], key_series[indices[-1]])
        summary[name] = entry
    return summary
