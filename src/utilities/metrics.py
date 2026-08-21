# =====================================================================
# Metrics — các chỉ số đánh giá mô hình hồi quy
# =====================================================================
# Module thuần toán học, dùng chung cho mọi thuật toán hồi quy trong
# project (OLS, Polynomial + Gradient Descent, ...).
# Không phụ thuộc vào bài toán, dữ liệu hay thư viện bên ngoài.
#
# Thứ tự khai báo bám đúng chuỗi tính toán thực tế:
#
#   ① SST  (biến động tổng của y quanh ȳ)      — nền để chuẩn hoá sai số
#   ② SSE  (biến động không giải thích được)   — sai số thô của mô hình
#   ③ MSE  = SSE / n                           — trung bình hoá SSE
#   ④ RMSE = √MSE                              — đưa về đơn vị của y
#   ⑤ MAE                                      — nhánh sai số tuyệt đối
#   ⑥ R²      = 1 - SSE / SST                  — cần ① và ②
#   ⑦ R²_adj                                   — cần ⑥ và số tham số p
#   ⑧ MAPE                                     — nhánh sai số tương đối
#   ⑨ DA                                       — chuyển từ độ lớn sang hướng
# =====================================================================


# ---------------------------------------------------------------------
# ① SST — điểm khởi đầu: đo tổng biến động vốn có của biến mục tiêu
# ---------------------------------------------------------------------
def calculate_sst(y_true):
    """
    Tổng Bình Phương Toàn Phần (SST - Sum of Squares Total).
    SST = Σ(y_i - ȳ)² — tổng biến động của y quanh giá trị trung bình.

    Đây là mốc tham chiếu: mô hình chỉ có ý nghĩa khi giải thích được
    một phần biến động này (xem R² ở bước ⑥).
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Không thể tính SST trên tập rỗng.")
    y_mean = sum(y_true) / n
    return sum((yt - y_mean) ** 2 for yt in y_true)


# ---------------------------------------------------------------------
# ② SSE — phần biến động mà mô hình KHÔNG giải thích được
# ---------------------------------------------------------------------
def calculate_sse(y_true, y_pred):
    """
    Tổng Bình Phương Sai Số (SSE - Sum of Squared Errors).
    SSE = Σ(y_i - ŷ_i)²

    Là đại lượng gốc: MSE, RMSE và R² đều được suy ra từ SSE.
    """
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))


# ---------------------------------------------------------------------
# ③ MSE — trung bình hoá SSE theo số quan sát
# ---------------------------------------------------------------------
def calculate_mse(y_true, y_pred):
    """
    Sai Số Bình Phương Trung Bình (MSE - Mean Squared Error).
    MSE = SSE / n
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Không thể tính MSE trên tập rỗng.")
    return calculate_sse(y_true, y_pred) / n


# ---------------------------------------------------------------------
# ④ RMSE — lấy căn để đưa MSE về cùng đơn vị với biến mục tiêu
# ---------------------------------------------------------------------
def calculate_rmse(y_true, y_pred):
    """
    Căn Bậc Hai Sai Số Bình Phương Trung Bình (RMSE - Root Mean Squared Error).
    RMSE = √MSE — cùng đơn vị với biến mục tiêu.
    """
    return calculate_mse(y_true, y_pred) ** 0.5


# ---------------------------------------------------------------------
# ⑤ MAE — nhánh độc lập, không đi qua SSE (ít nhạy với ngoại lai hơn)
# ---------------------------------------------------------------------
def calculate_mae(y_true, y_pred):
    """
    Sai Số Tuyệt Đối Trung Bình (MAE - Mean Absolute Error).
    MAE = (1/n) * Σ|y_i - ŷ_i|
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Không thể tính MAE trên tập rỗng.")
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


# ---------------------------------------------------------------------
# ⑥ R² — hợp nhất SST (①) và SSE (②) thành tỉ lệ biến động giải thích được
# ---------------------------------------------------------------------
def calculate_r_squared(y_true, y_pred):
    """
    Hệ số xác định R² (Coefficient of Determination).
    R² = 1 - SSE / SST
      - R² = 1.0 : dự đoán hoàn hảo
      - R² = 0.0 : không tốt hơn việc dự đoán bằng giá trị trung bình
      - R² < 0   : tệ hơn cả việc dự đoán bằng giá trị trung bình
    """
    sst = calculate_sst(y_true)
    if sst == 0:
        raise ValueError("SST = 0, biến mục tiêu là hằng số nên R² không xác định.")
    return 1 - (calculate_sse(y_true, y_pred) / sst)


# ---------------------------------------------------------------------
# ⑦ R²_adj — hiệu chỉnh R² (⑥) theo số tham số của mô hình
# ---------------------------------------------------------------------
def calculate_adjusted_r_squared(y_true, y_pred, num_parameters):
    """
    Hệ số xác định hiệu chỉnh R²_adj — phạt mô hình có nhiều tham số.
    R²_adj = 1 - (1 - R²) * (n - 1) / (n - p - 1)

    Parameters:
        y_true         : list giá trị thực tế
        y_pred         : list giá trị dự đoán
        num_parameters : số tham số của mô hình, KHÔNG tính hệ số chặn (bias)

    Returns:
        float, hoặc None nếu bậc tự do không đủ (n - p - 1 <= 0)
    """
    n = len(y_true)
    degrees_of_freedom = n - num_parameters - 1
    if degrees_of_freedom <= 0:
        return None
    r_squared = calculate_r_squared(y_true, y_pred)
    return 1 - (1 - r_squared) * (n - 1) / degrees_of_freedom


# ---------------------------------------------------------------------
# ⑧ MAPE — nhánh sai số tương đối, chuẩn hoá theo độ lớn của chính y
# ---------------------------------------------------------------------
def calculate_mape(y_true, y_pred, epsilon=1e-12):
    """
    Sai Số Phần Trăm Tuyệt Đối Trung Bình (MAPE – Mean Absolute
    Percentage Error).
    MAPE = (100/n) * Σ |(y_i - ŷ_i) / y_i|

    Dễ diễn giải vì không phụ thuộc đơn vị, nhưng mất ý nghĩa khi giá
    trị thực tế gần 0 — những mẫu như vậy bị bỏ qua.

    Parameters:
        epsilon : ngưỡng dưới của |y_i| để mẫu được tính vào MAPE

    Returns:
        float (đơn vị %), hoặc None nếu không còn mẫu hợp lệ
    """
    contributions = [
        abs((yt - yp) / yt) for yt, yp in zip(y_true, y_pred) if abs(yt) > epsilon
    ]
    if not contributions:
        return None
    return 100.0 * sum(contributions) / len(contributions)


# ---------------------------------------------------------------------
# ⑨ DA — rời thang đo sai số, chỉ hỏi mô hình có đoán đúng HƯỚNG không
# ---------------------------------------------------------------------
def calculate_directional_accuracy(y_true, y_pred, reference_values=None):
    """
    Độ Chính Xác Về Hướng (Directional Accuracy) — tỷ lệ dự đoán đúng
    DẤU của biến động so với một mốc tham chiếu.

    DA = (1/m) * Σ 1[ sign(y_i - ref_i) == sign(ŷ_i - ref_i) ]

    Chỉ số này bắc cầu giữa bài toán hồi quy và bài toán phân loại xu
    hướng: một mô hình có RMSE thấp vẫn có thể đoán sai hướng, và ngược
    lại.

    Parameters:
        y_true           : list giá trị thực tế
        y_pred           : list giá trị dự đoán
        reference_values : list giá trị mốc để so hướng. None → dùng
                           chính chuỗi y_true dịch lùi một bước, tức so
                           hướng giữa hai quan sát liên tiếp.

    Returns:
        float trong [0, 1], hoặc None nếu không có biến động nào để so
    """
    if reference_values is None:
        if len(y_true) < 2:
            return None
        reference_values = y_true[:-1]
        y_true = y_true[1:]
        y_pred = y_pred[1:]

    matches = 0
    comparable = 0
    for actual, predicted, reference in zip(y_true, y_pred, reference_values):
        actual_change = actual - reference
        predicted_change = predicted - reference
        if actual_change == 0:
            continue
        comparable += 1
        if (actual_change > 0) == (predicted_change > 0):
            matches += 1

    if comparable == 0:
        return None
    return matches / comparable


# ---------------------------------------------------------------------
# Gộp: chạy lại đúng chuỗi ① → ⑧ và trả về toàn bộ chỉ số
# ---------------------------------------------------------------------
def calculate_all_metrics(y_true, y_pred, num_parameters=None):
    """
    Tính trọn bộ chỉ số đánh giá và trả về dưới dạng dict.

    Các chỉ số được tính theo đúng thứ tự phụ thuộc ① → ⑧ ở đầu module,
    nên dict trả về cũng đọc được từ trên xuống như một mạch suy diễn.
    Riêng ⑨ (Directional Accuracy) cần mốc tham chiếu riêng nên không
    nằm trong bộ gộp này.

    Parameters:
        y_true         : list giá trị thực tế
        y_pred         : list giá trị dự đoán
        num_parameters : số tham số (không tính bias) — dùng cho R²_adj.
                         Bỏ qua nếu không cần R²_adj.

    Returns:
        dict { 'sst', 'sse', 'mse', 'rmse', 'mae', 'mape', 'r_squared',
               'adjusted_r_squared' (chỉ khi truyền num_parameters) }
    """
    results = {
        # ① → ② : biến động tổng, rồi phần sai số của mô hình
        'sst':       calculate_sst(y_true),
        'sse':       calculate_sse(y_true, y_pred),

        # ③ → ⑤ : các dạng sai số trung bình
        'mse':       calculate_mse(y_true, y_pred),
        'rmse':      calculate_rmse(y_true, y_pred),
        'mae':       calculate_mae(y_true, y_pred),

        # ⑧ : sai số tương đối, không phụ thuộc đơn vị
        'mape':      calculate_mape(y_true, y_pred),

        # ⑥ : tỉ lệ biến động giải thích được
        'r_squared': calculate_r_squared(y_true, y_pred),
    }

    # ⑦ : chỉ tính khi biết số tham số của mô hình
    if num_parameters is not None:
        results['adjusted_r_squared'] = calculate_adjusted_r_squared(
            y_true, y_pred, num_parameters
        )
    return results
