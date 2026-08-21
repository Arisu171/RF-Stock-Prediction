# =====================================================================
# Metrics Classification — các chỉ số đánh giá mô hình phân loại
# =====================================================================
# Module thuần toán học, bổ sung cho metrics.py (vốn dành cho hồi quy).
# Không phụ thuộc vào bài toán, dữ liệu hay thư viện bên ngoài.
#
# Quy ước: lớp dương (positive) là lớp mà Precision/Recall/F1 hướng tới.
# Khi không chỉ định, lớp dương mặc định là nhãn LỚN NHẤT trong không
# gian nhãn đã sắp xếp.
#
# Thứ tự khai báo bám đúng chuỗi tính toán thực tế:
#
#   ①–②    Chốt không gian nhãn và lớp dương  — nền cho mọi phép đo
#   ③–④    Ma trận nhầm lẫn → TP/FP/FN/TN     — bảng gốc của tất cả
#   ⑤–⑥    Accuracy, Error rate               — góc nhìn tổng thể
#   ⑦–⑨    Precision, Recall, Specificity     — ba góc nhìn bất đối xứng
#   ⑩–⑬   F1, F-beta, Balanced acc., MCC     — các cách gộp ⑦–⑨
#   ⑭–⑮   Báo cáo theo từng lớp              — chạy ⑦–⑩ cho mọi lớp
#   ⑯–⑳   ROC, AUC, Log-loss, Brier          — rời nhãn cứng, dùng điểm số
#   ㉑–㉓   Ngưỡng quyết định                  — quay lại nhãn cứng từ ⑯–⑳
#   ㉔      Gộp toàn bộ
#
# Điểm mấu chốt: ⑤–⑬ cần NHÃN dự đoán, còn ⑯–⑳ cần ĐIỂM SỐ liên tục.
# Hai nhóm này đo hai thứ khác nhau và đều cần có mặt trong báo cáo.
# =====================================================================

import math


# ---------------------------------------------------------------------
# ① Không gian nhãn — chốt danh sách và THỨ TỰ lớp cho toàn bộ phép đo
# ---------------------------------------------------------------------
def resolve_label_space(y_true, y_pred=None, label_space=None):
    """
    Xác định danh sách nhãn dùng chung cho mọi phép đo.

    Parameters:
        y_true      : list nhãn thực tế
        y_pred      : list nhãn dự đoán (tùy chọn, để không bỏ sót nhãn)
        label_space : danh sách nhãn chỉ định sẵn (tùy chọn)

    Returns:
        list nhãn đã sắp xếp
    """
    if label_space is not None:
        return list(label_space)

    labels = set(y_true)
    if y_pred is not None:
        labels |= set(y_pred)
    return sorted(labels)


# ---------------------------------------------------------------------
# ② Lớp dương — mọi chỉ số bất đối xứng bên dưới đều quy về mốc này
# ---------------------------------------------------------------------
def resolve_positive_label(label_space, positive_label=None):
    """
    Xác định nhãn được coi là lớp dương.

    Parameters:
        label_space    : list nhãn đã sắp xếp
        positive_label : nhãn chỉ định (None → lấy nhãn lớn nhất)

    Returns:
        nhãn lớp dương
    """
    if positive_label is None:
        return label_space[-1]
    if positive_label not in label_space:
        raise ValueError(
            f"Nhãn dương '{positive_label}' không có trong không gian nhãn "
            f"{label_space}."
        )
    return positive_label


# ---------------------------------------------------------------------
# ③ Ma trận nhầm lẫn — bảng gốc, mọi chỉ số đều là một tỷ số rút từ đây
# ---------------------------------------------------------------------
def build_confusion_matrix(y_true, y_pred, label_space=None):
    """
    Ma trận nhầm lẫn.

    matrix[i][j] = số mẫu THỰC TẾ thuộc lớp label_space[i] nhưng được
    DỰ ĐOÁN là lớp label_space[j].

    Đường chéo chính là số mẫu dự đoán đúng; các ô ngoài đường chéo cho
    biết mô hình nhầm theo hướng nào.

    Parameters:
        y_true      : list nhãn thực tế
        y_pred      : list nhãn dự đoán
        label_space : danh sách nhãn (None → suy ra từ dữ liệu)

    Returns:
        matrix      : list of lists kích thước K×K
        label_space : list nhãn tương ứng với chỉ số dòng/cột
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Số nhãn thực tế ({len(y_true)}) phải bằng số nhãn dự đoán "
            f"({len(y_pred)})."
        )

    label_space = resolve_label_space(y_true, y_pred, label_space)
    position_of_label = {label: index for index, label in enumerate(label_space)}
    matrix = [[0 for _ in label_space] for _ in label_space]

    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[position_of_label[true_label]][position_of_label[predicted_label]] += 1

    return matrix, label_space


# ---------------------------------------------------------------------
# ④ TP/FP/FN/TN — bốn ô của ③ ở dạng nhị phân, nguyên liệu của ⑤–⑫
# ---------------------------------------------------------------------
def calculate_binary_counts(y_true, y_pred, positive_label=None,
                            label_space=None):
    """
    Bốn đại lượng nền tảng của bài toán nhị phân.

        TP : dự đoán dương, thực tế dương
        FP : dự đoán dương, thực tế âm
        FN : dự đoán âm,   thực tế dương
        TN : dự đoán âm,   thực tế âm

    Returns:
        dict { 'true_positive', 'false_positive',
                'false_negative', 'true_negative' }
    """
    label_space = resolve_label_space(y_true, y_pred, label_space)
    positive_label = resolve_positive_label(label_space, positive_label)

    counts = {
        'true_positive':  0,
        'false_positive': 0,
        'false_negative': 0,
        'true_negative':  0,
    }
    for true_label, predicted_label in zip(y_true, y_pred):
        actual_positive = (true_label == positive_label)
        predicted_positive = (predicted_label == positive_label)

        if predicted_positive and actual_positive:
            counts['true_positive'] += 1
        elif predicted_positive and not actual_positive:
            counts['false_positive'] += 1
        elif not predicted_positive and actual_positive:
            counts['false_negative'] += 1
        else:
            counts['true_negative'] += 1
    return counts


# ---------------------------------------------------------------------
# ⑤ Accuracy — tỷ lệ đúng tổng thể; chỉ đáng tin khi hai lớp cân bằng
# ---------------------------------------------------------------------
def calculate_accuracy(y_true, y_pred):
    """
    Độ chính xác (Accuracy) — tỷ lệ mẫu được dự đoán đúng.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Chỉ đáng tin khi các lớp tương đối cân bằng. Với dữ liệu lệch
    90/10, dự đoán luôn lớp đa số đã cho Accuracy = 0.90.
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Không thể tính Accuracy trên tập rỗng.")
    return sum(
        1 for true_label, predicted_label in zip(y_true, y_pred)
        if true_label == predicted_label
    ) / n


# ---------------------------------------------------------------------
# ⑥ Error rate — phần bù của ⑤, tiện khi vẽ đường cong lỗi
# ---------------------------------------------------------------------
def calculate_error_rate(y_true, y_pred):
    """
    Tỷ lệ phân loại sai: 1 - Accuracy
    """
    return 1.0 - calculate_accuracy(y_true, y_pred)


# ---------------------------------------------------------------------
# ⑦ Precision — nhìn từ phía DỰ ĐOÁN dương: bao nhiêu phần là đúng
# ---------------------------------------------------------------------
def calculate_precision(y_true, y_pred, positive_label=None, label_space=None):
    """
    Độ chuẩn xác (Precision) — trong các mẫu ĐƯỢC DỰ ĐOÁN là dương, bao
    nhiêu phần là dương thật.

    Precision = TP / (TP + FP)

    Returns:
        float — 0.0 khi mô hình không dự đoán mẫu dương nào
    """
    counts = calculate_binary_counts(y_true, y_pred, positive_label, label_space)
    predicted_positive = counts['true_positive'] + counts['false_positive']
    if predicted_positive == 0:
        return 0.0
    return counts['true_positive'] / predicted_positive


# ---------------------------------------------------------------------
# ⑧ Recall — nhìn từ phía THỰC TẾ dương: bắt được bao nhiêu phần
# ---------------------------------------------------------------------
def calculate_recall(y_true, y_pred, positive_label=None, label_space=None):
    """
    Độ phủ (Recall / Sensitivity) — trong các mẫu THỰC SỰ dương, bao
    nhiêu phần được mô hình bắt được.

    Recall = TP / (TP + FN)

    Returns:
        float — 0.0 khi tập không có mẫu dương nào
    """
    counts = calculate_binary_counts(y_true, y_pred, positive_label, label_space)
    actual_positive = counts['true_positive'] + counts['false_negative']
    if actual_positive == 0:
        return 0.0
    return counts['true_positive'] / actual_positive


# ---------------------------------------------------------------------
# ⑨ Specificity — Recall của lớp âm, mảnh còn thiếu để có ⑪
# ---------------------------------------------------------------------
def calculate_specificity(y_true, y_pred, positive_label=None, label_space=None):
    """
    Độ đặc hiệu (Specificity) — Recall của lớp âm.

    Specificity = TN / (TN + FP)
    """
    counts = calculate_binary_counts(y_true, y_pred, positive_label, label_space)
    actual_negative = counts['true_negative'] + counts['false_positive']
    if actual_negative == 0:
        return 0.0
    return counts['true_negative'] / actual_negative


# ---------------------------------------------------------------------
# ⑩ F1 — trung bình điều hoà ⑦ và ⑧, buộc cả hai cùng phải khá
# ---------------------------------------------------------------------
def calculate_f1_score(y_true, y_pred, positive_label=None, label_space=None):
    """
    F1-score — trung bình điều hòa của Precision và Recall.

    F1 = 2 · P · R / (P + R)

    Trung bình điều hòa phạt nặng trường hợp một trong hai chỉ số quá
    thấp, nên F1 cao đòi hỏi cả hai đều khá.
    """
    precision = calculate_precision(y_true, y_pred, positive_label, label_space)
    recall = calculate_recall(y_true, y_pred, positive_label, label_space)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------
# ⑪ F-beta — ⑨ tổng quát, chọn nghiêng về Precision hay Recall
# ---------------------------------------------------------------------
def calculate_fbeta_score(y_true, y_pred, beta=1.0, positive_label=None,
                          label_space=None):
    """
    F-beta score — F1 tổng quát, cho phép ưu tiên Recall hoặc Precision.

    F_β = (1 + β²) · P · R / (β²·P + R)

      - β > 1 : coi trọng Recall  (sợ bỏ sót)
      - β < 1 : coi trọng Precision (sợ báo động giả)
    """
    precision = calculate_precision(y_true, y_pred, positive_label, label_space)
    recall = calculate_recall(y_true, y_pred, positive_label, label_space)
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0
    return (1.0 + beta_squared) * precision * recall / denominator


# ---------------------------------------------------------------------
# ⑫ Balanced accuracy — trung bình ⑧ và ⑨, không bị lớp đa số chi phối
# ---------------------------------------------------------------------
def calculate_balanced_accuracy(y_true, y_pred, positive_label=None,
                                label_space=None):
    """
    Độ chính xác cân bằng — trung bình cộng của Recall và Specificity.

    BalancedAccuracy = (Recall + Specificity) / 2

    Không bị lớp đa số chi phối như Accuracy thường.
    """
    recall = calculate_recall(y_true, y_pred, positive_label, label_space)
    specificity = calculate_specificity(y_true, y_pred, positive_label, label_space)
    return (recall + specificity) / 2.0


# ---------------------------------------------------------------------
# ⑬ MCC — chỉ số cân bằng nhất, dùng trọn cả bốn ô của ④
# ---------------------------------------------------------------------
def calculate_matthews_correlation(y_true, y_pred, positive_label=None,
                                   label_space=None):
    """
    Hệ số tương quan Matthews (MCC) — chỉ số cân bằng nhất cho bài toán
    nhị phân, tính trên cả bốn ô của ma trận nhầm lẫn.

    MCC = (TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Giá trị trong [-1, 1]:
      -  1 : dự đoán hoàn hảo
      -  0 : không tốt hơn ngẫu nhiên
      - -1 : dự đoán ngược hoàn toàn
    """
    counts = calculate_binary_counts(y_true, y_pred, positive_label, label_space)
    true_positive = counts['true_positive']
    false_positive = counts['false_positive']
    false_negative = counts['false_negative']
    true_negative = counts['true_negative']

    numerator = true_positive * true_negative - false_positive * false_negative
    denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------
# ⑭ Báo cáo theo lớp — chạy ⑦–⑨ cho TỪNG lớp, lộ ra lớp bị bỏ rơi
# ---------------------------------------------------------------------
def build_classification_report(y_true, y_pred, label_space=None):
    """
    Precision, Recall, F1 và số mẫu (support) cho TỪNG lớp, kèm hai giá
    trị trung bình tổng hợp.

    Cần thiết để phát hiện trường hợp mô hình "bỏ rơi" một lớp: Accuracy
    tổng thể vẫn cao trong khi F1 của lớp thiểu số gần bằng 0.

    Returns:
        dict {
            'per_class' : { nhãn: {'precision','recall','f1_score','support'} },
            'macro_avg' : {'precision','recall','f1_score'},
            'weighted_avg' : {'precision','recall','f1_score'},
            'accuracy'  : float,
        }
    """
    label_space = resolve_label_space(y_true, y_pred, label_space)
    per_class = {}

    for label in label_space:
        per_class[label] = {
            'precision': calculate_precision(y_true, y_pred, label, label_space),
            'recall':    calculate_recall(y_true, y_pred, label, label_space),
            'f1_score':  calculate_f1_score(y_true, y_pred, label, label_space),
            'support':   sum(1 for value in y_true if value == label),
        }

    total_support = sum(scores['support'] for scores in per_class.values())
    macro_average = {}
    weighted_average = {}

    for key in ('precision', 'recall', 'f1_score'):
        values = [scores[key] for scores in per_class.values()]
        macro_average[key] = sum(values) / len(values) if values else 0.0

        if total_support:
            weighted_average[key] = sum(
                scores[key] * scores['support'] for scores in per_class.values()
            ) / total_support
        else:
            weighted_average[key] = 0.0

    return {
        'per_class':    per_class,
        'macro_avg':    macro_average,
        'weighted_avg': weighted_average,
        'accuracy':     calculate_accuracy(y_true, y_pred),
    }


# ---------------------------------------------------------------------
# ⑮ Định dạng bảng — chỉ là lớp trình bày của ⑭
# ---------------------------------------------------------------------
def format_classification_report(report, decimals=4):
    """
    Định dạng kết quả của build_classification_report() thành bảng văn
    bản để in ra màn hình hoặc ghi vào báo cáo.

    Parameters:
        report   : dict trả về bởi build_classification_report()
        decimals : số chữ số thập phân

    Returns:
        str nhiều dòng
    """
    lines = [
        f"{'Lớp':<12}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>10}",
        '-' * 58,
    ]
    for label, scores in report['per_class'].items():
        lines.append(
            f"{str(label):<12}"
            f"{scores['precision']:>12.{decimals}f}"
            f"{scores['recall']:>12.{decimals}f}"
            f"{scores['f1_score']:>12.{decimals}f}"
            f"{scores['support']:>10}"
        )

    lines.append('-' * 58)
    for name, key in (('Macro avg', 'macro_avg'), ('Weighted avg', 'weighted_avg')):
        scores = report[key]
        lines.append(
            f"{name:<12}"
            f"{scores['precision']:>12.{decimals}f}"
            f"{scores['recall']:>12.{decimals}f}"
            f"{scores['f1_score']:>12.{decimals}f}"
            f"{'':>10}"
        )
    lines.append(f"{'Accuracy':<12}{report['accuracy']:>12.{decimals}f}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------
# ⑯ Đường ROC — rời thang nhãn cứng, chuyển sang quét mọi ngưỡng
# ---------------------------------------------------------------------
def calculate_roc_curve(y_true, y_score, positive_label=None, label_space=None):
    """
    Đường cong ROC — quỹ tích (FPR, TPR) khi ngưỡng quyết định quét từ
    cao xuống thấp.

        TPR = TP / (TP + FN)      FPR = FP / (FP + TN)

    Parameters:
        y_true         : list nhãn thực tế
        y_score        : list điểm số dự đoán cho LỚP DƯƠNG (càng cao
                         càng nghiêng về lớp dương)
        positive_label : nhãn lớp dương

    Returns:
        false_positive_rates : list FPR, tăng dần từ 0 tới 1
        true_positive_rates  : list TPR tương ứng
        thresholds           : list ngưỡng tương ứng
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            f"Số nhãn ({len(y_true)}) phải bằng số điểm số ({len(y_score)})."
        )

    label_space = resolve_label_space(y_true, None, label_space)
    positive_label = resolve_positive_label(label_space, positive_label)

    total_positive = sum(1 for label in y_true if label == positive_label)
    total_negative = len(y_true) - total_positive
    if total_positive == 0 or total_negative == 0:
        raise ValueError(
            "Đường cong ROC cần tập dữ liệu có cả mẫu dương lẫn mẫu âm."
        )

    ordered = sorted(zip(y_score, y_true), key=lambda pair: pair[0], reverse=True)

    false_positive_rates = [0.0]
    true_positive_rates = [0.0]
    thresholds = [float('inf')]

    true_positive = 0
    false_positive = 0
    previous_score = None

    for score, label in ordered:
        if previous_score is not None and score != previous_score:
            false_positive_rates.append(false_positive / total_negative)
            true_positive_rates.append(true_positive / total_positive)
            thresholds.append(previous_score)

        if label == positive_label:
            true_positive += 1
        else:
            false_positive += 1
        previous_score = score

    false_positive_rates.append(false_positive / total_negative)
    true_positive_rates.append(true_positive / total_positive)
    thresholds.append(previous_score)

    return false_positive_rates, true_positive_rates, thresholds


# ---------------------------------------------------------------------
# ⑰ AUC — thu ⑯ về một số, tính qua hạng nên xử lý được điểm số trùng
# ---------------------------------------------------------------------
def calculate_roc_auc(y_true, y_score, positive_label=None, label_space=None):
    """
    Diện tích dưới đường cong ROC.

    Tính theo thống kê Mann-Whitney U với xử lý hạng trung bình cho các
    điểm số trùng nhau — cách này chính xác hơn phép lấy tích phân hình
    thang khi có nhiều giá trị bằng nhau.

        AUC = (Σ hạng của mẫu dương - n₊(n₊+1)/2) / (n₊ · n₋)

    Ý nghĩa: xác suất một mẫu dương lấy ngẫu nhiên được chấm điểm cao
    hơn một mẫu âm lấy ngẫu nhiên.

    Returns:
        float trong [0, 1] — 0.5 nghĩa là không phân biệt được
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            f"Số nhãn ({len(y_true)}) phải bằng số điểm số ({len(y_score)})."
        )

    label_space = resolve_label_space(y_true, None, label_space)
    positive_label = resolve_positive_label(label_space, positive_label)

    total_positive = sum(1 for label in y_true if label == positive_label)
    total_negative = len(y_true) - total_positive
    if total_positive == 0 or total_negative == 0:
        raise ValueError("AUC cần tập dữ liệu có cả mẫu dương lẫn mẫu âm.")

    ranks = calculate_average_ranks(y_score)
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, y_true) if label == positive_label
    )
    return (
        positive_rank_sum - total_positive * (total_positive + 1) / 2.0
    ) / (total_positive * total_negative)


# ---------------------------------------------------------------------
# ⑱ Hạng trung bình — chi tiết kỹ thuật mà ⑰ dựa vào để xử lý giá trị trùng
# ---------------------------------------------------------------------
def calculate_average_ranks(values):
    """
    Hạng của từng phần tử khi sắp xếp tăng dần, các giá trị bằng nhau
    nhận hạng trung bình của nhóm.

    Ví dụ: [10, 20, 20, 30] → [1.0, 2.5, 2.5, 4.0]

    Returns:
        list hạng, cùng thứ tự với values
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)

    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1

        average_rank = (position + end) / 2.0 + 1.0
        for tied_position in range(position, end + 1):
            ranks[order[tied_position]] = average_rank
        position = end + 1

    return ranks


# ---------------------------------------------------------------------
# ⑲ Log-loss — phạt dự đoán vừa sai vừa tự tin, thứ mà ⑤ không thấy
# ---------------------------------------------------------------------
def calculate_log_loss(y_true, y_score, positive_label=None, label_space=None,
                       epsilon=1e-15):
    """
    Log-loss (Binary Cross-Entropy) — phạt mạnh những dự đoán vừa sai
    vừa tự tin.

    LogLoss = -(1/n) · Σ [ y·log(p) + (1-y)·log(1-p) ]

    Parameters:
        y_score : list xác suất dự đoán cho lớp dương
        epsilon : biên kẹp để tránh log(0)
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Không thể tính Log-loss trên tập rỗng.")

    label_space = resolve_label_space(y_true, None, label_space)
    positive_label = resolve_positive_label(label_space, positive_label)

    total = 0.0
    for label, score in zip(y_true, y_score):
        probability = min(max(score, epsilon), 1.0 - epsilon)
        actual = 1.0 if label == positive_label else 0.0
        total -= (actual * math.log(probability)
                  + (1.0 - actual) * math.log(1.0 - probability))
    return total / n


# ---------------------------------------------------------------------
# ⑳ Brier — bản dịu hơn của ⑲, đo sai số bình phương của xác suất
# ---------------------------------------------------------------------
def calculate_brier_score(y_true, y_score, positive_label=None, label_space=None):
    """
    Điểm Brier — sai số bình phương trung bình của xác suất dự đoán.

    Brier = (1/n) · Σ (p_i - y_i)²

    Càng gần 0 càng tốt. Khác Log-loss, Brier phạt nhẹ hơn với các dự
    đoán sai nhưng tự tin.
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Không thể tính điểm Brier trên tập rỗng.")

    label_space = resolve_label_space(y_true, None, label_space)
    positive_label = resolve_positive_label(label_space, positive_label)

    return sum(
        (score - (1.0 if label == positive_label else 0.0)) ** 2
        for label, score in zip(y_true, y_score)
    ) / n


# ---------------------------------------------------------------------
# ㉑ Áp ngưỡng — cầu nối từ điểm số liên tục trở về nhãn để dùng ⑤–⑬
# ---------------------------------------------------------------------
def apply_threshold(y_score, threshold, positive_label, negative_label):
    """
    Quy đổi điểm số liên tục thành nhãn theo một ngưỡng.

    Parameters:
        y_score        : list điểm số cho lớp dương
        threshold      : ngưỡng quyết định
        positive_label : nhãn trả về khi điểm số >= ngưỡng
        negative_label : nhãn trả về khi điểm số < ngưỡng

    Returns:
        list nhãn dự đoán
    """
    return [
        positive_label if score >= threshold else negative_label
        for score in y_score
    ]


# ---------------------------------------------------------------------
# ㉒ Quét ngưỡng — chạy ㉑ trên cả dải để thấy chỉ số biến thiên ra sao
# ---------------------------------------------------------------------
def scan_thresholds(y_true, y_score, metric_function, thresholds=None,
                    positive_label=None, label_space=None):
    """
    Quét một dải ngưỡng và đo chỉ số tại từng ngưỡng.

    Ngưỡng mặc định 0.5 hiếm khi là lựa chọn tốt nhất khi hai lớp lệch
    nhau hoặc khi cái giá của hai loại sai lầm không như nhau.

    Parameters:
        y_true          : list nhãn thực tế
        y_score         : list điểm số cho lớp dương
        metric_function : hàm (y_true, y_pred) -> float
        thresholds      : list ngưỡng cần thử (None → 0.05 … 0.95)

    Returns:
        thresholds : list ngưỡng đã thử
        scores     : list giá trị chỉ số tương ứng
    """
    label_space = resolve_label_space(y_true, None, label_space)
    positive_label = resolve_positive_label(label_space, positive_label)
    negative_label = [
        label for label in label_space if label != positive_label
    ][0]

    if thresholds is None:
        thresholds = [index / 100.0 for index in range(5, 100, 5)]

    scores = []
    for threshold in thresholds:
        y_pred = apply_threshold(y_score, threshold, positive_label, negative_label)
        scores.append(metric_function(y_true, y_pred))
    return thresholds, scores


# ---------------------------------------------------------------------
# ㉓ Chọn ngưỡng — lấy đỉnh của ㉒; CHỈ được chạy trên tập validate
# ---------------------------------------------------------------------
def find_best_threshold(y_true, y_score, metric_function, thresholds=None,
                        positive_label=None, label_space=None):
    """
    Tìm ngưỡng cực đại hóa một chỉ số.

    LƯU Ý CHỐNG RÒ RỈ: chỉ được dò ngưỡng trên tập VALIDATE, sau đó áp
    dụng nguyên vẹn cho tập TEST. Dò ngưỡng trên chính tập test sẽ cho
    kết quả tốt giả tạo.

    Returns:
        best_threshold : ngưỡng tốt nhất
        best_score     : giá trị chỉ số tại ngưỡng đó
    """
    thresholds, scores = scan_thresholds(
        y_true, y_score, metric_function, thresholds, positive_label, label_space
    )
    best_position = max(range(len(scores)), key=lambda index: scores[index])
    return thresholds[best_position], scores[best_position]


# ---------------------------------------------------------------------
# ㉔ Gộp: chạy lại đúng chuỗi ①→⑳ và trả về toàn bộ chỉ số
# ---------------------------------------------------------------------
def calculate_all_classification_metrics(y_true, y_pred, y_score=None,
                                         positive_label=None, label_space=None):
    """
    Tính trọn bộ chỉ số phân loại và trả về dưới dạng dict.

    Parameters:
        y_true         : list nhãn thực tế
        y_pred         : list nhãn dự đoán
        y_score        : list điểm số cho lớp dương (tùy chọn) — có thì
                         mới tính được ROC-AUC, Log-loss và Brier
        positive_label : nhãn lớp dương

    Returns:
        dict { 'accuracy', 'error_rate', 'precision', 'recall',
               'specificity', 'f1_score', 'balanced_accuracy',
               'matthews_correlation', 'confusion_matrix', 'label_space',
               'positive_label'
               [, 'roc_auc', 'log_loss', 'brier_score'] }
    """
    label_space = resolve_label_space(y_true, y_pred, label_space)
    positive_label = resolve_positive_label(label_space, positive_label)
    matrix, _ = build_confusion_matrix(y_true, y_pred, label_space)

    results = {
        'accuracy':             calculate_accuracy(y_true, y_pred),
        'error_rate':           calculate_error_rate(y_true, y_pred),
        'precision':            calculate_precision(
                                    y_true, y_pred, positive_label, label_space),
        'recall':               calculate_recall(
                                    y_true, y_pred, positive_label, label_space),
        'specificity':          calculate_specificity(
                                    y_true, y_pred, positive_label, label_space),
        'f1_score':             calculate_f1_score(
                                    y_true, y_pred, positive_label, label_space),
        'balanced_accuracy':    calculate_balanced_accuracy(
                                    y_true, y_pred, positive_label, label_space),
        'matthews_correlation': calculate_matthews_correlation(
                                    y_true, y_pred, positive_label, label_space),
        'confusion_matrix':     matrix,
        'label_space':          label_space,
        'positive_label':       positive_label,
    }

    if y_score is not None:
        results['roc_auc'] = calculate_roc_auc(
            y_true, y_score, positive_label, label_space)
        results['log_loss'] = calculate_log_loss(
            y_true, y_score, positive_label, label_space)
        results['brier_score'] = calculate_brier_score(
            y_true, y_score, positive_label, label_space)

    return results
