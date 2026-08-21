# =====================================================================
# Test Metrics — kiểm thử bộ chỉ số đánh giá
# =====================================================================
# Chỉ số đánh giá là thước đo của cả dự án. Một thước đo sai khiến mọi
# kết luận rút ra từ nó đều sai, mà lại rất khó phát hiện vì chương
# trình vẫn chạy trơn tru và vẫn in ra những con số trông hợp lý.
#
# Vì vậy phần lớn test ở đây đối chiếu với giá trị TÍNH TAY trên những
# ví dụ đủ nhỏ để kiểm tra bằng mắt.
#
# Thứ tự khai báo:
#
#   ①  Chỉ số hồi quy      — SSE, MSE, RMSE, MAE, R², MAPE
#   ②  Độ chính xác về hướng
#   ③  Ma trận nhầm lẫn và bốn đại lượng TP/FP/FN/TN
#   ④  Precision, Recall, F1 và các cách gộp
#   ⑤  ROC, AUC và các chỉ số dựa trên điểm số liên tục
#   ⑥  Ngưỡng quyết định
#   ⑦  Trường hợp biên
# =====================================================================

import pytest

from utilities import metrics, metricsClassification


# ---------------------------------------------------------------------
# ① Chỉ số hồi quy — đối chiếu với giá trị tính tay
# ---------------------------------------------------------------------
def test_regression_metrics_on_hand_example():
    """
    y = [1, 2, 3, 4], ŷ = [1, 2, 4, 4]
    sai số = [0, 0, -1, 0] → SSE = 1, MSE = 0.25, RMSE = 0.5, MAE = 0.25
    """
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.0, 4.0, 4.0]

    assert metrics.calculate_sse(y_true, y_pred) == pytest.approx(1.0)
    assert metrics.calculate_mse(y_true, y_pred) == pytest.approx(0.25)
    assert metrics.calculate_rmse(y_true, y_pred) == pytest.approx(0.5)
    assert metrics.calculate_mae(y_true, y_pred) == pytest.approx(0.25)


def test_r_squared_of_perfect_prediction_is_one():
    y_true = [1.0, 2.0, 3.0, 4.0]
    assert metrics.calculate_r_squared(y_true, y_true) == pytest.approx(1.0)


def test_r_squared_of_mean_prediction_is_zero():
    """
    Dự đoán bằng trung bình cho R² = 0 theo đúng định nghĩa. Đây là mốc
    đối chứng tối thiểu của mọi mô hình hồi quy.
    """
    y_true = [1.0, 2.0, 3.0, 4.0, 10.0]
    mean_value = sum(y_true) / len(y_true)

    assert metrics.calculate_r_squared(
        y_true, [mean_value] * len(y_true)) == pytest.approx(0.0)


def test_r_squared_can_be_negative():
    """R² âm nghĩa là mô hình tệ hơn cả việc đoán bằng hằng số."""
    assert metrics.calculate_r_squared([1.0, 2.0, 3.0], [10.0, 10.0, 10.0]) < 0


def test_mape_skips_values_near_zero():
    """
    MAPE bỏ qua các mẫu có giá trị thực tế gần 0 vì phép chia làm phần
    trăm sai số phóng đại vô hạn.
    """
    assert metrics.calculate_mape([0.0, 0.0], [1.0, 1.0]) is None
    assert metrics.calculate_mape([100.0, 200.0],
                                  [110.0, 180.0]) == pytest.approx(10.0)


def test_adjusted_r_squared_penalises_extra_parameters():
    """R²_adj phải nhỏ hơn R² khi mô hình có tham số."""
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    y_pred = [1.1, 2.1, 2.9, 4.2, 4.8, 6.9]

    plain = metrics.calculate_r_squared(y_true, y_pred)
    adjusted = metrics.calculate_adjusted_r_squared(y_true, y_pred, 2)

    assert adjusted < plain


def test_adjusted_r_squared_returns_none_without_degrees_of_freedom():
    """Không đủ bậc tự do thì trả về None thay vì một con số vô nghĩa."""
    assert metrics.calculate_adjusted_r_squared(
        [1.0, 2.0, 3.0], [1.0, 2.0, 3.1], 5) is None


# ---------------------------------------------------------------------
# ② Độ chính xác về hướng — cầu nối giữa hồi quy và phân loại
# ---------------------------------------------------------------------
def test_directional_accuracy_with_explicit_reference():
    """
    Mốc tham chiếu 0: đoán đúng hướng nghĩa là đoán đúng DẤU.
    Ba trong bốn mẫu cùng dấu → 0.75.
    """
    y_true = [1.0, -1.0, 2.0, -2.0]
    y_pred = [0.5, -0.5, -0.1, -3.0]

    assert metrics.calculate_directional_accuracy(
        y_true, y_pred, reference_values=[0.0] * 4) == pytest.approx(0.75)


def test_directional_accuracy_ignores_flat_moves():
    """Mẫu không có biến động thực tế thì không tính vào mẫu số."""
    assert metrics.calculate_directional_accuracy(
        [1.0, 0.0], [5.0, -5.0], reference_values=[0.0, 0.0]
    ) == pytest.approx(1.0)


def test_directional_accuracy_defaults_to_previous_observation():
    """Không truyền mốc thì so hướng giữa hai quan sát liên tiếp."""
    assert metrics.calculate_directional_accuracy(
        [10.0, 11.0, 12.0], [10.0, 11.5, 12.5]) == pytest.approx(1.0)


# ---------------------------------------------------------------------
# ③ Ma trận nhầm lẫn — bảng gốc, mọi chỉ số đều là tỷ số rút từ đây
# ---------------------------------------------------------------------
def test_confusion_matrix_layout():
    """matrix[i][j] = số mẫu THỰC TẾ lớp i được DỰ ĐOÁN là lớp j."""
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]

    matrix, label_space = metricsClassification.build_confusion_matrix(y_true, y_pred)

    assert label_space == [0, 1]
    assert matrix == [[1, 1], [1, 2]]


def test_binary_counts_match_confusion_matrix():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]

    counts = metricsClassification.calculate_binary_counts(
        y_true, y_pred, positive_label=1)

    assert counts == {'true_positive': 2, 'false_positive': 1,
                      'false_negative': 1, 'true_negative': 1}


def test_confusion_matrix_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        metricsClassification.build_confusion_matrix([0, 1], [0, 1, 1])


# ---------------------------------------------------------------------
# ④ Precision, Recall, F1 — ba góc nhìn và các cách gộp chúng
# ---------------------------------------------------------------------
def test_precision_recall_f1_on_hand_example():
    """TP=2, FP=1, FN=1 → P = 2/3, R = 2/3, F1 = 2/3."""
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]

    assert metricsClassification.calculate_precision(
        y_true, y_pred, 1) == pytest.approx(2 / 3)
    assert metricsClassification.calculate_recall(
        y_true, y_pred, 1) == pytest.approx(2 / 3)
    assert metricsClassification.calculate_f1_score(
        y_true, y_pred, 1) == pytest.approx(2 / 3)
    assert metricsClassification.calculate_accuracy(y_true, y_pred) == pytest.approx(0.6)


def test_precision_is_zero_when_nothing_predicted_positive():
    """Không dự đoán mẫu dương nào thì Precision quy ước bằng 0."""
    assert metricsClassification.calculate_precision(
        [0, 1, 1], [0, 0, 0], 1) == pytest.approx(0.0)


def test_f1_punishes_imbalance_between_precision_and_recall():
    """
    Trung bình điều hoà phạt nặng trường hợp một trong hai chỉ số quá
    thấp — đó là lý do dùng F1 thay vì trung bình cộng.
    """
    # Precision = 1.0 nhưng Recall = 0.2 → F1 ≈ 0.333, thấp hơn hẳn 0.6
    y_true = [1] * 10 + [0] * 10
    y_pred = [1] * 2 + [0] * 8 + [0] * 10

    f1 = metricsClassification.calculate_f1_score(y_true, y_pred, 1)
    arithmetic_mean = (1.0 + 0.2) / 2

    assert f1 == pytest.approx(1 / 3, abs=1e-9)
    assert f1 < arithmetic_mean


def test_fbeta_shifts_weight_between_precision_and_recall():
    """β > 1 coi trọng Recall, β < 1 coi trọng Precision."""
    y_true = [1] * 10 + [0] * 10
    y_pred = [1] * 2 + [0] * 18

    recall_weighted = metricsClassification.calculate_fbeta_score(
        y_true, y_pred, beta=2.0, positive_label=1)
    precision_weighted = metricsClassification.calculate_fbeta_score(
        y_true, y_pred, beta=0.5, positive_label=1)

    assert recall_weighted < precision_weighted


def test_balanced_accuracy_ignores_class_imbalance():
    """
    Với dữ liệu lệch 90/10, đoán toàn lớp đa số cho Accuracy 0.9 nhưng
    Balanced Accuracy chỉ 0.5 — đúng bản chất "không học được gì".
    """
    y_true = [0] * 90 + [1] * 10
    y_pred = [0] * 100

    assert metricsClassification.calculate_accuracy(y_true, y_pred) == pytest.approx(0.9)
    assert metricsClassification.calculate_balanced_accuracy(
        y_true, y_pred, 1) == pytest.approx(0.5)


def test_matthews_correlation_range():
    """MCC bằng 1 khi dự đoán hoàn hảo và -1 khi dự đoán ngược hoàn toàn."""
    y_true = [0, 0, 1, 1]

    assert metricsClassification.calculate_matthews_correlation(
        y_true, [0, 0, 1, 1], 1) == pytest.approx(1.0)
    assert metricsClassification.calculate_matthews_correlation(
        y_true, [1, 1, 0, 0], 1) == pytest.approx(-1.0)


def test_classification_report_covers_every_class():
    """Báo cáo phải có mục cho từng lớp kèm số mẫu thực tế."""
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0]

    report = metricsClassification.build_classification_report(y_true, y_pred)

    assert set(report['per_class']) == {0, 1}
    assert report['per_class'][0]['support'] == 3
    assert report['per_class'][1]['support'] == 2
    assert report['accuracy'] == pytest.approx(0.6)
    assert 'Precision' in metricsClassification.format_classification_report(report)


# ---------------------------------------------------------------------
# ⑤ ROC và AUC — nhóm chỉ số dựa trên điểm số liên tục
# ---------------------------------------------------------------------
def test_roc_auc_on_textbook_example():
    """Ví dụ kinh điển: y = [0,0,1,1], điểm = [0.1,0.4,0.35,0.8] → AUC = 0.75."""
    assert metricsClassification.calculate_roc_auc(
        [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_roc_auc_of_perfect_ranking_is_one():
    assert metricsClassification.calculate_roc_auc(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_roc_auc_of_constant_score_is_half():
    """Điểm số như nhau cho mọi mẫu thì không phân biệt được gì."""
    assert metricsClassification.calculate_roc_auc(
        [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_average_ranks_handles_ties():
    """Giá trị bằng nhau nhận hạng trung bình của cả nhóm."""
    assert metricsClassification.calculate_average_ranks(
        [10, 20, 20, 30]) == pytest.approx([1.0, 2.5, 2.5, 4.0])


def test_roc_curve_starts_and_ends_at_corners():
    """Đường ROC phải đi từ (0,0) tới (1,1)."""
    false_positive_rates, true_positive_rates, _ = \
        metricsClassification.calculate_roc_curve(
            [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])

    assert (false_positive_rates[0], true_positive_rates[0]) == (0.0, 0.0)
    assert (false_positive_rates[-1], true_positive_rates[-1]) == (1.0, 1.0)


def test_roc_requires_both_classes():
    """Không có cả hai lớp thì ROC không xác định."""
    with pytest.raises(ValueError):
        metricsClassification.calculate_roc_auc([1, 1, 1], [0.1, 0.5, 0.9])


def test_log_loss_punishes_confident_mistakes():
    """Dự đoán vừa sai vừa tự tin bị phạt nặng hơn dự đoán sai mà lưỡng lự."""
    confident_mistake = metricsClassification.calculate_log_loss(
        [1, 1], [0.01, 0.01])
    hesitant_mistake = metricsClassification.calculate_log_loss(
        [1, 1], [0.45, 0.45])

    assert confident_mistake > hesitant_mistake


def test_brier_score_of_perfect_prediction_is_zero():
    assert metricsClassification.calculate_brier_score(
        [0, 1], [0.0, 1.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------
# ⑥ Ngưỡng quyết định — quay từ điểm số liên tục về nhãn cứng
# ---------------------------------------------------------------------
def test_apply_threshold_boundary_is_inclusive():
    """Quy ước: điểm số BẰNG ngưỡng được xếp vào lớp dương."""
    assert metricsClassification.apply_threshold(
        [0.4, 0.5, 0.6], 0.5, 1, 0) == [0, 1, 1]


def test_find_best_threshold_beats_default():
    """
    Với dữ liệu lệch, ngưỡng tối ưu khác 0.5 và cho điểm cao hơn ngưỡng
    mặc định — đó là lý do phải dò ngưỡng.
    """
    y_true = [0] * 8 + [1] * 2
    y_score = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6]

    best_threshold, best_score = metricsClassification.find_best_threshold(
        y_true, y_score, metricsClassification.calculate_f1_score, positive_label=1)

    default_score = metricsClassification.calculate_f1_score(
        y_true, metricsClassification.apply_threshold(y_score, 0.5, 1, 0), 1)

    assert best_score >= default_score
    assert 0.0 < best_threshold < 1.0


def test_scan_thresholds_returns_matching_lengths():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.9]

    thresholds, scores = metricsClassification.scan_thresholds(
        y_true, y_score, metricsClassification.calculate_accuracy, positive_label=1)

    assert len(thresholds) == len(scores)


# ---------------------------------------------------------------------
# ⑦ Trường hợp biên — lỗi phải nổ rõ ràng thay vì trả về giá trị lạ
# ---------------------------------------------------------------------
def test_empty_input_raises():
    with pytest.raises(ValueError):
        metrics.calculate_mse([], [])
    with pytest.raises(ValueError):
        metricsClassification.calculate_accuracy([], [])


def test_constant_target_makes_r_squared_undefined():
    """SST = 0 thì R² không xác định, phải báo lỗi thay vì chia cho 0."""
    with pytest.raises(ValueError):
        metrics.calculate_r_squared([5.0, 5.0, 5.0], [4.0, 5.0, 6.0])


def test_unknown_positive_label_raises():
    with pytest.raises(ValueError):
        metricsClassification.calculate_precision([0, 1], [0, 1], positive_label=9)


def test_calculate_all_metrics_returns_expected_keys():
    results = metrics.calculate_all_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2], 2)

    for key in ('sst', 'sse', 'mse', 'rmse', 'mae', 'mape', 'r_squared',
                'adjusted_r_squared'):
        assert key in results


def test_calculate_all_classification_metrics_returns_expected_keys():
    results = metricsClassification.calculate_all_classification_metrics(
        [0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.7, 0.9], positive_label=1)

    for key in ('accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
                'log_loss', 'brier_score', 'confusion_matrix',
                'matthews_correlation'):
        assert key in results
