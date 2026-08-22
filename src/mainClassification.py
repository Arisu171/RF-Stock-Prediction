# =====================================================================
# Main Classification — chạy đầu-cuối NHÁNH A: phân loại xu hướng
# =====================================================================
# Script này KHÔNG chứa giá trị cụ thể nào của đề tài. Mọi đường dẫn,
# tên cột, danh sách chỉ báo và siêu tham số đều đọc từ file cấu hình
# JSON truyền qua tham số dòng lệnh.
#
#   python src/mainClassification.py --config config/classification.json
#
# Thứ tự khai báo bám đúng trình tự chạy:
#
#   ①  Đọc cấu hình           — nguồn sự thật duy nhất của lượt chạy
#   ②  Khởi tạo mô hình       — dịch nhánh 'model' thành đối tượng rừng
#   ③  Đánh giá một tập       — tính trọn bộ chỉ số cho một tập dữ liệu
#   ④  So với các mốc đối chứng — điều kiện cần để kết quả có ý nghĩa
#   ⑤  Vẽ và lưu đồ thị       — toàn bộ hình đầu ra của nhánh A
#   ⑥  Mạch chính             — nối ①→⑤ theo đúng thứ tự
#
# Cờ --save-model đóng gói mô hình đã huấn luyện kèm CÔNG THỨC ĐẶC
# TRƯNG và siêu dữ liệu, để mainPredict.py dùng lại mà không phải
# huấn luyện lại.
# =====================================================================

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from libraries import modelStore, rfPlot
from libraries.randomForest import RandomForestClassifier
from pipeline import experiment, labeling, modelBundle
from utilities import metricsClassification


# ---------------------------------------------------------------------
# ① Đọc cấu hình — mọi giá trị cụ thể của lượt chạy vào từ đây
# ---------------------------------------------------------------------
def load_configuration(path):
    """
    Đọc file cấu hình JSON.

    Parameters:
        path : đường dẫn tới file cấu hình

    Returns:
        dict cấu hình
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


# ---------------------------------------------------------------------
# ② Khởi tạo mô hình — dịch nhánh 'model' của ① thành đối tượng rừng
# ---------------------------------------------------------------------
def create_model(config):
    """
    Tạo một rừng ngẫu nhiên phân loại CHƯA huấn luyện theo cấu hình.

    Returns:
        RandomForestClassifier
    """
    settings = dict(config['model'])
    return RandomForestClassifier(**settings)


# ---------------------------------------------------------------------
# ③ Đánh giá một tập — dùng chung cho train, validate và test
# ---------------------------------------------------------------------
def evaluate_split(model, samples, targets, positive_label):
    """
    Tính trọn bộ chỉ số phân loại trên một tập dữ liệu.

    Returns:
        dict như metricsClassification.calculate_all_classification_metrics(),
        bổ sung 'report' (báo cáo theo từng lớp), 'predictions' và 'scores'
    """
    predictions = model.predict(samples)
    scores = model.predict_scores(samples, positive_label=positive_label)

    results = metricsClassification.calculate_all_classification_metrics(
        targets, predictions, scores, positive_label=positive_label
    )
    results['report'] = metricsClassification.build_classification_report(
        targets, predictions
    )
    results['predictions'] = predictions
    results['scores'] = scores
    return results


# ---------------------------------------------------------------------
# ④ Mốc đối chứng — kết quả chỉ có nghĩa khi VƯỢT được các mốc này
# ---------------------------------------------------------------------
def evaluate_baselines(train_targets, targets, samples, feature_names,
                       positive_label, negative_label):
    """
    Chấm điểm các mô hình đối chứng tầm thường trên cùng một tập.

    Ba mốc:
        - majority    : luôn đoán lớp chiếm đa số của tập HUẤN LUYỆN
        - persistence : lặp lại chiều biến động của bước liền trước
        - alternating : đoán xen kẽ, tương đương tung đồng xu có quy luật

    Vượt được ba mốc này là điều kiện CẦN, chưa phải điều kiện đủ.

    Returns:
        dict { tên_đối_chứng: accuracy }
    """
    balance = labeling.calculate_class_balance(train_targets)
    majority_label = balance['majority_label']

    baselines = {
        'majority': metricsClassification.calculate_accuracy(
            targets, [majority_label] * len(targets)
        ),
        'alternating': metricsClassification.calculate_accuracy(
            targets,
            [positive_label if index % 2 else negative_label
             for index in range(len(targets))]
        ),
    }

    if 'return_lag_1' in feature_names:
        position = feature_names.index('return_lag_1')
        persistence = [
            positive_label if sample[position] > 0 else negative_label
            for sample in samples
        ]
        baselines['persistence'] = metricsClassification.calculate_accuracy(
            targets, persistence
        )
    return baselines


# ---------------------------------------------------------------------
# ⑤ Vẽ và lưu — toàn bộ hình đầu ra của nhánh A gom về một chỗ
# ---------------------------------------------------------------------
def save_figures(config, model, dataset, evaluations, walk_forward,
                 output_dir, prefix):
    """
    Vẽ và lưu bộ đồ thị đánh giá.

    Returns:
        list đường dẫn các file đã lưu
    """
    import matplotlib
    matplotlib.use('Agg')

    settings = config.get('evaluation', {})
    feature_names = dataset['feature_names']
    saved = []

    def target(name):
        return os.path.join(output_dir, f"{prefix}_{name}.png")

    # Tầm quan trọng đặc trưng — hai cách đo đặt cạnh nhau
    impurity_importances = model.calculate_feature_importances()
    validation_samples, validation_targets = dataset['validation']
    permutation_importances = model.calculate_permutation_importances(
        validation_samples, validation_targets,
        num_repeats=settings.get('permutation_repeats', 5),
        random_state=config['model'].get('random_state'),
    )
    total = sum(max(value, 0.0) for value in permutation_importances)
    normalized = [
        max(value, 0.0) / total if total else 0.0
        for value in permutation_importances
    ]

    rfPlot.plot_importance_comparison(
        feature_names, impurity_importances, normalized,
        first_label='MDI', second_label='Permutation (validate)',
        top_k=settings.get('top_features', 15),
        filename=target('feature_importance'),
    )
    saved.append(target('feature_importance'))

    # Đường cong OOB theo số cây
    tree_counts, errors = model.calculate_out_of_bag_error_curve()
    rfPlot.plot_out_of_bag_curve(
        tree_counts, errors, filename=target('oob_curve')
    )
    saved.append(target('oob_curve'))

    # Ma trận nhầm lẫn và ROC trên tập kiểm tra
    test_evaluation = evaluations['test']
    rfPlot.plot_confusion_matrix(
        test_evaluation['confusion_matrix'],
        [str(label) for label in test_evaluation['label_space']],
        title='Ma trận nhầm lẫn — tập test',
        filename=target('confusion_matrix'),
    )
    saved.append(target('confusion_matrix'))

    test_samples, test_targets = dataset['test']
    false_positive_rates, true_positive_rates, _ = \
        metricsClassification.calculate_roc_curve(
            test_targets, test_evaluation['scores']
        )
    rfPlot.plot_roc_curve(
        false_positive_rates, true_positive_rates,
        test_evaluation['roc_auc'],
        title='Đường cong ROC — tập test',
        filename=target('roc_curve'),
    )
    saved.append(target('roc_curve'))

    # Chỉ số theo ngưỡng quyết định, dò trên tập validate
    validation_evaluation = evaluations['validation']
    thresholds, accuracy_values = metricsClassification.scan_thresholds(
        validation_targets, validation_evaluation['scores'],
        metricsClassification.calculate_accuracy,
    )
    _, f1_values = metricsClassification.scan_thresholds(
        validation_targets, validation_evaluation['scores'],
        metricsClassification.calculate_f1_score,
    )
    _, balanced_values = metricsClassification.scan_thresholds(
        validation_targets, validation_evaluation['scores'],
        metricsClassification.calculate_balanced_accuracy,
    )
    rfPlot.plot_threshold_curve(
        thresholds,
        {'Accuracy': accuracy_values, 'F1': f1_values,
         'Balanced accuracy': balanced_values},
        title='Chỉ số theo ngưỡng quyết định — tập validate',
        filename=target('threshold_curve'),
    )
    saved.append(target('threshold_curve'))

    # Độ ổn định qua các vòng walk-forward
    rfPlot.plot_fold_scores(
        [f"Vòng {position}" for position in
         range(1, len(walk_forward['fold_scores']) + 1)],
        walk_forward['fold_scores'],
        title='Accuracy qua từng vòng kiểm định tiến dần',
        y_label='Accuracy',
        filename=target('walk_forward'),
    )
    saved.append(target('walk_forward'))

    # Phân bố lớp
    balance = labeling.calculate_class_balance(dataset['targets'])
    rfPlot.plot_class_distribution(
        list(balance['counts']), list(balance['counts'].values()),
        title='Phân bố lớp trên toàn bộ dữ liệu',
        filename=target('class_distribution'),
    )
    saved.append(target('class_distribution'))
    return saved


# ---------------------------------------------------------------------
# ⑥ Mạch chính — nối ①→⑤ và in báo cáo ra màn hình
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Huấn luyện và đánh giá mô hình phân loại xu hướng.'
    )
    parser.add_argument('--config', required=True,
                        help='Đường dẫn tới file cấu hình JSON.')
    parser.add_argument('--no-figures', action='store_true',
                        help='Bỏ qua bước vẽ và lưu đồ thị.')
    parser.add_argument('--save-model', metavar='ĐƯỜNG_DẪN',
                        help='Lưu gói mô hình ra file JSON để dùng lại '
                             'với mainPredict.py.')
    arguments = parser.parse_args()

    config = load_configuration(arguments.config)
    label_settings = config['labeling']
    positive_label = label_settings.get('positive_label', 1)
    negative_label = label_settings.get('negative_label', 0)

    print('=' * 70)
    print(f"THÍ NGHIỆM: {config.get('description', '(không mô tả)')}")
    print(f"Dữ liệu   : {config['dataset']['path']} "
          f"({config['dataset'].get('label', '')})")
    print(f"Tầm nhìn  : {label_settings.get('horizon', 1)} bước")
    print('=' * 70)

    # ── Chuẩn bị dữ liệu ────────────────────────────────────────────
    print('\n[1] CHUẨN BỊ DỮ LIỆU')
    dataset = experiment.prepare_dataset(config, PROJECT_ROOT)
    balance = labeling.calculate_class_balance(dataset['targets'])
    print(f"Cân bằng lớp: {balance['counts']} — "
          f"lớp đa số chiếm {balance['majority_ratio']:.2%}")

    print('\n[2] TÁCH TẬP THEO THỜI GIAN')
    parts = experiment.split_prepared_dataset(config, dataset)
    dataset['validation'] = parts['validation']
    dataset['test'] = parts['test']

    train_samples, train_targets = parts['train']
    validation_samples, validation_targets = parts['validation']
    test_samples, test_targets = parts['test']

    # ── Kiểm định tiến dần ──────────────────────────────────────────
    print('\n[3] KIỂM ĐỊNH TIẾN DẦN (WALK-FORWARD)')
    walk_forward = experiment.run_walk_forward(
        config,
        train_samples + validation_samples,
        train_targets + validation_targets,
        model_factory=lambda: create_model(config),
        score_function=metricsClassification.calculate_accuracy,
    )
    print(f"  → Trung bình = {walk_forward['mean']:.4f} "
          f"± {walk_forward['standard_deviation']:.4f}")

    # ── Huấn luyện mô hình cuối ─────────────────────────────────────
    print('\n[4] HUẤN LUYỆN MÔ HÌNH CUỐI')
    model = create_model(config)
    model.fit(train_samples, train_targets)

    structure = model.describe()
    print(f"  Số cây: {structure['num_trees']} | "
          f"m = {structure['max_features_resolved']}/{structure['num_features']} | "
          f"độ sâu TB = {structure['average_depth']:.1f} | "
          f"lá TB = {structure['average_leaves']:.1f}")
    print(f"  Tỷ lệ OOB thực tế: {structure['average_out_of_bag_ratio']:.4f} "
          f"(lý thuyết 1/e ≈ 0.3679)")
    print(f"  Lỗi OOB: {model.calculate_out_of_bag_error():.4f}")

    # ── Đánh giá ────────────────────────────────────────────────────
    print('\n[5] ĐÁNH GIÁ')
    evaluations = {}
    for name, (samples, targets) in (('train', parts['train']),
                                     ('validation', parts['validation']),
                                     ('test', parts['test'])):
        evaluations[name] = evaluate_split(model, samples, targets, positive_label)

    header = (f"{'Tập':<12}{'Accuracy':>10}{'ROC-AUC':>10}{'F1(+)':>10}"
              f"{'F1(-)':>10}{'BalAcc':>10}{'MCC':>10}")
    print(header)
    print('-' * len(header))
    for name in ('train', 'validation', 'test'):
        results = evaluations[name]
        per_class = results['report']['per_class']
        print(f"{name:<12}{results['accuracy']:>10.4f}{results['roc_auc']:>10.4f}"
              f"{per_class[positive_label]['f1_score']:>10.4f}"
              f"{per_class[negative_label]['f1_score']:>10.4f}"
              f"{results['balanced_accuracy']:>10.4f}"
              f"{results['matthews_correlation']:>+10.4f}")

    print('\nBáo cáo chi tiết — tập validate:')
    print(metricsClassification.format_classification_report(
        evaluations['validation']['report']
    ))

    # ── Đối chứng ───────────────────────────────────────────────────
    print('\n[6] SO VỚI CÁC MỐC ĐỐI CHỨNG')
    for split_name, (samples, targets) in (('validate', parts['validation']),
                                           ('test', parts['test'])):
        baselines = evaluate_baselines(
            train_targets, targets, samples, dataset['feature_names'],
            positive_label, negative_label
        )
        model_accuracy = evaluations[
            'validation' if split_name == 'validate' else 'test'
        ]['accuracy']
        parts_text = '  '.join(
            f"{name} = {value:.4f}" for name, value in sorted(baselines.items())
        )
        best_baseline = max(baselines.values())
        verdict = 'VƯỢT' if model_accuracy > best_baseline else 'KHÔNG VƯỢT'
        print(f"  {split_name:<9} mô hình = {model_accuracy:.4f} | {parts_text} "
              f"→ {verdict}")

    # ── Ngưỡng quyết định ───────────────────────────────────────────
    print('\n[7] DÒ NGƯỠNG QUYẾT ĐỊNH (chỉ trên tập validate)')
    threshold, score = metricsClassification.find_best_threshold(
        validation_targets, evaluations['validation']['scores'],
        metricsClassification.calculate_balanced_accuracy,
        positive_label=positive_label,
    )
    tuned_test = metricsClassification.apply_threshold(
        evaluations['test']['scores'], threshold, positive_label, negative_label
    )
    print(f"  Ngưỡng tốt nhất theo balanced accuracy: {threshold:.2f} "
          f"(validate = {score:.4f})")
    print(f"  Áp lên tập test: accuracy = "
          f"{metricsClassification.calculate_accuracy(test_targets, tuned_test):.4f} | "
          f"balanced accuracy = "
          f"{metricsClassification.calculate_balanced_accuracy(test_targets, tuned_test):.4f}")

    # ── Lưu gói mô hình ─────────────────────────────────────────────
    if arguments.save_model:
        print('\n[8] LƯU GÓI MÔ HÌNH')
        size = modelStore.count_parameters(model)
        training_summary = {
            'data_label':   config['dataset'].get('label', ''),
            'data_path':    config['dataset']['path'],
            'num_samples':  len(train_samples),
            'period':       f"{parts['summary']['train']['key_range'][0]} → "
                            f"{parts['summary']['train']['key_range'][1]}",
            'horizon':      label_settings.get('horizon', 1),
            'parameters_per_sample': size['parameters'] / len(train_samples),
        }
        recorded_metrics = {
            'validation_accuracy': evaluations['validation']['accuracy'],
            'validation_roc_auc':  evaluations['validation']['roc_auc'],
            'test_accuracy':       evaluations['test']['accuracy'],
            'test_roc_auc':        evaluations['test']['roc_auc'],
            'out_of_bag_error':    model.calculate_out_of_bag_error(),
        }

        bundle = modelBundle.build_bundle(
            model, config, dataset['feature_names'],
            training_summary=training_summary,
            metrics=recorded_metrics,
        )
        bundle['threshold'] = threshold

        path = experiment.resolve_path(arguments.save_model, PROJECT_ROOT)
        modelBundle.save_bundle(bundle, path)
        print(f"  Đã lưu: {path}")
        print(f"  Dung lượng: {os.path.getsize(path) / 1024:.0f} KB | "
              f"{size['parameters']:,} tham số | "
              f"ngưỡng quyết định {threshold:.2f}")

    # ── Đồ thị ──────────────────────────────────────────────────────
    if not arguments.no_figures:
        print('\n[8] LƯU ĐỒ THỊ')
        output_settings = config.get('output', {})
        output_dir = experiment.resolve_path(
            output_settings.get('figure_dir', 'data/output'), PROJECT_ROOT
        )
        os.makedirs(output_dir, exist_ok=True)
        saved = save_figures(
            config, model, dataset, evaluations, walk_forward,
            output_dir, output_settings.get('prefix', 'classification')
        )
        for path in saved:
            print(f"  {os.path.basename(path)}")

    print('\nHoàn tất.')


if __name__ == '__main__':
    main()
