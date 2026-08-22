# =====================================================================
# Main Regression — chạy đầu-cuối NHÁNH B: hồi quy tỷ suất biến động
# =====================================================================
# Cùng nguyên tắc với mainClassification.py: script không chứa giá trị
# cụ thể nào của đề tài, mọi thứ đến từ file cấu hình JSON.
#
#   python src/mainRegression.py --config config/regression.json
#
# Nhánh B bổ trợ cho nhánh A ở hai điểm:
#   - Dùng được bộ chỉ số hồi quy sẵn có (RMSE, MAE, R², MAPE).
#   - Cho phép SUY NGƯỢC ra chiều biến động từ giá trị dự đoán, nhờ đó
#     so sánh trực tiếp với nhánh A qua chỉ số Directional Accuracy.
#
# Thứ tự khai báo bám đúng trình tự chạy:
#
#   ①  Đọc cấu hình
#   ②  Khởi tạo mô hình      — rừng hồi quy
#   ③  Đánh giá một tập      — chỉ số hồi quy + độ chính xác về hướng
#   ④  Mốc đối chứng         — dự đoán bằng hằng số và bằng giá trị trước
#   ⑤  Vẽ và lưu đồ thị
#   ⑥  Mạch chính
# =====================================================================

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from libraries import rfPlot
from libraries.randomForest import RandomForestRegressor
from pipeline import experiment
from utilities import metrics


# ---------------------------------------------------------------------
# ① Đọc cấu hình — mọi giá trị cụ thể của lượt chạy vào từ đây
# ---------------------------------------------------------------------
def load_configuration(path):
    """
    Đọc file cấu hình JSON, có hỗ trợ kế thừa qua khoá 'extends'.

    Uỷ quyền cho experiment.load_configuration() để nhiều mã cùng chia
    sẻ đúng một bộ phương pháp và siêu tham số — xem giải thích ở ⑨ của
    pipeline/experiment.py.

    Parameters:
        path : đường dẫn tới file cấu hình

    Returns:
        dict cấu hình đã hợp nhất
    """
    return experiment.load_configuration(path, PROJECT_ROOT)


# ---------------------------------------------------------------------
# ② Khởi tạo mô hình — dịch nhánh 'model' của ① thành rừng hồi quy
# ---------------------------------------------------------------------
def create_model(config):
    """
    Tạo một rừng ngẫu nhiên hồi quy CHƯA huấn luyện theo cấu hình.

    Returns:
        RandomForestRegressor
    """
    return RandomForestRegressor(**dict(config['model']))


# ---------------------------------------------------------------------
# ③ Đánh giá một tập — chỉ số độ lớn sai số VÀ độ chính xác về hướng
# ---------------------------------------------------------------------
def evaluate_split(model, samples, targets):
    """
    Tính bộ chỉ số hồi quy trên một tập dữ liệu.

    Ngoài các chỉ số độ lớn sai số, hàm tính thêm Directional Accuracy
    với mốc tham chiếu bằng 0: mục tiêu là tỷ suất biến động, nên "đoán
    đúng hướng" tương đương "đoán đúng dấu". Đây chính là cầu nối để so
    sánh nhánh B với nhánh A.

    Returns:
        dict các chỉ số, bổ sung 'directional_accuracy' và 'predictions'
    """
    predictions = model.predict(samples)
    results = metrics.calculate_all_metrics(targets, predictions)
    results['directional_accuracy'] = metrics.calculate_directional_accuracy(
        targets, predictions, reference_values=[0.0] * len(targets)
    )
    results['predictions'] = predictions
    return results


# ---------------------------------------------------------------------
# ④ Mốc đối chứng — hai cách dự đoán tầm thường cần phải vượt qua
# ---------------------------------------------------------------------
def evaluate_baselines(train_targets, targets):
    """
    Chấm điểm hai mô hình đối chứng trên cùng một tập.

        - mean_of_train : luôn dự đoán trung bình của tập huấn luyện.
                          Theo định nghĩa của R², mốc này cho R² ≈ 0.
        - zero          : luôn dự đoán 0, tức "không biến động".

    Với dữ liệu nhiễu mạnh, hai mốc tầm thường này rất khó vượt. R² âm
    nghĩa là mô hình còn tệ hơn việc đoán bừa bằng hằng số.

    Returns:
        dict { tên_đối_chứng: {'rmse', 'r_squared'} }
    """
    train_mean = sum(train_targets) / len(train_targets)
    baselines = {}
    for name, constant in (('mean_of_train', train_mean), ('zero', 0.0)):
        predictions = [constant] * len(targets)
        baselines[name] = {
            'rmse':      metrics.calculate_rmse(targets, predictions),
            'r_squared': metrics.calculate_r_squared(targets, predictions),
        }
    return baselines


# ---------------------------------------------------------------------
# ⑤ Vẽ và lưu — bộ đồ thị đầu ra của nhánh B
# ---------------------------------------------------------------------
def save_figures(config, model, dataset, evaluations, walk_forward,
                 output_dir, prefix):
    """
    Vẽ và lưu bộ đồ thị đánh giá hồi quy.

    Returns:
        list đường dẫn các file đã lưu
    """
    import matplotlib
    matplotlib.use('Agg')

    settings = config.get('evaluation', {})
    saved = []

    def target(name):
        return os.path.join(output_dir, f"{prefix}_{name}.png")

    rfPlot.plot_feature_importances(
        dataset['feature_names'], model.calculate_feature_importances(),
        top_k=settings.get('top_features', 15),
        title='Tầm quan trọng đặc trưng (MDI) — nhánh hồi quy',
        filename=target('feature_importance'),
    )
    saved.append(target('feature_importance'))

    tree_counts, errors = model.calculate_out_of_bag_error_curve()
    rfPlot.plot_out_of_bag_curve(
        tree_counts, errors,
        title='Sai số bình phương OOB theo số cây',
        y_label='MSE ngoài túi',
        filename=target('oob_curve'),
    )
    saved.append(target('oob_curve'))

    test_samples, test_targets = dataset['test']
    test_predictions = evaluations['test']['predictions']

    rfPlot.plot_predicted_versus_actual(
        test_targets, test_predictions,
        title='Tỷ suất dự đoán so với thực tế — tập test',
        filename=target('predicted_vs_actual'),
    )
    saved.append(target('predicted_vs_actual'))

    rfPlot.plot_series_comparison(
        test_targets, test_predictions,
        index_labels=[str(key) for key in dataset['test_keys']],
        title='Chuỗi tỷ suất thực tế và dự đoán — tập test',
        x_label='Thời điểm', y_label='Tỷ suất biến động',
        filename=target('series_comparison'),
    )
    saved.append(target('series_comparison'))

    rfPlot.plot_residuals(
        test_targets, test_predictions,
        title='Phân bố sai số — tập test',
        filename=target('residuals'),
    )
    saved.append(target('residuals'))

    rfPlot.plot_fold_scores(
        [f"Vòng {position}" for position in
         range(1, len(walk_forward['fold_scores']) + 1)],
        walk_forward['fold_scores'],
        title='RMSE qua từng vòng kiểm định tiến dần',
        y_label='RMSE',
        filename=target('walk_forward'),
    )
    saved.append(target('walk_forward'))
    return saved


# ---------------------------------------------------------------------
# ⑥ Mạch chính — nối ①→⑤ và in báo cáo ra màn hình
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Huấn luyện và đánh giá mô hình hồi quy tỷ suất biến động.'
    )
    parser.add_argument('--config', required=True,
                        help='Đường dẫn tới file cấu hình JSON.')
    parser.add_argument('--no-figures', action='store_true',
                        help='Bỏ qua bước vẽ và lưu đồ thị.')
    arguments = parser.parse_args()

    config = load_configuration(arguments.config)

    print('=' * 70)
    print(f"THÍ NGHIỆM: {config.get('description', '(không mô tả)')}")
    print(f"Dữ liệu   : {config['dataset']['path']} "
          f"({config['dataset'].get('label', '')})")
    print(f"Tầm nhìn  : {config['labeling'].get('horizon', 1)} bước")
    print('=' * 70)

    print('\n[1] CHUẨN BỊ DỮ LIỆU')
    dataset = experiment.prepare_dataset(config, PROJECT_ROOT)
    summary = {
        'min':  min(dataset['targets']),
        'max':  max(dataset['targets']),
        'mean': sum(dataset['targets']) / len(dataset['targets']),
    }
    print(f"Mục tiêu: min = {summary['min']:.4f} | "
          f"max = {summary['max']:.4f} | trung bình = {summary['mean']:.4f}")

    print('\n[2] TÁCH TẬP THEO THỜI GIAN')
    parts = experiment.split_prepared_dataset(config, dataset)
    dataset['test'] = parts['test']
    dataset['test_keys'] = [
        dataset['keys'][index] for index in parts['indices'][2]
    ]

    train_samples, train_targets = parts['train']
    validation_samples, validation_targets = parts['validation']
    test_samples, test_targets = parts['test']

    print('\n[3] KIỂM ĐỊNH TIẾN DẦN (WALK-FORWARD)')
    walk_forward = experiment.run_walk_forward(
        config,
        train_samples + validation_samples,
        train_targets + validation_targets,
        model_factory=lambda: create_model(config),
        score_function=metrics.calculate_rmse,
    )
    print(f"  → RMSE trung bình = {walk_forward['mean']:.6f} "
          f"± {walk_forward['standard_deviation']:.6f}")

    print('\n[4] HUẤN LUYỆN MÔ HÌNH CUỐI')
    model = create_model(config)
    model.fit(train_samples, train_targets)

    structure = model.describe()
    print(f"  Số cây: {structure['num_trees']} | "
          f"m = {structure['max_features_resolved']}/{structure['num_features']} | "
          f"độ sâu TB = {structure['average_depth']:.1f}")
    print(f"  MSE ngoài túi: {model.calculate_out_of_bag_error():.6f}")

    print('\n[5] ĐÁNH GIÁ')
    evaluations = {}
    for name, (samples, targets) in (('train', parts['train']),
                                     ('validation', parts['validation']),
                                     ('test', parts['test'])):
        evaluations[name] = evaluate_split(model, samples, targets)

    header = (f"{'Tập':<12}{'RMSE':>12}{'MAE':>12}{'R²':>10}"
              f"{'MAPE(%)':>12}{'DirAcc':>10}")
    print(header)
    print('-' * len(header))
    for name in ('train', 'validation', 'test'):
        results = evaluations[name]
        mape = results.get('mape')
        mape_text = f"{mape:>12.1f}" if mape is not None else f"{'—':>12}"
        directional = results.get('directional_accuracy')
        directional_text = (f"{directional:>10.4f}" if directional is not None
                            else f"{'—':>10}")
        print(f"{name:<12}{results['rmse']:>12.6f}{results['mae']:>12.6f}"
              f"{results['r_squared']:>10.4f}{mape_text}{directional_text}")

    print('\n[6] SO VỚI CÁC MỐC ĐỐI CHỨNG')
    for split_name, (samples, targets) in (('validate', parts['validation']),
                                           ('test', parts['test'])):
        baselines = evaluate_baselines(train_targets, targets)
        key = 'validation' if split_name == 'validate' else 'test'
        model_rmse = evaluations[key]['rmse']
        text = '  '.join(
            f"{name}: RMSE = {scores['rmse']:.6f}"
            for name, scores in sorted(baselines.items())
        )
        best_baseline = min(scores['rmse'] for scores in baselines.values())
        verdict = 'TỐT HƠN' if model_rmse < best_baseline else 'KHÔNG TỐT HƠN'
        print(f"  {split_name:<9} mô hình: RMSE = {model_rmse:.6f} | {text} "
              f"→ {verdict}")

    print('\n[7] SO SÁNH VỚI NHÁNH PHÂN LOẠI')
    print('  Directional Accuracy là chỉ số chung của hai nhánh: nhánh B suy')
    print('  chiều biến động từ DẤU của tỷ suất dự đoán, còn nhánh A học')
    print('  thẳng chiều đó. Hai con số đặt cạnh nhau cho biết việc học')
    print('  thêm độ lớn có giúp đoán hướng tốt hơn hay không.')
    for name in ('validation', 'test'):
        value = evaluations[name].get('directional_accuracy')
        if value is not None:
            print(f"    {name:<12} directional accuracy = {value:.4f}")

    if not arguments.no_figures:
        print('\n[8] LƯU ĐỒ THỊ')
        output_settings = config.get('output', {})
        output_dir = experiment.resolve_path(
            output_settings.get('figure_dir', 'data/output'), PROJECT_ROOT
        )
        os.makedirs(output_dir, exist_ok=True)
        saved = save_figures(
            config, model, dataset, evaluations, walk_forward,
            output_dir, output_settings.get('prefix', 'regression')
        )
        for path in saved:
            print(f"  {os.path.basename(path)}")

    print('\nHoàn tất.')


if __name__ == '__main__':
    main()
