# =====================================================================
# Main Predict — dự đoán bằng gói mô hình ĐÃ HUẤN LUYỆN, không train lại
# =====================================================================
# Đây là script biến dự án từ một THUẬT TOÁN thành một MÔ HÌNH: nó nạp
# gói mô hình đã lưu rồi dự đoán ngay, không cần dữ liệu huấn luyện và
# không tốn thời gian dựng lại rừng.
#
#   python src/mainPredict.py --model models/vnm.json \
#                             --data data/input/VNM.csv --rows 10
#
# Gói mô hình mang theo CÔNG THỨC ĐẶC TRƯNG của chính nó, nên script
# này không cần tới file cấu hình. Nếu dữ liệu đưa vào không dựng ra
# đúng bộ đặc trưng mà mô hình chờ đợi, chương trình DỪNG với thông báo
# rõ ràng thay vì âm thầm cho ra số sai.
#
# HAI ĐIỀU CẦN NHỚ KHI ĐỌC KẾT QUẢ:
#
#   1. Dòng mới nhất là DỰ BÁO THẬT — nhãn của nó chưa tồn tại vì tương
#      lai chưa xảy ra. Các dòng phía trên đã có thể đối chiếu.
#   2. Dự đoán ứng với tầm nhìn `horizon` phiên ghi trong gói, không
#      phải phiên kế tiếp, trừ khi horizon = 1.
#
# Thứ tự khai báo bám đúng trình tự chạy:
#
#   ①  Nạp bảng dữ liệu thô   — chỉ đọc file, chưa xử lý gì
#   ②  In kết quả dự đoán     — tách riêng phần trình bày
#   ③  Mạch chính             — nạp gói, kiểm tra, dự đoán, in
# =====================================================================

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from pipeline import experiment, modelBundle
from utilities import dataLoader


# ---------------------------------------------------------------------
# ① Nạp bảng thô — mọi việc làm sạch do gói mô hình quyết định, không phải ở đây
# ---------------------------------------------------------------------
def load_table(path, verbose=True):
    """
    Đọc file dữ liệu thô theo phần mở rộng.

    Cố ý KHÔNG làm sạch hay ép kiểu ở đây: các bước đó phải chạy đúng
    như lúc huấn luyện, mà công thức của chúng nằm trong gói mô hình.

    Parameters:
        path : đường dẫn tới file .csv hoặc .xlsx

    Returns:
        dict bảng { tên_cột: list giá trị }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    extension = os.path.splitext(path)[1].lower()
    if extension == '.csv':
        _, table = dataLoader.load_csv_data(path, verbose=verbose)
    else:
        _, table = dataLoader.load_excel_data(path, verbose=verbose)
    return table


# ---------------------------------------------------------------------
# ② In kết quả — đánh dấu rõ dòng nào là dự báo thật, dòng nào đã kiểm chứng được
# ---------------------------------------------------------------------
def print_predictions(results, bundle):
    """
    In bảng dự đoán ra màn hình.

    Dòng cuối cùng được đánh dấu riêng vì đó là dự báo cho tương lai
    chưa xảy ra — thứ duy nhất không thể đối chiếu ngay.
    """
    labeling_settings = bundle['recipe']['labeling']
    horizon = labeling_settings.get('horizon', 1)
    is_classifier = bundle['model']['task'] == 'classifier'
    positive_label = labeling_settings.get('positive_label', 1)

    if is_classifier:
        print(f"{'Mốc thời gian':<16}{'Dự đoán':>12}{'Xác suất tăng':>16}"
              f"{'Mức tin cậy':>14}")
    else:
        print(f"{'Mốc thời gian':<16}{'Dự đoán':>14}")
    print('-' * (58 if is_classifier else 30))

    for position, item in enumerate(results):
        is_last = position == len(results) - 1
        marker = '  ← dự báo' if is_last else ''

        if is_classifier:
            direction = 'TĂNG' if item['prediction'] == positive_label else 'GIẢM'
            distance = abs(item['score'] - 0.5)
            confidence = ('cao' if distance > 0.15
                          else 'trung bình' if distance > 0.05 else 'thấp')
            print(f"{str(item['key']):<16}{direction:>12}"
                  f"{item['score']:>16.4f}{confidence:>14}{marker}")
        else:
            print(f"{str(item['key']):<16}{item['prediction']:>14.6f}{marker}")

    print(f"\nDự đoán ứng với biến động sau {horizon} phiên kể từ mốc tương ứng.")


# ---------------------------------------------------------------------
# ③ Mạch chính — nạp gói, dự đoán, in; không có bước huấn luyện nào
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Dự đoán bằng gói mô hình đã huấn luyện sẵn.'
    )
    parser.add_argument('--model', required=True,
                        help='Đường dẫn tới gói mô hình .json.')
    parser.add_argument('--data', required=True,
                        help='Đường dẫn tới dữ liệu cùng định dạng lúc huấn luyện.')
    parser.add_argument('--rows', type=int, default=10,
                        help='Số mốc thời gian gần nhất cần in (mặc định 10).')
    parser.add_argument('--threshold', type=float,
                        help='Ngưỡng quyết định. Bỏ trống thì dùng ngưỡng '
                             'lưu trong gói mô hình.')
    parser.add_argument('--describe', action='store_true',
                        help='In thông tin gói mô hình rồi thoát.')
    arguments = parser.parse_args()

    model_path = experiment.resolve_path(arguments.model, PROJECT_ROOT)
    bundle = modelBundle.load_bundle(model_path)

    print('=' * 60)
    print('GÓI MÔ HÌNH')
    print('=' * 60)
    print(modelBundle.describe_bundle(bundle))

    if arguments.describe:
        return

    print('\n' + '=' * 60)
    print('DỰ ĐOÁN')
    print('=' * 60)

    table = load_table(experiment.resolve_path(arguments.data, PROJECT_ROOT))
    results = modelBundle.predict_with_bundle(
        bundle, table,
        num_rows=arguments.rows,
        threshold=arguments.threshold,
    )

    print()
    print_predictions(results, bundle)

    accuracy = bundle.get('metrics', {}).get('test_accuracy')
    if accuracy is not None:
        print(f"\nNhắc lại để đọc kết quả cho đúng: mô hình này đạt "
              f"accuracy {accuracy:.4f} trên tập test lúc huấn luyện — "
              f"mức chưa phân biệt được với ngẫu nhiên. Dùng để học tập, "
              f"không dùng để ra quyết định đầu tư.")


if __name__ == '__main__':
    main()
