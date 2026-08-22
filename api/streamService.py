# =====================================================================
# Stream Service — biến bộ máy phát lại thành luồng sự kiện SSE
# =====================================================================
# Tầng này CỐ Ý MỎNG. Toàn bộ logic thí nghiệm — dự đoán, hàng đợi chờ,
# ba đường accuracy — nằm ở pipeline/replayEngine.py và được kiểm thử
# bằng pytest mà không cần dựng máy chủ. Ở đây chỉ có ba việc: nạp gói
# mô hình, điều tiết nhịp phát, và gói sự kiện theo định dạng SSE.
#
# VÌ SAO DÙNG SSE CHỨ KHÔNG PHẢI WEBSOCKET. Luồng chỉ đi một chiều
# máy chủ → trình duyệt, không có thông điệp nào đi ngược lại. SSE làm
# đúng việc đó, tự động kết nối lại khi rớt mạng, và không cần thư viện
# nào ở phía trình duyệt.
#
# Định dạng một khung SSE:
#
#     event: step
#     data: {"index": 42, "key": "2024-05-13", ...}
#     <dòng trống>
#
# Thứ tự khai báo:
#
#   ①  Kho gói mô hình     — nạp một lần rồi dùng lại
#   ②  Dựng mô hình thích nghi
#   ③  Đóng khung SSE
#   ④  Bộ sinh luồng       — vòng lặp chính, có điều tiết nhịp
# =====================================================================

import json
import os
import time

from libraries.slidingForest import SlidingRandomForestClassifier
from pipeline import modelBundle
from pipeline.replayEngine import ReplayEngine


# ---------------------------------------------------------------------
# ① Kho gói mô hình — nạp lại mỗi lần xem là lãng phí
# ---------------------------------------------------------------------
class BundleStore:
    """
    Nạp và giữ các gói mô hình trong bộ nhớ.

    Một gói nặng khoảng 750 KB và mất chừng 10 ms để dựng lại thành đối
    tượng rừng. Giữ sẵn giúp mỗi lần bắt đầu phát lại không phải trả
    lại chi phí đó.
    """

    def __init__(self, directory):
        self.directory = directory
        self.loaded = {}

    def list_available(self):
        """
        Liệt kê các gói mô hình có trong thư mục.

        Returns:
            list of dict { 'name', 'label', 'description', 'horizon',
                           'parameters', 'metrics' }
        """
        if not os.path.isdir(self.directory):
            return []

        entries = []
        for filename in sorted(os.listdir(self.directory)):
            if not filename.endswith('.json'):
                continue
            name = os.path.splitext(filename)[0]
            try:
                bundle = self.get(name)
            except Exception:
                continue

            summary = bundle.get('training_summary', {})
            entries.append({
                'name':        name,
                'label':       summary.get('label') or summary.get('data_label') or name.upper(),
                'description': bundle.get('description', ''),
                'horizon':     bundle['recipe']['labeling'].get('horizon', 1),
                'parameters':  bundle.get('size', {}).get('parameters'),
                'num_samples': summary.get('num_samples'),
                'period':      summary.get('period', ''),
                'metrics':     bundle.get('metrics', {}),
            })
        return entries

    def get(self, name):
        """
        Lấy gói mô hình theo tên, nạp từ đĩa nếu chưa có trong bộ nhớ.

        Raises:
            FileNotFoundError nếu không có gói nào mang tên đó
        """
        if name in self.loaded:
            return self.loaded[name]

        if not _is_safe_name(name):
            raise FileNotFoundError(f"Tên mô hình không hợp lệ: '{name}'.")

        path = os.path.join(self.directory, f'{name}.json')
        bundle = modelBundle.load_bundle(path)
        self.loaded[name] = bundle
        return bundle


# ---------------------------------------------------------------------
# ② Dựng mô hình thích nghi — bọc chính rừng tĩnh vừa nạp
# ---------------------------------------------------------------------
def build_adaptive_model(bundle, settings):
    """
    Tạo bản rừng trượt từ gói mô hình đã nạp.

    Điểm mấu chốt: nó bọc CHÍNH rừng trong gói, nên hai đường trên biểu
    đồ xuất phát từ đúng cùng một tập cây. Mọi khác biệt về sau đều
    thuần tuý do việc thay máu, không lẫn nguyên nhân nào khác.

    Parameters:
        bundle   : gói mô hình đã nạp
        settings : dict tham số thay máu (nhánh 'sliding' của cấu hình)

    Returns:
        SlidingRandomForestClassifier, hoặc None nếu gói không phải bài
        toán phân loại
    """
    if bundle['model']['task'] != 'classifier':
        return None

    return SlidingRandomForestClassifier.from_forest(
        bundle['estimator'],
        trees_per_update=settings.get('trees_per_update', 5),
        window_size=settings.get('window_size', 400),
        update_every=settings.get('update_every', 40),
    )


# ---------------------------------------------------------------------
# ③ Đóng khung SSE — một sự kiện là một khối văn bản kết thúc bằng dòng trống
# ---------------------------------------------------------------------
def format_event(name, payload):
    """
    Gói một sự kiện theo đúng định dạng Server-Sent Events.

    Ngày tháng được đổi sang chuỗi vì `json` không tuần tự hoá được
    `datetime.date`.

    Returns:
        str khung SSE hoàn chỉnh
    """
    body = json.dumps(payload, ensure_ascii=False, default=str)
    return f'event: {name}\ndata: {body}\n\n'


# ---------------------------------------------------------------------
# ④ Bộ sinh luồng — vòng lặp chính, điều tiết nhịp cho người xem theo kịp
# ---------------------------------------------------------------------
def generate_stream(bundle, table, adaptive_settings=None, start_after=None,
                    speed=20, feature_window=80):
    """
    Sinh luồng sự kiện SSE cho một lượt phát lại.

    Thứ tự sự kiện:
        meta  → một lần ở đầu, mô tả phạm vi phát lại và mô hình
        step  → mỗi phiên một sự kiện
        done  → một lần ở cuối, kèm tổng kết
        error → thay cho done khi có sự cố, HOẶC khi không phát lại được
                phiên nào — im lặng trả về "0 bước" sẽ khiến người dùng
                không biết mình sai ở đâu

    Parameters:
        bundle            : gói mô hình đã nạp
        table             : dict bảng dữ liệu thô
        adaptive_settings : dict tham số thay máu; None = không chạy
                            đường thích nghi
        start_after       : mốc thời gian bắt đầu phát; None = lấy mốc
                            kết thúc huấn luyện trong gói
        speed             : số bước mỗi giây; 0 = phát nhanh nhất có thể
        feature_window    : số dòng lịch sử dùng để dựng đặc trưng

    Yields:
        str từng khung SSE
    """
    try:
        adaptive = (build_adaptive_model(bundle, adaptive_settings)
                    if adaptive_settings else None)
        engine = ReplayEngine(
            bundle, table,
            adaptive_model=adaptive,
            start_after=start_after,
            window_size=feature_window,
        )
    except Exception as error:
        yield format_event('error', {'message': str(error)})
        return

    yield format_event('meta', engine.describe())

    interval = 1.0 / speed if speed and speed > 0 else 0.0
    final = None
    count = 0

    try:
        for step in engine.run():
            final = step
            count += 1
            yield format_event('step', step)
            if interval:
                time.sleep(interval)
    except Exception as error:
        yield format_event('error', {'message': str(error)})
        return

    if count == 0:
        yield format_event('error', {
            'message': (
                'Không phát lại được phiên nào. Dữ liệu có thể quá ngắn so '
                'với cửa sổ chỉ báo dài nhất, hoặc toàn bộ đã nằm trong giai '
                'đoạn huấn luyện và bị bỏ qua.'
            ),
        })
        return

    yield format_event('done', {
        'steps':          count,
        'resolved_count': final['resolved_count'] if final else 0,
        'accuracy':       final['accuracy'] if final else {},
        'adaptation':     final['adaptation'] if final else None,
    })


# ---------------------------------------------------------------------
# Phép phụ dùng chung cho toàn module
# ---------------------------------------------------------------------
def _is_safe_name(name):
    """
    Chỉ cho phép tên gói gồm chữ, số, gạch ngang và gạch dưới.

    Ngăn việc ghép tên vào đường dẫn để đọc file ngoài thư mục models —
    dạng tấn công cơ bản nhất với mọi máy chủ nhận tham số từ URL.
    """
    return bool(name) and all(
        character.isalnum() or character in '-_' for character in name
    )
