# =====================================================================
# Main — ứng dụng web phát lại và dự đoán trực quan
# =====================================================================
# Chạy:
#
#   python -m uvicorn api.main:application --reload --port 8000
#
# rồi mở http://127.0.0.1:8000
#
# NGUYÊN TẮC PHÂN TẦNG. Thư mục api/ nằm NGOÀI src/ và chỉ import vào
# một chiều. Lõi học máy trong src/ không biết gì về web, không phụ
# thuộc FastAPI, và chạy độc lập được bằng dòng lệnh hay notebook như
# trước. Gỡ bỏ toàn bộ api/ thì phần còn lại vẫn nguyên vẹn.
#
# Tầng này cũng cố ý mỏng: mỗi route chỉ đọc tham số, gọi một hàm ở
# streamService hoặc uploadService, rồi trả kết quả. Không có logic
# thí nghiệm nào ở đây.
#
# Thứ tự khai báo:
#
#   ①  Cấu hình và tài nguyên dùng chung
#   ②  Danh sách mô hình
#   ③  Tải dữ liệu lên
#   ④  Luồng phát lại       — route chính
#   ⑤  Trang tĩnh
# =====================================================================

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api import streamService, uploadService

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'input')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
ADAPTIVE_CONFIG = os.path.join(PROJECT_ROOT, 'config', 'adaptive.json')


# ---------------------------------------------------------------------
# ① Cấu hình và tài nguyên dùng chung — dựng một lần lúc khởi động
# ---------------------------------------------------------------------
def load_adaptive_settings():
    """
    Đọc tham số thay máu từ config/adaptive.json.

    Thiếu file thì dùng giá trị mặc định thay vì dừng máy chủ — phần
    thích nghi chỉ là một đường phụ trên biểu đồ, không đáng để làm
    hỏng cả ứng dụng.

    Returns:
        dict { 'sliding': {...}, 'replay': {...} }
    """
    fallback = {
        'sliding': {'trees_per_update': 5, 'window_size': 400,
                    'update_every': 40},
        'replay':  {'feature_window': 80},
    }
    if not os.path.exists(ADAPTIVE_CONFIG):
        return fallback

    with open(ADAPTIVE_CONFIG, encoding='utf-8') as handle:
        settings = json.load(handle)
    return {
        'sliding': settings.get('sliding', fallback['sliding']),
        'replay':  settings.get('replay', fallback['replay']),
    }


application = FastAPI(
    title='Dự đoán xu hướng giá cổ phiếu — Random Forest',
    description='Phát lại chuỗi thời gian và chấm điểm dự đoán theo thời gian thực.',
    version='1.0',
)

bundle_store = streamService.BundleStore(MODELS_DIR)
table_store = uploadService.TableStore()
adaptive_settings = load_adaptive_settings()


# ---------------------------------------------------------------------
# ② Danh sách mô hình — để giao diện dựng ô chọn mã
# ---------------------------------------------------------------------
@application.get('/api/models')
def list_models():
    """
    Liệt kê các gói mô hình có sẵn kèm chỉ số đạt được lúc huấn luyện.

    Chỉ số được trả về cùng danh sách để giao diện hiển thị ngay bên
    cạnh tên mã — người xem biết mình đang chọn mô hình mạnh hay yếu
    trước khi bấm chạy.
    """
    entries = bundle_store.list_available()
    if not entries:
        raise HTTPException(
            status_code=404,
            detail=('Chưa có gói mô hình nào trong models/. Hãy chạy '
                    'mainClassification.py với cờ --save-model trước.'),
        )
    return {'models': entries}


@application.get('/api/datasets')
def list_datasets():
    """
    Liệt kê các file dữ liệu có sẵn trong data/input/.

    Dùng cho ô chọn nhanh, để người xem không phải tự tải file lên mới
    thử được.
    """
    if not os.path.isdir(DATA_DIR):
        return {'datasets': []}

    names = sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(DATA_DIR)
        if name.lower().endswith(('.csv', '.xlsx'))
    )
    return {'datasets': names}


# ---------------------------------------------------------------------
# ③ Tải dữ liệu lên — ranh giới tin cậy, mọi lỗi phải nói được thành lời
# ---------------------------------------------------------------------
@application.post('/api/upload')
async def upload_dataset(file: UploadFile = File(...)):
    """
    Nhận file người dùng tải lên, phân tích rồi giữ tạm trong bộ nhớ.

    Trả về mã tra cứu để dùng cho lượt phát lại kế tiếp. Dữ liệu KHÔNG
    được ghi ra đĩa và tự bị đẩy khỏi bộ nhớ khi có file mới hơn.
    """
    content = await file.read()
    try:
        table = uploadService.parse_upload(content, file.filename or 'upload')
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    token = table_store.add(table, file.filename or 'upload')
    num_rows = max((len(values) for values in table.values()), default=0)

    return {
        'token':    token,
        'label':    os.path.splitext(file.filename or 'upload')[0],
        'rows':     num_rows,
        'columns':  sorted(table),
        'message':  f'Đã nhận {num_rows:,} dòng.',
    }


# ---------------------------------------------------------------------
# ④ Luồng phát lại — route chính của cả ứng dụng
# ---------------------------------------------------------------------
@application.get('/api/stream')
def stream_replay(
    model: str = Query(..., description='Tên gói mô hình, ví dụ "vcb".'),
    dataset: str = Query(None, description='Tên file trong data/input/.'),
    upload: str = Query(None, description='Mã tra cứu của file đã tải lên.'),
    speed: float = Query(20, ge=0, le=500, description='Số bước mỗi giây.'),
    adaptive: bool = Query(True, description='Có chạy đường thích nghi không.'),
    start_after: str = Query(None, description='Chỉ phát từ sau mốc này.'),
):
    """
    Mở luồng SSE phát lại một chuỗi thời gian.

    Nguồn dữ liệu lấy theo thứ tự ưu tiên: `upload` trước, rồi `dataset`,
    và cuối cùng là chính file mà mô hình được huấn luyện trên đó.

    Hàm khai báo dạng đồng bộ có chủ ý: FastAPI sẽ chạy nó trong luồng
    riêng, nên vòng lặp phát lại không chặn vòng lặp sự kiện chính.
    """
    try:
        bundle = bundle_store.get(model)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    table = _resolve_table(bundle, dataset, upload)
    settings = adaptive_settings['sliding'] if adaptive else None

    stream = streamService.generate_stream(
        bundle, table,
        adaptive_settings=settings,
        start_after=start_after,
        speed=speed,
        feature_window=adaptive_settings['replay'].get('feature_window', 80),
    )
    return StreamingResponse(
        stream,
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',      # tắt đệm khi chạy sau nginx
        },
    )


def _resolve_table(bundle, dataset, upload):
    """
    Chọn nguồn dữ liệu cho lượt phát lại.

    Trả về BẢN SAO nông, vì bộ máy phát lại có sửa đổi bảng khi làm
    sạch — nếu đưa thẳng bảng trong kho thì lượt xem sau sẽ nhận dữ
    liệu đã bị ép kiểu dở dang.
    """
    if upload:
        try:
            return dict(table_store.get(upload)['table'])
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    name = dataset or bundle.get('training_summary', {}).get('data_label', '')
    if not name:
        raise HTTPException(
            status_code=400,
            detail='Không xác định được nguồn dữ liệu — hãy chọn dataset '
                   'hoặc tải file lên.',
        )

    for extension in ('.csv', '.xlsx'):
        path = os.path.join(DATA_DIR, f'{name}{extension}')
        if os.path.exists(path):
            with open(path, 'rb') as handle:
                return uploadService.parse_upload(handle.read(),
                                                  f'{name}{extension}')

    raise HTTPException(
        status_code=404,
        detail=f"Không tìm thấy dữ liệu '{name}' trong data/input/.",
    )


# ---------------------------------------------------------------------
# ⑤ Trang tĩnh — giao diện không cần bước build, không phụ thuộc mạng
# ---------------------------------------------------------------------
@application.get('/')
def serve_index():
    """Trả về trang chính."""
    path = os.path.join(STATIC_DIR, 'index.html')
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='Chưa có giao diện.')
    return FileResponse(path)


if os.path.isdir(STATIC_DIR):
    application.mount('/static', StaticFiles(directory=STATIC_DIR),
                      name='static')
