# =====================================================================
# Test API — kiểm thử tầng web
# =====================================================================
# Tầng api/ cố ý mỏng, nên bộ test này cũng tập trung đúng phần việc của
# nó: ĐỌC ĐÚNG dữ liệu người ngoài đưa vào, và BÁO LỖI THÀNH LỜI khi dữ
# liệu hỏng. Logic thí nghiệm đã được kiểm thử ở test_replayEngine.py.
#
# Nhóm quan trọng nhất là ② — ranh giới tin cậy. Mọi thứ đi qua đó đều
# do người ngoài cung cấp, nên nó phải chặn được file quá lớn, sai định
# dạng, sai bảng mã, và tên gói mô hình có ý đồ đọc file ngoài thư mục.
#
# Thứ tự khai báo:
#
#   ①  Dữ liệu và gói mô hình mẫu
#   ②  Ranh giới tin cậy — đọc file tải lên
#   ③  Kho bảng tạm
#   ④  Đóng khung SSE và tên gói an toàn
#   ⑤  Luồng phát lại đầu-cuối
#   ⑥  Các route HTTP
# =====================================================================

import io
import json
import os

import pytest
from fastapi.testclient import TestClient

from api import main, streamService, uploadService
from test_replayEngine import make_config, make_table   # dùng lại đồ nghề sẵn có


# ---------------------------------------------------------------------
# ① Dữ liệu và gói mô hình mẫu
# ---------------------------------------------------------------------
def csv_bytes(rows=80):
    table = make_table(rows)
    lines = ['Date,Open,High,Low,Close,Volume']
    for index in range(rows):
        lines.append(','.join(str(table[name][index]) for name in
                              ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')))
    return '\n'.join(lines).encode('utf-8')


@pytest.fixture
def bundle(tmp_path):
    """Gói mô hình nhỏ, dựng ngay trong bộ nhớ rồi lưu ra thư mục tạm."""
    from libraries import rfMath
    from libraries.randomForest import RandomForestClassifier
    from pipeline import experiment, labeling, modelBundle
    from pipeline.timePreprocess import parse_date_series

    config = make_config()
    prepared = make_table()
    for name in config['dataset']['numeric_columns']:
        prepared[name] = [float(value) for value in prepared[name]]
    prepared['Date'] = parse_date_series(prepared['Date'])

    cleaned, _ = experiment.clean_table(config, prepared, verbose=False)
    feature_table, _ = experiment.build_feature_table(config, cleaned)
    names = sorted(feature_table)

    samples = rfMath.columns_to_samples([feature_table[name] for name in names])
    targets = experiment.build_targets(config, cleaned, verbose=False)
    samples, targets, _ = labeling.align_features_and_targets(samples, targets)

    model = RandomForestClassifier(**config['model']).fit(
        samples[:len(samples) // 2], targets[:len(targets) // 2])

    payload = modelBundle.build_bundle(model, config, names)
    payload['threshold'] = 0.5
    payload['training_summary'] = {'data_label': 'TEST', 'num_samples': 40}

    path = os.path.join(str(tmp_path), 'test.json')
    modelBundle.save_bundle(payload, path)
    return modelBundle.load_bundle(path)


@pytest.fixture
def client(tmp_path, bundle):
    """Ứng dụng với kho mô hình trỏ vào thư mục tạm của test."""
    from pipeline import modelBundle

    store = streamService.BundleStore(str(tmp_path))
    modelBundle.save_bundle(
        {key: value for key, value in bundle.items() if key != 'estimator'},
        os.path.join(str(tmp_path), 'test.json'))

    original = main.bundle_store
    main.bundle_store = store
    yield TestClient(main.application)
    main.bundle_store = original


# ---------------------------------------------------------------------
# ② Ranh giới tin cậy — mọi lỗi phải nói được thành lời
# ---------------------------------------------------------------------
def test_reads_csv():
    table = uploadService.parse_upload(csv_bytes(60), 'x.csv')

    assert sorted(table) == ['Close', 'Date', 'High', 'Low', 'Open', 'Volume']
    assert len(table['Close']) == 60


def test_reads_csv_with_semicolon_delimiter():
    """File xuất từ Excel ở một số vùng dùng dấu chấm phẩy."""
    content = 'Date;Close\n2024-01-01;10\n2024-01-02;11'.encode('utf-8')
    table = uploadService.parse_upload(content, 'x.csv')

    assert table['Close'] == ['10', '11']


def test_reads_json_in_both_layouts():
    by_rows = json.dumps([{'Date': '2024-01-01', 'Close': 10},
                          {'Date': '2024-01-02', 'Close': 11}]).encode()
    by_columns = json.dumps({'Date': ['2024-01-01', '2024-01-02'],
                             'Close': [10, 11]}).encode()

    assert uploadService.parse_upload(by_rows, 'x.json')['Close'] == [10, 11]
    assert uploadService.parse_upload(by_columns, 'x.json')['Close'] == [10, 11]


def test_reads_excel(tmp_path):
    openpyxl = pytest.importorskip('openpyxl')

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(['Date', 'Close'])
    sheet.append(['2024-01-01', 10])
    sheet.append(['2024-01-02', 11])

    buffer = io.BytesIO()
    workbook.save(buffer)

    table = uploadService.parse_upload(buffer.getvalue(), 'x.xlsx')
    assert table['Close'] == [10, 11]


def test_rejects_empty_file():
    with pytest.raises(ValueError, match='rỗng'):
        uploadService.parse_upload(b'', 'x.csv')


def test_rejects_oversized_file():
    oversized = b'x' * (uploadService.MAX_UPLOAD_BYTES + 1)
    with pytest.raises(ValueError, match='vượt giới hạn'):
        uploadService.parse_upload(oversized, 'x.csv')


def test_rejects_unknown_extension():
    with pytest.raises(ValueError, match='Không hỗ trợ đuôi file'):
        uploadService.parse_upload(b'gi cung duoc', 'x.txt')


def test_rejects_header_only_csv():
    with pytest.raises(ValueError, match='không có dòng dữ liệu'):
        uploadService.parse_upload(b'Date,Close', 'x.csv')


def test_rejects_malformed_json():
    with pytest.raises(ValueError, match='sai cú pháp'):
        uploadService.parse_upload(b'{khong phai json}', 'x.json')


def test_rejects_json_columns_of_unequal_length():
    content = json.dumps({'Date': ['a', 'b'], 'Close': [1]}).encode()
    with pytest.raises(ValueError, match='cùng độ dài'):
        uploadService.parse_upload(content, 'x.json')


def test_handles_non_utf8_encoding():
    """File xuất từ Excel bản Windows thường dùng cp1252."""
    content = 'Date,Ghi chú\n2024-01-01,café'.encode('cp1252')
    table = uploadService.parse_upload(content, 'x.csv')

    assert len(table) == 2


# ---------------------------------------------------------------------
# ③ Kho bảng tạm — giữ trong bộ nhớ, tự đẩy bảng cũ khi đầy
# ---------------------------------------------------------------------
def test_table_store_round_trip():
    store = uploadService.TableStore()
    token = store.add({'Close': [1, 2]}, 'x.csv')

    assert store.get(token)['table']['Close'] == [1, 2]
    assert store.get(token)['label'] == 'x.csv'


def test_table_store_evicts_oldest():
    store = uploadService.TableStore(capacity=2)
    first = store.add({'Close': [1]}, 'a')
    store.add({'Close': [2]}, 'b')
    store.add({'Close': [3]}, 'c')

    with pytest.raises(KeyError):
        store.get(first)


def test_unknown_token_message_is_actionable():
    with pytest.raises(KeyError, match='tải file lên lại'):
        uploadService.TableStore().get('không-có')


# ---------------------------------------------------------------------
# ④ Đóng khung SSE và tên gói an toàn
# ---------------------------------------------------------------------
def test_event_frame_follows_sse_format():
    frame = streamService.format_event('step', {'index': 3})

    assert frame.startswith('event: step\ndata: ')
    assert frame.endswith('\n\n')
    assert json.loads(frame.split('data: ')[1].strip()) == {'index': 3}


def test_event_frame_serialises_dates():
    import datetime

    frame = streamService.format_event('step', {'key': datetime.date(2024, 1, 2)})
    assert '2024-01-02' in frame


def test_model_name_must_be_safe():
    """
    Chặn việc ghép tên vào đường dẫn để đọc file ngoài thư mục models —
    dạng tấn công cơ bản nhất với máy chủ nhận tham số từ URL.
    """
    store = streamService.BundleStore('models')

    for name in ('../secret', 'a/b', '..', '', 'x\\y'):
        with pytest.raises(FileNotFoundError):
            store.get(name)


# ---------------------------------------------------------------------
# ⑤ Luồng phát lại đầu-cuối — meta trước, step ở giữa, done ở cuối
# ---------------------------------------------------------------------
def parse_stream(chunks):
    """Tách luồng SSE thành danh sách (tên sự kiện, payload)."""
    events, name, buffer = [], None, []
    for line in ''.join(chunks).split('\n'):
        if line.startswith('event: '):
            name = line[7:]
        elif line.startswith('data: '):
            buffer.append(line[6:])
        elif line == '' and name:
            events.append((name, json.loads(''.join(buffer))))
            name, buffer = None, []
    return events


def test_stream_emits_meta_steps_and_done(bundle):
    events = parse_stream(streamService.generate_stream(
        bundle, make_table(), speed=0))

    assert events[0][0] == 'meta'
    assert events[-1][0] == 'done'
    assert all(name == 'step' for name, _ in events[1:-1])
    assert events[-1][1]['steps'] == len(events) - 2


def test_stream_reports_error_instead_of_crashing(bundle):
    """Dữ liệu hỏng phải thành sự kiện error, không phải ngoại lệ thô."""
    events = parse_stream(streamService.generate_stream(
        bundle, {'Date': [], 'Open': [], 'High': [], 'Low': [],
                 'Close': [], 'Volume': []}, speed=0))

    assert events[-1][0] == 'error'
    assert events[-1][1]['message']


def test_stream_with_adaptive_line(bundle):
    events = parse_stream(streamService.generate_stream(
        bundle, make_table(), speed=0,
        adaptive_settings={'trees_per_update': 2, 'window_size': 30,
                           'update_every': 10}))

    assert events[0][1]['has_adaptive'] is True
    assert events[-1][1]['accuracy']['adaptive'] is not None


def test_stream_without_adaptive_line(bundle):
    events = parse_stream(streamService.generate_stream(
        bundle, make_table(), speed=0))

    assert events[0][1]['has_adaptive'] is False
    assert events[-1][1]['accuracy']['adaptive'] is None


# ---------------------------------------------------------------------
# ⑥ Các route HTTP
# ---------------------------------------------------------------------
def test_lists_models(client):
    response = client.get('/api/models')

    assert response.status_code == 200
    assert response.json()['models'][0]['name'] == 'test'


def test_lists_datasets(client):
    response = client.get('/api/datasets')

    assert response.status_code == 200
    assert isinstance(response.json()['datasets'], list)


def test_upload_route_accepts_csv(client):
    response = client.post(
        '/api/upload',
        files={'file': ('x.csv', csv_bytes(60), 'text/csv')})

    assert response.status_code == 200
    assert response.json()['rows'] == 60
    assert response.json()['token'].startswith('upload-')


def test_upload_route_reports_bad_file(client):
    response = client.post(
        '/api/upload', files={'file': ('x.txt', b'gi do', 'text/plain')})

    assert response.status_code == 400
    assert 'Không hỗ trợ đuôi file' in response.json()['detail']


def test_stream_route_rejects_unknown_model(client):
    response = client.get('/api/stream?model=khong-co-that')

    assert response.status_code == 404


def test_stream_route_rejects_unknown_upload_token(client):
    response = client.get('/api/stream?model=test&upload=upload-999')

    assert response.status_code == 404


def test_stream_route_streams_uploaded_data(client):
    upload = client.post(
        '/api/upload',
        files={'file': ('x.csv', csv_bytes(120), 'text/csv')}).json()

    response = client.get(
        f"/api/stream?model=test&upload={upload['token']}&speed=0&adaptive=false")

    assert response.status_code == 200
    events = parse_stream([response.text])
    assert events[0][0] == 'meta'
    assert events[-1][0] in ('done', 'error')


def test_index_page_is_served(client):
    response = client.get('/')

    assert response.status_code == 200
    assert 'Dự đoán xu hướng giá cổ phiếu' in response.text
