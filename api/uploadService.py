# =====================================================================
# Upload Service — nhận file người dùng tải lên và đưa về dạng bảng
# =====================================================================
# Tầng này là RANH GIỚI TIN CẬY của hệ thống: mọi thứ đi qua đây đều do
# người ngoài cung cấp và không được tin. Vì vậy nó kiểm tra trước rồi
# mới chuyển tiếp, và luôn báo lỗi bằng câu người dùng hiểu được thay
# vì để ngoại lệ thô nổi lên.
#
# Hỗ trợ ba định dạng, tất cả đều phải cho ra CÙNG một cấu trúc:
#   dict { tên_cột: list giá trị }  — đúng dạng dataLoader trả về
#
# Bảng trả về vẫn ở dạng THÔ (chuỗi chưa ép kiểu). Việc làm sạch và
# dựng đặc trưng thuộc về gói mô hình, vì chỉ nó mới biết công thức.
#
# Thứ tự khai báo:
#
#   ①  Giới hạn an toàn      — chặn file quá lớn trước khi phân tích
#   ②  Đọc CSV
#   ③  Đọc Excel
#   ④  Đọc JSON              — chấp nhận hai dạng bố cục
#   ⑤  Điều phối theo đuôi file
#   ⑥  Kho bảng tạm          — giữ trong bộ nhớ, không ghi đĩa
# =====================================================================

import csv
import io
import json
import os

# Giới hạn cố ý đặt thấp: một chuỗi 20 năm dữ liệu ngày chỉ khoảng
# 300 KB, nên vượt 8 MB gần như chắc chắn là nhầm file.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_ROWS = 50_000
MAX_STORED_TABLES = 16


# ---------------------------------------------------------------------
# ① Giới hạn an toàn — kiểm tra kích thước trước khi phân tích nội dung
# ---------------------------------------------------------------------
def assert_within_limits(content, filename):
    """
    Chặn file quá lớn trước khi tốn công phân tích.

    Parameters:
        content  : bytes nội dung file
        filename : tên file, dùng trong thông báo lỗi

    Raises:
        ValueError nếu vượt giới hạn
    """
    if not content:
        raise ValueError(f"File '{filename}' rỗng.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File '{filename}' nặng {len(content) / 1024 / 1024:.1f} MB, "
            f"vượt giới hạn {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
        )


# ---------------------------------------------------------------------
# ② Đọc CSV — định dạng phổ biến nhất, tự dò dấu phân cách
# ---------------------------------------------------------------------
def read_csv(content):
    """
    Phân tích nội dung CSV thành bảng.

    Tự dò dấu phân cách giữa dấu phẩy, chấm phẩy và tab — ba lựa chọn
    hay gặp khi file được xuất từ Excel ở các vùng ngôn ngữ khác nhau.

    Returns:
        dict { tên_cột: list giá trị dạng chuỗi }
    """
    text = _decode(content)
    sample = text[:4096]

    delimiter = ','
    for candidate in (',', ';', '\t'):
        if sample.count(candidate) > sample.count(delimiter):
            delimiter = candidate

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise ValueError('File CSV không có dòng dữ liệu nào sau dòng tiêu đề.')

    headers = [cell.strip() for cell in rows[0]]
    return _rows_to_table(headers, rows[1:])


# ---------------------------------------------------------------------
# ③ Đọc Excel — cùng yêu cầu tên cột, chỉ khác cách lấy dữ liệu ra
# ---------------------------------------------------------------------
def read_excel(content):
    """
    Phân tích nội dung .xlsx thành bảng, lấy sheet đầu tiên.

    Returns:
        dict { tên_cột: list giá trị }
    """
    try:
        import openpyxl
    except ImportError as error:
        raise ValueError(
            'Chưa cài openpyxl nên không đọc được file Excel. '
            'Hãy dùng định dạng CSV hoặc cài thêm gói này.'
        ) from error

    workbook = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
    rows = [
        list(row) for row in workbook.worksheets[0].iter_rows(values_only=True)
        if any(cell is not None for cell in row)
    ]
    if len(rows) < 2:
        raise ValueError('File Excel không có dòng dữ liệu nào sau dòng tiêu đề.')

    headers = [str(cell).strip() if cell is not None else '' for cell in rows[0]]
    return _rows_to_table(headers, rows[1:])


# ---------------------------------------------------------------------
# ④ Đọc JSON — chấp nhận cả bố cục theo dòng lẫn theo cột
# ---------------------------------------------------------------------
def read_json(content):
    """
    Phân tích nội dung JSON thành bảng.

    Chấp nhận hai bố cục, vì cả hai đều hay gặp:

        [{"Date": "...", "Close": 100}, ...]     — mảng các dòng
        {"Date": ["..."], "Close": [100, ...]}   — dict các cột

    Returns:
        dict { tên_cột: list giá trị }
    """
    try:
        payload = json.loads(_decode(content))
    except json.JSONDecodeError as error:
        raise ValueError(f'File JSON sai cú pháp: {error}') from error

    if isinstance(payload, dict):
        if not payload:
            raise ValueError('File JSON rỗng.')
        lengths = {len(values) for values in payload.values()
                   if isinstance(values, list)}
        if len(lengths) != 1:
            raise ValueError(
                'Các cột trong file JSON phải là mảng có cùng độ dài.'
            )
        return {name: list(values) for name, values in payload.items()}

    if isinstance(payload, list):
        if not payload:
            raise ValueError('File JSON không có dòng dữ liệu nào.')
        if not isinstance(payload[0], dict):
            raise ValueError(
                'Mảng JSON phải chứa các đối tượng dạng {"Date": ..., ...}.'
            )
        headers = list(payload[0])
        return _rows_to_table(
            headers, [[row.get(name) for name in headers] for row in payload]
        )

    raise ValueError('Cấu trúc JSON không nhận diện được.')


# ---------------------------------------------------------------------
# ⑤ Điều phối — chọn bộ đọc theo đuôi file
# ---------------------------------------------------------------------
READERS = {
    '.csv':  read_csv,
    '.json': read_json,
    '.xlsx': read_excel,
    '.xlsm': read_excel,
}


def parse_upload(content, filename):
    """
    Đọc file tải lên thành bảng, chọn bộ đọc theo phần mở rộng.

    Parameters:
        content  : bytes nội dung file
        filename : tên file gốc

    Returns:
        dict { tên_cột: list giá trị }

    Raises:
        ValueError với thông báo tiếng Việt cho mọi trường hợp hỏng
    """
    assert_within_limits(content, filename)

    extension = os.path.splitext(filename)[1].lower()
    if extension not in READERS:
        raise ValueError(
            f"Không hỗ trợ đuôi file '{extension}'. "
            f"Hãy dùng một trong: {', '.join(sorted(READERS))}."
        )

    table = READERS[extension](content)
    num_rows = max((len(values) for values in table.values()), default=0)
    if num_rows > MAX_ROWS:
        raise ValueError(
            f'File có {num_rows:,} dòng, vượt giới hạn {MAX_ROWS:,} dòng.'
        )
    return table


# ---------------------------------------------------------------------
# ⑥ Kho bảng tạm — giữ trong bộ nhớ, không ghi ra đĩa
# ---------------------------------------------------------------------
class TableStore:
    """
    Giữ tạm các bảng đã tải lên để luồng phát lại dùng lại được.

    Cố ý KHÔNG ghi ra đĩa: dữ liệu người dùng tải lên chỉ phục vụ đúng
    phiên xem đó, không có lý do gì để tồn tại lâu hơn. Kho cũng bị chặn
    số lượng và tự đẩy bảng cũ nhất ra khi đầy, nên bộ nhớ không phình.
    """

    def __init__(self, capacity=MAX_STORED_TABLES):
        self.capacity = capacity
        self.tables = {}
        self.order = []
        self.counter = 0

    def add(self, table, label):
        """
        Lưu một bảng và trả về mã tra cứu.

        Returns:
            token : chuỗi mã tra cứu
        """
        self.counter += 1
        token = f'upload-{self.counter}'

        self.tables[token] = {'table': table, 'label': label}
        self.order.append(token)

        while len(self.order) > self.capacity:
            self.tables.pop(self.order.pop(0), None)
        return token

    def get(self, token):
        """
        Lấy lại bảng theo mã.

        Raises:
            KeyError nếu mã không còn trong kho
        """
        if token not in self.tables:
            raise KeyError(
                f"Không tìm thấy dữ liệu '{token}'. Có thể phiên đã hết hạn "
                f"hoặc máy chủ đã khởi động lại — hãy tải file lên lại."
            )
        return self.tables[token]


# ---------------------------------------------------------------------
# Phép phụ dùng chung cho toàn module
# ---------------------------------------------------------------------
def _decode(content):
    """Giải mã bytes, thử lần lượt các bảng mã hay gặp."""
    for encoding in ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1'):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError('Không giải mã được nội dung file — kiểm tra lại bảng mã.')


def _rows_to_table(headers, rows):
    """Đổi danh sách dòng thành dict các cột, bỏ qua ô thừa."""
    if not headers or all(not name for name in headers):
        raise ValueError('Không đọc được dòng tiêu đề của file.')

    table = {name: [] for name in headers}
    for row in rows:
        for position, name in enumerate(headers):
            value = row[position] if position < len(row) else None
            table[name].append(value)
    return table
