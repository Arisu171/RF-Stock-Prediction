# =====================================================================
# Data Loader — nạp dữ liệu từ file vào cấu trúc list thuần
# =====================================================================
# Tầng vào/ra duy nhất của project. Các module thuật toán chỉ nhận
# list số, không tự đọc file.
#
# Thứ tự khai báo đi từ đọc thô tới dữ liệu sẵn sàng cho mô hình:
#
#   ① Đọc Excel      — hai đầu vào thô, khác nhau ở định dạng file
#   ② Đọc CSV        — cùng vai trò với ①, trả về cùng cấu trúc dict
#   ③ Lấy một cột    — lối tắt khi chỉ cần đúng một cột
#   ④ Trích nhiều cột — làm sạch theo DÒNG để các cột luôn khớp chỉ số
#   ⑤ Gộp ①/② với ④  — hàm dùng nhiều nhất, nhận file trả về cột số
#
# Điểm mấu chốt nằm ở ④: làm sạch theo dòng chứ không theo cột. Nếu mỗi
# cột tự loại phần thiếu của riêng nó thì các cột sẽ lệch chỉ số và mọi
# mẫu phía sau đều sai — một lỗi âm thầm, rất khó phát hiện về sau.
# =====================================================================

import csv
import os

import openpyxl


# ---------------------------------------------------------------------
# ① Đọc Excel — đầu vào thô thứ nhất, trả về dict of lists
# ---------------------------------------------------------------------
def load_excel_data(file_path, sheet_name=0, header_row=1, verbose=True):
    """
    Đọc file Excel và trả về dữ liệu dạng dict of lists.

    Parameters:
        file_path  : đường dẫn tới file .xlsx
        sheet_name : tên sheet hoặc index (mặc định sheet đầu tiên)
        header_row : số thứ tự dòng tiêu đề (mặc định dòng 1)
        verbose    : in thông tin file đã đọc

    Returns:
        headers : list tên cột
        data    : dict { tên_cột: [giá trị theo dòng] }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    workbook = openpyxl.load_workbook(file_path, data_only=True)
    sheet = (workbook.worksheets[sheet_name] if isinstance(sheet_name, int)
             else workbook[sheet_name])

    rows = list(sheet.iter_rows(values_only=True))
    headers = [str(header) for header in rows[header_row - 1]]

    data = {header: [] for header in headers}
    for row in rows[header_row:]:
        for column_index, value in enumerate(row):
            if column_index < len(headers):
                data[headers[column_index]].append(value)

    if verbose:
        print(f"Đã đọc file: {os.path.basename(file_path)} | "
              f"Sheet: '{sheet.title}' | Số dòng dữ liệu: {len(rows) - header_row}")
        print(f"Các cột: {headers}")
    return headers, data


# ---------------------------------------------------------------------
# ② Đọc CSV — đầu vào thô thứ hai, cùng cấu trúc trả về với ①
# ---------------------------------------------------------------------
def load_csv_data(file_path, delimiter=',', verbose=True):
    """
    Đọc file CSV và trả về dữ liệu dạng dict of lists (giá trị giữ nguyên
    dạng chuỗi, chuỗi rỗng được quy về None).

    Returns:
        headers : list tên cột
        data    : dict { tên_cột: [giá trị theo dòng] }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with open(file_path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        rows = list(reader)

    headers = rows[0]
    data = {header: [] for header in headers}
    for row in rows[1:]:
        for column_index, value in enumerate(row):
            if column_index < len(headers):
                data[headers[column_index]].append(value if value != '' else None)

    if verbose:
        print(f"Đã đọc file: {os.path.basename(file_path)} | "
              f"Số dòng dữ liệu: {len(rows) - 1}")
        print(f"Các cột: {headers}")
    return headers, data


# ---------------------------------------------------------------------
# ③ Lấy một cột — lối tắt; KHÔNG dùng khi cần nhiều cột khớp dòng
# ---------------------------------------------------------------------
def get_column(data, column_name):
    """
    Lấy một cột dưới dạng list số thực, bỏ qua ô trống.

    Lưu ý: dùng cho trường hợp chỉ cần đúng một cột. Khi cần nhiều cột
    phải khớp dòng với nhau, hãy dùng extract_columns() để tránh lệch dòng.
    """
    if column_name not in data:
        raise KeyError(
            f"Không tìm thấy cột '{column_name}'. "
            f"Các cột hiện có: {list(data.keys())}"
        )
    return [float(value) for value in data[column_name] if value is not None]


# ---------------------------------------------------------------------
# ④ Trích nhiều cột — làm sạch theo DÒNG, chỗ tránh lệch chỉ số
# ---------------------------------------------------------------------
def extract_columns(data, feature_cols, target_col, verbose=True):
    """
    Trích các cột đặc trưng và cột mục tiêu, loại bỏ theo DÒNG những mẫu
    có ô trống hoặc không phải số — nhờ vậy các cột luôn khớp chỉ số.

    Parameters:
        data         : dict { tên_cột: [giá trị] }
        feature_cols : tên cột đặc trưng — str (1 biến) hoặc list (nhiều biến)
        target_col   : tên cột mục tiêu
        verbose      : in tóm tắt dữ liệu đã trích

    Returns:
        feature_columns : list of lists — mỗi phần tử là 1 cột đặc trưng
        targets         : list giá trị mục tiêu
    """
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    for column_name in list(feature_cols) + [target_col]:
        if column_name not in data:
            raise KeyError(
                f"Không tìm thấy cột '{column_name}'. "
                f"Các cột hiện có: {list(data.keys())}"
            )

    selected = [data[name] for name in feature_cols] + [data[target_col]]
    num_rows = min(len(column) for column in selected)

    feature_columns = [[] for _ in feature_cols]
    targets = []
    skipped = 0

    for row_index in range(num_rows):
        try:
            values = [float(column[row_index]) for column in selected]
        except (TypeError, ValueError):
            skipped += 1
            continue
        for position in range(len(feature_cols)):
            feature_columns[position].append(values[position])
        targets.append(values[-1])

    if not targets:
        raise ValueError("Không còn dòng dữ liệu hợp lệ nào sau khi làm sạch.")

    if verbose:
        print(f"\nĐã lấy {len(targets)} mẫu hợp lệ"
              + (f" (bỏ qua {skipped} dòng thiếu/không hợp lệ)" if skipped else ""))
        for position, column_name in enumerate(feature_cols):
            column = feature_columns[position]
            print(f"  X{position + 1}: \"{column_name}\" → "
                  f"min = {min(column):.2f}, max = {max(column):.2f}")
        print(f"  Y : \"{target_col}\" → "
              f"min = {min(targets):.2f}, max = {max(targets):.2f}")

    return feature_columns, targets


# ---------------------------------------------------------------------
# ⑤ Gộp: nhận đường dẫn, trả thẳng cột số đã sẵn sàng cho mô hình
# ---------------------------------------------------------------------
def load_dataset(file_path, feature_cols, target_col,
                 sheet_name=0, header_row=1, verbose=True):
    """
    Nạp dữ liệu từ file (.xlsx hoặc .csv) và trả về thẳng các cột số đã
    làm sạch, sẵn sàng đưa vào mô hình.

    Returns:
        feature_columns : list of lists — mỗi phần tử là 1 cột đặc trưng
        targets         : list giá trị mục tiêu
        headers         : list tên tất cả các cột trong file
    """
    if verbose:
        print(f"Đang tải dữ liệu từ: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()
    if extension == '.csv':
        headers, data = load_csv_data(file_path, verbose=verbose)
    else:
        headers, data = load_excel_data(
            file_path, sheet_name=sheet_name, header_row=header_row, verbose=verbose
        )

    feature_columns, targets = extract_columns(
        data, feature_cols, target_col, verbose=verbose
    )
    return feature_columns, targets, headers
