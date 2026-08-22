# =====================================================================
# Data Loader — nạp dữ liệu từ file vào cấu trúc list thuần
# =====================================================================
# Tầng vào/ra duy nhất của project. Các module thuật toán chỉ nhận
# list số, không tự đọc file.
#
# Thứ tự khai báo: hai đầu vào thô, khác nhau ở định dạng file.
#
#   ① Đọc Excel — trả về dict of lists
#   ② Đọc CSV   — cùng vai trò với ①, cùng cấu trúc trả về
#
# Module cố ý DỪNG ở mức đọc thô. Việc ép kiểu, làm sạch và dựng đặc
# trưng thuộc về tầng pipeline, vì chúng phụ thuộc vào cấu hình của
# từng bài toán còn việc đọc file thì không.
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

