# HƯỚNG DẪN DỰ ĐOÁN BẰNG MÔ HÌNH ĐÃ HUẤN LUYỆN

> Tài liệu này chỉ nói về việc **dùng** mô hình. Phần huấn luyện xem [README §9.3](../README.md), phần kết quả thực nghiệm xem
> [`reportResult.md`](.reports/reportResult.md).

---

## MỤC LỤC

1. [Chạy nhanh trong 30 giây](#1-chạy-nhanh-trong-30-giây)
2. [Ba thứ cần có](#2-ba-thứ-cần-có)
3. [Định dạng dữ liệu đầu vào](#3-định-dạng-dữ-liệu-đầu-vào)
4. [Các cách gọi](#4-các-cách-gọi)
5. [Đọc kết quả cho đúng](#5-đọc-kết-quả-cho-đúng)
6. [Dùng từ mã Python](#6-dùng-từ-mã-python)
7. [Bảng tra lỗi](#7-bảng-tra-lỗi)
8. [Những điều KHÔNG nên làm](#8-những-điều-không-nên-làm)
9. [Huấn luyện lại khi cần](#9-huấn-luyện-lại-khi-cần)

---

## 1. CHẠY NHANH TRONG 30 GIÂY

```powershell
.\.venv\Scripts\Activate.ps1

python src/mainPredict.py --model models/vcb.json --data data/input/VCB.csv --rows 5
```

Kết quả:

```text
Mốc thời gian        Dự đoán   Xác suất tăng   Mức tin cậy
----------------------------------------------------------
2026-08-17              TĂNG          0.5695    trung bình
2026-08-18              TĂNG          0.5464          thấp
2026-08-19              TĂNG          0.5852    trung bình
2026-08-20              TĂNG          0.5626    trung bình
2026-08-21              TĂNG          0.5846    trung bình  ← dự báo

Dự đoán ứng với biến động sau 5 phiên kể từ mốc tương ứng.
```

Dòng có dấu `← dự báo` là **dự báo thật** — mốc mới nhất, chưa thể kiểm chứng vì tương lai chưa xảy ra. Các dòng phía trên đã có thể đối chiếu với thực tế.

Chạy mất **dưới một giây**. Không có bước huấn luyện nào.

---

## 2. BA THỨ CẦN CÓ

| Thứ                     | Ở đâu                                 | Ghi chú                                                               |
| ------------------------ | ---------------------------------------- | ---------------------------------------------------------------------- |
| **Môi trường**  | `.venv` đã cài `requirements.txt` | Chỉ cần`openpyxl` nếu dùng `.xlsx`                             |
| **Gói mô hình** | `models/*.json`                        | Sinh ra bằng`--save-model`, xem [§9](#9-huấn-luyện-lại-khi-cần) |
| **Dữ liệu**      | `.csv` hoặc `.xlsx`                 | Xem[§3](#3-định-dạng-dữ-liệu-đầu-vào)                          |

Bốn gói mô hình có sẵn:

| File                | Mã | Mẫu huấn luyện | Tham số | Ngưỡng | Accuracy trên test |
| ------------------- | --- | ----------------- | -------- | -------- | ------------------- |
| `models/vnm.json` | VNM | 1 719             | 19 670   | 0.45     | 0.5237              |
| `models/vcb.json` | VCB | 2 084             | 25 450   | 0.50     | 0.4840              |
| `models/msn.json` | MSN | 2 062             | 25 935   | 0.50     | 0.5219              |
| `models/mwg.json` | MWG | 2 093             | 25 145   | 0.55     | 0.5318              |

VNM nhỏ hơn ba mã còn lại vì dữ liệu của nó bắt đầu từ 2016 thay vì 2014. Đây cũng là mã **đã dùng để dò siêu tham số**, nên kết quả của nó lạc quan hơn ba mã kia — xem [`reportResult.md`](.reports/reportResult.md) §9.

Xem thông tin một gói mà không chạy dự đoán:

```powershell
python src/mainPredict.py --model models/vcb.json --data data/input/VCB.csv --describe
```

```text
Mô tả        : VCB — Ngân hàng TMCP Ngoại thương Việt Nam. Phương pháp lấy từ base.json.
Bài toán     : classifier
Đặc trưng    : 27 cột
Kích thước   : 300 cây, 25,450 tham số
Huấn luyện   : 2084 mẫu | 2014-09-19 → 2022-12-30
Tham số/mẫu  : 12.2  (vượt 2 — mô hình đủ chỗ ghi nhớ nhiễu)
Chỉ số lúc huấn luyện:
  validation_accuracy   : 0.5359
  validation_roc_auc    : 0.5587
  test_accuracy         : 0.4840
  test_roc_auc          : 0.5589
```

---

## 3. ĐỊNH DẠNG DỮ LIỆU ĐẦU VÀO

### 3.1. Tên cột — bắt buộc đúng từng chữ

```csv
Date,Open,High,Low,Close,Volume
2026-08-19,68500.0,69200.0,68300.0,69000.0,4521300
2026-08-20,69000.0,69800.0,68900.0,69500.0,3897100
```

**Phân biệt hoa thường.** `close` không được chấp nhận, phải là `Close`.

| Cột       | Kiểu          | Ý nghĩa                                             |
| ---------- | -------------- | ----------------------------------------------------- |
| `Date`   | `YYYY-MM-DD` | Ngày giao dịch                                      |
| `Open`   | số            | Giá mở cửa                                         |
| `High`   | số            | Giá cao nhất phiên                                 |
| `Low`    | số            | Giá thấp nhất phiên                               |
| `Close`  | số            | Giá đóng cửa (nên dùng giá đã điều chỉnh) |
| `Volume` | số nguyên    | Khối lượng khớp lệnh                             |

Cột thừa được bỏ qua, không gây lỗi.

### 3.2. Số dòng tối thiểu — **50**, khuyến nghị **≥ 60**

Đây là ràng buộc hay bị bỏ sót nhất. Mô hình dùng chỉ báo `close_over_sma50` với cửa sổ 50 phiên, nên **không thể dự đoán chỉ từ một dòng dữ liệu**.

| Số dòng nạp lên | Số dự đoán nhận được |
| ------------------- | ---------------------------- |
| 20                  | 0 — báo lỗi               |
| 40                  | 0 — báo lỗi               |
| **50**        | **1**                  |
| 60                  | 11                           |
| 100                 | 51                           |

Công thức: `số dự đoán = số dòng − 49`.

### 3.3. Thứ tự dòng

Không cần sắp xếp trước — chương trình tự sắp theo `Date` tăng dần và tự khử ngày trùng. Nhưng dữ liệu phải **liên tục**: thiếu vài phiên ở giữa sẽ làm các chỉ báo cửa sổ trượt trộn lẫn hai giai đoạn cách xa nhau.

### 3.4. File Excel

Dùng được `.xlsx`, cùng yêu cầu tên cột. Chương trình tự nhận diện theo phần mở rộng của file.

---

## 4. CÁC CÁCH GỌI

```powershell
# Mặc định: 10 mốc gần nhất
python src/mainPredict.py --model models/msn.json --data data/input/MSN.csv

# Chỉ xem dự báo mới nhất
python src/mainPredict.py --model models/msn.json --data data/input/MSN.csv --rows 1

# Toàn bộ lịch sử dự đoán được
python src/mainPredict.py --model models/msn.json --data data/input/MSN.csv --rows 99999

# Đổi ngưỡng quyết định
python src/mainPredict.py --model models/msn.json --data data/input/MSN.csv --threshold 0.55

# Chỉ xem thông tin mô hình
python src/mainPredict.py --model models/msn.json --data data/input/MSN.csv --describe
```

| Tham số        | Bắt buộc | Mặc định   | Ý nghĩa                                 |
| --------------- | ---------- | ------------- | ----------------------------------------- |
| `--model`     | ✔         | —            | Đường dẫn gói mô hình`.json`     |
| `--data`      | ✔         | —            | Đường dẫn dữ liệu`.csv`/`.xlsx` |
| `--rows`      |            | 10            | Số mốc gần nhất cần in               |
| `--threshold` |            | lấy từ gói | Ngưỡng xác suất để gọi là "TĂNG" |
| `--describe`  |            | tắt          | In thông tin gói rồi thoát            |

### Về `--threshold`

Ngưỡng lưu trong gói được **dò trên tập validate** lúc huấn luyện — mỗi mã một giá trị khác nhau (VNM `0.45`, VCB và MSN `0.50`, MWG `0.55`). Nâng ngưỡng lên làm mô hình thận trọng hơn — báo "TĂNG" ít hơn nhưng chắc hơn; hạ xuống thì ngược lại.

Đây là công cụ điều chỉnh **độ thận trọng**, không phải cách làm mô hình chính xác hơn.

---

## 5. ĐỌC KẾT QUẢ CHO ĐÚNG

### 5.1. Dự đoán ứng với 5 phiên, không phải phiên kế tiếp

Dòng `2026-08-21 → TĂNG` nghĩa là: **giá đóng cửa sau 5 phiên kể từ 21/08 sẽ cao hơn giá đóng cửa ngày 21/08**. Không phải "ngày mai tăng".

Tầm nhìn ghi ngay dưới bảng và lưu trong gói (`recipe.labeling.horizon`).

### 5.2. Cột "Xác suất tăng"

Tỷ lệ cây trong rừng bỏ phiếu cho chiều tăng, sau khi lấy trung bình xác suất.
`0.5635` nghĩa là rừng nghiêng nhẹ về phía tăng.

### 5.3. Cột "Mức tin cậy"

Suy từ khoảng cách tới 0.5:

| Khoảng cách | Nhãn       |
| ------------- | ----------- |
| > 0.15        | cao         |
| 0.05 – 0.15   | trung bình  |
| < 0.05        | thấp        |

Đây chỉ là mức độ **đồng thuận giữa các cây**, **không phải xác suất dự đoán đúng**. Rừng có thể đồng thuận cao mà vẫn sai.

### 5.4. Điều quan trọng nhất

Chương trình tự in dòng cảnh báo cuối mỗi lần chạy, và nó không phải câu khách sáo:

> *Mô hình này đạt accuracy 0.4840 trên tập test lúc huấn luyện — mức chưa
> phân biệt được với ngẫu nhiên.*

Kiểm định McNemar trên cả bốn mã cho thấy **không mã nào có ưu thế so với việc đoán bừa đạt mức ý nghĩa thống kê** (p từ 0.06 đến 1.00). Chi tiết ở [`reportResult.md`](.reports/reportResult.md).

Dùng để học tập và trình bày phương pháp. **Không dùng để ra quyết định đầu tư.**

---

## 6. DÙNG TỪ MÃ PYTHON

Khi cần nhúng vào chương trình khác thay vì chạy dòng lệnh:

```python
import csv
import io
import sys

sys.path.insert(0, 'src')
from pipeline import modelBundle

# 1. Nạp gói mô hình — mất khoảng 10 ms
bundle = modelBundle.load_bundle('models/vcb.json')

# 2. Đọc dữ liệu thành dict { tên_cột: list giá trị }
rows = list(csv.DictReader(io.open('data/input/VCB.csv', encoding='utf-8')))
table = {name: [row[name] for row in rows] for name in rows[0]}

# 3. Dự đoán
results = modelBundle.predict_with_bundle(bundle, table, num_rows=5)

for item in results:
    print(f"{item['key']}  {item['prediction']}  {item['score']:.4f}")
```

Kết quả trả về là `list` các `dict`:

```python
{'key': datetime.date(2026, 8, 21), 'score': 0.5845675995432519, 'prediction': 1}
```

| Khoá          | Ý nghĩa                           |
| -------------- | ----------------------------------- |
| `key`        | Mốc thời gian (`datetime.date`) |
| `prediction` | Nhãn:`1` = tăng, `0` = giảm  |
| `score`      | Xác suất thuộc lớp tăng        |

### Các hàm hữu ích khác

```python
# Thông tin gói dưới dạng chuỗi nhiều dòng
print(modelBundle.describe_bundle(bundle))

# Chỉ dựng đặc trưng, không dự đoán — để kiểm tra dữ liệu
samples, keys, names = modelBundle.prepare_features_for_prediction(
    bundle['recipe'], table)
print(f'{len(samples)} mẫu, {len(names)} đặc trưng')

# Truy cập thẳng đối tượng rừng
forest = bundle['estimator']
print(forest.describe())
print(forest.calculate_feature_importances())
```

---

## 7. BẢNG TRA LỖI

### `Dữ liệu thiếu cột [...]`

```text
ValueError: Dữ liệu thiếu cột ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'].
Mô hình cần đúng các cột ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] (phân biệt hoa thường). File đang có: ['c', 'date', 'h', 'l', 'o', 'v'].
```

Đổi tên cột trong file cho khớp. Thông báo liệt kê sẵn cột hiện có để đối chiếu.

### `Không dựng được mẫu nào từ N dòng dữ liệu`

```text
ValueError: Không dựng được mẫu nào từ 19 dòng dữ liệu. Cửa sổ chỉ báo dài nhất trong công thức cần nhiều lịch sử hơn — hãy nạp ít nhất 60 dòng.
```

File quá ngắn. Xem [§3.2](#32-số-dòng-tối-thiểu--50-khuyến-nghị--60).

### `Không tìm thấy gói mô hình: ...`

Sai đường dẫn. Kiểm tra bằng `dir models` (Windows) hoặc `ls models/`.

### `Đặc trưng không khớp với lúc huấn luyện: ...`

Gói mô hình bị sửa tay, hoặc dùng gói của phiên bản mã nguồn khác. Huấn luyện lại theo [§9](#9-huấn-luyện-lại-khi-cần).

### `Gói mô hình phiên bản N không đọc được`

Định dạng gói đã đổi kể từ lúc lưu. Huấn luyện lại.

---

## 8. NHỮNG ĐIỀU KHÔNG NÊN LÀM

**Đừng nhập tay 27 đặc trưng.** Chúng là đại lượng dẫn xuất (`close_over_sma50 = 1.0312`, `bollinger_b = 0.4471`…), không ai gõ tay được.
Tệ hơn, tổ hợp gõ bừa sẽ tạo ra trạng thái không tồn tại trong thực tế — ví dụ `rsi_14 = 95` đi cùng `close_over_sma5 = 0.85` là mâu thuẫn. Đầu vào đúng luôn là OHLCV.

**Đừng bơm dữ liệu theo phút vào mô hình huấn luyện trên dữ liệu ngày.** Tên cột vẫn đúng nên chương trình **không báo lỗi**, nhưng `close_over_sma50` sẽ thành trung bình 50 *phút* thay vì 50 *ngày*. Kết quả vô nghĩa mà không có dấu hiệu nào.

**Đừng dùng gói của mã này cho mã khác rồi tin kết quả.** Về kỹ thuật thì chạy được — cùng định dạng là dùng được. Nhưng mô hình VNM áp lên FPT cho AUC 0.4987, đúng bằng tung đồng xu. Tín hiệu không chung giữa các mã.

**Đừng bỏ qua dòng cảnh báo cuối.** Nó ghi đúng con số accuracy của gói đang dùng, và con số đó chưa phân biệt được với ngẫu nhiên.

**Đừng dùng mô hình cũ mãi mà không kiểm tra.** Đo trên VNM cho thấy accuracy tụt từ 0.63 xuống 0.51 qua ba năm. Xem [§9](#9-huấn-luyện-lại-khi-cần).

---

## 9. HUẤN LUYỆN LẠI KHI CẦN

### Khi nào

- Có thêm nhiều dữ liệu mới (vài tháng trở lên)
- Đổi bộ đặc trưng hoặc siêu tham số trong `config/base.json`
- Theo dõi thấy ưu thế so với mốc đoán bừa đã biến mất

### Cách làm

```powershell
python src/mainClassification.py --config config/vcb.json `
                                 --no-figures `
                                 --save-model models/vcb.json
```

Mất khoảng 60 giây. Script in ra toàn bộ chỉ số đánh giá trước khi lưu, nên xem kỹ phần **so với các mốc đối chứng** trước khi dùng gói mới.

### Huấn luyện cho mã mới

1. Đặt file `.csv` vào `data/input/`
2. Tạo `config/<mã>.json`, chép từ `config/vcb.json` rồi đổi ba chỗ:

```json
{
  "extends": "config/base.json",
  "description": "ABC — Tên công ty.",
  "dataset": {
    "path": "data/input/ABC.csv",
    "label": "ABC",
    "key_column": "Date",
    "numeric_columns": ["Open", "High", "Low", "Close", "Volume"],
    "series": {
      "open": "Open", "high": "High", "low": "Low",
      "close": "Close", "volume": "Volume"
    }
  },
  "output": { "figure_dir": "data/output", "prefix": "ABC_classification" }
}
```

3. Chạy lệnh huấn luyện ở trên với `--config config/abc.json`

**Không sửa `config/base.json`** trừ khi muốn đổi phương pháp cho *tất cả* các mã — đó chính là mục đích của file đó.

---

<div align="center">
