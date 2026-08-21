# DỰ ĐOÁN XU HƯỚNG GIÁ CỔ PHIẾU BẰNG RANDOM FOREST

> **Mini Project — Machine Learning**
> Sử dụng dữ liệu giá cổ phiếu theo ngày (OHLCV) để dự đoán **xu hướng tăng/giảm**
> của phiên kế tiếp, kết hợp **phân tích kỹ thuật** và mô hình **Random Forest**
> (có đối chiếu với **Gradient Boosting**).

---

## MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)
2. [Bám sát yêu cầu đề bài](#2-bám-sát-yêu-cầu-đề-bài)
3. [Cấu trúc dự án](#3-cấu-trúc-dự-án)
4. [Dữ liệu](#4-dữ-liệu)
5. [Quy trình học máy](#5-quy-trình-học-máy)
6. [Đặc trưng — Phân tích kỹ thuật](#6-đặc-trưng--phân-tích-kỹ-thuật)
7. [Mô hình](#7-mô-hình)
8. [Đánh giá mô hình](#8-đánh-giá-mô-hình)
9. [Cài đặt và chạy](#9-cài-đặt-và-chạy)
10. [Quy trình Commit — Review — Merge](#10-quy-trình-commit--review--merge)
11. [Lộ trình thực hiện](#11-lộ-trình-thực-hiện)
12. [Hạn chế và hướng phát triển](#12-hạn-chế-và-hướng-phát-triển)
13. [Tài liệu tham khảo](#13-tài-liệu-tham-khảo)

> **Kết quả thực nghiệm** — số liệu đầy đủ của cả hai nhánh, đối chiếu với sáu tiêu
> chí chấp nhận ở [Mục 8.4](#84-tiêu-chí-chấp-nhận): xem
> [`src/.reports/reportResult.md`](src/.reports/reportResult.md).

---

## 1. GIỚI THIỆU

### 1.1. Bài toán

Cho chuỗi dữ liệu giá cổ phiếu theo ngày gồm `Date, Open, High, Low, Close, Volume`,
dự án giải quyết **hai bài toán song song** trên cùng một tập đặc trưng:

| Nhánh                | Bài toán                                   | Biến mục tiêu                                                 | Kiểu học             |
| --------------------- | -------------------------------------------- | ---------------------------------------------------------------- | ---------------------- |
| **A — Chính** | Dự đoán xu hướng phiên kế tiếp       | `y = 1` nếu `Close(t+1) > Close(t)`, ngược lại `y = 0` | Phân loại nhị phân |
| **B — Phụ**   | Dự đoán giá đóng cửa phiên kế tiếp | `y = Close(t+1)` (hoặc `return(t+1)`)                       | Hồi quy               |

Nhánh A bám đúng phát biểu của đề bài. Nhánh B tận dụng bộ chỉ số hồi quy
(`RMSE`, `MAE`, `R²`) đã có sẵn trong `src/utilities/metrics.py`, đồng thời cho
phép **suy ngược ra xu hướng** từ giá dự đoán để so sánh chéo với nhánh A.

**Tầm nhìn dự báo (`horizon`) là tham số, không phải hằng số.** Bảng trên mô tả
trường hợp `horizon = 1` (phiên kế tiếp). Thực nghiệm cho thấy tín hiệu ở tầm
nhìn 1 phiên yếu hơn hẳn tầm nhìn 5 phiên — nhiễu ngắn hạn lấn át. Vì vậy cấu
hình mặc định của dự án dùng `horizon = 5`; đổi sang giá trị khác chỉ cần sửa
một dòng trong `config/*.json`. Số liệu của cả hai tầm nhìn có trong
[`reportResult.md`](src/.reports/reportResult.md).

### 1.2. Triết lý cài đặt

Dự án cài đặt Random Forest theo **hai tầng**:

- **Tầng lõi — thuần Python (`src/libraries/`)**: tự xây dựng Decision Tree (CART),
  Bootstrap Sampling và Random Feature Selection từ đầu, không dùng thư viện ML.
  Mục tiêu là **hiểu và chứng minh được cơ chế thuật toán**, đúng theo quy ước của
  các dự án trước trong repo (`ols_*`, `poly_gd_*`).
- **Tầng đối chiếu — scikit-learn (`src/notebooks/`)**: dùng `RandomForestClassifier`
  / `GradientBoostingClassifier` làm **baseline** để kiểm chứng rằng cài đặt thuần
  Python cho kết quả tương đương, và để chạy các bước tinh chỉnh siêu tham số
  quy mô lớn.

> Cơ sở lý thuyết đầy đủ (công thức, chứng minh giảm phương sai, OOB error,
> feature importance, so sánh RF ↔ GB) được trình bày trong
> [`src/.reports/reportAlgorithm.md`](src/.reports/reportAlgorithm.md).

---

## 2. BÁM SÁT YÊU CẦU ĐỀ BÀI

| # | Yêu cầu                                              | Được đáp ứng tại                                                                                                                                                                    |
| - | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1 | **Đầy đủ quy trình học máy**              | [Mục 5](#5-quy-trình-học-máy) — 8 bước từ nạp dữ liệu → tiền xử lý thời gian → tạo đặc trưng → gán nhãn → tách tập → huấn luyện → tinh chỉnh → đánh giá |
| 2 | **Tiền xử lý dữ liệu thời gian**           | [Mục 5.2](#52-bước-2--tiền-xử-lý-dữ-liệu-thời-gian) — chuẩn hóa ngày, sắp xếp tăng dần, xử lý phiên thiếu, khử rò rỉ dữ liệu                                      |
| 3 | **Phân tích kỹ thuật**                       | [Mục 6](#6-đặc-trưng--phân-tích-kỹ-thuật) — SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ROC, OBV, Stochastic                                                                        |
| 4 | **Random Forest / Gradient Boosting**            | [Mục 7](#7-mô-hình) — RF thuần Python + baseline scikit-learn + so sánh với GB                                                                                                       |
| 5 | **Kết quả chạy tốt với dữ liệu validate** | [Mục 8.3](#83-chiến-lược-kiểm-định-walk-forward) và [8.4](#84-tiêu-chí-chấp-nhận) — walk-forward validation, OOB error, tiêu chí chấp nhận định lượng                   |
| 6 | **Đánh giá mô hình**                        | [Mục 8](#8-đánh-giá-mô-hình) — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Feature Importance                                                                        |
| 7 | **Commit — Review — Merge chuẩn**             | [Mục 10](#10-quy-trình-commit--review--merge) — Git Flow rút gọn, Conventional Commits, PR checklist, squash merge                                                                     |

---

## 3. CẤU TRÚC DỰ ÁN

```text
BOT--Machine-Learning/
│
├── config/                             # THAM SỐ CỦA THÍ NGHIỆM — dữ liệu, không phải mã
│   ├── classification.json             [x] Nhánh A: đường dẫn, đặc tả đặc trưng, siêu tham số
│   └── regression.json                 [x] Nhánh B
│
├── data/
│   ├── input/                          # Dữ liệu thô (.csv / .xlsx) — không commit
│   └── output/                         # Đồ thị, bảng kết quả — không commit
│
├── src/
│   ├── libraries/                      # TẦNG LÕI — thuật toán thuần Python
│   │   ├── __init__.py                 [x]
│   │   ├── rfMath.py                   [x] Gini, Entropy, độ lợi, bootstrap, bộ tích luỹ quét ngưỡng
│   │   ├── decisionTree.py             [x] Cây quyết định CART (phân loại + hồi quy)
│   │   ├── randomForest.py             [x] Bagging + random feature + OOB + importance
│   │   ├── gradientBoosting.py         [x] GB hồi quy và GB phân loại (bước Newton tại lá)
│   │   └── rfPlot.py                   [x] 14 hàm vẽ, không gắn với đề tài
│   │
│   ├── pipeline/                       # TẦNG BÀI TOÁN — chuỗi thời gian và phân tích kỹ thuật
│   │   ├── __init__.py                 [x]
│   │   ├── timePreprocess.py           [x] Sắp xếp, khử trùng, điền tiến, tương quan
│   │   ├── technicalIndicators.py      [x] 20 chỉ báo, tất cả đều nhân quả
│   │   ├── featureBuilder.py           [x] Thi hành BẢN ĐẶC TẢ đặc trưng từ config
│   │   ├── labeling.py                 [x] Nhãn xu hướng, mục tiêu hồi quy, cân bằng lớp
│   │   ├── splitter.py                 [x] Tách theo thời gian, walk-forward
│   │   └── experiment.py               [x] Điều phối: nạp → sạch → đặc trưng → nhãn → tách
│   │
│   ├── utilities/                      # TIỆN ÍCH DÙNG CHUNG
│   │   ├── __init__.py                 [x]
│   │   ├── dataLoader.py               [x] Đọc .csv/.xlsx, làm sạch theo dòng
│   │   ├── metrics.py                  [x] SSE, MSE, RMSE, MAE, MAPE, R², R²_adj, DirAcc
│   │   ├── metricsClassification.py    [x] Accuracy, Precision, Recall, F1, ROC-AUC, MCC, ngưỡng
│   │   └── plotStyle.py                [x] Bảng màu, kích thước, save_figure()
│   │
│   ├── notebooks/                      # TẦNG DUY NHẤT được khai báo giá trị cụ thể
│   │   ├── 01_eda.ipynb                [x] Khảo sát dữ liệu
│   │   ├── 02_features.ipynb           [x] Chỉ báo kỹ thuật + kiểm tra tính nhân quả
│   │   ├── 03_train_classify.ipynb     [x] Nhánh A đầy đủ, có phân tích quá khớp
│   │   ├── 04_train_regress.ipynb      [x] Nhánh B, minh hoạ giới hạn không ngoại suy
│   │   └── 05_baseline_sklearn.ipynb   [x] Đối chiếu scikit-learn + Gradient Boosting
│   │
│   ├── mainClassification.py           [x] Chạy nhánh A đầu-cuối từ file cấu hình
│   ├── mainRegression.py               [x] Chạy nhánh B đầu-cuối từ file cấu hình
│   │
│   └── .reports/
│       ├── reportAlgorithm.md          [x] Chuyên đề lý thuyết Random Forest
│       └── reportResult.md             [x] Báo cáo kết quả thực nghiệm
│
├── tests/                              # KIỂM THỬ — 120 test, chạy trong ~6 giây
│   ├── conftest.py                     [x] Đưa src/ vào đường dẫn tìm kiếm
│   ├── test_indicators.py              [x] Nhân quả, giữ độ dài, đúng công thức
│   ├── test_metrics.py                 [x] Đối chiếu giá trị tính tay
│   └── test_forest.py                  [x] Gini/Entropy, bootstrap ≈ 1/e, tái lập
│
├── .gitignore                          [x]
├── requirements.txt                    [x]
└── README.md                           [x] tài liệu này
```

**Nguyên tắc phân tầng:**

- `libraries/` **không biết gì** về cổ phiếu — chỉ nhận `list` số và trả về `list` số.
  Nhờ vậy có thể tái sử dụng cho bài toán khác.
- `pipeline/` chứa logic của chuỗi thời gian và phân tích kỹ thuật, nhưng vẫn chỉ
  nhận dãy số và tên cột do người gọi truyền vào.
- `utilities/` là tầng vào/ra và đo lường, không chứa thuật toán học máy.

**Quy tắc "không hằng số của đề tài trong mã nguồn":**

> Không file `.py` nào trong `libraries/`, `pipeline/` hay `utilities/` được phép
> khai báo biến mang tên hoặc giá trị cụ thể của đề tài. Tên biến phải thuộc về
> **thuật toán** (`series`, `window`, `targets`, `impurity`), không thuộc về **dữ
> liệu** (`close_price`, `vnm_volume`). Riêng các tham số mô hình có giá trị mặc
> định chuẩn — `max_features='sqrt'`, `min_samples_leaf`, `learning_rate` — vẫn
> được khai báo vì chúng thuộc về thuật toán.
>
> Mọi giá trị cụ thể chỉ xuất hiện ở hai nơi: **file cấu hình `config/*.json`** và
> **notebook**. Nhờ vậy chuyển dự án sang đề tài khác chỉ cần đổi cấu hình, không
> đụng tới một dòng mã nào.

**Quy ước đặt tên:**

| Đối tượng | Quy ước | Ví dụ |
| --- | --- | --- |
| Tên file `.py` | camelCase | `randomForest.py`, `metricsClassification.py` |
| Hàm, biến | snake_case | `calculate_gini_impurity`, `feature_names` |
| Lớp | PascalCase | `RandomForestClassifier`, `TreeNode` |
| Hằng số module | UPPER_SNAKE | `IMPURITY_FUNCTIONS`, `FIG_SIZE_COMPACT` |

**Quy ước trình bày trong mỗi file** (theo mẫu `utilities/metrics.py`): khối tiêu đề
`# ===` nêu vai trò module và **chuỗi phụ thuộc ①→⑨** giữa các định nghĩa, sau đó
mỗi định nghĩa có một divider `# ---` mang số thứ tự và **lý do nó đứng ở vị trí
đó**. Đọc dọc các divider là nắm được mạch suy diễn của cả file.

---

## 4. DỮ LIỆU

### 4.1. Định dạng đầu vào

Đặt file dữ liệu vào `data/input/`. Định dạng hỗ trợ: `.csv` và `.xlsx`
(đọc bởi `utilities/dataLoader.py`).

| Cột       | Kiểu | Mô tả                                              |
| ---------- | ----- | ---------------------------------------------------- |
| `Date`   | date  | Ngày giao dịch (`YYYY-MM-DD`)                    |
| `Open`   | float | Giá mở cửa                                        |
| `High`   | float | Giá cao nhất phiên                                |
| `Low`    | float | Giá thấp nhất phiên                              |
| `Close`  | float | Giá đóng cửa (ưu tiên giá đã điều chỉnh) |
| `Volume` | int   | Khối lượng khớp lệnh                            |

Ví dụ:

```csv
Date,Open,High,Low,Close,Volume
2020-01-02,27.50,27.95,27.40,27.85,4321000
2020-01-03,27.90,28.10,27.60,27.65,3980500
```

### 4.2. Nguồn dữ liệu mẫu

- **Cổ phiếu Việt Nam**: `vnstock`, CafeF, VNDirect (xuất CSV lịch sử).
- **Quốc tế**: Yahoo Finance (`yfinance`), Stooq, Nasdaq Data Link.

### 4.3. Yêu cầu về độ dài chuỗi

Khuyến nghị tối thiểu **1.000 phiên (~4 năm)** để:

- đủ mẫu sau khi mất khoảng `30` dòng đầu do cửa sổ trượt của các chỉ báo;
- tập validate và test mỗi tập có ít nhất vài trăm phiên, đủ để chỉ số ổn định;
- bao phủ được nhiều pha thị trường (tăng, giảm, đi ngang).

---

## 5. QUY TRÌNH HỌC MÁY

```text
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ 1. Nạp DL  │ → │ 2. Tiền xử │ → │ 3. Đặc     │ → │ 4. Gán     │
│  CSV/XLSX  │   │  lý t.gian │   │  trưng KT  │   │  nhãn      │
└────────────┘   └────────────┘   └────────────┘   └─────┬──────┘
                                                         │
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌─────▼──────┐
│ 8. Báo cáo │ ← │ 7. Đánh    │ ← │ 6. Tinh    │ ← │ 5. Tách    │
│  & lưu KQ  │   │  giá       │   │  chỉnh HP  │   │  tập t.gian│
└────────────┘   └────────────┘   └─────┬──────┘   └────────────┘
                                        │ ↑
                                  ┌─────▼──────┐
                                  │ Huấn luyện │
                                  │ Random For.│
                                  └────────────┘
```

### 5.1. Bước 1 — Nạp dữ liệu

Dùng `utilities/dataLoader.load_csv_data()`: tự nhận diện `.csv`/`.xlsx`, trả về
các cột số đã làm sạch theo **dòng** — loại bỏ đồng bộ những dòng thiếu giá trị nên
các cột luôn khớp chỉ số, tránh lệch dòng (một lỗi thường gặp khi làm chuỗi thời gian).

### 5.2. Bước 2 — Tiền xử lý dữ liệu thời gian

| Việc                      | Cách xử lý                                                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chuẩn hóa ngày          | Ép`Date` về kiểu ngày, **sắp xếp tăng dần**, loại ngày trùng                                                                         |
| Phiên nghỉ / thiếu      | Chỉ giữ ngày giao dịch thực tế;**không** nội suy ngày nghỉ lễ, cuối tuần                                                             |
| Giá trị thiếu rải rác | Điền tiến (forward fill) tối đa 1–2 phiên; quá ngưỡng thì loại bỏ dòng                                                                    |
| Giá trị bất thường    | Phát hiện qua biến động ngày vượt 40%, đối chiếu sự kiện chia tách cổ phiếu                                                             |
| Điều chỉnh giá         | Ưu tiên giá đã điều chỉnh để tránh gãy chuỗi do chia cổ tức/thưởng                                                                     |
| Chuẩn hóa thang đo      | RF**không cần** chuẩn hóa (bất biến với phép biến đổi đơn điệu); nếu vẫn dùng thì chỉ ước lượng tham số trên tập train |

> **Nguyên tắc chống rò rỉ dữ liệu (data leakage)** — điểm sống còn của dự án:
>
> 1. Mọi chỉ báo tại thời điểm `t` chỉ được tính từ dữ liệu `≤ t` (cửa sổ nhân quả).
> 2. Nhãn `y(t)` lấy từ `Close(t+1)` — **dịch nhãn về sau, không dịch đặc trưng về trước**.
> 3. Tách tập **theo thứ tự thời gian**, tuyệt đối **không xáo trộn** (`shuffle=False`).
> 4. Mọi tham số chuẩn hóa/điền thiếu chỉ được ước lượng trên tập train.

### 5.3. Bước 3 — Sinh đặc trưng kỹ thuật

Xem chi tiết tại [Mục 6](#6-đặc-trưng--phân-tích-kỹ-thuật). Sau bước này, `N` dòng
đầu (bằng cửa sổ chỉ báo dài nhất) sẽ bị loại vì chưa đủ dữ liệu lịch sử.

### 5.4. Bước 4 — Gán nhãn

```python
# Nhánh A — phân loại xu hướng
y_cls[t] = 1 if close[t + 1] > close[t] else 0

# Nhánh B — hồi quy
y_reg[t] = close[t + 1]                              # hoặc dùng tỷ suất sinh lời:
# y_reg[t] = (close[t + 1] - close[t]) / close[t]
```

Dòng cuối cùng bị loại do không có nhãn. Sau đó kiểm tra **cân bằng lớp**: tỷ lệ lớp `1`
thường nằm trong khoảng `48–54%`. Nếu lệch quá `60/40`, bật `class_weight='balanced'`
hoặc cân bằng lại mẫu.

### 5.5. Bước 5 — Tách tập theo thời gian

```text
|<---------- TRAIN 70% ---------->|<-- VALIDATE 15% -->|<-- TEST 15% -->|
2016-01                        2022-06             2023-09         2024-12
```

- **Train** — huấn luyện mô hình.
- **Validate** — chọn siêu tham số và ngưỡng quyết định; **không** dùng để huấn luyện.
- **Test** — chỉ chạy **một lần duy nhất** ở cuối, mô phỏng dữ liệu tương lai chưa từng thấy.
- Có thể chèn **gap** vài phiên giữa các tập để triệt tiêu tương quan biên.

### 5.6. Bước 6 — Huấn luyện

```python
from libraries.randomForest import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,        # B — số cây trong rừng
    max_features='sqrt',     # m — số đặc trưng xét mỗi nút
    max_depth=5,             # dữ liệu tài chính nhiễu → cây phải nông
    min_samples_leaf=40,     # và lá phải đủ lớn để không học thuộc nhiễu
    bootstrap=True,          # bật để dùng được OOB error
    random_state=42,         # cố định để tái lập kết quả
)
model.fit(train_samples, train_targets)
```

Trong thực tế không gọi trực tiếp như trên mà chạy qua script, để mọi tham số
đến từ file cấu hình thay vì nằm rải rác trong mã:

```powershell
python src/mainClassification.py --config config/classification.json
```

### 5.7. Bước 7 — Tinh chỉnh siêu tham số

Theo đúng quy trình 6 bước trong `reportAlgorithm.md` §9:

| Siêu tham số       | Lưới dò tìm                | Ảnh hưởng                                                      |
| -------------------- | ------------------------------ | ----------------------------------------------------------------- |
| `n_estimators`     | 100 → 200 → 500 → 1000      | Tăng đến khi đường cong OOB đi ngang                       |
| `max_features`     | `{1, log₂p, √p, p/3, p/2}` | **Quan trọng nhất** — cân bằng strength ↔ correlation |
| `min_samples_leaf` | `{1, 3, 5, 10, 20}`          | Chống quá khớp với dữ liệu nhiễu như giá cổ phiếu      |
| `max_depth`        | `{None, 8, 12, 16}`          | Giới hạn khi mô hình bắt đầu học thuộc nhiễu            |

Kiểm định bằng **`TimeSeriesSplit`** (hoặc walk-forward tự cài), **không** dùng
`KFold` ngẫu nhiên — vì xáo trộn thời gian sẽ cho phép mô hình "nhìn thấy tương lai".

### 5.8. Bước 8 — Đánh giá và báo cáo

Xem [Mục 8](#8-đánh-giá-mô-hình). Toàn bộ đồ thị lưu về `data/output/` thông qua
`utilities.plotStyle.save_figure()`; số liệu tổng hợp ghi vào
`src/.reports/reportResult.md`.

---

## 6. ĐẶC TRƯNG — PHÂN TÍCH KỸ THUẬT

### 6.1. Nhóm xu hướng (Trend)

| Chỉ báo           | Công thức / Ý nghĩa                                                        | Tham số      |
| ------------------- | ------------------------------------------------------------------------------ | ------------- |
| **SMA**       | `SMA(n) = (1/n)·Σ Close(t-i)` — trung bình động đơn giản            | 5, 10, 20, 50 |
| **EMA**       | `EMA(t) = α·Close(t) + (1-α)·EMA(t-1)`, `α = 2/(n+1)`                 | 12, 26        |
| **MACD**      | `EMA(12) - EMA(26)`; đường tín hiệu `EMA(9)` của MACD; histogram     | (12, 26, 9)   |
| **Price/SMA** | `Close / SMA(n)` — vị thế giá so với trung bình, đã chuẩn hóa sẵn | 20, 50        |
| **SMA cross** | `1` nếu `SMA(5) > SMA(20)` — tín hiệu giao cắt                        | (5, 20)       |

### 6.2. Nhóm động lượng (Momentum)

| Chỉ báo               | Công thức / Ý nghĩa                                                             | Tham số |
| ----------------------- | ----------------------------------------------------------------------------------- | -------- |
| **RSI**           | `100 - 100/(1 + RS)` với `RS = AvgGain/AvgLoss` — quá mua >70, quá bán <30 | 14       |
| **ROC**           | `(Close(t) - Close(t-n)) / Close(t-n) · 100`                                     | 5, 10    |
| **Stochastic %K** | `(Close - Low(n)) / (High(n) - Low(n)) · 100`                                    | 14       |
| **Lag returns**   | `return(t-1), return(t-2), …, return(t-5)`                                       | 1–5     |

### 6.3. Nhóm biến động (Volatility)

| Chỉ báo                 | Công thức / Ý nghĩa                                              | Tham số |
| ------------------------- | -------------------------------------------------------------------- | -------- |
| **Bollinger Bands** | `SMA(20) ± 2σ`; dùng `%B` và `Bandwidth` làm đặc trưng | (20, 2)  |
| **ATR**             | Trung bình`TrueRange` — biên độ dao động thực              | 14       |
| **Rolling std**     | Độ lệch chuẩn của`return` trong cửa sổ trượt              | 10, 20   |
| **High-Low range**  | `(High - Low) / Close` — biên độ tương đối trong phiên    | —       |

### 6.4. Nhóm khối lượng (Volume)

| Chỉ báo              | Công thức / Ý nghĩa                                              | Tham số |
| ---------------------- | -------------------------------------------------------------------- | -------- |
| **Volume ratio** | `Volume / SMA_Volume(20)` — phát hiện đột biến khối lượng | 20       |
| **OBV**          | Cộng dồn khối lượng theo dấu của biến động giá            | —       |

### 6.5. Lưu ý khi thiết kế đặc trưng

- **Ưu tiên đặc trưng dạng tỷ lệ** (`Close/SMA20`, `%B`, `return`) thay vì giá tuyệt đối.
  Random Forest **không ngoại suy được** (`reportAlgorithm.md` §4.3): nếu huấn luyện
  trên vùng giá 20–30 thì mô hình sẽ không bao giờ dự đoán ra giá 50.
- **Loại bỏ đặc trưng tương quan quá cao** (`|r| > 0.95`) — chúng làm loãng chỉ số tầm
  quan trọng (MDI bias, §4.4) dù ít ảnh hưởng tới độ chính xác.
- Tổng số đặc trưng mục tiêu: **20–35**, đủ đa dạng để random feature selection phát huy
  tác dụng nhưng không quá thưa.

---

## 7. MÔ HÌNH

### 7.1. Random Forest — cài đặt thuần Python

Ba trụ cột của thuật toán được cài trong `src/libraries/`:

| Thành phần                   | Module               | Nội dung                                                                    |
| ------------------------------ | -------------------- | ---------------------------------------------------------------------------- |
| Cây quyết định             | `decisionTree.py` | CART đệ quy, tìm ngưỡng chia tối ưu, điều kiện dừng               |
| Tiêu chí phân tách         | `rfMath.py`       | Gini, Entropy, Information Gain (phân loại); variance reduction (hồi quy) |
| Bagging                        | `randomForest.py` | Bootstrap`n` mẫu có hoàn lại cho mỗi cây                             |
| Ngẫu nhiên hóa đặc trưng | `randomForest.py` | Xét ngẫu nhiên`m = √p` đặc trưng tại mỗi nút                     |
| Tổng hợp dự đoán          | `randomForest.py` | Majority voting (phân loại) / trung bình cộng (hồi quy)                 |
| OOB & Importance               | `randomForest.py` | OOB error, MDI, permutation importance                                       |

### 7.2. Baseline scikit-learn

Notebook `05_baseline_sklearn.ipynb` chạy `RandomForestClassifier`,
`RandomForestRegressor` và `GradientBoostingClassifier` trên **cùng tập đặc trưng và
cùng cách tách tập**, nhằm:

1. **Kiểm chứng** cài đặt thuần Python — chênh lệch Accuracy chấp nhận được: `≤ 2%`.
2. Chạy `GridSearchCV` + `TimeSeriesSplit` cho không gian siêu tham số lớn.
3. So sánh **Random Forest ↔ Gradient Boosting** theo đúng phân tích tại
   `reportAlgorithm.md` §10.2 (RF giảm **phương sai**, GB giảm **độ chệch**).

### 7.3. Mô hình đối chứng (bắt buộc có)

Với dữ liệu tài chính, mọi kết quả phải được so với các mốc tầm thường:

| Đối chứng                  | Cách dự đoán                                | Vai trò                                                          |
| ----------------------------- | ----------------------------------------------- | ----------------------------------------------------------------- |
| **Majority class**      | Luôn dự đoán lớp chiếm đa số            | Mốc sàn tuyệt đối                                            |
| **Random guess**        | Dự đoán ngẫu nhiên 50/50                   | Kiểm tra ROC-AUC ≈ 0.5                                          |
| **Naive persistence**   | Xu hướng hôm nay lặp lại vào ngày mai    | Mốc sàn theo chuỗi thời gian                                  |
| **Logistic Regression** | Mô hình tuyến tính trên cùng đặc trưng | Kiểm tra RF có thực sự khai thác được quan hệ phi tuyến |

---

## 8. ĐÁNH GIÁ MÔ HÌNH

### 8.1. Chỉ số cho nhánh phân loại (A)

| Chỉ số                   | Công thức                     | Vì sao dùng                                                                                                   |
| -------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Accuracy**         | `(TP+TN)/(TP+TN+FP+FN)`       | Chỉ số tổng quát, hợp lệ khi hai lớp cân bằng                                                          |
| **Precision**        | `TP/(TP+FP)`                  | Trong các phiên dự đoán "tăng", bao nhiêu phiên tăng thật — gắn trực tiếp với rủi ro vào lệnh |
| **Recall**           | `TP/(TP+FN)`                  | Bắt được bao nhiêu phần trăm số phiên tăng thực tế                                                  |
| **F1-score**         | `2·P·R/(P+R)`               | Cân bằng giữa Precision và Recall                                                                           |
| **ROC-AUC**          | Diện tích dưới đường ROC | Đánh giá khả năng xếp hạng, độc lập với ngưỡng quyết định                                       |
| **Confusion Matrix** | Ma trận 2×2                   | Cho thấy mô hình sai lệch về phía nào                                                                    |

### 8.2. Chỉ số cho nhánh hồi quy (B)

Dùng trực tiếp `utilities/metrics.py` đã có: **SSE, MSE, RMSE, MAE, R², R²_adj**
(`calculate_all_metrics()` trả về trọn bộ trong một lần gọi).

Bổ sung riêng cho chuỗi thời gian:

- **MAPE** — sai số phần trăm, dễ diễn giải với giá.
- **Directional Accuracy** — tỷ lệ dự đoán đúng **dấu** biến động, dùng để so sánh trực
  tiếp nhánh B với nhánh A.

### 8.3. Chiến lược kiểm định (walk-forward)

```text
Vòng 1: train [=======]        → validate [--]
Vòng 2: train [=========]      → validate    [--]
Vòng 3: train [===========]    → validate       [--]
Vòng 4: train [=============]  → validate          [--]
                                 (cửa sổ mở rộng, luôn tiến theo thời gian)
```

Kết quả cuối = **trung bình ± độ lệch chuẩn** qua các vòng. Độ lệch chuẩn lớn là dấu
hiệu mô hình không ổn định giữa các pha thị trường.

Bổ sung **OOB error** — ước lượng lỗi "miễn phí" từ khoảng 36.8% số mẫu không được
bootstrap chọn ở mỗi cây (`reportAlgorithm.md` §7.7), dùng để chọn `n_estimators`
mà không tiêu tốn thêm dữ liệu validate.

### 8.4. Tiêu chí chấp nhận

Mô hình được coi là **"chạy tốt với dữ liệu validate"** khi thỏa **đồng thời**:

| # | Tiêu chí                             | Ngưỡng                                                                  |
| - | -------------------------------------- | ------------------------------------------------------------------------- |
| 1 | Accuracy trên validate                | `> 55%` và **cao hơn** mọi mô hình đối chứng ở §7.3     |
| 2 | ROC-AUC trên validate                 | `> 0.55` — tách rõ khỏi mức ngẫu nhiên 0.50                      |
| 3 | Chênh lệch Train ↔ Validate         | Accuracy lệch`< 10%`; lớn hơn là dấu hiệu quá khớp              |
| 4 | Ổn định qua các vòng walk-forward | Độ lệch chuẩn Accuracy`< 5%`                                        |
| 5 | F1-score của cả hai lớp             | Không lớp nào có F1`< 0.40` (mô hình không "bỏ rơi" một lớp) |
| 6 | Kết quả trên test                   | Không suy giảm quá`5%` so với validate                              |

> **Ghi chú trung thực về kỳ vọng**: thị trường tài chính có tỷ lệ nhiễu trên tín hiệu
> rất cao. Accuracy trong khoảng **55–60%** đã là kết quả tốt và đáng tin với bài toán
> này. Nếu mô hình cho Accuracy `> 70%`, khả năng cao nhất là **có rò rỉ dữ liệu** —
> cần rà lại toàn bộ 4 nguyên tắc ở [Mục 5.2](#52-bước-2--tiền-xử-lý-dữ-liệu-thời-gian)
> trước khi báo cáo kết quả.

### 8.5. Biểu đồ đầu ra (lưu tại `data/output/`)

Hai script chạy đặt tên file theo mẫu `<prefix>_<tên>.png`, với `prefix` khai báo
trong `config/*.json`.

| File | Nội dung |
| --- | --- |
| `<prefix>_class_distribution.png` | Phân bố lớp sau khi loại quan sát đứng yên |
| `<prefix>_oob_curve.png` | Lỗi OOB theo số cây `B` |
| `<prefix>_feature_importance.png` | Top đặc trưng theo MDI **và** permutation importance |
| `<prefix>_confusion_matrix.png` | Ma trận nhầm lẫn |
| `<prefix>_roc_curve.png` | Đường ROC kèm giá trị AUC |
| `<prefix>_threshold_curve.png` | Accuracy / F1 / Balanced accuracy theo ngưỡng quyết định |
| `<prefix>_walk_forward.png` | Điểm đánh giá qua từng vòng kiểm định tiến dần |
| `<prefix>_predicted_vs_actual.png` | Tán xạ dự đoán ↔ thực tế (nhánh B) |
| `<prefix>_series_comparison.png` | Chuỗi thực tế và chuỗi dự đoán theo thời gian (nhánh B) |
| `<prefix>_residuals.png` | Phân bố sai số (nhánh B) |

Các notebook lưu thêm `<SYMBOL>_01_price_history.png`, `<SYMBOL>_02_indicators.png`,
`<SYMBOL>_02_correlation.png` và `<SYMBOL>_05_roc_comparison.png`.

---

## 9. CÀI ĐẶT VÀ CHẠY

### 9.1. Yêu cầu

- Python **3.10+** (dự án đang phát triển trên 3.13)
- Git

### 9.2. Cài đặt

```powershell
# Clone
git clone <repository-url>
cd BOT--Machine-Learning

# Tạo môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

# Cài thư viện
pip install -r requirements.txt
```

**Thư viện sử dụng:**

| Nhóm        | Gói                                                                   | Vai trò                                                        |
| ------------ | ---------------------------------------------------------------------- | --------------------------------------------------------------- |
| Bắt buộc   | `openpyxl`, `matplotlib`, `jupyter`, `notebook`, `ipykernel` | Đọc Excel, vẽ đồ thị, chạy notebook                      |
| Cho baseline | `numpy`, `pandas`, `scikit-learn`                                | Chỉ dùng trong`05_baseline_sklearn.ipynb` để đối chiếu |

Tầng lõi `src/libraries/` **không phụ thuộc** vào bất kỳ thư viện ngoài nào — chạy được
với Python chuẩn.

### 9.3. Chạy

Đặt file dữ liệu vào `data/input/` rồi trỏ `config/*.json` tới nó.

```powershell
# Nhánh A — phân loại xu hướng tăng/giảm
python src/mainClassification.py --config config/classification.json

# Nhánh B — hồi quy tỷ suất biến động
python src/mainRegression.py --config config/regression.json

# Thêm --no-figures nếu chỉ cần số liệu, bỏ qua bước vẽ
python src/mainClassification.py --config config/classification.json --no-figures

# Hoặc chạy từng bước bằng notebook
jupyter notebook src/notebooks/
```

Kết quả (đồ thị, bảng chỉ số) được ghi vào `data/output/`.

### 9.4. Chạy kiểm thử

```powershell
python -m pytest tests/ -q
```

Bộ kiểm thử gồm **120 test** chạy trong khoảng 6 giây, tập trung vào ba tính chất
mà nếu sai thì mọi con số đánh giá phía sau đều vô nghĩa:

| Nhóm | Kiểm tra điều gì |
| --- | --- |
| **Tính nhân quả** | Cắt chuỗi tại một vị trí rồi tính lại chỉ báo trên đoạn đầu; giá trị trước điểm cắt phải không đổi. Vi phạm = rò rỉ dữ liệu tương lai. |
| **Đúng công thức** | Đối chiếu Gini, Entropy, Precision/Recall/F1, ROC-AUC với giá trị tính tay trên ví dụ nhỏ. |
| **Đúng lý thuyết** | Tỷ lệ mẫu ngoài túi hội tụ về `1/e`; R² của dự đoán bằng trung bình đúng bằng 0; cây hồi quy không ngoại suy được. |
| **Tái lập được** | Cùng `random_state` cho cùng kết quả; đổi hạt giống cho rừng khác. |

### 9.5. Đổi sang mã cổ phiếu hoặc đề tài khác

Không cần sửa một dòng mã nào — chỉ sửa `config/*.json`:

| Muốn đổi | Sửa khoá |
| --- | --- |
| Nguồn dữ liệu | `dataset.path`, `dataset.label` |
| Tên cột trong file | `dataset.series` (ánh xạ vai trò → tên cột thật) |
| Bộ chỉ báo và cửa sổ | `features` (danh sách bản đặc tả) |
| Tầm nhìn dự báo | `labeling.horizon` |
| Tỷ lệ tách tập | `split` |
| Siêu tham số mô hình | `model` |

Một mục trong `features` có dạng:

```json
{
  "name":      "close_over_sma20",
  "indicator": "ratio_to_moving_average",
  "inputs":    ["close"],
  "params":    {"window": 20},
  "keep":      true
}
```

`inputs` tra lần lượt trong các dãy gốc rồi tới các dãy đã dựng ở những mục
**trước đó**, nên bản đặc tả có thể xếp tầng. Danh sách chỉ báo hợp lệ lấy bằng
`pipeline.featureBuilder.describe_registry()`.

---

## 10. QUY TRÌNH COMMIT — REVIEW — MERGE

### 10.1. Mô hình nhánh

```text
main ────●─────────────────●─────────────────●────────→   (luôn chạy được, có tag)
          \               /                 /
develop ───●───●───●───●─●───●───●───●───●─●──────────→   (tích hợp)
                \     /       \     /
feature/*        ●───●         ●───●                      (mỗi tính năng một nhánh)
```

| Nhánh             | Vai trò                                    | Quy tắc                                                                                                 |
| ------------------ | ------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `main`           | Phiên bản ổn định, luôn chạy được | **Không commit trực tiếp**; chỉ nhận merge từ `develop`; mỗi lần merge gắn tag `v0.x` |
| `develop`        | Nhánh tích hợp                           | Nhận PR từ các nhánh`feature/*`                                                                    |
| `feature/<tên>` | Một đơn vị công việc                  | Xóa sau khi merge                                                                                       |
| `fix/<tên>`     | Sửa lỗi                                   | Như trên                                                                                               |
| `docs/<tên>`    | Tài liệu                                  | Như trên                                                                                               |

**Ví dụ tên nhánh:**

```text
feature/data-loader-timeseries
feature/technical-indicators
feature/decision-tree-core
feature/random-forest-bagging
feature/classification-metrics
fix/leakage-in-label-shift
docs/algorithm-report
```

### 10.2. Quy ước commit (Conventional Commits)

```text
<type>(<scope>): <mô tả ngắn, thể mệnh lệnh, không dấu chấm cuối>

<phần thân — giải thích VÌ SAO thay đổi, không phải LÀM GÌ (tùy chọn)>

<footer — refs #issue (tùy chọn)>
```

| `type`     | Dùng khi                              | Ví dụ                                                            |
| ------------ | -------------------------------------- | ------------------------------------------------------------------ |
| `feat`     | Thêm chức năng mới                 | `feat(indicators): thêm RSI và MACD`                           |
| `fix`      | Sửa lỗi                              | `fix(splitter): bỏ shuffle khi tách tập theo thời gian`      |
| `refactor` | Sửa cấu trúc, không đổi hành vi | `refactor(tree): tách hàm tìm ngưỡng chia tối ưu`         |
| `perf`     | Cải thiện hiệu năng                | `perf(forest): sắp xếp trước danh sách ngưỡng ứng viên` |
| `test`     | Thêm/sửa kiểm thử                  | `test(metrics): kiểm thử F1 với lớp mất cân bằng`         |
| `docs`     | Tài liệu                             | `docs(readme): bổ sung quy trình đánh giá`                  |
| `chore`    | Cấu hình, phụ thuộc                | `chore: thêm scikit-learn vào requirements`                    |
| `style`    | Định dạng, không đổi logic       | `style: căn lại thụt lề trong rfMath`                       |

**Nguyên tắc:**

- Mỗi commit là **một thay đổi logic hoàn chỉnh** — không gộp nhiều việc khác nhau.
- Commit phải ở **trạng thái chạy được**; không commit code lỗi cú pháp.
- **Không commit dữ liệu, model nhị phân, `.venv/`, `__pycache__/`** (đã có `.gitignore`).
- Thông điệp viết bằng tiếng Việt có dấu, thống nhất toàn dự án.

Ví dụ một commit đầy đủ:

```text
fix(labeling): dịch nhãn về sau thay vì dịch đặc trưng về trước

Cách cũ dịch cột đặc trưng lùi một phiên khiến chỉ báo tại thời điểm t
chứa thông tin của phiên t+1 — rò rỉ dữ liệu, làm accuracy trên tập
validate tăng ảo từ 0.56 lên 0.81.

Nay giữ nguyên đặc trưng tại t và lấy nhãn từ Close(t+1), sau đó bỏ
dòng cuối cùng do không có nhãn.

refs #12
```

### 10.3. Pull Request và Review

**Mẫu mô tả PR:**

```markdown
## Mục đích
Nhánh này thêm bộ chỉ báo động lượng (RSI, ROC, Stochastic).

## Thay đổi
- `pipeline/technicalIndicators.py`: thêm 3 hàm chỉ báo + kiểm tra tham số đầu vào
- `notebooks/02_features.ipynb`: đối chiếu kết quả với TradingView

## Cách kiểm tra
```powershell
python -m pytest tests/test_indicators.py
```

## Ảnh hưởng
Số đặc trưng tăng từ 18 → 24. Accuracy trên validate: 0.561 → 0.583.

## Checklist
- [ ] Code chạy được, không lỗi cú pháp
- [ ] Không có rò rỉ dữ liệu (chỉ báo tại t chỉ dùng dữ liệu ≤ t)
- [ ] Đã cập nhật tài liệu liên quan
- [ ] Không commit dữ liệu / file tạm
```

**Checklist cho người review:**

| Nhóm                          | Câu hỏi kiểm tra                                                                                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Đúng đắn**         | Công thức chỉ báo có khớp định nghĩa chuẩn? Biên cửa sổ trượt có bị lệch một phiên?                                              |
| **Rò rỉ dữ liệu**    | Có dùng thông tin tương lai ở bất kỳ đâu? Có`shuffle` khi tách tập? Có ước lượng tham số chuẩn hóa trên toàn bộ dữ liệu? |
| **Phân tầng**          | `libraries/` có bị lẫn logic chứng khoán? `pipeline/` có tự đọc file thay vì đi qua `dataLoader`?                                  |
| **Khả năng tái lập** | `random_state` đã cố định? Chạy lại có ra đúng kết quả cũ?                                                                            |
| **Đọc hiểu**          | Tên hàm/biến rõ nghĩa? Docstring nêu đủ tham số và giá trị trả về?                                                                     |
| **Tài liệu**           | README / Report có cần cập nhật theo thay đổi này?                                                                                            |

**Quy tắc review:**

- PR nên **dưới khoảng 400 dòng thay đổi** để review được kỹ.
- Người review để lại nhận xét theo 3 mức: `[Bắt buộc]`, `[Nên sửa]`, `[Góp ý]`.
- Tác giả **không tự merge PR của mình** khi chưa có ít nhất **1 approve**.
- Mọi nhận xét `[Bắt buộc]` phải được xử lý hoặc phản biện rõ ràng trước khi merge.

### 10.4. Merge

| Tình huống                 | Chiến lược                        | Lý do                                                        |
| ---------------------------- | ------------------------------------ | ------------------------------------------------------------- |
| `feature/*` → `develop` | **Squash merge**               | Gộp thành một commit sạch, lịch sử`develop` dễ đọc |
| `develop` → `main`      | **Merge commit** (`--no-ff`) | Giữ lại mốc phát hành, dễ truy vết                     |
| Cập nhật nhánh feature    | **Rebase** lên `develop`    | Tránh commit merge rác trong nhánh làm việc              |

```powershell
# Luồng làm việc chuẩn cho một tính năng
git checkout develop
git pull origin develop
git checkout -b feature/technical-indicators

# ... viết code, commit theo từng bước logic ...
git add src/pipeline/technicalIndicators.py
git commit -m "feat(indicators): thêm SMA, EMA và Bollinger Bands"

# Đồng bộ trước khi mở PR
git fetch origin
git rebase origin/develop

git push -u origin feature/technical-indicators
# → Mở Pull Request trên GitHub, gán reviewer, chờ approve

# Sau khi được approve: squash merge trên giao diện GitHub, rồi dọn nhánh
git checkout develop
git pull origin develop
git branch -d feature/technical-indicators
```

### 10.5. Gắn thẻ phiên bản

| Tag      | Mốc hoàn thành                                     |
| -------- | ----------------------------------------------------- |
| `v0.1` | Nạp dữ liệu và tiền xử lý thời gian           |
| `v0.2` | Bộ chỉ báo phân tích kỹ thuật                  |
| `v0.3` | Decision Tree thuần Python                           |
| `v0.4` | Random Forest hoàn chỉnh (bagging, OOB, importance) |
| `v0.5` | Bộ chỉ số đánh giá và walk-forward validation  |
| `v1.0` | Hoàn thiện hai nhánh, có báo cáo kết quả      |

---

## 11. LỘ TRÌNH THỰC HIỆN

| Giai đoạn | Công việc | Trạng thái |
| --- | --- | --- |
| 0 | Khởi tạo repo, `.gitignore`, `requirements.txt`, môi trường `.venv` | [x] Xong |
| 0 | Tầng tiện ích: `dataLoader`, `metrics`, `plotStyle` | [x] Xong |
| 0 | Chuyên đề lý thuyết `reportAlgorithm.md` | [x] Xong |
| 1 | Thu thập dữ liệu, EDA (`01_eda.ipynb`) | [x] Xong |
| 2 | `pipeline/timePreprocess.py` — tiền xử lý thời gian | [x] Xong |
| 3 | `pipeline/technicalIndicators.py` — 20 chỉ báo | [x] Xong |
| 3 | `pipeline/featureBuilder.py` — thi hành bản đặc tả đặc trưng | [x] Xong |
| 4 | `pipeline/labeling.py`, `pipeline/splitter.py` | [x] Xong |
| 5 | `libraries/rfMath.py`, `libraries/decisionTree.py` | [x] Xong |
| 6 | `libraries/randomForest.py` — bagging, OOB, importance | [x] Xong |
| 6 | `libraries/gradientBoosting.py` — GB hồi quy và phân loại | [x] Xong |
| 7 | `utilities/metricsClassification.py` | [x] Xong |
| 8 | Huấn luyện + tinh chỉnh siêu tham số cho hai nhánh | [x] Xong |
| 9 | Đối chiếu scikit-learn và Gradient Boosting | [x] Xong |
| 10 | `libraries/rfPlot.py` + bộ đồ thị đầu ra | [x] Xong |
| 11 | `reportResult.md` — báo cáo kết quả thực nghiệm | [x] Xong |
| 12 | Bộ kiểm thử `tests/` — 120 test | [x] Xong |
| 13 | Backtest có tính phí giao dịch | [ ] Ngoài phạm vi |

---

## 12. HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN

### 12.1. Hạn chế đã biết

- **Random Forest không ngoại suy được**: mô hình chỉ dự đoán trong khoảng giá trị đã
  thấy khi huấn luyện. Đây là lý do nhánh hồi quy dùng đặc trưng dạng **tỷ lệ** thay vì
  giá tuyệt đối (`reportAlgorithm.md` §4.3).
- **Giả định tính dừng**: quan hệ giữa chỉ báo và xu hướng thay đổi theo pha thị trường;
  mô hình huấn luyện trên giai đoạn tăng giá có thể suy giảm mạnh trong giai đoạn giảm.
- **Bỏ qua yếu tố ngoài phân tích kỹ thuật**: tin tức, báo cáo tài chính, dữ liệu vĩ mô,
  dòng tiền khối ngoại — đều không có trong tập đặc trưng.
- **Chưa tính chi phí giao dịch**: Accuracy cao chưa đồng nghĩa với chiến lược sinh lời
  sau phí và trượt giá.

### 12.2. Hướng phát triển

- Bổ sung **Gradient Boosting / XGBoost** và so sánh đầy đủ hai họ ensemble.
- **Nhãn ba lớp** (`tăng / đi ngang / giảm`) với ngưỡng theo `ATR` — thực tế hơn nhãn nhị phân.
- **Backtest** chiến lược giao dịch dựa trên tín hiệu mô hình, có tính phí giao dịch.
- **Huấn luyện đa mã cổ phiếu** để tăng số mẫu và giảm rủi ro quá khớp theo một mã.
- Thêm **SHAP values** để diễn giải mô hình ở mức từng dự đoán.

---

## 13. TÀI LIỆU THAM KHẢO

1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
   Learning* (2nd ed.). Springer — Chương 15: Random Forests.
3. Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. NYIF.
4. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley
   — Chương 7: Cross-Validation in Finance.
5. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
6. [`src/.reports/reportAlgorithm.md`](src/.reports/reportAlgorithm.md) — Chuyên đề
   Random Forest của dự án này.

---

<div align="center">
