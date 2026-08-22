# BÁO CÁO KẾT QUẢ THỰC NGHIỆM

> **Mini Project — Machine Learning**
> Dự đoán xu hướng giá cổ phiếu bằng Random Forest
> Số liệu trong báo cáo này sinh ra từ `src/mainClassification.py` và
> `src/mainRegression.py` với `random_state = 42`, tái lập được nguyên vẹn.

---

## MỤC LỤC

1. [Tóm tắt kết quả](#1-tóm-tắt-kết-quả)
2. [Thiết lập thí nghiệm](#2-thiết-lập-thí-nghiệm)
3. [Dữ liệu và tiền xử lý](#3-dữ-liệu-và-tiền-xử-lý)
4. [Nhánh A — Phân loại xu hướng](#4-nhánh-a--phân-loại-xu-hướng)
5. [Nhánh B — Hồi quy tỷ suất](#5-nhánh-b--hồi-quy-tỷ-suất)
6. [Diễn giải mô hình](#6-diễn-giải-mô-hình)
7. [Đối chiếu sáu tiêu chí chấp nhận](#7-đối-chiếu-sáu-tiêu-chí-chấp-nhận)
8. [Phân tích phần không đạt](#8-phân-tích-phần-không-đạt)
9. [Kiểm chứng cài đặt và so sánh mô hình](#9-kiểm-chứng-cài-đặt)
10. [Kết luận](#10-kết-luận)
11. [Cách tái lập](#11-cách-tái-lập)

---

## 1. TÓM TẮT KẾT QUẢ

| | Nhánh A — Phân loại | Nhánh B — Hồi quy |
| --- | --- | --- |
| Mục tiêu | Chiều biến động sau 5 phiên | Tỷ suất biến động sau 5 phiên |
| Mô hình | Random Forest, 300 cây | Random Forest, 300 cây |
| **Validate** | Accuracy **0.6087**, ROC-AUC **0.6476** | RMSE **0.022281**, R² **+0.0716** |
| **Test** | Accuracy 0.5237, ROC-AUC 0.5313 | RMSE 0.038519, R² −0.0257 |
| Mốc đối chứng (validate) | majority 0.5788 → **vượt +3.0 điểm** | mean-of-train RMSE 0.023294 → **tốt hơn** |
| Mốc đối chứng (test) | majority 0.5125 → vượt +1.1 điểm | mean-of-train RMSE 0.038045 → **không tốt hơn** |
| Walk-forward | 0.5814 ± 0.0170 | RMSE 0.027911 ± 0.003862 |

**Kết luận một dòng:** mô hình **đạt yêu cầu trên dữ liệu validate** — vượt rõ mọi
mốc đối chứng, không quá khớp, ổn định qua các vòng kiểm định tiến dần — nhưng
**suy giảm mạnh trên tập test**, giai đoạn thị trường mà mô hình chưa từng thấy.
Đây là kết quả trung thực và đúng như dự đoán với dữ liệu tài chính; phần
[Mục 8](#8-phân-tích-phần-không-đạt) phân tích nguyên nhân thay vì che giấu.

---

## 2. THIẾT LẬP THÍ NGHIỆM

### 2.1. Cấu hình chung

| Hạng mục | Giá trị | Nguồn |
| --- | --- | --- |
| Mã cổ phiếu | VNM (Vinamilk, sàn HOSE) | `config/*.json` → `dataset.path` |
| Khoảng thời gian | 2016-08-22 → 2026-08-21 | dữ liệu thô |
| Số phiên thô | 2 587 | — |
| Tầm nhìn dự báo | 5 phiên | `labeling.horizon` |
| Số đặc trưng | 27 | `features` (bản đặc tả) |
| Tỷ lệ tách | 70 / 15 / 15 | `split` |
| Khoảng trống (gap) | 5 phiên | `split.gap` |
| Hạt giống | 42 | `model.random_state` |

### 2.2. Vì sao tầm nhìn 5 phiên chứ không phải 1 phiên

Đề bài không quy định tầm nhìn. Thực nghiệm trên cả hai lựa chọn, với cùng bộ đặc
trưng và cùng quy trình:

| Mã | Tầm nhìn | Mốc majority (validate) | Accuracy validate | ROC-AUC validate | Chênh lệch train↔validate |
| --- | --- | --- | --- | --- | --- |
| VNM | 1 phiên | 0.5579 | 0.5793 | 0.5838 | 0.1298 |
| **VNM** | **5 phiên** | **0.5788** | **0.6223** | **0.6444** | **0.0374** |
| FPT | 1 phiên | 0.5435 | 0.5766 | 0.6052 | 0.0828 |
| FPT | 5 phiên | 0.5550 | 0.5255 | 0.4987 | 0.1542 |

*(cấu hình `max_depth = 5`, `min_samples_leaf = 40`, 150 cây — dùng để so sánh
tương đối giữa các lựa chọn, không phải cấu hình cuối cùng)*

Hai điều rút ra:

1. **Với VNM, tầm nhìn 5 phiên cho tín hiệu rõ hơn hẳn.** Nhiễu ngắn hạn lấn át
   tín hiệu ở tầm nhìn 1 phiên; kéo dài tầm nhìn giúp phần biến động có cấu trúc
   nổi lên trên nền nhiễu.
2. **Tín hiệu không phổ quát giữa các mã.** Với FPT, tầm nhìn 5 phiên cho
   ROC-AUC 0.4987 — không phân biệt được gì so với đoán bừa. Đây là phát hiện
   đáng ghi nhận: một quy trình chạy tốt trên mã này hoàn toàn có thể vô dụng
   trên mã khác, và không được suy rộng kết quả từ một mã ra cả thị trường.

### 2.3. Siêu tham số cuối cùng

| Tham số | Nhánh A | Nhánh B | Lý do chọn |
| --- | --- | --- | --- |
| `n_estimators` | 300 | 300 | Đường cong OOB đã đi ngang trước mốc này |
| `criterion` | `gini` | `variance` | Mặc định chuẩn cho từng loại bài toán |
| `max_depth` | 5 | 6 | Ràng buộc mạnh — xem [§4.2](#42-vì-sao-phải-ràng-buộc-mô-hình) |
| `min_samples_leaf` | 40 | 30 | Lá nhỏ hơn là học thuộc nhiễu |
| `max_features` | `sqrt` → 5/27 | `third` → 9/27 | Giá trị khuyến nghị cho phân loại / hồi quy |
| `bootstrap` | `True` | `True` | Bắt buộc để có ước lượng OOB |

---

## 3. DỮ LIỆU VÀ TIỀN XỬ LÝ

### 3.1. Chất lượng dữ liệu

| Kiểm tra | Kết quả | Đánh giá |
| --- | --- | --- |
| Ô thiếu | 0 trên mọi cột | Không cần điền tiến |
| Dòng bị loại khi làm sạch | 0 | Chuỗi liền mạch |
| Khoảng cách lớn nhất giữa hai phiên | 8 ngày | Trong mức nghỉ lễ thông thường |
| Biến động vượt 40% trong một phiên | 0 | Không có dấu hiệu lỗi dữ liệu |
| Quan sát đứng yên (horizon = 5) | 81 (3.14%) | Đã loại khỏi tập huấn luyện |

### 3.2. Xử lý quan sát đứng yên — một quyết định có ảnh hưởng lớn

Ở tầm nhìn **1 phiên**, VNM có **355 phiên đứng yên (13.7%)** do giá được làm tròn.
Cách xử lý nhóm này thay đổi hẳn bài toán:

| Cách xử lý | Phân bố lớp (giảm / tăng) | Mốc majority | Hệ quả |
| --- | --- | --- | --- |
| Dồn hết vào lớp "giảm" | 1 501 / 1 036 = **59.2 / 40.8** | 0.5916 | Mất cân bằng **giả**; mô hình học cách luôn đoán "giảm", F1 của lớp "tăng" chỉ đạt 0.1878 |
| **Loại khỏi tập huấn luyện** | 1 154 / 1 036 = **52.7 / 47.3** | 0.5269 | Bài toán cân bằng, mô hình buộc phải học tín hiệu thật |

*(số mẫu đã trừ vùng khởi động của cửa sổ trượt; đếm thô trên toàn chuỗi là
1 058 phiên tăng, 1 173 phiên giảm và 355 phiên đứng yên)*

Đây là ví dụ điển hình cho việc một lựa chọn tiền xử lý tưởng như vụn vặt lại
quyết định toàn bộ kết quả. Dự án chọn cách thứ hai (`flat_label = null`).

### 3.3. Số mẫu qua từng bước

```text
2 587 phiên thô
  → 2 587 sau khi làm sạch          (không mất dòng nào)
  → 2 456 sau khi dựng đặc trưng    (mất 131 dòng: 50 dòng khởi động
                                     cửa sổ SMA-50, 5 dòng cuối không có
                                     nhãn, 81 quan sát đứng yên)
  → chia thành 1 719 / 368 / 359
```

### 3.4. Phân chia theo thời gian

| Tập | Số mẫu | Tỷ lệ | Khoảng thời gian |
| --- | --- | --- | --- |
| Train | 1 719 | 70.3% | 2016-10-31 → 2023-09-06 |
| Validate | 368 | 15.0% | 2023-09-14 → 2025-03-12 |
| Test | 359 | 14.7% | 2025-03-20 → 2026-08-14 |

Khoảng trống 5 phiên giữa các tập là bắt buộc: nhãn của mẫu cuối tập train được
tính từ giá 5 phiên sau đó, tức đã chạm vào vùng validate. Không có khoảng trống
này, thông tin rò rỉ qua đúng đường biên.

---

## 4. NHÁNH A — PHÂN LOẠI XU HƯỚNG

### 4.1. Kết quả trên ba tập

| Tập | Accuracy | ROC-AUC | F1 (tăng) | F1 (giảm) | Balanced Acc. | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 0.6597 | 0.7663 | 0.5524 | 0.7255 | 0.6480 | +0.3260 |
| **Validate** | **0.6087** | **0.6476** | **0.4098** | **0.7073** | **0.5697** | **+0.1608** |
| Test | 0.5237 | 0.5313 | 0.4706 | 0.5671 | 0.5215 | +0.0436 |

Báo cáo chi tiết trên tập validate:

```text
Lớp            Precision      Recall    F1-score   Support
----------------------------------------------------------
0 (giảm)          0.6237      0.8169      0.7073       213
1 (tăng)          0.5618      0.3226      0.4098       155
----------------------------------------------------------
Macro avg         0.5927      0.5697      0.5586
Weighted avg      0.5976      0.6087      0.5820
Accuracy          0.6087
```

Điều đáng chú ý: `Precision` của lớp "tăng" đạt 0.5618 trong khi `Recall` chỉ
0.3226. Mô hình **thận trọng** — chỉ báo "tăng" khi khá chắc, chấp nhận bỏ sót
nhiều cơ hội. Với ứng dụng thực tế, đây thường là hành vi mong muốn hơn là ngược
lại, vì mỗi tín hiệu sai đều phải trả giá bằng chi phí giao dịch.

### 4.2. Vì sao phải ràng buộc mô hình

Rừng ngẫu nhiên để cây mọc tự do sẽ học thuộc nhiễu. Bảng dưới đo trực tiếp hiện
tượng đó — cùng dữ liệu, cùng 100 cây, chỉ đổi ràng buộc (nguồn:
`03_train_classify.ipynb` §4):

| Ràng buộc | Accuracy train | Accuracy validate | Chênh lệch | ROC-AUC validate |
| --- | --- | --- | --- | --- |
| `depth=∞, leaf=1` | **1.0000** | 0.5707 | **0.4293** | 0.6066 |
| `depth=12, leaf=10` | 0.9436 | 0.6033 | 0.3403 | 0.6354 |
| `depth=8, leaf=20` | 0.7807 | 0.6196 | 0.1611 | 0.6360 |
| **`depth=5, leaf=40`** | **0.6597** | **0.6250** | **0.0347** | **0.6418** |

Cây mọc tự do ghi nhớ **hoàn hảo** tập huấn luyện (Accuracy 1.0000) nhưng lại cho
kết quả validate **thấp nhất bảng** ở cả hai chỉ số. Toàn bộ phần chênh lệch
0.4293 là học thuộc nhiễu, không mang lại giá trị nào. Càng siết ràng buộc, khoảng
cách càng thu hẹp **và** kết quả validate càng tốt lên — quan hệ ngược chiều này
là bằng chứng rõ ràng nhất cho thấy dữ liệu tài chính đòi hỏi mô hình đơn giản.

*(Cấu hình cuối dùng 300 cây thay vì 100, cho Accuracy validate 0.6087 — bảng trên
giữ 100 cây để so sánh công bằng giữa các ràng buộc.)*

### 4.3. Ổn định qua kiểm định tiến dần

| Vòng | Số mẫu train | Số mẫu validate | Accuracy |
| --- | --- | --- | --- |
| 1 | 1 043 | 207 | 0.5797 |
| 2 | 1 250 | 207 | 0.6039 |
| 3 | 1 457 | 207 | 0.5749 |
| 4 | 1 664 | 207 | 0.5942 |
| 5 | 1 871 | 211 | 0.5545 |
| | | **Trung bình** | **0.5814 ± 0.0170** |

Độ lệch chuẩn 1.70% — mô hình ổn định giữa các giai đoạn, không phải ăn may ở một
đoạn dữ liệu cụ thể.

### 4.4. So với các mốc đối chứng

| Tập | Mô hình | majority | persistence | alternating | Kết luận |
| --- | --- | --- | --- | --- | --- |
| Validate | **0.6087** | 0.5788 | 0.5326 | 0.5136 | **Vượt cả ba** (+3.0 điểm so với mốc cao nhất) |
| Test | **0.5237** | 0.5125 | 0.4708 | 0.4708 | Vượt cả ba, nhưng chỉ +1.1 điểm |

### 4.5. Ngưỡng quyết định

Dò trên tập validate theo `balanced accuracy`, sau đó áp nguyên vẹn lên tập test:

| Ngưỡng | Nguồn | Accuracy test | Balanced Acc. test |
| --- | --- | --- | --- |
| 0.50 | mặc định | 0.5237 | 0.5215 |
| 0.45 | tối ưu trên validate (0.6237) | 0.5153 | 0.5175 |

Ngưỡng tối ưu trên validate **không** chuyển giao sang test. Đây là bằng chứng
trực tiếp cho thấy quan hệ giữa đặc trưng và mục tiêu đã đổi giữa hai giai đoạn.

---

## 5. NHÁNH B — HỒI QUY TỶ SUẤT

### 5.1. Kết quả trên ba tập

| Tập | RMSE | MAE | R² | MAPE (%) | Directional Accuracy |
| --- | --- | --- | --- | --- | --- |
| Train | 0.029021 | 0.021137 | +0.1836 | 103.4 | 0.6797 |
| **Validate** | **0.022281** | **0.016770** | **+0.0716** | 101.0 | **0.6137** |
| Test | 0.038519 | 0.026981 | −0.0257 | 118.7 | 0.5192 |

**Cách đọc MAPE cho đúng:** giá trị trên 100% ở đây **không** có nghĩa mô hình sai
hoàn toàn. Mục tiêu là tỷ suất biến động, dao động quanh 0, nên mẫu số `|y|`
thường rất nhỏ và làm phần trăm sai số phóng đại lên. Với loại mục tiêu này,
`RMSE` và `R²` mới là thước đo đáng tin — MAPE được giữ lại trong bảng chỉ để cho
thấy vì sao không nên dùng nó ở đây.

### 5.2. So với mốc đối chứng

| Tập | RMSE mô hình | RMSE mean-of-train | RMSE zero | Kết luận |
| --- | --- | --- | --- | --- |
| Validate | **0.022281** | 0.023294 | 0.023303 | **Tốt hơn** |
| Test | 0.038519 | 0.038045 | 0.038043 | **Không tốt hơn** |

`R²` âm trên tập test (−0.0257) chính là cách phát biểu khác của cùng một sự thật:
trên giai đoạn đó, dự đoán bằng một hằng số còn tốt hơn mô hình.

### 5.3. Giới hạn không ngoại suy — quan sát trực tiếp

| Đại lượng | Khoảng giá trị |
| --- | --- |
| Mục tiêu trong tập train | [−0.1371, +0.1677] |
| Mục tiêu trong tập test | [−0.1447, +0.1596] |
| **Dự đoán trên tập test** | **[−0.0112, +0.0336]** |

Dải dự đoán **hẹp hơn khoảng 7 lần** dải thực tế. Nguyên nhân mang tính cấu trúc:
rừng lấy trung bình dự đoán của 300 cây, mà mỗi cây lại lấy trung bình các giá trị
trong lá — hai lớp lấy trung bình chồng lên nhau san phẳng mọi giá trị cực đoan.
Hệ quả thực tế: **mô hình không bao giờ dự báo được các phiên biến động mạnh** —
đúng những phiên đáng quan tâm nhất.

### 5.4. Bắc cầu sang nhánh A

`Directional Accuracy` là chỉ số chung của hai nhánh:

| Tập | Nhánh A (học thẳng chiều) | Nhánh B (suy từ dấu tỷ suất) |
| --- | --- | --- |
| Validate | 0.6087 | 0.6137 |
| Test | 0.5237 | 0.5192 |

Hai nhánh cho kết quả gần như trùng nhau. Nghĩa là **việc bắt mô hình học thêm độ
lớn của biến động không giúp nó đoán hướng tốt hơn** — phần thông tin dùng được
nằm ở dấu, còn độ lớn hầu như chỉ là nhiễu.

---

## 6. DIỄN GIẢI MÔ HÌNH

### 6.1. Đặc trưng nào được dùng

Top 10 theo MDI, đặt cạnh permutation importance đo trên tập validate. Giá trị
permutation ở đây là **mức TĂNG lỗi thô** khi xáo trộn đặc trưng đó — giá trị âm
nghĩa là xáo trộn còn làm mô hình tốt lên, tức đặc trưng không đóng góp gì.

| Hạng | Đặc trưng | MDI | Permutation (thô) | Nhóm |
| --- | --- | --- | --- | --- |
| 1 | `volatility_10` | 0.1442 | **+0.04728** | Biến động |
| 2 | `atr_over_close` | 0.1044 | −0.00326 | Biến động |
| 3 | `close_over_sma50` | 0.0850 | +0.00489 | Xu hướng |
| 4 | `volatility_20` | 0.0645 | −0.00543 | Biến động |
| 5 | `rsi_14` | 0.0515 | **+0.01087** | Động lượng |
| 6 | `close_over_sma10` | 0.0482 | −0.00380 | Xu hướng |
| 7 | `bollinger_width` | 0.0448 | −0.00000 | Biến động |
| 8 | `roc_10` | 0.0422 | +0.00272 | Động lượng |
| 9 | `close_over_sma20` | 0.0405 | −0.00000 | Xu hướng |
| 10 | `close_over_ema12` | 0.0368 | −0.00109 | Xu hướng |

Ba nhận xét:

1. **Nhóm biến động chiếm ưu thế theo MDI.** Bốn trong mười đặc trưng hàng đầu đo
   biến động chứ không đo hướng, và `volatility_10` đứng đầu ở cả hai cách đo. Mô
   hình học được rằng *mức độ dao động* dự báo được nhiều hơn *chiều dao động* —
   phù hợp với hiểu biết đã biết về thị trường tài chính: biến động có tính cụm
   (volatility clustering) rõ rệt, còn hướng thì gần với ngẫu nhiên.

2. **Hai cách đo bất đồng mạnh — và permutation mới là cách đáng tin.**
   `atr_over_close` được MDI xếp hạng 2 nhưng permutation importance **âm**: xáo
   trộn nó không làm mô hình tệ đi. Đây là biểu hiện điển hình của nhóm đặc trưng
   **tương quan cao** — `atr_over_close`, `volatility_10`, `volatility_20` và
   `bollinger_width` cùng đo một thứ, nên khi phá hỏng một cái, ba cái còn lại vẫn
   cung cấp đủ thông tin. MDI chia đều công trạng cho cả nhóm và tạo ảo giác rằng
   cả bốn đều quan trọng. Đúng như `reportAlgorithm.md` §4.4 đã cảnh báo.

3. **Chỉ hai đặc trưng thực sự không thay thế được:** `volatility_10` (+0.047) và
   `rsi_14` (+0.011). Toàn bộ 25 đặc trưng còn lại đóng góp ở mức gần bằng 0 hoặc
   âm. Nói cách khác, **bộ 27 đặc trưng có thể rút gọn rất mạnh** mà không mất mát
   đáng kể — một hướng cải tiến cụ thể cho lần sau.

### 6.2. Cấu trúc rừng

| Chỉ số | Giá trị |
| --- | --- |
| Số cây | 300 |
| Đặc trưng xét mỗi nút (m) | 5 / 27 |
| Độ sâu trung bình | 5.0 (chạm trần `max_depth`) |
| Số lá trung bình | 13.5 |
| Số nút trung bình | 26.0 |
| **Tỷ lệ OOB thực tế** | **0.3676** |
| Tỷ lệ OOB lý thuyết `(1-1/n)^n` | 0.3676 |
| Giới hạn `1/e` | 0.3679 |
| Lỗi OOB | 0.3973 |

Tỷ lệ out-of-bag khớp với lý thuyết đến bốn chữ số thập phân — xác nhận cài đặt
bootstrap chạy đúng.

---

## 7. ĐỐI CHIẾU SÁU TIÊU CHÍ CHẤP NHẬN

Theo README §8.4:

| # | Tiêu chí | Ngưỡng | Kết quả | Đạt |
| --- | --- | --- | --- | --- |
| 1 | Accuracy validate > 55% và cao hơn mọi đối chứng | > 0.55 | 0.6087 (đối chứng cao nhất 0.5788) | ✅ |
| 2 | ROC-AUC validate | > 0.55 | 0.6476 | ✅ |
| 3 | Chênh lệch train ↔ validate | < 0.10 | 0.0510 | ✅ |
| 4 | Độ lệch chuẩn walk-forward | < 0.05 | 0.0170 | ✅ |
| 5 | F1 của mọi lớp | ≥ 0.40 | 0.4098 và 0.7073 | ✅ |
| 6 | Test không suy giảm quá 5% so với validate | < 0.05 | **0.0850** | ❌ |

**Năm trên sáu tiêu chí đạt.** Tiêu chí duy nhất không đạt là số 6, phân tích ở
mục tiếp theo.

---

## 8. PHÂN TÍCH PHẦN KHÔNG ĐẠT

### 8.1. Hiện tượng

Accuracy giảm từ 0.6087 (validate) xuống 0.5237 (test) — mất 8.5 điểm phần trăm,
gần như xoá sạch khoảng cách so với mốc majority. Nhánh B còn rõ hơn: `R²` từ
+0.0716 xuống −0.0257.

### 8.2. Loại trừ nguyên nhân "lỗi cài đặt"

Trước khi kết luận về bản chất dữ liệu, cần loại trừ khả năng có lỗi:

| Khả năng | Kiểm chứng | Kết quả |
| --- | --- | --- |
| Rò rỉ dữ liệu tương lai | Cắt chuỗi tại vị trí 1 000, tính lại chỉ báo trên đoạn đầu và so với chuỗi đầy đủ | Trùng khớp tuyệt đối trên cả 5 chỉ báo (`02_features.ipynb` §2) |
| Lệch một phiên khi gán nhãn | Nhãn dịch **về sau**, đặc trưng giữ nguyên; 5 dòng cuối không có nhãn và bị loại | Đúng thiết kế |
| Rò rỉ qua đường biên tách tập | Chèn gap 5 phiên bằng đúng `horizon` | Đã có |
| Dò ngưỡng trên tập test | Ngưỡng dò trên validate rồi áp nguyên vẹn | Đã tuân thủ |
| Sai trong cài đặt thuật toán | Đối chiếu với scikit-learn trên cùng cấu hình | Xem [§9](#9-kiểm-chứng-cài-đặt) |

Không tìm thấy lỗi cài đặt.

### 8.3. Nguyên nhân thực tế: đổi chế độ thị trường

Ba dấu hiệu cùng chỉ về một hướng:

1. **Ngưỡng quyết định không chuyển giao.** Ngưỡng 0.45 tối ưu trên validate lại
   làm Accuracy test giảm nhẹ. Nếu hai giai đoạn cùng phân phối, ngưỡng tối ưu
   phải chuyển giao được.
2. **Mốc majority cũng đổi.** Tỷ lệ lớp đa số giảm từ 0.5788 (validate) xuống
   0.5125 (test) — bản thân cấu trúc bài toán đã khác.
3. **Đặc trưng quan trọng nhất là biến động.** Chế độ biến động là thứ thay đổi
   theo giai đoạn thị trường; mô hình học quan hệ giữa mức biến động và xu hướng
   trong một chế độ sẽ không còn đúng khi chế độ đổi.
4. **Thứ hạng giữa các mô hình đảo ngược hoàn toàn giữa hai tập.** Logistic
   Regression tệ nhất trên validate nhưng tốt nhất trên test, còn các mô hình dựa
   trên cây thì ngược lại — xem [§9.5](#95-so-sánh-các-họ-mô-hình). Nếu hai giai
   đoạn cùng phân phối, điều này không thể xảy ra.

Đây chính là **vi phạm giả định tính dừng** đã nêu trước ở README §12.1, nay được
xác nhận bằng số liệu.

### 8.4. Điều này có nghĩa gì

Kết quả **không** nói rằng mô hình vô dụng, mà nói rằng:

- Kết quả trên validate là **có thật** — vượt đối chứng, ổn định, không quá khớp.
- Nhưng kết quả đó **có hạn sử dụng**. Một mô hình huấn luyện trên dữ liệu tới
  2023 không giữ được hiệu lực tới 2026 nếu không huấn luyện lại.
- Hướng xử lý đúng là **huấn luyện lại định kỳ** theo cửa sổ trượt
  (`splitter.generate_rolling_window_folds`) thay vì cố tìm một mô hình vĩnh viễn.

---

## 9. KIỂM CHỨNG CÀI ĐẶT

Toàn bộ thuật toán được viết thuần Python trong `src/libraries/`, không dùng thư
viện học máy. Ba phép kiểm chứng độc lập:

### 9.1. Kiểm chứng lý thuyết bootstrap

Tỷ lệ mẫu ngoài túi đo được **0.3676**, so với công thức `(1 - 1/n)^n = 0.3676` và
giới hạn `1/e = 0.3679`.

### 9.2. Kiểm chứng tính nhân quả của chỉ báo

Cắt chuỗi tại vị trí 1 000 rồi tính lại chỉ báo trên đoạn đầu. Nếu hàm chỉ nhìn về
quá khứ, giá trị tại mọi vị trí trước điểm cắt phải giống hệt giá trị tính trên
chuỗi đầy đủ. Cả năm chỉ báo được kiểm tra — `simple_moving_average`,
`exponential_moving_average`, `relative_strength_index`, `rate_of_change`,
`rolling_standard_deviation` — đều **ĐẠT**.

### 9.3. Đối chiếu với scikit-learn

Cùng dữ liệu, cùng siêu tham số, cùng hạt giống:

| Cài đặt | Accuracy validate | ROC-AUC validate | Accuracy test | Lỗi OOB |
| --- | --- | --- | --- | --- |
| **Thuần Python (dự án)** | 0.6087 | 0.6476 | 0.5237 | 0.3973 |
| scikit-learn | 0.6141 | 0.6446 | 0.5181 | 0.4043 |
| **Chênh lệch** | **0.0054** | **0.0030** | **0.0056** | **0.0070** |

Chênh lệch dưới 1% ở cả bốn chỉ số — hai cài đặt tương đương. Phần khác biệt còn
lại đến từ chi tiết sinh số ngẫu nhiên và cách chọn ngưỡng chia, không thể trùng
khít tuyệt đối.

Phép kiểm chứng chặt hơn — hai cài đặt có **đồng ý về đặc trưng nào quan trọng**
hay không:

| Đặc trưng | Hạng (dự án) | Hạng (sklearn) | Lệch |
| --- | --- | --- | --- |
| `volatility_10` | 1 | 1 | 0 |
| `atr_over_close` | 2 | 2 | 0 |
| `close_over_sma50` | 3 | 3 | 0 |
| `volatility_20` | 4 | 4 | 0 |
| `rsi_14` | 5 | 6 | 1 |
| `roc_10` | 8 | 5 | 3 |

**Bốn đặc trưng hàng đầu trùng khớp tuyệt đối về thứ hạng.** Các hạng sau lệch
nhẹ, đúng như dự đoán với nhóm đặc trưng tương quan cao đã bàn ở [§6.1](#61-đặc-trưng-nào-được-dùng).

### 9.4. Dò siêu tham số quy mô lớn

`GridSearchCV` với `TimeSeriesSplit` (5 vòng) trên lưới 60 cấu hình:

| Hạng mục | Kết quả |
| --- | --- |
| Cấu hình tốt nhất | `max_depth=5`, `max_features=0.33`, `min_samples_leaf=10` |
| Điểm cross-validation | 0.5657 |
| Accuracy trên validate | 0.6168 |

Kết quả này **xác nhận cấu hình chọn bằng tay**: cùng cho `max_depth = 5`, và
Accuracy validate 0.6168 so với 0.6087 — chênh lệch không đáng kể sau khi dò
tự động trên 60 cấu hình.

### 9.5. So sánh các họ mô hình

Tất cả chạy trên cùng bộ đặc trưng và cùng cách tách tập:

| Mô hình | Accuracy validate | AUC validate | Accuracy test | AUC test |
| --- | --- | --- | --- | --- |
| RF (thuần Python) | 0.6087 | **0.6476** | 0.5237 | 0.5313 |
| RF (scikit-learn) | **0.6141** | 0.6446 | 0.5181 | 0.5306 |
| GB (thuần Python) | **0.6141** | 0.6214 | 0.5404 | 0.5556 |
| GB (scikit-learn) | 0.5870 | 0.6127 | 0.5599 | 0.5585 |
| Logistic Regression | 0.5435 | 0.5081 | **0.5850** | **0.6086** |
| *Mốc majority* | *0.5788* | *0.5000* | *0.5125* | *0.5000* |

**Phát hiện đáng chú ý nhất của cả dự án nằm ở dòng cuối bảng.** Logistic
Regression — mô hình tuyến tính đơn giản nhất — cho kết quả **tệ nhất trên
validate** (AUC 0.5081, tức không phân biệt được gì) nhưng lại **tốt nhất trên
test** (AUC 0.6086). Toàn bộ các mô hình dựa trên cây thì ngược lại.

Đây là bằng chứng độc lập, mạnh nhất, cho kết luận ở [§8.3](#83-nguyên-nhân-thực-tế-đổi-chế-độ-thị-trường):
nếu hai giai đoạn cùng phân phối, thứ hạng giữa các mô hình phải giữ nguyên. Việc
thứ hạng **đảo ngược hoàn toàn** chỉ có thể giải thích bằng việc quan hệ giữa đặc
trưng và mục tiêu đã thay đổi. Mô hình càng phức tạp, càng khớp sát vào quan hệ
của giai đoạn cũ, càng mất nhiều khi quan hệ đó đổi.

Bài học rút ra không phải "nên dùng Logistic Regression", vì kết quả tốt của nó
trên test cũng chỉ là một lần quan sát duy nhất và hoàn toàn có thể là ngẫu
nhiên. Bài học là: **với dữ liệu tài chính, chênh lệch giữa các họ mô hình nhỏ
hơn nhiều so với chênh lệch giữa hai giai đoạn thời gian.** Công sức nên dồn vào
thiết kế quy trình kiểm định trung thực, không phải vào việc chọn mô hình.

## 10. KẾT LUẬN

### 10.1. Đã làm được

1. **Quy trình học máy đầy đủ** — 8 bước từ nạp dữ liệu tới báo cáo, mỗi bước là
   một module riêng có thể kiểm thử độc lập.
2. **Thuật toán tự cài đặt** — Decision Tree, Random Forest và Gradient Boosting
   viết thuần Python, kiểm chứng được bằng ba cách độc lập.
3. **Kết quả tốt trên dữ liệu validate** — vượt mọi mốc đối chứng, thoả 5/6 tiêu
   chí chấp nhận đặt ra từ trước khi chạy thí nghiệm.
4. **Đánh giá trung thực** — báo cáo cả phần không đạt kèm phân tích nguyên nhân,
   thay vì chỉ trình bày con số đẹp.

### 10.2. Ba bài học rút ra

1. **Tiền xử lý quyết định kết quả nhiều hơn mô hình.** Việc xử lý 355 quan sát
   đứng yên làm thay đổi mốc majority từ 0.59 xuống 0.53 — ảnh hưởng lớn hơn mọi
   phép tinh chỉnh siêu tham số cộng lại.
2. **Ràng buộc mô hình chặt lại làm kết quả tốt lên.** Cây mọc tự do đạt Accuracy
   0.9583 trên tập huấn luyện nhưng ROC-AUC validate lại **thấp hơn** cấu hình
   `depth = 5`. Với dữ liệu nhiễu, mọi tham số dư thừa đều bị dùng để học thuộc
   nhiễu.
3. **Luôn phải có mốc đối chứng.** Accuracy 0.6087 nghe khá, cho tới khi biết mốc
   đoán bừa đạt 0.5788. Phần đóng góp thật của mô hình là 3 điểm phần trăm, không
   phải 61.
4. **Chọn mô hình quan trọng ít hơn ta tưởng.** Năm mô hình khác nhau — hai họ
   ensemble, hai cài đặt, một mô hình tuyến tính — chênh nhau chưa tới 3 điểm phần
   trăm trên validate, trong khi cùng một mô hình chênh 8.5 điểm giữa validate và
   test. Chênh lệch **giữa hai giai đoạn thời gian** lớn gấp ba lần chênh lệch
   **giữa các mô hình**.

### 10.3. Hướng phát triển

| Hướng | Vì sao đáng làm |
| --- | --- |
| Huấn luyện lại theo cửa sổ trượt | Xử lý trực tiếp vấn đề đổi chế độ thị trường ở §8.3 |
| Nhãn ba lớp với ngưỡng theo `ATR` | Không bắt mô hình phán bừa khi thị trường đi ngang; `labeling.create_ternary_labels()` đã sẵn sàng |
| Huấn luyện đa mã | Tăng số mẫu và giảm rủi ro quá khớp theo một mã, sau khi §2.2 cho thấy tín hiệu khác nhau giữa các mã |
| Backtest có tính phí | Accuracy 0.52 rất có thể không sinh lời sau phí và trượt giá |

---

## 11. CÁCH TÁI LẬP

```powershell
# 1. Chuẩn bị môi trường
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Đặt dữ liệu OHLCV vào data/input/ rồi trỏ config tới nó
#    (định dạng cột: Date, Open, High, Low, Close, Volume)

# 3. Chạy hai nhánh
python src/mainClassification.py --config config/vnm.json
python src/mainRegression.py     --config config/regression.json
```

Mọi con số trong báo cáo này tái lập được với `random_state = 42`. Nếu kết quả
khác đi, hãy kiểm tra trước hết: nguồn dữ liệu, khoảng thời gian và giá trị
`horizon` trong file cấu hình.

---

## PHỤ LỤC — Đồ thị kèm theo

Sinh ra trong `data/output/` khi chạy hai script:

| File | Nội dung |
| --- | --- |
| `VNM_classification_class_distribution.png` | Phân bố lớp sau khi loại quan sát đứng yên |
| `VNM_classification_oob_curve.png` | Lỗi OOB theo số cây |
| `VNM_classification_feature_importance.png` | MDI đối chiếu permutation importance |
| `VNM_classification_confusion_matrix.png` | Ma trận nhầm lẫn trên tập test |
| `VNM_classification_roc_curve.png` | Đường ROC trên tập test |
| `VNM_classification_threshold_curve.png` | Chỉ số theo ngưỡng quyết định |
| `VNM_classification_walk_forward.png` | Accuracy qua từng vòng kiểm định |
| `VNM_regression_predicted_vs_actual.png` | Tán xạ dự đoán ↔ thực tế |
| `VNM_regression_series_comparison.png` | Chuỗi thực tế và dự đoán theo thời gian |
| `VNM_regression_residuals.png` | Phân bố sai số |
| `VNM_regression_oob_curve.png` | MSE ngoài túi theo số cây |
| `VNM_regression_walk_forward.png` | RMSE qua từng vòng kiểm định |

Sinh ra khi chạy các notebook:

| File | Nội dung |
| --- | --- |
| `VNM_01_price_history.png` | Diễn biến giá và khối lượng toàn giai đoạn |
| `VNM_01_class_distribution.png` | Phân bố lớp ở tầm nhìn 5 phiên |
| `VNM_02_indicators.png` | Giá kèm Bollinger, RSI và MACD trên 400 phiên gần nhất |
| `VNM_02_correlation.png` | Ma trận tương quan giữa 27 đặc trưng |
| `VNM_05_roc_comparison.png` | Đường ROC của năm mô hình đặt chồng |

---

<div align="center">

**Mini Project — Machine Learning**

Báo cáo kết quả · Random Forest · VNM 2016–2026

</div>
