# BÁO CÁO THUẬT TOÁN: RANDOM FOREST (RỪNG NGẪU NHIÊN)

> **Tài liệu chuyên đề — Machine Learning**
> Phạm vi: khái niệm, cơ chế hoạt động, ưu/nhược điểm, ngữ cảnh áp dụng,
> hệ thống thuật ngữ, hệ thống công thức toán học và các ví dụ ứng dụng.

---

## MỤC LỤC

1. [Khái niệm](#1-khái-niệm)
2. [Cơ chế hoạt động](#2-cơ-chế-hoạt-động)
3. [Ưu điểm](#3-ưu-điểm)
4. [Nhược điểm](#4-nhược-điểm)
5. [Ngữ cảnh — Sử dụng khi nào](#5-ngữ-cảnh--sử-dụng-khi-nào)
6. [Thuật ngữ](#6-thuật-ngữ)
7. [Công thức](#7-công-thức)
8. [Áp dụng — Ví dụ minh họa](#8-áp-dụng--ví-dụ-minh-họa)
9. [Siêu tham số và hướng dẫn tinh chỉnh](#9-siêu-tham-số-và-hướng-dẫn-tinh-chỉnh)
10. [So sánh với các thuật toán khác](#10-so-sánh-với-các-thuật-toán-khác)
11. [Kết luận](#11-kết-luận)
12. [Tài liệu tham khảo](#12-tài-liệu-tham-khảo)

---

## 1. KHÁI NIỆM

### 1.1. Định nghĩa

**Random Forest** (Rừng Ngẫu Nhiên) là một **thuật toán học có giám sát**
(*supervised learning*) thuộc nhóm **học kết hợp** (*ensemble learning*), do
**Leo Breiman** công bố năm 2001. Thuật toán xây dựng một **tập hợp gồm nhiều
cây quyết định** (*decision trees*) được huấn luyện độc lập trên các tập dữ liệu
con ngẫu nhiên, sau đó **tổng hợp dự đoán** của toàn bộ các cây để đưa ra kết
quả cuối cùng:

- **Bài toán phân loại** (*classification*): lấy **đa số phiếu** (*majority
  voting*) — lớp nào được nhiều cây bầu chọn nhất sẽ là kết quả.
- **Bài toán hồi quy** (*regression*): lấy **trung bình cộng** (*averaging*)
  giá trị dự đoán của tất cả các cây.

Tư tưởng cốt lõi được Breiman đúc kết trong câu:

> *"Một tập hợp lớn các mô hình yếu, **thiếu tương quan với nhau**, khi kết hợp
> lại sẽ tạo thành một mô hình mạnh."*

### 1.2. Ba trụ cột nền tảng

Random Forest = **Cây quyết định** + **Bagging** + **Ngẫu nhiên hóa đặc trưng**.

| Trụ cột                                                  | Vai trò                                                    | Tác dụng                                                                    |
| ---------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Decision Tree** (Cây quyết định)               | Bộ học cơ sở (*base learner*)                         | Mô hình phi tuyến, dễ diễn giải, nhưng**phương sai rất cao**  |
| **Bagging** (Bootstrap Aggregating)                  | Mỗi cây học trên một mẫu bootstrap khác nhau         | **Giảm phương sai** thông qua trung bình hóa                      |
| **Random Subspace** (Ngẫu nhiên hóa đặc trưng) | Mỗi nút chỉ xét một tập con đặc trưng ngẫu nhiên | **Giảm tương quan** giữa các cây → tăng hiệu quả của bagging |

**Điểm mấu chốt phân biệt Random Forest với Bagging thuần túy:** Bagging chỉ
ngẫu nhiên hóa **dữ liệu** (dòng); Random Forest ngẫu nhiên hóa **cả dữ liệu
lẫn đặc trưng** (dòng + cột). Chính lớp ngẫu nhiên thứ hai này làm các cây
"khác nhau nhiều hơn", giảm hệ số tương quan $\rho$ trong công thức phương sai
tổ hợp (xem [mục 7.6](#76-công-thức-giảm-phương-sai--linh-hồn-của-random-forest)).

### 1.3. Trực giác: "Trí tuệ đám đông"

Hãy tưởng tượng một hội đồng chuyên gia chẩn đoán bệnh:

- Mỗi bác sĩ (một cây) chỉ được xem **một phần hồ sơ bệnh án** (mẫu bootstrap)
  và tại mỗi bước chỉ được hỏi về **một vài triệu chứng ngẫu nhiên** (tập con
  đặc trưng).
- Từng bác sĩ riêng lẻ có thể sai lệch, nhưng **sai lệch của họ không giống
  nhau** — chúng triệt tiêu lẫn nhau khi bỏ phiếu.
- Kết luận của cả hội đồng chính xác và ổn định hơn nhiều so với bất kỳ cá nhân
  nào.

Điều kiện để "đám đông thông minh" hoạt động: (1) mỗi thành viên **tốt hơn đoán
mò**, và (2) các thành viên **độc lập/ít tương quan**. Random Forest được thiết
kế để bảo đảm đồng thời cả hai điều kiện này.

### 1.4. Sơ đồ kiến trúc

```
                        Tập huấn luyện D (n mẫu, p đặc trưng)
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼
  Bootstrap D₁    Bootstrap D₂    Bootstrap D₃        ...       Bootstrap D_B
      n mẫu,         n mẫu,          n mẫu,                        n mẫu,
   có hoàn lại     có hoàn lại     có hoàn lại                   có hoàn lại
        │              │               │                              │
        ▼              ▼               ▼                              ▼
    ┌───────┐      ┌───────┐       ┌───────┐                      ┌───────┐
    │ Cây 1 │      │ Cây 2 │       │ Cây 3 │          ...         │ Cây B │
    └───────┘      └───────┘       └───────┘                      └───────┘
       (-----mỗi nút chỉ xét m ngẫu nhiên trong p đặc trưng, m << p----)
        │              │               │                              │
        ▼              ▼               ▼                              ▼
       ŷ₁             ŷ₂              ŷ₃                             ŷ_B
        └──────────────┴───────┬───────┴──────────────────────────────┘
                               ▼
                    TỔNG HỢP (Aggregation)
            • Phân loại → Majority Voting / TB xác suất
            • Hồi quy   → Trung bình cộng
                               ▼
                       Dự đoán cuối cùng  ŷ
```

---

## 2. CƠ CHẾ HOẠT ĐỘNG

### 2.1. Thuật toán huấn luyện (mã giả)

```text
THUẬT TOÁN: RandomForestTrain(D, B, m, n_min)
────────────────────────────────────────────────────────────────
ĐẦU VÀO:
    D  = {(x₁,y₁), ..., (xₙ,yₙ)}   — tập huấn luyện, xᵢ ∈ ℝᵖ
    B                              — số cây trong rừng
    m                              — số đặc trưng xét tại mỗi nút (m ≤ p)
    n_min                          — số mẫu tối thiểu tại một lá

ĐẦU RA:
    Rừng F = {T₁, T₂, ..., T_B}

BƯỚC THỰC HIỆN:
  1  FOR b = 1 TO B DO
  2      Dᵦ ← Lấy mẫu bootstrap: rút n mẫu từ D CÓ HOÀN LẠI
  3      OOBᵦ ← D \ Dᵦ              // các mẫu không lọt vào Dᵦ
  4      Tᵦ ← BuildTree(Dᵦ, m, n_min)
  5  END FOR
  6  RETURN F = {T₁, ..., T_B}

THỦ TỤC: BuildTree(node_data, m, n_min)
────────────────────────────────────────
  1  IF |node_data| < n_min  HOẶC  node_data thuần khiết  HOẶC
        đạt độ sâu tối đa THEN
  2      RETURN Lá(giá trị = mode(y) nếu phân loại,
                            mean(y) nếu hồi quy)
  3  END IF
  4  S ← chọn NGẪU NHIÊN m đặc trưng trong p đặc trưng (KHÔNG hoàn lại)
  5  (j*, s*) ← tối ưu trên S theo tiêu chí phân tách
                (Gini / Entropy cho phân loại, SSE cho hồi quy)
  6  Chia node_data thành:
         L = {x : x_{j*} ≤ s*}      R = {x : x_{j*} > s*}
  7  RETURN Nút(j*, s*,
                trái  = BuildTree(L, m, n_min),
                phải  = BuildTree(R, m, n_min))
```

**Lưu ý quan trọng ở dòng 4:** tập con đặc trưng $S$ được **rút lại tại MỖI
nút**, không phải một lần cho cả cây. Đây là chi tiết kỹ thuật quyết định sức
mạnh khử tương quan của thuật toán.

### 2.2. Thuật toán dự đoán

```text
THUẬT TOÁN: RandomForestPredict(F, x)
──────────────────────────────────────
  1  FOR b = 1 TO B DO
  2      ŷᵦ ← Tᵦ.predict(x)      // đi từ gốc xuống lá theo các điều kiện
  3  END FOR
  4  IF bài toán là PHÂN LOẠI THEN
  5      RETURN mode{ŷ₁, ..., ŷ_B}                     // đa số phiếu
         (hoặc argmax_c (1/B)·Σ P̂ᵦ(c|x) — TB xác suất, thường tốt hơn)
  6  ELSE  // HỒI QUY
  7      RETURN (1/B)·Σ ŷᵦ                             // trung bình cộng
  8  END IF
```

### 2.3. Hai nguồn ngẫu nhiên và ý nghĩa

| Nguồn ngẫu nhiên             | Cấp độ áp dụng | Mục tiêu                            | Hệ quả nếu bỏ đi                                                       |
| ------------------------------- | ------------------- | ------------------------------------- | --------------------------------------------------------------------------- |
| **Bootstrap sampling**    | Mỗi**cây**  | Đa dạng hóa dữ liệu huấn luyện | Tất cả cây học trên cùng D → gần như giống hệt nhau              |
| **Random feature subset** | Mỗi**nút**  | Đa dạng hóa cấu trúc cây        | Đặc trưng mạnh chiếm nút gốc mọi cây → các cây tương quan cao |

**Ví dụ minh họa nguồn ngẫu nhiên thứ hai:** giả sử trong bài toán dự đoán giá
nhà, `Diện tích` là đặc trưng mạnh áp đảo. Nếu mọi nút đều được xét toàn bộ $p$
đặc trưng, thì `Diện tích` sẽ là nút gốc của **cả 500 cây** → 500 cây gần như
sao chép nhau → trung bình hóa gần như không giảm được phương sai. Khi mỗi nút
chỉ xét $m = \sqrt{p}$ đặc trưng, xác suất `Diện tích` xuất hiện tại nút gốc chỉ
còn $m/p$, buộc các cây phải khai thác cả những đặc trưng yếu hơn như `Vị trí`,
`Số phòng`, `Năm xây` — tạo ra sự đa dạng thực sự.

---

## 3. ƯU ĐIỂM

### 3.1. Độ chính xác cao và ổn định

Random Forest thường xuyên nằm trong nhóm mô hình **tốt nhất trên dữ liệu dạng
bảng** (*tabular data*) mà **không cần tinh chỉnh nhiều**. Cấu hình mặc định
($B = 500$, $m = \sqrt{p}$) đã cho kết quả cạnh tranh trong đa số bài toán thực
tế — đây là lý do nó được xem là **baseline mạnh** bắt buộc phải thử.

### 3.2. Chống quá khớp (overfitting) hiệu quả

Mỗi cây riêng lẻ **quá khớp nghiêm trọng** (phương sai cao, độ chệch thấp),
nhưng phép trung bình hóa trên $B$ cây ít tương quan **triệt tiêu phần lớn
phương sai** mà **không làm tăng độ chệch**. Breiman chứng minh: khi
$B \to \infty$, sai số tổng quát hóa **hội tụ** về một giới hạn hữu hạn — nghĩa
là **thêm cây không bao giờ gây quá khớp** (chỉ tốn thời gian tính toán).

### 3.3. Ước lượng lỗi miễn phí bằng OOB

Nhờ cơ chế bootstrap, khoảng **36.8%** dữ liệu không lọt vào mỗi cây, tạo thành
tập **Out-Of-Bag**. Tập này đóng vai trò như một **tập kiểm định tích hợp sẵn**,
cho ước lượng lỗi tổng quát hóa **gần tương đương k-fold cross-validation** mà
**không tốn thêm một lần huấn luyện nào**.

### 3.4. Đo lường tầm quan trọng đặc trưng

Cung cấp hai cơ chế xếp hạng đặc trưng (**MDI** và **Permutation Importance**),
rất hữu ích cho **lựa chọn đặc trưng** (*feature selection*) và **giải thích
nghiệp vụ** — trả lời được câu hỏi "yếu tố nào ảnh hưởng lớn nhất đến kết quả?".

### 3.5. Xử lý tốt dữ liệu "khó"

| Đặc điểm dữ liệu             | Cách Random Forest xử lý                                                                              |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Quan hệ phi tuyến**      | Cây tự chia không gian thành các ô chữ nhật → xấp xỉ mọi dạng hàm                          |
| **Tương tác giữa biến** | Các nhánh lồng nhau tự động mô hình hóa tương tác bậc cao                                   |
| **Không cần chuẩn hóa**  | Phép chia dựa trên**thứ tự** giá trị, bất biến với phép biến đổi đơn điệu        |
| **Đặc trưng hỗn hợp**   | Xử lý đồng thời biến số và biến hạng mục                                                      |
| **Ngoại lai (outliers)**    | Ít nhạy cảm — outlier bị cô lập vào một lá riêng, ảnh hưởng bị pha loãng qua trung bình |
| **Đa cộng tuyến**         | Không sụp đổ như hồi quy tuyến tính (dù làm nhiễu chỉ số tầm quan trọng)                  |
| **Giá trị thiếu**         | Có thể xử lý qua*surrogate splits* hoặc điền khuyết dựa trên proximity                       |

### 3.6. Song song hóa hoàn toàn

Các cây **độc lập hoàn toàn** với nhau → huấn luyện có thể phân tán trên nhiều
lõi CPU/nhiều máy với hiệu suất gần tuyến tính (`n_jobs=-1`). Đây là lợi thế
lớn so với các phương pháp tuần tự như **Gradient Boosting** (cây sau phụ thuộc
cây trước).

### 3.7. Ít siêu tham số nhạy cảm

Chỉ có $m$ (`max_features`) là thực sự cần tinh chỉnh; $B$ chỉ cần "đủ lớn".
So với mạng nơ-ron (learning rate, kiến trúc, batch size, regularization...),
chi phí tinh chỉnh thấp hơn nhiều.

---

## 4. NHƯỢC ĐIỂM

### 4.1. Mất khả năng diễn giải trực tiếp ("hộp đen")

Một cây quyết định đơn lẻ có thể vẽ ra và đọc thành các luật `IF-THEN` rõ ràng.
Nhưng một rừng gồm 500 cây, mỗi cây hàng trăm nút, thì **không thể diễn giải
bằng mắt**. Chỉ có thể giải thích **gián tiếp** qua feature importance, SHAP,
hoặc partial dependence plot.

> **Hệ quả thực tiễn:** trong các lĩnh vực bị quản lý chặt (tín dụng ngân hàng,
> bảo hiểm, y tế), nơi luật pháp yêu cầu **giải thích được lý do từ chối hồ sơ**,
> Random Forest có thể không được chấp nhận, buộc phải dùng Logistic Regression
> hoặc Decision Tree đơn.

### 4.2. Chi phí tính toán và bộ nhớ lớn

- **Thời gian huấn luyện:** $O(B \cdot m \cdot n \log^2 n)$ — tăng tuyến tính
  theo số cây.
- **Bộ nhớ:** phải lưu **toàn bộ cấu trúc** của $B$ cây. Một rừng 500 cây sâu
  trên 1 triệu mẫu có thể chiếm **hàng GB RAM**.
- **Thời gian dự đoán:** phải duyệt qua cả $B$ cây cho mỗi mẫu → **chậm hơn
  hàng trăm lần** so với hồi quy tuyến tính. Bất lợi cho các hệ thống
  **thời gian thực** với ràng buộc độ trễ khắt khe (quảng cáo realtime, giao
  dịch tần suất cao).

### 4.3. Không ngoại suy được (giới hạn nghiêm trọng của hồi quy)

Dự đoán của cây luôn là **trung bình các giá trị $y$ trong tập huấn luyện** tại
một lá. Do đó:

$$
\min(y_{train}) \le \hat{y} \le \max(y_{train})
$$

**Random Forest KHÔNG THỂ dự đoán giá trị nằm ngoài khoảng đã thấy.** Ví dụ:
huấn luyện trên giá nhà 1–5 tỷ, mô hình **không bao giờ** dự đoán 8 tỷ, kể cả
với căn biệt thự 500 m². Với **dữ liệu chuỗi thời gian có xu hướng tăng**
(doanh thu, dân số, lạm phát), đây là **lỗi chí mạng** — mô hình sẽ dự báo
"phẳng" ở mức lịch sử. Trường hợp này phải dùng hồi quy tuyến tính, ARIMA,
hoặc khử xu hướng (*detrending*) trước khi áp dụng Random Forest.

### 4.4. Thiên lệch trong đo lường tầm quan trọng (MDI bias)

Chỉ số MDI **thiên vị** theo hai hướng đã được chứng minh:

1. **Ưu ái đặc trưng có nhiều giá trị phân biệt** (biến liên tục, biến hạng mục
   nhiều cấp) — vì chúng có nhiều điểm cắt hơn nên dễ "vô tình" giảm được tạp
   chất.
2. **Chia sẻ sai tầm quan trọng giữa các biến tương quan** — khi $X_1, X_2$
   tương quan 0.95, tầm quan trọng thực bị chia đôi, khiến cả hai trông "yếu"
   hơn thực tế.

**Khắc phục:** dùng **Permutation Importance trên tập kiểm tra** hoặc **SHAP
values**.

### 4.5. Yếu với dữ liệu thưa và chiều rất cao

Với dữ liệu văn bản dạng TF-IDF (hàng chục nghìn chiều, ma trận thưa >99%),
Random Forest kém hơn hẳn **Linear SVM** hoặc **Logistic Regression**: khi
$m = \sqrt{p}$ đặc trưng được rút ngẫu nhiên, xác suất trúng đặc trưng có ý
nghĩa rất thấp → nhiều cây gần như học từ nhiễu.

### 4.6. Thiên lệch với dữ liệu mất cân bằng

Với tỷ lệ lớp 99:1, phép đa số phiếu có xu hướng nghiêng về lớp đa số. Cần
**Balanced Random Forest**, `class_weight='balanced_subsample'`, hoặc lấy mẫu
lại (SMOTE / undersampling).

### 4.7. Không phù hợp với dữ liệu phi cấu trúc

Với **ảnh, âm thanh, văn bản dài**, các mô hình học sâu (CNN, Transformer) vượt
trội hoàn toàn nhờ khả năng **tự học biểu diễn đặc trưng** (*representation
learning*) — điều Random Forest không có.

---

## 5. NGỮ CẢNH — SỬ DỤNG KHI NÀO

### 5.1. NÊN dùng Random Forest khi

| #  | Điều kiện                                                         | Lý do                                                               |
| -- | -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 1  | **Dữ liệu dạng bảng** (CSV, Excel, SQL)                    | Đây là "sân nhà" của thuật toán                              |
| 2  | **Cần một baseline mạnh nhanh chóng**                      | Chạy được ngay với tham số mặc định, làm mốc so sánh     |
| 3  | **Nghi ngờ quan hệ phi tuyến / có tương tác**           | Cây tự phát hiện mà không cần kỹ sư đặc trưng thủ công |
| 4  | **Đặc trưng hỗn hợp, thang đo khác nhau**               | Không cần chuẩn hóa, không cần xử lý phức tạp              |
| 5  | **Dữ liệu có nhiễu / ngoại lai**                          | Bản chất trung bình hóa làm mô hình bền vững                |
| 6  | **Số mẫu vừa phải (10³ – 10⁶)**                         | Cân bằng tốt giữa độ chính xác và chi phí                  |
| 7  | **Cần xếp hạng mức độ quan trọng của biến**           | Feature importance sẵn có                                          |
| 8  | **Không đủ dữ liệu cho deep learning**                    | Random Forest hoạt động tốt với vài nghìn mẫu                |
| 9  | **Có tài nguyên đa nhân, cần huấn luyện nhanh**        | Song song hóa hoàn hảo                                            |
| 10 | **Chấp nhận đánh đổi diễn giải lấy độ chính xác** | Không bị ràng buộc pháp lý về giải thích                    |

### 5.2. KHÔNG NÊN dùng khi

| # | Tình huống                                                                         | Nên dùng thay thế                                        |
| - | ------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| 1 | **Bắt buộc giải thích từng quyết định** (tín dụng, y khoa pháp lý) | Logistic Regression, Decision Tree đơn, GAM               |
| 2 | **Dự báo có xu hướng ngoại suy** (chuỗi thời gian tăng trưởng)      | Linear Regression, ARIMA, Prophet, LSTM                     |
| 3 | **Dữ liệu phi cấu trúc** (ảnh, âm thanh, văn bản)                      | CNN, RNN, Transformer                                       |
| 4 | **Suy luận thời gian thực, độ trễ < 1 ms**                               | Linear/Logistic Regression, mô hình đã chưng cất      |
| 5 | **Chiều cực cao và thưa** (TF-IDF, one-hot hàng chục nghìn cột)        | Linear SVM, Naive Bayes, Logistic + L1                      |
| 6 | **Cần độ chính xác tối đa trên bảng, chấp nhận tinh chỉnh**        | XGBoost, LightGBM, CatBoost                                 |
| 7 | **Quan hệ thực sự tuyến tính, đơn giản**                               | Linear Regression (đơn giản hơn, diễn giải tốt hơn) |
| 8 | **Thiết bị nhúng, RAM hạn chế**                                           | Decision Tree đơn, mô hình tuyến tính                 |

### 5.3. Quy tắc lựa chọn thực dụng

```text
Dữ liệu là ẢNH / ÂM THANH / VĂN BẢN DÀI?
    └─ CÓ  → Deep Learning (CNN / Transformer)
    └─ KHÔNG ↓
Dữ liệu dạng BẢNG?
    └─ CÓ ↓
Cần GIẢI THÍCH từng quyết định (ràng buộc pháp lý)?
    └─ CÓ  → Logistic Regression / Decision Tree đơn / GAM
    └─ KHÔNG ↓
Cần NGOẠI SUY ngoài khoảng dữ liệu (xu hướng thời gian)?
    └─ CÓ  → Linear Regression / ARIMA / detrend rồi mới dùng RF
    └─ KHÔNG ↓
Cần độ chính xác TỐI ĐA và có thời gian tinh chỉnh?
    └─ CÓ  → XGBoost / LightGBM  (RF vẫn nên chạy làm baseline)
    └─ KHÔNG → ✅ RANDOM FOREST
```

---

## 6. THUẬT NGỮ

### 6.1. Nhóm thuật ngữ nền tảng

**① Ensemble Learning (Học kết hợp)**
Phương pháp kết hợp nhiều mô hình cơ sở để tạo ra một mô hình tổng hợp có hiệu
năng vượt trội từng thành viên. Ba họ chính:

- **Bagging** — huấn luyện song song, độc lập → *giảm phương sai* (Random Forest).
- **Boosting** — huấn luyện tuần tự, mô hình sau sửa lỗi mô hình trước → *giảm
  độ chệch* (AdaBoost, XGBoost).
- **Stacking** — dùng một meta-model học cách kết hợp đầu ra các mô hình cơ sở.

**② Base Learner / Weak Learner (Bộ học cơ sở / Bộ học yếu)**
Mô hình thành phần trong ensemble. Trong Random Forest, base learner là một
**cây quyết định CART không (hoặc rất ít) cắt tỉa**.

**③ Decision Tree (Cây quyết định)**
Mô hình dạng cây trong đó mỗi **nút trong** (*internal node*) là một câu hỏi
kiểm tra điều kiện trên một đặc trưng ($x_j \le s$?), mỗi **nhánh** là một câu
trả lời, mỗi **nút lá** (*leaf*) chứa kết quả dự đoán.

**④ CART (Classification And Regression Trees)**
Thuật toán xây cây do Breiman đề xuất (1984), luôn tạo **cây nhị phân**. Dùng
Gini cho phân loại, SSE cho hồi quy. Đây là base learner chuẩn của Random Forest.

**⑤ Root / Internal Node / Leaf (Gốc / Nút trong / Lá)**

- **Root**: nút trên cùng, chứa toàn bộ dữ liệu tại cây đó.
- **Internal node**: nút có nhánh con, chứa điều kiện phân tách.
- **Leaf**: nút cuối, chứa giá trị dự đoán (mode cho phân loại, mean cho hồi quy).

**⑥ Depth (Độ sâu)**
Số cạnh dài nhất từ gốc đến một lá. Độ sâu càng lớn → cây càng phức tạp → càng
dễ quá khớp. Trong Random Forest, cây thường được **để mọc tự do** vì phương sai
sẽ được xử lý bởi bước tổng hợp.

### 6.2. Nhóm thuật ngữ về ngẫu nhiên hóa

**⑦ Bootstrap Sampling (Lấy mẫu bootstrap / lấy mẫu có hoàn lại)**
Từ tập $D$ có $n$ mẫu, rút ra $n$ mẫu **có hoàn lại** (mỗi mẫu có thể được rút
nhiều lần). Kết quả: tập $D_b$ cũng có $n$ phần tử nhưng chứa các bản sao trùng
lặp và **bỏ sót ~36.8%** dữ liệu gốc.

**⑧ Bagging (Bootstrap AGGregatING)**
Kỹ thuật: tạo $B$ mẫu bootstrap → huấn luyện $B$ mô hình độc lập → tổng hợp
bằng vote/average. Mục tiêu duy nhất: **giảm phương sai**.

**⑨ Out-Of-Bag (OOB) — Mẫu ngoài túi**
Các mẫu **không** được chọn vào $D_b$. Với mỗi cây $T_b$, tập $OOB_b$ đóng vai
trò tập kiểm định "sạch" (cây chưa từng thấy chúng).

**⑩ OOB Error / OOB Score (Lỗi OOB)**
Ước lượng lỗi tổng quát hóa tính trên các dự đoán OOB. Với mỗi mẫu $x_i$, chỉ
dùng những cây **không** chứa $x_i$ trong tập bootstrap của nó để dự đoán, rồi
so sánh với $y_i$ thực. Kết quả tiệm cận **Leave-One-Out CV** khi $B$ đủ lớn.

**⑪ Random Subspace Method (Phương pháp không gian con ngẫu nhiên)**
Kỹ thuật do Tin Kam Ho đề xuất (1998): tại mỗi nút, chỉ xét một tập con ngẫu
nhiên $m$ trong $p$ đặc trưng. Tham số này trong scikit-learn là `max_features`.

**⑫ Decorrelation (Khử tương quan)**
Quá trình làm giảm hệ số tương quan $\rho$ giữa dự đoán của các cây. Đây là
**mục đích tối thượng** của việc ngẫu nhiên hóa đặc trưng và là điều làm Random
Forest mạnh hơn Bagging thuần.

### 6.3. Nhóm thuật ngữ về tiêu chí phân tách

**⑬ Impurity (Độ tạp chất / Độ hỗn tạp)**
Thước đo mức độ "trộn lẫn" của các nhãn trong một nút. Nút chỉ chứa một lớp
duy nhất → *thuần khiết* (impurity = 0). Nút chứa các lớp cân bằng → impurity
đạt cực đại.

**⑭ Gini Impurity (Chỉ số Gini)**
Xác suất phân loại sai một mẫu nếu ta gán nhãn ngẫu nhiên theo phân phối nhãn
trong nút. Tính nhanh (không cần logarit) → mặc định của CART.

**⑮ Entropy (Độ hỗn loạn thông tin)**
Đại lượng từ lý thuyết thông tin của Shannon, đo lượng "bất định" trung bình.
Đơn vị: bit (log cơ số 2).

**⑯ Information Gain (Độ lợi thông tin)**
Lượng entropy **giảm được** sau khi phân tách. Chọn phép chia có IG lớn nhất.

**⑰ Variance Reduction / SSE Reduction (Giảm phương sai — hồi quy)**
Với bài toán hồi quy, tiêu chí là giảm tổng bình phương sai số (SSE) hoặc
phương sai trong nút.

**⑱ Split Point / Threshold (Điểm cắt / Ngưỡng)**
Cặp $(j, s)$ gồm chỉ số đặc trưng và giá trị ngưỡng để chia dữ liệu thành hai
nhánh: $x_j \le s$ và $x_j > s$.

### 6.4. Nhóm thuật ngữ về đánh giá và giải thích

**⑲ Bias – Variance Tradeoff (Đánh đổi độ chệch – phương sai)**

- **Bias (độ chệch)**: sai lệch hệ thống do mô hình quá đơn giản → *thiếu khớp*.
- **Variance (phương sai)**: độ nhạy của mô hình với biến động dữ liệu huấn
  luyện → *quá khớp*.

Random Forest giữ nguyên bias thấp của cây sâu, đồng thời **kéo giảm mạnh
variance** — đây là lý do nó hiệu quả.

**⑳ Feature Importance (Tầm quan trọng đặc trưng)**
Điểm số phản ánh mức đóng góp của mỗi đặc trưng vào chất lượng dự đoán.

**㉑ MDI — Mean Decrease in Impurity (Giảm tạp chất trung bình)**
Tổng mức giảm impurity mà một đặc trưng mang lại trên tất cả các nút, trên tất
cả các cây, có trọng số theo số mẫu. Nhanh nhưng **thiên lệch**.

**㉒ Permutation Importance (Tầm quan trọng hoán vị) / MDA**
Đo mức **suy giảm hiệu năng** khi xáo trộn ngẫu nhiên giá trị của một đặc trưng.
Chậm hơn nhưng **khách quan hơn** MDI, đặc biệt khi tính trên tập kiểm tra.

**㉓ Proximity Matrix (Ma trận lân cận)**
Ma trận $n \times n$ đo mức độ "giống nhau" giữa hai mẫu: $prox(i,j)$ = tỷ lệ
cây mà $x_i$ và $x_j$ cùng rơi vào một lá. Dùng để phát hiện bất thường, phân
cụm, và điền giá trị thiếu.

**㉔ Margin & Strength & Correlation (Biên, Sức mạnh, Tương quan)**
Bộ ba đại lượng trong chứng minh lý thuyết của Breiman về cận trên sai số tổng
quát hóa (xem [mục 7.9](#79-cận-trên-sai-số-tổng-quát-hóa-của-breiman)).

**㉕ Extremely Randomized Trees (Extra Trees)**
Biến thể "cực đoan" hơn: ngưỡng cắt cũng được chọn **hoàn toàn ngẫu nhiên**
thay vì tối ưu, và thường **không dùng bootstrap**. Giảm variance mạnh hơn nữa,
đổi lại bias tăng nhẹ; huấn luyện nhanh hơn đáng kể.

---

## 7. CÔNG THỨC

> **Quy ước ký hiệu chung**
>
> | Ký hiệu                                                  | Ý nghĩa                                    |
> | ---------------------------------------------------------- | -------------------------------------------- |
> | $n$                                                      | Số mẫu trong tập huấn luyện             |
> | $p$                                                      | Số đặc trưng (chiều của$\mathbf{x}$) |
> | $B$                                                      | Số cây trong rừng                         |
> | $m$ | Số đặc trưng xét tại mỗi nút ($m \le p$) |                                              |
> | $C$                                                      | Số lớp (bài toán phân loại)            |
> | $T_b$                                                    | Cây thứ$b$                               |
> | $t$                                                      | Một nút bất kỳ trong cây                |
> | $N_t$                                                    | Số mẫu tại nút$t$                      |
> | $\hat{y}$                                                | Giá trị dự đoán                         |

### 7.1. Công thức lấy mẫu Bootstrap

**Xác suất một mẫu cụ thể KHÔNG được chọn trong một lần rút:**

$$
P(\text{không chọn 1 lần}) = 1 - \frac{1}{n}
$$

**Xác suất KHÔNG được chọn trong cả $n$ lần rút độc lập:**

$$
P(\text{OOB}) = \left(1 - \frac{1}{n}\right)^{n}
$$

**Giới hạn khi $n \to \infty$:**

$$
\lim_{n \to \infty}\left(1 - \frac{1}{n}\right)^{n} = e^{-1} \approx 0{,}3679
$$

**Giải thích chi tiết:**

- Mỗi lần rút, xác suất trúng một mẫu cố định là $1/n$ → trượt là $1 - 1/n$.
- Bootstrap rút $n$ lần **độc lập, có hoàn lại** → nhân xác suất $n$ lần.
- Kết quả hội tụ về hằng số Euler nghịch đảo $e^{-1}$.

**Kết luận định lượng:**

- **≈ 36.8%** dữ liệu là **OOB** → dùng làm tập kiểm định miễn phí.
- **≈ 63.2%** dữ liệu **duy nhất** lọt vào mỗi cây (phần còn lại là bản sao trùng).

**Bảng giá trị thực tế:**

| $n$      | $(1 - 1/n)^n$ | % OOB   |
| ---------- | --------------- | ------- |
| 10         | 0.3487          | 34.87%  |
| 100        | 0.3660          | 36.60%  |
| 1 000      | 0.3677          | 36.77%  |
| 10 000     | 0.3679          | 36.79%  |
| $\infty$ | 0.36788         | 36.788% |

---

### 7.2. Chỉ số Gini (Gini Impurity)

$$
G(t) = 1 - \sum_{c=1}^{C} p_c^2 \qquad \text{trong đó } p_c = \frac{N_{t,c}}{N_t}
$$

**Giải thích từng thành phần:**

- $p_c$ — tỷ lệ mẫu thuộc lớp $c$ tại nút $t$.
- $\sum p_c^2$ — xác suất **hai mẫu rút ngẫu nhiên độc lập** cùng thuộc một lớp
  (xác suất "đoán đúng").
- $1 - \sum p_c^2$ — xác suất **đoán sai** = mức độ tạp chất.

**Miền giá trị:**

- $G = 0$ ⟺ nút thuần khiết (mọi mẫu cùng một lớp) — **tốt nhất**.
- $G_{\max} = 1 - \frac{1}{C}$ khi các lớp phân bố đều — **xấu nhất**.
  - Nhị phân ($C=2$): $G_{\max} = 0.5$
  - Ba lớp ($C=3$): $G_{\max} \approx 0.667$

**Ví dụ tính tay** — nút có 10 mẫu: 6 lớp "Có", 4 lớp "Không"

$$
p_{Có} = 0.6,\quad p_{Không} = 0.4
$$

$$
G = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 1 - 0.52 = \mathbf{0.48}
$$

---

### 7.3. Entropy và Information Gain

**Entropy tại nút $t$:**

$$
H(t) = -\sum_{c=1}^{C} p_c \log_2 p_c \qquad (\text{quy ước } 0\log_2 0 = 0)
$$

**Giải thích:**

- $-\log_2 p_c$ là **lượng thông tin bất ngờ** (*surprisal*) khi quan sát lớp
  $c$: lớp càng hiếm → càng "bất ngờ" → thông tin càng lớn.
- $H(t)$ là **kỳ vọng** của lượng bất ngờ đó — tức mức bất định trung bình.
- Đơn vị: **bit**.

**Miền giá trị:** $H = 0$ (thuần khiết) đến $H_{\max} = \log_2 C$ (phân bố đều).

**Information Gain của phép chia $S$:**

$$
IG(t, S) = H(t) - \left[\frac{N_L}{N_t} H(t_L) + \frac{N_R}{N_t} H(t_R)\right]
$$

**Giải thích:** entropy của nút cha trừ đi **trung bình có trọng số** entropy
các nút con. Trọng số $N_L/N_t$, $N_R/N_t$ phản ánh việc nút chứa nhiều mẫu hơn
thì quan trọng hơn. Thuật toán chọn phép chia có $IG$ **lớn nhất**.

**Ví dụ tính tay** — cùng nút 10 mẫu (6 Có, 4 Không):

$$
H = -(0.6\log_2 0.6 + 0.4\log_2 0.4)
$$

$$
= -\big(0.6 \times (-0{,}737) + 0.4 \times (-1{,}322)\big) = 0{,}442 + 0{,}529 = \mathbf{0{,}971 \text{ bit}}
$$

Giả sử chia được thành: nhánh trái 5 mẫu (5 Có, 0 Không), nhánh phải 5 mẫu
(1 Có, 4 Không):

$$
H(t_L) = 0 \quad (\text{thuần khiết})
$$

$$
H(t_R) = -(0.2\log_2 0.2 + 0.8\log_2 0.8) = 0{,}464 + 0{,}258 = 0{,}722
$$

$$
IG = 0{,}971 - \left[\tfrac{5}{10}(0) + \tfrac{5}{10}(0{,}722)\right] = 0{,}971 - 0{,}361 = \mathbf{0{,}610 \text{ bit}}
$$

**So sánh Gini và Entropy:** cho kết quả gần như nhau trong thực tế (khác biệt
ở dưới 2% số trường hợp). Gini **nhanh hơn** vì không cần tính logarit → được
chọn làm mặc định.

---

### 7.4. Tiêu chí phân tách cho bài toán hồi quy

**Tổng bình phương sai số tại nút $t$:**

$$
SSE(t) = \sum_{i \in t} (y_i - \bar{y}_t)^2 \qquad \text{với } \bar{y}_t = \frac{1}{N_t}\sum_{i \in t} y_i
$$

**Bài toán tối ưu tại mỗi nút** — tìm cặp $(j, s)$ tốt nhất:

$$
(j^*, s^*) = \arg\min_{j \in S,\; s} \left[ \sum_{i:\, x_{ij} \le s} (y_i - \bar{y}_L)^2 + \sum_{i:\, x_{ij} > s} (y_i - \bar{y}_R)^2 \right]
$$

**Giải thích:**

- $S$ là tập con **$m$ đặc trưng ngẫu nhiên** — đây chính là điểm khác biệt của
  Random Forest so với cây CART thông thường (vốn duyệt cả $p$ đặc trưng).
- $\bar{y}_L, \bar{y}_R$ là trung bình $y$ của nhánh trái/phải.
- Thuật toán duyệt mọi ngưỡng $s$ khả dĩ (thường là trung điểm giữa hai giá trị
  liên tiếp đã sắp xếp) và chọn tổ hợp làm SSE tổng nhỏ nhất.

**Mức giảm tạp chất (dùng cho feature importance):**

$$
\Delta(t) = SSE(t) - \left[SSE(t_L) + SSE(t_R)\right] \ge 0
$$

**Dự đoán tại lá:**

$$
\hat{y}_{leaf} = \bar{y}_{leaf} = \frac{1}{N_{leaf}}\sum_{i \in leaf} y_i
$$

*(Chính công thức này giải thích vì sao Random Forest không ngoại suy được:
dự đoán luôn là trung bình các $y$ đã quan sát.)*

---

### 7.5. Công thức tổng hợp dự đoán (Aggregation)

**a) Hồi quy — trung bình cộng:**

$$
\hat{y}_{RF}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x})
$$

**b) Phân loại — đa số phiếu cứng (*hard voting*):**

$$
\hat{y}_{RF}(\mathbf{x}) = \arg\max_{c \in \{1..C\}} \sum_{b=1}^{B} \mathbb{1}\left[T_b(\mathbf{x}) = c\right]
$$

trong đó $\mathbb{1}[\cdot]$ là **hàm chỉ thị**: bằng 1 nếu điều kiện đúng, bằng
0 nếu sai. Nói cách khác: đếm số cây bầu cho từng lớp, chọn lớp nhiều phiếu nhất.

**c) Phân loại — trung bình xác suất (*soft voting*, thường chính xác hơn):**

$$
\hat{P}(c \mid \mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} \hat{p}_{b}(c \mid \mathbf{x}), \qquad \hat{y}_{RF} = \arg\max_c \hat{P}(c \mid \mathbf{x})
$$

với $\hat{p}_b(c|\mathbf{x})$ là tỷ lệ mẫu lớp $c$ tại lá mà $\mathbf{x}$ rơi
vào trong cây $T_b$. Soft voting tận dụng được **mức độ tự tin** của từng cây
thay vì chỉ nhãn cuối → đây là cách scikit-learn triển khai mặc định.

---

### 7.6. Công thức giảm phương sai — LINH HỒN CỦA RANDOM FOREST

Giả sử $B$ cây, mỗi cây có phương sai $\sigma^2$, và **hệ số tương quan từng
đôi** giữa các cây là $\rho$. Phương sai của trung bình:

$$
\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} T_b\right) = \rho\,\sigma^{2} + \frac{1-\rho}{B}\,\sigma^{2}
$$

**Chứng minh vắn tắt:**

$$
\operatorname{Var}\!\left(\frac{1}{B}\sum T_b\right) = \frac{1}{B^2}\left[\sum_b \operatorname{Var}(T_b) + \sum_{b \ne b'} \operatorname{Cov}(T_b, T_{b'})\right]
$$

$$
= \frac{1}{B^2}\left[B\sigma^2 + B(B-1)\rho\sigma^2\right] = \frac{\sigma^2}{B} + \frac{(B-1)\rho\sigma^2}{B} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

**Phân tích hai số hạng — đây là phần quan trọng nhất của cả báo cáo:**

| Số hạng                     | Bản chất                                                 | Cách kiểm soát                                      |
| ----------------------------- | ---------------------------------------------------------- | ------------------------------------------------------ |
| $\dfrac{1-\rho}{B}\sigma^2$ | Phần**triệt tiêu được** bằng trung bình hóa | Tăng$B$ → tiến về 0                              |
| $\rho\,\sigma^2$            | **Sàn không thể vượt qua**                      | Chỉ giảm được bằng cách**giảm $\rho$** |

**Kết luận then chốt:**

$$
\lim_{B \to \infty} \operatorname{Var} = \rho\,\sigma^{2}
$$

Dù ta có trồng **một triệu cây**, phương sai vẫn bị chặn dưới bởi $\rho\sigma^2$.
Vì vậy **cách duy nhất để cải thiện tiếp là hạ $\rho$** — và đó chính xác là
nhiệm vụ của cơ chế **chọn $m$ đặc trưng ngẫu nhiên tại mỗi nút**.

**Minh họa số:** với $\sigma^2 = 1$, $B = 500$

| Phương pháp                  | $\rho$ | Phương sai tổ hợp               | Mức giảm    |
| ------------------------------- | -------- | ----------------------------------- | ------------- |
| Một cây đơn                 | —       | 1.000                               | —            |
| Bagging (không random feature) | 0.60     | $0.60 + \frac{0.4}{500} = 0.6008$ | 40%           |
| Random Forest ($m=\sqrt{p}$)  | 0.20     | $0.20 + \frac{0.8}{500} = 0.2016$ | **80%** |
| Extra Trees                     | 0.10     | $0.10 + \frac{0.9}{500} = 0.1018$ | **90%** |

→ Việc hạ $\rho$ từ 0.6 xuống 0.2 giúp giảm phương sai thêm **gấp 3 lần**, trong
khi tăng $B$ từ 500 lên 5000 gần như **không cải thiện gì**.

---

### 7.7. Lỗi Out-Of-Bag (OOB Error)

**Dự đoán OOB cho mẫu $i$** — chỉ dùng các cây không chứa $x_i$:

$$
\hat{y}_i^{OOB} = \frac{1}{|\mathcal{B}_i|}\sum_{b \in \mathcal{B}_i} T_b(\mathbf{x}_i), \qquad \mathcal{B}_i = \{b : \mathbf{x}_i \notin D_b\}
$$

Kỳ vọng $|\mathcal{B}_i| \approx 0.368 \times B$ cây.

**Lỗi OOB cho phân loại:**

$$
\text{OOB Error} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}\left[\hat{y}_i^{OOB} \ne y_i\right]
$$

**Lỗi OOB cho hồi quy:**

$$
\text{OOB MSE} = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i^{OOB}\right)^2, \qquad \text{OOB } R^2 = 1 - \frac{\sum (y_i - \hat{y}_i^{OOB})^2}{\sum (y_i - \bar{y})^2}
$$

**Ý nghĩa thực tiễn:** OOB Error là **ước lượng không chệch** của lỗi tổng quát
hóa, thu được **miễn phí** trong quá trình huấn luyện — thay thế được k-fold
cross-validation, tiết kiệm $k$ lần huấn luyện.

---

### 7.8. Công thức tầm quan trọng đặc trưng

**a) MDI — Mean Decrease in Impurity**

Với một cây $T_b$, tầm quan trọng của đặc trưng $X_j$:

$$
Imp_b(X_j) = \sum_{t \in T_b \,:\, v(t) = j} \frac{N_t}{n}\,\Delta I(t)
$$

trong đó:

- $v(t) = j$ — các nút $t$ sử dụng đặc trưng $X_j$ để phân tách.
- $\frac{N_t}{n}$ — **trọng số** theo tỷ lệ mẫu đi qua nút $t$ (nút gần gốc,
  nhiều mẫu → quan trọng hơn).
- $\Delta I(t) = I(t) - \frac{N_L}{N_t}I(t_L) - \frac{N_R}{N_t}I(t_R)$ — mức
  giảm tạp chất tại nút $t$.

**Trung bình trên toàn rừng và chuẩn hóa:**

$$
Imp(X_j) = \frac{1}{B}\sum_{b=1}^{B} Imp_b(X_j), \qquad \widetilde{Imp}(X_j) = \frac{Imp(X_j)}{\sum_{k=1}^{p} Imp(X_k)}
$$

Sau chuẩn hóa: $\sum_j \widetilde{Imp}(X_j) = 1$ → đọc được như "phần trăm đóng
góp".

**b) Permutation Importance (MDA — Mean Decrease in Accuracy)**

$$
PI(X_j) = \frac{1}{B}\sum_{b=1}^{B}\left[ err_{OOB_b}^{\pi_j} - err_{OOB_b} \right]
$$

**Quy trình chi tiết:**

1. Tính lỗi gốc $err_{OOB_b}$ của cây $b$ trên tập OOB của nó.
2. **Xáo trộn ngẫu nhiên** (hoán vị $\pi_j$) cột giá trị của đặc trưng $X_j$
   trong tập OOB — phá vỡ mọi liên hệ giữa $X_j$ và $y$, nhưng **giữ nguyên
   phân phối biên** của $X_j$.
3. Tính lại lỗi $err_{OOB_b}^{\pi_j}$.
4. Hiệu số = mức thiệt hại do "làm hỏng" $X_j$. Càng lớn → $X_j$ càng quan trọng.
5. Trung bình trên toàn rừng.

**Ưu điểm so với MDI:** không thiên vị biến nhiều giá trị, đo trên **hiệu năng
thực tế** thay vì tiêu chí nội bộ của cây, và áp dụng được cho **mọi mô hình**.

---

### 7.9. Cận trên sai số tổng quát hóa của Breiman

Định nghĩa **hàm biên** (*margin function*) cho phân loại:

$$
mg(\mathbf{x}, y) = P_{\Theta}\left[h(\mathbf{x},\Theta) = y\right] - \max_{c \ne y} P_{\Theta}\left[h(\mathbf{x},\Theta) = c\right]
$$

*(tỷ lệ cây bầu đúng trừ tỷ lệ cây bầu cho lớp sai mạnh nhất — biên càng lớn,
dự đoán càng "chắc chắn").*

**Sai số tổng quát hóa:** $PE^* = P_{\mathbf{x},y}\left[mg(\mathbf{x},y) < 0\right]$

**Định lý (Breiman, 2001):**

$$
PE^{*} \;\le\; \frac{\bar{\rho}\,(1 - s^{2})}{s^{2}}
$$

trong đó:

- $s = \mathbb{E}_{\mathbf{x},y}[mg(\mathbf{x},y)]$ — **strength** (sức mạnh
  trung bình của từng cây).
- $\bar{\rho}$ — **tương quan trung bình** giữa các cây.

**Diễn giải — đây là kim chỉ nam thiết kế thuật toán:**

- Sai số giảm khi **$s$ tăng** (cây mạnh hơn) và khi **$\bar{\rho}$ giảm**
  (cây độc lập hơn).
- Hai mục tiêu này **đối kháng nhau**: giảm $m$ → cây độc lập hơn
  ($\rho \downarrow$) nhưng cũng yếu đi ($s \downarrow$).
- Việc chọn $m$ chính là **tìm điểm cân bằng tối ưu** giữa $s$ và $\bar{\rho}$
  — đó là lý do $m$ là siêu tham số quan trọng nhất của Random Forest.

**Định lý hội tụ:** khi $B \to \infty$, $PE^*$ **hội tụ hầu chắc chắn** về một
giới hạn hữu hạn → *Random Forest không quá khớp khi tăng số cây*.

---

### 7.10. Giá trị mặc định của $m$ và độ phức tạp

**Quy tắc chọn $m$ (`max_features`):**

$$
m = \begin{cases}
\left\lfloor \sqrt{p} \right\rfloor & \text{Phân loại (mặc định)}\\[4pt]
\left\lfloor p/3 \right\rfloor \text{ hoặc } \left\lfloor\sqrt{p}\right\rfloor & \text{Hồi quy}\\[4pt]
\left\lfloor \log_2 p \right\rfloor + 1 & \text{Biến thể thay thế}\\[4pt]
p & \text{Suy biến thành Bagging thuần}
\end{cases}
$$

**Độ phức tạp tính toán:**

| Giai đoạn            | Độ phức tạp                   | Ghi chú                                            |
| ---------------------- | --------------------------------- | --------------------------------------------------- |
| Huấn luyện 1 cây    | $O(m \cdot n \log^2 n)$         | $n\log n$ cho sắp xếp, $\log n$ cho độ sâu |
| Huấn luyện cả rừng | $O(B \cdot m \cdot n \log^2 n)$ | Chia cho số lõi nếu song song                    |
| Dự đoán 1 mẫu      | $O(B \cdot \log n)$             | Duyệt độ sâu của$B$ cây                     |
| Bộ nhớ               | $O(B \cdot n)$                  | Số nút xấp xỉ tỷ lệ với$n$                 |

---

## 8. ÁP DỤNG — VÍ DỤ MINH HỌA

### 8.1. Ví dụ 1 — Tính tay một bước phân tách (giáo khoa)

**Bài toán:** dự đoán khách hàng có mua sản phẩm hay không.

| ID | Tuổi | Thu nhập (triệu) | Đã từng mua | **Mua?** |
| -- | ----- | ------------------ | -------------- | -------------- |
| 1  | 25    | 10                 | Không         | Không         |
| 2  | 35    | 25                 | Có            | Có            |
| 3  | 45    | 30                 | Có            | Có            |
| 4  | 22    | 8                  | Không         | Không         |
| 5  | 52    | 40                 | Có            | Có            |
| 6  | 28    | 12                 | Không         | Không         |

**Bước 1 — Gini tại nút gốc** (3 Có, 3 Không):

$$
G_{root} = 1 - \left[(3/6)^2 + (3/6)^2\right] = 1 - 0.5 = 0.5
$$

**Bước 2 — Giả sử tập con đặc trưng ngẫu nhiên tại nút này là**
$S = \{\text{Tuổi},\ \text{Đã từng mua}\}$ — đặc trưng `Thu nhập` **không được
xét** ở nút này, đúng tinh thần Random Forest.

*Ứng viên A: `Tuổi ≤ 30`*

- Trái (ID 1, 4, 6): 0 Có, 3 Không → $G_L = 1 - (0^2 + 1^2) = 0$
- Phải (ID 2, 3, 5): 3 Có, 0 Không → $G_R = 0$
- Gini có trọng số: $\frac{3}{6}(0) + \frac{3}{6}(0) = \mathbf{0}$
- **Mức giảm:** $0.5 - 0 = \mathbf{0.5}$

*Ứng viên B: `Đã từng mua = Có`*

- Cho kết quả phân tách y hệt → mức giảm cũng là 0.5.

**Bước 3 — Chọn phép chia:** cả hai đều hoàn hảo; thuật toán chọn ứng viên đầu
tiên hoặc phá hòa ngẫu nhiên. Hai nhánh con đều **thuần khiết** → dừng, tạo lá.

**Cây kết quả:**

```
              [Tuổi ≤ 30 ?]
             /             \
          Đúng             Sai
           /                 \
    ┌────────────┐      ┌────────────┐
    │ Lá: KHÔNG  │      │  Lá: CÓ    │
    │ (3 mẫu)    │      │  (3 mẫu)   │
    └────────────┘      └────────────┘
```

Trong rừng thật, cây thứ hai (với bootstrap khác và tập con đặc trưng khác) có
thể chọn `Thu nhập ≤ 18.5` làm nút gốc — tạo ra sự đa dạng cần thiết.

---

### 8.2. Ví dụ 2 — Chấm điểm tín dụng (Credit Scoring)

| Hạng mục                       | Nội dung                                                                                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bài toán**             | Phân loại nhị phân: khách hàng có vỡ nợ trong 12 tháng tới không?                                                                                       |
| **Đặc trưng**           | Tuổi, thu nhập, thời gian làm việc, dư nợ hiện tại, tỷ lệ nợ/thu nhập, số lần trễ hạn, số thẻ tín dụng, tình trạng hôn nhân, loại nhà ở |
| **Nhãn**                  | `0` = trả đúng hạn, `1` = vỡ nợ (thường mất cân bằng ~95:5)                                                                                          |
| **Vì sao dùng RF**       | Dữ liệu bảng hỗn hợp, quan hệ phi tuyến (thu nhập cao nhưng nợ cao vẫn rủi ro — tương tác), có ngoại lai                                          |
| **Cấu hình**             | `n_estimators=500`, `max_features='sqrt'`, `class_weight='balanced_subsample'`, `min_samples_leaf=5`                                                        |
| **Chỉ số đánh giá**   | AUC-ROC, Precision-Recall AUC, KS statistic (không dùng Accuracy vì mất cân bằng)                                                                             |
| **Kết quả điển hình** | AUC ≈ 0.78 – 0.85, tốt hơn Logistic Regression khoảng 3–5 điểm AUC                                                                                          |
| **Đầu ra bổ trợ**      | Feature importance chỉ ra`tỷ lệ nợ/thu nhập` và `số lần trễ hạn` là hai yếu tố quyết định                                                       |
| **Cảnh báo**             | Cần bổ sung SHAP để đáp ứng yêu cầu giải trình theo quy định của ngân hàng                                                                          |

---

### 8.3. Ví dụ 3 — Dự báo giá nhà (Hồi quy)

| Hạng mục                       | Nội dung                                                                                                                                                                                                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bài toán**             | Hồi quy: dự đoán giá bán (tỷ VNĐ)                                                                                                                                                                                                                  |
| **Đặc trưng**           | Diện tích, số phòng ngủ, số tầng, năm xây, khoảng cách tới trung tâm, quận/huyện, hướng nhà, mặt tiền đường                                                                                                                         |
| **Vì sao dùng RF**       | Giá nhà**phi tuyến mạnh** theo diện tích và có **tương tác vị trí × diện tích**; không cần chuẩn hóa các thang đo rất khác nhau (m² vs năm vs km)                                                                    |
| **Cấu hình**             | `n_estimators=300`, `max_features=p/3`, `min_samples_leaf=3`, `oob_score=True`                                                                                                                                                                     |
| **Chỉ số đánh giá**   | RMSE, MAE, MAPE,$R^2$                                                                                                                                                                                                                                    |
| **Kết quả điển hình** | $R^2 \approx 0.85$, RMSE giảm khoảng 30% so với hồi quy tuyến tính                                                                                                                                                                                 |
| **Cảnh báo quan trọng** | **Không dự đoán được** biệt thự 800 m² nếu tập huấn luyện chỉ có tối đa 200 m² — mô hình sẽ "kẹt trần" ở giá cao nhất từng thấy (xem [mục 4.3](#43-không-ngoại-suy-được-giới-hạn-nghiêm-trọng-của-hồi-quy)) |

---

### 8.4. Ví dụ 4 — Phát hiện gian lận thẻ tín dụng

| Hạng mục                        | Nội dung                                                                                                                                                                     |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bài toán**              | Phân loại nhị phân cực kỳ mất cân bằng (~0.17% giao dịch là gian lận)                                                                                             |
| **Đặc trưng**            | Số tiền, thời điểm, khoảng cách địa lý so với giao dịch trước, tần suất giao dịch trong 1 giờ, loại merchant, thiết bị                                   |
| **Vì sao dùng RF**        | Mẫu gian lận thể hiện qua**tổ hợp điều kiện phi tuyến** ("số tiền lớn + nửa đêm + merchant lạ + xa vị trí thường"), rất hợp với cấu trúc cây |
| **Xử lý mất cân bằng** | Balanced Random Forest (undersample lớp đa số trong mỗi bootstrap) hoặc SMOTE                                                                                            |
| **Chỉ số đánh giá**    | Recall của lớp gian lận (ưu tiên hàng đầu), Precision@k, PR-AUC                                                                                                       |
| **Ràng buộc thực tế**   | Thời gian dự đoán phải dưới 100 ms → giới hạn`n_estimators` và `max_depth`, hoặc chưng cất mô hình                                                        |

---

### 8.5. Ví dụ 5 — Chẩn đoán y khoa hỗ trợ

| Hạng mục                              | Nội dung                                                                                                                                                         |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bài toán**                    | Phân loại: nguy cơ mắc bệnh tim (Có/Không)                                                                                                                 |
| **Đặc trưng**                  | Tuổi, giới tính, huyết áp, cholesterol, đường huyết, nhịp tim tối đa, kết quả điện tâm đồ, tiền sử gia đình                                |
| **Vì sao dùng RF**              | Bộ dữ liệu**nhỏ** (vài trăm đến vài nghìn bệnh nhân) — không đủ cho deep learning; cần **xếp hạng yếu tố nguy cơ** cho bác sĩ |
| **Cấu hình**                    | `n_estimators=500`, `max_features='sqrt'`, đánh giá bằng **Stratified 10-fold CV** (dữ liệu ít → không tin OOB tuyệt đối)                   |
| **Chỉ số đánh giá**          | Sensitivity (Recall) ưu tiên cao — bỏ sót ca bệnh nguy hiểm hơn báo động giả; kèm Specificity và AUC                                                |
| **Giá trị bổ trợ**            | Permutation importance giúp bác sĩ hiểu yếu tố nào ảnh hưởng lớn nhất                                                                                 |
| **Lưu ý đạo đức/pháp lý** | Chỉ là**công cụ hỗ trợ**, không thay thế chẩn đoán lâm sàng; cần diễn giải bằng SHAP                                                       |

---

### 8.6. Ví dụ 6 — Dự đoán khách hàng rời bỏ (Churn Prediction)

| Hạng mục                       | Nội dung                                                                                                                                                       |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bài toán**             | Phân loại: thuê bao viễn thông có hủy dịch vụ trong tháng tới không?                                                                                |
| **Đặc trưng**           | Thời gian gắn bó, cước phí hàng tháng, tổng cước, loại hợp đồng, số cuộc gọi hỗ trợ, dịch vụ đăng ký thêm, phương thức thanh toán |
| **Vì sao dùng RF**       | Nhiều biến hạng mục, quan hệ phi tuyến rõ rệt (nhóm khách hàng mới**và** hợp đồng theo tháng có rủi ro rời bỏ cao đột biến)       |
| **Ứng dụng nghiệp vụ** | Xếp hạng xác suất rời bỏ → nhắm mục tiêu chiến dịch giữ chân cho top 10% rủi ro cao nhất                                                        |
| **Chỉ số đánh giá**   | Lift@10%, AUC, và**giá trị kinh doanh** (chi phí khuyến mãi so với doanh thu giữ được)                                                         |

---

### 8.7. Mã nguồn tham chiếu (scikit-learn)

```python
"""
Ví dụ tham chiếu: Random Forest cho bài toán phân loại.
(Lưu ý: project hiện tại triển khai thuật toán bằng Python thuần,
 đoạn mã dưới đây chỉ mang tính đối chiếu kết quả.)
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score

# 1. Chia dữ liệu — stratify để giữ tỷ lệ lớp
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Khởi tạo mô hình
model = RandomForestClassifier(
    n_estimators=500,          # B — số cây
    max_features='sqrt',       # m = √p — tham số quan trọng nhất
    max_depth=None,            # để cây mọc tự do
    min_samples_leaf=1,        # số mẫu tối thiểu tại lá
    bootstrap=True,            # bật lấy mẫu bootstrap
    oob_score=True,            # bật ước lượng lỗi OOB miễn phí
    class_weight='balanced_subsample',  # xử lý mất cân bằng lớp
    n_jobs=-1,                 # dùng toàn bộ lõi CPU
    random_state=42            # bảo đảm tái lập kết quả
)

# 3. Huấn luyện
model.fit(X_train, y_train)
print(f"OOB Score : {model.oob_score_:.4f}")   # lỗi kiểm định "miễn phí"

# 4. Đánh giá trên tập kiểm tra
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"AUC-ROC   : {roc_auc_score(y_test, y_proba):.4f}")

# 5. Tầm quan trọng đặc trưng — MDI (nhanh, có thiên lệch)
for name, imp in sorted(zip(feature_names, model.feature_importances_),
                        key=lambda kv: -kv[1]):
    print(f"{name:<28} {imp:.4f}")

# 6. Tầm quan trọng đặc trưng — Permutation (chậm, khách quan hơn)
perm = permutation_importance(model, X_test, y_test,
                              n_repeats=30, random_state=42, n_jobs=-1)
for name, mean, std in sorted(
        zip(feature_names, perm.importances_mean, perm.importances_std),
        key=lambda kv: -kv[1]):
    print(f"{name:<28} {mean:.4f} ± {std:.4f}")
```

---

## 9. SIÊU THAM SỐ VÀ HƯỚNG DẪN TINH CHỈNH

| Siêu tham số        | Ký hiệu toán học | Ý nghĩa                           | Giá trị khuyến nghị                          | Mức độ quan trọng |
| --------------------- | -------------------- | ----------------------------------- | ------------------------------------------------ | --------------------- |
| `n_estimators`      | $B$                | Số cây trong rừng                | 100 → 500 (tăng đến khi OOB error phẳng)    | ★★★☆☆            |
| `max_features`      | $m$                | Số đặc trưng xét mỗi nút     | `sqrt` (phân loại), `p/3` (hồi quy)       | ★★★★★            |
| `max_depth`         | —                   | Độ sâu tối đa                  | `None` (tự do); giới hạn nếu nhiễu nhiều | ★★☆☆☆            |
| `min_samples_split` | —                   | Số mẫu tối thiểu để chia nút | 2 (mặc định), tăng nếu quá khớp           | ★★☆☆☆            |
| `min_samples_leaf`  | $n_{min}$          | Số mẫu tối thiểu tại lá       | 1 (phân loại), 5 (hồi quy)                    | ★★★☆☆            |
| `bootstrap`         | —                   | Bật/tắt lấy mẫu bootstrap       | `True` (bắt buộc nếu muốn dùng OOB)       | ★★★★☆            |
| `class_weight`      | —                   | Trọng số lớp                     | `balanced_subsample` nếu mất cân bằng      | ★★★★☆            |
| `criterion`         | $I(t)$             | Tiêu chí phân tách              | `gini` / `entropy` / `squared_error`       | ★☆☆☆☆            |
| `n_jobs`            | —                   | Số luồng song song                | `-1` (dùng hết lõi)                         | —                    |
| `random_state`      | —                   | Hạt giống ngẫu nhiên            | Cố định để tái lập kết quả              | —                    |

### Quy trình tinh chỉnh khuyến nghị

```text
BƯỚC 1. Chạy cấu hình mặc định (B=100, m=√p) → lấy làm mốc.
BƯỚC 2. Tăng B (100 → 200 → 500 → 1000), vẽ đường cong OOB error theo B.
        Dừng tại điểm đường cong đi ngang (thêm cây chỉ tốn thời gian).
BƯỚC 3. Dò tìm m trên lưới {1, log₂p, √p, p/3, p/2, p} — đây là tham số
        có ảnh hưởng lớn nhất tới cân bằng strength ↔ correlation.
BƯỚC 4. Nếu vẫn quá khớp: tăng min_samples_leaf (1 → 3 → 5 → 10)
        hoặc giới hạn max_depth.
BƯỚC 5. Nếu dữ liệu mất cân bằng: bật class_weight, hoặc dùng
        Balanced Random Forest, hoặc lấy mẫu lại (SMOTE).
BƯỚC 6. Xác nhận cuối bằng k-fold CV trên tập kiểm tra độc lập.
```

---

## 10. SO SÁNH VỚI CÁC THUẬT TOÁN KHÁC

### 10.1. Bảng so sánh tổng quan

| Tiêu chí                                   | Decision Tree | Random Forest         | Gradient Boosting (XGBoost)    | Logistic/Linear Regression | Neural Network       |
| -------------------------------------------- | ------------- | --------------------- | ------------------------------ | -------------------------- | -------------------- |
| **Độ chính xác (dữ liệu bảng)** | Thấp         | Cao                   | Rất cao                       | Trung bình                | Cao                  |
| **Khả năng diễn giải**             | Rất cao      | Thấp                 | Thấp                          | Rất cao                   | Rất thấp           |
| **Nguy cơ quá khớp**                | Rất cao      | Thấp                 | Trung bình (cần tinh chỉnh) | Thấp                      | Cao                  |
| **Tốc độ huấn luyện**             | Rất nhanh    | Nhanh (song song)     | Trung bình (tuần tự)        | Rất nhanh                 | Chậm                |
| **Tốc độ dự đoán**               | Rất nhanh    | Trung bình           | Trung bình                    | Rất nhanh                 | Trung bình          |
| **Cần chuẩn hóa dữ liệu**         | Không        | Không                | Không                         | **Có**              | **Có**        |
| **Xử lý phi tuyến**                 | Có           | Có                   | Có                            | Không                     | Có                  |
| **Khả năng ngoại suy**              | Không        | **Không**      | Không                         | **Có**              | Có                  |
| **Số siêu tham số cần dò**        | Ít           | **Ít**         | Nhiều                         | Rất ít                   | Rất nhiều          |
| **Song song hóa**                     | —            | **Hoàn toàn** | Hạn chế                      | —                         | GPU                  |
| **Dữ liệu phi cấu trúc**           | Kém          | Kém                  | Kém                           | Kém                       | **Xuất sắc** |

### 10.2. Random Forest vs Gradient Boosting — so sánh trọng tâm

| Khía cạnh                                 | Random Forest                                       | Gradient Boosting                                               |
| ------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| **Chiến lược ensemble**            | Song song, độc lập (**Bagging**)           | Tuần tự, cây sau sửa lỗi cây trước (**Boosting**) |
| **Mục tiêu chính**                 | Giảm**variance**                             | Giảm**bias**                                             |
| **Cây cơ sở**                      | Cây**sâu**, mạnh (low bias, high variance) | Cây**nông** (2–8 tầng), yếu (high bias)              |
| **Nhạy với nhiễu/outlier**         | Bền vững                                          | **Nhạy cảm** (dễ học theo nhiễu)                     |
| **Nhạy với siêu tham số**         | Thấp — chạy tốt "ngay lập tức"                | Cao — cần dò learning rate, depth, regularization            |
| **Rủi ro quá khớp khi tăng cây** | **Không**                                    | **Có** (cần early stopping)                             |
| **Khi nào chọn**                    | Baseline nhanh, dữ liệu nhiễu, cần ổn định   | Cần độ chính xác tối đa, có thời gian tinh chỉnh      |

**Kinh nghiệm thực chiến:** luôn chạy Random Forest **trước** làm baseline. Nếu
XGBoost/LightGBM không vượt được nó đáng kể sau khi tinh chỉnh, hãy giữ Random
Forest — đơn giản, ổn định và bền vững hơn trong vận hành.

### 10.3. Random Forest vs Extra Trees

| Khía cạnh                      | Random Forest                               | Extra Trees                                                                           |
| -------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Bootstrap**              | Có (mặc định)                           | Không (dùng toàn bộ tập)                                                         |
| **Chọn ngưỡng cắt**    | **Tối ưu** trên $m$ đặc trưng | **Ngẫu nhiên hoàn toàn**, chọn tốt nhất trong các ngưỡng ngẫu nhiên |
| **Bias / Variance**        | Bias thấp hơn, variance cao hơn          | Bias cao hơn, variance thấp hơn                                                    |
| **Tốc độ huấn luyện** | Chuẩn                                      | **Nhanh hơn nhiều** (không phải duyệt tìm ngưỡng tối ưu)              |
| **Hỗ trợ OOB**           | Có                                         | Không (mặc định, vì không bootstrap)                                            |

---

## 11. KẾT LUẬN

**Random Forest** là một trong những thuật toán **cân bằng nhất** giữa độ chính
xác, độ bền vững và chi phí sử dụng. Ba luận điểm cốt lõi cần ghi nhớ:

1. **Về mặt lý thuyết:** sức mạnh của Random Forest nằm gọn trong công thức
   $\operatorname{Var} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. Tăng $B$ chỉ
   xử lý được số hạng thứ hai; **giảm $\rho$ (bằng ngẫu nhiên hóa đặc trưng)**
   mới là chìa khóa hạ sàn phương sai. Cận trên $PE^* \le \bar{\rho}(1-s^2)/s^2$
   của Breiman khẳng định: mục tiêu thiết kế là **cây mạnh nhưng ít tương quan**.
2. **Về mặt thực hành:** đây là lựa chọn **mặc định hợp lý** cho mọi bài toán
   học có giám sát trên **dữ liệu dạng bảng**. Chạy tốt với tham số mặc định,
   không cần chuẩn hóa, chống quá khớp tự nhiên, và cung cấp OOB error cùng
   feature importance miễn phí.
3. **Về mặt giới hạn:** hai ràng buộc phải luôn ghi nhớ — **không ngoại suy
   được** (loại bỏ ứng dụng cho dữ liệu có xu hướng) và **khó diễn giải**
   (cần SHAP/LIME trong lĩnh vực bị quản lý). Với dữ liệu phi cấu trúc, hãy
   chuyển sang deep learning.

> **Nguyên tắc thực dụng:** Trong mọi dự án machine learning trên dữ liệu bảng,
> hãy chạy Random Forest **đầu tiên** để thiết lập baseline. Nếu mô hình phức
> tạp hơn không vượt qua được nó một cách thuyết phục, baseline chính là câu
> trả lời cuối cùng.

---

## 12. TÀI LIỆU THAM KHẢO

1. **Breiman, L.** (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
   — Bài báo gốc, chứa toàn bộ chứng minh lý thuyết về strength, correlation
   và cận trên sai số tổng quát hóa.
2. **Breiman, L.** (1996). *Bagging Predictors*. Machine Learning, 24(2), 123–140.
   — Nền tảng của kỹ thuật bagging.
3. **Breiman, L., Friedman, J., Olshen, R., Stone, C.** (1984).
   *Classification and Regression Trees (CART)*. Wadsworth.
   — Thuật toán xây cây cơ sở, chỉ số Gini.
4. **Ho, T. K.** (1998). *The Random Subspace Method for Constructing Decision
   Forests*. IEEE TPAMI, 20(8), 832–844.
   — Nguồn gốc của ý tưởng ngẫu nhiên hóa đặc trưng.
5. **Hastie, T., Tibshirani, R., Friedman, J.** (2009).
   *The Elements of Statistical Learning*, 2nd ed., Chương 15: Random Forests.
   — Phân tích toán học về công thức giảm phương sai.
6. **Geurts, P., Ernst, D., Wehenkel, L.** (2006).
   *Extremely Randomized Trees*. Machine Learning, 63(1), 3–42.
7. **Strobl, C., Boulesteix, A., Zeileis, A., Hothorn, T.** (2007).
   *Bias in Random Forest Variable Importance Measures*. BMC Bioinformatics, 8:25.
   — Phân tích chi tiết thiên lệch của MDI.
8. **Pedregosa, F. et al.** (2011). *Scikit-learn: Machine Learning in Python*.
   JMLR, 12, 2825–2830. — Tài liệu triển khai thực tế.

---

*Báo cáo được biên soạn phục vụ mục đích học thuật và tham chiếu kỹ thuật.*
