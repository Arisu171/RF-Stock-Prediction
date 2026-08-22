# =====================================================================
# Replay Engine — phát lại chuỗi thời gian như một luồng trực tiếp
# =====================================================================
# Module thuần Python, không phụ thuộc thư viện ngoài.
#
# ĐÂY KHÔNG PHẢI MỘT MẸO TRÌNH DIỄN. Việc phát lại từng phiên rồi chấm
# điểm ngay khi kết quả xuất hiện chính là KIỂM ĐỊNH TIẾN DẦN được
# trực quan hoá — cùng một phương pháp với splitter.generate_expanding_
# window_folds(), chỉ khác ở chỗ kết quả hiện ra dần thay vì tổng kết
# một lần ở cuối.
#
# BA TÍNH CHẤT BẮT BUỘC, xếp theo mức nguy hiểm nếu vi phạm:
#
#   1. KHÔNG NHÌN TRƯỚC. Ở bước t, engine chỉ được chạm dữ liệu tới t.
#      Vi phạm điều này thì toàn bộ màn hình trở nên vô nghĩa, mà triệu
#      chứng duy nhất là accuracy đẹp bất thường — rất khó phát hiện.
#
#   2. BỎ QUA GIAI ĐOẠN HUẤN LUYỆN. Phát lại trên chính dữ liệu mô hình
#      đã học thuộc sẽ cho accuracy cao giả tạo. Mặc định engine nhảy
#      thẳng tới sau mốc kết thúc huấn luyện ghi trong gói mô hình.
#
#   3. CHẤM ĐIỂM ĐÚNG ĐỘ TRỄ. Với tầm nhìn h phiên, dự đoán tại t chỉ
#      kiểm chứng được ở t+h. Vì vậy engine giữ một HÀNG ĐỢI CHỜ, không
#      chấm ngay.
#
# Ba đường accuracy chạy song song, và đường thứ ba mới là đường khiến
# hai đường kia có ý nghĩa:
#   - static   : mô hình đứng yên kể từ lúc huấn luyện
#   - adaptive : cùng mô hình đó nhưng tiếp tục thay máu
#   - baseline : luôn đoán lớp đa số của giai đoạn huấn luyện
#
# Thứ tự khai báo:
#
#   ①  Chuẩn bị chuỗi phát lại  — làm sạch, chốt mốc bắt đầu
#   ②  Bộ đếm accuracy          — cộng dồn theo từng lần chấm
#   ③  Lớp ReplayEngine         — vòng lặp chính, hàng đợi chờ
#   ④  Sinh sự kiện             — bộ sinh để tầng web đẩy ra luồng
# =====================================================================

import datetime

from . import featureBuilder
from . import timePreprocess


# ---------------------------------------------------------------------
# ① Chuẩn bị chuỗi — làm sạch một lần, sau đó chỉ cắt cửa sổ
# ---------------------------------------------------------------------
def prepare_replay_series(recipe, table):
    """
    Làm sạch bảng dữ liệu thô theo đúng công thức của gói mô hình.

    Chỉ làm sạch — KHÔNG dựng đặc trưng ở đây. Đặc trưng phải được dựng
    lại tại từng bước trên cửa sổ trượt, vì đó là cách duy nhất bảo đảm
    tính chất không nhìn trước.

    Parameters:
        recipe : dict công thức lấy từ gói mô hình
        table  : dict bảng dữ liệu thô

    Returns:
        table : dict bảng đã làm sạch, đã sắp xếp tăng dần theo thời gian
    """
    key_column = recipe['key_column']

    for name in recipe['numeric_columns']:
        table[name] = [
            None if value is None or value == '' else float(value)
            for value in table[name]
        ]
    table[key_column] = timePreprocess.parse_date_series(table[key_column])

    table = timePreprocess.sort_table_by_column(table, key_column)
    table = timePreprocess.remove_duplicate_keys(table, key_column)
    for name in recipe['numeric_columns']:
        table[name] = timePreprocess.forward_fill_series(
            table[name], recipe['preprocess'].get('max_forward_fill', 2)
        )
    table, _ = timePreprocess.drop_rows_with_missing(table)
    return table


def resolve_start_index(keys, start_after):
    """
    Tìm vị trí bắt đầu phát lại: quan sát đầu tiên SAU mốc chỉ định.

    Mốc này thường là ngày kết thúc giai đoạn huấn luyện. Phát lại từ
    trước mốc đó nghĩa là cho mô hình dự đoán chính dữ liệu nó đã học
    thuộc — accuracy sẽ cao giả tạo và toàn bộ màn hình mất ý nghĩa.

    Parameters:
        keys        : list mốc thời gian đã sắp xếp tăng dần
        start_after : datetime.date, hoặc chuỗi 'YYYY-MM-DD', hoặc None

    Returns:
        int chỉ số bắt đầu; 0 nếu không chỉ định mốc
    """
    if start_after is None:
        return 0
    if isinstance(start_after, str):
        start_after = datetime.datetime.strptime(
            start_after.strip(), '%Y-%m-%d').date()

    for index, key in enumerate(keys):
        if key is not None and key > start_after:
            return index
    return len(keys)


# ---------------------------------------------------------------------
# ② Bộ đếm accuracy — cộng dồn, không lưu toàn bộ lịch sử
# ---------------------------------------------------------------------
class RunningAccuracy:
    """
    Đếm số lần đúng trên tổng số lần đã chấm.

    Cố ý chỉ giữ hai số nguyên thay vì cả danh sách: bộ đếm này được
    tuần tự hoá và đẩy qua luồng ở MỌI bước, nên phải nhẹ.
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def record(self, is_correct):
        """Ghi nhận một lần chấm."""
        self.total += 1
        if is_correct:
            self.correct += 1

    @property
    def value(self):
        """Tỷ lệ đúng hiện tại, hoặc None khi chưa chấm lần nào."""
        if self.total == 0:
            return None
        return self.correct / self.total


# ---------------------------------------------------------------------
# ③ ReplayEngine — vòng lặp chính, giữ hàng đợi dự đoán chờ kiểm chứng
# ---------------------------------------------------------------------
class ReplayEngine:
    """
    Phát lại một chuỗi thời gian, dự đoán tại từng bước và chấm điểm
    ngay khi kết quả xuất hiện.

    Parameters:
        bundle          : gói mô hình đã nạp bằng modelBundle.load_bundle()
        table           : dict bảng dữ liệu thô
        adaptive_model  : rừng trượt (tuỳ chọn). None = chỉ chạy mô hình
                          tĩnh và mốc đối chứng.
        start_after     : mốc thời gian phải vượt qua mới bắt đầu phát.
                          None → lấy mốc kết thúc huấn luyện trong gói.
        window_size     : số dòng lịch sử dùng để dựng đặc trưng tại mỗi
                          bước. Phải lớn hơn cửa sổ chỉ báo dài nhất.
                          Dựng lại toàn bộ lịch sử mỗi bước chậm hơn
                          khoảng 40 lần mà không chính xác hơn.
    """

    def __init__(self, bundle, table, adaptive_model=None, start_after=None,
                 window_size=80):
        self.bundle = bundle
        self.recipe = bundle['recipe']
        self.static_model = bundle['estimator']
        self.adaptive_model = adaptive_model

        self.horizon = self.recipe['labeling'].get('horizon', 1)
        self.positive_label = self.recipe['labeling'].get('positive_label', 1)
        self.negative_label = self.recipe['labeling'].get('negative_label', 0)
        self.threshold = bundle.get('threshold', 0.5)
        self.window_size = max(window_size, 60)

        self.table = prepare_replay_series(self.recipe, table)
        self.keys = self.table[self.recipe['key_column']]
        self.price_series = self.table[self.recipe['series']['close']]

        if start_after is None:
            start_after = _training_end_of(bundle)
        self.start_index = resolve_start_index(self.keys, start_after)
        self.start_after = start_after

        self.baseline_label = self._resolve_baseline_label()

        self.accuracy = {
            'static':   RunningAccuracy(),
            'adaptive': RunningAccuracy(),
            'baseline': RunningAccuracy(),
        }
        self.pending = {}

    # ── Mốc đối chứng ───────────────────────────────────────────────

    def _resolve_baseline_label(self):
        """
        Lớp chiếm đa số, tính TRÊN GIAI ĐOẠN TRƯỚC khi phát lại.

        Dùng phần dữ liệu sau đó để chọn mốc đối chứng cũng là một dạng
        nhìn trước, dù tinh vi hơn.
        """
        increases = decreases = 0
        limit = min(self.start_index, len(self.price_series) - self.horizon)
        for index in range(max(0, limit - self.horizon)):
            current = self.price_series[index]
            future = self.price_series[index + self.horizon]
            if future > current:
                increases += 1
            elif future < current:
                decreases += 1
        return self.positive_label if increases > decreases else self.negative_label

    # ── Dựng đặc trưng cho ĐÚNG một bước ────────────────────────────

    def _build_sample_at(self, index):
        """
        Dựng vector đặc trưng cho quan sát tại vị trí index.

        Chỉ cắt dữ liệu tới index — đây là chỗ tính chất KHÔNG NHÌN
        TRƯỚC được bảo đảm, và cũng là chỗ dễ làm hỏng nhất nếu về sau
        có ai sửa thành `index + 1` cho "tiện".

        Returns:
            list giá trị đặc trưng, hoặc None nếu chưa đủ lịch sử
        """
        begin = max(0, index - self.window_size + 1)
        window = {
            name: values[begin:index + 1]
            for name, values in self.table.items()
        }

        source_series = {
            role: window[column] for role, column in self.recipe['series'].items()
        }
        try:
            feature_table, _ = featureBuilder.build_features(
                self.recipe['features'], source_series)
        except (ValueError, KeyError):
            return None

        names = self.bundle['feature_names']
        if sorted(feature_table) != names:
            raise ValueError(
                "Đặc trưng dựng ra không khớp với gói mô hình — kiểm tra "
                "lại công thức trong gói."
            )

        row = [feature_table[name][-1] for name in names]
        return None if any(value is None for value in row) else row

    # ── Chấm điểm một dự đoán đã tới hạn ────────────────────────────

    def _resolve_due_prediction(self, index):
        """
        Kiểm chứng dự đoán đặt ra cách đây `horizon` bước, nếu có.

        Trả về None khi chưa tới hạn, hoặc khi giá không đổi — quan sát
        đứng yên không thuộc lớp nào nên bị loại khỏi phép chấm, đúng
        như quy ước lúc huấn luyện.
        """
        source_index = index - self.horizon
        entry = self.pending.pop(source_index, None)
        if entry is None:
            return None

        past_price = self.price_series[source_index]
        current_price = self.price_series[index]
        if current_price == past_price:
            return None

        actual = (self.positive_label if current_price > past_price
                  else self.negative_label)

        outcome = {
            'key':          str(self.keys[source_index]),
            'resolved_key': str(self.keys[index]),
            'actual':       actual,
        }
        for name in ('static', 'adaptive'):
            prediction = entry['prediction'].get(name)
            if prediction is None:
                continue
            is_correct = prediction['label'] == actual
            self.accuracy[name].record(is_correct)
            outcome[f'{name}_hit'] = is_correct

        self.accuracy['baseline'].record(self.baseline_label == actual)
        outcome['baseline_hit'] = self.baseline_label == actual

        if self.adaptive_model is not None:
            self.adaptive_model.observe(entry['sample'], actual)

        return outcome

    # ── ④ Sinh sự kiện ──────────────────────────────────────────────

    def describe(self):
        """
        Thông tin mở đầu luồng: phạm vi dữ liệu, phần bị bỏ qua, mô hình.

        Returns:
            dict
        """
        skipped = self.start_index
        return {
            'type':            'meta',
            'label':           self.bundle.get('training_summary', {})
                                          .get('data_label', ''),
            'total_rows':      len(self.keys),
            'skipped_rows':    skipped,
            'skipped_reason':  (f"bỏ qua giai đoạn huấn luyện tới "
                                f"{self.start_after}") if skipped else '',
            'replay_rows':     max(0, len(self.keys) - skipped),
            'first_key':       str(self.keys[skipped]) if skipped < len(self.keys) else None,
            'last_key':        str(self.keys[-1]) if self.keys else None,
            'horizon':         self.horizon,
            'threshold':       self.threshold,
            'baseline_label':  self.baseline_label,
            'positive_label':  self.positive_label,
            'has_adaptive':    self.adaptive_model is not None,
            'model_size':      self.bundle.get('size', {}),
            'metrics':         self.bundle.get('metrics', {}),
        }

    def run(self):
        """
        Bộ sinh sự kiện phát lại, mỗi bước một sự kiện.

        Cấu trúc một sự kiện bước:
            { 'type': 'step', 'index', 'key', 'bar', 'prediction',
              'resolved', 'accuracy', 'adaptation' }

        Yields:
            dict sự kiện
        """
        for index in range(self.start_index, len(self.keys)):
            resolved = self._resolve_due_prediction(index)
            sample = self._build_sample_at(index)

            prediction = {}
            if sample is not None:
                prediction['static'] = self._predict_one(self.static_model, sample)
                if self.adaptive_model is not None:
                    prediction['adaptive'] = self._predict_one(
                        self.adaptive_model, sample)
                self.pending[index] = {'sample': sample, 'prediction': prediction}

            yield {
                'type':       'step',
                'index':      index,
                'key':        str(self.keys[index]),
                'bar':        self._bar_at(index),
                'prediction': prediction,
                'resolved':   resolved,
                'accuracy': {
                    name: counter.value
                    for name, counter in self.accuracy.items()
                },
                'resolved_count': self.accuracy['static'].total,
                'adaptation': (self.adaptive_model.describe_adaptation()
                               if self.adaptive_model is not None else None),
            }

    def _predict_one(self, model, sample):
        """Dự đoán cho một mẫu, trả về cả nhãn lẫn điểm số."""
        score = model.predict_scores([sample],
                                     positive_label=self.positive_label)[0]
        return {
            'label': self.positive_label if score >= self.threshold
                     else self.negative_label,
            'score': score,
        }

    def _bar_at(self, index):
        """Nến giá tại một vị trí, đặt tên theo VAI TRÒ chứ không theo cột."""
        return {
            role: self.table[column][index]
            for role, column in self.recipe['series'].items()
        }


# ---------------------------------------------------------------------
# Phép phụ dùng chung cho toàn module
# ---------------------------------------------------------------------
def _training_end_of(bundle):
    """
    Rút mốc kết thúc huấn luyện từ siêu dữ liệu của gói mô hình.

    Ưu tiên khoá 'training_end' nếu có; nếu không thì tách từ chuỗi
    'period' dạng 'BẮT_ĐẦU → KẾT_THÚC'. Trả về None khi không xác định
    được — lúc đó phát lại từ đầu, và người gọi phải tự chịu trách
    nhiệm chỉ định start_after.
    """
    summary = bundle.get('training_summary', {})
    if summary.get('training_end'):
        return summary['training_end']

    period = summary.get('period', '')
    if '→' in period:
        return period.split('→')[-1].strip()
    return None
