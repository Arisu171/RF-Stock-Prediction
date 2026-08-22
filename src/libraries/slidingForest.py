# =====================================================================
# Sliding Forest — rừng ngẫu nhiên biết già đi và thay máu
# =====================================================================
# Module thuần Python, không phụ thuộc thư viện ngoài.
#
# VẤN ĐỀ MÀ MODULE NÀY GIẢI QUYẾT. Một rừng đã huấn luyện là vật thể
# tĩnh: nó ghi lại quan hệ giữa đặc trưng và mục tiêu ĐÚNG TẠI thời
# điểm huấn luyện. Khi quan hệ đó đổi, mô hình không hề biết và cứ tiếp
# tục áp dụng luật cũ.
#
# Rừng trượt xử lý bằng cách LUÂN CHUYỂN cây: cứ sau mỗi update_every
# quan sát mới, nuôi thêm trees_per_update cây trên window_size quan
# sát gần nhất, đồng thời loại bỏ đúng chừng ấy cây GIÀ NHẤT. Tổng số
# cây không đổi, nhưng thành phần của rừng đổi dần theo thời gian.
#
# VÌ SAO KHÔNG HUẤN LUYỆN LẠI TOÀN BỘ. Huấn luyện lại tốn thời gian tỷ
# lệ với số cây, còn thay máu chỉ tốn tỷ lệ với trees_per_update. Quan
# trọng hơn: thay máu giữ được tính LIÊN TỤC — rừng luôn là hỗn hợp của
# nhiều giai đoạn, nên không quên đột ngột những gì đã học.
#
# ĐIỀU MODULE NÀY KHÔNG LÀM. Nó không tạo ra tín hiệu không tồn tại.
# Nếu quan hệ nền tảng phần lớn là nhiễu, thay máu chỉ giúp mô hình
# bám theo nhiễu mới. Giá trị nằm ở chỗ ĐO ĐƯỢC sự thoái hoá của mô
# hình tĩnh, không phải ở chỗ hứa hẹn dự báo tốt hơn.
#
# Thứ tự khai báo:
#
#   ①  Lớp rừng trượt      — bọc một rừng thường đã huấn luyện
#   ②  Ghi nhận quan sát   — nạp dữ liệu mới, tự kích hoạt ③ khi đủ
#   ③  Thay máu            — nuôi cây mới, loại cây già
#   ④  Nhật ký thích nghi  — theo dõi rừng đã đổi bao nhiêu
# =====================================================================

import random

from .randomForest import RandomForestClassifier, RandomForestRegressor


# ---------------------------------------------------------------------
# ① Khung rừng trượt — dùng chung cho phân loại và hồi quy
# ---------------------------------------------------------------------
class SlidingForestMixin:
    """
    Bổ sung khả năng thay máu cho một rừng đã huấn luyện.

    Thiết kế cố ý KHÔNG tạo ra một loại mô hình mới: rừng trượt lúc khởi
    tạo chính là rừng thường, dự đoán bằng đúng cơ chế cũ, lưu ra đĩa
    bằng đúng modelStore cũ. Nó chỉ có thêm hai phương thức observe() và
    apply_update().

    Nhờ vậy phép so sánh "tĩnh ↔ thích nghi" là chính xác tuyệt đối:
    hai mô hình xuất phát từ cùng một tập cây, chỉ khác ở chỗ một bên
    đứng yên còn một bên tiếp tục học.

    Parameters bổ sung so với rừng thường:
        trees_per_update : số cây nuôi mới mỗi lần thay máu; cũng là số
                           cây già bị loại, nên tổng số cây không đổi
        window_size      : số quan sát gần nhất dùng để nuôi cây mới
        update_every     : bao nhiêu quan sát mới thì thay máu một lần
    """

    def configure_sliding(self, trees_per_update=10, window_size=500,
                          update_every=25, retire_old=True):
        """
        Đặt tham số thích nghi và khởi tạo bộ đệm quan sát.

        Gọi sau khi rừng đã được huấn luyện hoặc nạp từ file.

        Parameters:
            retire_old : True  → thay máu, tổng số cây không đổi
                         False → chỉ THÊM cây mới, rừng lớn dần. Không
                         mất cây cũ nào, đổi lại chi phí dự đoán tăng
                         tuyến tính theo số cây.
        """
        if trees_per_update < 1:
            raise ValueError("trees_per_update phải ≥ 1.")
        if window_size < 2:
            raise ValueError("window_size phải ≥ 2.")
        if update_every < 1:
            raise ValueError("update_every phải ≥ 1.")
        if not self.trees:
            raise RuntimeError(
                "Rừng phải được huấn luyện hoặc nạp trước khi bật thay máu."
            )

        self.trees_per_update = trees_per_update
        self.window_size = window_size
        self.update_every = update_every
        self.retire_old = retire_old

        self.buffer_samples = []
        self.buffer_targets = []
        self.pending_count = 0
        self.update_log = []
        self.seed_generator = random.Random(self.random_state)
        return self

    @classmethod
    def from_forest(cls, forest, trees_per_update=10, window_size=500,
                    update_every=25, retire_old=True):
        """
        Bọc một rừng thường ĐÃ huấn luyện thành rừng trượt.

        Không sao chép cây — hai đối tượng sẽ dùng chung danh sách cây
        nếu truyền trực tiếp. Vì vậy hàm này tạo bản sao nông của danh
        sách, để rừng gốc giữ nguyên trạng thái tĩnh và dùng làm đối
        chứng được.

        Parameters:
            forest : RandomForestClassifier hoặc RandomForestRegressor

        Returns:
            đối tượng rừng trượt tương ứng
        """
        sliding = cls(
            n_estimators=forest.n_estimators,
            criterion=forest.criterion,
            max_depth=forest.max_depth,
            min_samples_split=forest.min_samples_split,
            min_samples_leaf=forest.min_samples_leaf,
            min_impurity_decrease=forest.min_impurity_decrease,
            max_features=forest.max_features,
            max_thresholds=forest.max_thresholds,
            bootstrap=forest.bootstrap,
            random_state=forest.random_state,
        )
        sliding.trees = list(forest.trees)
        sliding.out_of_bag_indices = [[] for _ in forest.trees]
        sliding.training_samples = []
        sliding.training_targets = []
        sliding.num_features = forest.num_features
        if hasattr(forest, 'label_space'):
            sliding.label_space = list(forest.label_space)

        return sliding.configure_sliding(
            trees_per_update, window_size, update_every, retire_old)

    def seed_buffer(self, samples, targets):
        """
        Nạp sẵn dữ liệu huấn luyện gốc vào bộ đệm.

        Không có bước này, cây mới luôn được nuôi trên ÍT dữ liệu hơn
        hẳn cây gốc, nên phép so sánh "tĩnh ↔ thích nghi" bị lẫn hai
        biến: tính gần đây và lượng dữ liệu. Nạp sẵn dữ liệu gốc tách
        được hai biến đó ra.

        Đánh đổi: gói mô hình phải mang theo tập huấn luyện, làm dung
        lượng tăng vọt. Vì vậy đây là lựa chọn của người gọi chứ không
        phải mặc định.

        Parameters:
            samples : list of lists — mẫu huấn luyện gốc
            targets : list nhãn tương ứng

        Returns:
            chính đối tượng rừng
        """
        self._assert_sliding_configured()
        self.buffer_samples = [list(sample) for sample in samples]
        self.buffer_targets = list(targets)
        if len(self.buffer_samples) > self.window_size:
            excess = len(self.buffer_samples) - self.window_size
            del self.buffer_samples[:excess]
            del self.buffer_targets[:excess]
        return self

    # ── ② Ghi nhận quan sát ─────────────────────────────────────────

    def observe(self, sample, target):
        """
        Nạp MỘT quan sát đã biết kết quả vào bộ đệm.

        LƯU Ý VỀ ĐỘ TRỄ: chỉ được gọi khi nhãn thực sự đã biết. Với tầm
        nhìn h phiên, tại thời điểm t ta chỉ biết nhãn của quan sát
        t − h. Rừng trượt vì thế luôn thích nghi CHẬM h bước so với
        hiện tại — đây là giới hạn của bài toán, không phải của cài đặt.

        Parameters:
            sample : list giá trị đặc trưng
            target : nhãn hoặc giá trị mục tiêu đã biết

        Returns:
            True nếu lần nạp này kích hoạt một đợt thay máu
        """
        self._assert_sliding_configured()

        self.buffer_samples.append(list(sample))
        self.buffer_targets.append(target)
        self.pending_count += 1

        if len(self.buffer_samples) > self.window_size:
            excess = len(self.buffer_samples) - self.window_size
            del self.buffer_samples[:excess]
            del self.buffer_targets[:excess]

        if self.pending_count < self.update_every:
            return False
        return self.apply_update()

    # ── ③ Thay máu ──────────────────────────────────────────────────

    def apply_update(self):
        """
        Nuôi cây mới trên bộ đệm rồi loại đúng chừng ấy cây già nhất.

        Bỏ qua (trả về False) khi bộ đệm chưa đủ đa dạng để dựng cây —
        ví dụ toàn bộ cửa sổ chỉ có một lớp. Thà giữ nguyên rừng cũ còn
        hơn nhét vào những cây chỉ biết đoán một phía.

        Returns:
            True nếu rừng thực sự đổi
        """
        self._assert_sliding_configured()

        if len(self.buffer_samples) < max(2, self.min_samples_leaf * 2):
            return False
        if not self._buffer_is_trainable():
            return False

        new_trees = []
        for _ in range(self.trees_per_update):
            seed = self.seed_generator.randrange(2 ** 31)
            sampler = random.Random(seed)

            if self.bootstrap:
                in_bag, _ = _bootstrap(len(self.buffer_samples), sampler)
            else:
                in_bag = list(range(len(self.buffer_samples)))

            tree = self._create_tree(seed)
            tree.fit(
                [self.buffer_samples[index] for index in in_bag],
                [self.buffer_targets[index] for index in in_bag],
            )
            new_trees.append(tree)

        retired = (min(len(new_trees), max(0, len(self.trees) - 1))
                   if self.retire_old else 0)
        self.trees = self.trees[retired:] + new_trees
        self.out_of_bag_indices = [[] for _ in self.trees]

        self.pending_count = 0
        self.update_log.append({
            'window_used':  len(self.buffer_samples),
            'trees_added':  len(new_trees),
            'trees_retired': retired,
            'forest_size':  len(self.trees),
        })
        return True

    def _buffer_is_trainable(self):
        """Bộ đệm có đủ đa dạng để dựng cây có ý nghĩa không."""
        return True

    # ── ④ Nhật ký thích nghi ────────────────────────────────────────

    def describe_adaptation(self):
        """
        Tóm tắt quá trình thay máu tính tới thời điểm hiện tại.

        Returns:
            dict { 'num_updates', 'trees_replaced', 'forest_size',
                   'buffer_size', 'pending_until_update' }
        """
        self._assert_sliding_configured()
        return {
            'num_updates':          len(self.update_log),
            'trees_replaced':       sum(item['trees_added']
                                        for item in self.update_log),
            'forest_size':          len(self.trees),
            'buffer_size':          len(self.buffer_samples),
            'pending_until_update': self.update_every - self.pending_count,
        }

    def _assert_sliding_configured(self):
        if not hasattr(self, 'update_every'):
            raise RuntimeError(
                "Chưa bật chế độ thay máu — gọi configure_sliding() hoặc "
                "dựng bằng from_forest()."
            )


# ---------------------------------------------------------------------
# Rừng trượt cho bài toán phân loại
# ---------------------------------------------------------------------
class SlidingRandomForestClassifier(SlidingForestMixin, RandomForestClassifier):
    """
    Rừng ngẫu nhiên phân loại có khả năng thay máu.

    Dự đoán, lưu trữ và mọi hành vi khác giống hệt lớp cha; điểm khác
    duy nhất là observe() và apply_update().
    """

    def _buffer_is_trainable(self):
        """Cửa sổ chỉ có một lớp thì không dựng được cây phân biệt gì."""
        return len(set(self.buffer_targets)) >= 2


# ---------------------------------------------------------------------
# Rừng trượt cho bài toán hồi quy
# ---------------------------------------------------------------------
class SlidingRandomForestRegressor(SlidingForestMixin, RandomForestRegressor):
    """
    Rừng ngẫu nhiên hồi quy có khả năng thay máu.

    Xem SlidingRandomForestClassifier; khác biệt duy nhất là điều kiện
    dựng cây — hồi quy chỉ cần mục tiêu không phải hằng số.
    """

    def _buffer_is_trainable(self):
        return len(set(self.buffer_targets)) >= 2


# ---------------------------------------------------------------------
# Phép phụ dùng chung cho toàn module
# ---------------------------------------------------------------------
def _bootstrap(num_samples, random_generator):
    """Rút mẫu bootstrap; tách riêng để không phải nạp cả rfMath."""
    in_bag = [random_generator.randrange(num_samples) for _ in range(num_samples)]
    selected = set(in_bag)
    return in_bag, [index for index in range(num_samples) if index not in selected]
