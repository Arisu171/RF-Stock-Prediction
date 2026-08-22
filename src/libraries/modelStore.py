# =====================================================================
# Model Store — tuần tự hoá cây và rừng thành cấu trúc lưu được
# =====================================================================
# Module thuần Python, không phụ thuộc thư viện ngoài.
#
# VÌ SAO CẦN TẦNG NÀY. Một mô hình đã huấn luyện là vật thể tĩnh: chỉ
# gồm các luật chia và giá trị tại lá. Nhưng nếu không ghi được ra đĩa
# thì mỗi lần dùng lại phải huấn luyện lại từ đầu — lúc đó thứ ta có
# chỉ là một THUẬT TOÁN, chưa phải một MÔ HÌNH.
#
# Module này chuyển đổi hai chiều giữa đối tượng trong bộ nhớ và cấu
# trúc dict lồng nhau (ghi thẳng ra JSON được).
#
# ĐIỀU CỐ Ý KHÔNG LƯU: tập mẫu huấn luyện và chỉ số out-of-bag. Chúng
# chiếm phần lớn dung lượng mà chỉ phục vụ việc đánh giá lúc huấn
# luyện. Hệ quả: mô hình nạp lại DỰ ĐOÁN được nhưng KHÔNG tính lại
# được lỗi OOB — calculate_out_of_bag_error() sẽ trả về None. Con số
# OOB đo lúc huấn luyện được lưu sẵn trong siêu dữ liệu của gói mô
# hình, nên không mất thông tin.
#
# Quy ước khoá viết tắt trong nút cây (để file gọn):
#   d = depth        n = num_samples    q = impurity
#   f = feature_index  t = threshold    g = impurity_decrease
#   L = nút con trái   R = nút con phải
#   v = giá trị dự đoán tại lá          p = vector xác suất tại lá
#
# Thứ tự khai báo:
#
#   ①  Nút → dict        — đơn vị nhỏ nhất, đệ quy xuống toàn cây
#   ②  dict → nút        — chiều ngược của ①
#   ③  Cây → dict        — bọc ① kèm siêu dữ liệu của cây
#   ④  dict → cây        — chiều ngược của ③
#   ⑤  Rừng → dict       — lặp ③ cho mọi cây kèm tham số của rừng
#   ⑥  dict → rừng       — chiều ngược của ⑤
#   ⑦  Đếm tham số       — số con số thực sự phải lưu
# =====================================================================

from .decisionTree import (DecisionTreeClassifier,
                           DecisionTreeRegressor, TreeNode)
from .randomForest import RandomForestClassifier, RandomForestRegressor

# Phiên bản định dạng — tăng lên khi cấu trúc thay đổi không tương thích.
FORMAT_VERSION = 1

TREE_CLASSES = {
    'classifier': DecisionTreeClassifier,
    'regressor':  DecisionTreeRegressor,
}
FOREST_CLASSES = {
    'classifier': RandomForestClassifier,
    'regressor':  RandomForestRegressor,
}


# ---------------------------------------------------------------------
# ① Nút → dict — đệ quy xuống hai nhánh, lá là điểm dừng
# ---------------------------------------------------------------------
def serialize_node(node):
    """
    Chuyển một nút và toàn bộ cây con bên dưới thành dict lồng nhau.

    Parameters:
        node : đối tượng TreeNode

    Returns:
        dict — nút lá không có khoá 'L' và 'R'
    """
    payload = {
        'd': node.depth,
        'n': node.num_samples,
        'q': node.impurity,
    }

    if node.is_leaf:
        payload['v'] = node.value
        if node.probabilities is not None:
            payload['p'] = list(node.probabilities)
        return payload

    payload['f'] = node.feature_index
    payload['t'] = node.threshold
    payload['g'] = node.impurity_decrease
    payload['L'] = serialize_node(node.left)
    payload['R'] = serialize_node(node.right)
    return payload


# ---------------------------------------------------------------------
# ② dict → nút — dựng lại đúng đối tượng mà ① đã tháo rời
# ---------------------------------------------------------------------
def deserialize_node(payload):
    """
    Dựng lại một nút và toàn bộ cây con bên dưới từ dict.

    Returns:
        đối tượng TreeNode
    """
    node = TreeNode(
        depth=payload['d'],
        num_samples=payload['n'],
        impurity=payload['q'],
    )

    if 'L' not in payload:
        node.value = payload['v']
        node.probabilities = payload.get('p')
        return node

    node.feature_index = payload['f']
    node.threshold = payload['t']
    node.impurity_decrease = payload['g']
    node.left = deserialize_node(payload['L'])
    node.right = deserialize_node(payload['R'])
    return node


# ---------------------------------------------------------------------
# ③ Cây → dict — thêm siêu dữ liệu để ④ dựng lại đúng loại cây
# ---------------------------------------------------------------------
def serialize_tree(tree):
    """
    Chuyển một cây quyết định đã huấn luyện thành dict.

    Ngoài cấu trúc nút, cần lưu thêm num_training_samples vì chỉ số tầm
    quan trọng theo MDI dùng nó làm mẫu số.

    Parameters:
        tree : DecisionTreeClassifier hoặc DecisionTreeRegressor đã fit

    Returns:
        dict
    """
    if tree.root is None:
        raise RuntimeError("Không thể lưu cây chưa được huấn luyện.")

    payload = {
        'task':            _resolve_task(tree),
        'criterion':       tree.criterion,
        'num_features':    tree.num_features,
        'num_training_samples': tree.num_training_samples,
        'root':            serialize_node(tree.root),
    }
    if isinstance(tree, DecisionTreeClassifier):
        payload['label_space'] = list(tree.label_space)
    return payload


# ---------------------------------------------------------------------
# ④ dict → cây — khôi phục đối tượng dùng ngay được để dự đoán
# ---------------------------------------------------------------------
def deserialize_tree(payload):
    """
    Dựng lại cây quyết định từ dict.

    Cây nạp lại chỉ giữ những gì cần cho dự đoán, đo tầm quan trọng và
    xuất văn bản. Các siêu tham số điều khiển việc DỰNG cây (max_depth,
    min_samples_leaf, …) không được khôi phục vì cây đã dựng xong.

    Returns:
        DecisionTreeClassifier hoặc DecisionTreeRegressor
    """
    tree_class = TREE_CLASSES[payload['task']]
    tree = tree_class(criterion=payload['criterion'])

    tree.root = deserialize_node(payload['root'])
    tree.num_features = payload['num_features']
    tree.num_training_samples = payload['num_training_samples']
    if 'label_space' in payload:
        tree.label_space = list(payload['label_space'])
    return tree


# ---------------------------------------------------------------------
# ⑤ Rừng → dict — lặp ③ cho mọi cây, KHÔNG kèm dữ liệu huấn luyện
# ---------------------------------------------------------------------
def serialize_forest(forest):
    """
    Chuyển một rừng đã huấn luyện thành dict.

    Các siêu tham số được lưu để tra cứu và để huấn luyện lại đúng cấu
    hình, nhưng bản thân việc dự đoán chỉ cần danh sách cây.

    Parameters:
        forest : RandomForestClassifier hoặc RandomForestRegressor đã fit

    Returns:
        dict
    """
    if not forest.trees:
        raise RuntimeError("Không thể lưu rừng chưa được huấn luyện.")

    payload = {
        'format_version': FORMAT_VERSION,
        'task':           _resolve_task(forest),
        'num_features':   forest.num_features,
        'settings': {
            'n_estimators':          forest.n_estimators,
            'criterion':             forest.criterion,
            'max_depth':             forest.max_depth,
            'min_samples_split':     forest.min_samples_split,
            'min_samples_leaf':      forest.min_samples_leaf,
            'min_impurity_decrease': forest.min_impurity_decrease,
            'max_features':          forest.max_features,
            'max_thresholds':        forest.max_thresholds,
            'bootstrap':             forest.bootstrap,
            'random_state':          forest.random_state,
        },
        'trees': [serialize_tree(tree) for tree in forest.trees],
    }
    if isinstance(forest, RandomForestClassifier):
        payload['label_space'] = list(forest.label_space)
    return payload


# ---------------------------------------------------------------------
# ⑥ dict → rừng — khôi phục rừng dùng ngay được để dự đoán
# ---------------------------------------------------------------------
def deserialize_forest(payload):
    """
    Dựng lại rừng từ dict.

    Rừng nạp lại KHÔNG giữ tập mẫu huấn luyện, nên
    calculate_out_of_bag_predictions() trả về hai danh sách rỗng và
    calculate_out_of_bag_error() trả về None. Đây là hành vi cố ý: thà
    trả về None còn hơn một con số tính từ dữ liệu rỗng.

    Returns:
        RandomForestClassifier hoặc RandomForestRegressor
    """
    version = payload.get('format_version')
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Định dạng mô hình phiên bản {version} không đọc được bằng "
            f"phiên bản {FORMAT_VERSION} của modelStore."
        )

    forest_class = FOREST_CLASSES[payload['task']]
    forest = forest_class(**payload['settings'])

    forest.trees = [deserialize_tree(item) for item in payload['trees']]
    forest.num_features = payload['num_features']
    forest.out_of_bag_indices = [[] for _ in forest.trees]
    forest.training_samples = []
    forest.training_targets = []
    if 'label_space' in payload:
        forest.label_space = list(payload['label_space'])
    return forest


# ---------------------------------------------------------------------
# ⑦ Đếm tham số — thước đo kích thước thật của mô hình
# ---------------------------------------------------------------------
def count_parameters(forest):
    """
    Đếm số con số phải lưu để tái tạo lại toàn bộ rừng.

    Quy ước đếm:
        - nút trong: 2 con số (chỉ số đặc trưng, ngưỡng chia)
        - nút lá   : 1 con số giá trị, cộng thêm vector xác suất nếu có

    Con số này đặt cạnh số mẫu huấn luyện cho biết mô hình có bị thừa
    tham số hay không: tỷ lệ tham số/mẫu vượt quá 1–2 nghĩa là mô hình
    đủ chỗ để ghi nhớ cả nhiễu.

    Returns:
        dict { 'num_trees', 'internal_nodes', 'leaf_nodes', 'parameters' }
    """
    internal = leaves = parameters = 0

    def walk(node):
        nonlocal internal, leaves, parameters
        if node.is_leaf:
            leaves += 1
            parameters += 1
            if node.probabilities is not None:
                parameters += len(node.probabilities)
        else:
            internal += 1
            parameters += 2
            walk(node.left)
            walk(node.right)

    for tree in forest.trees:
        walk(tree.root)

    return {
        'num_trees':      len(forest.trees),
        'internal_nodes': internal,
        'leaf_nodes':     leaves,
        'parameters':     parameters,
    }


# ---------------------------------------------------------------------
# Phép phụ dùng chung cho toàn module
# ---------------------------------------------------------------------
def _resolve_task(model):
    """Suy ra kiểu bài toán từ lớp của đối tượng."""
    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
        return 'classifier'
    if isinstance(model, (DecisionTreeRegressor, RandomForestRegressor)):
        return 'regressor'
    raise TypeError(
        f"Không nhận diện được kiểu bài toán của {type(model).__name__}."
    )
