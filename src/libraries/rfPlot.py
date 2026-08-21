# =====================================================================
# RF Plot — đồ thị cho cây quyết định, rừng ngẫu nhiên và boosting
# =====================================================================
# Mọi hàm chỉ nhận dãy số và nhãn hiển thị dạng chuỗi, không gắn với
# bài toán cụ thể. Cấu hình màu sắc, kích thước và việc lưu file được
# lấy từ utilities.plotStyle để toàn dự án dùng chung một phong cách.
#
# Quy ước chung:
#   - Hàm trả về đối tượng Figure để notebook có thể chỉnh thêm.
#   - Truyền filename để lưu ngay về thư mục output mặc định.
#
# Thứ tự khai báo bám theo trình tự đọc kết quả của một buổi thực nghiệm:
#
#   ①–②   Tầm quan trọng đặc trưng — mô hình đã dựa vào cái gì
#   ③–⑥   Hội tụ và ổn định        — đã đủ cây chưa, có quá khớp không
#   ⑦–⑨   Đánh giá phân loại       — ma trận nhầm lẫn, ROC, chọn ngưỡng
#   ⑩–⑫  Đánh giá hồi quy         — tán xạ, chuỗi thời gian, sai số
#   ⑬–⑭  Khảo sát dữ liệu         — dùng TRƯỚC khi huấn luyện
# =====================================================================

import matplotlib.pyplot as plt

from utilities import plotStyle


# ---------------------------------------------------------------------
# ① Xếp hạng đặc trưng — đồ thị đọc nhiều nhất sau khi huấn luyện xong
# ---------------------------------------------------------------------
def plot_feature_importances(feature_names, importance_values, top_k=15,
                             title='Tầm quan trọng đặc trưng',
                             filename=None, output_dir=None):
    """
    Biểu đồ cột ngang xếp hạng đặc trưng theo tầm quan trọng.

    Parameters:
        feature_names     : list tên đặc trưng
        importance_values : list giá trị tầm quan trọng, cùng độ dài
        top_k             : chỉ hiển thị top_k đặc trưng cao nhất
        title             : tiêu đề đồ thị
        filename          : tên file để lưu (None = không lưu)
        output_dir        : thư mục lưu (None = mặc định của plotStyle)

    Returns:
        matplotlib Figure
    """
    if len(feature_names) != len(importance_values):
        raise ValueError("Số tên đặc trưng phải bằng số giá trị tầm quan trọng.")

    ranked = sorted(
        zip(feature_names, importance_values),
        key=lambda pair: pair[1], reverse=True
    )[:top_k]
    ranked.reverse()

    names = [name for name, _ in ranked]
    values = [value for _, value in ranked]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(names) + 1)))
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    ax.barh(names, values, color=plotStyle.BAR_COLOR, alpha=0.85)
    ax.set_xlabel('Tầm quan trọng')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plotStyle.apply_axes_style(ax, grid_axis='x')

    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ② Đối chiếu hai cách đo — lộ thiên lệch MDI mà ① một mình không thấy
# ---------------------------------------------------------------------
def plot_importance_comparison(feature_names, first_values, second_values,
                               first_label='MDI',
                               second_label='Permutation',
                               top_k=15,
                               title='Đối chiếu hai cách đo tầm quan trọng',
                               filename=None, output_dir=None):
    """
    So sánh hai cách đo tầm quan trọng cạnh nhau trên cùng một trục.

    Hữu ích để phát hiện thiên lệch của MDI: đặc trưng có nhiều giá trị
    phân biệt thường được MDI cho điểm cao hơn thực chất, trong khi
    permutation importance không mắc lỗi này.

    Parameters:
        feature_names : list tên đặc trưng
        first_values  : list giá trị của cách đo thứ nhất
        second_values : list giá trị của cách đo thứ hai
        first_label   : nhãn chú giải cho cách đo thứ nhất
        second_label  : nhãn chú giải cho cách đo thứ hai
        top_k         : số đặc trưng hiển thị, xếp theo cách đo thứ nhất

    Returns:
        matplotlib Figure
    """
    ranked = sorted(
        zip(feature_names, first_values, second_values),
        key=lambda triple: triple[1], reverse=True
    )[:top_k]
    ranked.reverse()

    names = [name for name, _, _ in ranked]
    first = [value for _, value, _ in ranked]
    second = [value for _, _, value in ranked]
    positions = range(len(names))
    bar_height = 0.4

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(names) + 1)))
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    ax.barh([position + bar_height / 2 for position in positions], first,
            height=bar_height, label=first_label,
            color=plotStyle.BAR_COLOR, alpha=0.85)
    ax.barh([position - bar_height / 2 for position in positions], second,
            height=bar_height, label=second_label,
            color=plotStyle.POINT_COLOR, alpha=0.85)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(names)
    ax.set_xlabel('Tầm quan trọng (đã chuẩn hóa riêng từng cách đo)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    plotStyle.apply_axes_style(ax, grid_axis='x')

    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ③ Đường cong OOB — căn cứ chọn số cây, đọc trước khi chốt mô hình
# ---------------------------------------------------------------------
def plot_out_of_bag_curve(tree_counts, error_values,
                          title='Lỗi Out-Of-Bag theo số cây',
                          y_label='Lỗi OOB',
                          filename=None, output_dir=None):
    """
    Đường cong lỗi OOB theo số cây — dùng để chọn n_estimators.

    Điểm đường cong bắt đầu đi ngang là ngưỡng bão hòa: thêm cây nữa chỉ
    tốn thời gian mà không cải thiện kết quả.

    Parameters:
        tree_counts  : list số cây
        error_values : list lỗi tương ứng

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=plotStyle.FIG_SIZE_COMPACT)
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    ax.plot(tree_counts, error_values, marker='o', markersize=4,
            color=plotStyle.LOSS_COLOR, linewidth=2)

    best_position = min(
        range(len(error_values)), key=lambda index: error_values[index]
    )
    ax.axhline(error_values[best_position], color=plotStyle.NEUTRAL_COLOR,
               linestyle='--', linewidth=1,
               label=f'Thấp nhất: {error_values[best_position]:.4f} '
                     f'tại {tree_counts[best_position]} cây')

    ax.set_xlabel('Số cây trong rừng (B)')
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    plotStyle.apply_axes_style(ax)

    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ④ Đường mất mát — bản đối ứng của ③ cho boosting
# ---------------------------------------------------------------------
def plot_loss_history(loss_values, title='Hàm mất mát theo vòng lặp',
                      x_label='Vòng lặp', y_label='Mất mát',
                      filename=None, output_dir=None):
    """
    Đường mất mát theo vòng lặp huấn luyện — dùng cho boosting.

    Parameters:
        loss_values : list giá trị mất mát theo thứ tự vòng lặp

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=plotStyle.FIG_SIZE_COMPACT)
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    ax.plot(range(1, len(loss_values) + 1), loss_values,
            color=plotStyle.LOSS_COLOR, linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plotStyle.apply_axes_style(ax)

    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ⑤ Huấn luyện ↔ kiểm định — khoảng cách hai đường chính là mức quá khớp
# ---------------------------------------------------------------------
def plot_hyperparameter_curve(parameter_values, training_scores,
                              validation_scores,
                              parameter_label='Giá trị siêu tham số',
                              score_label='Điểm đánh giá',
                              title='Đường cong kiểm định siêu tham số',
                              filename=None, output_dir=None):
    """
    Điểm huấn luyện và điểm kiểm định theo một siêu tham số.

    Khoảng cách giữa hai đường chính là mức độ quá khớp: hai đường tách
    xa nhau nghĩa là mô hình học thuộc dữ liệu huấn luyện.

    Parameters:
        parameter_values  : list giá trị siêu tham số đã thử
        training_scores   : list điểm trên tập huấn luyện
        validation_scores : list điểm trên tập kiểm định

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=plotStyle.FIG_SIZE_COMPACT)
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    ax.plot(parameter_values, training_scores, marker='o', markersize=4,
            color=plotStyle.POINT_COLOR, linewidth=2, label='Huấn luyện')
    ax.plot(parameter_values, validation_scores, marker='s', markersize=4,
            color=plotStyle.LINE_COLOR, linewidth=2, label='Kiểm định')
    ax.set_xlabel(parameter_label)
    ax.set_ylabel(score_label)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    plotStyle.apply_axes_style(ax)

    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ⑥ Điểm từng vòng — đo độ ỔN ĐỊNH, thứ mà một con số trung bình che mất
# ---------------------------------------------------------------------
def plot_fold_scores(fold_labels, score_values, reference_value=None,
                     title='Điểm đánh giá qua từng vòng kiểm định',
                     y_label='Điểm đánh giá',
                     filename=None, output_dir=None):
    """
    Điểm đánh giá qua từng vòng kiểm định tiến dần theo thời gian.

    Độ phân tán lớn giữa các vòng cho thấy mô hình không ổn định khi
    điều kiện dữ liệu thay đổi.

    Parameters:
        fold_labels     : list nhãn hiển thị của từng vòng
        score_values    : list điểm tương ứng
        reference_value : đường tham chiếu nằm ngang (ví dụ mốc ngẫu nhiên)

    Returns:
        matplotlib Figure
    """
    average_score = sum(score_values) / len(score_values)

    fig, ax = plt.subplots(figsize=plotStyle.FIG_SIZE_COMPACT)
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    ax.bar(fold_labels, score_values, color=plotStyle.BAR_COLOR, alpha=0.85)
    ax.axhline(average_score, color=plotStyle.LINE_COLOR, linestyle='-',
               linewidth=1.5, label=f'Trung bình: {average_score:.4f}')
    if reference_value is not None:
        ax.axhline(reference_value, color=plotStyle.NEUTRAL_COLOR,
                   linestyle='--', linewidth=1.5,
                   label=f'Tham chiếu: {reference_value:.4f}')

    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    plotStyle.apply_axes_style(ax, grid_axis='y')

    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ⑦ Ma trận nhầm lẫn — điểm khởi đầu của mọi phân tích lỗi phân loại
# ---------------------------------------------------------------------
def plot_confusion_matrix(matrix, class_labels,
                          title='Ma trận nhầm lẫn',
                          normalize=False,
                          filename=None, output_dir=None):
    """
    Ma trận nhầm lẫn dạng bản đồ nhiệt kèm số liệu trên từng ô.

    Parameters:
        matrix       : list of lists — matrix[i][j] = số mẫu thuộc lớp i
                       nhưng được dự đoán là lớp j
        class_labels : list nhãn hiển thị của các lớp
        normalize    : True → hiển thị tỷ lệ theo từng dòng thay vì số đếm

    Returns:
        matplotlib Figure
    """
    display_matrix = matrix
    if normalize:
        display_matrix = []
        for row in matrix:
            row_total = sum(row)
            if row_total:
                display_matrix.append([value / row_total for value in row])
            else:
                display_matrix.append([0.0 for _ in row])

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    image = ax.imshow(display_matrix, cmap='Blues')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel('Nhãn dự đoán')
    ax.set_ylabel('Nhãn thực tế')
    ax.set_title(title, fontsize=13, fontweight='bold')

    highest = max(max(row) for row in display_matrix) if display_matrix else 0
    for row_index, row in enumerate(display_matrix):
        for column_index, value in enumerate(row):
            text = f"{value:.2f}" if normalize else f"{value}"
            ax.text(column_index, row_index, text, ha='center', va='center',
                    color='white' if value > highest / 2 else 'black',
                    fontweight='bold')

    ax.grid(False)
    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ⑧ Đường ROC — đánh giá độc lập với ngưỡng, bổ trợ cho ⑦
# ---------------------------------------------------------------------
def plot_roc_curve(false_positive_rates, true_positive_rates,
                   area_under_curve=None,
                   title='Đường cong ROC',
                   filename=None, output_dir=None):
    """
    Đường cong ROC kèm đường chéo tham chiếu của bộ phân loại ngẫu nhiên.

    Parameters:
        false_positive_rates : list tỷ lệ dương tính giả
        true_positive_rates  : list tỷ lệ dương tính thật
        area_under_curve     : giá trị AUC để ghi vào chú giải

    Returns:
        matplotlib Figure
    """
    label = 'Mô hình'
    if area_under_curve is not None:
        label = f'Mô hình (AUC = {area_under_curve:.4f})'

    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    ax.plot(false_positive_rates, true_positive_rates,
            color=plotStyle.LINE_COLOR, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], color=plotStyle.NEUTRAL_COLOR, linestyle='--',
            linewidth=1.5, label='Ngẫu nhiên (AUC = 0.5)')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel('Tỷ lệ dương tính giả (FPR)')
    ax.set_ylabel('Tỷ lệ dương tính thật (TPR)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    plotStyle.apply_axes_style(ax)

    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ⑨ Chỉ số theo ngưỡng — căn cứ rời bỏ ngưỡng mặc định 0.5
# ---------------------------------------------------------------------
def plot_threshold_curve(thresholds, metric_series,
                         title='Chỉ số theo ngưỡng quyết định',
                         filename=None, output_dir=None):
    """
    Diễn biến của nhiều chỉ số theo ngưỡng quyết định — dùng để chọn
    ngưỡng thay cho mặc định 0.5.

    Parameters:
        thresholds    : list ngưỡng đã thử
        metric_series : dict { tên chỉ số: list giá trị theo ngưỡng }

    Returns:
        matplotlib Figure
    """
    palette = [plotStyle.POINT_COLOR, plotStyle.LINE_COLOR,
               plotStyle.RESIDUAL_COLOR, plotStyle.LOSS_COLOR,
               plotStyle.BAR_COLOR]

    fig, ax = plt.subplots(figsize=plotStyle.FIG_SIZE_COMPACT)
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    for position, (name, values) in enumerate(metric_series.items()):
        ax.plot(thresholds, values, linewidth=2, label=name,
                color=palette[position % len(palette)])

    ax.set_xlabel('Ngưỡng quyết định')
    ax.set_ylabel('Giá trị chỉ số')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    plotStyle.apply_axes_style(ax)

    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ⑩ Tán xạ dự đoán ↔ thực tế — nhìn thấy ngay giới hạn không ngoại suy
# ---------------------------------------------------------------------
def plot_predicted_versus_actual(actual_values, predicted_values,
                                 title='Giá trị dự đoán so với thực tế',
                                 filename=None, output_dir=None):
    """
    Biểu đồ tán xạ dự đoán ↔ thực tế kèm đường chéo lý tưởng y = x.

    Điểm càng bám sát đường chéo, mô hình càng chính xác. Với mô hình
    dựa trên cây, đồ thị này thường lộ rõ hiện tượng không ngoại suy:
    dự đoán bị "chặn đầu" ở hai đầu dải giá trị.

    Returns:
        matplotlib Figure
    """
    lowest = min(min(actual_values), min(predicted_values))
    highest = max(max(actual_values), max(predicted_values))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    ax.scatter(actual_values, predicted_values, s=18, alpha=0.6,
               color=plotStyle.POINT_COLOR, edgecolors='none')
    ax.plot([lowest, highest], [lowest, highest],
            color=plotStyle.LINE_COLOR, linestyle='--', linewidth=1.5,
            label='Dự đoán hoàn hảo (y = x)')
    ax.set_xlabel('Giá trị thực tế')
    ax.set_ylabel('Giá trị dự đoán')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    plotStyle.apply_axes_style(ax)

    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ⑪ Hai chuỗi chồng nhau — cách đọc ⑩ theo trục thời gian
# ---------------------------------------------------------------------
def plot_series_comparison(actual_values, predicted_values, index_labels=None,
                           title='Chuỗi thực tế và chuỗi dự đoán',
                           x_label='Chỉ số quan sát', y_label='Giá trị',
                           max_ticks=12,
                           filename=None, output_dir=None):
    """
    Hai chuỗi thực tế và dự đoán vẽ chồng theo thứ tự quan sát.

    Parameters:
        actual_values    : list giá trị thực tế
        predicted_values : list giá trị dự đoán
        index_labels     : list nhãn trục hoành (None → dùng số thứ tự)
        max_ticks        : số vạch tối đa trên trục hoành

    Returns:
        matplotlib Figure
    """
    positions = list(range(len(actual_values)))

    fig, ax = plt.subplots(figsize=plotStyle.FIG_SIZE_COMPACT)
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    ax.plot(positions, actual_values, color=plotStyle.POINT_COLOR,
            linewidth=1.8, label='Thực tế')
    ax.plot(positions, predicted_values, color=plotStyle.LINE_COLOR,
            linewidth=1.5, alpha=0.85, label='Dự đoán')

    if index_labels is not None:
        step = max(1, len(index_labels) // max_ticks)
        tick_positions = positions[::step]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([index_labels[position] for position in tick_positions],
                           rotation=45, ha='right')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    plotStyle.apply_axes_style(ax)

    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ⑫ Phân bố sai số — sai số còn cấu trúc nghĩa là mô hình còn bỏ sót
# ---------------------------------------------------------------------
def plot_residuals(actual_values, predicted_values,
                   title='Phân bố sai số',
                   filename=None, output_dir=None):
    """
    Hai đồ thị sai số cạnh nhau: sai số theo thứ tự quan sát và biểu đồ
    tần suất của sai số.

    Sai số lý tưởng phân tán ngẫu nhiên quanh 0, không có cấu trúc.

    Returns:
        matplotlib Figure
    """
    residuals = [
        actual - predicted
        for actual, predicted in zip(actual_values, predicted_values)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)

    axes[0].scatter(range(len(residuals)), residuals, s=14, alpha=0.6,
                    color=plotStyle.RESIDUAL_COLOR, edgecolors='none')
    axes[0].axhline(0, color=plotStyle.NEUTRAL_COLOR, linestyle='--',
                    linewidth=1.5)
    axes[0].set_xlabel('Chỉ số quan sát')
    axes[0].set_ylabel('Sai số (thực tế - dự đoán)')
    axes[0].set_title('Sai số theo thứ tự quan sát')
    plotStyle.apply_axes_style(axes[0])

    axes[1].hist(residuals, bins=30, color=plotStyle.RESIDUAL_COLOR, alpha=0.8)
    axes[1].axvline(0, color=plotStyle.NEUTRAL_COLOR, linestyle='--',
                    linewidth=1.5)
    axes[1].set_xlabel('Sai số')
    axes[1].set_ylabel('Tần suất')
    axes[1].set_title('Phân bố sai số')
    plotStyle.apply_axes_style(axes[1])

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()

    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ⑬ Ma trận tương quan — dùng ở bước khảo sát, trước khi huấn luyện
# ---------------------------------------------------------------------
def plot_correlation_matrix(feature_names, correlation_matrix,
                            title='Ma trận tương quan',
                            show_values=False,
                            filename=None, output_dir=None):
    """
    Ma trận tương quan dạng bản đồ nhiệt với thang màu phân kỳ quanh 0.

    Parameters:
        feature_names      : list tên đặc trưng
        correlation_matrix : list of lists — hệ số tương quan trong [-1, 1]
        show_values        : True → ghi giá trị lên từng ô

    Returns:
        matplotlib Figure
    """
    size = max(6, 0.45 * len(feature_names) + 2)

    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    image = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title(title, fontsize=13, fontweight='bold')

    if show_values:
        for row_index, row in enumerate(correlation_matrix):
            for column_index, value in enumerate(row):
                ax.text(column_index, row_index, f"{value:.2f}",
                        ha='center', va='center', fontsize=6)

    ax.grid(False)
    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig


# ---------------------------------------------------------------------
# ⑭ Phân bố lớp — kiểm tra mất cân bằng, quyết định có cần cân bằng lại
# ---------------------------------------------------------------------
def plot_class_distribution(class_labels, counts,
                            title='Phân bố lớp',
                            filename=None, output_dir=None):
    """
    Biểu đồ cột số lượng mẫu theo từng lớp, kèm tỷ lệ phần trăm.

    Dùng để kiểm tra mức độ mất cân bằng trước khi huấn luyện.

    Parameters:
        class_labels : list nhãn hiển thị
        counts       : list số lượng mẫu tương ứng

    Returns:
        matplotlib Figure
    """
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    bars = ax.bar([str(label) for label in class_labels], counts,
                  color=plotStyle.BAR_COLOR, alpha=0.85)

    for bar, count in zip(bars, counts):
        percentage = 100.0 * count / total if total else 0.0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{count}\n({percentage:.1f}%)",
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Số mẫu')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.18 if counts else 1)
    plotStyle.apply_axes_style(ax, grid_axis='y')

    if filename:
        plotStyle.save_figure(fig, filename, output_dir)
    return fig
