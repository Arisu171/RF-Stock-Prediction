# =====================================================================
# Plot Style — cấu hình hiển thị dùng chung cho mọi module vẽ đồ thị
# =====================================================================
# Tách riêng bảng màu, kích thước và tiện ích lưu file để các module
# vẽ đồ thị không lặp lại cấu hình.
#
# Thứ tự khai báo: hằng số cấu hình trước, hàm dùng chúng sau.
#
#   Hằng số  — bảng màu, kích thước, độ phân giải, thư mục lưu
#   ① Lưu figure     — nơi duy nhất ghi file ảnh ra đĩa
#   ② Áp style trục  — nơi duy nhất quyết định diện mạo một trục
#
# Gom hai việc này về một chỗ giúp mọi đồ thị của dự án trông như nhau
# mà không module vẽ nào phải lặp lại cấu hình.
# =====================================================================

import os


# ── Bảng màu ────────────────────────────────────────────────────────
POINT_COLOR      = '#4C72B0'   # xanh dương — điểm dữ liệu
LINE_COLOR       = '#DD4949'   # đỏ — đường/đường cong hồi quy
RESIDUAL_COLOR   = '#2CA02C'   # xanh lá — sai số
LOSS_COLOR       = '#FF8C00'   # cam — đường loss
BAR_COLOR        = '#7B68EE'   # tím — biểu đồ cột
HIGHLIGHT_COLOR  = '#D62728'   # đỏ đậm — điểm được chọn
NEUTRAL_COLOR    = '#888888'   # xám — đường tham chiếu
FIGURE_BG_COLOR  = '#F8F9FA'
AXES_BG_COLOR    = '#FFFFFF'

# ── Kích thước & độ phân giải ───────────────────────────────────────
FIG_SIZE_WIDE    = (14, 10)
FIG_SIZE_LARGE   = (16, 12)
FIG_SIZE_COMPACT = (11, 5)
DPI              = 120

# ── Thư mục lưu mặc định: <project_root>/data/output ────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output')


# ---------------------------------------------------------------------
# ① Lưu figure — nơi DUY NHẤT ghi ảnh ra đĩa, tự tạo thư mục nếu thiếu
# ---------------------------------------------------------------------
def save_figure(fig, filename, output_dir=None, dpi=DPI, verbose=True):
    """
    Lưu figure ra thư mục output, tự tạo thư mục nếu chưa tồn tại.

    Parameters:
        fig        : matplotlib Figure
        filename   : tên file (vd: 'regression.png') hoặc đường dẫn tuyệt đối
        output_dir : thư mục lưu (mặc định <project_root>/data/output)
        dpi        : độ phân giải
        verbose    : in đường dẫn file đã lưu

    Returns:
        filepath : đường dẫn file đã lưu
    """
    if os.path.isabs(filename):
        filepath = filename
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    else:
        output_dir = output_dir or DEFAULT_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    if verbose:
        print(f"Đã lưu đồ thị: {filepath}")
    return filepath


# ---------------------------------------------------------------------
# ② Áp style trục — nơi DUY NHẤT quyết định diện mạo chung của đồ thị
# ---------------------------------------------------------------------
def apply_axes_style(ax, grid_axis='both'):
    """
    Áp dụng style chung cho một axes: nền trắng, lưới mờ, ẩn viền trên/phải.
    """
    ax.set_facecolor(AXES_BG_COLOR)
    ax.grid(True, linestyle='--', alpha=0.4, axis=grid_axis)
    ax.spines[['top', 'right']].set_visible(False)
    return ax
