# =====================================================================
# Conftest — cấu hình dùng chung cho toàn bộ bộ kiểm thử
# =====================================================================
# Đưa src/ vào đường dẫn tìm kiếm module để các test import được package
# của dự án mà không cần cài đặt dự án như một gói.
# =====================================================================

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
