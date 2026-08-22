# =====================================================================
# Dockerfile — đóng gói máy chủ dự đoán và bảng điều khiển phát lại
# =====================================================================
# Build:
#   docker build -t rf-stock .
#
# Chạy:
#   docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" rf-stock
#
# Ảnh CHỈ chứa phần cần để chạy máy chủ. Toàn bộ nhóm phân tích —
# matplotlib, jupyter, numpy, pandas, scikit-learn — bị bỏ ra ngoài, vì
# lõi học máy trong src/ là Python thuần và không đụng tới chúng. Nhờ
# vậy ảnh chỉ cần 4 gói trực tiếp thay vì hơn một trăm.
#
# Notebook, tài liệu và bộ kiểm thử cũng không vào ảnh — xem .dockerignore.
# =====================================================================

FROM python:3.13-slim

# ---------------------------------------------------------------------
# ① Biến môi trường — đặt trước mọi thứ để áp dụng cho cả bước build
# ---------------------------------------------------------------------
# PYTHONDONTWRITEBYTECODE : không sinh .pyc, ảnh gọn hơn và không có
#                           file thừa lẫn vào lớp cuối
# PYTHONUNBUFFERED        : log hiện ra ngay thay vì bị giữ trong đệm —
#                           bắt buộc, nếu không sẽ không thấy gì khi
#                           container gặp sự cố
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---------------------------------------------------------------------
# ② Phụ thuộc — copy RIÊNG và cài trước khi copy mã nguồn
# ---------------------------------------------------------------------
# Tách thành lớp riêng có chủ ý: sửa mã nguồn thì Docker chỉ dựng lại
# từ bước ③ trở đi, không phải cài lại toàn bộ gói.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# ---------------------------------------------------------------------
# ③ Mã nguồn và tài nguyên
# ---------------------------------------------------------------------
# src/    — lõi học máy, thuần Python
# api/    — tầng web, chỉ import vào src/ một chiều
# config/ — bản đặc tả đặc trưng và siêu tham số
COPY src/    ./src/
COPY api/    ./api/
COPY config/ ./config/

# models/ và data/input/ nằm trong .gitignore nên có thể RỖNG khi build
# từ một bản sao mới. Ảnh vẫn chạy được: /api/models sẽ trả về thông báo
# yêu cầu huấn luyện trước, và người dùng có thể gắn thư mục thật vào
# bằng -v lúc chạy. Xem docker-compose.yml.
COPY models/     ./models/
COPY data/input/ ./data/input/

# ---------------------------------------------------------------------
# ④ Người dùng không đặc quyền — không có lý do gì chạy máy chủ bằng root
# ---------------------------------------------------------------------
RUN useradd --create-home --shell /bin/bash service \
    && chown -R service:service /app
USER service

# ---------------------------------------------------------------------
# ⑤ Cổng, kiểm tra sức khoẻ và lệnh khởi động
# ---------------------------------------------------------------------
EXPOSE 8000

# Gọi /api/datasets thay vì /api/models: route này luôn trả 200 kể cả
# khi chưa có gói mô hình nào, nên nó đo đúng thứ cần đo — máy chủ còn
# sống hay không — chứ không lẫn với việc thiếu dữ liệu.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/datasets', timeout=4)"

# Bind 0.0.0.0 chứ không phải 127.0.0.1: trong container, địa chỉ vòng
# lặp chỉ nghe được từ chính container đó.
CMD ["python", "-m", "uvicorn", "api.main:application", \
     "--host", "0.0.0.0", "--port", "8000"]
