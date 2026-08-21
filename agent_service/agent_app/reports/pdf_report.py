"""
Bản PDF của báo cáo.

WeasyPrint dựng PDF từ chính HTML/CSS đã port, nên bố cục và bảng màu khớp bản
HTML. Nó KHÔNG chạy JavaScript, vì vậy biểu đồ phải là ảnh PNG vẽ sẵn
(`charts_png.render_chart_images`) chứ không phải canvas Chart.js.

Vì sao chọn WeasyPrint chứ không phải trình duyệt headless: máy này là Jetson
aarch64 đang chạy inference cho dây chuyền, và không có sẵn chromium/chrome/
wkhtmltopdf. Cài một trình duyệt chỉ để in báo cáo là thêm hàng trăm MB và một
tiến trình ngốn RAM cạnh tranh với inference. WeasyPrint là thuần Python và các
thư viện hệ thống nó cần (cairo, pango) đã có trên máy.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def render_pdf(html: str) -> bytes:
    """HTML (đã nhúng ảnh biểu đồ) → bytes PDF."""
    # Import muộn: WeasyPrint nạp cairo/pango qua ctypes, mất khoảng một giây và
    # chỉ cần khi thực sự xuất PDF. Đưa lên đầu file thì mọi lần khởi động
    # service đều phải trả cái giá đó.
    from weasyprint import HTML

    return HTML(string=html).write_pdf()


def pdf_css_overrides() -> str:
    """
    Vài chỉnh riêng cho khổ giấy, nối sau REPORT_CSS.

    Bản web dùng chiều cao cố định cho khung biểu đồ (`.chart-wrap` 340px) vì
    canvas cần chỗ. Với ảnh tĩnh trên giấy thì để ảnh tự co theo bề rộng trang,
    không thì nó bị cắt hoặc chừa khoảng trắng lớn.
    """
    return """
    @page { size: A4 portrait; margin: 10mm 8mm; }
    body { background: #ffffff !important; }
    .container { max-width: 100%; padding: 0; }
    .chart-wrap, .chart-wrap-sm { height: auto; padding: 6px 0; }
    .chart-wrap img, .chart-wrap-sm img { width: 100%; height: auto; display: block; }
    .section, .recipe-section { page-break-inside: avoid; }
    .report-header { padding: 12px 14px; }
    """
