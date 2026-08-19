"""
Tool xuất báo cáo sản xuất ra file tải về.

Tương đương panel Export Report của tab Historical, nhưng gọi được bằng câu
chat. Toàn bộ đường đi nằm trong agent service: số liệu gom thẳng từ MongoDB
(`reports/data.py`), render bằng bản port của `reportGenerator.ts`
(`reports/html_report.py`), file ghi vào thư mục do chính service này phục vụ.
Không gọi sang backend :8000 ở bước nào.
"""

import logging
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent_app.reports import tabular
from agent_app.reports.data import (
    PERIOD_LABELS,
    build_summary,
    build_timeseries,
    resolve_period,
)
from agent_app.reports.html_report import THEMES, generate_html_report
from agent_app.tools.analytics_tools import (
    _disambiguation,
    _matching_recipes,
    _TZ,
)
from agent_app.tools.base_tool import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)

#: Thư mục file xuất ra. Nằm trong agent_service để service tự phục vụ được, và
#: để không lẫn vào `backend/uploads` của backend.
REPORTS_DIR = Path(__file__).resolve().parents[2] / "generated_reports"

#: Đường dẫn HTTP tương ứng, mount ở `main.py`.
REPORTS_URL_PREFIX = "/api/reports"

FORMATS = ("html", "pdf", "xlsx", "csv", "json")

#: Kỳ báo cáo đề xuất khi user chỉ nói "xuất báo cáo" mà không nêu thời gian.
#:
#: Mỗi lựa chọn mang luôn mốc gom số liệu hợp lý cho độ dài của nó, và nói ra
#: trong nhãn. Gộp hai câu hỏi thành một: hỏi riêng "kỳ nào" rồi "theo giờ hay
#: theo ngày" là bắt user bấm ba lần cho một việc, mà mốc thời gian thì gần như
#: luôn suy được từ độ dài kỳ.
PERIOD_CHOICES = [
    {"period": "today",     "granularity": "hour", "label": "Hôm nay",      "hint": "chia theo từng giờ"},
    {"period": "yesterday", "granularity": "hour", "label": "Hôm qua",      "hint": "chia theo từng giờ"},
    {"period": "thisweek",  "granularity": "day",  "label": "Tuần này",     "hint": "chia theo từng ngày"},
    {"period": "7days",     "granularity": "day",  "label": "7 ngày qua",   "hint": "chia theo từng ngày"},
    {"period": "thismonth", "granularity": "day",  "label": "Tháng này",    "hint": "chia theo từng ngày"},
    # 30 ngày để 'day' cho khớp quy tắc tự suy bên dưới (≤31 ngày → day). Nhãn
    # nút và mốc thực tế phải nói cùng một điều, không thì user bấm "chia theo
    # tuần" rồi mở file ra thấy chia theo ngày. 'week' vẫn dùng cho khoảng tuỳ ý
    # dài hơn một tháng.
    {"period": "30days",    "granularity": "day",  "label": "30 ngày qua",  "hint": "chia theo từng ngày"},
]

#: Nhãn hiển thị trên nút bấm khi phải hỏi lại user muốn định dạng nào.
FORMAT_CHOICES = [
    {"format": "html",  "label": "HTML",  "hint": "biểu đồ tương tác, mở bằng trình duyệt"},
    {"format": "pdf",   "label": "PDF",   "hint": "in được, gửi kèm email"},
    {"format": "xlsx",  "label": "Excel", "hint": "nhiều sheet, tự tính toán thêm được"},
    {"format": "csv",   "label": "CSV",   "hint": "bảng phẳng, nhập vào hệ thống khác"},
    {"format": "json",  "label": "JSON",  "hint": "dữ liệu thô cho lập trình"},
]

#: Giữ tối đa bấy nhiêu file, xoá dần từ cũ nhất.
#:
#: Có giới hạn ngay từ đầu là cố ý: thư mục `logs/` của chính dự án này đã phình
#: tới 1,4 GB vì phần lớn file nằm ngoài mọi chính sách dọn dẹp. Một thư mục
#: báo cáo không ai dọn sẽ đi đúng con đường đó, mà báo cáo thì tái tạo được
#: bất cứ lúc nào nên không có lý do gì phải giữ mãi.
MAX_KEPT_REPORTS = 40

#: Kỳ báo cáo ngắn thì mới tách theo camera. Phần đó phải `$unwind` mảng
#: `camera_results`, tức nạp trọn document 62,8 KB — 7 ngày mất ~52 s.
CAMERA_BREAKDOWN_MAX_DAYS = 2


class GenerateReportArgs(BaseModel):
    period: Optional[str] = Field(
        default=None,
        description="Kỳ báo cáo có sẵn: today, yesterday, thisweek, lastweek, 7days, "
                    "thismonth, lastmonth, 30days. ĐỂ TRỐNG (và không đưa start_date/"
                    "end_date) nếu user chưa nói rõ khoảng thời gian — tool sẽ hỏi lại. "
                    "Đừng tự đoán là 'hôm nay'.",
    )
    start_date: Optional[str] = Field(default=None, description="Ngày bắt đầu YYYY-MM-DD (ưu tiên hơn `period`)")
    end_date: Optional[str] = Field(default=None, description="Ngày kết thúc YYYY-MM-DD")
    granularity: Optional[str] = Field(
        default=None,
        description="Mốc gom số liệu: hour, day, week. ĐỂ TRỐNG để tool tự chọn theo độ dài "
                    "kỳ báo cáo (1-2 ngày → hour, tới 31 ngày → day, dài hơn → week). "
                    "Chỉ điền khi user nêu rõ ('theo giờ', 'từng ngày', 'theo tuần').",
    )
    recipe: Optional[str] = Field(
        default=None,
        description="Chỉ lấy một recipe (tên hoặc ObjectId). Bỏ trống = tất cả recipe.",
    )
    format: Optional[str] = Field(
        default=None,
        description="Định dạng file: html, pdf, xlsx, csv, json. "
                    "ĐỂ TRỐNG nếu user chưa nói rõ muốn định dạng nào — tool sẽ hỏi lại. "
                    "Chỉ điền khi user đã nêu đích danh ('dạng Excel', 'file PDF', 'xuất csv').",
    )
    theme: str = Field(default="industrial", description="Giao diện báo cáo: industrial, dark, executive")
    include_charts: bool = Field(default=True, description="Có kèm biểu đồ hay không (chỉ ảnh hưởng html/pdf)")
    include_per_recipe: bool = Field(default=True, description="Có mục chi tiết từng recipe hay không")


def _prune_old_reports() -> int:
    """Xoá file cũ nhất cho tới khi còn `MAX_KEPT_REPORTS`."""
    files = sorted(
        (p for p in REPORTS_DIR.glob("report_*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for p in files[MAX_KEPT_REPORTS:]:
        try:
            p.unlink()
            removed += 1
        except OSError as e:
            logger.warning("Không xoá được báo cáo cũ %s: %s", p.name, e)
    return removed


_GRAN_WORD = {"hour": "giờ", "day": "ngày", "week": "tuần"}


def _local_dates(start_dt: datetime, end_dt: datetime) -> tuple[str, str]:
    """
    Hai mốc naive-UTC → chuỗi 'YYYY-MM-DD' theo giờ địa phương.

    Nút bấm phải ghi ngày mà người dùng nhìn thấy. In thẳng mốc UTC ra sẽ lệch
    một ngày ở đầu kỳ (00:00 giờ VN là 17:00 hôm trước theo UTC).
    """
    from agent_app.reports.html_report import _to_local
    return (_to_local(start_dt).strftime("%Y-%m-%d"),
            _to_local(end_dt).strftime("%Y-%m-%d"))


def _slug(text: str) -> str:
    s = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_").lower()
    return s[:48] or "report"


def generate_report(
    period: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: Optional[str] = None,
    recipe: Optional[str] = None,
    format: Optional[str] = None,
    theme: str = "industrial",
    include_charts: bool = True,
    include_per_recipe: bool = True,
    **_ignored: Any,
) -> Dict[str, Any]:
    """
    Xuất báo cáo sản xuất ra file và trả về đường dẫn tải.

    `**_ignored` để bỏ qua tham số lạ. Đã gặp thật: phần mô tả tool nhắc tới các
    key trong kết quả trả về (`needs_period_choice`) và LLM tưởng đó là tham số
    nên truyền vào, làm cả lượt chat sập với "unexpected keyword argument". Một
    tham số bịa ra không được phép giết cả câu trả lời.
    """
    try:
        # Chưa biết kỳ báo cáo thì hỏi, đừng lặng lẽ lấy hôm nay. "Xuất báo cáo"
        # không hàm ý ngày nào cả, mà một file của hôm nay và một file của tháng
        # này là hai thứ hoàn toàn khác nhau.
        if not period and not start_date and not end_date:
            return {
                "success": False,
                "needs_period_choice": True,
                "message": (
                    "Chưa biết user muốn báo cáo kỳ nào. Hãy hỏi họ chọn, KHÔNG tự "
                    "mặc định là hôm nay. Mỗi lựa chọn đã kèm mốc chia số liệu phù hợp."
                ),
                "periods": [
                    {
                        **c,
                        "value": (f"Xuất báo cáo {c['label'].lower()} "
                                  f"gom theo {_GRAN_WORD[c['granularity']]}"),
                    }
                    for c in PERIOD_CHOICES
                ],
            }

        gran_raw = (granularity or "").strip().lower()
        if gran_raw and gran_raw not in ("hour", "day", "week"):
            return {"success": False, "error": "granularity phải là hour, day hoặc week"}

        th = (theme or "industrial").lower()
        if th not in THEMES:
            return {"success": False, "error": f"theme phải là một trong: {', '.join(THEMES)}"}

        if period and not (start_date or end_date) and period.lower() not in PERIOD_LABELS:
            return {"success": False,
                    "error": f"period '{period}' không hợp lệ. Hợp lệ: {', '.join(PERIOD_LABELS)}"}

        start_dt, end_dt, label = resolve_period(period, start_date, end_date)
        if start_dt > end_dt:
            return {"success": False, "error": "Ngày bắt đầu muộn hơn ngày kết thúc"}

        recipe_ids: List[str] = []
        if recipe:
            matches = _matching_recipes(recipe, start_dt, end_dt)
            if len(matches) > 1:
                # Tên khớp nhiều recipe — trả về danh sách để user chọn, chứ
                # không cộng gộp rồi xuất một file mà họ không biết gồm những gì.
                return _disambiguation(recipe, matches)
            if matches:
                recipe_ids = [matches[0]["recipe_id"]]
            else:
                return {
                    "success": False,
                    "error": f"Không có recipe nào khớp '{recipe}' trong kỳ {label}",
                }

        # Chưa biết định dạng thì DỪNG và hỏi, thay vì lặng lẽ xuất HTML.
        # Định dạng đổi hẳn thứ người dùng nhận được — file để in khác file để
        # tính toán tiếp — nên đây là lựa chọn của họ, không phải mặc định của
        # ta. Hỏi ở đây (sau khi đã chốt kỳ và recipe) để nút bấm mang theo đủ
        # ngữ cảnh, user không phải nhắc lại kỳ báo cáo lần thứ hai.
        raw_fmt = (format or "").strip().lower().lstrip(".")
        if raw_fmt in ("excel", "xls"):
            raw_fmt = "xlsx"
        if not raw_fmt:
            local = _local_dates(start_dt, end_dt)
            scope = f" cho recipe {recipe}" if recipe else ""
            return {
                "success": False,
                "needs_format_choice": True,
                "message": (
                    f"Chưa biết user muốn định dạng nào. Hãy hỏi họ chọn, "
                    f"KHÔNG được tự mặc định. Kỳ báo cáo đã chốt: {label} "
                    f"({local[0]} → {local[1]}){scope}."
                ),
                "period": {"label": label, "start": local[0], "end": local[1]},
                "formats": [
                    {
                        **c,
                        # `value` là nguyên văn câu sẽ gửi lại khi user bấm nút —
                        # kèm ngày cụ thể để lượt sau không phải suy luận lại
                        # "7 ngày qua" và ra một kỳ khác.
                        "value": (f"Xuất báo cáo từ {local[0]} đến {local[1]}{scope} "
                                  f"dạng {c['format']}"),
                    }
                    for c in FORMAT_CHOICES
                ],
            }
        if raw_fmt not in FORMATS:
            return {"success": False,
                    "error": f"Định dạng '{format}' không hỗ trợ. Hợp lệ: {', '.join(FORMATS)}"}
        fmt = raw_fmt

        span_days = (end_dt - start_dt).days + 1
        # Mốc gom số liệu suy từ độ dài kỳ nếu user không nêu. Để cứng 'day' như
        # trước thì báo cáo một ngày chỉ có đúng MỘT cột — biểu đồ vô nghĩa; còn
        # kỳ ba tháng chia theo ngày thì ra gần trăm cột chen chúc.
        gran = gran_raw or ("hour" if span_days <= 2 else "day" if span_days <= 31 else "week")

        summary = build_summary(
            start_dt, end_dt, recipe_ids or None,
            include_camera=(fmt in ("xlsx", "csv", "json") and span_days <= CAMERA_BREAKDOWN_MAX_DAYS),
        )
        if summary["total"] == 0:
            return {
                "success": False,
                "error": f"Không có dữ liệu sản xuất nào trong kỳ {label}",
                "period": {"start": start_dt.isoformat(), "end": end_dt.isoformat()},
            }

        timeseries = build_timeseries(start_dt, end_dt, gran, recipe_ids or None)

        cfg: Dict[str, Any] = {
            "periodLabel": label,
            "startDate": start_dt.isoformat(),
            "endDate": end_dt.isoformat(),
            "granularity": gran,
            "generatedAt": datetime.now(_TZ).replace(tzinfo=None).isoformat(),
            "theme": th,
            "sections": {
                "kpi": True,
                "trendChart": include_charts,
                "passfailChart": include_charts,
                "perRecipe": include_per_recipe,
            },
            "selectedRecipeIds": recipe_ids,
        }

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(_TZ).strftime("%Y%m%d_%H%M%S")
        name = f"report_{_slug(label)}_{stamp}.{fmt}"
        path = REPORTS_DIR / name

        t0 = time.time()
        if fmt == "html":
            path.write_text(generate_html_report(cfg, summary, timeseries), encoding="utf-8")
        elif fmt == "pdf":
            from agent_app.reports.charts_png import render_chart_images
            from agent_app.reports.pdf_report import pdf_css_overrides, render_pdf

            images = render_chart_images(timeseries, cfg, summary) if include_charts or include_per_recipe else {}
            html = generate_html_report(cfg, summary, timeseries, chart_images=images)
            # Chèn phần CSS riêng cho khổ giấy vào ngay trước </style> để nó ghi
            # đè các quy tắc chiều cao cố định của bản web.
            html = html.replace("</style>", pdf_css_overrides() + "</style>", 1)
            path.write_bytes(render_pdf(html))
        else:
            tables = tabular.build_tables(cfg, summary, timeseries)
            writer = {"xlsx": tabular.to_xlsx, "csv": tabular.to_csv, "json": tabular.to_json}[fmt]
            path.write_bytes(writer(tables))
        render_s = round(time.time() - t0, 2)

        pruned = _prune_old_reports()
        size_kb = round(path.stat().st_size / 1024, 1)

        return {
            "success": True,
            "download_url": f"{REPORTS_URL_PREFIX}/{name}",
            "filename": name,
            "format": fmt,
            "size_kb": size_kb,
            "render_seconds": render_s,
            "period": {"label": label, "start": start_dt.isoformat(), "end": end_dt.isoformat()},
            "granularity": gran,
            "granularity_auto": not gran_raw,
            "recipe_scope": (summary["by_recipe"][0]["recipe_name"]
                             if recipe_ids and summary["by_recipe"] else "Tất cả recipe"),
            "summary": {
                "total": summary["total"], "pass": summary["pass"],
                "fail": summary["fail"], "pass_rate": summary["pass_rate"],
                "recipes": len(summary["by_recipe"]),
                "buckets": len(timeseries["data"]),
            },
            "camera_breakdown_included": bool(summary.get("by_camera")),
            "old_reports_pruned": pruned,
            "note": (
                f"Đã tạo file xong. Nút tải đã được hệ thống gắn sẵn dưới câu trả lời — "
                f"ĐỪNG viết link, đừng dùng markdown ảnh hay liên kết. Chỉ cần nói "
                f"'bấm nút bên dưới để tải' kèm số liệu chính "
                f"({summary['total']:,} sản phẩm, tỷ lệ pass {summary['pass_rate']}%). "
                + ("Bảng tách theo camera bị bỏ vì kỳ báo cáo dài hơn "
                   f"{CAMERA_BREAKDOWN_MAX_DAYS} ngày — phần đó phải nạp trọn document nên rất chậm. "
                   if fmt in ("xlsx", "csv", "json") and not summary.get("by_camera") else "")
                + f"Chỉ giữ {MAX_KEPT_REPORTS} báo cáo gần nhất, file cũ hơn bị xoá tự động."
            ),
        }

    except Exception as e:
        logger.error(f"generate_report lỗi: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e), "message": "Không tạo được báo cáo"}


generate_report_tool = BaseTool.create_tool(
    func=generate_report,
    metadata=ToolMetadata(
        name="generate_report",
        description=(
            "Xuất BÁO CÁO SẢN XUẤT ra file tải về. Dùng khi user nói 'xuất báo cáo', "
            "'tạo report', 'xuất Excel', 'làm báo cáo PDF', 'gửi tôi file báo cáo'. "
            "Định dạng: html (biểu đồ tương tác), pdf (in được), xlsx (Excel nhiều "
            "sheet), csv, json. "
            "User chưa nói rõ KHOẢNG THỜI GIAN thì BỎ TRỐNG period/start_date/end_date; "
            "chưa nói rõ ĐỊNH DẠNG thì BỎ TRỐNG format. Tool sẽ tự hỏi lại user bằng "
            "nút bấm — đừng tự mặc định là hôm nay hay HTML. Để trống cả granularity, "
            "tool tự chọn theo độ dài kỳ. "
            "CHỈ truyền đúng các tham số có trong schema, không thêm tham số nào khác. "
            "Khi file đã tạo xong, nút tải được hệ thống gắn tự động — ĐỪNG tự viết "
            "link hay markdown, chỉ nói 'bấm nút bên dưới để tải'. "
            "KHÁC với get_pass_fail_stats/get_production_summary: những tool đó trả số "
            "liệu để trả lời trong chat, tool này tạo FILE. User chỉ hỏi số thì đừng "
            "gọi tool này."
        ),
        category="analytics",
    ),
    args_schema=GenerateReportArgs,
)

logger.info("✅ Report tools registered")
