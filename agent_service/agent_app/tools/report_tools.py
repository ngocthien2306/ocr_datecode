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
                    "thismonth, lastmonth, 30days. Bỏ trống và không đưa ngày = hôm nay.",
    )
    start_date: Optional[str] = Field(default=None, description="Ngày bắt đầu YYYY-MM-DD (ưu tiên hơn `period`)")
    end_date: Optional[str] = Field(default=None, description="Ngày kết thúc YYYY-MM-DD")
    granularity: str = Field(default="day", description="Mốc gom số liệu: hour, day, week")
    recipe: Optional[str] = Field(
        default=None,
        description="Chỉ lấy một recipe (tên hoặc ObjectId). Bỏ trống = tất cả recipe.",
    )
    format: str = Field(
        default="html",
        description="Định dạng file: html (có biểu đồ tương tác), pdf (in được), "
                    "xlsx (Excel nhiều sheet), csv, json",
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


def _slug(text: str) -> str:
    s = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_").lower()
    return s[:48] or "report"


def generate_report(
    period: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: str = "day",
    recipe: Optional[str] = None,
    format: str = "html",
    theme: str = "industrial",
    include_charts: bool = True,
    include_per_recipe: bool = True,
) -> Dict[str, Any]:
    """Xuất báo cáo sản xuất ra file và trả về đường dẫn tải."""
    try:
        fmt = (format or "html").lower().lstrip(".")
        if fmt in ("excel", "xls"):
            fmt = "xlsx"
        if fmt not in FORMATS:
            return {"success": False,
                    "error": f"Định dạng '{format}' không hỗ trợ. Hợp lệ: {', '.join(FORMATS)}"}

        gran = (granularity or "day").lower()
        if gran not in ("hour", "day", "week"):
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

        span_days = (end_dt - start_dt).days + 1
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
                f"Đã tạo file. Hãy đưa `download_url` cho user dưới dạng liên kết tải về "
                f"và nói ngắn gọn số liệu chính ({summary['total']:,} sản phẩm, "
                f"tỷ lệ pass {summary['pass_rate']}%). "
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
            "sheet), csv, json. Trả về `download_url` — BẮT BUỘC đưa liên kết đó cho "
            "user, không có nó thì họ không lấy được file. "
            "KHÁC với get_pass_fail_stats/get_production_summary: những tool đó trả số "
            "liệu để trả lời trong chat, tool này tạo FILE. User chỉ hỏi số thì đừng "
            "gọi tool này."
        ),
        category="analytics",
    ),
    args_schema=GenerateReportArgs,
)

logger.info("✅ Report tools registered")
