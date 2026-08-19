"""
Biểu đồ dạng ảnh PNG cho bản PDF.

Bản HTML dùng Chart.js vẽ trên <canvas>, tức cần một trình duyệt chạy
JavaScript. WeasyPrint — thứ dựng PDF ở đây — chỉ hiểu HTML/CSS và không chạy
JavaScript, nên canvas sẽ ra trắng trơn. Máy này lại không có chromium headless
(đã kiểm tra: không có chromium, chrome, wkhtmltopdf), và cài thêm một trình
duyệt trên Jetson đang chạy inference là cái giá quá đắt cho việc xuất báo cáo.

matplotlib đã có sẵn trong môi trường, nên biểu đồ được vẽ sẵn thành PNG rồi
nhúng base64 vào chính template HTML đó. Kiểu biểu đồ, thứ tự dataset và bảng
màu bám theo `html_report.THEME_CHART_COLORS` để hai bản trông cùng một họ.
"""

import base64
import io
from typing import Any, Dict, List, Optional

import matplotlib
# Phải chọn backend TRƯỚC khi import pyplot: mặc định matplotlib thử mở màn hình,
# trong service chạy nền thì hỏng ngay.
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from agent_app.reports.html_report import (  # noqa: E402
    THEME_CHART_COLORS,
    THEME_PASS_FAIL,
    _fmt_label,
    _recipes_in,
)

# Nền và màu chữ theo từng theme, khớp biến CSS trong styles.py.
_THEME_CANVAS = {
    "industrial": {"bg": "#ffffff", "fg": "#374151", "grid": "#d1d5db"},
    "dark":       {"bg": "#1e293b", "fg": "#94a3b8", "grid": "#1e3a5f"},
    "executive":  {"bg": "#ffffff", "fg": "#1e293b", "grid": "#e2e8f0"},
}


def _hex(color: str) -> str:
    """Bỏ hậu tố alpha kiểu 'CC'/'55' của Chart.js — matplotlib không hiểu."""
    return color[:7] if len(color) in (9, 8) else color


def _alpha(color: str, default: float = 1.0) -> float:
    if len(color) in (8, 9):
        try:
            return int(color[-2:], 16) / 255
        except ValueError:
            return default
    return default


def _fig(theme: str, w: float, h: float):
    c = _THEME_CANVAS[theme]
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    fig.patch.set_facecolor(c["bg"])
    ax.set_facecolor(c["bg"])
    ax.tick_params(colors=c["fg"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(c["grid"])
    ax.grid(True, color=c["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    return fig, ax, c


def _to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _legend(ax, c: Dict[str, str], ncol: int) -> None:
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    ncol=min(ncol, 4), fontsize=7, frameon=False)
    for text in leg.get_texts():
        text.set_color(c["fg"])


def render_chart_images(
    timeseries: Dict[str, Any],
    cfg: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Toàn bộ biểu đồ của báo cáo, dạng {chart_id: data-URI}.

    `chart_id` trùng đúng id mà `html_report` chờ đợi (`chartTrend`,
    `chartPassFail`, `chartRecipe_{i}`), nên chỉ cần truyền dict này vào
    `generate_html_report(..., chart_images=...)` là biểu đồ vào đúng chỗ.
    """
    theme = cfg["theme"]
    sections = cfg["sections"]
    colors = THEME_CHART_COLORS[theme]
    pf = THEME_PASS_FAIL[theme]
    recipes = _recipes_in(timeseries, cfg, summary)
    points = timeseries["data"]
    labels = [_fmt_label(pt["timestamp"], cfg["granularity"]) for pt in points]
    out: Dict[str, str] = {}

    if not points or not recipes:
        return out

    def series(recipe_id: str, field: str) -> List[int]:
        vals = []
        for pt in points:
            r = next((x for x in pt["by_recipe"] if x["recipe_id"] == recipe_id), None)
            vals.append(r[field] if r else 0)
        return vals

    # Nhãn dày quá thì chỉ hiện thưa ra — 30 mốc giờ chồng chữ lên nhau đọc
    # không nổi, mà PDF không zoom được như trang web.
    step = max(1, len(labels) // 14)
    ticks = list(range(0, len(labels), step))
    tick_labels = [labels[i] for i in ticks]

    if sections["trendChart"]:
        fig, ax, c = _fig(theme, 9.5, 3.2)
        x = range(len(labels))
        for i, r in enumerate(recipes):
            ax.plot(x, series(r["id"], "total"), label=r["name"],
                    color=colors[i % len(colors)], linewidth=1.8, marker="o", markersize=3)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_ylim(bottom=0)
        _legend(ax, c, len(recipes))
        out["chartTrend"] = _to_data_uri(fig)

    if sections["passfailChart"]:
        fig, ax, c = _fig(theme, 9.5, 3.2)
        n = len(recipes)
        width = 0.8 / max(n, 1)
        for i, r in enumerate(recipes):
            base = _hex(colors[i % len(colors)])
            offs = [j - 0.4 + width * (i + 0.5) for j in range(len(labels))]
            p = series(r["id"], "pass")
            f = series(r["id"], "fail")
            ax.bar(offs, p, width=width, color=base, alpha=0.8, label=f"{r['name']} (Pass)")
            ax.bar(offs, f, width=width, bottom=p, color=base, alpha=0.35, label=f"{r['name']} (Fail)")
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_ylim(bottom=0)
        _legend(ax, c, n * 2)
        out["chartPassFail"] = _to_data_uri(fig)

    if sections["perRecipe"]:
        for i, r in enumerate(recipes):
            fig, ax, c = _fig(theme, 9.5, 1.9)
            x = list(range(len(labels)))
            p = series(r["id"], "pass")
            f = series(r["id"], "fail")
            ax.bar(x, p, color=_hex(pf["pass"]), alpha=_alpha(pf["pass"]), label="Pass")
            ax.bar(x, f, bottom=p, color=_hex(pf["fail"]), alpha=_alpha(pf["fail"]), label="Fail")
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
            ax.set_ylim(bottom=0)
            _legend(ax, c, 2)
            out[f"chartRecipe_{i}"] = _to_data_uri(fig)

    return out
