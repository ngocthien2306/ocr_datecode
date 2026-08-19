"""
Bản HTML của báo cáo sản xuất — port từ
`frontend-ts/src/utils/reportGenerator.ts`.

Giữ nguyên cấu trúc DOM, tên class và bảng màu của bản TypeScript để báo cáo
xuất từ chat và báo cáo xuất từ panel Historical trông giống hệt nhau. CSS nằm
riêng ở `styles.py`, trích nguyên văn.

Khác bản gốc đúng hai chỗ, đều là bổ sung chứ không phải sửa:

  * hỗ trợ thêm mốc `week`;
  * biểu đồ có thể vẽ sẵn thành ảnh PNG thay cho canvas Chart.js, để bản PDF
    (render bằng WeasyPrint — không chạy JavaScript) vẫn có biểu đồ.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from agent_app.core.config import settings
from agent_app.reports.styles import REPORT_CSS

_TZ = ZoneInfo(settings.TIMEZONE)

THEME_CHART_COLORS: Dict[str, List[str]] = {
    "industrial": ['#2563eb', '#16a34a', '#dc2626', '#d97706', '#7c3aed',
                   '#0891b2', '#be185d', '#4d7c0f', '#374151', '#b45309'],
    "dark":       ['#38bdf8', '#4ade80', '#f87171', '#fbbf24', '#c084fc',
                   '#67e8f9', '#f472b6', '#a3e635', '#94a3b8', '#fb923c'],
    "executive":  ['#2563eb', '#0d9488', '#d97706', '#7c3aed', '#dc2626',
                   '#0891b2', '#be185d', '#65a30d', '#374151', '#b45309'],
}

THEME_PASS_FAIL: Dict[str, Dict[str, str]] = {
    "industrial": {"pass": '#16a34aCC', "fail": '#dc2626CC'},
    "dark":       {"pass": '#4ade80CC', "fail": '#f87171CC'},
    "executive":  {"pass": '#059669CC', "fail": '#dc2626CC'},
}

THEMES = tuple(THEME_CHART_COLORS)
SECTION_KEYS = ("kpi", "trendChart", "passfailChart", "perRecipe")


def _status_class(rate: float) -> str:
    return "st-ok" if rate >= 90 else "st-warn" if rate >= 70 else "st-fail"


def _status(rate: float) -> str:
    return "OK" if rate >= 90 else "WARN" if rate >= 70 else "FAIL"


def _n(v: Any) -> str:
    """Số có dấu phân cách nghìn, khớp `Number.toLocaleString()` của bản TS."""
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)


def _safe_json(obj: Any) -> str:
    """
    JSON nhúng được vào thẻ <script>.

    `</script>` nằm trong dữ liệu sẽ đóng thẻ sớm và làm hỏng cả trang, nên
    phải thoát — tên recipe do người dùng đặt, không có gì bảo đảm nó lành.
    """
    return json.dumps(obj, ensure_ascii=False).replace("</script>", "<\\/script>")


def _esc(s: Any) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _fmt_label(bucket: str, granularity: str) -> str:
    """
    Nhãn trục hoành.

    `data.py` đã gom mốc theo giờ địa phương ngay trong `$dateToString`, nên ở
    đây chỉ cần cắt chuỗi — KHÔNG được cộng thêm offset múi giờ lần nữa như bản
    TypeScript làm, vì bản đó nhận timestamp UTC thô từ API.
    """
    if granularity == "hour":
        return bucket[11:16] if len(bucket) >= 16 else bucket
    if granularity == "week":
        return bucket                      # '2026-W33'
    if len(bucket) >= 10:                  # '2026-08-19' → 'Aug 19'
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        try:
            _, m, d = bucket[:10].split("-")
            return f"{months[int(m) - 1]} {int(d)}"
        except (ValueError, IndexError):
            return bucket
    return bucket


def _gran_word(granularity: str) -> str:
    return {"hour": "Hour", "week": "Week"}.get(granularity, "Day")


# ── Các khối ─────────────────────────────────────────────────────────────────

def _to_local(dt: datetime) -> datetime:
    """
    naive-UTC (cách MongoDB lưu) → giờ địa phương, để hiển thị.

    `startDate`/`endDate` trong cfg là mốc naive-UTC: yêu cầu "từ 2026-08-01"
    thành `2026-07-31T17:00:00`. In thẳng ra thì header ghi 31/07 trong khi nhãn
    kỳ ghi 2026-08-01 — hai con số cạnh nhau tự bác nhau, và người đọc không
    biết báo cáo thực sự bắt đầu từ ngày nào.
    """
    return dt + timedelta(seconds=_TZ.utcoffset(dt).total_seconds())


def _header(cfg: Dict[str, Any]) -> str:
    def fmt(iso: str) -> str:
        try:
            d = _to_local(datetime.fromisoformat(iso))
            return f"{d.day:02d}/{d.month:02d}/{d.year}"
        except (ValueError, TypeError):
            return str(iso)[:10]

    gen = datetime.fromisoformat(cfg["generatedAt"]).strftime("%m/%d/%Y %H:%M:%S")
    return f"""
  <div class="report-header">
    <div>
      <div class="report-title">PRODUCTION INSPECTION REPORT</div>
      <div class="report-subtitle">
        Period: {_esc(cfg['periodLabel'])} &nbsp;|&nbsp; {fmt(cfg['startDate'])} – {fmt(cfg['endDate'])}<br>
        Breakdown: By {_gran_word(cfg['granularity'])}
      </div>
    </div>
    <div class="report-meta">
      Generated: {gen}<br>
      OCR Datecode System
    </div>
  </div>"""


def _kpi(summary: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    total, passed = summary["total"], summary["pass"]
    failed, rate = summary["fail"], summary["pass_rate"]
    picked = cfg.get("selectedRecipeIds") or []
    if picked:
        f = [r for r in summary["by_recipe"] if r["recipe_id"] in picked]
        total = sum(r["total"] for r in f)
        passed = sum(r["pass"] for r in f)
        failed = sum(r["fail"] for r in f)
        rate = round(passed / total * 1000) / 10 if total else 0

    pct = lambda v: round(v / total * 100) if total else 0  # noqa: E731
    return f"""
  <div class="section">
    <div class="section-title">Inspection Summary</div>
    <div class="kpi-row">
      <div class="kpi-cell">
        <div class="kpi-label">Total Inspected</div>
        <div class="kpi-value">{_n(total)}</div>
        <div class="kpi-sub">{_esc(cfg['periodLabel'])}</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Pass</div>
        <div class="kpi-value st-ok">{_n(passed)}</div>
        <div class="kpi-sub">{pct(passed)}% of total</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Fail</div>
        <div class="kpi-value st-fail">{_n(failed)}</div>
        <div class="kpi-sub">{pct(failed)}% of total</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Pass Rate</div>
        <div class="kpi-value {_status_class(rate)}">{rate}%</div>
        <div class="kpi-sub {_status_class(rate)}">{_status(rate)}</div>
      </div>
    </div>
  </div>"""


def _recipe_summary(summary: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    recipes = summary["by_recipe"]
    picked = cfg.get("selectedRecipeIds") or []
    if picked:
        recipes = [r for r in recipes if r["recipe_id"] in picked]
    if not recipes:
        return ""

    colors = THEME_CHART_COLORS[cfg["theme"]]
    rows = []
    for i, r in enumerate(recipes):
        color = colors[i % len(colors)]
        rate = r.get("pass_rate") or (round(r["pass"] / r["total"] * 1000) / 10 if r["total"] else 0)
        rows.append(f"""
      <tr>
        <td>
          <span style="display:inline-block;width:9px;height:9px;background:{color};margin-right:7px;vertical-align:middle"></span>
          {_esc(r['recipe_name'])}
        </td>
        <td class="num">{_n(r['total'])}</td>
        <td class="num st-ok">{_n(r['pass'])}</td>
        <td class="num st-fail">{_n(r['fail'])}</td>
        <td class="num {_status_class(rate)}">{rate}%</td>
        <td class="{_status_class(rate)}">{_status(rate)}</td>
      </tr>""")

    return f"""
  <div class="section">
    <div class="section-title">Recipe Summary</div>
    <table class="data-table">
      <thead>
        <tr>
          <th>Recipe</th>
          <th style="text-align:right">Total</th>
          <th style="text-align:right">Pass</th>
          <th style="text-align:right">Fail</th>
          <th style="text-align:right">Pass Rate</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>"""


def _combined_charts(images: Optional[Dict[str, str]]) -> str:
    """
    Hai biểu đồ tổng hợp.

    `images` có giá trị khi render cho PDF: khi đó dùng thẻ <img> với ảnh PNG
    nhúng sẵn, vì WeasyPrint không chạy JavaScript nên <canvas> sẽ trắng trơn.
    """
    def slot(chart_id: str) -> str:
        if images and chart_id in images:
            return f'<img src="{images[chart_id]}" style="width:100%;height:100%;object-fit:contain">'
        return f'<canvas id="{chart_id}"></canvas>'

    return f"""
  <div class="section">
    <div class="section-title">Production Trend – All Recipes</div>
    <div class="chart-wrap">{slot('chartTrend')}</div>
  </div>
  <div class="section">
    <div class="section-title">Pass / Fail Breakdown – All Recipes</div>
    <div class="chart-wrap">{slot('chartPassFail')}</div>
  </div>"""


def _recipes_in(
    timeseries: Dict[str, Any],
    cfg: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Các recipe có mặt trong kỳ, THEO ĐÚNG thứ tự của bảng Recipe Summary.

    Màu của recipe được lấy theo chỉ số trong danh sách này. Bản TypeScript gốc
    dựng danh sách theo thứ tự gặp đầu tiên khi quét timeseries, còn bảng
    Recipe Summary lại xếp theo tổng sản lượng giảm dần — hai thứ tự khác nhau
    nên cùng một recipe nhận hai màu khác nhau ở bảng và ở biểu đồ. Trên dữ liệu
    thật: bảng cho ONION POWDER màu xanh dương, chú giải biểu đồ lại gán màu đó
    cho Paprika. Người đọc đối chiếu ô màu với đường biểu đồ sẽ hiểu sai hẳn.

    Truyền `summary` vào để cả hai chỗ dùng chung một thứ tự.
    """
    seen: Dict[str, str] = {}
    for pt in timeseries["data"]:
        for r in pt["by_recipe"]:
            seen.setdefault(r["recipe_id"], r["recipe_name"])

    if summary:
        order = [r["recipe_id"] for r in summary["by_recipe"]]
        rank = {rid: i for i, rid in enumerate(order)}
        # Recipe chỉ có trong timeseries mà không có trong summary (không nên
        # xảy ra, nhưng đừng để nó biến mất khỏi báo cáo) xếp xuống cuối.
        ids = sorted(seen, key=lambda rid: rank.get(rid, len(order)))
    else:
        ids = list(seen)

    out = [{"id": rid, "name": seen[rid]} for rid in ids]
    picked = cfg.get("selectedRecipeIds") or []
    return [r for r in out if r["id"] in picked] if picked else out


def _per_recipe(timeseries: Dict[str, Any], summary: Dict[str, Any],
                cfg: Dict[str, Any], images: Optional[Dict[str, str]]) -> str:
    recipes = _recipes_in(timeseries, cfg, summary)
    if not recipes:
        return ""

    gran_label = "Hour" if cfg["granularity"] == "hour" else ("Week" if cfg["granularity"] == "week" else "Date")
    colors = THEME_CHART_COLORS[cfg["theme"]]
    out = []

    for idx, recipe in enumerate(recipes):
        rs = next((r for r in summary["by_recipe"] if r["recipe_id"] == recipe["id"]), None)
        tot = rs["total"] if rs else 0
        pas = rs["pass"] if rs else 0
        fai = rs["fail"] if rs else 0
        rat = (rs or {}).get("pass_rate") or (round(pas / tot * 1000) / 10 if tot else 0)
        col = colors[idx % len(colors)]

        sum_t = sum_p = sum_f = 0
        rows = []
        for pt in timeseries["data"]:
            r = next((x for x in pt["by_recipe"] if x["recipe_id"] == recipe["id"]), None)
            if not r or r["total"] == 0:
                continue
            sum_t += r["total"]; sum_p += r["pass"]; sum_f += r["fail"]
            rr = round(r["pass"] / r["total"] * 1000) / 10 if r["total"] else 0
            rows.append(f"""
        <tr>
          <td>{_esc(_fmt_label(pt['timestamp'], cfg['granularity']))}</td>
          <td class="num">{_n(r['total'])}</td>
          <td class="num st-ok">{_n(r['pass'])}</td>
          <td class="num st-fail">{_n(r['fail'])}</td>
          <td class="num {_status_class(rr)}">{rr}%</td>
          <td class="{_status_class(rr)}">{_status(rr)}</td>
        </tr>""")

        sum_rate = round(sum_p / sum_t * 1000) / 10 if sum_t else 0
        chart_id = f"chartRecipe_{idx}"
        if images and chart_id in images:
            chart_html = f'<img src="{images[chart_id]}" style="width:100%;height:100%;object-fit:contain">'
        else:
            chart_html = f'<canvas id="{chart_id}"></canvas>'

        tfoot = f"""
      <tfoot>
        <tr>
          <td>TOTAL</td>
          <td class="num">{_n(sum_t)}</td>
          <td class="num">{_n(sum_p)}</td>
          <td class="num">{_n(sum_f)}</td>
          <td class="num {_status_class(sum_rate)}">{sum_rate}%</td>
          <td class="{_status_class(sum_rate)}">{_status(sum_rate)}</td>
        </tr>
      </tfoot>""" if sum_t > 0 else ""

        body = "".join(rows) or ('<tr><td colspan="6" style="text-align:center;padding:16px;'
                                 'color:var(--text-muted)">No data for this period</td></tr>')

        out.append(f"""
  <div class="recipe-section" style="border-left:3px solid {col}">
    <div class="recipe-section-header">
      <span class="recipe-name">{_esc(recipe['name'])}</span>
      <div class="recipe-kpi-inline">
        <span>Total: <span class="rki-val">{_n(tot)}</span></span>
        <span class="st-ok">Pass: <span class="rki-val">{_n(pas)}</span></span>
        <span class="st-fail">Fail: <span class="rki-val">{_n(fai)}</span></span>
        <span class="{_status_class(rat)}">Rate: <span class="rki-val">{rat}%</span></span>
      </div>
    </div>
    <div class="chart-wrap-sm">{chart_html}</div>
    <table class="data-table">
      <thead>
        <tr>
          <th>{gran_label}</th>
          <th style="text-align:right">Total</th>
          <th style="text-align:right">Pass</th>
          <th style="text-align:right">Fail</th>
          <th style="text-align:right">Pass Rate</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>{tfoot}
    </table>
  </div>""")

    return "".join(out)


def _chart_scripts(timeseries: Dict[str, Any], cfg: Dict[str, Any],
                   summary: Optional[Dict[str, Any]] = None) -> str:
    recipes = _recipes_in(timeseries, cfg, summary)
    colors = THEME_CHART_COLORS[cfg["theme"]]
    pf = THEME_PASS_FAIL[cfg["theme"]]
    sections = cfg["sections"]
    labels = [_fmt_label(pt["timestamp"], cfg["granularity"]) for pt in timeseries["data"]]

    def series(recipe_id: str, field: str) -> List[int]:
        out = []
        for pt in timeseries["data"]:
            r = next((x for x in pt["by_recipe"] if x["recipe_id"] == recipe_id), None)
            out.append(r[field] if r else 0)
        return out

    trend_ds = [{
        "label": r["name"], "data": series(r["id"], "total"),
        "borderColor": colors[i % len(colors)],
        "backgroundColor": colors[i % len(colors)] + "20",
        "borderWidth": 2, "pointRadius": 4, "tension": 0.3, "fill": False,
    } for i, r in enumerate(recipes)]

    pf_ds: List[Dict[str, Any]] = []
    for i, r in enumerate(recipes):
        c = colors[i % len(colors)]
        pf_ds.append({"label": f"{r['name']} (Pass)", "data": series(r["id"], "pass"),
                      "backgroundColor": c + "CC", "stack": f"r{i}"})
        pf_ds.append({"label": f"{r['name']} (Fail)", "data": series(r["id"], "fail"),
                      "backgroundColor": c + "55", "stack": f"r{i}"})

    per_recipe = "\n".join(
        f"""
    initBarChart('chartRecipe_{i}', LABELS, [
      {{ label: 'Pass', data: {_safe_json(series(r['id'], 'pass'))}, backgroundColor: '{pf['pass']}', stack: 's0' }},
      {{ label: 'Fail', data: {_safe_json(series(r['id'], 'fail'))}, backgroundColor: '{pf['fail']}', stack: 's0' }}
    ]);"""
        for i, r in enumerate(recipes)
    )

    dark = cfg["theme"] == "dark"
    grid_c = "#1e3a5f" if dark else "#e5e7eb"
    tick_c = "#94a3b8" if dark else "#374151"
    legend_c = "#e2e8f0" if dark else "#111"
    tooltip_bg = "#0f172a" if dark else "#1e2a3a"
    base_font = "{ family: \"'Segoe UI', Arial, sans-serif\", size: 11 }"

    def opts(stacked: bool) -> str:
        st = "stacked:true, " if stacked else ""
        return f"""{{
    responsive:true, maintainAspectRatio:false,
    interaction:{{ mode:'index', intersect:false }},
    plugins:{{
      legend:{{ position:'bottom', labels:{{ padding:14, font:{base_font}, color:'{legend_c}' }}}},
      tooltip:{{ backgroundColor:'{tooltip_bg}', padding:10, titleFont:{base_font}, bodyFont:{base_font} }}
    }},
    scales:{{
      x:{{ {st}grid:{{ color:'{grid_c}' }}, ticks:{{ font:{base_font}, color:'{tick_c}', maxRotation:45 }}, border:{{ color:'{tick_c}50' }}}},
      y:{{ {st}grid:{{ color:'{grid_c}' }}, ticks:{{ font:{base_font}, color:'{tick_c}' }}, beginAtZero:true, border:{{ color:'{tick_c}50' }}}}
    }}
  }}"""

    init_trend = f"initLineChart('chartTrend',    LABELS, {_safe_json(trend_ds)});" if sections["trendChart"] else ""
    init_pf = f"initBarChart('chartPassFail', LABELS, {_safe_json(pf_ds)});" if sections["passfailChart"] else ""

    return f"""
  <script>
    var LABELS = {_safe_json(labels)};
    function initLineChart(id, labels, datasets) {{
      var el = document.getElementById(id); if (!el) return;
      new Chart(el, {{ type:'line', data:{{ labels:labels, datasets:datasets }}, options:{opts(False)} }});
    }}
    function initBarChart(id, labels, datasets) {{
      var el = document.getElementById(id); if (!el) return;
      new Chart(el, {{ type:'bar', data:{{ labels:labels, datasets:datasets }}, options:{opts(True)} }});
    }}
    document.addEventListener('DOMContentLoaded', function() {{
      {init_trend}
      {init_pf}
      {per_recipe if sections['perRecipe'] else ''}
    }});
  </script>"""


def generate_html_report(
    cfg: Dict[str, Any],
    summary: Dict[str, Any],
    timeseries: Dict[str, Any],
    chart_images: Optional[Dict[str, str]] = None,
) -> str:
    """
    Dựng trang HTML hoàn chỉnh.

    `chart_images` là dict {chart_id: 'data:image/png;base64,...'}. Truyền vào
    thì biểu đồ là ảnh tĩnh và trang KHÔNG nạp Chart.js — đó là chế độ dùng cho
    PDF. Bỏ trống thì biểu đồ là canvas Chart.js tương tác như bản FE.
    """
    sections = cfg["sections"]
    has_charts = sections["trendChart"] or sections["passfailChart"] or sections["perRecipe"]
    static_mode = chart_images is not None

    chart_lib = ("" if static_mode or not has_charts else
                 '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>')
    scripts = "" if static_mode or not has_charts else _chart_scripts(timeseries, cfg, summary)
    generated = datetime.fromisoformat(cfg["generatedAt"]).strftime("%m/%d/%Y, %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Production Report – {_esc(cfg['periodLabel'])}</title>
  {chart_lib}
  <style>{REPORT_CSS}</style>
</head>
<body class="theme-{cfg['theme']}">
  <div class="container">
    {_header(cfg)}
    {_kpi(summary, cfg) if sections['kpi'] else ''}
    {_recipe_summary(summary, cfg) if sections['kpi'] else ''}
    {_combined_charts(chart_images) if (sections['trendChart'] or sections['passfailChart']) else ''}
    {_per_recipe(timeseries, summary, cfg, chart_images) if sections['perRecipe'] else ''}
    <div class="report-footer">
      <span>OCR Datecode Inspection System</span>
      <span>Generated: {generated} &nbsp;|&nbsp; {_esc(cfg['periodLabel'])}</span>
    </div>
  </div>
  {scripts}
</body>
</html>"""
