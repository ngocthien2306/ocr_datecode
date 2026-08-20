"""
Dựng báo cáo SO SÁNH nhiều máy: HTML, PDF, Excel, CSV.

Khác với `generate_report` ở edge — cái đó là báo cáo của MỘT máy. Ở đây nội dung
chính là phần so sánh, nên không thể gom năm file của năm máy lại là xong.

**Biểu đồ vẽ sẵn thành PNG bằng matplotlib rồi nhúng base64**, không dùng
Chart.js. Cùng lý do đã ghi trong `PIPELINE.md §4`: WeasyPrint chỉ hiểu HTML/CSS,
không chạy JavaScript, nên `<canvas>` sẽ ra trắng trơn trong bản PDF. Vẽ sẵn thì
một template dùng được cho cả HTML lẫn PDF, và file HTML cũng tự chứa — mở offline
vẫn thấy biểu đồ.
"""

from __future__ import annotations

import base64
import csv
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parents[2] / "generated_reports"
OUT_DIR.mkdir(exist_ok=True)

FORMATS = ("html", "pdf", "excel", "csv")
GRANULARITIES = ("day", "week")

# Bảng màu cố định theo TÊN MÁY, không theo thứ tự trong danh sách. Gán theo thứ
# tự thì bỏ một máy ra khỏi báo cáo là mọi máy còn lại đổi màu, và hai bản báo cáo
# cạnh nhau không đọc chéo được.
_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#7c3aed",
            "#0891b2", "#db2777", "#65a30d"]


def _color_for(name: str) -> str:
    return _PALETTE[sum(ord(c) for c in name) % len(_PALETTE)]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def _chart_output(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Sản lượng mỗi ngày — chuẩn hoá, vì các máy chạy số ngày khác nhau."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [(r["machine"], (r.get("production") or {}).get("per_day"))
            for r in rows if (r.get("production") or {}).get("per_day")]
    if not data:
        return None
    names = [d[0] for d in data]
    vals = [d[1] for d in data]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.bar(names, vals, color=[_color_for(n) for n in names], width=.55)
    ax.set_ylabel("sản phẩm / ngày")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.25)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    return _png(fig)


def _chart_fingerprint(fp: Dict[str, Any]) -> Optional[str]:
    """
    Vân tay kiểu lỗi, cột chồng ngang.

    Cột chồng chứ không phải nhóm cạnh nhau: mỗi máy cộng lại đúng 100%, nên hình
    dạng của cả thanh chính là "vân tay". Để cạnh nhau thì mắt so từng cặp cột và
    mất luôn cái nhìn tổng thể.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    causes = fp.get("causes") or []
    by = fp.get("by_machine") or {}
    if not causes or not by:
        return None
    labels = fp.get("cause_labels") or {}
    names = list(by.keys())
    fig, ax = plt.subplots(figsize=(7.2, .62 * len(names) + 1.4))
    left = [0.0] * len(names)
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(causes):
        vals = [(by[n]["by_cause"].get(c) or 0) for n in names]
        ax.barh(names, vals, left=left, height=.6,
                label=labels.get(c, c), color=cmap(i * 2 % 20))
        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("tỉ trọng giữa các nguyên nhân (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, ncol=2, loc="upper center",
              bbox_to_anchor=(.5, -.28), frameon=False)
    return _png(fig)


def _fmt(v, d=0, dash="—"):
    if v is None:
        return dash
    return f"{v:,.{d}f}"


def build_html(data: Dict[str, Any]) -> str:
    rows = data["machines"]
    tot = data["fleet_total"]
    fp = data.get("failure_fingerprint") or {}
    cov = data["coverage"]

    c_out = _chart_output(rows)
    c_fp = _chart_fingerprint(fp) if fp else None

    def tr(r):
        p = r.get("production") or {}
        if not p:
            return (f"<tr><td class='n'>{r['machine']}</td>"
                    f"<td colspan='4' class='muted'>{r.get('error') or 'không có dữ liệu'}</td></tr>")
        rec = (r.get("recipes") or [{}])
        return (f"<tr><td class='n'>{r['machine']}<span class='sub'>{r.get('line') or ''}</span></td>"
                f"<td>{_fmt(p.get('total_products'))}</td>"
                f"<td>{_fmt(p.get('per_day'), 1)}</td>"
                f"<td>{_fmt(p.get('pass_rate'), 2)}%</td>"
                f"<td class='muted'>{(rec[0].get('name') if rec else '') or '—'}</td></tr>")

    fp_rows = ""
    if fp:
        labels = fp.get("cause_labels") or {}
        head = "".join(f"<th>{labels.get(c, c)}</th>" for c in fp["causes"])
        body = ""
        for name, r in fp["by_machine"].items():
            cells = "".join(
                f"<td>{_fmt(r['by_cause'].get(c), 1)}{'%' if r['by_cause'].get(c) is not None else ''}</td>"
                for c in fp["causes"])
            note = "phủ hết kỳ" if r.get("sample_covers_all") else "lấy mẫu"
            body += (f"<tr><td class='n'>{name}</td>{cells}"
                     f"<td class='muted'>{_fmt(r.get('sample_products'))} · {note}</td></tr>")
        fp_rows = f"""
        <h2>Vân tay kiểu lỗi</h2>
        <p class="lead">Tỉ trọng giữa các nguyên nhân trên mẫu fail của từng máy —
        các ô của một máy cộng lại bằng 100. Đây là cách so sánh có nghĩa giữa các
        máy, vì những nguyên nhân này thuộc pipeline OCR chứ không thuộc mặt hàng.</p>
        {f'<img src="data:image/png;base64,{c_fp}">' if c_fp else ''}
        <table><thead><tr><th>Máy</th>{head}<th>Mẫu</th></tr></thead>
        <tbody>{body}</tbody></table>"""

    missing = ""
    if not cov.get("complete"):
        items = [f"{m['machine']} ({m['reason']})" for m in cov.get("machines_missing", [])]
        items += [f"{m['machine']} ({m['reason']})" for m in cov.get("machines_degraded", [])]
        missing = (f"<div class='warn'><b>Báo cáo này KHÔNG đủ cả đội hình.</b> "
                   f"Thiếu: {', '.join(items)}</div>")

    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<title>Báo cáo so sánh máy</title><style>
 @page {{ size: A4; margin: 16mm 14mm; }}
 body {{ font:13px/1.55 -apple-system,"Helvetica Neue",Arial,sans-serif; color:#141821; }}
 h1 {{ font-size:20px; margin:0 0 4px; }}
 h2 {{ font-size:15px; margin:26px 0 8px; padding-bottom:5px; border-bottom:2px solid #e3e7ee;
        break-after: avoid; }}
 /* Giữ tiêu đề, câu dẫn và biểu đồ đi cùng nhau. Không có mấy dòng này thì
    WeasyPrint tách "Vân tay kiểu lỗi" ở cuối trang 1 còn biểu đồ sang trang 2,
    và người đọc gặp một tiêu đề trống. */
 .lead {{ break-after: avoid; }}
 table {{ break-inside: auto; }}
 tr {{ break-inside: avoid; }}
 .meta {{ color:#5f6878; font-size:12px; margin-bottom:18px; }}
 .lead {{ color:#5f6878; font-size:12px; margin:0 0 12px; }}
 .kpis {{ display:flex; gap:10px; flex-wrap:wrap; margin:14px 0 4px; }}
 .kpi {{ flex:1; min-width:120px; border:1px solid #e3e7ee; border-radius:9px; padding:10px 13px; }}
 .kpi .k {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.05em; color:#98a1b3; }}
 .kpi .v {{ font-size:19px; font-weight:650; margin-top:2px; }}
 table {{ border-collapse:collapse; width:100%; font-size:12px; margin-top:8px; }}
 th,td {{ padding:7px 9px; text-align:right; border-bottom:1px solid #e9ecf2; }}
 th {{ background:#f7f9fc; font-size:10.5px; text-transform:uppercase;
       letter-spacing:.04em; color:#5f6878; }}
 th:first-child, td:first-child {{ text-align:left; }}
 td.n {{ font-weight:600; }} td.n .sub {{ display:block; font-weight:400; color:#98a1b3; font-size:10.5px; }}
 .muted {{ color:#5f6878; font-weight:400; }}
 tr.tot td {{ font-weight:650; background:#f7f9fc; }}
 img {{ width:100%; margin:10px 0 4px; }}
 .warn {{ background:#fff7ed; border:1px solid #fed7aa; color:#9a3412;
          padding:10px 13px; border-radius:8px; margin:14px 0; font-size:12px; }}
 .foot {{ margin-top:26px; padding-top:10px; border-top:1px solid #e3e7ee;
          color:#98a1b3; font-size:11px; }}
</style></head><body>
<h1>Báo cáo so sánh máy</h1>
<div class="meta">{data['period_label']} · {len(rows)} máy · lập lúc
  {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
{missing}
<div class="kpis">
  <div class="kpi"><div class="k">Tổng sản lượng</div><div class="v">{_fmt(tot.get('products'))}</div></div>
  <div class="kpi"><div class="k">Đạt</div><div class="v">{_fmt(tot.get('pass'))}</div></div>
  <div class="kpi"><div class="k">Không đạt</div><div class="v">{_fmt(tot.get('fail'))}</div></div>
  <div class="kpi"><div class="k">Tỉ lệ đạt</div><div class="v">{_fmt(tot.get('pass_rate'), 2)}%</div></div>
</div>

<h2>Sản lượng theo máy</h2>
<p class="lead">Không xếp hạng theo tỉ lệ đạt: các máy chạy recipe khác nhau, nên
tỉ lệ đạt phản ánh độ khó của mặt hàng chứ không phản ánh máy.</p>
{f'<img src="data:image/png;base64,{c_out}">' if c_out else ''}
<table><thead><tr><th>Máy</th><th>Sản lượng</th><th>Mỗi ngày</th>
  <th>Tỉ lệ đạt</th><th>Recipe</th></tr></thead>
<tbody>{''.join(tr(r) for r in rows)}
<tr class="tot"><td>Tổng</td><td>{_fmt(tot.get('products'))}</td><td>—</td>
  <td>{_fmt(tot.get('pass_rate'), 2)}%</td><td class="muted">—</td></tr></tbody></table>
{fp_rows}
<div class="foot">Sinh bởi Fleet Service · dữ liệu đọc trực tiếp từ từng máy tại
thời điểm lập báo cáo.</div>
</body></html>"""


def build_pdf(html: str, path: Path) -> None:
    from weasyprint import HTML
    HTML(string=html).write_pdf(str(path))


def build_excel(data: Dict[str, Any], path: Path) -> None:
    """
    Excel nhiều sheet: một sheet tổng hợp, một sheet vân tay lỗi.

    Tách sheet thay vì nhồi một bảng: hai bảng có ĐƠN VỊ khác nhau (số sản phẩm
    và tỉ trọng %), để chung một sheet là mời người đọc cộng nhầm cột.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="334155")

    ws = wb.active
    ws.title = "Sản lượng"
    ws.append(["Máy", "Dây chuyền", "Sản lượng", "Mỗi ngày", "Đạt", "Không đạt",
               "Tỉ lệ đạt (%)", "Recipe", "Ghi chú"])
    for r in data["machines"]:
        p = r.get("production") or {}
        rec = (r.get("recipes") or [{}])
        ws.append([r["machine"], r.get("line"), p.get("total_products"),
                   p.get("per_day"), p.get("pass"), p.get("fail"),
                   p.get("pass_rate"), (rec[0].get("name") if rec else None),
                   r.get("error") or ""])
    t = data["fleet_total"]
    ws.append([])
    ws.append(["TỔNG", "", t.get("products"), "", t.get("pass"), t.get("fail"),
               t.get("pass_rate"), "", ""])

    fp = data.get("failure_fingerprint") or {}
    if fp:
        labels = fp.get("cause_labels") or {}
        w2 = wb.create_sheet("Vân tay kiểu lỗi")
        w2.append(["Máy"] + [labels.get(c, c) for c in fp["causes"]]
                  + ["Cỡ mẫu", "Phủ hết kỳ"])
        for name, r in fp["by_machine"].items():
            w2.append([name] + [r["by_cause"].get(c) for c in fp["causes"]]
                      + [r.get("sample_products"), bool(r.get("sample_covers_all"))])
        w2.append([])
        w2.append(["Các ô là TỈ TRỌNG giữa các nguyên nhân (một máy cộng lại = 100), "
                   "không phải % sản phẩm trượt từng bước."])

    for sheet in wb.worksheets:
        for c in sheet[1]:
            c.font, c.fill = head_font, head_fill
            c.alignment = Alignment(horizontal="center")
        for col in sheet.columns:
            width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
            sheet.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 42)
        sheet.freeze_panes = "A2"

    wb.save(str(path))


def build_csv(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Máy", "Dây chuyền", "Sản lượng", "Mỗi ngày", "Đạt",
                    "Không đạt", "Tỉ lệ đạt (%)", "Recipe", "Ghi chú"])
        for r in data["machines"]:
            p = r.get("production") or {}
            rec = (r.get("recipes") or [{}])
            w.writerow([r["machine"], r.get("line"), p.get("total_products"),
                        p.get("per_day"), p.get("pass"), p.get("fail"),
                        p.get("pass_rate"), (rec[0].get("name") if rec else None),
                        r.get("error") or ""])
        t = data["fleet_total"]
        w.writerow([])
        w.writerow(["TỔNG", "", t.get("products"), "", t.get("pass"),
                    t.get("fail"), t.get("pass_rate"), "", ""])


_EXT = {"html": "html", "pdf": "pdf", "excel": "xlsx", "csv": "csv"}


def render(data: Dict[str, Any], fmt: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    names = "-".join(r["machine"] for r in data["machines"])[:60]
    path = OUT_DIR / f"fleet_{names}_{stamp}.{_EXT[fmt]}"
    if fmt == "html":
        path.write_text(build_html(data), encoding="utf-8")
    elif fmt == "pdf":
        build_pdf(build_html(data), path)
    elif fmt == "excel":
        build_excel(data, path)
    else:
        build_csv(data, path)
    return path
