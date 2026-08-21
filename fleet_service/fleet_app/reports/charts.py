"""
Biểu đồ báo cáo — PNG vẽ sẵn, nhúng base64.

Vì sao không Chart.js: WeasyPrint chỉ hiểu HTML/CSS, không chạy JavaScript, nên
`<canvas>` ra trắng trơn trong bản PDF. Vẽ sẵn thì MỘT template dùng được cho cả
HTML lẫn PDF, và file HTML cũng tự chứa — gửi qua Zalo, mở offline vẫn thấy hình.

Mọi màu ở đây phải trùng với bảng màu trong `builder.py`: hình và bảng nằm cạnh
nhau trên cùng trang, lệch một tông là người đọc mất đường nối giữa chúng.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

# Bảng màu cố định theo TÊN MÁY, không theo thứ tự trong danh sách. Gán theo thứ
# tự thì bỏ một máy ra khỏi báo cáo là mọi máy còn lại đổi màu, và hai bản báo
# cáo cạnh nhau không đọc chéo được nữa.
PALETTE = ["#2563eb", "#0891b2", "#7c3aed", "#ca8a04", "#dc2626",
           "#16a34a", "#db2777", "#65a30d"]

# Màu theo MÃ nguyên nhân, cùng lý do như trên: bản trước lấy `cmap(i*2%20)`,
# nên chỉ cần một máy có thêm một nguyên nhân là cả biểu đồ đổi màu.
CAUSE_COLOR = {
    "char_verification":     "#1e2a3a",
    "no_detection":          "#3d5a7a",
    "text_verification":     "#6b8caf",
    "template_verification": "#9db4cc",
    "product_verification":  "#c8a35a",
}
CAUSE_FALLBACK = ["#8a94a6", "#b3bac6", "#d5d9e0"]

INK, GRID, PASS, FAIL, LINE = "#141821", "#e3e7ee", "#2f6f4f", "#c2544d", "#1e2a3a"


# Nhãn NGẮN cho tiêu đề bảng. Nhãn đầy đủ dài 30–38 ký tự; năm cột như thế
# cộng lại rộng hơn khổ A4, và bảng bị đẩy tràn ra ngoài lề — đo được ở bản đầu.
# Nhãn đầy đủ vẫn còn ở chú giải biểu đồ ngay phía trên, nối bằng ô màu.
CAUSE_SHORT = {
    "char_verification":     "Ký tự",
    "no_detection":          "Không thấy vùng",
    "text_verification":     "Đọc sai chuỗi",
    "template_verification": "Lệch template",
    "product_verification":  "Sai sản phẩm",
}

_ASSIGNED: dict = {}


def configure(names) -> None:
    """Gán màu cho đúng tập máy của báo cáo này, KHÔNG để hai máy trùng màu.

    Băm tên rồi lấy dư (bản trước) cho màu ổn định theo tên, nhưng `M2` và
    `PC-Auto-1` băm về cùng một ô — hai máy cùng màu xanh lá trong đúng một báo
    cáo SO SÁNH, đọc thành một máy.

    Nên: vẫn ưu tiên ô băm ra, nhưng đã có người chiếm thì dò tiếp. Sắp tên
    trước khi gán để kết quả không phụ thuộc thứ tự truyền vào.

    Đánh đổi đã biết: khi có xung đột, bỏ một máy khỏi báo cáo CÓ THỂ làm máy
    khác đổi màu. Vẫn chọn thế, vì hai máy cùng màu thì báo cáo sai ngay trên
    trang, còn đổi màu chỉ khó đối chiếu giữa hai bản.
    """
    _ASSIGNED.clear()
    used = set()
    for n in sorted(names):
        i = sum(ord(c) for c in n) % len(PALETTE)
        for k in range(len(PALETTE)):
            c = PALETTE[(i + k) % len(PALETTE)]
            if c not in used:
                used.add(c)
                _ASSIGNED[n] = c
                break
        else:
            _ASSIGNED[n] = PALETTE[i]


def color_for(name: str) -> str:
    if name in _ASSIGNED:
        return _ASSIGNED[name]
    return PALETTE[sum(ord(c) for c in name) % len(PALETTE)]


def cause_color(cause: str, i: int = 0) -> str:
    return CAUSE_COLOR.get(cause, CAUSE_FALLBACK[i % len(CAUSE_FALLBACK)])


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # DejaVu Sans có đủ dấu tiếng Việt; đặt tường minh để không phụ thuộc font
    # hệ thống của máy lập báo cáo — trên máy khác là ô vuông.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.edgecolor": "#b9c0cc", "text.color": INK,
                         "axes.labelcolor": "#5f6878", "xtick.color": "#5f6878",
                         "ytick.color": "#5f6878"})
    return plt


def _png(fig) -> str:
    plt = _plt()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def _clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=.8)
    ax.set_axisbelow(True)


def output_per_day(views: List[Dict[str, Any]]) -> Optional[str]:
    """
    Sản lượng chuẩn hoá: mỗi máy hai cột — chia cho cả kỳ, và chia cho ngày CÓ CHẠY.

    Hai cột cạnh nhau chứ không một cột: máy chạy 2 trong 7 ngày mà chỉ in cột
    thứ nhất thì trông như máy yếu, in cả hai thì đọc ra "chạy nhanh nhưng chạy
    ít ngày" — hai kết luận dẫn tới hai việc khác nhau.
    """
    plt = _plt()
    live = [v for v in views if v.get("per_day")]
    if not live:
        return None
    names = [v["machine"] for v in live]
    a = [v.get("per_day") or 0 for v in live]
    b = [v.get("per_active_day") or 0 for v in live]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    ax.bar([i - .19 for i in x], a, width=.36,
           color=[color_for(n) for n in names], label="chia cho cả kỳ")
    ax.bar([i + .19 for i in x], b, width=.36,
           color=[color_for(n) for n in names], alpha=.42,
           edgecolor=[color_for(n) for n in names], lw=1.1,
           hatch="///", label="chia cho ngày có chạy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("sản phẩm / ngày")
    for i, v in enumerate(b):
        if v:
            ax.text(i + .19, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5)
    # Chú giải phải TRUNG TÍNH màu. Mỗi cột mang màu của máy nó, nên một ô chú
    # giải màu vàng đứng cạnh chữ "chia cho cả kỳ" đọc thành "màu vàng nghĩa là
    # cả kỳ" — trong khi vàng là màu của Auto2. Ở đây chú giải chỉ giải thích
    # KIỂU cột: đặc hay gạch chéo.
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#8a94a6", label="chia cho cả kỳ"),
        Patch(facecolor="#8a94a6", alpha=.42, edgecolor="#8a94a6", hatch="///",
              label="chia cho ngày có chạy"),
    ], fontsize=7.5, frameon=False, ncol=2, loc="upper center",
        bbox_to_anchor=(.5, 1.2))
    _clean(ax)
    return _png(fig)


def fleet_daily(views: List[Dict[str, Any]]) -> Optional[str]:
    """
    Nhịp cả nhà máy theo ngày, cột chồng theo máy.

    Chồng theo máy chứ không tổng một màu: một ngày sản lượng tụt thì câu hỏi
    ngay sau đó luôn là "máy nào", và một cột trơn thì phải tra sang bảng khác.
    """
    plt = _plt()
    keys = sorted({d["key"] for v in views for d in v.get("days") or []})
    live = [v for v in views if v.get("days")]
    if not keys or not live:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    bottom = [0.0] * len(keys)
    for v in live:
        by = {d["key"]: d["total"] for d in v["days"]}
        vals = [by.get(k, 0) for k in keys]
        ax.bar(keys, vals, bottom=bottom, width=.62,
               color=color_for(v["machine"]), label=v["machine"])
        bottom = [b + s for b, s in zip(bottom, vals)]
    ax.set_ylabel("sản phẩm")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k[8:10] + "/" + k[5:7] for k in keys], fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, ncol=min(len(live), 5),
              loc="upper center", bbox_to_anchor=(.5, 1.24))
    _clean(ax)
    return _png(fig)


def fingerprint(fp: Dict[str, Any]) -> Optional[str]:
    """
    Vân tay kiểu lỗi: cột chồng NGANG, mỗi máy một thanh 100%.

    Cột chồng chứ không nhóm cạnh nhau: mỗi máy cộng lại đúng 100, nên hình dạng
    của cả thanh chính là "vân tay". Để cạnh nhau thì mắt so từng cặp cột và mất
    luôn cái nhìn tổng thể — thứ duy nhất so được giữa các máy chạy khác mặt hàng.
    """
    plt = _plt()
    causes, rows = fp.get("causes") or [], fp.get("rows") or []
    if not causes or not rows:
        return None
    labels = fp.get("labels") or {}
    names = [r["machine"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, .42 * len(names) + 1.55))
    left = [0.0] * len(names)
    for i, c in enumerate(causes):
        vals = []
        for r in rows:
            cell = next((x for x in r["cells"] if x["cause"] == c), None)
            vals.append((cell or {}).get("value") or 0)
        ax.barh(names, vals, left=left, height=.6,
                label=labels.get(c, c), color=cause_color(c, i))
        for j, (l, v) in enumerate(zip(left, vals)):
            # Chỉ ghi số vào lát đủ rộng để chứa nó. Ghi hết thì các lát mỏng
            # đè chữ lên nhau và không đọc được lát nào.
            if v >= 11:
                ax.text(l + v / 2, j, f"{v:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("tỉ trọng giữa các nguyên nhân, trên mẫu fail (%)")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.set_axisbelow(True)
    # Máy không có mẫu vẫn có hàng trong biểu đồ, và phải nói ra là vì sao —
    # một hàng trống không có chữ đọc thành "máy này không lỗi bao giờ".
    for j, r in enumerate(rows):
        if not any((c.get("value") or 0) for c in r["cells"]):
            ax.text(1.5, j, "không có mẫu trong kỳ", va="center", fontsize=7.5,
                    color="#98a1b3", style="italic")
    # Chú giải đặt TRÊN vùng vẽ. Đặt dưới thì nó phải chen với nhãn trục x và
    # dòng "tỉ trọng giữa các nguyên nhân (%)" — đã đo được lần chồng chữ thật:
    # "Detector không thấy vùng nào trong khung" cắt ngang qua số 40 và 60.
    # Trên thì phía đó trống, và chỗ chừa tính theo chiều cao hình chứ không
    # bằng một hằng số, vì hình 2 máy và hình 5 máy cao khác nhau gần gấp đôi.
    h = .42 * len(names) + 1.55
    fig.subplots_adjust(top=1 - .66 / h)
    fig.legend(fontsize=7.5, ncol=3, loc="upper center",
               bbox_to_anchor=(.5, .995), frameon=False)
    return _png(fig)


def machine_trend(view: Dict[str, Any]) -> Optional[str]:
    """
    Chi tiết một máy: cột chồng đạt/không đạt + đường tỉ lệ đạt ở trục phải.

    Chia theo TUẦN khi kỳ có từ hai tuần trở lên; một tuần thì vẽ theo NGÀY.
    Một cột đơn độc không so được với gì cả — cùng lý do đã bỏ biểu đồ một chuỗi
    ở màn hình Line Station: thang đo lúc đó là chính nó.
    """
    plt = _plt()
    weeks = [w for w in view.get("weeks") or [] if w["total"]]
    days = [d for d in view.get("days") or [] if d["total"]]
    if len(weeks) >= 2:
        xs = [f"T{w['week']}\n{w['span']}" for w in weeks]
        src = weeks
    elif days:
        xs = [d["key"][8:10] + "/" + d["key"][5:7] for d in days]
        src = days
    else:
        return None

    ps = [s["pass"] for s in src]
    fs = [s["fail"] for s in src]
    rates = [s["rate"] for s in src]

    # Thấp có chủ ý: thẻ máy phải đủ ngắn để HAI thẻ vừa một trang A4. Ở bản
    # 2,2in mỗi thẻ cao 145mm, nên mỗi trang chỉ chứa một thẻ và phụ lục 5 máy
    # ngốn 5 trang — người đọc phải lật qua lật lại để so hai máy.
    fig, ax = plt.subplots(figsize=(7.0, 1.5))
    ax.bar(xs, ps, width=.55, color=PASS, label="đạt")
    ax.bar(xs, fs, width=.55, bottom=ps, color=FAIL, label="không đạt")
    ax.set_ylabel("sản phẩm")
    _clean(ax)

    ax2 = ax.twinx()
    ax2.plot(xs, rates, color=LINE, lw=1.6, marker="o", ms=3.6,
             label="tỉ lệ đạt")
    ax2.set_ylabel("tỉ lệ đạt (%)")
    ax2.spines[["top"]].set_visible(False)
    # Trục phải KHÔNG cố định 0–100: dao động 96–99 mà kéo cả thang thì đường
    # tỉ lệ thành một vạch thẳng và không thấy gì. Nhưng cũng không để matplotlib
    # tự chọn hoàn toàn — chừa lề để đường không chạm cạnh khung.
    ok = [r for r in rates if r is not None]
    if ok:
        lo, hi = min(ok), max(ok)
        pad = max((hi - lo) * .25, 1.5)
        ax2.set_ylim(max(0, lo - pad), min(100, hi + pad))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(.5, 1.26))
    return _png(fig)
