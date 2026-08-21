"""
Tính toán cho báo cáo so sánh — TÁCH KHỎI phần dựng HTML.

Vì sao tách: mọi con số trong báo cáo phải kiểm được bằng một hàm gọi trực tiếp,
không phải bằng cách đọc chuỗi HTML. Bản trước trộn hai việc, nên câu duy nhất
trả lời được "tuần này tính từ đâu tới đâu" là đọc f-string.

Không có phụ thuộc nào ngoài stdlib ở đây — không matplotlib, không WeasyPrint.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Ngưỡng tỉ lệ đạt, dùng chung cho màu ô KPI và chữ trạng thái.
RATE_GOOD, RATE_WATCH = 95.0, 85.0


def fmt_num(v: Optional[float], d: int = 0, dash: str = "—") -> str:
    """Dấu nghìn "." và dấu thập phân "," — lối viết số ở đây.

    Định nghĩa nằm ở module này chứ không ở lớp vẽ, vì các câu trong `findings()`
    cũng chứa số và phải viết giống hệt bảng ngay bên trên nó. Bản đầu để lớp vẽ
    tự định dạng, nên bảng in "63,88%" còn phát hiện chính in "63.88%" — hai lối
    viết trong cùng một trang.
    """
    if v is None:
        return dash
    return (f"{v:,.{d}f}".replace(",", "\x00").replace(".", ",")
            .replace("\x00", "."))


def fmt_pct(v: Optional[float], d: int = 2) -> str:
    return "—" if v is None else f"{fmt_num(v, d)}%"


def rate_level(rate: Optional[float]) -> str:
    """'none' khi không đo được — KHÔNG phải 'bad'.

    Trộn hai thứ này là cách nhanh nhất để một máy tắt agent trông như một máy
    đang chạy hỏng: cả hai ra ô đỏ, mà việc cần làm thì khác hẳn nhau.
    """
    if rate is None:
        return "none"
    if rate >= RATE_GOOD:
        return "good"
    if rate >= RATE_WATCH:
        return "watch"
    return "bad"


def _d(key: str) -> Optional[date]:
    try:
        return datetime.strptime(key[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def daily(trend: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chuỗi ngày đã sắp, mỗi phần tử có pass/fail/total/rate."""
    out = []
    for k in sorted(trend or {}):
        v = (trend or {})[k] or {}
        p, f = v.get("pass") or 0, v.get("fail") or 0
        tot = p + f
        out.append({"key": k, "date": _d(k), "pass": p, "fail": f,
                    "total": tot,
                    "rate": round(p * 100.0 / tot, 2) if tot else None})
    return out


def weekly(trend: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Gộp chuỗi ngày thành TUẦN ISO (thứ Hai đầu tuần).

    Tuần ISO chứ không phải "7 ngày tính ngược từ hôm nay": người ở xưởng nói
    "tuần trước" theo lịch, và một cửa sổ trượt thì hai báo cáo lập cách nhau
    một ngày không so được với nhau.

    Nhãn ghi kèm KHOẢNG NGÀY THẬT trong dữ liệu, không phải ngày đầu/cuối tuần
    lịch: tuần đầu của kỳ thường bị cắt, và ghi "17/08–23/08" cho một tuần chỉ
    có số của hai ngày là nói quá phạm vi dữ liệu.
    """
    buckets: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for d in daily(trend):
        if d["date"] is None:
            continue
        iso = d["date"].isocalendar()
        key = (iso[0], iso[1])
        b = buckets.setdefault(key, {
            "year": iso[0], "week": iso[1],
            "monday": d["date"] - timedelta(days=iso[2] - 1),
            "first": d["date"], "last": d["date"],
            "pass": 0, "fail": 0, "days": 0,
        })
        b["first"] = min(b["first"], d["date"])
        b["last"] = max(b["last"], d["date"])
        b["pass"] += d["pass"]
        b["fail"] += d["fail"]
        b["days"] += 1 if d["total"] else 0

    out = []
    for key in sorted(buckets):
        b = buckets[key]
        tot = b["pass"] + b["fail"]
        out.append({
            **b,
            "total": tot,
            "rate": round(b["pass"] * 100.0 / tot, 2) if tot else None,
            # Chuẩn hoá theo NGÀY CÓ CHẠY, không theo 7. Một tuần mới bắt đầu
            # được hai ngày mà chia cho 7 thì máy trông như vừa tụt sản lượng.
            "per_active_day": round(tot / b["days"], 1) if b["days"] else None,
            "label": f"Tuần {b['week']}",
            "span": (f"{b['first'].strftime('%d/%m')}–{b['last'].strftime('%d/%m')}"
                     if b["first"] != b["last"] else b["first"].strftime("%d/%m")),
            # `partial` = tuần bị KỲ BÁO CÁO cắt (đầu kỳ/cuối kỳ). Với kỳ 7
            # ngày thì mọi tuần đều thế, nên nó không đáng in ra.
            "partial": ((b["last"] - b["first"]).days + 1) < 7,
            # Còn đây là thứ đáng in: trong khoảng đã có số, có bao nhiêu ngày
            # KHÔNG ra sản phẩm. Đó là ngày đứng máy, không phải ngày ngoài kỳ.
            "idle_days": ((b["last"] - b["first"]).days + 1) - b["days"],
        })

    # Delta tính SAU khi đã sắp, và chỉ giữa hai tuần cạnh nhau có số. Bỏ qua
    # bước này thì tuần đầu kỳ mang delta so với hư không.
    prev = None
    for w in out:
        w["delta_points"] = (round(w["rate"] - prev, 2)
                             if w["rate"] is not None and prev is not None else None)
        if w["rate"] is not None:
            prev = w["rate"]
    return out


def machine_view(row: Dict[str, Any], period_days: int) -> Dict[str, Any]:
    """
    Một máy, đủ thứ phụ lục cần — kể cả máy không có dữ liệu.

    Máy lỗi vẫn trả về một view đầy đủ khoá với giá trị None, để lớp vẽ không
    phải rẽ nhánh và không thể vô tình bỏ máy đó ra khỏi báo cáo.
    """
    prod = row.get("production") or {}
    trend = prod.get("trend") or {}
    days = daily(trend)
    weeks = weekly(trend)
    active = [d for d in days if d["total"]]
    total = prod.get("total_products")

    fm = row.get("failure_modes") or {}
    causes = fm.get("by_cause") or []
    top = max(causes, key=lambda c: c.get("share_of_causes_pct") or 0) if causes else None

    worst = min((d for d in active if d["rate"] is not None),
                key=lambda d: d["rate"], default=None)

    return {
        "machine": row.get("machine"),
        "line": row.get("line"),
        "model": row.get("model"),
        "state": row.get("state"),
        "error": row.get("error"),
        "has_data": bool(prod),
        "total": total,
        "pass": prod.get("pass"),
        "fail": prod.get("fail"),
        "rate": prod.get("pass_rate"),
        "per_day": prod.get("per_day"),
        # Hai cách chuẩn hoá, cố ý để cạnh nhau. `per_day` chia cho cả kỳ, nên
        # một máy chạy 2 trong 7 ngày trông như sản lượng thấp; `per_active_day`
        # nói nó chạy nhanh nhưng chạy ít ngày. Chỉ in một trong hai là bỏ mất
        # một nửa câu trả lời cho "vì sao máy này thấp".
        "active_days": len(active),
        "period_days": period_days,
        "per_active_day": (round(total / len(active), 1)
                           if total and active else None),
        "days": days,
        "weeks": weeks,
        "recipes": row.get("recipes") or [],
        "top_cause": top,
        "sample_products": fm.get("sample_products"),
        "sample_covers_all": fm.get("sample_covers_all"),
        "sampling": fm.get("sampling"),
        "template_similarity_avg": fm.get("template_similarity_avg"),
        "worst_day": worst,
    }


# Cần bấy nhiêu máy CÓ SỐ thì trung vị của cột mới đáng in.
MEDIAN_MIN_N = 3


def _median(xs: List[float]) -> Optional[float]:
    """Trung vị, hoặc None khi quá ít máy có số cho cột này.

    Không có ngưỡng thì cột chỉ một máy có số cho ra "trung vị" đúng bằng giá
    trị của máy đó — in ra là mời người đọc so nó với chính nó, và ô đó không
    bao giờ bị tô đậm dù lệch tới đâu. Cùng lỗi đã bỏ biểu đồ một chuỗi ở màn
    hình Line Station.
    """
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < MEDIAN_MIN_N:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def fingerprint_view(fp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vân tay lỗi kèm TRUNG VỊ từng nguyên nhân và độ lệch của từng máy.

    Trung vị chứ không trung bình: năm máy thì một máy lệch mạnh đủ kéo trung
    bình đi theo nó, và ô lệch mạnh nhất tự làm mờ chính mình.
    """
    causes = fp.get("causes") or []
    by = fp.get("by_machine") or {}
    if not causes or not by:
        return {}
    med = {c: _median([(v.get("by_cause") or {}).get(c) for v in by.values()])
           for c in causes}
    rows = []
    for name, v in by.items():
        cells = []
        for c in causes:
            val = (v.get("by_cause") or {}).get(c)
            m = med.get(c)
            # "Lệch mạnh" = hơn trung vị 12 điểm trở lên. Ngưỡng tuyệt đối chứ
            # không tương đối: trung vị 0,5% thì lệch gấp ba vẫn là 1,5% và
            # không đáng tô đậm.
            cells.append({"cause": c, "value": val,
                          "hot": (val is not None and m is not None
                                  and val - m >= 12)})
        rows.append({"machine": name, "cells": cells,
                     "sample_products": v.get("sample_products"),
                     "covers_all": bool(v.get("sample_covers_all"))})
    return {"causes": causes, "labels": fp.get("cause_labels") or {},
            "median": med, "rows": rows}


# Việc cần làm, tra theo mã nguyên nhân. Nói "no_detection cao" mà không nói
# phải đi xem cái gì thì người đọc vẫn phải hỏi lại một câu nữa.
CAUSE_ACTION = {
    "no_detection": "kiểm tra camera/trigger/ánh sáng — detector không thấy vùng in",
    "char_verification": "thấy vùng in nhưng ký tự dưới ngưỡng: xem nét in, mực, tiêu cự",
    "text_verification": "OCR đọc ra chuỗi khác: xem lại chuỗi mong đợi trong recipe",
    "template_verification": "ảnh không khớp template: template có thể đã cũ so với bao bì",
    "product_verification": "không nhận ra sản phẩm: sai recipe đang chạy, hoặc bao bì mới",
}


def findings(views: List[Dict[str, Any]], fp: Dict[str, Any],
             coverage: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Phát hiện chính, dựng bằng CODE chứ không bằng mô hình.

    Đây là những dòng người đọc sẽ hành động theo, nên chúng phải suy được từ
    đúng những con số in ngay trên trang. Một câu do LLM viết thì hay hơn nhưng
    không kiểm được, và báo cáo là chỗ sai một câu là mất niềm tin cả tập.
    """
    out: List[Dict[str, Any]] = []

    for m in coverage.get("machines_missing") or []:
        out.append({
            "kind": "bad",
            "text": (f"{m['machine']} không có số trong kỳ này — báo cáo thiếu "
                     f"một dây chuyền, không phải dây chuyền đó không sản xuất."),
            "action": "kiểm tra agent trên máy đó rồi lập lại báo cáo",
        })

    live = [v for v in views if v["has_data"]]

    for v in sorted(live, key=lambda v: v["rate"] or 0):
        if v["rate"] is not None and v["rate"] < RATE_WATCH:
            top = v.get("top_cause") or {}
            act = CAUSE_ACTION.get(top.get("cause"), "xem chi tiết ở phụ lục máy")
            out.append({
                "kind": "bad",
                "text": (f"{v['machine']} đạt {fmt_pct(v['rate'])} — dưới ngưỡng "
                         f"{fmt_pct(RATE_WATCH, 0)}. Nguyên nhân chiếm tỉ trọng "
                         f"lớn nhất trên mẫu: {top.get('label') or '—'} "
                         f"({fmt_pct(top.get('share_of_causes_pct') or 0, 0)} của mẫu)."),
                "action": act,
            })
            break

    # Tuần trượt: chỉ tính khi có ít nhất hai tuần CÓ SỐ. So một tuần với hư
    # không là cách tạo ra một phát hiện từ không có gì.
    for v in live:
        ws = [w for w in v["weeks"] if w["rate"] is not None]
        if len(ws) >= 2 and ws[-1]["delta_points"] is not None:
            d = ws[-1]["delta_points"]
            if d <= -3:
                out.append({
                    "kind": "watch",
                    "text": (f"{v['machine']} tuần {ws[-1]['week']} giảm "
                             f"{fmt_num(abs(d), 2)} điểm so với tuần trước "
                             f"({fmt_pct(ws[-2]['rate'])} → {fmt_pct(ws[-1]['rate'])})."),
                    "action": "xem phụ lục theo tuần của máy này để biết trượt từ ngày nào",
                })
                break

    # Máy chạy ít ngày: nói ra để không ai đọc cột "mỗi ngày" như năng lực máy.
    few = [v for v in live
           if v["active_days"] and v["active_days"] * 2 <= v["period_days"]]
    if few:
        names = ", ".join(f"{v['machine']} ({v['active_days']}/{v['period_days']} ngày)"
                          for v in few)
        out.append({
            "kind": "info",
            "text": (f"Chạy dưới nửa số ngày trong kỳ: {names}. Cột “mỗi ngày” "
                     f"chia cho cả kỳ nên thấp hơn năng lực thật."),
            "action": "đọc cột “mỗi ngày có chạy” bên cạnh để so năng lực",
        })

    if fp and fp.get("rows"):
        hot = [(r["machine"], c) for r in fp["rows"] for c in r["cells"] if c["hot"]]
        for name, c in hot[:1]:
            lab = fp["labels"].get(c["cause"], c["cause"])
            out.append({
                "kind": "watch",
                "text": (f"{name} lệch hẳn khỏi phần còn lại ở “{lab}”: "
                         f"{fmt_pct(c['value'], 0)} của mẫu so với trung vị "
                         f"{fmt_pct(fp['median'][c['cause']], 0)}."),
                "action": CAUSE_ACTION.get(c["cause"], "so với máy chạy cùng recipe"),
            })

    if not out:
        out.append({"kind": "good",
                    "text": "Không có máy nào dưới ngưỡng, không có tuần nào trượt quá 3 điểm.",
                    "action": "không cần hành động"})
    return out
