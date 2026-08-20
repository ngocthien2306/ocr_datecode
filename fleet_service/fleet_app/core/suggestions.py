"""
Gợi ý câu hỏi tiếp theo, SUY TỪ SỐ LIỆU vừa trả về.

Cùng nguyên tắc `grounded_suggestions` ở tầng edge, và cùng lý do đã ghi ở đó:
để mô hình tự viết gợi ý thì nó viết mà không nhìn con số, nên sau câu "máy nào
tệ nhất" nó mời "Xem lịch sử load recipe" — chẳng dính gì tới thứ vừa hiện. Còn
"Vì sao M2 có ký tự dưới ngưỡng cao gấp đôi mặt bằng?" thì lấy đúng con số vừa
nằm trên màn hình.

Thứ gì dựng được bằng code từ kết quả tool thì đừng để mô hình viết — y hệt lý do
`kpis` và `charts` không đi qua mô hình.
"""

from __future__ import annotations

from typing import Any, Dict, List

MAX = 5


def _outlier(fp: Dict[str, Any]) -> tuple:
    """
    Máy lệch nhiều nhất so với trung vị của các máy khác, ở nguyên nhân nào.

    Dùng khoảng cách tới TRUNG VỊ chứ không phải tới giá trị lớn nhất: lấy max thì
    máy cao nhất luôn tự nó là outlier, kể cả khi cả năm máy gần như nhau.
    """
    by = fp.get("by_machine") or {}
    causes = fp.get("causes") or []
    if len(by) < 2 or not causes:
        return None, None, None
    best = (None, None, 0.0)
    for c in causes:
        vals = {n: (v["by_cause"].get(c) or 0) for n, v in by.items()}
        ordered = sorted(vals.values())
        mid = ordered[len(ordered) // 2]
        for n, v in vals.items():
            if v - mid > best[2]:
                best = (n, c, v - mid)
    if best[2] < 12:          # lệch nhỏ thì không đáng gọi là bất thường
        return None, None, None
    return best


def build(tool_calls: List[str], results: Dict[str, Any]) -> List[str]:
    """
    Gợi ý theo thứ tự cụ thể → chung. Rỗng nếu không suy được gì.

    `results` là dict {tên tool: kết quả}. Chỉ đọc, không gọi lại gì.
    """
    out: List[str] = []
    used = set(tool_calls or [])

    fp_res = results.get("compare_failure_modes") or {}
    fp = fp_res.get("fingerprint") or {}
    if fp:
        machine, cause, gap = _outlier(fp)
        if machine:
            label = (fp.get("cause_labels") or {}).get(cause, cause)
            if "ask_machine" not in used:
                out.append(f"Vì sao {machine} có '{label}' cao hơn hẳn các máy khác?")
            out.append(f"Xem ảnh sản phẩm lỗi gần đây của {machine}")

    prod = results.get("fleet_production") or {}
    if prod.get("machines"):
        names = [m["machine"] for m in prod["machines"] if m.get("production")]
        if len(names) >= 2:
            out.append(f"Xuất báo cáo so sánh {' và '.join(names[:2])}")
        if prod.get("period_days") and prod["period_days"] <= 7:
            out.append("So sánh 30 ngày qua thay vì 7 ngày")

    rep = results.get("generate_fleet_report") or {}
    if rep.get("file"):
        ms = rep.get("machines") or []
        # Mời định dạng KHÁC cái vừa xuất. Bản đầu luôn mời Excel, kể cả ngay sau
        # khi vừa xuất Excel — gợi ý mời làm lại đúng việc vừa làm thì vô dụng.
        other = {"excel": "PDF", "pdf": "Excel", "html": "PDF",
                 "csv": "Excel"}.get(rep.get("format"), "PDF")
        out.append(f"Xuất cùng báo cáo này ra {other}")
        if len(ms) < 5:
            out.append("Xuất lại báo cáo cho cả 5 máy")
        out.append("So sánh kỳ 30 ngày cho đúng nhóm máy này")

    health = results.get("fleet_health") or {}
    for m in (health.get("machines") or []):
        x = m.get("metrics") or {}
        if (x.get("disk_percent") or 0) >= 85:
            out.append(f"Đĩa {m['machine']} còn {x.get('disk_free_gb')} GB — nên dọn gì?")
            break
    for m in (health.get("machines") or []):
        t = (m.get("metrics") or {}).get("cpu_temp")
        if t is not None and t >= 85:
            out.append(f"{m['machine']} đang {t:.0f}°C — có nguy hiểm không?")
            break

    # Máy thiếu dữ liệu được ưu tiên hỏi tiếp: đó là thứ người vận hành cần biết
    # nhất mà lại dễ trôi qua nhất, vì bảng vẫn hiện đầy đủ các máy còn lại.
    for res in results.values():
        cov = (res or {}).get("coverage") or {}
        for m in (cov.get("machines_missing") or []) + (cov.get("machines_degraded") or []):
            q = f"Vì sao {m['machine']} không trả lời?"
            if q not in out:
                out.insert(0, q)
            break

    if not out:
        out = ["Sản lượng cả đội hình 7 ngày qua",
               "Máy nào có vân tay lỗi bất thường?",
               "Máy nào đang nóng hoặc sắp đầy đĩa?"]

    seen, uniq = set(), []
    for q in out:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq[:MAX]
