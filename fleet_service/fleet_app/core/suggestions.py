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


# Chuỗi theo ngôn ngữ. Câu trả lời đi theo nút EN/VI của giao diện, nên gợi ý
# cũng phải theo — trả lời tiếng Anh rồi mời tiếp bằng tiếng Việt là hai giọng
# trong cùng một lượt.
_L = {
    "vi": {
        "whyHigh": lambda m, c: f"Vì sao {m} có '{c}' cao hơn hẳn các máy khác?",
        "seeFails": lambda m: f"Xem ảnh sản phẩm lỗi gần đây của {m}",
        "compare": lambda a, b: f"Xuất báo cáo so sánh {a} và {b}",
        "longer": "So sánh 30 ngày qua thay vì 7 ngày",
        "sameOther": lambda f: f"Xuất cùng báo cáo này ra {f}",
        "allFive": "Xuất lại báo cáo cho cả 5 máy",
        "same30": "So sánh kỳ 30 ngày cho đúng nhóm máy này",
        "disk": lambda m, g: f"Đĩa {m} còn {g} GB — nên dọn gì?",
        "hot": lambda m, t: f"{m} đang {t:.0f}°C — có nguy hiểm không?",
        "silent": lambda m: f"Vì sao {m} không trả lời?",
        # nhật ký
        "framesAround": lambda u, m: f"Xem ảnh trước/sau lần {u} sửa recipe trên {m}",
        "whoElse": lambda m: f"Còn ai thao tác recipe trên {m} tuần này?",
        "acrossMachines": lambda u: f"{u} đã làm gì trên các máy còn lại?",
        "busiest": lambda m, n: f"{m} có {n} thao tác — máy nào nhiều nhất?",
        "actionDetail": lambda a, m: f"Chi tiết các lần {a} trên {m}",
        # nhân sự
        "onShift": "Ai đang trong ca lúc này?",
        "deptSplit": lambda m: f"{m} có những bộ phận nào, mỗi bộ phận mấy người?",
        # lỗi hệ thống
        "errWorst": lambda m, n: f"{m} có {n} dòng lỗi — lỗi nào lặp nhiều nhất?",
        "errSince": lambda m: f"Lỗi trên {m} bắt đầu từ khi nào?",
        # ảnh
        "tplSplit": lambda m: f"Trên {m}, lỗi dồn vào vị trí chụp nào?",
        "wall": "Xem ảnh mới nhất của tất cả máy",
        "fallback": ["Sản lượng cả đội hình 7 ngày qua",
                     "Máy nào có vân tay lỗi bất thường?",
                     "Máy nào đang nóng hoặc sắp đầy đĩa?"],
    },
    "en": {
        "whyHigh": lambda m, c: f"Why is '{c}' so much higher on {m} than elsewhere?",
        "seeFails": lambda m: f"Show recent failed products from {m}",
        "compare": lambda a, b: f"Export a report comparing {a} and {b}",
        "longer": "Compare the last 30 days instead of 7",
        "sameOther": lambda f: f"Export this same report as {f}",
        "allFive": "Export the report for all 5 machines",
        "same30": "Compare a 30-day period for this same set",
        "disk": lambda m, g: f"{m} has {g} GB of disk left — what should be cleared?",
        "hot": lambda m, t: f"{m} is at {t:.0f}°C — is that dangerous?",
        "silent": lambda m: f"Why is {m} not answering?",
        "framesAround": lambda u, m: f"Show the frames either side of {u}'s recipe change on {m}",
        "whoElse": lambda m: f"Who else touched a recipe on {m} this week?",
        "acrossMachines": lambda u: f"What has {u} done on the other machines?",
        "busiest": lambda m, n: f"{m} has {n} actions — which machine has the most?",
        "actionDetail": lambda a, m: f"Detail every {a} on {m}",
        "onShift": "Who is on shift right now?",
        "deptSplit": lambda m: f"Which departments does {m} have, and how many people each?",
        "errWorst": lambda m, n: f"{m} logged {n} problem lines — which repeats most?",
        "errSince": lambda m: f"When did the errors on {m} start?",
        "tplSplit": lambda m: f"On {m}, which capture position do the failures land on?",
        "wall": "Show the latest image from every machine",
        "fallback": ["Fleet output over the last 7 days",
                     "Which machine has an unusual failure fingerprint?",
                     "Which machine is running hot or filling its disk?"],
    },
}


def build(tool_calls: List[str], results: Dict[str, Any],
          lang: str = "vi") -> List[str]:
    """
    Gợi ý theo thứ tự cụ thể → chung. Rỗng nếu không suy được gì.

    `results` là dict {tên tool: kết quả}. Chỉ đọc, không gọi lại gì.
    """
    out: List[str] = []
    used = set(tool_calls or [])
    L = _L.get(lang) or _L["vi"]

    fp_res = results.get("compare_failure_modes") or {}
    fp = fp_res.get("fingerprint") or {}
    if fp:
        machine, cause, gap = _outlier(fp)
        if machine:
            label = (fp.get("cause_labels") or {}).get(cause, cause)
            if "ask_machine" not in used:
                out.append(L["whyHigh"](machine, label))
            out.append(L["seeFails"](machine))

    prod = results.get("fleet_production") or {}
    if prod.get("machines"):
        names = [m["machine"] for m in prod["machines"] if m.get("production")]
        if len(names) >= 2:
            out.append(L["compare"](names[0], names[1]))
        if prod.get("period_days") and prod["period_days"] <= 7:
            out.append(L["longer"])

    rep = results.get("generate_fleet_report") or {}
    if rep.get("file"):
        ms = rep.get("machines") or []
        # Mời định dạng KHÁC cái vừa xuất. Bản đầu luôn mời Excel, kể cả ngay sau
        # khi vừa xuất Excel — gợi ý mời làm lại đúng việc vừa làm thì vô dụng.
        other = {"excel": "PDF", "pdf": "Excel", "html": "PDF",
                 "csv": "Excel"}.get(rep.get("format"), "PDF")
        out.append(L["sameOther"](other))
        if len(ms) < 5:
            out.append(L["allFive"])
        out.append(L["same30"])

    # Nhật ký thao tác. Thiếu nhánh này thì mọi câu hỏi về nhật ký rơi xuống bộ
    # ba mặc định — gợi ý không sai, mà không tồn tại.
    aud = results.get("fleet_audit_log") or {}
    by_mach = aud.get("by_machine") or {}
    if by_mach:
        focus = aud.get("machine_filter")
        if not focus:
            focus = max(by_mach, key=lambda k: by_mach[k]["total"])
        row = by_mach.get(focus) or {}
        acts = row.get("by_action") or {}
        who = (aud.get("user_lookup") or {}).get("username")

        recipe_acts = {k: v for k, v in acts.items() if "recipe" in k}
        if recipe_acts and who:
            out.append(L["framesAround"](who, focus))
        if recipe_acts:
            top = max(recipe_acts, key=recipe_acts.get)
            out.append(L["actionDetail"](top, focus))
            out.append(L["whoElse"](focus))
        if who and len(by_mach) > 1:
            out.append(L["acrossMachines"](who))
        if len(by_mach) > 1:
            busiest = max(by_mach, key=lambda k: by_mach[k]["total"])
            out.append(L["busiest"](busiest, by_mach[busiest]["total"]))

    staff = results.get("fleet_staff") or {}
    if staff.get("users") or staff.get("by_department"):
        out.append(L["onShift"])
        byd = staff.get("by_department_and_machine") or {}
        if byd:
            out.append(L["deptSplit"](next(iter(byd))))

    errs = results.get("fleet_log_errors") or {}
    for m in (errs.get("machines") or []):
        n = m.get("total_problem_lines") or 0
        if n > 0:
            out.append(L["errWorst"](m.get("machine"), n))
            out.append(L["errSince"](m.get("machine")))
            break


    health = results.get("fleet_health") or {}
    for m in (health.get("machines") or []):
        x = m.get("metrics") or {}
        if (x.get("disk_percent") or 0) >= 85:
            out.append(L["disk"](m["machine"], x.get("disk_free_gb")))
            break
    for m in (health.get("machines") or []):
        t = (m.get("metrics") or {}).get("cpu_temp")
        if t is not None and t >= 85:
            out.append(L["hot"](m["machine"], t))
            break

    # Máy thiếu dữ liệu được ưu tiên hỏi tiếp: đó là thứ người vận hành cần biết
    # nhất mà lại dễ trôi qua nhất, vì bảng vẫn hiện đầy đủ các máy còn lại.
    for res in results.values():
        cov = (res or {}).get("coverage") or {}
        for m in (cov.get("machines_missing") or []) + (cov.get("machines_degraded") or []):
            q = L["silent"](m["machine"])
            if q not in out:
                out.insert(0, q)
            break

    if not out:
        out = list(L["fallback"])

    seen, uniq = set(), []
    for q in out:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq[:MAX]
