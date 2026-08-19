"""
Dựng ảnh + biểu đồ kèm câu trả lời.

Nguyên tắc: suy TẤT ĐỊNH từ kết quả tool, không hỏi LLM. Nếu để LLM tự bịa số
cho biểu đồ thì có nguy cơ hình vẽ một đằng, số trong văn bản một nẻo — mà biểu
đồ lại là thứ người ta tin ngay bằng mắt, không kiểm lại.

Hai loại:
- images : ảnh visualize của frame fail, lấy từ `explain_failures`
- charts : dữ liệu thô để FE vẽ (không render ảnh ở server)

`chart` cố tình chỉ có một dạng — danh sách nhãn/giá trị — để FE chỉ phải viết
một bộ render. So sánh hai kỳ thì LLM gọi tool hai lần, thành hai chart cạnh
nhau, không cần kiểu chart riêng.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from agent_app.core.i18n import t, tcols, tseg

#: Số cột tối đa của một biểu đồ.
#:
#: Đặt 26 để phủ trọn một ngày theo giờ (24) và một tháng theo tuần. Giá trị cũ là
#: 12, và `_bar` cắt bằng `[:12]` — tức lấy 12 cột ĐẦU rồi bỏ im lặng phần còn
#: lại. Trên dữ liệu thật, biểu đồ "fail theo giờ" chỉ vẽ 0h–11h và bỏ 12h–19h,
#: trong đó có đúng giờ fail nhiều nhất trong ngày (14h, 32 fail, pass 93,94%).
#: Người xem thấy nửa ngày êm ả và kết luận cả ngày không có gì.
_MAX_BARS = 26
_MAX_IMAGES = 8

# URL tương đối để trang tự resolve theo origin đang phục vụ — chạy đúng ở cả
# localhost:8100 lẫn tunnel HTTPS. Hardcode host là hỏng một trong hai.
_UPLOAD_PREFIX = "/api/uploads/"


def _period_label(args: Dict[str, Any], result: Dict[str, Any]) -> str:
    """
    Nhãn kỳ dữ liệu, gắn vào tiêu đề biểu đồ.

    BẮT BUỘC phải có: khi user bảo "so sánh hôm nay với hôm qua", LLM gọi tool
    hai lần và ta sinh hai biểu đồ. Cùng tiêu đề thì người xem không phân biệt
    được cái nào là ngày nào — biểu đồ mất sạch ý nghĩa.
    """
    start, end = args.get("start_date"), args.get("end_date")
    if start or end:
        # Chỉ phần GIỜ mới đáng ghép thêm vào sau ngày. Nhánh else cũ trả về cả
        # chuỗi ngày khi tham số không có giờ, nên 'explain_failures' với
        # start=end='2026-08-19' sinh ra tiêu đề lặp ba lần:
        # "2026-08-19 2026-08-19–2026-08-19".
        def hm(v):
            return v[11:16] if v and len(v) > 15 else ""

        first, last = (start or end or "")[:10], (end or start or "")[:10]
        day = first if first == last else f"{first}→{last}"
        span = f"{hm(start)}–{hm(end)}".strip("–")
        return f"{day} {span}".strip() if span else day

    if result.get("date"):
        return str(result["date"])

    period = result.get("period") or {}
    if period.get("start"):
        return str(period["start"])[:10]

    return "hôm nay"


def _bar(title: str, series: List[Dict[str, Any]], unit: str = "",
         ordered: bool = False, scale_max: Optional[float] = None,
         ref: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Một biểu đồ cột.

    Quá `_MAX_BARS` thì KHÔNG cắt bừa phần đuôi. Dãy có thứ tự thời gian
    (`ordered=True`) thì giữ phần MỚI NHẤT, vì đó là phần người ta cần; dãy phân
    loại thì giữ các cột LỚN NHẤT, vì cột lớn là cột đáng xem. Và cả hai trường hợp
    đều ghi vào tiêu đề là đã gộp bao nhiêu cột — một biểu đồ thiếu cột mà không
    nói gì thì đọc thành "phần đó không có dữ liệu".
    """
    series = [s for s in series if s.get("value") is not None]
    if not series:
        return None

    dropped = 0
    if len(series) > _MAX_BARS:
        dropped = len(series) - _MAX_BARS
        if ordered:
            series = series[-_MAX_BARS:]
            title += f" · {dropped} {t('mốc đầu không hiện')}"
        else:
            series = sorted(series, key=lambda s: -(s.get("value") or 0))[:_MAX_BARS]
            title += f" · {dropped} {t('mục nhỏ hơn không hiện')}"

    # tseg() dịch từng đoạn của tiêu đề ghép: phần chữ được dịch, phần dữ liệu
    # (tên recipe, ngày, tên ca) không khớp bảng nên giữ nguyên.
    out: Dict[str, Any] = {"type": "bar", "title": tseg(title),
                           "unit": t(unit), "series": series}
    if scale_max:
        # Thang đo cố định để cột đo được so với một MỐC, không phải so với cột lớn
        # nhất trong chính nó. Với tiến độ chỉ tiêu, track đầy = đạt chỉ tiêu; để
        # renderer tự lấy max thì cột cuối luôn đầy track và mọi ngày đều trông
        # như đã hoàn thành.
        out["max"] = scale_max
    if ref:
        out["ref"] = ref
    return out


def images_from_tool_result(tool_name: str, result: Any) -> List[Dict[str, str]]:
    """Ảnh minh hoạ frame fail."""
    if tool_name != "explain_failures" or not isinstance(result, dict):
        return []

    out = []
    for s in (result.get("samples") or [])[:_MAX_IMAGES]:
        path = str(s.get("image_path") or "").lstrip("/")
        if not path:
            continue

        bits = [f"Camera {s.get('camera')}"]
        if s.get("timestamp"):
            bits.append(str(s["timestamp"])[11:19])
        if s.get("expected") is not None:
            bits.append(f"mong '{s.get('expected')}' → đọc '{s.get('recognized') or '(rỗng)'}'")

        out.append({
            "url": _UPLOAD_PREFIX + path,
            "caption": " · ".join(bits),
            "recipe": s.get("recipe_name") or "",
        })
    return out


def files_from_tool_result(tool_name: str, result: Any) -> List[Dict[str, Any]]:
    """
    File tải về suy từ kết quả tool.

    Chỉ `generate_report` sinh file. Dựng ở đây thay vì để LLM tự viết link vào
    câu trả lời: LLM hay bỏ mất tiền tố đường dẫn hoặc tự đổi tên file, mà link
    sai thì user không tải được gì và cũng không hiểu tại sao.
    """
    if tool_name != "generate_report" or not isinstance(result, dict):
        return []
    url = result.get("download_url")
    if not url:
        return []

    period = (result.get("period") or {}).get("label") or ""
    summary = result.get("summary") or {}
    bits = [str(result.get("format", "")).upper()]
    if result.get("size_kb") is not None:
        bits.append(f"{result['size_kb']:g} KB")
    if summary.get("total") is not None:
        bits.append(f"{summary['total']:,} sản phẩm")

    return [{
        "url": url,
        "filename": result.get("filename") or url.rsplit("/", 1)[-1],
        "label": f"Báo cáo {period}".strip(),
        "meta": " · ".join(b for b in bits if b),
    }]


_AVATAR_PREFIX = "/api/uploads/avatar/"


def _avatar_url(raw: Optional[str]) -> Optional[str]:
    """
    URL avatar dùng được từ agent service.

    Backend lưu `avatar_url` dạng `/api/upload/avatars/{file}` — đường dẫn của
    endpoint riêng bên :8000. Agent service không có route đó, nhưng nó đã mount
    `backend/uploads` ở `/api/uploads`, và file avatar nằm ở
    `backend/uploads/avatar/{file}`. Nên chỉ cần đổi sang đường dẫn tĩnh đó là
    ảnh hiện được mà không phải proxy qua backend — quan trọng vì backend có thể
    đang restart trong khi user vẫn xem thẻ.
    """
    if not raw:
        return None
    name = str(raw).rstrip("/").rsplit("/", 1)[-1]
    if not name:
        return None
    return _AVATAR_PREFIX + name


def cards_from_tool_result(tool_name: str, result: Any) -> List[Dict[str, Any]]:
    """
    Thẻ thông tin người thao tác, suy từ `get_audit_logs`.

    Dựng ở đây thay vì để LLM kể bằng văn xuôi: chức vụ và ảnh là dữ liệu tra
    được, còn LLM thì có xu hướng bịa chức vụ cho một username nó chưa từng thấy.
    """
    if tool_name != "get_audit_logs" or not isinstance(result, dict):
        return []

    cards = []
    for p in (result.get("people") or [])[:12]:
        acts = p.get("actions") or {}

        # Dòng ngay dưới tên: chức vụ — bộ phận. Đây là thứ người vận hành đọc
        # đầu tiên để biết "ai đây", còn `role` chỉ là quyền trong hệ thống.
        role_line = " · ".join(x for x in (p.get("job_title"), p.get("department")) if x)

        # Khoảng hoạt động kèm THỜI LƯỢNG. Hai mốc giờ trơ trọi buộc người đọc tự
        # trừ nhẩm, mà thời lượng mới là thứ so được với độ dài ca ghi ngay trên
        # thẻ — 7h50m trên ca 8h là bình thường, 0h06m thì là mới vào ca.
        window = None
        if p.get("first_seen") and p.get("last_seen"):
            a, b = p["first_seen"][11:19], p["last_seen"][11:19]
            if a == b:
                window = a
            else:
                window = f"{a} → {b}"
                try:
                    lo = datetime.fromisoformat(p["first_seen"])
                    hi = datetime.fromisoformat(p["last_seen"])
                    mins = int((hi - lo).total_seconds() // 60)
                    if mins > 0:
                        window += f"  ({mins // 60}h{mins % 60:02d}m)"
                except (ValueError, TypeError):
                    pass

        exists = p.get("account_exists") is not False

        # Tài khoản còn tồn tại thì hiện ĐỦ bộ hàng, thiếu giá trị điền "—".
        #
        # Trước đây tôi bỏ hẳn hàng rỗng, và hệ quả là mỗi thẻ có số hàng khác
        # nhau — người xem tưởng hệ thống hiển thị lung tung chứ không đoán được
        # là "người này chưa khai số điện thoại". Khung thẻ giống nhau thì chỗ
        # trống tự nói lên là thiếu dữ liệu.
        #
        # Tài khoản đã xoá thì không có hồ sơ để mà thiếu, nên chỉ hiện phần có
        # thật — bảy dòng "—" liền nhau chỉ là nhiễu.
        fields = (
            ("Ca làm việc", p.get("shift")),
            ("Dây chuyền", p.get("production_line")),
            ("Email", p.get("email")),
            ("Điện thoại", p.get("phone_number")),
            ("Vào làm", p.get("hire_date")),
            ("Hoạt động", window),
            ("Số giờ có mặt", f"{p['active_hours']}h" if p.get("active_hours") else None),
            ("Thao tác", ", ".join(f"{k} ×{v}" for k, v in acts.items()) or None),
        )
        # Nhãn dòng dịch tại đây; giá trị là dữ liệu thật (email, số điện thoại,
        # tên ca) nên tuyệt đối không đưa qua bảng dịch.
        rows = [(t(k), v if v else "—") for k, v in fields] if exists else [
            (t(k), v) for k, v in fields if v
        ]

        cards.append({
            "title": p.get("full_name") or p.get("username"),
            "role_line": role_line or None,
            "subtitle": (f"@{p.get('username')}"
                         + (f" · {p['employee_code']}" if p.get("employee_code") else "")),
            # Ghi rõ đây là QUYỀN trong phần mềm, không phải chức vụ.
            #
            # `role` là mức phân quyền (viewer/operator/supervisor/admin), còn
            # dòng ngay trên là chức vụ ngoài xưởng. Hai trục khác nhau, mà đặt
            # cạnh nhau không nhãn thì thẻ đọc như hai chức danh đá nhau:
            # "Maintenance Technician" mà badge lại ghi "operator", hay
            # "Quality Inspector" mà badge ghi "supervisor".
            "badge": f"{t('Quyền')}: {p.get('role') or t('không rõ')}",
            "badge_role": p.get("role") or "unknown",
            "avatar": _avatar_url(p.get("avatar_url")),
            "inactive": not exists or p.get("is_active") is False,
            "stat": p.get("action_count"),
            "stat_label": t("thao tác"),
            "rows": rows,
        })
    return cards


def strip_for_llm(result: Any) -> Any:
    """
    Bản rút gọn của kết quả tool để đưa vào ToolMessage.

    Bỏ `samples` (đường dẫn ảnh): LLM nhìn thấy đường dẫn là tự viết
    `![Ảnh](inference_results/...)` vào câu trả lời — đường dẫn đó thiếu tiền tố
    /api/uploads nên không phải URL hợp lệ, hiện ra dưới dạng ký tự thô. Ảnh đã
    được hệ thống gắn tự động qua trường `images`, LLM không cần chạm vào.
    Bỏ luôn cũng giúp mỗi ToolMessage nhẹ đi ~2.500 ký tự.
    """
    if not isinstance(result, dict):
        return result
    if not any(k in result for k in ("samples", "download_url", "people", "by_recipe",
                                     "sample_covers_all", "equipment_alerts")):
        return result

    out = dict(result)

    # Mẫu không phủ hết kỳ: đổi số TUYỆT ĐỐI của nguyên nhân thành PHẦN TRĂM.
    #
    # Để nguyên số đếm thì câu trả lời ghi "tổng fail 1.036" rồi liệt kê
    # "144 / 106 / 102" — ba con số đó là của mẫu 294, nhưng nằm ngay dưới con số
    # 1.036 nên đọc thành 144 trên 1.036. Note đã bảo phải nói theo tỷ lệ và mô
    # hình vẫn không làm; bỏ hẳn số đếm đi thì nó không còn gì để trình bày sai.
    if result.get("sample_covers_all") is False:
        n = result.get("examined_products") or 0
        if n:
            out["causes"] = [
                {"cause": c.get("cause"), "label": c.get("label"),
                 "percent_of_sample": round((c.get("products") or 0) / n * 100, 1)}
                for c in (result.get("causes") or [])
            ]
            out["text_mismatch_kinds"] = [
                {"kind": k.get("kind"), "label": k.get("label"),
                 "percent_of_sample": round((k.get("count") or 0) / n * 100, 1)}
                for k in (result.get("text_mismatch_kinds") or [])
            ]
            out.pop("causes_by_product", None)
            out.pop("failed_frames_by_camera", None)
            out["_counts"] = (
                f"<số đếm tuyệt đối đã bỏ: mẫu chỉ {n} trên tổng "
                f"{result.get('total_failed_products')} sản phẩm fail. Nói theo TỶ LỆ "
                f"(`percent_of_sample`), đừng đưa số đếm — đặt một con số của mẫu cạnh "
                f"tổng số fail của kỳ sẽ bị đọc thành số của cả kỳ>"
            )

    # Bản giao ca: bỏ bốn con số mà ô KPI đã hiện.
    #
    # Ba lần trước bảo prompt "đừng đọc lại ô KPI" đều không giữ được, và bản giao
    # ca cũng vậy — nó liệt kê đủ tổng/pass/uptime/số phút dừng ngay trên bốn ô hiện
    # đúng bốn số đó. Bỏ số đi thì phần văn xuôi buộc phải làm việc của nó: kết luận
    # và việc ca sau cần để ý. Giữ `fail_causes` và `recipe_changes` vì đó là thứ ô
    # KPI không nói được.
    if "equipment_alerts" in result and "production" in result:
        pr = dict(result.get("production") or {})
        # GIỮ sản lượng ca, dù ô KPI đã hiện.
        #
        # Bỏ nó đi thì mô hình lấy `target.actual_day` — sản lượng CẢ NGÀY — và gọi
        # đó là sản lượng của ca: KPI ghi ca 2.828 còn câu văn ghi 10.171. Gán sai
        # phạm vi tệ hơn trùng lặp nhiều, vì trưởng ca có thể lấy con số đó báo cáo
        # lên trên. Phần ồn nhất (uptime, số phút dừng) vẫn được bỏ.
        out["production"] = {
            "total_this_shift": pr.get("total"),
            "pass_rate": pr.get("pass_rate"),
            "by_recipe": pr.get("by_recipe"),
        }
        dt = dict(result.get("downtime") or {})
        out["downtime"] = {"stop_count": dt.get("stop_count"),
                           "stops": dt.get("stops")} if dt else None
        # Đổi tên khoá của chỉ tiêu cho không thể lẫn phạm vi.
        if out.get("target"):
            tg = dict(out["target"])
            out["target"] = {
                "target_whole_day": tg.get("target"),
                "actual_whole_day": tg.get("actual_day"),
                "achieved_percent_whole_day": tg.get("achieved_percent"),
                "projected_whole_day": tg.get("projected_end_of_day"),
            }
        out["_numbers"] = ("<ô KPI đã hiện uptime và số phút dừng — đừng liệt kê lại. "
                           "`total_this_shift` là sản lượng CA NÀY; các khoá "
                           "`*_whole_day` là của CẢ NGÀY, đừng dùng chúng làm số của ca>")

    if "samples" in result:
        out["samples"] = f"<{len(result.get('samples') or [])} ảnh minh hoạ đã được hệ thống đính kèm tự động>"

    # Cùng lý do, và đã quan sát thấy thật: nhìn thấy `download_url` là LLM tự
    # viết markdown link — có lần nó bịa ra cả hostname
    # `https://example.com/api/reports/...`, một liên kết chết mà user bấm vào
    # không hiểu tại sao lỗi. Nút tải đã được dựng tất định ở trường `files`, nên
    # LLM không cần và không nên chạm vào URL.
    if "download_url" in result:
        # Bỏ HẲN các key này, không thay bằng chuỗi mô tả: thử để lại một
        # placeholder thì LLM đem đúng chuỗi đó nhét vào markdown, ra
        # `![Tải báo cáo](<nút tải file đã được hệ thống đính kèm...>)`. Không
        # còn key nào để bám thì nó mới chịu chỉ nói bằng lời. `success`,
        # `format` và `size_kb` vẫn ở lại nên nó biết file đã tạo xong.
        for k in ("download_url", "filename"):
            out.pop(k, None)

    # Kết quả compare_periods: giữ phần ĐỊNH TÍNH, bỏ con số thô.
    #
    # Bảo prompt "đừng đọc lại ô KPI" không giữ được — LLM vẫn liệt kê đủ bốn
    # dòng tổng/pass/fail/tỷ lệ, mỗi con số hai lần trên cùng một câu trả lời.
    # Cách chắc chắn là cách đã dùng cho `download_url`: không đưa con số cho nó.
    # Ô KPI và bảng đã hiện đầy đủ, còn việc của văn xuôi là kết luận và nguyên
    # nhân — thứ không cần tới con số tuyệt đối.
    if "by_recipe" in result and "period_a" in result:
        keep = {
            "success", "period_a", "period_b", "same_length", "baseline_usable",
            "baseline_note", "recipe_scope", "note",
        }
        out = {k: v for k, v in result.items() if k in keep}
        out["_numbers"] = ("<ô KPI và bảng so sánh đã hiện đầy đủ con số — "
                           "ĐỪNG liệt kê lại, hãy viết kết luận>")
        # Hướng thay đổi bằng lời, đủ để kết luận mà không đọc được ra số.
        out["direction"] = {
            k: result[k].get("direction") for k in ("total", "pass", "fail", "pass_rate")
            if isinstance(result.get(k), dict)
        }
        pr = result.get("pass_rate") or {}
        if pr.get("diff") is not None:
            # Một con số được phép: chênh lệch pass rate là kết luận chất lượng,
            # và nói "tăng 1,15 điểm" mới cụ thể hơn "tăng nhẹ".
            out["pass_rate_diff_points"] = pr["diff"]
        # Bỏ hai mảng ngày thô — biểu đồ đã vẽ chúng. Giữ lại kết luận về dạng thay
        # đổi và ba ngày lệch nhất, vì đó là thứ văn xuôi cần nói.
        pt = result.get("pattern") or {}
        if pt:
            out["pattern"] = {
                "shape": pt.get("shape"),
                "biggest_day": pt.get("biggest_day"),
                "biggest_day_share_pct": pt.get("biggest_day_share_pct"),
                "top_day_diffs": sorted(pt.get("per_day_diff") or [],
                                        key=lambda x: -abs(x["diff"]))[:3],
            }
        out.pop("by_day", None)
        out["per_working_day"] = result.get("per_working_day")
        out["per_day_direction"] = (
            "tăng" if (result.get("per_day") or {}).get("current", 0)
            > (result.get("per_day") or {}).get("previous", 0) else "giảm"
        )
        # Điểm bất thường — đây mới là thứ văn xuôi cần nêu.
        # Đảo cách diễn đạt: `absent_from` (VẮNG ở kỳ nào) thay cho `only_in`
        # (CHỈ CÓ ở kỳ nào).
        #
        # Cùng một sự thật, nhưng "only_in: 7 ngày qua" khiến mô hình suy ra
        # "recipe mới xuất hiện" — và nó viết đúng như vậy, kể cả sau khi note cấm
        # đích danh cụm từ đó. ONION POWDER vắng ở kỳ trước KHÔNG phải recipe mới:
        # nó vẫn chạy hôm nay và cả tháng trước. Nói "vắng ở kỳ X" thì không còn
        # chỗ để suy ra tính mới, vì câu đó chỉ nói về một kỳ chứ không nói về vòng
        # đời sản phẩm.
        pa_label = (result.get("period_a") or {}).get("label")
        pb_label = (result.get("period_b") or {}).get("label")
        notable = []
        for r in (result.get("by_recipe") or []):
            if not (r.get("only_in") or r.get("all_failed") or r.get("baseline_too_small")):
                continue
            item: Dict[str, Any] = {"recipe_name": r["recipe_name"]}
            if r.get("only_in"):
                item["absent_from"] = pb_label if r["only_in"] == pa_label else pa_label
            if r.get("all_failed"):
                item["all_failed"] = True
            if r.get("baseline_too_small"):
                item["baseline_too_small"] = True
            notable.append(item)
        out["notable"] = notable
        out["recipes_in_both"] = [
            r["recipe_name"] for r in (result.get("by_recipe") or [])
            if not r.get("only_in")
        ]
        return out

    # `people` đã được biến thành thẻ hiển thị; để nguyên trong ToolMessage thì
    # LLM lại kể lại đúng những gì thẻ đang hiện, và mỗi người kèm cả URL ảnh.
    # Giữ lại phần tối thiểu để nó vẫn nói đúng số người và chức vụ.
    if "people" in result:
        out["people"] = [
            {"username": p.get("username"), "full_name": p.get("full_name"),
             "role": p.get("role"), "department": p.get("department"),
             "job_title": p.get("job_title"), "action_count": p.get("action_count"),
             # Phải giữ: câu "ai làm việc nhiều giờ nhất" cần con số này, mà cắt
             # nó đi thì LLM chỉ còn `action_count` để xếp hạng và trả lời bằng
             # số thao tác — sai đơn vị, và nghe vẫn như một câu trả lời.
             "active_hours": p.get("active_hours")}
            for p in (result.get("people") or [])
        ]
        out["_cards"] = "<thẻ thông tin người thao tác đã được hệ thống đính kèm tự động>"
    return out


def _tile(label: str, value: Any, sub: Optional[str] = None,
          delta: Optional[Dict[str, Any]] = None, accent: str = "",
          lower_is_better: bool = False, delta_kind: str = "rel",
          delta_fmt: str = "{:+,.0f}") -> Dict[str, Any]:
    """
    Một ô KPI.

    `accent` chỉ là màu nhận dạng của con số (ok/bad/rỗng). `lower_is_better` là
    thứ quyết định mũi tên xanh hay đỏ. Hai việc này phải tách: gộp vào một
    trường thì ô Fail vừa bị sơn đỏ vĩnh viễn — kể cả ngày fail giảm — vừa dùng
    chính màu đó để đảo cực mũi tên. Trong xưởng, đỏ là màu báo động; một ô đỏ
    thường trực là tiếng ồn, và nó che đúng cái ngày lẽ ra phải báo động.

    `delta_kind` là ĐƠN VỊ của chênh lệch: 'pp' cho điểm phần trăm (pass rate),
    'rel' cho phần trăm tương đối (sản lượng). Thiếu nó thì lớp vẽ phải tự đoán,
    và nó đã đoán sai — ô pass rate hiển thị con số tương đối ngay dưới một giá
    trị phần trăm, đọc thành điểm phần trăm. Vì pass rate luôn quanh 98%, hai con
    số chỉ lệch nhau chút ít nên sai mà không ai phát hiện.
    """
    # Dịch ở ĐÂY, không ở chỗ gọi: có hơn 30 lời gọi _tile() rải khắp file, bọc
    # t() vào từng cái vừa dài vừa dễ bỏ sót một ô. Nhãn nào chưa có trong bảng
    # dịch thì t() trả về nguyên văn nên không có ô nào bị trống.
    out: Dict[str, Any] = {"label": t(label), "value": value, "accent": accent}
    if sub:
        out["sub"] = tseg(sub)
    if delta and delta.get("diff") is not None:
        d = delta["diff"]
        out["delta"] = {
            # Con số người vận hành cần là chênh lệch tuyệt đối; phần trăm tương
            # đối là phụ. Bỏ tuyệt đối đi thì "▲ +21,6%" không nói của bao nhiêu,
            # và văn xuôi buộc phải nhắc lại — chính là chỗ trùng lặp phải dẹp.
            "text": (f"{d:+.2f} {t('điểm')}" if delta_kind == "pp"
                     else delta_fmt.format(d)),
            "rel": (f"{delta['change_pct']:+.1f}%"
                    if delta.get("change_pct") is not None
                    and abs(delta["change_pct"]) < 1000 else None),
            "direction": delta.get("direction"),
            "good": (delta.get("direction") == "down") if lower_is_better
                    else (delta.get("direction") == "up"),
            "previous": delta.get("previous"),
        }
    return out


def _rate_accent(rate: Optional[float]) -> str:
    """Màu của pass rate theo NGƯỠNG, không phải cố định xanh.

    Trước đây ô Pass rate luôn tô xanh, nên một recipe fail sạch 18/18 vẫn hiện
    "Pass rate 0%" màu xanh thành công."""
    if rate is None:
        return ""
    return "ok" if rate >= 95 else "warn" if rate >= 90 else "bad"


def kpis_from_tool_result(tool_name: str, result: Any) -> List[Dict[str, Any]]:
    """
    Dãy ô KPI suy tất định từ kết quả tool.

    Lý do tồn tại: con số quan trọng nhất của câu trả lời chỉ nằm trong văn xuôi
    do LLM viết. Dựng ô riêng từ chính kết quả tool thì số trên giao diện không
    thể lệch khỏi số trong cơ sở dữ liệu.
    """
    if not isinstance(result, dict) or not result.get("success"):
        return []

    def fmt(n: Any) -> str:
        try:
            return f"{int(n):,}"
        except (TypeError, ValueError):
            return "—" if n is None else str(n)

    if tool_name == "compare_periods":
        pa, pb = result.get("period_a") or {}, result.get("period_b") or {}
        # Phụ đề mang khoảng ngày thật: "so với kỳ liền trước" không cho biết là
        # 7 ngày nào, mà đó là thứ người xem cần để tin con số.
        base = f"so với {pb.get('label')}"
        if not result.get("baseline_usable"):
            base = result.get("baseline_note") or "kỳ đối chiếu không đủ dữ liệu"
        rate = result["pass_rate"]["current"]
        return [
            _tile("Tổng sản phẩm", fmt(result["total"]["current"]), base, result["total"]),
            _tile("Pass", fmt(result["pass"]["current"]), base, result["pass"], "ok"),
            _tile("Fail", fmt(result["fail"]["current"]), base, result["fail"],
                  "", lower_is_better=True),
            _tile("Pass rate", f"{rate}%" if rate is not None else "—",
                  f"{pa.get('label')} · {base}", result["pass_rate"],
                  _rate_accent(rate), delta_kind="pp"),
        ]

    if tool_name == "get_shift_handover":
        # Ca chưa bắt đầu: vẽ ô KPI của LẦN GẦN NHẤT, nhãn ghi rõ ngày.
        # Không vẽ gì thì câu trả lời "chưa bắt đầu" trơ trọi không có số nào,
        # mà số của ca đêm hôm trước lại đúng là thứ người hỏi cần.
        if result.get("not_started"):
            prev = result.get("previous_occurrence")
            if not prev:
                return []
            result = {**prev, "shift": f"{prev['shift']} · {prev['date']}",
                      "in_progress": False}
        pr = result.get("production") or {}
        if not pr.get("total"):
            return []
        dt = result.get("downtime") or {}
        alerts = len(result.get("equipment_alerts") or [])
        rate = pr.get("pass_rate")
        return [
            _tile("Sản lượng ca", fmt(pr.get("total")),
                  result.get("shift") + (" · đang chạy" if result.get("in_progress") else "")),
            _tile("Pass rate", f"{rate}%", None, None, _rate_accent(rate)),
            _tile("Uptime", f"{dt.get('uptime_percent')}%" if dt.get("uptime_percent") is not None else "—",
                  f"dừng {dt.get('minutes', 0):.0f} phút" if dt else None,
                  None, _rate_accent(dt.get("uptime_percent"))),
            # Ô cảnh báo tô đỏ khi có việc trong ca. 0 thì để trung tính — một ô
            # đỏ thường trực là tiếng ồn, đúng lỗi đã sửa ở ô Fail.
            _tile("Cảnh báo trong ca", str(alerts),
                  "không có gì bất thường" if not alerts else "cần xem",
                  None, "" if not alerts else "bad"),
        ]

    if tool_name == "get_target_progress":
        pct = result.get("achieved_percent")
        tgt, act = result.get("target"), result.get("actual")
        gap = result.get("gap") or 0
        proj = result.get("projected_end_of_day")
        rate, minr = result.get("pass_rate"), result.get("min_pass_rate")
        # Màu của ô sản lượng theo DỰ PHÓNG khi ngày còn đang chạy, không theo %
        # đã đạt. Giữa ngày mới xong 70% là bình thường; tô đỏ ở đó là báo động
        # sai, mà dự phóng lại đang cho thấy sẽ vượt chỉ tiêu.
        if result.get("reached"):
            vol_accent = "ok"
        elif proj is not None:
            vol_accent = "ok" if proj >= (tgt or 0) else "bad"
        else:
            vol_accent = "warn" if (pct or 0) >= 80 else "bad"
        tiles = [
            _tile("Sản lượng", f"{fmt(act)} / {fmt(tgt)}",
                  result.get("scope"), None, vol_accent),
            _tile("Hoàn thành", f"{pct}%" if pct is not None else "—",
                  # Nói rõ còn thiếu / đã vượt, thay vì để người đọc tự trừ.
                  ("đã vượt " + fmt(-gap)) if gap < 0 else f"còn thiếu {fmt(gap)}"),
        ]
        if proj:
            tiles.append(_tile(
                "Dự phóng cuối ngày", fmt(proj),
                # Nhãn phải nằm ngay trên ô: một con số tròn trịa không có nhãn sẽ
                # được đọc như cam kết, chứ không như phép ngoại suy.
                "ngoại suy theo nhịp hiện tại",
                None, "ok" if proj >= (tgt or 0) else "bad"))
        if rate is not None:
            tiles.append(_tile(
                "Pass rate", f"{rate}%",
                f"ngưỡng {minr}%" if minr is not None else None, None,
                ("ok" if result.get("quality_ok") else "bad")
                if result.get("quality_ok") is not None else _rate_accent(rate)))
        return tiles

    if tool_name == "get_downtime":
        up = result.get("uptime_percent")
        down = result.get("downtime_minutes") or 0
        h, m = int(down // 60), int(down % 60)
        return [
            _tile("Uptime", f"{up}%" if up is not None else "—",
                  "trên khoảng có sản xuất", None, _rate_accent(up)),
            _tile("Thời gian dừng", f"{h}h{m:02d}m" if h else f"{m} phút"),
            _tile("Số lần dừng", fmt(result.get("stop_count"))),
            _tile("Sản phẩm", fmt(result.get("products")),
                  f"{str(result.get('first_product',''))[11:16]} → "
                  f"{str(result.get('last_product',''))[11:16]}"),
        ]

    if tool_name in ("get_pass_fail_stats", "get_production_summary"):
        summ = result.get("summary") or {}
        total = summ.get("total_products")
        if not total:
            return []
        rate = summ.get("pass_rate")
        pct = lambda n: f"{round((n or 0) / total * 100, 2)}% tổng"  # noqa: E731
        return [
            _tile("Tổng sản phẩm", fmt(total)),
            _tile("Pass", fmt(summ.get("pass_count")), pct(summ.get("pass_count")), None, "ok"),
            _tile("Fail", fmt(summ.get("fail_count")), pct(summ.get("fail_count"))),
            _tile("Pass rate", f"{rate}%", None, None, _rate_accent(rate)),
        ]

    return []


def _period_caption(p: Dict[str, Any]) -> str:
    """
    "nhãn: từ → đến (N ngày)", nhưng bỏ phần ngày nếu nhãn ĐÃ là khoảng ngày.

    Kỳ đối chiếu tự động lấy nhãn chính là khoảng ngày của nó, nên ghép máy móc
    sẽ ra "2026-08-06 → 2026-08-12: 2026-08-06 → 2026-08-12 (7 ngày)".
    """
    lo, hi = str(p.get("start", ""))[:10], str(p.get("end", ""))[:10]
    days = f"{p.get('days')} ngày"
    label = str(p.get("label", ""))
    rng = lo if lo == hi else f"{lo} → {hi}"
    if label == rng:
        return f"{rng} ({days})"
    return f"{label}: {rng} ({days})"


def _num(v: Optional[float], suffix: str = "") -> str:
    """Số cho ô bảng; vắng mặt là "—" chứ không phải 0.

    Một số 0 in ra phải luôn có nghĩa "đo được và bằng 0". Trước đây sản lượng
    vắng mặt hiện 0 còn pass rate vắng mặt hiện "—" — cùng một sự thật, hai cách
    viết, trong cùng một hàng."""
    if v is None:
        return "—"
    return (f"{v:,.0f}" if suffix == "" else f"{v}{suffix}")


def tables_from_tool_result(tool_name: str, result: Any) -> List[Dict[str, Any]]:
    """
    Bảng dữ liệu tất định.

    Thay cho việc để LLM tự dựng bảng markdown: nó hay bỏ sót hàng khi danh sách
    dài, và định dạng số mỗi lượt một kiểu.
    """
    if not isinstance(result, dict) or not result.get("success"):
        return []

    if tool_name == "get_shift_handover":
        if result.get("not_started"):
            result = result.get("previous_occurrence") or {}
            if not result:
                return []
        out = []
        ch = result.get("recipe_changes") or []
        if ch:
            out.append({
                "title": f"Thay đổi recipe trong ca · {result.get('shift')}",
                "columns": tcols(["Giờ", "Người", "Thao tác", "Nội dung"]),
                "align": ["l", "l", "l", "l"],
                "rows": [[c["time"], c["username"], c["action"], c["description"][:70]]
                         for c in ch[:12]],
                "caption": "Đổi recipe thường đi kèm dừng máy và một cú vọt fail ngay sau.",
            })
        st = ((result.get("downtime") or {}).get("stops")) or []
        if st:
            out.append({
                "title": f"Các lần dừng trong ca · {result.get('shift')}",
                "columns": tcols(["Từ", "Đến", "Số phút"]),
                "align": ["l", "l", "r"],
                "rows": [[s["from"][11:16], s["to"][11:16], f"{s['minutes']:,.0f}"] for s in st],
            })
        return out

    if tool_name == "get_downtime":
        stops = result.get("stops") or []
        if not stops:
            return []
        return [{
            "title": f"Các lần dừng dây chuyền · {str(result.get('period',{}).get('start',''))[:10]}",
            "columns": tcols(["Từ", "Đến", "Số phút"]),
            "align": ["l", "l", "r"],
            "rows": [[s["from"][:16].replace("T", " "), s["to"][11:16], f"{s['minutes']:,.0f}"]
                     for s in stops],
            "caption": (f"Khe hở ≥ {result.get('min_gap_minutes')} phút giữa hai sản phẩm. "
                        f"Cho biết không có sản phẩm đi qua, không cho biết nguyên nhân."),
        }]

    if tool_name == "compare_periods":
        rows = result.get("by_recipe") or []
        if not rows:
            return []
        pa, pb = result.get("period_a") or {}, result.get("period_b") or {}
        scope = result.get("recipe_scope")
        title = f"So sánh theo recipe · {pa.get('label')} vs {pb.get('label')}"
        if scope and scope != "tất cả recipe":
            title += f" · chỉ {scope}"

        body = []
        for r in rows:
            tot, pr = r["total"], r["pass_rate"]
            if r.get("only_in"):
                note = f"chỉ chạy ở {r['only_in']}"
            elif r.get("baseline_too_small"):
                note = f"nền quá ít ({_num(pr['previous'] and tot['previous'])} sp)"
            else:
                note = ""
            if r.get("all_failed"):
                note = ("⚠ fail toàn bộ" + (f" · {note}" if note else ""))
            body.append([
                r["recipe_name"],
                _num(tot["previous"]), _num(tot["current"]),
                f"{tot['change_pct']:+.1f}%" if tot.get("change_pct") is not None else "—",
                _num(pr["previous"], "%"), _num(pr["current"], "%"),
                # "đ" là ký hiệu đồng ở Việt Nam — "+1.36đ" đọc thành "1,36 đồng"
                # trong một cột về chất lượng. Viết đủ chữ "điểm".
                f"{pr['diff']:+.2f} {t('điểm')}" if pr.get("diff") is not None else "—",
                note,
            ])
        return [{
            "title": title,
            # Tiêu đề ngắn: nhãn kỳ đầy đủ đã nằm ở title, nhồi thêm vào từng cột
            # làm bảng rộng ra tới 1.300px và cột quyết định bị đẩy khỏi màn hình.
            "columns": tcols(["Recipe", "SL trước", "SL nay", "Thay đổi SL",
                        "Pass trước", "Pass nay", "Chênh lệch (điểm %)", "Ghi chú"]),
            "align": ["l", "r", "r", "r", "r", "r", "r", "l"],
            "rows": body,
            "caption": " · ".join(_period_caption(p) for p in (pa, pb)),
        }]

    return []


def charts_from_tool_result(tool_name: str, args: Dict[str, Any], result: Any) -> List[Dict[str, Any]]:
    """Biểu đồ suy từ kết quả tool."""
    if not isinstance(result, dict) or not result.get("success"):
        return []

    charts: List[Dict[str, Any]] = []

    # Tổng quan gom nhóm → cột số sản phẩm fail của từng nhóm
    if tool_name == "get_production_summary":
        group = args.get("group_by", "recipe")
        key = {"camera": "camera", "hour": "hour", "shift": "shift"}.get(group, "recipe")
        rows = result.get("breakdown") or []
        label = {"camera": "camera", "hour": "giờ", "recipe": "recipe", "shift": "ca"}[key]
        # Dịch CẢ CỤM chứ không riêng chữ "camera"/"giờ": tiếng Anh đảo trật tự
        # ("FAIL units by hour"), ghép từng từ sẽ ra câu sai ngữ pháp.
        head = t(f"Sản phẩm FAIL theo {label}")

        c = _bar(
            f"{head} · {_period_label(args, result)}",
            [
                {
                    "label": f"{r.get(key)}h" if key == "hour" else str(r.get(key)),
                    "value": r.get("fail", 0),
                    "sub": f"{r.get('pass_rate')}% pass · {r.get('total'):,} sp",
                }
                for r in rows
            ],
            "sp",
            ordered=(key == "hour"),
        )
        if c:
            charts.append(c)

    # Xu hướng → cột pass rate theo từng mốc thời gian
    elif tool_name == "get_pass_fail_stats":
        trend = result.get("trend") or {}
        series = []
        for when in sorted(trend):
            v = trend[when]
            total = (v.get("pass", 0) or 0) + (v.get("fail", 0) or 0)
            if not total:
                continue
            series.append({
                "label": when,
                "value": round(v.get("pass", 0) / total * 100, 2),
                "sub": f"{v.get('fail', 0):,} fail / {total:,} sp",
            })
        recipe = args.get("recipe_id")
        if len(series) > 1:
            title = "Pass rate theo thời gian" + (f" · {recipe}" if recipe else "")
            c = _bar(title, series, "%", ordered=True)
            if c:
                charts.append(c)
        else:
            # Chỉ một mốc thời gian thì không vẽ được xu hướng, nhưng vẫn phải
            # có hình: thiếu nó LLM tự chế biểu đồ bằng ký tự ███ trong văn bản.
            summ = result.get("summary") or {}
            if summ.get("total_products"):
                title = f"PASS / FAIL · {_period_label(args, result)}"
                if recipe:
                    title += f" · {recipe}"
                c = _bar(
                    title,
                    [
                        {"label": "PASS", "value": summ.get("pass_count", 0),
                         "sub": f"{summ.get('pass_rate')}%"},
                        {"label": "FAIL", "value": summ.get("fail_count", 0),
                         "sub": f"{summ.get('fail_rate')}%"},
                    ],
                    "sp",
                )
                if c:
                    charts.append(c)

    # Tiến độ chỉ tiêu → luỹ tiến theo giờ, thang đo là chỉ tiêu
    elif tool_name == "get_target_progress":
        rows = result.get("cumulative_by_hour") or []
        target = result.get("target")
        if rows and target:
            c = _bar(
                f"Luỹ tiến so với chỉ tiêu · {result.get('scope')} · {result.get('date')}",
                [{"label": f"{r['hour']:02d}h", "value": r["cumulative"],
                  "sub": f"+{r['in_hour']:,} trong giờ"} for r in rows],
                "sp",
                ordered=True,
                scale_max=target,
                # Mốc tham chiếu: nếu sản xuất đều cả ngày thì tới giờ này phải đạt
                # bao nhiêu. Không có nó thì người xem không biết đang sớm hay muộn.
                ref={"label": f"chỉ tiêu {target:,}", "value": target},
            )
            if c:
                charts.append(c)

    # So sánh hai kỳ → cột kép sản lượng theo recipe
    #
    # Luồng so sánh trước đây không có hình nào: `charts_from_tool_result` chỉ
    # biết ba tool cũ, mà đây lại đúng là câu hỏi cần một cái nhìn nhất.
    elif tool_name == "compare_periods":
        pa, pb = result.get("period_a") or {}, result.get("period_b") or {}
        # Chỉ recipe CÓ sản lượng kỳ này. Recipe chỉ chạy kỳ trước sẽ ra cột bằng
        # 0 nằm cạnh các cột lớn — đọc thành "sụt về 0" chứ không phải "không
        # chạy". Chúng vẫn có mặt trong bảng, nơi cột Ghi chú nói rõ.
        rows = [r for r in (result.get("by_recipe") or []) if (r["total"]["current"] or 0)]
        dropped = [r["recipe_name"] for r in (result.get("by_recipe") or [])
                   if not (r["total"]["current"] or 0)]
        if rows:
            series = []
            for r in rows[:_MAX_BARS]:
                cur = r["total"]["current"] or 0
                prev = r["total"]["previous"] or 0
                # Vẽ theo sản lượng KỲ NÀY, phần kỳ trước đưa vào `sub`. Biểu đồ
                # cột đơn không diễn tả được hai giá trị, mà ghép hai cột cạnh
                # nhau thì recipe chỉ chạy một kỳ sẽ có một cột bằng 0 trông như
                # "sụt về 0" — đúng thứ bảng đang cố tránh.
                note = f"kỳ trước: {prev:,}" if prev else "kỳ trước: không có"
                if r.get("only_in"):
                    note = f"chỉ chạy ở {r['only_in']}"
                series.append({"label": r["recipe_name"], "value": cur, "sub": note})
            title = f"Sản lượng theo recipe · {pa.get('label')}"
            if dropped:
                # Nói ra recipe bị loại khỏi hình, không để nó lặng lẽ mất: một
                # biểu đồ thiếu recipe mà không giải thích đọc thành recipe đó
                # không tồn tại.
                title += f" (không gồm {len(dropped)} recipe không chạy kỳ này)"
            c = _bar(title, series, "sp")
            if c:
                charts.append(c)

        # Cột thứ hai: sản lượng theo NGÀY, kèm ngày đối chiếu ở dòng phụ.
        #
        # Vẽ giá trị TUYỆT ĐỐI chứ không vẽ chênh lệch: renderer tính bề rộng theo
        # tỷ lệ giá trị/max, nên một cột âm ra bề rộng âm và biến mất hoàn toàn —
        # đúng ngày tụt mạnh nhất sẽ không hiện.
        by_day = ((result.get("by_day") or {}).get("current")) or []
        prev_day = ((result.get("by_day") or {}).get("previous")) or []
        if len(by_day) > 1:
            series = []
            for i, d in enumerate(by_day):
                p0 = prev_day[i] if i < len(prev_day) else None
                sub = (f"{p0['date'][5:]}: {p0['total']:,}" if p0 else "kỳ trước: không có")
                if p0:
                    sub += f"  ({d['total'] - p0['total']:+,})"
                series.append({"label": d["date"][5:], "value": d["total"], "sub": sub})
            c = _bar(f"Sản lượng theo ngày · {pa.get('label')}", series, "sp", ordered=True)
            if c:
                charts.append(c)

        # Cột thứ ba: pass rate của các recipe chạy CẢ HAI kỳ, để nhìn ra chất
        # lượng đổi chiều. Recipe một kỳ không có gì để so nên loại khỏi biểu đồ
        # này thay vì vẽ một cột trơ.
        both = [r for r in (result.get("by_recipe") or [])
                if not r.get("only_in") and r["pass_rate"]["diff"] is not None]
        if both:
            c = _bar(
                f"Thay đổi pass rate · {pa.get('label')} so với {pb.get('label')}",
                [{"label": r["recipe_name"], "value": r["pass_rate"]["diff"],
                  "sub": f"{r['pass_rate']['previous']}% → {r['pass_rate']['current']}%"}
                 for r in both[:_MAX_BARS]],
                "điểm",
            )
            if c:
                charts.append(c)

    # Sản lượng theo ca / dừng máy đã có KPI và bảng riêng, không vẽ thêm.

    # Nguyên nhân fail → cột theo bước kiểm tra bị trượt
    elif tool_name == "explain_failures":
        nice = {
            "text_verification": "OCR đọc sai chuỗi",
            "char_verification": "Ký tự dưới ngưỡng",
            "template_verification": "Không khớp template",
            "product_verification": "Không nhận ra sản phẩm",
            "no_detection": "Không thấy nhãn trong khung",
            "unknown": "Chưa xác định",
        }
        # Đếm theo sản phẩm chứ không theo frame: cột trước đây gắn nhãn "frame"
        # nhưng người đọc vẫn hiểu là số sản phẩm, mà tổng các cột lại vượt xa
        # tổng sản phẩm fail (328 so với 167) nên biểu đồ trông như sai số liệu.
        # `causes` giờ là list (mỗi hàng có products + frames); chỉ
        # `causes_by_product` mới là dict {nguyên_nhân: số sản phẩm} mà _bar cần.
        causes = result.get("causes_by_product") or {}
        c = _bar(
            f"Nguyên nhân fail · {_period_label(args, result)}",
            [
                {"label": nice.get(k, k), "value": v, "sub": ""}
                for k, v in sorted(causes.items(), key=lambda x: -x[1])
            ],
            "sản phẩm",
        )
        if c:
            charts.append(c)

        by_cam = result.get("failed_frames_by_camera") or {}
        if len(by_cam) > 1:
            c = _bar(
                f"Frame fail theo camera · {_period_label(args, result)}",
                [{"label": k, "value": v, "sub": ""} for k, v in by_cam.items()],
                "frame",
            )
            if c:
                charts.append(c)

    return charts
