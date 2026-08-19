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

from typing import Any, Dict, List, Optional

_MAX_BARS = 12
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


def _bar(title: str, series: List[Dict[str, Any]], unit: str = "") -> Optional[Dict[str, Any]]:
    series = [s for s in series if s.get("value") is not None][:_MAX_BARS]
    if not series:
        return None
    return {"type": "bar", "title": title, "unit": unit, "series": series}


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

        window = None
        if p.get("first_seen") and p.get("last_seen"):
            a, b = p["first_seen"][11:19], p["last_seen"][11:19]
            window = a if a == b else f"{a} → {b}"

        cards.append({
            "title": p.get("full_name") or p.get("username"),
            "role_line": role_line or None,
            "subtitle": (f"@{p.get('username')}"
                         + (f" · {p['employee_code']}" if p.get("employee_code") else "")),
            "badge": p.get("role") or "unknown",
            "avatar": _avatar_url(p.get("avatar_url")),
            "inactive": p.get("account_exists") is False or p.get("is_active") is False,
            "stat": p.get("action_count"),
            "stat_label": "thao tác",
            # Chỉ giữ dòng có giá trị: user chưa khai bộ phận thì đừng hiện một
            # hàng "Bộ phận —" trống, thẻ trông như dữ liệu bị lỗi.
            "rows": [r for r in (
                ("Ca làm việc", p.get("shift")),
                ("Dây chuyền", p.get("production_line")),
                ("Email", p.get("email")),
                ("Điện thoại", p.get("phone_number")),
                ("Vào làm", p.get("hire_date")),
                ("Hoạt động", window),
                ("Thao tác", ", ".join(f"{k} ×{v}" for k, v in acts.items()) or None),
            ) if r[1]],
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
    if not any(k in result for k in ("samples", "download_url", "people")):
        return result

    out = dict(result)
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

    # `people` đã được biến thành thẻ hiển thị; để nguyên trong ToolMessage thì
    # LLM lại kể lại đúng những gì thẻ đang hiện, và mỗi người kèm cả URL ảnh.
    # Giữ lại phần tối thiểu để nó vẫn nói đúng số người và chức vụ.
    if "people" in result:
        out["people"] = [
            {"username": p.get("username"), "full_name": p.get("full_name"),
             "role": p.get("role"), "department": p.get("department"),
             "job_title": p.get("job_title"), "action_count": p.get("action_count")}
            for p in (result.get("people") or [])
        ]
        out["_cards"] = "<thẻ thông tin người thao tác đã được hệ thống đính kèm tự động>"
    return out


def charts_from_tool_result(tool_name: str, args: Dict[str, Any], result: Any) -> List[Dict[str, Any]]:
    """Biểu đồ suy từ kết quả tool."""
    if not isinstance(result, dict) or not result.get("success"):
        return []

    charts: List[Dict[str, Any]] = []

    # Tổng quan gom nhóm → cột số sản phẩm fail của từng nhóm
    if tool_name == "get_production_summary":
        group = args.get("group_by", "recipe")
        key = {"camera": "camera", "hour": "hour"}.get(group, "recipe")
        rows = result.get("breakdown") or []
        label = {"camera": "camera", "hour": "giờ", "recipe": "recipe"}[key]

        c = _bar(
            f"Sản phẩm FAIL theo {label} · {_period_label(args, result)}",
            [
                {
                    "label": f"{r.get(key)}h" if key == "hour" else str(r.get(key)),
                    "value": r.get("fail", 0),
                    "sub": f"{r.get('pass_rate')}% pass · {r.get('total'):,} sp",
                }
                for r in rows
            ],
            "sp",
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
            c = _bar(title, series, "%")
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
