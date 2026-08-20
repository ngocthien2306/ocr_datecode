"""
Endpoint cho tầng fleet đọc: số liệu sản xuất gọn, KHÔNG đi qua LLM.

Vì sao cần, khi đã có `/api/agent/chat`: fleet phải poll 5 máy mỗi phút. Đi qua
agent thì mỗi vòng là 5 lượt LLM — đo thật trên đội hình này là 4–20s mỗi lượt,
và với `gpt-4o-mini` thì một cái dashboard poll 60s tốn cỡ 200 USD/tháng để hiện
lại đúng con số mà `get_pass_fail_stats` tính ra trong 0,2s miễn phí.

Endpoint ở đây gọi THẲNG các tool đã có, không dựng lại phép tính. Đó là chủ ý:
logic trong `analytics_tools` không phải "đọc dữ liệu" mà là tri thức đúng/sai đã
tích luỹ — `created_at` thay vì `timestamp` (198s → 0,2s), đếm theo SẢN PHẨM chứ
không theo FRAME, mốc ngày là giờ địa phương còn DB lưu naive-UTC. Viết lại ở
tầng fleet là rước lại đủ các lỗi đó, mà không test nào bắt được vì chúng đều
"hiện ra vẫn đẹp".

Khuôn theo `/api/agent/service/status` đã có: gọi thẳng tool, không tốn token.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from agent_app.api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fleet", tags=["Fleet"])

# Bộ nhớ đệm trong tiến trình. Fleet poll mỗi 60s và câu hỏi luôn cùng tham số,
# nên không cache thì mỗi vòng lại mổ lại đúng ngần ấy document fail. TTL ngắn
# hơn nhịp poll để số không bao giờ cũ quá một vòng.
_CACHE: Dict[str, tuple] = {}
_CACHE_TTL = 45.0


def _cached(key: str, build):
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit and hit[0] > now:
        return hit[1]
    val = build()
    # Chỉ cache kết quả thành công: lỗi thường là tạm thời (Mongo nghẽn), giữ
    # lại 45s nghĩa là người dùng thử lại ngay vẫn nhận đúng cái lỗi cũ.
    if isinstance(val, dict) and val.get("success"):
        _CACHE[key] = (now + _CACHE_TTL, val)
    return val


def _dates(days: int) -> tuple:
    end = datetime.now()
    start = end - timedelta(days=max(1, days) - 1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _fingerprint(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Vân tay kiểu lỗi: phân bố nguyên nhân, dạng TỈ LỆ chứ không phải số đếm.

    Đây là thứ duy nhất so sánh được giữa các máy khi chúng chạy sản phẩm khác
    nhau. Tỉ lệ pass tuyệt đối thì không: M2 pass 69% trên hạt tiêu và
    PC-Auto-1 pass 98% trên muối không nói lên máy nào tệ hơn. Nhưng
    "M2 có no_detection 43% còn mặt bằng ~10%" thì đúng dù khác sản phẩm, vì
    no_detection thuộc về camera/trigger/ánh sáng chứ không thuộc về hàng.

    Chỉ trả tỉ lệ, KHÔNG trả số đếm tuyệt đối. `explain_failures`
    lấy mẫu tối đa `sample_limit`, nên số đếm trong `causes` là của MẪU chứ không
    phải của cả kỳ — đưa nó ra cạnh `total_fail` là dựng sẵn cái bẫy đã ghi trong
    PIPELINE.md: "tổng fail 1.036" rồi liệt kê "144/106/102", ba số sau là của
    mẫu 294.
    """
    if not raw.get("success"):
        return None
    causes = raw.get("causes") or []
    total = sum(c.get("products", 0) for c in causes) or 1
    return {
        "sample_products": raw.get("failed_camera_frames_examined"),
        "sample_covers_all": raw.get("sample_covers_all"),
        "sampling": raw.get("sampling"),
        "by_cause": [
            {
                "cause": c["cause"],
                "label": c.get("label"),
                # Chuẩn hoá để tổng bằng 100: đây là TỈ TRỌNG giữa các nguyên
                # nhân, không phải "bao nhiêu % sản phẩm trượt bước này". Một sản
                # phẩm trượt được nhiều bước, nên con số chưa chuẩn hoá của
                # `explain_failures` cộng lại vượt 100 — trộn hai thứ đó vào một
                # câu trả lời là ra hai số khác nhau cho cùng một nguyên nhân.
                "share_of_causes_pct": round(c.get("products", 0) * 100.0 / total, 1),
            }
            for c in causes
        ],
        "template_similarity_avg": raw.get("template_similarity_avg"),
    }


@router.get("/rollup", summary="Số liệu sản xuất gọn cho tầng fleet (không LLM)")
async def rollup(
    days: int = Query(7, ge=1, le=90, description="Số ngày tính ngược từ hôm nay"),
    causes: bool = Query(True, description="Kèm vân tay kiểu lỗi (chậm hơn)"),
    sample_limit: int = Query(200, ge=50, le=1000),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Gói dữ liệu vài KB cho fleet: sản lượng, pass/fail, xu hướng, vân tay lỗi.

    Cố ý KHÔNG trả ảnh mẫu. `explain_failures` có trường `samples` chứa ảnh sản
    phẩm lỗi; đẩy chúng qua link tới Jetson (đo được vài chục KB/s) là biến một
    gói 3 KB thành vài MB. Fleet cần tỉ lệ, không cần ảnh — ai muốn xem ảnh thì
    đi đường ủy quyền hỏi thẳng agent của máy đó.
    """
    from agent_app.tools.analytics_tools import (
        explain_failures,
        get_pass_fail_stats,
        list_recipes,
    )

    start_date, end_date = _dates(days)
    out: Dict[str, Any] = {
        "success": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "period": {"start": start_date, "end": end_date, "days": days},
    }

    stats = _cached(f"stats:{start_date}:{end_date}",
                    lambda: get_pass_fail_stats(start_date=start_date,
                                                end_date=end_date, group_by="day"))
    if stats.get("success"):
        s = stats.get("summary") or {}
        out["production"] = {
            "total_products": s.get("total_products"),
            "pass": s.get("pass_count"),
            "fail": s.get("fail_count"),
            "pass_rate": s.get("pass_rate"),
            # Chuẩn hoá theo ngày: hai máy chạy số ngày khác nhau trong kỳ thì
            # sản lượng tuyệt đối không so được, per-day thì so được.
            "per_day": round((s.get("total_products") or 0) / max(days, 1), 1),
            "trend": stats.get("trend") or {},
        }
    else:
        out["production"] = None
        out["production_error"] = stats.get("error")

    if causes:
        raw = _cached(f"causes:{start_date}:{end_date}:{sample_limit}",
                      lambda: explain_failures(start_date=start_date,
                                               end_date=end_date,
                                               sample_limit=sample_limit))
        out["failure_modes"] = _fingerprint(raw)
        if not raw.get("success"):
            out["failure_modes_error"] = raw.get("error")

    recs = _cached(f"recipes:{days}", lambda: list_recipes(days=days))
    if recs.get("success"):
        items: List[Dict[str, Any]] = recs.get("recipes") or recs.get("items") or []
        out["recipes"] = [
            {"name": r.get("name") or r.get("recipe_name"),
             "products": r.get("total") or r.get("total_products")}
            for r in items[:10]
        ]

    return out
