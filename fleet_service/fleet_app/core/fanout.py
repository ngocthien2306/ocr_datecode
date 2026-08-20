"""
Gọi nhiều máy cùng lúc, và trả về cả những máy không trả lời.

Hai điều bắt buộc, cả hai đều là chuyện đúng/sai chứ không phải tối ưu:

**Song song.** Đo trên đội hình này: một câu hỏi tới agent edge mất 4–20s. Hỏi
tuần tự 5 máy là gần một phút — người dùng bỏ đi trước khi có kết quả.

**Máy hỏng vẫn phải xuất hiện trong kết quả.** Đây là chỗ dễ sai nhất của cả
tầng fleet: bỏ máy lỗi ra khỏi danh sách thì "tổng sản lượng 5 máy" thiếu một
máy mà nhìn vẫn hoàn toàn bình thường. Không ai phát hiện được. Nên hàm ở đây
KHÔNG BAO GIỜ ném exception và KHÔNG BAO GIỜ lược bớt máy — máy lỗi trở thành
một phần tử có `ok=False` kèm lý do đọc được.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Sequence

from fleet_app.core.registry import Machine

logger = logging.getLogger(__name__)


async def fan_out(machines: Sequence[Machine],
                  call: Callable[[Machine], Awaitable[Any]],
                  timeout: float) -> List[Dict[str, Any]]:
    """
    Chạy `call` trên mọi máy cùng lúc, mỗi máy một timeout riêng.

    Timeout đặt cho TỪNG máy chứ không cho cả lượt: đặt cho cả lượt thì một máy
    chậm sẽ cắt ngang cả những máy đã trả lời xong.
    """
    async def one(m: Machine) -> Dict[str, Any]:
        base = {"node_id": m.node_id, "machine": m.name, "ip": m.ip}
        try:
            data = await asyncio.wait_for(call(m), timeout=timeout)
            return {**base, "ok": True, "data": data}
        except asyncio.TimeoutError:
            return {**base, "ok": False, "error": f"quá {timeout:.0f}s không trả lời"}
        except Exception as e:                      # noqa: BLE001 — cố ý bắt hết
            logger.warning("fan_out lỗi ở %s: %s", m.name, e)
            return {**base, "ok": False, "error": f"{type(e).__name__}: {e}"}

    if not machines:
        return []
    return list(await asyncio.gather(*(one(m) for m in machines)))


def coverage(results: Sequence[Dict[str, Any]],
             degraded: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """
    Mẫu này che bao nhiêu phần đội hình.

    Đi kèm MỌI kết quả tổng hợp. Con số tổng mà không nói rõ nó gộp từ mấy máy
    thì người đọc không có cách nào biết mình đang nhìn thiếu — đúng lớp lỗi
    "hiện ra vẫn đẹp nhưng nội dung sai" mà agent service đã phải sửa nhiều lần.

    Phân biệt HAI mức thiếu, vì gộp chúng lại là tự rơi vào đúng cái bẫy trên:

      missing  — không lấy được gì từ máy đó
      degraded — lấy được một phần, phần còn lại lỗi

    Lần đầu viết hàm này tôi chỉ đếm `ok` của lời gọi fan-out, nên khi tắt thử
    agent của LineTine thì `system_metrics` vẫn về (backend còn sống) còn
    `service_status` thì lỗi — lời gọi không ném exception nên bị tính là đủ, và
    `complete` báo True trong khi một máy đang hỏng. `complete` chỉ đúng khi
    không có máy nào thiếu VÀ không có máy nào lỗi một phần.
    """
    bad = [r for r in results if not r.get("ok")]
    return {
        "machines_total": len(results),
        "machines_ok": len(results) - len(bad) - len(degraded),
        "machines_missing": [{"machine": r["machine"], "reason": r.get("error")}
                             for r in bad],
        "machines_degraded": list(degraded),
        "complete": not bad and not degraded,
    }
