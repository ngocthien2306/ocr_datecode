"""
Cache kết quả tool trong thời gian ngắn.

Vì sao cần: câu "thống kê 7 ngày qua" quét hàng trăm nghìn document. Người dùng
hỏi lại cùng câu đó — vì bấm chip gợi ý, vì mở lại tab, vì hai người cùng xem một
màn hình — là chạy lại toàn bộ truy vấn. Trong một phiên chat, việc đó lặp rất
nhiều: đúng chuỗi hai câu hỏi mà user gặp hôm nay đã gọi `explain_failures` hai
lần với tham số gần như y nhau.

TTL đặt theo việc KỲ ĐÃ ĐÓNG hay CHƯA:

- Kỳ còn mở (có chứa hôm nay): dây chuyền vẫn đang chạy, số liệu tăng từng giây.
  TTL ngắn — đủ để hai câu hỏi liền nhau trong một lượt chat dùng chung kết quả,
  không đủ để người dùng thấy một con số cũ.
- Kỳ đã đóng (hôm qua, tuần trước): dữ liệu không đổi nữa. TTL dài.

Cache theo tiến trình, không phải Redis: agent service là một tiến trình đơn trên
Jetson, thêm một dịch vụ nữa chỉ để cache là cái giá không đáng. Mất cache khi
restart cũng không sao — nó chỉ là tối ưu.

KHÔNG cache tool có tác dụng phụ (`start_service`, `stop_service`) và tool sinh
file (`generate_report`): trả lại một kết quả cũ ở đó nghĩa là nói "đã dừng service"
mà thật ra chưa làm gì, hoặc đưa lại đường dẫn một file đã bị dọn.
"""

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Kỳ còn mở: ngắn. Hai câu hỏi liền nhau trong một lượt chat cách nhau vài giây,
# nên 45s là đủ dùng chung; còn người dùng nhìn lại sau một phút thì đã hết hạn.
TTL_OPEN = 45.0
# Kỳ đã đóng: dữ liệu không đổi. 30 phút, đủ để một phiên xem báo cáo không phải
# quét lại cùng một tuần chục lần.
TTL_CLOSED = 1800.0

# Số đo phần cứng làm mới mỗi 2 giây. TTL 45s là quá dài (câu "máy còn nóng
# không" sau khi vừa hạ tải sẽ trả lại con số nóng cũ), nhưng KHÔNG cache thì tệ
# hơn: mô hình gọi get_system_metrics ba lần trong một lượt và đọc ra ba con số
# RAM khác nhau — đã gặp thật, câu trả lời ghi 85,83% ở đoạn đầu rồi 87,73% ở
# đoạn sau. 10 giây đủ để một lượt chat dùng chung một lần đo, và vẫn tươi.
TTL_LIVE = 10.0

# Category được cache. Cố tình liệt kê CÓ chứ không loại trừ: thêm category mới
# thì mặc định KHÔNG cache, và đó là mặc định an toàn — bỏ sót một tool đọc thì
# chỉ chậm, còn cache lỡ một tool ghi thì sai kết quả.
_CACHEABLE_CATEGORIES = {"analytics", "logs", "equipment", "system"}

# Tool đo trạng thái tức thời: dùng TTL_LIVE thay vì TTL_OPEN.
_LIVE_TOOLS = {"get_system_metrics", "get_system_alerts"}

# Tool không bao giờ cache dù thuộc category ở trên.
_NEVER_CACHE = {
    "generate_report",     # sinh file thật, đường dẫn cũ có thể đã bị dọn
    "start_service",
    "stop_service",
}

_MAX_ENTRIES = 256

_lock = threading.Lock()
# key -> (het_han, ket_qua)
_store: Dict[str, Tuple[float, Any]] = {}
_hits = 0
_misses = 0


def _today_str() -> str:
    """Ngày hôm nay theo giờ địa phương, dạng YYYY-MM-DD."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d")


def _ttl_for(kwargs: Dict[str, Any], name: str = "") -> float:
    """
    TTL dựa trên việc tham số có chạm tới hôm nay hay không.

    Không có tham số ngày cũng coi là kỳ mở: mọi tool ở đây mặc định là "hôm nay"
    khi bỏ trống ngày, nên bỏ trống chính là trường hợp mở nhất.

    Tool đo trạng thái tức thời không có khái niệm "kỳ" nên xét theo tên trước.
    """
    if name in _LIVE_TOOLS:
        return TTL_LIVE

    today = _today_str()
    blob = json.dumps(kwargs, ensure_ascii=False, default=str)
    if today in blob:
        return TTL_OPEN
    has_date = any(k in kwargs and kwargs[k] for k in
                   ("start_date", "end_date", "date", "days", "period"))
    return TTL_CLOSED if has_date else TTL_OPEN


def _key(name: str, kwargs: Dict[str, Any]) -> Optional[str]:
    """
    Khoá cache. None nếu tham số không tuần tự hoá được.

    Sắp khoá theo thứ tự để `{a:1, b:2}` và `{b:2, a:1}` ra cùng một khoá — mô
    hình không giữ thứ tự tham số ổn định giữa các lượt.
    """
    try:
        return name + "|" + json.dumps(kwargs, sort_keys=True,
                                       ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None


def stats() -> Dict[str, Any]:
    with _lock:
        total = _hits + _misses
        return {
            "entries": len(_store),
            "hits": _hits,
            "misses": _misses,
            "hit_rate": round(_hits / total * 100, 1) if total else None,
        }


def clear() -> int:
    global _hits, _misses
    with _lock:
        n = len(_store)
        _store.clear()
        _hits = _misses = 0
    return n


def _prune(now: float) -> None:
    """Bỏ mục hết hạn; nếu vẫn quá đông thì bỏ mục hết hạn sớm nhất."""
    dead = [k for k, (exp, _) in _store.items() if exp <= now]
    for k in dead:
        _store.pop(k, None)
    if len(_store) > _MAX_ENTRIES:
        for k, _ in sorted(_store.items(), key=lambda kv: kv[1][0])[:len(_store) - _MAX_ENTRIES]:
            _store.pop(k, None)


def should_cache(name: str, category: str, requires_approval: bool) -> bool:
    if name in _NEVER_CACHE or requires_approval:
        return False
    return category in _CACHEABLE_CATEGORIES


def wrap(func: Callable, name: str) -> Callable:
    """
    Bọc hàm tool để cache theo tham số.

    Chỉ cache khi kết quả là dict có `success` là True. Kết quả lỗi không được
    cache: lỗi thường là tạm thời (Mongo nghẽn, file log đang ghi), giữ lại 45s
    nghĩa là người dùng thử lại ngay vẫn nhận đúng cái lỗi cũ và tưởng là hỏng thật.
    """
    global _hits, _misses

    def cached(*args, **kwargs):
        # Tool ở đây đều được LangChain gọi bằng keyword. Có positional thì bỏ
        # cache cho lượt đó thay vì đoán tên tham số.
        if args:
            return func(*args, **kwargs)

        key = _key(name, kwargs)
        if key is None:
            return func(**kwargs)

        now = time.monotonic()
        with _lock:
            entry = _store.get(key)
            if entry and entry[0] > now:
                globals()["_hits"] = _hits + 1
                logger.info("cache HIT %s", name)
                return entry[1]

        result = func(**kwargs)

        if isinstance(result, dict) and result.get("success") is True:
            ttl = _ttl_for(kwargs, name)
            with _lock:
                globals()["_misses"] = _misses + 1
                _prune(now)
                _store[key] = (now + ttl, result)
            logger.debug("cache STORE %s ttl=%.0fs", name, ttl)
        else:
            with _lock:
                globals()["_misses"] = _misses + 1

        return result

    cached.__name__ = getattr(func, "__name__", name)
    cached.__doc__ = getattr(func, "__doc__", None)
    return cached
