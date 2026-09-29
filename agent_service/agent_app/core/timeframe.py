"""
Mốc thời gian trong câu hỏi — nhận ra, quy về ngày cụ thể, và mang sang lượt sau.

## Vì sao cần file này

`_local_bound(None)` trong `tools/analytics_tools.py` mặc định là HÔM NAY, và mặc
định đó im lặng: agent con nào quên truyền `start_date`/`end_date` thì tool vẫn trả
về một kết quả trông hoàn toàn bình thường, chỉ là của sai ngày. Đã đo được trên
PC-Auto-1 ngày 2026-08-29, cùng một `session_id`:

    Lượt 1  "hôm qua ai đã load recipe?"  → đúng, 22 lần
    Lượt 2  "hôm qua thì sao?"            → gọi ĐÚNG tool, trả lời "Hôm nay không
                                            có ai load recipe" (0 bản ghi)

Chữ "hôm qua" nằm ngay trong câu mà vẫn rơi mất. Nên chỗ hỏng KHÔNG phải là history
— `api/chat.py` replay đầy đủ 40 message, `agent_tools.bind()` đẩy 6 message cuối
xuống agent con — mà là bước NGÔN NGỮ → THAM SỐ NGÀY, bước duy nhất trong chuỗi
được giao trọn cho mô hình tự làm.

## Cách sửa

Quy đổi ở đây, bằng bảng cố định, rồi CHÈN ngày cụ thể vào câu hỏi trước khi giao
cho agent con. Mô hình không còn phải tự suy "hôm qua là ngày nào" — nó đọc thẳng
`2026-08-28` trong câu. Đây cùng một lập luận với `core/intent.py`: thứ gì một phép
so chuỗi làm được chắc chắn thì đừng giao cho mô hình làm gần đúng.

Hai việc, dùng chung một bảng:

1. Câu CÓ mốc  → giải ra ngày cụ thể, chèn vào ("hôm qua" → 2026-08-28)
2. Câu KHÔNG có mốc → lấy mốc của câu hỏi user gần nhất còn trong history
   ("đã thay đổi gì trên recipe?" ngay sau một câu về "hôm qua" → vẫn là hôm qua)

## Giới hạn cố ý

Bảng chỉ nhận các mốc theo NGÀY. Ca kíp ("ca sáng", "ca C"), giờ lẻ ("từ 16h đến
18h") và tuần/tháng đang cố tình để mô hình tự xử: chúng cần biết lịch ca của nhà
máy và cần chọn mốc bắt đầu tuần, nhiều dữ kiện hơn hẳn một phép trừ ngày. Không
nhận ra thì trả None và mọi thứ chạy y như trước — im lặng, không hỏng.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple
from zoneinfo import ZoneInfo

from agent_app.core.config import settings

_TZ = ZoneInfo(settings.TIMEZONE)


# Cụm → số ngày lùi so với hôm nay. Thứ tự xét là thứ tự trong tuple, nên cụm DÀI
# đặt trước cụm ngắn: "hôm kia" phải được thử trước "hôm qua" chứ không thì
# "hôm kia" không bao giờ tới lượt (cả hai đều chứa "hôm").
_RELATIVE_DAYS: Tuple[Tuple[Tuple[str, ...], int], ...] = (
    (("hôm kia", "hôm kìa", "day before yesterday"), 2),
    (("hôm qua", "bữa qua", "tối qua", "sáng qua", "chiều qua", "yesterday"), 1),
    (("hôm nay", "bữa nay", "sáng nay", "chiều nay", "tối nay",
      "nãy giờ", "today"), 0),
)

# "N ngày qua" / "N ngày gần đây" / "last N days" → khoảng [hôm nay - (N-1), hôm nay]
_LAST_N_DAYS = re.compile(
    r"(?:(\d{1,3})\s*ngày\s*(?:qua|gần đây|vừa rồi|trở lại đây)"
    r"|last\s+(\d{1,3})\s+days?)",
    re.IGNORECASE,
)

# Ngày tuyệt đối: 2026-08-28 | 28/08/2026 | 28/8 (năm hiện tại)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b")

# Mốc mà bảng này KHÔNG giải được. Gặp thì trả None để mô hình tự xử — chèn một
# khoảng ngày thô vào câu hỏi về ca kíp là làm hỏng chính câu đó.
_OUT_OF_SCOPE = (
    "ca sáng", "ca chiều", "ca đêm", "ca a", "ca b", "ca c", "ca này", "ca trước",
    "tuần này", "tuần trước", "tuần rồi", "tháng này", "tháng trước", "tháng rồi",
    "this week", "last week", "this month", "last month", "shift",
)


def _today() -> datetime:
    return datetime.now(_TZ).replace(tzinfo=None, hour=0, minute=0,
                                     second=0, microsecond=0)


def detect(text: Optional[str]) -> Optional[Tuple[str, str, str]]:
    """
    Mốc thời gian trong câu, dạng `(start_date, end_date, nguyên văn cụm)`.

    Ngày dạng 'YYYY-MM-DD'. Trả None khi câu không có mốc nào, hoặc có mốc nhưng
    thuộc loại bảng này cố ý không nhận (xem `_OUT_OF_SCOPE`) — hai trường hợp đó
    xử lý giống nhau ở tầng gọi: không chèn gì cả.
    """
    if not text:
        return None
    low = text.lower()

    if any(p in low for p in _OUT_OF_SCOPE):
        return None

    m = _ISO_DATE.search(low)
    if m:
        d = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return (d, d, m.group(0))

    m = _DMY_DATE.search(low)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else _today().year
        try:
            d = datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            # 32/13 và các kiểu tương tự: đây là con số khác chứ không phải ngày.
            return None
        return (d, d, m.group(0))

    m = _LAST_N_DAYS.search(low)
    if m:
        n = int(m.group(1) or m.group(2))
        if 1 <= n <= 365:
            end = _today()
            start = end - timedelta(days=n - 1)
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), m.group(0))

    for phrases, back in _RELATIVE_DAYS:
        for p in phrases:
            if p in low:
                d = (_today() - timedelta(days=back)).strftime("%Y-%m-%d")
                return (d, d, p)

    return None


def mentions_timeframe(text: Optional[str]) -> bool:
    """Câu có nói tới thời gian không — kể cả loại `detect()` không giải được.

    Phân biệt hai thứ mà `detect()` trả None giống hệt nhau:
      - "camera nào fail nhiều nhất"  → không nhắc gì tới thời gian
      - "sản lượng tuần trước"        → CÓ nhắc, nhưng bảng cố ý không nhận

    Trường hợp sau tuyệt đối không được kế thừa mốc của lượt trước: user vừa nói
    "tuần trước" mà mình chèn "hôm qua" vào là ghi đè lên chính lời họ, và ghi
    đè một cách vô hình — câu trả lời sẽ về đúng một ngày, tự tin, và sai.
    """
    if not text:
        return False
    low = text.lower()
    return any(p in low for p in _OUT_OF_SCOPE)


def _texts_of_user_messages(history: Optional[List[Any]]) -> List[str]:
    """Nội dung các message của USER trong history, mới nhất trước.

    Chỉ lấy message của user: câu TRẢ LỜI của agent thường nhắc lại ngày ("Hôm
    qua có 22 lần...") và nếu tính cả chúng thì một câu trả lời lạc đề cũng đủ
    ghim sai mốc cho mọi lượt sau.
    """
    out: List[str] = []
    for m in reversed(history or []):
        if m.__class__.__name__ != "HumanMessage":
            continue
        c = getattr(m, "content", None)
        if isinstance(c, str) and c.strip():
            out.append(c)
    return out


def resolve(question: str, history: Optional[List[Any]] = None) -> str:
    """
    Câu hỏi kèm khoảng ngày đã giải sẵn, để agent con khỏi phải tự suy.

    Không đổi ý câu hỏi, chỉ nối thêm một dòng ở cuối. Không có mốc nào tìm được
    (kể cả trong history) thì trả nguyên văn — khi đó tool giữ mặc định hôm nay
    đúng như trước.
    """
    if not question:
        return question

    found = detect(question)
    inherited = False

    if not found and mentions_timeframe(question):
        # User CÓ nêu mốc, chỉ là loại bảng này không giải. Giao nguyên văn cho
        # mô hình — nó biết lịch ca và biết tuần bắt đầu từ đâu.
        return question

    if not found:
        for prev in _texts_of_user_messages(history):
            found = detect(prev)
            if found:
                inherited = True
                break

    if not found:
        return question

    start, end, phrase = found
    span = start if start == end else f"{start} → {end}"
    if inherited:
        hint = (f'[khoảng thời gian kế thừa từ câu hỏi trước ("{phrase}"): {span}. '
                f'Truyền start_date/end_date đúng khoảng này. User chưa đổi mốc, '
                f'nên KHÔNG được mặc định về hôm nay.]')
    else:
        hint = (f'[khoảng thời gian đã xác định từ "{phrase}": {span}. '
                f'Truyền start_date/end_date đúng khoảng này, đừng để tool tự '
                f'mặc định hôm nay.]')
    return f"{question}\n\n{hint}"
