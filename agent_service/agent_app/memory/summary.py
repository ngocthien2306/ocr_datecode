"""
Tóm tắt các lượt hội thoại bị cắt khỏi cửa sổ ngữ cảnh.

Vấn đề: `_trim_history` chỉ giữ `MAX_HISTORY_MESSAGES` message cuối. Cắt là cần
thiết — không cắt thì prompt phình vô hạn — nhưng cắt cứng làm ngữ cảnh mất ĐỘT
NGỘT chứ không suy giảm dần. Người vận hành nói chuyện 30 lượt về một recipe, tới
lượt 31 agent quên mất đang nói về recipe nào, và không có dấu hiệu gì báo trước.

Cách chữa: khi cửa sổ trượt, phần bị bỏ được tóm tắt thành vài dòng và dán lại vào
đầu ngữ cảnh. Tóm tắt được TÍCH LUỸ và lưu trong `metadata` của conversation, nên
mỗi lượt không phải tóm tắt lại từ đầu — chỉ khi cửa sổ trượt thêm mới có một lượt
gọi LLM, và lượt đó thay cho việc gửi hàng chục message cũ.

Tóm tắt dán vào dưới dạng `HumanMessage`, KHÔNG phải `SystemMessage`. Lý do rất cụ
thể: `orchestrator_agent.call_model` chỉ thêm system prompt của nó khi trong
messages CHƯA có `SystemMessage` nào. Dán tóm tắt dưới dạng SystemMessage là
orchestrator bỏ luôn system prompt của chính nó — mất hết chỉ dẫn định tuyến, mất
cả dòng ngôn ngữ.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from agent_app.core.config import settings
from agent_app.core.llm import make_llm
from agent_app.db.mongodb import get_database
from agent_app.memory.conversation_service import ConversationService

logger = logging.getLogger(__name__)

# Nhãn đặt trước tóm tắt. Phải nhìn ra ngay đây không phải lời user vừa nói, nếu
# không mô hình sẽ trả lời chính bản tóm tắt như thể đó là câu hỏi mới.
MARKER = "[NGỮ CẢNH CÁC LƯỢT TRƯỚC — không phải câu hỏi mới, chỉ để bạn nhớ]"

# Giữ tóm tắt ngắn. Nó chiếm chỗ trong cùng cửa sổ ngữ cảnh mà nó vừa giải phóng,
# nên để nó dài ra là tự thua.
_MAX_CHARS = 900

_PROMPT = """Bạn đang nén lịch sử một cuộc hội thoại giữa người vận hành nhà máy và
trợ lý phân tích dữ liệu sản xuất.

Viết lại thành tóm tắt NGẮN (tối đa 6 gạch đầu dòng, dưới 150 từ) chỉ gồm những thứ
CÒN CẦN để hiểu các câu hỏi tiếp theo:

- Recipe / camera / ca / khoảng thời gian đang được nói tới (ghi CHÍNH XÁC tên và số)
- Con số quan trọng đã nêu, kèm đơn vị
- Kết luận hoặc vấn đề đang theo dõi
- Việc user đã yêu cầu mà chưa xong

BỎ: lời chào, lời cảm ơn, câu dẫn, mô tả lại cách trình bày.

TUYỆT ĐỐI không bịa thêm số nào không có trong lịch sử. Nếu một con số không rõ đơn
vị thì ghi nguyên như đã thấy.

{previous}
--- LỊCH SỬ CẦN NÉN ---
{history}
--- HẾT ---

Tóm tắt:"""


def _render(messages: List[Any]) -> str:
    """Lịch sử thành text gọn. Bỏ ToolMessage: nội dung tool rất dài và đã được
    phản ánh trong câu trả lời của assistant."""
    out = []
    for m in messages:
        role = getattr(m, "role", None)
        content = str(getattr(m, "content", "") or "")
        if role == "tool" or not content.strip():
            continue
        who = {"user": "User", "assistant": "Trợ lý"}.get(role, role or "?")
        out.append(f"{who}: {content[:600]}")
    return "\n".join(out)


async def _read_meta(session_id: str) -> Dict[str, Any]:
    db = get_database()
    doc = await db[ConversationService.COLLECTION_NAME].find_one(
        {"session_id": session_id}, {"metadata": 1})
    return (doc or {}).get("metadata") or {}


async def _write_meta(session_id: str, summary: str, upto: int) -> None:
    db = get_database()
    await db[ConversationService.COLLECTION_NAME].update_one(
        {"session_id": session_id},
        {"$set": {"metadata.summary": summary, "metadata.summary_upto": upto}},
    )


async def ensure_summary(session_id: str, all_messages: List[Any],
                         kept: int) -> Optional[str]:
    """
    Tóm tắt phần lịch sử bị cắt, hoặc None nếu chưa có gì bị cắt.

    `kept` là số message cuối vẫn được replay. Phần bị cắt là `all_messages[:-kept]`.

    Chỉ gọi LLM khi phần bị cắt DÀI RA so với lần trước — nghĩa là mỗi lượt bình
    thường không tốn gì, và một phiên dài chỉ tốn thêm một lượt mỗi khi cửa sổ trượt.
    """
    dropped_count = max(0, len(all_messages) - kept)
    if dropped_count == 0:
        return None

    meta = await _read_meta(session_id)
    previous = meta.get("summary") or ""
    covered = int(meta.get("summary_upto") or 0)

    if covered >= dropped_count and previous:
        return previous

    # Chỉ nén phần CHƯA được tóm tắt, rồi gộp với tóm tắt cũ. Nén lại từ đầu mỗi lần
    # thì phiên càng dài càng đắt, và bản tóm tắt bị viết lại liên tục nên chi tiết
    # cũ dần bị mài mất.
    fresh = all_messages[covered:dropped_count]
    body = _render(fresh)
    if not body.strip():
        return previous or None

    prev_block = (f"--- TÓM TẮT ĐÃ CÓ (giữ lại thông tin còn giá trị) ---\n{previous}\n"
                  if previous else "")
    try:
        llm = make_llm(settings.DEFAULT_MODEL, 0.0)
        resp = llm.invoke(_PROMPT.format(previous=prev_block, history=body))
        text = str(getattr(resp, "content", "") or "").strip()[:_MAX_CHARS]
    except Exception as e:
        # Tóm tắt thất bại thì dùng lại bản cũ và đi tiếp. Ngữ cảnh kém hơn vẫn tốt
        # hơn một câu hỏi bị 500 vì lỗi ở bước phụ trợ.
        logger.warning("Không tóm tắt được hội thoại %s: %s", session_id, e)
        return previous or None

    if not text:
        return previous or None

    try:
        await _write_meta(session_id, text, dropped_count)
    except Exception as e:
        logger.warning("Không lưu được tóm tắt %s: %s", session_id, e)

    logger.info("Đã tóm tắt %d message bị cắt của session %s",
                dropped_count - covered, session_id)
    return text


def as_message(summary: str) -> HumanMessage:
    """Bọc tóm tắt thành message dán vào đầu ngữ cảnh."""
    return HumanMessage(content=f"{MARKER}\n{summary}")
