"""
`fleet_orchestrator` — MỘT agent duy nhất ở trung tâm.

Vì sao chỉ một, khi hệ thống này được gọi là multi-agent: tính multi-agent nằm ở
CHIỀU SÂU ỦY QUYỀN, không nằm ở chiều rộng của trung tâm. Cả hệ có 26 agent —
một ở đây và 25 ở edge (5 máy × 5 agent) — và trung tâm với tới chúng qua hai
tool `ask_machine` / `ask_all_machines`.

Thêm một tầng agent phụ ở trung tâm là thêm MỘT LƯỢT LLM cho mỗi câu hỏi, mà
đường ủy quyền đã tốn sẵn tới 5 lượt. Với 7 tool thì một prompt chứa thoải mái.
Ngưỡng nên tách: quá ~12 tool, hoặc prompt vượt ~2.000 token, hoặc xuất hiện
nhóm tool cần chính sách riêng (ứng viên đầu tiên: điều khiển máy từ xa, vì nó
có tác dụng phụ nên phải qua cổng xác nhận).
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from fleet_app.core.config import settings
from fleet_app.core import suggestions
from fleet_app.tools import fleet_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý vận hành cho một ĐỘI HÌNH máy kiểm tra date code (OCR).
Mỗi máy là một dây chuyền riêng, có agent riêng, database riêng.

## Chọn đường nào

Tool RẺ (không tốn gì, 0,2–2s) — dùng mặc định:
  list_machines · fleet_health · machine_detail · fleet_production · compare_failure_modes

Tool ĐẮT (một lượt LLM trên MỖI máy được hỏi, 4–20s):
  ask_machine · ask_all_machines

Lấy số thì luôn dùng tool rẻ. Chỉ ủy quyền khi câu hỏi cần LẬP LUẬN về một máy
cụ thể — "vì sao", "nên làm gì", phân tích nguyên nhân. Đừng bao giờ gọi
ask_all_machines cho câu hỏi mà fleet_production trả lời được.

Câu hỏi kiểu "máy nào tệ nhất, vì sao?" đi hai chặng:
  1. compare_failure_modes  (rẻ) → tìm máy lệch khỏi mặt bằng
  2. ask_machine vào đúng máy đó (đắt) → nguyên nhân cụ thể

## Bốn quy tắc không được vi phạm

1. **KHÔNG xếp hạng máy bằng tỉ lệ pass.** Các máy chạy recipe khác nhau — trên
   đội hình này đúng MỘT recipe được chia sẻ giữa hai máy. "Auto2 pass 98% còn M2
   pass 69%" là so hành tây với quế. Muốn so máy với máy thì dùng vân tay kiểu lỗi.

2. **Máy thiếu dữ liệu phải được NÊU TÊN.** Mọi kết quả tổng hợp đều có
   `coverage`. Nếu `complete` là false, câu trả lời PHẢI nói rõ máy nào thiếu và
   vì sao. Con số tổng thiếu một máy trông vẫn hoàn toàn bình thường — không ai
   phát hiện được nếu bạn im lặng.

3. **Mẫu nhỏ thì không kết luận.** Vân tay lỗi tính trên mẫu; `sample_covers_all`
   false nghĩa là tỉ lệ của MẪU chứ không phải của cả kỳ. Vài chục sản phẩm thì
   nói là "chưa đủ để kết luận", đừng xếp hạng.

4. **`agent_down` KHÁC `unreachable`.** agent_down nghĩa là máy VẪN ĐANG SẢN
   XUẤT, chỉ trợ lý trên máy đó tắt (thường sau reboot). Đừng báo thành máy chết.

## Cách trả lời

Ngắn gọn, số liệu đặt trong bảng markdown khi có từ 3 máy trở lên.
Nêu đơn vị rõ ràng: sản phẩm hay frame, phần trăm của mẫu hay của cả kỳ.
Không bịa số — chỉ dùng số mà tool trả về."""


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        llm = ChatOpenAI(model=settings.FLEET_MODEL, temperature=0.2,
                         api_key=settings.OPENAI_API_KEY)
        _agent = create_react_agent(llm, fleet_tools.FLEET_TOOLS,
                                    prompt=SYSTEM_PROMPT)
        logger.info("fleet_orchestrator sẵn sàng — %d tool", len(fleet_tools.FLEET_TOOLS))
    return _agent


# Ngôn ngữ trả lời do GIAO DIỆN quyết định, không để mô hình tự đoán theo câu
# hỏi. Đo được: giao diện đang ở VI, câu hỏi tiếng Việt, mô hình vẫn trả lời tiếng
# Anh. Mà tên recipe, tên máy, chức danh trong dữ liệu vốn đã lẫn hai thứ tiếng —
# càng không có gì để đoán.
_LANG_RULE = {
    "vi": "Trả lời bằng TIẾNG VIỆT, kể cả khi câu hỏi viết bằng tiếng Anh.",
    "en": "Answer in ENGLISH, even if the question is written in Vietnamese.",
}


async def run(message: str, history: Optional[List[Dict[str, str]]] = None,
              lang: str = "vi") -> Dict[str, Any]:
    """
    Chạy một lượt hỏi đáp.

    Attachment của edge KHÔNG đi qua mô hình — chúng được `fleet_tools` hút vào
    một hộp riêng rồi ghép thẳng vào phản hồi, kèm nhãn máy.
    """
    box, results = fleet_tools.start_collecting(lang)

    msgs: List[Any] = []
    for h in (history or [])[-8:]:
        role, content = h.get("role"), h.get("content") or ""
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=message))
    msgs.append(SystemMessage(content=_LANG_RULE.get(lang, _LANG_RULE["vi"])))

    result = await get_agent().ainvoke({"messages": msgs})
    out = result["messages"][-1]

    tools_used = []
    for m in result["messages"]:
        for tc in (getattr(m, "tool_calls", None) or []):
            tools_used.append(tc.get("name"))

    rep = results.get("generate_fleet_report") or {}
    return {
        "response": out.content if isinstance(out.content, str) else str(out.content),
        "tool_calls": tools_used,
        "attachments": box,
        # Gợi ý dựng bằng CODE từ kết quả tool, không phải do mô hình viết.
        "suggestions": suggestions.build(tools_used, results, lang=lang),
        # Đường tải gắn ở đây chứ không đưa cho mô hình: nó từng bịa ra URL kiểu
        # example.com khi nhìn thấy trường đường dẫn.
        "file": ({"name": rep["file"], "url": f"/api/fleet/report/{rep['file']}"}
                 if rep.get("file") else None),
    }
