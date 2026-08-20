"""
Bảy tool của `fleet_orchestrator`, chia làm hai nhóm rất rõ.

**Máy là THAM SỐ, không phải agent.** Nếu mỗi máy là một agent thì thêm máy mới
phải đăng ký agent, sửa prompt, sửa danh sách tool — và mô hình phải thuộc lòng
tên máy mới định tuyến được. Máy là tham số thì thêm máy chỉ là thêm một dòng
trong registry, và "hỏi tất cả các máy" diễn đạt được bằng một lời gọi thay vì
liệt kê năm lời gọi. Đây là thứ giữ cho hệ thống còn cắm-là-chạy ở tầng agent,
chứ không chỉ ở tầng hạ tầng.

| nhóm | tool | LLM ở edge | dùng khi |
|---|---|---|---|
| xác định | list_machines, fleet_health, machine_detail, fleet_production, compare_failure_modes | 0 | số liệu, bảng, xếp hạng |
| ủy quyền | ask_machine, ask_all_machines | 1 hoặc N | "vì sao", câu hỏi mở |

Ranh giới giữa hai nhóm là **tần suất**, không phải kiểu dữ liệu: thứ hỏi đi hỏi
lại đi đường xác định, thứ hỏi một lần đi đường ủy quyền. Đo được trên đội hình
này: đường xác định 0,2–2s và miễn phí; đường ủy quyền 4–20s và tốn tiền cho mỗi
máy được hỏi.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from fleet_app.core import queries
from fleet_app.core.config import settings
from fleet_app.core.edge_client import client
from fleet_app.core.fanout import coverage, fan_out
from fleet_app.core.registry import Machine, registry

logger = logging.getLogger(__name__)

# Attachment (KPI, biểu đồ, bảng, ảnh) mà agent edge trả về KHÔNG đi qua giá trị
# trả về của tool — chúng được hút vào đây rồi ghép thẳng vào phản hồi cuối. Cùng
# lý do `strip_for_llm()` ở tầng edge: số đến người dùng phải qua code, không qua
# miệng mô hình. Khác biệt duy nhất ở tầng fleet là phải GẮN NHÃN MÁY, nếu không
# thì 20 ô KPI của 5 máy trộn vào nhau và không ô nào biết của ai.
_collected: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar("_collected", default=None)

# Văn xuôi của edge bị cắt trước khi vào mô hình. Hỏi 5 máy mà mỗi máy trả về một
# bài dài thì lượt tổng hợp phình context, và phần thừa không thêm thông tin —
# con số thật đã đi đường attachment rồi.
_MAX_EDGE_PROSE = 800


def start_collecting() -> List[Dict[str, Any]]:
    box: List[Dict[str, Any]] = []
    _collected.set(box)
    return box


def _collect(machine: str, payload: Dict[str, Any]) -> None:
    box = _collected.get()
    if box is None:
        return
    for kind in ("kpis", "charts", "tables", "cards", "images", "files"):
        for item in (payload.get(kind) or []):
            box.append({"machine": machine, "kind": kind, "item": item})


def _trim(text: str) -> str:
    if len(text) <= _MAX_EDGE_PROSE:
        return text
    return text[:_MAX_EDGE_PROSE] + " […cắt bớt]"


# ---------------------------------------------------------------- xác định ---

async def list_machines() -> Dict[str, Any]:
    """Máy nào đang có trong đội hình, tên, dây chuyền, model, trạng thái."""
    return {
        "machines": [
            {"name": m.name, "line": m.line, "model": m.model, "ip": m.ip,
             "state": m.state(), "capability_level": m.level,
             "agents": m.agents, "last_seen_ago_sec":
                 None if not m.last_seen else int(__import__("time").time() - m.last_seen)}
            for m in registry.all()
        ],
        "note": ("state=agent_down nghĩa là MÁY VẪN CHẠY SẢN XUẤT, chỉ agent tắt "
                 "(agent chạy uvicorn trần nên không sống qua reboot). Khác hẳn "
                 "unreachable là mất liên lạc với cả máy."),
    }


async def fleet_health() -> Dict[str, Any]:
    """Sức khoẻ phần cứng và service của cả đội hình: nhiệt độ, RAM, đĩa, camera service."""
    d = await queries.fleet_status()
    return {
        "coverage": d["coverage"],
        "machines": [
            {"machine": m["name"], "state": m["state"], "metrics": m.get("metrics"),
             "camera_service_running": (m.get("service") or {}).get("is_running"),
             "errors": m.get("errors")}
            for m in d["machines"]
        ],
    }


class MachineArgs(BaseModel):
    host: str = Field(description="Tên máy, ví dụ 'M2', 'Auto2', 'LineTine', 'PC-Auto-1'.")


async def machine_detail(host: str, **_ignored) -> Dict[str, Any]:
    """Thông tin đầy đủ của MỘT máy: phần cứng, service, năng lực, trạng thái."""
    return await queries.machine_detail(host)


class DaysArgs(BaseModel):
    days: int = Field(default=7, ge=1, le=90, description="Số ngày tính ngược từ hôm nay.")


async def fleet_production(days: int = 7, **_ignored) -> Dict[str, Any]:
    """Sản lượng, pass/fail, sản lượng mỗi ngày và recipe đang chạy của từng máy."""
    d = await queries.fleet_production(days=days, causes=False)
    return {
        "period_days": d["period_days"], "coverage": d["coverage"],
        "fleet_total": d["fleet_total"], "note": d["note"],
        "machines": [{"machine": r["machine"], "line": r["line"],
                      "production": r["production"], "recipes": r["recipes"],
                      "error": r["error"]}
                     for r in d["machines"]],
    }


async def compare_failure_modes(days: int = 7, **_ignored) -> Dict[str, Any]:
    """
    So sánh PHÂN BỐ NGUYÊN NHÂN LỖI giữa các máy — cách so sánh có nghĩa duy nhất.

    Dùng tool này khi được hỏi máy nào tệ nhất, máy nào bất thường, hay so sánh
    các máy với nhau. KHÔNG dùng tỉ lệ pass để xếp hạng: các máy chạy recipe khác
    nhau nên tỉ lệ pass phản ánh độ khó mặt hàng chứ không phản ánh máy.
    """
    d = await queries.fleet_production(days=days, causes=True)
    return {
        "period_days": d["period_days"], "coverage": d["coverage"],
        "fingerprint": d["failure_fingerprint"],
        "how_to_read": (
            "Các khoá như char_verification / no_detection là BƯỚC KIỂM TRA trong "
            "pipeline OCR, KHÔNG phải tên recipe và KHÔNG lọc được. Khi ủy quyền "
            "xuống agent của máy, hãy mô tả bằng lời ('ký tự dưới ngưỡng tin cậy') "
            "chứ đừng truyền mã đó như tên recipe. "
            "Mỗi ô là TỈ TRỌNG giữa các nguyên nhân trên mẫu fail của máy đó "
            "(các ô của một máy cộng lại bằng 100). KHÔNG phải 'bao nhiêu % sản "
            "phẩm trượt bước này' — một sản phẩm trượt được nhiều bước. Nếu agent "
            "của máy trả về con số khác cho cùng nguyên nhân thì đó là mẫu số "
            "khác, đừng đặt hai con số cạnh nhau như thể cùng một thứ. "
            "Máy nào lệch hẳn khỏi mặt bằng ở một nguyên nhân thì đó là manh mối: "
            "no_detection cao = camera/trigger/ánh sáng không bắt được vùng; "
            "char_verification cao = bắt được vùng nhưng ký tự dưới ngưỡng tin cậy. "
            "Hai máy cùng 'pass rate thấp' có thể hỏng hai thứ hoàn toàn khác nhau."
        ),
    }


# ---------------------------------------------------------------- ủy quyền ---

class AskMachineArgs(BaseModel):
    host: str = Field(description="Tên máy cần hỏi, ví dụ 'M2'.")
    question: str = Field(description=(
        "Câu hỏi gửi cho agent của máy đó, viết đầy đủ và TỰ ĐỨNG ĐƯỢC. Phải nêu "
        "rõ khoảng thời gian, tên recipe, số camera nếu có — agent con không thấy "
        "câu hỏi gốc nên đừng dùng 'đó', 'cái này', 'như trên'. "
        "Viết bằng lời người: đừng truyền mã kỹ thuật như 'char_verification' vào "
        "chỗ tên recipe — đó là bước kiểm tra, không phải sản phẩm."))


async def ask_machine(host: str, question: str, **_ignored) -> Dict[str, Any]:
    """
    Hỏi agent của MỘT máy. Đắt: một lượt LLM trên máy đó, 4–20s.

    Dùng cho câu hỏi mở — 'vì sao', 'nên làm gì', phân tích nguyên nhân. Số liệu
    thuần thì dùng các tool xác định, rẻ hơn và nhanh hơn ~10 lần.
    """
    r = queries.resolve(host)
    if not r["ok"]:
        return r
    m: Machine = r["machine"]
    res = await client.chat(m.node_id, m.ip, question)
    if not res.ok:
        return {"ok": False, "machine": m.name, "error": res.error}
    data = res.data if isinstance(res.data, dict) else {}
    _collect(m.name, data)
    return {"ok": True, "machine": m.name, "answer": _trim(data.get("response") or "")}


class AskAllArgs(BaseModel):
    question: str = Field(description=(
        "Câu hỏi gửi cho agent của MỌI máy. Viết tự đứng được, nêu rõ khoảng thời "
        "gian. Mỗi máy trả lời độc lập về chính nó."))


async def ask_all_machines(question: str, **_ignored) -> Dict[str, Any]:
    """
    Hỏi agent của TẤT CẢ các máy cùng lúc. RẤT đắt: một lượt LLM trên mỗi máy.

    Chỉ dùng khi câu hỏi thật sự cần lập luận riêng ở từng máy. Muốn số liệu của
    cả đội hình thì dùng fleet_production hoặc compare_failure_modes — rẻ hơn,
    nhanh hơn, và cho số chính xác hơn.
    """
    ms = [m for m in registry.all() if m.online]

    async def one(m: Machine) -> Dict[str, Any]:
        res = await client.chat(m.node_id, m.ip, question)
        if not res.ok:
            raise RuntimeError(res.error or "chat lỗi")
        data = res.data if isinstance(res.data, dict) else {}
        _collect(m.name, data)
        return {"answer": _trim(data.get("response") or "")}

    results = await fan_out(ms, one, timeout=settings.EDGE_CHAT_TIMEOUT + 5)
    return {
        # Máy lỗi NẰM TRONG kết quả, không bị lược đi. Bỏ nó ra thì câu trả lời
        # tổng hợp thiếu một máy mà đọc vẫn hoàn toàn bình thường.
        "coverage": coverage(results),
        "answers": [
            {"machine": r["machine"],
             "answer": (r.get("data") or {}).get("answer") if r.get("ok") else None,
             "error": r.get("error")}
            for r in results
        ],
        "note": "Nếu coverage.complete là false, PHẢI nói rõ máy nào không trả lời.",
    }


class NoArgs(BaseModel):
    """Tool không nhận tham số."""


def _tool(fn, name: str, desc: str, schema=None) -> StructuredTool:
    """
    Bọc một coroutine thành tool.

    args_schema LUÔN được truyền tường minh, kể cả khi tool không có tham số.
    Để LangChain tự suy ra schema từ chữ ký hàm thì tool không tham số nhận được
    một schema rỗng sai — bản đầu bọc thêm `func=lambda **kw` cho nhánh sync, và
    LangChain suy ra một field tên `kw`, nên mọi lời gọi đều chết với
    "fleet_health() got an unexpected keyword argument 'kw'".

    Chỉ truyền `coroutine`: fleet chạy hoàn toàn async, không có đường nào gọi
    các tool này từ ngữ cảnh đồng bộ.
    """
    return StructuredTool.from_function(
        coroutine=fn, name=name, description=desc, args_schema=schema or NoArgs)


FLEET_TOOLS: List[StructuredTool] = [
    _tool(list_machines, "list_machines",
          "Liệt kê các máy trong đội hình: tên, dây chuyền, model, trạng thái, bậc năng lực. Rẻ."),
    _tool(fleet_health, "fleet_health",
          "Sức khoẻ phần cứng cả đội hình: nhiệt độ CPU/GPU, RAM, đĩa, camera service. Rẻ."),
    _tool(machine_detail, "machine_detail",
          "Thông tin đầy đủ của MỘT máy. Rẻ.", MachineArgs),
    _tool(fleet_production, "fleet_production",
          "Sản lượng, pass/fail, recipe của từng máy và tổng cả nhà máy. Rẻ.", DaysArgs),
    _tool(compare_failure_modes, "compare_failure_modes",
          "So sánh phân bố nguyên nhân lỗi giữa các máy. Dùng tool NÀY khi được hỏi "
          "máy nào tệ nhất hoặc bất thường — KHÔNG xếp hạng bằng tỉ lệ pass. Rẻ.", DaysArgs),
    _tool(ask_machine, "ask_machine",
          "Hỏi agent của MỘT máy một câu mở ('vì sao…'). ĐẮT: 1 lượt LLM trên máy đó, "
          "4–20s. Chỉ dùng khi cần lập luận, không dùng để lấy số.", AskMachineArgs),
    _tool(ask_all_machines, "ask_all_machines",
          "Hỏi agent của TẤT CẢ các máy. RẤT ĐẮT: 1 lượt LLM mỗi máy. Cân nhắc "
          "fleet_production / compare_failure_modes trước.", AskAllArgs),
]
