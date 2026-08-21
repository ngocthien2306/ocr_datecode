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
from fleet_app.reports import builder as reports_builder

logger = logging.getLogger(__name__)

# Attachment (KPI, biểu đồ, bảng, ảnh) mà agent edge trả về KHÔNG đi qua giá trị
# trả về của tool — chúng được hút vào đây rồi ghép thẳng vào phản hồi cuối. Cùng
# lý do `strip_for_llm()` ở tầng edge: số đến người dùng phải qua code, không qua
# miệng mô hình. Khác biệt duy nhất ở tầng fleet là phải GẮN NHÃN MÁY, nếu không
# thì 20 ô KPI của 5 máy trộn vào nhau và không ô nào biết của ai.
_collected: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar("_collected", default=None)

# Kết quả thô của từng tool, để `core/suggestions.py` suy gợi ý TỪ SỐ LIỆU thay vì
# để mô hình tự viết. Mô hình viết gợi ý mà không nhìn con số nên mời những thứ
# chẳng dính gì tới thứ vừa hiện trên màn hình.
_results: ContextVar[Optional[Dict[str, Any]]] = ContextVar("_results", default=None)

# Ngôn ngữ của lượt hỏi. File xuất ra (báo cáo) phải theo đúng ngôn ngữ người
# dùng đang hỏi — hỏi tiếng Anh mà nhận về một PDF tiếng Việt là một sản phẩm
# không dùng được, không phải một chi tiết nhỏ.
_lang: ContextVar[str] = ContextVar("_lang", default="vi")

# Văn xuôi của edge bị cắt trước khi vào mô hình. Hỏi 5 máy mà mỗi máy trả về một
# bài dài thì lượt tổng hợp phình context, và phần thừa không thêm thông tin —
# con số thật đã đi đường attachment rồi.
_MAX_EDGE_PROSE = 800


def start_collecting(lang: str = "vi") -> tuple:
    box: List[Dict[str, Any]] = []
    res: Dict[str, Any] = {}
    _collected.set(box)
    _results.set(res)
    _lang.set(lang or "vi")
    return box, res


def _remember(name: str, value: Any) -> None:
    res = _results.get()
    if res is not None:
        res[name] = value


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
    out = {
        "coverage": d["coverage"],
        "machines": [
            {"machine": m["name"], "state": m["state"], "metrics": m.get("metrics"),
             "camera_service_running": (m.get("service") or {}).get("is_running"),
             "errors": m.get("errors")}
            for m in d["machines"]
        ],
    }
    _remember("fleet_health", out)
    return out


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
    out = {
        "period_days": d["period_days"], "coverage": d["coverage"],
        "fleet_total": d["fleet_total"], "note": d["note"],
        "machines": [{"machine": r["machine"], "line": r["line"],
                      "production": r["production"], "recipes": r["recipes"],
                      "error": r["error"]}
                     for r in d["machines"]],
    }
    _remember("fleet_production", out)
    return out


async def compare_failure_modes(days: int = 7, **_ignored) -> Dict[str, Any]:
    """
    So sánh PHÂN BỐ NGUYÊN NHÂN LỖI giữa các máy — cách so sánh có nghĩa duy nhất.

    Dùng tool này khi được hỏi máy nào tệ nhất, máy nào bất thường, hay so sánh
    các máy với nhau. KHÔNG dùng tỉ lệ pass để xếp hạng: các máy chạy recipe khác
    nhau nên tỉ lệ pass phản ánh độ khó mặt hàng chứ không phản ánh máy.
    """
    d = await queries.fleet_production(days=days, causes=True)
    out = {
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
    _remember("compare_failure_modes", out)
    return out


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


# ------------------------------------------------------------------ báo cáo ---

class ReportArgs(BaseModel):
    # Mỗi mô tả phải nói CẢ HAI nhánh. Bản trước chỉ nói nhánh "để trống", nên
    # mô hình gọi rỗng kể cả khi user đã nói đủ ba thứ trong đúng câu đó — đo
    # được thật: "Xuất báo cáo pdf cho tất cả máy, kỳ 7 ngày qua" vẫn ra ba
    # tham số None và tool hỏi lại. Ba lượt hỏi, không ra file.
    machines: Optional[List[str]] = Field(
        default=None,
        description="Danh sách tên máy đưa vào báo cáo, ví dụ ['M1','M2']. "
                    "TRUYỀN khi user đã nêu, kể cả ở lượt trước; “tất cả máy” "
                    "→ ['Tất cả']. Để trống CHỈ khi user chưa nêu máy nào.")
    period: Optional[str] = Field(
        default=None,
        description="Kỳ báo cáo, ví dụ '7 ngày qua', 'hôm nay', '30 ngày qua'. "
                    "TRUYỀN khi user đã nêu, kể cả ở lượt trước. Để trống CHỈ "
                    "khi user chưa nêu kỳ.")
    format: Optional[str] = Field(
        default=None,
        description="Định dạng file: html/pdf/excel/csv. TRUYỀN khi user đã nêu, "
                    "kể cả ở lượt trước. Để trống CHỈ khi user chưa nêu.")


async def generate_fleet_report(machines: Optional[List[str]] = None,
                                period: Optional[str] = None,
                                format: Optional[str] = None,
                                **_ignored: Any) -> Dict[str, Any]:
    """
    Xuất báo cáo SO SÁNH nhiều máy ra file.

    Ba tham số đều mặc định None, và tool TỪ CHỐI chạy khi thiếu bất kỳ cái nào —
    nó trả về danh sách lựa chọn để hỏi lại người dùng. Đặt mặc định (ví dụ
    format="html") là dạy mô hình tự điền, và câu hỏi không bao giờ tới tay người
    dùng — đúng bài học đã ghi ở `PIPELINE.md §4` của tầng edge.

    `**_ignored` để một tham số bịa ra không giết cả lượt chat: đã xảy ra thật khi
    mô tả tool nhắc tên khoá trong kết quả và mô hình tưởng đó là tham số.
    """
    all_names = [m.name for m in registry.all()]

    missing: Dict[str, Any] = {}
    if not machines:
        missing["machines"] = {
            "prompt": "Báo cáo gồm những máy nào?",
            "options": all_names + ["Tất cả"],
        }
    if not period:
        missing["period"] = {
            "prompt": "Báo cáo cho kỳ nào?",
            # Chỉ đưa NHÃN, không đưa khoá nội bộ: mô hình sẽ nhắc lại đúng thứ
            # nó nhìn thấy, nên đưa khoá ra là mời nó gửi lại khoá sai chính tả.
            "options": [p["label"] for p in queries.PERIOD_CHOICES],
        }
    if not format:
        missing["format"] = {
            "prompt": "Xuất ra định dạng nào?",
            "options": ["html", "pdf", "excel", "csv"],
        }
    if missing:
        # Nói rõ hai việc, không chỉ một.
        #
        # Bản trước chỉ nói "hãy hỏi lại và ĐỪNG tự chọn thay họ", nên mô hình
        # hỏi lại đúng — rồi lượt sau người dùng đáp "Tất cả", nó lại gọi tool
        # với ba tham số rỗng và hỏi lại lần nữa. Vòng lặp đo được thật: hai
        # lượt, cùng một câu hỏi, không ra file.
        #
        # Thiếu là câu thứ hai: khi người dùng ĐÃ đáp, phải gọi lại tool với
        # ĐẦY ĐỦ ba tham số, gom cả những giá trị họ nêu ở các lượt trước.
        have = {k: v for k, v in (("machines", machines), ("period", period),
                                  ("format", format)) if v}
        return {
            "ok": False,
            "ask_user": missing,
            "đã_có": have,
            "message": (
                "Chưa đủ thông tin để xuất báo cáo. Hỏi người dùng đúng những mục "
                "còn thiếu và liệt kê các lựa chọn; ĐỪNG tự chọn thay họ. "
                "NHƯNG khi họ đã trả lời — kể cả trả lời ngắn như “Tất cả” hay "
                "“pdf” — hãy gọi lại tool NGAY với ĐẦY ĐỦ cả ba tham số "
                "(machines, period, format), gom lại những giá trị họ đã nêu ở "
                "các lượt trước (xem `đã_có`). Hỏi lại một mục mà họ vừa trả lời "
                "là lỗi."),
        }

    # --- đã đủ tham số ---
    fmt = (format or "").strip().lower()
    fmt = {"xlsx": "excel", "spreadsheet": "excel"}.get(fmt, fmt)
    if fmt not in reports_builder.FORMATS:
        return {"ok": False, "error": f"Định dạng '{format}' không hỗ trợ",
                "options": list(reports_builder.FORMATS)}

    if isinstance(machines, str):
        machines = [machines]
    if any(str(x).strip().lower() in ("tất cả", "all", "tat ca") for x in machines):
        chosen = all_names
    else:
        chosen, unknown = [], []
        for want in machines:
            r = queries.resolve(str(want))
            if r["ok"]:
                chosen.append(r["machine"].name)
            elif r.get("ambiguous"):
                return {"ok": False, "error": r["error"], "candidates": r["ambiguous"]}
            else:
                unknown.append(want)
        if unknown:
            return {"ok": False, "error": f"Không có máy: {', '.join(map(str, unknown))}",
                    "known_machines": all_names}
    if not chosen:
        return {"ok": False, "error": "Chưa chọn được máy nào"}

    p = queries.resolve_period(period)
    if not p:
        return {"ok": False, "error": f"Kỳ '{period}' không hiểu được",
                "options": [c["label"] for c in queries.PERIOD_CHOICES]}

    lang = _lang.get()
    data = await queries.report_data(chosen, p["days"], p["label"],
                                     p.get("label_en"))
    path = reports_builder.render(data, fmt, lang=lang)
    _remember("generate_fleet_report",
              {"file": path.name, "machines": chosen, "format": fmt})
    return {
        "ok": True,
        "machines": chosen,
        "period": p["label"],
        "format": fmt,
        # TÊN FILE KHÔNG NẰM Ở ĐÂY. Nó đi qua `_remember` để tầng API gắn link
        # thật vào phản hồi, còn mô hình không được nhìn thấy.
        #
        # Bản đầu để `"_file": path.name` với dấu gạch dưới, tưởng vậy là đủ kín.
        # Không đủ: mô hình đọc được tên file rồi tự bịa ra đường dẫn
        # `sandbox:/fleet_….pdf` và đưa cho người dùng bấm. Đúng bài học
        # `PIPELINE.md §3` — thay placeholder thì nó nhúng luôn placeholder, chỉ
        # khi XOÁ HẲN khỏi tầm nhìn của mô hình thì vấn đề mới hết.
        "file_ready": True,
        "how_to_present": ("Nói báo cáo đã sẵn sàng và mời người dùng bấm nút tải "
                           "bên dưới. TUYỆT ĐỐI không tự viết ra đường dẫn hay link "
                           "tải — bạn không có nó."),
        "summary": {
            "total_products": data["fleet_total"]["products"],
            "pass_rate": data["fleet_total"]["pass_rate"],
            "machines_missing": data["coverage"]["machines_missing"],
        },
    }


FLEET_TOOLS.append(_tool(
    generate_fleet_report, "generate_fleet_report",
    "Xuất báo cáo SO SÁNH nhiều máy ra file (html/pdf/excel/csv). "
    "TRUYỀN đúng những gì user ĐÃ nói — máy nào, kỳ nào, định dạng nào — kể cả khi "
    "họ nói ở lượt trước; “tất cả máy” là machines=['Tất cả']. "
    "Chỉ để TRỐNG những tham số user CHƯA nói: tool sẽ trả về danh sách lựa chọn "
    "để bạn hỏi lại đúng mục đó. Không tự chọn thay user, nhưng cũng không hỏi lại "
    "thứ họ vừa trả lời.",
    ReportArgs))


# ------------------------------------------------- nhân sự · nhật ký · log ---

async def fleet_staff(**_ignored: Any) -> Dict[str, Any]:
    """
    Nhân sự toàn nhà máy: ai, ở máy nào, bộ phận nào, ca nào, quyền gì.

    Dùng cho câu hỏi về con người và tổ chức: bộ phận nào bao nhiêu người, ai
    trực ca nào, một người có mặt ở những máy nào.
    """
    d = await queries.fleet_staff()
    users = [
        {"machine": u.get("machine"), "username": u.get("username"),
         "full_name": u.get("full_name"), "role": u.get("role"),
         "employee_code": u.get("employee_code"),
         "department": u.get("department"), "job_title": u.get("job_title"),
         "shift": u.get("shift"), "production_line": u.get("production_line"),
         "is_active": u.get("is_active")}
        for u in d["users"]
    ]

    # Đếm sẵn bằng CODE, không để mô hình tự đếm 54 dòng.
    #
    # Đã xảy ra thật: hỏi "bộ phận QA có bao nhiêu người" thì mô hình trả lời 10
    # ở văn xuôi và 13 ở bảng, trong khi số thật là 11 — nó đọc danh sách rồi
    # đếm tay, và đếm sai theo hai cách khác nhau trong cùng một câu trả lời.
    # Cùng nguyên tắc với `kpis`/`charts`: thứ gì code tính được thì đừng để mô
    # hình suy ra.
    def tally(key: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for u in users:
            out[u.get(key) or "(trống)"] = out.get(u.get(key) or "(trống)", 0) + 1
        return dict(sorted(out.items(), key=lambda x: -x[1]))

    cross: Dict[str, Dict[str, int]] = {}
    for u in users:
        dep = u.get("department") or "(trống)"
        cross.setdefault(dep, {})
        cross[dep][u["machine"]] = cross[dep].get(u["machine"], 0) + 1

    out = {
        "coverage": d["coverage"],
        "count": d["count"],
        "by_machine": d["by_machine"],
        "by_department": tally("department"),
        "by_role": tally("role"),
        "by_shift": tally("shift"),
        "by_department_and_machine": cross,
        "users": users,
        "note": ("CÁC SỐ ĐẾM Ở TRÊN LÀ CHUẨN — dùng chúng, đừng tự đếm lại từ "
                 "danh sách `users`. Cùng một `username` trên hai máy là HAI tài "
                 "khoản khác nhau (admin có trên cả 5 máy); khi nói về một người "
                 "luôn nêu kèm tên máy. Định danh xuyên máy duy nhất là "
                 "employee_code."),
    }
    _remember("fleet_staff", out)
    return out


class AuditArgs(BaseModel):
    days: int = Field(default=7, ge=1, le=90, description="Số ngày tính ngược từ hôm nay.")
    username: Optional[str] = Field(
        default=None,
        description="Lọc theo một người: truyền username (vd 'qc_tine'), HỌ TÊN "
                    "đầy đủ, hoặc mã nhân viên — trung tâm tự tra ra tài khoản. "
                    "Đừng truyền tên recipe hay tên máy vào đây.")
    action_type: Optional[str] = Field(
        default=None,
        description="Lọc theo loại thao tác: login, logout, create_user, "
                    "update_user, delete_user, reset_password, load_recipe, "
                    "stop_recipe, update_recipe.")
    machine: Optional[str] = Field(
        default=None,
        description="Hỏi về MỘT máy thì BẮT BUỘC truyền tên máy vào đây "
                    "(vd 'Auto2'). Không truyền thì kết quả gộp mọi máy và danh "
                    "sách bị cắt còn 25 dòng mới nhất — rất dễ không còn dòng "
                    "nào của máy bạn đang hỏi.")


def _spread(entries: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Lấy `limit` dòng nhưng RẢI ĐỀU qua các máy, giữ thứ tự thời gian.

    Cắt 25 dòng mới nhất toàn cục thì máy nào vừa hoạt động sẽ chiếm sạch, và
    các máy khác biến mất khỏi tầm nhìn của mô hình — im lặng, không báo lỗi.
    """
    if len(entries) <= limit:
        return entries
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        buckets.setdefault(e.get("machine") or "—", []).append(e)
    out: List[Dict[str, Any]] = []
    i = 0
    while len(out) < limit:
        added = False
        for rows in buckets.values():
            if i < len(rows):
                out.append(rows[i])
                added = True
                if len(out) >= limit:
                    break
        if not added:
            break
        i += 1
    out.sort(key=lambda e: e.get("time") or "", reverse=True)
    return out


async def fleet_audit_log(days: int = 7, username: Optional[str] = None,
                          action_type: Optional[str] = None,
                          machine: Optional[str] = None,
                          **_ignored: Any) -> Dict[str, Any]:
    """
    Ai làm gì, trên máy nào, lúc nào — gộp từ mọi máy.

    Bản ghi demo (`simulated`) đã bị loại sẵn.
    """
    # Tra người TRƯỚC khi hỏi nhật ký: ô này nhận cả họ tên lẫn username, mà bảng
    # audit chỉ tra được username.
    lookup = None
    if username:
        lookup = await queries.resolve_person(username)
        if lookup["status"] == "found":
            username = lookup["username"]
        else:
            # Không tra ra người thì DỪNG, không đi hỏi nhật ký. Hỏi tiếp chỉ nhận
            # về 5 lần "không có tài khoản này", đọc y hệt "người này chưa làm gì"
            # — và đó là một câu bịa về một con người có thật.
            return {"user_lookup": lookup,
                    "answer_is_unknown": True,
                    "note": ("KHÔNG tra được người này thành một tài khoản. Đây "
                             "KHÔNG phải bằng chứng người đó không thao tác gì — "
                             "chưa hề tra nhật ký. Nếu status là 'ambiguous', hãy "
                             "hỏi lại người dùng chọn tài khoản nào trong "
                             "`candidates`.")}

    d = await queries.fleet_audit(days=days, username=username,
                                  action_type=action_type, machine=machine)
    out = {
        "user_lookup": lookup,
        "coverage": d["coverage"],
        "period_days": d["period_days"],
        "total_in_period": d["total_in_period"],
        "by_action": d["by_action"],
        # Đếm theo máy trên TOÀN BỘ bản ghi, không phải trên phần đã cắt. Đây là
        # con số duy nhất được dùng để nói "máy X có/không có thao tác".
        "by_machine": d["by_machine"],
        "machine_filter": d.get("machine_filter"),
        "machines_without_user": d["machines_without_user"],
        # Cắt còn 25 dòng cho mô hình: nó cần thấy hình dạng, không cần đọc hết.
        # Con số tổng đã nằm ở `by_action`, tính trên TOÀN BỘ kỳ chứ không phải
        # trên phần bị cắt.
        # Cắt bớt phải RẢI ĐỀU qua các máy, không lấy 25 dòng mới nhất toàn cục.
        # Đo được: hỏi về Auto2, 25 dòng mới nhất toàn là M2 và PC-Auto-1, không
        # còn dòng nào của Auto2 — và câu trả lời thành "Auto2 không có thao tác
        # nào" trong khi Auto2 có 16 thao tác recipe.
        "entries": _spread(d["entries"], 25),
        "entries_shown": min(25, d["count"]),
        "entries_total_fetched": d["count"],
        "note": d["note"] + (
            " `entries` CHỈ là mẫu rải đều để xem hình dạng — muốn biết một máy"
            " có thao tác hay không thì đọc `by_machine`, KHÔNG đếm trong"
            " `entries`. Hỏi về một máy cụ thể thì truyền tham số `machine`."
            " `machines_without_user` là các máy KHÔNG CÓ tài khoản này — đó là "
            "chuyện bình thường (tài khoản chỉ tồn tại trên một số máy), KHÔNG "
            "phải máy hỏng. Máy hỏng nằm ở coverage.machines_missing."),
    }
    _remember("fleet_audit_log", out)
    return out


class LogErrorArgs(BaseModel):
    date: Optional[str] = Field(default=None, description="Ngày YYYY-MM-DD, để trống là hôm nay.")
    top: int = Field(default=8, ge=1, le=30, description="Số nhóm lỗi lấy mỗi máy.")


async def fleet_log_errors(date: Optional[str] = None, top: int = 8,
                           **_ignored: Any) -> Dict[str, Any]:
    """
    Lỗi và cảnh báo trong LOG HỆ THỐNG của mọi máy, đã gom nhóm.

    Khác `fleet_audit_log` (thao tác của người) và khác `fleet_health` (phần
    cứng): đây là thứ các service tự ghi ra khi có chuyện — dùng khi được hỏi
    máy nào đang kêu gì, có lỗi gì lặp lại, service nào bất ổn.
    """
    d = await queries.fleet_log_errors(date=date, top=top)
    _remember("fleet_log_errors", d)
    return d


FLEET_TOOLS.extend([
    _tool(fleet_staff, "fleet_staff",
          "Nhân sự toàn nhà máy: ai ở máy nào, bộ phận, chức vụ, ca, quyền. Rẻ."),
    _tool(fleet_audit_log, "fleet_audit_log",
          "Nhật ký thao tác của người dùng trên mọi máy: ai làm gì lúc nào. "
          "Đã loại bản ghi demo. Rẻ.", AuditArgs),
    _tool(fleet_log_errors, "fleet_log_errors",
          "Lỗi/cảnh báo trong LOG HỆ THỐNG của mọi máy, đã gom nhóm. Khác với "
          "nhật ký thao tác (việc của người) và khác sức khoẻ phần cứng. Rẻ.",
          LogErrorArgs),
])
