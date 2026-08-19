"""
Ngôn ngữ hiển thị của agent service.

Bối cảnh: toàn bộ service được viết bằng tiếng Việt, và chuỗi tiếng Việt trong
mã nguồn chia làm hai loại rất khác nhau:

1. **Chỉ dẫn cho mô hình** (~1.150 chuỗi trong `prompts/`, docstring của tool,
   trường `note` trong kết quả tool). Đây là lời dặn cho LLM, không ai thấy.
   Đổi ngôn ngữ trả lời KHÔNG cần dịch phần này — mô hình đọc tiếng Việt và
   trả lời tiếng Anh hoàn toàn bình thường. Chỉ cần thêm một dòng vào system
   prompt là xong.

2. **Nhãn hiện lên UI** (~174 chuỗi trong `core/attachments.py` và
   `core/suggestions.py`): tiêu đề ô KPI, tên cột bảng, tiêu đề biểu đồ, nhãn
   thẻ, câu gợi ý. Những chuỗi này do CODE sinh ra, không đi qua LLM, nên mô
   hình không thể dịch giúp. Đây mới là phần cần lớp dịch — và là lý do file
   này tồn tại.

Cách dùng: `t("Sản lượng")`. Chính chuỗi tiếng Việt là khoá tra bảng, giống
gettext. Chọn như vậy vì mã nguồn đã viết bằng tiếng Việt: bọc `t()` vào là
xong, không phải bịa ra tên khoá, và nếu thiếu bản dịch thì trả về nguyên
tiếng Việt — nhãn cũ vẫn hiện, không bao giờ ra ô trống hay vỡ giao diện.

Ngôn ngữ được giữ trong `ContextVar` chứ không truyền qua tham số. Nhãn được
dựng ở tận trong vòng lặp của từng agent (`historical_agent`, `log_agent`,
`attachments`), truyền `lang` xuống tới đó phải sửa hàng chục chữ ký hàm —
trong khi mỗi request là một task asyncio riêng nên ContextVar cách ly đúng
theo request mà không rò rỉ sang request khác.
"""

from contextvars import ContextVar
from typing import Dict, List, Optional

# Ngôn ngữ hỗ trợ. `code` dùng trong API, `name` là thứ đưa vào system prompt
# cho mô hình, `label` để FE hiện trong dropdown.
LANGUAGES: Dict[str, Dict[str, str]] = {
    "vi": {"name": "Vietnamese (tiếng Việt)", "label": "Tiếng Việt"},
    "en": {"name": "English",                 "label": "English"},
}

# "auto" không phải một ngôn ngữ mà là một chính sách: trả lời bằng đúng ngôn
# ngữ user vừa dùng. Nhãn UI khi đó lấy theo DEFAULT_LANG vì code không đoán
# ngôn ngữ của câu hỏi — việc đoán để mô hình làm, nó làm tốt hơn regex.
AUTO = "auto"
DEFAULT_LANG = "vi"

_lang: ContextVar[str] = ContextVar("display_lang", default=DEFAULT_LANG)

# Mã user yêu cầu, giữ nguyên văn — có thể là "auto". Cần tách khỏi `_lang` vì
# hai thứ khác nhau: "auto" thì nhãn UI phải chọn một ngôn ngữ cụ thể (tiếng
# Việt) trong khi câu trả lời của mô hình lại bám theo ngôn ngữ user vừa gõ.
_requested: ContextVar[str] = ContextVar("requested_lang", default=DEFAULT_LANG)


# Dấu thanh và nguyên âm riêng của tiếng Việt. Có một chữ trong này là chắc chắn
# tiếng Việt — không ngôn ngữ nào khác trong hệ hỗ trợ dùng các ký tự này.
_VN_CHARS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữựỳýỷỹỵđ"
)

# Người trong xưởng hay gõ không dấu, lúc đó bảng trên vô dụng. Đây là các từ
# công cụ tiếng Việt không dấu — chọn những từ không trùng từ tiếng Anh nào
# ("do", "la", "may", "tai" bị loại vì trùng).
_VN_ASCII = {
    "khong", "bao nhieu", "hom nay", "hom qua", "the nao", "san luong",
    "san pham", "nhieu nhat", "kiem tra", "bi loi", "tai sao", "vi sao",
    "cua", "duoc", "nhung", "gio", "ngay", "thang", "tuan", "ca dem",
    "thong ke", "bao cao", "xuat", "dang chay", "cho toi", "giup toi",
}


def detect(text: str) -> str:
    """
    Đoán ngôn ngữ của một câu hỏi. Trả về mã trong `LANGUAGES`.

    Cần hàm này vì chế độ "auto" giao cho mô hình tự bám ngôn ngữ đã KHÔNG chạy:
    system prompt dài 18.000 ký tự tiếng Việt lấn át, câu hỏi tiếng Anh vẫn bị
    trả lời bằng tiếng Việt. Và kể cả nếu mô hình làm đúng, nhãn UI do code sinh
    (ô KPI, cột bảng) vẫn không biết phải theo ngôn ngữ nào — kết quả là câu trả
    lời tiếng Anh nằm cạnh ô KPI tiếng Việt.

    Nhận biết ở đây chỉ cần phân biệt vi/en nên không cần thư viện: dấu thanh
    tiếng Việt là bằng chứng tuyệt đối, còn trường hợp gõ không dấu thì dựa vào
    danh sách từ công cụ. Đoán sai thì hậu quả nhẹ — trả lời sai thứ tiếng, user
    chọn cứng ngôn ngữ trong dropdown là xong.
    """
    if not text:
        return DEFAULT_LANG
    low = text.lower()
    if any(ch in _VN_CHARS for ch in low):
        return "vi"
    if any(w in low for w in _VN_ASCII):
        return "vi"
    return "en"


def resolve(code: Optional[str]) -> str:
    """Chuẩn hoá mã ngôn ngữ từ request. Mã lạ → mặc định, không báo lỗi."""
    if not code:
        return DEFAULT_LANG
    code = code.strip().lower()
    if code == AUTO:
        return AUTO
    code = code.split("-")[0]          # 'en-US' → 'en'
    return code if code in LANGUAGES else DEFAULT_LANG


def set_lang(code: Optional[str], text: str = "") -> str:
    """
    Đặt ngôn ngữ cho request hiện tại. Trả về mã ngôn ngữ CỤ THỂ đã dùng.

    `text` là tin nhắn của user. Với code="auto", ngôn ngữ được đoán từ chính
    tin nhắn đó và từ đây trở đi "auto" biến thành một ngôn ngữ cụ thể: cả prompt
    của mô hình lẫn nhãn UI đều dùng chung một ngôn ngữ, nên không còn cảnh câu
    trả lời tiếng Anh nằm cạnh ô KPI tiếng Việt.
    """
    resolved = resolve(code)
    if resolved == AUTO:
        resolved = detect(text)
    _requested.set(resolved)
    _lang.set(resolved)
    return resolved


def get_lang() -> str:
    """Ngôn ngữ cho NHÃN UI — luôn là một ngôn ngữ cụ thể, không bao giờ 'auto'."""
    return _lang.get()


def requested() -> str:
    """Mã user yêu cầu, có thể là 'auto'. Dùng để sinh dòng ngôn ngữ cho prompt."""
    return _requested.get()


def answer_language_line(code: Optional[str]) -> str:
    """
    Khối chỉ dẫn ngôn ngữ, dùng để NHẮC LẠI ở cuối system prompt.

    Xem `apply_language()` — khối này một mình không đủ.
    """
    resolved = resolve(code)
    if resolved == AUTO:
        return (
            "\n\n## 🌐 NGÔN NGỮ TRẢ LỜI\n\n"
            "Trả lời bằng ĐÚNG ngôn ngữ mà user dùng trong câu hỏi cuối cùng. "
            "User đổi ngôn ngữ giữa cuộc thì bám theo câu mới nhất.\n"
        )
    name = LANGUAGES[resolved]["name"]
    return (
        "\n\n## 🌐 NGÔN NGỮ TRẢ LỜI\n\n"
        f"NHẮC LẠI: viết câu trả lời bằng {name}. Toàn bộ prompt ở trên viết bằng "
        f"tiếng Việt và các ví dụ cũng bằng tiếng Việt, nhưng đó là chỉ dẫn dành "
        f"cho bạn — KHÔNG phải mẫu ngôn ngữ để bắt chước. Câu trả lời phải bằng "
        f"{name}.\n"
    )


def apply_language(prompt: str, code: Optional[str] = None) -> str:
    """
    Gắn chỉ dẫn ngôn ngữ vào một system prompt, ở CẢ ĐẦU VÀ CUỐI.

    Đặt ở cuối thôi là không đủ, đã thử và thất bại: system prompt của agent
    historical dài 18.000 ký tự toàn tiếng Việt, kèm hàng chục câu trả lời mẫu
    bằng tiếng Việt. Một dòng "trả lời bằng English" nằm ở cuối bị chỗ đó lấn
    át hoàn toàn — mô hình bắt chước ngôn ngữ của ví dụ chứ không nghe chỉ dẫn,
    nên vẫn trả lời tiếng Việt dù đã chọn English.

    Đặt lên đầu để nó là thứ đầu tiên mô hình đọc, rồi nhắc lại ở cuối để nó là
    thứ cuối cùng mô hình đọc trước khi viết — và nói thẳng rằng các ví dụ tiếng
    Việt trong prompt là chỉ dẫn, không phải mẫu ngôn ngữ cần bắt chước.
    """
    resolved = resolve(code if code is not None else requested())
    if resolved == AUTO:
        head = (
            "# 🌐 NGÔN NGỮ\n\n"
            "Trả lời bằng ĐÚNG ngôn ngữ user dùng ở câu hỏi cuối cùng. Prompt dưới "
            "đây viết bằng tiếng Việt vì đó là chỉ dẫn cho bạn, không phải quy định "
            "ngôn ngữ trả lời.\n\n---\n\n"
        )
    else:
        name = LANGUAGES[resolved]["name"]
        head = (
            "# 🌐 NGÔN NGỮ — ĐỌC TRƯỚC TIÊN\n\n"
            f"**Viết MỌI câu trả lời bằng {name}.** Kể cả khi user hỏi bằng ngôn ngữ "
            f"khác: user đã chọn {name} trong cài đặt.\n\n"
            f"Phần chỉ dẫn phía dưới viết bằng tiếng Việt và mọi câu trả lời mẫu "
            f"trong đó cũng bằng tiếng Việt. Đó là CHỈ DẪN NGHIỆP VỤ cho bạn, "
            f"KHÔNG phải mẫu ngôn ngữ để bắt chước. Đọc nội dung, rồi viết bằng "
            f"{name}.\n\n"
            f"Giữ nguyên không dịch: tên recipe, tên camera, username, mã nhân "
            f"viên, và các từ trong bảng TỪ VỰNG CỐ ĐỊNH.\n\n---\n\n"
        )
    return head + prompt + answer_language_line(resolved)


# ---------------------------------------------------------------------------
# Bảng dịch. Khoá = chuỗi tiếng Việt xuất hiện trong mã nguồn.
# ---------------------------------------------------------------------------

_EN: Dict[str, str] = {
    # --- ô KPI ---
    "Sản lượng": "Output",
    "Sản lượng ca": "Shift output",
    "Sản phẩm": "Units",
    "Tổng sản phẩm": "Total units",
    "Pass rate": "Pass rate",
    "Pass": "Pass",
    "Fail": "Fail",
    "Uptime": "Uptime",
    "Thời gian dừng": "Downtime",
    "Số lần dừng": "Stops",
    "Hoàn thành": "Completion",
    "Cảnh báo trong ca": "Alerts this shift",
    "Dự phóng cuối ngày": "End-of-day projection",
    "Còn thiếu": "Remaining",
    "Chỉ tiêu": "Target",
    "Ca làm việc": "Shift",
    "Dây chuyền": "Line",
    "Hoạt động": "Active",

    # --- cột bảng ---
    "Giờ": "Time",
    "Người": "User",
    "Thao tác": "Action",
    "Nội dung": "Details",
    "Từ": "From",
    "Đến": "To",
    "Số phút": "Minutes",
    "Recipe": "Recipe",
    "SL trước": "Units before",
    "SL nay": "Units now",
    "Thay đổi SL": "Unit change",
    "Pass trước": "Pass before",
    "Pass nay": "Pass now",
    "Chênh lệch (điểm %)": "Change (pp)",
    "Ghi chú": "Note",
    "Nguyên nhân": "Cause",
    "Số lượng": "Count",
    "Tỷ lệ": "Share",
    "Camera": "Camera",
    "Ca": "Shift",
    "Ngày": "Date",

    # --- nhãn thẻ nhân sự ---
    "Bộ phận": "Department",
    "Chức vụ": "Job title",
    "Email": "Email",
    "Điện thoại": "Phone",
    "Mã nhân viên": "Employee ID",
    "Quyền": "Role",
    "Số thao tác": "Actions",
    "Vào làm": "Hired",
    "Số giờ có mặt": "Hours on shift",
    "thao tác": "actions",
    "Giờ hoạt động": "Active hours",
    "không rõ": "unknown",

    # --- nguyên nhân fail ---
    "Chưa xác định": "Unclassified",
    "Không khớp template": "Template mismatch",
    "Không nhận ra sản phẩm": "Product not recognised",
    "Không thấy nhãn trong khung": "No label in frame",
    "Sai ký tự": "Character mismatch",

    # --- trạng thái / phụ đề ---
    "· đang chạy": "· running",
    "đang chạy": "running",
    "chưa bắt đầu": "not started",
    "Chỉ hiện dòng ERROR": "ERROR lines only",
    "Gom nhóm lỗi trong ngày": "Group today's errors",
    "Dừng Camera service": "Stop Camera service",

    # --- câu gợi ý ---
    "Ai load recipe hôm nay?": "Who loaded a recipe today?",
    "Ai load recipe nhiều nhất?": "Who loads recipes most often?",
    "Ai thao tác gì lúc đó?": "Who did what at that moment?",
    "Ca nào dừng nhiều nhất?": "Which shift has the most downtime?",
    "Ca nào đóng góp nhiều nhất?": "Which shift produced the most?",
    "Camera nào fail nhiều nhất?": "Which camera fails most?",
    "Còn thiếu bao nhiêu?": "How far from target?",
    "Có module nào lỗi không?": "Any module in error?",
    "Có sản phẩm nào bị bỏ sót?": "Any units missed?",
    "Cảm biến có ổn không?": "Are the sensors healthy?",
    "Hôm nay có lỗi gì không?": "Any errors today?",
    "Hôm qua có lỗi tương tự không?": "Same errors yesterday?",
    "Hôm qua thế nào?": "How was yesterday?",
    "So với hôm qua thế nào?": "Compared with yesterday?",
    "So với tuần trước thế nào?": "Compared with last week?",
    "Xuất báo cáo ngày hôm nay": "Export today's report",
    "Xem chi tiết nguyên nhân fail": "Break down the fail causes",
    "Tình trạng thiết bị thế nào?": "How is the equipment doing?",
    "Bản giao ca ca này": "Handover for this shift",
    "Đạt chỉ tiêu chưa?": "On target?",
    "Hôm nay sản xuất bao nhiêu sản phẩm?": "How many units did we make today?",
    "Camera nào fail nhiều nhất hôm nay?": "Which camera failed most today?",
    "Xu hướng pass rate 7 ngày qua": "Pass-rate trend over the last 7 days",
    "Camera service có đang chạy không?": "Is the camera service running?",
    "Báo cáo 7 ngày qua": "Last 7 days report",
    "Kiểm tra thiết bị": "Check the equipment",
    "Kiểm tra trigger và cảm biến": "Check trigger and sensors",
    "Log của module đó báo gì?": "What does that module's log say?",
    "Lúc đó hệ thống báo lỗi gì?": "What error did the system report then?",
    "Lúc đó log báo gì?": "What did the log say at that time?",
    "Nhịp dây chuyền hôm qua thế nào?": "How was the line cadence yesterday?",
    "Phân tích theo camera": "Break down by camera",
    "Phân tích theo từng giờ": "Break down by hour",
    "Recipe nào tệ hơn kỳ trước?": "Which recipe got worse?",
    "Recipe nào đang chạy?": "Which recipe is running?",
    "Service có kết nối WebSocket không?": "Is the service WebSocket connected?",
    "So sánh 7 ngày qua": "Compare the last 7 days",
    "So sánh với ca trước": "Compare with the previous shift",
    "So sánh với hôm qua": "Compare with yesterday",
    "Trigger có ổn không?": "Is the trigger healthy?",
    "Trong log có lỗi gì không?": "Any errors in the log?",
    "Tìm trong 7 ngày qua": "Search the last 7 days",
    "Vì sao dừng máy lúc đó?": "Why did the line stop then?",
    "Vì sao service restart?": "Why did the service restart?",
    "Xem 100 dòng log cuối": "Show the last 100 log lines",
    "Xem log backend mới nhất": "Show the latest backend log",
    "Xem log gần đây của service": "Show the service's recent log",
    "Xem log gốc của lỗi này": "Show the raw log for this error",
    "Xu hướng 7 ngày qua": "7-day trend",
    "Xung reject có đúng không?": "Is the reject pulse correct?",
    "Xuất PDF để in": "Export a printable PDF",
    "Xuất báo cáo kỳ này": "Export a report for this period",
    "Xuất bản Excel luôn": "Export as Excel too",

    # --- nhãn kỳ ---
    "ngày": "days",
    "hôm nay": "today",
    "hôm qua": "yesterday",
    "tuần này": "this week",
    "tuần trước": "last week",
    "tháng này": "this month",
    "tháng trước": "last month",
    "7 ngày qua": "last 7 days",
    "30 ngày qua": "last 30 days",

    # --- mảnh câu ghép trong tiêu đề biểu đồ ---
    "Nguyên nhân fail": "Fail causes",
    "Sản phẩm FAIL theo camera": "FAIL units by camera",
    "Sản phẩm FAIL theo giờ": "FAIL units by hour",
    "Sản phẩm FAIL theo recipe": "FAIL units by recipe",
    "Sản phẩm FAIL theo ca": "FAIL units by shift",
    "Sản lượng theo ngày": "Output by day",
    "Sản lượng theo recipe": "Output by recipe",
    "Sản lượng theo giờ": "Output by hour",
    "Sản lượng theo ca": "Output by shift",
    "So sánh theo recipe": "Recipe comparison",
    "Thay đổi pass rate": "Pass-rate change",
    "Các lần dừng dây chuyền": "Line stops",
    "Các lần dừng trong ca": "Stops this shift",
    "Thay đổi recipe trong ca": "Recipe changes this shift",
    "Luỹ tiến so với chỉ tiêu": "Progress against target",
    "Báo cáo": "Report",
    "điểm": "pp",
    "trong giờ": "in hour",
    "so với": "vs",
    "chỉ": "only",
    "mốc đầu không hiện": "leading points hidden",
    "mục nhỏ hơn không hiện": "smaller items hidden",
}

_TABLES: Dict[str, Dict[str, str]] = {"en": _EN}


def t(s: str, lang: Optional[str] = None) -> str:
    """
    Dịch một nhãn. Thiếu bản dịch thì trả về nguyên chuỗi vào.

    Fallback im lặng là có chủ ý: nhãn chưa dịch hiện ra tiếng Việt vẫn đọc
    được, còn KeyError giữa lúc dựng câu trả lời thì làm mất cả câu trả lời.
    """
    table = _TABLES.get(lang or get_lang())
    return table.get(s, s) if table else s


def tf(template: str, lang: Optional[str] = None, **kw) -> str:
    """
    Dịch một mẫu câu rồi mới điền tham số.

    Dùng cho tiêu đề ghép như `f"Nguyên nhân fail · {period}"`: dịch phần chữ,
    giữ nguyên phần dữ liệu.
    """
    return t(template, lang).format(**kw)


_SEP = " · "


def tseg(s: str, lang: Optional[str] = None) -> str:
    """
    Dịch một tiêu đề ghép bằng dấu ' · ', từng đoạn một.

    Tiêu đề biểu đồ trong service này được dựng kiểu
    `f"Nguyên nhân fail · {period}"` — nửa đầu là chữ cố định, nửa sau là dữ
    liệu (tên recipe, nhãn kỳ, ngày). Tra cả câu thì không bao giờ khớp bảng,
    nên tách theo dấu phân cách rồi tra từng đoạn: đoạn nào là chữ cố định thì
    được dịch, đoạn nào là dữ liệu thì không khớp bảng và giữ nguyên. Nhãn kỳ
    ("7 ngày qua") có trong bảng nên cũng được dịch luôn — đúng mong muốn.
    """
    if not s or _SEP not in s:
        return t(s, lang)
    return _SEP.join(t(part.strip(), lang) for part in s.split(_SEP))


def tcols(cols: List[str], lang: Optional[str] = None) -> List[str]:
    """Dịch cả hàng tiêu đề của bảng."""
    return [t(c, lang) for c in cols]


def missing_translations(lang: str = "en") -> List[str]:
    """Khoá nào chưa có bản dịch — dùng cho test, không dùng lúc chạy."""
    table = _TABLES.get(lang) or {}
    return [k for k, v in table.items() if not v]
