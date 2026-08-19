"""
Tools đọc và phân tích log.

Hai nguồn log hoàn toàn khác nhau:

  1. Log file trên đĩa — `logs/{category}/{YYYY-MM-DD}.log`, cùng thư mục mà
     `backend/app/api/endpoints/system_logs.py` phục vụ cho tab System Logs.
  2. Audit log trong MongoDB — collection `action_logs`, là thứ tab Audit Log
     hiển thị: ai đã đăng nhập, sửa recipe, load recipe, đổi camera.

RÀNG BUỘC QUAN TRỌNG — kích thước file:

    logs/start_services/ai_camera.log     543 MB
    logs/camera_management/2026-08-18.log 155 MB
    logs/backend.log                      579 MB

Đọc trọn một file như vậy là treo cả agent service và ăn hết RAM của Jetson.
Mọi tool ở đây vì thế đều đọc có chặn: `read_log_tail` seek ngược từ cuối file,
`search_logs` và `summarize_log_errors` quét tuần tự nhưng dừng khi chạm trần
byte/số dòng khớp. Không tool nào gọi `.read()` hay `.readlines()` trần.
"""

import gzip
import io
import re
import traceback
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent_app.core.config import settings
from agent_app.db.mongodb import get_sync_database
from agent_app.tools.base_tool import BaseTool, ToolMetadata

# Dùng lại đúng hàm quy đổi múi giờ của analytics_tools thay vì viết bản thứ
# hai: hai chỗ lệch nhau một tiếng thì cùng một câu hỏi sẽ ra hai kết quả, mà
# loại lỗi đó rất khó nhìn ra khi đọc câu trả lời.
from agent_app.tools.analytics_tools import _local_bound, _to_local_str, _TZ

import logging

logger = logging.getLogger(__name__)


# ── Giới hạn an toàn ─────────────────────────────────────────────────────────

LOGS_ROOT = settings.project_root / "logs"

#: Số byte tối đa đọc ngược từ cuối file cho `read_log_tail`.
MAX_TAIL_BYTES = 8 * 1024 * 1024

#: Số byte tối đa quét cho mỗi file khi tìm kiếm / thống kê lỗi.
MAX_SCAN_BYTES = 64 * 1024 * 1024

#: Độ dài tối đa của một dòng trả về; dòng log có thể nhúng cả payload JSON.
MAX_LINE_CHARS = 400

CATEGORIES = [
    "backend",
    "camera_settings",
    "camera_management",
    "trigger_stats",
    "pulse_width",
    "reject_actions",
    "obb_rotation",
    "camera_check",
    "start_services",
]

#: Nhóm field hồ sơ nhân sự trên document `users`.
#:
#: Không nằm trong pydantic model của backend — được ghi trực tiếp vào MongoDB.
#: Cố ý như vậy để không phải sửa `backend/app/models/user.py`: hai codebase đang
#: được giữ tách rời, và `update_user` của backend dùng `$set` với đúng các field
#: nó biết, nên field lạ vẫn sống sót qua mọi lần sửa user trên UI.
#:
#: Hệ quả cần biết: UI của backend chưa hiển thị/sửa được nhóm này.
PROFILE_FIELDS = (
    "employee_code",    # mã nhân viên
    "department",       # bộ phận
    "job_title",        # chức vụ
    "shift",            # ca làm việc
    "production_line",  # dây chuyền phụ trách
    "hire_date",        # ngày vào làm
)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LEVEL_RE = re.compile(r"\b(CRITICAL|FATAL|ERROR|WARNING|WARN|INFO|DEBUG)\b")

# Định dạng dòng không thống nhất giữa các category:
#   backend/camera_management : '2026-08-19 11:13:52,189 - logger - INFO - msg'
#   pulse_width/reject_actions: '2026-08-19 11:14:30.901 | DI0 | pulse_width=...'
#   start_services/frontend   : '11:11:09 AM [vite] (client) hmr update ...'
# Nên timestamp được bóc bằng một regex nới, và thiếu thì cũng không sao.
# Phần mili-giây phải nằm TRONG regex dù không cần dùng tới: `_signature` cắt
# theo `match.end()`, bỏ sót nó thì mọi chữ ký còn dính đuôi ',189 - ' và hai
# dòng cùng một lỗi ở hai mili-giây khác nhau bị đếm thành hai vấn đề riêng.
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:[.,]\d{1,6})?")


def _safe_category(category: str) -> Path:
    if category not in CATEGORIES:
        raise ValueError(f"Category không hợp lệ '{category}'. Hợp lệ: {', '.join(CATEGORIES)}")
    return LOGS_ROOT / category


def _resolve_file(folder: Path, date: str) -> Optional[Path]:
    """File log của một ngày, ưu tiên bản chưa nén."""
    raw, gz = folder / f"{date}.log", folder / f"{date}.log.gz"
    if raw.exists():
        return raw
    if gz.exists():
        return gz
    return None


def _list_dates(folder: Path) -> List[str]:
    if not folder.is_dir():
        return []
    out = set()
    for p in folder.iterdir():
        stem = p.name.split(".log")[0]
        if _DATE_RE.match(stem):
            out.add(stem)
    return sorted(out, reverse=True)


def _named_files(folder: Path) -> List[Dict[str, Any]]:
    """
    Các file log đặt theo TÊN thay vì theo ngày.

    `logs/start_services/` chứa `ai_camera.log`, `frontend.log`, `ngrok_api.log`…
    — do script khởi động ghi ra, không xoay vòng theo ngày. Riêng
    `ai_camera.log` đang là 543 MB, tức nơi nhiều dữ liệu nhất lại là nơi mà
    cách dò theo `YYYY-MM-DD.log` không nhìn thấy. Bỏ qua thì agent sẽ trả lời
    "không có log" trong khi log vẫn nằm đó.
    """
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or ".log" not in p.name:
            continue
        if _DATE_RE.match(p.name.split(".log")[0]):
            continue                      # đã được `_list_dates` kể rồi
        st = p.stat()
        out.append({
            "file": p.name,
            "size_mb": round(st.st_size / 1_048_576, 1),
            "modified": datetime.fromtimestamp(st.st_mtime, _TZ).strftime("%Y-%m-%d %H:%M"),
        })
    return out


def _pick_file(folder: Path, date: Optional[str], file: Optional[str]) -> Optional[Path]:
    """Chọn file theo tên nếu có `file`, ngược lại theo ngày."""
    if file:
        # Chốt trong đúng thư mục category — tên file do LLM sinh ra, không
        # được để '../../etc/passwd' đi lọt.
        candidate = (folder / Path(file).name)
        return candidate if candidate.is_file() else None
    return _resolve_file(folder, date or _today())


def _today() -> str:
    return datetime.now(_TZ).strftime("%Y-%m-%d")


def _clip(line: str) -> str:
    line = line.rstrip("\n")
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + " …[cắt bớt]"


def _level_of(line: str) -> Optional[str]:
    m = _LEVEL_RE.search(line)
    if not m:
        return None
    lv = m.group(1)
    return {"WARN": "WARNING", "FATAL": "CRITICAL"}.get(lv, lv)


def _matches(line: str, level: Optional[str], contains: Optional[str]) -> bool:
    if level and _level_of(line) != level.upper():
        return False
    if contains and contains.lower() not in line.lower():
        return False
    return True


def _iter_lines(path: Path, max_bytes: int):
    """
    Duyệt từng dòng, dừng khi đã đọc quá `max_bytes`.

    File .gz phải giải nén tuần tự (không seek được), nên trần byte tính trên
    dữ liệu đã bung chứ không phải kích thước file trên đĩa.
    """
    read = 0
    opener = (lambda: io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")) \
        if path.suffix == ".gz" else \
        (lambda: open(path, "r", encoding="utf-8", errors="replace"))

    with opener() as f:
        for line in f:
            read += len(line)
            if read > max_bytes:
                break
            yield line


def _tail_lines(path: Path, want: int) -> List[str]:
    """
    N dòng cuối, đọc ngược từ cuối file theo từng khối.

    Không dùng `readlines()[-n:]`: với file 543 MB thì cách đó nạp toàn bộ vào
    RAM chỉ để lấy vài trăm dòng cuối.
    """
    if path.suffix == ".gz":
        # gzip không seek ngược được; buộc phải giải nén tuần tự, nhưng chỉ giữ
        # lại `want` dòng cuối trong một vòng đệm trượt.
        from collections import deque
        buf: deque = deque(maxlen=want)
        for line in _iter_lines(path, MAX_SCAN_BYTES):
            buf.append(line)
        return list(buf)

    size = path.stat().st_size
    chunk = 256 * 1024
    data = b""
    with open(path, "rb") as f:
        pos = size
        while pos > 0 and data.count(b"\n") <= want and (size - pos) < MAX_TAIL_BYTES:
            step = min(chunk, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
    text = data.decode("utf-8", errors="replace")
    return text.splitlines(keepends=True)[-want:]


def _scan_targets(cat: str, dates: List[str]) -> List[Path]:
    """
    Các file của một category cần quét cho `dates`.

    Category nào không có file theo ngày (như `start_services`) thì lấy các file
    đặt theo tên — nếu không, quét sẽ lặng lẽ bỏ qua đúng những category đó và
    câu trả lời "không tìm thấy gì" là sai sự thật chứ không phải kết luận.
    """
    folder = LOGS_ROOT / cat
    found = [p for d in dates if (p := _resolve_file(folder, d)) is not None]
    if found:
        return found
    return [folder / f["file"] for f in _named_files(folder)]


def _clock_of(line: str) -> Optional[str]:
    """'HH:MM:SS' của dòng, hoặc None nếu dòng không có timestamp."""
    m = _TS_RE.match(line)
    return m.group(1)[11:] if m else None


def _in_window(line: str, start_time: Optional[str], end_time: Optional[str]) -> bool:
    """
    Dòng có nằm trong khung giờ không.

    So sánh chuỗi 'HH:MM:SS' theo thứ tự từ điển — với giờ 24h zero-pad thì thứ
    tự từ điển trùng thứ tự thời gian, không cần parse ra datetime.

    Dòng không có timestamp (log của vài service ngoài không ghi giờ) bị LOẠI khi
    có lọc giờ: không định vị được thì không thể khẳng định nó thuộc khung user
    hỏi, mà đưa vào sẽ làm sai kết luận "lúc 5 giờ có chuyện gì".
    """
    if not start_time and not end_time:
        return True
    clock = _clock_of(line)
    if clock is None:
        return False
    if start_time and clock < _pad_time(start_time):
        return False
    if end_time and clock > _pad_time(end_time, end=True):
        return False
    return True


def _pad_time(value: str, end: bool = False) -> str:
    """'5' → '05:00:00', '05:30' → '05:30:00' (hoặc '05:30:59' cho mốc cuối)."""
    parts = value.strip().split(":")
    h = parts[0].zfill(2)
    m = parts[1].zfill(2) if len(parts) > 1 else ("59" if end else "00")
    sec = parts[2].zfill(2) if len(parts) > 2 else ("59" if end else "00")
    return f"{h}:{m}:{sec}"


# ── 1. Liệt kê nguồn log ─────────────────────────────────────────────────────

class ListLogSourcesArgs(BaseModel):
    pass


def list_log_sources() -> Dict[str, Any]:
    """Các nhóm log đang có, kèm dung lượng và ngày mới nhất."""
    try:
        out = []
        for cat in CATEGORIES:
            folder = LOGS_ROOT / cat
            dates = _list_dates(folder)
            size = 0
            if folder.is_dir():
                size = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
            out.append({
                "category": cat,
                "file_count": len(dates),
                "size_mb": round(size / 1_048_576, 1),
                "latest_date": dates[0] if dates else None,
                "available_dates": dates[:14],
                "named_files": _named_files(folder),
            })
        return {
            "success": True,
            "today": _today(),
            "categories": out,
            "note": (
                "`available_dates` chỉ liệt kê 14 ngày gần nhất. Một số category "
                "không xoay vòng theo ngày mà đặt theo tên service — chúng nằm ở "
                "`named_files`, muốn đọc thì truyền tham số `file` thay cho `date`. "
                "File log rất lớn (có file tới hàng trăm MB) nên mọi tool đọc log "
                "đều có giới hạn; đừng yêu cầu đọc trọn file."
            ),
        }
    except Exception as e:
        logger.error(f"list_log_sources lỗi: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── 2. Đọc phần cuối log ─────────────────────────────────────────────────────

class ReadLogTailArgs(BaseModel):
    category: str = Field(description=f"Nhóm log. Hợp lệ: {', '.join(CATEGORIES)}")
    date: Optional[str] = Field(default=None, description="Ngày YYYY-MM-DD, bỏ trống = hôm nay")
    file: Optional[str] = Field(
        default=None,
        description="Tên file cụ thể (vd 'ai_camera.log') cho các category đặt theo tên "
                    "thay vì theo ngày; xem `named_files` của list_log_sources. Có `file` thì bỏ qua `date`.",
    )
    lines: int = Field(default=100, description="Số dòng cuối cần lấy (tối đa 500)")
    level: Optional[str] = Field(default=None, description="Lọc theo mức: ERROR, WARNING, INFO, DEBUG, CRITICAL")
    contains: Optional[str] = Field(default=None, description="Chỉ giữ dòng chứa chuỗi này (không phân biệt hoa thường)")


def read_log_tail(
    category: str,
    date: Optional[str] = None,
    file: Optional[str] = None,
    lines: int = 100,
    level: Optional[str] = None,
    contains: Optional[str] = None,
) -> Dict[str, Any]:
    """Lấy các dòng cuối của một file log, lọc được theo mức và theo từ khoá."""
    try:
        folder = _safe_category(category)
        if not file:
            date = date or _today()
            if not _DATE_RE.match(date):
                return {"success": False, "error": "Ngày phải có dạng YYYY-MM-DD"}

        path = _pick_file(folder, date, file)
        if path is None:
            return {
                "success": False,
                "error": (f"Không có file '{file}' trong '{category}'" if file
                          else f"Không có log của '{category}' ngày {date}"),
                "available_dates": _list_dates(folder)[:14],
                "named_files": _named_files(folder),
            }

        want = max(1, min(int(lines), 500))
        # Lọc làm mỏng kết quả rất nhiều, nên phải kéo dư rồi mới lọc — nếu
        # không, hỏi "20 dòng ERROR cuối" sẽ chỉ soi đúng 20 dòng cuối cùng và
        # thường trả về rỗng dù trong ngày có hàng trăm lỗi.
        raw = _tail_lines(path, want * 40 if (level or contains) else want)
        kept = [_clip(ln) for ln in raw if _matches(ln, level, contains)][-want:]

        return {
            "success": True,
            "category": category,
            "date": None if file else date,
            "file": path.name,
            "file_size_mb": round(path.stat().st_size / 1_048_576, 2),
            "filters": {"level": level, "contains": contains},
            "returned_lines": len(kept),
            "lines": kept,
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"read_log_tail lỗi: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── 3. Tìm kiếm xuyên nhiều ngày / nhiều nhóm ────────────────────────────────

class SearchLogsArgs(BaseModel):
    pattern: str = Field(description="Chuỗi hoặc regex cần tìm, vd 'timeout', 'Traceback', 'camera 40762191'")
    categories: Optional[List[str]] = Field(default=None, description="Giới hạn nhóm log; bỏ trống = các nhóm chính")
    days: int = Field(default=1, description="Số ngày gần nhất cần quét (tối đa 14)")
    level: Optional[str] = Field(default=None, description="Chỉ giữ dòng ở mức này")
    start_time: Optional[str] = Field(default=None, description="Chỉ giữ dòng từ giờ này ('HH:MM'), áp cho mọi ngày trong phạm vi")
    end_time: Optional[str] = Field(default=None, description="Chỉ giữ dòng đến giờ này ('HH:MM')")
    max_matches: int = Field(default=60, description="Số dòng khớp tối đa trả về (tối đa 200)")


def search_logs(
    pattern: str,
    categories: Optional[List[str]] = None,
    days: int = 1,
    level: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    max_matches: int = 60,
) -> Dict[str, Any]:
    """Tìm một chuỗi/regex trong log của nhiều ngày và nhiều nhóm."""
    try:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return {"success": False, "error": f"Regex không hợp lệ: {e}"}

        cats = categories or ["backend", "camera_management", "trigger_stats", "reject_actions"]
        bad = [c for c in cats if c not in CATEGORIES]
        if bad:
            return {"success": False, "error": f"Category không hợp lệ: {', '.join(bad)}"}

        n_days = max(1, min(int(days), 14))
        cap = max(1, min(int(max_matches), 200))
        today = datetime.now(_TZ)
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]

        matches: List[Dict[str, Any]] = []
        scanned: List[str] = []
        truncated = False

        for cat in cats:
            for path in _scan_targets(cat, dates):
                scanned.append(f"{cat}/{path.name}")
                for line in _iter_lines(path, MAX_SCAN_BYTES):
                    if not rx.search(line):
                        continue
                    if level and _level_of(line) != level.upper():
                        continue
                    if not _in_window(line, start_time, end_time):
                        continue
                    ts = _TS_RE.match(line)
                    matches.append({
                        "category": cat,
                        "file": path.name,
                        "time": ts.group(1)[11:] if ts else None,
                        "level": _level_of(line),
                        "line": _clip(line),
                    })
                    if len(matches) >= cap:
                        truncated = True
                        break
                if truncated:
                    break
            if truncated:
                break

        return {
            "success": True,
            "pattern": pattern,
            "scanned_files": scanned,
            "match_count": len(matches),
            "matches": matches,
            "truncated": truncated,
            "note": (
                f"Đã dừng ở {cap} dòng khớp đầu tiên — còn dòng khớp khác chưa liệt kê. "
                "Hãy thu hẹp `pattern`, `categories` hoặc `days` để thấy hết."
                if truncated else
                "Đã quét hết các file trong phạm vi yêu cầu."
            ),
        }
    except Exception as e:
        logger.error(f"search_logs lỗi: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── 4. Gom nhóm lỗi ──────────────────────────────────────────────────────────
#
# Đây là tool trả lời câu "vì sao": một sự cố hiếm khi sinh ra một dòng log, nó
# sinh ra hàng nghìn dòng gần giống nhau chỉ khác ID/số/đường dẫn. Liệt kê thô
# thì tràn màn hình mà vẫn không thấy quy luật. Chuẩn hoá từng dòng thành một
# "chữ ký" rồi đếm sẽ lộ ra ngay: 3 nhóm lỗi khác nhau, nhóm nào lặp nhiều nhất,
# bắt đầu và kết thúc lúc mấy giờ.

_NORMALIZERS = [
    (re.compile(r"\b[0-9a-fA-F]{24}\b"), "<id>"),           # ObjectId của Mongo
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"),
    (re.compile(r"/[\w./-]{6,}"), "<path>"),
    (re.compile(r"\b\d+\.\d+\b"), "<num>"),
    (re.compile(r"\b\d+\b"), "<num>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<hex>"),
]


def _signature(line: str) -> str:
    """Bỏ timestamp + phần biến thiên để các dòng cùng một lỗi gộp về một chữ ký."""
    body = line.rstrip("\n")
    m = _TS_RE.match(body)
    if m:
        body = body[m.end():].lstrip(" -|")
    for rx, repl in _NORMALIZERS:
        body = rx.sub(repl, body)
    return body.strip()[:220]


class SummarizeLogErrorsArgs(BaseModel):
    category: Optional[str] = Field(default=None, description="Nhóm log; bỏ trống = quét các nhóm chính")
    date: Optional[str] = Field(default=None, description="Ngày YYYY-MM-DD, bỏ trống = hôm nay")
    start_time: Optional[str] = Field(default=None, description="Chỉ xét từ giờ này, dạng 'HH:MM' hoặc 'HH'. VD user hỏi 'lúc 5 giờ' → start_time='05:00'")
    end_time: Optional[str] = Field(default=None, description="Chỉ xét đến giờ này, dạng 'HH:MM' hoặc 'HH'. VD 'lúc 5 giờ' → end_time='05:59'")
    top: int = Field(default=12, description="Số nhóm lỗi hàng đầu cần trả về (tối đa 30)")


def summarize_log_errors(
    category: Optional[str] = None,
    date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    top: int = 12,
) -> Dict[str, Any]:
    """Gom ERROR/WARNING/CRITICAL thành các nhóm giống nhau và đếm số lần."""
    try:
        cats = [category] if category else ["backend", "camera_management", "start_services"]
        bad = [c for c in cats if c not in CATEGORIES]
        if bad:
            return {"success": False, "error": f"Category không hợp lệ: {', '.join(bad)}"}

        date = date or _today()
        if not _DATE_RE.match(date):
            return {"success": False, "error": "Ngày phải có dạng YYYY-MM-DD"}

        limit = max(1, min(int(top), 30))
        groups: Dict[str, Dict[str, Any]] = {}
        by_level: Counter = Counter()
        by_logger: Counter = Counter()
        total = 0
        files: List[str] = []

        skipped: List[str] = []
        for cat in cats:
            targets = _scan_targets(cat, [date])
            if not targets:
                skipped.append(cat)
                continue
            for path in targets:
                files.append(f"{cat}/{path.name}")
                for line in _iter_lines(path, MAX_SCAN_BYTES):
                    lv = _level_of(line)
                    if lv not in ("ERROR", "CRITICAL", "WARNING"):
                        continue
                    if not _in_window(line, start_time, end_time):
                        continue
                    total += 1
                    by_level[lv] += 1

                    parts = line.split(" - ", 3)
                    if len(parts) >= 3:
                        by_logger[parts[1].strip()] += 1

                    sig = _signature(line)
                    ts = _TS_RE.match(line)
                    clock = ts.group(1)[11:] if ts else None
                    g = groups.get(sig)
                    if g is None:
                        groups[sig] = {
                            "signature": sig, "level": lv, "category": cat,
                            "count": 1, "first_seen": clock, "last_seen": clock,
                            "example": _clip(line),
                        }
                    else:
                        g["count"] += 1
                        if clock:
                            g["last_seen"] = clock
                            if not g["first_seen"]:
                                g["first_seen"] = clock

        if not files:
            return {
                "success": False,
                "error": f"Không có file log nào cho ngày {date} trong: {', '.join(cats)}",
            }

        ranked = sorted(groups.values(), key=lambda g: -g["count"])[:limit]

        return {
            "success": True,
            "date": date,
            "time_window": (f"{_pad_time(start_time) if start_time else '00:00:00'}"
                            f" → {_pad_time(end_time, end=True) if end_time else '23:59:59'}")
                           if (start_time or end_time) else "cả ngày",
            "categories": cats,
            "scanned_files": files,
            "skipped_categories": skipped,
            "total_problem_lines": total,
            "by_level": dict(by_level),
            "top_loggers": dict(by_logger.most_common(8)),
            "distinct_problems": len(groups),
            "problems": ranked,
            "note": (
                f"{total} dòng ERROR/WARNING/CRITICAL gom thành {len(groups)} vấn đề khác nhau. "
                "`signature` là dòng đã chuẩn hoá — số, ID, đường dẫn bị thay bằng <num>/<id>/<path> "
                "để các dòng cùng một lỗi đếm chung. Hãy giải thích theo `signature` + `count`, "
                "và dùng `first_seen`/`last_seen` để nói lỗi xảy ra liên tục hay chỉ dồn vào một lúc."
            ),
        }
    except Exception as e:
        logger.error(f"summarize_log_errors lỗi: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── 5. Audit log (MongoDB) ───────────────────────────────────────────────────

class GetAuditLogsArgs(BaseModel):
    username: Optional[str] = Field(default=None, description="Lọc theo tên đăng nhập")
    action_type: Optional[str] = Field(
        default=None,
        description="Loại thao tác: login, logout, create_recipe, update_recipe, delete_recipe, "
                    "load_recipe, stop_recipe, create_user, update_user, delete_user, "
                    "reset_user_password, create_camera, update_camera, delete_camera",
    )
    resource_type: Optional[str] = Field(default=None, description="Loại đối tượng: user, recipe, camera, auth")
    resource: Optional[str] = Field(
        default=None,
        description="Lọc theo ĐỐI TƯỢNG bị tác động — tên recipe (vd 'ONION POWDER'), "
                    "tên camera, hoặc ObjectId 24 ký tự. Dùng ô này khi user hỏi "
                    "'ai load recipe X', TUYỆT ĐỐI không nhét tên recipe vào `username`.",
    )
    start_date: Optional[str] = Field(default=None, description="Từ ngày YYYY-MM-DD (giờ địa phương)")
    end_date: Optional[str] = Field(default=None, description="Đến ngày YYYY-MM-DD (giờ địa phương)")
    limit: int = Field(default=40, description="Số bản ghi tối đa (tối đa 200)")


def get_audit_logs(
    username: Optional[str] = None,
    action_type: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 40,
) -> Dict[str, Any]:
    """Lịch sử thao tác của người dùng: ai làm gì, lúc nào, trên đối tượng nào."""
    try:
        db = get_sync_database()
        query: Dict[str, Any] = {
            "timestamp": {"$gte": _local_bound(start_date, end=False),
                          "$lte": _local_bound(end_date, end=True)},
        }
        if username:
            # Tên recipe bị nhét vào ô `username` là lỗi đã xảy ra thật: LLM muốn
            # lọc theo recipe, thấy `username` là ô chữ tự do gần nghĩa nhất nên
            # truyền 'ONION POWDER' vào đây. Khớp 0 bản ghi, và câu trả lời thành
            # "hôm nay không ai load recipe" — sai hẳn, trong khi có 5 lần load.
            # Nên phải chặn thay vì trả về rỗng một cách tự tin.
            known = db["action_logs"].distinct("username")
            if username not in known:
                return {
                    "success": False,
                    "error": (f"Không có người dùng nào tên '{username}' trong audit log. "
                              f"Nếu đây là TÊN RECIPE hoặc tên camera thì truyền vào tham "
                              f"số `resource`, không phải `username`."),
                    "known_usernames": known,
                }
            query["username"] = username
        if action_type:
            query["action_type"] = action_type
        if resource_type:
            query["resource_type"] = resource_type
        if resource:
            # ObjectId thì khớp thẳng `resource_id`; còn lại tìm trong
            # `description` — nơi duy nhất lưu TÊN đối tượng
            # ("Loaded recipe 'ONION POWDER' with product code ...").
            if re.fullmatch(r"[0-9a-fA-F]{24}", resource.strip()):
                query["resource_id"] = resource.strip()
            else:
                query["description"] = {"$regex": re.escape(resource.strip()),
                                        "$options": "i"}

        cap = max(1, min(int(limit), 200))
        total = db["action_logs"].count_documents(query)
        cursor = db["action_logs"].find(
            query,
            # `old_value`/`new_value` là bản chụp trọn recipe — có recipe kèm cả
            # toạ độ annotation, đủ sức thổi một ToolMessage lên hàng chục KB.
            # Phần trả lời chỉ cần biết ai đổi cái gì lúc nào.
            {"old_value": 0, "new_value": 0, "user_agent": 0},
        ).sort("timestamp", -1).limit(cap)

        entries = [{
            "time": _to_local_str(d["timestamp"]),
            "username": d.get("username"),
            "action_type": d.get("action_type"),
            "resource_type": d.get("resource_type"),
            "resource_id": d.get("resource_id"),
            "description": d.get("description"),
            "ip_address": d.get("ip_address"),
        } for d in cursor]

        actions = Counter(e["action_type"] for e in entries)
        users = Counter(e["username"] for e in entries)

        # Hồ sơ người thao tác, nối từ collection `users`.
        #
        # Audit log chỉ lưu `username` — một chuỗi. Muốn hiện thẻ có tên đầy đủ,
        # chức vụ và ảnh thì phải tra sang `users`; tra ở đây (một truy vấn cho
        # cả danh sách) chứ không để LLM tự đoán, vì nó sẽ bịa ra chức vụ.
        people: List[Dict[str, Any]] = []
        if entries:
            names = list(users)
            profiles = {
                u["username"]: u
                for u in db["users"].find(
                    {"username": {"$in": names}},
                    # Ngoài các field của backend còn có nhóm hồ sơ nhân sự
                    # (employee_code, department, job_title, shift, hire_date,
                    # production_line) — xem ghi chú ở `PROFILE_FIELDS` bên dưới.
                    {"_id": 0, "password": 0, "hashed_password": 0},
                )
            }
            for name, count in users.most_common():
                mine = [e for e in entries if e["username"] == name]
                prof = profiles.get(name) or {}
                people.append({
                    "username": name,
                    # Tài khoản đã bị xoá vẫn còn dấu vết trong audit log — đó
                    # chính là điểm của audit log — nên phải chịu được việc
                    # không tra ra hồ sơ, thay vì bỏ người đó khỏi thẻ.
                    "full_name": prof.get("full_name") or f"(tài khoản đã xoá: {name})",
                    "role": prof.get("role") or "unknown",
                    "avatar_url": prof.get("avatar_url"),
                    "email": prof.get("email"),
                    "phone_number": prof.get("phone_number"),
                    "is_active": prof.get("is_active"),
                    **{f: prof.get(f) for f in PROFILE_FIELDS},
                    "account_exists": bool(prof),
                    "action_count": count,
                    "actions": dict(Counter(e["action_type"] for e in mine).most_common()),
                    "first_seen": mine[-1]["time"],
                    "last_seen": mine[0]["time"],
                })

        return {
            "success": True,
            "filters": {
                "username": username or "all", "action_type": action_type or "all",
                "resource_type": resource_type or "all", "resource": resource or "all",
                "start": _to_local_str(query["timestamp"]["$gte"]),
                "end": _to_local_str(query["timestamp"]["$lte"]),
            },
            "total_matching": total,
            "returned": len(entries),
            "by_action": dict(actions.most_common()),
            "by_user": dict(users.most_common()),
            "people": people,
            "entries": entries,
            "note": (
                f"Khớp {total} bản ghi, trả về {len(entries)} bản mới nhất."
                + (" Tăng `limit` hoặc thu hẹp khoảng ngày để xem thêm."
                   if total > len(entries) else "")
            ),
        }
    except Exception as e:
        logger.error(f"get_audit_logs lỗi: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── Đăng ký tool ─────────────────────────────────────────────────────────────

list_log_sources_tool = BaseTool.create_tool(
    func=list_log_sources,
    metadata=ToolMetadata(
        name="list_log_sources",
        description=(
            "Liệt kê các nhóm log đang có kèm dung lượng và những ngày có dữ liệu. "
            "Gọi tool này TRƯỚC khi đọc log nếu chưa chắc nhóm nào hoặc ngày nào có dữ liệu."
        ),
        category="logs",
    ),
    args_schema=ListLogSourcesArgs,
)

read_log_tail_tool = BaseTool.create_tool(
    func=read_log_tail,
    metadata=ToolMetadata(
        name="read_log_tail",
        description=(
            "Đọc các dòng CUỐI của một file log, lọc được theo mức (ERROR/WARNING/…) "
            "và theo từ khoá. Dùng khi user muốn 'xem log gần đây', 'log mới nhất', "
            "'có lỗi gì vừa xảy ra'. Chỉ trả về tối đa 500 dòng vì file log có thể "
            "lên tới hàng trăm MB."
        ),
        category="logs",
    ),
    args_schema=ReadLogTailArgs,
)

search_logs_tool = BaseTool.create_tool(
    func=search_logs,
    metadata=ToolMetadata(
        name="search_logs",
        description=(
            "Tìm một chuỗi hoặc regex xuyên nhiều nhóm log và nhiều ngày. Dùng khi user "
            "hỏi về một hiện tượng cụ thể: 'có timeout không', 'tìm Traceback', "
            "'camera 40762191 có lỗi gì'. Kết quả bị chặn ở 200 dòng khớp — nếu "
            "`truncated` là true thì phải nói rõ cho user biết là chưa liệt kê hết."
        ),
        category="logs",
    ),
    args_schema=SearchLogsArgs,
)

summarize_log_errors_tool = BaseTool.create_tool(
    func=summarize_log_errors,
    metadata=ToolMetadata(
        name="summarize_log_errors",
        description=(
            "Gom toàn bộ ERROR/WARNING/CRITICAL trong ngày thành các nhóm vấn đề "
            "giống nhau và đếm số lần lặp. ĐÂY LÀ TOOL CHÍNH để trả lời 'hôm nay có "
            "lỗi gì', 'tại sao hệ thống chậm', 'có vấn đề gì không' — nó cho thấy "
            "quy luật thay vì một đống dòng log rời rạc. Hãy giải thích nguyên nhân "
            "dựa trên `signature` + `count` + `first_seen`/`last_seen`. "
            "User hỏi về MỘT KHUNG GIỜ ('lúc 5 giờ sáng có chuyện gì', 'khoảng 10h "
            "máy khựng') thì BẮT BUỘC truyền `start_time`/`end_time` — không truyền "
            "thì tool trả về cả ngày và câu trả lời sẽ lạc đề."
        ),
        category="logs",
    ),
    args_schema=SummarizeLogErrorsArgs,
)

get_audit_logs_tool = BaseTool.create_tool(
    func=get_audit_logs,
    metadata=ToolMetadata(
        name="get_audit_logs",
        description=(
            "Lịch sử thao tác của người dùng lấy từ MongoDB: đăng nhập, tạo/sửa/xoá "
            "recipe, load recipe, đổi camera, quản lý user. Dùng khi user hỏi 'ai đã "
            "làm gì', 'ai đổi recipe', 'ai đăng nhập lúc nào', 'lịch sử load recipe X'. "
            "Đây là nguồn KHÁC với log file — log file ghi hoạt động của máy, còn đây "
            "ghi hành động của con người. "
            "QUAN TRỌNG: `username` là NGƯỜI thao tác (vd 'admin'). Tên recipe hoặc tên "
            "camera phải truyền vào `resource`. Nhét tên recipe vào `username` sẽ khớp "
            "0 bản ghi và cho ra câu trả lời sai là 'không có ai làm gì'."
        ),
        category="logs",
    ),
    args_schema=GetAuditLogsArgs,
)

logger.info("✅ Log tools registered")


# ── 6. Quản lý dung lượng log ────────────────────────────────────────────────
#
# Phần "quản lý" của agent, và cố ý CHỈ ĐỌC.
#
# Backend có sẵn endpoint xoá log (`DELETE /{category}/{date}`, `DELETE
# /{category}`) và endpoint sửa chính sách dọn dẹp, nhưng chúng không được bọc
# thành tool. Xoá log là thao tác không hoàn tác được, và log chính là thứ để
# điều tra khi có sự cố — một câu chat hiểu nhầm là mất bằng chứng vĩnh viễn.
# Tool này chỉ trình bày hiện trạng và chỉ ra file nào đang chiếm chỗ; người
# vận hành tự bấm xoá trên tab System Logs.

_CLEANUP_CONFIG = LOGS_ROOT / "_meta" / "cleanup_config.json"


class LogStorageReportArgs(BaseModel):
    pass


def get_log_storage_report() -> Dict[str, Any]:
    """Dung lượng log đang chiếm, chính sách dọn dẹp, và file nào chính sách bỏ sót."""
    try:
        import json
        import shutil

        cfg: Dict[str, Any] = {}
        if _CLEANUP_CONFIG.is_file():
            try:
                cfg = json.loads(_CLEANUP_CONFIG.read_text())
            except Exception as e:
                cfg = {"error": f"Không đọc được cleanup_config.json: {e}"}

        keep_days = cfg.get("keep_days")
        cutoff = None
        if isinstance(keep_days, int) and keep_days > 0:
            cutoff = (datetime.now(_TZ) - timedelta(days=keep_days)).strftime("%Y-%m-%d")

        per_cat: List[Dict[str, Any]] = []
        unmanaged: List[Dict[str, Any]] = []
        expired: List[Dict[str, Any]] = []
        total = 0

        for cat in CATEGORIES:
            folder = LOGS_ROOT / cat
            if not folder.is_dir():
                continue
            size = 0
            for p in folder.iterdir():
                if not p.is_file():
                    continue
                st = p.stat()
                size += st.st_size
                stem = p.name.split(".log")[0]

                if not _DATE_RE.match(stem):
                    # Scheduler dọn theo tên '{YYYY-MM-DD}.log'; file đặt theo
                    # tên service không khớp mẫu đó nên nằm ngoài chính sách và
                    # phình vô hạn. Đây thường là nguyên nhân thật khi đĩa đầy
                    # dù retention đã bật.
                    if st.st_size > 1_048_576:
                        unmanaged.append({
                            "path": f"{cat}/{p.name}",
                            "size_mb": round(st.st_size / 1_048_576, 1),
                            "modified": datetime.fromtimestamp(st.st_mtime, _TZ).strftime("%Y-%m-%d %H:%M"),
                        })
                elif cutoff and stem < cutoff:
                    expired.append({
                        "path": f"{cat}/{p.name}",
                        "date": stem,
                        "size_mb": round(st.st_size / 1_048_576, 1),
                    })
            total += size
            per_cat.append({"category": cat, "size_mb": round(size / 1_048_576, 1)})

        # File .log nằm thẳng trong logs/ (không thuộc category nào)
        loose = []
        for p in LOGS_ROOT.iterdir():
            if p.is_file() and p.name.endswith(".log") and p.stat().st_size > 1_048_576:
                st = p.stat()
                loose.append({
                    "path": p.name,
                    "size_mb": round(st.st_size / 1_048_576, 1),
                    "modified": datetime.fromtimestamp(st.st_mtime, _TZ).strftime("%Y-%m-%d %H:%M"),
                })
                total += st.st_size

        usage = shutil.disk_usage(str(LOGS_ROOT))

        per_cat.sort(key=lambda x: -x["size_mb"])
        unmanaged.sort(key=lambda x: -x["size_mb"])
        loose.sort(key=lambda x: -x["size_mb"])

        return {
            "success": True,
            "total_log_size_mb": round(total / 1_048_576, 1),
            "disk_free_gb": round(usage.free / 1_073_741_824, 1),
            "disk_used_percent": round(usage.used * 100 / usage.total, 1),
            "cleanup_config": cfg,
            "size_by_category": per_cat,
            "outside_cleanup_policy": unmanaged + loose,
            "past_retention": sorted(expired, key=lambda x: x["date"])[:20],
            "note": (
                "Bộ dọn dẹp chỉ xử lý file đặt tên '{YYYY-MM-DD}.log'. Mọi file trong "
                "`outside_cleanup_policy` nằm NGOÀI chính sách đó — chúng không bao giờ "
                "bị xoá hay nén dù `keep_days` là bao nhiêu, nên thường chính chúng mới "
                "là thứ làm đầy đĩa. `past_retention` là file đã quá hạn mà vẫn còn, "
                "dấu hiệu bộ dọn dẹp chưa chạy. "
                "Tool này CHỈ ĐỌC — agent không xoá được log. Muốn xoá thì thao tác "
                "trên tab System Logs."
            ),
        }
    except Exception as e:
        logger.error(f"get_log_storage_report lỗi: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


get_log_storage_report_tool = BaseTool.create_tool(
    func=get_log_storage_report,
    metadata=ToolMetadata(
        name="get_log_storage_report",
        description=(
            "Dung lượng log đang chiếm, dung lượng đĩa còn trống, chính sách dọn dẹp "
            "đang cấu hình, và quan trọng nhất là những file nằm NGOÀI chính sách đó. "
            "Dùng khi user hỏi 'log chiếm bao nhiêu', 'đĩa sắp đầy', 'sao log không "
            "tự xoá', 'quản lý log'. Chỉ đọc — tool này không xoá được gì."
        ),
        category="logs",
    ),
    args_schema=LogStorageReportArgs,
)
