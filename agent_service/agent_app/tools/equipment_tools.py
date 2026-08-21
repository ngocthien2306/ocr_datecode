"""
Tools đo SỨC KHOẺ THIẾT BỊ.

Khác hai nhóm tool đã có: `analytics_tools` đếm sản phẩm, `log_tools` đọc chữ
trong log. Nhóm này đọc các log có CẤU TRÚC mà chưa ai khai thác, và so chúng với
cấu hình trong MongoDB.

Vì sao tách riêng: nhóm sự cố ở đây xảy ra TRƯỚC khi sản lượng tụt. Xung reject
sai, cảm biến trôi, một hệ thống con không init được — đều là thứ chỉ hiện ra
trong log kỹ thuật, và tới lúc thấy trên biểu đồ pass/fail thì đã muộn.

Bốn nguồn log dùng ở đây có định dạng riêng, không phải logging chuẩn của Python:

    reject_actions  2026-08-19 17:48:29.486 | REJECT_START | #0082 | Group #3750 |
                    Reject(DO3) + Alarm(DO0) | Scheduled: ... | Delay: 1500.0ms | ...
                    2026-08-19 17:48:30.483 | REJECT_END | ... |
                    Pulse: 50.0ms | Actual: 255.5ms | Diff: +205.5ms
    trigger_stats   ... - STATS - Triggers: 3852 | Groups: 3852 created, 3852
                    completed, 0 timeout | Success: 100.0% | Inferences: 3852 |
                    CaptureFailures: 2 | Active: ...
    pulse_width     2026-08-19 18:00:19.419 | DI0 | pulse_width=844.198ms | normal=100000ms
    obb_rotation    2026-08-19 10:09:49  ERROR  INIT FAILED — engine load error: ...
"""

import logging
import re
import statistics
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent_app.db.mongodb import get_sync_database
from agent_app.tools.base_tool import BaseTool, ToolMetadata
from agent_app.tools.log_tools import (
    LOGS_ROOT,
    MAX_SCAN_BYTES,
    _iter_lines,
    _resolve_file,
    _today,
)

logger = logging.getLogger(__name__)

_TS = r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})[.,]?(\d*)"

_REJECT_END = re.compile(
    _TS + r".*REJECT_END.*?Pulse:\s*([\d.]+)ms\s*\|\s*Actual:\s*([\d.]+)ms"
)
_REJECT_START = re.compile(_TS + r".*REJECT_START.*?Delay:\s*([\d.]+)ms")
_STATS = re.compile(
    _TS + r" - STATS - Triggers: (\d+) \| Groups: (\d+) created, (\d+) completed, "
    r"(\d+) timeout \| Success: ([\d.]+)% \| Inferences: (\d+) \| CaptureFailures: (\d+)"
)
_PULSE = re.compile(_TS + r"\s*\|\s*(DI\d+)\s*\|\s*pulse_width=([\d.]+)ms\s*\|\s*normal=([\d.]+)ms")
_INIT_FAIL = re.compile(_TS + r"\s+(ERROR|CRITICAL)\s+(.*)", re.IGNORECASE)


def _clock(m: "re.Match") -> str:
    return m.group(1)[11:19]


def _spread(values: List[float]) -> Dict[str, Any]:
    """Thống kê gọn của một dãy số, dùng trung vị chứ không phải trung bình.

    Trung bình bị một lần đo lệch kéo đi; trung vị nói đúng "thường thì bao nhiêu",
    và đó mới là con số so được với cấu hình."""
    if not values:
        return {}
    vs = sorted(values)
    return {
        "n": len(vs),
        "min": round(vs[0], 1),
        "median": round(statistics.median(vs), 1),
        "max": round(vs[-1], 1),
        "p95": round(vs[min(len(vs) - 1, int(len(vs) * 0.95))], 1),
    }


# ── 1. Xung reject: cấu hình vs thực tế ──────────────────────────────────────

class RejectTimingArgs(BaseModel):
    date: Optional[str] = Field(default=None, description="Ngày YYYY-MM-DD, bỏ trống = hôm nay")


def check_reject_timing(date: Optional[str] = None, **_ignored: Any) -> Dict[str, Any]:
    """So độ rộng xung reject thực tế với giá trị cấu hình trong recipe."""
    try:
        date = date or _today()
        path = _resolve_file(LOGS_ROOT / "reject_actions", date)
        if path is None:
            return {"success": False, "error": f"Không có log reject_actions ngày {date}"}

        actual, configured_in_log, delays, times = [], [], [], []
        for line in _iter_lines(path, MAX_SCAN_BYTES):
            m = _REJECT_END.match(line)
            if m:
                configured_in_log.append(float(m.group(3)))
                actual.append(float(m.group(4)))
                times.append(_clock(m))
                continue
            m = _REJECT_START.match(line)
            if m:
                delays.append(float(m.group(3)))

        if not actual:
            return {"success": False, "error": f"Không có lần reject nào trong ngày {date}"}

        # Cấu hình lấy từ recipe, không suy từ log: log ghi giá trị đang áp dụng,
        # còn recipe là ý định của người cài đặt — muốn thấy lệch thì phải so hai
        # nguồn đó với nhau.
        db = get_sync_database()
        recipes = [
            {"recipe_name": r.get("name"), "reject_pulse": r.get("reject_pulse"),
             "delay_reject": r.get("delay_reject"), "reject_method": r.get("reject_method")}
            for r in db["recipes"].find({}, {"name": 1, "reject_pulse": 1,
                                             "delay_reject": 1, "reject_method": 1})
        ]

        a = _spread(actual)
        cfg = _spread(configured_in_log)
        cfg_val = cfg.get("median")
        over = None
        if cfg_val:
            over = round(a["median"] / cfg_val, 2)

        # Đếm số lần vượt quá cấu hình, để nói được "bao nhiêu trên bao nhiêu"
        # thay vì chỉ đưa một con số trung vị.
        exceeded = sum(1 for c, x in zip(configured_in_log, actual) if x > c * 1.2)

        return {
            "success": True,
            "date": date,
            "reject_count": len(actual),
            "configured_pulse_ms": cfg,
            "actual_pulse_ms": a,
            "delay_ms": _spread(delays),
            "over_config_ratio": over,
            "exceeded_count": exceeded,
            "first_reject": times[0] if times else None,
            "last_reject": times[-1] if times else None,
            "recipe_config": recipes,
            "note": (
                f"`configured_pulse_ms` là giá trị log ghi là đang áp dụng, "
                f"`actual_pulse_ms` là độ rộng xung ĐO ĐƯỢC. Lệch nhiều nghĩa là cơ cấu "
                f"đẩy phôi giữ điện lâu hơn ý định cài đặt. "
                f"KHÔNG kết luận đây là lỗi — trên máy nhúng luôn có một mức trễ nền do "
                f"hệ điều hành và GPIO, và mức đó có thể lớn hơn xung 50ms. Việc của báo "
                f"cáo này là NÊU RA chênh lệch để thợ cơ khí xác nhận hệ quả, không phải "
                f"tự phán. Log reject KHÔNG ghi mỗi lần thuộc recipe nào, nên "
                f"`recipe_config` liệt kê cấu hình của tất cả recipe để đối chiếu."
            ),
        }
    except Exception as e:
        logger.error(f"check_reject_timing lỗi: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── 2. Độ tin cậy trigger ────────────────────────────────────────────────────

class TriggerHealthArgs(BaseModel):
    date: Optional[str] = Field(default=None, description="Ngày YYYY-MM-DD, bỏ trống = hôm nay")


def check_trigger_health(date: Optional[str] = None, **_ignored: Any) -> Dict[str, Any]:
    """Trigger, group, timeout và lỗi chụp ảnh trong ngày."""
    try:
        date = date or _today()
        path = _resolve_file(LOGS_ROOT / "trigger_stats", date)
        if path is None:
            return {"success": False, "error": f"Không có log trigger_stats ngày {date}"}

        rows = []
        for line in _iter_lines(path, MAX_SCAN_BYTES):
            m = _STATS.match(line)
            if m:
                rows.append({
                    "clock": _clock(m),
                    "triggers": int(m.group(3)),
                    "created": int(m.group(4)),
                    "completed": int(m.group(5)),
                    "timeout": int(m.group(6)),
                    "success": float(m.group(7)),
                    "inferences": int(m.group(8)),
                    "capture_failures": int(m.group(9)),
                })
        if not rows:
            return {"success": False, "error": f"Không đọc được dòng STATS nào ngày {date}"}

        # Bộ đếm là CỘNG DỒN TỪ LÚC SERVICE KHỞI ĐỘNG và reset khi restart.
        #
        # Lấy giá trị cuối trừ giá trị đầu là sai: hôm nay bộ đếm reset 2 lần
        # (10:09 và 10:24, đúng lúc service restart), nên phép trừ thẳng cho ra số
        # âm. Phải cộng từng đoạn, và mỗi lần thấy giá trị TỤT thì coi như bắt đầu
        # đoạn mới.
        fields = ("triggers", "created", "completed", "timeout", "inferences", "capture_failures")
        totals = {f: 0 for f in fields}
        restarts: List[str] = []
        prev = None
        for r in rows:
            if prev is None:
                for f in fields:
                    totals[f] += r[f]
            elif r["triggers"] < prev["triggers"]:
                restarts.append(r["clock"])
                for f in fields:
                    totals[f] += r[f]          # đoạn mới, cộng từ đầu
            else:
                for f in fields:
                    totals[f] += max(0, r[f] - prev[f])
            prev = r

        trig = totals["triggers"]
        return {
            "success": True,
            "date": date,
            "window": f"{rows[0]['clock']} → {rows[-1]['clock']}",
            "service_restarts": len(restarts),
            "restart_times": restarts,
            "triggers": trig,
            "groups_created": totals["created"],
            "groups_completed": totals["completed"],
            "groups_timeout": totals["timeout"],
            "inferences": totals["inferences"],
            "capture_failures": totals["capture_failures"],
            "timeout_rate": round(totals["timeout"] / trig * 100, 3) if trig else None,
            "capture_failure_rate": round(totals["capture_failures"] / trig * 100, 3) if trig else None,
            "latest_success_percent": rows[-1]["success"],
            "note": (
                "Bộ đếm cộng dồn từ lúc service khởi động và reset khi restart, nên các "
                "con số ở đây đã được cộng theo từng đoạn giữa các lần restart. "
                "`groups_timeout` là nhóm trigger không hoàn tất kịp — mỗi cái là một sản "
                "phẩm KHÔNG được kiểm. `capture_failures` là lần chụp ảnh thất bại, cũng "
                "là sản phẩm đi qua mà không có dữ liệu. Hai con số này quan trọng hơn "
                "`latest_success_percent`, vì tỷ lệ đó chỉ tính trên nhóm đã hoàn tất."
            ),
        }
    except Exception as e:
        logger.error(f"check_trigger_health lỗi: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── 3. Cảm biến đầu vào ──────────────────────────────────────────────────────

class SensorPulseArgs(BaseModel):
    date: Optional[str] = Field(default=None, description="Ngày YYYY-MM-DD, bỏ trống = hôm nay")


def check_sensor_pulse(date: Optional[str] = None, **_ignored: Any) -> Dict[str, Any]:
    """Độ rộng xung cảm biến đầu vào, và độ trôi trong ngày."""
    try:
        date = date or _today()
        path = _resolve_file(LOGS_ROOT / "pulse_width", date)
        if path is None:
            return {"success": False, "error": f"Không có log pulse_width ngày {date}"}

        by_ch: Dict[str, List[float]] = {}
        hourly: Dict[str, Dict[int, List[float]]] = {}
        normal: Dict[str, float] = {}
        for line in _iter_lines(path, MAX_SCAN_BYTES):
            m = _PULSE.match(line)
            if not m:
                continue
            ch, val = m.group(3), float(m.group(4))
            by_ch.setdefault(ch, []).append(val)
            normal[ch] = float(m.group(5))
            hour = int(m.group(1)[11:13])
            hourly.setdefault(ch, {}).setdefault(hour, []).append(val)

        if not by_ch:
            return {"success": False, "error": f"Không đọc được dòng pulse nào ngày {date}"}

        out = []
        for ch, vals in sorted(by_ch.items()):
            sp = _spread(vals)
            per_hour = {h: round(statistics.median(v), 1)
                        for h, v in sorted(hourly[ch].items())}
            meds = list(per_hour.values())
            out.append({
                "channel": ch,
                "pulse_ms": sp,
                "configured_normal_ms": normal.get(ch),
                "median_by_hour": per_hour,
                # Độ trôi = biên độ dao động của trung vị theo giờ. Xung cảm biến
                # phản ánh khoảng cách và tốc độ sản phẩm; trôi dần là dấu hiệu
                # băng tải đổi tốc độ hoặc cảm biến bắt đầu kém.
                "drift_ms": round(max(meds) - min(meds), 1) if len(meds) > 1 else 0.0,
                "spread_ms": round(sp["max"] - sp["min"], 1),
            })

        return {
            "success": True,
            "date": date,
            "channels": out,
            "note": (
                "`pulse_ms` là độ rộng xung cảm biến đo được — nó phản ánh khoảng cách "
                "và tốc độ sản phẩm đi qua, không phải cấu hình reject. "
                "`drift_ms` là chênh lệch giữa giờ có trung vị cao nhất và thấp nhất: trôi "
                "lớn nghĩa là nhịp dây chuyền thay đổi trong ngày. "
                "`configured_normal_ms` là ngưỡng 'bình thường' đang cài; nếu nó lớn hơn "
                "xung thực tế nhiều lần thì ngưỡng đó thực tế không có tác dụng chặn gì. "
                "Đây là số liệu để so theo NGÀY: một ngày không nói được gì về xu hướng."
            ),
        }
    except Exception as e:
        logger.error(f"check_sensor_pulse lỗi: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── 4. Hệ thống con có init được không ───────────────────────────────────────

#: Các nhóm log ghi lỗi khởi tạo của hệ thống con.
_INIT_CATEGORIES = ("obb_rotation", "camera_settings", "camera_management",
                    "backend", "camera_check")


class SubsystemHealthArgs(BaseModel):
    date: Optional[str] = Field(default=None, description="Ngày YYYY-MM-DD, bỏ trống = hôm nay")


def check_subsystem_health(date: Optional[str] = None, **_ignored: Any) -> Dict[str, Any]:
    """Các hệ thống con báo lỗi khởi tạo — thứ nằm im mà không ai hay."""
    try:
        date = date or _today()
        problems: List[Dict[str, Any]] = []
        scanned, missing = [], []

        for cat in _INIT_CATEGORIES:
            path = _resolve_file(LOGS_ROOT / cat, date)
            if path is None:
                missing.append(cat)
                continue
            scanned.append(cat)
            seen: Dict[str, Dict[str, Any]] = {}
            for line in _iter_lines(path, MAX_SCAN_BYTES):
                m = _INIT_FAIL.match(line)
                if not m:
                    continue
                msg = m.group(4).strip()
                # Chỉ giữ lỗi KHỞI TẠO / nạp model / thiếu file. Lỗi vận hành
                # thường ngày đã có `summarize_log_errors` lo; ở đây tìm thứ làm
                # một hệ thống con không chạy được ngay từ đầu.
                if not re.search(r"init|load|not found|no such file|failed to (start|open|create)"
                                 r"|khong tim thay|engine", msg, re.IGNORECASE):
                    continue
                # Gộp theo thông điệp đã bỏ đường dẫn, để một lỗi lặp 50 lần
                # không chiếm hết báo cáo.
                key = re.sub(r"/[\w./-]+", "<path>", msg)[:160]
                p = seen.setdefault(key, {
                    "category": cat, "level": m.group(3).upper(), "message": key,
                    "count": 0, "first_seen": _clock(m), "last_seen": _clock(m),
                    "example": msg[:220],
                })
                p["count"] += 1
                p["last_seen"] = _clock(m)
            problems.extend(seen.values())

        problems.sort(key=lambda p: -p["count"])
        return {
            "success": True,
            "date": date,
            "scanned_categories": scanned,
            "no_log_for_date": missing,
            "problem_count": len(problems),
            "problems": problems[:20],
            "note": (
                "Đây là lỗi KHỞI TẠO: một hệ thống con không nạp được model, không mở "
                "được file, không start được. Khác lỗi vận hành thường ngày — hệ thống con "
                "kiểu này nằm im hoàn toàn mà dây chuyền vẫn chạy, nên không ai phát hiện "
                "cho tới khi cần đúng chức năng đó. "
                "`no_log_for_date` là các nhóm không có log ngày đó — có thể do hệ thống "
                "con không chạy, KHÔNG có nghĩa là nó khoẻ."
            ),
        }
    except Exception as e:
        logger.error(f"check_subsystem_health lỗi: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── Đăng ký tool ─────────────────────────────────────────────────────────────

check_reject_timing_tool = BaseTool.create_tool(
    func=check_reject_timing,
    metadata=ToolMetadata(
        name="check_reject_timing",
        description=(
            "So độ rộng xung reject ĐO ĐƯỢC với giá trị cấu hình trong recipe. Dùng khi "
            "user hỏi 'cơ cấu đẩy phôi có đúng không', 'reject có chính xác không', "
            "'xung reject bao nhiêu', 'thời gian đẩy phôi'. Lệch nhiều thì NÊU RA để thợ "
            "cơ khí xác nhận, đừng tự kết luận là lỗi."
        ),
        category="equipment",
    ),
    args_schema=RejectTimingArgs,
)

check_trigger_health_tool = BaseTool.create_tool(
    func=check_trigger_health,
    metadata=ToolMetadata(
        name="check_trigger_health",
        description=(
            "Độ tin cậy của trigger: số trigger, nhóm timeout, lỗi chụp ảnh, số lần "
            "service restart. Dùng khi user hỏi 'trigger có ổn không', 'có sản phẩm nào "
            "bị bỏ sót không', 'có bị mất ảnh không', 'service có restart không'. "
            "`groups_timeout` và `capture_failures` là sản phẩm ĐI QUA MÀ KHÔNG ĐƯỢC KIỂM."
        ),
        category="equipment",
    ),
    args_schema=TriggerHealthArgs,
)

check_sensor_pulse_tool = BaseTool.create_tool(
    func=check_sensor_pulse,
    metadata=ToolMetadata(
        name="check_sensor_pulse",
        description=(
            "Độ rộng xung cảm biến đầu vào (DI) và độ trôi trong ngày. Dùng khi user hỏi "
            "'cảm biến có ổn không', 'nhịp dây chuyền có đều không', 'tốc độ băng tải'. "
            "Xung phản ánh khoảng cách và tốc độ sản phẩm, không phải cấu hình reject."
        ),
        category="equipment",
    ),
    args_schema=SensorPulseArgs,
)

check_subsystem_health_tool = BaseTool.create_tool(
    func=check_subsystem_health,
    metadata=ToolMetadata(
        name="check_subsystem_health",
        description=(
            "Các hệ thống con báo lỗi KHỞI TẠO — không nạp được model, thiếu file, không "
            "start được. Dùng khi user hỏi 'có module nào lỗi không', 'hệ thống có phần "
            "nào không chạy', 'kiểm tra sức khoẻ hệ thống'. Loại lỗi này làm một hệ thống "
            "con nằm im hoàn toàn trong khi dây chuyền vẫn chạy, nên rất dễ bị bỏ qua."
        ),
        category="equipment",
    ),
    args_schema=SubsystemHealthArgs,
)

logger.info("✅ Equipment tools registered")
