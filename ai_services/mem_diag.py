"""
mem_diag.py — Công cụ chẩn đoán memory leak (CHỈ THÊM VÀO, dễ gỡ).

Chạy 1 daemon thread, mỗi `interval` giây ghi log:
  - RSS process, số threads, số file descriptor (FD)
  - Kích thước TỪNG cache crop nghi ngờ (_sim_crop_cache / _template_crop_cache)
    cùng tổng bytes của ndarray bên trong  -> xác nhận thủ phạm #1
  - Top allocation theo tracemalloc (chỉ thẳng file:line đang phình)

Bật/tắt bằng env MEM_DIAG (mặc định "1" = bật). Để tắt hẳn: MEM_DIAG=0.
Log ra:  ai_services/mem_diag.log  +  ai_services/mem_diag.csv
Gỡ bỏ:   xoá 2 dòng import/start trong camera_management_service.py rồi xoá file này.
"""
import os
import gc
import time
import threading
import tracemalloc
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = os.environ.get("MEM_DIAG_LOG", os.path.join(_HERE, "mem_diag.log"))
_CSV = os.path.join(_HERE, "mem_diag.csv")
_INTERVAL = int(os.environ.get("MEM_DIAG_INTERVAL", "120"))  # giây
_CACHE_ATTRS = ("_sim_crop_cache", "_template_crop_cache")

_started = False


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def _fd_count():
    try:
        return len(os.listdir("/proc/self/fd"))
    except Exception:
        return -1


def _scan_caches():
    """Tìm mọi instance verifier qua gc, trả về dict: attr -> (num_instances, total_entries, total_mb)."""
    out = {a: [0, 0, 0.0] for a in _CACHE_ATTRS}
    try:
        for obj in gc.get_objects():
            for attr in _CACHE_ATTRS:
                try:
                    cache = obj.__dict__.get(attr) if hasattr(obj, "__dict__") else None
                except Exception:
                    cache = None
                if isinstance(cache, dict):
                    out[attr][0] += 1
                    out[attr][1] += len(cache)
                    nbytes = 0
                    for v in list(cache.values()):
                        nbytes += getattr(v, "nbytes", 0) or 0
                    out[attr][2] += nbytes / (1024.0 * 1024.0)
    except Exception:
        pass
    return out


def _scan_big_arrays(min_mb=2.0, top=8):
    """numpy array KHÔNG bị gc track; dict chỉ chứa array+scalar cũng bị CPython untrack.
    -> Duyệt SÂU 2 TẦNG từ mọi container gốc (list/dict/deque còn tracked) để gom mảng
    lớn theo 'chữ ký', lộ ra container nào đang tích tụ (nơi leak)."""
    lines = []
    try:
        import numpy as np
        from collections import defaultdict
        agg = defaultdict(lambda: [0, 0.0])  # signature -> [số chỗ, tổng MB]
        uniq = {}  # id(arr) -> MB  (tổng bộ nhớ mảng lớn duy nhất)

        def scan_child(v, depth):
            """Trả (tổng_MB, set gợi ý key) của mảng lớn trong v tới độ sâu depth."""
            total = 0.0
            hints = set()
            tv = type(v)
            if tv is np.ndarray:
                m = (getattr(v, "nbytes", 0) or 0) / 1048576.0
                if m >= min_mb:
                    uniq[id(v)] = m
                    total += m
                return total, hints
            if depth <= 0:
                return total, hints
            try:
                if tv is dict:
                    if len(v) > 5000:
                        return total, hints
                    for k, val in list(v.items()):
                        m, h = scan_child(val, depth - 1)
                        if m > 0:
                            total += m
                            if isinstance(k, str):
                                hints.add(k)
                            hints |= h
                elif tv in (list, tuple, set, frozenset) or tv.__name__ == "deque":
                    if len(v) > 50000:
                        return total, hints
                    for val in list(v):
                        m, h = scan_child(val, depth - 1)
                        total += m
                        hints |= h
            except Exception:
                pass
            return total, hints

        for obj in gc.get_objects():
            t = type(obj)
            if t in (list, tuple, dict) or t.__name__ == "deque":
                mb, hints = scan_child(obj, 2)
                if mb >= min_mb:
                    inner = ",".join(sorted(hints)) if hints else "ndarray"
                    sig = ("dict{" + inner + "}") if t is dict else f"{t.__name__}[{inner}]"
                    agg[sig][0] += 1
                    agg[sig][1] += mb

        lines.append(f"  big-ndarray giữ lại (unique): {len(uniq)} mảng, {sum(uniq.values()):.0f}MB")
        for sig, (c, mb) in sorted(agg.items(), key=lambda x: -x[1][1])[:top]:
            lines.append(f"    {sig}: {c} container, ~{mb:.0f}MB")
    except Exception as e:
        lines.append(f"  scan_big_arrays error: {e}")
    return lines


def _log_line(text):
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')}  {text}"
    try:
        with open(_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print("[MEM_DIAG] " + text, flush=True)


def _loop():
    tracemalloc.start(25)
    baseline = None
    # header CSV
    try:
        if not os.path.exists(_CSV):
            with open(_CSV, "w") as f:
                f.write("ts,rss_mb,threads,fds,sim_entries,sim_mb,tmpl_entries,tmpl_mb\n")
    except Exception:
        pass

    n = 0
    while True:
        try:
            n += 1
            rss = _rss_mb()
            threads = threading.active_count()
            fds = _fd_count()
            caches = _scan_caches()
            sim = caches["_sim_crop_cache"]
            tmpl = caches["_template_crop_cache"]

            _log_line(
                f"#{n} RSS={rss:.0f}MB threads={threads} fds={fds} | "
                f"sim_crop_cache: {sim[0]}inst {sim[1]}entries {sim[2]:.1f}MB | "
                f"template_crop_cache: {tmpl[0]}inst {tmpl[1]}entries {tmpl[2]:.1f}MB"
            )
            try:
                with open(_CSV, "a") as f:
                    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')},"
                            f"{rss:.0f},{threads},{fds},"
                            f"{sim[1]},{sim[2]:.1f},{tmpl[1]},{tmpl[2]:.1f}\n")
            except Exception:
                pass

            # tracemalloc: so với baseline (lấy ở vòng 2 cho ổn định)
            snap = tracemalloc.take_snapshot()
            if baseline is None and n >= 2:
                baseline = snap
                _log_line("  (đã chốt baseline tracemalloc cho các lần so sánh sau)")
            elif baseline is not None:
                top = snap.compare_to(baseline, "lineno")[:8]
                _log_line("  TOP TĂNG BỘ NHỚ kể từ baseline (file:line  +KB  count):")
                for st in top:
                    fr = st.traceback[0]
                    _log_line(f"    {os.path.basename(fr.filename)}:{fr.lineno}  "
                              f"+{st.size_diff/1024:.0f}KB  ({st.count_diff:+d} obj)")
            # Truy ngược container đang giữ mảng lớn (chốt nơi leak)
            for ln in _scan_big_arrays():
                _log_line(ln)
        except Exception as e:
            _log_line(f"loop error: {e}")
        time.sleep(_INTERVAL)


def start():
    global _started
    if _started:
        return
    if os.environ.get("MEM_DIAG", "1") == "0":
        return
    _started = True
    t = threading.Thread(target=_loop, name="mem-diag", daemon=True)
    t.start()
    _log_line(f"==== MEM_DIAG BẮT ĐẦU (interval={_INTERVAL}s, pid={os.getpid()}) ====")
