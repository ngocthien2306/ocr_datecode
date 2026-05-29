#!/usr/bin/env python3
"""
Quick RAM check per service (AI / BE / FE / Mongo) + system + GPU.

Usage:
    python tools/ram_check.py              # one-shot snapshot
    python tools/ram_check.py -w 5         # repeat every 5s (Ctrl-C to stop)
    python tools/ram_check.py -j           # JSON output (for piping)
"""
import argparse
import json
import re
import subprocess
import sys
import time

try:
    import psutil
except ImportError:
    sys.exit("Need psutil:  pip install psutil")


SERVICES = {
    "AI":    r"camera_management_service|inference_service",
    "BE":    r"uvicorn|backend/app/main\.py|backend\.app",
    "FE":    r"vite|node.*frontend",
    "Mongo": r"mongod",
}


def _find_procs(pattern: str):
    rx = re.compile(pattern)
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if rx.search(cmd) and "ram_check" not in cmd:
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


def _service_stats(pattern: str):
    procs = _find_procs(pattern)
    if not procs:
        return None
    rss = sum(p.memory_info().rss for p in procs) / 1024 / 1024
    threads = sum(p.num_threads() for p in procs)
    return {"pids": [p.pid for p in procs], "rss_mb": round(rss, 1), "threads": threads}


def _gpu_info():
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip().splitlines()[0]
        used, total = (int(x.strip()) for x in raw.split(","))
        return {"used_mb": used, "total_mb": total, "pct": round(100 * used / total, 1)}
    except Exception:
        return None


def snapshot() -> dict:
    snap = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for label, pat in SERVICES.items():
        snap[label.lower()] = _service_stats(pat)
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    snap["sys"] = {
        "total_mb": round(vm.total / 1024 / 1024),
        "available_mb": round(vm.available / 1024 / 1024),
        "used_pct": vm.percent,
        "swap_used_mb": round(sm.used / 1024 / 1024),
        "swap_total_mb": round(sm.total / 1024 / 1024),
    }
    snap["gpu"] = _gpu_info()
    return snap


def print_pretty(snap: dict) -> None:
    print(f"\n========== {snap['ts']} ==========")
    for label in ("ai", "be", "fe", "mongo"):
        s = snap.get(label)
        if s:
            pids = ",".join(str(x) for x in s["pids"])
            print(f"  {label.upper():<6} {s['rss_mb']:>8.1f} MB  "
                  f"({len(s['pids'])} proc, {s['threads']} thr, pid={pids})")
        else:
            print(f"  {label.upper():<6} {'--':>8}      (not running)")
    sy = snap["sys"]
    print(f"  ---")
    print(f"  Sys total       {sy['total_mb']:>8} MB")
    print(f"  Sys available   {sy['available_mb']:>8} MB  ({sy['used_pct']:.1f}% used)")
    print(f"  Swap            {sy['swap_used_mb']:>8} MB / {sy['swap_total_mb']} MB")
    g = snap.get("gpu")
    if g:
        print(f"  GPU             {g['used_mb']:>8} MB / {g['total_mb']} MB  ({g['pct']:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--watch", type=float, default=0,
                    help="Repeat every N seconds (Ctrl-C to stop)")
    ap.add_argument("-j", "--json", action="store_true",
                    help="JSON output (one line per snapshot)")
    args = ap.parse_args()

    def once():
        s = snapshot()
        if args.json:
            print(json.dumps(s, ensure_ascii=False), flush=True)
        else:
            print_pretty(s)

    if args.watch <= 0:
        once()
        return
    try:
        while True:
            once()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
