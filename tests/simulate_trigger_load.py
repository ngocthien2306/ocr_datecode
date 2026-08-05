#!/usr/bin/env python3
"""
Fire simulated triggers at a controlled rate, for load / soak testing.

Hits POST /api/trigger-simulator/simulate — the same path the UI's trigger
simulator uses — so cameras must be in software_trigger mode.

Two cadences:
    fixed   --rate N        N triggers per minute, evenly spaced
    random  --min-ms/--max-ms   one trigger per random interval in that range

Usage:
    python tests/simulate_trigger_load.py --rate 120 --duration 60
    python tests/simulate_trigger_load.py --mode random --min-ms 100 --max-ms 500 --duration 30
    python tests/simulate_trigger_load.py --rate 60 --duration 10 --dry-run
"""
import argparse
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def login(base_url, username, password):
    r = requests.post(f"{base_url}/api/auth/login",
                      data={"username": username, "password": password}, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["fixed", "random"], default="fixed")
    ap.add_argument("--rate", type=float, default=60, help="fixed mode: triggers per minute")
    ap.add_argument("--min-ms", type=float, default=100, help="random mode: min interval (ms)")
    ap.add_argument("--max-ms", type=float, default=500, help="random mode: max interval (ms)")
    ap.add_argument("--duration", type=float, required=True, help="how long to keep triggering (seconds)")
    ap.add_argument("--serial", default=None, help="camera serial (default: all software-trigger cameras)")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--token", default=None, help="use an existing JWT instead of logging in")
    ap.add_argument("--dry-run", action="store_true", help="print the cadence, send nothing")
    args = ap.parse_args()

    def next_interval():
        if args.mode == "random":
            return random.uniform(args.min_ms, args.max_ms) / 1000.0
        return 60.0 / args.rate

    headers = {}
    if args.token:
        headers = {"Authorization": f"Bearer {args.token}"}
    elif not args.dry_run:
        try:
            headers = {"Authorization": f"Bearer {login(args.url, args.user, args.password)}"}
        except Exception as e:
            print(f"login failed at {args.url}: {e}\n"
                  f"pass --user/--password (or --token), or --dry-run to check the cadence only")
            return 2
    endpoint = f"{args.url}/api/trigger-simulator/simulate"
    payload = {"serial_number": args.serial, "trigger_type": "rising_edge"}

    latencies, errors = [], []

    def fire(n):
        if args.dry_run:
            return
        t0 = time.perf_counter()
        try:
            r = requests.post(endpoint, json=payload, headers=headers, timeout=10)
            latencies.append((time.perf_counter() - t0) * 1000)
            if r.status_code != 200:
                errors.append(f"#{n} HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e:
            errors.append(f"#{n} {type(e).__name__}: {e}")

    cadence = (f"{args.rate}/min ({60000 / args.rate:.0f}ms)" if args.mode == "fixed"
               else f"random {args.min_ms:.0f}-{args.max_ms:.0f}ms")
    print(f"mode={args.mode} cadence={cadence} duration={args.duration}s "
          f"target={args.serial or 'all'}{' [DRY-RUN]' if args.dry_run else ''}")

    # Sending runs on a small pool so a slow response never shifts the cadence.
    pool = ThreadPoolExecutor(max_workers=8)
    start = time.perf_counter()
    deadline = start + args.duration
    next_at = start
    sent = late = 0

    while next_at < deadline:
        now = time.perf_counter()
        if next_at > now:
            time.sleep(next_at - now)
        elif now - next_at > 0.005:
            late += 1                      # scheduler fell behind — machine can't keep up
        sent += 1
        pool.submit(fire, sent)
        next_at += next_interval()

    pool.shutdown(wait=True)

    # Rate is over the requested window, not the time of the last send — the
    # loop stops scheduling at the deadline, so the final gap belongs to it too.
    print(f"\nsent={sent} over {args.duration:g}s -> {sent / args.duration * 60:.1f} triggers/min "
          f"(behind schedule {late}x, in-flight time {time.perf_counter() - start:.1f}s)")
    if latencies:
        latencies.sort()
        print(f"latency ms: avg={statistics.mean(latencies):.1f} "
              f"p50={latencies[len(latencies) // 2]:.1f} "
              f"p95={latencies[int(len(latencies) * 0.95)]:.1f} max={latencies[-1]:.1f}")
    print(f"errors={len(errors)}")
    for e in errors[:5]:
        print(f"  {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
