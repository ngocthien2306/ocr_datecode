"""
Test script to simulate concurrent GPIO operations:
- Thread 1: Continuously read DI0 value
- Thread 2: Trigger reject pulse on DO1 every 400-500ms (random)
- Thread 3: Continuously read DO1 value to monitor state

Purpose: Detect any race conditions or errors with libapmi thread safety
"""

import sys
import os
import time
import random
import threading
import ctypes
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from camera_management.utils import (
    read_di_value, trigger_reject_pulse, _use_native_gpio,
    _gpio_lock, _apmi_dio_read_output, _LIBAPMI_PIN_MAP
)

# Statistics
stats = {
    'di_reads': 0,
    'di_errors': 0,
    'do_pulses': 0,
    'do_errors': 0,
    'do_reads': 0,
    'do_read_errors': 0,
    'do_value_high': 0,
    'do_value_low': 0,
    'start_time': None,
}
stats_lock = threading.Lock()

# Control flag
running = True


def read_do_value(do_number: int) -> int:
    """
    Read Digital Output pin current value (0 or 1)
    Uses global _gpio_lock because libapmi is not thread-safe.
    """
    with _gpio_lock:
        if _use_native_gpio and _apmi_dio_read_output is not None:
            try:
                value = ctypes.c_int()
                lib_pin = _LIBAPMI_PIN_MAP.get(do_number)
                if lib_pin is not None:
                    ret = _apmi_dio_read_output(lib_pin, ctypes.byref(value))
                    if ret == 0:
                        return value.value
                    else:
                        print(f"[DO Read] Native DO{do_number} read failed: ret={ret}")
                        return -1
            except Exception as e:
                print(f"[DO Read] Native DO{do_number} read error: {e}")
                return -1
        return -1  # No native support


def di_reader_thread():
    """Thread that continuously reads DI0"""
    global running

    print("[DI Reader] Started - reading DI0 continuously")

    while running:
        try:
            value = read_di_value(0)
            with stats_lock:
                stats['di_reads'] += 1
                # Check for error (ret=-1 would cause issues in the lib)
                if value is None or value < 0:
                    stats['di_errors'] += 1
                    print(f"[DI Reader] ERROR: Got invalid value: {value}")
        except Exception as e:
            with stats_lock:
                stats['di_errors'] += 1
            print(f"[DI Reader] EXCEPTION: {e}")

        # Small delay to not completely flood
        time.sleep(0.005)  # 5ms between reads

    print("[DI Reader] Stopped")


def do_reader_thread():
    """Thread that continuously reads DO1 to monitor state"""
    global running

    print("[DO Reader] Started - reading DO1 continuously")

    while running:
        try:
            value = read_do_value(1)
            with stats_lock:
                stats['do_reads'] += 1
                if value < 0:
                    stats['do_read_errors'] += 1
                elif value == 0:
                    stats['do_value_low'] += 1
                else:
                    stats['do_value_high'] += 1

            # Print current value (less frequently)
            # print(f"DO1 = {value}")

        except Exception as e:
            with stats_lock:
                stats['do_read_errors'] += 1
            print(f"[DO Reader] EXCEPTION: {e}")

        # Small delay
        time.sleep(0.003)  # 3ms between reads

    print("[DO Reader] Stopped")


def do_reject_thread():
    """Thread that triggers reject pulse every 400-500ms"""
    global running

    print("[DO Reject] Started - triggering DO1 pulse every 400-500ms")

    while running:
        try:
            # Random delay between 400-500ms
            delay_ms = random.randint(400, 500)
            time.sleep(delay_ms / 1000.0)

            # Trigger 300ms pulse on DO1
            trigger_reject_pulse(1, pulse_ms=300)

            with stats_lock:
                stats['do_pulses'] += 1

        except Exception as e:
            with stats_lock:
                stats['do_errors'] += 1
            print(f"[DO Reject] EXCEPTION: {e}")

    print("[DO Reject] Stopped")


def print_stats():
    """Print current statistics"""
    with stats_lock:
        elapsed = time.time() - stats['start_time'] if stats['start_time'] else 0
        print(f"\n{'='*70}")
        print(f"Statistics after {elapsed:.1f}s:")
        print(f"  DI Reads:  {stats['di_reads']:6d} (errors: {stats['di_errors']})")
        print(f"  DO Reads:  {stats['do_reads']:6d} (errors: {stats['do_read_errors']})")
        print(f"  DO Pulses: {stats['do_pulses']:6d} (errors: {stats['do_errors']})")

        if stats['do_reads'] > 0:
            low_pct = stats['do_value_low'] / stats['do_reads'] * 100
            high_pct = stats['do_value_high'] / stats['do_reads'] * 100
            print(f"  DO1 State: LOW={stats['do_value_low']} ({low_pct:.1f}%) | HIGH={stats['do_value_high']} ({high_pct:.1f}%)")

        if elapsed > 0:
            print(f"  Rates: DI={stats['di_reads']/elapsed:.1f}/s | DO_Read={stats['do_reads']/elapsed:.1f}/s | Pulse={stats['do_pulses']/elapsed:.1f}/s")

        total_ops = stats['di_reads'] + stats['do_reads'] + stats['do_pulses']
        total_errors = stats['di_errors'] + stats['do_read_errors'] + stats['do_errors']
        error_rate = total_errors / max(1, total_ops) * 100
        print(f"  Total Error Rate: {error_rate:.2f}% ({total_errors}/{total_ops})")
        print(f"{'='*70}\n")


def main():
    global running

    print("="*70)
    print("GPIO Concurrent Access Test (3 Threads)")
    print(f"Using native GPIO: {_use_native_gpio}")
    print("="*70)
    print("\nTest scenario:")
    print("  - Thread 1: Read DI0 continuously (~5ms interval)")
    print("  - Thread 2: Read DO1 continuously (~3ms interval)")
    print("  - Thread 3: Trigger DO1 pulse (300ms) every 400-500ms")
    print("\nPress Ctrl+C to stop and see results\n")

    stats['start_time'] = time.time()

    # Create threads
    t1 = threading.Thread(target=di_reader_thread, name="DI_Reader", daemon=True)
    t2 = threading.Thread(target=do_reader_thread, name="DO_Reader", daemon=True)
    t3 = threading.Thread(target=do_reject_thread, name="DO_Reject", daemon=True)

    # Start threads
    t1.start()
    t2.start()
    t3.start()

    try:
        # Run for a while, printing stats every 5 seconds
        while True:
            time.sleep(5)
            print_stats()

    except KeyboardInterrupt:
        print("\n\nStopping threads...")
        running = False

        # Wait for threads to finish
        t1.join(timeout=2)
        t2.join(timeout=2)
        t3.join(timeout=2)

        print_stats()
        print("Test completed!")


if __name__ == "__main__":
    main()
