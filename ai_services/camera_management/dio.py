"""
Digital I/O for the inspection rig: DI trigger pins and DO reject/alarm pins.

Split out of the old catch-all utils.py — this is the only place that talks
to the GPIO hardware (ASUS libapmi via ctypes, with a subprocess fallback),
so the locking rules below apply to every DI/DO access in the service.
"""

import ctypes
import logging
import subprocess
import threading
from typing import Optional

logger = logging.getLogger(__name__)


# ============= GPIO/DI/DO Utilities =============

# Global lock for ALL GPIO operations (libapmi is NOT thread-safe!)
# DI read and DO write must be serialized to prevent ret=-1 errors
_gpio_lock = threading.Lock()

# Per-pin locks for pulse sequences (prevent concurrent writes during pulse)
# Use RLock to allow reentrant calls from trigger_reject_pulse
_pulse_locks = {i: threading.RLock() for i in range(8)}

# ============= ASUS libapmi Native Library (Fast GPIO) =============
# Uses ctypes to call ASUS PE1100N native library directly
# Much faster than subprocess (~21ms vs ~37ms per call)

_libapmi = None
_apmi_dio_read_input = None
_apmi_dio_write_output = None
_apmi_dio_read_output = None
_use_native_gpio = False

# libapmi uses power-of-2 pin mapping: DO0->1, DO1->2, DO2->4, DO3->8
_LIBAPMI_PIN_MAP = {0: 1, 1: 2, 2: 4, 3: 8}

def _init_libapmi():
    """Initialize ASUS libapmi library for native GPIO access"""
    global _libapmi, _apmi_dio_read_input, _apmi_dio_write_output, _apmi_dio_read_output, _use_native_gpio

    try:
        _libapmi = ctypes.CDLL('/lib/libapmi.so')

        # Setup apmi_dio_read_input(int pin, int *value) -> int
        _apmi_dio_read_input = _libapmi.apmi_dio_read_input
        _apmi_dio_read_input.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        _apmi_dio_read_input.restype = ctypes.c_int

        # Setup apmi_dio_write_output(int pin, int value) -> int
        _apmi_dio_write_output = _libapmi.apmi_dio_write_output
        _apmi_dio_write_output.argtypes = [ctypes.c_int, ctypes.c_int]
        _apmi_dio_write_output.restype = ctypes.c_int

        # Setup apmi_dio_read_output(int pin, int *value) -> int
        _apmi_dio_read_output = _libapmi.apmi_dio_read_output
        _apmi_dio_read_output.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        _apmi_dio_read_output.restype = ctypes.c_int

        _use_native_gpio = True
        # print("ASUS libapmi initialized - using native GPIO (fast mode)")   
        logger.info("ASUS libapmi initialized - using native GPIO (fast mode)")

    except Exception as e:
        logger.warning(f"Failed to load libapmi, falling back to subprocess: {e}")
        _use_native_gpio = False

# Initialize on module load
_init_libapmi()

def read_di_value(di_number: int) -> int:
    """
    Read Digital Input pin value (0 or 1)

    Args:
        di_number: DI pin number (0-3)

    Returns:
        Pin value (0 or 1), or 0 on error

    Note:
        Uses global _gpio_lock because libapmi is not thread-safe.
    """
    with _gpio_lock:
        # Try native library first (faster)
        if _use_native_gpio and _apmi_dio_read_input is not None:
            try:
                value = ctypes.c_int()
                lib_pin = _LIBAPMI_PIN_MAP.get(di_number)
                if lib_pin is not None:
                    ret = _apmi_dio_read_input(lib_pin, ctypes.byref(value))
                    if ret == 0:
                        # print(f"DI{di_number} = {value.value} (native)")
                        return value.value
                    else:
                        logger.warning(f"Native DI{di_number} read failed: ret={ret}")
            except Exception as e:
                logger.warning(f"Native DI{di_number} read error: {e}")

        # Fallback to subprocess
        try:
            result = subprocess.run(
                ["sudo", "dio_in", str(di_number)],
                capture_output=True,
                text=True,
                timeout=0.5
            )

            if result.returncode == 0:
                # Parse output format: "The id-X input gpio status = Y\nCompletion code = 0x00"
                output = result.stdout.strip()

                # Extract value from "status = Y" line
                for line in output.split('\n'):
                    if 'status' in line and '=' in line:
                        # Extract the value after '='
                        value_str = line.split('=')[-1].strip()
                        try:
                            value = int(value_str)
                            return value
                        except ValueError:
                            logger.warning(f"Failed to parse DI{di_number} value: {value_str}")
                            return 0

                # Fallback: couldn't find status line
                logger.warning(f"Unexpected DI{di_number} output format: {output}")
                return 0
            else:
                logger.warning(f"Failed to read DI{di_number}: {result.stderr.strip()}")
                return 0

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout reading DI{di_number}")
            return 0
        except Exception as e:
            logger.error(f"Error reading DI{di_number}: {e}")
            return 0


def check_trigger_edge(current: int, previous: Optional[int], activation: str) -> bool:
    """
    Check if current value matches trigger activation edge

    Args:
        current: Current DI value (0 or 1)
        previous: Previous DI value (0, 1, or None)
        activation: Trigger activation type (RisingEdge, FallingEdge, AnyEdge)

    Returns:
        True if edge detected, False otherwise
    """
    if previous is None:
        return False

    if activation == "RisingEdge":
        return previous == 0 and current == 1
    elif activation == "FallingEdge":
        return previous == 1 and current == 0
    elif activation == "AnyEdge":
        return previous != current
    else:
        logger.warning(f"Unknown trigger activation: {activation}")
        return False


def _write_do_raw(do_number: int, value: int) -> bool:
    """
    Internal: Write DO value with only _gpio_lock (no pulse_lock).
    Used by trigger_reject_pulse which already holds pulse_lock.

    MUST be called while holding pulse_lock for the pin!
    """
    with _gpio_lock:
        # Try native library first (faster)
        if _use_native_gpio and _apmi_dio_write_output is not None:
            try:
                lib_pin = _LIBAPMI_PIN_MAP.get(do_number)
                if lib_pin is not None:
                    ret = _apmi_dio_write_output(lib_pin, value)
                    if ret == 0:
                        logger.debug(f"DO{do_number} = {value} (native)")
                        return True
                    else:
                        logger.warning(f"Native DO{do_number} write failed: ret={ret}")
            except Exception as e:
                logger.warning(f"Native DO{do_number} write error: {e}")

        # Fallback to subprocess
        try:
            result = subprocess.run(
                ["sudo", "dio_out", str(do_number), str(value)],
                capture_output=True,
                text=True,
                timeout=1.0
            )

            if "Completion code = 0x00" in result.stdout or "Completion code = 0x00" in result.stderr:
                logger.debug(f"DO{do_number} = {value}")
                return True
            elif "Completion code = 0xFFFFFFFF" in result.stdout or "Completion code = 0xFFFFFFFF" in result.stderr:
                logger.debug(f"DO{do_number} already at {value}")
                return True
            elif result.returncode == 0:
                logger.debug(f"DO{do_number} = {value} (returncode=0)")
                return True
            else:
                logger.error(f"Failed to set DO{do_number}: {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout setting DO{do_number}")
            return False
        except Exception as e:
            logger.error(f"Error setting DO{do_number}: {e}")
            return False


def write_do_value(do_number: int, value: int) -> bool:
    """
    Set Digital Output pin value (public API).

    Args:
        do_number: DO pin number (0-7)
        value: Pin value (0 or 1)

    Returns:
        True if success, False on error

    Note:
        Uses per-pin pulse lock to prevent writes during active pulse.
        If a pulse is in progress on this pin, this call will block until complete.
    """
    pulse_lock = _pulse_locks.get(do_number)
    if pulse_lock is None:
        logger.error(f"Invalid DO pin: {do_number}")
        return False

    with pulse_lock:  # Wait for any active pulse on this pin
        return _write_do_raw(do_number, value)


def trigger_reject_pulse(do_number: int, pulse_ms: int = 100):
    """
    Trigger reject pulse on DO pin (ACTIVE LOW logic).

    Pulse sequence: HIGH -> LOW (pulse_ms) -> HIGH
    Uses per-pin lock to prevent concurrent writes during pulse.

    Args:
        do_number: DO pin number (0-3)
        pulse_ms: Pulse duration in milliseconds

    Raises:
        RuntimeError: If write fails
    """
    import time

    pulse_lock = _pulse_locks.get(do_number)
    if pulse_lock is None:
        raise RuntimeError(f"Invalid DO pin: {do_number}")

    with pulse_lock:
        # Set LOW (active)
        if not _write_do_raw(do_number, 0):
            raise RuntimeError(f"Failed to set DO{do_number} LOW")

        try:
            # Hold pulse - _gpio_lock is FREE during sleep, DI reads can proceed
            time.sleep(pulse_ms / 1000.0)
        finally:
            # Always release pin to HIGH — even if sleep is interrupted or exception occurs.
            # Without this, a hardware write failure leaves the relay energized (stuck ON).
            if not _write_do_raw(do_number, 1):
                logger.error(f"CRITICAL: Failed to release DO{do_number} to HIGH — pin may be stuck active!")

    logger.info(f"DO{do_number} pulse complete ({pulse_ms}ms)")
