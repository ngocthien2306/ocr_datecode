"""
I/O Utilities for Digital Input/Output Operations
Handles reading and writing to DI/DO pins via subprocess
"""

import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def read_di_pin(pin_number: int) -> int:
    """
    Read Digital Input pin value

    Args:
        pin_number: DI pin number (0-3)

    Returns:
        Pin value (0 or 1)
    """
    try:
        result = subprocess.run(
            ["sudo", "-n", "dio_in", str(pin_number)],
            capture_output=True,
            text=True,
            timeout=1
        )

        if result.returncode == 0:
            # Parse output to get status value
            # Expected format: "status = 0" or "status = 1"
            for line in result.stdout.split("\n"):
                if "status" in line and "=" in line:
                    value_str = line.split("=")[-1].strip()
                    return int(value_str)

        logger.warning(f"Failed to read DI{pin_number}: {result.stderr}")
        return 0

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout reading DI{pin_number}")
        return 0
    except Exception as e:
        logger.error(f"Error reading DI{pin_number}: {e}")
        return 0


def read_do_pin(pin_number: int) -> int:
    """
    Read Digital Output pin value

    Args:
        pin_number: DO pin number (0-3)

    Returns:
        Pin value (0 or 1)
    """
    try:
        result = subprocess.run(
            ["sudo", "-n", "dio_out", str(pin_number)],
            capture_output=True,
            text=True,
            timeout=1
        )

        if result.returncode == 0:
            # Parse output to get status value
            for line in result.stdout.split("\n"):
                if "status" in line and "=" in line:
                    value_str = line.split("=")[-1].strip()
                    return int(value_str)

        logger.warning(f"Failed to read DO{pin_number}: {result.stderr}")
        return 0

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout reading DO{pin_number}")
        return 0
    except Exception as e:
        logger.error(f"Error reading DO{pin_number}: {e}")
        return 0


def write_do_pin(pin_number: int, value: int) -> bool:
    """
    Write value to Digital Output pin

    Args:
        pin_number: DO pin number (0-3)
        value: Value to write (0 or 1)

    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(
            ["sudo", "-n", "dio_out", str(pin_number), str(value)],
            capture_output=True,
            text=True,
            timeout=1
        )

        if result.returncode == 0:
            logger.info(f"DO{pin_number} set to {value}")
            return True
        else:
            logger.error(f"Failed to set DO{pin_number}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout writing DO{pin_number}")
        return False
    except Exception as e:
        logger.error(f"Error writing DO{pin_number}: {e}")
        return False


def read_all_di_pins() -> list[int]:
    """
    Read all DI pins (0-3)

    Returns:
        List of 4 values [DI0, DI1, DI2, DI3]
    """
    return [read_di_pin(i) for i in range(4)]


def read_all_do_pins() -> list[int]:
    """
    Read all DO pins (0-3)

    Returns:
        List of 4 values [DO0, DO1, DO2, DO3]
    """
    return [read_do_pin(i) for i in range(4)]
