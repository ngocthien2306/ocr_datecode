"""
Shared Memory Service
Reads frames from shared memory created by Camera processes
"""

from multiprocessing import shared_memory
import pickle
import struct
import time
import numpy as np
import cv2
import logging
from typing import Optional, Tuple, Dict, Any, List
import threading

logger = logging.getLogger(__name__)


class SharedMemoryService:
    """
    Service to read frames from shared memory

    Features:
    - Read latest frame from camera shared memory
    - Thread-safe operations
    - Automatic format conversion (BGR → JPEG)
    - Metadata extraction
    """

    # The AI service writer overwrites ring-buffer slots with no cross-process
    # lock. A slow reader can catch a slot mid-overwrite (torn read), which used
    # to feed corrupted bytes straight into pickle.loads() — pickle's C decoder
    # can hard-crash (native abort) on malformed input instead of raising a
    # catchable Python exception. _read_frame_from_slot uses the slot's own
    # frame_idx as a seqlock: read it before and after extracting the raw bytes,
    # and only unpickle if it didn't change. These bound the retries.
    _MAX_TORN_READ_RETRIES = 3
    _TORN_READ_RETRY_DELAY = 0.002  # seconds

    # unlink()-ing a POSIX shm segment does NOT invalidate an already-open
    # handle elsewhere — it stays fully readable, frozen at its last content.
    # When the AI writer recreates its segment (reconnect/restart), our cached
    # handle in _connections keeps "successfully" reading the orphaned old one
    # forever: frame_count stays >0, slots still parse, but nothing advances.
    #
    # This USED to be detected by watching frame_idx, which was wrong: a camera
    # in software_trigger mode writes no frame until a product passes, and a
    # stopped recipe writes none at all, so a healthy idle camera looked
    # identical to an orphaned segment. On M2 that false positive killed the
    # camera service 22 times in one day, once mid-production (2026-08-21
    # 00:39→00:55, 16 restarts spaced by the 60s debounce).
    #
    # The writer now publishes a heartbeat that ticks on every camera-loop
    # iteration regardless of grabbing (RingBufferSharedMemory.beat), so we can
    # separate the two cases properly:
    #   heartbeat advancing              → writer alive; frozen frames are normal
    #   heartbeat frozen ≥ _HEARTBEAT_MAX_AGE → orphaned segment / dead loop
    # frame_idx is no longer used to judge staleness at all.
    _HEALTH_MAGIC = 0x4F435231  # 'OCR1' — writer supports the liveness block
    _OFF_HEALTH_MAGIC = 24
    _OFF_GRAB_MODE = 28
    _OFF_HEARTBEAT_CTR = 32
    _OFF_HEARTBEAT_TS = 40
    GRAB_MODE_NAMES = {0: "idle", 1: "continuous", 2: "software_trigger"}

    # The camera loop beats at least every ~100ms, so anything beyond a couple of
    # seconds already means trouble. Kept generously wide so a GC pause, a busy
    # Jetson, or a slow frame conversion can never be mistaken for a dead writer.
    _HEARTBEAT_MAX_AGE = 15.0  # seconds

    def __init__(self):
        self._lock = threading.Lock()
        self._connections: Dict[str, shared_memory.SharedMemory] = {}
        # serial -> (heartbeat_counter, first_seen_monotonic)
        self._last_beat_seen: Dict[str, Tuple[int, float]] = {}

        logger.info("SharedMemoryService initialized")

    def read_health(self, serial_number: str) -> Dict[str, Any]:
        """
        Read the writer's liveness block from the ring-buffer header.

        Returns a dict with:
          supported   — False if the attached writer predates the heartbeat
                        block (all-zero bytes); callers MUST treat this as
                        "cannot judge" and never escalate to a restart.
          alive       — heartbeat advanced since the previous call, or has been
                        frozen for less than _HEARTBEAT_MAX_AGE.
          grab_mode   — 'idle' | 'continuous' | 'software_trigger'
          grabbing    — whether the camera is in a mode that produces frames at
                        all. Used to decide whether an empty/frozen buffer is
                        even worth reporting.
          frozen_for  — seconds the heartbeat has been stuck (0.0 when advancing)
        """
        unknown = {
            "supported": False, "alive": True, "grab_mode": "unknown",
            "grabbing": False, "frozen_for": 0.0,
        }
        with self._lock:
            shm = self._get_shm(serial_number)
            if not shm:
                return unknown
            try:
                magic = struct.unpack_from("<I", shm.buf, self._OFF_HEALTH_MAGIC)[0]
                if magic != self._HEALTH_MAGIC:
                    # Older AI service — stay conservative rather than guessing.
                    return unknown

                beat = struct.unpack_from("<Q", shm.buf, self._OFF_HEARTBEAT_CTR)[0]
                mode_code = struct.unpack_from("<I", shm.buf, self._OFF_GRAB_MODE)[0]
            except Exception as e:
                logger.debug(f"read_health failed for {serial_number}: {e}")
                return unknown

            now = time.monotonic()
            prev = self._last_beat_seen.get(serial_number)
            if prev is None or prev[0] != beat:
                self._last_beat_seen[serial_number] = (beat, now)
                frozen_for = 0.0
            else:
                frozen_for = now - prev[1]

            grab_mode = self.GRAB_MODE_NAMES.get(mode_code, "unknown")
            return {
                "supported": True,
                "alive": frozen_for < self._HEARTBEAT_MAX_AGE,
                "grab_mode": grab_mode,
                "grabbing": grab_mode in ("continuous", "software_trigger"),
                "frozen_for": frozen_for,
            }

    def check_staleness(self, serial_number: str, frame_idx: int = 0) -> bool:
        """
        True only when the writer's heartbeat has been frozen long enough to
        mean the attached segment is orphaned (or its camera loop is dead).

        `frame_idx` is accepted for call-site compatibility and deliberately
        ignored: a frozen frame_idx is normal for an idle or trigger-mode
        camera, and treating it as a fault is what caused the false-positive
        restarts this method was rewritten to stop.
        """
        health = self.read_health(serial_number)
        if not health["supported"]:
            return False
        return not health["alive"]

    def _get_shm(self, serial_number: str) -> Optional[shared_memory.SharedMemory]:
        """
        Get or create shared memory connection

        Args:
            serial_number: Camera serial number

        Returns:
            SharedMemory instance or None
        """
        shm_name = f"camera_{serial_number}"

        # Check if already connected
        if shm_name in self._connections:
            return self._connections[shm_name]

        # Try to connect
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
            self._connections[shm_name] = shm
            logger.info(f"Connected to shared memory: {shm_name}")
            return shm

        except FileNotFoundError:
            logger.warning(f"Shared memory not found: {shm_name}")
            return None

        except Exception as e:
            logger.error(f"Error connecting to shared memory {shm_name}: {e}")
            return None

    def read_frame(
        self,
        serial_number: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Read latest frame from ring buffer shared memory

        Args:
            serial_number: Camera serial number

        Returns:
            Tuple of (frame_array, metadata) or None if failed

        Ring Buffer Format:
        - Header (64 bytes): [write_idx, frame_count, buffer_size, frame_counter, ...]
        - Frame Slots (5 slots): Each slot contains one frame with metadata
        """
        with self._lock:
            shm = self._get_shm(serial_number)

            if not shm:
                return None

            try:
                # Read ring buffer header
                HEADER_SIZE = 64
                BUFFER_SIZE = 5

                # Read header fields
                write_idx = struct.unpack_from("<I", shm.buf, 0)[0]
                frame_count = struct.unpack_from("<I", shm.buf, 4)[0]
                # buffer_size at offset 8 (not used)
                # frame_counter at offset 12 (not used here)
                slot_size = struct.unpack_from("<I", shm.buf, 20)[0]  # NEW: Read slot_size

                # Check if buffer has any frames
                if frame_count == 0:
                    logger.warning(f"No frames available in ring buffer for {serial_number}")
                    return None

                # Calculate slot index of latest frame (last written)
                latest_slot_idx = (write_idx - 1) % BUFFER_SIZE

                # Read frame from slot
                result = self._read_frame_from_slot(shm, latest_slot_idx, HEADER_SIZE, slot_size)

                if result:
                    frame_array, metadata = result
                    logger.debug(
                        f"Read latest frame from {serial_number}: "
                        f"slot={latest_slot_idx}, frame_idx={metadata.get('frame_idx', 'N/A')}"
                    )
                    return frame_array, metadata

                return None

            except Exception as e:
                logger.error(f"Error reading frame from {serial_number}: {e}")
                # Remove failed connection
                if f"camera_{serial_number}" in self._connections:
                    try:
                        self._connections[f"camera_{serial_number}"].close()
                    except:
                        pass
                    del self._connections[f"camera_{serial_number}"]

                return None

    def _read_frame_from_slot(
        self,
        shm: shared_memory.SharedMemory,
        slot_idx: int,
        header_size: int,
        slot_size: int
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Read frame from specific ring buffer slot, retrying on torn reads.

        The writer (AI service) can be overwriting this exact slot while we
        read it. Each attempt is validated by _read_frame_from_slot_once via
        a seqlock check; a torn read returns None and we retry with a short
        backoff instead of risking a corrupted unpickle.
        """
        for attempt in range(self._MAX_TORN_READ_RETRIES):
            result = self._read_frame_from_slot_once(shm, slot_idx, header_size, slot_size)
            if result is not None:
                return result
            if attempt < self._MAX_TORN_READ_RETRIES - 1:
                time.sleep(self._TORN_READ_RETRY_DELAY)

        logger.warning(
            f"Slot {slot_idx}: torn/invalid read persisted after "
            f"{self._MAX_TORN_READ_RETRIES} attempts, giving up"
        )
        return None

    def _read_frame_from_slot_once(
        self,
        shm: shared_memory.SharedMemory,
        slot_idx: int,
        header_size: int,
        slot_size: int
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Single read attempt of a ring buffer slot.

        Args:
            shm: Shared memory instance
            slot_idx: Slot index (0-4)
            header_size: Size of ring buffer header
            slot_size: Size of each slot in bytes (read from header)

        Returns:
            Tuple of (frame_array, metadata) or None

        Slot Format:
        [frame_idx (8B)] [timestamp (8B)] [metadata_len (4B)] [metadata_bytes]
        [frame_len (4B)] [frame_bytes]

        Seqlock check: frame_idx is the slot's own version marker. We capture
        all raw bytes first (cheap memoryview copies), re-read frame_idx, and
        only then unpickle/reshape — if frame_idx changed, the writer touched
        this slot mid-read and the bytes we captured are a torn mix of two
        frames, so we bail out before pickle.loads ever sees them.
        """
        try:
            # Calculate slot offset using actual slot_size from header
            slot_offset = header_size + (slot_idx * slot_size)
            offset = slot_offset

            # Read frame_idx (8 bytes)
            frame_idx_before = struct.unpack_from("<Q", shm.buf, offset)[0]
            offset += 8

            # Read timestamp (8 bytes)
            timestamp_ns = struct.unpack_from("<Q", shm.buf, offset)[0]
            offset += 8

            # Read metadata length (4 bytes)
            metadata_len = struct.unpack_from("<I", shm.buf, offset)[0]
            offset += 4

            # Validate metadata_len to prevent reading garbage
            if metadata_len > 10000:  # Metadata should be < 10KB
                logger.warning(f"Invalid metadata_len={metadata_len}, slot may be empty")
                return None

            # Read metadata bytes (raw copy only — do NOT unpickle yet)
            metadata_bytes = bytes(shm.buf[offset:offset+metadata_len])
            offset += metadata_len

            # Read frame length (4 bytes)
            frame_len = struct.unpack_from("<I", shm.buf, offset)[0]
            offset += 4

            # Read frame bytes (raw copy)
            frame_bytes = bytes(shm.buf[offset:offset+frame_len])

            # Seqlock validation: bail before pickle.loads if the writer
            # overwrote this slot while we were copying the bytes above.
            frame_idx_after = struct.unpack_from("<Q", shm.buf, slot_offset)[0]
            if frame_idx_after != frame_idx_before:
                logger.debug(
                    f"Torn read on slot {slot_idx}: frame_idx "
                    f"{frame_idx_before} -> {frame_idx_after}"
                )
                return None

            # Bytes are consistent — safe to unpickle now.
            metadata = pickle.loads(metadata_bytes)

            # Add ring buffer specific metadata
            metadata['frame_idx'] = frame_idx_before
            metadata['timestamp_ns'] = timestamp_ns

            # Reconstruct frame array
            shape = metadata.get("shape")
            dtype = metadata.get("dtype")

            if not shape or not dtype:
                logger.error("Missing shape or dtype in metadata")
                return None

            frame_array = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)

            return frame_array, metadata

        except Exception as e:
            logger.error(f"Error reading frame from slot {slot_idx}: {e}")
            return None

    def read_frame_as_jpeg(
        self,
        serial_number: str,
        quality: int = 90
    ) -> Optional[bytes]:
        """
        Read frame and encode as JPEG

        Args:
            serial_number: Camera serial number
            quality: JPEG quality (0-100)

        Returns:
            JPEG bytes or None
        """
        result = self.read_frame(serial_number)

        if not result:
            return None

        frame_array, metadata = result

        try:
            # Encode as JPEG
            success, jpeg_buffer = cv2.imencode(
                ".jpg",
                frame_array,
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )

            if not success:
                logger.error(f"Failed to encode frame as JPEG for {serial_number}")
                return None

            return jpeg_buffer.tobytes()

        except Exception as e:
            logger.error(f"Error encoding frame as JPEG: {e}")
            return None

    def read_latest_frames(
        self,
        serial_number: str,
        count: int = 5
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Read N latest frames from ring buffer

        Args:
            serial_number: Camera serial number
            count: Number of frames to read (1-5)

        Returns:
            List of (frame_array, metadata) tuples, ordered from newest to oldest
        """
        with self._lock:
            shm = self._get_shm(serial_number)

            if not shm:
                return []

            try:
                # Constants
                HEADER_SIZE = 64
                BUFFER_SIZE = 5

                # Read ring buffer header
                write_idx = struct.unpack_from("<I", shm.buf, 0)[0]
                frame_count = struct.unpack_from("<I", shm.buf, 4)[0]
                slot_size = struct.unpack_from("<I", shm.buf, 20)[0]  # NEW: Read slot_size

                # Check if buffer has any frames
                if frame_count == 0:
                    logger.warning(f"No frames available in ring buffer for {serial_number}")
                    return []

                # Determine actual count (can't read more than what's available)
                actual_count = min(count, frame_count, BUFFER_SIZE)

                # Calculate slot indices (newest to oldest)
                # write_idx points to next write position, so last written is (write_idx - 1)
                frames = []
                for i in range(actual_count):
                    slot_idx = (write_idx - 1 - i) % BUFFER_SIZE
                    result = self._read_frame_from_slot(shm, slot_idx, HEADER_SIZE, slot_size)

                    if result:
                        frames.append(result)
                    else:
                        logger.warning(f"Failed to read frame from slot {slot_idx}")

                logger.info(
                    f"Read {len(frames)} frames from {serial_number} "
                    f"(requested={count}, available={frame_count})"
                )

                return frames

            except Exception as e:
                logger.error(f"Error reading frames from {serial_number}: {e}")
                return []

    def cleanup(self, serial_number: Optional[str] = None):
        """
        Cleanup shared memory connections

        Args:
            serial_number: If specified, cleanup only this camera.
                          Otherwise cleanup all.
        """
        with self._lock:
            if serial_number:
                shm_name = f"camera_{serial_number}"
                if shm_name in self._connections:
                    try:
                        self._connections[shm_name].close()
                        del self._connections[shm_name]
                        logger.info(f"Closed shared memory connection: {shm_name}")
                    except Exception as e:
                        logger.error(f"Error closing shared memory {shm_name}: {e}")
                # Drop liveness tracking too, so the next read after a fresh
                # reattach starts clean instead of comparing against a
                # heartbeat from the orphaned segment we just dropped.
                self._last_beat_seen.pop(serial_number, None)

            else:
                # Cleanup all connections
                for shm_name, shm in list(self._connections.items()):
                    try:
                        shm.close()
                        logger.info(f"Closed shared memory connection: {shm_name}")
                    except Exception as e:
                        logger.error(f"Error closing shared memory {shm_name}: {e}")

                self._connections.clear()
                self._last_beat_seen.clear()


# Singleton instance
shared_memory_service = SharedMemoryService()
