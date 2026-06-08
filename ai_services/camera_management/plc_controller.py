"""
PLC Controller Module

Manages Modbus TCP connection to PLC for reject control.
Provides fallback to DIO if PLC connection fails.

Features:
- Singleton connection pattern
- Write pulse with timing measurement
- Auto-fallback to DIO on connection failure
- Statistics tracking (write time, failures)
"""

import logging
import threading
import time
from typing import Optional, Tuple
from pymodbus.client import ModbusTcpClient

logger = logging.getLogger(__name__)


class PLCController:
    """
    PLC Controller for Modbus TCP communication

    Features:
    - Singleton connection (connect once, reuse)
    - Write pulse: 1 → sleep(pulse_ms) → 0
    - Statistics tracking
    - Thread-safe operations
    """

    # PLC Configuration (fixed)
    PLC_IP = "192.168.1.5"
    PLC_PORT = 502
    PLC_TIMEOUT = 3.0  # seconds

    # Keep-alive — prevents Delta PLC's Modbus TCP idle timeout (~30-60s) from
    # silently closing the connection between writes. Combines OS TCP-keepalive
    # (early dead-connection detection) and an application-level heartbeat
    # (Modbus read so PLC sees traffic and doesn't idle-close us).
    HEARTBEAT_INTERVAL_S = 5.0  # how often to send a probe read
    HEARTBEAT_PROBE_ADDR = 0    # register to read (any valid addr will do)
    TCP_KEEPIDLE_S = 10         # OS starts probing after this many idle seconds
    TCP_KEEPINTVL_S = 5         # probe interval
    TCP_KEEPCNT = 3             # dead after N missed probes

    def __init__(self):
        """Initialize PLC Controller"""
        self._client: Optional[ModbusTcpClient] = None
        self._connected = False
        self._lock = threading.RLock()

        # Heartbeat thread
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._last_activity = time.time()  # updated by writes + heartbeat reads

        # Statistics
        self._stats = {
            'total_writes': 0,
            'total_pulses': 0,
            'total_failures': 0,
            'total_write_time_ms': 0.0,
            'avg_write_time_ms': 0.0,
            'max_write_time_ms': 0.0,
            'min_write_time_ms': float('inf')
        }
        self._stats_lock = threading.Lock()

        logger.info("PLCController initialized")

    def _apply_socket_keepalive(self) -> None:
        """
        Enable TCP keepalive on the underlying socket so the OS probes idle
        connections and reports them dead before our next write fails. Best-
        effort: silently skips if the pymodbus client doesn't expose `.socket`
        or the platform doesn't support the TCP_KEEP* options.
        """
        sock = None
        # pymodbus 3.x exposes `socket` on ModbusTcpClient. Older releases stash
        # it under different names; try a couple of fallbacks.
        for attr in ('socket', '_sock'):
            sock = getattr(self._client, attr, None)
            if sock is not None:
                break
        if sock is None:
            logger.warning(
                "PLC keepalive: underlying socket not exposed by pymodbus — skipping"
            )
            return
        try:
            import socket as _socket
            sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)
            # Linux-only TCP_KEEP* options — wrap each in try so unknown
            # platforms still get SO_KEEPALIVE.
            for name, val in (
                ('TCP_KEEPIDLE',  self.TCP_KEEPIDLE_S),
                ('TCP_KEEPINTVL', self.TCP_KEEPINTVL_S),
                ('TCP_KEEPCNT',   self.TCP_KEEPCNT),
            ):
                opt = getattr(_socket, name, None)
                if opt is not None:
                    try:
                        sock.setsockopt(_socket.IPPROTO_TCP, opt, val)
                    except OSError:
                        pass
            logger.info(
                f"PLC keepalive: SO_KEEPALIVE ON, "
                f"idle={self.TCP_KEEPIDLE_S}s intvl={self.TCP_KEEPINTVL_S}s "
                f"count={self.TCP_KEEPCNT}"
            )
        except Exception as e:
            logger.warning(f"PLC keepalive: setsockopt failed ({e})")

    def _heartbeat_loop(self) -> None:
        """
        Application-level heartbeat. Every HEARTBEAT_INTERVAL_S, if the
        connection has been idle long enough, send a cheap Modbus read so the
        PLC sees ongoing traffic and won't idle-close us. Real writes update
        `_last_activity` so we only probe on truly idle stretches.
        """
        logger.info(
            f"PLC heartbeat: started (every {self.HEARTBEAT_INTERVAL_S}s when idle)"
        )
        try:
            while not self._heartbeat_stop.wait(self.HEARTBEAT_INTERVAL_S):
                # Skip if other traffic already kept the link warm
                if (time.time() - self._last_activity) < self.HEARTBEAT_INTERVAL_S:
                    continue
                with self._lock:
                    if not self._connected or self._client is None:
                        continue
                    try:
                        result = self._client.read_holding_registers(
                            self.HEARTBEAT_PROBE_ADDR, count=1
                        )
                        if result.isError():
                            logger.warning(
                                f"PLC heartbeat: probe returned error — marking disconnected ({result})"
                            )
                            self._connected = False
                        else:
                            self._last_activity = time.time()
                            logger.debug("PLC heartbeat: probe OK")
                    except Exception as e:
                        # PLC dropped us between writes → mark disconnected,
                        # next write will trigger reconnect (or sensor_based
                        # send_verdict will retry).
                        logger.warning(
                            f"PLC heartbeat: probe exception ({e}) — marking disconnected"
                        )
                        self._connected = False
                        try:
                            self._client.close()
                        except Exception:
                            pass
        finally:
            logger.info("PLC heartbeat: stopped")

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name='PLCHeartbeat'
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None

    def connect(self) -> bool:
        """
        Connect to PLC via Modbus TCP

        Returns:
            True if connected successfully, False otherwise
        """
        with self._lock:
            # Already connected
            if self._connected and self._client and self._client.connected:
                logger.info(f"✅ Already connected to PLC at {self.PLC_IP}:{self.PLC_PORT}")
                return True

            # Disconnect old client if exists
            if self._client:
                try:
                    self._client.close()
                except:
                    pass

            # Create new client
            try:
                logger.info(f"🔌 Connecting to PLC at {self.PLC_IP}:{self.PLC_PORT}...")
                self._client = ModbusTcpClient(
                    host=self.PLC_IP,
                    port=self.PLC_PORT,
                    timeout=self.PLC_TIMEOUT
                )

                # Attempt connection
                if self._client.connect():
                    self._connected = True
                    self._last_activity = time.time()
                    self._apply_socket_keepalive()
                    self._start_heartbeat()
                    logger.info(f"✅ Connected to PLC at {self.PLC_IP}:{self.PLC_PORT}")
                    return True
                else:
                    logger.error(f"❌ Failed to connect to PLC at {self.PLC_IP}:{self.PLC_PORT}")
                    self._connected = False
                    return False

            except Exception as e:
                logger.error(f"❌ Error connecting to PLC: {e}")
                self._connected = False
                return False

    def disconnect(self):
        """Disconnect from PLC"""
        # Stop heartbeat BEFORE taking the lock so the thread can finish its
        # current probe without deadlocking on `self._lock`.
        self._stop_heartbeat()
        with self._lock:
            if self._client:
                try:
                    logger.info("🔌 Disconnecting from PLC...")
                    self._client.close()
                    self._connected = False
                    logger.info("✅ Disconnected from PLC")
                except Exception as e:
                    logger.error(f"Error disconnecting from PLC: {e}")

            self._client = None
            self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to PLC"""
        with self._lock:
            return self._connected and self._client and self._client.connected

    def write_register(self, address: int, value: int) -> Tuple[bool, float]:
        """
        Write single register to PLC

        Args:
            address: Register address (D0=0, D1=1, etc.)
            value: Value to write (0 or 1)

        Returns:
            Tuple of (success: bool, write_time_ms: float)
        """
        with self._lock:
            if not self.is_connected():
                logger.warning("PLC not connected, cannot write register")
                return False, 0.0

            try:
                t_start = time.perf_counter()
                result = self._client.write_register(address, value)
                t_end = time.perf_counter()

                write_time_ms = (t_end - t_start) * 1000.0

                # Check result
                if result.isError():
                    logger.error(f"❌ PLC write error: D{address}={value} | {result}")
                    self._update_stats(write_time_ms, success=False)
                    return False, write_time_ms
                else:
                    logger.debug(f"✅ PLC write OK: D{address}={value} | {write_time_ms:.2f}ms")
                    self._update_stats(write_time_ms, success=True)
                    self._last_activity = time.time()
                    return True, write_time_ms

            except Exception as e:
                logger.error(f"❌ Exception writing to PLC: {e}")
                self._update_stats(0.0, success=False)
                # Drop the connected flag so next call triggers reconnect
                # instead of attempting another write through a dead socket.
                self._connected = False
                return False, 0.0

    def write_coil(self, address: int, value: bool) -> bool:
        """
        Write single coil (bit) to PLC. Used by sensor_based reject for the
        ready/ack handshake signals.

        Args:
            address: Coil address (M0=0, M1=1, etc.)
            value: Coil state (True=ON, False=OFF)

        Returns:
            True if write succeeded, False otherwise.
        """
        with self._lock:
            if not self.is_connected():
                logger.warning(f"PLC not connected, cannot write coil at {address}")
                return False
            try:
                t_start = time.perf_counter()
                result = self._client.write_coil(address, bool(value))
                t_end = time.perf_counter()
                dt_ms = (t_end - t_start) * 1000.0
                if result.isError():
                    logger.error(f"❌ PLC write_coil error: coil@{address}={value} | {result}")
                    self._update_stats(dt_ms, success=False)
                    return False
                logger.debug(f"✅ PLC write_coil OK: coil@{address}={value} | {dt_ms:.2f}ms")
                self._update_stats(dt_ms, success=True)
                self._last_activity = time.time()
                return True
            except Exception as e:
                logger.error(f"❌ Exception writing PLC coil: {e}")
                self._update_stats(0.0, success=False)
                self._connected = False
                return False

    def read_coil(self, address: int) -> Optional[bool]:
        """
        Read single coil (bit) from PLC. Used by sensor_based reject to poll
        the ACK handshake signal.

        Args:
            address: Coil address.

        Returns:
            True/False on success. None on read error / not connected — caller
            should treat None as "ack not yet observed" (still in poll loop).
        """
        with self._lock:
            if not self.is_connected():
                return None
            try:
                result = self._client.read_coils(address, count=1)
                if result.isError():
                    logger.debug(f"PLC read_coil error: coil@{address} | {result}")
                    return None
                self._last_activity = time.time()
                return bool(result.bits[0])
            except Exception as e:
                logger.debug(f"Exception reading PLC coil: {e}")
                self._connected = False
                return None

    def write_pulse(self, register_address: int, pulse_ms: float) -> Tuple[bool, float]:
        """
        Write pulse to PLC register

        Pulse sequence: Write 1 → sleep(pulse_ms) → Write 0

        Args:
            register_address: Register address (D0=0, D1=1, etc.)
            pulse_ms: Pulse duration in milliseconds

        Returns:
            Tuple of (success: bool, total_duration_ms: float)
        """
        with self._lock:
            if not self.is_connected():
                logger.warning(f"PLC not connected, cannot write pulse to D{register_address}")
                return False, 0.0

            try:
                t_pulse_start = time.perf_counter()

                # Step 1: Write 1 (activate)
                success_1, write_time_1 = self.write_register(register_address, 1)
                if not success_1:
                    logger.error(f"❌ Failed to write D{register_address}=1")
                    return False, 0.0

                logger.info(f"🔴 PLC D{register_address}=1 (activate) | {write_time_1:.2f}ms")

                # Step 2: Hold pulse
                time.sleep(pulse_ms / 1000.0)

                # Step 3: Write 0 (deactivate)
                success_0, write_time_0 = self.write_register(register_address, 0)
                if not success_0:
                    logger.error(f"❌ Failed to write D{register_address}=0")
                    # Try to recover by writing 0 again
                    time.sleep(0.01)
                    success_0, write_time_0 = self.write_register(register_address, 0)
                    if not success_0:
                        logger.error(f"❌ Recovery failed for D{register_address}=0")
                        return False, 0.0

                logger.info(f"⚪ PLC D{register_address}=0 (deactivate) | {write_time_0:.2f}ms")

                # Calculate total duration
                t_pulse_end = time.perf_counter()
                total_duration_ms = (t_pulse_end - t_pulse_start) * 1000.0

                # Update pulse stats
                with self._stats_lock:
                    self._stats['total_pulses'] += 1

                logger.info(
                    f"✅ PLC pulse complete: D{register_address} | "
                    f"Duration: {total_duration_ms:.2f}ms "
                    f"(write: {write_time_1 + write_time_0:.2f}ms, pulse: {pulse_ms:.0f}ms)"
                )

                return True, total_duration_ms

            except Exception as e:
                logger.error(f"❌ Exception during PLC pulse: {e}")
                import traceback
                traceback.print_exc()
                return False, 0.0

    def write_dual_pulse(self, address1: int, address2: int, pulse_ms: float) -> Tuple[bool, float]:
        """
        Write pulse to TWO PLC registers simultaneously (optimized for consecutive addresses)

        Uses write_registers() for consecutive addresses (D0,D1) or two separate writes for non-consecutive.

        Pulse sequence: Write [1,1] → sleep(pulse_ms) → Write [0,0]

        Args:
            address1: First register address (e.g., D0=0)
            address2: Second register address (e.g., D1=1)
            pulse_ms: Pulse duration in milliseconds

        Returns:
            Tuple of (success: bool, total_duration_ms: float)
        """
        with self._lock:
            if not self.is_connected():
                logger.warning(f"PLC not connected, cannot write dual pulse to D{address1},D{address2}")
                return False, 0.0

            try:
                t_pulse_start = time.perf_counter()

                # Check if addresses are consecutive
                if abs(address2 - address1) == 1:
                    # Consecutive addresses, use write_registers() for better performance
                    start_addr = min(address1, address2)
                    logger.info(f"🟡 PLC D{address1},D{address2}=[1,1] (consecutive, optimized)")

                    # Step 1: Write [1, 1] (activate both)
                    t_start = time.perf_counter()
                    result = self._client.write_registers(start_addr, [1, 1])
                    t_end = time.perf_counter()
                    write_time_1 = (t_end - t_start) * 1000.0

                    if result.isError():
                        logger.error(f"❌ PLC write error: D{start_addr}-D{start_addr+1}=[1,1] | {result}")
                        self._update_stats(write_time_1, success=False)
                        return False, 0.0

                    logger.info(f"🔴 PLC D{address1},D{address2}=[1,1] (activate) | {write_time_1:.2f}ms")
                    self._update_stats(write_time_1, success=True)

                    # Step 2: Hold pulse
                    time.sleep(pulse_ms / 1000.0)

                    # Step 3: Write [0, 0] (deactivate both)
                    t_start = time.perf_counter()
                    result = self._client.write_registers(start_addr, [0, 0])
                    t_end = time.perf_counter()
                    write_time_0 = (t_end - t_start) * 1000.0

                    if result.isError():
                        logger.error(f"❌ PLC write error: D{start_addr}-D{start_addr+1}=[0,0] | {result}")
                        self._update_stats(write_time_0, success=False)
                        return False, 0.0

                    logger.info(f"⚪ PLC D{address1},D{address2}=[0,0] (deactivate) | {write_time_0:.2f}ms")
                    self._update_stats(write_time_0, success=True)

                    # Calculate total duration
                    t_pulse_end = time.perf_counter()
                    total_duration_ms = (t_pulse_end - t_pulse_start) * 1000.0

                    # Update pulse stats
                    with self._stats_lock:
                        self._stats['total_pulses'] += 2  # Count as 2 pulses

                    logger.info(
                        f"✅ PLC dual pulse complete: D{address1},D{address2} | "
                        f"Duration: {total_duration_ms:.2f}ms "
                        f"(write: {write_time_1 + write_time_0:.2f}ms, pulse: {pulse_ms:.0f}ms)"
                    )

                    return True, total_duration_ms

                else:
                    # Non-consecutive addresses, write separately (fast succession)
                    logger.info(f"🟡 PLC D{address1},D{address2} (non-consecutive, separate writes)")

                    # Step 1: Write both to 1 (fast succession)
                    success_1a, write_time_1a = self.write_register(address1, 1)
                    success_1b, write_time_1b = self.write_register(address2, 1)

                    if not (success_1a and success_1b):
                        logger.error(f"❌ Failed to activate D{address1},D{address2}")
                        return False, 0.0

                    logger.info(f"🔴 PLC D{address1},D{address2}=[1,1] (activate) | {write_time_1a + write_time_1b:.2f}ms")

                    # Step 2: Hold pulse
                    time.sleep(pulse_ms / 1000.0)

                    # Step 3: Write both to 0 (fast succession)
                    success_0a, write_time_0a = self.write_register(address1, 0)
                    success_0b, write_time_0b = self.write_register(address2, 0)

                    if not (success_0a and success_0b):
                        logger.error(f"❌ Failed to deactivate D{address1},D{address2}")
                        return False, 0.0

                    logger.info(f"⚪ PLC D{address1},D{address2}=[0,0] (deactivate) | {write_time_0a + write_time_0b:.2f}ms")

                    # Calculate total duration
                    t_pulse_end = time.perf_counter()
                    total_duration_ms = (t_pulse_end - t_pulse_start) * 1000.0

                    # Update pulse stats
                    with self._stats_lock:
                        self._stats['total_pulses'] += 2

                    logger.info(
                        f"✅ PLC dual pulse complete: D{address1},D{address2} | "
                        f"Duration: {total_duration_ms:.2f}ms"
                    )

                    return True, total_duration_ms

            except Exception as e:
                logger.error(f"❌ Exception during PLC dual pulse: {e}")
                import traceback
                traceback.print_exc()
                return False, 0.0

    def _update_stats(self, write_time_ms: float, success: bool):
        """Update write statistics"""
        with self._stats_lock:
            self._stats['total_writes'] += 1

            if success:
                self._stats['total_write_time_ms'] += write_time_ms

                # Update min/max
                if write_time_ms > self._stats['max_write_time_ms']:
                    self._stats['max_write_time_ms'] = write_time_ms
                if write_time_ms < self._stats['min_write_time_ms']:
                    self._stats['min_write_time_ms'] = write_time_ms

                # Update average
                successful_writes = self._stats['total_writes'] - self._stats['total_failures']
                if successful_writes > 0:
                    self._stats['avg_write_time_ms'] = (
                        self._stats['total_write_time_ms'] / successful_writes
                    )
            else:
                self._stats['total_failures'] += 1

    def get_stats(self) -> dict:
        """Get PLC statistics"""
        with self._stats_lock:
            stats = self._stats.copy()

            # Add connection status
            stats['connected'] = self.is_connected()
            stats['plc_ip'] = self.PLC_IP
            stats['plc_port'] = self.PLC_PORT

            return stats

    def reset_stats(self):
        """Reset statistics"""
        with self._stats_lock:
            self._stats = {
                'total_writes': 0,
                'total_pulses': 0,
                'total_failures': 0,
                'total_write_time_ms': 0.0,
                'avg_write_time_ms': 0.0,
                'max_write_time_ms': 0.0,
                'min_write_time_ms': float('inf')
            }
        logger.info("PLC statistics reset")
