import asyncio
import logging
import time
import psutil
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import deque

from ..models.jetson_monitoring import (
    SystemMetrics,
    CPUMetrics,
    GPUMetrics,
    RAMMetrics,
    DiskMetrics,
    PowerMetrics,
    Alert,
    MonitoringConfig
)

logger = logging.getLogger(__name__)

try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False
    logger.warning("jtop not available, falling back to psutil only")


class JetsonMonitoringService:
    def __init__(self):
        self.config = MonitoringConfig()
        self.current_metrics: Optional[SystemMetrics] = None
        self.active_alerts: deque = deque(maxlen=100)
        self.last_alert_time: Dict[str, datetime] = {}

        self.is_monitoring = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._socketio = None
        self._jetson = None

        self._disk_io_counters_prev = None
        self._disk_io_time_prev = None

        logger.info("JetsonMonitoringService initialized")

    def set_socketio(self, sio):
        """Set SocketIO instance for real-time updates"""
        self._socketio = sio
        logger.info("SocketIO instance set for JetsonMonitoringService")

    async def start_monitoring(self):
        """Start background monitoring task"""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return

        self.is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Jetson monitoring started")

    async def stop_monitoring(self):
        """Stop background monitoring task"""
        if not self.is_monitoring:
            logger.warning("Monitoring not running")
            return

        self.is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        if self._jetson:
            self._jetson.close()
            self._jetson = None

        logger.info("Jetson monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop using jtop"""
        logger.info("Monitoring loop started")

        if JTOP_AVAILABLE:
            await self._monitoring_loop_jtop()
        else:
            await self._monitoring_loop_psutil()

    async def _monitoring_loop_jtop(self):
        """Monitoring loop using jtop"""
        try:
            with jtop() as jetson:
                self._jetson = jetson
                logger.info("jtop initialized successfully")

                while self.is_monitoring and jetson.ok():
                    try:
                        if self.config.enabled:
                            metrics = self._collect_metrics_jtop(jetson)
                            self.current_metrics = metrics

                            alerts = self._check_alerts(metrics)
                            if alerts:
                                for alert in alerts:
                                    self.active_alerts.append(alert)

                            if self._socketio:
                                await self._emit_metrics(metrics, alerts)

                        await asyncio.sleep(self.config.update_interval_seconds)

                    except Exception as e:
                        logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                        await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}", exc_info=True)
            self.is_monitoring = False
        finally:
            self._jetson = None
            logger.info("Monitoring loop stopped")

    async def _monitoring_loop_psutil(self):
        """Fallback monitoring loop using psutil only"""
        logger.info("Using psutil fallback (jtop not available)")

        try:
            while self.is_monitoring:
                try:
                    if self.config.enabled:
                        metrics = self._collect_metrics_psutil()
                        self.current_metrics = metrics

                        alerts = self._check_alerts(metrics)
                        if alerts:
                            for alert in alerts:
                                self.active_alerts.append(alert)

                        if self._socketio:
                            await self._emit_metrics(metrics, alerts)

                    await asyncio.sleep(self.config.update_interval_seconds)

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}", exc_info=True)
            self.is_monitoring = False
        finally:
            logger.info("Monitoring loop stopped")

    def _collect_metrics_jtop(self, jetson) -> SystemMetrics:
        """Collect metrics using jtop"""
        stats = jetson.stats

        # CPU metrics
        per_core = {f"CPU{i}": stats.get(f"CPU{i}", 0) for i in range(1, 9) if f"CPU{i}" in stats}
        avg_cpu = sum(per_core.values()) / len(per_core) if per_core else 0

        cpu_freq = jetson.cpu.get("freq", {})
        cpu_freq_mhz = cpu_freq.get("cur", 0) if isinstance(cpu_freq, dict) else 0

        cpu_temp = stats.get("Temp cpu")

        cpu_metrics = CPUMetrics(
            usage_percent=avg_cpu,
            frequency_mhz=cpu_freq_mhz,
            temperature_celsius=cpu_temp,
            per_core_usage=per_core
        )

        # GPU metrics
        gpu_usage = stats.get("GPU", 0)
        gpu_freq = jetson.gpu.get("freq", {})
        gpu_freq_mhz = gpu_freq.get("cur", 0) if isinstance(gpu_freq, dict) else 0
        gpu_temp = stats.get("Temp gpu")

        gpu_metrics = GPUMetrics(
            usage_percent=gpu_usage,
            frequency_mhz=gpu_freq_mhz,
            temperature_celsius=gpu_temp
        )

        # RAM metrics
        ram_percent = stats.get("RAM", 0) * 100
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        ram_metrics = RAMMetrics(
            total_mb=mem.total / (1024 * 1024),
            used_mb=mem.used / (1024 * 1024),
            available_mb=mem.available / (1024 * 1024),
            usage_percent=ram_percent,
            swap_total_mb=swap.total / (1024 * 1024),
            swap_used_mb=swap.used / (1024 * 1024),
            swap_percent=swap.percent
        )

        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()

        read_speed = None
        write_speed = None
        if disk_io and self._disk_io_counters_prev and self._disk_io_time_prev:
            time_delta = time.time() - self._disk_io_time_prev
            if time_delta > 0:
                read_speed = (disk_io.read_bytes - self._disk_io_counters_prev.read_bytes) / time_delta / (1024 * 1024)
                write_speed = (disk_io.write_bytes - self._disk_io_counters_prev.write_bytes) / time_delta / (1024 * 1024)

        self._disk_io_counters_prev = disk_io
        self._disk_io_time_prev = time.time()

        disk_metrics = DiskMetrics(
            total_gb=disk.total / (1024 ** 3),
            used_gb=disk.used / (1024 ** 3),
            available_gb=disk.free / (1024 ** 3),
            usage_percent=disk.percent,
            read_speed_mbps=read_speed,
            write_speed_mbps=write_speed
        )

        # Power metrics
        power_metrics = PowerMetrics(
            total_mw=stats.get("Power TOT"),
            cpu_gpu_cv_mw=stats.get("Power VDD_CPU_GPU_CV"),
            soc_mw=stats.get("Power VDD_SOC")
        )

        # Uptime
        uptime = stats.get("uptime")
        uptime_seconds = uptime.total_seconds() if uptime else 0

        nvp_model = stats.get("nvp model")

        return SystemMetrics(
            cpu=cpu_metrics,
            gpu=gpu_metrics,
            ram=ram_metrics,
            disk=disk_metrics,
            power=power_metrics,
            uptime_seconds=uptime_seconds,
            nvp_model=nvp_model
        )

    def _collect_metrics_psutil(self) -> SystemMetrics:
        """Fallback: Collect metrics using psutil only"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        per_core_dict = {f"CPU{i+1}": val for i, val in enumerate(per_core)}
        cpu_freq = psutil.cpu_freq()

        cpu_metrics = CPUMetrics(
            usage_percent=cpu_percent,
            frequency_mhz=cpu_freq.current if cpu_freq else 0,
            temperature_celsius=None,
            per_core_usage=per_core_dict
        )

        # GPU
        gpu_metrics = GPUMetrics(
            usage_percent=0,
            frequency_mhz=0,
            temperature_celsius=None
        )

        # RAM
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        ram_metrics = RAMMetrics(
            total_mb=mem.total / (1024 * 1024),
            used_mb=mem.used / (1024 * 1024),
            available_mb=mem.available / (1024 * 1024),
            usage_percent=mem.percent,
            swap_total_mb=swap.total / (1024 * 1024),
            swap_used_mb=swap.used / (1024 * 1024),
            swap_percent=swap.percent
        )

        # Disk
        disk = psutil.disk_usage('/')
        disk_metrics = DiskMetrics(
            total_gb=disk.total / (1024 ** 3),
            used_gb=disk.used / (1024 ** 3),
            available_gb=disk.free / (1024 ** 3),
            usage_percent=disk.percent,
            read_speed_mbps=None,
            write_speed_mbps=None
        )

        uptime_seconds = time.time() - psutil.boot_time()

        return SystemMetrics(
            cpu=cpu_metrics,
            gpu=gpu_metrics,
            ram=ram_metrics,
            disk=disk_metrics,
            power=None,
            uptime_seconds=uptime_seconds,
            nvp_model=None
        )

    def _check_alerts(self, metrics: SystemMetrics) -> List[Alert]:
        """Check if any metrics exceed thresholds"""
        alerts = []
        thresholds = self.config.alert_thresholds
        now = datetime.now()

        checks = [
            ("cpu_usage", metrics.cpu.usage_percent, thresholds.cpu_percent,
             f"CPU usage is {metrics.cpu.usage_percent:.1f}%"),
            ("gpu_usage", metrics.gpu.usage_percent, thresholds.gpu_percent,
             f"GPU usage is {metrics.gpu.usage_percent:.1f}%"),
            ("ram_usage", metrics.ram.usage_percent, thresholds.ram_percent,
             f"RAM usage is {metrics.ram.usage_percent:.1f}%"),
            ("disk_usage", metrics.disk.usage_percent, thresholds.disk_percent,
             f"Disk usage is {metrics.disk.usage_percent:.1f}%"),
        ]

        if metrics.cpu.temperature_celsius and thresholds.cpu_temp_celsius:
            checks.append(("cpu_temp", metrics.cpu.temperature_celsius, thresholds.cpu_temp_celsius,
                          f"CPU temperature is {metrics.cpu.temperature_celsius:.1f}°C"))

        if metrics.gpu.temperature_celsius and thresholds.gpu_temp_celsius:
            checks.append(("gpu_temp", metrics.gpu.temperature_celsius, thresholds.gpu_temp_celsius,
                          f"GPU temperature is {metrics.gpu.temperature_celsius:.1f}°C"))

        for alert_type, current_value, threshold_value, message in checks:
            if current_value > threshold_value:
                if self._should_trigger_alert(alert_type, now):
                    severity = "critical" if current_value > threshold_value * 1.1 else "warning"

                    alert = Alert(
                        alert_type=alert_type,
                        severity=severity,
                        message=message,
                        current_value=current_value,
                        threshold_value=threshold_value
                    )
                    alerts.append(alert)
                    self.last_alert_time[alert_type] = now
                    logger.warning(f"Alert: {message} (threshold: {threshold_value})")

        return alerts

    def _should_trigger_alert(self, alert_type: str, now: datetime) -> bool:
        """Check if enough time passed since last alert"""
        if alert_type not in self.last_alert_time:
            return True

        time_since_last = (now - self.last_alert_time[alert_type]).total_seconds()
        return time_since_last >= self.config.alert_cooldown_seconds

    async def _emit_metrics(self, metrics: SystemMetrics, alerts: List[Alert]):
        """Emit metrics via SocketIO"""
        if not self._socketio:
            return

        try:
            await self._socketio.emit("jetson_metrics", metrics.model_dump(mode='json'))

            if alerts:
                await self._socketio.emit("jetson_alerts", {
                    "alerts": [alert.model_dump(mode='json') for alert in alerts]
                })

        except Exception as e:
            logger.error(f"Error emitting metrics: {e}")

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        return self.current_metrics

    def get_active_alerts(self, limit: int = 50) -> List[Alert]:
        return list(self.active_alerts)[-limit:]

    def get_config(self) -> MonitoringConfig:
        return self.config.model_copy()

    def update_config(self, **kwargs) -> MonitoringConfig:
        for key, value in kwargs.items():
            if value is not None and hasattr(self.config, key):
                setattr(self.config, key, value)
        return self.config.model_copy()

    def get_status(self) -> Dict[str, Any]:
        return {
            "monitoring_enabled": self.config.enabled,
            "update_interval_seconds": self.config.update_interval_seconds,
            "last_update": self.current_metrics.timestamp.isoformat() if self.current_metrics else None,
            "active_alerts_count": len(self.active_alerts),
            "task_alive": self._monitoring_task is not None and not self._monitoring_task.done() if self._monitoring_task else False,
            "jtop_available": JTOP_AVAILABLE
        }


jetson_monitoring_service = JetsonMonitoringService()
