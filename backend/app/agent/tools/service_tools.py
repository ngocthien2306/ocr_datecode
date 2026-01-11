"""
Service Management Tools
Tools for managing AI services (camera_management, etc.)
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from app.agent.tools.base_tool import BaseTool, ToolMetadata, ToolRegistry
from app.api.websocket.camera_ws import camera_ws_manager
import subprocess
import psutil
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Argument Schemas
# ============================================================================

class CheckServiceStatusArgs(BaseModel):
    """Arguments for check_service_status tool"""
    service_name: str = Field(
        default="camera_management",
        description="Name of the service to check (camera_management, inference, etc.)"
    )


class ServiceActionArgs(BaseModel):
    """Arguments for start/stop service tools"""
    service_name: str = Field(
        description="Name of the service (camera_management, inference, etc.)"
    )


class GetLogsArgs(BaseModel):
    """Arguments for get_service_logs tool"""
    service_name: str = Field(
        description="Name of the service"
    )
    lines: int = Field(
        default=50,
        description="Number of log lines to return (max 500)",
        ge=1,
        le=500
    )


# ============================================================================
# Tool Implementation Functions
# ============================================================================

def check_service_status(service_name: str = "camera_management") -> Dict[str, Any]:
    """
    Check if a service is running and its connection status

    Args:
        service_name: Name of the service to check

    Returns:
        dict with status information
    """
    logger.info(f"Checking status for service: {service_name}")

    if service_name == "camera_management":
        # Check WebSocket connection
        ws_connected = camera_ws_manager.is_connected()

        # Check if process is running
        is_running = False
        pid = None
        cpu_percent = None
        memory_mb = None

        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'camera_management_service.py' in ' '.join(cmdline):
                        is_running = True
                        pid = proc.info['pid']

                        # Get resource usage
                        process = psutil.Process(pid)
                        cpu_percent = process.cpu_percent(interval=0.1)
                        memory_mb = process.memory_info().rss / 1024 / 1024

                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error checking process: {e}")

        # Determine overall status
        if is_running and ws_connected:
            status = "healthy"
            message = "Service đang chạy và đã kết nối WebSocket ✅"
        elif is_running and not ws_connected:
            status = "degraded"
            message = "Service đang chạy nhưng WebSocket chưa kết nối (có thể đang khởi động) ⚠️"
        else:
            status = "stopped"
            message = "Service không chạy ❌"

        return {
            "service_name": service_name,
            "is_running": is_running,
            "pid": pid,
            "websocket_connected": ws_connected,
            "status": status,
            "message": message,
            "cpu_percent": cpu_percent,
            "memory_mb": round(memory_mb, 2) if memory_mb else None
        }

    return {
        "error": f"Unknown service: {service_name}",
        "available_services": ["camera_management"]
    }


def start_service(service_name: str) -> Dict[str, Any]:
    """
    Start a service

    Args:
        service_name: Name of the service to start

    Returns:
        dict with result
    """
    logger.info(f"Starting service: {service_name}")

    if service_name == "camera_management":
        # Check if already running
        status = check_service_status(service_name)
        if status.get("is_running"):
            return {
                "success": False,
                "message": f"Service đã chạy rồi (PID: {status['pid']}) ⚠️",
                "pid": status['pid']
            }

        # Get script path
        project_root = Path(__file__).parent.parent.parent.parent.parent
        script_path = project_root / "ai_services" / "camera_management_service.py"

        if not script_path.exists():
            return {
                "success": False,
                "message": f"Không tìm thấy script service: {script_path}"
            }

        try:
            # Start service as background process
            log_dir = project_root / "ai_services" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "camera_management.log"

            process = subprocess.Popen(
                ["python3", str(script_path)],
                cwd=str(script_path.parent),
                stdout=open(log_file, 'a'),
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent
                env=os.environ.copy()
            )

            logger.info(f"Started service with PID: {process.pid}")

            return {
                "success": True,
                "message": f"✅ Service đã khởi động thành công!\n📝 Logs: {log_file}",
                "pid": process.pid,
                "log_file": str(log_file)
            }

        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            return {
                "success": False,
                "message": f"❌ Lỗi khi khởi động service: {str(e)}"
            }

    return {
        "success": False,
        "error": f"Unknown service: {service_name}",
        "available_services": ["camera_management"]
    }


def stop_service(service_name: str) -> Dict[str, Any]:
    """
    Stop a running service

    Args:
        service_name: Name of the service to stop

    Returns:
        dict with result
    """
    logger.info(f"Stopping service: {service_name}")

    if service_name == "camera_management":
        # Find the process
        found = False

        try:
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'camera_management_service.py' in ' '.join(cmdline):
                        pid = proc.info['pid']
                        found = True

                        # Graceful shutdown
                        process = psutil.Process(pid)
                        process.terminate()  # SIGTERM

                        # Wait for shutdown (max 5 seconds)
                        try:
                            process.wait(timeout=5)
                            logger.info(f"Service stopped gracefully (PID: {pid})")
                            return {
                                "success": True,
                                "message": f"✅ Service đã dừng thành công (PID: {pid})",
                                "pid": pid
                            }
                        except psutil.TimeoutExpired:
                            # Force kill if not responding
                            process.kill()  # SIGKILL
                            process.wait()
                            logger.warning(f"Service force killed (PID: {pid})")
                            return {
                                "success": True,
                                "message": f"⚠️ Service đã bị force kill (không phản hồi shutdown, PID: {pid})",
                                "pid": pid
                            }

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error stopping service: {e}")
            return {
                "success": False,
                "message": f"❌ Lỗi khi dừng service: {str(e)}"
            }

        if not found:
            return {
                "success": False,
                "message": "Service không chạy ⚠️"
            }

    return {
        "success": False,
        "error": f"Unknown service: {service_name}",
        "available_services": ["camera_management"]
    }


def get_service_logs(service_name: str, lines: int = 50) -> Dict[str, Any]:
    """
    Get recent logs from a service

    Args:
        service_name: Name of the service
        lines: Number of lines to return

    Returns:
        dict with logs
    """
    logger.info(f"Getting logs for service: {service_name} (last {lines} lines)")

    if service_name == "camera_management":
        project_root = Path(__file__).parent.parent.parent.parent.parent
        log_file = project_root / "ai_services" / "logs" / "camera_management.log"

        if not log_file.exists():
            return {
                "logs": [],
                "message": "Chưa có file log (service có thể chưa từng chạy) 📝",
                "log_file": str(log_file)
            }

        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

            return {
                "logs": [line.strip() for line in last_lines],
                "total_lines": len(all_lines),
                "returned_lines": len(last_lines),
                "log_file": str(log_file),
                "message": f"📝 Đọc được {len(last_lines)} dòng log (tổng {len(all_lines)} dòng)"
            }

        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return {
                "error": f"Lỗi khi đọc logs: {str(e)}"
            }

    return {
        "error": f"Unknown service: {service_name}",
        "available_services": ["camera_management"]
    }


# ============================================================================
# Create and Register Tools
# ============================================================================

# Check service status tool
check_service_status_tool = BaseTool.create_tool(
    func=check_service_status,
    metadata=ToolMetadata(
        name="check_service_status",
        description=(
            "Kiểm tra xem service có đang chạy không và trạng thái kết nối WebSocket. "
            "Trả về thông tin process (PID, CPU, RAM) và trạng thái WebSocket. "
            "LUÔN LUÔN dùng tool này TRƯỚC KHI thực hiện bất kỳ hành động nào khác."
        ),
        category="service",
        requires_approval=False
    ),
    args_schema=CheckServiceStatusArgs
)

# Start service tool
start_service_tool = BaseTool.create_tool(
    func=start_service,
    metadata=ToolMetadata(
        name="start_service",
        description=(
            "Khởi động một service. "
            "QUAN TRỌNG: PHẢI check status trước khi start. "
            "PHẢI hỏi xác nhận người dùng trước khi thực hiện."
        ),
        category="service",
        requires_approval=True
    ),
    args_schema=ServiceActionArgs
)

# Stop service tool
stop_service_tool = BaseTool.create_tool(
    func=stop_service,
    metadata=ToolMetadata(
        name="stop_service",
        description=(
            "Dừng một service đang chạy (graceful shutdown). "
            "QUAN TRỌNG: PHẢI hỏi xác nhận người dùng trước khi thực hiện. "
            "Hành động này sẽ disconnect tất cả cameras và dừng inference."
        ),
        category="service",
        requires_approval=True
    ),
    args_schema=ServiceActionArgs
)

# Get logs tool
get_service_logs_tool = BaseTool.create_tool(
    func=get_service_logs,
    metadata=ToolMetadata(
        name="get_service_logs",
        description=(
            "Lấy các dòng log gần nhất từ service. "
            "Hữu ích để debug lỗi hoặc kiểm tra xem service đang làm gì. "
            "Trả về N dòng cuối cùng của file log."
        ),
        category="service",
        requires_approval=False
    ),
    args_schema=GetLogsArgs
)

# Register all tools
ToolRegistry.register(check_service_status_tool)
ToolRegistry.register(start_service_tool)
ToolRegistry.register(stop_service_tool)
ToolRegistry.register(get_service_logs_tool)

logger.info("✅ Service tools registered")
