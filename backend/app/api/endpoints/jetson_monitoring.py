import logging
from fastapi import APIRouter, HTTPException, Query

from ...schemas.jetson_monitoring import (
    SystemMetricsResponse,
    AlertsResponse,
    MonitoringConfigResponse,
    UpdateMonitoringConfigRequest,
    UpdateMonitoringConfigResponse,
    MonitoringStatusResponse
)
from ...services.jetson_monitoring_service import jetson_monitoring_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/jetson-monitoring/metrics", response_model=SystemMetricsResponse)
async def get_current_metrics():
    """
    Get current Jetson system metrics (CPU, GPU, RAM, Disk).

    Returns real-time metrics including:
    - CPU usage, frequency, temperature
    - GPU usage, frequency, temperature
    - RAM usage, available, swap
    - Disk usage, I/O speed
    """
    try:
        metrics = jetson_monitoring_service.get_current_metrics()

        if not metrics:
            raise HTTPException(
                status_code=503,
                detail="Monitoring service not ready. Metrics not available yet."
            )

        return SystemMetricsResponse(
            success=True,
            data=metrics,
            message="Current system metrics retrieved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve system metrics: {str(e)}"
        )


@router.get("/jetson-monitoring/alerts", response_model=AlertsResponse)
async def get_alerts(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of alerts to return")
):
    """
    Get recent alerts triggered by threshold violations.

    Returns alerts when metrics exceed configured thresholds.
    """
    try:
        alerts = jetson_monitoring_service.get_active_alerts(limit=limit)

        return AlertsResponse(
            success=True,
            data=alerts,
            count=len(alerts),
            message=f"Retrieved {len(alerts)} recent alerts"
        )

    except Exception as e:
        logger.error(f"Error getting alerts: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve alerts: {str(e)}"
        )


@router.get("/jetson-monitoring/config", response_model=MonitoringConfigResponse)
async def get_monitoring_config():
    """
    Get current monitoring configuration.

    Returns:
    - Monitoring enabled/disabled status
    - Update interval
    - Alert thresholds for CPU, GPU, RAM, Disk, temperatures
    - Alert cooldown period
    """
    try:
        config = jetson_monitoring_service.get_config()

        return MonitoringConfigResponse(
            success=True,
            data=config,
            message="Monitoring configuration retrieved successfully"
        )

    except Exception as e:
        logger.error(f"Error getting config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve monitoring configuration: {str(e)}"
        )


@router.put("/jetson-monitoring/config", response_model=UpdateMonitoringConfigResponse)
async def update_monitoring_config(request: UpdateMonitoringConfigRequest):
    """
    Update monitoring configuration.

    Allows updating:
    - Enable/disable monitoring
    - Update interval (0.5s - 60s)
    - Alert thresholds
    - Alert cooldown period (10s - 600s)
    """
    try:
        update_dict = request.model_dump(exclude_none=True)

        if "alert_thresholds" in update_dict and update_dict["alert_thresholds"]:
            thresholds_dict = update_dict["alert_thresholds"].model_dump(exclude_none=True)
            current_config = jetson_monitoring_service.get_config()
            current_thresholds = current_config.alert_thresholds.model_dump()

            current_thresholds.update(thresholds_dict)

            from ...models.jetson_monitoring import AlertThreshold
            updated_thresholds = AlertThreshold(**current_thresholds)
            update_dict["alert_thresholds"] = updated_thresholds

        updated_config = jetson_monitoring_service.update_config(**update_dict)

        return UpdateMonitoringConfigResponse(
            success=True,
            data=updated_config,
            message="Monitoring configuration updated successfully"
        )

    except Exception as e:
        logger.error(f"Error updating config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update monitoring configuration: {str(e)}"
        )


@router.get("/jetson-monitoring/status", response_model=MonitoringStatusResponse)
async def get_monitoring_status():
    """
    Get monitoring service status.

    Returns:
    - Whether monitoring is enabled
    - Update interval
    - Last update timestamp
    - Active alerts count
    - Thread health status
    """
    try:
        status = jetson_monitoring_service.get_status()

        return MonitoringStatusResponse(
            success=True,
            data=status,
            message="Monitoring service status retrieved successfully"
        )

    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve monitoring status: {str(e)}"
        )


@router.post("/jetson-monitoring/start")
async def start_monitoring():
    """
    Start the monitoring service background task.

    This will begin collecting and emitting metrics at the configured interval.
    """
    try:
        await jetson_monitoring_service.start_monitoring()

        return {
            "success": True,
            "message": "Monitoring service started successfully"
        }

    except Exception as e:
        logger.error(f"Error starting monitoring: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start monitoring service: {str(e)}"
        )


@router.post("/jetson-monitoring/stop")
async def stop_monitoring():
    """
    Stop the monitoring service background task.

    This will stop collecting and emitting metrics.
    """
    try:
        await jetson_monitoring_service.stop_monitoring()

        return {
            "success": True,
            "message": "Monitoring service stopped successfully"
        }

    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop monitoring service: {str(e)}"
        )
