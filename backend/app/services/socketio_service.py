"""
SocketIO Service for Real-time Frontend Updates
"""

import logging
import socketio

logger = logging.getLogger(__name__)

# Create SocketIO server instance
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',  # Configure based on your CORS requirements
    logger=True,
    engineio_logger=True
)

# Wrap with ASGI app
socket_app = socketio.ASGIApp(sio)

# Import camera stream service
from app.services.camera_stream_service import camera_stream_service


@sio.event
async def connect(sid, environ, auth):
    """Handle client connection"""
    logger.info(f"Client connected: {sid}")
    await sio.emit('connection_response', {'status': 'connected', 'sid': sid}, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {sid}")

    # Clean up camera stream subscriptions - check all cameras
    for serial_number in list(camera_stream_service.stream_subscribers.keys()):
        if sid in camera_stream_service.stream_subscribers[serial_number]:
            camera_stream_service.remove_subscriber(serial_number, sid)

            # Stop streaming if no more subscribers
            if not camera_stream_service.has_subscribers(serial_number):
                await camera_stream_service.stop_streaming(serial_number)


@sio.event
async def subscribe_inference_results(sid, data=None):
    """Client subscribes to inference result updates"""
    logger.info(f"Client {sid} subscribed to inference_results")
    await sio.enter_room(sid, 'inference_results')
    await sio.emit('subscribed', {'channel': 'inference_results'}, room=sid)


@sio.event
async def unsubscribe_inference_results(sid, data=None):
    """Client unsubscribes from inference result updates"""
    logger.info(f"Client {sid} unsubscribed from inference_results")
    await sio.leave_room(sid, 'inference_results')


async def emit_inference_result(result_data: dict):
    """
    Emit new inference result to all subscribed clients

    Args:
        result_data: Inference result data to broadcast
    """
    try:
        # Create a log-friendly version without image_base64
        log_data = {
            'id': result_data.get('id'),
            'recipe_id': result_data.get('recipe_id'),
            'recipe_name': result_data.get('recipe_name'),
            'product_pass_fail': result_data.get('product_pass_fail'),
            'camera_count': len(result_data.get('camera_results', [])),
            'timestamp': result_data.get('timestamp')
        }
        logger.info(f"Broadcasting inference result: {log_data}")
        await sio.emit('new_inference_result', result_data, room='inference_results')
    except Exception as e:
        logger.error(f"Error emitting inference result: {e}")


@sio.event
async def start_camera_stream(sid, data):
    """Client requests to start camera stream"""
    try:
        serial_number = data.get('serial_number')
        frame_rate = data.get('frame_rate', 10)

        logger.info(f"Client {sid} requesting camera stream for {serial_number}")

        # Add subscriber
        camera_stream_service.add_subscriber(serial_number, sid)

        # Start streaming if not already active
        if serial_number not in camera_stream_service.active_streams:
            await camera_stream_service.start_streaming(serial_number, frame_rate)

        await sio.emit('camera_stream_started', {'serial_number': serial_number}, room=sid)

    except Exception as e:
        logger.error(f"Error starting camera stream: {e}")
        await sio.emit('camera_stream_error', {'error': str(e)}, room=sid)


@sio.event
async def stop_camera_stream(sid, data):
    """Client requests to stop camera stream"""
    try:
        serial_number = data.get('serial_number')

        logger.info(f"Client {sid} stopping camera stream for {serial_number}")

        # Remove subscriber
        camera_stream_service.remove_subscriber(serial_number, sid)

        # Stop streaming if no more subscribers
        if not camera_stream_service.has_subscribers(serial_number):
            await camera_stream_service.stop_streaming(serial_number)

        await sio.emit('camera_stream_stopped', {'serial_number': serial_number}, room=sid)

    except Exception as e:
        logger.error(f"Error stopping camera stream: {e}")
        await sio.emit('camera_stream_error', {'error': str(e)}, room=sid)


async def emit_camera_frame(frame_data: dict):
    """
    Emit camera frame to subscribed clients

    Args:
        frame_data: Frame data
            {
                'serial_number': str,
                'frame_base64': str,
                'timestamp': float,
                'frame_idx': int
            }
    """
    try:
        serial_number = frame_data.get('serial_number')

        # Send to all clients subscribed to this camera
        for sid in camera_stream_service.get_subscribers(serial_number):
            await sio.emit('camera_frame', frame_data, room=sid)

    except Exception as e:
        logger.error(f"Error emitting camera frame: {e}")


async def emit_recipe_status_change(status_data: dict):
    """
    Emit recipe status change to all connected clients

    Args:
        status_data: Recipe status data to broadcast
            {
                'event': 'recipe_loaded' | 'recipe_stopped',
                'recipe_id': str,
                'recipe_name': str,
                ...
            }
    """
    try:
        event_type = status_data.get('event', 'recipe_status_change')
        logger.info(f"Broadcasting recipe status change: {event_type} - {status_data.get('recipe_name')}")
        await sio.emit('recipe_status_change', status_data)
    except Exception as e:
        logger.error(f"Error emitting recipe status change: {e}")


# Export sio instance for use in other modules
__all__ = ['sio', 'socket_app', 'emit_inference_result', 'emit_recipe_status_change']
