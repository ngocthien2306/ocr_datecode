"""
SocketIO Service for Real-time Frontend Updates
"""

import asyncio
import logging
import socketio

logger = logging.getLogger(__name__)

# Create SocketIO server instance
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',  # Configure based on your CORS requirements
    logger=False,  # Disable verbose socketio logging (hides base64 data in console)
    engineio_logger=False  # Disable engineio logging
)

# Wrap with ASGI app
socket_app = socketio.ASGIApp(sio)

# Import camera stream service
from app.services.camera_stream_service import camera_stream_service

# Import and initialize Jetson monitoring service
from app.services.jetson_monitoring_service import jetson_monitoring_service
jetson_monitoring_service.set_socketio(sio)


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
        save_enabled = data.get('save_enabled', False)

        logger.info(f"Client {sid} requesting camera stream for {serial_number} (save_enabled={save_enabled})")

        # Add subscriber
        camera_stream_service.add_subscriber(serial_number, sid)

        # Start streaming if not already active
        if serial_number not in camera_stream_service.active_streams:
            await camera_stream_service.start_streaming(serial_number, frame_rate, save_enabled)
        else:
            # Update save_enabled flag for existing stream
            camera_stream_service.save_enabled[serial_number] = save_enabled
            logger.info(f"Updated save_enabled={save_enabled} for existing stream {serial_number}")

        await sio.emit('camera_stream_started', {
            'serial_number': serial_number,
            'save_enabled': save_enabled
        }, room=sid)

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


async def emit_camera_service_status(status_data: dict):
    """
    Emit camera service connection status to all connected clients

    Args:
        status_data: Service status data
            {
                'connected': bool,
                'message': str
            }
    """
    try:
        logger.info(f"Broadcasting camera service status: {status_data}")
        await sio.emit('camera_service_status', status_data)
    except Exception as e:
        logger.error(f"Error emitting camera service status: {e}")


async def emit_camera_status_update(camera_data: dict):
    """
    Emit individual camera connection status update to all connected clients

    Args:
        camera_data: Camera status data
            {
                'serial_number': str,
                'camera_id': str,
                'is_connected': bool,
                'event': 'camera_connected' | 'camera_disconnected'
            }
    """
    try:
        logger.info(f"Broadcasting camera status update: {camera_data.get('serial_number')} - connected={camera_data.get('is_connected')}")
        await sio.emit('camera_status_update', camera_data)
    except Exception as e:
        logger.error(f"Error emitting camera status update: {e}")


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


@sio.event
async def request_io_status(sid, data=None):
    """Client requests I/O status tracking"""
    try:
        logger.info(f"Client {sid} requesting I/O status tracking")

        # Add to I/O tracking room
        await sio.enter_room(sid, 'io_tracking')

        # Start I/O tracking service
        from app.services.io_tracking_service import io_tracking_service
        await io_tracking_service.start_tracking()

        await sio.emit('io_tracking_started', {'status': 'tracking'}, room=sid)

    except Exception as e:
        logger.error(f"Error starting I/O tracking: {e}")
        await sio.emit('io_tracking_error', {'error': str(e)}, room=sid)


@sio.event
async def stop_io_tracking(sid, data=None):
    """Client stops I/O status tracking"""
    try:
        logger.info(f"Client {sid} stopping I/O tracking")

        # Leave I/O tracking room
        await sio.leave_room(sid, 'io_tracking')

        # Check if any clients still tracking
        room_clients = sio.manager.rooms.get('/').get('io_tracking', set())
        if not room_clients:
            # No more clients tracking, stop I/O service
            from app.services.io_tracking_service import io_tracking_service
            await io_tracking_service.stop_tracking()

    except Exception as e:
        logger.error(f"Error stopping I/O tracking: {e}")


@sio.event
async def set_do_pin(sid, data):
    """Client sets Digital Output pin value"""
    try:
        pin_number = data.get('pin_number')
        value = data.get('value')

        logger.info(f"Client {sid} setting DO{pin_number} to {value}")

        # Write to DO pin. io_utils.write_do_pin is a sync subprocess call
        # (~1-3s); run it in a thread so it doesn't block the event loop and
        # freeze every other SocketIO client.
        from app.utils.io_utils import write_do_pin
        success = await asyncio.to_thread(write_do_pin, pin_number, value)

        if success:
            await sio.emit('do_pin_set_success', {'pin_number': pin_number, 'value': value}, room=sid)
        else:
            await sio.emit('do_pin_set_error', {'error': 'Failed to set DO pin'}, room=sid)

    except Exception as e:
        logger.error(f"Error setting DO pin: {e}")
        await sio.emit('do_pin_set_error', {'error': str(e)}, room=sid)


async def emit_io_status_update(io_data: dict):
    """
    Emit I/O status update to all tracking clients

    Args:
        io_data: I/O status data
            {
                'di': [0, 1, 0, 1],  # DI0-DI3 values
                'do': [0, 0, 1, 0],  # DO0-DO3 values
                'timestamp': 1234567890.123
            }
    """
    try:
        await sio.emit('io_status_update', io_data, room='io_tracking')
    except Exception as e:
        logger.error(f"Error emitting I/O status update: {e}")


@sio.event
async def subscribe_jetson_monitoring(sid, data=None):
    """Client subscribes to Jetson monitoring updates"""
    logger.info(f"Client {sid} subscribed to jetson_monitoring")
    await sio.enter_room(sid, 'jetson_monitoring')
    await sio.emit('subscribed', {'channel': 'jetson_monitoring'}, room=sid)


@sio.event
async def unsubscribe_jetson_monitoring(sid, data=None):
    """Client unsubscribes from Jetson monitoring updates"""
    logger.info(f"Client {sid} unsubscribed from jetson_monitoring")
    await sio.leave_room(sid, 'jetson_monitoring')


# Export sio instance for use in other modules
__all__ = [
    'sio',
    'socket_app',
    'emit_inference_result',
    'emit_recipe_status_change',
    'emit_io_status_update',
    'jetson_monitoring_service'
]
