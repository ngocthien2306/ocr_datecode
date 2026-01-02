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

# Track which client is viewing which camera
live_view_subscriptions = {}  # {sid: serial_number}


@sio.event
async def connect(sid, environ, auth):
    """Handle client connection"""
    logger.info(f"Client connected: {sid}")
    await sio.emit('connection_response', {'status': 'connected', 'sid': sid}, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {sid}")

    # Clean up live view subscriptions
    if sid in live_view_subscriptions:
        serial_number = live_view_subscriptions[sid]
        del live_view_subscriptions[sid]

        # Check if anyone else is viewing this camera
        other_viewers = [s for s, cam in live_view_subscriptions.items() if cam == serial_number]

        if not other_viewers:
            # No one else viewing, stop streaming
            from app.api.websocket.camera_ws import send_stop_live_view
            try:
                await send_stop_live_view(serial_number)
                logger.info(f"Stopped live view for {serial_number} (no more viewers)")
            except Exception as e:
                logger.error(f"Error stopping live view: {e}")


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
        logger.info(f"Broadcasting inference result: {result_data.get('id')}")
        await sio.emit('new_inference_result', result_data, room='inference_results')
    except Exception as e:
        logger.error(f"Error emitting inference result: {e}")


@sio.event
async def start_live_view(sid, data):
    """Client requests to start live view for a camera"""
    try:
        serial_number = data.get('serial_number')
        frame_rate = data.get('frame_rate', 10)

        logger.info(f"Client {sid} requesting live view for {serial_number}")

        # Track subscription
        live_view_subscriptions[sid] = serial_number

        # Forward command to Camera Service
        from app.api.websocket.camera_ws import send_start_live_view
        success = await send_start_live_view(serial_number, frame_rate)

        if success:
            await sio.emit('live_view_started', {'serial_number': serial_number}, room=sid)
        else:
            await sio.emit('live_view_error', {'error': 'Failed to start live view'}, room=sid)

    except Exception as e:
        logger.error(f"Error starting live view: {e}")
        await sio.emit('live_view_error', {'error': str(e)}, room=sid)


@sio.event
async def stop_live_view(sid, data):
    """Client requests to stop live view"""
    try:
        serial_number = data.get('serial_number')

        logger.info(f"Client {sid} stopping live view for {serial_number}")

        # Remove subscription
        live_view_subscriptions.pop(sid, None)

        # Check if anyone else is viewing this camera
        other_viewers = [s for s, cam in live_view_subscriptions.items() if cam == serial_number]

        if not other_viewers:
            # No one else viewing, stop streaming
            from app.api.websocket.camera_ws import send_stop_live_view
            await send_stop_live_view(serial_number)
            logger.info(f"Stopped streaming for {serial_number} (no more viewers)")

        await sio.emit('live_view_stopped', {'serial_number': serial_number}, room=sid)

    except Exception as e:
        logger.error(f"Error stopping live view: {e}")
        await sio.emit('live_view_error', {'error': str(e)}, room=sid)


async def emit_live_frame(frame_data: dict):
    """
    Emit live frame to clients viewing this camera

    Args:
        frame_data: Frame data from Camera Service
            {
                'serial_number': str,
                'frame_base64': str,
                'timestamp': float,
                'resolution': [width, height]
            }
    """
    try:
        serial_number = frame_data.get('serial_number')

        # Send to all clients viewing this camera
        for sid, cam in live_view_subscriptions.items():
            if cam == serial_number:
                await sio.emit('live_frame', frame_data, room=sid)

    except Exception as e:
        logger.error(f"Error emitting live frame: {e}")


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
