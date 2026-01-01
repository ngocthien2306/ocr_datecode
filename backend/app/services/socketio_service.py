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


@sio.event
async def connect(sid, environ, auth):
    """Handle client connection"""
    logger.info(f"Client connected: {sid}")
    await sio.emit('connection_response', {'status': 'connected', 'sid': sid}, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {sid}")


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


# Export sio instance for use in other modules
__all__ = ['sio', 'socket_app', 'emit_inference_result']
