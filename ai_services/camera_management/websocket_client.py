"""
WebSocket Client for CameraManagement Service
Handles bidirectional communication with Backend FastAPI server
"""

import asyncio
import json
import logging
from typing import Optional, Callable, Dict, Any
import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


class CameraWebSocketClient:
    """
    WebSocket client for camera management

    Features:
    - Auto-reconnect with exponential backoff
    - Heartbeat to detect disconnections
    - Event-driven message handling
    - Thread-safe operations
    """

    def __init__(
        self,
        server_url: str,
        message_callback: Optional[Callable[[Dict[str, Any]], asyncio.coroutine]] = None,
        heartbeat_interval: int = 30
    ):
        """
        Initialize WebSocket client

        Args:
            server_url: WebSocket server URL (e.g. "ws://localhost:8000/ws/camera-management")
            message_callback: Async callback for received messages
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.server_url = server_url
        self.message_callback = message_callback
        self.heartbeat_interval = heartbeat_interval

        self.ws: Optional[WebSocketClientProtocol] = None
        self.running = False
        self.connected = False

        # Reconnection settings
        self.reconnect_delay = 1  # Start with 1 second
        self.max_reconnect_delay = 60  # Max 60 seconds
        self.reconnect_backoff = 2  # Exponential backoff multiplier

    async def connect(self) -> bool:
        """
        Connect to WebSocket server

        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"Connecting to WebSocket server: {self.server_url}")

            self.ws = await websockets.connect(
                self.server_url,
                ping_interval=self.heartbeat_interval,
                ping_timeout=10
            )

            self.connected = True
            self.reconnect_delay = 1  # Reset backoff on successful connect

            logger.info("WebSocket connected successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket server: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.ws:
            try:
                await self.ws.close()
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting WebSocket: {e}")

        self.connected = False
        self.ws = None

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send message to server

        Args:
            message: Message dict (will be JSON serialized)

        Returns:
            True if sent successfully
        """
        if not self.ws or not self.connected:
            logger.warning("Cannot send message: not connected")
            return False

        try:
            message_json = json.dumps(message)
            await self.ws.send(message_json)
            logger.debug(f"Sent message: {message.get('event', 'unknown')}")
            return True

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.connected = False
            return False

    async def _receive_loop(self):
        """Main receive loop"""
        while self.running and self.ws:
            try:
                # Receive message
                message_raw = await self.ws.recv()
                message = json.loads(message_raw)

                event_type = message.get("event", "unknown")
                logger.debug(f"Received message: {event_type}")

                # Call message callback
                if self.message_callback:
                    try:
                        await self.message_callback(message)
                    except Exception as e:
                        logger.error(f"Error in message callback: {e}")

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.connected = False
                break

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")

            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                self.connected = False
                break

    async def _reconnect_loop(self):
        """Auto-reconnect loop"""
        while self.running:
            if not self.connected:
                logger.info(f"Attempting to reconnect in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)

                if await self.connect():
                    # Start receive loop
                    asyncio.create_task(self._receive_loop())

                    # Send connected event
                    await self.send_message({
                        "event": "service_connected",
                        "data": {
                            "service": "camera_management"
                        }
                    })

                else:
                    # Increase backoff delay
                    self.reconnect_delay = min(
                        self.reconnect_delay * self.reconnect_backoff,
                        self.max_reconnect_delay
                    )

            else:
                # Check connection every 1 second
                await asyncio.sleep(1)

    async def start(self):
        """Start WebSocket client"""
        self.running = True

        # Initial connection
        if await self.connect():
            # Start receive loop
            asyncio.create_task(self._receive_loop())

            # Send connected event
            await self.send_message({
                "event": "service_connected",
                "data": {
                    "service": "camera_management"
                }
            })

        # Start reconnect loop
        asyncio.create_task(self._reconnect_loop())

        logger.info("WebSocket client started")

    async def stop(self):
        """Stop WebSocket client"""
        self.running = False

        # Send disconnect event
        if self.connected:
            await self.send_message({
                "event": "service_disconnected",
                "data": {
                    "service": "camera_management"
                }
            })

        await self.disconnect()

        logger.info("WebSocket client stopped")

    async def wait_until_connected(self, timeout: float = 30.0) -> bool:
        """
        Wait until connected

        Args:
            timeout: Timeout in seconds

        Returns:
            True if connected within timeout
        """
        start_time = asyncio.get_event_loop().time()

        while not self.connected:
            if asyncio.get_event_loop().time() - start_time > timeout:
                return False

            await asyncio.sleep(0.1)

        return True
