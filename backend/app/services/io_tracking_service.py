"""
I/O Tracking Service
Continuously polls DI/DO pins and emits updates via SocketIO
"""

import asyncio
import logging
import time
from typing import Optional

from app.utils.io_utils import read_all_di_pins, read_all_do_pins

logger = logging.getLogger(__name__)


class IOTrackingService:
    """
    Service to track I/O pin status in real-time

    Continuously reads DI/DO pins and emits updates via SocketIO
    """

    def __init__(self):
        self.is_tracking = False
        self.tracking_task: Optional[asyncio.Task] = None
        self.polling_interval = 0.1  # 100ms = 10Hz
        logger.info("IOTrackingService initialized")

    async def start_tracking(self):
        """Start I/O tracking loop"""
        if self.is_tracking:
            logger.warning("I/O tracking already running")
            return

        self.is_tracking = True
        self.tracking_task = asyncio.create_task(self._tracking_loop())
        logger.info("I/O tracking started")

    async def stop_tracking(self):
        """Stop I/O tracking loop"""
        if not self.is_tracking:
            logger.warning("I/O tracking not running")
            return

        self.is_tracking = False

        if self.tracking_task:
            self.tracking_task.cancel()
            try:
                await self.tracking_task
            except asyncio.CancelledError:
                pass
            self.tracking_task = None

        logger.info("I/O tracking stopped")

    async def _tracking_loop(self):
        """Main tracking loop - polls I/O and emits updates"""
        try:
            from app.services.socketio_service import emit_io_status_update

            while self.is_tracking:
                # Read all pins
                di_values = read_all_di_pins()
                do_values = read_all_do_pins()

                # Emit to frontend
                await emit_io_status_update({
                    'di': di_values,
                    'do': do_values,
                    'timestamp': time.time()
                })

                # Wait before next poll
                await asyncio.sleep(self.polling_interval)

        except asyncio.CancelledError:
            logger.info("I/O tracking loop cancelled")
        except Exception as e:
            logger.error(f"Error in I/O tracking loop: {e}")
            import traceback
            traceback.print_exc()
            self.is_tracking = False


# Global service instance
io_tracking_service = IOTrackingService()
