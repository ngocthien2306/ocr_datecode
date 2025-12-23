"""
AI Services Core Module
Camera management and OCR processing for industrial automation
"""

from .camera import BaslerCamera
from .camera_manager import CameraManager
from .socket_manager import SocketManager, SimpleWebSocketManager

__all__ = [
    'BaslerCamera',
    'CameraManager',
    'SocketManager',
    'SimpleWebSocketManager',
]

__version__ = '1.0.0'
