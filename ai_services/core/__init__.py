"""
AI Services Core Module
Camera management and OCR processing for industrial automation
"""

from .camera import BaslerCamera
from .camera_manager import CameraManager


__all__ = [
    'BaslerCamera',
    'CameraManager',

]

__version__ = '1.0.0'
