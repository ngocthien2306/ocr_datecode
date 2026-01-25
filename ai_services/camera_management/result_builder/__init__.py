"""
Results Module

This module provides Builder pattern for constructing inference results:
- InferenceResultBuilder: Fluent builder for inference results
- FrameResultBuilder: Builder for frame-level results
- CameraResultBuilder: Builder for camera-level results
"""

from .builder import InferenceResultBuilder, FrameResultBuilder, CameraResultBuilder

__all__ = [
    'InferenceResultBuilder',
    'FrameResultBuilder',
    'CameraResultBuilder'
]
