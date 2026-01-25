"""
OCR Backend Module

This module provides a Strategy Pattern for OCR backends:
- OCRBackendStrategy: Abstract base class defining the interface
- TensorRTOCRBackend: TensorRT-based implementation (high performance)
- ONNXOCRBackend: ONNX-based implementation (cross-platform)
- OCRBackendFactory: Factory for creating OCR backends
"""

from .base import OCRBackendStrategy, OCRResult
from .factory import OCRBackendFactory, OCRBackendType

__all__ = [
    'OCRBackendStrategy',
    'OCRResult',
    'OCRBackendFactory',
    'OCRBackendType'
]
