"""
Inference Pipeline Module

This module provides Template Method pattern for inference pipelines:
- InferencePipelineTemplate: Abstract base class defining pipeline steps
- SingleCameraPipeline: Pipeline for single camera (single or multi-template)
- MultiCameraPipeline: Pipeline for multiple cameras with batch processing
"""

from .base import InferencePipelineTemplate, PipelineContext
from .single_camera import SingleCameraPipeline
from .multi_camera import MultiCameraPipeline

__all__ = [
    'InferencePipelineTemplate',
    'PipelineContext',
    'SingleCameraPipeline',
    'MultiCameraPipeline'
]
