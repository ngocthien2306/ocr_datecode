"""
Matchers Module

This module provides factory pattern for creating template matchers:
- MatcherFactory: Factory for creating SuperPointMatcherTRT instances
- AnnotationParser: Parse and convert annotations to pixel coordinates
- MatcherConfig: Configuration dataclass for matcher creation
"""

from .factory import MatcherFactory
from .annotation_parser import AnnotationParser
from .config import MatcherConfig, CropArea, BoundingBox

__all__ = [
    'MatcherFactory',
    'AnnotationParser',
    'MatcherConfig',
    'CropArea',
    'BoundingBox'
]
