"""
Verification Services Module

This module contains services for verifying inference results:
- TextVerificationService: OCR-based text verification
- TemplateVerificationService: Template matching verification
- ProductVerificationService: Product alignment verification
"""

from .text_verifier import TextVerificationService
from .template_verifier import TemplateVerificationService
from .product_verifier import ProductVerificationService
from .wrinkle_segmenter import WrinkledSegmenterTRT
from .ml_classifier import MLClassifierService

__all__ = [
    'TextVerificationService',
    'TemplateVerificationService',
    'ProductVerificationService',
    'WrinkledSegmenterTRT',
    'MLClassifierService',
]
