"""
OCR Backend Strategy - Abstract Base Class

Defines the interface for OCR backends using Strategy Pattern.
All OCR implementations must inherit from this class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class OCRResult:
    """Result from OCR recognition"""
    text: str
    confidence: float

    def __iter__(self):
        """Allow unpacking as tuple: text, confidence = result"""
        return iter((self.text, self.confidence))


class OCRBackendStrategy(ABC):
    """
    Abstract base class for OCR backends.

    Implements Strategy Pattern to allow swapping between different
    OCR implementations (TensorRT, ONNX, etc.) at runtime.

    All implementations must provide:
    - recognize(): Single image recognition
    - recognize_batch(): Batch image recognition
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend (e.g., 'tensorrt', 'onnx')"""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available and properly initialized"""
        pass

    @abstractmethod
    def recognize(
        self,
        image: np.ndarray,
        return_confidence: bool = True
    ) -> Tuple[str, float]:
        """
        Recognize text from a single image.

        Args:
            image: Input image (BGR numpy array)
            return_confidence: Whether to return confidence score

        Returns:
            Tuple of (recognized_text, confidence_score)
            If return_confidence=False, confidence will be 0.0
        """
        pass

    @abstractmethod
    def recognize_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        Recognize text from multiple images in a single batch.

        Args:
            images: List of input images (BGR numpy arrays)

        Returns:
            List of (recognized_text, confidence_score) tuples
        """
        pass

    def preprocess(self, image: np.ndarray, target_height: int = 48) -> np.ndarray:
        """
        Preprocess image for OCR model.

        Default implementation - can be overridden by subclasses.

        Args:
            image: Input image (BGR or grayscale)
            target_height: Target height for resizing

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        import cv2

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize to target height while maintaining aspect ratio
        h, w = gray.shape
        ratio = target_height / h
        new_w = max(int(w * ratio), 10)

        resized = cv2.resize(gray, (new_w, target_height))

        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5

        # Convert to 3 channels
        tensor = np.stack([normalized, normalized, normalized], axis=0).astype(np.float32)
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        return tensor

    def decode_ctc_with_confidence(
        self,
        preds: np.ndarray,
        char_list: List[str]
    ) -> Tuple[str, float]:
        """
        Decode CTC predictions with confidence score.

        Args:
            preds: Model predictions [timesteps, num_classes]
            char_list: List of characters (with 'blank' at index 0)

        Returns:
            Tuple of (decoded_text, average_confidence)
        """
        pred = preds[0] if len(preds.shape) == 3 else preds

        best_path = np.argmax(pred, axis=-1)

        # Collect confidences
        all_confidences = []
        for t, char_idx in enumerate(best_path):
            all_confidences.append(float(pred[t, char_idx]))

        # Decode text (remove blanks and duplicates)
        decoded_chars = []
        prev_idx = -1
        for char_idx in best_path:
            if char_idx != 0 and char_idx != prev_idx:
                if char_idx < len(char_list):
                    decoded_chars.append(char_list[char_idx])
            prev_idx = char_idx

        text = ''.join(decoded_chars)
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0

        return text, float(avg_confidence)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backend={self.backend_name}, available={self.is_available})"
