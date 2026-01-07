"""
TensorRT-based Text Recognition for Camera Management
Optimized for real-time inference in production environment
"""
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # Initialize CUDA context
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT/PyCUDA not available. Text recognition will be disabled.")


class TextRecognizerTRT:
    """
    TensorRT-based text recognizer for production use

    Features:
    - Dynamic shape support (variable width text regions)
    - CTC decoding
    - Batch processing support
    - High performance (~5-10ms per image)
    """

    def __init__(self, engine_path: str, dict_path: str, min_width: int = 320, max_width: int = 2000):
        """
        Initialize TensorRT text recognizer

        Args:
            engine_path: Path to rec.engine (TensorRT engine file)
            dict_path: Path to dict.txt (character dictionary)
            min_width: Minimum width (must match minShapes in engine)
            max_width: Maximum width (must match maxShapes in engine)
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT is not available. Please install tensorrt and pycuda.")

        self.engine_path = engine_path
        self.dict_path = dict_path
        self.min_width = min_width
        self.max_width = max_width

        # Load character dictionary
        self.char_dict = self._load_dict(dict_path)
        self.char_list = ['blank'] + self.char_dict  # CTC blank token at index 0

        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Get input/output binding info
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            is_input = self.engine.binding_is_input(i)

            if is_input:
                self.input_name = name
                self.input_shape = shape
            else:
                self.output_name = name
                self.output_shape = shape

        # CUDA stream for async operations
        self.stream = cuda.Stream()

        logger.info(f"✅ TextRecognizerTRT initialized")
        logger.info(f"   Engine: {Path(engine_path).name}")
        logger.info(f"   Dictionary: {len(self.char_dict)} characters")
        logger.info(f"   Input: {self.input_name} {self.input_shape}")
        logger.info(f"   Output: {self.output_name} {self.output_shape}")
        logger.info(f"   Width range: {min_width} - {max_width}px")

    def _load_dict(self, dict_path: str) -> list:
        """Load character dictionary from file"""
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f.readlines()]
        return chars

    def _allocate_buffers(self, input_shape: tuple):
        """
        Allocate input/output buffers for specific input shape

        Args:
            input_shape: Input tensor shape [1, 3, H, W]

        Returns:
            (inputs, outputs, bindings) tuple
        """
        from .ocr_utils import HostDeviceMem

        inputs = []
        outputs = []
        bindings = []

        # Set dynamic input shape
        self.context.set_binding_shape(0, input_shape)

        for i in range(self.engine.num_bindings):
            # Get binding shape (use execution context for dynamic shapes)
            if self.engine.binding_is_input(i):
                shape = input_shape
            else:
                shape = self.context.get_binding_shape(i)

            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings

    def preprocess(self, image: np.ndarray, target_height: int = 48) -> np.ndarray:
        """
        Preprocess image for recognition model

        Args:
            image: Input image (BGR or grayscale)
            target_height: Target height (default: 48)

        Returns:
            Preprocessed tensor [1, 3, 48, W]
        """
        from .ocr_utils import preprocess_text_image

        return preprocess_text_image(
            image=image,
            target_height=target_height,
            min_width=self.min_width,
            max_width=self.max_width
        )

    def decode_ctc(self, preds: np.ndarray) -> str:
        """
        Decode CTC output to text

        Args:
            preds: Model output [batch_size, sequence_length, num_classes]

        Returns:
            Decoded text string
        """
        from .ocr_utils import decode_ctc_greedy

        return decode_ctc_greedy(preds, self.char_list)

    def recognize(self, image: np.ndarray, return_confidence: bool = True):
        """
        Recognize text from cropped image using TensorRT

        Args:
            image: Cropped text region (BGR or grayscale)
            return_confidence: Return confidence score

        Returns:
            dict with keys:
                - text: Recognized text string
                - confidence: Average confidence score (if return_confidence=True)
        """
        # Preprocess
        tensor = self.preprocess(image)
        input_shape = tensor.shape  # [1, 3, 48, W]

        # Allocate buffers for this specific input shape
        inputs, outputs, bindings = self._allocate_buffers(input_shape)

        # Copy input data to host buffer
        np.copyto(inputs[0].host, tensor.ravel())

        # Transfer input data to device
        cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        # Transfer predictions back from device
        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, self.stream)

        # Synchronize stream
        self.stream.synchronize()

        # Reshape output
        output_shape = self.context.get_binding_shape(1)

        # Handle tuple output shape
        if isinstance(output_shape, tuple):
            output_shape = list(output_shape)

        # Reshape and decode
        try:
            preds = outputs[0].host.reshape(output_shape)
        except Exception as e:
            logger.error(f"Error reshaping output: {e}")
            logger.error(f"Output shape: {output_shape}, Host buffer size: {outputs[0].host.shape}")
            raise

        # Decode
        text = self.decode_ctc(preds)

        result = {'text': text}

        if return_confidence:
            # Calculate average confidence from softmax probabilities
            from .ocr_utils import calculate_confidence
            confidence = calculate_confidence(preds[0])
            result['confidence'] = float(confidence)

        return result

    def recognize_batch(self, images: list) -> list:
        """
        Recognize text from multiple cropped images

        Note: Processes images sequentially due to variable widths.
        Still faster than ONNX due to TensorRT optimization.

        Args:
            images: List of cropped images

        Returns:
            List of result dicts with 'text' and 'confidence' keys
        """
        results = []
        for img in images:
            result = self.recognize(img, return_confidence=True)
            results.append(result)
        return results


# Singleton instance for production use
_text_recognizer_instance = None


def get_text_recognizer(
    engine_path: str = None,
    dict_path: str = None,
    min_width: int = 320,
    max_width: int = 2000
) -> TextRecognizerTRT:
    """
    Get or create singleton TextRecognizerTRT instance

    Args:
        engine_path: Path to TensorRT engine (required on first call)
        dict_path: Path to character dictionary (required on first call)
        min_width: Minimum width
        max_width: Maximum width

    Returns:
        TextRecognizerTRT instance
    """
    global _text_recognizer_instance

    if _text_recognizer_instance is None:
        if not TENSORRT_AVAILABLE:
            logger.error("TensorRT not available")
            return None

        if engine_path is None or dict_path is None:
            raise ValueError("engine_path and dict_path required for first call")

        _text_recognizer_instance = TextRecognizerTRT(
            engine_path=engine_path,
            dict_path=dict_path,
            min_width=min_width,
            max_width=max_width
        )

    return _text_recognizer_instance
