"""
TensorRT-based Text Recognition for Camera Management
Optimized for real-time inference in production environment
Compatible with TensorRT 8.x, 9.x, and 10.x+
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
    - CTC decoding with confidence scores
    - Batch processing support
    - High performance (~5-10ms per image)
    - Compatible with TensorRT 8.x, 9.x, and 10.x+
    - Auto-padding for character dictionary
    """

    def __init__(self, engine_path: str, dict_path: str, min_width: int = 50, max_width: int = 2000):
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

        # Detect TensorRT version and use appropriate API
        self.use_new_api = hasattr(self.engine, 'num_io_tensors')
        
        # Get input/output binding info
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.input_idx = 0
        self.output_idx = 1

        if self.use_new_api:
            # TensorRT 10.x+ API
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                mode = self.engine.get_tensor_mode(name)
                
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                    self.input_shape = shape
                    self.input_idx = i
                else:
                    self.output_name = name
                    self.output_shape = shape
                    self.output_idx = i
        else:
            # TensorRT 8.x/9.x API (legacy)
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                is_input = self.engine.binding_is_input(i)

                if is_input:
                    self.input_name = name
                    self.input_shape = shape
                    self.input_idx = i
                else:
                    self.output_name = name
                    self.output_shape = shape
                    self.output_idx = i

        # Auto-pad character list if model expects more classes
        expected_classes = self.output_shape[-1] if isinstance(self.output_shape[-1], int) else None
        
        if expected_classes and len(self.char_list) < expected_classes:
            num_missing = expected_classes - len(self.char_list)
            self.char_list.extend([' ' for i in range(num_missing)])
            logger.warning(f"⚠️  Padded {num_missing} unknown tokens to match model output")

        # CUDA stream for async operations
        self.stream = cuda.Stream()

        logger.info(f"✅ TextRecognizerTRT initialized")
        logger.info(f"   TensorRT API: {'10.x+' if self.use_new_api else '8.x/9.x'}")
        logger.info(f"   Engine: {Path(engine_path).name}")
        logger.info(f"   Dictionary: {len(self.char_dict)} characters")
        logger.info(f"   char_list length: {len(self.char_list)}")
        logger.info(f"   Model output classes: {expected_classes}")
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

        if self.use_new_api:
            # TensorRT 10.x+ API
            # Set dynamic input shape
            self.context.set_input_shape(self.input_name, input_shape)
            
            # Allocate input buffer
            input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
            input_size = trt.volume(input_shape)
            input_host = cuda.pagelocked_empty(input_size, input_dtype)
            input_device = cuda.mem_alloc(input_host.nbytes)
            inputs.append(HostDeviceMem(input_host, input_device))
            self.context.set_tensor_address(self.input_name, int(input_device))
            
            # Allocate output buffer
            output_shape = self.context.get_tensor_shape(self.output_name)
            output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
            output_size = trt.volume(output_shape)
            output_host = cuda.pagelocked_empty(output_size, output_dtype)
            output_device = cuda.mem_alloc(output_host.nbytes)
            outputs.append(HostDeviceMem(output_host, output_device))
            self.context.set_tensor_address(self.output_name, int(output_device))
            
        else:
            # TensorRT 8.x/9.x API (legacy)
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

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Apply softmax to logits
        
        Args:
            x: Logits array [..., num_classes]
            
        Returns:
            Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def decode_ctc(self, preds: np.ndarray) -> str:
        """
        Decode CTC output to text (basic version without confidence)

        Args:
            preds: Model output [batch_size, sequence_length, num_classes]

        Returns:
            Decoded text string
        """
        indices = np.argmax(preds, axis=2)[0]
        
        chars = []
        prev_idx = -1
        
        for idx in indices:
            if idx != 0 and idx != prev_idx:
                if idx < len(self.char_list):
                    chars.append(self.char_list[idx])
            prev_idx = idx
        
        return ''.join(chars)

    def decode_ctc_with_confidence(self, preds: np.ndarray) -> tuple:
        """
        Decode CTC output to text with confidence scores

        NOTE: This implementation matches ONNX text_recognizer.py for consistency:
        - Uses raw logits (not softmax) for confidence calculation
        - Collects confidence for all timesteps (not just decoded chars)

        Args:
            preds: Model output [batch_size, sequence_length, num_classes]

        Returns:
            Tuple of (text, average_confidence)
        """
        pred = preds[0]  # Shape: (timesteps, num_classes)

        best_path = np.argmax(pred, axis=-1)

        # Collect ALL confidences (same as ONNX)
        all_confidences = []
        for t, char_idx in enumerate(best_path):
            all_confidences.append(float(pred[t, char_idx]))

        # Decode text (same as ONNX)
        decoded_chars = []
        prev_idx = -1
        for char_idx in best_path:
            if char_idx != 0 and char_idx != prev_idx:
                if char_idx < len(self.char_list):
                    decoded_chars.append(self.char_list[char_idx])
            prev_idx = char_idx

        text = ''.join(decoded_chars)
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0

        return text, float(avg_confidence)

    def recognize_with_char_conf(self, image: np.ndarray) -> tuple:
        """
        Recognize text và trả về per-character confidence (chỉ alnum).

        Returns:
            (text, avg_conf, char_confs)
            char_confs: list of (char, conf) — chỉ chữ/số, bỏ ký tự đặc biệt
        """
        tensor = self.preprocess(image)
        input_shape = tensor.shape

        inputs, outputs, bindings = self._allocate_buffers(input_shape)
        np.copyto(inputs[0].host, tensor.ravel())
        cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, self.stream)

        if self.use_new_api:
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, self.stream)
        self.stream.synchronize()

        if self.use_new_api:
            output_shape = self.context.get_tensor_shape(self.output_name)
        else:
            output_shape = self.context.get_binding_shape(1)
        if isinstance(output_shape, tuple):
            output_shape = list(output_shape)

        preds = outputs[0].host.reshape(output_shape)
        pred  = preds[0]  # (T, C)

        best_path = np.argmax(pred, axis=-1)
        best_prob = pred[np.arange(len(pred)), best_path]

        decoded_chars, char_confs = [], []
        prev_idx = -1
        for t, char_idx in enumerate(best_path):
            if char_idx != 0 and char_idx != prev_idx:
                if char_idx < len(self.char_list):
                    label = self.char_list[char_idx]
                    decoded_chars.append(label)
                    if label.isalnum():
                        char_confs.append((label, float(best_prob[t])))
            prev_idx = char_idx

        text     = ''.join(decoded_chars)
        avg_conf = float(np.mean(best_prob)) if len(best_prob) > 0 else 0.0
        return text, avg_conf, char_confs

    def recognize(self, image: np.ndarray, return_confidence: bool = True):
        """
        Recognize text from cropped image using TensorRT

        Args:
            image: Cropped text region (BGR or grayscale)
            return_confidence: Return confidence score

        Returns:
            If return_confidence=True: tuple of (text, confidence)
            If return_confidence=False: text string only
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

        if self.use_new_api:
            # TensorRT 10.x+ API
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            # TensorRT 8.x/9.x API (legacy)
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        # Transfer predictions back from device
        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, self.stream)

        # Synchronize stream
        self.stream.synchronize()

        # Reshape output
        if self.use_new_api:
            output_shape = self.context.get_tensor_shape(self.output_name)
        else:
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

        # Decode with or without confidence
        if return_confidence:
            text, confidence = self.decode_ctc_with_confidence(preds)
            return text, confidence
        else:
            text = self.decode_ctc(preds)
            return text

    def recognize_batch(self, images: list) -> list:
        """
        Recognize text from multiple cropped images

        Note: Processes images sequentially due to variable widths.
        Still faster than ONNX due to TensorRT optimization.

        Args:
            images: List of cropped images

        Returns:
            List of tuples: [(text, confidence), ...]
        """
        results = []
        for img in images:
            text, confidence, char_confs = self.recognize_with_char_conf(img)
            results.append({"text": text, "confidence": confidence, "char_confs": char_confs})
        return results


# Singleton instance for production use
_text_recognizer_instance = None


def get_text_recognizer(
    engine_path: str = None,
    dict_path: str = None,
    min_width: int = 50,
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


# Demo/Testing
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python text_recognizer_trt.py <engine_path> <dict_path> [test_image]")
        sys.exit(1)
    
    engine_path = sys.argv[1]
    dict_path = sys.argv[2]
    test_image_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Initialize recognizer
    recognizer = TextRecognizerTRT(
        engine_path=engine_path,
        dict_path=dict_path
    )
    
    if test_image_path:
        # Test with sample image
        test_image = cv2.imread(test_image_path)
        
        if test_image is not None:
            # Crop a region (example)
            h, w = test_image.shape[:2]
            cropped = test_image[h//4:h//2, w//4:3*w//4]
            
            # Recognize with confidence
            text, confidence = recognizer.recognize(cropped, return_confidence=True)
            print(f"\nRecognized: '{text}'")
            print(f"Confidence: {confidence:.3f}")
            
            # Show result
            cv2.imshow('Cropped Region', cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Could not load image: {test_image_path}")
    else:
        print("✅ Recognizer initialized successfully. Provide test image path to test.")