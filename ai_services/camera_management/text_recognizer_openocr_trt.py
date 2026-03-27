"""
OpenOCR RepSVTR TensorRT with Pre-allocated Buffers
Optimized for real-time inference - No CUDA context conflict
Pattern: Pre-allocate buffers like YOLO OBB (no dynamic allocation)
Performance: ~127 imgs/sec (batch=4), ~8ms per image
Compatible with TensorRT 8.x, 9.x, and 10.x+
"""
import cv2
import numpy as np
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # Initialize CUDA context
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT/PyCUDA not available. OpenOCR TensorRT will be disabled.")


class TextRecognizerOpenOCRTRT:
    """
    OpenOCR RepSVTR TensorRT recognizer with pre-allocated buffers.

    Key Features:
    - Pre-allocated CUDA buffers (like YOLO OBB) - NO dynamic allocation
    - No CUDA context conflict with SuperPoint TensorRT
    - Ultra-fast performance (~127 imgs/sec in batch mode)
    - Batch processing support (dynamic batch 1-16)
    - CTC decoding with confidence scores
    - Direct numpy array input (no temp files needed)

    Memory Pattern:
    - Buffers allocated ONCE in __init__()
    - Buffers REUSED for every inference
    - Buffers freed only on object destruction

    Model: OpenOCR RepSVTR (9.77x faster than ONNX)
    """

    def __init__(self, engine_path: str, dict_path: str, use_space_char: bool = True):
        """
        Initialize OpenOCR TensorRT recognizer with pre-allocated buffers.

        Args:
            engine_path: Path to .engine file (TensorRT engine)
            dict_path: Path to character dictionary
            use_space_char: Add space character to dictionary
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT is not available. Please install tensorrt and pycuda.")

        self.engine_path = engine_path
        self.dict_path = dict_path

        # Load character dictionary
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = f.read().splitlines()

        if use_space_char:
            chars.append(' ')

        self.character = ['blank'] + chars

        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Detect TensorRT version
        self.use_new_api = hasattr(self.engine, 'num_io_tensors')

        # Get input/output info
        self._init_tensor_info()

        # Get batch range from engine profile
        if -1 in self.input_shape:
            profile = self.engine.get_tensor_profile_shape(self.input_name, 0)
            self.min_batch = profile[0][0]
            self.max_batch = profile[2][0]
            logger.info(f"Dynamic batch range: {self.min_batch} - {self.max_batch}")
        else:
            self.min_batch = self.input_shape[0]
            self.max_batch = self.input_shape[0]
            logger.info(f"Fixed batch size: {self.max_batch}")

        # PRE-ALLOCATE CUDA buffers (like YOLO OBB) ✅
        self._allocate_persistent_buffers()

        logger.info(f"✅ TextRecognizerOpenOCRTRT initialized")
        logger.info(f"   TensorRT API: {'10.x+' if self.use_new_api else '8.x/9.x'}")
        logger.info(f"   Engine: {Path(engine_path).name}")
        logger.info(f"   Dictionary: {len(self.character)} characters")
        logger.info(f"   Batch range: {self.min_batch}-{self.max_batch}")

    def _init_tensor_info(self):
        """Initialize tensor names and shapes"""
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None

        if self.use_new_api:
            # TensorRT 10.x+ API
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                mode = self.engine.get_tensor_mode(name)

                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                    self.input_shape = shape
                else:
                    self.output_name = name
                    self.output_shape = shape
        else:
            # TensorRT 8.x/9.x API (legacy)
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

    def _allocate_persistent_buffers(self):
        """
        Pre-allocate CUDA buffers (ONE-TIME, like YOLO OBB).
        Buffers persist for the lifetime of the object.

        Memory Layout (max size):
        - Input: [max_batch, 3, 48, 320] = typical text region size
        - Output: [max_batch, 25, 97] = sequence length x num_classes
        """
        # Input buffer dimensions (typical text region)
        max_height = 48
        max_width = 320

        input_shape = (self.max_batch, 3, max_height, max_width)
        input_size = int(np.prod(input_shape))

        # Allocate pinned host memory (faster H2D transfer)
        self.h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)

        # Allocate device memory
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)

        # Output buffer dimensions
        # OpenOCR output: [batch, 25, 97] for RepSVTR
        max_seq_len = 25
        num_classes = len(self.character)
        output_shape = (self.max_batch, max_seq_len, num_classes)
        output_size = int(np.prod(output_shape))

        # Allocate pinned host memory
        self.h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)

        # Allocate device memory
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        # Create CUDA stream for async operations
        self.stream = cuda.Stream()

        logger.info(f"✅ Pre-allocated CUDA buffers (persistent, like YOLO OBB):")
        logger.info(f"   Input:  {self.h_input.nbytes / 1024 / 1024:.2f} MB")
        logger.info(f"   Output: {self.h_output.nbytes / 1024 / 1024:.2f} MB")
        logger.info(f"   Total:  {(self.h_input.nbytes + self.h_output.nbytes) / 1024 / 1024:.2f} MB")

    def preprocess_numpy(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess numpy array image directly (OpenOCR RepSVTR format).

        Args:
            image: numpy array (BGR or RGB or Grayscale), shape [H, W] or [H, W, 3]

        Returns:
            numpy array [1, 3, 48, W_resized]
        """
        # Handle grayscale
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR input (OpenCV default)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image

        h, w = img_rgb.shape[:2]

        # Resize to height=48, maintain aspect ratio
        target_h = 48
        ratio = w / float(h)
        target_w = int(target_h * ratio)

        # Limit width to reasonable range
        target_w = max(10, min(target_w, 320))

        # Resize
        img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Normalize: [0, 255] -> [0, 1] -> [-1, 1]
        img_array = img_resized.astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5

        # HWC to CHW
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension
        img_array = img_array[np.newaxis, ...]

        return img_array.astype(np.float32)

    def postprocess(self, output: np.ndarray) -> dict:
        """
        Decode CTC output to text (OpenOCR RepSVTR).

        Args:
            output: model output [1, seq_len, num_classes]

        Returns:
            dict: {'text': str, 'confidence': float}
        """
        # Get predictions
        preds_idx = np.argmax(output, axis=2)[0]
        preds_prob = np.max(output, axis=2)[0]

        # CTC decode
        char_list = []
        conf_list = []
        last_idx = 0

        for i, idx in enumerate(preds_idx):
            if idx != 0 and idx != last_idx:  # Skip blank and repeats
                if idx < len(self.character):
                    char_list.append(self.character[idx])
                    conf_list.append(preds_prob[i])
            last_idx = idx

        text = ''.join(char_list)
        confidence = float(np.mean(conf_list)) if conf_list else 0.0
        char_confs = [(c, cf) for c, cf in zip(char_list, conf_list) if c.isalnum()]

        return {'text': text, 'confidence': confidence, 'char_confs': char_confs}

    def recognize_with_char_conf(self, image: np.ndarray) -> tuple:
        """
        Recognize text và trả về per-character confidence (chỉ alnum).

        Returns:
            (text, avg_conf, char_confs)
            char_confs: list of (char, conf) — chỉ chữ/số, bỏ ký tự đặc biệt
        """
        batch_size = 1
        preprocessed = self.preprocess_numpy(image)
        max_width = preprocessed.shape[3]

        batch_data = np.zeros((1, 3, 48, max_width), dtype=np.float32)
        batch_data[0, :, :, :max_width] = preprocessed[0]

        if self.use_new_api:
            self.context.set_input_shape(self.input_name, batch_data.shape)
        else:
            self.context.set_binding_shape(0, batch_data.shape)

        np.copyto(self.h_input[:batch_data.size], batch_data.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        if self.use_new_api:
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(
                bindings=[int(self.d_input), int(self.d_output)],
                stream_handle=self.stream.handle
            )

        self.stream.synchronize()

        if self.use_new_api:
            output_shape = self.context.get_tensor_shape(self.output_name)
        else:
            output_shape = self.context.get_binding_shape(1)

        output_size = int(np.prod(output_shape))
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        output = self.h_output[:output_size].reshape(output_shape)  # (1, T, C)

        preds_idx  = np.argmax(output, axis=2)[0]
        preds_prob = np.max(output, axis=2)[0]

        char_list, conf_list = [], []
        last_idx = 0
        for i, idx in enumerate(preds_idx):
            if idx != 0 and idx != last_idx:
                if idx < len(self.character):
                    label = self.character[idx]
                    char_list.append(label)
                    conf_list.append(float(preds_prob[i]))
            last_idx = idx

        text     = ''.join(char_list)
        avg_conf = float(np.mean(conf_list)) if conf_list else 0.0
        char_confs = [(c, cf) for c, cf in zip(char_list, conf_list) if c.isalnum()]
        return text, avg_conf, char_confs

    def recognize(self, image: np.ndarray, return_confidence: bool = True) -> tuple:
        """
        Run OCR on single image.

        Args:
            image: numpy array (BGR format from OpenCV)
            return_confidence: Return confidence score

        Returns:
            tuple: (text, confidence) if return_confidence=True
        """
        result = self.recognize_batch([image], use_true_batch=True)[0]

        if return_confidence:
            return (result['text'], result['confidence'])
        else:
            return (result['text'],)

    def recognize_batch(self, images: list, use_true_batch: bool = True) -> list:
        """
        Run OCR on batch of images using PRE-ALLOCATED buffers.

        Args:
            images: List of numpy arrays (BGR format)
            use_true_batch: Use true batch inference (default: True)

        Returns:
            List of dicts: [{'text': str, 'confidence': float}, ...]
        """
        batch_size = len(images)

        # Check batch limit
        if batch_size < self.min_batch or batch_size > self.max_batch:
            raise ValueError(
                f"Batch size {batch_size} out of range [{self.min_batch}, {self.max_batch}]"
            )

        # Preprocess all images
        preprocessed = []
        for img in images:
            if isinstance(img, np.ndarray):
                img_array = self.preprocess_numpy(img)
            else:
                raise ValueError("Only numpy arrays supported")
            preprocessed.append(img_array)

        # Find max width in batch
        max_width = max(img.shape[3] for img in preprocessed)
        height = 48

        # Create padded batch
        batch_data = np.zeros((batch_size, 3, height, max_width), dtype=np.float32)
        for i, img in enumerate(preprocessed):
            w = img.shape[3]
            batch_data[i, :, :, :w] = img[0]

        input_shape = batch_data.shape

        # Set dynamic input shape
        if self.use_new_api:
            self.context.set_input_shape(self.input_name, input_shape)
        else:
            self.context.set_binding_shape(0, input_shape)

        # REUSE pre-allocated buffers (NO NEW ALLOCATION!) ✅
        np.copyto(self.h_input[:batch_data.size], batch_data.ravel())

        # Copy to device (async)
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        # Run inference with timing
        start_time = time.perf_counter()

        if self.use_new_api:
            # TensorRT 10.x+ API
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            # TensorRT 8.x/9.x API (legacy)
            self.context.execute_async_v2(
                bindings=[int(self.d_input), int(self.d_output)],
                stream_handle=self.stream.handle
            )

        self.stream.synchronize()
        inference_time = time.perf_counter() - start_time

        # Get output shape
        if self.use_new_api:
            output_shape = self.context.get_tensor_shape(self.output_name)
        else:
            output_shape = self.context.get_binding_shape(1)

        output_size = int(np.prod(output_shape))

        # Copy output to host (async)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        # Extract results
        output_data = self.h_output[:output_size].reshape(output_shape)

        # NO FREE - buffers are persistent! ✅

        # Postprocess each result in batch
        results = []

        for i in range(batch_size):
            single_pred = output_data[i:i+1]
            result = self.postprocess(single_pred)
            result['time'] = inference_time / batch_size
            results.append(result)

        return results

    def __del__(self):
        """Cleanup CUDA buffers on object destruction"""
        try:
            if hasattr(self, 'd_input'):
                self.d_input.free()
            if hasattr(self, 'd_output'):
                self.d_output.free()
            logger.debug("OpenOCR TensorRT CUDA buffers freed")
        except Exception as e:
            logger.debug(f"Error freeing CUDA buffers: {e}")
