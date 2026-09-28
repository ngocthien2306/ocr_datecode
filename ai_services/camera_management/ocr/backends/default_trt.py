"""
TensorRT-based Text Recognition for Camera Management
Optimized for real-time inference in production environment
Compatible with TensorRT 8.x, 9.x, and 10.x+
"""
import cv2
import functools
import numpy as np
import threading
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


_primary_ctx = None
_primary_ctx_lock = threading.Lock()


def _get_primary_context():
    """One retained handle to device 0's primary context, shared by every thread.

    ``pycuda.autoinit`` calls ``dev.make_context()``, which pushes a context on
    the *importing* thread only. This backend is built on InferenceWorker_0 —
    which pushes ``retain_primary_context()`` in
    ``inference_handler._initialize_cuda_context_for_thread`` — but its OCR
    calls are dispatched from the ``verify_phase`` pool, whose threads hold no
    context at all. Every per-call ``pagelocked_empty``/``mem_alloc`` in
    ``_allocate_buffers`` then died with "explicit_context_dependent failed:
    invalid device context", the whole batch returned empty, and only the
    rescue pass (which does run on InferenceWorker_0) ever read any text.

    Handing back a single shared handle, rather than calling
    retain_primary_context() per thread, is deliberate: pycuda records the
    building thread on each Context object and frees allocations against it, so
    per-thread handles leak buffers allocated on pool workers that later exit.
    Same reasoning as anomaly_inference._get_primary_context.
    """
    global _primary_ctx
    if _primary_ctx is None:
        with _primary_ctx_lock:
            if _primary_ctx is None:
                cuda.init()  # idempotent
                _primary_ctx = cuda.Device(0).retain_primary_context()
    return _primary_ctx


def _bind_cuda_context(fn):
    """Remember, on the instance, which CUDA context the engine is built under.

    It must be *that* context the inference calls push later, not simply the
    primary one: ``pycuda.autoinit`` calls ``dev.make_context()``, which is a
    different context from ``retain_primary_context()`` (verified — different
    handles), and a TensorRT execution context plus its device buffers are only
    valid in the context they were created in. In production this object is
    built on InferenceWorker_0, which holds the primary context; a bare script
    that just imports the module builds it under autoinit's. Binding covers
    both instead of guessing.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        ctx = cuda.Context.get_current()
        pushed = False
        if ctx is None:
            # Built on a thread with no context of its own — adopt the shared
            # primary one and hold it only for the duration of __init__.
            ctx = _get_primary_context()
            ctx.push()
            pushed = True
        self._cuda_ctx = ctx
        try:
            return fn(self, *args, **kwargs)
        finally:
            if pushed:
                ctx.pop()
    return wrapper


def _with_cuda_context(fn):
    """Push the engine's own context for the call and pop it again — never
    "push and leave", even on a persistent pool thread: a thread dying with a
    pushed but unpopped context makes pycuda hard-abort the process on pool
    shutdown. Nesting is legal, so on threads that already hold the context
    this costs a pair of no-op stack operations.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        ctx = getattr(self, "_cuda_ctx", None) or _get_primary_context()
        ctx.push()
        try:
            return fn(self, *args, **kwargs)
        finally:
            ctx.pop()
    return wrapper


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

    @_bind_cuda_context
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

        # The execution context and the buffers below are shared instance
        # state now that they are allocated once, so only one forward pass may
        # be in flight at a time. Batches interleave safely: _infer() copies
        # its output out before releasing, so it is atomic per image.
        self._lock = threading.RLock()

        self._alloc_max_buffers()

        logger.info(f"✅ TextRecognizerTRT initialized")
        logger.info(f"   TensorRT API: {'10.x+' if self.use_new_api else '8.x/9.x'}")
        logger.info(f"   Engine: {Path(engine_path).name}")
        logger.info(f"   Dictionary: {len(self.char_dict)} characters")
        logger.info(f"   char_list length: {len(self.char_list)}")
        logger.info(f"   Model output classes: {expected_classes}")
        logger.info(f"   Input: {self.input_name} {self.input_shape}")
        logger.info(f"   Output: {self.output_name} {self.output_shape}")
        logger.info(f"   Width range: {self.min_width} - {self.max_width}px")

    def _load_dict(self, dict_path: str) -> list:
        """Load character dictionary from file"""
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f.readlines()]
        return chars

    def _alloc_max_buffers(self):
        """Allocate host/device buffers once, sized for the engine's largest
        profile shape, and bind them to the execution context for good.

        This used to be a per-call ``_allocate_buffers(input_shape)``: one
        ``pagelocked_empty`` + ``mem_alloc`` pair for *every* text region, i.e.
        thousands of cudaHostAllocs an hour on the line, each freed only
        whenever the GC happened to reach it. Pinned-memory churn at that rate
        is what fragments the host allocator, and the allocation is also the
        only part of inference that needs a live CUDA context — plain
        ``memcpy_*_async``/``synchronize`` do not (measured). Every sibling
        backend already pre-allocates (svtrv2_trt.py:86, openocr_trt.py:164);
        this brings PaddleV5 in line with them.

        Sizing comes from the engine's own optimization profile rather than the
        constructor's ``max_width``, so a mismatched config can no longer ask
        for a shape the engine cannot build.
        """
        if self.use_new_api:
            prof = self.engine.get_tensor_profile_shape(self.input_name, 0)
            min_in, max_in = tuple(prof[0]), tuple(prof[-1])
            # preprocess() pads/resizes into [min_width, max_width], so keeping
            # those inside the profile keeps every set_input_shape() valid.
            self.min_width = max(self.min_width, int(min_in[-1]))
            self.max_width = min(self.max_width, int(max_in[-1]))

            self.context.set_input_shape(self.input_name, max_in)
            max_out = tuple(self.context.get_tensor_shape(self.output_name))

            in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
            out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
            self._h_input = cuda.pagelocked_empty(trt.volume(max_in), in_dtype)
            self._d_input = cuda.mem_alloc(self._h_input.nbytes)
            self._h_output = cuda.pagelocked_empty(trt.volume(max_out), out_dtype)
            self._d_output = cuda.mem_alloc(self._h_output.nbytes)

            # Addresses are fixed for the engine's lifetime; only the shape
            # changes per call, which set_input_shape() handles on its own.
            self.context.set_tensor_address(self.input_name, int(self._d_input))
            self.context.set_tensor_address(self.output_name, int(self._d_output))
            self._bindings = None
        else:
            # TensorRT 8.x/9.x: no profile query, so trust max_width.
            max_in = (1, 3, 48, self.max_width)
            self.context.set_binding_shape(self.input_idx, max_in)
            max_out = tuple(self.context.get_binding_shape(self.output_idx))

            in_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
            out_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_idx))
            self._h_input = cuda.pagelocked_empty(trt.volume(max_in), in_dtype)
            self._d_input = cuda.mem_alloc(self._h_input.nbytes)
            self._h_output = cuda.pagelocked_empty(trt.volume(max_out), out_dtype)
            self._d_output = cuda.mem_alloc(self._h_output.nbytes)

            self._bindings = [0] * self.engine.num_bindings
            self._bindings[self.input_idx] = int(self._d_input)
            self._bindings[self.output_idx] = int(self._d_output)

        logger.info(
            f"   Buffers: pre-allocated for {tuple(max_in)} -> {tuple(max_out)} "
            f"({(self._h_input.nbytes + self._h_output.nbytes) / 1024:.0f} KB pinned)"
        )

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        """Run one forward pass into the pre-allocated buffers.

        Returns a *copy* of the output, shaped for this input: the host buffer
        is reused by the next call, so handing out a view would let a later
        inference rewrite predictions the caller is still decoding.
        """
        with self._lock:
            if self.use_new_api:
                self.context.set_input_shape(self.input_name, tensor.shape)
                out_shape = tuple(self.context.get_tensor_shape(self.output_name))
            else:
                self.context.set_binding_shape(self.input_idx, tensor.shape)
                out_shape = tuple(self.context.get_binding_shape(self.output_idx))

            n_in = int(tensor.size)
            n_out = int(np.prod(out_shape))
            self._h_input[:n_in] = tensor.ravel()

            cuda.memcpy_htod_async(self._d_input, self._h_input[:n_in], self.stream)
            if self.use_new_api:
                self.context.execute_async_v3(stream_handle=self.stream.handle)
            else:
                self.context.execute_async_v2(
                    bindings=self._bindings, stream_handle=self.stream.handle
                )
            cuda.memcpy_dtoh_async(self._h_output[:n_out], self._d_output, self.stream)
            self.stream.synchronize()

            return self._h_output[:n_out].reshape(out_shape).copy()

    def preprocess(self, image: np.ndarray, target_height: int = 48) -> np.ndarray:
        """
        Preprocess image for recognition model

        Args:
            image: Input image (BGR or grayscale)
            target_height: Target height (default: 48)

        Returns:
            Preprocessed tensor [1, 3, 48, W]
        """
        from ..ocr_utils import preprocess_text_image

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

    @_with_cuda_context
    def recognize_with_char_conf(self, image: np.ndarray) -> tuple:
        """
        Recognize text và trả về per-character confidence (chỉ alnum).

        Returns:
            (text, avg_conf, char_confs)
            char_confs: list of (char, conf) — chỉ chữ/số, bỏ ký tự đặc biệt
        """
        tensor = self.preprocess(image)
        preds = self._infer(tensor)
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

    @_with_cuda_context
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

        # Forward pass into the pre-allocated buffers
        preds = self._infer(tensor)

        # Decode with or without confidence
        if return_confidence:
            text, confidence = self.decode_ctc_with_confidence(preds)
            return text, confidence
        else:
            text = self.decode_ctc(preds)
            return text

    @_with_cuda_context
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