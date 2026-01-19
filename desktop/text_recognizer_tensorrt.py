"""
TensorRT-based Text Recognition (High Performance)
Sử dụng TensorRT engine để tăng tốc độ inference
"""
import cv2
import numpy as np
from pathlib import Path

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # Initialize CUDA context
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT/PyCUDA not available. Install with: pip install tensorrt pycuda")


class HostDeviceMem:
    """Simple helper data class for storing host and device memory pointers"""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TextRecognizerTensorRT:
    def __init__(self, engine_path, dict_path, min_width=320, max_width=2000):
        """
        Initialize TensorRT-based text recognizer
        
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
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        # Note: We'll allocate buffers dynamically based on input size
        # since we have dynamic shapes (minShapes/optShapes/maxShapes)
        
        print(f"✅ TextRecognizerTensorRT initialized")
        print(f"   Engine: {Path(engine_path).name}")
        print(f"   Dictionary: {len(self.char_dict)} characters")
        print(f"   Input: {self.input_name} {self.input_shape}")
        print(f"   Output: {self.output_name} {self.output_shape}")
        print(f"   Width range: {min_width} - {max_width}px")
    
    def _load_dict(self, dict_path):
        """Load character dictionary from file"""
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f.readlines()]
        return chars
    
    def _allocate_buffers(self, input_shape):
        """Allocate input/output buffers for specific input shape"""
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
    
    def preprocess(self, image, target_height=48):
        """
        Preprocess image for recognition model
        
        Args:
            image: Input image (BGR or grayscale)
            target_height: Target height (default: 48)
        
        Returns:
            Preprocessed tensor [1, 3, 48, W]
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to target height while maintaining aspect ratio
        h, w = gray.shape
        ratio = target_height / h
        new_w = int(w * ratio)
        
        # Clamp to min/max width bounds
        # Pad if too small, resize if too large
        original_w = new_w
        new_w = max(new_w, self.min_width)
        new_w = min(new_w, self.max_width)
        
        resized = cv2.resize(gray, (original_w, target_height))
        
        # Pad to min_width if necessary
        if original_w < self.min_width:
            pad_width = self.min_width - original_w
            resized = cv2.copyMakeBorder(
                resized, 
                0, 0, 0, pad_width,  # top, bottom, left, right
                cv2.BORDER_CONSTANT, 
                value=0  # Pad with black
            )
        elif original_w > self.max_width:
            # Resize to max_width if too large
            resized = cv2.resize(resized, (self.max_width, target_height))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Subtract mean and divide by std
        mean = np.array([0.5], dtype=np.float32)
        std = np.array([0.5], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Convert to 3 channels (RGB format)
        tensor = np.stack([normalized, normalized, normalized], axis=0).astype(np.float32)
        
        # Add batch dimension: [1, 3, H, W]
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
        
        return tensor
    
    def decode_ctc(self, preds):
        """
        Decode CTC output to text
        
        Args:
            preds: Model output [batch_size, sequence_length, num_classes]
        
        Returns:
            Decoded text string
        """
        # Get best path (argmax)
        indices = np.argmax(preds, axis=2)[0]  # Take first batch
        
        # CTC decoding: remove duplicates and blanks
        chars = []
        prev_idx = -1
        
        for idx in indices:
            # Skip blank (index 0) and consecutive duplicates
            if idx != 0 and idx != prev_idx:
                if idx < len(self.char_list):
                    chars.append(self.char_list[idx])
            prev_idx = idx
        
        return ''.join(chars)
    
    def recognize(self, image, return_confidence=True):
        """
        Recognize text from cropped image using TensorRT
        
        Args:
            image: Cropped text region (BGR or grayscale)
            return_confidence: Return confidence score
        
        Returns:
            If return_confidence=True: (text, confidence)
            If return_confidence=False: text
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
        
        # Validate output shape
        try:
            preds = outputs[0].host.reshape(output_shape)
        except Exception as e:
            print(f"Error reshaping output: {e}")
            print(f"Output shape: {output_shape}, Host buffer size: {outputs[0].host.shape}")
            raise
        
        # Decode
        text = self.decode_ctc(preds)
        
        if return_confidence:
            # Calculate average confidence from softmax probabilities
            softmax_preds = self._softmax(preds[0])  # [seq_len, num_classes]
            max_probs = np.max(softmax_preds, axis=1)
            confidence = np.mean(max_probs)
            return text, float(confidence)
        else:
            return text
    
    def _softmax(self, x):
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def recognize_batch(self, images):
        """
        Recognize text from multiple cropped images (batch processing)
        
        Note: TensorRT batch processing requires same width for all images
        For variable widths, we process one by one (still faster than ONNX)
        
        Args:
            images: List of cropped images
        
        Returns:
            List of (text, confidence) tuples
        """
        results = []
        for img in images:
            text, conf = self.recognize(img)
            results.append((text, conf))
        return results


# Example usage
if __name__ == '__main__':
    import time
    
    # Initialize TensorRT recognizer
    recognizer = TextRecognizerTensorRT(
        engine_path='../languages/english/rec.engine',
        dict_path='../languages/english/dict.txt'
    )
    
    # Test with sample image
    test_image = cv2.imread('../images/1.jpg')
    
    if test_image is not None:
        # Crop a region (example)
        cropped = test_image[100:150, 100:400]
        
        # Warm up
        _ = recognizer.recognize(cropped)
        
        # Benchmark
        iterations = 10
        start = time.time()
        for _ in range(iterations):
            text, confidence = recognizer.recognize(cropped)
        avg_time = (time.time() - start) / iterations * 1000
        
        print(f"\n{'='*60}")
        print(f"Recognized: '{text}'")
        print(f"Confidence: {confidence:.3f}")
        print(f"Average inference time: {avg_time:.2f}ms")
        print(f"{'='*60}")
        
        # Show result
        cv2.imshow('Cropped Region', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Please provide a test image")
