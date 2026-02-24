"""
OpenOCR RepSVTR ONNX Runtime Implementation
Fallback option when TensorRT is not available
Performance: ~13 imgs/sec (batch=4)
Compatible with ONNX Runtime GPU/CPU
"""
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. OpenOCR ONNX will be disabled.")


class TextRecognizerOpenOCRONNX:
    """
    OpenOCR RepSVTR ONNX recognizer.

    Key Features:
    - ONNX Runtime (GPU/CPU)
    - Batch processing support
    - Direct numpy array input
    - CTC decoding with confidence scores

    Performance: ~13 imgs/sec (batch=4) - fallback option
    Model: OpenOCR RepSVTR
    """

    def __init__(self, onnx_path: str, dict_path: str, use_space_char: bool = True, use_gpu: bool = True):
        """
        Initialize OpenOCR ONNX recognizer.

        Args:
            onnx_path: Path to .onnx file
            dict_path: Path to character dictionary
            use_space_char: Add space character to dictionary
            use_gpu: Use GPU if available
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available. Please install onnxruntime or onnxruntime-gpu.")

        self.onnx_path = onnx_path
        self.dict_path = dict_path

        # Load character dictionary
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = f.read().splitlines()

        if use_space_char:
            chars.append(' ')

        self.character = ['blank'] + chars

        # Create ONNX Runtime session
        providers = []
        if use_gpu:
            # Try CUDA first, fallback to CPU
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
                logger.info("Using CUDA for ONNX inference")
            else:
                logger.warning("CUDA not available, using CPU")
                providers.append('CPUExecutionProvider')
        else:
            providers.append('CPUExecutionProvider')

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"✅ TextRecognizerOpenOCRONNX initialized")
        logger.info(f"   Model: {Path(onnx_path).name}")
        logger.info(f"   Dictionary: {len(self.character)} characters")
        logger.info(f"   Provider: {self.session.get_providers()[0]}")

    def preprocess_numpy(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess numpy array image (OpenOCR RepSVTR format).

        Args:
            image: numpy array (BGR or RGB or Grayscale)

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

        return {'text': text, 'confidence': confidence}

    def recognize(self, image: np.ndarray, return_confidence: bool = True) -> tuple:
        """
        Run OCR on single image.

        Args:
            image: numpy array (BGR format from OpenCV)
            return_confidence: Return confidence score

        Returns:
            tuple: (text, confidence) if return_confidence=True
        """
        # Preprocess
        input_data = self.preprocess_numpy(image)

        # Run inference
        output = self.session.run([self.output_name], {self.input_name: input_data})[0]

        # Postprocess
        result = self.postprocess(output)

        if return_confidence:
            return (result['text'], result['confidence'])
        else:
            return (result['text'],)

    def recognize_batch(self, images: list, batch_size: int = 4) -> list:
        """
        Run OCR on batch of images.

        Args:
            images: List of numpy arrays (BGR format)
            batch_size: Batch size for processing (default: 4)

        Returns:
            List of dicts: [{'text': str, 'confidence': float}, ...]
        """
        results = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]

            # Preprocess batch
            preprocessed = []
            for img in batch_images:
                if isinstance(img, np.ndarray):
                    img_array = self.preprocess_numpy(img)
                else:
                    raise ValueError("Only numpy arrays supported")
                preprocessed.append(img_array)

            # Find max width in batch
            max_width = max(img.shape[3] for img in preprocessed)
            height = 48
            batch_len = len(preprocessed)

            # Create padded batch
            batch_data = np.zeros((batch_len, 3, height, max_width), dtype=np.float32)
            for j, img in enumerate(preprocessed):
                w = img.shape[3]
                batch_data[j, :, :, :w] = img[0]

            # Run batch inference
            output = self.session.run([self.output_name], {self.input_name: batch_data})[0]

            # Postprocess each result
            for j in range(batch_len):
                single_pred = output[j:j+1]
                result = self.postprocess(single_pred)
                results.append(result)

        return results
