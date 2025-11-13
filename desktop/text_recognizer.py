"""
Lightweight Text Recognition (chỉ Recognition, không Detection)
Dùng cho các vùng text đã được crop sẵn từ template matching
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class TextRecognizer:
    def __init__(self, model_path, dict_path, use_gpu=False):
        """
        Initialize text recognizer
        
        Args:
            model_path: Path to rec.onnx model
            dict_path: Path to dict.txt (character dictionary)
            use_gpu: Use CUDA provider if available
        """
        self.model_path = model_path
        self.dict_path = dict_path
        
        # Load character dictionary
        self.char_dict = self._load_dict(dict_path)
        self.char_list = ['blank'] + self.char_dict  # CTC blank token at index 0
        
        # Initialize ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✅ TextRecognizer initialized")
        print(f"   Model: {Path(model_path).name}")
        print(f"   Dictionary: {len(self.char_dict)} characters")
        print(f"   Provider: {self.session.get_providers()[0]}")
    
    def _load_dict(self, dict_path):
        """Load character dictionary from file"""
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f.readlines()]
        return chars
    
    def preprocess(self, image, target_height=48):
        """
        Preprocess image for recognition model
        
        Args:
            image: Input image (BGR or grayscale)
            target_height: Target height (default: 32 for PP-OCR)
        
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
        
        # Ensure minimum width
        new_w = max(new_w, 10)
        
        resized = cv2.resize(gray, (new_w, target_height))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Subtract mean and divide by std (ImageNet stats)
        mean = np.array([0.5], dtype=np.float32)
        std = np.array([0.5], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Convert to 3 channels (RGB format expected by model)
        # Shape: [H, W] -> [3, H, W]
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
        # Shape: [batch_size, sequence_length]
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
        Recognize text from cropped image
        
        Args:
            image: Cropped text region (BGR or grayscale)
            return_confidence: Return confidence score
        
        Returns:
            If return_confidence=True: (text, confidence)
            If return_confidence=False: text
        """
        # Preprocess
        tensor = self.preprocess(image)
        
        # Inference
        preds = self.session.run([self.output_name], {self.input_name: tensor})[0]
        
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
    
    def recognize_batch(self, images, max_width=None):
        """
        Recognize text from multiple cropped images (batch processing)
        
        Args:
            images: List of cropped images
            max_width: Maximum width for padding (auto-detect if None)
        
        Returns:
            List of (text, confidence) tuples
        """
        if not images:
            return []
        
        # Preprocess all images
        tensors = [self.preprocess(img) for img in images]
        
        # Find max width for padding
        if max_width is None:
            max_width = max(t.shape[3] for t in tensors)
        
        # Pad to same width
        batch_tensors = []
        for tensor in tensors:
            h, w = tensor.shape[2], tensor.shape[3]
            if w < max_width:
                # Pad right with zeros
                pad_width = max_width - w
                padded = np.pad(tensor, ((0, 0), (0, 0), (0, 0), (0, pad_width)), mode='constant')
                batch_tensors.append(padded)
            else:
                batch_tensors.append(tensor)
        
        # Stack into batch
        batch = np.vstack(batch_tensors)
        
        # Inference
        preds = self.session.run([self.output_name], {self.input_name: batch})[0]
        
        # Decode each prediction
        results = []
        for i in range(len(images)):
            text = self.decode_ctc(preds[i:i+1])
            
            # Calculate confidence
            softmax_preds = self._softmax(preds[i])
            max_probs = np.max(softmax_preds, axis=1)
            confidence = np.mean(max_probs)
            
            results.append((text, float(confidence)))
        
        return results


# Example usage
if __name__ == '__main__':
    # Initialize recognizer
    recognizer = TextRecognizer(
        model_path='../languages/english/rec.onnx',
        dict_path='../languages/english/dict.txt',
        use_gpu=False
    )
    
    # Test with sample image
    test_image = cv2.imread('../images/1.jpg')
    
    if test_image is not None:
        # Crop a region (example)
        cropped = test_image[100:150, 100:400]
        
        # Recognize
        text, confidence = recognizer.recognize(cropped)
        print(f"\nRecognized: '{text}'")
        print(f"Confidence: {confidence:.3f}")
        
        # Show result
        cv2.imshow('Cropped Region', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Please provide a test image")
