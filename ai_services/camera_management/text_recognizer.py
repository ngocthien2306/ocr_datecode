import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())
class TextRecognizer:
    def __init__(self, model_path, dict_path, use_gpu=True):
        self.model_path = model_path
        self.dict_path = dict_path
        
        self.char_dict = self._load_dict(dict_path)
        self.char_list = ['blank'] + self.char_dict

        # Use CUDAExecutionProvider instead of TensorrtExecutionProvider
        # to avoid conflict with SuperPointMatcherTRT
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        output_shape = self.session.get_outputs()[0].shape
        expected_classes = output_shape[-1] if isinstance(output_shape[-1], int) else None
        
        if expected_classes and len(self.char_list) < expected_classes:
            num_missing = expected_classes - len(self.char_list)
            self.char_list.extend([f' ' for i in range(num_missing)])
            print(f"⚠️  Padded {num_missing} unknown tokens to match model output")
        
        print(f"✅ TextRecognizer initialized")
        print(f"   Model: {Path(model_path).name}")
        print(f"   Dictionary: {len(self.char_dict)} characters")
        print(f"   char_list length: {len(self.char_list)}")
        print(f"   Model output classes: {expected_classes}")
        print(f"   Provider: {self.session.get_providers()[0]}")
    
    def _load_dict(self, dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f.readlines()]
        return chars
    
    def preprocess(self, image, target_height=48):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        ratio = target_height / h
        new_w = max(int(w * ratio), 10)
        
        resized = cv2.resize(gray, (new_w, target_height))
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        
        tensor = np.stack([normalized, normalized, normalized], axis=0).astype(np.float32)
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
        
        return tensor
    
    def decode_ctc(self, preds):
        indices = np.argmax(preds, axis=2)[0]
        
        chars = []
        prev_idx = -1
        
        for idx in indices:
            if idx != 0 and idx != prev_idx:
                if idx < len(self.char_list):
                    chars.append(self.char_list[idx])
            prev_idx = idx
        
        return ''.join(chars)
        
    def decode_ctc_with_confidence(self, preds):
        pred = preds[0]  # Shape: (timesteps, num_classes)
        
        best_path = np.argmax(pred, axis=-1)
        
        all_confidences = []
        for t, char_idx in enumerate(best_path):
            all_confidences.append(float(pred[t, char_idx]))
        
        # Decode text như cũ
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

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def recognize(self, image, return_confidence=True):
        print("ONNX INFERENCE ...")
        tensor = self.preprocess(image)
        preds = self.session.run([self.output_name], {self.input_name: tensor})[0]
        
        if return_confidence:
            return self.decode_ctc_with_confidence(preds)
        else:
            return self.decode_ctc(preds)
    
    def recognize_batch(self, images, max_width=None):
        if not images:
            return []
        
        tensors = [self.preprocess(img) for img in images]
        
        if max_width is None:
            max_width = max(t.shape[3] for t in tensors)
        
        batch_tensors = []
        for tensor in tensors:
            w = tensor.shape[3]
            if w < max_width:
                pad_width = max_width - w
                padded = np.pad(tensor, ((0, 0), (0, 0), (0, 0), (0, pad_width)), mode='constant')
                batch_tensors.append(padded)
            else:
                batch_tensors.append(tensor)
        
        batch = np.vstack(batch_tensors)
        preds = self.session.run([self.output_name], {self.input_name: batch})[0]
        
        results = []
        for i in range(len(images)):
            text, confidence = self.decode_ctc_with_confidence(preds[i:i+1])
            results.append((text, float(confidence)))
        
        return results

if __name__ == '__main__':
    # Initialize recognizer
    recognizer = TextRecognizer(
        model_path='../languages/english/rec.onnx',
        dict_path='../languages/english/dict.txt',
        use_gpu=False
    )
    
    # Test with sample image
    test_image = cv2.imread('/home/demo/Source/ocr_datecode/desktop/tests/debug_ocr_0.png')
    
    if test_image is not None:
        
        start_time = time.time()
        # Recognize
        text, confidence = recognizer.recognize(test_image)
        end_time = time.time()
        print(f"Processing time: {(end_time - start_time)*1000:.3f}  ms")
        print(f"\nRecognized: '{text}'")
        print(f"Confidence: {confidence:.3f}")
        
    else:
        print("Please provide a test image")


# import cv2
# from pathlib import Path

# recognizer = TextRecognizer('../languages/english/rec.onnx', 
#                            '../languages/english/dict.txt', 
#                            use_gpu=True)

# images = [cv2.imread(str(p)) for p in sorted(Path('/Users/ngocthien.ai/Source/Projects/ocr_datecode/desktop/results').glob('cropped_text_*.jpg'))]
# start_time = time.time()
# results = recognizer.recognize_batch(images)
# end_time = time.time()
# print(f"Processing time for {len(images)} images: {(end_time - start_time)*1000:.3f} ms")
# for i, (text, conf) in enumerate(results):
#     print(f"{i}: '{text}' ({conf:.3f})")