#!/usr/bin/env python3
"""
Simple OCR Inference Script - ONNX
Input: image path
Output: recognized text
"""

import numpy as np
from PIL import Image
import onnxruntime


class SimpleOCR:
    def __init__(self, onnx_path, dict_path='./languages/english/ppocr_keys_v1.txt', use_space_char=True):
        """
        Initialize ONNX OCR model

        Args:
            onnx_path: path to .onnx model file
            dict_path: path to character dictionary
            use_space_char: whether to add space character to dictionary
        """
        # Load ONNX model
        providers = ['CPUExecutionProvider', 'CUDAExecutionProvider']
        self.session = onnxruntime.InferenceSession(onnx_path, providers=providers)

        # Load character dictionary
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = f.read().splitlines()

        # Add space char if needed (must match training config)
        if use_space_char:
            chars.append(' ')

        self.character = ['blank'] + chars

    def preprocess_single(self, img_path):
        """
        Preprocess single image: dynamic ratio resize with padding

        Args:
            img_path: path to input image

        Returns:
            tuple: (preprocessed array [3, H, W], actual_width, target_width, target_height)
        """
        # Load image
        img = Image.open(img_path).convert('RGB')

        # Dynamic ratio resize logic (height=48 for this model)
        w, h = img.size
        base_h = 48
        max_ratio = 12

        # Calculate ratio
        gen_ratio = max(1, round(float(w) / float(h)))
        ratio_resize = min(gen_ratio, max_ratio)

        # Get target size (width = height * ratio)
        target_h = base_h
        target_w = base_h * ratio_resize

        # Resize with padding
        ratio = w / float(h)
        import math
        if math.ceil(target_h * ratio) > target_w:
            resized_w = target_w
        else:
            resized_w = int(math.ceil(target_h * ratio))

        # Resize image
        img = img.resize((resized_w, target_h), Image.BICUBIC)

        # Convert to tensor and normalize
        img_array = np.array(img).astype(np.float32)
        img_array = img_array / 255.0
        img_array = (img_array - 0.5) / 0.5

        # HWC to CHW
        img_array = img_array.transpose(2, 0, 1)

        return img_array, resized_w, target_w, target_h

    def postprocess(self, preds):
        """
        Decode CTC output to text with confidence

        Args:
            preds: model output [batch, seq_len, num_classes]

        Returns:
            tuple: (text, confidence)
        """
        # Get argmax and max values (same as original code)
        preds_idx = np.argmax(preds[0], axis=2)[0]
        preds_prob = np.max(preds[0], axis=2)[0]

        # CTC decode: remove blanks and duplicates, calculate confidence
        char_list = []
        conf_list = []
        last_idx = 0
        for i, idx in enumerate(preds_idx):
            if idx != 0 and idx != last_idx:  # 0 is blank
                if idx < len(self.character):
                    char_list.append(self.character[idx])
                    conf_list.append(preds_prob[i])
            last_idx = idx

        # Calculate average confidence
        text = ''.join(char_list)
        confidence = float(np.mean(conf_list)) if conf_list else 0.0

        return text, confidence

    def recognize(self, img_paths, batch_size=1):
        """
        Run OCR on single image or batch of images

        Args:
            img_paths: path to input image (str) or list of image paths
            batch_size: batch size for processing

        Returns:
            dict or list of dict: {'text': str, 'confidence': float, 'time': float}
        """
        import time

        # Handle single image
        if isinstance(img_paths, str):
            img_paths = [img_paths]
            return_single = True
        else:
            return_single = False

        all_results = []

        # Process in batches
        for batch_start in range(0, len(img_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(img_paths))
            batch_paths = img_paths[batch_start:batch_end]

            # Preprocess batch
            batch_data = []
            for img_path in batch_paths:
                img_array, resized_w, target_w, target_h = self.preprocess_single(img_path)
                batch_data.append((img_array, resized_w, target_w, target_h))

            # Find max dimensions in batch
            max_width = max(d[2] for d in batch_data)
            max_height = max(d[3] for d in batch_data)

            # Create padded batch
            padded_batch = np.zeros((len(batch_data), 3, max_height, max_width), dtype=np.float32)
            for i, (img_array, resized_w, target_w, target_h) in enumerate(batch_data):
                padded_batch[i, :, :target_h, :resized_w] = img_array[:, :, :resized_w]

            # Inference with timing
            start_time = time.time()
            input_name = self.session.get_inputs()[0].name
            preds = self.session.run(None, {input_name: padded_batch})
            inference_time = time.time() - start_time

            # Postprocess each image in batch
            for i in range(len(batch_data)):
                # Extract single prediction
                single_pred = [preds[0][i:i+1]]
                text, confidence = self.postprocess(single_pred)

                all_results.append({
                    'text': text,
                    'confidence': confidence,
                    'time': inference_time / len(batch_data)  # Average time per image
                })

        return all_results[0] if return_single else all_results


if __name__ == '__main__':
    import sys

    # Usage: python simple_ocr_inference.py image1.png [image2.png ...]
    if len(sys.argv) < 2:
        print("Usage: python simple_ocr_inference.py <image_path> [image_path2 ...]")
        print("Example: python simple_ocr_inference.py test1.png")
        print("Example: python simple_ocr_inference.py img1.png img2.png img3.png")
        sys.exit(1)

    img_paths = sys.argv[1:]
    onnx_path = './languages/english/openocr_rec_model.onnx'

    # Initialize OCR
    ocr = SimpleOCR(onnx_path)

    # Recognize (batch if multiple images)
    if len(img_paths) == 1:
        result = ocr.recognize(img_paths[0])
        print(f"Text: {result['text']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference time: {result['time']:.4f}s")
    else:
        print(f"Processing {len(img_paths)} images with batch size 8...")
        results = ocr.recognize(img_paths, batch_size=8)
        for i, (img_path, result) in enumerate(zip(img_paths, results)):
            print(f"\n[{i+1}/{len(img_paths)}] {img_path}")
            print(f"  Text: {result['text']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Time: {result['time']:.4f}s")
