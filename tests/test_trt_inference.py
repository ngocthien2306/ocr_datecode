#!/usr/bin/env python3
"""
Simple TensorRT OCR Inference Script
Input: image path
Output: recognized text
Compatible with TensorRT 8.x, 9.x, and 10.x+
"""

import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TensorRTOCR:
    def __init__(self, engine_path, dict_path='./languages/english/ppocr_keys_v1.txt', use_space_char=True):
        """
        Initialize TensorRT OCR model

        Args:
            engine_path: path to .engine model file
            dict_path: path to character dictionary
            use_space_char: whether to add space character to dictionary
        """
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Detect TensorRT version and use appropriate API
        self.use_new_api = hasattr(self.engine, 'num_io_tensors')

        # Get input/output info
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

        # Load character dictionary
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = f.read().splitlines()

        if use_space_char:
            chars.append(' ')

        self.character = ['blank'] + chars

        # CUDA stream
        self.stream = cuda.Stream()

        print(f"TensorRT API: {'10.x+' if self.use_new_api else '8.x/9.x'}")
        print(f"Input: {self.input_name} {self.input_shape}")
        print(f"Output: {self.output_name} {self.output_shape}")

    def preprocess_numpy(self, image: np.ndarray):
        """
        Preprocess numpy array image directly

        Args:
            image: numpy array (BGR or RGB), shape [H, W, 3]

        Returns:
            numpy array [1, 3, 48, W]
        """
        import cv2

        # Convert to RGB if BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR input (OpenCV default)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image

        h, w = img_rgb.shape[:2]

        # Dynamic ratio resize (height=48)
        base_h = 48
        max_ratio = 12

        gen_ratio = max(1, round(float(w) / float(h)))
        ratio_resize = min(gen_ratio, max_ratio)

        target_h = base_h
        target_w = base_h * ratio_resize

        # Calculate actual resize width
        ratio = w / float(h)
        import math
        if math.ceil(target_h * ratio) > target_w:
            resized_w = target_w
        else:
            resized_w = int(math.ceil(target_h * ratio))

        # Resize using cv2
        img_resized = cv2.resize(img_rgb, (resized_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Normalize
        img_array = img_resized.astype(np.float32)
        img_array = img_array / 255.0
        img_array = (img_array - 0.5) / 0.5

        # HWC to CHW
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension and pad to target_w
        padded = np.zeros((1, 3, target_h, target_w), dtype=np.float32)
        padded[0, :, :, :resized_w] = img_array

        return padded

    def preprocess(self, img_path):
        """
        Preprocess single image from file path

        Args:
            img_path: path to input image

        Returns:
            numpy array [1, 3, 48, W]
        """
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # Dynamic ratio resize (height=48)
        base_h = 48
        max_ratio = 12

        gen_ratio = max(1, round(float(w) / float(h)))
        ratio_resize = min(gen_ratio, max_ratio)

        target_h = base_h
        target_w = base_h * ratio_resize

        # Calculate actual resize width
        ratio = w / float(h)
        import math
        if math.ceil(target_h * ratio) > target_w:
            resized_w = target_w
        else:
            resized_w = int(math.ceil(target_h * ratio))

        # Resize
        img = img.resize((resized_w, target_h), Image.BICUBIC)

        # Normalize
        img_array = np.array(img).astype(np.float32)
        img_array = img_array / 255.0
        img_array = (img_array - 0.5) / 0.5

        # HWC to CHW
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension and pad to target_w
        padded = np.zeros((1, 3, target_h, target_w), dtype=np.float32)
        padded[0, :, :, :resized_w] = img_array

        return padded

    def postprocess(self, output):
        """
        Decode CTC output to text

        Args:
            output: model output [1, seq_len, num_classes]

        Returns:
            tuple: (text, confidence)
        """
        preds_idx = np.argmax(output, axis=2)[0]
        preds_prob = np.max(output, axis=2)[0]

        # CTC decode
        char_list = []
        conf_list = []
        last_idx = 0

        for i, idx in enumerate(preds_idx):
            if idx != 0 and idx != last_idx:
                if idx < len(self.character):
                    char_list.append(self.character[idx])
                    conf_list.append(preds_prob[i])
            last_idx = idx

        text = ''.join(char_list)
        confidence = float(np.mean(conf_list)) if conf_list else 0.0

        return text, confidence

    def recognize(self, img_input):
        """
        Run OCR on single image

        Args:
            img_input: path to input image (str) or numpy array

        Returns:
            dict: {'text': str, 'confidence': float, 'time': float}
        """
        result = self.recognize_batch([img_input])
        return result[0]

    def recognize_batch(self, img_inputs, use_true_batch=False):
        """
        Run OCR on batch of images

        Args:
            img_inputs: list of image paths (str) or numpy arrays
            use_true_batch: if True and engine supports it, use true batch inference.
                           if False, process sequentially (fallback mode)

        Returns:
            list of dict: [{'text': str, 'confidence': float, 'time': float}, ...]
        """
        import time

        # Check if engine supports batch > 1
        max_batch = self.input_shape[0] if isinstance(self.input_shape[0], int) else -1

        if use_true_batch and max_batch != 1:
            # TRUE BATCH INFERENCE
            batch_size = len(img_inputs)

            # Check batch limit
            if max_batch > 0 and batch_size > max_batch:
                print(f"Warning: batch size {batch_size} exceeds engine max {max_batch}, processing in chunks")
                # Process in chunks
                all_results = []
                for i in range(0, len(img_inputs), max_batch):
                    chunk = img_inputs[i:i+max_batch]
                    chunk_results = self.recognize_batch(chunk, use_true_batch=True)
                    all_results.extend(chunk_results)
                return all_results

            # Preprocess all images
            preprocessed = []
            for img_input in img_inputs:
                # Detect if input is numpy array or file path
                if isinstance(img_input, np.ndarray):
                    img_array = self.preprocess_numpy(img_input)
                else:
                    img_array = self.preprocess(img_input)
                preprocessed.append(img_array)

            # Find max width in batch
            max_width = max(img.shape[3] for img in preprocessed)
            height = 48

            # Create padded batch
            batch_data = np.zeros((batch_size, 3, height, max_width), dtype=np.float32)
            for i, img in enumerate(preprocessed):
                w = img.shape[3]
                batch_data[i, :, :, :w] = img[0]

            input_shape = batch_data.shape  # [N, 3, 48, W]

            # Allocate buffers
            if self.use_new_api:
                # TensorRT 10.x+ API
                self.context.set_input_shape(self.input_name, input_shape)

                # Allocate input
                input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
                d_input = cuda.mem_alloc(batch_data.nbytes)
                self.context.set_tensor_address(self.input_name, int(d_input))

                # Allocate output
                output_shape = self.context.get_tensor_shape(self.output_name)
                output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
                output_size = trt.volume(output_shape)
                output_data = np.empty(output_shape, dtype=output_dtype)
                d_output = cuda.mem_alloc(output_data.nbytes)
                self.context.set_tensor_address(self.output_name, int(d_output))

            else:
                # TensorRT 8.x/9.x API (legacy)
                self.context.set_binding_shape(0, input_shape)

                # Allocate input
                d_input = cuda.mem_alloc(batch_data.nbytes)

                # Allocate output
                output_shape = self.context.get_binding_shape(1)
                output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize
                d_output = cuda.mem_alloc(output_size)
                output_data = np.empty(output_shape, dtype=np.float32)

            # Copy input to device
            cuda.memcpy_htod_async(d_input, batch_data, self.stream)

            # Run inference with timing
            start_time = time.time()
            if self.use_new_api:
                self.context.execute_async_v3(stream_handle=self.stream.handle)
            else:
                self.context.execute_async_v2(
                    bindings=[int(d_input), int(d_output)],
                    stream_handle=self.stream.handle
                )
            self.stream.synchronize()
            inference_time = time.time() - start_time

            # Copy output to host
            cuda.memcpy_dtoh_async(output_data, d_output, self.stream)
            self.stream.synchronize()

            # Free memory
            d_input.free()
            d_output.free()

            # Postprocess each result in batch
            results = []
            avg_time = inference_time / batch_size

            for i in range(batch_size):
                # Extract single prediction
                single_pred = output_data[i:i+1]
                text, confidence = self.postprocess(single_pred)

                results.append({
                    'text': text,
                    'confidence': confidence,
                    'time': avg_time
                })

            return results

        else:
            # SEQUENTIAL PROCESSING (fallback for batch_size=1 engines)
            results = []

            for img_input in img_inputs:
                # Preprocess
                if isinstance(img_input, np.ndarray):
                    input_data = self.preprocess_numpy(img_input)
                else:
                    input_data = self.preprocess(img_input)
                input_shape = input_data.shape  # [1, 3, 48, W]

                # Allocate buffers
                if self.use_new_api:
                    # TensorRT 10.x+ API
                    self.context.set_input_shape(self.input_name, input_shape)

                    # Allocate input
                    input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
                    d_input = cuda.mem_alloc(input_data.nbytes)
                    self.context.set_tensor_address(self.input_name, int(d_input))

                    # Allocate output
                    output_shape = self.context.get_tensor_shape(self.output_name)
                    output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
                    output_size = trt.volume(output_shape)
                    output_data = np.empty(output_shape, dtype=output_dtype)
                    d_output = cuda.mem_alloc(output_data.nbytes)
                    self.context.set_tensor_address(self.output_name, int(d_output))

                else:
                    # TensorRT 8.x/9.x API (legacy)
                    self.context.set_binding_shape(0, input_shape)

                    # Allocate input
                    d_input = cuda.mem_alloc(input_data.nbytes)

                    # Allocate output
                    output_shape = self.context.get_binding_shape(1)
                    output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize
                    d_output = cuda.mem_alloc(output_size)
                    output_data = np.empty(output_shape, dtype=np.float32)

                # Copy input to device
                cuda.memcpy_htod_async(d_input, input_data, self.stream)

                # Run inference with timing
                start_time = time.time()
                if self.use_new_api:
                    self.context.execute_async_v3(stream_handle=self.stream.handle)
                else:
                    self.context.execute_async_v2(
                        bindings=[int(d_input), int(d_output)],
                        stream_handle=self.stream.handle
                    )
                self.stream.synchronize()
                inference_time = time.time() - start_time

                # Copy output to host
                cuda.memcpy_dtoh_async(output_data, d_output, self.stream)
                self.stream.synchronize()

                # Free memory
                d_input.free()
                d_output.free()

                # Postprocess
                text, confidence = self.postprocess(output_data)

                results.append({
                    'text': text,
                    'confidence': confidence,
                    'time': inference_time
                })

            return results


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_trt_inference.py <image_path> [image_path2 ...]")
        print("Example (single): python test_trt_inference.py test.png")
        print("Example (batch):  python test_trt_inference.py img1.png img2.png img3.png")
        sys.exit(1)

    img_paths = sys.argv[1:]

    # Use batch engine if available, otherwise use single-batch engine
    import os
    batch_engine = './languages/english/openocr_rec_model_batch.engine'
    single_engine = './languages/english/openocr_rec_model.engine'

    if os.path.exists(batch_engine):
        engine_path = batch_engine
        print(f"Using batch-enabled engine")
    else:
        engine_path = single_engine
        print(f"Using single-batch engine (sequential processing)")

    # Initialize OCR
    print(f"Loading TensorRT engine: {engine_path}")
    ocr = TensorRTOCR(engine_path)

    # Single or batch inference
    if len(img_paths) == 1:
        # Single image
        print(f"Processing image: {img_paths[0]}")
        result = ocr.recognize(img_paths[0])
        print(f"\nText: {result['text']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference time: {result['time']*1000:.2f}ms")
    else:
        # Batch processing
        print(f"Processing {len(img_paths)} images in batch...")
        import time
        start = time.time()
        results = ocr.recognize_batch(img_paths)
        total_time = time.time() - start

        print(f"\nBatch Results:")
        print(f"{'#':<4} {'Image':<30} {'Text':<30} {'Conf':<8} {'Time(ms)':<10}")
        print("-" * 88)
        for i, (img_path, result) in enumerate(zip(img_paths, results), 1):
            img_name = img_path.split('/')[-1]
            print(f"{i:<4} {img_name:<30} {result['text']:<30} {result['confidence']:<8.4f} {result['time']*1000:<10.2f}")

        print(f"\nTotal batch time: {total_time*1000:.2f}ms")
        print(f"Average per image: {total_time*1000/len(img_paths):.2f}ms")
        print(f"Throughput: {len(img_paths)/total_time:.2f} images/sec")
