"""
YOLO OBB TensorRT Inference
TensorRT 10.x compatible
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from typing import List, Tuple


class YOLOOBBTensorRT:
    def __init__(self, engine_path: str, class_names: List[str] = None):
        self.class_names = class_names or []
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Get input/output info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # Get shapes
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)

        # Check if dynamic
        self.is_dynamic = -1 in self.input_shape

        if self.is_dynamic:
            # Get min/max batch from profile
            profile = self.engine.get_tensor_profile_shape(self.input_name, 0)
            self.min_batch = profile[0][0]
            self.max_batch = profile[2][0]
            print(f"Dynamic batch: {self.min_batch} - {self.max_batch}")
        else:
            self.batch_size = self.input_shape[0]
            print(f"Fixed batch: {self.batch_size}")

        # Allocate buffers (will be resized for dynamic batch)
        self.allocate_buffers(self.max_batch if self.is_dynamic else self.batch_size)

    def allocate_buffers(self, batch_size: int):
        """Allocate device memory"""
        input_shape = (batch_size, 3, 320, 320)
        output_shape = (batch_size, 300, 7)

        input_size = int(np.prod(input_shape))
        output_size = int(np.prod(output_shape))

        self.h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)

        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        self.stream = cuda.Stream()

    def preprocess(self, images: List[np.ndarray]):
        """Preprocess batch of images"""
        batch_tensor = []
        scales = []
        pads = []

        for image in images:
            h, w = image.shape[:2]
            scale = min(320 / w, 320 / h)
            new_w, new_h = int(w * scale), int(h * scale)

            resized = cv2.resize(image, (new_w, new_h))
            padded = np.full((320, 320, 3), 114, dtype=np.uint8)
            pad_top = (320 - new_h) // 2
            pad_left = (320 - new_w) // 2
            padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            batch_tensor.append(tensor)
            scales.append(scale)
            pads.append((pad_top, pad_left))

        return np.array(batch_tensor, dtype=np.float32), scales, pads

    def postprocess(self, output: np.ndarray, conf_threshold: float = 0.7):
        """Postprocess single image output"""
        # Format: [cx, cy, w, h, conf, class_id, angle]
        boxes = np.concatenate([output[:, :4], output[:, 6:7]], axis=1)
        scores = output[:, 4]
        class_ids = output[:, 5].astype(int)

        mask = (scores >= conf_threshold) & (scores > 0)
        return boxes[mask], scores[mask], class_ids[mask]

    def predict(self, images: List[np.ndarray], conf_threshold: float = 0.7,
                return_timing: bool = False):
        """
        Predict on batch of images
        """
        if not isinstance(images, list):
            images = [images]

        batch_size = len(images)

        # Check batch size
        if self.is_dynamic:
            if batch_size < self.min_batch or batch_size > self.max_batch:
                raise ValueError(f"Batch size {batch_size} out of range [{self.min_batch}, {self.max_batch}]")
            # Set dynamic shape
            self.context.set_input_shape(self.input_name, (batch_size, 3, 320, 320))
        else:
            if batch_size != self.batch_size:
                raise ValueError(f"Batch size must be {self.batch_size}, got {batch_size}")

        timing = {}

        # Preprocess
        t0 = time.perf_counter()
        batch_tensor, scales, pads = self.preprocess(images)
        timing['preprocess'] = (time.perf_counter() - t0) * 1000

        # Copy to device
        t0 = time.perf_counter()
        np.copyto(self.h_input[:batch_tensor.size], batch_tensor.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        timing['h2d'] = (time.perf_counter() - t0) * 1000

        # Inference
        t0 = time.perf_counter()
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        timing['inference'] = (time.perf_counter() - t0) * 1000

        # Copy from device
        t0 = time.perf_counter()
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        timing['d2h'] = (time.perf_counter() - t0) * 1000

        # Postprocess
        t0 = time.perf_counter()
        # Get actual output size for current batch
        output_size = batch_size * 300 * 7
        output = self.h_output[:output_size].reshape(batch_size, 300, 7)
        results = []

        for i in range(batch_size):
            boxes, scores, class_ids = self.postprocess(output[i], conf_threshold)

            # Scale back to original coordinates
            if len(boxes) > 0:
                pad_top, pad_left = pads[i]
                scale = scales[i]
                boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
                boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
                boxes[:, 2] = boxes[:, 2] / scale
                boxes[:, 3] = boxes[:, 3] / scale

            results.append((boxes, scores, class_ids))

        timing['postprocess'] = (time.perf_counter() - t0) * 1000
        timing['total'] = sum(timing.values())
        timing['per_image'] = timing['total'] / batch_size

        return (results, timing) if return_timing else results

    def draw_boxes(self, image: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                   class_ids: np.ndarray):
        """Draw rotated boxes"""
        img = image.copy()

        for box, score, cls_id in zip(boxes, scores, class_ids):
            cx, cy, w, h, angle = box
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            hw, hh = w / 2, h / 2

            corners = np.array([
                [cx + (-hw * cos_a + hh * sin_a), cy + (-hw * sin_a - hh * cos_a)],
                [cx + (hw * cos_a + hh * sin_a), cy + (hw * sin_a - hh * cos_a)],
                [cx + (hw * cos_a - hh * sin_a), cy + (hw * sin_a + hh * cos_a)],
                [cx + (-hw * cos_a - hh * sin_a), cy + (-hw * sin_a + hh * cos_a)]
            ]).astype(np.int32)

            cv2.polylines(img, [corners], True, (0, 255, 0), 2)

            label = f"{self.class_names[cls_id]}: {score:.2f}" if cls_id < len(self.class_names) \
                    else f"Class {cls_id}: {score:.2f}"
            cv2.putText(img, label, tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)

        return img


def main():
    # Initialize
    model = YOLOOBBTensorRT(
        engine_path="weights/best_bottle_obb_l_320.engine",
        class_names=['bottle', 'label']
    )

    print(f"\n=== TensorRT OBB Inference ===")

    # Test single image
    print("\n--- Single Image ---")
    image = cv2.imread('/home/demo/Source/ocr_datecode/backend/uploads/inference_results/69c4891ca70901e2a320fb94/2026-03-26/40767171/fail_f0_20260326_033835934715_org.jpg')
    results, timing = model.predict([image], conf_threshold=0.1, return_timing=True)

    boxes, scores, class_ids = results[0]
    print(f"Detected: {len(boxes)} objects")
    for i, (box, score, cls) in enumerate(zip(boxes, scores, class_ids)):
        print(f"  {i}: {model.class_names[cls]} conf={score:.4f} "
              f"cx={box[0]:.1f} cy={box[1]:.1f} angle={box[4]:.4f}")

    print(f"\nTiming:")
    print(f"  Preprocess: {timing['preprocess']:.2f} ms")
    print(f"  H2D:        {timing['h2d']:.2f} ms")
    print(f"  Inference:  {timing['inference']:.2f} ms")
    print(f"  D2H:        {timing['d2h']:.2f} ms")
    print(f"  Postprocess:{timing['postprocess']:.2f} ms")
    print(f"  Total:      {timing['total']:.2f} ms ({1000/timing['total']:.1f} FPS)")

    annotated = model.draw_boxes(image, boxes, scores, class_ids)
    cv2.imwrite("output_trt.png", annotated)
    print("Saved: output_trt.png")

    # Test batch
    print("\n--- Batch Inference (4 images) ---")
    images = [cv2.imread('/home/demo/Source/ocr_datecode/backend/uploads/inference_results/69c4891ca70901e2a320fb94/2026-03-26/40767171/fail_f0_20260326_033835934715_org.jpg') for _ in range(2)]
    results, timing = model.predict(images, conf_threshold=0.3, return_timing=True)

    for i, (boxes, scores, _) in enumerate(results):
        print(f"Image {i}: {len(boxes)} detections")

    print(f"\nBatch Timing:")
    print(f"  Total:     {timing['total']:.2f} ms")
    print(f"  Per image: {timing['per_image']:.2f} ms ({1000/timing['per_image']:.1f} FPS)")
    print(f"  Inference: {timing['inference']:.2f} ms")


if __name__ == "__main__":
    main()
