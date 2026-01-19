import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import requests
import time
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class TensorRTInferenceAPI:
    def __init__(self, engine_path: str, api_base: str = "http://localhost:8000"):
        self.api_base = api_base
        self.engine_path = engine_path
        self._load_engine()
        self._warmup()

    def _load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                if -1 in shape:
                    max_shape = list(shape)
                    max_shape[0] = 8
                    size = trt.volume(max_shape)
                    self.input_shape = [-1, 1, 1080, 1920]
                else:
                    size = trt.volume(shape)
                    self.input_shape = shape

                host_mem = np.empty(size, dtype=dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                self.bindings.append(int(device_mem))
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'name': tensor_name,
                    'dtype': dtype
                })
            else:
                max_size = 100 * 1024 * 1024 // np.dtype(dtype).itemsize
                device_mem = cuda.mem_alloc(max_size * np.dtype(dtype).itemsize)

                self.bindings.append(int(device_mem))
                self.outputs.append({
                    'host': None,
                    'device': device_mem,
                    'shape': shape,
                    'name': tensor_name,
                    'dtype': dtype,
                    'max_size': max_size
                })

    def _warmup(self):
        h, w = self.input_shape[2:]
        dummy_input = np.random.rand(2, 1, h, w).astype(np.float32)
        self._infer(dummy_input)

    def _infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        batch_size = input_data.shape[0]
        if -1 in self.input_shape or self.input_shape[0] != batch_size:
            self.context.set_input_shape(self.inputs[0]['name'], input_data.shape)

        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            input_data.ravel(),
            self.stream
        )

        for i, binding in enumerate(self.bindings):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i),
                binding
            )

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        results = []
        for output in self.outputs:
            actual_shape = self.context.get_tensor_shape(output['name'])
            if -1 in actual_shape:
                host_mem = np.empty(output['max_size'], dtype=output['dtype'])
                cuda.memcpy_dtoh_async(host_mem, output['device'], self.stream)
                self.stream.synchronize()
                results.append(host_mem)
            else:
                actual_size = trt.volume(actual_shape)
                if actual_size <= 0:
                    actual_size = output['max_size']
                host_mem = np.empty(actual_size, dtype=output['dtype'])
                cuda.memcpy_dtoh_async(host_mem, output['device'], self.stream)
                self.stream.synchronize()
                try:
                    results.append(host_mem.reshape(actual_shape))
                except:
                    results.append(host_mem)

        self.stream.synchronize()
        return results

    def get_latest_recipe(self) -> dict:
        url = f"{self.api_base}/api/recipes/loads/latest"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def download_template_image(self, image_url: str) -> np.ndarray:
        if not image_url.startswith('http'):
            image_url = self.api_base.rstrip('/') + image_url

        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        nparr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def _resize_to_engine_size(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        h, w = img.shape
        target_h, target_w = self.input_shape[2:]

        if h != target_h or w != target_w:
            resized = cv2.resize(img, (target_w, target_h))
            return resized, (w / target_w, h / target_h)
        return img, (1.0, 1.0)

    def visualize_failure(self, template_img: np.ndarray, target_img: np.ndarray, num_matches: int):
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        axes[0].imshow(cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Template', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Target (Failed - Only {num_matches} matches)', fontsize=14, fontweight='bold', color='red')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig("inference_failed.jpg", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved failure visualization to: inference_failed.jpg")

    def visualize_success(self, template_img: np.ndarray, target_img: np.ndarray,
                         template_bbox: dict, other_bboxes: list, transformed_bboxes: list,
                         confidence: float, inliers: int, total_matches: int):

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        ax = axes[0]
        ax.imshow(cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB))

        colors = {'template': 'green', 'text': 'orange'}

        if template_bbox:
            pts = np.array(template_bbox['points'])
            polygon = patches.Polygon(pts, fill=False, edgecolor=colors['template'], linewidth=2)
            ax.add_patch(polygon)
            center = pts.mean(axis=0)
            ax.text(center[0], center[1], 'template', fontsize=10, color='white',
                   bbox=dict(boxstyle='round', facecolor=colors['template'], alpha=0.7))

        for bbox in other_bboxes:
            pts = np.array(bbox['points'])
            color = colors.get(bbox['type'], 'yellow')
            polygon = patches.Polygon(pts, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(polygon)
            center = pts.mean(axis=0)
            label = bbox.get('text', bbox['type'])
            ax.text(center[0], center[1], label, fontsize=10, color='white',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

        ax.set_title('Template with Annotations', fontsize=14, fontweight='bold')
        ax.axis('off')

        ax = axes[1]
        ax.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))

        for bbox in transformed_bboxes:
            pts = np.array(bbox['points'])
            color = colors.get(bbox['type'], 'yellow')
            polygon = patches.Polygon(pts, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(polygon)
            center = pts.mean(axis=0)
            label = bbox.get('text', bbox['type'])
            ax.text(center[0], center[1], label, fontsize=10, color='white',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

        info_text = f"Confidence: {confidence:.1%}\nInliers: {inliers}/{total_matches}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', color='white',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

        ax.set_title('Target with Detected Regions', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig("inference_result.jpg", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved result to: inference_result.jpg")

    def process_image(self, target_path: str, score_threshold: float = None, ransac_threshold: float = 5.0):
        t_total = time.time()

        recipe_data = self.get_latest_recipe()
        metadata = recipe_data['metadata']

        if score_threshold is None:
            score_threshold = metadata.get('model_thresholds', {}).get('detection_threshold', 0.3)
            print(f"Using threshold from API: {score_threshold}")

        camera_template = metadata['camera_templates'][0]
        template_info = camera_template['templates'][0]

        template_img = self.download_template_image(template_info['image_url'])
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

        annotations = template_info['annotations']
        template_bbox = None
        other_bboxes = []

        img_h, img_w = template_img.shape[:2]

        for ann in annotations:
            ann_type = ann.get('type', '')

            if ann_type == 'template':
                x = ann.get('x', 0)
                y = ann.get('y', 0)
                w = ann.get('width', 0)
                h = ann.get('height', 0)

                x1 = int(x * img_w)
                y1 = int(y * img_h)
                x2 = int((x + w) * img_w)
                y2 = int((y + h) * img_h)

                template_bbox = {
                    'type': 'template',
                    'points': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                }
            elif ann_type == 'text' and ann.get('points'):
                points = ann.get('points', [])
                pixel_points = []
                for pt in points:
                    px = int(pt[0] * img_w)
                    py = int(pt[1] * img_h)
                    pixel_points.append([px, py])

                other_bboxes.append({
                    'type': ann_type,
                    'text': ann.get('text', ''),
                    'points': pixel_points
                })
            elif ann_type == 'text':
                x = ann.get('x', 0)
                y = ann.get('y', 0)
                w = ann.get('width', 0)
                h = ann.get('height', 0)

                x1 = int(x * img_w)
                y1 = int(y * img_h)
                x2 = int((x + w) * img_w)
                y2 = int((y + h) * img_h)

                other_bboxes.append({
                    'type': ann_type,
                    'text': ann.get('text', ''),
                    'points': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                })

        target_img = cv2.imread(target_path)
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        template_resized, _ = self._resize_to_engine_size(template_gray)
        target_resized, _ = self._resize_to_engine_size(target_gray)

        template_tensor = template_resized.astype(np.float32)[None, None] / 255.0
        target_tensor = target_resized.astype(np.float32)[None, None] / 255.0
        batch_input = np.concatenate([template_tensor, target_tensor], axis=0)

        outputs = self._infer(batch_input)

        kpts_raw = outputs[0]
        matches_raw = outputs[1]
        mscores_raw = outputs[2]

        kpts_flat = kpts_raw.ravel()
        for num_kpts in [4096, 2048, 1024, 512]:
            try:
                kpts = kpts_flat[:batch_input.shape[0]*num_kpts*2].reshape(batch_input.shape[0], num_kpts, 2)
                break
            except:
                continue
        else:
            kpts = kpts_flat[:batch_input.shape[0]*1024*2].reshape(batch_input.shape[0], 1024, 2)

        kpts0 = kpts[0].astype(np.float32)
        kpts1 = kpts[1].astype(np.float32)

        matches_flat = matches_raw.ravel()
        mscores_flat = mscores_raw.ravel()

        valid_mask = mscores_flat > 1e-6
        num_matches = np.sum(valid_mask)

        if num_matches == 0:
            num_matches = len(mscores_flat)

        matches = matches_flat[:num_matches*3].reshape(num_matches, 3).astype(np.int32)
        mscores = mscores_flat[:num_matches]

        batch_mask = matches[:, 0] == 0
        batch_matches = matches[batch_mask]
        batch_mscores = mscores[batch_mask]

        valid_mask = batch_mscores > score_threshold
        valid_matches = batch_matches[valid_mask]

        if len(valid_matches) < 10:
            print(f"Too few matches: {len(valid_matches)}")
            self.visualize_failure(template_img, target_img, len(valid_matches))
            return None

        m_kpts0 = kpts0[valid_matches[:, 1]].copy()
        m_kpts1 = kpts1[valid_matches[:, 2]].copy()

        template_h, template_w = template_gray.shape[:2]
        engine_h, engine_w = self.input_shape[2:]
        template_scale_factor = (template_w / engine_w, template_h / engine_h)

        target_h, target_w = target_gray.shape[:2]
        target_scale_factor = (target_w / engine_w, target_h / engine_h)

        m_kpts0[:, 0] *= template_scale_factor[0]
        m_kpts0[:, 1] *= template_scale_factor[1]
        m_kpts1[:, 0] *= target_scale_factor[0]
        m_kpts1[:, 1] *= target_scale_factor[1]

        H, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, ransac_threshold)

        if H is None:
            print("Homography estimation failed")
            self.visualize_failure(template_img, target_img, len(valid_matches))
            return None

        inliers = np.sum(mask)
        confidence = inliers / len(m_kpts0)

        H_full = H

        transformed_bboxes = []

        if template_bbox:
            template_pts = np.array(template_bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
            template_transformed = cv2.perspectiveTransform(template_pts, H_full)
            transformed_bboxes.append({
                'type': 'template',
                'points': template_transformed.reshape(-1, 2).tolist()
            })

        for bbox in other_bboxes:
            pts = np.array(bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
            pts_transformed = cv2.perspectiveTransform(pts, H_full)
            transformed_bboxes.append({
                'type': bbox['type'],
                'text': bbox['text'],
                'points': pts_transformed.reshape(-1, 2).tolist()
            })

        self.visualize_success(template_img, target_img, template_bbox, other_bboxes,
                              transformed_bboxes, confidence, inliers, len(m_kpts0))

        print(f"✅ Success! Confidence: {confidence:.1%}")
        print(f"   Inliers: {inliers}/{len(m_kpts0)}")
        print(f"   Total time: {(time.time() - t_total)*1000:.1f}ms")

        for bbox in other_bboxes:
            if bbox['type'] == 'text':
                print(f"   Expected text: {bbox['text']}")

        return {
            'success': True,
            'confidence': confidence,
            'transformed_bboxes': transformed_bboxes,
            'recipe_name': metadata.get('name'),
            'product_code': metadata.get('product_code')
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str,
                       default='backend/uploads/inference_results/694850c56beac57d01bdf65c/2026-01-06/24241268/fail_f0_20260106_123218019388_org.jpg',
                       help='Target image path')
    parser.add_argument('--engine', type=str,
                       default='weights/pipeline_fp16_dynamic.engine',
                       help='TensorRT engine path')
    parser.add_argument('--api-base', type=str,
                       default='http://localhost:8000',
                       help='API base URL')
    parser.add_argument('--threshold', type=float,
                       default=None,
                       help='Score threshold (default: load from API)')
    args = parser.parse_args()

    if not os.path.exists(args.target):
        print(f"Target image not found: {args.target}")
        return

    if not os.path.exists(args.engine):
        print(f"Engine file not found: {args.engine}")
        return

    print(f"🚀 Initializing TensorRT engine...")
    print(f"   Engine: {args.engine}")
    print(f"   API: {args.api_base}")
    print(f"   Target: {args.target}")
    print()

    inference = TensorRTInferenceAPI(args.engine, args.api_base)
    result = inference.process_image(args.target, score_threshold=args.threshold)

    if result:
        print(f"\n📦 Recipe: {result['recipe_name']} ({result['product_code']})")


if __name__ == '__main__':
    main()
