import cv2
import numpy as np
import json
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union
import matplotlib.pyplot as plt
import time


class SuperPointMatcherTRT:
    def __init__(self, json_path: str, engine_path: str, scale: float = 1.0, verbose: bool = False):
        self.verbose = verbose
        self.scale = scale
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self._img_cache = {}
        t_start = time.time()
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.template_path = data['_template_image']
        self.annotations = data[self.template_path]
        
        template_img = cv2.imread(self.template_path)
        
        if self.use_cuda:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(template_img)
            
            if scale != 1.0:
                gpu_img = cv2.cuda.resize(gpu_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            
            gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
            self.template_gray = gpu_gray.download()
            self.template_img = gpu_img.download()
        else:
            if scale != 1.0:
                template_img = cv2.resize(template_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            self.template_img = template_img
            self.template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        
        self.template_bbox = None
        self.other_bboxes = []
        for ann in self.annotations:
            if ann['type'] == 'template':
                self.template_bbox = ann
            elif ann['type'] not in ['crop_area']:
                self.other_bboxes.append(ann)
        
        self._load_engine(engine_path)
        
        self.template_resized, self.template_scale = self._resize_to_engine_size(self.template_gray)
        self.template_tensor = self.template_resized.astype(np.float32)[None, None] / 255.0
        
        h, w = self.input_shape[2:]
        dummy = np.random.rand(2, 1, h, w).astype(np.float32)
        self._infer(dummy)
        
        print(f"✅ TRT Matcher ready ({(time.time()-t_start)*1000:.0f}ms) | CUDA: {self.use_cuda} | Scale: {scale}x | Shape: {self.input_shape}")
    
    def _load_engine(self, engine_path: str):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f:
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
                size = trt.volume(shape)
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
                self.input_shape = shape
            else:
                max_size = 100 * 1024 * 1024 // np.dtype(dtype).itemsize
                device_mem = cuda.mem_alloc(max_size * np.dtype(dtype).itemsize)
                
                self.bindings.append(int(device_mem))
                self.outputs.append({
                    'device': device_mem,
                    'name': tensor_name,
                    'dtype': dtype
                })

    def _infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        cuda.memcpy_htod_async(self.inputs[0]['device'], input_data.ravel(), self.stream)
        
        for i, binding in enumerate(self.bindings):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), binding)
        
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        results = []
        for output in self.outputs:
            actual_shape = self.context.get_tensor_shape(output['name'])
            actual_size = trt.volume(actual_shape)
            host_mem = np.empty(actual_size, dtype=output['dtype'])
            cuda.memcpy_dtoh_async(host_mem, output['device'], self.stream)
            results.append(host_mem.reshape(actual_shape))
        
        self.stream.synchronize()
        return results

    def _resize_to_engine_size(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        h, w = img.shape
        target_h, target_w = self.input_shape[2:]
        
        if h != target_h or w != target_w:
            resized = cv2.resize(img, (target_w, target_h))
            return resized, (w / target_w, h / target_h)
        return img, (1.0, 1.0)
    
    def _load_image(self, source: Union[str, np.ndarray], use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Load image from path or numpy array, return (color_img, gray_img)"""
        
        # Check if numpy array
        if isinstance(source, np.ndarray):
            target_img_full = source
            cache_key = None
        else:
            cache_key = source if use_cache else None
            
            # Check cache
            if cache_key and cache_key in self._img_cache:
                return self._img_cache[cache_key]
            
            target_img_full = cv2.imread(source, cv2.IMREAD_COLOR)
            if target_img_full is None:
                raise ValueError(f"Failed to load image: {source}")
        
        # Process with CUDA or CPU
        if self.use_cuda:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(target_img_full)
            
            if self.scale != 1.0:
                gpu_img_scaled = cv2.cuda.resize(gpu_img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
                target_img = gpu_img_scaled.download()
            else:
                gpu_img_scaled = gpu_img
                target_img = target_img_full
            
            gpu_gray = cv2.cuda.cvtColor(gpu_img_scaled, cv2.COLOR_BGR2GRAY)
            target_gray = gpu_gray.download()
        else:
            if self.scale != 1.0:
                target_img = cv2.resize(target_img_full, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
            else:
                target_img = target_img_full
            
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        
        # Cache if path-based
        if cache_key:
            self._img_cache[cache_key] = (target_img, target_gray, target_img_full)
        
        return target_img, target_gray, target_img_full
    
    def match(self, source: Union[str, np.ndarray], score_threshold: float = 0.3, 
              ransac_threshold: float = 5.0, use_cache: bool = True) -> Dict:
        """
        Match template against target image
        
        Args:
            source: Image path (str) or numpy array (np.ndarray)
            score_threshold: Minimum match score
            ransac_threshold: RANSAC threshold for homography
            use_cache: Cache loaded images (only for path-based)
        
        Returns:
            Dict with success, homography, confidence, bboxes, timings
        """
        timings = {}
        t_total = time.time()
        
        t0 = time.time()
        target_img, target_gray, target_img_full = self._load_image(source, use_cache)
        timings['load'] = (time.time() - t0) * 1000
        
        t0 = time.time()
        target_resized, target_scale = self._resize_to_engine_size(target_gray)
        target_tensor = target_resized.astype(np.float32)[None, None] / 255.0
        batch_input = np.concatenate([self.template_tensor, target_tensor], axis=0)
        timings['prep'] = (time.time() - t0) * 1000
        
        t0 = time.time()
        kpts, matches, mscores = self._infer(batch_input)
        timings['trt'] = (time.time() - t0) * 1000
        
        t0 = time.time()
        batch_mask = matches[:, 0] == 0
        batch_matches = matches[batch_mask]
        batch_mscores = mscores[batch_mask]
        
        kpts0 = kpts[0].astype(np.float32)
        kpts1 = kpts[1].astype(np.float32)
        
        valid_mask = batch_mscores > score_threshold
        valid_matches = batch_matches[valid_mask]
        
        if len(valid_matches) < 10:
            timings['total'] = (time.time() - t_total) * 1000
            return {
                'success': False,
                'error': f'Too few matches: {len(valid_matches)}',
                'target_img': target_img_full,
                'timings': timings
            }
        
        m_kpts0 = kpts0[valid_matches[:, 1]].copy()
        m_kpts1 = kpts1[valid_matches[:, 2]].copy()
        
        m_kpts0[:, 0] *= self.template_scale[0]
        m_kpts0[:, 1] *= self.template_scale[1]
        m_kpts1[:, 0] *= target_scale[0]
        m_kpts1[:, 1] *= target_scale[1]
        timings['post'] = (time.time() - t0) * 1000
        
        t0 = time.time()
        H, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, ransac_threshold)
        timings['ransac'] = (time.time() - t0) * 1000
        
        if H is None:
            timings['total'] = (time.time() - t_total) * 1000
            return {
                'success': False,
                'error': 'Homography failed',
                'target_img': target_img_full,
                'timings': timings
            }
        
        inliers = np.sum(mask)
        confidence = inliers / len(m_kpts0)
        
        scale_matrix = np.array([
            [1/self.scale, 0, 0],
            [0, 1/self.scale, 0],
            [0, 0, 1]
        ])
        
        H_full = scale_matrix @ H @ np.linalg.inv(scale_matrix)
        
        transformed_bboxes = []
        template_pts = np.array(self.template_bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
        template_transformed = cv2.perspectiveTransform(template_pts, H_full)
        transformed_bboxes.append({
            'type': 'template',
            'points': template_transformed.reshape(-1, 2).tolist()
        })
        
        for bbox in self.other_bboxes:
            pts = np.array(bbox['points'], dtype=np.float32).reshape(-1, 1, 2)
            pts_transformed = cv2.perspectiveTransform(pts, H_full)
            transformed_bboxes.append({
                'type': bbox['type'],
                'points': pts_transformed.reshape(-1, 2).tolist()
            })
        
        timings['total'] = (time.time() - t_total) * 1000
        
        if self.verbose:
            print(f"Load:{timings['load']:.0f}ms | Prep:{timings['prep']:.0f}ms | TRT:{timings['trt']:.0f}ms | Post:{timings['post']:.0f}ms | RANSAC:{timings['ransac']:.0f}ms | Total:{timings['total']:.0f}ms")
        
        return {
            'success': True,
            'homography': H_full,
            'confidence': confidence,
            'inliers': inliers,
            'total_matches': len(m_kpts0),
            'transformed_bboxes': transformed_bboxes,
            'target_img': target_img_full,
            'timings': timings
        }
    
    def match_batch(self, sources: List[Union[str, np.ndarray]], 
                   score_threshold: float = 0.3,
                   ransac_threshold: float = 5.0) -> List[Dict]:
        """Match multiple images efficiently"""
        results = []
        for source in sources:
            result = self.match(source, score_threshold, ransac_threshold, use_cache=False)
            results.append(result)
        return results
    
    def clear_cache(self):
        """Clear image cache"""
        self._img_cache.clear()
    
    def crop_regions(self, result: Dict, output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        if not result['success']:
            return {}
        
        target_img = result['target_img']
        cropped_regions = {}
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, bbox in enumerate(result['transformed_bboxes']):
            if bbox['type'] == 'template':
                continue
            
            pts = np.array(bbox['points'], dtype=np.float32)
            
            width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
            height = int(max(np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[3] - pts[0])))
            
            dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(pts, dst_pts)
            cropped = cv2.warpPerspective(target_img, M, (width, height))
            
            key = f"{bbox['type']}_{i}"
            cropped_regions[key] = cropped
            
            if output_dir:
                cv2.imwrite(str(Path(output_dir) / f"{key}.png"), cropped)
        
        return cropped_regions
    
    def visualize(self, result: Dict, save_path: Optional[str] = None):
        if not result['success']:
            return
        
        colors = {'template': (0, 255, 0), 'text': (255, 165, 0), 'barcode': (255, 0, 255), 'datecode': (0, 255, 255)}
        
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        vis_template = self.template_img.copy()
        for bbox in self.annotations:
            if bbox['type'] in ['crop_area']:
                continue
            pts = np.array(bbox['points'], dtype=np.float32)
            if self.scale != 1.0:
                pts *= self.scale
            pts = pts.astype(np.int32)
            color = colors.get(bbox['type'], (255, 255, 255))
            cv2.polylines(vis_template, [pts], True, color, 3)
            cv2.putText(vis_template, bbox['type'], tuple(pts.mean(axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        axes[0].imshow(cv2.cvtColor(vis_template, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Template (scale={self.scale}x)', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        vis_target = result['target_img'].copy()
        for bbox in result['transformed_bboxes']:
            pts = np.array(bbox['points'], dtype=np.int32)
            color = colors.get(bbox['type'], (255, 255, 255))
            cv2.polylines(vis_target, [pts], True, color, 3)
            
            for j, pt in enumerate(pts):
                cv2.circle(vis_target, tuple(pt), 8, [(0,255,0), (255,0,0), (0,0,255), (255,255,0)][j], -1)
            
            cv2.putText(vis_target, bbox['type'], tuple(pts.mean(axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        timings = result.get('timings', {})
        text = f"Conf: {result['confidence']:.0%} | Inliers: {result['inliers']}/{result['total_matches']}\nTotal: {timings.get('total',0):.0f}ms | TRT: {timings.get('trt',0):.0f}ms"
        axes[1].text(0.02, 0.98, text, transform=axes[1].transAxes, fontsize=14, verticalalignment='top', color='white', bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))
        
        axes[1].imshow(cv2.cvtColor(vis_target, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Target', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == '__main__':
    matcher = SuperPointMatcherTRT('images/annotations.json', 'weights/pipeline_fp16_small.engine', scale=0.3, verbose=True)
    
    print("\n" + "="*60)
    print("Test 1: Match from path (with cache)")
    print("="*60)
    for i in range(5):
        result = matcher.match('images/4.jpg', use_cache=True)
        print(f"Run {i+1}: {result['timings']['total']:.0f}ms (Load: {result['timings']['load']:.0f}ms)")
    
    print("\n" + "="*60)
    print("Test 2: Match from numpy array")
    print("="*60)
    img_array = cv2.imread('images/4.jpg')
    for i in range(5):
        result = matcher.match(img_array, use_cache=False)
        print(f"Run {i+1}: {result['timings']['total']:.0f}ms (Load: {result['timings']['load']:.0f}ms)")
    
    print("\n" + "="*60)
    print("Test 3: Batch processing")
    print("="*60)
    t0 = time.time()
    results = matcher.match_batch(['images/4.jpg', 'images/2.jpg'])
    total_time = (time.time() - t0) * 1000
    print(f"Batch 2 images: {total_time:.0f}ms ({total_time/2:.0f}ms per image)")
    
    if result['success']:
        print(f"\n✅ Conf: {result['confidence']:.0%}")
        matcher.visualize(result, save_path='results/result_trt.png')
        cropped = matcher.crop_regions(result, output_dir='results/')
        print(f"✂️  {len(cropped)} regions")