import cv2
import numpy as np
import json
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import time


class SuperPointMatcherONNX:
    def __init__(self, json_path: str, pipeline_path: str, scale: float = 1.0, verbose: bool = False):
        self.verbose = verbose
        t_start = time.time()
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.template_path = data['_template_image']
        self.annotations = data[self.template_path]
        self.scale = scale
        
        t0 = time.time()
        template_img = cv2.imread(self.template_path)
        if scale != 1.0:
            template_img = cv2.resize(template_img, None, fx=scale, fy=scale)
        
        self.template_img = template_img
        self.template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        if self.verbose:
            print(f"⏱️  Load template: {(time.time()-t0)*1000:.1f}ms")
        
        self.template_bbox = None
        self.other_bboxes = []
        for ann in self.annotations:
            if ann['type'] == 'template':
                self.template_bbox = ann
            elif ann['type'] not in ['crop_area']:
                self.other_bboxes.append(ann)
        
        t0 = time.time()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.pipeline_sess = ort.InferenceSession(pipeline_path, providers=providers)
        if self.verbose:
            print(f"⏱️  Load pipeline: {(time.time()-t0)*1000:.1f}ms")
        
        print(f"✅ Initialized SuperPointMatcherONNX ({(time.time()-t_start)*1000:.1f}ms)")
        print(f"   Provider: {self.pipeline_sess.get_providers()[0]}")
        print(f"   Scale: {scale}x")
        print(f"   Template: {Path(self.template_path).name} ({self.template_gray.shape[1]}x{self.template_gray.shape[0]})")
        print(f"   Bboxes: template + {len(self.other_bboxes)} regions")
    
    def _resize_to_32(self, img):
        h, w = img.shape
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32
        if new_h != h or new_w != w:
            resized = cv2.resize(img, (new_w, new_h))
            return resized, (w / new_w, h / new_h)
        return img, (1.0, 1.0)
    
    def match(self, target_path: str, score_threshold: float = 0.3, 
              ransac_threshold: float = 5.0) -> Dict:
        timings = {}
        t_total = time.time()
        
        t0 = time.time()
        target_img_full = cv2.imread(target_path)
        
        if self.scale != 1.0:
            target_img = cv2.resize(target_img_full, None, fx=self.scale, fy=self.scale)
        else:
            target_img = target_img_full
            
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        timings['load_target'] = (time.time() - t0) * 1000
        
        t0 = time.time()
        template_resized, template_scale = self._resize_to_32(self.template_gray)
        target_resized, target_scale = self._resize_to_32(target_gray)
        timings['resize_to_32'] = (time.time() - t0) * 1000
        
        t0 = time.time()
        template_tensor = template_resized.astype(np.float32)[None, None] / 255.0
        target_tensor = target_resized.astype(np.float32)[None, None] / 255.0
        batch_input = np.concatenate([template_tensor, target_tensor], axis=0)
        timings['to_tensor'] = (time.time() - t0) * 1000
        
        t0 = time.time()
        outputs = self.pipeline_sess.run(None, {'images': batch_input})
        kpts, matches, mscores = outputs
        timings['total_inference'] = (time.time() - t0) * 1000
        
        t0 = time.time()
        batch_mask = matches[:, 0] == 0
        batch_matches = matches[batch_mask]
        batch_mscores = mscores[batch_mask]
        
        kpts0 = kpts[0].astype(np.float32)
        kpts1 = kpts[1].astype(np.float32)
        
        valid_mask = batch_mscores > score_threshold
        valid_matches = batch_matches[valid_mask]
        
        m_kpts0 = kpts0[valid_matches[:, 1]].copy()
        m_kpts1 = kpts1[valid_matches[:, 2]].copy()
        
        m_kpts0[:, 0] *= template_scale[0]
        m_kpts0[:, 1] *= template_scale[1]
        m_kpts1[:, 0] *= target_scale[0]
        m_kpts1[:, 1] *= target_scale[1]
        timings['postprocess_matches'] = (time.time() - t0) * 1000
        
        if len(m_kpts0) < 10:
            timings['total'] = (time.time() - t_total) * 1000
            return {
                'success': False,
                'error': f'Too few matches: {len(m_kpts0)}',
                'homography': None,
                'confidence': 0.0,
                'transformed_bboxes': [],
                'target_img': target_img_full,
                'timings': timings
            }
        
        t0 = time.time()
        H, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, ransac_threshold)
        timings['ransac_homography'] = (time.time() - t0) * 1000
        
        if H is None:
            timings['total'] = (time.time() - t_total) * 1000
            return {
                'success': False,
                'error': 'Homography estimation failed',
                'homography': None,
                'confidence': 0.0,
                'transformed_bboxes': [],
                'target_img': target_img_full,
                'timings': timings
            }
        
        inliers = np.sum(mask)
        confidence = inliers / len(m_kpts0)
        
        t0 = time.time()
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
        timings['transform_bboxes'] = (time.time() - t0) * 1000
        
        timings['total'] = (time.time() - t_total) * 1000
        
        if self.verbose:
            self._print_timings(timings)
        
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
    
    def _print_timings(self, timings: Dict):
        print(f"\n⏱️  Timing Breakdown:")
        print(f"   Load target:           {timings['load_target']:7.1f}ms")
        print(f"   Resize to 32x:         {timings['resize_to_32']:7.1f}ms")
        print(f"   To tensor:             {timings['to_tensor']:7.1f}ms")
        print(f"   Total inference:       {timings['total_inference']:7.1f}ms")
        print(f"   Postprocess matches:   {timings['postprocess_matches']:7.1f}ms")
        print(f"   RANSAC homography:     {timings['ransac_homography']:7.1f}ms")
        print(f"   Transform bboxes:      {timings['transform_bboxes']:7.1f}ms")
        print(f"   {'='*40}")
        print(f"   TOTAL:                 {timings['total']:7.1f}ms ({timings['total']/1000:.2f}s)")
    
    def crop_regions(self, result: Dict, output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        t_start = time.time()
        
        if not result['success']:
            print(f"❌ Cannot crop - matching failed: {result.get('error')}")
            return {}
        
        target_img = result['target_img']
        cropped_regions = {}
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, bbox in enumerate(result['transformed_bboxes']):
            if bbox['type'] == 'template':
                continue
            
            pts = np.array(bbox['points'], dtype=np.float32)
            
            width = int(max(
                np.linalg.norm(pts[0] - pts[1]),
                np.linalg.norm(pts[2] - pts[3])
            ))
            height = int(max(
                np.linalg.norm(pts[1] - pts[2]),
                np.linalg.norm(pts[3] - pts[0])
            ))
            
            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(pts, dst_pts)
            cropped = cv2.warpPerspective(target_img, M, (width, height))
            
            key = f"{bbox['type']}_{i}"
            cropped_regions[key] = cropped
            
            if output_dir:
                output_path = Path(output_dir) / f"{key}.png"
                cv2.imwrite(str(output_path), cropped)
        
        if self.verbose:
            print(f"⏱️  Crop regions: {(time.time()-t_start)*1000:.1f}ms")
        
        return cropped_regions
    
    def visualize(self, result: Dict, save_path: Optional[str] = None, show: bool = True):
        t_start = time.time()
        
        if not result['success']:
            print(f"❌ Cannot visualize - matching failed: {result.get('error')}")
            return
        
        colors = {
            'template': (0, 255, 0),
            'text': (255, 165, 0),
            'barcode': (255, 0, 255),
            'datecode': (0, 255, 255)
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        ax = axes[0]
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
            center = pts.mean(axis=0).astype(int)
            cv2.putText(vis_template, bbox['type'], tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        ax.imshow(cv2.cvtColor(vis_template, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Template (scale={self.scale}x)', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        ax = axes[1]
        vis_target = result['target_img'].copy()
        for bbox in result['transformed_bboxes']:
            pts = np.array(bbox['points'], dtype=np.int32)
            color = colors.get(bbox['type'], (255, 255, 255))
            cv2.polylines(vis_target, [pts], True, color, 3)
            
            corner_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            for j, pt in enumerate(pts):
                cv2.circle(vis_target, tuple(pt), 8, corner_colors[j], -1)
            
            center = pts.mean(axis=0).astype(int)
            cv2.putText(vis_target, bbox['type'], tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        timings = result.get('timings', {})
        total_time = timings.get('total', 0)
        confidence_text = f"Confidence: {result['confidence']:.1%}\nInliers: {result['inliers']}/{result['total_matches']}\nTime: {total_time:.0f}ms"
        ax.text(0.02, 0.98, confidence_text, transform=ax.transAxes,
               fontsize=14, verticalalignment='top', color='white',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))
        
        ax.imshow(cv2.cvtColor(vis_target, cv2.COLOR_BGR2RGB))
        ax.set_title('Target with Detected Regions', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        if self.verbose:    
            print(f"⏱️  Visualize: {(time.time()-t_start)*1000:.1f}ms")


if __name__ == '__main__':
    matcher = SuperPointMatcherONNX(
        'images/annotations.json',
        'weights/superpoint_lightglue_pipeline.onnx',
        scale=0.3,
        verbose=True
    )
    for _ in range(20):
        result = matcher.match('images/2.jpg')

    
    if result['success']:
        print(f"\n✅ Success! Confidence: {result['confidence']:.1%}")
        matcher.visualize(result, save_path='result.png')
        cropped = matcher.crop_regions(result, output_dir='results/')
        print(f"✂️  Cropped {len(cropped)} regions")
    else:
        print(f"❌ Failed: {result['error']}")