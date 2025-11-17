import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Optional
from PyQt5.QtCore import QRect

# Try import SuperPoint + LightGlue
try:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    SUPERPOINT_AVAILABLE = True
except ImportError:
    SUPERPOINT_AVAILABLE = False
    print("Warning: SuperPoint/LightGlue not available.")


class BoundingBox:
    def __init__(self, rect=None, bbox_type="text", shape="rectangle", points=None, bbox_id=None):
        self.bbox_type = bbox_type
        self.shape = shape
        self.bbox_id = bbox_id or id(self)

        if shape == "polygon" and points is not None:
            self.points = points
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            self.rect = QRect(int(min(xs)), int(min(ys)),
                            int(max(xs) - min(xs)), int(max(ys) - min(ys)))
        else:
            self.points = None
            self.rect = rect if rect else QRect()

    @property
    def polygon(self):
        return self.points


class SuperPointMatcher:
    """Template matcher using SuperPoint + LightGlue on FULL template image"""

    def __init__(self, template_image_path: str, template_bboxes, scale: float = 1.0, verbose: bool = False):
        """
        Args:
            template_image_path: Path to template image
            template_bboxes: List of BoundingBox objects from annotations
            scale: Scale factor for images (1.0 = original size)
            verbose: Print detailed timing information
        """
        if not SUPERPOINT_AVAILABLE:
            raise ImportError("SuperPoint/LightGlue not available. Install with: pip install kornia git+https://github.com/cvg/LightGlue.git")

        self.verbose = verbose
        self.scale = scale
        t_start = time.time()

        # Load full template image
        t0 = time.time()
        template_img = cv2.imread(template_image_path)

        # Apply scale if needed
        if scale != 1.0:
            template_img = cv2.resize(template_img, None, fx=scale, fy=scale)

        self.template_img = template_img
        self.template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

        if self.verbose:
            print(f"⏱️  Load template: {(time.time()-t0)*1000:.1f}ms")

        # Parse bboxes
        self.template_bbox = None
        self.other_bboxes = []
        for bbox in template_bboxes:
            if bbox.bbox_type == 'template':
                self.template_bbox = bbox
            elif bbox.bbox_type not in ['crop_area']:
                self.other_bboxes.append(bbox)

        # Initialize SuperPoint + LightGlue
        t0 = time.time()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        if self.verbose:
            print(f"⏱️  Load SuperPoint+LightGlue: {(time.time()-t0)*1000:.1f}ms")

        print(f"✅ Initialized SuperPointMatcher ({(time.time()-t_start)*1000:.1f}ms)")
        print(f"   Device: {self.device}")
        print(f"   Scale: {scale}x")
        print(f"   Template: {Path(template_image_path).name} ({self.template_gray.shape[1]}x{self.template_gray.shape[0]})")
        print(f"   Bboxes: template + {len(self.other_bboxes)} regions")

    def _resize_to_32(self, img):
        """Resize to multiples of 32 (SuperPoint requirement)"""
        h, w = img.shape
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32
        if new_h != h or new_w != w:
            resized = cv2.resize(img, (new_w, new_h))
            return resized, (w / new_w, h / new_h)
        return img, (1.0, 1.0)

    def match(self, target_image_path: str, method='superpoint', threshold: float = 0.3,
              debug: bool = False):
        """
        Match template to target image

        Returns:
            tuple: (bboxes, confidence, target_image)
        """
        timings = {}
        t_total = time.time()

        # Load target
        t0 = time.time()
        target_img_full = cv2.imread(target_image_path)

        # Apply scale if needed
        if self.scale != 1.0:
            target_img = cv2.resize(target_img_full, None, fx=self.scale, fy=self.scale)
        else:
            target_img = target_img_full

        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        timings['load_target'] = (time.time() - t0) * 1000

        # Resize to multiples of 32
        t0 = time.time()
        template_resized, template_scale = self._resize_to_32(self.template_gray)
        target_resized, target_scale = self._resize_to_32(target_gray)
        timings['resize_to_32'] = (time.time() - t0) * 1000

        if debug:
            print(f"\n🔍 SuperPoint Debug (FULL IMAGE):")
            print(f"   Device: {self.device}")
            print(f"   Template: {self.template_gray.shape} → {template_resized.shape}")
            print(f"   Target: {target_gray.shape} → {target_resized.shape}")

        # Convert to tensors
        t0 = time.time()
        template_tensor = torch.from_numpy(template_resized).float()[None, None] / 255.0
        target_tensor = torch.from_numpy(target_resized).float()[None, None] / 255.0
        template_tensor = template_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        timings['to_tensor'] = (time.time() - t0) * 1000

        # Extract and match features
        t0 = time.time()
        with torch.no_grad():
            feats0 = self.extractor.extract(template_tensor)
            feats1 = self.extractor.extract(target_tensor)
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        timings['feature_matching'] = (time.time() - t0) * 1000

        # Get matches
        t0 = time.time()
        kpts0 = feats0['keypoints']
        kpts1 = feats1['keypoints']
        matches = matches01['matches']
        match_scores = matches01.get('scores', None)

        if debug:
            print(f"   Template keypoints: {kpts0.shape[0]}")
            print(f"   Target keypoints: {kpts1.shape[0]}")
            print(f"   Raw matches: {matches.shape[0]}")

        # Filter by score
        if match_scores is not None:
            good_mask = match_scores > threshold
            matches = matches[good_mask]
            match_scores = match_scores[good_mask]
            if debug:
                print(f"   Filtered matches (>{threshold}): {matches.shape[0]}")

        if matches.shape[0] < 10:
            if debug:
                print(f"   ❌ Too few matches: {matches.shape[0]}")
            timings['postprocess'] = (time.time() - t0) * 1000
            timings['total'] = (time.time() - t_total) * 1000
            return None, 0.0, target_img_full

        # Scale keypoints back
        m_kpts0 = kpts0[matches[..., 0]].cpu().numpy()
        m_kpts1 = kpts1[matches[..., 1]].cpu().numpy()
        m_kpts0[:, 0] *= template_scale[0]
        m_kpts0[:, 1] *= template_scale[1]
        m_kpts1[:, 0] *= target_scale[0]
        m_kpts1[:, 1] *= target_scale[1]
        timings['postprocess'] = (time.time() - t0) * 1000

        # Homography
        t0 = time.time()
        H, mask = cv2.findHomography(m_kpts0.reshape(-1, 1, 2).astype(np.float32),
                                     m_kpts1.reshape(-1, 1, 2).astype(np.float32),
                                     cv2.RANSAC, ransacReprojThreshold=5.0)
        timings['ransac'] = (time.time() - t0) * 1000

        if H is None:
            if debug:
                print("   ❌ Homography failed")
            timings['total'] = (time.time() - t_total) * 1000
            return None, 0.0, target_img_full

        # Confidence
        ransac_conf = np.sum(mask) / len(mask)
        if ransac_conf < 0.3:
            if debug:
                print(f"   ❌ Low confidence: {ransac_conf:.3f}")
            timings['total'] = (time.time() - t_total) * 1000
            return None, ransac_conf, target_img_full

        if match_scores is not None:
            inlier_scores = match_scores.cpu().numpy()[mask.flatten().astype(bool)]
            lg_conf = inlier_scores.mean() if len(inlier_scores) > 0 else 0.0
            confidence = 0.6 * ransac_conf + 0.4 * lg_conf
        else:
            confidence = ransac_conf

        if debug:
            print(f"   ✓ Inliers: {np.sum(mask)}/{len(mask)} ({ransac_conf:.3f})")
            print(f"   ✓ Final confidence: {confidence:.3f}")

        # Transform bboxes
        t0 = time.time()
        transformed_bboxes = []

        # Scale homography if needed (to work with full-resolution images)
        if self.scale != 1.0:
            scale_matrix = np.array([
                [1/self.scale, 0, 0],
                [0, 1/self.scale, 0],
                [0, 0, 1]
            ])
            H_full = scale_matrix @ H @ np.linalg.inv(scale_matrix)
        else:
            H_full = H

        # Template bbox
        if self.template_bbox.shape == 'polygon' and self.template_bbox.points:
            pts = np.array(self.template_bbox.points, dtype=np.float32).reshape(-1, 1, 2)
        else:
            x, y = self.template_bbox.rect.x(), self.template_bbox.rect.y()
            w, h = self.template_bbox.rect.width(), self.template_bbox.rect.height()
            pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(pts, H_full)
        polygon = transformed.reshape(-1, 2).tolist()
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        rect = QRect(int(min(xs)), int(min(ys)),
                    int(max(xs)-min(xs)), int(max(ys)-min(ys)))
        transformed_bboxes.append(BoundingBox(rect=rect, bbox_type='template',
                                             shape='polygon', points=polygon))

        # Other bboxes
        for bbox in self.other_bboxes:
            if bbox.shape == 'polygon' and bbox.points:
                pts = np.array(bbox.points, dtype=np.float32).reshape(-1, 1, 2)
            else:
                x, y = bbox.rect.x(), bbox.rect.y()
                w, h = bbox.rect.width(), bbox.rect.height()
                pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32).reshape(-1, 1, 2)

            transformed = cv2.perspectiveTransform(pts, H_full)
            polygon = transformed.reshape(-1, 2).tolist()
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            rect = QRect(int(min(xs)), int(min(ys)),
                        int(max(xs)-min(xs)), int(max(ys)-min(ys)))
            transformed_bboxes.append(BoundingBox(rect=rect, bbox_type=bbox.bbox_type,
                                                 shape='polygon', points=polygon))

        timings['transform_bboxes'] = (time.time() - t0) * 1000
        timings['total'] = (time.time() - t_total) * 1000

        if self.verbose or debug:
            self._print_timings(timings)

        return transformed_bboxes, confidence, target_img_full

    def _print_timings(self, timings: Dict):
        print(f"\n⏱️  SuperPoint Timing:")
        print(f"   Load target:       {timings.get('load_target', 0):7.1f}ms")
        print(f"   Resize to 32x:     {timings.get('resize_to_32', 0):7.1f}ms")
        print(f"   To tensor:         {timings.get('to_tensor', 0):7.1f}ms")
        print(f"   Feature+Match:     {timings.get('feature_matching', 0):7.1f}ms")
        print(f"   Postprocess:       {timings.get('postprocess', 0):7.1f}ms")
        print(f"   RANSAC:            {timings.get('ransac', 0):7.1f}ms")
        print(f"   Transform bboxes:  {timings.get('transform_bboxes', 0):7.1f}ms")
        print(f"   {'='*40}")
        print(f"   TOTAL:             {timings['total']:7.1f}ms")
