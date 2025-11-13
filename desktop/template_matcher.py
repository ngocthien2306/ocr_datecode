import cv2
import numpy as np
from PyQt5.QtCore import QRect

# Try import SuperPoint + LightGlue (optional)
try:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    SUPERPOINT_AVAILABLE = True
except ImportError:
    SUPERPOINT_AVAILABLE = False
    print("Warning: SuperPoint/LightGlue not available. Install with: pip install kornia git+https://github.com/cvg/LightGlue.git")


class BoundingBox:
    def __init__(self, rect=None, bbox_type="text", shape="rectangle", points=None, bbox_id=None):
        """
        Args:
            rect: QRect for rectangle shape (backward compatibility)
            bbox_type: str - type of bbox (template, text, datecode, barcode)
            shape: str - "rectangle" or "polygon"
            points: list of [x,y] for polygon shape (4 points)
            bbox_id: unique ID (for UI tracking)
        """
        self.bbox_type = bbox_type
        self.shape = shape  # "rectangle" or "polygon"
        self.bbox_id = bbox_id or id(self)  # Unique ID for UI

        if shape == "polygon" and points is not None:
            self.points = points  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Calculate bounding rect from polygon for compatibility
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            self.rect = QRect(int(min(xs)), int(min(ys)),
                            int(max(xs) - min(xs)), int(max(ys) - min(ys)))
        else:
            self.points = None
            self.rect = rect if rect else QRect()

    # Backward compatibility
    @property
    def polygon(self):
        """Alias for points (backward compatibility)"""
        return self.points

    @staticmethod
    def from_dict(data):
        """Load from JSON format"""
        shape = data.get('shape', 'rectangle')  # Default to rectangle for old data
        bbox_type = data['type']

        if shape == 'polygon':
            points = data.get('points', None)
            return BoundingBox(rect=None, bbox_type=bbox_type, shape='polygon', points=points)
        else:
            # Rectangle mode (old format or explicit rectangle)
            rect = QRect(data['x'], data['y'], data['width'], data['height'])
            return BoundingBox(rect=rect, bbox_type=bbox_type, shape='rectangle')

    def to_dict(self):
        """Save to JSON format"""
        result = {
            'type': self.bbox_type,
            'shape': self.shape
        }

        if self.shape == 'polygon' and self.points is not None:
            result['points'] = self.points
        else:
            # Save rectangle data
            result['x'] = self.rect.x()
            result['y'] = self.rect.y()
            result['width'] = self.rect.width()
            result['height'] = self.rect.height()

        return result

    def get_polygon_points(self):
        """Get polygon points as numpy array (for drawing)"""
        if self.shape == 'polygon' and self.points is not None:
            return np.array(self.points, dtype=np.int32)
        else:
            # Convert rectangle to polygon
            x, y, w, h = self.rect.x(), self.rect.y(), self.rect.width(), self.rect.height()
            return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)


class TemplateMatcher:
    def __init__(self, template_image_path, template_bboxes):
        self.template_image = cv2.imread(template_image_path)
        self.template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
        self.template_bboxes = template_bboxes

        self.template_bbox = None
        self.template_region = None
        self.other_bboxes = []

        self._extract_template_data()

    def _extract_template_data(self):
        for bbox in self.template_bboxes:
            if bbox.bbox_type == 'template':
                self.template_bbox = bbox
            else:
                self.other_bboxes.append(bbox)

        if self.template_bbox:
            x = self.template_bbox.rect.x()
            y = self.template_bbox.rect.y()
            w = self.template_bbox.rect.width()
            h = self.template_bbox.rect.height()
            self.template_region = self.template_gray[y:y+h, x:x+w]

    def match_simple(self, target_gray):
        result = cv2.matchTemplate(target_gray, self.template_region, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return max_loc, max_val, None

    def match_multi_scale(self, target_gray):
        scales = np.linspace(0.5, 2.0, 20)
        best_match = None
        best_score = -1
        best_scale = 1.0

        h, w = self.template_region.shape
        for scale in scales:
            if scale < 1.0:
                resized = cv2.resize(target_gray, None, fx=scale, fy=scale)
                if resized.shape[0] < h or resized.shape[1] < w:
                    continue
                result = cv2.matchTemplate(resized, self.template_region, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                max_loc = (int(max_loc[0] / scale), int(max_loc[1] / scale))
            else:
                resized_template = cv2.resize(self.template_region, None, fx=scale, fy=scale)
                if resized_template.shape[0] > target_gray.shape[0] or resized_template.shape[1] > target_gray.shape[1]:
                    continue
                result = cv2.matchTemplate(target_gray, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_match = max_loc
                best_scale = scale

        return best_match, best_score, None, best_scale

    def match_feature_based(self, target_gray):
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(self.template_region, None)
        kp2, des2 = sift.detectAndCompute(target_gray, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None, 0.0, None

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            return None, 0.0, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return None, 0.0, None

        h, w = self.template_region.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)

        x_coords = transformed[:, 0, 0]
        y_coords = transformed[:, 0, 1]
        x, y = int(x_coords.min()), int(y_coords.min())

        confidence = np.sum(mask) / len(mask) if mask is not None else 0.0

        return (x, y), confidence, H

    def match_superpoint_lightglue(self, target_gray):
        """Match using SuperPoint + LightGlue (deep learning approach)"""
        if not SUPERPOINT_AVAILABLE:
            print("SuperPoint not available, falling back to SIFT")
            return self.match_feature_based(target_gray)

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Initialize SuperPoint + LightGlue
            extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
            matcher = LightGlue(features='superpoint').eval().to(device)

            # Convert to torch tensors (grayscale, normalized to [0, 1])
            template_tensor = torch.from_numpy(self.template_region).float()[None, None] / 255.0
            target_tensor = torch.from_numpy(target_gray).float()[None, None] / 255.0

            template_tensor = template_tensor.to(device)
            target_tensor = target_tensor.to(device)

            # Extract features
            feats0 = extractor.extract(template_tensor)
            feats1 = extractor.extract(target_tensor)

            # Match features
            matches01 = matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

            # Get matched keypoints
            kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

            # Convert to numpy
            src_pts = m_kpts0.cpu().numpy().reshape(-1, 1, 2)
            dst_pts = m_kpts1.cpu().numpy().reshape(-1, 1, 2)

            if len(src_pts) < 10:
                return None, 0.0, None

            # Compute homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                return None, 0.0, None

            h, w = self.template_region.shape
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, H)

            x_coords = transformed[:, 0, 0]
            y_coords = transformed[:, 0, 1]
            x, y = int(x_coords.min()), int(y_coords.min())

            confidence = np.sum(mask) / len(mask) if mask is not None else 0.0

            return (x, y), confidence, H

        except Exception as e:
            print(f"SuperPoint matching failed: {e}")
            print("Falling back to SIFT")
            return self.match_feature_based(target_gray)

    def _is_homography_near_similarity(self, H, threshold_angle=3, threshold_scale=0.1, debug=False):
        """
        Kiểm tra xem homography có gần với similarity transform không
        (chỉ có rotation + scale + translation, không có perspective/shear)

        Args:
            H: 3x3 homography matrix
            threshold_angle: Góc nghiêng tối đa (degrees) để coi là "thẳng"
            threshold_scale: Scale difference tối đa để coi là "uniform"
            debug: In ra thông tin debug

        Returns:
            bool: True nếu gần similarity (không cần polygon)
        """
        # Normalize H
        H = H / H[2, 2]

        a = H[0, 0]
        b = H[0, 1]
        c = H[1, 0]
        d = H[1, 1]

        # Check 1: Perspective components (should be near zero)
        perspective_h20 = abs(H[2, 0])
        perspective_h21 = abs(H[2, 1])

        # Nới lỏng threshold - giá trị < 0.001 thường là noise/numerical error
        if perspective_h20 > 0.001 or perspective_h21 > 0.001:
            if debug:
                print(f"  ✗ Perspective detected: h20={perspective_h20:.6f}, h21={perspective_h21:.6f}")
            return False
        elif debug:
            print(f"  ✓ Perspective negligible: h20={perspective_h20:.6f}, h21={perspective_h21:.6f}")

        # Check 2: Compute scales
        scale_x = np.sqrt(a**2 + c**2)
        scale_y = np.sqrt(b**2 + d**2)

        scale_diff = abs(scale_x - scale_y) / max(scale_x, scale_y)
        if scale_diff > threshold_scale:
            if debug:
                print(f"  ✗ Non-uniform scale: sx={scale_x:.3f}, sy={scale_y:.3f}, diff={scale_diff:.3f}")
            return False
        elif debug:
            print(f"  ✓ Scale uniform: sx={scale_x:.3f}, sy={scale_y:.3f}, diff={scale_diff:.3f}")

        # Check 3: Compute rotation angle
        angle = np.arctan2(c, a) * 180 / np.pi

        if abs(angle) > threshold_angle:
            if debug:
                print(f"  ✗ Rotation too large: {angle:.2f}°")
            return False
        elif debug:
            print(f"  ✓ Rotation acceptable: {angle:.2f}°")

        # Check 4: Check for shear/skew
        # For similarity: a*d - b*c should equal scale_x * scale_y (no shear)
        det = a * d - b * c
        expected_det = scale_x * scale_y
        det_diff = abs(det - expected_det) / expected_det if expected_det != 0 else 1

        if det_diff > 0.1:
            if debug:
                print(f"  ✗ Shear/skew detected: det={det:.3f}, expected={expected_det:.3f}, diff={det_diff:.3f}")
            return False
        elif debug:
            print(f"  ✓ No shear: det={det:.3f}, expected={expected_det:.3f}, diff={det_diff:.3f}")

        # Check 5: Orthogonality check (for similarity, rotation matrix should be orthogonal)
        # a*b + c*d should be near 0
        dot_product = abs(a * b + c * d)
        if dot_product > 0.01:
            if debug:
                print(f"  ✗ Non-orthogonal: dot={dot_product:.6f}")
            return False
        elif debug:
            print(f"  ✓ Orthogonal: dot={dot_product:.6f}")

        if debug:
            print(f"  ✅ DECISION: Use RECTANGLE (near similarity transform)")

        return True

    def _transform_bbox_with_homography(self, bbox, H, template_offset_x, template_offset_y, debug=False):
        # Get corners from either rectangle or polygon
        if bbox.shape == 'polygon' and bbox.points is not None:
            # Template có polygon → dùng polygon points
            corners = np.float32([
                [pt[0] - template_offset_x, pt[1] - template_offset_y]
                for pt in bbox.points
            ]).reshape(-1, 1, 2)
        else:
            # Template có rectangle → convert to corners
            x = bbox.rect.x() - template_offset_x
            y = bbox.rect.y() - template_offset_y
            w = bbox.rect.width()
            h = bbox.rect.height()

            corners = np.float32([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ]).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(corners, H)
        polygon_points = transformed.reshape(-1, 2).tolist()

        # Check if homography is near similarity (no significant perspective)
        is_near_similarity = self._is_homography_near_similarity(H, debug=debug)

        if is_near_similarity:
            # Use rectangle instead of polygon
            x_coords = [p[0] for p in polygon_points]
            y_coords = [p[1] for p in polygon_points]
            new_rect = QRect(
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords) - min(x_coords)),
                int(max(y_coords) - min(y_coords))
            )
            if debug:
                print(f"  → Using RECTANGLE for {bbox.bbox_type}")
            return BoundingBox(rect=new_rect, bbox_type=bbox.bbox_type, shape='rectangle')
        else:
            # Use polygon for perspective distortion
            if debug:
                print(f"  → Using POLYGON for {bbox.bbox_type}")
            return BoundingBox(rect=None, bbox_type=bbox.bbox_type, shape='polygon', points=polygon_points)

    def match(self, target_image_path, method='auto', threshold=0.7, debug=False):
        target_image = cv2.imread(target_image_path)
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        homography_matrix = None
        scale = 1.0

        if method == 'simple':
            max_loc, confidence, _ = self.match_simple(target_gray)
        elif method == 'multi_scale':
            max_loc, confidence, _, scale = self.match_multi_scale(target_gray)
        elif method == 'feature':
            max_loc, confidence, homography_matrix = self.match_feature_based(target_gray)
        elif method == 'superpoint':
            max_loc, confidence, homography_matrix = self.match_superpoint_lightglue(target_gray)
        else:
            results = []

            loc1, conf1, _ = self.match_simple(target_gray)
            results.append(('simple', loc1, conf1, None, 1.0))

            loc2, conf2, _, scale2 = self.match_multi_scale(target_gray)
            results.append(('multi_scale', loc2, conf2, None, scale2))

            loc3, conf3, H3 = self.match_feature_based(target_gray)
            if loc3 is not None:
                results.append(('feature', loc3, conf3, H3, 1.0))

            # Try SuperPoint if available
            if SUPERPOINT_AVAILABLE:
                loc4, conf4, H4 = self.match_superpoint_lightglue(target_gray)
                if loc4 is not None:
                    results.append(('superpoint', loc4, conf4, H4, 1.0))

            results.sort(key=lambda x: x[2], reverse=True)
            method_name, max_loc, confidence, homography_matrix, scale = results[0]
            print(f"Auto-selected: {method_name} (conf: {confidence:.3f})")

        if max_loc is None or confidence < threshold:
            return None, confidence, target_image

        transformed_bboxes = []

        if homography_matrix is not None:
            template_x_orig = self.template_bbox.rect.x()
            template_y_orig = self.template_bbox.rect.y()

            if debug:
                print("\n🔍 Homography Transform Analysis:")

            for bbox in self.other_bboxes:
                if debug:
                    print(f"\nTransforming {bbox.bbox_type}:")
                transformed_bbox = self._transform_bbox_with_homography(
                    bbox, homography_matrix, template_x_orig, template_y_orig, debug=debug
                )
                transformed_bboxes.append(transformed_bbox)

            h, w = self.template_region.shape
            template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            template_transformed = cv2.perspectiveTransform(template_corners, homography_matrix)
            template_polygon = template_transformed.reshape(-1, 2).tolist()

            # Check if homography is near similarity for template too
            if debug:
                print(f"\nTransforming template:")
            if self._is_homography_near_similarity(homography_matrix, debug=debug):
                # Use rectangle
                x_coords = [p[0] for p in template_polygon]
                y_coords = [p[1] for p in template_polygon]
                matched_template_rect = QRect(
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords) - min(x_coords)),
                    int(max(y_coords) - min(y_coords))
                )
                matched_template_bbox = BoundingBox(
                    rect=matched_template_rect,
                    bbox_type='template',
                    shape='rectangle'
                )
            else:
                # Use polygon
                matched_template_bbox = BoundingBox(
                    rect=None,
                    bbox_type='template',
                    shape='polygon',
                    points=template_polygon
                )
            transformed_bboxes.append(matched_template_bbox)

        else:
            template_x_orig = self.template_bbox.rect.x()
            template_y_orig = self.template_bbox.rect.y()

            for bbox in self.other_bboxes:
                relative_x = bbox.rect.x() - template_x_orig
                relative_y = bbox.rect.y() - template_y_orig

                new_x = max_loc[0] + int(relative_x * scale)
                new_y = max_loc[1] + int(relative_y * scale)

                new_rect = QRect(
                    new_x,
                    new_y,
                    int(bbox.rect.width() * scale),
                    int(bbox.rect.height() * scale)
                )
                transformed_bboxes.append(BoundingBox(new_rect, bbox.bbox_type))

            matched_template_rect = QRect(
                max_loc[0],
                max_loc[1],
                int(self.template_bbox.rect.width() * scale),
                int(self.template_bbox.rect.height() * scale)
            )
            transformed_bboxes.append(BoundingBox(matched_template_rect, 'template'))

        return transformed_bboxes, confidence, target_image

    @staticmethod
    def draw_bboxes(image, bboxes, draw_polygon=True):
        colors = {
            'template': (255, 80, 80),
            'text': (80, 255, 120),
            'datecode': (80, 150, 255),
            'barcode': (255, 220, 80)
        }

        result = image.copy()

        for bbox in bboxes:
            color = colors.get(bbox.bbox_type, (255, 255, 255))

            if bbox.shape == 'polygon' and bbox.points is not None and draw_polygon:
                points = np.array(bbox.points, dtype=np.int32)
                cv2.polylines(result, [points], True, color, 2)

                # Draw corner keypoints with different colors
                cv2.circle(result, tuple(points[0]), 5, (0, 255, 0), -1)  # Green
                cv2.circle(result, tuple(points[1]), 5, (255, 0, 0), -1)  # Blue
                cv2.circle(result, tuple(points[2]), 5, (0, 0, 255), -1)  # Red
                cv2.circle(result, tuple(points[3]), 5, (255, 255, 0), -1)  # Yellow

                center_x = int(np.mean(points[:, 0]))
                center_y = int(np.mean(points[:, 1]))
                cv2.putText(result, bbox.bbox_type, (center_x - 30, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Draw rectangle
                x = bbox.rect.x()
                y = bbox.rect.y()
                w = bbox.rect.width()
                h = bbox.rect.height()

                cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
                cv2.putText(result, bbox.bbox_type, (x+5, y+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result

    @staticmethod
    def crop_region_with_perspective(image, bbox):
        if bbox.shape == 'rectangle' or bbox.points is None:
            # Simple rectangular crop
            x, y, w, h = bbox.rect.x(), bbox.rect.y(), bbox.rect.width(), bbox.rect.height()
            return image[y:y+h, x:x+w]

        # Polygon crop with perspective correction
        src_points = np.array(bbox.points, dtype=np.float32)

        width = int(max(
            np.linalg.norm(src_points[0] - src_points[1]),
            np.linalg.norm(src_points[2] - src_points[3])
        ))
        height = int(max(
            np.linalg.norm(src_points[1] - src_points[2]),
            np.linalg.norm(src_points[3] - src_points[0])
        ))

        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, M, (width, height))

        return warped

    def visualize_matching(self, target_image_path):
        target_image = cv2.imread(target_image_path)
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(target_gray, self.template_region, cv2.TM_CCOEFF_NORMED)

        result_normalized = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(result_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        h, w = self.template_region.shape
        heatmap_resized = cv2.resize(heatmap, (target_image.shape[1], target_image.shape[0]))

        overlay = cv2.addWeighted(target_image, 0.6, heatmap_resized, 0.4, 0)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        cv2.rectangle(overlay, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 3)
        cv2.putText(overlay, f"Score: {max_val:.3f}", (max_loc[0], max_loc[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return overlay
