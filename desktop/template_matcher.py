import cv2
import numpy as np
from PyQt5.QtCore import QRect


class BoundingBox:
    def __init__(self, rect, bbox_type):
        self.rect = rect
        self.bbox_type = bbox_type

    @staticmethod
    def from_dict(data):
        rect = QRect(data['x'], data['y'], data['width'], data['height'])
        return BoundingBox(rect, data['type'])

    def to_dict(self):
        return {
            'x': self.rect.x(),
            'y': self.rect.y(),
            'width': self.rect.width(),
            'height': self.rect.height(),
            'type': self.bbox_type
        }


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

    def _transform_bbox_with_homography(self, bbox, H, template_offset_x, template_offset_y):
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

        x_coords = transformed[:, 0, 0]
        y_coords = transformed[:, 0, 1]

        new_x = int(x_coords.min())
        new_y = int(y_coords.min())
        new_w = int(x_coords.max() - x_coords.min())
        new_h = int(y_coords.max() - y_coords.min())

        return QRect(new_x, new_y, new_w, new_h)

    def match(self, target_image_path, method='auto', threshold=0.7):
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
        else:
            results = []

            loc1, conf1, _ = self.match_simple(target_gray)
            results.append(('simple', loc1, conf1, None, 1.0))

            loc2, conf2, _, scale2 = self.match_multi_scale(target_gray)
            results.append(('multi_scale', loc2, conf2, None, scale2))

            loc3, conf3, H3 = self.match_feature_based(target_gray)
            if loc3 is not None:
                results.append(('feature', loc3, conf3, H3, 1.0))

            results.sort(key=lambda x: x[2], reverse=True)
            method_name, max_loc, confidence, homography_matrix, scale = results[0]
            print(f"Auto-selected: {method_name} (conf: {confidence:.3f})")

        if max_loc is None or confidence < threshold:
            return None, confidence, target_image

        transformed_bboxes = []

        if homography_matrix is not None:
            template_x_orig = self.template_bbox.rect.x()
            template_y_orig = self.template_bbox.rect.y()

            for bbox in self.other_bboxes:
                new_rect = self._transform_bbox_with_homography(
                    bbox, homography_matrix, template_x_orig, template_y_orig
                )
                transformed_bboxes.append(BoundingBox(new_rect, bbox.bbox_type))

            h, w = self.template_region.shape
            template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            template_transformed = cv2.perspectiveTransform(template_corners, homography_matrix)

            x_coords = template_transformed[:, 0, 0]
            y_coords = template_transformed[:, 0, 1]
            matched_template_rect = QRect(
                int(x_coords.min()),
                int(y_coords.min()),
                int(x_coords.max() - x_coords.min()),
                int(y_coords.max() - y_coords.min())
            )
            transformed_bboxes.append(BoundingBox(matched_template_rect, 'template'))

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
    def draw_bboxes(image, bboxes):
        colors = {
            'template': (255, 80, 80),
            'text': (80, 255, 120),
            'datecode': (80, 150, 255),
            'barcode': (255, 220, 80)
        }

        result = image.copy()

        for bbox in bboxes:
            color = colors.get(bbox.bbox_type, (255, 255, 255))
            x = bbox.rect.x()
            y = bbox.rect.y()
            w = bbox.rect.width()
            h = bbox.rect.height()

            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result, bbox.bbox_type, (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result

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
