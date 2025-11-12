"""
Annotation Manager
Quản lý lưu và load annotations
"""
import json
import os
from template_matcher import BoundingBox


class AnnotationManager:
    """Class quản lý annotations của các ảnh"""

    def __init__(self):
        self.annotations = {}  # {image_path: [BoundingBox, ...]}
        self.annotations_file = None
        self.template_image_path = None  # Đường dẫn ảnh template

    def load_annotations(self, images_folder):
        """Load annotations từ file JSON"""
        self.annotations_file = os.path.join(images_folder, 'annotations.json')

        if os.path.exists(self.annotations_file):
            try:
                with open(self.annotations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load template image path nếu có
                self.template_image_path = data.get('_template_image', None)

                # Chuyển đổi từ dict sang BoundingBox objects
                self.annotations = {}
                for image_path, bboxes_data in data.items():
                    if image_path == '_template_image':
                        continue  # Skip metadata
                    bboxes = [BoundingBox.from_dict(bbox_data) for bbox_data in bboxes_data]
                    self.annotations[image_path] = bboxes

                print(f"Loaded annotations from {self.annotations_file}")
                if self.template_image_path:
                    print(f"Template image: {self.template_image_path}")
                return True
            except Exception as e:
                print(f"Error loading annotations: {e}")
                self.annotations = {}
                self.template_image_path = None
                return False
        else:
            self.annotations = {}
            self.template_image_path = None
            return True

    def save_annotations(self):
        """Lưu annotations ra file JSON"""
        if not self.annotations_file:
            return False

        try:
            # Chuyển đổi BoundingBox objects sang dict
            data = {}

            # Lưu template image path nếu có
            if self.template_image_path:
                data['_template_image'] = self.template_image_path

            for image_path, bboxes in self.annotations.items():
                if bboxes:  # Chỉ lưu những ảnh có bbox
                    data[image_path] = [bbox.to_dict() for bbox in bboxes]

            # Lưu ra file
            with open(self.annotations_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"Saved annotations to {self.annotations_file}")
            return True
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False

    def get_annotations(self, image_path):
        """Lấy annotations của một ảnh"""
        return self.annotations.get(image_path, [])

    def set_annotations(self, image_path, bboxes):
        """Set annotations cho một ảnh"""
        if bboxes:
            self.annotations[image_path] = bboxes
        elif image_path in self.annotations:
            # Xóa nếu không còn bbox
            del self.annotations[image_path]

    def has_annotations(self, image_path):
        """Kiểm tra ảnh có annotations không"""
        return image_path in self.annotations and len(self.annotations[image_path]) > 0

    def get_annotations_file(self):
        """Lấy đường dẫn file annotations"""
        return self.annotations_file

    def get_statistics(self):
        """Lấy thống kê về annotations"""
        stats = {
            'total_images': len(self.annotations),
            'total_bboxes': sum(len(bboxes) for bboxes in self.annotations.values()),
            'by_type': {
                'template': 0,
                'text': 0,
                'datecode': 0,
                'barcode': 0
            }
        }

        for bboxes in self.annotations.values():
            for bbox in bboxes:
                if bbox.bbox_type in stats['by_type']:
                    stats['by_type'][bbox.bbox_type] += 1

        return stats

    def set_template_image(self, image_path):
        """Đặt ảnh làm template"""
        self.template_image_path = image_path

    def unset_template_image(self):
        """Bỏ template image"""
        self.template_image_path = None

    def is_template_image(self, image_path):
        """Kiểm tra xem ảnh có phải là template không"""
        return self.template_image_path == image_path

    def get_template_image(self):
        """Lấy đường dẫn template image"""
        return self.template_image_path
