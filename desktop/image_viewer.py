"""
Image Viewer Widget
Widget để hiển thị ảnh và vẽ bounding boxes với khả năng di chuyển, resize, select
"""
from PyQt5.QtWidgets import QLabel, QMessageBox
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QFont, QPolygon
from type_dialog import TypeDialog
from template_matcher import BoundingBox


class ImageViewer(QLabel):
    """Widget để hiển thị ảnh và cho phép vẽ bounding boxes"""

    # Signals
    bboxCreated = pyqtSignal()
    bboxSelected = pyqtSignal(object)  # Emit bbox khi được chọn
    bboxesChanged = pyqtSignal()  # Emit khi có thay đổi

    # Màu sắc cho từng loại bbox
    BBOX_COLORS = {
        'template': QColor(255, 80, 80),      # Đỏ sáng
        'text': QColor(80, 255, 120),         # Xanh lá sáng
        'datecode': QColor(80, 150, 255),     # Xanh dương sáng
        'barcode': QColor(255, 220, 80)       # Vàng sáng
    }

    # Kích thước handle để resize
    HANDLE_SIZE = 8

    # Modes
    MODE_DRAW = 0
    MODE_SELECT = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setObjectName("imageViewer")

        # State
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.bboxes = []  # Danh sách các BoundingBox
        self.selected_bbox = None  # BBox đang được chọn

        # Drawing state
        self.mode = self.MODE_SELECT  # Default to select mode
        self.drawing = False
        self.start_point = None
        self.current_rect = None

        # Moving/Resizing state
        self.moving = False
        self.resizing = False
        self.resize_handle = None  # 'tl', 'tr', 'bl', 'br', 'l', 'r', 't', 'b'
        self.drag_start_pos = None
        self.drag_start_rect = None

        # Polygon editing state
        self.editing_polygon_point = False
        self.polygon_point_index = None  # Index của điểm đang kéo (0-3)

        # Zoom state
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # Drawing shape
        self.draw_shape = 'rectangle'  # 'rectangle' or 'polygon'
        self.polygon_points = []  # For polygon drawing (4 points)

    def set_image(self, image_path):
        """Load và hiển thị ảnh"""
        self.original_pixmap = QPixmap(image_path)
        if self.original_pixmap.isNull():
            QMessageBox.warning(self, "Lỗi", f"Không thể load ảnh: {image_path}")
            return False

        self.bboxes = []
        self.selected_bbox = None
        self.zoom_factor = 1.0  # Reset zoom
        self.update_display()
        return True

    def update_display(self):
        """Cập nhật hiển thị với scaling và vẽ bboxes"""
        if self.original_pixmap is None:
            return

        # Scale ảnh cho vừa với widget, có tính zoom
        base_size = self.size()
        zoomed_width = int(base_size.width() * self.zoom_factor)
        zoomed_height = int(base_size.height() * self.zoom_factor)

        self.scaled_pixmap = self.original_pixmap.scaled(
            zoomed_width,
            zoomed_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Vẽ bboxes lên ảnh
        pixmap_with_boxes = self.scaled_pixmap.copy()
        painter = QPainter(pixmap_with_boxes)
        painter.setRenderHint(QPainter.Antialiasing)

        # Tính tỷ lệ scale
        scale_x = self.scaled_pixmap.width() / self.original_pixmap.width()
        scale_y = self.scaled_pixmap.height() / self.original_pixmap.height()

        # Vẽ các bbox (không được chọn trước)
        for bbox in self.bboxes:
            if bbox != self.selected_bbox:
                self.draw_bbox(painter, bbox, scale_x, scale_y, False)

        # Vẽ selected bbox sau cùng (để nó nổi bật)
        if self.selected_bbox:
            self.draw_bbox(painter, self.selected_bbox, scale_x, scale_y, True)

        # Vẽ bbox đang được vẽ
        if self.drawing and self.current_rect:
            pen = QPen(QColor(255, 255, 255), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)

        # Vẽ polygon points đang được vẽ
        if len(self.polygon_points) > 0:
            # Convert original coords to scaled coords
            scaled_polygon_points = []
            for pt in self.polygon_points:
                scaled_x = int(pt[0] * scale_x)
                scaled_y = int(pt[1] * scale_y)
                scaled_polygon_points.append(QPoint(scaled_x, scaled_y))

            # Vẽ các điểm đã click
            painter.setBrush(QBrush(QColor(0, 255, 0)))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            for i, pt in enumerate(scaled_polygon_points):
                painter.drawEllipse(pt, 5, 5)
                # Vẽ số thứ tự
                painter.drawText(pt + QPoint(8, 5), str(i + 1))

            # Vẽ đường nối giữa các điểm
            if len(scaled_polygon_points) > 1:
                pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
                painter.setPen(pen)
                for i in range(len(scaled_polygon_points) - 1):
                    painter.drawLine(scaled_polygon_points[i], scaled_polygon_points[i + 1])

            # Vẽ đường nối điểm cuối với điểm đầu nếu đã có ít nhất 3 điểm
            if len(scaled_polygon_points) >= 3:
                pen = QPen(QColor(0, 255, 0), 2, Qt.DotLine)
                painter.setPen(pen)
                painter.drawLine(scaled_polygon_points[-1], scaled_polygon_points[0])

        painter.end()
        self.setPixmap(pixmap_with_boxes)

    def draw_bbox(self, painter, bbox, scale_x, scale_y, is_selected):
        """Vẽ một bounding box (rectangle hoặc polygon)"""
        pixmap_rect = QRect(0, 0, self.scaled_pixmap.width(), self.scaled_pixmap.height())
        color = self.BBOX_COLORS.get(bbox.bbox_type, QColor(255, 255, 255))

        if bbox.shape == 'polygon' and bbox.points is not None:
            # Vẽ polygon
            scaled_points = []
            for pt in bbox.points:
                scaled_x = int(pt[0] * scale_x)
                scaled_y = int(pt[1] * scale_y)
                scaled_points.append(QPoint(scaled_x, scaled_y))

            polygon = QPolygon(scaled_points)

            if is_selected:
                pen = QPen(color, 3)
                painter.setPen(pen)
                painter.drawPolygon(polygon)

                # Vẽ keypoints tại 4 góc với handles lớn hơn để dễ kéo
                corner_colors = [
                    QColor(0, 255, 0),    # Green
                    QColor(255, 0, 0),    # Blue
                    QColor(0, 0, 255),    # Red
                    QColor(255, 255, 0)   # Yellow
                ]
                for i, pt in enumerate(scaled_points):
                    painter.setBrush(QBrush(corner_colors[i]))
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.drawEllipse(pt, 7, 7)  # Lớn hơn để dễ click
                    # Vẽ số thứ tự
                    painter.setPen(QPen(QColor(255, 255, 255)))
                    painter.drawText(pt + QPoint(10, 5), str(i + 1))
            else:
                pen = QPen(color, 2)
                painter.setPen(pen)
                painter.drawPolygon(polygon)

            # Vẽ label
            center_x = sum(pt.x() for pt in scaled_points) // len(scaled_points)
            center_y = sum(pt.y() for pt in scaled_points) // len(scaled_points)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(QPoint(center_x - 30, center_y), bbox.bbox_type)

        else:
            # Vẽ rectangle (chế độ cũ)
            scaled_rect = QRect(
                int(bbox.rect.x() * scale_x),
                int(bbox.rect.y() * scale_y),
                int(bbox.rect.width() * scale_x),
                int(bbox.rect.height() * scale_y)
            )

            # Giới hạn trong pixmap bounds
            scaled_rect = scaled_rect.intersected(pixmap_rect)

            if is_selected:
                # Vẽ bbox được chọn với hiệu ứng nổi bật
                pen = QPen(color, 3)
                painter.setPen(pen)
                painter.drawRect(scaled_rect)

                # Vẽ các handle để resize
                self.draw_handles(painter, scaled_rect, color)

                # Vẽ label với background
                font = QFont()
                font.setBold(True)
                painter.setFont(font)

                label_text = f"{bbox.bbox_type}"
                metrics = painter.fontMetrics()
                label_rect = metrics.boundingRect(label_text)
                label_rect.moveTopLeft(scaled_rect.topLeft() + QPoint(0, -label_rect.height() - 2))
                label_rect.adjust(-4, -2, 4, 2)

                # Đảm bảo label nằm trong pixmap
                if label_rect.top() < 0:
                    label_rect.moveTop(scaled_rect.top() + 2)
                if label_rect.left() < 0:
                    label_rect.moveLeft(2)
                if label_rect.right() > pixmap_rect.right():
                    label_rect.moveRight(pixmap_rect.right() - 2)

                painter.fillRect(label_rect, color)
                painter.setPen(QPen(QColor(0, 0, 0)))
                painter.drawText(label_rect, Qt.AlignCenter, label_text)
            else:
                # Vẽ bbox thường
                pen = QPen(color, 2)
                painter.setPen(pen)
                painter.drawRect(scaled_rect)

                # Vẽ label nhỏ
                label_pos = scaled_rect.topLeft() + QPoint(5, 15)
                # Giới hạn label trong pixmap
                if label_pos.x() + 50 > pixmap_rect.right():
                    label_pos.setX(pixmap_rect.right() - 50)
                if label_pos.y() > pixmap_rect.bottom():
                    label_pos.setY(pixmap_rect.bottom() - 5)

                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(label_pos, bbox.bbox_type)

    def draw_handles(self, painter, rect, color):
        """Vẽ các handle để resize"""
        handle_color = color
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setBrush(QBrush(handle_color))

        handles = self.get_resize_handles(rect)
        for handle_rect in handles.values():
            painter.drawRect(handle_rect)

    def get_resize_handles(self, rect):
        """Lấy vị trí các handle để resize"""
        hs = self.HANDLE_SIZE
        handles = {
            'tl': QRect(rect.left() - hs//2, rect.top() - hs//2, hs, hs),
            'tr': QRect(rect.right() - hs//2, rect.top() - hs//2, hs, hs),
            'bl': QRect(rect.left() - hs//2, rect.bottom() - hs//2, hs, hs),
            'br': QRect(rect.right() - hs//2, rect.bottom() - hs//2, hs, hs),
            'l': QRect(rect.left() - hs//2, rect.center().y() - hs//2, hs, hs),
            'r': QRect(rect.right() - hs//2, rect.center().y() - hs//2, hs, hs),
            't': QRect(rect.center().x() - hs//2, rect.top() - hs//2, hs, hs),
            'b': QRect(rect.center().x() - hs//2, rect.bottom() - hs//2, hs, hs),
        }
        return handles

    def set_mode(self, mode):
        """Đặt chế độ (draw/select)"""
        self.mode = mode
        if mode == self.MODE_DRAW:
            self.setCursor(Qt.CrossCursor)
            self.selected_bbox = None
            self.polygon_points = []  # Reset polygon points
        else:
            self.setCursor(Qt.ArrowCursor)
            self.polygon_points = []
        self.update_display()

    def set_draw_shape(self, shape):
        """Đặt shape để vẽ ('rectangle' or 'polygon')"""
        self.draw_shape = shape
        self.polygon_points = []  # Reset polygon points when changing shape

    def mousePressEvent(self, event):
        """Xử lý khi nhấn chuột"""
        if self.original_pixmap is None:
            return

        pixmap_pos = self.get_pixmap_position(event.pos())
        if pixmap_pos is None:
            return

        if event.button() == Qt.LeftButton:
            if self.mode == self.MODE_DRAW:
                if self.draw_shape == 'rectangle':
                    # Chế độ vẽ rectangle
                    self.drawing = True
                    self.start_point = pixmap_pos
                    self.current_rect = QRect(pixmap_pos, pixmap_pos)
                elif self.draw_shape == 'polygon':
                    # Chế độ vẽ polygon (4 điểm)
                    # Convert pixmap coords to original image coords
                    scale_x = self.original_pixmap.width() / self.scaled_pixmap.width()
                    scale_y = self.original_pixmap.height() / self.scaled_pixmap.height()
                    original_x = int(pixmap_pos.x() * scale_x)
                    original_y = int(pixmap_pos.y() * scale_y)

                    self.polygon_points.append([original_x, original_y])

                    # Nếu đã đủ 4 điểm, hiển thị dialog để chọn type
                    if len(self.polygon_points) == 4:
                        dialog = TypeDialog(self)
                        if dialog.exec_():
                            bbox_type = dialog.get_selected_type()
                            bbox = BoundingBox(
                                rect=None,
                                bbox_type=bbox_type,
                                shape='polygon',
                                points=self.polygon_points.copy()
                            )
                            self.bboxes.append(bbox)
                            self.bboxCreated.emit()
                            self.bboxesChanged.emit()

                        # Reset polygon points
                        self.polygon_points = []

                    self.update_display()

            elif self.mode == self.MODE_SELECT:
                # Chế độ select/move/resize
                scale_x = self.scaled_pixmap.width() / self.original_pixmap.width()
                scale_y = self.scaled_pixmap.height() / self.original_pixmap.height()

                # Kiểm tra xem có click vào handle không (nếu có bbox được chọn)
                if self.selected_bbox:
                    if self.selected_bbox.shape == 'polygon' and self.selected_bbox.points is not None:
                        # Polygon editing: check click vào keypoint
                        scaled_points = []
                        for pt in self.selected_bbox.points:
                            scaled_x = int(pt[0] * scale_x)
                            scaled_y = int(pt[1] * scale_y)
                            scaled_points.append(QPoint(scaled_x, scaled_y))

                        # Kiểm tra click vào điểm nào
                        for i, pt in enumerate(scaled_points):
                            distance = (pixmap_pos - pt).manhattanLength()
                            if distance <= 10:  # Trong vòng 10px
                                self.editing_polygon_point = True
                                self.polygon_point_index = i
                                self.drag_start_pos = pixmap_pos
                                return

                    elif self.selected_bbox.shape == 'rectangle':
                        # Rectangle editing: resize/move
                        scaled_rect = self.get_scaled_rect(self.selected_bbox.rect, scale_x, scale_y)
                        handles = self.get_resize_handles(scaled_rect)

                        for handle_name, handle_rect in handles.items():
                            if handle_rect.contains(pixmap_pos):
                                self.resizing = True
                                self.resize_handle = handle_name
                                self.drag_start_pos = pixmap_pos
                                self.drag_start_rect = QRect(self.selected_bbox.rect)
                                return

                        # Kiểm tra click vào trong bbox để di chuyển
                        if scaled_rect.contains(pixmap_pos):
                            self.moving = True
                            self.drag_start_pos = pixmap_pos
                            self.drag_start_rect = QRect(self.selected_bbox.rect)
                            return

                # Kiểm tra click vào bbox nào
                clicked_bbox = self.get_bbox_at_position(pixmap_pos, scale_x, scale_y)
                if clicked_bbox:
                    self.selected_bbox = clicked_bbox
                    self.bboxSelected.emit(clicked_bbox)
                    self.update_display()
                else:
                    self.selected_bbox = None
                    self.bboxSelected.emit(None)
                    self.update_display()

    def mouseMoveEvent(self, event):
        """Xử lý khi di chuyển chuột"""
        pixmap_pos = self.get_pixmap_position(event.pos())
        if pixmap_pos is None:
            return

        if self.drawing:
            # Vẽ bbox mới
            self.current_rect = QRect(self.start_point, pixmap_pos).normalized()
            self.update_display()

        elif self.editing_polygon_point and self.selected_bbox:
            # Kéo polygon point
            scale_x = self.original_pixmap.width() / self.scaled_pixmap.width()
            scale_y = self.original_pixmap.height() / self.scaled_pixmap.height()

            # Convert pixmap coords to original coords
            original_x = int(pixmap_pos.x() * scale_x)
            original_y = int(pixmap_pos.y() * scale_y)

            # Giới hạn trong ảnh
            original_x = max(0, min(original_x, self.original_pixmap.width()))
            original_y = max(0, min(original_y, self.original_pixmap.height()))

            # Cập nhật điểm
            self.selected_bbox.points[self.polygon_point_index] = [original_x, original_y]

            # Cập nhật bounding rect
            xs = [p[0] for p in self.selected_bbox.points]
            ys = [p[1] for p in self.selected_bbox.points]
            from PyQt5.QtCore import QRect
            self.selected_bbox.rect = QRect(
                int(min(xs)), int(min(ys)),
                int(max(xs) - min(xs)), int(max(ys) - min(ys))
            )

            self.update_display()

        elif self.moving and self.selected_bbox:
            # Di chuyển bbox
            scale_x = self.original_pixmap.width() / self.scaled_pixmap.width()
            scale_y = self.original_pixmap.height() / self.scaled_pixmap.height()

            delta = pixmap_pos - self.drag_start_pos
            delta_original = QPoint(int(delta.x() * scale_x), int(delta.y() * scale_y))

            new_rect = QRect(self.drag_start_rect)
            new_rect.translate(delta_original)

            # Giới hạn trong ảnh
            if new_rect.left() < 0:
                new_rect.moveLeft(0)
            if new_rect.top() < 0:
                new_rect.moveTop(0)
            if new_rect.right() > self.original_pixmap.width():
                new_rect.moveRight(self.original_pixmap.width())
            if new_rect.bottom() > self.original_pixmap.height():
                new_rect.moveBottom(self.original_pixmap.height())

            self.selected_bbox.rect = new_rect
            self.update_display()

        elif self.resizing and self.selected_bbox:
            # Resize bbox
            scale_x = self.original_pixmap.width() / self.scaled_pixmap.width()
            scale_y = self.original_pixmap.height() / self.scaled_pixmap.height()

            delta = pixmap_pos - self.drag_start_pos
            delta_original = QPoint(int(delta.x() * scale_x), int(delta.y() * scale_y))

            new_rect = QRect(self.drag_start_rect)

            # Resize theo handle
            if 'l' in self.resize_handle:
                new_rect.setLeft(self.drag_start_rect.left() + delta_original.x())
            if 'r' in self.resize_handle:
                new_rect.setRight(self.drag_start_rect.right() + delta_original.x())
            if 't' in self.resize_handle:
                new_rect.setTop(self.drag_start_rect.top() + delta_original.y())
            if 'b' in self.resize_handle:
                new_rect.setBottom(self.drag_start_rect.bottom() + delta_original.y())

            # Normalize và giới hạn kích thước tối thiểu
            new_rect = new_rect.normalized()
            if new_rect.width() >= 10 and new_rect.height() >= 10:
                # Giới hạn trong ảnh
                if new_rect.left() < 0:
                    new_rect.setLeft(0)
                if new_rect.top() < 0:
                    new_rect.setTop(0)
                if new_rect.right() > self.original_pixmap.width():
                    new_rect.setRight(self.original_pixmap.width())
                if new_rect.bottom() > self.original_pixmap.height():
                    new_rect.setBottom(self.original_pixmap.height())

                self.selected_bbox.rect = new_rect
                self.update_display()

        else:
            # Cập nhật cursor khi hover
            if self.mode == self.MODE_SELECT and self.selected_bbox:
                scale_x = self.scaled_pixmap.width() / self.original_pixmap.width()
                scale_y = self.scaled_pixmap.height() / self.original_pixmap.height()
                scaled_rect = self.get_scaled_rect(self.selected_bbox.rect, scale_x, scale_y)
                handles = self.get_resize_handles(scaled_rect)

                cursor_set = False
                for handle_name, handle_rect in handles.items():
                    if handle_rect.contains(pixmap_pos):
                        # Set cursor theo hướng resize
                        if handle_name in ['tl', 'br']:
                            self.setCursor(Qt.SizeFDiagCursor)
                        elif handle_name in ['tr', 'bl']:
                            self.setCursor(Qt.SizeBDiagCursor)
                        elif handle_name in ['l', 'r']:
                            self.setCursor(Qt.SizeHorCursor)
                        elif handle_name in ['t', 'b']:
                            self.setCursor(Qt.SizeVerCursor)
                        cursor_set = True
                        break

                if not cursor_set:
                    if scaled_rect.contains(pixmap_pos):
                        self.setCursor(Qt.SizeAllCursor)
                    else:
                        self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        """Xử lý khi thả chuột"""
        if event.button() != Qt.LeftButton:
            return

        if self.drawing:
            self.drawing = False

            # Kiểm tra rectangle có hợp lệ không
            if self.current_rect and self.current_rect.width() > 5 and self.current_rect.height() > 5:
                # Hiển thị dialog để chọn type
                dialog = TypeDialog(self)
                if dialog.exec_():
                    bbox_type = dialog.get_selected_type()

                    # Chuyển đổi về tọa độ ảnh gốc
                    scale_x = self.original_pixmap.width() / self.scaled_pixmap.width()
                    scale_y = self.original_pixmap.height() / self.scaled_pixmap.height()

                    original_rect = QRect(
                        int(self.current_rect.x() * scale_x),
                        int(self.current_rect.y() * scale_y),
                        int(self.current_rect.width() * scale_x),
                        int(self.current_rect.height() * scale_y)
                    )

                    # Tạo bbox mới (rectangle mode)
                    bbox = BoundingBox(rect=original_rect, bbox_type=bbox_type, shape='rectangle')
                    self.bboxes.append(bbox)

                    # Emit signals
                    self.bboxCreated.emit()
                    self.bboxesChanged.emit()

            self.current_rect = None
            self.update_display()

        elif self.moving or self.resizing or self.editing_polygon_point:
            self.moving = False
            self.resizing = False
            self.editing_polygon_point = False
            self.resize_handle = None
            self.polygon_point_index = None
            self.drag_start_pos = None
            self.drag_start_rect = None
            self.bboxesChanged.emit()

            if self.mode == self.MODE_SELECT:
                self.setCursor(Qt.ArrowCursor)

    def get_scaled_rect(self, original_rect, scale_x, scale_y):
        """Chuyển đổi rect từ tọa độ gốc sang tọa độ scaled"""
        return QRect(
            int(original_rect.x() * scale_x),
            int(original_rect.y() * scale_y),
            int(original_rect.width() * scale_x),
            int(original_rect.height() * scale_y)
        )

    def get_bbox_at_position(self, pos, scale_x, scale_y):
        """Lấy bbox tại vị trí cho trước (vị trí trong scaled pixmap)"""
        # Duyệt ngược để lấy bbox trên cùng
        for bbox in reversed(self.bboxes):
            if bbox.shape == 'polygon' and bbox.points is not None:
                # Kiểm tra point in polygon
                scaled_points = []
                for pt in bbox.points:
                    scaled_x = int(pt[0] * scale_x)
                    scaled_y = int(pt[1] * scale_y)
                    scaled_points.append(QPoint(scaled_x, scaled_y))
                polygon = QPolygon(scaled_points)
                if polygon.containsPoint(pos, Qt.OddEvenFill):
                    return bbox
            else:
                # Rectangle bbox
                scaled_rect = self.get_scaled_rect(bbox.rect, scale_x, scale_y)
                if scaled_rect.contains(pos):
                    return bbox
        return None

    def get_pixmap_position(self, widget_pos):
        """Chuyển đổi tọa độ widget sang tọa độ trong pixmap"""
        if self.scaled_pixmap is None:
            return None

        # Tính offset của pixmap trong label (do alignment center)
        label_width = self.width()
        label_height = self.height()
        pixmap_width = self.scaled_pixmap.width()
        pixmap_height = self.scaled_pixmap.height()

        offset_x = (label_width - pixmap_width) // 2
        offset_y = (label_height - pixmap_height) // 2

        # Tọa độ trong pixmap
        x = widget_pos.x() - offset_x
        y = widget_pos.y() - offset_y

        # Kiểm tra có nằm trong pixmap không
        if 0 <= x < pixmap_width and 0 <= y < pixmap_height:
            return QPoint(x, y)
        return None

    def delete_selected_bbox(self):
        """Xóa bbox đang được chọn"""
        if self.selected_bbox and self.selected_bbox in self.bboxes:
            self.bboxes.remove(self.selected_bbox)
            self.selected_bbox = None
            self.bboxSelected.emit(None)
            self.bboxesChanged.emit()
            self.update_display()
            return True
        return False

    def select_bbox_by_id(self, bbox_id):
        """Chọn bbox theo ID"""
        for bbox in self.bboxes:
            if bbox.bbox_id == bbox_id:
                self.selected_bbox = bbox
                self.bboxSelected.emit(bbox)
                self.update_display()
                return True
        return False

    def get_bboxes(self):
        """Lấy danh sách các bounding boxes"""
        return self.bboxes

    def set_bboxes(self, bboxes):
        """Set danh sách các bounding boxes"""
        self.bboxes = bboxes
        self.selected_bbox = None
        self.update_display()

    def clear_bboxes(self):
        """Xóa tất cả bounding boxes"""
        self.bboxes = []
        self.selected_bbox = None
        self.bboxSelected.emit(None)
        self.bboxesChanged.emit()
        self.update_display()

    def resizeEvent(self, event):
        """Xử lý khi resize widget"""
        super().resizeEvent(event)
        self.update_display()

    def zoom_in(self):
        """Zoom in ảnh"""
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor = min(self.zoom_factor * 1.2, self.max_zoom)
            self.update_display()
            return True
        return False

    def zoom_out(self):
        """Zoom out ảnh"""
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor = max(self.zoom_factor / 1.2, self.min_zoom)
            self.update_display()
            return True
        return False

    def reset_zoom(self):
        """Reset zoom về 100%"""
        self.zoom_factor = 1.0
        self.update_display()

    def get_zoom_percent(self):
        """Lấy zoom percent hiện tại"""
        return int(self.zoom_factor * 100)

    def wheelEvent(self, event):
        """Xử lý scroll wheel để zoom"""
        if event.modifiers() == Qt.ControlModifier:
            # Ctrl + Scroll để zoom
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        """Xử lý phím tắt"""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            if self.delete_selected_bbox():
                return
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            if event.modifiers() == Qt.ControlModifier:
                self.zoom_in()
                return
        elif event.key() == Qt.Key_Minus:
            if event.modifiers() == Qt.ControlModifier:
                self.zoom_out()
                return
        elif event.key() == Qt.Key_0:
            if event.modifiers() == Qt.ControlModifier:
                self.reset_zoom()
                return
        super().keyPressEvent(event)
