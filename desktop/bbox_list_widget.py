"""
BBox List Widget
Widget để hiển thị danh sách các bounding boxes
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
                              QListWidgetItem, QPushButton, QLabel)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor


class BBoxListWidget(QWidget):
    """Widget hiển thị danh sách bounding boxes"""

    # Signals
    bboxSelected = pyqtSignal(object)  # Emit khi chọn bbox từ list
    deleteRequested = pyqtSignal(object)  # Emit khi muốn xóa bbox

    # Màu sắc cho từng loại
    BBOX_COLORS = {
        'template': '#ff5050',
        'text': '#50ff78',
        'datecode': '#5096ff',
        'barcode': '#ffdc50'
    }

    # Icon cho từng loại
    BBOX_ICONS = {
        'template': '',
        'text': '',
        'datecode': '',
        'barcode': ''
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("bboxPanel")
        self.bboxes = []
        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel("Bounding Boxes")
        header.setObjectName("headerLabel")
        layout.addWidget(header)

        # List widget
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("bboxList")
        self.list_widget.setWordWrap(True)
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        layout.addWidget(self.list_widget)

        # Info label
        self.info_label = QLabel("No bounding boxes")
        self.info_label.setObjectName("infoLabel")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setObjectName("dangerButton")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self.on_delete_clicked)
        button_layout.addWidget(self.delete_btn)

        layout.addLayout(button_layout)

    def update_bboxes(self, bboxes):
        """Cập nhật danh sách bounding boxes"""
        self.bboxes = bboxes
        self.refresh_list()

    def refresh_list(self):
        """Làm mới danh sách hiển thị"""
        self.list_widget.clear()

        if not self.bboxes:
            self.info_label.setText("No bounding boxes")
            self.delete_btn.setEnabled(False)
            return

        for i, bbox in enumerate(self.bboxes):
            # Tạo item text
            text = f"BBox #{i+1} - {bbox.bbox_type.upper()}"
            details = f"    Position: ({bbox.rect.x()}, {bbox.rect.y()})"
            details += f"\n    Size: {bbox.rect.width()} × {bbox.rect.height()}"

            item = QListWidgetItem(f"{text}\n{details}")
            item.setData(Qt.UserRole, bbox.bbox_id)

            # Set màu
            color = QColor(self.BBOX_COLORS.get(bbox.bbox_type, '#ffffff'))
            item.setForeground(color)

            self.list_widget.addItem(item)

        # Update info
        count = len(self.bboxes)
        type_counts = {}
        for bbox in self.bboxes:
            type_counts[bbox.bbox_type] = type_counts.get(bbox.bbox_type, 0) + 1

        info_parts = [f"Total: {count}"]
        for bbox_type, count in sorted(type_counts.items()):
            info_parts.append(f"{bbox_type}: {count}")

        self.info_label.setText(" | ".join(info_parts))

    def on_item_clicked(self, item):
        """Xử lý khi click vào item"""
        bbox_id = item.data(Qt.UserRole)
        # Tìm bbox tương ứng
        for bbox in self.bboxes:
            if bbox.bbox_id == bbox_id:
                self.bboxSelected.emit(bbox)
                self.delete_btn.setEnabled(True)
                return

    def on_delete_clicked(self):
        """Xử lý khi click nút delete"""
        current_item = self.list_widget.currentItem()
        if current_item:
            bbox_id = current_item.data(Qt.UserRole)
            # Tìm bbox tương ứng
            for bbox in self.bboxes:
                if bbox.bbox_id == bbox_id:
                    self.deleteRequested.emit(bbox)
                    return

    def select_bbox(self, bbox):
        """Chọn bbox trong list"""
        if bbox is None:
            self.list_widget.clearSelection()
            self.delete_btn.setEnabled(False)
            return

        # Tìm và select item tương ứng
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == bbox.bbox_id:
                self.list_widget.setCurrentItem(item)
                self.delete_btn.setEnabled(True)
                return

    def clear_selection(self):
        """Bỏ chọn tất cả"""
        self.list_widget.clearSelection()
        self.delete_btn.setEnabled(False)
