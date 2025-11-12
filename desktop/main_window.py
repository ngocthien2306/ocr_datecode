"""
Main Window
Cửa sổ chính của ứng dụng với UI được cải tiến
"""
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                              QListWidget, QPushButton, QFileDialog, QLabel,
                              QMessageBox, QListWidgetItem, QSplitter, QComboBox)
from PyQt5.QtCore import Qt, QDir
from PyQt5.QtGui import QFont
from image_viewer import ImageViewer
from annotation_manager import AnnotationManager
from bbox_list_widget import BBoxListWidget
from styles import APP_STYLESHEET


class MainWindow(QMainWindow):
    """Cửa sổ chính của ứng dụng"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Annotation Tool")
        self.setGeometry(100, 100, 1400, 900)

        # Apply stylesheet
        self.setStyleSheet(APP_STYLESHEET)

        # Khởi tạo annotation manager
        self.annotation_manager = AnnotationManager()

        # State
        self.images_folder = None
        self.image_files = []
        self.current_image_path = None

        # Setup UI
        self.setup_ui()

        # Load images folder mặc định nếu có
        default_images_folder = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'images'
        )
        if os.path.exists(default_images_folder):
            self.load_images_folder(default_images_folder)

    def setup_ui(self):
        """Thiết lập giao diện"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === LEFT PANEL ===
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_panel.setMinimumWidth(220)
        left_panel.setMaximumWidth(300)

        # Header
        header_label = QLabel("Image Folder")
        header_label.setObjectName("headerLabel")
        left_layout.addWidget(header_label)

        # Folder path
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setWordWrap(True)
        self.folder_path_label.setObjectName("infoLabel")
        left_layout.addWidget(self.folder_path_label)

        # Select folder button
        select_folder_btn = QPushButton("Select Folder")
        select_folder_btn.setObjectName("primaryButton")
        select_folder_btn.clicked.connect(self.select_folder)
        left_layout.addWidget(select_folder_btn)

        left_layout.addSpacing(16)

        # Image list header
        list_header = QLabel("Images")
        list_header.setObjectName("headerLabel")
        left_layout.addWidget(list_header)

        # Image list
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        left_layout.addWidget(self.image_list)

        # Image info
        self.info_label = QLabel("No image selected")
        self.info_label.setWordWrap(True)
        self.info_label.setObjectName("infoLabel")
        left_layout.addWidget(self.info_label)

        main_layout.addWidget(left_panel)

        # === CENTER + RIGHT PANELS (Splitter) ===
        center_right_splitter = QSplitter(Qt.Horizontal)

        # === CENTER PANEL (Image Viewer) ===
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(12, 12, 12, 12)

        # Toolbar
        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 8, 12, 8)

        # Mode buttons
        self.select_mode_btn = QPushButton("Select")
        self.select_mode_btn.setCheckable(True)
        self.select_mode_btn.setChecked(True)
        self.select_mode_btn.clicked.connect(self.set_select_mode)
        self.select_mode_btn.setEnabled(False)
        self.select_mode_btn.setMinimumWidth(60)
        toolbar_layout.addWidget(self.select_mode_btn)

        self.draw_mode_btn = QPushButton("Draw")
        self.draw_mode_btn.setCheckable(True)
        self.draw_mode_btn.clicked.connect(self.set_draw_mode)
        self.draw_mode_btn.setEnabled(False)
        self.draw_mode_btn.setMinimumWidth(60)
        toolbar_layout.addWidget(self.draw_mode_btn)

        toolbar_layout.addSpacing(8)

        # Shape selection (Rectangle/Polygon)
        shape_label = QLabel("Shape:")
        toolbar_layout.addWidget(shape_label)

        self.shape_combo = QComboBox()
        self.shape_combo.addItem("Rectangle")
        self.shape_combo.addItem("Polygon (4 pts)")
        self.shape_combo.setEnabled(False)
        self.shape_combo.currentIndexChanged.connect(self.on_shape_changed)
        self.shape_combo.setMinimumWidth(120)
        toolbar_layout.addWidget(self.shape_combo)

        toolbar_layout.addSpacing(8)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("dangerButton")
        self.clear_btn.clicked.connect(self.clear_bboxes)
        self.clear_btn.setEnabled(False)
        self.clear_btn.setMinimumWidth(60)
        toolbar_layout.addWidget(self.clear_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("primaryButton")
        self.save_btn.clicked.connect(self.save_annotations)
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumWidth(60)
        toolbar_layout.addWidget(self.save_btn)

        toolbar_layout.addSpacing(8)

        # Set as template button
        self.set_template_btn = QPushButton("Template")
        self.set_template_btn.setCheckable(True)
        self.set_template_btn.clicked.connect(self.toggle_template)
        self.set_template_btn.setEnabled(False)
        self.set_template_btn.setFixedWidth(75)
        self.set_template_btn.setToolTip("Mark this image as template for matching")
        toolbar_layout.addWidget(self.set_template_btn)

        toolbar_layout.addStretch()

        self.bbox_count_label = QLabel("BBoxes: 0")
        self.bbox_count_label.setObjectName("countLabel")
        toolbar_layout.addWidget(self.bbox_count_label)

        center_layout.addWidget(toolbar)

        # Image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.bboxCreated.connect(self.on_bbox_created)
        self.image_viewer.bboxSelected.connect(self.on_bbox_selected)
        self.image_viewer.bboxesChanged.connect(self.on_bboxes_changed)
        center_layout.addWidget(self.image_viewer, stretch=1)

        # Zoom controls bar (bottom)
        zoom_bar = QWidget()
        zoom_bar.setObjectName("toolbar")
        zoom_bar_layout = QHBoxLayout(zoom_bar)
        zoom_bar_layout.setContentsMargins(12, 6, 12, 6)

        zoom_bar_layout.addStretch()

        zoom_bar_layout.addWidget(QLabel("Zoom:"))

        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedWidth(30)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_out_btn.setEnabled(False)
        self.zoom_out_btn.setToolTip("Zoom Out (Ctrl + -)")
        zoom_bar_layout.addWidget(self.zoom_out_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(45)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        zoom_bar_layout.addWidget(self.zoom_label)

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(30)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_in_btn.setEnabled(False)
        self.zoom_in_btn.setToolTip("Zoom In (Ctrl + +)")
        zoom_bar_layout.addWidget(self.zoom_in_btn)

        self.zoom_reset_btn = QPushButton("1:1")
        self.zoom_reset_btn.setFixedWidth(40)
        self.zoom_reset_btn.clicked.connect(self.reset_zoom)
        self.zoom_reset_btn.setEnabled(False)
        self.zoom_reset_btn.setToolTip("Reset Zoom (Ctrl + 0)")
        zoom_bar_layout.addWidget(self.zoom_reset_btn)

        zoom_bar_layout.addStretch()

        center_layout.addWidget(zoom_bar)

        center_right_splitter.addWidget(center_panel)

        # === RIGHT PANEL (BBox List) ===
        self.bbox_list_widget = BBoxListWidget()
        self.bbox_list_widget.setMinimumWidth(200)
        self.bbox_list_widget.setMaximumWidth(350)
        self.bbox_list_widget.bboxSelected.connect(self.on_bbox_list_selected)
        self.bbox_list_widget.deleteRequested.connect(self.on_bbox_delete_requested)

        center_right_splitter.addWidget(self.bbox_list_widget)

        # Set initial splitter sizes (cho phép resize linh hoạt)
        center_right_splitter.setSizes([900, 250])
        center_right_splitter.setStretchFactor(0, 3)  # Image viewer có stretch factor lớn hơn
        center_right_splitter.setStretchFactor(1, 1)  # BBox list

        main_layout.addWidget(center_right_splitter, stretch=1)

    def select_folder(self):
        """Chọn folder chứa ảnh"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Images Folder",
            QDir.homePath()
        )

        if folder:
            self.load_images_folder(folder)

    def load_images_folder(self, folder):
        """Load danh sách ảnh từ folder"""
        self.images_folder = folder
        self.folder_path_label.setText(folder)

        # Load danh sách file ảnh
        self.image_files = []
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

        for filename in sorted(os.listdir(folder)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_formats:
                self.image_files.append(filename)

        # Cập nhật list widget
        self.image_list.clear()
        for filename in self.image_files:
            item = QListWidgetItem(filename)
            # Đánh dấu nếu đã có annotations
            image_path = os.path.join(self.images_folder, filename)
            if self.annotation_manager.has_annotations(image_path):
                item.setText(f"✓ {filename}")
                item.setForeground(Qt.green)
            self.image_list.addItem(item)

        # Load annotations cho folder này
        self.annotation_manager.load_annotations(folder)

        # Update info
        self.info_label.setText(f"{len(self.image_files)} images found")

    def on_image_selected(self, item):
        """Xử lý khi chọn ảnh từ list"""
        # Lưu annotations của ảnh hiện tại nếu có
        if self.current_image_path:
            self.save_current_annotations()

        # Load ảnh mới
        filename = item.text().replace("✓ ", "")  # Remove checkmark
        self.current_image_path = os.path.join(self.images_folder, filename)

        if self.image_viewer.set_image(self.current_image_path):
            # Load annotations nếu có
            bboxes = self.annotation_manager.get_annotations(self.current_image_path)
            self.image_viewer.set_bboxes(bboxes)

            # Enable buttons
            self.select_mode_btn.setEnabled(True)
            self.draw_mode_btn.setEnabled(True)
            self.shape_combo.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.zoom_in_btn.setEnabled(True)
            self.zoom_out_btn.setEnabled(True)
            self.zoom_reset_btn.setEnabled(True)
            self.set_template_btn.setEnabled(True)

            # Update displays
            self.update_bbox_count()
            self.bbox_list_widget.update_bboxes(bboxes)

            # Update info
            width = self.image_viewer.original_pixmap.width()
            height = self.image_viewer.original_pixmap.height()
            self.info_label.setText(
                f"{filename}\n"
                f"{width} × {height} px\n"
                f"{len(bboxes)} boxes"
            )

            # Update template status
            is_template = self.annotation_manager.is_template_image(self.current_image_path)
            self.set_template_btn.setChecked(is_template)
            self.update_zoom_label()

    def set_select_mode(self):
        """Chuyển sang chế độ select"""
        if not self.select_mode_btn.isChecked():
            self.select_mode_btn.setChecked(True)
            return

        self.image_viewer.set_mode(ImageViewer.MODE_SELECT)
        self.select_mode_btn.setChecked(True)
        self.draw_mode_btn.setChecked(False)

    def set_draw_mode(self):
        """Chuyển sang chế độ draw"""
        if not self.draw_mode_btn.isChecked():
            self.draw_mode_btn.setChecked(True)
            return

        self.image_viewer.set_mode(ImageViewer.MODE_DRAW)
        self.draw_mode_btn.setChecked(True)
        self.select_mode_btn.setChecked(False)

    def on_bbox_created(self):
        """Xử lý khi có bbox mới được tạo"""
        self.update_bbox_count()
        self.bbox_list_widget.update_bboxes(self.image_viewer.get_bboxes())
        # Auto save
        self.save_current_annotations()

    def on_bbox_selected(self, bbox):
        """Xử lý khi bbox được chọn trong image viewer"""
        self.bbox_list_widget.select_bbox(bbox)

    def on_bbox_list_selected(self, bbox):
        """Xử lý khi bbox được chọn từ list"""
        self.image_viewer.select_bbox_by_id(bbox.bbox_id)

    def on_bbox_delete_requested(self, bbox):
        """Xử lý khi yêu cầu xóa bbox từ list"""
        # Chọn bbox trước khi xóa
        self.image_viewer.select_bbox_by_id(bbox.bbox_id)
        # Xóa
        self.image_viewer.delete_selected_bbox()
        # Cập nhật list
        self.bbox_list_widget.update_bboxes(self.image_viewer.get_bboxes())
        self.update_bbox_count()
        self.save_current_annotations()

    def on_bboxes_changed(self):
        """Xử lý khi có thay đổi về bboxes (move, resize, delete)"""
        self.update_bbox_count()
        self.bbox_list_widget.update_bboxes(self.image_viewer.get_bboxes())
        # Auto save
        self.save_current_annotations()

    def on_shape_changed(self, index):
        """Xử lý khi đổi shape mode (Rectangle/Polygon)"""
        if index == 0:  # Rectangle
            self.image_viewer.set_draw_shape('rectangle')
        else:  # Polygon (4 pts)
            self.image_viewer.set_draw_shape('polygon')

    def update_bbox_count(self):
        """Cập nhật số lượng bbox"""
        count = len(self.image_viewer.get_bboxes())
        self.bbox_count_label.setText(f"BBoxes: {count}")

    def clear_bboxes(self):
        """Xóa tất cả bboxes"""
        if len(self.image_viewer.get_bboxes()) == 0:
            return

        reply = QMessageBox.question(
            self,
            'Clear All Bounding Boxes',
            'Are you sure you want to delete all bounding boxes?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.image_viewer.clear_bboxes()
            self.bbox_list_widget.update_bboxes([])
            self.update_bbox_count()
            self.save_current_annotations()

    def save_current_annotations(self):
        """Lưu annotations của ảnh hiện tại"""
        if self.current_image_path:
            bboxes = self.image_viewer.get_bboxes()
            self.annotation_manager.set_annotations(self.current_image_path, bboxes)
            self.annotation_manager.save_annotations()
            self.update_image_list_item()

    def save_annotations(self):
        """Lưu tất cả annotations ra file"""
        self.save_current_annotations()
        QMessageBox.information(
            self,
            "Saved",
            f"Annotations saved successfully!\n\n"
            f"File: {self.annotation_manager.get_annotations_file()}"
        )

    def update_image_list_item(self):
        """Cập nhật trạng thái của item trong list"""
        if not self.current_image_path:
            return

        filename = os.path.basename(self.current_image_path)
        has_annotations = self.annotation_manager.has_annotations(self.current_image_path)

        # Tìm và cập nhật item
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            item_filename = item.text().replace("✓ ", "")
            if item_filename == filename:
                if has_annotations:
                    item.setText(f"✓ {filename}")
                    item.setForeground(Qt.green)
                else:
                    item.setText(filename)
                    item.setForeground(Qt.white)
                break

    def zoom_in(self):
        """Zoom in ảnh"""
        if self.image_viewer.zoom_in():
            self.update_zoom_label()

    def zoom_out(self):
        """Zoom out ảnh"""
        if self.image_viewer.zoom_out():
            self.update_zoom_label()

    def reset_zoom(self):
        """Reset zoom"""
        self.image_viewer.reset_zoom()
        self.update_zoom_label()

    def update_zoom_label(self):
        """Cập nhật label hiển thị zoom"""
        zoom_percent = self.image_viewer.get_zoom_percent()
        self.zoom_label.setText(f"{zoom_percent}%")

    def toggle_template(self):
        """Toggle template status cho ảnh hiện tại"""
        if not self.current_image_path:
            return

        if self.set_template_btn.isChecked():
            # Set as template
            self.annotation_manager.set_template_image(self.current_image_path)
            QMessageBox.information(
                self,
                "Template Set",
                f"Image marked as template:\n{os.path.basename(self.current_image_path)}"
            )
        else:
            # Unset template
            self.annotation_manager.unset_template_image()
            QMessageBox.information(
                self,
                "Template Unset",
                "Template image cleared"
            )

        self.annotation_manager.save_annotations()

    def closeEvent(self, event):
        """Xử lý khi đóng ứng dụng"""
        # Auto save trước khi thoát
        if self.current_image_path:
            self.save_current_annotations()

        event.accept()
