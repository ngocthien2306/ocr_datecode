import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTextEdit, QSlider, QGroupBox, QSpinBox, QDialog,
                             QFormLayout, QDialogButtonBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from test_segment import YOLOSegmentInference


class CircularROISelector(QDialog):
    """Dialog for selecting circular ROI on template image"""
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select ROI (Circular)")
        self.setGeometry(100, 100, 1000, 800)

        self.original_image = image.copy()
        self.circle_center = None  # (x, y)
        self.circle_radius = 50
        self.dragging = False

        # HSV range from histogram analysis
        self.h_min, self.h_max = 0, 180
        self.s_min, self.s_max = 0, 255
        self.v_min, self.v_max = 0, 255

        self.setup_ui()

        # Set initial circle at center
        h, w = image.shape[:2]
        self.circle_center = (w // 2, h // 2)
        self.analyze_roi()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Radius slider
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Circle Radius:"))
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setMinimum(10)
        self.radius_slider.setMaximum(200)
        self.radius_slider.setValue(50)
        self.radius_slider.valueChanged.connect(self.on_radius_changed)
        radius_layout.addWidget(self.radius_slider)
        self.radius_label = QLabel("50 px")
        radius_layout.addWidget(self.radius_label)
        layout.addLayout(radius_layout)

        # Instructions
        instructions = QLabel("Click and drag to move the circle. Adjust radius with slider.")
        layout.addWidget(instructions)

        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(900, 500)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release
        layout.addWidget(self.image_label)

        # Histogram display
        self.histogram_label = QLabel()
        self.histogram_label.setFixedHeight(120)
        self.histogram_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.histogram_label)

        # HSV range display (read-only, auto-updated)
        hsv_group = QGroupBox("HSV Range (auto-detected from circle)")
        hsv_layout = QVBoxLayout()

        self.hsv_info = QTextEdit()
        self.hsv_info.setMaximumHeight(80)
        self.hsv_info.setReadOnly(True)
        hsv_layout.addWidget(self.hsv_info)

        hsv_group.setLayout(hsv_layout)
        layout.addWidget(hsv_group)

        # Manual override inputs
        manual_group = QGroupBox("Manual Override (optional)")
        manual_layout = QFormLayout()

        # H range
        h_layout = QHBoxLayout()
        self.h_min_spin = QSpinBox()
        self.h_min_spin.setRange(0, 180)
        self.h_max_spin = QSpinBox()
        self.h_max_spin.setRange(0, 180)
        h_layout.addWidget(QLabel("Min:"))
        h_layout.addWidget(self.h_min_spin)
        h_layout.addWidget(QLabel("Max:"))
        h_layout.addWidget(self.h_max_spin)
        manual_layout.addRow("Hue:", h_layout)

        # S range
        s_layout = QHBoxLayout()
        self.s_min_spin = QSpinBox()
        self.s_min_spin.setRange(0, 255)
        self.s_max_spin = QSpinBox()
        self.s_max_spin.setRange(0, 255)
        s_layout.addWidget(QLabel("Min:"))
        s_layout.addWidget(self.s_min_spin)
        s_layout.addWidget(QLabel("Max:"))
        s_layout.addWidget(self.s_max_spin)
        manual_layout.addRow("Saturation:", s_layout)

        # V range
        v_layout = QHBoxLayout()
        self.v_min_spin = QSpinBox()
        self.v_min_spin.setRange(0, 255)
        self.v_max_spin = QSpinBox()
        self.v_max_spin.setRange(0, 255)
        v_layout.addWidget(QLabel("Min:"))
        v_layout.addWidget(self.v_min_spin)
        v_layout.addWidget(QLabel("Max:"))
        v_layout.addWidget(self.v_max_spin)
        manual_layout.addRow("Value:", v_layout)

        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def on_radius_changed(self, value):
        self.circle_radius = value
        self.radius_label.setText(f"{value} px")
        self.analyze_roi()

    def mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.get_image_coords(event.pos())
            if pos is None:
                return

            # Check if clicking inside circle
            if self.circle_center is not None:
                dx = pos[0] - self.circle_center[0]
                dy = pos[1] - self.circle_center[1]
                dist = np.sqrt(dx*dx + dy*dy)

                if dist <= self.circle_radius:
                    self.dragging = True
            else:
                # Set circle center
                self.circle_center = pos
                self.dragging = True
                self.analyze_roi()

    def mouse_move(self, event):
        if self.dragging:
            pos = self.get_image_coords(event.pos())
            if pos is None:
                return

            self.circle_center = pos
            self.analyze_roi()

    def mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def get_image_coords(self, widget_pos):
        """Convert widget coordinates to image coordinates"""
        if self.original_image is None:
            return None

        # Get label size
        label_rect = self.image_label.rect()

        # Calculate scaling
        img_h, img_w = self.original_image.shape[:2]
        scale = min(label_rect.width() / img_w, label_rect.height() / img_h)

        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)

        # Calculate offset (centering)
        offset_x = (label_rect.width() - scaled_w) // 2
        offset_y = (label_rect.height() - scaled_h) // 2

        # Convert to image coords
        img_x = int((widget_pos.x() - offset_x) / scale)
        img_y = int((widget_pos.y() - offset_y) / scale)

        # Clamp to image bounds
        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))

        return (img_x, img_y)

    def analyze_roi(self):
        """Analyze circular ROI to determine HSV range"""
        if self.circle_center is None:
            return

        # Create circular mask
        h, w = self.original_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, self.circle_center, self.circle_radius, 255, -1)

        # Extract ROI pixels
        roi_pixels = self.original_image[mask > 0]

        if len(roi_pixels) == 0:
            return

        # Convert to HSV - keep as 3D for calcHist
        roi_hsv = cv2.cvtColor(roi_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)

        # Calculate histograms
        h_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([roi_hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([roi_hsv], [2], None, [256], [0, 256])

        # Find range with 5% threshold
        threshold_pct = 0.05

        # H range
        h_threshold = h_hist.max() * threshold_pct
        h_indices = np.where(h_hist > h_threshold)[0]
        if len(h_indices) > 0:
            self.h_min = int(h_indices.min())
            self.h_max = int(h_indices.max())

        # S range
        s_threshold = s_hist.max() * threshold_pct
        s_indices = np.where(s_hist > s_threshold)[0]
        if len(s_indices) > 0:
            self.s_min = int(s_indices.min())
            self.s_max = int(s_indices.max())

        # V range
        v_threshold = v_hist.max() * threshold_pct
        v_indices = np.where(v_hist > v_threshold)[0]
        if len(v_indices) > 0:
            self.v_min = int(v_indices.min())
            self.v_max = int(v_indices.max())

        # Update spinboxes
        self.h_min_spin.setValue(self.h_min)
        self.h_max_spin.setValue(self.h_max)
        self.s_min_spin.setValue(self.s_min)
        self.s_max_spin.setValue(self.s_max)
        self.v_min_spin.setValue(self.v_min)
        self.v_max_spin.setValue(self.v_max)

        # Update info text
        self.hsv_info.setText(
            f"H: [{self.h_min}, {self.h_max}]  (range: {self.h_max - self.h_min})\n"
            f"S: [{self.s_min}, {self.s_max}]  (range: {self.s_max - self.s_min})\n"
            f"V: [{self.v_min}, {self.v_max}]  (range: {self.v_max - self.v_min})\n"
            f"Pixels in circle: {len(roi_pixels)}"
        )

        # Draw histogram
        self.draw_histogram(h_hist, s_hist, v_hist)

        # Update display
        self.update_display()

    def draw_histogram(self, h_hist, s_hist, v_hist):
        """Draw combined HSV histogram"""
        hist_h = 100
        hist_w = 900

        hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

        # Normalize histograms
        h_hist_norm = cv2.normalize(h_hist, None, 0, hist_h - 10, cv2.NORM_MINMAX)
        s_hist_norm = cv2.normalize(s_hist, None, 0, hist_h - 10, cv2.NORM_MINMAX)
        v_hist_norm = cv2.normalize(v_hist, None, 0, hist_h - 10, cv2.NORM_MINMAX)

        # Draw H (red) - 180 bins
        bin_w = hist_w / 180
        for i in range(180):
            x = int(i * bin_w)
            cv2.line(hist_img,
                    (x, hist_h),
                    (x, hist_h - int(h_hist_norm[i])),
                    (0, 0, 255), max(1, int(bin_w)))

        # Draw S (green) - 256 bins scaled to fit
        bin_w = hist_w / 256
        for i in range(256):
            x = int(i * bin_w)
            cv2.line(hist_img,
                    (x, hist_h),
                    (x, hist_h - int(s_hist_norm[i])),
                    (0, 255, 0), 1)

        # Draw V (blue) - 256 bins scaled to fit
        for i in range(256):
            x = int(i * bin_w)
            cv2.line(hist_img,
                    (x, hist_h),
                    (x, hist_h - int(v_hist_norm[i])),
                    (255, 0, 0), 1)

        # Add labels
        cv2.putText(hist_img, "H", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(hist_img, "S", (40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(hist_img, "V", (70, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Convert to QPixmap
        hist_img_rgb = cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)
        h, w, ch = hist_img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(hist_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.histogram_label.setPixmap(pixmap)

    def update_display(self):
        """Update image display with circle overlay"""
        display = self.original_image.copy()

        # Draw circle
        if self.circle_center is not None:
            cv2.circle(display, self.circle_center, self.circle_radius, (0, 255, 0), 2)
            cv2.circle(display, self.circle_center, 5, (0, 0, 255), -1)  # Center dot

        # Convert to QPixmap and display
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = display_rgb.shape

        # Scale to fit label
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        scale = min(label_w / w, label_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        display_scaled = cv2.resize(display_rgb, (new_w, new_h))

        bytes_per_line = ch * new_w
        q_img = QImage(display_scaled.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def get_result(self):
        """Return circle center, radius and HSV range"""
        return {
            'center': self.circle_center,
            'radius': self.circle_radius,
            'h_min': self.h_min_spin.value(),
            'h_max': self.h_max_spin.value(),
            's_min': self.s_min_spin.value(),
            's_max': self.s_max_spin.value(),
            'v_min': self.v_min_spin.value(),
            'v_max': self.v_max_spin.value()
        }


class DualCameraInspectorV3(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual Camera Inspector V3 (Circular ROI)")
        self.setGeometry(100, 100, 1600, 900)

        # Model
        self.model = YOLOSegmentInference(
            "weights/yolo26_label_seg.onnx",
            class_names=['bottle', 'label']
        )

        # Data
        self.folder1_paths = []
        self.folder2_paths = []
        self.current_idx = 0
        self.conf_threshold = 0.25
        self.pixel_threshold = 1000

        # Template data
        self.template1_data = None
        self.template2_data = None

        # Visualization
        self.show_masks = True
        self.show_filtered_regions = False

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # ===== LEFT PANEL =====
        left_panel = QGroupBox("Template Setup")
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout()

        # Conf threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Conf Threshold:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        conf_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel("0.25")
        conf_layout.addWidget(self.conf_label)
        left_layout.addLayout(conf_layout)

        # Pixel threshold
        pixel_layout = QHBoxLayout()
        pixel_layout.addWidget(QLabel("Pixel Threshold:"))
        self.pixel_threshold_spin = QSpinBox()
        self.pixel_threshold_spin.setRange(1, 100000)
        self.pixel_threshold_spin.setValue(1000)
        self.pixel_threshold_spin.setSingleStep(100)
        self.pixel_threshold_spin.valueChanged.connect(self.on_pixel_threshold_changed)
        pixel_layout.addWidget(self.pixel_threshold_spin)
        left_layout.addLayout(pixel_layout)

        # View 1
        view1_group = QGroupBox("View 1 (Camera 1)")
        view1_layout = QVBoxLayout()

        self.btn_select_template1 = QPushButton("Select Template 1")
        self.btn_select_template1.clicked.connect(lambda: self.select_template_roi(1))
        view1_layout.addWidget(self.btn_select_template1)

        self.template1_info = QTextEdit()
        self.template1_info.setMaximumHeight(120)
        self.template1_info.setReadOnly(True)
        self.template1_info.setText("No template set")
        view1_layout.addWidget(self.template1_info)

        view1_group.setLayout(view1_layout)
        left_layout.addWidget(view1_group)

        # View 2
        view2_group = QGroupBox("View 2 (Camera 2)")
        view2_layout = QVBoxLayout()

        self.btn_select_template2 = QPushButton("Select Template 2")
        self.btn_select_template2.clicked.connect(lambda: self.select_template_roi(2))
        view2_layout.addWidget(self.btn_select_template2)

        self.template2_info = QTextEdit()
        self.template2_info.setMaximumHeight(120)
        self.template2_info.setReadOnly(True)
        self.template2_info.setText("No template set")
        view2_layout.addWidget(self.template2_info)

        view2_group.setLayout(view2_layout)
        left_layout.addWidget(view2_group)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)

        # ===== RIGHT PANEL =====
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Top buttons
        top_layout = QHBoxLayout()
        self.btn_load_folder1 = QPushButton("Load Folder 1")
        self.btn_load_folder1.clicked.connect(lambda: self.load_folder(1))
        top_layout.addWidget(self.btn_load_folder1)

        self.btn_load_folder2 = QPushButton("Load Folder 2")
        self.btn_load_folder2.clicked.connect(lambda: self.load_folder(2))
        top_layout.addWidget(self.btn_load_folder2)

        self.btn_start_inference = QPushButton("Start Inference")
        self.btn_start_inference.clicked.connect(self.start_inference)
        self.btn_start_inference.setEnabled(False)
        top_layout.addWidget(self.btn_start_inference)

        self.label_status = QLabel("No folders loaded")
        top_layout.addWidget(self.label_status)
        top_layout.addStretch()

        right_layout.addLayout(top_layout)

        # Image display
        image_grid = QHBoxLayout()

        view1_display = QVBoxLayout()
        view1_display.addWidget(QLabel("View 1 (Camera 1)"))
        self.image_view1 = QLabel()
        self.image_view1.setAlignment(Qt.AlignCenter)
        self.image_view1.setMinimumSize(550, 400)
        self.image_view1.setStyleSheet("border: 1px solid black;")
        view1_display.addWidget(self.image_view1)
        image_grid.addLayout(view1_display)

        view2_display = QVBoxLayout()
        view2_display.addWidget(QLabel("View 2 (Camera 2)"))
        self.image_view2 = QLabel()
        self.image_view2.setAlignment(Qt.AlignCenter)
        self.image_view2.setMinimumSize(550, 400)
        self.image_view2.setStyleSheet("border: 1px solid black;")
        view2_display.addWidget(self.image_view2)
        image_grid.addLayout(view2_display)

        right_layout.addLayout(image_grid)

        # Results
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setReadOnly(True)
        right_layout.addWidget(self.results_text)

        # View options
        view_options = QHBoxLayout()

        self.btn_toggle_masks = QPushButton("Hide Masks")
        self.btn_toggle_masks.clicked.connect(self.toggle_masks)
        self.btn_toggle_masks.setEnabled(False)
        view_options.addWidget(self.btn_toggle_masks)

        self.btn_toggle_filtered = QPushButton("Show Filtered Regions")
        self.btn_toggle_filtered.clicked.connect(self.toggle_filtered_regions)
        self.btn_toggle_filtered.setEnabled(False)
        view_options.addWidget(self.btn_toggle_filtered)

        view_options.addStretch()
        right_layout.addLayout(view_options)

        # Navigation
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setEnabled(False)

        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setEnabled(False)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        right_layout.addLayout(nav_layout)

        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel)

        central_widget.setLayout(main_layout)

    def on_conf_changed(self, value):
        self.conf_threshold = value / 100.0
        self.conf_label.setText(f"{self.conf_threshold:.2f}")
        if self.folder1_paths and self.folder2_paths:
            self.show_current_pair()

    def on_pixel_threshold_changed(self, value):
        self.pixel_threshold = value
        if self.folder1_paths and self.folder2_paths:
            self.show_current_pair()

    def toggle_masks(self):
        self.show_masks = not self.show_masks
        self.btn_toggle_masks.setText("Show Masks" if not self.show_masks else "Hide Masks")
        if self.folder1_paths and self.folder2_paths:
            self.show_current_pair()

    def toggle_filtered_regions(self):
        self.show_filtered_regions = not self.show_filtered_regions
        self.btn_toggle_filtered.setText("Hide Filtered Regions" if self.show_filtered_regions else "Show Filtered Regions")
        if self.folder1_paths and self.folder2_paths:
            self.show_current_pair()

    def select_template_roi(self, view_num):
        """Open circular ROI selector"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Template for View {view_num}",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if not file_path:
            return

        image = cv2.imread(file_path)

        # Open circular ROI selector
        dialog = CircularROISelector(image, self)
        if dialog.exec_() == QDialog.Accepted:
            result = dialog.get_result()

            if view_num == 1:
                self.template1_data = result
                self.template1_info.setText(
                    f"Circle: center=({result['center'][0]}, {result['center'][1]}), radius={result['radius']}\n"
                    f"HSV Range:\n"
                    f"  H: [{result['h_min']}, {result['h_max']}]\n"
                    f"  S: [{result['s_min']}, {result['s_max']}]\n"
                    f"  V: [{result['v_min']}, {result['v_max']}]"
                )
            else:
                self.template2_data = result
                self.template2_info.setText(
                    f"Circle: center=({result['center'][0]}, {result['center'][1]}), radius={result['radius']}\n"
                    f"HSV Range:\n"
                    f"  H: [{result['h_min']}, {result['h_max']}]\n"
                    f"  S: [{result['s_min']}, {result['s_max']}]\n"
                    f"  V: [{result['v_min']}, {result['v_max']}]"
                )

            # Enable inference if both templates set
            if self.template1_data and self.template2_data and self.folder1_paths and self.folder2_paths:
                self.btn_start_inference.setEnabled(True)

    def load_folder(self, folder_num):
        """Load folder"""
        folder = QFileDialog.getExistingDirectory(self, f"Select Folder {folder_num}")
        if not folder:
            return

        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(list(Path(folder).glob(ext)))
            image_paths.extend(list(Path(folder).glob(ext.upper())))

        image_paths = sorted(image_paths)

        if not image_paths:
            self.results_text.setText(f"No images found in folder {folder_num}!")
            return

        if folder_num == 1:
            self.folder1_paths = image_paths
        else:
            self.folder2_paths = image_paths

        if self.folder1_paths and self.folder2_paths:
            self.label_status.setText(f"Loaded: {len(self.folder1_paths)} pairs")
            if self.template1_data and self.template2_data:
                self.btn_start_inference.setEnabled(True)

    def start_inference(self):
        """Start inference"""
        if not (self.folder1_paths and self.folder2_paths and self.template1_data and self.template2_data):
            return

        self.current_idx = 0
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.btn_toggle_masks.setEnabled(True)
        self.btn_toggle_filtered.setEnabled(True)
        self.show_current_pair()

    def show_current_pair(self):
        """Display current pair"""
        if not (self.folder1_paths and self.folder2_paths):
            return

        idx = min(self.current_idx, len(self.folder1_paths) - 1, len(self.folder2_paths) - 1)

        # Load images
        img1 = cv2.imread(str(self.folder1_paths[idx]))
        img2 = cv2.imread(str(self.folder2_paths[idx]))

        # Predict
        boxes1, masks1 = self.model.predict(img1, conf_threshold=self.conf_threshold)
        boxes2, masks2 = self.model.predict(img2, conf_threshold=self.conf_threshold)

        # Count matching pixels
        match_count1 = self.count_matching_pixels(img1, boxes1, masks1, self.template1_data)
        match_count2 = self.count_matching_pixels(img2, boxes2, masks2, self.template2_data)

        total_match = match_count1 + match_count2

        # Visualize
        if self.show_filtered_regions:
            img1_vis = self.draw_filtered_regions(img1, boxes1, masks1, self.template1_data)
            img2_vis = self.draw_filtered_regions(img2, boxes2, masks2, self.template2_data)
        else:
            img1_vis = self.draw_labels(img1, boxes1, masks1)
            img2_vis = self.draw_labels(img2, boxes2, masks2)

        self.display_image_in_label(img1_vis, self.image_view1)
        self.display_image_in_label(img2_vis, self.image_view2)

        # Pass/Fail
        view1_pass = match_count1 >= self.pixel_threshold
        view2_pass = match_count2 >= self.pixel_threshold
        final_result = view1_pass or view2_pass

        # Display results
        result_text = f"Image Pair: {idx + 1}/{len(self.folder1_paths)}\n"
        result_text += f"Files:\n"
        result_text += f"  - View 1: {Path(self.folder1_paths[idx]).name}\n"
        result_text += f"  - View 2: {Path(self.folder2_paths[idx]).name}\n\n"

        result_text += f"Matching Pixels:\n"
        result_text += f"  - View 1: {match_count1} px\n"
        result_text += f"  - View 2: {match_count2} px\n"
        result_text += f"  - Total: {total_match} px\n\n"

        result_text += f"Template Matching (Threshold: {self.pixel_threshold} px):\n"
        result_text += f"  View 1: {match_count1} px → {'✓ PASS' if view1_pass else '✗ FAIL'}\n"
        result_text += f"  View 2: {match_count2} px → {'✓ PASS' if view2_pass else '✗ FAIL'}\n"

        result_text += f"\nFinal Result: {'✓ OK' if final_result else '✗ NG'}"

        self.results_text.setText(result_text)
        self.label_status.setText(f"Image {idx + 1}/{len(self.folder1_paths)} | Result: {'OK ✓' if final_result else 'NG ✗'}")

    def count_matching_pixels(self, image, boxes, masks, template_data):
        """Count pixels within HSV range"""
        if boxes is None or len(boxes) == 0 or masks is None or template_data is None:
            return 0

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h_min, h_max = template_data['h_min'], template_data['h_max']
        s_min, s_max = template_data['s_min'], template_data['s_max']
        v_min, v_max = template_data['v_min'], template_data['v_max']

        total_count = 0

        for i, box in enumerate(boxes):
            class_id = int(box[5])
            class_name = self.model.class_names[class_id]

            if class_name == 'label' and i < len(masks):
                mask = masks[i]
                mask_binary = mask > 0

                mask_hsv = image_hsv[mask_binary]

                if len(mask_hsv) == 0:
                    continue

                h_match = (mask_hsv[:, 0] >= h_min) & (mask_hsv[:, 0] <= h_max)
                s_match = (mask_hsv[:, 1] >= s_min) & (mask_hsv[:, 1] <= s_max)
                v_match = (mask_hsv[:, 2] >= v_min) & (mask_hsv[:, 2] <= v_max)

                matching = h_match & s_match & v_match
                total_count += matching.sum()

        return int(total_count)

    def draw_labels(self, image, boxes, masks):
        """Draw masks and boxes"""
        if boxes is None or len(boxes) == 0:
            return image

        img = image.copy()

        if self.show_masks and masks is not None and len(masks) > 0:
            mask_overlay = np.zeros_like(img)

            for i, (box, mask) in enumerate(zip(boxes, masks)):
                class_id = int(box[5])
                class_name = self.model.class_names[class_id]

                if class_name == 'label':
                    color = (0, 255, 0)
                    mask_binary = mask.astype(np.uint8)
                    mask_overlay[mask_binary > 0] = color

            img = cv2.addWeighted(img, 0.6, mask_overlay, 0.4, 0)

        for box in boxes:
            x, y, w_box, h_box, conf, class_id = box
            class_id = int(class_id)
            class_name = self.model.class_names[class_id]

            if class_name == 'label':
                x1, y1 = int(x - w_box/2), int(y - h_box/2)
                x2, y2 = int(x + w_box/2), int(y + h_box/2)

                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label = f'{class_name}: {conf:.2f}'
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

    def draw_filtered_regions(self, image, boxes, masks, template_data):
        """Draw only pixels matching HSV range"""
        if boxes is None or len(boxes) == 0 or masks is None or template_data is None:
            return image

        img = image.copy()
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h_min, h_max = template_data['h_min'], template_data['h_max']
        s_min, s_max = template_data['s_min'], template_data['s_max']
        v_min, v_max = template_data['v_min'], template_data['v_max']

        overlay = np.zeros_like(img)

        for i, box in enumerate(boxes):
            class_id = int(box[5])
            class_name = self.model.class_names[class_id]

            if class_name == 'label' and i < len(masks):
                mask = masks[i]
                mask_binary = mask > 0

                mask_indices = np.where(mask_binary)

                if len(mask_indices[0]) == 0:
                    continue

                for y, x in zip(mask_indices[0], mask_indices[1]):
                    h, s, v = image_hsv[y, x]

                    if (h_min <= h <= h_max and
                        s_min <= s <= s_max and
                        v_min <= v <= v_max):
                        overlay[y, x] = (0, 255, 255)  # Yellow

        img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

        return img

    def display_image_in_label(self, img, label_widget):
        """Display image"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        label_w = label_widget.width()
        label_h = label_widget.height()

        scale = min(label_w / w, label_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_scaled = cv2.resize(img_rgb, (new_w, new_h))

        bytes_per_line = ch * new_w
        q_img = QImage(img_scaled.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        label_widget.setPixmap(pixmap)

    def next_image(self):
        max_idx = min(len(self.folder1_paths), len(self.folder2_paths)) - 1
        if self.current_idx < max_idx:
            self.current_idx += 1
            self.show_current_pair()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_pair()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DualCameraInspectorV3()
    viewer.show()
    sys.exit(app.exec_())
