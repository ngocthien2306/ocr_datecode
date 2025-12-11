"""
Inference Widget - Template matching and OCR inference
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QFileDialog, QListWidget, QTextEdit,
                              QSplitter, QMessageBox, QListWidgetItem, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import sys

# Import the SuperPointMatcherONNX class
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    import onnxruntime as ort
    from typing import Dict, Optional
    import time
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available")


class SuperPointMatcherONNX:
    """ONNX-based SuperPoint matcher for template matching"""
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
        
        return cropped_regions


class InferenceWorker(QThread):
    """Worker thread for running inference"""
    finished = pyqtSignal(dict, object)  # result, annotated_image
    error = pyqtSignal(str)
    
    def __init__(self, matcher, image_path):
        super().__init__()
        self.matcher = matcher
        self.image_path = image_path
    
    def run(self):
        try:
            result = self.matcher.match(self.image_path)
            
            if result['success']:
                # Draw bboxes on image
                annotated = result['target_img'].copy()
                colors = {
                    'template': (0, 255, 0),
                    'text': (255, 165, 0),
                    'barcode': (255, 0, 255),
                    'datecode': (0, 255, 255)
                }
                
                for bbox in result['transformed_bboxes']:
                    pts = np.array(bbox['points'], dtype=np.int32)
                    color = colors.get(bbox['type'], (255, 255, 255))
                    cv2.polylines(annotated, [pts], True, color, 3)
                    
                    center = pts.mean(axis=0).astype(int)
                    cv2.putText(annotated, bbox['type'], tuple(center), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                self.finished.emit(result, annotated)
            else:
                self.error.emit(result.get('error', 'Unknown error'))
        except Exception as e:
            self.error.emit(str(e))


class InferenceWidget(QWidget):
    """Widget for template inference"""
    
    def __init__(self):
        super().__init__()
        self.matcher = None
        self.image_folder = None
        self.image_files = []
        self.current_index = -1
        self.current_result = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for 3 panels
        splitter = QSplitter(Qt.Horizontal)
        
        # === LEFT PANEL ===
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_panel.setMinimumWidth(220)
        left_panel.setMaximumWidth(300)
        
        # Annotations JSON
        header1 = QLabel("Template Configuration")
        header1.setObjectName("headerLabel")
        left_layout.addWidget(header1)
        
        self.json_path_label = QLabel("No JSON loaded")
        self.json_path_label.setWordWrap(True)
        self.json_path_label.setObjectName("infoLabel")
        left_layout.addWidget(self.json_path_label)
        
        load_json_btn = QPushButton("Load Annotations JSON")
        load_json_btn.setObjectName("primaryButton")
        load_json_btn.clicked.connect(self.load_annotations_json)
        left_layout.addWidget(load_json_btn)
        
        left_layout.addSpacing(16)
        
        # Image folder
        header2 = QLabel("Test Images")
        header2.setObjectName("headerLabel")
        left_layout.addWidget(header2)
        
        self.folder_path_label = QLabel("No folder loaded")
        self.folder_path_label.setWordWrap(True)
        self.folder_path_label.setObjectName("infoLabel")
        left_layout.addWidget(self.folder_path_label)
        
        load_folder_btn = QPushButton("Load Image Folder")
        load_folder_btn.setObjectName("primaryButton")
        load_folder_btn.clicked.connect(self.load_image_folder)
        left_layout.addWidget(load_folder_btn)
        
        left_layout.addSpacing(16)
        
        # Image list
        list_header = QLabel("Images")
        list_header.setObjectName("headerLabel")
        left_layout.addWidget(list_header)
        
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        left_layout.addWidget(self.image_list)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◄ Prev")
        self.prev_btn.clicked.connect(self.prev_image)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next ►")
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        left_layout.addLayout(nav_layout)
        
        # Info label
        self.info_label = QLabel("Load JSON and images to start")
        self.info_label.setWordWrap(True)
        self.info_label.setObjectName("infoLabel")
        left_layout.addWidget(self.info_label)
        
        splitter.addWidget(left_panel)
        
        # === CENTER PANEL ===
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        center_header = QLabel("Image Preview")
        center_header.setObjectName("headerLabel")
        center_layout.addWidget(center_header)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3e3e3e;")
        self.image_label.setMinimumSize(400, 300)
        center_layout.addWidget(self.image_label, stretch=1)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        center_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setObjectName("infoLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.status_label)
        
        # Run inference button
        self.run_btn = QPushButton("Run Inference")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.clicked.connect(self.run_inference)
        self.run_btn.setEnabled(False)
        center_layout.addWidget(self.run_btn)
        
        splitter.addWidget(center_panel)
        
        # === RIGHT PANEL ===
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_panel.setMinimumWidth(250)
        right_panel.setMaximumWidth(400)
        
        # Header
        right_header = QLabel("OCR Results")
        right_header.setObjectName("headerLabel")
        right_layout.addWidget(right_header)
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("OCR results will appear here...")
        right_layout.addWidget(self.results_text, stretch=1)
        
        # Export button (for future)
        export_btn = QPushButton("Export Results")
        export_btn.setEnabled(False)
        right_layout.addWidget(export_btn)
        
        splitter.addWidget(right_panel)
        
        # Set splitter sizes
        splitter.setSizes([250, 600, 300])
        
        main_layout.addWidget(splitter)
    
    def load_annotations_json(self):
        """Load annotations JSON file"""
        if not ONNX_AVAILABLE:
            QMessageBox.critical(self, "Error", "ONNX Runtime not available!\nPlease install: pip install onnxruntime-gpu")
            return
        
        json_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Annotations JSON",
            "",
            "JSON Files (*.json)"
        )
        
        if not json_path:
            return
        
        # Check for pipeline ONNX model
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(json_path)),
            'weights',
            'superpoint_lightglue_pipeline.onnx'
        )
        
        if not os.path.exists(pipeline_path):
            QMessageBox.critical(
                self,
                "Model Not Found",
                f"ONNX model not found at:\n{pipeline_path}\n\n"
                "Please ensure the model file exists."
            )
            return
        
        try:
            self.matcher = SuperPointMatcherONNX(
                json_path,
                pipeline_path,
                scale=0.5,
                verbose=True
            )
            
            self.json_path_label.setText(os.path.basename(json_path))
            self.info_label.setText(f"✅ Template loaded: {self.matcher.template_bbox['type']}")
            
            if self.image_folder:
                self.run_btn.setEnabled(True)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load matcher:\n{str(e)}")
    
    def load_image_folder(self):
        """Load folder containing test images"""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        
        if not folder:
            return
        
        self.image_folder = folder
        self.folder_path_label.setText(folder)
        
        # Load image files
        self.image_files = []
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for filename in sorted(os.listdir(folder)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_formats:
                self.image_files.append(filename)
        
        # Update list
        self.image_list.clear()
        for filename in self.image_files:
            self.image_list.addItem(filename)
        
        self.info_label.setText(f"{len(self.image_files)} images found")
        
        if self.image_files:
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            if self.matcher:
                self.run_btn.setEnabled(True)
    
    def on_image_selected(self, item):
        """Handle image selection"""
        idx = self.image_list.row(item)
        self.current_index = idx
        self.load_current_image()
    
    def load_current_image(self):
        """Load and display current image"""
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return
        
        filename = self.image_files[self.current_index]
        image_path = os.path.join(self.image_folder, filename)
        
        # Load and display
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            # Select in list
            self.image_list.setCurrentRow(self.current_index)
            
            # Update status
            self.status_label.setText(f"Image {self.current_index + 1}/{len(self.image_files)}: {filename}")
            
            # Clear previous results
            self.results_text.clear()
            self.current_result = None
    
    def prev_image(self):
        """Go to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def next_image(self):
        """Go to next image"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
    
    def run_inference(self):
        """Run inference on current image"""
        if not self.matcher or self.current_index < 0:
            return
        
        filename = self.image_files[self.current_index]
        image_path = os.path.join(self.image_folder, filename)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.run_btn.setEnabled(False)
        self.status_label.setText("Running inference...")
        
        # Run in thread
        self.worker = InferenceWorker(self.matcher, image_path)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.error.connect(self.on_inference_error)
        self.worker.start()
    
    def on_inference_finished(self, result, annotated_image):
        """Handle inference completion"""
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.current_result = result
        
        # Display annotated image
        h, w, ch = annotated_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(annotated_image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_image)
        
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        
        # Display results
        timings = result.get('timings', {})
        results_text = f"""✅ Inference Successful

Confidence: {result['confidence']:.1%}
Inliers: {result['inliers']}/{result['total_matches']}
Processing Time: {timings.get('total', 0):.0f}ms

Detected Regions:
"""
        for bbox in result['transformed_bboxes']:
            if bbox['type'] != 'template':
                results_text += f"  • {bbox['type']}\n"
        
        results_text += "\n--- OCR Results ---\n(To be implemented)\n"
        
        self.results_text.setText(results_text)
        self.status_label.setText(f"✅ Success - Confidence: {result['confidence']:.1%}")
    
    def on_inference_error(self, error_msg):
        """Handle inference error"""
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        
        self.results_text.setText(f"❌ Inference Failed\n\n{error_msg}")
        self.status_label.setText(f"❌ Error: {error_msg}")
