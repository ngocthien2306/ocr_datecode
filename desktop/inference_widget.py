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
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
import sys
import multiprocessing as mp
from multiprocessing import Process, Queue
import pickle

# Import the SuperPointMatcherONNX class
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    import onnxruntime as ort
    from typing import Dict, Optional, List, Tuple
    import time
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available")

# Import TextRecognizer
try:
    from text_recognizer import TextRecognizer
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: text_recognizer not available")

# Import pyzbar for barcode decoding
try:
    from pyzbar import pyzbar
    BARCODE_AVAILABLE = True
except ImportError:
    BARCODE_AVAILABLE = False
    print("Warning: pyzbar not available. Install with: pip install pyzbar")

# Import image preprocessor for better OCR
try:
    from image_preprocessor import ImagePreprocessor
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    print("Warning: image_preprocessor not available")


def run_inference_process(json_path, pipeline_path, image_path, result_queue):
    """
    Function to run in separate process - completely isolated from Qt
    """
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
        import time
        import json
        
        # Load template info
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        template_path = data['_template_image']
        annotations = data[template_path]
        
        # Parse annotations
        template_bbox = None
        other_bboxes = []
        crop_area = None
        
        for ann in annotations:
            if ann['type'] == 'template':
                template_bbox = ann
            elif ann['type'] == 'crop_area':
                crop_area = ann
            else:
                other_bboxes.append(ann)
        
        # Load template
        scale = 1
        template_img_full = cv2.imread(template_path)
        if scale != 1.0:
            template_img_full = cv2.resize(template_img_full, None, fx=scale, fy=scale)
        
        # Apply crop area if exists
        crop_offset = (0, 0)  # (x_offset, y_offset) for mapping back to original
        
        if crop_area is not None:
            # Crop template
            x, y = int(crop_area['x']), int(crop_area['y'])
            w, h = int(crop_area['width']), int(crop_area['height'])
            crop_offset = (x, y)
            
            template_img = template_img_full[y:y+h, x:x+w]
            print(f"✂️  Template cropped: {template_img_full.shape} -> {template_img.shape}")
            
            # Save cropped template for debug
            cv2.imwrite("debug_template_crop_area.png", template_img)
        else:
            template_img = template_img_full
        
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        
        # Load ONNX model in this process
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        providers = ['CPUExecutionProvider']
        pipeline_sess = ort.InferenceSession(
            pipeline_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Helper function
        def resize_to_32(img):
            h, w = img.shape
            new_h = ((h + 31) // 32) * 32
            new_w = ((w + 31) // 32) * 32
            if new_h != h or new_w != w:
                resized = cv2.resize(img, (new_w, new_h))
                return resized, (w / new_w, h / new_h)
            return img, (1.0, 1.0)
        
        # Match process
        timings = {}
        t_total = time.time()
        
        # Load target
        target_img_original = cv2.imread(image_path)
        if scale != 1.0:
            target_img_full = cv2.resize(target_img_original, None, fx=scale, fy=scale)
        else:
            target_img_full = target_img_original
        
        # Apply crop area if exists (same as template)
        if crop_area is not None:
            x, y = int(crop_area['x']), int(crop_area['y'])
            w, h = int(crop_area['width']), int(crop_area['height'])
            target_img = target_img_full[y:y+h, x:x+w]
            print(f"✂️  Target cropped: {target_img_full.shape} -> {target_img.shape}")
            
            # Save cropped target for debug
            cv2.imwrite("debug_target_crop_area.png", target_img)
        else:
            target_img = target_img_full
        
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        
        # Resize
        template_resized, template_scale = resize_to_32(template_gray)
        target_resized, target_scale = resize_to_32(target_gray)
        
        # Prepare tensors
        template_tensor = np.ascontiguousarray(
            template_resized.astype(np.float32)[None, None] / 255.0
        )
        target_tensor = np.ascontiguousarray(
            target_resized.astype(np.float32)[None, None] / 255.0
        )
        batch_input = np.ascontiguousarray(
            np.concatenate([template_tensor, target_tensor], axis=0)
        )
        
        # Run ONNX inference
        outputs = pipeline_sess.run(None, {'images': batch_input})
        kpts, matches, mscores = outputs
        
        # Postprocess
        batch_mask = matches[:, 0] == 0
        batch_matches = matches[batch_mask]
        batch_mscores = mscores[batch_mask]
        
        kpts0 = kpts[0].astype(np.float32)
        kpts1 = kpts[1].astype(np.float32)
        
        score_threshold = 0.3
        valid_mask = batch_mscores > score_threshold
        valid_matches = batch_matches[valid_mask]
        
        m_kpts0 = kpts0[valid_matches[:, 1]].copy()
        m_kpts1 = kpts1[valid_matches[:, 2]].copy()
        
        m_kpts0[:, 0] *= template_scale[0]
        m_kpts0[:, 1] *= template_scale[1]
        m_kpts1[:, 0] *= target_scale[0]
        m_kpts1[:, 1] *= target_scale[1]
        
        if len(m_kpts0) < 10:
            result_queue.put({
                'success': False,
                'error': f'Too few matches: {len(m_kpts0)}'
            })
            return
        
        # Find homography
        H, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, 5.0)
        
        if H is None:
            result_queue.put({
                'success': False,
                'error': 'Homography estimation failed'
            })
            return
        
        inliers = np.sum(mask)
        confidence = inliers / len(m_kpts0)
        
        # Transform bboxes
        scale_matrix = np.array([
            [1/scale, 0, 0],
            [0, 1/scale, 0],
            [0, 0, 1]
        ])
        H_full = scale_matrix @ H @ np.linalg.inv(scale_matrix)
        
        transformed_bboxes = []
        
        # Adjust template bbox if crop area exists
        # (subtract crop offset from template points)
        adjusted_template_pts = np.array(template_bbox['points'], dtype=np.float32)
        if crop_area is not None:
            adjusted_template_pts[:, 0] -= crop_offset[0]
            adjusted_template_pts[:, 1] -= crop_offset[1]
        
        # Template bbox
        template_pts = adjusted_template_pts.reshape(-1, 1, 2)
        template_transformed = cv2.perspectiveTransform(template_pts, H_full)
        
        # Add crop offset back to transformed points (map to original image coordinates)
        if crop_area is not None:
            template_transformed[:, :, 0] += crop_offset[0]
            template_transformed[:, :, 1] += crop_offset[1]
        transformed_bboxes.append({
            'type': 'template',
            'points': template_transformed.reshape(-1, 2).tolist()
        })
        
        # Other bboxes
        for bbox in other_bboxes:
            # Adjust bbox if crop area exists
            adjusted_pts = np.array(bbox['points'], dtype=np.float32)
            if crop_area is not None:
                adjusted_pts[:, 0] -= crop_offset[0]
                adjusted_pts[:, 1] -= crop_offset[1]
            
            pts = adjusted_pts.reshape(-1, 1, 2)
            pts_transformed = cv2.perspectiveTransform(pts, H_full)
            
            # Add crop offset back (map to original image coordinates)
            if crop_area is not None:
                pts_transformed[:, :, 0] += crop_offset[0]
                pts_transformed[:, :, 1] += crop_offset[1]
            
            transformed_bboxes.append({
                'type': bbox['type'],
                'points': pts_transformed.reshape(-1, 2).tolist()
            })
        
        timings['total'] = (time.time() - t_total) * 1000
        
        # Perform OCR on detected regions
        ocr_results = []
        if OCR_AVAILABLE:
            try:
                ocr_start = time.time()
                from text_recognizer import TextRecognizer
                
                # Initialize OCR model
                recognizer = TextRecognizer(
                    model_path='../languages/english/rec.onnx',
                    dict_path='../languages/english/dict.txt',
                    use_gpu=False
                )
                
                # Crop regions for OCR (excluding template)
                cropped_regions = []

                for bbox in transformed_bboxes:
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
                    # Use original full image (not cropped) for warping
                    cropped = cv2.warpPerspective(target_img_full, M, (width, height))
                    
                    cropped_regions.append({
                        'image': cropped,
                        'type': bbox['type'],
                        'bbox_idx': len(cropped_regions)
                    })

                # for bbox in transformed_bboxes:
                #     if bbox['type'] in ['template']:
                #         continue
                        
                #     # Get bounding box points (4 corners)
                #     pts = np.array(bbox['points'], dtype=np.float32)
                    
                #     # Get straight bounding rectangle (no rotation)
                #     x_coords = pts[:, 0]
                #     y_coords = pts[:, 1]
                #     x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                #     y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                    
                #     width = x_max - x_min
                #     height = y_max - y_min
                    
                #     if width > 0 and height > 0:
                #         # Crop the region directly (no rotation)
                #         cropped = target_img_full[y_min:y_max, x_min:x_max]
                        
                #         # Add padding if height is too small
                #         min_height = 48
                #         if height < min_height:
                #             pad_top = (min_height - height) // 2
                #             pad_bottom = min_height - height - pad_top
                            
                #             # Create white canvas with padding
                #             padded = np.ones((min_height, width, 3), dtype=np.uint8) * 255
                #             # Place cropped image in center
                #             padded[pad_top:pad_top+height, :] = cropped
                #             cropped = padded
                        
                #         cropped_regions.append({
                #             'image': cropped,
                #             'type': bbox['type'],
                #             'bbox_idx': len(cropped_regions)
                #         })
                
                # Run batch OCR and Barcode decoding
                if cropped_regions:
                    images = [r['image'] for r in cropped_regions]

                    for i, image in enumerate(images):
                        cv2.imwrite(f"debug_ocr_{i}.png", image)
                    
                    # Process each region based on type
                    for i, region in enumerate(cropped_regions):
                        image = region['image']
                        region_type = region['type']
                        
                        # Decode barcode if type is 'barcode'
                        if region_type == 'barcode' and BARCODE_AVAILABLE:
                            try:
                                # Decode barcode
                                barcodes = pyzbar.decode(image)
                                
                                if barcodes:
                                    # Use first detected barcode
                                    barcode = barcodes[0]
                                    text = barcode.data.decode('utf-8')
                                    barcode_type = barcode.type
                                    
                                    ocr_results.append({
                                        'type': 'barcode',
                                        'text': text,
                                        'barcode_type': barcode_type,
                                        'confidence': 1.0  # pyzbar doesn't provide confidence
                                    })
                                else:
                                    # No barcode detected, fallback to OCR
                                    text, conf = recognizer.recognize(image)
                                    ocr_results.append({
                                        'type': region_type,
                                        'text': text,
                                        'confidence': float(conf),
                                        'note': 'No barcode detected, used OCR'
                                    })
                            except Exception as e:
                                print(f"Barcode decode error: {e}")
                                # Fallback to OCR on error
                                text, conf = recognizer.recognize(image)
                                ocr_results.append({
                                    'type': region_type,
                                    'text': text,
                                    'confidence': float(conf)
                                })
                        else:
                            # Use OCR for text/datecode or when barcode lib not available
                            text, conf = recognizer.recognize(image)
                            ocr_results.append({
                                'type': region_type,
                                'text': text,
                                'confidence': float(conf)
                            })
                
                timings['ocr'] = (time.time() - ocr_start) * 1000
                print(ocr_results)
                
            except Exception as e:
                print(f"OCR error: {e}")
                import traceback
                traceback.print_exc()
        
        # Draw on image (use original full image, not cropped)
        annotated = target_img_original.copy() if scale == 1.0 else target_img_full.copy()
        colors = {
            'template': (0, 255, 0),
            'text': (255, 165, 0),
            'barcode': (255, 0, 255),
            'datecode': (0, 255, 255)
        }
        
        for bbox in transformed_bboxes:
            pts = np.array(bbox['points'], dtype=np.int32)
            color = colors.get(bbox['type'], (255, 255, 255))
            cv2.polylines(annotated, [pts], True, color, 3)
            center = pts.mean(axis=0).astype(int)
            cv2.putText(annotated, bbox['type'], tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Return results via queue
        result_queue.put({
            'success': True,
            'confidence': confidence,
            'inliers': int(inliers),
            'total_matches': len(m_kpts0),
            'transformed_bboxes': transformed_bboxes,
            'annotated_image': annotated,
            'ocr_results': ocr_results,
            'timings': timings
        })
        
    except Exception as e:
        import traceback
        result_queue.put({
            'success': False,
            'error': f'{str(e)}\n{traceback.format_exc()}'
        })


class InferenceMonitor(QThread):
    """Thread to monitor the inference process and get results"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, result_queue, process):
        super().__init__()
        self.result_queue = result_queue
        self.process = process
        self.running = True
    
    def run(self):
        try:
            # Wait for result with timeout
            timeout = 60  # 60 seconds max
            start_time = time.time()
            
            while self.running and (time.time() - start_time) < timeout:
                if not self.result_queue.empty():
                    result = self.result_queue.get()
                    
                    if result['success']:
                        self.finished.emit(result)
                    else:
                        self.error.emit(result.get('error', 'Unknown error'))
                    return
                
                # Check if process is still alive
                if not self.process.is_alive():
                    if self.result_queue.empty():
                        self.error.emit("Process terminated without result")
                    return
                
                time.sleep(0.1)
            
            if time.time() - start_time >= timeout:
                self.error.emit("Inference timeout (60s)")
                
        except Exception as e:
            self.error.emit(f"Monitor error: {str(e)}")
    
    def stop(self):
        self.running = False


class SuperPointMatcherONNX:
    """Simple wrapper that stores paths for multiprocessing inference"""
    def __init__(self, json_path: str, pipeline_path: str, scale: float = 1.0, verbose: bool = False):
        self.json_path = json_path
        self.pipeline_path = pipeline_path
        self.scale = scale
        self.verbose = verbose
        
        # Just validate files exist
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON not found: {json_path}")
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"ONNX model not found: {pipeline_path}")
        
        # Load basic info for display
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.template_path = data['_template_image']
        self.annotations = data[self.template_path]
        
        self.template_bbox = None
        self.other_bboxes = []
        for ann in self.annotations:
            if ann['type'] == 'template':
                self.template_bbox = ann
            elif ann['type'] not in ['crop_area']:
                self.other_bboxes.append(ann)
        
        print(f"✅ Initialized SuperPointMatcherONNX (multiprocessing mode)")
        print(f"   Template: {Path(self.template_path).name}")
        print(f"   Bboxes: template + {len(self.other_bboxes)} regions")
        print(f"   ⚠️  Inference will run in separate process (no Qt conflicts)")


class InferenceWorker(QThread):
    """Worker that manages the multiprocessing inference"""
    finished = pyqtSignal(dict, object)  # result, annotated_image
    error = pyqtSignal(str)
    
    def __init__(self, matcher, image_path):
        super().__init__()
        self.matcher = matcher
        self.image_path = image_path
        self.process = None
        self.monitor = None
    
    def run(self):
        try:
            print(f"Starting inference on: {self.image_path}")
            
            # Create queue for results
            result_queue = mp.Queue()
            
            # Start inference in separate process
            self.process = mp.Process(
                target=run_inference_process,
                args=(
                    self.matcher.json_path,
                    self.matcher.pipeline_path,
                    self.image_path,
                    result_queue
                )
            )
            self.process.start()
            
            # Monitor the process
            self.monitor = InferenceMonitor(result_queue, self.process)
            self.monitor.finished.connect(self._on_finished)
            self.monitor.error.connect(self._on_error)
            self.monitor.start()
            
            # Wait for monitor to finish
            self.monitor.wait()
            
            # Clean up
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Exception in inference worker:\n{error_details}")
            self.error.emit(f"{str(e)}\n\nSee console for details")
    
    def _on_finished(self, result):
        """Handle successful inference"""
        annotated = result['annotated_image']
        print(f"Inference successful. Confidence: {result['confidence']:.1%}")
        self.finished.emit(result, annotated)
    
    def _on_error(self, error_msg):
        """Handle inference error"""
        print(f"Inference failed: {error_msg}")
        self.error.emit(error_msg)


class InferenceWidget(QWidget):
    """Widget for template inference"""
    
    def __init__(self):
        super().__init__()
        self.matcher = None
        self.image_folder = None
        self.image_files = []
        self.current_index = -1
        self.current_result = None
        self.results_cache = {}  # Store results: {filename: {result, annotated_image}}
        
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
        
        # Auto-run checkbox and Run button in horizontal layout
        button_layout = QHBoxLayout()
        
        from PyQt5.QtWidgets import QCheckBox
        self.auto_run_checkbox = QCheckBox("Auto-run on next/prev")
        self.auto_run_checkbox.setChecked(True)  # Default enabled
        button_layout.addWidget(self.auto_run_checkbox)
        
        self.run_btn = QPushButton("Run Inference")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.clicked.connect(self.run_inference)
        self.run_btn.setEnabled(False)
        button_layout.addWidget(self.run_btn)
        
        center_layout.addLayout(button_layout)
        
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
        # Try multiple possible locations
        pipeline_paths = [
            os.path.join(os.path.dirname(os.path.dirname(json_path)), 'weights', 'superpoint_lightglue_pipeline.onnx'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', 'superpoint_lightglue_pipeline.onnx'),
            "../weights/superpoint_lightglue_pipeline.onnx",
            "weights/superpoint_lightglue_pipeline.onnx"
        ]
        
        pipeline_path = None
        for path in pipeline_paths:
            if os.path.exists(path):
                pipeline_path = path
                break
        
        if not pipeline_path:
            QMessageBox.critical(
                self,
                "Model Not Found",
                f"ONNX model not found. Tried:\n" + "\n".join(pipeline_paths[:2]) + "\n\n"
                "Please ensure the model file exists."
            )
            return
        
        try:
            self.status_label.setText("Loading template...")
            self.matcher = SuperPointMatcherONNX(
                json_path,
                pipeline_path,
                scale=0.5,
                verbose=True
            )
            
            self.json_path_label.setText(os.path.basename(json_path))
            self.info_label.setText(f"✅ Template loaded: {self.matcher.template_bbox['type']}")
            self.status_label.setText("Ready")
            
            if self.image_folder:
                self.run_btn.setEnabled(True)
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error loading matcher:\n{error_details}")
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to load matcher:\n{str(e)}\n\nSee console for details."
            )
            self.status_label.setText("Error loading template")
    
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
        
        # Load saved results if exist
        self._load_results_from_file()
        
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
            
            # Check if we have cached results for this image
            if filename in self.results_cache:
                # Load cached results
                cached = self.results_cache[filename]
                self.current_result = cached['result']
                
                # Check if we have annotated image in memory
                if cached['annotated_image'] is not None:
                    # Display cached annotated image
                    annotated_image = cached['annotated_image']
                else:
                    # Generate annotated image from saved results
                    annotated_image = self._generate_annotated_image(image_path, cached['result'])
                    # Cache the generated image
                    self.results_cache[filename]['annotated_image'] = annotated_image
                
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
                
                # Display cached results
                self._display_results(cached['result'])
                self.status_label.setText(f"📁 Cached - Image {self.current_index + 1}/{len(self.image_files)}: {filename}")
            else:
                # Clear previous results
                self.results_text.clear()
                self.current_result = None
                
                # Auto-run inference if enabled and matcher is available
                if self.auto_run_checkbox.isChecked() and self.matcher:
                    self.run_inference()
    
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
        
        # Save to cache
        if self.current_index >= 0 and self.current_index < len(self.image_files):
            filename = self.image_files[self.current_index]
            self.results_cache[filename] = {
                'result': result,
                'annotated_image': annotated_image.copy()
            }
            # Save to file
            self._save_results_to_file()
        
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
        self._display_results(result)
        self.status_label.setText(f"✅ Success - Confidence: {result['confidence']:.1%}")
    
    def _display_results(self, result):
        """Display inference results in text panel"""
        timings = result.get('timings', {})
        ocr_results = result.get('ocr_results', [])
        # Processing Time: {timings.get('total', 0):.0f}ms
        results_text = f"""✅ Inference Successful

Confidence: {result['confidence']:.1%}
Inliers: {result['inliers']}/{result['total_matches']}

"""
        
        if timings.get('ocr'):
            results_text += f"OCR Time: {timings.get('ocr', 0):.0f}ms\n"
        
        results_text += "\nDetected Regions:\n"
        for bbox in result['transformed_bboxes']:
            if bbox['type'] != 'template':
                results_text += f"  • {bbox['type']}\n"
        
        # Display OCR results
        if ocr_results:
            results_text += "\n" + "="*40 + "\n"
            results_text += "📝 RECOGNITION RESULTS\n"
            results_text += "="*40 + "\n\n"
            
            for i, ocr in enumerate(ocr_results, 1):
                region_type = ocr['type'].upper()
                
                # Different display for barcode vs text
                if ocr['type'] == 'barcode' and 'barcode_type' in ocr:
                    results_text += f"Region {i} [BARCODE - {ocr['barcode_type']}]:\n"
                    results_text += f"  📊 Data: {ocr['text']}\n"
                    if 'note' in ocr:
                        results_text += f"  ⚠️  {ocr['note']}\n"
                else:
                    results_text += f"Region {i} [{region_type}]:\n"


                    ocr_text = ocr['text'].replace(":", "")
                    results_text += f"  📝 Text: {ocr_text}\n"
                    # results_text += f"  🎯 Confidence: {ocr['confidence']:.1%}\n"
                
                results_text += "\n"
        else:
            results_text += "\n" + "="*40 + "\n"
            results_text += "📝 RECOGNITION RESULTS\n"
            results_text += "="*40 + "\n"
            if OCR_AVAILABLE:
                results_text += "\nNo text/barcode regions detected.\n"
            else:
                results_text += "\nOCR not available (text_recognizer.py not found)\n"
        
        self.results_text.setText(results_text)
    
    def on_inference_error(self, error_msg):
        """Handle inference error"""
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        
        self.results_text.setText(f"❌ Inference Failed\n\n{error_msg}")
        self.status_label.setText(f"❌ Error: {error_msg}")
    
    def _get_results_file_path(self):
        """Get path to results JSON file"""
        if not self.image_folder:
            return None
        return os.path.join(self.image_folder, 'inference_results.json')
    
    def _save_results_to_file(self):
        """Save all cached results to JSON file"""
        results_path = self._get_results_file_path()
        if not results_path:
            return
        
        try:
            # Prepare data (exclude annotated images, save only results)
            save_data = {}
            for filename, cache in self.results_cache.items():
                result = cache['result']
                # Convert numpy types to native Python types for JSON serialization
                save_data[filename] = {
                    'confidence': float(result.get('confidence', 0)),
                    'inliers': int(result.get('inliers', 0)),
                    'total_matches': int(result.get('total_matches', 0)),
                    'transformed_bboxes': result.get('transformed_bboxes', []),
                    'ocr_results': result.get('ocr_results', []),
                    'timings': {k: float(v) for k, v in result.get('timings', {}).items()}
                }
            
            # Save to file
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved to: {results_path}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _load_results_from_file(self):
        """Load cached results from JSON file"""
        results_path = self._get_results_file_path()
        if not results_path or not os.path.exists(results_path):
            self.results_cache = {}
            return
        
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            # We only have result data, not annotated images
            # Images will be regenerated on display if needed
            self.results_cache = {}
            for filename, result_data in save_data.items():
                self.results_cache[filename] = {
                    'result': result_data,
                    'annotated_image': None  # Will be loaded/generated when needed
                }
            
            print(f"Loaded {len(self.results_cache)} cached results from: {results_path}")
            
        except Exception as e:
            print(f"Error loading results: {e}")
            self.results_cache = {}
    
    def _generate_annotated_image(self, image_path, result):
        """Generate annotated image from saved results"""
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Draw bounding boxes
        annotated = img.copy()
        colors = {
            'template': (0, 255, 0),
            'text': (255, 165, 0),
            'barcode': (255, 0, 255),
            'datecode': (0, 255, 255)
        }
        
        transformed_bboxes = result.get('transformed_bboxes', [])
        for bbox in transformed_bboxes:
            pts = np.array(bbox['points'], dtype=np.int32)
            color = colors.get(bbox['type'], (255, 255, 255))
            cv2.polylines(annotated, [pts], True, color, 3)
            center = pts.mean(axis=0).astype(int)
            cv2.putText(annotated, bbox['type'], tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return annotated


# Multiprocessing setup for macOS/Windows compatibility
if __name__ != '__main__':
    mp.set_start_method('spawn', force=True)
