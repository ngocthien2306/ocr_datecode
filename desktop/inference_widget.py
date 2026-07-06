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
                              QSplitter, QMessageBox, QListWidgetItem, QProgressBar,
                              QScrollArea, QGroupBox, QSpinBox)
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


def run_inference_process(json_path, pipeline_path, image_path, result_queue,
                            ocr_backend='auto'):
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

        def to_polygon(ann):
            # Normalize any rectangle bbox into polygon form so downstream
            # code can assume `points` is always present.
            if ann.get('shape') == 'polygon' and 'points' in ann:
                return ann
            x, y = float(ann['x']), float(ann['y'])
            w, h = float(ann['width']), float(ann['height'])
            return {
                **ann,
                'shape': 'polygon',
                'points': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            }

        for ann in annotations:
            if ann['type'] == 'template':
                template_bbox = to_polygon(ann)
            elif ann['type'] == 'crop_area':
                crop_area = ann
            else:
                other_bboxes.append(to_polygon(ann))
        
        # Per-image debug folder: desktop/debug_output/<target_filename>/...
        # All `cv2.imwrite("debug_*.png", ...)` calls go here so successive
        # inferences don't overwrite each other and you can audit per-image.
        debug_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'debug_output',
            os.path.splitext(os.path.basename(image_path))[0],
        )
        os.makedirs(debug_dir, exist_ok=True)

        def _debug_path(name):
            return os.path.join(debug_dir, name)

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
            
            cv2.imwrite(_debug_path("template_crop_area.png"), template_img)
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
            
            cv2.imwrite(_debug_path("target_crop_area.png"), target_img)
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
        ocr_backend_used = None
        if OCR_AVAILABLE:
            try:
                ocr_start = time.time()
                # Use the shared factory so user's backend choice (auto/tensorrt/
                # onnx_gpu/onnx_cpu) is honoured, with graceful fallback.
                import config as _cfg
                _cfg.OCR_BACKEND = ocr_backend
                recognizer, ocr_backend_used = _cfg.get_recognizer()
                print(f"[inference] OCR backend = {ocr_backend_used}")

                def recognize_region_text(region_image):
                    if hasattr(recognizer, 'recognize_with_chars'):
                        try:
                            text, conf, chars = recognizer.recognize_with_chars(region_image)
                            return text, conf, chars
                        except Exception as e:
                            print(f"[inference] recognize_with_chars failed: {e}")
                    text, conf = recognizer.recognize(region_image)
                    return text, conf, []
                
                # Crop regions for OCR (excluding template).
                # Also crop the SAME region from the template image so we can
                # later run per-character comparison between template & target.
                # If a region has annotated `chars`, we additionally pre-pair
                # individual char crops (template ↔ target via homography +
                # local matchTemplate refinement) so segmentation is bypassed.
                cropped_regions = []

                # Iterate paired (transformed bbox in target, original bbox in template).
                # `other_bboxes` holds template-coord bboxes; `transformed_bboxes`
                # starts with the template region then mirrors `other_bboxes` in order.
                for other_bbox, t_bbox in zip(other_bboxes, transformed_bboxes[1:]):
                    pts_tgt = np.array(t_bbox['points'], dtype=np.float32)
                    pts_tmpl = np.array(other_bbox['points'], dtype=np.float32)

                    width = int(max(
                        np.linalg.norm(pts_tgt[0] - pts_tgt[1]),
                        np.linalg.norm(pts_tgt[2] - pts_tgt[3])
                    ))
                    height = int(max(
                        np.linalg.norm(pts_tgt[1] - pts_tgt[2]),
                        np.linalg.norm(pts_tgt[3] - pts_tgt[0])
                    ))

                    dst_pts = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype=np.float32)

                    M = cv2.getPerspectiveTransform(pts_tgt, dst_pts)
                    cropped = cv2.warpPerspective(target_img_full, M, (width, height))

                    M_tmpl = cv2.getPerspectiveTransform(pts_tmpl, dst_pts)
                    tmpl_cropped = cv2.warpPerspective(template_img_full, M_tmpl, (width, height))

                    # If user annotated explicit char positions, pre-pair them
                    char_pairs = []
                    chars_in_template = other_bbox.get('chars') or []
                    if chars_in_template:
                        try:
                            from char_segmenter import (
                                char_polygon_to_points, warp_char_with_refinement,
                            )
                            for char_dict in chars_in_template:
                                char_pts = char_polygon_to_points(char_dict)
                                tmpl_c, tgt_c, _off, _score = warp_char_with_refinement(
                                    template_img_full, target_img_full, H_full, char_pts,
                                    search_radius=0.15,
                                    refine_min_score=0.3,
                                )
                                if tmpl_c is not None and tgt_c is not None:
                                    char_pairs.append((tmpl_c, tgt_c))
                        except Exception as e:
                            print(f"[inference] char-bbox pairing failed: {e}")
                            char_pairs = []

                    cropped_regions.append({
                        'image': cropped,
                        'template_image': tmpl_cropped,
                        'char_pairs': char_pairs,  # empty when no annotated chars
                        'type': t_bbox['type'],
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

                    for i, region in enumerate(cropped_regions):
                        # Tag with bbox type so debug folder is self-describing
                        cv2.imwrite(
                            _debug_path(f"ocr_{i}_{region['type']}.png"),
                            region['image'],
                        )
                        # Save matching template crop too, makes diffing trivial
                        if region.get('template_image') is not None:
                            cv2.imwrite(
                                _debug_path(f"ocr_{i}_{region['type']}_tmpl.png"),
                                region['template_image'],
                            )
                    
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
                                    text, conf, chars = recognize_region_text(image)
                                    ocr_results.append({
                                        'type': region_type,
                                        'text': text,
                                        'confidence': float(conf),
                                        'chars': chars,
                                        'note': 'No barcode detected, used OCR'
                                    })
                            except Exception as e:
                                print(f"Barcode decode error: {e}")
                                # Fallback to OCR on error
                                text, conf, chars = recognize_region_text(image)
                                ocr_results.append({
                                    'type': region_type,
                                    'text': text,
                                    'confidence': float(conf),
                                    'chars': chars
                                })
                        else:
                            # Use OCR for text/datecode or when barcode lib not available
                            text, conf, chars = recognize_region_text(image)
                            ocr_results.append({
                                'type': region_type,
                                'text': text,
                                'confidence': float(conf),
                                'chars': chars
                            })
                
                timings['ocr'] = (time.time() - ocr_start) * 1000
                print(ocr_results)

            except Exception as e:
                print(f"OCR error: {e}")
                import traceback
                traceback.print_exc()

        # Per-character comparison between template and target text regions.
        # Two paths:
        #  1) Region has annotated `chars` → use compare_char_pairs (no segmentation,
        #     char count guaranteed to match).
        #  2) Otherwise fall back to compare_arrays (segments both sides — old path).
        char_comparisons = []
        try:
            from char_segmenter import compare_arrays, compare_char_pairs
            from params_store import load_trained_params

            trained_record = load_trained_params(json_path)
            shared_params = trained_record['params'] if trained_record else None
            using_trained = trained_record is not None

            cmp_start = time.time()
            for region in cropped_regions:
                if region['type'] == 'barcode':
                    continue
                tmpl_crop = region.get('template_image')
                if tmpl_crop is None:
                    continue

                if region.get('char_pairs'):
                    # NEW path: explicit char bboxes → no segmentation
                    out = compare_char_pairs(region['char_pairs'], params=shared_params)
                    strip = out['strip']
                    results = out['results']
                    overall_pass = out['overall_pass']
                else:
                    # OLD path: segment both sides
                    strip, results, overall_pass = compare_arrays(
                        tmpl_crop, region['image'], params=shared_params,
                    )

                if strip is None:
                    continue
                char_comparisons.append({
                    'type': region['type'],
                    'bbox_idx': region['bbox_idx'],
                    'image': strip,
                    'template_image': tmpl_crop,
                    'target_image': region['image'],
                    'overall_pass': overall_pass,
                    'n_chars_template': sum(1 for _ in results),
                    'used_char_bboxes': bool(region.get('char_pairs')),
                })
            timings['char_compare'] = (time.time() - cmp_start) * 1000
        except ImportError:
            using_trained = False
        except Exception as e:
            print(f"Char comparison error: {e}")
            import traceback
            traceback.print_exc()
            using_trained = False

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
            'char_comparisons': char_comparisons,
            'using_trained_params': bool(using_trained) if 'using_trained' in dir() else False,
            'ocr_backend_used': ocr_backend_used,
            'timings': timings
        })
        
    except Exception as e:
        import traceback
        result_queue.put({
            'success': False,
            'error': f'{str(e)}\n{traceback.format_exc()}'
        })


def train_params_process(json_path, pipeline_path, target_image_paths,
                         n_trials, n_ng_per_pair, locked_pass_threshold,
                         result_queue, control_queue):
    """
    Subprocess: extract region pairs from N target images, then run
    auto_tune_multi_image to find best shared params.

    Streams progress via `result_queue` as dicts with key `kind` =
    'progress' | 'extract' | 'done' | 'error'.

    `control_queue` is checked between trials for a 'cancel' message.
    """
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
        import json
        import time

        # ---- Load template + annotations (same parsing as run_inference_process) ----
        with open(json_path, 'r') as f:
            data = json.load(f)
        template_path = data['_template_image']
        annotations = data[template_path]

        def to_polygon(ann):
            if ann.get('shape') == 'polygon' and 'points' in ann:
                return ann
            x, y = float(ann['x']), float(ann['y'])
            w, h = float(ann['width']), float(ann['height'])
            return {**ann, 'shape': 'polygon',
                    'points': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]}

        template_bbox = None
        other_bboxes = []
        crop_area = None
        for ann in annotations:
            if ann['type'] == 'template':
                template_bbox = to_polygon(ann)
            elif ann['type'] == 'crop_area':
                crop_area = ann
            else:
                other_bboxes.append(to_polygon(ann))

        if template_bbox is None or not other_bboxes:
            result_queue.put({'kind': 'error',
                              'error': 'No template or text regions in annotations'})
            return

        # ---- Load template image + crop area ----
        template_img_full = cv2.imread(template_path)
        if template_img_full is None:
            result_queue.put({'kind': 'error',
                              'error': f'Cannot read template: {template_path}'})
            return

        crop_offset = (0, 0)
        if crop_area is not None:
            cx, cy = int(crop_area['x']), int(crop_area['y'])
            cw, ch = int(crop_area['width']), int(crop_area['height'])
            crop_offset = (cx, cy)
            template_img = template_img_full[cy:cy + ch, cx:cx + cw]
        else:
            template_img = template_img_full
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

        # ---- Load pipeline ONNX ----
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        pipeline_sess = ort.InferenceSession(
            pipeline_path, sess_options=sess_options,
            providers=['CPUExecutionProvider'],
        )

        def resize_to_32(img):
            h, w = img.shape
            new_h = ((h + 31) // 32) * 32
            new_w = ((w + 31) // 32) * 32
            if new_h != h or new_w != w:
                resized = cv2.resize(img, (new_w, new_h))
                return resized, (w / new_w, h / new_h)
            return img, (1.0, 1.0)

        template_resized, template_scale_back = resize_to_32(template_gray)

        # ---- For each selected target image, run matching + extract crops ----
        prepared_pairs = []   # list of (tmpl_crop_bgr, tgt_crop_bgr)
        for img_idx, target_path in enumerate(target_image_paths):
            result_queue.put({'kind': 'extract', 'i': img_idx + 1,
                              'n': len(target_image_paths),
                              'name': os.path.basename(target_path)})

            target_img_full = cv2.imread(target_path)
            if target_img_full is None:
                continue
            if crop_area is not None:
                target_img = target_img_full[
                    crop_offset[1]:crop_offset[1] + int(crop_area['height']),
                    crop_offset[0]:crop_offset[0] + int(crop_area['width']),
                ]
            else:
                target_img = target_img_full
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            target_resized, target_scale_back = resize_to_32(target_gray)

            # SuperPoint+LightGlue ONNX (mirror run_inference_process exactly)
            template_tensor = np.ascontiguousarray(
                template_resized.astype(np.float32)[None, None] / 255.0
            )
            target_tensor = np.ascontiguousarray(
                target_resized.astype(np.float32)[None, None] / 255.0
            )
            batch_input = np.ascontiguousarray(
                np.concatenate([template_tensor, target_tensor], axis=0)
            )
            try:
                outputs = pipeline_sess.run(None, {'images': batch_input})
            except Exception as e:
                print(f'[train] match failed on {target_path}: {e}')
                continue
            kpts, matches, mscores = outputs

            # `matches` rows are [batch_idx, kpts0_idx, kpts1_idx]
            batch_mask = matches[:, 0] == 0
            batch_matches = matches[batch_mask]
            batch_mscores = mscores[batch_mask]

            kpts0 = kpts[0].astype(np.float32)
            kpts1 = kpts[1].astype(np.float32)

            score_threshold = 0.3
            valid_mask = batch_mscores > score_threshold
            valid_matches = batch_matches[valid_mask]
            if len(valid_matches) < 10:
                continue

            m_kpts0 = kpts0[valid_matches[:, 1]].copy()
            m_kpts1 = kpts1[valid_matches[:, 2]].copy()
            m_kpts0[:, 0] *= template_scale_back[0]
            m_kpts0[:, 1] *= template_scale_back[1]
            m_kpts1[:, 0] *= target_scale_back[0]
            m_kpts1[:, 1] *= target_scale_back[1]

            H, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, 5.0)
            if H is None:
                continue

            # For each non-template, non-barcode region: warp BOTH to canonical rect
            for other_bbox in other_bboxes:
                if other_bbox['type'] == 'barcode':
                    continue

                # Adjust template points for crop_area offset, transform to target,
                # then add offset back
                tmpl_pts_adj = np.array(other_bbox['points'], dtype=np.float32)
                if crop_area is not None:
                    tmpl_pts_adj[:, 0] -= crop_offset[0]
                    tmpl_pts_adj[:, 1] -= crop_offset[1]
                pts_t = cv2.perspectiveTransform(tmpl_pts_adj.reshape(-1, 1, 2), H)
                pts_t = pts_t.reshape(-1, 2)
                if crop_area is not None:
                    pts_t[:, 0] += crop_offset[0]
                    pts_t[:, 1] += crop_offset[1]

                tmpl_pts = np.array(other_bbox['points'], dtype=np.float32)

                width = int(max(
                    np.linalg.norm(pts_t[0] - pts_t[1]),
                    np.linalg.norm(pts_t[2] - pts_t[3]),
                ))
                height = int(max(
                    np.linalg.norm(pts_t[1] - pts_t[2]),
                    np.linalg.norm(pts_t[3] - pts_t[0]),
                ))
                if width < 8 or height < 8:
                    continue

                # If the region has annotated chars, generate ONE pair per char
                # via homography + local refinement. Skips segmentation entirely
                # for this region. Otherwise emit one whole-region pair (the
                # old behaviour, which auto-tune will still segment internally).
                chars_in_template = other_bbox.get('chars') or []
                if chars_in_template:
                    try:
                        from char_segmenter import (
                            char_polygon_to_points, warp_char_with_refinement,
                        )
                        for char_dict in chars_in_template:
                            char_pts = char_polygon_to_points(char_dict)
                            tmpl_c, tgt_c, _off, _score = warp_char_with_refinement(
                                template_img_full, target_img_full, H, char_pts,
                                search_radius=0.15, refine_min_score=0.3,
                            )
                            if tmpl_c is not None and tgt_c is not None:
                                prepared_pairs.append((tmpl_c, tgt_c, 'char'))
                        continue
                    except Exception as e:
                        print(f"[train] char-bbox pairing failed: {e}")
                        # fall through to whole-region path

                dst_pts = np.array([[0, 0], [width - 1, 0],
                                     [width - 1, height - 1], [0, height - 1]],
                                    dtype=np.float32)
                M_tgt = cv2.getPerspectiveTransform(pts_t.astype(np.float32), dst_pts)
                M_tmpl = cv2.getPerspectiveTransform(tmpl_pts, dst_pts)
                tgt_crop = cv2.warpPerspective(target_img_full, M_tgt, (width, height))
                tmpl_crop = cv2.warpPerspective(template_img_full, M_tmpl, (width, height))
                prepared_pairs.append((tmpl_crop, tgt_crop, 'region'))

        if not prepared_pairs:
            result_queue.put({'kind': 'error',
                              'error': 'No region pairs could be extracted from selected images'})
            return

        # ---- Run auto-tune over the prepared pairs ----
        from auto_tune import auto_tune_multi_image

        cancelled = [False]

        def cancel_check():
            if cancelled[0]:
                return True
            try:
                while not control_queue.empty():
                    msg = control_queue.get_nowait()
                    if msg == 'cancel':
                        cancelled[0] = True
                        return True
            except Exception:
                pass
            return False

        def progress_cb(done, total, best):
            result_queue.put({'kind': 'progress', 'done': done,
                              'total': total, 'best': float(best)})

        # User-pinned PASS threshold: lock so random search doesn't override it
        base_params = None
        locked_keys = None
        if locked_pass_threshold is not None:
            base_params = {'pass_threshold': float(locked_pass_threshold)}
            locked_keys = {'pass_threshold'}

        best_params, best_metrics, _hist = auto_tune_multi_image(
            prepared_pairs,
            base_params=base_params,
            n_trials=int(n_trials),
            n_ng_per_pair=int(n_ng_per_pair),
            seed=42,
            locked_keys=locked_keys,
            progress_cb=progress_cb,
            cancel_cb=cancel_check,
        )

        result_queue.put({
            'kind': 'done',
            'best_params': best_params,
            'best_metrics': best_metrics or {},
            'n_pairs': len(prepared_pairs),
            'cancelled': cancelled[0],
        })

    except Exception as e:
        import traceback
        result_queue.put({'kind': 'error',
                          'error': f'{str(e)}\n{traceback.format_exc()}'})


class TrainingMonitor(QThread):
    """Drains the training subprocess result_queue and emits Qt signals."""
    progress = pyqtSignal(int, int, float)             # done, total, best_score
    extracting = pyqtSignal(int, int, str)             # i, n, filename
    finished_ok = pyqtSignal(dict, dict, int, bool)    # params, metrics, n_pairs, cancelled
    failed = pyqtSignal(str)

    def __init__(self, result_queue, control_queue, process, parent=None):
        super().__init__(parent)
        self.result_queue = result_queue
        self.control_queue = control_queue
        self.process = process
        self.running = True

    def cancel(self):
        try:
            self.control_queue.put('cancel')
        except Exception:
            pass

    def stop(self):
        self.running = False

    def run(self):
        try:
            while self.running:
                if not self.result_queue.empty():
                    msg = self.result_queue.get()
                    kind = msg.get('kind')
                    if kind == 'progress':
                        self.progress.emit(int(msg['done']), int(msg['total']),
                                            float(msg['best']))
                    elif kind == 'extract':
                        self.extracting.emit(int(msg['i']), int(msg['n']),
                                              str(msg['name']))
                    elif kind == 'done':
                        self.finished_ok.emit(
                            dict(msg.get('best_params') or {}),
                            dict(msg.get('best_metrics') or {}),
                            int(msg.get('n_pairs', 0)),
                            bool(msg.get('cancelled', False)),
                        )
                        return
                    elif kind == 'error':
                        self.failed.emit(str(msg.get('error', 'unknown error')))
                        return
                if not self.process.is_alive() and self.result_queue.empty():
                    self.failed.emit("Training process exited unexpectedly")
                    return
                time.sleep(0.05)
        except Exception as e:
            self.failed.emit(f"Monitor error: {e}")


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
    
    def __init__(self, matcher, image_path, ocr_backend='auto'):
        super().__init__()
        self.matcher = matcher
        self.image_path = image_path
        self.ocr_backend = ocr_backend
        self.process = None
        self.monitor = None

    def run(self):
        try:
            print(f"Starting inference on: {self.image_path} [OCR backend: {self.ocr_backend}]")

            result_queue = mp.Queue()
            self.process = mp.Process(
                target=run_inference_process,
                args=(
                    self.matcher.json_path,
                    self.matcher.pipeline_path,
                    self.image_path,
                    result_queue,
                    self.ocr_backend,
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
        # Training state (subprocess + monitor)
        self._train_process = None
        self._train_monitor = None
        self._train_result_queue = None
        self._train_control_queue = None

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
        
        # Image list (checkable for training selection)
        list_header = QLabel("Images")
        list_header.setObjectName("headerLabel")
        left_layout.addWidget(list_header)

        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        # Each item also has a Qt.CheckState set in _populate_image_list();
        # itemChanged fires when the checkbox is toggled.
        self.image_list.itemChanged.connect(self._on_image_check_changed)
        left_layout.addWidget(self.image_list)

        # Select-all / clear row
        sel_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Check all")
        self.select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        sel_row.addWidget(self.select_all_btn)
        self.select_none_btn = QPushButton("Uncheck")
        self.select_none_btn.clicked.connect(lambda: self._set_all_checked(False))
        sel_row.addWidget(self.select_none_btn)
        left_layout.addLayout(sel_row)

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

        # OCR backend selector
        from PyQt5.QtWidgets import QComboBox
        ocr_row = QHBoxLayout()
        ocr_row.addWidget(QLabel("OCR Backend:"))
        self.ocr_backend_combo = QComboBox()
        self.ocr_backend_combo.addItem("Auto (TensorRT → ONNX)", "auto")
        self.ocr_backend_combo.addItem("TensorRT (GPU)", "tensorrt")
        self.ocr_backend_combo.addItem("ONNX Runtime (GPU)", "onnx_gpu")
        self.ocr_backend_combo.addItem("ONNX Runtime (CPU)", "onnx_cpu")
        self.ocr_backend_combo.addItem("SMTR/SVTRv2 ONNX (CPU)", "smtr_onnx_cpu")
        self.ocr_backend_combo.addItem("SMTR/SVTRv2 ONNX (GPU)", "smtr_onnx_gpu")
        self.ocr_backend_combo.addItem("SMTR/SVTRv2 ONNX (TensorRT EP)", "smtr_onnx_trt")
        self.ocr_backend_combo.addItem("SMTR Attention ONNX (CPU)", "smtr_attn_onnx_cpu")
        self.ocr_backend_combo.addItem("SMTR Attention ONNX (GPU)", "smtr_attn_onnx_gpu")
        self.ocr_backend_combo.addItem("SMTR Attention ONNX (TensorRT EP)", "smtr_attn_onnx_trt")
        self.ocr_backend_combo.addItem("Tesseract (auto-pick lib)", "tesseract")
        self.ocr_backend_combo.addItem("Tesseract — pytesseract", "tesseract_pytesseract")
        self.ocr_backend_combo.addItem("Tesseract — tesserocr", "tesseract_tesserocr")
        self.ocr_backend_combo.addItem("EasyOCR (CPU)", "easyocr_cpu")
        self.ocr_backend_combo.addItem("EasyOCR (GPU)", "easyocr_gpu")
        self.ocr_backend_combo.addItem("RapidOCR", "rapidocr")
        self.ocr_backend_combo.setToolTip(
            "OCR engine:\n"
            "  • TensorRT — fastest, needs NVIDIA GPU + nvidia-tensorrt + pycuda\n"
            "  • ONNX     — PP-OCRv5 model, GPU or CPU\n"
            "  • SMTR/SVTRv2 ONNX — weights/rec_smtr_fp16.onnx + EN_symbol_dict.txt\n"
            "  • SMTR Attention ONNX — weights/rec_smtr_attn_fp16.onnx; exposes char bbox metadata\n"
            "  • Tesseract — classic OCR (needs eng tessdata), good for clean printed text\n"
            "  • EasyOCR — PyTorch deep-learning OCR, robust on stylised text (CPU/GPU)\n"
            "  • RapidOCR — PP-OCR via ONNXRuntime, lightweight, fast on CPU.\n"
            "                  Bypasses detection (we already cropped) → recognition only."
        )
        # Changing backend invalidates cached OCR results
        self.ocr_backend_combo.currentIndexChanged.connect(
            lambda _i: self.results_cache.clear()
        )
        ocr_row.addWidget(self.ocr_backend_combo, stretch=1)
        left_layout.addLayout(ocr_row)

        # === Train section ===
        train_group = QGroupBox("Train")
        train_layout = QVBoxLayout(train_group)
        train_layout.setSpacing(4)

        self.train_selected_label = QLabel("Selected: 0")
        train_layout.addWidget(self.train_selected_label)

        train_row1 = QHBoxLayout()
        train_row1.addWidget(QLabel("Trials:"))
        self.train_trials_input = QSpinBox()
        self.train_trials_input.setRange(5, 200)
        self.train_trials_input.setValue(20)
        train_row1.addWidget(self.train_trials_input)
        train_row1.addWidget(QLabel("NG/img:"))
        self.train_ng_input = QSpinBox()
        self.train_ng_input.setRange(0, 12)
        self.train_ng_input.setValue(4)
        train_row1.addWidget(self.train_ng_input)
        train_layout.addLayout(train_row1)

        # User-pinned per-char PASS threshold (locked during search).
        # Lower = more lenient acceptance per char.
        from PyQt5.QtWidgets import QDoubleSpinBox
        train_row_pt = QHBoxLayout()
        train_row_pt.addWidget(QLabel("PASS threshold (per char):"))
        self.train_pass_input = QDoubleSpinBox()
        self.train_pass_input.setRange(0.30, 1.00)
        self.train_pass_input.setSingleStep(0.01)
        self.train_pass_input.setDecimals(2)
        self.train_pass_input.setValue(0.75)
        self.train_pass_input.setToolTip(
            "Per-char PASS threshold — locked during training (auto-tune won't change it).\n"
            "Lower = more lenient. Strict per-region rule is still applied: "
            "ALL chars in a region must PASS for the region to PASS."
        )
        train_row_pt.addWidget(self.train_pass_input)
        train_layout.addLayout(train_row_pt)

        train_row2 = QHBoxLayout()
        self.train_btn = QPushButton("🤖 Train on selected")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self._start_training)
        train_row2.addWidget(self.train_btn)
        self.train_cancel_btn = QPushButton("Cancel")
        self.train_cancel_btn.setEnabled(False)
        self.train_cancel_btn.clicked.connect(self._cancel_training)
        train_row2.addWidget(self.train_cancel_btn)
        train_layout.addLayout(train_row2)

        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        train_layout.addWidget(self.train_progress)

        self.train_status = QLabel("")
        self.train_status.setWordWrap(True)
        self.train_status.setStyleSheet("color: #aaa;")
        train_layout.addWidget(self.train_status)

        self.trained_badge = QLabel("")
        self.trained_badge.setWordWrap(True)
        train_layout.addWidget(self.trained_badge)

        left_layout.addWidget(train_group)

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
        center_layout.addWidget(self.image_label, stretch=3)

        # === Character comparison strip (template vs target, per text region) ===
        self.compare_header = QLabel("Character Comparison")
        self.compare_header.setObjectName("headerLabel")
        self.compare_header.setVisible(False)
        center_layout.addWidget(self.compare_header)

        self.compare_scroll = QScrollArea()
        self.compare_scroll.setWidgetResizable(True)
        self.compare_scroll.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3e3e3e;")
        self.compare_scroll.setMinimumHeight(220)
        self.compare_scroll.setVisible(False)

        self.compare_container = QWidget()
        self.compare_layout = QVBoxLayout(self.compare_container)
        self.compare_layout.setContentsMargins(6, 6, 6, 6)
        self.compare_layout.setSpacing(8)
        self.compare_scroll.setWidget(self.compare_container)
        center_layout.addWidget(self.compare_scroll, stretch=2)

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
                self._update_train_selected_count()
            self._refresh_trained_badge()
                
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
        
        # Update list (each item gets a checkbox for training selection)
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for filename in self.image_files:
            item = QListWidgetItem(filename)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.image_list.addItem(item)
        self.image_list.blockSignals(False)

        self.info_label.setText(f"{len(self.image_files)} images found")

        if self.image_files:
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            if self.matcher:
                self.run_btn.setEnabled(True)
                self.train_btn.setEnabled(True)
        self._update_train_selected_count()
        self._refresh_trained_badge()
    
    def on_image_selected(self, item):
        """Handle image selection"""
        idx = self.image_list.row(item)
        self.current_index = idx
        self.load_current_image()

    def _on_image_check_changed(self, _item):
        """Update Train selected count whenever any checkbox toggles."""
        self._update_train_selected_count()

    def _set_all_checked(self, checked):
        state = Qt.Checked if checked else Qt.Unchecked
        self.image_list.blockSignals(True)
        for i in range(self.image_list.count()):
            self.image_list.item(i).setCheckState(state)
        self.image_list.blockSignals(False)
        self._update_train_selected_count()

    def _update_train_selected_count(self):
        n = sum(
            1 for i in range(self.image_list.count())
            if self.image_list.item(i).checkState() == Qt.Checked
        )
        self.train_selected_label.setText(f"Selected: {n}")
        if hasattr(self, 'train_btn'):
            # Need at least 1 checked image AND a loaded matcher to train
            self.train_btn.setEnabled(n > 0 and self.matcher is not None
                                     and not self._is_training())

    def _is_training(self):
        return getattr(self, '_train_monitor', None) is not None and \
               self._train_monitor.isRunning()

    # --------------------------------------------------------- Training

    def _start_training(self):
        if self._is_training():
            return
        if not self.matcher or not self.image_folder:
            return

        # Collect checked image paths
        target_paths = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.checkState() == Qt.Checked:
                target_paths.append(os.path.join(self.image_folder, item.text()))
        if not target_paths:
            return

        n_trials = int(self.train_trials_input.value())
        n_ng = int(self.train_ng_input.value())
        locked_pt = float(self.train_pass_input.value())

        # Spawn training subprocess + monitor thread
        self._train_result_queue = mp.Queue()
        self._train_control_queue = mp.Queue()
        self._train_process = mp.Process(
            target=train_params_process,
            args=(self.matcher.json_path, self.matcher.pipeline_path,
                  target_paths, n_trials, n_ng, locked_pt,
                  self._train_result_queue, self._train_control_queue),
        )
        self._train_process.start()

        # parent=self so Qt cleans the thread up if the widget is destroyed
        self._train_monitor = TrainingMonitor(
            self._train_result_queue, self._train_control_queue, self._train_process,
            parent=self,
        )
        self._train_monitor.progress.connect(self._on_train_progress)
        self._train_monitor.extracting.connect(self._on_train_extracting)
        self._train_monitor.finished_ok.connect(
            lambda p, m, n, c: self._on_train_done(p, m, n, c, target_paths)
        )
        self._train_monitor.failed.connect(self._on_train_failed)
        self._train_monitor.start()

        # Reflect state in UI
        self.train_btn.setEnabled(False)
        self.train_cancel_btn.setEnabled(True)
        self.train_progress.setVisible(True)
        self.train_progress.setRange(0, n_trials)
        self.train_progress.setValue(0)
        self.train_status.setText(f"Extracting regions from {len(target_paths)} images…")

    def _cancel_training(self):
        if self._is_training():
            self._train_monitor.cancel()
            self.train_status.setText("Cancelling…")

    def _on_train_extracting(self, i, n, name):
        self.train_status.setText(f"Extracting {i}/{n}: {name}")

    def _on_train_progress(self, done, total, best):
        self.train_progress.setMaximum(total)
        self.train_progress.setValue(done)
        self.train_status.setText(f"Trial {done}/{total} — best score: {best:.3f}")

    def _on_train_done(self, params, metrics, n_pairs, cancelled, target_paths):
        self._cleanup_training()
        if not metrics:
            QMessageBox.warning(
                self, "Training",
                "Training finished but no working config was found. "
                "Try more trials, more selected images, or check that segmentation works on the selected images."
            )
            return
        # Show summary + Apply/Discard
        verb = "cancelled early" if cancelled else "finished"
        n_pass = metrics.get('n_clean_pass', 0)
        mismatch = metrics.get('n_clean_count_mismatch', 0)
        mismatch_note = (
            f"\n  ⚠️ {mismatch} regions failed due to char-count mismatch"
            if mismatch > 0 else ""
        )
        msg = (
            f"Training {verb}.\n\n"
            f"Region pairs evaluated: {n_pairs}\n"
            f"Score: {metrics.get('score', 0):.3f}\n"
            f"  Clean PASS: {n_pass}/{n_pairs} regions "
            f"({metrics.get('clean_pass_rate', 0) * 100:.0f}%)"
            f"{mismatch_note}\n"
            f"  NG catch rate: {metrics.get('ng_catch_rate', 0) * 100:.0f}%\n"
            f"  Confidence margin: {metrics.get('margin', 0):+.3f}\n\n"
            "Apply this config and save to compare_params.json?"
        )
        reply = QMessageBox.question(
            self, "Training complete", msg,
            QMessageBox.Apply | QMessageBox.Discard,
            QMessageBox.Apply,
        )
        if reply == QMessageBox.Apply:
            try:
                from params_store import save_trained_params
                trained_on = [os.path.basename(p) for p in target_paths]
                path = save_trained_params(
                    self.matcher.json_path, params, metrics=metrics,
                    trained_on=trained_on,
                )
                self.train_status.setText(
                    f"<b style='color:#0c0'>Saved.</b> Subsequent inferences will use the trained config."
                )
                # Force re-inference of current image to pick up new params
                self.results_cache.clear()
                self._refresh_trained_badge()
                if self.current_index >= 0:
                    self.run_inference()
            except Exception as e:
                QMessageBox.critical(self, "Save failed", str(e))
        else:
            self.train_status.setText("Discarded.")

    def _on_train_failed(self, msg):
        self._cleanup_training()
        QMessageBox.critical(self, "Training failed", msg)

    def _cleanup_training(self):
        self.train_btn.setEnabled(True)
        self.train_cancel_btn.setEnabled(False)
        self.train_progress.setVisible(False)
        # Stop the monitor thread cleanly before discarding its reference
        try:
            if self._train_monitor is not None:
                self._train_monitor.stop()
                self._train_monitor.wait(2000)
        except Exception:
            pass
        # Best-effort cleanup of subprocess
        try:
            if self._train_process is not None and self._train_process.is_alive():
                self._train_process.terminate()
                self._train_process.join(timeout=2)
        except Exception:
            pass
        self._train_process = None
        self._train_monitor = None
        self._update_train_selected_count()

    def closeEvent(self, event):
        """Make sure background workers are torn down before the widget dies,
        otherwise QThread will abort the process with
        'QThread: Destroyed while thread is still running'."""
        if self._is_training():
            try:
                self._train_monitor.cancel()
            except Exception:
                pass
            self._cleanup_training()
        # Inference monitor also needs to stop if mid-run
        try:
            worker = getattr(self, 'worker', None)
            if worker is not None and worker.isRunning():
                worker.stop() if hasattr(worker, 'stop') else None
                worker.wait(2000)
        except Exception:
            pass
        super().closeEvent(event)

    def _refresh_trained_badge(self):
        """Show whether trained params are currently active for this template,
        and pre-fill the PASS-threshold spinbox from the trained value if any."""
        try:
            from params_store import load_trained_params
        except ImportError:
            self.trained_badge.setText("")
            return
        if not self.matcher:
            self.trained_badge.setText("")
            return
        rec = load_trained_params(self.matcher.json_path)
        if not rec:
            self.trained_badge.setText(
                "<span style='color:#888'>No trained params (using defaults — PASS threshold = "
                f"{self.train_pass_input.value():.2f})</span>"
            )
            return
        m = rec.get('metrics') or {}
        n = rec.get('n_images', 0)
        score = m.get('score', 0.0)
        trained_pt = rec.get('params', {}).get('pass_threshold')
        # Pre-fill the spinbox with the trained value so further training
        # iterations start from there (user can still change it).
        if trained_pt is not None and hasattr(self, 'train_pass_input'):
            self.train_pass_input.blockSignals(True)
            self.train_pass_input.setValue(float(trained_pt))
            self.train_pass_input.blockSignals(False)
        self.trained_badge.setText(
            f"<span style='color:#0c0'>● Using trained params</span> "
            f"(score {score:.2f}, n={n}, PASS thr {trained_pt:.2f})"
        )
    
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
                self._render_char_comparisons([])
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
        ocr_backend = self.ocr_backend_combo.currentData() or 'auto'
        self.worker = InferenceWorker(self.matcher, image_path, ocr_backend=ocr_backend)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.error.connect(self.on_inference_error)
        self.worker.start()
    
    def on_inference_finished(self, result, annotated_image):
        """Handle inference completion"""
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.current_result = result
        result['ocr_backend_key'] = self.ocr_backend_combo.currentData() or 'auto'
        
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
        backend = result.get('ocr_backend_used') or '?'
        self.status_label.setText(
            f"✅ Success — Confidence: {result['confidence']:.1%}  •  OCR: {backend}"
        )
    
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

                chars = ocr.get('chars') or []
                if chars:
                    n_boxes = sum(
                        1 for ch in chars
                        if all(k in ch for k in ('x0', 'y0', 'x1', 'y1'))
                    )
                    if n_boxes:
                        results_text += f"  🔠 Char boxes: {n_boxes}\n"
                    else:
                        results_text += f"  🔠 Chars: {len(chars)}\n"
                
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

        # Render the per-region character comparison strips below the main image.
        self._render_char_comparisons(result.get('char_comparisons') or [])

    def _render_char_comparisons(self, comparisons):
        """Populate the bottom area with one comparison strip per text region."""
        # Clear existing children
        while self.compare_layout.count():
            item = self.compare_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not comparisons:
            self.compare_header.setVisible(False)
            self.compare_scroll.setVisible(False)
            return

        for comp in comparisons:
            img = comp['image']  # BGR numpy
            h, w, ch = img.shape
            qt_img = QImage(img.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
            pix = QPixmap.fromImage(qt_img)

            # Header row: title + tune button
            header_row = QHBoxLayout()
            label_title = QLabel(
                f"[{comp['type'].upper()}] region #{comp['bbox_idx']} — "
                f"{'PASS' if comp['overall_pass'] else 'FAIL'}"
            )
            color = "#00C800" if comp['overall_pass'] else "#FF4040"
            label_title.setStyleSheet(f"color: {color}; font-weight: bold;")
            header_row.addWidget(label_title)
            header_row.addStretch(1)

            tune_btn = QPushButton("🔧 Tune")
            tune_btn.setEnabled(
                comp.get('template_image') is not None
                and comp.get('target_image') is not None
            )
            # Capture the comp dict in default arg so each button keeps its own.
            tune_btn.clicked.connect(lambda _checked, c=comp: self._open_tune_dialog(c))
            header_row.addWidget(tune_btn)

            header_widget = QWidget()
            header_widget.setLayout(header_row)
            self.compare_layout.addWidget(header_widget)

            img_label = QLabel()
            img_label.setPixmap(pix)
            img_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            self.compare_layout.addWidget(img_label)

        self.compare_layout.addStretch(1)
        self.compare_header.setVisible(True)
        self.compare_scroll.setVisible(True)

    def _open_tune_dialog(self, comp):
        """Open a modeless tuning dialog for a single (template, target) pair."""
        tmpl = comp.get('template_image')
        tgt = comp.get('target_image')
        if tmpl is None or tgt is None:
            return
        try:
            from compare_tune_dialog import CompareTuneDialog
        except ImportError as e:
            print(f"Cannot open tune dialog: {e}")
            return
        json_path = self.matcher.json_path if self.matcher else None
        dlg = CompareTuneDialog(
            tmpl_img=tmpl, tgt_img=tgt,
            region_type=comp['type'], region_idx=comp['bbox_idx'],
            parent=self,
            annotations_json_path=json_path,
        )
        # Modeless so user can keep multiple regions open side-by-side
        dlg.setModal(False)
        dlg.show()
        # Keep a reference so it isn't GC'd while open
        if not hasattr(self, '_open_tune_dialogs'):
            self._open_tune_dialogs = []
        self._open_tune_dialogs.append(dlg)
        dlg.finished.connect(lambda _: self._open_tune_dialogs.remove(dlg)
                             if dlg in self._open_tune_dialogs else None)

    def on_inference_error(self, error_msg):
        """Handle inference error"""
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)

        self.results_text.setText(f"❌ Inference Failed\n\n{error_msg}")
        self.status_label.setText(f"❌ Error: {error_msg}")
        self._render_char_comparisons([])
    
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
                    'ocr_backend_key': result.get('ocr_backend_key'),
                    'ocr_backend_used': result.get('ocr_backend_used'),
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
            current_backend = self.ocr_backend_combo.currentData() or 'auto'
            for filename, result_data in save_data.items():
                saved_backend = result_data.get('ocr_backend_key')
                if saved_backend is not None and saved_backend != current_backend:
                    continue
                if saved_backend is None and current_backend != 'auto':
                    continue
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
