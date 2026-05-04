"""
Pipeline TensorRT: OBB Bottle Detect → Crop → Wrinkled Label Segment
Giống test_obb_then_segment.py nhưng dùng TensorRT cho cả 2 model.

Model 1 (OBB)  : best_bottle_obb_m_320.engine  (YOLOOBBTensorRT)
Model 2 (Seg)  : best_wrinkled_instance_segmentation_crop_bottle.engine
"""

import sys
import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_services.yolo_obb_tensorrt import YOLOOBBTensorRT


# ──────────────────────────────────────────────
# Helper dùng lại: crop chai theo OBB
# ──────────────────────────────────────────────
def crop_bottle_from_obb(image, cx, cy, w, h, angle, padding=10):
    img_h, img_w = image.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), np.degrees(angle), 1.0)
    rotated = cv2.warpAffine(image, M, (img_w, img_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    x1 = max(0, int(cx - w / 2) - padding)
    y1 = max(0, int(cy - h / 2) - padding)
    x2 = min(img_w, int(cx + w / 2) + padding)
    y2 = min(img_h, int(cy + h / 2) + padding)
    return rotated[y1:y2, x1:x2]


# ──────────────────────────────────────────────
# Wrinkled Segmenter — TensorRT
# ──────────────────────────────────────────────
class WrinkledSegmenterTRT:
    """
    YOLO Instance Segmentation via TensorRT (dynamic batch).
    Engine build command:
      trtexec --onnx=best_wrinkled_instance_segmentation_crop_bottle.onnx
              --saveEngine=...engine --fp16
              --minShapes=images:1x3x320x320
              --optShapes=images:4x3x320x320
              --maxShapes=images:8x3x320x320
    """

    def __init__(self, engine_path: str, class_names=None):
        self.class_names = class_names or ['label', 'wrinkled']
        trt_logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream   = cuda.Stream()

        # Lấy tên tất cả tensor (input + outputs)
        n = self.engine.num_io_tensors
        names = [self.engine.get_tensor_name(i) for i in range(n)]
        self.input_name = names[0]
        self.output_names = names[1:]  # det, proto

        # Kích thước input từ profile
        profile    = self.engine.get_tensor_profile_shape(self.input_name, 0)
        max_shape  = profile[2]           # (max_batch, 3, H, W)
        self.max_batch = max_shape[0]
        self.img_size  = max_shape[2]     # H (= W = 320)

        # Query output shapes với batch=1 để biết kích thước cố định
        self.context.set_input_shape(self.input_name, (1, 3, self.img_size, self.img_size))
        self._out_shapes = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))  # (1, N, C) hoặc (1, 32, mh, mw)
            self._out_shapes[name] = shape
            print(f"[SEG-TRT]   {name}: {shape}")

        # Cấp phát bộ nhớ
        self._alloc()
        print(f"[SEG-TRT] Loaded: {engine_path}  img_size={self.img_size}")

    def _alloc(self):
        s, mb = self.img_size, self.max_batch
        self.h_input = cuda.pagelocked_empty(mb * 3 * s * s, np.float32)
        self.d_input  = cuda.mem_alloc(self.h_input.nbytes)

        self.h_outs, self.d_outs = {}, {}
        for name, shape in self._out_shapes.items():
            # Nhân max_batch / 1 để đủ dung lượng cho batch lớn hơn
            size = int(np.prod(shape)) * self.max_batch
            h = cuda.pagelocked_empty(size, np.float32)
            d = cuda.mem_alloc(h.nbytes)
            self.h_outs[name] = h
            self.d_outs[name] = d

    # ── Preprocess ──────────────────────────────
    def _preprocess(self, image: np.ndarray):
        s = self.img_size
        h, w = image.shape[:2]
        scale = min(s / w, s / h)
        new_w, new_h = int(w * scale), int(h * scale)

        padded = np.full((s, s, 3), 114, dtype=np.uint8)
        pad_top, pad_left = (s - new_h) // 2, (s - new_w) // 2
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = cv2.resize(image, (new_w, new_h))

        tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor, scale, (pad_top, pad_left)

    # ── Predict ─────────────────────────────────
    def predict(self, image: np.ndarray, conf_threshold=0.3, return_timing=False):
        import time
        s = self.img_size
        timing = {}

        t0 = time.perf_counter()
        tensor, scale, (pad_top, pad_left) = self._preprocess(image)
        timing['preprocess'] = (time.perf_counter() - t0) * 1000

        # Set shape + H2D
        t0 = time.perf_counter()
        self.context.set_input_shape(self.input_name, (1, 3, s, s))
        np.copyto(self.h_input[:tensor.size], tensor.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        for name in self.output_names:
            self.context.set_tensor_address(name, int(self.d_outs[name]))
        self.stream.synchronize()
        timing['h2d'] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        timing['inference'] = (time.perf_counter() - t0) * 1000

        # D2H
        t0 = time.perf_counter()
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.h_outs[name], self.d_outs[name], self.stream)
        self.stream.synchronize()
        timing['d2h'] = (time.perf_counter() - t0) * 1000

        # Parse outputs
        t0 = time.perf_counter()
        det_name   = self.output_names[0]
        proto_name = self.output_names[1] if len(self.output_names) > 1 else None

        det_shape = tuple(self.context.get_tensor_shape(det_name))     # (1, N, 38)
        boxes_raw = self.h_outs[det_name][:int(np.prod(det_shape))].reshape(det_shape[1:])  # (N, 38)

        protos = None
        if proto_name:
            p_shape = tuple(self.context.get_tensor_shape(proto_name))  # (1, 32, mh, mw)
            protos  = self.h_outs[proto_name][:int(np.prod(p_shape))].reshape(p_shape[1:])  # (32, mh, mw)

        # Filter confidence — format: [x1, y1, x2, y2, conf, cls_id, coeff×32]
        valid = boxes_raw[:, 4] >= conf_threshold
        boxes = boxes_raw[valid]
        if len(boxes) == 0:
            timing['postprocess'] = (time.perf_counter() - t0) * 1000
            timing['total'] = sum(timing.values())
            return (np.array([]), None, timing) if return_timing else (np.array([]), None)

        box_coords  = boxes[:, :6].copy()
        mask_coeffs = boxes[:, 6:]

        masks = None
        if protos is not None:
            masks = self._process_masks(protos, mask_coeffs, box_coords,
                                        image.shape, scale, (pad_top, pad_left))

        # Scale về tọa độ ảnh gốc
        box_coords[:, 0] = (box_coords[:, 0] - pad_left) / scale
        box_coords[:, 1] = (box_coords[:, 1] - pad_top)  / scale
        box_coords[:, 2] = (box_coords[:, 2] - pad_left) / scale
        box_coords[:, 3] = (box_coords[:, 3] - pad_top)  / scale

        # xyxy → cxcywh
        x1, y1, x2, y2 = box_coords[:, 0], box_coords[:, 1], box_coords[:, 2], box_coords[:, 3]
        box_coords[:, 0] = (x1 + x2) / 2
        box_coords[:, 1] = (y1 + y2) / 2
        box_coords[:, 2] = x2 - x1
        box_coords[:, 3] = y2 - y1

        timing['postprocess'] = (time.perf_counter() - t0) * 1000
        timing['total'] = sum(timing.values())

        return (box_coords, masks, timing) if return_timing else (box_coords, masks)

    # ── Predict Batch ────────────────────────────
    def predict_batch(self, images: list, conf_threshold=0.3, return_timing=False):
        """
        Predict trên nhiều ảnh cùng lúc (batch inference).
        Returns: list of (boxes, masks) — 1 phần tử cho mỗi ảnh.
        """
        import time
        timing = {}
        batch_size = len(images)
        if batch_size > self.max_batch:
            raise ValueError(f"Batch {batch_size} > max_batch {self.max_batch}")

        s = self.img_size

        # Preprocess tất cả ảnh
        t0 = time.perf_counter()
        tensors, scales, pads = [], [], []
        for img in images:
            t, sc, pd = self._preprocess(img)
            tensors.append(t); scales.append(sc); pads.append(pd)
        batch_input = np.stack(tensors)          # (B, 3, H, W)
        timing['preprocess'] = (time.perf_counter() - t0) * 1000

        # H2D
        t0 = time.perf_counter()
        self.context.set_input_shape(self.input_name, (batch_size, 3, s, s))
        np.copyto(self.h_input[:batch_input.size], batch_input.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        for name in self.output_names:
            self.context.set_tensor_address(name, int(self.d_outs[name]))
        self.stream.synchronize()
        timing['h2d'] = (time.perf_counter() - t0) * 1000

        # Inference
        t0 = time.perf_counter()
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        timing['inference'] = (time.perf_counter() - t0) * 1000

        # D2H
        t0 = time.perf_counter()
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.h_outs[name], self.d_outs[name], self.stream)
        self.stream.synchronize()
        timing['d2h'] = (time.perf_counter() - t0) * 1000

        # Parse + postprocess từng ảnh
        t0 = time.perf_counter()
        det_name   = self.output_names[0]
        proto_name = self.output_names[1] if len(self.output_names) > 1 else None

        det_shape = tuple(self.context.get_tensor_shape(det_name))      # (B, N, 38)
        det_out   = self.h_outs[det_name][:int(np.prod(det_shape))].reshape(det_shape)

        protos_out = None
        if proto_name:
            p_shape    = tuple(self.context.get_tensor_shape(proto_name))  # (B, 32, mh, mw)
            protos_out = self.h_outs[proto_name][:int(np.prod(p_shape))].reshape(p_shape)

        results = []
        for i in range(batch_size):
            boxes_raw = det_out[i]
            valid = boxes_raw[:, 4] >= conf_threshold
            boxes = boxes_raw[valid]

            if len(boxes) == 0:
                results.append((np.array([]), None))
                continue

            box_coords  = boxes[:, :6].copy()
            mask_coeffs = boxes[:, 6:]
            protos      = protos_out[i] if protos_out is not None else None

            masks = None
            if protos is not None:
                masks = self._process_masks(protos, mask_coeffs, box_coords,
                                            images[i].shape, scales[i], pads[i])

            pad_top, pad_left = pads[i]
            sc = scales[i]
            box_coords[:, 0] = (box_coords[:, 0] - pad_left) / sc
            box_coords[:, 1] = (box_coords[:, 1] - pad_top)  / sc
            box_coords[:, 2] = (box_coords[:, 2] - pad_left) / sc
            box_coords[:, 3] = (box_coords[:, 3] - pad_top)  / sc

            x1, y1, x2, y2 = box_coords[:, 0], box_coords[:, 1], box_coords[:, 2], box_coords[:, 3]
            box_coords[:, 0] = (x1 + x2) / 2
            box_coords[:, 1] = (y1 + y2) / 2
            box_coords[:, 2] = x2 - x1
            box_coords[:, 3] = y2 - y1

            results.append((box_coords, masks))

        timing['postprocess'] = (time.perf_counter() - t0) * 1000
        timing['total']     = sum(timing.values())
        timing['per_image'] = timing['total'] / batch_size

        return (results, timing) if return_timing else results

    def _process_masks(self, protos, mask_coeffs, boxes_xyxy_s, orig_shape, scale, pad):
        pad_top, pad_left = pad
        s = self.img_size
        c, mh, mw = protos.shape
        h_orig, w_orig = orig_shape[:2]

        # sigmoid trên proto space (nhỏ, nhanh)
        masks = 1 / (1 + np.exp(-np.matmul(mask_coeffs, protos.reshape(c, -1)).reshape(-1, mh, mw)))

        # Tính vùng unpad trong proto space (tránh resize trung gian 320×320)
        # proto : (mh, mw) tương ứng img_size × img_size
        ratio = mh / s  # thường = 0.25 (80/320)
        pt = int(pad_top  * ratio)
        pl = int(pad_left * ratio)
        new_mh = int(h_orig * scale * ratio)
        new_mw = int(w_orig * scale * ratio)
        masks_unpad = masks[:, pt:pt + new_mh, pl:pl + new_mw]  # crop trong proto space

        # Threshold sớm trong proto space (nhỏ) → uint8 → resize NEAREST (2× nhanh hơn LINEAR)
        masks_unpad_bin = (masks_unpad > 0.5).astype(np.uint8)
        masks_orig = np.array([
            cv2.resize(m, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            for m in masks_unpad_bin
        ]).astype(np.float32)

        # Crop mask theo bbox
        for i, b in enumerate(boxes_xyxy_s):
            x1 = max(0, int((b[0] - pad_left) / scale))
            y1 = max(0, int((b[1] - pad_top)  / scale))
            x2 = min(w_orig, int((b[2] - pad_left) / scale))
            y2 = min(h_orig, int((b[3] - pad_top)  / scale))
            masks_orig[i, :y1, :] = 0; masks_orig[i, y2:, :] = 0
            masks_orig[i, :, :x1] = 0; masks_orig[i, :, x2:] = 0

        return masks_orig

    def draw(self, image, boxes, masks):
        img = image.copy()
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]

        if masks is not None and len(masks):
            overlay = np.zeros_like(img)
            for box, mask in zip(boxes, masks):
                overlay[mask > 0] = colors[int(box[5]) % len(colors)]
            img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        crop_area = image.shape[0] * image.shape[1]
        for i, box in enumerate(boxes):
            cx, cy, bw, bh, conf, cls_id = box
            cls_id = int(cls_id)
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            color  = colors[cls_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            area = int(np.sum(masks[i] > 0)) if masks is not None else int(bw * bh)
            name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            cv2.putText(img, f"{name} {conf:.2f} {area}px({area/crop_area*100:.1f}%)",
                        (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return img


# ──────────────────────────────────────────────
# Pipeline chính
# ──────────────────────────────────────────────
def run_pipeline(
    image_path: str,
    obb_engine:  str = "weights/best_bottle_obb_m_320.engine",
    seg_engine:  str = "weights/best_wrinkled_instance_segmentation_crop_bottle.engine",
    obb_conf:    float = 0.5,
    seg_conf:    float = 0.3,
    crop_padding: int  = 15,
    show_plot:   bool  = True,
    save_path:   str   = None,
):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    obb_detector = YOLOOBBTensorRT(obb_engine, class_names=['product', 'label'])
    seg_model    = WrinkledSegmenterTRT(seg_engine, class_names=['wrinkled'])

    # ── Step 1: OBB detect ──
    results_obb = obb_detector.predict([image], conf_threshold=obb_conf)
    all_boxes, all_scores, all_class_ids = results_obb[0]
    print(f"[OBB] Phát hiện {len(all_boxes)} object (conf >= {obb_conf})")

    annotated_orig = obb_detector.draw_boxes(image, all_boxes, all_scores, all_class_ids)

    # Ưu tiên class 'product', fallback sang 'label'
    PROD, LABL = obb_detector.class_names.index('product'), obb_detector.class_names.index('label')
    prod_mask = all_class_ids == PROD
    if prod_mask.any():
        crop_boxes, crop_scores, crop_cls = all_boxes[prod_mask], all_scores[prod_mask], all_class_ids[prod_mask]
        src = "product"
    else:
        lbl_mask = all_class_ids == LABL
        crop_boxes, crop_scores, crop_cls = all_boxes[lbl_mask], all_scores[lbl_mask], all_class_ids[lbl_mask]
        src = "label (fallback)"

    print(f"[OBB] Crop từ {src}: {len(crop_boxes)} vùng")

    # ── Step 2: Crop + Segment từng chai ──
    results = []
    crops_vis = []

    for idx, (box, score, _cls) in enumerate(zip(crop_boxes, crop_scores, crop_cls)):
        cx, cy, w, h, angle = box
        print(f"\n  Chai #{idx}: cx={cx:.1f} cy={cy:.1f} w={w:.1f} h={h:.1f} "
              f"angle={np.degrees(angle):.1f}° conf={score:.3f}")

        crop = crop_bottle_from_obb(image, cx, cy, w, h, angle, padding=crop_padding)
        if crop.size == 0:
            print("    [WARN] Crop rỗng, bỏ qua!")
            continue

        seg_boxes, seg_masks, seg_timing = seg_model.predict(crop, conf_threshold=seg_conf, return_timing=True)
        print(f"    [SEG] Phát hiện {len(seg_boxes)} vùng  "
              f"| pre={seg_timing['preprocess']:.1f}ms  "
              f"h2d={seg_timing['h2d']:.1f}ms  "
              f"infer={seg_timing['inference']:.1f}ms  "
              f"d2h={seg_timing['d2h']:.1f}ms  "
              f"post={seg_timing['postprocess']:.1f}ms  "
              f"total={seg_timing['total']:.1f}ms")

        has_wrinkle, total_wrinkle_area, crop_area = False, 0, crop.shape[0] * crop.shape[1]
        for i, b in enumerate(seg_boxes):
            name = seg_model.class_names[int(b[5])] if int(b[5]) < len(seg_model.class_names) else str(int(b[5]))
            area = int(np.sum(seg_masks[i] > 0)) if seg_masks is not None else 0
            print(f"      - {name}: conf={b[4]:.3f}  area={area}px ({area/crop_area*100:.1f}%)")
            if 'wrinkl' in name.lower():
                has_wrinkle = True
                total_wrinkle_area += area

        wrinkle_pct = total_wrinkle_area / crop_area * 100 if crop_area else 0
        print(f"    → {'CÓ NHĂN' if has_wrinkle else 'KHÔNG NHĂN'}"
              + (f"  {total_wrinkle_area}px ({wrinkle_pct:.1f}%)" if has_wrinkle else ""))

        crop_vis = seg_model.draw(crop, seg_boxes, seg_masks) if len(seg_boxes) else crop.copy()
        status_color = (0, 0, 255) if has_wrinkle else (0, 255, 0)
        header = f"#{idx} {'FAIL' if has_wrinkle else 'OK'}"
        if has_wrinkle:
            header += f"  {total_wrinkle_area}px ({wrinkle_pct:.1f}%)"
        cv2.putText(crop_vis, header, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        crops_vis.append(crop_vis)
        results.append({
            'bottle_idx' : idx,
            'obb_box'    : (cx, cy, w, h, angle),
            'obb_score'  : score,
            'crop'       : crop,
            'seg_boxes'  : seg_boxes,
            'seg_masks'  : seg_masks,
            'has_wrinkle': has_wrinkle,
        })

    # ── Step 3: Visualize ──
    if show_plot or save_path:
        _visualize(image_path, annotated_orig, crops_vis, results,
                   show_plot=show_plot, save_path=save_path)
    return results


def _visualize(image_path, annotated_orig, crops_vis, results, show_plot=True, save_path=None):
    n = max(1, len(crops_vis))
    fig, axes = plt.subplots(2, n, figsize=(max(12, n * 4), 8))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    axes[0, 0].imshow(cv2.cvtColor(annotated_orig, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Original + OBB ({len(results)} chai)", fontsize=11)
    axes[0, 0].axis('off')
    for j in range(1, n):
        axes[0, j].axis('off')

    for j, (cv, res) in enumerate(zip(crops_vis, results)):
        axes[1, j].imshow(cv2.cvtColor(cv, cv2.COLOR_BGR2RGB))
        ok = not res['has_wrinkle']
        axes[1, j].set_title(f"Chai #{res['bottle_idx']} — {'OK' if ok else 'FAIL'}",
                              color='green' if ok else 'red', fontsize=10)
        axes[1, j].axis('off')
    for j in range(len(crops_vis), n):
        axes[1, j].axis('off')

    plt.suptitle(f"TRT: OBB → Crop → Wrinkle Seg\n{image_path}", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    if show_plot:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    OBB_ENGINE = "./weights/best_bottle_m_320_new.engine"
    SEG_ENGINE = "./weights/best_wrinkled_instance_segmentation_crop_bottle_m.engine"

    if len(sys.argv) >= 3:
        # Batch: <input_folder> <output_folder>
        from tests.test_obb_then_segment import run_folder  # reuse batch logic nếu cần
        print("Batch mode chưa implement trong TRT version — dùng test_obb_then_segment.py")
    else:
        test_image = sys.argv[1] if len(sys.argv) > 1 else "test3.jpg"
        results = run_pipeline(
            image_path   = test_image,
            obb_engine   = OBB_ENGINE,
            seg_engine   = SEG_ENGINE,
            obb_conf     = 0.5,
            seg_conf     = 0.3,
            crop_padding = 15,
            show_plot    = True,
            save_path    = "output_pipeline_trt.png",
        )
        n_fail = sum(r['has_wrinkle'] for r in results)
        print(f"\n{'='*40}")
        print(f"Tổng số chai : {len(results)}")
        print(f"FAIL (nhăn)  : {n_fail}")
        print(f"OK           : {len(results) - n_fail}")
        print(f"{'='*40}")
