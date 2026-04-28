"""
Embedding-based OK/NG classifier using ONNX model.

Uses cosine similarity between template crop and target crop embeddings.
Supports projection and arcface head types only.

p_ok = (cosine_sim + 1) / 2  →  identical=1.0, opposite=0.0

Template embeddings are pre-computed once at recipe load via
cache_template_embeddings() and reused across all frames. classify_batch()
only runs ONNX on target crops.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = img
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def _extract_embeddings(
    sess: ort.InferenceSession,
    head: str,
    input_name: str,
    tensors: np.ndarray,
) -> np.ndarray:
    """Run ONNX inference and return L2-normalized embeddings."""
    if head == "projection":
        emb = sess.run(None, {input_name: tensors})[0]
    else:  # arcface
        emb = sess.run(None, {input_name: tensors})[1]
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb / norms


class EmbeddingClassifierService:
    """
    ONNX-based per-character OK/NG classifier using cosine similarity.

    Workflow:
      1. Call cache_template_embeddings() once when recipe is loaded.
      2. Call classify_batch() each frame — only runs ONNX on target crops,
         template embeddings are served from cache.
      3. Call clear_template_cache() when recipe is reloaded.
    """

    def __init__(
        self,
        onnx_path: str,
        config_path: str,
        save_debug_images: bool = True,
        debug_path: Optional[str] = None,
    ):
        cfg = OmegaConf.load(config_path)
        self.head = cfg.model.head.type
        self.size = int(cfg.data.image_size)

        if self.head not in ("projection", "arcface"):
            raise ValueError(
                f"EmbeddingClassifierService requires projection/arcface head, got: {self.head}"
            )

        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.sess.get_inputs()[0].name

        self.save_debug_images = save_debug_images
        self.debug_path = debug_path or f"{os.environ.get('HOME')}/Source/ocr_datecode/ai_services/test_result"
        if self.save_debug_images:
            os.makedirs(self.debug_path, exist_ok=True)

        # (serial_number, annotation_idx) → L2-normalized embedding (D,)
        self._template_emb_cache: Dict[Tuple[str, int], np.ndarray] = {}

        logger.info(
            f"EmbeddingClassifierService: head={self.head}, size={self.size}, "
            f"provider={self.sess.get_providers()[0]}, onnx={onnx_path}"
        )

    # ── Template cache management ──

    def cache_template_embeddings(
        self,
        serial_number: str,
        items: List[Dict[str, Any]],
    ) -> int:
        """
        Pre-compute and cache template embeddings for a camera's char bboxes.

        Call once when a recipe is loaded for a camera. Skips items already
        in cache (idempotent).

        items shape: [{'annotation_idx': int, 'template_crop': np.ndarray}, ...]

        Returns number of new embeddings cached.
        """
        crops: List[np.ndarray] = []
        keys: List[Tuple[str, int]] = []

        for item in items:
            ann_idx = item.get('annotation_idx')
            template = item.get('template_crop')
            if ann_idx is None or template is None or template.size == 0:
                continue
            key = (serial_number, ann_idx)
            if key in self._template_emb_cache:
                continue
            try:
                crops.append(_preprocess(template, self.size))
                keys.append(key)
            except Exception as e:
                logger.warning(f"[{serial_number}] Failed to preprocess template ann {ann_idx}: {e}")

        if not crops:
            return 0

        tensors = np.stack(crops)
        embs = _extract_embeddings(self.sess, self.head, self.input_name, tensors)
        for key, emb in zip(keys, embs):
            self._template_emb_cache[key] = emb

        logger.info(f"Cached {len(keys)} template embeddings for {serial_number}")
        return len(keys)

    def clear_template_cache(self, serial_number: Optional[str] = None) -> None:
        """
        Invalidate template embedding cache.
        Call when recipe is reloaded. Pass serial_number to clear only one
        camera, or None to clear all.
        """
        if serial_number is None:
            count = len(self._template_emb_cache)
            self._template_emb_cache.clear()
            logger.info(f"Template embedding cache cleared ({count} entries)")
        else:
            keys = [k for k in self._template_emb_cache if k[0] == serial_number]
            for k in keys:
                del self._template_emb_cache[k]
            logger.info(f"Template embedding cache cleared for {serial_number} ({len(keys)} entries)")

    # ── Per-frame classification ──

    def classify_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Batch classify N character crops via embedding cosine similarity.

        Template embeddings are served from cache (set via
        cache_template_embeddings). If a cache miss occurs (e.g. first frame
        before cache_template_embeddings was called), falls back to computing
        from item['template_crop'] and caches the result.

        Input item shape:
            {
                'region_img':     np.ndarray  — target crop (current frame)
                'template_crop':  np.ndarray  — fallback if cache miss (optional)
                'conf_threshold': float
                'serial_number':  str
                'annotation_idx': int
            }

        Returns list parallel to items:
            {'ml_pass', 'p_ok', 'label', 'threshold', 'time_ms', 'error'}
        """
        n = len(items)
        if n == 0:
            return []

        t0 = time.perf_counter()
        results: List[Optional[Dict[str, Any]]] = [None] * n

        # ── Step 1: preprocess targets; resolve template embs from cache ──
        target_tensors: List[np.ndarray] = []
        template_embs_resolved: List[Optional[np.ndarray]] = []  # per valid item
        miss_crops: List[np.ndarray] = []   # template crops for cache misses
        miss_keys: List[Tuple[str, int]] = []
        miss_positions: List[int] = []      # index into target_tensors list
        valid_idxs: List[int] = []

        for i, item in enumerate(items):
            region = item.get('region_img')
            conf_thr = float(item.get('conf_threshold', 0.5))
            serial = item.get('serial_number', '')
            ann_idx = item.get('annotation_idx', -1)

            if region is None or region.size == 0:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0, 'error': 'empty_region',
                }
                continue

            try:
                target_tensors.append(_preprocess(region, self.size))
            except Exception as e:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0,
                    'error': f'preprocess_failed:{e}',
                }
                continue

            pos = len(target_tensors) - 1
            cache_key = (serial, ann_idx)
            cached = self._template_emb_cache.get(cache_key)

            if cached is not None:
                template_embs_resolved.append(cached)
            else:
                # Cache miss — will compute from template_crop below
                template = item.get('template_crop')
                if template is None or template.size == 0:
                    results[i] = {
                        'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                        'threshold': conf_thr, 'time_ms': 0.0,
                        'error': 'template_cache_miss_and_no_crop',
                    }
                    # Roll back the target tensor we just added
                    target_tensors.pop()
                    continue
                try:
                    miss_crops.append(_preprocess(template, self.size))
                    miss_keys.append(cache_key)
                    miss_positions.append(pos)
                    template_embs_resolved.append(None)  # placeholder
                except Exception as e:
                    results[i] = {
                        'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                        'threshold': conf_thr, 'time_ms': 0.0,
                        'error': f'template_preprocess_failed:{e}',
                    }
                    target_tensors.pop()
                    continue

            valid_idxs.append(i)

        if not valid_idxs:
            for i, r in enumerate(results):
                if r is None:
                    results[i] = {
                        'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                        'threshold': float(items[i].get('conf_threshold', 0.5)),
                        'time_ms': 0.0, 'error': 'all_invalid',
                    }
            return results  # type: ignore

        # ── Step 2: ONNX on targets (always) + cache-miss templates (first frame only) ──
        m = len(valid_idxs)
        try:
            if miss_crops:
                # Combine targets + miss templates in one ONNX call
                combined = np.stack(target_tensors + miss_crops)
                all_embs = _extract_embeddings(self.sess, self.head, self.input_name, combined)
                target_embs = all_embs[:m]
                miss_embs   = all_embs[m:]
                # Cache newly computed template embeddings
                for key, emb in zip(miss_keys, miss_embs):
                    self._template_emb_cache[key] = emb
                    logger.debug(f"Cached template emb on-the-fly for {key}")
                # Fill placeholders in template_embs_resolved
                miss_iter = iter(miss_embs)
                for pos in miss_positions:
                    template_embs_resolved[pos] = next(miss_iter)
            else:
                # All templates cached — only run ONNX on targets
                target_embs = _extract_embeddings(
                    self.sess, self.head, self.input_name, np.stack(target_tensors)
                )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.error(f"EmbeddingClassifier ONNX inference failed: {e}")
            for i in valid_idxs:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': float(items[i].get('conf_threshold', 0.5)),
                    'time_ms': elapsed, 'error': f'inference_failed:{e}',
                }
            for i, r in enumerate(results):
                if r is None:
                    results[i] = {
                        'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                        'threshold': float(items[i].get('conf_threshold', 0.5)),
                        'time_ms': elapsed, 'error': 'inference_failed',
                    }
            return results  # type: ignore

        # ── Step 3: cosine similarity + threshold ──
        elapsed = (time.perf_counter() - t0) * 1000
        time_per_item = round(elapsed / m, 2)

        for row, i in enumerate(valid_idxs):
            item = items[i]
            tmpl_emb = template_embs_resolved[row]
            sim = float(np.dot(target_embs[row], tmpl_emb))
            p_ok = (sim + 1.0) / 2.0   # map [-1,1] → [0,1]
            conf_thr = float(item.get('conf_threshold', 0.5))
            label = "OK" if p_ok >= conf_thr else "NG"
            results[i] = {
                'ml_pass': (label == "OK"),
                'p_ok': round(p_ok, 4),
                'label': label,
                'threshold': conf_thr,
                'time_ms': time_per_item,
                'error': None,
            }

            if self.save_debug_images:
                try:
                    ts = int(time.time())
                    fname = (
                        f"emb_classify_{item.get('serial_number', '')}_"
                        f"{item.get('annotation_idx', -1)}_{label}_{p_ok:.2f}_{ts}.png"
                    )
                    cv2.imwrite(os.path.join(self.debug_path, fname), item['region_img'])
                except Exception:
                    pass

            logger.debug(
                f"[{item.get('serial_number', '')}] emb ann "
                f"{item.get('annotation_idx', -1)}: sim={sim:.4f} p_ok={p_ok:.4f} "
                f"{label} thr={conf_thr}"
            )

        cache_status = f"(+{len(miss_crops)} new)" if miss_crops else "(all cached)"
        logger.info(
            f"Embedding classify batch: N={m} {cache_status}, "
            f"elapsed={elapsed:.1f}ms ({time_per_item:.2f}ms/item)"
        )

        for i, r in enumerate(results):
            if r is None:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': float(items[i].get('conf_threshold', 0.5)),
                    'time_ms': 0.0, 'error': 'unknown',
                }

        return results  # type: ignore
