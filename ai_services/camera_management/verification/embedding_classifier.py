"""
Embedding-based OK/NG classifier using ONNX model.

Uses cosine similarity between template crop and target crop embeddings.
Supports projection and arcface head types only.

p_ok = (cosine_sim + 1) / 2  →  identical=1.0, opposite=0.0

Template and target crops are embedded together in every classify_batch() call.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

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

    classify_batch() embeds template and target crops together in one ONNX
    call every time — no caching.
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

        logger.info(
            f"EmbeddingClassifierService: head={self.head}, size={self.size}, "
            f"provider={self.sess.get_providers()[0]}, onnx={onnx_path}"
        )

    def classify_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Batch classify N character crops via embedding cosine similarity.

        Both template and target are embedded in one ONNX call per batch.
        Each item must supply template_crop.

        Input item shape:
            {
                'region_img':     np.ndarray  — target crop (current frame)
                'template_crop':  np.ndarray  — template crop (required)
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

        tmpl_tensors: List[np.ndarray] = []
        tgt_tensors: List[np.ndarray] = []
        valid_idxs: List[int] = []

        for i, item in enumerate(items):
            region = item.get('region_img')
            template = item.get('template_crop')
            conf_thr = float(item.get('conf_threshold', 0.5))

            if region is None or region.size == 0:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0, 'error': 'empty_region',
                }
                continue

            if template is None or template.size == 0:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0, 'error': 'missing_template_crop',
                }
                continue

            try:
                tgt_tensors.append(_preprocess(region, self.size))
                tmpl_tensors.append(_preprocess(template, self.size))
            except Exception as e:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': conf_thr, 'time_ms': 0.0,
                    'error': f'preprocess_failed:{e}',
                }
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

        m = len(valid_idxs)
        try:
            # Embed template + target together in one ONNX call
            combined = np.stack(tmpl_tensors + tgt_tensors)
            all_embs = _extract_embeddings(self.sess, self.head, self.input_name, combined)
            tmpl_embs = all_embs[:m]
            tgt_embs  = all_embs[m:]
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

        elapsed = (time.perf_counter() - t0) * 1000
        time_per_item = round(elapsed / m, 2)

        for row, i in enumerate(valid_idxs):
            item = items[i]
            sim = float(np.dot(tmpl_embs[row], tgt_embs[row]))
            p_ok = (sim + 1.0) / 2.0
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

        logger.info(
            f"Embedding classify batch: N={m}, elapsed={elapsed:.1f}ms ({time_per_item:.2f}ms/item)"
        )

        for i, r in enumerate(results):
            if r is None:
                results[i] = {
                    'ml_pass': False, 'p_ok': 0.0, 'label': 'NG',
                    'threshold': float(items[i].get('conf_threshold', 0.5)),
                    'time_ms': 0.0, 'error': 'unknown',
                }

        return results  # type: ignore
