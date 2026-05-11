"""
ML Classifier Service

Loads sklearn classifiers trained via the ML Training Studio and
predicts OK/NG on per-character crops (each bbox = 1 character).

Feature extraction: SupCon embedding (128-dim L2-normalized) via shared ONNX
weights at weights/supcon_128_efficientnet_b2_*. MUST stay in sync with
backend/app/services/ml_training_service.py.
"""

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ── SupCon embedding ──────────────────────────────────────────────────────
EMBED_DIM = 128
EMBED_SIZE = 64
_EMB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_EMB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# 4 levels up: verification → camera_management → ai_services → repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SUPCON_PATH = _REPO_ROOT / "weights/supcon_128_efficientnet_b2_20260429-073504"
# TensorRT engine cache — SHARED with BE training (backend/app/services/ml_training_service.py).
# Both services point _BACKEND_DIR / _REPO_ROOT to project root, so the cache dir resolves
# to the same physical path; the engine BE builds during training is reused at inference time.
_TRT_CACHE_DIR = _REPO_ROOT / "cache" / "supcon_trt"

# Explicit TRT optimization profile for SupCon (input: dynamic batch x 3 x 64 x 64).
# Must match BE config so both services hit the SAME engine file in shared cache.
#   min  =   1  → single-crop inference
#   opt  =  48  → average chars per frame (recipe-typical)
#   max  = 128  → batch ceiling; embed_crops chunks above this size
_TRT_PROFILE_MIN = "input:1x3x64x64"
_TRT_PROFILE_OPT = "input:48x3x64x64"
_TRT_PROFILE_MAX = "input:128x3x64x64"

_supcon_session = None  # singleton ONNX session


def _get_supcon_session():
    """
    Lazy-init singleton ONNX session for SupCon embedder.

    Provider preference: TensorRT → CUDA → CPU. TRT engine is cached on disk at
    `_TRT_CACHE_DIR` and shared with BE training, so the (slow) build runs only
    once per Jetson/JetPack version. Subsequent restarts load instantly.

    On Mac dev (no TRT/CUDA), falls back cleanly to CPU.
    """
    global _supcon_session
    if _supcon_session is None:
        import onnxruntime as ort
        onnx_path = _SUPCON_PATH / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"SupCon model not found at {onnx_path}")

        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = int(os.environ.get("ML_ORT_INTRA_THREADS", "4"))
        sess_options.log_severity_level = 3  # suppress dynamic-axes warnings

        available = set(ort.get_available_providers())
        providers: List[Any] = []
        cache_existed = False
        if "TensorrtExecutionProvider" in available:
            _TRT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_existed = any(_TRT_CACHE_DIR.iterdir())
            trt_fp16 = os.environ.get("ML_TRT_FP16", "1") not in ("0", "false", "False")
            providers.append((
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable":  True,
                    "trt_engine_cache_path":    str(_TRT_CACHE_DIR),
                    "trt_fp16_enable":          trt_fp16,
                    # 256 MB workspace — fits comfortably on Jetson Orin 8GB.
                    "trt_max_workspace_size":   256 * 1024 * 1024,
                    # Explicit shape profile — avoids mid-production rebuilds when
                    # batch size varies between recipes.
                    "trt_profile_min_shapes":   _TRT_PROFILE_MIN,
                    "trt_profile_opt_shapes":   _TRT_PROFILE_OPT,
                    "trt_profile_max_shapes":   _TRT_PROFILE_MAX,
                },
            ))
        if "CUDAExecutionProvider" in available:
            providers.append((
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search":    "HEURISTIC",
                    "do_copy_in_default_stream": True,
                },
            ))
        providers.append("CPUExecutionProvider")

        _supcon_session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=providers,
        )
        active = _supcon_session.get_providers()[0]
        cache_state = (
            f"cache={'HIT' if cache_existed else 'MISS'} ({_TRT_CACHE_DIR})"
            if active == "TensorrtExecutionProvider" else "no-cache"
        )
        logger.info(f"[SupCon] loaded {onnx_path.name} on {active} — {cache_state}")
    return _supcon_session


def _preprocess_for_supcon(bgr: np.ndarray) -> np.ndarray:
    """Resize keep-aspect → pad-255 to 64×64 → ImageNet normalize → CHW float32.
    Accepts grayscale (2D) or BGR (3D) input — grayscale is widened to 3 channels.
    """
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.ndim == 3 and bgr.shape[2] == 1:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((3, EMBED_SIZE, EMBED_SIZE), dtype=np.float32)
    s = EMBED_SIZE / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas_pad = np.full((EMBED_SIZE, EMBED_SIZE, 3), 255, dtype=np.uint8)
    canvas_pad[(EMBED_SIZE - nh) // 2:(EMBED_SIZE - nh) // 2 + nh,
               (EMBED_SIZE - nw) // 2:(EMBED_SIZE - nw) // 2 + nw] = img
    rgb = cv2.cvtColor(canvas_pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - _EMB_MEAN) / _EMB_STD).transpose(2, 0, 1).astype(np.float32)


def embed_crops(crops: List[np.ndarray], batch_size: int = 128) -> np.ndarray:
    """Batch-embed crops via SupCon ONNX. Returns (N, 128) L2-normalized."""
    if not crops:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    sess = _get_supcon_session()
    out_chunks: List[np.ndarray] = []
    for i in range(0, len(crops), batch_size):
        chunk = crops[i:i + batch_size]
        x = np.stack([_preprocess_for_supcon(c) for c in chunk]).astype(np.float32)
        out = sess.run(None, {"input": x})[0]
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
        out_chunks.append(out / norms)
    return np.vstack(out_chunks)


def _centroid_predict_proba(X: np.ndarray, bundle: Dict[str, Any]) -> np.ndarray:
    """Centroid scoring → p_ok ∈ [0, 1]. Mirror of BE."""
    c_ok = bundle['centroid_ok']
    c_ng = bundle['centroid_ng']
    T = float(bundle.get('temperature', 5.0))
    raw = (X @ c_ok) - (X @ c_ng)
    return 1.0 / (1.0 + np.exp(-np.clip(raw * T, -30, 30)))


class MLClassifierService:
    """
    Loads and caches sklearn classifiers from disk, predicts OK/NG
    on single-character region crops.

    Directory layout (shared filesystem with BE):
        {ml_base_dir}/{project_id}/models/{model_id}.joblib
    """

    def __init__(
        self,
        ml_base_dir: Path,
        save_debug_images: bool = True,
        debug_path: Optional[str] = None,
    ):
        self.ml_base_dir = Path(ml_base_dir)
        self.save_debug_images = save_debug_images
        self.debug_path = debug_path or f"{os.environ.get('HOME')}/Source/ocr_datecode/ai_services/test_result"

        self._cache: Dict[Tuple[str, str], Any] = {}
        self._lock = Lock()

        if self.save_debug_images:
            os.makedirs(self.debug_path, exist_ok=True)

        logger.info(
            f"MLClassifierService initialized: base_dir={self.ml_base_dir}, "
            f"save_debug={self.save_debug_images}"
        )

    def _model_path(self, project_id: str, model_id: str) -> Path:
        return self.ml_base_dir / project_id / "models" / f"{model_id}.joblib"

    def load_model(self, project_id: str, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load model bundle from disk (cached). Supports two shapes:
          - sklearn  : {'algorithm': rf|svm|mlp, 'clf': ...}
          - centroid : {'algorithm': 'centroid', 'centroid_ok', 'centroid_ng',
                        'temperature'}
        Legacy joblibs (raw classifier, v2 with goldens) are no longer supported.
        """
        key = (project_id, model_id)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

            path = self._model_path(project_id, model_id)
            if not path.exists():
                logger.error(f"ML model not found: {path}")
                return None

            try:
                data = joblib.load(str(path))
                if not isinstance(data, dict):
                    logger.error(f"Unsupported model bundle at {path}: not a dict")
                    return None
                algo = (data.get('algorithm') or '').lower()
                if algo == 'centroid':
                    if 'centroid_ok' not in data or 'centroid_ng' not in data:
                        logger.error(f"Centroid bundle at {path} missing centroid_ok/ng")
                        return None
                    bundle = {
                        'algorithm':   'centroid',
                        'centroid_ok': np.asarray(data['centroid_ok'], dtype=np.float32),
                        'centroid_ng': np.asarray(data['centroid_ng'], dtype=np.float32),
                        'temperature': float(data.get('temperature', 5.0)),
                    }
                elif 'clf' in data:
                    bundle = {'algorithm': algo or 'rf', 'clf': data['clf']}
                else:
                    logger.error(
                        f"Unsupported model bundle at {path}: "
                        "expected 'clf' (sklearn) or centroid keys."
                    )
                    return None
                self._cache[key] = bundle
                logger.info(
                    f"ML model loaded: project={project_id}, model={model_id}, "
                    f"algorithm={bundle['algorithm']}"
                )
                return bundle
            except Exception as e:
                logger.error(f"Failed to load ML model {path}: {e}")
                return None

    def clear_cache(self):
        """Drop all cached models (call on recipe reload)."""
        with self._lock:
            self._cache.clear()
        logger.info("MLClassifierService cache cleared")

    def classify_region(
        self,
        region_img: np.ndarray,
        project_id: str,
        model_id: str,
        conf_threshold: float,
        serial_number: str = "",
        annotation_idx: int = -1,
    ) -> Dict[str, Any]:
        """
        Predict OK/NG on a single-character region crop via SupCon embedding
        + sklearn classifier.

        Returns:
            {
                'ml_pass': bool,        # True if p_ok >= conf_threshold
                'p_ok': float,          # probability of OK class
                'label': 'OK' | 'NG',
                'threshold': float,
                'time_ms': float,
                'error': str | None,
            }
        """
        t0 = time.perf_counter()
        result = {
            'ml_pass': False,
            'p_ok': 0.0,
            'label': 'NG',
            'threshold': conf_threshold,
            'time_ms': 0.0,
            'error': None,
        }

        try:
            bundle = self.load_model(project_id, model_id)
            if bundle is None:
                result['error'] = 'model_not_found'
                return result

            if region_img is None or region_img.size == 0:
                result['error'] = 'empty_region'
                return result

            feat = embed_crops([region_img])     # (1, 128)
            if bundle.get('algorithm') == 'centroid':
                p_ok = float(_centroid_predict_proba(feat, bundle)[0])
            else:
                clf = bundle['clf']
                proba = clf.predict_proba(feat)[0]
                p_ok = float(proba[1]) if len(proba) > 1 else float(proba[0])
            label = "OK" if p_ok >= conf_threshold else "NG"

            result['ml_pass'] = (label == "OK")
            result['p_ok'] = round(p_ok, 4)
            result['label'] = label

            if self.save_debug_images:
                try:
                    ts = int(time.time())
                    fname = (
                        f"ml_classify_{serial_number}_{annotation_idx}_"
                        f"{label}_{p_ok:.2f}_{ts}.png"
                    )
                    cv2.imwrite(os.path.join(self.debug_path, fname), region_img)
                except Exception as e_save:
                    logger.warning(
                        f"[{serial_number}] Failed to save ML debug image ann {annotation_idx}: {e_save}"
                    )

        except Exception as e:
            logger.error(
                f"[{serial_number}] ML classify error ann {annotation_idx}: {e}"
            )
            result['error'] = str(e)

        result['time_ms'] = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[{serial_number}] ML classify ann {annotation_idx}: "
            f"label={result['label']}, p_ok={result['p_ok']:.4f}, "
            f"threshold={conf_threshold}, ml_pass={result['ml_pass']}, "
            f"time={result['time_ms']:.1f}ms"
        )
        return result

    def classify_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Batch-predict OK/NG for N region crops via SupCon embedding (single
        ONNX call per group) + sklearn classifier.

        Much faster than calling classify_region in parallel because:
          - SupCon ONNX runs all crops in one batched call
          - sklearn predict_proba(N samples) is near-O(1) overhead vs single

        Input item shape:
            {'region_img', 'project_id', 'model_id', 'conf_threshold',
             'serial_number', 'annotation_idx'}

        Returns a list parallel to `items` with one result dict each
        (same shape as classify_region's return).
        """
        n = len(items)
        results: List[Optional[Dict[str, Any]]] = [None] * n
        if n == 0:
            return []

        # Group items by (project_id, model_id) — different cameras may use
        # different models; each group gets its own predict_proba call.
        groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for i, item in enumerate(items):
            groups[(item['project_id'], item['model_id'])].append(i)

        for (pid, mid), idxs in groups.items():
            t0 = time.perf_counter()
            bundle = self.load_model(pid, mid)

            if bundle is None:
                for i in idxs:
                    item = items[i]
                    results[i] = {
                        'annotation_idx': item.get('annotation_idx'),
                        'ml_pass': False,
                        'p_ok': 0.0,
                        'label': 'NG',
                        'threshold': float(item.get('conf_threshold', 0.5)),
                        'time_ms': 0.0,
                        'error': 'model_not_found',
                    }
                continue

            # Collect valid crops in this group
            crops_list: List[np.ndarray] = []
            valid_idxs: List[int] = []
            for i in idxs:
                item = items[i]
                region = item.get('region_img')
                if region is None or region.size == 0:
                    results[i] = {
                        'annotation_idx': item.get('annotation_idx'),
                        'ml_pass': False,
                        'p_ok': 0.0,
                        'label': 'NG',
                        'threshold': float(item.get('conf_threshold', 0.5)),
                        'time_ms': 0.0,
                        'error': 'empty_region',
                    }
                    continue
                crops_list.append(region)
                valid_idxs.append(i)

            if not valid_idxs:
                continue

            # Time embed (SupCon ONNX) and predict_proba separately for diagnostics.
            # Old code measured t_feat BEFORE embed_crops → reported feat=0.0ms always.
            t_embed_start = time.perf_counter()
            X = embed_crops(crops_list)            # (M, 128) — batched SupCon ONNX
            t_embed_end = time.perf_counter()
            if bundle.get('algorithm') == 'centroid':
                p_ok_arr = _centroid_predict_proba(X, bundle)
                probas = np.stack([1.0 - p_ok_arr, p_ok_arr], axis=1)
            else:
                clf = bundle['clf']
                probas = clf.predict_proba(X)      # (M, 2) or (M, 1)
            t_pred_end = time.perf_counter()

            debug_write_count = 0
            for row, i in enumerate(valid_idxs):
                item = items[i]
                proba = probas[row]
                p_ok = float(proba[1]) if proba.shape[0] > 1 else float(proba[0])
                conf_thr = float(item.get('conf_threshold', 0.5))
                label = "OK" if p_ok >= conf_thr else "NG"
                results[i] = {
                    'annotation_idx': item.get('annotation_idx'),
                    'ml_pass': (label == "OK"),
                    'p_ok': round(p_ok, 4),
                    'label': label,
                    'threshold': conf_thr,
                    'time_ms': 0.0,  # set below
                    'error': None,
                }

                if self.save_debug_images:
                    try:
                        ts = int(time.time())
                        fname = (
                            f"ml_classify_{item.get('serial_number', '')}_"
                            f"{item.get('annotation_idx', -1)}_{label}_"
                            f"{p_ok:.2f}_{ts}.png"
                        )
                        cv2.imwrite(os.path.join(self.debug_path, fname), item['region_img'])
                        debug_write_count += 1
                    except Exception:
                        pass

            group_ms  = (time.perf_counter() - t0) * 1000
            embed_ms  = (t_embed_end - t_embed_start) * 1000
            pred_ms   = (t_pred_end  - t_embed_end)   * 1000
            setup_ms  = (t_embed_start - t0) * 1000      # crops_list build + valid_idxs filter
            debug_ms  = (time.perf_counter() - t_pred_end) * 1000   # debug image save loop
            for i in valid_idxs:
                if results[i] is not None:
                    results[i]['time_ms'] = round(group_ms / max(len(valid_idxs), 1), 2)

            logger.info(
                f"ML batch classify: project={pid}, model={mid}, algo={bundle.get('algorithm','rf')}, "
                f"N={len(valid_idxs)}, setup={setup_ms:.1f}ms, "
                f"embed={embed_ms:.1f}ms, predict={pred_ms:.1f}ms, "
                f"debug={debug_ms:.1f}ms, total={group_ms:.1f}ms"
            )

        # Fill any remaining None with fallback (shouldn't happen)
        for i, r in enumerate(results):
            if r is None:
                item = items[i]
                results[i] = {
                    'annotation_idx': item.get('annotation_idx'),
                    'ml_pass': False,
                    'p_ok': 0.0,
                    'label': 'NG',
                    'threshold': float(item.get('conf_threshold', 0.5)),
                    'time_ms': 0.0,
                    'error': 'unknown',
                }

        return results  # type: ignore
