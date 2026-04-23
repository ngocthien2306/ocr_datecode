"""
ML Classifier Service

Loads sklearn classifiers trained via the ML Training Studio and
predicts OK/NG on per-character crops (each bbox = 1 character).

Feature extraction MUST match backend/app/services/ml_training_service.py
(1120-dim: 32x32 pixels + Sobel histograms + H/V projections).
"""

import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

import cv2
import joblib
import numpy as np

logger = logging.getLogger(__name__)

FEAT_SIZE = (32, 32)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def extract_features(char_img: np.ndarray) -> np.ndarray:
    """
    Port of backend's extract_features — MUST stay in sync.
    1120-dim: 32x32 pixels (1024) + Sobel Gx/Gy hists (32) + H/V projections (64).
    """
    gray = _to_gray(char_img)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(1120, dtype=np.float32)

    scale = min(FEAT_SIZE[0] / w, FEAT_SIZE[1] / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(FEAT_SIZE[::-1], dtype=np.uint8)
    yo = (FEAT_SIZE[1] - nh) // 2
    xo = (FEAT_SIZE[0] - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized

    pixels = canvas.astype(np.float32).flatten() / 255.0

    gx = cv2.Sobel(canvas, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(canvas, cv2.CV_32F, 0, 1, ksize=3)
    hist_gx = np.histogram(gx, bins=16, range=(-255, 255))[0].astype(np.float32)
    hist_gy = np.histogram(gy, bins=16, range=(-255, 255))[0].astype(np.float32)
    hist_gx /= hist_gx.sum() + 1e-6
    hist_gy /= hist_gy.sum() + 1e-6

    h_proj = canvas.astype(np.float32).sum(axis=1) / (FEAT_SIZE[0] * 255.0 + 1e-6)
    v_proj = canvas.astype(np.float32).sum(axis=0) / (FEAT_SIZE[1] * 255.0 + 1e-6)

    return np.concatenate([pixels, hist_gx, hist_gy, h_proj, v_proj])


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

    def load_model(self, project_id: str, model_id: str) -> Optional[Any]:
        """Load classifier from disk (cached). Returns None if missing."""
        key = (project_id, model_id)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

            path = self._model_path(project_id, model_id)
            if not path.exists():
                logger.error(f"ML model not found: {path}")
                return None

            try:
                clf = joblib.load(str(path))
                self._cache[key] = clf
                logger.info(
                    f"ML model loaded: project={project_id}, model={model_id}, "
                    f"type={type(clf).__name__}"
                )
                return clf
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
        Predict OK/NG on a single-character region crop.

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
            clf = self.load_model(project_id, model_id)
            if clf is None:
                result['error'] = 'model_not_found'
                return result

            if region_img is None or region_img.size == 0:
                result['error'] = 'empty_region'
                return result

            feat = extract_features(region_img).reshape(1, -1)
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
