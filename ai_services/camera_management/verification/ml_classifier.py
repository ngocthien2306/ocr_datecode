"""
ML Classifier Service

Loads sklearn classifiers trained via the ML Training Studio and
predicts OK/NG on per-character crops (each bbox = 1 character).

Feature extraction MUST stay in sync with
backend/app/services/ml_training_service.py:
  - v1: 856-dim (legacy bundles without goldens)
  - v2: 1016-dim = 856 base + 160 diff-vs-golden
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

FEAT_SIZE = (48, 48)
GRID = 6
CELL_SIZE = FEAT_SIZE[0] // GRID
FEAT_DIM = 576 + 144 + 32 + 96 + 8   # v1 = 856
FEAT_DIM_V2 = FEAT_DIM + 160          # v2 = 1016


def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def extract_features(char_img: np.ndarray) -> np.ndarray:
    """
    Port of backend's extract_features — MUST stay in sync.

    856-dim feature vector:
      - 24×24 downsampled pixels (576) — global shape
      - 6×6 grid × 4 cell stats (144)  — LOCAL defect signal
      - Sobel Gx/Gy histograms (32)    — stroke orientation
      - H/V projections (96)           — row/column sums
      - 2×2 quadrant Sobel mag (8)     — coarse gradient energy
    """
    gray = _to_gray(char_img)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    scale = min(FEAT_SIZE[0] / w, FEAT_SIZE[1] / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(FEAT_SIZE[::-1], dtype=np.uint8)
    yo = (FEAT_SIZE[1] - nh) // 2
    xo = (FEAT_SIZE[0] - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized

    # 1) Global downsampled pixels (24×24 = 576)
    small = cv2.resize(canvas, (24, 24), interpolation=cv2.INTER_AREA)
    pixels = small.astype(np.float32).flatten() / 255.0

    # 2) 6×6 grid local stats (144)
    edges = cv2.Canny(canvas, 50, 150)
    cell_stats = []
    for i in range(GRID):
        for j in range(GRID):
            y0, y1 = i * CELL_SIZE, (i + 1) * CELL_SIZE
            x0, x1 = j * CELL_SIZE, (j + 1) * CELL_SIZE
            cell = canvas[y0:y1, x0:x1]
            edge_cell = edges[y0:y1, x0:x1]
            cell_stats.append(float(cell.mean()) / 255.0)
            cell_stats.append(float(cell.std()) / 128.0)
            cell_stats.append(float(np.count_nonzero(edge_cell)) / (CELL_SIZE * CELL_SIZE))
            cell_stats.append(float(cell.min()) / 255.0)
    cell_arr = np.asarray(cell_stats, dtype=np.float32)

    # 3) Sobel Gx/Gy histograms (32)
    gx = cv2.Sobel(canvas, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(canvas, cv2.CV_32F, 0, 1, ksize=3)
    hist_gx = np.histogram(gx, bins=16, range=(-255, 255))[0].astype(np.float32)
    hist_gy = np.histogram(gy, bins=16, range=(-255, 255))[0].astype(np.float32)
    hist_gx /= hist_gx.sum() + 1e-6
    hist_gy /= hist_gy.sum() + 1e-6

    # 4) H/V projections (96)
    h_proj = canvas.astype(np.float32).sum(axis=1) / (FEAT_SIZE[0] * 255.0 + 1e-6)
    v_proj = canvas.astype(np.float32).sum(axis=0) / (FEAT_SIZE[1] * 255.0 + 1e-6)

    # 5) Sobel magnitude per 2×2 quadrant (8)
    sob = np.hypot(gx, gy)
    qsize = FEAT_SIZE[0] // 2
    quad_stats = []
    for qi in range(2):
        for qj in range(2):
            q = sob[qi * qsize:(qi + 1) * qsize, qj * qsize:(qj + 1) * qsize]
            quad_stats.append(float(q.mean()) / 255.0)
            quad_stats.append(float(q.std()) / 255.0)
    quad_arr = np.asarray(quad_stats, dtype=np.float32)

    return np.concatenate([pixels, cell_arr, hist_gx, hist_gy, h_proj, v_proj, quad_arr])


# ───────────────────────── Golden template (v2) — port from BE ─────────────

def preprocess_canonical(img: np.ndarray) -> np.ndarray:
    """Gray → CLAHE → resize keep aspect → center-pad 48×48. Matches BE."""
    gray = _to_gray(img)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(FEAT_SIZE[::-1], dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normed = clahe.apply(gray)
    scale = min(FEAT_SIZE[0] / w, FEAT_SIZE[1] / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(normed, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(FEAT_SIZE[::-1], dtype=np.uint8)
    yo = (FEAT_SIZE[1] - nh) // 2
    xo = (FEAT_SIZE[0] - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def align_to_golden(input_48: np.ndarray, golden_48: np.ndarray, search: int = 5) -> np.ndarray:
    """±search-px template match → apply best shift. Returns aligned 48×48."""
    padded = cv2.copyMakeBorder(
        input_48, search, search, search, search, cv2.BORDER_REPLICATE,
    )
    result = cv2.matchTemplate(padded, golden_48, cv2.TM_CCOEFF_NORMED)
    _, _, _, (mx, my) = cv2.minMaxLoc(result)
    dx, dy = mx - search, my - search
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        input_48, M, FEAT_SIZE,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def extract_diff_features(diff_48: np.ndarray) -> np.ndarray:
    """160-dim diff features: 6×6 grid × 4 stats (144) + 4×4 region max (16)."""
    feats: List[float] = []
    # 6×6 grid × 4 stats: mean, max, std, count(>30)
    for i in range(GRID):
        for j in range(GRID):
            y0, y1 = i * CELL_SIZE, (i + 1) * CELL_SIZE
            x0, x1 = j * CELL_SIZE, (j + 1) * CELL_SIZE
            cell = diff_48[y0:y1, x0:x1]
            feats.append(float(cell.mean()) / 255.0)
            feats.append(float(cell.max()) / 255.0)
            feats.append(float(cell.std()) / 128.0)
            feats.append(float(np.count_nonzero(cell > 30)) / (CELL_SIZE * CELL_SIZE))
    # 4×4 region × 1 stat: max
    region_sz = FEAT_SIZE[0] // 4
    for i in range(4):
        for j in range(4):
            y0, y1 = i * region_sz, (i + 1) * region_sz
            x0, x1 = j * region_sz, (j + 1) * region_sz
            feats.append(float(diff_48[y0:y1, x0:x1].max()) / 255.0)
    return np.asarray(feats, dtype=np.float32)


def extract_features_v2(
    char_img: np.ndarray,
    char_id: Optional[str],
    goldens: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    1016-dim v2 features: 856 base (on aligned input) + 160 diff vs golden.
    When char_id missing or no golden → diff features = zeros (still 1016-dim).
    """
    canvas = preprocess_canonical(char_img)
    diff_feats: np.ndarray
    if char_id and goldens and char_id in goldens:
        aligned = align_to_golden(canvas, goldens[char_id])
        diff = np.abs(aligned.astype(np.int16) - goldens[char_id].astype(np.int16)).astype(np.uint8)
        base = extract_features(aligned)
        diff_feats = extract_diff_features(diff)
    else:
        base = extract_features(canvas)
        diff_feats = np.zeros(160, dtype=np.float32)
    return np.concatenate([base, diff_feats])


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
        Load model bundle from disk (cached). Returns a dict with keys:
            clf            — sklearn classifier
            goldens        — {char_id: 48×48 uint8} per-char templates (may be empty)
            feat_version   — 'v1' or 'v2'
        Legacy joblibs that stored the raw classifier directly are wrapped
        into the same shape so callers can ignore the format difference.
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
                if isinstance(data, dict) and ('clf' in data or 'goldens' in data):
                    bundle = {
                        'clf':           data.get('clf'),
                        'goldens':       data.get('goldens') or {},
                        'feat_version':  data.get('feat_version', 'v2'),
                    }
                else:
                    # Legacy: raw classifier
                    bundle = {'clf': data, 'goldens': {}, 'feat_version': 'v1'}
                self._cache[key] = bundle
                logger.info(
                    f"ML model loaded: project={project_id}, model={model_id}, "
                    f"feat_version={bundle['feat_version']}, "
                    f"goldens={len(bundle['goldens'])}, "
                    f"clf_type={type(bundle['clf']).__name__}"
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
        char_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Predict OK/NG on a single-character region crop.

        Args:
            char_id: identity of the character ('L', '0'...). Required for
                v2 models — picks the matching golden template for diff-based
                feature extraction. None / unknown char → falls back to v1.

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

            clf = bundle['clf']
            goldens = bundle['goldens']
            feat_version = bundle['feat_version']

            if feat_version == 'v2':
                feat = extract_features_v2(region_img, char_id, goldens).reshape(1, -1)
            else:
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

    def classify_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Batch-predict OK/NG for N region crops.

        Much faster than calling classify_region in parallel because:
          - sklearn predict_proba(N samples) is barely more expensive than
            predict_proba(1 sample) for RandomForest/MLP
          - Avoids Python thread contention (GIL bound on feature extraction)

        Input item shape:
            {'region_img', 'project_id', 'model_id', 'conf_threshold',
             'serial_number', 'annotation_idx', 'char_id'?}

        `char_id` is required for v2 models (picks the matching golden
        template); v1 models ignore it.

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

            clf = bundle['clf']
            goldens = bundle['goldens']
            use_v2 = (bundle['feat_version'] == 'v2')

            # Feature extraction (Python loop but cheap per-item: ~1-3ms)
            feats_list: List[np.ndarray] = []
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
                if use_v2:
                    feats_list.append(extract_features_v2(region, item.get('char_id'), goldens))
                else:
                    feats_list.append(extract_features(region))
                valid_idxs.append(i)

            if not valid_idxs:
                continue

            t_feat = time.perf_counter()
            X = np.stack(feats_list, axis=0)
            probas = clf.predict_proba(X)  # shape (M, 2) or (M, 1)
            t_pred = time.perf_counter()

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

            group_ms = (time.perf_counter() - t0) * 1000
            feat_ms = (t_feat - t0) * 1000
            pred_ms = (t_pred - t_feat) * 1000
            for i in valid_idxs:
                if results[i] is not None:
                    results[i]['time_ms'] = round(group_ms / max(len(valid_idxs), 1), 2)

            logger.info(
                f"ML batch classify: project={pid}, model={mid}, "
                f"N={len(valid_idxs)}, feat={feat_ms:.1f}ms, "
                f"predict={pred_ms:.1f}ms, total={group_ms:.1f}ms"
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
