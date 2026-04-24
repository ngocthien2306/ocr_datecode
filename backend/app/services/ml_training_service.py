"""
ML Training Service
Handles feature extraction, augmentation, model training and prediction.
Algorithms: Random Forest, SVM, MLP.
"""
import base64
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import joblib
import numpy as np

from app.models.ml_training import MLAnnotationInDB, TrainRequest
from app.services.ml_segment_service import crop_segment, segment_region

logger = logging.getLogger(__name__)

FEAT_SIZE = (32, 32)


# ──────────────────────────────────────── Feature extraction ──

def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def extract_features(char_img: np.ndarray) -> np.ndarray:
    """
    Extract 1120-dim feature vector:
      - 32×32 normalized pixels (1024)
      - Sobel Gx/Gy histograms (32)
      - H/V projection profiles (64)
    """
    gray = _to_gray(char_img)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(1120, dtype=np.float32)

    scale = min(FEAT_SIZE[0] / w, FEAT_SIZE[1] / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv.resize(gray, (nw, nh), interpolation=cv.INTER_AREA)
    canvas = np.zeros(FEAT_SIZE[::-1], dtype=np.uint8)
    yo = (FEAT_SIZE[1] - nh) // 2
    xo = (FEAT_SIZE[0] - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized

    pixels = canvas.astype(np.float32).flatten() / 255.0

    gx = cv.Sobel(canvas, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(canvas, cv.CV_32F, 0, 1, ksize=3)
    hist_gx = np.histogram(gx, bins=16, range=(-255, 255))[0].astype(np.float32)
    hist_gy = np.histogram(gy, bins=16, range=(-255, 255))[0].astype(np.float32)
    hist_gx /= hist_gx.sum() + 1e-6
    hist_gy /= hist_gy.sum() + 1e-6

    h_proj = canvas.astype(np.float32).sum(axis=1) / (FEAT_SIZE[0] * 255.0 + 1e-6)
    v_proj = canvas.astype(np.float32).sum(axis=0) / (FEAT_SIZE[1] * 255.0 + 1e-6)

    return np.concatenate([pixels, hist_gx, hist_gy, h_proj, v_proj])


# ──────────────────────────────────────── Augmentation (synthetic NG) ──

def augment_ng(char_img: np.ndarray, n: int = 5) -> List[np.ndarray]:
    """Generate n synthetic NG samples from an OK character image."""
    gray = _to_gray(char_img)
    h, w = gray.shape[:2]
    results = []
    num_aug_types = 4
    choices = np.random.choice(num_aug_types, size=n, replace=n > num_aug_types)
    for choice in choices:
        aug = gray.copy()
        if choice == 0:
            noise = np.random.normal(0, 60, aug.shape).astype(np.int16)
            aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif choice == 1:
            # Ước lượng màu background (chữ tối → background là vùng sáng)
            if aug.ndim == 3:
                bg_color = np.percentile(aug.reshape(-1, aug.shape[2]), 75, axis=0)
            else:
                bg_color = np.percentile(aug, 75)

            num_cuts = np.random.randint(1, 3)
            for _c in range(num_cuts):
                # Chọn ngẫu nhiên vệt ngang hay vệt dọc
                if np.random.rand() < 0.5:
                    # Vệt ngang: rộng hết ảnh, cao dày
                    rh = np.random.randint(max(8, h // 8), max(10, h // 3))
                    ry = np.random.randint(0, max(1, h - rh))
                    aug[ry:ry + rh, :] = bg_color
                else:
                    # Vệt dọc: cao hết ảnh, rộng dày
                    rw = np.random.randint(max(8, w // 8), max(10, w // 3))
                    rx = np.random.randint(0, max(1, w - rw))
                    aug[:, rx:rx + rw] = bg_color
        elif choice == 2:
            k = np.random.randint(7, 10)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
            aug = cv.erode(aug, kernel, iterations=1)
        elif choice == 3:
            k = np.random.randint(4, 7)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
            aug = cv.dilate(aug, kernel, iterations=1)
        # elif choice == 4:
        #     dx = np.random.randint(-w // 3, w // 3 + 1)
        #     dy = np.random.randint(-h // 3, h // 3 + 1)
        #     M = np.float32([[1, 0, dx], [0, 1, dy]])
        #     aug = cv.warpAffine(aug, M, (w, h), borderValue=255)
        # elif choice == 5:
        #     k = np.random.choice([17, 19, 21, 23])
        #     aug = cv.GaussianBlur(aug, (k, k), 0)
        results.append(aug)
    return results


# ──────────────────────────────────────── Image encoding ──

def img_to_b64(img: np.ndarray, quality: int = 85) -> str:
    """Encode BGR numpy array to base64 JPEG string."""
    ok, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ──────────────────────────────────────── Training ──

def build_dataset(
    annotations: List[MLAnnotationInDB],
    images_dir: Path,
    augment_factor: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], int, int]:
    """
    Build (X, y, crops_raw) dataset from project annotations.

    Returns:
        X: feature matrix
        y: labels (1=OK, 0=NG)
        crops_raw: list of raw crop images, same order as X/y rows
        n_ok: count of real OK samples
        n_ng: count of real+synthetic NG samples
    """
    ok_imgs: List[np.ndarray] = []
    ng_imgs: List[np.ndarray] = []

    for ann in annotations:
        img_path = images_dir / ann.filename
        for region in ann.regions:
            for seg in region.segments:
                if seg.label not in ("OK", "NG"):
                    continue
                crop = crop_segment(img_path, {
                    "x": seg.x, "y": seg.y, "w": seg.w, "h": seg.h,
                })
                if crop is None:
                    continue
                if seg.label == "OK":
                    ok_imgs.append(crop)
                else:
                    ng_imgs.append(crop)

    if not ok_imgs and not ng_imgs:
        raise ValueError("No labeled segments found. Please label images before training.")

    X_ok = np.array([extract_features(c) for c in ok_imgs], dtype=np.float32)
    y_ok = np.ones(len(X_ok), dtype=np.int32)

    # Augment OK→synthetic NG if augment_factor > 0
    aug_ng_imgs: List[np.ndarray] = []
    if augment_factor >= 2:
        n_per_sample = augment_factor - 1
        for c in ok_imgs:
            aug_ng_imgs.extend(augment_ng(c, n=n_per_sample))

    all_ng = ng_imgs + aug_ng_imgs
    if all_ng:
        X_ng = np.array([extract_features(c) for c in all_ng], dtype=np.float32)
        y_ng = np.zeros(len(X_ng), dtype=np.int32)
        X = np.vstack([X_ok, X_ng])
        y = np.concatenate([y_ok, y_ng])
        crops_all: List[np.ndarray] = ok_imgs + all_ng
    else:
        X = X_ok
        y = y_ok
        crops_all = list(ok_imgs)

    # Shuffle — keep crops in sync
    idx = np.random.permutation(len(X))
    crops_shuffled = [crops_all[i] for i in idx]
    return X[idx], y[idx], crops_shuffled, len(ok_imgs), len(all_ng)


def train_model(
    annotations: List[MLAnnotationInDB],
    images_dir: Path,
    request: TrainRequest,
    model_save_path: Path,
) -> Dict[str, Any]:
    """
    Train a classifier and save it to disk.
    Also saves the test-set crops with predictions to a sidecar JSON file.
    Returns metrics dict.
    """
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    X, y, crops_raw, n_ok, n_ng = build_dataset(annotations, images_dir, request.augment_factor)

    if len(X) < 4:
        raise ValueError(f"Need at least 4 samples, got {len(X)}.")

    # Split — pass crops_raw alongside X/y so indices stay in sync
    test_size = min(request.test_split, 0.4)
    if len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test, _, crops_test = train_test_split(
            X, y, crops_raw, test_size=test_size, random_state=42, stratify=y,
        )
    else:
        # Only one class — no meaningful split; use all crops for display
        X_train, X_test, y_train, y_test = X, X, y, y
        crops_test = crops_raw

    # Build & train classifier
    clf = _build_classifier(request)
    clf.fit(X_train, y_train)

    threshold = float(getattr(request, "threshold", 0.5))

    def _apply_threshold(X: np.ndarray) -> np.ndarray:
        proba = clf.predict_proba(X)
        p_ok = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        return (p_ok >= threshold).astype(np.int32)

    # Evaluate using the configured threshold
    y_pred_train = _apply_threshold(X_train)
    y_pred_test  = _apply_threshold(X_test)
    acc_train = float(accuracy_score(y_train, y_pred_train))
    acc_test  = float(accuracy_score(y_test,  y_pred_test))

    cm     = confusion_matrix(y_test, y_pred_test).tolist()
    report = classification_report(y_test, y_pred_test,
                                   target_names=["NG", "OK"], zero_division=0)

    # Build per-crop test-set records (saved as sidecar JSON)
    proba_test = clf.predict_proba(X_test)
    test_set_items = []
    for crop_img, true_y, pred_y, proba in zip(crops_test, y_test, y_pred_test, proba_test):
        p_ok = float(proba[1]) if len(proba) > 1 else float(proba[0])
        test_set_items.append({
            "crop_b64":   img_to_b64(crop_img),
            "true_label": "OK" if int(true_y) == 1 else "NG",
            "pred_label": "OK" if int(pred_y) == 1 else "NG",
            "prob_ok":    round(p_ok, 4),
            "correct":    bool(true_y == pred_y),
        })

    # Save model + sidecar JSON
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, str(model_save_path))
    test_set_path = model_save_path.parent / f"{model_save_path.stem}_test_set.json"
    test_set_path.write_text(json.dumps(test_set_items))

    return {
        "accuracy_train": acc_train,
        "accuracy_test":  acc_test,
        "n_ok":           n_ok,
        "n_ng":           n_ng,
        "n_total":        len(X),
        "confusion_matrix": cm,
        "report":         report,
    }


def _build_classifier(request: TrainRequest):
    algo = request.algorithm.lower()
    if algo == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=request.n_estimators,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif algo == "svm":
        from sklearn.svm import SVC
        return SVC(
            C=request.C,
            kernel="rbf",
            probability=True,
            max_iter=request.max_iter,
            random_state=42,
        )
    elif algo == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=tuple(request.hidden_layer_sizes),
            max_iter=request.max_iter,
            random_state=42,
            early_stopping=True,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


# ──────────────────────────────────────── Prediction ──

def predict_on_image(
    model_path: Path,
    image_path: Path,
    region: Optional[Dict] = None,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Segment an image (or a region of it) and predict OK/NG per character.

    Args:
        model_path: Path to saved joblib model.
        image_path: Image to predict on.
        region: Optional {x, y, w, h} normalized — segment only this area.
        threshold: Probability threshold for OK class.

    Returns:
        List of {id, x, y, w, h, prob_ok, label, crop_b64}
    """
    clf = joblib.load(str(model_path))

    if region:
        segments = segment_region(image_path, region)
    else:
        # Segment entire image using a full-image region
        segments = segment_region(image_path, {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})

    results = []
    for seg in segments:
        crop = crop_segment(image_path, seg)
        if crop is None:
            continue
        feat = extract_features(crop).reshape(1, -1)
        proba = clf.predict_proba(feat)[0]
        # index 1 = OK if both classes present, else use index 0
        p_ok = float(proba[1]) if len(proba) > 1 else float(proba[0])
        label = "OK" if p_ok >= threshold else "NG"
        results.append({
            "id": seg["id"],
            "x": seg["x"],
            "y": seg["y"],
            "w": seg["w"],
            "h": seg["h"],
            "prob_ok": round(p_ok, 4),
            "label": label,
            "crop_b64": img_to_b64(crop),
        })

    return results


def get_labeled_crops(
    annotations: List[MLAnnotationInDB],
    images_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Collect all labeled character crops for the Train tab preview grid.
    Returns list of {segment_id, region_id, filename, label, crop_b64}.
    """
    result = []
    for ann in annotations:
        img_path = images_dir / ann.filename
        for region in ann.regions:
            for seg in region.segments:
                if seg.label not in ("OK", "NG"):
                    continue
                crop = crop_segment(img_path, {
                    "x": seg.x, "y": seg.y, "w": seg.w, "h": seg.h,
                })
                if crop is None:
                    continue
                result.append({
                    "segment_id": seg.id,
                    "region_id": region.id,
                    "filename": ann.filename,
                    "label": seg.label,
                    "crop_b64": img_to_b64(crop),
                })
    return result


def generate_synthetic_crops(
    annotations: List[MLAnnotationInDB],
    images_dir: Path,
    augment_factor: int,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic NG crops from OK samples for preview.
    augment_factor must be >= 2 (n_per_sample = augment_factor - 1).
    Returns list of {source_segment_id, filename, label, crop_b64}.
    """
    if augment_factor < 2:
        return []
    n_per_sample = augment_factor - 1
    result = []
    for ann in annotations:
        img_path = images_dir / ann.filename
        for region in ann.regions:
            for seg in region.segments:
                if seg.label != "OK":
                    continue
                crop = crop_segment(img_path, {
                    "x": seg.x, "y": seg.y, "w": seg.w, "h": seg.h,
                })
                if crop is None:
                    continue
                aug_imgs = augment_ng(crop, n=n_per_sample)
                for aug in aug_imgs:
                    result.append({
                        "source_segment_id": seg.id,
                        "filename": ann.filename,
                        "label": "NG",
                        "crop_b64": img_to_b64(aug),
                    })
    return result
