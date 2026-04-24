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


# ──────────────────────────────────────── Augmentation helpers ──

def _estimate_bg_color(img: np.ndarray):
    """Ước lượng màu background (chữ tối → bg là vùng sáng, lấy percentile 75)."""
    if img.ndim == 3:
        return np.percentile(img.reshape(-1, img.shape[2]), 75, axis=0)
    return np.percentile(img, 75)


def _estimate_fg_color(img: np.ndarray):
    """Ước lượng màu foreground (chữ) — vùng tối, percentile 25."""
    if img.ndim == 3:
        return np.percentile(img.reshape(-1, img.shape[2]), 25, axis=0)
    return np.percentile(img, 25)


# ──────────────────────────────────────── Augmentation (synthetic OK) ──

def augment_ok(char_img: np.ndarray, n: int = 5) -> List[np.ndarray]:
    """
    Generate n mildly-augmented OK samples.

    Biên độ NHỎ — giữ nguyên semantic OK. Mô phỏng variation thực tế:
    rotation nhẹ, dịch vị trí nhỏ, ánh sáng, noise sensor, focus drift nhẹ.
    """
    gray = _to_gray(char_img)
    h, w = gray.shape[:2]
    results: List[np.ndarray] = []
    num_aug_types = 5
    choices = np.random.choice(num_aug_types, size=n, replace=n > num_aug_types)

    for choice in choices:
        aug = gray.copy()
        if choice == 0:
            # Rotation ±5° — replicate border để không tạo viền đen giả
            angle = float(np.random.uniform(-5, 5))
            M = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            aug = cv.warpAffine(
                aug, M, (w, h),
                flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE,
            )
        elif choice == 1:
            # Translation ±3px
            dx = int(np.random.randint(-3, 4))
            dy = int(np.random.randint(-3, 4))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aug = cv.warpAffine(
                aug, M, (w, h),
                flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE,
            )
        elif choice == 2:
            # Brightness/contrast jitter ±15%
            alpha = float(np.random.uniform(0.85, 1.15))
            beta = int(np.random.randint(-15, 15))
            aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        elif choice == 3:
            # Mild gaussian noise (σ=5-8) — sensor noise
            sigma = float(np.random.uniform(5, 8))
            noise = np.random.normal(0, sigma, aug.shape).astype(np.int16)
            aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif choice == 4:
            # Slight blur k=3/5 — focus drift nhẹ
            k = int(np.random.choice([3, 5]))
            aug = cv.GaussianBlur(aug, (k, k), 0)
        results.append(aug)
    return results


# ──────────────────────────────────────── Augmentation (synthetic NG) ──

def _ng_transform(aug: np.ndarray, choice: int) -> np.ndarray:
    """Apply a single NG transform. Split out so augment_ng can chain 2 at once."""
    h, w = aug.shape[:2]

    if choice == 0:
        # Heavy noise σ=40-80 — bụi/nhiễu
        sigma = float(np.random.uniform(40, 80))
        noise = np.random.normal(0, sigma, aug.shape).astype(np.int16)
        return np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if choice == 1:
        # Localized cut — vá bg_color 3-8px tại vị trí ngẫu nhiên trên stroke
        # Simulate "in mất nét 1 đoạn cục bộ"
        bg = _estimate_bg_color(aug)
        fg = _estimate_fg_color(aug)
        # Build text mask (dark pixels = stroke)
        gray = aug if aug.ndim == 2 else cv.cvtColor(aug, cv.COLOR_BGR2GRAY)
        thr = (float(np.mean([fg if np.isscalar(fg) else fg.mean(),
                              bg if np.isscalar(bg) else bg.mean()])))
        text_mask = gray < thr
        ys, xs = np.where(text_mask)
        if len(xs) < 5:
            # Fallback — không có stroke rõ thì cut strip nhỏ
            rh = int(np.random.randint(max(3, int(h * 0.05)), max(8, int(h * 0.10))))
            ry = int(np.random.randint(0, max(1, h - rh)))
            aug[ry:ry + rh, :] = bg
            return aug

        num_cuts = int(np.random.randint(1, 3))
        for _ in range(num_cuts):
            i = int(np.random.randint(0, len(xs)))
            cx, cy = int(xs[i]), int(ys[i])
            patch_w = int(np.random.randint(3, 7))
            patch_h = int(np.random.randint(3, 7))
            x0 = max(0, cx - patch_w // 2)
            y0 = max(0, cy - patch_h // 2)
            x1 = min(w, x0 + patch_w)
            y1 = min(h, y0 + patch_h)
            aug[y0:y1, x0:x1] = bg
        return aug

    if choice == 2:
        # Partial erosion — chỉ erode 1 nửa ảnh (lỗi ribbon/head 1 phía)
        k = int(np.random.randint(5, 9))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        eroded = cv.erode(aug, kernel, iterations=1)
        side = int(np.random.randint(0, 4))  # 0=left, 1=right, 2=top, 3=bottom
        out = aug.copy()
        if side == 0:
            out[:, :w // 2] = eroded[:, :w // 2]
        elif side == 1:
            out[:, w // 2:] = eroded[:, w // 2:]
        elif side == 2:
            out[:h // 2, :] = eroded[:h // 2, :]
        else:
            out[h // 2:, :] = eroded[h // 2:, :]
        return out

    if choice == 3:
        # Dilate full — mực chảy dày toàn ký tự
        k = int(np.random.randint(6, 9))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        return cv.dilate(aug, kernel, iterations=1)

    if choice == 4:
        # Strip cut — 1 đường bg_color cắt NGANG hoặc DỌC qua toàn ký tự
        # Simulate "mất nét 1 đường luôn" (ribbon/head miss trên 1 hàng pixels)
        bg = _estimate_bg_color(aug)
        out = aug.copy()
        if np.random.rand() < 0.5:
            # Horizontal strip — cut across full width
            thickness = int(np.random.randint(max(2, int(h * 0.04)),
                                              max(5, int(h * 0.12))))
            y0 = int(np.random.randint(0, max(1, h - thickness)))
            out[y0:y0 + thickness, :] = bg
        else:
            # Vertical strip — cut across full height
            thickness = int(np.random.randint(max(2, int(w * 0.04)),
                                              max(5, int(w * 0.12))))
            x0 = int(np.random.randint(0, max(1, w - thickness)))
            out[:, x0:x0 + thickness] = bg
        return out

    if choice == 5:
        # Ink blot — chấm đen (fg color) 3-5px tại vị trí ngẫu nhiên
        fg = _estimate_fg_color(aug)
        num_blots = int(np.random.randint(1, 3))
        for _ in range(num_blots):
            cx = int(np.random.randint(2, max(3, w - 2)))
            cy = int(np.random.randint(2, max(3, h - 2)))
            radius = int(np.random.randint(2, 5))
            cv.circle(aug, (cx, cy), radius, fg if np.isscalar(fg) else tuple(fg.tolist()), -1)
        return aug

    if choice == 6:
        # Ghosting — shift + overlay alpha 0.4 (in chồng)
        dx = int(np.random.randint(3, 6)) * int(np.random.choice([-1, 1]))
        dy = int(np.random.randint(2, 5)) * int(np.random.choice([-1, 1]))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv.warpAffine(
            aug, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE,
        )
        alpha = float(np.random.uniform(0.35, 0.55))
        return cv.addWeighted(aug, 1.0 - alpha, shifted, alpha, 0)

    return aug


def augment_ng(char_img: np.ndarray, n: int = 5) -> List[np.ndarray]:
    """
    Generate n synthetic NG samples.

    7 transform types (NO blur — blur cũng có thể gặp ở OK sample thật):
      0 heavy noise | 1 localized cut | 2 partial erosion | 3 dilate full
      4 strip cut   | 5 ink blot      | 6 ghosting
    Với ~20% xác suất mỗi sample sẽ chain 2 transforms khác nhau để
    tạo defect phức hợp (e.g. noise + cut) giống thực tế hơn.
    """
    gray = _to_gray(char_img)
    results: List[np.ndarray] = []
    num_aug_types = 7

    for _ in range(n):
        aug = gray.copy()
        first = int(np.random.randint(0, num_aug_types))
        aug = _ng_transform(aug, first)

        # 20% chance chain a second distinct transform
        if np.random.rand() < 0.20:
            remaining = [c for c in range(num_aug_types) if c != first]
            second = int(np.random.choice(remaining))
            aug = _ng_transform(aug, second)

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

    n_ok_real = len(ok_imgs)
    n_ng_real = len(ng_imgs)

    # Balance formula — aim for total_ok ≈ total_ng:
    #   n_aug_ng = (factor - 1) * n_ok_real
    #   n_aug_ok = n_ng_real + (factor - 2) * n_ok_real     (floored at 0)
    # See docs: x2 → 0 OK aug + n_ok NG aug; x3 → n_ok OK aug + 2n_ok NG aug.
    aug_ok_imgs: List[np.ndarray] = []
    aug_ng_imgs: List[np.ndarray] = []

    if augment_factor >= 2 and n_ok_real > 0:
        # --- Augment NG from OK templates ---
        n_per_ok_ng = augment_factor - 1
        for c in ok_imgs:
            aug_ng_imgs.extend(augment_ng(c, n=n_per_ok_ng))

        # --- Augment OK to balance class ---
        n_aug_ok_total = n_ng_real + max(0, augment_factor - 2) * n_ok_real
        if n_aug_ok_total > 0:
            # Distribute per-sample as evenly as possible
            base = n_aug_ok_total // n_ok_real
            extra = n_aug_ok_total - base * n_ok_real
            for i, c in enumerate(ok_imgs):
                n_this = base + (1 if i < extra else 0)
                if n_this > 0:
                    aug_ok_imgs.extend(augment_ok(c, n=n_this))

    all_ok = ok_imgs + aug_ok_imgs
    all_ng = ng_imgs + aug_ng_imgs

    X_ok = np.array([extract_features(c) for c in all_ok], dtype=np.float32)
    y_ok = np.ones(len(X_ok), dtype=np.int32)

    if all_ng:
        X_ng = np.array([extract_features(c) for c in all_ng], dtype=np.float32)
        y_ng = np.zeros(len(X_ng), dtype=np.int32)
        X = np.vstack([X_ok, X_ng])
        y = np.concatenate([y_ok, y_ng])
        crops_all: List[np.ndarray] = list(all_ok) + list(all_ng)
    else:
        X = X_ok
        y = y_ok
        crops_all = list(all_ok)

    logger.info(
        f"[build_dataset] OK: {n_ok_real} real + {len(aug_ok_imgs)} aug = {len(all_ok)} | "
        f"NG: {n_ng_real} real + {len(aug_ng_imgs)} aug = {len(all_ng)} | "
        f"factor={augment_factor}"
    )

    # Shuffle — keep crops in sync
    idx = np.random.permutation(len(X))
    crops_shuffled = [crops_all[i] for i in idx]
    return X[idx], y[idx], crops_shuffled, len(all_ok), len(all_ng)


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
    label: str = "NG",
) -> List[Dict[str, Any]]:
    """
    Generate synthetic crops from OK samples for preview.

    Args:
        annotations: project annotations.
        images_dir: project images directory.
        augment_factor: preview uses (augment_factor - 1) augments per OK sample.
        label: 'NG' (destructive augs), 'OK' (mild augs), or 'BOTH'.

    Returns list of {source_segment_id, filename, label, crop_b64}.
    """
    if augment_factor < 2:
        return []
    n_per_sample = augment_factor - 1
    label = (label or "NG").upper()
    want_ng = label in ("NG", "BOTH")
    want_ok = label in ("OK", "BOTH")

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
                if want_ng:
                    for aug in augment_ng(crop, n=n_per_sample):
                        result.append({
                            "source_segment_id": seg.id,
                            "filename": ann.filename,
                            "label": "NG",
                            "crop_b64": img_to_b64(aug),
                        })
                if want_ok:
                    for aug in augment_ok(crop, n=n_per_sample):
                        result.append({
                            "source_segment_id": seg.id,
                            "filename": ann.filename,
                            "label": "OK",
                            "crop_b64": img_to_b64(aug),
                        })
    return result
