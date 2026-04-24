"""
Random Forest character quality classifier.

Train with OK character images from templates,
auto-generate NG samples via augmentation,
predict PASS/FAIL per character with adjustable threshold.
"""

import os
import numpy as np
import cv2 as cv
from sklearn.ensemble import RandomForestClassifier
import joblib

from test_segment import segment_characters

# ─────────────────────────────────────────────── Feature extraction ──

FEAT_SIZE = (32, 32)


def _to_gray(img):
    if len(img.shape) == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def extract_features(char_img):
    """Extract feature vector from a single character image.

    Features:
      - 32x32 normalized pixel intensities (1024)
      - Sobel gradient histograms Gx/Gy (2x16 = 32)
      - Horizontal/vertical projection profiles (32+32 = 64)
    Total: 1120 features
    """
    gray = _to_gray(char_img)

    # Normalize to 32x32, keep aspect ratio, pad black
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

    # (1) Flatten pixels, normalize 0-1
    pixels = canvas.astype(np.float32).flatten() / 255.0

    # (2) Sobel gradient histograms
    gx = cv.Sobel(canvas, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(canvas, cv.CV_32F, 0, 1, ksize=3)
    hist_gx = np.histogram(gx, bins=16, range=(-255, 255))[0].astype(np.float32)
    hist_gy = np.histogram(gy, bins=16, range=(-255, 255))[0].astype(np.float32)
    total_gx = hist_gx.sum() + 1e-6
    total_gy = hist_gy.sum() + 1e-6
    hist_gx /= total_gx
    hist_gy /= total_gy

    # (3) Projection profiles
    h_proj = canvas.astype(np.float32).sum(axis=1) / (FEAT_SIZE[0] * 255.0 + 1e-6)
    v_proj = canvas.astype(np.float32).sum(axis=0) / (FEAT_SIZE[1] * 255.0 + 1e-6)

    return np.concatenate([pixels, hist_gx, hist_gy, h_proj, v_proj])


# ─────────────────────────────────────────────── Augmentation (NG) ──

def _augment_ng(char_img, n_augments=5):
    """Generate synthetic NG samples from an OK character image."""
    gray = _to_gray(char_img)
    h, w = gray.shape[:2]
    results = []

    for _ in range(n_augments):
        aug = gray.copy()
        choice = np.random.randint(0, 6)

        if choice == 0:
            # Heavy Gaussian noise
            noise = np.random.normal(0, 60, aug.shape).astype(np.int16)
            aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        elif choice == 1:
            # Strong blur
            k = np.random.choice([7, 9, 11, 13])
            aug = cv.GaussianBlur(aug, (k, k), 0)

        elif choice == 2:
            # Random erase (block out part of image)
            rh = max(2, h // 3)
            rw = max(2, w // 3)
            ry = np.random.randint(0, max(1, h - rh))
            rx = np.random.randint(0, max(1, w - rw))
            aug[ry:ry + rh, rx:rx + rw] = 0

        elif choice == 3:
            # Heavy morphological erosion (thin strokes)
            k = np.random.randint(3, 6)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
            aug = cv.erode(aug, kernel, iterations=1)

        elif choice == 4:
            # Heavy morphological dilation (fat strokes)
            k = np.random.randint(3, 6)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
            aug = cv.dilate(aug, kernel, iterations=1)

        elif choice == 5:
            # Random shift
            dx = np.random.randint(-w // 3, w // 3 + 1)
            dy = np.random.randint(-h // 3, h // 3 + 1)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aug = cv.warpAffine(aug, M, (w, h), borderValue=0)

        results.append(aug)

    return results


# ─────────────────────────────────────────────── Training ────────────

def collect_ok_chars(image_paths, save_dir=None):
    """Segment all template images and collect character crops.

    If save_dir is provided, saves each crop to disk organized by source image.
    Returns list of char images (grayscale).
    """
    all_chars = []
    if save_dir:
        ok_dir = os.path.join(save_dir, "ok_chars")
        os.makedirs(ok_dir, exist_ok=True)

    global_idx = 0
    for path in image_paths:
        boxes, chars, thresh_chars, img, thresh = segment_characters(path, save=False)
        if not chars:
            continue
        src_name = os.path.splitext(os.path.basename(path))[0]
        for i, c in enumerate(chars):
            gray_c = _to_gray(c)
            all_chars.append(gray_c)
            if save_dir:
                fname = f"{global_idx:04d}_{src_name}_char{i}.png"
                cv.imwrite(os.path.join(ok_dir, fname), c)
            global_idx += 1

    return all_chars


def train_model(ok_image_paths, n_augments=5, n_estimators=100,
                model_path=None, save_dir=None):
    """Train Random Forest from OK template images.

    1. Segment all OK images → collect OK char crops
    2. Extract features → label 1
    3. Augment → synthetic NG → label 0
    4. Train RF
    5. Save model

    If save_dir is provided, saves OK crops and NG augmented crops to disk.
    Returns (model, info_dict).
    """
    ok_chars = collect_ok_chars(ok_image_paths, save_dir=save_dir)
    if len(ok_chars) == 0:
        raise ValueError("No characters found in template images.")

    # OK features
    ok_features = np.array([extract_features(c) for c in ok_chars], dtype=np.float32)
    ok_labels = np.ones(len(ok_features), dtype=np.int32)

    # NG features from augmentation
    ng_features_list = []
    if save_dir:
        ng_dir = os.path.join(save_dir, "ng_chars")
        os.makedirs(ng_dir, exist_ok=True)
    ng_idx = 0
    for c in ok_chars:
        for aug in _augment_ng(c, n_augments=n_augments):
            ng_features_list.append(extract_features(aug))
            if save_dir:
                cv.imwrite(os.path.join(ng_dir, f"ng_{ng_idx:04d}.png"), aug)
            ng_idx += 1

    ng_features = np.array(ng_features_list, dtype=np.float32)
    ng_labels = np.zeros(len(ng_features), dtype=np.int32)

    # Combine
    X = np.vstack([ok_features, ng_features])
    y = np.concatenate([ok_labels, ng_labels])

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Train
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)

    info = {
        "n_ok": len(ok_features),
        "n_ng": len(ng_features),
        "n_total": len(X),
        "n_templates": len(ok_image_paths),
        "accuracy_train": float(clf.score(X, y)),
        "save_dir": save_dir,
    }

    if model_path:
        joblib.dump(clf, model_path)
        info["model_path"] = model_path

    return clf, info


# ─────────────────────────────────────────────── Prediction ──────────

def predict_image(model, image_path, threshold=0.5):
    """Segment target image and predict each character.

    Returns list of dicts: [{char_idx, prob_ok, label, char_img}, ...]
    """
    boxes, chars, thresh_chars, img, thresh = segment_characters(image_path, save=False)
    if not chars:
        return [], img, thresh

    results = []
    for i, c in enumerate(chars):
        feat = extract_features(c).reshape(1, -1)
        prob = model.predict_proba(feat)[0]
        # prob[1] = P(OK), prob[0] = P(NG)
        p_ok = float(prob[1]) if len(prob) > 1 else float(prob[0])
        label = "PASS" if p_ok >= threshold else "FAIL"
        results.append({
            "char_idx": i,
            "prob_ok": round(p_ok, 4),
            "label": label,
            "char_img": c,
            "box": boxes[i],
        })

    return results, img, thresh


def load_model(model_path):
    """Load a saved Random Forest model."""
    return joblib.load(model_path)
