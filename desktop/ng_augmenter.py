"""
Synthetic NG (defect) augmentation — generate corrupted versions of a clean
character crop to validate that the comparison pipeline detects them.

Augmentation types: noise, cut, erode (thinning), dilate (bleed), line (cross).
All operate on BGR uint8 arrays.
"""
import cv2 as cv
import numpy as np


NG_AUG_TYPES = ["noise", "cut", "erode", "dilate", "line"]

DEFAULT_AUG_PARAMS = {
    'n_samples': 24,             # how many samples to generate
    'seed': 0,                    # 0 = random each call; >0 = deterministic
    # noise
    'noise_sigma': 30,            # gaussian std-dev added to pixels
    # cut
    'cut_count_min': 1,
    'cut_count_max': 3,
    'cut_size_frac_min': 0.16,    # rect size = frac * image dim
    'cut_size_frac_max': 0.34,
    # erode / dilate
    'erode_k_min': 4,
    'erode_k_max': 6,
    'dilate_k_min': 5,
    'dilate_k_max': 7,
    # line (ribbon-miss / scratch)
    'line_count_min': 1,
    'line_count_max': 2,
    'line_thick_min': 2,
    'line_thick_max_frac': 0.125,  # thickness <= frac * min(h,w)
    # which augmentations to enable (subset of NG_AUG_TYPES)
    'enabled': list(NG_AUG_TYPES),
}


def _pick_patch_color(img):
    """Black / white / background-median, picked uniformly at random."""
    choice = int(np.random.randint(0, 3))
    if choice == 0:
        return (0, 0, 0)
    if choice == 1:
        return (255, 255, 255)
    if len(img.shape) == 3:
        return tuple(int(v) for v in np.median(img.reshape(-1, 3), axis=0))
    return (int(np.median(img)),) * 3


def _ensure_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img


def augment_one(img, aug_type, params=None):
    """Apply ONE specific augmentation type to a BGR image. Returns BGR result."""
    p = {**DEFAULT_AUG_PARAMS, **(params or {})}
    img = _ensure_bgr(img).copy()
    h, w = img.shape[:2]

    if aug_type == "noise":
        sigma = float(p['noise_sigma'])
        noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if aug_type == "cut":
        n = int(np.random.randint(int(p['cut_count_min']), int(p['cut_count_max']) + 1))
        f_min, f_max = float(p['cut_size_frac_min']), float(p['cut_size_frac_max'])
        for _ in range(n):
            rh = max(2, int(h * np.random.uniform(f_min, f_max)))
            rw = max(2, int(w * np.random.uniform(f_min, f_max)))
            ry = int(np.random.randint(0, max(1, h - rh)))
            rx = int(np.random.randint(0, max(1, w - rw)))
            img[ry:ry + rh, rx:rx + rw] = _pick_patch_color(img)
        return img

    if aug_type == "erode":
        k = int(np.random.randint(int(p['erode_k_min']), int(p['erode_k_max']) + 1))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (max(1, k), max(1, k)))
        return cv.erode(img, kernel, iterations=1)

    if aug_type == "dilate":
        k = int(np.random.randint(int(p['dilate_k_min']), int(p['dilate_k_max']) + 1))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (max(1, k), max(1, k)))
        return cv.dilate(img, kernel, iterations=1)

    if aug_type == "line":
        n = int(np.random.randint(int(p['line_count_min']), int(p['line_count_max']) + 1))
        max_thick = max(3, int(min(h, w) * float(p['line_thick_max_frac']))) + 1
        for _ in range(n):
            color = _pick_patch_color(img)
            thickness = int(np.random.randint(int(p['line_thick_min']), max_thick))
            if np.random.rand() < 0.5:
                band = int(np.random.choice([0, 1, 2]))
                center = [int(h * 0.2), int(h * 0.5), int(h * 0.8)][band]
                y0 = max(0, center - thickness // 2)
                y1 = min(h, y0 + thickness)
                img[y0:y1, :] = color
            else:
                band = int(np.random.choice([0, 1, 2]))
                center = [int(w * 0.2), int(w * 0.5), int(w * 0.8)][band]
                x0 = max(0, center - thickness // 2)
                x1 = min(w, x0 + thickness)
                img[:, x0:x1] = color
        return img

    return img  # unknown type → passthrough


def generate_samples(img, params=None):
    """
    Generate N synthetic NG variants of `img` (BGR).
    Returns list of dicts: [{'image': bgr, 'type': str}, ...]
    """
    p = {**DEFAULT_AUG_PARAMS, **(params or {})}
    enabled = [t for t in p['enabled'] if t in NG_AUG_TYPES]
    if not enabled:
        return []

    seed = int(p.get('seed') or 0)
    if seed > 0:
        np.random.seed(seed)

    n = max(1, int(p['n_samples']))
    samples = []
    for i in range(n):
        aug_type = enabled[i % len(enabled)] if n <= len(enabled) else \
                   enabled[int(np.random.randint(0, len(enabled)))]
        out = augment_one(img, aug_type, p)
        samples.append({'image': out, 'type': aug_type})
    return samples
