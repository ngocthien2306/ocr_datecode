"""
Synthetic print-defect generator: teach the recogniser to REFUSE to read a
badly printed character instead of guessing it.

Why this exists: a well-trained OCR model is trained to be robust — RecAug adds
blur, noise and distortion so it reads through bad images — which means it also
reads through broken strokes, missing dots and overprinted characters and the
line PASSes a bad print. Here, verified crops get 1..max_chars characters
damaged on purpose, and the label says what a strict inspector would accept:

    stroke_break  a band through the strokes is erased        -> char becomes REJECT_CHAR
    dot_dropout   most CIJ dots of the char are erased        -> char becomes REJECT_CHAR
    overprint     another char from the same crop is stamped
                  over it (shifted)                           -> char becomes REJECT_CHAR
    blot          an ink smear covers part of it              -> char becomes REJECT_CHAR
    char_missing  the whole char is erased                    -> char is dropped from the label

REJECT_CHAR is '□': present in every PP-OCRv5/v6 dict (so the pretrained head
keeps its size) and never printed on a product. On the line, ai_services'
compare_texts() then fails the read — a '□' in the middle breaks the match, and
at the end it is stripped but the real character it replaced is gone, so the
length differs. No runtime change is needed.

Character positions come from CTC: the recogniser emits each character at one
output timestep, and each timestep covers ~8 px of the H=48 input. Only crops
the localiser reads EXACTLY (spaces ignored) are used, so every position is
known to belong to the right label character.

Only damage heavy enough to matter is labelled as a defect (see MIN_INK_REMOVED):
a lightly scratched character that a person still reads confidently should still
be read, otherwise the model learns to fail good bottles.
"""
import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np

from app.services import dataset_fs

logger = logging.getLogger(__name__)

REJECT_CHAR = "□"
DEFECT_TYPES = ("stroke_break", "dot_dropout", "overprint", "blot", "char_missing")
# Fraction of the character's ink that must be gone/obscured before the sample
# is labelled a defect. Below it the sample is discarded (never labelled as a
# clean read either — a half-damaged char is exactly the ambiguous case).
MIN_INK_REMOVED = 0.35

H = 48  # localiser input height, same as production


@dataclass
class CharBox:
    char: str
    idx: int          # index in the label string
    x0: int
    x1: int
    y0: int
    y1: int


def synth_dir(project_id: str) -> Path:
    return dataset_fs.dataset_dir(project_id) / "synth"


def manifest_path(project_id: str) -> Path:
    return synth_dir(project_id) / "manifest.json"


def load_manifest(project_id: str) -> Optional[Dict]:
    p = manifest_path(project_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.exception(f"[synth] unreadable manifest {p}")
        return None


def clear(project_id: str) -> None:
    shutil.rmtree(synth_dir(project_id), ignore_errors=True)


# ── localisation ─────────────────────────────────────────────────────────────


def _prep(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """Production preprocessing (ocr_utils.preprocess_text_image, min_width 50).
    Returns the tensor and the resized width."""
    h, w = gray.shape
    new_w = max(int(w * (H / h)), 10)
    r = cv.resize(gray, (new_w, H))
    if new_w < 50:
        r = cv.copyMakeBorder(r, 0, 0, 0, 50 - new_w, cv.BORDER_CONSTANT, value=0)
    x = (r.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.stack([x, x, x])[None], new_w


class Localiser:
    """CTC char positions from an exported rec ONNX + its dict."""

    def __init__(self, onnx_path: Path, dict_path: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        n_cls = self.sess.get_outputs()[0].shape[-1]
        chars = ["<blank>"] + [l.strip() for l in dict_path.read_text(encoding="utf-8").splitlines()]
        self.chars = chars + [" "] * max(0, n_cls - len(chars))

    def peaks(self, gray: np.ndarray) -> Tuple[str, List[Tuple[str, float]]]:
        """Decoded text and, per emitted char, its x centre in ORIGINAL pixels."""
        x, new_w = _prep(gray)
        pred = self.sess.run(None, {self.inp: x})[0][0]
        best = pred.argmax(-1)
        T = pred.shape[0]
        in_w = x.shape[-1]
        scale = gray.shape[1] / float(new_w)
        out, prev, text = [], 0, []
        for t, c in enumerate(best):
            if c != 0 and c != prev and c < len(self.chars):
                ch = self.chars[c]
                text.append(ch)
                out.append((ch, (t + 0.5) * in_w / T * scale))
            prev = c
        return "".join(text), out


def _ink_mask(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Ink = the Otsu minority class (works for light-on-dark CIJ caps and
    dark-on-light labels alike). Returns (mask uint8 0/255, ink_is_bright)."""
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    bright = (bw > 0).mean() < 0.5
    mask = bw if bright else cv.bitwise_not(bw)
    return mask, bright


def char_boxes(gray: np.ndarray, label: str, loc: Localiser) -> Optional[List[CharBox]]:
    """Boxes for every non-space label char, or None if the crop cannot be
    trusted (localiser misread it)."""
    decoded, peaks = loc.peaks(gray)
    if decoded.replace(" ", "") != label.replace(" ", ""):
        return None
    peaks = [(c, x) for c, x in peaks if c != " "]
    label_idx = [i for i, c in enumerate(label) if c != " "]
    if len(peaks) != len(label_idx) or len(peaks) < 2:
        return None

    xs = [x for _, x in peaks]
    gaps = np.diff(xs)
    pitch = float(np.median(gaps)) if len(gaps) else gray.shape[1] / 4
    mask, _ = _ink_mask(gray)
    h, w = gray.shape
    boxes = []
    for (c, x), li in zip(peaks, label_idx):
        # CTC fires near the left/centre of a glyph; a box one pitch wide,
        # shifted slightly right, covers it without eating the neighbours.
        half = max(pitch * 0.48, 3)
        x0 = int(max(0, round(x - half * 0.9)))
        x1 = int(min(w, round(x + half * 1.1)))
        cols = mask[:, x0:x1]
        rows = np.where(cols.sum(1) > 0)[0]
        y0, y1 = (int(rows.min()), int(rows.max()) + 1) if len(rows) else (0, h)
        boxes.append(CharBox(c, li, x0, x1, y0, y1))
    return boxes


# ── defects ──────────────────────────────────────────────────────────────────


def _erase(gray: np.ndarray, erase_mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels with surrounding background texture."""
    m = cv.dilate(erase_mask, np.ones((3, 3), np.uint8))
    return cv.inpaint(gray, m, 3, cv.INPAINT_TELEA)


def _box_ink(mask: np.ndarray, b: CharBox) -> np.ndarray:
    sub = np.zeros_like(mask)
    sub[b.y0:b.y1, b.x0:b.x1] = mask[b.y0:b.y1, b.x0:b.x1]
    return sub


def _stroke_break(gray, mask, b, rng):
    ink = _box_ink(mask, b)
    band = np.zeros_like(mask)
    bw, bh = b.x1 - b.x0, b.y1 - b.y0
    for _ in range(rng.randint(1, 2)):
        thick = max(2, int(rng.uniform(0.18, 0.32) * max(bw, bh)))
        if rng.random() < 0.5:   # horizontal-ish cut
            y = rng.randint(b.y0, max(b.y0, b.y1 - 1))
            p1, p2 = (b.x0 - 2, y + rng.randint(-3, 3)), (b.x1 + 2, y + rng.randint(-3, 3))
        else:                    # vertical-ish cut
            x = rng.randint(b.x0, max(b.x0, b.x1 - 1))
            p1, p2 = (x + rng.randint(-3, 3), b.y0 - 2), (x + rng.randint(-3, 3), b.y1 + 2)
        cv.line(band, p1, p2, 255, thick)
    hit = cv.bitwise_and(ink, band)
    return _erase(gray, hit), hit.sum() / max(ink.sum(), 1)


def _dot_dropout(gray, mask, b, rng):
    ink = _box_ink(mask, b)
    n, lab = cv.connectedComponents(ink)
    if n <= 2:   # strokes fused into one blob — fall back to a break
        return _stroke_break(gray, mask, b, rng)
    frac = rng.uniform(0.45, 0.75)
    drop = [i for i in range(1, n) if rng.random() < frac]
    hit = np.isin(lab, drop).astype(np.uint8) * 255
    return _erase(gray, hit), hit.sum() / max(ink.sum(), 1)


def _char_missing(gray, mask, b, rng):
    ink = _box_ink(mask, b)
    return _erase(gray, ink), 1.0


def _overprint(gray, mask, b, rng, boxes, bright, src=None):
    others = [o for o in boxes if o is not b and o.char.isalnum() and o.char != b.char]
    if not others:
        return gray, 0.0
    o = rng.choice(others)
    patch = (gray if src is None else src)[o.y0:o.y1, o.x0:o.x1]
    ph, pw = patch.shape
    dx = int(rng.uniform(-0.3, 0.3) * pw)
    dy = int(rng.uniform(-0.2, 0.2) * ph)
    x0 = min(max(0, b.x0 + dx), gray.shape[1] - pw)
    y0 = min(max(0, b.y0 + dy), gray.shape[0] - ph)
    out = gray.copy()
    roi = out[y0:y0 + ph, x0:x0 + pw]
    out[y0:y0 + ph, x0:x0 + pw] = np.maximum(roi, patch) if bright else np.minimum(roi, patch)
    # "Obscured" = the foreign ink landing inside the char box.
    ink = _box_ink(mask, b)
    foreign = np.zeros_like(mask)
    pm, _ = _ink_mask(patch) if patch.size else (np.zeros_like(patch), bright)
    foreign[y0:y0 + ph, x0:x0 + pw] = pm
    covered = cv.bitwise_and(foreign, cv.bitwise_not(ink))[b.y0:b.y1, b.x0:b.x1]
    return out, covered.sum() / max(ink.sum(), 1)


def _blot(gray, mask, b, rng, bright):
    ink = _box_ink(mask, b)
    ink_val = float(np.median(gray[ink > 0])) if ink.any() else (255 if bright else 0)
    blot = np.zeros_like(mask)
    cx = rng.randint(b.x0, max(b.x0, b.x1 - 1))
    cy = rng.randint(b.y0, max(b.y0, b.y1 - 1))
    ax = max(2, int(rng.uniform(0.3, 0.5) * (b.x1 - b.x0)))
    ay = max(2, int(rng.uniform(0.25, 0.45) * (b.y1 - b.y0)))
    cv.ellipse(blot, (cx, cy), (ax, ay), rng.uniform(0, 180), 0, 360, 255, -1)
    blot = cv.GaussianBlur(blot, (5, 5), 0)
    alpha = blot.astype(np.float32) / 255.0
    out = (gray.astype(np.float32) * (1 - alpha) + ink_val * alpha).astype(np.uint8)
    region = (blot > 127)[b.y0:b.y1, b.x0:b.x1]
    box_area = max((b.x1 - b.x0) * (b.y1 - b.y0), 1)
    return out, region.sum() / box_area * 1.6   # area fraction, scaled ~ to ink terms


def _damage(img: np.ndarray, src: np.ndarray, mask: np.ndarray, bright: bool,
            b: CharBox, boxes: List[CharBox], dtype: str,
            rng: random.Random) -> Tuple[np.ndarray, float]:
    """Apply one defect to char `b` of `img`. `src` is the undamaged crop, so an
    overprint never stamps a char that an earlier pass already damaged."""
    if dtype == "stroke_break":
        return _stroke_break(img, mask, b, rng)
    if dtype == "dot_dropout":
        return _dot_dropout(img, mask, b, rng)
    if dtype == "char_missing":
        return _char_missing(img, mask, b, rng)
    if dtype == "overprint":
        return _overprint(img, mask, b, rng, boxes, bright, src=src)
    if dtype == "blot":
        return _blot(img, mask, b, rng, bright)
    raise ValueError(dtype)


def make_defect(gray: np.ndarray, label: str, boxes: List[CharBox], dtype: str,
                rng: random.Random, max_chars: int = 1,
                ) -> Optional[Tuple[np.ndarray, str, str, float, str]]:
    """Damage a random 1..max_chars distinct alnum chars with the same defect
    type. Returns (image, new_label, chars, severity, final_type), or None when
    every draw came out too light to call a defect.

    A char whose damage is too light is put back as it was (never labelled
    either way — a half-damaged char is exactly the ambiguous case), so only
    chars that clear MIN_INK_REMOVED change the label."""
    cands = [b for b in boxes if b.char.isalnum() and (b.x1 - b.x0) >= 4]
    if not cands:
        return None
    mask, bright = _ink_mask(gray)
    cands = [b for b in cands if _box_ink(mask, b).any()]
    if not cands:
        return None
    k = rng.randint(1, max(1, min(max_chars, len(cands))))
    targets = rng.sample(cands, k)

    img = gray
    hits: List[Tuple[CharBox, str, float]] = []
    for b in targets:
        out, sev = _damage(img, gray, mask, bright, b, boxes, dtype, rng)
        if sev < MIN_INK_REMOVED:
            continue
        # A char whose ink is (almost) all erased IS a missing char, whatever
        # generator got it there (dot_dropout on a solid-stroke font erases
        # whole glyphs). Labelling it □ would teach two answers for one picture.
        t = "char_missing" if dtype in ("stroke_break", "dot_dropout") and sev >= 0.9 else dtype
        img = out
        hits.append((b, t, float(min(sev, 1.0))))
    if not hits:
        return None

    chars = list(label)
    for b, t, _ in sorted(hits, key=lambda h: h[0].idx, reverse=True):
        if t == "char_missing":
            del chars[b.idx]
        else:
            chars[b.idx] = REJECT_CHAR
    final_type = "char_missing" if all(t == "char_missing" for _, t, _ in hits) else dtype
    hit_chars = "".join(b.char for b, _, _ in sorted(hits, key=lambda h: h[0].idx))
    severity = sum(sev for _, _, sev in hits) / len(hits)
    return img, "".join(chars), hit_chars, severity, final_type


# ── dataset generation ───────────────────────────────────────────────────────


def generate(project_id: str, items: Sequence, split_of: Dict[str, str], loc: Localiser,
             per_image: int = 1, types: Sequence[str] = DEFECT_TYPES, seed: int = 0,
             localiser_label: str = "", max_text_length: int = 25,
             max_chars: int = 1,
             on_progress: Optional[Callable[[int, int], None]] = None) -> Dict:
    """Write synthetic defect images for every item the localiser can place.

    Every variant records its source item and that item's split. Training only
    uses variants of train items and evaluation only variants of test items, so
    a test crop never leaks into training through its damaged copy.
    """
    types = [t for t in types if t in DEFECT_TYPES] or list(DEFECT_TYPES)
    rng = random.Random(seed)
    clear(project_id)
    out_dir = synth_dir(project_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries, n_localised, n_unreadable = [], 0, 0
    by_type = {t: 0 for t in types}
    for n_done, item in enumerate(items):
        if on_progress:
            on_progress(n_done, len(items))
        src = dataset_fs.image_abs_path(project_id, item.image_path)
        gray = cv.imread(str(src), cv.IMREAD_GRAYSCALE)
        if gray is None or gray.shape[0] < 8:
            n_unreadable += 1
            continue
        boxes = char_boxes(gray, item.gt_text, loc)
        if not boxes:
            continue
        n_localised += 1
        for k in range(per_image):
            for _ in range(4):   # a draw can come out too light; retry a few times
                dtype = rng.choice(types)
                got = make_defect(gray, item.gt_text, boxes, dtype, rng, max_chars=max_chars)
                if got is None:
                    continue
                img, new_label, ch, sev, dtype = got
                if not new_label.strip() or len(new_label) > max_text_length:
                    break
                name = f"{item.id}_{k}_{dtype}.png"
                cv.imwrite(str(out_dir / name), img)
                entries.append({
                    "file": f"synth/{name}",
                    "label": new_label,
                    "source_id": item.id,
                    "source_label": item.gt_text,
                    "source_image": item.image_path,
                    "split": split_of.get(item.id, "train"),
                    "type": dtype,
                    "char": ch,
                    "n_chars": len(ch),
                    "severity": round(sev, 3),
                    "w": int(img.shape[1]),
                    "h": int(img.shape[0]),
                })
                by_type[dtype] = by_type.get(dtype, 0) + 1
                break

    manifest = {
        "reject_char": REJECT_CHAR,
        "per_image": per_image,
        "max_chars": max_chars,
        "types": list(types),
        "seed": seed,
        "localiser": localiser_label,
        "n_items": len(items),
        "n_localised": n_localised,
        "n_unreadable": n_unreadable,
        "n_train": sum(e["split"] == "train" for e in entries),
        "n_test": sum(e["split"] == "test" for e in entries),
        "by_type": by_type,
        "entries": entries,
    }
    manifest_path(project_id).write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest


def summary(manifest: Optional[Dict]) -> Optional[Dict]:
    if not manifest:
        return None
    return {k: v for k, v in manifest.items() if k != "entries"}


def is_rejected(pred: str, source_label: str) -> bool:
    """Would the line FAIL this read? Mirrors compare_texts(): letters+digits
    only, case-insensitive. A '□' is not alnum, so it drops out here — and the
    real char it replaced is missing, so the strings differ."""
    norm = lambda s: "".join(c for c in s if c.isalnum()).upper()
    return norm(pred) != norm(source_label)
