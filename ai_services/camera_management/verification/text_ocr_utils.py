"""
OCR Text Processing Utilities

Pure helpers used by TextVerificationService:
  - augment_laser_text: 5 visual enhancement variants for laser-engraved text
  - apply_text_corrections: hard-coded OCR substitution rules
  - pick_winning_candidate: choose best candidate from OCR candidate list
  - calculate_text_similarity: SequenceMatcher ratio after normalization
"""

from difflib import SequenceMatcher

import cv2
import numpy as np

from ..ocr_utils import compare_texts


AUGMENT_SIMILARITY_THRESHOLD = 0.70


def augment_laser_text(img_bgr: np.ndarray) -> dict:
    """
    Generate 5 enhanced versions optimized for difficult backgrounds (laser-engraved, low contrast).
    Copied from tests/test_trt_inference.py.

    Returns:
        dict with keys: 'original', 'clahe', 'bg_subtract', 'unsharp_clahe', 'tophat'
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    results = {'original': img_bgr.copy()}

    # 1. CLAHE – adaptive local contrast enhancement
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    results['clahe'] = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 2. Background subtraction – amplify residual text signal
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    diff_amp = cv2.convertScaleAbs(cv2.subtract(gray, bg), alpha=8)
    results['bg_subtract'] = cv2.cvtColor(diff_amp, cv2.COLOR_GRAY2BGR)

    # 3. Unsharp masking + CLAHE
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 2.0, blurred, -1.0, 0)
    results['unsharp_clahe'] = cv2.cvtColor(clahe.apply(unsharp), cv2.COLOR_GRAY2BGR)

    # 4. Morphological TOPHAT – extracts bright regions smaller than kernel
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 20))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_morph)
    tophat_eq = cv2.convertScaleAbs(tophat, alpha=6)
    results['tophat'] = cv2.cvtColor(clahe.apply(tophat_eq), cv2.COLOR_GRAY2BGR)

    return results


def augment_dot_matrix_text(img_bgr: np.ndarray) -> dict:
    """
    5 versions tuned for CIJ ink-jet **dot-matrix** text (mỗi ký tự = lưới chấm rời).
    Mục tiêu: gộp các chấm gần nhau thành nét liền để OCR không nhầm chấm thưa
    thành ký tự sai (vd V dot-matrix → U vì đáy V chỉ 1-2 chấm dễ trông như cong).

    Khác `augment_laser_text` ở chỗ KHÔNG sharpen / KHÔNG subtract background —
    các op đó càng làm khoảng cách giữa chấm rõ hơn → đọc tệ hơn. Thay vào đó
    dùng blur / morph close / erode / downup để TRIỆT TIÊU khe trống.

    Returns:
        dict keys: 'original', 'gauss_merge', 'close_bridge',
                   'fatten_strokes', 'downup_antialias'
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    results = {'original': img_bgr.copy()}

    # 1. Gaussian merge — blur σ=1.8 đủ mạnh để chấm liền kề hợp thành blob stroke,
    # CLAHE phục hồi tương phản tổng thể sau blur.
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.8)
    results['gauss_merge'] = cv2.cvtColor(clahe.apply(blurred), cv2.COLOR_GRAY2BGR)

    # 2. Morph CLOSE trên ảnh inverted — text dark on light → invert để text bright,
    # close (dilate+erode) bridge khe LIGHT (vốn là gap giữa chấm) bên trong stroke,
    # rồi invert về. Kernel 3×5 ưu tiên dọc — phù hợp các chấm xếp dọc trong chữ.
    inv = cv2.bitwise_not(gray)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel_close)
    closed_back = cv2.bitwise_not(closed)
    results['close_bridge'] = cv2.cvtColor(clahe.apply(closed_back), cv2.COLOR_GRAY2BGR)

    # 3. Fatten dark strokes — median 3 xoá nhiễu muối tiêu, erode 2×2 ellipse
    # làm dark pixel "lan" ra → các chấm phình + tự overlap → thành nét.
    median = cv2.medianBlur(gray, 3)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    fattened = cv2.erode(median, kernel_erode)
    results['fatten_strokes'] = cv2.cvtColor(clahe.apply(fattened), cv2.COLOR_GRAY2BGR)

    # 4. Down→up anti-alias — downscale 0.5× rồi upscale lại kích thước gốc:
    # bước down area-average gộp chấm thành mức xám trung gian, bước up linear
    # bôi mịn → chấm biến mất nhưng stroke vẫn nhận diện được.
    new_w = max(1, w // 2)
    new_h = max(1, h // 2)
    small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    results['downup_antialias'] = cv2.cvtColor(clahe.apply(restored), cv2.COLOR_GRAY2BGR)

    return results


def apply_text_corrections(text: str) -> str:
    """Apply common OCR correction rules before comparison."""
    if "Pt" in text:
        text = text.replace("Pt", "PL")
    if "USsed" in text:
        text = text.replace("USsed", "Used")
    if "Iif" in text:
        text = text.replace("Iif", "If")
    return text


def pick_winning_candidate(
    candidates,
    conf_threshold: float,
    use_char_conf_check: bool,
    expected_text: str,
    case_sensitive: bool = True,
):
    """
    Choose the best candidate from an OCR candidate list.

    Returns:
        (match: bool, recognized_text: str, confidence: float, char_confs)
    """
    passing = []
    for text, conf, cc in candidates:
        if conf < conf_threshold:
            continue
        if use_char_conf_check and cc is not None:
            if any(c.isalnum() and cf < conf_threshold for c, cf in cc):
                continue
        passing.append((apply_text_corrections(text.strip()), conf, cc))

    if not passing:
        best = max(candidates, key=lambda x: x[1])
        return False, apply_text_corrections(best[0].strip()), best[1], best[2]

    matched = [
        (t, c, cc) for t, c, cc in passing
        if compare_texts(t, expected_text, case_sensitive=case_sensitive, strip=True)
    ]
    if matched:
        winner = max(matched, key=lambda x: x[1])
        return True, expected_text[:], winner[1], winner[2]

    best = max(passing, key=lambda x: x[1])
    return False, best[0], best[1], best[2]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two texts using SequenceMatcher.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity ratio (0.0 - 1.0)
    """
    # Normalize: strip whitespace, remove internal spaces, and convert to lowercase
    special_chars_to_space = ['_', '-', '－', '—', '–', ',', '.', ':', ';', '--']
    for char in special_chars_to_space:
        text1 = text1.replace(char, " ")
        text2 = text2.replace(char, " ")

    text1_norm = text1.strip().replace(" ", "").lower()
    text2_norm = text2.strip().replace(" ", "").lower()

    ratio = SequenceMatcher(None, text1_norm, text2_norm).ratio()
    return ratio


def levenshtein_distance(a: str, b: str) -> int:
    """
    Character-level edit distance. O(len(a) * len(b)) DP — cheap for OCR strings (<50 chars).
    Used by tolerance check to count exact #-of-chars differing.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                curr[j - 1] + 1,        # insert
                prev[j] + 1,            # delete
                prev[j - 1] + cost,     # substitute
            )
        prev = curr
    return prev[-1]
