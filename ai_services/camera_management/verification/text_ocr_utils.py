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
