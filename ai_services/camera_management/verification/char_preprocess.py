"""
Char crop preprocessing — remove neighbor-character fragments leaking into
the crop region.

Char bboxes projected from template via SuperPoint occasionally include a
sliver of the adjacent character (e.g. crop of "8" picks up a piece of "4"
on the right). Such fragments are pure noise to the SupCon embedding +
classifier, dragging p_ok down and producing false NGs.

`remove_fragments_local_bg` runs Otsu + connected components, keeps the
component(s) at the crop center (the actual character), and inpaints other
fragments with a local-median background color. Output keeps the same size
as the input, character + in-character defects are preserved verbatim.
"""

from typing import Tuple

import cv2
import numpy as np


def remove_fragments_local_bg(
    crop: np.ndarray,
    dilate_px: int = 4,
    ring_px: int = 12,
    min_area: int = 20,
    radius_ratio: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loại fragment kí tự lân cận khỏi crop ML char.

    Args:
        crop:         BGR (H, W, 3) hoặc grayscale (H, W) uint8.
        dilate_px:    Dilate fragment mask trước khi fill — bao anti-alias.
        ring_px:      Annular ring quanh fragment để sample background color.
        min_area:     Component nhỏ hơn ngưỡng này coi như noise → bỏ qua.
        radius_ratio: Component có centroid trong radius_ratio * min(h, w)
                      kể từ tâm ảnh sẽ được giữ (chữ chính). Còn lại = fragment.

    Returns:
        (output, drop_mask)
          output    — ảnh CÙNG SIZE, fragment được thay bằng local-bg color.
                      Chữ chính + defects bên trong giữ nguyên 100%.
          drop_mask — uint8 (H, W), 255 = pixel bị thay, 0 = giữ nguyên.
    """
    if crop is None or crop.size == 0:
        return crop, np.zeros((0, 0), dtype=np.uint8) if crop is None else np.zeros_like(crop[..., 0] if crop.ndim == 3 else crop)

    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    h, w = bw.shape

    n, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return crop.copy(), np.zeros_like(bw)

    valid = [i for i in range(1, n) if int(stats[i, 4]) >= min_area]
    if not valid:
        return crop.copy(), np.zeros_like(bw)

    cx_img, cy_img = w / 2, h / 2
    R = radius_ratio * min(h, w)

    # KEEP = component có centroid trong R từ tâm ảnh (chữ chính + pieces bị
    # crack/cut split đều gần tâm). Fragment bulk ở rìa → centroid xa → DROP.
    keep_ids = set()
    for i in valid:
        ccx, ccy = float(cents[i][0]), float(cents[i][1])
        if (ccx - cx_img) ** 2 + (ccy - cy_img) ** 2 < R * R:
            keep_ids.add(i)

    # Fallback: nếu không có gì gần tâm → giữ component lớn nhất.
    if not keep_ids:
        keep_ids = {max(valid, key=lambda i: int(stats[i, 4]))}

    # Merge accents (chấm của 'i', 'j', ':', '!'): RẤT NHỎ + sát thân chữ +
    # overlap cột chặt. Thresholds đủ chặt để fragment cỡ trung không lọt.
    kept_mask = np.isin(labels, list(keep_ids))
    kept_cols = kept_mask.any(axis=0)
    kept_rows = np.where(kept_mask.any(axis=1))[0]
    if len(kept_rows) > 0:
        ky0, ky1 = int(kept_rows.min()), int(kept_rows.max())
        for i in valid:
            if i in keep_ids:
                continue
            x, y, cw, ch, area = stats[i]
            if area > 200:
                continue
            cols = kept_cols[x:x + cw]
            if cols.sum() / max(cw, 1) < 0.7:
                continue
            dy = min(abs(int(y) - ky1), abs(int(y + ch) - ky0))
            if dy <= 10:
                keep_ids.add(i)

    drop_ids = [i for i in valid if i not in keep_ids]
    if not drop_ids:
        return crop.copy(), np.zeros_like(bw)

    out = crop.copy()
    final_mask = np.zeros_like(bw)
    k_fill = max(3, int(dilate_px) | 1)
    ker_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_fill, k_fill))
    k_ring = max(3, int(ring_px * 2 + 1) | 1)
    ker_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_ring, k_ring))
    all_ink_dil = cv2.dilate(bw, ker_fill)

    for fid in drop_ids:
        frag = (labels == fid).astype(np.uint8) * 255
        frag_dil = cv2.dilate(frag, ker_fill)
        frag_outer = cv2.dilate(frag, ker_ring)
        # Ring quanh fragment, ngoài ink, ngoài fragment dilated.
        ring = (frag_outer > 0) & (frag_dil == 0) & (all_ink_dil == 0)
        if int(ring.sum()) >= 20:
            if crop.ndim == 3:
                bg = np.median(crop[ring].reshape(-1, 3), axis=0).astype(np.uint8)
            else:
                bg = int(np.median(crop[ring]))
        else:
            # ring quá nhỏ → fallback median toàn vùng không có ink
            free = (bw == 0)
            if int(free.sum()) >= 20 and crop.ndim == 3:
                bg = np.median(crop[free].reshape(-1, 3), axis=0).astype(np.uint8)
            elif int(free.sum()) >= 20:
                bg = int(np.median(crop[free]))
            else:
                bg = np.array([200, 200, 200], dtype=np.uint8) if crop.ndim == 3 else 200

        out[frag_dil > 0] = bg
        final_mask[frag_dil > 0] = 255

    return out, final_mask
