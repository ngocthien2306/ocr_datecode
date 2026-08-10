"""
ONNX inference với bbox 2D per-character qua SMTR cross-attention map.

Cần model ONNX export từ tools/export_smtr_onnx_attn.py (3 outputs:
gtc_logits, ctc_logits, attn_maps).

Usage:
    python tools/infer_rec_onnx_attn.py --img ocr1.png --vis --verbose
    python tools/infer_rec_onnx_attn.py --img ./images/ --save-json bboxes.json
"""

import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

from infer_rec_onnx import (
    SMTRLabelDecode,
    CTCLabelDecode,
    preprocess,
    get_image_paths,
)
from infer_rec_onnx_charbbox import ctc_align_chars, CTC_STRIDE

ONNX_MODEL = './export_attn/rec_smtr_attn_fp16.onnx'
DICT_PATH = './tools/utils/EN_symbol_dict.txt'

# FP16 models produced by onnxconverter_common insert "InsertedPrecisionFreeCast_*"
# nodes around RMSNorm. ORT's SimplifiedLayerNormFusion (extended-level optimizer)
# trips over them -> "Attempting to get index by a name which does not exist".
# Disabling just that fusion pass keeps all other graph optimizations.
_DISABLED_OPTIMIZERS = ['SimplifiedLayerNormFusion']


def make_session(model_path, providers):
    """Create an ORT session, working around the FP16 SimplifiedLayerNorm bug."""
    so = ort.SessionOptions()
    try:
        return ort.InferenceSession(
            model_path, so, providers=providers,
            disabled_optimizers=_DISABLED_OPTIMIZERS,
        )
    except TypeError:
        # Older onnxruntime without `disabled_optimizers`: fall back to BASIC level,
        # which does not run the extended-level fusion that crashes.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        return ort.InferenceSession(model_path, so, providers=providers)


def detect_text_y_band(img_bgr, energy_thr=0.3):
    """Tìm dải Y chứa text bằng Sobel-Y projection (single-line text).

    Returns (y0, y1) trên ảnh gốc. Fallback (0, H) nếu không detect được.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    profile = sy.sum(axis=1)                                     # [h]
    k = max(3, h // 30) | 1                                      # odd kernel
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, k), 0).flatten()
    if profile.max() <= 1e-6:
        return 0, h
    thr = profile.max() * energy_thr
    high = profile >= thr
    # Lấy run liên tục dài nhất
    runs, cur = [], None
    for i, v in enumerate(high):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            runs.append((cur, i)); cur = None
    if cur is not None:
        runs.append((cur, len(high)))
    if not runs:
        return 0, h
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    y0, y1 = runs[0]
    # Sanity: dải text hợp lý từ 20%–95% chiều cao
    if y1 - y0 < h * 0.2 or y1 - y0 > h * 0.98:
        return 0, h
    # Pad nhẹ cho đỡ cắt nét
    pad = max(2, h // 30)
    return max(0, int(y0) - pad), min(h, int(y1) + pad)


def heatmap_to_bbox_cc(heat, threshold_ratio=0.5):
    """Threshold + giữ connected component CHỨA peak.

    Tránh bbox bị kéo ra cả ảnh khi heatmap có nhiễu rải rác.
    """
    if heat.max() <= 0:
        return None
    thr = heat.max() * threshold_ratio
    mask = (heat >= thr).astype(np.uint8)
    if not mask.any():
        return None
    # Tìm component chứa peak
    py, px = np.unravel_index(int(heat.argmax()), heat.shape)
    n_lab, lab = cv2.connectedComponents(mask, connectivity=8)
    peak_lab = lab[py, px]
    if peak_lab == 0:
        return None
    cc_mask = (lab == peak_lab)
    ys, xs = np.where(cc_mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def smtr_peak_x_ranges(attn_maps, smtr_steps, w_orig):
    """Lấy X từ SMTR attention peak (X-projection) cho mỗi char step.

    Khác CTC: attention biết vị trí từng char theo thứ tự decoder, kể cả khi
    có space trong ảnh. Bbox X = midpoint giữa các peak liền kề.
    """
    W_ = attn_maps.shape[2]
    peaks = []
    for j in smtr_steps:
        heat = attn_maps[j].astype(np.float32)
        x_profile = heat.max(axis=0)                          # [W']
        peaks.append(int(x_profile.argmax()))
    n = len(peaks)
    ranges = []
    sx = w_orig / W_
    for i, p in enumerate(peaks):
        left = 0 if i == 0 else (peaks[i - 1] + p) / 2
        right = W_ if i == n - 1 else (p + peaks[i + 1]) / 2
        ranges.append((int(left * sx), int(right * sx), int(p * sx + sx / 2)))
    return ranges


def attn_to_char_bboxes_hybrid(
    attn_maps, gtc_logits, ctc_logits, img_bgr,
    smtr_decode, ctc_decode,
    resized_w,
    threshold_ratio=0.5,
    x_source='auto',   # 'auto' | 'ctc' | 'attn'
):
    """X từ CTC hoặc SMTR attention, Y từ Sobel-Y projection (1 dải toàn dòng).

    x_source:
      - 'ctc'  : CTC alignment (chính xác khi không có space).
      - 'attn' : SMTR attention peak (handle space tốt hơn).
      - 'auto' : ctc nếu CTC↔SMTR text khớp, ngược lại attn.
    """
    h_orig, w_orig = img_bgr.shape[:2]
    pred_idx = gtc_logits.argmax(-1)
    pred_prob = gtc_logits.max(-1)

    smtr_chars, smtr_steps = [], []
    for j in range(len(pred_idx)):
        try:
            ch = smtr_decode.character[int(pred_idx[j])]
        except Exception:
            continue
        if ch == '</s>':
            break
        if ch in ('<s>', '<pad>'):
            continue
        smtr_chars.append(ch)
        smtr_steps.append(j)

    ctc_chars = ctc_align_chars(
        ctc_logits[None], ctc_decode, w_orig, resized_w, stride=CTC_STRIDE
    )
    text_match = (
        len(ctc_chars) == len(smtr_chars)
        and all(c['char'] == s for c, s in zip(ctc_chars, smtr_chars))
    )

    if x_source == 'auto':
        x_source = 'ctc' if text_match else 'attn'
    used = x_source

    if x_source == 'attn':
        attn_ranges = smtr_peak_x_ranges(attn_maps, smtr_steps, w_orig)

    y0_band, y1_band = detect_text_y_band(img_bgr)

    chars = []
    for k, (ch, j) in enumerate(zip(smtr_chars, smtr_steps)):
        heat = attn_maps[j].astype(np.float32)
        if x_source == 'ctc' and text_match:
            x0 = ctc_chars[k]['x_start']
            x1 = ctc_chars[k]['x_end']
        else:
            x0, x1, _peak_x = attn_ranges[k]

        chars.append({
            'char': ch,
            'x0': x0, 'y0': y0_band, 'x1': x1, 'y1': y1_band,
            'conf': float(pred_prob[j]),
            'heat_max': float(heat.max()),
            'bbox_src': used,
        })
    return chars, used, (y0_band, y1_band), text_match


def draw_bboxes(img_bgr, chars):
    vis = img_bgr.copy()
    palette = [
        (0, 200, 0), (0, 165, 255), (255, 0, 200),
        (200, 200, 0), (0, 100, 255), (200, 100, 200),
    ]
    for i, c in enumerate(chars):
        color = palette[i % len(palette)]
        cv2.rectangle(vis, (c['x0'], c['y0']), (c['x1'], c['y1']), color, 2)
        label = c['char']
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(c['y0'] - 3, th + 2)
        cv2.rectangle(
            vis, (c['x0'], ty - th - 2), (c['x0'] + tw + 4, ty + 2), color, -1
        )
        cv2.putText(
            vis, label, (c['x0'] + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return vis


def draw_heatmaps(img_bgr, attn_maps, chars):
    """Vẽ heatmap overlay cho từng char (grid)."""
    h, w = img_bgr.shape[:2]
    n = len(chars)
    if n == 0:
        return img_bgr.copy()
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, c in enumerate(chars):
        # Need access to original heatmap by step idx; chars only knows post-decoded.
        # We'll re-look-up by recovering j: chars are in order of valid decode step.
        # Simpler: caller passes attn_maps trimmed in order.
        heat = attn_maps[i].astype(np.float32)
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.5, heat_color, 0.5, 0)
        cv2.rectangle(overlay, (c['x0'], c['y0']), (c['x1'], c['y1']),
                      (255, 255, 255), 2)
        cv2.putText(overlay, c['char'], (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        r, col = divmod(i, cols)
        canvas[r * h:(r + 1) * h, col * w:(col + 1) * w] = overlay
    return canvas


def run(img_path, sess, gtc_decode, ctc_decode, vis_dir=None,
        verbose=False, threshold_ratio=0.5, save_heatmap=False,
        x_source='auto'):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {img_path}')
    h_orig, w_orig = img_bgr.shape[:2]
    inp = preprocess(img_bgr)
    resized_w = inp.shape[3]

    gtc_logits, ctc_logits, attn_maps = sess.run(None, {'image': inp})

    gtc_text, gtc_score = gtc_decode(gtc_logits, torch_tensor=False)[0]
    ctc_text, ctc_score = ctc_decode(ctc_logits, torch_tensor=False)[0]

    chars, used, y_band, text_match = attn_to_char_bboxes_hybrid(
        attn_maps[0], gtc_logits[0], ctc_logits[0], img_bgr,
        gtc_decode, ctc_decode, resized_w,
        threshold_ratio=threshold_ratio,
        x_source=x_source,
    )

    if verbose:
        print(f'  gtc: {gtc_text!r} ({gtc_score:.4f})')
        print(f'  ctc: {ctc_text!r} ({ctc_score:.4f})')
        print(f'  text_match (CTC==SMTR): {text_match}')
        print(f'  attn shape: {attn_maps.shape}  feat HxW: '
              f'{attn_maps.shape[2]}x{attn_maps.shape[3]}')
        print(f'  x_source: {used}  Y band: [{y_band[0]}, {y_band[1]}]  (H={h_orig})')
        for c in chars:
            print(f"    {c['char']!r:>4}  "
                  f"({c['x0']:4d},{c['y0']:3d})-({c['x1']:4d},{c['y1']:3d})  "
                  f"conf={c['conf']:.3f}  heat_max={c['heat_max']:.3f}")

    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        vis = draw_bboxes(img_bgr, chars)
        out_path = os.path.join(
            vis_dir, os.path.splitext(os.path.basename(img_path))[0] + '_attn_bbox.png'
        )
        cv2.imwrite(out_path, vis)

        if save_heatmap:
            # Map mỗi char về step index trong attn_maps (skip <s>/<pad>, dừng EOS)
            pred_idx = gtc_logits[0].argmax(-1)
            char_steps = []
            for j, idx in enumerate(pred_idx):
                try:
                    ch = gtc_decode.character[int(idx)]
                except Exception:
                    continue
                if ch == '</s>':
                    break
                if ch in ('<s>', '<pad>'):
                    continue
                char_steps.append(j)
            heat_attn = attn_maps[0][char_steps] if char_steps else attn_maps[0][:0]
            hm = draw_heatmaps(img_bgr, heat_attn, chars)
            hm_path = os.path.join(
                vis_dir, os.path.splitext(os.path.basename(img_path))[0] + '_heatmap.png'
            )
            cv2.imwrite(hm_path, hm)

    text, score = (
        (ctc_text, ctc_score) if ctc_score >= gtc_score
        else (gtc_text, gtc_score)
    )
    return text, score, chars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True)
    parser.add_argument('--model', default=ONNX_MODEL)
    parser.add_argument('--dict', default=DICT_PATH)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis-dir', default='./char_attn_vis')
    parser.add_argument('--save-heatmap', action='store_true',
                        help='Cũng lưu lưới heatmap overlay từng char')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Ngưỡng heat (× max) để tạo bbox (default 0.5)')
    parser.add_argument('--x-source', choices=['auto', 'ctc', 'attn'],
                        default='auto',
                        help='Nguồn X-bbox: ctc | attn | auto (default)')
    parser.add_argument('--save-json', default=None)
    args = parser.parse_args()

    providers = (
        ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if args.gpu else ['CPUExecutionProvider']
    )
    sess = make_session(args.model, providers)

    gtc_decode = SMTRLabelDecode(
        character_dict_path=args.dict, use_space_char=True, next_mode=True
    )
    ctc_decode = CTCLabelDecode(
        character_dict_path=args.dict, use_space_char=True
    )

    results = {}
    for img_path in get_image_paths(args.img):
        text, score, chars = run(
            img_path, sess, gtc_decode, ctc_decode,
            vis_dir=args.vis_dir if args.vis else None,
            verbose=args.verbose,
            threshold_ratio=args.threshold,
            save_heatmap=args.save_heatmap,
            x_source=args.x_source,
        )
        print(f'{img_path}\t{text}\t{score:.4f}\t{len(chars)} chars')
        results[img_path] = {
            'text': text, 'score': float(score), 'chars': chars,
        }

    if args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'Saved → {args.save_json}')


if __name__ == '__main__':
    main()
