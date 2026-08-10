"""
Cross-check a fine-tuned SMTR model across the three runtimes it passes
through: PyTorch checkpoint -> ONNX -> TensorRT .engine.

Decoding goes through app/services/smtr_runtime/ — the vendored copy of
ai_services' recognizer classes — so a mismatch here is a decode mismatch, not a
difference between two re-implementations. Run check_runtime_parity.py to confirm
the copy still matches what production uses.

Usage:
    python verify_export.py \
        --data ./data_ocr_merged --gt rec_gt_test.txt \
        --onnx ./output/verify_run/export/rec_smtr_fp16.onnx \
        --engine ./output/verify_run/export/rec_smtr_fp16.engine
"""
import argparse
import os
import string
import sys
import time

import cv2

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)

DEFAULT_DICT = os.path.join(__dir__, 'OpenOCR', 'tools', 'utils', 'EN_symbol_dict.txt')


def load_gt(data_dir, gt_file):
    items = []
    with open(os.path.join(data_dir, gt_file), encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            rel, _, label = line.partition('\t')
            items.append((os.path.join(data_dir, rel), label))
    return items


def _normalize(text):
    """OpenOCR RecMetric's normalization (is_filter=True, is_lower=True,
    ignore_space=True): keep letters+digits only, lowercase. The accuracy
    printed by train_rec.py/eval_rec.py is measured on this form, so an
    exported model has to be scored the same way to be comparable — raw
    exact-match is a different (stricter) number."""
    return ''.join(c for c in text if c in string.digits + string.ascii_letters).lower()


def score(preds, items, label):
    """preds: list of (gtc_text, ctc_text). Reports both heads and the
    either-head-correct upper bound, which is what the production candidate
    logic can reach at best."""
    n = len(items)
    for tag, norm in (('exact', lambda s: s), ('normalized', _normalize)):
        gtc_ok = sum(norm(p[0]) == norm(gt) for p, (_, gt) in zip(preds, items))
        ctc_ok = sum(norm(p[1]) == norm(gt) for p, (_, gt) in zip(preds, items))
        any_ok = sum(norm(gt) in (norm(p[0]), norm(p[1])) for p, (_, gt) in zip(preds, items))
        print(f'{label:<12} [{tag:<10}] gtc {gtc_ok}/{n} = {gtc_ok / n:.4f}   '
              f'ctc {ctc_ok}/{n} = {ctc_ok / n:.4f}   '
              f'either {any_ok}/{n} = {any_ok / n:.4f}')
    return gtc_ok / n, ctc_ok / n, any_ok / n


def run_backend(rec, items, batch=8):
    preds, t0 = [], time.perf_counter()
    for i in range(0, len(items), batch):
        imgs = [cv2.imread(p) for p, _ in items[i:i + batch]]
        for r in rec.recognize_batch(imgs):
            preds.append((r[0][0], r[1][0]))
    ms = (time.perf_counter() - t0) * 1000 / len(items)
    return preds, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--gt', default='rec_gt_test.txt')
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--engine', required=True)
    ap.add_argument('--dict', default=DEFAULT_DICT)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--show-diff', type=int, default=10,
                    help='print up to N images where ONNX and TRT disagree')
    args = ap.parse_args()

    items = load_gt(args.data, args.gt)
    print(f'{len(items)} test images from {args.gt}\n')

    from app.services.smtr_runtime.smtr_onnx import TextRecognizerSMTRONNX
    from app.services.smtr_runtime.smtr_trt import TextRecognizerSMTRTRT

    onnx_rec = TextRecognizerSMTRONNX(args.onnx, args.dict, device='cpu')
    onnx_preds, onnx_ms = run_backend(onnx_rec, items, args.batch)

    trt_rec = TextRecognizerSMTRTRT(args.engine, args.dict)
    trt_preds, trt_ms = run_backend(trt_rec, items, args.batch)

    print()
    o = score(onnx_preds, items, 'ONNX fp16')
    t = score(trt_preds, items, 'TensorRT')
    print(f'\nlatency/img: onnx(cpu) {onnx_ms:.1f} ms   trt(gpu) {trt_ms:.1f} ms')

    disagree = [(p, a, b) for (p, _), a, b in zip(items, onnx_preds, trt_preds)
                if a[0] != b[0] or a[1] != b[1]]
    print(f'ONNX vs TRT disagreements: {len(disagree)}/{len(items)}')
    for path, a, b in disagree[:args.show_diff]:
        print(f'  {os.path.basename(path)[:60]}\n'
              f'      onnx gtc={a[0]!r} ctc={a[1]!r}\n'
              f'      trt  gtc={b[0]!r} ctc={b[1]!r}')

    print(f'\nTRT vs ONNX either-head acc delta: {t[2] - o[2]:+.4f}')


if __name__ == '__main__':
    main()
