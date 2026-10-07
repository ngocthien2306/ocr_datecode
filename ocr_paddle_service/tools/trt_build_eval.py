"""
Build a TensorRT engine for an exported Paddle rec model and/or score it with
ai_services' own TextRecognizerTRT — run in the `vision` conda env (the one
camera_management_service.py runs in), never in paddle_train.

    python tools/trt_build_eval.py build --onnx rec.onnx --engine rec_fp16.engine
    python tools/trt_build_eval.py eval  --engine rec_fp16.engine --dict dict.txt \
        --list items.tsv            # "<abs image path>\\t<label>" per line

Why a subprocess in that env rather than tensorrt inside the service:
  - a serialized engine only deserializes under the TensorRT version that
    built it, so it must be built by the runtime ai_services actually has;
  - the service process must not own a CUDA context next to Paddle's.

Why the recognizer is imported from ai_services rather than copied: this is
the number that gates a model for the line, so it has to be produced by the
line's own preprocessing + CTC decode, not by a re-implementation of it.
camera_management/__init__.py is bypassed (it pulls in the camera stack);
the two parent packages are registered as bare namespaces so the real
ocr/backends/default_trt.py and its `..ocr_utils` import load unchanged.
If ai_services moves those files, this fails loudly at import.

Prints one JSON object on the last stdout line.
"""
import argparse
import json
import os
import sys
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AI_CM = os.path.join(REPO, "ai_services", "camera_management")

# Same profile as scripts/build_ocr_engines.py 'paddlev6' — batch 1 because
# TextRecognizerTRT runs crops one at a time at their own width.
MIN, OPT, MAX = (1, 3, 48, 50), (1, 3, 48, 320), (1, 3, 48, 2000)


def _build(args):
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from trt_build_utils import build_engine

    t0 = time.time()
    build_engine(args.onnx, args.engine, "x", MIN, OPT, MAX, fp16=True)
    return {
        "engine_path": args.engine,
        "size_mb": round(os.path.getsize(args.engine) / 1e6, 2),
        "build_s": round(time.time() - t0, 1),
    }


def _load_recognizer_cls():
    for name, path in (
        ("camera_management", AI_CM),
        ("camera_management.ocr", os.path.join(AI_CM, "ocr")),
    ):
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod
    from camera_management.ocr.backends.default_trt import TextRecognizerTRT
    return TextRecognizerTRT


def _eval(args):
    import cv2

    TextRecognizerTRT = _load_recognizer_cls()
    rec = TextRecognizerTRT(args.engine, args.dict)
    rows = []
    with open(args.list, encoding="utf-8") as f:
        for line in f:
            path, _, label = line.rstrip("\n").partition("\t")
            rows.append((path, label))

    preds = []
    t0 = time.perf_counter()
    for path, _ in rows:
        img = cv2.imread(path)
        if img is None:
            preds.append({"text": "", "confidence": 0.0, "missing": True})
            continue
        text, conf = rec.recognize(img)
        preds.append({"text": text, "confidence": float(conf)})
    ms = (time.perf_counter() - t0) * 1000 / max(len(rows), 1)
    return {"preds": preds, "ms_per_image": round(ms, 2), "num_classes": len(rec.char_list)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--onnx", required=True)
    b.add_argument("--engine", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--engine", required=True)
    e.add_argument("--dict", required=True)
    e.add_argument("--list", required=True)
    args = ap.parse_args()

    out = _build(args) if args.cmd == "build" else _eval(args)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
