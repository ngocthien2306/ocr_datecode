#!/usr/bin/env python3
"""
Build TensorRT engines for the English OCR recognition models in
languages/english/, using the TensorRT Python Builder API directly
(no trtexec binary needed -- works with the pip-installed
tensorrt==11.1.0 in the `vision` conda env).

These engines are loaded with raw `tensorrt.Runtime().deserialize_cuda_engine()`
by:
    ai_services/camera_management/text_recognizer_trt.py           (legacy)
    ai_services/camera_management/text_recognizer_openocr_trt.py   (openocr)
    ai_services/camera_management/text_recognizer_smtr_trt.py      (smtr)
Engines built by TensorRT 8.x/9.x/10.x are NOT forward compatible with the
11.1.0 runtime now installed, so they must be rebuilt from the .onnx sources.

Usage:
    conda activate vision
    python scripts/build_ocr_engines.py --list
    python scripts/build_ocr_engines.py --model openocr
    python scripts/build_ocr_engines.py --model openocr legacy smtr
    python scripts/build_ocr_engines.py --all

    # custom one-off build
    python scripts/build_ocr_engines.py --onnx foo.onnx --engine foo.engine \
        --input-name input --min 1x3x48x50 --opt 4x3x48x320 --max 16x3x48x2000
"""
import argparse
import os

from trt_build_utils import build_engine, parse_shape, run_spec

LANG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "languages", "english"
)

# name -> onnx/engine filenames (relative to LANG_DIR), input tensor name,
# and optimization profile shapes as (batch, channels, height, width).
#
# width range (50 / 320 / 2000) and openocr batch range (1 / 4 / 16) are
# taken from this repo's own prior build commands:
#   desktop/docs/convert_to_tensorrt.sh, TENSORRT_GUIDE.md  (legacy rec.onnx)
#   text_recognizer_openocr_trt.py docstring                (openocr batch 1-16)
MODEL_SPECS = {
    "openocr": dict(
        onnx="openocr_rec_model.onnx",
        engine="openocr_rec_model_batch.engine",
        input_name="input",
        min=(1, 3, 48, 50), opt=(4, 3, 48, 320), max=(16, 3, 48, 2000),
        fp16=True,
    ),
    "legacy": dict(
        onnx="rec.onnx",
        engine="rec.engine",
        input_name="x",
        min=(1, 3, 48, 50), opt=(1, 3, 48, 320), max=(1, 3, 48, 2000),
        fp16=True,
    ),
    "smtr": dict(
        onnx="rec_smtr_fp16.onnx",
        engine="rec_smtr_fp16.engine",
        input_name="image",
        min=(1, 3, 32, 32), opt=(4, 3, 32, 320), max=(16, 3, 32, 2000),
        fp16=True,
    ),
    "smtr_new": dict(
        onnx="rec_smtr_fp16_new.onnx",
        engine="rec_smtr_fp16_new.engine",
        input_name="image",
        min=(1, 3, 32, 32), opt=(4, 3, 32, 320), max=(16, 3, 32, 2000),
        fp16=True,
    ),
    "smtr_attn": dict(
        onnx="rec_smtr_attn_fp16.onnx",
        engine="rec_smtr_attn_fp16.engine",
        input_name="image",
        min=(1, 3, 32, 32), opt=(4, 3, 32, 320), max=(16, 3, 32, 2000),
        fp16=True,
        drop_outputs=["attn_maps"],
        note=(
            "Source ONNX has 3 outputs (gtc_logits, ctc_logits, attn_maps) but "
            "TextRecognizerSMTRTRT (text_recognizer_smtr_trt.py:42) asserts "
            "exactly 2 outputs. Dropping 'attn_maps' from the graph before "
            "building so the engine matches that assert."
        ),
    ),
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", nargs="+", choices=sorted(MODEL_SPECS), help="named model(s) to build")
    ap.add_argument("--all", action="store_true", help="build every known model spec")
    ap.add_argument("--list", action="store_true", help="list known model specs and exit")
    ap.add_argument("--onnx", help="custom ONNX path (use with --engine/--input-name/--min/--opt/--max)")
    ap.add_argument("--engine", help="custom output engine path")
    ap.add_argument("--input-name", default="input")
    ap.add_argument("--min", type=parse_shape, help="e.g. 1x3x48x50")
    ap.add_argument("--opt", type=parse_shape, help="e.g. 4x3x48x320")
    ap.add_argument("--max", type=parse_shape, help="e.g. 16x3x48x2000")
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = ap.parse_args()

    if args.list:
        for name, spec in MODEL_SPECS.items():
            print(f"{name:12s} {spec['onnx']:28s} -> {spec['engine']}")
            if spec.get("note"):
                print(f"             ⚠️  {spec['note']}")
        return

    if args.onnx:
        if not (args.engine and args.min and args.opt and args.max):
            ap.error("--onnx requires --engine --min --opt --max")
        build_engine(args.onnx, args.engine, args.input_name, args.min, args.opt, args.max, fp16=args.fp16)
        return

    names = list(MODEL_SPECS) if args.all else (args.model or [])
    if not names:
        ap.error("specify --model NAME [NAME ...], --all, or --onnx/--engine for a custom build")

    for name in names:
        run_spec(name, MODEL_SPECS[name], LANG_DIR)


if __name__ == "__main__":
    main()
