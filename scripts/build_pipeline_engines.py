#!/usr/bin/env python3
"""
Build TensorRT engines for the SuperPoint+LightGlue matching pipeline in
weights/, using the TensorRT Python Builder API directly (no trtexec binary
needed -- works with the pip-installed tensorrt==11.1.0 in the `vision`
conda env, which does not ship trtexec).

Replaces the old trtexec-based weights/build.sh (trtexec isn't available
with the pip TensorRT install). Same ONNX source, engine name and profile
shapes as that script.

Usage:
    conda activate vision
    python scripts/build_pipeline_engines.py --list
    python scripts/build_pipeline_engines.py --model small_300
    python scripts/build_pipeline_engines.py --all
"""
import argparse
import os

from trt_build_utils import build_engine, parse_shape, run_spec

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")

# name -> onnx/engine filenames (relative to WEIGHTS_DIR), input tensor name,
# and optimization profile shapes as (batch, channels, height, width).
# images is single-channel (grayscale).
#
# Engine filename suffix is _H_W (height then width) -- confirmed from a real
# run's log in docs/new_feature.md: "pipeline_fp16_dynamic_315_560.engine"
# logged "Input shape: (-1, 1, 315, 560)". Batch range (2-8) matches
# weights/build.sh's dynamic-batch convention, reused for the whole family
# since they're all built from the same superpoint_lightglue_small.onnx at
# different fixed resolutions.
MODEL_SPECS = {
    # matches weights/build.sh: batch 2 (1 template+1 target) .. 8 (4+4), 300x300
    "small_300": dict(
        onnx="superpoint_lightglue_small.onnx",
        engine="pipeline_fp16_dynamic_300_300.engine",
        input_name="images",
        min=(2, 1, 300, 300), opt=(6, 1, 300, 300), max=(8, 1, 300, 300),
        fp16=True,
    ),
    "small_315_560": dict(
        onnx="superpoint_lightglue_small.onnx",
        engine="pipeline_fp16_dynamic_315_560.engine",
        input_name="images",
        min=(2, 1, 315, 560), opt=(6, 1, 315, 560), max=(8, 1, 315, 560),
        fp16=True,
    ),
    "small_400_1000": dict(
        onnx="superpoint_lightglue_small.onnx",
        engine="pipeline_fp16_dynamic_400_1000.engine",
        input_name="images",
        min=(2, 1, 400, 1000), opt=(6, 1, 400, 1000), max=(8, 1, 400, 1000),
        fp16=True,
    ),
    "small_480_640": dict(
        onnx="superpoint_lightglue_small.onnx",
        engine="pipeline_fp16_dynamic_480_640.engine",
        input_name="images",
        min=(2, 1, 480, 640), opt=(6, 1, 480, 640), max=(8, 1, 480, 640),
        fp16=True,
    ),
    "small_512_512": dict(
        onnx="superpoint_lightglue_small.onnx",
        engine="pipeline_fp16_dynamic_512_512.engine",
        input_name="images",
        min=(2, 1, 512, 512), opt=(6, 1, 512, 512), max=(8, 1, 512, 512),
        fp16=True,
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
    ap.add_argument("--input-name", default="images")
    ap.add_argument("--min", type=parse_shape, help="e.g. 2x1x300x300")
    ap.add_argument("--opt", type=parse_shape, help="e.g. 6x1x300x300")
    ap.add_argument("--max", type=parse_shape, help="e.g. 8x1x300x300")
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = ap.parse_args()

    if args.list:
        for name, spec in MODEL_SPECS.items():
            print(f"{name:12s} {spec['onnx']:32s} -> {spec['engine']}")
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
        run_spec(name, MODEL_SPECS[name], WEIGHTS_DIR)


if __name__ == "__main__":
    main()
