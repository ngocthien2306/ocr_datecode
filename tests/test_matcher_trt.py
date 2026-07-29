#!/usr/bin/env python3
"""
End-to-end test of the SuperPoint+LightGlue TensorRT matcher pipeline
(ai_services/camera_management/matchers/) against the rebuilt engine.

Uses two real photos of the same bottle cap (different rotation) as a
template/target pair -- a realistic case for the registration matcher, not
just a load/warm-up smoke test.

Usage:
    conda activate vision
    python tests/test_matcher_trt.py
    python tests/test_matcher_trt.py --template test_image/bottle1.jpg --target test_image/bottle3.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "ai_services"))

from inference_engine_shared import SuperPointEngineTRT, TemplateConfig

DEFAULT_ENGINE = os.path.join(REPO_ROOT, "weights", "pipeline_fp16_dynamic_300_300.engine")


def make_template(path: str) -> TemplateConfig:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    full_bbox = {
        "type": "template",
        "points": [[0, 0], [w, 0], [w, h], [0, h]],
        "annotation_index": 0,
    }
    return TemplateConfig(
        template_path=path,
        template_img=img,
        template_gray=gray,
        template_bbox=full_bbox,
        other_bboxes=[],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default=os.path.join(REPO_ROOT, "test_image", "bottle1.jpg"))
    ap.add_argument("--target", default=os.path.join(REPO_ROOT, "test_image", "bottle2.jpg"))
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    args = ap.parse_args()

    print(f"engine:   {args.engine}")
    print(f"template: {args.template}")
    print(f"target:   {args.target}\n")

    engine = SuperPointEngineTRT.get_instance(args.engine, verbose=True)

    template = make_template(args.template)
    target_img = cv2.imread(args.target)
    if target_img is None:
        raise FileNotFoundError(args.target)

    result = engine.match_single(target_img, template)

    print("\n=== result ===")
    print(f"success:        {result.get('success')}")
    print(f"confidence:     {result.get('confidence')}")
    print(f"inliers:        {result.get('inliers')}")
    print(f"total_matches:  {result.get('total_matches')}")
    hom = result.get("homography")
    print(f"homography:\n{np.array(hom) if hom is not None else None}")
    print(f"timings (ms):   {result.get('timings')}")


if __name__ == "__main__":
    main()
