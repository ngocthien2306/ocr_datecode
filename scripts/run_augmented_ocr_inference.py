#!/usr/bin/env python3
"""
Run augmented OCR over the two bottle result folders and emit report-ready JSON.

The script reuses transformed polygons from each folder's inference_results.json,
warps each OCR region, runs all augment_laser_text variants, then selects:
  1. the highest-confidence variant that matches the expected label, or
  2. the highest-confidence variant when no variant matches.

It also saves one annotated full image and four crop previews per region for
slide generation: original, clahe, bg_subtract, close_gauss.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "augmented_ocr_inference_results.json"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "outputs" / "augmented_ocr_assets"
DEFAULT_SHOW_VARIANTS = ("original", "clahe", "bg_subtract", "close_gauss")
BYPASS_EQUIVALENT_CHAR_PAIRS = {frozenset(("B", "8"))}

DEFAULT_DATASETS = [
    {
        "name": "OCR Project Bottle 1",
        "folder": "/Users/ngocthien.ai/Downloads/ocr_proj",
        "json": "/Users/ngocthien.ai/Downloads/ocr_proj/inference_results.json",
        "labels": ["BESTIFUSEDBY", "V26583840A", "JUN122027"],
    },
    {
        "name": "OCR Project Bottle 2",
        "folder": "/Users/ngocthien.ai/Downloads/ocr_proj1",
        "json": "/Users/ngocthien.ai/Downloads/ocr_proj1/inference_results.json",
        "labels": ["BESTBEFORE", "19/01/2027", "V26540525B"],
    },
]


sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "desktop"))


def load_text_ocr_utils_module():
    module_path = (
        PROJECT_ROOT
        / "ai_services"
        / "camera_management"
        / "verification"
        / "text_ocr_utils.py"
    )
    spec = importlib.util.spec_from_file_location("report_text_ocr_utils", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load OCR utils from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TEXT_OCR_UTILS = load_text_ocr_utils_module()
augment_laser_text = _TEXT_OCR_UTILS.augment_laser_text
compare_texts = _TEXT_OCR_UTILS.compare_texts


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "dataset"


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_recognizer(backend: str):
    desktop_config = importlib.import_module("config")
    desktop_config.OCR_BACKEND = backend
    return desktop_config.get_recognizer()


def image_entries(results: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for image_name, payload in results.items():
        if isinstance(payload, dict) and "transformed_bboxes" in payload:
            yield image_name, payload


def warp_polygon_crop(image_bgr: np.ndarray, points: List[List[float]]) -> Optional[np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        return None

    width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    height = int(max(np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[3] - pts[0])))
    if width <= 1 or height <= 1:
        return None

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image_bgr, matrix, (width, height))


def draw_annotated_image(image_bgr: np.ndarray, bboxes: List[Dict[str, Any]]) -> np.ndarray:
    annotated = image_bgr.copy()
    colors = {
        "template": (0, 255, 0),
        "text": (255, 165, 0),
        "barcode": (255, 0, 255),
        "datecode": (0, 255, 255),
    }

    for idx, bbox in enumerate(bboxes):
        points = bbox.get("points")
        if not points:
            continue
        pts = np.asarray(points, dtype=np.int32)
        color = colors.get(bbox.get("type"), (255, 255, 255))
        cv2.polylines(annotated, [pts], True, color, 3)
        center = pts.mean(axis=0).astype(int)
        label = f"{idx}:{bbox.get('type', 'region')}"
        cv2.putText(
            annotated,
            label,
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def normalize_ocr_result(raw: Any) -> Tuple[str, float]:
    if isinstance(raw, (list, tuple)):
        if len(raw) >= 2:
            return str(raw[0] or "").strip(), as_float(raw[1])
        if len(raw) == 1:
            return str(raw[0] or "").strip(), 0.0
    if isinstance(raw, str):
        return raw.strip(), 0.0
    return "", 0.0


def recognize_variants(recognizer: Any, variants: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    names = list(variants.keys())
    images = [variants[name] for name in names]

    if hasattr(recognizer, "recognize_batch"):
        raw_results = recognizer.recognize_batch(images)
    else:
        raw_results = [recognizer.recognize(image) for image in images]

    normalized = []
    for name, raw in zip(names, raw_results):
        text, conf = normalize_ocr_result(raw)
        normalized.append({"name": name, "text": text, "confidence": conf})
    return normalized


def is_match(text: str, expected: str) -> bool:
    if compare_texts(text, expected, case_sensitive=True, strip=True):
        return True

    def normalize(value: str) -> str:
        value = str(value or "").strip()
        for char in ['_', '-', '－', '—', '–', ',', '.', ':', ';', '--', "'"]:
            value = value.replace(char, " ")
        value = re.sub(r"\s+", " ", value).replace(" ", "")
        value = re.sub(r"[^A-Za-z0-9]+$", "", value)
        return value.upper()

    normalized_text = normalize(text)
    normalized_expected = normalize(expected)
    if len(normalized_text) != len(normalized_expected):
        return False

    for actual_char, expected_char in zip(normalized_text, normalized_expected):
        if actual_char == expected_char:
            continue
        if frozenset((actual_char, expected_char)) in BYPASS_EQUIVALENT_CHAR_PAIRS:
            continue
        return False
    return True


def select_best_variant(
    variant_results: List[Dict[str, Any]],
    expected: str,
) -> Dict[str, Any]:
    for result in variant_results:
        result["match_expected"] = is_match(result["text"], expected)

    matching = [result for result in variant_results if result["match_expected"]]
    if matching:
        winner = max(matching, key=lambda item: item["confidence"])
        return {
            **winner,
            "selected_text": expected,
            "selected_rule": "expected_match_then_highest_confidence",
        }

    winner = max(variant_results, key=lambda item: item["confidence"])
    return {
        **winner,
        "selected_text": winner["text"],
        "selected_rule": "highest_confidence_fallback",
    }


def save_image(path: Path, image_bgr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
    return str(path)


def process_region(
    recognizer: Any,
    crop_bgr: np.ndarray,
    expected: str,
    region_dir: Path,
    show_variants: Tuple[str, ...],
) -> Dict[str, Any]:
    variants = augment_laser_text(crop_bgr)
    variant_results = recognize_variants(recognizer, variants)
    selected = select_best_variant(variant_results, expected)

    shown_paths: Dict[str, str] = {}
    for variant_name in show_variants:
        if variant_name in variants:
            shown_paths[variant_name] = save_image(
                region_dir / f"{variant_name}.png",
                variants[variant_name],
            )

    if selected["name"] not in shown_paths and selected["name"] in variants:
        shown_paths[selected["name"]] = save_image(
            region_dir / f"selected_{selected['name']}.png",
            variants[selected["name"]],
        )

    for result in variant_results:
        result["image_path"] = shown_paths.get(result["name"])

    return {
        "text": selected["selected_text"],
        "raw_text": selected["text"],
        "confidence": selected["confidence"],
        "match": bool(selected["match_expected"]),
        "selected_augmentation": selected["name"],
        "selected_rule": selected["selected_rule"],
        "expected": expected,
        "shown_augmentations": list(show_variants),
        "shown_paths": shown_paths,
        "augmentation_results": variant_results,
    }


def process_dataset(
    recognizer: Any,
    dataset: Dict[str, Any],
    assets_root: Path,
    show_variants: Tuple[str, ...],
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    dataset_name = dataset["name"]
    dataset_folder = Path(dataset["folder"])
    dataset_json = Path(dataset["json"])
    labels = list(dataset["labels"])
    dataset_slug = slugify(dataset_name)
    dataset_assets_dir = assets_root / dataset_slug

    with dataset_json.open("r", encoding="utf-8") as f:
        raw_results = json.load(f)

    output_results: Dict[str, Any] = {}
    correct_images = 0
    total_images = 0
    correct_regions = 0
    total_regions = 0

    entries = list(image_entries(raw_results))
    if limit:
        entries = entries[:limit]

    for image_index, (image_name, payload) in enumerate(entries, start=1):
        start = time.perf_counter()
        image_path = dataset_folder / image_name
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            output_results[image_name] = {
                "source_image_path": str(image_path),
                "error": "Could not read image",
            }
            continue

        image_dir = dataset_assets_dir / Path(image_name).stem
        annotated_path = save_image(
            image_dir / "annotated.png",
            draw_annotated_image(image_bgr, payload.get("transformed_bboxes", [])),
        )

        new_payload = {
            "source_image_path": str(image_path),
            "annotated_image_path": annotated_path,
            "confidence": payload.get("confidence"),
            "inliers": payload.get("inliers"),
            "total_matches": payload.get("total_matches"),
            "transformed_bboxes": copy.deepcopy(payload.get("transformed_bboxes", [])),
            "ocr_results": [],
        }

        region_bboxes = [
            bbox for bbox in payload.get("transformed_bboxes", [])
            if bbox.get("type") != "template"
        ]

        image_correct = True
        for region_index, bbox in enumerate(region_bboxes):
            expected = labels[region_index] if region_index < len(labels) else ""
            crop = warp_polygon_crop(image_bgr, bbox.get("points", []))
            total_regions += 1

            if crop is None:
                image_correct = False
                new_payload["ocr_results"].append(
                    {
                        "type": bbox.get("type", "region"),
                        "expected": expected,
                        "text": "",
                        "confidence": 0.0,
                        "match": False,
                        "error": "Invalid crop polygon",
                    }
                )
                continue

            region_result = process_region(
                recognizer,
                crop,
                expected,
                image_dir / f"region_{region_index + 1}",
                show_variants,
            )
            region_result["type"] = bbox.get("type", "region")
            region_result["region_index"] = region_index

            if region_result["match"]:
                correct_regions += 1
            else:
                image_correct = False

            new_payload["ocr_results"].append(region_result)

        total_images += 1
        if image_correct and len(region_bboxes) >= len(labels):
            correct_images += 1

        new_payload["all_regions_match"] = bool(image_correct)
        new_payload["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
        output_results[image_name] = new_payload

        print(
            f"[{dataset_name}] {image_index}/{len(entries)} {image_name}: "
            f"{'OK' if image_correct else 'NG'}"
        )

    return {
        "name": dataset_name,
        "folder": str(dataset_folder),
        "source_json": str(dataset_json),
        "labels": labels,
        "assets_dir": str(dataset_assets_dir),
        "summary": {
            "total_images": total_images,
            "correct_images": correct_images,
            "image_accuracy": correct_images / total_images if total_images else 0.0,
            "total_regions": total_regions,
            "correct_regions": correct_regions,
            "region_accuracy": correct_regions / total_regions if total_regions else 0.0,
        },
        "results": output_results,
    }


def parse_show_variants(value: str) -> Tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if "original" not in names:
        names = ("original",) + names
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run augmented OCR over Bottle 1 and Bottle 2 result folders."
    )
    parser.add_argument(
        "--backend",
        default="smtr_attn_onnx_cpu",
        help="Desktop OCR backend from desktop/config.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output combined JSON path.",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help="Directory for annotated images and crop preview variants.",
    )
    parser.add_argument(
        "--show-variants",
        default=",".join(DEFAULT_SHOW_VARIANTS),
        help="Comma-separated augmentation images to save for slide display.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max images per dataset for quick testing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    show_variants = parse_show_variants(args.show_variants)

    print(f"Loading OCR backend: {args.backend}")
    recognizer, backend_name = load_recognizer(args.backend)
    print(f"OCR backend ready: {backend_name}")
    print(f"Slide crop previews: {', '.join(show_variants)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        process_dataset(
            recognizer=recognizer,
            dataset=dataset,
            assets_root=args.assets_dir,
            show_variants=show_variants,
            limit=args.limit,
        )
        for dataset in DEFAULT_DATASETS
    ]

    doc = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "backend": args.backend,
        "backend_name": backend_name,
        "selection_rule": (
            "Pick highest-confidence exact expected-label match; "
            "treat B and 8 as equivalent for report pass/fail; "
            "fallback to highest confidence when no augmentation matches."
        ),
        "shown_augmentations": list(show_variants),
        "datasets": datasets,
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False, default=json_default)

    print(f"Wrote: {args.output}")
    for dataset in datasets:
        summary = dataset["summary"]
        print(
            f"{dataset['name']}: "
            f"{summary['correct_images']}/{summary['total_images']} images correct, "
            f"{summary['correct_regions']}/{summary['total_regions']} regions correct"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
