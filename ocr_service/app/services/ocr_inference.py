"""
Score exported artifacts using ai_services' OWN recognizer classes.

This module deliberately imports from ../ai_services rather than shipping a
second copy of the SMTR pre/post-processing. anomaly_service made the opposite
call for its TensorRT helper (ship a copy, stay independently deployable), and
that is right for a build utility — but not here. A second copy of the decode
path is exactly how a model scores 0.96 in the studio and 0.43 in production:
the numbers this service reports are only meaningful if they come from the code
that will actually read the labels on the line. Both services run on the same
workstation by design, the same way anomaly_service reads backend/uploads
directly.

Consequence to keep in mind: changing smtr_utils or the backends in ai_services
changes this service's reported accuracy. That is the intent.
"""
import logging
import string
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2 as cv

from app.core.config import AI_SERVICES_DIR, CHARACTER_DICT_PATH

logger = logging.getLogger(__name__)

_RECOGNIZER_CACHE: Dict[str, object] = {}


def _stub_camera_management_package() -> None:
    """Make camera_management.ocr.* importable without executing the package
    __init__, which imports pypylon (the Basler SDK — present only where the
    cameras are). The stub keeps __path__ so submodule imports still resolve."""
    if str(AI_SERVICES_DIR) not in sys.path:
        sys.path.insert(0, str(AI_SERVICES_DIR))
    for name, rel in (("camera_management", ("camera_management",)),
                      ("camera_management.ocr", ("camera_management", "ocr"))):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = [str(AI_SERVICES_DIR.joinpath(*rel))]
        sys.modules[name] = mod


def load_recognizer(model_path: Path, engine: str, dict_path: Optional[Path] = None):
    """Cached recognizer for one artifact. engine is 'onnx' or 'tensorrt'.

    Cached by path: a TensorRT engine holds ~55 MB of weights plus its
    activation buffers on the GPU, and rebuilding the context per request would
    make the reported inference_ms load time rather than steady-state speed.
    """
    key = f"{engine}:{model_path}"
    if key in _RECOGNIZER_CACHE:
        return _RECOGNIZER_CACHE[key]

    _stub_camera_management_package()
    dict_str = str(dict_path or CHARACTER_DICT_PATH)

    if engine == "tensorrt":
        from camera_management.ocr.backends.smtr_trt import TextRecognizerSMTRTRT
        rec = TextRecognizerSMTRTRT(str(model_path), dict_str)
    elif engine == "onnx":
        from camera_management.ocr.backends.smtr_onnx import TextRecognizerSMTRONNX
        rec = TextRecognizerSMTRONNX(str(model_path), dict_str, device="cuda")
    else:
        raise ValueError(f"engine must be 'onnx' or 'tensorrt', got {engine!r}")

    _RECOGNIZER_CACHE[key] = rec
    return rec


def release_model(model_id: str) -> int:
    """Drop cached recognizers whose path contains this model_id.

    Called from model deletion: without it a deleted model's engine stays
    resident on the GPU until the service restarts, and re-exporting under the
    same id would keep serving the stale one.
    """
    stale = [k for k in _RECOGNIZER_CACHE if model_id in k]
    for k in stale:
        _RECOGNIZER_CACHE.pop(k, None)
    return len(stale)


def normalize(text: str) -> str:
    """OpenOCR RecMetric's normalisation: letters+digits only, lowercased.

    The accuracy train_rec.py prints is measured on this form, so an exported
    model has to be scored the same way to be comparable. Raw exact-match is a
    different, stricter number — the two differ by a lot (0.968 vs 0.436 for the
    same model) because a model trained with use_space_char=False cannot emit
    spaces at all.
    """
    return "".join(c for c in text if c in string.digits + string.ascii_letters).lower()


def recognize_paths(
    recognizer, paths: Sequence[Path], batch: int = 1,
) -> List[Tuple[str, float, str, float]]:
    """Run the recognizer over image paths. Returns per-image
    (gtc_text, gtc_conf, ctc_text, ctc_conf).

    batch defaults to 1 on purpose. preprocess_batch pads every crop in a batch
    out to the widest one with -1, which costs real accuracy: the same engine
    scored 0.968 at batch 1 and 0.917 at batch 8. Batched numbers are fine for
    throughput but useless for gating a model.
    """
    out: List[Tuple[str, float, str, float]] = []
    for i in range(0, len(paths), batch):
        chunk = [cv.imread(str(p)) for p in paths[i:i + batch]]
        keep = [(j, img) for j, img in enumerate(chunk) if img is not None]
        results = recognizer.recognize_batch([img for _, img in keep]) if keep else []
        by_idx = {j: r for (j, _), r in zip(keep, results)}
        for j in range(len(chunk)):
            r = by_idx.get(j)
            if r is None:
                out.append(("", 0.0, "", 0.0))
            else:
                out.append((r[0][0], float(r[0][1]), r[1][0], float(r[1][1])))
    return out


def score_against_labels(
    preds: Sequence[Tuple[str, float, str, float]], labels: Sequence[str],
) -> Dict[str, float]:
    """Accuracy of both heads, normalised and exact.

    'either' is the ceiling the production candidate logic can reach, since
    pick_winning_candidate considers both heads' output.
    """
    n = max(len(labels), 1)

    def _acc(key, norm) -> Dict[str, float]:
        gtc = sum(norm(p[0]) == norm(gt) for p, gt in zip(preds, labels))
        ctc = sum(norm(p[2]) == norm(gt) for p, gt in zip(preds, labels))
        either = sum(norm(gt) in (norm(p[0]), norm(p[2])) for p, gt in zip(preds, labels))
        return {f"{key}_gtc": gtc / n, f"{key}_ctc": ctc / n, f"{key}_either": either / n}

    return {
        **_acc("norm", normalize),
        **_acc("exact", lambda s: s),
        "n": len(labels),
    }
