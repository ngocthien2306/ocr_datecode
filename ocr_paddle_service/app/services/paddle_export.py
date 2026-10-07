"""
Trained checkpoint -> inference model -> ONNX -> TensorRT engine, and scoring
of those artifacts the way the line runs them.

Every heavy step is a subprocess:
  export  paddle_train env (this one), tools/export_rec_onnx.py — works around
          two paddle2onnx 2.1.0 / Paddle 3.4 incompatibilities and refuses to
          write an ONNX that disagrees with Paddle (see that file's docstring).
  engine  `vision` env (settings.TRT_CONDA_ENV), tools/trt_build_eval.py — the
          engine must be built by the TensorRT ai_services deserializes with.
  eval    ONNX: in this process, CPU onnxruntime, with a copy of production's
          preprocessing + CTC decode below. TensorRT: `vision` env, with
          ai_services' own TextRecognizerTRT.
"""
import json
import logging
import os
import shutil
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np

from app.core.config import (
    PADDLEOCR_DIR,
    REPO_ROOT,
    SERVICE_ROOT,
    arch_dict_path,
    settings,
)
from app.services import dataset_fs

logger = logging.getLogger(__name__)

EXPORT_TOOL = SERVICE_ROOT / "tools" / "export_rec_onnx.py"
TRT_TOOL = SERVICE_ROOT / "tools" / "trt_build_eval.py"
CONDA_SH = Path.home() / "miniconda3" / "etc" / "profile.d" / "conda.sh"


def _run(cmd: List[str], cwd: Path, env: Optional[Dict] = None, timeout: int = 1800) -> str:
    proc = subprocess.run(
        cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        raise RuntimeError(f"{Path(cmd[1]).name if len(cmd) > 1 else cmd[0]} exited "
                           f"{proc.returncode}:\n{tail}")
    return proc.stdout


def _run_in_trt_env(args: List[str], timeout: int = 1800) -> Dict:
    """Run tools/trt_build_eval.py in the TensorRT env; return its JSON."""
    inner = (
        f"source '{CONDA_SH}' && conda activate {settings.TRT_CONDA_ENV} && "
        f"exec python '{TRT_TOOL}' " + " ".join(f"'{a}'" for a in args)
    )
    # A clean env: this process's LD_LIBRARY_PATH / CUDA libs are Paddle's
    # (nvidia-* wheels for CUDA 13.0), not what the vision env was built with.
    env = {k: v for k, v in os.environ.items()
           if k in ("HOME", "USER", "PATH", "LANG", "TERM")}
    env["PYTHONUNBUFFERED"] = "1"
    out = _run(["bash", "-lc", inner], cwd=REPO_ROOT, env=env, timeout=timeout)
    last = [l for l in out.splitlines() if l.startswith("{")]
    if not last:
        raise RuntimeError(f"trt_build_eval.py printed no result:\n{out[-2000:]}")
    return json.loads(last[-1])


# ── export ───────────────────────────────────────────────────────────────────


def export_onnx(project_id: str, model_id: str, arch: str,
                config_path: Path, checkpoint_path: Path) -> Dict:
    """Checkpoint -> inference/ + rec.onnx + dict.txt in the export dir.

    dict.txt is copied beside the model so the export dir is self-contained: the
    recipe hands ai_services a model path and a dict path, and a decoder built
    from the wrong dict shifts every character index without raising.
    """
    out_dir = dataset_fs.export_dir(project_id, model_id)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    inference_dir = out_dir / "inference"
    onnx_path = out_dir / "rec.onnx"

    ckpt = str(checkpoint_path)
    if ckpt.endswith(".pdparams"):
        ckpt = ckpt[: -len(".pdparams")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    _run(
        [sys.executable, str(EXPORT_TOOL), "-c", str(config_path), "-o",
         f"Global.pretrained_model={ckpt}",
         f"Global.save_inference_dir={inference_dir}",
         "Global.use_gpu=False",
         "--onnx", str(onnx_path)],
        cwd=PADDLEOCR_DIR, env=env,
    )

    dict_dest = out_dir / "dict.txt"
    shutil.copyfile(arch_dict_path(arch), dict_dest)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    num_classes = sess.get_outputs()[0].shape[-1]
    n_dict = len(dict_dest.read_text(encoding="utf-8").splitlines())
    # blank + dict + space. ai_services pads any shortfall with ' ', so a
    # mismatch would not raise there — it would decode wrong characters.
    if isinstance(num_classes, int) and num_classes != n_dict + 2:
        raise RuntimeError(
            f"ONNX has {num_classes} classes but dict has {n_dict} (+blank+space = "
            f"{n_dict + 2}) — dict and model disagree"
        )
    return {
        "inference_dir": str(inference_dir),
        "onnx_path": str(onnx_path),
        "dict_path": str(dict_dest),
        "num_classes": num_classes,
        "size_mb": round(onnx_path.stat().st_size / 1e6, 2),
    }


def build_engine(project_id: str, model_id: str, onnx_path: Path) -> Dict:
    engine_path = dataset_fs.export_dir(project_id, model_id) / "rec_fp16.engine"
    return _run_in_trt_env(["build", "--onnx", str(onnx_path), "--engine", str(engine_path)])


# ── scoring ──────────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Letters+digits, lowercased — what PaddleOCR's RecMetric compares with
    ignore_space, and close to what production's compare_texts() tolerates."""
    return "".join(c for c in text if c in string.digits + string.ascii_letters).lower()


def score(preds: Sequence[str], labels: Sequence[str]) -> Dict:
    n = max(len(labels), 1)
    return {
        "norm": sum(normalize(p) == normalize(g) for p, g in zip(preds, labels)) / n,
        "exact": sum(p == g for p, g in zip(preds, labels)) / n,
        "n": len(labels),
    }


# Copied from ai_services/camera_management/ocr/ocr_utils.py
# preprocess_text_image (target_height=48) with TextRecognizerTRT's
# min_width=50 / max_width=2000. Copied, not imported — see the module
# docstring of tools/trt_build_eval.py for the one place that does import it.
def preprocess_like_production(image: np.ndarray, min_width: int = 50,
                               max_width: int = 2000) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    new_w = max(int(w * (48 / h)), 10)
    resized = cv.resize(gray, (new_w, 48))
    if new_w < min_width:
        resized = cv.copyMakeBorder(resized, 0, 0, 0, min_width - new_w,
                                    cv.BORDER_CONSTANT, value=0)
        new_w = min_width
    if new_w > max_width:
        resized = cv.resize(resized, (max_width, 48))
    x = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.stack([x, x, x], axis=0)[None].astype(np.float32)


def _char_list(dict_path: Path, num_classes: int) -> List[str]:
    # TextRecognizerTRT: ['blank'] + [line.strip() ...], padded with ' '.
    chars = ["blank"] + [l.strip() for l in dict_path.read_text(encoding="utf-8").splitlines()]
    return chars + [" "] * max(0, num_classes - len(chars))


def _ctc_decode(pred: np.ndarray, char_list: List[str]) -> Tuple[str, float]:
    best = pred.argmax(-1)
    conf = float(np.mean([pred[t, c] for t, c in enumerate(best)])) if len(best) else 0.0
    out, prev = [], -1
    for c in best:
        if c != 0 and c != prev and c < len(char_list):
            out.append(char_list[c])
        prev = c
    return "".join(out), conf


_ONNX_CACHE: Dict[str, object] = {}


def _onnx_session(onnx_path: Path):
    key = str(onnx_path)
    if key not in _ONNX_CACHE:
        import onnxruntime as ort
        _ONNX_CACHE[key] = ort.InferenceSession(key, providers=["CPUExecutionProvider"])
    return _ONNX_CACHE[key]


def release_model(model_id: str) -> None:
    for k in [k for k in _ONNX_CACHE if model_id in k]:
        _ONNX_CACHE.pop(k, None)


def recognize_onnx(onnx_path: Path, dict_path: Path, images: Sequence[np.ndarray]
                   ) -> Tuple[List[Tuple[str, float]], float]:
    sess = _onnx_session(onnx_path)
    name = sess.get_inputs()[0].name
    chars = _char_list(dict_path, sess.get_outputs()[0].shape[-1])
    out = []
    t0 = time.perf_counter()
    for img in images:
        if img is None:
            out.append(("", 0.0))
            continue
        pred = sess.run(None, {name: preprocess_like_production(img)})[0][0]
        out.append(_ctc_decode(pred, chars))
    ms = (time.perf_counter() - t0) * 1000 / max(len(images), 1)
    return out, ms


def predict_paths(onnx_path: Optional[Path], engine_path: Optional[Path], dict_path: Path,
                  paths: Sequence[Path], engine: str) -> Tuple[List[str], List[float], float]:
    """(texts, confidences, ms/image) for absolute image paths at batch 1."""
    if engine == "onnx":
        if not onnx_path or not onnx_path.is_file():
            raise FileNotFoundError("No ONNX export for this model yet")
        results, ms = recognize_onnx(onnx_path, dict_path, [cv.imread(str(p)) for p in paths])
        return [r[0] for r in results], [r[1] for r in results], ms
    if engine == "tensorrt":
        if not engine_path or not engine_path.is_file():
            raise FileNotFoundError("No TensorRT engine for this model yet")
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            for p in paths:
                f.write(f"{p}\t\n")
            list_path = f.name
        try:
            got = _run_in_trt_env(["eval", "--engine", str(engine_path),
                                   "--dict", str(dict_path), "--list", list_path])
        finally:
            os.unlink(list_path)
        return ([p["text"] for p in got["preds"]], [p["confidence"] for p in got["preds"]],
                got["ms_per_image"])
    raise ValueError(f"engine must be 'onnx' or 'tensorrt', got {engine!r}")


def evaluate(project_id: str, onnx_path: Optional[Path], engine_path: Optional[Path],
             dict_path: Path, items: Sequence, engine: str) -> Dict:
    """Score one artifact over items at batch=1. engine: 'onnx' | 'tensorrt'.

    Source images are ocr_service's colour crops — production also receives
    colour and converts to gray itself, so no pre-conversion here."""
    paths = [dataset_fs.image_abs_path(project_id, i.image_path) for i in items]
    labels = [i.gt_text for i in items]
    preds, confs, ms = predict_paths(onnx_path, engine_path, dict_path, paths, engine)

    per_image = [{
        "id": item.id,
        "gt_text": item.gt_text,
        "pred_text": pred,
        "confidence": conf,
        "correct_norm": normalize(pred) == normalize(item.gt_text),
        "correct_exact": pred == item.gt_text,
        "image_path": item.image_path,
    } for item, pred, conf in zip(items, preds, confs)]
    return {
        "engine": engine,
        "scores": score(preds, labels),
        "ms_per_image": round(ms, 2),
        "items": per_image,
    }


def evaluate_synthetic(project_id: str, onnx_path: Optional[Path], engine_path: Optional[Path],
                       dict_path: Path, entries: Sequence[Dict], engine: str) -> Dict:
    """Reject rate on synthetic defect crops of TEST items: how often the line
    would FAIL a crop with one character damaged. Also what it read, per type."""
    from app.services.synthetic_defects import is_rejected
    base = dataset_fs.dataset_dir(project_id)
    preds, confs, ms = predict_paths(onnx_path, engine_path, dict_path,
                                     [base / e["file"] for e in entries], engine)
    by_type: Dict[str, List[int]] = {}
    items = []
    for e, pred, conf in zip(entries, preds, confs):
        rej = is_rejected(pred, e["source_label"])
        t = by_type.setdefault(e["type"], [0, 0])
        t[0] += 1
        t[1] += rej
        items.append({**{k: e[k] for k in ("file", "type", "char", "severity", "source_label", "label")},
                      "pred_text": pred, "confidence": conf, "rejected": rej})
    n = max(len(entries), 1)
    return {
        "engine": engine,
        "n": len(entries),
        "reject_rate": sum(i["rejected"] for i in items) / n,
        "by_type": {k: {"n": v[0], "reject_rate": v[1] / max(v[0], 1)} for k, v in by_type.items()},
        "ms_per_image": round(ms, 2),
        "items": items,
    }
