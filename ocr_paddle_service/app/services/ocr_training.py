"""
Fine-tune a PaddleOCR PP-OCRv5/v6 recogniser by driving PaddleOCR's
tools/train.py.

Runs as a SUBPROCESS, not in-process: train.py configures logging, claims the
GPU and initialises paddle.distributed; hosting that inside uvicorn would
pollute the API process and leave nothing clean to kill on cancel. A
subprocess gives a real terminate() and takes its CUDA context with it.

The config is the arch's upstream YAML with a small set of overrides, so a
PaddleOCR update that changes an architecture's head or loss is picked up
rather than frozen into a template here. The overrides that matter:

  Train set  MultiScaleDataSet with ds_width=True at a single height (48):
             batches are grouped by aspect ratio and resized to 48 x 48*aspect.
             The upstream default squeezes every crop into 320 px at three
             heights; production (TextRecognizerTRT) keeps the aspect at H=48,
             and 41% of factory datecode crops are wider than 320 px there.
             Measured on the same 2-epoch model: 118/154 with the upstream-
             style resize vs 113/154 through production preprocessing.
  RecConAug  removed — it concatenates two crops into one, which breaks the
             per-batch aspect the sampler just chose.
  Images     grayscale copies (see dataset_fs), matching production's input.
"""
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import yaml

from app.core.config import (
    PADDLE_ARCHS,
    PADDLEOCR_DIR,
    REC_IMAGE_H,
    arch_dict_path,
    arch_pretrained_path,
)
from app.models.ocr import OCRBaseRef, OCRTrainRequest
from app.services import dataset_fs

logger = logging.getLogger(__name__)


class TrainingCancelled(Exception):
    pass


class TrainingFailed(RuntimeError):
    pass


# PaddleOCR's train.py log lines:
#   "epoch: [3/50], global_step: 60, lr: ..., acc: 0.75, ..."
#   "cur metric, acc: 0.93, norm_edit_dis: 0.98, fps: ..."
#   "best metric, acc: 0.93, is_float16: False, norm_edit_dis: 0.98, ..., best_epoch: 7"
_RE_EPOCH = re.compile(r"epoch: \[(\d+)/(\d+)\]")
_RE_CUR = re.compile(r"cur metric, (.+)$")
_RE_BEST = re.compile(r"best metric, (.+)$")
_METRIC_KEYS = ("acc", "norm_edit_dis", "best_epoch")

# Training owns 0..85% of the progress bar; export + engine + eval the rest.
TRAIN_PROGRESS_CEILING = 85.0


def _parse_metrics(blob: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in blob.split(","):
        if ":" not in part:
            continue
        k, _, v = part.partition(":")
        k = k.strip()
        if k not in _METRIC_KEYS:
            continue
        try:
            out[k] = int(v) if k == "best_epoch" else float(v)
        except ValueError:
            continue
    return out


# ── base weights ─────────────────────────────────────────────────────────────


def resolve_base(arch: str, ref: OCRBaseRef, model_lookup: Optional[Dict] = None) -> Dict:
    """Return {path, label} for the weights a run starts from. `path` is the
    .pdparams file; PaddleOCR's pretrained_model takes it without the suffix.

    model_lookup is the already-fetched model record when ref.kind=='model'
    (kept DB-free so it can run in the training thread)."""
    if arch not in PADDLE_ARCHS:
        raise TrainingFailed(f"Unknown arch '{arch}' (have: {sorted(PADDLE_ARCHS)})")
    if not (PADDLEOCR_DIR / "tools" / "train.py").is_file():
        raise TrainingFailed(
            f"PaddleOCR checkout missing at {PADDLEOCR_DIR} — "
            f"git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git"
        )

    if ref.kind == "pretrained":
        path = arch_pretrained_path(arch)
        if not path.is_file():
            raise TrainingFailed(
                f"Pretrained weights missing: {path}. Download them with "
                f"scripts/setup_ocr_paddle_service.sh --download {arch}"
            )
        return {"path": path, "label": f"pretrained:{arch}"}

    if ref.kind == "model":
        if not model_lookup:
            raise TrainingFailed("base.kind='model' but no model record was provided")
        if model_lookup.get("arch") != arch:
            raise TrainingFailed(
                f"Base model is {model_lookup.get('arch')}, this run is {arch} — "
                f"weights of a different architecture cannot be loaded"
            )
        path = Path(model_lookup.get("checkpoint_path") or "")
        if not path.is_file():
            raise TrainingFailed(f"Base model's checkpoint is gone from disk: {path}")
        return {"path": path, "label": model_lookup.get("label") or f"model:{model_lookup.get('id')}"}

    raise TrainingFailed(f"base.kind must be 'pretrained' or 'model', got {ref.kind!r}")


# ── config rendering ─────────────────────────────────────────────────────────


def _strip_pdparams(path: Path) -> str:
    s = str(path)
    return s[: -len(".pdparams")] if s.endswith(".pdparams") else s


def render_config(
    project_id: str, model_id: str, req: OCRTrainRequest, pretrained_path: Path,
    n_train: int,
) -> Path:
    """Write the training YAML for one run: upstream arch config + overrides."""
    arch = PADDLE_ARCHS[req.arch]
    with open(PADDLEOCR_DIR / arch["config"], encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_dir = dataset_fs.dataset_dir(project_id)
    run_dir = dataset_fs.run_dir(project_id, model_id)
    steps_per_epoch = max(n_train // req.batch_size, 1)
    max_w = max(req.max_train_width, 64)

    g = cfg["Global"]
    g.update({
        "use_gpu": True,
        "distributed": False,
        "epoch_num": req.epoch_num,
        "print_batch_step": 10,
        "save_model_dir": str(run_dir),
        # Only best_accuracy + latest are kept; an iter_epoch_N per epoch is
        # ~60 MB each for nothing this service offers (no resume).
        "save_epoch_step": 100000,
        # Evaluate once per epoch — best-epoch selection needs the eval, and
        # upstream's 2000-step interval never fires on a factory dataset.
        "eval_batch_step": [0, steps_per_epoch],
        "cal_metric_during_train": True,
        "pretrained_model": _strip_pdparams(pretrained_path),
        "checkpoints": None,
        "character_dict_path": str(arch_dict_path(req.arch)),
        "max_text_length": req.max_text_length,
        "use_space_char": True,
        "use_visualdl": False,
        "save_res_path": str(run_dir / "predicts.txt"),
    })

    lr = cfg["Optimizer"]["lr"]
    lr["learning_rate"] = req.lr
    # Upstream warms up 5 epochs of a 100+ epoch schedule; keep it ~10%.
    if "warmup_epoch" in lr:
        lr["warmup_epoch"] = max(1, min(5, req.epoch_num // 10))

    # Every head that carries max_text_length (NRTRHead) must agree with Global.
    for head in cfg["Architecture"].get("Head", {}).get("head_list", []) or []:
        for spec in head.values():
            if isinstance(spec, dict) and "max_text_length" in spec:
                spec["max_text_length"] = req.max_text_length

    tr = cfg["Train"]
    tr["dataset"]["name"] = "MultiScaleDataSet"
    tr["dataset"]["ds_width"] = True
    tr["dataset"]["data_dir"] = str(dataset_dir)
    tr["dataset"]["label_file_list"] = [str(dataset_fs.label_file(project_id, model_id, "train"))]
    tr["dataset"].pop("ext_op_transform_idx", None)
    transforms = []
    for op in tr["dataset"]["transforms"]:
        name = next(iter(op))
        if name == "RecConAug":
            continue
        transforms.append(op)
    tr["dataset"]["transforms"] = transforms
    tr["sampler"] = {
        "name": "MultiScaleSampler",
        "scales": [[max_w, REC_IMAGE_H]],
        "first_bs": req.batch_size,
        "fix_bs": True,
        "divided_factor": [8, 16],
        "is_training": True,
        "max_w": float(max_w),
    }
    tr["loader"].update({
        "shuffle": True,
        "batch_size_per_card": req.batch_size,
        "drop_last": True,
        "num_workers": 4,
    })

    ev = cfg["Eval"]
    ev["dataset"]["data_dir"] = str(dataset_dir)
    ev["dataset"]["label_file_list"] = [str(dataset_fs.label_file(project_id, model_id, "test"))]
    for op in ev["dataset"]["transforms"]:
        if "RecResizeImg" in op:
            # eval_mode resizes to H=48 x (48*aspect) with no padding (width
            # floor 48), one crop per batch — i.e. what TextRecognizerTRT feeds
            # the engine. A fixed-width eval instead pads every crop with
            # mid-gray the model never saw in ds_width training, and measured
            # acc 0.03 for a model that scores 0.95 on the line's
            # preprocessing — best-epoch selection on that number is noise.
            op["RecResizeImg"]["image_shape"] = [3, REC_IMAGE_H, REC_IMAGE_H]
            op["RecResizeImg"]["eval_mode"] = True
    ev["loader"].update({"batch_size_per_card": 1, "num_workers": 2})

    path = dataset_fs.models_dir(project_id) / f"{model_id}_config.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


# ── the run ──────────────────────────────────────────────────────────────────


def train_model(
    project_id: str,
    model_id: str,
    req: OCRTrainRequest,
    base_path: Path,
    n_train: int,
    progress_cb: Callable[[str, float], None],
    cancel_check: Callable[[], bool],
) -> Dict:
    """Run one fine-tune. Returns the metrics of the best checkpoint."""
    run_dir = dataset_fs.run_dir(project_id, model_id)
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    progress_cb("preparing", 2.0)
    config_path = render_config(project_id, model_id, req, base_path, n_train)

    cmd = [sys.executable, "tools/train.py", "-c", str(config_path)]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        PYTHONUNBUFFERED="1",
        # Default is to grab most of the card up front; the line's live
        # inference shares this GPU.
        FLAGS_allocator_strategy="auto_growth",
    )
    logger.info(f"[paddle-ocr] launching: {' '.join(cmd)} (cwd={PADDLEOCR_DIR})")

    proc = subprocess.Popen(
        cmd, cwd=str(PADDLEOCR_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    best: Dict[str, float] = {}
    tail: list = []
    cancelled = False
    last_pct = -1.0

    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if not line:
            continue
        tail.append(line)
        if len(tail) > 60:
            tail.pop(0)

        if cancel_check() and not cancelled:
            cancelled = True
            logger.warning("[paddle-ocr] cancel requested — terminating trainer")
            proc.terminate()
            continue

        m = _RE_EPOCH.search(line)
        if m:
            epoch, total = int(m.group(1)), max(int(m.group(2)), 1)
            pct = min(epoch / total * TRAIN_PROGRESS_CEILING, TRAIN_PROGRESS_CEILING)
            if pct - last_pct >= 1.0:
                last_pct = pct
                progress_cb("training", pct)
            continue

        m = _RE_BEST.search(line)
        if m:
            parsed = _parse_metrics(m.group(1))
            if parsed != best:
                best = parsed
                logger.info(f"[paddle-ocr] best so far: {best}")
            continue

        m = _RE_CUR.search(line)
        if m:
            logger.info(f"[paddle-ocr] eval: {_parse_metrics(m.group(1))}")
            continue

        if "Error" in line or "Traceback" in line:
            logger.warning(f"[paddle-ocr] {line}")

    rc = proc.wait()

    if cancelled:
        raise TrainingCancelled()
    if rc != 0:
        raise TrainingFailed(
            f"tools/train.py exited {rc}. Last lines:\n" + "\n".join(tail[-25:])
        )

    src = run_dir / "best_accuracy.pdparams"
    if not src.is_file():
        raise TrainingFailed(
            f"training finished but {src} was never written — no eval ever ran "
            f"or the eval set is empty"
        )
    checkpoint_path = dataset_fs.models_dir(project_id) / f"{model_id}.pdparams"
    shutil.copyfile(src, checkpoint_path)
    # latest.* and the optimizer state are only useful for resuming, which this
    # service does not offer.
    shutil.rmtree(run_dir, ignore_errors=True)

    progress_cb("trained", TRAIN_PROGRESS_CEILING)
    return {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "metrics": best,
    }
