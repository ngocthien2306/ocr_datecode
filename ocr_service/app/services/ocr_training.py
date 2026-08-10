"""
Fine-tune SVTRv2 + GTCDecoder(SMTRDecoder + RCTCDecoder) by driving OpenOCR's
tools/train_rec.py.

Runs as a SUBPROCESS, not in-process. train_rec.py configures the root logger,
claims a CUDA device globally, and initialises torch.distributed; hosting that
inside uvicorn would pollute the API process and leave nothing clean to kill on
cancel. A subprocess gives a real terminate() and takes its CUDA context with it
when it goes.
"""
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

from app.core.config import (
    BASE_CKPT_DIR,
    BUILTIN_BASES,
    CHARACTER_DICT_PATH,
    OPENOCR_DIR,
)
from app.models.ocr import OCRBaseRef, OCRTrainRequest
from app.services import ckpt_vocab, dataset_fs

logger = logging.getLogger(__name__)


class TrainingCancelled(Exception):
    pass


class TrainingFailed(RuntimeError):
    pass


# ── stdout parsing ────────────────────────────────────────────────────────────
# train_rec.py logs an epoch line per print_batch_step, then "cur metric" after
# each eval and "best metric" whenever main_indicator improves.
_RE_EPOCH = re.compile(r"epoch: \[(\d+)/(\d+)\]")
_RE_CUR = re.compile(r"cur metric, (.+)$")
_RE_BEST = re.compile(r"best metric, (.+)$")
_METRIC_KEYS = ("acc", "gtc_acc", "min_acc", "norm_edit_dis", "gtc_norm_edit_dis", "best_epoch")

# Training owns 0..85% of the progress bar; ONNX and TensorRT export split the
# rest, so the bar doesn't sit at 100% through a 40-second engine build.
TRAIN_PROGRESS_CEILING = 85.0


def _parse_metrics(blob: str) -> Dict[str, float]:
    """Parse "acc: 0.97, norm_edit_dis: 0.99, best_epoch: 4" into a dict.

    Only the keys we store — the line also carries num_samples and fps, which
    change between OpenOCR versions and are not worth tracking.
    """
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


# ── base checkpoint resolution ───────────────────────────────────────────────


def builtin_base_path(builtin: str) -> Path:
    if builtin not in BUILTIN_BASES:
        raise TrainingFailed(f"Unknown builtin base '{builtin}' (have: {sorted(BUILTIN_BASES)})")
    return BASE_CKPT_DIR / BUILTIN_BASES[builtin]


def resolve_base(ref: OCRBaseRef, model_lookup: Optional[Dict] = None) -> Dict:
    """Return {path, label} for the checkpoint a run starts from.

    model_lookup is the already-fetched ocr_models record when ref.kind=='model';
    this function stays sync/DB-free so it can run inside the training thread.
    """
    if ref.kind == "builtin":
        name = ref.builtin or "datecode_2406"
        path = builtin_base_path(name)
        if not path.is_file():
            raise TrainingFailed(
                f"Base checkpoint missing: {path}. It ships with the installer, "
                f"not with git — see ocr_service/README.md"
            )
        return {"path": path, "label": f"builtin:{name}"}

    if ref.kind == "model":
        if not model_lookup:
            raise TrainingFailed("base.kind='model' but no model record was provided")
        path = Path(model_lookup.get("checkpoint_path") or "")
        if not path.is_file():
            raise TrainingFailed(f"Base model's checkpoint is gone from disk: {path}")
        return {"path": path, "label": model_lookup.get("label") or f"model:{model_lookup.get('id')}"}

    raise TrainingFailed(f"base.kind must be 'builtin' or 'model', got {ref.kind!r}")


# ── config rendering ─────────────────────────────────────────────────────────


def render_config(
    project_id: str,
    model_id: str,
    req: OCRTrainRequest,
    pretrained_path: Path,
    output_dir: Path,
) -> Path:
    """Write the training YAML for one run.

    Kept as a literal template rather than a YAML round-trip: the anchors
    (&character_dict_path, &max_text_length, &next, &subsl) are load-bearing —
    the encoder, decoder and post-processor must agree on them — and a
    dump/reload would expand the anchors into copies that can then drift apart.

    Three values here are established by measurement, not taste. See
    docs/ocr_training_plan.md §2 before changing any of them:
      main_indicator: min_acc  — 'acc' is the CTC head alone, so best-checkpoint
                                 selection would ignore the SMTR head entirely
                                 and can save a run at epoch 2 with gtc_acc 0.52.
      save_epoch_step: huge    — suppresses epoch_N.pth. Each is 252 MB, and the
                                 default settings wrote 2.7 GB per run.
      use_amp: True            — needs the .float() patch in the two
                                 postprocess files (patches/).
    """
    dataset_dir = dataset_fs.dataset_dir(project_id)
    cfg = f"""Global:
  device: gpu
  epoch_num: {req.epoch_num}
  log_smooth_window: 20
  print_batch_step: 10
  output_dir: {output_dir}
  save_epoch_step: [100000, 100000]
  eval_batch_step: [0, 500]
  eval_epoch_step: [0, 1]
  cal_metric_during_train: True
  pretrained_model: {pretrained_path}
  checkpoints:
  use_tensorboard: false
  infer_img:
  character_dict_path: &character_dict_path {CHARACTER_DICT_PATH}
  max_text_length: &max_text_length {req.max_text_length}
  use_space_char: &use_space_char {req.use_space_char}
  save_res_path: {output_dir}/predicts.txt
  use_amp: True

Optimizer:
  name: AdamW
  lr: {req.lr}
  weight_decay: 0.05
  filter_bias_and_bn: True

LRScheduler:
  name: OneCycleLR
  warmup_epoch: 1.5
  cycle_momentum: False

Architecture:
  model_type: rec
  algorithm: SVTRv2
  in_channels: 3
  Transform:
  Encoder:
    name: SVTRv2LNConvTwo33
    use_pos_embed: False
    dims: [128, 256, 384]
    depths: [6, 6, 6]
    num_heads: [4, 8, 12]
    mixer: [['Conv','Conv','Conv','Conv','Conv','Conv'],['Conv','Conv','FGlobal','Global','Global','Global'],['Global','Global','Global','Global','Global','Global']]
    local_k: [[5, 5], [5, 5], [-1, -1]]
    sub_k: [[1, 1], [2, 1], [-1, -1]]
    last_stage: false
    feat2d: True
  Decoder:
    name: GTCDecoder
    infer_gtc: True
    detach: False
    gtc_decoder:
      name: SMTRDecoder
      num_layer: 1
      ds: True
      max_len: *max_text_length
      next_mode: &next True
      sub_str_len: &subsl 5
    ctc_decoder:
      name: RCTCDecoder

Loss:
  name: GTCLoss
  ctc_weight: 0.1
  gtc_loss:
    name: SMTRLoss

PostProcess:
  name: GTCLabelDecode
  gtc_label_decode:
    name: SMTRLabelDecode
    next_mode: *next
  character_dict_path: *character_dict_path
  use_space_char: *use_space_char

Metric:
  name: RecGTCMetric
  main_indicator: min_acc
  is_filter: True

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {dataset_dir}
    label_file_list: ["{dataset_fs.label_file(project_id, 'train')}"]
    transforms:
      - DecodeImagePIL:
          img_mode: RGB
      - PARSeqAugPIL:
      - RecTVResize:
          image_shape: [{req.image_h}, {req.image_w}]
          padding: False
      - GTCLabelEncode:
          gtc_label_encode:
            name: SMTRLabelEncode
            sub_str_len: *subsl
          character_dict_path: *character_dict_path
          use_space_char: *use_space_char
          max_text_length: *max_text_length
      - KeepKeys:
          keep_keys: ['image', 'label', 'label_subs', 'label_next', 'length_subs',
          'label_subs_pre', 'label_next_pre', 'length_subs_pre', 'length', 'ctc_label', 'ctc_length']
  loader:
    shuffle: True
    batch_size_per_card: {req.batch_size}
    drop_last: True
    num_workers: 2

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {dataset_dir}
    label_file_list: ["{dataset_fs.label_file(project_id, 'test')}"]
    transforms:
      - DecodeImagePIL:
          img_mode: RGB
      - RecTVResize:
          image_shape: [{req.image_h}, {req.image_w}]
          padding: False
      - GTCLabelEncode:
          gtc_label_encode:
            name: ARLabelEncode
          character_dict_path: *character_dict_path
          use_space_char: *use_space_char
          max_text_length: *max_text_length
      - KeepKeys:
          keep_keys: ['image', 'label', 'length', 'ctc_label', 'ctc_length']
  loader:
    shuffle: False
    drop_last: False
    batch_size_per_card: {req.batch_size}
    num_workers: 2
"""
    path = dataset_fs.models_dir(project_id) / f"{model_id}_config.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg, encoding="utf-8")
    return path


def build_post_process(config_path: Path):
    """Build GTCLabelDecode from a rendered config, for vocab surgery.

    Imported here rather than at module import time: it pulls in torch and the
    OpenOCR package, and the API process must start on a machine where OpenOCR
    isn't cloned (label review works fine without it).
    """
    if str(OPENOCR_DIR) not in sys.path:
        sys.path.insert(0, str(OPENOCR_DIR))
    from openrec.postprocess import build_post_process as _build
    from tools.engine.config import Config

    return _build(Config(str(config_path)).cfg["PostProcess"])


# ── the run ──────────────────────────────────────────────────────────────────


def prepare_checkpoint(
    project_id: str, model_id: str, req: OCRTrainRequest, base_path: Path, config_path: Path,
) -> Dict:
    """Widen the base checkpoint's vocabulary if this run needs the space class.

    Returns {path, expanded_from}. path is what pretrained_model must point at.
    """
    if not ckpt_vocab.needs_expansion(base_path, req.use_space_char):
        return {"path": base_path, "expanded_from": None}

    out = dataset_fs.models_dir(project_id) / f"{model_id}_base_space.pth"
    ckpt_vocab.expand_for_space(base_path, out, build_post_process(config_path))
    return {"path": out, "expanded_from": str(base_path)}


def train_model(
    project_id: str,
    model_id: str,
    req: OCRTrainRequest,
    base_path: Path,
    progress_cb: Callable[[str, float], None],
    cancel_check: Callable[[], bool],
) -> Dict:
    """Run one fine-tune. Returns the metrics of the best checkpoint."""
    models_dir = dataset_fs.models_dir(project_id)
    output_dir = models_dir / f"{model_id}_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_cb("preparing", 2.0)
    config_path = render_config(project_id, model_id, req, base_path, output_dir)
    prepared = prepare_checkpoint(project_id, model_id, req, base_path, config_path)
    if prepared["expanded_from"]:
        # Re-render so pretrained_model points at the widened copy. Cheaper and
        # less surprising than editing the file we just wrote.
        config_path = render_config(project_id, model_id, req, prepared["path"], output_dir)

    cmd = [sys.executable, "tools/train_rec.py", "--c", str(config_path)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    logger.info(f"[ocr] launching: {' '.join(cmd)} (cwd={OPENOCR_DIR})")

    proc = subprocess.Popen(
        cmd, cwd=str(OPENOCR_DIR), env=env,
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
            logger.warning("[ocr] cancel requested — terminating trainer")
            proc.terminate()
            continue

        m = _RE_EPOCH.search(line)
        if m:
            epoch, total = int(m.group(1)), max(int(m.group(2)), 1)
            pct = min(epoch / total * TRAIN_PROGRESS_CEILING, TRAIN_PROGRESS_CEILING)
            # Only report on change: print_batch_step fires several times per
            # epoch and the DB write behind progress_cb is not free.
            if pct - last_pct >= 1.0:
                last_pct = pct
                progress_cb("training", pct)
            continue

        m = _RE_BEST.search(line)
        if m:
            best = _parse_metrics(m.group(1))
            logger.info(f"[ocr] best so far: {best}")
            continue

        m = _RE_CUR.search(line)
        if m:
            logger.info(f"[ocr] eval: {_parse_metrics(m.group(1))}")

    rc = proc.wait()

    if cancelled:
        # terminate() lands as a non-zero rc; that is the expected path here.
        raise TrainingCancelled()
    if rc != 0:
        raise TrainingFailed(
            f"train_rec.py exited {rc}. Last lines:\n" + "\n".join(tail[-25:])
        )

    src = output_dir / "best.pth"
    if not src.is_file():
        raise TrainingFailed(
            f"training finished but {src} was never written — no eval ever "
            f"improved on the initial best_metric, or the eval set is empty"
        )
    checkpoint_path = models_dir / f"{model_id}.pth"
    shutil.move(str(src), str(checkpoint_path))
    # latest.pth is another 252 MB and is only useful for resuming, which this
    # service does not offer.
    shutil.rmtree(output_dir, ignore_errors=True)
    if prepared["expanded_from"]:
        # The widened base was only an input to this run. Dropping it saves
        # 84 MB per model, and expand_for_space is deterministic so it can be
        # rebuilt from expanded_from + the config if anyone needs to audit it.
        # Left in place on failure, where it is worth inspecting.
        (models_dir / f"{model_id}_base_space.pth").unlink(missing_ok=True)

    progress_cb("trained", TRAIN_PROGRESS_CEILING)
    return {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "metrics": best,
        "vocab_size": ckpt_vocab.vocab_size(checkpoint_path),
        "expanded_from": prepared["expanded_from"],
    }
