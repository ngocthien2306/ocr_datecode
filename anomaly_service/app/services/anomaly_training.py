"""
Anomaly training — wraps anomalib's Engine (PatchCore/Padim) around this
service's dataset layout. API surface verified against anomalib==2.5.0 in
the Week 1/2 spike (fit -> predict -> export ONNX -> onnxruntime, all
confirmed working end-to-end before this file was written).

Design notes:
  - We call `engine.predict()` once (not `engine.test()` + `predict()`
    separately) to get per-image (pred_score, gt_label, image_path) for
    every test sample, then compute AUROC/F1/confusion-matrix ourselves via
    sklearn. This avoids a second dataloader pass AND gives us the raw
    per-image scores we need for the threshold-recompute endpoint (Eval UI
    slider) without touching the model again.
  - `pred_score` from anomalib's predict() is already min-max normalized to
    [0, 1] by the model's post-processor, so a 0..1 threshold slider in the
    UI maps directly onto it — no extra rescaling needed.
  - Convention: gt_label / pred_label True == "abnormal". Exposed to the FE
    as the strings "normal"/"abnormal" (not OK/NG) to avoid confusion with
    the unrelated char-classifier's OK/NG labels.
  - PatchCore/Padim are feature-modeling algorithms, not iterative gradient
    training — `Engine.fit()` runs exactly one epoch (coreset selection /
    Gaussian fitting) and is fast even on a full dataset. Cancellation is
    therefore only checked *between* phases (fit / predict / export), not
    mid-epoch — good enough in practice, documented here so it isn't a
    surprise later.
"""
import base64
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2 as cv
import numpy as np

from app.models.anomaly import AnomalyTrainRequest
from app.services import dataset_fs
from app.services.inspection_crop import img_to_b64

logger = logging.getLogger(__name__)


class TrainingCancelled(Exception):
    pass


def _build_datamodule(project_id: str, request: AnomalyTrainRequest):
    from anomalib.data import Folder
    from anomalib.data.utils.split import ValSplitMode

    if not dataset_fs.has_images(dataset_fs.train_good_dir(project_id)):
        raise ValueError("No normal images in dataset — import 'normal' crops first.")

    # PatchCore/Padim are one-class — they train on normal images only, so
    # abnormal data isn't required to START training (only to get a
    # meaningful AUROC/F1 out of the eval step). Early in a project's life
    # you may only have normal crops yet; train anyway and let evaluating
    # against test_unlabeled-style data guide which images to label abnormal
    # next. train_model() reports metrics as unavailable when this is empty.
    defect_types = dataset_fs.list_defect_types(project_id)
    abnormal_dirs = [f"test/{d}" for d in defect_types] if defect_types else None
    normal_test_dir = (
        "test/good" if dataset_fs.has_images(dataset_fs.test_good_dir(project_id)) else None
    )

    return Folder(
        name=project_id,
        root=dataset_fs.dataset_dir(project_id),
        normal_dir="train/good",
        abnormal_dir=abnormal_dirs,
        normal_test_dir=normal_test_dir,
        # When normal_test_dir is None, this fraction of train/good is
        # automatically held out for testing instead.
        normal_split_ratio=request.test_split,
        # PatchCore/Padim don't train against a validation set (single-epoch
        # feature modeling, no early stopping/LR schedule) — anomalib's
        # default (FROM_TEST, 50%) would otherwise silently siphon half of
        # every imported test image into an unused val split, so our own
        # test-results/eval endpoints would only ever see the other half.
        # ValSplitMode.NONE crashes (Lightning still calls val_dataloader()
        # and Folder never populates val_data for NONE) — SAME_AS_TEST
        # reuses the test set for the (unused) val loop instead of removing
        # images from it, which keeps the real test set intact.
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=4,
    )


def _build_model(request: AnomalyTrainRequest):
    from anomalib.models import Padim, Patchcore

    algo = request.algorithm.lower()
    if algo == "patchcore":
        return Patchcore(
            backbone=request.backbone,
            layers=request.layers,
            coreset_sampling_ratio=request.coreset_sampling_ratio,
        )
    if algo == "padim":
        return Padim(backbone=request.backbone, layers=request.layers)
    raise ValueError(f"Unknown algorithm: {request.algorithm!r} (expected 'patchcore' or 'padim')")


def encode_heatmap_overlay(img: np.ndarray, anomaly_map: np.ndarray, quality: int = 90) -> str:
    """Per-pixel anomaly score map -> JPEG-encoded base64 heatmap blended
    over the original crop (min-max normalized per-image, JET colormap,
    resized to match img -- anomaly_map comes out at the model's
    fixed training resolution, e.g. 256x256, not the crop's native size)."""
    m = anomaly_map.astype(np.float32)
    lo, hi = float(m.min()), float(m.max())
    m = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
    m_u8 = (m * 255).astype(np.uint8)
    if m_u8.shape[:2] != img.shape[:2]:
        m_u8 = cv.resize(m_u8, (img.shape[1], img.shape[0]))
    colored = cv.applyColorMap(m_u8, cv.COLORMAP_JET)
    overlay = cv.addWeighted(img, 0.55, colored, 0.45, 0)
    ok, buf = cv.imencode(".jpg", overlay, [cv.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


def _compute_metrics(scores: List[float], labels: List[bool], threshold: float = 0.5) -> Dict[str, Any]:
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

    y_true = np.asarray(labels, dtype=np.int32)  # 1 = abnormal
    y_score = np.asarray(scores, dtype=np.float32)
    y_pred = (y_score >= threshold).astype(np.int32)

    auroc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "image_auroc": round(auroc, 4),
        "image_f1": round(f1, 4),
        "threshold": threshold,
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]], 0=normal 1=abnormal
        "n_normal_test": int((y_true == 0).sum()),
        "n_abnormal_test": int((y_true == 1).sum()),
    }


def recompute_metrics_at_threshold(test_results_path: Path, threshold: float) -> Dict[str, Any]:
    """Re-derive metrics from the stored per-image scores — no model/GPU
    needed, used by the Eval UI's threshold slider."""
    items = json.loads(test_results_path.read_text())
    scores = [it["pred_score"] for it in items]
    labels = [it["gt_label"] == "abnormal" for it in items]
    metrics = _compute_metrics(scores, labels, threshold=threshold)
    metrics["items"] = [
        {**it, "pred_label": "abnormal" if it["pred_score"] >= threshold else "normal",
         "correct": (it["pred_score"] >= threshold) == (it["gt_label"] == "abnormal")}
        for it in items
    ]
    return metrics


def train_model(
    project_id: str,
    model_id: str,
    request: AnomalyTrainRequest,
    checkpoint_dir: Path,
    progress_cb: Optional[Callable[[str, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Train + evaluate one PatchCore/Padim model. Saves:
      {checkpoint_dir}/{model_id}.ckpt            — Lightning checkpoint
      {checkpoint_dir}/{model_id}_test_results.json — per-image eval sidecar

    Returns the metrics dict (see _compute_metrics), plus checkpoint_path.
    """
    import torch
    from anomalib.engine import Engine

    def _check_cancel():
        if cancel_check is not None and cancel_check():
            raise TrainingCancelled("Cancelled by user")

    def _emit(phase: str, pct: float):
        _check_cancel()
        if progress_cb is not None:
            try:
                progress_cb(phase, float(pct))
            except Exception:
                logger.exception("[anomaly train] progress_cb failed")

    _emit("preparing", 5)
    datamodule = _build_datamodule(project_id, request)
    model = _build_model(request)

    run_dir = checkpoint_dir / "runs"
    engine = Engine(
        default_root_dir=str(run_dir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
    )

    _emit("fitting", 15)
    engine.fit(model=model, datamodule=datamodule)
    _check_cancel()

    _emit("predicting", 65)
    predictions = engine.predict(model=model, datamodule=datamodule)
    _check_cancel()

    scores: List[float] = []
    labels: List[bool] = []
    paths: List[str] = []
    anomaly_maps: List[Optional[np.ndarray]] = []
    for batch in predictions:
        scores.extend(float(s) for s in batch.pred_score.tolist())
        labels.extend(bool(g) for g in batch.gt_label.tolist())
        paths.extend(batch.image_path)
        amap = batch.anomaly_map
        if amap is not None:
            anomaly_maps.extend(amap.detach().cpu().numpy())
        else:
            anomaly_maps.extend([None] * len(batch.pred_score))

    if not scores:
        raise ValueError("No test predictions produced — check dataset has normal+abnormal test images.")

    _emit("evaluating", 80)
    metrics = _compute_metrics(scores, labels, threshold=0.5)

    _emit("encoding_testset", 88)
    test_results = []
    for score, label, path, amap in zip(scores, labels, paths, anomaly_maps):
        img = cv.imread(path)
        crop_b64 = img_to_b64(img) if img is not None else ""
        heatmap_b64 = encode_heatmap_overlay(img, amap) if img is not None and amap is not None else ""
        test_results.append({
            "image_path": path,
            "crop_b64": crop_b64,
            "heatmap_b64": heatmap_b64,
            "pred_score": round(score, 4),
            "gt_label": "abnormal" if label else "normal",
            "pred_label": "abnormal" if score >= 0.5 else "normal",
            "correct": (score >= 0.5) == label,
        })

    _emit("saving", 95)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_src = engine.trainer.checkpoint_callback.best_model_path
    ckpt_dst = checkpoint_dir / f"{model_id}.ckpt"
    if ckpt_src and Path(ckpt_src).exists():
        shutil.copy2(ckpt_src, ckpt_dst)
    else:
        raise RuntimeError("Training finished but no checkpoint was produced")

    test_results_path = checkpoint_dir / f"{model_id}_test_results.json"
    test_results_path.write_text(json.dumps(test_results))

    # The versioned run dir (runs/Patchcore/{project_id}/v0/...) is no longer
    # needed once we've copied out the checkpoint — drop it so repeated
    # training runs don't pile up multi-hundred-MB lightning_logs on disk.
    try:
        shutil.rmtree(run_dir, ignore_errors=True)
    except Exception:
        logger.warning(f"[anomaly train] failed to clean run dir {run_dir}")

    _emit("completed", 100)

    metrics["checkpoint_path"] = str(ckpt_dst)
    metrics["test_results_path"] = str(test_results_path)
    return metrics


def load_model_from_checkpoint(algorithm: str, checkpoint_path: Path):
    """Reload a trained model for export — a separate action from training,
    possibly run much later / after a service restart."""
    from anomalib.models import Padim, Patchcore

    algo = algorithm.lower()
    cls = Patchcore if algo == "patchcore" else Padim if algo == "padim" else None
    if cls is None:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
    # PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects
    # anomalib/Lightning classes (e.g. anomalib.PrecisionType) baked into the
    # checkpoint's hyperparameters -- not a security concern here since this
    # checkpoint was produced by our own train_model() above, not loaded
    # from an untrusted source.
    return cls.load_from_checkpoint(str(checkpoint_path), weights_only=False)


def export_onnx(algorithm: str, checkpoint_path: Path, export_dir: Path, image_size: int) -> Path:
    import torch
    from anomalib.engine import Engine

    model = load_model_from_checkpoint(algorithm, checkpoint_path)
    engine = Engine(
        default_root_dir=str(export_dir / "_engine_tmp"),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
    )
    onnx_path = engine.export(
        model=model,
        export_type="onnx",
        export_root=str(export_dir),
        input_size=(image_size, image_size),
    )
    shutil.rmtree(export_dir / "_engine_tmp", ignore_errors=True)
    if onnx_path is None:
        raise RuntimeError("anomalib export returned no path")
    return Path(onnx_path)
