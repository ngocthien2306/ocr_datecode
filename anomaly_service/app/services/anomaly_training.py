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


def _stage_dataset(project_id: str, excluded_paths: List[str], staging_root: Path) -> Path:
    """Mirror the dataset into `staging_root` as symlinks, omitting excluded files.

    Training reads the filesystem (anomalib's Folder globs directories), so an
    "excluded" flag in the database is invisible to it. Rather than moving or
    deleting the user's images, build a throwaway tree of links to the ones that
    should take part and point Folder at that. Nothing in the real dataset is
    touched, so an interrupted run cannot lose data.

    Returns the staged dataset root.
    """
    src_root = dataset_fs.dataset_dir(project_id)
    proj_dir = dataset_fs.project_dir(project_id)
    # Excluded paths are stored project-relative (they include the leading
    # "dataset/"); resolve to absolute so they can be compared to what we walk.
    excluded_abs = {str((proj_dir / p).resolve()) for p in excluded_paths}

    if staging_root.exists():
        shutil.rmtree(staging_root, ignore_errors=True)

    # train/good, test/good, plus every test/<defect_type> folder.
    src_dirs = [src_root / "train" / "good", src_root / "test" / "good"]
    test_root = src_root / "test"
    if test_root.exists():
        src_dirs += [d for d in sorted(test_root.iterdir()) if d.is_dir() and d.name != "good"]

    n_linked = n_skipped = 0
    for src_dir in src_dirs:
        if not src_dir.exists():
            continue
        rel_dir = src_dir.relative_to(src_root)
        dst_dir = staging_root / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_dir.glob("*")):
            if not f.is_file() or f.suffix.lower() not in dataset_fs.ALLOWED_EXTS:
                continue
            if str(f.resolve()) in excluded_abs:
                n_skipped += 1
                continue
            (dst_dir / f.name).symlink_to(f.resolve())
            n_linked += 1

    logger.info(
        f"[anomaly train] staged dataset for {project_id}: "
        f"{n_linked} image(s) linked, {n_skipped} excluded -> {staging_root}"
    )
    return staging_root


def _build_datamodule(project_id: str, request: AnomalyTrainRequest, staging_root: Optional[Path] = None):
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
    # When images are excluded, everything below is resolved against the staged
    # symlink tree instead of the real dataset — otherwise a defect type whose
    # every image was excluded would still be listed, and anomalib hard-errors
    # on an empty abnormal_dir.
    root = staging_root or dataset_fs.dataset_dir(project_id)
    test_root = root / "test"

    if staging_root is None:
        defect_types = dataset_fs.list_defect_types(project_id)
        has_normal_test = dataset_fs.has_images(dataset_fs.test_good_dir(project_id))
    else:
        defect_types = sorted(
            d.name for d in test_root.iterdir()
            if d.is_dir() and d.name != "good" and dataset_fs.has_images(d)
        ) if test_root.exists() else []
        has_normal_test = dataset_fs.has_images(test_root / "good")
        if not dataset_fs.has_images(root / "train" / "good"):
            raise ValueError("Every normal training image was excluded — nothing left to train on.")

    abnormal_dirs = [f"test/{d}" for d in defect_types] if defect_types else None
    normal_test_dir = "test/good" if has_normal_test else None

    return Folder(
        name=project_id,
        root=root,
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
        # 0, not the anomalib default of 4: with num_workers>0 PyTorch spawns
        # separate multiprocessing DataLoader worker processes, and at least
        # one has been observed to survive after training finishes (idle,
        # 0% CPU) holding onto several GB of RSS (torch/TensorRT libs +
        # cached tensors) until manually killed. PatchCore/Padim run exactly
        # 1 epoch on small datasets, so in-process loading isn't a
        # bottleneck -- not worth the leak risk to parallelize it.
        num_workers=0,
    )


def _build_model(request: AnomalyTrainRequest):
    from anomalib.models import Padim, Patchcore

    algo = request.algorithm.lower()
    cls = Patchcore if algo == "patchcore" else Padim if algo == "padim" else None
    if cls is None:
        raise ValueError(f"Unknown algorithm: {request.algorithm!r} (expected 'patchcore' or 'padim')")

    # request.image_size MUST be wired in here, not only at export time. In
    # anomalib 2.x the training/eval resize lives in the model's PreProcessor
    # (Resize + Normalize) -- the Folder datamodule has no image_size of its
    # own -- so leaving it at the default silently trains at 256x256 no matter
    # what the user picked, while export_onnx() still bakes input_size=512.
    # That train/serve resolution mismatch is catastrophic for PatchCore/Padim:
    # the memory bank / Gaussian is fitted on patch features at one scale, so
    # feeding a different scale puts every patch far from every stored feature.
    # Measured on a 512-export of a 256-trained model: raw score 0.58-0.69 ->
    # 20.6-22.7 (~33x) on the model's own normal training images, saturating
    # pred_score to exactly 1.0000 for every image in the Test tab while Eval
    # (which runs at the training resolution) looked perfectly healthy.
    size = (request.image_size, request.image_size)
    kwargs: Dict[str, Any] = {
        "backbone": request.backbone,
        "layers": request.layers,
        "pre_processor": cls.configure_pre_processor(image_size=size),
    }
    if algo == "patchcore":
        kwargs["coreset_sampling_ratio"] = request.coreset_sampling_ratio
    return cls(**kwargs)


def encode_heatmap_overlay(img: np.ndarray, anomaly_map: np.ndarray, quality: int = 90) -> str:
    """Per-pixel anomaly score map -> JPEG-encoded base64 heatmap blended
    over the original crop (JET colormap, resized to match img -- anomaly_map
    comes out at the model's fixed training resolution, e.g. 256x256, not the
    crop's native size).

    The map is rendered on its ABSOLUTE scale: anomalib's post-processor
    already normalizes anomaly_map so that 0.5 is the decision threshold, so
    0 -> cold, 0.5 -> at threshold, 1 -> clearly anomalous, and two crops are
    directly comparable to each other.

    Do NOT min-max stretch per image. That was the original implementation and
    it made every crop look uniformly red-hot: a perfectly normal crop whose
    map peaks at 0.12 had that 0.12 rescaled to 255, i.e. pure red in JET.
    Measured on this project's own normal-only test split, per-image peaks ran
    0.12-0.47 with 0.00% of pixels above the 0.5 threshold -- so every red
    region in those heatmaps was an artifact of the rescale, and normal vs
    defective crops were visually indistinguishable.
    """
    m = np.clip(anomaly_map.astype(np.float32), 0.0, 1.0)
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

    # Both AUROC and F1 are undefined on a single-class test set -- which is
    # exactly what you get before any abnormal images are imported (see
    # _build_datamodule). Report them as None, not 0.0: a hard 0.0 is
    # indistinguishable in the UI from a model that genuinely scores 0.0, so it
    # reads as "completely broken" when the truth is "not measurable yet".
    # (F1 would technically evaluate to 0.0 here -- zero true positives are
    # possible when no sample is positive -- but that number carries no
    # information about the model, so it's suppressed alongside AUROC.)
    measurable = len(np.unique(y_true)) > 1
    auroc = round(float(roc_auc_score(y_true, y_score)), 4) if measurable else None
    f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4) if measurable else None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "image_auroc": auroc,
        "image_f1": f1,
        "metrics_available": measurable,
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


def release_torch_cuda_cache(where: str) -> None:
    """Hand torch's cached-but-unused GPU blocks back to the driver.

    torch's caching allocator never releases freed blocks on its own: after a
    training run its reserved pool stays at the training peak even though
    memory_allocated() reads 0. TensorRT and onnxruntime cudaMalloc *outside*
    that pool, so everything torch retains is memory they simply cannot have.

    Measured: freeing a 3 GB tensor left torch.reserved at 2864 MiB with
    nvidia-smi unchanged, and a TensorRT context needing 3.9 GB then failed to
    allocate — until empty_cache() ran. Training, ONNX export and the Test/
    Studio tabs all share this one process (train.py runs train_model through
    run_in_executor, not a subprocess), so the pool has to be handed back at
    the end of every torch-side job or the next TensorRT load starves.
    """
    import gc

    import torch

    if not torch.cuda.is_available():
        return
    reserved_before = torch.cuda.memory_reserved()
    gc.collect()
    torch.cuda.empty_cache()
    freed = (reserved_before - torch.cuda.memory_reserved()) / 2**20
    if freed >= 1:
        logger.info(f"[anomaly] released {freed:.0f} MiB of torch CUDA cache after {where}")


def train_model(
    project_id: str,
    model_id: str,
    request: AnomalyTrainRequest,
    checkpoint_dir: Path,
    progress_cb: Optional[Callable[[str, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    excluded_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Train + evaluate one PatchCore/Padim model. Saves:
      {checkpoint_dir}/{model_id}.ckpt            — Lightning checkpoint
      {checkpoint_dir}/{model_id}_test_results.json — per-image eval sidecar

    Returns the metrics dict (see _compute_metrics), plus checkpoint_path.

    Thin wrapper over _train_model so the GPU pool is handed back on *every*
    exit path — including TrainingCancelled and the mid-run raises — since a
    failed run holds just as much memory as a successful one.
    """
    try:
        return _train_model(
            project_id, model_id, request, checkpoint_dir,
            progress_cb=progress_cb, cancel_check=cancel_check,
            excluded_paths=excluded_paths,
        )
    finally:
        release_torch_cuda_cache("training")


def _train_model(
    project_id: str,
    model_id: str,
    request: AnomalyTrainRequest,
    checkpoint_dir: Path,
    progress_cb: Optional[Callable[[str, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    excluded_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
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
    # Only stage when something is actually excluded — the common case keeps
    # pointing straight at the dataset, so this adds no cost or risk to it.
    staging_root = None
    if excluded_paths:
        staging_root = dataset_fs.dataset_dir(project_id).parent / "_staging" / model_id
        _stage_dataset(project_id, excluded_paths, staging_root)
    datamodule = _build_datamodule(project_id, request, staging_root=staging_root)
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

    # The staging tree is symlinks only, so removing it never touches a real
    # image — but leaving it behind would confuse the next run's dataset counts.
    if staging_root is not None:
        shutil.rmtree(staging_root, ignore_errors=True)

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


def _trained_image_size(model) -> Optional[int]:
    """The square resize the model was actually TRAINED at, read back off the
    checkpoint's PreProcessor, or None if it can't be determined."""
    transform = getattr(getattr(model, "pre_processor", None), "transform", None)
    for t in getattr(transform, "transforms", []) or []:
        size = getattr(t, "size", None)
        if size and len(size) == 2 and size[0] == size[1]:
            return int(size[0])
    return None


def export_onnx(algorithm: str, checkpoint_path: Path, export_dir: Path, image_size: int) -> Path:
    """Export one checkpoint to ONNX.

    Wrapped so the backbone this loads onto the GPU doesn't stay parked in
    torch's caching allocator — export is normally followed immediately by a
    TensorRT build, which is exactly the allocation that fails when it does
    (see release_torch_cuda_cache).
    """
    try:
        return _export_onnx(algorithm, checkpoint_path, export_dir, image_size)
    finally:
        release_torch_cuda_cache("onnx export")


def _export_onnx(algorithm: str, checkpoint_path: Path, export_dir: Path, image_size: int) -> Path:
    import torch
    from anomalib.engine import Engine

    model = load_model_from_checkpoint(algorithm, checkpoint_path)

    # Exporting at a different resolution than the model was trained at
    # silently destroys accuracy (see _build_model) -- every image saturates to
    # pred_score 1.0. Trust the checkpoint over the caller's argument, and say
    # so loudly: a stale model record's image_size must not produce a
    # confidently-wrong engine.
    trained = _trained_image_size(model)
    if trained is not None and trained != image_size:
        logger.warning(
            f"[anomaly export] image_size mismatch: checkpoint {checkpoint_path.name} was "
            f"trained at {trained}x{trained} but export requested {image_size}x{image_size}. "
            f"Exporting at {trained} instead — a mismatched export saturates every score to 1.0."
        )
        image_size = trained

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
