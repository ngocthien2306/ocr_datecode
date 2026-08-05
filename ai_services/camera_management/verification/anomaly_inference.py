import gc
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 -- creates this process's CUDA context, on the main/import thread only
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logger.warning("[AnomalyInference] TensorRT/pycuda not available — falling back to ONNX Runtime only")

try:
    import torch
    import torchvision.transforms.v2.functional as tv_f
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error(
        "[AnomalyInference] torch/torchvision not installed — anomaly detection is disabled "
        "(no way to reproduce the training-time resize; see module docstring)"
    )

# Concurrent-inference cap per engine_path. Each slot is a full TensorRT
# execution context, and a context is NOT cheap: measured 1.07 GB of activation
# memory for the 512x512 PatchCore engine (on top of ~482 MB of shared weights),
# and the size is driven by the memory-bank distance matrix, not by the batch
# profile — rebuilding at max_batch=1 gave byte-identical activation. Four slots
# would therefore reserve ~4.8 GB for a single model, which OOM'd this 16 GB card
# during testing. Concurrency here is bounded by how many cameras share one
# model (two, for front/back label), so two is the working ceiling; anything
# beyond that queues rather than allocating.
_MAX_SLOTS_PER_ENGINE = 2

# How long _acquire_trt_slot waits on the pool before re-reading it. Only there
# so a pool dropped by release_models() can't strand a waiter forever; a normal
# wait is the length of one inference (~30 ms), far under this.
_SLOT_WAIT_TIMEOUT_S = 2.0

# A recipe change that swaps the anomaly model requires restarting ai_services
# to take effect — same as a newly trained char-classifier model today (see
# backend's _trigger_restart_all) — so in practice each of these holds at
# most a few entries per process lifetime.
_primary_ctx = None          # single retained primary-context handle, see _get_primary_context
_primary_ctx_lock = threading.Lock()

_onnx_sessions: Dict[str, Any] = {}
_trt_engines: Dict[str, Any] = {}                   # engine_path -> deserialized ICudaEngine (shared, read-only)
_trt_slot_pools: Dict[str, "queue.Queue"] = {}       # engine_path -> pool of free _TRTSlot
_trt_slot_counts: Dict[str, int] = {}                # engine_path -> slots created so far
_trt_pool_lock = threading.Lock()


class _TRTSlot:
    """One execution context + its own H2D/D2H buffers + its own CUDA
    stream. TensorRT execution contexts are not safe to share *across*
    threads (only within one thread at a time each), but multiple
    independent contexts created from the same engine ARE safe to drive
    concurrently from different threads — this is what lets two cameras
    on the same anomaly model run genuinely in parallel instead of
    queueing behind a single context."""

    def __init__(self, engine):
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.input_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_names = [n for n in names if n != self.input_name]

        # Square HxW read off the engine's profile, not off recipe config —
        # a stale recipe image_size must never silently feed the wrong-sized
        # tensor into a graph built for a fixed size.
        profile = engine.get_tensor_profile_shape(self.input_name, 0)
        self.image_size = int(profile[2][2])

        self.context.set_input_shape(self.input_name, (1, 3, self.image_size, self.image_size))
        self.h_input = cuda.pagelocked_empty(3 * self.image_size * self.image_size, np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.h_outs, self.d_outs, self.out_shapes = {}, {}, {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            self.out_shapes[name] = shape
            h = cuda.pagelocked_empty(int(np.prod(shape)), np.float32)
            self.h_outs[name] = h
            self.d_outs[name] = cuda.mem_alloc(h.nbytes)

    def close(self) -> None:
        """Free this slot's CUDA allocations explicitly. Caller must have the
        primary context pushed.

        Dropping the last Python reference is NOT enough. pycuda's GC-time
        cleanup remembers the thread that allocated, and every slot is
        allocated on an ephemeral pool worker — warmup() and
        product_verifier._run_anomaly_checks() both build a per-call
        ThreadPoolExecutor — so by teardown that thread is long gone and
        pycuda logs "device_allocation in out-of-thread context could not be
        cleaned up" and leaks the buffer. Freeing by hand under the pushed
        primary context is the path that actually returns the memory.
        """
        for buf in (self.d_input, *self.d_outs.values()):
            try:
                buf.free()
            except Exception:
                pass
        self.d_outs.clear()
        # pagelocked_empty returns an ndarray whose .base owns the pinned
        # allocation; that is what has to be freed, not the array.
        for host in (self.h_input, *self.h_outs.values()):
            base = getattr(host, "base", None)
            if base is not None and hasattr(base, "free"):
                try:
                    base.free()
                except Exception:
                    pass
        self.h_outs.clear()
        # Releasing the execution context here (rather than at GC) keeps its
        # ~1 GB of activation memory from outliving the buffers above.
        self.context = None
        self.stream = None

    def infer(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """x: (3, image_size, image_size) float32, already preprocessed."""
        self.context.set_input_shape(self.input_name, (1, 3, self.image_size, self.image_size))
        np.copyto(self.h_input, x.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        for name in self.output_names:
            self.context.set_tensor_address(name, int(self.d_outs[name]))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.h_outs[name], self.d_outs[name], self.stream)
        self.stream.synchronize()
        return {
            name: self.h_outs[name][: int(np.prod(self.out_shapes[name]))].reshape(self.out_shapes[name]).copy()
            for name in self.output_names
        }


def _get_trt_engine(engine_path: str):
    """Must be called with the calling thread's pycuda context already
    pushed (see predict()/_predict_trt) — deserialize_cuda_engine touches
    the CUDA context same as any other pycuda/TensorRT call."""
    engine = _trt_engines.get(engine_path)
    if engine is not None:
        return engine
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    _trt_engines[engine_path] = engine
    logger.info(f"[AnomalyInference] loaded TensorRT engine {Path(engine_path).name}")
    return engine


def _acquire_trt_slot(engine_path: str) -> _TRTSlot:
    """Must be called with the calling thread's pycuda context already
    pushed."""
    while True:
        with _trt_pool_lock:
            pool = _trt_slot_pools.setdefault(engine_path, queue.Queue())
            created = _trt_slot_counts.get(engine_path, 0)
            if pool.empty() and created < _MAX_SLOTS_PER_ENGINE:
                _trt_slot_counts[engine_path] = created + 1
                return _TRTSlot(_get_trt_engine(engine_path))
        # Either a free slot is sitting in the pool, or we're at the
        # concurrency cap — wait for one to come back rather than growing GPU
        # memory unbounded. The wait is bounded and retried because
        # release_models() can drop this pool while we hold a reference to it:
        # the slot we were queued behind is then torn down instead of returned,
        # and an unbounded get() on the orphaned queue would never wake. Looping
        # re-reads the live pool and, finding it empty, allocates a fresh slot.
        try:
            return pool.get(timeout=_SLOT_WAIT_TIMEOUT_S)
        except queue.Empty:
            continue


def _release_trt_slot(engine_path: str, slot: _TRTSlot) -> None:
    """Return a slot to its pool, or tear it down if the pool is gone.

    release_models() can drop the pool mid-inference — a recipe change while a
    trigger is in flight. Before this, that raced into
    `KeyError: <engine_path>` inside the finally-block of _predict_trt, which
    surfaced as a skipped frame (verified with two threads inferring against a
    model being released in a loop). A slot whose pool has vanished belongs to
    a model nobody will ask for again, so free it rather than resurrecting the
    pool — resurrecting would silently re-pin the very engine the release was
    meant to drop. _predict_trt still has the primary context pushed here,
    which is what close() requires.
    """
    with _trt_pool_lock:
        pool = _trt_slot_pools.get(engine_path)
        if pool is not None:
            pool.put(slot)
            return
    slot.close()


def _get_primary_context():
    """One retained handle to device 0's primary context, shared by every
    thread, created once.

    Calling cuda.Device(0).retain_primary_context() per thread returns a
    *different Python Context object* each time, and pycuda records on each one
    the thread that built it. Any allocation made under such an object can then
    only be cleaned up from that same thread — and every slot here is allocated
    on an ephemeral pool worker, so by teardown time that thread no longer
    exists. The result was pycuda skipping the free and warning
    "device_allocation in out-of-thread context could not be cleaned up",
    leaking ~8 MiB per recipe change (measured over 6 swap cycles).

    Sharing one handle fixes it: measured 0 warnings freeing from the main
    thread *or* an unrelated thread, versus 1 per allocation with per-thread
    handles. The underlying CUDA primary context is the same one
    pycuda.autoinit set up at import, so pushing it stays safe to nest.
    """
    global _primary_ctx
    if _primary_ctx is None:
        with _primary_ctx_lock:
            if _primary_ctx is None:
                _primary_ctx = cuda.Device(0).retain_primary_context()
    return _primary_ctx


def release_models(keep_paths: Optional[Set[str]] = None) -> int:
    """Drop cached engines/sessions that the current recipe no longer uses.

    ``keep_paths`` is the set of onnx_path/engine_path values the newly loaded
    recipe binds to; anything cached outside it is freed. Pass ``None`` to
    release everything (recipe stopped).

    Why this has to exist: the caches below are plain dicts with no eviction,
    so before this a recipe change that swapped the anomaly model left the old
    engine pinned for the life of the process — measured ~1.6 GB for one
    512x512 model, times two execution contexts. The module docstring used to
    say a restart made that moot, but nothing guarantees the restart actually
    happens, and a stale 3 GB is exactly what makes the *new* model's contexts
    fail to allocate.

    Returns the number of cached models dropped.

    Freeing is not just dropping references: _TRTSlot owns pycuda
    pagelocked/device allocations, and pycuda frees those against the CUDA
    context that was current at allocation time. So the primary context is
    pushed for the teardown, exactly as _predict_trt does for inference.
    """
    with _trt_pool_lock:
        stale_engines = [p for p in _trt_engines if keep_paths is None or p not in keep_paths]
        stale_onnx = [p for p in _onnx_sessions if keep_paths is None or p not in keep_paths]
        if not stale_engines and not stale_onnx:
            return 0

        slots_to_free = []
        for path in stale_engines:
            pool = _trt_slot_pools.pop(path, None)
            _trt_slot_counts.pop(path, None)
            while pool is not None and not pool.empty():
                # A slot still checked out by an in-flight inference is NOT in
                # the pool, so it stays alive until its thread returns it — at
                # which point put() lands in a pool nobody holds and it is
                # collected. Never yank a context out from under a running
                # execute_async_v3.
                slots_to_free.append(pool.get_nowait())
            _trt_engines.pop(path, None)

        for path in stale_onnx:
            _onnx_sessions.pop(path, None)

    if slots_to_free and TRT_AVAILABLE:
        ctx = None
        try:
            ctx = _get_primary_context()
            ctx.push()
            for slot in slots_to_free:
                slot.close()
            slots_to_free.clear()
            gc.collect()
        except Exception as e:
            logger.warning(f"[AnomalyInference] CUDA teardown warning while releasing models: {e}")
        finally:
            if ctx is not None:
                try:
                    ctx.pop()
                except Exception:
                    pass
    else:
        slots_to_free.clear()
        gc.collect()

    dropped = len(stale_engines) + len(stale_onnx)
    kept = "all" if keep_paths is None else f"{len(keep_paths)} still in use"
    logger.info(f"[AnomalyInference] released {dropped} cached model(s) ({kept} kept)")
    return dropped


def _get_onnx_session(onnx_path: str):
    sess = _onnx_sessions.get(onnx_path)
    if sess is not None:
        return sess

    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.enable_mem_pattern = False
    sess_options.enable_cpu_mem_arena = False
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.log_severity_level = 3

    # Plain CUDA EP, fp32 -- no TensorRT EP here. This path is only a
    # fallback for models that haven't had a TensorRT engine exported yet;
    # keeping it fp32/CUDA-only avoids reintroducing the fp16 accuracy loss
    # this module was specifically rewritten to get rid of.
    available = set(ort.get_available_providers())
    providers = (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else []) + ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    active = sess.get_providers()[0]
    logger.info(f"[AnomalyInference] loaded ONNX {Path(onnx_path).name} on {active} (fallback path, no engine_path set)")
    _onnx_sessions[onnx_path] = sess
    return sess


def crop_label(frame: np.ndarray, cx: float, cy: float, w: float, h: float, angle: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Same OBB crop WrinkledSegmenterTRT.predict_batch's caller uses —
    calling the shared @staticmethod directly so the two checks are always
    pixel-identical on the same product_box, whichever one is active."""
    from .wrinkle_segmenter import WrinkledSegmenterTRT
    return WrinkledSegmenterTRT.crop_from_obb(frame, cx, cy, w, h, angle)


def _preprocess(crop_bgr: np.ndarray, image_size: int) -> np.ndarray:
    """BGR uint8 crop -> (3, image_size, image_size) float32 in [0,1],
    resized with torchvision's antialiased bilinear resize -- see module
    docstring for why nothing else (cv2, PIL) reproduces training-time
    scores. Normalization is NOT applied here; it's baked into the graph."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    resized = tv_f.resize(t, [image_size, image_size], antialias=True)
    return np.ascontiguousarray(resized.numpy())


def _extract_maps(out: Dict[str, np.ndarray], orig_hw: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    h, w = orig_hw
    mask = out.get("pred_mask")
    amap = out.get("anomaly_map")
    if mask is not None:
        mask = np.asarray(mask).reshape(mask.shape[-2], mask.shape[-1]).astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
    if amap is not None:
        amap = np.asarray(amap).reshape(amap.shape[-2], amap.shape[-1]).astype(np.float32)
        if amap.shape != (h, w):
            amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
    return mask, amap


def extract_regions(
    mask: Optional[np.ndarray],
    min_region_area: float = 0.0,
    amap: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Connected-component analysis on a binary (H, W) mask in crop-pixel-
    space -> discrete regions, mirroring the per-instance shape
    WrinkledSegmenterTRT already produces from YOLO-seg (area, contour,
    corners) so the exact same min_region_area / max_region_area / total-area
    rules — and the same downstream drawing code — apply unchanged regardless
    of whether wrinkle-YOLO or this PatchCore/Padim model found the regions.

    Every geometry field returned here (`contour`, `corners`) is in CROP
    space, not frame space — the crop is a rotated, offset sub-image of the
    frame. Callers that draw these on the full frame must back-project them
    first (build_anomaly_check does); drawing them raw puts the mask in the
    wrong place. `area` is deliberately left in crop pixels so it stays
    comparable to the min/max area thresholds.

    A single blobby anomaly_map/pred_mask has no notion of "instances" the
    way YOLO-seg's separate detections do, so instances here are just
    connected components of the thresholded mask — adjacent-but-distinct
    defects that happen to touch will merge into one region, which YOLO-seg
    would have kept separate. Acceptable for area-threshold purposes (their
    combined area still counts), just not equivalent instance-for-instance.
    """
    if mask is None or not mask.any():
        return []
    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    regions = []
    for label_id in range(1, n_labels):
        blob = (labels == label_id).astype(np.uint8)
        area = int(blob.sum())
        if min_region_area > 0 and area < min_region_area:
            continue
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(contour)
        mean_score = float(amap[blob > 0].mean()) if amap is not None else None
        regions.append({
            "label_id": label_id,
            "area": area,
            "score": mean_score,
            "class": "anomaly",
            "contour": contour.tolist(),
            "corners": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        })
    return regions


def _predict_trt(engine_path: str, crop_bgr: np.ndarray) -> Dict[str, Any]:
    # Push this thread's pycuda context for the whole call and pop it again
    # before returning -- never "push and leave", even on a persistent
    # thread. product_verifier.py runs this from freshly-created, per-call
    # ThreadPoolExecutors (torn down right after each trigger) — verified
    # empirically that a thread dying with a pushed-but-unpopped context
    # makes pycuda hard-abort the *entire process* on pool shutdown, not
    # just warn. _get_primary_context() hands back the same shared context
    # `pycuda.autoinit` already created on the main thread at import time,
    # so this is safe to nest with it — and sharing one handle is what lets
    # release_models() free these slots later (see _get_primary_context).
    ctx = _get_primary_context()
    ctx.push()
    try:
        slot = _acquire_trt_slot(engine_path)
        try:
            x = _preprocess(crop_bgr, slot.image_size)
            out = slot.infer(x)
            score = float(out["pred_score"].reshape(-1)[0])
            mask, amap = _extract_maps(out, crop_bgr.shape[:2])
            return {"skipped": False, "pred_score": score, "backend": "tensorrt", "mask": mask, "anomaly_map": amap}
        finally:
            _release_trt_slot(engine_path, slot)
    finally:
        ctx.pop()


def _predict_onnx(onnx_path: str, crop_bgr: np.ndarray, recipe_image_size: int) -> Dict[str, Any]:
    sess = _get_onnx_session(onnx_path)
    input_name = sess.get_inputs()[0].name
    graph_h = sess.get_inputs()[0].shape[2]
    image_size = graph_h if isinstance(graph_h, int) else recipe_image_size
    x = _preprocess(crop_bgr, image_size)[None, ...]
    outputs = sess.run(None, {input_name: x})
    out_names = [o.name for o in sess.get_outputs()]
    out = dict(zip(out_names, outputs))
    score = float(np.asarray(out["pred_score"]).reshape(-1)[0])
    mask, amap = _extract_maps(out, crop_bgr.shape[:2])
    return {"skipped": False, "pred_score": score, "backend": "onnx", "mask": mask, "anomaly_map": amap}


def predict(
    crop_bgr: Optional[np.ndarray],
    onnx_path: Optional[str] = None,
    engine_path: Optional[str] = None,
    image_size: int = 256,
) -> Dict[str, Any]:
    """Run one anomaly-detection inference on a label crop. Prefers
    engine_path (TensorRT, via pycuda); falls back to onnx_path (ONNX
    Runtime, CUDA EP, fp32) when no engine has been exported yet.

    Never raises — a broken model must not take down the whole inspection
    pipeline. On any failure returns skipped=True with the error recorded,
    so the caller can decide whether to fail-open (skip, like a disabled
    check) or fail-closed (treat as abnormal); build_anomaly_check() fails
    open, matching how WrinkledSegmenterTRT.build_wrinkled_check treats a
    missing/failed detection today.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {"skipped": True, "reason": "empty crop", "pred_score": 0.0}
    if not TORCH_AVAILABLE:
        return {"skipped": True, "reason": "torch/torchvision not installed", "pred_score": 0.0}
    try:
        if engine_path and TRT_AVAILABLE and Path(engine_path).exists():
            return _predict_trt(engine_path, crop_bgr)
        if onnx_path and Path(onnx_path).exists():
            return _predict_onnx(onnx_path, crop_bgr, image_size)
        return {"skipped": True, "reason": "no onnx_path/engine_path configured or files missing", "pred_score": 0.0}
    except Exception as e:
        logger.exception(f"[AnomalyInference] inference failed (engine_path={engine_path}, onnx_path={onnx_path})")
        return {"skipped": True, "reason": f"inference error: {e}", "pred_score": 0.0, "error": str(e)}


def build_anomaly_check(
    pred: Dict[str, Any],
    threshold: float,
    min_area: float = 0.0,
    min_region_area: float = 0.0,
    max_region_area: float = 0.0,
    mask_polygons: Optional[List[np.ndarray]] = None,
    mask_overlap_threshold: float = 0.6,
    pixel_threshold: Optional[float] = None,
    cx: float = 0.0,
    cy: float = 0.0,
    angle: float = 0.0,
    crop_offset: Tuple[int, int] = (0, 0),
    frame_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    if pred.get("skipped"):
        return {
            "ok": True, "skipped": True, "reason": pred.get("reason", "skipped"),
            "has_wrinkled": False, "wrinkled_count": 0, "wrinkled_boxes": [],
            "anomaly_score": pred.get("pred_score", 0.0), "anomaly_threshold": threshold,
        }

    score = pred["pred_score"]
    amap = pred.get("anomaly_map")
    if pixel_threshold is not None and amap is not None:
        mask_for_regions = amap > pixel_threshold
    else:
        mask_for_regions = pred.get("mask")

    region_logic_on = (min_area > 0 or max_region_area > 0) and mask_for_regions is not None and frame_shape is not None

    if not region_logic_on:
        is_abnormal = score >= threshold
        return {
            "ok": not is_abnormal,
            "skipped": False,
            "has_wrinkled": is_abnormal,   # reused key name — see module docstring
            "wrinkled_count": 1 if is_abnormal else 0,
            "wrinkled_boxes": [],          # area logic off — no per-region boxes to draw
            "anomaly_score": score,
            "anomaly_threshold": threshold,
            "backend": pred.get("backend"),
        }

    from .wrinkle_segmenter import WrinkledSegmenterTRT

    regions = extract_regions(mask_for_regions, min_region_area, amap)

    # Back-project every candidate region from crop space to frame space. The
    # crop fed to the model is a rotated+offset sub-image of the frame, so a
    # region's crop-space contour is NOT where it lives on the frame — it has
    # to be placed at crop_offset and un-rotated about (cx, cy) first. Both the
    # exclusion-polygon overlap test AND the contour/corners handed downstream
    # for drawing must use the frame-space version; only `area` stays in crop
    # pixels, so it stays comparable to the min/max area thresholds (same split
    # build_wrinkled_check uses for YOLO-seg masks).
    frame_masks = []
    if regions:
        crop_masks = np.zeros((len(regions),) + mask_for_regions.shape, dtype=np.float32)
        for i, region in enumerate(regions):
            cv2.drawContours(crop_masks[i], [np.array(region["contour"], dtype=np.int32)], -1, 1, thickness=cv2.FILLED)
        frame_masks = WrinkledSegmenterTRT.back_project_masks(crop_masks, cx, cy, angle, crop_offset, frame_shape)

    union_mask = None
    if mask_polygons:
        union_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        for poly in mask_polygons:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(union_mask, [pts], 1)

    crop_area = int(mask_for_regions.shape[0] * mask_for_regions.shape[1])
    wrinkled_boxes = []
    excluded_by_mask = 0
    for region, frame_mask in zip(regions, frame_masks):
        if union_mask is not None:
            binary_region = frame_mask > 0.5
            region_pixels = int(binary_region.sum())
            if region_pixels > 0:
                overlap_ratio = int(np.logical_and(binary_region, union_mask > 0).sum()) / region_pixels
                if overlap_ratio >= mask_overlap_threshold:
                    excluded_by_mask += 1
                    continue

        frame_contour = WrinkledSegmenterTRT._mask_to_contour(frame_mask)
        if frame_contour is None:
            continue
        region.pop("label_id", None)
        region["contour"] = frame_contour
        region["corners"] = WrinkledSegmenterTRT._contour_to_bbox_corners(frame_contour)
        region["area_pct"] = round(region["area"] / max(crop_area, 1) * 100, 2)
        wrinkled_boxes.append(region)

    total_area = sum(r["area"] for r in wrinkled_boxes)
    has_critical = max_region_area > 0 and any(r["area"] >= max_region_area for r in wrinkled_boxes)
    total_exceeded = min_area > 0 and total_area >= min_area
    # AREA DECIDES once area logic is configured. The image-level pred_score used
    # to OR into this, which made a frame fail on a high score even when the
    # defect area was well under min_area — the operator tunes areas they can see
    # and measure on the frame, not an opaque 0-1 score. The score is still
    # reported below for debugging; it just no longer votes.
    # With area logic OFF (min_area == 0 and max_region_area == 0) the early
    # `not region_logic_on` return above still decides purely on the score, so
    # templates that were never given area thresholds are unaffected.
    has_wrinkled = has_critical or total_exceeded
    triggered_by = "critical" if has_critical else ("total" if total_exceeded else None)

    return {
        "ok": not has_wrinkled,
        "skipped": False,
        "has_wrinkled": has_wrinkled,
        "triggered_by": triggered_by,
        "wrinkled_count": len(wrinkled_boxes),
        "total_area": total_area,
        "min_area": min_area,
        "min_region_area": min_region_area,
        "max_region_area": max_region_area,
        "excluded_by_mask": excluded_by_mask,
        "wrinkled_boxes": wrinkled_boxes,
        "anomaly_score": score,
        "anomaly_threshold": threshold,
        "backend": pred.get("backend"),
    }


def warmup(
    onnx_path: Optional[str] = None,
    engine_path: Optional[str] = None,
    image_size: int = 256,
    slots: int = 1,
) -> bool:
    """Pay the first-inference cost up front instead of on the first real
    trigger. Everything predict() needs is built lazily on first use —
    deserializing the .engine, creating the execution context, allocating
    pinned host + device buffers, and TensorRT's own per-shape kernel
    selection — which measured ~9s in production and blew straight past
    delay_reject (1.45s), so the first bottle after every service restart
    was inspected but never rejected.

    `slots` should be the number of frames that can hit this same model
    concurrently (i.e. how many cameras share it). Warming only one slot
    leaves the second camera allocating its own context/buffers mid-trigger,
    which just moves a smaller version of the same stall to the first run.

    Always driven through a worker pool, even for a single slot, so it walks
    the exact same path a real trigger does (pool thread + CUDA context push)
    rather than a main-thread-only shortcut that could leave some per-thread
    setup unpaid.

    Never raises — a model that can't be warmed still gets retried lazily on
    the first real frame, same as before.
    """
    from concurrent.futures import ThreadPoolExecutor

    dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    name = Path(engine_path or onnx_path or "?").name
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max(1, slots)) as pool:
        preds = list(pool.map(
            lambda _: predict(dummy, onnx_path=onnx_path, engine_path=engine_path, image_size=image_size),
            range(max(1, slots)),
        ))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    failed = next((p for p in preds if p.get("skipped")), None)
    if failed is not None:
        logger.warning(
            f"[AnomalyInference] warmup FAILED for {name} after {elapsed_ms:.0f}ms: "
            f"{failed.get('reason')} — first real frame will pay the load cost instead"
        )
        return False
    logger.info(
        f"[AnomalyInference] warmed up {name} in {elapsed_ms:.0f}ms "
        f"({slots} slot(s), backend={preds[0].get('backend')})"
    )
    return True


def release_session(onnx_path: Optional[str] = None, engine_path: Optional[str] = None) -> bool:
    """Drop cached session/engine state (e.g. on graceful shutdown or a
    recipe model swap). Returns whether anything was actually held."""
    released = False
    if onnx_path is not None:
        released = _onnx_sessions.pop(onnx_path, None) is not None or released
    if engine_path is not None:
        with _trt_pool_lock:
            released = _trt_engines.pop(engine_path, None) is not None or released
            _trt_slot_pools.pop(engine_path, None)
            _trt_slot_counts.pop(engine_path, None)
    return released
