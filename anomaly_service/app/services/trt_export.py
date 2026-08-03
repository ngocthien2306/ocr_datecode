"""
Build a standalone TensorRT .engine from an exported anomaly-model ONNX,
using the TensorRT Python Builder API directly (no trtexec binary --
tensorrt's pip wheel doesn't ship one). Same approach and same TensorRT
11.1.0 API quirks handled here as ocr_datecode/scripts/trt_build_utils.py
(that script isn't imported directly -- this service ships its own copy so
it stays independently deployable rather than reaching into a sibling
service's scripts/ directory).

Complements the existing onnxruntime-TensorrtExecutionProvider "verify"
step in export.py (an internal, ORT-managed engine cache used at live
inference time) with a real portable .engine file the user can download,
matching how the OCR pipeline's TensorRT engines are shipped.
"""
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import onnx
import tensorrt as trt
from onnxconverter_common import float16 as onnx_float16

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
_INVALID_NODE_RE = re.compile(r"Invalid Node - (\S+)")

# onnxconverter_common's post-conversion "remove redundant Cast pairs"
# cleanup crashes (AttributeError: 'list' object has no attribute 'input')
# when a Cast node's output fans out to more than one downstream node. It's
# a pure optimization pass (leaves a few harmless extra Cast pairs if
# skipped; TensorRT folds those away anyway), so disable it rather than
# work around the crash.
onnx_float16.remove_unnecessary_cast_node = lambda graph_proto: None


def _load_onnx_bytes(onnx_path: Path, fp16: bool, node_block_list) -> bytes:
    model = onnx.load(str(onnx_path))
    if fp16:
        already_fp16 = any(
            init.data_type == onnx.TensorProto.FLOAT16 for init in model.graph.initializer
        )
        if not already_fp16:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = onnx_float16.convert_float_to_float16(
                    model, keep_io_types=True, node_block_list=list(node_block_list or [])
                )
    return model.SerializeToString()


def build_engine_from_onnx(
    onnx_path: Path,
    engine_path: Path,
    input_name: str,
    min_shape: Tuple[int, ...],
    opt_shape: Tuple[int, ...],
    max_shape: Tuple[int, ...],
    fp16: bool = False,
    workspace_gib: int = 4,
    max_fp32_retries: int = 25,
    log=None,
) -> Path:
    """Build a dynamic-batch TensorRT engine and write it to engine_path."""
    _log = log or (lambda msg: None)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    node_block_list = set()
    network = None
    builder = None
    for attempt in range(max_fp32_retries + 1):
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network()
        parser = trt.OnnxParser(network, TRT_LOGGER)
        onnx_bytes = _load_onnx_bytes(onnx_path, fp16, node_block_list)
        ok = parser.parse(onnx_bytes)
        if ok:
            break

        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        newly_bad = {m for e in errors for m in _INVALID_NODE_RE.findall(e)} - node_block_list
        if not fp16 or not newly_bad:
            for e in errors:
                print(e, file=sys.stderr)
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

        _log(f"fp16 dtype mismatch, forcing fp32 for: {sorted(newly_bad)} (retry {attempt + 1})")
        node_block_list |= newly_bad
    else:
        raise RuntimeError(f"Failed to parse ONNX after {max_fp32_retries} fp32-fallback retries: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib * (1 << 30))

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    _log(f"building engine: input='{input_name}' min={min_shape} opt={opt_shape} max={max_shape} fp16={fp16}")
    t0 = time.time()
    host_mem = builder.build_serialized_network(network, config)
    if host_mem is None:
        raise RuntimeError(f"Engine build failed for {onnx_path}")
    # TensorRT 10+ returns an IHostMemory wrapper (no __len__), not raw bytes.
    serialized = bytes(memoryview(host_mem))

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    _log(f"wrote {engine_path} ({len(serialized) / (1024 * 1024):.1f} MB, {time.time() - t0:.1f}s)")
    return engine_path


def onnx_input_info(onnx_path: Path) -> Tuple[str, list]:
    """Return (input_tensor_name, static_dims) -- dims[0] (batch) is None if dynamic."""
    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        dims.append(None if d.dim_param else d.dim_value)
    return inp.name, dims
