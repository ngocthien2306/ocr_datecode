"""
Build a standalone TensorRT .engine from an exported ONNX, using the TensorRT
Python Builder API directly (the pip wheel ships no trtexec).

Independent copy of the same logic as scripts/trt_build_utils.py and
anomaly_service/app/services/trt_export.py, for the same reason those are
separate: a training service that reaches into a sibling's scripts/ directory
stops being independently deployable.

The profile matches the `smtr_attn` spec that built the in-production engine:
min 1x3x32x32 / opt 4x3x32x320 / max 16x3x32x2000. Engines are NOT forward
compatible across TensorRT majors — one built by 8.x/9.x/10.x will not
deserialize under the 11.1 runtime this service pins.
"""
import logging
import re
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_INVALID_NODE_RE = re.compile(r"Invalid Node - (\S+)")

# Width range 32..2000 and batch 1..16 come from this repo's own prior build
# commands for the SMTR engines (scripts/build_ocr_engines.py).
DEFAULT_MIN = (1, 3, 32, 32)
DEFAULT_OPT = (4, 3, 32, 320)
DEFAULT_MAX = (16, 3, 32, 2000)


def _load_onnx_bytes(onnx_path: Path, fp16: bool, node_block_list) -> bytes:
    import onnx
    from onnxconverter_common import float16 as onnx_float16

    # onnxconverter_common's post-conversion "remove redundant Cast pairs" pass
    # crashes (AttributeError: 'list' object has no attribute 'input') when a
    # Cast output fans out to more than one node. It is a pure optimisation —
    # skipping it leaves a few harmless Cast pairs that TensorRT folds anyway.
    onnx_float16.remove_unnecessary_cast_node = lambda graph_proto: None

    model = onnx.load(str(onnx_path))
    if fp16:
        already_fp16 = any(
            init.data_type == onnx.TensorProto.FLOAT16 for init in model.graph.initializer
        )
        if not already_fp16:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = onnx_float16.convert_float_to_float16(
                    model, keep_io_types=True, node_block_list=list(node_block_list or []),
                )
    return model.SerializeToString()


def onnx_io_info(onnx_path: Path) -> Tuple[str, List[Optional[int]], List[str]]:
    """(input_name, input_dims, output_names). A dim is None when dynamic."""
    import onnx

    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    dims = [None if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim]
    return inp.name, dims, [o.name for o in model.graph.output]


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    input_name: str = "image",
    min_shape: Tuple[int, ...] = DEFAULT_MIN,
    opt_shape: Tuple[int, ...] = DEFAULT_OPT,
    max_shape: Tuple[int, ...] = DEFAULT_MAX,
    fp16: bool = True,
    workspace_gib: int = 4,
    max_fp32_retries: int = 25,
    log=None,
) -> Path:
    """Build a dynamic batch+width engine and write it to engine_path.

    workspace_gib is a memory-pool CEILING, not a reservation — measured at both
    1 and 4 GiB the build peaked at the same 612 MiB, so there is nothing to gain
    by lowering it.
    """
    import tensorrt as trt

    _log = log or (lambda msg: logger.info(f"[trt] {msg}"))
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    trt_logger = trt.Logger(trt.Logger.WARNING)
    node_block_list = set()
    network = builder = None

    for attempt in range(max_fp32_retries + 1):
        builder = trt.Builder(trt_logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, trt_logger)
        if parser.parse(_load_onnx_bytes(onnx_path, fp16, node_block_list)):
            break

        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        newly_bad = {m for e in errors for m in _INVALID_NODE_RE.findall(e)} - node_block_list
        if not fp16 or not newly_bad:
            for e in errors:
                _log(f"parse error: {e}")
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")
        # A node whose fp16 form TensorRT rejects gets pinned back to fp32 and
        # the graph is re-converted. Iterative because one fix can surface the
        # next.
        _log(f"fp16 dtype mismatch, forcing fp32 for {sorted(newly_bad)} (retry {attempt + 1})")
        node_block_list |= newly_bad
    else:
        raise RuntimeError(
            f"Failed to parse {onnx_path} after {max_fp32_retries} fp32-fallback retries"
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib * (1 << 30))
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    _log(f"building: input={input_name!r} min={min_shape} opt={opt_shape} max={max_shape} fp16={fp16}")
    t0 = time.time()
    host_mem = builder.build_serialized_network(network, config)
    if host_mem is None:
        raise RuntimeError(f"Engine build failed for {onnx_path}")
    # TensorRT 10+ returns an IHostMemory wrapper (no __len__), not raw bytes.
    serialized = bytes(memoryview(host_mem))

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    _log(f"wrote {engine_path.name} ({len(serialized) / 1e6:.1f} MB, {time.time() - t0:.1f}s)")
    return engine_path


def inspect_engine(engine_path: Path) -> dict:
    """Deserialize the engine and report its bindings.

    The output count is the check that matters: ai_services'
    TextRecognizerSMTRTRT asserts exactly two, and an engine with three (built
    from the older attn ONNX) fails at load time in production rather than here.
    """
    import tensorrt as trt

    with open(engine_path, "rb") as f:
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Engine failed to deserialize: {engine_path}")

    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        entry = {"name": name, "shape": list(engine.get_tensor_shape(name))}
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            entry["profile"] = [list(s) for s in engine.get_tensor_profile_shape(name, 0)]
            inputs.append(entry)
        else:
            outputs.append(entry)
    return {
        "inputs": inputs,
        "outputs": outputs,
        "size_mb": round(engine_path.stat().st_size / 1e6, 1),
        "runtime_compatible": len(outputs) == 2,
    }
