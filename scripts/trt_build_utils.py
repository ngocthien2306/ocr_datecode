"""
Shared TensorRT engine-build helpers using the TensorRT Python Builder API
directly (no trtexec binary needed -- works with the pip-installed
tensorrt==11.1.0 in the `vision` conda env, which does not ship trtexec).

Used by:
    scripts/build_ocr_engines.py
    scripts/build_pipeline_engines.py
"""
import os
import re
import sys
import time
import warnings

import onnx
import tensorrt as trt
from onnxconverter_common import float16 as onnx_float16

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

_INVALID_NODE_RE = re.compile(r"Invalid Node - (\S+)")

# onnxconverter_common's post-conversion "remove redundant Cast pairs" cleanup
# crashes (AttributeError: 'list' object has no attribute 'input') whenever a
# Cast node's output fans out to more than one downstream node -- common in
# transformer graphs (e.g. SuperPoint/LightGlue) with node_block_list set.
# It's a pure optimization pass (leaves a few harmless extra Cast pairs if
# skipped, TensorRT folds those away anyway), so disable it rather than work
# around the crash.
onnx_float16.remove_unnecessary_cast_node = lambda graph_proto: None


def parse_shape(s: str):
    return tuple(int(x) for x in s.lower().split("x"))


def load_onnx_bytes(onnx_path, drop_outputs=None, fp16=False, node_block_list=None):
    """
    Return serialized onnx bytes, optionally with some graph outputs removed
    and/or weights/activations converted to float16.

    TensorRT 11.1.0 dropped BuilderFlag.FP16 (weakly-typed builder precision
    control no longer exists in this pip build) -- precision is now driven
    entirely by the tensor dtypes baked into the ONNX graph itself. So "fp16
    build" here means converting the graph to float16 (keeping input/output
    tensors float32 via keep_io_types=True, matching the old FP16-flag
    behavior of fp32 I/O with fp16 internal compute) before handing it to
    the TensorRT parser.

    node_block_list: node names to force-keep in float32 (with Cast nodes
    auto-inserted at their boundary by onnxconverter_common). Needed for
    graphs with explicit Cast(to=float32) nodes feeding an elementwise op
    against a Constant that the converter would otherwise downcast to
    float16, producing a dtype-mismatched op (e.g. SuperPoint/LightGlue's
    positional-encoding Muls).
    """
    model = onnx.load(onnx_path)

    if drop_outputs:
        keep = [o for o in model.graph.output if o.name not in drop_outputs]
        dropped = [o.name for o in model.graph.output if o.name in drop_outputs]
        del model.graph.output[:]
        model.graph.output.extend(keep)
        print(f"  dropped outputs: {dropped} -> remaining: {[o.name for o in keep]}")

    if fp16:
        already_fp16 = any(
            init.data_type == onnx.TensorProto.FLOAT16 for init in model.graph.initializer
        )
        if already_fp16:
            print("  graph weights already float16, skipping conversion")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = onnx_float16.convert_float_to_float16(
                    model, keep_io_types=True, node_block_list=list(node_block_list or [])
                )
            print("  converted graph weights/activations to float16 (I/O stays float32)"
                  + (f", kept fp32 at: {sorted(node_block_list)}" if node_block_list else ""))

    return model.SerializeToString()


def _parse_network(onnx_path, drop_outputs, fp16, node_block_list, builder):
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    onnx_bytes = load_onnx_bytes(onnx_path, drop_outputs, fp16=fp16, node_block_list=node_block_list)
    ok = parser.parse(onnx_bytes)
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    return ok, network, errors


def seed_block_list(onnx_path, patterns):
    """
    Node names in onnx_path matching any of `patterns` (regexes, fullmatch).

    The retry loop below only learns one bad node per parse attempt, and each
    attempt re-loads + re-converts the whole graph -- fine for a handful of
    nodes, far too slow for a deep transformer where the same dtype mismatch
    repeats in every layer (the 9-layer LightGlue needs ~58 nodes forced to
    fp32). Pre-seeding the known families collapses that to a single parse.
    """
    model = onnx.load(onnx_path, load_external_data=False)
    res = [re.compile(p) for p in patterns]
    return {n.name for n in model.graph.node if n.name and any(r.fullmatch(n.name) for r in res)}


def build_engine(onnx_path, engine_path, input_name, min_shape, opt_shape, max_shape,
                  fp16=True, workspace_gib=4, drop_outputs=None, max_fp32_retries=25,
                  fp32_node_patterns=None):
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(onnx_path)

    # TensorRT 10+ removed the EXPLICIT_BATCH flag -- explicit batch is now
    # the only supported network mode, so create_network() takes no flags.
    node_block_list = set()
    if fp16 and fp32_node_patterns:
        node_block_list = seed_block_list(onnx_path, fp32_node_patterns)
        print(f"  pre-seeded {len(node_block_list)} nodes as fp32 from {len(fp32_node_patterns)} pattern(s)")
    network = None
    for attempt in range(max_fp32_retries + 1):
        builder = trt.Builder(TRT_LOGGER)
        ok, network, errors = _parse_network(onnx_path, drop_outputs, fp16, node_block_list, builder)
        if ok:
            break

        newly_bad = {m for e in errors for m in _INVALID_NODE_RE.findall(e)}
        newly_bad -= node_block_list
        if not fp16 or not newly_bad:
            for e in errors:
                print(e, file=sys.stderr)
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

        print(f"  parse failed on dtype mismatch, forcing fp32 for: {sorted(newly_bad)} (retry {attempt + 1})")
        node_block_list |= newly_bad
    else:
        raise RuntimeError(f"Failed to parse ONNX after {max_fp32_retries} fp32-fallback retries: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib * (1 << 30))

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print(f"  building: {onnx_path}")
    print(f"    input='{input_name}' min={min_shape} opt={opt_shape} max={max_shape} fp16={fp16}")
    t0 = time.time()
    host_mem = builder.build_serialized_network(network, config)
    if host_mem is None:
        raise RuntimeError(f"Engine build failed for {onnx_path}")
    # TensorRT 10+ returns an IHostMemory wrapper (no __len__), not raw bytes.
    serialized = bytes(memoryview(host_mem))

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    dt = time.time() - t0
    size_mb = len(serialized) / (1024 * 1024)
    print(f"  ✅ wrote {engine_path} ({size_mb:.1f} MB, {dt:.1f}s)")


def run_spec(name, spec, base_dir):
    print(f"\n=== {name} ===")
    if spec.get("note"):
        print(f"  ⚠️  {spec['note']}")
    onnx_path = os.path.join(base_dir, spec["onnx"])
    engine_path = os.path.join(base_dir, spec["engine"])
    if os.path.exists(engine_path):
        backup = engine_path + ".bak"
        os.replace(engine_path, backup)
        print(f"  existing engine backed up -> {backup}")
    build_engine(
        onnx_path, engine_path, spec["input_name"],
        spec["min"], spec["opt"], spec["max"], fp16=spec.get("fp16", True),
        drop_outputs=spec.get("drop_outputs"),
        fp32_node_patterns=spec.get("fp32_node_patterns"),
    )
