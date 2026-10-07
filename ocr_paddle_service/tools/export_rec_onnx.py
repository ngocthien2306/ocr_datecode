"""
Export a trained PaddleOCR rec checkpoint to inference model + ONNX.

    python tools/export_rec_onnx.py -c <run>/config.yml \
        -o Global.pretrained_model=<run>/best_accuracy \
           Global.save_inference_dir=<run>/inference \
        --onnx <run>/rec.onnx

Must run with cwd = PaddleOCR/ (configs carry relative dict paths).

Why the patches (Paddle 3.4 vs paddle2onnx 2.1.0, the latest):
  1. Paddle 3.4 hardcodes two new ops in PIR mode — F.linear -> linear_v2 and
     paddle.shape -> shape64 — which paddle2onnx does not know ("unsupported
     operators" abort). Both have exact old-op equivalents (matmul + add,
     shape), swapped in for the export trace only; training is untouched.
  2. Paddle 3.4 stores layer_norm epsilon as a_f64; paddle2onnx reads it as
     f32 and emits garbage (measured: 5.8e-36, -0.023, 6.9e+31 instead of
     1e-5/1e-6) — the ONNX then disagreed with Paddle on 36/40 test crops
     while converting "successfully". Converted from a copy of the graph with
     those attrs retagged a_f32.
The ONNX is checked against the Paddle predictor before this exits non-zero.
"""
import json
import os
import runpy
import sys

import numpy as np

import paddle
import paddle.nn.functional as F
import paddle.nn.functional.common as F_common
import paddle.tensor.attribute as T_attr
from paddle import _C_ops


def _linear_compat(x, weight, bias=None, name=None):
    out = _C_ops.matmul(x, weight, False, False)
    return _C_ops.add(out, bias) if bias is not None else out


def _shape_compat(input):
    out = _C_ops.shape(input)
    out.stop_gradient = True
    return out


def _patch():
    F.linear = F_common.linear = _linear_compat
    paddle.shape = T_attr.shape = _shape_compat


def _retag_f64_epsilon(src_json, dst_json):
    """Copy of the PIR graph with layer_norm epsilon as a_f32 (see docstring)."""
    graph = json.load(open(src_json))
    n = 0

    def walk(o):
        nonlocal n
        if isinstance(o, dict):
            if o.get("#", "").endswith("layer_norm"):
                for a in o.get("A", []):
                    if a.get("N") == "epsilon" and a["AT"].get("#") == "0.a_f64":
                        a["AT"]["#"] = "0.a_f32"
                        n += 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(graph)
    with open(dst_json, "w") as f:
        json.dump(graph, f)
    return n


def _check_parity(save_dir, onnx_path, width=320, tol=1e-3):
    import onnxruntime as ort
    import paddle.inference as pi

    cfg = pi.Config(os.path.join(save_dir, "inference.json"),
                    os.path.join(save_dir, "inference.pdiparams"))
    cfg.disable_gpu()
    cfg.disable_mkldnn()
    pred = pi.create_predictor(cfg)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for w in (width // 2, width):
        x = rng.uniform(-1, 1, (2, 3, 48, w)).astype(np.float32)
        h = pred.get_input_handle(pred.get_input_names()[0])
        h.copy_from_cpu(x)
        pred.run()
        ref = pred.get_output_handle(pred.get_output_names()[0]).copy_to_cpu()
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        worst = max(worst, float(np.abs(out - ref).max()))
    print(f"onnx vs paddle max|diff| = {worst:.2e} (tol {tol:g})")
    return worst <= tol


def main():
    argv = sys.argv[1:]
    onnx_path = None
    if "--onnx" in argv:
        i = argv.index("--onnx")
        onnx_path = argv[i + 1]
        del argv[i:i + 2]
    opset = 14

    _patch()
    paddle_ocr_dir = os.getcwd()
    sys.path.insert(0, paddle_ocr_dir)
    sys.argv = [os.path.join(paddle_ocr_dir, "tools", "export_model.py")] + argv
    runpy.run_path(sys.argv[0], run_name="__main__")

    if onnx_path:
        save_dir = next(
            a.split("=", 1)[1] for a in argv if a.startswith("Global.save_inference_dir=")
        )
        import paddle2onnx
        p2o_json = os.path.join(save_dir, "inference_p2o.json")
        n = _retag_f64_epsilon(os.path.join(save_dir, "inference.json"), p2o_json)
        print(f"retagged {n} layer_norm epsilon attrs f64 -> f32")
        paddle2onnx.export(
            p2o_json,
            os.path.join(save_dir, "inference.pdiparams"),
            onnx_path,
            opset_version=opset,
            enable_onnx_checker=True,
        )
        os.remove(p2o_json)
        print(f"onnx saved to {onnx_path}")
        if not _check_parity(save_dir, onnx_path):
            sys.exit("ONNX does not match Paddle — do not deploy this file")


if __name__ == "__main__":
    main()
