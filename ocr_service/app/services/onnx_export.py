"""
Export a trained SVTRv2+SMTR checkpoint to ONNX with the TWO outputs the
runtime consumes: gtc_logits [B, max_len-1, V] and ctc_logits [B, W/4, V].

No attn_maps. ai_services' TextRecognizerSMTRTRT asserts the engine has exactly
two output bindings, so exporting two here means the TensorRT build needs no
output-stripping pass. (ocr_service/export_smtr_onnx_attn.py still emits the
three-output graph for the offline char-bbox tooling.)
"""
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from app.core.config import OPENOCR_DIR

logger = logging.getLogger(__name__)

# onnxconverter_common's fp16 rewrite wraps RMSNorm in
# "InsertedPrecisionFreeCast_*" nodes that onnxruntime's SimplifiedLayerNormFusion
# then fails to resolve, so the session refuses to initialise. Disabling just
# that pass keeps every other graph optimisation.
DISABLED_ORT_OPTIMIZERS = ["SimplifiedLayerNormFusion"]


def ensure_openocr_on_path() -> None:
    if str(OPENOCR_DIR) not in sys.path:
        sys.path.insert(0, str(OPENOCR_DIR))


def absolutize_dict_paths(node) -> None:
    """Rewrite every relative character_dict_path against OpenOCR/.

    Training configs carry './tools/utils/EN_symbol_dict.txt', which only
    resolves when the process cwd is the OpenOCR checkout — that is how
    train_rec.py is launched. Export runs from anywhere.
    """
    if isinstance(node, dict):
        p = node.get("character_dict_path")
        if isinstance(p, str) and p and not os.path.isabs(p):
            node["character_dict_path"] = os.path.normpath(os.path.join(str(OPENOCR_DIR), p))
        for v in node.values():
            absolutize_dict_paths(v)
    elif isinstance(node, list):
        for v in node:
            absolutize_dict_paths(v)


def make_ort_session(path: str, providers=None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    providers = providers or ["CPUExecutionProvider"]
    try:
        return ort.InferenceSession(
            path, so, providers=providers, disabled_optimizers=DISABLED_ORT_OPTIMIZERS,
        )
    except TypeError:
        # onnxruntime predating `disabled_optimizers`: BASIC level skips the
        # extended-level fusion that breaks.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        return ort.InferenceSession(path, so, providers=providers)


def export_onnx(
    config_path: Path,
    ckpt_path: Path,
    out_dir: Path,
    fp16: bool = True,
    height: int = 32,
    width: int = 320,
    opset: int = 14,
) -> Dict[str, Optional[str]]:
    """Trace the checkpoint to ONNX (fp32, plus fp16 when asked).

    `width` is only the tracing width — the width axis stays dynamic, so the
    engine built from this graph accepts any crop width inside its profile.
    """
    ensure_openocr_on_path()
    import torch
    import torch.nn as nn
    from openrec.modeling import build_model as build_rec_model
    from openrec.postprocess import build_post_process
    from tools.engine.config import Config
    from tools.utils.ckpt import load_ckpt

    cfg = Config(str(config_path)).cfg
    cfg["Global"]["pretrained_model"] = str(ckpt_path)
    absolutize_dict_paths(cfg)

    post_process = build_post_process(cfg["PostProcess"])
    char_num = post_process.get_character_num()
    arch = cfg["Architecture"]
    if isinstance(arch["Decoder"].get("out_channels", None), list):
        arch["Decoder"]["out_channels"] = [char_num, char_num]
    else:
        arch["Decoder"]["out_channels"] = char_num

    model = build_rec_model(arch)
    load_ckpt(model, cfg)
    model.eval()

    # Re-parameterisable blocks (the SVTRv2 conv mixers) must be folded before
    # tracing or the graph keeps the training-time multi-branch form.
    for layer in model.modules():
        if hasattr(layer, "rep") and not getattr(layer, "is_repped", True):
            layer.rep()

    class _Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, x):
            return self.model.decoder.forward_onnx(self.model.encoder(x))

    wrapper = _Wrapper(model).eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "rec_smtr.onnx"
    dummy = torch.randn(1, 3, height, width)

    torch.onnx.export(
        wrapper,
        dummy,
        str(fp32_path),
        input_names=["image"],
        output_names=["gtc_logits", "ctc_logits"],
        dynamic_axes={
            "image": {0: "batch", 3: "width"},
            "gtc_logits": {0: "batch"},
            "ctc_logits": {0: "batch", 1: "width"},
        },
        opset_version=opset,
        # TorchScript tracer, not the dynamo exporter torch>=2.9 defaults to.
        # The decode loop is a Python for-loop that has to be unrolled into the
        # graph, which is exactly what tracing does.
        dynamo=False,
    )

    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    # torch can write weights as a sidecar .data file; the runtime loads a
    # single self-contained .onnx, so fold it back in and drop the sidecar.
    proto = onnx.load(str(fp32_path), load_external_data=True)
    load_external_data_for_model(proto, str(out_dir))
    onnx.save_model(proto, str(fp32_path), save_as_external_data=False)
    Path(str(fp32_path) + ".data").unlink(missing_ok=True)

    sess = make_ort_session(str(fp32_path))
    out = sess.run(None, {"image": dummy.numpy()})
    logger.info(f"[ocr] onnx fp32 ok: gtc {out[0].shape} ctc {out[1].shape} "
                f"({fp32_path.stat().st_size / 1e6:.1f} MB)")

    result: Dict[str, Optional[str]] = {
        "onnx_path": str(fp32_path),
        "onnx_fp16_path": None,
        "gtc_shape": list(out[0].shape),
        "ctc_shape": list(out[1].shape),
    }

    if fp16:
        from onnxconverter_common import float16

        fp16_model = float16.convert_float_to_float16(proto, keep_io_types=True)
        fp16_path = out_dir / "rec_smtr_fp16.onnx"
        onnx.save_model(fp16_model, str(fp16_path), save_as_external_data=False)
        sess16 = make_ort_session(str(fp16_path))
        out16 = sess16.run(None, {"image": dummy.numpy()})
        logger.info(f"[ocr] onnx fp16 ok: gtc {out16[0].shape} ctc {out16[1].shape} "
                    f"({fp16_path.stat().st_size / 1e6:.1f} MB)")
        result["onnx_fp16_path"] = str(fp16_path)

    return result
