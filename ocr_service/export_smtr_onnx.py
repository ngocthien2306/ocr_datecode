"""
Export SVTRv2 + GTCDecoder(SMTRDecoder + RCTCDecoder) to ONNX with the TWO
outputs the runtime actually consumes:

  - gtc_logits  [B, max_len-1, V_gtc]     (27 x 96 for max_text_length=25)
  - ctc_logits  [B, W/4, V_ctc]           (95 classes)

No attn_maps. The older export_smtr_onnx_attn.py emitted a third
per-character attention output used only by offline char-bbox tooling;
ai_services/camera_management/ocr/backends/smtr_trt.py asserts the engine has
exactly two output bindings, so exporting two here means the TensorRT build
step needs no output-stripping pass.

Usage:
    python export_smtr_onnx.py \
        --c ./configs/svtrv2_verify.yml \
        --ckpt ./output/verify_run/best.pth \
        --out ./output/verify_run/export \
        --fp16
"""

import argparse
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(__dir__, 'OpenOCR'))

import torch
import torch.nn as nn

from openrec.modeling import build_model as build_rec_model
from openrec.postprocess import build_post_process
from tools.engine.config import Config
from tools.utils.ckpt import load_ckpt
from tools.utils.logging import get_logger

logger = get_logger()

# onnxconverter_common's fp16 graph rewrite inserts casts around RMSNorm that
# onnxruntime's SimplifiedLayerNormFusion pass then trips over
# ("Attempting to get index by a name which does not exist"). Disabling just
# that pass keeps every other graph optimization. Mirrors
# infer_rec_onnx_attn.py::make_session.
_DISABLED_OPTIMIZERS = ['SimplifiedLayerNormFusion']


_OPENOCR_DIR = os.path.join(__dir__, 'OpenOCR')


def _absolutize_dict_paths(node):
    """Rewrite every relative character_dict_path against OpenOCR/.

    Training configs carry './tools/utils/EN_symbol_dict.txt', which only
    resolves when the process cwd is the OpenOCR checkout (that's how
    train_rec.py is launched). Export runs from anywhere, so resolve it here
    instead of forcing callers to chdir.
    """
    if isinstance(node, dict):
        p = node.get('character_dict_path')
        if isinstance(p, str) and p and not os.path.isabs(p):
            node['character_dict_path'] = os.path.normpath(
                os.path.join(_OPENOCR_DIR, p))
        for v in node.values():
            _absolutize_dict_paths(v)
    elif isinstance(node, list):
        for v in node:
            _absolutize_dict_paths(v)


class SMTRONNXWrapper(nn.Module):
    """encoder + GTCDecoder.forward_onnx (gtc_logits, ctc_logits)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feat = self.model.encoder(x)          # [B, C, H/8, W/4]
        return self.model.decoder.forward_onnx(feat)


def _make_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    try:
        return ort.InferenceSession(
            path, so, providers=['CPUExecutionProvider'],
            disabled_optimizers=_DISABLED_OPTIMIZERS,
        )
    except TypeError:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])


def export(config_path, ckpt_path, out_dir, fp16=True, height=32, width=320,
           opset=14):
    cfg = Config(config_path).cfg
    cfg['Global']['pretrained_model'] = ckpt_path
    _absolutize_dict_paths(cfg)

    post_process_class = build_post_process(cfg['PostProcess'])
    char_num = post_process_class.get_character_num()

    arch = cfg['Architecture']
    if isinstance(arch['Decoder'].get('out_channels', None), list):
        arch['Decoder']['out_channels'] = [char_num, char_num]
    else:
        arch['Decoder']['out_channels'] = char_num

    model = build_rec_model(arch)
    load_ckpt(model, cfg)
    model.eval()

    # Re-parameterizable blocks (SVTRv2 conv mixers) must be folded before
    # tracing, or the graph carries the training-time multi-branch form.
    for layer in model.modules():
        if hasattr(layer, 'rep') and not getattr(layer, 'is_repped', True):
            layer.rep()

    wrapper = SMTRONNXWrapper(model).eval()

    os.makedirs(out_dir, exist_ok=True)
    fp32_path = os.path.join(out_dir, 'rec_smtr.onnx')
    dummy = torch.randn(1, 3, height, width)

    torch.onnx.export(
        wrapper,
        dummy,
        fp32_path,
        input_names=['image'],
        output_names=['gtc_logits', 'ctc_logits'],
        dynamic_axes={
            'image':      {0: 'batch', 3: 'width'},
            'gtc_logits': {0: 'batch'},
            'ctc_logits': {0: 'batch', 1: 'width'},
        },
        opset_version=opset,
        # TorchScript tracer, not the dynamo exporter torch>=2.9 defaults to.
        # The decode loop is a Python for-loop meant to be unrolled into the
        # graph; that is exactly what tracing does, and it is how the
        # in-production rec_smtr_attn_fp16.onnx was produced.
        dynamo=False,
    )
    logger.info(f'Exported ONNX: {fp32_path}')

    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    # torch writes weights >2GB (and sometimes smaller graphs) as a sidecar
    # .data file; the runtime loads a single self-contained .onnx, so fold it
    # back in and drop the sidecar.
    model_proto = onnx.load(fp32_path, load_external_data=True)
    load_external_data_for_model(model_proto, out_dir)
    onnx.save_model(model_proto, fp32_path, save_as_external_data=False)
    if os.path.exists(fp32_path + '.data'):
        os.remove(fp32_path + '.data')
    logger.info(f'Single-file ONNX: {fp32_path} '
                f'({os.path.getsize(fp32_path) / 1e6:.1f} MB)')

    sess = _make_session(fp32_path)
    out = sess.run(None, {'image': dummy.numpy()})
    logger.info(f'fp32 verify: gtc {out[0].shape}  ctc {out[1].shape}')

    result = {'onnx_path': fp32_path, 'onnx_fp16_path': None}

    if fp16:
        from onnxconverter_common import float16
        model_fp16 = float16.convert_float_to_float16(
            model_proto, keep_io_types=True)
        fp16_path = os.path.join(out_dir, 'rec_smtr_fp16.onnx')
        onnx.save_model(model_fp16, fp16_path, save_as_external_data=False)
        logger.info(f'FP16 ONNX: {fp16_path} '
                    f'({os.path.getsize(fp16_path) / 1e6:.1f} MB)')

        sess16 = _make_session(fp16_path)
        out16 = sess16.run(None, {'image': dummy.numpy()})
        logger.info(f'fp16 verify: gtc {out16[0].shape}  ctc {out16[1].shape}')
        result['onnx_fp16_path'] = fp16_path

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c', dest='config', required=True, help='training config yml')
    ap.add_argument('--ckpt', required=True, help='checkpoint (best.pth)')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--fp16', action='store_true', default=True)
    ap.add_argument('--no-fp16', dest='fp16', action='store_false')
    ap.add_argument('--height', type=int, default=32)
    ap.add_argument('--width', type=int, default=320,
                    help='dummy trace width; the width axis stays dynamic')
    ap.add_argument('--opset', type=int, default=14)
    args = ap.parse_args()

    export(args.config, args.ckpt, args.out, fp16=args.fp16,
           height=args.height, width=args.width, opset=args.opset)


if __name__ == '__main__':
    main()
