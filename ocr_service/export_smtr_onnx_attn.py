"""
Export SVTRv2 + SMTRDecoder (GTCDecoder) to ONNX **with per-character attention maps**.

Outputs (4):
  - gtc_logits  [B, max_len-1, V_gtc]
  - ctc_logits  [B, T_ctc, V_ctc]
  - attn_maps   [B, max_len-1, H', W']   ← attention heatmap per decoding step
                H' = H_in / 8, W' = W_in / 4 (SVTRv2 strides)

Usage:
    python tools/export_smtr_onnx_attn.py \
        --c ./svtrv2_finetune_output_24_6/svtrv2_finetune_custom.yml \
        --o Global.pretrained_model=./svtrv2_finetune_output_24_6/best.pth \
           Global.export_dir=./export_attn \
           Global.fp16=True
"""

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import torch
import torch.nn as nn

from openrec.modeling import build_model as build_rec_model
from openrec.postprocess import build_post_process
from tools.engine.config import Config
from tools.utility import ArgsParser
from tools.utils.ckpt import load_ckpt
from tools.utils.logging import get_logger


class SVTRv2SMTRAttnONNXWrapper(nn.Module):
    """encoder + GTCDecoder.forward_onnx_with_attn."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feat = self.model.encoder(x)                      # [B, C, H', W']
        return self.model.decoder.forward_onnx_with_attn(feat)


def main(cfg):
    _cfg = cfg.cfg
    logger = get_logger()

    post_process_class = build_post_process(_cfg['PostProcess'])
    char_num = post_process_class.get_character_num()

    arch = _cfg['Architecture']
    if isinstance(arch['Decoder'].get('out_channels', None), list):
        arch['Decoder']['out_channels'] = [char_num, char_num]
    else:
        arch['Decoder']['out_channels'] = char_num

    model = build_rec_model(arch)
    load_ckpt(model, _cfg)
    model.eval()

    for layer in model.modules():
        if hasattr(layer, 'rep') and not getattr(layer, 'is_repped', True):
            layer.rep()

    wrapper = SVTRv2SMTRAttnONNXWrapper(model)
    wrapper.eval()

    try:
        scales = _cfg['Train']['dataset']['transforms'][-1].get('scales', [[320, 32]])
        rec_h = scales[0][1]
    except Exception:
        rec_h = 32
    dummy = torch.randn(1, 3, rec_h, 320)

    export_dir = _cfg['Global'].get('export_dir', './export_attn')
    os.makedirs(export_dir, exist_ok=True)
    save_path = os.path.join(export_dir, 'rec_smtr_attn.onnx')

    torch.onnx.export(
        wrapper,
        dummy,
        save_path,
        input_names=['image'],
        output_names=['gtc_logits', 'ctc_logits', 'attn_maps'],
        dynamic_axes={
            'image':      {0: 'batch', 3: 'width'},
            'gtc_logits': {0: 'batch'},
            'ctc_logits': {0: 'batch', 1: 'width'},
            'attn_maps':  {0: 'batch', 3: 'feat_w'},
        },
        opset_version=14,
    )
    logger.info(f'Exported ONNX model to: {save_path}')

    import onnx
    from onnx.external_data_helper import load_external_data_for_model
    logger.info('Merging external data into single ONNX file...')
    model_proto = onnx.load(save_path, load_external_data=True)
    load_external_data_for_model(model_proto, export_dir)
    onnx.save_model(model_proto, save_path, save_as_external_data=False)
    data_file = save_path + '.data'
    if os.path.exists(data_file):
        os.remove(data_file)
    logger.info(f'Single-file ONNX saved: {save_path}  '
                f'({os.path.getsize(save_path) / 1e6:.1f} MB)')

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(save_path, providers=['CPUExecutionProvider'])
        out = sess.run(None, {'image': dummy.numpy()})
        logger.info(
            f'gtc {out[0].shape}  ctc {out[1].shape}  attn {out[2].shape}'
        )
    except ImportError:
        logger.info('onnxruntime not installed, skipping verification.')

    if _cfg['Global'].get('fp16', False):
        try:
            from onnxconverter_common import float16
        except ImportError:
            logger.error(
                'onnxconverter_common not installed. '
                'Run: pip install onnxconverter-common'
            )
            return

        logger.info('Converting to FP16...')
        model_fp16 = float16.convert_float_to_float16(
            model_proto, keep_io_types=True
        )
        fp16_path = os.path.join(export_dir, 'rec_smtr_attn_fp16.onnx')
        onnx.save_model(model_fp16, fp16_path, save_as_external_data=False)
        logger.info(f'FP16 ONNX saved: {fp16_path}  '
                    f'({os.path.getsize(fp16_path) / 1e6:.1f} MB)')

        try:
            import onnxruntime as ort
            # FP16 conversion inserts "InsertedPrecisionFreeCast_*" nodes around
            # RMSNorm; ORT's SimplifiedLayerNormFusion crashes on them. Disable
            # just that fusion pass (falls back to BASIC level on older ORT).
            so16 = ort.SessionOptions()
            try:
                sess16 = ort.InferenceSession(
                    fp16_path, so16, providers=['CPUExecutionProvider'],
                    disabled_optimizers=['SimplifiedLayerNormFusion'],
                )
            except TypeError:
                so16.graph_optimization_level = \
                    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                sess16 = ort.InferenceSession(
                    fp16_path, so16, providers=['CPUExecutionProvider'])
            out16 = sess16.run(None, {'image': dummy.numpy()})
            logger.info(
                f'FP16 gtc {out16[0].shape}  ctc {out16[1].shape}  attn {out16[2].shape}'
            )
        except ImportError:
            pass


def parse_args():
    parser = ArgsParser()
    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg)


# python tools/export_smtr_onnx_attn.py --c ./svtrv2_finetune_output_24_6/svtrv2_finetune_custom.yml --o Global.pretrained_model=./svtrv2_finetune_output_24_6/best.pth Global.export_dir=./export_attn Global.fp16=True

