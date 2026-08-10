"""
Grow a SMTR/GTC checkpoint's vocabulary by one class (the space character),
so a model trained with use_space_char=False can be fine-tuned with
use_space_char=True.

Why this exists: turning on use_space_char adds ' ' to the character list,
which changes the output width of every vocab-sized tensor:

    decoder.gtc_decoder.char_embed.embedding.weight   (99, 384) -> (100, 384)
    decoder.gtc_decoder.ques1_head.{weight,bias}      (96, ...)  -> (97, ...)
    decoder.ctc_decoder.fc.{weight,bias}              (95, ...)  -> (96, ...)

torch's load_state_dict(strict=False) tolerates missing/extra KEYS but still
raises on a shape mismatch, so the checkpoint has to be reshaped up-front.
Dropping those tensors instead would throw away both trained heads; inserting
a row keeps every character the model already knows and leaves only the new
class to be learned.

Insert position is not the end: for the SMTR head ' ' sits between the last
real character and <s>, so appending would silently reassign <s>/<INF>/<INB>/
<pad>. The position is read from the post-processor's own character list
rather than hardcoded.

Usage:
    python expand_ckpt_vocab.py \
        --c ./configs/svtrv2_verify_space.yml \
        --in ./svtrv2_finetune_output_24_6/best.pth \
        --out ./output/base_space/best_space.pth
"""
import argparse
import copy
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(__dir__, 'OpenOCR'))

import torch

from openrec.postprocess import build_post_process
from tools.engine.config import Config

from export_smtr_onnx import _absolutize_dict_paths

# tensor name -> which decoder's character list defines its row order
VOCAB_TENSORS = {
    'decoder.gtc_decoder.char_embed.embedding.weight': 'gtc',
    'decoder.gtc_decoder.ques1_head.weight': 'gtc',
    'decoder.gtc_decoder.ques1_head.bias': 'gtc',
    'decoder.ctc_decoder.fc.weight': 'ctc',
    'decoder.ctc_decoder.fc.bias': 'ctc',
}


def _insert_row(t: torch.Tensor, idx: int, fill: str) -> torch.Tensor:
    """Insert one row at idx along dim 0.

    fill='mean' for embeddings — a brand-new embedding of zeros sits outside
    the distribution the attention layers were tuned on. fill='zero' for
    classifier heads — logit 0 is a neutral starting score for the new class
    rather than an arbitrarily confident one.
    """
    if fill == 'mean':
        new = t.mean(dim=0, keepdim=True)
    else:
        new = torch.zeros_like(t[:1])
    return torch.cat([t[:idx], new, t[idx:]], dim=0)


def expand(config_path, in_path, out_path):
    cfg = Config(config_path).cfg
    _absolutize_dict_paths(cfg)

    pp = build_post_process(cfg['PostProcess'])
    space_idx = {
        'gtc': pp.gtc_label_decode.character.index(' '),
        'ctc': pp.ctc_label_decode.character.index(' '),
    }
    print(f"space index: gtc={space_idx['gtc']}  ctc={space_idx['ctc']}")

    ckpt = torch.load(in_path, map_location='cpu')
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    state = copy.deepcopy(state)

    for name, which in VOCAB_TENSORS.items():
        if name not in state:
            raise KeyError(f'{name} not in checkpoint — wrong architecture?')
        old = state[name]
        fill = 'mean' if 'embedding' in name else 'zero'
        state[name] = _insert_row(old, space_idx[which], fill)
        print(f'  {name:<55} {tuple(old.shape)} -> {tuple(state[name].shape)} ({fill})')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if 'state_dict' in ckpt:
        ckpt['state_dict'] = state
        torch.save(ckpt, out_path)
    else:
        torch.save(state, out_path)
    print(f'wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c', dest='config', required=True,
                    help='the use_space_char=True config the output will train under')
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--out', dest='out_path', required=True)
    args = ap.parse_args()
    expand(args.config, args.in_path, args.out_path)


if __name__ == '__main__':
    main()
