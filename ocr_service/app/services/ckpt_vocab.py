"""
Vocabulary surgery on SMTR/GTC checkpoints.

Turning on use_space_char adds ' ' to the character list, which changes the row
count of every vocab-sized tensor:

    decoder.gtc_decoder.char_embed.embedding.weight   (99, 384) -> (100, 384)
    decoder.gtc_decoder.ques1_head.{weight,bias}      (96, ...)  -> (97, ...)
    decoder.ctc_decoder.fc.{weight,bias}              (95, ...)  -> (96, ...)

torch's load_state_dict(strict=False) tolerates missing and extra KEYS but still
raises on a shape mismatch, so a space-enabled run cannot load a space-less
checkpoint as-is. Inserting one row keeps both trained heads — 21,022,849 of
21,024,003 parameters carry over untouched, 1,154 are new — where dropping the
tensors would throw away every character the model already knows.

The insert position is NOT the end. For the SMTR head ' ' sits between the last
real character and <s>, so appending would silently reassign
<s>/<INF>/<INB>/<pad> by one. Position comes from the post-processor's own
character list rather than a constant.
"""
import copy
import logging
from pathlib import Path
from typing import Dict

import torch

logger = logging.getLogger(__name__)

# tensor name -> which decoder's character list defines its row order
VOCAB_TENSORS = {
    "decoder.gtc_decoder.char_embed.embedding.weight": "gtc",
    "decoder.gtc_decoder.ques1_head.weight": "gtc",
    "decoder.gtc_decoder.ques1_head.bias": "gtc",
    "decoder.ctc_decoder.fc.weight": "ctc",
    "decoder.ctc_decoder.fc.bias": "ctc",
}

# Row count of ques1_head.weight for a checkpoint trained WITHOUT the space
# class. Used only to report vocab size; the expansion decision compares against
# the target model's real shapes, never against this constant.
_NO_SPACE_HEAD_ROWS = 96


def _state_dict(ckpt) -> Dict:
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt


def head_rows(ckpt_path: Path) -> int:
    """Rows in ques1_head.weight — 96 without the space class, 97 with it."""
    state = _state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    key = "decoder.gtc_decoder.ques1_head.weight"
    if key not in state:
        raise KeyError(f"{key} missing from {ckpt_path} — not an SVTRv2+SMTR checkpoint?")
    return int(state[key].shape[0])


def vocab_size(ckpt_path: Path) -> int:
    """Character-list length implied by the checkpoint: 99 or 100."""
    return head_rows(ckpt_path) + 3


def needs_expansion(ckpt_path: Path, use_space_char: bool) -> bool:
    """Whether this checkpoint must be widened before a run with this setting.

    Decided from the checkpoint's SHAPE, not from any recorded flag: fine-tuning
    onward from a model that already has the space class must not widen a second
    time, and a stored flag can be wrong or absent for hand-placed files.
    """
    if not use_space_char:
        return False
    return head_rows(ckpt_path) == _NO_SPACE_HEAD_ROWS


def _insert_row(t: torch.Tensor, idx: int, fill: str) -> torch.Tensor:
    """Insert one row at idx along dim 0.

    fill='mean' for embeddings — a row of zeros sits outside the distribution
    the attention layers were tuned on. fill='zero' for classifier heads, where
    a zero row means logit 0: a neutral starting score for the new class rather
    than an arbitrarily confident one.
    """
    new = t.mean(dim=0, keepdim=True) if fill == "mean" else torch.zeros_like(t[:1])
    return torch.cat([t[:idx], new, t[idx:]], dim=0)


def expand_for_space(ckpt_path: Path, out_path: Path, post_process) -> Dict:
    """Write a space-enabled copy of ckpt_path. `post_process` is a built
    GTCLabelDecode, used to locate ' ' in each head's character list."""
    space_idx = {
        "gtc": post_process.gtc_label_decode.character.index(" "),
        "ctc": post_process.ctc_label_decode.character.index(" "),
    }

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = copy.deepcopy(_state_dict(ckpt))

    changed = {}
    for name, which in VOCAB_TENSORS.items():
        if name not in state:
            raise KeyError(f"{name} not in {ckpt_path} — wrong architecture?")
        old_shape = tuple(state[name].shape)
        fill = "mean" if "embedding" in name else "zero"
        state[name] = _insert_row(state[name], space_idx[which], fill)
        changed[name] = f"{old_shape} -> {tuple(state[name].shape)} ({fill})"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt["state_dict"] = state
        torch.save(ckpt, str(out_path))
    else:
        torch.save(state, str(out_path))

    logger.info(f"[ocr] expanded vocab for space: {ckpt_path.name} -> {out_path.name}")
    for name, desc in changed.items():
        logger.info(f"[ocr]   {name} {desc}")
    return {"space_index": space_idx, "changed": changed, "out_path": str(out_path)}
