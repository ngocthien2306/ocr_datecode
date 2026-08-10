"""
CLI wrapper over app.services.ckpt_vocab — grow a SMTR/GTC checkpoint's
vocabulary by one class (the space character) so a checkpoint trained with
use_space_char=False can be fine-tuned with it on.

The service does this automatically when a run needs it
(ocr_training.prepare_checkpoint); this entry point exists for doing it by hand
against a base checkpoint. Read app/services/ckpt_vocab.py for why a row is
inserted rather than the tensors dropped, and why the insert position is not the
end.

Usage:
    python expand_ckpt_vocab.py \
        --c ./configs/svtrv2_finetune.yml \
        --in ./weights/base/svtrv2_datecode_2406.pth \
        --out ./output/base_space/best_space.pth
"""
import argparse
import os
import sys
from pathlib import Path

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)
sys.path.insert(0, os.path.join(__dir__, "OpenOCR"))

from app.services import ckpt_vocab  # noqa: E402
from app.services.ocr_training import build_post_process  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", dest="config", required=True,
                    help="the use_space_char=True config the output will train under")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    rows = ckpt_vocab.head_rows(in_path)
    if not ckpt_vocab.needs_expansion(in_path, use_space_char=True):
        print(f"{in_path} already has the space class "
              f"(ques1_head rows={rows}, vocab={ckpt_vocab.vocab_size(in_path)}) — nothing to do")
        return

    result = ckpt_vocab.expand_for_space(
        in_path, Path(args.out_path), build_post_process(Path(args.config)),
    )
    print(f"space index: {result['space_index']}")
    for name, desc in result["changed"].items():
        print(f"  {name:<55} {desc}")
    print(f"wrote {result['out_path']}")


if __name__ == "__main__":
    main()
