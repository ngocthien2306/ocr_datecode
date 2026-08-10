"""
Measure the GPU footprint of an OCR fine-tune run, per batch size.

Launches the real `tools/train_rec.py` CLI as its own process — the same way a
run is started for real — and samples nvidia-smi's per-process figure while it
runs. An earlier version of this script drove train_rec.py in-process via
runpy; it reported an identical 81 MiB for every batch size because train_rec.py
exits through SystemExit before the training loop when driven that way, so
nothing was ever measured. If two batch sizes ever report the same number
again, suspect the harness before believing the result.

nvidia-smi's per-process number (not torch's max_memory_allocated) is the one
to budget with: it includes the allocator's cached-but-free blocks and the CUDA
context, i.e. everything the OTHER processes on the card actually lose.

Usage:
    python measure_vram.py --c ./configs/general.yml --batches 16 32 64 128
"""
import argparse
import os
import subprocess
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
_OPENOCR = os.path.join(__dir__, 'OpenOCR')


def run_one(cfg, bs, epochs=1):
    cmd = [
        sys.executable, 'tools/train_rec.py', '--c', os.path.abspath(cfg), '--o',
        f'Global.epoch_num={epochs}',
        f'Train.loader.batch_size_per_card={bs}',
        'Global.save_epoch_step=[10000,10000]',
        'Global.output_dir=' + os.path.join(__dir__, 'output', '_vram_probe'),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    proc = subprocess.Popen(cmd, cwd=_OPENOCR, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    peak, samples = 0, 0
    while proc.poll() is None:
        try:
            r = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
            for line in r.stdout.strip().splitlines():
                p, mem = [x.strip() for x in line.split(',')]
                if int(p) == proc.pid:
                    peak = max(peak, int(mem))
                    samples += 1
        except Exception:
            pass
        # 0.1s, not 0.25s: at 640 images an epoch is ~2s of GPU work, so a
        # coarse poll can miss the training peak entirely and report only the
        # CUDA-context allocation from process startup. If `samples` comes back
        # in the single digits, the run was too short to have been measured --
        # raise --epochs rather than trusting the number.
        time.sleep(0.1)
    tail = (proc.stdout.read() or '')[-800:]
    ok = ('cur metric' in tail or 'best metric' in tail) and samples >= 20
    return peak, samples, ok, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c', dest='cfg', required=True)
    ap.add_argument('--batches', type=int, nargs='+', default=[16, 32, 64, 128])
    ap.add_argument('--epochs', type=int, default=1)
    args = ap.parse_args()

    print(f'{"batch":>6} {"peak VRAM":>12} {"samples":>8}   status')
    for bs in args.batches:
        peak, samples, ok, tail = run_one(args.cfg, bs, args.epochs)
        status = 'ok' if ok and peak > 0 else 'FAILED — see stderr'
        print(f'{bs:>6} {peak:>8,} MiB {samples:>8}   {status}')
        if not (ok and peak > 0):
            print(tail, file=sys.stderr)


if __name__ == '__main__':
    main()
