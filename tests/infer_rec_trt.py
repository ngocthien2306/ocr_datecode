"""
SMTR (GTC + CTC dual-head) — Pure TensorRT backend (pycuda, TensorRT 10.x)

Build engine once:
    /usr/src/tensorrt/bin/trtexec \
        --onnx=languages/english/rec_smtr_fp16.onnx \
        --saveEngine=languages/english/rec_smtr_fp16.engine \
        --fp16 \
        --minShapes=image:1x3x32x32 \
        --optShapes=image:4x3x32x320 \
        --maxShapes=image:16x3x32x1280 \
        --memPoolSize=workspace:4096

Usage:
    python infer_rec_trt.py --img <file_or_dir> [--batch N] [--verbose]
"""

import argparse, os, time
import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

# reuse decode + preprocess từ ONNX module
from infer_rec_onnx import (
    build_postprocessors,
    preprocess_batch,
    get_image_paths,
    SMTRLabelDecode,
    CTCLabelDecode,
)

ENGINE_PATH = './languages/english/rec_smtr_fp16.engine'
DICT_PATH   = './languages/english/EN_symbol_dict.txt'


# ── Engine loader ──────────────────────────────────────────────────────────────

def load_engine(engine_path: str):
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        return trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())


# ── SMTRTRTSession ──────────────────────────────────────────────────────────────

class SMTRTRTSession:
    """
    Wraps a TRT engine for the SMTR model (1 input, 2 outputs, dynamic batch+width).

    Pre-allocates CUDA buffers at max profile size.
    On each call, sets actual input shape → TRT computes actual output shape →
    copies only the needed bytes back.
    """

    def __init__(self, engine_path: str):
        self.engine  = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        # Resolve tensor names in order: 1 input, 2 outputs
        self.input_name   = None
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        assert self.input_name, "No input tensor found"
        assert len(self.output_names) == 2, (
            f"Expected 2 output tensors, got {self.output_names}")

        # Allocate input buffer at max profile shape
        in_min, in_opt, in_max = self.engine.get_tensor_profile_shape(self.input_name, 0)
        self._in_max   = tuple(in_max)
        self._max_batch = in_max[0]
        self._d_input   = cuda.mem_alloc(int(np.prod(in_max)) * 4)

        # Allocate output buffers: set input to max so TRT can resolve max output sizes
        self.context.set_input_shape(self.input_name, in_max)
        self._d_outputs  = []
        self._out_max_elems = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            elems = int(np.prod(shape))
            self._d_outputs.append(cuda.mem_alloc(elems * 4))
            self._out_max_elems.append(elems)

        # Bind all tensor addresses
        self.context.set_tensor_address(self.input_name, int(self._d_input))
        for name, d_buf in zip(self.output_names, self._d_outputs):
            self.context.set_tensor_address(name, int(d_buf))

        # Warm-up
        self._infer_raw(np.zeros(in_max, dtype=np.float32))
        print(f"[TRT] Engine loaded: {os.path.basename(engine_path)} | "
              f"max_batch={self._max_batch} | outputs={self.output_names}")

    def _infer_raw(self, x: np.ndarray):
        """x: float32 [N, 3, H, W] — returns (gtc_logits, ctc_logits)."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        self.context.set_input_shape(self.input_name, x.shape)

        # Actual output shapes for this input
        out_shapes = [
            tuple(self.context.get_tensor_shape(n)) for n in self.output_names
        ]

        cuda.memcpy_htod_async(self._d_input, x, self.stream)
        self.context.execute_async_v3(self.stream.handle)

        outputs = []
        for d_buf, shape in zip(self._d_outputs, out_shapes):
            arr = np.empty(shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(arr, d_buf, self.stream)
            outputs.append(arr)

        self.stream.synchronize()
        return outputs[0], outputs[1]   # gtc_logits, ctc_logits

    def __del__(self):
        try:
            self._d_input.free()
            for d in self._d_outputs:
                d.free()
        except Exception:
            pass


# ── per-char confidence ────────────────────────────────────────────────────────

def run_with_char_conf(img_bgr: np.ndarray, trt_sess: SMTRTRTSession,
                       gtc_decode: SMTRLabelDecode, ctc_decode: CTCLabelDecode):
    """Single image → prints per-char conf table for both heads, returns (text, score)."""
    from infer_rec_onnx import preprocess, _print_char_conf

    inp = preprocess(img_bgr)
    gtc_logits, ctc_logits = trt_sess._infer_raw(inp)

    gtc_cc = gtc_decode.decode_char_confs(gtc_logits)[0]
    ctc_cc = ctc_decode.decode_char_confs(ctc_logits)[0]

    gtc_results = gtc_decode(gtc_logits, torch_tensor=False)
    ctc_results = ctc_decode(ctc_logits, torch_tensor=False)
    gtc_text, gtc_score = gtc_results[0]
    ctc_text, ctc_score = ctc_results[0]
    text, score = (ctc_text, ctc_score) if ctc_score >= gtc_score else (gtc_text, gtc_score)

    _print_char_conf('GTC', gtc_cc)
    _print_char_conf('CTC', ctc_cc)
    print(f"  winner='{text}'  score={score:.4f}")
    return text, score


# ── batch inference ────────────────────────────────────────────────────────────

def run_batch(imgs_bgr: list, trt_sess: SMTRTRTSession,
              gtc_decode: SMTRLabelDecode, ctc_decode: CTCLabelDecode,
              verbose: bool = False):
    if not imgs_bgr:
        return []

    t0 = time.perf_counter()
    inp = preprocess_batch(imgs_bgr)           # [N, 3, 32, max_W]
    t_pre = time.perf_counter() - t0

    t1 = time.perf_counter()
    gtc_logits, ctc_logits = trt_sess._infer_raw(inp)
    t_infer = time.perf_counter() - t1

    t2 = time.perf_counter()
    gtc_results = gtc_decode(gtc_logits, torch_tensor=False)
    ctc_results = ctc_decode(ctc_logits, torch_tensor=False)
    t_post = time.perf_counter() - t2

    results = []
    for gtc, ctc in zip(gtc_results, ctc_results):
        gtc_text, gtc_score = gtc
        ctc_text, ctc_score = ctc
        text, score = (ctc_text, ctc_score) if ctc_score >= gtc_score else (gtc_text, gtc_score)
        results.append((text, score))

    if verbose:
        n = len(imgs_bgr)
        total = t_pre + t_infer + t_post
        print(f'  batch_size : {n}')
        print(f'  preprocess : {t_pre*1000:6.2f} ms')
        print(f'  inference  : {t_infer*1000:6.2f} ms')
        print(f'  postprocess: {t_post*1000:6.2f} ms')
        print(f'  total      : {total*1000:6.2f} ms  ({total/n*1000:.2f} ms/img)')

    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',     required=True)
    parser.add_argument('--engine',  default=ENGINE_PATH)
    parser.add_argument('--dict',    default=DICT_PATH)
    parser.add_argument('--batch',     type=int, default=0,
                        help='chunk size (0 = all at once)')
    parser.add_argument('--char-conf', action='store_true',
                        help='print per-character confidence for both GTC and CTC heads')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    trt_sess = SMTRTRTSession(args.engine)
    gtc_decode, ctc_decode = build_postprocessors(args.dict)

    img_paths = get_image_paths(args.img)

    if args.char_conf:
        for path in img_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            print(f'\n{"="*55}')
            print(f'  {os.path.basename(path)}')
            run_with_char_conf(img, trt_sess, gtc_decode, ctc_decode)
        return

    chunk = args.batch if args.batch > 0 else len(img_paths)
    for start in range(0, len(img_paths), chunk):
        paths_chunk = img_paths[start:start + chunk]
        imgs  = [cv2.imread(p) for p in paths_chunk]
        valid = [(p, img) for p, img in zip(paths_chunk, imgs) if img is not None]
        if not valid:
            continue
        paths_ok, imgs_ok = zip(*valid)
        results = run_batch(list(imgs_ok), trt_sess, gtc_decode, ctc_decode,
                            verbose=args.verbose)
        for path, (text, score) in zip(paths_ok, results):
            print(f'{path}\t{text}\t{score:.4f}')


if __name__ == '__main__':
    main()
