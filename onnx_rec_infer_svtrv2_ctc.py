import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Backend abstraction: ONNX Runtime  |  ORT-TensorRT EP  |  pure TensorRT
# ---------------------------------------------------------------------------

def _build_onnx_session(model_path: str, device: str):
    """
    device: 'cpu' | 'cuda' | 'trt'
      - cpu   → CPUExecutionProvider
      - cuda  → CUDAExecutionProvider (cuDNN exhaustive search + all graph opts)
      - trt   → TensorrtExecutionProvider (fp16, engine cache) + CUDA fallback
    """
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if device == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device == "cuda":
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider",
        ]
    elif device == "trt":
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": False,
                "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,  # 4 GB
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache",
                "trt_min_subgraph_size": 3,
            }),
            "CUDAExecutionProvider",   # fallback for unsupported ops
            "CPUExecutionProvider",
        ]
    else:
        raise ValueError(f"Unknown device: {device}. Choose cpu / cuda / trt")

    sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
    active = sess.get_providers()
    print(f"[backend] {active[0]}")
    return sess


class ONNXBackend:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.sess = _build_onnx_session(model_path, device)
        self.in_name  = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self._use_iobinding = (device in ("cuda", "trt"))

    def infer(self, x: np.ndarray) -> np.ndarray:
        if not self._use_iobinding:
            return self.sess.run([self.out_name], {self.in_name: x})[0]
        # IO binding: pin input on CUDA, keep output on CUDA → one D2H copy at end
        io = self.sess.io_binding()
        x = np.ascontiguousarray(x, dtype=np.float32)
        x_ort = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
        io.bind_ortvalue_input(self.in_name, x_ort)
        io.bind_output(self.out_name, device_type="cuda", device_id=0,
                       element_type=np.float32)
        self.sess.run_with_iobinding(io)
        return io.get_outputs()[0].numpy()


def _try_import_trt():
    try:
        import tensorrt as trt      # noqa: F401
        import pycuda.driver as cuda  # noqa: F401
        import pycuda.autoinit       # noqa: F401
        return True
    except ImportError:
        return False


class TRTBackend:
    """Pure TensorRT 10.x backend (requires tensorrt + pycuda)."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        self._trt  = trt
        self._cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        # Resolve tensor names
        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        assert self.input_name and self.output_name

        # Pre-allocate device buffers at max profile shape
        in_max = list(self.engine.get_tensor_profile_shape(self.input_name, 0)[2])
        # Output shape is inferred — set max input first, then query context
        self.context.set_input_shape(self.input_name, in_max)
        out_max = list(self.context.get_tensor_shape(self.output_name))
        self._d_input  = cuda.mem_alloc(int(np.prod(in_max))  * 4)
        self._d_output = cuda.mem_alloc(int(np.prod(out_max)) * 4)
        self.context.set_tensor_address(self.input_name,  int(self._d_input))
        self.context.set_tensor_address(self.output_name, int(self._d_output))

        # Warm-up
        dummy = np.zeros(in_max, dtype=np.float32)
        self.infer(dummy)
        print(f"[backend] TensorrtExecutionProvider (pure TRT)")

    def infer(self, x: np.ndarray) -> np.ndarray:
        cuda = self._cuda
        x = np.ascontiguousarray(x, dtype=np.float32)
        self.context.set_input_shape(self.input_name, x.shape)
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))

        cuda.memcpy_htod_async(self._d_input, x, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        output = np.empty(out_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self._d_output, self.stream)
        self.stream.synchronize()
        return output


def create_backend(args):
    """
    Priority:
      --engine  → pure TensorRT (needs tensorrt + pycuda)
      --device trt/cuda/cpu  → ORT with corresponding EP
    """
    if args.engine:
        if not _try_import_trt():
            raise RuntimeError(
                "tensorrt / pycuda not found. Install them or use --onnx with --device trt")
        return TRTBackend(args.engine)
    return ONNXBackend(args.onnx, device=args.device)


# ---------------------------------------------------------------------------
# OCR confusion map + verify
# ---------------------------------------------------------------------------




def normalize_text(s: str, keep_space: bool = True) -> str:
    s = s.strip()
    return " ".join(s.split()) if keep_space else s.replace(" ", "")


def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


# ---------------------------------------------------------------------------
# Pre / post processing
# ---------------------------------------------------------------------------

def load_charset(dict_path, use_space_char=True):
    chars = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            ch = line.rstrip("\n")
            if ch == "":
                continue
            chars.append(ch)
    if use_space_char and " " not in chars:
        chars.append(" ")
    return [""] + chars  # 0 is CTC blank


def resize_keep_ratio_pad(img, imgH=48, imgW=320):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image shape: {img.shape}")
    ratio = w / float(h)
    new_w = min(int(np.ceil(imgH * ratio)), imgW)
    resized = cv2.resize(img, (new_w, imgH), interpolation=cv2.INTER_CUBIC)
    if new_w < imgW:
        pad = np.zeros((imgH, imgW - new_w, 3), dtype=resized.dtype)
        resized = np.concatenate([resized, pad], axis=1)
    else:
        resized = resized[:, :imgW, :]
    return resized


def preprocess(img_bgr, imgH=48, imgW=320, img_mode="RGB"):
    if img_mode.upper() == "RGB":
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img = img_bgr
    img = resize_keep_ratio_pad(img, imgH=imgH, imgW=imgW)
    img = img.astype("float32") / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0).astype("float32")


def preprocess_batch(img_paths, imgH=48, imgW=320, img_mode="RGB"):
    imgs = []
    for p in img_paths:
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            raise FileNotFoundError(p)
        imgs.append(preprocess(img_bgr, imgH=imgH, imgW=imgW, img_mode=img_mode))
    return np.concatenate(imgs, axis=0)


def ctc_greedy_decode(logits, charset):
    if logits.ndim != 3:
        raise ValueError(f"Unexpected output shape: {logits.shape}")
    _, d1, _ = logits.shape
    if d1 == len(charset):
        logits = np.transpose(logits, (0, 2, 1))

    pred = np.argmax(logits, axis=2)
    texts, confs, char_confs = [], [], []
    blank_id = 0
    for b in range(pred.shape[0]):
        out_chars, out_confs, last = [], [], None
        for t, p in enumerate(pred[b].tolist()):
            if p != blank_id and p != last and 0 <= p < len(charset):
                out_chars.append(charset[p])
                out_confs.append(float(np.max(logits[b, t])))
            last = p
        texts.append("".join(out_chars))
        confs.append(float(np.mean(out_confs)) if out_confs else 0.0)
        char_confs.append(list(zip(out_chars, out_confs)))
    return texts, confs, char_confs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # Model source — mutually exclusive: --onnx (ORT) or --engine (pure TRT)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx",   help="Path to ONNX model")
    group.add_argument("--engine", help="Path to TensorRT .engine file (pure TRT, needs pycuda)")

    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "trt"],
                    help="Backend for --onnx: cpu | cuda | trt (ORT-TRT EP)")
    ap.add_argument("--img", nargs="+", required=True, help="One or more image paths")
    ap.add_argument("--expected", nargs="+", default=None,
                    help="Expected texts (same order as --img). Enables verification.")
    ap.add_argument("--dict", default="./tools/utils/ppocr_keys_v1.txt")
    ap.add_argument("--use_space_char", action="store_true", default=True)
    ap.add_argument("--imgH", type=int, default=48)
    ap.add_argument("--imgW", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--debug", action="store_true", help="Print raw output tensor stats")
    args = ap.parse_args()

    if args.expected and len(args.expected) != len(args.img):
        ap.error("--expected must have the same number of values as --img")

    charset  = load_charset(args.dict, use_space_char=args.use_space_char)
    backend  = create_backend(args)


    for _ in range(10):
        img_paths = args.img
        all_texts, all_confs, all_char_confs = [], [], []
        total_pre = total_infer = total_post = 0.0

        for i in range(0, len(img_paths), args.batch_size):
            batch_paths = img_paths[i: i + args.batch_size]

            t0 = time.perf_counter()
            x  = preprocess_batch(batch_paths, imgH=args.imgH, imgW=args.imgW)
            t1 = time.perf_counter()
            y  = backend.infer(x)
            t2 = time.perf_counter()
            if args.debug:
                print(f"[DBG] output shape={y.shape} dtype={y.dtype} "
                    f"min={float(y.min()):.6f} max={float(y.max()):.6f} "
                    f"mean={float(y.mean()):.6f} nan={bool(np.isnan(y).any())}")
            texts, confs, char_confs = ctc_greedy_decode(y, charset)
            t3 = time.perf_counter()

            total_pre   += t1 - t0
            total_infer += t2 - t1
            total_post  += t3 - t2
            all_texts.extend(texts)
            all_confs.extend(confs)
            all_char_confs.extend(char_confs)

        n = len(img_paths)
        total = total_pre + total_infer + total_post

        print(f"\n{'='*52}")
        print(f"  Images      : {n}   |   Batch size : {args.batch_size}")
        print(f"{'='*52}")
        print(f"  Preprocess  : {total_pre*1000:7.2f} ms  ({total_pre*1000/n:.2f} ms/img)")
        print(f"  Inference   : {total_infer*1000:7.2f} ms  ({total_infer*1000/n:.2f} ms/img)")
        print(f"  Postprocess : {total_post*1000:7.2f} ms  ({total_post*1000/n:.2f} ms/img)")
        print(f"  Total       : {total*1000:7.2f} ms  ({total*1000/n:.2f} ms/img)")
        print(f"{'='*52}\n")

        for path, text, conf, ccs in zip(img_paths, all_texts, all_confs, all_char_confs):
            print(f"img : {path}")
            print(f"text: {text}")
            print(f"conf: {conf:.4f}")
            print(f"chars: {' '.join(f'{ch}({c:.2f})' for ch, c in ccs)}")
            print()


if __name__ == "__main__":
    main()


# ── ONNX (CPU) ──────────────────────────────────────────────────────────────
#   python onnx_rec_infer_svtrv2_ctc.py --onnx languages/english/rec_model.onnx --device cpu --img test_image/ocr1.png test_image/ocr2.png --dict ./languages/english/ppocr_keys_v1.txt
#
# ── ONNX + ORT-TRT EP (fp16, auto-fallback) ─────────────────────────────────
#   python onnx_rec_infer_svtrv2_ctc.py \
#       --onnx languages/english/rec_model.onnx --device trt \
#       --img test_image/ocr1.png test_image/ocr2.png --dict ./languages/english/ppocr_keys_v1.txt
#
# ── Pure TensorRT engine ─────────────────────────────────────────────────────
#   python onnx_rec_infer_svtrv2_ctc.py \
#       --engine exports_trt/rec_model.engine \
#       --img test_image/ocr1.png test_image/ocr2.png --dict ./languages/english/ppocr_keys_v1.txt
