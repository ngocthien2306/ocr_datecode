#!/usr/bin/env python3
"""
Benchmark SuperPoint+LightGlue TensorRT engines side by side.

Built for the A/B between the two ONNX sources in weights/:
    superpoint_lightglue_pipeline.onnx  -> 9-layer LightGlue, 1024 keypoints
    superpoint_lightglue_small.onnx     -> 2-layer LightGlue,  512 keypoints
(build them with: python scripts/build_pipeline_engines.py --model
 sp_lg_pipeline_300 sp_lg_small_300 sp_lg_pipeline_480_640 sp_lg_small_480_640)

Feeds a real template/target image pair (not noise, so the match counts mean
something), warms up, then times execute_async_v3 with CUDA events (pure GPU
time) and wall clock (incl. H2D/D2H copies).

Usage:
    conda activate vision
    python scripts/bench_pipeline_engines.py \
        --engine weights/sp_lg_small_fp16_dynamic_300_300.engine \
                 weights/sp_lg_pipeline_fp16_dynamic_300_300.engine
    python scripts/bench_pipeline_engines.py --engine ... --batch 2 6 --iters 100
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
import pycuda.driver as cuda
import tensorrt as trt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


class _Allocator(trt.IOutputAllocator):
    """
    Output allocator for data-dependent shapes (`matches`/`mscores` have a
    NonZero-derived extent, so TensorRT only knows their real size at runtime).
    Grows the device buffer on demand and records the shape TRT reports.
    """

    def __init__(self):
        super().__init__()
        self.buffers = {}   # name -> (device_ptr, nbytes)
        self.shapes = {}    # name -> tuple

    def reallocate_output(self, tensor_name, memory, size, alignment):
        ptr, have = self.buffers.get(tensor_name, (None, 0))
        if have >= size and ptr is not None:
            return ptr
        if ptr is not None:
            ptr.free()
        mem = cuda.mem_alloc(int(size))
        self.buffers[tensor_name] = (mem, int(size))
        return int(mem)

    # TRT 10+ calls the _async variant when present
    def reallocate_output_async(self, tensor_name, memory, size, alignment, stream):
        return self.reallocate_output(tensor_name, memory, size, alignment)

    def notify_shape(self, tensor_name, shape):
        self.shapes[tensor_name] = tuple(shape)
        return True


class EngineRunner:
    def __init__(self, engine_path):
        self.path = engine_path
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = None
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = tuple(self.engine.get_tensor_shape(name))
            else:
                self.output_names.append(name)

        self.h, self.w = self.input_shape[2], self.input_shape[3]
        self.allocator = _Allocator()
        for name in self.output_names:
            self.context.set_output_allocator(name, self.allocator)
        self._d_in = None
        self._d_in_bytes = 0

    def infer(self, batch):
        """batch: float32 (B,1,H,W). Returns {name: np.ndarray}."""
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        if batch.nbytes > self._d_in_bytes:
            if self._d_in is not None:
                self._d_in.free()
            self._d_in = cuda.mem_alloc(batch.nbytes)
            self._d_in_bytes = batch.nbytes

        self.context.set_input_shape(self.input_name, batch.shape)
        cuda.memcpy_htod_async(self._d_in, batch, self.stream)
        self.context.set_tensor_address(self.input_name, int(self._d_in))
        if not self.context.execute_async_v3(stream_handle=self.stream.handle):
            raise RuntimeError("execute_async_v3 failed")
        self.stream.synchronize()

        out = {}
        for name in self.output_names:
            shape = self.allocator.shapes.get(name)
            if shape is None or -1 in shape:
                shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.empty(shape, dtype=dtype)
            if host.size:
                cuda.memcpy_dtoh_async(host, self.allocator.buffers[name][0], self.stream)
            out[name] = host
        self.stream.synchronize()
        return out

    def time_gpu(self, batch, iters):
        """Pure GPU time of execute_async_v3 (ms/iter), input already on device."""
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        self.context.set_input_shape(self.input_name, batch.shape)
        cuda.memcpy_htod(self._d_in, batch)
        self.context.set_tensor_address(self.input_name, int(self._d_in))
        start, end = cuda.Event(), cuda.Event()
        self.stream.synchronize()
        start.record(self.stream)
        for _ in range(iters):
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        end.record(self.stream)
        end.synchronize()
        return start.time_till(end) / iters


def load_pair(template, target, h, w):
    a = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(template)
    if b is None:
        raise FileNotFoundError(target)
    a = cv2.resize(a, (w, h)).astype(np.float32) / 255.0
    b = cv2.resize(b, (w, h)).astype(np.float32) / 255.0
    return a, b


def make_batch(a, b, batch):
    if batch % 2:
        raise ValueError("batch must be even (pairs of template+target)")
    return np.stack([a, b] * (batch // 2))[:, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", nargs="+", required=True, help="engine paths to compare")
    ap.add_argument("--template", default=os.path.join(REPO_ROOT, "test_image", "bottle1.jpg"))
    ap.add_argument("--target", default=os.path.join(REPO_ROOT, "test_image", "bottle2.jpg"))
    ap.add_argument("--batch", nargs="+", type=int, default=[2, 6])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--score-threshold", type=float, default=0.2,
                    help="mscore cut for the reported 'good matches' count")
    args = ap.parse_args()

    print(f"template: {args.template}\ntarget:   {args.target}\n")
    rows = []
    for path in args.engine:
        runner = EngineRunner(path)
        a, b = load_pair(args.template, args.target, runner.h, runner.w)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"=== {os.path.basename(path)} ===")
        print(f"  input {runner.input_shape}  outputs {runner.output_names}  engine {size_mb:.1f} MB")

        for batch in args.batch:
            x = make_batch(a, b, batch)
            for _ in range(args.warmup):
                runner.infer(x)

            out = runner.infer(x)
            kpts = out.get("keypoints")
            matches = out.get("matches")
            mscores = out.get("mscores")
            n_kpts = kpts.shape[1] if kpts is not None and kpts.ndim == 3 else -1
            pair0 = matches[:, 0] == 0 if matches is not None and matches.size else np.zeros(0, bool)
            n_pair0 = int(pair0.sum())
            good = int((mscores[pair0] > args.score_threshold).sum()) if n_pair0 else 0
            mean_s = float(mscores[pair0].mean()) if n_pair0 else 0.0

            gpu_ms = runner.time_gpu(x, args.iters)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                runner.infer(x)
            wall_ms = (time.perf_counter() - t0) / args.iters * 1000

            print(f"  batch={batch}: gpu {gpu_ms:7.2f} ms | e2e {wall_ms:7.2f} ms | "
                  f"{batch / 2 / (wall_ms / 1000):6.1f} pair/s | kpts {n_kpts} | "
                  f"matches(pair0) {n_pair0} | >{args.score_threshold} {good} | mean {mean_s:.3f}")
            rows.append((os.path.basename(path), batch, gpu_ms, wall_ms, n_kpts, n_pair0, good, mean_s))
        print()

    print("=" * 108)
    print(f"{'engine':52s} {'B':>2s} {'gpu ms':>8s} {'e2e ms':>8s} {'pair/s':>8s} "
          f"{'kpts':>6s} {'match':>6s} {'good':>6s} {'score':>6s}")
    for name, batch, gpu, wall, nk, nm, good, ms in rows:
        print(f"{name:52s} {batch:2d} {gpu:8.2f} {wall:8.2f} {batch / 2 / (wall / 1000):8.1f} "
              f"{nk:6d} {nm:6d} {good:6d} {ms:6.3f}")


if __name__ == "__main__":
    sys.exit(main())
