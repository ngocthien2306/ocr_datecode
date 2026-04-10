import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT/PyCUDA not available.")

from .smtr_utils import build_postprocessors, preprocess, preprocess_batch, _decode_dual


class TextRecognizerSMTRTRT:
    def __init__(self, engine_path: str, dict_path: str):
        if not TENSORRT_AVAILABLE:
            raise ImportError("tensorrt / pycuda not installed")

        self._gtc, self._ctc = build_postprocessors(dict_path)

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self._engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())

        self._ctx    = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        self._in_name  = None
        self._out_names = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._in_name = name
            else:
                self._out_names.append(name)
        assert self._in_name and len(self._out_names) == 2

        _, _, in_max = self._engine.get_tensor_profile_shape(self._in_name, 0)
        self._ctx.set_input_shape(self._in_name, in_max)
        self._d_in = cuda.mem_alloc(int(np.prod(in_max)) * 4)
        self._d_outs = []
        for name in self._out_names:
            shape = tuple(self._ctx.get_tensor_shape(name))
            self._d_outs.append(cuda.mem_alloc(int(np.prod(shape)) * 4))
        self._ctx.set_tensor_address(self._in_name, int(self._d_in))
        for name, buf in zip(self._out_names, self._d_outs):
            self._ctx.set_tensor_address(name, int(buf))

        self._infer(np.zeros(in_max, dtype=np.float32))
        logger.info(f"TextRecognizerSMTRTRT | {Path(engine_path).name} | max_batch={in_max[0]}")

    def _infer(self, x: np.ndarray):
        x = np.ascontiguousarray(x, dtype=np.float32)
        self._ctx.set_input_shape(self._in_name, x.shape)
        out_shapes = [tuple(self._ctx.get_tensor_shape(n)) for n in self._out_names]
        cuda.memcpy_htod_async(self._d_in, x, self._stream)
        self._ctx.execute_async_v3(self._stream.handle)
        outputs = []
        for buf, shape in zip(self._d_outs, out_shapes):
            arr = np.empty(shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(arr, buf, self._stream)
            outputs.append(arr)
        self._stream.synchronize()
        return outputs[0], outputs[1]

    def recognize(self, image: np.ndarray):
        gtc, ctc = self._infer(preprocess(image))
        return _decode_dual(gtc, ctc, self._gtc, self._ctc)[0]

    def recognize_with_char_conf(self, image: np.ndarray):
        return self.recognize(image)

    def recognize_batch(self, images: list):
        if not images:
            return []
        gtc, ctc = self._infer(preprocess_batch(images))
        return _decode_dual(gtc, ctc, self._gtc, self._ctc)

    def __del__(self):
        try:
            self._d_in.free()
            for b in self._d_outs:
                b.free()
        except Exception:
            pass
