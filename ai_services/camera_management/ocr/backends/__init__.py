"""
Concrete text-recognition backends.

One module per (model family x runtime): the model family decides the pre/post
processing, the runtime (ONNX Runtime vs TensorRT) decides only how the graph
is loaded and executed. These are the raw recognizer classes — ocr/factory.py
adapts each one to the OCRBackendStrategy interface and picks between them.

Imported lazily by the factory rather than eagerly here: each backend pulls in
its own heavy runtime (tensorrt/pycuda or onnxruntime), and a deployment that
only uses one shouldn't pay for — or hard-fail on — the others being absent.
"""
