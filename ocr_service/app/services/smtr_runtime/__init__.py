"""
Vendored copy of ai_services' SMTR runtime: the pre/post-processing and the two
recognizer backends, byte-for-byte from
`ai_services/camera_management/ocr/{smtr_utils.py,backends/smtr_{trt,onnx}.py}`
apart from one relative import (`..smtr_utils` -> `.smtr_utils`).

Why copied and not imported: importing across service trees would mean either
executing `camera_management/__init__.py` — which pulls in pypylon, the Basler
camera SDK — or stubbing the package to skip it, and a stub that bypasses
__init__ breaks silently the moment ai_services changes its layout or adds an
import. A 416-line copy with no dependencies outside itself is the cheaper risk.

The cost is drift: this service's reported accuracy is only meaningful if it
decodes exactly the way production does, and nothing stops the two copies from
diverging. `python check_runtime_parity.py` compares them and is the thing to run
after touching either side.
"""
