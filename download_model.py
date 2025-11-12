from huggingface_hub import hf_hub_download

# Download English models
det_path = hf_hub_download("monkt/paddleocr-onnx", "detection/v5/det.onnx")
rec_path = hf_hub_download("monkt/paddleocr-onnx", "languages/english/rec.onnx")
dict_path = hf_hub_download("monkt/paddleocr-onnx", "languages/english/dict.txt")

# Use with RapidOCR
from rapidocr_onnxruntime import RapidOCR
ocr = RapidOCR(det_model_path=det_path, rec_model_path=rec_path, rec_keys_path=dict_path)
result, elapsed = ocr("ocr.jpg")

print(result)
print(elapsed)
