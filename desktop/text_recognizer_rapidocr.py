"""
RapidOCR-based text recognition.

RapidOCR wraps PP-OCR models in ONNX Runtime — same family of models we already
use, but bundled with detection + classification + recognition end-to-end and
its own preprocessing. We mostly skip detection (our crops are already
segmented per-region) and run recognition only for speed.

Same `recognize(image_bgr) → (text, confidence)` API as the other recognizers.
"""
import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False
    print("Warning: rapidocr_onnxruntime not installed. `pip install rapidocr_onnxruntime`")


# Module-level cache so we don't re-init the engine (slow: ONNX model load)
# per recognize call. Keyed by config_path.
_ENGINE_CACHE = {}


def _get_engine(config_path=None):
    key = config_path or '_default_'
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = RapidOCR(config_path=config_path) if config_path else RapidOCR()
    return _ENGINE_CACHE[key]


class TextRecognizerRapidOCR:
    def __init__(self, use_det=False, use_cls=False, use_rec=True,
                 config_path=None):
        """
        Args:
            use_det: run detection (find text bboxes). Default False — our crops
                     already only contain one text line.
            use_cls: run angle-classifier (180° rotation detection). Default False.
            use_rec: run recognition. Default True.
            config_path: optional path to a custom RapidOCR config.yaml
        """
        if not RAPIDOCR_AVAILABLE:
            raise ImportError("rapidocr_onnxruntime is not available")

        self.use_det = bool(use_det)
        self.use_cls = bool(use_cls)
        self.use_rec = bool(use_rec)
        self.engine = _get_engine(config_path)
        # We bypass det/cls and call the recognizer directly so we can ask
        # for per-character confidences (return_word_box=True). The high-level
        # __call__ doesn't expose that flag.
        self.text_rec = self.engine.text_rec

        print(f"✅ RapidOCR initialized "
              f"(det={self.use_det}, cls={self.use_cls}, rec={self.use_rec})")

    @staticmethod
    def _prepare(image, upscale_min_h=48):
        if image is None or image.size == 0:
            return None
        h = image.shape[0]
        if h < upscale_min_h:
            scale = upscale_min_h / h
            image = cv2.resize(image, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
        return image

    def recognize(self, image_bgr, return_confidence=True):
        """OCR a single image. Returns (text, mean_confidence)."""
        text, mean_conf, _per_char = self.recognize_with_chars(image_bgr)
        return (text, mean_conf) if return_confidence else text

    def recognize_with_chars(self, image_bgr):
        """
        OCR a single image and return per-character information.

        Returns:
            (text, mean_confidence, chars)
            where `chars` is a list of dicts:
              [{'char': 'L', 'conf': 0.99, 'col': 1}, ...]
            `col` is the position in the model's feature map (proxy for
            x-position in the input image — useful for aligning the per-char
            confidence with our own char bboxes).
        """
        prepped = self._prepare(image_bgr)
        if prepped is None:
            return "", 0.0, []

        try:
            # Direct call to text_rec with return_word_box=True so the decoder
            # exposes the per-char confidence list (mean is what the high-level
            # __call__ returns).
            results, _elapsed = self.text_rec([prepped], return_word_box=True)
        except Exception as e:
            print(f"[rapidocr] error: {e}")
            return "", 0.0, []

        if not results:
            return "", 0.0, []

        # results[0] = (text, mean_conf, [feature_w, [chars], [cols], states, conf_list])
        first = results[0]
        if len(first) >= 3:
            text, mean_conf, extra = first[0], first[1], first[2]
            char_list = (extra[1][0] if extra and len(extra) > 1 and extra[1] else
                         list(text))
            col_list = (extra[2][0] if extra and len(extra) > 2 and extra[2] else
                        [0] * len(char_list))
            conf_list = extra[4] if extra and len(extra) > 4 else [mean_conf] * len(char_list)
        else:
            text, mean_conf = first[0], first[1]
            char_list = list(text)
            col_list = [0] * len(char_list)
            conf_list = [mean_conf] * len(char_list)

        # Pad shorter lists defensively
        n = max(len(char_list), len(conf_list), len(col_list))
        char_list = (char_list + [''] * n)[:n]
        col_list = (list(col_list) + [0] * n)[:n]
        conf_list = (list(conf_list) + [mean_conf] * n)[:n]

        chars = [
            {'char': c, 'conf': float(p), 'col': int(col)}
            for c, p, col in zip(char_list, conf_list, col_list)
            if c != ''
        ]
        text = (text or '').strip()
        mean_conf = max(0.0, min(1.0, float(mean_conf)))
        return text, mean_conf, chars

    def recognize_batch(self, images_bgr):
        return [self.recognize(img, return_confidence=True) for img in images_bgr]
