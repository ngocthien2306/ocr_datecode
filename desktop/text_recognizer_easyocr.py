"""
EasyOCR-based text recognition.

EasyOCR is a deep-learning OCR (PyTorch under the hood) with strong multilingual
support out of the box. Slower than Tesseract on CPU and bigger models, but
often more robust on noisy / stylised text.

Same `recognize(image_bgr) → (text, confidence)` API as the other recognizers
so it's a drop-in replacement.
"""
import cv2
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr not installed. `pip install easyocr`")


# Module-level cache so we don't re-init the reader (slow: model load) per
# recognize call. Keyed by (langs_tuple, gpu).
_READER_CACHE = {}


def _get_reader(langs, gpu, model_storage_directory=None):
    """Lazy + cached easyocr.Reader so model weights load only once per process."""
    key = (tuple(langs), bool(gpu),
           model_storage_directory or '')
    if key in _READER_CACHE:
        return _READER_CACHE[key]
    kwargs = {
        'lang_list': list(langs),
        'gpu': bool(gpu),
        'verbose': False,
    }
    if model_storage_directory:
        kwargs['model_storage_directory'] = model_storage_directory
    reader = easyocr.Reader(**kwargs)
    _READER_CACHE[key] = reader
    return reader


class TextRecognizerEasyOCR:
    def __init__(self, langs=('en',), gpu=False,
                 allowlist=None, model_storage_directory=None,
                 detail=1, paragraph=False):
        """
        Args:
            langs: list/tuple of language codes, e.g. ('en',) or ('en', 'vi')
            gpu:   use CUDA (will fall back to CPU if torch sees no GPU)
            allowlist: optional str of allowed chars (like Tesseract whitelist)
            model_storage_directory: where to cache .pth weights (default ~/.EasyOCR)
            detail: 1 = return bbox+text+conf (we use this); 0 = text only
            paragraph: True = combine adjacent boxes into a paragraph
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError("easyocr is not available")

        self.langs = tuple(langs)
        self.gpu = bool(gpu)
        self.allowlist = allowlist
        self.detail = int(detail)
        self.paragraph = bool(paragraph)

        # Lazy load (cached at module level so reusing the same langs/gpu
        # combo across recognizers shares the model).
        self.reader = _get_reader(self.langs, self.gpu, model_storage_directory)

        print(f"✅ EasyOCR initialized (langs={self.langs}, gpu={self.gpu})")
        if self.allowlist:
            print(f"   Allowlist: {self.allowlist!r}")

    @staticmethod
    def _prepare(image, upscale_min_h=48):
        """Light preprocessing: upscale tiny crops. EasyOCR accepts BGR or
        grayscale numpy arrays directly."""
        if image is None or image.size == 0:
            return None
        h = image.shape[0]
        if h < upscale_min_h:
            scale = upscale_min_h / h
            image = cv2.resize(image, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
        return image

    def recognize(self, image_bgr, return_confidence=True):
        """OCR a single image. Returns (text, confidence)."""
        prepped = self._prepare(image_bgr)
        if prepped is None:
            return ("", 0.0) if return_confidence else ""

        kwargs = {
            'detail': 1,                # always need confidence internally
            'paragraph': self.paragraph,
        }
        if self.allowlist:
            kwargs['allowlist'] = self.allowlist

        try:
            results = self.reader.readtext(prepped, **kwargs)
        except Exception as e:
            print(f"[easyocr] error: {e}")
            return ("", 0.0) if return_confidence else ""

        # Each result = (bbox, text, conf) when detail=1
        texts = []
        confs = []
        for item in results:
            try:
                _bbox, text, conf = item
            except (TypeError, ValueError):
                continue
            text = (text or '').strip()
            if not text:
                continue
            texts.append(text)
            try:
                confs.append(float(conf))
            except Exception:
                pass

        # Sort by left edge if we have bboxes (most use cases want L→R order)
        # already preserved by EasyOCR by default; just join.
        full_text = ' '.join(texts)
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
        avg_conf = max(0.0, min(1.0, avg_conf))

        return (full_text, avg_conf) if return_confidence else full_text

    def recognize_batch(self, images_bgr):
        return [self.recognize(img, return_confidence=True) for img in images_bgr]
