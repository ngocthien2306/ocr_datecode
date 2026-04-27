"""
Tesseract-based text recognition. Supports two backing libraries:

  • tesserocr  — C-API binding, in-process (~5-10ms/call). Faster but needs
                 a C++ build chain to install.
  • pytesseract — subprocess wrapper around the `tesseract` CLI
                 (~30-100ms/call). Slower per call (subprocess spawn each time)
                 but trivial to install: `pip install pytesseract`.

Same `recognize(image_bgr) → (text, confidence)` API regardless of backend,
so it's a drop-in replacement for TextRecognizer / TextRecognizerTensorRT.
"""
import cv2
import numpy as np
import os
import subprocess

# ---- Try the faster C-binding first; pytesseract is the fallback ----
try:
    import tesserocr  # noqa: F401
    TESSEROCR_AVAILABLE = True
except ImportError:
    TESSEROCR_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    Image = None

if TESSEROCR_AVAILABLE:
    from PIL import Image  # tesserocr also needs PIL


def _autodetect_tessdata():
    """Find tessdata dir without forcing the user to set TESSDATA_PREFIX."""
    env = os.environ.get('TESSDATA_PREFIX')
    if env and os.path.isdir(env):
        return env
    candidates = [
        '/opt/homebrew/opt/tesseract/share/tessdata',
        '/opt/homebrew/share/tessdata',
        '/usr/local/share/tessdata',
        '/usr/share/tessdata',
        '/usr/share/tesseract-ocr/4.00/tessdata',
        '/usr/share/tesseract-ocr/5/tessdata',
    ]
    for p in candidates:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'eng.traineddata')):
            return p
    try:
        out = subprocess.check_output(
            ['tesseract', '--list-langs'], stderr=subprocess.STDOUT, text=True
        )
        first = out.splitlines()[0]
        if '"' in first:
            return first.split('"')[1]
    except Exception:
        pass
    return None


def _bgr_to_pil(image):
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _prepare(image, upscale_min_h=48):
    """Light preprocessing: upscale tiny crops + grayscale."""
    if image is None or image.size == 0:
        return None
    h = image.shape[0]
    if h < upscale_min_h:
        scale = upscale_min_h / h
        image = cv2.resize(image, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


class TextRecognizerTesseract:
    """
    Common front-end. Constructor picks a backend based on `library`:
      • 'auto'        — tesserocr if available, else pytesseract (default)
      • 'tesserocr'   — force C-binding
      • 'pytesseract' — force subprocess wrapper
    """
    def __init__(self, lang='eng', psm=7, oem=1,
                 char_whitelist=None, tessdata_path=None,
                 library='auto'):
        self.lang = lang
        self.psm = int(psm)
        self.oem = int(oem)
        self.char_whitelist = char_whitelist or ''
        self.tessdata_path = tessdata_path or _autodetect_tessdata()

        if library == 'auto':
            library = 'tesserocr' if TESSEROCR_AVAILABLE else 'pytesseract'

        if library == 'tesserocr':
            if not TESSEROCR_AVAILABLE:
                raise ImportError("tesserocr not installed. `pip install tesserocr`")
            self._impl = _TesserocrImpl(self.lang, self.psm, self.oem,
                                         self.char_whitelist, self.tessdata_path)
        elif library == 'pytesseract':
            if not PYTESSERACT_AVAILABLE:
                raise ImportError("pytesseract not installed. `pip install pytesseract Pillow`")
            self._impl = _PytesseractImpl(self.lang, self.psm, self.oem,
                                           self.char_whitelist, self.tessdata_path)
        else:
            raise ValueError(f"Unknown library: {library}")

        self.library = library
        print(f"✅ Tesseract initialized via {library} "
              f"(lang={lang}, psm={psm}, oem={oem})")
        if self.char_whitelist:
            print(f"   Whitelist: {self.char_whitelist!r}")

    def recognize(self, image_bgr, return_confidence=True):
        gray = _prepare(image_bgr)
        if gray is None:
            return ("", 0.0) if return_confidence else ""
        text, conf = self._impl.recognize(gray)
        return (text, conf) if return_confidence else text

    def recognize_batch(self, images_bgr):
        return [self.recognize(img, return_confidence=True) for img in images_bgr]


# --------------------------------------------------------------------- impls

class _TesserocrImpl:
    def __init__(self, lang, psm, oem, char_whitelist, tessdata_path):
        kwargs = {'lang': lang, 'psm': int(psm), 'oem': int(oem)}
        if tessdata_path:
            kwargs['path'] = tessdata_path.rstrip('/') + '/'
        self.api = tesserocr.PyTessBaseAPI(**kwargs)
        if char_whitelist:
            self.api.SetVariable('tessedit_char_whitelist', char_whitelist)

    def recognize(self, gray):
        pil = _bgr_to_pil(gray)
        self.api.SetImage(pil)
        text = self.api.GetUTF8Text().strip()
        conf = float(self.api.MeanTextConf()) / 100.0
        return text, max(0.0, min(1.0, conf))

    def __del__(self):
        try:
            self.api.End()
        except Exception:
            pass


class _PytesseractImpl:
    def __init__(self, lang, psm, oem, char_whitelist, tessdata_path):
        self.lang = lang
        # Build a CLI-style config string. pytesseract passes this verbatim.
        bits = [f'--psm {int(psm)}', f'--oem {int(oem)}']
        if tessdata_path:
            bits.append(f'--tessdata-dir "{tessdata_path}"')
        if char_whitelist:
            bits.append(f'-c tessedit_char_whitelist={char_whitelist}')
        self.config = ' '.join(bits)

    def recognize(self, gray):
        pil = _bgr_to_pil(gray)
        # image_to_data gives both text + per-word confidence
        try:
            data = pytesseract.image_to_data(
                pil, lang=self.lang, config=self.config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as e:
            print(f"[pytesseract] error: {e}")
            return "", 0.0
        # Filter non-empty word entries; conf is "-1" for image-level rows
        words, confs = [], []
        for w, c in zip(data.get('text', []), data.get('conf', [])):
            w = (w or '').strip()
            try:
                c = float(c)
            except Exception:
                continue
            if w and c >= 0:
                words.append(w)
                confs.append(c)
        text = ' '.join(words)
        conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        return text, max(0.0, min(1.0, conf))
