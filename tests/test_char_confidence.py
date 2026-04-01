"""
Test per-character confidence từ các OCR recognizer.

Usage:
    python tests/test_char_confidence.py
    python tests/test_char_confidence.py /path/to/image.png
"""
import sys
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
REC_ONNX   = ROOT / "languages/english/rec.onnx"
OPEN_ONNX  = ROOT / "languages/english/openocr_rec_model.onnx"
DICT_PATH      = ROOT / "languages/english/dict.txt"
DICT_PATH_OPEN = ROOT / "languages/english/ppocr_keys_v1.txt"
TEST_IMAGE = ROOT / "text_1.jpg"

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_dict(dict_path: Path, pad_to: int = None):
    with open(dict_path, encoding="utf-8") as f:
        chars = f.read().splitlines()
    char_list = ["blank"] + chars
    if pad_to and len(char_list) < pad_to:
        char_list.extend([" "] * (pad_to - len(char_list)))
    return char_list


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def decode_per_char(pred: np.ndarray, char_list: list, apply_softmax: bool = False):
    """
    CTC decode trả về per-character confidence.

    Args:
        pred        : shape (T, C) — một sample
        char_list   : ['blank', 'A', 'B', ...]
        apply_softmax: True nếu pred là logit (rec.onnx); False nếu đã là prob

    Returns:
        text         : str
        per_char     : list of (char, conf_float)  — chỉ ký tự thực sự decode
        all_timesteps: list of (char_or_blank, conf) — toàn bộ T timestep (debug)
    """
    if apply_softmax:
        prob = softmax(pred)          # (T, C)
    else:
        prob = pred                   # đã là probability

    best_idx  = np.argmax(prob, axis=-1)   # (T,)
    best_prob = prob[np.arange(len(prob)), best_idx]  # (T,)

    # all timesteps (for debug)
    all_timesteps = []
    for t, idx in enumerate(best_idx):
        label = char_list[idx] if idx < len(char_list) else f"[{idx}]"
        all_timesteps.append((label, float(best_prob[t])))

    # CTC collapse: bỏ blank (idx=0) và repeat
    chars, confs = [], []
    prev_idx = 0
    for t, idx in enumerate(best_idx):
        if idx != 0 and idx != prev_idx:
            label = char_list[idx] if idx < len(char_list) else f"[{idx}]"
            chars.append(label)
            # chỉ giữ conf cho ký tự chữ/số, bỏ ký tự đặc biệt
            if label.isalnum():
                confs.append((label, float(best_prob[t])))
        prev_idx = idx

    return "".join(chars), confs, all_timesteps


# ── Preprocess ─────────────────────────────────────────────────────────────────

def preprocess_rec(image: np.ndarray, target_h: int = 48) -> np.ndarray:
    """rec.onnx: grayscale → 3-channel duplicate, normalize [-1,1]"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w  = gray.shape
    new_w = max(int(w * target_h / h), 10)
    resized = cv2.resize(gray, (new_w, target_h))
    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    tensor = np.stack([norm, norm, norm], axis=0)[np.newaxis]  # (1,3,H,W)
    return tensor.astype(np.float32)


def preprocess_openocr(image: np.ndarray, target_h: int = 48) -> np.ndarray:
    """openocr: BGR→RGB, normalize [-1,1]"""
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w  = rgb.shape[:2]
    new_w = max(min(int(target_h * w / h), 320), 10)
    resized = cv2.resize(rgb, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    tensor = norm.transpose(2, 0, 1)[np.newaxis]  # (1,3,H,W)
    return tensor.astype(np.float32)


# ── Print helpers ───────────────────────────────────────────────────────────────

def print_per_char(label: str, text: str, per_char: list):
    bar_width = 20
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  Text    : \"{text}\"")
    print(f"  Avg conf: {np.mean([c for _, c in per_char]):.4f}  (alphanumeric only)" if per_char else "  (no chars)")
    print(f"{'─'*55}")
    print(f"  {'Char':<6} {'Conf':>6}  {'Bar'}")
    print(f"{'─'*55}")
    for ch, cf in per_char:
        filled = int(cf * bar_width)
        bar    = "█" * filled + "░" * (bar_width - filled)
        flag   = "  <-- LOW" if cf < 0.5 else ""
        print(f"  {repr(ch):<6} {cf:>6.4f}  {bar}{flag}")
    print(f"{'='*55}")


def print_all_timesteps(all_ts: list, n: int = 40):
    """In tối đa n timestep đầu để debug."""
    print(f"\n  [DEBUG] First {min(n, len(all_ts))} timesteps:")
    for i, (ch, cf) in enumerate(all_ts[:n]):
        marker = " <blank>" if ch == "blank" else ""
        print(f"    t={i:3d}  {repr(ch):<6} {cf:.4f}{marker}")


# ── Main ────────────────────────────────────────────────────────────────────────

def run_test(image_path: Path, show_timesteps: bool = False):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        sys.exit(1)

    print(f"\nImage  : {image_path.name}")
    print(f"Shape  : {image.shape}")

    char_list = load_dict(DICT_PATH)
    print(f"Dict   : {len(char_list)} chars (incl. blank)")

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])

    # ── Model 1: rec.onnx (TextRecognizer) ────────────────────────────────────
    if REC_ONNX.exists():
        sess  = ort.InferenceSession(str(REC_ONNX), providers=providers)
        iname = sess.get_inputs()[0].name
        oname = sess.get_outputs()[0].name

        num_classes = sess.get_outputs()[0].shape[-1]
        char_list   = load_dict(DICT_PATH, pad_to=num_classes)
        print(f"Dict   : {len(char_list)} chars (padded to {num_classes})")

        tensor = preprocess_rec(image)
        out    = sess.run([oname], {iname: tensor})[0]   # (1, T, C)
        pred   = out[0]                                   # (T, C)

        text, per_char, all_ts = decode_per_char(pred, char_list, apply_softmax=False)
        print_per_char("rec.onnx  (TextRecognizer) — probability output", text, per_char)
        if show_timesteps:
            print_all_timesteps(all_ts)
    else:
        print(f"[SKIP] {REC_ONNX} not found")

    # ── Model 2: openocr_rec_model.onnx (TextRecognizerOpenOCRONNX) ───────────
    if OPEN_ONNX.exists():
        sess_open      = ort.InferenceSession(str(OPEN_ONNX), providers=providers)
        num_classes_op = sess_open.get_outputs()[0].shape[-1]
        open_char_list = load_dict(DICT_PATH_OPEN, pad_to=num_classes_op)
        print(f"\nOpenOCR dict: {len(open_char_list)} chars (padded to {num_classes_op})")

        sess  = sess_open
        iname = sess.get_inputs()[0].name
        oname = sess.get_outputs()[0].name

        tensor = preprocess_openocr(image)
        out    = sess.run([oname], {iname: tensor})[0]   # (1, T, C)
        pred   = out[0]                                   # (T, C) probabilities

        text, per_char, all_ts = decode_per_char(pred, open_char_list, apply_softmax=False)
        print_per_char("openocr_rec_model.onnx  (TextRecognizerOpenOCRONNX)", text, per_char)
        if show_timesteps:
            print_all_timesteps(all_ts)
    else:
        print(f"[SKIP] {OPEN_ONNX} not found")


if __name__ == "__main__":
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else TEST_IMAGE
    show_ts  = "--timesteps" in sys.argv
    run_test(img_path, show_timesteps=show_ts)
