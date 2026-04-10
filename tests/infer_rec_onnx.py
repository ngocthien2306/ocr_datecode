import argparse
import os
import re
import time

import cv2
import numpy as np
import onnxruntime as ort

try:
    import torch
except ImportError:
    torch = None


# ── Label-decode classes (inlined from openrec/postprocess) ───────────────────

class BaseRecLabelDecode:
    """Convert between text-label and text-index."""

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = 'sos'
        self.end_str = 'eos'
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = '0123456789abcdefghijklmnopqrstuvwxyz'
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, 'rb') as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip('\n').strip('\r\n')
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(' ')
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)
        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)
            if self.reverse:
                text = self.pred_reverse(text)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank

    def get_character_num(self):
        return len(self.character)


class CTCLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index."""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super().__init__(character_dict_path, use_space_char)

    def __call__(self, preds, batch=None, **kwargs):
        if kwargs.get('torch_tensor', True):
            preds = preds.detach().cpu().numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if batch is None:
            return text
        label = self.decode(batch[1])
        return text, label

    def decode_char_confs(self, logits_batch: np.ndarray):
        """logits_batch: [N, T, C]  → List[N] of [(char, conf), ...]"""
        results = []
        for logits in logits_batch:                 # logits: [T, C]
            best_idx  = logits.argmax(axis=-1)      # [T]
            best_prob = logits.max(axis=-1)         # [T]
            chars = []
            prev = -1
            for t, idx in enumerate(best_idx):
                if idx != 0 and idx != prev:        # bỏ blank + duplicate
                    if idx < len(self.character):
                        chars.append((self.character[idx], float(best_prob[t])))
                prev = idx
            results.append(chars)
        return results

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class SMTRLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index."""

    BOS = '<s>'
    EOS = '</s>'
    IN_F = '<INF>'
    IN_B = '<INB>'
    PAD = '<pad>'

    def __init__(self, character_dict_path=None, use_space_char=True,
                 next_mode=True, **kwargs):
        super().__init__(character_dict_path, use_space_char)
        self.next_mode = next_mode

    def __call__(self, preds, batch=None, *args, **kwargs):
        if isinstance(preds, list):
            preds = preds[-1]
        if torch is not None and isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if batch is None:
            return text
        label = batch[1]
        label = self.decode(label[:, 1:])
        return text, label

    def decode_char_confs(self, logits_batch: np.ndarray):
        """logits_batch: [N, T, C]  → List[N] of [(char, conf), ...]"""
        SKIP = {'<s>', '<pad>', '<INF>', '<INB>'}
        results = []
        for logits in logits_batch:
            best_idx  = logits.argmax(axis=-1)
            best_prob = logits.max(axis=-1)
            chars = []
            for t, idx in enumerate(best_idx):
                if idx < len(self.character):
                    ch = self.character[idx]
                    if ch == '</s>':    # EOS → dừng
                        break
                    if ch in SKIP:
                        continue
                    chars.append((ch, float(best_prob[t])))
            results.append(chars)
        return results

    def add_special_char(self, dict_character):
        dict_character = [self.EOS] + dict_character + [
            self.BOS, self.IN_F, self.IN_B, self.PAD
        ]
        self.num_character = len(dict_character)
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                try:
                    char_idx = self.character[int(text_index[batch_idx][idx])]
                except Exception:
                    continue
                if char_idx == '</s>':
                    break
                if char_idx == '<s>' or char_idx == '<pad>':
                    continue
                char_list.append(char_idx)
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            if self.next_mode or text_prob is None:
                text = ''.join(char_list)
            else:
                text = ''.join(char_list[::-1])
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

ONNX_MODEL   = './languages/english/rec_smtr_fp16.onnx'
DICT_PATH    = './languages/english/EN_symbol_dict.txt'
IMG_HEIGHT   = 32


# ── Postprocessors ────────────────────────────────────────────────────────────

def build_postprocessors(dict_path):
    gtc_decode = SMTRLabelDecode(
        character_dict_path=dict_path,
        use_space_char=True,
        next_mode=True,
    )
    ctc_decode = CTCLabelDecode(
        character_dict_path=dict_path,
        use_space_char=True,
    )
    return gtc_decode, ctc_decode


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Resize to fixed height 32, normalize to [-1, 1], return [1,3,H,W]."""
    h, w = img_bgr.shape[:2]
    new_w = max(int(w * IMG_HEIGHT / h), 1)
    img = cv2.resize(img_bgr, (new_w, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5          # normalize to [-1, 1]
    img = img.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]
    return img


def preprocess_batch(imgs_bgr: list) -> np.ndarray:
    """Preprocess a list of BGR images into a single batched tensor [N,3,H,max_W].

    Each image is resized to height IMG_HEIGHT while keeping aspect ratio,
    then zero-padded on the right to match the widest image in the batch.
    """
    tensors = []
    for img_bgr in imgs_bgr:
        h, w = img_bgr.shape[:2]
        new_w = max(int(w * IMG_HEIGHT / h), 1)
        img = cv2.resize(img_bgr, (new_w, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5   # [H, W, 3]
        img = img.transpose(2, 0, 1)        # [3, H, W]
        tensors.append(img)

    max_w = max(t.shape[2] for t in tensors)
    padded = []
    for t in tensors:
        pad_w = max_w - t.shape[2]
        # pad value 0 corresponds to (0 - 0.5)/0.5 = -1 (black)
        t_pad = np.pad(t, ((0, 0), (0, 0), (0, pad_w)), mode='constant', constant_values=-1)
        padded.append(t_pad)

    return np.stack(padded, axis=0)  # [N, 3, H, max_W]


# ── Inference ─────────────────────────────────────────────────────────────────

def get_image_paths(path: str):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    if os.path.isfile(path):
        return [path]
    return sorted(
        os.path.join(path, f) for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in exts
    )


def run(img_path: str, sess: ort.InferenceSession,
        gtc_decode: SMTRLabelDecode, ctc_decode: CTCLabelDecode,
        verbose: bool = False):
    t0 = time.perf_counter()
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {img_path}')
    inp = preprocess(img_bgr)
    t_pre = time.perf_counter() - t0

    t1 = time.perf_counter()
    gtc_logits, ctc_logits = sess.run(None, {'image': inp})
    t_infer = time.perf_counter() - t1

    t2 = time.perf_counter()
    gtc_results = gtc_decode(gtc_logits, torch_tensor=False)
    ctc_results = ctc_decode(ctc_logits, torch_tensor=False)
    t_post = time.perf_counter() - t2

    # Pick the decoder with higher confidence
    gtc_text, gtc_score = gtc_results[0]
    ctc_text, ctc_score = ctc_results[0]
    text, score = (ctc_text, ctc_score) if ctc_score >= gtc_score else (gtc_text, gtc_score)

    if verbose:
        total = t_pre + t_infer + t_post
        print(f'  preprocess : {t_pre*1000:6.2f} ms')
        print(f'  inference  : {t_infer*1000:6.2f} ms')
        print(f'  postprocess: {t_post*1000:6.2f} ms')
        print(f'  total      : {total*1000:6.2f} ms')
        print(f'  gtc: {gtc_text!r} ({gtc_score:.4f})  ctc: {ctc_text!r} ({ctc_score:.4f})')

    return text, score


def _print_char_conf(label: str, char_confs: list):
    BAR = '█'
    RESET, GREEN, YELLOW, RED = '\033[0m', '\033[92m', '\033[93m', '\033[91m'
    text = ''.join(c for c, _ in char_confs)
    avg  = float(np.mean([s for _, s in char_confs])) if char_confs else 0.0
    print(f"  [{label}] '{text}'  avg={avg:.4f}")
    for ch, conf in char_confs:
        color = GREEN if conf >= 0.80 else (YELLOW if conf >= 0.50 else RED)
        filled = round(conf * 10)
        bar = f"{color}{BAR * filled}{'░' * (10 - filled)}{RESET}"
        print(f"    {ch!r:>4}  {conf:.4f}  {bar}")


def run_with_char_conf(img_bgr: np.ndarray, sess: ort.InferenceSession,
                       gtc_decode: SMTRLabelDecode, ctc_decode: CTCLabelDecode):
    """Single image → prints per-char conf table for both heads, returns (text, score)."""
    inp = preprocess(img_bgr)
    gtc_logits, ctc_logits = sess.run(None, {'image': inp})

    gtc_cc = gtc_decode.decode_char_confs(gtc_logits)[0]
    ctc_cc = ctc_decode.decode_char_confs(ctc_logits)[0]

    gtc_results = gtc_decode(gtc_logits, torch_tensor=False)
    ctc_results = ctc_decode(ctc_logits, torch_tensor=False)
    gtc_text, gtc_score = gtc_results[0]
    ctc_text, ctc_score = ctc_results[0]
    text, score = (ctc_text, ctc_score) if ctc_score >= gtc_score else (gtc_text, gtc_score)

    _print_char_conf('GTC', gtc_cc)
    _print_char_conf('CTC', ctc_cc)
    print(f"  winner='{text}'  score={score:.4f}")
    return text, score


def run_batch(imgs_bgr: list, sess: ort.InferenceSession,
              gtc_decode: SMTRLabelDecode, ctc_decode: CTCLabelDecode,
              verbose: bool = False):
    """Run inference on a batch of BGR images (no size limit).

    Returns a list of (text, score) tuples, one per image.
    """
    if not imgs_bgr:
        return []

    t0 = time.perf_counter()
    inp = preprocess_batch(imgs_bgr)   # [N, 3, H, max_W]
    t_pre = time.perf_counter() - t0

    t1 = time.perf_counter()
    gtc_logits, ctc_logits = sess.run(None, {'image': inp})
    t_infer = time.perf_counter() - t1

    t2 = time.perf_counter()
    gtc_results = gtc_decode(gtc_logits, torch_tensor=False)
    ctc_results = ctc_decode(ctc_logits, torch_tensor=False)
    t_post = time.perf_counter() - t2

    results = []
    for gtc, ctc in zip(gtc_results, ctc_results):
        gtc_text, gtc_score = gtc
        ctc_text, ctc_score = ctc
        text, score = (ctc_text, ctc_score) if ctc_score >= gtc_score else (gtc_text, gtc_score)
        results.append((text, score))

    if verbose:
        total = t_pre + t_infer + t_post
        n = len(imgs_bgr)
        print(f'  batch_size : {n}')
        print(f'  preprocess : {t_pre*1000:6.2f} ms')
        print(f'  inference  : {t_infer*1000:6.2f} ms')
        print(f'  postprocess: {t_post*1000:6.2f} ms')
        print(f'  total      : {total*1000:6.2f} ms  ({total/n*1000:.2f} ms/img)')

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',   required=True,    help='image file or directory')
    parser.add_argument('--model', default=ONNX_MODEL)
    parser.add_argument('--dict',  default=DICT_PATH)
    parser.add_argument('--gpu',     action='store_true')
    parser.add_argument('--batch',     type=int, default=0,
                        help='batch size for inference (0 = all images at once)')
    parser.add_argument('--char-conf', action='store_true',
                        help='print per-character confidence for both GTC and CTC heads')
    parser.add_argument('--verbose', action='store_true', help='print per-component timing')
    args = parser.parse_args()

    providers = [
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,  # 4 GB
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
            "trt_min_subgraph_size": 3,
        }),
        "CUDAExecutionProvider",   # fallback for unsupported ops
        "CPUExecutionProvider",
    ]

    sess = ort.InferenceSession(args.model, providers=providers)

    gtc_decode, ctc_decode = build_postprocessors(args.dict)

    img_paths = get_image_paths(args.img)

    if args.char_conf:
        # Per-character confidence mode — ảnh từng cái, show bảng chi tiết
        for path in img_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            print(f'\n{"="*55}')
            print(f'  {os.path.basename(path)}')
            run_with_char_conf(img, sess, gtc_decode, ctc_decode)
        return

    batch_size = args.batch if args.batch > 0 else len(img_paths)
    for _ in range(10):
        for start in range(0, len(img_paths), batch_size):
            chunk_paths = img_paths[start:start + batch_size]
            imgs = [cv2.imread(p) for p in chunk_paths]
            valid = [(p, img) for p, img in zip(chunk_paths, imgs) if img is not None]
            if not valid:
                continue
            paths_ok, imgs_ok = zip(*valid)
            results = run_batch(list(imgs_ok), sess, gtc_decode, ctc_decode,
                                verbose=args.verbose)
            for path, (text, score) in zip(paths_ok, results):
                print(f'{path}\t{text}\t{score:.4f}')


if __name__ == '__main__':
    main()
