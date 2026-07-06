"""
Desktop ONNX recognizer for SVTRv2 + SMTR/CTC dual-head models.

The public methods mirror the existing desktop recognizers:
    recognize(image) -> (text, confidence)
    recognize_batch(images) -> [(text, confidence), ...]
    recognize_with_char_conf(image) -> (text, confidence, [(char, confidence), ...])
"""

import re
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort


IMG_HEIGHT = 32
CTC_STRIDE = 4
_DISABLED_OPTIMIZERS = ['SimplifiedLayerNormFusion']


class BaseRecLabelDecode:
    """Convert between text labels and text indices."""

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
                for line in fin.readlines():
                    char = line.decode('utf-8').strip('\n').strip('\r\n')
                    self.character_str.append(char)
            if use_space_char:
                self.character_str.append(' ')
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {char: i for i, char in enumerate(dict_character)}
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
                self.character[int(text_id)]
                for text_id in text_index[batch_idx][selection]
                if int(text_id) < len(self.character)
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
        return [0]

    def get_character_num(self):
        return len(self.character)


class CTCLabelDecode(BaseRecLabelDecode):
    """CTC greedy decoder."""

    def __call__(self, preds, batch=None, **kwargs):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if batch is None:
            return text
        label = self.decode(batch[1])
        return text, label

    def add_special_char(self, dict_character):
        return ['blank'] + dict_character

    def decode_char_confs(self, logits_batch: np.ndarray):
        results = []
        for logits in logits_batch:
            best_idx = logits.argmax(axis=-1)
            best_prob = logits.max(axis=-1)
            chars = []
            prev_idx = -1
            for t, idx in enumerate(best_idx):
                idx = int(idx)
                if idx != 0 and idx != prev_idx and idx < len(self.character):
                    chars.append((self.character[idx], float(best_prob[t])))
                prev_idx = idx
            results.append(chars)
        return results


class SMTRLabelDecode(BaseRecLabelDecode):
    """SMTR GTC decoder."""

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
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if batch is None:
            return text
        label = self.decode(batch[1][:, 1:])
        return text, label

    def add_special_char(self, dict_character):
        dict_character = [
            self.EOS,
            *dict_character,
            self.BOS,
            self.IN_F,
            self.IN_B,
            self.PAD,
        ]
        self.num_character = len(dict_character)
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        result_list = []
        for batch_idx in range(len(text_index)):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                try:
                    char_idx = self.character[int(text_index[batch_idx][idx])]
                except Exception:
                    continue
                if char_idx == self.EOS:
                    break
                if char_idx in (self.BOS, self.PAD):
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
            score = np.mean(conf_list).tolist() if conf_list else 0.0
            result_list.append((text, score))
        return result_list

    def decode_char_confs(self, logits_batch: np.ndarray):
        skipped = {self.BOS, self.PAD, self.IN_F, self.IN_B}
        results = []
        for logits in logits_batch:
            best_idx = logits.argmax(axis=-1)
            best_prob = logits.max(axis=-1)
            chars = []
            for t, idx in enumerate(best_idx):
                idx = int(idx)
                if idx >= len(self.character):
                    continue
                char = self.character[idx]
                if char == self.EOS:
                    break
                if char not in skipped:
                    chars.append((char, float(best_prob[t])))
            results.append(chars)
        return results


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


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Resize to height 32, normalize to [-1, 1], return [1, 3, H, W]."""
    if img_bgr is None:
        raise ValueError("Cannot recognize an empty image")
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {img_bgr.shape}")
    new_w = max(int(w * IMG_HEIGHT / h), 1)
    img = cv2.resize(img_bgr, (new_w, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def preprocess_batch(images: List[np.ndarray]) -> np.ndarray:
    tensors = [preprocess(image)[0] for image in images]
    max_w = max(tensor.shape[2] for tensor in tensors)
    padded = [
        np.pad(
            tensor,
            ((0, 0), (0, 0), (0, max_w - tensor.shape[2])),
            mode='constant',
            constant_values=-1,
        )
        for tensor in tensors
    ]
    return np.stack(padded, axis=0).astype(np.float32)


def detect_text_y_band(img_bgr, energy_thr=0.3):
    """Detect the single-line text Y band on the original crop."""
    h, _w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    profile = sy.sum(axis=1)
    k = max(3, h // 30) | 1
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, k), 0).flatten()
    if profile.max() <= 1e-6:
        return 0, h

    high = profile >= profile.max() * energy_thr
    runs = []
    cur = None
    for i, active in enumerate(high):
        if active and cur is None:
            cur = i
        elif not active and cur is not None:
            runs.append((cur, i))
            cur = None
    if cur is not None:
        runs.append((cur, len(high)))
    if not runs:
        return 0, h

    runs.sort(key=lambda run: run[1] - run[0], reverse=True)
    y0, y1 = runs[0]
    if y1 - y0 < h * 0.2 or y1 - y0 > h * 0.98:
        return 0, h
    pad = max(2, h // 30)
    return max(0, int(y0) - pad), min(h, int(y1) + pad)


def ctc_align_chars(ctc_logits, ctc_decode, w_orig, resized_w, stride=CTC_STRIDE):
    """Approximate per-character X ranges from CTC best-path centers."""
    logits = ctc_logits[0] if ctc_logits.ndim == 3 else ctc_logits
    best_idx = logits.argmax(axis=-1)
    best_prob = logits.max(axis=-1)
    scale = w_orig / max(float(resized_w), 1.0)

    chars = []
    prev_idx = -1
    for t, idx in enumerate(best_idx):
        idx = int(idx)
        if idx != 0 and idx != prev_idx and idx < len(ctc_decode.character):
            center = (t + 0.5) * stride * scale
            chars.append({
                'char': ctc_decode.character[idx],
                'center': float(center),
                'conf': float(best_prob[t]),
            })
        prev_idx = idx

    for i, char in enumerate(chars):
        left = 0 if i == 0 else (chars[i - 1]['center'] + char['center']) / 2
        right = w_orig if i == len(chars) - 1 else (
            char['center'] + chars[i + 1]['center']
        ) / 2
        char['x_start'] = max(0, int(round(left)))
        char['x_end'] = min(w_orig, int(round(right)))
    return chars


def smtr_peak_x_ranges(attn_maps, smtr_steps, w_orig):
    """Use SMTR attention X peaks, split by midpoints between adjacent peaks."""
    feat_w = attn_maps.shape[2]
    peaks = []
    for step in smtr_steps:
        heat = attn_maps[step].astype(np.float32)
        x_profile = heat.max(axis=0)
        peaks.append(int(x_profile.argmax()))

    scale_x = w_orig / float(feat_w)
    ranges = []
    for i, peak in enumerate(peaks):
        left = 0 if i == 0 else (peaks[i - 1] + peak) / 2
        right = feat_w if i == len(peaks) - 1 else (peak + peaks[i + 1]) / 2
        ranges.append((int(left * scale_x), int(right * scale_x)))
    return ranges


def attention_char_bboxes(attn_maps, gtc_logits, ctc_logits, img_bgr,
                          smtr_decode, ctc_decode, resized_w, x_source='auto'):
    """Return char dictionaries with bbox fields from attention/CTC alignment."""
    h_orig, w_orig = img_bgr.shape[:2]
    pred_idx = gtc_logits.argmax(-1)
    pred_prob = gtc_logits.max(-1)

    smtr_chars = []
    smtr_steps = []
    for step in range(len(pred_idx)):
        try:
            char = smtr_decode.character[int(pred_idx[step])]
        except Exception:
            continue
        if char == SMTRLabelDecode.EOS:
            break
        if char in (SMTRLabelDecode.BOS, SMTRLabelDecode.PAD):
            continue
        if step >= len(attn_maps):
            continue
        smtr_chars.append(char)
        smtr_steps.append(step)

    if not smtr_chars:
        return []

    ctc_chars = ctc_align_chars(ctc_logits[None], ctc_decode, w_orig, resized_w)
    text_match = (
        len(ctc_chars) == len(smtr_chars)
        and all(ctc_char['char'] == smtr_char
                for ctc_char, smtr_char in zip(ctc_chars, smtr_chars))
    )

    if x_source == 'auto':
        x_source = 'ctc' if text_match else 'attn'
    if x_source == 'attn' or not text_match:
        attn_ranges = smtr_peak_x_ranges(attn_maps, smtr_steps, w_orig)
    else:
        attn_ranges = None

    y0, y1 = detect_text_y_band(img_bgr)
    chars = []
    for index, (char, step) in enumerate(zip(smtr_chars, smtr_steps)):
        heat = attn_maps[step].astype(np.float32)
        if x_source == 'ctc' and text_match:
            x0 = ctc_chars[index]['x_start']
            x1 = ctc_chars[index]['x_end']
            bbox_src = 'ctc'
        else:
            x0, x1 = attn_ranges[index]
            bbox_src = 'attn'

        chars.append({
            'char': char,
            'conf': float(pred_prob[step]),
            'col': index,
            'x0': int(x0),
            'y0': int(y0),
            'x1': int(x1),
            'y1': int(y1),
            'heat_max': float(heat.max()),
            'bbox_src': bbox_src,
            'text_match': bool(text_match),
            'image_h': int(h_orig),
            'image_w': int(w_orig),
        })
    return chars


def _providers_for_device(device: str):
    available = ort.get_available_providers()
    providers = []

    if device == 'trt':
        if 'TensorrtExecutionProvider' in available:
            providers.append((
                'TensorrtExecutionProvider',
                {
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': './trt_cache',
                },
            ))
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
    elif device == 'cuda':
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
    elif device != 'cpu':
        raise ValueError(f"Unknown SMTR ONNX device: {device}")

    if 'CPUExecutionProvider' in available:
        providers.append('CPUExecutionProvider')
    return providers or available


class TextRecognizerSMTRONNX:
    """SVTRv2 + SMTR/CTC recognizer for ONNX Runtime."""

    def __init__(self, model_path: str, dict_path: str, device: str = 'cpu',
                 model_label: str = 'SMTR ONNX', x_source: str = 'auto'):
        self.model_path = str(model_path)
        self.dict_path = str(dict_path)
        self.device = device
        self.model_label = model_label
        self.x_source = x_source

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"SMTR ONNX model not found: {self.model_path}")
        if not Path(self.dict_path).exists():
            raise FileNotFoundError(f"SMTR dictionary not found: {self.dict_path}")

        self.gtc_decode, self.ctc_decode = build_postprocessors(self.dict_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = _providers_for_device(device)
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers,
                disabled_optimizers=_DISABLED_OPTIMIZERS,
            )
        except TypeError:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers,
            )
        self.input_name = self.session.get_inputs()[0].name

        print(f"{self.model_label} recognizer initialized")
        print(f"   Model: {Path(self.model_path).name}")
        print(f"   Dictionary: {len(self.gtc_decode.character_str)} characters")
        print(f"   Provider: {self.provider}")

    @property
    def provider(self):
        return self.session.get_providers()[0]

    def _infer(self, tensor: np.ndarray):
        outputs = self.session.run(None, {self.input_name: tensor.astype(np.float32)})
        if len(outputs) < 2:
            raise ValueError(f"Expected SMTR model to return 2 outputs, got {len(outputs)}")
        attn_maps = outputs[2] if len(outputs) >= 3 else None
        return outputs[0], outputs[1], attn_maps

    def _decode(self, gtc_logits, ctc_logits, index=0, with_chars=False):
        gtc_results = self.gtc_decode(gtc_logits, torch_tensor=False)
        ctc_results = self.ctc_decode(ctc_logits, torch_tensor=False)

        gtc_text, gtc_score = gtc_results[index]
        ctc_text, ctc_score = ctc_results[index]
        use_ctc = ctc_score >= gtc_score

        if use_ctc:
            text, score = ctc_text, float(ctc_score)
            char_confs = self.ctc_decode.decode_char_confs(ctc_logits)[index]
        else:
            text, score = gtc_text, float(gtc_score)
            char_confs = self.gtc_decode.decode_char_confs(gtc_logits)[index]

        if with_chars:
            char_confs = [(char, conf) for char, conf in char_confs if char.isalnum()]
            return text, score, char_confs
        return text, score

    def recognize(self, image: np.ndarray, return_confidence: bool = True):
        gtc_logits, ctc_logits, _attn_maps = self._infer(preprocess(image))
        text, score = self._decode(gtc_logits, ctc_logits)
        return (text, score) if return_confidence else text

    def recognize_with_char_conf(self, image: np.ndarray):
        gtc_logits, ctc_logits, _attn_maps = self._infer(preprocess(image))
        return self._decode(gtc_logits, ctc_logits, with_chars=True)

    def recognize_with_chars(self, image: np.ndarray):
        tensor = preprocess(image)
        gtc_logits, ctc_logits, attn_maps = self._infer(tensor)
        text, avg_conf, char_confs = self._decode(gtc_logits, ctc_logits, with_chars=True)
        if attn_maps is not None:
            chars = attention_char_bboxes(
                attn_maps[0],
                gtc_logits[0],
                ctc_logits[0],
                image,
                self.gtc_decode,
                self.ctc_decode,
                tensor.shape[3],
                x_source=self.x_source,
            )
            if chars:
                return text, avg_conf, chars
        chars = [
            {'char': char, 'conf': float(conf), 'col': i}
            for i, (char, conf) in enumerate(char_confs)
        ]
        return text, avg_conf, chars

    def recognize_batch(self, images: List[np.ndarray], batch_size=None):
        if not images:
            return []

        if batch_size is None or batch_size <= 0:
            batch_size = len(images)

        results = []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            gtc_logits, ctc_logits, _attn_maps = self._infer(preprocess_batch(chunk))
            for index in range(len(chunk)):
                results.append(self._decode(gtc_logits, ctc_logits, index=index))
        return results
