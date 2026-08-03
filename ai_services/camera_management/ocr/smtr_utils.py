import re
import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

IMG_HEIGHT = 32


class BaseRecLabelDecode:
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
                    line = line.decode('utf-8').strip('\n').strip('\r\n')
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(' ')
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {char: i for i, char in enumerate(dict_character)}
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re, c_current = [], ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current:
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current:
            pred_re.append(c_current)
        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        for batch_idx in range(len(text_index)):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for tok in ignored_tokens:
                selection &= text_index[batch_idx] != tok
            char_list = [self.character[i] for i in text_index[batch_idx][selection]]
            conf_list = text_prob[batch_idx][selection] if text_prob is not None else [1] * len(selection)
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
        return text, self.decode(batch[1])

    def decode_char_confs(self, logits_batch: np.ndarray):
        results = []
        for logits in logits_batch:
            best_idx  = logits.argmax(axis=-1)
            best_prob = logits.max(axis=-1)
            chars, prev = [], -1
            for t, idx in enumerate(best_idx):
                if idx != 0 and idx != prev and idx < len(self.character):
                    chars.append((self.character[idx], float(best_prob[t])))
                prev = idx
            results.append(chars)
        return results

    def add_special_char(self, dict_character):
        return ['blank'] + dict_character


class SMTRLabelDecode(BaseRecLabelDecode):
    BOS = '<s>'
    EOS = '</s>'
    IN_F = '<INF>'
    IN_B = '<INB>'
    PAD = '<pad>'

    def __init__(self, character_dict_path=None, use_space_char=True, next_mode=True, **kwargs):
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
        label = self.decode(batch[1][:, 1:])
        return text, label

    def decode_char_confs(self, logits_batch: np.ndarray):
        SKIP = {'<s>', '<pad>', '<INF>', '<INB>'}
        results = []
        for logits in logits_batch:
            best_idx  = logits.argmax(axis=-1)
            best_prob = logits.max(axis=-1)
            chars = []
            for t, idx in enumerate(best_idx):
                if idx < len(self.character):
                    ch = self.character[idx]
                    if ch == '</s>':
                        break
                    if ch not in SKIP:
                        chars.append((ch, float(best_prob[t])))
            results.append(chars)
        return results

    def add_special_char(self, dict_character):
        dict_character = [self.EOS] + dict_character + [self.BOS, self.IN_F, self.IN_B, self.PAD]
        self.num_character = len(dict_character)
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        result_list = []
        for batch_idx in range(len(text_index)):
            char_list, conf_list = [], []
            for idx in range(len(text_index[batch_idx])):
                try:
                    ch = self.character[int(text_index[batch_idx][idx])]
                except Exception:
                    continue
                if ch == '</s>':
                    break
                if ch in ('<s>', '<pad>'):
                    continue
                char_list.append(ch)
                conf_list.append(text_prob[batch_idx][idx] if text_prob is not None else 1)
            text = ''.join(char_list) if self.next_mode or text_prob is None else ''.join(char_list[::-1])
            result_list.append((text, np.mean(conf_list).tolist() if conf_list else 0.0))
        return result_list


def build_postprocessors(dict_path):
    gtc = SMTRLabelDecode(character_dict_path=dict_path, use_space_char=True, next_mode=True)
    ctc = CTCLabelDecode(character_dict_path=dict_path, use_space_char=True)
    return gtc, ctc


def preprocess(img_bgr: np.ndarray, max_width: int = None) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    new_w = max(int(w * IMG_HEIGHT / h), 1)
    # Clamp về profile width tối đa của engine (TRT) — tránh tensor vượt
    # optimization profile gây setInputShape bị từ chối → output rác.
    if max_width is not None:
        new_w = min(new_w, max_width)
    img = cv2.resize(img_bgr, (new_w, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5
    return img.transpose(2, 0, 1)[np.newaxis]


def preprocess_batch(imgs_bgr: list, max_width: int = None) -> np.ndarray:
    tensors = []
    for img_bgr in imgs_bgr:
        h, w = img_bgr.shape[:2]
        new_w = max(int(w * IMG_HEIGHT / h), 1)
        # Clamp từng crop trước khi pad: crop hẹp giữ nguyên (chỉ bị pad), chỉ
        # crop quá rộng mới bị nén — nhờ vậy width batch luôn ≤ max_width và
        # không crop nào "đầu độc" cả batch.
        if max_width is not None:
            new_w = min(new_w, max_width)
        img = cv2.resize(img_bgr, (new_w, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5
        tensors.append(img.transpose(2, 0, 1))
    max_w = max(t.shape[2] for t in tensors)
    padded = [
        np.pad(t, ((0,0),(0,0),(0, max_w - t.shape[2])), mode='constant', constant_values=-1)
        for t in tensors
    ]
    return np.stack(padded, axis=0)


def _decode_dual(gtc_logits, ctc_logits, gtc_decode, ctc_decode):
    gtc_texts = gtc_decode(gtc_logits, torch_tensor=False)
    ctc_texts = ctc_decode(ctc_logits, torch_tensor=False)
    gtc_cc    = gtc_decode.decode_char_confs(gtc_logits)
    ctc_cc    = ctc_decode.decode_char_confs(ctc_logits)
    return [
        [(gt, gc, gcc), (ct, cc, ccc)]
        for (gt, gc), (ct, cc), gcc, ccc
        in zip(gtc_texts, ctc_texts, gtc_cc, ctc_cc)
    ]


def _to_candidates(result) -> list:
    if isinstance(result, list) and result and isinstance(result[0], (list, tuple)):
        return [(item[0], item[1], item[2] if len(item) > 2 else None) for item in result]
    if isinstance(result, dict):
        return [(result['text'], result['confidence'], result.get('char_confs'))]
    cc = result[2] if len(result) > 2 else None
    return [(result[0], result[1], cc)]
