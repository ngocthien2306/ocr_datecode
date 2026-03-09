#!/usr/bin/env python3
import math
import time
import numpy as np
import cv2
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TensorRTOCR:
    def __init__(self, engine_path, dict_path='./languages/english/ppocr_keys_v1.txt', use_space_char=True):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.use_new_api = hasattr(self.engine, 'num_io_tensors')
        self.stream = cuda.Stream()

        self.input_name = self.output_name = None
        self.input_shape = self.output_shape = None

        if self.use_new_api:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_name, self.input_shape = name, shape
                else:
                    self.output_name, self.output_shape = name, shape
        else:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                if self.engine.binding_is_input(i):
                    self.input_name, self.input_shape = name, shape
                else:
                    self.output_name, self.output_shape = name, shape

        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = f.read().splitlines()
        if use_space_char:
            chars.append(' ')
        self.character = ['blank'] + chars

    # ------------------------------------------------------------------ #
    #  Augmentation for laser-engraved text on light/grey background      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def augment_laser_text(img_bgr):
        """
        Generate 5 enhanced versions optimized for white laser-engraved text
        on light/grey background.

        Returns:
            dict with keys: 'original', 'clahe', 'invert_otsu',
                            'adaptive_thresh', 'bg_subtract', 'sharpen_clahe'
            All values are BGR uint8 arrays with same H/W as input.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        results = {'original': img_bgr.copy()}

        # 1. CLAHE – adaptive local contrast enhancement
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab_eq = cv2.merge([clahe.apply(l), a, b])
        results['clahe'] = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # 2. Background subtraction – blur to estimate BG, amplify residual text signal
        bg = cv2.GaussianBlur(gray, (51, 51), 0)
        diff_amp = cv2.convertScaleAbs(cv2.subtract(gray, bg), alpha=8)
        results['bg_subtract'] = cv2.cvtColor(diff_amp, cv2.COLOR_GRAY2BGR)

        # 5. Unsharp masking – subtract blurred version to amplify edges,
        #    less halo artifact than convolution kernel
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        unsharp = cv2.addWeighted(gray, 2.0, blurred, -1.0, 0)
        results['unsharp_clahe'] = cv2.cvtColor(clahe.apply(unsharp), cv2.COLOR_GRAY2BGR)

        # 6. Morphological TOPHAT – extracts bright regions (text) smaller than kernel,
        #    very effective for bright laser text on light background
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 20))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_morph)
        tophat_eq = cv2.convertScaleAbs(tophat, alpha=6)
        results['tophat'] = cv2.cvtColor(clahe.apply(tophat_eq), cv2.COLOR_GRAY2BGR)

        return results

    @staticmethod
    def save_augmented(img_input, out_dir='./aug_debug'):
        """Save all augmented versions to out_dir for visual inspection."""
        import os
        os.makedirs(out_dir, exist_ok=True)
        if isinstance(img_input, str):
            base = os.path.splitext(os.path.basename(img_input))[0]
            img_bgr = cv2.imread(img_input)
        else:
            base = 'image'
            img_bgr = img_input
        for name, img in TensorRTOCR.augment_laser_text(img_bgr).items():
            cv2.imwrite(os.path.join(out_dir, f'{base}_{name}.png'), img)

    def recognize_augmented(self, img_input, return_all=False):
        """
        Run OCR on original + 5 augmented versions, return highest-confidence result.

        Args:
            img_input: file path (str) or BGR numpy array
            return_all: if True, returns (best_result, all_results_dict)

        Returns:
            best result dict {'text', 'confidence', 'time', 'version'}
        """
        if isinstance(img_input, str):
            img_bgr = cv2.imread(img_input)
        else:
            img_bgr = img_input

        versions = self.augment_laser_text(img_bgr)
        all_results = {}
        best = None

        for ver_name, img_ver in versions.items():
            r = self.recognize(img_ver)
            r['version'] = ver_name
            all_results[ver_name] = r
            if best is None or r['confidence'] > best['confidence']:
                best = r

        return (best, all_results) if return_all else best

    # ------------------------------------------------------------------ #
    #  Core preprocessing / inference                                     #
    # ------------------------------------------------------------------ #

    def _preprocess_array(self, img_rgb):
        h, w = img_rgb.shape[:2]
        base_h, max_ratio = 48, 12
        ratio_resize = min(max(1, round(w / h)), max_ratio)
        target_w = base_h * ratio_resize
        resized_w = min(target_w, int(math.ceil(base_h * w / h)))
        img = cv2.resize(img_rgb, (resized_w, base_h), interpolation=cv2.INTER_CUBIC)
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        padded = np.zeros((1, 3, base_h, target_w), dtype=np.float32)
        padded[0, :, :, :resized_w] = img.transpose(2, 0, 1)
        return padded

    def preprocess(self, img_input):
        if isinstance(img_input, np.ndarray):
            img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = np.array(Image.open(img_input).convert('RGB'))
        return self._preprocess_array(img_rgb)

    def postprocess(self, output):
        preds_idx = np.argmax(output, axis=2)[0]
        preds_prob = np.max(output, axis=2)[0]
        chars, confs, last = [], [], 0
        for i, idx in enumerate(preds_idx):
            if idx != 0 and idx != last and idx < len(self.character):
                chars.append(self.character[idx])
                confs.append(preds_prob[i])
            last = idx
        return ''.join(chars), float(np.mean(confs)) if confs else 0.0

    def _alloc_and_infer(self, input_data):
        input_shape = input_data.shape
        if self.use_new_api:
            self.context.set_input_shape(self.input_name, input_shape)
            d_input = cuda.mem_alloc(input_data.nbytes)
            self.context.set_tensor_address(self.input_name, int(d_input))
            output_shape = self.context.get_tensor_shape(self.output_name)
            output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
            output_data = np.empty(output_shape, dtype=output_dtype)
            d_output = cuda.mem_alloc(output_data.nbytes)
            self.context.set_tensor_address(self.output_name, int(d_output))
        else:
            self.context.set_binding_shape(0, input_shape)
            d_input = cuda.mem_alloc(input_data.nbytes)
            output_shape = self.context.get_binding_shape(1)
            output_data = np.empty(output_shape, dtype=np.float32)
            d_output = cuda.mem_alloc(output_data.nbytes)

        cuda.memcpy_htod_async(d_input, input_data, self.stream)
        t0 = time.time()
        if self.use_new_api:
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(
                bindings=[int(d_input), int(d_output)],
                stream_handle=self.stream.handle
            )
        self.stream.synchronize()
        elapsed = time.time() - t0
        cuda.memcpy_dtoh_async(output_data, d_output, self.stream)
        self.stream.synchronize()
        d_input.free()
        d_output.free()
        return output_data, elapsed

    def recognize(self, img_input):
        input_data = self.preprocess(img_input)
        output_data, elapsed = self._alloc_and_infer(input_data)
        text, conf = self.postprocess(output_data)
        return {'text': text, 'confidence': conf, 'time': elapsed}

    def recognize_batch(self, img_inputs, use_true_batch=False):
        max_batch = self.input_shape[0] if isinstance(self.input_shape[0], int) else -1

        if use_true_batch and max_batch != 1:
            if max_batch > 0 and len(img_inputs) > max_batch:
                results = []
                for i in range(0, len(img_inputs), max_batch):
                    results.extend(self.recognize_batch(img_inputs[i:i + max_batch], use_true_batch=True))
                return results

            preprocessed = [self.preprocess(x) for x in img_inputs]
            max_w = max(p.shape[3] for p in preprocessed)
            batch = np.zeros((len(img_inputs), 3, 48, max_w), dtype=np.float32)
            for i, p in enumerate(preprocessed):
                batch[i, :, :, :p.shape[3]] = p[0]

            output_data, elapsed = self._alloc_and_infer(batch)
            avg = elapsed / len(img_inputs)
            results = []
            for i in range(len(img_inputs)):
                text, conf = self.postprocess(output_data[i:i + 1])
                results.append({'text': text, 'confidence': conf, 'time': avg})
            return results

        return [self.recognize(x) for x in img_inputs]


if __name__ == '__main__':
    import sys, os

    if len(sys.argv) < 2:
        print("Usage: python trt_ocr.py <image_path> [--augment] [--save-aug [dir]] [image_path2 ...]")
        sys.exit(1)

    use_augment = '--augment' in sys.argv
    save_aug    = '--save-aug' in sys.argv
    aug_dir     = './aug_debug'
    skip = {'--augment', '--save-aug'}
    if save_aug:
        idx = sys.argv.index('--save-aug')
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith('--'):
            aug_dir = sys.argv[idx + 1]
            skip.add(aug_dir)
    img_paths = [
        "tests/crop_output/5_40767171_Frame_2_text_3_exp[PRODUCT_OF_ITALY]_got[PROOUCT_OF_ITALY]_FAIL_c0.96.jpg"
    ]

    batch_engine = './languages/english/openocr_rec_model_batch.engine'
    single_engine = './languages/english/openocr_rec_model.engine'
    engine_path = batch_engine if os.path.exists(batch_engine) else single_engine

    ocr = TensorRTOCR(engine_path)

    if save_aug:
        for path in img_paths:
            TensorRTOCR.save_augmented(path, aug_dir)
        print(f"Augmented images saved to: {aug_dir}")

    if len(img_paths) == 1 and use_augment:
        best, all_results = ocr.recognize_augmented(img_paths[0], return_all=True)
        for ver, r in all_results.items():
            marker = ' <-- best' if ver == best['version'] else ''
            print(f"  [{ver:<16}] '{r['text']}'  conf={r['confidence']:.4f}{marker}")
        print(f"\nBest: '{best['text']}'  conf={best['confidence']:.4f}  version={best['version']}")
    elif len(img_paths) == 1:
        r = ocr.recognize(img_paths[0])
        print(f"Text: {r['text']}  Conf: {r['confidence']:.4f}  Time: {r['time']*1000:.2f}ms")
    else:
        t0 = time.time()
        for path in img_paths:
            if use_augment:
                best, _ = ocr.recognize_augmented(path)
                print(f"{path}: '{best['text']}'  conf={best['confidence']:.4f}  ver={best['version']}")
            else:
                r = ocr.recognize(path)
                print(f"{path}: {r['text']}  ({r['confidence']:.4f}, {r['time']*1000:.2f}ms)")
        total = time.time() - t0
        print(f"Total: {total*1000:.2f}ms | Avg: {total*1000/len(img_paths):.2f}ms | {len(img_paths)/total:.2f} img/s")