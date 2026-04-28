#!/usr/bin/env python3
import sys, cv2, numpy as np, onnxruntime as ort, torch
from pathlib import Path
from omegaconf import OmegaConf
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "desktop"))
from char_segmenter import segment_characters

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr, size):
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas[(size-nh)//2:(size-nh)//2+nh, (size-nw)//2:(size-nw)//2+nw] = img
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_centroids(sess, input_name, feat_dim, dataset_root, multi_to_idx, size, cache_path):
    if cache_path.exists():
        return np.load(str(cache_path))
    sums   = np.zeros((len(multi_to_idx), feat_dim), dtype=np.float64)
    counts = np.zeros(len(multi_to_idx), dtype=np.int64)
    for char_dir in sorted(Path(dataset_root).iterdir()):
        if not char_dir.is_dir() or not char_dir.name.startswith("char_"):
            continue
        for sub in ("ok", "ng"):
            key = f"{char_dir.name}__{sub}"
            if key not in multi_to_idx:
                continue
            idx  = multi_to_idx[key]
            imgs = [p for p in sorted((char_dir / sub).iterdir()) if p.suffix.lower() in VALID_EXT]
            if not imgs:
                continue
            batch = np.stack([preprocess(cv2.imread(str(p)), size) for p in imgs])
            embs  = sess.run(None, {input_name: batch})[0]
            embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            sums[idx]  += embs.sum(0)
            counts[idx] += len(imgs)
    c  = (sums / np.maximum(counts[:, None], 1)).astype(np.float32)
    c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-8
    np.save(str(cache_path), c)
    return c


def to_pixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))


class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR OK/NG Viewer")
        self.resize(1300, 720)
        self.sess = self.head = self.cfg_size = self.input_name = None
        self.centroids = self.idx_to_ok_ng = None
        self.idx_to_char = {}
        self.template_emb = None
        self.threshold = 0.5
        self.pad_w = self.pad_h = 6
        self.temperature = 5.0
        self.folder = None
        self._build_ui()

    def _build_ui(self):
        tb = self.addToolBar("Tools")
        tb.setMovable(False)

        def btn(label, slot):
            b = QPushButton(label)
            b.clicked.connect(slot)
            return b

        def spin_d(lo, hi, val, step, slot):
            s = QDoubleSpinBox()
            s.setRange(lo, hi); s.setSingleStep(step); s.setValue(val)
            s.valueChanged.connect(slot)
            return s

        def spin_i(lo, hi, val, slot):
            s = QSpinBox()
            s.setRange(lo, hi); s.setValue(val)
            s.valueChanged.connect(slot)
            return s

        tb.addWidget(btn("Load Model",  self._load_model))
        tb.addSeparator()
        self.tmpl_btn = QPushButton("Load Template")
        self.tmpl_btn.clicked.connect(self._load_template)
        self.tmpl_btn.setEnabled(False)
        self.tmpl_btn.setToolTip("Only available for projection / arcface models")
        tb.addWidget(self.tmpl_btn)
        tb.addSeparator()
        tb.addWidget(btn("Load Folder", self._load_folder))
        tb.addSeparator()

        tb.addWidget(QLabel(" Threshold:"))
        self.th_spin = spin_d(0.0, 1.0, 0.5, 0.05, lambda v: setattr(self, "threshold", v))
        tb.addWidget(self.th_spin)

        tb.addWidget(QLabel("  Temp:"))
        tb.addWidget(spin_d(0.5, 20.0, 5.0, 0.5, lambda v: setattr(self, "temperature", v)))

        tb.addWidget(QLabel("  Pad W:"))
        tb.addWidget(spin_i(0, 30, 6, lambda v: setattr(self, "pad_w", v)))
        tb.addWidget(QLabel("  Pad H:"))
        tb.addWidget(spin_i(0, 30, 6, lambda v: setattr(self, "pad_h", v)))

        splitter = QSplitter(Qt.Horizontal)

        self.img_list = QListWidget()
        self.img_list.setMaximumWidth(260)
        self.img_list.currentTextChanged.connect(self._on_select)
        splitter.addWidget(self.img_list)

        right = QWidget()
        vbox  = QVBoxLayout(right)
        vbox.setContentsMargins(4, 4, 4, 4)

        self.img_label = QLabel(alignment=Qt.AlignCenter)
        self.img_label.setMinimumSize(600, 400)
        self.img_label.setStyleSheet("background:#1e1e1e;")
        vbox.addWidget(self.img_label, stretch=4)

        self.log = QTextEdit(readOnly=True)
        self.log.setFont(QFont("Courier New", 10))
        self.log.setMaximumHeight(180)
        vbox.addWidget(self.log, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 4)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Load a model run directory to start.")

    def _load_model(self):
        run_dir = QFileDialog.getExistingDirectory(self, "Select run directory")
        if not run_dir:
            return
        run_dir = Path(run_dir)
        try:
            cfg           = OmegaConf.load(run_dir / "config.yaml")
            self.cfg_size = cfg.data.image_size
            self.head     = cfg.model.head.type
            self.template_emb = None
            self.tmpl_btn.setEnabled(self.head in ("projection", "arcface"))

            ckpt             = torch.load(run_dir / "best.pt", map_location="cpu")
            multi_to_idx      = ckpt.get("multi_to_idx", {})
            self.idx_to_ok_ng = {v: (0 if k.endswith("__ok") else 1) for k, v in multi_to_idx.items()}
            self.idx_to_char  = {v: k.split("__")[0].replace("char_", "") for k, v in multi_to_idx.items()}
            raw_th  = ckpt.get("metrics", {}).get("threshold", 0.0)
            if cfg.model.head.type in ("projection", "arcface"):
                # stored threshold is raw (sim_ng - sim_ok) ∈ [-2,2] → convert to sigmoid scale
                best_th = float(sigmoid(np.array([raw_th]) * self.temperature))
            else:
                best_th = raw_th
            self.threshold = best_th
            self.th_spin.setValue(best_th)

            self.sess       = ort.InferenceSession(str(run_dir / "model.onnx"),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
            feat_dim        = self.sess.get_outputs()[0].shape[-1]

            self.centroids = None
            if self.head == "projection":
                cache = run_dir / "centroids.npy"
                if cache.exists():
                    self.centroids = np.load(str(cache))
                    self.statusBar().showMessage(f"Loaded (cached centroids) — {run_dir.name}")
                else:
                    dataset = QFileDialog.getExistingDirectory(self, "Select dataset root to build centroids")
                    if dataset:
                        self.statusBar().showMessage("Building centroids, please wait...")
                        QApplication.processEvents()
                        self.centroids = build_centroids(self.sess, self.input_name, feat_dim,
                                                         dataset, multi_to_idx, self.cfg_size, cache)
                        self.statusBar().showMessage(f"Loaded — {run_dir.name}")
                    else:
                        self.statusBar().showMessage("Warning: no centroids built — projection head needs dataset")
            else:
                self.statusBar().showMessage(f"Loaded [{self.head}] — {run_dir.name}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _load_template(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select template image(s) — OK samples only", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not paths:
            return
        try:
            tensors = np.stack([preprocess(cv2.imread(p), self.cfg_size) for p in paths])
            if self.head == "projection":
                embs = self.sess.run(None, {self.input_name: tensors})[0]
            else:  # arcface
                embs = self.sess.run(None, {self.input_name: tensors})[1]
            embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            tmpl = embs.mean(axis=0)
            tmpl /= np.linalg.norm(tmpl) + 1e-8
            self.template_emb = tmpl
            self.th_spin.setValue(0.5)
            self.threshold = 0.5
            names = ", ".join(Path(p).name for p in paths[:2])
            suffix = f" (+{len(paths)-2} more)" if len(paths) > 2 else ""
            self.statusBar().showMessage(f"[Template] {names}{suffix}  — {len(paths)} image(s)")
        except Exception as e:
            QMessageBox.critical(self, "Template error", str(e))

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        self.folder = Path(folder)
        self.img_list.clear()
        for p in sorted(self.folder.iterdir()):
            if p.suffix.lower() in VALID_EXT:
                self.img_list.addItem(p.name)
        self.statusBar().showMessage(f"{self.img_list.count()} images — {self.folder.name}")

    def _on_select(self, name):
        if not name or not self.sess or not self.folder:
            return
        self._run(self.folder / name)

    def _run(self, path):
        try:
            boxes, char_imgs, _, proc_img, _ = segment_characters(str(path))
        except Exception as e:
            self.log.setText(f"Segmentation error:\n{e}")
            return

        if not boxes:
            self.log.setText("No characters segmented.")
            self._show(proc_img)
            return

        def pad(img):
            return cv2.copyMakeBorder(img, self.pad_h, self.pad_h, self.pad_w, self.pad_w,
                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))

        tensors = np.stack([preprocess(pad(c), self.cfg_size) for c in char_imgs])

        pred_chars = ["?"] * len(tensors)
        mode_label = "P_NG"

        if self.template_emb is not None and self.head in ("projection", "arcface"):
            # Template matching mode — no centroids needed
            if self.head == "projection":
                embs = self.sess.run(None, {self.input_name: tensors})[0]
            else:
                embs = self.sess.run(None, {self.input_name: tensors})[1]
            embs  /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            sims   = embs @ self.template_emb          # cosine sim ∈ [-1, 1]
            scores = (1.0 - sims) / 2.0                # 0 = identical, 1 = opposite
            mode_label = "dist"
        elif self.head == "linear":
            logits = self.sess.run(None, {self.input_name: tensors})[0]
            scores = softmax(logits)[:, 1]
        elif self.head == "projection":
            embs    = self.sess.run(None, {self.input_name: tensors})[0]
            embs   /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            sims    = embs @ self.centroids.T
            ng_mask = np.array([self.idx_to_ok_ng[i] for i in range(len(self.idx_to_ok_ng))], dtype=bool)
            raw     = np.where(ng_mask, sims, -2.).max(1) - np.where(~ng_mask, sims, -2.).max(1)
            scores  = sigmoid(raw * self.temperature)
            pred_chars = [self.idx_to_char.get(int(i), "?") for i in sims.argmax(axis=1)]
        else:  # arcface — cos_sim baked into ONNX (output[1]), no centroids needed
            cos_sim = self.sess.run(None, {self.input_name: tensors})[1]
            ng_mask = np.array([self.idx_to_ok_ng[i] for i in range(len(self.idx_to_ok_ng))], dtype=bool)
            raw     = np.where(ng_mask, cos_sim, -2.).max(1) - np.where(~ng_mask, cos_sim, -2.).max(1)
            scores  = sigmoid(raw * self.temperature)
            pred_chars = [self.idx_to_char.get(int(i), "?") for i in cos_sim.argmax(axis=1)]

        vis   = proc_img.copy()
        n_ng  = 0
        lines = [f"{'#':>3}  {'char':>6}  {mode_label:>6}  {'conf%':>6}  verdict"]
        for i, (box, s, ch) in enumerate(zip(boxes, scores, pred_chars)):
            x, y, w, h = box
            ng    = bool(s >= self.threshold)
            n_ng += ng
            color = (0, 0, 255) if ng else (0, 200, 0)
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
            cv2.putText(vis, f"{ch}:{s*100:.0f}%", (x, max(15, y-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            lines.append(f"{i:>3}  {ch:>6}  {s:>6.4f}  {s*100:>5.1f}%  {'NG' if ng else 'OK'}")

        lines.append(f"\n→ {n_ng}/{len(boxes)} NG   threshold={self.threshold:.4f}")
        self.log.setText("\n".join(lines))
        self._show(vis)

    def _show(self, bgr):
        px = to_pixmap(bgr)
        self.img_label.setPixmap(
            px.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Viewer()
    w.show()
    sys.exit(app.exec_())
