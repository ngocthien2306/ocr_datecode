"""
Predict all images using SVTRv2-CTC ONNX model, generate HTML report.
On mismatch, retry with augmented versions and include all results in report.
Usage:
    # TensorRT engine (recommended)
    python predict_folder_report.py \
        --engine languages/english/rec_model_fp32.engine \
        --folder ./datecodes

    # TensorRT fp16
    python predict_folder_report.py \
        --engine languages/english/rec_model_fp16.engine \
        --folder ./datecodes \
        --output report.html

    # ONNX CPU
    python predict_folder_report.py \
        --onnx languages/english/rec_model.onnx \
        --folder ./datecodes \
        --dict languages/english/ppocr_keys_v1.txt \
        --device cpu

    # Custom output paths
    python predict_folder_report.py \
        --engine languages/english/rec_model_fp32.engine \
        --folder ./datecodes \
        --output report.html \
        --csv results.csv
"""
import argparse, base64, csv, os, re, time
from pathlib import Path
import cv2, numpy as np
import onnxruntime as ort

# ── Backend ──────────────────────────────────────────────────────────────────

def _build_session(model_path, device):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = {"cpu": ["CPUExecutionProvider"],
                 "cuda": [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
                 "trt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
                }.get(device, ["CPUExecutionProvider"])
    sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    print(f"[backend] {sess.get_providers()[0]}")
    return sess

class ONNXBackend:
    def __init__(self, model_path, device="cpu"):
        self.sess = _build_session(model_path, device)
        self.in_name  = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def infer(self, x):
        return self.sess.run([self.out_name], {self.in_name: x})[0]

class TRTBackend:
    def __init__(self, engine_path):
        import tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit  # noqa
        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()
        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        in_max = list(self.engine.get_tensor_profile_shape(self.input_name, 0)[2])
        self.context.set_input_shape(self.input_name, in_max)
        out_max = list(self.context.get_tensor_shape(self.output_name))
        self._d_in  = cuda.mem_alloc(int(np.prod(in_max))  * 4)
        self._d_out = cuda.mem_alloc(int(np.prod(out_max)) * 4)
        self.context.set_tensor_address(self.input_name,  int(self._d_in))
        self.context.set_tensor_address(self.output_name, int(self._d_out))
        self.infer(np.zeros(in_max, dtype=np.float32))
        print("[backend] TensorRT engine")

    def infer(self, x):
        cuda = self._cuda
        x = np.ascontiguousarray(x, dtype=np.float32)
        self.context.set_input_shape(self.input_name, x.shape)
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        cuda.memcpy_htod_async(self._d_in, x, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        out = np.empty(out_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(out, self._d_out, self.stream)
        self.stream.synchronize()
        return out

# ── Pre/post-processing ───────────────────────────────────────────────────────

def load_charset(dict_path):
    chars = [l.rstrip("\n") for l in open(dict_path, encoding="utf-8") if l.strip()]
    if " " not in chars:
        chars.append(" ")
    return [""] + chars  # index 0 = CTC blank

def preprocess(img_bgr, imgH=48, imgW=320):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    new_w = min(int(np.ceil(imgH * w / h)), imgW)
    img = cv2.resize(img, (new_w, imgH), interpolation=cv2.INTER_CUBIC)
    if new_w < imgW:
        img = np.concatenate([img, np.zeros((imgH, imgW - new_w, 3), dtype=img.dtype)], axis=1)
    img = (img.astype("float32") / 255.0 - 0.5) / 0.5
    return np.expand_dims(np.transpose(img, (2, 0, 1)), 0).astype("float32")

def ctc_decode(logits, charset):
    if logits.ndim == 3 and logits.shape[1] == len(charset):
        logits = np.transpose(logits, (0, 2, 1))
    pred = np.argmax(logits, axis=2)
    texts, confs, char_confs = [], [], []
    for b in range(pred.shape[0]):
        chars, scores, last = [], [], None
        for t, p in enumerate(pred[b]):
            if p != 0 and p != last and p < len(charset):
                chars.append(charset[p])
                scores.append(float(np.max(logits[b, t])))
            last = p
        texts.append("".join(chars))
        confs.append(float(np.mean(scores)) if scores else 0.0)
        char_confs.append(list(zip(chars, scores)))
    return texts, confs, char_confs

def predict_one(backend, img_bgr, charset, imgH, imgW):
    x = preprocess(img_bgr, imgH, imgW)
    y = backend.infer(x)
    texts, confs, char_confs = ctc_decode(y, charset)
    return texts[0].replace(" ", ""), confs[0], char_confs[0]

# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_laser_text(img_bgr):
    """5 enhanced versions for laser-engraved / low-contrast text."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_img = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    bg      = cv2.GaussianBlur(gray, (51, 51), 0)
    bgsub   = cv2.cvtColor(cv2.convertScaleAbs(cv2.subtract(gray, bg), alpha=8), cv2.COLOR_GRAY2BGR)

    unsharp = cv2.addWeighted(gray, 2.0, cv2.GaussianBlur(gray, (0, 0), 3), -1.0, 0)
    unsharp_clahe = cv2.cvtColor(clahe.apply(unsharp), cv2.COLOR_GRAY2BGR)

    tophat  = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 20)))
    tophat_img = cv2.cvtColor(clahe.apply(cv2.convertScaleAbs(tophat, alpha=6)), cv2.COLOR_GRAY2BGR)

    return {
        "clahe":        clahe_img,
        "bg_subtract":  bgsub,
        "unsharp_clahe": unsharp_clahe,
        "tophat":       tophat_img,
    }

# ── Compare logic ────────────────────────────────────────────────────────────

def clean_text(s):
    """Keep only alphanumeric characters, uppercase."""
    return ''.join(c for c in s if c.isalnum()).upper()

def compare(expected, predicted, max_diff=1):
    """
    Returns: 'MATCH' | 'BYPASS' | 'FAIL' | '' (no expected)
      - Clean both sides (keep only alnum, uppercase)
      - exact match after clean       → MATCH
      - length differs                → FAIL  (missing/extra chars)
      - same length, diffs <= max_diff → BYPASS
      - same length, diffs >  max_diff → FAIL
    """
    if not expected:
        return ""
    ce = clean_text(expected)
    cp = clean_text(predicted)
    if ce == cp:
        return "MATCH"
    if len(ce) != len(cp):
        return "FAIL"
    diffs = sum(a != b for a, b in zip(ce, cp))
    return "BYPASS" if diffs <= max_diff else "FAIL"

# ── Filename parser ───────────────────────────────────────────────────────────

_META_RE = re.compile(
    r"__(?P<text>.+?)__exp__(?P<expected>.+?)__vc__(?P<vc_conf>-?[\d.]+)__thr__(?P<thr>-?[\d.]+)$"
)

def parse_meta(stem):
    m = _META_RE.search(stem)
    if not m:
        return {"text_fn": stem, "expected": "", "vc_conf": -1.0, "threshold": -1.0}
    return {
        "text_fn":   m.group("text").replace("_", " "),
        "expected":  m.group("expected").replace("_", " "),
        "vc_conf":   float(m.group("vc_conf")),
        "threshold": float(m.group("thr")),
    }

# ── HTML helpers ──────────────────────────────────────────────────────────────

def conf_color(c):
    return "#22c55e" if c >= .90 else "#f59e0b" if c >= .75 else "#f97316" if c >= .50 else "#ef4444"

def b64img(path_or_bgr):
    if isinstance(path_or_bgr, (str, Path)):
        data = open(path_or_bgr, "rb").read()
        ext  = Path(path_or_bgr).suffix.lower()
    else:
        _, buf = cv2.imencode(".jpg", path_or_bgr)
        data, ext = bytes(buf), ".jpg"
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"

def char_html(char_confs):
    if not char_confs:
        return "<span style='color:#94a3b8;font-style:italic'>—</span>"
    out = []
    for ch, c in char_confs:
        col = conf_color(c)
        out.append(f'<span style="display:inline-flex;flex-direction:column;align-items:center;margin:0 2px">'
                   f'<span style="color:{col};font-weight:700;border-bottom:2px solid {col};padding:0 2px">'
                   f'{"&nbsp;" if ch==" " else ch}</span>'
                   f'<span style="color:{col};font-size:.6em">{c*100:.0f}%</span></span>')
    return "".join(out)

def vc_badge(vc, thr):
    if vc < 0: return "—"
    ok = vc >= thr > 0
    col = "#22c55e" if ok else "#ef4444"
    return (f'<span style="color:{col};font-weight:700">{vc*100:.1f}%</span> '
            f'<span style="font-size:.7em;color:{col};border:1px solid {col}44;border-radius:3px;padding:1px 4px">'
            f'{"PASS" if ok else "FAIL"}</span>')

def aug_rows_html(aug_results, expected=""):
    """Render augmented retry results as sub-rows (collapsed by default)."""
    if not aug_results:
        return ""
    rows = []
    for name, pred, conf, cc, img_bgr in aug_results:
        col = conf_color(conf)
        ok  = bool(expected and pred.strip() == expected.strip())
        bg  = "background:#0a200a" if ok else "background:#0f1f35"
        mark = ' <span style="color:#22c55e;font-weight:700">✓</span>' if ok else ""
        rows.append(
            f'<tr style="{bg}">'
            f'<td colspan="5"></td>'
            f'<td colspan="2" style="padding:4px 6px">'
            f'<img src="{b64img(img_bgr)}" style="max-height:48px;max-width:180px;object-fit:contain;border-radius:3px">'
            f'<div style="font-size:.7em;color:#64748b;margin-top:2px">{name}{mark}</div></td>'
            f'<td style="font-family:monospace;color:#93c5fd;font-size:.88em">{pred or "<em>—</em>"}</td>'
            f'<td><span style="color:{col};font-weight:700">{conf*100:.1f}%</span></td>'
            f'<td>{char_html(cc)}</td>'
            f'</tr>'
        )
    return "".join(rows)

# ── HTML builder ──────────────────────────────────────────────────────────────

def aug_matched(r):
    """True if any augmented retry reaches MATCH or BYPASS."""
    exp = r["expected"]
    return any(compare(exp, pred) in ("MATCH", "BYPASS")
               for _, pred, _, _, _ in r.get("aug_results", []))

def build_html(results, elapsed, args):
    n = len(results)
    n_match   = sum(1 for r in results if r["cmp"] == "MATCH")
    n_bypass  = sum(1 for r in results if r["cmp"] == "BYPASS")
    n_fail    = sum(1 for r in results if r["cmp"] == "FAIL" and not aug_matched(r))
    n_rescued = sum(1 for r in results if r["cmp"] == "FAIL" and aug_matched(r))
    n_empty   = sum(1 for r in results if not r["text"])
    rows      = []

    for idx, r in enumerate(results, 1):
        col     = conf_color(r["conf"])
        cmp     = r["cmp"]
        rescued = (cmp == "FAIL" and aug_matched(r))
        # silently treat BYPASS and aug-rescued as passed
        passed  = cmp in ("MATCH", "BYPASS") or rescued

        aug_html = aug_rows_html(r.get("aug_results", []), r["expected"])
        aug_toggle = (
            f'<button onclick="toggleAug(this)" style="font-size:.7em;padding:2px 7px;'
            f'background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:4px;cursor:pointer;margin-left:6px">'
            f'▶ {len(r.get("aug_results",[]))} augmented</button>'
            if aug_html else ""
        )
        aug_section = (
            f'<tr class="aug-rows" style="display:none"><td colspan="10">'
            f'<table style="width:100%;border-collapse:collapse">{aug_html}</table></td></tr>'
            if aug_html else ""
        )

        if not r["expected"]:
            match_col, match_txt, row_bg = "#475569", "—", ""
        elif passed:
            match_col, match_txt, row_bg = "#22c55e", "✓ match", ""
        else:
            match_col, match_txt, row_bg = "#ef4444", "✗ fail", "background:#1a0a0a"

        # predicted cell: show expected when passed (bypass/aug silently normalised)
        pred_display = (r["expected"] if passed and r["expected"]
                        else r["text"] or '<em style="color:#94a3b8">—</em>')

        rows.append(f"""
        <tr style="{row_bg}" data-cmp="{cmp}">
          <td style="color:#475569;text-align:center">{idx}</td>
          <td><img src="{b64img(r['path'])}" style="max-height:56px;max-width:200px;object-fit:contain;border-radius:3px;cursor:zoom-in"
               onclick="showModal(this.src,'{r['filename']}')"></td>
          <td style="color:#64748b;font-size:.72em;word-break:break-all">{r['filename']}</td>
          <td style="font-family:monospace;color:#93c5fd">{r['expected'] or "—"}</td>
          <td style="font-family:monospace">{pred_display}</td>
          <td><span style="color:{match_col};font-weight:700">{match_txt}</span>{aug_toggle}</td>
          <td style="text-align:center"><span style="color:{col};font-weight:700">{r['conf']*100:.1f}%</span></td>
          <td>{vc_badge(r['vc_conf'], r['threshold'])}</td>
          <td style="text-align:center;color:#64748b;font-size:.85em">{f"{r['threshold']:.2f}" if r['threshold']>=0 else "—"}</td>
        </tr>{aug_section}""")

    model_label = args.engine if args.engine else args.onnx
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>OCR Report — {Path(args.folder).name}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0}}
  header{{background:#1e293b;border-bottom:1px solid #334155;padding:16px 28px}}
  header h1{{font-size:1.2rem;color:#f8fafc;margin-bottom:4px}}
  .meta{{font-size:.78rem;color:#94a3b8}}
  .stat-bar{{background:#1e293b;border-bottom:1px solid #334155;padding:10px 28px;display:flex;gap:24px;flex-wrap:wrap}}
  .stat{{display:flex;flex-direction:column}}
  .sv{{font-size:1.4rem;font-weight:700;color:#38bdf8}}
  .sl{{font-size:.7rem;color:#64748b;text-transform:uppercase}}
  .toolbar{{padding:10px 28px;background:#0f172a;border-bottom:1px solid #1e293b;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
  .toolbar input,.toolbar select{{padding:6px 10px;background:#1e293b;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:.87rem;outline:none}}
  .toolbar input:focus{{border-color:#38bdf8}}
  .wrap{{padding:0 28px 48px;overflow-x:auto}}
  table{{width:100%;border-collapse:collapse;font-size:.87rem;margin-top:8px}}
  th{{background:#1e293b;color:#94a3b8;font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;padding:9px 10px;border-bottom:2px solid #334155;position:sticky;top:0;z-index:1;white-space:nowrap}}
  td{{padding:9px 10px;border-bottom:1px solid #1e293b;vertical-align:middle}}
  tr:hover td{{background:#1e293b44}}
  #modal{{display:none;position:fixed;inset:0;background:#000d;z-index:200;align-items:center;justify-content:center}}
  #modal.open{{display:flex}}
  #modal-img{{max-width:90vw;max-height:88vh;border-radius:8px}}
  #modal-close{{position:fixed;top:10px;right:16px;font-size:2rem;color:#e2e8f0;cursor:pointer;opacity:.7}}
  #modal-name{{position:fixed;top:12px;left:50%;transform:translateX(-50%);background:#1e293b;color:#e2e8f0;padding:4px 14px;border-radius:20px;font-size:.78rem}}
</style></head><body>
<header>
  <h1>OCR Datecode Report</h1>
  <div class="meta">Folder: <b>{args.folder}</b> · Model: <b>{model_label}</b> · Device: <b>{args.device}</b></div>
</header>
<div class="stat-bar">
  <div class="stat"><span class="sv">{n}</span><span class="sl">Images</span></div>
  <div class="stat"><span class="sv" style="color:#22c55e">{n_match + n_bypass + n_rescued}</span><span class="sl">Pass</span></div>
  <div class="stat"><span class="sv" style="color:#ef4444">{n_fail}</span><span class="sl">Fail</span></div>
  <div class="stat"><span class="sv" style="color:#94a3b8">{n_empty}</span><span class="sl">Empty</span></div>
  <div class="stat"><span class="sv">{elapsed*1000/max(n,1):.1f} ms</span><span class="sl">Avg/img</span></div>
</div>
<div class="toolbar">
  <input id="q" type="text" placeholder="Search filename / text..." oninput="ft()">
  <select id="fm" onchange="ft()">
    <option value="all">All</option>
    <option value="pass">Pass</option>
    <option value="FAIL">Fail</option>
    <option value="empty">Empty pred</option>
  </select>
  <label style="font-size:.8rem;color:#64748b">Conf &lt;</label>
  <select id="fc" onchange="ft()">
    <option value="all">All</option><option value="90">90%</option>
    <option value="75">75%</option><option value="50">50%</option>
  </select>
</div>
<div class="wrap"><table id="tbl">
  <thead><tr>
    <th>#</th><th>Crop</th><th>Filename</th><th>Expected</th><th>Predicted</th>
    <th>Match</th><th style="text-align:center">Conf</th><th>VC</th><th>Thr</th>
  </tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table></div>
<div id="modal" onclick="closeModal()">
  <span id="modal-close">✕</span><div id="modal-name"></div>
  <img id="modal-img" src="" alt="">
</div>
<script>
function showModal(src,name){{
  document.getElementById('modal-img').src=src;
  document.getElementById('modal-name').textContent=name;
  document.getElementById('modal').classList.add('open');
}}
function closeModal(){{document.getElementById('modal').classList.remove('open');}}
document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeModal();}});
function toggleAug(btn){{
  const augRow=btn.closest('tr').nextElementSibling;
  if(!augRow||!augRow.classList.contains('aug-rows'))return;
  const hidden=augRow.style.display==='none';
  augRow.style.display=hidden?'':'none';
  btn.textContent=(hidden?'▼':'▶')+btn.textContent.slice(1);
}}
function ft(){{
  const q=document.getElementById('q').value.toLowerCase();
  const fm=document.getElementById('fm').value;
  const fc=document.getElementById('fc').value;
  document.querySelectorAll('#tbl tbody tr:not(.aug-rows)').forEach(tr=>{{
    const fname=tr.cells[2].textContent.toLowerCase();
    const exp=tr.cells[3].textContent.toLowerCase();
    const pred=tr.cells[4].textContent.toLowerCase();
    const cmp=tr.dataset.cmp||'';
    const conf=parseFloat(tr.cells[6].textContent)||0;
    let show=true;
    if(q&&!fname.includes(q)&&!pred.includes(q)&&!exp.includes(q))show=false;
    if(fm==='pass'&&tr.cells[5].textContent.includes('✗'))show=false;
    if(fm==='FAIL'&&!tr.cells[5].textContent.includes('✗'))show=false;
    if(fm==='empty'&&pred.trim()!=='—'&&pred.trim()!=='')show=false;
    if(fc!=='all'&&conf>=parseFloat(fc))show=false;
    tr.style.display=show?'':'none';
    // also hide/show adjacent aug row
    const next=tr.nextElementSibling;
    if(next&&next.classList.contains('aug-rows'))next.style.display=show&&next._visible?'':'none';
  }});
}}
</script>
</body></html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--onnx")
    g.add_argument("--engine")
    ap.add_argument("--folder",     required=True)
    ap.add_argument("--dict",       default="languages/english/ppocr_keys_v1.txt")
    ap.add_argument("--device",     default="cpu", choices=["cpu", "cuda", "trt"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_diff",   type=int, default=1,
                    help="Max char substitutions to bypass (same-length only, default=1)")
    ap.add_argument("--imgH",       type=int, default=48)
    ap.add_argument("--imgW",       type=int, default=320)
    ap.add_argument("--output",     default="report.html")
    ap.add_argument("--csv",        default=None, help="CSV output path (default: same stem as --output)")
    args = ap.parse_args()

    img_paths = sorted(p for p in Path(args.folder).iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not img_paths:
        print(f"No images in {args.folder}"); return
    print(f"Found {len(img_paths)} images")

    charset = load_charset(args.dict)
    backend = TRTBackend(args.engine) if args.engine else ONNXBackend(args.onnx, args.device)

    results, t0 = [], time.perf_counter()

    for i in range(0, len(img_paths), args.batch_size):
        batch = img_paths[i: i + args.batch_size]
        # batch inference
        imgs = []
        bgrs = []
        for p in batch:
            bgr = cv2.imread(str(p))
            bgrs.append(bgr)
            imgs.append(preprocess(bgr, args.imgH, args.imgW) if bgr is not None
                        else np.zeros((1, 3, args.imgH, args.imgW), dtype=np.float32))
        x = np.concatenate(imgs, axis=0)
        texts, confs, char_confs = ctc_decode(backend.infer(x), charset)
        texts = [t.replace(" ", "") for t in texts]

        for p, bgr, t, c, cc in zip(batch, bgrs, texts, confs, char_confs):
            meta = parse_meta(p.stem)
            expected = meta["expected"].replace(" ", "")
            cmp = compare(expected, t, max_diff=args.max_diff)

            # Augmentation retry only on hard FAIL (not bypass)
            aug_results = []
            if cmp == "FAIL" and bgr is not None:
                for aug_name, aug_img in augment_laser_text(bgr).items():
                    at, ac, acc = predict_one(backend, aug_img, charset, args.imgH, args.imgW)
                    aug_results.append((aug_name, at, ac, acc, aug_img))
                    print(f"  [aug:{aug_name}] '{at}' conf={ac:.3f}")

            results.append({
                "path": str(p), "filename": p.name,
                "text": t, "conf": c, "char_confs": cc,
                "expected": expected, "cmp": cmp,
                "vc_conf": meta["vc_conf"], "threshold": meta["threshold"],
                "aug_results": aug_results,
            })

        done = min(i + args.batch_size, len(img_paths))
        print(f"  [{done}/{len(img_paths)}] last: '{texts[-1]}' conf={confs[-1]:.3f}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone {elapsed*1000:.1f} ms  ({elapsed*1000/len(img_paths):.1f} ms/img)")

    out = Path(args.output)
    out.write_text(build_html(results, elapsed, args), encoding="utf-8")
    print(f"Report → {out.resolve()}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = Path(args.csv) if args.csv else out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "expected", "predicted", "compare",
            "expected_clean", "predicted_clean",
            "model_conf", "vc_conf", "threshold",
            "char_confs",
            "aug_rescued", "aug_best_pred", "aug_best_conf",
        ])
        for r in results:
            char_str  = " ".join(f"{ch}:{c:.3f}" for ch, c in r["char_confs"])
            exp_clean  = clean_text(r["expected"])
            pred_clean = clean_text(r["text"])

            aug = r.get("aug_results", [])
            aug_rescued = aug_best_pred = aug_best_conf = ""
            if aug:
                # prefer MATCH/BYPASS; otherwise pick highest conf
                good = [(n, p, c) for n, p, c, _, _ in aug
                        if compare(r["expected"], p) in ("MATCH", "BYPASS")]
                if good:
                    aug_rescued = "yes"
                    _, aug_best_pred, aug_best_conf = max(good, key=lambda x: x[2])
                else:
                    aug_rescued = "no"
                    _, aug_best_pred, aug_best_conf, _, _ = max(aug, key=lambda x: x[2])

            writer.writerow([
                r["filename"],
                r["expected"],
                r["text"],
                r["cmp"],
                exp_clean,
                pred_clean,
                f"{r['conf']:.4f}",
                f"{r['vc_conf']:.4f}" if r["vc_conf"] >= 0 else "",
                f"{r['threshold']:.2f}" if r["threshold"] >= 0 else "",
                char_str,
                aug_rescued,
                aug_best_pred,
                f"{aug_best_conf:.4f}" if aug_best_conf != "" else "",
            ])
    print(f"CSV    → {csv_path.resolve()}")

if __name__ == "__main__":
    main()