"""
OCR Label Tool — web app đơn giản để VERIFY + điền text đúng cho các crop OCR,
rồi EXPORT sang format training (giống data_ocr: train/ test/ + rec_gt_*.txt).

Chạy:
    python backend/scripts/ocr_label_tool.py --dataset ./ocr_training_data --port 8005

Tunnel ra ngoài (pinggy):
    ssh -p 443 -R0:localhost:8005 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 <token>@pro.pinggy.io

Dataset: thư mục chứa *.jpg (đệ quy). Tên file do crop_ocr_training_data.py sinh ra
mã hoá sẵn exp-/got- để prefill. Nhãn lưu vào <dataset>/_labels.json — chỉ lưu khi
bạn bấm OK (verify xong). Chọn lại ảnh trong gallery để sửa.

Export → <export_dir>/{train,test}/*.jpg + rec_gt_train.txt / rec_gt_test.txt
(dòng: "train/<file>.jpg\\t<label>"). Item chưa set split → tự chia theo ratio.
"""

import argparse
import json
import re
import shutil
import zlib
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# ─── Filename parse ─────────────────────────────────────────────────────────────
FN_RE = re.compile(
    r"^(?P<tag>pass|fail)__(?P<recipe>[^_]+)__(?P<rid>[^_]+)__(?P<cam>[^_]+)"
    r"__f(?P<fi>\d+)__ann(?P<ann>\d+)__exp-(?P<exp>.*?)__got-(?P<got>.*?)"
    r"__c(?P<conf>[-\d.]+)\.jpg$"
)


def parse_name(basename: str) -> dict:
    m = FN_RE.match(basename)
    if not m:
        return {"tag": "?", "recipe": "?", "expected": "", "recognized": "", "conf": None}
    d = m.groupdict()
    return {
        "tag": d["tag"],
        "recipe": d["recipe"],
        "cam": d["cam"],
        "ann": d["ann"],
        "expected": d["exp"].replace("_", " ").strip(),
        "recognized": d["got"].replace("_", " ").strip(),
        "conf": float(d["conf"]) if d["conf"] not in ("", "-1.000") else None,
    }


# ─── State ──────────────────────────────────────────────────────────────────────
class LabelTool:
    def __init__(self, dataset: Path):
        self.dataset = dataset
        self.labels_path = dataset / "_labels.json"
        self.labels: dict = {}
        if self.labels_path.exists():
            self.labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
        self.items = []
        for p in sorted(dataset.rglob("*.jpg")):
            rel = str(p.relative_to(dataset))
            meta = parse_name(p.name)
            folder = str(Path(rel).parent)   # vd "fail/6a31ec96..."
            self.items.append({"path": rel, "folder": folder, **meta})

    def save_labels(self):
        self.labels_path.write_text(
            json.dumps(self.labels, ensure_ascii=False, indent=0), encoding="utf-8"
        )

    def list_items(self):
        out = []
        for it in self.items:
            lab = self.labels.get(it["path"], {})
            out.append({
                **it,
                "label": lab.get("label", ""),       # rỗng nếu chưa verify
                "split": lab.get("split", "auto"),
                "status": lab.get("status", "todo"),
            })
        return out


TOOL: LabelTool = None
EXPORT_DIR = Path("./data_ocr_export")
app = FastAPI(title="OCR Label Tool")


class SaveReq(BaseModel):
    path: str
    label: str
    split: str = "auto"
    status: str = "done"


class ExportReq(BaseModel):
    export_dir: str = None
    ratio: float = 0.85


@app.get("/api/list")
def api_list():
    items = TOOL.list_items()
    folders = sorted({i["folder"] for i in items})
    done = sum(1 for i in items if i["status"] == "done")
    skip = sum(1 for i in items if i["status"] == "skip")
    return {"total": len(items), "done": done, "skip": skip, "folders": folders, "items": items}


@app.get("/img")
def img(path: str):
    f = (TOOL.dataset / path).resolve()
    if TOOL.dataset.resolve() not in f.parents or not f.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(f))


@app.post("/api/save")
def api_save(req: SaveReq):
    TOOL.labels[req.path] = {"label": req.label.strip(), "split": req.split, "status": req.status}
    TOOL.save_labels()
    return {"ok": True}


def _auto_split(name: str, ratio: float) -> str:
    h = zlib.crc32(name.encode("utf-8")) % 1000 / 1000.0
    return "train" if h < ratio else "test"


@app.post("/api/export")
def api_export(req: ExportReq):
    out = Path(req.export_dir) if req.export_dir else EXPORT_DIR
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    lines = {"train": [], "test": []}
    counts = {"train": 0, "test": 0, "skipped_no_label": 0}
    for path, lab in TOOL.labels.items():
        if lab.get("status") != "done" or not lab.get("label", "").strip():
            counts["skipped_no_label"] += 1
            continue
        src = TOOL.dataset / path
        if not src.exists():
            continue
        name = Path(path).name
        split = lab.get("split", "auto")
        if split not in ("train", "test"):
            split = _auto_split(name, req.ratio)
        shutil.copy(str(src), str(out / split / name))
        lines[split].append(f"{split}/{name}\t{lab['label'].strip()}")
        counts[split] += 1
    (out / "rec_gt_train.txt").write_text("\n".join(lines["train"]) + ("\n" if lines["train"] else ""), encoding="utf-8")
    (out / "rec_gt_test.txt").write_text("\n".join(lines["test"]) + ("\n" if lines["test"] else ""), encoding="utf-8")
    return {"ok": True, "output": str(out.resolve()), **counts}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR Label Tool</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--border:#e3e6ea;--text:#1f2733;--muted:#6b7280;
--accent:#2563eb;--accent2:#1d4ed8;--ok:#16a34a;--skip:#d97706;--shadow:0 1px 3px rgba(0,0,0,.08)}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;font-family:system-ui,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--text);display:flex;flex-direction:column}
header{background:var(--card);border-bottom:1px solid var(--border);padding:9px 16px;
display:flex;align-items:center;gap:14px;box-shadow:var(--shadow);z-index:5}
header h1{font-size:15px;margin:0;font-weight:600;white-space:nowrap}
.prog{color:var(--muted);font-size:13px;white-space:nowrap}
.bar{height:8px;background:#eceef1;border-radius:6px;overflow:hidden;width:200px}
.bar>i{display:block;height:100%;background:var(--ok);transition:.2s}
.spacer{flex:1}
select,input[type=number]{font-size:13px;padding:7px 9px;border:1px solid var(--border);border-radius:8px;background:#fff}
button{font-size:13.5px;padding:9px 14px;border-radius:9px;border:1px solid var(--border);
background:var(--card);cursor:pointer;font-weight:500}
button:hover{background:#f0f2f5}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
button.primary:hover{background:var(--accent2)}
button.ok{background:var(--ok);color:#fff;border-color:var(--ok)}
button.skip{color:var(--skip)}
main{flex:1;display:flex;min-height:0}
/* ── left gallery ── */
aside{width:320px;min-width:320px;border-right:1px solid var(--border);background:var(--card);
display:flex;flex-direction:column;min-height:0}
.filters{padding:10px;border-bottom:1px solid var(--border);display:flex;flex-direction:column;gap:8px}
.filters .r{display:flex;gap:8px}.filters select{flex:1}
.gallery{overflow-y:auto;padding:8px;display:grid;grid-template-columns:1fr 1fr;gap:8px;align-content:start}
.thumb{position:relative;border:2px solid var(--border);border-radius:8px;overflow:hidden;cursor:pointer;background:#11151c}
.thumb.sel{border-color:var(--accent);box-shadow:0 0 0 2px rgba(37,99,235,.25)}
.thumb img{width:100%;height:46px;object-fit:contain;display:block;image-rendering:pixelated}
.thumb .cap{font-size:10px;color:#cbd5e1;padding:2px 4px;background:#0d1117;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.badge{position:absolute;top:3px;right:3px;font-size:10px;font-weight:700;padding:1px 6px;border-radius:10px;color:#fff}
.badge.done{background:var(--ok)}.badge.skip{background:var(--skip)}
.tagdot{position:absolute;top:3px;left:3px;font-size:9px;font-weight:700;padding:1px 5px;border-radius:8px;color:#fff}
.tagdot.fail{background:#dc2626}.tagdot.pass{background:#059669}
/* ── right editor ── */
section{flex:1;overflow-y:auto;padding:22px;min-width:0}
.card{max-width:760px;margin:0 auto;background:var(--card);border:1px solid var(--border);
border-radius:12px;box-shadow:var(--shadow);padding:20px}
.imgbox{background:#11151c;border-radius:8px;padding:18px;text-align:center;margin-bottom:8px;overflow:auto}
.imgbox img{max-width:100%;image-rendering:pixelated;transform-origin:center}
.meta{font-size:12.5px;color:var(--muted);display:flex;gap:16px;flex-wrap:wrap;margin:8px 2px 14px}
.meta b{color:var(--text);font-weight:600}
label.fld{display:block;font-size:12px;color:var(--muted);margin-bottom:6px}
input#label{width:100%;font-size:22px;padding:12px 14px;border:2px solid var(--border);
border-radius:9px;font-family:ui-monospace,Menlo,monospace;letter-spacing:.5px}
input#label:focus{outline:none;border-color:var(--accent)}
.row{display:flex;gap:10px;align-items:center;margin-top:14px;flex-wrap:wrap}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:9px;overflow:hidden}
.seg button{border:0;border-radius:0;padding:8px 14px}.seg button.on{background:var(--accent);color:#fff}
.exp{font-family:ui-monospace,monospace}
.hint{font-size:12px;color:var(--muted);margin-top:6px}
.export{background:#ecfdf5;border:1px solid #a7f3d0;border-radius:10px;padding:12px 14px;margin-top:18px;font-size:13px}
.export button{background:var(--ok);color:#fff;border-color:var(--ok)}
.empty{color:var(--muted);text-align:center;padding:40px}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1f2733;
color:#fff;padding:10px 16px;border-radius:8px;font-size:13px;opacity:0;transition:.2s;pointer-events:none}
.toast.show{opacity:.95}
.zoom{display:flex;gap:6px;align-items:center;justify-content:center;margin-top:6px;font-size:12px;color:var(--muted)}
</style></head><body>
<header>
  <h1>🏷️ OCR Label</h1>
  <div class="bar"><i id="pbar"></i></div>
  <div class="prog" id="prog">—</div>
  <div class="spacer"></div>
  <span style="font-size:12px;color:var(--muted)">train ratio</span>
  <input id="ratio" type="number" value="0.85" step="0.05" min="0" max="1" style="width:62px">
  <button class="ok" onclick="doExport()">⬇ Export</button>
  <span id="expmsg" style="font-size:12px;color:var(--muted)"></span>
</header>
<main>
  <aside>
    <div class="filters">
      <div class="r"><select id="ffolder" onchange="applyFilter()"></select></div>
      <div class="r">
        <select id="fstatus" onchange="applyFilter()">
          <option value="all">Tất cả trạng thái</option>
          <option value="todo">Chưa verify</option>
          <option value="done">Đã verify (done)</option>
          <option value="skip">Skip</option>
        </select>
      </div>
      <div style="font-size:12px;color:var(--muted)" id="cnt"></div>
    </div>
    <div class="gallery" id="gallery"></div>
  </aside>
  <section>
    <div class="card" id="editor"><div class="empty">Chọn 1 ảnh ở danh sách bên trái để verify</div></div>
  </section>
</main>
<div class="toast" id="toast"></div>
<script>
let ALL=[], view=[], curPath=null, split="auto", zoom=3;
const $=id=>document.getElementById(id);
function toast(t){const e=$('toast');e.textContent=t;e.classList.add('show');setTimeout(()=>e.classList.remove('show'),1300)}
function short(p){return p.split('/').pop().replace(/__exp-.*/,'').replace(/^(pass|fail)__[^_]+__/,'')}

async function load(){const r=await fetch('/api/list');const d=await r.json();ALL=d.items;
  const sel=$('ffolder');sel.innerHTML='<option value="all">📁 Tất cả folder ('+d.total+')</option>'+
    d.folders.map(f=>`<option value="${f}">${f} (${ALL.filter(x=>x.folder===f).length})</option>`).join('');
  updateProg();applyFilter();}
function updateProg(){const done=ALL.filter(x=>x.status==='done').length,skip=ALL.filter(x=>x.status==='skip').length;
  $('prog').textContent=`${done}/${ALL.length} done · ${skip} skip`;
  $('pbar').style.width=(ALL.length?100*done/ALL.length:0)+'%';}
function applyFilter(){const ff=$('ffolder').value,fs=$('fstatus').value;
  view=ALL.filter(x=>(ff==='all'||x.folder===ff)&&(fs==='all'||x.status===fs));
  $('cnt').textContent=`${view.length} ảnh`;
  renderGallery();
  if(view.length&&!view.find(x=>x.path===curPath))select(view[0].path);
  else if(!view.length){curPath=null;$('editor').innerHTML='<div class="empty">Không có ảnh khớp bộ lọc</div>'}}
function renderGallery(){const g=$('gallery');g.innerHTML='';
  view.forEach(it=>{const d=document.createElement('div');d.className='thumb'+(it.path===curPath?' sel':'');
    d.id='th_'+btoa(unescape(encodeURIComponent(it.path))).replace(/=/g,'');
    let badge=it.status==='done'?'<span class="badge done">✓ done</span>':
              it.status==='skip'?'<span class="badge skip">skip</span>':'';
    d.innerHTML=`<span class="tagdot ${it.tag}">${it.tag}</span>${badge}`+
      `<img loading="lazy" src="/img?path=${encodeURIComponent(it.path)}">`+
      `<div class="cap">${short(it.path)}</div>`;
    d.onclick=()=>select(it.path);g.appendChild(d);});}
function thumbId(p){return 'th_'+btoa(unescape(encodeURIComponent(p))).replace(/=/g,'')}
function select(path){curPath=path;const it=ALL.find(x=>x.path===path);if(!it)return;
  document.querySelectorAll('.thumb').forEach(t=>t.classList.remove('sel'));
  const td=$(thumbId(path));if(td){td.classList.add('sel');td.scrollIntoView({block:'nearest'})}
  split=it.split||'auto';
  const labVal=it.label||it.expected||'';
  $('editor').innerHTML=`
    <div class="imgbox"><img id="img" src="/img?path=${encodeURIComponent(path)}" style="transform:scale(${zoom})"></div>
    <div class="zoom">zoom <button onclick="setZoom(-1)">−</button><span id="zlbl">${zoom}×</span><button onclick="setZoom(1)">+</button></div>
    <div class="meta" style="margin-top:12px">
      <span>loại: <b>${it.tag}</b></span><span>recipe: <b>${it.recipe}</b></span>
      <span>ann: <b>${it.ann||'?'}</b></span>
      <span>OCR đọc: <b class="exp">${it.recognized||'∅'}</b></span>
      <span>conf: <b>${it.conf==null?'—':it.conf.toFixed(3)}</b></span>
      <span>trạng thái: <b>${it.status}</b></span>
    </div>
    <label class="fld">Text ĐÚNG (verify & sửa) — Enter = OK & ảnh kế</label>
    <input id="label" autocomplete="off" autocapitalize="off" spellcheck="false" value="${(labVal).replace(/"/g,'&quot;')}">
    <div class="hint">Hệ thống đoán (expected): <span class="exp">${it.expected||'∅'}</span></div>
    <div class="row">
      <button class="ok" onclick="saveNext()">✓ OK & Lưu (Enter)</button>
      <button class="skip" onclick="skip()">Bỏ qua</button>
      <button onclick="nav(-1)">← Trước</button>
      <button onclick="nav(1)">Sau →</button>
      <div class="spacer"></div>
      <div class="seg" id="seg">
        ${['auto','train','test'].map(s=>`<button class="${split===s?'on':''}" onclick="setSplit('${s}')">${s}</button>`).join('')}
      </div>
    </div>`;
  const inp=$('label');inp.focus();inp.select();
  inp.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();saveNext()}});}
function setZoom(d){zoom=Math.max(1,Math.min(8,zoom+d));const im=$('img');if(im)im.style.transform=`scale(${zoom})`;$('zlbl').textContent=zoom+'×'}
function setSplit(s){split=s;document.querySelectorAll('#seg button').forEach(b=>b.classList.toggle('on',b.textContent===s))}
async function save(status){const it=ALL.find(x=>x.path===curPath);if(!it)return;
  const label=$('label')?$('label').value:'';
  await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:curPath,label,split,status})});
  it.label=label;it.split=split;it.status=status;
  updateProg();
  // cập nhật badge thumbnail tại chỗ
  const td=$(thumbId(curPath));if(td){td.querySelector('.badge')?.remove();
    if(status!=='todo'){const b=document.createElement('span');b.className='badge '+status;
      b.textContent=status==='done'?'✓ done':'skip';td.appendChild(b)}}}
function curIdx(){return view.findIndex(x=>x.path===curPath)}
async function saveNext(){await save('done');nav(1)}
async function skip(){await save('skip');nav(1)}
function nav(d){const j=curIdx()+d;if(j>=0&&j<view.length)select(view[j].path);else toast(d>0?'Cuối danh sách':'Đầu danh sách')}
async function doExport(){$('expmsg').textContent=' …đang export';
  const r=await fetch('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ratio:parseFloat($('ratio').value)||0.85})});const d=await r.json();
  $('expmsg').textContent=` ✓ ${d.train} train + ${d.test} test`;toast(`Export: ${d.train}+${d.test} → ${d.output}`)}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
  if(e.key==='ArrowRight')nav(1);if(e.key==='ArrowLeft')nav(-1)});
load();
</script></body></html>"""


def main():
    global TOOL, EXPORT_DIR
    ap = argparse.ArgumentParser(description="OCR labeling web tool + export training format")
    ap.add_argument("--dataset", default="./ocr_training_data", help="Thư mục chứa crop *.jpg")
    ap.add_argument("--export_dir", default="./data_ocr_export", help="Thư mục export training")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8005, help="Port")
    args = ap.parse_args()

    ds = Path(args.dataset)
    if not ds.exists():
        raise SystemExit(f"Dataset không tồn tại: {ds}")
    TOOL = LabelTool(ds)
    EXPORT_DIR = Path(args.export_dir)
    print(f"Dataset: {ds.resolve()} | {len(TOOL.items)} ảnh | đã label: {len(TOOL.labels)}")
    print(f"Export dir: {EXPORT_DIR.resolve()}")
    print(f"Mở: http://localhost:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
