import { useRef, useState } from 'react';
import {
  paddleOcrAPI, PaddleArtifact, PaddleEngine, PaddleEvalResult, PaddleModel, PaddlePredictResult,
} from '@/services/paddleOcrTraining';

/** Eval / Export / Test for PaddleOCR runs (ocr_paddle_service, :8003).
 *  A Paddle model has ONE CTC head, so these show one prediction per crop —
 *  the SMTR tabs' GTC/CTC split does not apply. */

interface Props {
  projectId: string;
  model: PaddleModel | null;
}

function pct(v: number | null | undefined): string {
  return v == null ? 'n/a' : `${(v * 100).toFixed(2)}%`;
}

function NoModel() {
  return <div className="at-empty-state" style={{ padding: 40 }}>
    Select a PaddleOCR run in the Train tab first.
  </div>;
}

function ModelHeader({ model }: { model: PaddleModel }) {
  return (
    <div className="at-hint" style={{ marginBottom: 8 }}>
      Run <b style={{ fontFamily: 'monospace' }}>{model.id.slice(-8)}</b> ·{' '}
      {model.arch.replace('_rec', '')} · {model.status} · from {model.base_label}
    </div>
  );
}

// ─────────────────────────────── Eval ─────────────────────────────────────

export function PaddleEvalTab({ projectId, model }: Props) {
  const [engine, setEngine] = useState<PaddleEngine>('tensorrt');
  const [result, setResult] = useState<PaddleEvalResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [onlyWrong, setOnlyWrong] = useState(true);
  const [strict, setStrict] = useState(false);

  if (!model) return <NoModel />;

  const run = async () => {
    setRunning(true); setError(null);
    try {
      setResult(await paddleOcrAPI.evaluate(projectId, model.id, engine, true));
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Evaluation failed');
    } finally {
      setRunning(false);
    }
  };

  const s = result?.scores;
  const items = (result?.items ?? []).filter(
    (i) => !onlyWrong || !(strict ? i.correct_exact : i.correct_norm));

  return (
    <>
      <ModelHeader model={model} />
      <div className="ot-toolbar">
        <label className="at-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          engine
          <select className="at-form-input" style={{ width: 120 }} value={engine}
                  onChange={(e) => setEngine(e.target.value as PaddleEngine)}>
            <option value="tensorrt">TensorRT</option>
            <option value="onnx">ONNX</option>
          </select>
        </label>
        <button className="at-btn at-btn-primary" onClick={run} disabled={running}>
          {running ? 'Scoring…' : 'Run evaluation'}
        </button>
        <span className="ot-spacer" />
        <span className="at-hint">
          TensorRT is scored by ai_services' own TextRecognizerTRT (in the vision env),
          ONNX with a copy of its preprocessing — grayscale, H=48, aspect kept, batch 1.
        </span>
      </div>

      {error && <div className="at-alert-error">{error}</div>}

      {!result && !running && (
        <div className="at-hint">
          Training already stored: engine {pct(model.metrics.acc_trt)} normalized /
          {' '}{pct(model.metrics.acc_exact_trt)} exact, ONNX {pct(model.metrics.acc_onnx)}.
          Run this to see which test images fail.
        </div>
      )}

      {s && (
        <>
          <div className="ot-prepare-grid">
            <div className="ot-stat-box">
              <div className="ot-stat-num">{pct(s.norm)}</div>
              <div className="ot-stat-label">normalized</div>
            </div>
            <div className="ot-stat-box">
              <div className="ot-stat-num">{pct(s.exact)}</div>
              <div className="ot-stat-label">exact</div>
            </div>
            <div className="ot-stat-box">
              <div className="ot-stat-num">{result!.ms_per_image} ms</div>
              <div className="ot-stat-label">per image ({result!.engine})</div>
            </div>
            <div className="ot-stat-box">
              <div className="ot-stat-num">{s.n}</div>
              <div className="ot-stat-label">test images</div>
            </div>
          </div>

          <div className="at-hint" style={{ marginBottom: 8 }}>
            <b>normalized</b> keeps letters and digits only and lowercases — close to what
            production's compare_texts() tolerates. <b>exact</b> is raw string equality,
            spaces and punctuation included. Training's own eval (best epoch{' '}
            {result!.train_metrics.best_epoch ?? 'n/a'}): {pct(result!.train_metrics.acc)}.
          </div>

          <div className="ot-toolbar">
            <label className="at-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <input type="checkbox" checked={onlyWrong} onChange={(e) => setOnlyWrong(e.target.checked)} />
              only failures
            </label>
            <label className="at-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <input type="checkbox" checked={strict} onChange={(e) => setStrict(e.target.checked)} />
              judge by exact match
            </label>
            <span className="at-hint">{items.length} of {result!.items.length} shown</span>
          </div>

          {items.map((it) => {
            const ok = strict ? it.correct_exact : it.correct_norm;
            return (
              <div key={it.id} className={`ot-label-row ${ok ? 'is-verified' : 'is-rejected'}`}
                   style={{ opacity: 1 }}>
                <div className="ot-crop-wrap">
                  {it.thumb_b64
                    ? <img className="ot-crop-img" src={`data:image/jpeg;base64,${it.thumb_b64}`} alt={it.gt_text} />
                    : <span className="at-hint">no thumbnail</span>}
                </div>
                <div className="ot-label-fields">
                  <div style={{ fontFamily: 'monospace', fontSize: 13 }}>
                    <span style={{ color: '#64748b' }}>gt </span><b>{it.gt_text}</b>
                  </div>
                  <div className="ot-hint-row">
                    <span className={it.correct_exact ? 'ot-hint-exp' : 'ot-hint-rec'}>
                      pred <b>{it.pred_text || '(empty)'}</b> ({it.confidence.toFixed(3)})
                    </span>
                  </div>
                  <div className="ot-meta-chips">
                    <span className={`ot-chip-sm ${it.correct_norm ? 'ot-pass' : 'ot-fail'}`}>
                      {it.correct_norm ? 'normalized ok' : 'normalized wrong'}
                    </span>
                    <span className={`ot-chip-sm ${it.correct_exact ? 'ot-pass' : 'ot-fail'}`}>
                      {it.correct_exact ? 'exact ok' : 'exact wrong'}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}

          {items.length === 0 && (
            <div className="at-empty-state" style={{ padding: 30 }}>No failures under this filter.</div>
          )}
        </>
      )}
    </>
  );
}

// ─────────────────────────────── Export ───────────────────────────────────

const ARTIFACTS: Array<{ id: PaddleArtifact; label: string; field: keyof PaddleModel; hint: string }> = [
  { id: 'engine', label: 'TensorRT .engine (fp16)', field: 'engine_path',
    hint: 'What a recipe loads at inference time — built by the vision env\'s TensorRT' },
  { id: 'onnx', label: 'ONNX', field: 'onnx_path',
    hint: 'Source of the engine; also the ONNX fallback if no engine exists' },
  { id: 'dict', label: 'Character dict', field: 'dict_path',
    hint: 'Must travel with the model — a decoder built from the wrong dict shifts every index' },
  { id: 'checkpoint', label: 'Checkpoint .pdparams', field: 'checkpoint_path',
    hint: 'Fine-tune onward from this (pick it as "Start from" in Train)' },
];

export function PaddleExportTab({
  projectId, model, onModelChange,
}: Props & { onModelChange: () => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  if (!model) return <NoModel />;
  const completed = model.status === 'completed';

  const act = async (label: string, fn: () => Promise<any>, describe: (r: any) => string) => {
    setBusy(label); setError(null); setMessage(null);
    try {
      const r = await fn();
      setMessage(describe(r));
      onModelChange();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || `${label} failed`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <>
      <ModelHeader model={model} />
      {error && <div className="at-alert-error">{error}</div>}
      {message && <div className="at-alert-ok">{message}</div>}

      <div className="at-hint" style={{ marginBottom: 10 }}>
        Export runs automatically at the end of training. Use these only to rebuild —
        e.g. after the TensorRT version in the vision env changes and the old engine no
        longer loads. Re-exporting ONNX deletes the engine; build it again afterwards.
      </div>

      <div className="ot-toolbar">
        <button className="at-btn at-btn-secondary" disabled={!completed || !!busy}
                onClick={() => act('Export ONNX', () => paddleOcrAPI.exportOnnx(projectId, model.id),
                  (r) => `ONNX exported — ${r.size_mb} MB, ${r.num_classes} classes (checked against Paddle)`)}>
          {busy === 'Export ONNX' ? 'Exporting…' : 'Re-export ONNX'}
        </button>
        <button className="at-btn at-btn-secondary" disabled={!completed || !model.onnx_path || !!busy}
                onClick={() => act('Build engine', () => paddleOcrAPI.exportTensorRT(projectId, model.id),
                  (r) => `Engine built — ${r.size_mb} MB in ${r.build_s} s`)}>
          {busy === 'Build engine' ? 'Building… (30 s – 4 min)' : 'Rebuild TensorRT engine'}
        </button>
        <button className="at-btn at-btn-secondary" disabled={!model.engine_path || !!busy}
                onClick={() => act('Score engine', () => paddleOcrAPI.evaluate(projectId, model.id, 'tensorrt'),
                  (r) => `Engine: ${pct(r.scores.norm)} normalized / ${pct(r.scores.exact)} exact over ${r.scores.n}`)}>
          {busy === 'Score engine' ? 'Scoring…' : 'Re-score engine'}
        </button>
      </div>

      <table style={{ borderCollapse: 'collapse', fontSize: 13, width: '100%' }}>
        <tbody>
          {ARTIFACTS.map((a) => {
            const path = model[a.field] as string | null;
            return (
              <tr key={a.id} style={{ borderBottom: '1px solid #e2e8f0' }}>
                <td style={{ padding: '8px 12px 8px 0', whiteSpace: 'nowrap' }}><b>{a.label}</b></td>
                <td style={{ padding: '8px 12px 8px 0' }}>
                  <div className="at-hint">{a.hint}</div>
                  <div style={{ fontFamily: 'monospace', fontSize: 11, color: '#64748b',
                                wordBreak: 'break-all' }}>{path || '— not exported'}</div>
                </td>
                <td style={{ padding: '8px 0', textAlign: 'right' }}>
                  <button className="at-btn at-btn-secondary at-btn-sm" disabled={!path}
                          onClick={() => paddleOcrAPI.download(projectId, model.id, a.id)
                            .catch((e) => setError(e?.message || 'Download failed'))}>
                    Download
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="at-hint" style={{ marginTop: 10 }}>
        On a recipe, pick it under <b>Paddle · project</b> in OCR Model. The recipe
        load hands ai_services the engine + dict paths; no restart needed.
      </div>
    </>
  );
}

// ─────────────────────────────── Test ─────────────────────────────────────

export function PaddleTestTab({ projectId, model }: Props) {
  const [result, setResult] = useState<PaddlePredictResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  if (!model) return <NoModel />;

  const pick = (f: File | null) => {
    setFile(f); setResult(null); setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(f ? URL.createObjectURL(f) : null);
  };

  const run = async () => {
    if (!file) return;
    setBusy(true); setError(null);
    try {
      setResult(await paddleOcrAPI.predict(projectId, model.id, file));
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Inference failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <ModelHeader model={model} />
      <div className="ot-toolbar">
        <input ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }}
               onChange={(e) => pick(e.target.files?.[0] ?? null)} />
        <button className="at-btn at-btn-secondary" onClick={() => inputRef.current?.click()}>
          Choose crop image
        </button>
        <button className="at-btn at-btn-primary" onClick={run} disabled={!file || busy || !model.onnx_path}>
          {busy ? 'Reading…' : 'Recognize'}
        </button>
        <span className="ot-spacer" />
        <span className="at-hint">
          Runs the ONNX export with ai_services' preprocessing (on CPU). On the
          test split it reads identically to the TensorRT engine.
        </span>
      </div>

      {error && <div className="at-alert-error">{error}</div>}
      {!model.onnx_path && <div className="at-hint">This run has no ONNX export yet.</div>}

      {preview && (
        <div style={{ margin: '10px 0' }}>
          <img src={preview} alt="crop" style={{ maxWidth: '100%', maxHeight: 160,
                                                  imageRendering: 'pixelated', border: '1px solid #e2e8f0' }} />
        </div>
      )}

      {result && (
        <div className="ot-prepare-grid">
          <div className="ot-stat-box" style={{ gridColumn: 'span 2' }}>
            <div className="ot-stat-num" style={{ fontFamily: 'monospace' }}>{result.text || '(empty)'}</div>
            <div className="ot-stat-label">recognized</div>
          </div>
          <div className="ot-stat-box">
            <div className="ot-stat-num">{(result.confidence * 100).toFixed(1)}%</div>
            <div className="ot-stat-label">confidence</div>
          </div>
          <div className="ot-stat-box">
            <div className="ot-stat-num">{result.inference_ms} ms</div>
            <div className="ot-stat-label">{result.size[0]}×{result.size[1]} px (onnx, cpu)</div>
          </div>
        </div>
      )}
    </>
  );
}
