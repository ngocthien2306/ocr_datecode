import { useRef, useState } from 'react';
import {
  ocrTrainingModelAPI, InferenceEngine, OCRModel, OCRPredictResult,
} from '@/services/ocrTraining';

interface Props {
  projectId: string;
  model: OCRModel | null;
}

interface Run { engine: InferenceEngine; result: OCRPredictResult }

export default function TestTab({ projectId, model }: Props) {
  const [engine, setEngine] = useState<InferenceEngine>('tensorrt');
  const [runs, setRuns] = useState<Run[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  if (!model) {
    return <div className="at-empty-state" style={{ padding: 40 }}>
      Select a run in the Train tab first.
    </div>;
  }

  const pick = (f: File | null) => {
    setFile(f);
    setRuns([]);
    setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(f ? URL.createObjectURL(f) : null);
  };

  const run = async (which: InferenceEngine | 'both') => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const engines: InferenceEngine[] = which === 'both' ? ['tensorrt', 'onnx'] : [which];
      const out: Run[] = [];
      for (const e of engines) {
        out.push({ engine: e, result: await ocrTrainingModelAPI.predict(projectId, model.id, file, e) });
      }
      setRuns(out);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Inference failed');
    } finally {
      setBusy(false);
    }
  };

  // Destructured rather than indexed: noUncheckedIndexedAccess types runs[0] as
  // possibly-undefined even after the length check.
  const [first, second] = runs;
  const disagree = !!first && !!second
    && (first.result.gtc_text !== second.result.gtc_text
     || first.result.ctc_text !== second.result.ctc_text);

  return (
    <>
      {error && <div className="at-alert-error">{error}</div>}

      <div className="at-hint" style={{ marginBottom: 10 }}>
        Read a single crop. Recognizers stay cached per artifact, so the first call after a
        rebuild includes load time and later ones show steady-state speed.
      </div>

      <div className="ot-toolbar">
        <input ref={inputRef} type="file" accept="image/*"
               onChange={(e) => pick(e.target.files?.[0] ?? null)} />
        <label className="at-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          engine
          <select className="at-form-input" style={{ width: 120 }} value={engine}
                  onChange={(e) => setEngine(e.target.value as InferenceEngine)}>
            <option value="tensorrt">TensorRT</option>
            <option value="onnx">ONNX</option>
          </select>
        </label>
        <button className="at-btn at-btn-primary" onClick={() => run(engine)} disabled={busy || !file}>
          {busy ? 'Reading…' : 'Read'}
        </button>
        <button className="at-btn at-btn-secondary" onClick={() => run('both')} disabled={busy || !file}
                title="A disagreement between the two means the fp16 engine diverged from its ONNX source">
          Compare both
        </button>
        {!model.engine_path && (
          <span className="at-hint">No engine on this model — only the ONNX backend will work.</span>
        )}
      </div>

      {preview && (
        <div className="ot-crop-wrap" style={{ marginBottom: 10 }}>
          <img className="ot-crop-img ot-crop-lg" src={preview} alt="upload preview" />
        </div>
      )}

      {disagree && (
        <div className="at-alert-error">
          TensorRT and ONNX read this crop differently. They agreed exactly across the whole
          test set during validation, so a disagreement here points at the fp16 engine having
          diverged from its ONNX source — rebuild the engine before trusting it.
        </div>
      )}

      {runs.map(({ engine: e, result }) => (
        <div key={e} className="ot-label-row" style={{ gridTemplateColumns: '1fr 220px' }}>
          <div className="ot-label-fields">
            <div style={{ display: 'flex', gap: 8, alignItems: 'baseline' }}>
              <span className="ot-chip-sm">{e}</span>
              <b style={{ fontFamily: 'monospace', fontSize: 16 }}>
                {result.gtc_conf >= result.ctc_conf ? result.gtc_text : result.ctc_text}
              </b>
              <span className="at-hint">(higher-confidence head)</span>
            </div>
            <div className="ot-hint-row">
              <span className="ot-hint-exp">gtc <b>{result.gtc_text || '(empty)'}</b> ({result.gtc_conf.toFixed(3)})</span>
              <span className="ot-hint-rec">ctc <b>{result.ctc_text || '(empty)'}</b> ({result.ctc_conf.toFixed(3)})</span>
            </div>
            <div className="ot-hint-row">
              <span>{result.size[0]}×{result.size[1]} px</span>
              <span>{result.inference_ms} ms</span>
            </div>
          </div>
          <div className="ot-row-actions">
            <span className="at-hint">
              Production picks between both heads, so either reading being right is what
              matters on the line.
            </span>
          </div>
        </div>
      ))}
    </>
  );
}
