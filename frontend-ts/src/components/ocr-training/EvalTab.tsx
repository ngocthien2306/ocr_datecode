import { useState } from 'react';
import {
  ocrTrainingModelAPI, InferenceEngine, OCREvalResult, OCRModel,
} from '@/services/ocrTraining';

interface Props {
  projectId: string;
  model: OCRModel | null;
}

function pct(v: number | null | undefined): string {
  return v == null ? 'n/a' : `${(v * 100).toFixed(2)}%`;
}

export default function EvalTab({ projectId, model }: Props) {
  const [engine, setEngine] = useState<InferenceEngine>('tensorrt');
  const [result, setResult] = useState<OCREvalResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [onlyWrong, setOnlyWrong] = useState(true);
  const [strict, setStrict] = useState(false);

  if (!model) {
    return <div className="at-empty-state" style={{ padding: 40 }}>
      Select a run in the Train tab first.
    </div>;
  }

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      setResult(await ocrTrainingModelAPI.evaluate(projectId, model.id, engine, true));
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
      <div className="ot-toolbar">
        <label className="at-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          engine
          <select className="at-form-input" style={{ width: 120 }} value={engine}
                  onChange={(e) => setEngine(e.target.value as InferenceEngine)}>
            <option value="tensorrt">TensorRT</option>
            <option value="onnx">ONNX</option>
          </select>
        </label>
        <button className="at-btn at-btn-primary" onClick={run} disabled={running}>
          {running ? 'Scoring…' : 'Run evaluation'}
        </button>
        <span className="ot-spacer" />
        <span className="at-hint">
          Scored at batch=1. Batched inference pads every crop out to the widest one in
          the batch, which costs real accuracy — the same engine scored 0.968 at batch 1
          and 0.917 at batch 8.
        </span>
      </div>

      {error && <div className="at-alert-error">{error}</div>}

      {!result && !running && (
        <div className="at-hint">
          Nothing measured yet. Training already stored the engine's accuracy
          ({pct(model.metrics.acc_trt)} normalized, {pct(model.metrics.acc_exact_trt)} exact);
          run this to reproduce it, see which images fail, or compare ONNX against TensorRT.
        </div>
      )}

      {s && (
        <>
          <div className="ot-prepare-grid">
            <div className="ot-stat-box">
              <div className="ot-stat-num">{pct(s.norm_either)}</div>
              <div className="ot-stat-label">normalized · either head</div>
            </div>
            <div className="ot-stat-box">
              <div className="ot-stat-num">{pct(s.exact_either)}</div>
              <div className="ot-stat-label">exact · either head</div>
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
            <b>normalized</b> keeps letters and digits only and lowercases — the same form
            train_rec.py reports, and effectively what production's compare_texts() does
            (it strips spaces and punctuation before comparing). <b>exact</b> is raw string
            equality; it is much lower for a model trained without the space class, because
            such a model cannot emit spaces at all. Do not compare the two figures.
          </div>

          <table style={{ borderCollapse: 'collapse', fontSize: 12, marginBottom: 12 }}>
            <thead>
              <tr style={{ textAlign: 'left', color: '#64748b' }}>
                <th style={{ padding: '4px 12px 4px 0' }} />
                <th style={{ padding: '4px 12px 4px 0' }}>GTC (SMTR)</th>
                <th style={{ padding: '4px 12px 4px 0' }}>CTC</th>
                <th style={{ padding: '4px 12px 4px 0' }}>either</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ padding: '4px 12px 4px 0' }}><b>normalized</b></td>
                <td style={{ padding: '4px 12px 4px 0' }}>{pct(s.norm_gtc)}</td>
                <td style={{ padding: '4px 12px 4px 0' }}>{pct(s.norm_ctc)}</td>
                <td style={{ padding: '4px 12px 4px 0' }}>{pct(s.norm_either)}</td>
              </tr>
              <tr>
                <td style={{ padding: '4px 12px 4px 0' }}><b>exact</b></td>
                <td style={{ padding: '4px 12px 4px 0' }}>{pct(s.exact_gtc)}</td>
                <td style={{ padding: '4px 12px 4px 0' }}>{pct(s.exact_ctc)}</td>
                <td style={{ padding: '4px 12px 4px 0' }}>{pct(s.exact_either)}</td>
              </tr>
              <tr>
                <td style={{ padding: '4px 12px 4px 0', color: '#64748b' }}>checkpoint</td>
                <td style={{ padding: '4px 12px 4px 0', color: '#64748b' }}>{pct(result!.train_metrics.gtc_acc)}</td>
                <td style={{ padding: '4px 12px 4px 0', color: '#64748b' }}>{pct(result!.train_metrics.acc)}</td>
                <td style={{ padding: '4px 12px 4px 0', color: '#64748b' }}>
                  min {pct(result!.train_metrics.min_acc)}
                </td>
              </tr>
            </tbody>
          </table>

          <div className="at-hint" style={{ marginBottom: 8 }}>
            The checkpoint row is <b>min_acc</b> — the worse of the two heads — while
            “either” is the better one. The engine reading above the checkpoint is those two
            questions differing, not a contradiction. What would matter is the engine reading
            well <i>below</i> the checkpoint: that means the fp16 export lost something.
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
                    <span className={it.gtc_text === it.gt_text ? 'ot-hint-exp' : 'ot-hint-rec'}>
                      gtc <b>{it.gtc_text || '(empty)'}</b> ({it.gtc_conf.toFixed(3)})
                    </span>
                    <span className={it.ctc_text === it.gt_text ? 'ot-hint-exp' : 'ot-hint-rec'}>
                      ctc <b>{it.ctc_text || '(empty)'}</b> ({it.ctc_conf.toFixed(3)})
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
            <div className="at-empty-state" style={{ padding: 30 }}>
              No failures under this filter.
            </div>
          )}
        </>
      )}
    </>
  );
}
