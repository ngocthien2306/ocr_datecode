import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ocrTrainingAPI, ocrTrainingModelAPI, OCRBaseCheckpoints, OCRModel,
  OCRPrepareReport, OCRTrainLogEntry,
} from '@/services/ocrTraining';

interface Props {
  projectId: string;
  models: OCRModel[];
  onModelsChange: () => void;
  selectedModelId: string | null;
  onSelectModel: (id: string | null) => void;
}

/** Defaults established by measurement, not preference — see
 *  docs/ocr_training_plan.md §2. */
const DEFAULTS = {
  epoch_num: 50,
  batch_size: 32,
  lr: 0.0001,
  test_split: 0.2,
  use_space_char: true,
};

// batch 32 measured at ~2.6 GB against ai_services' 3.5 GB on a 16 GB card.
// The numbers are the measured training peak, not an estimate.
const VRAM_BY_BATCH: Record<number, number> = { 16: 1664, 32: 2592, 64: 4614, 128: 8946 };

function fmtPct(v: number | null | undefined): string {
  return v == null ? 'n/a' : `${(v * 100).toFixed(1)}%`;
}

export default function TrainTab({
  projectId, models, onModelsChange, selectedModelId, onSelectModel,
}: Props) {
  const [bases, setBases] = useState<OCRBaseCheckpoints | null>(null);
  const [baseValue, setBaseValue] = useState('builtin:datecode_2406');
  const [params, setParams] = useState(DEFAULTS);
  const [report, setReport] = useState<OCRPrepareReport | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const [logs, setLogs] = useState<OCRTrainLogEntry[]>([]);
  const [live, setLive] = useState<{ phase: string | null; progress: number; status: string } | null>(null);
  const sinceRef = useRef(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const logBoxRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    ocrTrainingModelAPI.listBaseCheckpoints()
      .then(setBases)
      .catch((e) => console.error('[OCRTraining] base list failed', e));
  }, [projectId, models.length]);

  const refreshReport = useCallback(async () => {
    try {
      setReport(await ocrTrainingAPI.prepare(projectId, {
        dryRun: true, testSplit: params.test_split, useSpaceChar: params.use_space_char,
      }));
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Dataset check failed');
    }
  }, [projectId, params.test_split, params.use_space_char]);

  useEffect(() => { refreshReport(); }, [refreshReport]);

  const activeModel = models.find((m) => m.id === selectedModelId) || null;
  const running = models.find((m) => m.status === 'training' || m.status === 'pending') || null;

  // Poll the selected model while it is live. The `since` cursor means each poll
  // ships only new lines, so a long run's log doesn't get re-sent every 2s.
  const pollTargetId = running?.id || selectedModelId;
  useEffect(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    if (!pollTargetId) { setLogs([]); setLive(null); return; }
    sinceRef.current = 0;
    setLogs([]);

    const poll = async () => {
      try {
        const res = await ocrTrainingModelAPI.getTrainLogs(projectId, pollTargetId, sinceRef.current);
        if (res.logs.length) {
          sinceRef.current = res.next_since;
          setLogs((prev) => [...prev, ...res.logs]);
        }
        setLive({ phase: res.phase, progress: res.progress, status: res.status });
        if (res.status !== 'training' && res.status !== 'pending') {
          if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
          onModelsChange();
        }
      } catch (e) {
        console.error('[OCRTraining] log poll failed', e);
      }
    };
    poll();
    pollRef.current = setInterval(poll, 2000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [projectId, pollTargetId]);

  useEffect(() => {
    if (logBoxRef.current) logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
  }, [logs.length]);

  const selectedBaseInfo = (() => {
    if (!bases) return null;
    if (baseValue.startsWith('builtin:')) {
      const id = baseValue.slice('builtin:'.length);
      return bases.builtin.find((b) => b.id === id) ?? null;
    }
    const mid = baseValue.slice('model:'.length);
    for (const g of bases.projects) {
      const m = g.models.find((x) => x.model_id === mid);
      if (m) return m;
    }
    return null;
  })();

  // Narrowing the vocabulary would delete the space class and every weight that
  // learned it, so the backend refuses it with a 400. Say so before the click.
  const vocabConflict = !!selectedBaseInfo
    && (selectedBaseInfo as any).use_space_char === true
    && !params.use_space_char;

  const start = async () => {
    setStarting(true);
    setError(null);
    setInfo(null);
    try {
      const base = baseValue.startsWith('builtin:')
        ? { kind: 'builtin' as const, builtin: baseValue.slice('builtin:'.length) }
        : { kind: 'model' as const, model_id: baseValue.slice('model:'.length) };
      const res = await ocrTrainingModelAPI.startTraining(projectId, { ...params, base });
      setInfo(
        `Run started — ${res.n_train} train / ${res.n_test} test`
        + (res.dropped_count ? `, ${res.dropped_count} dropped by validation` : '')
        + (res.gpu_holder ? `. Queued behind ${res.gpu_holder}.` : '.'),
      );
      onSelectModel(res.model_id);
      onModelsChange();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Could not start the run');
    } finally {
      setStarting(false);
    }
  };

  const cancel = async (id: string) => {
    if (!confirm('Cancel this run? The partially trained checkpoint is discarded.')) return;
    try {
      await ocrTrainingModelAPI.cancelTraining(projectId, id);
      onModelsChange();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Cancel failed');
    }
  };

  const remove = async (id: string) => {
    if (!confirm('Delete this model and every file the run wrote (checkpoint, ONNX, engine)?')) return;
    try {
      await ocrTrainingModelAPI.deleteModel(projectId, id);
      if (selectedModelId === id) onSelectModel(null);
      onModelsChange();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Delete failed');
    }
  };

  const estVram = VRAM_BY_BATCH[params.batch_size];
  const lowEpochsWithSpace = params.use_space_char && params.epoch_num < 20;

  return (
    <>
      {error && <div className="at-alert-error">{error}</div>}
      {info && <div className="at-alert-ok">{info}</div>}

      <div className="ot-prepare-grid">
        <div className="ot-stat-box">
          <div className="ot-stat-num">{report?.n_train ?? '—'}</div>
          <div className="ot-stat-label">train images</div>
        </div>
        <div className="ot-stat-box">
          <div className="ot-stat-num">{report?.n_test ?? '—'}</div>
          <div className="ot-stat-label">test ({report?.split_source ?? '—'})</div>
        </div>
        <div className={`ot-stat-box ${report?.dropped_count ? 'ot-bad' : ''}`}>
          <div className="ot-stat-num">{report?.dropped_count ?? 0}</div>
          <div className="ot-stat-label">dropped labels</div>
        </div>
        <div className="ot-stat-box">
          <div className="ot-stat-num">{estVram ? `${(estVram / 1024).toFixed(1)}G` : '—'}</div>
          <div className="ot-stat-label">measured VRAM @ batch {params.batch_size}</div>
        </div>
      </div>

      {report?.blocking_reason && <div className="at-alert-error">{report.blocking_reason}</div>}

      <div className="at-form-row" style={{ flexWrap: 'wrap', gap: 12 }}>
        <label className="at-label" style={{ minWidth: 320, flex: '1 1 320px' }}>
          Base checkpoint
          <select className="at-form-input" value={baseValue}
                  onChange={(e) => setBaseValue(e.target.value)}>
            <optgroup label="Built-in">
              {(bases?.builtin ?? []).map((b) => (
                <option key={b.id} value={`builtin:${b.id}`} disabled={!b.available}>
                  {b.id}{b.recommended ? ' (recommended)' : ''}{b.available ? '' : ' — file missing'}
                </option>
              ))}
            </optgroup>
            {(bases?.projects ?? []).map((g) => (
              <optgroup key={g.project_id} label={`Trained · ${g.project_name}`}>
                {g.models.map((m) => (
                  <option key={m.model_id} value={`model:${m.model_id}`}>
                    {m.label}{m.use_space_char ? ' · space' : ''}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
          <small className="at-hint">
            Always a fine-tune — a few thousand crops cannot train SVTRv2 from scratch.
            Models from other projects are listed on purpose: continuing from last week's
            model after adding images is the common case.
          </small>
        </label>

        <label className="at-label">
          Epochs
          <input className="at-form-input" type="number" min={1} max={500} style={{ width: 90 }}
                 value={params.epoch_num}
                 onChange={(e) => setParams((p) => ({ ...p, epoch_num: Math.max(1, +e.target.value || 1) }))} />
        </label>

        <label className="at-label">
          Batch size
          <select className="at-form-input" style={{ width: 100 }} value={params.batch_size}
                  onChange={(e) => setParams((p) => ({ ...p, batch_size: +e.target.value }))}>
            {[16, 32, 64, 128].map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>

        <label className="at-label">
          Learning rate
          <input className="at-form-input" type="number" step={0.00001} min={0.000001} style={{ width: 110 }}
                 value={params.lr}
                 onChange={(e) => setParams((p) => ({ ...p, lr: +e.target.value || DEFAULTS.lr }))} />
        </label>

        <label className="at-label">
          Test split
          <input className="at-form-input" type="number" step={0.05} min={0} max={0.9} style={{ width: 90 }}
                 value={params.test_split}
                 onChange={(e) => setParams((p) => ({ ...p, test_split: Math.min(0.9, Math.max(0, +e.target.value || 0)) }))} />
        </label>

        <label className="at-label" style={{ alignSelf: 'flex-end' }}>
          <input type="checkbox" checked={params.use_space_char}
                 onChange={(e) => setParams((p) => ({ ...p, use_space_char: e.target.checked }))} />
          {' '}Train with spaces
        </label>
      </div>

      <div className="at-hint" style={{ marginBottom: 8 }}>
        <b>Train with spaces</b> lets the model reproduce spaces in the label:
        exact-match went 0.43 → 0.93 with no measurable cost in character accuracy.
        A base that lacks the space class is widened automatically — 1,154 new
        parameters out of 21M, not a retrain from scratch.
      </div>

      {params.batch_size >= 128 && (
        <div className="at-hint">
          Batch 128 measured at 8.9 GB. With ai_services live (3.5 GB) that is 76% of a
          16 GB card and leaves no room for anything else on the GPU.
        </div>
      )}

      {lowEpochsWithSpace && (
        <div className="at-hint">
          With spaces on, the best epoch landed around 18 in testing — the autoregressive
          head needs a few epochs to learn <i>where</i> spaces go. Below ~20 epochs the run
          may stop before it recovers.
        </div>
      )}

      {vocabConflict && (
        <div className="at-alert-error">
          The selected base was trained with spaces. Training from it with spaces off would
          drop the space class it learned, so the service refuses it — re-enable “Train with
          spaces”, or pick a different base.
        </div>
      )}

      <div className="ot-toolbar">
        <button className="at-btn at-btn-primary" onClick={start}
                disabled={starting || !!running || !!report?.blocking_reason || vocabConflict}>
          {starting ? 'Starting…' : running ? 'A run is already active' : 'Start training'}
        </button>
        {running && (
          <button className="at-btn at-btn-secondary" onClick={() => cancel(running.id)}>
            Cancel run
          </button>
        )}
        <span className="ot-spacer" />
        {live && (
          <span className="at-hint">
            {live.status} · {live.phase ?? '—'} · {Math.round(live.progress)}%
            {live.phase === 'waiting_for_gpu' && ' (queued behind another GPU job)'}
          </span>
        )}
      </div>

      {live && (live.status === 'training' || live.status === 'pending') && (
        <div style={{ height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden', marginBottom: 10 }}>
          <div style={{ height: '100%', width: `${live.progress}%`, background: '#3b82f6',
                        transition: 'width .4s' }} />
        </div>
      )}

      {logs.length > 0 && (
        <div ref={logBoxRef}
             style={{ maxHeight: 220, overflowY: 'auto', background: '#0f172a', color: '#cbd5e1',
                      fontFamily: 'ui-monospace, Menlo, monospace', fontSize: 11, padding: 10,
                      borderRadius: 6, marginBottom: 12 }}>
          {logs.map((l) => (
            <div key={l.idx} className={l.level === 'ERROR' ? 'at-log-error'
                                       : l.level === 'WARNING' ? 'at-log-warning' : undefined}>
              {l.msg}
            </div>
          ))}
        </div>
      )}

      <h3 style={{ margin: '4px 0 8px' }}>Runs</h3>
      {models.length === 0 && <div className="at-empty-state" style={{ padding: 24 }}>No runs yet</div>}

      {models.map((m) => (
        <div key={m.id} className={`ot-label-row ${selectedModelId === m.id ? 'is-focused' : ''}`}
             style={{ gridTemplateColumns: '1fr auto', cursor: 'pointer' }}
             onClick={() => onSelectModel(m.id)}>
          <div className="ot-label-fields">
            <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
              <b style={{ fontFamily: 'monospace' }}>{m.id.slice(-8)}</b>
              <span className="ot-chip-sm">{m.status}</span>
              {m.phase && m.status === 'training' && <span className="ot-chip-sm">{m.phase}</span>}
              {m.use_space_char && <span className="ot-chip-sm ot-pass">space</span>}
              <span className="ot-chip-sm">vocab {m.vocab_size}</span>
              <span className="ot-chip-sm">{m.base_label}</span>
            </div>
            <div className="ot-hint-row">
              <span>min_acc <b>{fmtPct(m.metrics.min_acc)}</b></span>
              <span>ctc <b>{fmtPct(m.metrics.acc)}</b></span>
              <span>gtc <b>{fmtPct(m.metrics.gtc_acc)}</b></span>
              {m.metrics.best_epoch != null && <span>best epoch <b>{m.metrics.best_epoch}</b></span>}
              {m.metrics.acc_trt != null && <span>engine <b>{fmtPct(m.metrics.acc_trt)}</b></span>}
              {m.metrics.acc_exact_trt != null && <span>exact <b>{fmtPct(m.metrics.acc_exact_trt)}</b></span>}
              <span>{m.metrics.n_train}/{m.metrics.n_test}</span>
            </div>
            {m.error && <div className="at-log-error" style={{ fontSize: 11 }}>{m.error}</div>}
            <div className="ot-hint-row">
              {m.engine_path && <span className="ot-chip-sm ot-pass">engine</span>}
              {m.onnx_fp16_path && <span className="ot-chip-sm">onnx fp16</span>}
              {!m.engine_path && m.status === 'completed' && (
                <span className="ot-chip-sm ot-fail">no engine — export it in the Export tab</span>
              )}
              <span>{new Date(m.created_at).toLocaleString()}</span>
            </div>
          </div>
          <div className="ot-row-actions">
            {(m.status === 'training' || m.status === 'pending') ? (
              <button className="at-btn at-btn-secondary at-btn-sm"
                      onClick={(e) => { e.stopPropagation(); cancel(m.id); }}>Cancel</button>
            ) : (
              <button className="at-btn at-btn-sm" style={{ background: '#fee2e2', color: '#b91c1c' }}
                      onClick={(e) => { e.stopPropagation(); remove(m.id); }}>Delete</button>
            )}
          </div>
        </div>
      ))}

      {activeModel && activeModel.metrics.acc_trt != null && activeModel.metrics.min_acc != null
        && activeModel.metrics.acc_trt < activeModel.metrics.min_acc - 0.02 && (
        <div className="at-hint">
          The engine for {activeModel.id.slice(-8)} scores{' '}
          {((activeModel.metrics.min_acc - activeModel.metrics.acc_trt) * 100).toFixed(1)}pp below its
          checkpoint — the fp16 export probably lost accuracy. Compare ONNX against TensorRT in the
          Eval tab before putting it on a recipe.
        </div>
      )}
    </>
  );
}
