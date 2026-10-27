import { useEffect, useRef, useState } from 'react';
import { friendlyArch, friendlyBase } from './modelNames';
import {
  paddleOcrAPI, PaddleArch, PaddleLogEntry, PaddleModel, PaddleTrainRequest,
} from '@/services/paddleOcrTraining';
import { ocrTrainingAPI } from '@/services/ocrTraining';
import SyntheticDefectsPanel from './SyntheticDefectsPanel';
import TrainLogPanel from './TrainLogPanel';
import { fmtServiceTime } from './time';

interface Props {
  projectId: string;
  /** Owned by OCRTrainingPage so Eval/Export/Test see the same runs + selection. */
  models: PaddleModel[];
  onModelsChange: () => void;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

// 15 epochs from pretrained reached 93-95% engine accuracy on MINCED ONION
// (635 train / 154 test). The learning rate is not exposed: the service applies
// the upstream PP-OCRv6 value (5e-4) with a warmup scaled to the epoch count.
const DEFAULTS = {
  epoch_num: 15,
  batch_size: 16,
  test_split: 0.2,
  max_train_width: 768,
  max_text_length: 25,
};

/** Floor the service enforces regardless of batch size. */
const MIN_TRAIN_IMAGES = 8;

// Batch size and test split stay at DEFAULTS (16 / 0.2) and are not shown.
const SHOW_BATCH_SPLIT = false;

function fmtPct(v: number | null | undefined): string {
  return v == null ? 'n/a' : `${(v * 100).toFixed(1)}%`;
}

/** Train tab body for the PaddleOCR family (ocr_paddle_service, :8003).
 *  Export (ONNX + TensorRT) and the accuracy check run automatically at the end
 *  of every run, so there is no separate Eval/Export step for these models:
 *  once a run shows "completed" with an engine it is selectable on a recipe as
 *  OCR model "Paddle · <project>". */
export default function PaddleTrainPanel({
  projectId, models, onModelsChange, selectedId, onSelect,
}: Props) {
  const [archs, setArchs] = useState<PaddleArch[]>([]);
  const [arch, setArch] = useState('PP-OCRv6_small_rec');
  const [baseValue, setBaseValue] = useState('pretrained');
  const [params, setParams] = useState(DEFAULTS);
  const [useSynthetic, setUseSynthetic] = useState(false);
  const [starting, setStarting] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const [logs, setLogs] = useState<PaddleLogEntry[]>([]);
  const [live, setLive] = useState<{ phase: string | null; progress: number; status: string } | null>(null);
  const sinceRef = useRef(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadModels = onModelsChange;

  // Train-image count for the warning below. The studio's dry-run prepare
  // shares this project's labels; this service's own validation (dict chars,
  // max length 25) can drop a few more, so it is an estimate — the start
  // request is still refused by the service if the real count is too low.
  const [nTrain, setNTrain] = useState<number | null>(null);
  useEffect(() => {
    let cancelled = false;
    ocrTrainingAPI.prepare(projectId, { dryRun: true, testSplit: params.test_split, maxTextLength: 25 })
      .then((r) => { if (!cancelled) setNTrain(r.n_train); })
      .catch(() => { if (!cancelled) setNTrain(null); });
    return () => { cancelled = true; };
  }, [projectId, params.test_split, models.length]);
  const minTrain = Math.max(MIN_TRAIN_IMAGES, params.batch_size);
  const minTrainShort = nTrain != null && nTrain < minTrain;

  // Re-read archs whenever the run list changes: a finished run becomes a
  // selectable base for its architecture.
  useEffect(() => {
    paddleOcrAPI.listArchs().then(setArchs).catch((e) => console.error('[PaddleOCR] archs', e));
  }, [projectId, models.length]);

  const archInfo = archs.find((a) => a.id === arch) || null;
  useEffect(() => {
    // A base belongs to one arch; switching arch invalidates a model base.
    if (baseValue !== 'pretrained' && !archInfo?.trained_bases.some((b) => `model:${b.model_id}` === baseValue)) {
      setBaseValue('pretrained');
    }
  }, [arch, archInfo, baseValue]);

  const running = models.find((m) => m.status === 'training' || m.status === 'pending') || null;
  const pollTargetId = running?.id || selectedId;

  useEffect(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    if (!pollTargetId) { setLogs([]); setLive(null); return; }
    sinceRef.current = 0;
    setLogs([]);
    const poll = async () => {
      try {
        const res = await paddleOcrAPI.getTrainLogs(projectId, pollTargetId, sinceRef.current);
        if (res.logs.length) {
          sinceRef.current = res.next_since;
          setLogs((prev) => [...prev, ...res.logs]);
        }
        setLive({ phase: res.phase, progress: res.progress, status: res.status });
        if (res.status !== 'training' && res.status !== 'pending') {
          if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
          loadModels();
        }
      } catch (e) {
        console.error('[PaddleOCR] log poll failed', e);
      }
    };
    poll();
    pollRef.current = setInterval(poll, 2000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, pollTargetId]);

  const start = async () => {
    setStarting(true); setError(null); setInfo(null);
    try {
      const base: PaddleTrainRequest['base'] = baseValue === 'pretrained'
        ? { kind: 'pretrained' }
        : { kind: 'model', model_id: baseValue.slice('model:'.length) };
      const res = await paddleOcrAPI.startTraining(projectId, {
        ...params, arch, base, use_synthetic: useSynthetic,
      });
      setInfo(`Run started — ${res.n_train} train / ${res.n_test} test`
        + (res.n_synth_train ? ` + ${res.n_synth_train} synthetic defect crops` : '')
        + (res.dropped_count ? `, ${res.dropped_count} dropped by validation` : '')
        + (res.gpu_holder ? `. Queued behind ${res.gpu_holder}.` : '.'));
      onSelect(res.model_id);
      loadModels();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Could not start the run');
    } finally {
      setStarting(false);
    }
  };

  const act = async (label: string, fn: () => Promise<any>, confirmMsg?: string) => {
    if (confirmMsg && !confirm(confirmMsg)) return;
    setBusy(label); setError(null);
    try {
      await fn();
      await loadModels();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || `${label} failed`);
    } finally {
      setBusy(null);
    }
  };

  const canStart = !starting && !running && !minTrainShort
    && (baseValue !== 'pretrained' || !!archInfo?.pretrained_available);

  return (
    <>
      {error && <div className="at-alert-error">{error}</div>}
      {info && <div className="at-alert-ok">{info}</div>}

      {minTrainShort && (
        <div className="at-alert-error">
          Cần ít nhất <b>{minTrain}</b> ảnh train (batch size {params.batch_size}) — hiện chỉ có
          {' '}<b>{nTrain}</b> (ước tính). Ít ảnh hơn một batch thì run không train được gì và sẽ lỗi.
          Verify thêm nhãn ở tab Label, giảm test split, hoặc chọn batch size nhỏ hơn.
        </div>
      )}

      <div className="ot-train-layout">
        <section className="ot-card ot-train-settings">
          <div className="ot-card-title">Training settings</div>
          <div className="ot-settings-grid">
            <label className="at-label">
              Architecture
              <select className="at-form-input" value={arch} onChange={(e) => setArch(e.target.value)}>
                {archs.map((a) => (
                  <option key={a.id} value={a.id}>
                    {friendlyArch(a.label)}{a.recommended ? ' (recommended)' : ''}
                    {!a.pretrained_available && !a.trained_bases.length ? ' — no weights' : ''}
                  </option>
                ))}
              </select>
            </label>

            <label className="at-label ot-span-2">
              Start from
              <select className="at-form-input" value={baseValue} onChange={(e) => setBaseValue(e.target.value)}>
                <option value="pretrained" disabled={!archInfo?.pretrained_available}>
                  Starter model{archInfo?.pretrained_available ? '' : ' — file missing'}
                </option>
                {(archInfo?.trained_bases ?? []).map((b) => (
                  <option key={b.model_id} value={`model:${b.model_id}`}>
                    Trained · {b.project_name} · {friendlyArch(b.label)}
                  </option>
                ))}
              </select>
            </label>

            <label className="at-label">
              Epochs
              <input className="at-form-input" type="number" min={1} max={500}
                     value={params.epoch_num}
                     onChange={(e) => setParams((p) => ({ ...p, epoch_num: Math.max(1, +e.target.value || 1) }))} />
            </label>

            {SHOW_BATCH_SPLIT && (
              <label className="at-label">
                Batch size
                <select className="at-form-input" value={params.batch_size}
                        onChange={(e) => setParams((p) => ({ ...p, batch_size: +e.target.value }))}>
                  {[16, 32, 64].map((b) => <option key={b} value={b}>{b}</option>)}
                </select>
              </label>
            )}

            {SHOW_BATCH_SPLIT && (
              <label className="at-label">
                Test split
                <input className="at-form-input" type="number" step={0.05} min={0.05} max={0.9}
                       value={params.test_split}
                       onChange={(e) => setParams((p) => ({ ...p, test_split: Math.min(0.9, Math.max(0.05, +e.target.value || 0.2)) }))} />
              </label>
            )}
          </div>


          <SyntheticDefectsPanel projectId={projectId} testSplit={params.test_split}
                                 useSynthetic={useSynthetic} onUseSyntheticChange={setUseSynthetic} />

          <div className="ot-toolbar">
            <button className="at-btn at-btn-primary" onClick={start} disabled={!canStart}>
              {starting ? 'Starting…' : running ? 'A run is already active' : 'Start training'}
            </button>
            {running && (
              <button className="at-btn at-btn-secondary"
                      onClick={() => act('Cancel', () => paddleOcrAPI.cancelTraining(projectId, running.id),
                        'Cancel this run? The partially trained checkpoint is discarded.')}>
                Cancel run
              </button>
            )}
          </div>
        </section>

        <TrainLogPanel live={live} logs={logs.filter((l) => !l.msg.includes('epoch: ['))} />
      </div>

      <div className="ot-card-title ot-runs-title">Runs <span className="ot-count">{models.length}</span></div>
      {models.length === 0 && <div className="at-empty-state" style={{ padding: 24 }}>No runs yet</div>}

      {models.map((m) => {
        const live_ = m.status === 'training' || m.status === 'pending';
        return (
          <div key={m.id} className={`ot-label-row ${selectedId === m.id ? 'is-focused' : ''}`}
               style={{ gridTemplateColumns: '1fr auto', cursor: 'pointer' }}
               onClick={() => onSelect(m.id)}>
            <div className="ot-label-fields">
              <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
                <b style={{ fontFamily: 'monospace' }}>{m.id.slice(-8)}</b>
                <span className="ot-chip-sm">{friendlyArch(m.arch)}</span>
                <span className="ot-chip-sm">{m.status}</span>
                {m.phase && live_ && <span className="ot-chip-sm">{m.phase}</span>}
                <span className="ot-chip-sm">{friendlyBase(m.base_label)}</span>
              </div>
              <div className="ot-hint-row">
                <span>engine <b>{fmtPct(m.metrics.acc_trt)}</b></span>
                {(m.metrics.syn_reject_trt ?? m.metrics.syn_reject_onnx) != null && (
                  <span title="Share of synthetic defect crops (made from test images) the line would FAIL">
                    bắt lỗi <b>{fmtPct(m.metrics.syn_reject_trt ?? m.metrics.syn_reject_onnx)}</b>
                  </span>
                )}
                {!!m.metrics.n_synth_train && <span className="ot-chip-sm">+{m.metrics.n_synth_train} lỗi</span>}
                {m.metrics.acc_exact_trt != null && <span>exact <b>{fmtPct(m.metrics.acc_exact_trt)}</b></span>}
                <span>onnx <b>{fmtPct(m.metrics.acc_onnx)}</b></span>
                <span>train-eval <b>{fmtPct(m.metrics.acc)}</b></span>
                {m.metrics.best_epoch != null && <span>best epoch <b>{m.metrics.best_epoch}</b></span>}
                <span>{m.metrics.n_train}/{m.metrics.n_test}</span>
              </div>
              {m.error && <div className="at-log-error" style={{ fontSize: 11 }}>{m.error}</div>}
              <div className="ot-hint-row">
                {m.engine_path && <span className="ot-chip-sm ot-pass">engine</span>}
                {m.onnx_path && <span className="ot-chip-sm">onnx</span>}
                {m.status === 'completed' && (m.engine_path || m.onnx_path) && (
                  <span className="ot-chip-sm ot-pass">selectable on recipes</span>
                )}
                {m.status === 'completed' && !m.engine_path && (
                  <span className="ot-chip-sm ot-fail">no engine</span>
                )}
                <span>{fmtServiceTime(m.created_at)}</span>
              </div>
            </div>
            <div className="ot-row-actions" style={{ display: 'flex', gap: 4, flexWrap: 'wrap', justifyContent: 'flex-end' }}
                 onClick={(e) => e.stopPropagation()}>
              {live_ ? (
                <button className="at-btn at-btn-secondary at-btn-sm"
                        onClick={() => act('Cancel', () => paddleOcrAPI.cancelTraining(projectId, m.id),
                          'Cancel this run?')}>Cancel</button>
              ) : (
                <>
                  {m.status === 'completed' && !m.onnx_path && (
                    <button className="at-btn at-btn-secondary at-btn-sm" disabled={!!busy}
                            onClick={() => act('Export ONNX', () => paddleOcrAPI.exportOnnx(projectId, m.id))}>
                      {busy === 'Export ONNX' ? '…' : 'Export ONNX'}
                    </button>
                  )}
                  {m.status === 'completed' && m.onnx_path && !m.engine_path && (
                    <button className="at-btn at-btn-secondary at-btn-sm" disabled={!!busy}
                            onClick={() => act('Build engine', () => paddleOcrAPI.exportTensorRT(projectId, m.id))}>
                      {busy === 'Build engine' ? 'Building…' : 'Build engine'}
                    </button>
                  )}
                  {m.status === 'completed' && m.engine_path && (
                    <button className="at-btn at-btn-secondary at-btn-sm" disabled={!!busy}
                            onClick={() => act('Re-evaluate', () => paddleOcrAPI.evaluate(projectId, m.id, 'tensorrt'))}>
                      {busy === 'Re-evaluate' ? '…' : 'Re-evaluate'}
                    </button>
                  )}
                  {m.onnx_path && (
                    <button className="at-btn at-btn-secondary at-btn-sm"
                            onClick={() => paddleOcrAPI.download(projectId, m.id, 'onnx')}>ONNX</button>
                  )}
                  {m.dict_path && (
                    <button className="at-btn at-btn-secondary at-btn-sm"
                            onClick={() => paddleOcrAPI.download(projectId, m.id, 'dict')}>Dict</button>
                  )}
                  <button className="at-btn at-btn-sm" style={{ background: '#fee2e2', color: '#b91c1c' }}
                          onClick={() => act('Delete', async () => {
                            await paddleOcrAPI.deleteModel(projectId, m.id);
                            if (selectedId === m.id) onSelect(null);
                          },
                            'Delete this model and every file the run wrote?')}>
                    Delete
                  </button>
                </>
              )}
            </div>
          </div>
        );
      })}
    </>
  );
}
