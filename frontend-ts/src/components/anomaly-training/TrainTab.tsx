import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import {
  anomalyTrainingAPI, AnomalyAlgorithm, AnomalyModel, AnomalyTrainRequest, DatasetStats,
  TrainLogEntry, fmtMetric,
} from '@/services/anomalyTraining';
import ConfirmDialog from '@/components/shared/ConfirmDialog';

interface Props {
  projectId: string;
  models: AnomalyModel[];
  onModelsChange: () => void;
  selectedModelId: string | null;
  onSelectModel: (id: string) => void;
}

interface ConfirmDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  confirmText?: string;
  onConfirm: (() => void) | null;
}

// Backbone is a timm feature-extractor name, and valid `layers` depend on
// that specific backbone's internal module names -- there's no way for a
// user to guess a working combination by typing. Curated to the backbones
// anomalib's own PatchCore/Padim examples are built/tested against, each
// paired with its correct feature-extraction layers.
interface BackboneOption { value: string; label: string; layers: string[] }

const BACKBONE_OPTIONS: Record<AnomalyAlgorithm, BackboneOption[]> = {
  patchcore: [
    { value: 'wide_resnet50_2', label: 'Wide ResNet50-2 — default, more accurate', layers: ['layer2', 'layer3'] },
    { value: 'resnet18', label: 'ResNet18 — faster, lighter', layers: ['layer2', 'layer3'] },
    { value: 'resnet50', label: 'ResNet50', layers: ['layer2', 'layer3'] },
  ],
  padim: [
    { value: 'resnet18', label: 'ResNet18 — default, faster', layers: ['layer1', 'layer2', 'layer3'] },
    { value: 'wide_resnet50_2', label: 'Wide ResNet50-2 — more accurate', layers: ['layer1', 'layer2', 'layer3'] },
  ],
};

const BACKBONE_PRESETS: Record<AnomalyAlgorithm, { backbone: string; layers: string[] }> = {
  patchcore: { backbone: BACKBONE_OPTIONS.patchcore[0]!.value, layers: BACKBONE_OPTIONS.patchcore[0]!.layers },
  padim: { backbone: BACKBONE_OPTIONS.padim[0]!.value, layers: BACKBONE_OPTIONS.padim[0]!.layers },
};

const STATUS_COLORS: Record<string, string> = {
  pending: '#94a3b8', training: '#b45309', completed: '#15803d', failed: '#b91c1c', cancelled: '#64748b',
};

// Fixed presets instead of a free number input -- these are the sizes
// anomalib's backbones (resnet/wide_resnet feature pyramids, stride 32)
// are actually validated against; arbitrary sizes can misalign feature
// map dimensions between layers.
const IMAGE_SIZE_OPTIONS = [128, 224, 256, 320, 384, 448, 512];

// PatchCore/Padim run in a single pass (trainer_arguments={'max_epochs': 1}
// in anomalib) -- coreset subsampling / Gaussian-statistics fitting, not
// iterative gradient descent. So there's no per-epoch loss curve and no
// early stopping; progress instead tracks these pipeline phases, matching
// the phase/progress values train.py's _progress_cb emits.
const PHASES = [
  { key: 'preparing', label: 'Prepare', pct: 5 },
  { key: 'fitting', label: 'Fit', pct: 15 },
  { key: 'predicting', label: 'Predict', pct: 65 },
  { key: 'evaluating', label: 'Evaluate', pct: 80 },
  { key: 'encoding_testset', label: 'Encode', pct: 88 },
  { key: 'saving', label: 'Save', pct: 95 },
  { key: 'completed', label: 'Done', pct: 100 },
] as const;

export default function TrainTab({ projectId, models, onModelsChange, selectedModelId, onSelectModel }: Props) {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [algorithm, setAlgorithm] = useState<AnomalyAlgorithm>('patchcore');
  const [backbone, setBackbone] = useState(BACKBONE_PRESETS.patchcore.backbone);
  const [layersText, setLayersText] = useState(BACKBONE_PRESETS.patchcore.layers.join(','));
  const [coresetRatio, setCoresetRatio] = useState(0.1);
  const [imageSize, setImageSize] = useState(256);
  const [testSplit, setTestSplit] = useState(0.2);
  const [starting, setStarting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [logs, setLogs] = useState<TrainLogEntry[]>([]);
  const [liveModel, setLiveModel] = useState<AnomalyModel | null>(null);
  const sinceRef = useRef(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const handleAlgorithmChange = (algo: AnomalyAlgorithm) => {
    setAlgorithm(algo);
    setBackbone(BACKBONE_PRESETS[algo].backbone);
    setLayersText(BACKBONE_PRESETS[algo].layers.join(','));
  };

  const handleBackboneChange = (value: string) => {
    setBackbone(value);
    const opt = BACKBONE_OPTIONS[algorithm].find((o) => o.value === value);
    if (opt) setLayersText(opt.layers.join(','));
  };

  const trainingModel = models.find((m) => m.status === 'training' || m.status === 'pending');

  // Poll logs for whichever model is currently training (or the selected
  // one, so the log panel doesn't go blank right after training finishes).
  const pollTargetId = trainingModel?.id || selectedModelId;

  useEffect(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    if (!pollTargetId) { setLogs([]); return; }
    sinceRef.current = 0;
    setLogs([]);

    const poll = async () => {
      try {
        const res = await anomalyTrainingAPI.getTrainLogs(projectId, pollTargetId, sinceRef.current);
        if (res.logs.length > 0) {
          setLogs((prev) => [...prev, ...res.logs]);
          sinceRef.current = res.next_since;
        }
        setLiveModel((prev) => ({
          ...(prev || {} as AnomalyModel), id: pollTargetId,
          phase: res.phase, progress: res.progress, status: res.status as any, error: res.error,
        } as AnomalyModel));
        if (res.status !== 'training' && res.status !== 'pending') {
          if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
          onModelsChange();
        }
      } catch (e) {
        console.error('[AnomalyTraining] log poll failed', e);
      }
    };

    poll();
    pollRef.current = setInterval(poll, 2000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [pollTargetId, projectId]);

  // Refresh whenever the tab mounts — images can be excluded or generated on
  // other tabs, and a stale count here would misstate what is about to run.
  useEffect(() => {
    anomalyTrainingAPI.datasetStats(projectId).then(setStats).catch(() => setStats(null));
  }, [projectId]);

  // Mirrors _build_datamodule: every image in train/good fits the model;
  // test/good only evaluates it, and when test/good is empty anomalib holds
  // out `test_split` of train/good to stand in for it.
  const projection = useMemo(() => {
    if (!stats) return null;
    const trainPool = Math.max(0, (stats.train_normal ?? 0) - (stats.excluded_train_normal ?? 0));
    const testNormalPool = Math.max(0, (stats.test_normal ?? 0) - (stats.excluded_test_normal ?? 0));
    const abnormal = Math.max(0, stats.abnormal_count - (stats.excluded_abnormal ?? 0));
    const heldOut = testNormalPool > 0 ? 0 : Math.round(trainPool * testSplit);
    return {
      fit: trainPool - heldOut,
      evalNormal: testNormalPool > 0 ? testNormalPool : heldOut,
      evalAbnormal: abnormal,
      synthetic: stats.synthetic_count ?? 0,
      excluded: stats.excluded_count ?? 0,
      autoHeldOut: testNormalPool === 0,
    };
  }, [stats, testSplit]);

  const handleStart = async () => {
    setStarting(true);
    setErrorMsg(null);
    try {
      const request: AnomalyTrainRequest = {
        algorithm,
        backbone,
        layers: layersText.split(',').map((s) => s.trim()).filter(Boolean),
        coreset_sampling_ratio: coresetRatio,
        image_size: imageSize,
        test_split: testSplit,
      };
      const res = await anomalyTrainingAPI.startTraining(projectId, request);
      onSelectModel(res.model_id);
      onModelsChange();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Failed to start training');
    } finally {
      setStarting(false);
    }
  };

  const handleCancel = async () => {
    if (!trainingModel) return;
    try {
      await anomalyTrainingAPI.cancelTraining(projectId, trainingModel.id);
      onModelsChange();
    } catch (e) {
      console.error('[AnomalyTraining] cancel failed', e);
    }
  };

  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false, title: '', message: '', type: 'danger', onConfirm: null,
  });

  const doDeleteModel = async (m: AnomalyModel) => {
    setDeletingId(m.id);
    try {
      await anomalyTrainingAPI.deleteModel(projectId, m.id);
      onModelsChange();
    } catch (err: any) {
      setErrorMsg(err?.response?.data?.detail || err?.message || 'Failed to delete model');
    } finally {
      setDeletingId(null);
    }
  };

  const handleDeleteModel = (e: MouseEvent, m: AnomalyModel) => {
    e.stopPropagation(); // don't also select the model we're deleting
    setConfirmDialog({
      isOpen: true,
      title: 'Delete model',
      message: `Delete this ${m.algorithm} model and its checkpoint/exports? This can't be undone.`,
      type: 'danger',
      confirmText: 'Delete',
      onConfirm: () => doDeleteModel(m),
    });
  };

  return (
    <div style={{ display: 'flex', gap: 24, flex: 1, minHeight: 0 }}>
      {/* Left: config form + model list */}
      <div style={{ width: 320, display: 'flex', flexDirection: 'column', gap: 16, overflowY: 'auto' }}>
        <div>
          <label className="at-label">Algorithm</label>
          <select className="at-form-select" value={algorithm}
                  onChange={(e) => handleAlgorithmChange(e.target.value as AnomalyAlgorithm)}
                  disabled={!!trainingModel}>
            <option value="patchcore">PatchCore</option>
            <option value="padim">Padim</option>
          </select>
        </div>
        <div>
          <label className="at-label">Backbone</label>
          <select className="at-form-select" value={backbone}
                  onChange={(e) => handleBackboneChange(e.target.value)}
                  disabled={!!trainingModel}>
            {BACKBONE_OPTIONS[algorithm].map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
        {algorithm === 'patchcore' && (
          <div>
            <label className="at-label">Coreset sampling ratio</label>
            <input className="at-form-input" type="number" min={0.01} max={1} step={0.01}
                   value={coresetRatio} onChange={(e) => setCoresetRatio(Number(e.target.value))} disabled={!!trainingModel} />
          </div>
        )}
        <div>
          <label className="at-label">Image size (px)</label>
          <select className="at-form-select" value={imageSize}
                  onChange={(e) => setImageSize(Number(e.target.value))} disabled={!!trainingModel}>
            {IMAGE_SIZE_OPTIONS.map((s) => (
              <option key={s} value={s}>{s} × {s}{s === 256 ? ' (default)' : ''}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="at-label">Test split (held-out normal fraction)</label>
          <input className="at-form-input" type="number" min={0.05} max={0.5} step={0.05}
                 value={testSplit} onChange={(e) => setTestSplit(Number(e.target.value))} disabled={!!trainingModel} />
        </div>

        {projection && (
          <div className="at-train-projection">
            <div className="at-train-projection-line">
              Fits the model on <b>{projection.fit}</b> normal image(s)
              {projection.autoHeldOut && projection.evalNormal > 0 && (
                <> · <b>{projection.evalNormal}</b> auto-held-out for evaluation</>
              )}
            </div>
            <div className="at-train-projection-line">
              Evaluates on <b>{projection.evalNormal}</b> normal + <b>{projection.evalAbnormal}</b> abnormal
              {projection.synthetic > 0 && <> (of which <b>{projection.synthetic}</b> synthetic)</>}
            </div>
            {projection.excluded > 0 && (
              <div className="at-train-projection-line muted">
                {projection.excluded} image(s) excluded from this run
              </div>
            )}
            {projection.evalAbnormal === 0 ? (
              <div className="at-train-projection-warn">
                No abnormal images — AUROC/F1 will be N/A and the threshold degenerates to the
                highest normal score. Generate some on the Synthetic NG tab first.
              </div>
            ) : projection.evalAbnormal < 5 ? (
              <div className="at-train-projection-warn">
                Only {projection.evalAbnormal} abnormal image(s) — the threshold is being fitted to
                very few points, so it will sit almost on top of them.
              </div>
            ) : null}
            {projection.synthetic > 0 && (
              <div className="at-train-projection-line muted">
                Synthetic images calibrate the threshold; they never train the model. Every score is
                re-normalised after training, so recipe thresholds need re-checking.
              </div>
            )}
          </div>
        )}

        {errorMsg && <div className="at-alert-error">{errorMsg}</div>}

        {trainingModel ? (
          <button className="at-btn at-btn-secondary" onClick={handleCancel}>Cancel training</button>
        ) : (
          <button className="at-btn at-btn-primary" onClick={handleStart} disabled={starting}>
            {starting ? 'Starting...' : 'Start Training'}
          </button>
        )}

        <div>
          <label className="at-label">Models</label>
          <div className="at-hint" style={{ marginBottom: 4 }}>
            Click a version to view/export it in the Eval and Export tabs.
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 2 }}>
            {models.length === 0 && <div className="at-hint">No models trained yet.</div>}
            {models.map((m, i) => {
              const version = models.length - i; // oldest = v1, newest = highest
              const selected = selectedModelId === m.id;
              const deletable = m.status !== 'training' && m.status !== 'pending';
              return (
                <div
                  key={m.id}
                  onClick={() => onSelectModel(m.id)}
                  style={{
                    padding: '8px 10px', borderRadius: 6, cursor: 'pointer', fontSize: 12,
                    border: `1px solid ${selected ? '#2563eb' : '#e2e8f0'}`,
                    background: selected ? '#eff6ff' : '#fff',
                    opacity: deletingId === m.id ? 0.5 : 1,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span><b>v{version}</b> · {m.algorithm}</span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ color: STATUS_COLORS[m.status] || '#64748b', fontWeight: 600 }}>{m.status}</span>
                      {deletable && (
                        <button
                          type="button"
                          className="at-cand-card-delete"
                          style={{ flex: '0 0 auto', padding: '1px 7px' }}
                          disabled={deletingId === m.id}
                          onClick={(e) => handleDeleteModel(e, m)}
                          title="Delete this model (frees checkpoint/export files on disk)"
                        >×</button>
                      )}
                    </span>
                  </div>
                  <div className="at-hint">{new Date(m.created_at).toLocaleString()}</div>
                  {m.status === 'completed' && (
                    <div className="at-hint">AUROC {fmtMetric(m.metrics.image_auroc)} · F1 {fmtMetric(m.metrics.image_f1)}</div>
                  )}
                  {selected && <div className="at-hint" style={{ color: '#2563eb', fontWeight: 600 }}>selected for Eval/Export</div>}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Right: live log panel */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <label className="at-label">Training log</label>
        <div className="at-hint" style={{ marginBottom: 10 }}>
          PatchCore/Padim fit in a single pass (coreset/statistics fitting, not iterative gradient
          descent) — no epochs, no early stopping. Progress below tracks pipeline phases instead of
          a per-epoch loss curve.
        </div>
        {liveModel && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{
                height: '100%', width: `${Math.round(liveModel.progress || 0)}%`,
                background: liveModel.status === 'failed' ? '#ef4444'
                  : liveModel.status === 'completed' ? '#22c55e' : '#2563eb',
                transition: 'width .3s ease',
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
              {PHASES.map((p) => {
                const reached = (liveModel.progress ?? 0) >= p.pct;
                const current = liveModel.phase === p.key;
                return (
                  <span key={p.key} style={{
                    fontSize: 10, fontWeight: current ? 700 : 400,
                    color: current ? '#2563eb' : reached ? '#15803d' : '#94a3b8',
                  }}>{p.label}</span>
                );
              })}
            </div>
            <div className="at-hint" style={{ marginTop: 6 }}>
              Phase: <b>{liveModel.phase || '-'}</b> · {Math.round(liveModel.progress || 0)}%
              {liveModel.status === 'training' && <span className="at-loading-spinner" style={{ marginLeft: 8, verticalAlign: 'middle' }} />}
            </div>
          </div>
        )}
        <div className="at-train-log">
          {logs.length === 0 && <div style={{ opacity: .5 }}>No logs yet — start a training run.</div>}
          {logs.map((l) => (
            <div key={l.idx} className={l.level === 'ERROR' ? 'at-log-error' : l.level === 'WARNING' ? 'at-log-warning' : undefined}>
              {l.msg}
            </div>
          ))}
        </div>
      </div>

      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
        confirmText={confirmDialog.confirmText ?? 'Confirm'}
        onClose={() => setConfirmDialog((prev) => ({ ...prev, isOpen: false }))}
        onConfirm={() => confirmDialog.onConfirm?.()}
      />
    </div>
  );
}
