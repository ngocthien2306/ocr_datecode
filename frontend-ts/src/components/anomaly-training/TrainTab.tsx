import { useCallback, useEffect, useRef, useState } from 'react';
import {
  anomalyTrainingAPI, AnomalyAlgorithm, AnomalyModel, AnomalyTrainRequest, TrainLogEntry,
} from '@/services/anomalyTraining';

interface Props {
  projectId: string;
  models: AnomalyModel[];
  onModelsChange: () => void;
  selectedModelId: string | null;
  onSelectModel: (id: string) => void;
}

const BACKBONE_PRESETS: Record<AnomalyAlgorithm, { backbone: string; layers: string[] }> = {
  patchcore: { backbone: 'wide_resnet50_2', layers: ['layer2', 'layer3'] },
  padim: { backbone: 'resnet18', layers: ['layer1', 'layer2', 'layer3'] },
};

const STATUS_COLORS: Record<string, string> = {
  pending: '#94a3b8', training: '#b45309', completed: '#15803d', failed: '#b91c1c', cancelled: '#64748b',
};

export default function TrainTab({ projectId, models, onModelsChange, selectedModelId, onSelectModel }: Props) {
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
          <input className="at-form-input" value={backbone} onChange={(e) => setBackbone(e.target.value)} disabled={!!trainingModel} />
        </div>
        <div>
          <label className="at-label">Layers (comma-separated)</label>
          <input className="at-form-input" value={layersText} onChange={(e) => setLayersText(e.target.value)} disabled={!!trainingModel} />
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
          <input className="at-form-input" type="number" min={64} max={512} step={32}
                 value={imageSize} onChange={(e) => setImageSize(Number(e.target.value))} disabled={!!trainingModel} />
        </div>
        <div>
          <label className="at-label">Test split (held-out normal fraction)</label>
          <input className="at-form-input" type="number" min={0.05} max={0.5} step={0.05}
                 value={testSplit} onChange={(e) => setTestSplit(Number(e.target.value))} disabled={!!trainingModel} />
        </div>

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
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 6 }}>
            {models.length === 0 && <div className="at-hint">No models trained yet.</div>}
            {models.map((m) => (
              <div
                key={m.id}
                onClick={() => onSelectModel(m.id)}
                style={{
                  padding: '8px 10px', borderRadius: 6, cursor: 'pointer', fontSize: 12,
                  border: `1px solid ${selectedModelId === m.id ? '#2563eb' : '#e2e8f0'}`,
                  background: selectedModelId === m.id ? '#eff6ff' : '#fff',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <b>{m.algorithm}</b>
                  <span style={{ color: STATUS_COLORS[m.status] || '#64748b', fontWeight: 600 }}>{m.status}</span>
                </div>
                <div className="at-hint">{new Date(m.created_at).toLocaleString()}</div>
                {m.status === 'completed' && (
                  <div className="at-hint">AUROC {m.metrics.image_auroc.toFixed(3)} · F1 {m.metrics.image_f1.toFixed(3)}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right: live log panel */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <label className="at-label">Training log</label>
        {liveModel && (
          <div className="at-hint" style={{ marginBottom: 6 }}>
            Phase: <b>{liveModel.phase || '-'}</b> · {Math.round(liveModel.progress || 0)}%
            {liveModel.status === 'training' && <span className="at-loading-spinner" style={{ marginLeft: 8, verticalAlign: 'middle' }} />}
          </div>
        )}
        <div style={{
          flex: 1, background: '#0f172a', color: '#cbd5e1', fontFamily: 'monospace', fontSize: 12,
          padding: 12, borderRadius: 8, overflowY: 'auto', minHeight: 200,
        }}>
          {logs.length === 0 && <div style={{ opacity: .5 }}>No logs yet — start a training run.</div>}
          {logs.map((l) => (
            <div key={l.idx} style={{ color: l.level === 'ERROR' ? '#f87171' : l.level === 'WARNING' ? '#fbbf24' : '#cbd5e1' }}>
              {l.msg}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
