import { useCallback, useEffect, useState } from 'react';
import {
  ocrTrainingModelAPI, OCRArtifact, OCREngineInfo, OCRModel,
} from '@/services/ocrTraining';

interface Props {
  projectId: string;
  model: OCRModel | null;
  onModelChange: () => void;
}

const ARTIFACTS: Array<{ id: OCRArtifact; label: string; field: keyof OCRModel; hint: string }> = [
  { id: 'engine', label: 'TensorRT .engine', field: 'engine_path',
    hint: 'What a recipe loads at inference time' },
  { id: 'onnx_fp16', label: 'ONNX fp16', field: 'onnx_fp16_path',
    hint: 'Source the engine is built from; also the ONNX fallback backend' },
  { id: 'onnx', label: 'ONNX fp32', field: 'onnx_path', hint: 'Unquantised graph' },
  { id: 'dict', label: 'Character dict', field: 'dict_path',
    hint: 'Must travel with the engine — a decoder built from the wrong dict shifts every index' },
  { id: 'checkpoint', label: 'Checkpoint .pth', field: 'checkpoint_path',
    hint: 'Fine-tune onward from this, or re-export it' },
];

export default function ExportTab({ projectId, model, onModelChange }: Props) {
  const [info, setInfo] = useState<OCREngineInfo | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const loadInfo = useCallback(async () => {
    if (!model?.engine_path) { setInfo(null); return; }
    try {
      setInfo(await ocrTrainingModelAPI.inspectExport(projectId, model.id));
    } catch (e: any) {
      // An engine built by a different TensorRT major will not deserialize; that
      // is exactly what this call is for, so surface it rather than hiding it.
      setError(e?.response?.data?.detail || 'Could not inspect the engine');
      setInfo(null);
    }
  }, [projectId, model?.id, model?.engine_path]);

  useEffect(() => { setError(null); setMessage(null); loadInfo(); }, [loadInfo]);

  if (!model) {
    return <div className="at-empty-state" style={{ padding: 40 }}>
      Select a run in the Train tab first.
    </div>;
  }

  const act = async (label: string, fn: () => Promise<any>, describe: (r: any) => string) => {
    setBusy(label);
    setError(null);
    setMessage(null);
    try {
      const r = await fn();
      setMessage(describe(r));
      onModelChange();
      await loadInfo();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || `${label} failed`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <>
      {error && <div className="at-alert-error">{error}</div>}
      {message && <div className="at-alert-ok">{message}</div>}

      <div className="at-hint" style={{ marginBottom: 10 }}>
        Both exports run automatically at the end of a successful run. These buttons are for
        re-exporting — after deleting an artifact, or when the machine's TensorRT version
        changes and existing engines stop deserializing.
      </div>

      <div className="ot-toolbar">
        <button className="at-btn at-btn-secondary" disabled={busy !== null || model.status !== 'completed'}
                onClick={() => act('ONNX export',
                  () => ocrTrainingModelAPI.exportOnnx(projectId, model.id),
                  (r) => `ONNX written — gtc ${JSON.stringify(r.gtc_shape)}, ctc ${JSON.stringify(r.ctc_shape)}`)}>
          {busy === 'ONNX export' ? 'Exporting…' : 'Re-export ONNX'}
        </button>
        <button className="at-btn at-btn-secondary"
                disabled={busy !== null || !model.onnx_fp16_path}
                title={!model.onnx_fp16_path ? 'Export ONNX first — the engine is built from it' : undefined}
                onClick={() => act('TensorRT build',
                  () => ocrTrainingModelAPI.exportTensorRT(projectId, model.id, true),
                  (r) => `Engine built — ${r.size_mb} MB, outputs ${r.outputs.map((o: any) => o.name).join(', ')}`)}>
          {busy === 'TensorRT build' ? 'Building…' : 'Rebuild TensorRT engine'}
        </button>
        <span className="ot-spacer" />
        <span className="at-hint">
          The build takes the GPU lock, so it queues behind any active training run.
        </span>
      </div>

      <h3 style={{ margin: '10px 0 6px' }}>Artifacts</h3>
      {ARTIFACTS.map((a) => {
        const present = !!model[a.field];
        return (
          <div key={a.id} className={`ot-label-row ${present ? 'is-verified' : 'is-need_review'}`}
               style={{ gridTemplateColumns: '1fr 160px' }}>
            <div className="ot-label-fields">
              <div style={{ display: 'flex', gap: 8, alignItems: 'baseline' }}>
                <b>{a.label}</b>
                <span className={`ot-chip-sm ${present ? 'ot-pass' : 'ot-fail'}`}>
                  {present ? 'present' : 'not exported'}
                </span>
              </div>
              <div className="ot-hint-row"><span>{a.hint}</span></div>
              {present && (
                <div className="ot-hint-row" style={{ wordBreak: 'break-all' }}>
                  <span>{String(model[a.field])}</span>
                </div>
              )}
            </div>
            <div className="ot-row-actions">
              <button className="at-btn at-btn-secondary at-btn-sm" disabled={!present}
                      onClick={() => ocrTrainingModelAPI.download(projectId, model.id, a.id)
                        .catch((e) => setError(e?.message || 'Download failed'))}>
                Download
              </button>
            </div>
          </div>
        );
      })}

      {info && (
        <>
          <h3 style={{ margin: '14px 0 6px' }}>Engine bindings</h3>
          {!info.runtime_compatible && (
            <div className="at-alert-error">
              This engine exposes {info.outputs.length} outputs. ai_services' SMTR backend
              asserts exactly two (gtc_logits, ctc_logits), so it would fail at load time on
              the line. Rebuild from the two-output ONNX.
            </div>
          )}
          <div className="ot-prepare-grid">
            <div className="ot-stat-box">
              <div className="ot-stat-num">{info.size_mb} MB</div>
              <div className="ot-stat-label">engine size</div>
            </div>
            <div className={`ot-stat-box ${info.runtime_compatible ? '' : 'ot-bad'}`}>
              <div className="ot-stat-num">{info.outputs.length}</div>
              <div className="ot-stat-label">outputs (must be 2)</div>
            </div>
            <div className="ot-stat-box">
              <div className="ot-stat-num">{model.vocab_size}</div>
              <div className="ot-stat-label">vocab ({model.use_space_char ? 'with space' : 'no space'})</div>
            </div>
          </div>
          <div style={{ fontFamily: 'ui-monospace, Menlo, monospace', fontSize: 11, color: '#475569' }}>
            {info.inputs.map((i) => (
              <div key={i.name}>
                in <b>{i.name}</b> {JSON.stringify(i.shape)}
                {i.profile && <> · profile min {JSON.stringify(i.profile[0])} opt{' '}
                  {JSON.stringify(i.profile[1])} max {JSON.stringify(i.profile[2])}</>}
              </div>
            ))}
            {info.outputs.map((o) => (
              <div key={o.name}>out <b>{o.name}</b> {JSON.stringify(o.shape)}</div>
            ))}
          </div>
          <div className="at-hint" style={{ marginTop: 8 }}>
            −1 in a shape is a dynamic axis. The width axis stays dynamic so one engine
            handles any crop width inside its profile; batch is dynamic up to 16.
          </div>
        </>
      )}
    </>
  );
}
