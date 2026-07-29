import { useState } from 'react';
import { anomalyTrainingAPI, AnomalyModel, VerifyTensorRTResult } from '@/services/anomalyTraining';

interface Props {
  projectId: string;
  model: AnomalyModel | null;
  onModelChange: () => void;
}

export default function ExportTab({ projectId, model, onModelChange }: Props) {
  const [exporting, setExporting] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [exportingEngine, setExportingEngine] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [trtResult, setTrtResult] = useState<VerifyTensorRTResult | null>(null);

  if (!model) {
    return <div className="at-empty-state">Select a model in the Train tab first.</div>;
  }
  if (model.status !== 'completed') {
    return <div className="at-empty-state">This model hasn't finished training yet (status: {model.status}).</div>;
  }

  const handleExport = async () => {
    setExporting(true);
    setErrorMsg(null);
    try {
      await anomalyTrainingAPI.exportOnnx(projectId, model.id);
      onModelChange();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Export failed');
    } finally {
      setExporting(false);
    }
  };

  const handleVerifyTrt = async () => {
    setVerifying(true);
    setErrorMsg(null);
    setTrtResult(null);
    try {
      const res = await anomalyTrainingAPI.verifyTensorRT(projectId, model.id);
      setTrtResult(res);
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'TensorRT verify failed');
    } finally {
      setVerifying(false);
    }
  };

  const handleDownload = async () => {
    try {
      await anomalyTrainingAPI.downloadOnnx(projectId, model.id);
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Download failed');
    }
  };

  const handleExportEngine = async () => {
    setExportingEngine(true);
    setErrorMsg(null);
    try {
      await anomalyTrainingAPI.exportTensorRT(projectId, model.id);
      onModelChange();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'TensorRT export failed');
    } finally {
      setExportingEngine(false);
    }
  };

  const handleDownloadEngine = async () => {
    try {
      await anomalyTrainingAPI.downloadEngine(projectId, model.id);
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Download failed');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 560 }}>
      <div className="at-hint">
        Exporting: <b>{model.algorithm}</b>, trained {new Date(model.created_at).toLocaleString()}
        {' '}(pick a different version in the Train tab's model list)
      </div>

      {errorMsg && <div className="at-alert-error">{errorMsg}</div>}

      <div className="at-stat-card">
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>1. Export ONNX</div>
        <div className="at-hint" style={{ marginBottom: 10 }}>
          {model.onnx_path ? `Exported: ${model.onnx_path}` : 'Not exported yet.'}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="at-btn at-btn-primary" onClick={handleExport} disabled={exporting}>
            {exporting ? 'Exporting...' : model.onnx_path ? 'Re-export ONNX' : 'Export ONNX'}
          </button>
          {model.onnx_path && (
            <button className="at-btn at-btn-secondary" onClick={handleDownload}>Download .onnx</button>
          )}
        </div>
      </div>

      <div className="at-stat-card">
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>2. Verify TensorRT</div>
        <div className="at-hint" style={{ marginBottom: 10 }}>
          Builds (or reuses) the standalone .engine from the ONNX file and confirms it
          deserializes correctly on this machine's GPU — a quick sanity check before
          downloading it in step 3.
        </div>
        <button className="at-btn at-btn-primary" onClick={handleVerifyTrt} disabled={verifying || !model.onnx_path}>
          {verifying ? 'Verifying...' : 'Verify TensorRT'}
        </button>
        {!model.onnx_path && <div className="at-hint" style={{ marginTop: 6 }}>Export ONNX first.</div>}
        {trtResult && (
          <div className="at-hint" style={{ marginTop: 10, lineHeight: 1.8 }}>
            Provider: <b>{trtResult.active_provider}</b><br />
            Engine cache: <b>{trtResult.engine_cache_hit ? 'HIT (reused)' : 'MISS (built now)'}</b><br />
            Build/load time: <b>{trtResult.build_or_load_ms} ms</b><br />
            Output shapes: {trtResult.output_shapes.map((s) => `[${s.join(',')}]`).join(', ')}
          </div>
        )}
      </div>

      <div className="at-stat-card">
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>3. Export TensorRT (.engine)</div>
        <div className="at-hint" style={{ marginBottom: 10 }}>
          {model.engine_path
            ? `Exported: ${model.engine_path}`
            : 'Builds a standalone .engine file (dynamic batch 1-8, same fixed size as training) '
              + 'you can copy anywhere and load directly with tensorrt.Runtime — same format as '
              + 'the OCR/pipeline engines in weights/, unlike the internal cache above.'}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="at-btn at-btn-primary" onClick={handleExportEngine}
                  disabled={exportingEngine || !model.onnx_path}>
            {exportingEngine ? 'Building engine...' : model.engine_path ? 'Re-export .engine' : 'Export .engine'}
          </button>
          {model.engine_path && (
            <button className="at-btn at-btn-secondary" onClick={handleDownloadEngine}>Download .engine</button>
          )}
        </div>
        {!model.onnx_path && <div className="at-hint" style={{ marginTop: 6 }}>Export ONNX first.</div>}
      </div>
    </div>
  );
}
