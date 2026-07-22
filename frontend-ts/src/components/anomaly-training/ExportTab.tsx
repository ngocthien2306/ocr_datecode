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

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 560 }}>
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
          Builds/loads a TensorRT engine from the ONNX file on this machine and runs one
          inference — confirms the model deploys correctly before wiring it into a recipe.
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
            Inference time: <b>{trtResult.inference_ms} ms</b><br />
            Output shapes: {trtResult.output_shapes.map((s) => `[${s.join(',')}]`).join(', ')}
          </div>
        )}
      </div>
    </div>
  );
}
