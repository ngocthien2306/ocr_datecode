import { useCallback, useEffect, useRef, useState } from 'react';
import { anomalyTrainingAPI, AnomalyModel, InferenceEngine, PredictResult, TestResultItem, fmtMetric } from '@/services/anomalyTraining';

interface Props {
  projectId: string;
  model: AnomalyModel | null;
}

function b64ToFile(b64: string, filename: string): File {
  const byteChars = atob(b64);
  const bytes = new Uint8Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) bytes[i] = byteChars.charCodeAt(i);
  return new File([bytes], filename, { type: 'image/jpeg' });
}

/**
 * "Try it" tab — drop a new image (or pick one from the model's own test
 * set) and run it through the exported ONNX live, Roboflow-style. Distinct
 * from Eval, which only replays the training-time test-set predictions.
 */
export default function TestTab({ projectId, model }: Props) {
  const [testSet, setTestSet] = useState<TestResultItem[]>([]);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [previewName, setPreviewName] = useState<string>('');
  const [predicting, setPredicting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const [engineChoice, setEngineChoice] = useState<InferenceEngine>('onnx');
  const [lastFile, setLastFile] = useState<{ file: File; label: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setResult(null);
    setThreshold(model?.metrics?.threshold ?? 0.5);
    setTestSet([]);
    if (!model || model.status !== 'completed') return;
    (async () => {
      try {
        const res = await anomalyTrainingAPI.getTestResults(projectId, model.id);
        setTestSet(res.items.slice(0, 8));
      } catch {
        // Test-set quick-picks are a convenience, not critical -- ignore failures.
      }
    })();
  }, [projectId, model?.id]);

  const runPredict = useCallback(async (file: File, label: string, engine: InferenceEngine) => {
    if (!model) return;
    setPredicting(true);
    setErrorMsg(null);
    setPreviewName(label);
    setLastFile({ file, label });
    try {
      const res = await anomalyTrainingAPI.predictImage(projectId, model.id, file, engine);
      setResult(res);
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Prediction failed');
      setResult(null);
    } finally {
      setPredicting(false);
    }
  }, [projectId, model?.id]);

  // Re-run the same image when the engine choice changes, so switching
  // ONNX <-> TensorRT is a direct speed comparison on identical input.
  const handleEngineChange = (next: InferenceEngine) => {
    setEngineChoice(next);
    if (lastFile) runPredict(lastFile.file, lastFile.label, next);
  };

  if (!model) {
    return <div className="at-empty-state">Select a model in the Train tab first.</div>;
  }
  if (model.status !== 'completed') {
    return <div className="at-empty-state">This model hasn't finished training yet (status: {model.status}).</div>;
  }
  if (!model.onnx_path) {
    return (
      <div className="at-empty-state">
        Export ONNX in the Export tab first — Test runs live inference through the exported model.
      </div>
    );
  }

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) runPredict(file, file.name, engineChoice);
  };

  const trtReady = !!model.engine_path;

  const predLabel = result && result.pred_score >= threshold ? 'abnormal' : 'normal';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, flex: 1, minHeight: 0 }}>
      <div className="at-hint">
        Testing: <b>{model.algorithm}</b>, trained {new Date(model.created_at).toLocaleString()}
        {' '}· AUROC {fmtMetric(model.metrics.image_auroc)} · F1 {fmtMetric(model.metrics.image_f1)}
        {' '}(pick a different version in the Train tab's model list)
      </div>

      <div style={{ display: 'flex', gap: 16, flex: 1, minHeight: 0 }}>
        {/* Left: test-set quick picks + upload */}
        <div style={{ width: 220, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 12, overflowY: 'auto' }}>
          <div
            className={`at-test-dropzone ${dragOver ? 'dragover' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
          >
            Drop an image or click to select
            <input
              ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }}
              onChange={(e) => handleFiles(e.target.files)}
            />
          </div>

          {testSet.length > 0 && (
            <div>
              <label className="at-label">Test set (quick pick)</label>
              <div className="at-test-quickpicks">
                {testSet.map((item, i) => (
                  <img
                    key={i}
                    src={`data:image/jpeg;base64,${item.crop_b64}`}
                    alt=""
                    className="at-test-quickpick-img"
                    onClick={() => runPredict(
                      b64ToFile(item.crop_b64, `test-set-${i}.jpg`),
                      `test set #${i + 1} (gt: ${item.gt_label})`,
                      engineChoice,
                    )}
                  />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Center: image + heatmap + badge */}
        <div className="at-test-center">
          {predicting ? (
            <div className="at-loading-spinner" />
          ) : result ? (
            <>
              <span className={`at-test-badge ${predLabel}`}>
                {predLabel === 'abnormal' ? 'Anomaly detected' : 'Normal'}
              </span>
              <img
                src={`data:image/jpeg;base64,${showHeatmap && result.heatmap_b64 ? result.heatmap_b64 : result.crop_b64}`}
                alt=""
                style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', borderRadius: 6 }}
              />
            </>
          ) : (
            <div style={{ color: '#94a3b8', fontSize: 13 }}>Drop an image or pick from the test set to try the model.</div>
          )}
        </div>

        {/* Right: engine/threshold/heatmap controls + JSON output */}
        <div style={{ width: 280, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 14, overflowY: 'auto' }}>
          <div>
            <label className="at-label">Inference engine</label>
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                type="button"
                className={`at-cand-toggle normal ${engineChoice === 'onnx' ? 'active' : ''}`}
                disabled={predicting}
                onClick={() => handleEngineChange('onnx')}
              >ONNX</button>
              <button
                type="button"
                className={`at-cand-toggle normal ${engineChoice === 'tensorrt' ? 'active' : ''}`}
                disabled={predicting || !trtReady}
                title={!trtReady ? 'Export TensorRT in the Export tab (step 3) first' : undefined}
                onClick={() => handleEngineChange('tensorrt')}
              >TensorRT</button>
            </div>
            {!trtReady && <div className="at-hint" style={{ marginTop: 4 }}>Export TensorRT (Export tab, step 3) to compare speed.</div>}
            {result && (
              <div className="at-hint" style={{ marginTop: 6 }}>
                Inference time (<b>{result.engine}</b>, {result.active_provider}):{' '}
                <b style={{ color: '#2563eb', fontSize: 13 }}>{result.inference_ms} ms</b>
              </div>
            )}
          </div>
          <div>
            <label className="at-label">Threshold: {threshold.toFixed(2)}</label>
            <input
              type="range" min={0} max={1} step={0.01} value={threshold} style={{ width: '100%' }}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
            <div className="at-hint">Client-side only — reclassifies the current score, no re-inference needed.</div>
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, cursor: 'pointer' }}>
            <input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} />
            Show anomaly heatmap
          </label>

          {errorMsg && <div className="at-alert-error">{errorMsg}</div>}

          {result && (
            <div>
              <label className="at-label">Output</label>
              <pre className="at-test-json">{JSON.stringify({
                file: previewName,
                pred_score: result.pred_score,
                pred_label: predLabel,
                threshold,
                image_size: result.image_size,
                engine: result.engine,
                provider: result.active_provider,
                inference_ms: result.inference_ms,
              }, null, 2)}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
