import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  mlTrainingAPI,
  LabeledCrop,
  MLModel,
  MLProject,
  TrainRequest,
} from '@/services/mlTraining';

interface Props {
  project: MLProject;
  onRefresh: () => void;
}

const AUGMENT_OPTIONS = [
  { value: 0, label: 'Off' },
  { value: 2, label: '×2' },
  { value: 3, label: '×3' },
  { value: 4, label: '×4' },
  { value: 5, label: '×5' },
];

export default function TrainTab({ project, onRefresh }: Props) {
  // Labeled crops preview
  const [crops, setCrops] = useState<LabeledCrop[]>([]);
  const [loadingCrops, setLoadingCrops] = useState(false);

  // Training config
  const [algorithm, setAlgorithm] = useState<'rf' | 'svm' | 'mlp'>('rf');
  const [augmentFactor, setAugmentFactor] = useState(0);
  const [nEstimators, setNEstimators] = useState(100);
  const [maxIter, setMaxIter] = useState(500);
  const [svmC, setSvmC] = useState(1.0);

  // Training state
  const [training, setTraining] = useState(false);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [models, setModels] = useState<MLModel[]>([]);
  const [latestModel, setLatestModel] = useState<MLModel | null>(null);

  // Poll interval ref
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Predict
  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictResults, setPredictResults] = useState<any[] | null>(null);
  const predictInputRef = useRef<HTMLInputElement>(null);

  // ── Load crops ────────────────────────────────────────────────────────
  const loadCrops = useCallback(async () => {
    setLoadingCrops(true);
    try {
      const data = await mlTrainingAPI.getLabeledCrops(project.id);
      setCrops(data.crops);
    } catch { /* ignore */ }
    finally { setLoadingCrops(false); }
  }, [project.id]);

  // ── Load models ───────────────────────────────────────────────────────
  const loadModels = useCallback(async () => {
    try {
      const list = await mlTrainingAPI.listModels(project.id);
      setModels(list);
      const completed = list.find(m => m.status === 'completed');
      setLatestModel(completed ?? null);
    } catch { /* ignore */ }
  }, [project.id]);

  useEffect(() => {
    loadCrops();
    loadModels();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [project.id]);

  // ── Start training ────────────────────────────────────────────────────
  const handleTrain = async () => {
    setTraining(true);
    setPredictResults(null);
    try {
      const req: TrainRequest = {
        algorithm,
        augment_factor: augmentFactor,
        n_estimators: nEstimators,
        max_iter: maxIter,
        C: svmC,
      };
      const { model_id } = await mlTrainingAPI.startTraining(project.id, req);
      setActiveModelId(model_id);

      // Poll for completion
      pollRef.current = setInterval(async () => {
        try {
          const model = await mlTrainingAPI.getModelStatus(project.id, model_id);
          if (model.status === 'completed' || model.status === 'failed') {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            setTraining(false);
            setActiveModelId(null);
            await loadModels();
            await onRefresh();
          }
        } catch { /* ignore */ }
      }, 2000);
    } catch (e: any) {
      console.error('Training failed', e);
      setTraining(false);
      alert(e?.response?.data?.detail ?? 'Training failed');
    }
  };

  // ── Predict ───────────────────────────────────────────────────────────
  const handlePredict = async () => {
    if (!predictFile || !latestModel) return;
    setPredicting(true);
    setPredictResults(null);
    try {
      const res = await mlTrainingAPI.predict(project.id, predictFile);
      setPredictResults(res.results);
    } catch (e: any) {
      alert(e?.response?.data?.detail ?? 'Prediction failed');
    } finally {
      setPredicting(false);
    }
  };

  // ── Stats from crops ──────────────────────────────────────────────────
  const okCrops = crops.filter(c => c.label === 'OK');
  const ngCrops = crops.filter(c => c.label === 'NG');
  const canTrain = okCrops.length + ngCrops.length >= 2 && !training;

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div className="ml-train-tab">
      {/* Left: Config */}
      <div className="ml-train-left">
        <div className="ml-section-title">Dataset</div>
        <div className="ml-metric-row">
          <div className="ml-metric-card">
            <div className="ml-metric-value" style={{ color: '#4ade80' }}>{okCrops.length}</div>
            <div className="ml-metric-label">OK samples</div>
          </div>
          <div className="ml-metric-card">
            <div className="ml-metric-value" style={{ color: '#f87171' }}>{ngCrops.length}</div>
            <div className="ml-metric-label">NG samples</div>
          </div>
        </div>

        <div className="ml-section-title">Algorithm</div>
        <div className="ml-form-group">
          {(['rf', 'svm', 'mlp'] as const).map(alg => (
            <label key={alg} style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '13px', color: algorithm === alg ? '#e2e8f0' : '#6b7280' }}>
              <input
                type="radio"
                name="algorithm"
                value={alg}
                checked={algorithm === alg}
                onChange={() => setAlgorithm(alg)}
              />
              {alg === 'rf' ? 'Random Forest' : alg === 'svm' ? 'SVM (RBF kernel)' : 'Neural Net (MLP)'}
            </label>
          ))}
        </div>

        {/* Algorithm-specific params */}
        {algorithm === 'rf' && (
          <div className="ml-form-group">
            <label className="ml-form-label">N estimators</label>
            <input
              className="ml-form-input"
              type="number"
              min={10} max={500} step={10}
              value={nEstimators}
              onChange={e => setNEstimators(Number(e.target.value))}
            />
          </div>
        )}
        {(algorithm === 'svm' || algorithm === 'mlp') && (
          <div className="ml-form-group">
            <label className="ml-form-label">{algorithm === 'svm' ? 'C (regularization)' : 'Max iterations'}</label>
            <input
              className="ml-form-input"
              type="number"
              value={algorithm === 'svm' ? svmC : maxIter}
              onChange={e => algorithm === 'svm' ? setSvmC(Number(e.target.value)) : setMaxIter(Number(e.target.value))}
            />
          </div>
        )}

        <div className="ml-section-title">Augmentation (NG)</div>
        <div className="ml-form-group">
          <div style={{ fontSize: '11px', color: '#6b7280', marginBottom: '4px' }}>
            Generate synthetic NG from OK samples
          </div>
          <div className="ml-augment-options">
            {AUGMENT_OPTIONS.map(opt => (
              <button
                key={opt.value}
                className={`ml-augment-chip ${augmentFactor === opt.value ? 'selected' : ''}`}
                onClick={() => setAugmentFactor(opt.value)}
              >
                {opt.label}
              </button>
            ))}
          </div>
          {augmentFactor >= 2 && (
            <div style={{ fontSize: '11px', color: '#6b7280' }}>
              → {okCrops.length * (augmentFactor - 1)} synthetic NG added
            </div>
          )}
        </div>

        <button
          className="ml-btn ml-btn-primary"
          style={{ width: '100%', justifyContent: 'center', padding: '10px' }}
          onClick={handleTrain}
          disabled={!canTrain}
        >
          {training
            ? <><span className="ml-loading-spinner" style={{width:14,height:14,borderWidth:2}}/> Training...</>
            : <><svg width="14" height="14" viewBox="0 0 24 24" fill="none"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/></svg> Start Training</>
          }
        </button>

        {!canTrain && !training && (
          <div style={{ fontSize: '11px', color: '#6b7280', textAlign: 'center' }}>
            Need at least 2 labeled samples in Label tab
          </div>
        )}

        {training && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div style={{ fontSize: '12px', color: '#9ca3af' }}>Training in progress...</div>
            <div className="ml-progress-bar-wrapper">
              <div className="ml-progress-bar" />
            </div>
          </div>
        )}

        {/* Models history */}
        {models.length > 0 && (
          <>
            <div className="ml-section-title">History</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {models.slice(0, 5).map(m => (
                <div
                  key={m.id}
                  style={{ background: '#1a1d27', border: '1px solid #2d3148', borderRadius: '6px', padding: '6px 8px', fontSize: '11px' }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#9ca3af' }}>{m.algorithm.toUpperCase()}</span>
                    <span style={{
                      color: m.status === 'completed' ? '#4ade80' : m.status === 'failed' ? '#f87171' : '#fbbf24',
                    }}>
                      {m.status}
                    </span>
                  </div>
                  {m.status === 'completed' && (
                    <div style={{ color: '#6b7280', marginTop: '2px' }}>
                      Test acc: {(m.metrics.accuracy_test * 100).toFixed(1)}%
                    </div>
                  )}
                  {m.status === 'failed' && m.error && (
                    <div style={{ color: '#f87171', marginTop: '2px', wordBreak: 'break-word' }}>{m.error}</div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Right: Results + Preview + Test */}
      <div className="ml-train-right">
        {/* Character crops preview */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: '#1a1d27', borderBottom: '1px solid #2d3148', flexShrink: 0 }}>
          <span style={{ fontSize: '12px', fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '.05em' }}>
            Labeled Crops
          </span>
          <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={loadCrops} disabled={loadingCrops}>
            {loadingCrops
              ? <span className="ml-loading-spinner" style={{width:12,height:12,borderWidth:2}}/>
              : <><svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M1 4v6h6M23 20v-6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg> Refresh</>
            }
          </button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
          {/* Crops grid */}
          {loadingCrops ? (
            <div className="ml-empty-state"><div className="ml-loading-spinner" /></div>
          ) : crops.length === 0 ? (
            <div className="ml-empty-state">
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" style={{opacity:.4}}><path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/><circle cx="7" cy="7" r="1.5" fill="currentColor"/></svg>
              No labeled characters yet. Go to Label tab to annotate images.
            </div>
          ) : (
            <div className="ml-crops-grid">
              {crops.map((crop, i) => (
                <div key={i} className="ml-crop-card">
                  <img src={`data:image/jpeg;base64,${crop.crop_b64}`} alt={`char-${i}`} />
                  <span className={`ml-label-badge ${crop.label === 'OK' ? 'ok' : 'ng'}`}>
                    {crop.label}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Training results */}
          {latestModel && latestModel.status === 'completed' && (
            <div className="ml-results-panel">
              <div className="ml-section-title">Training Results — {latestModel.algorithm.toUpperCase()}</div>

              <div className="ml-metric-row">
                <div className="ml-metric-card">
                  <div className="ml-metric-value">{(latestModel.metrics.accuracy_train * 100).toFixed(1)}%</div>
                  <div className="ml-metric-label">Train accuracy</div>
                </div>
                <div className="ml-metric-card">
                  <div className="ml-metric-value" style={{ color: '#4ade80' }}>{(latestModel.metrics.accuracy_test * 100).toFixed(1)}%</div>
                  <div className="ml-metric-label">Test accuracy</div>
                </div>
                <div className="ml-metric-card">
                  <div className="ml-metric-value" style={{ color: '#9ca3af', fontSize: '16px' }}>
                    {latestModel.metrics.n_ok} / {latestModel.metrics.n_ng}
                  </div>
                  <div className="ml-metric-label">OK / NG samples</div>
                </div>
              </div>

              {latestModel.metrics.confusion_matrix.length > 0 && (
                <div>
                  <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '6px', fontWeight: 600 }}>Confusion Matrix</div>
                  <div className="ml-confusion-matrix">
                    <table>
                      <thead>
                        <tr>
                          <th></th>
                          <th>Pred NG</th>
                          <th>Pred OK</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <th>Actual NG</th>
                          {latestModel.metrics.confusion_matrix[0]?.map((v, i) => (
                            <td key={i} style={{ color: i === 0 ? '#4ade80' : '#f87171' }}>{v}</td>
                          ))}
                        </tr>
                        <tr>
                          <th>Actual OK</th>
                          {latestModel.metrics.confusion_matrix[1]?.map((v, i) => (
                            <td key={i} style={{ color: i === 1 ? '#4ade80' : '#f87171' }}>{v}</td>
                          ))}
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {latestModel.metrics.report && (
                <div>
                  <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '6px', fontWeight: 600 }}>Classification Report</div>
                  <pre className="ml-report-pre">{latestModel.metrics.report}</pre>
                </div>
              )}

              {/* Test section */}
              <div className="ml-section-title" style={{ marginTop: '8px' }}>Test Prediction</div>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <button
                  className="ml-btn ml-btn-secondary ml-btn-sm"
                  onClick={() => predictInputRef.current?.click()}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/></svg> Choose Image
                </button>
                {predictFile && (
                  <span style={{ fontSize: '11px', color: '#9ca3af' }}>{predictFile.name}</span>
                )}
                {predictFile && (
                  <button
                    className="ml-btn ml-btn-primary ml-btn-sm"
                    onClick={handlePredict}
                    disabled={predicting}
                  >
                    {predicting
                      ? <><span className="ml-loading-spinner" style={{width:12,height:12,borderWidth:2}}/> Predicting...</>
                      : <><svg width="12" height="12" viewBox="0 0 24 24" fill="none"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/></svg> Run</>
                    }
                  </button>
                )}
                <input
                  ref={predictInputRef}
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={e => {
                    setPredictFile(e.target.files?.[0] ?? null);
                    setPredictResults(null);
                  }}
                />
              </div>

              {predictResults && (
                <div>
                  <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '6px' }}>
                    {predictResults.length} characters detected
                  </div>
                  <div className="ml-predict-results-grid">
                    {predictResults.map((r, i) => (
                      <div key={i} className={`ml-predict-card ${r.label.toLowerCase()}`}>
                        <img src={`data:image/jpeg;base64,${r.crop_b64}`} alt={`pred-${i}`} />
                        <span className={`ml-label-badge ${r.label === 'OK' ? 'ok' : 'ng'}`}>{r.label}</span>
                        <span className="ml-predict-prob">{(r.prob_ok * 100).toFixed(0)}% OK</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
