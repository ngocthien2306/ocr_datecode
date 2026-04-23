import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  mlTrainingAPI,
  LabeledCrop,
  SyntheticCrop,
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

// ── Lazy image: only renders <img> when it enters the viewport ─────────────
function LazyImage({ src, alt }: { src: string; alt: string }) {
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVisible(true); obs.disconnect(); } },
      { rootMargin: '300px' }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div ref={ref} style={{ width: '52px', height: '36px', background: '#0f1117', borderRadius: '3px' }}>
      {visible && (
        <img
          src={src}
          alt={alt}
          style={{ width: '52px', height: '36px', objectFit: 'contain', display: 'block' }}
        />
      )}
    </div>
  );
}

// ── CropGrid: shared grid used for both real and synthetic crops ───────────
interface CropGridProps {
  items: Array<{ crop_b64: string; label: string }>;
  emptyText: string;
}

function CropGrid({ items, emptyText }: CropGridProps) {
  if (items.length === 0) {
    return (
      <div className="ml-empty-state" style={{ minHeight: '100px' }}>
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.4 }}>
          <path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z"
            stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          <circle cx="7" cy="7" r="1.5" fill="currentColor" />
        </svg>
        <span style={{ fontSize: '12px', color: '#6b7280', marginTop: '6px' }}>{emptyText}</span>
      </div>
    );
  }

  return (
    <div className="ml-crops-grid">
      {items.map((crop, i) => (
        <div key={i} className="ml-crop-card">
          <LazyImage src={`data:image/jpeg;base64,${crop.crop_b64}`} alt={`crop-${i}`} />
          <span className={`ml-label-badge ${crop.label === 'OK' ? 'ok' : 'ng'}`}>
            {crop.label}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────
export default function TrainTab({ project, onRefresh }: Props) {
  // Labeled crops
  const [crops, setCrops] = useState<LabeledCrop[]>([]);
  const [loadingCrops, setLoadingCrops] = useState(false);

  // Labeled crops UI state
  const [cropsTab, setCropsTab] = useState<'real' | 'synthetic'>('real');
  const [cropFilter, setCropFilter] = useState<'all' | 'OK' | 'NG'>('all');

  // Synthetic preview
  const [syntheticCrops, setSyntheticCrops] = useState<SyntheticCrop[]>([]);
  const [loadingSynthetic, setLoadingSynthetic] = useState(false);

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

  // ── Preview synthetic ─────────────────────────────────────────────────
  const handlePreviewSynthetic = async () => {
    if (augmentFactor < 2) return;
    setLoadingSynthetic(true);
    setSyntheticCrops([]);
    setCropsTab('synthetic');
    try {
      const data = await mlTrainingAPI.previewSynthetic(project.id, augmentFactor);
      setSyntheticCrops(data.crops);
    } catch { /* ignore */ }
    finally { setLoadingSynthetic(false); }
  };

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

  // ── Stats ──────────────────────────────────────────────────────────────
  const okCrops = crops.filter(c => c.label === 'OK');
  const ngCrops = crops.filter(c => c.label === 'NG');
  const canTrain = okCrops.length + ngCrops.length >= 2 && !training;

  const filteredCrops =
    cropFilter === 'all' ? crops :
    cropFilter === 'OK'  ? okCrops : ngCrops;

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div className="ml-train-tab">
      {/* ── Left: Config ────────────────────────────────────────────── */}
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

        {/* ── Augmentation ──────────────────────────────────────── */}
        <div className="ml-section-title">Augmentation (NG)</div>
        <div className="ml-form-group">
          <div style={{ fontSize: '11px', color: '#6b7280', marginBottom: '6px' }}>
            Generate synthetic NG from OK samples
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
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
              <button
                className="ml-btn ml-btn-secondary ml-btn-sm"
                onClick={handlePreviewSynthetic}
                disabled={loadingSynthetic}
                title="Preview synthetic NG crops before training"
                style={{ marginLeft: 'auto' }}
              >
                {loadingSynthetic
                  ? <span className="ml-loading-spinner" style={{ width: 12, height: 12, borderWidth: 2 }} />
                  : <>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" />
                    </svg>
                    Preview
                  </>
                }
              </button>
            )}
          </div>
          {augmentFactor >= 2 && (
            <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '4px' }}>
              → {okCrops.length * (augmentFactor - 1)} synthetic NG will be added
            </div>
          )}
        </div>

        {/* ── Train button ──────────────────────────────────────── */}
        <button
          className="ml-btn ml-btn-primary"
          style={{ width: '100%', justifyContent: 'center', padding: '10px' }}
          onClick={handleTrain}
          disabled={!canTrain}
        >
          {training
            ? <><span className="ml-loading-spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Training...</>
            : <><svg width="14" height="14" viewBox="0 0 24 24" fill="none"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor" /></svg> Start Training</>
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

        {/* ── History ───────────────────────────────────────────── */}
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
                    <span style={{ color: m.status === 'completed' ? '#4ade80' : m.status === 'failed' ? '#f87171' : '#fbbf24' }}>
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

      {/* ── Right: Crops + Results ────────────────────────────────────── */}
      <div className="ml-train-right">

        {/* ── Labeled Crops header ───────────────────────────────── */}
        <div style={{ background: '#1a1d27', borderBottom: '1px solid #2d3148', flexShrink: 0 }}>
          {/* Title row */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px 0' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '.05em' }}>
              Labeled Crops
            </span>
            <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={loadCrops} disabled={loadingCrops}>
              {loadingCrops
                ? <span className="ml-loading-spinner" style={{ width: 12, height: 12, borderWidth: 2 }} />
                : <><svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M1 4v6h6M23 20v-6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /><path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg> Refresh</>
              }
            </button>
          </div>

          {/* Inner tabs */}
          <div style={{ display: 'flex', gap: '0', padding: '8px 14px 0' }}>
            {([
              { key: 'real', label: `Real Data (${crops.length})` },
              { key: 'synthetic', label: `Synthetic (${syntheticCrops.length})` },
            ] as const).map(tab => (
              <button
                key={tab.key}
                onClick={() => setCropsTab(tab.key)}
                style={{
                  padding: '5px 12px',
                  fontSize: '12px',
                  border: 'none',
                  borderBottom: cropsTab === tab.key ? '2px solid #3b82f6' : '2px solid transparent',
                  background: 'transparent',
                  color: cropsTab === tab.key ? '#60a5fa' : '#6b7280',
                  cursor: 'pointer',
                  fontWeight: cropsTab === tab.key ? 600 : 400,
                  transition: 'color .12s',
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Filter row — only on Real tab */}
          {cropsTab === 'real' && (
            <div style={{ display: 'flex', gap: '6px', padding: '8px 14px' }}>
              {(['all', 'OK', 'NG'] as const).map(f => (
                <button
                  key={f}
                  onClick={() => setCropFilter(f)}
                  style={{
                    padding: '3px 10px',
                    fontSize: '11px',
                    borderRadius: '12px',
                    border: '1px solid',
                    cursor: 'pointer',
                    transition: 'all .12s',
                    borderColor: cropFilter === f
                      ? (f === 'OK' ? '#4ade80' : f === 'NG' ? '#f87171' : '#3b82f6')
                      : '#2d3148',
                    background: cropFilter === f
                      ? (f === 'OK' ? 'rgba(74,222,128,.12)' : f === 'NG' ? 'rgba(248,113,113,.12)' : 'rgba(59,130,246,.12)')
                      : 'transparent',
                    color: cropFilter === f
                      ? (f === 'OK' ? '#4ade80' : f === 'NG' ? '#f87171' : '#60a5fa')
                      : '#6b7280',
                  }}
                >
                  {f === 'all' ? `All (${crops.length})` : f === 'OK' ? `OK (${okCrops.length})` : `NG (${ngCrops.length})`}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* ── Crops content area ─────────────────────────────────── */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>

          {/* Real tab */}
          {cropsTab === 'real' && (
            loadingCrops
              ? <div className="ml-empty-state"><div className="ml-loading-spinner" /></div>
              : <CropGrid
                  items={filteredCrops}
                  emptyText={
                    crops.length === 0
                      ? 'No labeled characters yet. Go to Label tab to annotate images.'
                      : `No ${cropFilter} crops found.`
                  }
                />
          )}

          {/* Synthetic tab */}
          {cropsTab === 'synthetic' && (
            loadingSynthetic
              ? <div className="ml-empty-state"><div className="ml-loading-spinner" /></div>
              : syntheticCrops.length === 0
                ? (
                  <div className="ml-empty-state" style={{ minHeight: '120px' }}>
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.4 }}>
                      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                    </svg>
                    <span style={{ fontSize: '12px', color: '#6b7280', marginTop: '6px' }}>
                      {augmentFactor < 2
                        ? 'Select an augmentation factor (×2 or higher), then click Preview.'
                        : 'Click Preview in the Augmentation section to generate synthetic NG samples.'}
                    </span>
                  </div>
                )
                : <CropGrid items={syntheticCrops} emptyText="No synthetic crops." />
          )}

          {/* ── Training results ─────────────────────────────────── */}
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
                        <tr><th></th><th>Pred NG</th><th>Pred OK</th></tr>
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
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                    <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                  </svg> Choose Image
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
                      ? <><span className="ml-loading-spinner" style={{ width: 12, height: 12, borderWidth: 2 }} /> Predicting...</>
                      : <><svg width="12" height="12" viewBox="0 0 24 24" fill="none"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor" /></svg> Run</>
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
