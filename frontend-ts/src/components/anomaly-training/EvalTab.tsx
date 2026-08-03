import { useCallback, useEffect, useState } from 'react';
import { anomalyTrainingAPI, AnomalyModel, TestResultsResponse, fmtMetric } from '@/services/anomalyTraining';

interface Props {
  projectId: string;
  model: AnomalyModel | null;
}

export default function EvalTab({ projectId, model }: Props) {
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState<TestResultsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);

  const load = useCallback(async (t: number) => {
    if (!model || model.status !== 'completed') { setResult(null); return; }
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await anomalyTrainingAPI.getTestResults(projectId, model.id, t);
      setResult(res);
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Failed to load test results');
    } finally {
      setLoading(false);
    }
  }, [projectId, model?.id]);

  useEffect(() => {
    setThreshold(model?.metrics?.threshold ?? 0.5);
    load(model?.metrics?.threshold ?? 0.5);
  }, [model?.id]);

  if (!model) {
    return <div className="at-empty-state">Select a model in the Train tab first.</div>;
  }
  if (model.status !== 'completed') {
    return <div className="at-empty-state">This model hasn't finished training yet (status: {model.status}).</div>;
  }

  const [[tn, fp], [fn, tp]] = result?.confusion_matrix ?? [[0, 0], [0, 0]];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, flex: 1, overflowY: 'auto' }}>
      <div className="at-hint">
        Evaluating: <b>{model.algorithm}</b>, trained {new Date(model.created_at).toLocaleString()}
        {' '}(pick a different version in the Train tab's model list)
      </div>

      {result && !result.metrics_available && (
        <div className="at-hint" style={{ color: '#b45309', fontWeight: 600 }}>
          AUROC/F1 are not measurable for this model: its test set contains{' '}
          {result.n_normal_test} normal and 0 abnormal images, and both metrics need
          both classes. Import abnormal crops (Dataset tab) and re-train — that also
          gives the threshold something real to calibrate against; right now it
          degenerates to "worse than the worst normal image seen during training".
        </div>
      )}

      <div className="at-stats-row">
        <div className="at-stat-card">
          <div className="at-stat-value">{fmtMetric(result?.image_auroc)}</div>
          <div className="at-stat-label">Image AUROC</div>
        </div>
        <div className="at-stat-card">
          <div className="at-stat-value">{fmtMetric(result?.image_f1)}</div>
          <div className="at-stat-label">Image F1 @ threshold</div>
        </div>
        <div className="at-stat-card normal">
          <div className="at-stat-value">{result?.n_normal_test ?? 0}</div>
          <div className="at-stat-label">Normal (test)</div>
        </div>
        <div className="at-stat-card abnormal">
          <div className="at-stat-value">{result?.n_abnormal_test ?? 0}</div>
          <div className="at-stat-label">Abnormal (test)</div>
        </div>
      </div>

      <div>
        <label className="at-label">Threshold: {threshold.toFixed(2)}</label>
        <input
          type="range" min={0} max={1} step={0.01} value={threshold} style={{ width: '100%' }}
          onChange={(e) => setThreshold(Number(e.target.value))}
          onMouseUp={() => load(threshold)}
          onTouchEnd={() => load(threshold)}
        />
        <div className="at-hint">Drag and release to recompute metrics — pure read, no retrain.</div>
      </div>

      {errorMsg && <div className="at-alert-error">{errorMsg}</div>}
      {loading && <div className="at-loading-spinner" />}

      {result && (
        <>
          <div>
            <label className="at-label">Confusion matrix</label>
            <table style={{ borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr>
                  <th></th>
                  <th style={{ padding: 6 }}>Pred Normal</th>
                  <th style={{ padding: 6 }}>Pred Abnormal</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: 6, fontWeight: 600 }}>GT Normal</td>
                  <td style={{ padding: 6, textAlign: 'center', background: '#dcfce7' }}>{tn}</td>
                  <td style={{ padding: 6, textAlign: 'center', background: '#fee2e2' }}>{fp}</td>
                </tr>
                <tr>
                  <td style={{ padding: 6, fontWeight: 600 }}>GT Abnormal</td>
                  <td style={{ padding: 6, textAlign: 'center', background: '#fee2e2' }}>{fn}</td>
                  <td style={{ padding: 6, textAlign: 'center', background: '#dcfce7' }}>{tp}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
              <label className="at-label" style={{ marginBottom: 0 }}>Test images ({result.items.length})</label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, cursor: 'pointer' }}>
                <input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} />
                Show anomaly heatmap
              </label>
            </div>
            <div className="at-cand-grid">
              {result.items.map((item, i) => {
                const b64 = showHeatmap && item.heatmap_b64 ? item.heatmap_b64 : item.crop_b64;
                return (
                  <div key={i} className="at-cand-card" style={{ borderColor: item.correct ? '#22c55e' : '#ef4444', borderWidth: 2 }}>
                    <img
                      src={`data:image/jpeg;base64,${b64}`}
                      alt=""
                      className="at-cand-card-img"
                      style={{ cursor: 'zoom-in' }}
                      onClick={() => setLightboxIdx(i)}
                    />
                    <div className="at-cand-card-meta">
                      gt: {item.gt_label} · pred: {item.pred_label}
                      <br />score: {item.pred_score.toFixed(3)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}

      {result && lightboxIdx !== null && (() => {
        const item = result.items[lightboxIdx];
        if (!item) return null;
        const b64 = showHeatmap && item.heatmap_b64 ? item.heatmap_b64 : item.crop_b64;
        return (
          <div className="at-lightbox-overlay" onClick={() => setLightboxIdx(null)}>
            <div className="at-lightbox-card" onClick={(e) => e.stopPropagation()}>
              <div className="at-lightbox-card-header">
                <span style={{ flex: 1, fontSize: 12 }}>
                  gt: <b>{item.gt_label}</b> · pred: <b>{item.pred_label}</b> · score: <b>{item.pred_score.toFixed(3)}</b>
                  {' '}· <span style={{ color: item.correct ? '#15803d' : '#b91c1c' }}>{item.correct ? 'correct' : 'incorrect'}</span>
                </span>
                {!!item.heatmap_b64 && (
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, cursor: 'pointer' }}>
                    <input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} />
                    Heatmap
                  </label>
                )}
                <button
                  type="button"
                  className="at-cand-card-delete"
                  style={{ flex: '0 0 30px' }}
                  disabled={lightboxIdx <= 0}
                  onClick={() => setLightboxIdx((idx) => (idx !== null ? Math.max(0, idx - 1) : idx))}
                  title="Previous"
                >‹</button>
                <button
                  type="button"
                  className="at-cand-card-delete"
                  style={{ flex: '0 0 30px' }}
                  disabled={lightboxIdx >= result.items.length - 1}
                  onClick={() => setLightboxIdx((idx) => (idx !== null ? Math.min(result.items.length - 1, idx + 1) : idx))}
                  title="Next"
                >›</button>
                <button className="at-lightbox-close" onClick={() => setLightboxIdx(null)}>×</button>
              </div>
              <div className="at-lightbox-card-body">
                <img src={`data:image/jpeg;base64,${b64}`} alt="" className="at-lightbox-img" />
              </div>
              <div className="at-lightbox-card-footer">
                {lightboxIdx + 1} / {result.items.length} · {item.image_path}
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
