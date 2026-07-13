import { useCallback, useEffect, useState } from 'react';
import { anomalyTrainingAPI, AnomalyModel, TestResultsResponse } from '@/services/anomalyTraining';

interface Props {
  projectId: string;
  model: AnomalyModel | null;
}

export default function EvalTab({ projectId, model }: Props) {
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState<TestResultsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

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
      <div className="at-stats-row">
        <div className="at-stat-card">
          <div className="at-stat-value">{(result?.image_auroc ?? 0).toFixed(3)}</div>
          <div className="at-stat-label">Image AUROC</div>
        </div>
        <div className="at-stat-card">
          <div className="at-stat-value">{(result?.image_f1 ?? 0).toFixed(3)}</div>
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
            <label className="at-label">Test images ({result.items.length})</label>
            <div className="at-cand-grid">
              {result.items.map((item, i) => (
                <div key={i} className="at-cand-card" style={{ borderColor: item.correct ? '#22c55e' : '#ef4444', borderWidth: 2 }}>
                  <img src={`data:image/jpeg;base64,${item.crop_b64}`} alt="" className="at-cand-card-img" />
                  <div className="at-cand-card-meta">
                    gt: {item.gt_label} · pred: {item.pred_label}
                    <br />score: {item.pred_score.toFixed(3)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
