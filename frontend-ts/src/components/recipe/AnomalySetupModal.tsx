import { useEffect, useState } from 'react';
import { anomalyTrainingAPI, AnomalyProject, AnomalyModel, fmtMetric } from '@/services/anomalyTraining';
import '@/styles/AnomalySetupModal.css';

interface AnomalyConfig {
  enabled: boolean;
  anomaly_project_id?: string | null;
  anomaly_model_id?: string | null;
  onnx_path?: string | null;
  engine_path?: string | null;
  image_size?: number;
  threshold?: number;
  min_area?: number;
  min_region_area?: number;
  max_region_area?: number;
  pixel_threshold?: number | null;
}

interface Props {
  isOpen: boolean;
  templateName: string;
  initialConfig: AnomalyConfig | null;
  onClose: () => void;
  onSave: (config: AnomalyConfig) => void;
}

/**
 * Pick an anomaly_service project + a specific trained (and exported) model
 * version for this template's label-defect check. Replaces the wrinkle
 * check when `enabled` — see ai_services/.../anomaly_inference.py.
 *
 * The model's onnx_path/image_size are captured here at selection time and
 * stored directly on the recipe, so ai_services never has to call
 * anomaly_service at inference time — just reads a plain file path.
 */
export default function AnomalySetupModal({ isOpen, templateName, initialConfig, onClose, onSave }: Props) {
  const [projects, setProjects] = useState<AnomalyProject[]>([]);
  const [models, setModels] = useState<AnomalyModel[]>([]);
  const [projectId, setProjectId] = useState(initialConfig?.anomaly_project_id || '');
  const [modelId, setModelId] = useState(initialConfig?.anomaly_model_id || '');
  const [threshold, setThreshold] = useState(initialConfig?.threshold ?? 0.5);
  const [enabled, setEnabled] = useState(initialConfig?.enabled ?? false);
  const [minArea, setMinArea] = useState(initialConfig?.min_area ?? 0);
  const [minRegionArea, setMinRegionArea] = useState(initialConfig?.min_region_area ?? 0);
  const [maxRegionArea, setMaxRegionArea] = useState(initialConfig?.max_region_area ?? 0);
  const [pixelThreshold, setPixelThreshold] = useState(initialConfig?.pixel_threshold ?? 0.5);
  // Area-based logic is its own opt-in on top of `enabled` — off by default,
  // and inferred from whatever was already saved so re-opening the modal
  // shows the right state. Turning it off clears min_area/max_region_area/
  // pixel_threshold on save (area logic fully off, pure image-level
  // threshold — see backend AnomalyConfig docstring).
  const [areaLogicOn, setAreaLogicOn] = useState(
    !!(initialConfig && ((initialConfig.min_area ?? 0) > 0 || (initialConfig.max_region_area ?? 0) > 0))
  );
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    setLoadingProjects(true);
    anomalyTrainingAPI.listProjects()
      .then(setProjects)
      .catch((e) => setErrorMsg(e?.response?.data?.detail || e?.message || 'Failed to load anomaly projects'))
      .finally(() => setLoadingProjects(false));
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || !projectId) { setModels([]); return; }
    setLoadingModels(true);
    anomalyTrainingAPI.listModels(projectId)
      .then(setModels)
      .catch((e) => setErrorMsg(e?.response?.data?.detail || e?.message || 'Failed to load models'))
      .finally(() => setLoadingModels(false));
  }, [isOpen, projectId]);

  if (!isOpen) return null;

  // Only exported models can actually run in production.
  const usableModels = models.filter((m) => m.status === 'completed' && !!m.onnx_path);
  const selectedModel = usableModels.find((m) => m.id === modelId);

  const handleSave = () => {
    if (enabled && !selectedModel) {
      setErrorMsg('Pick a trained + exported model, or turn off "Enabled" to keep using the wrinkle check.');
      return;
    }
    if (areaLogicOn && !(minArea > 0 || maxRegionArea > 0)) {
      setErrorMsg('Area-based logic is on but both Min total area and Max region area are 0 — set at least one, or turn area logic off.');
      return;
    }
    onSave({
      enabled,
      anomaly_project_id: projectId || null,
      anomaly_model_id: modelId || null,
      onnx_path: selectedModel?.onnx_path || null,
      engine_path: selectedModel?.engine_path || null,
      image_size: Number(selectedModel?.params?.image_size) || 256,
      threshold,
      min_area: areaLogicOn ? minArea : 0,
      min_region_area: areaLogicOn ? minRegionArea : 0,
      max_region_area: areaLogicOn ? maxRegionArea : 0,
      pixel_threshold: areaLogicOn ? pixelThreshold : null,
    });
  };

  return (
    <div className="anomaly-setup-backdrop" onClick={onClose}>
      <div className="anomaly-setup-modal" onClick={(e) => e.stopPropagation()}>
        <div className="anomaly-setup-header">
          <h3>Anomaly Detection — {templateName}</h3>
          <button type="button" className="anomaly-setup-close" onClick={onClose}>×</button>
        </div>

        <div className="anomaly-setup-body">
          <label className="anomaly-setup-toggle">
            <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
            <span>Enabled — replaces the wrinkle check with this model when on</span>
          </label>

          <div className="anomaly-setup-row">
            <label>Project</label>
            <select value={projectId} onChange={(e) => { setProjectId(e.target.value); setModelId(''); }} disabled={loadingProjects}>
              <option value="">{loadingProjects ? 'Loading...' : 'Select a project'}</option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>{p.name} ({p.normal_count} normal / {p.abnormal_count} abnormal)</option>
              ))}
            </select>
          </div>

          <div className="anomaly-setup-row">
            <label>Model version</label>
            <select value={modelId} onChange={(e) => setModelId(e.target.value)} disabled={!projectId || loadingModels}>
              <option value="">{loadingModels ? 'Loading...' : 'Select a model'}</option>
              {usableModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {/* Date AND time — several models get trained on the same day while
                      tuning, so the date alone doesn't identify which one you just made.
                      Same toLocaleString() the Train/Eval/Test/Export tabs use. */}
                  {m.algorithm} · AUROC {fmtMetric(m.metrics.image_auroc)} · {new Date(m.created_at).toLocaleString()}
                </option>
              ))}
            </select>
            {projectId && !loadingModels && usableModels.length === 0 && (
              <span className="anomaly-setup-hint">No exported model in this project yet — train + export ONNX in Anomaly Training Studio first.</span>
            )}
            {selectedModel && !selectedModel.engine_path && (
              <span className="anomaly-setup-hint">No TensorRT engine exported for this version — inference will fall back to ONNX (slower). Export TensorRT in Anomaly Training Studio for production.</span>
            )}
          </div>

          <div className="anomaly-setup-row">
            <label>Threshold: {threshold.toFixed(2)}</label>
            <input type="range" min={0} max={1} step={0.01} value={threshold}
                   onChange={(e) => setThreshold(Number(e.target.value))} />
            <span className="anomaly-setup-hint">
              {areaLogicOn
                ? 'Not used while area logic is on — area decides the verdict. The score is still recorded on each frame for debugging.'
                : 'Anomaly score ≥ threshold → FAIL. Tune from the model\'s Eval tab first.'}
            </span>
          </div>

          <label className="anomaly-setup-toggle">
            <input type="checkbox" checked={areaLogicOn} onChange={(e) => setAreaLogicOn(e.target.checked)} />
            <span>Area-based logic — total/max abnormal area decides FAIL, <b>replacing</b> the Threshold above</span>
          </label>

          {areaLogicOn && (
            <div className="anomaly-setup-area-group">
              <div className="anomaly-setup-row">
                <label>Pixel threshold: {pixelThreshold.toFixed(2)}</label>
                <input type="range" min={0} max={1} step={0.01} value={pixelThreshold}
                       onChange={(e) => setPixelThreshold(Number(e.target.value))} />
                <span className="anomaly-setup-hint">
                  Required for area logic to do anything — the model's own built-in mask was found unreliable
                  (never fires without labeled defect masks in training), so regions are found by thresholding
                  the raw anomaly heatmap at this value instead. Same 0–1 scale as Threshold.
                </span>
              </div>

              <div className="anomaly-setup-row">
                <label>Min total area (px)</label>
                <input type="number" min={0} step={100} value={minArea}
                       onChange={(e) => setMinArea(Number(e.target.value))} />
                <span className="anomaly-setup-hint">Sum of all valid regions ≥ this → FAIL. 0 = disabled.</span>
              </div>

              <div className="anomaly-setup-row">
                <label>Min region area (px)</label>
                <input type="number" min={0} step={10} value={minRegionArea}
                       onChange={(e) => setMinRegionArea(Number(e.target.value))} />
                <span className="anomaly-setup-hint">Regions smaller than this are ignored as noise.</span>
              </div>

              <div className="anomaly-setup-row">
                <label>Max region area (px) — critical</label>
                <input type="number" min={0} step={100} value={maxRegionArea}
                       onChange={(e) => setMaxRegionArea(Number(e.target.value))} />
                <span className="anomaly-setup-hint">Any single region ≥ this → immediate FAIL. 0 = disabled.</span>
              </div>
            </div>
          )}

          {errorMsg && <div className="anomaly-setup-error">{errorMsg}</div>}
        </div>

        <div className="anomaly-setup-footer">
          <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
          <button type="button" className="btn btn-primary" onClick={handleSave}>Save</button>
        </div>
      </div>
    </div>
  );
}
