import { useCallback, useEffect, useState } from 'react';
import '@/styles/OCRTraining.css';
import {
  ocrTrainingAPI, ocrTrainingModelAPI, OCRDatasetStats, OCRImportResult, OCRModel, OCRProject,
} from '@/services/ocrTraining';
import DatasetTab from './DatasetTab';
import LabelTab from './LabelTab';
import TrainTab from './TrainTab';
import EvalTab from './EvalTab';
import ExportTab from './ExportTab';
import TestTab from './TestTab';
import ImportFromRecipeModal from './ImportFromRecipeModal';

interface Props {
  onClose: () => void;
}

type TabId = 'dataset' | 'label' | 'train' | 'eval' | 'export' | 'test';

const TAB_IDS: TabId[] = ['dataset', 'label', 'train', 'eval', 'export', 'test'];
const TAB_LABELS: Record<TabId, string> = {
  dataset: 'Dataset', label: 'Label', train: 'Train',
  eval: 'Eval', export: 'Export', test: 'Test',
};

export default function OCRTrainingPage({ onClose }: Props) {
  const [tab, setTab] = useState<TabId>('dataset');
  const [projects, setProjects] = useState<OCRProject[]>([]);
  const [activeProject, setActiveProject] = useState<OCRProject | null>(null);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [serviceError, setServiceError] = useState<string | null>(null);

  const [showNewProjectForm, setShowNewProjectForm] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDesc, setNewProjectDesc] = useState('');
  const [creating, setCreating] = useState(false);

  const [stats, setStats] = useState<OCRDatasetStats | null>(null);
  const [showImportModal, setShowImportModal] = useState(false);
  const [lastImportResult, setLastImportResult] = useState<OCRImportResult | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [models, setModels] = useState<OCRModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  const loadProjects = useCallback(async () => {
    setLoadingProjects(true);
    try {
      const list = await ocrTrainingAPI.listProjects();
      setProjects(list);
      setServiceError(null);
      setActiveProject((cur) => cur ?? list[0] ?? null);
    } catch (e: any) {
      // A studio that renders an empty sidebar when the service is down looks
      // like "no projects yet", which sends people hunting in the wrong place.
      setServiceError(
        e?.code === 'ERR_NETWORK'
          ? 'Cannot reach ocr_service on :8002. Start it with `python -m app.main` in ocr_service/.'
          : (e?.response?.data?.detail || e?.message || 'Failed to load projects'),
      );
    } finally {
      setLoadingProjects(false);
    }
  }, []);

  useEffect(() => { loadProjects(); }, [loadProjects]);

  const refreshStats = useCallback(async () => {
    if (!activeProject) { setStats(null); return; }
    try {
      const s = await ocrTrainingAPI.datasetStats(activeProject.id);
      setStats(s);
      setProjects((prev) => prev.map((p) => p.id === activeProject.id
        ? { ...p, total_count: s.total_count, verified_count: s.verified_count,
            need_review_count: s.need_review_count }
        : p));
    } catch (e) {
      console.error('[OCRTraining] Failed to load dataset stats', e);
    }
  }, [activeProject]);

  const loadModels = useCallback(async () => {
    if (!activeProject) { setModels([]); return; }
    try {
      const list = await ocrTrainingModelAPI.listModels(activeProject.id);
      setModels(list);
      // Default to the newest run so Eval/Export/Test have something to show
      // without the operator having to click back into Train first.
      setSelectedModelId((cur) => (cur && list.some((m) => m.id === cur)) ? cur : (list[0]?.id ?? null));
    } catch (e) {
      console.error('[OCRTraining] Failed to load models', e);
    }
  }, [activeProject]);

  useEffect(() => { refreshStats(); loadModels(); }, [activeProject?.id]);

  // A live run's status/metrics change without any user action, so the run list
  // has to keep up while the Train tab streams its log.
  useEffect(() => {
    if (!models.some((m) => m.status === 'training' || m.status === 'pending')) return;
    const t = setInterval(loadModels, 4000);
    return () => clearInterval(t);
  }, [models, loadModels]);

  const handleCreateProject = async () => {
    if (!newProjectName.trim()) return;
    setCreating(true);
    try {
      const project = await ocrTrainingAPI.createProject(
        newProjectName.trim(), newProjectDesc.trim() || undefined);
      setProjects((prev) => [project, ...prev]);
      setActiveProject(project);
      setShowNewProjectForm(false);
      setNewProjectName('');
      setNewProjectDesc('');
    } catch (e: any) {
      setServiceError(e?.response?.data?.detail || e?.message || 'Failed to create project');
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteProject = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this OCR project, its dataset images and its trained models?')) return;
    try {
      await ocrTrainingAPI.deleteProject(id);
      const next = projects.filter((p) => p.id !== id);
      setProjects(next);
      if (activeProject?.id === id) setActiveProject(next[0] ?? null);
    } catch (e: any) {
      setServiceError(e?.response?.data?.detail || e?.message || 'Failed to delete project');
    }
  };

  const handleImported = (result: OCRImportResult) => {
    setLastImportResult(result);
    refreshStats();
    setRefreshKey((k) => k + 1);
  };

  const verified = stats?.verified_count ?? activeProject?.verified_count ?? 0;
  const needReview = stats?.need_review_count ?? activeProject?.need_review_count ?? 0;
  const total = stats?.total_count ?? activeProject?.total_count ?? 0;
  const selectedModel = models.find((m) => m.id === selectedModelId) ?? null;

  return (
    <div className="at-overlay">
      <div className="at-header">
        <div className="at-header-left">
          <h1 className="at-title">OCR Training Studio</h1>
          {activeProject && (
            <span className={`at-badge ${activeProject.status}`}>{activeProject.name}</span>
          )}
        </div>
        <button className="at-close-btn" onClick={onClose}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
            <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          Close
        </button>
      </div>

      {activeProject && (
        <div className="at-tabs">
          {TAB_IDS.map((id) => {
            // Train needs verified labels; the three after it need a run to exist.
            const needsData = id === 'train' && verified === 0;
            const needsModel = (id === 'eval' || id === 'export' || id === 'test') && models.length === 0;
            const disabled = needsData || needsModel;
            return (
              <button key={id} className={`at-tab-btn ${tab === id ? 'active' : ''}`}
                      disabled={disabled}
                      title={needsData ? 'Verify some labels first — training only reads verified items'
                           : needsModel ? 'Train a model first' : undefined}
                      onClick={() => setTab(id)}>
                {TAB_LABELS[id]}
                {id === 'label' && needReview > 0 && (
                  <span className="at-chip" style={{ marginLeft: 6 }}>{needReview}</span>
                )}
                {id === 'train' && models.some((m) => m.status === 'training' || m.status === 'pending') && (
                  <span className="at-chip" style={{ marginLeft: 6 }}>running</span>
                )}
              </button>
            );
          })}
        </div>
      )}

      <div className="at-body">
        <div className="at-sidebar">
          <div className="at-sidebar-header">
            Projects
            <button className="at-new-project-btn" onClick={() => setShowNewProjectForm(true)}
                    title="New project">+</button>
          </div>

          {showNewProjectForm && (
            <div style={{ padding: 10, borderBottom: '1px solid #e2e8f0', display: 'flex',
                          flexDirection: 'column', gap: 6 }}>
              <input className="at-form-input" placeholder="Project name" value={newProjectName}
                     onChange={(e) => setNewProjectName(e.target.value)}
                     onKeyDown={(e) => e.key === 'Enter' && handleCreateProject()} autoFocus />
              <input className="at-form-input" placeholder="Description (optional)" value={newProjectDesc}
                     onChange={(e) => setNewProjectDesc(e.target.value)} />
              <div style={{ display: 'flex', gap: 6 }}>
                <button className="at-btn at-btn-primary at-btn-sm" onClick={handleCreateProject}
                        disabled={creating || !newProjectName.trim()}>
                  {creating ? '…' : 'Create'}
                </button>
                <button className="at-btn at-btn-secondary at-btn-sm"
                        onClick={() => setShowNewProjectForm(false)}>Cancel</button>
              </div>
            </div>
          )}

          <div className="at-project-list">
            {loadingProjects && (
              <div className="at-empty-state" style={{ padding: 20 }}><div className="at-loading-spinner" /></div>
            )}
            {!loadingProjects && !serviceError && projects.length === 0 && (
              <div className="at-empty-state" style={{ padding: 20 }}>No projects yet</div>
            )}
            {projects.map((p) => (
              <div key={p.id} className={`at-project-item ${activeProject?.id === p.id ? 'active' : ''}`}
                   onClick={() => { setActiveProject(p); setTab('dataset'); }}>
                <div className="at-project-item-info">
                  <div className="at-project-item-name">{p.name}</div>
                  <div className="at-project-item-meta">
                    {p.verified_count} verified
                    {p.need_review_count > 0 && ` · ${p.need_review_count} to review`}
                  </div>
                </div>
                <button className="at-project-delete-btn" onClick={(e) => handleDeleteProject(p.id, e)}
                        title="Delete project">✕</button>
              </div>
            ))}
          </div>
        </div>

        <div className="at-content">
          {serviceError ? (
            <div className="at-alert-error" style={{ margin: 20 }}>
              {serviceError}
              <div style={{ marginTop: 8 }}>
                <button className="at-btn at-btn-secondary at-btn-sm" onClick={loadProjects}>Retry</button>
              </div>
            </div>
          ) : !activeProject ? (
            <div className="at-empty-state">Select or create a project to get started</div>
          ) : (
            <>
              <div className="at-dataset-toolbar">
                <div className="at-dataset-toolbar-stats">
                  <span className="at-stat-pill normal"><b>{verified}</b> verified</span>
                  <span className="at-stat-pill abnormal"><b>{needReview}</b> to review</span>
                  <span className="at-stat-pill"><b>{total}</b> total</span>
                </div>
              </div>

              {lastImportResult && (
                <div className="at-hint" style={{ marginBottom: 8 }}>
                  Last import: {lastImportResult.imported} added, {lastImportResult.skipped} skipped
                  {lastImportResult.errors.length > 0 && `, ${lastImportResult.errors.length} error(s)`}.
                  They arrived as <b>need_review</b> — verify them in the Label tab before training.
                </div>
              )}

              {verified === 0 && total > 0 && (
                <div className="at-hint" style={{ marginBottom: 12 }}>
                  Nothing is verified yet, so a training run has nothing to read. Head to the
                  Label tab — items the recipe already confirmed can be accepted in one click.
                </div>
              )}

              {tab === 'dataset' ? (
                <DatasetTab projectId={activeProject.id} refreshKey={refreshKey}
                            onCountsChanged={() => { refreshStats(); }}
                            onOpenImport={() => setShowImportModal(true)} />
              ) : tab === 'label' ? (
                /* onCountsChanged refreshes the counts only. Bumping refreshKey
                   here would refetch the page after every save, undoing
                   LabelTab's patch-in-place and reshuffling rows out from under
                   the cursor mid-review. */
                <LabelTab projectId={activeProject.id} refreshKey={refreshKey}
                          onCountsChanged={refreshStats} />
              ) : tab === 'train' ? (
                <TrainTab projectId={activeProject.id} models={models}
                          onModelsChange={loadModels}
                          selectedModelId={selectedModelId} onSelectModel={setSelectedModelId} />
              ) : tab === 'eval' ? (
                <EvalTab projectId={activeProject.id} model={selectedModel} />
              ) : tab === 'export' ? (
                <ExportTab projectId={activeProject.id} model={selectedModel}
                           onModelChange={loadModels} />
              ) : (
                <TestTab projectId={activeProject.id} model={selectedModel} />
              )}
            </>
          )}
        </div>
      </div>

      {activeProject && (
        <ImportFromRecipeModal projectId={activeProject.id} open={showImportModal}
                               onClose={() => setShowImportModal(false)} onImported={handleImported} />
      )}
    </div>
  );
}
