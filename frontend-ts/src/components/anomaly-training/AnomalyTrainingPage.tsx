import { useCallback, useEffect, useState } from 'react';
import '@/styles/AnomalyTraining.css';
import {
  anomalyTrainingAPI, AnomalyProject, DatasetStats, AnomalyImportResult, AnomalyModel,
} from '@/services/anomalyTraining';
import ImportFromRecipeModal from './ImportFromRecipeModal';
import DatasetGallery from './DatasetGallery';
import TrainTab from './TrainTab';
import EvalTab from './EvalTab';
import ExportTab from './ExportTab';
import TestTab from './TestTab';
import StudioTab from './StudioTab';
import SyntheticTab from './SyntheticTab';

interface Props {
  onClose: () => void;
}

type TabId = 'dataset' | 'synthetic' | 'train' | 'eval' | 'export' | 'test' | 'studio';

export default function AnomalyTrainingPage({ onClose }: Props) {
  const [tab, setTab] = useState<TabId>('dataset');
  const [projects, setProjects] = useState<AnomalyProject[]>([]);
  const [activeProject, setActiveProject] = useState<AnomalyProject | null>(null);
  const [loadingProjects, setLoadingProjects] = useState(false);

  const [showNewProjectForm, setShowNewProjectForm] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDesc, setNewProjectDesc] = useState('');
  const [creating, setCreating] = useState(false);

  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [showImportModal, setShowImportModal] = useState(false);
  const [lastImportResult, setLastImportResult] = useState<AnomalyImportResult | null>(null);

  const [models, setModels] = useState<AnomalyModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  const loadProjects = useCallback(async () => {
    setLoadingProjects(true);
    try {
      const list = await anomalyTrainingAPI.listProjects();
      setProjects(list);
      if (!activeProject && list.length > 0) {
        setActiveProject(list[0]);
      }
    } catch (e) {
      console.error('[AnomalyTraining] Failed to load projects', e);
    } finally {
      setLoadingProjects(false);
    }
  }, [activeProject]);

  useEffect(() => { loadProjects(); }, []);

  const refreshStats = useCallback(async () => {
    if (!activeProject) { setStats(null); return; }
    try {
      const s = await anomalyTrainingAPI.datasetStats(activeProject.id);
      setStats(s);
      setProjects((prev) => prev.map((p) => p.id === activeProject.id
        ? { ...p, normal_count: s.normal_count, abnormal_count: s.abnormal_count }
        : p));
    } catch (e) {
      console.error('[AnomalyTraining] Failed to load dataset stats', e);
    }
  }, [activeProject]);

  const loadModels = useCallback(async () => {
    if (!activeProject) { setModels([]); return; }
    try {
      const list = await anomalyTrainingAPI.listModels(activeProject.id);
      setModels(list);
      if (!selectedModelId && list.length > 0) setSelectedModelId(list[0].id);
    } catch (e) {
      console.error('[AnomalyTraining] Failed to load models', e);
    }
  }, [activeProject, selectedModelId]);

  useEffect(() => {
    setSelectedModelId(null);
    refreshStats();
    loadModels();
  }, [activeProject?.id]);

  const handleCreateProject = async () => {
    if (!newProjectName.trim()) return;
    setCreating(true);
    try {
      const project = await anomalyTrainingAPI.createProject(newProjectName.trim(), newProjectDesc.trim() || undefined);
      setProjects((prev) => [project, ...prev]);
      setActiveProject(project);
      setShowNewProjectForm(false);
      setNewProjectName('');
      setNewProjectDesc('');
    } catch (e) {
      console.error('[AnomalyTraining] Failed to create project', e);
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteProject = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this anomaly project and all its dataset images?')) return;
    try {
      await anomalyTrainingAPI.deleteProject(id);
      const next = projects.filter((p) => p.id !== id);
      setProjects(next);
      if (activeProject?.id === id) setActiveProject(next[0] ?? null);
    } catch (e) {
      console.error('[AnomalyTraining] Failed to delete project', e);
    }
  };

  const [galleryRefreshKey, setGalleryRefreshKey] = useState(0);

  const handleImported = (result: AnomalyImportResult) => {
    setLastImportResult(result);
    refreshStats();
    setGalleryRefreshKey((k) => k + 1);
  };

  const selectedModel = models.find((m) => m.id === selectedModelId) || null;
  // PatchCore/Padim fit on normal images only -- abnormal is optional, only
  // needed to get a meaningful AUROC/F1 out of eval (backend reports those
  // as 0.0 / not-yet-meaningful when there's no abnormal test data).
  const hasDataset = (stats?.normal_count ?? activeProject?.normal_count ?? 0) > 0;

  const TAB_LABELS: Record<TabId, string> = {
    dataset: 'Dataset', synthetic: 'Synthetic NG', train: 'Train', eval: 'Eval',
    export: 'Export', test: 'Test', studio: 'Studio',
  };

  return (
    <div className="at-overlay">
      <div className="at-header">
        <div className="at-header-left">
          <h1 className="at-title">Anomaly Training Studio</h1>
          {activeProject && (
            <span className={`at-badge ${activeProject.status}`}>{activeProject.name}</span>
          )}
        </div>
        <button className="at-close-btn" onClick={onClose}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
          Close
        </button>
      </div>

      {activeProject && (
        <div className="at-tabs">
          {(['dataset', 'synthetic', 'train', 'eval', 'export', 'test', 'studio'] as TabId[]).map((id) => (
            <button
              key={id}
              className={`at-tab-btn ${tab === id ? 'active' : ''}`}
              disabled={id !== 'dataset' && !hasDataset}
              title={id !== 'dataset' && !hasDataset ? 'Import normal + abnormal images first' : undefined}
              onClick={() => setTab(id)}
            >
              {TAB_LABELS[id]}
            </button>
          ))}
        </div>
      )}

      <div className="at-body">
        <div className="at-sidebar">
          <div className="at-sidebar-header">
            Projects
            <button className="at-new-project-btn" onClick={() => setShowNewProjectForm(true)} title="New project">+</button>
          </div>

          {showNewProjectForm && (
            <div style={{ padding: 10, borderBottom: '1px solid #e2e8f0', display: 'flex', flexDirection: 'column', gap: 6 }}>
              <input className="at-form-input" placeholder="Project name" value={newProjectName}
                     onChange={(e) => setNewProjectName(e.target.value)}
                     onKeyDown={(e) => e.key === 'Enter' && handleCreateProject()} autoFocus />
              <input className="at-form-input" placeholder="Description (optional)" value={newProjectDesc}
                     onChange={(e) => setNewProjectDesc(e.target.value)} />
              <div style={{ display: 'flex', gap: 6 }}>
                <button className="at-btn at-btn-primary at-btn-sm" onClick={handleCreateProject}
                        disabled={creating || !newProjectName.trim()}>
                  {creating ? '...' : 'Create'}
                </button>
                <button className="at-btn at-btn-secondary at-btn-sm" onClick={() => setShowNewProjectForm(false)}>Cancel</button>
              </div>
            </div>
          )}

          <div className="at-project-list">
            {loadingProjects && (
              <div className="at-empty-state" style={{ padding: 20 }}><div className="at-loading-spinner" /></div>
            )}
            {!loadingProjects && projects.length === 0 && (
              <div className="at-empty-state" style={{ padding: 20 }}>No projects yet</div>
            )}
            {projects.map((p) => (
              <div key={p.id} className={`at-project-item ${activeProject?.id === p.id ? 'active' : ''}`}
                   onClick={() => { setActiveProject(p); setTab('dataset'); }}>
                <div className="at-project-item-info">
                  <div className="at-project-item-name">{p.name}</div>
                  <div className="at-project-item-meta">{p.normal_count} normal · {p.abnormal_count} abnormal</div>
                </div>
                <button className="at-project-delete-btn" onClick={(e) => handleDeleteProject(p.id, e)} title="Delete project">✕</button>
              </div>
            ))}
          </div>
        </div>

        <div className="at-content">
          {!activeProject ? (
            <div className="at-empty-state">Select or create a project to get started</div>
          ) : tab === 'dataset' ? (
            <>
              <div className="at-dataset-toolbar">
                <div className="at-dataset-toolbar-stats">
                  <span className="at-stat-pill normal">
                    <b>{stats?.normal_count ?? activeProject.normal_count}</b> normal
                  </span>
                  <span className="at-stat-pill abnormal">
                    <b>{stats?.abnormal_count ?? activeProject.abnormal_count}</b> abnormal
                  </span>
                  {!!stats?.defect_types?.length && (
                    <span className="at-dataset-toolbar-defects">
                      {stats.defect_types.map((t) => <span key={t} className="at-defect-chip">{t}</span>)}
                    </span>
                  )}
                </div>
                <button className="at-btn at-btn-primary" onClick={() => setShowImportModal(true)}>
                  + Import from Recipe
                </button>
              </div>

              {lastImportResult && (
                <div className="at-hint" style={{ marginBottom: 8 }}>
                  Last import: {lastImportResult.imported} added, {lastImportResult.skipped} skipped
                  {lastImportResult.errors.length > 0 && `, ${lastImportResult.errors.length} error(s)`}.
                </div>
              )}

              {!hasDataset && (
                <div className="at-hint" style={{ marginBottom: 12 }}>
                  Import at least one normal image to unlock Train (abnormal images are optional,
                  only needed for a meaningful eval score).
                </div>
              )}

              <DatasetGallery
                projectId={activeProject.id}
                refreshKey={galleryRefreshKey}
                onCountsChanged={refreshStats}
              />
            </>
          ) : tab === 'synthetic' ? (
            <SyntheticTab
              projectId={activeProject.id}
              onGenerated={() => { setGalleryRefreshKey((k) => k + 1); refreshStats(); }}
            />
          ) : tab === 'train' ? (
            <TrainTab
              projectId={activeProject.id}
              models={models}
              onModelsChange={loadModels}
              selectedModelId={selectedModelId}
              onSelectModel={setSelectedModelId}
            />
          ) : tab === 'eval' ? (
            <EvalTab projectId={activeProject.id} model={selectedModel} />
          ) : tab === 'export' ? (
            <ExportTab projectId={activeProject.id} model={selectedModel} onModelChange={loadModels} />
          ) : tab === 'test' ? (
            <TestTab projectId={activeProject.id} model={selectedModel} />
          ) : (
            <StudioTab projectId={activeProject.id} model={selectedModel} />
          )}
        </div>
      </div>

      {activeProject && (
        <ImportFromRecipeModal
          projectId={activeProject.id}
          open={showImportModal}
          onClose={() => setShowImportModal(false)}
          onImported={handleImported}
        />
      )}
    </div>
  );
}
