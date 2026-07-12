import { useCallback, useEffect, useState } from 'react';
import '@/styles/AnomalyTraining.css';
import { anomalyTrainingAPI, AnomalyProject, DatasetStats, AnomalyImportResult } from '@/services/anomalyTraining';
import ImportFromRecipeModal from './ImportFromRecipeModal';

interface Props {
  onClose: () => void;
}

export default function AnomalyTrainingPage({ onClose }: Props) {
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

  useEffect(() => { refreshStats(); }, [activeProject?.id]);

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

  const handleImported = (result: AnomalyImportResult) => {
    setLastImportResult(result);
    refreshStats();
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
                   onClick={() => setActiveProject(p)}>
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
          ) : (
            <>
              <div className="at-stats-row">
                <div className="at-stat-card normal">
                  <div className="at-stat-value">{stats?.normal_count ?? activeProject.normal_count}</div>
                  <div className="at-stat-label">Normal images</div>
                </div>
                <div className="at-stat-card abnormal">
                  <div className="at-stat-value">{stats?.abnormal_count ?? activeProject.abnormal_count}</div>
                  <div className="at-stat-label">Abnormal images</div>
                </div>
              </div>

              {!!stats?.defect_types?.length && (
                <div style={{ marginBottom: 20 }}>
                  <label className="at-label">Defect types</label>
                  <div>
                    {stats.defect_types.map((t) => <span key={t} className="at-defect-chip">{t}</span>)}
                  </div>
                </div>
              )}

              <button className="at-btn at-btn-primary" style={{ alignSelf: 'flex-start' }}
                      onClick={() => setShowImportModal(true)}>
                Import from Recipe
              </button>

              {lastImportResult && (
                <div className="at-hint" style={{ marginTop: 10 }}>
                  Last import: {lastImportResult.imported} added, {lastImportResult.skipped} skipped
                  {lastImportResult.errors.length > 0 && `, ${lastImportResult.errors.length} error(s)`}.
                </div>
              )}

              <div className="at-hint" style={{ marginTop: 24 }}>
                Train / Test / Export come online in Week 2–3 (see docs/anomaly_training_plan.md).
              </div>
            </>
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
