import React, { useCallback, useEffect, useState } from 'react';
import '@/styles/MLTraining.css';
import { mlTrainingAPI, MLProject } from '@/services/mlTraining';
import ImageTab from './ImageTab';
import LabelTab from './LabelTab';
import TrainTab from './TrainTab';

type TabId = 'images' | 'label' | 'train';

interface Props {
  onClose: () => void;
}

export default function MLTrainingPage({ onClose }: Props) {
  const [tab, setTab] = useState<TabId>('images');
  const [projects, setProjects] = useState<MLProject[]>([]);
  const [activeProject, setActiveProject] = useState<MLProject | null>(null);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [showNewProjectForm, setShowNewProjectForm] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDesc, setNewProjectDesc] = useState('');
  const [creating, setCreating] = useState(false);

  // ── Load projects ──────────────────────────────────────────────────────
  const loadProjects = useCallback(async () => {
    setLoadingProjects(true);
    try {
      const list = await mlTrainingAPI.listProjects();
      setProjects(list);
      if (!activeProject && list.length > 0) {
        setActiveProject(list[0]);
      }
    } catch (e) {
      console.error('Failed to load projects', e);
    } finally {
      setLoadingProjects(false);
    }
  }, [activeProject]);

  useEffect(() => { loadProjects(); }, []);

  // ── Create project ─────────────────────────────────────────────────────
  const handleCreateProject = async () => {
    if (!newProjectName.trim()) return;
    setCreating(true);
    try {
      const project = await mlTrainingAPI.createProject(newProjectName.trim(), newProjectDesc.trim() || undefined);
      setProjects(prev => [project, ...prev]);
      setActiveProject(project);
      setShowNewProjectForm(false);
      setNewProjectName('');
      setNewProjectDesc('');
    } catch (e) {
      console.error('Failed to create project', e);
    } finally {
      setCreating(false);
    }
  };

  // ── Delete project ─────────────────────────────────────────────────────
  const handleDeleteProject = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this project and all its data?')) return;
    try {
      await mlTrainingAPI.deleteProject(id);
      const next = projects.filter(p => p.id !== id);
      setProjects(next);
      if (activeProject?.id === id) {
        setActiveProject(next[0] ?? null);
      }
    } catch (e) {
      console.error('Failed to delete project', e);
    }
  };

  // ── Refresh project stats ──────────────────────────────────────────────
  const refreshActiveProject = useCallback(async () => {
    if (!activeProject) return;
    try {
      const updated = await mlTrainingAPI.getProject(activeProject.id);
      setActiveProject(updated);
      setProjects(prev => prev.map(p => p.id === updated.id ? updated : p));
    } catch { /* ignore */ }
  }, [activeProject]);

  const TAB_ICONS: Record<TabId, React.ReactNode> = {
    images: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
        <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
        <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
        <path d="M21 15l-5-5L5 21" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
      </svg>
    ),
    label: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
        <path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
        <circle cx="7" cy="7" r="1.5" fill="currentColor"/>
      </svg>
    ),
    train: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
        <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
        <path d="M2 17l10 5 10-5" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
        <path d="M2 12l10 5 10-5" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
      </svg>
    ),
  };

  return (
    <div className="ml-training-overlay">
      {/* Header */}
      <div className="ml-training-header">
        <div className="ml-training-header-left">
          <h1 className="ml-training-title">ML Training Studio</h1>
          {activeProject && (
            <span className={`ml-training-badge ${activeProject.status}`}>
              {activeProject.name}
            </span>
          )}
        </div>
        <button className="ml-close-btn" onClick={onClose}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
          Close
        </button>
      </div>

      {/* Tabs */}
      <div className="ml-training-tabs">
        {(['images', 'label', 'train'] as TabId[]).map(id => (
          <button
            key={id}
            className={`ml-tab-btn ${tab === id ? 'active' : ''}`}
            onClick={() => setTab(id)}
          >
            {TAB_ICONS[id]}
            {id.charAt(0).toUpperCase() + id.slice(1)}
            {id === 'images' && activeProject && (
              <span className="ml-tab-count">{activeProject.image_count}</span>
            )}
            {id === 'label' && activeProject && (
              <span className="ml-tab-count">{activeProject.labeled_count}</span>
            )}
          </button>
        ))}
      </div>

      {/* Body */}
      <div className="ml-training-body">
        {/* Project sidebar */}
        <div className="ml-project-sidebar">
          <div className="ml-sidebar-header">
            Projects
            <button className="ml-new-project-btn" onClick={() => setShowNewProjectForm(true)} title="New project">+</button>
          </div>

          {/* New project form */}
          {showNewProjectForm && (
            <div style={{ padding: '10px', borderBottom: '1px solid #2d3148', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <input
                className="ml-form-input"
                placeholder="Project name"
                value={newProjectName}
                onChange={e => setNewProjectName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleCreateProject()}
                autoFocus
              />
              <input
                className="ml-form-input"
                placeholder="Description (optional)"
                value={newProjectDesc}
                onChange={e => setNewProjectDesc(e.target.value)}
              />
              <div style={{ display: 'flex', gap: '6px' }}>
                <button className="ml-btn ml-btn-primary ml-btn-sm" onClick={handleCreateProject} disabled={creating || !newProjectName.trim()}>
                  {creating ? '...' : 'Create'}
                </button>
                <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={() => setShowNewProjectForm(false)}>Cancel</button>
              </div>
            </div>
          )}

          <div className="ml-project-list">
            {loadingProjects && (
              <div className="ml-empty-state" style={{ padding: '20px' }}>
                <div className="ml-loading-spinner" />
              </div>
            )}
            {!loadingProjects && projects.length === 0 && (
              <div className="ml-empty-state" style={{ padding: '20px' }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style={{opacity:.4}}><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/></svg>
                No projects yet
              </div>
            )}
            {projects.map(p => (
              <div
                key={p.id}
                className={`ml-project-item ${activeProject?.id === p.id ? 'active' : ''}`}
                onClick={() => setActiveProject(p)}
              >
                <div className="ml-project-item-info">
                  <div className="ml-project-item-name">{p.name}</div>
                  <div className="ml-project-item-meta">
                    {p.image_count} imgs · {p.labeled_count} labeled
                  </div>
                </div>
                <button
                  className="ml-project-delete-btn"
                  onClick={e => handleDeleteProject(p.id, e)}
                  title="Delete project"
                >✕</button>
              </div>
            ))}
          </div>
        </div>

        {/* Tab content */}
        <div className="ml-tab-content">
          {!activeProject ? (
            <div className="ml-empty-state" style={{ flex: 1 }}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style={{opacity:.4}}><path d="M9 12H3m0 0l3-3m-3 3l3 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/><rect x="9" y="4" width="12" height="16" rx="2" stroke="currentColor" strokeWidth="2"/></svg>
              Select or create a project to get started
            </div>
          ) : (
            <>
              {tab === 'images' && (
                <ImageTab project={activeProject} onRefresh={refreshActiveProject} />
              )}
              {tab === 'label' && (
                <LabelTab project={activeProject} onRefresh={refreshActiveProject} />
              )}
              {tab === 'train' && (
                <TrainTab project={activeProject} onRefresh={refreshActiveProject} />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
