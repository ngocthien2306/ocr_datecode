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

  const tabs: { id: TabId; label: string; icon: string }[] = [
    { id: 'images', label: 'Images', icon: '🖼️' },
    { id: 'label',  label: 'Label',  icon: '🏷️' },
    { id: 'train',  label: 'Train',  icon: '🧠' },
  ];

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
        <button className="ml-close-btn" onClick={onClose}>✕ Close</button>
      </div>

      {/* Tabs */}
      <div className="ml-training-tabs">
        {tabs.map(t => (
          <button
            key={t.id}
            className={`ml-tab-btn ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            <span>{t.icon}</span>
            {t.label}
            {t.id === 'images' && activeProject && (
              <span className="ml-tab-count">{activeProject.image_count}</span>
            )}
            {t.id === 'label' && activeProject && (
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
                <span className="ml-empty-icon">📁</span>
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
              <span className="ml-empty-icon">👈</span>
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
