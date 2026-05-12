import React, { useCallback, useEffect, useState } from 'react';
import '@/styles/MLTraining.css';
import { mlTrainingAPI, MLProject } from '@/services/mlTraining';
import ImageTab from './ImageTab';
import LabelTab from './LabelTab';
import ImportedCharsTab from './ImportedCharsTab';
import TrainTab from './TrainTab';
import { DeepLink } from './CropEditPopover';

type TabId = 'images' | 'label' | 'imports' | 'train';

interface Props {
  onClose: () => void;
}

export default function MLTrainingPage({ onClose }: Props) {
  const [tab, setTab] = useState<TabId>('images');
  const [projects, setProjects] = useState<MLProject[]>([]);
  const [activeProject, setActiveProject] = useState<MLProject | null>(null);
  // Deep-link payload set when user clicks "Open in {tab}" inside the Train-tab
  // crop editor. The target tab consumes it on mount, then calls clearDeepLink
  // so re-mounts don't re-trigger the focus.
  const [deepLink, setDeepLink] = useState<DeepLink | null>(null);
  const handleJumpTo = useCallback((link: DeepLink) => {
    setTab(link.tab);
    setDeepLink(link);
  }, []);
  const clearDeepLink = useCallback(() => setDeepLink(null), []);

  // ── AI service lifecycle state ─────────────────────────────────────────
  // Tracks whether we (this studio session) were the ones who stopped the
  // AI camera service. Persisted in BE via /tmp lock file too, so this is
  // really just a UX shortcut to avoid an extra status round-trip on close.
  const [aiStoppedHere, setAiStoppedHere] = useState(false);
  // Reflects status of any model currently training in the active project.
  // Updated by TrainTab via the onTrainingStateChange callback.
  const [trainingActive, setTrainingActive] = useState(false);

  // Modal flags. Entry-confirm is owned by the parent (Dashboard) so the
  // page doesn't flash before the dialog appears.
  const [showRestartingModal, setShowRestartingModal] = useState(false);
  const [showStartingAIModal, setShowStartingAIModal] = useState(false);
  const [showCloseDuringTrainModal, setShowCloseDuringTrainModal] = useState(false);
  const [showRestartFailedModal, setShowRestartFailedModal] = useState(false);

  // ── Probe AI service status on mount ──────────────────────────────────
  // Dashboard already asked the user about stopping AI before mounting us.
  // We just check the BE flag to know whether we should auto-start on exit.
  useEffect(() => {
    (async () => {
      try {
        const status = await mlTrainingAPI.aiServiceStatus();
        // BE keeps a flag file when /stop is called. Presence == "we owe a
        // restart on close". Both fresh-stop (from Dashboard) and stale-flag
        // (previous session) end up here, which is the desired behavior.
        if (!status.active && status.in_training_mode) {
          setAiStoppedHere(true);
        }
      } catch (e) {
        console.warn('[MLTrainingPage] ai-service status probe failed', e);
      }
    })();
  }, []);

  // ── Training-complete handler: BE will auto-restart ocr-all ────────────
  // Detection logic:
  //   1. Poll /api/health every 2s.
  //   2. Until we see at least one FAILED health check, BE hasn't been
  //      killed yet — restart hasn't kicked in.
  //   3. After we see a fail (BE down) and then a success (BE back) → real
  //      restart cycle completed → reload page.
  //   4. If 15s pass with NO failed health checks → restart silently
  //      failed (bad sudo password, systemctl error, …). Show error modal.
  //   5. 60s overall timeout as a safety net.
  // In practice Firefox kiosk is killed by stop_services.sh well before the
  // page reload above runs; this is the fallback when something prevents it.
  const handleTrainingComplete = useCallback(() => {
    setTrainingActive(false);
    setShowRestartingModal(true);

    const startedAt = Date.now();
    let sawHealthFail = false;
    let interval: ReturnType<typeof setInterval> | null = null;
    let safetyTimeout: ReturnType<typeof setTimeout> | null = null;

    const cleanup = () => {
      if (interval) { clearInterval(interval); interval = null; }
      if (safetyTimeout) { clearTimeout(safetyTimeout); safetyTimeout = null; }
    };

    interval = setInterval(async () => {
      const elapsed = Date.now() - startedAt;
      try {
        await mlTrainingAPI.apiHealth();
        // Health OK
        if (sawHealthFail) {
          // BE went down then came back up — real restart cycle done.
          setShowRestartingModal(false);
          cleanup();
          window.location.reload();
        } else if (elapsed > 15_000) {
          // 15s passed, BE never went down → restart never happened.
          // sudo password wrong, systemctl error, or some other failure.
          setShowRestartingModal(false);
          setShowRestartFailedModal(true);
          cleanup();
        }
        // else: BE still alive but restart may still be on its way — wait.
      } catch {
        sawHealthFail = true;
      }
    }, 2000);

    safetyTimeout = setTimeout(() => {
      setShowRestartingModal(false);
      cleanup();
    }, 60_000);
  }, []);

  const handleClose = useCallback(async () => {
    // Drop in-memory training caches (existing behavior — 409 if mid-train).
    if (activeProject) {
      try {
        await mlTrainingAPI.releaseTrainingResources(activeProject.id);
      } catch (e) {
        console.debug('[MLTrainingPage] release skipped:', e);
      }
    }

    // If training is mid-flight, warn user — the system will auto-restart
    // once training completes (BE side), so leaving is safe.
    if (trainingActive) {
      setShowCloseDuringTrainModal(true);
      return;
    }

    // If we stopped AI on entry, start it back up — block close until it's
    // confirmed running so user isn't left without a camera service.
    if (aiStoppedHere) {
      setShowStartingAIModal(true);
      try {
        await mlTrainingAPI.aiServiceStart();
        const deadline = Date.now() + 30_000;
        while (Date.now() < deadline) {
          await new Promise(r => setTimeout(r, 1500));
          try {
            const st = await mlTrainingAPI.aiServiceStatus();
            if (st.active) break;
          } catch { /* still booting */ }
        }
      } catch (e) {
        console.error('[MLTrainingPage] ai-service start failed', e);
      }
      setShowStartingAIModal(false);
    }

    onClose();
  }, [activeProject, trainingActive, aiStoppedHere, onClose]);

  // User chose to close during training — BE auto-restart will handle the
  // rest. Don't try to manually start AI service: ocr-all restart will
  // respawn it.
  const handleConfirmCloseDuringTrain = useCallback(() => {
    setShowCloseDuringTrainModal(false);
    onClose();
  }, [onClose]);
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

  useEffect(() => {
    // Snapshot camera buffer once when ML Training page opens
    mlTrainingAPI.snapshotImages().catch(e => console.warn('Snapshot failed', e));
    loadProjects();
  }, []);

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

  // ── Clone project (copy training data only — no trained models) ────────
  const [cloningId, setCloningId] = useState<string | null>(null);
  const handleCloneProject = async (p: MLProject, e: React.MouseEvent) => {
    e.stopPropagation();
    const newName = prompt('Tên project mới:', `${p.name} (copy)`);
    if (newName === null) return;
    const trimmed = newName.trim();
    if (!trimmed) return;
    setCloningId(p.id);
    try {
      const cloned = await mlTrainingAPI.cloneProject(p.id, trimmed);
      setProjects(prev => [cloned, ...prev]);
      setActiveProject(cloned);
    } catch (err) {
      console.error('Failed to clone project', err);
      alert('Clone project thất bại');
    } finally {
      setCloningId(null);
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
    imports: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
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

  const TAB_LABELS: Record<TabId, string> = {
    images:  'Images',
    label:   'Label',
    imports: 'Imported Chars',
    train:   'Train',
  };

  return (
    <div className="ml-training-overlay">
      {/* Header */}
      <div className="ml-training-header">
        <div className="ml-training-header-left">
          <h1 className="ml-training-title">AI Training Studio</h1>
          {activeProject && (
            <span className={`ml-training-badge ${activeProject.status}`}>
              {activeProject.name}
            </span>
          )}
        </div>
        <button className="ml-close-btn" onClick={handleClose}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
          Close
        </button>
      </div>

      {/* Tabs */}
      <div className="ml-training-tabs">
        {(['images', 'label', 'imports', 'train'] as TabId[]).map(id => (
          <button
            key={id}
            className={`ml-tab-btn ${tab === id ? 'active' : ''}`}
            onClick={() => setTab(id)}
          >
            {TAB_ICONS[id]}
            {TAB_LABELS[id]}
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
                <div className="ml-project-actions">
                  <button
                    className="ml-project-clone-btn"
                    onClick={e => handleCloneProject(p, e)}
                    title="Copy project data"
                    disabled={cloningId === p.id}
                  >
                    {cloningId === p.id ? '…' : (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                        <rect x="9" y="9" width="11" height="11" rx="2" stroke="currentColor" strokeWidth="2"/>
                        <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" stroke="currentColor" strokeWidth="2"/>
                      </svg>
                    )}
                  </button>
                  <button
                    className="ml-project-delete-btn"
                    onClick={e => handleDeleteProject(p.id, e)}
                    title="Delete project"
                  >✕</button>
                </div>
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
                <LabelTab project={activeProject} onRefresh={refreshActiveProject}
                  deepLink={deepLink && deepLink.tab === 'label' ? deepLink : null}
                  onDeepLinkConsumed={clearDeepLink} />
              )}
              {tab === 'imports' && (
                <ImportedCharsTab project={activeProject} onRefresh={refreshActiveProject}
                  deepLink={deepLink && deepLink.tab === 'imports' ? deepLink : null}
                  onDeepLinkConsumed={clearDeepLink} />
              )}
              {tab === 'train' && (
                <TrainTab project={activeProject} onRefresh={refreshActiveProject}
                  onJumpTo={handleJumpTo}
                  onTrainingStateChange={setTrainingActive}
                  onTrainingComplete={handleTrainingComplete} />
              )}
            </>
          )}
        </div>
      </div>

      {/* ── Modals: AI service lifecycle (entry-confirm owned by Dashboard) ── */}

      {showRestartingModal && (
        <div className="ml-modal-overlay">
          <div className="ml-modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 420 }}>
            <div className="ml-modal-header">
              <h3>🔄 Training xong — Đang khởi động lại</h3>
            </div>
            <div className="ml-modal-body" style={{ padding: 16, fontSize: 13, lineHeight: 1.5, textAlign: 'center' }}>
              <div className="ml-loading-spinner" style={{ margin: '12px auto', width: 32, height: 32, borderWidth: 3 }} />
              <p style={{ margin: 0 }}>
                Hệ thống đang khởi động lại để áp dụng model mới.<br />
                Vui lòng chờ trong giây lát…
              </p>
            </div>
          </div>
        </div>
      )}

      {showStartingAIModal && (
        <div className="ml-modal-overlay">
          <div className="ml-modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 420 }}>
            <div className="ml-modal-header">
              <h3>📷 Đang khởi động Camera</h3>
            </div>
            <div className="ml-modal-body" style={{ padding: 16, fontSize: 13, lineHeight: 1.5, textAlign: 'center' }}>
              <div className="ml-loading-spinner" style={{ margin: '12px auto', width: 32, height: 32, borderWidth: 3 }} />
              <p style={{ margin: 0 }}>
                Đang bật lại camera, vui lòng chờ…
              </p>
            </div>
          </div>
        </div>
      )}

      {showRestartFailedModal && (
        <div className="ml-modal-overlay" onClick={() => setShowRestartFailedModal(false)}>
          <div className="ml-modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 460 }}>
            <div className="ml-modal-header">
              <h3>⚠️ Khởi động lại thất bại</h3>
            </div>
            <div className="ml-modal-body" style={{ padding: 16, fontSize: 13, lineHeight: 1.5 }}>
              <p style={{ marginTop: 0 }}>
                Training đã hoàn tất nhưng hệ thống chưa khởi động lại được tự động.
              </p>
              <p style={{ marginBottom: 0 }}>
                Vui lòng khởi động lại thủ công để áp dụng model mới.
              </p>
            </div>
            <div className="ml-modal-footer">
              <button className="ml-btn ml-btn-primary"
                onClick={() => setShowRestartFailedModal(false)}>
                OK
              </button>
            </div>
          </div>
        </div>
      )}

      {showCloseDuringTrainModal && (
        <div className="ml-modal-overlay" onClick={() => setShowCloseDuringTrainModal(false)}>
          <div className="ml-modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 440 }}>
            <div className="ml-modal-header">
              <h3>⏳ Đang training</h3>
            </div>
            <div className="ml-modal-body" style={{ padding: 16, fontSize: 13, lineHeight: 1.5 }}>
              <p style={{ margin: 0 }}>
                Quá trình training đang chạy. Khi xong, hệ thống sẽ tự khởi động lại.
              </p>
            </div>
            <div className="ml-modal-footer">
              <button className="ml-btn ml-btn-secondary"
                onClick={() => setShowCloseDuringTrainModal(false)}>
                Ở lại
              </button>
              <button className="ml-btn ml-btn-primary"
                onClick={handleConfirmCloseDuringTrain}>
                Đóng
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
