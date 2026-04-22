import React, { useCallback, useEffect, useRef, useState } from 'react';
import { mlTrainingAPI, AvailableImage, MLProject, ProjectImage } from '@/services/mlTraining';

interface Props {
  project: MLProject;
  onRefresh: () => void;
}

export default function ImageTab({ project, onRefresh }: Props) {
  // Available images (from camera buffer)
  const [available, setAvailable] = useState<AvailableImage[]>([]);
  const [loadingAvailable, setLoadingAvailable] = useState(false);
  const [selectedAvailable, setSelectedAvailable] = useState<Set<string>>(new Set());

  // Project images
  const [projectImages, setProjectImages] = useState<ProjectImage[]>([]);
  const [loadingProject, setLoadingProject] = useState(false);
  const [selectedProject, setSelectedProject] = useState<Set<string>>(new Set());

  const [copying, setCopying] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Grid column control (shared state for both panels)
  const [availCols, setAvailCols] = useState(3);
  const [projectCols, setProjectCols] = useState(3);

  // ── Loaders ───────────────────────────────────────────────────────────
  const loadAvailable = useCallback(async () => {
    setLoadingAvailable(true);
    try {
      const data = await mlTrainingAPI.listAvailableImages();
      setAvailable(data);
    } catch (e) { console.error(e); }
    finally { setLoadingAvailable(false); }
  }, []);

  const loadProjectImages = useCallback(async () => {
    setLoadingProject(true);
    try {
      const data = await mlTrainingAPI.listProjectImages(project.id);
      setProjectImages(data);
    } catch (e) { console.error(e); }
    finally { setLoadingProject(false); }
  }, [project.id]);

  useEffect(() => {
    loadAvailable();
    loadProjectImages();
    setSelectedAvailable(new Set());
    setSelectedProject(new Set());
  }, [project.id]);

  // ── Available image selection ──────────────────────────────────────────
  const toggleAvailable = (filename: string) => {
    setSelectedAvailable(prev => {
      const next = new Set(prev);
      next.has(filename) ? next.delete(filename) : next.add(filename);
      return next;
    });
  };

  const selectAllAvailable = () => {
    if (selectedAvailable.size === available.length) {
      setSelectedAvailable(new Set());
    } else {
      setSelectedAvailable(new Set(available.map(a => a.filename)));
    }
  };

  // ── Copy selected available → project ─────────────────────────────────
  const handleCopy = async () => {
    if (!selectedAvailable.size) return;
    setCopying(true);
    try {
      await mlTrainingAPI.copyImages(project.id, Array.from(selectedAvailable));
      setSelectedAvailable(new Set());
      await loadProjectImages();
      onRefresh();
    } catch (e) { console.error(e); }
    finally { setCopying(false); }
  };

  // ── Upload from local ──────────────────────────────────────────────────
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    setUploading(true);
    try {
      await mlTrainingAPI.uploadImages(project.id, files);
      await loadProjectImages();
      onRefresh();
    } catch (err) { console.error(err); }
    finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // ── Delete selected project images ────────────────────────────────────
  const handleDeleteSelected = async () => {
    if (!selectedProject.size) return;
    if (!confirm(`Delete ${selectedProject.size} image(s)?`)) return;
    setDeleting(true);
    try {
      await Promise.all(
        Array.from(selectedProject).map(fn => mlTrainingAPI.deleteImage(project.id, fn))
      );
      setSelectedProject(new Set());
      await loadProjectImages();
      onRefresh();
    } catch (e) { console.error(e); }
    finally { setDeleting(false); }
  };

  const toggleProject = (filename: string) => {
    setSelectedProject(prev => {
      const next = new Set(prev);
      next.has(filename) ? next.delete(filename) : next.add(filename);
      return next;
    });
  };

  const selectAllProject = () => {
    if (selectedProject.size === projectImages.length) {
      setSelectedProject(new Set());
    } else {
      setSelectedProject(new Set(projectImages.map(i => i.filename)));
    }
  };

  return (
    <div className="ml-image-tab">
      {/* Left: Available images from camera buffer */}
      <div className="ml-image-source-panel">
        <div className="ml-panel-header">
          Camera Buffer
          <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={loadAvailable} disabled={loadingAvailable}>
            {loadingAvailable ? <span className="ml-loading-spinner" style={{width:14,height:14,borderWidth:2}}/> : <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M1 4v6h6M23 20v-6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>}
          </button>
        </div>

        <div className="ml-select-all-bar">
          <input
            type="checkbox"
            checked={available.length > 0 && selectedAvailable.size === available.length}
            onChange={selectAllAvailable}
            id="avail-select-all"
          />
          <label htmlFor="avail-select-all" style={{ cursor: 'pointer' }}>
            {selectedAvailable.size > 0 ? `${selectedAvailable.size} selected` : 'Select all'}
          </label>
          {selectedAvailable.size > 0 && (
            <button className="ml-btn ml-btn-primary ml-btn-sm" onClick={handleCopy} disabled={copying} style={{ marginLeft: 'auto' }}>
              {copying ? 'Copying...' : `Copy ${selectedAvailable.size}`}
            </button>
          )}
        </div>

        {/* Grid column slider */}
        <div className="ml-grid-cols-bar">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/><rect x="14" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/><rect x="3" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/><rect x="14" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/></svg>
          <input
            type="range" min={2} max={8} step={1}
            value={availCols}
            onChange={e => setAvailCols(Number(e.target.value))}
            className="ml-cols-slider"
            title={`${availCols} columns`}
          />
          <span style={{fontSize:11,minWidth:14,textAlign:'center'}}>{availCols}</span>
        </div>

        {loadingAvailable ? (
          <div className="ml-empty-state"><div className="ml-loading-spinner" /></div>
        ) : available.length === 0 ? (
          <div className="ml-empty-state">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" style={{opacity:.4}}><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/><circle cx="12" cy="13" r="4" stroke="currentColor" strokeWidth="2"/></svg>
            No images in camera buffer
          </div>
        ) : (
          <div className="ml-image-grid" style={{gridTemplateColumns:`repeat(${availCols}, 1fr)`}}>
            {available.map(img => (
              <div
                key={img.filename}
                className={`ml-image-thumb ${selectedAvailable.has(img.filename) ? 'selected' : ''}`}
                onClick={() => toggleAvailable(img.filename)}
                title={img.filename}
              >
                <img src={`data:image/jpeg;base64,${img.thumbnail_b64}`} alt={img.filename} />
                {selectedAvailable.has(img.filename) && (
                  <div className="ml-image-thumb-check"><svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/></svg></div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Right: Project images */}
      <div className="ml-image-project-panel">
        <div className="ml-panel-header">
          Project Images
          <div className="ml-panel-actions">
            <button
              className="ml-btn ml-btn-secondary ml-btn-sm"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
            >
              {uploading ? 'Uploading...' : <><svg width="13" height="13" viewBox="0 0 24 24" fill="none"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg> Upload</>}
            </button>
            {selectedProject.size > 0 && (
              <button
                className="ml-btn ml-btn-danger ml-btn-sm"
                onClick={handleDeleteSelected}
                disabled={deleting}
              >
                {deleting ? '...' : <><svg width="12" height="12" viewBox="0 0 24 24" fill="none"><polyline points="3 6 5 6 21 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6M10 11v6M14 11v6M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg> {selectedProject.size}</>}
              </button>
            )}
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />

        <div className="ml-select-all-bar">
          <input
            type="checkbox"
            checked={projectImages.length > 0 && selectedProject.size === projectImages.length}
            onChange={selectAllProject}
            id="proj-select-all"
          />
          <label htmlFor="proj-select-all" style={{ cursor: 'pointer' }}>
            {selectedProject.size > 0
              ? `${selectedProject.size} selected`
              : `${projectImages.length} images`}
          </label>
        </div>

        {/* Grid column slider */}
        <div className="ml-grid-cols-bar">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/><rect x="14" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/><rect x="3" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/><rect x="14" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/></svg>
          <input
            type="range" min={2} max={8} step={1}
            value={projectCols}
            onChange={e => setProjectCols(Number(e.target.value))}
            className="ml-cols-slider"
            title={`${projectCols} columns`}
          />
          <span style={{fontSize:11,minWidth:14,textAlign:'center'}}>{projectCols}</span>
        </div>

        {loadingProject ? (
          <div className="ml-empty-state"><div className="ml-loading-spinner" /></div>
        ) : projectImages.length === 0 ? (
          <div className="ml-empty-state">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" style={{opacity:.4}}><rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/><circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/><path d="M21 15l-5-5L5 21" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/></svg>
            No images yet.<br />Copy from camera buffer or upload files.
          </div>
        ) : (
          <div className="ml-image-grid" style={{gridTemplateColumns:`repeat(${projectCols}, 1fr)`}}>
            {projectImages.map(img => (
              <div
                key={img.filename}
                className={`ml-image-thumb ${selectedProject.has(img.filename) ? 'selected' : ''} ${img.has_annotation ? 'annotated' : ''}`}
                onClick={() => toggleProject(img.filename)}
                title={img.filename}
              >
                <img src={`data:image/jpeg;base64,${img.thumbnail_b64}`} alt={img.filename} />
                {selectedProject.has(img.filename) && (
                  <div className="ml-image-thumb-check"><svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/></svg></div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
