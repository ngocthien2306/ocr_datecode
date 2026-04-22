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
            {loadingAvailable ? '...' : '↻'}
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

        {loadingAvailable ? (
          <div className="ml-empty-state"><div className="ml-loading-spinner" /></div>
        ) : available.length === 0 ? (
          <div className="ml-empty-state">
            <span className="ml-empty-icon">📷</span>
            No images in camera buffer
          </div>
        ) : (
          <div className="ml-image-grid">
            {available.map(img => (
              <div
                key={img.filename}
                className={`ml-image-thumb ${selectedAvailable.has(img.filename) ? 'selected' : ''}`}
                onClick={() => toggleAvailable(img.filename)}
                title={img.filename}
              >
                <img src={`data:image/jpeg;base64,${img.thumbnail_b64}`} alt={img.filename} />
                {selectedAvailable.has(img.filename) && (
                  <div className="ml-image-thumb-check">✓</div>
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
              {uploading ? 'Uploading...' : '⬆ Upload'}
            </button>
            {selectedProject.size > 0 && (
              <button
                className="ml-btn ml-btn-danger ml-btn-sm"
                onClick={handleDeleteSelected}
                disabled={deleting}
              >
                {deleting ? '...' : `🗑 ${selectedProject.size}`}
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

        {loadingProject ? (
          <div className="ml-empty-state"><div className="ml-loading-spinner" /></div>
        ) : projectImages.length === 0 ? (
          <div className="ml-empty-state">
            <span className="ml-empty-icon">🖼️</span>
            No images yet.<br />Copy from camera buffer or upload files.
          </div>
        ) : (
          <div className="ml-image-grid" style={{ flex: 1 }}>
            {projectImages.map(img => (
              <div
                key={img.filename}
                className={`ml-image-thumb ${selectedProject.has(img.filename) ? 'selected' : ''} ${img.has_annotation ? 'annotated' : ''}`}
                onClick={() => toggleProject(img.filename)}
                title={img.filename}
              >
                <img src={`data:image/jpeg;base64,${img.thumbnail_b64}`} alt={img.filename} />
                {selectedProject.has(img.filename) && (
                  <div className="ml-image-thumb-check">✓</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
