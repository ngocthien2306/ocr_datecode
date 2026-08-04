import { useEffect, useState } from 'react';
import '@/styles/AnomalyTraining.css';
import { anomalyTrainingAPI, DatasetImage } from '@/services/anomalyTraining';
import ConfirmDialog from '@/components/shared/ConfirmDialog';

interface Props {
  projectId: string;
  refreshKey: number; // bump to force a re-fetch (e.g. after an import)
  onCountsChanged: () => void; // parent re-fetches dataset-stats
}

interface ConfirmDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  confirmText?: string;
  onConfirm: (() => void) | null;
}

const DEFAULT_DEFECT_TYPE = 'wrinkled';
const PAGE_SIZE = 60;

/**
 * Roboflow-style dataset review: browse every already-imported crop as a
 * paginated thumbnail grid, click one to see it full-size, relabel or
 * delete one at a time or in bulk via multi-select.
 * Complements ImportFromRecipeModal, which only adds new items.
 */
export default function DatasetGallery({ projectId, refreshKey, onCountsChanged }: Props) {
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'normal' | 'abnormal'>('all');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [defectDraft, setDefectDraft] = useState<Map<string, string>>(new Map());
  const [busyId, setBusyId] = useState<string | null>(null);

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkDefectType, setBulkDefectType] = useState(DEFAULT_DEFECT_TYPE);
  const [bulkBusy, setBulkBusy] = useState(false);

  const [lightboxId, setLightboxId] = useState<string | null>(null);
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const [lightboxLoading, setLightboxLoading] = useState(false);
  const lightboxImg = images.find((i) => i.id === lightboxId) ?? null;

  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false, title: '', message: '', type: 'danger', onConfirm: null,
  });

  const load = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const data = await anomalyTrainingAPI.listDatasetImages(projectId, {
        label: filter === 'all' ? undefined : filter,
        page,
        pageSize: PAGE_SIZE,
      });
      setImages(data.images);
      setTotalPages(data.total_pages);
      setTotal(data.total);
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Failed to load dataset');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [projectId, filter, page, refreshKey]);
  useEffect(() => { setPage(1); setSelected(new Set()); }, [filter, projectId]);

  const pageIds = images.map((i) => i.id);
  const allOnPageSelected = pageIds.length > 0 && pageIds.every((id) => selected.has(id));

  const toggleSelectPage = () => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (allOnPageSelected) pageIds.forEach((id) => next.delete(id));
      else pageIds.forEach((id) => next.add(id));
      return next;
    });
  };

  // Distinct from the page toggle on purpose: with 60 images per page, a single
  // "select all" that quietly meant only the visible ones would make a bulk
  // delete look like it had missed most of the dataset.
  const selectAllMatching = async () => {
    setBulkBusy(true);
    setErrorMsg(null);
    try {
      const res = await anomalyTrainingAPI.listDatasetImageIds(
        projectId, filter === 'all' ? undefined : filter,
      );
      setSelected(new Set(res.ids));
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || 'Could not select every image');
    } finally {
      setBulkBusy(false);
    }
  };

  const toggleSelected = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const lightboxIndex = images.findIndex((i) => i.id === lightboxId);
  const canPrev = lightboxIndex > 0;
  const canNext = lightboxIndex >= 0 && lightboxIndex < images.length - 1;

  const stepLightbox = (delta: number) => {
    const next = images[lightboxIndex + delta];
    if (next) openLightbox(next);
  };

  const openLightbox = async (img: DatasetImage) => {
    setLightboxId(img.id);
    setLightboxSrc(null);
    setLightboxLoading(true);
    try {
      const res = await anomalyTrainingAPI.getDatasetImageFull(projectId, img.id);
      setLightboxSrc(res.full_b64);
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Failed to load full image');
      setLightboxId(null);
    } finally {
      setLightboxLoading(false);
    }
  };

  const handleRelabel = async (img: DatasetImage, label: 'normal' | 'abnormal') => {
    if (label === img.label) return;
    const defectType = label === 'abnormal' ? (defectDraft.get(img.id) || img.defect_type || DEFAULT_DEFECT_TYPE) : undefined;
    setBusyId(img.id);
    setErrorMsg(null);
    try {
      await anomalyTrainingAPI.relabelDatasetImage(projectId, img.id, label, defectType);
      onCountsChanged();
      await load();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Relabel failed');
    } finally {
      setBusyId(null);
    }
  };

  const handleDefectTypeCommit = async (img: DatasetImage, value: string) => {
    if (!value || value === img.defect_type) return;
    setBusyId(img.id);
    setErrorMsg(null);
    try {
      await anomalyTrainingAPI.relabelDatasetImage(projectId, img.id, 'abnormal', value);
      onCountsChanged();
      await load();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Relabel failed');
    } finally {
      setBusyId(null);
    }
  };

  useEffect(() => {
    if (!lightboxId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') stepLightbox(-1);
      else if (e.key === 'ArrowRight') stepLightbox(1);
      else if (e.key === 'Escape') setLightboxId(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [lightboxId, lightboxIndex, images]);

  const doDelete = async (img: DatasetImage, closeLightboxAfter: boolean) => {
    setBusyId(img.id);
    setErrorMsg(null);
    try {
      await anomalyTrainingAPI.deleteDatasetImage(projectId, img.id);
      onCountsChanged();
      setSelected((prev) => { const next = new Set(prev); next.delete(img.id); return next; });
      await load();
      if (closeLightboxAfter) setLightboxId(null);
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Delete failed');
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = (img: DatasetImage, closeLightboxAfter = false) => {
    setConfirmDialog({
      isOpen: true,
      title: 'Delete image',
      message: `Delete this ${img.label} image? This can't be undone.`,
      type: 'danger',
      confirmText: 'Delete',
      onConfirm: () => doDelete(img, closeLightboxAfter),
    });
  };

  const handleBulkRelabel = async (label: 'normal' | 'abnormal') => {
    if (selected.size === 0) return;
    setBulkBusy(true);
    setErrorMsg(null);
    try {
      const res = await anomalyTrainingAPI.bulkRelabelDatasetImages(
        projectId, Array.from(selected), label, label === 'abnormal' ? bulkDefectType : undefined,
      );
      if (res.errors.length > 0) {
        setErrorMsg(`${res.updated} updated, ${res.errors.length} failed (${res.errors[0]?.reason})`);
      }
      onCountsChanged();
      setSelected(new Set());
      await load();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Bulk relabel failed');
    } finally {
      setBulkBusy(false);
    }
  };

  const doBulkDelete = async () => {
    setBulkBusy(true);
    setErrorMsg(null);
    try {
      const res = await anomalyTrainingAPI.bulkDeleteDatasetImages(projectId, Array.from(selected));
      if (res.errors.length > 0) {
        setErrorMsg(`${res.deleted} deleted, ${res.errors.length} failed (${res.errors[0]?.reason})`);
      }
      onCountsChanged();
      setSelected(new Set());
      await load();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Bulk delete failed');
    } finally {
      setBulkBusy(false);
    }
  };

  const handleBulkExclude = async (excluded: boolean) => {
    setBulkBusy(true);
    setErrorMsg(null);
    try {
      await anomalyTrainingAPI.bulkExcludeFromTraining(projectId, [...selected], excluded);
      setSelected(new Set());
      await load();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || 'Could not change training selection');
    } finally {
      setBulkBusy(false);
    }
  };

  const handleBulkDelete = () => {
    if (selected.size === 0) return;
    setConfirmDialog({
      isOpen: true,
      title: 'Delete images',
      message: `Delete ${selected.size} image(s)? This can't be undone.`,
      type: 'danger',
      confirmText: 'Delete',
      onConfirm: doBulkDelete,
    });
  };

  return (
    <div>
      <div className="at-gallery-filters">
        {(['all', 'normal', 'abnormal'] as const).map((f) => (
          <button
            key={f}
            type="button"
            className={`at-gallery-filter ${filter === f ? 'active' : ''}`}
            onClick={() => setFilter(f)}
          >{f === 'all' ? 'All' : f === 'normal' ? 'Normal' : 'Abnormal'}</button>
        ))}
        {!loading && <span className="at-hint" style={{ marginLeft: 8, alignSelf: 'center' }}>{total} image(s)</span>}

        <div className="at-gallery-select-actions">
          <button type="button" className="at-btn at-btn-secondary at-btn-sm"
                  disabled={loading || !images.length}
                  onClick={toggleSelectPage}>
            {allOnPageSelected ? 'Deselect page' : `Select page (${images.length})`}
          </button>
          {total > images.length && (
            <button type="button" className="at-btn at-btn-secondary at-btn-sm"
                    disabled={loading || bulkBusy}
                    title="Spans every page of the current filter, not just what is visible"
                    onClick={selectAllMatching}>
              Select all {total}
            </button>
          )}
          {selected.size > 0 && (
            <button type="button" className="at-btn at-btn-secondary at-btn-sm"
                    onClick={() => setSelected(new Set())}>Clear</button>
          )}
        </div>
      </div>

      {selected.size > 0 && (
        <div className="at-bulk-bar">
          <b>{selected.size} selected</b>
          {selected.size > images.length && (
            <span className="at-hint">across all pages</span>
          )}
          <button className="at-btn at-btn-secondary at-btn-sm" disabled={bulkBusy}
                  onClick={() => handleBulkRelabel('normal')}>Set Normal</button>
          <button className="at-btn at-btn-secondary at-btn-sm" disabled={bulkBusy}
                  onClick={() => handleBulkRelabel('abnormal')}>Set Abnormal</button>
          <input
            className="at-cand-defect-input"
            value={bulkDefectType}
            placeholder="defect type"
            onChange={(e) => setBulkDefectType(e.target.value)}
          />
          <button className="at-btn at-btn-secondary at-btn-sm" disabled={bulkBusy}
                  title="Hold these out of the next training run. The files are not moved or deleted."
                  onClick={() => handleBulkExclude(true)}>Exclude from training</button>
          <button className="at-btn at-btn-secondary at-btn-sm" disabled={bulkBusy}
                  onClick={() => handleBulkExclude(false)}>Include again</button>
          <button className="at-btn at-btn-secondary at-btn-sm" disabled={bulkBusy} onClick={handleBulkDelete}
                  style={{ color: '#b91c1c' }}>Delete</button>
          <button className="at-btn at-btn-secondary at-btn-sm" disabled={bulkBusy}
                  onClick={() => setSelected(new Set())} style={{ marginLeft: 'auto' }}>Clear</button>
        </div>
      )}

      {errorMsg && <div className="at-alert-error" style={{ marginBottom: 8 }}>{errorMsg}</div>}

      {loading ? (
        <div className="at-empty-state" style={{ padding: '40px 0' }}>Loading…</div>
      ) : images.length === 0 ? (
        <div className="at-empty-state" style={{ padding: '40px 0' }}>
          No images yet — use "Import from Recipe" to add some.
        </div>
      ) : (
        <>
          <div className="at-cand-grid">
            {images.map((img) => {
              const busy = busyId === img.id || bulkBusy;
              const checked = selected.has(img.id);
              return (
                <div
                  key={img.id}
                  className={`at-cand-card at-gallery-card selected-${img.label} selectable ${checked ? 'checked' : ''}`}
                  title={`${img.recipe_name ?? ''} · ${img.camera_serial} · frame ${img.frame_idx} · ${img.created_at}`}
                  style={busy ? { opacity: 0.5, pointerEvents: 'none' } : undefined}
                >
                  <input
                    type="checkbox"
                    className="at-cand-card-select"
                    checked={checked}
                    onChange={() => toggleSelected(img.id)}
                  />
                  <img
                    src={`data:image/jpeg;base64,${img.thumb_b64}`}
                    alt=""
                    className="at-cand-card-img"
                    onClick={() => openLightbox(img)}
                  />
                  <span className="at-cand-badge">
                    {img.label === 'abnormal' ? `abnormal (${img.defect_type})` : 'normal'}
                  </span>
                  {img.source === 'synthetic' && (
                    <span className="at-syn-badge" title="Drawn, not a real defect">SYN</span>
                  )}
                  {img.exclude_from_training && (
                    <span
                      className={`at-excluded-badge ${img.source === 'synthetic' ? '' : 'no-syn'}`}
                      title="Held out of the next training run"
                    >
                      EXCLUDED
                    </span>
                  )}
                  <div className="at-gallery-card-footer">
                    <div className="at-cand-card-meta">{img.camera_serial} · f{img.frame_idx}</div>
                    <div className="at-cand-card-actions">
                      <button
                        type="button"
                        className={`at-cand-toggle normal ${img.label === 'normal' ? 'active' : ''}`}
                        disabled={busy}
                        onClick={() => handleRelabel(img, 'normal')}
                      >Normal</button>
                      <button
                        type="button"
                        className={`at-cand-toggle abnormal ${img.label === 'abnormal' ? 'active' : ''}`}
                        disabled={busy}
                        onClick={() => handleRelabel(img, 'abnormal')}
                      >Abnormal</button>
                      <button
                        type="button"
                        className="at-cand-card-delete"
                        disabled={busy}
                        onClick={() => handleDelete(img)}
                        title="Delete image"
                      >×</button>
                    </div>
                    {img.label === 'abnormal' && (
                      <input
                        className="at-cand-defect-input"
                        defaultValue={img.defect_type ?? ''}
                        placeholder="defect type (e.g. wrinkled)"
                        disabled={busy}
                        onChange={(e) => setDefectDraft((prev) => new Map(prev).set(img.id, e.target.value))}
                        onBlur={(e) => handleDefectTypeCommit(img, e.target.value)}
                      />
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {totalPages > 1 && (
            <div className="at-pagination">
              <button className="at-btn at-btn-secondary at-btn-sm" disabled={page <= 1}
                      onClick={() => setPage((p) => Math.max(1, p - 1))}>‹ Prev</button>
              <span className="at-hint">Page {page} / {totalPages}</span>
              <button className="at-btn at-btn-secondary at-btn-sm" disabled={page >= totalPages}
                      onClick={() => setPage((p) => Math.min(totalPages, p + 1))}>Next ›</button>
            </div>
          )}
        </>
      )}

      {lightboxId && lightboxImg && (
        <div className="at-lightbox-overlay" onClick={() => setLightboxId(null)}>
          <div className="at-lightbox-card" onClick={(e) => e.stopPropagation()}>
            <div className="at-lightbox-card-header">
              <div className="at-cand-card-actions" style={{ padding: 0, flex: 1 }}>
                <button
                  type="button"
                  className={`at-cand-toggle normal ${lightboxImg.label === 'normal' ? 'active' : ''}`}
                  disabled={busyId === lightboxImg.id}
                  onClick={() => handleRelabel(lightboxImg, 'normal')}
                >Normal</button>
                <button
                  type="button"
                  className={`at-cand-toggle abnormal ${lightboxImg.label === 'abnormal' ? 'active' : ''}`}
                  disabled={busyId === lightboxImg.id}
                  onClick={() => handleRelabel(lightboxImg, 'abnormal')}
                >Abnormal</button>
                {lightboxImg.label === 'abnormal' && (
                  <input
                    className="at-cand-defect-input"
                    style={{ margin: 0, width: 160, flex: '0 0 auto' }}
                    defaultValue={lightboxImg.defect_type ?? ''}
                    placeholder="defect type"
                    disabled={busyId === lightboxImg.id}
                    onBlur={(e) => handleDefectTypeCommit(lightboxImg, e.target.value)}
                  />
                )}
              </div>
              <button
                type="button"
                className="at-cand-card-delete"
                style={{ flex: '0 0 30px' }}
                disabled={busyId === lightboxImg.id}
                onClick={() => handleDelete(lightboxImg, true)}
                title="Delete image"
              >×</button>
              <button className="at-lightbox-close" onClick={() => setLightboxId(null)}>×</button>
            </div>
            <div className="at-lightbox-card-body">
              <button
                type="button"
                className="at-lightbox-nav prev"
                disabled={!canPrev}
                title="Previous image (←)"
                onClick={() => stepLightbox(-1)}
              >‹</button>
              {lightboxLoading || !lightboxSrc ? (
                <div className="at-lightbox-loading">Loading…</div>
              ) : (
                <img
                  src={`data:image/jpeg;base64,${lightboxSrc}`}
                  alt=""
                  className="at-lightbox-img"
                />
              )}
              <button
                type="button"
                className="at-lightbox-nav next"
                disabled={!canNext}
                title="Next image (→)"
                onClick={() => stepLightbox(1)}
              >›</button>
            </div>
            <div className="at-lightbox-card-footer">
              {lightboxImg.recipe_name && <span>{lightboxImg.recipe_name} · </span>}
              {lightboxImg.camera_serial} · frame {lightboxImg.frame_idx} ·{' '}
              {new Date(lightboxImg.created_at).toLocaleString()}
              <span className="at-lightbox-pos">
                {lightboxIndex + 1} / {images.length} on this page
              </span>
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
        confirmText={confirmDialog.confirmText ?? 'Confirm'}
        onClose={() => setConfirmDialog((prev) => ({ ...prev, isOpen: false }))}
        onConfirm={() => confirmDialog.onConfirm?.()}
      />
    </div>
  );
}
