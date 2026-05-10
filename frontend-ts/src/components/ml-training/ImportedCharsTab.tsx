import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  mlTrainingAPI, MLProject, CharImportBatch, CharImportItem,
} from '@/services/mlTraining';
import ImportFromInspectionsModal from './ImportFromInspectionsModal';
import ConfirmDialog from '@/components/shared/ConfirmDialog';
import { useToast } from '@/contexts/ToastContext';

interface ConfirmDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  confirmText?: string;
  onConfirm: (() => void) | null;
}

interface RenameDialogState {
  isOpen: boolean;
  initialValue: string;
  onSubmit: ((next: string) => void) | null;
}

interface Props {
  project: MLProject;
  onRefresh: () => void;
  // Deep-link from the Train-tab "Open in Imported Chars" button — auto-opens
  // the matching batch and scrolls the char card into view.
  deepLink?: { tab: 'imports'; batchId: string; charId: string } | null;
  onDeepLinkConsumed?: () => void;
}

/**
 * Imported Chars tab — manages the active-learning pool.
 *
 * Each batch is a single import session (one click of "Import" in the
 * ImportFromInspectionsModal). Inside a batch, chars can be re-labelled
 * OK/NG, char_id edited inline, or deleted individually / in bulk.
 *
 * The Train tab pulls labelled chars (any batch, any char) into its dataset
 * automatically when `include_imported_chars` is on.
 */
export default function ImportedCharsTab({ project, onRefresh, deepLink, onDeepLinkConsumed }: Props) {
  const toast = useToast();
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false, title: '', message: '', type: 'warning', onConfirm: null,
  });
  const [renameDialog, setRenameDialog] = useState<RenameDialogState>({
    isOpen: false, initialValue: '', onSubmit: null,
  });
  const [renameValue, setRenameValue] = useState('');

  const [batches, setBatches] = useState<CharImportBatch[]>([]);
  const [loadingBatches, setLoadingBatches] = useState(false);
  const [importOpen, setImportOpen] = useState(false);

  // Open batch → its chars cached by batch_id
  const [openBatchId, setOpenBatchId] = useState<string | null>(null);
  const [charsByBatch, setCharsByBatch] = useState<Map<string, CharImportItem[]>>(new Map());
  const [loadingChars, setLoadingChars] = useState(false);

  // Selection (across the open batch)
  const [selected, setSelected] = useState<Set<string>>(new Set());
  // Inline edit state for char_id
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingValue, setEditingValue] = useState('');

  // Top-level filter: all / OK / NG
  const [labelFilter, setLabelFilter] = useState<'all' | 'OK' | 'NG'>('all');

  const loadBatches = useCallback(async () => {
    setLoadingBatches(true);
    try {
      const list = await mlTrainingAPI.listCharImportBatches(project.id);
      setBatches(list);
      // If currently-open batch was deleted, collapse
      if (openBatchId && !list.some(b => b.id === openBatchId)) {
        setOpenBatchId(null);
      }
    } catch (e) {
      console.error('[ImportedCharsTab] loadBatches failed', e);
    } finally {
      setLoadingBatches(false);
    }
  }, [project.id, openBatchId]);

  const loadChars = useCallback(async (batchId: string) => {
    setLoadingChars(true);
    try {
      const items = await mlTrainingAPI.listCharImports(project.id, { batch_id: batchId });
      setCharsByBatch(prev => {
        const next = new Map(prev);
        next.set(batchId, items);
        return next;
      });
    } catch (e) {
      console.error('[ImportedCharsTab] loadChars failed', e);
    } finally {
      setLoadingChars(false);
    }
  }, [project.id]);

  useEffect(() => {
    loadBatches();
  }, [project.id]);

  // Auto-load chars when batch is opened
  useEffect(() => {
    if (openBatchId && !charsByBatch.has(openBatchId)) {
      loadChars(openBatchId);
    }
  }, [openBatchId]);

  // Track a charId we should scroll-to once its batch finishes loading. This
  // is the deep-link "Open in Imported Chars" path from the Train-tab popover.
  const [pendingScrollChar, setPendingScrollChar] = useState<string | null>(null);
  useEffect(() => {
    if (!deepLink) return;
    setOpenBatchId(deepLink.batchId);
    setPendingScrollChar(deepLink.charId);
    onDeepLinkConsumed?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deepLink]);

  // Scroll + highlight pulse once the batch chars are in state.
  useEffect(() => {
    if (!pendingScrollChar || !openBatchId) return;
    if (!charsByBatch.has(openBatchId)) return;  // wait for load
    const el = document.querySelector<HTMLElement>(
      `[data-imp-char-id="${CSS.escape(pendingScrollChar)}"]`,
    );
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      el.classList.add('ml-imp-card-flash');
      setTimeout(() => el.classList.remove('ml-imp-card-flash'), 1600);
    }
    setPendingScrollChar(null);
  }, [pendingScrollChar, openBatchId, charsByBatch]);

  const handleToggleBatch = (batchId: string) => {
    setOpenBatchId(prev => (prev === batchId ? null : batchId));
    setSelected(new Set());
  };

  const handleRenameBatch = (batch: CharImportBatch) => {
    setRenameValue(batch.name);
    setRenameDialog({
      isOpen: true,
      initialValue: batch.name,
      onSubmit: async (next: string) => {
        const trimmed = next.trim();
        if (!trimmed || trimmed === batch.name) return;
        try {
          await mlTrainingAPI.renameCharImportBatch(project.id, batch.id, trimmed);
          setBatches(prev => prev.map(b => b.id === batch.id ? { ...b, name: trimmed } : b));
          toast.success('Batch renamed');
        } catch (e: any) {
          toast.error(e?.response?.data?.detail ?? 'Rename failed');
        }
      },
    });
  };

  const handleDeleteBatch = (batch: CharImportBatch) => {
    setConfirmDialog({
      isOpen: true,
      title: 'Delete batch',
      message: `Delete batch "${batch.name}" and all ${batch.total} chars in it?`,
      type: 'danger',
      confirmText: 'Delete',
      onConfirm: async () => {
        try {
          await mlTrainingAPI.deleteCharImportBatch(project.id, batch.id);
          setBatches(prev => prev.filter(b => b.id !== batch.id));
          setCharsByBatch(prev => {
            const next = new Map(prev);
            next.delete(batch.id);
            return next;
          });
          if (openBatchId === batch.id) setOpenBatchId(null);
          onRefresh();
          toast.success('Batch deleted');
        } catch (e: any) {
          toast.error(e?.response?.data?.detail ?? 'Delete failed');
        }
      },
    });
  };

  const handleImported = async (result: { imported: number; skipped: number; errors: any[] }) => {
    await loadBatches();
    onRefresh();
    if (result.imported > 0) {
      const skip = result.skipped > 0 ? ` · skipped ${result.skipped}` : '';
      toast.success(`Imported ${result.imported} char(s)${skip}`);
    } else if (result.errors?.length) {
      toast.error(`Import failed: ${result.errors[0].reason ?? 'unknown'}`);
    } else {
      toast.warning('Nothing imported (all candidates already in pool)');
    }
  };

  // ── Per-char actions ───────────────────────────────────────────────────
  // Direct setter — clicking OK/NG button picks that label rather than
  // toggling. No-op when already at the desired label.
  const setLabel = async (item: CharImportItem, nextLabel: 'OK' | 'NG') => {
    if (item.label === nextLabel) return;
    try {
      const updated = await mlTrainingAPI.updateCharImport(project.id, item.id, { label: nextLabel });
      patchCharInState(updated);
      bumpBatchCount(item.batch_id, item.label, nextLabel);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Update failed');
    }
  };

  const startEditChar = (item: CharImportItem) => {
    setEditingId(item.id);
    setEditingValue(item.char_id ?? '');
  };

  const commitEditChar = async (item: CharImportItem) => {
    const next = editingValue.trim();
    setEditingId(null);
    if ((next || null) === (item.char_id || null)) return;
    try {
      const updated = await mlTrainingAPI.updateCharImport(project.id, item.id, { char_id: next || null });
      patchCharInState(updated);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Update failed');
    }
  };

  const deleteChar = (item: CharImportItem) => {
    setConfirmDialog({
      isOpen: true,
      title: 'Delete char',
      message: `Delete this "${item.char_id ?? '?'}" char?`,
      type: 'danger',
      confirmText: 'Delete',
      onConfirm: async () => {
        try {
          await mlTrainingAPI.deleteCharImport(project.id, item.id);
          removeCharFromState(item);
        } catch (e: any) {
          toast.error(e?.response?.data?.detail ?? 'Delete failed');
        }
      },
    });
  };

  // ── Bulk actions ───────────────────────────────────────────────────────
  const handleBulkSet = async (label: 'OK' | 'NG') => {
    if (!openBatchId || selected.size === 0) return;
    const ids = Array.from(selected);
    try {
      await mlTrainingAPI.bulkCharImports(project.id, { char_ids: ids, label });
      // Update local state + batch count
      const items = charsByBatch.get(openBatchId) || [];
      const before = items.filter(i => ids.includes(i.id));
      let okDelta = 0, ngDelta = 0;
      for (const i of before) {
        if (i.label === label) continue;
        if (label === 'OK') { okDelta += 1; ngDelta -= 1; }
        else { okDelta -= 1; ngDelta += 1; }
      }
      setCharsByBatch(prev => {
        const next = new Map(prev);
        next.set(openBatchId, items.map(i => ids.includes(i.id) ? { ...i, label } : i));
        return next;
      });
      setBatches(prev => prev.map(b =>
        b.id === openBatchId
          ? { ...b, ok_count: b.ok_count + okDelta, ng_count: b.ng_count + ngDelta }
          : b
      ));
      setSelected(new Set());
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Bulk update failed');
    }
  };

  const handleBulkDelete = () => {
    if (!openBatchId || selected.size === 0) return;
    const targetBatchId = openBatchId;
    const ids = Array.from(selected);
    setConfirmDialog({
      isOpen: true,
      title: 'Delete chars',
      message: `Delete ${ids.length} char(s)?`,
      type: 'danger',
      confirmText: 'Delete',
      onConfirm: async () => {
        try {
          await mlTrainingAPI.bulkCharImports(project.id, { char_ids: ids, delete: true });
          const items = charsByBatch.get(targetBatchId) || [];
          const removed = items.filter(i => ids.includes(i.id));
          const okDelta = -removed.filter(i => i.label === 'OK').length;
          const ngDelta = -removed.filter(i => i.label === 'NG').length;
          setCharsByBatch(prev => {
            const next = new Map(prev);
            next.set(targetBatchId, items.filter(i => !ids.includes(i.id)));
            return next;
          });
          setBatches(prev => prev.map(b =>
            b.id === targetBatchId
              ? { ...b, total: b.total + okDelta + ngDelta, ok_count: b.ok_count + okDelta, ng_count: b.ng_count + ngDelta }
              : b
          ));
          setSelected(new Set());
          toast.success(`Deleted ${ids.length} char(s)`);
        } catch (e: any) {
          toast.error(e?.response?.data?.detail ?? 'Bulk delete failed');
        }
      },
    });
  };

  // ── State helpers ──────────────────────────────────────────────────────
  const patchCharInState = (updated: CharImportItem) => {
    setCharsByBatch(prev => {
      const next = new Map(prev);
      const arr = next.get(updated.batch_id) || [];
      next.set(updated.batch_id, arr.map(i => i.id === updated.id ? updated : i));
      return next;
    });
  };

  const removeCharFromState = (item: CharImportItem) => {
    setCharsByBatch(prev => {
      const next = new Map(prev);
      const arr = next.get(item.batch_id) || [];
      next.set(item.batch_id, arr.filter(i => i.id !== item.id));
      return next;
    });
    setBatches(prev => prev.map(b =>
      b.id === item.batch_id
        ? {
            ...b,
            total: b.total - 1,
            ok_count: b.ok_count - (item.label === 'OK' ? 1 : 0),
            ng_count: b.ng_count - (item.label === 'NG' ? 1 : 0),
          }
        : b
    ));
  };

  const bumpBatchCount = (batchId: string, fromLabel: string, toLabel: string) => {
    setBatches(prev => prev.map(b => {
      if (b.id !== batchId) return b;
      const ok = b.ok_count + (toLabel === 'OK' ? 1 : 0) - (fromLabel === 'OK' ? 1 : 0);
      const ng = b.ng_count + (toLabel === 'NG' ? 1 : 0) - (fromLabel === 'NG' ? 1 : 0);
      return { ...b, ok_count: ok, ng_count: ng };
    }));
  };

  // ── Derived ────────────────────────────────────────────────────────────
  const totals = useMemo(() => {
    const t = batches.reduce(
      (acc, b) => ({ total: acc.total + b.total, ok: acc.ok + b.ok_count, ng: acc.ng + b.ng_count }),
      { total: 0, ok: 0, ng: 0 },
    );
    return t;
  }, [batches]);

  const visibleChars = useMemo(() => {
    if (!openBatchId) return [];
    const all = charsByBatch.get(openBatchId) || [];
    if (labelFilter === 'all') return all;
    return all.filter(c => c.label === labelFilter);
  }, [openBatchId, charsByBatch, labelFilter]);

  const allSelectedInView = visibleChars.length > 0 && visibleChars.every(c => selected.has(c.id));

  const toggleSelectAllVisible = () => {
    if (allSelectedInView) {
      setSelected(new Set());
    } else {
      setSelected(new Set(visibleChars.map(c => c.id)));
    }
  };

  const fmtDate = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleString('en-GB', {
      day: '2-digit', month: '2-digit', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  };

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="ml-imp-tab">
      {/* Header */}
      <div className="ml-imp-toolbar">
        <button className="ml-btn ml-btn-primary" onClick={() => setImportOpen(true)}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" style={{ marginRight: 4 }}>
            <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          Import from Inspections
        </button>
        <span className="ml-imp-totals">
          <b>{totals.total}</b> chars total
          {' · '}<span className="ok-count">{totals.ok} OK</span>
          {' · '}<span className="ng-count">{totals.ng} NG</span>
          {' · '}<span className="muted">{batches.length} batch(es)</span>
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className="ml-imp-filter-label">Filter:</span>
          {(['all', 'OK', 'NG'] as const).map(f => (
            <button key={f}
              className={`ml-augment-chip ${labelFilter === f ? 'selected' : ''}`}
              onClick={() => setLabelFilter(f)}
            >{f}</button>
          ))}
        </div>
      </div>

      {/* Batches */}
      {loadingBatches && (
        <div className="ml-empty-state"><div className="ml-loading-spinner" /></div>
      )}

      {!loadingBatches && batches.length === 0 && (
        <div className="ml-empty-state" style={{ padding: 40 }}>
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" style={{ opacity: .4 }}>
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span style={{ marginTop: 8, fontSize: 13 }}>
            No imported chars yet. Click <b>Import from Inspections</b> to start.
          </span>
        </div>
      )}

      {batches.map(b => {
        const isOpen = openBatchId === b.id;
        return (
          <div key={b.id} className="ml-imp-batch">
            <div className="ml-imp-batch-header">
              <button className="ml-imp-batch-toggle" onClick={() => handleToggleBatch(b.id)}>
                <span style={{ width: 12, textAlign: 'center' }}>{isOpen ? '▼' : '▶'}</span>
                <span className="ml-imp-batch-name">{b.name}</span>
                <span className="ml-imp-batch-date">{fmtDate(b.created_at)}</span>
                <span className="ml-imp-batch-meta">· {b.total} chars</span>
                <span className="ok-count" style={{ color: '#16a34a' }}>{b.ok_count} OK</span>
                <span className="ng-count" style={{ color: '#dc2626' }}>{b.ng_count} NG</span>
              </button>
              <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={() => handleRenameBatch(b)} title="Rename batch">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                  <path d="M12 20h9M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z"
                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
              <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={() => handleDeleteBatch(b)} title="Delete batch">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                  <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2M10 11v6M14 11v6"
                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </div>

            {/* Char grid (only when open) */}
            {isOpen && (
              <div className="ml-imp-batch-body">
                {/* Bulk action bar */}
                <div className="ml-imp-bulk-bar">
                  <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                    <input type="checkbox" checked={allSelectedInView} onChange={toggleSelectAllVisible}
                           disabled={visibleChars.length === 0} />
                    <span>Select all ({selected.size}/{visibleChars.length})</span>
                  </label>
                  <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={() => handleBulkSet('OK')} disabled={selected.size === 0}>
                    Set OK
                  </button>
                  <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={() => handleBulkSet('NG')} disabled={selected.size === 0}>
                    Set NG
                  </button>
                  <button className="ml-btn ml-btn-danger ml-btn-sm" onClick={handleBulkDelete} disabled={selected.size === 0}>
                    Delete selected
                  </button>
                  {loadingChars && <span style={{ opacity: .6 }}>Loading…</span>}
                </div>

                {/* Char grid */}
                <div className="ml-imp-grid">
                  {visibleChars.map(item => (
                    <CharCard
                      key={item.id}
                      item={item}
                      selected={selected.has(item.id)}
                      editing={editingId === item.id}
                      editingValue={editingValue}
                      onToggleSelect={() => {
                        setSelected(prev => {
                          const next = new Set(prev);
                          if (next.has(item.id)) next.delete(item.id); else next.add(item.id);
                          return next;
                        });
                      }}
                      onSetLabel={(label) => setLabel(item, label)}
                      onStartEdit={() => startEditChar(item)}
                      onChangeEdit={setEditingValue}
                      onCommitEdit={() => commitEditChar(item)}
                      onCancelEdit={() => setEditingId(null)}
                      onDelete={() => deleteChar(item)}
                    />
                  ))}
                </div>

                {!loadingChars && visibleChars.length === 0 && (
                  <div className="ml-empty-state" style={{ padding: 16, fontSize: 12 }}>
                    No chars match the filter.
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}

      <ImportFromInspectionsModal
        projectId={project.id}
        open={importOpen}
        onClose={() => setImportOpen(false)}
        onImported={handleImported}
      />

      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
        confirmText={confirmDialog.confirmText ?? 'Confirm'}
        onClose={() => setConfirmDialog(prev => ({ ...prev, isOpen: false }))}
        onConfirm={() => confirmDialog.onConfirm?.()}
      />

      {renameDialog.isOpen && (
        <div className="ml-modal-overlay" onClick={() => setRenameDialog(prev => ({ ...prev, isOpen: false }))}>
          <div className="ml-modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 420 }}>
            <div className="ml-modal-header">
              <h3>Rename batch</h3>
              <button className="ml-modal-close" onClick={() => setRenameDialog(prev => ({ ...prev, isOpen: false }))}>×</button>
            </div>
            <div className="ml-modal-body">
              <label className="ml-label">New name</label>
              <input
                className="ml-form-input"
                autoFocus
                value={renameValue}
                onChange={e => setRenameValue(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    renameDialog.onSubmit?.(renameValue);
                    setRenameDialog(prev => ({ ...prev, isOpen: false }));
                  } else if (e.key === 'Escape') {
                    setRenameDialog(prev => ({ ...prev, isOpen: false }));
                  }
                }}
              />
            </div>
            <div className="ml-modal-footer">
              <button className="ml-btn ml-btn-secondary"
                onClick={() => setRenameDialog(prev => ({ ...prev, isOpen: false }))}>Cancel</button>
              <button className="ml-btn ml-btn-primary"
                onClick={() => {
                  renameDialog.onSubmit?.(renameValue);
                  setRenameDialog(prev => ({ ...prev, isOpen: false }));
                }}>Rename</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Single char card ─────────────────────────────────────────────────────

interface CharCardProps {
  item: CharImportItem;
  selected: boolean;
  editing: boolean;
  editingValue: string;
  onToggleSelect: () => void;
  onSetLabel: (label: 'OK' | 'NG') => void;
  onStartEdit: () => void;
  onChangeEdit: (v: string) => void;
  onCommitEdit: () => void;
  onCancelEdit: () => void;
  onDelete: () => void;
}

function CharCard({
  item, selected, editing, editingValue,
  onToggleSelect, onSetLabel, onStartEdit, onChangeEdit, onCommitEdit, onCancelEdit, onDelete,
}: CharCardProps) {
  return (
    <div className={`ml-imp-card ${selected ? 'selected' : ''}`}
      data-imp-char-id={item.id}>
      <input
        type="checkbox"
        className="ml-imp-card-checkbox"
        checked={selected}
        onChange={onToggleSelect}
      />

      <button className="ml-imp-card-delete" onClick={onDelete} title="Delete">×</button>

      <div className="ml-imp-card-img-wrap">
        <img className="ml-imp-card-img" src={item.crop_url} alt={item.char_id ?? '?'} />
      </div>

      <div className="ml-imp-card-footer">
        {editing ? (
          <input
            autoFocus
            className="ml-imp-charid-input"
            value={editingValue}
            onChange={e => onChangeEdit(e.target.value)}
            onBlur={onCommitEdit}
            onKeyDown={e => {
              if (e.key === 'Enter') onCommitEdit();
              else if (e.key === 'Escape') onCancelEdit();
            }}
          />
        ) : (
          <button className="ml-imp-charid" onClick={onStartEdit} title="Click to edit char_id">
            {item.char_id || '—'}
          </button>
        )}
        <div className="ml-imp-label-toggle" role="group" aria-label="Pick label">
          <button
            type="button"
            className={`ok ${item.label === 'OK' ? 'active' : ''}`}
            onClick={() => onSetLabel('OK')}
            title="Mark as OK"
          >OK</button>
          <button
            type="button"
            className={`ng ${item.label === 'NG' ? 'active' : ''}`}
            onClick={() => onSetLabel('NG')}
            title="Mark as NG"
          >NG</button>
        </div>
      </div>

      <div className="ml-imp-card-meta">
        ML: {item.ml_label} · {(item.ml_p_ok * 100).toFixed(0)}%
      </div>
    </div>
  );
}
