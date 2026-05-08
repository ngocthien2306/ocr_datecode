import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  mlTrainingAPI, MLProject, CharImportBatch, CharImportItem,
} from '@/services/mlTraining';
import ImportFromInspectionsModal from './ImportFromInspectionsModal';

interface Props {
  project: MLProject;
  onRefresh: () => void;
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
export default function ImportedCharsTab({ project, onRefresh }: Props) {
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

  const handleToggleBatch = (batchId: string) => {
    setOpenBatchId(prev => (prev === batchId ? null : batchId));
    setSelected(new Set());
  };

  const handleRenameBatch = async (batch: CharImportBatch) => {
    const next = prompt('Batch name:', batch.name);
    if (next === null) return;
    const trimmed = next.trim();
    if (!trimmed || trimmed === batch.name) return;
    try {
      await mlTrainingAPI.renameCharImportBatch(project.id, batch.id, trimmed);
      setBatches(prev => prev.map(b => b.id === batch.id ? { ...b, name: trimmed } : b));
    } catch (e: any) {
      alert(e?.response?.data?.detail ?? 'Rename failed');
    }
  };

  const handleDeleteBatch = async (batch: CharImportBatch) => {
    if (!confirm(`Delete batch "${batch.name}" and all ${batch.total} chars in it?`)) return;
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
    } catch (e: any) {
      alert(e?.response?.data?.detail ?? 'Delete failed');
    }
  };

  const handleImported = async () => {
    await loadBatches();
    onRefresh();
  };

  // ── Per-char actions ───────────────────────────────────────────────────
  const flipLabel = async (item: CharImportItem) => {
    const nextLabel = item.label === 'OK' ? 'NG' : 'OK';
    try {
      const updated = await mlTrainingAPI.updateCharImport(project.id, item.id, { label: nextLabel });
      patchCharInState(updated);
      bumpBatchCount(item.batch_id, item.label, nextLabel);
    } catch (e: any) {
      alert(e?.response?.data?.detail ?? 'Update failed');
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
      alert(e?.response?.data?.detail ?? 'Update failed');
    }
  };

  const deleteChar = async (item: CharImportItem) => {
    if (!confirm(`Delete this "${item.char_id ?? '?'}" char?`)) return;
    try {
      await mlTrainingAPI.deleteCharImport(project.id, item.id);
      removeCharFromState(item);
    } catch (e: any) {
      alert(e?.response?.data?.detail ?? 'Delete failed');
    }
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
      alert(e?.response?.data?.detail ?? 'Bulk update failed');
    }
  };

  const handleBulkDelete = async () => {
    if (!openBatchId || selected.size === 0) return;
    if (!confirm(`Delete ${selected.size} char(s)?`)) return;
    const ids = Array.from(selected);
    try {
      await mlTrainingAPI.bulkCharImports(project.id, { char_ids: ids, delete: true });
      const items = charsByBatch.get(openBatchId) || [];
      const removed = items.filter(i => ids.includes(i.id));
      const okDelta = -removed.filter(i => i.label === 'OK').length;
      const ngDelta = -removed.filter(i => i.label === 'NG').length;
      setCharsByBatch(prev => {
        const next = new Map(prev);
        next.set(openBatchId, items.filter(i => !ids.includes(i.id)));
        return next;
      });
      setBatches(prev => prev.map(b =>
        b.id === openBatchId
          ? { ...b, total: b.total + okDelta + ngDelta, ok_count: b.ok_count + okDelta, ng_count: b.ng_count + ngDelta }
          : b
      ));
      setSelected(new Set());
    } catch (e: any) {
      alert(e?.response?.data?.detail ?? 'Bulk delete failed');
    }
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
    <div className="ml-imported-tab" style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 12, overflow: 'auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <button className="ml-btn ml-btn-primary" onClick={() => setImportOpen(true)}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" style={{ marginRight: 4 }}>
            <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          Import from Inspections
        </button>
        <span style={{ fontSize: 13, opacity: .8 }}>
          <b>{totals.total}</b> chars total
          {' · '}<span style={{ color: '#22c55e' }}>{totals.ok} OK</span>
          {' · '}<span style={{ color: '#ef4444' }}>{totals.ng} NG</span>
          {' · '}<span style={{ opacity: .7 }}>{batches.length} batch(es)</span>
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 11, opacity: .7 }}>Filter:</span>
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
          <div key={b.id} className="ml-char-group" style={{ borderRadius: 6 }}>
            <div className="ml-char-group-header" style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px' }}>
              <button onClick={() => handleToggleBatch(b.id)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', display: 'flex', alignItems: 'center', gap: 8, flex: 1, textAlign: 'left' }}>
                <span style={{ width: 12, textAlign: 'center' }}>{isOpen ? '▼' : '▶'}</span>
                <span style={{ fontWeight: 600 }}>{b.name}</span>
                <span style={{ fontSize: 11, opacity: .65 }}>{fmtDate(b.created_at)}</span>
                <span style={{ fontSize: 12, opacity: .8 }}>· {b.total} chars</span>
                <span style={{ color: '#22c55e' }}>{b.ok_count} OK</span>
                <span style={{ color: '#ef4444' }}>{b.ng_count} NG</span>
              </button>
              <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={() => handleRenameBatch(b)} title="Rename batch">
                ✏️
              </button>
              <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={() => handleDeleteBatch(b)} title="Delete batch">
                🗑
              </button>
            </div>

            {/* Char grid (only when open) */}
            {isOpen && (
              <div style={{ padding: 10, borderTop: '1px solid #2d3148' }}>
                {/* Bulk action bar */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                    <input type="checkbox" checked={allSelectedInView} onChange={toggleSelectAllVisible}
                           disabled={visibleChars.length === 0} />
                    <span style={{ fontSize: 12 }}>Select all ({selected.size}/{visibleChars.length})</span>
                  </label>
                  <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={() => handleBulkSet('OK')} disabled={selected.size === 0}>
                    Set OK
                  </button>
                  <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={() => handleBulkSet('NG')} disabled={selected.size === 0}>
                    Set NG
                  </button>
                  <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={handleBulkDelete} disabled={selected.size === 0}
                          style={{ color: '#ef4444' }}>
                    Delete selected
                  </button>
                  {loadingChars && <span style={{ fontSize: 11, opacity: .6 }}>Loading…</span>}
                </div>

                {/* Char grid — bigger cards (~96px) per user request */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(96px, 1fr))',
                  gap: 8,
                }}>
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
                      onFlipLabel={() => flipLabel(item)}
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
  onFlipLabel: () => void;
  onStartEdit: () => void;
  onChangeEdit: (v: string) => void;
  onCommitEdit: () => void;
  onCancelEdit: () => void;
  onDelete: () => void;
}

function CharCard({
  item, selected, editing, editingValue,
  onToggleSelect, onFlipLabel, onStartEdit, onChangeEdit, onCommitEdit, onCancelEdit, onDelete,
}: CharCardProps) {
  return (
    <div
      className={`ml-imp-card ${selected ? 'selected' : ''}`}
      style={{
        position: 'relative',
        background: '#0f1117',
        border: selected ? '2px solid #3b82f6' : '1px solid #2d3148',
        borderRadius: 5,
        padding: 4,
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}
    >
      {/* Selection checkbox top-left */}
      <input type="checkbox" checked={selected} onChange={onToggleSelect}
        style={{ position: 'absolute', top: 4, left: 4, zIndex: 2, cursor: 'pointer' }} />

      {/* Delete button top-right (visible on hover via opacity) */}
      <button onClick={onDelete}
        title="Delete"
        style={{
          position: 'absolute', top: 2, right: 2, zIndex: 2,
          width: 18, height: 18, padding: 0,
          background: 'rgba(239,68,68,.85)', color: '#fff',
          border: 'none', borderRadius: 3, cursor: 'pointer',
          fontSize: 11, lineHeight: 1,
        }}>×</button>

      {/* Crop image */}
      <div style={{ width: '100%', aspectRatio: '1', background: '#000', borderRadius: 3, overflow: 'hidden' }}>
        <img src={item.crop_url} alt={item.char_id ?? '?'}
             style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }} />
      </div>

      {/* Footer: char_id (editable inline) + label badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, justifyContent: 'space-between', minHeight: 22 }}>
        {editing ? (
          <input
            autoFocus
            value={editingValue}
            onChange={e => onChangeEdit(e.target.value)}
            onBlur={onCommitEdit}
            onKeyDown={e => {
              if (e.key === 'Enter') onCommitEdit();
              else if (e.key === 'Escape') onCancelEdit();
            }}
            style={{
              flex: 1, minWidth: 0, fontSize: 13, fontWeight: 600,
              padding: '1px 4px', background: '#1f2937', color: '#fff',
              border: '1px solid #3b82f6', borderRadius: 3,
              fontFamily: 'monospace', textAlign: 'center',
            }}
          />
        ) : (
          <button onClick={onStartEdit} title="Click to edit char_id"
            style={{
              flex: 1, minWidth: 0, padding: '1px 4px',
              background: 'transparent', border: '1px dashed transparent',
              color: '#fff', fontFamily: 'monospace', fontSize: 13,
              fontWeight: 600, cursor: 'pointer', textAlign: 'center',
              borderRadius: 3,
            }}
            onMouseEnter={e => (e.currentTarget.style.borderColor = '#3b82f6')}
            onMouseLeave={e => (e.currentTarget.style.borderColor = 'transparent')}
          >{item.char_id || '—'}</button>
        )}
        <button onClick={onFlipLabel} title={`Click to flip → ${item.label === 'OK' ? 'NG' : 'OK'}`}
          className={`ml-label-badge ${item.label === 'OK' ? 'ok' : 'ng'}`}
          style={{ cursor: 'pointer', border: 'none' }}>
          {item.label}
        </button>
      </div>

      {/* ML hint (small, secondary) */}
      <div style={{ fontSize: 9, opacity: .55, textAlign: 'center', lineHeight: 1.2 }}>
        ML: {item.ml_label} · {(item.ml_p_ok * 100).toFixed(0)}%
      </div>
    </div>
  );
}
