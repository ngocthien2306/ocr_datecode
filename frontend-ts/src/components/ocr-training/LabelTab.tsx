import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ocrTrainingAPI, LabelStatus, OCRDatasetItem, SplitName,
} from '@/services/ocrTraining';

interface Props {
  projectId: string;
  refreshKey: number;
  onCountsChanged: () => void;
}

const PAGE_SIZE = 25;

const STATUS_TABS: Array<{ id: LabelStatus | 'all'; label: string }> = [
  { id: 'need_review', label: 'Need review' },
  { id: 'verified', label: 'Verified' },
  { id: 'rejected', label: 'Rejected' },
  { id: 'all', label: 'All' },
];

export default function LabelTab({ projectId, refreshKey, onCountsChanged }: Props) {
  const [status, setStatus] = useState<LabelStatus | 'all'>('need_review');
  const [split, setSplit] = useState<SplitName | ''>('');
  const [page, setPage] = useState(1);
  const [data, setData] = useState<{ items: OCRDatasetItem[]; total: number; totalPages: number }>(
    { items: [], total: 0, totalPages: 1 },
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  /** Edits not yet written. Kept separate from the fetched items so a failed
   *  save doesn't silently discard what the operator typed. */
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [focusIdx, setFocusIdx] = useState(0);
  const [bigCrops, setBigCrops] = useState(false);
  const [lightbox, setLightbox] = useState<{ item: OCRDatasetItem; b64: string } | null>(null);
  const inputRefs = useRef<Array<HTMLInputElement | null>>([]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await ocrTrainingAPI.listItems(projectId, {
        status: status === 'all' ? undefined : status,
        split: split || undefined,
        page,
        pageSize: PAGE_SIZE,
      });
      setData({ items: res.items, total: res.total, totalPages: res.total_pages || 1 });
      setDrafts({});
      setFocusIdx(0);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Failed to load items');
    } finally {
      setLoading(false);
    }
  }, [projectId, status, split, page]);

  useEffect(() => { load(); }, [load, refreshKey]);
  useEffect(() => { setPage(1); }, [status, split]);

  const items = data.items;
  const dirtyCount = useMemo(
    () => items.filter((i) => drafts[i.id] !== undefined && drafts[i.id] !== i.gt_text).length,
    [items, drafts],
  );

  const textOf = (i: OCRDatasetItem) => drafts[i.id] ?? i.gt_text;

  const save = async (item: OCRDatasetItem, opts: { verify?: boolean } = {}) => {
    const text = textOf(item);
    if (opts.verify && !text.trim()) {
      setError('Cannot verify an empty label — OpenOCR drops empty-label samples, '
             + 'so it would shrink the training set instead of erroring.');
      return false;
    }
    setBusyId(item.id);
    setError(null);
    try {
      await ocrTrainingAPI.relabelItem(projectId, item.id, {
        gt_text: text !== item.gt_text ? text : undefined,
        status: opts.verify ? 'verified' : undefined,
      });
      // Patch in place rather than refetching: a refetch on every keystroke-save
      // would reshuffle the page under the operator's cursor mid-review.
      setData((prev) => ({
        ...prev,
        items: prev.items.map((i) => i.id === item.id
          ? { ...i, gt_text: text, status: opts.verify ? 'verified' : i.status }
          : i),
      }));
      setDrafts((p) => { const n = { ...p }; delete n[item.id]; return n; });
      onCountsChanged();
      return true;
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Save failed');
      return false;
    } finally {
      setBusyId(null);
    }
  };

  const reject = async (item: OCRDatasetItem) => {
    setBusyId(item.id);
    try {
      await ocrTrainingAPI.relabelItem(projectId, item.id, { status: 'rejected' });
      setData((prev) => ({
        ...prev,
        items: prev.items.map((i) => i.id === item.id ? { ...i, status: 'rejected' } : i),
      }));
      onCountsChanged();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Reject failed');
    } finally {
      setBusyId(null);
    }
  };

  const focusRow = (idx: number) => {
    const clamped = Math.max(0, Math.min(items.length - 1, idx));
    setFocusIdx(clamped);
    inputRefs.current[clamped]?.focus();
    inputRefs.current[clamped]?.select();
  };

  const onKeyDown = async (e: React.KeyboardEvent, item: OCRDatasetItem, idx: number) => {
    // Enter = verify + advance is the whole point of this tab: a few hundred
    // crops is only tolerable if the common case is one keystroke.
    if (e.key === 'Enter') {
      e.preventDefault();
      if (await save(item, { verify: true })) focusRow(idx + 1);
    } else if (e.key === 'ArrowDown' && e.ctrlKey) {
      e.preventDefault(); focusRow(idx + 1);
    } else if (e.key === 'ArrowUp' && e.ctrlKey) {
      e.preventDefault(); focusRow(idx - 1);
    } else if (e.key === 'd' && e.ctrlKey && item.expected_text) {
      // The recipe's expected text is right there; retyping it is wasted work.
      e.preventDefault();
      setDrafts((p) => ({ ...p, [item.id]: item.expected_text as string }));
    } else if (e.key === 'Escape') {
      setDrafts((p) => { const n = { ...p }; delete n[item.id]; return n; });
    }
  };

  const verifyAllMatched = async () => {
    // Items the recipe already confirmed: expected === recognized, so the
    // prefill is the recipe's own ground truth and there is nothing to read.
    const ids = items.filter((i) => i.verify_match === true && i.status !== 'verified'
                                 && i.gt_text.trim()).map((i) => i.id);
    if (!ids.length) return;
    if (!confirm(`Verify ${ids.length} item(s) the recipe already confirmed on this page?`)) return;
    try {
      const r = await ocrTrainingAPI.bulkStatus(projectId, ids, 'verified');
      if (r.skipped_empty_text) {
        setError(`${r.skipped_empty_text} item(s) skipped — empty label cannot be verified.`);
      }
      onCountsChanged();
      load();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Bulk verify failed');
    }
  };

  const openLightbox = async (item: OCRDatasetItem) => {
    try {
      const full = await ocrTrainingAPI.getItemFull(projectId, item.id);
      setLightbox({ item, b64: full.full_b64 });
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Could not load the full-size crop');
    }
  };

  const matchedOnPage = items.filter(
    (i) => i.verify_match === true && i.status !== 'verified' && i.gt_text.trim()).length;

  return (
    <>
      <div className="ot-toolbar">
        <div style={{ display: 'flex', gap: 4 }}>
          {STATUS_TABS.map((t) => (
            <button key={t.id}
                    className={`at-btn at-btn-sm ${status === t.id ? 'at-btn-primary' : 'at-btn-secondary'}`}
                    onClick={() => setStatus(t.id)}>
              {t.label}
            </button>
          ))}
        </div>

        <select className="at-form-input" style={{ maxWidth: 120 }} value={split}
                onChange={(e) => setSplit(e.target.value as SplitName | '')}>
          <option value="">train + test</option>
          <option value="train">train only</option>
          <option value="test">test only</option>
        </select>

        <label className="at-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <input type="checkbox" checked={bigCrops} onChange={(e) => setBigCrops(e.target.checked)} />
          big crops
        </label>

        <span className="ot-spacer" />

        {matchedOnPage > 0 && (
          <button className="at-btn at-btn-secondary at-btn-sm" onClick={verifyAllMatched}
                  title="These were confirmed by the recipe itself — expected equals what was read">
            Verify {matchedOnPage} recipe-confirmed
          </button>
        )}
        <span className="ot-kbd-hint">
          <span className="ot-kbd">Enter</span> verify + next ·{' '}
          <span className="ot-kbd">Ctrl+D</span> copy expected ·{' '}
          <span className="ot-kbd">Esc</span> undo
        </span>
      </div>

      {error && <div className="at-alert-error">{error}</div>}

      {dirtyCount > 0 && (
        <div className="at-hint" style={{ marginBottom: 8 }}>
          {dirtyCount} unsaved edit(s) on this page — press Enter on a row to save and verify it.
        </div>
      )}

      {loading && <div className="at-empty-state" style={{ padding: 30 }}><div className="at-loading-spinner" /></div>}

      {!loading && items.length === 0 && (
        <div className="at-empty-state" style={{ padding: 40 }}>
          {status === 'need_review'
            ? 'Nothing left to review. Import more crops, or switch to Verified to re-check.'
            : 'No items match this filter.'}
        </div>
      )}

      {!loading && items.map((item, idx) => {
        const text = textOf(item);
        const dirty = drafts[item.id] !== undefined && drafts[item.id] !== item.gt_text;
        return (
          <div key={item.id}
               className={`ot-label-row is-${item.status} ${idx === focusIdx ? 'is-focused' : ''}`}>
            <div className="ot-crop-wrap">
              {item.thumb_b64
                ? <img className={`ot-crop-img ${bigCrops ? 'ot-crop-lg' : ''}`}
                       src={`data:image/jpeg;base64,${item.thumb_b64}`}
                       alt={item.gt_text} onClick={() => openLightbox(item)} title="Click to zoom" />
                : <span className="at-hint">image missing on disk</span>}
            </div>

            <div className="ot-label-fields">
              <input
                ref={(el) => { inputRefs.current[idx] = el; }}
                className={`ot-gt-input ${dirty ? 'ot-dirty' : ''}`}
                value={text}
                disabled={busyId === item.id}
                placeholder="(empty — type the printed text)"
                onFocus={() => setFocusIdx(idx)}
                onChange={(e) => setDrafts((p) => ({ ...p, [item.id]: e.target.value }))}
                onKeyDown={(e) => onKeyDown(e, item, idx)}
              />

              <div className="ot-hint-row">
                {item.expected_text != null && <span className="ot-hint-exp">exp <b>{item.expected_text}</b></span>}
                {item.recognized_text != null && <span className="ot-hint-rec">read <b>{item.recognized_text}</b></span>}
                {item.ocr_confidence != null && item.ocr_confidence >= 0 && (
                  <span>conf <b>{item.ocr_confidence.toFixed(3)}</b></span>
                )}
                <span>{text.length} chars</span>
              </div>

              <div className="ot-row-actions">
                <button className="at-btn at-btn-primary at-btn-sm" disabled={busyId === item.id}
                        onClick={async () => { if (await save(item, { verify: true })) focusRow(idx + 1); }}>
                  Verify
                </button>
                <button className="at-btn at-btn-secondary at-btn-sm" disabled={busyId === item.id || !dirty}
                        onClick={() => save(item)}>
                  Save only
                </button>
                <button className="at-btn at-btn-secondary at-btn-sm" disabled={busyId === item.id}
                        onClick={() => reject(item)} title="Exclude this crop from training entirely">
                  Reject
                </button>
                <span className="ot-meta-chips">
                  <span className="ot-chip-sm">{item.status}</span>
                  {item.split === 'test' && <span className="ot-chip-sm ot-test">test</span>}
                  {item.verify_match === true && <span className="ot-chip-sm ot-pass">match</span>}
                  {item.verify_match === false && <span className="ot-chip-sm ot-fail">no match</span>}
                  {item.recipe_name && <span className="ot-chip-sm">{item.recipe_name}</span>}
                  {item.source !== 'import' && <span className="ot-chip-sm">{item.source}</span>}
                </span>
              </div>
            </div>
          </div>
        );
      })}

      {data.totalPages > 1 && (
        <div className="ot-toolbar">
          <button className="at-btn at-btn-secondary at-btn-sm" disabled={page <= 1}
                  onClick={() => setPage((p) => p - 1)}>← Prev</button>
          <span className="at-hint">Page {page} / {data.totalPages} · {data.total} item(s)</span>
          <button className="at-btn at-btn-secondary at-btn-sm" disabled={page >= data.totalPages}
                  onClick={() => setPage((p) => p + 1)}>Next →</button>
        </div>
      )}

      {lightbox && (
        <div className="at-lightbox-overlay" onClick={() => setLightbox(null)}>
          <div className="at-lightbox-card" onClick={(e) => e.stopPropagation()}>
            <div className="at-lightbox-card-header">
              <span>{lightbox.item.gt_text || '(no label)'}</span>
              <button className="at-lightbox-close" onClick={() => setLightbox(null)}>✕</button>
            </div>
            <div className="at-lightbox-card-body" style={{ overflow: 'auto' }}>
              <img src={`data:image/jpeg;base64,${lightbox.b64}`} alt={lightbox.item.gt_text}
                   style={{ maxWidth: '100%', imageRendering: 'crisp-edges' }} />
            </div>
            <div className="at-lightbox-card-footer">
              <span className="at-hint">
                {lightbox.item.recipe_name} · ann{lightbox.item.annotation_index} · {lightbox.item.region_type}
              </span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
