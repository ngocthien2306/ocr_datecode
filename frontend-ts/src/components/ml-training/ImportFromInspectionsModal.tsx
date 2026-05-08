import { useEffect, useMemo, useState } from 'react';
import { mlTrainingAPI, InspectionCandidate } from '@/services/mlTraining';
import { recipesAPI } from '@/services/recipes';
import type { Recipe } from '@/types/index';

interface ImportResult {
  batch_id: string | null;
  batch_name?: string;
  imported: number;
  skipped: number;
  errors: Array<{ inspection_id: string; reason: string }>;
}

interface Props {
  projectId: string;
  open: boolean;
  onClose: () => void;
  onImported: (result: ImportResult) => void;
}

/**
 * Search past inspection results for char-level training candidates and
 * create a new batch in the Imported Chars pool.
 *
 * Filter is simplified to the model's predicted label (OK/NG) — both checked
 * by default so the user sees everything, then picks specific cards. Already-
 * imported candidates show a badge + disabled checkbox so the user can't
 * re-import the same (inspection, annotation) pair twice.
 */
export default function ImportFromInspectionsModal({
  projectId, open, onClose, onImported,
}: Props) {
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [recipeId, setRecipeId] = useState<string>('');

  const today = useMemo(() => new Date().toISOString().slice(0, 10), []);
  const sevenDaysAgo = useMemo(() => {
    const d = new Date();
    d.setDate(d.getDate() - 7);
    return d.toISOString().slice(0, 10);
  }, []);

  const [dateFrom, setDateFrom] = useState<string>(sevenDaysAgo);
  const [dateTo, setDateTo] = useState<string>(today);
  const [includePredOk, setIncludePredOk] = useState(true);
  const [includePredNg, setIncludePredNg] = useState(true);
  const [limit, setLimit] = useState(100);

  const [candidates, setCandidates] = useState<InspectionCandidate[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [batchName, setBatchName] = useState<string>('');
  const [importing, setImporting] = useState(false);

  useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const list = await recipesAPI.getAllRecipes(0, 100);
        setRecipes(list);
      } catch (e: any) {
        console.error('[ImportFromInspectionsModal] load recipes failed:', e);
        setErrorMsg(e?.response?.data?.detail || e?.message || 'Failed to load recipes');
      }
    })();
  }, [open]);

  // Reset transient state when re-opened
  useEffect(() => {
    if (!open) {
      setCandidates([]);
      setSelected(new Set());
      setErrorMsg(null);
      setBatchName('');
    }
  }, [open]);

  const handleSearch = async () => {
    if (!includePredOk && !includePredNg) {
      setErrorMsg('Pick at least one predicted label (OK or NG)');
      return;
    }
    setLoading(true);
    setErrorMsg(null);
    setCandidates([]);
    setSelected(new Set());
    try {
      const data = await mlTrainingAPI.inspectionCandidates(projectId, {
        recipe_id: recipeId || undefined,
        date_from: dateFrom ? `${dateFrom}T00:00:00` : undefined,
        date_to:   dateTo   ? `${dateTo}T23:59:59`   : undefined,
        include_pred_ok: includePredOk,
        include_pred_ng: includePredNg,
        limit,
      });
      setCandidates(data.candidates);
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  // Composite key — annotation_idx alone is not unique within an inspection
  // (a single inspection has multiple frames sharing the same indices).
  const keyOf = (c: InspectionCandidate) =>
    `${c.inspection_id}:${c.camera_serial}:${c.frame_idx}:${c.annotation_idx}`;

  const toggleOne = (k: string, disabled: boolean) => {
    if (disabled) return;
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });
  };

  // Selectable = candidates not already imported
  const selectableKeys = useMemo(
    () => candidates.filter(c => !c.imported_batch_id).map(keyOf),
    [candidates],
  );

  const toggleAll = () => {
    if (selected.size === selectableKeys.length && selectableKeys.length > 0) {
      setSelected(new Set());
    } else {
      setSelected(new Set(selectableKeys));
    }
  };

  const handleImport = async () => {
    if (selected.size === 0) return;
    setImporting(true);
    try {
      const selections = candidates
        .filter(c => selected.has(keyOf(c)))
        .map(c => ({
          inspection_id: c.inspection_id,
          annotation_idx: c.annotation_idx,
          camera_serial: c.camera_serial,
          frame_idx: c.frame_idx,
        }));
      const result = await mlTrainingAPI.createCharImportBatch(
        projectId, selections, batchName.trim() || undefined,
      );
      onImported(result);
      onClose();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Import failed');
    } finally {
      setImporting(false);
    }
  };

  if (!open) return null;

  const totalSelectable = selectableKeys.length;
  const allSelected = totalSelectable > 0 && selected.size === totalSelectable;
  const predOkCount = candidates.filter(c => c.ml_label === 'OK').length;
  const predNgCount = candidates.filter(c => c.ml_label === 'NG').length;
  const dupCount    = candidates.filter(c => c.imported_batch_id).length;

  return (
    <div className="ml-modal-overlay" onClick={onClose}>
      <div className="ml-modal ml-modal-lg" onClick={e => e.stopPropagation()}>
        <div className="ml-modal-header">
          <h3>Import from Inspections</h3>
          <button className="ml-modal-close" onClick={onClose}>×</button>
        </div>

        <div className="ml-modal-body">
          {/* Filters */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div className="ml-form-row">
              <label className="ml-label">Recipe</label>
              <select className="ml-form-select" value={recipeId}
                      onChange={e => setRecipeId(e.target.value)}>
                <option value="">All recipes</option>
                {recipes.map(r => (
                  <option key={r.id} value={r.id}>{r.name}</option>
                ))}
              </select>
              {recipes.length === 0 && (
                <span className="ml-hint">No recipes loaded yet.</span>
              )}
            </div>
            <div className="ml-form-row" style={{ flexDirection: 'row', gap: 8 }}>
              <div style={{ flex: 1 }}>
                <label className="ml-label">From</label>
                <input className="ml-form-input" type="date"
                       value={dateFrom} onChange={e => setDateFrom(e.target.value)} />
              </div>
              <div style={{ flex: 1 }}>
                <label className="ml-label">To</label>
                <input className="ml-form-input" type="date"
                       value={dateTo} onChange={e => setDateTo(e.target.value)} />
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16, flexWrap: 'wrap', marginTop: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 11, opacity: .7 }}>ML Predicted:</span>
              <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                <input type="checkbox" checked={includePredOk}
                       onChange={e => setIncludePredOk(e.target.checked)} />
                <span style={{ color: '#22c55e', fontWeight: 600 }}>OK</span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                <input type="checkbox" checked={includePredNg}
                       onChange={e => setIncludePredNg(e.target.checked)} />
                <span style={{ color: '#ef4444', fontWeight: 600 }}>NG</span>
              </label>
            </div>
            <div className="ml-form-row" style={{ minWidth: 100 }}>
              <label className="ml-label">Limit</label>
              <input className="ml-form-input" type="number" min={1} max={500} step={10}
                     value={limit} onChange={e => setLimit(Number(e.target.value) || 100)} />
            </div>
            <button className="ml-btn ml-btn-primary" onClick={handleSearch} disabled={loading}>
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>

          {errorMsg && (
            <div className="ml-alert ml-alert-error" style={{ marginTop: 8 }}>{errorMsg}</div>
          )}

          {/* Result summary + select all + batch name */}
          {candidates.length > 0 && (
            <>
              <div className="ml-subhint" style={{
                display: 'flex', alignItems: 'center', gap: 12, marginTop: 12, marginBottom: 6, flexWrap: 'wrap',
              }}>
                <span><b>{candidates.length}</b> candidates</span>
                <span style={{ color: '#22c55e' }}>{predOkCount} pred OK</span>
                <span style={{ color: '#ef4444' }}>{predNgCount} pred NG</span>
                {dupCount > 0 && (
                  <span style={{ opacity: .6 }}>· {dupCount} already imported</span>
                )}
                <span style={{ marginLeft: 'auto' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
                    <input type="checkbox" checked={allSelected} onChange={toggleAll}
                           disabled={totalSelectable === 0} />
                    <span>Select all ({selected.size}/{totalSelectable})</span>
                  </label>
                </span>
              </div>

              <div className="ml-form-row" style={{ marginTop: 4 }}>
                <label className="ml-label">Batch name (optional)</label>
                <input className="ml-form-input" type="text" maxLength={80}
                       placeholder="Auto-generated from current time"
                       value={batchName} onChange={e => setBatchName(e.target.value)} />
              </div>
            </>
          )}

          {/* Char crop grid */}
          <div className="ml-insp-grid">
            {candidates.map(c => {
              const k = keyOf(c);
              const dup = !!c.imported_batch_id;
              const isSel = selected.has(k);
              return (
                <button
                  key={k}
                  type="button"
                  className={`ml-insp-card ${c.ml_label === 'OK' ? 'pred-ok' : 'pred-ng'} ${isSel ? 'selected' : ''} ${dup ? 'duplicate' : ''}`}
                  onClick={() => toggleOne(k, dup)}
                  disabled={dup}
                  style={dup ? { opacity: 0.4, cursor: 'not-allowed' } : undefined}
                  title={
                    dup
                      ? `Already imported in batch "${c.imported_batch_name}"`
                      : `${c.recipe_name} · ${c.camera_serial} · frame ${c.frame_idx} · ${c.timestamp ?? ''}`
                  }
                >
                  <img
                    src={`data:image/jpeg;base64,${c.crop_b64}`}
                    alt={c.expected || '?'}
                    className="ml-insp-card-img"
                  />
                  <div className="ml-insp-card-meta">
                    <span className="ml-insp-card-char">{c.expected || '?'}</span>
                    <span className={`ml-insp-card-label ${c.ml_label === 'OK' ? 'ok' : 'ng'}`}>
                      {c.ml_label}
                    </span>
                    <span className="ml-insp-card-prob">{(c.ml_p_ok * 100).toFixed(0)}%</span>
                  </div>
                  {dup && (
                    <span style={{
                      position: 'absolute', top: 2, left: 2,
                      fontSize: 9, fontWeight: 600, padding: '1px 5px',
                      borderRadius: 3, background: 'rgba(75,85,99,.9)', color: '#fff',
                    }}>imported</span>
                  )}
                </button>
              );
            })}
          </div>

          {!loading && candidates.length === 0 && (
            <div className="ml-empty" style={{ padding: '40px 0', textAlign: 'center', opacity: .55 }}>
              No candidates. Try widening the date range or toggle filters.
            </div>
          )}
        </div>

        <div className="ml-modal-footer">
          <button className="ml-btn ml-btn-secondary" onClick={onClose} disabled={importing}>Cancel</button>
          <button className="ml-btn ml-btn-primary" onClick={handleImport}
                  disabled={importing || selected.size === 0}>
            {importing ? 'Importing...' : `Import ${selected.size} char(s)`}
          </button>
        </div>
      </div>
    </div>
  );
}
