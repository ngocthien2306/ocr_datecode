import { useEffect, useMemo, useState } from 'react';
import { mlTrainingAPI, InspectionCandidate } from '@/services/mlTraining';
import { recipesAPI } from '@/services/recipes';
import type { Recipe } from '@/types/index';

interface Props {
  projectId: string;
  open: boolean;
  onClose: () => void;
  onImported: (result: { imported: number; skipped: number; errors: any[] }) => void;
}

/**
 * Pull mispredicted chars from past inspections back into the project for
 * re-labeling. Operator can filter by recipe + date, toggle hard-fail vs.
 * borderline, multi-select chars to import. Each import copies the source
 * frame into the project and creates an annotation with the char bbox +
 * char_id pre-filled (label left blank for human review).
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
  const [includeHardFail, setIncludeHardFail] = useState(true);
  const [includeBorderline, setIncludeBorderline] = useState(true);
  const [limit, setLimit] = useState(100);

  const [candidates, setCandidates] = useState<InspectionCandidate[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [importing, setImporting] = useState(false);

  useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const list = await recipesAPI.getAllRecipes(0, 200);
        setRecipes(list);
      } catch { /* ignore */ }
    })();
  }, [open]);

  const handleSearch = async () => {
    if (!includeHardFail && !includeBorderline) {
      setErrorMsg('Pick at least one of Hard Fail or Borderline');
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
        include_hard_fail: includeHardFail,
        include_borderline: includeBorderline,
        limit,
      });
      setCandidates(data.candidates);
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const keyOf = (c: InspectionCandidate) => `${c.inspection_id}:${c.annotation_idx}`;

  const toggleOne = (k: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === candidates.length) setSelected(new Set());
    else setSelected(new Set(candidates.map(keyOf)));
  };

  const handleImport = async () => {
    if (selected.size === 0) return;
    setImporting(true);
    try {
      const selections = candidates
        .filter(c => selected.has(keyOf(c)))
        .map(c => ({ inspection_id: c.inspection_id, annotation_idx: c.annotation_idx }));
      const result = await mlTrainingAPI.importFromInspections(projectId, selections);
      onImported(result);
      onClose();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Import failed');
    } finally {
      setImporting(false);
    }
  };

  if (!open) return null;

  const allSelected = candidates.length > 0 && selected.size === candidates.length;
  const hardFailCount  = candidates.filter(c => c.kind === 'hard_fail').length;
  const borderlineCount = candidates.filter(c => c.kind === 'borderline').length;

  return (
    <div className="ml-modal-overlay" onClick={onClose}>
      <div className="ml-modal ml-modal-lg" onClick={e => e.stopPropagation()}>
        <div className="ml-modal-header">
          <h3>Import from Inspections</h3>
          <button className="ml-modal-close" onClick={onClose}>×</button>
        </div>

        <div className="ml-modal-body">
          {/* Filters */}
          <div className="ml-form-row" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div className="ml-form-group">
              <label className="ml-form-label">Recipe</label>
              <select className="ml-form-select" value={recipeId}
                      onChange={e => setRecipeId(e.target.value)}>
                <option value="">All recipes</option>
                {recipes.map(r => (
                  <option key={r.id} value={r.id}>{r.name}</option>
                ))}
              </select>
            </div>
            <div className="ml-form-group" style={{ display: 'flex', gap: 8 }}>
              <div style={{ flex: 1 }}>
                <label className="ml-form-label">From</label>
                <input className="ml-form-input" type="date"
                       value={dateFrom} onChange={e => setDateFrom(e.target.value)} />
              </div>
              <div style={{ flex: 1 }}>
                <label className="ml-form-label">To</label>
                <input className="ml-form-input" type="date"
                       value={dateTo} onChange={e => setDateTo(e.target.value)} />
              </div>
            </div>
          </div>

          <div className="ml-form-row" style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input type="checkbox" checked={includeHardFail}
                     onChange={e => setIncludeHardFail(e.target.checked)} />
              <span>Hard Fail <span style={{ opacity: .6, fontSize: 11 }}>(ML said NG)</span></span>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input type="checkbox" checked={includeBorderline}
                     onChange={e => setIncludeBorderline(e.target.checked)} />
              <span>Borderline <span style={{ opacity: .6, fontSize: 11 }}>(0.03 ≤ p_ok ≤ 0.3)</span></span>
            </label>
            <div className="ml-form-group" style={{ marginLeft: 'auto', minWidth: 120 }}>
              <label className="ml-form-label">Limit</label>
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

          {/* Result summary + select all */}
          {candidates.length > 0 && (
            <div className="ml-subhint" style={{
              display: 'flex', alignItems: 'center', gap: 12, marginTop: 12, marginBottom: 6,
            }}>
              <span><b>{candidates.length}</b> candidates</span>
              <span style={{ color: '#ef4444' }}>{hardFailCount} hard-fail</span>
              <span style={{ color: '#fbbf24' }}>{borderlineCount} borderline</span>
              <span style={{ marginLeft: 'auto' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
                  <input type="checkbox" checked={allSelected} onChange={toggleAll} />
                  <span>Select all ({selected.size})</span>
                </label>
              </span>
            </div>
          )}

          {/* Char crop grid */}
          <div className="ml-insp-grid">
            {candidates.map(c => {
              const k = keyOf(c);
              const isSel = selected.has(k);
              return (
                <button
                  key={k}
                  type="button"
                  className={`ml-insp-card ${c.kind} ${isSel ? 'selected' : ''}`}
                  onClick={() => toggleOne(k)}
                  title={`${c.recipe_name} · ${c.camera_serial} · frame ${c.frame_idx} · ${c.timestamp ?? ''}`}
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
