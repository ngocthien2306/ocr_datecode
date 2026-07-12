import { useEffect, useMemo, useState } from 'react';
import '@/styles/AnomalyTraining.css';
import { anomalyTrainingAPI, AnomalyCandidate, AnomalyImportResult } from '@/services/anomalyTraining';
import { recipesAPI } from '@/services/recipes';
import type { Recipe } from '@/types/index';

interface Props {
  projectId: string;
  open: boolean;
  onClose: () => void;
  onImported: (result: AnomalyImportResult) => void;
}

interface Selection {
  label: 'normal' | 'abnormal';
  defect_type: string;
}

const DEFAULT_DEFECT_TYPE = 'wrinkled';

/**
 * Search past inspection results for whole-label crops (the same `label`
 * region wrinkle_segmenter uses today), filtered by recipe, and import the
 * chosen ones into this anomaly project's dataset as normal/abnormal.
 */
export default function ImportFromRecipeModal({ projectId, open, onClose, onImported }: Props) {
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
  const [limit, setLimit] = useState(100);

  const [candidates, setCandidates] = useState<AnomalyCandidate[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [selections, setSelections] = useState<Map<string, Selection>>(new Map());
  const [importing, setImporting] = useState(false);

  useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const list = await recipesAPI.getAllRecipes(0, 100);
        setRecipes(list);
      } catch (e: any) {
        setErrorMsg(e?.response?.data?.detail || e?.message || 'Failed to load recipes');
      }
    })();
  }, [open]);

  useEffect(() => {
    if (!open) {
      setCandidates([]);
      setSelections(new Map());
      setErrorMsg(null);
    }
  }, [open]);

  const keyOf = (c: AnomalyCandidate) => `${c.inspection_id}:${c.camera_serial}:${c.frame_idx}`;

  const handleSearch = async () => {
    setLoading(true);
    setErrorMsg(null);
    setCandidates([]);
    setSelections(new Map());
    try {
      const data = await anomalyTrainingAPI.getCandidates({
        project_id: projectId,
        recipe_id: recipeId || undefined,
        date_from: dateFrom ? `${dateFrom}T00:00:00` : undefined,
        date_to: dateTo ? `${dateTo}T23:59:59` : undefined,
        limit,
      });
      setCandidates(data.candidates);
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const setLabel = (key: string, label: 'normal' | 'abnormal') => {
    setSelections((prev) => {
      const next = new Map(prev);
      const existing = next.get(key);
      if (existing?.label === label) {
        next.delete(key); // toggle off
      } else {
        next.set(key, { label, defect_type: existing?.defect_type || DEFAULT_DEFECT_TYPE });
      }
      return next;
    });
  };

  const setDefectType = (key: string, defect_type: string) => {
    setSelections((prev) => {
      const next = new Map(prev);
      const existing = next.get(key);
      if (existing) next.set(key, { ...existing, defect_type });
      return next;
    });
  };

  const handleImport = async () => {
    if (selections.size === 0) return;
    setImporting(true);
    setErrorMsg(null);
    try {
      const payload = Array.from(selections.entries()).map(([key, sel]) => {
        const [inspection_id, camera_serial, frame_idx] = key.split(':');
        return {
          inspection_id,
          camera_serial,
          frame_idx: Number(frame_idx),
          label: sel.label,
          defect_type: sel.label === 'abnormal' ? (sel.defect_type || DEFAULT_DEFECT_TYPE) : undefined,
        };
      });
      const result = await anomalyTrainingAPI.importCandidates(projectId, payload);
      onImported(result);
      onClose();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Import failed');
    } finally {
      setImporting(false);
    }
  };

  if (!open) return null;

  const normalN = Array.from(selections.values()).filter((s) => s.label === 'normal').length;
  const abnormalN = Array.from(selections.values()).filter((s) => s.label === 'abnormal').length;

  return (
    <div className="at-modal-overlay" onClick={onClose}>
      <div className="at-modal" onClick={(e) => e.stopPropagation()}>
        <div className="at-modal-header">
          <h3>Import from Recipe</h3>
          <button className="at-modal-close" onClick={onClose}>×</button>
        </div>

        <div className="at-modal-body">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
            <div className="at-form-row">
              <label className="at-label">Recipe</label>
              <select className="at-form-select" value={recipeId} onChange={(e) => setRecipeId(e.target.value)}>
                <option value="">All recipes</option>
                {recipes.map((r) => (
                  <option key={r.id} value={r.id}>{r.name}</option>
                ))}
              </select>
            </div>
            <div className="at-form-row">
              <label className="at-label">From</label>
              <input className="at-form-input" type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
            </div>
            <div className="at-form-row">
              <label className="at-label">To</label>
              <input className="at-form-input" type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16, marginTop: 10 }}>
            <div className="at-form-row" style={{ width: 100 }}>
              <label className="at-label">Limit</label>
              <input className="at-form-input" type="number" min={1} max={500} step={10}
                     value={limit} onChange={(e) => setLimit(Number(e.target.value) || 100)} />
            </div>
            <button className="at-btn at-btn-primary" onClick={handleSearch} disabled={loading}>
              {loading ? 'Searching...' : 'Search'}
            </button>
            {candidates.length > 0 && (
              <span className="at-hint" style={{ marginLeft: 'auto' }}>
                <b>{candidates.length}</b> candidates ·
                <span style={{ color: '#15803d' }}> {normalN} normal</span> ·
                <span style={{ color: '#b91c1c' }}> {abnormalN} abnormal</span> selected
              </span>
            )}
          </div>

          {errorMsg && <div className="at-alert-error" style={{ marginTop: 8 }}>{errorMsg}</div>}

          <div className="at-cand-grid">
            {candidates.map((c) => {
              const key = keyOf(c);
              const dup = !!c.imported_split;
              const sel = selections.get(key);
              return (
                <div
                  key={key}
                  className={`at-cand-card ${sel ? `selected-${sel.label}` : ''} ${dup ? 'duplicate' : ''}`}
                  title={`${c.recipe_name} · ${c.camera_serial} · frame ${c.frame_idx} · ${c.timestamp ?? ''}`}
                >
                  <img src={`data:image/jpeg;base64,${c.crop_b64}`} alt="" className="at-cand-card-img" />
                  {dup && <span className="at-cand-badge">imported ({c.imported_split})</span>}
                  <div className="at-cand-card-meta">{c.camera_serial} · f{c.frame_idx}</div>
                  <div className="at-cand-card-actions">
                    <button
                      type="button"
                      className={`at-cand-toggle normal ${sel?.label === 'normal' ? 'active' : ''}`}
                      disabled={dup}
                      onClick={() => setLabel(key, 'normal')}
                    >Normal</button>
                    <button
                      type="button"
                      className={`at-cand-toggle abnormal ${sel?.label === 'abnormal' ? 'active' : ''}`}
                      disabled={dup}
                      onClick={() => setLabel(key, 'abnormal')}
                    >Abnormal</button>
                  </div>
                  {sel?.label === 'abnormal' && (
                    <input
                      className="at-cand-defect-input"
                      value={sel.defect_type}
                      placeholder="defect type (e.g. wrinkled)"
                      onChange={(e) => setDefectType(key, e.target.value)}
                    />
                  )}
                </div>
              );
            })}
          </div>

          {!loading && candidates.length === 0 && (
            <div className="at-empty-state" style={{ padding: '40px 0' }}>
              No candidates yet — pick a recipe/date range and Search.
            </div>
          )}
        </div>

        <div className="at-modal-footer">
          <button className="at-btn at-btn-secondary" onClick={onClose} disabled={importing}>Cancel</button>
          <button className="at-btn at-btn-primary" onClick={handleImport} disabled={importing || selections.size === 0}>
            {importing ? 'Importing...' : `Import ${selections.size} image(s)`}
          </button>
        </div>
      </div>
    </div>
  );
}
