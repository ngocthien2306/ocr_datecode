import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ocrTrainingAPI, MatchFilter, OCRCandidate, OCRCandidatesResponse,
  OCRImportResult, OCRRecipeOption, RegionType, SplitName,
} from '@/services/ocrTraining';

interface Props {
  projectId: string;
  open: boolean;
  onClose: () => void;
  onImported: (result: OCRImportResult) => void;
}

const MATCH_FILTERS: Array<{ id: MatchFilter; label: string; hint: string }> = [
  { id: 'pass', label: 'Verified OK', hint: 'The recipe confirmed this text — expected IS the ground truth' },
  { id: 'fail', label: 'Failed', hint: 'Hard crops and real misprints. Most valuable, most in need of review' },
  { id: 'all', label: 'All', hint: 'Both' },
];

function candKey(c: OCRCandidate): string {
  return `${c.inspection_id}:${c.camera_serial}:${c.frame_idx}:${c.annotation_index}`;
}

export default function ImportFromRecipeModal({ projectId, open, onClose, onImported }: Props) {
  const [recipes, setRecipes] = useState<OCRRecipeOption[]>([]);
  const [recipeId, setRecipeId] = useState('');
  const [matchFilter, setMatchFilter] = useState<MatchFilter>('pass');
  const [regionType, setRegionType] = useState<RegionType | ''>('');
  const [limit, setLimit] = useState(60);
  const [split, setSplit] = useState<SplitName>('train');

  const [resp, setResp] = useState<OCRCandidatesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  // Text the operator corrected right here in the grid, so an obviously-wrong
  // prefill can be fixed before it ever reaches the Label tab.
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [importing, setImporting] = useState(false);

  useEffect(() => {
    if (!open) return;
    ocrTrainingAPI.listRecipesWithOcrData(50)
      .then((r) => setRecipes(r.recipes))
      .catch((e) => console.error('[OCRTraining] recipe list failed', e));
  }, [open]);

  const search = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await ocrTrainingAPI.getCandidates({
        project_id: projectId,
        recipe_id: recipeId || undefined,
        region_type: regionType || undefined,
        match_filter: matchFilter,
        limit,
      });
      setResp(r);
      // Pre-tick everything not already imported: on the `pass` filter the
      // prefill is the recipe's own expected text, so bulk-accepting is the
      // normal action and unticking the odd one out is the exception.
      setSelected(new Set(r.candidates.filter((c) => !c.imported_status).map(candKey)));
      setEdits({});
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  }, [projectId, recipeId, regionType, matchFilter, limit]);

  useEffect(() => { if (open) search(); }, [open]);

  const toggle = (key: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  const selectable = useMemo(
    () => (resp?.candidates ?? []).filter((c) => !c.imported_status),
    [resp],
  );

  const doImport = async () => {
    if (!resp || selected.size === 0) return;
    setImporting(true);
    setError(null);
    try {
      const selections = resp.candidates
        .filter((c) => selected.has(candKey(c)))
        .map((c) => ({
          inspection_id: c.inspection_id,
          camera_serial: c.camera_serial,
          frame_idx: c.frame_idx,
          annotation_index: c.annotation_index,
          gt_text: edits[candKey(c)] ?? c.prefill_text,
        }));
      const result = await ocrTrainingAPI.importCandidates(projectId, selections, split);
      onImported(result);
      await search();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Import failed');
    } finally {
      setImporting(false);
    }
  };

  if (!open) return null;

  return (
    <div className="at-modal-overlay" onClick={onClose}>
      <div className="at-modal" style={{ maxWidth: 1100 }} onClick={(e) => e.stopPropagation()}>
        <div className="at-modal-header">
          <span>Import OCR crops from recipe</span>
          <button className="at-modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="at-modal-body">
          <div className="ot-toolbar">
            <select className="at-form-input" style={{ maxWidth: 260 }}
                    value={recipeId} onChange={(e) => setRecipeId(e.target.value)}>
              <option value="">All recipes with OCR data</option>
              {recipes.map((r) => (
                <option key={r.recipe_id} value={r.recipe_id}>
                  {r.recipe_name || r.recipe_id} ({r.inspection_count})
                </option>
              ))}
            </select>

            <select className="at-form-input" style={{ maxWidth: 130 }}
                    value={regionType} onChange={(e) => setRegionType(e.target.value as RegionType | '')}>
              <option value="">text + datecode</option>
              <option value="text">text</option>
              <option value="datecode">datecode</option>
            </select>

            <div style={{ display: 'flex', gap: 4 }}>
              {MATCH_FILTERS.map((f) => (
                <button key={f.id} title={f.hint}
                        className={`at-btn at-btn-sm ${matchFilter === f.id ? 'at-btn-primary' : 'at-btn-secondary'}`}
                        onClick={() => setMatchFilter(f.id)}>
                  {f.label}
                </button>
              ))}
            </div>

            <input className="at-form-input" style={{ maxWidth: 80 }} type="number" min={1} max={400}
                   value={limit} onChange={(e) => setLimit(Math.max(1, Math.min(400, +e.target.value || 60)))} />

            <button className="at-btn at-btn-primary at-btn-sm" onClick={search} disabled={loading}>
              {loading ? 'Searching…' : 'Search'}
            </button>
          </div>

          {error && <div className="at-alert-error">{error}</div>}

          {resp && (
            <div className="at-hint" style={{ marginBottom: 8 }}>
              {resp.count} candidate(s) from {resp.docs_scanned} inspection(s).
              {resp.skipped_degenerate > 0 && (
                <>
                  {' '}<b>{resp.skipped_degenerate} region(s) skipped</b> — their annotation quad
                  was projected outside the frame, which means template alignment failed on those
                  frames rather than that the crops are missing.
                </>
              )}
              {matchFilter === 'fail' && (
                <> Prefill on failed regions comes from what OCR read, so expect to correct these.</>
              )}
            </div>
          )}

          <div style={{ maxHeight: '48vh', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: 6 }}>
            {(resp?.candidates ?? []).map((c) => {
              const key = candKey(c);
              const imported = !!c.imported_status;
              return (
                <div key={key} className={`ot-cand-row ${imported ? 'is-imported' : ''}`}>
                  <input type="checkbox" checked={selected.has(key)} disabled={imported}
                         onChange={() => toggle(key)} />
                  <div className="ot-crop-wrap">
                    <img className="ot-crop-img" src={`data:image/jpeg;base64,${c.crop_b64}`}
                         alt={c.prefill_text || 'crop'} />
                  </div>
                  <div className="ot-label-fields">
                    <input className="ot-gt-input" disabled={imported}
                           value={edits[key] ?? c.prefill_text}
                           placeholder="(no prefill — type the printed text)"
                           onChange={(e) => setEdits((p) => ({ ...p, [key]: e.target.value }))} />
                    <div className="ot-hint-row">
                      {c.expected_text != null && <span className="ot-hint-exp">exp <b>{c.expected_text}</b></span>}
                      {c.recognized_text != null && <span className="ot-hint-rec">read <b>{c.recognized_text}</b></span>}
                    </div>
                    <div className="ot-meta-chips">
                      <span className="ot-chip-sm">{c.region_type}</span>
                      <span className="ot-chip-sm">ann{c.annotation_index}</span>
                      {c.verify_match === true && <span className="ot-chip-sm ot-pass">match</span>}
                      {c.verify_match === false && <span className="ot-chip-sm ot-fail">no match</span>}
                      {imported && <span className="ot-chip-sm">already imported ({c.imported_status})</span>}
                    </div>
                  </div>
                </div>
              );
            })}
            {!loading && resp && resp.count === 0 && (
              <div className="at-empty-state" style={{ padding: 30 }}>
                No candidates. Try a different recipe or the “All” filter.
              </div>
            )}
          </div>
        </div>

        <div className="at-modal-footer">
          <span className="at-hint" style={{ marginRight: 'auto' }}>
            {selected.size} of {selectable.length} selectable ticked. Imports land as{' '}
            <b>need_review</b> — training only reads verified labels.
          </span>
          <label className="at-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            split
            <select className="at-form-input" style={{ width: 90 }} value={split}
                    onChange={(e) => setSplit(e.target.value as SplitName)}>
              <option value="train">train</option>
              <option value="test">test</option>
            </select>
          </label>
          <button className="at-btn at-btn-secondary" onClick={onClose}>Close</button>
          <button className="at-btn at-btn-primary" onClick={doImport}
                  disabled={importing || selected.size === 0}>
            {importing ? 'Importing…' : `Import ${selected.size}`}
          </button>
        </div>
      </div>
    </div>
  );
}
