import { useEffect, useState } from 'react';
import {
  mlTrainingAPI,
  ProjectImage,
  ImportFromRecipeResponse,
} from '@/services/mlTraining';
import { recipesAPI } from '@/services/recipes';
import type { Recipe } from '@/types/index';

interface Props {
  projectId: string;
  projectImages: ProjectImage[];
  open: boolean;
  onClose: () => void;
  onImported: (result: ImportFromRecipeResponse) => void;
}

/**
 * Modal to auto-populate ML project annotations from a recipe template.
 *
 * Flow:
 *   1. User selects a recipe (loaded from /recipes API)
 *   2. User selects a camera (from recipe.cameras)
 *   3. User multi-selects filenames from the project images
 *   4. Submit → BE reads recipe's template annotations (type=text/datecode),
 *      creates regions/segments per selected file with char_id = expected_text
 */
export default function ImportFromRecipeModal({
  projectId, projectImages, open, onClose, onImported,
}: Props) {
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [loadingRecipes, setLoadingRecipes] = useState(false);

  const [recipeId, setRecipeId] = useState<string>('');
  const [cameraSerial, setCameraSerial] = useState<string>('');
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());

  const [importing, setImporting] = useState(false);
  const [lastResult, setLastResult] = useState<ImportFromRecipeResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Load recipes once when modal opens. BE caps limit at 100.
  useEffect(() => {
    if (!open) return;
    (async () => {
      setLoadingRecipes(true);
      setErrorMsg(null);
      try {
        const list = await recipesAPI.getAllRecipes(0, 100);
        setRecipes(list);
      } catch (e: any) {
        console.error('[ImportModal] load recipes failed:', e);
        setErrorMsg(e?.response?.data?.detail || e?.message || 'Failed to load recipes');
      } finally {
        setLoadingRecipes(false);
      }
    })();
  }, [open]);

  // When recipe changes, auto-select first camera
  useEffect(() => {
    if (!recipeId) { setCameraSerial(''); return; }
    const r = recipes.find(x => x.id === recipeId);
    const firstCam = r?.cameras?.[0];
    if (firstCam) {
      setCameraSerial(firstCam.serial_number);
    } else {
      setCameraSerial('');
    }
  }, [recipeId, recipes]);

  const selectedRecipe = recipes.find(r => r.id === recipeId);

  const toggleFile = (filename: string) => {
    setSelectedFiles(prev => {
      const next = new Set(prev);
      if (next.has(filename)) next.delete(filename);
      else next.add(filename);
      return next;
    });
  };

  const selectAllFiles = () => {
    if (selectedFiles.size === projectImages.length) {
      setSelectedFiles(new Set());
    } else {
      setSelectedFiles(new Set(projectImages.map(p => p.filename)));
    }
  };

  const handleImport = async () => {
    setErrorMsg(null);
    setLastResult(null);
    if (!recipeId || !cameraSerial || selectedFiles.size === 0) {
      setErrorMsg('Vui lòng chọn recipe, camera và ít nhất 1 file');
      return;
    }
    setImporting(true);
    try {
      const result = await mlTrainingAPI.importFromRecipe(projectId, {
        recipe_id: recipeId,
        camera_serial: cameraSerial,
        filenames: Array.from(selectedFiles),
      });
      setLastResult(result);
      onImported(result);
      // Keep modal open so user sees summary; they close manually
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Import failed');
    } finally {
      setImporting(false);
    }
  };

  const handleClose = () => {
    setLastResult(null);
    setErrorMsg(null);
    setSelectedFiles(new Set());
    onClose();
  };

  if (!open) return null;

  return (
    <div className="ml-modal-overlay" onClick={handleClose}>
      <div className="ml-modal" onClick={e => e.stopPropagation()}>
        <div className="ml-modal-header">
          <h3>Import annotations from Recipe</h3>
          <button className="ml-modal-close" onClick={handleClose} title="Close">×</button>
        </div>

        <div className="ml-modal-body">
          <p className="ml-hint" style={{ marginTop: 0 }}>
            Auto-populate segments with <code>char_id</code> from recipe template's
            text/datecode bboxes. User labels OK/NG afterwards.
          </p>

          {/* Recipe picker */}
          <div className="ml-form-row">
            <label className="ml-label">Recipe</label>
            <select
              className="ml-form-select"
              value={recipeId}
              onChange={e => setRecipeId(e.target.value)}
              disabled={loadingRecipes || importing}
            >
              <option value="">-- Select recipe --</option>
              {recipes.map(r => (
                <option key={r.id} value={r.id}>
                  {r.name} ({r.product_code})
                </option>
              ))}
            </select>
            {loadingRecipes && <span className="ml-hint">Loading recipes…</span>}
            {!loadingRecipes && recipes.length === 0 && (
              <span className="ml-hint">No recipes found. Create one in Recipes page first.</span>
            )}
          </div>

          {/* Camera picker */}
          {selectedRecipe && (
            <div className="ml-form-row">
              <label className="ml-label">Camera</label>
              <select
                className="ml-form-select"
                value={cameraSerial}
                onChange={e => setCameraSerial(e.target.value)}
                disabled={importing}
              >
                {(selectedRecipe.cameras || []).map(c => (
                  <option key={c.serial_number} value={c.serial_number}>
                    {c.serial_number} — {c.camera_id || 'unknown'}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* File picker */}
          <div className="ml-form-row">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <label className="ml-label" style={{ flex: 1, marginBottom: 0 }}>
                Files to annotate ({selectedFiles.size}/{projectImages.length})
              </label>
              <button
                className="ml-btn ml-btn-secondary ml-btn-sm"
                onClick={selectAllFiles}
                disabled={importing || projectImages.length === 0}
              >
                {selectedFiles.size === projectImages.length ? 'Unselect all' : 'Select all'}
              </button>
            </div>
            <div className="ml-file-picker">
              {projectImages.length === 0 && (
                <div className="ml-hint" style={{ padding: 6 }}>
                  No project images. Upload/copy images first.
                </div>
              )}
              {projectImages.map(img => {
                const isSel = selectedFiles.has(img.filename);
                return (
                  <label key={img.filename} className={`ml-file-row ${isSel ? 'selected' : ''}`}>
                    <input
                      type="checkbox"
                      checked={isSel}
                      onChange={() => toggleFile(img.filename)}
                      disabled={importing}
                    />
                    <span className="ml-file-row-name">{img.filename}</span>
                    {img.has_annotation && <span className="ml-file-row-tag">labeled</span>}
                  </label>
                );
              })}
            </div>
            <div className="ml-hint">
              ⚠️ Existing annotations will be OVERWRITTEN for selected files.
            </div>
          </div>

          {/* Status / result */}
          {errorMsg && <div className="ml-alert ml-alert-error">{errorMsg}</div>}
          {lastResult && (
            <div className="ml-alert ml-alert-success">
              <div><b>Imported:</b> {lastResult.imported} · <b>Skipped:</b> {lastResult.skipped}</div>
              <div><b>Chars:</b> {lastResult.char_ids.join(', ') || '—'}</div>
              {lastResult.errors.length > 0 && (
                <details style={{ marginTop: 4 }}>
                  <summary>{lastResult.errors.length} error(s)</summary>
                  <ul style={{ margin: '4px 0 0 16px', padding: 0 }}>
                    {lastResult.errors.map((e, i) => (
                      <li key={i} style={{ fontSize: 11 }}>
                        <code>{e.filename}</code>: {e.reason}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          )}
        </div>

        <div className="ml-modal-footer">
          {(!recipeId || !cameraSerial || selectedFiles.size === 0) && !importing && (
            <span className="ml-hint" style={{ marginRight: 'auto', alignSelf: 'center' }}>
              {!recipeId
                ? '→ Pick a recipe first'
                : !cameraSerial
                ? '→ Pick a camera'
                : '→ Select at least 1 file'}
            </span>
          )}
          <button className="ml-btn ml-btn-secondary" onClick={handleClose} disabled={importing}>
            Close
          </button>
          <button
            className="ml-btn ml-btn-primary"
            onClick={handleImport}
            disabled={importing || !recipeId || !cameraSerial || selectedFiles.size === 0}
          >
            {importing ? 'Importing…' : `Import (${selectedFiles.size})`}
          </button>
        </div>
      </div>
    </div>
  );
}
