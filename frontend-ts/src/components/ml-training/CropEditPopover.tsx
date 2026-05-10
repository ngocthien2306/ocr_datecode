import { useEffect, useRef, useState } from 'react';
import { mlTrainingAPI } from '@/services/mlTraining';
import { useToast } from '@/contexts/ToastContext';
import ConfirmDialog from '@/components/shared/ConfirmDialog';

/**
 * Inline editor for a single labeled or imported crop. Used by the Train tab
 * (Real Data + Imported sub-tabs) and the Test Set "wrong" cards.
 *
 * Two sources, one popover:
 *   - 'labeled'  → PATCH /annotations/{filename}/segments/{segment_id}
 *   - 'imported' → PATCH /char-imports/chars/{char_id}
 *
 * "Open in source" jumps the parent tab so the user can see full context
 * (filename canvas for labeled, batch view for imported).
 */
export type EditTarget =
  | { source: 'labeled';  projectId: string; filename: string; segmentId: string;
      initialCharId: string | null; initialLabel: 'OK' | 'NG' | null;
      previewSrc: string | null }
  | { source: 'imported'; projectId: string; charId: string; batchId: string;
      initialCharId: string | null; initialLabel: 'OK' | 'NG' | null;
      previewSrc: string | null };

export type DeepLink =
  | { tab: 'label';   filename: string; segmentId: string }
  | { tab: 'imports'; batchId: string;  charId: string };

interface Props {
  target: EditTarget | null;
  onClose: () => void;
  onSaved: () => void;
  onJumpTo?: (link: DeepLink) => void;
}

export default function CropEditPopover({ target, onClose, onSaved, onJumpTo }: Props) {
  const toast = useToast();
  const [charId, setCharId] = useState('');
  const [label, setLabel] = useState<'OK' | 'NG' | null>(null);
  const [saving, setSaving] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  // Track mount + the active target identity. Used to ignore late settles from
  // saves whose target was already replaced (or whose popover was unmounted).
  const mountedRef = useRef(true);
  const targetRef  = useRef<EditTarget | null>(target);
  useEffect(() => { targetRef.current = target; }, [target]);
  // Reset on every mount — required because React Strict Mode in dev runs
  // mount → unmount → mount, which would otherwise leave mountedRef stuck at
  // false after the first cleanup and break every guarded async settle.
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // Reset internal state every time the editor target changes.
  useEffect(() => {
    if (!target) return;
    setCharId(target.initialCharId ?? '');
    setLabel(target.initialLabel ?? null);
    setTimeout(() => inputRef.current?.focus(), 30);
  }, [target]);

  if (!target) return null;

  const dirty =
    (charId.trim() || null) !== (target.initialCharId || null) ||
    label !== target.initialLabel;

  // True if the in-flight call should still affect UI (popover not unmounted
  // and target not swapped to another crop). Without this, a slow save on
  // crop A finishing AFTER user opens popover for crop B would close B.
  const isStillActive = (myTarget: EditTarget) =>
    mountedRef.current && targetRef.current === myTarget;

  const save = async () => {
    if (saving || !dirty) { onClose(); return; }
    const myTarget = target;
    setSaving(true);

    // Build patch with `undefined` for unchanged fields → axios omits them
    // from the JSON body. Sending `null` worked too (BE Pydantic treats both
    // as "no change") but showed up as noise in DevTools.
    const initialCid = (myTarget.initialCharId ?? '').trim();
    const newCid = charId.trim();
    const charIdChanged = newCid !== initialCid;
    const labelChanged  = label !== null && label !== myTarget.initialLabel;

    // Both BE endpoints share the same convention: empty string after strip
    // → stored as None (clear). So we send the trimmed string for both paths.
    try {
      if (myTarget.source === 'labeled') {
        await mlTrainingAPI.patchSegment(myTarget.projectId, myTarget.filename, myTarget.segmentId, {
          ...(charIdChanged ? { char_id: newCid } : {}),
          ...(labelChanged  ? { label: label as 'OK' | 'NG' } : {}),
        });
      } else {
        await mlTrainingAPI.updateCharImport(myTarget.projectId, myTarget.charId, {
          ...(charIdChanged ? { char_id: newCid } : {}),
          ...(labelChanged  ? { label: label as 'OK' | 'NG' } : {}),
        });
      }
      // onSaved (refresh upstream) runs unconditionally — refresh is harmless.
      // onClose only fires if the user is still looking at this same target.
      onSaved();
      if (isStillActive(myTarget)) {
        toast.success('Saved');
        onClose();
      }
    } catch (e: any) {
      if (isStillActive(myTarget)) {
        toast.error(e?.response?.data?.detail ?? 'Save failed');
      }
    } finally {
      if (mountedRef.current) setSaving(false);
    }
  };

  const doDelete = async () => {
    const myTarget = target;
    setSaving(true);
    try {
      if (myTarget.source === 'labeled') {
        await mlTrainingAPI.deleteSegment(myTarget.projectId, myTarget.filename, myTarget.segmentId);
      } else {
        await mlTrainingAPI.deleteCharImport(myTarget.projectId, myTarget.charId);
      }
      onSaved();
      if (isStillActive(myTarget)) {
        toast.success('Deleted');
        onClose();
      }
    } catch (e: any) {
      if (isStillActive(myTarget)) {
        toast.error(e?.response?.data?.detail ?? 'Delete failed');
      }
    } finally {
      if (mountedRef.current) {
        setSaving(false);
        setConfirmOpen(false);
      }
    }
  };

  const jumpToSource = () => {
    if (!onJumpTo) return;
    if (target.source === 'labeled') {
      onJumpTo({ tab: 'label', filename: target.filename, segmentId: target.segmentId });
    } else {
      onJumpTo({ tab: 'imports', batchId: target.batchId, charId: target.charId });
    }
    onClose();
  };

  const sourceLabel = target.source === 'labeled' ? 'Label tab' : 'Imported Chars tab';

  return (
    <>
      <div className="ml-modal-overlay" onClick={onClose}>
        <div className="ml-modal ml-crop-edit-modal" onClick={e => e.stopPropagation()}>
          <div className="ml-modal-header">
            <h3>Fix this crop</h3>
            <button className="ml-modal-close" onClick={onClose} aria-label="Close">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </button>
          </div>

          <div className="ml-modal-body ml-crop-edit-body">
            {target.previewSrc && (
              <div className="ml-crop-edit-preview">
                <img src={target.previewSrc} alt={target.initialCharId ?? '?'} />
                <span className="ml-crop-edit-source-tag">{target.source}</span>
              </div>
            )}

            <div className="ml-form-group" style={{ marginTop: 0 }}>
              <label className="ml-form-label">char_id</label>
              <input ref={inputRef}
                className="ml-form-input"
                value={charId}
                onChange={e => setCharId(e.target.value)}
                placeholder="e.g. 0, A, B"
                onKeyDown={e => {
                  if (e.key === 'Enter') save();
                  else if (e.key === 'Escape') onClose();
                }}
              />
            </div>

            <div className="ml-form-group">
              <label className="ml-form-label">Label</label>
              <div className="ml-crop-edit-label-toggle" role="group" aria-label="Pick label">
                <button type="button"
                  className={`ok ${label === 'OK' ? 'active' : ''}`}
                  onClick={() => setLabel('OK')}>OK</button>
                <button type="button"
                  className={`ng ${label === 'NG' ? 'active' : ''}`}
                  onClick={() => setLabel('NG')}>NG</button>
              </div>
            </div>
          </div>

          <div className="ml-modal-footer ml-crop-edit-footer">
            <button className="ml-btn ml-btn-danger ml-btn-sm"
              onClick={() => setConfirmOpen(true)} disabled={saving}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" style={{ marginRight: 4 }}>
                <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2M10 11v6M14 11v6"
                  stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Delete
            </button>
            {onJumpTo && (
              <button className="ml-btn ml-btn-secondary ml-btn-sm"
                onClick={jumpToSource} disabled={saving}
                title={`Open in ${sourceLabel}`}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" style={{ marginRight: 4 }}>
                  <path d="M14 3h7v7M10 14L21 3M21 14v5a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h5"
                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Open in {sourceLabel}
              </button>
            )}
            <div style={{ flex: 1 }} />
            <button className="ml-btn ml-btn-secondary" onClick={onClose} disabled={saving}>
              Cancel
            </button>
            <button className="ml-btn ml-btn-primary" onClick={save} disabled={saving || !dirty}>
              {saving ? 'Saving…' : 'Save'}
            </button>
          </div>
        </div>
      </div>

      <ConfirmDialog
        isOpen={confirmOpen}
        title="Delete this crop?"
        message={
          target.source === 'labeled'
            ? 'This removes the segment from the annotation. Cannot be undone.'
            : 'This removes the imported char from the pool. Cannot be undone.'
        }
        type="danger"
        confirmText="Delete"
        onClose={() => setConfirmOpen(false)}
        onConfirm={doDelete}
      />
    </>
  );
}
