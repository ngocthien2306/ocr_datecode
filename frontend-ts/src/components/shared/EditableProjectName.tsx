import { useState } from 'react';

interface Props {
  name: string;
  /** Persist the new name; throw to keep the editor open and show the error. */
  onSave: (name: string) => Promise<void>;
}

/** Project name in a training-studio sidebar with in-place rename (✎).
 *  Enter saves, Esc cancels. Clicks inside never select the project row. */
export default function EditableProjectName({ name, onSave }: Props) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(name);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = (e: React.MouseEvent) => {
    e.stopPropagation();
    setValue(name);
    setError(null);
    setEditing(true);
  };

  const save = async () => {
    const next = value.trim();
    if (!next) { setError('Name cannot be empty'); return; }
    if (next === name) { setEditing(false); return; }
    setSaving(true);
    setError(null);
    try {
      await onSave(next);
      setEditing(false);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Rename failed');
    } finally {
      setSaving(false);
    }
  };

  if (!editing) {
    return (
      <div className="at-project-name-row">
        <div className="at-project-item-name" title={name}>{name}</div>
        <button type="button" className="at-project-rename-btn" onClick={start} title="Rename project">✎</button>
      </div>
    );
  }

  return (
    <div className="at-project-rename" onClick={(e) => e.stopPropagation()}>
      <input
        className="at-form-input"
        value={value}
        autoFocus
        disabled={saving}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') { e.preventDefault(); save(); }
          else if (e.key === 'Escape') { e.preventDefault(); setEditing(false); }
        }}
      />
      <div className="at-project-rename-actions">
        <button type="button" className="at-btn at-btn-primary at-btn-sm" onClick={save} disabled={saving}>
          {saving ? '…' : 'Save'}
        </button>
        <button type="button" className="at-btn at-btn-secondary at-btn-sm"
                onClick={() => setEditing(false)} disabled={saving}>Cancel</button>
      </div>
      {error && <div className="at-project-rename-error">{error}</div>}
    </div>
  );
}
