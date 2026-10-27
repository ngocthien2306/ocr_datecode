import { useEffect, useMemo, useRef, useState } from 'react';
import { FAMILY_LABEL, friendlyArch, friendlyRuntime } from '@/components/ocr-training/modelNames';
import { createPortal } from 'react-dom';
import recipesAPI, { type OcrModelUsage } from '@/services/recipes';
import type { OCRBaseCheckpoints } from '@/services/ocrTraining';
import type { PaddleTrainedModels } from '@/services/paddleOcrTraining';
import '@/styles/OcrModelPicker.css';

/** OCR model picker for the recipe form: a one-line summary plus a
 *  "Choose…" dialog with a searchable, sortable table of every model.
 *
 *  Values use the same encoding as the old <select>: a built-in type name
 *  ('' = default), or CUSTOM:{project}:{model} / CUSTOM_PADDLE:{project}:{model}
 *  for trained models, so the caller's change handler is unchanged. */

type Kind = 'builtin' | 'smtr' | 'paddle';
type KindFilter = 'all' | Kind;
type SortKey = 'project' | 'kind' | 'acc' | 'trained' | 'used';

interface Row {
  value: string;
  modelId: string | null;
  kind: Kind;
  project: string;
  detail: string;
  acc: number | null;
  createdAt: string | null;
  haystack: string;
}

// Operators pick by plain names; engine names (SMTR, Paddle…) are never shown,
// only kept in the search text so engineers can still find a model by them.
const BUILTINS: Array<{ value: string; title: string; detail: string }> = [
  { value: '', title: 'Automatic', detail: 'system default' },
  { value: 'SMTR', title: 'General · Extra large', detail: 'ready to use' },
  { value: 'SVTRV2_CTC', title: 'General · Large', detail: 'ready to use' },
  { value: 'OPENOCR_REPSVTR', title: 'General · Medium', detail: 'ready to use' },
  { value: 'PADDLEV5', title: 'General · Small v5', detail: 'ready to use' },
  { value: 'PADDLEV6', title: 'General · Small v6', detail: 'ready to use' },
];

const KIND_LABEL: Record<Kind, string> = { builtin: 'General', ...FAMILY_LABEL };

const fmtDate = (iso: string | null) => {
  if (!iso) return '';
  const d = new Date(iso);
  return isNaN(d.getTime()) ? '' : d.toLocaleString(undefined, {
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  });
};

const fmtAcc = (acc: number | null) => (acc == null ? '' : `${(acc * 100).toFixed(1)}%`);
const ts = (iso: string | null) => (iso ? Date.parse(iso) || 0 : 0);

function buildRows(smtr: OCRBaseCheckpoints | null, paddle: PaddleTrainedModels | null): Row[] {
  const rows: Row[] = [];
  for (const p of smtr?.projects ?? []) {
    for (const m of p.models) {
      rows.push({
        value: `CUSTOM:${p.project_id}:${m.model_id}`, modelId: m.model_id, kind: 'smtr',
        project: p.project_name,
        detail: m.use_space_char ? 'reads spaces' : '',
        acc: m.min_acc, createdAt: m.created_at,
        haystack: `${p.project_name} custom large smtr`.toLowerCase(),
      });
    }
  }
  for (const p of paddle?.projects ?? []) {
    for (const m of p.models) {
      rows.push({
        value: `CUSTOM_PADDLE:${p.project_id}:${m.model_id}`, modelId: m.model_id, kind: 'paddle',
        project: p.project_name,
        detail: `${friendlyArch(m.arch)} · ${friendlyRuntime(m.runtime)}`,
        acc: m.acc_trt ?? m.acc_onnx, createdAt: m.created_at,
        haystack: `${p.project_name} custom small paddle ${m.arch} ${m.runtime}`.toLowerCase(),
      });
    }
  }
  for (const b of BUILTINS) {
    rows.push({
      value: b.value, modelId: null, kind: 'builtin', project: b.title, detail: b.detail,
      acc: null, createdAt: null, haystack: `${b.title} ${b.value} general built-in`.toLowerCase(),
    });
  }
  return rows;
}

interface Props {
  value: string;
  onChange: (value: string) => void;
  smtrModels: OCRBaseCheckpoints | null;
  paddleModels: PaddleTrainedModels | null;
  /** The recipe being edited, so "Used by" can mark it. */
  currentRecipeId?: string | null;
}

export default function OcrModelPicker({ value, onChange, smtrModels, paddleModels, currentRecipeId }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [kindFilter, setKindFilter] = useState<KindFilter>('all');
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'trained', dir: -1 });
  const [pending, setPending] = useState(value);
  const [usage, setUsage] = useState<OcrModelUsage | null>(null);
  const [usageError, setUsageError] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);

  const rows = useMemo(() => buildRows(smtrModels, paddleModels), [smtrModels, paddleModels]);
  const selected = rows.find(r => r.value === value) ?? null;
  const isMissing = !selected && value !== '';

  const usedBy = (r: Row) => (r.modelId && usage ? usage[r.modelId] ?? [] : []);

  // Fetch usage every time the dialog opens: it changes whenever any recipe is saved.
  useEffect(() => {
    if (!open) return;
    setPending(value);
    setQuery('');
    setUsageError(false);
    let cancelled = false;
    recipesAPI.getOcrModelUsage()
      .then(u => { if (!cancelled) setUsage(u); })
      .catch(() => { if (!cancelled) { setUsage(null); setUsageError(true); } });
    setTimeout(() => searchRef.current?.focus(), 0);
    return () => { cancelled = true; };
  }, [open, value]);

  const visible = useMemo(() => {
    const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const list = rows.filter(r =>
      (kindFilter === 'all' || r.kind === kindFilter)
      && terms.every(t => r.haystack.includes(t)));
    const cmp = (a: Row, b: Row): number => {
      switch (sort.key) {
        case 'project': return a.project.localeCompare(b.project) * sort.dir;
        case 'kind': return KIND_LABEL[a.kind].localeCompare(KIND_LABEL[b.kind]) * sort.dir;
        case 'used': return (usedBy(a).length - usedBy(b).length) * sort.dir;
        case 'acc':
          if (a.acc == null || b.acc == null) return (a.acc == null ? 1 : 0) - (b.acc == null ? 1 : 0);
          return (a.acc - b.acc) * sort.dir;
        case 'trained':
          // Built-ins have no date; keep them at the bottom in both directions.
          if (!a.createdAt || !b.createdAt) return (a.createdAt ? 0 : 1) - (b.createdAt ? 0 : 1);
          return (ts(a.createdAt) - ts(b.createdAt)) * sort.dir;
      }
    };
    return [...list].sort((a, b) => cmp(a, b) || a.project.localeCompare(b.project));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, query, kindFilter, sort, usage]);

  useEffect(() => {
    bodyRef.current?.querySelector<HTMLElement>('tr.is-pending')?.scrollIntoView({ block: 'nearest' });
  }, [pending, open]);

  const toggleSort = (key: SortKey) =>
    setSort(s => (s.key === key ? { key, dir: (s.dir === 1 ? -1 : 1) } : { key, dir: key === 'project' || key === 'kind' ? 1 : -1 }));

  const confirm = (v: string) => {
    onChange(v);
    setOpen(false);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    const i = visible.findIndex(r => r.value === pending);
    if (e.key === 'ArrowDown') { e.preventDefault(); const r = visible[Math.min(visible.length - 1, i + 1)]; if (r) setPending(r.value); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); const r = visible[Math.max(0, i - 1)]; if (r) setPending(r.value); }
    else if (e.key === 'Enter') { e.preventDefault(); if (i >= 0) confirm(pending); }
    else if (e.key === 'Escape') { e.preventDefault(); setOpen(false); }
  };

  const counts = useMemo(() => ({
    all: rows.length,
    smtr: rows.filter(r => r.kind === 'smtr').length,
    paddle: rows.filter(r => r.kind === 'paddle').length,
    builtin: rows.filter(r => r.kind === 'builtin').length,
  }), [rows]);

  const sortHeader = (key: SortKey, label: string, className = '') => (
    <th className={`${className} is-sortable${sort.key === key ? ' is-sorted' : ''}`} onClick={() => toggleSort(key)}>
      {label}
      <span className="ocr-picker__sort">{sort.key === key ? (sort.dir === 1 ? '▲' : '▼') : ''}</span>
    </th>
  );

  return (
    <div className="ocr-picker">
      <button
        type="button"
        className={`ocr-picker__trigger${isMissing ? ' is-missing' : ''}`}
        onClick={() => setOpen(true)}
      >
        {selected ? (
          <span className="ocr-picker__current">
            <span className={`ocr-picker__badge ocr-picker__badge--${selected.kind}`}>{KIND_LABEL[selected.kind]}</span>
            <span className="ocr-picker__current-title">{selected.project}</span>
            {selected.acc != null && <span className="ocr-picker__acc">{fmtAcc(selected.acc)}</span>}
            {selected.createdAt && <span className="ocr-picker__muted">{fmtDate(selected.createdAt)}</span>}
          </span>
        ) : (
          <span className="ocr-picker__current-title">
            {isMissing ? 'Unavailable model — choose another' : 'Automatic'}
          </span>
        )}
        <span className="ocr-picker__choose">Choose…</span>
      </button>

      {open && createPortal(
        <div className="ocr-picker__overlay" onMouseDown={e => { if (e.target === e.currentTarget) setOpen(false); }}>
          <div className="ocr-picker__dialog" role="dialog" aria-label="Choose OCR model" onKeyDown={onKeyDown}>
            <div className="ocr-picker__header">
              <h3>Choose OCR model</h3>
              <button type="button" className="ocr-picker__close" onClick={() => setOpen(false)} aria-label="Close">×</button>
            </div>

            <div className="ocr-picker__toolbar">
              <input
                ref={searchRef}
                type="text"
                className="ocr-picker__search"
                placeholder="Search by project or model name…"
                value={query}
                onChange={e => setQuery(e.target.value)}
              />
              <div className="ocr-picker__filters">
                {(['all', 'smtr', 'paddle', 'builtin'] as const).map(k => (
                  <button
                    key={k}
                    type="button"
                    className={`ocr-picker__chip${kindFilter === k ? ' is-on' : ''}`}
                    onClick={() => setKindFilter(k)}
                  >
                    {k === 'all' ? 'All' : KIND_LABEL[k]} <span>{counts[k]}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="ocr-picker__table-wrap" ref={bodyRef}>
              <table className="ocr-picker__table">
                <thead>
                  <tr>
                    {sortHeader('project', 'Model')}
                    {sortHeader('kind', 'Type')}
                    <th>Details</th>
                    {sortHeader('acc', 'Accuracy', 'num')}
                    {sortHeader('trained', 'Trained')}
                    {sortHeader('used', 'Used by', 'num')}
                  </tr>
                </thead>
                <tbody>
                  {visible.length === 0 && (
                    <tr><td colSpan={6} className="ocr-picker__empty">No model matches “{query}”</td></tr>
                  )}
                  {visible.map(r => {
                    const users = usedBy(r);
                    const isCurrent = r.value === value;
                    const usedHere = !!currentRecipeId && users.some(u => u.id === currentRecipeId);
                    return (
                      <tr
                        key={r.value || '__default'}
                        className={`${r.value === pending ? 'is-pending' : ''}${isCurrent ? ' is-current' : ''}`}
                        onClick={() => setPending(r.value)}
                        onDoubleClick={() => confirm(r.value)}
                      >
                        <td className="ocr-picker__project">
                          {r.project}
                          {isCurrent && <span className="ocr-picker__current-tag">current</span>}
                        </td>
                        <td><span className={`ocr-picker__badge ocr-picker__badge--${r.kind}`}>{KIND_LABEL[r.kind]}</span></td>
                        <td className="ocr-picker__muted">{r.detail}</td>
                        <td className="num ocr-picker__acc">{fmtAcc(r.acc)}</td>
                        <td className="ocr-picker__muted">{fmtDate(r.createdAt)}</td>
                        <td className="num" title={users.map(u => u.name).join('\n')}>
                          {r.kind === 'builtin' ? '' : usage == null ? (usageError ? '?' : '…') : users.length === 0
                            ? <span className="ocr-picker__muted">—</span>
                            : <span className="ocr-picker__used">
                                {users.length} recipe{users.length > 1 ? 's' : ''}{usedHere ? ' (incl. this)' : ''}
                              </span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="ocr-picker__footer">
              <span className="ocr-picker__muted">
                Double-click a row or press Enter to select · hover “Used by” to see recipe names
              </span>
              <div className="ocr-picker__actions">
                <button type="button" className="ocr-picker__btn" onClick={() => setOpen(false)}>Cancel</button>
                <button
                  type="button"
                  className="ocr-picker__btn ocr-picker__btn--primary"
                  disabled={!rows.some(r => r.value === pending)}
                  onClick={() => confirm(pending)}
                >
                  Select
                </button>
              </div>
            </div>
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}
