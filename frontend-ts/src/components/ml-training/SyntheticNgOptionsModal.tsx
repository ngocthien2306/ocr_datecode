import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  mlTrainingAPI, SeverityDist, SyntheticCrop, SyntheticNgOptions,
  NG_DEFECT_TYPES, NgDefectType,
} from '@/services/mlTraining';
import { useToast } from '@/contexts/ToastContext';

const DEFAULT_SEVERITY: SeverityDist = {
  subtle: 0,
  light:  0,
  medium: 100,
  heavy:  0,
};

// Heavy-tier cut band defaults — fraction of char bbox along cut axis.
// Mirrors DEFAULT_HEAVY_CUT_FRAC_MIN/MAX in backend ml_ng_augment.py.
const DEFAULT_CUT_FRAC_MIN = 0.10;
const DEFAULT_CUT_FRAC_MAX = 0.50;

// Cap the auto-derived char filter so the per-defect grid stays tight even
// when the project has many distinct chars.
const PREVIEW_CHARS_MAX = 8;

const DEFECT_LABELS: Record<NgDefectType, { label: string; hint: string }> = {
  cut_horizontal:   { label: 'Cut horizontal', hint: 'Horizontal band cuts through ink' },
  cut_vertical:     { label: 'Cut vertical',   hint: 'Vertical band cuts through ink' },
  segment_removal:  { label: 'Segment removal', hint: 'Removes a connected chunk of ink' },
  dropout_dots:     { label: 'Dropout dots',   hint: 'Random small holes inside the glyph' },
  crack:            { label: 'Crack',          hint: 'Thin random crack line across the glyph' },
  block_overlay:    { label: 'Block overlay',  hint: 'Opaque rectangle covering part of the char' },
  local_blob:       { label: 'Local blob',     hint: 'Smudge / ink blob near the strokes' },
  edge_erosion:     { label: 'Edge erosion',   hint: 'Top/bottom strokes eroded away' },
  tape_overlay:     { label: 'Tape overlay',   hint: 'Semi-transparent coloured tape band' },
  stroke_thinning:  { label: 'Stroke thinning', hint: 'Stroke breaks / fade — light defect' },
};

interface Props {
  open: boolean;
  projectId: string;
  initial: SyntheticNgOptions;
  // Chars actually present as OK samples in the project. The modal auto-uses
  // up to PREVIEW_CHARS_MAX of them as the source-segment filter so previews
  // never request chars the project doesn't have. Empty array = no filter.
  availableChars: string[];
  onClose: () => void;
  onSave: (opts: SyntheticNgOptions) => void;
}

function normalizeSeverity(d?: SeverityDist): SeverityDist {
  if (!d) return { ...DEFAULT_SEVERITY };
  return {
    subtle: Math.max(0, d.subtle ?? 0),
    light:  Math.max(0, d.light  ?? 0),
    medium: Math.max(0, d.medium ?? 0),
    heavy:  Math.max(0, d.heavy  ?? 0),
  };
}

export default function SyntheticNgOptionsModal({
  open, projectId, initial, availableChars, onClose, onSave,
}: Props) {
  const toast = useToast();

  const [severity, setSeverity] = useState<SeverityDist>(() =>
    normalizeSeverity(initial.severity_dist));

  // Heavy-tier cut size range (% of char bbox). Only affects cut_horizontal /
  // cut_vertical when severity='heavy'. Stored as 0..1 floats, displayed as %.
  const [cutFracMin, setCutFracMin] = useState<number>(
    initial.cut_frac_min ?? DEFAULT_CUT_FRAC_MIN);
  const [cutFracMax, setCutFracMax] = useState<number>(
    initial.cut_frac_max ?? DEFAULT_CUT_FRAC_MAX);

  // Defect type whitelist. null/undefined → treat as "all selected". We
  // keep this normalized as a Set internally for fast lookups, but persist
  // it back as either an explicit array or null (= all).
  const [enabledSet, setEnabledSet] = useState<Set<string>>(() => {
    const initSel = initial.enabled_defect_types;
    if (initSel == null) return new Set(NG_DEFECT_TYPES);  // default: all
    return new Set(initSel);
  });
  const allSelected = enabledSet.size === NG_DEFECT_TYPES.length;

  // One-shot guard so the auto-preview only fires once per modal open. Reset
  // on the open transition below.
  const autoFiredRef = useRef(false);

  // Char filter is derived from project-available chars (passed in by parent).
  // Capped at PREVIEW_CHARS_MAX so the per-defect grid stays compact. Empty
  // pool → null = "no filter" (BE falls back to every OK sample in project).
  const charFilter = useMemo<string[] | null>(() => {
    if (!availableChars || availableChars.length === 0) return null;
    return availableChars.slice(0, PREVIEW_CHARS_MAX);
  }, [availableChars]);

  const [openSection, setOpenSection] = useState<Record<string, boolean>>({
    severity: true, cut: true, defects: true, live: false,
  });
  const toggleSection = (k: string) =>
    setOpenSection(p => ({ ...p, [k]: !p[k] }));

  // Per-defect preview cache — keyed by defect type
  const [defectPreview, setDefectPreview] =
    useState<Record<string, SyntheticCrop[]>>({});
  const [defectLoading, setDefectLoading] = useState<Record<string, boolean>>({});

  // Per-severity preview cache — keyed by 'subtle' | 'light' | 'medium' | 'heavy'
  const [severityPreview, setSeverityPreview] =
    useState<Record<string, SyntheticCrop[]>>({});
  const [severityLoading, setSeverityLoading] =
    useState<Record<string, boolean>>({});

  // Overall live preview (random pick weighted by severity)
  const [livePreview, setLivePreview] = useState<SyntheticCrop[]>([]);
  const [loadingLive, setLoadingLive] = useState(false);

  useEffect(() => {
    if (!open) return;
    setSeverity(normalizeSeverity(initial.severity_dist));
    setEnabledSet(initial.enabled_defect_types == null
      ? new Set(NG_DEFECT_TYPES)
      : new Set(initial.enabled_defect_types));
    setCutFracMin(initial.cut_frac_min ?? DEFAULT_CUT_FRAC_MIN);
    setCutFracMax(initial.cut_frac_max ?? DEFAULT_CUT_FRAC_MAX);
    setOpenSection({ severity: true, cut: true, defects: true, live: false });
    setDefectPreview({});
    setDefectLoading({});
    setSeverityPreview({});
    setSeverityLoading({});
    setLivePreview([]);
    autoFiredRef.current = false;
  }, [open]);


  const toggleDefect = (defect: string) => {
    setEnabledSet(prev => {
      const next = new Set(prev);
      if (next.has(defect)) next.delete(defect); else next.add(defect);
      return next;
    });
  };
  const selectAllDefects = () => setEnabledSet(new Set(NG_DEFECT_TYPES));
  const clearAllDefects = () => setEnabledSet(new Set());

  const severityTotal = severity.subtle + severity.light + severity.medium + severity.heavy;
  const pct = (v: number) => severityTotal > 0
    ? `${((v / severityTotal) * 100).toFixed(0)}%`
    : '0%';

  const fetchSeverityPreview = useCallback(async (
    sev: keyof SeverityDist,
    opts: { silent?: boolean } = {},
  ) => {
    setSeverityLoading(p => ({ ...p, [sev]: true }));
    try {
      // Force the distribution to a single tier so every returned crop is
      // generated at that exact severity — lets the user see what each level
      // produces in isolation. Honors enabled_defect_types so previews match
      // what training would actually generate.
      const forced: SeverityDist = { subtle: 0, light: 0, medium: 0, heavy: 0 };
      forced[sev] = 100;
      const r = await mlTrainingAPI.previewSynthetic(
        projectId, 2, 'NG', forced,
        {
          enabled_defect_types: allSelected
            ? null
            : (Array.from(enabledSet) as NgDefectType[]),
          char_filter: charFilter,
          cut_frac_min: cutFracMin,
          cut_frac_max: cutFracMax,
        }
      );
      setSeverityPreview(p => ({ ...p, [sev]: r.crops.slice(0, 6) }));
      if (!opts.silent && r.crops.length === 0) {
        toast.warning(`No preview generated for ${sev} — check preview chars or label some OK samples first`);
      }
    } catch (e: any) {
      if (!opts.silent) toast.error(e?.response?.data?.detail ?? `Preview failed for ${sev}`);
    } finally {
      setSeverityLoading(p => ({ ...p, [sev]: false }));
    }
  }, [projectId, allSelected, enabledSet, charFilter, cutFracMin, cutFracMax, toast]);

  const fetchDefectPreview = useCallback(async (
    defect: NgDefectType,
    opts: { silent?: boolean } = {},
  ) => {
    setDefectLoading(p => ({ ...p, [defect]: true }));
    try {
      const r = await mlTrainingAPI.previewSynthetic(
        projectId, 2, 'NG', severity,
        {
          force_defect_type: defect,
          char_filter: charFilter,
          cut_frac_min: cutFracMin,
          cut_frac_max: cutFracMax,
        }
      );
      // Cap to 6 thumbnails per defect to keep grid tight
      setDefectPreview(p => ({ ...p, [defect]: r.crops.slice(0, 6) }));
      if (!opts.silent && r.crops.length === 0) {
        toast.warning(`No preview for ${DEFECT_LABELS[defect].label} — check preview chars or label some OK samples first`);
      }
    } catch (e: any) {
      if (!opts.silent) toast.error(e?.response?.data?.detail ?? `Preview failed for ${defect}`);
    } finally {
      setDefectLoading(p => ({ ...p, [defect]: false }));
    }
  }, [projectId, severity, charFilter, cutFracMin, cutFracMax, toast]);

  // Auto-fire only the 4 severity previews on open. The 10 defect-type cards
  // start empty — user clicks Re-create on each card when they want to
  // inspect it. Originally we auto-fired all 14 calls, but that spiked BE
  // memory by ~1GB every time the modal opened (each preview-synthetic
  // request populates the synth cache with style fingerprint + BG pool).
  useEffect(() => {
    if (!open || autoFiredRef.current) return;
    autoFiredRef.current = true;
    const sevs: (keyof SeverityDist)[] = ['subtle', 'light', 'medium', 'heavy'];
    sevs.forEach(s => { void fetchSeverityPreview(s, { silent: true }); });
  }, [open, fetchSeverityPreview]);

  const handleGenerateLive = async () => {
    setLoadingLive(true);
    try {
      const r = await mlTrainingAPI.previewSynthetic(projectId, 2, 'NG', severity, {
        enabled_defect_types: allSelected
          ? null
          : (Array.from(enabledSet) as NgDefectType[]),
        char_filter: charFilter,
        cut_frac_min: cutFracMin,
        cut_frac_max: cutFracMax,
      });
      setLivePreview(r.crops.slice(0, 12));
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Preview failed');
    } finally {
      setLoadingLive(false);
    }
  };

  const handleSave = () => {
    onSave({
      severity_dist: { ...severity },
      // Persist null when everything is selected so future code can rely on
      // "null/undefined = all", and explicit array otherwise.
      enabled_defect_types: allSelected
        ? null
        : (Array.from(enabledSet) as NgDefectType[]),
      cut_frac_min: cutFracMin,
      cut_frac_max: cutFracMax,
    });
    onClose();
  };

  const handleReset = () => {
    setSeverity({ ...DEFAULT_SEVERITY });
    setEnabledSet(new Set(NG_DEFECT_TYPES));
    setCutFracMin(DEFAULT_CUT_FRAC_MIN);
    setCutFracMax(DEFAULT_CUT_FRAC_MAX);
    setDefectPreview({});
    setSeverityPreview({});
    setLivePreview([]);
  };

  const severityRows = useMemo(() => ([
    { key: 'subtle' as const, label: 'Subtle',  color: '#9ca3af', hint: '1 light defect (mild break / tiny cut)' },
    { key: 'light'  as const, label: 'Light',   color: '#3b82f6', hint: '1-2 mild defects' },
    { key: 'medium' as const, label: 'Medium',  color: '#f59e0b', hint: '2-3 medium defects' },
    { key: 'heavy'  as const, label: 'Heavy',   color: '#ef4444', hint: '2-3 strong defects (cuts, tape, blocks)' },
  ]), []);

  if (!open) return null;

  return (
    <div className="ml-modal-overlay" onClick={onClose}>
      <div className="ml-modal ml-synth-ok-modal" onClick={e => e.stopPropagation()}>
        <div className="ml-modal-header">
          <h3>Synthetic NG Options</h3>
          <button className="ml-modal-close" onClick={onClose} aria-label="Close">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        <div className="ml-modal-body ml-synth-ok-body">

          {/* ── SEVERITY MIX ── */}
          <Section title={`Severity mix (total ${severityTotal})`}
            isOpen={openSection.severity} onToggle={() => toggleSection('severity')}>
            <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 4 }}>
              Weights are auto-normalized — relative ratio matters, not the total.
            </div>
            {severityRows.map(r => {
              const samples = severityPreview[r.key] || [];
              const loading = severityLoading[r.key];
              return (
                <div key={r.key} className="ml-ng-severity-row">
                  <div className="ml-synth-slider-row">
                    <span className="ml-form-label" style={{ color: r.color, width: 80 }} title={r.hint}>
                      {r.label}
                    </span>
                    <input type="range" min={0} max={100} step={1}
                      value={severity[r.key]}
                      onChange={e => setSeverity(p => ({ ...p, [r.key]: Number(e.target.value) }))}
                      style={{ flex: 1, accentColor: r.color }} />
                    <span className="ml-synth-slider-value">{pct(severity[r.key])}</span>
                    <button className="ml-btn ml-btn-secondary ml-btn-sm"
                      onClick={() => fetchSeverityPreview(r.key)}
                      disabled={loading}
                      title={`Preview crops at ${r.label.toLowerCase()} severity`}>
                      {loading
                        ? <span className="ml-loading-spinner" style={{ width: 10, height: 10, borderWidth: 2 }} />
                        : <>
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" style={{ marginRight: 3 }}>
                              <path d="M1 4v6h6M23 20v-6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                              <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                            Re-create
                          </>
                      }
                    </button>
                  </div>
                  <div className="ml-ng-severity-preview-row">
                    {samples.map((c, i) => (
                      <img key={i} src={`data:image/jpeg;base64,${c.crop_b64}`}
                        alt={`${r.key}-${i}`} className="ml-thumb-sm"
                        title={c.aug_type ? `${c.aug_type}${c.char_id ? ` · ${c.char_id}` : ''}` : ''} />
                    ))}
                    {!loading && samples.length === 0 && (
                      <span style={{ fontSize: 10, color: '#6b7280', fontStyle: 'italic' }}>
                        No preview yet
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
            {severityTotal === 0 && (
              <div className="ml-imported-toggle-warn" style={{ marginTop: 4 }}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden>
                  <path d="M12 9v4M12 17h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                All weights zero — backend will fall back to default 10/50/35/5.
              </div>
            )}
          </Section>

          {/* ── HEAVY CUT SIZE RANGE ── */}
          <Section title={`Heavy cut size (${Math.round(cutFracMin * 100)}–${Math.round(cutFracMax * 100)}% of char)`}
            isOpen={openSection.cut} onToggle={() => toggleSection('cut')}>
            <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 8 }}>
              Heavy-tier <b>cut_horizontal</b> / <b>cut_vertical</b> wipe ink along
              a band sized as a % of the character bbox. Background outside the
              glyph is left untouched, so the cut looks like a missing stroke
              (e.g. B→D, E→[, 0→U). Other severities use small fixed bands.
            </div>
            <div className="ml-synth-slider-row">
              <span className="ml-form-label" style={{ width: 80 }} title="Lower bound of the heavy cut band">
                Min
              </span>
              <input type="range" min={1} max={99} step={1}
                value={Math.round(cutFracMin * 100)}
                onChange={e => {
                  const v = Number(e.target.value) / 100;
                  setCutFracMin(v);
                  // Keep min <= max so the BE doesn't have to swap silently.
                  if (v > cutFracMax) setCutFracMax(v);
                }}
                style={{ flex: 1, accentColor: '#ef4444' }} />
              <span className="ml-synth-slider-value">{Math.round(cutFracMin * 100)}%</span>
            </div>
            <div className="ml-synth-slider-row">
              <span className="ml-form-label" style={{ width: 80 }} title="Upper bound of the heavy cut band">
                Max
              </span>
              <input type="range" min={1} max={99} step={1}
                value={Math.round(cutFracMax * 100)}
                onChange={e => {
                  const v = Number(e.target.value) / 100;
                  setCutFracMax(v);
                  if (v < cutFracMin) setCutFracMin(v);
                }}
                style={{ flex: 1, accentColor: '#ef4444' }} />
              <span className="ml-synth-slider-value">{Math.round(cutFracMax * 100)}%</span>
            </div>
            <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
              <button className="ml-btn ml-btn-secondary ml-btn-sm"
                onClick={() => { setCutFracMin(DEFAULT_CUT_FRAC_MIN); setCutFracMax(DEFAULT_CUT_FRAC_MAX); }}
                title="Reset to backend default range (10–50%)">
                Default 10–50%
              </button>
              <button className="ml-btn ml-btn-secondary ml-btn-sm"
                onClick={() => fetchSeverityPreview('heavy')}
                disabled={severityLoading.heavy}
                title="Re-render the heavy preview with the new range">
                Refresh heavy preview
              </button>
            </div>
          </Section>

          {/* ── DEFECT TYPES (per-type preview + enable checkbox) ── */}
          <Section title={`Defect types (${enabledSet.size}/${NG_DEFECT_TYPES.length} enabled${allSelected ? ' — all' : ''})`}
            isOpen={openSection.defects} onToggle={() => toggleSection('defects')}>
            <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 4 }}>
              {charFilter && charFilter.length > 0 ? (
                <>
                  Previewing on{' '}
                  <span style={{ fontFamily: 'monospace', fontWeight: 600, color: '#3b82f6' }}>
                    {charFilter.join(' ')}
                  </span>{' '}
                  ({charFilter.length} char{charFilter.length !== 1 ? 's' : ''} from your labeled OK)
                </>
              ) : (
                <>No labeled OK samples in this project — preview will be empty until you label some.</>
              )}
            </div>
            <div className="ml-synth-section-actions">
              <button className="ml-btn ml-btn-secondary ml-btn-sm"
                onClick={selectAllDefects} disabled={allSelected}>
                Select all
              </button>
              <button className="ml-btn ml-btn-secondary ml-btn-sm"
                onClick={clearAllDefects} disabled={enabledSet.size === 0}>
                Clear all
              </button>
              <span style={{ marginLeft: 'auto', fontSize: 11, color: '#6b7280' }}>
                Only checked defects are used during preview / training
              </span>
            </div>
            {enabledSet.size === 0 && (
              <div className="ml-imported-toggle-warn" style={{ marginTop: 4 }}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden>
                  <path d="M12 9v4M12 17h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                No defects selected — backend will fall back to all 10 types.
              </div>
            )}
            <div style={{ fontSize: 11, color: '#9ca3af', margin: '4px 0 6px' }}>
              Click <b>Re-create</b> on any card to preview that defect (the per-card preview ignores the checkbox and lets you sample each type).
            </div>
            <div className="ml-ng-defect-grid">
              {NG_DEFECT_TYPES.map(defect => {
                const meta = DEFECT_LABELS[defect];
                const samples = defectPreview[defect] || [];
                const loading = defectLoading[defect];
                const isEnabled = enabledSet.has(defect);
                return (
                  <div key={defect}
                    className={`ml-ng-defect-card${isEnabled ? ' enabled' : ' disabled'}`}>
                    <div className="ml-ng-defect-head">
                      <label className="ml-ng-defect-title" title={meta.hint}
                        style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
                        <input type="checkbox"
                          checked={isEnabled}
                          onChange={() => toggleDefect(defect)} />
                        {meta.label}
                      </label>
                      <button className="ml-btn ml-btn-secondary ml-btn-sm"
                        onClick={() => fetchDefectPreview(defect)}
                        disabled={loading}
                        title="Generate preview crops for this defect type">
                        {loading
                          ? <span className="ml-loading-spinner" style={{ width: 10, height: 10, borderWidth: 2 }} />
                          : <>
                              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" style={{ marginRight: 3 }}>
                                <path d="M1 4v6h6M23 20v-6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                              </svg>
                              Re-create
                            </>
                        }
                      </button>
                    </div>
                    <div className="ml-thumb-row" style={{ minHeight: 38 }}>
                      {samples.map((c, i) => (
                        <img key={i} src={`data:image/jpeg;base64,${c.crop_b64}`}
                          alt={`${defect}-${i}`} className="ml-thumb-sm"
                          title={c.char_id ? `char: ${c.char_id}` : ''} />
                      ))}
                      {!loading && samples.length === 0 && (
                        <span style={{ fontSize: 10, color: '#6b7280', fontStyle: 'italic' }}>
                          No preview yet
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </Section>

          {/* ── LIVE PREVIEW (random weighted) ── */}
          <Section title="Live preview (random mix)"
            isOpen={openSection.live} onToggle={() => toggleSection('live')}>
            <button className="ml-btn ml-btn-secondary ml-btn-sm"
              onClick={handleGenerateLive} disabled={loadingLive}>
              {loadingLive ? 'Generating…' : 'Generate sample crops'}
            </button>
            <div className="ml-thumb-row" style={{ marginTop: 6 }}>
              {livePreview.map((c, i) => (
                <div key={i} className="ml-live-card">
                  <img src={`data:image/jpeg;base64,${c.crop_b64}`}
                    alt={c.aug_type ?? 'ng'} className="ml-thumb-sm" />
                  <span style={{ fontSize: 9, opacity: 0.7 }}>
                    {c.aug_type ?? '—'}
                  </span>
                </div>
              ))}
            </div>
          </Section>

        </div>

        <div className="ml-modal-footer ml-synth-ok-footer">
          <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={handleReset}>
            Reset
          </button>
          <div style={{ flex: 1 }} />
          <button className="ml-btn ml-btn-secondary" onClick={onClose}>Cancel</button>
          <button className="ml-btn ml-btn-primary" onClick={handleSave}>Save</button>
        </div>
      </div>
    </div>
  );
}

function Section({ title, isOpen, onToggle, children }: {
  title: string; isOpen?: boolean; onToggle: () => void; children: React.ReactNode;
}) {
  return (
    <div className="ml-synth-section">
      <button className="ml-synth-section-header" onClick={onToggle}>
        <span style={{ width: 12 }}>{isOpen ? '▼' : '▶'}</span>
        <span>{title}</span>
      </button>
      {isOpen && <div className="ml-synth-section-body">{children}</div>}
    </div>
  );
}
