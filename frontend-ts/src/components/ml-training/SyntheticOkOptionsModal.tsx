import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  mlTrainingAPI, FontInfo, SynthOkStyle, SyntheticOkOptions, SyntheticOkCrop,
} from '@/services/mlTraining';
import { useToast } from '@/contexts/ToastContext';
import ConfirmDialog from '@/components/shared/ConfirmDialog';

const DEFAULTS: Required<Omit<SyntheticOkOptions, 'font_paths'>> & { font_paths: string[] | null } = {
  font_paths: null,
  style_sample_n: 64,
  sample_strategy: 'random',
  rotation_max_deg: 5,
  size_jitter: 0.30,
  char_fill_min: 0.85,
  char_fill_max: 0.95,
  bg_per_char: 24,
  fill_min: 0.10,
  fill_max: 0.65,
  min_contrast: 20,
  max_retries: 4,
};

interface Props {
  open: boolean;
  projectId: string;
  initial: SyntheticOkOptions;
  previewChars: string;
  onClose: () => void;
  onSave: (opts: SyntheticOkOptions) => void;
}

const rgbCss = (bgr: number[] | undefined) => {
  if (!bgr || bgr.length < 3) return '#888';
  const [b, g, r] = bgr;
  return `rgb(${r},${g},${b})`;
};

export default function SyntheticOkOptionsModal({
  open, projectId, initial, previewChars, onClose, onSave,
}: Props) {
  const toast = useToast();
  const [opts, setOpts] = useState<SyntheticOkOptions>({ ...DEFAULTS, ...initial });

  const [fonts, setFonts] = useState<FontInfo[]>([]);
  const [loadingFonts, setLoadingFonts] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [fontSearch, setFontSearch] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [confirmDelete, setConfirmDelete] = useState<FontInfo | null>(null);

  const filteredFonts = useMemo(() => {
    const q = fontSearch.trim().toLowerCase();
    if (!q) return fonts;
    return fonts.filter(f =>
      f.name.toLowerCase().includes(q) ||
      f.filename.toLowerCase().includes(q) ||
      f.source.includes(q),
    );
  }, [fonts, fontSearch]);

  const [openSection, setOpenSection] = useState<Record<string, boolean>>({
    fonts: true, style: false, bg: false, gen: false, val: false, live: false,
  });
  const toggleSection = (k: string) => setOpenSection(p => ({ ...p, [k]: !p[k] }));

  const [style, setStyle] = useState<SynthOkStyle | null>(null);
  const [loadingStyle, setLoadingStyle] = useState(false);
  const [bgPool, setBgPool] = useState<Record<string, string[]>>({});
  const [loadingBg, setLoadingBg] = useState(false);
  const [livePreview, setLivePreview] = useState<SyntheticOkCrop[]>([]);
  const [loadingLive, setLoadingLive] = useState(false);

  useEffect(() => {
    if (!open) return;
    setOpts({ ...DEFAULTS, ...initial });
    setOpenSection({ fonts: true, style: false, bg: false, gen: false, val: false, live: false });
    setStyle(null);
    setBgPool({});
    setLivePreview([]);
  }, [open]);

  const fontsInitRef = useRef(false);
  useEffect(() => {
    if (!open) return;
    setLoadingFonts(true);
    mlTrainingAPI.fontsDiscover(previewChars)
      .then(list => {
        setFonts(list);
        if (!fontsInitRef.current && opts.font_paths === undefined && list.length > 0) {
          const notoFonts = list.filter(f => /noto/i.test(f.name));
          if (notoFonts.length > 0) {
            setOpts(p => ({ ...p, font_paths: notoFonts.map(f => f.path) }));
            const names = notoFonts.map(f => f.name).join(', ');
            toast.info(`Default fonts: ${names}`);
          }
          fontsInitRef.current = true;
        }
      })
      .catch(() => toast.error('Failed to load fonts'))
      .finally(() => setLoadingFonts(false));
  }, [open, previewChars]);

  const styleDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fetchStyle = useCallback(() => {
    setLoadingStyle(true);
    mlTrainingAPI.synthOkStyle(projectId, {
      style_sample_n: opts.style_sample_n,
      sample_strategy: opts.sample_strategy,
      include_imported: true,
      n_thumbnails: 4,
    }).then(setStyle)
      .catch((e: any) => toast.error(e?.response?.data?.detail ?? 'Style fetch failed'))
      .finally(() => setLoadingStyle(false));
  }, [projectId, opts.style_sample_n, opts.sample_strategy]);

  useEffect(() => {
    if (!open || !openSection.style) return;
    if (styleDebounceRef.current) clearTimeout(styleDebounceRef.current);
    styleDebounceRef.current = setTimeout(fetchStyle, 500);
    return () => { if (styleDebounceRef.current) clearTimeout(styleDebounceRef.current); };
  }, [open, openSection.style, opts.style_sample_n, opts.sample_strategy, fetchStyle]);

  const fetchBgPool = useCallback(() => {
    setLoadingBg(true);
    mlTrainingAPI.synthOkBgPool(projectId, { n_per_char: 4, chars: '' })
      .then(setBgPool)
      .catch((e: any) => toast.error(e?.response?.data?.detail ?? 'BG pool fetch failed'))
      .finally(() => setLoadingBg(false));
  }, [projectId]);

  useEffect(() => {
    if (open && openSection.bg && Object.keys(bgPool).length === 0 && !loadingBg) {
      fetchBgPool();
    }
  }, [open, openSection.bg]);

  const handleGenerateLive = () => {
    setLoadingLive(true);
    mlTrainingAPI.previewSyntheticOk(projectId, {
      target_n_per_char: 1,
      only_below_threshold: false,
      char_filter: null,
      ...opts,
    }).then(r => setLivePreview(r.crops.slice(0, 8)))
      .catch((e: any) => toast.error(e?.response?.data?.detail ?? 'Preview failed'))
      .finally(() => setLoadingLive(false));
  };

  const handleUpload = async (file: File) => {
    setUploading(true);
    try {
      const f = await mlTrainingAPI.fontUpload(file);
      setFonts(prev => [...prev, f]);
      if (f.warning) toast.warning(f.warning);
      else toast.success(`Uploaded ${f.filename}`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Upload failed');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const doDeleteFont = async (f: FontInfo) => {
    try {
      await mlTrainingAPI.fontDelete(f.filename);
      setFonts(prev => prev.filter(x => x.path !== f.path));
      setOpts(prev => ({
        ...prev,
        font_paths: prev.font_paths?.filter(p => p !== f.path) ?? null,
      }));
      toast.success('Font deleted');
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Delete failed');
    } finally {
      setConfirmDelete(null);
    }
  };

  const selectedSet = useMemo(() => {
    if (opts.font_paths == null) return new Set(fonts.map(f => f.path));
    return new Set(opts.font_paths);
  }, [opts.font_paths, fonts]);
  const usingAll = opts.font_paths == null || (fonts.length > 0 && selectedSet.size === fonts.length);

  const toggleFont = (path: string) => {
    setOpts(prev => {
      const base = prev.font_paths ?? fonts.map(f => f.path);
      const cur = new Set(base);
      if (cur.has(path)) cur.delete(path); else cur.add(path);
      const next = Array.from(cur);
      const allSelected = next.length === fonts.length && fonts.every(f => cur.has(f.path));
      return { ...prev, font_paths: allSelected ? null : next };
    });
  };

  const selectAllFonts = () => setOpts(p => ({ ...p, font_paths: null }));
  const clearAllFonts  = () => setOpts(p => ({ ...p, font_paths: [] }));

  const handleSave = () => {
    onSave(opts);
    onClose();
  };

  const handleResetDefaults = () => setOpts({ ...DEFAULTS });

  const handleClearCache = async () => {
    try {
      await mlTrainingAPI.synthOkCacheClear(projectId);
      toast.success('Cache cleared');
      if (openSection.style) fetchStyle();
      if (openSection.bg) fetchBgPool();
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Cache clear failed');
    }
  };

  if (!open) return null;

  return (
    <>
      <div className="ml-modal-overlay" onClick={onClose}>
        <div className="ml-modal ml-synth-ok-modal" onClick={e => e.stopPropagation()}>
          <div className="ml-modal-header">
            <h3>Synthetic OK Options</h3>
            <button className="ml-modal-close" onClick={onClose} aria-label="Close">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </button>
          </div>

          <div className="ml-modal-body ml-synth-ok-body">

            {/* ── FONTS ── */}
            <Section
              title={`Fonts (${selectedSet.size}/${fonts.length} selected${usingAll ? ' — all' : ''})`}
              isOpen={openSection.fonts} onToggle={() => toggleSection('fonts')}>
              <div className="ml-synth-section-actions">
                <button className="ml-btn ml-btn-secondary ml-btn-sm"
                  onClick={selectAllFonts} disabled={usingAll}>
                  Select all
                </button>
                <button className="ml-btn ml-btn-secondary ml-btn-sm"
                  onClick={clearAllFonts} disabled={selectedSet.size === 0}>
                  Clear all
                </button>
                <button className="ml-btn ml-btn-secondary ml-btn-sm"
                  onClick={() => fileInputRef.current?.click()} disabled={uploading}
                  style={{ marginLeft: 'auto' }}>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" style={{ marginRight: 4 }}>
                    <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                  {uploading ? 'Uploading…' : 'Upload .ttf/.otf'}
                </button>
                <input ref={fileInputRef} type="file" accept=".ttf,.otf,.ttc"
                  style={{ display: 'none' }}
                  onChange={e => {
                    const f = e.target.files?.[0];
                    if (f) handleUpload(f);
                  }} />
              </div>
              {selectedSet.size === 0 && fonts.length > 0 && (
                <div className="ml-imported-toggle-warn" style={{ marginTop: 2 }}>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <path d="M12 9v4M12 17h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
                      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  No font selected — will fall back to all auto-discovered fonts.
                </div>
              )}
              {loadingFonts && <div className="ml-empty-state" style={{ minHeight: 60 }}>
                <div className="ml-loading-spinner" />
              </div>}
              {!loadingFonts && fonts.length > 0 && (
                <div className="ml-font-search-row">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
                    <path d="M21 21l-4.35-4.35" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                  <input className="ml-form-input ml-font-search-input"
                    placeholder="Search font name (e.g. noto, bold, project)…"
                    value={fontSearch}
                    onChange={e => setFontSearch(e.target.value)} />
                  {fontSearch && (
                    <button className="ml-font-search-clear" onClick={() => setFontSearch('')}
                      aria-label="Clear search" title="Clear">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                        <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                      </svg>
                    </button>
                  )}
                  <span className="ml-font-search-count">
                    {fontSearch ? `${filteredFonts.length}/${fonts.length}` : `${fonts.length}`}
                  </span>
                </div>
              )}
              <div className="ml-font-list">
                {filteredFonts.map(f => {
                  const isPicked = selectedSet.has(f.path);
                  return (
                    <div key={f.path}
                      className={`ml-font-item${isPicked ? ' selected' : ''}`}
                      onClick={() => toggleFont(f.path)}
                      role="button" tabIndex={0}
                      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleFont(f.path); } }}
                      style={{ cursor: 'pointer' }}>
                      <input type="checkbox"
                        checked={isPicked}
                        onChange={() => toggleFont(f.path)}
                        onClick={(e) => e.stopPropagation()} />
                      <div className="ml-font-meta">
                        <span className="ml-font-name">{f.name}</span>
                        <span className={`ml-font-source-tag ${f.source}`}>{f.source}</span>
                        {f.stroke_ratio != null && (
                          <span className="ml-font-stroke">stroke {f.stroke_ratio.toFixed(2)}</span>
                        )}
                      </div>
                      {f.preview_b64 && (
                        <img className="ml-font-preview" alt={f.name}
                          src={`data:image/png;base64,${f.preview_b64}`} />
                      )}
                      {f.source === 'project' && (
                        <button className="ml-btn ml-btn-secondary ml-btn-sm ml-font-delete-btn"
                          onClick={(e) => { e.stopPropagation(); setConfirmDelete(f); }}
                          title="Delete this uploaded font file (permanent)">
                          <svg width="11" height="11" viewBox="0 0 24 24" fill="none">
                            <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2M10 11v6M14 11v6"
                              stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        </button>
                      )}
                    </div>
                  );
                })}
                {fonts.length === 0 && !loadingFonts && (
                  <div className="ml-empty-state" style={{ minHeight: 50, fontSize: 12 }}>
                    No fonts. Upload a .ttf/.otf above.
                  </div>
                )}
                {fonts.length > 0 && filteredFonts.length === 0 && (
                  <div className="ml-empty-state" style={{ minHeight: 50, fontSize: 12 }}>
                    No font matches "{fontSearch}".
                  </div>
                )}
              </div>
            </Section>

            {/* ── STYLE FINGERPRINT ── */}
            <Section title="Style fingerprint"
              isOpen={openSection.style} onToggle={() => toggleSection('style')}>
              <div className="ml-synth-row">
                <label className="ml-form-label">Sample N</label>
                <input className="ml-form-input ml-num-sm" type="number" min={4} max={500}
                  value={opts.style_sample_n}
                  onChange={e => setOpts(p => ({ ...p, style_sample_n: Math.max(4, Number(e.target.value) || 64) }))} />
                <span className="ml-synth-strategy">
                  {(['first', 'random', 'stratified'] as const).map(s => (
                    <label key={s} className="ml-radio">
                      <input type="radio" checked={opts.sample_strategy === s}
                        onChange={() => setOpts(p => ({ ...p, sample_strategy: s }))} /> {s}
                    </label>
                  ))}
                </span>
                <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={fetchStyle} disabled={loadingStyle}>
                  {loadingStyle ? 'Loading…' : 'Re-extract'}
                </button>
              </div>
              {style && (
                <>
                  <div className="ml-synth-row">
                    <span className="ml-color-swatch" style={{ background: rgbCss(style.ink_bgr) }} title="ink" />
                    <span>ink</span>
                    <span className="ml-color-swatch" style={{ background: rgbCss(style.bg_bgr) }} title="bg" />
                    <span>bg</span>
                    <span className="ml-synth-stat">size {style.mean_w}×{style.mean_h}</span>
                    <span className="ml-synth-stat">blur σ {style.blur_sigma.toFixed(2)}</span>
                    <span className="ml-synth-stat">noise σ {style.noise_std.toFixed(2)}</span>
                    <span className="ml-synth-stat">N={style.n_analyzed}</span>
                  </div>
                  <div className="ml-thumb-row">
                    {style.sample_b64s.map((b, i) => (
                      <img key={i} alt={`s${i}`} src={`data:image/jpeg;base64,${b}`} className="ml-thumb-sm" />
                    ))}
                  </div>
                </>
              )}
              <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={handleClearCache}>
                Clear cache & rebuild
              </button>
            </Section>

            {/* ── BG POOL ── */}
            <Section title="Background pool"
              isOpen={openSection.bg} onToggle={() => toggleSection('bg')}>
              <div className="ml-synth-row">
                <label className="ml-form-label">BG/char</label>
                <input className="ml-form-input ml-num-sm" type="number" min={4} max={48}
                  value={opts.bg_per_char}
                  onChange={e => setOpts(p => ({ ...p, bg_per_char: Math.max(4, Number(e.target.value) || 24) }))} />
                <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={fetchBgPool} disabled={loadingBg}>
                  {loadingBg ? 'Loading…' : 'Refresh preview'}
                </button>
              </div>
              {Object.entries(bgPool).map(([cid, b64s]) => (
                <div key={cid} className="ml-bg-pool-row">
                  <span className="ml-bg-pool-cid">'{cid}'</span>
                  <div className="ml-thumb-row">
                    {b64s.map((b, i) => (
                      <img key={i} src={`data:image/jpeg;base64,${b}`} alt="" className="ml-thumb-sm" />
                    ))}
                  </div>
                </div>
              ))}
            </Section>

            {/* ── GENERATION ── */}
            <Section title="Generation params"
              isOpen={openSection.gen} onToggle={() => toggleSection('gen')}>
              <SliderRow label="Rotation max (°)" value={opts.rotation_max_deg ?? 5}
                min={0} max={15} step={0.5}
                onChange={v => setOpts(p => ({ ...p, rotation_max_deg: v }))} />
              <SliderRow label="Size jitter" value={opts.size_jitter ?? 0.30}
                min={0} max={0.5} step={0.01} fmt={v => `${(v * 100).toFixed(0)}%`}
                onChange={v => setOpts(p => ({ ...p, size_jitter: v }))} />
              <SliderRow label="Char fill min" value={opts.char_fill_min ?? 0.85}
                min={0.5} max={1.0} step={0.01} fmt={v => `${(v * 100).toFixed(0)}%`}
                onChange={v => setOpts(p => ({ ...p, char_fill_min: v }))} />
              <SliderRow label="Char fill max" value={opts.char_fill_max ?? 0.95}
                min={0.5} max={1.0} step={0.01} fmt={v => `${(v * 100).toFixed(0)}%`}
                onChange={v => setOpts(p => ({ ...p, char_fill_max: v }))} />
            </Section>

            {/* ── VALIDATION ── */}
            <Section title="Validation"
              isOpen={openSection.val} onToggle={() => toggleSection('val')}>
              <SliderRow label="Fill min" value={opts.fill_min ?? 0.10}
                min={0.0} max={0.5} step={0.01} fmt={v => `${(v * 100).toFixed(0)}%`}
                onChange={v => setOpts(p => ({ ...p, fill_min: v }))} />
              <SliderRow label="Fill max" value={opts.fill_max ?? 0.65}
                min={0.3} max={0.9} step={0.01} fmt={v => `${(v * 100).toFixed(0)}%`}
                onChange={v => setOpts(p => ({ ...p, fill_max: v }))} />
              <SliderRow label="Min contrast" value={opts.min_contrast ?? 20}
                min={5} max={80} step={1}
                onChange={v => setOpts(p => ({ ...p, min_contrast: v }))} />
              <SliderRow label="Max retries" value={opts.max_retries ?? 4}
                min={1} max={10} step={1}
                onChange={v => setOpts(p => ({ ...p, max_retries: Math.round(v) }))} />
            </Section>

            {/* ── LIVE PREVIEW ── */}
            <Section title="Live preview"
              isOpen={openSection.live} onToggle={() => toggleSection('live')}>
              <button className="ml-btn ml-btn-secondary ml-btn-sm"
                onClick={handleGenerateLive} disabled={loadingLive}>
                {loadingLive ? 'Generating…' : 'Generate sample crops'}
              </button>
              <div className="ml-thumb-row" style={{ marginTop: 6 }}>
                {livePreview.map((c, i) => (
                  <div key={i} className="ml-live-card">
                    <img src={`data:image/jpeg;base64,${c.crop_b64}`} alt={c.char_id} className="ml-thumb-sm" />
                    <span>'{c.char_id}'</span>
                  </div>
                ))}
              </div>
            </Section>

          </div>

          <div className="ml-modal-footer ml-synth-ok-footer">
            <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={handleResetDefaults}>
              Reset
            </button>
            <div style={{ flex: 1 }} />
            <button className="ml-btn ml-btn-secondary" onClick={onClose}>Cancel</button>
            <button className="ml-btn ml-btn-primary" onClick={handleSave}>Save</button>
          </div>
        </div>
      </div>

      <ConfirmDialog
        isOpen={confirmDelete !== null}
        title="Delete font?"
        message={confirmDelete ? `Delete "${confirmDelete.filename}" from project fonts? This cannot be undone.` : ''}
        type="danger"
        confirmText="Delete"
        onClose={() => setConfirmDelete(null)}
        onConfirm={() => confirmDelete && doDeleteFont(confirmDelete)}
      />
    </>
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

function SliderRow({ label, value, min, max, step, fmt, onChange }: {
  label: string; value: number; min: number; max: number; step: number;
  fmt?: (v: number) => string; onChange: (v: number) => void;
}) {
  return (
    <div className="ml-synth-slider-row">
      <span className="ml-form-label">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ flex: 1, accentColor: '#3b82f6' }} />
      <span className="ml-synth-slider-value">{fmt ? fmt(value) : value}</span>
    </div>
  );
}
