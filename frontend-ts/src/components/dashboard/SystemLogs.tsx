import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import api, { API_BASE_URL } from '@/services/http';
import { useToast } from '@/contexts/ToastContext';
import ConfirmDialog from '@/components/shared/ConfirmDialog';
import '@/styles/SystemLogs.css';

interface CategoryItem {
  category: string;
  file_count: number;
  size: number;
  latest_date: string | null;
}

interface DateItem {
  date: string;
  size: number;
  modified_at: number;
  compressed: boolean;
}

interface CleanupConfig {
  enabled: boolean;
  keep_days: number;
  compress_after_days: number;
  schedule_hour: number;
  schedule_minute: number;
}

type LevelFilter = '' | 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';

interface ConfirmDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  onConfirm: (() => void) | null;
}

const formatBytes = (n: number): string => {
  if (!n) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
};

const todayStr = (): string => new Date().toISOString().slice(0, 10);

const lineLevel = (line: string): LevelFilter | '' => {
  if (line.includes(' CRITICAL ') || line.includes('- CRITICAL -')) return 'CRITICAL';
  if (line.includes(' ERROR ')    || line.includes('- ERROR -'))    return 'ERROR';
  if (line.includes(' WARNING ')  || line.includes('- WARNING -'))  return 'WARNING';
  if (line.includes(' INFO ')     || line.includes('- INFO -'))     return 'INFO';
  if (line.includes(' DEBUG ')    || line.includes('- DEBUG -'))    return 'DEBUG';
  return '';
};

// ── Inline SVG icons (theme-aware via stroke="currentColor") ──────────────
const IconPlay = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
    <path d="M5 4l14 8-14 8V4z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
  </svg>
);
const IconStop = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
    <rect x="6" y="6" width="12" height="12" rx="1" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
  </svg>
);
const IconDownload = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);
const IconTrash = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
    <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2M10 11v6M14 11v6"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);
const IconInfo = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
    <path d="M12 16v-5M12 8h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

const SystemLogs: React.FC = () => {
  const toast = useToast();

  // Tree state
  const [categories, setCategories] = useState<CategoryItem[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [dates, setDates] = useState<DateItem[]>([]);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);

  // Viewer state
  const [lines, setLines] = useState<string[]>([]);
  const [loadingFile, setLoadingFile] = useState(false);
  const [tailLines, setTailLines] = useState<number>(500);
  const [search, setSearch] = useState('');
  const [levelFilter, setLevelFilter] = useState<LevelFilter>('');
  const [autoScroll, setAutoScroll] = useState(true);

  // Live tail state
  const [liveOn, setLiveOn] = useState(false);
  const liveAbortRef = useRef<AbortController | null>(null);

  // Cleanup config
  const [cleanupCfg, setCleanupCfg] = useState<CleanupConfig | null>(null);
  const [cfgSaving, setCfgSaving] = useState(false);

  // Confirm dialog
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false, title: '', message: '', type: 'warning', onConfirm: null,
  });

  const viewerRef = useRef<HTMLDivElement>(null);

  // ── Loaders ────────────────────────────────────────────────────────────────
  const loadCategories = useCallback(async () => {
    try {
      const { data } = await api.get<CategoryItem[]>('/system-logs/categories');
      setCategories(data);
      if (!selectedCategory && data.length > 0) {
        const first = data.find(c => c.file_count > 0) ?? data[0];
        if (first) setSelectedCategory(first.category);
      }
    } catch (e: any) {
      toast.error(`Failed to load categories: ${e?.response?.data?.detail || e.message}`);
    }
  }, [selectedCategory, toast]);

  const loadDates = useCallback(async (category: string) => {
    try {
      const { data } = await api.get<DateItem[]>(`/system-logs/${category}/dates`);
      setDates(data);
      setSelectedDate(data[0]?.date ?? null);
    } catch (e: any) {
      toast.error(`Failed to load dates: ${e?.response?.data?.detail || e.message}`);
      setDates([]);
      setSelectedDate(null);
    }
  }, [toast]);

  const loadFile = useCallback(async (category: string, date: string) => {
    setLoadingFile(true);
    try {
      const { data } = await api.get(`/system-logs/${category}/${date}`, {
        params: { tail_lines: tailLines, level: levelFilter || undefined },
      });
      setLines(data.lines || []);
    } catch (e: any) {
      toast.error(`Failed to load file: ${e?.response?.data?.detail || e.message}`);
      setLines([]);
    } finally {
      setLoadingFile(false);
    }
  }, [tailLines, levelFilter, toast]);

  const loadCleanupCfg = useCallback(async () => {
    try {
      const { data } = await api.get<CleanupConfig>('/system-logs/cleanup-config');
      setCleanupCfg(data);
    } catch {
      setCleanupCfg({
        enabled: false, keep_days: 30, compress_after_days: 7,
        schedule_hour: 0, schedule_minute: 30,
      });
    }
  }, []);

  // ── Effects ────────────────────────────────────────────────────────────────
  useEffect(() => { loadCategories(); loadCleanupCfg(); }, [loadCategories, loadCleanupCfg]);

  useEffect(() => {
    if (selectedCategory) loadDates(selectedCategory);
  }, [selectedCategory, loadDates]);

  useEffect(() => {
    stopLive();
    if (selectedCategory && selectedDate) {
      loadFile(selectedCategory, selectedDate);
    } else {
      setLines([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCategory, selectedDate, tailLines, levelFilter]);

  useEffect(() => {
    if (autoScroll && viewerRef.current) {
      viewerRef.current.scrollTop = viewerRef.current.scrollHeight;
    }
  }, [lines, autoScroll]);

  useEffect(() => () => stopLive(), []);

  // ── Live tail (SSE via fetch + ReadableStream) ─────────────────────────────
  const startLive = useCallback(async () => {
    if (!selectedCategory || !selectedDate || liveOn) return;
    if (selectedDate !== todayStr()) {
      toast.info("Live tail is only available for today's log");
      return;
    }
    const ac = new AbortController();
    liveAbortRef.current = ac;
    setLiveOn(true);
    try {
      const token = localStorage.getItem('access_token');
      const resp = await fetch(`${API_BASE_URL}/system-logs/${selectedCategory}/${selectedDate}/tail`, {
        signal: ac.signal,
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`Live tail failed: ${resp.status} ${resp.statusText}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (!ac.signal.aborted) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let idx = buffer.indexOf('\n\n');
        while (idx !== -1) {
          const block = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          for (const ln of block.split('\n')) {
            if (ln.startsWith('data: ')) {
              try {
                const obj = JSON.parse(ln.slice(6));
                if (obj.line) {
                  setLines(prev => {
                    const next = prev.concat([obj.line as string]);
                    return next.length > 5000 ? next.slice(-5000) : next;
                  });
                }
              } catch { /* ignore */ }
            }
          }
          idx = buffer.indexOf('\n\n');
        }
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') {
        toast.error(`Live tail error: ${e.message}`);
      }
    } finally {
      setLiveOn(false);
      liveAbortRef.current = null;
    }
  }, [selectedCategory, selectedDate, liveOn, toast]);

  const stopLive = useCallback(() => {
    if (liveAbortRef.current) {
      liveAbortRef.current.abort();
      liveAbortRef.current = null;
    }
    setLiveOn(false);
  }, []);

  // ── Mutations ──────────────────────────────────────────────────────────────
  const downloadFile = () => {
    if (!selectedCategory || !selectedDate) return;
    const token = localStorage.getItem('access_token');
    fetch(`${API_BASE_URL}/system-logs/${selectedCategory}/${selectedDate}/download`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    }).then(async r => {
      if (!r.ok) throw new Error(`Download failed: ${r.status}`);
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedCategory}_${selectedDate}.log`;
      a.click();
      URL.revokeObjectURL(url);
    }).catch(e => toast.error(e.message));
  };

  const askDeleteFile = () => {
    if (!selectedCategory || !selectedDate) return;
    setConfirmDialog({
      isOpen: true,
      title: 'Delete log file',
      message: `Delete ${selectedCategory}/${selectedDate}.log? This cannot be undone.`,
      type: 'danger',
      onConfirm: async () => {
        try {
          await api.delete(`/system-logs/${selectedCategory}/${selectedDate}`);
          toast.success('Deleted');
          setSelectedDate(null);
          if (selectedCategory) {
            await loadDates(selectedCategory);
          }
          await loadCategories();
        } catch (e: any) {
          toast.error(`Delete failed: ${e?.response?.data?.detail || e.message}`);
        }
      },
    });
  };

  const askDeleteCategory = () => {
    if (!selectedCategory) return;
    setConfirmDialog({
      isOpen: true,
      title: 'Delete entire category',
      message: `Delete ALL log files in '${selectedCategory}'? This cannot be undone.`,
      type: 'danger',
      onConfirm: async () => {
        try {
          await api.delete(`/system-logs/${selectedCategory}`);
          toast.success('Category cleared');
          setDates([]);
          setSelectedDate(null);
          await loadCategories();
        } catch (e: any) {
          toast.error(`Delete failed: ${e?.response?.data?.detail || e.message}`);
        }
      },
    });
  };

  const saveCleanupCfg = async () => {
    if (!cleanupCfg) return;
    if (cleanupCfg.compress_after_days >= cleanupCfg.keep_days) {
      toast.error('compress_after_days must be < keep_days');
      return;
    }
    setCfgSaving(true);
    try {
      const { data } = await api.put<CleanupConfig>('/system-logs/cleanup-config', cleanupCfg);
      setCleanupCfg(data);
      toast.success('Cleanup config saved');
    } catch (e: any) {
      toast.error(`Save failed: ${e?.response?.data?.detail || e.message}`);
    } finally {
      setCfgSaving(false);
    }
  };

  // ── Derived ────────────────────────────────────────────────────────────────
  const filteredLines = useMemo(() => {
    if (!search) return lines;
    const q = search.toLowerCase();
    return lines.filter(ln => ln.toLowerCase().includes(q));
  }, [lines, search]);

  const totalDisk = useMemo(
    () => categories.reduce((acc, c) => acc + c.size, 0),
    [categories],
  );

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="syslogs-root" style={{ display: 'flex', height: 'calc(100vh - 160px)', minHeight: 480 }}>
      {/* Sidebar: Categories + Cleanup config */}
      <div className="syslogs-pane" style={{ width: 240, flexShrink: 0, display: 'flex', flexDirection: 'column' }}>
        <div className="syslogs-pane-header" style={{ padding: '12px 16px' }}>
          <div style={{ fontSize: 11, marginBottom: 4 }}>CATEGORIES</div>
          <div style={{ fontSize: 11, opacity: 0.7 }}>Total: {formatBytes(totalDisk)}</div>
        </div>
        <div style={{ overflow: 'auto', flex: 1 }}>
          {categories.map(c => (
            <button
              key={c.category}
              type="button"
              onClick={() => setSelectedCategory(c.category)}
              className={`syslogs-item ${selectedCategory === c.category ? 'active' : ''}`}
            >
              <div className="syslogs-item-name">{c.category}</div>
              <div className="syslogs-item-meta">
                {c.file_count} file{c.file_count !== 1 ? 's' : ''} · {formatBytes(c.size)}
              </div>
            </button>
          ))}
        </div>

        {/* Cleanup config panel */}
        {cleanupCfg && (
          <div className="syslogs-pane-section" style={{ padding: 12, fontSize: 12 }}>
            <div className="syslogs-cleanup-label">AUTO CLEANUP</div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={cleanupCfg.enabled}
                onChange={e => setCleanupCfg({ ...cleanupCfg, enabled: e.target.checked })}
              />
              <span>Enabled</span>
            </label>
            <div style={{ marginBottom: 6 }}>
              <label className="syslogs-cleanup-label">Keep (days)</label>
              <input
                type="number" min={1} max={3650}
                value={cleanupCfg.keep_days}
                onChange={e => setCleanupCfg({ ...cleanupCfg, keep_days: parseInt(e.target.value) || 1 })}
                className="syslogs-num"
                style={{ width: '100%' }}
              />
            </div>
            <div style={{ marginBottom: 6 }}>
              <label className="syslogs-cleanup-label">Compress after (days)</label>
              <input
                type="number" min={1}
                value={cleanupCfg.compress_after_days}
                onChange={e => setCleanupCfg({ ...cleanupCfg, compress_after_days: parseInt(e.target.value) || 1 })}
                className="syslogs-num"
                style={{ width: '100%' }}
              />
            </div>
            <div style={{ marginBottom: 8 }}>
              <label className="syslogs-cleanup-label">Schedule (HH:MM)</label>
              <div style={{ display: 'flex', gap: 4 }}>
                <input
                  type="number" min={0} max={23}
                  value={cleanupCfg.schedule_hour}
                  onChange={e => setCleanupCfg({ ...cleanupCfg, schedule_hour: parseInt(e.target.value) || 0 })}
                  className="syslogs-num"
                  style={{ width: '50%' }}
                />
                <input
                  type="number" min={0} max={59}
                  value={cleanupCfg.schedule_minute}
                  onChange={e => setCleanupCfg({ ...cleanupCfg, schedule_minute: parseInt(e.target.value) || 0 })}
                  className="syslogs-num"
                  style={{ width: '50%' }}
                />
              </div>
            </div>
            <button
              type="button"
              onClick={saveCleanupCfg}
              disabled={cfgSaving}
              className="syslogs-btn syslogs-btn-primary"
              style={{ width: '100%', justifyContent: 'center' }}
            >
              {cfgSaving ? 'Saving…' : 'Save'}
            </button>
          </div>
        )}
      </div>

      {/* Middle: Dates */}
      <div className="syslogs-pane" style={{ width: 200, flexShrink: 0, display: 'flex', flexDirection: 'column' }}>
        <div className="syslogs-pane-header"
          style={{ padding: '12px 16px', fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span>DATES</span>
          {selectedCategory && dates.length > 0 && (
            <button
              type="button"
              onClick={askDeleteCategory}
              title="Clear all files in this category"
              className="syslogs-btn-link"
            >
              Clear all
            </button>
          )}
        </div>
        <div style={{ overflow: 'auto', flex: 1 }}>
          {dates.map(d => (
            <button
              key={d.date}
              type="button"
              onClick={() => setSelectedDate(d.date)}
              className={`syslogs-item ${selectedDate === d.date ? 'active' : ''}`}
              style={{ padding: '8px 16px' }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span className="syslogs-item-name-sm">{d.date}</span>
                {d.compressed && <span className="syslogs-tag-gz">GZ</span>}
                {d.date === todayStr() && <span className="syslogs-tag-live">● LIVE</span>}
              </div>
              <div className="syslogs-item-meta">{formatBytes(d.size)}</div>
            </button>
          ))}
          {dates.length === 0 && selectedCategory && (
            <div className="syslogs-hint-empty">No log files</div>
          )}
        </div>
      </div>

      {/* Viewer */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* Toolbar */}
        <div className="syslogs-toolbar"
          style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', padding: '8px 16px' }}>
          {selectedDate && selectedDate === todayStr() && (
            <button
              type="button"
              onClick={liveOn ? stopLive : startLive}
              className={`syslogs-btn ${liveOn ? 'syslogs-btn-stop' : 'syslogs-btn-success'}`}
            >
              {liveOn ? <IconStop /> : <IconPlay />}
              <span>{liveOn ? 'Stop' : 'Live'}</span>
            </button>
          )}
          <input
            type="text"
            placeholder="Search..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="syslogs-search"
            style={{ minWidth: 180, flex: 1, maxWidth: 300 }}
          />
          <select
            value={levelFilter}
            onChange={e => setLevelFilter(e.target.value as LevelFilter)}
            className="syslogs-select"
          >
            <option value="">All levels</option>
            <option value="DEBUG">Debug</option>
            <option value="INFO">Info</option>
            <option value="WARNING">Warning</option>
            <option value="ERROR">Error</option>
            <option value="CRITICAL">Critical</option>
          </select>
          <select
            value={tailLines}
            onChange={e => setTailLines(parseInt(e.target.value))}
            className="syslogs-select"
          >
            <option value={200}>Last 200</option>
            <option value={500}>Last 500</option>
            <option value={2000}>Last 2000</option>
            <option value={10000}>Last 10000</option>
          </select>
          <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontSize: 12 }}>
            <input
              type="checkbox" checked={autoScroll}
              onChange={e => setAutoScroll(e.target.checked)}
            />
            <span>Auto-scroll</span>
          </label>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
            {selectedDate && (
              <>
                <button type="button" onClick={downloadFile} className="syslogs-btn">
                  <IconDownload />
                  <span>Download</span>
                </button>
                <button type="button" onClick={askDeleteFile} className="syslogs-btn syslogs-btn-danger">
                  <IconTrash />
                  <span>Delete</span>
                </button>
              </>
            )}
          </div>
        </div>

        {/* Lines */}
        <div ref={viewerRef} className="syslogs-viewer" style={{ flex: 1, overflow: 'auto', padding: '8px 12px' }}>
          {loadingFile && <div className="syslogs-empty">Loading…</div>}
          {!loadingFile && filteredLines.length === 0 && (
            <div className="syslogs-empty"
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <IconInfo />
              {selectedDate ? 'No matching lines' : 'Select a category and date'}
            </div>
          )}
          {!loadingFile && filteredLines.map((ln, i) => {
            const lvl = lineLevel(ln);
            return (
              <div key={i} className={`syslogs-line ${lvl ? `syslogs-line-${lvl}` : ''}`}>
                {ln}
              </div>
            );
          })}
        </div>

        {/* Status bar */}
        <div className="syslogs-statusbar"
          style={{ padding: '4px 12px', display: 'flex', gap: 12 }}>
          <span>{filteredLines.length}/{lines.length} lines</span>
          {liveOn && <span className="syslogs-status-live">● Live tailing</span>}
          {selectedCategory && selectedDate && (
            <span style={{ marginLeft: 'auto' }}>
              {selectedCategory}/{selectedDate}.log
            </span>
          )}
        </div>
      </div>

      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
        onClose={() => setConfirmDialog({ ...confirmDialog, isOpen: false })}
        onConfirm={() => {
          confirmDialog.onConfirm?.();
          setConfirmDialog({ ...confirmDialog, isOpen: false });
        }}
      />
    </div>
  );
};

export default SystemLogs;
