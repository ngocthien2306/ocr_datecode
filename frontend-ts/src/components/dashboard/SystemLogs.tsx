import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import api, { API_BASE_URL } from '@/services/http';
import { useToast } from '@/contexts/ToastContext';
import ConfirmDialog from '@/components/shared/ConfirmDialog';

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

const levelColor = (level: string): string => {
  switch (level) {
    case 'CRITICAL': return '#dc2626';
    case 'ERROR':    return '#ef4444';
    case 'WARNING':  return '#f59e0b';
    case 'INFO':     return '#9ca3af';
    case 'DEBUG':    return '#6b7280';
    default:         return 'inherit';
  }
};

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
  const [confirm, setConfirm] = useState<null | {
    title: string; message: string; onConfirm: () => void;
  }>(null);

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
    } catch (e: any) {
      // Not fatal — use defaults
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
    // Stop live tail whenever selection changes
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

  // Cleanup on unmount
  useEffect(() => () => stopLive(), []);

  // ── Live tail (SSE via fetch + ReadableStream so Authorization header works) ─
  const startLive = useCallback(async () => {
    if (!selectedCategory || !selectedDate || liveOn) return;
    if (selectedDate !== todayStr()) {
      toast.info('Live tail is only available for today\'s log');
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
        // SSE messages are separated by \n\n
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

  const deleteFile = () => {
    if (!selectedCategory || !selectedDate) return;
    setConfirm({
      title: 'Delete log file',
      message: `Delete ${selectedCategory}/${selectedDate}.log? This cannot be undone.`,
      onConfirm: async () => {
        try {
          await api.delete(`/system-logs/${selectedCategory}/${selectedDate}`);
          toast.success('Deleted');
          setSelectedDate(null);
          await loadDates(selectedCategory);
          await loadCategories();
        } catch (e: any) {
          toast.error(`Delete failed: ${e?.response?.data?.detail || e.message}`);
        }
      },
    });
  };

  const deleteCategory = () => {
    if (!selectedCategory) return;
    setConfirm({
      title: 'Delete entire category',
      message: `Delete ALL log files in '${selectedCategory}'? This cannot be undone.`,
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
    <div style={{ display: 'flex', height: 'calc(100vh - 160px)', minHeight: 480, gap: 0 }}>
      {/* Sidebar: Categories + Cleanup config */}
      <div style={{
        width: 240,
        flexShrink: 0,
        borderRight: '1px solid var(--color-border, #2a2f3a)',
        display: 'flex', flexDirection: 'column',
      }}>
        <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--color-border, #2a2f3a)' }}>
          <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>CATEGORIES</div>
          <div style={{ fontSize: 11, opacity: 0.5 }}>Total: {formatBytes(totalDisk)}</div>
        </div>
        <div style={{ overflow: 'auto', flex: 1 }}>
          {categories.map(c => (
            <button
              key={c.category}
              type="button"
              onClick={() => setSelectedCategory(c.category)}
              style={{
                display: 'block', width: '100%', textAlign: 'left',
                padding: '10px 16px', border: 'none',
                background: selectedCategory === c.category ? 'rgba(59,130,246,.15)' : 'transparent',
                color: 'var(--color-text, #e5e7eb)',
                cursor: 'pointer', fontSize: 13,
                borderLeft: selectedCategory === c.category
                  ? '3px solid var(--color-primary, #3b82f6)' : '3px solid transparent',
              }}
            >
              <div style={{ fontWeight: 500 }}>{c.category}</div>
              <div style={{ fontSize: 10, opacity: 0.6, marginTop: 2 }}>
                {c.file_count} file{c.file_count !== 1 ? 's' : ''} · {formatBytes(c.size)}
              </div>
            </button>
          ))}
        </div>

        {/* Cleanup config panel */}
        {cleanupCfg && (
          <div style={{
            borderTop: '1px solid var(--color-border, #2a2f3a)',
            padding: 12, fontSize: 12,
          }}>
            <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 8 }}>AUTO CLEANUP</div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={cleanupCfg.enabled}
                onChange={e => setCleanupCfg({ ...cleanupCfg, enabled: e.target.checked })}
              />
              <span>Enabled</span>
            </label>
            <div style={{ marginBottom: 6 }}>
              <label style={{ display: 'block', fontSize: 11, opacity: 0.7 }}>Keep (days)</label>
              <input
                type="number" min={1} max={3650}
                value={cleanupCfg.keep_days}
                onChange={e => setCleanupCfg({ ...cleanupCfg, keep_days: parseInt(e.target.value) || 1 })}
                style={{ width: '100%', padding: 4, fontSize: 12 }}
              />
            </div>
            <div style={{ marginBottom: 6 }}>
              <label style={{ display: 'block', fontSize: 11, opacity: 0.7 }}>Compress after (days)</label>
              <input
                type="number" min={1}
                value={cleanupCfg.compress_after_days}
                onChange={e => setCleanupCfg({ ...cleanupCfg, compress_after_days: parseInt(e.target.value) || 1 })}
                style={{ width: '100%', padding: 4, fontSize: 12 }}
              />
            </div>
            <div style={{ marginBottom: 8 }}>
              <label style={{ display: 'block', fontSize: 11, opacity: 0.7 }}>Schedule (HH:MM)</label>
              <div style={{ display: 'flex', gap: 4 }}>
                <input
                  type="number" min={0} max={23}
                  value={cleanupCfg.schedule_hour}
                  onChange={e => setCleanupCfg({ ...cleanupCfg, schedule_hour: parseInt(e.target.value) || 0 })}
                  style={{ width: '50%', padding: 4, fontSize: 12 }}
                />
                <input
                  type="number" min={0} max={59}
                  value={cleanupCfg.schedule_minute}
                  onChange={e => setCleanupCfg({ ...cleanupCfg, schedule_minute: parseInt(e.target.value) || 0 })}
                  style={{ width: '50%', padding: 4, fontSize: 12 }}
                />
              </div>
            </div>
            <button
              type="button"
              onClick={saveCleanupCfg}
              disabled={cfgSaving}
              style={{
                width: '100%', padding: '6px 8px', fontSize: 12,
                background: 'var(--color-primary, #3b82f6)', color: '#fff',
                border: 'none', borderRadius: 4, cursor: 'pointer',
                opacity: cfgSaving ? 0.6 : 1,
              }}
            >
              {cfgSaving ? 'Saving…' : 'Save'}
            </button>
          </div>
        )}
      </div>

      {/* Middle: Dates */}
      <div style={{
        width: 200, flexShrink: 0,
        borderRight: '1px solid var(--color-border, #2a2f3a)',
        display: 'flex', flexDirection: 'column',
      }}>
        <div style={{
          padding: '12px 16px', borderBottom: '1px solid var(--color-border, #2a2f3a)',
          fontSize: 11, opacity: 0.6, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span>DATES</span>
          {selectedCategory && dates.length > 0 && (
            <button
              type="button"
              onClick={deleteCategory}
              title="Clear all files in this category"
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#ef4444', fontSize: 11 }}
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
              style={{
                display: 'block', width: '100%', textAlign: 'left',
                padding: '8px 16px', border: 'none',
                background: selectedDate === d.date ? 'rgba(59,130,246,.15)' : 'transparent',
                color: 'var(--color-text, #e5e7eb)',
                cursor: 'pointer', fontSize: 12,
                borderLeft: selectedDate === d.date
                  ? '3px solid var(--color-primary, #3b82f6)' : '3px solid transparent',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>{d.date}</span>
                {d.compressed && (
                  <span style={{ fontSize: 9, opacity: 0.6, padding: '1px 4px', border: '1px solid currentColor', borderRadius: 2 }}>
                    GZ
                  </span>
                )}
                {d.date === todayStr() && (
                  <span style={{ fontSize: 9, color: '#22c55e', marginLeft: 'auto' }}>● LIVE</span>
                )}
              </div>
              <div style={{ fontSize: 10, opacity: 0.6, marginTop: 2 }}>
                {formatBytes(d.size)}
              </div>
            </button>
          ))}
          {dates.length === 0 && selectedCategory && (
            <div style={{ padding: 16, fontSize: 12, opacity: 0.5 }}>No log files</div>
          )}
        </div>
      </div>

      {/* Viewer */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* Toolbar */}
        <div style={{
          display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap',
          padding: '8px 16px', borderBottom: '1px solid var(--color-border, #2a2f3a)',
          fontSize: 12,
        }}>
          {selectedDate && selectedDate === todayStr() && (
            <button
              type="button"
              onClick={liveOn ? stopLive : startLive}
              style={{
                padding: '6px 12px',
                background: liveOn ? '#ef4444' : '#22c55e',
                color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer',
                fontSize: 12, fontWeight: 500,
              }}
            >
              {liveOn ? '■ Stop' : '▶ Live'}
            </button>
          )}
          <input
            type="text"
            placeholder="Search..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{ padding: 6, fontSize: 12, minWidth: 180, flex: 1, maxWidth: 300 }}
          />
          <select
            value={levelFilter}
            onChange={e => setLevelFilter(e.target.value as LevelFilter)}
            style={{ padding: 6, fontSize: 12 }}
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
            style={{ padding: 6, fontSize: 12 }}
          >
            <option value={200}>Last 200</option>
            <option value={500}>Last 500</option>
            <option value={2000}>Last 2000</option>
            <option value={10000}>Last 10000</option>
          </select>
          <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
            <input
              type="checkbox" checked={autoScroll}
              onChange={e => setAutoScroll(e.target.checked)}
            />
            <span>Auto-scroll</span>
          </label>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
            {selectedDate && (
              <>
                <button type="button" onClick={downloadFile} style={{ padding: '6px 10px', fontSize: 12 }}>
                  ⬇ Download
                </button>
                <button
                  type="button"
                  onClick={deleteFile}
                  style={{ padding: '6px 10px', fontSize: 12, color: '#ef4444' }}
                >
                  🗑 Delete
                </button>
              </>
            )}
          </div>
        </div>

        {/* Lines */}
        <div
          ref={viewerRef}
          style={{
            flex: 1, overflow: 'auto', padding: '8px 12px',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
            fontSize: 12, lineHeight: 1.5,
            background: 'var(--color-bg-secondary, #0f1117)',
          }}
        >
          {loadingFile && <div style={{ opacity: 0.6 }}>Loading…</div>}
          {!loadingFile && filteredLines.length === 0 && (
            <div style={{ opacity: 0.5 }}>
              {selectedDate ? 'No matching lines' : 'Select a category and date'}
            </div>
          )}
          {!loadingFile && filteredLines.map((ln, i) => {
            const lvl = lineLevel(ln);
            return (
              <div
                key={i}
                style={{
                  whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                  color: levelColor(lvl),
                }}
              >
                {ln}
              </div>
            );
          })}
        </div>

        {/* Status bar */}
        <div style={{
          padding: '4px 12px', fontSize: 11, opacity: 0.6,
          borderTop: '1px solid var(--color-border, #2a2f3a)',
          display: 'flex', gap: 12,
        }}>
          <span>{filteredLines.length}/{lines.length} lines</span>
          {liveOn && <span style={{ color: '#22c55e' }}>● Live tailing</span>}
          {selectedCategory && selectedDate && (
            <span style={{ marginLeft: 'auto' }}>
              {selectedCategory}/{selectedDate}.log
            </span>
          )}
        </div>
      </div>

      {confirm && (
        <ConfirmDialog
          isOpen={true}
          title={confirm.title}
          message={confirm.message}
          type="danger"
          onConfirm={() => { confirm.onConfirm(); setConfirm(null); }}
          onClose={() => setConfirm(null)}
        />
      )}
    </div>
  );
};

export default SystemLogs;
