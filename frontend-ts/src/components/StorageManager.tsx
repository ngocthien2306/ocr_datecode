import React, { useEffect, useState } from 'react';
import { useToast } from '@/contexts/ToastContext';
import {
  CleanupConfig,
  CleanupRunResult,
  DiskUsage,
  StorageItem,
  StorageStats,
} from '@/types/storage';
import { API_BASE_URL } from '@/config/api';
import ConfirmDialog from '@/components/shared/ConfirmDialog';

const DEFAULT_CONFIG: CleanupConfig = {
  enabled: false,
  dry_run: true,
  schedule_mode: 'daily',
  schedule_hour: 2,
  schedule_minute: 0,
  interval_hours: 6,
  trigger_usage_percent: 70,
  target_usage_percent: 60,
  min_images_per_folder: 25,
  keep_recent_dates_per_recipe: 3,
};

const authHeaders = (): Record<string, string> => {
  const token = localStorage.getItem('access_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
};

const StorageManager: React.FC = () => {
  const toast = useToast();
  const [storageTree, setStorageTree] = useState<StorageItem | null>(null);
  const [stats, setStats] = useState<StorageStats | null>(null);
  const [diskUsage, setDiskUsage] = useState<DiskUsage | null>(null);
  const [cleanupConfig, setCleanupConfig] = useState<CleanupConfig>(DEFAULT_CONFIG);
  const [lastRun, setLastRun] = useState<CleanupRunResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSavingConfig, setIsSavingConfig] = useState(false);
  const [isRunningCleanup, setIsRunningCleanup] = useState(false);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [deletingPath, setDeletingPath] = useState<string | null>(null);
  const [confirmDialog, setConfirmDialog] = useState<{
    isOpen: boolean;
    item: StorageItem | null;
  }>({ isOpen: false, item: null });

  useEffect(() => {
    refreshAll();
  }, []);

  const refreshAll = async () => {
    await Promise.all([
      fetchStorageTree(),
      fetchStats(),
      fetchDiskUsage(),
      fetchCleanupConfig(),
      fetchLastRun(),
    ]);
  };

  const fetchStorageTree = async (path: string = '', maxDepth: number = 3) => {
    try {
      setIsLoading(true);
      const response = await fetch(
        `${API_BASE_URL}/api/storage/tree?path=${encodeURIComponent(path)}&max_depth=${maxDepth}`,
        { headers: authHeaders() },
      );
      if (response.ok) {
        setStorageTree(await response.json());
      } else {
        toast.error('Failed to load storage tree');
      }
    } catch (error) {
      console.error('Error fetching storage tree:', error);
      toast.error('Failed to load storage tree');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchStats = async (path: string = '') => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/storage/stats?path=${encodeURIComponent(path)}`,
        { headers: authHeaders() },
      );
      if (response.ok) setStats(await response.json());
    } catch (error) {
      console.error('Error fetching storage stats:', error);
    }
  };

  const fetchDiskUsage = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/storage/disk-usage`, {
        headers: authHeaders(),
      });
      if (response.ok) setDiskUsage(await response.json());
    } catch (error) {
      console.error('Error fetching disk usage:', error);
    }
  };

  const fetchCleanupConfig = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/storage/cleanup-config`, {
        headers: authHeaders(),
      });
      if (response.ok) setCleanupConfig(await response.json());
    } catch (error) {
      console.error('Error fetching cleanup config:', error);
    }
  };

  const fetchLastRun = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/storage/cleanup/last-run`, {
        headers: authHeaders(),
      });
      if (response.ok) {
        const data = await response.json();
        setLastRun(data && data.ran_at ? data : null);
      }
    } catch (error) {
      console.error('Error fetching cleanup last-run:', error);
    }
  };

  const saveCleanupConfig = async () => {
    if (cleanupConfig.target_usage_percent >= cleanupConfig.trigger_usage_percent) {
      toast.error('Target % must be lower than Trigger %');
      return;
    }
    try {
      setIsSavingConfig(true);
      const response = await fetch(`${API_BASE_URL}/api/storage/cleanup-config`, {
        method: 'PUT',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify(cleanupConfig),
      });
      if (response.ok) {
        setCleanupConfig(await response.json());
        toast.success('Auto cleanup settings saved');
      } else {
        const err = await response.json().catch(() => ({}));
        toast.error(err.detail || 'Failed to save settings');
      }
    } catch (error) {
      console.error('Error saving cleanup config:', error);
      toast.error('Failed to save settings');
    } finally {
      setIsSavingConfig(false);
    }
  };

  const runCleanupNow = async () => {
    try {
      setIsRunningCleanup(true);
      const response = await fetch(`${API_BASE_URL}/api/storage/cleanup/run-now`, {
        method: 'POST',
        headers: authHeaders(),
      });
      if (response.ok) {
        const result: CleanupRunResult = await response.json();
        setLastRun(result);
        if (result.trigger === 'threshold-skip') {
          toast.info(result.message);
        } else {
          toast.success(result.message);
        }
        await Promise.all([fetchDiskUsage(), fetchStorageTree(), fetchStats()]);
      } else {
        const err = await response.json().catch(() => ({}));
        toast.error(err.detail || 'Cleanup failed');
      }
    } catch (error) {
      console.error('Error running cleanup:', error);
      toast.error('Cleanup failed');
    } finally {
      setIsRunningCleanup(false);
    }
  };

  const toggleFolder = (path: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const handleDeleteClick = (item: StorageItem) => {
    setConfirmDialog({ isOpen: true, item });
  };

  const handleConfirmDelete = async () => {
    const item = confirmDialog.item;
    if (!item) return;
    try {
      setDeletingPath(item.path);
      const response = await fetch(
        `${API_BASE_URL}/api/storage/folder?path=${encodeURIComponent(item.path)}`,
        { method: 'DELETE', headers: authHeaders() },
      );
      if (response.ok) {
        toast.success(`Deleted ${item.name}`);
        await Promise.all([fetchStorageTree(), fetchStats(), fetchDiskUsage()]);
      } else {
        const err = await response.json().catch(() => ({}));
        toast.error(err.detail || 'Failed to delete folder');
      }
    } catch (error) {
      console.error('Error deleting folder:', error);
      toast.error('Failed to delete folder');
    } finally {
      setDeletingPath(null);
      setConfirmDialog({ isOpen: false, item: null });
    }
  };

  const formatSize = (mb: number): string => {
    if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
    return `${mb.toFixed(2)} MB`;
  };

  const usageLevel = (pct: number): 'low' | 'mid' | 'high' => {
    if (pct >= 85) return 'high';
    if (pct >= 70) return 'mid';
    return 'low';
  };

  const formatRanAt = (iso: string): string => {
    try {
      return new Date(iso).toLocaleString();
    } catch {
      return iso;
    }
  };

  const renderFolderTree = (item: StorageItem, level: number = 0) => {
    const isExpanded = expandedFolders.has(item.path);
    const hasChildren = item.children && item.children.length > 0;
    const canExpand = item.children !== null;
    const isDeleting = deletingPath === item.path;

    return (
      <div key={item.path} className="storage-folder-item" style={{ marginLeft: `${level * 20}px` }}>
        <div className="folder-row">
          <div className="folder-info">
            {canExpand && (
              <button
                className="expand-btn"
                onClick={() => toggleFolder(item.path)}
                disabled={isDeleting}
              >
                {isExpanded ? (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M8 10.5l-4-4h8l-4 4z" />
                  </svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M6 4l4 4-4 4V4z" />
                  </svg>
                )}
              </button>
            )}
            {!canExpand && (
              <span className="no-expand">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="5" width="10" height="8" rx="1" />
                  <path d="M5 5V3a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
                </svg>
              </span>
            )}

            <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" className="folder-icon">
              <path d="M2 4a2 2 0 0 1 2-2h4.586a1 1 0 0 1 .707.293l1.414 1.414A1 1 0 0 0 11.414 4H16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V4z" />
            </svg>
            <span className="folder-name">{item.name}</span>
            <span className="folder-size">{formatSize(item.size_mb)}</span>
            {item.file_count !== undefined && (
              <span className="file-count">{item.file_count} files</span>
            )}
          </div>

          <div className="folder-actions">
            {level > 0 && (
              <button
                className="delete-btn"
                onClick={() => handleDeleteClick(item)}
                disabled={isDeleting}
                title="Delete folder"
              >
                {isDeleting ? '...' : (
                  <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 4h12M5 4V3a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v1M14 4v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4h10z" />
                    <line x1="7" y1="7" x2="7" y2="12" />
                    <line x1="11" y1="7" x2="11" y2="12" />
                  </svg>
                )}
              </button>
            )}
          </div>
        </div>

        {isExpanded && hasChildren && (
          <div className="folder-children">
            {item.children!.map((child) => renderFolderTree(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="storage-manager">
        <div className="storage-loading">
          <div className="spinner" />
          <p>Loading storage information...</p>
        </div>
      </div>
    );
  }

  const usedPct = diskUsage?.used_percent ?? 0;
  const level = usageLevel(usedPct);

  return (
    <div className="storage-manager">
      {/* Header */}
      <div className="storage-header">
        <div>
          <h2>Storage Management</h2>
          <p>Manage inference result files and auto-cleanup</p>
        </div>
        <button onClick={refreshAll} className="refresh-btn">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9" />
            <path d="M13.5 2.5V5.5H10.5" />
          </svg>
          <span>Refresh</span>
        </button>
      </div>

      {/* Disk Usage Card */}
      {diskUsage && (
        <div className={`storage-disk-card disk-level-${level}`}>
          <div className="disk-header">
            <h3>
              <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <ellipse cx="10" cy="5" rx="7" ry="2.5" />
                <path d="M3 5v5c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5V5" />
                <path d="M3 10v5c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-5" />
              </svg>
              <span>Disk Usage</span>
            </h3>
            <span className="disk-mount" title={diskUsage.mount_point}>
              {diskUsage.mount_point}
            </span>
          </div>
          <div className="disk-bar-wrapper">
            <div className="disk-bar-fill" style={{ width: `${Math.min(100, usedPct)}%` }} />
          </div>
          <div className="disk-stats">
            <span>
              <strong>{diskUsage.used_gb.toFixed(1)} GB</strong> used of {diskUsage.total_gb.toFixed(1)} GB
            </span>
            <span className="disk-stats-percent">{usedPct.toFixed(1)}%</span>
            <span>Free: {diskUsage.free_gb.toFixed(1)} GB</span>
          </div>
        </div>
      )}

      {/* Auto Cleanup Card */}
      <div className="storage-cleanup-card">
        <div className="cleanup-header">
          <h3>
            <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M3 5h14M8 5V3a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v2" />
              <path d="M5 5l1 11.5A2 2 0 0 0 8 18h4a2 2 0 0 0 2-1.5L15 5" />
              <path d="M8 9v6M12 9v6" />
            </svg>
            <span>Auto Cleanup</span>
          </h3>
          <label className="cleanup-toggle">
            <input
              type="checkbox"
              checked={cleanupConfig.enabled}
              onChange={(e) =>
                setCleanupConfig({ ...cleanupConfig, enabled: e.target.checked })
              }
            />
            <span>Enable</span>
          </label>
        </div>

        <div className="cleanup-grid">
          {/* Schedule */}
          <div className="cleanup-field">
            <label>Schedule</label>
            <div className="cleanup-radio-row">
              <label>
                <input
                  type="radio"
                  name="schedule_mode"
                  checked={cleanupConfig.schedule_mode === 'daily'}
                  onChange={() =>
                    setCleanupConfig({ ...cleanupConfig, schedule_mode: 'daily' })
                  }
                />
                Daily
              </label>
              <label>
                <input
                  type="radio"
                  name="schedule_mode"
                  checked={cleanupConfig.schedule_mode === 'hourly'}
                  onChange={() =>
                    setCleanupConfig({ ...cleanupConfig, schedule_mode: 'hourly' })
                  }
                />
                Hourly
              </label>
            </div>
            {cleanupConfig.schedule_mode === 'daily' ? (
              <input
                type="time"
                className="cleanup-input"
                value={`${String(cleanupConfig.schedule_hour).padStart(2, '0')}:${String(
                  cleanupConfig.schedule_minute,
                ).padStart(2, '0')}`}
                onChange={(e) => {
                  const parts = e.target.value.split(':').map(Number);
                  const h = parts[0] ?? 0;
                  const m = parts[1] ?? 0;
                  setCleanupConfig({ ...cleanupConfig, schedule_hour: h, schedule_minute: m });
                }}
              />
            ) : (
              <div className="cleanup-input-row">
                <span>Every</span>
                <input
                  type="number"
                  className="cleanup-input cleanup-input-narrow"
                  min={1}
                  max={24}
                  value={cleanupConfig.interval_hours}
                  onChange={(e) =>
                    setCleanupConfig({
                      ...cleanupConfig,
                      interval_hours: Number(e.target.value),
                    })
                  }
                />
                <span>hour(s)</span>
              </div>
            )}
          </div>

          {/* Trigger threshold */}
          <div className="cleanup-field">
            <label>
              Trigger when disk usage &gt;{' '}
              <strong>{cleanupConfig.trigger_usage_percent}%</strong>
            </label>
            <input
              type="range"
              min={50}
              max={95}
              step={1}
              value={cleanupConfig.trigger_usage_percent}
              onChange={(e) =>
                setCleanupConfig({
                  ...cleanupConfig,
                  trigger_usage_percent: Number(e.target.value),
                })
              }
            />
          </div>

          {/* Target threshold */}
          <div className="cleanup-field">
            <label>
              Stop when disk usage ≤{' '}
              <strong>{cleanupConfig.target_usage_percent}%</strong>
            </label>
            <input
              type="range"
              min={30}
              max={90}
              step={1}
              value={cleanupConfig.target_usage_percent}
              onChange={(e) =>
                setCleanupConfig({
                  ...cleanupConfig,
                  target_usage_percent: Number(e.target.value),
                })
              }
            />
          </div>

          {/* Min images per folder */}
          <div className="cleanup-field">
            <label>
              Min images per folder:{' '}
              <strong>{cleanupConfig.min_images_per_folder}</strong>
            </label>
            <input
              type="range"
              min={20}
              max={30}
              step={1}
              value={cleanupConfig.min_images_per_folder}
              onChange={(e) =>
                setCleanupConfig({
                  ...cleanupConfig,
                  min_images_per_folder: Number(e.target.value),
                })
              }
            />
          </div>

          {/* Keep recent dates */}
          <div className="cleanup-field">
            <label>
              Keep recent date-folders per recipe:{' '}
              <strong>{cleanupConfig.keep_recent_dates_per_recipe}</strong>
            </label>
            <input
              type="range"
              min={1}
              max={30}
              step={1}
              value={cleanupConfig.keep_recent_dates_per_recipe}
              onChange={(e) =>
                setCleanupConfig({
                  ...cleanupConfig,
                  keep_recent_dates_per_recipe: Number(e.target.value),
                })
              }
            />
          </div>

          {/* Dry run */}
          <div className="cleanup-field">
            <label className="cleanup-checkbox">
              <input
                type="checkbox"
                checked={cleanupConfig.dry_run}
                onChange={(e) =>
                  setCleanupConfig({ ...cleanupConfig, dry_run: e.target.checked })
                }
              />
              <span>
                Dry run (simulate — don&apos;t actually delete files)
              </span>
            </label>
          </div>
        </div>

        <div className="cleanup-actions">
          <button
            className="cleanup-save-btn"
            onClick={saveCleanupConfig}
            disabled={isSavingConfig}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M11 2H3a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V5l-3-3z" />
              <path d="M5 2v4h5V2M5 10h6" />
            </svg>
            <span>{isSavingConfig ? 'Saving…' : 'Save Settings'}</span>
          </button>
          <button
            className="cleanup-run-btn"
            onClick={runCleanupNow}
            disabled={isRunningCleanup}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M4 3l9 5-9 5V3z" />
            </svg>
            <span>{isRunningCleanup ? 'Running…' : 'Run Cleanup Now'}</span>
          </button>
        </div>
      </div>

      {/* Last Run Card */}
      {lastRun && (
        <div className="storage-lastrun-card">
          <div className="lastrun-header">
            <h3>
              <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="10" cy="10" r="7.5" />
                <path d="M10 5.5V10l3 2" />
              </svg>
              <span>Last Cleanup Run</span>
            </h3>
            <div className="lastrun-badges">
              <span className={`lastrun-badge lastrun-trigger-${lastRun.trigger}`}>
                {lastRun.trigger}
              </span>
              {lastRun.dry_run && <span className="lastrun-badge lastrun-dry">dry-run</span>}
            </div>
          </div>
          <div className="lastrun-stats">
            <div>
              <span className="lastrun-label">When</span>
              <span className="lastrun-value">{formatRanAt(lastRun.ran_at)}</span>
            </div>
            <div>
              <span className="lastrun-label">Files deleted</span>
              <span className="lastrun-value">{lastRun.files_deleted}</span>
            </div>
            <div>
              <span className="lastrun-label">Folders deleted</span>
              <span className="lastrun-value">{lastRun.folders_deleted}</span>
            </div>
            <div>
              <span className="lastrun-label">Freed</span>
              <span className="lastrun-value">{lastRun.freed_mb.toFixed(1)} MB</span>
            </div>
            <div>
              <span className="lastrun-label">Disk</span>
              <span className="lastrun-value">
                {lastRun.disk_used_percent_before.toFixed(1)}% →{' '}
                {lastRun.disk_used_percent_after.toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="lastrun-label">Duration</span>
              <span className="lastrun-value">{lastRun.duration_seconds}s</span>
            </div>
          </div>
          <p className="lastrun-msg">{lastRun.message}</p>
        </div>
      )}

      {/* Stats Card */}
      {stats && (
        <div className="storage-stats-card">
          <div className="stat-item">
            <span className="stat-label">Total Size</span>
            <span className="stat-value">{stats.total_size_gb.toFixed(2)} GB</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Total Files</span>
            <span className="stat-value">{stats.total_files.toLocaleString()}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Directories</span>
            <span className="stat-value">{stats.total_directories.toLocaleString()}</span>
          </div>
        </div>
      )}

      {/* Folder Tree */}
      <div className="storage-tree-card">
        <h3>Folder Structure</h3>
        <div className="storage-tree">
          {storageTree ? renderFolderTree(storageTree) : <p>No data available</p>}
        </div>
      </div>

      {/* Info */}
      <div className="storage-info-card">
        <h3>Information</h3>
        <ul>
          <li>
            <svg className="info-icon" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M1.5 4A1.5 1.5 0 0 1 3 2.5h3.3a1 1 0 0 1 .7.3l1.4 1.4a1 1 0 0 0 .7.3H13A1.5 1.5 0 0 1 14.5 6v7A1.5 1.5 0 0 1 13 14.5H3A1.5 1.5 0 0 1 1.5 13V4z" />
            </svg>
            <span>Folders are organized by recipe ID → date → timestamp</span>
          </li>
          <li>
            <svg className="info-icon" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <rect x="2" y="3" width="12" height="10" rx="1.5" />
              <circle cx="6" cy="7" r="1.3" />
              <path d="M14 11l-3.5-3.5L4 13" />
            </svg>
            <span>Leaf folders contain captured image files</span>
          </li>
          <li>
            <svg className="info-icon" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M14 8a6 6 0 1 1-1.8-4.3" />
              <path d="M14 2.5V6h-3.5" />
            </svg>
            <span>
              Auto cleanup keeps a random sample of N newest images per leaf folder
              (20–30) and preserves the most recent date-folders per recipe
            </span>
          </li>
          <li>
            <svg className="info-icon" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M8 1.5l6.5 11.5h-13L8 1.5z" />
              <path d="M8 6.5v3M8 11.5v.1" />
            </svg>
            <span>Cleanup only runs when disk usage exceeds the trigger threshold</span>
          </li>
          <li>
            <svg className="info-icon" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <circle cx="8" cy="8" r="6.5" />
              <path d="M8 4.5v3.8M8 11v.1" />
            </svg>
            <span>Use Dry-run mode to preview what would be deleted</span>
          </li>
        </ul>
      </div>

      {/* Confirm Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={() => setConfirmDialog({ isOpen: false, item: null })}
        onConfirm={handleConfirmDelete}
        type="danger"
        title="Delete Folder"
        message={
          confirmDialog.item
            ? `Are you sure you want to delete "${confirmDialog.item.name}"?\n\nThis will delete ${confirmDialog.item.file_count} files (${formatSize(confirmDialog.item.size_mb)}) and cannot be undone!`
            : ''
        }
        confirmText="Delete"
        cancelText="Cancel"
      />
    </div>
  );
};

export default StorageManager;
