import React, { useEffect, useState } from 'react';
import { useToast } from '@/contexts/ToastContext';
import { StorageItem, StorageStats } from '@/types/storage';
import { API_BASE_URL } from '@/config/api';
import ConfirmDialog from '@/components/shared/ConfirmDialog';

const StorageManager: React.FC = () => {
  const toast = useToast();
  const [storageTree, setStorageTree] = useState<StorageItem | null>(null);
  const [stats, setStats] = useState<StorageStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [confirmDialog, setConfirmDialog] = useState<{
    isOpen: boolean;
    item: StorageItem | null;
  }>({ isOpen: false, item: null });

  useEffect(() => {
    fetchStorageTree();
    fetchStats();
  }, []);

  const fetchStorageTree = async (path: string = '', maxDepth: number = 3) => {
    try {
      setIsLoading(true);
      const token = localStorage.getItem('access_token');
      const response = await fetch(
        `${API_BASE_URL}/api/storage/tree?path=${encodeURIComponent(path)}&max_depth=${maxDepth}`,
        {
          headers: { 'Authorization': `Bearer ${token}` }
        }
      );

      if (response.ok) {
        const data = await response.json();
        setStorageTree(data);
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
      const token = localStorage.getItem('access_token');
      const response = await fetch(
        `${API_BASE_URL}/api/storage/stats?path=${encodeURIComponent(path)}`,
        {
          headers: { 'Authorization': `Bearer ${token}` }
        }
      );

      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Error fetching storage stats:', error);
    }
  };

  const toggleFolder = (path: string) => {
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const handleDeleteClick = (item: StorageItem) => {
    setConfirmDialog({ isOpen: true, item });
  };

  const handleConfirmDelete = async () => {
    const item = confirmDialog.item;
    if (!item) return;

    // TODO: Implement delete functionality
    // Temporarily disabled - just show message
    toast.info(`Delete functionality temporarily disabled. Would delete: ${item.name} (${formatSize(item.size_mb)})`);
    console.log('Delete requested for:', item.path, item.file_count, 'files');

    /* COMMENTED OUT - Delete implementation
    try {
      setDeletingPath(item.path);
      const token = localStorage.getItem('access_token');
      const response = await fetch(
        `${API_BASE_URL}/api/storage/folder?path=${encodeURIComponent(item.path)}`,
        {
          method: 'DELETE',
          headers: { 'Authorization': `Bearer ${token}` }
        }
      );

      if (response.ok) {
        toast.success(`Deleted ${item.name} successfully`);
        await fetchStorageTree();
        await fetchStats();
      } else {
        const error = await response.json();
        toast.error(error.detail || 'Failed to delete folder');
      }
    } catch (error) {
      console.error('Error deleting folder:', error);
      toast.error('Failed to delete folder');
    } finally {
      setDeletingPath(null);
    }
    */
  };

  const formatSize = (mb: number): string => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(2)} GB`;
    }
    return `${mb.toFixed(2)} MB`;
  };

  const renderFolderTree = (item: StorageItem, level: number = 0) => {
    const isExpanded = expandedFolders.has(item.path);
    const hasChildren = item.children && item.children.length > 0;
    const canExpand = item.children !== null;
    const isDeleting = false; // Delete feature is disabled

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
                    <path d="M8 10.5l-4-4h8l-4 4z"/>
                  </svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M6 4l4 4-4 4V4z"/>
                  </svg>
                )}
              </button>
            )}
            {!canExpand && (
              <span className="no-expand">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="5" width="10" height="8" rx="1"/>
                  <path d="M5 5V3a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
                </svg>
              </span>
            )}

            <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" className="folder-icon">
              <path d="M2 4a2 2 0 0 1 2-2h4.586a1 1 0 0 1 .707.293l1.414 1.414A1 1 0 0 0 11.414 4H16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V4z"/>
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
                {isDeleting ? (
                  '...'
                ) : (
                  <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 4h12M5 4V3a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v1M14 4v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4h10z"/>
                    <line x1="7" y1="7" x2="7" y2="12"/>
                    <line x1="11" y1="7" x2="11" y2="12"/>
                  </svg>
                )}
              </button>
            )}
          </div>
        </div>

        {isExpanded && hasChildren && (
          <div className="folder-children">
            {item.children!.map(child => renderFolderTree(child, level + 1))}
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

  return (
    <div className="storage-manager">
      {/* Header */}
      <div className="storage-header">
        <div>
          <h2>Storage Management</h2>
          <p>Manage inference result files and folders</p>
        </div>
        <button onClick={() => { fetchStorageTree(); fetchStats(); }} className="refresh-btn">
          🔄 Refresh
        </button>
      </div>

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
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" style={{ verticalAlign: 'middle', marginRight: '8px' }}>
              <path d="M1.5 3A1.5 1.5 0 0 1 3 1.5h3.293a1 1 0 0 1 .707.293l1.414 1.414A1 1 0 0 0 9.121 3.5H13A1.5 1.5 0 0 1 14.5 5v8a1.5 1.5 0 0 1-1.5 1.5H3A1.5 1.5 0 0 1 1.5 13V3z"/>
            </svg>
            Folders are organized by recipe ID → date → timestamp
          </li>
          <li>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ verticalAlign: 'middle', marginRight: '8px' }}>
              <rect x="2" y="4" width="12" height="10" rx="1"/>
              <path d="M4 4V2.5a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1V4"/>
            </svg>
            Leaf folders contain captured image files
          </li>
          <li>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ verticalAlign: 'middle', marginRight: '8px' }}>
              <path d="M2 3h12M4 3V2a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v1M12 3v9a1.5 1.5 0 0 1-1.5 1.5h-5A1.5 1.5 0 0 1 4 12V3h8z"/>
              <line x1="6" y1="6" x2="6" y2="10"/>
              <line x1="10" y1="6" x2="10" y2="10"/>
            </svg>
            Click delete button to remove a folder (currently disabled for safety)
          </li>
          <li>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ verticalAlign: 'middle', marginRight: '8px' }}>
              <circle cx="8" cy="8" r="6"/>
              <path d="M8 5v3l2 2"/>
            </svg>
            Click refresh to update the folder tree
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
        message={confirmDialog.item ?
          `Are you sure you want to delete "${confirmDialog.item.name}"?\n\nThis will delete ${confirmDialog.item.file_count} files (${formatSize(confirmDialog.item.size_mb)}) and cannot be undone!` :
          ''
        }
        confirmText="Delete"
        cancelText="Cancel"
      />
    </div>
  );
};

export default StorageManager;
