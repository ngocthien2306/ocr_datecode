import { useState, useEffect, useMemo } from 'react';
import { receiptsAPI } from '@/services/api';
import type { RecipeSortBy } from '@/services/recipes';
import RecipeFormModal from './RecipeFormModal';
import RecipeViewModal from './RecipeViewModal';
import ConfirmDialog from '@/components/shared/ConfirmDialog';
import RecipeLoadingTemplates from '@/components/shared/RecipeLoadingTemplates';
import { useToast } from '@/contexts/ToastContext';
import { useUser } from '@/contexts/UserContext';
import type { Receipt } from '@/types';
import '@/styles/HistoryView.css';
import '@/styles/RecipeLoadingAnimations.css';
import { API_BASE_URL } from '@/config/api';

interface ConfirmDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  onConfirm: (() => void) | null;
}

interface Statistics {
  totalReceipts: number;
  totalProducts: number;
  successRate: number;
}

type ViewMode = 'list' | 'create' | 'edit' | 'view' | 'history';

export default function Receipts() {
  const toast = useToast();
  const { canPerformAction } = useUser();
  const [receipts, setReceipts] = useState<Receipt[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statistics, setStatistics] = useState<Statistics>({
    totalReceipts: 0,
    totalProducts: 0,
    successRate: 0
  });

  const [searchTerm, setSearchTerm] = useState('');
  const [activeSearch, setActiveSearch] = useState('');   // committed search query — drives effect-based reload
  const [sortBy, setSortBy] = useState<RecipeSortBy>('created_desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const itemsPerPage = 10;

  // View mode state
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [modalMode, setModalMode] = useState<'create' | 'edit'>('create');
  const [selectedRecipe, setSelectedRecipe] = useState<Receipt | null>(null);

  // Selection states
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [selectAll, setSelectAll] = useState(false);

  // Confirmation dialog states
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning',
    onConfirm: null
  });
  
  // Load history state
  const [historyItems, setHistoryItems] = useState<any[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyCount, setHistoryCount] = useState(0);
  const [rawOpenMap, setRawOpenMap] = useState<Record<string, boolean>>({});

  // Loading animation state
  const [showLoadingAnimation, setShowLoadingAnimation] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingTemplate, setLoadingTemplate] = useState<'ocr-scanner' | 'camera-vision' | 'barcode-scanner' | 'neural-network' | 'qr-detector' | 'industrial-factory'>('ocr-scanner');
  const [overlayMode, setOverlayMode] = useState<'fullscreen' | 'dashboard-main'>('fullscreen');
  const [isLightMode] = useState(() => {
    const savedMode = localStorage.getItem('appThemeMode');
    return savedMode ? savedMode === 'light' : true;
  });

  // Running recipe state
  const [runningRecipeId, setRunningRecipeId] = useState<string | null>(null);
  

  // Load receipts from API — re-runs when page, sort, or committed search changes
  useEffect(() => {
    loadReceipts();
  }, [currentPage, sortBy, activeSearch]);

  // Statistics + running recipe — once per mount
  useEffect(() => {
    loadStatistics();
    checkRunningRecipe();
  }, []);

  // Check for running recipe
  const checkRunningRecipe = async () => {
    try {
      const latestLoad = await receiptsAPI.getLatestLoadedRecipe();
      if (latestLoad && latestLoad.recipe_id) {
        setRunningRecipeId(latestLoad.recipe_id);
      } else {
        setRunningRecipeId(null);
      }
    } catch (error) {
      console.error('Error checking running recipe:', error);
      setRunningRecipeId(null);
    }
  };

  // Load recipe template and overlay mode from settings
  useEffect(() => {
    const savedTemplate = localStorage.getItem('recipeLoadTemplate');
    if (savedTemplate) {
      setLoadingTemplate(savedTemplate as any);
    }

    const savedOverlayMode = localStorage.getItem('recipeLoadOverlayMode');
    console.log('Receipts - Loading overlay mode from localStorage:', savedOverlayMode);
    if (savedOverlayMode) {
      setOverlayMode(savedOverlayMode as 'fullscreen' | 'dashboard-main');
    }

    const handleTemplateChange = (event: CustomEvent) => {
      setLoadingTemplate(event.detail.template as any);
    };

    const handleOverlayModeChange = (event: CustomEvent) => {
      console.log('Receipts - Overlay mode changed via event:', event.detail.mode);
      setOverlayMode(event.detail.mode as 'fullscreen' | 'dashboard-main');
    };

    window.addEventListener('recipeLoadTemplateChanged', handleTemplateChange as EventListener);
    window.addEventListener('recipeLoadOverlayModeChanged', handleOverlayModeChange as EventListener);

    return () => {
      window.removeEventListener('recipeLoadTemplateChanged', handleTemplateChange as EventListener);
      window.removeEventListener('recipeLoadOverlayModeChanged', handleOverlayModeChange as EventListener);
    };
  }, []);

  const loadReceipts = async () => {
    try {
      setLoading(true);
      const skip = (currentPage - 1) * itemsPerPage;
      const trimmed = activeSearch.trim();
      const data = trimmed
        ? await receiptsAPI.searchReceipts(trimmed, skip, itemsPerPage, sortBy)
        : await receiptsAPI.getAllReceipts(skip, itemsPerPage, true, sortBy);

      const transformedReceipts = data.map(recipe => ({
        id: recipe.id,
        name: recipe.name,
        productCode: recipe.product_code,
        date: recipe.created_at ? new Date(recipe.created_at).toISOString().split('T')[0] : new Date().toISOString().split('T')[0],
        camera: `Camera Settings: ${recipe.camera_settings?.exposure_time || 'N/A'}ms`,
        products: 0, // This should come from production data
        passed: 0,
        failed: 0,
        operator: recipe.created_by_name || recipe.created_by || 'System',
        status: (recipe.is_active ? 'Active' : 'Inactive') as 'Active' | 'Inactive',
        description: recipe.description || '',
        // Include all recipe fields for editing
        cameras: recipe.cameras || [],
        camera_templates: recipe.camera_templates || [],
        delay_reject: recipe.delay_reject,
        reject_pulse: recipe.reject_pulse,
        reject_method: recipe.reject_method || 'DIO_OUT',
        do_reject_number: recipe.do_reject_number,
        do_alarm_number: recipe.do_alarm_number ?? 0,
        normal_pulse_ms: recipe.normal_pulse_ms ?? 250.0,
        cameraSettings: recipe.camera_settings,
        modelThresholds: recipe.model_thresholds,
        template_config: recipe.template_config,
        roi_config: recipe.roi_config,
        is_active: recipe.is_active,
        ocr_model_type: recipe.ocr_model_type || '',
        ml_project_id: recipe.ml_project_id || '',
        ml_model_id: recipe.ml_model_id || '',
        defect_model: recipe.defect_model || 'arcface',
        classifier_backend: recipe.classifier_backend || 'embedding',
        wrinkle_conf: recipe.wrinkle_conf ?? 0.25,
        createdAt: recipe.created_at || new Date().toISOString(),
        updatedAt: recipe.updated_at || new Date().toISOString()
      }));

      setReceipts(transformedReceipts);

      // Pagination total — only meaningful in list mode (BE search has no count endpoint).
      // For search results, treat what came back as the full set.
      if (trimmed) {
        // Approximate: if we got fewer than itemsPerPage we're on the last page;
        // otherwise allow next page to fetch (BE will return [] if past the end).
        setTotalPages(data.length < itemsPerPage ? currentPage : currentPage + 1);
      } else {
        const countData = await receiptsAPI.getReceiptsCount(true);
        setTotalPages(Math.max(1, Math.ceil(countData.count / itemsPerPage)));
      }

      setError(null);
    } catch (err) {
      console.error('Error loading receipts:', err);
      setError('Failed to load receipts. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const loadStatistics = async () => {
    try {
      const stats = await receiptsAPI.getStatistics();
      setStatistics(stats);
    } catch (err) {
      console.error('Error loading statistics:', err);
    }
  };

  // Commit the search input as the active query and reset to page 1.
  // The list-loading effect (which depends on `activeSearch` + `sortBy` +
  // `currentPage`) re-runs and fetches via the right endpoint.
  const handleSearch = () => {
    setCurrentPage(1);
    setActiveSearch(searchTerm);
  };

  const handleCreateReceipt = () => {
    setModalMode('create');
    setSelectedRecipe(null);
    setViewMode('create');
  };

  const handleEditReceipt = (receipt: Receipt) => {
    setModalMode('edit');
    setSelectedRecipe(receipt);
    setViewMode('edit');
  };

  const handleModalSubmit = async (formData: any) => {
    try {
      if (modalMode === 'create') {
        await receiptsAPI.createReceipt(formData);
        toast.success(`Recipe "${formData.name}" created successfully!`);
      } else {
        await receiptsAPI.updateReceipt(selectedRecipe!.id, formData);
        toast.success(`Recipe "${formData.name}" updated successfully!`);
      }

      // Reload receipts after create/update
      await loadReceipts();
      await loadStatistics();

      setViewMode('list');
    } catch (error: any) {
      console.error('Error saving recipe:', error);
      console.error('Error details:', error.response?.data);
      const errorMsg = error.response?.data?.detail || 'Failed to save recipe. Please try again.';
      toast.error(errorMsg);
      throw error; // Let the modal handle the error display
    }
  };

  const handleViewReceipt = (receipt: Receipt) => {
    setSelectedRecipe(receipt);
    setViewMode('view');
  };

  const handleViewModalEdit = () => {
    if (selectedRecipe) {
      handleEditReceipt(selectedRecipe);
    }
  };

  const handleBackToList = () => {
    setViewMode('list');
    setSelectedRecipe(null);
  };

  const handleCloneReceipt = async (receipt: Receipt) => {
    try {
      const clonedRecipe = await receiptsAPI.cloneReceipt(receipt.id);
      toast.success(`Recipe "${receipt.name}" cloned successfully as "${clonedRecipe.name}"!`);

      // Reload receipts to show the new cloned recipe
      await loadReceipts();
      await loadStatistics();

      // Transform the cloned recipe to Receipt type for editing
      const clonedReceipt: Receipt = {
        id: clonedRecipe.id,
        name: clonedRecipe.name,
        productCode: clonedRecipe.product_code,
        date: clonedRecipe.created_at ? new Date(clonedRecipe.created_at).toISOString().split('T')[0] : new Date().toISOString().split('T')[0],
        camera: `Camera Settings: ${clonedRecipe.camera_settings?.exposure_time || 'N/A'}ms`,
        products: 0,
        passed: 0,
        failed: 0,
        operator: clonedRecipe.created_by_name || clonedRecipe.created_by || 'System',
        status: (clonedRecipe.is_active ? 'Active' : 'Inactive') as 'Active' | 'Inactive',
        description: clonedRecipe.description || '',
        cameras: clonedRecipe.cameras || [],
        camera_templates: clonedRecipe.camera_templates || [],
        delay_reject: clonedRecipe.delay_reject,
        reject_pulse: clonedRecipe.reject_pulse,
        reject_method: clonedRecipe.reject_method || 'DIO_OUT',
        do_reject_number: clonedRecipe.do_reject_number,
        do_alarm_number: clonedRecipe.do_alarm_number ?? 0,
        normal_pulse_ms: clonedRecipe.normal_pulse_ms ?? 250.0,
        cameraSettings: clonedRecipe.camera_settings,
        modelThresholds: clonedRecipe.model_thresholds,
        template_config: clonedRecipe.template_config,
        roi_config: clonedRecipe.roi_config,
        is_active: clonedRecipe.is_active,
        ocr_model_type: clonedRecipe.ocr_model_type || '',
        ml_project_id: clonedRecipe.ml_project_id || '',
        ml_model_id: clonedRecipe.ml_model_id || '',
        defect_model: clonedRecipe.defect_model || 'arcface',
        classifier_backend: clonedRecipe.classifier_backend || 'embedding',
        createdAt: clonedRecipe.created_at || new Date().toISOString(),
        updatedAt: clonedRecipe.updated_at || new Date().toISOString()
      };

      // Open edit modal for the cloned recipe
      handleEditReceipt(clonedReceipt);
    } catch (error: any) {
      console.error('Error cloning receipt:', error);
      const errorMsg = error.response?.data?.detail || 'Failed to clone recipe. Please try again.';
      toast.error(errorMsg);
    }
  };

  const handleLoadReceipt = async (receipt: Receipt) => {
    // Check if another recipe is already running
    if (runningRecipeId && runningRecipeId !== receipt.id) {
      const runningRecipe = receipts.find(r => r.id === runningRecipeId);
      const runningName = runningRecipe ? `"${runningRecipe.name}"` : `(ID: ${runningRecipeId})`;
      setConfirmDialog({
        isOpen: true,
        title: 'Recipe Already Running',
        message: `Recipe ${runningName} is currently running.\n\nDo you want to stop it and load "${receipt.name}" (${receipt.productCode}) instead?`,
        type: 'warning',
        onConfirm: async () => {
          await doLoadRecipe(receipt);
        }
      });
      return;
    }

    setConfirmDialog({
      isOpen: true,
      title: 'Load Recipe',
      message: `Load recipe "${receipt.name}" (${receipt.productCode})?\n\nThis will load the recipe configuration into the system.`,
      type: 'info',
      onConfirm: async () => {
        await doLoadRecipe(receipt);
      }
    });
  };

  const doLoadRecipe = async (receipt: Receipt) => {
    try {
      const data = await receiptsAPI.loadReceipt(receipt.id);

      // Show loading animation with progress on success
      setShowLoadingAnimation(true);
      setLoadingProgress(0);

      // Animate progress from 0 to 100%
      const progressInterval = setInterval(() => {
        setLoadingProgress(prev => {
          if (prev >= 100) {
            clearInterval(progressInterval);
            return 100;
          }
          // Smooth acceleration
          const increment = prev < 30 ? 2 : prev < 70 ? 3 : 5;
          return Math.min(prev + increment, 100);
        });
      }, 50);

      setTimeout(async () => {
        clearInterval(progressInterval);
        setShowLoadingAnimation(false);
        setLoadingProgress(0);
        setRunningRecipeId(receipt.id); // Set running recipe

        // Set inference mode to OFFLINE by default when loading recipe
        try {
          await receiptsAPI.setInferenceMode(receipt.id, false);
          // Save OFFLINE state to localStorage for UI persistence
          localStorage.setItem('inference_mode_online', 'false');
          console.log('[LoadRecipe] Set inference mode to OFFLINE and saved to localStorage');
        } catch (error) {
          console.error('[LoadRecipe] Failed to set inference mode to OFFLINE:', error);
        }

        // Redirect to Realtime tab
        const realtimeLink = document.querySelector('a[href="#realtime"]') as HTMLAnchorElement;
        if (realtimeLink) {
          realtimeLink.click();
        }
      }, 3000);

      toast.success(`Recipe "${receipt.name}" loaded and recorded (event id: ${data.id}).`);
      await loadReceipts();
    } catch (error: any) {
      console.error('Error loading receipt:', error);
      const msg = error?.response?.data?.detail || 'Failed to load receipt. Please try again.';
      toast.error(msg);
    }
  };

  const handleStopReceipt = async (receipt: Receipt) => {
    setConfirmDialog({
      isOpen: true,
      title: 'Stop Recipe',
      message: `Stop running recipe "${receipt.name}" (${receipt.productCode})?\n\nThis will stop the recipe and set cameras to idle mode.`,
      type: 'warning',
      onConfirm: async () => {
        try {
          await receiptsAPI.stopReceipt(receipt.id);
          setRunningRecipeId(null); // Clear running recipe
          toast.success(`Recipe "${receipt.name}" stopped successfully!`);
          await checkRunningRecipe(); // Refresh status
        } catch (error: any) {
          console.error('Error stopping recipe:', error);
          const errorMsg = error.response?.data?.detail || 'Failed to stop recipe. Please try again.';
          toast.error(errorMsg);
        }
      }
    });
  };

  const handleDeleteReceipt = async (receipt: Receipt) => {
    if (receipt.id === runningRecipeId) {
      toast.warning(`Recipe "${receipt.name}" is currently running. Please stop it before deleting.`);
      return;
    }

    setConfirmDialog({
      isOpen: true,
      title: 'Delete Recipe',
      message: `Delete recipe "${receipt.name}" (${receipt.productCode})?\n\nThis action cannot be undone.`,
      type: 'danger',
      onConfirm: async () => {
        try {
          await receiptsAPI.deleteReceipt(receipt.id);
          await loadReceipts();
          await loadStatistics();
          setSelectedIds(selectedIds.filter(id => id !== receipt.id));
          toast.success(`Recipe "${receipt.name}" deleted successfully!`);
        } catch (error) {
          console.error('Error deleting receipt:', error);
          toast.error('Failed to delete receipt. Please try again.');
        }
      }
    });
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.length === 0) {
      toast.warning('Please select at least one receipt to delete.');
      return;
    }

    if (runningRecipeId && selectedIds.includes(runningRecipeId)) {
      const runningRecipe = receipts.find(r => r.id === runningRecipeId);
      toast.warning(`Recipe "${runningRecipe?.name ?? runningRecipeId}" is currently running. Please stop it before deleting.`);
      return;
    }

    setConfirmDialog({
      isOpen: true,
      title: 'Delete Multiple Recipes',
      message: `Delete ${selectedIds.length} selected recipe(s)?\n\nThis action cannot be undone.`,
      type: 'danger',
      onConfirm: async () => {
        try {
          // Delete all selected receipts
          await Promise.all(selectedIds.map(id => receiptsAPI.deleteReceipt(id)));
          await loadReceipts();
          await loadStatistics();
          setSelectedIds([]);
          setSelectAll(false);
          toast.success(`${selectedIds.length} recipe(s) deleted successfully!`);
        } catch (error) {
          console.error('Error deleting receipts:', error);
          toast.error('Failed to delete some receipts. Please try again.');
        }
      }
    });
  };

  const handleSelectAll = () => {
    if (selectAll) {
      setSelectedIds([]);
    } else {
      setSelectedIds(filteredReceipts.map(r => r.id));
    }
    setSelectAll(!selectAll);
  };

  const handleSelectReceipt = (receiptId: string) => {
    if (selectedIds.includes(receiptId)) {
      setSelectedIds(selectedIds.filter(id => id !== receiptId));
      setSelectAll(false);
    } else {
      const newSelectedIds = [...selectedIds, receiptId];
      setSelectedIds(newSelectedIds);
      if (newSelectedIds.length === filteredReceipts.length) {
        setSelectAll(true);
      }
    }
  };

  const openHistory = async (receipt: Receipt) => {
    try {
      setHistoryLoading(true);
      const data = await receiptsAPI.getLoadHistory(receipt.id, 0, 50);
      const items = data.items || [];
      setHistoryItems(items);
      setHistoryCount(data.count || 0);
    } catch (err) {
      console.error('Error loading history:', err);
      toast.error('Failed to load history.');
    } finally {
      setHistoryLoading(false);
    }
  };

  // reference to prevent TS "declared but its value is never read" when UI wiring removed
  void openHistory;

  const openGlobalHistory = async () => {
    try {
      setHistoryLoading(true);
      setViewMode('history');
      const data = await receiptsAPI.getGlobalLoadHistory(0, 100);
      const items = data.items || [];
      setHistoryItems(items);
      setHistoryCount(data.count || 0);
    } catch (err: any) {
      console.error('Error loading global history:', err);
      const msg = err?.response?.data?.detail || 'Failed to load global history.';
      toast.error(msg);
      setViewMode('list');
    } finally {
      setHistoryLoading(false);
    }
  };

  

  // Pin the currently-running recipe to the top of the visible page (if it's
  // present in the loaded slice). Server already sorted by `sort_by`; we only
  // reorder client-side so the running recipe is immediately findable. If the
  // running recipe lives on a different page in the chosen sort, it stays
  // there — toggling sort to `updated_desc` after a load brings it forward.
  const filteredReceipts = useMemo(() => {
    if (!runningRecipeId) return receipts;
    const idx = receipts.findIndex(r => r.id === runningRecipeId);
    if (idx <= 0) return receipts;
    const running = receipts[idx];
    if (!running) return receipts;
    return [running, ...receipts.slice(0, idx), ...receipts.slice(idx + 1)];
  }, [receipts, runningRecipeId]);

  // Render different views based on viewMode
  if (viewMode === 'create' || viewMode === 'edit') {
    return (
      <div className="receipts-page">
        <RecipeFormModal
          isOpen={true}
          onClose={handleBackToList}
          onSubmit={handleModalSubmit}
          recipe={selectedRecipe as any}
          mode={modalMode}
        />
      </div>
    );
  }

  if (viewMode === 'view') {
    return (
      <div className="receipts-page">
        <RecipeViewModal
          isOpen={true}
          onClose={handleBackToList}
          recipe={selectedRecipe}
          onEdit={handleViewModalEdit}
        />
      </div>
    );
  }

  if (viewMode === 'history') {
    const makeAbsolute = (p?: string) => {
      if (!p) return p;
      if (p.startsWith('http')) return p;
      return `${API_BASE_URL}${p}`;
    };

    return (
      <div className="receipts-page">
        <div className="history-view-page">
          <div className="page-header">
            <button className="back-btn" onClick={handleBackToList}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Back to List
            </button>
            <h2>Load History ({historyCount})</h2>
          </div>

          <div className="history-container">
            {historyLoading ? (
              <div style={{ textAlign: 'center', padding: '40px' }}>Loading history...</div>
            ) : historyItems.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#6b7280' }}>
                No load events recorded.
              </div>
            ) : (
              <table className="history-table">
                <thead>
                  <tr>
                    <th style={{ width: '120px' }}>Image</th>
                    <th>ID</th>
                    <th>Loaded By</th>
                    <th>Loaded At</th>
                    <th>Metadata</th>
                  </tr>
                </thead>
                <tbody>
                  {historyItems.map(item => {
                    const viz = item?.metadata?.camera_templates?.[0]?.templates?.[0]?.visualization_url;
                    return (
                      <tr key={item.id}>
                        <td style={{ verticalAlign: 'top', padding: '8px', width: '200px' }}>
                          {viz ? (
                            <img
                              src={makeAbsolute(viz)}
                              alt="visual"
                              style={{ width: '200px', height: 'auto', borderRadius: 4, cursor: 'pointer', objectFit: 'contain' }}
                              onClick={() => window.open(makeAbsolute(viz), '_blank')}
                            />
                          ) : (
                            <div style={{ color: '#888' }}>No image</div>
                          )}
                        </td>
                        <td style={{ verticalAlign: 'top', padding: '8px' }}>{item.id}</td>
                        <td style={{ verticalAlign: 'top', padding: '8px' }}>{item.loaded_by_full_name || item.loaded_by}</td>
                        <td style={{ verticalAlign: 'top', padding: '8px' }}>{new Date(item.loaded_at_local).toLocaleString()}</td>
                        <td style={{ maxWidth: '640px', verticalAlign: 'top', padding: '8px' }}>
                          {/* Compact metadata summary */}
                          <div style={{ fontSize: '0.9rem', marginBottom: '6px' }}>
                            <div><strong>Name:</strong> {item.metadata?.name || '—'}</div>
                            <div><strong>Product:</strong> {item.metadata?.product_code || '—'}</div>
                            <div><strong>Camera templates:</strong> {Array.isArray(item.metadata?.camera_templates) ? item.metadata.camera_templates.length : '—'}</div>
                          </div>

                          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                            <button
                              className="dashboard-btn"
                              style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                              onClick={() => setRawOpenMap(prev => ({ ...prev, [item.id]: !prev[item.id] }))}
                            >
                              {rawOpenMap[item.id] ? 'Hide raw' : 'Show raw'}
                            </button>
                            <div style={{ flex: 1 }} />
                          </div>
                          {rawOpenMap[item.id] && (
                            <div style={{ marginTop: '8px', maxHeight: '200px', overflow: 'auto', background: '#fafafa', borderRadius: '6px', padding: '8px' }}>
                              {Array.isArray(item.metadata?.cameras) && item.metadata.cameras.length > 0 ? (
                                item.metadata.cameras.map((cam: any, idx: number) => (
                                  <div key={idx} style={{ marginBottom: 8, fontSize: '0.85rem' }}>
                                    <div><strong>Camera #{idx + 1}</strong></div>
                                    <div>Exposure: {cam?.exposure_time ?? '—'} ms</div>
                                    <div>Delay: {cam?.delay_trigger ?? cam?.delay ?? '—'} ms</div>
                                    <div>Gain: {cam?.gain ?? '—'}</div>
                                    {cam?.trigger_config && (
                                      <div style={{ marginLeft: 8, marginTop: 4 }}>
                                        <div><strong>Trigger Config</strong></div>
                                        <div>Mode: {String(cam.trigger_config.trigger_mode)}</div>
                                        <div>Source: {cam.trigger_config.trigger_source || '—'}</div>
                                        <div>Selector: {cam.trigger_config.trigger_selector || '—'}</div>
                                        <div>Activation: {cam.trigger_config.trigger_activation || '—'}</div>
                                      </div>
                                    )}
                                  </div>
                                ))
                              ) : (
                                <div style={{ fontSize: '0.85rem' }}>No camera config available.</div>
                              )}
                            </div>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="receipts-page" style={{ position: 'relative' }}>
      {/* Recipe Loading Animation */}
      {showLoadingAnimation && (
        <RecipeLoadingTemplates
          template={loadingTemplate}
          progress={loadingProgress}
          isLightMode={isLightMode}
          overlayMode={overlayMode}
        />
      )}

      <div className="section-header">
        <h1>Production Recipes</h1>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          {selectedIds.length > 0 && canPerformAction('delete', 'receipt') && (
            <button 
              className="dashboard-btn delete-selected-btn" 
              onClick={handleDeleteSelected}
              style={{
                background: 'linear-gradient(135deg, #ef4444, #dc2626)',
                boxShadow: '0 2px 8px rgba(239, 68, 68, 0.3)'
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path d="M3 6H5H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M8 6V4C8 3.46957 8.21071 2.96086 8.58579 2.58579C8.96086 2.21071 9.46957 2 10 2H14C14.5304 2 15.0391 2.21071 15.4142 2.58579C15.7893 2.96086 16 3.46957 16 4V6M19 6V20C19 20.5304 18.7893 21.0391 18.4142 21.4142C18.0391 21.7893 17.5304 22 17 22H7C6.46957 22 5.96086 21.7893 5.58579 21.4142C5.21071 21.0391 5 20.5304 5 20V6H19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
                    Delete Selected ({selectedIds.length})
            </button>
          )}
          {canPerformAction('create', 'receipt') && (
            <button className="dashboard-btn" onClick={handleCreateReceipt}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <line x1="12" y1="5" x2="12" y2="19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                <line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
              Create Recipe
            </button>
          )}
          <button className="dashboard-btn" onClick={openGlobalHistory} title="Show global load history">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" style={{ marginRight: 6 }}>
              <path d="M21 10H3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M21 6H3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M21 14H3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            History
          </button>
        </div>
      </div>

      {error && (
        <div style={{ 
          padding: '12px', 
          marginBottom: '16px', 
          backgroundColor: '#fee', 
          color: '#c33', 
          borderRadius: '8px',
          border: '1px solid #fcc' 
        }}>
          {error}
        </div>
      )}

      {/* Summary Cards */}
      <div className="summary-cards">
        <div className="summary-card">
          <div className="card-icon receipts">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </div>
          <div className="card-info">
            <h3>Total Receipts</h3>
            <div className="card-value">{statistics.totalReceipts}</div>
            <span className="card-status normal">Active recipes</span>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-icon products">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <polyline points="20,6 9,17 4,12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="card-info">
            <h3>Total Products</h3>
            <div className="card-value">{statistics.totalProducts}</div>
            <span className="card-status increase">All products</span>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-icon success">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M22 11.08V12C21.9988 14.1564 21.3005 16.2547 20.0093 17.9818C18.7182 19.7088 16.9033 20.9725 14.8354 21.5839C12.7674 22.1953 10.5573 22.1219 8.53447 21.3746C6.51168 20.6273 4.78465 19.2461 3.61096 17.4371C2.43727 15.628 1.87979 13.4881 2.02168 11.3363C2.16356 9.18455 2.99721 7.13631 4.39828 5.49706C5.79935 3.85781 7.69279 2.71537 9.79619 2.24013C11.8996 1.7649 14.1003 1.98232 16.07 2.85999" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="22,4 12,14.01 9,11.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="card-info">
            <h3>Detection Threshold</h3>
            <div className="card-value">{statistics.successRate}%</div>
            <span className="card-status success">Average</span>
          </div>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="page-controls">
        <div className="search-box">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
            <path d="m21 21-4.35-4.35" stroke="currentColor" strokeWidth="2"/>
          </svg>
          <input
            type="text"
            placeholder="Search receipts by ID, name, product code..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button 
            onClick={handleSearch}
            style={{
              marginLeft: '8px',
              padding: '8px 16px',
              backgroundColor: '#2563eb',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            Search
          </button>
        </div>
        <div className="filter-buttons">
          <select
            className="filter-select"
            value={sortBy}
            onChange={(e) => {
              setSortBy(e.target.value as RecipeSortBy);
              setCurrentPage(1);
            }}
            title="Sort recipes"
          >
            <option value="created_desc">Created (newest)</option>
            <option value="created_asc">Created (oldest)</option>
            <option value="updated_desc">Updated (newest)</option>
            <option value="updated_asc">Updated (oldest)</option>
          </select>
          <button className="filter-btn" onClick={() => toast.info('Export feature - to be implemented')}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M21 15V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="7,10 12,15 17,10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="12" y1="15" x2="12" y2="3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Export
          </button>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <div>Loading receipts...</div>
        </div>
      )}

      {/* Receipts Table */}
      {!loading && (
        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: '50px' }}>
                  <input 
                    type="checkbox" 
                    checked={selectAll}
                    onChange={handleSelectAll}
                    style={{ cursor: 'pointer', width: '18px', height: '18px' }}
                  />
                </th>
                <th>Receipt ID</th>
                <th>Recipe Name</th>
                <th>Product Code</th>
                <th>Date</th>
                <th>Camera Settings</th>
                <th>Recognition Threshold</th>
                <th>Operator</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredReceipts.length === 0 ? (
                <tr>
                  <td colSpan={10} style={{ textAlign: 'center', padding: '40px' }}>
                    No receipts found. {searchTerm && 'Try adjusting your search.'}
                  </td>
                </tr>
              ) : (
                filteredReceipts.map(receipt => (
                  <tr key={receipt.id} className={selectedIds.includes(receipt.id) ? 'selected-row' : ''}>
                    <td>
                      <input 
                        type="checkbox"
                        checked={selectedIds.includes(receipt.id)}
                        onChange={() => handleSelectReceipt(receipt.id)}
                        style={{ cursor: 'pointer', width: '18px', height: '18px' }}
                      />
                    </td>
                    <td><strong>{receipt.id}</strong></td>
                    <td>{receipt.name}</td>
                    <td>{receipt.productCode}</td>
                    <td>{receipt.date}</td>
                    <td>
                      {receipt.cameras && receipt.cameras.length > 0 && (
                        <div style={{ fontSize: '0.85em' }}>
                          <div>Exp: {receipt.cameras[0]?.exposure_time || "N/A"}ms</div>
                          <div>Delay: {receipt.cameras[0]?.delay_trigger || "N/A"}ms</div>
                        </div>
                      )}
                    </td>
                    <td>
                      {receipt.modelThresholds && receipt.modelThresholds.recognition_threshold && (
                        <span className="text-success">
                          {(receipt.modelThresholds.recognition_threshold * 100).toFixed(0)}%
                        </span>
                      )}
                    </td>
                    <td>{receipt.operator}</td>
                    <td>
                      <span className={`status-badge ${receipt.status === 'Active' ? 'completed' : 'inactive'}`}>
                        {receipt.status}
                      </span>
                    </td>
                    <td>
                      <div className="action-buttons">
                        {runningRecipeId === receipt.id ? (
                          <button
                            className="action-btn stop-btn"
                            title="Stop Recipe"
                            onClick={() => handleStopReceipt(receipt)}
                            style={{ backgroundColor: '#dc3545', color: 'white', width: '85px' }}
                          >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                              <rect x="6" y="6" width="12" height="12" stroke="currentColor" strokeWidth="2" fill="currentColor"/>
                            </svg>
                            Stop
                          </button>
                        ) : (
                          <button
                            className="action-btn load-btn"
                            title="Load Receipt"
                            style={{ width: '85px' }}
                            onClick={() => handleLoadReceipt(receipt)}
                          >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                              <path d="M3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12Z" stroke="currentColor" strokeWidth="2"/>
                              <path d="M12 7V12L15 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                            Load
                          </button>
                        )}
                        
                        {canPerformAction('update', 'receipt') && (
                          <button 
                            className="action-btn edit" 
                            title="Edit"
                            onClick={() => handleEditReceipt(receipt)}
                          >
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                              <path d="M11 4H4C3.46957 4 2.96086 4.21071 2.58579 4.58579C2.21071 4.96086 2 5.46957 2 6V20C2 20.5304 2.21071 21.0391 2.58579 21.4142C2.96086 21.7893 3.46957 22 4 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                              <path d="M18.5 2.50001C18.8978 2.10219 19.4374 1.87869 20 1.87869C20.5626 1.87869 21.1022 2.10219 21.5 2.50001C21.8978 2.89784 22.1213 3.4374 22.1213 4.00001C22.1213 4.56262 21.8978 5.10219 21.5 5.50001L12 15L8 16L9 12L18.5 2.50001Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          </button>
                        )}
                        <button 
                          className="action-btn view" 
                          title="View"
                          onClick={() => handleViewReceipt(receipt)}
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                            <path d="M1 12C1 12 5 4 12 4C19 4 23 12 23 12C23 12 19 20 12 20C5 20 1 12 1 12Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </button>
                        {canPerformAction('create', 'receipt') && (
                          <button
                            className="action-btn clone"
                            title="Clone Recipe"
                            onClick={() => handleCloneReceipt(receipt)}
                          >
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                              <path d="M5 15H4C3.46957 15 2.96086 14.7893 2.58579 14.4142C2.21071 14.0391 2 13.5304 2 13V4C2 3.46957 2.21071 2.96086 2.58579 2.58579C2.96086 2.21071 3.46957 2 4 2H13C13.5304 2 14.0391 2.21071 14.4142 2.58579C14.7893 2.96086 15 3.46957 15 4V5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          </button>
                        )}
                        {canPerformAction('delete', 'receipt') && (
                          <button 
                            className="action-btn delete" 
                            title="Delete"
                            onClick={() => handleDeleteReceipt(receipt)}
                          >
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                              <path d="M3 6H5H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                              <path d="M8 6V4C8 3.46957 8.21071 2.96086 8.58579 2.58579C8.96086 2.21071 9.46957 2 10 2H14C14.5304 2 15.0391 2.21071 15.4142 2.58579C15.7893 2.96086 16 3.46957 16 4V6M19 6V20C19 20.5304 18.7893 21.0391 18.4142 21.4142C18.0391 21.7893 17.5304 22 17 22H7C6.46957 22 5.96086 21.7893 5.58579 21.4142C5.21071 21.0391 5 20.5304 5 20V6H19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      <div className="pagination">
        <button 
          className="pagination-btn" 
          disabled={currentPage === 1}
          onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
        >
          Previous
        </button>
        <div className="pagination-numbers">
          {[...Array(Math.min(totalPages, 5))].map((_, idx) => {
            const pageNum = idx + 1;
            return (
              <button 
                key={pageNum}
                className={`pagination-number ${currentPage === pageNum ? 'active' : ''}`}
                onClick={() => setCurrentPage(pageNum)}
              >
                {pageNum}
              </button>
            );
          })}
        </div>
        <button 
          className="pagination-btn"
          disabled={currentPage === totalPages}
          onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
        >
          Next
        </button>
      </div>

      {/* Confirmation Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={() => setConfirmDialog({ ...confirmDialog, isOpen: false })}
        onConfirm={confirmDialog.onConfirm || (() => {})}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
      />
    </div>
  );
}