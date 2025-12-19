import React, { useState, useEffect, ChangeEvent } from 'react';
import { receiptsAPI } from '@/services/api';
import RecipeFormModal from './RecipeFormModal';
import RecipeViewModal from './RecipeViewModal';
import ConfirmDialog from '@/components/shared/ConfirmDialog';
import { useToast } from '@/contexts/ToastContext';
import type { Receipt, Statistics, RecipeCreate, RecipeUpdate } from '@/types';

interface ConfirmDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  onConfirm: (() => void) | null;
}

const Receipts: React.FC = () => {
  const toast = useToast();
  const [receipts, setReceipts] = useState<Receipt[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statistics, setStatistics] = useState<Statistics>({
    totalReceipts: 0,
    totalProducts: 0,
    successRate: 0
  });
  
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const itemsPerPage = 10;

  // Modal states
  const [isFormModalOpen, setIsFormModalOpen] = useState(false);
  const [isViewModalOpen, setIsViewModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState<'create' | 'edit'>('create');
  const [selectedRecipe, setSelectedRecipe] = useState<Receipt | null>(null);
  
  // Selection states
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [selectAll, setSelectAll] = useState(false);

  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning',
    onConfirm: null
  });

  useEffect(() => {
    loadReceipts();
    loadStatistics();
  }, [currentPage]);

  const loadReceipts = async () => {
    try {
      setLoading(true);
      const skip = (currentPage - 1) * itemsPerPage;
      const data = await receiptsAPI.getAllReceipts(skip, itemsPerPage, true);
      
      // Transform recipe data to receipt format for UI
      const transformedReceipts: Receipt[] = data.map((recipe): Receipt => ({
        id: recipe.id,
        name: recipe.name,
        productCode: recipe.product_code,
        date: recipe.created_at ? new Date(recipe.created_at).toISOString().split('T')[0] : new Date().toISOString().split('T')[0],
        camera: `Camera Settings: ${recipe.camera_settings?.exposure_time || 'N/A'}ms`,
        products: 0,
        passed: 0,
        failed: 0,
        operator: recipe.created_by || 'System',
        status: recipe.is_active ? 'Active' : 'Inactive',
        description: recipe.description || '',
        cameras: recipe.cameras || [],
        camera_templates: [],
        delay_reject: recipe.delay_reject,
        cameraSettings: recipe.camera_settings,
        modelThresholds: recipe.model_thresholds,
        template_config: recipe.template_config,
        roi_config: recipe.roi_config,
        is_active: recipe.is_active,
        createdAt: recipe.created_at || new Date().toISOString(),
        updatedAt: recipe.updated_at || new Date().toISOString()
      }));
      
      setReceipts(transformedReceipts);
      
      // Calculate total pages
      const countData = await receiptsAPI.getReceiptsCount(true);
      setTotalPages(Math.ceil(countData.count / itemsPerPage));
      
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

  const filteredReceipts = receipts.filter(receipt => {
    const search = searchTerm.toLowerCase();
    return (
      receipt.name.toLowerCase().includes(search) ||
      receipt.productCode.toLowerCase().includes(search) ||
      receipt.operator.toLowerCase().includes(search)
    );
  });

  const handleCreateReceipt = () => {
    setModalMode('create');
    setSelectedRecipe(null);
    setIsFormModalOpen(true);
  };

  const handleEditReceipt = (receipt: Receipt) => {
    setModalMode('edit');
    setSelectedRecipe(receipt);
    setIsFormModalOpen(true);
  };

  const handleModalSubmit = async (formData: RecipeCreate | RecipeUpdate) => {
    try {
      if (modalMode === 'create') {
        await receiptsAPI.createReceipt(formData as RecipeCreate);
        toast.success(`Recipe "${(formData as RecipeCreate).name}" created successfully!`);
      } else if (selectedRecipe) {
        await receiptsAPI.updateReceipt(selectedRecipe.id, formData as RecipeUpdate);
        toast.success(`Recipe updated successfully!`);
      }
      
      await loadReceipts();
      await loadStatistics();
      setIsFormModalOpen(false);
    } catch (error: any) {
      console.error('Error saving recipe:', error);
      const errorMsg = error.response?.data?.detail || 'Failed to save recipe. Please try again.';
      toast.error(errorMsg);
      throw error;
    }
  };

  const handleViewReceipt = (receipt: Receipt) => {
    setSelectedRecipe(receipt);
    setIsViewModalOpen(true);
  };

  const handleViewModalEdit = () => {
    setIsViewModalOpen(false);
    if (selectedRecipe) {
      handleEditReceipt(selectedRecipe);
    }
  };

  const handleDeleteReceipt = async (receipt: Receipt) => {
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

    setConfirmDialog({
      isOpen: true,
      title: 'Delete Multiple Recipes',
      message: `Delete ${selectedIds.length} selected recipe(s)?\n\nThis action cannot be undone.`,
      type: 'danger',
      onConfirm: async () => {
        try {
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
      setSelectedIds([...selectedIds, receiptId]);
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading recipes...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <p className="error-message">{error}</p>
        <button onClick={loadReceipts} className="retry-button">Retry</button>
      </div>
    );
  }

  return (
    <div className="receipts-section">
      {/* Header */}
      <div className="section-header">
        <div>
          <h2>Recipes Management</h2>
          <p>Manage product recipes and configurations</p>
        </div>
        <button className="create-button" onClick={handleCreateReceipt}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          Create Recipe
        </button>
      </div>

      {/* Statistics */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </div>
          <div className="stat-content">
            <div className="stat-label">Total Recipes</div>
            <div className="stat-value">{statistics.totalReceipts}</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon success">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
          <div className="stat-content">
            <div className="stat-label">Success Rate</div>
            <div className="stat-value">{statistics.successRate.toFixed(1)}%</div>
          </div>
        </div>
      </div>

      {/* Search and Actions */}
      <div className="table-controls">
        <div className="search-box">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
            <path d="m21 21-4.35-4.35" stroke="currentColor" strokeWidth="2"/>
          </svg>
          <input
            type="text"
            placeholder="Search recipes..."
            value={searchTerm}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setSearchTerm(e.target.value)}
          />
        </div>
        
        {selectedIds.length > 0 && (
          <button className="action-btn delete" onClick={handleDeleteSelected}>
            Delete Selected ({selectedIds.length})
          </button>
        )}
      </div>

      {/* Recipes Table */}
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ width: '50px' }}>
                <input
                  type="checkbox"
                  checked={selectAll}
                  onChange={handleSelectAll}
                />
              </th>
              <th>Recipe Name</th>
              <th>Product Code</th>
              <th>Status</th>
              <th>Created</th>
              <th>Operator</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredReceipts.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', padding: '2rem' }}>
                  {searchTerm ? 'No recipes found' : 'No recipes available'}
                </td>
              </tr>
            ) : (
              filteredReceipts.map((receipt) => (
                <tr key={receipt.id} className={selectedIds.includes(receipt.id) ? 'selected-row' : ''}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(receipt.id)}
                      onChange={() => handleSelectReceipt(receipt.id)}
                    />
                  </td>
                  <td>
                    <strong>{receipt.name}</strong>
                  </td>
                  <td>{receipt.productCode}</td>
                  <td>
                    <span className={`status-badge ${receipt.status === 'Active' ? 'success' : 'inactive'}`}>
                      {receipt.status}
                    </span>
                  </td>
                  <td>{receipt.date}</td>
                  <td>{receipt.operator}</td>
                  <td>
                    <div className="action-buttons">
                      <button
                        className="action-btn view"
                        onClick={() => handleViewReceipt(receipt)}
                        title="View"
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" stroke="currentColor" strokeWidth="2"/>
                          <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                      </button>
                      <button
                        className="action-btn edit"
                        onClick={() => handleEditReceipt(receipt)}
                        title="Edit"
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                          <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" stroke="currentColor" strokeWidth="2"/>
                          <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                      </button>
                      <button
                        className="action-btn delete"
                        onClick={() => handleDeleteReceipt(receipt)}
                        title="Delete"
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                          <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="pagination">
          <button 
            disabled={currentPage === 1}
            onClick={() => setCurrentPage(currentPage - 1)}
          >
            Previous
          </button>
          <span>Page {currentPage} of {totalPages}</span>
          <button 
            disabled={currentPage === totalPages}
            onClick={() => setCurrentPage(currentPage + 1)}
          >
            Next
          </button>
        </div>
      )}

      {/* Modals */}
      <RecipeFormModal
        isOpen={isFormModalOpen}
        onClose={() => setIsFormModalOpen(false)}
        onSubmit={handleModalSubmit}
        recipe={selectedRecipe}
        mode={modalMode}
      />

      {selectedRecipe && (
        <RecipeViewModal
          isOpen={isViewModalOpen}
          onClose={() => setIsViewModalOpen(false)}
          recipe={selectedRecipe}
          onEdit={handleViewModalEdit}
        />
      )}

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
};

export default Receipts;
