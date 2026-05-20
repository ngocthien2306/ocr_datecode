import React, { useState, useEffect, useCallback } from 'react';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { inferenceResultsAPI, type InferenceResultResponse, type FailReason, type CenterDirection } from '@/services/inferenceResults';
import { recipesAPI } from '@/services/api';
import type { Recipe } from '@/types';
import InspectionResultRow from './InspectionResultRow';
import ConfirmDialog from '@/components/shared/ConfirmDialog';
import Toast from '@/components/shared/Toast';
import { getDateRangeUTC, convertLocalToUTC, getTimezoneOffset, getTimezoneDisplay } from '@/utils/timezone';

interface InspectionResultsTabProps {
  dateRange: string;
}

const InspectionResultsTab: React.FC<InspectionResultsTabProps> = ({ dateRange }) => {
  const [results, setResults] = useState<InferenceResultResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [totalCount, setTotalCount] = useState(0);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());

  // Recipes for filter dropdown
  const [recipes, setRecipes] = useState<Recipe[]>([]);

  // Filters
  const [selectedRecipe, setSelectedRecipe] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<'all' | 'PASS' | 'FAIL'>('all');
  const [customStartDate, setCustomStartDate] = useState<Date | null>(null);
  const [customEndDate, setCustomEndDate] = useState<Date | null>(null);
  const [useCustomDateRange, setUseCustomDateRange] = useState(false);

  // Fail reason filters
  const [failReasons, setFailReasons] = useState<Set<FailReason>>(new Set());
  const [wrinkledMinArea, setWrinkledMinArea] = useState<string>('');
  const [centerDirection, setCenterDirection] = useState<CenterDirection>('any');

  // Confirm dialog state
  const [confirmDialog, setConfirmDialog] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    type: 'warning' | 'danger' | 'info';
    onConfirm: () => void;
  }>({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning',
    onConfirm: () => {}
  });

  // Toast state
  const [toast, setToast] = useState<{
    show: boolean;
    message: string;
    type: 'success' | 'error' | 'warning' | 'info';
  }>({
    show: false,
    message: '',
    type: 'info'
  });

  const showToast = useCallback((message: string, type: 'success' | 'error' | 'warning' | 'info' = 'info') => {
    setToast({ show: true, message, type });
  }, []);

  const closeToast = useCallback(() => {
    setToast(prev => ({ ...prev, show: false }));
  }, []);

  const showConfirm = useCallback((
    title: string,
    message: string,
    onConfirm: () => void,
    type: 'warning' | 'danger' | 'info' = 'warning'
  ) => {
    setConfirmDialog({ isOpen: true, title, message, type, onConfirm });
  }, []);

  const closeConfirm = useCallback(() => {
    setConfirmDialog(prev => ({ ...prev, isOpen: false }));
  }, []);

  // Fetch recipes on mount
  useEffect(() => {
    fetchRecipes();
  }, []);

  // Fetch results when filters change
  useEffect(() => {
    fetchResults();
  }, [dateRange, currentPage, pageSize, selectedRecipe, statusFilter, useCustomDateRange, customStartDate, customEndDate, failReasons, wrinkledMinArea, centerDirection]);

  const fetchRecipes = async () => {
    try {
      // Fetch all active recipes
      const data = await recipesAPI.getAllRecipes(0, 100, true);
      setRecipes(data);
    } catch (error) {
      console.error('Error fetching recipes:', error);
    }
  };

  const fetchResults = async () => {
    try {
      setLoading(true);

      // Calculate date range
      const { start_date, end_date } = getDateRange(dateRange);

      // Build filters
      const filters: any = {
        skip: (currentPage - 1) * pageSize,
        limit: pageSize,
        start_date,
        end_date
      };

      if (selectedRecipe !== 'all') {
        filters.recipe_id = selectedRecipe;
      }

      if (statusFilter !== 'all') {
        filters.pass_fail = statusFilter;
      }

      if (failReasons.size > 0) {
        filters.fail_reasons = Array.from(failReasons).join(',');
        if (failReasons.has('wrinkled') && wrinkledMinArea !== '') {
          filters.wrinkled_min_area = Number(wrinkledMinArea);
        }
        if (failReasons.has('center') && centerDirection !== 'any') {
          filters.center_direction = centerDirection;
        }
      }

      // Fetch results
      const data = await inferenceResultsAPI.getResults(filters);

      // Debug: Check if IDs are unique
      console.log('Fetched results:', data.length);
      console.log('Unique IDs:', new Set(data.map(r => r.id)).size);
      console.log('First few IDs:', data.slice(0, 3).map(r => ({ id: r.id, recipe: r.recipe_name })));

      setResults(data);

      // Fetch count for pagination
      const countFilters: any = { start_date, end_date };
      if (selectedRecipe !== 'all') countFilters.recipe_id = selectedRecipe;
      if (statusFilter !== 'all') countFilters.pass_fail = statusFilter;
      if (failReasons.size > 0) {
        countFilters.fail_reasons = Array.from(failReasons).join(',');
        if (failReasons.has('wrinkled') && wrinkledMinArea !== '') {
          countFilters.wrinkled_min_area = Number(wrinkledMinArea);
        }
        if (failReasons.has('center') && centerDirection !== 'any') {
          countFilters.center_direction = centerDirection;
        }
      }

      const countData = await inferenceResultsAPI.getCount(countFilters);
      setTotalCount(countData.count);

    } catch (error) {
      console.error('Error fetching inspection results:', error);
    } finally {
      setLoading(false);
    }
  };

  const getDateRange = (range: string): { start_date: string; end_date: string } => {
    // Use custom date range if enabled
    if (useCustomDateRange && customStartDate && customEndDate) {
      const offset = getTimezoneOffset();

      console.log('[Timezone Debug] Custom Date Range:');
      console.log('  User selected START:', customStartDate.toString());
      console.log('  User selected END:', customEndDate.toString());
      console.log('  Timezone offset (minutes):', offset);

      // User picks date/time thinking it's in local timezone (e.g., Vietnam time)
      // We need to convert: Local time → UTC
      // Example: User picks "17/01/2026 02:00" VN → "16/01/2026 19:00" UTC
      const startUTC = convertLocalToUTC(customStartDate, offset);
      const endUTC = convertLocalToUTC(customEndDate, offset);

      console.log('  Converted START UTC:', startUTC.toISOString());
      console.log('  Converted END UTC:', endUTC.toISOString());

      return {
        start_date: startUTC.toISOString(),
        end_date: endUTC.toISOString()
      };
    }

    // Use preset range with proper timezone handling
    const result = getDateRangeUTC(range);
    console.log(`[Timezone Debug] Preset Range "${range}":`, result);
    return result;
  };

  const toggleRowExpanded = (id: string) => {
    console.log('Toggle expand for ID:', id);
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(id)) {
      newExpanded.delete(id);
      console.log('Collapsed:', id);
    } else {
      newExpanded.add(id);
      console.log('Expanded:', id);
    }
    console.log('All expanded IDs:', Array.from(newExpanded));
    setExpandedRows(newExpanded);
  };

  const handleDelete = (id: string) => {
    showConfirm(
      'Delete Result',
      'Are you sure you want to delete this inspection result?',
      async () => {
        try {
          await inferenceResultsAPI.delete(id);
          showToast('Result deleted successfully', 'success');
          fetchResults();
        } catch (error) {
          console.error('Error deleting result:', error);
          showToast('Failed to delete result', 'error');
        }
      },
      'danger'
    );
  };

  const toggleRowSelected = (id: string) => {
    console.log('Toggle select for ID:', id);
    const newSelected = new Set(selectedRows);
    if (newSelected.has(id)) {
      newSelected.delete(id);
      console.log('Deselected:', id);
    } else {
      newSelected.add(id);
      console.log('Selected:', id);
    }
    console.log('All selected IDs:', Array.from(newSelected));
    setSelectedRows(newSelected);
  };

  const toggleAllRows = () => {
    if (selectedRows.size === results.length) {
      setSelectedRows(new Set());
    } else {
      setSelectedRows(new Set(results.map(r => r.id)));
    }
  };

  const handleBulkDelete = () => {
    if (selectedRows.size === 0) {
      showToast('Please select at least one result to delete', 'warning');
      return;
    }

    showConfirm(
      'Delete Selected Results',
      `Are you sure you want to delete ${selectedRows.size} inspection result(s)?`,
      async () => {
        try {
          // Delete all selected rows, tracking successes and failures
          const ids = Array.from(selectedRows);
          const deleteResults = await Promise.allSettled(ids.map(id => inferenceResultsAPI.delete(id)));

          const failures = deleteResults.filter((r): r is PromiseRejectedResult => r.status === 'rejected');
          const successes = deleteResults.filter((r): r is PromiseFulfilledResult<any> => r.status === 'fulfilled');

          if (failures.length > 0) {
            console.error('Failed to delete some results:', failures.map((f) => ({
              id: ids[deleteResults.indexOf(f)],
              error: f.reason
            })));

            if (successes.length > 0) {
              showToast(`Deleted ${successes.length} result(s), but ${failures.length} failed.`, 'warning');
            } else {
              showToast(`Failed to delete ${failures.length} result(s).`, 'error');
            }
          } else {
            showToast(`Successfully deleted ${successes.length} result(s).`, 'success');
          }

          // Clear selection and refresh
          setSelectedRows(new Set());
          fetchResults();
        } catch (error) {
          console.error('Error bulk deleting results:', error);
          showToast('Failed to delete results', 'error');
        }
      },
      'danger'
    );
  };

  // Get user role for permissions
  const toggleFailReason = (reason: FailReason) => {
    setFailReasons(prev => {
      const next = new Set(prev);
      if (next.has(reason)) {
        next.delete(reason);
      } else {
        next.add(reason);
      }
      return next;
    });
    setCurrentPage(1);
  };

  const getUserRole = (): string => {
    try {
      const userStr = localStorage.getItem('user');
      if (userStr) {
        const user = JSON.parse(userStr);
        return user.role || 'viewer';
      }
    } catch (error) {
      console.error('Error getting user role:', error);
    }
    return 'viewer';
  };

  const userRole = getUserRole();
  const canDelete = userRole === 'admin' || userRole === 'supervisor';

  // Pagination helpers
  const totalPages = Math.ceil(totalCount / pageSize);
  const startIndex = (currentPage - 1) * pageSize + 1;
  const endIndex = Math.min(currentPage * pageSize, totalCount);

  const goToPage = (page: number) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
    }
  };

  return (
    <div className="inspection-results-tab">
      {/* Bulk Actions Bar */}
      {canDelete && selectedRows.size > 0 && (
        <div className="bulk-actions-bar">
          <span className="selected-count">
            {selectedRows.size} item(s) selected
          </span>
          <button className="bulk-delete-btn" onClick={handleBulkDelete}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <polyline points="3,6 5,6 21,6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6m3 0V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Delete Selected
          </button>
          <button className="bulk-clear-btn" onClick={() => setSelectedRows(new Set())}>
            Clear Selection
          </button>
        </div>
      )}

      {/* Filters */}
      <div className="filters-bar">

        {/* ── Row 1: Primary filters ── */}
        <div className="filter-row">
          <div className="filter-group">
            <label>Recipe:</label>
            <select value={selectedRecipe} onChange={(e) => {
              setSelectedRecipe(e.target.value);
              setCurrentPage(1);
            }}>
              <option value="all">All Recipes</option>
              {recipes.map((recipe) => (
                <option key={recipe.id} value={recipe.id}>{recipe.name}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Status:</label>
            <select value={statusFilter} onChange={(e) => {
              setStatusFilter(e.target.value as 'all' | 'PASS' | 'FAIL');
              setCurrentPage(1);
            }}>
              <option value="all">All</option>
              <option value="PASS">PASS</option>
              <option value="FAIL">FAIL</option>
            </select>
          </div>

          <div className="filter-group-divider" />

          <div className="filter-group">
            <label>
              <input
                type="checkbox"
                checked={useCustomDateRange}
                onChange={(e) => {
                  setUseCustomDateRange(e.target.checked);
                  setCurrentPage(1);
                }}
                className="custom-date-checkbox"
              />
              Custom Date ({getTimezoneDisplay()}):
            </label>
          </div>

          {useCustomDateRange && (
            <>
              <div className="filter-group">
                <label>From:</label>
                <DatePicker
                  selected={customStartDate}
                  onChange={(date: Date | null) => { setCustomStartDate(date); setCurrentPage(1); }}
                  showTimeSelect
                  timeFormat="HH:mm"
                  timeIntervals={15}
                  dateFormat="dd/MM/yyyy HH:mm"
                  placeholderText="Start date & time"
                  className="custom-datepicker-input"
                  calendarClassName="custom-datepicker-calendar"
                />
              </div>
              <div className="filter-group">
                <label>To:</label>
                <DatePicker
                  selected={customEndDate}
                  onChange={(date: Date | null) => { setCustomEndDate(date); setCurrentPage(1); }}
                  showTimeSelect
                  timeFormat="HH:mm"
                  timeIntervals={15}
                  dateFormat="dd/MM/yyyy HH:mm"
                  placeholderText="End date & time"
                  className="custom-datepicker-input"
                  calendarClassName="custom-datepicker-calendar"
                  minDate={customStartDate || undefined}
                />
              </div>
            </>
          )}

          <button className="dashboard-btn" onClick={fetchResults}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M21.5 2V6M21.5 6H17.5M21.5 6L18.5 3C16.8 1.5 14.5 0.5 12 0.5C5.5 0.5 0.5 5.5 0.5 12C0.5 18.5 5.5 23.5 12 23.5C17.5 23.5 22 19.5 23 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Apply
          </button>
        </div>

        {/* ── Row 2: Fail Reason ── */}
        <div className="filter-row filter-row--secondary">
          <span className="fail-section-label">Fail Reason:</span>

          <div className="fail-pills">
            {([
              { key: 'text',     label: 'Text Verify' },
              { key: 'char',     label: 'Char Verify' },
              { key: 'template', label: 'Template' },
              { key: 'wrinkled', label: 'Wrinkled' },
              { key: 'center',   label: 'Center Align' },
              { key: 'color',    label: 'Color Check' },
            ] as { key: FailReason; label: string }[]).map(({ key, label }) => (
              <button
                key={key}
                className={`fail-pill${failReasons.has(key) ? ' fail-pill--active' : ''}`}
                onClick={() => toggleFailReason(key)}
              >
                <span className="fail-pill-dot" />
                {label}
              </button>
            ))}
          </div>

          {failReasons.has('wrinkled') && (
            <>
              <span className="filter-sub-sep">›</span>
              <div className="filter-sub-group">
                <label>Min Area (px²):</label>
                <input
                  type="number"
                  min={0}
                  value={wrinkledMinArea}
                  placeholder="e.g. 2000"
                  onChange={(e) => { setWrinkledMinArea(e.target.value); setCurrentPage(1); }}
                  className="fail-reason-input"
                />
              </div>
            </>
          )}

          {failReasons.has('center') && (
            <>
              <span className="filter-sub-sep">›</span>
              <div className="filter-sub-group">
                <label>Direction:</label>
                <select
                  className="fail-sub-select"
                  value={centerDirection}
                  onChange={(e) => { setCenterDirection(e.target.value as CenterDirection); setCurrentPage(1); }}
                >
                  <option value="any">Any</option>
                  <option value="left">Left</option>
                  <option value="right">Right</option>
                </select>
              </div>
            </>
          )}
        </div>

      </div>

      {/* Results Table */}
      <div className="results-table-container">
        {loading ? (
          <div className="loading-state">Loading...</div>
        ) : results.length === 0 ? (
          <div className="empty-state">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
              <line x1="12" y1="8" x2="12" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <circle cx="12" cy="16" r="0.5" fill="currentColor"/>
            </svg>
            <p>No inspection results found</p>
          </div>
        ) : (
          <>
            <table className="results-table">
              <thead>
                <tr>
                  {canDelete && (
                    <th style={{ width: '50px' }}>
                      <input
                        type="checkbox"
                        checked={selectedRows.size === results.length && results.length > 0}
                        onChange={(e) => {
                          e.stopPropagation();
                          toggleAllRows();
                        }}
                        onClick={(e) => e.stopPropagation()}
                        className="row-checkbox"
                      />
                    </th>
                  )}
                  <th style={{ width: '50px' }}></th>
                  <th style={{ width: '150px' }}>Time</th>
                  <th style={{ width: '200px' }}>Recipe</th>
                  <th style={{ width: '100px' }}>Result</th>
                  <th style={{ width: '100px' }}>Cameras</th>
                  <th style={{ width: '120px' }}>Text Verify</th>
                  <th style={{ width: '120px' }}>Char Verify</th>
                  <th style={{ width: '100px' }}>Confidence</th>
                  <th style={{ width: '150px' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result) => (
                  <InspectionResultRow
                    key={result.id}
                    result={result}
                    isExpanded={expandedRows.has(result.id)}
                    isSelected={selectedRows.has(result.id)}
                    onToggleExpand={() => toggleRowExpanded(result.id)}
                    onToggleSelect={() => toggleRowSelected(result.id)}
                    onDelete={() => handleDelete(result.id)}
                    canDelete={canDelete}
                  />
                ))}
              </tbody>
            </table>

            {/* Pagination */}
            <div className="pagination-controls">
              <div className="pagination-info">
                Showing {startIndex}-{endIndex} of {totalCount} results
              </div>

              <div className="pagination-buttons">
                <button
                  className="pagination-btn"
                  onClick={() => goToPage(1)}
                  disabled={currentPage === 1}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <polyline points="11,17 6,12 11,7" stroke="currentColor" strokeWidth="2"/>
                    <polyline points="18,17 13,12 18,7" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </button>

                <button
                  className="pagination-btn"
                  onClick={() => goToPage(currentPage - 1)}
                  disabled={currentPage === 1}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <polyline points="15,18 9,12 15,6" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </button>

                <span className="page-indicator">
                  Page {currentPage} of {totalPages}
                </span>

                <button
                  className="pagination-btn"
                  onClick={() => goToPage(currentPage + 1)}
                  disabled={currentPage === totalPages}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <polyline points="9,18 15,12 9,6" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </button>

                <button
                  className="pagination-btn"
                  onClick={() => goToPage(totalPages)}
                  disabled={currentPage === totalPages}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <polyline points="13,17 18,12 13,7" stroke="currentColor" strokeWidth="2"/>
                    <polyline points="6,17 11,12 6,7" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </button>
              </div>

              <div className="page-size-selector">
                <label>Per page:</label>
                <select value={pageSize} onChange={(e) => {
                  setPageSize(Number(e.target.value));
                  setCurrentPage(1);
                }}>
                  <option value="10">10</option>
                  <option value="25">25</option>
                  <option value="50">50</option>
                  <option value="100">100</option>
                </select>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Confirm Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={closeConfirm}
        onConfirm={confirmDialog.onConfirm}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
        confirmText="Delete"
        cancelText="Cancel"
      />

      {/* Toast */}
      {toast.show && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={closeToast}
        />
      )}
    </div>
  );
};

export default InspectionResultsTab;
