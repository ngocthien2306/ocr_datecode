import React, { useState, useEffect } from 'react';
import { actionLogsAPI } from '@/services/api';
import { useToast } from '@/contexts/ToastContext';
import { IndustrialLoading } from '@/components/ui/IndustrialLoading';
import type { ActionLog, ActionLogFilters, ActionType, ResourceType } from '@/types';
import { usersAPI } from '@/services/api';

interface LogsProps {}

const Logs: React.FC<LogsProps> = () => {
  const toast = useToast();
  const [logs, setLogs] = useState<ActionLog[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filters, setFilters] = useState<ActionLogFilters>({});
  const [currentPage, setCurrentPage] = useState(1);
  const [totalLogs, setTotalLogs] = useState(0);
  const [selectedLog, setSelectedLog] = useState<ActionLog | null>(null);
  const [loadingTemplate, setLoadingTemplate] = useState('logs');
  const [expandedDetails, setExpandedDetails] = useState<Set<string>>(new Set());
  const [userOptions, setUserOptions] = useState<{id: string; username: string; full_name?: string}[]>([]);
  const [usernameToFullname, setUsernameToFullname] = useState<Record<string, string>>({});
  const [viewMode, setViewMode] = useState<'list' | 'detail'>('list');
  const logsPerPage = 50;

  const actionTypeLabels: Record<ActionType, string> = {
    login: 'Login',
    logout: 'Logout',
    create_user: 'Create User',
    update_user: 'Update User',
    delete_user: 'Delete User',
    reset_user_password: 'Reset Password',
    create_recipe: 'Create Recipe',
    update_recipe: 'Update Recipe',
    delete_recipe: 'Delete Recipe',
    load_recipe: 'Load Recipe',
    create_camera: 'Create Camera',
    update_camera: 'Update Camera',
    delete_camera: 'Delete Camera'
  };

  const resourceTypeLabels: Record<ResourceType, string> = {
    user: 'User',
    recipe: 'Recipe',
    camera: 'Camera',
    auth: 'Authentication'
  };

  const loadLogs = async () => {
    try {
      setIsLoading(true);
      const skip = (currentPage - 1) * logsPerPage;
      const fetchedLogs = await actionLogsAPI.getActionLogs(skip, logsPerPage, filters);
      setLogs(fetchedLogs);
      // Note: API doesn't return total count, so we'll estimate
      setTotalLogs(fetchedLogs.length >= logsPerPage ? (currentPage + 1) * logsPerPage : currentPage * logsPerPage);
    } catch (error) {
      console.error('Error loading logs:', error);
      toast.error('Error loading activity logs');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadLogs();
    // Reset expanded details when page or filters change
    setExpandedDetails(new Set());
  }, [currentPage, filters]);

  useEffect(() => {
    const savedTemplate = localStorage.getItem('logsLoading');
    if (savedTemplate) {
      setLoadingTemplate(savedTemplate);
    }

    const handleTemplateChange = (event: CustomEvent) => {
      if (event.detail.tab === 'logsLoading') {
        setLoadingTemplate(event.detail.template);
      }
    };

    window.addEventListener('tabLoadingChanged', handleTemplateChange as EventListener);

    return () => {
      window.removeEventListener('tabLoadingChanged', handleTemplateChange as EventListener);
    };
  }, []);

  useEffect(() => {
    // fetch simple user list for select; ignore errors (e.g., insufficient permissions)
    let mounted = true;
    (async () => {
      try {
        const users = await usersAPI.getSimpleUsers();
        if (!mounted) return;
        setUserOptions(users);
        const map: Record<string, string> = {};
        users.forEach(u => { if (u.username) map[u.username] = u.full_name || ''; });
        setUsernameToFullname(map);
      } catch (err) {
        // silently ignore (will mean select is empty for users without permission)
        console.debug('Could not load simple users for select', err);
      }
    })();
    return () => { mounted = false; };
  }, []);

  const handleFilterChange = (field: keyof ActionLogFilters, value: string) => {
    console.log('Filter change:', field, value);
    setFilters(prev => ({
      ...prev,
      [field]: value || undefined
    }));
    setCurrentPage(1); // Reset to first page when filters change
  };

  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatValue = (value: any): string => {
    if (!value) return '-';
    if (typeof value === 'object') {
      return JSON.stringify(value, null, 2);
    }
    return String(value);
  };

  const getActionTypeColor = (actionType: ActionType): string => {
    if (actionType.includes('create')) return 'text-green-600 bg-green-100';
    if (actionType.includes('update')) return 'text-blue-600 bg-blue-100';
    if (actionType.includes('delete')) return 'text-red-600 bg-red-100';
    if (actionType === 'login') return 'text-purple-600 bg-purple-100';
    if (actionType === 'logout') return 'text-gray-600 bg-gray-100';
    return 'text-gray-600 bg-gray-100';
  };

  const getResourceTypeColor = (resourceType: ResourceType): string => {
    switch (resourceType) {
      case 'user': return 'text-blue-600 bg-blue-100';
      case 'recipe': return 'text-orange-600 bg-orange-100';
      case 'camera': return 'text-teal-600 bg-teal-100';
      case 'auth': return 'text-purple-600 bg-purple-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const toggleDetailsExpansion = (logId: string) => {
    const newExpanded = new Set(expandedDetails);
    if (newExpanded.has(logId)) {
      newExpanded.delete(logId);
    } else {
      newExpanded.add(logId);
    }
    setExpandedDetails(newExpanded);
  };

  const shouldShowDetailsToggle = (log: ActionLog): boolean => {
    const oldValueText = formatValue(log.old_value);
    const newValueText = formatValue(log.new_value);
    const detailsText = oldValueText + newValueText;
    return detailsText.length > 150; // Show toggle if content is longer than 150 chars
  };

  const truncateText = (text: string, maxLength: number = 50): string => {
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength) + '...';
  };

  const handleViewDetails = (log: ActionLog) => {
    setSelectedLog(log);
    setViewMode('detail');
  };

  const handleBackToList = () => {
    setViewMode('list');
    setSelectedLog(null);
  };

  // Detail View Mode
  if (viewMode === 'detail' && selectedLog) {
    return (
      <div className="p-6">
        <div className="mb-6">
          <button
            onClick={handleBackToList}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Back to List
          </button>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-6">Action Log Details</h2>

          {/* Log Info */}
          <div className="mb-6 bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Time:</span>
                <div className="text-gray-900 dark:text-gray-100 mt-1">{formatDateTime(selectedLog.timestamp)}</div>
              </div>
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">User:</span>
                <div className="text-gray-900 dark:text-gray-100 mt-1">{selectedLog.username}</div>
                <div className="text-gray-500 dark:text-gray-400 text-xs mt-1">{selectedLog.user_id}</div>
              </div>
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Action:</span>
                <div className="mt-1">
                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getActionTypeColor(selectedLog.action_type)}`}>
                    {actionTypeLabels[selectedLog.action_type]}
                  </span>
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Resource:</span>
                <div className="mt-1">
                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getResourceTypeColor(selectedLog.resource_type)}`}>
                    {resourceTypeLabels[selectedLog.resource_type]}
                  </span>
                </div>
              </div>
            </div>
            <div className="mt-4">
              <span className="font-medium text-gray-700 dark:text-gray-300">Description:</span>
              <div className="text-gray-900 dark:text-gray-100 mt-1">{selectedLog.description}</div>
            </div>
            {selectedLog.ip_address && (
              <div className="mt-4">
                <span className="font-medium text-gray-700 dark:text-gray-300">IP Address:</span>
                <div className="text-gray-900 dark:text-gray-100 mt-1">{selectedLog.ip_address}</div>
              </div>
            )}
          </div>

          {/* JSON Comparison */}
          {(selectedLog.old_value || selectedLog.new_value) && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Before */}
              <div>
                <h4 className="text-md font-medium text-red-600 dark:text-red-400 mb-2">Before</h4>
                <textarea
                  className="w-full h-96 p-3 border border-red-200 dark:border-red-700 rounded-md bg-red-50 dark:bg-red-900 text-sm font-mono text-gray-900 dark:text-gray-100"
                  value={selectedLog.old_value ? JSON.stringify(selectedLog.old_value, null, 2) : ''}
                  readOnly
                  placeholder="No previous value"
                />
              </div>

              {/* After */}
              <div>
                <h4 className="text-md font-medium text-green-600 dark:text-green-400 mb-2">After</h4>
                <textarea
                  className="w-full h-96 p-3 border border-green-200 dark:border-green-700 rounded-md bg-green-50 dark:bg-green-900 text-sm font-mono text-gray-900 dark:text-gray-100"
                  value={selectedLog.new_value ? JSON.stringify(selectedLog.new_value, null, 2) : ''}
                  readOnly
                  placeholder="No new value"
                />
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">Activity Logs</h1>
        <p className="text-gray-600 dark:text-gray-400">Monitor all system activities</p>
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Action Type
            </label>
            <select
              value={filters.action_type || ''}
              onChange={(e) => handleFilterChange('action_type', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              <option value="">All</option>
              {Object.entries(actionTypeLabels).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              User
            </label>
            <select
              value={(filters as any).username || ''}
              onChange={(e) => handleFilterChange('username' as keyof ActionLogFilters, e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              <option value="">All</option>
              {userOptions.map(u => (
                <option key={u.id} value={u.username}>
                  {u.username}{u.full_name ? ` — ${u.full_name}` : ''}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Resource Type
            </label>
            <select
              value={filters.resource_type || ''}
              onChange={(e) => handleFilterChange('resource_type', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              <option value="">All</option>
              {Object.entries(resourceTypeLabels).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              From Date
            </label>
            <input
              type="date"
              value={filters.start_date || ''}
              onChange={(e) => handleFilterChange('start_date', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              To Date
            </label>
            <input
              type="date"
              value={filters.end_date || ''}
              onChange={(e) => handleFilterChange('end_date', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
          </div>
        </div>

        <div className="mt-4 flex justify-end">
          <button
            onClick={() => {
              setFilters({});
              setCurrentPage(1);
            }}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md"
          >
            Clear Filters
          </button>
        </div>
      </div>

      {/* Logs Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        {isLoading ? (
          <IndustrialLoading template={loadingTemplate} />
        ) : logs.length === 0 ? (
          <div className="p-8 text-center text-gray-500 dark:text-gray-400">
            No activity logs found
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    User
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Action
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Resource
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Details
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {logs.map((log, index) => (
                  <tr key={log.id} className={`${index % 2 === 0 ? 'bg-white dark:bg-gray-800' : 'bg-gray-50 dark:bg-gray-700'} hover:bg-gray-100 dark:hover:bg-gray-600`}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100 align-top">
                      {formatDateTime(log.timestamp)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100 align-top">
                      <div className="font-medium">{log.username}</div>
                      <div className="text-gray-500 dark:text-gray-400 text-xs mt-1">{usernameToFullname[log.username] ?? '-'}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap align-top">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getActionTypeColor(log.action_type)}`}>
                        {actionTypeLabels[log.action_type]}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap align-top">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getResourceTypeColor(log.resource_type)}`}>
                        {resourceTypeLabels[log.resource_type]}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900 dark:text-gray-100 max-w-xs align-top">
                      <div className="truncate" title={log.description}>
                        {log.description}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400 align-top max-w-md">
                      <div className="space-y-1">
                        {log.old_value && (
                          <div>
                            <span className="font-medium text-red-600 dark:text-red-400">Before: </span>
                            <span className="text-xs break-all">
                              {expandedDetails.has(log.id)
                                ? formatValue(log.old_value)
                                : truncateText(formatValue(log.old_value))}
                            </span>
                          </div>
                        )}
                        {log.new_value && (
                          <div>
                            <span className="font-medium text-green-600 dark:text-green-400">After: </span>
                            <span className="text-xs break-all">
                              {expandedDetails.has(log.id)
                                ? formatValue(log.new_value)
                                : truncateText(formatValue(log.new_value))}
                            </span>
                          </div>
                        )}
                        {log.ip_address && (
                          <div className="text-xs text-gray-400 dark:text-gray-500">
                            IP: {log.ip_address}
                          </div>
                        )}
                        {shouldShowDetailsToggle(log) && (
                          <button
                            onClick={() => toggleDetailsExpansion(log.id)}
                            className="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 mt-1 font-medium"
                          >
                            {expandedDetails.has(log.id) ? 'Show less' : 'Show more'}
                          </button>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium align-top">
                      <button
                        onClick={() => handleViewDetails(log)}
                        className="text-blue-600 dark:text-blue-400 hover:text-blue-900 dark:hover:text-blue-300 px-3 py-1 rounded-md text-sm font-medium bg-blue-50 dark:bg-blue-900 hover:bg-blue-100 dark:hover:bg-blue-800 transition-colors"
                      >
                        Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

      </div>

      {/* Pagination */}
      {logs.length > 0 && (
        <div className="mt-6 flex justify-between items-center">
          <div className="text-sm text-gray-700 dark:text-gray-300">
            Showing {((currentPage - 1) * logsPerPage) + 1} - {Math.min(currentPage * logsPerPage, totalLogs)} of {totalLogs} results
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800"
            >
              Previous
            </button>
            <span className="px-3 py-1 text-sm text-gray-700 dark:text-gray-300">
              Page {currentPage}
            </span>
            <button
              onClick={() => setCurrentPage(prev => prev + 1)}
              disabled={logs.length < logsPerPage}
              className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Logs;