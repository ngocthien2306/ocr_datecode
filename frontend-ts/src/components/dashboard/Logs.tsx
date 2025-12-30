import React, { useState, useEffect } from 'react';
import { actionLogsAPI } from '@/services/api';
import { useToast } from '@/contexts/ToastContext';
import { IndustrialLoading } from '@/components/ui/IndustrialLoading';
import type { ActionLog, ActionLogFilters, ActionType, ResourceType } from '@/types';

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

  const handleFilterChange = (field: keyof ActionLogFilters, value: string) => {
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

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Activity Logs</h1>
        <p className="text-gray-600">Monitor all system activities</p>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Action Type
            </label>
            <select
              value={filters.action_type || ''}
              onChange={(e) => handleFilterChange('action_type', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All</option>
              {Object.entries(actionTypeLabels).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Resource Type
            </label>
            <select
              value={filters.resource_type || ''}
              onChange={(e) => handleFilterChange('resource_type', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All</option>
              {Object.entries(resourceTypeLabels).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              From Date
            </label>
            <input
              type="date"
              value={filters.start_date || ''}
              onChange={(e) => handleFilterChange('start_date', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              To Date
            </label>
            <input
              type="date"
              value={filters.end_date || ''}
              onChange={(e) => handleFilterChange('end_date', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <div className="mt-4 flex justify-end">
          <button
            onClick={() => {
              setFilters({});
              setCurrentPage(1);
            }}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md"
          >
            Clear Filters
          </button>
        </div>
      </div>

      {/* Logs Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        {isLoading ? (
          <IndustrialLoading template={loadingTemplate} />
        ) : logs.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            No activity logs found
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    User
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Action
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Resource
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Details
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {logs.map((log) => (
                  <tr key={log.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatDateTime(log.timestamp)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <div className="font-medium">{log.username}</div>
                      <div className="text-gray-500 text-xs">{log.user_id}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getActionTypeColor(log.action_type)}`}>
                        {actionTypeLabels[log.action_type]}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getResourceTypeColor(log.resource_type)}`}>
                        {resourceTypeLabels[log.resource_type]}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900 max-w-xs">
                      <div className="truncate" title={log.description}>
                        {log.description}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      <div className="space-y-1">
                        {log.old_value && (
                          <div>
                            <span className="font-medium text-red-600">Before: </span>
                            <span className="text-xs">{formatValue(log.old_value)}</span>
                          </div>
                        )}
                        {log.new_value && (
                          <div>
                            <span className="font-medium text-green-600">After: </span>
                            <span className="text-xs">{formatValue(log.new_value)}</span>
                          </div>
                        )}
                        {log.ip_address && (
                          <div className="text-xs text-gray-400">
                            IP: {log.ip_address}
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <button
                        onClick={() => setSelectedLog(log)}
                        className="text-blue-600 hover:text-blue-900 px-3 py-1 rounded-md text-sm font-medium bg-blue-50 hover:bg-blue-100 transition-colors"
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
          <div className="text-sm text-gray-700">
            Showing {((currentPage - 1) * logsPerPage) + 1} - {Math.min(currentPage * logsPerPage, totalLogs)} of {totalLogs} results
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 text-sm border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
            >
              Previous
            </button>
            <span className="px-3 py-1 text-sm text-gray-700">
              Page {currentPage}
            </span>
            <button
              onClick={() => setCurrentPage(prev => prev + 1)}
              disabled={logs.length < logsPerPage}
              className="px-3 py-1 text-sm border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Detail Modal */}
      {selectedLog && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50" onClick={() => setSelectedLog(null)}>
          <div className="relative top-20 mx-auto p-5 border w-11/12 max-w-6xl shadow-lg rounded-md bg-white" onClick={e => e.stopPropagation()}>
            <div className="mt-3">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium text-gray-900">
                  Action Log Details
                </h3>
                <button
                  onClick={() => setSelectedLog(null)}
                  className="text-gray-400 hover:text-gray-600 text-2xl"
                >
                  ×
                </button>
              </div>

              {/* Log Info */}
              <div className="mb-6 bg-gray-50 p-4 rounded-lg">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="font-medium text-gray-700">Time:</span>
                    <div className="text-gray-900">{formatDateTime(selectedLog.timestamp)}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">User:</span>
                    <div className="text-gray-900">{selectedLog.username}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Action:</span>
                    <div className="text-gray-900">{actionTypeLabels[selectedLog.action_type]}</div>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Resource:</span>
                    <div className="text-gray-900">{resourceTypeLabels[selectedLog.resource_type]}</div>
                  </div>
                </div>
                <div className="mt-2">
                  <span className="font-medium text-gray-700">Description:</span>
                  <div className="text-gray-900 mt-1">{selectedLog.description}</div>
                </div>
                {selectedLog.ip_address && (
                  <div className="mt-2">
                    <span className="font-medium text-gray-700">IP Address:</span>
                    <div className="text-gray-900 mt-1">{selectedLog.ip_address}</div>
                  </div>
                )}
              </div>

              {/* JSON Comparison */}
              {(selectedLog.old_value || selectedLog.new_value) && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Before */}
                  <div>
                    <h4 className="text-md font-medium text-red-600 mb-2">Before</h4>
                    <textarea
                      className="w-full h-64 p-3 border border-red-200 rounded-md bg-red-50 text-sm font-mono"
                      value={selectedLog.old_value ? JSON.stringify(selectedLog.old_value, null, 2) : ''}
                      readOnly
                      placeholder="No previous value"
                    />
                  </div>

                  {/* After */}
                  <div>
                    <h4 className="text-md font-medium text-green-600 mb-2">After</h4>
                    <textarea
                      className="w-full h-64 p-3 border border-green-200 rounded-md bg-green-50 text-sm font-mono"
                      value={selectedLog.new_value ? JSON.stringify(selectedLog.new_value, null, 2) : ''}
                      readOnly
                      placeholder="No new value"
                    />
                  </div>
                </div>
              )}

              {/* Close Button */}
              <div className="flex justify-end mt-6">
                <button
                  onClick={() => setSelectedLog(null)}
                  className="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Logs;