import React, { useState, useEffect } from 'react';
import { camerasAPI } from '../services/api';
import { useToast } from '../contexts/ToastContext';
import ConfirmDialog from './ConfirmDialog';
import './Dashboard.css';

export default function CameraManagement() {
  const toast = useToast();
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIds, setSelectedIds] = useState([]);
  const [selectAll, setSelectAll] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [editingCamera, setEditingCamera] = useState(null);
  const [statistics, setStatistics] = useState({
    total: 0,
    connected: 0,
    active: 0
  });

  const [confirmDialog, setConfirmDialog] = useState({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning',
    onConfirm: null
  });

  const [formData, setFormData] = useState({
    camera_id: '',
    model_name: '',
    serial_number: '',
    resolution_width: 1920,
    resolution_height: 1200,
    is_connected: false,
    is_active: true,
    ip_address: '',
    location: '',
    description: '',
    max_frame_rate: 40.0,
    pixel_format: 'Mono8'
  });

  useEffect(() => {
    fetchCameras();
  }, []);

  const fetchCameras = async () => {
    try {
      setLoading(true);
      const [allCameras, countData] = await Promise.all([
        camerasAPI.getAllCameras(0, 100),
        camerasAPI.getCamerasCount()
      ]);
      
      setCameras(allCameras);
      
      // Calculate statistics
      const connectedCount = allCameras.filter(c => c.is_connected).length;
      const activeCount = allCameras.filter(c => c.is_active).length;
      
      setStatistics({
        total: countData.count || allCameras.length,
        connected: connectedCount,
        active: activeCount
      });
      
      setError(null);
    } catch (err) {
      console.error('Error fetching cameras:', err);
      setError('Failed to load cameras');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
  };

  const filteredCameras = cameras.filter(camera => {
    const search = searchTerm.toLowerCase();
    return (
      camera.camera_id?.toLowerCase().includes(search) ||
      camera.model_name?.toLowerCase().includes(search) ||
      camera.serial_number?.toLowerCase().includes(search) ||
      camera.location?.toLowerCase().includes(search) ||
      camera.ip_address?.toLowerCase().includes(search)
    );
  });

  const handleSelectAll = (e) => {
    e.stopPropagation();
    const checked = e.target.checked;
    console.log('handleSelectAll called:', checked);
    setSelectAll(checked);
    if (checked) {
      setSelectedIds(filteredCameras.map(c => c.camera_id));
    } else {
      setSelectedIds([]);
    }
  };

  const handleSelectCamera = (cameraId) => {
    console.log('handleSelectCamera called:', cameraId);
    setSelectedIds(prev => {
      if (prev.includes(cameraId)) {
        const newSelected = prev.filter(id => id !== cameraId);
        setSelectAll(false);
        return newSelected;
      } else {
        const newSelected = [...prev, cameraId];
        if (newSelected.length === filteredCameras.length) {
          setSelectAll(true);
        }
        return newSelected;
      }
    });
  };

  const handleDeleteCamera = async (cameraId) => {
    const camera = cameras.find(c => c.camera_id === cameraId);
    setConfirmDialog({
      isOpen: true,
      title: 'Delete Camera',
      message: `Are you sure you want to delete camera "${camera?.camera_id || cameraId}"?\n\nThis action cannot be undone.`,
      type: 'danger',
      onConfirm: async () => {
        try {
          await camerasAPI.deleteCamera(cameraId);
          await fetchCameras();
          setSelectedIds([]);
          setSelectAll(false);
          toast.success('Camera deleted successfully!');
        } catch (err) {
          console.error('Error deleting camera:', err);
          toast.error('Failed to delete camera');
        }
      }
    });
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.length === 0) return;
    
    setConfirmDialog({
      isOpen: true,
      title: 'Delete Multiple Cameras',
      message: `Are you sure you want to delete ${selectedIds.length} camera(s)?\n\nThis action cannot be undone.`,
      type: 'danger',
      onConfirm: async () => {
        try {
          await Promise.all(selectedIds.map(id => camerasAPI.deleteCamera(id)));
          await fetchCameras();
          setSelectedIds([]);
          setSelectAll(false);
          toast.success(`${selectedIds.length} camera(s) deleted successfully!`);
        } catch (err) {
          console.error('Error deleting cameras:', err);
          toast.error('Failed to delete cameras');
        }
      }
    });
  };

  const handleToggleConnection = async (camera) => {
    try {
      await camerasAPI.updateCameraConnection(camera.camera_id, !camera.is_connected);
      await fetchCameras();
      toast.success(`Camera ${camera.is_connected ? 'disconnected' : 'connected'} successfully!`);
    } catch (err) {
      console.error('Error toggling camera connection:', err);
      toast.error('Failed to update camera connection status');
    }
  };

  const openCreateModal = () => {
    setEditingCamera(null);
    setFormData({
      camera_id: '',
      model_name: '',
      serial_number: '',
      resolution_width: 1920,
      resolution_height: 1200,
      is_connected: false,
      is_active: true,
      ip_address: '',
      location: '',
      description: '',
      max_frame_rate: 40.0,
      pixel_format: 'Mono8'
    });
    setShowModal(true);
  };

  const openEditModal = (camera) => {
    setEditingCamera(camera);
    setFormData({
      camera_id: camera.camera_id,
      model_name: camera.model_name,
      serial_number: camera.serial_number,
      resolution_width: camera.resolution_width,
      resolution_height: camera.resolution_height,
      is_connected: camera.is_connected,
      is_active: camera.is_active,
      ip_address: camera.ip_address || '',
      location: camera.location || '',
      description: camera.description || '',
      max_frame_rate: camera.max_frame_rate || 40.0,
      pixel_format: camera.pixel_format || 'Mono8'
    });
    setShowModal(true);
  };

  const handleFormChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : 
              (name === 'resolution_width' || name === 'resolution_height') ? parseInt(value) || 0 :
              name === 'max_frame_rate' ? parseFloat(value) || 0 :
              value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      if (editingCamera) {
        await camerasAPI.updateCamera(editingCamera.camera_id, formData);
        toast.success('Camera updated successfully!');
      } else {
        await camerasAPI.createCamera(formData);
        toast.success('Camera created successfully!');
      }
      
      setShowModal(false);
      await fetchCameras();
    } catch (err) {
      console.error('Error saving camera:', err);
      toast.error(err.response?.data?.detail || 'Failed to save camera');
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading cameras...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <p className="error-message">{error}</p>
        <button onClick={fetchCameras} className="retry-button">Retry</button>
      </div>
    );
  }

  return (
    <div className="camera-management">
      <div className="section-header">
        <div>
          <h2>Camera Management</h2>
          <p>Manage your camera devices and configurations</p>
        </div>
        <button className="create-button" onClick={openCreateModal}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          Add Camera
        </button>
      </div>

      {/* Statistics Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <rect x="2" y="4" width="20" height="16" rx="2" stroke="white" strokeWidth="2"/>
              <circle cx="12" cy="12" r="3" stroke="white" strokeWidth="2"/>
            </svg>
          </div>
          <div className="stat-content">
            <div className="stat-label">Total Cameras</div>
            <div className="stat-value">{statistics.total}</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)' }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="white" strokeWidth="2"/>
              <path d="M8 12l2 2 4-4" stroke="white" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
          <div className="stat-content">
            <div className="stat-label">Connected</div>
            <div className="stat-value">{statistics.connected}</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)' }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="stat-content">
            <div className="stat-label">Active</div>
            <div className="stat-value">{statistics.active}</div>
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
            placeholder="Search by Camera ID, Model, Serial, Location, IP..."
            value={searchTerm}
            onChange={handleSearch}
          />
        </div>
        
        {selectedIds.length > 0 && (
          <button className="action-btn delete" onClick={handleDeleteSelected}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Delete Selected ({selectedIds.length})
          </button>
        )}
      </div>

      {/* Cameras Table */}
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ width: '50px' }} onClick={(e) => e.stopPropagation()}>
                <input
                  type="checkbox"
                  checked={selectAll}
                  onChange={(e) => {
                    e.stopPropagation();
                    handleSelectAll(e);
                  }}
                />
              </th>
              <th>Camera ID</th>
              <th>Model</th>
              <th>Serial Number</th>
              <th>Resolution</th>
              <th>Status</th>
              <th>Location</th>
              <th>IP Address</th>
              <th>Frame Rate</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredCameras.length === 0 ? (
              <tr>
                <td colSpan="10" style={{ textAlign: 'center', padding: '2rem' }}>
                  {searchTerm ? 'No cameras found matching your search' : 'No cameras available'}
                </td>
              </tr>
            ) : (
              filteredCameras.map((camera) => (
                <tr
                  key={camera.camera_id}
                  className={selectedIds.includes(camera.camera_id) ? 'selected-row' : ''}
                >
                  <td onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(camera.camera_id)}
                      onChange={(e) => {
                        e.stopPropagation();
                        handleSelectCamera(camera.camera_id);
                      }}
                    />
                  </td>
                  <td>
                    <span className="camera-id">{camera.camera_id}</span>
                  </td>
                  <td>{camera.model_name}</td>
                  <td>
                    <code className="serial-number">{camera.serial_number}</code>
                  </td>
                  <td>{camera.resolution_width} × {camera.resolution_height}</td>
                  <td>
                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                      <span className={`status-badge ${camera.is_connected ? 'success' : 'error'}`}>
                        {camera.is_connected ? 'Connected' : 'Disconnected'}
                      </span>
                      {camera.is_active && (
                        <span className="status-badge info">Active</span>
                      )}
                    </div>
                  </td>
                  <td>{camera.location || '-'}</td>
                  <td>
                    {camera.ip_address ? (
                      <code className="ip-address">{camera.ip_address}</code>
                    ) : '-'}
                  </td>
                  <td>{camera.max_frame_rate} fps</td>
                  <td>
                    <div className="action-buttons">
                      <button
                        className={`action-btn ${camera.is_connected ? 'disconnect' : 'connect'}`}
                        onClick={() => handleToggleConnection(camera)}
                        title={camera.is_connected ? 'Disconnect' : 'Connect'}
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                          <path d="M12 6v6l4 2" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                      </button>
                      <button
                        className="action-btn edit"
                        onClick={() => openEditModal(camera)}
                        title="Edit"
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                          <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" stroke="currentColor" strokeWidth="2"/>
                          <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                      </button>
                      <button
                        className="action-btn delete"
                        onClick={() => handleDeleteCamera(camera.camera_id)}
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

      {/* Camera Form Modal */}
      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal-content camera-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{editingCamera ? 'Edit Camera' : 'Add New Camera'}</h3>
              <button className="close-button" onClick={() => setShowModal(false)}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2"/>
                </svg>
              </button>
            </div>

            <form onSubmit={handleSubmit}>
              <div className="form-grid">
                <div className="form-group">
                  <label>Camera ID *</label>
                  <input
                    type="text"
                    name="camera_id"
                    value={formData.camera_id}
                    onChange={handleFormChange}
                    placeholder="CAM001"
                    required
                    disabled={!!editingCamera}
                  />
                </div>

                <div className="form-group">
                  <label>Model Name *</label>
                  <input
                    type="text"
                    name="model_name"
                    value={formData.model_name}
                    onChange={handleFormChange}
                    placeholder="Basler acA1920-40gm"
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Serial Number *</label>
                  <input
                    type="text"
                    name="serial_number"
                    value={formData.serial_number}
                    onChange={handleFormChange}
                    placeholder="40123456"
                    required
                  />
                </div>

                <div className="form-group">
                  <label>IP Address</label>
                  <input
                    type="text"
                    name="ip_address"
                    value={formData.ip_address}
                    onChange={handleFormChange}
                    placeholder="192.168.1.100"
                  />
                </div>

                <div className="form-group">
                  <label>Resolution Width *</label>
                  <input
                    type="number"
                    name="resolution_width"
                    value={formData.resolution_width}
                    onChange={handleFormChange}
                    placeholder="1920"
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Resolution Height *</label>
                  <input
                    type="number"
                    name="resolution_height"
                    value={formData.resolution_height}
                    onChange={handleFormChange}
                    placeholder="1200"
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Max Frame Rate (fps)</label>
                  <input
                    type="number"
                    step="0.1"
                    name="max_frame_rate"
                    value={formData.max_frame_rate}
                    onChange={handleFormChange}
                    placeholder="40.0"
                  />
                </div>

                <div className="form-group">
                  <label>Pixel Format</label>
                  <select
                    name="pixel_format"
                    value={formData.pixel_format}
                    onChange={handleFormChange}
                  >
                    <option value="Mono8">Mono8</option>
                    <option value="Mono12">Mono12</option>
                    <option value="RGB8">RGB8</option>
                    <option value="BGR8">BGR8</option>
                    <option value="YUV422">YUV422</option>
                  </select>
                </div>

                <div className="form-group full-width">
                  <label>Location</label>
                  <input
                    type="text"
                    name="location"
                    value={formData.location}
                    onChange={handleFormChange}
                    placeholder="Production Line 1 - Position A"
                  />
                </div>

                <div className="form-group full-width">
                  <label>Description</label>
                  <textarea
                    name="description"
                    value={formData.description}
                    onChange={handleFormChange}
                    placeholder="Camera description and usage details..."
                    rows="3"
                  />
                </div>

                <div className="form-group checkbox-group">
                  <label>
                    <input
                      type="checkbox"
                      name="is_connected"
                      checked={formData.is_connected}
                      onChange={handleFormChange}
                    />
                    <span>Connected</span>
                  </label>
                </div>

                <div className="form-group checkbox-group">
                  <label>
                    <input
                      type="checkbox"
                      name="is_active"
                      checked={formData.is_active}
                      onChange={handleFormChange}
                    />
                    <span>Active</span>
                  </label>
                </div>
              </div>

              <div className="modal-actions">
                <button type="button" className="cancel-button" onClick={() => setShowModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="submit-button">
                  {editingCamera ? 'Update Camera' : 'Create Camera'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Confirmation Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={() => setConfirmDialog({ ...confirmDialog, isOpen: false })}
        onConfirm={confirmDialog.onConfirm}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
      />
    </div>
  );
}
