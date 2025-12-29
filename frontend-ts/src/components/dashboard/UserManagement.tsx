import React, { useState, useEffect, ChangeEvent, FormEvent, useRef } from 'react';
import { usersAPI, uploadAPI } from '@/services/api';
import { useToast } from '@/contexts/ToastContext';
import ConfirmDialog from '@/components/shared/ConfirmDialog';
import type { User, UserCreate, UserUpdate } from '@/types';
import { API_BASE_URL } from '@/config/api';

interface ConfirmDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  onConfirm: (() => void) | null;
}

interface UserFormData {
  username: string;
  email: string;
  full_name: string;
  phone_number: string;
  avatar_url: string;
  password: string;
  role: 'admin' | 'operator' | 'supervisor' | 'viewer';
  is_active: boolean;
}

interface ValidationErrors {
  username?: string;
  email?: string;
  full_name?: string;
  phone_number?: string;
  password?: string;
}

const UserManagement: React.FC = () => {
  const toast = useToast();
  const [users, setUsers] = useState<User[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [selectAll, setSelectAll] = useState(false);
  const [viewMode, setViewMode] = useState<'list' | 'form'>('list');

  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning',
    onConfirm: null
  });

  const [formData, setFormData] = useState<UserFormData>({
    username: '',
    email: '',
    full_name: '',
    phone_number: '',
    avatar_url: '',
    password: '',
    role: 'operator',
    is_active: true
  });

  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({});
  const [editingUser, setEditingUser] = useState<(User & { password?: string }) | null>(null);
  const [avatarPreview, setAvatarPreview] = useState<string>('');
  const [uploadingAvatar, setUploadingAvatar] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      setIsLoading(true);
      const data = await usersAPI.getAllUsers();
      setUsers(data);
    } catch (err) {
      console.error('Error fetching users:', err);
      toast.error('Failed to load users');
    } finally {
      setIsLoading(false);
    }
  };

  const openCreateForm = () => {
    setEditingUser(null);
    setFormData({
      username: '',
      email: '',
      full_name: '',
      phone_number: '',
      avatar_url: '',
      password: '',
      role: 'operator',
      is_active: true
    });
    setValidationErrors({});
    setAvatarPreview('');
    setViewMode('form');
  };

  const handleEditClick = (user: User) => {
    setEditingUser(user);
    setFormData({
      username: user.username,
      email: user.email,
      full_name: user.full_name,
      phone_number: user.phone_number || '',
      avatar_url: user.avatar_url || '',
      password: '',
      role: user.role,
      is_active: user.is_active
    });
    setValidationErrors({});
    setAvatarPreview(user.avatar_url || '');
    setViewMode('form');
  };

  const handleBackToList = () => {
    setViewMode('list');
    setEditingUser(null);
    setValidationErrors({});
    setAvatarPreview('');
  };

  const handleAvatarClick = () => {
    fileInputRef.current?.click();
  };

  const handleAvatarChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast.error('Please select an image file');
      return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      toast.error('Image size must be less than 5MB');
      return;
    }

    try {
      setUploadingAvatar(true);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setAvatarPreview(reader.result as string);
      };
      reader.readAsDataURL(file);

      // Upload to server
      const avatarUrl = await uploadAPI.uploadAvatar(file);

      setFormData(prev => ({
        ...prev,
        avatar_url: avatarUrl
      }));

      toast.success('Avatar uploaded successfully!');
    } catch (err: any) {
      console.error('Error uploading avatar:', err);
      toast.error(err.response?.data?.detail || 'Failed to upload avatar');
      setAvatarPreview('');
    } finally {
      setUploadingAvatar(false);
    }
  };

  const handleRemoveAvatar = () => {
    setAvatarPreview('');
    setFormData(prev => ({
      ...prev,
      avatar_url: ''
    }));
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const validateField = (name: string, value: string): string | undefined => {
    switch (name) {
      case 'username':
        if (!value.trim()) return 'Username is required';
        if (value.length < 3) return 'Username must be at least 3 characters';
        if (value.length > 50) return 'Username must not exceed 50 characters';
        break;

      case 'full_name':
        if (!value.trim()) return 'Full name is required';
        if (value.length > 100) return 'Full name must not exceed 100 characters';
        break;

      case 'email':
        if (value.trim()) {
          const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
          if (!emailRegex.test(value)) return 'Invalid email format';
        }
        if (value.length > 100) return 'Email must not exceed 100 characters';
        break;

      case 'phone_number':
        if (value && !/^\d*$/.test(value)) return 'Phone number must contain only digits';
        if (value.length > 15) return 'Phone number must not exceed 15 digits';
        break;

      case 'password':
        if (!editingUser) {
          if (!value.trim()) return 'Password is required';
          if (value.length < 6) return 'Password must be at least 6 characters';
        }
        if (value.length > 100) return 'Password must not exceed 100 characters';
        break;
    }
    return undefined;
  };

  const handleFormChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const target = e.target as HTMLInputElement;
    const { name, value, type } = target;
    const checked = target.checked;

    let newValue = value;

    // Phone number: only allow digits
    if (name === 'phone_number' && value) {
      newValue = value.replace(/\D/g, '');
      if (newValue.length > 15) {
        newValue = newValue.slice(0, 15);
      }
    }

    // Update form data
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : newValue
    }));

    // Validate field
    if (type !== 'checkbox') {
      const error = validateField(name, newValue);
      setValidationErrors(prev => ({
        ...prev,
        [name]: error
      }));
    }
  };

  const isFormValid = (): boolean => {
    // Check if there are any validation errors
    const hasErrors = Object.values(validationErrors).some(error => error !== undefined);
    if (hasErrors) return false;

    // Check required fields
    if (!formData.username.trim()) return false;
    if (!formData.full_name.trim()) return false;
    if (!editingUser && !formData.password.trim()) return false;

    return true;
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!isFormValid()) {
      toast.error('Please fix all validation errors before submitting');
      return;
    }

    try {
      if (editingUser) {
        const updateData: UserUpdate = {
          email: formData.email,
          full_name: formData.full_name,
          phone_number: formData.phone_number || undefined,
          avatar_url: formData.avatar_url || undefined,
          role: formData.role,
          is_active: formData.is_active
        };

        await usersAPI.updateUser((editingUser as any)._id || (editingUser as any).id, updateData);
        toast.success('User updated successfully!');
      } else {
        await usersAPI.createUser(formData as UserCreate);
        toast.success('User created successfully!');
      }

      setViewMode('list');
      setEditingUser(null);
      await fetchUsers();
    } catch (err: any) {
      console.error('Error saving user:', err);
      toast.error(err.response?.data?.detail || 'Failed to save user');
    }
  };

  const handleDeleteUser = async (userId: string, username: string) => {
    setConfirmDialog({
      isOpen: true,
      title: 'Delete User',
      message: `Are you sure you want to delete user "${username}"?\n\nThis action cannot be undone.`,
      type: 'danger',
      onConfirm: async () => {
        try {
          await usersAPI.deleteUser(userId);
          setSelectedIds([]);
          setSelectAll(false);
          await fetchUsers();
          toast.success(`User "${username}" deleted successfully!`);
        } catch (err: any) {
          console.error('Error deleting user:', err);
          toast.error(err.response?.data?.detail || 'Failed to delete user');
        }
      }
    });
  };

  const handleSelectAll = (e: ChangeEvent<HTMLInputElement>) => {
    e.stopPropagation();
    const checked = e.target.checked;
    setSelectAll(checked);
    if (checked) {
      setSelectedIds(filteredUsers.map(u => (u as any)._id || (u as any).id));
    } else {
      setSelectedIds([]);
    }
  };

  const handleSelectUser = (userId: string) => {
    setSelectedIds(prev => {
      if (prev.includes(userId)) {
        const newSelected = prev.filter(id => id !== userId);
        setSelectAll(false);
        return newSelected;
      } else {
        const newSelected = [...prev, userId];
        if (newSelected.length === filteredUsers.length) {
          setSelectAll(true);
        }
        return newSelected;
      }
    });
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.length === 0) return;
    
    setConfirmDialog({
      isOpen: true,
      title: 'Delete Multiple Users',
      message: `Are you sure you want to delete ${selectedIds.length} user(s)?\n\nThis action cannot be undone.`,
      type: 'danger',
      onConfirm: async () => {
        try {
          await Promise.all(selectedIds.map(id => usersAPI.deleteUser(id)));
          setSelectedIds([]);
          setSelectAll(false);
          await fetchUsers();
          toast.success(`${selectedIds.length} user(s) deleted successfully!`);
        } catch (err) {
          console.error('Error deleting users:', err);
          toast.error('Failed to delete some users');
        }
      }
    });
  };

  const filteredUsers = users.filter(user =>
    user.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.username?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.email?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.role?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Form Mode
  if (viewMode === 'form') {
    return (
      <div className="user-management-page">
        <div className="camera-form-view">
          <div className="form-header">
            <button className="back-button" onClick={handleBackToList}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <path d="M19 12H5M5 12l7 7M5 12l7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Back to User List
            </button>
            <h2>{editingUser ? 'Edit User' : 'Add New User'}</h2>
          </div>

          <div className="form-content">
            <form onSubmit={handleSubmit}>
              {/* Avatar Upload Section */}
              <div className="avatar-upload-section">
                <div className="avatar-preview-container">
                  {avatarPreview ? (
                    <div className="avatar-preview">
                      <img src={`${API_BASE_URL}${avatarPreview}`} alt="Avatar preview" />
                      <button
                        type="button"
                        className="avatar-remove-btn"
                        onClick={handleRemoveAvatar}
                        disabled={uploadingAvatar}
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                          <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                      </button>
                    </div>
                  ) : (
                    <div className="avatar-placeholder" onClick={handleAvatarClick}>
                      {uploadingAvatar ? (
                        <div className="avatar-uploading">
                          <div className="spinner"></div>
                          <span>Uploading...</span>
                        </div>
                      ) : (
                        <>
                          <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
                            <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" stroke="currentColor" strokeWidth="2"/>
                            <circle cx="12" cy="7" r="4" stroke="currentColor" strokeWidth="2"/>
                          </svg>
                          <span>Click to upload avatar</span>
                          <span className="avatar-hint">PNG, JPG up to 5MB</span>
                        </>
                      )}
                    </div>
                  )}
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleAvatarChange}
                  style={{ display: 'none' }}
                />
                {avatarPreview && !uploadingAvatar && (
                  <button
                    type="button"
                    className="avatar-change-btn"
                    onClick={handleAvatarClick}
                  >
                    Change Avatar
                  </button>
                )}
              </div>

              <div className="form-grid">
                <div className="form-group">
                  <label>Username *</label>
                  <input
                    type="text"
                    name="username"
                    value={formData.username}
                    onChange={handleFormChange}
                    placeholder="Enter username"
                    disabled={!!editingUser}
                    maxLength={50}
                    className={validationErrors.username ? 'error' : ''}
                  />
                  {validationErrors.username && (
                    <span className="error-message">{validationErrors.username}</span>
                  )}
                </div>

                <div className="form-group">
                  <label>Full Name *</label>
                  <input
                    type="text"
                    name="full_name"
                    value={formData.full_name}
                    onChange={handleFormChange}
                    placeholder="Enter full name"
                    maxLength={100}
                    className={validationErrors.full_name ? 'error' : ''}
                  />
                  {validationErrors.full_name && (
                    <span className="error-message">{validationErrors.full_name}</span>
                  )}
                </div>

                <div className="form-group">
                  <label>Email</label>
                  <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleFormChange}
                    placeholder="Enter email"
                    maxLength={100}
                    className={validationErrors.email ? 'error' : ''}
                  />
                  {validationErrors.email && (
                    <span className="error-message">{validationErrors.email}</span>
                  )}
                </div>

                <div className="form-group">
                  <label>Phone Number</label>
                  <input
                    type="tel"
                    name="phone_number"
                    value={formData.phone_number}
                    onChange={handleFormChange}
                    placeholder="Enter phone number (digits only)"
                    maxLength={15}
                    inputMode="numeric"
                    pattern="[0-9]*"
                    className={validationErrors.phone_number ? 'error' : ''}
                  />
                  {validationErrors.phone_number && (
                    <span className="error-message">{validationErrors.phone_number}</span>
                  )}
                </div>

                {!editingUser && (
                  <div className="form-group">
                    <label>Password *</label>
                    <input
                      type="password"
                      name="password"
                      value={formData.password}
                      onChange={handleFormChange}
                      placeholder="Enter password (min 6 characters)"
                      maxLength={100}
                      className={validationErrors.password ? 'error' : ''}
                    />
                    {validationErrors.password && (
                      <span className="error-message">{validationErrors.password}</span>
                    )}
                  </div>
                )}

                <div className="form-group">
                  <label>Role</label>
                  <select
                    name="role"
                    value={formData.role}
                    onChange={handleFormChange}
                  >
                    <option value="viewer">Viewer</option>
                    <option value="operator">Operator</option>
                    <option value="supervisor">Supervisor</option>
                    <option value="admin">Admin</option>
                  </select>
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

              <div className="form-actions">
                <button type="button" className="cancel-button" onClick={handleBackToList}>
                  Cancel
                </button>
                <button
                  type="submit"
                  className="submit-button"
                  disabled={!isFormValid()}
                >
                  {editingUser ? 'Update User' : 'Create User'}
                </button>
              </div>
            </form>
          </div>
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

  // List Mode
  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading users...</p>
      </div>
    );
  }

  return (
    <div className="user-management-page">
      <div className="section-header">
        <h1>User Management</h1>
        <button className="dashboard-btn" onClick={openCreateForm}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <line x1="12" y1="5" x2="12" y2="19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          Add User
        </button>
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
            placeholder="Search users by name, email, or role..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        {selectedIds.length > 0 && (
          <button className="action-btn delete" onClick={handleDeleteSelected}>
            Delete Selected ({selectedIds.length})
          </button>
        )}
      </div>

      {/* Users Table */}
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
              <th>Username</th>
              <th>Name</th>
              <th>Email</th>
              <th>Role</th>
              <th>Status</th>
              <th>Last Login</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
              {filteredUsers.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ textAlign: 'center', padding: '2rem' }}>
                  {searchTerm ? 'No users found' : 'No users available'}
                </td>
              </tr>
            ) : (
              filteredUsers.map((user) => (
                <tr key={(user as any)._id || (user as any).id} className={selectedIds.includes((user as any)._id || (user as any).id) ? 'selected-row' : ''}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedIds.includes((user as any)._id || (user as any).id)}
                      onChange={() => handleSelectUser((user as any)._id || (user as any).id)}
                    />
                  </td>
                  <td>
                    <div className="user-cell">
                      <div className="user-avatar">{user.full_name?.charAt(0) || user.username.charAt(0)}</div>
                      <span className="user-name">{user.username}</span>
                    </div>
                  </td>
                  <td>{user.full_name || '-'}</td>
                  <td>{user.email}</td>
                  <td>
                    <span className={`role-badge ${user.role}`}>
                      {user.role}
                    </span>
                  </td>
                  <td>
                    <span className={`status-badge ${user.is_active ? 'active' : 'inactive'}`}>
                      {user.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </td>
                  <td>{user.last_login ? new Date(user.last_login).toLocaleString() : 'Never'}</td>
                  <td>
                    <div className="action-buttons">
                      <button
                        className="action-btn edit"
                        onClick={() => handleEditClick(user)}
                        title="Edit"
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                          <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" stroke="currentColor" strokeWidth="2"/>
                          <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                      </button>
                      <button
                        className="action-btn delete"
                        onClick={() => handleDeleteUser((user as any)._id || (user as any).id, user.username)}
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
};

export default UserManagement;
