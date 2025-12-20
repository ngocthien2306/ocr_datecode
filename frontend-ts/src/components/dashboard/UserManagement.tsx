import React, { useState, useEffect, ChangeEvent, FormEvent } from 'react';
import { usersAPI } from '@/services/api';
import { useToast } from '@/contexts/ToastContext';
import ConfirmDialog from '@/components/shared/ConfirmDialog';
import type { User, UserCreate, UserUpdate } from '@/types';

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
  role: 'admin' | 'operator' | 'viewer';
  is_active: boolean;
}

const UserManagement: React.FC = () => {
  const toast = useToast();
  const [users, setUsers] = useState<User[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [selectAll, setSelectAll] = useState(false);
  
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning',
    onConfirm: null
  });

  const [newUser, setNewUser] = useState<UserFormData>({
    username: '',
    email: '',
    full_name: '',
    phone_number: '',
    avatar_url: '',
    password: '',
    role: 'operator',
    is_active: true
  });

  const [editingUser, setEditingUser] = useState<(User & { password?: string }) | null>(null);

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

  const handleCreateUser = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      await usersAPI.createUser(newUser as UserCreate);
      setShowAddModal(false);
      setNewUser({
        username: '',
        email: '',
        full_name: '',
        phone_number: '',
        avatar_url: '',
        password: '',
        role: 'operator',
        is_active: true
      });
      await fetchUsers();
      toast.success('User created successfully!');
    } catch (err: any) {
      console.error('Error creating user:', err);
      toast.error(err.response?.data?.detail || 'Failed to create user');
    }
  };

  const handleEditClick = (user: User) => {
    setEditingUser({
      ...user,
      password: ''
    });
    setShowEditModal(true);
  };

  const handleEditUser = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!editingUser) return;

    try {
      const updateData: UserUpdate = {
        email: editingUser.email,
        full_name: editingUser.full_name,
        role: editingUser.role,
        is_active: editingUser.is_active
      };

      await usersAPI.updateUser(editingUser.id, updateData);
      setShowEditModal(false);
      setEditingUser(null);
      await fetchUsers();
      toast.success('User updated successfully!');
    } catch (err: any) {
      console.error('Error updating user:', err);
      toast.error(err.response?.data?.detail || 'Failed to update user');
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
      setSelectedIds(filteredUsers.map(u => u.id));
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
        <button className="dashboard-btn" onClick={() => setShowAddModal(true)}>
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
                <tr key={user.id} className={selectedIds.includes(user.id) ? 'selected-row' : ''}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(user.id)}
                      onChange={() => handleSelectUser(user.id)}
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
                        onClick={() => handleDeleteUser(user.id, user.username)}
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

      {/* Add User Modal */}
      {showAddModal && (
        <div className="modal-overlay" onClick={() => setShowAddModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Add New User</h3>
              <button className="close-btn" onClick={() => setShowAddModal(false)}>×</button>
            </div>
            <form onSubmit={handleCreateUser}>
              <div className="form-group">
                <label>Username *</label>
                <input
                  type="text"
                  value={newUser.username}
                  onChange={(e) => setNewUser({ ...newUser, username: e.target.value })}
                  required
                  placeholder="Enter username"
                />
              </div>
              <div className="form-group">
                <label>Full Name *</label>
                <input
                  type="text"
                  value={newUser.full_name}
                  onChange={(e) => setNewUser({ ...newUser, full_name: e.target.value })}
                  required
                  placeholder="Enter full name"
                />
              </div>
              <div className="form-group">
                <label>Email *</label>
                <input
                  type="email"
                  value={newUser.email}
                  onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                  required
                  placeholder="Enter email"
                />
              </div>
              <div className="form-group">
                <label>Phone Number</label>
                <input
                  type="text"
                  value={newUser.phone_number}
                  onChange={(e) => setNewUser({ ...newUser, phone_number: e.target.value })}
                  placeholder="Enter phone number"
                />
              </div>
              <div className="form-group">
                <label>Avatar URL</label>
                <input
                  type="text"
                  value={newUser.avatar_url}
                  onChange={(e) => setNewUser({ ...newUser, avatar_url: e.target.value })}
                  placeholder="Enter avatar URL"
                />
              </div>
              <div className="form-group">
                <label>Password *</label>
                <input
                  type="password"
                  value={newUser.password}
                  onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label>Role</label>
                <select
                  value={newUser.role}
                  onChange={(e) => setNewUser({ ...newUser, role: e.target.value as 'admin' | 'operator' | 'viewer' })}
                >
                  <option value="viewer">Viewer</option>
                  <option value="operator">Operator</option>
                  <option value="admin">Admin</option>
                </select>
              </div>
              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    checked={newUser.is_active}
                    onChange={(e) => setNewUser({ ...newUser, is_active: e.target.checked })}
                  />
                  <span>Active</span>
                </label>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" onClick={() => setShowAddModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Create User
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit User Modal */}
      {showEditModal && editingUser && (
        <div className="modal-overlay" onClick={() => setShowEditModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Edit User</h3>
              <button className="close-btn" onClick={() => setShowEditModal(false)}>×</button>
            </div>
            <form onSubmit={handleEditUser}>
              <div className="form-group">
                <label>Username</label>
                <input type="text" value={editingUser.username} disabled />
              </div>
              <div className="form-group">
                <label>Email *</label>
                <input
                  type="email"
                  value={editingUser.email}
                  onChange={(e) => setEditingUser({ ...editingUser, email: e.target.value })}
                  required
                  placeholder="Enter email"
                />
              </div>
              <div className="form-group">
                <label>Phone Number</label>
                <input
                  type="text"
                  value={editingUser.phone_number || ''}
                  onChange={(e) => setEditingUser({ ...editingUser, phone_number: e.target.value })}
                  placeholder="Enter phone number"
                />
              </div>
              <div className="form-group">
                <label>Avatar URL</label>
                <input
                  type="text"
                  value={editingUser.avatar_url || ''}
                  onChange={(e) => setEditingUser({ ...editingUser, avatar_url: e.target.value })}
                  placeholder="Enter avatar URL"
                />
              </div>
              <div className="form-group">
                <label>Full Name *</label>
                <input
                  type="text"
                  value={editingUser.full_name}
                  onChange={(e) => setEditingUser({ ...editingUser, full_name: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label>Role</label>
                <select
                  value={editingUser.role}
                  onChange={(e) => setEditingUser({ ...editingUser, role: e.target.value as 'admin' | 'operator' | 'viewer' })}
                >
                  <option value="viewer">Viewer</option>
                  <option value="operator">Operator</option>
                  <option value="admin">Admin</option>
                </select>
              </div>
              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    checked={editingUser.is_active}
                    onChange={(e) => setEditingUser({ ...editingUser, is_active: e.target.checked })}
                  />
                  <span>Active</span>
                </label>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" onClick={() => setShowEditModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Update User
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
        onConfirm={confirmDialog.onConfirm || (() => {})}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
      />
    </div>
  );
};

export default UserManagement;
