import React, { useState } from 'react';

export default function UserManagement() {
  const [users, setUsers] = useState([
    { id: 1, name: 'Admin User', email: 'admin@suntech.com', role: 'Administrator', status: 'Active', lastLogin: '2024-01-15 14:30' },
    { id: 2, name: 'John Smith', email: 'john.smith@suntech.com', role: 'Operator', status: 'Active', lastLogin: '2024-01-15 13:45' },
    { id: 3, name: 'Sarah Johnson', email: 'sarah.j@suntech.com', role: 'Supervisor', status: 'Active', lastLogin: '2024-01-15 12:20' },
    { id: 4, name: 'Mike Wilson', email: 'mike.w@suntech.com', role: 'Operator', status: 'Inactive', lastLogin: '2024-01-10 09:15' },
    { id: 5, name: 'Emily Davis', email: 'emily.d@suntech.com', role: 'Quality Control', status: 'Active', lastLogin: '2024-01-15 11:00' },
  ]);

  const [searchTerm, setSearchTerm] = useState('');

  const filteredUsers = users.filter(user =>
    user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.role.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="user-management-page">
      <div className="section-header">
        <h1>User Management</h1>
        <button className="dashboard-btn">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <line x1="12" y1="5" x2="12" y2="19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          Add User
        </button>
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
            placeholder="Search users by name, email, or role..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="filter-buttons">
          <button className="filter-btn active">All Users</button>
          <button className="filter-btn">Active</button>
          <button className="filter-btn">Inactive</button>
        </div>
      </div>

      {/* Users Table */}
      <div className="data-table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Email</th>
              <th>Role</th>
              <th>Status</th>
              <th>Last Login</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredUsers.map(user => (
              <tr key={user.id}>
                <td>#{user.id}</td>
                <td>
                  <div className="user-cell">
                    <div className="user-avatar">{user.name.charAt(0)}</div>
                    <span className="user-name">{user.name}</span>
                  </div>
                </td>
                <td>{user.email}</td>
                <td>
                  <span className="role-badge">{user.role}</span>
                </td>
                <td>
                  <span className={`status-badge ${user.status.toLowerCase()}`}>
                    {user.status}
                  </span>
                </td>
                <td>{user.lastLogin}</td>
                <td>
                  <div className="action-buttons">
                    <button className="action-btn edit" title="Edit">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M11 4H4C3.46957 4 2.96086 4.21071 2.58579 4.58579C2.21071 4.96086 2 5.46957 2 6V20C2 20.5304 2.21071 21.0391 2.58579 21.4142C2.96086 21.7893 3.46957 22 4 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M18.5 2.50001C18.8978 2.10219 19.4374 1.87869 20 1.87869C20.5626 1.87869 21.1022 2.10219 21.5 2.50001C21.8978 2.89784 22.1213 3.4374 22.1213 4.00001C22.1213 4.56262 21.8978 5.10219 21.5 5.50001L12 15L8 16L9 12L18.5 2.50001Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </button>
                    <button className="action-btn delete" title="Delete">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M3 6H5H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M8 6V4C8 3.46957 8.21071 2.96086 8.58579 2.58579C8.96086 2.21071 9.46957 2 10 2H14C14.5304 2 15.0391 2.21071 15.4142 2.58579C15.7893 2.96086 16 3.46957 16 4V6M19 6V20C19 20.5304 18.7893 21.0391 18.4142 21.4142C18.0391 21.7893 17.5304 22 17 22H7C6.46957 22 5.96086 21.7893 5.58579 21.4142C5.21071 21.0391 5 20.5304 5 20V6H19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="pagination">
        <button className="pagination-btn" disabled>Previous</button>
        <div className="pagination-numbers">
          <button className="pagination-number active">1</button>
          <button className="pagination-number">2</button>
          <button className="pagination-number">3</button>
        </div>
        <button className="pagination-btn">Next</button>
      </div>
    </div>
  );
}
