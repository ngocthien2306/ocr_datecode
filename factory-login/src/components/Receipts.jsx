import React, { useState } from 'react';

export default function Receipts() {
  const [receipts, setReceipts] = useState([
    { id: 'RCP-2024-001', date: '2024-01-15', camera: 'Camera 1 - Line A', products: 456, passed: 452, failed: 4, operator: 'John Smith', status: 'Completed' },
    { id: 'RCP-2024-002', date: '2024-01-15', camera: 'Camera 2 - Line B', products: 423, passed: 417, failed: 6, operator: 'Sarah Johnson', status: 'Completed' },
    { id: 'RCP-2024-003', date: '2024-01-15', camera: 'Camera 3 - QC', products: 368, passed: 360, failed: 8, operator: 'Emily Davis', status: 'Completed' },
    { id: 'RCP-2024-004', date: '2024-01-14', camera: 'Camera 1 - Line A', products: 445, passed: 440, failed: 5, operator: 'John Smith', status: 'Completed' },
    { id: 'RCP-2024-005', date: '2024-01-14', camera: 'Camera 2 - Line B', products: 412, passed: 408, failed: 4, operator: 'Mike Wilson', status: 'Completed' },
    { id: 'RCP-2024-006', date: '2024-01-14', camera: 'Camera 3 - QC', products: 355, passed: 348, failed: 7, operator: 'Emily Davis', status: 'Completed' },
  ]);

  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDate, setSelectedDate] = useState('all');

  const filteredReceipts = receipts.filter(receipt => {
    const matchesSearch = receipt.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      receipt.camera.toLowerCase().includes(searchTerm.toLowerCase()) ||
      receipt.operator.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesDate = selectedDate === 'all' || receipt.date === selectedDate;

    return matchesSearch && matchesDate;
  });

  return (
    <div className="receipts-page">
      <div className="section-header">
        <h1>Production Receipts</h1>
        <button className="dashboard-btn">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <line x1="12" y1="5" x2="12" y2="19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          Create Receipt
        </button>
      </div>

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
            <div className="card-value">{receipts.length}</div>
            <span className="card-status normal">This period</span>
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
            <div className="card-value">{receipts.reduce((sum, r) => sum + r.products, 0)}</div>
            <span className="card-status increase">All receipts</span>
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
            <h3>Success Rate</h3>
            <div className="card-value">98.2%</div>
            <span className="card-status success">Above target</span>
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
            placeholder="Search receipts by ID, camera, or operator..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="filter-buttons">
          <select
            className="filter-select"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
          >
            <option value="all">All Dates</option>
            <option value="2024-01-15">2024-01-15</option>
            <option value="2024-01-14">2024-01-14</option>
          </select>
          <button className="filter-btn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M21 15V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="7,10 12,15 17,10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="12" y1="15" x2="12" y2="3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Export
          </button>
        </div>
      </div>

      {/* Receipts Table */}
      <div className="data-table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Receipt ID</th>
              <th>Date</th>
              <th>Camera/Line</th>
              <th>Total Products</th>
              <th>Passed</th>
              <th>Failed</th>
              <th>Success Rate</th>
              <th>Operator</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredReceipts.map(receipt => (
              <tr key={receipt.id}>
                <td><strong>{receipt.id}</strong></td>
                <td>{receipt.date}</td>
                <td>{receipt.camera}</td>
                <td>{receipt.products}</td>
                <td className="text-success">{receipt.passed}</td>
                <td className="text-error">{receipt.failed}</td>
                <td>
                  <div className="progress-cell">
                    <span>{((receipt.passed / receipt.products) * 100).toFixed(1)}%</span>
                    <div className="progress-bar">
                      <div
                        className="progress-fill success"
                        style={{ width: `${(receipt.passed / receipt.products) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </td>
                <td>{receipt.operator}</td>
                <td>
                  <span className="status-badge completed">{receipt.status}</span>
                </td>
                <td>
                  <div className="action-buttons">
                    <button className="action-btn view" title="View">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M1 12C1 12 5 4 12 4C19 4 23 12 23 12C23 12 19 20 12 20C5 20 1 12 1 12Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </button>
                    <button className="action-btn download" title="Download">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M21 15V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <polyline points="7,10 12,15 17,10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <line x1="12" y1="15" x2="12" y2="3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
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
