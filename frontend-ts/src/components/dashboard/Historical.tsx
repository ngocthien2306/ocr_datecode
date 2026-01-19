import React, { useState, useEffect } from 'react';
import { inferenceResultsAPI } from '@/services/inferenceResults';
import InspectionResultsTab from './historical/InspectionResultsTab';
import ProductionAnalyticsTab from './historical/ProductionAnalyticsTab';
import '@/styles/Historical.css';

interface SummaryStats {
  total: number;
  pass: number;
  fail: number;
  passRate: number;
}

const Historical: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'inspection' | 'analytics'>('inspection');
  const [summaryStats, setSummaryStats] = useState<SummaryStats>({
    total: 0,
    pass: 0,
    fail: 0,
    passRate: 0
  });
  const [dateRange, setDateRange] = useState('7days');
  const [loading, setLoading] = useState(false);

  // Fetch summary stats based on date range
  useEffect(() => {
    fetchSummaryStats();
  }, [dateRange]);

  const fetchSummaryStats = async () => {
    try {
      setLoading(true);

      // Calculate date range
      const { start_date, end_date } = getDateRange(dateRange);

      // Fetch summary statistics from new API
      const stats = await inferenceResultsAPI.getSummaryStatistics({
        start_date,
        end_date
      });

      setSummaryStats({
        total: stats.total,
        pass: stats.pass,
        fail: stats.fail,
        passRate: stats.pass_rate
      });
    } catch (error) {
      console.error('Error fetching summary stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const getDateRange = (range: string): { start_date: string; end_date: string } => {
    const now = new Date();
    const end = new Date(now);
    let start = new Date(now);

    switch (range) {
      case 'today':
        start.setHours(0, 0, 0, 0);
        break;
      case 'yesterday':
        start.setDate(now.getDate() - 1);
        start.setHours(0, 0, 0, 0);
        end.setDate(now.getDate() - 1);
        end.setHours(23, 59, 59, 999);
        break;
      case '1days':
        start.setDate(now.getDate() - 1);
        break;
      case '3days':
        start.setDate(now.getDate() - 3);
        break;
      case '7days':
        start.setDate(now.getDate() - 7);
        break;
      case 'thisweek':
        const dayOfWeek = now.getDay();
        start.setDate(now.getDate() - dayOfWeek);
        start.setHours(0, 0, 0, 0);
        break;
      case 'lastweek':
        const lastWeekDay = now.getDay();
        start.setDate(now.getDate() - lastWeekDay - 7);
        start.setHours(0, 0, 0, 0);
        end.setDate(now.getDate() - lastWeekDay - 1);
        end.setHours(23, 59, 59, 999);
        break;
      case '30days':
        start.setDate(now.getDate() - 30);
        break;
      case 'thismonth':
        start.setDate(1);
        start.setHours(0, 0, 0, 0);
        break;
      case 'lastmonth':
        start.setMonth(now.getMonth() - 1);
        start.setDate(1);
        start.setHours(0, 0, 0, 0);
        end.setMonth(now.getMonth());
        end.setDate(0);
        end.setHours(23, 59, 59, 999);
        break;
      default:
        start.setDate(now.getDate() - 7);
    }

    return {
      start_date: start.toISOString(),
      end_date: end.toISOString()
    };
  };

  return (
    <div className="historical-page">
      {/* Header */}
      <div className="section-header">
        <h1>Historical Data</h1>
        <div className="header-controls">
          <div className="control-group">
            <label>Date Range:</label>
            <select value={dateRange} onChange={(e) => setDateRange(e.target.value)}>
              <option value="today">Today</option>
              <option value="yesterday">Yesterday</option>
              <option value="1days">Last 1 Day</option>
              <option value="3days">Last 3 Days</option>
              <option value="7days">Last 7 Days</option>
              <option value="thisweek">This Week</option>
              <option value="lastweek">Last Week</option>
              <option value="30days">Last 30 Days</option>
              <option value="thismonth">This Month</option>
              <option value="lastmonth">Last Month</option>
            </select>
          </div>
          <button className="dashboard-btn" onClick={fetchSummaryStats}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M21.5 2V6M21.5 6H17.5M21.5 6L18.5 3C16.8 1.5 14.5 0.5 12 0.5C5.5 0.5 0.5 5.5 0.5 12C0.5 18.5 5.5 23.5 12 23.5C17.5 23.5 22 19.5 23 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Refresh
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="summary-cards">
        <div className="summary-card">
          <div className="card-icon products">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="14" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="14" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              <rect x="3" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </div>
          <div className="card-info">
            <h3>Total Inspections</h3>
            <div className="card-value">{loading ? '...' : summaryStats.total.toLocaleString()}</div>
            <span className="card-status normal">
              {dateRange === 'today' ? 'Today' :
               dateRange === 'yesterday' ? 'Yesterday' :
               dateRange === '1days' ? 'Last 1 day' :
               dateRange === '3days' ? 'Last 3 days' :
               dateRange === '7days' ? 'Last 7 days' :
               dateRange === 'thisweek' ? 'This week' :
               dateRange === 'lastweek' ? 'Last week' :
               dateRange === '30days' ? 'Last 30 days' :
               dateRange === 'thismonth' ? 'This month' :
               dateRange === 'lastmonth' ? 'Last month' : 'Period'}
            </span>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-icon success">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M22 11.08V12C21.9988 14.1564 21.3005 16.2547 20.0093 17.9818C18.7182 19.7088 16.9033 20.9725 14.8354 21.5839C12.7674 22.1953 10.5573 22.1219 8.53447 21.3746C6.51168 20.6273 4.78465 19.2461 3.61096 17.4371C2.43727 15.628 1.87979 13.4881 2.02168 11.3363C2.16356 9.18455 2.99721 7.13631 4.39828 5.49706C5.79935 3.85781 7.69279 2.71537 9.79619 2.24013C11.8996 1.7649 14.1003 1.98232 16.07 2.85999" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M22 4L12 14.01L9 11.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="card-info">
            <h3>PASS</h3>
            <div className="card-value">{loading ? '...' : summaryStats.pass.toLocaleString()}</div>
            <span className="card-status success">
              {summaryStats.total > 0 ? `${Math.round((summaryStats.pass / summaryStats.total) * 100)}%` : '0%'} of total
            </span>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-icon error">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
              <line x1="15" y1="9" x2="9" y2="15" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <line x1="9" y1="9" x2="15" y2="15" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
          <div className="card-info">
            <h3>FAIL</h3>
            <div className="card-value">{loading ? '...' : summaryStats.fail.toLocaleString()}</div>
            <span className="card-status error">
              {summaryStats.total > 0 ? `${Math.round((summaryStats.fail / summaryStats.total) * 100)}%` : '0%'} of total
            </span>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-icon camera">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="card-info">
            <h3>Pass Rate</h3>
            <div className="card-value">{loading ? '...' : `${summaryStats.passRate}%`}</div>
            <span className={`card-status ${summaryStats.passRate >= 90 ? 'success' : summaryStats.passRate >= 70 ? 'normal' : 'error'}`}>
              {summaryStats.passRate >= 90 ? 'Excellent' : summaryStats.passRate >= 70 ? 'Good' : 'Needs attention'}
            </span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="tabs-container">
        <div className="tabs-header">
          <button
            className={`tab-button ${activeTab === 'inspection' ? 'active' : ''}`}
            onClick={() => setActiveTab('inspection')}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="16" y1="13" x2="8" y2="13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="16" y1="17" x2="8" y2="17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="10,9 9,9 8,9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Inspection Results
          </button>
          <button
            className={`tab-button ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <line x1="12" y1="20" x2="12" y2="10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="18" y1="20" x2="18" y2="4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="6" y1="20" x2="6" y2="16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Production Analytics
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'inspection' ? (
            <InspectionResultsTab dateRange={dateRange} />
          ) : (
            <ProductionAnalyticsTab />
          )}
        </div>
      </div>
    </div>
  );
};

export default Historical;
