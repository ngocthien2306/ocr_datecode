import React, { useState } from 'react';
import SystemMonitor from '@/components/SystemMonitor';
import IOMonitor from '@/components/IOMonitor';
import StorageManager from '@/components/StorageManager';
import '@/styles/System.css';

type TabType = 'system' | 'io' | 'storage';

const System: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('system');

  return (
    <div className="system-page">
      {/* Header */}
      <div className="system-header">
        <div className="system-title-section">
          <h1>System</h1>
          <p>Monitor system metrics and I/O status</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="system-tabs">
        <button
          className={`tab-btn ${activeTab === 'system' ? 'active' : ''}`}
          onClick={() => setActiveTab('system')}
        >
          System Monitor
        </button>
        <button
          className={`tab-btn ${activeTab === 'io' ? 'active' : ''}`}
          onClick={() => setActiveTab('io')}
        >
          I/O Monitor
        </button>
        <button
          className={`tab-btn ${activeTab === 'storage' ? 'active' : ''}`}
          onClick={() => setActiveTab('storage')}
        >
          Storage
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'system' && <SystemMonitor />}
        {activeTab === 'io' && <IOMonitor />}
        {activeTab === 'storage' && <StorageManager />}
      </div>
    </div>
  );
};

export default System;
