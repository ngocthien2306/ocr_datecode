import React, { useState } from 'react';
import AuditLog from './AuditLog';
import SystemLogs from './SystemLogs';
import '@/styles/SystemLogs.css';

type LogsSubTab = 'audit' | 'system';

const Logs: React.FC = () => {
  const [subTab, setSubTab] = useState<LogsSubTab>('audit');

  return (
    <div className="logs-page" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="logs-subtabs" style={{ display: 'flex', gap: 4, padding: '8px 16px 0' }}>
        {([
          { key: 'audit',  label: 'Audit Log' },
          { key: 'system', label: 'System Logs' },
        ] as const).map(t => (
          <button
            key={t.key}
            type="button"
            onClick={() => setSubTab(t.key)}
            className={`logs-subtab-btn ${subTab === t.key ? 'active' : ''}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ flex: 1, overflow: 'auto' }}>
        {subTab === 'audit' ? <AuditLog /> : <SystemLogs />}
      </div>
    </div>
  );
};

export default Logs;
