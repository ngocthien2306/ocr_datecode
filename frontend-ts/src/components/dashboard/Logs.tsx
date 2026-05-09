import React, { useState } from 'react';
import AuditLog from './AuditLog';
import SystemLogs from './SystemLogs';

type LogsSubTab = 'audit' | 'system';

const Logs: React.FC = () => {
  const [subTab, setSubTab] = useState<LogsSubTab>('audit');

  return (
    <div className="logs-page">
      <div
        className="logs-subtabs"
        style={{
          display: 'flex',
          gap: 4,
          padding: '8px 16px 0',
          borderBottom: '1px solid var(--color-border, #2a2f3a)',
        }}
      >
        {([
          { key: 'audit',  label: 'Audit Log' },
          { key: 'system', label: 'System Logs' },
        ] as const).map(t => (
          <button
            key={t.key}
            type="button"
            onClick={() => setSubTab(t.key)}
            className={`logs-subtab-btn ${subTab === t.key ? 'active' : ''}`}
            style={{
              padding: '8px 16px',
              border: 'none',
              borderBottom: subTab === t.key
                ? '2px solid var(--color-primary, #3b82f6)'
                : '2px solid transparent',
              background: 'transparent',
              color: subTab === t.key
                ? 'var(--color-text, #e5e7eb)'
                : 'var(--color-text-muted, #9ca3af)',
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: subTab === t.key ? 600 : 400,
              marginBottom: -1,
            }}
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
