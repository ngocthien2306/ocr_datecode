/**
 * SVG Icons for Agent Chat Widget
 * Converts emoji/text icons to proper SVG icons
 */

import React from 'react';

interface IconProps {
  size?: number;
  color?: string;
  className?: string;
}

// Robot/Bot Icon (replaces 🤖)
export const BotIcon: React.FC<IconProps> = ({ size = 24, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <rect x="5" y="9" width="14" height="12" rx="2" stroke={color} strokeWidth="2"/>
    <circle cx="9" cy="13" r="1" fill={color}/>
    <circle cx="15" cy="13" r="1" fill={color}/>
    <path d="M9 17h6" stroke={color} strokeWidth="2" strokeLinecap="round"/>
    <path d="M12 4v3" stroke={color} strokeWidth="2" strokeLinecap="round"/>
    <circle cx="12" cy="3" r="1" fill={color}/>
  </svg>
);

// Trash/Delete Icon (replaces 🗑️)
export const TrashIcon: React.FC<IconProps> = ({ size = 20, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke={color} strokeWidth="2" strokeLinecap="round"/>
    <path d="M19 6v12a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" stroke={color} strokeWidth="2"/>
    <path d="M10 11v6M14 11v6" stroke={color} strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

// Maximize Icon (replaces ⛶)
export const MaximizeIcon: React.FC<IconProps> = ({ size = 20, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" stroke={color} strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

// Minimize/Restore Icon (replaces ⊡)
export const RestoreIcon: React.FC<IconProps> = ({ size = 20, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M8 3v3m0 0H5m3 0l-5 5M16 21v-3m0 0h3m-3 0l5-5" stroke={color} strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

// Minimize Icon (replaces −)
export const MinimizeIcon: React.FC<IconProps> = ({ size = 20, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M5 12h14" stroke={color} strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

// Window Icon (for restore from minimize)
export const WindowIcon: React.FC<IconProps> = ({ size = 20, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <rect x="3" y="3" width="18" height="18" rx="2" stroke={color} strokeWidth="2"/>
    <path d="M3 9h18" stroke={color} strokeWidth="2"/>
  </svg>
);

// Close/X Icon (replaces ✕)
export const CloseIcon: React.FC<IconProps> = ({ size = 20, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M18 6L6 18M6 6l12 12" stroke={color} strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

// Send/Arrow Up Icon (replaces ↑)
export const SendIcon: React.FC<IconProps> = ({ size = 20, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M12 19V5M5 12l7-7 7 7" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

// Loading/Spinner Icon (replaces ⟳)
export const LoadingIcon: React.FC<IconProps> = ({ size = 20, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={`${className} agent-icon-spin`}>
    <path d="M12 2v4m0 12v4M6 12H2m20 0h-4" stroke={color} strokeWidth="2" strokeLinecap="round"/>
    <path d="M17.657 17.657L15.536 15.536m-7.072 0L6.343 17.657M17.657 6.343L15.536 8.464m-7.072 0L6.343 6.343"
          stroke={color} strokeWidth="2" strokeLinecap="round" opacity="0.4"/>
  </svg>
);

// Refresh Icon (replaces 🔄)
export const RefreshIcon: React.FC<IconProps> = ({ size = 18, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"
          stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

// Quick Action Icons
export const SearchIcon: React.FC<IconProps> = ({ size = 18, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <circle cx="11" cy="11" r="8" stroke={color} strokeWidth="2"/>
    <path d="m21 21-4.35-4.35" stroke={color} strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

export const DocumentIcon: React.FC<IconProps> = ({ size = 18, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z" stroke={color} strokeWidth="2"/>
    <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" stroke={color} strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

export const HelpIcon: React.FC<IconProps> = ({ size = 18, color = 'currentColor', className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <circle cx="12" cy="12" r="10" stroke={color} strokeWidth="2"/>
    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3M12 17h.01" stroke={color} strokeWidth="2" strokeLinecap="round"/>
  </svg>
);
