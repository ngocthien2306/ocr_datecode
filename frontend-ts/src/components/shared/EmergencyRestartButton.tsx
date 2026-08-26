import React, { useState } from 'react';
import { useEmergencyRestart } from '@/contexts/EmergencyRestartContext';
import './EmergencyRestartButton.css';

/**
 * Always-visible emergency control, pinned to the top-right corner.
 *
 * Pressing it does NOT restart anything on its own — it opens a confirmation
 * card, and only that card arms the cancellable countdown. Restarting stops the
 * line for 1–2 minutes, so it must never be one stray tap away on a touchscreen
 * standing next to a running conveyor.
 */
export const EmergencyRestartButton: React.FC = () => {
  const { phase, arm } = useEmergencyRestart();
  const [confirming, setConfirming] = useState(false);

  // The countdown overlay owns the screen once armed; hide the launcher so the
  // operator only ever sees one control.
  if (phase !== 'idle') return null;

  return (
    <div className="erb-root">
      <button
        type="button"
        className="erb-button"
        onClick={() => setConfirming(v => !v)}
        title="Khởi động lại toàn bộ dịch vụ (khẩn cấp)"
        aria-label="Khởi động lại toàn bộ dịch vụ"
      >
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M12 3a9 9 0 1 0 9 9"
            stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"
          />
          <path
            d="M12 3v6m0-6 4 3-4 3"
            stroke="currentColor" strokeWidth="2.2"
            strokeLinecap="round" strokeLinejoin="round"
          />
        </svg>
        <span>Khẩn cấp</span>
      </button>

      {confirming && (
        <div className="erb-popover" role="dialog" aria-label="Xác nhận khởi động lại">
          <div className="erb-popover-title">Khởi động lại toàn bộ dịch vụ?</div>
          <div className="erb-popover-body">
            Backend, camera service và giao diện sẽ khởi động lại.
            Dây chuyền <strong>ngừng kiểm tra khoảng 1–2 phút</strong>.
            Chỉ dùng khi hệ thống không còn cách khắc phục nào khác.
          </div>
          <div className="erb-popover-actions">
            <button
              type="button"
              className="erb-cancel"
              onClick={() => setConfirming(false)}
            >
              Huỷ
            </button>
            <button
              type="button"
              className="erb-confirm"
              onClick={() => { setConfirming(false); arm(); }}
            >
              Khởi động lại
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
