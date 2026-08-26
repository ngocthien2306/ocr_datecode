import React from 'react';
import { useEmergencyRestart } from '@/contexts/EmergencyRestartContext';
import './FrameUnavailableDialog.css';

/** Shape of the `detail` payload from backend/app/services/frame_guidance.py */
export interface FrameRemedy {
  step: number;
  action: 'reconnect_camera' | 'reload_recipe' | 'restart_all' | string;
  label: string;
  detail: string;
  danger?: boolean;
}

export interface FrameUnavailableDetail {
  code: string;
  title: string;
  reason: string;
  serial_number?: string;
  grab_mode?: string;
  remedies: FrameRemedy[];
}

/**
 * Pull the structured guidance out of an axios error, or null when the backend
 * answered with a plain string (older build, or an unrelated failure) so the
 * caller can fall back to a toast.
 */
export const parseFrameUnavailable = (error: any): FrameUnavailableDetail | null => {
  const detail = error?.response?.data?.detail;
  if (detail && typeof detail === 'object' && Array.isArray(detail.remedies)) {
    return detail as FrameUnavailableDetail;
  }
  return null;
};

/**
 * The same three remedies the backend sends, for the cases the frontend detects
 * on its own (camera flagged disconnected, an empty 200 response, a network
 * error). Kept in sync with backend/app/services/frame_guidance.py so the
 * operator reads identical wording no matter which side noticed the problem.
 */
export const LOCAL_FRAME_REMEDIES = (serialNumber: string): FrameRemedy[] => [
  {
    step: 1,
    action: 'reconnect_camera',
    label: 'Ngắt kết nối rồi kết nối lại camera',
    detail:
      `Mở tab Camera, bấm Disconnect cho camera ${serialNumber}, ` +
      'chờ vài giây rồi bấm Connect lại. Sau đó quay lại đây và chụp hình.',
  },
  {
    step: 2,
    action: 'reload_recipe',
    label: 'Load lại recipe rồi chụp hình',
    detail:
      'Mở tab Recipe và Load lại recipe đang dùng. Camera chỉ sinh hình khi recipe ' +
      'đang chạy — recipe vừa dừng thì bộ đệm sẽ không có hình mới.',
  },
  {
    step: 3,
    action: 'restart_all',
    label: 'Khẩn cấp: khởi động lại toàn bộ dịch vụ',
    detail:
      'Chỉ dùng khi hai cách trên không hiệu quả. Toàn bộ hệ thống sẽ khởi động lại ' +
      'và dây chuyền ngừng kiểm tra khoảng 1–2 phút.',
    danger: true,
  },
];

interface Props {
  detail: FrameUnavailableDetail | null;
  onClose: () => void;
  /** Optional: retry the capture without closing the modal behind us. */
  onRetry?: () => void;
}

/**
 * Explains, in Vietnamese, why no frame came back and what to do about it —
 * ordered cheapest-fix-first. Step 3 hands over to the shared emergency-restart
 * countdown rather than restarting on the spot.
 */
export const FrameUnavailableDialog: React.FC<Props> = ({ detail, onClose, onRetry }) => {
  const { arm } = useEmergencyRestart();

  if (!detail) return null;

  return (
    <div className="fud-overlay" role="dialog" aria-modal="true">
      <div className="fud-card">
        <div className="fud-header">
          <div className="fud-icon" aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path
                d="M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"
                stroke="currentColor" strokeWidth="2"
                strokeLinecap="round" strokeLinejoin="round"
              />
            </svg>
          </div>
          <div>
            <div className="fud-title">{detail.title}</div>
            {detail.serial_number && (
              <div className="fud-subtitle">Camera {detail.serial_number}</div>
            )}
          </div>
        </div>

        <div className="fud-reason">{detail.reason}</div>

        <div className="fud-steps-label">Cách xử lý, thử lần lượt từ trên xuống:</div>

        <ol className="fud-steps">
          {detail.remedies.map(remedy => (
            <li
              key={remedy.step}
              className={`fud-step${remedy.danger ? ' fud-step-danger' : ''}`}
            >
              <div className="fud-step-badge">{remedy.step}</div>
              <div className="fud-step-body">
                <div className="fud-step-label">{remedy.label}</div>
                <div className="fud-step-detail">{remedy.detail}</div>
                {remedy.action === 'restart_all' && (
                  <button
                    type="button"
                    className="fud-restart-btn"
                    onClick={() => { onClose(); arm(); }}
                  >
                    Khởi động lại toàn bộ dịch vụ
                  </button>
                )}
              </div>
            </li>
          ))}
        </ol>

        <div className="fud-actions">
          <button type="button" className="fud-close" onClick={onClose}>
            Đóng
          </button>
          {onRetry && (
            <button type="button" className="fud-retry" onClick={onRetry}>
              Thử lấy hình lại
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
