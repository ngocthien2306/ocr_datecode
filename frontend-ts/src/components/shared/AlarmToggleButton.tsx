import React, { useState } from 'react';
import { mlTrainingAPI } from '@/services/mlTraining';
import { useToast } from '@/contexts/ToastContext';
import './AlarmToggleButton.css';

/**
 * Bật/tắt còi báo động (DO3), ghim ngay dưới nút Khẩn cấp.
 *
 * Tồn tại vì thư viện GPIO đôi khi để chân còi kẹt ở mức kêu sau một xung báo
 * động. Bấm là lật luôn, không hỏi lại: lật còi không làm dừng dây chuyền, và
 * lúc còi đang kêu inh ỏi thì thêm một bước xác nhận chỉ gây khó chịu.
 *
 * Không tự hỏi trạng thái định kỳ — mỗi lần đọc là một lệnh `dio_out` tranh EC
 * với luồng GPIO của AI service. Trạng thái chỉ biết được sau lần bấm đầu.
 */
export const AlarmToggleButton: React.FC = () => {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const [alarmOn, setAlarmOn] = useState<boolean | null>(null);

  const handleClick = async () => {
    if (busy) return;
    setBusy(true);
    try {
      const { alarm_on } = await mlTrainingAPI.alarmToggle();
      setAlarmOn(alarm_on);
      if (alarm_on) toast.warning('Đã bật còi');
      else toast.success('Đã tắt còi');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? 'Không đổi được trạng thái còi');
    } finally {
      setBusy(false);
    }
  };

  const label = alarmOn ? 'Tắt còi' : 'Bật/tắt còi';

  return (
    <button
      type="button"
      className={`atb-button${alarmOn ? ' atb-on' : ''}${busy ? ' atb-busy' : ''}`}
      onClick={handleClick}
      disabled={busy}
      title="Bật/tắt còi báo động"
      aria-label="Bật/tắt còi báo động"
    >
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path
          d="M6 16V11a6 6 0 1 1 12 0v5l1.5 2h-15L6 16Z"
          stroke="currentColor" strokeWidth="2" strokeLinejoin="round"
        />
        <path d="M10 21h4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        {!alarmOn && (
          <path d="M4 4l16 16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        )}
      </svg>
      <span className="atb-label">{label}</span>
    </button>
  );
};
