import React, {
  createContext, useCallback, useContext, useEffect, useRef, useState,
} from 'react';
import { mlTrainingAPI } from '@/services/mlTraining';
import './EmergencyRestartContext.css';

/**
 * Emergency restart of the whole stack (`systemctl restart ocr-all`).
 *
 * Owned by a context rather than by each button because two places can start it
 * — the floating button in the corner and step 3 of the "no frame" dialog — and
 * a production line must never end up with two countdowns racing each other.
 *
 * Restarting stops inspection for 1–2 minutes, so arming is deliberately a
 * two-stage action: press, then watch a cancellable countdown run out.
 */

const COUNTDOWN_SECONDS = 10;
const HEALTH_POLL_MS = 3000;

type Phase = 'idle' | 'counting' | 'restarting';

interface EmergencyRestartValue {
  phase: Phase;
  secondsLeft: number;
  /** Start the cancellable countdown. No-op if already armed. */
  arm: () => void;
  /** Abort before the countdown reaches zero. */
  cancel: () => void;
}

const EmergencyRestartContext = createContext<EmergencyRestartValue | null>(null);

export const useEmergencyRestart = (): EmergencyRestartValue => {
  const ctx = useContext(EmergencyRestartContext);
  if (!ctx) {
    throw new Error('useEmergencyRestart must be used inside <EmergencyRestartProvider>');
  }
  return ctx;
};

export const EmergencyRestartProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [phase, setPhase] = useState<Phase>('idle');
  const [secondsLeft, setSecondsLeft] = useState(COUNTDOWN_SECONDS);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTick = useCallback(() => {
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
  }, []);

  const arm = useCallback(() => {
    setPhase(prev => (prev === 'idle' ? 'counting' : prev));
    setSecondsLeft(COUNTDOWN_SECONDS);
  }, []);

  const cancel = useCallback(() => {
    clearTick();
    setPhase('idle');
    setSecondsLeft(COUNTDOWN_SECONDS);
  }, [clearTick]);

  // Countdown → fire.
  useEffect(() => {
    if (phase !== 'counting') return;

    tickRef.current = setInterval(() => {
      setSecondsLeft(prev => {
        if (prev > 1) return prev - 1;
        clearTick();
        setPhase('restarting');
        // The backend kills itself ~1s after answering, so a rejected promise
        // here is expected and must not surface as an error to the operator.
        mlTrainingAPI.restartAll().catch(() => { /* backend is going down */ });
        return 0;
      });
    }, 1000);

    return clearTick;
  }, [phase, clearTick]);

  // While restarting, poll /health and reload once the new backend answers, so
  // every stream and socket is rebuilt from scratch rather than half-resumed.
  useEffect(() => {
    if (phase !== 'restarting') return;

    let cancelled = false;
    // Give the old process time to actually die, otherwise the first poll hits
    // the backend that is about to exit and we reload straight into the outage.
    const startDelay = setTimeout(() => {
      const poll = setInterval(async () => {
        try {
          await mlTrainingAPI.apiHealth();
          if (!cancelled) {
            clearInterval(poll);
            window.location.reload();
          }
        } catch {
          /* still down — keep polling */
        }
      }, HEALTH_POLL_MS);

      if (cancelled) clearInterval(poll);
    }, 8000);

    return () => {
      cancelled = true;
      clearTimeout(startDelay);
    };
  }, [phase]);

  return (
    <EmergencyRestartContext.Provider value={{ phase, secondsLeft, arm, cancel }}>
      {children}
      {phase !== 'idle' && (
        <div className="emr-overlay">
          <div className="emr-card">
            {phase === 'counting' ? (
              <>
                <div className="emr-ring">
                  <span className="emr-count">{secondsLeft}</span>
                </div>
                <div className="emr-title">Sắp khởi động lại toàn bộ dịch vụ</div>
                <div className="emr-message">
                  Hệ thống sẽ khởi động lại sau <strong>{secondsLeft}</strong> giây.
                  Dây chuyền sẽ ngừng kiểm tra khoảng 1–2 phút.
                </div>
                <button type="button" className="emr-cancel" onClick={cancel}>
                  Huỷ, không khởi động lại
                </button>
              </>
            ) : (
              <>
                <div className="emr-spinner" />
                <div className="emr-title">Đang khởi động lại dịch vụ…</div>
                <div className="emr-message">
                  Vui lòng chờ. Trang sẽ tự tải lại khi hệ thống hoạt động trở lại.
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </EmergencyRestartContext.Provider>
  );
};
