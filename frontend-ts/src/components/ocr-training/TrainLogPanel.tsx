import { useEffect, useRef } from 'react';

interface LogLine { idx: number; level: string; msg: string }

interface Props {
  live: { phase: string | null; progress: number; status: string } | null;
  logs: LogLine[];
}

/** Right-hand card of the Train tab: run status, progress bar and the log.
 *  Shared by both trainers so the two Train tabs look and behave the same. */
export default function TrainLogPanel({ live, logs }: Props) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const stickRef = useRef(true);

  // Follow new lines only while the user is at the bottom, so scrolling up to
  // read an earlier epoch isn't yanked back every 2 s poll.
  useEffect(() => {
    const el = boxRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [logs.length]);

  const onScroll = () => {
    const el = boxRef.current;
    if (el) stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  const isLive = !!live && (live.status === 'training' || live.status === 'pending');

  return (
    <section className="ot-card ot-train-progress">
      <div className="ot-card-title">
        Progress &amp; log
        {live && <span className={`ot-status ot-status--${live.status}`}>{live.status}</span>}
      </div>

      {live && (
        <div className="ot-progress-meta">
          <span>{live.phase === 'waiting_for_gpu' ? 'Waiting for GPU (another job is running)' : (live.phase ?? '—')}</span>
          <b>{Math.round(live.progress)}%</b>
        </div>
      )}
      {isLive && (
        <div className="ot-progress-bar">
          <div style={{ width: `${live!.progress}%` }} />
        </div>
      )}

      <div className="ot-log-box" ref={boxRef} onScroll={onScroll}>
        {logs.length === 0 ? (
          <div className="ot-log-empty">
            {live ? 'No log lines for this run.' : 'Select a run, or start training — its log appears here.'}
          </div>
        ) : logs.map((l) => (
          <div key={l.idx} className={l.level === 'ERROR' ? 'at-log-error'
                                     : l.level === 'WARNING' ? 'at-log-warning' : undefined}>
            {l.msg}
          </div>
        ))}
      </div>
    </section>
  );
}
