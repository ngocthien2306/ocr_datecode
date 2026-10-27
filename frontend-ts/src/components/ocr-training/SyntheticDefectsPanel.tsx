import { useEffect, useState } from 'react';
import {
  paddleOcrAPI, SyntheticDefectType, SyntheticInfo, SyntheticProgress,
} from '@/services/paddleOcrTraining';

interface Props {
  projectId: string;
  /** The Train form's test split — synthetic crops are tagged train/test with
   *  it, so a damaged copy of a test crop never ends up in training. */
  testSplit: number;
  useSynthetic: boolean;
  onUseSyntheticChange: (v: boolean) => void;
}

const TYPES: Array<{ id: SyntheticDefectType; label: string; hint: string }> = [
  { id: 'stroke_break', label: 'Đứt nét', hint: 'xoá 1–2 dải cắt ngang/dọc qua nét chữ → nhãn □' },
  { id: 'dot_dropout', label: 'Mất chấm', hint: 'xoá phần lớn các chấm CIJ của ký tự → nhãn □' },
  { id: 'overprint', label: 'Chồng chữ', hint: 'đè một ký tự khác (lệch vị trí) lên ký tự → nhãn □' },
  // 'blot' (lem mực) is not offered: the line prints with a laser, which
  // cannot smear ink.
  { id: 'char_missing', label: 'Mất chữ', hint: 'xoá hẳn ký tự → nhãn bỏ luôn ký tự đó' },
];

// 'blot' kept for naming crops generated before it was hidden.
const TYPE_LABEL: Record<string, string> = { blot: 'Lem mực', ...Object.fromEntries(TYPES.map((t) => [t.id, t.label])) };

/** "Làm giàu dữ liệu" — synthetic print defects so the model learns to answer
 *  □ for a badly printed char instead of reading through it. */
export default function SyntheticDefectsPanel({
  projectId, testSplit, useSynthetic, onUseSyntheticChange,
}: Props) {
  const [info, setInfo] = useState<SyntheticInfo | null>(null);
  const [perImage, setPerImage] = useState(1);
  const [maxChars, setMaxChars] = useState(1);
  const [types, setTypes] = useState<SyntheticDefectType[]>(TYPES.map((t) => t.id));
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<SyntheticProgress | null>(null);

  // The generate request only returns when it is done; poll its progress meanwhile.
  useEffect(() => {
    if (busy !== 'generate') { setProgress(null); return; }
    const t = setInterval(() => {
      paddleOcrAPI.getSyntheticProgress(projectId)
        .then((p) => { if (p.running) setProgress(p); })
        .catch(() => { /* progress is cosmetic; the request itself reports errors */ });
    }, 800);
    return () => clearInterval(t);
  }, [busy, projectId]);

  useEffect(() => {
    setInfo(null);
    paddleOcrAPI.getSynthetic(projectId)
      .then((d) => {
        setInfo(d);
        if (!d.exists) onUseSyntheticChange(false);
        // Start from the settings the current data was made with.
        if (d.exists) { setMaxChars(d.max_chars ?? 1); setPerImage(d.per_image ?? 1); }
      })
      .catch((e) => console.error('[PaddleOCR] synthetic info', e));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  const generate = async () => {
    setBusy('generate'); setError(null);
    try {
      const d = await paddleOcrAPI.generateSynthetic(projectId, { per_image: perImage, types, test_split: testSplit, max_chars: maxChars });
      setInfo(d);
      onUseSyntheticChange(true);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Generation failed');
    } finally {
      setBusy(null);
    }
  };

  const remove = async () => {
    if (!confirm('Xoá toàn bộ dữ liệu lỗi tổng hợp của project này?')) return;
    setBusy('delete'); setError(null);
    try {
      await paddleOcrAPI.deleteSynthetic(projectId);
      setInfo({ exists: false, reject_char: '□' });
      onUseSyntheticChange(false);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Delete failed');
    } finally {
      setBusy(null);
    }
  };

  const toggle = (t: SyntheticDefectType) =>
    setTypes((cur) => cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]);

  return (
    <div style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, marginBottom: 12 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
        <h4 style={{ margin: 0 }}>Làm giàu dữ liệu lỗi</h4>
      </div>

      {error && <div className="at-alert-error">{error}</div>}

      {busy === 'generate' && (
        <div className="ot-synth-progress">
          <div className="ot-progress-meta">
            <span>
              {!progress || progress.phase === 'loading' ? 'Đang chuẩn bị…'
                : progress.phase === 'previews' ? 'Đang tạo ảnh xem trước…'
                : `Đang tạo mẫu lỗi: ${progress.done} / ${progress.total} ảnh`}
            </span>
            <b>{progress?.percent ?? 0}%</b>
          </div>
          <div className="ot-progress-bar">
            <div style={{ width: `${progress?.percent ?? 0}%` }} />
          </div>
        </div>
      )}

      <div className="ot-inline-row">
        {TYPES.map((t) => (
          <label key={t.id} className="at-label ot-check" title={t.hint}>
            <input type="checkbox" checked={types.includes(t.id)} onChange={() => toggle(t.id)} />
            {t.label}
          </label>
        ))}
        <label className="at-label ot-check"
               title="Mỗi ảnh lỗi làm hỏng ngẫu nhiên từ 1 đến số ký tự này (cùng một kiểu lỗi)">
          Số ký tự lỗi
          <select className="at-form-input" style={{ width: 90 }} value={maxChars}
                  onChange={(e) => setMaxChars(+e.target.value)}>
            {[1, 2, 3, 4, 5].map((n) => (
              <option key={n} value={n}>{n === 1 ? '1' : `1 – ${n}`}</option>
            ))}
          </select>
        </label>
        <label className="at-label ot-check">
          Mẫu lỗi / ảnh
          <select className="at-form-input" style={{ width: 70 }} value={perImage}
                  onChange={(e) => setPerImage(+e.target.value)}>
            {[1, 2, 3].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <button className="at-btn at-btn-secondary" onClick={generate}
                disabled={!!busy || types.length === 0}>
          {busy === 'generate' ? 'Đang tạo…' : info?.exists ? 'Tạo lại' : 'Tạo dữ liệu lỗi'}
        </button>
        {info?.exists && (
          <button className="at-btn at-btn-sm" style={{ background: '#fee2e2', color: '#b91c1c' }}
                  onClick={remove} disabled={!!busy}>Xoá</button>
        )}
      </div>

      {info?.exists && (
        <>
          <label className="at-label" style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8 }}>
            <input type="checkbox" checked={useSynthetic}
                   onChange={(e) => onUseSyntheticChange(e.target.checked)} />
            <b>Dùng dữ liệu lỗi khi train</b>
          </label>

          {(info.samples ?? []).length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
                          gap: 8, marginTop: 10 }}>
              {info.samples!.map((s, i) => (
                <div key={i} style={{ fontSize: 11, border: '1px solid #f1f5f9', borderRadius: 4, padding: 4 }}>
                  <img src={`data:image/png;base64,${s.image_b64}`} alt={s.label}
                       style={{ maxWidth: '100%', imageRendering: 'pixelated', display: 'block' }} />
                  <div style={{ marginTop: 2 }}>
                    <span className="ot-chip-sm">{TYPE_LABEL[s.type] ?? s.type}</span>{' '}
                    ký tự <b>{s.char}</b> · {(s.severity * 100).toFixed(0)}% · {s.split}
                  </div>
                  <div style={{ fontFamily: 'monospace' }}>
                    {s.source_label} → <b>{s.label}</b>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
