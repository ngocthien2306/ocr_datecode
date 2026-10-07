import { useEffect, useState } from 'react';
import {
  paddleOcrAPI, SyntheticDefectType, SyntheticInfo,
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
  { id: 'blot', label: 'Lem mực', hint: 'vệt mực che một phần ký tự → nhãn □' },
  { id: 'char_missing', label: 'Mất chữ', hint: 'xoá hẳn ký tự → nhãn bỏ luôn ký tự đó' },
];

const TYPE_LABEL: Record<string, string> = Object.fromEntries(TYPES.map((t) => [t.id, t.label]));

/** "Làm giàu dữ liệu" — synthetic print defects so the model learns to answer
 *  □ for a badly printed char instead of reading through it. */
export default function SyntheticDefectsPanel({
  projectId, testSplit, useSynthetic, onUseSyntheticChange,
}: Props) {
  const [info, setInfo] = useState<SyntheticInfo | null>(null);
  const [perImage, setPerImage] = useState(1);
  const [types, setTypes] = useState<SyntheticDefectType[]>(TYPES.map((t) => t.id));
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setInfo(null);
    paddleOcrAPI.getSynthetic(projectId)
      .then((d) => { setInfo(d); if (!d.exists) onUseSyntheticChange(false); })
      .catch((e) => console.error('[PaddleOCR] synthetic info', e));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  const generate = async () => {
    setBusy('generate'); setError(null);
    try {
      const d = await paddleOcrAPI.generateSynthetic(projectId, { per_image: perImage, types, test_split: testSplit });
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
        <span className="at-hint">
          tạo ảnh bị đứt nét / mất chữ / chồng chữ từ ảnh đã verify, để model trả “□” thay vì
          cố đọc chữ in xấu → dây chuyền FAIL chai đó
        </span>
      </div>

      {error && <div className="at-alert-error">{error}</div>}

      <div className="at-form-row" style={{ flexWrap: 'wrap', gap: 12, alignItems: 'flex-end' }}>
        {TYPES.map((t) => (
          <label key={t.id} className="at-label" title={t.hint}
                 style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <input type="checkbox" checked={types.includes(t.id)} onChange={() => toggle(t.id)} />
            {t.label}
          </label>
        ))}
        <label className="at-label">
          Mẫu lỗi / ảnh
          <select className="at-form-input" style={{ width: 80 }} value={perImage}
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
          <div className="ot-hint-row" style={{ marginTop: 8 }}>
            <span><b>{info.n_train}</b> train + <b>{info.n_test}</b> test (chỉ để đo)</span>
            <span>định vị được {info.n_localised}/{info.n_items} ảnh</span>
            {Object.entries(info.by_type ?? {}).filter(([, n]) => n > 0).map(([t, n]) => (
              <span key={t} className="ot-chip-sm">{TYPE_LABEL[t] ?? t} {n}</span>
            ))}
            <span className="at-hint">vị trí ký tự lấy từ {info.localiser}</span>
          </div>

          <label className="at-label" style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8 }}>
            <input type="checkbox" checked={useSynthetic}
                   onChange={(e) => onUseSyntheticChange(e.target.checked)} />
            <b>Dùng dữ liệu lỗi khi train</b>
            <span className="at-hint">
              — sau khi train, run được chấm thêm “bắt lỗi”: % ảnh lỗi (từ tập test) mà dây chuyền sẽ FAIL
            </span>
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
