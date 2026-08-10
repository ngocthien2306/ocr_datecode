import { useCallback, useEffect, useState } from 'react';
import {
  ocrTrainingAPI, LabelStatus, OCRDatasetItem, OCRPrepareReport, SplitName,
} from '@/services/ocrTraining';

interface Props {
  projectId: string;
  refreshKey: number;
  onCountsChanged: () => void;
  onOpenImport: () => void;
}

const PAGE_SIZE = 40;

export default function DatasetTab({ projectId, refreshKey, onCountsChanged, onOpenImport }: Props) {
  const [status, setStatus] = useState<LabelStatus | ''>('');
  const [split, setSplit] = useState<SplitName | ''>('');
  const [page, setPage] = useState(1);
  const [items, setItems] = useState<OCRDatasetItem[]>([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [report, setReport] = useState<OCRPrepareReport | null>(null);
  const [seeding, setSeeding] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await ocrTrainingAPI.listItems(projectId, {
        status: status || undefined, split: split || undefined, page, pageSize: PAGE_SIZE,
      });
      setItems(res.items);
      setTotal(res.total);
      setTotalPages(res.total_pages || 1);
      setSel(new Set());
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Failed to load dataset');
    } finally {
      setLoading(false);
    }
  }, [projectId, status, split, page]);

  useEffect(() => { load(); }, [load, refreshKey]);
  useEffect(() => { setPage(1); }, [status, split]);

  const toggle = (id: string) => setSel((p) => {
    const n = new Set(p);
    n.has(id) ? n.delete(id) : n.add(id);
    return n;
  });

  const selectAllMatching = async () => {
    try {
      const r = await ocrTrainingAPI.listItemIds(projectId, status || undefined, split || undefined);
      setSel(new Set(r.ids));
      setInfo(`${r.total} item(s) selected across all pages.`);
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Select-all failed');
    }
  };

  const runBulk = async (fn: () => Promise<any>, label: string) => {
    setError(null);
    setInfo(null);
    try {
      const r = await fn();
      setInfo(`${label}: ${r.modified ?? r.deleted ?? 0} item(s).`
            + (r.skipped_empty_text ? ` ${r.skipped_empty_text} skipped (empty label).` : ''));
      onCountsChanged();
      await load();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || `${label} failed`);
    }
  };

  const ids = Array.from(sel);

  const seedFromFolder = async () => {
    if (!confirm('Seed this project from ocr_service/data_ocr_merged?\n\n'
               + 'Those labels come from rec_gt files that are already reviewed, so they '
               + 'arrive as verified and are immediately trainable.')) return;
    setSeeding(true);
    setError(null);
    try {
      const r = await ocrTrainingAPI.importFolder(projectId);
      setInfo(`Seeded ${r.imported} image(s) — train ${r.per_split.train ?? 0}, test ${r.per_split.test ?? 0}.`
            + (r.error_count ? ` ${r.error_count} error(s).` : ''));
      onCountsChanged();
      await load();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Seed failed');
    } finally {
      setSeeding(false);
    }
  };

  const runPrepare = async () => {
    setError(null);
    try {
      setReport(await ocrTrainingAPI.prepare(projectId, { dryRun: true }));
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Prepare check failed');
    }
  };

  return (
    <>
      <div className="ot-toolbar">
        <button className="at-btn at-btn-primary" onClick={onOpenImport}>+ Import from Recipe</button>
        <button className="at-btn at-btn-secondary" onClick={seedFromFolder} disabled={seeding}
                title="Bulk-load the bundled data_ocr_merged sample set (796 pre-labelled crops)">
          {seeding ? 'Seeding…' : 'Seed from data_ocr_merged'}
        </button>
        <button className="at-btn at-btn-secondary" onClick={runPrepare}
                title="Validate labels and show what a training run would actually see">
          Check dataset
        </button>

        <span className="ot-spacer" />

        <select className="at-form-input" style={{ maxWidth: 150 }} value={status}
                onChange={(e) => setStatus(e.target.value as LabelStatus | '')}>
          <option value="">all statuses</option>
          <option value="need_review">need_review</option>
          <option value="verified">verified</option>
          <option value="rejected">rejected</option>
        </select>
        <select className="at-form-input" style={{ maxWidth: 130 }} value={split}
                onChange={(e) => setSplit(e.target.value as SplitName | '')}>
          <option value="">train + test</option>
          <option value="train">train</option>
          <option value="test">test</option>
        </select>
      </div>

      {error && <div className="at-alert-error">{error}</div>}
      {info && <div className="at-alert-ok">{info}</div>}

      {report && (
        <div style={{ marginBottom: 12 }}>
          <div className="ot-prepare-grid">
            <div className="ot-stat-box">
              <div className="ot-stat-num">{report.n_train}</div>
              <div className="ot-stat-label">train images</div>
            </div>
            <div className="ot-stat-box">
              <div className="ot-stat-num">{report.n_test}</div>
              <div className="ot-stat-label">test images ({report.split_source})</div>
            </div>
            <div className={`ot-stat-box ${report.dropped_count ? 'ot-bad' : ''}`}>
              <div className="ot-stat-num">{report.dropped_count}</div>
              <div className="ot-stat-label">dropped</div>
            </div>
            <div className={`ot-stat-box ${report.unknown_char_count ? 'ot-warn' : ''}`}>
              <div className="ot-stat-num">{report.unknown_char_count}</div>
              <div className="ot-stat-label">non-dict chars</div>
            </div>
          </div>

          {report.blocking_reason && <div className="at-alert-error">{report.blocking_reason}</div>}

          {report.dropped_count > 0 && (
            <div className="at-hint">
              <b>{report.dropped_count} label(s) dropped.</b> A label longer than the model's
              max_text_length makes OpenOCR fetch a <i>different</i> image into that slot instead —
              random during training, the next one during eval — so the dataset size never changes
              and the accuracy quietly stops describing your test set. Shorten or reject these:
              <ul style={{ margin: '6px 0 0 18px' }}>
                {report.dropped.slice(0, 6).map((d) => (
                  <li key={d.id} style={{ fontFamily: 'monospace', fontSize: 11 }}>
                    [{d.length} chars] {d.gt_text}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {report.unknown_chars.length > 0 && (
            <div className="at-hint">
              Characters outside the model's dictionary: <b>{report.unknown_chars.join(' ')}</b>.
              These are stripped silently, so the model is taught to read the image as the
              stripped text. Worth fixing in the Label tab.
            </div>
          )}

          {Object.keys(report.by_recipe_train).length > 0 && (
            <div className="at-hint">
              Train mix:{' '}
              {Object.entries(report.by_recipe_train)
                .sort((a, b) => b[1] - a[1])
                .map(([k, v]) => `${k} ${v}`).join(' · ')}
              {' '}— a set dominated by one recipe trains a model that only reads that font.
            </div>
          )}
        </div>
      )}

      {sel.size > 0 && (
        <div className="at-bulk-bar">
          <b>{sel.size} selected</b>
          <button className="at-btn at-btn-primary at-btn-sm"
                  onClick={() => runBulk(() => ocrTrainingAPI.bulkStatus(projectId, ids, 'verified'), 'Verified')}>
            Verify
          </button>
          <button className="at-btn at-btn-secondary at-btn-sm"
                  onClick={() => runBulk(() => ocrTrainingAPI.bulkStatus(projectId, ids, 'need_review'), 'Reset to need_review')}>
            Un-verify
          </button>
          <button className="at-btn at-btn-secondary at-btn-sm"
                  onClick={() => runBulk(() => ocrTrainingAPI.bulkSplit(projectId, ids, 'test'), 'Moved to test')}>
            → test
          </button>
          <button className="at-btn at-btn-secondary at-btn-sm"
                  onClick={() => runBulk(() => ocrTrainingAPI.bulkSplit(projectId, ids, 'train'), 'Moved to train')}>
            → train
          </button>
          <button className="at-btn at-btn-secondary at-btn-sm"
                  onClick={() => runBulk(() => ocrTrainingAPI.bulkExclude(projectId, ids, true), 'Excluded')}
                  title="Hold out of the next run without deleting">
            Exclude
          </button>
          <button className="at-btn at-btn-secondary at-btn-sm"
                  onClick={() => runBulk(() => ocrTrainingAPI.bulkExclude(projectId, ids, false), 'Included')}>
            Include
          </button>
          <span className="ot-spacer" />
          <button className="at-btn at-btn-sm" style={{ background: '#fee2e2', color: '#b91c1c' }}
                  onClick={() => {
                    if (confirm(`Delete ${sel.size} item(s) and their image files? This cannot be undone.`)) {
                      runBulk(() => ocrTrainingAPI.bulkDelete(projectId, ids), 'Deleted');
                    }
                  }}>
            Delete
          </button>
        </div>
      )}

      <div className="ot-toolbar" style={{ paddingTop: 0 }}>
        <button className="at-btn at-btn-secondary at-btn-sm" onClick={selectAllMatching}>
          Select all matching ({total})
        </button>
        <button className="at-btn at-btn-secondary at-btn-sm" onClick={() => setSel(new Set())}
                disabled={sel.size === 0}>
          Clear selection
        </button>
      </div>

      {loading && <div className="at-empty-state" style={{ padding: 30 }}><div className="at-loading-spinner" /></div>}
      {!loading && items.length === 0 && (
        <div className="at-empty-state" style={{ padding: 40 }}>
          Nothing here yet — import from a recipe, or seed from data_ocr_merged to get going.
        </div>
      )}

      {!loading && items.map((item) => (
        <div key={item.id} className={`ot-label-row is-${item.status}`}
             style={{ gridTemplateColumns: '28px 1fr 300px' }}>
          <input type="checkbox" checked={sel.has(item.id)} onChange={() => toggle(item.id)} />
          <div className="ot-crop-wrap">
            {item.thumb_b64
              ? <img className="ot-crop-img" src={`data:image/jpeg;base64,${item.thumb_b64}`} alt={item.gt_text} />
              : <span className="at-hint">image missing</span>}
          </div>
          <div className="ot-label-fields">
            <div style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 600 }}>
              {item.gt_text || <span className="at-hint">(no label)</span>}
            </div>
            <div className="ot-meta-chips">
              <span className="ot-chip-sm">{item.status}</span>
              <span className="ot-chip-sm">{item.split}</span>
              {item.exclude_from_training && <span className="ot-chip-sm ot-fail">excluded</span>}
              {item.verify_match === true && <span className="ot-chip-sm ot-pass">match</span>}
              {item.verify_match === false && <span className="ot-chip-sm ot-fail">no match</span>}
              {item.recipe_name && <span className="ot-chip-sm">{item.recipe_name}</span>}
              <span className="ot-chip-sm">{item.source}</span>
            </div>
          </div>
        </div>
      ))}

      {totalPages > 1 && (
        <div className="ot-toolbar">
          <button className="at-btn at-btn-secondary at-btn-sm" disabled={page <= 1}
                  onClick={() => setPage((p) => p - 1)}>← Prev</button>
          <span className="at-hint">Page {page} / {totalPages} · {total} item(s)</span>
          <button className="at-btn at-btn-secondary at-btn-sm" disabled={page >= totalPages}
                  onClick={() => setPage((p) => p + 1)}>Next →</button>
        </div>
      )}
    </>
  );
}
