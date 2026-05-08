import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  mlTrainingAPI,
  LabeledCrop,
  SyntheticCrop,
  SyntheticOkCrop,
  MLModel,
  MLProject,
  TestSetCropResult,
  TrainingLogEntry,
  TrainRequest,
} from '@/services/mlTraining';

interface CharBucket<T> {
  char_id: string | null;  // null = no char_id (still trained, just shown separately)
  items: T[];
  ok: number;
  ng: number;
}

/**
 * Group crops by char_id for display only (no training-status semantics).
 * char_id == null lands in a "No char_id" bucket sorted last.
 */
function bucketByChar<T extends { char_id?: string | null; label: string }>(
  items: T[],
  sortMode: 'alpha' | 'wrong-desc' = 'alpha',
  wrongCounts?: Map<string, number>,
): CharBucket<T>[] {
  const map = new Map<string, CharBucket<T>>();
  const keyOf = (cid: string | null | undefined): string => (cid && cid.trim() ? cid : '__null__');

  for (const item of items) {
    const key = keyOf(item.char_id);
    if (!map.has(key)) {
      map.set(key, { char_id: key === '__null__' ? null : key, items: [], ok: 0, ng: 0 });
    }
    const b = map.get(key)!;
    b.items.push(item);
    if (item.label === 'OK') b.ok += 1;
    else if (item.label === 'NG') b.ng += 1;
  }

  const buckets = Array.from(map.values());
  if (sortMode === 'wrong-desc' && wrongCounts) {
    buckets.sort((a, b) => {
      const wa = wrongCounts.get(a.char_id ?? '__null__') ?? 0;
      const wb = wrongCounts.get(b.char_id ?? '__null__') ?? 0;
      if (wa !== wb) return wb - wa;  // more wrongs first
      return (a.char_id ?? '~').localeCompare(b.char_id ?? '~');
    });
  } else {
    buckets.sort((a, b) => {
      // null last
      if (a.char_id === null && b.char_id !== null) return 1;
      if (a.char_id !== null && b.char_id === null) return -1;
      return (a.char_id ?? '').localeCompare(b.char_id ?? '');
    });
  }
  return buckets;
}

interface Props {
  project: MLProject;
  onRefresh: () => void;
}

const AUGMENT_OPTIONS = [
  { value: 0, label: 'Off' },
  { value: 2, label: '×2' },
  { value: 3, label: '×3' },
  { value: 4, label: '×4' },
  { value: 5, label: '×5' },
];

/**
 * Char-balanced NG augmentation math — mirrors backend build_dataset.
 * target_ng_per_char = factor * max(n_ok_real across chars).
 * Each char's NG is topped up to target; only chars with OK samples count.
 *
 * Returns per-class totals for display in the training preview.
 */
function computeAugmentStats(
  crops: { char_id?: string | null; label: string }[],
  factor: number,
): { nOkReal: number; nNgReal: number; nNgAug: number; totalOk: number; totalNg: number } {
  const okByChar = new Map<string, number>();
  const ngByChar = new Map<string, number>();
  for (const c of crops) {
    const cid = (c.char_id || '').trim();
    if (!cid) continue;  // no char_id → excluded from training
    if (c.label === 'OK') okByChar.set(cid, (okByChar.get(cid) || 0) + 1);
    else if (c.label === 'NG') ngByChar.set(cid, (ngByChar.get(cid) || 0) + 1);
  }
  const nOkReal = Array.from(okByChar.values()).reduce((a, b) => a + b, 0);
  const nNgReal = Array.from(ngByChar.values()).reduce((a, b) => a + b, 0);

  let nNgAug = 0;
  if (factor >= 2 && okByChar.size > 0) {
    const maxOk = Math.max(...okByChar.values());
    const target = factor * maxOk;
    for (const cid of okByChar.keys()) {
      const deficit = Math.max(0, target - (ngByChar.get(cid) || 0));
      nNgAug += deficit;
    }
  }
  return { nOkReal, nNgReal, nNgAug, totalOk: nOkReal, totalNg: nNgReal + nNgAug };
}

// ── Lazy image: only renders <img> when it enters the viewport ─────────────
function LazyImage({ src, alt }: { src: string; alt: string }) {
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry!.isIntersecting) { setVisible(true); obs.disconnect(); } },
      { rootMargin: '300px' }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div ref={ref} style={{ width: '52px', height: '36px', background: '#0f1117', borderRadius: '3px' }}>
      {visible && (
        <img src={src} alt={alt}
          style={{ width: '52px', height: '36px', objectFit: 'contain', display: 'block' }} />
      )}
    </div>
  );
}

// ── Stats banner showing per-char coverage at a glance ─────────────────────
interface CharStatsBannerProps {
  buckets: CharBucket<any>[];
  viewMode: 'grouped' | 'flat';
  onToggleView: (mode: 'grouped' | 'flat') => void;
}
function CharStatsBanner({ buckets, viewMode, onToggleView }: CharStatsBannerProps) {
  const total = buckets.length;
  const totalOk = buckets.reduce((s, b) => s + b.ok, 0);
  const totalNg = buckets.reduce((s, b) => s + b.ng, 0);

  return (
    <div className="ml-char-stats-banner">
      <span><b>{total}</b> char{total !== 1 ? 's' : ''}</span>
      <span style={{ color: '#22c55e' }}>{totalOk} OK</span>
      <span style={{ color: '#ef4444' }}>{totalNg} NG</span>
      <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
        {(['grouped', 'flat'] as const).map(mode => (
          <button key={mode}
            className={`ml-view-toggle-btn${viewMode === mode ? ' active' : ''}`}
            onClick={() => onToggleView(mode)}
          >{mode === 'grouped' ? '📋 Grouped' : '📃 Flat'}</button>
        ))}
      </div>
    </div>
  );
}

// ── Grouped crops (sections per char_id, collapsible) ──────────────────────
interface CharGroupedCropsProps {
  buckets: CharBucket<{ crop_b64: string; label: string; char_id?: string | null }>[];
  expanded: Set<string>;
  onToggleExpand: (key: string) => void;
  emptyText: string;
}
function CharGroupedCrops({ buckets, expanded, onToggleExpand, emptyText }: CharGroupedCropsProps) {
  if (buckets.length === 0) {
    return (
      <div className="ml-empty-state" style={{ minHeight: '100px' }}>
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.4 }}>
          <path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z"
            stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          <circle cx="7" cy="7" r="1.5" fill="currentColor" />
        </svg>
        <span style={{ fontSize: '12px', marginTop: '6px' }}>{emptyText}</span>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {buckets.map(b => {
        const key = b.char_id ?? '__null__';
        const isOpen = expanded.has(key);
        return (
          <div key={key} className="ml-char-group">
            <button className="ml-char-group-header" onClick={() => onToggleExpand(key)}>
              <span style={{ width: 12, textAlign: 'center' }}>{isOpen ? '▼' : '▶'}</span>
              <span className="ml-char-badge">{b.char_id ?? '—'}</span>
              <span style={{ color: '#22c55e' }}>{b.ok} OK</span>
              <span style={{ color: '#ef4444' }}>{b.ng} NG</span>
            </button>
            {isOpen && (
              <div className="ml-crops-grid" style={{ padding: 6 }}>
                {b.items.map((item, i) => (
                  <div key={i} className="ml-crop-card" style={{ position: 'relative' }}>
                    <LazyImage src={`data:image/jpeg;base64,${item.crop_b64}`} alt={`crop-${i}`} />
                    <span className={`ml-label-badge ${item.label === 'OK' ? 'ok' : 'ng'}`}>{item.label}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── CropGrid ───────────────────────────────────────────────────────────────
function CropGrid({ items, emptyText }: {
  items: Array<{ crop_b64: string; label: string; char_id?: string | null }>;
  emptyText: string;
}) {
  if (items.length === 0) {
    return (
      <div className="ml-empty-state" style={{ minHeight: '100px' }}>
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.4 }}>
          <path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z"
            stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          <circle cx="7" cy="7" r="1.5" fill="currentColor" />
        </svg>
        <span style={{ fontSize: '12px', color: '#6b7280', marginTop: '6px' }}>{emptyText}</span>
      </div>
    );
  }
  return (
    <div className="ml-crops-grid">
      {items.map((crop, i) => (
        <div key={i} className="ml-crop-card" style={{ position: 'relative' }}>
          <LazyImage src={`data:image/jpeg;base64,${crop.crop_b64}`} alt={`crop-${i}`} />
          <span className={`ml-label-badge ${crop.label === 'OK' ? 'ok' : 'ng'}`}>{crop.label}</span>
          {crop.char_id && (
            <span
              title={`char_id: ${crop.char_id}`}
              style={{
                position: 'absolute', top: 2, right: 2,
                fontSize: 10, fontWeight: 600, padding: '1px 5px',
                borderRadius: 3, background: '#3b82f6', color: '#fff',
                fontFamily: 'monospace', lineHeight: 1.3,
                boxShadow: '0 1px 2px rgba(0,0,0,.3)',
              }}
            >{crop.char_id}</span>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Prediction result card with optional diff heatmap ─────────────────────
interface PredictCardItem {
  crop_b64: string;
  label: string;
  prob_ok: number;
  char_id?: string | null;
  aligned_b64?: string | null;
  golden_b64?: string | null;
  diff_b64?: string | null;
}

function PredictResultCard({ result }: { result: PredictCardItem }) {
  const [expanded, setExpanded] = useState(false);
  const hasHeatmap = Boolean(result.diff_b64 && result.golden_b64);

  return (
    <div className={`ml-predict-card ${result.label.toLowerCase()}`} style={{ position: 'relative' }}>
      <img src={`data:image/jpeg;base64,${result.crop_b64}`} alt="predict crop" />
      <span className={`ml-label-badge ${result.label === 'OK' ? 'ok' : 'ng'}`}>{result.label}</span>
      <span className="ml-predict-prob">{(result.prob_ok * 100).toFixed(0)}% OK</span>
      {result.char_id && (
        <span style={{
          position: 'absolute', top: 2, right: 2,
          fontSize: 10, fontWeight: 600, padding: '1px 5px',
          borderRadius: 3, background: '#3b82f6', color: '#fff',
          fontFamily: 'monospace', lineHeight: 1.3,
        }}>{result.char_id}</span>
      )}
      {hasHeatmap && (
        <button
          onClick={() => setExpanded(v => !v)}
          title={expanded ? 'Hide heatmap' : 'Show diff heatmap'}
          style={{
            position: 'absolute', bottom: 2, right: 2,
            width: 18, height: 18, padding: 0,
            background: 'rgba(59,130,246,.9)', color: '#fff',
            border: 'none', borderRadius: 3, cursor: 'pointer',
            fontSize: 11, lineHeight: 1,
          }}
        >{expanded ? '−' : '🔥'}</button>
      )}
      {expanded && hasHeatmap && (
        <div className="ml-golden-card" style={{
          position: 'absolute', top: '100%', left: 0, right: 0,
          marginTop: 4, padding: 6,
          zIndex: 10, boxShadow: '0 4px 12px rgba(0,0,0,.15)',
        }}>
          <div style={{ display: 'flex', gap: 4, justifyContent: 'space-between', width: '100%' }}>
            <HeatmapPanel label="Input" src={result.aligned_b64 || result.crop_b64} />
            <HeatmapPanel label="Golden" src={result.golden_b64!} />
            <HeatmapPanel label="Diff" src={result.diff_b64!} />
          </div>
        </div>
      )}
    </div>
  );
}

function HeatmapPanel({ label, src }: { label: string; src: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, flex: 1 }}>
      <img src={`data:image/jpeg;base64,${src}`} alt={label}
        className="ml-golden-img"
        style={{ maxWidth: 64, height: 64 }} />
      <span style={{ fontSize: 9, opacity: 0.7 }}>{label}</span>
    </div>
  );
}


// ── Single test-set crop card ──────────────────────────────────────────────
function TestSetCropCard({ item }: { item: TestSetCropResult }) {
  const wrong = !item.correct;
  return (
    <div className={`ml-testset-card${wrong ? ' wrong' : ''}`}>
      {/* Wrong indicator */}
      {wrong && (
        <div style={{
          position: 'absolute', top: '-5px', right: '-5px',
          width: '14px', height: '14px', borderRadius: '50%',
          background: '#ef4444', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <svg width="8" height="8" viewBox="0 0 24 24" fill="none">
            <path d="M18 6L6 18M6 6l12 12" stroke="white" strokeWidth="3" strokeLinecap="round" />
          </svg>
        </div>
      )}
      <img src={`data:image/jpeg;base64,${item.crop_b64}`} alt="crop" />
      {/* True → Pred */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
        <span className={`ml-label-badge ${item.true_label === 'OK' ? 'ok' : 'ng'}`}
          style={{ fontSize: '9px', padding: '1px 4px' }}>{item.true_label}</span>
        <svg width="8" height="8" viewBox="0 0 24 24" fill="none">
          <path d="M5 12h14M12 5l7 7-7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            opacity="0.4" />
        </svg>
        <span className={`ml-label-badge ${item.pred_label === 'OK' ? 'ok' : 'ng'}`}
          style={{ fontSize: '9px', padding: '1px 4px', opacity: wrong ? 1 : 0.7 }}>{item.pred_label}</span>
      </div>
      <span style={{ fontSize: '9px', opacity: 0.6 }}>{(item.prob_ok * 100).toFixed(0)}% OK</span>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────
export default function TrainTab({ project, onRefresh }: Props) {
  // Crops
  const [crops, setCrops] = useState<LabeledCrop[]>([]);
  const [loadingCrops, setLoadingCrops] = useState(false);
  const [cropsTab, setCropsTab] = useState<'real' | 'synthetic'>('real');
  const [cropFilter, setCropFilter] = useState<'all' | 'OK' | 'NG'>('all');
  const [viewMode, setViewMode] = useState<'grouped' | 'flat'>('grouped');
  const [expandedChars, setExpandedChars] = useState<Set<string>>(new Set());
  const toggleCharExpanded = useCallback((key: string) => {
    setExpandedChars(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }, []);

  // Synthetic NG preview
  const [syntheticCrops, setSyntheticCrops] = useState<SyntheticCrop[]>([]);
  const [loadingSynthetic, setLoadingSynthetic] = useState(false);

  // Synthetic OK preview (font-render)
  const [syntheticOkCrops, setSyntheticOkCrops] = useState<SyntheticOkCrop[]>([]);
  const [loadingSynthOk, setLoadingSynthOk] = useState(false);
  const [okSynthTarget, setOkSynthTarget] = useState(15);

  // Training config
  const [algorithm, setAlgorithm] = useState<'rf' | 'svm' | 'mlp' | 'centroid'>('rf');
  const [augmentFactor, setAugmentFactor] = useState(0);
  const [nEstimators, setNEstimators] = useState(100);
  const [maxIter, setMaxIter] = useState(500);
  const [svmC, setSvmC] = useState(1.0);
  const [threshold, setThreshold] = useState(50); // percent, 0–100
  const [centroidTemperature, setCentroidTemperature] = useState(5.0);
  const [includeImportedChars, setIncludeImportedChars] = useState(true);
  const [importedStats, setImportedStats] = useState<{ ok: number; ng: number; batches: number }>({ ok: 0, ng: 0, batches: 0 });
  // NG augmentation severity weights — auto-normalized BE side
  const [severitySubtle, setSeveritySubtle] = useState(10);
  const [severityLight,  setSeverityLight ] = useState(50);
  const [severityMedium, setSeverityMedium] = useState(35);
  const [severityHeavy,  setSeverityHeavy ] = useState(5);
  const severityDist = useMemo(() => ({
    subtle: severitySubtle,
    light:  severityLight,
    medium: severityMedium,
    heavy:  severityHeavy,
  }), [severitySubtle, severityLight, severityMedium, severityHeavy]);

  // Training state
  const [training, setTraining] = useState(false);
  const [models, setModels] = useState<MLModel[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Live training progress
  const [trainLogs, setTrainLogs] = useState<TrainingLogEntry[]>([]);
  const [trainPhase, setTrainPhase] = useState<string | null>(null);
  const [trainProgress, setTrainProgress] = useState(0);
  const [trainStartedAt, setTrainStartedAt] = useState<number | null>(null);
  const [, setNowTick] = useState(0);     // forces re-render every 1s during training (for elapsed/ETA)
  const logScrollRef = useRef<HTMLDivElement>(null);

  // Model selection
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const completedModels = useMemo(() => models.filter(m => m.status === 'completed'), [models]);
  const selectedModel = useMemo(
    () => completedModels.find(m => m.id === selectedModelId) ?? completedModels[0] ?? null,
    [completedModels, selectedModelId]
  );

  // Results panel tabs
  const [resultsTab, setResultsTab] = useState<'metrics' | 'testset'>('metrics');

  // Predict (single image)
  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictResults, setPredictResults] = useState<any[] | null>(null);
  const predictInputRef = useRef<HTMLInputElement>(null);

  // Test set crops
  const [testSetCrops, setTestSetCrops] = useState<TestSetCropResult[]>([]);
  const [loadingTestSet, setLoadingTestSet] = useState(false);
  const [testSetFilter, setTestSetFilter] = useState<'all' | 'correct' | 'wrong'>('all');
  const [testsetExpanded, setTestsetExpanded] = useState<Set<string>>(new Set());

  // ── Load crops ────────────────────────────────────────────────────────
  const loadCrops = useCallback(async () => {
    setLoadingCrops(true);
    try {
      const data = await mlTrainingAPI.getLabeledCrops(project.id);
      setCrops(data.crops);
    } catch { /* ignore */ }
    finally { setLoadingCrops(false); }
  }, [project.id]);

  // ── Load models ───────────────────────────────────────────────────────
  const loadModels = useCallback(async () => {
    try {
      const list = await mlTrainingAPI.listModels(project.id);
      setModels(list);
      // Auto-select first completed model on initial load only
      setSelectedModelId(prev => {
        if (prev) return prev;
        return list.find(m => m.status === 'completed')?.id ?? null;
      });
    } catch { /* ignore */ }
  }, [project.id]);

  // ── Load imported chars stats (count summary for the toggle) ──────────
  const loadImportedStats = useCallback(async () => {
    try {
      const list = await mlTrainingAPI.listCharImportBatches(project.id);
      const ok = list.reduce((s, b) => s + b.ok_count, 0);
      const ng = list.reduce((s, b) => s + b.ng_count, 0);
      setImportedStats({ ok, ng, batches: list.length });
    } catch { /* ignore */ }
  }, [project.id]);

  useEffect(() => {
    loadCrops();
    loadModels();
    loadImportedStats();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [project.id]);

  // Auto-select newly completed model after training
  const handleModelCompleted = useCallback((modelId: string) => {
    setSelectedModelId(modelId);
    setResultsTab('metrics');
    setTestSetCrops([]);
  }, []);

  // ── Preview synthetic NG (only NG is generated — no OK augmentation) ──
  const handlePreviewSynthetic = async () => {
    if (augmentFactor < 2) return;
    setLoadingSynthetic(true);
    setSyntheticCrops([]);
    setCropsTab('synthetic');
    try {
      const data = await mlTrainingAPI.previewSynthetic(project.id, augmentFactor, 'NG', severityDist);
      setSyntheticCrops(data.crops);
    } catch { /* ignore */ }
    finally { setLoadingSynthetic(false); }
  };

  const handlePreviewSynthOk = async () => {
    setLoadingSynthOk(true);
    setSyntheticOkCrops([]);
    setCropsTab('synthetic');
    try {
      const data = await mlTrainingAPI.previewSyntheticOk(project.id, {
        target_n_per_char: okSynthTarget,
        only_below_threshold: true,
      });
      setSyntheticOkCrops(data.crops);
    } catch (e: any) {
      alert(e?.response?.data?.detail ?? 'OK synthesis failed');
    }
    finally { setLoadingSynthOk(false); }
  };

  // ── Start training ────────────────────────────────────────────────────
  const handleTrain = async () => {
    setTraining(true);
    setPredictResults(null);
    setTestSetCrops([]);
    setTrainLogs([]);
    setTrainPhase(null);
    setTrainProgress(0);
    setTrainStartedAt(Date.now());
    try {
      const req: TrainRequest = {
        algorithm,
        augment_factor: augmentFactor,
        threshold: threshold / 100,
        n_estimators: nEstimators,
        max_iter: maxIter,
        C: svmC,
        severity_dist: severityDist,
        ok_synth_target: okSynthTarget,
        centroid_temperature: centroidTemperature,
        include_imported_chars: includeImportedChars,
      };
      const { model_id } = await mlTrainingAPI.startTraining(project.id, req);

      let since = 0;
      pollRef.current = setInterval(async () => {
        try {
          const data = await mlTrainingAPI.getTrainingLogs(project.id, model_id, since);
          if (data.logs.length > 0) {
            setTrainLogs(prev => [...prev, ...data.logs]);
            since = data.next_since;
          }
          if (data.phase != null) setTrainPhase(data.phase);
          if (data.progress != null) setTrainProgress(data.progress);
          if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            setTraining(false);
            await loadModels();
            await onRefresh();
            if (data.status === 'completed') handleModelCompleted(model_id);
          }
        } catch { /* ignore */ }
      }, 1500);
    } catch (e: any) {
      setTraining(false);
      alert(e?.response?.data?.detail ?? 'Training failed');
    }
  };

  // Tick every 1s during training so elapsed/ETA update without log activity.
  useEffect(() => {
    if (!training) return;
    const t = setInterval(() => setNowTick(v => v + 1), 1000);
    return () => clearInterval(t);
  }, [training]);

  // Auto-scroll log panel to bottom when new logs arrive.
  useEffect(() => {
    if (logScrollRef.current) {
      logScrollRef.current.scrollTop = logScrollRef.current.scrollHeight;
    }
  }, [trainLogs.length]);

  // Friendly phase label + estimated duration helpers
  const phaseLabel = (p: string | null): string => {
    switch (p) {
      case 'preparing':           return 'Đang chuẩn bị dữ liệu';
      case 'training_classifier': return 'Đang train classifier';
      case 'evaluating':          return 'Đang đánh giá';
      case 'encoding_testset':    return 'Đang encode test set';
      case 'saving':              return 'Đang lưu model';
      case 'completed':           return 'Hoàn tất';
      case 'failed':              return 'Thất bại';
      default:                    return p ? p : 'Đang khởi động';
    }
  };
  const fmtDuration = (sec: number): string => {
    if (!isFinite(sec) || sec < 0) return '—';
    const s = Math.round(sec);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60), r = s % 60;
    return r === 0 ? `${m}m` : `${m}m${r}s`;
  };
  const elapsedSec = trainStartedAt ? (Date.now() - trainStartedAt) / 1000 : 0;
  const etaSec = (trainProgress > 0 && trainProgress < 100)
    ? (elapsedSec / trainProgress) * (100 - trainProgress)
    : null;

  // ── Download model report (JSON with embedded images) ─────────────────
  const [downloadingReport, setDownloadingReport] = useState(false);
  const handleDownloadReport = async () => {
    if (!selectedModel) return;
    setDownloadingReport(true);
    try {
      const data = await mlTrainingAPI.getModelReport(project.id, selectedModel.id);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `model_${selectedModel.id}_report.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e: any) {
      alert(e?.response?.data?.detail ?? 'Report download failed');
    } finally { setDownloadingReport(false); }
  };

  // ── Predict (single image) ────────────────────────────────────────────
  const handlePredict = async () => {
    if (!predictFile || !selectedModel) return;
    setPredicting(true);
    setPredictResults(null);
    try {
      const res = await mlTrainingAPI.predict(project.id, predictFile, selectedModel.id);
      setPredictResults(res.results);
    } catch (e: any) {
      alert(e?.response?.data?.detail ?? 'Prediction failed');
    } finally { setPredicting(false); }
  };

  // ── Load test set crops ───────────────────────────────────────────────
  const loadTestSetCrops = useCallback(async (modelId: string) => {
    setLoadingTestSet(true);
    setTestSetCrops([]);
    try {
      const data = await mlTrainingAPI.getTestSetCrops(project.id, modelId);
      setTestSetCrops(data.crops);
    } catch { /* ignore */ }
    finally { setLoadingTestSet(false); }
  }, [project.id]);

  // Auto-load when switching to test set tab or changing model
  useEffect(() => {
    if (resultsTab === 'testset' && selectedModel) {
      loadTestSetCrops(selectedModel.id);
    }
  }, [resultsTab, selectedModel?.id]);

  // ── Per-char accuracy (computed from testSetCrops) ───────────────────────
  const perCharAccuracy = useMemo(() => {
    const map = new Map<string, { char_id: string | null; n: number; correct: number; wrong: number }>();
    for (const item of testSetCrops) {
      const key = item.char_id ?? '__null__';
      if (!map.has(key)) {
        map.set(key, { char_id: item.char_id ?? null, n: 0, correct: 0, wrong: 0 });
      }
      const e = map.get(key)!;
      e.n += 1;
      if (item.correct) e.correct += 1; else e.wrong += 1;
    }
    return Array.from(map.values())
      .map(e => ({ ...e, acc: e.n ? e.correct / e.n : 0 }))
      .sort((a, b) => {
        if (a.wrong !== b.wrong) return b.wrong - a.wrong;
        return (a.char_id ?? '~').localeCompare(b.char_id ?? '~');
      });
  }, [testSetCrops]);

  // ── Stats ──────────────────────────────────────────────────────────────
  const okCrops  = crops.filter(c => c.label === 'OK');
  const ngCrops  = crops.filter(c => c.label === 'NG');
  const canTrain = okCrops.length + ngCrops.length >= 2 && !training;
  const filteredCrops = cropFilter === 'all' ? crops : cropFilter === 'OK' ? okCrops : ngCrops;

  // Model label helper
  const modelLabel = (m: MLModel) => {
    const date = new Date(m.created_at).toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
    const acc  = m.metrics?.accuracy_test != null ? ` · ${(m.metrics.accuracy_test * 100).toFixed(1)}%` : '';
    const thr  = m.params?.threshold != null ? ` · thr ${Math.round((m.params.threshold as number) * 100)}%` : '';
    return `${m.algorithm.toUpperCase()}${acc}${thr} · ${date}`;
  };

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div className="ml-train-tab">

      {/* ── Left: Config ────────────────────────────────────────────────── */}
      <div className="ml-train-left">
        <div className="ml-section-title">Dataset</div>
        <div className="ml-metric-row">
          <div className="ml-metric-card">
            <div className="ml-metric-value" style={{ color: '#4ade80' }}>{okCrops.length}</div>
            <div className="ml-metric-label">OK samples</div>
          </div>
          <div className="ml-metric-card">
            <div className="ml-metric-value" style={{ color: '#f87171' }}>{ngCrops.length}</div>
            <div className="ml-metric-label">NG samples</div>
          </div>
        </div>

        {/* Imported Chars pool toggle */}
        <div className="ml-form-group" style={{ marginTop: 4 }}>
          <label style={{ display: 'flex', alignItems: 'flex-start', gap: 8, cursor: 'pointer', padding: '4px 0' }}>
            <input type="checkbox" checked={includeImportedChars}
                   onChange={e => setIncludeImportedChars(e.target.checked)}
                   disabled={importedStats.batches === 0} />
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>
                Include imported chars
              </span>
              <span style={{ fontSize: 10, opacity: 0.65 }}>
                {importedStats.batches === 0
                  ? 'No imported chars yet — open the Imported Chars tab to add some.'
                  : `${importedStats.ok} OK · ${importedStats.ng} NG across ${importedStats.batches} batch(es)`}
              </span>
            </div>
          </label>
        </div>

        <div className="ml-section-title">Algorithm</div>
        <div className="ml-form-group">
          {([
            { key: 'rf',       label: 'Random Forest',     tag: 'Binary (needs OK + NG)' },
            { key: 'svm',      label: 'SVM (RBF kernel)',  tag: 'Binary (needs OK + NG)' },
            { key: 'mlp',      label: 'Neural Net (MLP)',  tag: 'Binary (needs OK + NG)' },
            { key: 'centroid', label: 'Nearest Centroid',  tag: 'Global mean OK + mean NG (no model fit)' },
          ] as const).map(opt => (
            <label key={opt.key}
              style={{
                display: 'flex', alignItems: 'flex-start', gap: 8, cursor: 'pointer',
                padding: '4px 0', opacity: algorithm === opt.key ? 1 : 0.7,
              }}>
              <input type="radio" name="algorithm" value={opt.key}
                checked={algorithm === opt.key}
                onChange={() => setAlgorithm(opt.key)} />
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span style={{ fontSize: '13px', fontWeight: algorithm === opt.key ? 600 : 400 }}>
                  {opt.label}
                </span>
                <span style={{ fontSize: '10px', opacity: 0.65 }}>{opt.tag}</span>
              </div>
            </label>
          ))}
        </div>

        {algorithm === 'rf' && (
          <div className="ml-form-group">
            <label className="ml-form-label">N estimators</label>
            <input className="ml-form-input" type="number" min={10} max={500} step={10}
              value={nEstimators} onChange={e => setNEstimators(Number(e.target.value))} />
          </div>
        )}
        {(algorithm === 'svm' || algorithm === 'mlp') && (
          <div className="ml-form-group">
            <label className="ml-form-label">{algorithm === 'svm' ? 'C (regularization)' : 'Max iterations'}</label>
            <input className="ml-form-input" type="number"
              value={algorithm === 'svm' ? svmC : maxIter}
              onChange={e => algorithm === 'svm' ? setSvmC(Number(e.target.value)) : setMaxIter(Number(e.target.value))} />
          </div>
        )}
        {algorithm === 'centroid' && (
          <div className="ml-form-group">
            <label className="ml-form-label">Temperature (sigmoid scale)</label>
            <input className="ml-form-input" type="number" step="0.5" min={1} max={20}
              value={centroidTemperature}
              onChange={e => setCentroidTemperature(Number(e.target.value) || 5.0)} />
            <div style={{ fontSize: 10, color: '#9ca3af', marginTop: 4 }}>
              p_ok = sigmoid((sim_ok − sim_ng) × T). Higher T → sharper decision; lower → softer.
            </div>
          </div>
        )}

        {/* ── Synthetic OK (font-render) ── */}
        <div className="ml-section-title">Synthetic OK (font-render)</div>
        <div className="ml-form-group">
          <div style={{ fontSize: '11px', color: '#6b7280', marginBottom: '6px' }}>
            Top-up under-represented chars by rendering glyphs onto real backgrounds
            (style fingerprint extracted from real OK).
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <label style={{ fontSize: '11px', color: '#9ca3af' }}>Target / char</label>
            <input type="number" min={0} max={500}
              value={okSynthTarget}
              onChange={e => setOkSynthTarget(Math.max(0, Number(e.target.value) || 0))}
              className="ml-form-input"
              style={{ width: 64, textAlign: 'center', padding: '4px 6px' }} />
            <button className="ml-btn ml-btn-secondary ml-btn-sm"
              onClick={handlePreviewSynthOk} disabled={loadingSynthOk || okSynthTarget <= 0}
              title="Render synthetic OK crops to fill chars below the target count">
              {loadingSynthOk
                ? <span className="ml-loading-spinner" style={{ width: 12, height: 12, borderWidth: 2 }} />
                : <><svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                    <path d="M4 7V4H20V7M12 4V20M8 20H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg> Preview OK</>}
            </button>
            {syntheticOkCrops.length > 0 && (
              <span style={{ fontSize: 11, color: '#22c55e' }}>+{syntheticOkCrops.length} synthetic</span>
            )}
          </div>
          <div style={{ fontSize: 10, color: '#9ca3af', marginTop: 4 }}>
            Set <code>0</code> to disable. When &gt; 0, training also tops up char OK counts before training.
          </div>
        </div>

        {/* ── Augmentation ── */}
        <div className="ml-section-title">Augmentation (NG)</div>
        <div className="ml-form-group">
          <div style={{ fontSize: '11px', color: '#6b7280', marginBottom: '6px' }}>Generate synthetic NG from OK samples</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
            <div className="ml-augment-options">
              {AUGMENT_OPTIONS.map(opt => (
                <button key={opt.value}
                  className={`ml-augment-chip ${augmentFactor === opt.value ? 'selected' : ''}`}
                  onClick={() => setAugmentFactor(opt.value)}>{opt.label}</button>
              ))}
            </div>
            {augmentFactor >= 2 && (
              <button className="ml-btn ml-btn-secondary ml-btn-sm"
                style={{ marginLeft: 'auto' }}
                onClick={() => handlePreviewSynthetic()} disabled={loadingSynthetic}
                title="Preview synthetic NG crops generated from OK samples">
                {loadingSynthetic
                  ? <span className="ml-loading-spinner" style={{ width: 12, height: 12, borderWidth: 2 }} />
                  : <><svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" />
                    </svg> Preview NG</>
                }
              </button>
            )}
          </div>
          {/* {augmentFactor >= 2 && (() => {
            const { nOkReal, nNgReal, nNgAug, totalOk, totalNg } =
              computeAugmentStats(crops, augmentFactor);
            return (
              <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '4px' }}>
                → OK: <b>{totalOk}</b> (no aug) &nbsp;|&nbsp;
                NG: {nNgReal} real + {nNgAug} aug = <b>{totalNg}</b>
                {nOkReal === 0 && <span style={{ color: '#f59e0b' }}> · no OK samples with char_id</span>}
              </div>
            );
          })()} */}

          {/* ── NG severity distribution sliders (auto-normalized BE side) ── */}
          {augmentFactor >= 2 && (() => {
            const total = severitySubtle + severityLight + severityMedium + severityHeavy;
            const pct = (v: number) => total > 0 ? (v / total * 100).toFixed(0) : '0';
            const rows: { key: string; label: string; value: number; setter: (n: number) => void; color: string; hint: string }[] = [
              { key: 'subtle', label: 'Subtle',  value: severitySubtle, setter: setSeveritySubtle, color: '#9ca3af', hint: '1 light defect (mild break / tiny cut)' },
              { key: 'light',  label: 'Light',   value: severityLight,  setter: setSeverityLight,  color: '#3b82f6', hint: '1-2 mild defects' },
              { key: 'medium', label: 'Medium',  value: severityMedium, setter: setSeverityMedium, color: '#f59e0b', hint: '2-3 medium defects' },
              { key: 'heavy',  label: 'Heavy',   value: severityHeavy,  setter: setSeverityHeavy,  color: '#ef4444', hint: '2-3 strong defects (cuts, tape, blocks)' },
            ];
            return (
              <div style={{ marginTop: 8, padding: '6px 8px', borderRadius: 4, background: 'rgba(148,163,184,.06)', border: '1px solid rgba(148,163,184,.18)' }}>
                <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 4, display: 'flex', justifyContent: 'space-between' }}>
                  <span>NG severity mix</span>
                  <span style={{ opacity: 0.6 }}>{total === 0 ? '⚠ all zero — uses default' : `total ${total}`}</span>
                </div>
                {rows.map(r => (
                  <div key={r.key} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, marginTop: 2 }}>
                    <span style={{ color: r.color, width: 56 }} title={r.hint}>{r.label}</span>
                    <input type="range" min={0} max={100} step={1} value={r.value}
                           onChange={e => r.setter(Number(e.target.value))}
                           style={{ flex: 1, accentColor: r.color }} />
                    <span style={{ width: 36, textAlign: 'right', color: '#6b7280' }}>{pct(r.value)}%</span>
                  </div>
                ))}
              </div>
            );
          })()}
        </div>

        {/* ── OK Threshold ── */}
        <div className="ml-section-title">OK Threshold</div>
        <div className="ml-form-group">
          <div style={{ fontSize: '11px', color: '#6b7280', marginBottom: '4px' }}>
            prob_ok ≥ threshold → classified as OK
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              type="range"
              min={1} max={99} step={1}
              value={threshold}
              onChange={e => setThreshold(Number(e.target.value))}
              style={{ flex: 1, accentColor: '#3b82f6' }}
            />
            <input
              type="number"
              min={1} max={99} step={1}
              value={threshold}
              onChange={e => {
                const v = Math.min(99, Math.max(1, Number(e.target.value)));
                setThreshold(v);
              }}
              className="ml-form-input"
              style={{ width: '58px', textAlign: 'center', padding: '4px 6px' }}
            />
            <span style={{ fontSize: '12px', color: '#6b7280', flexShrink: 0 }}>%</span>
          </div>
          {threshold !== 50 && (
            <div style={{ fontSize: '11px', color: '#fbbf24' }}>
              {threshold < 50
                ? `↓ Looser — more crops will be OK`
                : `↑ Stricter — fewer crops will be OK`}
            </div>
          )}
        </div>

        {/* ── Train Preview — per-char breakdown before user clicks Train ── */}
        {(() => {
          const buckets = bucketByChar(crops);
          if (buckets.length === 0) return null;
          const named = buckets.filter(b => b.char_id !== null);
          const unnamedSamples = buckets
            .filter(b => b.char_id === null)
            .reduce((sum, b) => sum + b.items.length, 0);
          const { nOkReal, nNgReal, nNgAug, totalNg } =
            computeAugmentStats(crops, augmentFactor);
          const f = augmentFactor;
          const maxOkPerChar = named.reduce((m, b) => Math.max(m, b.ok), 0);
          const targetPerChar = f >= 2 ? f * maxOkPerChar : 0;
          return (
            <div className="ml-train-preview">
              <div className="ml-train-preview-title">🚀 Training preview</div>
              {named.length > 0 && (
                <div style={{ color: '#22c55e' }}>
                  ✓ {named.length} char{named.length !== 1 ? 's' : ''}: {named.map(b => b.char_id).join(', ')}
                </div>
              )}
              {unnamedSamples > 0 && (
                <div style={{ color: '#9ca3af' }}>
                  • {unnamedSamples} sample{unnamedSamples !== 1 ? 's' : ''} without char_id (still trained, no per-char stats)
                </div>
              )}
              <div style={{ marginTop: 2 }}>
                Dataset: {nOkReal} OK · {nNgReal}+{nNgAug}={totalNg} NG{f >= 2 ? ` (target ${targetPerChar}/char)` : ''}
                {f < 2 && <span style={{ color: '#9ca3af' }}> · no augmentation</span>}
              </div>
              {f >= 2 && nOkReal > 0 && (
                <div style={{ fontSize: 10, opacity: 0.55, marginTop: 2 }}>
                  Char-balanced NG — every char tops up to {targetPerChar} NG samples
                </div>
              )}
            </div>
          );
        })()}

        {/* ── Train button ── */}
        <button className="ml-btn ml-btn-primary"
          style={{ width: '100%', justifyContent: 'center', padding: '10px' }}
          onClick={handleTrain} disabled={!canTrain}>
          {training
            ? <><span className="ml-loading-spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Training...</>
            : <><svg width="14" height="14" viewBox="0 0 24 24" fill="none"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor" /></svg> Start Training</>
          }
        </button>

        {!canTrain && !training && (
          <div style={{ fontSize: '11px', color: '#6b7280', textAlign: 'center' }}>
            Need at least 2 labeled samples in Label tab
          </div>
        )}
        {training && (
          <div className="ml-training-progress-panel">
            <div className="ml-training-phase-row">
              <span className={`ml-training-phase-badge phase-${trainPhase || 'preparing'}`}>
                {phaseLabel(trainPhase)}
              </span>
              <span className="ml-training-progress-pct">{trainProgress.toFixed(0)}%</span>
            </div>
            <div className="ml-training-progress-track">
              <div className="ml-training-progress-fill" style={{ width: `${trainProgress}%` }} />
            </div>
            <div className="ml-training-eta">
              Đã chạy {fmtDuration(elapsedSec)}
              {etaSec != null && <> · còn ~{fmtDuration(etaSec)}</>}
            </div>
            <div className="ml-training-log-panel" ref={logScrollRef}>
              {trainLogs.length === 0 ? (
                <div className="ml-training-log-empty">Đang chờ log từ server…</div>
              ) : trainLogs.map(log => (
                <div key={log.idx} className={`ml-training-log-line level-${log.level.toLowerCase()}`}>
                  <span className="ml-training-log-ts">
                    {new Date(log.ts * 1000).toLocaleTimeString('en-GB', { hour12: false })}
                  </span>
                  <span className="ml-training-log-msg">{log.msg}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── History ── */}
        {models.length > 0 && (
          <>
            <div className="ml-section-title">History</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {models.slice(0, 5).map(m => (
                <div key={m.id} className="ml-history-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ opacity: 0.75 }}>{m.algorithm.toUpperCase()}</span>
                    <span style={{ color: m.status === 'completed' ? '#22c55e' : m.status === 'failed' ? '#ef4444' : '#f59e0b' }}>
                      {m.status}
                    </span>
                  </div>
                  {m.status === 'completed' && (
                    <div style={{ opacity: 0.6, marginTop: '2px' }}>
                      Test acc: {(m.metrics.accuracy_test * 100).toFixed(1)}%
                    </div>
                  )}
                  {m.status === 'failed' && m.error && (
                    <div style={{ color: '#ef4444', marginTop: '2px', wordBreak: 'break-word' }}>{m.error}</div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {/* ── Right: 2-column layout ──────────────────────────────────────── */}
      <div className="ml-train-right">

        {/* ── Column 1 (60%): Labeled Crops ───────────────────────────── */}
        <div className="ml-crops-panel">
          <div className="ml-crops-panel-header">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px 0' }}>
              <span style={{ fontSize: '12px', fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '.05em' }}>
                Labeled Crops
              </span>
              <button className="ml-btn ml-btn-secondary ml-btn-sm" onClick={loadCrops} disabled={loadingCrops}>
                {loadingCrops
                  ? <span className="ml-loading-spinner" style={{ width: 12, height: 12, borderWidth: 2 }} />
                  : <><svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                      <path d="M1 4v6h6M23 20v-6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg> Refresh</>
                }
              </button>
            </div>
            <div style={{ display: 'flex', padding: '8px 14px 0' }}>
              {([
                { key: 'real', label: `Real Data (${crops.length})` },
                { key: 'synthetic', label: `Synthetic (${syntheticCrops.length + syntheticOkCrops.length})` },
              ] as const).map(tab => (
                <button key={tab.key} onClick={() => setCropsTab(tab.key)} style={{
                  padding: '5px 12px', fontSize: '12px', border: 'none',
                  borderBottom: cropsTab === tab.key ? '2px solid #3b82f6' : '2px solid transparent',
                  background: 'transparent', color: cropsTab === tab.key ? '#60a5fa' : '#6b7280',
                  cursor: 'pointer', fontWeight: cropsTab === tab.key ? 600 : 400, transition: 'color .12s',
                }}>{tab.label}</button>
              ))}
            </div>
            {cropsTab === 'real' ? (
              <div style={{ display: 'flex', gap: '6px', padding: '8px 14px' }}>
                {(['all', 'OK', 'NG'] as const).map(f => (
                  <button key={f} onClick={() => setCropFilter(f)} style={{
                    padding: '3px 10px', fontSize: '11px', borderRadius: '12px', border: '1px solid', cursor: 'pointer', transition: 'all .12s',
                    borderColor: cropFilter === f ? (f === 'OK' ? '#4ade80' : f === 'NG' ? '#f87171' : '#3b82f6') : '#2d3148',
                    background: cropFilter === f ? (f === 'OK' ? 'rgba(74,222,128,.12)' : f === 'NG' ? 'rgba(248,113,113,.12)' : 'rgba(59,130,246,.12)') : 'transparent',
                    color: cropFilter === f ? (f === 'OK' ? '#4ade80' : f === 'NG' ? '#f87171' : '#60a5fa') : '#6b7280',
                  }}>
                    {f === 'all' ? `All (${crops.length})` : f === 'OK' ? `OK (${okCrops.length})` : `NG (${ngCrops.length})`}
                  </button>
                ))}
              </div>
            ) : <div style={{ height: '8px' }} />}
            {/* Stats banner showing char coverage + view mode toggle */}
            {(() => {
              const synthAll: Array<{ crop_b64: string; label: string; char_id?: string | null }> = [
                ...syntheticCrops,
                ...syntheticOkCrops.map(c => ({
                  crop_b64: c.crop_b64, label: 'OK', char_id: c.char_id,
                })),
              ];
              const src: Array<{ crop_b64: string; label: string; char_id?: string | null }> =
                cropsTab === 'real' ? crops : synthAll;
              const sourceBuckets = bucketByChar(src);
              return (
                <CharStatsBanner
                  buckets={sourceBuckets}
                  viewMode={viewMode}
                  onToggleView={setViewMode}
                />
              );
            })()}
          </div>
          <div className="ml-crops-panel-body">
            {cropsTab === 'real' && (
              loadingCrops
                ? <div className="ml-empty-state"><div className="ml-loading-spinner" /></div>
                : viewMode === 'grouped'
                    ? <CharGroupedCrops
                        buckets={bucketByChar(filteredCrops)}
                        expanded={expandedChars}
                        onToggleExpand={toggleCharExpanded}
                        emptyText={crops.length === 0 ? 'No labeled characters yet. Go to Label tab.' : `No ${cropFilter} crops found.`}
                      />
                    : <CropGrid items={filteredCrops}
                        emptyText={crops.length === 0 ? 'No labeled characters yet. Go to Label tab.' : `No ${cropFilter} crops found.`} />
            )}
            {cropsTab === 'synthetic' && (() => {
              const merged: Array<{ crop_b64: string; label: string; char_id?: string | null }> = [
                ...syntheticCrops,
                ...syntheticOkCrops.map(c => ({
                  crop_b64: c.crop_b64, label: 'OK', char_id: c.char_id,
                })),
              ];
              if (loadingSynthetic || loadingSynthOk) {
                return <div className="ml-empty-state"><div className="ml-loading-spinner" /></div>;
              }
              if (merged.length === 0) {
                return (
                  <div className="ml-empty-state" style={{ minHeight: '120px' }}>
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.4 }}>
                      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                    </svg>
                    <span style={{ fontSize: '12px', color: '#6b7280', marginTop: '6px' }}>
                      Click <b>Preview NG</b> or <b>Preview OK</b> in the config panel.
                    </span>
                  </div>
                );
              }
              return viewMode === 'grouped'
                ? <CharGroupedCrops
                    buckets={bucketByChar(merged)}
                    expanded={expandedChars}
                    onToggleExpand={toggleCharExpanded}
                    emptyText="No synthetic crops."
                  />
                : <CropGrid items={merged} emptyText="No synthetic crops." />;
            })()}
          </div>
        </div>

        {/* ── Column 2 (40%): Results ──────────────────────────────────── */}
        <div className="ml-results-column">
          {completedModels.length === 0 ? (
            <div className="ml-empty-state" style={{ height: '100%' }}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.3 }}>
                <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <span style={{ fontSize: '12px', color: '#6b7280', marginTop: '8px' }}>
                Training results will appear here after training completes.
              </span>
            </div>
          ) : (
            <div className="ml-results-panel">
              {/* ── Model selectbox ── */}
              <div>
                <label className="ml-form-label" style={{ marginBottom: '6px', display: 'block' }}>Model</label>
                <select
                  className="ml-form-select"
                  value={selectedModelId ?? ''}
                  onChange={e => {
                    setSelectedModelId(e.target.value);
                    setPredictResults(null);
                    setTestSetCrops([]);
                  }}
                >
                  {completedModels.map(m => (
                    <option key={m.id} value={m.id}>{modelLabel(m)}</option>
                  ))}
                </select>
              </div>

              {/* ── Tabs: Metrics | Test Set ── */}
              {selectedModel && (
                <>
                  <div style={{ display: 'flex', borderBottom: '1px solid #2d3148', marginBottom: '4px' }}>
                    {([
                      { key: 'metrics', label: 'Metrics' },
                      { key: 'testset', label: 'Test Set' },
                    ] as const).map(t => (
                      <button key={t.key} onClick={() => setResultsTab(t.key)} style={{
                        padding: '6px 14px', fontSize: '12px', border: 'none',
                        borderBottom: resultsTab === t.key ? '2px solid #3b82f6' : '2px solid transparent',
                        background: 'transparent', color: resultsTab === t.key ? '#60a5fa' : '#6b7280',
                        cursor: 'pointer', fontWeight: resultsTab === t.key ? 600 : 400, transition: 'color .12s',
                        marginBottom: '-1px',
                      }}>{t.label}</button>
                    ))}
                  </div>

                  {/* ── Metrics tab ── */}
                  {resultsTab === 'metrics' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      {/* Top bar: threshold badge + download report */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        {selectedModel.params?.threshold != null && (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', opacity: 0.75 }}>
                            <svg width="11" height="11" viewBox="0 0 24 24" fill="none">
                              <line x1="4" y1="12" x2="20" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                              <circle cx="12" cy="12" r="3" fill="currentColor" />
                            </svg>
                            OK threshold:
                            <span style={{ color: '#3b82f6', fontWeight: 600 }}>
                              {Math.round((selectedModel.params.threshold as number) * 100)}%
                            </span>
                          </div>
                        )}
                        <button
                          className="ml-btn ml-btn-secondary ml-btn-sm"
                          onClick={handleDownloadReport}
                          disabled={downloadingReport}
                          title="Download JSON report with metrics + test-set crops (base64)"
                          style={{ marginLeft: 'auto' }}
                        >
                          {downloadingReport
                            ? <><span className="ml-loading-spinner" style={{ width: 11, height: 11, borderWidth: 2 }} /> Building…</>
                            : <>
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"
                                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                                Download Report
                              </>
                          }
                        </button>
                      </div>
                      <div className="ml-metric-row">
                        <div className="ml-metric-card">
                          <div className="ml-metric-value">{(selectedModel.metrics.accuracy_train * 100).toFixed(1)}%</div>
                          <div className="ml-metric-label">Train accuracy</div>
                        </div>
                        <div className="ml-metric-card">
                          <div className="ml-metric-value" style={{ color: '#4ade80' }}>{(selectedModel.metrics.accuracy_test * 100).toFixed(1)}%</div>
                          <div className="ml-metric-label">Test accuracy</div>
                        </div>
                        <div className="ml-metric-card">
                          <div className="ml-metric-value" style={{ color: '#9ca3af', fontSize: '16px' }}>
                            {selectedModel.metrics.n_ok} / {selectedModel.metrics.n_ng}
                          </div>
                          <div className="ml-metric-label">OK / NG samples</div>
                        </div>
                      </div>

                      {selectedModel.metrics.confusion_matrix.length > 0 && (
                        <div>
                          <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '6px', fontWeight: 600 }}>Confusion Matrix</div>
                          <div className="ml-confusion-matrix">
                            <table>
                              <thead><tr><th></th><th>Pred NG</th><th>Pred OK</th></tr></thead>
                              <tbody>
                                <tr>
                                  <th>Actual NG</th>
                                  {selectedModel.metrics.confusion_matrix[0]?.map((v, i) => (
                                    <td key={i} style={{ color: i === 0 ? '#4ade80' : '#f87171' }}>{v}</td>
                                  ))}
                                </tr>
                                <tr>
                                  <th>Actual OK</th>
                                  {selectedModel.metrics.confusion_matrix[1]?.map((v, i) => (
                                    <td key={i} style={{ color: i === 1 ? '#4ade80' : '#f87171' }}>{v}</td>
                                  ))}
                                </tr>
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}

                      {selectedModel.metrics.report && (
                        <div>
                          <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '6px', fontWeight: 600 }}>Classification Report</div>
                          <pre className="ml-report-pre">{selectedModel.metrics.report}</pre>
                        </div>
                      )}

                      {/* Per-char accuracy table (requires test-set data loaded) */}
                      {perCharAccuracy.length > 0 && (
                        <div>
                          <div style={{
                            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                            fontSize: '12px', color: '#6b7280', marginBottom: '6px', fontWeight: 600,
                          }}>
                            <span>Per-Char Accuracy</span>
                            <button
                              className="ml-btn ml-btn-secondary ml-btn-sm"
                              onClick={() => loadTestSetCrops(selectedModel.id)}
                              disabled={loadingTestSet}
                              style={{ padding: '2px 8px', fontSize: 10 }}
                            >
                              {loadingTestSet ? '…' : 'Refresh'}
                            </button>
                          </div>
                          <div style={{ overflowX: 'auto' }}>
                            <table className="ml-perchar-table">
                              <thead>
                                <tr>
                                  <th style={{ textAlign: 'left' }}>Char</th>
                                  <th>n</th>
                                  <th style={{ color: '#22c55e' }}>✓</th>
                                  <th style={{ color: '#ef4444' }}>✗</th>
                                  <th>Acc</th>
                                </tr>
                              </thead>
                              <tbody>
                                {perCharAccuracy.map(r => (
                                  <tr key={r.char_id ?? '__null__'}>
                                    <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>
                                      {r.char_id ?? <span style={{ color: '#ef4444' }}>—</span>}
                                    </td>
                                    <td>{r.n}</td>
                                    <td style={{ color: '#22c55e' }}>{r.correct}</td>
                                    <td style={{ color: r.wrong > 0 ? '#ef4444' : 'inherit', opacity: r.wrong > 0 ? 1 : 0.5 }}>
                                      {r.wrong}
                                    </td>
                                    <td style={{
                                      color: r.acc >= 0.9 ? '#22c55e' : r.acc >= 0.7 ? '#f59e0b' : '#ef4444',
                                      fontWeight: 600,
                                    }}>{(r.acc * 100).toFixed(0)}%</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          {testSetCrops.length === 0 && !loadingTestSet && (
                            <div style={{ fontSize: 11, color: '#6b7280', marginTop: 4 }}>
                              Load test-set tab to populate this table.
                            </div>
                          )}
                        </div>
                      )}

                      {/* Single-image predict */}
                      <div className="ml-section-title" style={{ marginTop: '4px' }}>Test Prediction</div>
                      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                        <button className="ml-btn ml-btn-secondary ml-btn-sm"
                          onClick={() => predictInputRef.current?.click()}>
                          <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                            <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                          </svg> Choose Image
                        </button>
                        {predictFile && <span style={{ fontSize: '11px', color: '#9ca3af' }}>{predictFile.name}</span>}
                        {predictFile && (
                          <button className="ml-btn ml-btn-primary ml-btn-sm" onClick={handlePredict} disabled={predicting}>
                            {predicting
                              ? <><span className="ml-loading-spinner" style={{ width: 12, height: 12, borderWidth: 2 }} /> Predicting...</>
                              : <><svg width="12" height="12" viewBox="0 0 24 24" fill="none"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor" /></svg> Run</>
                            }
                          </button>
                        )}
                        <input ref={predictInputRef} type="file" accept="image/*" style={{ display: 'none' }}
                          onChange={e => { setPredictFile(e.target.files?.[0] ?? null); setPredictResults(null); }} />
                      </div>

                      {predictResults && (
                        <div>
                          <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '6px' }}>
                            {predictResults.length} characters detected
                          </div>
                          <div className="ml-predict-results-grid">
                            {predictResults.map((r, i) => (
                              <PredictResultCard key={i} result={r} />
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* ── Test Set tab ── */}
                  {resultsTab === 'testset' && (() => {
                    const correct = testSetCrops.filter(c => c.correct);
                    const wrong   = testSetCrops.filter(c => !c.correct);
                    const shown   = testSetFilter === 'all' ? testSetCrops
                                  : testSetFilter === 'correct' ? correct : wrong;
                    return (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        {loadingTestSet ? (
                          <div className="ml-empty-state" style={{ minHeight: '100px' }}>
                            <div className="ml-loading-spinner" />
                          </div>
                        ) : testSetCrops.length === 0 ? (
                          <div style={{ fontSize: '12px', color: '#6b7280', textAlign: 'center', padding: '32px 0' }}>
                            Train a model to generate test-set crops.
                          </div>
                        ) : (
                          <>
                            {/* Summary stats */}
                            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                              <span style={{ fontSize: '12px', color: '#9ca3af' }}>
                                {testSetCrops.length} crops
                              </span>
                              <span style={{ fontSize: '12px', color: '#4ade80' }}>
                                ✓ {correct.length} correct
                              </span>
                              {wrong.length > 0 && (
                                <span style={{ fontSize: '12px', color: '#f87171' }}>
                                  ✗ {wrong.length} wrong
                                </span>
                              )}
                              <span style={{ fontSize: '11px', color: '#6b7280', marginLeft: 'auto' }}>
                                {((correct.length / testSetCrops.length) * 100).toFixed(1)}%
                              </span>
                              <button className="ml-btn ml-btn-secondary ml-btn-sm"
                                onClick={() => loadTestSetCrops(selectedModel!.id)} disabled={loadingTestSet}>
                                <svg width="11" height="11" viewBox="0 0 24 24" fill="none">
                                  <path d="M1 4v6h6M23 20v-6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                  <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                              </button>
                            </div>

                            {/* Filter chips */}
                            <div style={{ display: 'flex', gap: '6px' }}>
                              {([
                                { key: 'all',     label: `All (${testSetCrops.length})` },
                                { key: 'correct', label: `Correct (${correct.length})` },
                                { key: 'wrong',   label: `Wrong (${wrong.length})` },
                              ] as const).map(f => (
                                <button key={f.key} onClick={() => setTestSetFilter(f.key)} style={{
                                  padding: '3px 10px', fontSize: '11px', borderRadius: '12px',
                                  border: '1px solid', cursor: 'pointer', transition: 'all .12s',
                                  borderColor: testSetFilter === f.key
                                    ? (f.key === 'wrong' ? '#f87171' : f.key === 'correct' ? '#4ade80' : '#3b82f6')
                                    : '#2d3148',
                                  background: testSetFilter === f.key
                                    ? (f.key === 'wrong' ? 'rgba(248,113,113,.12)' : f.key === 'correct' ? 'rgba(74,222,128,.12)' : 'rgba(59,130,246,.12)')
                                    : 'transparent',
                                  color: testSetFilter === f.key
                                    ? (f.key === 'wrong' ? '#f87171' : f.key === 'correct' ? '#4ade80' : '#60a5fa')
                                    : '#6b7280',
                                }}>{f.label}</button>
                              ))}
                            </div>

                            {/* Grouped by char — sorted wrong-count desc, auto-expand chars with errors */}
                            {(() => {
                              // Build per-char groups for SHOWN items (after filter)
                              const groups = new Map<string, { char_id: string | null; items: TestSetCropResult[]; wrong: number; correct: number }>();
                              for (const it of shown) {
                                const key = it.char_id ?? '__null__';
                                if (!groups.has(key)) {
                                  groups.set(key, { char_id: it.char_id ?? null, items: [], wrong: 0, correct: 0 });
                                }
                                const g = groups.get(key)!;
                                g.items.push(it);
                                if (it.correct) g.correct += 1; else g.wrong += 1;
                              }
                              const sorted = Array.from(groups.values()).sort((a, b) => {
                                if (a.wrong !== b.wrong) return b.wrong - a.wrong;
                                return (a.char_id ?? '~').localeCompare(b.char_id ?? '~');
                              });

                              return (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                  {sorted.map(g => {
                                    const key = g.char_id ?? '__null__';
                                    const acc = g.items.length ? g.correct / g.items.length : 0;
                                    // Auto-expand if has wrongs OR user manually toggled
                                    const autoExpand = g.wrong > 0;
                                    const manuallyToggled = testsetExpanded.has(key);
                                    const isOpen = autoExpand ? !manuallyToggled : manuallyToggled;
                                    const accColor = acc >= 0.9 ? '#22c55e' : acc >= 0.7 ? '#f59e0b' : '#ef4444';
                                    return (
                                      <div key={key} className="ml-testset-group">
                                        <button
                                          className="ml-testset-group-header"
                                          onClick={() => setTestsetExpanded(prev => {
                                            const next = new Set(prev);
                                            if (next.has(key)) next.delete(key); else next.add(key);
                                            return next;
                                          })}
                                        >
                                          <span style={{ width: 12, textAlign: 'center' }}>{isOpen ? '▼' : '▶'}</span>
                                          <span className="ml-char-badge" style={{ color: accColor }}>
                                            {g.char_id ?? '—'}
                                          </span>
                                          <span style={{ color: accColor, fontWeight: 600 }}>
                                            {(acc * 100).toFixed(0)}% acc
                                          </span>
                                          <span>{g.items.length} test · ✓ {g.correct} · ✗ {g.wrong}</span>
                                        </button>
                                        {isOpen && (
                                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, padding: 6 }}>
                                            {/* Sort wrongs first within group */}
                                            {[...g.items].sort((a, b) => Number(a.correct) - Number(b.correct)).map((item, i) => (
                                              <TestSetCropCard key={i} item={item} />
                                            ))}
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })}
                                  {shown.length === 0 && (
                                    <div style={{ fontSize: '12px', color: '#6b7280', padding: '16px 0' }}>
                                      No crops match this filter.
                                    </div>
                                  )}
                                </div>
                              );
                            })()}
                          </>
                        )}
                      </div>
                    );
                  })()}

                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
