import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  anomalyTrainingAPI, AnomalyModel, DatasetImage, DefectStroke,
  InferenceEngine, SimulateDefectResult,
} from '@/services/anomalyTraining';
import '@/styles/AnomalyTraining.css';

interface Props {
  projectId: string;
  model: AnomalyModel | null;
}

type EdgeKind = 'hard' | 'soft' | 'wrinkle' | 'bubble';

const EDGE_HELP: Record<EdgeKind, string> = {
  hard: 'Sharp-edged line — a scratch or print artefact. Easiest to see, least like a crease.',
  soft: 'Blurred line — a smudge or soft mark. No step gradient for the model to latch onto.',
  wrinkle: 'Highlight + shadow pair — how a real crease catches light. Ignores the dark/bright setting.',
  bubble: 'Blister trapped under the label — a shallow dome, bright on the lit slope and dark on the far one. '
    + 'Drag to set its span; thickness softens the rim and Light angle picks which side is lit.',
};

/** Bubble geometry comes from the drag's two ends (they span the major axis)
 *  and it lights both sides itself, so the dark/bright choice has no meaning —
 *  same as wrinkle. */
const IGNORES_POLARITY: EdgeKind[] = ['wrinkle', 'bubble'];

/**
 * Defect Studio — draw a synthetic wrinkle onto a clean label, run the model,
 * and see what it saw: score, heatmap, and the predicted mask as polygons.
 *
 * Answers "how bad must a defect be before this model catches it?" without
 * waiting for a real defective bottle. Drawing happens server-side so the
 * marks match the offline detection-limit study exactly — the canvas only
 * captures geometry.
 */
export default function StudioTab({ projectId, model }: Props) {
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [baseId, setBaseId] = useState<string>('');
  const [baseB64, setBaseB64] = useState<string>('');
  const [loadingBase, setLoadingBase] = useState(false);

  // Stroke parameters
  const [width, setWidth] = useState(6);
  const [delta, setDelta] = useState(30);
  const [polarity, setPolarity] = useState<'dark' | 'bright'>('dark');
  const [edge, setEdge] = useState<EdgeKind>('wrinkle');
  const [curvature, setCurvature] = useState(0.15);

  // Geometry only. Parameters are NOT baked in here — they are applied at send
  // time from the toolbar, so changing Δ/thickness/type re-renders the same
  // marks instead of silently keeping whatever was set when they were drawn.
  // Sweeping one parameter over one mark is the whole point of this tab.
  const [paths, setPaths] = useState<number[][][]>([]);
  const [drawing, setDrawing] = useState<number[][] | null>(null);
  // Signature of what produced `result`, so the panel can say when the shown
  // result no longer matches the controls instead of looking current.
  const [predictedSig, setPredictedSig] = useState<string>('');

  // Default to TensorRT whenever an engine exists. Measured on this project's
  // exports, an onnxruntime CUDA-EP session costs ~2.2 GB of GPU memory versus
  // ~0.6 GB for the equivalent TensorRT engine, and Studio re-predicts on every
  // parameter change — so defaulting to ONNX quietly pinned the heavier backend
  // for a whole drawing session, on top of whatever the Test tab already held.
  const trtReady = !!model?.engine_path;
  const [engineChoice, setEngineChoice] = useState<InferenceEngine>(trtReady ? 'tensorrt' : 'onnx');
  const [threshold, setThreshold] = useState(0.5);
  const [pixelThreshold, setPixelThreshold] = useState(0.4);
  const [usePixelThreshold, setUsePixelThreshold] = useState(true);

  const [result, setResult] = useState<SimulateDefectResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [view, setView] = useState<'defect' | 'heatmap'>('heatmap');
  const [showMask, setShowMask] = useState(true);
  const [showMarks, setShowMarks] = useState(true);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  const imgRef = useRef<HTMLImageElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [natural, setNatural] = useState({ w: 0, h: 0 });
  const [rendered, setRendered] = useState({ w: 0, h: 0 });

  useEffect(() => {
    setResult(null);
    setPaths([]);
    setPredictedSig('');
    setThreshold(model?.metrics?.threshold ?? 0.5);
    // Re-apply the backend default here too: the useState initializer only runs
    // on mount, and this tab commonly mounts before the model has loaded (or
    // switches to a model with a different export state).
    setEngineChoice(model?.engine_path ? 'tensorrt' : 'onnx');
  }, [model?.id, model?.engine_path]);

  // Clean base images only: drawing a synthetic defect on an already-defective
  // label would confound the answer.
  useEffect(() => {
    if (!projectId) return;
    (async () => {
      try {
        const page = await anomalyTrainingAPI.listDatasetImages(projectId, {
          label: 'normal', page: 1, pageSize: 24,
        });
        setImages(page.images);
      } catch (e: any) {
        setErrorMsg(e?.response?.data?.detail || 'Could not load dataset images');
      }
    })();
  }, [projectId]);

  const pickBase = useCallback(async (id: string) => {
    setLoadingBase(true);
    setErrorMsg(null);
    setResult(null);
    setPaths([]);
    setPredictedSig('');
    try {
      const full = await anomalyTrainingAPI.getDatasetImageFull(projectId, id);
      setBaseId(id);
      setBaseB64(full.full_b64);
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || 'Could not load that image');
    } finally {
      setLoadingBase(false);
    }
  }, [projectId]);

  const measure = useCallback(() => {
    const el = imgRef.current;
    if (!el) return;
    setNatural({ w: el.naturalWidth, h: el.naturalHeight });
    setRendered({ w: el.clientWidth, h: el.clientHeight });
  }, []);

  useEffect(() => {
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [measure]);

  // Display px -> source image px. Everything sent to the backend is in source
  // pixels so `width` and `delta` mean the same thing regardless of zoom.
  const toSource = useCallback((clientX: number, clientY: number): number[] | null => {
    const el = imgRef.current;
    if (!el || !natural.w) return null;
    const r = el.getBoundingClientRect();
    const x = ((clientX - r.left) / r.width) * natural.w;
    const y = ((clientY - r.top) / r.height) * natural.h;
    if (x < 0 || y < 0 || x > natural.w || y > natural.h) return null;
    return [Math.round(x), Math.round(y)];
  }, [natural]);

  const onPointerDown = (e: React.PointerEvent) => {
    if (!baseB64 || busy) return;
    const p = toSource(e.clientX, e.clientY);
    if (!p) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    setDrawing([p]);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!drawing) return;
    const p = toSource(e.clientX, e.clientY);
    if (!p) return;
    // Thin the freehand path: a point every few px is plenty and keeps the
    // payload (and the server-side polyline) small.
    const last = drawing[drawing.length - 1];
    if (last && Math.hypot((p[0] ?? 0) - (last[0] ?? 0), (p[1] ?? 0) - (last[1] ?? 0)) < 6) return;
    setDrawing([...drawing, p]);
  };

  const onPointerUp = () => {
    if (!drawing) return;
    if (drawing.length >= 2) setPaths((p) => [...p, drawing]);
    setDrawing(null);
  };

  // Current toolbar settings applied to every drawn path.
  const strokes: DefectStroke[] = useMemo(() => paths.map((points) => ({
    points, width, delta, polarity, edge,
    // Curvature only bows a straight 2-point stroke — a freehand path already
    // has its own shape. For a bubble the same field carries the light angle,
    // which applies however the span was drawn.
    curvature: edge === 'bubble' || points.length === 2 ? curvature : 0,
  })), [paths, width, delta, polarity, edge, curvature]);

  // Everything that changes what the model will be shown or how it is judged.
  const sig = useMemo(() => JSON.stringify([
    baseId, strokes, engineChoice, threshold, usePixelThreshold ? pixelThreshold : null,
  ]), [baseId, strokes, engineChoice, threshold, usePixelThreshold, pixelThreshold]);
  const stale = !!result && sig !== predictedSig;

  const runSimulate = useCallback(async () => {
    if (!model || !baseId) return;
    setBusy(true);
    setErrorMsg(null);
    try {
      const res = await anomalyTrainingAPI.simulateDefect(projectId, model.id, {
        strokes,
        engine: engineChoice,
        item_id: baseId,
        threshold,
        pixel_threshold: usePixelThreshold ? pixelThreshold : null,
      });
      setResult(res);
      setPredictedSig(sig);
      setView('heatmap');
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || e?.message || 'Simulation failed');
      setResult(null);
    } finally {
      setBusy(false);
    }
  }, [projectId, model, baseId, strokes, engineChoice, threshold, usePixelThreshold, pixelThreshold, sig]);

  // Keep a mark that sits right at the model's limit — those are the samples
  // worth adding to the dataset, and you find them by hand here rather than by
  // sweeping a grid.
  const saveAsSample = useCallback(async () => {
    if (!model || !baseId || !paths.length) return;
    setBusy(true); setSaveMsg(null); setErrorMsg(null);
    try {
      const res = await anomalyTrainingAPI.simulateDefect(projectId, model.id, {
        strokes, engine: engineChoice, item_id: baseId, threshold,
        pixel_threshold: usePixelThreshold ? pixelThreshold : null,
        save_to_dataset: true,
      });
      setResult(res);
      setPredictedSig(sig);
      setSaveMsg('Added to the dataset as an abnormal test sample. Retrain to recalibrate the threshold.');
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || 'Could not save to dataset');
    } finally { setBusy(false); }
  }, [projectId, model, baseId, paths.length, strokes, engineChoice, threshold, usePixelThreshold, pixelThreshold, sig]);

  // Preview of not-yet-predicted strokes, drawn in display coords.
  const previewScale = natural.w ? rendered.w / natural.w : 1;
  const pendingPaths = useMemo(() => {
    const all = drawing ? [...paths, drawing] : paths;
    return all.map((pts) => pts.map((pt) => `${(pt[0] ?? 0) * previewScale},${(pt[1] ?? 0) * previewScale}`).join(' '));
  }, [paths, drawing, previewScale]);

  const maskScale = result && result.mask_width
    ? { x: rendered.w / result.mask_width, y: rendered.h / result.mask_height }
    : { x: 1, y: 1 };

  const shownImage = result
    ? (view === 'heatmap' && result.heatmap_b64 ? result.heatmap_b64 : result.defect_b64)
    : baseB64;

  if (!model || model.status !== 'completed') {
    return <div className="at-empty">Select a trained model to use the Studio.</div>;
  }

  return (
    <div className="at-studio">
      <div className="at-studio-main">
        <div className="at-studio-toolbar">
          <div className="at-studio-group">
            <label>Defect type</label>
            <select value={edge} onChange={(e) => setEdge(e.target.value as EdgeKind)}>
              <option value="wrinkle">Wrinkle (highlight + shadow)</option>
              <option value="bubble">Bubble (blister under the label)</option>
              <option value="soft">Soft mark (blurred)</option>
              <option value="hard">Scratch (sharp)</option>
            </select>
          </div>
          <div className="at-studio-group">
            <label>Contrast Δ: {delta}</label>
            <input type="range" min={5} max={90} step={1} value={delta}
                   onChange={(e) => setDelta(Number(e.target.value))} />
          </div>
          <div className="at-studio-group">
            <label>Thickness: {width}px</label>
            <input type="range" min={1} max={30} step={1} value={width}
                   onChange={(e) => setWidth(Number(e.target.value))} />
          </div>
          <div className="at-studio-group">
            <label>
              {edge === 'bubble' ? 'Light angle' : 'Curve'}: {curvature.toFixed(2)}
            </label>
            <input type="range"
                   min={edge === 'bubble' ? -1 : -0.5} max={edge === 'bubble' ? 1 : 0.5}
                   step={0.05} value={curvature}
                   onChange={(e) => setCurvature(Number(e.target.value))} />
          </div>
          <div className="at-studio-group">
            <label>Direction</label>
            <select value={polarity} disabled={IGNORES_POLARITY.includes(edge)}
                    onChange={(e) => setPolarity(e.target.value as 'dark' | 'bright')}>
              <option value="dark">Darker than background</option>
              <option value="bright">Brighter than background</option>
            </select>
          </div>
        </div>
        <div className="at-studio-hint">{EDGE_HELP[edge]}</div>

        <div className="at-studio-canvas-wrap" ref={wrapRef}>
          {!baseB64 ? (
            <div className="at-empty">
              {loadingBase ? 'Loading image…' : 'Pick a clean label below to start drawing.'}
            </div>
          ) : (
            <div className="at-studio-canvas"
                 onPointerDown={onPointerDown}
                 onPointerMove={onPointerMove}
                 onPointerUp={onPointerUp}
                 onPointerLeave={onPointerUp}>
              <img ref={imgRef} src={`data:image/jpeg;base64,${shownImage}`}
                   alt="label" onLoad={measure} draggable={false} />
              <svg className="at-studio-overlay" width={rendered.w} height={rendered.h}>
                {/* Drawn marks — hidden once a result is shown so the model's
                    own mask isn't visually confused with our input. */}
                {(!result || showMarks) && pendingPaths.map((pts, i) => (
                  <polyline key={`s${i}`} points={pts} fill="none"
                            stroke={polarity === 'bright' && !IGNORES_POLARITY.includes(edge) ? '#fde047' : '#38bdf8'}
                            strokeWidth={Math.max(2, width * previewScale)}
                            strokeDasharray={result ? '6 5' : undefined}
                            strokeLinecap="round" strokeLinejoin="round"
                            opacity={result ? 0.9 : 0.75} />
                ))}
                {result && showMask && result.mask_polygons.map((p, i) => (
                  <polygon key={`m${i}`}
                           points={p.points.map((pt) => `${(pt[0] ?? 0) * maskScale.x},${(pt[1] ?? 0) * maskScale.y}`).join(' ')}
                           fill="rgba(239,68,68,0.28)" stroke="#ef4444" strokeWidth={2} />
                ))}
              </svg>
            </div>
          )}
        </div>

        <div className="at-studio-actions">
          <button className="btn btn-secondary" disabled={!paths.length || busy}
                  onClick={() => setPaths((p) => p.slice(0, -1))}>
            Undo
          </button>
          <button className="btn btn-secondary" disabled={(!paths.length && !result) || busy}
                  onClick={() => { setPaths([]); setResult(null); setPredictedSig(''); }}>
            Clear
          </button>
          <span className="at-studio-count">{paths.length} mark(s) drawn</span>
          <select value={engineChoice} onChange={(e) => setEngineChoice(e.target.value as InferenceEngine)}>
            <option value="onnx">ONNX</option>
            <option value="tensorrt" disabled={!trtReady}>
              {trtReady ? 'TensorRT' : 'TensorRT (not exported)'}
            </option>
          </select>
          <button className="btn btn-primary" disabled={!baseId || busy} onClick={runSimulate}>
            {busy ? 'Running…' : stale ? 'Re-predict' : paths.length ? 'Draw + Predict' : 'Predict clean (baseline)'}
          </button>
          <button className="btn btn-secondary" disabled={!paths.length || busy}
                  title="Save this drawn image into the dataset as an abnormal test sample"
                  onClick={saveAsSample}>
            Save as NG sample
          </button>
        </div>

        {stale && (
          <div className="at-warn">
            Settings changed since this result — the marks above are redrawn with the
            new values but the score and mask below are still from the previous run.
            Press Re-predict.
          </div>
        )}
        {saveMsg && <div className="at-alert-ok">{saveMsg}</div>}
        {errorMsg && <div className="at-error">{errorMsg}</div>}

        <div className="at-studio-bases">
          {images.map((im) => (
            <button key={im.id}
                    className={`at-studio-base ${baseId === im.id ? 'active' : ''}`}
                    title={`${im.camera_serial} · frame ${im.frame_idx}`}
                    onClick={() => pickBase(im.id)}>
              <img src={`data:image/jpeg;base64,${im.thumb_b64}`} alt="" />
            </button>
          ))}
          {!images.length && <span className="at-hint">No normal images in this project yet.</span>}
        </div>
      </div>

      <div className="at-studio-side">
        <h4>Result</h4>
        {!result ? (
          <div className="at-hint">
            Draw one or more marks on the label, then Predict. Run it with no marks
            first to confirm the clean image scores near zero — that is the baseline
            everything else is judged against.
          </div>
        ) : (
          <>
            <div className={`at-studio-verdict ${result.caught_by_mask || result.caught_by_score ? 'bad' : 'good'}`}>
              {result.caught_by_mask
                ? 'CAUGHT — mask landed on the mark'
                : result.caught_by_score
                  ? 'CAUGHT by score only — mask missed the mark'
                  : 'MISSED — below the detection limit'}
            </div>

            <div className="at-studio-metrics">
              <div><span>Anomaly score</span><b>{result.pred_score.toFixed(4)}</b></div>
              <div><span>Threshold</span><b>{result.threshold.toFixed(2)}</b></div>
              <div><span>Mask ∩ mark</span><b>{(result.stroke_overlap * 100).toFixed(0)}%</b></div>
              <div><span>Mask regions</span><b>{result.region_count}</b></div>
              <div><span>Total area</span><b>{result.total_area} px² ({result.total_area_pct}%)</b></div>
              <div><span>Heatmap max</span><b>{result.anomaly_map_max?.toFixed(3) ?? '—'}</b></div>
              <div><span>Inference</span><b>{result.inference_ms.toFixed(1)} ms</b></div>
              <div><span>Backend</span><b>{result.active_provider}</b></div>
            </div>

            {result.caught_by_score && !result.caught_by_mask && (
              <div className="at-warn">
                The score crossed the threshold but the mask did not overlap the drawn
                mark — the model reacted to something else in the frame. Treat this as
                a miss for localization purposes.
              </div>
            )}

            <div className="at-studio-view">
              <button className={view === 'heatmap' ? 'active' : ''} onClick={() => setView('heatmap')}>Heatmap</button>
              <button className={view === 'defect' ? 'active' : ''} onClick={() => setView('defect')}>Drawn image</button>
              <label>
                <input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} />
                Mask
              </label>
              <label>
                <input type="checkbox" checked={showMarks} onChange={(e) => setShowMarks(e.target.checked)} />
                My marks
              </label>
            </div>
            <div className="at-hint">Mask from: {result.mask_source}</div>

            {result.mask_polygons.length > 0 && (
              <>
                <h4>Regions</h4>
                <table className="at-studio-regions">
                  <thead>
                    <tr>
                      <th>#</th><th>Area</th><th>Conf</th><th>On mark</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.mask_polygons.map((p, i) => (
                      <tr key={i} className={(p.stroke_overlap ?? 0) < 0.1 ? 'off-mark' : ''}>
                        <td>{i + 1}</td>
                        <td>
                          {p.area}
                          <span className="at-studio-sub"> px² · {p.area_pct}%</span>
                        </td>
                        <td>
                          {p.score_mean?.toFixed(3) ?? '—'}
                          <span className="at-studio-sub"> max {p.score_max?.toFixed(3) ?? '—'}</span>
                        </td>
                        <td>{p.stroke_overlap != null ? `${Math.round(p.stroke_overlap * 100)}%` : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="at-hint">
                  Regions are sorted largest first. <b>Conf</b> is the heatmap value inside
                  the region — a wide faint region and a small intense one are different
                  findings that area alone cannot separate. <b>On mark</b> is how much of
                  that region sits on what you drew; a row near 0% is the model reacting
                  to something else and is greyed out.
                </div>
              </>
            )}
          </>
        )}

        <h4>Detection settings</h4>
        <div className="at-studio-group">
          <label>Score threshold: {threshold.toFixed(2)}</label>
          <input type="range" min={0} max={1} step={0.01} value={threshold}
                 onChange={(e) => setThreshold(Number(e.target.value))} />
        </div>
        <label className="at-studio-check">
          <input type="checkbox" checked={usePixelThreshold}
                 onChange={(e) => setUsePixelThreshold(e.target.checked)} />
          Build mask from heatmap instead of the model's own mask
        </label>
        {usePixelThreshold && (
          <div className="at-studio-group">
            <label>Pixel threshold: {pixelThreshold.toFixed(2)}</label>
            <input type="range" min={0.05} max={0.95} step={0.05} value={pixelThreshold}
                   onChange={(e) => setPixelThreshold(Number(e.target.value))} />
            <span className="at-hint">
              The model's built-in mask is calibrated during training and stays empty
              on projects with no pixel-level ground truth. Thresholding the heatmap
              yourself is what makes the mask (and any area rule) usable.
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
