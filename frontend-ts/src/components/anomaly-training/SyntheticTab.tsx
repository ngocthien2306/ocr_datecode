import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  anomalyTrainingAPI, DatasetImage, DefectEdge, SyntheticOptions,
  SyntheticPreviewResult,
} from '@/services/anomalyTraining';
import '@/styles/AnomalyTraining.css';

interface Props {
  projectId: string;
  onGenerated: () => void;
}

const DELTA_CHOICES = [10, 15, 20, 25, 30, 40, 55, 75];
const WIDTH_CHOICES = [2, 3, 4, 6, 8, 10, 14];
const EDGE_CHOICES: { id: DefectEdge; label: string; help: string }[] = [
  { id: 'wrinkle', label: 'Wrinkle', help: 'Crease — highlight + shadow pair' },
  { id: 'bubble', label: 'Bubble', help: 'Blister under the label — a shallow dome that redirects light' },
  { id: 'soft', label: 'Smudge', help: 'Blurred mark, no hard edge' },
  { id: 'hard', label: 'Scratch', help: 'Sharp-edged line' },
];
const MULTIPLIERS = [1, 2, 3, 4];

function toggle<T>(list: T[], value: T): T[] {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value].sort(
    (a, b) => (typeof a === 'number' && typeof b === 'number' ? a - b : 0),
  );
}

/**
 * Generate synthetic NG samples — controlled defects drawn onto clean labels.
 *
 * These do NOT teach the model what a defect looks like. PatchCore/Padim fit
 * their memory bank on normal images only; anomalib forces every abnormal
 * sample into the test split. What a population of NG samples buys is a
 * decision threshold calibrated against something other than a single point,
 * plus AUROC/F1 that mean anything. That is the reason to use this tab, and
 * the UI says so rather than letting the name imply otherwise.
 */
export default function SyntheticTab({ projectId, onGenerated }: Props) {
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const [deltas, setDeltas] = useState<number[]>([20, 30, 40, 55]);
  const [widths, setWidths] = useState<number[]>([4, 10]);
  const [edges, setEdges] = useState<DefectEdge[]>(['wrinkle', 'bubble']);
  const [polarities, setPolarities] = useState<('dark' | 'bright')[]>(['dark']);
  const [marksMin, setMarksMin] = useState(1);
  const [marksMax, setMarksMax] = useState(3);
  const [multiplier, setMultiplier] = useState(2);
  const [defectType, setDefectType] = useState('synthetic_wrinkle');
  const [seed, setSeed] = useState(20260803);

  const [preview, setPreview] = useState<SyntheticPreviewResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [doneMsg, setDoneMsg] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const page = await anomalyTrainingAPI.listDatasetImages(projectId, {
          label: 'normal', page: 1, pageSize: 48,
        });
        setImages(page.images);
      } catch (e: any) {
        setErrorMsg(e?.response?.data?.detail || 'Could not load normal images');
      }
    })();
  }, [projectId]);

  const options: SyntheticOptions = useMemo(
    () => ({
      deltas, widths, edges, polarities,
      marks_min: Math.min(marksMin, marksMax),
      marks_max: Math.max(marksMin, marksMax),
    }),
    [deltas, widths, edges, polarities, marksMin, marksMax],
  );

  // Output size is simply a multiple of the selected base images — the option
  // pools widen the variety of each image, not the count.
  const estimate = selected.size * multiplier;

  const body = () => ({
    base_item_ids: [...selected], options, multiplier, defect_type: defectType, seed,
  });

  const runPreview = useCallback(async () => {
    setBusy(true); setErrorMsg(null); setDoneMsg(null);
    try {
      setPreview(await anomalyTrainingAPI.previewSynthetic(projectId, body()));
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || 'Preview failed');
      setPreview(null);
    } finally { setBusy(false); }
  }, [projectId, selected, options, multiplier, defectType, seed]);

  const runGenerate = useCallback(async () => {
    setBusy(true); setErrorMsg(null); setDoneMsg(null);
    try {
      const r = await anomalyTrainingAPI.generateSynthetic(projectId, body());
      setDoneMsg(
        `Generated ${r.generated} image(s) into test/${r.defect_type} (batch ${r.batch_id}). ` +
        `Dataset now has ${r.normal} normal / ${r.abnormal} abnormal. Retrain to recalibrate the threshold.`,
      );
      setPreview(null);
      onGenerated();
    } catch (e: any) {
      setErrorMsg(e?.response?.data?.detail || 'Generation failed');
    } finally { setBusy(false); }
  }, [projectId, selected, options, multiplier, defectType, seed, onGenerated]);

  const canRun = selected.size > 0 && deltas.length > 0 && widths.length > 0 && edges.length > 0;

  return (
    <div className="at-syn">
      <div className="at-syn-note">
        <b>What this changes.</b> Synthetic NG samples are added as <i>abnormal test</i>
        {' '}images. They never train the model — PatchCore fits on normal images only.
        What they do is give the decision threshold a population to separate instead of
        a single defect, and make AUROC/F1 measurable. After generating, retrain: every
        score is re-normalised, so recipe thresholds need re-checking.
      </div>

      <div className="at-syn-cols">
        <div className="at-syn-col">
          <h4>1. Clean base images</h4>
          <div className="at-hint">
            Normal images only — drawing a defect on an already-defective label would
            make the sample uninterpretable. Marks are auto-placed on blank label
            surface, avoiding printed text.
          </div>
          <div className="at-syn-actions-inline">
            <button className="at-btn at-btn-secondary at-btn-sm"
                    onClick={() => setSelected(new Set(images.map((i) => i.id)))}>Select all</button>
            <button className="at-btn at-btn-secondary at-btn-sm"
                    onClick={() => setSelected(new Set())}>Clear</button>
            <span className="at-hint">{selected.size} of {images.length} selected</span>
          </div>
          <div className="at-syn-bases">
            {images.map((im) => (
              <button key={im.id}
                      className={`at-syn-base ${selected.has(im.id) ? 'active' : ''}`}
                      title={`${im.camera_serial} · frame ${im.frame_idx}`}
                      onClick={() => setSelected((s) => {
                        const n = new Set(s);
                        n.has(im.id) ? n.delete(im.id) : n.add(im.id);
                        return n;
                      })}>
                <img src={`data:image/jpeg;base64,${im.thumb_b64}`} alt="" />
              </button>
            ))}
            {!images.length && <span className="at-hint">No normal images in this project yet.</span>}
          </div>
        </div>

        <div className="at-syn-col">
          <h4>2. Defect options</h4>

          <div className="at-syn-row">
            <label>Defect types — tick any, each mark picks one at random</label>
            <div className="at-syn-chips">
              {EDGE_CHOICES.map((e) => (
                <button key={e.id} title={e.help}
                        className={`at-chip ${edges.includes(e.id) ? 'on' : ''}`}
                        onClick={() => setEdges(toggle(edges, e.id))}>{e.label}</button>
              ))}
              <button className="at-chip ghost"
                      onClick={() => setEdges(EDGE_CHOICES.map((e) => e.id))}>All</button>
            </div>
            <span className="at-hint">
              {EDGE_CHOICES.filter((e) => edges.includes(e.id)).map((e) => e.help).join(' · ') || 'Pick at least one'}
            </span>
          </div>

          <div className="at-syn-row">
            <label>Contrast Δ (grey offset vs local background)</label>
            <div className="at-syn-chips">
              {DELTA_CHOICES.map((d) => (
                <button key={d} className={`at-chip ${deltas.includes(d) ? 'on' : ''}`}
                        onClick={() => setDeltas(toggle(deltas, d))}>{d}</button>
              ))}
              <button className="at-chip ghost" onClick={() => setDeltas(DELTA_CHOICES)}>All</button>
            </div>
            <span className="at-hint">
              Straddle the model's limit — samples both sides of it are what make the
              threshold land in a meaningful place.
            </span>
          </div>

          <div className="at-syn-row">
            <label>Thickness (px)</label>
            <div className="at-syn-chips">
              {WIDTH_CHOICES.map((w) => (
                <button key={w} className={`at-chip ${widths.includes(w) ? 'on' : ''}`}
                        onClick={() => setWidths(toggle(widths, w))}>{w}</button>
              ))}
              <button className="at-chip ghost" onClick={() => setWidths(WIDTH_CHOICES)}>All</button>
            </div>
            <span className="at-hint">For bubbles this sets how soft the rim is, not a line width.</span>
          </div>

          {/* Compact controls sit side by side — each is a single small value,
              and stacking them full-width made the panel scroll for no reason. */}
          <div className="at-syn-fields">
            <div className="at-syn-field">
              <label>How many to generate</label>
              <div className="at-syn-chips">
                {MULTIPLIERS.map((m) => (
                  <button key={m} className={`at-chip ${multiplier === m ? 'on' : ''}`}
                          onClick={() => setMultiplier(m)}>x{m}</button>
                ))}
              </div>
            </div>

            <div className="at-syn-field">
              <label>Marks per image</label>
              <div className="at-syn-range">
                <input type="number" min={1} max={5} value={marksMin}
                       onChange={(e) => setMarksMin(Math.min(5, Math.max(1, Number(e.target.value))))} />
                <span>–</span>
                <input type="number" min={1} max={5} value={marksMax}
                       onChange={(e) => setMarksMax(Math.min(5, Math.max(1, Number(e.target.value))))} />
              </div>
            </div>

            <div className="at-syn-field">
              <label>
                Direction
                {!edges.some((e) => e === 'hard' || e === 'soft') && (
                  <span className="at-syn-na"> — n/a for wrinkle/bubble</span>
                )}
              </label>
              <div className="at-syn-chips">
                {(['dark', 'bright'] as const).map((pol) => (
                  <button key={pol} className={`at-chip ${polarities.includes(pol) ? 'on' : ''}`}
                          disabled={!edges.some((e) => e === 'hard' || e === 'soft')}
                          title="Only applies to scratch and smudge — wrinkle and bubble draw both sides"
                          onClick={() => setPolarities(toggle(polarities, pol))}>{pol}</button>
                ))}
              </div>
            </div>

            <div className="at-syn-field">
              <label>Seed</label>
              <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
            </div>

            <div className="at-syn-field grow">
              <label>Defect type folder</label>
              <input type="text" value={defectType} onChange={(e) => setDefectType(e.target.value)} />
            </div>
          </div>

          <span className="at-hint">
            Marks per image is drawn at random in that range — real labels rarely carry
            exactly one defect, and a fixed count is another regularity the model could
            key on. Amount is a multiple of the {selected.size} base image(s) selected;
            each output re-rolls its base, its mark count and every mark's parameters.
            The same seed reproduces the same batch. The folder name becomes a defect
            class under <code>test/</code> as soon as the first file lands.
          </span>

          <div className="at-syn-footer">
            <div className="at-syn-estimate">
              {selected.size === 0
                ? <span className="at-syn-over">Select at least one base image</span>
                : <>Will generate <b>{estimate}</b> image(s) — {selected.size} base × x{multiplier}</>}
              {estimate > 500 && <span className="at-syn-over"> · over the 500 limit</span>}
            </div>
            <button className="at-btn at-btn-secondary" disabled={!canRun || busy} onClick={runPreview}>
              {busy ? 'Working…' : 'Preview'}
            </button>
            <button className="at-btn at-btn-primary" disabled={!canRun || busy || estimate > 500}
                    onClick={runGenerate}>
              Generate + add to dataset
            </button>
          </div>

          {errorMsg && <div className="at-alert-error">{errorMsg}</div>}
          {doneMsg && <div className="at-alert-ok">{doneMsg}</div>}
        </div>
      </div>

      {preview && (
        <div className="at-syn-preview">
          <h4>
            Preview — {preview.samples.length} of {preview.total_to_generate} shown
            {preview.over_limit && <span className="at-syn-over"> (over limit: {preview.max_generate})</span>}
          </h4>
          <div className="at-hint">Nothing has been written yet. Check the marks look like plausible defects before generating.</div>
          <div className="at-syn-grid">
            {preview.samples.map((s, i) => (
              <figure key={i}>
                <img src={`data:image/jpeg;base64,${s.image_b64}`} alt="" />
                <figcaption>
                  {s.n_marks} mark{s.n_marks > 1 ? 's' : ''} · {s.edges.join(', ')}
                  <br />Δ {s.deltas.map((d) => Math.round(d)).join('/')}
                </figcaption>
              </figure>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
