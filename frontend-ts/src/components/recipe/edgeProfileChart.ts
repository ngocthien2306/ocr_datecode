/**
 * Shared 1D edge-profile chart, used by both edge-detection setup UIs:
 *  - EdgeSetupModal      (product walls, anchor_mode='edge_regions'|'label_strip')
 *  - ColorSetupModal     (cap edges, localization_method='edge_regions')
 *
 * Both read the same `profiles` payload shape from the backend
 * (image_proc_detector builds it in one place), so the renderer is shared too —
 * a threshold drawn differently in the two modals would be worse than useless
 * when the same number decides both.
 */

import { useEffect, useState } from 'react';

export type Pt = [number, number];

/** The app toggles themes by adding `dark-mode` to <body> (see Dashboard). */
export function isDarkMode(): boolean {
  return typeof document !== 'undefined' && document.body.classList.contains('dark-mode');
}

/**
 * Re-render when the theme flips. Canvas content is painted imperatively, so
 * unlike CSS it cannot restyle itself — without this a chart drawn in light mode
 * stays light on a dark modal until the next detect.
 */
export function useDarkMode(): boolean {
  const [dark, setDark] = useState(isDarkMode);
  useEffect(() => {
    const obs = new MutationObserver(() => setDark(isDarkMode()));
    obs.observe(document.body, { attributes: true, attributeFilter: ['class'] });
    setDark(isDarkMode());
    return () => obs.disconnect();
  }, []);
  return dark;
}

interface ChartPalette {
  bg: string; grid: string; curve: string; title: string; muted: string;
  peakPass: string; peakFail: string; peakLabel: string; ref: string; secondary: string;
}

function palette(): ChartPalette {
  return isDarkMode()
    ? {
        bg: '#111827', grid: '#374151', curve: '#d1d5db', title: '#9ca3af',
        muted: '#6b7280', peakPass: '#fbbf24', peakFail: '#6b7280',
        peakLabel: '#fcd34d', ref: '#f472b6', secondary: '#4ade80',
      }
    : {
        bg: '#f3f4f6', grid: '#e5e7eb', curve: '#374151', title: '#6b7280',
        muted: '#9ca3af', peakPass: '#f59e0b', peakFail: '#9ca3af',
        peakLabel: '#b45309', ref: '#ec4899', secondary: '#16a34a',
      };
}

export interface SideProfile {
  profile: number[];
  peaks: number[];
  peak_hratio?: number[];          // height_ratio per peak (what outer_hratio filters on)
  gap_offset: number;
  outer_col: number | null;
  inner_col: number | null;
  pred_col: number | null;
  outer_min_hratio?: number;       // current threshold → drawn for reference
  detect_mode?: string;
  specular_regions?: Pt[][];       // suppressed glare blobs in image pixel coords
}

export interface ProfileChartLabels {
  /** Big dot for the chosen edge. Default 'OUT'. */
  chosen?: string;
  /** Big dot for the secondary edge (inner wall). Omit to hide. */
  secondary?: string;
  /** Dashed vertical reference line. Default 'pred'. */
  reference?: string;
}

/**
 * Render a profile curve + all peaks + the chosen edge, in the light theme the
 * Python debug viz uses. Labels show the actual px value (= col + gap_offset).
 */
export function drawProfileChart(
  canvas: HTMLCanvasElement | null,
  data: SideProfile | null | undefined,
  title: string,
  labels: ProfileChartLabels = {},
) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width, H = canvas.height;
  const C = palette();
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  ctx.font = '11px sans-serif';
  ctx.fillStyle = C.title;
  ctx.fillText(title, 6, 13);
  if (!data || !data.profile || data.profile.length < 2) {
    ctx.fillStyle = C.muted;
    ctx.fillText('no data', 6, Math.round(H / 2));
    return;
  }
  const prof = data.profile;
  const n = prof.length;
  const padT = 18, padB = 14;
  const plotH = H - padT - padB;
  const go = data.gap_offset || 0;
  const xOf = (col: number) => (col / (n - 1)) * W;
  const yOf = (v: number) => padT + plotH - v * plotH;

  // profile curve
  ctx.strokeStyle = C.curve;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xOf(i), y = yOf(prof[i]!);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Candidate peaks coloured by whether their height_ratio passes the coverage
  // threshold — that is the quantity deciding if a peak can become the edge.
  // Qualifying peaks are amber + labelled with their hr; rejected ones small grey.
  const thr = data.outer_min_hratio ?? 0;
  const hr = data.peak_hratio || [];
  data.peaks.forEach((p, i) => {
    if (p < 0 || p >= n) return;
    const h = hr[i];
    const pass = h != null && h >= thr;
    ctx.fillStyle = pass ? C.peakPass : C.peakFail;
    ctx.beginPath(); ctx.arc(xOf(p), yOf(prof[p]!), pass ? 3.5 : 2.5, 0, Math.PI * 2); ctx.fill();
    if (pass && h != null) {
      ctx.font = '9px sans-serif';
      ctx.fillStyle = C.peakLabel;
      ctx.fillText(h.toFixed(2), Math.min(xOf(p) + 4, W - 20), Math.max(10, yOf(prof[p]!) - 8));
      ctx.font = '11px sans-serif';
    }
  });
  // threshold reference (top-right): peaks with hr ≥ this become edge candidates
  ctx.font = '10px sans-serif';
  ctx.fillStyle = C.peakLabel;
  const thrTxt = `coverage ≥ ${thr.toFixed(2)} = edge`;
  ctx.fillText(thrTxt, Math.max(6, W - ctx.measureText(thrTxt).width - 6), 13);
  ctx.font = '11px sans-serif';

  // reference position (pink dashed vertical line)
  if (data.pred_col != null && data.pred_col >= 0 && data.pred_col < n) {
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = C.ref;
    ctx.lineWidth = 1;
    const px = xOf(data.pred_col);
    ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, H - padB); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = C.ref;
    ctx.fillText(`${labels.reference ?? 'pred'} ${Math.round(data.pred_col + go)}`, 6, H - 3);
  }

  const bigDot = (col: number | null, color: string, label: string) => {
    if (col == null || col < 0 || col >= n) return;
    const x = xOf(col), y = yOf(prof[Math.round(col)] ?? 0);
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillText(label, Math.min(x + 7, W - 44), Math.max(12, y - 6));
  };
  const chosen = labels.chosen ?? 'OUT';
  bigDot(data.outer_col, C.peakPass,
    data.outer_col != null ? `${chosen} ${Math.round(data.outer_col + go)}` : chosen);
  if (labels.secondary) {
    bigDot(data.inner_col, C.secondary,
      data.inner_col != null ? `${labels.secondary} ${Math.round(data.inner_col + go)}` : labels.secondary);
  }
}
