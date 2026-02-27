import type { SummaryStatistics, TimeseriesStatistics } from '@/services/inferenceResults';
import { getTimezoneOffset } from '@/utils/timezone';

// ─── Types ────────────────────────────────────────────────────────────────────

export type ReportTheme = 'industrial' | 'dark' | 'executive';

export interface ReportSections {
  kpi: boolean;
  trendChart: boolean;
  passfailChart: boolean;
  perRecipe: boolean;
}

export interface ReportConfig {
  periodLabel: string;
  startDate: string;
  endDate: string;
  granularity: 'hour' | 'day';
  generatedAt: string;
  theme: ReportTheme;
  sections: ReportSections;
  selectedRecipeIds: string[];
}

// ─── Per-theme color palettes ─────────────────────────────────────────────────

const THEME_CHART_COLORS: Record<ReportTheme, string[]> = {
  industrial: ['#2563eb','#16a34a','#dc2626','#d97706','#7c3aed','#0891b2','#be185d','#4d7c0f','#374151','#b45309'],
  dark:       ['#38bdf8','#4ade80','#f87171','#fbbf24','#c084fc','#67e8f9','#f472b6','#a3e635','#94a3b8','#fb923c'],
  executive:  ['#2563eb','#0d9488','#d97706','#7c3aed','#dc2626','#0891b2','#be185d','#65a30d','#374151','#b45309'],
};

const THEME_PASS_FAIL: Record<ReportTheme, { pass: string; fail: string }> = {
  industrial: { pass: '#16a34aCC', fail: '#dc2626CC' },
  dark:       { pass: '#4ade80CC', fail: '#f87171CC' },
  executive:  { pass: '#059669CC', fail: '#dc2626CC' },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function passRateStatusClass(rate: number): string {
  if (rate >= 90) return 'st-ok';
  if (rate >= 70) return 'st-warn';
  return 'st-fail';
}

function passRateStatus(rate: number): string {
  if (rate >= 90) return 'OK';
  if (rate >= 70) return 'WARN';
  return 'FAIL';
}

function fmtLabel(ts: string, granularity: 'hour' | 'day'): string {
  // Convert UTC → local timezone (UTC+7 VN by default, or from settings)
  const offsetMs = getTimezoneOffset() * 60_000;
  const local = new Date(new Date(ts).getTime() + offsetMs);
  if (granularity === 'hour') {
    return `${local.getUTCHours().toString().padStart(2,'0')}:${local.getUTCMinutes().toString().padStart(2,'0')}`;
  }
  const M = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  return `${M[local.getUTCMonth()]} ${local.getUTCDate()}`;
}

function safeJson(obj: unknown): string {
  return JSON.stringify(obj).replace(/<\/script>/gi, '<\\/script>');
}

// ─── CSS ─────────────────────────────────────────────────────────────────────

function buildStyles(): string {
  return `
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    /* ══ CSS VARIABLES – Theme defaults (Industrial) ══ */
    :root {
      --bg:               #ffffff;
      --surface:          #ffffff;
      --surface-alt:      #f3f4f6;
      --border:           #d1d5db;
      --border-strong:    #9ca3af;
      --text:             #111111;
      --text-muted:       #6b7280;
      --text-inv:         #ffffff;

      --pass-color:       #15803d;
      --fail-color:       #b91c1c;
      --warn-color:       #92400e;

      --header-bg:        #1e2a3a;
      --header-text:      #ffffff;
      --header-sub:       #94a3b8;
      --header-accent:    #2563eb;

      --sec-title-bg:     #374151;
      --sec-title-color:  #ffffff;
      --sec-border:       #d1d5db;

      --kpi-bg:           #ffffff;
      --kpi-label-bg:     transparent;
      --kpi-label-color:  #6b7280;
      --kpi-label-pad:    0 0 6px;

      --th-bg:            #f3f4f6;
      --th-color:         #374151;
      --th-border:        #d1d5db;
      --td-even:          #f9fafb;
      --tfoot-bg:         #f3f4f6;

      --recipe-hd-bg:     #f3f4f6;
      --recipe-border:    #d1d5db;

      --chart-grid:       #d1d5db;
      --chart-tick:       #374151;
    }

    /* ══ Theme 2: Dark Enterprise ══ */
    body.theme-dark {
      --bg:               #0f172a;
      --surface:          #1e293b;
      --surface-alt:      #0f172a;
      --border:           #334155;
      --border-strong:    #475569;
      --text:             #e2e8f0;
      --text-muted:       #94a3b8;
      --text-inv:         #0f172a;

      --pass-color:       #4ade80;
      --fail-color:       #f87171;
      --warn-color:       #fbbf24;

      --header-bg:        #020617;
      --header-text:      #f1f5f9;
      --header-sub:       #64748b;
      --header-accent:    #38bdf8;

      --sec-title-bg:     #1e293b;
      --sec-title-color:  #38bdf8;
      --sec-border:       #334155;

      --kpi-bg:           #1e293b;
      --kpi-label-bg:     transparent;
      --kpi-label-color:  #94a3b8;
      --kpi-label-pad:    0 0 6px;

      --th-bg:            #0f172a;
      --th-color:         #38bdf8;
      --th-border:        #334155;
      --td-even:          #162032;
      --tfoot-bg:         #0f172a;

      --recipe-hd-bg:     #1e293b;
      --recipe-border:    #334155;

      --chart-grid:       #1e3a5f;
      --chart-tick:       #94a3b8;
    }

    /* ══ Theme 3: Executive ══ */
    body.theme-executive {
      --bg:               #eef2f7;
      --surface:          #ffffff;
      --surface-alt:      #f8fafc;
      --border:           #e2e8f0;
      --border-strong:    #cbd5e1;
      --text:             #1e293b;
      --text-muted:       #64748b;
      --text-inv:         #ffffff;

      --pass-color:       #059669;
      --fail-color:       #dc2626;
      --warn-color:       #d97706;

      --header-bg:        #1e3a8a;
      --header-text:      #ffffff;
      --header-sub:       #bfdbfe;
      --header-accent:    #60a5fa;

      --sec-title-bg:     #ffffff;
      --sec-title-color:  #1e40af;
      --sec-border:       #bfdbfe;

      --kpi-bg:           #ffffff;
      --kpi-label-bg:     #1e40af;
      --kpi-label-color:  #ffffff;
      --kpi-label-pad:    8px 16px;

      --th-bg:            #1e40af;
      --th-color:         #ffffff;
      --th-border:        #1e40af;
      --td-even:          #f0f7ff;
      --tfoot-bg:         #dbeafe;

      --recipe-hd-bg:     #eff6ff;
      --recipe-border:    #bfdbfe;

      --chart-grid:       #e2e8f0;
      --chart-tick:       #1e293b;
    }

    /* ══ Base styles ══ */
    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      font-size: 13px;
      color: var(--text);
      background: var(--bg);
    }
    .container { max-width: 1100px; margin: 0 auto; padding: 20px 28px; }

    /* ── Status classes ── */
    .st-ok   { color: var(--pass-color); font-weight: 700; }
    .st-warn { color: var(--warn-color); font-weight: 700; }
    .st-fail { color: var(--fail-color); font-weight: 700; }

    /* ── Header ── */
    .report-header {
      background: var(--header-bg);
      color: var(--header-text);
      padding: 20px 28px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 3px solid var(--header-accent);
    }
    .report-title   { font-size: 20px; font-weight: 700; letter-spacing: 0.02em; margin-bottom: 4px; }
    .report-subtitle { font-size: 12px; color: var(--header-sub); line-height: 1.8; }
    .report-meta    { text-align: right; font-size: 12px; color: var(--header-sub); line-height: 1.8; }

    /* Executive header: gradient accent bar below */
    body.theme-executive .report-header { background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%); }
    /* Dark header: slight glow on accent border */
    body.theme-dark .report-header { border-bottom-width: 2px; border-bottom-style: solid; }

    /* ── Section wrapper ── */
    .section {
      border: 1px solid var(--sec-border);
      background: var(--surface);
      margin-bottom: 16px;
    }
    body.theme-executive .section { box-shadow: 0 1px 4px rgba(0,0,0,0.06); }

    .section-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: var(--sec-title-bg);
      color: var(--sec-title-color);
      padding: 8px 14px;
    }
    /* Executive: white bg with blue left border */
    body.theme-executive .section-title {
      border-left: 4px solid #2563eb;
      border-bottom: 1px solid var(--sec-border);
      font-size: 12px;
      padding-left: 12px;
    }
    /* Dark: add bottom border */
    body.theme-dark .section-title {
      border-bottom: 1px solid var(--border);
      font-size: 12px;
      letter-spacing: 0.1em;
    }

    /* ── KPI row ── */
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
    }
    .kpi-cell {
      background: var(--kpi-bg);
      border-right: 1px solid var(--border);
      text-align: center;
      padding: 16px 14px;
    }
    .kpi-cell:last-child { border-right: none; }
    .kpi-label {
      background: var(--kpi-label-bg);
      color: var(--kpi-label-color);
      padding: var(--kpi-label-pad);
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      margin-bottom: 6px;
    }
    .kpi-value {
      font-size: 30px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      line-height: 1;
      color: var(--text);
    }
    .kpi-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

    /* Executive KPI: colored header strip = label floats to top edge */
    body.theme-executive .kpi-cell {
      padding: 0;
      overflow: hidden;
      border: none;
      border-right: 1px solid var(--border);
    }
    body.theme-executive .kpi-cell:last-child { border-right: none; }
    body.theme-executive .kpi-label {
      display: block;
      margin: 0;
      padding: 8px 14px;
      text-align: left;
    }
    body.theme-executive .kpi-value { padding: 14px 14px 4px; text-align: center; }
    body.theme-executive .kpi-sub   { padding: 0 14px 14px; text-align: center; }

    /* Dark KPI: outlined box style */
    body.theme-dark .kpi-cell {
      border-right: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
    }

    /* ── Charts ── */
    .chart-wrap    { position: relative; height: 340px; padding: 12px 8px 6px; background: var(--surface); }
    .chart-wrap-sm { position: relative; height: 200px; padding: 8px 8px 4px;  background: var(--surface); }

    /* ── Tables ── */
    .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .data-table th {
      background: var(--th-bg);
      color: var(--th-color);
      text-align: left;
      padding: 8px 12px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border: 1px solid var(--th-border);
    }
    .data-table td {
      padding: 7px 12px;
      border: 1px solid var(--border);
      color: var(--text);
      font-variant-numeric: tabular-nums;
    }
    .data-table tr:nth-child(even) td { background: var(--td-even); }
    .data-table tfoot td { background: var(--tfoot-bg); font-weight: 700; border-top: 2px solid var(--border-strong); }
    .num { text-align: right; }

    /* ── Recipe section ── */
    .recipe-section { border: 1px solid var(--recipe-border); margin-bottom: 16px; background: var(--surface); }
    body.theme-executive .recipe-section { box-shadow: 0 1px 4px rgba(0,0,0,0.06); }

    .recipe-section-header {
      background: var(--recipe-hd-bg);
      padding: 10px 14px;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      border-bottom: 1px solid var(--recipe-border);
    }
    /* Executive: colored left border per recipe */
    body.theme-executive .recipe-section-header { border-bottom: 1px solid var(--sec-border); }

    .recipe-name {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text);
    }
    .recipe-kpi-inline { font-size: 12px; color: var(--text-muted); display: flex; gap: 20px; }
    .rki-val { font-weight: 700; }

    /* ── Footer ── */
    .report-footer {
      border-top: 2px solid var(--header-bg);
      margin-top: 8px;
      padding: 10px 0;
      font-size: 11px;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
    }
    body.theme-dark .report-footer { border-top-color: var(--border-strong); }

    /* ── Print ── */
    @media print {
      body { font-size: 11px; }
      .container { padding: 0; max-width: 100%; }
      .recipe-section, .section { page-break-inside: avoid; }
      .chart-wrap    { height: 260px; }
      .chart-wrap-sm { height: 160px; }
      body.theme-dark { background: white !important; color: black !important; }
    }
  `;
}

// ─── Header ───────────────────────────────────────────────────────────────────

function buildHeader(config: ReportConfig): string {
  const startD  = new Date(config.startDate);
  const endD    = new Date(config.endDate);
  const fmtDate = (d: Date) =>
    `${d.getUTCDate().toString().padStart(2,'0')}/${(d.getUTCMonth()+1).toString().padStart(2,'0')}/${d.getUTCFullYear()}`;
  const genStr  = new Date(config.generatedAt).toLocaleString('en-US', {
    year:'numeric', month:'2-digit', day:'2-digit',
    hour:'2-digit', minute:'2-digit', second:'2-digit', hour12: false,
  });

  return `
  <div class="report-header">
    <div>
      <div class="report-title">PRODUCTION INSPECTION REPORT</div>
      <div class="report-subtitle">
        Period: ${config.periodLabel} &nbsp;|&nbsp; ${fmtDate(startD)} – ${fmtDate(endD)}<br>
        Breakdown: By ${config.granularity === 'hour' ? 'Hour' : 'Day'}
      </div>
    </div>
    <div class="report-meta">
      Generated: ${genStr}<br>
      OCR Datecode System
    </div>
  </div>`;
}

// ─── KPI Summary ─────────────────────────────────────────────────────────────

function buildKPI(summary: SummaryStatistics, config: ReportConfig): string {
  let total = summary.total, pass = summary.pass, fail = summary.fail, rate = summary.pass_rate;
  if (config.selectedRecipeIds.length > 0) {
    const f = summary.by_recipe.filter(r => config.selectedRecipeIds.includes(r.recipe_id));
    total = f.reduce((s, r) => s + r.total, 0);
    pass  = f.reduce((s, r) => s + r.pass,  0);
    fail  = f.reduce((s, r) => s + r.fail,  0);
    rate  = total > 0 ? Math.round(pass / total * 1000) / 10 : 0;
  }

  return `
  <div class="section">
    <div class="section-title">Inspection Summary</div>
    <div class="kpi-row">
      <div class="kpi-cell">
        <div class="kpi-label">Total Inspected</div>
        <div class="kpi-value">${total.toLocaleString()}</div>
        <div class="kpi-sub">${config.periodLabel}</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Pass</div>
        <div class="kpi-value st-ok">${pass.toLocaleString()}</div>
        <div class="kpi-sub">${total > 0 ? Math.round(pass/total*100) : 0}% of total</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Fail</div>
        <div class="kpi-value st-fail">${fail.toLocaleString()}</div>
        <div class="kpi-sub">${total > 0 ? Math.round(fail/total*100) : 0}% of total</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Pass Rate</div>
        <div class="kpi-value ${passRateStatusClass(rate)}">${rate}%</div>
        <div class="kpi-sub ${passRateStatusClass(rate)}">${passRateStatus(rate)}</div>
      </div>
    </div>
  </div>`;
}

// ─── Recipe Summary Table ─────────────────────────────────────────────────────

function buildRecipeSummaryCards(summary: SummaryStatistics, config: ReportConfig): string {
  let recipes = summary.by_recipe;
  if (config.selectedRecipeIds.length > 0) {
    recipes = recipes.filter(r => config.selectedRecipeIds.includes(r.recipe_id));
  }
  if (recipes.length === 0) return '';

  const rows = recipes.map((r, idx) => {
    const color = THEME_CHART_COLORS[config.theme][idx % THEME_CHART_COLORS[config.theme].length];
    const rate  = r.pass_rate ?? (r.total > 0 ? Math.round(r.pass / r.total * 1000) / 10 : 0);
    return `
      <tr>
        <td>
          <span style="display:inline-block;width:9px;height:9px;background:${color};margin-right:7px;vertical-align:middle"></span>
          ${r.recipe_name}
        </td>
        <td class="num">${r.total.toLocaleString()}</td>
        <td class="num st-ok">${r.pass.toLocaleString()}</td>
        <td class="num st-fail">${r.fail.toLocaleString()}</td>
        <td class="num ${passRateStatusClass(rate)}">${rate}%</td>
        <td class="${passRateStatusClass(rate)}">${passRateStatus(rate)}</td>
      </tr>`;
  }).join('');

  return `
  <div class="section">
    <div class="section-title">Recipe Summary</div>
    <table class="data-table">
      <thead>
        <tr>
          <th>Recipe</th>
          <th style="text-align:right">Total</th>
          <th style="text-align:right">Pass</th>
          <th style="text-align:right">Fail</th>
          <th style="text-align:right">Pass Rate</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;
}

// ─── Combined Chart Canvases ──────────────────────────────────────────────────

function buildCombinedChartsHtml(): string {
  return `
  <div class="section">
    <div class="section-title">Production Trend – All Recipes</div>
    <div class="chart-wrap"><canvas id="chartTrend"></canvas></div>
  </div>
  <div class="section">
    <div class="section-title">Pass / Fail Breakdown – All Recipes</div>
    <div class="chart-wrap"><canvas id="chartPassFail"></canvas></div>
  </div>`;
}

// ─── Per-Recipe Sections ──────────────────────────────────────────────────────

function buildPerRecipeSections(
  timeseries: TimeseriesStatistics,
  summary: SummaryStatistics,
  config: ReportConfig,
): string {
  const recipeMap = new Map<string, string>();
  timeseries.data.forEach(pt => pt.by_recipe.forEach(r => {
    if (!recipeMap.has(r.recipe_id)) recipeMap.set(r.recipe_id, r.recipe_name);
  }));

  let recipes = Array.from(recipeMap.entries()).map(([id, name]) => ({ id, name }));
  if (config.selectedRecipeIds.length > 0) {
    recipes = recipes.filter(r => config.selectedRecipeIds.includes(r.id));
  }
  if (recipes.length === 0) return '';

  const granLabel = config.granularity === 'hour' ? 'Hour' : 'Date';
  const colors = THEME_CHART_COLORS[config.theme];

  return recipes.map((recipe, idx) => {
    const rs  = summary.by_recipe.find(r => r.recipe_id === recipe.id);
    const tot = rs?.total ?? 0, pas = rs?.pass ?? 0, fai = rs?.fail ?? 0;
    const rat = rs?.pass_rate ?? (tot > 0 ? Math.round(pas / tot * 1000) / 10 : 0);
    const col = colors[idx % colors.length];

    let sumT = 0, sumP = 0, sumF = 0;
    const tableRows = timeseries.data.map(pt => {
      const r = pt.by_recipe.find(x => x.recipe_id === recipe.id);
      if (!r || r.total === 0) return '';
      sumT += r.total; sumP += r.pass; sumF += r.fail;
      const rr = r.total > 0 ? Math.round(r.pass / r.total * 1000) / 10 : 0;
      return `
        <tr>
          <td>${fmtLabel(pt.timestamp, config.granularity)}</td>
          <td class="num">${r.total.toLocaleString()}</td>
          <td class="num st-ok">${r.pass.toLocaleString()}</td>
          <td class="num st-fail">${r.fail.toLocaleString()}</td>
          <td class="num ${passRateStatusClass(rr)}">${rr}%</td>
          <td class="${passRateStatusClass(rr)}">${passRateStatus(rr)}</td>
        </tr>`;
    }).join('');

    const sumRate = sumT > 0 ? Math.round(sumP / sumT * 1000) / 10 : 0;

    return `
  <div class="recipe-section" style="border-left:3px solid ${col}">
    <div class="recipe-section-header">
      <span class="recipe-name">${recipe.name}</span>
      <div class="recipe-kpi-inline">
        <span>Total: <span class="rki-val">${tot.toLocaleString()}</span></span>
        <span class="st-ok">Pass: <span class="rki-val">${pas.toLocaleString()}</span></span>
        <span class="st-fail">Fail: <span class="rki-val">${fai.toLocaleString()}</span></span>
        <span class="${passRateStatusClass(rat)}">Rate: <span class="rki-val">${rat}%</span></span>
      </div>
    </div>
    <div class="chart-wrap-sm"><canvas id="chartRecipe_${idx}"></canvas></div>
    <table class="data-table">
      <thead>
        <tr>
          <th>${granLabel}</th>
          <th style="text-align:right">Total</th>
          <th style="text-align:right">Pass</th>
          <th style="text-align:right">Fail</th>
          <th style="text-align:right">Pass Rate</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>${tableRows || `<tr><td colspan="6" style="text-align:center;padding:16px;color:var(--text-muted)">No data for this period</td></tr>`}</tbody>
      ${sumT > 0 ? `
      <tfoot>
        <tr>
          <td>TOTAL</td>
          <td class="num">${sumT.toLocaleString()}</td>
          <td class="num">${sumP.toLocaleString()}</td>
          <td class="num">${sumF.toLocaleString()}</td>
          <td class="num ${passRateStatusClass(sumRate)}">${sumRate}%</td>
          <td class="${passRateStatusClass(sumRate)}">${passRateStatus(sumRate)}</td>
        </tr>
      </tfoot>` : ''}
    </table>
  </div>`;
  }).join('');
}

// ─── Chart Init Scripts ───────────────────────────────────────────────────────

function buildChartScripts(timeseries: TimeseriesStatistics, config: ReportConfig): string {
  const recipeMap = new Map<string, string>();
  timeseries.data.forEach(pt => pt.by_recipe.forEach(r => {
    if (!recipeMap.has(r.recipe_id)) recipeMap.set(r.recipe_id, r.recipe_name);
  }));

  let recipes = Array.from(recipeMap.entries()).map(([id, name]) => ({ id, name }));
  if (config.selectedRecipeIds.length > 0) {
    recipes = recipes.filter(r => config.selectedRecipeIds.includes(r.id));
  }

  const colors  = THEME_CHART_COLORS[config.theme];
  const pf      = THEME_PASS_FAIL[config.theme];
  const labels  = timeseries.data.map(pt => fmtLabel(pt.timestamp, config.granularity));

  // Trend line datasets
  const trendDS = recipes.map((recipe, idx) => {
    const color = colors[idx % colors.length];
    const data  = timeseries.data.map(pt => pt.by_recipe.find(x => x.recipe_id === recipe.id)?.total ?? 0);
    return { label: recipe.name, data, borderColor: color, backgroundColor: color + '20',
             borderWidth: 2, pointRadius: 4, tension: 0.3, fill: false };
  });

  // Pass/Fail stacked bar datasets
  const pfDS: object[] = [];
  recipes.forEach((recipe, idx) => {
    const color = colors[idx % colors.length];
    const passD = timeseries.data.map(pt => pt.by_recipe.find(x => x.recipe_id === recipe.id)?.pass ?? 0);
    const failD = timeseries.data.map(pt => pt.by_recipe.find(x => x.recipe_id === recipe.id)?.fail ?? 0);
    pfDS.push({ label: `${recipe.name} (Pass)`, data: passD, backgroundColor: color + 'CC', stack: `r${idx}` });
    pfDS.push({ label: `${recipe.name} (Fail)`, data: failD, backgroundColor: color + '55', stack: `r${idx}` });
  });

  // Per-recipe mini chart scripts
  const perRecipeScripts = recipes.map((recipe, idx) => {
    const passD = timeseries.data.map(pt => pt.by_recipe.find(x => x.recipe_id === recipe.id)?.pass ?? 0);
    const failD = timeseries.data.map(pt => pt.by_recipe.find(x => x.recipe_id === recipe.id)?.fail ?? 0);
    return `
    initBarChart('chartRecipe_${idx}', LABELS, [
      { label: 'Pass', data: ${safeJson(passD)}, backgroundColor: '${pf.pass}', stack: 's0' },
      { label: 'Fail', data: ${safeJson(failD)}, backgroundColor: '${pf.fail}', stack: 's0' }
    ]);`;
  }).join('\n');

  const gridC = config.theme === 'dark' ? '#1e3a5f' : '#e5e7eb';
  const tickC = config.theme === 'dark' ? '#94a3b8' : '#374151';
  const legendC = config.theme === 'dark' ? '#e2e8f0' : '#111';
  const tooltipBg = config.theme === 'dark' ? '#0f172a' : '#1e2a3a';

  const baseFont = `{ family: "'Segoe UI', Arial, sans-serif", size: 11 }`;

  const commonOpts = `{
    responsive:true, maintainAspectRatio:false,
    interaction:{ mode:'index', intersect:false },
    plugins:{
      legend:{ position:'bottom', labels:{ usePointStyle:false, padding:14, font:${baseFont}, color:'${legendC}' }},
      tooltip:{ backgroundColor:'${tooltipBg}', padding:10, titleFont:${baseFont}, bodyFont:${baseFont} }
    },
    scales:{
      x:{ grid:{ color:'${gridC}' }, ticks:{ font:${baseFont}, color:'${tickC}', maxRotation:45 }, border:{ color:'${tickC}50' }},
      y:{ grid:{ color:'${gridC}' }, ticks:{ font:${baseFont}, color:'${tickC}' }, beginAtZero:true, border:{ color:'${tickC}50' }}
    }
  }`;

  const stackedOpts = `{
    responsive:true, maintainAspectRatio:false,
    interaction:{ mode:'index', intersect:false },
    plugins:{
      legend:{ position:'bottom', labels:{ padding:14, font:${baseFont}, color:'${legendC}' }},
      tooltip:{ backgroundColor:'${tooltipBg}', padding:10, titleFont:${baseFont}, bodyFont:${baseFont} }
    },
    scales:{
      x:{ stacked:true, grid:{ color:'${gridC}' }, ticks:{ font:${baseFont}, color:'${tickC}', maxRotation:45 }, border:{ color:'${tickC}50' }},
      y:{ stacked:true, grid:{ color:'${gridC}' }, ticks:{ font:${baseFont}, color:'${tickC}' }, beginAtZero:true, border:{ color:'${tickC}50' }}
    }
  }`;

  return `
  <script>
    var LABELS = ${safeJson(labels)};
    function initLineChart(id, labels, datasets) {
      var el = document.getElementById(id); if (!el) return;
      new Chart(el, { type:'line', data:{ labels:labels, datasets:datasets }, options:${commonOpts} });
    }
    function initBarChart(id, labels, datasets) {
      var el = document.getElementById(id); if (!el) return;
      new Chart(el, { type:'bar', data:{ labels:labels, datasets:datasets }, options:${stackedOpts} });
    }
    document.addEventListener('DOMContentLoaded', function() {
      ${config.sections.trendChart   ? `initLineChart('chartTrend',    LABELS, ${safeJson(trendDS)});` : ''}
      ${config.sections.passfailChart ? `initBarChart('chartPassFail', LABELS, ${safeJson(pfDS)});`    : ''}
      ${config.sections.perRecipe    ? perRecipeScripts : ''}
    });
  </script>`;
}

// ─── Main Export ──────────────────────────────────────────────────────────────

export function generateHTMLReport(
  config: ReportConfig,
  summary: SummaryStatistics,
  timeseries: TimeseriesStatistics,
): string {
  const hasCharts = config.sections.trendChart || config.sections.passfailChart || config.sections.perRecipe;

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Production Report – ${config.periodLabel}</title>
  ${hasCharts ? '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"><\/script>' : ''}
  <style>${buildStyles()}</style>
</head>
<body class="theme-${config.theme}">
  <div class="container">
    ${buildHeader(config)}
    ${config.sections.kpi ? buildKPI(summary, config) : ''}
    ${config.sections.kpi ? buildRecipeSummaryCards(summary, config) : ''}
    ${(config.sections.trendChart || config.sections.passfailChart) ? buildCombinedChartsHtml() : ''}
    ${config.sections.perRecipe ? buildPerRecipeSections(timeseries, summary, config) : ''}
    <div class="report-footer">
      <span>OCR Datecode Inspection System</span>
      <span>Generated: ${new Date(config.generatedAt).toLocaleString()} &nbsp;|&nbsp; ${config.periodLabel}</span>
    </div>
  </div>
  ${hasCharts ? buildChartScripts(timeseries, config) : ''}
</body>
</html>`;
}
