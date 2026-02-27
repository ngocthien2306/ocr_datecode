import type { SummaryStatistics, TimeseriesStatistics } from '@/services/inferenceResults';

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
  sections: ReportSections;
  selectedRecipeIds: string[]; // empty = all
}

const CHART_COLORS = [
  '#2563eb', '#16a34a', '#dc2626', '#d97706',
  '#7c3aed', '#0891b2', '#be185d', '#4d7c0f',
  '#374151', '#b45309',
];

function passRateColor(rate: number): string {
  if (rate >= 90) return '#15803d';
  if (rate >= 70) return '#92400e';
  return '#b91c1c';
}

function passRateStatus(rate: number): string {
  if (rate >= 90) return 'OK';
  if (rate >= 70) return 'WARN';
  return 'FAIL';
}

function passRateStatusClass(rate: number): string {
  if (rate >= 90) return 'st-ok';
  if (rate >= 70) return 'st-warn';
  return 'st-fail';
}

/** Format a UTC ISO timestamp → display label based on granularity */
function fmtLabel(ts: string, granularity: 'hour' | 'day'): string {
  const d = new Date(ts);
  if (granularity === 'hour') {
    const h = d.getUTCHours().toString().padStart(2, '0');
    const m = d.getUTCMinutes().toString().padStart(2, '0');
    return `${h}:${m}`;
  }
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  return `${months[d.getUTCMonth()]} ${d.getUTCDate()}`;
}

/** Prevent injecting </script> which breaks the HTML */
function safeJson(obj: unknown): string {
  return JSON.stringify(obj).replace(/<\/script>/gi, '<\\/script>');
}

// ─── CSS ─────────────────────────────────────────────────────────────────────

function buildStyles(): string {
  return `
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      font-size: 13px; color: #111; background: #fff;
    }
    .container { max-width: 1100px; margin: 0 auto; padding: 20px 28px; }

    /* ── Header ── */
    .report-header {
      background: #1e2a3a;
      color: #fff;
      padding: 20px 28px;
      margin-bottom: 0;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 3px solid #2563eb;
    }
    .report-title { font-size: 20px; font-weight: 700; letter-spacing: 0.02em; margin-bottom: 4px; }
    .report-subtitle { font-size: 12px; color: #94a3b8; line-height: 1.8; }
    .report-meta { text-align: right; font-size: 12px; color: #94a3b8; line-height: 1.8; }

    /* ── Section ── */
    .section { border: 1px solid #d1d5db; margin-bottom: 16px; background: #fff; }
    .section-title {
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.08em; color: #fff; background: #374151;
      padding: 7px 14px;
    }
    .section-body { padding: 16px 14px; }

    /* ── KPI row ── */
    .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid #d1d5db; }
    .kpi-cell {
      padding: 16px 14px; border-right: 1px solid #d1d5db; text-align: center;
    }
    .kpi-cell:last-child { border-right: none; }
    .kpi-label {
      font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.07em; color: #6b7280; margin-bottom: 6px;
    }
    .kpi-value { font-size: 30px; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1; }
    .kpi-sub   { font-size: 12px; color: #6b7280; margin-top: 4px; }

    /* ── Status indicators ── */
    .st-ok   { color: #15803d; font-weight: 700; }
    .st-warn { color: #92400e; font-weight: 700; }
    .st-fail { color: #b91c1c; font-weight: 700; }

    /* ── Charts ── */
    .chart-wrap    { position: relative; height: 340px; padding: 12px 6px 6px; }
    .chart-wrap-sm { position: relative; height: 200px; padding: 8px 6px 4px; }

    /* ── Tables ── */
    .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .data-table th {
      background: #f3f4f6; text-align: left; padding: 8px 12px;
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.05em; color: #374151;
      border: 1px solid #d1d5db;
    }
    .data-table td {
      padding: 7px 12px; border: 1px solid #e5e7eb;
      font-variant-numeric: tabular-nums;
    }
    .data-table tr:nth-child(even) td { background: #f9fafb; }
    .data-table tfoot td {
      background: #f3f4f6; font-weight: 700; border-top: 2px solid #9ca3af;
    }
    .num { text-align: right; }

    /* ── Recipe section ── */
    .recipe-section { border: 1px solid #d1d5db; margin-bottom: 16px; background: #fff; }
    .recipe-section-header {
      background: #f3f4f6; padding: 10px 14px;
      display: flex; justify-content: space-between; align-items: baseline;
      border-bottom: 1px solid #d1d5db;
    }
    .recipe-name {
      font-size: 13px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.04em; color: #111;
    }
    .recipe-kpi-inline {
      font-size: 12px; color: #374151;
      display: flex; gap: 20px;
    }
    .rki-item { white-space: nowrap; }
    .rki-val  { font-weight: 700; }
    .recipe-section-body { padding: 0; }

    /* ── Footer ── */
    .report-footer {
      border-top: 2px solid #1e2a3a; margin-top: 8px;
      padding: 10px 0; font-size: 11px; color: #6b7280;
      display: flex; justify-content: space-between;
    }

    /* ── Print ── */
    @media print {
      body { font-size: 11px; }
      .container { padding: 0; max-width: 100%; }
      .recipe-section, .section { page-break-inside: avoid; }
      .chart-wrap    { height: 260px; }
      .chart-wrap-sm { height: 160px; }
    }
  `;
}

// ─── Header HTML ─────────────────────────────────────────────────────────────

function buildHeader(config: ReportConfig): string {
  const startD = new Date(config.startDate);
  const endD   = new Date(config.endDate);
  const fmtDate = (d: Date) =>
    `${d.getUTCDate().toString().padStart(2,'0')}/${(d.getUTCMonth()+1).toString().padStart(2,'0')}/${d.getUTCFullYear()}`;
  const genD   = new Date(config.generatedAt);
  const genStr = genD.toLocaleString('en-US', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
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

// ─── KPI Section ─────────────────────────────────────────────────────────────

function buildKPI(summary: SummaryStatistics, config: ReportConfig): string {
  let total = summary.total, pass = summary.pass, fail = summary.fail, rate = summary.pass_rate;
  if (config.selectedRecipeIds.length > 0) {
    const filtered = summary.by_recipe.filter(r => config.selectedRecipeIds.includes(r.recipe_id));
    total = filtered.reduce((s, r) => s + r.total, 0);
    pass  = filtered.reduce((s, r) => s + r.pass, 0);
    fail  = filtered.reduce((s, r) => s + r.fail, 0);
    rate  = total > 0 ? Math.round((pass / total) * 1000) / 10 : 0;
  }

  return `
  <div class="section">
    <div class="section-title">INSPECTION SUMMARY</div>
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
        <div class="kpi-value ${passRateStatusClass(rate)}" style="color:${passRateColor(rate)}">${rate}%</div>
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

  const rows = recipes.map(r => {
    const rate = r.pass_rate ?? (r.total > 0 ? Math.round(r.pass / r.total * 1000) / 10 : 0);
    return `
      <tr>
        <td>${r.recipe_name}</td>
        <td class="num">${r.total.toLocaleString()}</td>
        <td class="num st-ok">${r.pass.toLocaleString()}</td>
        <td class="num st-fail">${r.fail.toLocaleString()}</td>
        <td class="num" style="color:${passRateColor(rate)};font-weight:700">${rate}%</td>
        <td class="${passRateStatusClass(rate)}">${passRateStatus(rate)}</td>
      </tr>`;
  }).join('');

  return `
  <div class="section">
    <div class="section-title">RECIPE SUMMARY</div>
    <div class="section-body" style="padding:0">
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
    </div>
  </div>`;
}

// ─── Combined Charts ──────────────────────────────────────────────────────────

function buildCombinedChartsHtml(): string {
  return `
  <div class="section">
    <div class="section-title">PRODUCTION TREND – ALL RECIPES</div>
    <div class="chart-wrap"><canvas id="chartTrend"></canvas></div>
  </div>

  <div class="section">
    <div class="section-title">PASS / FAIL BREAKDOWN – ALL RECIPES</div>
    <div class="chart-wrap"><canvas id="chartPassFail"></canvas></div>
  </div>`;
}

// ─── Per-Recipe Sections ──────────────────────────────────────────────────────

function buildPerRecipeSections(
  timeseries: TimeseriesStatistics,
  summary: SummaryStatistics,
  config: ReportConfig,
): string {
  // Collect unique recipes from timeseries in order
  const recipeMap = new Map<string, string>();
  timeseries.data.forEach(pt => {
    pt.by_recipe.forEach(r => {
      if (!recipeMap.has(r.recipe_id)) recipeMap.set(r.recipe_id, r.recipe_name);
    });
  });

  let recipes = Array.from(recipeMap.entries()).map(([id, name]) => ({ id, name }));
  if (config.selectedRecipeIds.length > 0) {
    recipes = recipes.filter(r => config.selectedRecipeIds.includes(r.id));
  }

  if (recipes.length === 0) return '<div class="section"><p style="color:#6b7280">No recipe data available.</p></div>';

  const granLabel = config.granularity === 'hour' ? 'Hour' : 'Date';

  return recipes.map((recipe, idx) => {
    const recipeStats    = summary.by_recipe.find(r => r.recipe_id === recipe.id);
    const totalForRecipe = recipeStats?.total ?? 0;
    const passForRecipe  = recipeStats?.pass  ?? 0;
    const failForRecipe  = recipeStats?.fail  ?? 0;
    const rateForRecipe  = recipeStats?.pass_rate
      ?? (totalForRecipe > 0 ? Math.round(passForRecipe / totalForRecipe * 1000) / 10 : 0);

    // Table rows + accumulate totals
    let sumTotal = 0, sumPass = 0, sumFail = 0;
    const tableRows = timeseries.data.map(pt => {
      const r = pt.by_recipe.find(x => x.recipe_id === recipe.id);
      if (!r || r.total === 0) return '';
      sumTotal += r.total; sumPass += r.pass; sumFail += r.fail;
      const rowRate = r.total > 0 ? Math.round(r.pass / r.total * 1000) / 10 : 0;
      return `
        <tr>
          <td>${fmtLabel(pt.timestamp, config.granularity)}</td>
          <td class="num">${r.total.toLocaleString()}</td>
          <td class="num st-ok">${r.pass.toLocaleString()}</td>
          <td class="num st-fail">${r.fail.toLocaleString()}</td>
          <td class="num" style="color:${passRateColor(rowRate)};font-weight:700">${rowRate}%</td>
          <td class="${passRateStatusClass(rowRate)}">${passRateStatus(rowRate)}</td>
        </tr>`;
    }).join('');

    const sumRate = sumTotal > 0 ? Math.round(sumPass / sumTotal * 1000) / 10 : 0;

    return `
  <div class="recipe-section">
    <div class="recipe-section-header">
      <span class="recipe-name">${recipe.name}</span>
      <div class="recipe-kpi-inline">
        <span class="rki-item">Total: <span class="rki-val">${totalForRecipe.toLocaleString()}</span></span>
        <span class="rki-item st-ok">Pass: <span class="rki-val">${passForRecipe.toLocaleString()}</span></span>
        <span class="rki-item st-fail">Fail: <span class="rki-val">${failForRecipe.toLocaleString()}</span></span>
        <span class="rki-item" style="color:${passRateColor(rateForRecipe)}">Rate: <span class="rki-val">${rateForRecipe}%</span></span>
      </div>
    </div>
    <div class="recipe-section-body">
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
        <tbody>${tableRows || '<tr><td colspan="6" style="color:#9ca3af;text-align:center;padding:16px">No data for this period</td></tr>'}</tbody>
        ${sumTotal > 0 ? `
        <tfoot>
          <tr>
            <td>TOTAL</td>
            <td class="num">${sumTotal.toLocaleString()}</td>
            <td class="num">${sumPass.toLocaleString()}</td>
            <td class="num">${sumFail.toLocaleString()}</td>
            <td class="num" style="color:${passRateColor(sumRate)}">${sumRate}%</td>
            <td class="${passRateStatusClass(sumRate)}">${passRateStatus(sumRate)}</td>
          </tr>
        </tfoot>` : ''}
      </table>
    </div>
  </div>`;
  }).join('');
}

// ─── Chart Init Scripts ───────────────────────────────────────────────────────

function buildChartScripts(
  timeseries: TimeseriesStatistics,
  config: ReportConfig,
): string {
  // Collect unique recipes (filtered)
  const recipeMap = new Map<string, string>();
  timeseries.data.forEach(pt => {
    pt.by_recipe.forEach(r => {
      if (!recipeMap.has(r.recipe_id)) recipeMap.set(r.recipe_id, r.recipe_name);
    });
  });
  let recipes = Array.from(recipeMap.entries()).map(([id, name]) => ({ id, name }));
  if (config.selectedRecipeIds.length > 0) {
    recipes = recipes.filter(r => config.selectedRecipeIds.includes(r.id));
  }

  const labels = timeseries.data.map(pt => fmtLabel(pt.timestamp, config.granularity));

  // ── Trend chart datasets (one line per recipe, showing total) ──
  const trendDatasets = recipes.map((recipe, idx) => {
    const color = CHART_COLORS[idx % CHART_COLORS.length];
    const data = timeseries.data.map(pt => {
      const r = pt.by_recipe.find(x => x.recipe_id === recipe.id);
      return r ? r.total : 0;
    });
    return { label: recipe.name, data, borderColor: color, backgroundColor: color + '20',
             borderWidth: 2, pointRadius: 4, tension: 0.3, fill: false };
  });

  // ── Pass/Fail stacked bar datasets ──
  const pfDatasets: object[] = [];
  recipes.forEach((recipe, idx) => {
    const color = CHART_COLORS[idx % CHART_COLORS.length];
    const passD = timeseries.data.map(pt => { const r = pt.by_recipe.find(x => x.recipe_id === recipe.id); return r ? r.pass : 0; });
    const failD = timeseries.data.map(pt => { const r = pt.by_recipe.find(x => x.recipe_id === recipe.id); return r ? r.fail : 0; });
    pfDatasets.push({ label: `${recipe.name} (Pass)`, data: passD, backgroundColor: color + 'CC', stack: `r${idx}` });
    pfDatasets.push({ label: `${recipe.name} (Fail)`, data: failD, backgroundColor: color + '55', stack: `r${idx}` });
  });

  // ── Per-recipe chart datasets ──
  const perRecipeScripts = recipes.map((recipe, idx) => {
    const color = CHART_COLORS[idx % CHART_COLORS.length];
    const passD = timeseries.data.map(pt => { const r = pt.by_recipe.find(x => x.recipe_id === recipe.id); return r ? r.pass : 0; });
    const failD = timeseries.data.map(pt => { const r = pt.by_recipe.find(x => x.recipe_id === recipe.id); return r ? r.fail : 0; });
    return `
    initBarChart('chartRecipe_${idx}', ${safeJson(labels)}, [
      { label: 'Pass', data: ${safeJson(passD)}, backgroundColor: '${color}CC', stack: 's0' },
      { label: 'Fail', data: ${safeJson(failD)}, backgroundColor: '${color}55', stack: 's0' }
    ]);`;
  }).join('\n');

  const baseFont = `{ family: "Segoe UI, Arial, sans-serif", size: 11 }`;
  const gridColor = `'#d1d5db'`;
  const tickColor = `'#374151'`;

  const commonOpts = `{
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'bottom', labels: { usePointStyle: false, padding: 16, font: ${baseFont}, color: '#111' } },
      tooltip: { backgroundColor: '#1e2a3a', padding: 10, titleFont: ${baseFont}, bodyFont: ${baseFont} }
    },
    scales: {
      x: { grid: { color: ${gridColor} }, ticks: { font: ${baseFont}, color: ${tickColor}, maxRotation: 45 }, border: { color: '#9ca3af' } },
      y: { grid: { color: ${gridColor} }, ticks: { font: ${baseFont}, color: ${tickColor} }, beginAtZero: true, border: { color: '#9ca3af' } }
    }
  }`;

  const stackedOpts = `{
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'bottom', labels: { padding: 16, font: ${baseFont}, color: '#111' } },
      tooltip: { backgroundColor: '#1e2a3a', padding: 10, titleFont: ${baseFont}, bodyFont: ${baseFont} }
    },
    scales: {
      x: { stacked: true, grid: { color: ${gridColor} }, ticks: { font: ${baseFont}, color: ${tickColor}, maxRotation: 45 }, border: { color: '#9ca3af' } },
      y: { stacked: true, grid: { color: ${gridColor} }, ticks: { font: ${baseFont}, color: ${tickColor} }, beginAtZero: true, border: { color: '#9ca3af' } }
    }
  }`;

  return `
  <script>
    var chartInstances = {};
    function initLineChart(id, labels, datasets) {
      var el = document.getElementById(id);
      if (!el) return;
      chartInstances[id] = new Chart(el, { type: 'line', data: { labels: labels, datasets: datasets }, options: ${commonOpts} });
    }
    function initBarChart(id, labels, datasets) {
      var el = document.getElementById(id);
      if (!el) return;
      chartInstances[id] = new Chart(el, { type: 'bar', data: { labels: labels, datasets: datasets }, options: ${stackedOpts} });
    }

    document.addEventListener('DOMContentLoaded', function() {
      ${config.sections.trendChart ? `initLineChart('chartTrend', ${safeJson(labels)}, ${safeJson(trendDatasets)});` : ''}
      ${config.sections.passfailChart ? `initBarChart('chartPassFail', ${safeJson(labels)}, ${safeJson(pfDatasets)});` : ''}
      ${config.sections.perRecipe ? perRecipeScripts : ''}
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
<body>
  <div class="container">
    ${buildHeader(config)}
    ${config.sections.kpi ? buildKPI(summary, config) : ''}
    ${config.sections.kpi ? buildRecipeSummaryCards(summary, config) : ''}
    ${(config.sections.trendChart || config.sections.passfailChart) ? buildCombinedChartsHtml() : ''}
    ${config.sections.perRecipe ? buildPerRecipeSections(timeseries, summary, config) : ''}
    <div class="report-footer">
      <span>OCR Datecode Inspection System</span>
      <span>Generated: ${new Date(config.generatedAt).toLocaleString()} &nbsp;|&nbsp; Period: ${config.periodLabel}</span>
    </div>
  </div>
  ${hasCharts ? buildChartScripts(timeseries, config) : ''}
</body>
</html>`;
}
