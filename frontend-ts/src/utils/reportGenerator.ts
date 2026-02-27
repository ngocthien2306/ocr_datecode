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
  '#6366f1', '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b',
  '#10b981', '#ef4444', '#06b6d4', '#f97316', '#84cc16',
];

function passRateColor(rate: number): string {
  if (rate >= 90) return '#10b981';
  if (rate >= 70) return '#f59e0b';
  return '#ef4444';
}

function passRateStatus(rate: number): string {
  if (rate >= 90) return 'Excellent';
  if (rate >= 70) return 'Good';
  return 'Needs Attention';
}

function passRateBadgeClass(rate: number): string {
  if (rate >= 90) return 'badge-success';
  if (rate >= 70) return 'badge-warn';
  return 'badge-danger';
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
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 14px; color: #1f2937; background: #f3f4f6;
    }
    .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
    a { color: inherit; text-decoration: none; }

    /* ── Header ── */
    .report-header {
      background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
      color: white; border-radius: 12px; padding: 28px 32px;
      margin-bottom: 24px; display: flex; justify-content: space-between; align-items: flex-start;
    }
    .report-title { font-size: 26px; font-weight: 700; margin-bottom: 6px; }
    .report-subtitle { font-size: 14px; opacity: 0.85; }
    .report-meta { text-align: right; font-size: 13px; opacity: 0.85; line-height: 1.7; }

    /* ── Section ── */
    .section { background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.07); }
    .section-title {
      font-size: 17px; font-weight: 600; color: #111827;
      margin-bottom: 18px; padding-bottom: 12px;
      border-bottom: 2px solid #e5e7eb; display: flex; align-items: center; gap: 8px;
    }
    .section-title svg { flex-shrink: 0; }

    /* ── KPI Cards ── */
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
    @media (max-width: 768px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
    .kpi-card {
      border-radius: 10px; padding: 20px; border-left: 4px solid;
      background: #fafafa;
    }
    .kpi-card.total { border-color: #6366f1; }
    .kpi-card.pass  { border-color: #10b981; }
    .kpi-card.fail  { border-color: #ef4444; }
    .kpi-card.rate  { border-color: #f59e0b; }
    .kpi-label { font-size: 12px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.05em; color: #6b7280; margin-bottom: 8px; }
    .kpi-value { font-size: 32px; font-weight: 700; color: #111827; line-height: 1; margin-bottom: 6px; }
    .kpi-sub { font-size: 13px; color: #6b7280; }

    /* ── Recipe Summary Cards ── */
    .recipe-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
    .recipe-card {
      border-radius: 10px; padding: 18px; background: #fafafa;
      border: 1px solid #e5e7eb;
    }
    .recipe-card-name { font-weight: 600; font-size: 14px; margin-bottom: 12px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .recipe-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px; }
    .recipe-stat { text-align: center; }
    .recipe-stat-val { font-size: 20px; font-weight: 700; }
    .recipe-stat-lbl { font-size: 11px; color: #9ca3af; text-transform: uppercase; }
    .progress-bar-wrap { background: #e5e7eb; border-radius: 99px; height: 7px; margin-bottom: 8px; overflow: hidden; }
    .progress-bar-fill { height: 100%; border-radius: 99px; }

    /* ── Badges ── */
    .badge { display: inline-block; padding: 3px 9px; border-radius: 99px; font-size: 12px; font-weight: 600; }
    .badge-success { background: #d1fae5; color: #065f46; }
    .badge-warn    { background: #fef3c7; color: #92400e; }
    .badge-danger  { background: #fee2e2; color: #991b1b; }

    /* ── Charts ── */
    .chart-wrap { position: relative; height: 380px; margin-top: 8px; }
    .chart-wrap-sm { position: relative; height: 220px; margin-top: 8px; }

    /* ── Per-Recipe Section ── */
    .recipe-section { background: white; border-radius: 12px; margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.07); overflow: hidden; }
    .recipe-section-header {
      padding: 18px 24px; display: flex; justify-content: space-between; align-items: center;
      border-bottom: 1px solid #e5e7eb;
    }
    .recipe-section-header h3 { font-size: 16px; font-weight: 600; }
    .recipe-section-body { padding: 20px 24px; }
    .recipe-header-stats { display: flex; gap: 20px; align-items: center; }
    .rhs-item { text-align: center; }
    .rhs-val { font-size: 18px; font-weight: 700; }
    .rhs-lbl { font-size: 11px; color: #9ca3af; text-transform: uppercase; }

    /* ── Table ── */
    .data-table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }
    .data-table th {
      background: #f9fafb; text-align: left; padding: 10px 14px;
      font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em;
      color: #6b7280; font-weight: 600; border-bottom: 2px solid #e5e7eb;
    }
    .data-table td { padding: 9px 14px; border-bottom: 1px solid #f3f4f6; }
    .data-table tr:last-child td { border-bottom: none; }
    .data-table tr:hover td { background: #fafafa; }
    .data-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .rate-cell { font-weight: 600; }

    /* ── Footer ── */
    .report-footer {
      text-align: center; padding: 16px; color: #9ca3af; font-size: 12px;
      border-top: 1px solid #e5e7eb; margin-top: 8px;
    }

    /* ── Print ── */
    @media print {
      body { background: white; font-size: 12px; }
      .container { padding: 0; max-width: 100%; }
      .report-header { border-radius: 0; }
      .section, .recipe-section { box-shadow: none; border: 1px solid #e5e7eb; break-inside: avoid; }
      .recipe-section { page-break-inside: avoid; }
    }
  `;
}

// ─── Header HTML ─────────────────────────────────────────────────────────────

function buildHeader(config: ReportConfig): string {
  const startD = new Date(config.startDate);
  const endD = new Date(config.endDate);
  const fmtDate = (d: Date) =>
    `${d.getUTCDate().toString().padStart(2,'0')}/${(d.getUTCMonth()+1).toString().padStart(2,'0')}/${d.getUTCFullYear()}`;
  const genD = new Date(config.generatedAt);
  const genStr = genD.toLocaleString('en-US', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  });

  return `
  <div class="report-header">
    <div>
      <div class="report-title">Production Report</div>
      <div class="report-subtitle">${config.periodLabel} &nbsp;·&nbsp; ${fmtDate(startD)} – ${fmtDate(endD)}</div>
      <div class="report-subtitle" style="margin-top:6px;font-size:13px">
        Breakdown: By ${config.granularity === 'hour' ? 'Hour' : 'Day'}
      </div>
    </div>
    <div class="report-meta">
      <div>Generated: ${genStr}</div>
      <div style="margin-top:4px;font-size:12px;opacity:0.7">OCR Datecode System</div>
    </div>
  </div>`;
}

// ─── KPI Section ─────────────────────────────────────────────────────────────

function buildKPI(summary: SummaryStatistics, config: ReportConfig): string {
  // If specific recipes selected, aggregate only those
  let total = summary.total, pass = summary.pass, fail = summary.fail, rate = summary.pass_rate;
  if (config.selectedRecipeIds.length > 0) {
    const filtered = summary.by_recipe.filter(r => config.selectedRecipeIds.includes(r.recipe_id));
    total = filtered.reduce((s, r) => s + r.total, 0);
    pass  = filtered.reduce((s, r) => s + r.pass, 0);
    fail  = filtered.reduce((s, r) => s + r.fail, 0);
    rate  = total > 0 ? Math.round((pass / total) * 100 * 10) / 10 : 0;
  }

  return `
  <div class="section">
    <div class="section-title">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <rect x="2" y="3" width="20" height="14" rx="2" stroke="currentColor" stroke-width="2"/>
        <path d="M8 21h8M12 17v4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
      </svg>
      Summary
    </div>
    <div class="kpi-grid">
      <div class="kpi-card total">
        <div class="kpi-label">Total Inspections</div>
        <div class="kpi-value">${total.toLocaleString()}</div>
        <div class="kpi-sub">${config.periodLabel}</div>
      </div>
      <div class="kpi-card pass">
        <div class="kpi-label">PASS</div>
        <div class="kpi-value" style="color:#10b981">${pass.toLocaleString()}</div>
        <div class="kpi-sub">${total > 0 ? Math.round(pass/total*100) : 0}% of total</div>
      </div>
      <div class="kpi-card fail">
        <div class="kpi-label">FAIL</div>
        <div class="kpi-value" style="color:#ef4444">${fail.toLocaleString()}</div>
        <div class="kpi-sub">${total > 0 ? Math.round(fail/total*100) : 0}% of total</div>
      </div>
      <div class="kpi-card rate">
        <div class="kpi-label">Pass Rate</div>
        <div class="kpi-value" style="color:${passRateColor(rate)}">${rate}%</div>
        <div class="kpi-sub">
          <span class="badge ${passRateBadgeClass(rate)}">${passRateStatus(rate)}</span>
        </div>
      </div>
    </div>
  </div>`;
}

// ─── Recipe Summary Cards ─────────────────────────────────────────────────────

function buildRecipeSummaryCards(summary: SummaryStatistics, config: ReportConfig): string {
  let recipes = summary.by_recipe;
  if (config.selectedRecipeIds.length > 0) {
    recipes = recipes.filter(r => config.selectedRecipeIds.includes(r.recipe_id));
  }
  if (recipes.length === 0) return '';

  const cards = recipes.map((r, idx) => {
    const color = CHART_COLORS[idx % CHART_COLORS.length];
    const rate = r.pass_rate ?? (r.total > 0 ? Math.round(r.pass / r.total * 1000) / 10 : 0);
    return `
    <div class="recipe-card">
      <div class="recipe-card-name" title="${r.recipe_name}">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};margin-right:6px;"></span>
        ${r.recipe_name}
      </div>
      <div class="recipe-stats">
        <div class="recipe-stat"><div class="recipe-stat-val">${r.total.toLocaleString()}</div><div class="recipe-stat-lbl">Total</div></div>
        <div class="recipe-stat"><div class="recipe-stat-val" style="color:#10b981">${r.pass.toLocaleString()}</div><div class="recipe-stat-lbl">Pass</div></div>
        <div class="recipe-stat"><div class="recipe-stat-val" style="color:#ef4444">${r.fail.toLocaleString()}</div><div class="recipe-stat-lbl">Fail</div></div>
      </div>
      <div class="progress-bar-wrap">
        <div class="progress-bar-fill" style="width:${rate}%;background:${passRateColor(rate)}"></div>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span class="badge ${passRateBadgeClass(rate)}">${passRateStatus(rate)}</span>
        <span style="font-size:13px;font-weight:600;color:${passRateColor(rate)}">${rate}%</span>
      </div>
    </div>`;
  }).join('');

  return `
  <div class="section">
    <div class="section-title">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <path d="M21 16V8C21 6.9 20.1 6 19 6H5C3.9 6 3 6.9 3 8v8c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2z" stroke="currentColor" stroke-width="2"/>
        <path d="M12 6V18M3 12h18" stroke="currentColor" stroke-width="1.5"/>
      </svg>
      Recipe Overview
    </div>
    <div class="recipe-cards">${cards}</div>
  </div>`;
}

// ─── Combined Charts ──────────────────────────────────────────────────────────

function buildCombinedChartsHtml(): string {
  return `
  <div class="section">
    <div class="section-title">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      Production Trend – All Recipes
    </div>
    <div class="chart-wrap"><canvas id="chartTrend"></canvas></div>
  </div>

  <div class="section">
    <div class="section-title">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <line x1="12" y1="20" x2="12" y2="10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        <line x1="18" y1="20" x2="18" y2="4"  stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        <line x1="6"  y1="20" x2="6"  y2="16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
      </svg>
      Pass / Fail Breakdown – All Recipes
    </div>
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
    const color = CHART_COLORS[idx % CHART_COLORS.length];

    // Build table rows
    const recipeStats = summary.by_recipe.find(r => r.recipe_id === recipe.id);
    const totalForRecipe = recipeStats?.total ?? 0;
    const passForRecipe  = recipeStats?.pass ?? 0;
    const failForRecipe  = recipeStats?.fail ?? 0;
    const rateForRecipe  = recipeStats?.pass_rate ?? (totalForRecipe > 0 ? Math.round(passForRecipe/totalForRecipe*1000)/10 : 0);

    const tableRows = timeseries.data.map(pt => {
      const r = pt.by_recipe.find(x => x.recipe_id === recipe.id);
      if (!r || r.total === 0) return '';
      const rowRate = r.total > 0 ? Math.round(r.pass / r.total * 1000) / 10 : 0;
      const lbl = fmtLabel(pt.timestamp, config.granularity);
      return `
        <tr>
          <td>${lbl}</td>
          <td class="num">${r.total.toLocaleString()}</td>
          <td class="num" style="color:#10b981;font-weight:500">${r.pass.toLocaleString()}</td>
          <td class="num" style="color:#ef4444;font-weight:500">${r.fail.toLocaleString()}</td>
          <td class="num rate-cell" style="color:${passRateColor(rowRate)}">${rowRate}%</td>
          <td><span class="badge ${passRateBadgeClass(rowRate)}">${passRateStatus(rowRate)}</span></td>
        </tr>`;
    }).join('');

    return `
  <div class="recipe-section">
    <div class="recipe-section-header" style="border-left:4px solid ${color}">
      <h3>${recipe.name}</h3>
      <div class="recipe-header-stats">
        <div class="rhs-item"><div class="rhs-val">${totalForRecipe.toLocaleString()}</div><div class="rhs-lbl">Total</div></div>
        <div class="rhs-item"><div class="rhs-val" style="color:#10b981">${passForRecipe.toLocaleString()}</div><div class="rhs-lbl">Pass</div></div>
        <div class="rhs-item"><div class="rhs-val" style="color:#ef4444">${failForRecipe.toLocaleString()}</div><div class="rhs-lbl">Fail</div></div>
        <div class="rhs-item">
          <div class="rhs-val" style="color:${passRateColor(rateForRecipe)}">${rateForRecipe}%</div>
          <div class="rhs-lbl">Pass Rate</div>
        </div>
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
        <tbody>${tableRows || '<tr><td colspan="6" style="color:#9ca3af;text-align:center;padding:20px">No data for this period</td></tr>'}</tbody>
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

  const commonOpts = `{
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'top', labels: { usePointStyle: true, padding: 14, font: { size: 12 } } },
      tooltip: { backgroundColor: 'rgba(0,0,0,0.8)', padding: 10 }
    },
    scales: {
      x: { grid: { color: '#e5e7eb' }, ticks: { font: { size: 11 }, color: '#6b7280', maxRotation: 45 } },
      y: { grid: { color: '#e5e7eb' }, ticks: { font: { size: 11 }, color: '#6b7280' }, beginAtZero: true }
    }
  }`;

  const stackedOpts = `{
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'top', labels: { padding: 14, font: { size: 12 } } },
      tooltip: { backgroundColor: 'rgba(0,0,0,0.8)', padding: 10 }
    },
    scales: {
      x: { stacked: true, grid: { color: '#e5e7eb' }, ticks: { font: { size: 11 }, color: '#6b7280', maxRotation: 45 } },
      y: { stacked: true, grid: { color: '#e5e7eb' }, ticks: { font: { size: 11 }, color: '#6b7280' }, beginAtZero: true }
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

  const chartCdnNotice = hasCharts ? `
    <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;padding:10px 16px;margin-bottom:16px;font-size:13px;color:#92400e">
      <strong>Note:</strong> Charts require an internet connection to display (Chart.js CDN). Tables are always visible.
    </div>` : '';

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
    ${chartCdnNotice}
    ${config.sections.kpi ? buildKPI(summary, config) : ''}
    ${config.sections.kpi ? buildRecipeSummaryCards(summary, config) : ''}
    ${(config.sections.trendChart || config.sections.passfailChart) ? buildCombinedChartsHtml() : ''}
    ${config.sections.perRecipe ? buildPerRecipeSections(timeseries, summary, config) : ''}
    <div class="report-footer">
      OCR Datecode &nbsp;·&nbsp; Generated ${new Date(config.generatedAt).toLocaleString()} &nbsp;·&nbsp; ${config.periodLabel}
    </div>
  </div>
  ${hasCharts ? buildChartScripts(timeseries, config) : ''}
</body>
</html>`;
}
