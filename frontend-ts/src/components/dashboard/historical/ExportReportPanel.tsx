import React, { useState, useEffect } from 'react';
import { recipesAPI } from '@/services/recipes';
import { inferenceResultsAPI } from '@/services/inferenceResults';
import type { Recipe } from '@/types';
import { generateHTMLReport, type ReportSections } from '@/utils/reportGenerator';
import { getTimezoneOffset } from '@/utils/timezone';

interface ExportReportPanelProps {
  isOpen: boolean;
  onClose: () => void;
  /** Inherit the date range currently selected on the Historical page */
  defaultDateRange: string;
}

type GranularityOption = 'auto' | 'hour' | 'day';

const PERIOD_OPTIONS = [
  { value: 'today',     label: 'Today' },
  { value: 'yesterday', label: 'Yesterday' },
  { value: 'thisweek',  label: 'This Week' },
  { value: 'lastweek',  label: 'Last Week' },
  { value: '7days',     label: 'Last 7 Days' },
  { value: 'thismonth', label: 'This Month' },
  { value: 'lastmonth', label: 'Last Month' },
  { value: '30days',    label: 'Last 30 Days' },
] as const;

const PERIOD_LABELS: Record<string, string> = Object.fromEntries(
  PERIOD_OPTIONS.map(o => [o.value, o.label])
);

/** Compute UTC date range with timezone awareness */
function computeDateRange(period: string): { start_date: string; end_date: string } {
  const tzOffset = getTimezoneOffset(); // minutes
  const nowUTC   = Date.now();
  // current time in target timezone
  const localMs  = nowUTC + tzOffset * 60_000;
  const local    = new Date(localMs);

  const yr  = local.getUTCFullYear();
  const mo  = local.getUTCMonth();
  const day = local.getUTCDate();
  const dow = local.getUTCDay(); // 0=Sun

  // helper: midnight local → UTC ISO
  const toUTC = (y: number, m: number, d: number, h = 0, min = 0, sec = 0) =>
    new Date(Date.UTC(y, m, d, h, min, sec) - tzOffset * 60_000).toISOString();
  const eod = (y: number, m: number, d: number) => toUTC(y, m, d, 23, 59, 59);

  switch (period) {
    case 'today':
      return { start_date: toUTC(yr, mo, day), end_date: eod(yr, mo, day) };

    case 'yesterday': {
      const yd = new Date(Date.UTC(yr, mo, day - 1));
      return {
        start_date: toUTC(yd.getUTCFullYear(), yd.getUTCMonth(), yd.getUTCDate()),
        end_date:   eod(yd.getUTCFullYear(), yd.getUTCMonth(), yd.getUTCDate()),
      };
    }

    case 'thisweek': {
      // Mon = start of week
      const daysToMon = dow === 0 ? 6 : dow - 1;
      const mon = new Date(Date.UTC(yr, mo, day - daysToMon));
      return {
        start_date: toUTC(mon.getUTCFullYear(), mon.getUTCMonth(), mon.getUTCDate()),
        end_date:   eod(yr, mo, day),
      };
    }

    case 'lastweek': {
      const daysToMon = dow === 0 ? 6 : dow - 1;
      const thisMon = new Date(Date.UTC(yr, mo, day - daysToMon));
      const lastMon = new Date(thisMon.getTime() - 7 * 86_400_000);
      const lastSun = new Date(thisMon.getTime() - 86_400_000);
      return {
        start_date: toUTC(lastMon.getUTCFullYear(), lastMon.getUTCMonth(), lastMon.getUTCDate()),
        end_date:   eod(lastSun.getUTCFullYear(), lastSun.getUTCMonth(), lastSun.getUTCDate()),
      };
    }

    case '7days': {
      const s = new Date(localMs - 7 * 86_400_000);
      return { start_date: toUTC(s.getUTCFullYear(), s.getUTCMonth(), s.getUTCDate()), end_date: eod(yr, mo, day) };
    }

    case 'thismonth':
      return { start_date: toUTC(yr, mo, 1), end_date: eod(yr, mo, day) };

    case 'lastmonth': {
      const lastMoStart = new Date(Date.UTC(yr, mo - 1, 1));
      const lastMoEnd   = new Date(Date.UTC(yr, mo, 0));
      return {
        start_date: toUTC(lastMoStart.getUTCFullYear(), lastMoStart.getUTCMonth(), 1),
        end_date:   eod(lastMoEnd.getUTCFullYear(), lastMoEnd.getUTCMonth(), lastMoEnd.getUTCDate()),
      };
    }

    case '30days': {
      const s = new Date(localMs - 30 * 86_400_000);
      return { start_date: toUTC(s.getUTCFullYear(), s.getUTCMonth(), s.getUTCDate()), end_date: eod(yr, mo, day) };
    }

    default: {
      const s = new Date(localMs - 7 * 86_400_000);
      return { start_date: toUTC(s.getUTCFullYear(), s.getUTCMonth(), s.getUTCDate()), end_date: eod(yr, mo, day) };
    }
  }
}

function autoGranularity(period: string): 'hour' | 'day' {
  return ['today', 'yesterday'].includes(period) ? 'hour' : 'day';
}

const ExportReportPanel: React.FC<ExportReportPanelProps> = ({ isOpen, onClose, defaultDateRange }) => {
  const [period, setPeriod]         = useState(defaultDateRange || 'today');
  const [granularity, setGranularity] = useState<GranularityOption>('auto');
  const [recipes, setRecipes]       = useState<Recipe[]>([]);
  const [filterByRecipe, setFilterByRecipe] = useState(false);
  const [selectedRecipes, setSelectedRecipes] = useState<string[]>([]);
  const [sections, setSections]     = useState<ReportSections>({
    kpi: true, trendChart: true, passfailChart: true, perRecipe: true,
  });
  const [generating, setGenerating] = useState(false);
  const [error, setError]           = useState<string | null>(null);

  // Sync period when panel opens
  useEffect(() => {
    if (isOpen) {
      const mapped = PERIOD_OPTIONS.find(o => o.value === defaultDateRange)
        ? defaultDateRange
        : 'today';
      setPeriod(mapped);
    }
  }, [isOpen, defaultDateRange]);

  useEffect(() => {
    recipesAPI.getAllRecipes(0, 100, true)
      .then(setRecipes)
      .catch(e => console.error('Failed to fetch recipes', e));
  }, []);

  const actualGranularity = granularity === 'auto' ? autoGranularity(period) : granularity;

  const toggleSection = (key: keyof ReportSections) =>
    setSections(prev => ({ ...prev, [key]: !prev[key] }));

  const toggleRecipe = (id: string) =>
    setSelectedRecipes(prev =>
      prev.includes(id) ? prev.filter(r => r !== id) : [...prev, id]
    );

  const handleGenerate = async () => {
    setGenerating(true);
    setError(null);
    try {
      const { start_date, end_date } = computeDateRange(period);
      const recipeIds = filterByRecipe && selectedRecipes.length > 0
        ? selectedRecipes
        : [];
      const recipeIdsParam = recipeIds.length > 0 ? recipeIds.join(',') : undefined;

      const [summary, timeseries] = await Promise.all([
        inferenceResultsAPI.getSummaryStatistics({ start_date, end_date }),
        inferenceResultsAPI.getTimeseriesStatistics({
          start_date,
          end_date,
          granularity: actualGranularity,
          recipe_ids: recipeIdsParam,
        }),
      ]);

      const html = generateHTMLReport(
        {
          periodLabel:      PERIOD_LABELS[period] ?? period,
          startDate:        start_date,
          endDate:          end_date,
          granularity:      actualGranularity,
          generatedAt:      new Date().toISOString(),
          sections,
          selectedRecipeIds: recipeIds,
        },
        summary,
        timeseries,
      );

      const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
      const url  = URL.createObjectURL(blob);
      const newTab = window.open(url, '_blank');
      if (!newTab) {
        setError('Popup was blocked. Please allow popups for this site and try again.');
        URL.revokeObjectURL(url);
      } else {
        // Clean up blob URL after tab has loaded
        newTab.addEventListener('load', () => URL.revokeObjectURL(url), { once: true });
      }
    } catch (err) {
      console.error('Export report error:', err);
      setError('Failed to generate report. Please check your connection and try again.');
    } finally {
      setGenerating(false);
    }
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Overlay */}
      <div className="export-overlay" onClick={onClose} />

      {/* Side Panel */}
      <div className="export-panel">
        {/* ── Header ── */}
        <div className="export-panel-head">
          <span className="export-panel-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" style={{ marginRight: 8, flexShrink: 0 }}>
              <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="2"/>
              <line x1="16" y1="13" x2="8" y2="13" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <line x1="16" y1="17" x2="8" y2="17" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            Export Report
          </span>
          <button className="export-panel-close" onClick={onClose} title="Close">✕</button>
        </div>

        {/* ── Body ── */}
        <div className="export-panel-body">
          {error && <div className="export-error">{error}</div>}

          {/* 1. Time Period */}
          <div className="ep-section">
            <div className="ep-section-label">
              <span className="ep-step">1</span> Time Period
            </div>
            <select
              className="ep-select"
              value={period}
              onChange={e => setPeriod(e.target.value)}
            >
              {PERIOD_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          {/* 2. Chart Breakdown */}
          <div className="ep-section">
            <div className="ep-section-label">
              <span className="ep-step">2</span> Chart Breakdown
            </div>
            <div className="ep-radio-group">
              {([
                { val: 'auto', label: 'Auto', hint: `→ by ${actualGranularity}` },
                { val: 'hour', label: 'Force Hour', hint: '' },
                { val: 'day',  label: 'Force Day',  hint: '' },
              ] as { val: GranularityOption; label: string; hint: string }[]).map(opt => (
                <label key={opt.val} className="ep-radio-label">
                  <input
                    type="radio"
                    name="granularity"
                    value={opt.val}
                    checked={granularity === opt.val}
                    onChange={() => setGranularity(opt.val)}
                  />
                  <span>{opt.label}</span>
                  {opt.hint && <span className="ep-hint">{opt.hint}</span>}
                </label>
              ))}
            </div>
          </div>

          {/* 3. Recipes */}
          <div className="ep-section">
            <div className="ep-section-label">
              <span className="ep-step">3</span> Recipes
            </div>
            <label className="ep-toggle-label">
              <input
                type="checkbox"
                checked={filterByRecipe}
                onChange={e => {
                  setFilterByRecipe(e.target.checked);
                  if (!e.target.checked) setSelectedRecipes([]);
                }}
              />
              Filter by specific recipes
            </label>

            {filterByRecipe && (
              <>
                <div className="ep-recipe-actions">
                  <button className="ep-btn-sm" onClick={() => setSelectedRecipes(recipes.map(r => r.id))}>
                    Select All
                  </button>
                  <button className="ep-btn-sm" onClick={() => setSelectedRecipes([])}>
                    Clear
                  </button>
                </div>
                <div className="ep-recipe-list">
                  {recipes.map(recipe => (
                    <label key={recipe.id} className="ep-recipe-item">
                      <input
                        type="checkbox"
                        checked={selectedRecipes.includes(recipe.id)}
                        onChange={() => toggleRecipe(recipe.id)}
                      />
                      <span className="ep-recipe-name" title={recipe.name}>{recipe.name}</span>
                    </label>
                  ))}
                  {recipes.length === 0 && (
                    <div className="ep-empty">No recipes found</div>
                  )}
                </div>
                <div className="ep-recipe-count">
                  {selectedRecipes.length === 0
                    ? 'All recipes will be included'
                    : `${selectedRecipes.length} recipe(s) selected`}
                </div>
              </>
            )}
            {!filterByRecipe && (
              <div className="ep-hint-block">All recipes will be included</div>
            )}
          </div>

          {/* 4. Report Sections */}
          <div className="ep-section">
            <div className="ep-section-label">
              <span className="ep-step">4</span> Include Sections
            </div>
            <div className="ep-check-list">
              {([
                { key: 'kpi',          label: 'KPI Summary + Recipe Cards' },
                { key: 'trendChart',   label: 'Combined Trend Chart' },
                { key: 'passfailChart',label: 'Pass/Fail Breakdown Chart' },
                { key: 'perRecipe',    label: 'Per-Recipe Detail (chart + table)' },
              ] as { key: keyof ReportSections; label: string }[]).map(({ key, label }) => (
                <label key={key} className="ep-check-item">
                  <input
                    type="checkbox"
                    checked={sections[key]}
                    onChange={() => toggleSection(key)}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
            <p className="ep-hint-block" style={{ marginTop: 8 }}>
              Charts require internet. Tables are always visible.
            </p>
          </div>
        </div>

        {/* ── Footer ── */}
        <div className="export-panel-foot">
          <button
            className="ep-generate-btn"
            onClick={handleGenerate}
            disabled={generating}
          >
            {generating ? (
              <>
                <span className="ep-spinner" />
                Generating…
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <polyline points="7,10 12,15 17,10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <line x1="12" y1="15" x2="12" y2="3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Generate Report
              </>
            )}
          </button>
          <p className="ep-foot-note">Opens in a new browser tab</p>
        </div>
      </div>
    </>
  );
};

export default ExportReportPanel;
