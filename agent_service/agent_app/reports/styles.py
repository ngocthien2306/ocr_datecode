"""
CSS của báo cáo sản xuất — trích nguyên văn từ
`frontend-ts/src/utils/reportGenerator.ts` (hàm `buildStyles`).

Copy đúng từng ký tự chứ không viết lại: báo cáo do agent xuất phải trông
giống hệt báo cáo do panel Historical xuất, nếu không thì cùng một kỳ sản xuất
lại có hai bản báo cáo khác nhau và người đọc không biết tin bản nào.

Ba theme (industrial / dark / executive) đều nằm trong chuỗi này, chọn bằng
class trên thẻ <body>.
"""

REPORT_CSS = r"""
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
  """
