/* ═══════════════════════════════════════════════════════════════════════════
   Các khối hiển thị: thẻ máy, thống kê, ảnh lỗi, nhân sự, nhật ký.
   ═══════════════════════════════════════════════════════════════════════════ */

import { store, has, num, fmt, NA, level, uptime, esc, clock,
         coverageHTML, deltaPts, causeLabel } from './core.js';
import { jetsonSVG } from './jetson-art.js';

const TONE = { ok: 'ok', warn: 'warn', agent_down: 'mute',
               unreachable: 'bad', offline: 'bad' };

/* ── Thẻ máy ─────────────────────────────────────────────────────────────── */

export function machineCard(m) {
  const t = store.t;
  const tone = TONE[m.state] || 'bad';
  const x = m.metrics || {};
  const p = m.production || {};
  const rec = (m.recipes || [])[0];
  const jetsonKind = /agx\s+orin/i.test(m.model || '') ? 'agx'
    : /\bsuper\b/i.test(m.model || '') ? 'super' : 'nano';

  const cls = tone === 'warn' ? 'mcard s-warn'
            : tone === 'bad' ? 'mcard s-bad'
            : tone === 'mute' ? 'mcard s-mute' : 'mcard';

  // Phần trăm LUÔN kèm số tuyệt đối: "RAM 87%" trên máy 8 GB và trên máy 32 GB
  // là hai tình huống khác hẳn, mà phần trăm thì giống hệt.
  const hw = (label, val, sub, lv) => `
    <div><div class="eyebrow">${label}</div>
      <div class="v ${lv || ''}">${val ?? NA}</div>
      ${sub ? `<div class="sub">${sub}</div>` : ''}</div>`;
  const pie = (label, percent, sub, lv) => {
    if (!has(percent)) return hw(label, null, sub, lv);
    const value = Math.max(0, Math.min(100, Math.round(percent)));
    const detail = sub ? `${label}: ${value}% · ${sub}` : `${label}: ${value}%`;
    return `<div class="hw-pie">
      <div class="eyebrow">${label}</div>
      <div class="donut ${lv || ''}" style="--pct:${value}" role="img" aria-label="${esc(detail)}">
        <span>${value}<small>%</small></span>
      </div>
      ${sub ? `<div class="sub">${sub}</div>` : ''}
    </div>`;
  };

  const target = has(p.target) && p.target > 0
    ? Math.round((p.total_products || 0) * 100 / p.target) : null;

  const body = m.metrics || m.production ? `
    <div class="mrow">
      <div>
        <div class="big ${has(p.pass_rate) ? level(100 - p.pass_rate, 15, 25) : ''}">
          ${has(p.pass_rate) ? num(p.pass_rate, 1) : '—'}<small>%</small></div>
        <div class="eyebrow" style="margin-top:2px">${t.passThisShift}</div>
        ${deltaPts(p.delta_pts, p.baseline_count)}
      </div>
      <div class="side">
        <div style="font-size:15px;font-weight:600">
          ${fmt(p.total_products)}${has(p.target) ? ` <span class="muted" style="font-weight:400">/ ${fmt(p.target)} ${t.products}</span>` : ` <span class="muted" style="font-weight:400">${t.products}</span>`}</div>
        ${has(x.uptime_seconds) ? `<div class="card-uptime"><span class="eyebrow">${t.uptime}</span><b>${uptime(x.uptime_seconds)}</b></div>` : ''}
        ${has(target) ? `<div class="bar"><i class="${level(100 - target, 20, 40)}"
            style="width:${Math.min(target, 100)}%"></i></div>
          <div class="sub" style="font-size:10.5px;color:var(--faint);margin-top:4px">
            ${target}% ${t.shiftTarget}</div>` : ''}
      </div>
    </div>
    <div class="hw">
      ${hw('CPU', has(x.cpu_temp) ? `${num(x.cpu_temp, 0)}°C` : null,
           has(x.cpu_percent) ? `${t.load} ${num(x.cpu_percent, 0)}%` : '',
           level(x.cpu_temp))}
      ${hw('GPU', has(x.gpu_temp) ? `${num(x.gpu_temp, 0)}°C` : null,
           has(x.gpu_percent) ? `${t.load} ${num(x.gpu_percent, 0)}%` : '',
           level(x.gpu_temp))}
      ${pie('RAM', x.ram_percent,
            has(x.ram_used_gb) ? `${num(x.ram_used_gb, 1)}/${num(x.ram_total_gb, 0)} GB` : '',
            level(x.ram_percent))}
      ${pie(t.disk, x.disk_percent,
            has(x.disk_free_gb) ? `${num(x.disk_free_gb, 0)} GB ${t.free}` : '',
            level(x.disk_percent))}
    </div>` : `
    <div class="mrow"><div class="muted" style="padding:18px 0">${t.noMetrics}</div></div>`;

  return `<article class="${cls}" data-node="${esc(m.node_id)}">
    <div class="jetson-visual jetson-${jetsonKind} is-${tone}" aria-hidden="true">
      ${jetsonSVG(jetsonKind, tone)}
    </div>
    <div class="mcard-content">
      <div class="mcard-top">
        <span class="eyebrow">${esc(m.line || '')} · ${esc(m.model || m.hostname || '')}</span>
        <span class="state ${tone}">${esc(t.state[m.state] || m.state)}</span>
      </div>
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <div class="mname">${esc(m.name)}</div>
        ${rec ? `<span class="recipe-chip">${esc(rec.name)}</span>` : ''}
      </div>
      ${body}
    </div>
  </article>`;
}

export const skeletonCard = () =>
  `<article class="mcard skel"><div class="ln" style="width:40%"></div>
   <div class="ln" style="width:65%;height:24px"></div>
   <div class="ln" style="width:50%;height:34px"></div>
   <div class="ln" style="width:90%"></div><div class="ln" style="width:80%"></div>
   </article>`;

/* ── Thống kê toàn nhà máy ───────────────────────────────────────────────── */

export function statsHTML(prod, view, inventory = []) {
  const t = store.t;
  const tot = prod?.fleet_total || {};
  const reports = prod?.machines || [];
  const byMachine = new Map(reports.map(report => [report.machine, report]));
  const rows = (inventory.length ? inventory : reports.map(report => ({
    name: report.machine, line: report.line, state: report.state,
  }))).map(machine => {
    const report = byMachine.get(machine.name);
    return {
      ...machine,
      line: report?.line || machine.line,
      state: report?.state || machine.state || 'offline',
      production: report?.production || null,
      recipes: report?.recipes || null,
      error: report?.error || null,
    };
  });
  const known = new Set(rows.map(row => row.name));
  reports.forEach(report => {
    if (!known.has(report.machine)) rows.push({
      name: report.machine, line: report.line, state: report.state,
      production: report.production, recipes: report.recipes, error: report.error,
    });
  });
  const rate = value => has(value) ? `${num(value, 1)}%` : NA;
  const rateFail = production => {
    if (!production) return null;
    if (has(production.fail) && has(production.total_products) && production.total_products > 0) {
      return production.fail * 100 / production.total_products;
    }
    return has(production.pass_rate) ? 100 - production.pass_rate : null;
  };

  const table = `<div class="tbl-wrap"><table>
    <thead><tr><th>&nbsp;</th><th>${t.output}</th><th>${t.perDay}</th>
      <th>${t.passRate}</th><th>${t.failRate}</th><th>${t.recipe}</th></tr></thead><tbody>
    ${rows.map(r => {
      const p = r.production;
      if (!p) return `<tr><td class="name">${esc(r.name)}</td>
        <td colspan="5" class="muted">${esc(r.error || t.noMetrics)}</td></tr>`;
      const rec = (r.recipes || [])[0];
      return `<tr><td class="name">${esc(r.name)}
          <div class="sub" style="font-weight:400;color:var(--faint);font-size:11px">${esc(r.line || '')}</div></td>
        <td>${fmt(p.total_products)}</td><td>${fmt(p.per_day, 1)}</td>
        <td>${rate(p.pass_rate)}</td><td>${rate(rateFail(p))}</td>
        <td class="muted">${esc(rec ? rec.name : '—')}</td></tr>`;
    }).join('')}
    <tr class="total"><td>${t.fleetTotal}</td><td>${fmt(tot.products)}</td>
      <td>—</td><td>${rate(tot.pass_rate)}</td><td>${rate(rateFail({ fail: tot.fail, total_products: tot.products, pass_rate: tot.pass_rate }))}</td><td class="muted">—</td></tr>
    </tbody></table></div>`;

  const reporting = rows.filter(row => row.production).length;
  const chart = `<div class="quality-chart">
    <div class="quality-chart-meta">
      <div class="quality-legend"><span class="pass"><i></i>${t.pass}</span><span class="fail"><i></i>${t.fail}</span></div>
      <span class="hint">${t.qualityHint}</span>
    </div>
    <div class="quality-grid">
      ${rows.map(r => {
        const p = r.production;
        const pass = p && has(p.pass_rate) ? Math.max(0, Math.min(100, p.pass_rate)) : null;
        const fail = rateFail(p);
        const recipe = (r.recipes || [])[0];
        const state = t.state[r.state] || r.state || t.state.offline;
        const tooltip = p ? `<div class="quality-tooltip" role="tooltip">
          <strong>${esc(r.name)}</strong><span>${esc(r.line || '')}</span>
          <dl><div><dt>${t.pass}</dt><dd>${rate(pass)} · ${fmt(p.pass)}</dd></div>
          <div><dt>${t.fail}</dt><dd>${rate(fail)} · ${fmt(p.fail)}</dd></div>
          <div><dt>${t.output}</dt><dd>${fmt(p.total_products)}</dd></div>
          <div><dt>${t.recipe}</dt><dd>${esc(recipe?.name || '—')}</dd></div></dl>
        </div>` : `<div class="quality-tooltip" role="tooltip"><strong>${esc(r.name)}</strong>
          <span>${esc(state)}</span><p>${esc(r.error || t.noMetrics)}</p></div>`;
        return `<button type="button" class="quality-machine ${p ? '' : 'is-offline'}" aria-label="${esc(p ? `${r.name}: ${rate(pass)} ${t.pass}, ${rate(fail)} ${t.fail}` : `${r.name}: ${state}`)}">
          <div class="quality-machine-head"><span>${esc(r.name)}</span><em>${esc(state)}</em></div>
          ${p ? `<div class="quality-meter" aria-hidden="true"><i class="quality-pass" style="width:${pass}%"></i><i class="quality-fail" style="width:${Math.max(0, fail)}%"></i></div>
            <div class="quality-rates"><strong>${rate(pass)} <small>${t.pass}</small></strong><span>${rate(fail)} ${t.fail}</span></div>
            <div class="quality-output">${fmt(p.total_products)} ${t.output.toLowerCase()}</div>`
            : `<div class="quality-empty">—<span>${t.noMetrics}</span></div>`}
          ${tooltip}
        </button>`;
      }).join('')}
    </div>
    <div class="quality-footer"><span>${t.qualityCoverage(reporting, rows.length)}</span><span>${t.noRank}</span></div>
  </div>`;

  return view === 'table' ? table : chart;
}

/** Màu gán theo TÊN máy, không theo thứ tự trong danh sách: gán theo thứ tự thì
 *  bỏ một máy khỏi bộ lọc là mọi máy còn lại đổi màu, và hai biểu đồ cạnh nhau
 *  không đọc chéo được nữa. */
const PALETTE = ['#5980a6', '#b3261e', '#2f7d4f', '#9a6a00', '#6b4fa0',
                 '#0e7490', '#a3457d', '#5c7a1e'];
export const seriesColor = name =>
  PALETTE[[...String(name)].reduce((a, c) => a + c.charCodeAt(0), 0) % PALETTE.length];

/* ── Vân tay kiểu lỗi ────────────────────────────────────────────────────── */

export function fingerprintHTML(prod) {
  const t = store.t;
  const fp = prod && prod.failure_fingerprint;
  if (!fp || !fp.causes || !fp.causes.length) return '';
  const labels = fp.cause_labels || {};
  return `<div class="panel" style="margin-top:14px">
    <div class="panel-head"><h2>${t.fingerprint}</h2>
      <span class="hint">${t.fpNote}</span></div>
    <div class="tbl-wrap"><table>
      <thead><tr><th>&nbsp;</th>
        ${fp.causes.map(c => `<th>${esc(causeLabel(c, labels[c]))}</th>`).join('')}
        <th>${t.sample}</th></tr></thead><tbody>
      ${Object.entries(fp.by_machine).map(([nm, r]) => `<tr>
        <td class="name">${esc(nm)}</td>
        ${fp.causes.map(c => {
          const v = r.by_cause[c];
          return has(v)
            ? `<td class="heat" style="--v:${v}"><span>${num(v, 1)}%</span></td>`
            : `<td class="muted">—</td>`;
        }).join('')}
        <td class="muted">${fmt(r.sample_products)} · ${r.sample_covers_all ? t.fullSample : t.partialSample}</td>
      </tr>`).join('')}
    </tbody></table></div></div>`;
}

/* ── Lưới ảnh sản phẩm lỗi ───────────────────────────────────────────────── */

export function failureGridHTML(d) {
  const t = store.t;
  if (!d || !d.images || !d.images.length)
    return `<div class="coverage">${t.noMetrics}</div>`;
  return `<div class="fgrid">${d.images.map(i => `
    <figure class="fcell">
      <img loading="lazy" src="${esc(i.url)}?w=480" alt="${esc(causeLabel(i.cause, i.cause_label))}">
      <figcaption class="cap">
        <b>${esc(i.machine)}</b> · ${esc(String(i.timestamp || '').slice(11, 16))}
        · ${esc(causeLabel(i.cause, i.cause_label))}
        ${i.expected ? `<div class="exp">${t.expected}: ${esc(String(i.expected).slice(0, 34))}</div>` : ''}
        ${i.recognized ? `<div class="exp">${t.gotRead}: ${esc(String(i.recognized).slice(0, 34))}</div>` : ''}
      </figcaption>
    </figure>`).join('')}</div>` + coverageHTML(d.coverage);
}

/* ── Nhân sự ─────────────────────────────────────────────────────────────── */

const KEYS = { machine: 'machine', dept: 'department', shift: 'shift' };

/* Ca làm dạng "Shift B (14:00–22:00)". Tách ra giờ để biết AI ĐANG TRONG CA —
   đó là câu hỏi đầu tiên khi nhìn danh sách nhân sự lúc 2 giờ sáng, và không có
   trường nào trong DB trả lời sẵn. Nhận cả gạch nối thường lẫn gạch ngang dài,
   vì dữ liệu thật dùng "–". */
const SHIFT_RE = /(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})/;

function shiftWindow(text) {
  const m = SHIFT_RE.exec(String(text || ''));
  if (!m) return null;
  return { from: +m[1] * 60 + +m[2], to: +m[3] * 60 + +m[4],
           label: `${m[1].padStart(2, '0')}:${m[2]}`, endLabel: `${m[3].padStart(2, '0')}:${m[4]}` };
}

function onShiftNow(text, now = new Date()) {
  const w = shiftWindow(text);
  if (!w) return null;                       // "Office hours" — không suy ra được
  const cur = now.getHours() * 60 + now.getMinutes();
  // Ca đêm vắt qua nửa đêm: 22:00–06:00 thì "trong ca" là >=22:00 HOẶC <06:00.
  return w.from <= w.to ? (cur >= w.from && cur < w.to)
                        : (cur >= w.from || cur < w.to);
}

const initials = name => String(name || '?')
  .trim().split(/\s+/).slice(-2).map(w => w[0] || '').join('').toUpperCase() || '?';

function staffStatus(u, t) {
  const on = onShiftNow(u.shift);
  const w = shiftWindow(u.shift);
  const login = u.last_login ? String(u.last_login) : null;
  const today = login && login.slice(0, 10) === new Date().toISOString().slice(0, 10);
  const hhmm = login ? login.slice(11, 16) : null;

  if (on === true)
    return `<span class="st-on">${today && hhmm ? t.onShiftSince(hhmm) : t.onShift}</span>`;
  if (on === false)
    return `<span class="st-off">${today && hhmm ? t.offShiftActive(hhmm)
      : w ? t.offShiftStarts(w.label) : t.offShift}</span>`;
  // Không có giờ ca (hành chính) — chỉ nói lần đăng nhập gần nhất, không đoán.
  return `<span class="st-off">${today && hhmm ? t.lastActive(hhmm) : (u.shift || '—')}</span>`;
}

export function staffHTML(d, opts) {
  const t = store.t;
  if (!d) return `<div class="coverage">${t.loading}</div>`;
  let users = d.users || [];

  const q = (opts.q || '').trim().toLowerCase();
  if (q) users = users.filter(u =>
    [u.full_name, u.username, u.employee_code, u.job_title].some(v =>
      String(v || '').toLowerCase().includes(q)));
  for (const f of ['machine', 'department', 'shift', 'role'])
    if (opts[f]) users = users.filter(u => (u[f] || '') === opts[f]);
  if (opts.onShiftOnly) users = users.filter(u => onShiftNow(u.shift) === true);

  const [g1, g2] = opts.group === 'dept' ? [KEYS.dept, KEYS.machine]
                 : opts.group === 'shift' ? [KEYS.shift, KEYS.machine]
                 : [KEYS.machine, KEYS.dept];

  const tree = {};
  for (const u of users) {
    const a = u[g1] || '—', b = u[g2] || '—';
    ((tree[a] ??= {})[b] ??= []).push(u);
  }

  /* Một DÒNG, không phải một thẻ. Danh sách này để quét dọc tìm người và tìm
     ai đang trong ca; thẻ ảnh xếp lưới thì mỗi người chiếm chỗ gấp bốn và mắt
     không còn cột nào để bám. */
  /* Mỗi dòng vẫn là một NÚT hỏi trợ lý về người đó — đọc thấy tên rồi muốn
     biết họ vừa làm gì là phản xạ tự nhiên, gõ lại username thì vừa chậm vừa
     dễ sai. Viết lại staffHTML theo mockup đã làm rơi mất mấy thuộc tính này
     và cả tính năng biến mất theo. */
  const row = u => `<div class="prow" role="button" tabindex="0"
    data-name="${esc(u.full_name || u.username)}"
    data-username="${esc(u.username)}" data-machine="${esc(u.machine || '')}"
    title="${esc(t.askStaff(u.full_name || u.username, u.username, u.machine || ''))}">
    <span class="prow-av">${u.avatar_url
      ? `<img loading="lazy" src="/api/fleet/avatar/${esc(u.machine)}?p=${encodeURIComponent(u.avatar_url)}" alt="">`
      : esc(initials(u.full_name || u.username))}</span>
    <span class="prow-id">
      <b>${esc(u.full_name || u.username)}</b>
      <i>${esc(u.employee_code || '—')} · @${esc(u.username)}</i>
    </span>
    <span class="prow-job">${esc(u.job_title || '—')}</span>
    <span class="prow-shift">${u.shift ? `<em>${esc(u.shift)}</em>` : ''}</span>
    <span class="prow-acc">${t.access}: ${esc(u.role || '—')}</span>
    <span class="prow-st">${staffStatus(u, t)}</span>
  </div>`;

  const machines = opts.machines || [];
  return Object.entries(tree).sort().map(([a, subs]) => {
    const list = Object.values(subs).flat();
    const on = list.filter(u => onShiftNow(u.shift) === true).length;
    const meta = machines.find(m => m.name === a);
    const sub = Object.entries(subs).sort()
      .map(([b, l]) => `${b} ${l.length}`).join(' · ');
    return `<details class="sgroup" open><summary>
      <span class="sg-name">${esc(a)}${meta?.line ? ` · ${esc(meta.line)}` : ''}</span>
      <span class="sg-model">${esc(meta?.model || sub)}</span>
      <span class="spacer"></span>
      <span class="sg-count"><b>${list.length}</b> ${t.people} ·
        <b class="st-on">${on}</b> ${t.onShiftShort}</span></summary>
      ${Object.entries(subs).sort().map(([b, l]) => `
        <div class="eyebrow sg-dept">${esc(b)} — ${l.length}</div>
        ${l.map(row).join('')}`).join('')}
    </details>`;
  }).join('') + coverageHTML(d.coverage);
}

/* ── Nhật ký ─────────────────────────────────────────────────────────────── */

const ACTIVITY_COLORS = ['#2f69cf', '#0597b4', '#d17500', '#7942d7', '#2f8a55', '#a43a75', '#63707a'];
const activityColor = value => {
  const index = [...String(value || '')].reduce((sum, char) => sum + char.charCodeAt(0), 0);
  return ACTIVITY_COLORS[index % ACTIVITY_COLORS.length];
};
const actionTone = action => {
  if (/recipe/i.test(action)) return 'recipe';
  if (/login|logout/i.test(action)) return 'access';
  if (/user|password/i.test(action)) return 'identity';
  return 'neutral';
};
const activityTime = time => String(time || '').slice(0, 16).replace('T', ' ');

/* Thao tác đổi cách máy chấm sản phẩm. Chỉ những dòng này mới đáng ghép ảnh
   trước/sau: đăng nhập hay xoá camera thì không đổi gì trên tấm ảnh. */
const RECIPE_ACTIONS = new Set(['update_recipe', 'load_recipe', 'stop_recipe']);

export function auditHTML(d, opts = {}) {
  const t = store.t;
  // `null` = chưa tải xong, khác hẳn với đã tải mà rỗng. Gộp hai thứ vào một câu
  // "không lấy được số liệu" thì trong lúc gọi 5 máy (mất vài chục giây) màn hình
  // đang báo hỏng trong khi nó chỉ đang chờ.
  if (!d) return `<div class="coverage">${t.loading}</div>`;
  const all = d.entries || [];
  // "Today" là ngày của bản ghi mới nhất, không dùng thời gian máy người xem.
  // Nếu dashboard mở lại dữ liệu cache, dùng Date.now() sẽ làm bảng trống sai.
  const latestDate = String(all[0]?.time || '').slice(0, 10);
  const query = String(opts.query || '').trim().toLocaleLowerCase();
  const entries = all.filter(e => {
    if (opts.machine && e.machine !== opts.machine) return false;
    if (opts.action && e.action_type !== opts.action) return false;
    if (opts.range === 'today' && latestDate && !String(e.time || '').startsWith(latestDate)) return false;
    if (!query) return true;
    return [e.machine, e.username, e.action_type, e.description, e.resource_id]
      .some(value => String(value || '').toLocaleLowerCase().includes(query));
  });
  const rows = entries.map(e => {
    const at = activityTime(e.time);
    return `<tr class="logrow audit-row">
    <td><time title="${esc(at)}">${esc(at.slice(-5) || '—')}</time></td>
    <td class="name"><span class="machine-dot" style="--machine-color:${activityColor(e.machine)}"></span>${esc(e.machine)}</td>
    <td>${esc(e.username || '—')}</td>
    <td><span class="action-tag ${actionTone(e.action_type)}">${esc(e.action_type || '—')}</span></td>
    <td class="muted" style="text-align:left">${esc(e.description || e.resource_id || '')}</td>
    <td class="audit-act">
      ${RECIPE_ACTIONS.has(e.action_type) ? `<button class="ask seeframes"
        data-machine="${esc(e.machine)}" data-ts="${esc(e.time || '')}"
        title="${esc(t.framesAroundHint)}">${t.framesAround}</button>` : ''}
      <button class="ask" data-ask="${esc(t.askAudit(e.machine, e.username || '—',
        e.action_type || '—', at))}">${t.askMore}</button></td>
  </tr>
  <tr class="frames-row" hidden><td colspan="6"></td></tr>`;
  }).join('');
  return `<div class="tbl-wrap"><table>
    <thead><tr><th>${t.activityTime}</th><th>${t.machine}</th><th>${t.user}</th>
      <th>${t.action}</th><th style="text-align:left">${t.detail}</th><th></th></tr></thead>
    <tbody>${rows || `<tr><td colspan="6" class="muted">${t.noRecords}</td></tr>`}</tbody>
  </table></div>` +
  `<div class="activity-summary">${esc(t.activityShowing(entries.length, all.length))}</div>` +
  (d.machines_without_user?.length
    ? `<div class="coverage">${esc(t.noUserRecords(d.machines_without_user.join(', ')))}</div>` : '') +
  coverageHTML(d.coverage);
}

export function logErrorsHTML(d, selected = null) {
  const t = store.t;
  if (!d) return `<div class="coverage">${t.loading}</div>`;
  const machines = (d.machines || []).filter(machine => !machine.error && machine.problems?.length);
  const selectedMachine = machines.find(machine => machine.machine === selected?.machine) || machines[0];
  const selectedProblem = selectedMachine?.problems?.[selected?.index] || selectedMachine?.problems?.[0];
  const isSelected = (machine, index) => machine.machine === selectedMachine?.machine &&
    machine.problems[index] === selectedProblem;
  // Bảng bên trái là một "radar" rất ngắn: một vấn đề nổi bật nhất/máy.
  // Toàn bộ chi tiết vẫn ở panel phải, tránh lặp 3 dòng/máy như một bảng log.
  const rows = machines.map(machine => {
    const problem = [...machine.problems].sort((a, b) => (b.count || 0) - (a.count || 0))[0];
    const index = machine.problems.indexOf(problem);
    return `<tr class="error-row ${isSelected(machine, index) ? 'selected' : ''}"
      tabindex="0" role="button" data-error-machine="${esc(machine.machine)}" data-error-index="${index}">
      <td class="name"><span class="machine-dot" style="--machine-color:${activityColor(machine.machine)}"></span>${esc(machine.machine)}</td>
      <td class="error-signature"><span class="level-tag ${String(problem.level || '').toLowerCase()}">${esc(problem.level || '—')}</span>${esc(problem.signature || '—')}</td>
      <td>×${fmt(problem.count)}</td>
      <td>${esc(problem.last_seen || '—')}</td>
    </tr>`;
  }).join('');
  const unavailable = (d.machines || []).filter(machine => machine.error).map(machine =>
    `<div class="coverage miss"><b>${esc(machine.machine)}</b> · ${esc(machine.error)}</div>`).join('');
  const detail = selectedMachine && selectedProblem ? `<aside class="error-detail">
      <div class="error-detail-head">
        <div>
          <div class="eyebrow">${esc(selectedMachine.machine)} · ${esc(selectedProblem.category || t.systemErrors)}</div>
          <h3>${esc(selectedProblem.signature || '—')}</h3>
        </div>
        <span class="level-tag ${String(selectedProblem.level || '').toLowerCase()}">${esc(selectedProblem.level || '—')}</span>
      </div>
      <dl class="error-meta">
        <div><dt>${t.activityOccurrences}</dt><dd>×${fmt(selectedProblem.count)}</dd></div>
        <div><dt>${t.activityFirstSeen}</dt><dd>${esc(selectedProblem.first_seen || '—')}</dd></div>
        <div><dt>${t.activityLastSeen}</dt><dd>${esc(selectedProblem.last_seen || '—')}</dd></div>
      </dl>
      <div class="example-label">${t.recordedExample}</div>
      <pre class="log-example">${esc(selectedProblem.example || selectedProblem.signature || '—')}</pre>
      <div class="error-actions">
        <button data-ask="${esc(t.askLogErrors(selectedMachine.machine))}">${esc(t.askAnalyze(selectedMachine.machine))}</button>
        <button data-copy-log="${esc(selectedProblem.example || selectedProblem.signature || '')}">${t.copyExample}</button>
      </div>
      <p class="error-source-note">${t.activityLogSource}</p>
    </aside>` : `<aside class="error-detail empty"><p>${t.noErrorRecords}</p></aside>`;
  return `<div class="error-workspace">
    <div class="tbl-wrap error-list"><table>
      <thead><tr><th>${t.machine}</th><th style="text-align:left">${t.activityErrorGroup}</th>
        <th>${t.activityCount}</th><th>${t.activityLastSeen}</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="4" class="muted">${t.noRecords}</td></tr>`}</tbody>
    </table></div>
    ${detail}
  </div>${unavailable}${coverageHTML(d.coverage)}`;
}

/* ── Ngăn kéo chi tiết máy ───────────────────────────────────────────────── */

export function drawerHTML(m, prod, { headOnly = false, bodyOnly = false } = {}) {
  const t = store.t;
  const x = m.metrics || {};
  const row = (k, v) => `<div class="kv"><span class="k">${k}</span><span>${v}</span></div>`;
  const p = (prod?.machines || []).find(r => r.machine === m.name) || {};
  const pr = p.production || {};
  const trend = pr.trend || {};
  const keys = Object.keys(trend).sort();
  const maxV = Math.max(1, ...keys.map(k => (trend[k].pass || 0) + (trend[k].fail || 0)));

  const spark = keys.length ? `<div class="spark">${keys.map(k => {
    const v = trend[k], tot = (v.pass || 0) + (v.fail || 0);
    const h = Math.max(2, Math.round(tot * 100 / maxV));
    const failH = tot ? Math.round((v.fail || 0) * 100 / tot) : 0;
    const detail = `${k} · ${fmt(v.pass || 0)} pass · ${fmt(v.fail || 0)} fail · ${fmt(tot)} total`;
    return `<button class="spark-bar" type="button" style="height:${h}%;background:linear-gradient(to top,
      var(--bad) 0 ${failH}%, var(--accent) ${failH}% 100%)" aria-label="${esc(detail)}">
      <span class="spark-tip" role="tooltip">${esc(detail)}</span></button>`;
  }).join('')}</div><div class="eyebrow">${esc(keys[0] || '')} → ${esc(keys.at(-1) || '')}</div>` : '';

  const fp = prod?.failure_fingerprint;
  const mine = fp?.by_machine?.[m.name];
  const bars = mine ? fp.causes.map(c => {
    const v = mine.by_cause[c] || 0;
    return `<div style="margin-top:6px">
      <div style="display:flex;justify-content:space-between;font-size:11.5px">
        <span>${esc(causeLabel(c, (fp.cause_labels || {})[c]))}</span><b>${num(v, 1)}%</b></div>
      <div class="bar"><i style="width:${v}%"></i></div></div>`;
  }).join('') : '';

  /* Header là thanh ĐẦU TIÊN của drawer và dính lại khi cuộn: nút đóng mà trôi
     mất theo nội dung thì người ta phải cuộn ngược lên mới thoát ra được. Nút
     hỏi trợ lý cũng lên đây — trước nó nằm tận cuối, sau cả vân tay lỗi, tức là
     phải cuộn hết trang mới thấy thứ mình định bấm ngay từ đầu. */
  return `${bodyOnly ? '' : `<div class="drawer-head">
      <div style="display:flex;align-items:center;gap:10px">
        <h2 style="font-size:20px">${esc(m.name)}</h2>
        <span class="state ${TONE[m.state] || 'bad'}">${esc(t.state[m.state] || m.state)}</span>
        <span class="spacer"></span>
        <button class="ask-btn" onclick="window.__askAbout('${esc(m.name)}')">
          ${t.chatTitle} →</button>
        <button class="close-btn" onclick="window.__closeDrawer()"
          aria-label="${esc(t.close)}">✕</button>
      </div>
      <div class="eyebrow" style="margin-top:4px">${esc(m.line || '')} ·
        ${esc(m.model || '')} · ${esc(m.ip)}</div>
    </div>`}${headOnly ? '' : `
    <div class="drawer-body">
      ${spark}
      ${row(t.output, fmt(pr.total_products))}
      ${row(t.passRate, has(pr.pass_rate) ? num(pr.pass_rate, 2) + '%' : NA)}
      ${row(t.perDay, fmt(pr.per_day, 1))}
      <div class="eyebrow" style="margin:16px 0 4px">${t.hardware}</div>
      ${row('CPU', has(x.cpu_temp) ? `${num(x.cpu_temp, 1)}°C · ${t.load} ${num(x.cpu_percent, 0)}%` : NA)}
      ${row('GPU', has(x.gpu_temp) ? `${num(x.gpu_temp, 1)}°C · ${t.load} ${num(x.gpu_percent, 0)}%` : NA)}
      ${row('RAM', has(x.ram_percent) ? `${num(x.ram_percent, 0)}% · ${num(x.ram_used_gb, 1)}/${num(x.ram_total_gb, 0)} GB` : NA)}
      ${row(t.disk, has(x.disk_percent) ? `${num(x.disk_percent, 0)}% · ${num(x.disk_free_gb, 0)} GB ${t.free}` : NA)}
      ${row(t.uptime, uptime(x.uptime_seconds))}
      ${m.service ? row('Camera service', m.service.is_running ? t.cameraOn : t.cameraOff) : ''}
      ${bars ? `<div class="eyebrow" style="margin:16px 0 4px">${t.fingerprint}</div>${bars}
        <div class="muted" style="font-size:11px;margin-top:6px">
          ${fmt(mine.sample_products)} ${t.sample} · ${mine.sample_covers_all ? t.fullSample : t.partialSample}</div>` : ''}
      ${(m.errors || []).map(e => `<div class="coverage miss" style="margin-top:10px">⚠ ${esc(e)}</div>`).join('')}
    </div>`}`;
}
