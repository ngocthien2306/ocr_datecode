/* ═══════════════════════════════════════════════════════════════════════════
   Các khối hiển thị: thẻ máy, thống kê, ảnh lỗi, nhân sự, nhật ký.
   ═══════════════════════════════════════════════════════════════════════════ */

import { store, has, num, fmt, NA, level, uptime, esc, clock,
         coverageHTML, deltaPts, causeLabel } from './core.js';

const TONE = { ok: 'ok', warn: 'warn', agent_down: 'mute',
               unreachable: 'bad', offline: 'bad' };

/* ── Thẻ máy ─────────────────────────────────────────────────────────────── */

export function machineCard(m) {
  const t = store.t;
  const tone = TONE[m.state] || 'bad';
  const x = m.metrics || {};
  const p = m.production || {};
  const rec = (m.recipes || [])[0];

  const cls = tone === 'warn' ? 'mcard s-warn'
            : tone === 'bad' ? 'mcard s-bad'
            : tone === 'mute' ? 'mcard s-mute' : 'mcard';

  // Phần trăm LUÔN kèm số tuyệt đối: "RAM 87%" trên máy 8 GB và trên máy 32 GB
  // là hai tình huống khác hẳn, mà phần trăm thì giống hệt.
  const hw = (label, val, sub, lv) => `
    <div><div class="eyebrow">${label}</div>
      <div class="v ${lv || ''}">${val ?? NA}</div>
      ${sub ? `<div class="sub">${sub}</div>` : ''}</div>`;

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
      ${hw('RAM', has(x.ram_percent) ? `${num(x.ram_percent, 0)}%` : null,
           has(x.ram_used_gb) ? `${num(x.ram_used_gb, 1)}/${num(x.ram_total_gb, 0)} GB` : '',
           level(x.ram_percent))}
      ${hw(t.disk, has(x.disk_percent) ? `${num(x.disk_percent, 0)}%` : null,
           has(x.disk_free_gb) ? `${num(x.disk_free_gb, 0)} GB ${t.free}` : '',
           level(x.disk_percent))}
    </div>` : `
    <div class="mrow"><div class="muted" style="padding:18px 0">${t.noMetrics}</div></div>`;

  return `<article class="${cls}" data-node="${esc(m.node_id)}">
    <div class="mcard-top">
      <span class="eyebrow">${esc(m.line || '')} · ${esc(m.model || m.hostname || '')}</span>
      <span class="state ${tone}">${esc(t.state[m.state] || m.state)}</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
      <div class="mname">${esc(m.name)}</div>
      ${rec ? `<span class="recipe-chip">${esc(rec.name)}</span>` : ''}
    </div>
    ${body}
  </article>`;
}

export const skeletonCard = () =>
  `<article class="mcard skel"><div class="ln" style="width:40%"></div>
   <div class="ln" style="width:65%;height:24px"></div>
   <div class="ln" style="width:50%;height:34px"></div>
   <div class="ln" style="width:90%"></div><div class="ln" style="width:80%"></div>
   </article>`;

/* ── Thống kê toàn nhà máy ───────────────────────────────────────────────── */

export function statsHTML(prod, view) {
  const t = store.t;
  if (!prod) return `<div class="coverage">${t.noMetrics}</div>`;
  const tot = prod.fleet_total || {};
  const rows = prod.machines || [];

  const table = `<div class="tbl-wrap"><table>
    <thead><tr><th>&nbsp;</th><th>${t.output}</th><th>${t.perDay}</th>
      <th>${t.passRate}</th><th>${t.recipe}</th></tr></thead><tbody>
    ${rows.map(r => {
      const p = r.production;
      if (!p) return `<tr><td class="name">${esc(r.machine)}</td>
        <td colspan="4" class="muted">${esc(r.error || t.noMetrics)}</td></tr>`;
      const rec = (r.recipes || [])[0];
      return `<tr><td class="name">${esc(r.machine)}
          <div class="sub" style="font-weight:400;color:var(--faint);font-size:11px">${esc(r.line || '')}</div></td>
        <td>${fmt(p.total_products)}</td><td>${fmt(p.per_day, 1)}</td>
        <td>${num(p.pass_rate, 2)}%</td>
        <td class="muted">${esc(rec ? rec.name : '—')}</td></tr>`;
    }).join('')}
    <tr class="total"><td>${t.fleetTotal}</td><td>${fmt(tot.products)}</td>
      <td>—</td><td>${num(tot.pass_rate, 2)}%</td><td class="muted">—</td></tr>
    </tbody></table></div>`;

  /* Biểu đồ cột: sản lượng mỗi ngày, đã chuẩn hoá vì các máy chạy số ngày khác
     nhau. Chiều cao cột tính bằng PX chứ không phải %: cột nằm trong một flex
     item cao auto, nên `height:%` quy về auto và mọi cột biến mất — bản trước
     chỉ còn trơ lại con số lơ lửng.                                           */
  const vals = rows.map(r => (r.production || {}).per_day || 0);
  const max = Math.max(1, ...vals);
  const H = 132;
  const cols = `repeat(${rows.length}, minmax(0, 1fr))`;
  const chart = `<div style="padding:14px 16px">
    <div style="display:grid;grid-template-columns:${cols};gap:14px;align-items:end">
      ${rows.map((r, i) => {
        const v = vals[i];
        const h = Math.max(3, Math.round(v * H / max));
        return `<div style="text-align:center">
          <div style="font-size:11px;font-weight:600;margin-bottom:4px">${fmt(v, 0)}</div>
          <div title="${esc(r.machine)}: ${fmt(v, 0)}" style="height:${h}px;
            max-width:72px;margin:0 auto;
            background:${seriesColor(r.machine)};border-radius:3px 3px 0 0"></div>
        </div>`;
      }).join('')}
    </div>
    <div style="display:grid;grid-template-columns:${cols};gap:14px;
      border-top:1px solid var(--line);padding-top:6px;margin-top:0">
      ${rows.map(r => `<div class="eyebrow" style="text-align:center">${esc(r.machine)}</div>`).join('')}
    </div>
    <div class="eyebrow" style="margin-top:10px">${t.output} / ${t.perDay}</div></div>`;

  return (view === 'table' ? table : chart) +
    `<div style="padding:0 16px 12px"><span class="muted" style="font-size:11.5px">
      ${t.noRank}</span></div>` + coverageHTML(prod.coverage);
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

export function staffHTML(d, opts) {
  const t = store.t;
  if (!d) return `<div class="coverage">${t.loading}</div>`;
  let users = d.users || [];

  const q = (opts.q || '').trim().toLowerCase();
  if (q) users = users.filter(u =>
    [u.full_name, u.username, u.employee_code].some(v =>
      String(v || '').toLowerCase().includes(q)));
  for (const f of ['machine', 'department', 'shift', 'role'])
    if (opts[f]) users = users.filter(u => (u[f] || '') === opts[f]);

  const [g1, g2] = opts.group === 'dept' ? [KEYS.dept, KEYS.machine]
                 : opts.group === 'shift' ? [KEYS.shift, KEYS.machine]
                 : [KEYS.machine, KEYS.dept];

  const tree = {};
  for (const u of users) {
    const a = u[g1] || '—', b = u[g2] || '—';
    ((tree[a] ??= {})[b] ??= []).push(u);
  }

  const card = u => `<div class="person">
    ${u.avatar_url
      ? `<img loading="lazy" src="/api/fleet/avatar/${esc(u.machine)}?p=${encodeURIComponent(u.avatar_url)}" alt="">`
      : `<div class="avatar-fallback">${esc((u.full_name || u.username || '?')[0])}</div>`}
    <div style="min-width:0">
      <div class="nm">${esc(u.full_name || u.username)}</div>
      <div class="meta">${esc(u.employee_code || '—')} · @${esc(u.username)}</div>
      <div class="meta">${esc(u.job_title || '—')}${u.shift ? ' · ' + esc(u.shift) : ''}</div>
      <span class="access">${t.access}: ${esc(u.role || '—')}</span>
    </div></div>`;

  return Object.entries(tree).sort().map(([a, subs]) => {
    const n = Object.values(subs).reduce((s, v) => s + v.length, 0);
    return `<details class="sgroup" open><summary>${esc(a)}
      <span class="count">${n} ${t.people}</span></summary>
      ${Object.entries(subs).sort().map(([b, list]) => `
        <div class="eyebrow" style="padding:4px 16px">${esc(b)} · ${list.length}</div>
        <div class="people">${list.map(card).join('')}</div>`).join('')}
    </details>`;
  }).join('') + coverageHTML(d.coverage);
}

/* ── Nhật ký ─────────────────────────────────────────────────────────────── */

export function auditHTML(d) {
  const t = store.t;
  // `null` = chưa tải xong, khác hẳn với đã tải mà rỗng. Gộp hai thứ vào một câu
  // "không lấy được số liệu" thì trong lúc gọi 5 máy (mất vài chục giây) màn hình
  // đang báo hỏng trong khi nó chỉ đang chờ.
  if (!d) return `<div class="coverage">${t.loading}</div>`;
  const rows = (d.entries || []).map(e => `<tr class="logrow">
    <td>${esc(String(e.time || '').slice(0, 16).replace('T', ' '))}</td>
    <td class="name">${esc(e.machine)}</td>
    <td>${esc(e.username || '—')}</td>
    <td>${esc(e.action_type || '—')}</td>
    <td class="muted" style="text-align:left">${esc(e.description || e.resource_id || '')}</td>
    <td><button class="ask" data-ask="${esc(t.askAudit(e.machine, e.username || '—',
        e.action_type || '—', String(e.time || '').slice(0, 16).replace('T', ' ')))}">${t.askMore}</button></td>
  </tr>`).join('');
  return `<div class="tbl-wrap"><table>
    <thead><tr><th>${t.updated}</th><th>${t.machine}</th><th>${t.user}</th>
      <th>${t.action}</th><th style="text-align:left">${t.detail}</th><th></th></tr></thead>
    <tbody>${rows || `<tr><td colspan="6" class="muted">${t.noRecords}</td></tr>`}</tbody>
  </table></div>` +
  (d.machines_without_user?.length
    ? `<div class="coverage">${esc(t.noUserRecords(d.machines_without_user.join(', ')))}</div>` : '') +
  coverageHTML(d.coverage);
}

export function logErrorsHTML(d) {
  const t = store.t;
  if (!d) return `<div class="coverage">${t.loading}</div>`;
  return `<div class="tbl-wrap"><table>
    <thead><tr><th>${t.machine}</th><th>${t.lines}</th><th>${t.groups}</th>
      <th style="text-align:left">${t.topProblems}</th><th></th></tr></thead><tbody>
    ${(d.machines || []).map(m => m.error
      ? `<tr><td class="name">${esc(m.machine)}</td>
         <td colspan="4" class="muted">${esc(m.error)}</td></tr>`
      : `<tr class="logrow"><td class="name">${esc(m.machine)}</td>
         <td>${fmt(m.total_problem_lines)}</td><td>${fmt(m.distinct_problems)}</td>
         <td class="muted" style="text-align:left;white-space:normal;max-width:520px">
           ${(m.problems || []).slice(0, 2).map(p =>
             `[${esc(p.level)}] ×${p.count} ${esc(String(p.signature || '').slice(0, 90))}`).join('<br>')}</td>
         <td><button class="ask" data-ask="${esc(t.askLogErrors(m.machine))}">${t.askMore}</button></td>
       </tr>`).join('')}
  </tbody></table></div>` + coverageHTML(d.coverage);
}

/* ── Ngăn kéo chi tiết máy ───────────────────────────────────────────────── */

export function drawerHTML(m, prod) {
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
    return `<i style="height:${h}%;background:linear-gradient(to top,
      var(--bad) 0 ${failH}%, var(--accent) ${failH}% 100%)"
      title="${esc(k)}: ${v.pass}/${tot}"></i>`;
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

  return `<div class="drawer-head">
      <div style="display:flex;align-items:center;gap:10px">
        <h2 style="font-size:20px">${esc(m.name)}</h2>
        <span class="state ${TONE[m.state] || 'bad'}">${esc(t.state[m.state] || m.state)}</span>
        <span class="spacer"></span>
        <button onclick="window.__closeDrawer()">✕</button>
      </div>
      <div class="eyebrow" style="margin-top:4px">${esc(m.line || '')} ·
        ${esc(m.model || '')} · ${esc(m.ip)}</div>
    </div>
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
      <button style="margin-top:16px" onclick="window.__askAbout('${esc(m.name)}')">
        ${t.chatTitle} → ${esc(m.name)}</button>
    </div>`;
}
