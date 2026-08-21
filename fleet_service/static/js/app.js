/* ═══════════════════════════════════════════════════════════════════════════
   Fleet Console — điều phối.

   Nhịp làm mới khác nhau theo chi phí thật của từng nguồn:
     trạng thái + phần cứng   30s   (đo được 1,3s, đổi liên tục)
     sản xuất + vân tay lỗi   5 phút (rollup lạnh mổ vài trăm document/máy)
     nhân sự / nhật ký        khi mở tab (gần như không đổi)

   Dashboard KHÔNG đi qua LLM. Hôm OpenAI hết credit, cả 5 agent im tiếng cùng
   lúc mà màn hình này vẫn phải sống.
   ═══════════════════════════════════════════════════════════════════════════ */

import { store, I18N, api, esc, clock, coverageHTML } from './core.js';
import * as map from '/shared/factory-map-3d.js';  // dùng chung với agent_service
import * as V from './views.js';
import * as chat from './chat.js';
import * as frames from './frame-view.js';
import * as wall from './frame-wall.js';

const $ = s => document.querySelector(s);
const state = { status: null, prod: null, staff: null, audit: null,
                logerr: null, images: null,
                selected: null, tab: 'overview', statsView: 'chart', fpView: 'chart',
                period: 7, staffOpts: { group: 'machine' }, simulated: false,
                activityView: 'audit', auditOpts: { machine: '', action: '', range: 'today', query: '' },
                selectedLogProblem: null,
                stale: null };

/* Inventory vẫn phải hiện khi Tailnet không tới được: đây là dữ liệu tạm ở
   frontend, không giả telemetry và tự biến mất từng máy khi API trả máy thật
   cùng tên. Model được chuẩn hoá theo inventory vật lý của xưởng. */
const DISPLAY_MODEL = {
  Auto2: 'Jetson Orin Nano 8GB',
  M1: 'Jetson Orin Nano 8GB',
  M2: 'Jetson Orin Nano 8GB',
  LineTine: 'Jetson Orin Nano 8GB',
  'PC-Auto-1': 'Jetson AGX Orin 16GB',
  'Auto 1': 'Jetson Orin Nano 8GB Super',
  'Tin 2': 'Jetson Orin Nano 8GB Super',
};
const PROVISIONAL_MACHINES = [
  { node_id: '__offline_auto2__', name: 'Auto2', line: 'Line 1',
    hostname: 'auto2', model: DISPLAY_MODEL.Auto2, state: 'offline', floor: { x: 0, y: 0 }, provisional: true },
  { node_id: '__offline_m1__', name: 'M1', line: 'Line 2',
    hostname: 'm1', model: DISPLAY_MODEL.M1, state: 'offline', floor: { x: 0, y: 0 }, provisional: true },
  { node_id: '__offline_m2__', name: 'M2', line: 'Line 3',
    hostname: 'm2', model: DISPLAY_MODEL.M2, state: 'offline', floor: { x: 0, y: 0 }, provisional: true },
  { node_id: '__offline_linetine__', name: 'LineTine', line: 'Tine line',
    hostname: 'linetine', model: DISPLAY_MODEL.LineTine, state: 'offline', floor: { x: 0, y: 0 }, provisional: true },
  { node_id: '__offline_pc_auto_1__', name: 'PC-Auto-1', line: 'Auto line',
    hostname: 'pc-auto-1', model: DISPLAY_MODEL['PC-Auto-1'], state: 'offline', floor: { x: 0, y: 0 }, provisional: true },
  { node_id: '__offline_auto_1__', name: 'Auto 1', line: 'Carton line 5',
    hostname: 'auto-1', model: DISPLAY_MODEL['Auto 1'], state: 'offline', floor: { x: 0, y: 0 }, provisional: true },
  { node_id: '__offline_tin_2__', name: 'Tin 2', line: 'Carton line 6 · annex building',
    hostname: 'tin-2', model: DISPLAY_MODEL['Tin 2'], state: 'offline', floor: { x: 0, y: 0 }, provisional: true },
];

function applyProvisionalFleet(status) {
  const live = status?.machines || [];
  const names = new Set(live.map(machine => machine.name));
  const machines = live.map(machine => ({ ...machine,
    model: DISPLAY_MODEL[machine.name] || machine.model,
  }));
  PROVISIONAL_MACHINES.forEach(machine => {
    if (!names.has(machine.name)) machines.push({ ...machine });
  });
  return { ...(status || {}), machines };
}

/* ── Thanh đầu trang ─────────────────────────────────────────────────────── */

function paintTop() {
  const t = store.t;
  $('#no-login').textContent = t.noLogin;
  $('#btn-refresh').textContent = t.refresh;
  const ms = state.status?.machines || [];
  const ok = ms.filter(m => m.state === 'ok').length;
  const bad = ms.length - ok;
  $('#tally').innerHTML =
    `<span><span class="pip" style="background:var(--ok)"></span><b>${ok}</b> ${t.running}</span>` +
    (bad ? `<span><span class="pip" style="background:var(--warn)"></span><b>${bad}</b> ${t.attention}</span>` : '');
  $('#clock').textContent = new Date().toLocaleTimeString(t.locale);
  ['en', 'vi'].forEach(l => $('#btn-' + l).setAttribute('aria-pressed', store.lang === l));
  ['light', 'dark'].forEach(x => $('#btn-' + x).setAttribute('aria-pressed', store.theme === x));
  $('#tab-overview').textContent = t.tabOverview;
  $('#tab-frames').textContent = t.tabFrames;
  $('#tab-staff').textContent = t.tabStaff;
  $('#tab-log').textContent = t.tabLog;
  $('#stale').innerHTML = state.stale
    ? `<span class="muted" style="font-size:11.5px">${t.staleAt(clock(state.stale))}</span>` : '';
}

/* ── Fold đầu: sơ đồ + lưới thẻ ──────────────────────────────────────────── */

function paintFold() {
  const t = store.t;
  $('#floor-title').textContent = t.floorTitle;
  $('#floor-hint').textContent = t.floorHint;
  $('#mach-title').textContent = t.machinesTitle;
  $('#mach-hint').textContent = t.clickHint;
  $('#map-legend').innerHTML = map.legendHTML();

  const ms = (state.status?.machines || []).map(m => {
    const p = (state.prod?.machines || []).find(r => r.machine === m.name);
    return { ...m, production: p?.production, recipes: p?.recipes };
  });

  if (!ms.length) {
    $('#grid').innerHTML = Array.from({ length: 5 }, V.skeletonCard).join('');
    return;
  }
  map.render($('#map'), {
    machines: ms, selected: state.selected,
    onSelect: id => openDrawer(id),
    // Module dùng chung không import store/bậc-1 của riêng service nào — hai
    // thứ đó do bên gọi cấp, nên cùng một file chạy được ở cả hai bề mặt.
    store,
    fallback: () => import('./factory-map.js'),
  });
  $('#grid').innerHTML = ms.map(V.machineCard).join('');
  $('#grid').querySelectorAll('.mcard').forEach(c =>
    c.onclick = () => openDrawer(c.dataset.node));
}

/* ── Ngăn kéo ────────────────────────────────────────────────────────────── */

function openDrawer(nodeId) {
  const m = (state.status?.machines || []).find(x => x.node_id === nodeId);
  if (!m) return;
  state.selected = nodeId;
  // Ảnh đặt TRÊN CÙNG: mở drawer ra là để xem máy đang chạy thế nào, và một
  // tấm ảnh sản phẩm trả lời câu đó nhanh hơn mọi con số bên dưới. Nằm dưới
  // vân tay lỗi thì phải cuộn qua cả trang mới thấy.
  // Thứ tự: header (tên máy, nút hỏi, nút đóng) → ảnh sản phẩm → số liệu.
  // Ảnh trả lời "máy đang chạy thế nào" nhanh nhất, nhưng nó không được đẩy
  // header xuống giữa trang — nút đóng phải luôn ở nơi mắt tìm nó đầu tiên.
  $('#drawer').innerHTML =
    V.drawerHTML(m, state.prod, { headOnly: true })
    + '<section class="panel frame-panel" id="frame-panel"></section>'
    + V.drawerHTML(m, state.prod, { bodyOnly: true });
  $('#drawer').classList.add('open');
  // Ảnh chỉ tải khi drawer MỞ, và dừng hẳn khi đóng: kéo ảnh về cho một máy
  // không ai còn nhìn là chạm vào Jetson đang chạy dây chuyền, không lý do.
  frames.open(m.name);
  paintFold();
}
/** Mở drawer theo TÊN máy — tường ảnh chỉ biết tên, không biết node id. */
function openDrawerByName(name) {
  const m = (state.status?.machines || []).find(x => x.name === name);
  if (m) openDrawer(m.node_id);
}

window.__closeDrawer = () => {
  frames.stop();
  state.selected = null;
  $('#drawer').classList.remove('open');
  paintFold();
};
window.__askAbout = name => {
  // Drawer nằm trên chat theo z-index; giữ nó mở thì panel chat đã focus vẫn bị
  // che hoàn toàn. Đóng trước, bỏ chọn máy rồi mới mở chat theo ngữ cảnh đó.
  window.__closeDrawer();
  chat.setContext(name);
};

/* ── Tab ─────────────────────────────────────────────────────────────────── */

function paintTabs() {
  const t = store.t;
  ['overview', 'staff', 'frames', 'log'].forEach(k =>
    $('#tab-' + k).setAttribute('aria-selected', state.tab === k));
  $('#pane-overview').hidden = state.tab !== 'overview';
  $('#pane-staff').hidden = state.tab !== 'staff';
  $('#pane-frames').hidden = state.tab !== 'frames';
  $('#pane-log').hidden = state.tab !== 'log';
  // Rời tab là dừng hẳn: tường ảnh ở chế độ camera hỏi BẢY máy một lúc, để nó
  // chạy ngầm dưới một tab không ai xem là phí đúng bảy lần.
  if (state.tab !== 'frames') wall.stop();

  if (state.tab === 'overview') {
    $('#stats-title').textContent = t.quality;
    $('#fail-title').textContent = t.failures;
    $('#btn-export').textContent = t.export;
    $('#stats-body').innerHTML = V.statsHTML(state.prod, state.statsView, state.status?.machines || []);
    /* Panel "Failure fingerprint" tạm ẩn theo yêu cầu — chưa đạt về mặt hình.
       KHÔNG xoá `V.fingerprintHTML` cũng không xoá dữ liệu: vân tay lỗi vẫn là
       thứ duy nhất so sánh được giữa các máy chạy sản phẩm khác nhau, và trợ lý
       vẫn dùng đúng số liệu đó để trả lời cũng như để dựng gợi ý. Muốn bật lại
       thì thêm <div id="fingerprint"> vào index.html và gỡ chú thích này. */
    $('#fail-body').innerHTML = V.failureGridHTML(state.images);
  }
  if (state.tab === 'staff') {
    $('#staff-body').innerHTML = V.staffHTML(state.staff,
      { ...state.staffOpts, machines: state.status?.machines || [] });
    paintStaffFilters();
    $('#staff-body').querySelectorAll('.prow').forEach(el => {
      const open = () => chat.setPerson({ name: el.dataset.name,
        username: el.dataset.username, machine: el.dataset.machine });
      el.onclick = open;
      el.onkeydown = e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
      };
    });
  }
  if (state.tab === 'log') {
    paintActivity();
  }
}

function paintActivity() {
  const t = store.t;
  const audit = state.activityView === 'audit';
  $('#activity-audit').textContent = t.userActions;
  $('#activity-errors').textContent = t.systemErrors;
  $('#activity-audit').setAttribute('aria-pressed', audit);
  $('#activity-errors').setAttribute('aria-pressed', !audit);
  $('#activity-note').textContent = audit ? t.activityAuditNote : t.activityErrorNote;
  $('#sim-control').hidden = !audit;
  $('#sim-label').textContent = t.showSimulated;
  $('#activity-audit').onclick = () => {
    state.activityView = 'audit';
    paintTabs();
  };
  $('#activity-errors').onclick = () => {
    state.activityView = 'errors';
    paintTabs();
  };

  const entries = state.audit?.entries || [];
  const values = key => [...new Set(entries.map(entry => entry[key]).filter(Boolean))].sort();
  $('#activity-filters').innerHTML = audit ? `<div class="activity-filters">
    <select id="audit-machine" aria-label="${esc(t.machine)}">
      <option value="">${esc(t.activityAllMachines)}</option>
      ${values('machine').map(value => `<option value="${esc(value)}" ${value === state.auditOpts.machine ? 'selected' : ''}>${esc(value)}</option>`).join('')}
    </select>
    <select id="audit-action" aria-label="${esc(t.action)}">
      <option value="">${esc(t.activityAllActions)}</option>
      ${values('action_type').map(value => `<option value="${esc(value)}" ${value === state.auditOpts.action ? 'selected' : ''}>${esc(value)}</option>`).join('')}
    </select>
    <select id="audit-range" aria-label="${esc(t.activityPeriod)}">
      <option value="today" ${state.auditOpts.range === 'today' ? 'selected' : ''}>${esc(t.activityToday)}</option>
      <option value="week" ${state.auditOpts.range === 'week' ? 'selected' : ''}>${esc(t.activityWeek)}</option>
    </select>
    <input id="audit-query" type="search" value="${esc(state.auditOpts.query)}" placeholder="${esc(t.searchActivity)}" aria-label="${esc(t.searchActivity)}">
  </div>` : '';

  const renderBody = () => {
    $('#activity-body').innerHTML = audit
      ? V.auditHTML(state.audit, state.auditOpts)
      : V.logErrorsHTML(state.logerr, state.selectedLogProblem);
    $('#activity-body').querySelectorAll('[data-ask]').forEach(button =>
      button.onclick = () => chat.askNow(button.dataset.ask));
    // Ảnh trước/sau một lần sửa recipe: mở ngay dưới chính dòng nhật ký đó, để
    // hành động và hậu quả của nó nằm cạnh nhau chứ không phải hai màn hình.
    $('#activity-body').querySelectorAll('button.seeframes').forEach(b =>
      b.onclick = () => toggleFrames(b));
    restoreFrames();
    $('#activity-body').querySelectorAll('[data-error-machine]').forEach(row => {
      const selectProblem = () => {
        state.selectedLogProblem = { machine: row.dataset.errorMachine, index: Number(row.dataset.errorIndex) };
        renderBody();
      };
      row.onclick = selectProblem;
      row.onkeydown = event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectProblem(); }
      };
    });
    $('#activity-body').querySelectorAll('[data-copy-log]').forEach(button => {
      button.onclick = async () => {
        try {
          await navigator.clipboard.writeText(button.dataset.copyLog);
          button.textContent = t.copied;
          setTimeout(() => { button.textContent = t.copyExample; }, 1400);
        } catch (_) {}
      };
    });
  };
  renderBody();

  if (!audit) return;
  const update = (key, value) => {
    state.auditOpts[key] = value;
    renderBody();
  };
  $('#audit-machine').onchange = event => update('machine', event.target.value);
  $('#audit-action').onchange = event => update('action', event.target.value);
  $('#audit-range').onchange = event => update('range', event.target.value);
  $('#audit-query').oninput = event => update('query', event.target.value);
}

/* Hàng ảnh nào đang mở. Bảng nhật ký được vẽ lại mỗi nhịp làm mới trạng thái
   (30 giây), và nếu không nhớ thì hàng ảnh vừa mở ra bị đóng sập ngay giữa lúc
   người ta đang so hai tấm. */
const openFrames = new Set();

function restoreFrames() {
  if (!openFrames.size) return;
  document.querySelectorAll('#activity-body button.seeframes').forEach(b => {
    if (openFrames.has(`${b.dataset.machine}|${b.dataset.ts}`)) toggleFrames(b, true);
  });
}

async function toggleFrames(btn, restoring = false) {
  const holder = btn.closest('tr')?.nextElementSibling;
  if (!holder || !holder.classList.contains('frames-row')) return;
  const cell = holder.firstElementChild;
  const key = `${btn.dataset.machine}|${btn.dataset.ts}`;
  if (!holder.hidden && !restoring) {
    holder.hidden = true;
    openFrames.delete(key);
    return;
  }
  holder.hidden = false;
  openFrames.add(key);
  const t = store.t;
  cell.innerHTML = `<div class="coverage">${t.loading}</div>`;

  const m = btn.dataset.machine, ts = btn.dataset.ts;
  const r = await api(`/api/fleet/frames-around/${encodeURIComponent(m)}`
    + `?ts=${encodeURIComponent(ts)}`);
  const d = r.data || {};
  const shot = (f, label) => {
    if (!f) return `<figure class="shot"><div class="shot-empty">—</div>
      <figcaption class="shot-cap">${esc(label)}</figcaption></figure>`;
    const src = `/api/fleet/failure-image/${encodeURIComponent(m)}`
      + `/${encodeURIComponent(f.id)}?w=420`;
    const said = (f.expected != null || f.recognized != null)
      ? ` · ${t.expected} ${f.expected ?? '—'} → ${t.readAs} ${f.recognized || t.emptyRead}` : '';
    return `<figure class="shot ${f.verdict === 'FAIL' ? 'bad' : ''}">
      <img loading="lazy" src="${src}" alt="">
      <figcaption class="shot-badge">${esc(f.verdict || '')}</figcaption>
      <figcaption class="shot-cap">${esc(label)} · ${esc(String(f.timestamp || '').slice(11, 19))}${esc(said)}</figcaption>
    </figure>`;
  };
  cell.innerHTML = (!d.before && !d.after)
    ? `<div class="coverage">${t.framesAroundNone}</div>`
    : `<div class="shots frames-around">${shot(d.before, t.beforeChange)}
        ${shot(d.after, t.afterChange)}</div>`;
}

function paintStaffFilters() {
  const t = store.t;
  const users = state.staff?.users || [];
  const uniq = k => [...new Set(users.map(u => u[k]).filter(Boolean))].sort();
  const sel = (id, key, label) => {
    const el = $('#f-' + id);
    if (!el) return;
    const cur = state.staffOpts[key] || '';
    el.innerHTML = `<option value="">${label} · ${t.all}</option>` +
      uniq(key).map(v => `<option ${v === cur ? 'selected' : ''}>${esc(v)}</option>`).join('');
    el.onchange = () => { state.staffOpts[key] = el.value || null; paintTabs(); };
  };
  sel('machine', 'machine', t.machine);
  sel('dept', 'department', t.department);
  sel('shift', 'shift', t.shift);
  sel('role', 'role', t.access);
  $('#f-q').placeholder = t.searchStaff;
  const nMach = new Set(users.map(u => u.machine).filter(Boolean)).size;
  $('#staff-count').textContent = t.staffCount(users.length, nMach);
  $('#staff-readonly').textContent = t.staffReadOnly;
  $('#staff-loadnote').textContent = t.staffLoadNote;
  $('#staff-keynote').textContent = t.staffKeyNote;
  $('#onshift-label').textContent = t.onShiftOnly;
  const oc = $('#f-onshift');
  oc.checked = !!state.staffOpts.onShiftOnly;
  oc.onchange = () => { state.staffOpts.onShiftOnly = oc.checked; paintTabs(); };
  $('#f-q').oninput = () => { state.staffOpts.q = $('#f-q').value; paintTabs(); };
  $('#g-machine').textContent = t.byMachineDept;
  $('#g-dept').textContent = t.byDeptMachine;
  $('#g-shift').textContent = t.byShiftMachine;
  ['machine', 'dept', 'shift'].forEach(g =>
    $('#g-' + g).setAttribute('aria-pressed', state.staffOpts.group === g));
}

/* ── Nạp dữ liệu ─────────────────────────────────────────────────────────── */

async function loadStatus() {
  const r = await api('/api/fleet/status');
  state.status = applyProvisionalFleet(r.data || state.status);
  if (r.data) state.stale = r.stale ? r.at : null;
  paintTop(); paintFold(); paintTabs();
}

async function loadProduction() {
  const r = await api(`/api/fleet/production?days=${state.period}`);
  if (r.data) state.prod = r.data;
  const im = await api('/api/fleet/failure-images?days=7&per_machine=3');
  if (im.data) state.images = im.data;
  paintFold(); paintTabs();
}

async function loadStaff() {
  const r = await api('/api/fleet/staff');
  if (r.data) state.staff = r.data;
  paintTabs();
}

async function loadLogs() {
  const a = await api(`/api/fleet/audit?days=7&include_simulated=${state.simulated}`);
  if (a.data) state.audit = a.data;
  const e = await api('/api/fleet/log-errors?top=6');
  if (e.data) state.logerr = e.data;
  paintTabs();
}

/* ── Khởi động ───────────────────────────────────────────────────────────── */

function setLang(l) {
  store.lang = l; localStorage.setItem('fleet_lang', l);
  document.documentElement.lang = l;
  // Vẽ lại từ dữ liệu ĐÃ TẢI, không gọi lại API: mỗi lần bấm EN/VI mà refetch là
  // 5 lượt gọi ra Jetson qua đường chậm chỉ để đổi chữ.
  paintTop(); paintFold(); paintTabs(); chat.mount();
}

function setTheme(x) {
  store.theme = x; localStorage.setItem('fleet_theme', x);
  document.documentElement.setAttribute('data-theme', x);
  paintTop(); paintFold();
}

function boot() {
  document.documentElement.setAttribute('data-theme', store.theme);
  document.documentElement.lang = store.lang;

  $('#btn-en').onclick = () => setLang('en');
  $('#btn-vi').onclick = () => setLang('vi');
  $('#btn-light').onclick = () => setTheme('light');
  $('#btn-dark').onclick = () => setTheme('dark');
  $('#btn-refresh').onclick = async () => {
    await fetch('/api/fleet/refresh', { method: 'POST' });
    loadStatus(); loadProduction();
  };
  ['overview', 'staff', 'frames', 'log'].forEach(k => $('#tab-' + k).onclick = () => {
    state.tab = k; paintTabs();
    if (k === 'staff' && !state.staff) loadStaff();
    if (k === 'log' && !state.audit) loadLogs();
    if (k === 'frames') wall.show(openDrawerByName);
  });
  ['machine', 'dept', 'shift'].forEach(g => $('#g-' + g).onclick = () => {
    state.staffOpts.group = g; paintTabs();
  });
  $('#sim-toggle').onchange = e => {
    state.simulated = e.target.checked; state.audit = null; loadLogs();
  };
  ['chart', 'table'].forEach(v => $('#v-' + v).onclick = () => {
    state.statsView = v;
    ['chart', 'table'].forEach(k => $('#v-' + k).setAttribute('aria-pressed', k === v));
    paintTabs();
  });
  $('#v-chart').setAttribute('aria-pressed', 'true');
  $('#btn-export').onclick = () =>
    chat.askNow(store.lang === 'vi' ? 'Xuất báo cáo so sánh các máy'
                                    : 'Export a comparison report');
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') window.__closeDrawer();
  });

  chat.mount();
  paintTop(); paintTabs();
  loadStatus(); loadProduction();
  setInterval(loadStatus, 30_000);
  setInterval(loadProduction, 300_000);
  setInterval(() => $('#clock').textContent =
    new Date().toLocaleTimeString(store.t.locale), 1000);
}

boot();
