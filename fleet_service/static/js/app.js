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
import * as map from './factory-map.js';
import * as V from './views.js';
import * as chat from './chat.js';

const $ = s => document.querySelector(s);
const state = { status: null, prod: null, staff: null, audit: null,
                logerr: null, images: null,
                selected: null, tab: 'overview', statsView: 'chart',
                period: 7, staffOpts: { group: 'machine' }, simulated: false,
                stale: null };

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
  $('#drawer').innerHTML = V.drawerHTML(m, state.prod);
  $('#drawer').classList.add('open');
  paintFold();
}
window.__closeDrawer = () => {
  state.selected = null;
  $('#drawer').classList.remove('open');
  paintFold();
};
window.__askAbout = name => chat.setContext(name);

/* ── Tab ─────────────────────────────────────────────────────────────────── */

function paintTabs() {
  const t = store.t;
  ['overview', 'staff', 'log'].forEach(k =>
    $('#tab-' + k).setAttribute('aria-selected', state.tab === k));
  $('#pane-overview').hidden = state.tab !== 'overview';
  $('#pane-staff').hidden = state.tab !== 'staff';
  $('#pane-log').hidden = state.tab !== 'log';

  if (state.tab === 'overview') {
    $('#stats-title').textContent = t.production;
    $('#fail-title').textContent = t.failures;
    $('#btn-export').textContent = t.export;
    $('#stats-body').innerHTML = V.statsHTML(state.prod, state.statsView);
    $('#fingerprint').innerHTML = V.fingerprintHTML(state.prod);
    $('#fail-body').innerHTML = V.failureGridHTML(state.images);
  }
  if (state.tab === 'staff') {
    $('#staff-body').innerHTML = V.staffHTML(state.staff, state.staffOpts);
    paintStaffFilters();
    $('#staff-body').querySelectorAll('.person').forEach(el => {
      const open = () => chat.setPerson({ name: el.dataset.name,
        username: el.dataset.username, machine: el.dataset.machine });
      el.onclick = open;
      el.onkeydown = e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
      };
    });
  }
  if (state.tab === 'log') {
    $('#log-users-title').textContent = t.userActions;
    $('#log-sys-title').textContent = t.systemErrors;
    $('#sim-label').textContent = t.showSimulated;
    $('#log-users').innerHTML = V.auditHTML(state.audit);
    $('#log-sys').innerHTML = V.logErrorsHTML(state.logerr);
    document.querySelectorAll('[data-ask]').forEach(b =>
      b.onclick = () => chat.askNow(b.dataset.ask));
  }
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
  if (r.data) { state.status = r.data; state.stale = r.stale ? r.at : null; }
  paintTop(); paintFold();
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
  paintTop();
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
  ['overview', 'staff', 'log'].forEach(k => $('#tab-' + k).onclick = () => {
    state.tab = k; paintTabs();
    if (k === 'staff' && !state.staff) loadStaff();
    if (k === 'log' && !state.audit) loadLogs();
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
