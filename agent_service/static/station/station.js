/* ═══════════════════════════════════════════════════════════════════════════
   Line Station — logic màn hình cạnh dây chuyền.

   Bốn luật của bề mặt này, và chúng quyết định gần hết code bên dưới:

   1. KHÔNG BAO GIỜ hiện 0 thay cho "chưa có". "0 sản phẩm, đạt 0%" đọc như máy
      đang hỏng; "0°C" đọc như máy rất mát. Chưa đo được thì là dấu gạch.
   2. Gọi hụt thì GIỮ SỐ CŨ kèm dấu thời gian, không xoá màn hình. Wi-Fi xưởng
      chập là chuyện thường; màn hình trắng thì người đứng máy mất luôn thông tin
      vừa có, còn số cũ kèm "cập nhật 16:42" thì vẫn dùng được.
   3. Không so sánh với line khác. Line này chạy quế, line kia chạy muối.
   4. Không nút nào gây tác dụng phụ. Cả trang chỉ đọc; "bàn giao ca" là ghi
      nhận, không điều khiển máy.

   Mọi số liệu lấy từ CHÍNH máy này (:8100), nên rút mạng ra ngoài thì màn hình
   vẫn đầy đủ.
   ═══════════════════════════════════════════════════════════════════════════ */

const $ = s => document.querySelector(s);
const NA = '—';

const I18N = {
  vi: {
    locale: 'vi-VN',
    gateNote: 'Đăng nhập bằng tài khoản của máy này.',
    signIn: 'Vào màn hình', wrong: 'Sai tên đăng nhập hoặc mật khẩu',
    output: 'Sản lượng ca này', products: 'sản phẩm',
    rate: 'Tỉ lệ đạt', split: (a, b) => `${a} đạt · ${b} không đạt`,
    deltaUp: n => `▲ ${n} điểm so ca trước`,
    deltaDown: n => `▼ ${n} điểm so ca trước`,
    deltaFlat: 'ngang ca trước',
    noPrev: 'chưa có ca trước để so',
    hourly: 'Đạt / không đạt theo giờ trong ca',
    hourlyRange: (a, b, h) => `${a} → ${b} · đã trôi ${h} giờ`,
    hourlyNote: 'Giờ chưa tới để trống, không vẽ cột 0.',
    fails: 'Sản phẩm lỗi gần đây',
    want: 'mong', got: 'đọc', empty: '(trống)',
    noFails: 'Chưa có ảnh sản phẩm lỗi nào trong 24 giờ qua.',
    hw: 'Tình trạng máy', crew: 'Người trong ca',
    cpu: 'CPU', gpu: 'GPU', ram: 'RAM', disk: 'Đĩa', cam: 'Camera',
    camOn: 'camera service chạy', camOff: 'camera service đã dừng',
    hwStale: at => `chưa đo được từ ${at}`,
    measuredAt: at => `đo lúc ${at}`, uptime: 'chạy liên tục',
    hwNone: 'Vòng giám sát phần cứng chưa trả số — máy vẫn chạy.',
    ramWatch: p => `RAM đang ${p}% — theo dõi thêm`,
    diskWatch: p => `Đĩa đang ${p}% — nên dọn bớt`,
    hotWatch: t => `CPU ${t}°C — kiểm tra tản nhiệt`,
    noCrew: 'Không có ai đang trong ca theo khung giờ đã khai.',
    inShift: t => `vào ca ${t}`,
    shiftLabel: (n, a, b) => `Ca ${n} · ${a}–${b}`,
    notStarted: n => `Ca ${n} chưa bắt đầu`,
    prevShift: (p, n) => `ca trước: ${p} sản phẩm, đạt ${n}%`,
    refresh: (at, ok) => ok
      ? `Tự làm mới 15s · cập nhật ${at} · dữ liệu lấy từ chính máy này`
      : `Không gọi được máy · số liệu từ ${at} · sẽ thử lại`,
    handover: 'Bàn giao ca', print: 'In', running: 'đang chạy',
    hoTitle: n => `Bàn giao ca ${n}`,
    hoFail: 'Chưa dựng được bản bàn giao.',
    notConfigured: 'Chưa khai STATION_NAME — màn hình đang lấy tên từ hostname.',
  },
  en: {
    locale: 'en-GB',
    gateNote: "Sign in with this machine's own account.",
    signIn: 'Open station', wrong: 'Wrong username or password',
    output: 'Output this shift', products: 'products',
    rate: 'Pass rate', split: (a, b) => `${a} pass · ${b} fail`,
    deltaUp: n => `▲ ${n} pts vs previous shift`,
    deltaDown: n => `▼ ${n} pts vs previous shift`,
    deltaFlat: 'level with previous shift',
    noPrev: 'no previous shift to compare',
    hourly: 'Pass / fail by hour of shift',
    hourlyRange: (a, b, h) => `${a} → ${b} · ${h} h elapsed`,
    hourlyNote: 'Hours not yet reached are left blank, not drawn as zero.',
    fails: 'Recent failed products',
    want: 'expected', got: 'read', empty: '(empty)',
    noFails: 'No failed-product image in the last 24 hours.',
    hw: 'Machine health', crew: 'Crew on shift',
    cpu: 'CPU', gpu: 'GPU', ram: 'RAM', disk: 'Disk', cam: 'Camera',
    camOn: 'camera service running', camOff: 'camera service stopped',
    hwStale: at => `no reading since ${at}`,
    measuredAt: at => `measured ${at}`, uptime: 'up',
    hwNone: 'Hardware monitor has not reported — the machine is still running.',
    ramWatch: p => `RAM at ${p}% — keep watching`,
    diskWatch: p => `Disk at ${p}% — worth clearing`,
    hotWatch: t => `CPU ${t}°C — check cooling`,
    noCrew: 'Nobody is on shift by the declared hours.',
    inShift: t => `on shift since ${t}`,
    shiftLabel: (n, a, b) => `Shift ${n} · ${a}–${b}`,
    notStarted: n => `Shift ${n} has not started`,
    prevShift: (p, n) => `previous shift: ${p} products, ${n}% pass`,
    refresh: (at, ok) => ok
      ? `Auto-refresh 15s · updated ${at} · data from this machine`
      : `Could not reach the machine · figures from ${at} · will retry`,
    handover: 'Shift handover', print: 'Print', running: 'running',
    hoTitle: n => `Shift ${n} handover`,
    hoFail: 'Could not build the handover.',
    notConfigured: 'STATION_NAME is not set — falling back to hostname.',
  },
};

/* Tiếng Việt là mặc định Ở ĐÂY (Fleet Console mặc định EN): người đứng máy là
   công nhân trong xưởng, còn Fleet Console là màn hình của quản lý. */
const store = {
  lang: localStorage.getItem('station_lang') || 'vi',
  token: localStorage.getItem('station_token') || '',
  get t() { return I18N[this.lang]; },
};

const state = { over: null, fails: null, hw: null, crew: null, cam: null,
                fetchedAt: 0,
                at: null, ok: true };

const esc = s => String(s ?? '').replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = n => n == null ? NA : Number(n).toLocaleString(store.t.locale);
const hhmm = iso => String(iso || '').slice(11, 16);

/* Ngưỡng màu. Giống Fleet Console để cùng một con số không đổi nghĩa giữa hai
   màn hình của cùng nhà máy. */
const rateTone = v => v == null ? 'none' : v >= 92 ? 'ok' : v >= 80 ? 'warn' : 'bad';

/* ── Gọi API ─────────────────────────────────────────────────────────────── */

async function api(path) {
  const r = await fetch(path, { headers: { Authorization: `Bearer ${store.token}` } });
  if (r.status === 401) { logout(); throw new Error('401'); }
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function logout() {
  store.token = '';
  localStorage.removeItem('station_token');
  $('#app').hidden = true;
  $('#gate').hidden = false;
}

async function login(user, pass) {
  const body = new URLSearchParams({ username: user, password: pass });
  const r = await fetch('/api/auth/login', { method: 'POST', body });
  if (!r.ok) return false;
  const d = await r.json();
  if (!d.access_token) return false;
  store.token = d.access_token;
  localStorage.setItem('station_token', d.access_token);
  return true;
}

/* ── Vẽ ──────────────────────────────────────────────────────────────────── */

function paintHeader() {
  const t = store.t, o = state.over;
  $('#lang-vi').setAttribute('aria-pressed', store.lang === 'vi');
  $('#lang-en').setAttribute('aria-pressed', store.lang === 'en');
  /* Đồng hồ lấy giờ CỦA MÁY, không lấy giờ máy đang xem. Mọi con số trên màn
     hình này là giờ của Auto2; để đồng hồ chạy theo trình duyệt thì xem từ xa
     sẽ thấy "10:40" ngay cạnh "06:00 → 09:40" — hai giờ khác nhau cho cùng một
     thời điểm. Giữa hai lần làm mới thì tự đếm thêm để kim không đứng. */
  if (state.over?.generated_at) {
    const base = new Date(state.over.generated_at).getTime();
    const drift = state.fetchedAt ? Date.now() - state.fetchedAt : 0;
    $('#m-clock').textContent = new Date(base + drift)
      .toLocaleTimeString(t.locale, { hour: '2-digit', minute: '2-digit' });
  } else {
    $('#m-clock').textContent = '--:--';
  }
  if (!o) return;

  $('#m-name').textContent = [o.machine?.name, o.machine?.line]
    .filter(Boolean).join(' · ') || NA;
  $('#m-recipe').textContent = o.recipe_name || '';
  $('#m-recipe').hidden = !o.recipe_name;

  const sh = o.shift || {};
  $('#m-shift').textContent = sh.not_started
    ? t.notStarted(sh.name)
    : t.shiftLabel(sh.name, sh.from, sh.to);

  // Trạng thái ở thanh đầu nói về CAMERA SERVICE, và chỉ nói khi biết.
  const st = $('#m-state');
  st.textContent = state.cam === false ? t.camOff
    : state.cam === true ? t.running : '';
  st.className = 'state ' + (state.cam === false ? 'warn'
    : state.cam === true ? 'ok' : '');
}

function paintProduction() {
  const t = store.t, o = state.over;
  $('#t-output').textContent = t.output;
  $('#t-rate').textContent = t.rate;
  if (!o) return;

  const out = o.output || {}, prev = o.previous_shift || {};

  // Ca chưa bắt đầu là một CÂU, không phải con số 0.
  $('#v-output').innerHTML = out.total == null
    ? `<span class="huge none">${NA}</span>`
    : `${fmt(out.total)}<small>${t.products}</small>`;
  $('#v-output').className = 'huge' + (out.total == null ? ' none' : '');
  $('#s-output').textContent = out.total == null
    ? (prev.total != null ? t.prevShift(fmt(prev.total), prev.pass_rate ?? NA) : '')
    : `${o.shift?.from ?? ''} → ${hhmm(o.generated_at)}`;

  const rate = out.pass_rate;
  $('#v-rate').innerHTML = rate == null ? NA
    : `${rate.toString().replace('.', store.lang === 'vi' ? ',' : '.')}<small>%</small>`;
  $('#v-rate').className = 'huge ' + rateTone(rate);

  const d = prev.delta_points;
  const dl = $('#s-delta');
  if (d == null) { dl.textContent = t.noPrev; dl.className = 'sub'; }
  else if (d > 0.05) { dl.textContent = t.deltaUp(Math.abs(d)); dl.className = 'sub up'; }
  else if (d < -0.05) { dl.textContent = t.deltaDown(Math.abs(d)); dl.className = 'sub down'; }
  else { dl.textContent = t.deltaFlat; dl.className = 'sub'; }

  $('#s-split').textContent = out.total == null ? ''
    : t.split(fmt(out.pass), fmt(out.fail));
}

function paintHourly() {
  const t = store.t, o = state.over;
  $('#t-hourly').textContent = t.hourly;
  $('#f-hourly').textContent = t.hourlyNote;
  if (!o) return;
  const rows = o.hourly || [];
  const sh = o.shift || {};
  $('#s-hourly').textContent = t.hourlyRange(sh.from ?? '', hhmm(o.generated_at),
                                             sh.hours_elapsed ?? 0);
  $('#v-hourly').innerHTML = rows.map(r => {
    if (r.pass_rate == null)
      return `<div class="hcol future"><div class="pct">&nbsp;</div>
        <div class="bar"></div><div class="hl">${esc(r.hour)}</div></div>`;
    const h = Math.max(4, Math.round(r.pass_rate * 1.2));
    return `<div class="hcol" title="${esc(r.hour)} · ${fmt(r.total)} ${t.products}">
      <div class="pct">${Math.round(r.pass_rate)}%</div>
      <div class="bar ${rateTone(r.pass_rate)}" style="height:${h}px"></div>
      <div class="hl">${esc(r.hour)}</div></div>`;
  }).join('');
}

function paintFails() {
  const t = store.t;
  $('#t-fails').textContent = t.fails;
  const items = state.fails?.images || [];
  if (!items.length) {
    // Rỗng CÓ LÝ DO, không phải một ô trống im lặng.
    $('#v-fails').innerHTML = `<div class="empty">${t.noFails}</div>`;
    return;
  }
  $('#v-fails').innerHTML = items.map(i => {
    const said = (i.expected != null || i.recognized != null)
      ? `<div class="diff">${t.want} <b>${esc(i.expected ?? NA)}</b> ·
         ${t.got} <b class="got">${esc(i.recognized || t.empty)}</b></div>` : '';
    return `<figure class="fail">
      <img loading="lazy" alt=""
        src="/api/station/image/${encodeURIComponent(i.id)}?w=480&token=${encodeURIComponent(store.token)}">
      <div class="meta"><div class="t">${esc(String(i.timestamp || '').slice(11, 16))}</div>
        ${said}</div></figure>`;
  }).join('');
}

function paintHardware() {
  const t = store.t;
  $('#t-hw').textContent = t.hw;
  const raw = state.hw;
  if (!raw || raw.success === false) {
    $('#v-hw').innerHTML = `<div class="empty">${t.hwNone}</div>`;
    return;
  }
  /* Payload của `get_system_metrics` là LỒNG: cpu.temperature_celsius,
     ram.usage_percent, disk.available_gb… Bản đầu tôi đọc phẳng (h.cpu_temp)
     nên mọi ô hiện dấu gạch trong khi máy đang trả đủ số — đoán hình dạng thay
     vì mở ra xem. Dàn phẳng một lần ở đây, và giữ `null` là `null`. */
  const h = {
    cpu_temp: raw.cpu?.temperature_celsius ?? null,
    cpu_load: raw.cpu?.usage_percent ?? null,
    gpu_temp: raw.gpu?.temperature_celsius ?? null,
    gpu_load: raw.gpu?.usage_percent ?? null,
    ram_percent: raw.ram?.usage_percent ?? null,
    ram_used_gb: raw.ram?.used_gb ?? null,
    ram_total_gb: raw.ram?.total_gb ?? null,
    disk_percent: raw.disk?.usage_percent ?? null,
    disk_free_gb: raw.disk?.available_gb ?? null,
    measured_at: raw.measured_at || null,
    uptime: raw.uptime || null,
    camera_service_running: state.cam,
  };
  // Nhiệt độ `null` giữ nguyên là dấu gạch: máy x86 không có cảm biến kiểu
  // Jetson, và "0°C" nói rằng máy rất mát.
  const row = (k, v, tone, extra) => `<div class="kv">
    <span class="k">${k}</span>
    <span class="v ${tone || ''}">${v}</span>
    ${extra ? `<span class="x">${extra}</span>` : ''}</div>`;
  const deg = v => v == null ? NA : `${Math.round(v)}°C`;
  const pct = v => v == null ? NA : `${Math.round(v)}%`;

  const notes = [];
  if ((h.ram_percent ?? 0) >= 85) notes.push(t.ramWatch(Math.round(h.ram_percent)));
  if ((h.disk_percent ?? 0) >= 85) notes.push(t.diskWatch(Math.round(h.disk_percent)));
  if ((h.cpu_temp ?? 0) >= 85) notes.push(t.hotWatch(Math.round(h.cpu_temp)));

  // Chưa biết thì nói là chưa biết. Bản đầu mặc định "đang chạy" khi thiếu dữ
  // liệu — tức là khẳng định một điều chưa kiểm, trên đúng cái ô mà người vận
  // hành dùng để quyết định có gọi bảo trì hay không.
  const camTxt = h.camera_service_running === true ? t.camOn
    : h.camera_service_running === false ? t.camOff : NA;
  $('#v-hw').innerHTML =
    row(t.cam, camTxt, h.camera_service_running === false ? 'warn' : '')
    + row(t.cpu, deg(h.cpu_temp), (h.cpu_temp ?? 0) >= 85 ? 'bad' : '',
          h.cpu_load == null ? '' : `load ${Math.round(h.cpu_load)}%`)
    + row(t.gpu, deg(h.gpu_temp), '', h.gpu_load == null ? '' : `load ${Math.round(h.gpu_load)}%`)
    + row(t.ram, pct(h.ram_percent), (h.ram_percent ?? 0) >= 85 ? 'warn' : '',
          h.ram_used_gb == null ? '' : `${h.ram_used_gb}/${h.ram_total_gb} GB`)
    + row(t.disk, pct(h.disk_percent), (h.disk_percent ?? 0) >= 85 ? 'warn' : '',
          h.disk_free_gb == null ? '' : `${h.disk_free_gb} GB ${store.lang === 'vi' ? 'trống' : 'free'}`)
    + notes.map(n => `<div class="note">${esc(n)}</div>`).join('')
    + (h.measured_at ? `<div class="foot">${t.measuredAt(hhmm(h.measured_at))}${
        h.uptime ? ` · ${t.uptime} ${esc(h.uptime)}` : ''}</div>` : '');
}

function paintCrew() {
  const t = store.t;
  $('#t-crew').textContent = t.crew;
  const rows = state.crew?.crew || [];
  if (!rows.length) { $('#v-crew').innerHTML = `<div class="empty">${t.noCrew}</div>`; return; }
  const initials = n => String(n || '?').trim().split(/\s+/).slice(-2)
    .map(w => w[0] || '').join('').toUpperCase();
  $('#v-crew').innerHTML = rows.map(p => `<div class="person">
    <span class="av">${esc(initials(p.full_name))}</span>
    <span><span class="nm">${esc(p.full_name)}</span>
      <div class="jt">${esc(p.job_title || '')}${
        p.since ? ` · ${t.inShift(p.since)}` : ''}</div></span>
  </div>`).join('');
}

function paintFooter() {
  const t = store.t;
  $('#f-refresh').textContent = t.refresh(state.at || NA, state.ok);
  $('#btn-handover').textContent = t.handover;
}

function paintAll() {
  paintHeader(); paintProduction(); paintHourly();
  paintFails(); paintHardware(); paintCrew(); paintFooter();
}

/* ── Nạp dữ liệu ─────────────────────────────────────────────────────────── */

async function load() {
  try {
    // Bốn lời gọi song song, tất cả tới CHÍNH máy này — không có bước nào ra
    // mạng ngoài, nên rút mạng thì màn hình vẫn đủ.
    const [o, f, h, c, sv] = await Promise.all([
      api('/api/station/overview'),
      api('/api/station/failures?limit=4').catch(() => null),
      api('/api/station/health-metrics').catch(() => null),
      api('/api/station/crew').catch(() => null),
      api('/api/agent/service/status').catch(() => null),
    ]);
    state.over = o; state.ok = true;
    if (f) state.fails = f;
    if (h) state.hw = h;
    if (c) state.crew = c;
    // Trạng thái camera service là câu trả lời của agent, không phải mặc định.
    if (sv) state.cam = sv.running ?? sv.is_running ?? sv.camera_service_running ?? null;
    state.fetchedAt = Date.now();
    // Dấu thời gian cũng lấy giờ MÁY, cùng lý do với đồng hồ.
    state.at = (o.generated_at || '').slice(11, 19)
      || new Date().toLocaleTimeString(store.t.locale);
  } catch (e) {
    // Giữ nguyên số cũ. Xoá màn hình vì một lần gọi hụt là lấy đi thông tin
    // người đứng máy vừa có.
    state.ok = false;
    console.warn('[station]', e);
  }
  paintAll();
}

/* ── Bàn giao ca ─────────────────────────────────────────────────────────── */

async function openHandover() {
  const t = store.t;
  $('#handover').hidden = false;
  $('#h-title').textContent = t.hoTitle(state.over?.shift?.name ?? '');
  $('#h-print').textContent = t.print;
  $('#h-body').innerHTML = `<div class="skel" style="height:120px"></div>`;
  try {
    const d = await api('/api/station/handover');
    $('#h-body').innerHTML = renderHandover(d);
  } catch {
    $('#h-body').innerHTML = `<div class="empty">${t.hoFail}</div>`;
  }
}

/** Dựng bản bàn giao từ dữ liệu tool — KHÔNG qua mô hình, nên nó vẫn chạy khi
 *  trợ lý tắt hoặc hết credit. Bàn giao ca là việc bắt buộc lúc 22:00. */
function renderHandover(d) {
  if (!d || d.success === false) return `<div class="empty">${store.t.hoFail}</div>`;
  const sec = (title, rows) => rows.length
    ? `<div class="hsec"><h3>${esc(title)}</h3>${rows.map(([k, v]) =>
        `<div class="hrow"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('')}</div>`
    : '';
  const out = [];
  for (const [key, val] of Object.entries(d)) {
    if (key === 'success' || val == null) continue;
    if (typeof val !== 'object') { out.push([key, String(val)]); continue; }
    if (!Array.isArray(val)) {
      const rows = Object.entries(val)
        .filter(([, v]) => v != null && typeof v !== 'object')
        .map(([k, v]) => [k, String(v)]);
      if (rows.length) out.push(...rows.map(([k, v]) => [`${key}.${k}`, v]));
    }
  }
  return sec(store.t.hoTitle(state.over?.shift?.name ?? ''), out)
    || `<pre style="white-space:pre-wrap;font-size:13px">${esc(JSON.stringify(d, null, 2))}</pre>`;
}

/* ── Khởi động ───────────────────────────────────────────────────────────── */

function setLang(l) {
  store.lang = l;
  localStorage.setItem('station_lang', l);
  document.documentElement.lang = l;
  paintAll();     // vẽ lại từ dữ liệu ĐÃ CÓ, không gọi lại máy
}

function boot() {
  document.documentElement.lang = store.lang;
  const t = store.t;
  $('#gate-note').textContent = t.gateNote;
  $('#gate-btn').textContent = t.signIn;

  $('#gate-form').onsubmit = async e => {
    e.preventDefault();
    $('#gate-err').textContent = '';
    const ok = await login($('#gate-user').value, $('#gate-pass').value);
    if (!ok) { $('#gate-err').textContent = store.t.wrong; return; }
    $('#gate').hidden = true; $('#app').hidden = false;
    load();
  };

  $('#lang-vi').onclick = () => setLang('vi');
  $('#lang-en').onclick = () => setLang('en');
  $('#btn-handover').onclick = openHandover;
  $('#h-close').onclick = () => { $('#handover').hidden = true; };
  $('#h-print').onclick = () => window.print();
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') $('#handover').hidden = true;
  });

  if (!store.token) { $('#gate').hidden = false; return; }
  $('#app').hidden = false;
  paintAll();
  load();
  // 15 giây, không cần bấm. Dừng khi màn hình bị che để không hỏi máy vô ích.
  setInterval(() => { if (!document.hidden) load(); }, 15_000);
  setInterval(paintHeader, 1000);
}

boot();
