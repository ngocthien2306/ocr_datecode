/* ═══════════════════════════════════════════════════════════════════════════
   Tường ảnh — mọi máy trên một màn hình.

   Vì sao đáng có một màn hình riêng: quản đốc không đi mở bảy cái drawer. Đặt
   ảnh của các line cạnh nhau thì thứ đập vào mắt là SO SÁNH — line nào in mờ
   hơn, line nào lệch khung — mà nhìn từng máy một thì không bao giờ thấy. Nó
   mạnh nhất khi hai line chạy cùng recipe: lúc đó khác biệt giữa hai tấm ảnh
   không còn là do sản phẩm nữa.

   Hai chế độ, cùng nghĩa với trong drawer:
     "Đã kiểm"  ảnh sản phẩm lỗi gần nhất của từng máy (cache 60 giây ở fleet)
     "Camera"   ảnh camera ngay lúc này

   Chế độ camera nhân chi phí lên theo SỐ MÁY, nên nó chậm hơn hẳn trong drawer
   và dừng ngay khi rời tab. Bảy Jetson đang chạy inference cho dây chuyền
   không phải là một dàn camera an ninh.
   ═══════════════════════════════════════════════════════════════════════════ */

import { store, esc, api } from './core.js';

const $ = s => document.querySelector(s);

let timer = null;
let data = null;
let mode = 'inspected';
let fetchedAt = 0;
let onOpen = null;      // bấm một ô → mở drawer của máy đó

const LIVE_MS = 6000;     // camera: 6 giây/khung, nhân cho từng máy
const INSPECT_MS = 60000; // đúng bằng TTL của fleet, hỏi dày hơn cũng chỉ nhận cache

export function stop() {
  clearInterval(timer);
  timer = null;
}

function tile(m) {
  const t = store.t;
  const f = m.frame || {};
  const name = m.machine || '—';

  if (!m.success)
    return `<article class="wtile off"><header>${esc(name)}</header>
      <div class="shot-empty">${esc(m.error || t.noFrame)}</div></article>`;
  // Serial lấy từ `cameras` — nó có kể cả khi máy KHÔNG có lỗi nào gần đây,
  // còn `frame.camera` thì chỉ có khi vừa có sản phẩm trượt.
  const serial = (m.cameras && m.cameras[0]) || f.camera || null;

  if (!m.found && mode === 'inspected')
    return `<article class="wtile off"><header>${esc(name)}</header>
      <div class="shot-empty">${t.noFrameYet}</div></article>`;

  const src = mode === 'live'
    ? (serial ? `/api/fleet/live-frame/${encodeURIComponent(name)}`
        + `?serial=${encodeURIComponent(serial)}&w=460`
        + `&_=${Math.floor(Date.now() / LIVE_MS)}` : null)
    : (f.id ? `/api/fleet/failure-image/${encodeURIComponent(name)}`
        + `/${encodeURIComponent(f.id)}?w=460` : null);

  // Dòng dưới mỗi ô nói ĐÚNG thứ đang xem: ở chế độ camera thì không có
  // "mong / đọc", vì tấm ảnh đó chưa hề được chấm.
  const foot = mode === 'live'
    ? `<span class="live-dot"></span>${t.liveNow}`
    : [String(f.timestamp || '').slice(11, 19),
       f.expected != null ? `${t.expected} ${f.expected}` : null,
       f.recognized != null ? `${t.readAs} ${f.recognized || t.emptyRead}` : null,
      ].filter(Boolean).map(esc).join(' · ');

  return `<article class="wtile ${mode === 'live' ? 'is-live' : ''}"
      data-machine="${esc(name)}" role="button" tabindex="0">
    <header>${esc(name)}
      <span class="wtile-rec">${esc(f.recipe_name || '')}</span></header>
    ${src ? `<img loading="lazy" src="${esc(src)}" alt="${esc(name)}"
        onerror="this.replaceWith(Object.assign(document.createElement('div'),
          {className:'shot-empty', textContent:${JSON.stringify(t.camNoAnswer)}}))">`
          : `<div class="shot-empty">${mode === 'live' ? t.noCamera : '—'}</div>`}
    <footer>${foot}</footer>
  </article>`;
}

function paint() {
  const t = store.t;
  const el = $('#wall-body');
  if (!el) return;
  $('#wall-title').textContent = t.wallTitle;
  $('#wall-hint').textContent = mode === 'live' ? t.liveNote : t.wallHint;
  $('#w-inspected').textContent = t.modeInspected;
  $('#w-live').textContent = t.modeLive;
  $('#w-inspected').setAttribute('aria-pressed', String(mode === 'inspected'));
  $('#w-live').setAttribute('aria-pressed', String(mode === 'live'));

  if (!data) { el.innerHTML = `<div class="coverage">${t.loading}</div>`; return; }
  const rows = data.machines || [];
  el.innerHTML = `<div class="wall">${rows.map(tile).join('')}</div>`
    + `<div class="coverage">${t.wallFoot(rows.length,
        Math.round((Date.now() - fetchedAt) / 1000))}</div>`;

  el.querySelectorAll('.wtile[data-machine]').forEach(c => {
    const go = () => onOpen && onOpen(c.dataset.machine);
    c.onclick = go;
    c.onkeydown = e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); }
    };
  });
}

async function load() {
  const r = await api('/api/fleet/frames');
  if (r.data) { data = r.data; fetchedAt = Date.now(); }
  paint();
}

function setMode(next) {
  if (mode === next) return;
  mode = next;
  restartTimer();
  paint();
}

function restartTimer() {
  stop();
  timer = setInterval(() => {
    if (document.hidden) return;
    // Chế độ camera chỉ cần vẽ lại (URL đổi theo mốc thời gian là ảnh tự tải
    // mới); chế độ đã kiểm mới phải hỏi lại metadata.
    if (mode === 'live') paint(); else load();
  }, mode === 'live' ? LIVE_MS : INSPECT_MS);
}

/** Mở tường ảnh. `open` là hàm để bấm vào một ô thì mở drawer máy đó. */
export function show(openMachine) {
  onOpen = openMachine;
  paint();
  if (!data) load();
  restartTimer();
  $('#w-inspected').onclick = () => setMode('inspected');
  $('#w-live').onclick = () => setMode('live');
}
