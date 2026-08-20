/* ═══════════════════════════════════════════════════════════════════════════
   Khung ảnh sản phẩm trong drawer.

   Hai chế độ, và phải gọi đúng tên vì chúng trả lời hai câu khác nhau:

     "Đã kiểm"  ảnh sản phẩm LỖI gần nhất, đặt cạnh ẢNH TEMPLATE đang chạy lúc
                nó bị chụp. Đo trên M2: 402/402 sản phẩm fail có lưu ảnh, còn
                0/49 sản phẩm đạt có ảnh — pipeline chỉ ghi ảnh cho frame fail.
                Nên vế đối chiếu là template, tức chính thứ hệ thống đem ra so,
                chứ không phải "ảnh đạt gần nhất" (thứ không tồn tại).

     "Camera"   ảnh camera NGAY LÚC NÀY. Có thể là băng tải trống. Giữ lại vì
                nó cho thấy hai thứ ảnh inference không cho thấy: góc đặt
                camera và ánh sáng.

   Nhịp 60 giây do MÁY CHỦ giữ (queries.FRAME_TTL), không phải do chỗ này. Ở
   đây chỉ hỏi lại; ba tab cùng mở thì vẫn chỉ một lần chạm tới Jetson.
   ═══════════════════════════════════════════════════════════════════════════ */

import { store, esc, api } from './core.js';

const $ = s => document.querySelector(s);

let timer = null;        // vòng làm mới ảnh "đã kiểm"
let liveTimer = null;    // vòng làm mới ảnh camera
let current = null;      // {machine, mode, template, data, fetchedAt}

/** Dừng mọi vòng lặp. Gọi khi đóng drawer — để chạy tiếp là vẫn kéo ảnh về
 *  cho một máy không ai còn nhìn. */
export function stop() {
  clearInterval(timer); clearInterval(liveTimer);
  timer = liveTimer = null; current = null;
}

function ago(sec) {
  const t = store.t;
  if (sec == null) return '—';
  if (sec < 60) return t.secAgo(Math.round(sec));
  if (sec < 3600) return t.minAgo(Math.round(sec / 60));
  return t.hourAgo(Math.round(sec / 3600));
}

function shot(src, caption, badge, cls = '') {
  // Ảnh hỏng phải NÓI RA. Icon ảnh vỡ của trình duyệt trông như giao diện lỗi,
  // trong khi sự thật thường là camera của máy đó không trả frame — đo được
  // trên M1: /api/cameras/.../frame treo 8 giây rồi timeout.
  const bad = `this.replaceWith(Object.assign(document.createElement('div'),
    {className:'shot-empty', textContent:${JSON.stringify(store.t.camNoAnswer)}}))`;
  return `<figure class="shot ${cls}">
    ${src ? `<img loading="lazy" src="${esc(src)}" alt="${esc(caption)}"
              onerror="${bad}">`
          : `<div class="shot-empty">—</div>`}
    ${badge ? `<figcaption class="shot-badge">${esc(badge)}</figcaption>` : ''}
    <figcaption class="shot-cap">${esc(caption)}</figcaption>
  </figure>`;
}

function body(machine, d, mode) {
  const t = store.t;
  if (!d || d.success === false)
    return `<div class="muted" style="padding:10px 0">${esc(d?.error || t.noFrame)}</div>`;
  if (!d.found)
    return `<div class="muted" style="padding:10px 0">${t.noFrameYet}</div>`;

  const f = d.frame || {}, tpl = d.template;

  /* Một camera có thể soi NHIỀU vị trí, mỗi vị trí một template (đo trên M2:
     cùng camera 40767173 có Frame 2/3/4). Chia lỗi theo template chính là câu
     trả lời cho "hỏng cả camera hay hỏng một vị trí": 89% dồn vào Frame 4 thì
     đi chỉnh đúng vị trí đó, không phải tháo cả camera.
     Vẫn KHÔNG ghép ảnh lỗi với template khác — mỗi frame do đúng một template
     chấm, ghép chéo là nói sai. */
  const rows = d.templates || [];
  const chips = rows.length > 1 ? `
    <div class="tpl-split">
      ${rows.map(r => `<button type="button" class="tpl-chip"
        data-tpl="${esc(r.template_name)}"
        aria-pressed="${r.template_name === (current?.template ?? (tpl?.name || ''))}"
        title="${esc(t.camShort(r.camera))}">
        <b>${esc(r.template_name)}</b>
        <span>${r.share_pct}%</span>
        <i>${r.fails}</i>
      </button>`).join('')}
      ${current?.template ? `<button type="button" class="tpl-chip clear"
        data-tpl="">${t.allFrames}</button>` : ''}
    </div>` : '';
  const age = f.age_seconds == null ? null
    : f.age_seconds + (Date.now() - (current?.fetchedAt || Date.now())) / 1000;

  if (mode === 'live') {
    // Cache-buster theo giây: ảnh camera là no-store, nhưng vẫn phải đổi URL,
    // không thì trình duyệt dùng lại chính thẻ <img> cũ.
    const serial = (d.cameras && d.cameras[0]) || f.camera || null;
    const src = serial
      ? `/api/fleet/live-frame/${encodeURIComponent(machine)}`
        + `?serial=${encodeURIComponent(serial)}&w=560&_=${Math.floor(Date.now() / 1000)}`
      : null;
    return `<div class="shots one">${shot(src, t.liveNow, t.liveBadge, 'live')}</div>
      <div class="shot-note">${t.liveNote}</div>`;
  }

  const failSrc = f.id
    ? `/api/fleet/failure-image/${encodeURIComponent(machine)}/${encodeURIComponent(f.id)}?w=520`
    : null;
  const tplSrc = tpl?.url
    ? `/api/fleet/template/${encodeURIComponent(machine)}?p=${encodeURIComponent(tpl.url)}&w=520`
    : null;

  // Dòng "mong → đọc" chỉ hiện khi lỗi LÀ sai chuỗi. Nguyên nhân khác (không
  // thấy vùng, lệch template) thì hai ô này rỗng, và in ra "mong: null" chỉ
  // làm người xem tưởng hỏng dữ liệu.
  const diff = (f.expected != null || f.recognized != null) ? `
    <div class="shot-diff">
      <span class="want">${t.expected}: <b>${esc(f.expected ?? '—')}</b></span>
      <span class="got">${t.readAs}: <b>${esc(f.recognized || t.emptyRead)}</b></span>
    </div>` : '';

  const conf = f.confidence != null
    ? `<span class="pill">conf ${(f.confidence * 100).toFixed(1)}%</span>` : '';

  return `
    ${chips}
    <div class="shots">
      ${shot(failSrc, `${t.failFrame} · ${String(f.timestamp || '').slice(11, 19)}`,
             'FAIL', 'bad')}
      ${shot(tplSrc, tpl ? `${t.template} · ${tpl.name || ''}` : t.noTemplate,
             tpl ? 'TEMPLATE' : '', 'ref')}
    </div>
    ${diff}
    <div class="shot-note">
      ${conf}
      <span>${esc(f.recipe_name || '')}</span>
      <span>${ago(age)}</span>
      ${tpl?.loaded_by ? `<span>${t.tplLoaded(tpl.loaded_at || '', tpl.loaded_by)}</span>` : ''}
      ${d.cached ? `<span class="muted">${t.serverCached}</span>` : ''}
    </div>`;
}

function paint() {
  const el = $('#frame-panel');
  if (!el || !current) return;
  const t = store.t;
  el.innerHTML = `
    <div class="panel-head">
      <h2>${t.productImage}</h2>
      <span class="spacer"></span>
      <div class="seg">
        <button type="button" data-mode="inspected"
          aria-pressed="${current.mode === 'inspected'}">${t.modeInspected}</button>
        <button type="button" data-mode="live"
          aria-pressed="${current.mode === 'live'}">${t.modeLive}</button>
      </div>
    </div>
    <div class="frame-body">${body(current.machine, current.data, current.mode)}</div>`;
  el.querySelectorAll('button[data-mode]').forEach(b =>
    b.onclick = () => setMode(b.dataset.mode));
  el.querySelectorAll('button[data-tpl]').forEach(b =>
    b.onclick = () => pickTemplate(b.dataset.tpl || null));
}

function setMode(mode) {
  if (!current || current.mode === mode) return;
  current.mode = mode;
  clearInterval(liveTimer); liveTimer = null;
  if (mode === 'live') {
    /* 3 giây một khung. Đây là máy đang chạy inference cho dây chuyền, nên xem
       trực tiếp cũng phải có chừng mực — 0,3 hình/giây đủ để chỉnh góc camera
       mà không giành CPU với việc chính của máy. */
    liveTimer = setInterval(paint, 3000);
  }
  paint();
}

function pickTemplate(name) {
  if (!current || current.template === name) return;
  current.template = name;
  current.mode = 'inspected';
  paint();
  load(current.machine);
}

async function load(machine) {
  const q = current?.template
    ? `?template=${encodeURIComponent(current.template)}` : '';
  const r = await api(`/api/fleet/frame/${encodeURIComponent(machine)}${q}`);
  if (!current || current.machine !== machine) return;
  current.data = r.data || current.data;
  current.fetchedAt = Date.now();
  paint();
}

/** Mở khung ảnh cho một máy. Gọi lại với máy khác thì tự chuyển. */
export function open(machine) {
  stop();
  current = { machine, mode: 'inspected', template: null, data: null,
               fetchedAt: Date.now() };
  paint();
  load(machine);
  // Hỏi lại đúng bằng TTL của máy chủ: hỏi dày hơn thì chỉ nhận lại bản cache.
  timer = setInterval(() => {
    if (!document.hidden) load(machine);
  }, 60_000);
}
