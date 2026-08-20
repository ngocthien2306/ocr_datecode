/* ═══════════════════════════════════════════════════════════════════════════
   Panel chat với fleet_orchestrator.

   Lịch sử giữ ở client và gửi kèm mỗi lượt: trung tâm không giữ session, nên
   tắt tab là hết — đúng nguyên tắc fleet chỉ đọc và không phải thành phần thiết
   yếu của dây chuyền.
   ═══════════════════════════════════════════════════════════════════════════ */

import { store, esc, post } from './core.js';

const history = [];
/* Ngữ cảnh gắn kèm câu hỏi. Trước đây chỉ là tên máy; giờ là {label, hint} để
   gắn được cả một CON NGƯỜI — câu "người này đã làm gì" chỉ trả lời đúng khi
   agent biết cả username lẫn máy, chứ nguyên cái tên đầy đủ thì không tra được
   bảng audit. */
let context = null;          // {label, hint} | null
const $ = s => document.querySelector(s);

export function mount() {
  const t = store.t;
  $('#chat-title').textContent = t.chatTitle;
  $('#chat-hint').textContent = t.chatHint;
  $('#btn-send').textContent = t.send;
  $('#q').placeholder = t.placeholder;
  $('#chips').innerHTML = t.chips
    .map(c => `<button type="button">${esc(c)}</button>`).join('');
  $('#chips').onclick = e => {
    if (e.target.tagName === 'BUTTON') { $('#q').value = e.target.textContent; send(); }
  };
  $('#chat > header').onclick = e => {
    if (e.target.tagName === 'BUTTON') return;
    $('#chat').classList.toggle('min');
    $('#chat-caret').textContent = $('#chat').classList.contains('min') ? '▲' : '▼';
  };
  $('#ask').onsubmit = e => { e.preventDefault(); send(); };
  renderCtx();
}

/** Ngữ cảnh gắn sẵn hiện thành chip — gắn ngầm mà không hiện ra thì người dùng
 *  không hiểu vì sao câu trả lời chỉ nói về một máy. */
export function setContext(machine) {
  context = { label: machine, hint: store.t.ctxMachine(machine) };
  renderCtx();
  $('#chat').classList.remove('min');
  $('#chat-caret').textContent = '▼';
  $('#q').focus();
}

/** Gắn ngữ cảnh vào MỘT NGƯỜI và mồi sẵn câu hỏi — không tự gửi, để người dùng
 *  còn sửa được trước khi hỏi. */
export function setPerson({ name, username, machine }) {
  context = { label: `${name} · ${machine}`,
              hint: store.t.ctxPerson(name, username, machine) };
  renderCtx();
  $('#chat').classList.remove('min');
  $('#chat-caret').textContent = '▼';
  $('#q').value = store.t.askStaff(name, username, machine);
  $('#q').focus();
}

function renderCtx() {
  const el = $('#ctxchip');
  if (!context) { el.innerHTML = ''; return; }
  el.innerHTML = `${store.t.askingAbout} <b>${esc(context.label)}</b>
    <button style="padding:1px 6px;font-size:10px;margin-left:6px">✕</button>`;
  el.querySelector('button').onclick = () => { context = null; renderCtx(); };
}

export function askNow(text, machine) {
  if (machine) setContext(machine);
  $('#q').value = text;
  send();
}

/* Markdown tối giản: đủ cho thứ agent trả về. Không kéo cả thư viện markdown vào
   một trang tĩnh chỉ để render bảng và chữ đậm. */
function mini(md) {
  const lines = esc(md).split('\n');
  let out = '', tbl = null;
  const flush = () => { if (tbl) { out += `<table>${tbl}</table>`; tbl = null; } };
  for (const ln of lines) {
    if (/^\s*\|/.test(ln)) {
      if (/^\s*\|[\s|:-]+\|\s*$/.test(ln)) continue;
      const cells = ln.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
      const tag = tbl === null ? 'th' : 'td';
      tbl = (tbl || '') + `<tr>${cells.map(c => `<${tag}>${c}</${tag}>`).join('')}</tr>`;
    } else { flush(); out += ln ? `<div>${ln}</div>` : '<div style="height:5px"></div>'; }
  }
  flush();
  return out.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`(.+?)`/g, '<code>$1</code>');
}

function bubble(cls, html) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.innerHTML = html;
  $('#log').appendChild(d);
  $('#log').scrollTop = $('#log').scrollHeight;
  return d;
}

async function send() {
  const t = store.t;
  const raw = $('#q').value.trim();
  if (!raw) return;
  const q = context ? `${raw} (${context.hint})` : raw;
  $('#chat').classList.remove('min');
  $('#chat-caret').textContent = '▼';
  $('#q').value = '';
  bubble('u', esc(raw));
  const wait = bubble('a', `<span class="na">${t.thinking}</span>`);
  try {
    const d = await post('/api/fleet/chat', { message: q, history, lang: store.lang });
    if (d.detail) { wait.className = 'msg a err'; wait.textContent = d.detail; return; }
    // Link tải do SERVER đưa, không do mô hình viết: cho mô hình thấy tên file
    // thì nó bịa ra đường dẫn và người dùng bấm vào một link không tồn tại.
    const dl = d.file
      ? `<a class="dl" href="${esc(d.file.url)}" download>⬇ ${esc(d.file.name)}</a>` : '';
    // Gợi ý cũng do server dựng từ số liệu, không do mô hình tự nghĩ.
    const sg = (d.suggestions || []).length
      ? `<div class="sugg">${d.suggestions.map(s =>
          `<button type="button">${esc(s)}</button>`).join('')}</div>` : '';
    wait.innerHTML = mini(d.response || '') + dl +
      `<div class="meta">${t.usedTools}: ${(d.tool_calls || []).join(', ') || '—'}` +
      (d.attachments?.length ? ` · ${d.attachments.length} ${t.fromMachines}` : '') +
      '</div>' + sg;
    wait.querySelectorAll('.sugg button').forEach(b =>
      b.onclick = () => { $('#q').value = b.textContent; send(); });
    history.push({ role: 'user', content: q },
                 { role: 'assistant', content: d.response || '' });
  } catch (e) {
    wait.className = 'msg a err';
    wait.textContent = String(e);
  }
}
