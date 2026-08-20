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
  const initialChips = history.length ? [] : t.chips.slice(0, 2);
  $('#chips').hidden = initialChips.length === 0;
  $('#chips').innerHTML = initialChips
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
   một trang tĩnh chỉ để render bảng và chữ đậm. Tách cấu trúc trước rồi mới
   escape từng nội dung; escape cả chuỗi ngay từ đầu sẽ khiến `###` thành text
   thường và bảng tab-separated không có cách nhận ra cột. */
function mini(md) {
  const inline = value => esc(value)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  const tableCells = line => {
    const clean = line.trim();
    if (!clean) return null;
    if (clean.includes('|')) {
      const cells = clean.replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim());
      return cells.length > 1 ? cells : null;
    }
    const cells = clean.split('\t').map(cell => cell.trim());
    return cells.length > 1 ? cells : null;
  };
  const divider = cells => cells.every(cell => /^:?-{3,}:?$/.test(cell));
  const tableCell = value => /^(critical|error|warning|info)$/i.test(value.trim())
    ? `<span class="chat-level ${value.trim().toLowerCase()}">${inline(value)}</span>`
    : inline(value);
  const lines = String(md || '').replace(/\r/g, '').split('\n');
  const out = [];
  let table = null, list = null;
  const flushTable = () => {
    if (!table?.length) { table = null; return; }
    const [head, ...body] = table;
    out.push(`<div class="chat-table-wrap"><table class="chat-table"><thead><tr>${head.map(cell =>
      `<th>${tableCell(cell)}</th>`).join('')}</tr></thead><tbody>${body.map(row =>
      `<tr>${row.map(cell => `<td>${tableCell(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
    table = null;
  };
  const flushList = () => {
    if (list?.length) out.push(`<ul class="chat-list">${list.map(item => `<li>${inline(item)}</li>`).join('')}</ul>`);
    list = null;
  };
  for (const raw of lines) {
    const cells = tableCells(raw);
    if (cells) {
      flushList();
      if (!divider(cells)) (table ||= []).push(cells);
      continue;
    }
    flushTable();
    const heading = raw.match(/^\s*(#{1,3})\s+(.+)$/);
    if (heading) {
      flushList();
      const tag = heading[1].length === 1 ? 'h2' : 'h3';
      out.push(`<${tag} class="chat-heading">${inline(heading[2])}</${tag}>`);
      continue;
    }
    const bullet = raw.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) { (list ||= []).push(bullet[1]); continue; }
    flushList();
    if (raw.trim()) out.push(`<p>${inline(raw)}</p>`);
  }
  flushTable(); flushList();
  return out.join('');
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
  // Chỉ cần gợi ý global trước câu đầu. Sau đó để phần trả lời dùng gợi ý theo
  // ngữ cảnh, tránh 2 cụm câu hỏi lặp nhau chiếm hết cửa sổ chat.
  $('#chips').hidden = true;
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
    const suggestions = (d.suggestions || []).slice(0, 2);
    const sg = suggestions.length
      ? `<div class="sugg">${suggestions.map(s =>
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
