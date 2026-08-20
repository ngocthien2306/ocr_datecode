/* ═══════════════════════════════════════════════════════════════════════════
   FactoryMap — sơ đồ mặt bằng đẳng cự (bậc 1).

   Interface `render({machines, selected, onSelect})` giữ nguyên cho cả bậc 2
   (three.js): đổi bậc chỉ là thay ruột module này, không đụng chỗ gọi.

   Vì sao bậc 1 là mặc định: 5 máy trên một mặt sàn thì góc nhìn đẳng cự cố định
   truyền đạt đủ vị trí, mà không kéo ~600 KB three.js + WebGL lên một tablet
   đặt cạnh dây chuyền. Bậc 2 dành cho Fleet Console trên màn hình lớn.
   ═══════════════════════════════════════════════════════════════════════════ */

import { store, esc } from './core.js';

// Phép chiếu đẳng cự: 1 ô lưới = TILE, trục x đi chéo phải-xuống, y chéo trái-xuống.
//
// Ô phải đủ RỘNG cho nhãn nằm dưới khối. Bản đầu dùng TILE_W=108 nên bước ngang
// chỉ 54px trong khi nhãn rộng ~90px — năm nhãn chồng lên nhau thành một đống
// chữ không đọc được. Nhãn là ràng buộc, không phải khối hộp.
const TILE_W = 230, TILE_H = 112, BOX_H = 40;
const OX = 0, OY = 0;

const iso = (x, y) => [OX + (x - y) * TILE_W * 0.5,
                       OY + (x + y) * TILE_H * 0.5];

/* Bốn trạng thái phải phân biệt bằng CẢ HÌNH LẪN MÀU, không chỉ màu: xưởng có
   người mù màu, và màn hình cạnh dây chuyền thường bị chói.                   */
const STYLE = {
  ok:          { fill: '#cfe0f0', side: '#a9c4de', edge: '#7fa0c0', dash: '', dim: 0 },
  warn:        { fill: '#f6e7c8', side: '#e2cb9d', edge: '#9a6a00', dash: '', dim: 0 },
  agent_down:  { fill: '#e4e6e9', side: '#cfd3d8', edge: '#98989b', dash: '', dim: 1 },
  unreachable: { fill: '#eceef0', side: '#dfe2e5', edge: '#98989b', dash: '5 4', dim: 1 },
  offline:     { fill: '#eceef0', side: '#dfe2e5', edge: '#98989b', dash: '5 4', dim: 1 },
};

function box(cx, cy, s, selected) {
  const w = TILE_W * 0.44, h = TILE_H * 0.44, z = BOX_H;
  const top = `${cx},${cy - h - z} ${cx + w},${cy - z} ${cx},${cy + h - z} ${cx - w},${cy - z}`;
  const left = `${cx - w},${cy - z} ${cx},${cy + h - z} ${cx},${cy + h} ${cx - w},${cy}`;
  const right = `${cx + w},${cy - z} ${cx},${cy + h - z} ${cx},${cy + h} ${cx + w},${cy}`;
  const lift = selected ? ' transform="translate(0,-7)"' : '';
  return `<g${lift}${s.dim ? ' opacity=".62"' : ''}>
    ${selected ? `<ellipse cx="${cx}" cy="${cy + h + 8}" rx="${w * 1.1}" ry="9"
        fill="#1d1f20" opacity=".13"/>` : ''}
    <polygon class="side" points="${left}"  fill="${s.side}" stroke="${s.edge}"
      stroke-width="${selected ? 2 : 1}" stroke-dasharray="${s.dash}"/>
    <polygon class="side" points="${right}" fill="${s.side}" stroke="${s.edge}"
      stroke-width="${selected ? 2 : 1}" stroke-dasharray="${s.dash}"/>
    <polygon class="top"  points="${top}"   fill="${s.fill}" stroke="${s.edge}"
      stroke-width="${selected ? 2 : 1}" stroke-dasharray="${s.dash}"/>
  </g>`;
}

/* Dấu hiệu hình học đi kèm màu — mỗi trạng thái một hình khác nhau. */
function marker(cx, cy, state) {
  const y = cy - BOX_H - TILE_H * 0.44 - 14;
  if (state === 'ok')
    return `<circle cx="${cx}" cy="${y}" r="4" fill="#2f7d4f"/>`;
  if (state === 'warn')
    return `<g><polygon points="${cx},${y - 7} ${cx + 6},${y + 4} ${cx - 6},${y + 4}"
        fill="none" stroke="#9a6a00" stroke-width="1.6"/>
      <circle cx="${cx}" cy="${y + 1}" r="1.2" fill="#9a6a00"/></g>`;
  if (state === 'agent_down')  // khung chat gạch chéo
    return `<g stroke="#98989b" stroke-width="1.4" fill="none">
      <rect x="${cx - 7}" y="${y - 6}" width="14" height="10" rx="2"/>
      <line x1="${cx - 8}" y1="${y + 5}" x2="${cx + 8}" y2="${y - 7}"/></g>`;
  return `<g stroke="#98989b" stroke-width="1.4">
    <line x1="${cx - 6}" y1="${y - 6}" x2="${cx + 6}" y2="${y + 6}"/>
    <line x1="${cx + 6}" y1="${y - 6}" x2="${cx - 6}" y2="${y + 6}"/></g>`;
}

function warehouse(x, y, label) {
  const [cx, cy] = iso(x, y);
  const s = { fill: '#e4e6e9', side: '#d2d5d9', edge: '#b7b7ba', dash: '', dim: 0 };
  return `<g opacity=".9">${box(cx, cy, s, false)}
    <text x="${cx}" y="${cy + 30}" text-anchor="middle"
      font-size="10" letter-spacing="1.4" fill="#98989b">${esc(label)}</text></g>`;
}

export function render(el, { machines, selected, onSelect }) {
  const t = store.t;
  const placed = machines.filter(m => m.floor);
  const xs = placed.map(m => m.floor.x), ys = placed.map(m => m.floor.y);
  const maxX = Math.max(1, ...xs), maxY = Math.max(1, ...ys);

  // Lưới sàn
  const G0 = -0.8, gx1 = maxX + 1.0, gy1 = maxY + 1.0;
  let grid = '';
  for (let i = G0; i <= gx1 + 1e-6; i += 0.5) {
    const [x1, y1] = iso(i, G0), [x2, y2] = iso(i, gy1);
    grid += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
  }
  for (let j = G0; j <= gy1 + 1e-6; j += 0.5) {
    const [x1, y1] = iso(G0, j), [x2, y2] = iso(gx1, j);
    grid += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
  }

  const [ax1, ay1] = iso(G0 + 0.2, (maxY + 1) / 2);
  const [ax2, ay2] = iso(gx1 - 0.2, (maxY + 1) / 2);

  const wh = [[G0 - 0.3, 0.2, 'RAW MATERIALS'], [gx1 + 0.3, maxY - 0.2, 'FINISHED GOODS']];

  /* Thứ tự vẽ phải theo CHIỀU SÂU (x+y tăng dần), không theo thứ tự trong
     machines.json: ở góc đẳng cự, vật ở gần phải nằm trên vật ở xa. Vẽ theo thứ
     tự file thì M1 đè lên Auto2 và trông như hai khối cắm vào nhau.            */
  const depth = placed.slice().sort((a, b) =>
    (a.floor.x + a.floor.y) - (b.floor.x + b.floor.y));

  const solids = depth.map(m => {
    const [cx, cy] = iso(m.floor.x, m.floor.y);
    const s = STYLE[m.state] || STYLE.unreachable;
    return `<g class="mach" data-node="${esc(m.node_id)}">
      <title>${esc(m.name)} — ${esc(t.state[m.state] || m.state)}</title>
      ${box(cx, cy, s, m.node_id === selected)}
      ${marker(cx, cy, m.state)}
    </g>`;
  }).join('');

  /* Nhãn vẽ ở LƯỢT RIÊNG sau tất cả khối. Nhãn nằm dưới chân khối của nó, tức
     rơi đúng vào chỗ khối hàng sau chiếm — trộn chung một lượt thì tên máy bị
     nuốt mất, và tên máy là thứ người ta tìm đầu tiên trên sơ đồ.              */
  const labels = placed.map(m => {
    const [cx, cy] = iso(m.floor.x, m.floor.y);
    const w = 62, y = cy + 22;
    return `<g class="mach lbl" data-node="${esc(m.node_id)}">
      <rect x="${cx - w}" y="${y}" width="${w * 2}" height="34" rx="5"
        fill="var(--surface)" opacity=".88"/>
      <text x="${cx}" y="${y + 14}" text-anchor="middle" font-size="13"
        font-weight="600" fill="currentColor">${esc(m.name)}</text>
      <text x="${cx}" y="${y + 28}" text-anchor="middle" font-size="10"
        fill="#98989b">${esc(m.line || '')}</text>
    </g>`;
  }).join('');

  /* viewBox ôm đúng thứ ĐÃ VẼ — kể cả bề rộng nhãn. Bản trước chỉ lấy bounds của
     lưới nên nhãn "LineTine" thò ra ngoài khung và bị cắt cụt.                 */
  const pts = [iso(G0, G0), iso(gx1, G0), iso(G0, gy1), iso(gx1, gy1)];
  placed.forEach(m => {
    const [cx, cy] = iso(m.floor.x, m.floor.y);
    pts.push([cx - 66, cy - BOX_H - TILE_H * 0.44 - 22], [cx + 66, cy + 60]);
  });
  wh.forEach(([x, y]) => {
    const [cx, cy] = iso(x, y);
    pts.push([cx - 62, cy - BOX_H - TILE_H * 0.44], [cx + 62, cy + 36]);
  });
  const pad = 14;
  const bx = Math.min(...pts.map(p => p[0])) - pad;
  const by = Math.min(...pts.map(p => p[1])) - pad;
  const bw = Math.max(...pts.map(p => p[0])) + pad - bx;
  const bh = Math.max(...pts.map(p => p[1])) + pad - by;

  el.innerHTML = `
    <svg id="factory" viewBox="${bx} ${by} ${bw} ${bh}" role="img"
         aria-label="${esc(t.floorTitle)}">
      <g stroke="#d4d4d7" stroke-width=".6" opacity=".55">${grid}</g>
      <line x1="${ax1}" y1="${ay1}" x2="${ax2}" y2="${ay2}"
        stroke="#b7b7ba" stroke-width="1" stroke-dasharray="6 5"/>
      <text x="${(ax1 + ax2) / 2}" y="${(ay1 + ay2) / 2 - 6}" text-anchor="middle"
        font-size="9.5" letter-spacing="2" fill="#b7b7ba">AISLE</text>
      ${wh.map(w => warehouse(w[0], w[1], w[2])).join('')}
      ${solids}
      ${labels}
    </svg>`;

  el.querySelectorAll('.mach').forEach(g =>
    g.addEventListener('click', () => onSelect(g.dataset.node)));
}

export function legendHTML() {
  const t = store.t;
  const dot = c => `<svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="${c}"/></svg>`;
  const tri = `<svg width="12" height="12"><polygon points="6,1 11,10 1,10" fill="none"
    stroke="#9a6a00" stroke-width="1.5"/></svg>`;
  const chat = `<svg width="14" height="12" stroke="#98989b" stroke-width="1.3" fill="none">
    <rect x="1" y="1" width="12" height="8" rx="2"/><line x1="0" y1="11" x2="14" y2="0"/></svg>`;
  const cross = `<svg width="12" height="12" stroke="#98989b" stroke-width="1.4">
    <line x1="2" y1="2" x2="10" y2="10"/><line x1="10" y1="2" x2="2" y2="10"/></svg>`;
  return `
    <span class="k">${dot('#2f7d4f')} ${t.state.ok}</span>
    <span class="k">${tri} ${t.state.warn}</span>
    <span class="k">${chat} ${t.state.agent_down}</span>
    <span class="k">${cross} ${t.state.unreachable}</span>`;
}
