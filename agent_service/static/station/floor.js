/* ═══════════════════════════════════════════════════════════════════════════
   Sơ đồ vị trí cho Line Station.

   Cùng ý với sơ đồ ở Fleet Console, nhưng luật khác hẳn ở một điểm: ở đây CHỈ
   máy của chính station là thao tác được. Các máy khác vẽ xám và — quan trọng
   nhất — KHÔNG có đèn trạng thái.

   Bỏ đèn của máy khác là cố ý, không phải quên: trạng thái line khác không phải
   việc của người đang đứng ở đây, và hiện lên là mời so sánh sai. Line này chạy
   quế, line kia chạy muối; hai đèn cạnh nhau thì mắt tự so, mà so là sai.

   Dùng SVG đẳng cự (~vài KB) chứ không three.js: màn hình cạnh dây chuyền hay
   là tablet cũ, và một sơ đồ định vị không đáng kéo 600 KB WebGL. Bản 3D là
   việc của Fleet Console trên màn hình lớn.
   ═══════════════════════════════════════════════════════════════════════════ */

import { planCells } from '/shared/factory-plan.js';

const TILE_W = 210, TILE_H = 104, BOX_H = 34;
const iso = (x, y) => [(x - y) * TILE_W * 0.5, (x + y) * TILE_H * 0.5];
const pt = (x, y) => iso(x, y).map(n => n.toFixed(1)).join(',');

/* Hai bảng vật liệu RIÊNG, không phải một bảng rồi giảm opacity: giảm opacity
   làm mất luôn chiều sâu của khối, và một khối bẹt thì không còn đọc ra là máy. */
const ACTIVE = { top: '#cfe0f0', l: '#a9c4de', r: '#93b3d1', edge: '#5980a6' };
const MUTED = { top: '#e4e6e9', l: '#d2d5d9', r: '#c6cacd', edge: '#b7b7ba' };

const esc = s => String(s ?? '').replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function box(cx, cy, m, lift = 0) {
  const w = TILE_W * 0.42, h = TILE_H * 0.42, z = BOX_H + lift;
  return `<g>
    <polygon points="${cx - w},${cy - z} ${cx},${cy + h - z} ${cx},${cy + h} ${cx - w},${cy}"
      fill="${m.l}" stroke="${m.edge}" stroke-width="1"/>
    <polygon points="${cx + w},${cy - z} ${cx},${cy + h - z} ${cx},${cy + h} ${cx + w},${cy}"
      fill="${m.r}" stroke="${m.edge}" stroke-width="1"/>
    <polygon points="${cx},${cy - h - z} ${cx + w},${cy - z} ${cx},${cy + h - z} ${cx - w},${cy - z}"
      fill="${m.top}" stroke="${m.edge}" stroke-width="1"/>
  </g>`;
}

/** Ba dấu hiệu cho máy ĐANG BẬT — ba thứ khác nhau, vì màn hình xưởng bị chói
 *  và người xem có thể nghiêng góc: vòng thép trên sàn, bốn dấu chuẩn ở góc, và
 *  cột sáng dựng đứng. Một dấu hiệu đơn lẻ rất dễ mất trong điều kiện đó. */
/* Vòng thép nằm TRÊN SÀN nên vẽ trước khối (khối che một phần là đúng thực tế).
   Cột sáng thì phải vẽ SAU khối, không thì nó chạy sau lưng máy và mất hẳn —
   lỗi ở bản đầu. Nên hàm này tách làm hai lượt. */
function floorRing(cx, cy) {
  const w = TILE_W * 0.42, h = TILE_H * 0.42;
  return `
    <ellipse cx="${cx}" cy="${cy + h * 0.5}" rx="${w * 1.34}" ry="${h * 1.34}"
      fill="#5980a6" opacity=".1"/>
    <ellipse cx="${cx}" cy="${cy + h * 0.5}" rx="${w * 1.34}" ry="${h * 1.34}"
      fill="none" stroke="#5980a6" stroke-width="3"/>`;
}

function activeMarks(cx, cy) {
  const w = TILE_W * 0.42, h = TILE_H * 0.42;
  const marks = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) => {
    const mx = cx + sx * w * 1.5, my = cy + sy * h * 1.5;
    return `<path d="M ${mx - sx * 11} ${my} H ${mx} V ${my - sy * 9}"
      fill="none" stroke="#5980a6" stroke-width="2.5" stroke-linecap="round"/>`;
  }).join('');
  return `
    <line x1="${cx}" y1="${cy - BOX_H - h - 6}" x2="${cx}" y2="${cy - BOX_H - h - 62}"
      stroke="url(#beam)" stroke-width="9" stroke-linecap="round"/>
    ${marks}`;
}

/**
 * render(el, {machines, self, onOtherTap})
 *   machines    [{name, line, floor:{x,y}}]
 *   self        tên máy của chính station này
 *   onOtherTap  gọi khi chạm vào máy KHÁC — phải nói gì đó, xem dưới
 */
export function render(el, { machines, self, onOtherTap, t }) {
/* Vị trí máy lấy từ mặt bằng dùng chung, KHÔNG từ `machines.json`. Hai bản đồ
   2D và sơ đồ 3D phải bày ra cùng một xưởng; đọc hai nguồn khác nhau thì sửa
   một bên là hai bản đồ lệch nhau mà không ai báo lỗi. `m.floor` chỉ còn là
   đường lui cho máy chưa có tên trong mặt bằng. */
  const cells = planCells();
  const placed = (machines || [])
    .filter(m => cells[m.name] || m.floor)
    .map(m => ({ ...m, floor: cells[m.name] || m.floor }));
  if (!placed.length) { el.innerHTML = ''; return; }

  const xs = placed.map(m => m.floor.x), ys = placed.map(m => m.floor.y);
  const maxX = Math.max(1, ...xs), maxY = Math.max(1, ...ys);

  let grid = '';
  for (let i = -0.8; i <= maxX + 1; i += 0.5) {
    const [x1, y1] = iso(i, -0.8), [x2, y2] = iso(i, maxY + 1);
    grid += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
  }
  for (let j = -0.8; j <= maxY + 1; j += 0.5) {
    const [x1, y1] = iso(-0.8, j), [x2, y2] = iso(maxX + 1, j);
    grid += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
  }

  // Vẽ theo chiều sâu: vật gần phải nằm trên vật xa.
  const depth = placed.slice().sort((a, b) =>
    (a.floor.x + a.floor.y) - (b.floor.x + b.floor.y));

  const solids = depth.map(m => {
    const [cx, cy] = iso(m.floor.x, m.floor.y);
    const mine = m.name === self;
    return `<g class="fm ${mine ? 'mine' : 'other'}" data-name="${esc(m.name)}">
      <title>${esc(m.name)}${mine ? '' : ` — ${esc(t.otherMachine)}`}</title>
      ${mine ? floorRing(cx, cy) : ''}
      ${box(cx, cy, mine ? ACTIVE : MUTED, mine ? 4 : 0)}
      ${mine ? activeMarks(cx, cy) : ''}
    </g>`;
  }).join('');

  /* Nhãn vẽ ở lượt riêng, sau tất cả khối. Nhãn của máy khác mờ 62%.

     Cỡ nhãn nhân LBL: từ khi bản đồ bày theo mặt bằng thật, bảy máy trải trên
     một vùng rộng gấp ~4,8 lần lưới 3×2 cũ. viewBox tự co cho vừa khung, nên
     mọi thứ bị thu nhỏ theo — chữ 13px hoá ra còn ~3px trên màn hình và không
     ai đọc được. Khối thì vẫn nhận ra là khối khi nhỏ, chữ thì không, nên chỉ
     phóng nhãn. */
  const LBL = 2.2;
  const labels = placed.map(m => {
    const [cx, cy] = iso(m.floor.x, m.floor.y);
    const mine = m.name === self;
    return `<g class="fm ${mine ? 'mine' : 'other'}" data-name="${esc(m.name)}"
        opacity="${mine ? 1 : 0.62}">
      <rect x="${cx - 58 * LBL}" y="${cy + 20}" width="${116 * LBL}" height="${32 * LBL}" rx="5"
        fill="#ffffff" opacity="${mine ? 0.94 : 0.8}"/>
      <text x="${cx}" y="${cy + 20 + 14 * LBL}" text-anchor="middle" font-size="${13 * LBL}"
        font-weight="${mine ? 700 : 500}" fill="#1d1f20">${esc(m.name)}</text>
      <text x="${cx}" y="${cy + 20 + 27 * LBL}" text-anchor="middle" font-size="${10 * LBL}"
        fill="#7a7a7d">${esc(m.line || '')}</text>
    </g>`;
  }).join('');

  const pts = [iso(-0.8, -0.8), iso(maxX + 1, -0.8), iso(-0.8, maxY + 1),
               iso(maxX + 1, maxY + 1)];
  placed.forEach(m => {
    const [cx, cy] = iso(m.floor.x, m.floor.y);
    pts.push([cx - 62 * 2.2, cy - BOX_H - 60], [cx + 62 * 2.2, cy + 20 + 32 * 2.2 + 8]);
  });
  const bx = Math.min(...pts.map(p => p[0])) - 12;
  const by = Math.min(...pts.map(p => p[1])) - 12;
  const bw = Math.max(...pts.map(p => p[0])) + 12 - bx;
  const bh = Math.max(...pts.map(p => p[1])) + 12 - by;

  el.innerHTML = `
    <svg class="floor" viewBox="${bx} ${by} ${bw} ${bh}" role="img"
         aria-label="${esc(t.floorTitle)}">
      <defs><linearGradient id="beam" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0" stop-color="#5980a6" stop-opacity=".75"/>
        <stop offset="1" stop-color="#5980a6" stop-opacity="0"/>
      </linearGradient></defs>
      <g stroke="#d4d4d7" stroke-width=".7" opacity=".5">${grid}</g>
      ${solids}${labels}
    </svg>
    <p class="floor-note" id="floor-note">${esc(t.floorHint)}</p>`;

  /* Chạm vào máy khác KHÔNG được im lặng. Không đổi con trỏ, không mở gì, nhưng
     phải nói một câu — im lặng thì người dùng tưởng màn hình đơ và chạm tiếp. */
  el.querySelectorAll('.fm.other').forEach(g =>
    g.addEventListener('click', () => onOtherTap && onOtherTap(g.dataset.name)));
}
