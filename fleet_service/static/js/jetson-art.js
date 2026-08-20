/* ═══════════════════════════════════════════════════════════════════════════
   Hình minh hoạ Jetson trên thẻ máy.

   Dựng bằng SVG chứ không bằng CSS 3D transform. Bản trước xoay chồng chục thẻ
   <span>, mà cái cần vẽ lại là những chi tiết người ta dùng để NHẬN RA phần
   cứng: cánh tản nhiệt, quạt, dãy 40 chân, cụm cổng phía sau. Với transform thì
   mỗi chi tiết như thế là thêm một lớp giả; với SVG thì nó là hình học.

   Toạ độ nhập theo MILIMÉT THẬT của dev kit rồi mới chiếu, nên tỉ lệ giữa board,
   tản nhiệt và hàng chân đúng như vật — đó là thứ làm hình trông ra hàng thật.

   Orin Nano và Orin Nano Super dùng CHUNG hình học bo mạch, vì chúng đúng là
   cùng một bộ dev kit — "Super" là mở khoá xung nhịp, không phải bo mạch khác.
   Cái tách hai loại ra là phần NHẬN DIỆN: vỏ quạt và viền xanh NVIDIA, nhãn
   riêng. Làm khác ở chỗ đó thì nhìn một cái là phân biệt được thế hệ mà vẫn
   không vẽ ra một khác biệt phần cứng không tồn tại.
   ═══════════════════════════════════════════════════════════════════════════ */

import { esc } from './core.js';

/* Chiếu đẳng cự: nhìn từ trên xuống, hai trục đáy chếch 30°, trục cao thẳng
   đứng. Cùng phép chiếu với sơ đồ mặt bằng, nên thẻ máy và sơ đồ nhìn "cùng
   một thế giới". */
const K = 0.62;                       // mm → px
const CX = 0.866 * K, CY = 0.5 * K;   // cos30, sin30
const iso = (x, y, z) => [(x - y) * CX, (x + y) * CY - z * K];
const pt = (x, y, z) => iso(x, y, z).map(n => n.toFixed(2)).join(',');

/** Khối hộp đẳng cự: mặt trên + hai mặt bên. Trả về theo thứ tự vẽ từ xa tới
 *  gần, nên cứ nối chuỗi là chồng đúng. */
function slab(x, y, z, w, d, h, c) {
  const top = `${pt(x, y, z + h)} ${pt(x + w, y, z + h)} ${pt(x + w, y + d, z + h)} ${pt(x, y + d, z + h)}`;
  const left = `${pt(x, y + d, z + h)} ${pt(x + w, y + d, z + h)} ${pt(x + w, y + d, z)} ${pt(x, y + d, z)}`;
  const right = `${pt(x + w, y, z + h)} ${pt(x + w, y + d, z + h)} ${pt(x + w, y + d, z)} ${pt(x + w, y, z)}`;
  return `<polygon points="${left}" fill="${c.l}"/>
          <polygon points="${right}" fill="${c.r}"/>
          <polygon points="${top}" fill="${c.t}"/>`;
}

/** Vệt phẳng nằm trên mặt board — nhãn lụa, khe cắm, viền. */
const patch = (x, y, z, w, d, fill, extra = '') =>
  `<polygon points="${pt(x, y, z)} ${pt(x + w, y, z)} ${pt(x + w, y + d, z)} ${pt(x, y + d, z)}"
     fill="${fill}" ${extra}/>`;

/* Bảng màu: nhựa PCB xanh rêu, nhôm anod đen, thép, chân mạ vàng. Lấy từ ảnh
   sản phẩm chứ không lấy từ tokens giao diện — đây là vật thật, không phải một
   thành phần UI. */
const C = {
  pcb:    { t: '#1f4a34', l: '#123024', r: '#173d2b' },
  sink:   { t: '#33373c', l: '#191c20', r: '#232629' },
  fanhub: { t: '#4a4f55', l: '#26292d', r: '#303338' },
  metal:  { t: '#c3c7cb', l: '#7e8489', r: '#9ca2a7' },
  agx:    { t: '#b9bec4', l: '#6d7378', r: '#8b9197' },
  agxTop: { t: '#2f3338', l: '#1a1d21', r: '#24272b' },
  port:   { t: '#b0b5b9', l: '#5f6469', r: '#7d8287' },
  black:  { t: '#2a2d31', l: '#141618', r: '#1e2124' },
};

/* ── Orin Nano Developer Kit ─────────────────────────────────────────────── */
/* Carrier 100 × 79 mm; cụm tản nhiệt + quạt ~70 × 70 × 26; hàng 40 chân dọc
   cạnh dài; cụm cổng dồn về một cạnh ngắn. */
function nano(led, badge, sup) {
  // `sup`: bộ Super — vỏ quạt và viền theo xanh NVIDIA, PCB đen hơn.
  const pcb = sup ? { t: '#16281f', l: '#0b1611', r: '#101d16' } : C.pcb;
  const hub = sup ? { t: '#4b6a3c', l: '#22301c', r: '#2f4527' } : C.fanhub;
  const trim = sup ? '#76b900' : '#2c6647';        // xanh NVIDIA
  const P = [];
  P.push(slab(0, 0, 0, 100, 79, 1.6, pcb));

  // Lụa in + bốn lỗ bắt ốc: thứ khiến mảng xanh đọc ra là bo mạch chứ không
  // phải một tấm bìa.
  P.push(patch(3, 3, 1.7, 94, 73, 'none', `stroke="${trim}" stroke-width=".6"`));
  for (const [sx, sy] of [[5, 5], [95, 5], [5, 74], [95, 74]])
    P.push(`<circle cx="${iso(sx, sy, 1.7)[0]}" cy="${iso(sx, sy, 1.7)[1]}"
            r="1.6" fill="#0d2419" stroke="#c8c2a8" stroke-width=".5"/>`);

  // Khe M.2 dưới gầm, ló ra ở mép
  P.push(patch(58, 62, 1.7, 34, 6, '#0f2a1e'));

  // Dãy 40 chân: đế nhựa đen + hai hàng chân mạ vàng
  P.push(slab(24, 2.5, 1.6, 52, 5.4, 2.6, C.black));
  for (let i = 0; i < 20; i++) {
    const px = 25.4 + i * 2.6;
    P.push(slab(px, 3.2, 4.2, 1.3, 1.3, 2.6,
      { t: '#f2d178', l: '#8f6f24', r: '#c79c3c' }));
    P.push(slab(px, 5.6, 4.2, 1.3, 1.3, 2.6,
      { t: '#f2d178', l: '#8f6f24', r: '#c79c3c' }));
  }

  // Cụm tản nhiệt: đế nhôm + cánh. Cánh vẽ bằng KHE TỐI xen kẽ chứ không phải
  // vạch sáng — nhôm anod đen thì cái mắt bắt được là bóng giữa hai cánh.
  P.push(slab(16, 14, 1.6, 70, 60, 14, C.sink));
  for (let i = 0; i < 12; i++)
    P.push(patch(19 + i * 5.6, 15.5, 15.62, 2.2, 57, '#15181b'));

  // Quạt: khung + trục + cánh nghiêng
  P.push(slab(24, 22, 15.6, 54, 44, 10, hub));
  // Vạch xanh chạy dọc vỏ quạt — dấu hiệu bản Super, thấy ngay ở cỡ thẻ máy.
  if (sup) P.push(patch(26, 24, 25.7, 50, 2.6, '#76b900'));
  const [fx, fy] = iso(51, 44, 25.6);
  P.push(`<ellipse cx="${fx.toFixed(2)}" cy="${fy.toFixed(2)}" rx="15" ry="8.6"
          fill="#1b1e21" stroke="#4d5359" stroke-width=".8"/>`);
  for (let i = 0; i < 9; i++) {
    const a = (i / 9) * Math.PI * 2;
    P.push(`<path d="M ${fx.toFixed(2)} ${fy.toFixed(2)}
      q ${(Math.cos(a) * 9).toFixed(2)} ${(Math.sin(a) * 5.2).toFixed(2)}
        ${(Math.cos(a + .5) * 14).toFixed(2)} ${(Math.sin(a + .5) * 8).toFixed(2)}"
      fill="none" stroke="${sup ? '#63823f' : '#3a4046'}" stroke-width="1.5"
      stroke-linecap="round"/>`);
  }
  P.push(`<ellipse cx="${fx.toFixed(2)}" cy="${fy.toFixed(2)}" rx="4.2" ry="2.4"
          fill="#5a6067"/>`);

  // Cụm cổng phía sau: RJ45, 2 tầng USB-A, USB-C, DisplayPort, jack nguồn
  P.push(slab(-2, 20, 1.6, 4, 13, 11, C.port));         // RJ45
  P.push(slab(-2, 36, 1.6, 4, 11, 7, C.port));          // USB-A tầng dưới
  P.push(slab(-2, 36, 9.2, 4, 11, 7, C.port));          // USB-A tầng trên
  P.push(slab(-2, 50, 1.6, 4, 7, 4, C.port));           // USB-C
  P.push(slab(-2, 60, 1.6, 4, 10, 6, C.port));          // DisplayPort
  P.push(slab(90, 66, 1.6, 8, 9, 6, C.black));          // jack nguồn
  P.push(patch(2, 22, 12.6, 0, 0, 'none'));

  // Khe microSD ở cạnh bên
  P.push(slab(40, 75, 1.6, 16, 4, 2.2, C.metal));

  // Đèn trạng thái — chấm màu duy nhất trên hình, và nó nói đúng một điều
  const [lx, ly] = iso(94, 12, 2.2);
  P.push(`<circle cx="${lx.toFixed(2)}" cy="${ly.toFixed(2)}" r="2.4"
          fill="${led}"/>`);

  // Nhãn nằm trên dải PCB trống sát mép trước. Đặt lên mặt tản nhiệt thì chữ
  // đen trên nền đen, và ở cỡ thẻ máy thì nó chỉ còn là một vệt bẩn.
  const [bx, by] = iso(5, 77, 1.8);
  P.push(`<text x="${bx.toFixed(2)}" y="${by.toFixed(2)}" font-size="4.1"
    font-weight="700" fill="${sup ? '#9fdc3a' : '#a9e0bf'}" letter-spacing=".35"
    transform="rotate(30 ${bx.toFixed(2)} ${by.toFixed(2)})">${esc(badge)}</text>`);
  return P.join('');
}

/* ── AGX Orin Developer Kit ──────────────────────────────────────────────── */
/* Không phải bo mạch trần mà là một khối vỏ nhôm 110 × 110 × 72, mặt trên đen
   đục lỗ, một góc vát. Nhìn khác hẳn Nano — và đó chính là điều hình này phải
   truyền đạt ngay từ xa. */
function agx(led, badge) {
  const P = [];
  P.push(slab(0, 0, 0, 110, 110, 62, C.agx));
  P.push(slab(4, 4, 62, 102, 102, 8, C.agxTop));

  // Lưới đục lỗ trên nắp
  for (let i = 0; i < 9; i++)
    for (let j = 0; j < 9; j++) {
      const [hx, hy] = iso(12 + i * 10.5, 12 + j * 10.5, 70);
      P.push(`<circle cx="${hx.toFixed(2)}" cy="${hy.toFixed(2)}" r="1.5"
              fill="#101215"/>`);
    }

  // Đường vát và khe tản nhiệt bên hông
  P.push(patch(0, 0, 62, 110, 4, '#dfe3e6'));
  for (let i = 0; i < 7; i++)
    P.push(`<polygon points="${pt(110, 16 + i * 12, 54)} ${pt(110, 22 + i * 12, 54)}
            ${pt(110, 22 + i * 12, 10)} ${pt(110, 16 + i * 12, 10)}"
            fill="#6a7075"/>`);

  // Cụm cổng phía sau
  P.push(slab(-2, 18, 8, 4, 14, 12, C.port));
  P.push(slab(-2, 38, 8, 4, 12, 8, C.port));
  P.push(slab(-2, 38, 17, 4, 12, 8, C.port));
  P.push(slab(-2, 56, 8, 4, 8, 5, C.port));
  P.push(slab(-2, 70, 8, 4, 12, 7, C.port));

  // Gờ vát chạy quanh chân và hõm đế — vỏ AGX không phải một khối hộp trơn,
  // và chính đường gờ đó làm nó ra dáng thiết bị công nghiệp thay vì cái thùng.
  P.push(slab(2, 2, 0, 106, 106, 5, { t: '#cfd3d7', l: '#5c6165', r: '#787e83' }));
  P.push(`<polygon points="${pt(0, 110, 62)} ${pt(110, 110, 62)}
          ${pt(110, 110, 56)} ${pt(0, 110, 56)}" fill="#dfe3e6" opacity=".55"/>`);

  const [lx, ly] = iso(100, 100, 70.5);
  P.push(`<circle cx="${lx.toFixed(2)}" cy="${ly.toFixed(2)}" r="2.6"
          fill="${led}"/>`);

  // Nhãn ép lên MẶT BÊN, không lên nắp: nắp đục lỗ dày đặc, chữ đặt lên đó thì
  // không đọc nổi ở cỡ thẻ máy.
  const [bx, by] = iso(8, 110, 40);
  P.push(`<text x="${bx.toFixed(2)}" y="${by.toFixed(2)}" font-size="5.2"
    font-weight="700" fill="#4c5257" letter-spacing=".4"
    transform="rotate(30 ${bx.toFixed(2)} ${by.toFixed(2)})">${esc(badge)}</text>`);
  return P.join('');
}

/* ── API ─────────────────────────────────────────────────────────────────── */

const LED = { ok: '#37c06a', warn: '#e0a33a', bad: '#8d969d', mute: '#8d969d' };

/**
 * @param kind 'nano' | 'super' | 'agx'
 * @param tone 'ok' | 'warn' | 'bad' | 'mute' — chỉ quyết định màu ĐÈN, không
 *   đổi hình: máy mất mạng vẫn là đúng cái máy đó, làm nó biến dạng thì thẻ
 *   trông như thay phần cứng.
 */
export function jetsonSVG(kind, tone) {
  const led = LED[tone] || LED.bad;
  // Nhãn ngắn: bo mạch chỉ rộng 100 mm, chuỗi dài hơn là tràn khỏi mép và
  // rơi xuống cái bóng đổ.
  const badge = kind === 'agx' ? 'JETSON AGX ORIN'
    : kind === 'super' ? 'ORIN NANO SUPER' : 'ORIN NANO';
  const body = kind === 'agx' ? agx(led, badge)
    : nano(led, badge, kind === 'super');
  // viewBox ôm cả hai dáng: Nano bè ngang, AGX cao. Dùng chung một khung để
  // hai loại thẻ cạnh nhau không nhảy kích thước.
  return `<svg class="jetson-svg" viewBox="-42 -34 112 90" aria-hidden="true">
    <ellipse cx="14" cy="41" rx="40" ry="10" fill="#1d1f20" opacity=".13"/>
    <g>${body}</g></svg>`;
}
