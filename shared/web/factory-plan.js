/* ═══════════════════════════════════════════════════════════════════════════
   Mặt bằng nhà máy — MỘT nguồn cho cả ba bản đồ.

   Có ba thứ vẽ cùng một xưởng:
     · sơ đồ 3D dùng chung  (shared/web/factory-map-3d.js)
     · bản 2D dự phòng Fleet (fleet_service/static/js/factory-map.js)
     · bản 2D bậc 1 Station  (agent_service/static/station/floor.js)

   Trước đây bản 3D giữ bố trí riêng trong code còn hai bản 2D đọc toạ độ trong
   `machines.json`. Hai nguồn ấy không có gì ràng buộc nhau, nên sửa một bên là
   hai bản đồ bày khác hẳn nhau mà không ai báo lỗi — người đứng ở Station nhìn
   ra một xưởng, người ở phòng vận hành nhìn ra một xưởng khác.

   Tách ra đây vì đó là DỮ LIỆU, không phải chi tiết dựng hình. Module này cố ý
   không import gì: bản 2D bậc 1 chạy trên tablet cũ và không được kéo theo
   three.js chỉ để biết máy đứng ở đâu.

   Hệ toạ độ: +x sang đông, +z xuống nam. Gốc giữa sảnh chính, đơn vị ~mét.
   ═══════════════════════════════════════════════════════════════════════════ */

export const PLANT = {
  hall: { x: -2, z: -3, w: 78, d: 64, label: 'MAIN BUILDING' },
  areas: [
    { label: 'LINE 6 BAY', x: -30, z: -14, w: 16, d: 11, tint: '#d9a441' },
    { label: 'LINE 1 · LINE 2 BAY', x: -12, z: -8.5, w: 16, d: 22, tint: '#a86fc0' },
    { label: 'WEST BAY', x: -30, z: 10, w: 16, d: 21, tint: '#d9a441' },
    { label: 'INNER BAY', x: 17, z: 12.5, w: 34, d: 28, tint: '#d97b3f' },
    { label: 'LINE 3 · LINE 4 BAY', x: 25, z: 15.5, w: 16, d: 22, tint: '#3fa383' },
  ],
  /* `open` là hướng CỬA của mỗi khu — ba vách kín, một mặt hở ra hành lang.
     Trước đây hướng này suy ra từ dấu của z, tức đã giả định sẵn hai dãy quay
     mặt vào nhau. Mặt bằng thật thì mỗi khu quay một kiểu, nên phải khai. */
  // Khoá LUÔN đặt trong nháy, kể cả khi tên hợp lệ làm định danh. Tên máy là
  // dữ liệu, không phải cú pháp — và bất kỳ công cụ nào đổi tên máy bằng cách
  // thay chuỗi sẽ biến một khoá trần thành `Line 2:`, tức lỗi cú pháp câm.
  lines: {
    'Line 6': { x: -30, z: -14, open: 'east' },
    'Line 1': { x: -12, z: -14, open: 'east' },
    'Line 2': { x: -12, z:  -3, open: 'east' },
    'Line 7': { x: -30, z:  15, open: 'north' },
    'Line 3': { x:  25, z:  10, open: 'west' },
    'Line 4': { x:  25, z:  21, open: 'west' },
    'Line 5': { x:  26, z: -52, open: 'south' },
  },
  // `fill` là mức chất hàng: kho nguyên liệu đầy, kho thành phẩm vơi hơn vì
  // hàng vừa được lấy đi. Ba kho trông khác nhau là có lý do.
  stores: [
    { label: 'KHO 1', x: 4.5, z: 4, bays: 2, fill: 0.9 },
    { label: 'KHO 2', x: 12.5, z: 4, bays: 2, fill: 0.55 },
    { label: 'KHO 3', x: -30, z: 3, bays: 3, fill: 0.75 },
  ],
  entrance: { x: -6, z: -35 },
  ops: { x: 14, z: -25 },
};

// Cửa quay về đâu → xoay cả khu quanh trục Y. Dựng hình một lần theo hướng
// chuẩn (cửa mở về +z) rồi xoay, thay vì viết bốn biến thể hình học.
export const OPEN_YAW = { south: 0, north: Math.PI, east: Math.PI / 2, west: -Math.PI / 2 };

/** Một ô lưới của bản đồ 2D bằng bao nhiêu mét.
 *
 *  Khoảng cách nhỏ nhất giữa hai máy trong mặt bằng là 11 m (Line 1 ↔ Line 2,
 *  và Line 3 ↔ Line 4). Trên lưới đẳng cự, hai khối chỉ hết chồng nhau khi cách nhau
 *  từ 2 Ô trở lên: lệch 1 ô cho ra 105 px ngang, mà riêng bề rộng một khối đã
 *  176 px. Nên 11 m phải bằng 2 ô → mỗi ô 5,5 m. Đặt 11 (1 ô) thì hai máy đó
 *  đè lên nhau, đúng như bản đầu. */
export const PLAN_CELL = 5.5;

/** Toạ độ ô lưới cho hai bản đồ 2D, suy thẳng từ mặt bằng ở trên.
 *
 *  Hai bản 2D vẽ theo lưới đẳng cự gốc (0,0), còn mặt bằng dùng mét có số âm —
 *  nên chia cho PLAN_CELL rồi dịch về gốc. Suy ra chứ không chép tay, để sửa
 *  mặt bằng là cả ba bản đồ đi theo. */
export function planCells() {
  const entries = Object.entries(PLANT.lines);
  const minX = Math.min(...entries.map(([, p]) => p.x));
  const minZ = Math.min(...entries.map(([, p]) => p.z));
  const out = {};
  for (const [name, p] of entries) {
    out[name] = {
      x: Number(((p.x - minX) / PLAN_CELL).toFixed(2)),
      y: Number(((p.z - minZ) / PLAN_CELL).toFixed(2)),
    };
  }
  return out;
}
