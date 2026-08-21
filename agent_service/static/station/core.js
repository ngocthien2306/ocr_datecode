/* Shim cho `factory-3d.js` — bản copy nguyên vẹn từ Fleet Console.
 *
 * Module 3D chỉ cần đúng ba thứ từ `store`: theme, tiêu đề sơ đồ, và nhãn trạng
 * thái. Cấp qua shim thay vì sửa module, để lần sau đồng bộ với Fleet Console
 * chỉ là copy lại một file — sửa vào ruột nó là mỗi lần sync một lần merge tay.
 *
 * ĐÁNH ĐỔI đã biết: đây là 1.280 dòng NHÂN BẢN. Hai bản sẽ trôi khác nhau. Cách
 * đúng về lâu dài là tách thành package dùng chung cho cả fleet_service và
 * agent_service; ở đây chọn copy để Line Station không phải phụ thuộc vào fleet
 * service — thứ mà cả màn hình này được thiết kế để sống thiếu.
 */

const T = {
  vi: {
    floorTitle: 'Sơ đồ nhà máy',
    state: { ok: 'đang chạy', warn: 'cần chú ý', agent_down: 'trợ lý tắt',
             unreachable: 'không với tới được', offline: 'ngoài mạng' },
  },
  en: {
    floorTitle: 'Factory floor',
    state: { ok: 'running', warn: 'needs attention', agent_down: 'assistant off',
             unreachable: 'unreachable', offline: 'off network' },
  },
};

export const store = {
  get lang() { return localStorage.getItem('station_lang') || 'vi'; },
  get theme() {
    // Lựa chọn của NGƯỜI trước, hệ thống sau. Thiếu bước này thì bấm nút sáng
    // mà sơ đồ 3D vẫn tối — nửa màn hình một tông, nửa kia một tông.
    const chosen = document.documentElement.getAttribute('data-theme');
    if (chosen) return chosen;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches
      ? 'dark' : 'light';
  },
  get t() { return T[this.lang] || T.vi; },
};
