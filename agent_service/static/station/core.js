/* `store` cho sơ đồ nhà máy dùng chung (`/shared/factory-map-3d.js`).
 *
 * Module đó không import gì của service nào: nó nhận `store` qua render(). Ba
 * thứ nó cần là theme, tiêu đề sơ đồ, và nhãn trạng thái — cấp ở đây, với nhãn
 * tiếng Việt của Line Station.
 *
 * Trước đây bề mặt này giữ MỘT BẢN COPY 1.282 dòng của module 3D, và shim này
 * tồn tại để lần sau đồng bộ chỉ là copy lại một file. Hai bản đã trôi khác
 * nhau 80 dòng sau một ngày, nên giờ chỉ còn một bản ở `shared/web/`.
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
