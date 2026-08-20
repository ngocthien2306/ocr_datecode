/* ═══════════════════════════════════════════════════════════════════════════
   Nền chung: chuỗi song ngữ, quy tắc trình bày số, và tầng gọi API.

   ES module thuần, không bundler. Fleet service là một tiến trình FastAPI phục
   vụ file tĩnh; thêm một toolchain build chỉ để tách file là cái giá không đáng
   ở quy mô này.
   ═══════════════════════════════════════════════════════════════════════════ */

/* ── Song ngữ ───────────────────────────────────────────────────────────────
   MỌI chuỗi nằm ở đây, không rải trong hàm dựng component. Rải ra thì lần thêm
   ngôn ngữ sau sẽ sót, mà sót chỗ nào chỉ lộ khi có máy rơi đúng vào trạng thái
   hiếm đó — `agent_down` chỉ hiện khi thật sự có máy hỏng.                    */

export const I18N = {
  en: {
    locale: 'en-GB',
    noLogin: 'tailnet-only · no login by design',
    running: 'running', attention: 'needs attention',
    refresh: 'Refresh', updated: 'updated', seen: 'seen', never: 'never',
    floorTitle: 'Factory floor — spice packing · isometric (tier 1)',
    floorHint: 'not to scale · positions from machines.json',
    machinesTitle: 'Machines — current shift',
    clickHint: 'click a machine for detail',
    load: 'load', free: 'free', of: 'of', uptime: 'Uptime', disk: 'Disk',
    products: 'products', shiftTarget: 'of shift target',
    passThisShift: 'pass · this shift', vsPrev: 'vs prev shift',
    pts: 'pts', hardware: 'Hardware', noMetrics: 'no readings available',
    cameraOn: 'camera service running', cameraOff: 'camera service stopped',
    cameraUnknown: 'camera service unknown',
    d: 'd', h: 'h', m: 'm',
    production: 'Production', fleet7d: 'Fleet · last 7 days',
    fingerprint: 'Failure fingerprint',
    fpNote: "Share of each machine's sampled failures — comparable across machines because these causes belong to the OCR pipeline, not the product.",
    noRank: 'Lines run different recipes — pass rates are not comparable.',
    output: 'Output', perDay: 'per day', passRate: 'Pass rate', recipe: 'Recipe',
    sample: 'sample', partialSample: 'sampled', fullSample: 'complete',
    fleetTotal: 'Fleet total', days: 'days', export: 'Export report',
    tabOverview: 'Overview', tabStaff: 'Staff', tabLog: 'Activity log',
    groupBy: 'Group by', byMachineDept: 'Machine → Dept',
    byDeptMachine: 'Dept → Machine', byShiftMachine: 'Shift → Machine',
    searchStaff: 'Search name, code, username…',
    all: 'All', machine: 'Machine', department: 'Department', shift: 'Shift',
    access: 'Access', onShiftOnly: 'On shift now only', people: 'people',
    userActions: 'User actions', systemErrors: 'System errors',
    showSimulated: 'Show demo records', askMore: 'ask',
    loading: 'loading…', noRecords: 'no records in this period',
    detail: 'Detail', user: 'User', action: 'Action',
    lines: 'Problem lines', groups: 'Distinct', topProblems: 'Most frequent',
    noUserRecords: m => `no user records on ${m}`,
    askAudit: (mach, user, act, at) =>
      `On ${mach}, ${user} performed ${act} at ${at}. Explain what that changed.`,
    askLogErrors: m => `Analyse today's system errors on ${m}.`,
    failures: 'Failure investigation', topImages: 'Failure samples',
    expected: 'expected', gotRead: 'read',
    chatTitle: 'Fleet assistant', chatHint: 'ask about all machines',
    send: 'Send', placeholder: 'e.g. which machine has the worst quality?',
    thinking: 'thinking…', usedTools: 'tools', fromMachines: 'attachments',
    askingAbout: 'asking about',
    ctxMachine: m => `machine ${m}`,
    ctxPerson: (n, u, m) => `${n} (@${u}) on ${m}`,
    askStaff: (n, u, m) =>
      `What has @${u} (${n}) done on ${m} in the last 7 days — did they load or change any recipe?`,
    chips: ['Output for all machines this week',
            'Which machine has the worst quality? Why?',
            'Any machine running hot or low on disk?'],
    complete: n => `All ${n} machines reporting.`,
    missing: (k, n) => `<b>No data from ${k}/${n} machines:</b> `,
    partial: (k, n) => `<b>Partial data from ${k}/${n} machines:</b> `,
    loadFail: e => `<b>Could not load: ${e}</b>`,
    staleAt: t => `showing last good data from ${t}`,
    state: {
      ok: 'Running', warn: 'Needs attention',
      agent_down: 'Assistant off · machine running',
      unreachable: 'Unreachable', offline: 'Off network',
      not_started: 'Shift not started', partial: 'Partial data',
    },
    /* Nguyên nhân lỗi: máy trạm trả về CẢ khoá ổn định (`cause`) lẫn nhãn tiếng
       Việt (`label`). Dịch từ khoá; nhãn của máy trạm chỉ dùng khi gặp khoá lạ —
       nếu không thì bật sang EN vẫn thấy nguyên một cột chữ Việt. */
    cause: {
      char_verification: 'Character below confidence threshold',
      no_detection: 'Detector found no region in frame',
      text_verification: 'OCR read the wrong string',
      template_verification: 'Image did not match template',
      product_verification: 'Product not recognised',
      unknown: 'Unclassified',
    },
  },
  vi: {
    locale: 'vi-VN',
    noLogin: 'chỉ trong tailnet · không có đăng nhập, có chủ đích',
    running: 'đang chạy', attention: 'cần chú ý',
    refresh: 'Làm mới', updated: 'cập nhật', seen: 'thấy', never: 'chưa từng',
    floorTitle: 'Mặt bằng xưởng — đóng gói gia vị · đẳng cự (bậc 1)',
    floorHint: 'không theo tỉ lệ · vị trí lấy từ machines.json',
    machinesTitle: 'Các máy — ca hiện tại',
    clickHint: 'bấm vào máy để xem chi tiết',
    load: 'tải', free: 'trống', of: '/', uptime: 'Chạy liên tục', disk: 'Đĩa',
    products: 'sản phẩm', shiftTarget: 'chỉ tiêu ca',
    passThisShift: 'đạt · ca này', vsPrev: 'so ca trước',
    pts: 'điểm', hardware: 'Phần cứng', noMetrics: 'không lấy được số liệu',
    cameraOn: 'camera service đang chạy', cameraOff: 'camera service đã tắt',
    cameraUnknown: 'camera service không rõ',
    d: 'ngày', h: 'giờ', m: 'phút',
    production: 'Sản xuất', fleet7d: 'Toàn nhà máy · 7 ngày',
    fingerprint: 'Vân tay kiểu lỗi',
    fpNote: 'Tỉ trọng trên mẫu fail của từng máy — so được giữa các máy vì các nguyên nhân này thuộc pipeline OCR, không thuộc mặt hàng.',
    noRank: 'Các line chạy recipe khác nhau — tỉ lệ đạt không so trực tiếp được.',
    output: 'Sản lượng', perDay: 'mỗi ngày', passRate: 'Tỉ lệ đạt', recipe: 'Recipe',
    sample: 'mẫu', partialSample: 'lấy mẫu', fullSample: 'phủ hết kỳ',
    fleetTotal: 'Toàn nhà máy', days: 'ngày', export: 'Xuất báo cáo',
    tabOverview: 'Tổng quan', tabStaff: 'Nhân sự', tabLog: 'Nhật ký',
    groupBy: 'Nhóm theo', byMachineDept: 'Máy → Bộ phận',
    byDeptMachine: 'Bộ phận → Máy', byShiftMachine: 'Ca → Máy',
    searchStaff: 'Tìm tên, mã NV, username…',
    all: 'Tất cả', machine: 'Máy', department: 'Bộ phận', shift: 'Ca',
    access: 'Quyền', onShiftOnly: 'Chỉ người đang trong ca', people: 'người',
    userActions: 'Thao tác người dùng', systemErrors: 'Lỗi hệ thống',
    showSimulated: 'Hiện bản ghi demo', askMore: 'hỏi',
    loading: 'đang tải…', noRecords: 'không có bản ghi nào trong khoảng này',
    detail: 'Chi tiết', user: 'Người dùng', action: 'Thao tác',
    lines: 'Dòng có vấn đề', groups: 'Số nhóm', topProblems: 'Hay gặp nhất',
    noUserRecords: m => `${m} không có bản ghi người dùng`,
    askAudit: (mach, user, act, at) =>
      `Trên máy ${mach}, ${user} đã ${act} lúc ${at}. Giải thích việc đó đã thay đổi gì.`,
    askLogErrors: m => `Phân tích lỗi hệ thống trên máy ${m} hôm nay.`,
    failures: 'Điều tra nguyên nhân lỗi', topImages: 'Ảnh sản phẩm lỗi',
    expected: 'mong', gotRead: 'đọc',
    chatTitle: 'Trợ lý đội hình', chatHint: 'hỏi về mọi máy',
    send: 'Gửi', placeholder: 'vd: máy nào chất lượng tệ nhất?',
    thinking: 'đang nghĩ…', usedTools: 'tool', fromMachines: 'đính kèm',
    askingAbout: 'đang hỏi về',
    ctxMachine: m => `máy ${m}`,
    ctxPerson: (n, u, m) => `${n} (@${u}) trên ${m}`,
    askStaff: (n, u, m) =>
      `@${u} (${n}) đã thao tác gì trên máy ${m} trong 7 ngày qua — có load hay sửa recipe nào không?`,
    chips: ['Sản lượng cả đội hình tuần này',
            'Máy nào chất lượng tệ nhất? Tại sao?',
            'Máy nào đang nóng hoặc sắp đầy đĩa?'],
    complete: n => `Đủ cả ${n} máy.`,
    missing: (k, n) => `<b>Không lấy được ${k}/${n} máy:</b> `,
    partial: (k, n) => `<b>Thiếu một phần ${k}/${n} máy:</b> `,
    loadFail: e => `<b>Không tải được: ${e}</b>`,
    staleAt: t => `đang hiện số liệu cũ lúc ${t}`,
    state: {
      ok: 'đang chạy', warn: 'cần chú ý',
      agent_down: 'trợ lý tắt · máy vẫn chạy',
      unreachable: 'không với tới được', offline: 'ngoài mạng',
      not_started: 'ca chưa bắt đầu', partial: 'thiếu một phần',
    },
    cause: {
      char_verification: 'Ký tự dưới ngưỡng tin cậy',
      no_detection: 'Detector không thấy vùng nào trong khung',
      text_verification: 'OCR đọc sai chuỗi',
      template_verification: 'Ảnh không khớp template',
      product_verification: 'Không nhận ra sản phẩm',
      unknown: 'Chưa phân loại',
    },
  },
};

/** Nhãn nguyên nhân theo ngôn ngữ đang chọn, lùi về nhãn máy trạm gửi kèm. */
export const causeLabel = (key, fallback) =>
  store.t.cause[key] || fallback || key;

export const store = {
  lang: localStorage.getItem('fleet_lang') || 'en',
  theme: localStorage.getItem('fleet_theme') || 'light',
  get t() { return I18N[this.lang]; },
};

/* ── Quy tắc trình bày số ───────────────────────────────────────────────────
   Mỗi hàm dưới đây thi hành một quy tắc đã trả giá mới có, ghi trong
   docs/ui/04-design-system.md.                                               */

export const has = v => v !== null && v !== undefined;

/** Không có số thì "—", KHÔNG phải 0. Nhiệt độ null vẽ thành 0°C đọc như máy
 *  đang rất mát — sai đúng theo hướng nguy hiểm nhất. */
export const NA = '<span class="na">—</span>';
export const num = (v, d = 0) => has(v) ? Number(v).toFixed(d) : null;
export const fmt = (v, d = 0) =>
  has(v) ? Number(v).toLocaleString(store.t.locale,
            { minimumFractionDigits: d, maximumFractionDigits: d }) : NA;

/** Delta ghi bằng ĐIỂM phần trăm, không phải %. "+1,36đ" từng bị đọc thành
 *  "1,36 đồng". Và nền so sánh dưới 30 bản ghi thì KHÔNG đưa delta — nền 2 bản
 *  ghi từng cho ra "+3.309.850%". */
export const MIN_BASELINE = 30;
export function deltaPts(value, baselineCount) {
  if (!has(value) || (has(baselineCount) && baselineCount < MIN_BASELINE)) return '';
  const up = value >= 0;
  const arrow = up ? '▲' : '▼';
  return `<span class="delta ${up ? 'up' : 'down'}">${arrow} ${Math.abs(value).toFixed(1)} ${store.t.pts} ${store.t.vsPrev}</span>`;
}

/** Ngưỡng theo giới hạn phần cứng thật: Orin hạ xung ~85°C, CPU x86 crit=100;
 *  Mongo ngừng ghi khi hết đĩa. Không phải số tròn cho đẹp. */
export const level = (v, warn = 80, bad = 92) =>
  !has(v) ? '' : v >= bad ? 'bad' : v >= warn ? 'warn' : '';

export function uptime(s) {
  if (!s) return NA;
  const t = store.t;
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600);
  return d ? `${d}<small>${t.d}</small> ${h}<small>${t.h}</small>`
           : `${h}<small>${t.h}</small> ${Math.floor((s % 3600) / 60)}<small>${t.m}</small>`;
}

export const esc = s => String(s ?? '').replace(/[<>&"]/g,
  c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' }[c]));

export const clock = ts =>
  new Date((ts || Date.now() / 1000) * 1000)
    .toLocaleTimeString(store.t.locale, { hour: '2-digit', minute: '2-digit' });

/** Dòng phạm vi mẫu — BẮT BUỘC dưới mọi số liệu tổng hợp. Bảng thiếu một máy
 *  trông vẫn hoàn toàn bình thường; không có dòng này thì không ai phát hiện. */
export function coverageHTML(c) {
  const t = store.t;
  if (!c) return '';
  if (c.complete) return `<div class="coverage">${t.complete(c.machines_total)}</div>`;
  const miss = (c.machines_missing || []).map(m => `${m.machine} (${m.reason})`);
  const deg = (c.machines_degraded || []).map(m => `${m.machine} (${m.reason})`);
  return `<div class="coverage miss">` +
    [miss.length ? t.missing(miss.length, c.machines_total) + miss.join(' · ') : '',
     deg.length ? t.partial(deg.length, c.machines_total) + deg.join(' · ') : '']
      .filter(Boolean).join('<br>') + `</div>`;
}

/* ── Tầng API ───────────────────────────────────────────────────────────────
   Một lần gọi hụt KHÔNG được xoá trắng dữ liệu cũ. Giữ bản cuối cùng lấy được
   kèm dấu thời gian — màn hình trống không nói gì, còn số cũ có ghi giờ thì
   người trực vẫn biết mình đang nhìn cái gì.                                 */

const cache = new Map();

export async function api(path, { keep = true } = {}) {
  try {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    cache.set(path, { data: d, at: Date.now() / 1000 });
    return { data: d, stale: false };
  } catch (e) {
    const hit = keep && cache.get(path);
    if (hit) return { data: hit.data, stale: true, at: hit.at, error: String(e) };
    return { data: null, stale: false, error: String(e) };
  }
}

export const post = (path, body) =>
  fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body) }).then(r => r.json());
