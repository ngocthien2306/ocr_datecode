/* ═══════════════════════════════════════════════════════════════════════════
   FactoryMap bậc 2 — sơ đồ nhà máy 3D thật (three.js).

   Cùng interface với bản đẳng cự: render(el, {machines, selected, onSelect}).
   Đó là lý do đổi được bậc mà không đụng vào app.js.

   Vì sao dựng hình bằng CODE chứ không tải model 3D: mỗi dây chuyền ở đây là
   một băng tải + tủ soi OCR + cổng loại + tủ điện. Dựng bằng code thì thêm máy
   thứ sáu là thêm một dòng trong machines.json, còn tải model là phải có ai đó
   mở phần mềm 3D lên mỗi lần nhà máy đổi bố trí.

   Ba điều bản 2D không làm được, và là lý do đáng dựng bậc này:
     - thấy được máy nằm ở đâu so với lối đi và kho, tức là đi bộ tới thế nào
     - đèn tháp trên tủ điện là thứ công nhân xưởng vốn đã quen nhìn
     - xoay được, nên che khuất không còn là vấn đề như ở góc đẳng cự cố định

   three.js nằm ở static/vendor (2,1 MB), KHÔNG lấy từ CDN: cụm máy này sống
   trong tailnet, và một sơ đồ mặt bằng không được phép phụ thuộc vào việc nhà
   máy có ra được internet hay không.
   ═══════════════════════════════════════════════════════════════════════════ */

import { store, esc } from './core.js';
import * as flat from './factory-map.js';

/* Chú giải dùng chung với bậc 1: cùng 4 trạng thái, cùng cách gọi tên. Viết lại
   một bộ chữ thứ hai là cách chắc chắn để hai bản nói khác nhau về cùng một máy. */
export const legendHTML = flat.legendHTML;

const SPAN_X = 14.5;   // khoảng cách giữa hai dây chuyền cùng dãy (mét)
const AISLE  = 15.0;   // bề rộng từ tim dãy này sang tim dãy kia

let TH = null;         // module three, nạp một lần
let S = null;          // {renderer, scene, camera, controls, root, lines}
let booting = false;
let broken = false;    // WebGL hỏng / nạp thư viện lỗi → lùi về bản đẳng cự

/* ── Vật liệu ────────────────────────────────────────────────────────────── */

function materials(T) {
  const std = (name, o) => new T.MeshStandardMaterial({ name, ...o });
  return {
    floor:    std('floor',        { color: 0x2b3440, roughness: .94, metalness: .02 }),
    aisle:    std('aisle_paint',  { color: 0x39434f, roughness: .85 }),
    steel:    std('steel',        { color: 0xb7bcc2, roughness: .42, metalness: .34 }),
    panel:    std('panel',        { color: 0xe4e6e9, roughness: .55, metalness: .12 }),
    accent:   std('steel_blue',   { color: 0x5980a6, roughness: .48, metalness: .22 }),
    dark:     std('dark_trim',    { color: 0x3f464e, roughness: .6,  metalness: .2 }),
    belt:     std('belt',         { color: 0x6b7078, roughness: .9 }),
    guard:    std('guard_yellow', { color: 0xd8ab2c, roughness: .7 }),
    glass:    std('glass',        { color: 0xd7e2ec, roughness: .18, metalness: .1,
                                    transparent: true, opacity: .42 }),
    carton:   std('carton',       { color: 0xc9a882, roughness: .95 }),
    building: std('building',     { color: 0xb9bec5, roughness: .9 }),
    ok:       std('status_ok',    { color: 0x4f8a52, emissive: 0x4f8a52,
                                    emissiveIntensity: 1.6, roughness: .4 }),
    warn:     std('status_warn',  { color: 0xc98a20, emissive: 0xc98a20,
                                    emissiveIntensity: 1.9, roughness: .4 }),
    mute:     std('status_muted', { color: 0x8a9099, roughness: .55 }),
    sel:      std('selection',    { color: 0x5980a6, emissive: 0x5980a6,
                                    emissiveIntensity: .9, transparent: true,
                                    opacity: .85 }),
  };
}

const beaconOf = (M, state) =>
  state === 'warn' ? M.warn
  : (state === 'ok' ? M.ok : M.mute);

/* ── Nhãn máy: sprite vẽ trên canvas ─────────────────────────────────────── */

function labelSprite(T, text, sub) {
  const c = document.createElement('canvas');
  c.width = 512; c.height = 160;
  const g = c.getContext('2d');
  g.fillStyle = 'rgba(242,242,243,0.96)'; g.fillRect(0, 0, 512, 160);
  g.strokeStyle = '#5980a6'; g.lineWidth = 4; g.strokeRect(2, 2, 508, 156);
  g.fillStyle = '#1d1f20'; g.font = '700 64px "Barlow Condensed", Barlow, sans-serif';
  g.fillText(text, 22, 74);
  g.fillStyle = '#41505f'; g.font = '600 34px Barlow, sans-serif';
  g.fillText(String(sub || '').slice(0, 30), 22, 128);
  const tex = new T.CanvasTexture(c);
  tex.colorSpace = T.SRGBColorSpace;
  const sp = new T.Sprite(new T.SpriteMaterial({ map: tex, transparent: true,
                                                 depthTest: false }));
  sp.name = 'label_' + text;
  sp.scale.set(6.4, 2.0, 1);
  sp.renderOrder = 10;   // nhãn không được để tủ điện che mất
  return sp;
}

/* ── Một dây chuyền kiểm tra ─────────────────────────────────────────────── */

function inspectionLine(T, M, cfg) {
  const box = (n, w, h, d, mat, x = 0, y = 0, z = 0) => {
    const m = new T.Mesh(new T.BoxGeometry(w, h, d), mat);
    m.name = n; m.position.set(x, y, z); return m;
  };
  const cyl = (n, rt, rb, h, mat, x = 0, y = 0, z = 0, seg = 24) => {
    const m = new T.Mesh(new T.CylinderGeometry(rt, rb, h, seg), mat);
    m.name = n; m.position.set(x, y, z); return m;
  };

  const g = new T.Group();
  g.name = 'line_' + cfg.key;

  // Băng tải, dài 9 m theo trục x
  const frame = new T.Group(); frame.name = 'conveyor';
  frame.add(box('belt_surface', 9, .08, .86, M.belt, 0, .92, 0));
  frame.add(box('belt_rail_n', 9, .14, .07, M.steel, 0, 1.0, .47));
  frame.add(box('belt_rail_s', 9, .14, .07, M.steel, 0, 1.0, -.47));
  frame.add(box('belt_bed', 9, .16, .9, M.steel, 0, .82, 0));
  for (let i = -4; i <= 4; i += 2) {
    frame.add(cyl('conveyor_leg', .055, .055, .74, M.steel, i, .37, .34, 10));
    frame.add(cyl('conveyor_leg', .055, .055, .74, M.steel, i, .37, -.34, 10));
    frame.add(box('conveyor_crossbrace', .06, .06, .68, M.steel, i, .2, 0));
  }
  // Rulô kéo: trục phải nằm NGANG băng (theo z), nên xoay quanh x. Xoay quanh z
  // thì rulô nằm dọc theo hướng chạy — nhìn là biết sai ngay.
  const roller = cyl('drive_roller', .16, .16, .92, M.dark, 4.5, .9, 0, 20);
  roller.rotation.x = Math.PI / 2;
  frame.add(roller);
  g.add(frame);

  const motor = new T.Group(); motor.name = 'drive_motor';
  motor.add(cyl('motor_body', .22, .22, .5, M.dark, 0, 0, 0, 18));
  motor.rotation.z = Math.PI / 2;
  motor.position.set(4.9, .9, .58);
  g.add(motor);

  // Tủ soi OCR — chính là chỗ hệ thống này làm việc
  const cab = new T.Group(); cab.name = 'inspection_cabinet';
  cab.add(box('cabinet_body', 2.3, 1.5, 1.9, M.panel, 0, 1.65, 0));
  cab.add(box('cabinet_base', 2.4, .9, 2.0, M.steel, 0, .45, 0));
  cab.add(box('cabinet_window', 1.5, .9, .06, M.glass, 0, 1.75, .98));
  cab.add(box('cabinet_frame_top', 2.42, .1, 2.02, M.dark, 0, 2.45, 0));
  cab.add(box('cabinet_hmi', .6, .42, .06, M.dark, .78, 1.35, .99));
  cab.add(box('cabinet_hmi_screen', .5, .32, .02, M.accent, .78, 1.35, 1.03));
  cab.add(box('camera_bridge', .16, .16, 2.4, M.steel, 0, 2.62, 0));
  cab.add(cyl('camera_post_l', .07, .07, .5, M.steel, 0, 2.35, 1.1, 12));
  cab.add(cyl('camera_post_r', .07, .07, .5, M.steel, 0, 2.35, -1.1, 12));
  cab.add(box('camera_head_1', .34, .3, .42, M.dark, 0, 2.34, .3));
  cab.add(cyl('camera_lens_1', .11, .13, .24, M.steel, 0, 2.1, .3, 16));
  cab.add(box('camera_head_2', .3, .26, .36, M.dark, 0, 2.36, -.42));
  cab.add(cyl('camera_lens_2', .09, .11, .2, M.steel, 0, 2.14, -.42, 16));
  cab.add(box('light_bar', .9, .08, .1, M.panel, 0, 2.16, .62));
  cab.position.set(-.6, 0, 1.35);
  g.add(cab);

  // Cổng loại + thùng phế
  const rej = new T.Group(); rej.name = 'reject_station';
  rej.add(box('reject_arm', .9, .1, .12, M.accent, 0, 1.05, .55));
  rej.add(cyl('reject_pivot', .09, .09, .36, M.steel, -.45, .9, .55, 12));
  rej.add(box('reject_bin', 1.0, .85, 1.0, M.steel, .2, .43, 1.5));
  rej.add(box('reject_bin_rim', 1.06, .06, 1.06, M.dark, .2, .88, 1.5));
  rej.position.set(2.5, 0, 0);
  g.add(rej);

  // Tủ điện + đèn tháp. Đèn là thứ mang trạng thái, nên nó được giữ lại để
  // đổi vật liệu khi máy đổi trạng thái — không dựng lại cả cảnh.
  const ctl = new T.Group(); ctl.name = 'control_cabinet';
  ctl.add(box('control_body', 1.0, 2.0, .7, M.panel, 0, 1.0, 0));
  ctl.add(box('control_door_seam', .03, 1.7, .72, M.dark, 0, 1.05, 0));
  ctl.add(box('control_plinth', 1.05, .12, .75, M.dark, 0, .06, 0));
  ctl.add(cyl('beacon_post', .05, .05, .7, M.steel, 0, 2.35, 0, 10));
  const beacon = cyl('status_beacon', .17, .17, .36, M.mute, 0, 2.85, 0, 20);
  ctl.add(beacon);
  ctl.add(cyl('beacon_cap', .18, .18, .06, M.dark, 0, 3.06, 0, 20));
  ctl.position.set(-4.1, 0, 1.45);
  g.add(ctl);

  const stack = new T.Group(); stack.name = 'carton_stack';
  for (let i = 0; i < 3; i++)
    stack.add(box('carton', .6, .4, .5, M.carton, i * .05, .2 + i * .4, 0));
  stack.position.set(-4.4, 0, -1.5);
  g.add(stack);

  const rail = (x, z, len) => {
    const r = new T.Group(); r.name = 'guard_rail';
    r.add(box('rail_top', len, .07, .07, M.guard, 0, 1.05, 0));
    r.add(box('rail_mid', len, .05, .05, M.guard, 0, .62, 0));
    r.add(cyl('rail_post', .05, .05, 1.1, M.guard, -len / 2 + .1, .55, 0, 10));
    r.add(cyl('rail_post', .05, .05, 1.1, M.guard, len / 2 - .1, .55, 0, 10));
    r.position.set(x, 0, z);
    return r;
  };
  g.add(rail(1.2, -1.9, 3.4), rail(-3.2, -1.9, 2.6), rail(3.4, 2.6, 2.4));

  /* Trạng thái phải nhận ra được bằng HÌNH, không chỉ bằng màu đèn: vòng cảnh
     báo dưới sàn và chảo "trợ lý tắt" là hai dấu hiệu hình học riêng. */
  const ring = new T.Mesh(new T.RingGeometry(3.6, 4.0, 48), M.warn);
  ring.name = 'attention_ring';
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(0, .02, .6);
  ring.visible = false;
  g.add(ring);

  const dish = cyl('assistant_offline_marker', .02, .34, .5, M.mute, -4.1, 3.4, 1.45, 14);
  dish.visible = false;
  g.add(dish);

  const selRing = new T.Mesh(new T.RingGeometry(4.3, 4.7, 56), M.sel);
  selRing.name = 'selection_ring';
  selRing.rotation.x = -Math.PI / 2;
  selRing.position.set(0, .03, .6);
  selRing.visible = false;
  g.add(selRing);

  const label = labelSprite(T, cfg.name, cfg.sub);
  label.position.set(-1.0, 4.7, .4);
  g.add(label);

  g.position.set(cfg.x, 0, cfg.z);
  g.rotation.y = cfg.rotation || 0;

  // Chỉ mesh trong dây chuyền mới nhận raycast — sàn, tường, kho thì không.
  g.traverse(o => {
    if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; o.userData.node = cfg.node; }
  });
  return { group: g, beacon, ring, dish, selRing, label, cfg };
}

/* ── Nền xưởng ───────────────────────────────────────────────────────────── */

function hall(T, M, halfX, halfZ) {
  const box = (n, w, h, d, mat, x = 0, y = 0, z = 0) => {
    const m = new T.Mesh(new T.BoxGeometry(w, h, d), mat);
    m.name = n; m.position.set(x, y, z); return m;
  };
  const g = new T.Group(); g.name = 'hall';
  const W = halfX * 2, D = halfZ * 2;

  const slab = box('floor_slab', W, .2, D, M.floor, 0, -.1, 0);
  slab.receiveShadow = true;
  g.add(slab);
  g.add(box('aisle_marking', W - 4, .02, 2.6, M.aisle, 0, .01, 0));
  for (let x = -halfX + 3; x <= halfX - 3; x += 2.5) {
    g.add(box('aisle_chevron', .7, .03, .14, M.guard, x, .02, 1.25));
    g.add(box('aisle_chevron', .7, .03, .14, M.guard, x, .02, -1.25));
  }
  g.add(box('wall_north', W, 1.1, .3, M.building, 0, .55, -halfZ));
  g.add(box('wall_south', W, 1.1, .3, M.building, 0, .55, halfZ));
  g.add(box('wall_west', .3, 1.1, D, M.building, -halfX, .55, 0));
  g.add(box('wall_east', .3, 1.1, D, M.building, halfX, .55, 0));
  return g;
}

function store3d(T, M, name, x, z, text, sub) {
  const box = (n, w, h, d, mat, px = 0, py = 0, pz = 0) => {
    const m = new T.Mesh(new T.BoxGeometry(w, h, d), mat);
    m.name = n; m.position.set(px, py, pz); return m;
  };
  const g = new T.Group(); g.name = name;
  g.add(box('store_body', 7, 3.2, 5, M.building, 0, 1.6, 0));
  g.add(box('store_roof', 7.3, .16, 5.3, M.dark, 0, 3.28, 0));
  g.add(box('store_door', 2.2, 2.2, .1, M.accent, 0, 1.1, 2.52));
  const lb = labelSprite(T, text, sub);
  lb.scale.set(5.2, 1.6, 1);
  lb.position.set(0, 4.6, 0);
  g.add(lb);
  g.position.set(x, 0, z);
  g.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return g;
}

/* ── Vị trí: machines.json → toạ độ mét ──────────────────────────────────── */

function place(machines) {
  const p = machines.filter(m => m.floor);
  if (!p.length) return [];
  const xs = p.map(m => m.floor.x), ys = p.map(m => m.floor.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const cy = (y0 + y1) / 2;
  const zScale = y1 > y0 ? AISLE / (y1 - y0) : 0;
  return p.map(m => ({
    node: m.node_id,
    key: String(m.name).toLowerCase().replace(/[^a-z0-9]+/g, '_'),
    name: m.name,
    x: (m.floor.x - cx) * SPAN_X,
    z: (m.floor.y - cy) * zScale,
    // Dãy bên kia lối đi quay 180° để hai dãy cùng hướng mặt ra lối đi — đó là
    // cách người ta thật sự bố trí xưởng, và nó làm lối đi đọc ra là lối đi.
    rotation: m.floor.rotation || ((m.floor.y - cy) > 0 ? Math.PI : 0),
  }));
}

const subOf = m => {
  const rec = (m.recipes || [])[0];
  return [m.line, rec && rec.name].filter(Boolean).join(' · ') || '—';
};

/* ── Dựng cảnh ───────────────────────────────────────────────────────────── */

async function boot(el, machines) {
  const [T, oc] = await Promise.all([
    import('three'),
    import('three/addons/controls/OrbitControls.js'),
  ]);
  TH = T;

  const renderer = new T.WebGLRenderer({ antialias: true, alpha: true,
                                         preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = T.PCFSoftShadowMap;
  /* Canvas tự khai kích thước ở đây chứ không trông vào app.css: kích thước là
     điều kiện để cảnh này tồn tại, còn app.css là chỗ nhiều người cùng sửa. Đã
     mất một vòng vì đúng chuyện đó — canvas nở ra 1462×1300 và tràn khỏi panel. */
  const cv = renderer.domElement;
  cv.id = 'factory3d';
  Object.assign(cv.style, { display: 'block', width: '100%', height: '100%',
                            cursor: 'grab' });
  el.innerHTML = '';
  if (!el.style.height && el.clientHeight < 240) el.style.height = '460px';
  el.style.padding = '0';
  el.appendChild(cv);

  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(45, 1, .1, 500);
  const controls = new oc.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = .08;
  controls.minDistance = 16;
  controls.maxDistance = 120;
  controls.maxPolarAngle = Math.PI * .49;   // không cho chui xuống dưới sàn

  scene.add(new T.HemisphereLight(0xffffff, 0xd8d2c4, 1.0));
  const key = new T.DirectionalLight(0xffffff, 2.2);
  key.position.set(26, 40, 24);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.bias = -0.0004;
  // Khung bóng phải trùm cả xưởng ~46×30 m. Để mặc định (±5 m) thì chỉ một máy
  // có bóng, bốn máy còn lại trông như đang lơ lửng.
  Object.assign(key.shadow.camera, { left: -34, right: 34, top: 34,
                                     bottom: -34, near: 1, far: 140 });
  key.shadow.camera.updateProjectionMatrix();
  scene.add(key);
  const fill = new T.DirectionalLight(0xfff4e6, .5);
  fill.position.set(-28, 18, -22);
  scene.add(fill);

  const M = materials(T);
  const root = new T.Group(); root.name = 'factory_floor';
  scene.add(root);

  const spots = place(machines);
  const halfX = Math.max(23, ...spots.map(s => Math.abs(s.x) + 8));
  const halfZ = 15;
  root.add(hall(T, M, halfX, halfZ));
  root.add(store3d(T, M, 'raw_material_store', -halfX + 5, 11, 'KHO NL',
                   'Kho nguyên liệu'));
  root.add(store3d(T, M, 'finished_goods_store', halfX - 5, 11, 'KHO TP',
                   'Kho thành phẩm'));

  const grid = new T.GridHelper(200, 80, 0x5980a6, 0x416180);
  grid.name = 'reference_grid';
  grid.position.y = -.24;
  grid.material.opacity = .35;
  grid.material.transparent = true;
  scene.add(grid);

  const lines = new Map();
  for (const s of spots) {
    const m = machines.find(x => x.node_id === s.node);
    const built = inspectionLine(T, M, { ...s, sub: subOf(m) });
    root.add(built.group);
    lines.set(s.node, built);
  }

  /* Khung hình tính từ HỘP BAO của cảnh, không đặt bằng số cố định: xưởng rộng
     bao nhiêu là do machines.json quyết định, và panel rộng bao nhiêu là do
     trình duyệt. Đặt cứng thì thêm một máy nữa là máy đó nằm ngoài khung. */
  const frame = () => {
    const b = new T.Box3().setFromObject(root);
    const size = b.getSize(new T.Vector3());
    const mid = b.getCenter(new T.Vector3());
    const fov = camera.fov * Math.PI / 180;
    const distV = (size.y / 2) / Math.tan(fov / 2);
    const distH = (Math.max(size.x, size.z) / 2)
                  / Math.tan(Math.atan(Math.tan(fov / 2) * camera.aspect));
    const d = Math.max(distV, distH) * 1.15;
    controls.target.set(mid.x, 1.5, mid.z);
    // Góc nhìn ba-phần-tư: đủ cao để đọc bố trí, đủ thấp để còn thấy tủ soi và
    // đèn tháp dựng đứng — nhìn thẳng từ trên xuống thì mất hết chiều cao máy.
    camera.position.set(mid.x + d * .62, d * .46, mid.z + d * .70);
    camera.updateProjectionMatrix();
    controls.update();
  };

  S = { T, M, renderer, scene, camera, controls, root, lines, el, raf: null,
        onSelect: null, subs: new Map() };

  // Kích thước bám panel, không bám cửa sổ.
  let framed = false;
  const fit = () => {
    const w = el.clientWidth || 600, h = el.clientHeight || 420;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    // Chỉ tự đóng khung LẦN ĐẦU. Sau đó người dùng đã xoay/zoom theo ý họ, kéo
    // camera về chỗ cũ mỗi lần đổi kích thước là giật mất quyền điều khiển.
    if (!framed) { frame(); framed = true; }
  };
  fit();
  new ResizeObserver(fit).observe(el);

  bindPicking(el);

  /* Vòng vẽ chỉ chạy khi sơ đồ CÒN NHÌN THẤY. Một canvas WebGL quay vô ích dưới
     tab đang ẩn vẫn ăn GPU và pin — mà màn hình này thường mở cả ca. */
  const io = new IntersectionObserver(([e]) => {
    if (e.isIntersecting) start(); else stop();
  }, { threshold: 0 });
  io.observe(el);
  document.addEventListener('visibilitychange',
    () => (document.hidden ? stop() : start()));
  start();
}

function start() {
  if (!S || S.raf) return;
  const loop = () => {
    S.raf = requestAnimationFrame(loop);
    S.controls.update();
    S.renderer.render(S.scene, S.camera);
  };
  S.raf = requestAnimationFrame(loop);
}

function stop() {
  if (S && S.raf) { cancelAnimationFrame(S.raf); S.raf = null; }
}

/* ── Bấm chọn máy ────────────────────────────────────────────────────────── */

function bindPicking(el) {
  const ray = new TH.Raycaster();
  const v = new TH.Vector2();
  let down = null;

  el.addEventListener('pointerdown', e => (down = [e.clientX, e.clientY]));
  el.addEventListener('pointerup', e => {
    if (!down) return;
    const moved = Math.hypot(e.clientX - down[0], e.clientY - down[1]);
    down = null;
    // Kéo để xoay KHÔNG được tính là bấm chọn: xoay xong mà ngăn kéo bật ra
    // che mất cảnh thì không ai xoay nữa.
    if (moved > 5) return;

    const r = el.getBoundingClientRect();
    v.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    v.y = -((e.clientY - r.top) / r.height) * 2 + 1;
    ray.setFromCamera(v, S.camera);
    for (const hit of ray.intersectObject(S.root, true)) {
      const node = hit.object.userData.node;
      if (node) { S.onSelect && S.onSelect(node); return; }
    }
  });

  el.addEventListener('pointermove', e => {
    const r = el.getBoundingClientRect();
    v.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    v.y = -((e.clientY - r.top) / r.height) * 2 + 1;
    ray.setFromCamera(v, S.camera);
    const hit = ray.intersectObject(S.root, true)
      .find(h => h.object.userData.node);
    el.style.cursor = hit ? 'pointer' : 'grab';
  });
}

/* ── Cập nhật theo dữ liệu ───────────────────────────────────────────────── */

function apply(machines, selected) {
  for (const [node, L] of S.lines) {
    const m = machines.find(x => x.node_id === node);
    if (!m) continue;
    L.beacon.material = beaconOf(S.M, m.state);
    L.ring.visible = m.state === 'warn';
    L.dish.visible = m.state === 'agent_down';
    L.selRing.visible = node === selected;
    // Máy không với tới được thì mờ hẳn cả cụm — nhìn một cái là biết chỗ đó
    // đang không có số liệu, chứ không phải đang chạy tốt.
    const dim = m.state === 'unreachable' || m.state === 'offline';
    L.group.traverse(o => {
      if (o.isMesh && o !== L.beacon) {
        o.material.transparent = dim || o.material.name === 'glass';
        o.material.opacity = dim ? .45
          : (o.material.name === 'glass' ? .42 : 1);
      }
    });
    // Nhãn chỉ vẽ lại khi CHỮ đổi: mỗi lần vẽ là một texture mới lên GPU, mà
    // hàm này chạy 30 giây một lần suốt ca.
    const sub = subOf(m);
    if (S.subs.get(node) !== sub) {
      S.subs.set(node, sub);
      const fresh = labelSprite(S.T, m.name, sub);
      fresh.position.copy(L.label.position);
      L.group.remove(L.label);
      L.label.material.map.dispose();
      L.label.material.dispose();
      L.group.add(fresh);
      L.label = fresh;
    }
  }
}

/* ── Interface công khai ─────────────────────────────────────────────────── */

export function render(el, { machines, selected, onSelect }) {
  if (broken) return flat.render(el, { machines, selected, onSelect });

  if (!S) {
    if (booting) return;
    booting = true;
    // Trong lúc nạp thư viện, vẫn vẽ bản đẳng cự — màn hình không được trống
    // vài giây chỉ vì đang tải 2 MB.
    flat.render(el, { machines, selected, onSelect });
    boot(el, machines)
      .then(() => { booting = false; S.onSelect = onSelect; apply(machines, selected); })
      .catch(err => {
        // WebGL bị tắt, GPU cũ, hay thiếu file vendor — lùi về bậc 1 và nói ra.
        console.warn('[factory-3d] không dựng được cảnh 3D, dùng bản đẳng cự:', err);
        broken = true; booting = false;
        flat.render(el, { machines, selected, onSelect });
      });
    return;
  }
  S.onSelect = onSelect;
  apply(machines, selected);
}

/** Trả về true nếu đang thực sự chạy 3D — để giao diện nói đúng "bậc" nào. */
export const is3D = () => !!S && !broken;
