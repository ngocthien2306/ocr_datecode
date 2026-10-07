/* ═══════════════════════════════════════════════════════════════════════════
   DÙNG CHUNG giữa fleet_service và agent_service. Sửa ở đây, không copy.

   Trước đây file này tồn tại hai bản 1.282 dòng — một cho Fleet Console, một
   cho Line Station — và chúng đã bắt đầu trôi khác nhau (80 dòng) chỉ sau một
   ngày. Bản của station là tập cha: nó thêm `enabledKeys`, và khi không truyền
   thì hành vi giống hệt bản fleet.

   Đặt ở `shared/web/` chứ không nhét vào một service: cả hai service đều đọc từ
   ĐĨA CỦA CHÍNH MÁY mình, nên Line Station vẫn không phụ thuộc fleet service
   lúc chạy — đó là điều kiện bắt buộc của bề mặt đó, và dùng chung code không
   phá nó.
   ═══════════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════════
   FactoryMap — sơ đồ nhà máy 3D tương tác.

   Giữ nguyên public interface của `factory-map.js` để bên gọi vẫn chỉ biết
   render({ machines, selected, onSelect }). Three.js tải lười: nếu tablet ở
   xưởng không ra Internet/CDN, sơ đồ SVG bậc 1 vẫn là phương án dự phòng.
   ═══════════════════════════════════════════════════════════════════════════ */

/* `store` được TRUYỀN VÀO qua render(), không import.
 *
 * Đây là điều kiện để một file dùng chung cho hai service: nếu nó import
 * './core.js' thì mỗi bên phải đặt một core.js cạnh nó, và thế là lại có hai
 * bản phải giữ đồng bộ — đúng cái vấn đề file này sinh ra để bỏ.
 *
 * Module chỉ cần ba thứ: `theme`, `t.floorTitle`, `t.state`. */
let store = {
  theme: 'light',
  t: { floorTitle: 'Factory floor', state: {} },
};

/* three.js đọc từ ĐĨA CỦA MÁY, không phải unpkg.com.
 *
 * Bản r184 đã được vendor vào `shared/web/vendor/three/` từ trước nhưng hằng số
 * này vẫn trỏ ra CDN — nên cả hai bề mặt vẫn tải 2 MB qua Internet. Ở Fleet
 * Console trên máy Mac thì không ai thấy; ở Line Station — màn hình cạnh dây
 * chuyền, trên máy trong xưởng có thể không có đường ra ngoài — nó là chênh
 * lệch giữa CÓ sơ đồ và KHÔNG.
 *
 * Đường dẫn tương đối so với chính file này, nên hai service không cần khai
 * hằng số riêng: /shared/factory-map-3d.js → /shared/vendor/three/... */
import { PLANT, OPEN_YAW } from './factory-plan.js';

const THREE_URL = new URL('./vendor/three/three.module.js', import.meta.url).href;
const pending = new WeakMap();
const scenes = new WeakMap();
let threePromise;

/* Bảng màu RIÊNG cho máy bị tắt — không phải bảng màu rồi giảm opacity. Giảm
   opacity làm mất chiều sâu của khối, và một khối bẹt thì không còn đọc ra là
   máy. Ở Line Station, mọi máy trừ máy của chính station đều dùng bảng này. */
const DISABLED = { beacon: null, body: '#c8ccd0', trim: '#e4e6e9' };

/* Danh sách máy ĐƯỢC thao tác. `null` = tất cả (Fleet Console). Line Station
   truyền đúng một tên: máy của chính nó. */
let ENABLED = null;
/* Bản đồ nhỏ chỉ hợp với màn hình lớn của Fleet Console. Ở Line Station thì
   khung sơ đồ đã nhỏ, và người đứng đó chỉ cần biết MÁY CỦA MÌNH nằm đâu —
   một bản đồ toàn xưởng ở góc là mời họ so sánh với line khác, đúng thứ
   floor.js cố tình tránh (xem chú thích đầu file đó). */
let MINIMAP = true;

const isEnabled = name => !ENABLED || ENABLED.includes(name);

/* Loader cho sơ đồ bậc 1, do bên gọi cấp qua `options.fallback`. */
let fallbackLoader = null;

const STATUS = {
  ok:          { beacon: '#2f8a58', body: '#86aaca', trim: '#d7e8f5' },
  warn:        { beacon: '#b57a00', body: '#c7a26a', trim: '#f6e7c8' },
  agent_down:  { beacon: '#8c9299', body: '#9ea8b2', trim: '#e5e7e9' },
  unreachable: { beacon: '#8c9299', body: '#9ea8b2', trim: '#e5e7e9' },
  offline:     { beacon: '#8c9299', body: '#9ea8b2', trim: '#e5e7e9' },
};

/* Mỗi carton line có bao bì khác nhau. Màu/nhãn là cue trực quan trên sơ đồ,
   không phải dữ liệu batch hay mô phỏng nhãn thương mại thật. */
const LINE_PRODUCTS = {
  'Line 1': { label: 'ONION POWDER', pack: '#c89a50', tape: '#f1d389', mark: '#7d5424', shape: [0.74, 0.58, 0.68] },
  'Line 3': { label: 'CHILI POWDER', pack: '#bd5139', tape: '#f2c95c', mark: '#7f271d', shape: [0.82, 0.5, 0.64] },
  'Line 4': { label: 'CINNAMON', pack: '#805034', tape: '#d59b43', mark: '#e9c39a', shape: [0.68, 0.72, 0.76] },
  'Line 2': { label: 'PURE SEA SALT', pack: '#e2e9eb', tape: '#3d8aa5', mark: '#1e5d73', shape: [0.72, 0.62, 0.72] },
  'Line 7': { label: 'CARTON', pack: '#6b7fa8', tape: '#d7e0f2', mark: '#263c65', shape: [0.8, 0.54, 0.7] },
  'Line 5': { label: 'TIN CARTON', pack: '#61997c', tape: '#e5d36e', mark: '#24563e', shape: [0.7, 0.68, 0.74] },
};

const OFFLINE_CARTON_LINES = [
  { node_id: '__offline_auto_1__', name: 'Line 7', line: 'West bay',
    model: 'Jetson Orin Nano 8GB Super', state: 'offline', floor: { x: 0, y: 0 }, virtual: true },
  { node_id: '__offline_tin_2__', name: 'Line 5', line: 'Annex building',
    model: 'Jetson Orin Nano 8GB Super', state: 'offline', floor: { x: 0, y: 0 }, virtual: true, building: 'tin2' },
];

// Mặt bằng nằm ở factory-plan.js — xem chú thích đầu file đó.
/* Style của minimap do chính module chèn, một lần. `.factory-zone-nav` thì nằm
   trong CSS của từng app — nhưng đã có HAI bản (fleet + station), và thêm bản
   thứ ba cho một widget mà module tự dựng chỉ tạo thêm chỗ để lệch nhau. */
function ensureMiniStyle() {
  if (document.getElementById('factory-mini-style')) return;
  const style = document.createElement('style');
  style.id = 'factory-mini-style';
  style.textContent = `
/* Dải dưới đã có .factory-zone-nav (trái) và .factory-view-nav (phải), nút ở
   station cao 40px — nên minimap nằm TRÊN dải đó, không cạnh nó. */
.factory-mini{
  position:absolute; left:12px; bottom:62px; width:260px; height:220px;
  border:1px solid rgba(128,145,160,.45); border-radius:10px; cursor:pointer;
  box-shadow:0 6px 20px rgba(12,20,28,.18); backdrop-filter:blur(2px);
}
.factory-mini:hover{ border-color:rgba(128,145,160,.75) }
@media (max-width:700px){ .factory-mini{ width:168px; height:142px; bottom:58px } }`;
  document.head.append(style);
}

function loadThree() {
  if (!threePromise) threePromise = import(THREE_URL);
  return threePromise;
}

function disposeObject(root) {
  const disposed = new Set();
  root.traverse(node => {
    if (node.geometry && !disposed.has(node.geometry)) {
      node.geometry.dispose();
      disposed.add(node.geometry);
    }
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    materials.forEach(material => {
      if (!material || disposed.has(material)) return;
      material.map?.dispose();
      material.dispose();
      disposed.add(material);
    });
  });
  root.clear();
}

function textSprite(THREE, title, subtitle, dark) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  const scale = 2;
  canvas.width = 360 * scale;
  canvas.height = 94 * scale;
  context.scale(scale, scale);
  context.font = '600 20px Barlow, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  const width = Math.min(172, Math.max(92, context.measureText(title).width + 34));
  context.fillStyle = dark ? 'rgba(27, 31, 35, .88)' : 'rgba(255, 255, 255, .92)';
  context.strokeStyle = dark ? 'rgba(255,255,255,.12)' : 'rgba(28, 32, 36, .12)';
  context.lineWidth = 1;
  context.beginPath();
  context.roundRect((180 - width / 2), 10, width, 74, 8);
  context.fill();
  context.stroke();
  context.fillStyle = dark ? '#f2f5f7' : '#202326';
  context.fillText(title, 180, 37);
  context.font = '500 13px Barlow, sans-serif';
  context.fillStyle = dark ? '#aab3bc' : '#70767d';
  context.fillText(subtitle || '', 180, 61);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
  sprite.scale.set(3.8, 1, 1);
  return sprite;
}

class FactoryFloorScene {
  constructor(THREE, el) {
    this.THREE = THREE;
    this.el = el;
    this.machineRoot = new THREE.Group();
    this.pickables = [];
    this.productAnimators = [];
    this.robotAnimators = [];
    this.peopleAnimators = [];
    this.lastAnimatedAt = 0;
    // Khung mặc định ôm sảnh chính (78×64) chứ không phải cụm 2×3 cũ.
    this.cameraTarget = new THREE.Vector3(PLANT.hall.x, 0, PLANT.hall.z);
    this.cameraMove = null;
    this.distance = 108;
    // Máy quay đứng phía +z nhìn về −z: hướng bắc (Line 5, cổng vào) nằm ở
    // PHÍA TRÊN màn hình, khớp với minimap và với bản vẽ mặt bằng. Giá trị
    // âm cũ lật ngược trục bắc–nam giữa hai khung nhìn.
    this.azimuth = 0.72;
    this.elevation = 0.72;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.drag = null;

    el.replaceChildren();
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'factory-3d-canvas';
    this.canvas.setAttribute('role', 'img');
    this.canvas.setAttribute('aria-label', store.t.floorTitle);
    el.append(this.canvas);
    this.zoneNav = document.createElement('nav');
    this.zoneNav.className = 'factory-zone-nav';
    this.zoneNav.setAttribute('aria-label', 'Factory building navigation');
    this.zoneNav.innerHTML = '<button type="button" data-focus="main" aria-pressed="true">Main hall</button><button type="button" data-focus="tin2" aria-pressed="false">Line 5 annex</button>';
    this.zoneNav.addEventListener('click', event => {
      const button = event.target.closest('button[data-focus]');
      if (button) this.focusZone(button.dataset.focus);
    });
    el.append(this.zoneNav);

    /* Thanh góc nhìn. Xoay bằng chuột thì dễ lạc — nhất là trên màn hình cạnh
       dây chuyền, nơi người ta chạm một cái rồi đi. Ba nút này là đường về. */
    this.viewNav = document.createElement('nav');
    this.viewNav.className = 'factory-zone-nav factory-view-nav';
    this.viewNav.setAttribute('aria-label', 'Camera views');
    this.viewNav.innerHTML =
      '<button type="button" data-view="iso" aria-pressed="true" title="3/4 view (0)">3D</button>'
      + '<button type="button" data-view="top" aria-pressed="false" title="Top-down (1)">Top</button>'
      + '<button type="button" data-view="front" aria-pressed="false" title="Eye level (2)">Eye</button>'
      + '<button type="button" data-zoom="-1" title="Zoom in (+)">+</button>'
      + '<button type="button" data-zoom="1" title="Zoom out (\u2212)">\u2212</button>'
      + (MINIMAP ? '<button type="button" data-mini="1" aria-pressed="true" title="Overview map (M)">Map</button>' : '');
    this.viewNav.addEventListener('click', event => {
      const v = event.target.closest('button[data-view]');
      if (v) { this.setView(v.dataset.view); return; }
      const z = event.target.closest('button[data-zoom]');
      if (z) { this.zoomBy(Number(z.dataset.zoom) * 6); return; }
      if (event.target.closest('button[data-mini]')) this.toggleMini();
    });
    el.append(this.viewNav);

    /* Bản đồ 2D góc dưới. Cảnh 3D đẹp nhưng xoay một lúc là mất phương hướng —
       nhất là khi Line 5 nằm ngoài sảnh chính. Minimap vẽ từ CÙNG dữ liệu PLANT
       với cảnh 3D, nên không bao giờ lệch nhau; chấm máy tô theo trạng thái, và
       hình quạt cho biết camera đang nhìn từ đâu. Bấm vào là bay tới máy đó. */
    if (MINIMAP) {
      ensureMiniStyle();
      this.mini = document.createElement('canvas');
      this.mini.className = 'factory-mini';
      this.mini.width = 260;
      this.mini.height = 220;
      this.mini.setAttribute('role', 'img');
      this.mini.setAttribute('aria-label', 'Factory floor overview map');
      this.mini.addEventListener('click', event => this.miniClick(event));
      this.mini.addEventListener('mousemove', event => this.miniHover(event));
      this.mini.addEventListener('mouseleave', () => this.setHover(null));
      el.append(this.mini);
      this.miniBtn = this.viewNav.querySelector('button[data-mini]');
      // Màn hình hẹp thì có người muốn tắt hẳn bản đồ nhỏ. Nhớ lựa chọn đó, vì
      // họ chạm một cái rồi đi làm việc khác.
      let saved = '1';
      try { saved = localStorage.getItem('factory_mini') ?? '1'; } catch { /* chế độ riêng tư */ }
      this.toggleMini(saved !== '0');
    }

    /* Thẻ ảnh nổi khi rê chuột lên một máy. Cái tủ soi OCR trong sơ đồ trở
       thành một cửa sổ thật: chỉ tay vào máy là thấy nó vừa chụp được gì, khỏi
       phải mở drawer rồi đóng lại chỉ để liếc một cái. */
    this.peek = document.createElement('div');
    this.peek.className = 'map-peek';
    this.peek.hidden = true;
    el.append(this.peek);
    this.peekCache = new Map();   // máy → dữ liệu khung ảnh, tránh hỏi lại
    this.peekFor = null;

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;

    this.scene = new THREE.Scene();
    // Mặt bằng mới rộng 78×64 cộng nhà Line 5 ở z ≈ −52; far 140 cắt mất nửa
    // cảnh khi camera lùi ra đủ xa để thấy hết.
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.1, 420);
    this.scene.add(this.machineRoot);

    /* Ánh sáng. Bản trước cộng hemi 2.1 + key 2.8 + fill 1.1: tổng nền quá cao
       nên bóng đổ gần như biến mất, và mọi bề mặt về cùng một độ sáng — đó là
       lý do cảnh trông bẹt. Hạ nền xuống, dồn năng lượng vào MỘT nguồn chính
       thì mới có bóng, mà bóng mới là thứ nói cho mắt biết vật nào đứng trên
       sàn và vật nào cao bao nhiêu. */
    this.scene.environment = this.makeEnvironment();
    if ('environmentIntensity' in this.scene) this.scene.environmentIntensity = 0.55;

    const hemi = new THREE.HemisphereLight(0xf3f8ff, 0x93a0ad, 0.75);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xfff4e3, 3.1);
    key.position.set(16, 26, 12);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -42;
    key.shadow.camera.right = 42;
    key.shadow.camera.top = 42;
    key.shadow.camera.bottom = -42;
    key.shadow.camera.far = 90;
    // normalBias khử vân sọc trên mặt cong (chai, rulô) — thiếu nó thì bề mặt
    // tự đổ bóng lên chính nó thành những vệt tối chạy dọc.
    key.shadow.normalBias = 0.035;
    key.shadow.bias = -0.0004;
    key.shadow.camera.updateProjectionMatrix();
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xa8ccec, 0.5);
    fill.position.set(-18, 9, -14);
    this.scene.add(fill);
    // Đèn hắt ngược nhẹ để viền máy không chìm hẳn vào nền.
    const rim = new THREE.DirectionalLight(0xffffff, 0.35);
    rim.position.set(-6, 12, 22);
    this.scene.add(rim);

    this.renderer.toneMappingExposure = 1.0;

    this.floorTheme = store.theme;
    this.floor = this.makeFloor();
    this.scene.add(this.floor);
    this.bindEvents();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(el);
    this.resize();
    this.animationFrame = requestAnimationFrame(now => this.animate(now));
  }

  makeFloor() {
    const THREE = this.THREE;
    const floor = new THREE.Group();
    const dark = store.theme === 'dark';
    /* Sàn phải phủ cả sảnh chính (x −41…37, z −35…29) lẫn nhà Line 5 ở phía bắc
       (tâm z ≈ −52). Bản cũ là dải 102×44 lệch sang đông vì Line 5 khi đó đặt ở
       x = 55. */
    const slab = new THREE.Mesh(
      new THREE.BoxGeometry(104, 0.34, 108),
      new THREE.MeshStandardMaterial({ color: dark ? 0x19242e : 0xe7edf2, roughness: 0.96 }),
    );
    slab.position.x = 0;
    slab.position.z = -14;
    slab.position.y = -0.25;
    slab.receiveShadow = true;
    floor.add(slab);
    const grid = new THREE.GridHelper(108, 108, dark ? 0x536373 : 0xc2ced8, dark ? 0x344454 : 0xd7e0e7);
    grid.position.x = 0;
    grid.position.z = -14;
    grid.position.y = -0.055;
    grid.material.transparent = true;
    grid.material.opacity = dark ? 0.42 : 0.75;
    floor.add(grid);

    const aisleMaterial = new THREE.MeshBasicMaterial({ color: dark ? 0xa5b4c2 : 0x9aa9b7, transparent: true, opacity: 0.58 });
    const aisleSurface = new THREE.MeshBasicMaterial({ color: dark ? 0x263b4a : 0xd4dfe8, transparent: true, opacity: 0.7 });

    /* Mạng hành lang, không phải một lối đi thẳng. Mỗi đoạn nối cửa của các khu
       với nhau theo đúng bản vẽ: cổng vào → phòng vận hành → kho, và một trục
       dọc chạy trước cửa từng dãy máy. Đây là thứ quyết định cảnh có đọc được
       như một nhà máy hay không — máy đặt đúng chỗ mà không có đường đi tới thì
       vẫn là một bức tranh trang trí. */
    const CORRIDORS = [
      { x: -21.5, z:   0.5, w:  3.4, d: 42 },   // trục dọc phía tây
      { x:  -3.5, z:  -6.0, w:  3.4, d: 30 },   // trục dọc trước Line 1 / Line 2
      { x:   4.0, z: -30.0, w: 26.0, d: 3.4 },  // cổng vào → phòng vận hành
      { x:  10.0, z: -12.0, w:  3.4, d: 32 },   // phòng vận hành → kho
      { x:  -3.0, z:   7.5, w: 40.0, d: 3.4 },  // trục ngang nối tây ↔ kho ↔ M
      { x:  16.5, z:  16.0, w:  3.4, d: 22 },   // trục dọc trước Line 3 / Line 4
    ];
    for (const c of CORRIDORS) {
      const lane = new THREE.Mesh(new THREE.BoxGeometry(c.w, 0.025, c.d), aisleSurface);
      lane.position.set(c.x, 0.005, c.z);
      floor.add(lane);
      // Vạch đứt chạy dọc theo chiều dài đoạn hành lang.
      const along = c.w >= c.d;
      const length = along ? c.w : c.d;
      for (let t = -length / 2 + 0.7; t < length / 2 - 0.7; t += 1.4) {
        const dash = new THREE.Mesh(
          new THREE.BoxGeometry(along ? 0.72 : 0.1, 0.025, along ? 0.1 : 0.72), aisleMaterial);
        dash.position.set(c.x + (along ? t : 0), 0.02, c.z + (along ? 0 : t));
        floor.add(dash);
      }
    }

    // Đường ngoài trời nối xưởng chính với nhà Line 5 ở phía bắc.
    const road = new THREE.Mesh(
      new THREE.BoxGeometry(8, 0.04, 20),
      new THREE.MeshStandardMaterial({ color: dark ? 0x1c252d : 0xb9c0c5, roughness: 0.96 }),
    );
    road.position.set(26, 0.025, -40);
    floor.add(road);
    for (let z = -49; z < -31; z += 1.5) {
      const dash = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.025, 0.7), aisleMaterial);
      dash.position.set(26, 0.06, z);
      floor.add(dash);
    }
    return floor;
  }

  /** Môi trường phản chiếu: một dải chuyển màu trời→sàn, đưa qua PMREM.
   *
   *  Không có nó thì mọi vật liệu metalness cao đổ về xám phẳng — inox băng
   *  tải, khung nhôm, vỏ tủ đều trông như nhựa sơn. Dựng bằng canvas nên không
   *  phải tải thêm file HDR nào. */
  makeEnvironment() {
    const THREE = this.THREE;
    const c = document.createElement('canvas');
    c.width = 32; c.height = 128;
    const g = c.getContext('2d');
    const grad = g.createLinearGradient(0, 0, 0, 128);
    grad.addColorStop(0, '#eef4fa');   // trần sáng
    grad.addColorStop(0.48, '#c9d4de');
    grad.addColorStop(0.52, '#8d99a4');
    grad.addColorStop(1, '#5d666e');   // sàn tối
    g.fillStyle = grad; g.fillRect(0, 0, 32, 128);
    const tex = new THREE.CanvasTexture(c);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const env = pmrem.fromEquirectangular(tex).texture;
    pmrem.dispose();
    tex.dispose();
    return env;
  }

  refreshFloorTheme() {
    if (this.floorTheme === store.theme) return;
    this.scene.remove(this.floor);
    disposeObject(this.floor);
    this.floorTheme = store.theme;
    this.floor = this.makeFloor();
    this.scene.add(this.floor);
  }

  animate(now) {
    if (this.cameraMove) {
      const progress = Math.min(1, (now - this.cameraMove.started) / this.cameraMove.duration);
      const ease = 1 - (1 - progress) ** 3;
      const mv = this.cameraMove;
      this.cameraTarget.lerpVectors(mv.fromTarget, mv.toTarget, ease);
      this.distance = mv.fromDistance + (mv.toDistance - mv.fromDistance) * ease;
      // Góc cũng bay theo, nên nút "Top"/"Eye" là một cú lia mượt chứ không
      // phải một cú nhảy — nhảy thì người xem mất luôn cảm giác đang ở đâu.
      if (mv.toAzimuth != null)
        this.azimuth = mv.fromAzimuth + (mv.toAzimuth - mv.fromAzimuth) * ease;
      if (mv.toElevation != null)
        this.elevation = mv.fromElevation + (mv.toElevation - mv.fromElevation) * ease;
      if (progress === 1) this.cameraMove = null;
    }
    if (now - this.lastAnimatedAt >= 32) {
      const seconds = now / 1000;
      this.productAnimators.forEach(item => {
        if (item.update) { item.update(seconds); return; }
        const span = item.max - item.min;
        item.object.position.x = item.min + ((seconds * item.speed + item.phase) % span);
      });
      this.peopleAnimators.forEach(p => {
        const rig = p.person.userData.rig;
        if (p.kind === 'walk') {
          /* Đi theo lộ trình có ĐIỂM DỪNG: chạy hết thời gian dừng ở một chặng
             rồi mới đi tiếp chặng đó. Quãng đường cộng dồn dùng để đánh nhịp
             chân, nên bàn chân không trượt và lúc đứng thì chân cũng đứng. */
          let clock = (seconds + p.offset) % p.cycle;
          let pos = null, dir = null, walked = 0, standing = false;
          for (let i = 0; i < p.legs.length; i++) {
            const leg = p.legs[i];
            const wait = leg.a.pause || 0;
            if (clock < wait) {
              const prev = p.legs[(i - 1 + p.legs.length) % p.legs.length];
              pos = leg.a;
              dir = { x: leg.a.x - prev.a.x, z: leg.a.z - prev.a.z };
              standing = true;
              break;
            }
            clock -= wait;
            const dur = leg.len / p.speed;
            if (clock < dur) {
              const k = clock / dur;
              pos = { x: leg.a.x + (leg.b.x - leg.a.x) * k,
                      z: leg.a.z + (leg.b.z - leg.a.z) * k };
              dir = { x: leg.b.x - leg.a.x, z: leg.b.z - leg.a.z };
              walked = clock * p.speed;
              break;
            }
            clock -= dur;
            walked += leg.len;
          }
          if (!pos) { pos = p.stops[0]; dir = { x: 1, z: 0 }; standing = true; }

          p.person.position.x = pos.x;
          p.person.position.z = pos.z;
          /* Mặt người quay về -Z (áo phản quang đặt ở z âm), nên góc quay là
             atan2(-dx, -dz). Bản trước tính như thể mặt quay về +Z, và thế là
             cả ba người đi giật lùi suốt ca.                                  */
          if (dir && (dir.x || dir.z))
            p.person.rotation.y = Math.atan2(-dir.x, -dir.z);

          const rig2 = p.person.userData.rig;
          if (rig2) {
            const swing = standing ? 0 : Math.sin(walked * 1.7) * 0.55;
            rig2.legs[0].rotation.x = swing;
            rig2.legs[1].rotation.x = -swing;
            rig2.arms[0].rotation.x = -swing * 0.7;
            rig2.arms[1].rotation.x = swing * 0.7;
          }
          p.person.position.y = standing ? 0
            : Math.abs(Math.cos(walked * 1.7)) * 0.035;
        } else if (rig) {
          // Đứng máy: dồn trọng tâm và với tay, biên độ nhỏ.
          const t2 = seconds * 0.6 + p.phase * 6;
          const sway = Math.sin(t2) * 0.12;
          rig.arms[0].rotation.x = -0.25 + Math.sin(t2 * 1.7) * 0.28;
          rig.arms[1].rotation.x = -0.15 + Math.sin(t2 * 1.7 + 1.1) * 0.2;
          rig.legs[0].rotation.x = sway * 0.25;
          rig.legs[1].rotation.x = -sway * 0.25;
          p.person.rotation.z = sway * 0.05;
        }
      });
      this.robotAnimators.forEach(robot => {
        if (robot.transferBottle) {
          // Một chu kỳ nhìn được ngay từ xa: chai đi trên băng 1, robot nhấc
          // sang băng 2, rồi chai tiếp tục chạy qua cụm camera OCR.
          const phase = (seconds * robot.speed + robot.phase) % 1;
          let x, y, z;
          if (phase < 0.28) {
            const p = phase / 0.28;
            x = -5.2 + 4.05 * p; y = 1.04; z = -2.05;
          } else if (phase < 0.72) {
            const p = (phase - 0.28) / 0.44;
            const ease = p * p * (3 - 2 * p);
            x = -1.15 + 1.35 * ease;
            y = 1.04 + Math.sin(Math.PI * ease) * 1.0;
            z = -2.05 + 2.05 * ease;
          } else {
            const p = (phase - 0.72) / 0.28;
            x = 0.2 + 4.75 * p; y = 1.04; z = 0;
          }
          robot.transferBottle.position.set(x, y, z);
          // Tay xoay về đúng vị trí chai ở mỗi nhịp, thay vì chỉ lắc sine độc
          // lập nên không bao giờ trông như đang gắp/đặt hàng.
          const aim = Math.atan2(-(z - robot.baseZ), x - robot.baseX);
          robot.shoulder.rotation.y = aim;
          const nearBelt = phase < 0.31 || phase > 0.68;
          robot.elbow.rotation.z = nearBelt ? 1.55 : 1.15;
          robot.wrist.rotation.z = nearBelt ? -0.40 : -0.25;
          return;
        }
        const cycle = seconds * robot.speed + robot.phase;
        robot.shoulder.rotation.y = -0.55 + Math.sin(cycle) * 0.75;
        robot.elbow.rotation.y = 0.85 + Math.sin(cycle + 1.1) * 0.5;
        robot.wrist.rotation.y = -0.3 - Math.sin(cycle + 1.8) * 0.45;
      });
      this.lastAnimatedAt = now;
      this.draw();
    }
    this.animationFrame = requestAnimationFrame(next => this.animate(next));
  }

  /** Bay camera tới một trạng thái mới. Mọi thứ điều khiển camera đều đi qua
   *  đây, nên không có đường nào làm cảnh nhảy giật. */
  flyTo({ target, distance, azimuth, elevation, duration = 620 }) {
    const T = this.THREE;
    this.cameraMove = {
      started: performance.now(),
      duration,
      fromTarget: this.cameraTarget.clone(),
      toTarget: target ? target.clone() : this.cameraTarget.clone(),
      fromDistance: this.distance,
      toDistance: distance ?? this.distance,
      fromAzimuth: this.azimuth,
      toAzimuth: azimuth ?? null,
      fromElevation: this.elevation,
      toElevation: elevation ?? null,
    };
    void T;
  }

  /* Ba góc nhìn cố định. "Top" để đọc bố trí mặt bằng, "Eye" để nhìn ngang tầm
     mắt như đứng trong xưởng, "3D" là góc mặc định ba-phần-tư. */
  setView(preset) {
    const T = this.THREE;
    // Ba preset đều lấy tâm và khoảng cách từ PLANT, không phải số cố định của
    // bố trí cũ — đổi mặt bằng thì ba góc nhìn tự theo.
    const home = new T.Vector3(PLANT.hall.x, 0, PLANT.hall.z);
    // "Top" là góc nhìn TOÀN NHÀ MÁY, nên phải gồm cả nhà Line 5 ở phía bắc —
    // căn theo sảnh chính thì nhà Line 5 bị cắt mất khỏi khung.
    if (preset === 'top') {
      const b = this.miniBounds();
      this.flyTo({
        target: new T.Vector3((b.x0 + b.x1) / 2, 0, (b.z0 + b.z1) / 2),
        distance: 150, azimuth: Math.PI / 2, elevation: 1.45,
      });
    }
    else if (preset === 'front') this.flyTo({ target: new T.Vector3(PLANT.hall.x, 1.6, PLANT.hall.z), distance: 52, azimuth: Math.PI / 2, elevation: 0.28 });
    else this.flyTo({ target: home, distance: 108, azimuth: 0.72, elevation: 0.72 });
    this.viewNav?.querySelectorAll('button[data-view]').forEach(b =>
      b.setAttribute('aria-pressed', String(b.dataset.view === (preset || 'iso'))));
  }

  zoomBy(amount) {
    this.distance = Math.max(14, Math.min(150, this.distance + amount));
    this.cameraMove = null;
  }

  /** Điểm trên mặt sàn nằm dưới con trỏ — để zoom đi VỀ PHÍA chỗ đang nhìn. */
  groundAt(event) {
    const T = this.THREE;
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = new T.Vector3();
    return this.raycaster.ray.intersectPlane(
      new T.Plane(new T.Vector3(0, 1, 0), 0), hit) ? hit : null;
  }

  /** Dời tâm nhìn theo mặt sàn. Kéo chuột phải hoặc giữ Shift.
   *  Không có pan thì camera chỉ quay quanh đúng một điểm, và muốn xem kỹ máy ở
   *  góc xưởng là phải zoom ra rồi zoom vào — thao tác đó lặp cả ngày. */
  panBy(dx, dy) {
    const k = this.distance * 0.0016;
    const cos = Math.cos(this.azimuth), sin = Math.sin(this.azimuth);
    this.cameraTarget.x += (-dx * -sin - dy * cos) * k;
    this.cameraTarget.z += (-dx * cos - dy * -sin) * k;
    // Giới hạn quanh xưởng: pan tự do thì rất dễ đẩy cảnh ra ngoài rồi không
    // biết đường về.
    this.cameraTarget.x = Math.max(-46, Math.min(46, this.cameraTarget.x));
    this.cameraTarget.z = Math.max(-30, Math.min(30, this.cameraTarget.z));
    this.cameraMove = null;
  }

  /** Bay tới một máy cụ thể — dùng khi bấm đúp trên sơ đồ. */
  focusMachine(nodeId) {
    const g = this.machineRoot.children.find(c => c.userData.nodeId === nodeId);
    if (!g) return;
    this.flyTo({ target: g.position.clone(), distance: 22, elevation: 0.55 });
  }

  hidePeek() {
    if (this.peek) { this.peek.hidden = true; this.peekFor = null; }
  }

  /** Rê lên một máy → thẻ ảnh nhỏ bám theo con trỏ. */
  async showPeek(hit, event) {
    /* Ở Line Station KHÔNG có thẻ peek. Hai lý do: nó gọi /api/fleet/frame của
       fleet service (không tồn tại trên máy này, nên hiện "undefined"), và ảnh
       của máy mình đã nằm ngay tab Sản xuất. Còn ảnh của máy khác thì cố ý
       không hiện ở bề mặt này. */
    if (ENABLED) { this.hidePeek(); return; }
    const name = hit?.machineName;
    if (!name) { this.hidePeek(); return; }

    // Đặt thẻ trong khung sơ đồ, kẹp lại để không tràn ra ngoài mép.
    const box = this.el.getBoundingClientRect();
    const x = Math.min(Math.max(event.clientX - box.left + 16, 8), box.width - 236);
    const y = Math.min(Math.max(event.clientY - box.top + 16, 8), box.height - 190);
    this.peek.style.left = `${x}px`;
    this.peek.style.top = `${y}px`;
    this.peek.hidden = false;

    if (this.peekFor === name) return;      // vẫn cùng một máy, chỉ dời chỗ
    this.peekFor = name;

    const t = store.t;
    const cached = this.peekCache.get(name);
    this.peek.innerHTML = `<b>${name}</b><div class="muted">${
      cached ? '' : t.loading}</div>`;

    let d = cached;
    if (!d) {
      try {
        const r = await fetch(`/api/fleet/frame/${encodeURIComponent(name)}`);
        d = await r.json();
        this.peekCache.set(name, d);
      } catch { d = { success: false }; }
      // Con trỏ có thể đã rời đi trong lúc chờ — đừng vẽ đè lên máy khác.
      if (this.peekFor !== name) return;
    }

    const f = d?.frame;
    if (!f) {
      this.peek.innerHTML = `<b>${name}</b><div class="muted">${t.noFrameYet}</div>`;
      return;
    }
    const said = (f.expected != null || f.recognized != null)
      ? `${t.expected} ${f.expected ?? '—'} → ${t.readAs} ${f.recognized || t.emptyRead}`
      : (f.recipe_name || '');
    this.peek.innerHTML = `<b>${name}</b>
      <img src="/api/fleet/failure-image/${encodeURIComponent(name)}/${encodeURIComponent(f.id)}?w=300" alt="">
      <div class="muted">${String(f.timestamp || '').slice(11, 19)} · ${said}</div>`;
  }

  focusZone(name) {
    const target = name === 'tin2'
      ? { point: new this.THREE.Vector3(PLANT.lines['Line 5'].x, 0, PLANT.lines['Line 5'].z), distance: 40 }
      : { point: new this.THREE.Vector3(PLANT.hall.x, 0, PLANT.hall.z), distance: 108 };
    this.cameraMove = {
      started: performance.now(), duration: 700,
      fromTarget: this.cameraTarget.clone(), toTarget: target.point,
      fromDistance: this.distance, toDistance: target.distance,
    };
    this.zoneNav.querySelectorAll('button').forEach(button =>
      button.setAttribute('aria-pressed', String(button.dataset.focus === name)));
  }

  /** Kho: giá kệ hai tầng có pallet, không phải một khối hộp.
   *
   *  Khối hộp trơn đọc ra là "có cái gì đó ở đây" chứ không phải "đây là kho".
   *  Giá kệ, pallet xếp không đều và lối giữa hai dãy thì nói đúng công năng —
   *  và cho thấy kho nào đang đầy, kho nào còn chỗ. */
  addWarehouse(label, x, z, { bays = 3, fill = 0.75 } = {}) {
    const THREE = this.THREE;
    const dark = store.theme === 'dark';
    const group = new THREE.Group();
    const steel = new THREE.MeshStandardMaterial({ color: dark ? 0x8a6f52 : 0xb4823f, roughness: 0.55, metalness: 0.3 });
    const deck = new THREE.MeshStandardMaterial({ color: dark ? 0x4c5863 : 0xc8ced4, roughness: 0.9 });
    const wood = new THREE.MeshStandardMaterial({ color: 0x8c633a, roughness: 0.92 });
    const carton = new THREE.MeshStandardMaterial({ color: dark ? 0xa98a5f : 0xd0a86a, roughness: 0.94 });
    const add = (geometry, material, px, py, pz, parent = group) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(px, py, pz);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      parent.add(mesh);
      return mesh;
    };
    const width = bays * 2.3;
    // Sàn kho + vạch kẻ ô.
    add(new THREE.BoxGeometry(width + 1.6, 0.05, 7.4), new THREE.MeshStandardMaterial({
      color: dark ? 0x2a3945 : 0xdde3e8, roughness: 0.95,
    }), 0, 0.03, 0);
    // Hai dãy giá quay lưng vào nhau, lối giữa để xe nâng vào.
    for (const row of [-2.4, 2.4]) {
      for (let b = 0; b <= bays; b++) {
        const px = -width / 2 + b * 2.3;
        add(new THREE.BoxGeometry(0.16, 3.4, 0.16), steel, px, 1.7, row - 0.55);
        add(new THREE.BoxGeometry(0.16, 3.4, 0.16), steel, px, 1.7, row + 0.55);
      }
      for (const level of [1.0, 2.2, 3.35]) {
        add(new THREE.BoxGeometry(width, 0.1, 0.12), steel, 0, level, row - 0.55);
        add(new THREE.BoxGeometry(width, 0.1, 0.12), steel, 0, level, row + 0.55);
        if (level < 3.3) add(new THREE.BoxGeometry(width, 0.06, 1.2), deck, 0, level + 0.06, row);
      }
      // Pallet hàng: thưa dần lên cao, và có ô trống — kho đầy đều tăm tắp là
      // kho trong phối cảnh quảng cáo, không phải kho thật.
      const levels = [1.06, 2.26];
      for (let b = 0; b < bays; b++) {
        levels.forEach((level, li) => {
          // Băm tất định theo vị trí kho + ô kệ: mỗi kho một kiểu chất hàng,
          // nhưng vẽ lại vẫn ra y như cũ (Math.random sẽ nhảy mỗi khung hình).
          const slot = Math.abs(Math.round(x * 3 + z * 5) + b * 7 + li * 3) % 10;
          if (slot / 10 > fill) return;
          const px = -width / 2 + b * 2.3 + 1.15;
          add(new THREE.BoxGeometry(1.6, 0.12, 1.05), wood, px, level + 0.06, row);
          const h = 0.55 + ((b + li) % 3) * 0.16;
          add(new THREE.BoxGeometry(1.42, h, 0.94), carton, px, level + 0.12 + h / 2, row);
        });
      }
    }
    const labelSprite = textSprite(THREE, label, '', dark);
    labelSprite.scale.set(3.7, 0.98, 1);
    labelSprite.position.set(0, 4.6, 0);
    group.add(labelSprite);
    group.position.set(x, 0, z);
    this.machineRoot.add(group);
  }

  addTin2Annex(x, z) {
    const THREE = this.THREE;
    const dark = store.theme === 'dark';
    const building = new THREE.Group();
    building.userData.focus = 'tin2';
    const wall = new THREE.MeshStandardMaterial({
      color: dark ? 0x4a5762 : 0xd4d9dc, roughness: 0.9, transparent: true, opacity: 0.72,
    });
    const frame = new THREE.MeshStandardMaterial({ color: dark ? 0x758694 : 0x89959d, roughness: 0.5, metalness: 0.35 });
    const add = (geometry, material, px, py, pz) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(px, py, pz);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      building.add(mesh);
      return mesh;
    };
    add(new THREE.BoxGeometry(23, 0.18, 16), new THREE.MeshStandardMaterial({ color: dark ? 0x26333d : 0xe0e5e8, roughness: 0.96 }), 0, 0.02, 0);
    // Bao ngoài là nhà riêng; mặt phía đường có cửa cuốn để vẫn thấy không gian bên trong.
    add(new THREE.BoxGeometry(0.22, 3.6, 16), wall, -11.4, 1.8, 0);
    add(new THREE.BoxGeometry(0.22, 3.6, 16), wall, 11.4, 1.8, 0);
    add(new THREE.BoxGeometry(23, 3.6, 0.22), wall, 0, 1.8, 7.9);
    add(new THREE.BoxGeometry(7.2, 3.6, 0.22), wall, 7.8, 1.8, -7.9);
    // Cửa dock mở về phía đường nối xưởng chính.
    for (const gx of [-4.2, 4.2]) add(new THREE.BoxGeometry(0.25, 3.4, 0.25), frame, gx, 1.7, -7.9);
    add(new THREE.BoxGeometry(8.6, 0.24, 0.3), frame, 0, 3.35, -7.9);
    for (const cx of [-11.4, 11.4]) for (const cz of [-7.9, 7.9]) add(new THREE.BoxGeometry(0.42, 3.9, 0.42), frame, cx, 1.95, cz);
    const dock = add(new THREE.BoxGeometry(3.2, 0.26, 4.2), new THREE.MeshStandardMaterial({ color: dark ? 0x303b44 : 0xb9c2c7, roughness: 0.9 }), -13.1, 0.2, -3.8);
    dock.receiveShadow = true;
    const sign = textSprite(THREE, 'LINE 5', 'ANNEX BUILDING', dark);
    sign.position.set(0, 5.0, -7.65);
    sign.scale.set(5.1, 1.34, 1);
    building.add(sign);
    building.position.set(x, 0, z);
    // Nhà Line 5 nằm phía BẮC xưởng chính, nên mặt có cửa dock phải quay xuống
    // nam. Hình dựng với dock ở −z, xoay 180° là dock nhìn ra đường nối về
    // xưởng — nếu không thì xe nâng phải đi vòng ra sau nhà.
    building.rotation.y = Math.PI;
    this.machineRoot.add(building);
    this.pickables.push(building);
  }

  addPerson(target, { x, z, role = 'worker', shirt = 0x2d6c9d, facing = 0 }) {
    const THREE = this.THREE;
    const person = new THREE.Group();
    const skin = new THREE.MeshStandardMaterial({ color: 0xd6a07c, roughness: 0.82 });
    const cloth = new THREE.MeshStandardMaterial({ color: shirt, roughness: 0.78 });
    const trousers = new THREE.MeshStandardMaterial({ color: role === 'supervisor' ? 0x30363e : 0x334b5b, roughness: 0.86 });
    const helmet = new THREE.MeshStandardMaterial({ color: role === 'supervisor' ? 0xf4f4ed : 0xf0c748, roughness: 0.58 });
    const vest = new THREE.MeshStandardMaterial({ color: role === 'supervisor' ? 0xe55d3f : 0xf0b72f, roughness: 0.55 });
    const boot = new THREE.MeshStandardMaterial({ color: 0x1e252b, roughness: 0.9 });
    const add = (geometry, material, px, py, pz) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(px, py, pz);
      mesh.castShadow = true;
      person.add(mesh);
      return mesh;
    };
    const limb = (from, to, radius, material, parent = person) => {
      const delta = new THREE.Vector3().subVectors(to, from);
      const part = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 1.08, delta.length(), 10), material);
      part.position.copy(from).add(to).multiplyScalar(0.5);
      part.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
      part.castShadow = true;
      parent.add(part);
      return part;
    };
    /* Khớp xoay ở hông và vai. Bản trước dựng tay chân bằng toạ độ TUYỆT ĐỐI
       trong thân người, nên không có gì để xoay — muốn bước đi thì chỉ còn cách
       trượt cả khối, và trượt mà chân đứng yên thì nhìn như đi patin. Bọc mỗi
       chi vào một Group đặt đúng khớp là xoay được quanh đúng trục. */
    const joint = (px, py, pz) => {
      const g = new THREE.Group();
      g.position.set(px, py, pz);
      person.add(g);
      return g;
    };
    // Tỉ lệ cơ thể, khớp gối/khuỷu và áo phản quang giúp nhân vật đọc được ở
    // góc isometric, thay vì chỉ là cylinder có hai chân.
    add(new THREE.CylinderGeometry(0.2, 0.27, 0.76, 12), cloth, 0, 1.25, 0);
    add(new THREE.BoxGeometry(0.42, 0.52, 0.06), vest, 0, 1.3, -0.235);
    add(new THREE.BoxGeometry(0.44, 0.08, 0.08), new THREE.MeshBasicMaterial({ color: 0xece3b9 }), 0, 1.43, -0.275);
    add(new THREE.SphereGeometry(0.17, 14, 10), skin, 0, 1.78, 0);
    add(new THREE.CylinderGeometry(0.07, 0.08, 0.12), skin, 0, 1.6, 0);
    const hardhat = add(new THREE.SphereGeometry(0.2, 14, 8, 0, Math.PI * 2, 0, Math.PI / 2), helmet, 0, 1.91, 0);
    hardhat.rotation.y = Math.PI / 4;
    const rig = { legs: [], arms: [] };
    for (const side of [-1, 1]) {
      const legPivot = joint(side * 0.1, 0.88, 0);
      rig.legs.push(legPivot);
      const knee = new THREE.Vector3(side * 0.02, -0.41, side * 0.04);
      const ankle = new THREE.Vector3(0, -0.72, -0.03);
      limb(new THREE.Vector3(0, 0, 0), knee, 0.075, trousers, legPivot);
      limb(knee, ankle, 0.065, trousers, legPivot);
      const shoe = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.09, 0.26), boot);
      shoe.position.set(0, -0.77, -0.1);
      shoe.castShadow = true;
      legPivot.add(shoe);

      const armPivot = joint(side * 0.22, 1.52, 0);
      rig.arms.push(armPivot);
      const elbow = new THREE.Vector3(side * 0.11, -0.32, side * 0.08);
      const handY = (role === 'supervisor' ? 1.08 : 1.05) - 1.52;
      const hand = new THREE.Vector3(side * 0.06, handY, -0.18);
      limb(new THREE.Vector3(0, 0, 0), elbow, 0.06, cloth, armPivot);
      limb(elbow, hand, 0.052, cloth, armPivot);
      const fist = new THREE.Mesh(new THREE.SphereGeometry(0.065, 10, 8), skin);
      fist.position.copy(hand);
      armPivot.add(fist);
    }
    person.userData.rig = rig;
    if (role === 'supervisor') {
      const clipboard = add(new THREE.BoxGeometry(0.26, 0.36, 0.04), new THREE.MeshStandardMaterial({ color: 0xd96c42, roughness: 0.6 }), 0.25, 1.2, -0.23);
      clipboard.rotation.z = -0.2;
    }
    person.position.set(x, 0, z);
    person.rotation.y = facing;
    target.add(person);
    return person;
  }

  addZone(machine, x, z, index, open = 'south') {
    const THREE = this.THREE;
    const dark = store.theme === 'dark';
    const isBottleLine = machine.name === 'Line 6';
    const product = LINE_PRODUCTS[machine.name] || { pack: '#d7b34d', tape: '#9e2f2f', mark: '#7b5322' };
    // Dựng theo hướng chuẩn — vách kín ở -z, cửa mở ra +z — rồi xoay cả nhóm
    // theo `open`. Nhờ vậy pallet, vạch sàn và người đứng máy đi theo luôn.
    const awayFromAisle = -1;
    const towardAisle = 1;
    const zone = new THREE.Group();
    const wall = new THREE.MeshStandardMaterial({
      color: dark ? 0x48545e : 0xd8dde0, roughness: 0.92, transparent: true, opacity: 0.84,
    });
    const trim = new THREE.MeshStandardMaterial({ color: dark ? 0x71818f : 0xa7b2ba, roughness: 0.6, metalness: 0.25 });
    const zoneFloor = new THREE.Mesh(
      new THREE.BoxGeometry(13.8, 0.06, 7.8),
      new THREE.MeshStandardMaterial({ color: product.pack, roughness: 0.96, transparent: true, opacity: dark ? 0.16 : 0.09 }),
    );
    zoneFloor.position.y = 0.05;
    zone.add(zoneFloor);
    const add = (geometry, material, px, py, pz) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(px, py, pz);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      zone.add(mesh);
      return mesh;
    };
    // Ba vách, mặt hướng aisle để hở làm lối vào. Vách bán trong để thấy line.
    add(new THREE.BoxGeometry(0.18, 2.5, 7.7), wall, -6.8, 1.25, 0);
    add(new THREE.BoxGeometry(0.18, 2.5, 7.7), wall, 6.8, 1.25, 0);
    add(new THREE.BoxGeometry(13.8, 2.5, 0.18), wall, 0, 1.25, awayFromAisle * 3.8);
    for (const sideX of [-6.8, 6.8]) {
      add(new THREE.BoxGeometry(0.28, 0.16, 7.8), trim, sideX, 2.5, 0);
      add(new THREE.BoxGeometry(0.28, 0.16, 7.8), trim, sideX, 0.13, 0);
    }
    add(new THREE.BoxGeometry(13.8, 0.16, 0.28), trim, 0, 2.5, awayFromAisle * 3.8);
    // Hai trụ cổng và vạch sàn báo lối đi từ aisle vào khu vực.
    for (const gateX of [-3.0, 3.0]) add(new THREE.BoxGeometry(0.2, 2.1, 0.2), trim, gateX, 1.05, towardAisle * 3.72);
    add(new THREE.BoxGeometry(5.8, 0.04, 0.16), trim, 0, 0.08, towardAisle * 3.72);

    const pallet = (px, pz, levels) => {
      add(new THREE.BoxGeometry(1.32, 0.12, 0.96), new THREE.MeshStandardMaterial({ color: 0x8c633a, roughness: 0.9 }), px, 0.17, pz);
      if (isBottleLine) {
        const crate = add(new THREE.BoxGeometry(1.04, 0.34, 0.78), new THREE.MeshStandardMaterial({ color: 0x9a7354, roughness: 0.85 }), px, 0.4, pz);
        for (const bx of [-0.25, 0, 0.25]) for (const bz of [-0.18, 0.18]) {
          add(new THREE.CylinderGeometry(0.07, 0.08, 0.38, 10), new THREE.MeshStandardMaterial({ color: 0xd1a63e, roughness: 0.35, transparent: true, opacity: 0.9 }), px + bx, 0.72, pz + bz);
        }
        return crate;
      }
      for (let level = 0; level < levels; level++) {
        const pack = add(new THREE.BoxGeometry(0.8, 0.5, 0.7), new THREE.MeshStandardMaterial({ color: product.pack, roughness: 0.9 }), px + (level % 2 ? 0.04 : -0.04), 0.48 + level * 0.5, pz);
        const band = add(new THREE.BoxGeometry(0.84, 0.05, 0.09), new THREE.MeshBasicMaterial({ color: product.tape }), pack.position.x, pack.position.y + 0.12, pz);
        band.material.transparent = true;
        band.material.opacity = 0.92;
      }
    };
    pallet(5.45, awayFromAisle * 2.65, 3);
    pallet(-5.3, awayFromAisle * 2.65, 2);

    const workerColors = [0x2d6c9d, 0xc85c32, 0x5d8c55, 0x8962a8, 0x237878];
    const operator = this.addPerson(zone, { x: -4.85, z: towardAisle * 2.25, shirt: workerColors[index % workerColors.length], facing: towardAisle > 0 ? 0 : Math.PI });
    // Người đứng máy không đi đâu cả — họ thao tác tại chỗ. Nhưng máy DỪNG thì
    // họ cũng đứng yên hẳn, cùng một luật với băng tải: cảnh không được nói
    // rằng có việc đang diễn ra ở một line đã tắt.
    if (!['unreachable', 'offline'].includes(machine.state))
      this.peopleAnimators.push({ kind: 'idle', person: operator, phase: index * 0.7 });
    zone.position.set(x, 0, z);
    zone.rotation.y = OPEN_YAW[open] ?? 0;
    this.machineRoot.add(zone);
  }

  /** Xe nâng: khung, cabin, cột nâng, càng, và một pallet hàng khi đang chở.
   *  Trả về nhóm để `update()` gắn vào tuyến chạy. */
  makeForklift(loaded) {
    const THREE = this.THREE;
    const dark = store.theme === 'dark';
    const truck = new THREE.Group();
    const body = new THREE.MeshStandardMaterial({ color: 0xd8a12a, roughness: 0.55, metalness: 0.2 });
    const dim = new THREE.MeshStandardMaterial({ color: dark ? 0x2b343c : 0x39424a, roughness: 0.8 });
    const steel = new THREE.MeshStandardMaterial({ color: dark ? 0x7d8b96 : 0x98a4ad, roughness: 0.45, metalness: 0.5 });
    const add = (geometry, material, px, py, pz) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(px, py, pz);
      mesh.castShadow = true;
      truck.add(mesh);
      return mesh;
    };
    add(new THREE.BoxGeometry(1.05, 0.5, 1.5), body, 0, 0.5, 0.1);
    add(new THREE.BoxGeometry(0.95, 0.42, 0.62), dim, 0, 0.95, 0.42);        // ghế / khoang lái
    // Khung bảo vệ trên đầu — thứ nhìn ra ngay là xe nâng chứ không phải xe đẩy.
    for (const cx of [-0.44, 0.44]) for (const cz of [-0.2, 0.62])
      add(new THREE.BoxGeometry(0.07, 1.15, 0.07), steel, cx, 1.35, cz);
    add(new THREE.BoxGeometry(1.05, 0.08, 0.95), steel, 0, 1.95, 0.21);
    for (const wx of [-0.56, 0.56]) {
      add(new THREE.CylinderGeometry(0.26, 0.26, 0.18, 14), dim, wx, 0.26, 0.55).rotation.z = Math.PI / 2;
      add(new THREE.CylinderGeometry(0.2, 0.2, 0.16, 12), dim, wx, 0.2, -0.5).rotation.z = Math.PI / 2;
    }
    for (const mx of [-0.34, 0.34]) add(new THREE.BoxGeometry(0.1, 1.9, 0.1), steel, mx, 0.95, -0.78);
    for (const fx of [-0.3, 0.3]) add(new THREE.BoxGeometry(0.13, 0.07, 1.15), steel, fx, 0.14, -1.35);
    if (loaded) {
      add(new THREE.BoxGeometry(1.25, 0.12, 1.0), new THREE.MeshStandardMaterial({ color: 0x8c633a, roughness: 0.92 }), 0, 0.24, -1.35);
      add(new THREE.BoxGeometry(1.1, 0.72, 0.9), new THREE.MeshStandardMaterial({ color: dark ? 0xa98a5f : 0xd0a86a, roughness: 0.94 }), 0, 0.66, -1.35);
    }
    return truck;
  }

  /** Xe nâng chạy vòng trên hành lang. Dùng lại đúng cơ chế chặng/nghỉ của
   *  người đi bộ, nên nó dừng ở kho đủ lâu để đọc ra là đang lấy hàng. */
  addForklifts() {
    const routes = [
      { loaded: true, speed: 4.2, offset: 0, stops: [
        { x: 10, z: 4.5, pause: 4 },            // ăn hàng ở kho
        { x: 10, z: 7.5 },
        { x: 16.5, z: 7.5 },
        { x: 16.5, z: 19 },
        { x: 19.4, z: 19, pause: 3.5 },         // trả hàng cạnh Line 4
        { x: 16.5, z: 19 },
        { x: 16.5, z: 7.5 },
        { x: 10, z: 7.5 },
      ] },
      { loaded: false, speed: 4.8, offset: 11, stops: [
        { x: -21.5, z: -16 },
        { x: -21.5, z: 7.5 },
        { x: -3, z: 7.5 },
        { x: -3.5, z: -6 },
        { x: -3.5, z: -19, pause: 2.5 },
        { x: -3.5, z: -6 },
        { x: -3, z: 7.5 },
        { x: -21.5, z: 7.5 },
      ] },
    ];
    for (const r of routes) {
      const truck = this.makeForklift(r.loaded);
      truck.position.set(r.stops[0].x, 0, r.stops[0].z);
      this.machineRoot.add(truck);
      const legs = r.stops.map((a, i) => {
        const b = r.stops[(i + 1) % r.stops.length];
        return { a, b, len: Math.hypot(b.x - a.x, b.z - a.z) };
      });
      const cycle = legs.reduce((n, l) => n + l.len / r.speed, 0)
                  + r.stops.reduce((n, st) => n + (st.pause || 0), 0);
      this.peopleAnimators.push({ kind: 'walk', person: truck, ...r, legs, cycle });
    }
  }

  /** Đồ đạc còn lại của một xưởng đang hoạt động: cột nhà, pallet chờ, xe đẩy,
   *  cọc chắn dọc hành lang, điểm chữa cháy. Không cái nào mang dữ liệu — chúng
   *  ở đây để cảnh có tỉ lệ và có chiều sâu, thay vì mấy khối trôi trên nền phẳng. */
  addPlantProps() {
    const THREE = this.THREE;
    const dark = store.theme === 'dark';
    const props = new THREE.Group();
    const add = (geometry, material, px, py, pz, parent = props) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(px, py, pz);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      parent.add(mesh);
      return mesh;
    };
    const concrete = new THREE.MeshStandardMaterial({ color: dark ? 0x3d4852 : 0xccd3d9, roughness: 0.95 });
    const wood = new THREE.MeshStandardMaterial({ color: 0x8c633a, roughness: 0.92 });
    const carton = new THREE.MeshStandardMaterial({ color: dark ? 0xa98a5f : 0xd0a86a, roughness: 0.94 });
    const yellow = new THREE.MeshStandardMaterial({ color: 0xd8a12a, roughness: 0.6 });
    const steel = new THREE.MeshStandardMaterial({ color: dark ? 0x7d8b96 : 0x98a4ad, roughness: 0.45, metalness: 0.5 });
    const red = new THREE.MeshStandardMaterial({ color: 0xb63a2e, roughness: 0.6 });

    /* Cột nhà. Bản trước rải lưới 5×5 khắp sàn, cột cao 6,4 và dày 0,7 — hoá ra
       cột là thứ CAO NHẤT và NHIỀU NHẤT trong cảnh, che mất chính mấy dây chuyền
       mà sơ đồ sinh ra để hiển thị. Giờ chỉ chạy quanh viền nhà, thêm ba cột
       trong ở đúng chỗ trống, mảnh hơn và thấp hơn. */
    const columns = [];
    for (const cz of [-30, -14, 2, 18, 27]) { columns.push([-39, cz], [35, cz]); }
    for (const cx of [-24, -8, 8, 24]) columns.push([cx, -33]);
    for (const cx of [-24, -8, 8]) columns.push([cx, 27]);
    columns.push([-1.5, -25], [0, -2], [0, 18]);
    for (const [cx, cz] of columns) {
      add(new THREE.BoxGeometry(0.38, 3.8, 0.38), concrete, cx, 1.9, cz);
      add(new THREE.BoxGeometry(0.54, 0.36, 0.54), yellow, cx, 0.18, cz);   // chân cột sơn vạch
    }

    // Pallet chờ xếp, chồng cao thấp khác nhau.
    const stacks = [[-17.5, 12, 3], [-15.6, 12.9, 2], [8.5, 10.5, 4], [12.4, 12.2, 2],
                    [21, -6, 3], [23.5, -5, 1], [-25, -21, 2], [1.5, 20, 3]];
    for (const [px, pz, n] of stacks) {
      for (let i = 0; i < n; i++) add(new THREE.BoxGeometry(1.5, 0.14, 1.05), wood, px, 0.09 + i * 0.16, pz);
      if (n > 2) add(new THREE.BoxGeometry(1.32, 0.75, 0.92), carton, px, 0.16 * n + 0.42, pz);
    }

    // Xe đẩy lồng: đứng cạnh lối vào các khu.
    for (const [px, pz, rot] of [[-7.6, -9.5, 0], [-7.6, 1.5, 0], [19.8, 14.5, Math.PI / 2], [-26, -18, 0]]) {
      const cage = new THREE.Group();
      add(new THREE.BoxGeometry(1.1, 0.1, 0.78), steel, 0, 0.34, 0, cage);
      for (const gx of [-0.53, 0.53]) add(new THREE.BoxGeometry(0.05, 1.25, 0.78), steel, gx, 0.95, 0, cage);
      add(new THREE.BoxGeometry(1.1, 1.25, 0.05), steel, 0, 0.95, -0.37, cage);
      add(new THREE.BoxGeometry(0.95, 0.55, 0.66), carton, 0, 0.67, 0.02, cage);
      for (const wx of [-0.42, 0.42]) for (const wz of [-0.28, 0.28])
        add(new THREE.CylinderGeometry(0.12, 0.12, 0.07, 10), steel, wx, 0.12, wz, cage).rotation.z = Math.PI / 2;
      cage.position.set(px, 0, pz);
      cage.rotation.y = rot;
      props.add(cage);
    }

    // Cọc chắn vàng dọc hai trục hành lang chính — ranh giới người đi / xe chạy.
    for (let z = -18; z <= 20; z += 3.4) {
      add(new THREE.CylinderGeometry(0.11, 0.13, 0.85, 10), yellow, -19.6, 0.42, z);
      add(new THREE.CylinderGeometry(0.11, 0.13, 0.85, 10), yellow, -5.4, 0.42, Math.min(z, 8));
    }

    // Điểm chữa cháy: bảng đỏ + hai bình, đặt ở đầu mỗi khu.
    for (const [px, pz] of [[-21.9, -20.5], [-3.9, -20.5], [16.9, 3.5], [-21.9, 20.5]]) {
      add(new THREE.BoxGeometry(0.9, 1.1, 0.08), red, px, 1.5, pz);
      for (const dx of [-0.22, 0.22]) add(new THREE.CylinderGeometry(0.13, 0.13, 0.62, 10), red, px + dx, 0.31, pz + 0.12);
    }
    this.machineRoot.add(props);
  }

  /** Bao nhà chính: bờ tường thấp + dầm mái viền.
   *
   *  Không dựng tường cao — nhìn nghiêng thì tường che mất chính thứ cần xem.
   *  Nhưng thiếu hẳn bao ngoài thì các cột đứng chơ vơ giữa sàn trống và cảnh
   *  đọc ra là "mấy khu máy đặt ngoài trời", không phải một toà nhà. */
  addHallShell() {
    const THREE = this.THREE;
    const dark = store.theme === 'dark';
    const shell = new THREE.Group();
    const h = PLANT.hall;
    const kerb = new THREE.MeshStandardMaterial({ color: dark ? 0x38434d : 0xc6ced5, roughness: 0.92 });
    const beam = new THREE.MeshStandardMaterial({
      color: dark ? 0x5d6b77 : 0x9fabb4, roughness: 0.6, metalness: 0.3,
      transparent: true, opacity: 0.75,
    });
    const add = (geometry, material, px, py, pz) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(px, py, pz);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      shell.add(mesh);
    };
    const hw = h.w / 2, hd = h.d / 2;
    // Bờ tường cao 0,9 — đủ đọc ra ranh giới, chưa đủ che gì.
    for (const dz of [-hd, hd]) add(new THREE.BoxGeometry(h.w, 0.9, 0.4), kerb, h.x, 0.45, h.z + dz);
    for (const dx of [-hw, hw]) add(new THREE.BoxGeometry(0.4, 0.9, h.d), kerb, h.x + dx, 0.45, h.z);
    // Dầm mái viền ở cao độ cột, bán trong.
    for (const dz of [-hd, hd]) add(new THREE.BoxGeometry(h.w, 0.32, 0.32), beam, h.x, 3.9, h.z + dz);
    for (const dx of [-hw, hw]) add(new THREE.BoxGeometry(0.32, 0.32, h.d), beam, h.x + dx, 3.9, h.z);
    const sign = textSprite(THREE, h.label, '', dark);
    sign.scale.set(6.2, 1.15, 1);
    sign.position.set(h.x - hw + 7, 5.4, h.z - hd);
    shell.add(sign);
    this.machineRoot.add(shell);
  }

  /** Khu có tên: sàn nhuộm màu + viền + biển, tương ứng các ô đặt tên trong
   *  bản vẽ mặt bằng. Chỉ là mặt sàn và viền thấp, không dựng vách — vách là
   *  việc của từng khu máy, còn đây là ranh giới hành chính của khu vực. */
  addArea({ label, x, z, w, d, tint }) {
    const THREE = this.THREE;
    const dark = store.theme === 'dark';
    const group = new THREE.Group();
    const pad = new THREE.Mesh(
      new THREE.BoxGeometry(w, 0.04, d),
      new THREE.MeshStandardMaterial({
        color: tint, roughness: 0.98, transparent: true, opacity: dark ? 0.1 : 0.07,
      }),
    );
    pad.position.y = 0.03;
    pad.receiveShadow = true;
    group.add(pad);
    const edge = new THREE.MeshBasicMaterial({ color: tint, transparent: true, opacity: dark ? 0.5 : 0.42 });
    for (const dz of [-d / 2, d / 2]) {
      const bar = new THREE.Mesh(new THREE.BoxGeometry(w, 0.05, 0.16), edge);
      bar.position.set(0, 0.06, dz);
      group.add(bar);
    }
    for (const dx of [-w / 2, w / 2]) {
      const bar = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.05, d), edge);
      bar.position.set(dx, 0.06, 0);
      group.add(bar);
    }
    const sign = textSprite(THREE, label, '', dark);
    sign.scale.set(Math.min(7.2, label.length * 0.42), 1.05, 1);
    sign.position.set(0, 0.9, -d / 2 + 0.9);
    group.add(sign);
    group.position.set(x, 0, z);
    this.machineRoot.add(group);
  }

  /** Phòng vận hành: hộp kính nhìn ra xưởng. Đây là chỗ người ta thật sự ngồi
   *  đọc màn hình, nên nó có mặt trên sơ đồ chứ không phải chỉ có máy móc. */
  addOpsRoom(x, z) {
    const THREE = this.THREE;
    const dark = store.theme === 'dark';
    const room = new THREE.Group();
    const glass = new THREE.MeshStandardMaterial({
      color: dark ? 0x5a7385 : 0xbcd2df, roughness: 0.22, metalness: 0.15,
      transparent: true, opacity: 0.44,
    });
    const frame = new THREE.MeshStandardMaterial({ color: dark ? 0x76858f : 0x8e9aa2, roughness: 0.5, metalness: 0.3 });
    const add = (geometry, material, px, py, pz) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(px, py, pz);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      room.add(mesh);
    };
    add(new THREE.BoxGeometry(11, 0.16, 7), new THREE.MeshStandardMaterial({ color: dark ? 0x2b3a45 : 0xdfe6ea, roughness: 0.95 }), 0, 0.05, 0);
    add(new THREE.BoxGeometry(11, 2.9, 0.14), glass, 0, 1.5, 3.4);
    add(new THREE.BoxGeometry(11, 2.9, 0.14), glass, 0, 1.5, -3.4);
    add(new THREE.BoxGeometry(0.14, 2.9, 7), glass, -5.4, 1.5, 0);
    add(new THREE.BoxGeometry(0.14, 2.9, 7), glass, 5.4, 1.5, 0);
    // Mái là KHUNG viền, không phải tấm đặc: tấm đặc thì không nhìn thấy người
    // ngồi bên trong, mà chính họ là lý do căn phòng có mặt trên sơ đồ.
    for (const bz of [-3.4, 3.4]) add(new THREE.BoxGeometry(11.3, 0.18, 0.3), frame, 0, 3.0, bz);
    for (const bx of [-5.4, 5.4]) add(new THREE.BoxGeometry(0.3, 0.18, 7.3), frame, bx, 3.0, 0);
    for (const cx of [-5.4, 5.4]) for (const cz of [-3.4, 3.4])
      add(new THREE.BoxGeometry(0.3, 3.0, 0.3), frame, cx, 1.5, cz);
    // Hai màn hình treo — đây là nơi Fleet Console được nhìn.
    for (const mx of [-2.2, 2.2])
      add(new THREE.BoxGeometry(2.2, 1.25, 0.1), new THREE.MeshStandardMaterial({
        color: dark ? 0x16212a : 0x2a3742, roughness: 0.35, emissive: 0x1d4e6b, emissiveIntensity: dark ? 0.55 : 0.3,
      }), mx, 1.85, -3.1);
    this.addPerson(room, { x: -1.6, z: -1.2, role: 'supervisor', shirt: 0x424d66, facing: Math.PI });
    this.addPerson(room, { x: 1.9, z: -0.6, role: 'supervisor', shirt: 0x6b536d, facing: Math.PI });
    const sign = textSprite(THREE, 'OPERATIONS ROOM', '', dark);
    sign.scale.set(5.4, 1.1, 1);
    sign.position.set(0, 4.0, 0);
    room.add(sign);
    room.position.set(x, 0, z);
    this.machineRoot.add(room);
  }

  /** Cổng vào: mọi thứ đi vào xưởng đều qua đây, nên nó là gốc của các tuyến
   *  đi lại vẽ trên sàn. */
  addEntrance(x, z) {
    const THREE = this.THREE;
    const dark = store.theme === 'dark';
    const gate = new THREE.Group();
    const frame = new THREE.MeshStandardMaterial({ color: dark ? 0x76858f : 0x8e9aa2, roughness: 0.5, metalness: 0.3 });
    const add = (geometry, material, px, py, pz) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(px, py, pz);
      mesh.castShadow = true;
      gate.add(mesh);
    };
    for (const px of [-3.4, 3.4]) add(new THREE.BoxGeometry(0.5, 4.2, 0.5), frame, px, 2.1, 0);
    add(new THREE.BoxGeometry(7.8, 0.4, 0.5), frame, 0, 4.2, 0);
    add(new THREE.BoxGeometry(6.8, 0.05, 3.4), new THREE.MeshBasicMaterial({
      color: dark ? 0xa5b4c2 : 0x9aa9b7, transparent: true, opacity: 0.4,
    }), 0, 0.04, 1.9);
    const sign = textSprite(THREE, 'ENTRANCE', '', dark);
    sign.scale.set(3.6, 1.0, 1);
    sign.position.set(0, 5.2, 0);
    gate.add(sign);
    gate.position.set(x, 0, z);
    this.machineRoot.add(gate);
  }

  addAisleStaff() {
    const staff = new this.THREE.Group();

    /* Người đi TUẦN CÁC LINE, không phải đi tới đi lui trên lối đi. Đi dọc mãi
       một đường thẳng thì nhìn ra ngay là hoạt ảnh; còn rẽ vào một line, đứng
       lại một lúc bên máy rồi ra tiếp sang line khác thì đúng là việc người ta
       làm trong ca.

       Toạ độ bám mặt bằng thật (PLANT ở đầu file). Mỗi điểm dừng nằm ngay
       NGOÀI cửa của khu tương ứng — cửa cách tâm khu 3,9 đơn vị theo hướng
       `open` — nên người dừng lại là đứng cạnh máy, không đứng xuyên tường.
       Các đoạn di chuyển đều bám hành lang trong CORRIDORS, không cắt chéo qua
       khu sản xuất.                                                           */
    const walkers = [
      { // tổ trưởng: trục dọc giữa, tuần Line 1 rồi Line 2
        shirt: 0x424d66, role: 'supervisor', speed: 2.6, offset: 0,
        stops: [
          { x: -6.5, z: -14, pause: 5 },
          { x: -3.5, z: -14 },
          { x: -3.5, z: -3 },
          { x: -6.5, z: -3, pause: 4.5 },
          { x: -3.5, z: -3 },
          { x: -3.5, z: -14 },
        ],
      },
      { // QA: trục tây rồi vòng sang trục ngang, tuần Line 6 và Line 7
        shirt: 0x6b536d, role: 'supervisor', speed: 2.2, offset: 9,
        stops: [
          { x: -24, z: -14, pause: 4 },
          { x: -21.5, z: -14 },
          { x: -21.5, z: 7.5 },
          { x: -30, z: 7.5 },
          { x: -30, z: 9.5, pause: 5.5 },
          { x: -30, z: 7.5 },
          { x: -21.5, z: 7.5 },
          { x: -21.5, z: -14 },
        ],
      },
      { // công nhân: trục đông, tuần Line 3 / Line 4 rồi ra kho
        shirt: 0x2d6c9d, role: 'worker', speed: 3.0, offset: 4,
        stops: [
          { x: 19, z: 10, pause: 3.5 },
          { x: 16.5, z: 10 },
          { x: 16.5, z: 21 },
          { x: 19, z: 21, pause: 3 },
          { x: 16.5, z: 21 },
          { x: 16.5, z: 7.5 },
          { x: 10, z: 7.5 },
          { x: 10, z: 4.5, pause: 2.5 },
          { x: 10, z: 7.5 },
          { x: 16.5, z: 7.5 },
          { x: 16.5, z: 10 },
        ],
      },
    ];

    for (const w of walkers) {
      const person = this.addPerson(staff, {
        x: w.stops[0].x, z: w.stops[0].z, role: w.role, shirt: w.shirt, facing: 0,
      });
      // Dựng sẵn độ dài từng chặng: mỗi khung hình mà tính lại là tính thừa.
      const legs = w.stops.map((a, i) => {
        const b = w.stops[(i + 1) % w.stops.length];
        return { a, b, len: Math.hypot(b.x - a.x, b.z - a.z) };
      });
      const cycle = legs.reduce((n, l) => n + l.len / w.speed, 0)
                  + w.stops.reduce((n, st) => n + (st.pause || 0), 0);
      this.peopleAnimators.push({ kind: 'walk', person, ...w, legs, cycle });
    }
    this.machineRoot.add(staff);
  }


  addMachine(machine, x, z, selected, open = 'south') {
    const THREE = this.THREE;
    const live = isEnabled(machine.name);
    // Máy bị tắt: bảng xám, và `beacon: null` để khối đèn không được dựng.
    const style = live ? (STATUS[machine.state] || STATUS.unreachable) : DISABLED;
    const dark = store.theme === 'dark';
    const isBottleLine = machine.name === 'Line 6';
    const isStopped = ['unreachable', 'offline'].includes(machine.state);
    /* Máy dừng thì KHÔNG có gì chuyển động — không có ngoại lệ cho line chai.
       Trước đây `|| isBottleLine` cho Line 6 chạy băng tải kể cả khi đã mất
       Tailnet, để minh hoạ luồng băng 1 → robot → băng 2. Nhưng đây là màn hình
       giám sát, không phải video giới thiệu: một dây chuyền đang chạy trên sơ
       đồ trong khi thẻ bên cạnh ghi "Off network" là hai câu trái ngược nhau về
       cùng một máy, và cái chuyển động bao giờ cũng thắng.

       Hàng hoá vẫn nằm nguyên trên băng — dây chuyền dừng thì sản phẩm còn đó,
       chỉ là không đi. */
    const animateFlow = !isStopped;
    const productProfile = LINE_PRODUCTS[machine.name] || LINE_PRODUCTS['Line 1'];
    const group = new THREE.Group();
    if (machine.virtual) group.userData.focus = machine.building || 'main';
    // Máy bị tắt KHÔNG mang nodeId, nên raycast không chọn được nó; thay vào đó
    // mang `disabledName` để lớp trên còn nói được một dòng nhắc khi bị chạm.
    else if (live) group.userData.nodeId = machine.node_id;
    else group.userData.disabledName = machine.name;
    // Tên máy đi kèm để thẻ ảnh khi rê chuột hỏi đúng máy — API ảnh khoá theo
    // TÊN, còn raycast thì chỉ biết node id.
    group.userData.machineName = machine.name;

    /* Máy của station được đánh dấu bằng BA thứ khác nhau — vòng thép trên sàn,
       bốn dấu chuẩn ở góc, cột sáng dựng đứng. Ba dấu hiệu vì màn hình xưởng bị
       chói và người xem có thể nhìn nghiêng; một dấu hiệu đơn lẻ rất dễ mất
       trong điều kiện đó. Chỉ bật khi ENABLED có giới hạn, tức chỉ ở Line
       Station — Fleet Console cho thao tác mọi máy nên không cần đánh dấu. */
    if (live && ENABLED) {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(4.6, 5.0, 56),
        new THREE.MeshBasicMaterial({ color: 0x5980a6, transparent: true,
                                      opacity: 0.85, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.03;
      group.add(ring);

      const beam = new THREE.Mesh(
        new THREE.CylinderGeometry(0.5, 0.16, 7.5, 14, 1, true),
        new THREE.MeshBasicMaterial({ color: 0x5980a6, transparent: true,
                                      opacity: 0.18, side: THREE.DoubleSide,
                                      depthWrite: false }));
      beam.position.set(0, 5.6, 0);
      group.add(beam);

      for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        const mk = new THREE.Mesh(
          new THREE.BoxGeometry(1.5, 0.05, 0.16),
          new THREE.MeshBasicMaterial({ color: 0x5980a6 }));
        mk.position.set(sx * 5.4, 0.04, sz * 3.4);
        group.add(mk);
        const mk2 = mk.clone();
        mk2.geometry = new THREE.BoxGeometry(0.16, 0.05, 1.2);
        mk2.position.set(sx * 5.4, 0.04, sz * 3.4);
        group.add(mk2);
      }
    }
    group.position.set(x, selected ? 0.18 : 0, z);
    // Xoay cùng hướng với khu bao quanh, nếu không băng tải sẽ chạy đâm vào
    // vách. Các animator dùng toạ độ cục bộ nên xoay nhóm là chúng theo hết.
    group.rotation.y = OPEN_YAW[open] ?? 0;

    const metal = new THREE.MeshStandardMaterial({ color: dark ? 0x9ca9b1 : 0xc7d0d5, roughness: 0.42, metalness: 0.72 });
    const panel = new THREE.MeshStandardMaterial({ color: style.body, roughness: 0.55, metalness: 0.22 });
    const belt = new THREE.MeshStandardMaterial({ color: dark ? 0x1c2b35 : 0x364852, roughness: 0.82, metalness: 0.15 });
    const trim = new THREE.MeshStandardMaterial({ color: style.trim, roughness: 0.47, metalness: 0.12 });
    const ink = new THREE.MeshStandardMaterial({ color: dark ? 0x10171d : 0x222d35, roughness: 0.72, metalness: 0.25 });
    const display = new THREE.MeshStandardMaterial({ color: 0x175a84, emissive: 0x09243b, emissiveIntensity: 0.7, roughness: 0.25 });
    const carton = new THREE.MeshStandardMaterial({ color: productProfile.pack, roughness: 0.9 });
    const cartonTape = new THREE.MeshBasicMaterial({ color: productProfile.tape });
    const cartonMark = new THREE.MeshBasicMaterial({ color: productProfile.mark });
    const bottle = new THREE.MeshStandardMaterial({ color: 0xd1a63e, roughness: 0.32, metalness: 0.05, transparent: true, opacity: 0.92 });
    const cap = new THREE.MeshStandardMaterial({ color: 0x9f2f2f, roughness: 0.48 });
    const lens = new THREE.MeshStandardMaterial({ color: 0x0d2a3d, emissive: 0x1474a8, emissiveIntensity: 0.6, roughness: 0.18 });
    const beam = new THREE.MeshBasicMaterial({ color: 0x4ab4e6, transparent: true, opacity: 0.22 });
    const add = (geometry, material, px, py, pz, target = group) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(px, py, pz);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      target.add(mesh);
      return mesh;
    };

    // Dây băng tải dài 11 m với khung, ray, chân và các vạch chạy động.
    const conveyorLength = 11;
    add(new THREE.BoxGeometry(conveyorLength, 0.16, 1.44), metal, 0, 0.72, 0);
    add(new THREE.BoxGeometry(conveyorLength, 0.09, 1.13), belt, 0, 0.86, 0);
    for (const railZ of [-0.66, 0.66]) {
      const rail = add(new THREE.CylinderGeometry(0.05, 0.05, conveyorLength, 12), metal, 0, 1.2, railZ);
      rail.rotation.z = Math.PI / 2;
    }
    for (let px = -5; px <= 5; px += 2.5) {
      for (const railZ of [-0.47, 0.47]) add(new THREE.CylinderGeometry(0.07, 0.07, 0.72, 10), metal, px, 0.36, railZ);
      add(new THREE.BoxGeometry(0.08, 0.08, 1.05), metal, px, 0.2, 0);
    }
    for (let i = 0; i < 7; i++) {
      const marker = add(new THREE.BoxGeometry(0.34, 0.02, 0.9), new THREE.MeshBasicMaterial({ color: dark ? 0x5e7481 : 0x758994, transparent: true, opacity: 0.48 }), -5.2 + i * 1.55, 0.918, 0);
      if (animateFlow) this.productAnimators.push({ object: marker, min: -5.2, max: 5.2, speed: 1.1, phase: i * 1.55 });
    }

    if (isBottleLine) {
      /* Line 6: băng 1 đưa chai vào robot; robot chuyển sang băng 2 (băng
         chính z=0) để cụm 4 camera đọc datecode. Hai băng chạy song song để
         người xem thấy ngay luồng pick → inspection. */
      const infeedLength = 5.2, infeedX = -3.0, infeedZ = -2.05;
      add(new THREE.BoxGeometry(infeedLength, 0.16, 1.22), metal, infeedX, 0.72, infeedZ);
      add(new THREE.BoxGeometry(infeedLength, 0.09, 0.92), belt, infeedX, 0.86, infeedZ);
      for (const railZ of [-0.55, 0.55]) {
        const rail = add(new THREE.CylinderGeometry(0.045, 0.045, infeedLength, 12), metal, infeedX, 1.14, infeedZ + railZ);
        rail.rotation.z = Math.PI / 2;
      }
      for (const px of [-5.15, -3, -0.85]) {
        for (const railZ of [-0.38, 0.38]) add(new THREE.CylinderGeometry(0.065, 0.065, 0.7, 10), metal, px, 0.36, infeedZ + railZ);
      }
      // Vạch belt sáng hơn phần thân để chuyển động nhìn rõ, kể cả khi camera
      // đang zoom xa và chai che phần lớn mặt băng tải.
      for (let i = 0; i < 4; i++) {
        const tread = add(new THREE.BoxGeometry(0.38, 0.026, 0.72), new THREE.MeshBasicMaterial({ color: 0x87cde0, transparent: true, opacity: 0.68 }), -5.2 + i * 1.28, 0.92, infeedZ);
        if (animateFlow) this.productAnimators.push({ object: tread, min: -5.2, max: -0.8, speed: 0.72, phase: i * 1.28 });
      }
      for (let i = 0; i < 4; i++) {
        const incoming = new THREE.Group();
        add(new THREE.CylinderGeometry(0.2, 0.24, 0.62, 16), bottle, 0, 0.31, 0, incoming);
        add(new THREE.CylinderGeometry(0.1, 0.1, 0.17, 14), bottle, 0, 0.7, 0, incoming);
        add(new THREE.CylinderGeometry(0.11, 0.11, 0.08, 14), cap, 0, 0.83, 0, incoming);
        incoming.position.set(-5.25 + i * 1.05, 1.04, infeedZ);
        group.add(incoming);
        if (animateFlow) this.productAnimators.push({ object: incoming, min: -5.25, max: -1.1, speed: 0.68, phase: i * 1.05 });
      }

      const robot = new THREE.Group();
      robot.position.set(-0.35, 0, -1.0);
      group.add(robot);
      const robotOrange = new THREE.MeshStandardMaterial({ color: 0xd96727, roughness: 0.48, metalness: 0.25 });
      const robotDark = new THREE.MeshStandardMaterial({ color: 0x26343d, roughness: 0.6, metalness: 0.36 });
      const robotAdd = (geometry, material, px, py, pz, target = robot) => {
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(px, py, pz);
        mesh.castShadow = true;
        target.add(mesh);
        return mesh;
      };
      robotAdd(new THREE.CylinderGeometry(0.56, 0.7, 0.3, 20), robotDark, 0, 0.15, 0);
      robotAdd(new THREE.CylinderGeometry(0.36, 0.46, 0.5, 20), robotOrange, 0, 0.53, 0);
      robotAdd(new THREE.CylinderGeometry(0.16, 0.2, 1.38, 14), robotDark, 0, 1.18, 0);
      const shoulder = new THREE.Group();
      shoulder.position.set(0, 1.5, 0);
      robot.add(shoulder);
      robotAdd(new THREE.CylinderGeometry(0.26, 0.3, 0.28, 16), robotDark, 0, 0, 0, shoulder);
      const upper = robotAdd(new THREE.BoxGeometry(0.82, 0.24, 0.32), robotOrange, 0.41, 0.11, 0, shoulder);
      upper.rotation.z = -0.16;
      const elbow = new THREE.Group();
      elbow.position.set(0.78, 0.2, 0);
      shoulder.add(elbow);
      robotAdd(new THREE.CylinderGeometry(0.23, 0.26, 0.28, 16), robotDark, 0, 0, 0, elbow);
      const forearm = robotAdd(new THREE.BoxGeometry(0.58, 0.2, 0.28), robotOrange, 0.29, 0.05, 0, elbow);
      forearm.rotation.z = 0.12;
      const wrist = new THREE.Group();
      wrist.position.set(0.54, 0.08, 0);
      elbow.add(wrist);
      robotAdd(new THREE.CylinderGeometry(0.1, 0.1, 0.76), robotDark, 0, -0.37, 0, wrist);
      robotAdd(new THREE.BoxGeometry(0.42, 0.08, 0.34), robotDark, 0, -0.76, 0, wrist);
      for (const fingerZ of [-0.13, 0.13]) robotAdd(new THREE.BoxGeometry(0.22, 0.16, 0.07), robotOrange, 0.13, -0.88, fingerZ, wrist);
      // Chai transfer là thực thể độc lập, đi theo quỹ đạo từ băng 1 sang băng
      // 2. Vì vậy không còn chai treo cố định dưới kẹp như mô hình cũ.
      const transferBottle = new THREE.Group();
      add(new THREE.CylinderGeometry(0.2, 0.24, 0.62, 16), bottle, 0, 0.31, 0, transferBottle);
      add(new THREE.CylinderGeometry(0.1, 0.1, 0.17, 14), bottle, 0, 0.7, 0, transferBottle);
      add(new THREE.CylinderGeometry(0.11, 0.11, 0.08, 14), cap, 0, 0.83, 0, transferBottle);
      group.add(transferBottle);
      // Cánh tay robot cũng theo cùng một luật: máy dừng thì tay đứng yên.
      if (animateFlow)
        this.robotAnimators.push({ shoulder, elbow, wrist, transferBottle,
          baseX: -0.35, baseZ: -1.0, speed: 0.115, phase: 0.18 });
    }

    // Trạm OCR: carton dùng tunnel camera phía trên; line chai dùng buồng mở để
    // bốn camera (trên, hai bên, dưới) nhìn thấy chai và vùng datecode.
    const station = new THREE.Group();
    group.add(station);
    if (!isBottleLine) {
      add(new THREE.BoxGeometry(1.9, 0.18, 2.15), panel, 0, 3.05, 0, station);
      for (const side of [-1, 1]) {
        add(new THREE.BoxGeometry(0.18, 2.15, 2.15), trim, 0, 1.95, side * 0.98, station);
      }
      add(new THREE.BoxGeometry(1.55, 0.72, 0.06), display, 0, 2.0, -1.1, station);
      const camera = add(new THREE.BoxGeometry(0.5, 0.36, 0.52), ink, 0, 2.68, 0, station);
      camera.castShadow = true;
      add(new THREE.CylinderGeometry(0.13, 0.13, 0.12, 18), lens, 0, 2.43, 0, station);
      const lightBar = add(new THREE.BoxGeometry(1.25, 0.08, 0.08), new THREE.MeshBasicMaterial({ color: 0xbbefff }), 0, 2.42, -0.58, station);
      lightBar.material.transparent = true;
      lightBar.material.opacity = 0.75;
    } else {
      for (const side of [-1, 1]) {
        add(new THREE.CylinderGeometry(0.065, 0.065, 3.35, 12), metal, 0, 2.1, side * 1.5, station);
        const rail = add(new THREE.CylinderGeometry(0.06, 0.06, 3.1, 12), metal, 0, 3.72, side * 0.76, station);
        rail.rotation.z = Math.PI / 2;
      }
      add(new THREE.BoxGeometry(0.18, 0.18, 3.25), metal, 0, 3.75, 0, station);
      const cameraAt = (px, py, pz, direction) => {
        const head = add(new THREE.BoxGeometry(0.48, 0.34, 0.48), ink, px, py, pz, station);
        const glass = add(new THREE.CylinderGeometry(0.13, 0.13, 0.16, 18), lens, px, py, pz, station);
        const scan = add(new THREE.CylinderGeometry(0.1, 0.26, 0.82, 18, 1, true), beam, px, py, pz, station);
        if (direction === 'down') {
          glass.position.y -= 0.25;
          scan.position.y -= 0.55;
        } else if (direction === 'up') {
          glass.position.y += 0.25;
          scan.position.y += 0.55;
        } else {
          glass.rotation.x = Math.PI / 2;
          scan.rotation.x = Math.PI / 2;
          const sign = direction === 'left' ? 1 : -1;
          glass.position.z += sign * 0.26;
          scan.position.z += sign * 0.55;
        }
        head.castShadow = true;
      };
      cameraAt(0, 3.42, 0, 'down');       // camera phía trên
      cameraAt(0, 1.84, 1.55, 'left');    // camera bên trái
      cameraAt(0, 1.84, -1.55, 'right');  // camera bên phải
      cameraAt(0, 0.33, 0, 'up');         // camera bên dưới đọc datecode
      add(new THREE.BoxGeometry(0.8, 0.08, 2.85), new THREE.MeshBasicMaterial({ color: 0x5fc1ea, transparent: true, opacity: 0.28 }), 0, 0.52, 0, station);
    }

    // Tủ điều khiển và đèn trạng thái, đặt tách khỏi trạm đọc để đường line rõ.
    add(new THREE.BoxGeometry(0.95, 2.05, 0.76), panel, -4.55, 1.03, 1.42);
    add(new THREE.BoxGeometry(0.62, 0.42, 0.06), display, -4.55, 1.4, 1.83);
    add(new THREE.CylinderGeometry(0.055, 0.055, 0.72, 10), metal, -4.55, 2.38, 1.42);
    /* Máy bị tắt KHÔNG có đèn trạng thái. Cố ý, không phải quên: trạng thái
       line khác không phải việc của người đang đứng ở đây, và hiện lên là mời
       so sánh sai — line này chạy quế, line kia chạy muối. */
    if (style.beacon) {
      const beacon = add(new THREE.CylinderGeometry(0.17, 0.17, 0.36, 18), new THREE.MeshStandardMaterial({ color: style.beacon, emissive: style.beacon, emissiveIntensity: 1.1 }), -4.55, 2.92, 1.42);
      beacon.castShadow = false;
    }

    // Cổng reject và thùng loại đặt ngay sau trạm OCR.
    add(new THREE.BoxGeometry(0.86, 0.1, 0.1), new THREE.MeshStandardMaterial({ color: 0xe5a32c, roughness: 0.45 }), 2.08, 1.18, 0.78);
    add(new THREE.CylinderGeometry(0.1, 0.1, 0.42, 12), metal, 1.67, 0.98, 0.78);
    add(new THREE.BoxGeometry(1.05, 0.9, 0.95), metal, 2.3, 0.45, 1.72);

    // Hàng hoá chạy thật trên băng tải: bốn carton line và một bottle line.
    for (let i = 0; i < 5; i++) {
      const product = new THREE.Group();
      if (isBottleLine) {
        const body = add(new THREE.CylinderGeometry(0.22, 0.26, 0.68, 16), bottle, 0, 0.34, 0, product);
        body.castShadow = true;
        add(new THREE.CylinderGeometry(0.11, 0.11, 0.18, 14), bottle, 0, 0.77, 0, product);
        add(new THREE.CylinderGeometry(0.12, 0.12, 0.08, 14), cap, 0, 0.9, 0, product);
      } else {
        const [width, height, depth] = productProfile.shape;
        add(new THREE.BoxGeometry(width, height, depth), carton, 0, height / 2, 0, product);
        add(new THREE.BoxGeometry(width + 0.04, 0.05, 0.1), cartonTape, 0, height * 0.77, 0, product);
        add(new THREE.BoxGeometry(0.08, height * 0.62, depth + 0.04), cartonMark, width * 0.2, height / 2, 0, product);
      }
      product.position.set(-5 + i * 2.12, 1.04, 0);
      group.add(product);
      if (animateFlow) this.productAnimators.push({ object: product, min: -5.2, max: 5.2, speed: isBottleLine ? 0.82 : 1.05, phase: i * 2.12 });
    }

    if (!isBottleLine) {
      const [width, height, depth] = productProfile.shape;
      for (let stack = 0; stack < 3; stack++) {
        add(new THREE.BoxGeometry(width, height, depth), carton, 4.9 + stack * 0.05, height / 2 + stack * height, -1.6);
        add(new THREE.BoxGeometry(width + 0.04, 0.05, 0.1), cartonTape, 4.9 + stack * 0.05, height * 0.77 + stack * height, -1.6);
      }
    }

    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(12, 4.2, isBottleLine ? 6.4 : 4.3)),
      new THREE.LineBasicMaterial({ color: 0x1677b8, transparent: true, opacity: 0.9 }),
    );
    outline.position.y = 2.1;
    outline.visible = selected;
    group.add(outline);
    const subtitle = isBottleLine ? 'BOTTLE LINE · ROBOT + 4-CAMERA CHECK' : `CARTON · ${productProfile.label}`;
    const label = textSprite(THREE, machine.name, live ? subtitle : '', dark);
    if (!live) label.material.opacity = 0.62;   // nhãn máy khác mờ 62%
    label.position.set(0, 5.3, 0);
    group.add(label);
    this.machineRoot.add(group);
    this.pickables.push(group);
  }

  update({ machines, selected, onSelect }) {
    this.onSelect = onSelect;
    this.miniSelected = selected;
    // Danh sách máy vừa đổi thì cái đang trỏ tới có thể không còn — bỏ hover
    // trước, nếu không vòng sáng đứng lại ở chỗ một máy đã biến mất.
    this.hoverId = null;
    if (this.halo) this.halo.visible = false;
    this.canvas.setAttribute('aria-label', store.t.floorTitle);
    this.refreshFloorTheme();
    disposeObject(this.machineRoot);
    this.pickables = [];
    this.productAnimators = [];
    this.robotAnimators = [];
    const realPlaced = machines.filter(machine => machine.floor);
    const realNames = new Set(realPlaced.map(machine => machine.name));
    const placed = [...realPlaced, ...OFFLINE_CARTON_LINES.filter(machine => !realNames.has(machine.name))];
    this.miniMachines = placed;
    if (!placed.length) {
      this.draw();
      return;
    }
    /* Toạ độ trong config là nhãn logic của tier 1 (bản đồ 2D dùng nó). Sơ đồ
       3D thì dựng theo mặt bằng thật, khai trong PLANT ở đầu file. */
    PLANT.areas.forEach(area => this.addArea(area));
    PLANT.stores.forEach(s => this.addWarehouse(s.label, s.x, s.z, { bays: s.bays, fill: s.fill }));
    this.addEntrance(PLANT.entrance.x, PLANT.entrance.z);
    this.addOpsRoom(PLANT.ops.x, PLANT.ops.z);
    this.addHallShell();
    this.addPlantProps();
    this.addForklifts();
    placed.forEach((machine, index) => {
      const pos = PLANT.lines[machine.name] || { x: -18 + index * 9, z: 10, open: 'south' };
      if (machine.name === 'Line 5') this.addTin2Annex(pos.x, pos.z);
      this.addZone(machine, pos.x, pos.z, index, pos.open);
      this.addMachine(machine, pos.x, pos.z, machine.node_id === selected, pos.open);
    });
    this.addAisleStaff();
    this.draw();
  }

  resize() {
    const width = Math.max(1, this.el.clientWidth);
    const height = Math.max(1, this.el.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.draw();
  }

  aimCamera() {
    const horizontal = this.distance * Math.cos(this.elevation);
    this.camera.position.set(
      this.cameraTarget.x + horizontal * Math.cos(this.azimuth),
      this.cameraTarget.y + this.distance * Math.sin(this.elevation),
      this.cameraTarget.z + horizontal * Math.sin(this.azimuth),
    );
    this.camera.lookAt(this.cameraTarget);
  }

  /* ── Bản đồ 2D ────────────────────────────────────────────────────────── */

  /** Khung nhìn của minimap: bao trọn sảnh chính CỘNG nhà Line 5, cộng lề. Tính
   *  từ PLANT nên thêm/bớt khu là minimap tự co theo, không phải sửa số. */
  miniBounds() {
    const h = PLANT.hall;
    let x0 = h.x - h.w / 2, x1 = h.x + h.w / 2;
    let z0 = h.z - h.d / 2, z1 = h.z + h.d / 2;
    for (const p of Object.values(PLANT.lines)) {
      x0 = Math.min(x0, p.x - 13); x1 = Math.max(x1, p.x + 13);
      z0 = Math.min(z0, p.z - 10); z1 = Math.max(z1, p.z + 10);
    }
    return { x0: x0 - 3, x1: x1 + 3, z0: z0 - 3, z1: z1 + 3 };
  }

  miniProject(b, w, h) {
    const s = Math.min(w / (b.x1 - b.x0), h / (b.z1 - b.z0));
    return {
      s,
      ox: (w - (b.x1 - b.x0) * s) / 2 - b.x0 * s,
      oy: (h - (b.z1 - b.z0) * s) / 2 - b.z0 * s,
    };
  }

  drawMini() {
    const c = this.mini;
    if (!c || !this.miniOn) return;   // đang ẩn thì khỏi vẽ
    const g = c.getContext('2d');
    const dark = store.theme === 'dark';
    const W = c.width, H = c.height;
    g.clearRect(0, 0, W, H);
    g.fillStyle = dark ? 'rgba(14,22,30,0.82)' : 'rgba(255,255,255,0.88)';
    g.fillRect(0, 0, W, H);
    const b = this.miniBounds();
    const { s, ox, oy } = this.miniProject(b, W - 14, H - 14);
    const X = x => ox + x * s + 7;
    const Y = z => oy + z * s + 7;

    // Bao nhà chính
    const h = PLANT.hall;
    g.strokeStyle = dark ? '#4b5f70' : '#9fb0bd';
    g.lineWidth = 1.4;
    g.strokeRect(X(h.x - h.w / 2), Y(h.z - h.d / 2), h.w * s, h.d * s);

    // Các khu có tên
    for (const a of PLANT.areas) {
      g.fillStyle = a.tint + (dark ? '30' : '22');
      g.fillRect(X(a.x - a.w / 2), Y(a.z - a.d / 2), a.w * s, a.d * s);
      g.strokeStyle = a.tint + '99';
      g.lineWidth = 1;
      g.strokeRect(X(a.x - a.w / 2), Y(a.z - a.d / 2), a.w * s, a.d * s);
    }

    // Kho
    g.fillStyle = dark ? '#55636f' : '#c2cbd3';
    for (const st of PLANT.stores) g.fillRect(X(st.x) - 7, Y(st.z) - 4, 14, 8);

    // Hình quạt hướng camera — đứng ở đâu nhìn vào đâu.
    const t = this.cameraTarget;
    const camX = t.x + Math.cos(this.azimuth) * this.distance * 0.42;
    const camZ = t.z + Math.sin(this.azimuth) * this.distance * 0.42;
    g.beginPath();
    g.moveTo(X(camX), Y(camZ));
    const ang = Math.atan2(t.z - camZ, t.x - camX);
    for (const d of [-0.42, 0.42]) {
      g.lineTo(X(camX + Math.cos(ang + d) * 26), Y(camZ + Math.sin(ang + d) * 26));
    }
    g.closePath();
    g.fillStyle = dark ? 'rgba(126,178,214,0.20)' : 'rgba(45,108,157,0.16)';
    g.fill();

    // Máy: chấm tô theo trạng thái, có nhãn.
    this.miniHits = [];
    for (const m of this.miniMachines || []) {
      const p = PLANT.lines[m.name];
      if (!p) continue;
      const cx = X(p.x), cy = Y(p.z);
      const ok = !['unreachable', 'offline'].includes(m.state);
      const hovered = m.node_id === this.hoverId;
      const chosen = m.node_id === this.miniSelected;
      // Vòng ngoài khi đang trỏ tới — cùng màu với vòng sáng dưới sàn cảnh 3D,
      // để mắt nối được hai khung nhìn mà không phải đọc tên.
      if (hovered) {
        g.beginPath();
        g.arc(cx, cy, 9, 0, Math.PI * 2);
        g.fillStyle = 'rgba(45,140,205,0.22)';
        g.fill();
        g.strokeStyle = '#2d8ccd';
        g.lineWidth = 1.6;
        g.stroke();
      }
      g.beginPath();
      g.arc(cx, cy, chosen || hovered ? 6 : 4.4, 0, Math.PI * 2);
      g.fillStyle = ok ? (m.state === 'agent_down' ? '#d08b2c' : '#2f8f5b') : '#b0392c';
      g.fill();
      if (chosen) {
        g.strokeStyle = dark ? '#e8f1f7' : '#1d2b36';
        g.lineWidth = 1.6;
        g.stroke();
      }
      g.fillStyle = hovered ? (dark ? '#eaf3fa' : '#12232f') : (dark ? '#c9d6e0' : '#3a4650');
      g.font = `${hovered ? 'bold ' : ''}9px system-ui, sans-serif`;
      g.textAlign = 'center';
      g.fillText(m.name, cx, cy - (hovered ? 12 : 7));
      this.miniHits.push({ x: cx, y: cy, node_id: m.node_id });
    }
  }

  /** Đổi máy đang trỏ tới — DÙNG CHUNG cho cả hai chiều.
   *
   *  Trỏ vào máy trong cảnh 3D thì chấm tương ứng trên minimap sáng lên, và
   *  ngược lại. Một đường vào duy nhất, nên hai khung nhìn không thể lệch nhau:
   *  mọi thứ đều đọc `this.hoverId`. */
  setHover(nodeId) {
    if (this.hoverId === nodeId) return;
    this.hoverId = nodeId || null;
    const p = this.hoverId && PLANT.lines[this.hoverName(this.hoverId)];
    if (this.halo) this.halo.visible = false;
    if (p) {
      if (!this.halo) {
        const THREE = this.THREE;
        const ring = new THREE.Mesh(
          new THREE.RingGeometry(6.6, 7.5, 48),
          new THREE.MeshBasicMaterial({ color: 0x2d8ccd, transparent: true, opacity: 0.55, side: THREE.DoubleSide }),
        );
        ring.rotation.x = -Math.PI / 2;
        // Gắn vào scene chứ không vào machineRoot: machineRoot bị huỷ mỗi lần
        // update(), vòng sáng thì phải sống qua các lần vẽ lại.
        this.scene.add(ring);
        this.halo = ring;
      }
      this.halo.position.set(p.x, 0.09, p.z);
      this.halo.visible = true;
    }
    this.draw();
  }

  hoverName(nodeId) {
    return (this.miniMachines || []).find(m => m.node_id === nodeId)?.name || '';
  }

  toggleMini(on) {
    if (!this.mini) return;          // bề mặt này tắt hẳn minimap
    this.miniOn = on === undefined ? !this.miniOn : on;
    if (this.mini) this.mini.hidden = !this.miniOn;
    this.miniBtn?.setAttribute('aria-pressed', String(this.miniOn));
    try { localStorage.setItem('factory_mini', this.miniOn ? '1' : '0'); } catch { /* chế độ riêng tư */ }
    if (this.miniOn) this.drawMini();
  }

  miniHover(event) {
    const hit = this.miniHitAt(event);
    this.mini.style.cursor = hit ? 'pointer' : 'default';
    this.setHover(hit?.node_id || null);
  }

  miniHitAt(event) {
    const rect = this.mini.getBoundingClientRect();
    const px = (event.clientX - rect.left) * (this.mini.width / rect.width);
    const py = (event.clientY - rect.top) * (this.mini.height / rect.height);
    let best = null, bestD = 16 * 16;
    for (const hit of this.miniHits || []) {
      const d = (hit.x - px) ** 2 + (hit.y - py) ** 2;
      if (d < bestD) { bestD = d; best = hit; }
    }
    return best;
  }

  miniClick(event) {
    const best = this.miniHitAt(event);
    if (best) {
      this.focusMachine(best.node_id);
      if (this.onSelect) this.onSelect(best.node_id);
    }
  }

  draw() {
    this.aimCamera();
    this.renderer.render(this.scene, this.camera);
    this.drawMini();
  }

  targetAt(event) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObjects(this.pickables, true)[0];
    if (!hit) return null;
    let node = hit.object;
    while (node && !node.userData.nodeId && !node.userData.focus) node = node.parent;
    return node?.userData || null;
  }

  bindEvents() {
    this.canvas.tabIndex = 0;   // nhận được phím, và có vòng focus khi tab tới
    this.canvas.addEventListener('contextmenu', e => e.preventDefault());
    this.canvas.addEventListener('pointerdown', event => {
      this.canvas.setPointerCapture(event.pointerId);
      this.canvas.focus({ preventScroll: true });
      this.cameraMove = null;
      // Chuột phải / chuột giữa / giữ Shift = dời tâm nhìn, chuột trái = xoay.
      const pan = event.button === 2 || event.button === 1 || event.shiftKey;
      this.drag = { x: event.clientX, y: event.clientY, moved: false, pan };
      this.canvas.classList.add('is-dragging');
    });
    this.canvas.addEventListener('pointermove', event => {
      if (this.drag) {
        const dx = event.clientX - this.drag.x;
        const dy = event.clientY - this.drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) this.drag.moved = true;
        if (this.drag.pan) {
          this.panBy(dx, dy);
        } else {
          this.azimuth -= dx * 0.009;
          this.elevation = Math.max(0.28, Math.min(1.45, this.elevation + dy * 0.008));
        }
        this.drag.x = event.clientX;
        this.drag.y = event.clientY;
        this.draw();
      } else {
        const hit = this.targetAt(event);
        // Máy bị tắt KHÔNG đổi con trỏ thành pointer: con trỏ đó hứa một hành
        // động không tồn tại.
        this.canvas.style.cursor = (hit && !hit.disabledName) ? 'pointer' : 'grab';
        this.setHover(hit?.nodeId || null);
        this.showPeek(hit, event);
      }
    });
    this.canvas.addEventListener('pointerup', event => {
      const wasClick = this.drag && !this.drag.moved;
      this.drag = null;
      this.canvas.classList.remove('is-dragging');
      if (wasClick) {
        const target = this.targetAt(event);
        if (target?.focus) this.focusZone(target.focus);
        else if (target?.nodeId) this.onSelect?.(target.nodeId);
        // Chạm máy bị tắt phải NÓI GÌ ĐÓ. Im lặng thì người dùng tưởng màn hình
        // đơ rồi chạm tiếp.
        else if (target?.disabledName) this.onDisabledTap?.(target.disabledName);
      }
    });
    this.canvas.addEventListener('pointerleave', () => { this.hidePeek(); this.setHover(null); });
    this.canvas.addEventListener('pointercancel', () => {
      this.drag = null;
      this.canvas.classList.remove('is-dragging');
    });
    /* Zoom đi VỀ PHÍA con trỏ, không về giữa màn hình. Zoom vào tâm thì muốn
       xem kỹ một máy ở rìa là phải zoom rồi kéo, zoom rồi kéo — còn zoom theo
       con trỏ thì chỉ cần trỏ vào máy đó và cuộn. */
    this.canvas.addEventListener('wheel', event => {
      event.preventDefault();
      const before = this.groundAt(event);
      const next = Math.max(14, Math.min(78, this.distance + event.deltaY * 0.018));
      const ratio = 1 - next / this.distance;
      this.distance = next;
      if (before && ratio > 0) {
        this.cameraTarget.x += (before.x - this.cameraTarget.x) * ratio;
        this.cameraTarget.z += (before.z - this.cameraTarget.z) * ratio;
      }
      this.cameraMove = null;
      this.draw();
    }, { passive: false });

    // Bấm đúp một máy = bay tới xem gần. Bấm một lần vẫn là chọn, nên thao tác
    // cũ không đổi nghĩa.
    this.canvas.addEventListener('dblclick', event => {
      const target = this.targetAt(event);
      if (target?.nodeId) this.focusMachine(target.nodeId);
      else if (target?.focus) this.focusZone(target.focus);
    });

    /* Bàn phím: màn hình cạnh dây chuyền hay đặt xa tầm với, và không phải chỗ
       nào cũng có chuột tử tế. */
    this.canvas.addEventListener('keydown', event => {
      const step = event.shiftKey ? 0.22 : 0.08;
      const map = {
        ArrowLeft:  () => (this.azimuth -= step),
        ArrowRight: () => (this.azimuth += step),
        ArrowUp:    () => (this.elevation = Math.min(1.45, this.elevation + step * .6)),
        ArrowDown:  () => (this.elevation = Math.max(0.28, this.elevation - step * .6)),
        '+': () => this.zoomBy(-4), '=': () => this.zoomBy(-4),
        '-': () => this.zoomBy(4),  '_': () => this.zoomBy(4),
        '0': () => this.setView('iso'),
        '1': () => this.setView('top'),
        '2': () => this.setView('front'),
        m: () => this.toggleMini(), M: () => this.toggleMini(),
      };
      const fn = map[event.key];
      if (!fn) return;
      event.preventDefault();
      this.cameraMove = null;
      fn();
      this.draw();
    });
  }
}

async function start(el) {
  const queued = pending.get(el);
  try {
    const THREE = await loadThree();
    const scene = new FactoryFloorScene(THREE, el);
    scenes.set(el, scene);
    pending.delete(el);
    scene.update(queued.options);
  } catch (error) {
    console.warn('3D factory map unavailable; using the SVG fallback.', error);
    /* Bậc 1 do BÊN GỌI cấp, vì hai bề mặt có luật khác nhau: Fleet Console vẽ
       đèn cho mọi máy, Line Station chỉ vẽ máy của chính nó. Trước đây chỗ này
       import cứng './factory-map.js' — file chỉ có ở fleet — nên ở Line Station
       khi WebGL lỗi (tablet cũ, driver hỏng) sơ đồ trắng trơn, không báo gì.
       Nhánh này là async, nên try/catch của bên gọi cũng không bắt được. */
    const latest = pending.get(el)?.options || queued.options;
    pending.delete(el);
    if (!fallbackLoader) {
      el.innerHTML = '<div class="map-loading">Không dựng được sơ đồ nhà máy.</div>';
      console.error('factory-map-3d: thiếu options.fallback, không có sơ đồ bậc 1.');
      return;
    }
    const fallback = await fallbackLoader();
    fallback.render(el, latest);
  }
}

export function render(el, options) {
  // `enabledKeys`: chỉ những máy này thao tác được. Không truyền = tất cả.
  ENABLED = options?.enabledKeys || null;
  if (options?.store) store = options.store;
  if (options?.fallback) fallbackLoader = options.fallback;
  MINIMAP = options?.minimap !== false;
  const scene = scenes.get(el);
  if (scene) {
    scene.update(options);
    return;
  }
  const queued = pending.get(el);
  if (queued) {
    queued.options = options;
    return;
  }
  el.innerHTML = '<div class="map-loading">Loading 3D factory floor…</div>';
  pending.set(el, { options });
  void start(el);
}

/** Chú giải trạng thái.
 *
 *  `t` truyền vào chứ không đọc `store` cấp module: `store` chỉ được gán khi
 *  `render()` chạy, mà `paintFold()` vẽ chú giải TRƯỚC lần render đầu tiên —
 *  nên nó đọc phải stub `{ state: {} }` và in ra bốn chữ "undefined". Nhận
 *  tham số thì hết phụ thuộc vào thứ tự gọi.
 */
export function legendHTML(t) {
  t = t || store.t;
  const dot = color => `<svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="${color}"/></svg>`;
  const tri = '<svg width="12" height="12"><polygon points="6,1 11,10 1,10" fill="none" stroke="#9a6a00" stroke-width="1.5"/></svg>';
  const chat = '<svg width="14" height="12" stroke="#98989b" stroke-width="1.3" fill="none"><rect x="1" y="1" width="12" height="8" rx="2"/><line x1="0" y1="11" x2="14" y2="0"/></svg>';
  const cross = '<svg width="12" height="12" stroke="#98989b" stroke-width="1.4"><line x1="2" y1="2" x2="10" y2="10"/><line x1="10" y1="2" x2="2" y2="10"/></svg>';
  return `<span class="k">${dot('#2f7d4f')} ${t.state.ok}</span>
    <span class="k">${tri} ${t.state.warn}</span>
    <span class="k">${chat} ${t.state.agent_down}</span>
    <span class="k">${cross} ${t.state.unreachable}</span>`;
}
