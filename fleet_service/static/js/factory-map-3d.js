/* ═══════════════════════════════════════════════════════════════════════════
   FactoryMap — sơ đồ nhà máy 3D tương tác.

   Giữ nguyên public interface của `factory-map.js` để `app.js` vẫn chỉ biết
   render({ machines, selected, onSelect }). Three.js tải lười: nếu tablet ở
   xưởng không ra Internet/CDN, sơ đồ SVG bậc 1 vẫn là phương án dự phòng.
   ═══════════════════════════════════════════════════════════════════════════ */

import { store } from './core.js';

const THREE_URL = 'https://unpkg.com/three@0.184.0/build/three.module.js';
const pending = new WeakMap();
const scenes = new WeakMap();
let threePromise;

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
  Auto2:    { label: 'ONION POWDER', pack: '#c89a50', tape: '#f1d389', mark: '#7d5424', shape: [0.74, 0.58, 0.68] },
  M1:       { label: 'CHILI POWDER', pack: '#bd5139', tape: '#f2c95c', mark: '#7f271d', shape: [0.82, 0.5, 0.64] },
  M2:       { label: 'CINNAMON', pack: '#805034', tape: '#d59b43', mark: '#e9c39a', shape: [0.68, 0.72, 0.76] },
  LineTine: { label: 'PURE SEA SALT', pack: '#e2e9eb', tape: '#3d8aa5', mark: '#1e5d73', shape: [0.72, 0.62, 0.72] },
  'Auto 1': { label: 'AUTO 1 CARTON', pack: '#6b7fa8', tape: '#d7e0f2', mark: '#263c65', shape: [0.8, 0.54, 0.7] },
  'Tin 2':  { label: 'TIN 2 CARTON', pack: '#61997c', tape: '#e5d36e', mark: '#24563e', shape: [0.7, 0.68, 0.74] },
};

const OFFLINE_CARTON_LINES = [
  { node_id: '__offline_auto_1__', name: 'Auto 1', line: 'Carton line 5',
    model: 'Jetson Orin Nano 8GB Super', state: 'offline', floor: { x: 0, y: 0 }, virtual: true },
  { node_id: '__offline_tin_2__', name: 'Tin 2', line: 'Carton line 6 · annex building',
    model: 'Jetson Orin Nano 8GB Super', state: 'offline', floor: { x: 0, y: 0 }, virtual: true, building: 'tin2' },
];

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
    this.cameraTarget = new THREE.Vector3(0, 0, 0);
    this.cameraMove = null;
    this.distance = 47;
    this.azimuth = -0.72;
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
    this.zoneNav.innerHTML = '<button type="button" data-focus="main" aria-pressed="true">Main hall</button><button type="button" data-focus="tin2" aria-pressed="false">Tin 2 annex</button>';
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
      + '<button type="button" data-zoom="1" title="Zoom out (\u2212)">\u2212</button>';
    this.viewNav.addEventListener('click', event => {
      const v = event.target.closest('button[data-view]');
      if (v) { this.setView(v.dataset.view); return; }
      const z = event.target.closest('button[data-zoom]');
      if (z) this.zoomBy(Number(z.dataset.zoom) * 6);
    });
    el.append(this.viewNav);

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
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.1, 140);
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
    const slab = new THREE.Mesh(
      new THREE.BoxGeometry(102, 0.34, 44),
      new THREE.MeshStandardMaterial({ color: dark ? 0x19242e : 0xe7edf2, roughness: 0.96 }),
    );
    slab.position.x = 20;
    slab.position.y = -0.25;
    slab.receiveShadow = true;
    floor.add(slab);
    const grid = new THREE.GridHelper(102, 102, dark ? 0x536373 : 0xc2ced8, dark ? 0x344454 : 0xd7e0e7);
    grid.position.x = 20;
    grid.position.y = -0.055;
    grid.material.transparent = true;
    grid.material.opacity = dark ? 0.42 : 0.75;
    floor.add(grid);

    const aisleMaterial = new THREE.MeshBasicMaterial({ color: dark ? 0xa5b4c2 : 0x9aa9b7, transparent: true, opacity: 0.58 });
    const aisle = new THREE.Mesh(
      new THREE.BoxGeometry(52, 0.025, 3.4),
      new THREE.MeshBasicMaterial({ color: dark ? 0x263b4a : 0xd4dfe8, transparent: true, opacity: 0.7 }),
    );
    aisle.position.y = 0.005;
    floor.add(aisle);
    for (let x = -25; x < 25; x += 1.4) {
      for (const z of [-1.38, 1.38]) {
        const dash = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.025, 0.1), aisleMaterial);
        dash.position.set(x, 0.02, z);
        floor.add(dash);
      }
    }
    const road = new THREE.Mesh(
      new THREE.BoxGeometry(17, 0.04, 8),
      new THREE.MeshStandardMaterial({ color: dark ? 0x1c252d : 0xb9c0c5, roughness: 0.96 }),
    );
    road.position.set(34, 0.025, 0);
    floor.add(road);
    for (let x = 27; x < 41; x += 1.5) {
      const dash = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.025, 0.14), aisleMaterial);
      dash.position.set(x, 0.06, 0);
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
    const home = new T.Vector3(0, 0, 0);
    if (preset === 'top') this.flyTo({ target: home, distance: 62, azimuth: -Math.PI / 2, elevation: 1.45 });
    else if (preset === 'front') this.flyTo({ target: new T.Vector3(0, 1.6, 0), distance: 34, azimuth: -Math.PI / 2, elevation: 0.3 });
    else this.flyTo({ target: home, distance: 47, azimuth: -0.72, elevation: 0.72 });
    this.viewNav?.querySelectorAll('button[data-view]').forEach(b =>
      b.setAttribute('aria-pressed', String(b.dataset.view === (preset || 'iso'))));
  }

  zoomBy(amount) {
    this.distance = Math.max(14, Math.min(78, this.distance + amount));
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
      ? { point: new this.THREE.Vector3(55, 0, 0), distance: 30 }
      : { point: new this.THREE.Vector3(0, 0, 0), distance: 47 };
    this.cameraMove = {
      started: performance.now(), duration: 700,
      fromTarget: this.cameraTarget.clone(), toTarget: target.point,
      fromDistance: this.distance, toDistance: target.distance,
    };
    this.zoneNav.querySelectorAll('button').forEach(button =>
      button.setAttribute('aria-pressed', String(button.dataset.focus === name)));
  }

  addWarehouse(label, x, z) {
    const THREE = this.THREE;
    const dark = store.theme === 'dark';
    const group = new THREE.Group();
    const material = new THREE.MeshStandardMaterial({ color: dark ? 0x4c5863 : 0xd2d7dc, roughness: 0.9 });
    const roof = new THREE.Mesh(new THREE.BoxGeometry(4.1, 1.25, 3.2), material);
    roof.position.y = 0.6;
    roof.castShadow = true;
    roof.receiveShadow = true;
    group.add(roof);
    const labelSprite = textSprite(THREE, label, '', dark);
    labelSprite.scale.set(3.7, 0.98, 1);
    labelSprite.position.set(0, 2.2, 0);
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
    const sign = textSprite(THREE, 'TIN 2', 'ANNEX BUILDING · CARTON LINE 6', dark);
    sign.position.set(0, 5.0, -7.65);
    sign.scale.set(5.1, 1.34, 1);
    building.add(sign);
    building.position.set(x, 0, z);
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

  addZone(machine, x, z, index) {
    const THREE = this.THREE;
    const dark = store.theme === 'dark';
    const isBottleLine = machine.name === 'PC-Auto-1';
    const product = LINE_PRODUCTS[machine.name] || { pack: '#d7b34d', tape: '#9e2f2f', mark: '#7b5322' };
    const awayFromAisle = z < 0 ? -1 : 1;
    const towardAisle = -awayFromAisle;
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
    this.machineRoot.add(zone);
  }

  addAisleStaff() {
    const staff = new this.THREE.Group();

    /* Người đi TUẦN CÁC LINE, không phải đi tới đi lui trên lối đi. Đi dọc mãi
       một đường thẳng thì nhìn ra ngay là hoạt ảnh; còn rẽ vào một line, đứng
       lại một lúc bên máy rồi ra tiếp sang line khác thì đúng là việc người ta
       làm trong ca.

       Toạ độ bám bố trí thật: hai dãy máy ở z = ±10, lối đi ở z ≈ 0, mép zone
       phía lối đi ở z = ±6,3 — nên điểm dừng đặt ở đó là đứng ngay cạnh máy,
       không phải đứng xuyên qua tường.                                        */
    const A = 1.25, B = -1.25;          // hai làn của lối đi
    const IN_TOP = -6.3, IN_BOT = 6.3;  // mép zone phía lối đi

    const walkers = [
      { // tổ trưởng: hai line dãy trên
        shirt: 0x424d66, role: 'supervisor', speed: 2.6, offset: 0,
        stops: [
          { x: -17, z: IN_TOP, pause: 5 },
          { x: -17, z: B },
          { x: 0, z: B },
          { x: 0, z: IN_TOP, pause: 4.5 },
          { x: 0, z: B },
          { x: -17, z: B },
        ],
      },
      { // QA: hai line dãy dưới, đi ngược chiều
        shirt: 0x6b536d, role: 'supervisor', speed: 2.2, offset: 9,
        stops: [
          { x: 17, z: IN_BOT, pause: 4 },
          { x: 17, z: A },
          { x: -17, z: A },
          { x: -17, z: IN_BOT, pause: 5.5 },
          { x: -17, z: A },
          { x: 17, z: A },
        ],
      },
      { // công nhân: cắt ngang lối đi giữa hai dãy
        shirt: 0x2d6c9d, role: 'worker', speed: 3.0, offset: 4,
        stops: [
          { x: 17, z: IN_TOP, pause: 3.5 },
          { x: 17, z: B },
          { x: 8, z: A },
          { x: 0, z: IN_BOT, pause: 3 },
          { x: 0, z: A },
          { x: 12, z: B },
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


  addMachine(machine, x, z, selected) {
    const THREE = this.THREE;
    const style = STATUS[machine.state] || STATUS.unreachable;
    const dark = store.theme === 'dark';
    const isBottleLine = machine.name === 'PC-Auto-1';
    const isStopped = ['unreachable', 'offline'].includes(machine.state);
    /* Máy dừng thì KHÔNG có gì chuyển động — không có ngoại lệ cho line chai.
       Trước đây `|| isBottleLine` cho PC-Auto-1 chạy băng tải kể cả khi đã mất
       Tailnet, để minh hoạ luồng băng 1 → robot → băng 2. Nhưng đây là màn hình
       giám sát, không phải video giới thiệu: một dây chuyền đang chạy trên sơ
       đồ trong khi thẻ bên cạnh ghi "Off network" là hai câu trái ngược nhau về
       cùng một máy, và cái chuyển động bao giờ cũng thắng.

       Hàng hoá vẫn nằm nguyên trên băng — dây chuyền dừng thì sản phẩm còn đó,
       chỉ là không đi. */
    const animateFlow = !isStopped;
    const productProfile = LINE_PRODUCTS[machine.name] || LINE_PRODUCTS.Auto2;
    const group = new THREE.Group();
    if (machine.virtual) group.userData.focus = machine.building || 'main';
    else group.userData.nodeId = machine.node_id;
    // Tên máy đi kèm để thẻ ảnh khi rê chuột hỏi đúng máy — API ảnh khoá theo
    // TÊN, còn raycast thì chỉ biết node id.
    group.userData.machineName = machine.name;
    group.position.set(x, selected ? 0.18 : 0, z);

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
      /* PC-Auto-1: băng 1 đưa chai vào robot; robot chuyển sang băng 2 (băng
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
    const beacon = add(new THREE.CylinderGeometry(0.17, 0.17, 0.36, 18), new THREE.MeshStandardMaterial({ color: style.beacon, emissive: style.beacon, emissiveIntensity: 1.1 }), -4.55, 2.92, 1.42);
    beacon.castShadow = false;

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
    const label = textSprite(THREE, machine.name, subtitle, dark);
    label.position.set(0, 5.3, 0);
    group.add(label);
    this.machineRoot.add(group);
    this.pickables.push(group);
  }

  update({ machines, selected, onSelect }) {
    this.onSelect = onSelect;
    this.canvas.setAttribute('aria-label', store.t.floorTitle);
    this.refreshFloorTheme();
    disposeObject(this.machineRoot);
    this.pickables = [];
    this.productAnimators = [];
    this.robotAnimators = [];
    const realPlaced = machines.filter(machine => machine.floor);
    const realNames = new Set(realPlaced.map(machine => machine.name));
    const placed = [...realPlaced, ...OFFLINE_CARTON_LINES.filter(machine => !realNames.has(machine.name))];
    if (!placed.length) {
      this.draw();
      return;
    }
    /* Toạ độ config là nhãn logic của tier 1. Sơ đồ 3D dùng layout vận hành:
       mỗi line có vùng an toàn riêng và hai dãy bị ngăn bởi aisle trung tâm. */
    const layout = {
      Auto2:      { x: -17, z: -10 },
      M1:         { x:   0, z: -10 },
      M2:         { x:  17, z: -10 },
      LineTine:   { x: -17, z:  10 },
      'Auto 1':   { x:   0, z:  10 },
      'PC-Auto-1': { x:  17, z:  10 },
      'Tin 2':    { x:  55, z:   0 },
    };
    this.addWarehouse('RAW MATERIALS', -23, -17);
    this.addWarehouse('FINISHED GOODS', 23, 17);
    placed.forEach((machine, index) => {
      const pos = layout[machine.name] || { x: -18 + index * 9, z: 10 };
      if (machine.name === 'Tin 2') this.addTin2Annex(pos.x, pos.z);
      this.addZone(machine, pos.x, pos.z, index);
      this.addMachine(machine, pos.x, pos.z, machine.node_id === selected);
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

  draw() {
    this.aimCamera();
    this.renderer.render(this.scene, this.camera);
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
        this.canvas.style.cursor = hit ? 'pointer' : 'grab';
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
      }
    });
    this.canvas.addEventListener('pointerleave', () => this.hidePeek());
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
    const fallback = await import('./factory-map.js');
    const latest = pending.get(el)?.options || queued.options;
    pending.delete(el);
    fallback.render(el, latest);
  }
}

export function render(el, options) {
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

export function legendHTML() {
  const t = store.t;
  const dot = color => `<svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="${color}"/></svg>`;
  const tri = '<svg width="12" height="12"><polygon points="6,1 11,10 1,10" fill="none" stroke="#9a6a00" stroke-width="1.5"/></svg>';
  const chat = '<svg width="14" height="12" stroke="#98989b" stroke-width="1.3" fill="none"><rect x="1" y="1" width="12" height="8" rx="2"/><line x1="0" y1="11" x2="14" y2="0"/></svg>';
  const cross = '<svg width="12" height="12" stroke="#98989b" stroke-width="1.4"><line x1="2" y1="2" x2="10" y2="10"/><line x1="10" y1="2" x2="2" y2="10"/></svg>';
  return `<span class="k">${dot('#2f7d4f')} ${t.state.ok}</span>
    <span class="k">${tri} ${t.state.warn}</span>
    <span class="k">${chat} ${t.state.agent_down}</span>
    <span class="k">${cross} ${t.state.unreachable}</span>`;
}
