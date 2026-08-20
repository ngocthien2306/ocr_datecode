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

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.92;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.1, 140);
    this.scene.add(this.machineRoot);

    const hemi = new THREE.HemisphereLight(0xf6fbff, 0x8c98a5, 2.1);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 2.8);
    key.position.set(10, 18, 8);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -30;
    key.shadow.camera.right = 30;
    key.shadow.camera.top = 30;
    key.shadow.camera.bottom = -30;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x9dc7e9, 1.1);
    fill.position.set(-14, 7, -12);
    this.scene.add(fill);

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
      this.cameraTarget.lerpVectors(this.cameraMove.fromTarget, this.cameraMove.toTarget, ease);
      this.distance = this.cameraMove.fromDistance + (this.cameraMove.toDistance - this.cameraMove.fromDistance) * ease;
      if (progress === 1) this.cameraMove = null;
    }
    if (now - this.lastAnimatedAt >= 32) {
      const seconds = now / 1000;
      this.productAnimators.forEach(item => {
        if (item.update) { item.update(seconds); return; }
        const span = item.max - item.min;
        item.object.position.x = item.min + ((seconds * item.speed + item.phase) % span);
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
    const limb = (from, to, radius, material) => {
      const delta = new THREE.Vector3().subVectors(to, from);
      const part = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 1.08, delta.length(), 10), material);
      part.position.copy(from).add(to).multiplyScalar(0.5);
      part.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
      part.castShadow = true;
      person.add(part);
      return part;
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
    for (const side of [-1, 1]) {
      const hip = new THREE.Vector3(side * 0.1, 0.88, 0);
      const knee = new THREE.Vector3(side * 0.12, 0.47, side * 0.04);
      const ankle = new THREE.Vector3(side * 0.1, 0.16, -0.03);
      limb(hip, knee, 0.075, trousers);
      limb(knee, ankle, 0.065, trousers);
      add(new THREE.BoxGeometry(0.15, 0.09, 0.26), boot, side * 0.1, 0.11, -0.1);
      const shoulder = new THREE.Vector3(side * 0.22, 1.52, 0);
      const elbow = new THREE.Vector3(side * 0.33, 1.2, side * 0.08);
      const hand = new THREE.Vector3(side * 0.28, role === 'supervisor' ? 1.08 : 1.05, -0.18);
      limb(shoulder, elbow, 0.06, cloth);
      limb(elbow, hand, 0.052, cloth);
      add(new THREE.SphereGeometry(0.065, 10, 8), skin, hand.x, hand.y, hand.z);
    }
    if (role === 'supervisor') {
      const clipboard = add(new THREE.BoxGeometry(0.26, 0.36, 0.04), new THREE.MeshStandardMaterial({ color: 0xd96c42, roughness: 0.6 }), 0.25, 1.2, -0.23);
      clipboard.rotation.z = -0.2;
    }
    person.position.set(x, 0, z);
    person.rotation.y = facing;
    target.add(person);
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
    this.addPerson(zone, { x: -4.85, z: towardAisle * 2.25, shirt: workerColors[index % workerColors.length], facing: towardAisle > 0 ? 0 : Math.PI });
    zone.position.set(x, 0, z);
    this.machineRoot.add(zone);
  }

  addAisleStaff() {
    const staff = new this.THREE.Group();
    this.addPerson(staff, { x: -6.4, z: -0.3, role: 'supervisor', shirt: 0x424d66, facing: 0.2 });
    this.addPerson(staff, { x: 8.6, z: 0.45, role: 'supervisor', shirt: 0x6b536d, facing: -0.4 });
    this.machineRoot.add(staff);
  }

  addMachine(machine, x, z, selected) {
    const THREE = this.THREE;
    const style = STATUS[machine.state] || STATUS.unreachable;
    const dark = store.theme === 'dark';
    const isBottleLine = machine.name === 'PC-Auto-1';
    const isStopped = ['unreachable', 'offline'].includes(machine.state);
    // PC-Auto-1 là mô hình quy trình: vẫn phát hoạt ảnh để người xem hiểu luồng
    // băng 1 → robot → băng 2, kể cả khi máy thật đang mất Tailnet.
    const animateFlow = !isStopped || isBottleLine;
    const productProfile = LINE_PRODUCTS[machine.name] || LINE_PRODUCTS.Auto2;
    const group = new THREE.Group();
    if (machine.virtual) group.userData.focus = machine.building || 'main';
    else group.userData.nodeId = machine.node_id;
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
    this.canvas.addEventListener('pointerdown', event => {
      this.canvas.setPointerCapture(event.pointerId);
      this.cameraMove = null;
      this.drag = { x: event.clientX, y: event.clientY, moved: false };
      this.canvas.classList.add('is-dragging');
    });
    this.canvas.addEventListener('pointermove', event => {
      if (this.drag) {
        const dx = event.clientX - this.drag.x;
        const dy = event.clientY - this.drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) this.drag.moved = true;
        this.azimuth -= dx * 0.009;
        this.elevation = Math.max(0.28, Math.min(1.2, this.elevation + dy * 0.008));
        this.drag.x = event.clientX;
        this.drag.y = event.clientY;
        this.draw();
      } else {
        this.canvas.style.cursor = this.targetAt(event) ? 'pointer' : 'grab';
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
    this.canvas.addEventListener('pointercancel', () => {
      this.drag = null;
      this.canvas.classList.remove('is-dragging');
    });
    this.canvas.addEventListener('wheel', event => {
      event.preventDefault();
      this.distance = Math.max(28, Math.min(70, this.distance + event.deltaY * 0.018));
      this.draw();
    }, { passive: false });
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
