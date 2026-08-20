# Hướng dẫn implement — Fleet Console & mẫu báo cáo

Tài liệu này đi kèm bộ mockup trong project. Nó nói **cái gì có trong file nào**,
**dịch từng khối UI sang code thế nào**, và **thứ tự làm** để không dựng lại hai lần.

Nguồn thiết kế: `uploads/01`–`uploads/10`. Nguồn dữ liệu thật: repo
`ngocthien2306/ocr_datecode@develop` (xem `github.md`).

---

## 1. File trong project

| File | Là gì | Dùng để làm gì |
|---|---|---|
| `Fleet Console Home.dc.html` | Bảng mockup chính, 4 lượt (turn) trên một canvas | nguồn tham chiếu UI: bố cục, cỡ chữ, màu, chuỗi song ngữ |
| `factory-3d.html` | Sơ đồ nhà máy 3D bậc 2, three.js, chạy độc lập | tham chiếu cho `FactoryMap` bản 3D; xuất OBJ+MTL / GLB được |
| `three-d-stage.js` | Shell viewer 3D (renderer, đèn, OrbitControls, nút export) | đọc để biết cách set up scene; **không** bê nguyên vào app |
| `_ds/industry-.../styles.css` | Token + component của design system Industry | nguồn `var(--*)` duy nhất |
| `screenshots/factory-3d-c.png` | Ảnh khung mặc định của bản 3D | dán vào tài liệu / slide |
| `github.md` | Ghi nhận repo nguồn + screen map | để đồng bộ ngược khi repo đổi |

### Bảng mockup gồm những gì

| Id | Màn hình | Ghi chú |
|---|---|---|
| `1a` | Fleet home — first fold (sơ đồ đẳng cự + lưới thẻ máy) | không cuộn; đây là câu hỏi đầu ca |
| `1b` | `MachineCard` — **8 trạng thái** | ok · warn · agent_down · unreachable · offline · loading · partial · not_started |
| `1c` | Ô máy trên sơ đồ — 5 trạng thái hiển thị | phân biệt bằng cả hình lẫn màu |
| `1d` | Pill đầu trang + dòng phạm vi mẫu | gồm cả trạng thái "làm mới hụt" |
| `2a`–`2c` | Báo cáo **một máy** — trang 1, trang 2, bản Executive | A4 dọc, biểu đồ ảnh tĩnh |
| `2d`–`2f` | Báo cáo **toàn nhà máy** — tổng quan, vân tay lỗi, phụ lục | banner phạm vi có 2 biến thể |
| `3a` | Tab **Staff** — nhóm Máy → Bộ phận | chỉ xem |
| `3b` | Tab **Activity log** — thao tác người dùng | có bộ lọc `simulated` |
| `3c` | Tab **Activity log** — lỗi hệ thống + log tail + nút ủy quyền | hai phần, một tab |
| `4a` | Sơ đồ nhà máy 3D (bậc 2) | kèm bảng so sánh cái giá với bậc 1 |

Mỗi mockup có thẻ **SPEC NOTES** cạnh nó ghi quy tắc mà khối đó đang thi hành và
điều khoản trong tài liệu gốc. Đọc thẻ đó trước khi sửa layout — phần lớn là kết
luận sau lỗi đã xảy ra thật, không phải sở thích.

Tắt/bật ghi chú và chuỗi tiếng Việt bằng hai tweak: `annotations`, `viCopy`.

---

## 2. Trước khi viết UI: ba endpoint chặn đường

Không dựng được UI nếu thiếu. Đúng thứ tự GĐ 1 trong `uploads/07`.

| # | Endpoint cần thêm (ở `agent_service`, theo khuôn `/api/fleet/rollup`) | Mở khoá |
|---|---|---|
| 1 | `GET /api/fleet/staff` — trả đủ `employee_code`, `department`, `job_title`, `shift`, `production_line` | tab Staff (`3a`) |
| 2 | `GET /api/fleet/failure-images?days=&limit=&cause=` + `GET /api/fleet/failure-image/{id}?w=480` (**có thu nhỏ**) | lưới ảnh lỗi (`2b`), Line Station |
| 3 | `granularity=hour\|day\|week\|shift` cho `/api/fleet/rollup` và `/api/fleet/production` | bộ lọc Giờ/Ngày/Tuần, khối "Theo ca" |

Điểm 2 không phải tuỳ chọn: ảnh gốc 1–2 MB × 12 ảnh ≈ 20 MB ≈ ~10 phút qua đường
tới Jetson. Thu nhỏ 480px ≈ 60 KB/ảnh.

Không sửa response model của backend ở giai đoạn này — sẽ phải restart backend
trên 5 máy đang sản xuất.

---

## 3. Cây component

```
FleetConsole
├─ TopBar            pill trạng thái · đồng hồ · EN|VI · sáng/tối · Làm mới
├─ FirstFold
│  ├─ FactoryMap     {machines, selected, onSelect}      ← đổi bậc 1 ↔ 2 chỉ thay ruột
│  └─ MachineGrid → MachineCard × N
├─ StatsBlock        PeriodPicker + ChartOrTable + CoverageNote + ExportButton
├─ FailureBlock      FingerprintHeatmap → FailureGrid → DelegateButton
├─ Tabs
│  ├─ StaffTab       GroupBySwitch + FilterBar + StaffGroup → StaffCard
│  └─ LogTab
│     ├─ UserActions LogTable + SimulatedFilter + AskButton
│     └─ SystemErrors ErrorSummaryTable + LogTail + DelegateButton
├─ MachineDrawer     mở từ map hoặc card, ~520px, KHÔNG che sơ đồ
└─ ChatPanel         ContextChip + AskPicker + ToolStatusLabel + SuggestionChips
```

Nguyên tắc dùng chung, đã có trong `uploads/04`:

- **`PeriodPicker` chỉ một bản** (Giờ / Ngày / Tuần / Ca) dùng khắp nơi.
- **`CoverageNote` bắt buộc** dưới mọi số liệu tổng hợp. Không có nó thì bảng
  thiếu một máy trông vẫn bình thường.
- **`AskPicker` một component, hai lối vào**: từ chat và từ nút "Xuất báo cáo"
  (điền sẵn theo bộ lọc đang xem).

---

## 4. Hợp đồng của từng component chính

### `MachineCard(machine)`

Đọc trực tiếp từ `1b`. Mỗi trạng thái là một nhánh render, **không phải một cờ boolean**:

```ts
type MachineState =
  | 'ok'            // "đang chạy"
  | 'warn'          // "cần chú ý"  — vượt ngưỡng hoặc pass tụt
  | 'agent_down'    // "trợ lý tắt · máy vẫn chạy"   ← KHÔNG gộp vào máy chết
  | 'unreachable'   // "không với tới được từ 14:22"
  | 'offline'       // "ngoài mạng"
  | 'loading'
  | 'partial'       // phần cứng live, sản xuất cũ (hoặc ngược lại)
  | 'not_started';  // ca chưa bắt đầu
```

Quy tắc render, cứng:

| Việc | Cách làm |
|---|---|
| Không có số | `—`, tuyệt đối không `0` |
| Phần trăm | luôn kèm số tuyệt đối: `RAM 87% · 7.0/8 GB` |
| Delta | ghi `▼ 8,2 điểm`, không `8,2%` |
| Nền so sánh < 30 bản ghi | **không đưa delta** |
| Hai kỳ khác độ dài | đưa **per-day** |
| Ca chưa bắt đầu | "Ca chưa bắt đầu" + số ca trước để đối chiếu |
| Gọi hụt | giữ số cũ + dấu thời gian, **không xoá trắng** |
| Skeleton | giữ đúng chiều cao thẻ để lưới không nhảy |

Ngưỡng lấy theo giới hạn phần cứng thật: nhiệt `<80 / 80–92 / ≥92 °C`,
RAM và đĩa `<80 / 80–92 / ≥92 %`.

### `FactoryMap({machines, selected, onSelect})`

Hai bản, cùng interface:

- **Bậc 1 — SVG đẳng cự** (`1a`, ~15 KB). Mặc định. Dùng cho Line Station và
  tablet cũ. Khối vẽ bằng ba path (mặt trên + hai mặt bên), toạ độ từ
  `machines.json`.
- **Bậc 2 — three.js** (`factory-3d.html`, ~600 KB + WebGL, 4–6 ngày). Dùng cho
  Fleet Console trên màn hình lớn.

Bốn trạng thái trên sơ đồ phải phân biệt **bằng cả hình lẫn màu**:

| Trạng thái | Bậc 1 (SVG) | Bậc 2 (3D) |
|---|---|---|
| ok | khối màu line, viền mảnh, chấm xanh | đèn tháp xanh |
| warn | viền vàng + tam giác + chấm nhấp nháy chậm | đèn tháp vàng + **vòng cảnh báo trên sàn** |
| agent_down | khối mờ + icon chat gạch chéo | đèn tháp xám + dấu chảo xám trên tủ điện |
| unreachable | khối xám, gạch chéo, viền nét đứt | khối xám, bỏ vỏ máy |
| đang chọn | nâng lên + đổ bóng + viền đậm | camera focus + viền phát sáng |

Toạ độ máy thêm vào `config/machines.json`, khoá theo **Tailscale node id**
(4/5 máy trùng hostname `suntech-desktop`, không dùng hostname làm khoá):

```json
"nmupyJbod721CNTRL": {
  "label": "M2", "line": "Line 3", "model": "Jetson Orin Nano 8GB Super",
  "floor": { "x": 3, "y": 1, "rotation": 0, "zone": "Dãy A" }
}
```

### Chuyển bản 3D sang React

`factory-3d.html` viết bằng three.js thuần vì phải chạy độc lập. Trong app React,
dịch 1-1 sang `@react-three/fiber` + `@react-three/drei`:

| Trong `factory-3d.html` | Trong React |
|---|---|
| `new THREE.Group()` cho mỗi line | `<InspectionLine machine={m} />` |
| `stage.setObject(factory)` | `<Canvas><Factory machines={machines} /></Canvas>` |
| OrbitControls của stage | `<OrbitControls makeDefault />` từ drei |
| nhãn bằng `THREE.Sprite` + canvas texture | `<Html>` của drei → **nhãn là DOM**: song ngữ, chọn được chữ, dùng chung i18n |
| — | `onPointerDown={() => onSelect(m.key)}` trên group của máy |
| đèn + shadow trong `_boot()` | `<hemisphereLight/> <directionalLight castShadow/>` |

Vài điểm phải giữ khi port:

- Đơn vị **mét thật**, y-up, gốc ở tâm sàn — để đổi mặt bằng chỉ là đổi toạ độ.
- Vật liệu dùng lại (5–6 material cho cả cảnh), **không** tạo material mới trong
  vòng render.
- Nền dùng `var(--color-accent-900)`; lưới nền dùng `--color-accent` /
  `--color-accent-700`. Không thêm màu trang trí ngoài steel.
- Không tải model ngoài: toàn bộ hình học dựng bằng code, nặng nhất vẫn là thư viện.
- Bọc bằng `<Suspense>` + fallback là **bản đẳng cự bậc 1**, để máy không có WebGL
  vẫn dùng được.

### `StaffCard`

- Badge quyền **luôn có nhãn**: `Quyền: operator` / `Access: operator`. Badge
  "operator" trần đứng cạnh chức vụ "Kỹ thuật viên bảo trì" đọc như hai chức danh
  đá nhau — lỗi đã gặp thật.
- Khoá bản ghi theo **(máy, username)**. `admin`, `operator`, `supervisor` tồn tại
  trên cả 5 máy và là 5 tài khoản khác nhau. Muốn "một người trên nhiều máy" thì
  liên kết bằng `employee_code` (NV-xxxx).
- Giai đoạn này **chỉ xem**. Tạo / sửa / đổi mật khẩu / khoá là thao tác có tác
  dụng phụ, cần cổng xác nhận riêng + ghi audit.

### `LogTab`

Hai phần trong một tab, **không trộn một dòng thời gian**:

| | Thao tác người dùng | Lỗi hệ thống |
|---|---|---|
| Nguồn | `action_logs` (Mongo) | file log service từng máy |
| Tool | `get_audit_logs` | `summarize_log_errors`, `search_logs`, `read_log_tail` |
| Người đọc | quản trị, trưởng ca | kỹ thuật |

- Bộ lọc `simulated: true` — mặc định **ẩn** ở môi trường thật.
- Fleet **không** kéo nguyên file log về (một file trên Jetson từng 1,4 GB). Chỉ
  nhận tóm tắt do edge làm; muốn sâu hơn thì `read_log_tail` rồi ủy quyền.
- Mỗi hàng có nút "hỏi thêm" mang ngữ cảnh (user + máy + thời điểm) sang chat.

### `ChatPanel` + `AskPicker`

```python
async def generate_fleet_report(machines=None, period=None, format=None, **_ignored):
    missing = {}
    if not machines: missing["machines"] = {...}
    if not period:   missing["period"]   = {...}
    if not format:   missing["format"]   = {...}
    if missing:
        return {"ok": False, "ask_user": missing, ...}
```

Ba điều bắt buộc: mặc định là `None` (đặt giá trị hợp lệ là dạy mô hình khỏi hỏi);
chỉ khoá **còn thiếu** mới vào `missing`; luôn có `**_ignored`.

Giao diện render `ask_user` thành **nút bấm**, mỗi lựa chọn kèm `hint` giải thích
hệ quả ("PDF — để in / gửi mail"). Nhãn chờ ghi theo tool đang chạy
("đang hỏi M2…", "đang dựng báo cáo PDF…") vì hỏi agent mất 4–20s, có lúc 27s.

Ba thứ không để mô hình làm: viết đường dẫn tải file · tự chọn máy/kỳ/định dạng ·
nhắc lại số từ trí nhớ.

---

## 5. Nhịp làm mới

| Dữ liệu | Nhịp | Vì sao |
|---|---|---|
| Trạng thái + phần cứng | 30s | rẻ (1,3s), đổi liên tục |
| Sản xuất + vân tay lỗi | 5 phút | rollup lạnh mổ vài trăm document trên Jetson |
| Line Station | 15s | người vận hành đứng nhìn liên tục |
| Nhân sự | khi mở tab | gần như không đổi |

Dashboard **không đi qua LLM**. Chỉ câu hỏi mở mới gọi agent — hôm OpenAI hết
credit, cả 5 agent im tiếng cùng lúc mà dashboard vẫn phải sống.

---

## 6. Màu chuỗi dữ liệu

```js
const color = name => PALETTE[hash(name) % PALETTE.length];   // theo TÊN máy
```

Gán theo thứ tự trong danh sách thì bỏ một máy khỏi bộ lọc là mọi máy còn lại đổi
màu, và hai biểu đồ cạnh nhau không đọc chéo được. Quá 8 máy thì **không thêm màu**
— chuyển sang quy tắc đậm/mờ (chọn máy để tô đậm, còn lại xám mờ + dải min–max).

---

## 7. Song ngữ

- Mọi chuỗi gom vào **một bảng**, không rải trong hàm dựng component. Rải ra thì
  lần thêm ngôn ngữ sau sẽ sót, mà sót chỗ nào chỉ lộ khi có máy rơi vào trạng thái
  hiếm (`agent_down` chỉ hiện khi thật sự có máy hỏng).
- Đổi ngôn ngữ **vẽ lại từ dữ liệu đã tải**, không refetch — mỗi lần bấm EN/VI mà
  gọi lại API là 5 lượt ra Jetson qua đường chậm chỉ để đổi chữ.
- Số và giờ theo locale (`en-GB` / `vi-VN`).

Bảng chuỗi trạng thái, lấy từ mockup `1b`:

| key | EN | VI |
|---|---|---|
| `ok` | Running | đang chạy |
| `warn` | Needs attention | cần chú ý |
| `agent_down` | Assistant off · machine running | trợ lý tắt · máy vẫn chạy |
| `unreachable` | Unreachable since {t} | không với tới được từ {t} |
| `offline` | Off network | ngoài mạng |
| `not_started` | Shift not started | ca chưa bắt đầu |
| `partial` | Partial data | thiếu một phần |

---

## 8. Mẫu báo cáo (`2a`–`2f`)

- **Một bộ style, hai đường xuất.** CSS phải vào được cả `reportGenerator.ts`
  (panel Historical) lẫn `styles.py` (agent). Cùng một kỳ sản xuất mà ra hai bản
  khác nhau thì người đọc không biết tin bản nào.
- **Biểu đồ là ảnh tĩnh** trong cả HTML lẫn PDF của fleet: WeasyPrint không chạy
  JS → vẽ PNG bằng matplotlib theo theme rồi nhúng base64.
- **Tiêu đề + câu dẫn + biểu đồ chung một khối** `break-after: avoid`.
- Banner thiếu máy **ngay đầu trang**, không phải chú thích cuối.
- Không xếp hạng máy theo pass rate; in cố định dòng chú thích dưới bảng máy.
- Vân tay lỗi là **tỉ trọng của mẫu**, ghi cỡ mẫu từng hàng ("196 · lấy mẫu" /
  "126 · phủ hết kỳ").
- Theme `dark` chỉ để xem trên màn hình. Bản in mặc định `industrial`.

---

## 9. Thứ tự làm

| GĐ | Việc | Ngày | Mốc kiểm chứng |
|---|---|---|---|
| 1 | 3 endpoint ở mục 2 | 4,5 | 54 người / 5 máy có đủ field · lưới 12 ảnh tải < 3s · gộp ca đúng ca C |
| 2 | Trang chủ: `FactoryMap` + `MachineCard` + `MachineDrawer` | 5 | tắt agent một máy → hiện "trợ lý tắt · máy vẫn chạy", **không** "máy chết" |
| 3 | Thống kê + vân tay lỗi + lưới ảnh | 4,5 | người chưa dùng hệ thống nhìn bảng nhiệt nói được "M1 và M2 hỏng hai thứ khác nhau" |
| 4 | Staff + Activity log + tool agent | 6 | chat trả đúng "hôm nay ai đăng nhập vào M2?" |
| 5 | Line Station | 4 | rút mạng ra ngoài → vẫn chạy đầy đủ, chỉ chat báo tạm không dùng được |
| 6 | `AskPicker` + đánh bóng 5 trạng thái | 4 | "xuất PDF so sánh M1 M2 tuần này" xuất luôn, không hỏi lại |

**28 ngày công** cho toàn bộ phạm vi. GĐ 3 trước GĐ 4 vì vân tay lỗi là thứ khác
biệt nhất. GĐ 5 gần cuối vì Line Station dùng lại gần hết component GĐ 2–3.

Bản 3D bậc 2 là việc **thêm vào GĐ 2** (4–6 ngày), làm sau khi bậc 1 đã chạy —
không thay thế nó: Line Station vẫn cần bản đẳng cự.

---

## 10. Rà soát trước khi merge

- [ ] Mọi khối dữ liệu có đủ **5 trạng thái** (tải · có · rỗng-có-lý-do · một phần · lỗi)
- [ ] Không có `0` nào đang đứng thay cho `—`
- [ ] Mọi số tổng hợp có `CoverageNote`, và máy thiếu được **nêu tên**
- [ ] Delta ghi "điểm"; nền < 30 bản ghi thì không có delta
- [ ] Đơn vị ghi rõ **sản phẩm** hay **frame**
- [ ] Biểu đồ cắt bớt cột → ghi vào tiêu đề
- [ ] `agent_down` không bị gộp vào "máy chết" ở bất kỳ đâu
- [ ] Mọi bản ghi người dùng khoá theo (máy, username)
- [ ] Bộ lọc `simulated` có, và mặc định ẩn
- [ ] Chuỗi mới đã vào bảng i18n, có cả EN và VI
- [ ] Không hard-code hex / font / px mà token đã có (`var(--*)`)
- [ ] Không có nút gây tác dụng phụ nào lọt vào Line Station
