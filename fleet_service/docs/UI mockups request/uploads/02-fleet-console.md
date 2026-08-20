# 02 — Fleet Console (máy tổng)

## Bố cục trang chủ

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Fleet — OCR Datecode      ● 4 tốt  ● 1 cần chú ý     ☀☾  EN|VI   Làm mới │
├───────────────────────┬──────────────────────────────────────────────────┤
│                       │  ┌────────┐┌────────┐┌────────┐┌────────┐        │
│   SƠ ĐỒ NHÀ MÁY 3D    │  │ Auto2  ││  M1    ││  M2    ││LineTine│        │
│                       │  │ 98,4%  ││ 77,8%  ││ 69,4%  ││ 81,5%  │        │
│   ┌──┐  ┌──┐  ┌──┐    │  └────────┘└────────┘└────────┘└────────┘        │
│   │L1│  │L2│  │L3│    │  ┌────────┐                                      │
│   └──┘  └──┘  └──┘    │  │PC-Auto1│    ← lưới thẻ máy, click mở ngăn kéo │
│      ┌──┐  ┌──┐       │  └────────┘                                      │
│      │L4│  │L5│       │                                                  │
│      └──┘  └──┘       │                                                  │
│   [xoay] [đặt lại]    │                                                  │
├───────────────────────┴──────────────────────────────────────────────────┤
│  THỐNG KÊ TOÀN NHÀ MÁY        [Giờ ▾][Ngày ▾][Tuần ▾]   [Biểu đồ|Bảng]   │
│  ...                                                                     │
├──────────────────────────────────────────────────────────────────────────┤
│  ĐIỀU TRA NGUYÊN NHÂN LỖI     top N ảnh · mở rộng theo máy               │
├──────────────────────────────────────────────────────────────────────────┤
│  [Tổng quan] [Nhân sự] [Nhật ký thao tác]        ← tab                   │
└──────────────────────────────────────────────────────────────────────────┘
                                              ┌──────────────────┐
                                              │ Trợ lý đội hình  │ ← luôn hiện
                                              └──────────────────┘
```

Sơ đồ 3D và lưới thẻ máy **luôn cùng màn hình đầu tiên**, không cuộn. Đó là câu
hỏi đầu ca: "line nào cần tôi tới". Mọi thứ khác nằm dưới.

---

## 1. Sơ đồ nhà máy 3D

### Bố trí đề xuất (chưa có mặt bằng thật)

Nhà xưởng chữ nhật, hai dãy dây chuyền song song, lối đi ở giữa:

```
        ┌─────────────────── XƯỞNG ĐÓNG GÓI GIA VỊ ───────────────────┐
        │                                                             │
        │   [Line 1]        [Line 2]        [Line 3]                  │
        │    Auto2            M1              M2                      │
        │   ONION POWDER    CHILI Pdr      ORG CINNAMON               │
        │                                                             │
        │  ══════════════ lối đi ══════════════                       │
        │                                                             │
        │        [Tine Line]         [Auto Line]                      │
        │         LineTine            PC-Auto-1                       │
        │        PURE SEA SALT        Minced Onion                    │
        │                                                             │
        │  ┌────────┐                              ┌──────────┐       │
        │  │ Kho NL │                              │ Kho TP   │       │
        │  └────────┘                              └──────────┘       │
        └─────────────────────────────────────────────────────────────┘
```

Vị trí lưu trong `config/machines.json` (`x`, `y`, `rotation`) để đổi mặt bằng
không phải sửa code.

### Chọn công nghệ — hai bậc

| | **Bậc 1: đẳng cự SVG/CSS** ✅ khuyến nghị | **Bậc 2: Three.js** |
|---|---|---|
| Hình khối | hộp đẳng cự vẽ sẵn | mesh 3D thật, xoay tự do |
| Kích thước | ~15 KB | ~600 KB (three.js) |
| Chạy trên | mọi thiết bị, kể cả tablet cũ | cần WebGL |
| Thời gian dựng | 1–2 ngày | 4–6 ngày |
| Khi nào cần | 5–20 máy, một tầng | nhiều tầng, cần xoay/zoom thật |

**Bắt đầu bằng bậc 1.** Với 5 máy trên một mặt sàn, góc nhìn đẳng cự cố định
truyền đạt đủ vị trí, mà không kéo theo WebGL lên một tablet đặt cạnh dây chuyền.
Giữ interface `FactoryMap({machines, onSelect})` để đổi sang bậc 2 chỉ là thay
một component.

### Trạng thái hiển thị trên sơ đồ

| Trạng thái | Thể hiện |
|---|---|
| Đang chạy tốt | khối màu line, viền mảnh |
| Cần chú ý (nóng / đĩa đầy / pass tụt) | viền vàng, chấm nhấp nháy chậm |
| Agent tắt, **máy vẫn chạy** | khối mờ đi, biểu tượng "trợ lý tắt" |
| Mất liên lạc | khối xám, gạch chéo |
| Đang chọn | nâng lên, đổ bóng, viền đậm |

**Bốn trạng thái này phải phân biệt được bằng cả hình dạng lẫn màu**, không chỉ
màu — xưởng có người mù màu, và màn hình cạnh dây chuyền thường bị chói.

### Tương tác
- Click khối → mở ngăn kéo chi tiết máy (mục 2)
- Hover → tooltip: tên line, recipe đang chạy, pass rate ca hiện tại
- Không có máy nào được chọn → sơ đồ ở trạng thái tổng quan

---

## 2. Ngăn kéo chi tiết một máy

Mở từ sơ đồ hoặc từ thẻ máy. Trượt vào từ phải, chiếm ~520px, **không che sơ đồ**
để người dùng còn thấy mình đang xem máy nào.

```
┌─ M2 · Line 3 ─────────────────────────────── ✕ ─┐
│ Jetson Orin Nano 8GB Super · 100.112.46.5       │
│ ● đang chạy · ORG CINNAMON CC                   │
├─────────────────────────────────────────────────┤
│ [Giờ] [Ngày] [Tuần]              ← bộ lọc        │
│                                                 │
│  Sản lượng ca này        1.284 sp               │
│  Tỉ lệ đạt               69,4%   ▼ 8,2 điểm     │
│  Chỉ tiêu ca             1.284 / 1.800  (71%)   │
│  ▁▂▃▅▇▆▄▃▂  ← biểu đồ pass/fail theo giờ        │
│                                                 │
│  PHẦN CỨNG                                      │
│  CPU 56°C · GPU 56°C · RAM 87% · Đĩa 74%        │
│                                                 │
│  NGUYÊN NHÂN LỖI (vân tay)                      │
│  ████████████░░░ ký tự dưới ngưỡng      56,5%   │
│  ███░░░░░░░░░░░░ OCR đọc sai chuỗi      18,7%   │
│  ...                                            │
│                                                 │
│  [Hỏi trợ lý về máy này]  [Mở giao diện line]   │
└─────────────────────────────────────────────────┘
```

Nút **"Hỏi trợ lý về máy này"** mở chat với ngữ cảnh đã gắn sẵn máy — người dùng
không phải gõ lại tên máy, và tránh luôn chuyện gõ nhầm tên.

**Bộ lọc Giờ/Ngày/Tuần** đổi cả KPI lẫn biểu đồ. Khi chọn "Giờ", trục là 24 giờ
của hôm nay; "Ngày" là 7 ngày gần nhất; "Tuần" là 8 tuần gần nhất.

---

## 3. Thống kê toàn nhà máy

Ngay dưới lưới máy. Người dùng chọn **cách xem** (biểu đồ / bảng) và **đơn vị thời
gian** (giờ / ngày / tuần) qua dropdown.

### Dạng biểu đồ
- Cột chồng pass/fail theo đơn vị thời gian đã chọn, một nhóm cột cho mỗi máy
- Hoặc đường tỉ lệ pass theo thời gian, mỗi máy một đường, màu cố định theo tên máy

**Màu gán theo TÊN máy, không theo thứ tự trong danh sách.** Gán theo thứ tự thì
bỏ một máy khỏi bộ lọc là mọi máy còn lại đổi màu, và hai biểu đồ cạnh nhau không
đọc chéo được. (Quy tắc này đã áp dụng trong bộ dựng báo cáo.)

### Dạng bảng
Cột: Máy · Line · Sản lượng · Mỗi ngày · Đạt · Không đạt · Tỉ lệ đạt · Recipe.
Sắp xếp được theo mọi cột. Hàng tổng ở cuối.

### Bắt buộc, không được bỏ
Dưới cả hai dạng luôn có **dòng phạm vi mẫu**:
> Đủ cả 5 máy. — hoặc — **Thiếu LineTine** (agent tắt lúc 14:22)

Và với dạng bảng, cột "Tỉ lệ đạt" phải có chú thích: *các máy chạy recipe khác
nhau, không so trực tiếp được.*

---

## 4. Điều tra nguyên nhân lỗi

Phần này là của QA và kỹ thuật. Ba tầng, từ tổng tới cụ thể.

### Tầng 1 — Vân tay kiểu lỗi toàn nhà máy
Bảng nhiệt: hàng là máy, cột là nguyên nhân, ô là tỉ trọng. Ô lệch mạnh khỏi
trung vị được tô đậm hơn. Đây là thứ chỉ ra "M1 lỗi kiểu khác M2" mà xếp hạng
pass rate không nói ra được.

### Tầng 2 — Top N ảnh sản phẩm lỗi
Lưới ảnh, mặc định N = 12, chọn được 6/12/24.

```
┌────────┐┌────────┐┌────────┐┌────────┐
│ [ảnh]  ││ [ảnh]  ││ [ảnh]  ││ [ảnh]  │
│ M2·14h ││ M1·09h ││ M2·15h ││LT·11h  │
│ mong:  ││ mong:  ││ mong:  ││ mong:  │
│ BB/2609││ BB/2609││ ...    ││ ...    │
│ đọc:   ││ đọc:   ││        ││        │
│ BB/2G09││ (trống)││        ││        │
└────────┘└────────┘└────────┘└────────┘
```

Mỗi ảnh kèm: máy, giờ, `expected` → `recognized`, nguyên nhân. **Ảnh có
`expected` được ưu tiên hiển thị trước** — ảnh kèm dòng "mong X → đọc Y" tự giải
thích, còn ảnh không có thì người xem chỉ thấy một khung hình và phải tự đoán.
(Logic này đã có sẵn trong `_pick_samples` ở edge, tái dùng chứ đừng viết lại.)

Ảnh được **rải đều qua các nguyên nhân**, không dồn hết vào loại phổ biến nhất —
cũng là logic sẵn có.

### Tầng 3 — Mở rộng theo máy
Click một máy trong bảng nhiệt → chỉ hiện ảnh của máy đó, và một nút "hỏi agent
của máy này phân tích sâu". Đây là lúc đường ủy quyền được dùng đúng chỗ.

> ⚠️ **Khoảng trống kỹ thuật:** hiện **chưa có endpoint nào phục vụ ảnh sản phẩm
> lỗi**. Backend chỉ phục vụ avatar, template và ảnh ML. Xem
> [06 — Data Contracts](06-data-contracts.md).

---

## 5. Tab Nhân sự

54 tài khoản trên 5 máy. Nhóm theo **máy → bộ phận**, đổi được sang nhóm theo
**bộ phận → máy** hoặc **ca → máy**.

```
▼ M2 · Line 3                                    9 người
   ▼ Sản xuất                                    5 người
     [ảnh] Tạ Minh Khoa      NV-3145  CN vận hành  Ca C  ● đang trong ca
     [ảnh] Phan Thị Thuỳ Dương NV-3120 CN vận hành Ca B
   ▼ QA/QC                                       2 người
     ...
▶ M1 · Line 2                                    9 người
```

Mỗi thẻ người: ảnh, tên, mã NV, chức vụ, ca, **quyền hệ thống có nhãn rõ ràng**
("Quyền: operator"), và khoảng hoạt động hôm nay.

> Nhãn quyền phải ghi rõ chữ "Quyền:". Không có nhãn thì badge "operator" đứng
> cạnh chức vụ "Kỹ thuật viên bảo trì" đọc như hai chức danh đá nhau — lỗi đã
> gặp thật và đã sửa ở giao diện `/test`.

Bộ lọc: theo máy, bộ phận, ca, vai trò, trạng thái (đang trong ca / không).

**Phạm vi giai đoạn này: chỉ XEM.** "Quản lý user" đầy đủ (tạo / sửa / đổi mật
khẩu / khoá trên nhiều máy) là thao tác **có tác dụng phụ** — nó cần cổng xác
nhận riêng, ghi audit, và trả lời câu "tạo trên máy nào / mọi máy?". Đưa vào cùng
đợt với phần xem là trộn một tính năng an toàn với một tính năng nguy hiểm. Ghi
nhận là giai đoạn kế tiếp, thiết kế cổng xác nhận trước khi làm.

> ⚠️ **Khoảng trống kỹ thuật:** API `/api/users/` của backend **không trả về**
> `department`, `production_line`, `shift`, `employee_code`, `job_title` — dù
> MongoDB có đủ. Không có chúng thì mọi cách nhóm ở trên đều không dựng được.

---

## 6. Tab Nhật ký — hai phần, đừng gộp

Yêu cầu gốc là quản lý **cả log hệ thống lẫn log thao tác**. Hai thứ này khác
nhau về nguồn, người đọc và câu hỏi, nên là hai phần trong một tab chứ không
trộn chung một dòng thời gian:

| | Thao tác người dùng | Lỗi hệ thống |
|---|---|---|
| Nguồn | `action_logs` (Mongo) | file log của service trên từng máy |
| Ai đọc | quản trị, trưởng ca | kỹ thuật |
| Câu hỏi | "ai làm gì" | "máy nào đang kêu gì" |
| Tool sẵn có | `get_audit_logs` | `summarize_log_errors`, `search_logs`, `read_log_tail` |

### 6a. Thao tác người dùng

Ai làm gì, trên máy nào, lúc nào. Dòng thời gian gộp từ 5 máy.

```
[Tất cả máy ▾] [Tất cả thao tác ▾] [Hôm nay ▾]        🔍 tìm

14:32  M2   truongca_m2   load_recipe    ORG CINNAMON CC
14:07  M1   nv_dungmay1   login
13:58  Auto2 admin        update_user    qa_operator2
...
```

Loại thao tác lấy từ dữ liệu thật: `login`, `logout`, `create_user`,
`update_user`, `delete_user`, `reset_password`, `load_recipe`, `stop_recipe`.

**Tích hợp chat:** mỗi hàng có nút "hỏi thêm" đưa ngữ cảnh sang trợ lý. Và ngược
lại, câu hỏi kiểu *"hôm nay ai sửa recipe trên M2?"* trả lời được ngay trong chat
mà không cần mở tab này.

> ⚠️ Bản ghi giả lập mang cờ `simulated: true`. Giao diện phải **có bộ lọc tách
> chúng ra** — nếu không, số liệu demo trộn vào số liệu thật và không ai phân
> biệt được.

### 6b. Lỗi hệ thống

Fan-out `summarize_log_errors` qua các máy, gộp thành bảng: máy · nhóm lỗi · số
lần · lần cuối. Click một hàng → `read_log_tail` của đúng máy đó, và nút "hỏi
agent máy này phân tích" (ủy quyền, đường đắt).

Không kéo nguyên file log về fleet: file log trên Jetson từng phình tới 1,4 GB —
đọc trọn một file như vậy là treo agent và ăn hết RAM (đã ghi ở `log_tools.py`).
Fleet chỉ nhận bản tóm tắt do edge làm; muốn sâu hơn thì ủy quyền.

## 7. Xuất báo cáo — không chỉ từ chat

Khối thống kê (mục 3) có nút **"Xuất báo cáo"** mở đúng cái picker của chat
(chọn máy / kỳ / định dạng) với **giá trị điền sẵn theo bộ lọc đang xem** — người
dùng đang nhìn 30 ngày dạng bảng thì picker mở ra đã chọn sẵn 30 ngày. Cùng một
component `AskPicker`, hai lối vào.

Cạnh nút là **danh sách báo cáo đã sinh** (tên, máy, kỳ, định dạng, lúc nào) để
tải lại mà không phải xuất lần nữa — file đã nằm sẵn trong `generated_reports/`,
chỉ thiếu chỗ liệt kê.
