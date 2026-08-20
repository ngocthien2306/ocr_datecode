# 07 — Lộ trình

Sáu giai đoạn. Mỗi giai đoạn có **mốc kiểm chứng đo được** — không phải "xong
màn hình X" mà "câu hỏi Y trả lời đúng trên dữ liệu thật".

Ước lượng theo ngày công một người, đã tính cả phần kiểm chứng.

---

## GĐ 1 — Gỡ ba khoảng trống dữ liệu (4,5 ngày)

Làm trước tiên vì ba tính năng trong bản mô tả **không dựng được** nếu thiếu.
Toàn bộ là endpoint ở `agent_service`, theo khuôn `/api/fleet/rollup` đã có —
không đụng backend, không restart backend trên máy sản xuất.

| # | Việc | Ngày |
|---|---|---|
| 1.1 | `GET /api/fleet/staff` — trả đủ field hồ sơ đang bị API `/api/users/` bỏ rơi | 1,5 |
| 1.2 | `GET /api/fleet/failure-images` + endpoint phục vụ ảnh **có thu nhỏ** | 2,0 |
| 1.3 | Thêm `granularity=hour\|day\|week\|shift` vào rollup | 1,0 |

**Kiểm chứng:**
- `/api/fleet/staff` trả về 54 người trên 5 máy, có đủ `department`, `shift`, `production_line`
- Lưới 12 ảnh lỗi tải xong **dưới 3 giây** qua đường tới Jetson (ảnh gốc sẽ mất ~10 phút)
- Gộp theo ca trả đúng ca C (22:00–06:00), không mất ca đêm

---

## GĐ 2 — Trang chủ Fleet Console (5 ngày)

| # | Việc | Ngày |
|---|---|---|
| 2.1 | `FactoryMap` đẳng cự + toạ độ trong `machines.json` | 1,5 |
| 2.2 | Lưới thẻ máy (đã có, chuyển sang component) | 0,5 |
| 2.3 | `MachineDrawer` + `PeriodPicker` giờ/ngày/tuần + biểu đồ | 2,0 |
| 2.4 | 4 trạng thái máy trên sơ đồ, phân biệt bằng cả hình lẫn màu | 1,0 |

**Kiểm chứng:** tắt agent một máy → sơ đồ hiện đúng "trợ lý tắt · máy vẫn chạy",
**không** hiện "máy chết"; số liệu phần cứng của máy đó vẫn về.

---

## GĐ 3 — Thống kê + Điều tra lỗi (4,5 ngày)

| # | Việc | Ngày |
|---|---|---|
| 3.1 | Khối thống kê toàn nhà máy, chuyển biểu đồ ↔ bảng, dropdown giờ/ngày/tuần | 1,5 |
| 3.2 | Bảng nhiệt vân tay kiểu lỗi | 1,0 |
| 3.3 | Lưới top N ảnh lỗi, kèm `mong → đọc` | 1,5 |
| 3.4 | Mở rộng theo máy → nút ủy quyền cho agent máy đó | 0,5 |

**Kiểm chứng:** người chưa từng dùng hệ thống nhìn bảng nhiệt và nói được
"M1 và M2 hỏng hai thứ khác nhau" mà không cần giải thích thêm.

---

## GĐ 4 — Nhân sự + Nhật ký (thao tác & lỗi hệ thống) + agent (6 ngày)

| # | Việc | Ngày |
|---|---|---|
| 4.1 | Tab Nhân sự, nhóm theo máy → bộ phận, đổi được cách nhóm | 1,5 |
| 4.2 | Tab Nhật ký thao tác, gộp 5 máy, **lọc `simulated`** | 1,0 |
| 4.3 | Tool `fleet_user_management` cho agent | 1,0 |
| 4.4 | Tool `fleet_audit_log` cho agent | 1,0 |
| 4.5 | Nút "hỏi thêm" từ mỗi hàng nhật ký sang chat, kèm ngữ cảnh | 0,5 |
| 4.6 | Phần "Lỗi hệ thống": fan-out `summarize_log_errors`, click xem tail, nút ủy quyền | 1,0 |

**Ràng buộc:** mọi bản ghi người dùng khoá theo **(máy, username)** — username
trùng trên cả 5 máy (`admin` ×5), gộp theo username trần là trộn 5 người thành một.

**Kiểm chứng:** chat trả lời đúng ba câu:
- "Hôm nay ai đăng nhập vào M2?"
- "Ai sửa recipe trên Auto2 tuần này?"
- "Bộ phận QA có bao nhiêu người, ở những line nào?"

---

## GĐ 5 — Line Station (4 ngày)

| # | Việc | Ngày |
|---|---|---|
| 5.1 | Bố cục đọc-từ-2-mét, tự làm mới 15s | 1,5 |
| 5.2 | Sản lượng ca + chỉ tiêu, xử lý **ca C vắt nửa đêm** | 1,0 |
| 5.3 | Sản phẩm lỗi gần đây (ảnh thu nhỏ) | 0,5 |
| 5.4 | Người trong ca + nút bàn giao ca | 0,5 |
| 5.5 | Chat gọn, khoá ngữ cảnh vào máy này | 0,5 |

**Kiểm chứng:** rút mạng ra ngoài → Line Station **vẫn chạy đầy đủ**; chỉ phần
chat báo tạm không dùng được.

---

## GĐ 6 — Picker cho agent + đánh bóng (4 ngày)

| # | Việc | Ngày |
|---|---|---|
| 6.1 | `AskPicker`: render `ask_user` thành nút bấm, có `hint` | 1,5 |
| 6.2 | Chip ngữ cảnh gắn sẵn ("đang hỏi về M2") | 0,5 |
| 6.3 | Nhãn trạng thái chờ theo tool đang chạy | 0,5 |
| 6.4 | Rà lại 5 trạng thái của mọi khối dữ liệu | 0,5 |
| 6.5 | Nút "Xuất báo cáo" từ khối thống kê (điền sẵn theo bộ lọc) + danh sách báo cáo đã sinh | 1,0 |

**Kiểm chứng:** "xuất PDF so sánh M1 M2 tuần này" xuất luôn **không hỏi lại**;
"xuất báo cáo" hiện picker đủ 3 nhóm; "xuất báo cáo PDF" chỉ hỏi 2 nhóm còn thiếu.

---

## Tổng

**28 ngày công** cho toàn bộ phạm vi đã mô tả.

| Nếu chỉ có… | Làm gì | Được gì |
|---|---|---|
| 1 tuần | GĐ 1 | Gỡ chặn, chưa thấy gì trên màn hình |
| 2 tuần | GĐ 1 + 2 | Trang chủ có sơ đồ và ngăn kéo máy |
| 3 tuần | + GĐ 3 | Đủ phần gây ấn tượng nhất (vân tay lỗi + ảnh) |
| 4 tuần | + GĐ 4 | Đủ phần quản lý |
| 5–6 tuần | + GĐ 5, 6 | Toàn bộ |

## Thứ tự này có lý do

**GĐ 1 trước tiên** vì ba tính năng phụ thuộc vào nó, và phát hiện muộn thì phải
làm lại giao diện. Riêng ảnh lỗi, nếu bỏ qua bước thu nhỏ thì tới lúc thử trên
xưởng mới biết lưới ảnh mất 10 phút để tải.

**GĐ 3 trước GĐ 4** vì vân tay kiểu lỗi là thứ khác biệt nhất của hệ thống này —
nó trả lời được câu mà xếp hạng pass rate không trả lời được.

**GĐ 5 gần cuối** vì Line Station dùng lại gần hết component của GĐ 2–3. Làm
trước là dựng hai lần.

## Việc còn treo, ngoài lộ trình

| | Ghi chú |
|---|---|
| Cron dọn đĩa hàng tuần | ~10–17 GB/tháng/máy; 2–3 tháng nữa M2 lại đầy |
| Cảnh báo chủ động (Zalo/Telegram) | dùng chung vòng poll đã có, xem [08](08-proposals.md) |
| PC-Auto-1 71–100°C | vấn đề vật lý: tản nhiệt / quạt / bụi |
| Agent chạy uvicorn trần | không sống qua reboot; nên có systemd unit |
