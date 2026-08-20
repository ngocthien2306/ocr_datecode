# 08 — Đề xuất thêm

Mỗi đề xuất dưới đây bắt nguồn từ **một con số đã đo được trên hệ thống thật**,
không phải từ danh sách tính năng chung chung. Cột "căn cứ" ghi rõ số đó.

---

## ★★★ Nhóm 1 — Đáng làm nhất

### P1. Phân tích ngưỡng — "tôi có đang vứt hàng tốt không?"

**Căn cứ:** mẫu 300 frame fail trên M2, 7 ngày:
```
ngưỡng template = 0.60
  cách ngưỡng < 0.05  →  20,9%   ← sát ngưỡng
  cách 0.05–0.15      →  54,9%
  cách > 0.15         →  24,1%
```

Một phần năm số hàng bị loại nằm cách ngưỡng chưa tới 0,05. M2 hôm khảo sát loại
203 sản phẩm ⇒ khoảng **43 sản phẩm/ngày có thể đang bị loại oan**, trên một máy.

**Làm gì:** màn hình phân bố khoảng cách tới ngưỡng theo máy × recipe, kèm mô
phỏng *"hạ ngưỡng 0,60 → 0,55: 53 frame chuyển fail→pass"*.

**Ràng buộc phải cứng trong code:** công cụ **không bao giờ tự đổi ngưỡng**, và
phải trình bày **cả hai chiều** — hạ ngưỡng giảm loại oan nhưng tăng nguy cơ lọt
hàng sai date code. Trong ngành thực phẩm, lọt hàng đắt hơn loại oan rất nhiều
(thu hồi, phạt, mất khách). Công cụ đưa số, người quyết.

Dữ liệu đã có sẵn: `similarity` và `threshold` lưu trong từng frame.

### P2. Hàng đợi phúc tra — đo độ chính xác của chính hệ thống

**Căn cứ:** `inference_results` **không có field nào lưu phán quyết của người**.
Nghĩa là hiện không ai biết "27% fail" là hàng xấu thật hay máy bắt sai.

**Làm gì:** mỗi ngày lấy mẫu N frame fail (ưu tiên nhóm sát ngưỡng), hiện ảnh +
`expected` vs đọc được + confidence, người bấm **đúng / sai**. Từ đó tính:

```
Độ chính xác trên M2:  loại đúng 78%  ·  loại oan 22%
```

**Cần:** thêm field `human_verdict` — thay đổi nhỏ nhất trong tài liệu này, nhưng
mở khoá thứ giá trị nhất: con số để nói chuyện với khách hàng và với kiểm toán.
Nó cũng biến P1 từ giả thuyết thành bằng chứng.

### P3. Cảnh báo chủ động (Zalo / Telegram)

**Căn cứ:** khảo sát tìm thấy **M2 đĩa 98% (còn 6,3 GB)** và **Auto2 có 12 cảnh
báo đang active mà không ai đọc**. Cả hai tồn tại nhiều tuần.

Mọi thứ trong hệ thống hiện tại là *hỏi mới biết*. Khách hàng cần biết **trước
khi mất cả ca**:
- fail tăng gấp 3 trong 20 phút so với chính máy đó
- `no_detection` vọt lên — camera/ánh sáng/trigger, không phải mặt hàng
- máy im >5 phút, hoặc agent chết sau reboot
- đĩa < 20 GB, RAM > 92%, nhiệt > 92°C

**Rẻ để làm** vì vòng poll đã có sẵn. Đây là thứ biến hệ thống từ *công cụ tra
cứu* thành *hệ thống giám sát* — và là khác biệt lớn nhất trong mắt người mua.

---

## ★★ Nhóm 2 — Đáng làm tiếp

### P4. Tương quan thay đổi ↔ kết quả
*"Recipe sửa lúc 14:00, pass rate tụt từ 14:05 — ai sửa, sửa gì?"*
Dữ liệu đã đủ: `get_audit_logs` + `get_recipe_load_history` + timeline inference.
Trả lời được câu khó nhất khi truy nguyên nhân: **thứ gì đã đổi**.

### P5. So sánh ca / kíp
*"Cùng máy, cùng recipe: ca A đạt 95%, ca C đạt 78%"* → vấn đề vận hành hoặc đào
tạo, không phải máy. Dữ liệu đã có: `_shift_expr()` + hồ sơ ca của 54 người.
Đây là phép so sánh **có nghĩa** vì cùng máy cùng mặt hàng — khác hẳn so giữa các
line.

### P6. Chỉ tiêu ngày toàn nhà máy
`get_target_progress` + `production_targets.json` đã có ở edge. Fleet gộp lại:
5 line so với kế hoạch ngày. Con số quản đốc nhìn mỗi ngày.

### P7. Gói bằng chứng audit (BRC / HACCP / FDA)
*"Chứng minh lô ngày 15/08 trên M2 đã kiểm đủ"* → xuất gói: thống kê + ảnh mẫu +
nhật ký thao tác trong kỳ. Ngành thực phẩm cần cái này để qua đánh giá.

---

## ★ Nhóm 3 — Về sau

| | Tính năng | Ghi chú |
|---|---|---|
| P8 | Bảo trì dự đoán | từ xu hướng `check_reject_timing` / `trigger` / `sensor`; cần tích luỹ lịch sử |
| P9 | Điều khiển từ xa (restart service) | `start_service`/`stop_service` đã có; **cần cổng xác nhận** vì có tác dụng phụ |
| P10 | SSE gộp tiến trình 5 máy | edge đã có SSE; fleet đi cùng đường |
| P11 | Lịch sử xu hướng dài hạn | fleet hiện không lưu; cần SQLite tích luỹ rollup |

---

## Ba điều nên **không** làm

**Đừng xếp hạng dây chuyền bằng tỉ lệ pass, dù ai xin.** Trên đội hình này đúng
một recipe được chia sẻ giữa hai máy. Bảng xếp hạng ấy sẽ được đem đi họp và
dùng để đánh giá con người, trong khi nó chỉ đo độ khó mặt hàng. Nếu buộc phải
có, dùng **độ lệch so với chính máy đó kỳ trước**.

**Đừng để agent tự đổi cấu hình máy.** Ngưỡng, recipe, tham số camera — agent đề
xuất được, nhưng nút bấm phải thuộc về người, và phải ghi audit.

**Đừng gộp ảnh sản phẩm lỗi vào báo cáo gửi kèm mail.** Ảnh gốc 1–2 MB; một báo
cáo 12 ảnh là 20 MB. Báo cáo đưa link, ảnh xem trên hệ thống.

---

## Đề xuất về vận hành, ngoài phần mềm

Ba việc phát hiện trong lúc khảo sát, không thuộc phạm vi UI nhưng ảnh hưởng
trực tiếp tới độ tin cậy của hệ thống:

| | Vấn đề | Hệ quả nếu bỏ qua |
|---|---|---|
| 1 | **PC-Auto-1 chạy 71–100°C** ở tải 1–3% | CPU hạ xung hoặc tắt máy; đây là hỏng tản nhiệt, phần mềm chỉ làm nó hiện ra |
| 2 | **Agent chạy uvicorn trần**, không có systemd | reboot là mất trợ lý trên máy đó, và không ai biết |
| 3 | **Đĩa tích luỹ ~10–17 GB/tháng/máy** | đã dọn 139 GB một lần; không có cron thì 2–3 tháng nữa lặp lại |
