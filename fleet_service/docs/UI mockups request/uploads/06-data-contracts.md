# 06 — Data Contracts: cái gì đã có, cái gì còn thiếu

Đọc mục này **trước khi ước lượng công việc**. Ba tính năng trong bản mô tả hiện
chưa có dữ liệu để dựng, và đó là phần tốn thời gian nhất chứ không phải giao diện.

## Đã có, dùng được ngay

### Fleet service (`:8200`)
| Endpoint | Trả về | Đo được |
|---|---|---|
| `GET /api/fleet/machines` | danh sách máy, bậc năng lực, trạng thái | tức thì |
| `GET /api/fleet/status` | phần cứng + service 5 máy | 1,3s |
| `GET /api/fleet/production?days=` | sản lượng + vân tay lỗi 5 máy | 1,6s |
| `GET /api/fleet/machine/{key}` | chi tiết một máy | ~0,5s |
| `GET /api/fleet/ask/{key}?q=` | ủy quyền câu hỏi | 4–20s |
| `POST /api/fleet/chat` | trợ lý đội hình, 8 tool | 5–36s |
| `GET /api/fleet/report/{name}` | tải file báo cáo | tức thì |

### Agent service trên mỗi máy (`:8100`)
| Endpoint | Ghi chú |
|---|---|
| `GET /api/agent/health` | **không cần token** — dùng cho liveness |
| `GET /api/fleet/rollup?days=&causes=` | 1,4 KB · 2,0s lạnh / 0,2s cache |
| `POST /api/agent/chat` | 5 agent, có `kpis`/`charts`/`tables` dựng bằng code |
| `GET /api/agent/service/status` | trạng thái camera service, không LLM |

### Backend trên mỗi máy (`:8000`)
| Endpoint | Ghi chú |
|---|---|
| `GET /api/jetson-monitoring/metrics` | CPU/GPU/RAM/đĩa/điện, **không cần token** |
| `GET /api/jetson-monitoring/alerts` | cảnh báo vượt ngưỡng |
| `GET /api/users/` | ⚠️ **thiếu field hồ sơ** — xem bên dưới |

### Tool sẵn có ở edge, tái dùng được
`get_pass_fail_stats` · `explain_failures` · `compare_periods` · `get_downtime` ·
`get_shift_handover` · `get_target_progress` · `get_audit_logs` · `search_logs` ·
`summarize_log_errors` · `check_reject_timing` · `check_trigger_health` ·
`check_sensor_pulse` · `check_subsystem_health` · `get_system_metrics`

Định nghĩa ca đã có: **Ca A 06–14 · Ca B 14–22 · Ca C 22–06** (ca C vắt qua nửa
đêm; biểu thức Mongo đã xử lý bằng `$switch`).

---

## ❌ Thiếu 1 — Field hồ sơ nhân sự không ra khỏi API

**Đã kiểm chứng trên M2:**

```
MongoDB có:  employee_code, department, job_title, shift,
             production_line, hire_date, avatar_url, role, …
API trả về:  _id, username, email, full_name, phone_number,
             role, is_active, avatar_url, last_login, created_at, updated_at
```

Sáu field hồ sơ bị rơi. Nguyên nhân: model response của backend map field tường
minh, không tự serialize — cùng lớp lỗi đã ghi cho `recipe_to_response`.

**Chặn tính năng:** toàn bộ tab Nhân sự (group theo line / bộ phận / ca).

**Cách xử lý, xếp theo mức xâm lấn tăng dần:**

| | Cách | Đánh giá |
|---|---|---|
| ✅ | Thêm `GET /api/fleet/staff` vào **agent_service**, đọc thẳng các field hồ sơ | Đúng khuôn `/api/fleet/rollup` đã có. Không đụng backend, không restart backend trên 5 máy sản xuất |
| | Sửa response model của backend | Đúng về lâu dài nhưng phải restart backend 5 máy đang chạy |
| ❌ | Fleet đọc thẳng MongoDB | Phá nguyên tắc kiến trúc, và bỏ mất tri thức đúng/sai trong tool |

---

## ❌ Thiếu 2 — Không có endpoint phục vụ ảnh sản phẩm lỗi

Backend phục vụ `avatars/`, `templates/images/`, `templates/visualizations/`,
ảnh ML — **nhưng không có gì phục vụ `uploads/inference_results/`**.

`explain_failures` trả về `image_path` trỏ tới file trên đĩa Jetson, nhưng giao
diện không có đường lấy file đó.

**Chặn tính năng:** "top N ảnh minh hoạ" ở Fleet Console, và "sản phẩm lỗi gần
đây" ở Line Station — tức hai trong số các tính năng bạn nêu.

**Cần làm:**
1. `GET /api/fleet/failure-images?days=&limit=&cause=` ở agent_service — trả
   metadata + đường dẫn ảnh, tái dùng `_pick_samples` (đã rải đều theo nguyên
   nhân và ưu tiên ảnh có `expected`)
2. `GET /api/fleet/failure-image/{id}?w=480` — phục vụ file, **có thu nhỏ**

Điểm 2 không phải tuỳ chọn. Ảnh gốc 1–2 MB, đường tới Jetson vài chục KB/s; lưới
12 ảnh gốc là ~20 MB, tức khoảng 10 phút tải. Thu nhỏ về 480px là ~60 KB/ảnh.

Chặn đường dẫn: chỉ nhận id, không nhận đường dẫn tuỳ ý — tránh đọc file khác
trên máy.

---

## ❌ Thiếu 3 — Không có gộp theo GIỜ và theo CA ở tầng fleet

`/api/fleet/rollup` hiện gộp **theo ngày**. Bản mô tả yêu cầu lọc **theo giờ**
(Line Station, ngăn kéo máy) và **theo ca** (báo cáo).

`get_pass_fail_stats` đã nhận `group_by="hour"`, và `_shift_expr()` đã có sẵn.
Việc còn lại là mở tham số ra tới tầng fleet.

**Cần làm:** thêm `granularity=hour|day|week|shift` vào `/api/fleet/rollup` và
`/api/fleet/production`.

Lưu ý ca C: cửa sổ 22:00–06:00 vắt qua nửa đêm. Gộp theo ca mà dùng khoảng liên
tục sẽ mất toàn bộ ca đêm.

---

## ⚠️ Cần chú ý 4 — Bản ghi giả lập lẫn với bản ghi thật

`action_logs` có cả bản ghi mang cờ `simulated: true` (dữ liệu demo đã seed).

Nhật ký thao tác và mọi thống kê về người dùng **phải lọc được** theo cờ này, và
mặc định nên **loại chúng ra** ở môi trường thật. Không lọc thì số liệu demo trộn
vào số liệu thật và không cách nào phân biệt.

## ⚠️ Cần chú ý 5 — Username không duy nhất giữa các máy

Đã kiểm chứng: `admin`, `operator`, `supervisor` tồn tại trên **cả 5 máy** — là
các tài khoản khác nhau trùng tên, không phải một người. Mọi API nhân sự và nhật
ký ở tầng fleet phải khoá theo **(máy, username)**; muốn liên kết "một người trên
nhiều máy" thì dùng `employee_code`. Chi tiết: [09 — Mở rộng](09-scale.md), mục 9.

## ⚠️ Cần chú ý 6 — Vị trí máy trên sơ đồ

Chưa có toạ độ. Thêm vào `config/machines.json`:

```json
"nmupyJbod721CNTRL": {
  "label": "M2", "line": "Line 3", "model": "Jetson Orin Nano 8GB Super",
  "floor": {"x": 3, "y": 1, "rotation": 0, "zone": "Dãy A"}
}
```

Khoá theo Tailscale node id, giống các nhãn hiện có — bốn máy trùng hostname
`suntech-desktop` nên hostname không dùng làm khoá được.

---

## Tóm tắt: tính năng nào chặn bởi cái gì

| Tính năng | Chặn bởi | Ước lượng gỡ chặn |
|---|---|---|
| Sơ đồ nhà máy 3D | toạ độ trong config | 0,5 ngày |
| Ngăn kéo máy, lọc theo giờ | Thiếu 3 | 1 ngày |
| Thống kê toàn nhà máy | — (đã có) | 0 |
| Top N ảnh lỗi | **Thiếu 2** | 2 ngày |
| Tab Nhân sự | **Thiếu 1** | 1,5 ngày |
| Nhật ký thao tác | cần gộp 5 máy + lọc `simulated` | 1 ngày |
| Báo cáo theo ca | Thiếu 3 | (gộp với trên) |
