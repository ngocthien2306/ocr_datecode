# Fleet Service

Quản lý nhiều máy trạm chạy OCR Datecode từ một chỗ. Chạy trên máy dev hoặc VPS,
**không chạy trên Jetson**.

```bash
cp .env.sample .env      # điền FLEET_EDGE_PASSWORD
pip install -r requirements.txt
./run.sh                 # → http://localhost:8200
```

## Ba nguyên tắc

**Fleet chỉ đọc, và không được là thành phần thiết yếu.** 5 máy phải chạy bình
thường kể cả khi service này tắt cả tuần. Vì vậy fleet **kéo** dữ liệu về chứ
không bắt edge **đẩy** lên — chiều phụ thuộc đi từ trung tâm xuống thiết bị,
không bao giờ ngược lại.

**Số liệu đi đường xác định, lời giải thích đi đường ủy quyền.** Endpoint không
qua LLM cho dashboard và cảnh báo (rẻ, nhanh, chạy lại ra y hệt, và vẫn sống khi
OpenAI hết credit — đã xảy ra thật). `/ask` mới gọi agent của máy đích.

**Giao diện với edge là NĂNG LỰC, không phải SCHEMA.** Fleet không hỏi "collection
của mày có field gì", nó hỏi "mày làm được gì". Đó là thứ duy nhất sống sót qua
việc M1/M2/LineTine ở `release_v1` còn PC-Auto-1 ở `release_v2`.

## Cắm máy mới vào là chạy

Phát hiện qua `tailscale status --json`. Máy mới vào tailnet và chạy agent service
là **tự xuất hiện ở vòng quét kế tiếp** — không sửa file, không restart, không
deploy. `config/machines.json` chỉ để đặt tên đẹp và gắn nhãn, không phải để khai
báo sự tồn tại.

Khoá theo **Tailscale node id**: bốn máy trong đội hình này cùng hostname
`suntech-desktop`, còn IP thì đổi khi máy rời/vào lại tailnet.

### Thang năng lực

Fleet không giả định máy hỗ trợ gì. Nó dò từ dưới lên, dừng ở bậc cao nhất máy
đó đỡ được, và máy chưa nâng cấp thì tự động đi đường thấp hơn thay vì gãy.

| Bậc | Dò bằng | Biết được |
|---|---|---|
| L0 | cổng 8100 | máy tồn tại |
| L1 | `GET /api/agent/health` (không auth) | sống, mấy agent, backend có với tới không |
| L2 | login + `GET /api/agent/agents` | hỏi được loại câu nào |
| L3 | `POST /api/agent/chat` | ủy quyền được (không dò — tốn tiền) |
| L4 | `/api/jetson-monitoring/metrics` | số liệu phần cứng |

## Ba tình huống hỏng, ba cách nói khác nhau

| Trạng thái | Nghĩa |
|---|---|
| `ok` | bình thường |
| `agent_down` | **máy vẫn sản xuất**, chỉ agent tắt (thường sau reboot, vì agent chạy uvicorn trần) |
| `unreachable` | không với tới được |
| `offline` | không còn trên tailnet |

Máy hỏng **không bao giờ bị lược khỏi kết quả** — nó thành một dòng có lý do đọc
được. Bỏ nó ra thì "tổng sản lượng 5 máy" thiếu một máy mà nhìn vẫn hoàn toàn
bình thường, và không ai phát hiện.

`coverage` phân biệt **không lấy được gì** với **lấy được một phần**. Bản đầu chỉ
đếm cái sau là "đủ", nên khi tắt thử agent LineTine thì `system_metrics` vẫn về
còn `service_status` lỗi — và dashboard báo "đủ cả 5 máy" trong khi một máy đang hỏng.

## Endpoint

| | |
|---|---|
| `GET /api/fleet/machines` | danh sách + bậc năng lực |
| `GET /api/fleet/status` | cả đội hình, gộp song song (~1,3s cho 5 máy) |
| `GET /api/fleet/machine/{key}` | thẻ chi tiết một máy |
| `POST /api/fleet/refresh` | quét lại tailnet ngay |
| `GET /api/fleet/ask/{key}?q=` | ủy quyền câu hỏi cho agent máy đó (**đắt**, 4–20s) |

Tên mơ hồ trả **409 kèm danh sách ứng viên** chứ không tự chọn. `suntech-desktop`
khớp 4 máy; đoán bừa nghĩa là đưa số liệu của máy khác mà người dùng không biết.

`/ask` trả **nguyên văn** câu trả lời của edge, không tóm tắt lại: agent của máy
đó chính là chuyên gia về máy đó. Đứng giữa viết lại chỉ mất chi tiết, tốn thêm
một lượt LLM, và thêm một chỗ để bịa.
