# 09 — Mở rộng: cái gì gãy ở 10 / 20 / 50 máy

Rà soát lại toàn bộ thiết kế và code đang chạy dưới câu hỏi "nhiều máy hơn thì
sao". Mỗi mục ghi rõ **gãy ở quy mô nào** và **sửa lúc nào** — sửa sớm quá là
lãng phí, sửa muộn quá là làm lại.

## Đã chống sẵn từ thiết kế (không phải làm gì)

| Thứ | Vì sao đã ổn |
|---|---|
| Phát hiện máy | Quét tailnet — máy mới vào tailnet + chạy agent là tự xuất hiện, không sửa file |
| Định danh | Khoá theo Tailscale node id — hostname đã trùng 4/5 máy ngay từ hôm nay |
| Tầng agent | Máy là **tham số**, không phải agent — 50 máy vẫn là 8 tool, không phải 50 tool |
| Thang năng lực | Máy phiên bản cũ tự đi đường thấp hơn — đội hình không cần đồng nhất |
| Line Station | Mỗi máy một trang riêng, tự nhiên scale theo số máy |
| Biểu đồ vân tay trong báo cáo | Chiều cao đã tính theo số máy (`0.62 × N + 1.4`) |

## Gãy dần theo quy mô

### ~10 máy trở lên

**1. Trần song song của fan-out** — ✅ **đã sửa trong code** (20/08). `gather`
trần nghĩa là 50 máy = 50 kết nối cùng lúc từ một tiến trình; các link tới Jetson
vốn vài chục KB/s sẽ chèn nhau và máy khoẻ cũng thành timeout. Nay có
`FANOUT_CONCURRENCY = 8` trong cấu hình, áp cho cả fan-out lẫn vòng dò.

**2. `ask_all_machines` — chi phí tuyến tính theo số máy.** 5 máy = 5 lượt LLM
(~25 giây, vài xu). 30 máy = 30 lượt (~2 phút, tiền thật) **cho một câu hỏi**.

Quy tắc khi vượt ~10 máy: tool phải **báo trước chi phí và hỏi lại** —
*"Câu này sẽ hỏi 30 máy, mất ~2 phút. Tiếp tục, hay lọc theo dãy/line trước?"* —
đúng khuôn `ask_user` của báo cáo. Và gợi ý mặc định phải đổi: tìm outlier bằng
tool rẻ trước, chỉ ủy quyền vào máy lệch.

**3. Picker chọn máy trong chat.** Checkbox 5 cái thì đẹp; 30 cái là bức tường.
Từ ~10 máy: nhóm theo **dãy/khu** (đã có `zone` trong thiết kế `machines.json`),
thêm ô tìm, và nút chọn cả nhóm ("Cả dãy A").

**4. Biểu đồ nhiều đường.** Một đường mỗi máy đọc được tới ~8 đường. Quá đó thì
quy tắc chuẩn: **chọn máy để tô đậm, phần còn lại xám mờ** + dải min–max của cả
đội hình. Tuyệt đối không vẽ 30 đường 30 màu.

**5. Bảng màu 8 màu, gán theo hash tên.** Hai máy có thể trùng màu ngay từ hôm
nay (hash va chạm), và chắc chắn trùng khi quá 8 máy. Chấp nhận được cho báo cáo
ít máy; khi quá 8, chuyển sang quy tắc ở mục 4 (đậm/mờ) thay vì thêm màu — mắt
người không phân biệt nổi 15 màu trên một biểu đồ.

### ~20 máy trở lên

**6. Mỗi request `/api/fleet/status` tự fan-out lại.** Hiện mỗi lượt tải
dashboard là một lượt gọi thật ra mọi máy — 3 người mở dashboard cùng lúc là
×3 tải lên các Jetson, cộng với vòng dò nền cũng đang chạy. Ở 5 máy thì vô hại.

Cách sửa (GĐ khi cần): vòng poll nền **lưu snapshot**, endpoint trả snapshot kèm
`generated_at`; chỉ nút "Làm mới" mới ép gọi thật. Người xem thứ 10 không tốn
thêm gì.

**7. Lưới thẻ máy và sơ đồ.** 20 thẻ vẫn cuộn được nhưng mất "nhìn một cái biết
ngay". Từ ~12 máy: nhóm thẻ theo dãy/khu, mặc định **thu gọn nhóm nào toàn máy
khoẻ** — màn hình đầu chỉ hiện máy cần chú ý. Sơ đồ isometric vẫn ổn tới ~20–30
khối trên một tầng; nhiều tầng/xưởng thì thêm chuyển tầng, và đó mới là lúc cân
nhắc lại Three.js.

**8. Nhật ký thao tác gộp N máy.** Hiện gộp bằng cách hỏi từng máy mỗi lần mở
tab. 20 máy × vài nghìn bản ghi là chậm và lặp. Khi tới đó: fleet giữ **SQLite
tích luỹ** (đọc tăng dần theo `timestamp` từng máy), tab đọc từ đó. Cũng chính là
nền cho lịch sử xu hướng dài hạn (P11).

### Bất kể quy mô — lỗi logic đã kiểm chứng

**9. Username KHÔNG duy nhất giữa các máy.** Đã kiểm chứng: `admin`, `operator`,
`supervisor` tồn tại trên **cả 5 máy hiện tại** — là 5 tài khoản khác nhau trùng
tên, không phải một người. Càng nhiều máy càng nhiều trùng.

Hệ quả bắt buộc cho tab Nhân sự và nhật ký thao tác:
- Khoá mọi bản ghi người dùng theo **(máy, username)**, không theo username trần
- Câu hỏi chat *"hôm nay admin làm gì?"* phải trả lời **theo từng máy** hoặc hỏi
  lại "admin của máy nào?", không được gộp
- Muốn có danh tính "một người trên nhiều máy" thì dùng `employee_code` (NV-xxxx)
  làm khoá liên kết — nhưng đó là tính năng riêng, có chủ đích, không phải mặc định

**10. Nhãn tay trong `machines.json`.** 5 máy đặt tên tay được; 50 máy là gánh
vận hành. Quy ước từ bây giờ: máy chưa có nhãn vẫn **được quản đầy đủ** (hiện
hostname + 4 số cuối node id), nhãn chỉ làm đẹp — nguyên tắc này đã có, giữ chặt.
Thêm: trang quản trị nhỏ để đặt nhãn ngay trên giao diện thay vì sửa JSON.

**11. Ca làm việc đang là hằng số toàn cục.** `SHIFTS` (A/B/C) hardcode chung
cho mọi máy. Nhà máy mới có thể chạy 2 ca 12 tiếng. Khi điều đó xảy ra: đưa định
nghĩa ca vào cấu hình **theo máy** (mặc định thừa kế bộ chung). Chưa cần làm ngay
— nhưng mọi code mới đụng tới ca nên đọc qua một hàm `shifts_for(machine)` thay
vì import thẳng hằng số, để hôm đó chỉ sửa một chỗ.

## Ngưỡng hành động — tóm tắt

| Quy mô | Việc phải làm trước khi tới đó |
|---|---|
| **Hiện tại (5)** | ✅ trần fan-out (xong) · khoá nhân sự theo (máy, username) ngay từ GĐ 4 |
| **~10** | picker nhóm theo dãy · `ask_all` hỏi lại khi quá 10 máy · biểu đồ đậm/mờ |
| **~20** | snapshot cho `/status` · lưới thẻ thu gọn nhóm khoẻ · SQLite cho nhật ký |
| **~50** | cân nhắc lại transport (message bus) — đã ghi trong phân tích kiến trúc: quá ~30 thiết bị hoặc ra ngoài tailnet là ranh giới đổi sang MQTT/NATS, giữ nguyên tầng agent |

Nguyên tắc chung cho mọi mục trên: **giao diện và tool không được giả định danh
sách máy là ngắn**. Mọi chỗ liệt kê máy phải trả lời được ba câu: nhóm theo gì,
tìm kiếm thế nào, và cái gì được ẩn đi khi mọi thứ bình thường.
