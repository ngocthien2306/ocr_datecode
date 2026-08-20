# 01 — Bối cảnh sản phẩm

## Hai bề mặt, hai công việc khác hẳn nhau

Đây là quyết định gốc của toàn bộ thiết kế: **không dùng chung một giao diện rồi
ẩn/hiện theo quyền.** Hai nhóm người dùng không phải là cùng một người với quyền
khác nhau — họ trả lời hai câu hỏi khác nhau, ở hai khoảng chú ý khác nhau.

| | **Fleet Console** (máy tổng) | **Line Station** (tại một dây chuyền) |
|---|---|---|
| Người dùng | Quản đốc, trưởng ca, kỹ thuật, QA | Công nhân vận hành tại chính line đó |
| Câu hỏi | "Dây chuyền nào cần tôi tới?" | "Line của tôi đang chạy có ổn không?" |
| Phạm vi | Toàn bộ 5 máy, có so sánh | Đúng một máy, không so sánh |
| Chu kỳ nhìn | Vài lần mỗi ca, ngồi bàn | Liên tục, đứng cạnh máy |
| Thiết bị | Màn hình lớn, chuột | Tablet / màn cảm ứng, đeo găng |
| Đơn vị thời gian | Ca, ngày, tuần | Giờ hiện tại, ca hiện tại |
| Hành động chính | Điều tra, so sánh, xuất báo cáo | Nhận biết sớm, ghi nhận, bàn giao ca |

Hệ quả cụ thể: Line Station **không có** phần so sánh giữa các máy. Đưa nó vào là
mời người vận hành so line mình với line khác — mà như đã nói, các line chạy mặt
hàng khác nhau nên phép so đó vô nghĩa và dễ gây tị nạnh.

## Người dùng và việc họ cần làm

### Quản đốc / trưởng ca — Fleet Console
1. Mở máy đầu ca: 5 line đang thế nào, có line nào cần tới ngay không
2. Thấy một line tụt: tìm hiểu vì sao, có phải vấn đề máy hay vấn đề mặt hàng
3. Cuối ca: bàn giao — sản lượng, sự cố, ai trực ca tới
4. Cuối tuần: xuất báo cáo so sánh cho cấp trên

### Kỹ thuật bảo trì — Fleet Console
1. Máy nào có dấu hiệu hỏng trước khi nó dừng
2. Nhiệt độ, RAM, đĩa của cả đội hình
3. Vân tay kiểu lỗi: lỗi thuộc camera hay thuộc mặt hàng

### QA — Fleet Console
1. Nguyên nhân lỗi chính, kèm **ảnh sản phẩm lỗi** để mắt người xác nhận
2. Truy vết: lô nào, ca nào, máy nào

### Công nhân vận hành — Line Station
1. Line đang chạy recipe gì, sản lượng ca này tới đâu, so với chỉ tiêu
2. Vừa có sản phẩm lỗi không, ảnh trông ra sao
3. Máy có báo gì bất thường không

### Quản trị hệ thống — Fleet Console
1. 54 tài khoản trên 5 máy: ai có quyền gì, ở line nào
2. Ai vừa làm gì trên máy nào (audit)

## Ràng buộc đã đo được, ảnh hưởng trực tiếp tới thiết kế

| Ràng buộc | Số đo | Hệ quả thiết kế |
|---|---|---|
| Đường truyền tới Jetson | vài chục KB/s | Không tải ảnh gốc 1–2 MB; ảnh phải có bản thu nhỏ |
| Độ trễ hỏi agent | 4–20s, có lúc 27s | Chat phải có trạng thái chờ rõ ràng; dashboard không đi qua agent |
| Rollup không LLM | 1,4 KB, 0,2–2,0s | Nhịp làm mới dashboard 30s; phần sản xuất 5 phút |
| Jetson RAM trống | ~1 GB | Không thêm dịch vụ nặng ở edge |
| Recipe trùng nhau | 1/5 máy | Không xếp hạng bằng pass rate |
| Ca sản xuất | A 06–14, B 14–22, C 22–06 | Ca C vắt qua nửa đêm — mọi bộ lọc theo ca phải xử lý |

## Ngoài phạm vi lần này

- Điều khiển máy từ xa (start/stop recipe) — có tác dụng phụ, cần thiết kế cổng xác nhận riêng
- Phân quyền theo vai trò trên Fleet Console — hiện dùng chung một tài khoản admin
- Tạo / sửa / xoá user từ Fleet Console — thao tác ghi, cần cổng xác nhận; giai đoạn đầu chỉ xem
- Ứng dụng di động

## Một quyết định phải nói rõ: Fleet Console hiện KHÔNG có đăng nhập

Cổng `:8200` không có trang login. Đây là quyết định có chủ đích ở giai đoạn
này: fleet chỉ với tới được **qua tailnet** — mạng riêng, thiết bị phải được mời
vào — nên tầng mạng đang làm việc của tầng đăng nhập.

Nhưng nó là ranh giới cứng: **trước khi đưa Fleet Console ra ngoài tailnet (ngrok,
VPN khách, mạng xưởng mở) thì phải thêm đăng nhập trước**, vì bên trong nó có
danh sách nhân sự, nhật ký thao tác và nút xuất báo cáo. Line Station thì khác:
nó chạy trên chính máy đó và dùng đăng nhập sẵn có của backend máy đó.
