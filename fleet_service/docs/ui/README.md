# Tài liệu UI/UX — Hệ thống quản lý tập trung

Bộ tài liệu để dựng hai giao diện của hệ thống OCR Datecode nhiều dây chuyền.

| Tài liệu | Nội dung | Đọc khi |
|---|---|---|
| [01 — Bối cảnh sản phẩm](01-product-brief.md) | Ai dùng cái gì, hai bề mặt khác nhau ra sao | trước tiên |
| [02 — Fleet Console](02-fleet-console.md) | Màn hình máy tổng: sơ đồ 3D, thống kê, điều tra lỗi, nhân sự | dựng giao diện quản lý |
| [03 — Line Station](03-line-station.md) | Màn hình tại một dây chuyền, cho nhân viên vận hành | dựng giao diện tại line |
| [04 — Design System](04-design-system.md) | Token màu, component, trạng thái rỗng/lỗi/đang tải, song ngữ | viết bất kỳ giao diện nào |
| [05 — Agent UX](05-agent-ux.md) | Quy tắc hội thoại: khi nào hỏi lại, khi nào không | làm phần chat và picker |
| [06 — Data Contracts](06-data-contracts.md) | API nào đã có, API nào còn thiếu, cho từng tính năng | **trước khi ước lượng công việc** |
| [07 — Lộ trình](07-roadmap.md) | Kế hoạch theo giai đoạn, có mốc kiểm chứng | lập kế hoạch |
| [08 — Đề xuất thêm](08-proposals.md) | Tính năng đề xuất, dựa trên dữ liệu thật đã khảo sát | quyết định phạm vi |

## Ba nguyên tắc xuyên suốt, kế thừa từ hệ thống hiện tại

Ba điều dưới đây không phải sở thích thiết kế. Chúng là kết luận rút ra sau một
loạt lỗi đã xảy ra thật, ghi trong `agent_service/docs/PIPELINE.md`, và bộ tài
liệu này giữ nguyên chúng.

**1. Số liệu đi đường xác định, lời giải thích đi đường ủy quyền.** Thứ hiển thị
lặp đi lặp lại (dashboard, thẻ máy, biểu đồ) lấy từ endpoint không qua LLM: rẻ,
nhanh, chạy lại ra y hệt, và vẫn sống khi OpenAI hết credit — chuyện đã xảy ra,
và lúc đó cả 5 agent im tiếng cùng lúc. Chỉ câu hỏi mở mới gọi agent.

**2. Máy thiếu dữ liệu phải được nêu tên.** Mọi con số tổng hợp đi kèm phạm vi
mẫu. Một bảng "toàn nhà máy" thiếu một dây chuyền trông vẫn hoàn toàn bình
thường — không ai phát hiện nếu giao diện im lặng.

**3. Không xếp hạng dây chuyền bằng tỉ lệ pass.** Năm máy chạy năm mặt hàng khác
nhau; khảo sát thực tế cho thấy **đúng một recipe** được chia sẻ giữa hai máy
trên cả đội hình. So tỉ lệ pass giữa hạt tiêu và muối là so độ khó mặt hàng, chứ
không phải so máy. Cái so được là **vân tay kiểu lỗi**.

## Trạng thái hệ thống khi viết tài liệu này (20/08/2026)

- 5 dây chuyền: Auto2, M1, M2, LineTine (Jetson Orin Nano 8GB) và PC-Auto-1 (x86)
- 54 tài khoản người dùng trên toàn nhà máy
- ~117.000 sản phẩm/7 ngày, pass 95,5% toàn nhà máy
- Mỗi máy có agent service riêng với 5 agent; fleet service điều phối qua Tailscale
