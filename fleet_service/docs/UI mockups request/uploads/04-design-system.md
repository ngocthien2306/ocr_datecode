# 04 — Design System

Dùng chung cho cả Fleet Console và Line Station. Bảng token hiện đã áp dụng trong
`fleet_service/static/index.html`; phần dưới chuẩn hoá lại và bổ sung phần còn thiếu.

## Màu

Sáng là mặc định, tối là lựa chọn. **Bảng màu đầy đủ đặt ở `:root` trần**, khối
`[data-theme="dark"]` chỉ ghi đè — định nghĩa màu chỉ trong khối theme thì lúc
chưa chọn gì trang không có màu nào.

```css
:root{
  --bg:#f4f6fa; --card:#fff; --card2:#f7f9fc; --line:#e3e7ee;
  --fg:#141821; --dim:#5f6878; --faint:#98a1b3;
  --ok:#1a7f37; --warn:#9a6700; --bad:#cf222e; --accent:#0969da;
}
:root[data-theme="dark"]{
  --bg:#0b0d12; --card:#151822; --card2:#1b1f2b; --line:#252a37;
  --fg:#e8ebf2; --dim:#8b93a7; --faint:#5a6275;
  --ok:#3fb950; --warn:#d29922; --bad:#f85149; --accent:#4c8dff;
}
```

**Màu chuỗi dữ liệu gán theo TÊN máy**, không theo thứ tự trong danh sách:

```js
const color = name => PALETTE[hash(name) % PALETTE.length];
```

Gán theo thứ tự thì bỏ một máy khỏi bộ lọc là mọi máy còn lại đổi màu, và hai
biểu đồ cạnh nhau không đọc chéo được nữa.

## Ngưỡng cảnh báo

Lấy theo giới hạn thật của phần cứng, không phải số tròn cho đẹp:

| Chỉ số | Bình thường | Chú ý | Nguy hiểm | Căn cứ |
|---|---|---|---|---|
| Nhiệt CPU/GPU | < 80°C | 80–92°C | ≥ 92°C | Orin hạ xung ~85°C; x86 ở đây `crit=100` |
| RAM | < 80% | 80–92% | ≥ 92% | Jetson 8GB, camera service giữ ~2GB |
| Đĩa | < 80% | 80–92% | ≥ 92% | Mongo ngừng ghi khi hết đĩa |

## Quy tắc trình bày số — bắt buộc

Nhóm quy tắc này rút từ các lỗi đã xảy ra thật ở tầng edge. Mỗi dòng là một lỗi
đã tốn thời gian truy nguyên.

| Quy tắc | Vì sao |
|---|---|
| Không có số ⇒ hiện **`—`**, không hiện `0` | Nhiệt độ `null` vẽ thành `0°C` đọc như máy rất mát — sai theo hướng nguy hiểm nhất |
| Phần trăm luôn kèm số tuyệt đối | "RAM 87%" trên máy 8GB và 32GB là hai tình huống khác hẳn |
| Điểm phần trăm ghi **"điểm"**, không ghi `%` | "+1,36đ" từng bị đọc thành "1,36 đồng" |
| Ghi rõ đơn vị: **sản phẩm** hay **frame** | Một sản phẩm có nhiều frame; "328 sản phẩm fail" trong khi tổng fail là 167 |
| Tỉ lệ trên mẫu ghi rõ **"của mẫu"** | `explain_failures` lấy mẫu; số tuyệt đối đặt cạnh tổng kỳ là dựng sẵn cái bẫy |
| Nền so sánh < 30 bản ghi ⇒ **không đưa delta** | "+3.309.850%" vì nền có 2 bản ghi |
| So hai kỳ khác độ dài ⇒ đưa **per-day** | "tăng 59,75%" khi so 19 ngày với 31 ngày; thực tế per-day tăng 160% |
| Biểu đồ cắt bớt cột ⇒ **ghi vào tiêu đề** | Biểu đồ giờ từng thiếu 12h–19h, gồm cả giờ tệ nhất, mà im lặng |
| Ca chưa bắt đầu ⇒ **"chưa bắt đầu"** | Không phải "0 sản phẩm, đạt 0%" |

## Trạng thái của mọi khối dữ liệu

Mỗi khối phải định nghĩa đủ **năm** trạng thái. Thiếu trạng thái nào thì trạng
thái đó sẽ tự hiện ra dưới dạng xấu nhất — thường là một khối trống không giải
thích gì.

| Trạng thái | Thể hiện |
|---|---|
| Đang tải | khung xương, giữ đúng chiều cao để trang không nhảy |
| Có dữ liệu | bình thường |
| Rỗng có lý do | "Ca chưa bắt đầu" / "Không có sản phẩm lỗi trong kỳ" — **nêu lý do**, không để trống |
| Một phần | hiện phần có được **+ nêu tên** phần thiếu |
| Lỗi | câu tiếng người + việc cần làm; không hiện tên exception |

Trạng thái **"một phần"** là quan trọng nhất và hay bị bỏ nhất. Ví dụ thật: tắt
agent của một máy thì số liệu phần cứng vẫn về (backend còn sống) còn trạng thái
service thì lỗi. Nếu chỉ có hai trạng thái "được / không được", màn hình sẽ báo
"đủ cả 5 máy" trong khi một máy đang hỏng.

## Ba cách gọi tên trạng thái máy — không được gộp

| Trạng thái | Nghĩa | Nói với người dùng |
|---|---|---|
| `ok` | bình thường | "đang chạy" |
| `agent_down` | **máy vẫn sản xuất**, chỉ trợ lý tắt | "trợ lý tắt · máy vẫn chạy" |
| `unreachable` | mất liên lạc với máy | "không với tới được từ 14:22" |
| `offline` | không còn trên mạng nội bộ | "ngoài mạng" |

Gộp `agent_down` vào "máy chết" là điều người trực sẽ chạy nhầm chỗ.

## Component

| Component | Ghi chú |
|---|---|
| `MachineCard` | thẻ máy: tên, model, trạng thái, 4 ô chỉ số, service |
| `FactoryMap` | interface `{machines, selected, onSelect}` — đổi đẳng cự ↔ 3D chỉ là thay ruột |
| `MachineDrawer` | ngăn kéo phải, không che sơ đồ |
| `PeriodPicker` | Giờ / Ngày / Tuần / Ca — **một component duy nhất** dùng khắp nơi |
| `CoverageNote` | dòng phạm vi mẫu; **bắt buộc dưới mọi số liệu tổng hợp** |
| `FailureGrid` | lưới ảnh lỗi, kèm `mong → đọc` |
| `StaffCard` | ảnh, tên, mã NV, chức vụ, ca, **"Quyền: operator"** có nhãn |
| `ChatPanel` | chat + picker + chip gợi ý + nút tải file |
| `AskPicker` | render `ask_user` thành nút bấm (xem [05](05-agent-ux.md)) |

## Song ngữ

Tiếng Anh mặc định, tiếng Việt chuyển được, lưu trong `localStorage`.

**Mọi chuỗi gom vào một bảng**, không rải trong hàm dựng component. Rải ra thì
lần thêm ngôn ngữ sau sẽ sót, mà sót chỗ nào chỉ lộ khi có máy rơi đúng vào trạng
thái hiếm đó — ví dụ `agent_down` chỉ hiện khi thật sự có máy hỏng.

Đổi ngôn ngữ **vẽ lại từ dữ liệu đã tải**, không gọi lại API: mỗi lần bấm EN/VI
mà refetch là 5 lượt gọi ra Jetson qua đường truyền chậm chỉ để đổi chữ.

Số và giờ theo locale (`en-GB` / `vi-VN`).

## Nhịp làm mới

| Dữ liệu | Nhịp | Vì sao |
|---|---|---|
| Trạng thái + phần cứng | 30s | rẻ (1,3s), thay đổi liên tục |
| Sản xuất + vân tay lỗi | 5 phút | rollup lạnh mổ vài trăm document trên Jetson |
| Line Station | 15s | người vận hành đứng nhìn liên tục |
| Nhân sự | khi mở tab | gần như không đổi |

Một lần gọi hụt **không được xoá trắng bảng cũ** — giữ số cũ kèm dấu thời gian.
