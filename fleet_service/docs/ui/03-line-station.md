# 03 — Line Station (giao diện tại một dây chuyền)

Màn hình đặt cạnh dây chuyền, cho công nhân vận hành line đó. Chạy trên chính máy
đó (`:8100/station`), không cần fleet service — **line vẫn dùng được khi mạng ra
ngoài chết**.

## Nguyên tắc riêng của bề mặt này

**Đọc được từ 2 mét.** Người vận hành đứng, không ngồi. Số quan trọng cỡ chữ tối
thiểu 32px; nhãn phụ 14px. Không có bảng nhiều cột.

**Vùng chạm tối thiểu 48×48px.** Người dùng đeo găng tay.

**Không có phần so sánh với line khác.** Line này chạy quế, line kia chạy muối —
so tỉ lệ pass là so độ khó mặt hàng. Đưa vào chỉ gây tị nạnh mà không giúp gì.

**Không có nút nào gây tác dụng phụ.** Không start/stop recipe, không sửa cấu
hình. Màn hình này để **nhìn và ghi nhận**, không để điều khiển.

**Tự làm mới, không cần bấm.** Người vận hành có việc khác để làm; màn hình phải
tự cập nhật 15s một lần.

## Bố cục

```
┌──────────────────────────────────────────────────────────────┐
│  M2 · Line 3          Ca B 14:00–22:00        16:42          │
│  ORG CINNAMON CC                              ● đang chạy    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│    SẢN LƯỢNG CA NÀY              TỈ LỆ ĐẠT                   │
│                                                              │
│       1.284                        69,4%                     │
│    ────────────── 71%              ▼ 8,2 điểm so ca trước    │
│    chỉ tiêu 1.800                                            │
│                                                              │
│    ▁▂▃▅▇▆▄▃▂▁▂▃  ← pass/fail theo giờ trong ca               │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  SẢN PHẨM LỖI GẦN ĐÂY                          [xem tất cả]  │
│  ┌──────┐┌──────┐┌──────┐┌──────┐                            │
│  │[ảnh] ││[ảnh] ││[ảnh] ││[ảnh] │                            │
│  │16:38 ││16:31 ││16:29 ││16:22 │                            │
│  │mong  ││mong  ││ ...  ││ ...  │                            │
│  │BB/26 ││BB/26 ││      ││      │                            │
│  │đọc   ││đọc   ││      ││      │                            │
│  │BB/2G ││(trống)│      ││      │                            │
│  └──────┘└──────┘└──────┘└──────┘                            │
├──────────────────────────────────────────────────────────────┤
│  TÌNH TRẠNG MÁY                                              │
│  ● camera service chạy   CPU 56°C   RAM 87%   Đĩa 74%        │
│                                                              │
│  ⚠ RAM đang 87% — theo dõi thêm                              │
├──────────────────────────────────────────────────────────────┤
│  NGƯỜI TRONG CA                                              │
│  [ảnh] Phan Thị Thuỳ Dương · CN vận hành · vào ca 14:02      │
│  [ảnh] Chu Thị Hải Yến · Giám sát QA · vào ca 14:05          │
├──────────────────────────────────────────────────────────────┤
│  [Bàn giao ca]                          [Hỏi trợ lý]         │
└──────────────────────────────────────────────────────────────┘
```

## Chi tiết từng khối

### Đầu trang
Tên line, recipe đang chạy, ca hiện tại kèm khung giờ, đồng hồ. Chấm trạng thái.

**Ca C vắt qua nửa đêm (22:00–06:00).** Lúc 01:00 sáng, "ca hiện tại" là ca C bắt
đầu từ 22:00 *hôm qua*. Mọi phép tính "sản lượng ca này" phải theo cửa sổ đó, nếu
không lúc 00:05 màn hình sẽ nhảy về 0 và người vận hành tưởng mất dữ liệu.

### Sản lượng ca
Hai số lớn: sản lượng và tỉ lệ đạt. Thanh tiến độ so với chỉ tiêu ca (lấy từ
`config/production_targets.json`).

**Ca chưa bắt đầu thì nói "chưa bắt đầu", không hiện "0 sản phẩm, đạt 0%".** Đây
là lỗi đã sửa ở tầng edge (`not_started`), và giao diện phải tôn trọng nó — "0%"
đọc như máy đang hỏng.

Delta so ca trước ghi rõ đơn vị: **"▼ 8,2 điểm"**, không phải "▼ 8,2%". Điểm phần
trăm và phần trăm là hai thứ khác nhau; lỗi "+1,36đ" bị đọc thành "1,36 đồng" đã
xảy ra thật.

### Sản phẩm lỗi gần đây
4 ảnh mới nhất, kèm giờ và `mong → đọc`. Chạm vào một ảnh mở toàn màn hình.

Ảnh phải là **bản thu nhỏ**. Ảnh gốc 1–2 MB, mà màn hình này tự làm mới 15s —
tải ảnh gốc là ép đường truyền và ép đĩa Jetson.

### Tình trạng máy
Một dòng: camera service, CPU, RAM, đĩa. Chỉ hiện cảnh báo khi vượt ngưỡng, chứ
không liệt kê mọi chỉ số bình thường.

**Nhiệt độ `null` hiện "—", không hiện 0°C.** Máy x86 không có cảm biến kiểu
Jetson; "0°C" đọc như máy đang rất mát, sai đúng theo hướng nguy hiểm nhất.

### Người trong ca
Ảnh + tên + chức vụ + giờ vào ca. Đây là phần làm màn hình "có người", và giúp
trưởng ca biết ai đang ở đâu.

### Bàn giao ca
Nút mở bản giao ca của ca hiện tại: sản lượng, chỉ tiêu, dừng máy, nguyên nhân
lỗi chính, cảnh báo thiết bị, người trong ca. Tái dùng `get_shift_handover` đã có
ở edge — **một lời gọi, không phải sáu**.

### Trợ lý
Chat gọn, chỉ hỏi về máy này. Gợi ý sẵn:
- "Ca này có gì bất thường không?"
- "Vì sao mấy sản phẩm vừa rồi bị loại?"
- "Máy có cảnh báo phần cứng nào không?"

## Chế độ suy giảm

| Tình huống | Màn hình làm gì |
|---|---|
| Mất mạng ra ngoài | Vẫn chạy đầy đủ — mọi dữ liệu lấy từ chính máy này |
| Agent service tắt | Ẩn phần chat, **giữ nguyên** mọi số liệu; báo "trợ lý tạm không dùng được" |
| OpenAI hết credit | Như trên |
| Vòng giám sát phần cứng chết | Khối "tình trạng máy" ghi "chưa đo được", **không** ghi 0 |
| Chưa có dữ liệu ca | "Ca chưa bắt đầu" kèm số liệu ca trước để đối chiếu |
