# 10 — Thiết kế mẫu báo cáo: một máy & toàn nhà máy

Tài liệu này có hai phần: **(A)** những gì code xuất báo cáo hiện tại đã làm —
đọc từ `agent_service/agent_app/reports/` và `fleet_service/fleet_app/reports/` —
và **(B)** hai prompt sẵn để dán vào Claude Design, dựng mẫu báo cáo mới trên
đúng nền đó.

---

## A. Code gốc đang xuất gì

### Báo cáo MỘT MÁY (`agent_service/agent_app/reports/`, 1.478 dòng)

Nguồn gốc quan trọng: CSS được **trích nguyên văn** từ
`frontend-ts/src/utils/reportGenerator.ts` — cố ý, để báo cáo do agent xuất trông
giống hệt báo cáo do panel Historical xuất. *Cùng một kỳ sản xuất mà ra hai bản
báo cáo khác nhau thì người đọc không biết tin bản nào.* Mẫu mới phải giữ nguyên
tính chất này: **một bộ style, hai đường xuất.**

Cấu trúc trang (theo `html_report.py`):

| Khối | Nội dung | Bật/tắt |
|---|---|---|
| Header | "PRODUCTION INSPECTION REPORT" · kỳ · khoảng ngày · breakdown · giờ sinh | luôn có |
| KPI row | 4 ô: Total Inspected / Pass / Fail / Pass Rate — ô Pass Rate đổi màu theo ngưỡng kèm chữ trạng thái | `kpi` |
| Recipe summary | bảng theo recipe: total/pass/fail/rate + trạng thái | `kpi` |
| Combined charts | trend + pass/fail | `trendChart` / `passfailChart` |
| Per-recipe | mỗi recipe một khối riêng kèm biểu đồ | `perRecipe` |
| Footer | "OCR Datecode Inspection System" · giờ sinh · kỳ | luôn có |

Ba theme (class trên `<body>`):

| Theme | Nhận diện |
|---|---|
| `industrial` (mặc định) | nền trắng, header navy `#1e2a3a`, accent xanh `#2563eb` |
| `dark` | nền slate `#0f172a`, accent sky `#38bdf8` |
| `executive` | header gradient `#1e3a8a → #1d4ed8`, nhãn KPI trong chip xanh |

Kỹ thuật: bản HTML dùng Chart.js; bản PDF (WeasyPrint) **không chạy JS** nên
biểu đồ vẽ sẵn PNG bằng matplotlib rồi nhúng base64 (`charts_png.py` còn theo
theme). Excel: sheet "Tổng quan" (meta + tổng) rồi sheet theo recipe / camera /
chuỗi thời gian, freeze panes hàng đầu.

Dữ liệu có sẵn (`data.py`): tổng pass/fail/rate · theo **recipe** · theo
**camera** · chuỗi thời gian theo **giờ/ngày/tuần**.

### Báo cáo TOÀN NHÀ MÁY (`fleet_service/fleet_app/reports/`, 3 module)

**Đã dựng lại theo Prompt 2 ở mục B** (2026-08-21). Ba module tách việc:
`aggregate.py` tính (stdlib thuần, gọi trực tiếp kiểm được từng con số),
`charts.py` vẽ PNG, `builder.py` dựng trang.

| Tờ | Nội dung |
|---|---|
| 1 | Header navy + chỗ logo · banner phạm vi dữ liệu (xanh đủ / cam thiếu) · 4 KPI (ô tỉ lệ đổi màu theo ngưỡng) · nhịp sản xuất theo ngày (cột chồng theo máy) · bảng máy 7 cột + hàng TỔNG + chú thích không xếp hạng |
| 2 | Vân tay kiểu lỗi (cột chồng ngang 100% + bảng nhiệt có trung vị và ô lệch mạnh) · khung "đọc thế nào" · phát hiện chính dựng bằng code · **phụ lục theo máy chia theo TUẦN ISO** |

Phụ lục: mỗi máy một thẻ — ba số, biểu đồ cột chồng đạt/không đạt + đường tỉ lệ,
bảng theo tuần (sản lượng · đạt · không đạt · ngày có chạy · mỗi ngày có chạy ·
tỉ lệ · delta điểm so tuần trước), recipe trong kỳ, nguyên nhân lớn nhất, ngày
thấp nhất. Máy không có số vẫn có thẻ, nền xám, kèm lý do.

Excel 3 sheet (Sản lượng · Theo tuần · Vân tay), CSV một file hai khối.

Quy tắc mới phải giữ:

| Quy tắc | Lỗi gốc |
|---|---|
| Màu máy gán qua `charts.configure()`, dò khi trùng | `M2` và `PC-Auto-1` băm về cùng ô → hai máy cùng màu xanh lá trong một báo cáo so sánh |
| Dấu nghìn `.`, thập phân `,` — một hàm duy nhất dùng cho cả bảng lẫn câu văn | U+202F làm dấu nghìn: mất hẳn ở cỡ 17pt in đậm ("126587"); và bảng in "63,88%" còn phát hiện chính in "63.88%" |
| Trung vị cột chỉ tính khi ≥3 máy có số | cột một máy cho "trung vị" đúng bằng chính nó |
| Hai cột "mỗi ngày": cả kỳ và ngày có chạy | máy chạy 2/7 ngày trông như máy yếu |
| Chú giải biểu đồ đặt TRÊN vùng vẽ | chú giải dưới cắt ngang nhãn trục x |
| Chú giải kiểu cột phải trung tính màu | ô chú giải màu Auto2 đọc thành "vàng nghĩa là cả kỳ" |
| Không máy nào đọc được → banner đỏ | `complete=True` với 0 máy in ra "Đủ cả 0/0 máy" |
| Thẻ máy ≤ ~120mm để hai thẻ vừa một trang | thẻ 145mm → mỗi trang một thẻ, thừa 150mm trắng |

### Quy tắc nội dung đã trả giá mới có — mẫu mới KHÔNG được làm mất

| Quy tắc | Lỗi gốc |
|---|---|
| Báo cáo toàn nhà máy **không xếp hạng máy theo pass rate**, in rõ chú thích | 5 máy chạy 5 mặt hàng; đúng 1 recipe trùng giữa 2 máy |
| Vân tay lỗi là **tỉ trọng của mẫu**, ghi "N mẫu · lấy mẫu/phủ hết kỳ" từng hàng | "tổng fail 1.036" rồi liệt kê số của mẫu 294 |
| Thiếu máy nào → **banner ngay đầu trang**, không phải chú thích cuối | bảng thiếu 1 máy trông vẫn hoàn toàn bình thường |
| Không có số → `—`, không phải `0` | nhiệt độ null in `0°C` = "máy rất mát" |
| Delta điểm phần trăm ghi chữ "điểm" | "+1,36đ" bị đọc thành tiền |
| Tiêu đề + câu dẫn + biểu đồ **không tách trang** (`break-after: avoid`) | WeasyPrint từng bỏ tiêu đề mồ côi cuối trang 1 |
| Màu máy gán theo **tên**, không theo thứ tự | bỏ 1 máy khỏi báo cáo là các máy còn lại đổi màu |
| Biểu đồ là **ảnh tĩnh** trong cả HTML lẫn PDF của fleet | canvas trắng trơn trong PDF |

---

## B. Hai prompt cho Claude Design

Cách dùng: template **UI mockups** (không phải 3D object), design system
*Industry*. Mỗi prompt tự đứng được — dán nguyên văn.

### Prompt 1 — Báo cáo MỘT MÁY

```
Thiết kế mẫu BÁO CÁO SẢN XUẤT cho một dây chuyền kiểm tra date code bằng OCR
(nhà máy đóng gói gia vị). Khổ A4 dọc, dùng cho cả bản in PDF lẫn bản HTML.
Biểu đồ là ảnh tĩnh (PDF không chạy JavaScript). Tiếng Việt.

3 artboard:

1. TRANG 1 — theme "Industrial" (nền trắng, header navy #1e2a3a, accent #2563eb)
   - Header thanh navy: trái là tiêu đề "BÁO CÁO KIỂM TRA SẢN XUẤT" + dòng phụ
     "Máy M2 · Line 3 · ORG CINNAMON CC", giữa là kỳ "7 ngày · 14/08–20/08 ·
     chia theo ngày", phải là khối meta "Lập lúc 20/08/2026 19:40 · OCR Datecode
     System" + ô trống đặt logo công ty
   - Dải 4 ô KPI: Tổng kiểm 5.258 · Đạt 3.651 · Không đạt 1.607 · Tỉ lệ đạt
     69,4% — ô tỉ lệ có màu trạng thái (xanh ≥95, vàng 85–95, đỏ <85) kèm chữ
     trạng thái bên dưới
   - Biểu đồ cột chồng pass/fail theo ngày, kèm đường tỉ lệ đạt trục phải
   - Bảng "Theo recipe": recipe · sản lượng · đạt · không đạt · tỉ lệ · trạng thái
   - Bảng "Theo camera": serial · sản lượng · đạt · tỉ lệ
   - Footer mảnh: OCR Datecode Inspection System · số trang

2. TRANG 2 — cùng theme
   - Khối "Theo ca": 3 hàng Ca A (06–14) / Ca B (14–22) / Ca C (22–06), mỗi hàng
     sản lượng + tỉ lệ + sparkline; ca chưa bắt đầu ghi "chưa bắt đầu" chứ không
     ghi 0
   - Khối "Nguyên nhân lỗi": thanh ngang tỉ trọng 4 nguyên nhân (ký tự dưới
     ngưỡng tin cậy / OCR đọc sai chuỗi / ảnh không khớp template / detector
     không thấy vùng), ghi rõ "trên mẫu 196 sản phẩm · lấy mẫu rải đều theo ngày"
   - Lưới 6 ảnh sản phẩm lỗi, mỗi ảnh caption 2 dòng: "mong BB/2609 → đọc
     BB/2G09" + giờ chụp + camera
   - Khối "So với kỳ trước": 3 dòng delta, viết dạng "▼ 8,2 điểm" (chữ "điểm",
     không dùng %), có ghi per-day khi hai kỳ khác độ dài

3. TRANG 1 lặp lại theme "Executive" — header gradient xanh #1e3a8a→#1d4ed8,
   nhãn KPI đặt trong chip xanh đậm chữ trắng, nền #eef2f7, sang trọng hơn để
   gửi khách hàng / cấp trên

Quy tắc cứng: giá trị không đo được hiện "—" chứ không hiện 0; mọi con số %
của mẫu phải ghi "của mẫu"; tiêu đề mục không được đứng mồ côi cuối trang.
```

### Prompt 2 — Báo cáo TOÀN NHÀ MÁY (máy tổng)

```
Thiết kế mẫu BÁO CÁO SO SÁNH NHIỀU DÂY CHUYỀN cho nhà máy đóng gói gia vị,
5 máy kiểm tra date code bằng OCR. Khổ A4 dọc, in PDF được, biểu đồ là ảnh
tĩnh. Tiếng Việt. Theme Industrial (trắng + navy #1e2a3a + accent #2563eb),
đồng bộ với mẫu báo cáo một máy.

3 artboard:

1. TRANG 1 — Tổng quan nhà máy
   - Header navy: "BÁO CÁO SO SÁNH DÂY CHUYỀN" · "5 máy · 7 ngày ·
     14/08–20/08" · meta giờ lập + chỗ logo
   - NGAY DƯỚI HEADER: banner phạm vi dữ liệu. Hai biến thể trong cùng artboard:
     (a) nền xanh nhạt "Đủ cả 5 máy" và (b) nền cam "KHÔNG đủ đội hình — thiếu
     LineTine (agent tắt lúc 14:22)". Banner (b) phải nổi bật, không được là
     chú thích nhỏ
   - Dải 4 KPI: Tổng sản lượng 117.443 · Đạt 112.109 · Không đạt 5.334 ·
     Tỉ lệ đạt chung 95,46%
   - Biểu đồ cột "sản phẩm / ngày" theo máy (đã chuẩn hoá per-day vì các máy
     chạy số ngày khác nhau), mỗi cột màu riêng cố định theo tên máy
   - Bảng máy: Máy · Line · Sản lượng · Mỗi ngày · Tỉ lệ đạt · Recipe đang
     chạy + hàng TỔNG. Dưới bảng in nghiêng dòng: "Không xếp hạng theo tỉ lệ
     đạt: các máy chạy recipe khác nhau, tỉ lệ phản ánh độ khó mặt hàng."

2. TRANG 2 — Vân tay kiểu lỗi (phần đinh của báo cáo)
   - Câu dẫn: "Tỉ trọng giữa các nguyên nhân trên mẫu fail của từng máy — các ô
     một máy cộng lại bằng 100. Đây là cách so sánh có nghĩa giữa các máy vì
     những nguyên nhân này thuộc pipeline OCR, không thuộc mặt hàng."
   - Biểu đồ cột chồng NGANG: mỗi máy một thanh 100%, 4 màu cho 4 nguyên nhân,
     chú giải dưới
   - Bảng nhiệt: hàng máy × cột nguyên nhân, ô lệch mạnh khỏi trung vị tô đậm;
     cột cuối "Mẫu": "196 · lấy mẫu" hoặc "96 · phủ hết kỳ"
   - Khung "Đọc thế nào": no_detection cao = camera/trigger/ánh sáng;
     char_verification cao = thấy vùng nhưng ký tự dưới ngưỡng — hai máy cùng
     'pass thấp' có thể hỏng hai thứ hoàn toàn khác nhau
   - Khối kết: 2–3 gạch đầu dòng phát hiện chính, mỗi dòng kèm hành động đề xuất

3. TRANG 3 — Phụ lục theo máy
   - 5 thẻ ngang, mỗi máy một thẻ: tên + line + model thiết bị, mini-bar
     pass/fail 7 ngày, 3 số nhỏ (sản lượng · mỗi ngày · tỉ lệ), recipe đang chạy
   - Thẻ của máy thiếu dữ liệu: nền xám, ghi lý do "agent tắt lúc 14:22 · máy
     vẫn sản xuất" — vẫn CÓ MẶT trong báo cáo, không bị bỏ ra
   - Footer: "Sinh bởi Fleet Service · dữ liệu đọc trực tiếp từ từng máy tại
     thời điểm lập báo cáo" + số trang

Quy tắc cứng: giá trị thiếu hiện "—"; tỉ trọng ghi "của mẫu"; máy lỗi vẫn xuất
hiện kèm lý do; delta điểm phần trăm viết chữ "điểm".
```

---

## Ghi chú triển khai sau khi có mẫu

1. **Một bộ style, hai đường xuất** — style mới phải vào được cả
   `reportGenerator.ts` (frontend) lẫn `styles.py` (agent), như bản gốc đã làm.
   Sửa một nơi rồi quên nơi kia là quay lại đúng vấn đề mà việc copy-nguyên-văn
   ban đầu đã giải.
2. Khối "Theo ca" và "ảnh sản phẩm lỗi" trong Prompt 1 cần **GĐ 1** (gộp theo
   ca + endpoint ảnh thu nhỏ) — mẫu cứ thiết kế trước, dữ liệu theo sau.
3. Banner thiếu máy có **hai biến thể ngay trong mẫu** để người duyệt thấy cả
   hai trạng thái — trạng thái xấu mới là trạng thái cần thiết kế kỹ.
4. Theme `dark` giữ cho bản xem trên màn hình; bản in mặc định `industrial`.
   In nền tối là đốt mực và mất tương phản trên giấy.
