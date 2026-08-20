# 05 — Agent UX: khi nào hỏi lại, khi nào không

## Quy tắc gốc

> **Hỏi lại đúng những gì còn thiếu. Không hỏi lại thứ người dùng đã nói.**

Nghe hiển nhiên, nhưng cả hai vế đều dễ hỏng theo hai hướng ngược nhau:

| Hỏng kiểu A — tự đoán | Hỏng kiểu B — hỏi thừa |
|---|---|
| Đặt `format="html"` làm mặc định ⇒ mô hình tự điền ⇒ câu hỏi **không bao giờ** tới tay người dùng, và họ nhận file sai định dạng | Người dùng đã nói "xuất PDF cho M1 và M2 tuần này" mà vẫn hỏi lại cả ba mục ⇒ phải trả lời lại thứ mình vừa nói |

Kiểu A đã xảy ra ở tầng edge và được ghi trong `PIPELINE.md §4`. Kiểu B là rủi ro
mới khi thêm picker.

## Cách hiện thực: tham số mặc định `None`, và chỉ hỏi phần thiếu

```python
async def generate_fleet_report(machines=None, period=None, format=None, **_ignored):
    missing = {}
    if not machines: missing["machines"] = {...}
    if not period:   missing["period"]   = {...}
    if not format:   missing["format"]   = {...}
    if missing:
        return {"ok": False, "ask_user": missing, ...}
```

Ba điểm bắt buộc:

1. **Mặc định là `None`, không phải giá trị hợp lệ.** Đặt bất kỳ giá trị mặc định
   nào là dạy mô hình rằng câu hỏi đó không cần hỏi.
2. **Chỉ những khoá còn thiếu mới vào `missing`.** Người dùng nói rõ 2/3 thì chỉ
   hỏi 1.
3. **`**_ignored`.** Một tham số bịa ra không được giết cả lượt chat — đã xảy ra
   thật khi mô tả tool nhắc tên khoá trong kết quả và mô hình tưởng đó là tham số.

## Bảng hành vi mong đợi

| Người dùng gõ | Hệ thống làm |
|---|---|
| "Xuất báo cáo" | Hỏi cả 3: máy nào, kỳ nào, định dạng nào |
| "Xuất báo cáo PDF" | Hỏi 2: máy nào, kỳ nào |
| "Xuất báo cáo M1 M2 tuần này" | Hỏi 1: định dạng nào |
| "Xuất PDF so sánh M1 M2 tuần này" | **Không hỏi gì**, xuất luôn |
| "Xuất báo cáo cho tất cả máy hôm nay ra excel" | **Không hỏi gì**, xuất luôn |

## Picker: hỏi bằng nút bấm, không bằng chữ

Yêu cầu rõ ràng của người dùng: *"xuất report hiện ra các danh sách máy để người
dùng click, và định dạng để click"*.

Nghĩa là `ask_user` **không được** để mô hình viết lại thành văn xuôi. Nó phải đi
thẳng ra giao diện dưới dạng dữ liệu có cấu trúc:

```json
{
  "ask_user": {
    "machines": {
      "prompt": "Báo cáo gồm những máy nào?",
      "type": "multi",
      "options": [
        {"value": "Auto2", "label": "Auto2", "hint": "Line 1 · ONION POWDER"},
        {"value": "M1", "label": "M1", "hint": "Line 2 · CHILI Pdr"}
      ],
      "quick": ["Tất cả"]
    },
    "period": {"prompt": "...", "type": "single", "options": [...]},
    "format": {"prompt": "...", "type": "single",
               "options": [{"value":"pdf","label":"PDF","hint":"để in / gửi mail"},
                           {"value":"excel","label":"Excel","hint":"để lọc, tính thêm"}]}
  }
}
```

Giao diện render thành:

```
┌─ Xuất báo cáo so sánh ───────────────────────────┐
│ Chọn máy                          [Tất cả]       │
│ ☑ Auto2   ☑ M1   ☐ M2   ☐ LineTine  ☐ PC-Auto-1  │
│   Line 1    Line 2                               │
│                                                  │
│ Kỳ báo cáo                                       │
│ ( )Hôm nay ( )Hôm qua (•)7 ngày ( )30 ngày       │
│                                                  │
│ Định dạng                                        │
│ [ HTML ] [ PDF ] [ Excel ] [ CSV ]               │
│   xem nhanh  in/mail  lọc thêm  đưa hệ khác      │
│                                                  │
│                        [Huỷ]  [Xuất báo cáo]     │
└──────────────────────────────────────────────────┘
```

Mỗi lựa chọn có **`hint` giải thích hệ quả**, không chỉ tên. "PDF — để in / gửi
mail" hữu ích hơn "PDF" với người chưa từng dùng.

Người dùng bấm xong → giao diện gọi lại chat với đủ tham số → agent xuất luôn,
không hỏi thêm.

## Gợi ý câu hỏi tiếp theo (3–5 câu)

**Do code dựng từ số liệu, không do mô hình viết.** Lý do đã ghi ở
`core/suggestions.py` và ở tầng edge: mô hình viết gợi ý mà không nhìn con số, nên
sau câu "máy nào tệ nhất" nó mời "xem lịch sử load recipe" — chẳng dính gì tới
thứ vừa hiện trên màn hình.

Thứ tự ưu tiên:

1. **Máy thiếu dữ liệu** — đẩy lên đầu. Đây là thứ dễ trôi qua nhất vì bảng vẫn
   hiện đầy đủ các máy còn lại.
2. **Outlier vừa tìm được** — "Vì sao M2 có 'ký tự dưới ngưỡng' cao hơn hẳn?"
   Outlier đo bằng khoảng cách tới **trung vị**, không phải tới giá trị lớn nhất:
   đo với max thì máy cao nhất luôn tự nó là outlier, kể cả khi cả 5 máy sát nhau.
3. **Bước tiếp hợp lý của việc vừa làm** — vừa xuất PDF thì mời Excel, **không**
   mời PDF lần nữa.
4. **Ngưỡng vừa vượt** — "Đĩa M1 còn 50 GB, nên dọn gì?"
5. Câu chung, chỉ khi không suy được gì.

## Ngữ cảnh gắn sẵn

Chat mở từ một ngữ cảnh cụ thể thì mang theo ngữ cảnh đó, người dùng không phải
gõ lại:

| Mở từ | Ngữ cảnh gắn sẵn |
|---|---|
| Nút trong ngăn kéo máy | máy đó |
| Một hàng trong nhật ký thao tác | user + máy + thời điểm |
| Một ảnh sản phẩm lỗi | máy + recipe + giờ |
| Line Station | luôn là máy đó, không đổi được |

Ngữ cảnh hiện thành một chip nhỏ phía trên ô nhập ("đang hỏi về **M2**"), gỡ được.
Gắn ngầm mà không hiện ra thì người dùng không hiểu vì sao câu trả lời chỉ nói về
một máy.

## Trạng thái chờ

Đo được: hỏi agent mất 4–20s, có lúc 27s. Ô chat phải nói rõ đang làm gì, không
chỉ quay vòng:

```
đang hỏi M2…                    ← khi ủy quyền
đang đọc số liệu 5 máy…         ← khi fan-out
đang dựng báo cáo PDF…          ← khi xuất file
```

Tầng edge đã có SSE báo tiến trình. Tầng fleet nên đi cùng đường đó ở giai đoạn
sau; trước mắt hiện nhãn theo tool đang chạy là đủ.

## Ba thứ tuyệt đối không để mô hình làm

| Không | Vì sao |
|---|---|
| Viết đường dẫn tải file | Đã xảy ra thật: mô hình bịa `sandbox:/fleet_….pdf`. Tên file phải **ra khỏi tầm nhìn** của mô hình, link do server gắn |
| Tự chọn máy / kỳ / định dạng | Câu hỏi sẽ không bao giờ tới người dùng |
| Nhắc lại số từ trí nhớ | Số đi qua trường có cấu trúc, giao diện tự dựng bảng |
