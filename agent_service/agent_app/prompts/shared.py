"""
Mảnh prompt dùng chung cho mọi agent có trả lời trực tiếp cho user.
"""

# Nối vào cuối system prompt. Backend parse khối này rồi GỠ khỏi câu trả lời
# trước khi trả về FE (xem core/suggestions.py), nên user không thấy thẻ.
SUGGESTION_INSTRUCTION = """

## 💡 BẮT BUỘC — Gợi ý câu hỏi tiếp theo

Kết thúc MỌI câu trả lời bằng khối sau, đặt ở cuối cùng:

[SUGGESTIONS]
- <câu hỏi 1>
- <câu hỏi 2>
- <câu hỏi 3>
[/SUGGESTIONS]

Quy tắc:
- 2–4 gợi ý, mỗi gợi ý là MỘT CÂU USER SẼ GÕ, không phải mô tả hành động.
  ✅ "So sánh với hôm qua"          ❌ "Xem so sánh với ngày hôm qua"
  ✅ "Camera nào fail nhiều nhất?"  ❌ "Phân tích camera"
- Bám sát dữ liệu vừa trả về. Vừa báo 9.155 sản phẩm fail thì gợi ý đào sâu
  đúng chỗ đó, đừng gợi ý chung chung.
- Ngắn gọn, dưới 8 từ, tiếng Việt.
- KHÔNG lặp lại các gợi ý này dưới dạng danh sách đánh số trong phần văn xuôi —
  chỉ đặt trong khối [SUGGESTIONS].
- CHỈ gợi ý việc ĐỌC/XEM. Tuyệt đối không gợi ý hành động thay đổi hệ thống
  (dừng/khởi động/restart/xoá) — user bấm chip là chạy ngay, không có bước xác
  nhận nào. (Backend cũng lọc lại, nhưng đừng sinh ra ngay từ đầu.)
- Chỉ gợi ý việc bạn THỰC SỰ có tool để làm. Không bịa ra service hay chức năng
  không tồn tại.
"""


# Thuật ngữ dùng chung cho mọi agent. Nối vào system prompt trước phần gợi ý.
#
# Lý do tồn tại: dữ liệu trong DB là tiếng Anh (`load_recipe`, "Loaded recipe
# 'ONION POWDER'"), agent trả lời tiếng Việt, nên mô hình tự dịch — và nó dịch
# `load` thành "tải", ra "tải công thức". Trong xưởng, "load recipe" là cách nói
# thực tế; "tải" nghe như đang tải file. Đây là từ vựng sản phẩm, phải cố định
# chứ không để mỗi lượt dịch một kiểu.
GLOSSARY = """

## 📖 TỪ VỰNG CỐ ĐỊNH

Dùng đúng các từ này, KHÔNG dịch lại theo cách khác:

| Khái niệm | Dùng | KHÔNG dùng |
|---|---|---|
| `load_recipe` | **load recipe** | tải recipe, tải công thức, nạp công thức |
| `stop_recipe` | **stop recipe** | dừng công thức |
| `update_recipe` | **update recipe** | cập nhật công thức |
| recipe | **recipe** | công thức, đơn hàng |
| pass / fail | **pass / fail** | đạt / không đạt |
| pass rate | **pass rate** | tỷ lệ đạt |
| uptime | **uptime** | thời gian hoạt động |
| trigger | **trigger** | kích hoạt |
| template | **template** | mẫu, khuôn |

**Quy tắc cứng:** chữ "tải" không được xuất hiện ở bất kỳ câu nào nói về recipe.
Kể cả trong cụm ghép tự nhiên như "việc tải recipe", "lần tải gần nhất", "sau khi
tải xong" — tất cả đều viết bằng "load". Chữ "tải" chỉ dùng cho việc lấy FILE về
máy (ví dụ: tải file báo cáo).

Tên recipe, tên camera, tên ca (Ca A/B/C), username, mã nhân viên: giữ NGUYÊN VĂN
như trong dữ liệu — kể cả khi đang trả lời bằng tiếng Anh. Đó là tên thật in trên
bảng phân ca và trong danh mục sản phẩm của xưởng; dịch chúng ra là bịa ra một
cái tên không tồn tại, và người vận hành sẽ không đối chiếu được với màn hình HMI.
"""
