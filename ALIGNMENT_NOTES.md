# Label Misalignment Detection — Notes

## Vấn đề gốc rễ

Factory reject ~500 sản phẩm/giờ do `check_center_alignment` false positive.

**Nguyên nhân:**
- `_check_center_alignment` dùng YOLO OBB để detect vùng chai (product box)
- YOLO OBB train không đủ — nhà máy có 200+ loại chai, không thể lấy mẫu train hết
- YOLO detect sai → center sai → reject nhầm

---

## Luồng thực thi hiện tại

```
single_camera.py → verify_batch(frames_data)
                       ↓
              _process_single_frame()
                  ├── _batch_wrinkle_check()   ✅ đang dùng (segmentation model)
                  └── _check_center_alignment() ⚠️ đang dùng YOLO OBB (không ổn)
```

**Check đang bật:** `check_center_alignment=True`, `check_wrinkled=True`  
**Check đang tắt:** `check_rotation`, `check_misalignment`, `check_label_boundary`

---

## Giải pháp đang nghiên cứu: Edge Detection

Dùng 2 cạnh trắng bên trong vỏ nhựa trong suốt của chai làm reference thay cho YOLO.

**Approach đã validate:**
- Sobel X → profile 1D → `scipy.signal.find_peaks`
- Search trong **outer 30%** của crop (trái/phải) → tránh bắt nhầm edge của label/barcode ở giữa
- Width std ≈ 9–12px, 100/100 ảnh detect thành công

**Vấn đề crop drift:**
- `transformed_bboxes['product']` bị SuperPoint kéo theo nhãn
- Khi nhãn lệch → crop dịch → mất cạnh chai trong ảnh
- Đang nghiên cứu dùng phase correlation (template vs live crop) để bù lại shift

**Phase correlation kết quả:**
- 40767171: shift_x std=1.4px (tốt), response 0.3–0.7
- 40733814: shift_x std=0.2px (rất tốt), response thấp 0.1–0.3
- Vấn đề: label content thay đổi (datecode) → response không ổn định

---

## Những gì đã sửa trong code

### `factory.py` — `_save_template_sample()`
Khi init matcher, tự động lưu vào `crop_samples/{serial}/`:
- `template.jpg` — ảnh template đã crop đến product bbox
- `template_product_bbox.json` — toạ độ product bbox (`points`, `x_min/max`, `width`, `height`)

### `product_verifier.py` — `_save_crop_sample()`
Trước `_check_center_alignment`, lưu mỗi frame vào `crop_samples/{serial}/`:
- `crop_XXXX_TIMESTAMP.jpg` — ảnh crop từ product bbox (margin=0)
- `crop_XXXX_TIMESTAMP.json` — metadata gồm:
  - `crop_offset`: `[x1, y1]` offset trong frame gốc
  - `product_points`: product bbox trong frame space
  - `label_points`: label bbox **trong crop space** (đã trừ offset)
  - `label_points_frame`: label bbox trong frame space

Giới hạn 100 ảnh/camera (`crop_sample_max=100`). Tắt bằng `save_crop_samples = False`.

---

## Test script

**`test_alignment.py`** — chạy offline trên crop samples đã lưu:

```bash
python3 /home/demo/Source/ocr_datecode/test_alignment.py
```

Output: `crop_samples/align_vis/{serial}/` — 3 panel ghép ngang:
- **RAW**: ảnh gốc + label bbox (cyan)
- **TEMPLATE**: template crop + bottle walls (xanh lá)
- **ANNOTATED**: bottle center từ phase correlation (xanh lam) + label center (đỏ) + walls (xanh lá) + `mis=` + `E=` + `resp=`

---

---

## Phân tích crop drift + thuật toán mới đề xuất

### Quan sát thực tế (từ ảnh)

Chai nhựa trong suốt mỗi bên có **2 cạnh trắng**:
- Cạnh ngoài = mặt sau chai nhìn xuyên qua nhựa
- Cạnh trong = thành chai thực sự (inner wall) ← cái cần detect

Thuật toán hiện tại lấy peak mạnh nhất outer 30% → đang lấy **cạnh ngoài**, chưa đúng.

### Vấn đề khi nhãn lệch

```
Nhãn lệch phải → product bbox dịch phải → crop dịch phải
→ Cạnh TRÁI của chai bị cắt khỏi crop
→ Chỉ còn thấy 2 cạnh bên PHẢI
→ Thuật toán search outer-30% bên trái không tìm được gì (hoặc bắt nhầm)
→ Kết quả sai

Nhãn lệch trái → ngược lại → mất cạnh phải
```

### Giải pháp đề xuất: extrapolate từ cạnh còn lại + template width

**Ý tưởng:** Nếu chỉ tìm được 1 cạnh (bên không bị cắt), dùng template inner width để suy ra cạnh còn lại.

**Công thức:**
```
# Tìm inner wall trong template 1 lần → template_inner_width
# (detect bằng Sobel trên template crop, lấy cạnh thứ 2 từ mỗi bên)

# Target crop: chỉ thấy left inner wall tại lx
right_expected = lx + template_inner_width

# Nếu right_expected > crop_width:
#   right wall nằm ngoài crop → tọa độ trong frame = crop_x2_frame + (right_expected - crop_width)

# Bottle center trong frame = crop_offset_x + lx + template_inner_width / 2
# Misalignment = label_center_frame - bottle_center_frame
```

**Ưu điểm:**
- Không cần cả 2 cạnh cùng visible
- Dùng template inner width (đo 1 lần, stable) thay vì template product bbox width (outer-to-outer)
- Không cần YOLO, không cần phase correlation

### ⚠️ Câu hỏi chưa rõ

**Template product bbox được vẽ theo outer walls hay inner walls?**
- Hiện tại `template_product_bbox.json` lấy từ annotation do operator vẽ trên UI
- Nếu operator vẽ theo **outer walls**: template width = outer-to-outer → cần tính thêm plastic thickness
- Nếu operator vẽ theo **inner walls**: dùng trực tiếp được

→ Cần xác nhận trước khi implement. Hoặc tính `template_inner_width` bằng cách detect edge trực tiếp trên `template.jpg` (không phụ thuộc annotation).

---

## Việc cần làm tiếp

- [ ] Xác nhận: template product bbox vẽ theo outer hay inner wall?
- [ ] Detect inner walls trên template.jpg → lấy `template_inner_width` thực tế
- [ ] Implement thuật toán extrapolate: tìm 1 cạnh → suy ra cạnh kia → tính center
- [ ] Collect thêm ảnh có nhãn **thực sự lệch** để validate
- [ ] Implement vào `_check_center_alignment` thay thế YOLO product_box
- [ ] Bỏ dependency YOLO OBB khỏi center alignment check



 "  đúng, còn vùng product (product transformer box) thì nhảy bị sai nên khiến ảnh vùng product được crop ra bị dịch theo label   
                                                                                                                                   
                                                                                                                                   
                                                                                                                                   
    tôi có suy nghĩ như sau, khi mà vùng chai bị dịch theo nhãn thì có thể lúc crop ra theo product box nó có thể bị mất cạnh      
                                                                                                                                   
    của product nên lúc tìm cạnh sẽ bị sai, tôi ví dụ nó crop ra khiến mất cạnh bên trái (nhãn lệch sang phải nhưng không đáng kể, 
                                                                                                                                   
    nhãn lệch chưa chạm mép chai) -> ko bắt. Rồi tình huống lệch phải cũng vậy.                                                    
                                                                                                                                   
    Tôi quan sát thấy tình huống nó nếu nhãn lệch phải crop ra thì chạy trái của chai lúc mà thuật toán chạy dường như là cạnh     
  ngoài                                                                                                                            
     cùng bên phải, nhưng chả có cạnh nào ở đó cả vì nhãn nó lệch crop bị sai                                                      
                                                                                                                                   
                                                                                                                                   
                                                                                                                                   
    Ngược lại bên trái nó sẽ thấy được 2 cạnh luôn và tôi quan sát là nó đang lấy cạnh ngoài cùng, theo cơ chế của [Image #6]      
                                                                                                                                   
    /home/demo/Source/ocr_datecode/crop_samples/40733814/template_product_bbox.json                                                
                                                                                                                                   
    nó sẽ dựa vào ảnh template và toạ độ product bbox để dịch phải ko, vì vậy nó detect sai và lấy cạnh thứ 1 nên lúc nó dịch chưa 
                                                                                                                                   
    đúng. Theo lý thuyết nó lấy  đc cạnh thứ 2 bên trái thì lúc tìm cạnh bên phải nó dịch theo và toạ độ phải thường hơn toạ độ    
  x2                                                                                                                               
    vượt ra bên ngoài. Thế thì ta lấy x2 đó + với đoạn chêch lệch. như product template width là 100, tuy nhiên ta tìm đc cạnh     
  trái                                                                                                                             
    là x1 = 20 của target, lẽ thường tình là 20 + 100 -> sẽ ra 120. nhưng ảnh target chỉ là 110 thôi vì đây là ảnh crop nên ta +   
                                                                                                                                   
    thêm 10 nữa để quy đổi ra ảnh ko phải crop của product bbox..                                                                  
                                                                                                                                   
                                                                                                                                                                                                                                     
    Hãy phân tích và đặt câu hỏi nếu chưa rõ vấn đề   vấn đề ngược lại khi mầ lệch phải "  