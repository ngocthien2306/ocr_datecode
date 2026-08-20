# Agent service — quá trình phát triển

Ghi lại toàn bộ mạch công việc trên `agent_service/` trong ngày 19/08/2026, theo
đúng thứ tự đã làm. Mục đích không phải liệt kê tính năng — phần đó đọc code là
biết — mà là ghi lại **vì sao** từng thứ được làm như vậy, đặc biệt là những chỗ
cách làm đầu tiên đã thất bại. Người đọc sau sẽ khỏi phải mắc lại.

Nguyên tắc xuyên suốt: `agent_service/` **tự chứa**. Nó không import từ
`backend/app/`, không gọi endpoint của backend. Cần chức năng gì đã có ở backend
hay frontend thì port sang, chấp nhận trùng lặp, để hai codebase không đụng nhau.
Ngoại lệ duy nhất là `core/backend_client.py`, hỏi backend qua HTTP xem WebSocket
của camera service có đang kết nối — trạng thái đó là singleton trong tiến trình
backend, không có đường nào khác đọc được.

---

## 1. Port service sang `release_v1`, dựng chạy được

Copy `agent_service/` từ `release_v2`. Chạy lên thì 401 mọi request: `SECRET_KEY`
trong `agent_service/.env` còn là chuỗi placeholder, không trùng `backend/.env`.
Hai bên phải cùng khoá vì agent service **tự verify** JWT do backend phát, chứ
không hỏi backend xem token có hợp lệ.

Thêm `.env.bak` vào `.gitignore` — file đó chứa `OPENAI_API_KEY`.

## 2. Truy vấn analytics: 198s → 0,2s

Câu hỏi "thống kê 7 ngày qua" mất 198 giây. Nguyên nhân: các tool filter theo
`timestamp`, nhưng collection `inference_results` chỉ có index bao phủ trên
`{created_at, product_pass_fail, recipe_id, recipe_name}`. `timestamp` có index
riêng `{timestamp: -1}` nhưng không bao phủ, nên mỗi bản ghi phải nạp cả document
— mà document ở đây **trung bình 62,8 KB**, do
`char_verification.results[].mask_diff_b64` nhúng một ảnh PNG base64 cho từng ký
tự. Cả collection 13,5 GB.

Đổi sang filter `created_at`: 18,82s → 0,20s trên cùng câu hỏi. Với các tool bắt
buộc phải đọc document (như `explain_failures`), thêm `_FAIL_PROJECTION` loại bỏ
`mask_diff_b64` ngay ở tầng projection.

```python
_TIME_FIELD = "created_at"   # được index bao phủ; timestamp thì không
_FAIL_PROJECTION = {"camera_results.frames.char_verification.results.mask_diff_b64": 0}
```

## 3. Một lớp bug "trả lời sai mà nghe rất hợp lý"

Đây là phần chiếm nhiều thời gian nhất, và là phần đáng đọc nhất. Điểm chung của
cả nhóm: con số hiện ra **đúng về mặt tính toán** nhưng **sai về mặt ý nghĩa**, nên
không có test nào bắt được, và người đọc không có cách nào biết là mình đang bị
lừa.

| Hiện tượng | Thực chất | Cách sửa |
|---|---|---|
| "328 sản phẩm fail" trong khi tổng fail là 167 | đếm theo FRAME, trình bày như SẢN PHẨM | mỗi hàng nguyên nhân mang cả `products` và `frames` |
| "Chưa xác định: 78" | là frame có `detected_regions: []` | đặt tên đúng: `no_detection` |
| Biểu đồ giờ thiếu 12h–19h, gồm cả 14h là giờ tệ nhất | `_bar` cắt còn 12 cột, im lặng | nâng lên 26 cột, và **ghi vào tiêu đề** đã gộp bao nhiêu |
| Hỏi 7 ngày, thực chất chỉ soi 1,5 ngày | `explain_failures` lấy N bản ghi MỚI NHẤT | chia mẫu đều theo từng ngày — làm xong thì thứ hạng nguyên nhân đổi hẳn |
| "tổng fail 1.036" rồi liệt kê "144/106/102" | ba số kia là của mẫu 294, không phải của kỳ | mô hình chỉ còn nhận `percent_of_sample`, không nhận số đếm |
| "+3.309.850%" | nền so sánh có 2 bản ghi | `MIN_BASELINE = 30`, dưới ngưỡng thì không đưa delta |
| "tăng 59,75%" khi so tháng này (19 ngày) với tháng trước (31 ngày) | thực tế per-day tăng 160% | thêm `same_length`, số ngày, và `per_day` |
| "+1,36đ" dưới ô pass rate | đọc thành 1,36 **đồng** | viết đủ chữ "điểm"; thêm `delta_kind='pp'` |
| recipe không chạy kỳ này hiện −100% | `only_in` chỉ null pass_rate, còn để volume | null cả hai; giá trị vắng hiện "—" chứ không phải 0 |
| Ca B báo cáo cảnh báo 10:10 của ca A | cảnh báo phạm vi ngày bị trộn vào phạm vi ca | tách `equipment_alerts` với `day_wide_alerts` |
| Ca chưa bắt đầu báo "0 sản phẩm, pass 0%" | cửa sổ `22:00 → 19:18` | thêm `not_started` và `previous_occurrence` có ghi ngày |
| "Hôm nay bao nhiêu người đăng nhập?" → "📭 Không có dữ liệu" | route sai vào historical_analytics, agent đó không có tool nào về đăng nhập | siết keyword; và **rào** template "không có dữ liệu" |
| Thẻ ghi "14:12 → 14:23" trong khi bản ghi đầu là 11:05 | thống kê tính từ `entries` đã bị `limit` cắt | `$group` trên toàn bộ match |

### Cách chữa gốc: giữ dữ liệu khỏi mô hình, thay vì dặn mô hình

Nhiều lỗi trên có dạng "mô hình nói một con số nó không nên nói". Cách chữa đầu
tiên luôn là thêm chỉ dẫn vào prompt — và cách đó **thất bại lặp lại**. Ví dụ rõ
nhất: mô hình bịa ra `https://example.com/api/reports/...`. Tôi thay bằng một
chuỗi placeholder; nó nhúng luôn placeholder vào markdown. Chỉ khi **xoá hẳn**
`download_url` và `filename` khỏi tầm nhìn của mô hình thì vấn đề mới hết.

Từ đó thành `strip_for_llm()`: kết quả tool bị lược trước khi vào mô hình. Con số
vẫn đến được người dùng, nhưng qua **attachment dựng bằng code** (`kpis`,
`charts`, `tables`, `cards`, `files`), nên số hiện ra không thể lệch khỏi số trong
DB. Cùng lúc nó giải luôn bài toán mô hình nhắc lại y nguyên bảng số trong văn
xuôi — không phải bằng cách cấm, mà bằng cách không đưa.

Tương tự với cách gọi tên: cấm cụm "recipe mới" thì nó chuyển sang "sản phẩm
mới". Chỉ hết khi đổi tên khái niệm trong dữ liệu: `only_in` → `absent_from`.

## 4. Xuất báo cáo, không cần trình duyệt

Port `reportGenerator.ts` của frontend sang `agent_app/reports/`. Bản HTML dùng
Chart.js vẽ trên `<canvas>` — cần một trình duyệt chạy JavaScript. WeasyPrint
(thứ dựng PDF ở đây) chỉ hiểu HTML/CSS, canvas sẽ ra trắng trơn; máy này không có
chromium/wkhtmltopdf, và cài thêm một trình duyệt lên Jetson đang chạy inference
là cái giá quá đắt cho việc xuất báo cáo.

Nên biểu đồ được vẽ sẵn thành PNG bằng matplotlib (`reports/charts_png.py`) rồi
nhúng base64 vào chính template đó. Excel/CSV qua openpyxl (`reports/tabular.py`).

Về luồng hỏi: `generate_report` **không tự chọn** định dạng hay kỳ. Tham số để
`None` trong args schema và trả về danh sách `options` để người dùng bấm. Đặt
default `format: str = "html"` là dạy mô hình tự điền, và câu hỏi không bao giờ
tới tay người dùng.

Một lỗi đáng nhớ: tool crash cả lượt chat với
`generate_report() got an unexpected keyword argument 'needs_period_choice'` —
description của tool có nhắc tên các khoá trong response, mô hình tưởng đó là
tham số. Sửa bằng `**_ignored` và bỏ tên khoá khỏi description.

## 5. Thêm agent, và bản giao ca

- `equipment_health` (`tools/equipment_tools.py`): `check_reject_timing`,
  `check_trigger_health`, `check_sensor_pulse`, `check_subsystem_health` — đọc
  bốn log có cấu trúc bằng regex, dùng **median** chứ không phải mean vì một lần
  dừng máy đủ kéo lệch mean.
- `get_shift_handover`: một lời gọi gộp sản lượng + chỉ tiêu + dừng máy + nguyên
  nhân fail + cảnh báo thiết bị + người trong ca.
- `get_target_progress` + `config/production_targets.json` (đọc lại mỗi lần gọi).
  Trước đó câu "đạt mục tiêu chưa" được mô hình trả lời "đã đạt được mục tiêu"
  trong khi hệ thống **không có** mục tiêu nào. Nay chưa cấu hình thì tool từ chối
  trả lời.

Ba phát hiện trên dây chuyền thật, ngoài phạm vi agent, đã báo lại:
`weights/best_bottle_m.engine` mất khiến `obb_rotation` không khởi tạo được từ
10:09; xung reject cấu hình 50 ms nhưng đo được median 255,5 ms trên 257/257 lần
(đã nêu để kỹ thuật viên xác nhận, **không** kết luận là lỗi); `logs/` 1,4 GB
trong đó 1,1 GB nằm ngoài chính sách dọn.

## 6. Dữ liệu demo: user, hồ sơ, ảnh, hoạt động theo ca

Xem `docs/DEMO_DATA.md`. Ràng buộc quan trọng: **không** chạm `load_recipe` /
`stop_recipe` / recipe / camera, vì ONION POWDER đang chạy thật trên dây chuyền.
Chỉ dùng nhóm thao tác về user và đăng nhập/đăng xuất.

## 7. Đa ngôn ngữ (phần cuối, `core/i18n.py`)

Điểm mở đầu là một lỗi từ vựng: DB lưu tiếng Anh (`load_recipe`,
`"Loaded recipe 'ONION POWDER'"`), agent trả lời tiếng Việt, nên mô hình **tự
dịch** và chọn "tải" — ra "tải công thức". Mỗi lượt lại dịch một kiểu. Sửa bằng
bảng **TỪ VỰNG CỐ ĐỊNH** trong `prompts/shared.py`, nối vào system prompt của mọi
agent.

Bảng cấm/cho phép một mình **chưa đủ**: mô hình vẫn viết "việc tải recipe", vì
"tải" đi vào cụm ghép rất tự nhiên. Phải thêm một quy tắc cứng — chữ "tải" không
được xuất hiện trong bất kỳ câu nào về recipe, và chỉ dùng cho việc lấy file về
máy.

Việc đa ngôn ngữ rẻ hơn tưởng tượng, vì chuỗi tiếng Việt trong service chia làm
hai loại rất khác nhau:

| Loại | Số lượng | Cần dịch? |
|---|---|---|
| Chỉ dẫn cho mô hình (prompt, docstring, `note` của tool) | ~1.150 | **Không.** Mô hình đọc tiếng Việt, trả lời tiếng Anh bình thường |
| Nhãn do CODE sinh (ô KPI, cột bảng, tiêu đề, chip gợi ý) | 174 | **Có.** Không đi qua mô hình nên mô hình không dịch giúp được |

Bảng dịch lấy **chính chuỗi tiếng Việt làm khoá**, kiểu gettext, vì mã nguồn đã
viết bằng tiếng Việt: bọc `t()` vào là xong, không phải bịa tên khoá, và thiếu
bản dịch thì trả về nguyên tiếng Việt — nhãn cũ vẫn đọc được, không bao giờ ra ô
trống. Ngôn ngữ giữ trong `ContextVar`: nhãn được dựng tận trong vòng lặp của
từng agent, truyền tham số xuống tới đó phải sửa hàng chục chữ ký hàm, mà mỗi
request là một task asyncio riêng nên ContextVar đã cách ly đúng theo request.

Việc dịch nằm **bên trong** `_tile()` và `_bar()`, không ở chỗ gọi — có hơn 30
lời gọi `_tile()` rải khắp file, bọc từng cái thì chắc chắn bỏ sót.

### Hai chỗ cách làm đầu tiên KHÔNG chạy

**Đặt chỉ dẫn ngôn ngữ ở cuối prompt thì bị lấn át hoàn toàn.** Prompt của agent
historical dài 18.000 ký tự toàn tiếng Việt, kèm hàng chục câu trả lời mẫu bằng
tiếng Việt. Mô hình bắt chước ngôn ngữ của ví dụ chứ không nghe một dòng ở cuối —
chọn English mà vẫn trả lời tiếng Việt. `apply_language()` đặt chỉ dẫn ở **cả đầu
và cuối**, và nói thẳng rằng phần tiếng Việt bên dưới là chỉ dẫn nghiệp vụ, không
phải mẫu ngôn ngữ để bắt chước.

**Chế độ `auto` giao cho mô hình tự bám ngôn ngữ cũng thất bại**, vì cùng lý do.
Và kể cả nếu nó làm đúng thì nhãn UI vẫn không biết theo ngôn ngữ nào — ra cảnh
câu trả lời tiếng Anh nằm cạnh ô KPI tiếng Việt. Nên `auto` được giải **ở phía
code** bằng `detect()`: có dấu thanh tiếng Việt là chắc chắn tiếng Việt, gõ không
dấu thì tra danh sách từ công cụ không dấu. Từ đó `auto` trở thành một ngôn ngữ cụ
thể, prompt và nhãn dùng chung một thứ tiếng.

### Những chỗ khác phải chạm tới

- **Từ khoá định tuyến** của orchestrator trước đó chỉ có tiếng Việt, nên câu hỏi
  tiếng Anh không khớp keyword nào. Thêm bộ tiếng Anh cho cả bốn agent, kèm bảng
  những câu dễ route sai — đáng chú ý nhất là "How many users logged in today?",
  đúng cái lỗi đã gặp ở bản tiếng Việt; thêm keyword mà không thêm cảnh báo thì
  lỗi cũ quay lại nguyên vẹn qua cửa tiếng Anh.
- **Tên nguyên nhân fail** được dịch ngay trong kết quả tool, không chỉ ở nhãn
  biểu đồ: mô hình đọc trường đó rồi nhắc lại trong văn xuôi, để nguyên tiếng
  Việt thì tên trong văn xuôi khác tên trên biểu đồ ngay bên cạnh.
- **Nhãn kỳ** ("tuần này", "so với") do tool sinh và nằm lẫn giữa chuỗi tự do nên
  tra cả chuỗi không khớp. `tphrase()` thay theo cụm con, duyệt dài-trước-ngắn để
  "7 ngày qua" không bị "ngày" ăn mất một nửa. Bảng cụm cố tình để nhỏ và chỉ gồm
  cụm thời gian: thay cụm con trên bảng lớn sẽ đụng vào dữ liệu thật.
- **File xuất ra** trước đây HTML/PDF toàn tiếng Anh còn Excel/CSV toàn tiếng
  Việt — cùng một lệnh ra hai file hai thứ tiếng. Nay cả hai theo lựa chọn.
- **Nút tải file** ghép cứng "Báo cáo" với nhãn kỳ tiếng Anh, ra "Báo cáo Today".
  `treport_period()` dịch ngược nhãn kỳ.

### Giữ nguyên, không dịch

Tên recipe, tên camera, username, tên người, và **tên ca** (`SHIFTS` trong
`analytics_tools.py`: Ca A/B/C). Đó là tên thật trên bảng phân ca và trong danh
mục sản phẩm của xưởng; dịch thành "Shift B" là bịa ra một cái tên người vận hành
không đối chiếu được với màn hình HMI.

*(Lưu ý dễ nhầm: field `shift` trong hồ sơ **user** thì đã được đổi sang tiếng
Anh — xem `scripts/demo/to_english.py`. Đó là dữ liệu demo, khác với `SHIFTS` là
định nghĩa ca của hệ thống.)*

### Cách dùng

Dropdown trên `/test`, hoặc trường `language` trong `POST /api/agent/chat`:

| `language` | Hành vi |
|---|---|
| bỏ trống | tiếng Việt (client cũ không cần sửa gì) |
| `"vi"` / `"en"` | hỏi tiếng nào cũng trả lời bằng ngôn ngữ đã chọn |
| `"auto"` | trả lời đúng thứ tiếng user vừa gõ |

Thêm ngôn ngữ thứ ba: thêm một dict vào `_TABLES` và một dòng vào `LANGUAGES`,
khoảng 140 chuỗi. Nhưng `detect()` hiện chỉ phân biệt vi/en, nên ngôn ngữ thứ ba
phải chọn cứng trong dropdown.

### Kiểm chứng

Chạy thật trên service, không phải unit test:

- 15/15 câu hỏi tiếng Anh vào đúng tool, gồm cả câu bẫy về đăng nhập.
- 5/5 tổ hợp ngôn ngữ đúng: hỏi Việt/chốt Việt, hỏi Việt/chốt Anh, hỏi Anh/chốt
  Anh, `auto`+Anh, `auto`+Việt.
- 10/10 câu thử `detect()` đúng, gồm cả trường hợp gõ không dấu.
- Hồi quy 6 câu tiếng Việt: nhãn và câu trả lời y như trước.
- Xuất CSV thật ở cả hai ngôn ngữ rồi đọc lại đầu file.
- 0 vi phạm quy tắc "tải" trên 3 câu về recipe.


## 8. Định tuyến sai, và đường ra cho người dùng

Người dùng gặp một chuỗi hai câu hỏi làm lộ ra vấn đề rõ nhất trong ngày:

1. "Từ 16h đến 18h camera nào fail nhiều nhất?" → đúng, ra 5 sản phẩm fail.
2. "Show 5 sản phẩm lỗi đó từ 16h đến 18h" → route sang `log_analysis`, gọi
   `search_logs`, không thấy dòng log nào, trả lời **"không có sự cố nào được ghi
   lại"**. Năm sản phẩm fail đó có thật, có cả ảnh, và nằm trong MongoDB.

Đáng chú ý: **ngữ cảnh hội thoại vẫn hoạt động bình thường** — câu 2 đã tự lấy
`camera 40762191` và khung giờ 16–18h từ câu 1. Lỗi không nằm ở việc quên ngữ
cảnh, mà ở việc mang đúng ngữ cảnh đó đến sai agent.

Ba việc đã làm:

**Rào kết luận ngay trong dữ liệu.** `search_logs` khi rỗng từng ghi note "Đã quét
hết các file trong phạm vi yêu cầu" — đọc lên đúng như một lời xác nhận không có
gì. Nay note nói thẳng đây là kết quả tìm trong FILE LOG, không cho biết gì về
việc sản phẩm có fail hay không, cấm kết luận "không có lỗi", và chỉ sang
`explain_failures`. Rào bằng dữ liệu hiệu quả hơn dặn trong prompt: prompt nằm cách
chỗ đó 18.000 ký tự, còn note nằm ngay cạnh kết quả rỗng mà mô hình đang đọc.

**Dạy orchestrator phân biệt hai nghĩa của chữ "lỗi".** Tiếng Việt dùng chung một
chữ cho sản phẩm FAIL (database) và dòng ERROR (file log). Quy tắc: "lỗi"/"fail"
đứng cạnh "sản phẩm / hàng / con / chai / thùng" → `historical_analytics`; chỉ khi
user nói rõ "log / traceback / service / module" → `log_analysis`.

**Đường ra khi vẫn sai** (`core/reroute.py`). Siết từ khoá chỉ giảm tỷ lệ sai chứ
không triệt được, nên phần này quan trọng không kém: khi có dấu hiệu route sai, bày
nút hỏi lại chính câu đó bằng agent khác, gửi thẳng vào agent được chọn. Nút chỉ
hiện trong ba tình huống — ý định lệch với agent đã trả lời (kể cả khi tool trả về
đầy dữ liệu, đây là ca tệ nhất), tool chạy nhưng rỗng, và không tool nào chạy trong
khi agent đang hỏi lại. Một nút "không đúng ý?" sau câu trả lời đúng là gieo nghi
ngờ vào chính thứ đang đúng.

**Chip gợi ý đọc số liệu, không đọc tên tool.** `_FALLBACK` tra theo tên tool nên
sau `group_by='camera'` nó vẫn mời "Phân tích theo camera". Nay có `_REDUNDANT` loại
câu đã thành vô nghĩa, và `_followups()` suy gợi ý từ kết quả — "Xem 5 sản phẩm lỗi
đó" lấy đúng con số vừa hiện. Khối [SUGGESTIONS] của LLM bị hạ xuống sau nhóm này:
nó bám ngữ cảnh hội thoại tốt hơn nhưng viết gợi ý mà không nhìn con số.


## 9. Đợt kiến trúc: agent-như-tool, cache, tóm tắt, stream

Sau một lượt soát kiến trúc, bốn việc được làm. Ba trong số đó (đa chặng, chẻ việc,
bỏ lượt LLM định tuyến) hoá ra là **cùng một thay đổi**.

### Agent con thành tool của orchestrator

Bản trước định tuyến MỘT chặng: một lượt LLM riêng trả JSON `{agent_id, confidence}`
rồi giao đứt cho một agent. Nay bốn agent con là tool, nên chọn agent chính là chọn
tool. "Sản lượng hôm nay có bị ảnh hưởng bởi lỗi thiết bị không?" trước không trả lời
được vì cần hai agent; nay gọi cả hai một lượt rồi gộp.

**Nhánh trả thẳng** là chỗ đáng chú ý. Vòng lặp tool tiêu chuẩn luôn quay lại LLM để
viết câu cuối, nhưng với một agent thì lượt đó vừa tốn vừa **có hại**: agent con đã
viết xong câu trả lời kèm số liệu, để orchestrator kể lại là mở đúng cửa cho lớp bug ở
mục 3. Nên một agent ⇒ trả nguyên văn, không thêm lượt LLM; từ hai agent mới tổng hợp.

**Đường tắt** (`core/intent.py`) bỏ hẳn lượt LLM điều phối khi câu hỏi khớp cụm rõ
nghĩa: 7,54s → 4,96s. Bảng cụm cố tình hẹp — không nhận từ đơn như "lỗi"/"log"/"ca",
đúng những từ đã gây route sai — và khớp nhiều agent thì nhường lại cho LLM, vì đó
chính là lúc cần nó. `core/reroute` dùng CHUNG bảng này; để hai bảng riêng thì chúng sẽ
lệch nhau theo cách tệ nhất, đường tắt gửi tới agent A trong khi reroute khẳng định câu
đó thuộc agent B.

Ba lỗi tự gây ra trong lúc làm, đáng ghi vì đều thuộc loại khó đoán trước:

| Lỗi | Vì sao |
|---|---|
| `ToolNode` làm chết mọi request | `AgentState.messages` không gắn reducer `add_messages`, nên giá trị trả về THAY THẾ cả danh sách; ToolNode chỉ trả ToolMessage mới ⇒ xoá sạch hội thoại |
| `/test` trống, và 4 nút hỏi-lại hiện dưới câu trả lời ĐÚNG | tool call của agent con nằm trong state riêng nên mất khỏi response, `reroute` tưởng không có tool nào chạy |
| câu về đăng nhập hiện kèm `explain_failures` | vòng gom quét cả history đã replay, báo tool của lượt TRƯỚC như vừa chạy |

### Cache kết quả tool

TTL theo kỳ: kỳ còn mở 45s (số liệu tăng từng giây), kỳ đã đóng 30 phút. Đặt trong
`BaseTool.create_tool` — nút duy nhất mọi tool đi qua. `should_cache` **liệt kê có**
thay vì loại trừ, nên category mới mặc định không cache: bỏ sót một tool đọc thì chỉ
chậm, còn cache lỡ một tool ghi thì trả kết quả sai. Không cache `start/stop_service`,
`generate_report`, và mọi kết quả lỗi.

### Tóm tắt hội thoại

Cắt cứng 40 message làm ngữ cảnh mất ĐỘT NGỘT: nói 30 lượt về một recipe, lượt 31 quên
đang nói recipe nào. Nay phần bị cắt được nén thành vài dòng, tích luỹ, lưu trong
`metadata` của conversation. Chỉ nén phần chưa tóm tắt rồi gộp với bản cũ — nén lại từ
đầu mỗi lần thì phiên càng dài càng đắt, và chi tiết cũ dần bị mài mất.

Dán dưới dạng `HumanMessage` có nhãn, **không** phải `SystemMessage`:
`orchestrator.call_model` chỉ thêm system prompt khi chưa có SystemMessage nào, nên dán
kiểu đó là orchestrator bỏ luôn prompt của chính nó.

### Stream tiến trình

Stream tiến trình, không stream token: câu trả lời chỉ đến ở cuối, sau khi tool xong,
nên stream token chỉ làm mượt ~1 giây cuối còn 20 giây đầu vẫn trắng. Điểm phát cũng
nằm trong `create_tool`, bọc NGOÀI cache để cache hit cũng được báo.

Hàng đợi là `queue.Queue` chứ không `asyncio.Queue` vì node LangGraph đồng bộ chạy
trong thread executor — chỗ phát khác thread chỗ đọc. Kênh phải mở TRƯỚC
`asyncio.create_task`, vì task chụp context lúc tạo.

`/chat/stream` cũ có đường code riêng nên không nạp/không lưu lịch sử và không có
attachment. Nay `_run_chat` dùng chung cho cả hai endpoint.

Câu hai agent, đo thật: 0,01s start → 1,94s "đang hỏi số liệu sản xuất" → 2,90s xong
`get_production_summary` → 4,58s "đang hỏi thiết bị" → 6,63s xong bốn tool thiết bị →
14,09s kết quả. Trước đó 14 giây là màn hình trắng.

---

## Còn để mở

**Trong agent service**

- Nội dung bên trong bản HTML/PDF vẫn cố định tiếng Anh (port nguyên từ frontend),
  chưa đi qua lớp i18n.
- Đã đề xuất, chưa làm: agent `config_audit`, agent `mlops`, `anomaly_watch`,
  `get_jetson_metrics`, tương quan giữa reject và fail.

**Đã nêu, người dùng quyết định CHƯA làm**

- **Không có kiểm tra quyền ở bất kỳ đâu.** `role` được đọc từ JWT và đặt vào
  `state.context` nhưng không agent hay tool nào dùng tới; `get_current_user` chỉ xác
  thực token và kiểm tra tài khoản còn hoạt động. Trong khi đó `stop_service` gửi
  SIGTERM rồi SIGKILL trực tiếp bằng psutil vào `camera_management_service.py`, không
  đi qua backend nên không có tầng nào chặn. Nghĩa là một tài khoản quyền `viewer` chat
  "dừng camera service" là dừng được dịch vụ kiểm tra trên dây chuyền đang chạy. Xác
  minh bằng đọc code, KHÔNG thử thật. Ba bước nếu làm: chặn theo quyền ở tầng tool,
  bắt xác nhận hai bước qua `ui_options`, ghi audit log mọi hành động agent thực hiện.
- Ngưỡng `confidence < 0.7` và không có reflection lúc chạy — hai mục còn lại của lượt
  soát kiến trúc, cũng được quyết định bỏ qua.

**Ngoài agent service**

- `weights/best_bottle_m.engine` vẫn thiếu.
- Xung reject 255 ms chờ kỹ thuật viên xác nhận.
- `SECRET_KEY` ở cả `backend/.env` và `agent_service/.env` đang là
  `change-this-secret-key-in-production`. Hai bên khớp nên JWT chạy đúng, nhưng
  đây là dây chuyền thật.
