# Dữ liệu demo: user, hồ sơ nhân sự, ảnh chân dung, hoạt động theo ca

Hướng dẫn dựng lại bộ dữ liệu demo dùng để test agent — đặc biệt là các tool đọc
audit log (`get_audit_logs`) và phần thẻ nhân sự hiện trên `/test`.

Script nằm ở `agent_service/scripts/demo/`. Chúng chạy được từ bất kỳ thư mục nào:
mọi đường dẫn suy ra từ vị trí file, không phải từ `cwd`. Kết quả trung gian ghi
vào `scripts/demo/_out/`.

> **Cảnh báo an toàn — đọc trước khi chạy bất cứ thứ gì.**
>
> Đây là dây chuyền đang chạy thật. Toàn bộ script trong thư mục này **cố ý không
> chạm** tới `load_recipe`, `stop_recipe`, create/update/delete recipe, và
> create/update/delete camera. Recipe ONION POWDER đang chạy trên máy; một lệnh
> `stop_recipe` để "sinh audit log cho đẹp" là dừng sản xuất thật.
>
> Chỉ dùng nhóm thao tác về **user** và **đăng nhập/đăng xuất**. Nếu bạn thêm
> script mới vào đây, giữ nguyên ràng buộc đó.

## Nhiều máy: mỗi máy một bộ nhân sự riêng

Bộ script này ban đầu hardcode **một** bộ nhân sự (`STAFF`, `PEOPLE`, `NAMES`, và
seed random cố định). Dựng lên bốn máy thì bốn máy ra đúng cùng một tập người,
cùng một mẫu giờ vào/ra tới từng phút — đứng cạnh nhau là lộ ngay đây là dữ liệu
dựng.

`scripts/demo/profiles.py` tách phần đó ra theo máy. Chọn bằng biến môi trường:

```bash
export DEMO_MACHINE=M1          # M1 | M2 | LineTine | PC-Auto-1
export DEMO_ADMIN_PASSWORD='...'
python3 seed_staff.py
python3 to_english.py
python3 gen_new.py _out/avatars_M1     # sinh ảnh (chạy ở máy có mạng nhanh)
python3 replace_avatars.py             # đọc _out/avatars/
python3 sim_activity.py
```

**Không đặt `DEMO_MACHINE` thì mọi thứ giữ nguyên hành vi cũ** — cố ý, để Auto2
(máy đã setup xong trước khi có cơ chế này) không phải chạy lại gì.

Mỗi profile giữ: `staff` (tài khoản tạo mới), `existing` (hồ sơ cho tài khoản có
sẵn — danh sách này khác nhau từng máy), `names`, `avatars`, `part_time`, và
`seed`. Seed khác nhau là phần dễ quên nhất: cùng seed thì `sim_activity.py` sinh
ra đúng cùng một mẫu giờ trên mọi máy.

`gen_avatars.py` **thoát ngay** khi `DEMO_MACHINE` bật: ở chế độ theo máy,
`gen_new.py` đã bao gồm cả tài khoản có sẵn, chạy tiếp sẽ ghi đè
`supervisor.png` / `operator.png` bằng khuôn mặt của bộ gốc.

### Hai thứ vấp phải khi rollout bốn máy

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `RateLimitError` giữa chừng, mất luôn các ảnh còn lại của lượt | gpt-image-1 giới hạn **5 ảnh/phút cho cả organization**; chạy song song bốn máy là chắc chắn chạm trần | `gen_new.py` lùi lại và thử lại (20s → 120s). Ảnh đã có được bỏ qua nên chạy lại không tốn tiền |
| `rsync` ảnh sang Jetson mất hơn 5 phút rồi timeout | ảnh gốc 1024×1024 PNG ≈ **1,5 MB/tấm**, link tới Jetson chỉ vài chục KB/s | thu về 512×512 trước khi đẩy: 13 MB → 2,7 MB. Trang 12 thẻ nhân sự cũng khỏi phải tải 18 MB ảnh |

Ảnh nên sinh ở máy dev rồi rsync sang, chứ không bắt Jetson gọi OpenAI.

---

## Yêu cầu

- Backend đang chạy ở `http://localhost:8000`, đăng nhập được bằng `admin`.
- MongoDB truy cập được (đọc `MONGODB_URL`, `DATABASE_NAME` từ `backend/.env`).
- Sinh ảnh cần `OPENAI_API_KEY` trong `agent_service/.env`, và `pip install openai
  python-dotenv`.

## Chạy theo thứ tự

Mật khẩu admin truyền qua biến môi trường `DEMO_ADMIN_PASSWORD` — thư mục script
nằm trong git nên không nhúng mật khẩu vào mã nguồn. Thiếu biến này thì script
dừng ngay với thông báo, không chạy nửa vời.

```bash
cd agent_service/scripts/demo
export DEMO_ADMIN_PASSWORD='...'

python3 seed_staff.py        # 1. tạo tài khoản + ghi hồ sơ nhân sự
python3 to_english.py        # 2. đổi bộ phận/chức vụ/ca sang tiếng Anh
python3 gen_new.py           # 3. sinh ảnh chân dung (gọi OpenAI, mất vài phút)
python3 gen_avatars.py       #    (ảnh cho nhóm tài khoản có sẵn)
python3 replace_avatars.py   # 4. upload ảnh, gán vào user, dọn ảnh mồ côi
python3 sim_activity.py      # 5. giả lập hoạt động trải theo ca
```

`seed_users.py` và `run_ops.py` là bộ **đầu tiên**, tạo nhóm `qa_*` và chạy các
thao tác có ghi audit log (`create_user`, `update_user`, `delete_user`,
`reset_password`, `login`, `logout`). Giữ lại vì chúng là cách sinh audit log
nhiều loại action nhất; `seed_staff.py` tập trung vào hồ sơ nhân sự.

Mật khẩu sinh ngẫu nhiên và **chỉ in ra một lần** ở cuối mỗi script — copy ngay.
Bộ hiện tại đã lưu ở `TEST_ACCOUNTS.txt` tại gốc repo (file này nằm trong
`.gitignore` vì chứa mật khẩu dạng rõ).

## Trạng thái hiện tại

12 user, tất cả đều có đủ hồ sơ và ảnh; 167 bản ghi audit giả lập trong tổng 1.663.

| username | quyền | mã NV | bộ phận | chức vụ | ca |
|---|---|---|---|---|---|
| `admin` | admin | NV-0001 | Management | System Administrator | Office hours |
| `supervisor` | supervisor | NV-0205 | Production | Production Supervisor | Shift C (22:00–06:00) |
| `truongca_a` | supervisor | NV-0318 | Production | Shift Leader | Shift A (06:00–14:00) |
| `qa_supervisor` | supervisor | NV-0688 | QA/QC | QA Supervisor | Office hours |
| `qc_kiemtra` | supervisor | NV-0741 | QA/QC | Quality Inspector | Office hours |
| `kt_baotri` | operator | NV-0526 | Maintenance | Maintenance Technician | Shift P (12:00–16:00) |
| `nv_vanhanh1` | operator | NV-1042 | Production | Machine Operator | Shift A (06:00–14:00) |
| `nv_vanhanh2` | operator | NV-1057 | Production | Machine Operator | Shift B (14:00–22:00) |
| `operator` | operator | NV-1103 | Production | Machine Operator | Shift C (22:00–06:00) |
| `qa_operator1` | operator | NV-1210 | QA/QC | QA Technician | Shift A (06:00–14:00) |
| `qa_operator2` | operator | NV-1211 | QA/QC | QA Technician | Shift B (14:00–22:00) |
| `nv_kho` | viewer | NV-0903 | Warehouse | Warehouse Clerk | Shift P (11:00–15:00) |

---

## 1. Tài khoản và hồ sơ — `seed_staff.py`

Tài khoản tạo **qua API backend** (`POST /api/users/`), không insert thẳng vào
Mongo. Lý do: chỉ đi qua API thì mới sinh audit log `create_user` đúng luồng, mà
audit log chính là thứ ta cần để test.

Nhóm field hồ sơ thì **ghi thẳng vào MongoDB**: `employee_code`, `department`,
`job_title`, `shift`, `production_line`, `hire_date`. Pydantic model của backend
chưa có các field này, và hai codebase đang giữ tách rời nên không sửa backend chỉ
để phục vụ dữ liệu demo. An toàn được vì `update_user` của backend dùng `$set`,
nên sửa user trên UI không xoá mất chúng.

Script cũng bổ sung hồ sơ cho các tài khoản **có sẵn** (`admin`, `supervisor`,
`operator`, `qa_*`) để không có thẻ nào hiện thiếu nửa thông tin.

### Một lỗi đã gặp

Email `.local` bị pydantic từ chối. Dùng `example.com`.

## 2. Bộ phận / chức vụ sang tiếng Anh — `to_english.py`

Ban đầu hồ sơ ghi bằng tiếng Việt. Sau đó đổi sang tiếng Anh theo yêu cầu — và
đổi **cả ca lẫn dây chuyền**, không chỉ hai field bộ phận/chức vụ, vì chúng nằm
trên cùng một thẻ; để lẫn thì thẻ đọc nửa nọ nửa kia.

Lưu ý phân biệt: đây là field `shift` trong **hồ sơ user** (dữ liệu demo). Nó khác
`SHIFTS` trong `agent_app/tools/analytics_tools.py` — đó là định nghĩa ca của hệ
thống, vẫn giữ tiếng Việt (Ca A/B/C) vì đó là tên trên bảng phân ca của xưởng.

## 3. Ảnh chân dung — `gen_new.py`, `gen_avatars.py`

Sinh bằng OpenAI image API (`gpt-image-1`, 1024×1024). Prompt ghép từ bốn phần
tách rời để giữ nhất quán giữa các ảnh:

1. **Khung ảnh** — ảnh thẻ, vai lên, nhìn thẳng, ánh sáng studio đều, nền xám
   trơn. Kèm câu `"Fictional person, not a real individual."`
2. **Nhân dạng** — người Việt, nét Đông Á, tóc đen, da sáng ấm.
3. **Đồng phục**, theo đúng quy ước của xưởng:
   - operator → áo xanh dương (`royal-blue`) + lưới trùm tóc trắng
   - supervisor → áo blouse trắng an toàn thực phẩm + lưới trùm tóc trắng
   - admin → blazer navy, sơ mi trắng, **không** lưới trùm tóc (khối văn phòng)
4. **Đặc điểm riêng từng người** — tuổi, kính, kiểu tóc, phụ kiện. Có phần này thì
   12 ảnh mới không ra 12 người trông giống nhau.

Script **bỏ qua file đã tồn tại**, nên chạy lại không tốn tiền API thêm. Ảnh ghi
vào `scripts/demo/_out/avatars/<username>.png`; đặt tên theo username để
`replace_avatars.py` biết gán cho ai.

Muốn vẽ lại một người: xoá đúng file `.png` của người đó rồi chạy lại.

## 4. Upload và gán ảnh — `replace_avatars.py`

Ba việc trong một lượt: upload qua `POST /api/upload/avatar`, gán `avatar_url` qua
`PUT /api/users/{id}`, rồi **xoá các file avatar không còn ai trỏ tới**.

Bước xoá là chủ ý. Mỗi lần upload sinh một file UUID mới trong
`backend/uploads/avatar/`; không dọn thì mỗi vòng thử lại để lại một ảnh mồ côi.
Đúng cơ chế đã làm `logs/` của dự án này phình lên 1,4 GB.

## 5. Hoạt động theo ca — `sim_activity.py`

**Vì sao cần.** Mọi bản ghi login thật đều do script tạo trong vài giây, nên thẻ
nhân sự hiện `Hoạt động 14:12:36 → 14:23:41` — vô nghĩa. Ca thật dài 4h hoặc 8h,
và đó mới là thứ người vận hành muốn thấy.

Cách làm:

- Giờ ca được **bóc từ chính chuỗi ca** (`"Shift A (06:00–14:00)"`) bằng regex, để
  dữ liệu sinh ra và mô tả trên thẻ không thể lệch nhau. Ca đêm vắt qua nửa đêm
  nên kết thúc rơi sang ngày sau.
- Mỗi ca sinh: một `login` đầu ca lệch vài phút cho tự nhiên, 1–3 lần login lại
  giữa ca (nghỉ giữa giờ, đổi máy), và một `logout` cuối ca — `logout` **chỉ khi
  ca đã kết thúc**, để ca đang diễn ra hiện đúng dạng "đang trong ca" thay vì báo
  đã tan.
- Trải 4 ngày (`DAYS_BACK = 4`), nên có cả ca đã trọn vẹn để đối chiếu.
- Hai người được đặt ca 4h (`Shift P`) phủ qua giờ hiện tại, để trên thẻ có cả ca
  4h lẫn ca 8h.
- `admin` **bị loại trừ** — đó là tài khoản đang dùng thật, để nguyên.

Hai điểm cần biết khi chạy lại:

- Mọi bản ghi đều mang cờ **`simulated: True`**, để phân biệt với log thật và xoá
  lại được.
- Script **tự xoá lần giả lập trước** (`delete_many({"simulated": True})`) rồi mới
  chèn, nên chạy lại không nhân đôi dữ liệu.
- Seed random **cố định** (`Random(20260819)`), nên chạy lại ra cùng kết quả.

Cuối script có phần đối chiếu, in khoảng hoạt động hôm nay và hôm qua của từng
người — dùng để xác nhận ca đang diễn ra ra khoảng ngắn hơn độ dài ca, còn ca hôm
qua ra trọn 4h/8h.

### Dọn dữ liệu giả lập

```bash
cd agent_service/scripts/demo
python3 -c "
from pathlib import Path
from pymongo import MongoClient
ENV = Path.cwd().resolve().parents[2] / 'backend' / '.env'
def env(k):
    for l in open(ENV):
        l = l.strip()
        if l.startswith(k + '='): return l.split('=', 1)[1]
db = MongoClient(env('MONGODB_URL'))[env('DATABASE_NAME')]
print('đã xoá:', db['action_logs'].delete_many({'simulated': True}).deleted_count)
"
```

Log thật không mang cờ `simulated` nên không bị ảnh hưởng.

---

## Những lỗi đã gặp khi làm phần này

Ghi lại vì chúng đều là lỗi "hiện ra vẫn đẹp nhưng nội dung sai":

| Hiện tượng | Nguyên nhân |
|---|---|
| Thẻ ghi `14:12 → 14:23` trong khi bản ghi đầu tiên là 11:05 | thống kê từng người tính trên `entries` đã bị `limit` cắt, không phải trên toàn bộ match. Sửa bằng `$group` trên full match |
| Badge hiện quyền hệ thống cạnh chức vụ, đọc như hai chức danh đá nhau ("Maintenance Technician" mà badge ghi "operator") | thiếu nhãn. Nay ghi rõ "Quyền: operator" |
| Thẻ dùng theme tối trên trang `/test` nền sáng | CSS viết riêng thay vì dùng biến CSS của trang |
| `get_audit_logs(username='ONION POWDER')` trả "không có lịch sử" trong khi có 5 lần load | ONION POWDER là **tài nguyên**, không phải người. Thêm tham số `resource`; username lạ nay bị từ chối kèm danh sách hợp lệ |
| Câu "ai hoạt động nhiều giờ nhất" được trả lời bằng **số lượng thao tác** | thiếu `active_hours`. Thêm rồi vẫn sai, vì chính `strip_for_llm` của tôi đã lược mất field đó |
| Bảng biến mất trên điện thoại | rule 720px ẩn `.tbl-scroll` cho mọi bảng, nhưng chỉ bảng ≥5 cột mới có bản card thay thế. Thu hẹp lại thành `.tbl-wrap.has-cards` |
