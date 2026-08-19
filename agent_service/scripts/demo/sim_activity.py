"""
Giả lập hoạt động trải dài theo ca làm việc.

Vì sao cần: mọi bản ghi login thật đều do script tạo trong vài giây, nên thẻ hiện
"Hoạt động 14:12:36" — vô nghĩa. Ca thực tế dài 4h hoặc 8h, và đó mới là thứ
người vận hành muốn thấy.

Bản ghi giả lập được đánh dấu `simulated: True` để phân biệt với log thật và xoá
lại được. Chỉ dùng login/logout — KHÔNG bịa các action chạm recipe/camera.
"""

# --- đường dẫn ---
# Script này chạy được từ bất kỳ cwd nào: mọi đường dẫn suy ra từ vị trí chính
# file này, không phải từ cwd. Bản gốc trỏ vào thư mục tạm của phiên làm việc, và
# thư mục đó bị dọn sau khi phiên kết thúc.
from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[3]          # …/ocr_datecode
OUTDIR = _Path(__file__).resolve().parent / "_out"   # nơi ghi kết quả trung gian
OUTDIR.mkdir(exist_ok=True)
BACKEND_ENV = REPO / "backend" / ".env"
AGENT_ENV = REPO / "agent_service" / ".env"

import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pymongo import MongoClient

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DAYS_BACK = 4          # hôm nay + 3 ngày trước

def env(k, f=None):
    for l in open(f or BACKEND_ENV):
        l = l.strip()
        if l.startswith(k + "="): return l.split("=", 1)[1]

db = MongoClient(env("MONGODB_URL"))[env("DATABASE_NAME")]
rnd = random.Random(20260819)          # cố định seed để chạy lại ra cùng kết quả

def to_utc(local_dt):
    """Giờ địa phương → naive UTC, đúng cách collection này lưu."""
    return local_dt.replace(tzinfo=TZ).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

def shift_window(shift, day):
    """
    (bắt đầu, kết thúc) giờ địa phương của ca trong ngày `day`.

    Giờ được bóc từ chính chuỗi ca ('Shift A (06:00–14:00)') để dữ liệu và mô tả
    không thể lệch nhau. Ca đêm vắt qua nửa đêm nên kết thúc rơi sang ngày sau.
    """
    import re
    m = re.search(r"(\d{2}):(\d{2})\D+(\d{2}):(\d{2})", shift or "")
    if not m:
        return (datetime.combine(day, datetime.min.time()) + timedelta(hours=8),
                datetime.combine(day, datetime.min.time()) + timedelta(hours=17))
    h1, m1, h2, m2 = map(int, m.groups())
    start = datetime.combine(day, datetime.min.time()) + timedelta(hours=h1, minutes=m1)
    end = datetime.combine(day, datetime.min.time()) + timedelta(hours=h2, minutes=m2)
    if end <= start:
        end += timedelta(days=1)
    return start, end

# Hai người làm bán thời gian 4h, để trên thẻ có cả ca 4h lẫn ca 8h
# Đặt hai ca 4h phủ qua giờ hiện tại, để hôm nay chúng hiện dạng "đang trong ca"
# thay vì bị các bản ghi login thật ngoài ca kéo dài khoảng hoạt động ra.
db["users"].update_one({"username": "nv_kho"},
                       {"$set": {"shift": "Shift P (11:00–15:00)"}})
db["users"].update_one({"username": "kt_baotri"},
                       {"$set": {"shift": "Shift P (12:00–16:00)"}})

users = list(db["users"].find({}, {"username": 1, "full_name": 1, "role": 1, "shift": 1}))
now_local = datetime.now(TZ).replace(tzinfo=None)
today = now_local.date()

# Dọn lần giả lập trước để chạy lại không nhân đôi
removed = db["action_logs"].delete_many({"simulated": True}).deleted_count
print(f"  xoá {removed} bản ghi giả lập cũ\n")

docs = []
for u in users:
    uname = u["username"]
    if uname == "admin":
        continue                       # admin là tài khoản bạn đang dùng thật, để nguyên
    for back in range(DAYS_BACK):
        day = today - timedelta(days=back)
        start, end = shift_window(u.get("shift"), day)
        if start > now_local:
            continue                   # ca chưa tới
        finish = min(end, now_local)   # ca đang diễn ra thì chỉ tới hiện tại
        if (finish - start) < timedelta(minutes=30):
            continue

        # login đầu ca, lệch vài phút cho tự nhiên
        t_in = start + timedelta(minutes=rnd.randint(0, 9), seconds=rnd.randint(0, 59))
        docs.append((uname, u, "login", t_in, f"User '{uname}' logged in"))

        # 1-3 lần đăng nhập lại giữa ca (nghỉ giữa giờ, đổi máy)
        span = (finish - t_in).total_seconds()
        for _ in range(rnd.randint(1, 3)):
            off = rnd.uniform(0.15, 0.85) * span
            docs.append((uname, u, "login", t_in + timedelta(seconds=off),
                         f"User '{uname}' logged in"))

        # logout cuối ca, chỉ khi ca đã kết thúc
        if end <= now_local:
            t_out = end - timedelta(minutes=rnd.randint(0, 6), seconds=rnd.randint(0, 59))
            docs.append((uname, u, "logout", t_out, f"User '{uname}' logged out"))

rows = [{
    "user_id": str(u.get("_id", "")),
    "username": uname,
    "action_type": act,
    "resource_type": "auth",
    "resource_id": None,
    "description": desc,
    "old_value": None,
    "new_value": None,
    "ip_address": f"192.168.100.{rnd.randint(20, 90)}",
    "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/141.0 Safari/537.36",
    "timestamp": to_utc(t),
    "simulated": True,
} for uname, u, act, t, desc in docs]

if rows:
    db["action_logs"].insert_many(rows)
print(f"  thêm {len(rows)} bản ghi giả lập ({DAYS_BACK} ngày, {len(users)-1} người)\n")

# Đối chiếu: khoảng hoạt động hôm nay của mỗi người
from collections import defaultdict
start_utc = to_utc(datetime.combine(today, datetime.min.time()))
spans = defaultdict(list)
for d in db["action_logs"].find({"timestamp": {"$gte": start_utc}}, {"username": 1, "timestamp": 1}):
    spans[d["username"]].append(d["timestamp"])
def report(label, lo_utc, hi_utc):
    got = defaultdict(list)
    for d in db["action_logs"].find({"timestamp": {"$gte": lo_utc, "$lt": hi_utc}},
                                    {"username": 1, "timestamp": 1}):
        got[d["username"]].append(d["timestamp"])
    print(f"  {label}")
    f = lambda x: (x + timedelta(hours=7)).strftime("%H:%M")
    for uname, ts in sorted(got.items()):
        lo, hi = min(ts), max(ts)
        h, rem = divmod(int((hi - lo).total_seconds()), 3600)
        sh = next((x.get("shift") for x in users if x["username"] == uname), "") or ""
        print(f"    {uname:14} {f(lo)} → {f(hi)}   {h}h{rem//60:02d}m   {len(ts):2} bản ghi   {sh}")
    print()

report("HÔM NAY (ca đang diễn ra ⇒ khoảng ngắn hơn độ dài ca):",
       start_utc, to_utc(datetime.combine(today + timedelta(days=1), datetime.min.time())))
report("HÔM QUA (ca đã kết thúc ⇒ thấy trọn 4h / 8h):",
       to_utc(datetime.combine(today - timedelta(days=1), datetime.min.time())), start_utc)
