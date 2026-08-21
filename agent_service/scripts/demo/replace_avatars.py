"""
Thay avatar: upload ảnh mới, gán vào user, rồi XOÁ file avatar cũ.

Bước xoá là chủ ý. Mỗi lần upload sinh một file UUID mới trong
backend/uploads/avatar/; không dọn thì mỗi vòng thử lại để lại một ảnh mồ côi
không ai trỏ tới nữa. Đúng cơ chế đã làm `logs/` của dự án này phình lên 1,4 GB.
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

# Mật khẩu admin lấy từ biến môi trường, KHÔNG nhúng vào mã nguồn: thư mục này
# nằm trong git, và đây là hệ thống đang chạy thật.
import os as _os
def admin_password() -> str:
    pw = _os.environ.get("DEMO_ADMIN_PASSWORD")
    if not pw:
        raise SystemExit(
            "Thiếu DEMO_ADMIN_PASSWORD.\n"
            "  Chạy:  DEMO_ADMIN_PASSWORD='...' python3 " + _os.path.basename(__file__)
        )
    return pw

import json, mimetypes, urllib.request, urllib.parse, urllib.error, uuid
from pathlib import Path

BASE = "http://localhost:8000"
SPD = OUTDIR
SRC = OUTDIR / "avatars"
AVDIR = REPO / "backend" / "uploads" / "avatar"

def call(method, path, token=None, body=None, form=None):
    data, headers = None, {}
    if token: headers["Authorization"] = "Bearer " + token
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif body is not None:
        data = json.dumps(body).encode(); headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            t = r.read().decode(); return r.status, (json.loads(t) if t.strip() else {})
    except urllib.error.HTTPError as e:
        t = e.read().decode()
        try: return e.code, json.loads(t)
        except json.JSONDecodeError: return e.code, {"raw": t[:150]}

def post_file(path, token, fp):
    boundary = "----" + uuid.uuid4().hex
    ctype = mimetypes.guess_type(fp.name)[0] or "image/png"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{fp.name}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        fp.read_bytes(), b"\r\n", f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(BASE + path, data=body, method="POST", headers={
        "Authorization": "Bearer " + token,
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    })
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"raw": e.read().decode()[:200]}

st, r = call("POST", "/api/auth/login", form={"username": "admin", "password": admin_password()})
ADMIN = r["access_token"]
st, users = call("GET", "/api/users/", ADMIN)
by_name = {u["username"]: u for u in users}

old = {u["username"]: u.get("avatar_url") for u in users}
replaced = []

for fp in sorted(SRC.glob("*.png")):
    uname = fp.stem
    u = by_name.get(uname)
    if not u:
        print(f"  {uname:15} khong co user, bo qua"); continue
    st, up = post_file("/api/upload/avatar", ADMIN, fp)
    if st != 200:
        print(f"  {uname:15} upload LOI {st}: {up}"); continue
    uid = u.get("_id") or u.get("id")
    st2, _ = call("PUT", f"/api/users/{uid}", ADMIN, body={"avatar_url": up["url"]})
    print(f"  {uname:15} -> {up['url'].rsplit('/',1)[-1]}  gan: HTTP {st2}")
    if st2 == 200:
        replaced.append(old.get(uname))

print("\n── dọn file avatar mồ côi ──")
st, users2 = call("GET", "/api/users/", ADMIN)
inuse = {(u.get("avatar_url") or "").rsplit("/", 1)[-1] for u in users2}
removed = 0
for p in AVDIR.glob("*"):
    if p.is_file() and p.name not in inuse:
        p.unlink(); removed += 1
print(f"  xoá {removed} file không còn ai trỏ tới")
print(f"  còn lại {len(list(AVDIR.glob('*')))} file, {len([x for x in inuse if x])} user có avatar")
