"""
Tạo user test + chạy các thao tác CÓ ghi audit log.

CỐ Ý KHÔNG chạm tới: load_recipe, stop_recipe, create/update/delete_recipe,
create/update/delete_camera — recipe ONION POWDER đang chạy trên dây chuyền.
Chỉ dùng nhóm thao tác về user và đăng nhập/đăng xuất.
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

import json, urllib.request, urllib.parse, urllib.error, secrets, string

BASE = "http://localhost:8000"

def call(method, path, token=None, body=None, form=None, query=None):
    url = BASE + path + (("?" + urllib.parse.urlencode(query)) if query else "")
    data, headers = None, {}
    if token:
        headers["Authorization"] = "Bearer " + token
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt.strip() else {})
    except urllib.error.HTTPError as e:
        txt = e.read().decode()
        try:
            return e.code, json.loads(txt)
        except json.JSONDecodeError:
            return e.code, {"raw": txt[:200]}

def pwd():
    al = string.ascii_letters + string.digits
    return "Qa" + "".join(secrets.choice(al) for _ in range(10)) + "!7"

st, tok = call("POST", "/api/auth/login", form={"username": "admin", "password": admin_password()})
assert st == 200, (st, tok)
ADMIN = tok["access_token"]
print(f"đăng nhập admin: HTTP {st}")

NEW = [
    ("qa_supervisor", "supervisor", "QA Supervisor",  "qa.supervisor@example.com"),
    ("qa_operator1",  "operator",   "QA Operator 1",  "qa.operator1@example.com"),
    ("qa_operator2",  "operator",   "QA Operator 2",  "qa.operator2@example.com"),
    ("qa_temp",       "operator",   "QA Temp (xoá)",  "qa.temp@example.com"),
]

print("\n── 1. create_user (admin only) ──")
created = {}
for uname, role, full, mail in NEW:
    p = pwd()
    st, r = call("POST", "/api/users/", ADMIN, body={
        "username": uname, "email": mail, "full_name": full,
        "role": role, "is_active": True, "password": p,
    })
    if st == 201:
        created[uname] = {"id": r["_id"] if "_id" in r else r.get("id"), "pw": p, "role": role}
        print(f"  HTTP {st}  {uname:14} {role:11} id={created[uname]['id']}")
    else:
        print(f"  HTTP {st}  {uname:14} -> {r}")

json.dump(created, open(str(OUTDIR / "created_users.json"),"w"), indent=1)
print("\n  MẬT KHẨU (chỉ in một lần):")
for u, v in created.items():
    print(f"    {u:14} {v['pw']}")
