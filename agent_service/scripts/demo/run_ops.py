"""Chạy các thao tác CÓ ghi audit log. Không chạm recipe/camera."""

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

import json, urllib.request, urllib.parse, urllib.error

BASE = "http://localhost:8000"
SPD = str(OUTDIR)

def call(method, path, token=None, body=None, form=None, query=None):
    url = BASE + path + (("?" + urllib.parse.urlencode(query)) if query else "")
    data, headers = None, {}
    if token: headers["Authorization"] = "Bearer " + token
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif body is not None:
        data = json.dumps(body).encode(); headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            t = r.read().decode(); return r.status, (json.loads(t) if t.strip() else {})
    except urllib.error.HTTPError as e:
        t = e.read().decode()
        try: return e.code, json.loads(t)
        except json.JSONDecodeError: return e.code, {"raw": t[:150]}

users = json.load(open(f"{SPD}/created_users.json"))
st, r = call("POST", "/api/auth/login", form={"username":"admin","password":admin_password()})
ADMIN = r["access_token"]

toks = {}
print("── 2. login (mỗi user) → ActionType.LOGIN ──")
for u, v in users.items():
    st, r = call("POST", "/api/auth/login", form={"username": u, "password": v["pw"]})
    if st == 200:
        toks[u] = r["access_token"]
        print(f"  HTTP {st}  {u:14} {v['role']:11} token OK")
    else:
        print(f"  HTTP {st}  {u:14} -> {r}")

print("\n── 3. update_user (admin) → ActionType.UPDATE_USER ──")
for u in ("qa_supervisor", "qa_operator1"):
    st, r = call("PUT", f"/api/users/{users[u]['id']}", ADMIN,
                 body={"full_name": users[u]["role"].title() + " (đã cập nhật)",
                       "phone_number": "0900000001"})
    print(f"  HTTP {st}  {u:14} full_name -> {r.get('full_name')}")

print("\n── 4. reset_user_password (admin) → ActionType.RESET_USER_PASSWORD ──")
newpw = "QaReset2026!x"
st, r = call("POST", f"/api/users/{users['qa_operator2']['id']}/reset-password", ADMIN,
             query={"new_password": newpw})
print(f"  HTTP {st}  qa_operator2 -> {r.get('message', r)}")
st, r = call("POST", "/api/auth/login", form={"username":"qa_operator2","password":newpw})
print(f"  đăng nhập lại bằng mật khẩu mới: HTTP {st}")
if st == 200: toks["qa_operator2"] = r["access_token"]

print("\n── 5. Kiểm tra phân quyền (không ghi log, xác nhận role có hiệu lực) ──")
checks = [
    ("GET",  "/api/users/",              "danh sách user (cần supervisor+)"),
    ("POST", "/api/users/",              "tạo user (cần admin)"),
]
for who in ("qa_operator1", "qa_supervisor"):
    for method, path, label in checks:
        body = {"username":"x_probe","full_name":"x","role":"operator","password":"abc123"} if method=="POST" else None
        st, _ = call(method, path, toks.get(who), body=body)
        verdict = "CHO PHEP" if st < 300 else ("TU CHOI 403" if st == 403 else f"HTTP {st}")
        print(f"  {who:14} {label:34} -> {verdict}")

print("\n── 6. logout → ActionType.LOGOUT ──")
for u in ("qa_supervisor", "qa_operator1"):
    st, r = call("POST", "/api/auth/logout", toks.get(u))
    print(f"  HTTP {st}  {u}")

print("\n── 7. delete_user (admin, chỉ xoá qa_temp) → ActionType.DELETE_USER ──")
st, r = call("DELETE", f"/api/users/{users['qa_temp']['id']}", ADMIN)
print(f"  HTTP {st}  qa_temp -> {r.get('message', r)}")
