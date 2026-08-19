"""
Tạo user giả lập + ghi hồ sơ nhân sự.

Tài khoản tạo qua API backend (để sinh audit log `create_user` đúng luồng), còn
nhóm field hồ sơ (bộ phận, chức vụ, ca…) ghi thẳng vào MongoDB vì pydantic model
của backend chưa có chúng và ta đang giữ hai codebase tách rời. `update_user` của
backend dùng `$set` nên các field này không bị xoá khi sửa user trên UI.

KHÔNG chạm recipe/camera.
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

import json, secrets, string, urllib.request, urllib.parse, urllib.error
from pymongo import MongoClient

BASE = "http://localhost:8000"
SPD = str(OUTDIR)

def env(k, f=None):
    for l in open(f or BACKEND_ENV):
        l = l.strip()
        if l.startswith(k + "="): return l.split("=", 1)[1]

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
        with urllib.request.urlopen(req, timeout=30) as r:
            t = r.read().decode(); return r.status, (json.loads(t) if t.strip() else {})
    except urllib.error.HTTPError as e:
        t = e.read().decode()
        try: return e.code, json.loads(t)
        except json.JSONDecodeError: return e.code, {"raw": t[:150]}

def pwd():
    al = string.ascii_letters + string.digits
    return "Qa" + "".join(secrets.choice(al) for _ in range(10)) + "!7"

# username, role, họ tên, email, mã NV, bộ phận, chức vụ, ca, dây chuyền, ngày vào
STAFF = [
    ("nv_vanhanh1", "operator",   "Nguyễn Thị Lan",   "lan.nt",   "NV-1042", "Sản xuất", "Công nhân vận hành",        "Ca A (06:00–14:00)", "Line 1", "2023-03-15"),
    ("nv_vanhanh2", "operator",   "Trần Văn Hùng",    "hung.tv",  "NV-1057", "Sản xuất", "Công nhân vận hành",        "Ca B (14:00–22:00)", "Line 1", "2024-01-08"),
    ("truongca_a",  "supervisor", "Phạm Minh Đức",    "duc.pm",   "NV-0318", "Sản xuất", "Trưởng ca",                 "Ca A (06:00–14:00)", "Line 1-2", "2019-07-01"),
    ("qc_kiemtra",  "supervisor", "Lê Thị Hồng Nhung","nhung.lth","NV-0741", "QA/QC",    "Nhân viên kiểm tra chất lượng", "Ca hành chính",  "Toàn bộ", "2022-09-12"),
    ("kt_baotri",   "operator",   "Võ Quốc Thắng",    "thang.vq", "NV-0526", "Bảo trì",  "Kỹ thuật viên bảo trì",     "Ca hành chính",      "Toàn bộ", "2021-05-20"),
    ("nv_kho",      "viewer",     "Đặng Văn Sáu",     "sau.dv",   "NV-0903", "Kho vận",  "Nhân viên kho",             "Ca hành chính",      "—",       "2020-11-03"),
]

st, r = call("POST", "/api/auth/login", form={"username": "admin", "password": admin_password()})
ADMIN = r["access_token"]

created = {}
print("── tạo tài khoản (qua API → ghi audit log create_user) ──")
for uname, role, full, mail, code, dept, title, shift, line, hired in STAFF:
    p = pwd()
    st, r = call("POST", "/api/users/", ADMIN, body={
        "username": uname, "email": f"{mail}@example.com", "full_name": full,
        "phone_number": "09" + code.split("-")[1][:2] + str(abs(hash(uname)))[:6],
        "role": role, "is_active": True, "password": p,
    })
    if st == 201:
        created[uname] = {"pw": p, "id": r.get("_id") or r.get("id")}
        print(f"  HTTP {st}  {uname:13} {role:11} {full}")
    else:
        print(f"  HTTP {st}  {uname:13} -> {r}")

json.dump(created, open(f"{SPD}/created_staff.json", "w"), indent=1)

print("\n── ghi hồ sơ nhân sự (trực tiếp vào MongoDB) ──")
db = MongoClient(env("MONGODB_URL"))[env("DATABASE_NAME")]
for uname, role, full, mail, code, dept, title, shift, line, hired in STAFF:
    res = db["users"].update_one({"username": uname}, {"$set": {
        "employee_code": code, "department": dept, "job_title": title,
        "shift": shift, "production_line": line, "hire_date": hired,
    }})
    print(f"  {uname:13} {dept:10} {title:32} {shift}")

# Hồ sơ cho các user đã có sẵn, để mọi thẻ đều đầy đủ
print("\n── bổ sung hồ sơ cho user cũ ──")
EXISTING = [
    ("admin",         "NV-0001", "Ban điều hành", "Quản trị hệ thống",   "Ca hành chính", "—",      "2018-01-02"),
    ("supervisor",    "NV-0205", "Sản xuất",      "Giám sát sản xuất",   "Ca C (22:00–06:00)", "Line 1-2", "2020-02-17"),
    ("operator",      "NV-1103", "Sản xuất",      "Công nhân vận hành",  "Ca C (22:00–06:00)", "Line 2", "2024-06-10"),
    ("qa_supervisor", "NV-0688", "QA/QC",         "Giám sát QA",         "Ca hành chính", "Toàn bộ", "2021-10-05"),
    ("qa_operator1",  "NV-1210", "QA/QC",         "Nhân viên QA",        "Ca A (06:00–14:00)", "Line 1", "2024-08-19"),
    ("qa_operator2",  "NV-1211", "QA/QC",         "Nhân viên QA",        "Ca B (14:00–22:00)", "Line 2", "2025-02-03"),
]
for uname, code, dept, title, shift, line, hired in EXISTING:
    db["users"].update_one({"username": uname}, {"$set": {
        "employee_code": code, "department": dept, "job_title": title,
        "shift": shift, "production_line": line, "hire_date": hired,
    }})
    print(f"  {uname:14} {dept:14} {title}")

print("\n── đăng nhập từng user mới → audit log login ──")
for uname, v in created.items():
    st, r = call("POST", "/api/auth/login", form={"username": uname, "password": v["pw"]})
    print(f"  HTTP {st}  {uname}")

print("\nMẬT KHẨU (in một lần):")
for u, v in created.items():
    print(f"  {u:13} {v['pw']}")
