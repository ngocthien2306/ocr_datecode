"""Đổi giá trị hồ sơ nhân sự sang tiếng Anh."""

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

from pymongo import MongoClient

def env(k, f=None):
    for l in open(f or BACKEND_ENV):
        l = l.strip()
        if l.startswith(k + "="): return l.split("=", 1)[1]

db = MongoClient(env("MONGODB_URL"))[env("DATABASE_NAME")]

DEPT = {
    "Sản xuất": "Production",
    "QA/QC": "QA/QC",
    "Bảo trì": "Maintenance",
    "Kho vận": "Warehouse",
    "Ban điều hành": "Management",
}
TITLE = {
    "Công nhân vận hành": "Machine Operator",
    "Trưởng ca": "Shift Leader",
    "Nhân viên kiểm tra chất lượng": "Quality Inspector",
    "Kỹ thuật viên bảo trì": "Maintenance Technician",
    "Nhân viên kho": "Warehouse Clerk",
    "Quản trị hệ thống": "System Administrator",
    "Giám sát sản xuất": "Production Supervisor",
    "Giám sát QA": "QA Supervisor",
    "Nhân viên QA": "QA Technician",
}
# Ca và dây chuyền cũng đổi luôn: chúng nằm cùng một thẻ với hai field trên,
# để lẫn tiếng Việt thì thẻ đọc nửa nọ nửa kia.
SHIFT = {
    "Ca A (06:00–14:00)": "Shift A (06:00–14:00)",
    "Ca B (14:00–22:00)": "Shift B (14:00–22:00)",
    "Ca C (22:00–06:00)": "Shift C (22:00–06:00)",
    "Ca hành chính": "Office hours",
}
LINE = {"Toàn bộ": "All lines"}

# Tên hiển thị: bỏ hậu tố "(đã cập nhật)" còn sót lại từ lần test update_user,
# và đặt tên người thật sự cho các tài khoản mẫu.
NAMES = {
    "operator": "Hoàng Văn Nam",
    "supervisor": "Bùi Thị Thu Hà",
    "qa_supervisor": "Ngô Thanh Tuyền",
    "qa_operator1": "Đỗ Trung Kiên",
    "qa_operator2": "Vũ Thị Mai Anh",
}

changed = 0
for u in db["users"].find({}, {"username": 1, "department": 1, "job_title": 1,
                              "shift": 1, "production_line": 1, "full_name": 1}):
    upd = {}
    for field, table in (("department", DEPT), ("job_title", TITLE),
                         ("shift", SHIFT), ("production_line", LINE)):
        cur = u.get(field)
        if cur in table and table[cur] != cur:
            upd[field] = table[cur]
    if u["username"] in NAMES:
        upd["full_name"] = NAMES[u["username"]]
    if upd:
        db["users"].update_one({"_id": u["_id"]}, {"$set": upd})
        changed += 1

print(f"  cập nhật {changed} user\n")
for u in db["users"].find({}, {"username": 1, "full_name": 1, "role": 1, "department": 1,
                               "job_title": 1, "shift": 1, "production_line": 1}).sort("department", 1):
    print(f"  {u['username']:14} {str(u.get('full_name'))[:20]:22} {str(u.get('department')):12} "
          f"{str(u.get('job_title')):24} {str(u.get('shift')):22} {u.get('production_line')}")
