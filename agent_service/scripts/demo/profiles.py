"""
Hồ sơ nhân sự demo, tách riêng theo từng máy.

Vì sao cần: các script trong thư mục này vốn hardcode một bộ nhân sự duy nhất
(`STAFF` trong `seed_staff.py`, `PEOPLE` trong `gen_new.py`, `NAMES` trong
`to_english.py`, seed random cố định trong `sim_activity.py`). Chạy nguyên xi
lên nhiều máy thì mọi máy ra **cùng một bộ người, cùng một mẫu hoạt động** — đứng
cạnh nhau là lộ ngay đây là dữ liệu dựng.

Cách dùng: đặt biến môi trường `DEMO_MACHINE` trước khi chạy script.

    DEMO_MACHINE=M1 DEMO_ADMIN_PASSWORD='...' python3 seed_staff.py

Không đặt `DEMO_MACHINE` thì các script giữ nguyên hành vi cũ (bộ nhân sự gốc
đang chạy trên Auto2) — cố ý, để không phải sửa gì trên máy đã setup xong.

Mỗi profile gồm:
  seed      — seed random của `sim_activity.py`; khác nhau ⇒ mẫu giờ vào/ra khác nhau
  staff     — tài khoản TẠO MỚI qua API backend (sinh audit log `create_user`)
  existing  — hồ sơ bổ sung cho tài khoản CÓ SẴN trên máy đó (không tạo mới)
  names     — tên hiển thị đặt cho tài khoản có sẵn
  avatars   — (đồng phục, đặc điểm riêng) để dựng prompt sinh ảnh
  part_time — ca 4h ghi đè sau cùng, để thẻ có cả ca 4h lẫn ca 8h

Đồng phục theo đúng quy ước của xưởng (xem docs/DEMO_DATA.md §3):
  blue   → công nhân vận hành / kho: áo xanh + lưới trùm tóc
  white  → giám sát / QA: blouse trắng an toàn thực phẩm + lưới trùm tóc
  office → khối văn phòng: blazer navy, không lưới trùm tóc
"""

import os

# Tuple staff:    username, role, họ tên, prefix email, mã NV, bộ phận, chức vụ, ca, dây chuyền, ngày vào
# Tuple existing: username, mã NV, bộ phận, chức vụ, ca, dây chuyền, ngày vào
#
# Bộ phận / chức vụ / ca viết tiếng Việt, đúng khoá mà `to_english.py` biết đổi.
# Viết thẳng tiếng Anh vào đây thì bước 2 của pipeline không còn gì để làm, và
# lần sau ai thêm người mới sẽ không biết phải theo bảng nào.

PROFILES = {
    # ─────────────────────────────────────────────────────────────────────────
    "M1": {
        "seed": 20260821,
        "staff": [
            ("nv_dungmay1", "operator",   "Nguyễn Thị Bích Ngọc", "ngoc.ntb",  "NV-2011", "Sản xuất", "Công nhân vận hành",            "Ca A (06:00–14:00)", "Line 1",   "2022-04-11"),
            ("nv_dungmay2", "operator",   "Lý Văn Cường",         "cuong.lv",  "NV-2034", "Sản xuất", "Công nhân vận hành",            "Ca B (14:00–22:00)", "Line 1",   "2023-08-22"),
            ("truongca_m1", "supervisor", "Trịnh Quang Huy",      "huy.tq",    "NV-0412", "Sản xuất", "Trưởng ca",                     "Ca A (06:00–14:00)", "Line 1-2", "2018-05-14"),
            ("qc_soi1",     "supervisor", "Hoàng Thị Kim Chi",    "chi.htk",   "NV-0803", "QA/QC",    "Nhân viên kiểm tra chất lượng", "Ca hành chính",      "Toàn bộ",  "2021-11-30"),
            ("bt_dien1",    "operator",   "Đinh Công Toàn",       "toan.dc",   "NV-0611", "Bảo trì",  "Kỹ thuật viên bảo trì",         "Ca hành chính",      "Toàn bộ",  "2020-02-19"),
            ("kho_nhap1",   "viewer",     "Mai Văn Lợi",          "loi.mv",    "NV-0955", "Kho vận",  "Nhân viên kho",                 "Ca hành chính",      "—",        "2019-09-08"),
        ],
        "existing": [
            ("admin",      "NV-0001", "Ban điều hành", "Quản trị hệ thống",  "Ca hành chính",      "—",        "2018-01-02"),
            ("supervisor", "NV-0207", "Sản xuất",      "Giám sát sản xuất",  "Ca C (22:00–06:00)", "Line 1-2", "2020-03-09"),
            ("operator",   "NV-2108", "Sản xuất",      "Công nhân vận hành", "Ca C (22:00–06:00)", "Line 1",   "2024-07-01"),
        ],
        "names": {
            "supervisor": "Nguyễn Thị Thanh Vân",
            "operator":   "Trần Đức Anh",
        },
        "avatars": {
            "nv_dungmay1": ("blue",  "A woman in her early 30s, round face, small silver necklace, gentle smile."),
            "nv_dungmay2": ("blue",  "A man in his early 30s, broad forehead, short black hair under the hairnet, calm gaze."),
            "truongca_m1": ("white", "A man in his mid 40s, receding hairline under the hairnet, thin moustache, steady authoritative expression."),
            "qc_soi1":     ("white", "A woman in her late 20s, oval face, thin wire-rimmed glasses, hair tucked neatly under the hairnet."),
            "bt_dien1":    ("blue",  "A man in his late 30s, weathered complexion, light stubble, sturdy build, direct look."),
            "kho_nhap1":   ("blue",  "A man in his early 50s, greying short hair under the hairnet, friendly crow's feet."),
            "supervisor":  ("white", "A woman in her late 30s, long black hair tucked under the hairnet, composed confident expression."),
            "operator":    ("blue",  "A man in his mid 20s, slim face, short black hair under the hairnet, attentive expression."),
        },
        "part_time": {
            "bt_dien1":  "Shift P (12:00–16:00)",
            "kho_nhap1": "Shift P (11:00–15:00)",
        },
    },
    # ─────────────────────────────────────────────────────────────────────────
    "M2": {
        "seed": 20260822,
        "staff": [
            ("nv_maych1",    "operator",   "Phan Thị Thuỳ Dương", "duong.ptt", "NV-3120", "Sản xuất", "Công nhân vận hành",     "Ca B (14:00–22:00)", "Line 2",   "2023-01-16"),
            ("nv_maych2",    "operator",   "Tạ Minh Khoa",        "khoa.tm",   "NV-3145", "Sản xuất", "Công nhân vận hành",     "Ca C (22:00–06:00)", "Line 2",   "2024-05-27"),
            ("truongca_m2",  "supervisor", "Lương Thế Vinh",      "vinh.lt",   "NV-0527", "Sản xuất", "Trưởng ca",              "Ca C (22:00–06:00)", "Line 2-3", "2017-12-04"),
            ("qa_giamsat2",  "supervisor", "Chu Thị Hải Yến",     "yen.cth",   "NV-0692", "QA/QC",    "Giám sát QA",            "Ca hành chính",      "Toàn bộ",  "2020-07-21"),
            ("bt_cokhi2",    "operator",   "Nguyễn Hữu Phước",    "phuoc.nh",  "NV-0648", "Bảo trì",  "Kỹ thuật viên bảo trì",  "Ca hành chính",      "Toàn bộ",  "2022-03-09"),
            ("kho_xuat2",    "viewer",     "Trương Văn Bảo",      "bao.tv",    "NV-0977", "Kho vận",  "Nhân viên kho",          "Ca hành chính",      "—",        "2021-06-15"),
        ],
        "existing": [
            ("admin",      "NV-0001", "Ban điều hành", "Quản trị hệ thống",  "Ca hành chính",      "—",        "2018-01-02"),
            ("supervisor", "NV-0219", "Sản xuất",      "Giám sát sản xuất",  "Ca A (06:00–14:00)", "Line 2-3", "2019-08-26"),
            ("operator",   "NV-3172", "Sản xuất",      "Công nhân vận hành", "Ca A (06:00–14:00)", "Line 2",   "2025-01-13"),
        ],
        "names": {
            "supervisor": "Phạm Thị Lệ Quyên",
            "operator":   "Nguyễn Văn Thành",
        },
        "avatars": {
            "nv_maych1":   ("blue",  "A woman in her mid 20s, straight black hair tucked under the hairnet, small stud earrings, bright alert eyes."),
            "nv_maych2":   ("blue",  "A man in his late 20s, square jaw, short black hair under the hairnet, serious expression."),
            "truongca_m2": ("white", "A man in his early 50s, salt-and-pepper eyebrows, square black-framed glasses, experienced steady look."),
            "qa_giamsat2": ("white", "A woman in her early 40s, black hair tucked under the hairnet, wearing a lanyard with an ID card."),
            "bt_cokhi2":   ("blue",  "A man in his early 40s, tanned skin, strong jawline, faint scar on the left cheek."),
            "kho_xuat2":   ("blue",  "A man in his late 40s, rounded face, thick eyebrows, warm easygoing smile."),
            "supervisor":  ("white", "A woman in her mid 40s, fine lines around the eyes, hair tucked under the hairnet, quietly authoritative."),
            "operator":    ("blue",  "A man in his early 30s, thin black-framed glasses, short hair under the hairnet."),
            # Chỉ máy này mới có `admin` trong danh sách: `avatar_url` của admin
            # trên M2 trỏ vào một file đã không còn trong backend/uploads/avatar
            # (hỏng sẵn từ trước), nên thẻ hiện ảnh vỡ. Các máy khác admin vẫn có
            # ảnh riêng đọc được ⇒ để nguyên, không đụng tài khoản đang dùng thật.
            "admin":       ("office", "A man in his late 40s, short neatly combed black hair with slight grey at the temples, thin rimless glasses, calm authoritative expression."),
        },
        "part_time": {
            "bt_cokhi2": "Shift P (13:00–17:00)",
            "kho_xuat2": "Shift P (09:00–13:00)",
        },
    },
    # ─────────────────────────────────────────────────────────────────────────
    "LineTine": {
        "seed": 20260823,
        "staff": [
            ("nv_dongoi1",    "operator",   "Bùi Thị Ngọc Trâm", "tram.btn",  "NV-4108", "Sản xuất", "Công nhân vận hành",            "Ca A (06:00–14:00)", "Line 1",   "2024-02-05"),
            ("nv_dongoi2",    "operator",   "Đoàn Văn Tiến",     "tien.dv",   "NV-4133", "Sản xuất", "Công nhân vận hành",            "Ca B (14:00–22:00)", "Line 1",   "2023-10-12"),
            ("truongca_tine", "supervisor", "Vương Đức Thịnh",   "thinh.vd",  "NV-0356", "Sản xuất", "Trưởng ca",                     "Ca B (14:00–22:00)", "Line 1-2", "2019-03-28"),
            ("qc_tine",       "supervisor", "Lâm Thị Tuyết Mai", "mai.ltt",   "NV-0759", "QA/QC",    "Nhân viên kiểm tra chất lượng", "Ca hành chính",      "Toàn bộ",  "2022-08-01"),
            ("bt_tine",       "operator",   "Hồ Sỹ Nguyên",      "nguyen.hs", "NV-0583", "Bảo trì",  "Kỹ thuật viên bảo trì",         "Ca hành chính",      "Toàn bộ",  "2021-01-25"),
            ("kho_tine",      "viewer",     "Ngô Văn Quý",       "quy.nv",    "NV-0918", "Kho vận",  "Nhân viên kho",                 "Ca hành chính",      "—",        "2020-05-19"),
        ],
        "existing": [
            ("admin",      "NV-0001", "Ban điều hành", "Quản trị hệ thống",  "Ca hành chính",      "—",        "2018-01-02"),
            ("supervisor", "NV-0233", "Sản xuất",      "Giám sát sản xuất",  "Ca C (22:00–06:00)", "Line 1-2", "2021-02-11"),
            ("operator",   "NV-4160", "Sản xuất",      "Công nhân vận hành", "Ca C (22:00–06:00)", "Line 2",   "2024-11-04"),
            ("sup_test",   "NV-0771", "QA/QC",         "Giám sát QA",        "Ca hành chính",      "Toàn bộ",  "2022-05-30"),
            ("op_test",    "NV-4185", "Sản xuất",      "Công nhân vận hành", "Ca A (06:00–14:00)", "Line 2",   "2025-03-17"),
        ],
        "names": {
            "supervisor": "Đỗ Thị Hồng Loan",
            "operator":   "Vũ Minh Quân",
            "sup_test":   "Nguyễn Thị Bảo Châu",
            "op_test":    "Lê Hoàng Phúc",
        },
        "avatars": {
            "nv_dongoi1":    ("blue",  "A woman in her early 20s, heart-shaped face, hair tucked under the hairnet, shy friendly smile."),
            "nv_dongoi2":    ("blue",  "A man in his mid 30s, high cheekbones, short black hair under the hairnet, focused expression."),
            "truongca_tine": ("white", "A man in his late 40s, close-cropped greying hair under the hairnet, deep-set eyes, firm expression."),
            "qc_tine":       ("white", "A woman in her mid 30s, round wire glasses, hair tucked under the hairnet, precise attentive look."),
            "bt_tine":       ("blue",  "A man in his early 30s, lean face, short beard shadow, alert practical expression."),
            "kho_tine":      ("blue",  "A man in his mid 50s, weathered kind face, greying hair under the hairnet."),
            "supervisor":    ("white", "A woman in her early 50s, hair tucked under the hairnet, reading glasses hanging on a cord."),
            "operator":      ("blue",  "A man in his late 20s, wide friendly face, short hair under the hairnet."),
            "sup_test":      ("white", "A woman in her late 20s, long face, thin silver-rimmed glasses, neat appearance."),
            "op_test":       ("blue",  "A young man in his early 20s, boyish features, short hair under the hairnet, eager expression."),
        },
        "part_time": {
            "bt_tine":  "Shift P (10:00–14:00)",
            "kho_tine": "Shift P (14:00–18:00)",
        },
    },
    # ─────────────────────────────────────────────────────────────────────────
    "PC-Auto-1": {
        "seed": 20260824,
        "staff": [
            ("nv_auto1",      "operator",   "Cao Thị Mỹ Linh",   "linh.ctm",  "NV-5102", "Sản xuất", "Công nhân vận hành",    "Ca A (06:00–14:00)", "Line 1",   "2023-06-13"),
            ("nv_auto2",      "operator",   "Nguyễn Bá Lộc",     "loc.nb",    "NV-5127", "Sản xuất", "Công nhân vận hành",    "Ca C (22:00–06:00)", "Line 1",   "2024-09-02"),
            ("truongca_auto", "supervisor", "Đặng Trọng Nghĩa",  "nghia.dt",  "NV-0284", "Sản xuất", "Trưởng ca",             "Ca C (22:00–06:00)", "Line 1-2", "2016-10-17"),
            ("qa_auto",       "supervisor", "Tô Thị Hạnh Dung",  "dung.tth",  "NV-0715", "QA/QC",    "Giám sát QA",           "Ca hành chính",      "Toàn bộ",  "2021-04-06"),
            ("bt_auto",       "operator",   "Phạm Anh Tuấn",     "tuan.pa",   "NV-0669", "Bảo trì",  "Kỹ thuật viên bảo trì", "Ca hành chính",      "Toàn bộ",  "2019-11-11"),
            ("kho_auto",      "viewer",     "Lê Văn Hoà",        "hoa.lv",    "NV-0932", "Kho vận",  "Nhân viên kho",         "Ca hành chính",      "—",        "2022-12-20"),
        ],
        "existing": [
            ("admin",      "NV-0001", "Ban điều hành", "Quản trị hệ thống",  "Ca hành chính",      "—",        "2018-01-02"),
            ("supervisor", "NV-0246", "Sản xuất",      "Giám sát sản xuất",  "Ca B (14:00–22:00)", "Line 1-2", "2018-09-24"),
            ("operator",   "NV-5188", "Sản xuất",      "Công nhân vận hành", "Ca B (14:00–22:00)", "Line 2",   "2025-05-08"),
            ("suntech",    "NV-5203", "Sản xuất",      "Công nhân vận hành", "Ca A (06:00–14:00)", "Line 2",   "2024-04-15"),
            ("suntech1",   "NV-5219", "Sản xuất",      "Công nhân vận hành", "Ca B (14:00–22:00)", "Line 2",   "2024-10-28"),
            ("admin123",   "NV-0838", "QA/QC",         "Nhân viên QA",       "Ca hành chính",      "Toàn bộ",  "2023-02-06"),
            ("admin321",   "NV-0851", "QA/QC",         "Nhân viên QA",       "Ca A (06:00–14:00)", "Line 1",   "2023-07-19"),
        ],
        "names": {
            "supervisor": "Trần Thị Kim Oanh",
            "operator":   "Bùi Xuân Trường",
            "suntech":    "Nguyễn Tấn Đạt",
            "suntech1":   "Phạm Thị Diễm My",
            "admin123":   "Lê Quốc Hùng",
            "admin321":   "Đặng Thị Ánh Tuyết",
        },
        "avatars": {
            "nv_auto1":      ("blue",  "A woman in her late 20s, soft oval face, hair tucked under the hairnet, quiet composed smile."),
            "nv_auto2":      ("blue",  "A man in his early 40s, strong brow, short black hair under the hairnet, weathered hands visible."),
            "truongca_auto": ("white", "A man in his mid 50s, greying hair under the hairnet, rectangular glasses, veteran supervisor presence."),
            "qa_auto":       ("white", "A woman in her late 30s, sharp features, hair tucked under the hairnet, lanyard with an ID card."),
            "bt_auto":       ("blue",  "A man in his mid 30s, athletic build, short hair under the hairnet, confident practical look."),
            "kho_auto":      ("blue",  "A man in his early 50s, thin frame, greying temples under the hairnet, patient expression."),
            "supervisor":    ("white", "A woman in her early 40s, hair tucked under the hairnet, small pearl earrings, calm assured expression."),
            "operator":      ("blue",  "A man in his late 30s, rounded face, short hair under the hairnet, easy smile."),
            "suntech":       ("blue",  "A man in his mid 20s, narrow face, short spiky hair under the hairnet, keen expression."),
            "admin123":      ("white", "A man in his early 30s, square black-framed glasses, hair tucked under the hairnet, methodical look."),
            "admin321":      ("white", "A woman in her mid 20s, long face, hair tucked under the hairnet, notebook-in-hand attentiveness."),
        },
        "part_time": {
            "bt_auto":  "Shift P (08:00–12:00)",
            "kho_auto": "Shift P (15:00–19:00)",
        },
    },
}

MACHINE = (os.environ.get("DEMO_MACHINE") or "").strip()


def active():
    """
    Profile của máy đang chạy, hoặc None nếu không đặt `DEMO_MACHINE`.

    Trả None (chứ không raise) là cố ý: script gọi hàm này rồi tự quyết định
    dùng bộ hardcode cũ, nên máy đã setup xong không cần sửa gì.
    """
    if not MACHINE:
        return None
    if MACHINE not in PROFILES:
        raise SystemExit(
            f"DEMO_MACHINE={MACHINE!r} không có trong profiles.py.\n"
            f"  Hợp lệ: {', '.join(sorted(PROFILES))}"
        )
    return PROFILES[MACHINE]


def banner():
    """In một dòng cho biết đang chạy theo profile nào — để không seed nhầm máy."""
    print(f"── profile: {MACHINE or 'MẶC ĐỊNH (bộ gốc, Auto2)'} ──")
