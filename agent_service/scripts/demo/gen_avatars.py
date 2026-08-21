"""
Sinh ảnh chân dung nhân sự bằng OpenAI image API.

Nhân vật HƯ CẤU dùng làm avatar mẫu cho hệ thống test. Đồng phục theo đúng quy
ước của xưởng: operator áo xanh dương, supervisor áo trắng; admin là khối văn
phòng nên mặc sơ mi / blazer thay vì đồ xưởng.
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

import base64, os, sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(AGENT_ENV)

# Ở chế độ theo máy, `gen_new.py` đã sinh ảnh cho CẢ tài khoản mới lẫn tài khoản
# có sẵn (profile giữ chung một bảng `avatars`). Chạy tiếp script này sẽ ghi đè
# `supervisor.png` / `operator.png` bằng khuôn mặt của bộ gốc — đúng thứ ta đang
# cố tránh. Nên thoát sớm thay vì lặng lẽ làm hỏng.
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from profiles import active as _profile
if _profile():
    raise SystemExit("DEMO_MACHINE đang bật ⇒ dùng gen_new.py (đã bao gồm tài khoản có sẵn). Bỏ qua.")
# Thư mục ảnh: tham số 1 nếu có, không thì mặc định _out/avatars — để
# replace_avatars.py tìm thấy mà không phải truyền đường dẫn hai lần.
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTDIR / "avatars"
OUT.mkdir(parents=True, exist_ok=True)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Chung cho mọi ảnh: khung ảnh thẻ, ánh sáng đều, nền xám trơn.
FRAME = ("Photorealistic ID badge headshot, shoulders up, facing camera directly, "
         "neutral friendly expression, even soft studio lighting, plain light grey "
         "seamless background, sharp focus, natural skin texture, realistic pores, "
         "documentary corporate photography. Fictional person, not a real individual. ")

VIET = "A Vietnamese person with East Asian features, black hair, warm light-tan skin tone. "

# Đồng phục xưởng thực phẩm
BLUE = ("Wearing a clean royal-blue short-sleeve factory work shirt with a collar, "
        "and a white disposable hairnet covering the hair. ")
WHITE = ("Wearing a clean white food-safety lab coat over a shirt, and a white "
         "disposable hairnet covering the hair. ")
# Khối văn phòng: sang hơn, không đồ xưởng
OFFICE = ("Wearing a well-tailored dark navy blazer over a crisp white dress shirt, "
          "no hairnet, neatly groomed hair, poised executive presence. ")

PEOPLE = {
    "admin":         OFFICE + VIET + "A man in his late 40s, short neatly combed black hair with slight grey at the temples, thin rimless glasses, calm authoritative expression.",
    "supervisor":    WHITE + VIET + "A woman in her late 30s, black hair tucked under the hairnet, confident composed expression.",
    "operator":      BLUE + VIET + "A man in his mid 20s, short black hair under the hairnet, attentive expression.",
    "qa_supervisor": WHITE + VIET + "A woman in her early 40s, black hair tucked under the hairnet, wearing a lanyard with an ID card.",
    "qa_operator1":  BLUE + VIET + "A man in his early 30s, black-framed glasses, short black hair under the hairnet.",
    "qa_operator2":  BLUE + VIET + "A woman in her mid 20s, straight black hair tucked under the hairnet, small stud earrings.",
}

for name, desc in PEOPLE.items():
    dst = OUT / f"{name}.png"
    if dst.exists():
        print(f"  {name:14} da co, bo qua"); continue
    r = client.images.generate(model="gpt-image-1", prompt=FRAME + desc,
                               size="1024x1024", n=1)
    dst.write_bytes(base64.b64decode(r.data[0].b64_json))
    print(f"  {name:14} {dst.stat().st_size/1024:7.1f} KB")
