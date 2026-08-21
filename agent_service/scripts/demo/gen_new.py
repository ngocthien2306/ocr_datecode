"""Sinh avatar cho các user giả lập mới. Áo xanh = operator, trắng = supervisor."""

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

import base64, os, sys, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(AGENT_ENV)
# Thư mục ảnh: tham số 1 nếu có, không thì mặc định _out/avatars — để
# replace_avatars.py tìm thấy mà không phải truyền đường dẫn hai lần.
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTDIR / "avatars"
OUT.mkdir(parents=True, exist_ok=True)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

FRAME = ("Photorealistic ID badge headshot, shoulders up, facing camera directly, "
         "neutral friendly expression, even soft studio lighting, plain light grey "
         "seamless background, sharp focus, natural skin texture. "
         "Fictional person, not a real individual. ")
VIET = "A Vietnamese person with East Asian features, black hair, warm light-tan skin tone. "
BLUE = ("Wearing a clean royal-blue short-sleeve factory work shirt with a collar, "
        "and a white disposable hairnet covering the hair. ")
WHITE = ("Wearing a clean white food-safety lab coat over a shirt, and a white "
         "disposable hairnet covering the hair. ")

PEOPLE = {
    "nv_vanhanh1": BLUE + VIET + "A woman in her early 30s, round face, small silver necklace.",
    "nv_vanhanh2": BLUE + VIET + "A man in his late 20s, square jaw, short black hair.",
    "truongca_a":  WHITE + VIET + "A man in his mid 40s, receding hairline under the hairnet, thin moustache, steady expression.",
    "qc_kiemtra":  WHITE + VIET + "A woman in her late 20s, oval face, wearing thin wire glasses.",
    "kt_baotri":   BLUE + VIET + "A man in his late 30s, weathered complexion, slight stubble, sturdy build.",
    "nv_kho":      BLUE + VIET + "A man in his early 50s, greying short hair under the hairnet, friendly crow's feet.",
}

# Khối văn phòng: không đồ xưởng, không lưới trùm tóc.
OFFICE = ("Wearing a well-tailored dark navy blazer over a crisp white dress shirt, "
          "no hairnet, neatly groomed hair, poised executive presence. ")

# Bộ ảnh riêng cho từng máy. Profile chỉ giữ (đồng phục, đặc điểm riêng) chứ
# không giữ prompt hoàn chỉnh: FRAME và VIET là phần phải giống hệt nhau giữa
# mọi ảnh, để 40 tấm trông như chụp cùng một buổi chứ không phải 40 phong cách.
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from profiles import active as _profile, banner as _banner
_P = _profile()
_banner()
if _P:
    _UNIFORM = {"blue": BLUE, "white": WHITE, "office": OFFICE}
    PEOPLE = {u: _UNIFORM[k] + VIET + d for u, (k, d) in _P["avatars"].items()}


def generate(prompt, label):
    """
    Sinh một ảnh, lùi lại và thử lại khi bị rate limit.

    gpt-image-1 giới hạn 5 ảnh/phút cho cả organization. Sinh cho nhiều máy là
    chắc chắn chạm trần, và bản cũ để nguyên 429 bay lên thành traceback — mất
    luôn cả những ảnh còn lại trong lượt. Ảnh đã có thì được bỏ qua nên chạy lại
    không tốn tiền, nhưng phải chạy lại bằng tay thì vẫn phiền.
    """
    from openai import RateLimitError
    delay = 20
    for attempt in range(6):
        try:
            return client.images.generate(model="gpt-image-1", prompt=prompt,
                                          size="1024x1024", n=1)
        except RateLimitError:
            if attempt == 5:
                raise
            print(f"  {label:13} rate limit, chờ {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)


for name, desc in PEOPLE.items():
    dst = OUT / f"{name}.png"
    if dst.exists():
        print(f"  {name:13} da co"); continue
    r = generate(FRAME + desc, name)
    dst.write_bytes(base64.b64decode(r.data[0].b64_json))
    print(f"  {name:13} {dst.stat().st_size/1024:7.1f} KB", flush=True)
