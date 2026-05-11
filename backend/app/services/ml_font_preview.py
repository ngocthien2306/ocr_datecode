import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from app.services.ml_ok_synthesize import (
    _FONT_DIR, _SYSTEM_FONT_FALLBACKS, _measure_stroke_ratio,
)


_PREVIEW_CACHE: Dict[Tuple[str, float, str, int, int, str, str], str] = {}


_DEFAULT_PREVIEW = "0123456789\nABCDEFGHIJKLMNOPQRSTUVWXYZ\nabcdefghijklmnopqrstuvwxyz"


def _ensure_chars(chars: str) -> str:
    chars = (chars or "").strip()
    if not chars:
        return _DEFAULT_PREVIEW
    return chars[:200]


def render_preview_b64(font_path: str, chars: str,
                       width: int = 420, height: int = 84,
                       bg_hex: str = "#ffffff", ink_hex: str = "#111111") -> Optional[str]:
    chars = _ensure_chars(chars)
    if not os.path.exists(font_path):
        return None
    try:
        mtime = os.path.getmtime(font_path)
    except OSError:
        return None
    key = (font_path, mtime, chars, width, height, bg_hex, ink_hex)
    if key in _PREVIEW_CACHE:
        return _PREVIEW_CACHE[key]

    text = "\n".join(" ".join(line) for line in chars.split("\n"))
    img = Image.new("RGB", (width, height), bg_hex)
    draw = ImageDraw.Draw(img)
    fs_lo, fs_hi = 8, height
    while fs_hi - fs_lo > 1:
        fs = (fs_lo + fs_hi) // 2
        try:
            font = ImageFont.truetype(font_path, fs)
        except Exception:
            return None
        l, t, r, b = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
        if (r - l) > width - 8 or (b - t) > height - 4:
            fs_hi = fs
        else:
            fs_lo = fs
    try:
        font = ImageFont.truetype(font_path, fs_lo)
    except Exception:
        return None
    l, t, r, b = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
    cx = (width - (r - l)) // 2 - l
    cy = (height - (b - t)) // 2 - t
    draw.multiline_text((cx, cy), text, font=font, fill=ink_hex, spacing=2)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    _PREVIEW_CACHE[key] = b64
    return b64


def list_fonts(preview_chars: str = "") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    if _FONT_DIR.is_dir():
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            for p in sorted(_FONT_DIR.glob(ext)):
                sp = str(p.resolve())
                if sp in seen:
                    continue
                seen.add(sp)
                ratio = _measure_stroke_ratio(sp)
                out.append({
                    "path": sp,
                    "filename": p.name,
                    "name": p.stem,
                    "stroke_ratio": round(ratio, 3) if ratio is not None else None,
                    "source": "project",
                    "preview_b64": render_preview_b64(sp, preview_chars),
                })
    for sp in _SYSTEM_FONT_FALLBACKS:
        if not os.path.exists(sp) or sp in seen:
            continue
        seen.add(sp)
        ratio = _measure_stroke_ratio(sp)
        out.append({
            "path": sp,
            "filename": Path(sp).name,
            "name": Path(sp).stem,
            "stroke_ratio": round(ratio, 3) if ratio is not None else None,
            "source": "system",
            "preview_b64": render_preview_b64(sp, preview_chars),
        })
    return out


def is_project_font(path: str) -> bool:
    try:
        return str(Path(path).resolve()).startswith(str(_FONT_DIR.resolve()))
    except Exception:
        return False


def clear_preview_cache():
    _PREVIEW_CACHE.clear()
