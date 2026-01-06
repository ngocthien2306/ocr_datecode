import os
import sys
import json
import requests
import cv2
import numpy as np
from typing import Tuple

API_BASE = os.environ.get("API_BASE", "https://suntech-vision-api.ngrok.app")


def get_latest_json() -> dict:
    url = f"{API_BASE}/api/recipes/loads/latest"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def hex_to_bgr(hexcolor: str) -> Tuple[int, int, int]:
    try:
        h = hexcolor.lstrip('#')
        if len(h) == 3:
            h = ''.join(c*2 for c in h)
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return (b, g, r)
    except Exception:
        return (0, 0, 255)


def make_absolute(url: str) -> str:
    if not url:
        return url
    if url.startswith('http://') or url.startswith('https://'):
        return url
    return API_BASE.rstrip('/') + url


def draw_annotations(img: np.ndarray, anns: list) -> np.ndarray:
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ann in anns or []:
        shape = (ann.get('shape') or ann.get('type') or '').lower()
        color = ann.get('color') or ann.get('stroke') or '#ff0000'
        bgr = hex_to_bgr(color)
        label = ann.get('text') or ann.get('type') or ''

        if shape in ('rectangle', 'rect', 'template'):
            try:
                x = float(ann.get('x', 0))
                y = float(ann.get('y', 0))
                ww = float(ann.get('width') or ann.get('w') or 0)
                hh = float(ann.get('height') or ann.get('h') or 0)
            except Exception:
                continue
            x1 = int(round(x * w if 0.0 <= x <= 1.0 else x))
            y1 = int(round(y * h if 0.0 <= y <= 1.0 else y))
            x2 = int(round((x + ww) * w if 0.0 <= x + ww <= 1.0 else (x + ww)))
            y2 = int(round((y + hh) * h if 0.0 <= y + hh <= 1.0 else (y + hh)))
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, max(2, int(w/800)))

            if label:
                txt = str(label)
                scale = max(0.5, w / 1000.0)
                thickness = max(1, int(scale))
                (tw, th), baseline = cv2.getTextSize(txt, font, scale, thickness)
                bx1 = x1
                by1 = max(0, y1 - th - 6)
                bx2 = bx1 + tw + 6
                by2 = by1 + th + 4
                cv2.rectangle(img, (bx1, by1), (bx2, by2), bgr, -1)
                cv2.putText(img, txt, (bx1 + 3, by1 + th - 1), font, scale, (255,255,255), thickness, cv2.LINE_AA)

        elif shape in ('polygon', 'poly') or ann.get('points'):
            pts = ann.get('points') or ann.get('polygon') or []
            proc = []
            for p in pts:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                try:
                    px = float(p[0]); py = float(p[1])
                except Exception:
                    continue
                px_i = int(round(px * w if 0.0 <= px <= 1.0 else px))
                py_i = int(round(py * h if 0.0 <= py <= 1.0 else py))
                proc.append((px_i, py_i))
            if len(proc) >= 2:
                pts_np = np.array(proc, np.int32).reshape((-1,1,2))
                cv2.polylines(img, [pts_np], isClosed=True, color=bgr, thickness=max(2, int(w/800)))
                if label:
                    lx, ly = proc[0]
                    txt = str(label)
                    scale = max(0.5, w / 1000.0)
                    thickness = max(1, int(scale))
                    (tw, th), baseline = cv2.getTextSize(txt, font, scale, thickness)
                    bx1 = lx
                    by1 = max(0, ly - th - 6)
                    bx2 = bx1 + tw + 6
                    by2 = by1 + th + 4
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), bgr, -1)
                    cv2.putText(img, txt, (bx1 + 3, by1 + th - 1), font, scale, (255,255,255), thickness, cv2.LINE_AA)

    return img


def show_latest_template():
    data = get_latest_json()
    tpl = None
    try:
        tpl = data['metadata']['camera_templates'][0]['templates'][0]
    except Exception:
        print('No template found in latest load metadata')
        return

    image_url = tpl.get('image_url')
    if not image_url:
        print('No image_url for template')
        return
    img_url = make_absolute(image_url)
    print('Downloading image from', img_url)
    r = requests.get(img_url, timeout=10)
    r.raise_for_status()
    nparr = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print('Failed to decode image')
        return

    annotations = tpl.get('annotations') or []
    annotated = draw_annotations(img, annotations)
    cv2.imwrite("latest_template_visualization.jpg", annotated)
    win = 'Latest template visualization'
    # cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # cv2.imshow(win, annotated)
    # print('Press any key in image window to exit')
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def main():
    # print JSON
    data = get_latest_json()
    print('Latest load event:')
    print(json.dumps(data, indent=2, ensure_ascii=False))

    if len(sys.argv) > 1 and sys.argv[1] in ('--show', 'show'):
        try:
            show_latest_template()
        except Exception as e:
            print('Error showing visualization:', e)


if __name__ == '__main__':
    main()
