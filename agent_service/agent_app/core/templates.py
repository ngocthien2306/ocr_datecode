"""
Tra ảnh template ĐANG HOẠT ĐỘNG tại thời điểm một sản phẩm bị fail.

Vì sao cần: ảnh frame fail một mình không nói được sản phẩm sai ở đâu. Nhìn
"mong 'BESTifUsedbyAUG072028' → đọc 'BESTifUsedbyAUG182028'" thì biết là lệch một
cụm ký tự, nhưng không biết vùng OCR được đặt ở đâu trên nhãn, có bị cắt mất một
phần chữ không, ngưỡng conf là bao nhiêu. Đặt ảnh template cạnh ảnh fail thì trả
lời được ngay.

Điểm khó nằm ở chữ "tại thời điểm đó". Template bị chỉnh liên tục trong ngày —
recipe này có 190 lần load. Lấy template HIỆN TẠI để đối chiếu với một frame fail
lúc 14:47 là so sai vật: rất có thể template đã được sửa chính vì frame đó fail.
Nên phải tìm đúng bản load đang chạy vào lúc frame được chụp:

    loaded_at <= t  AND  (stopped_at IS NULL OR stopped_at > t)

Cùng logic với `ReceiptLoadRepository.get_template_images_at_timestamp` của
backend, nhưng đọc MongoDB trực tiếp — agent service không gọi endpoint backend
(xem docs/PIPELINE.md).
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent_app.db.mongodb import get_sync_database

logger = logging.getLogger(__name__)

# Vùng nào đáng vẽ lên ảnh. Bỏ `char` vì một nhãn có tới hàng chục ô ký tự — vẽ
# hết thì ảnh thành một tấm lưới, che mất đúng thứ cần xem.
_ROI_TYPES = ("crop_area", "template", "text")

# Số bản load nạp về cho mỗi recipe. Các mẫu fail của một câu hỏi nằm trong cùng
# một kỳ, nên vài chục bản gần nhất phủ hết; nạp cả 190 bản chỉ để tra 8 mẫu là
# đọc thừa.
_LOADS_PER_RECIPE = 60


def _loads_for(recipe_id: str, upto: datetime) -> List[Dict[str, Any]]:
    """
    Các bản load của một recipe, mới nhất trước, tính tới mốc `upto`.

    Nạp một lần rồi tra trong bộ nhớ thay vì mỗi mẫu một truy vấn: 8 mẫu là 8 lần
    đi Mongo, mà chúng gần như luôn rơi vào cùng vài bản load.
    """
    db = get_sync_database()
    cur = (db["receipt_loads"]
           .find({"recipe_id": recipe_id, "loaded_at": {"$lte": upto}},
                 {"loaded_at": 1, "stopped_at": 1, "loaded_by_full_name": 1,
                  "metadata.cameras.camera_id": 1,
                  "metadata.cameras.serial_number": 1,
                  "metadata.camera_templates": 1})
           .sort("loaded_at", -1)
           .limit(_LOADS_PER_RECIPE))
    return list(cur)


def _active_at(loads: List[Dict[str, Any]], ts: datetime) -> Optional[Dict[str, Any]]:
    """Bản load đang chạy vào lúc `ts`, hoặc None."""
    for doc in loads:                      # đã sort mới-nhất-trước
        loaded_at = doc.get("loaded_at")
        if not loaded_at or loaded_at > ts:
            continue
        stopped_at = doc.get("stopped_at")
        if stopped_at is not None and stopped_at <= ts:
            # Bản này đã bị dừng trước thời điểm cần tra. Không `continue` sang
            # bản cũ hơn: các bản cũ hơn cũng đã dừng trước đó, nên lúc `ts` dây
            # chuyền không chạy recipe này.
            return None
        return doc
    return None


def _camera_id_for(meta: Dict[str, Any], serial: str) -> Optional[str]:
    """`camera_templates` khoá theo `camera_id`, còn bản ghi fail chỉ có serial."""
    for cam in (meta.get("cameras") or []):
        if str(cam.get("serial_number")) == str(serial):
            return cam.get("camera_id")
    return None


def _rois(tpl: Dict[str, Any], expected: Optional[str]) -> List[Dict[str, Any]]:
    """
    Các vùng của template, toạ độ đã chuẩn hoá 0..1 nên vẽ được ở mọi kích thước.

    Vùng nào có `text` trùng giá trị hệ thống MONG ĐỢI thì được đánh dấu
    `highlight` — đó chính là vùng vừa trượt, và là chỗ người xem cần nhìn trước
    tiên. Không có nó thì người xem phải tự dò trong 3-4 khung xem khung nào liên
    quan tới dòng chữ "mong ... → đọc ...".
    """
    out = []
    for a in (tpl.get("annotations") or []):
        if a.get("type") not in _ROI_TYPES:
            continue
        text = a.get("text") or ""
        if expected:
            # Lỗi lệch chuỗi: vùng đáng nhìn là đúng vùng khai báo chuỗi đó.
            hl = bool(text and text == expected)
        else:
            # Không có `expected` nghĩa là frame trượt ở bước khác — thường là
            # detector không thấy gì. Khi đó vùng đáng nhìn là vùng OCR, vì nó cho
            # thấy chữ LẼ RA phải nằm ở đâu; đó chính là câu hỏi người xem đang có.
            hl = a.get("type") == "text"
        out.append({
            "type": a.get("type"),
            "text": text,
            "conf": a.get("conf"),
            "x": a.get("x"), "y": a.get("y"),
            "w": a.get("width"), "h": a.get("height"),
            "highlight": hl,
        })
    return out


def attach_templates(samples: List[Dict[str, Any]]) -> int:
    """
    Gắn `sample["template"]` cho từng mẫu fail. Trả về số mẫu gắn được.

    Mẫu nào không tra được thì KHÔNG có khoá `template` — lớp vẽ tự hiểu là chỉ
    hiện ảnh fail. Không bịa ra một template rỗng, và không dùng template hiện tại
    làm hàng thay thế: một tấm ảnh sai chỗ còn tệ hơn không có ảnh, vì người xem
    sẽ tin nó.
    """
    if not samples:
        return 0

    by_recipe: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        rid = s.get("recipe_id")
        ts = s.get("_ts_utc")
        if rid and isinstance(ts, datetime):
            by_recipe.setdefault(rid, []).append(s)

    done = 0
    for recipe_id, group in by_recipe.items():
        try:
            loads = _loads_for(recipe_id, max(s["_ts_utc"] for s in group))
        except Exception as e:
            logger.warning("Không đọc được receipt_loads của %s: %s", recipe_id, e)
            continue

        for s in group:
            load = _active_at(loads, s["_ts_utc"])
            if not load:
                continue
            meta = load.get("metadata") or {}
            cam_id = _camera_id_for(meta, s.get("camera"))
            if not cam_id:
                continue

            entry = next((ct for ct in (meta.get("camera_templates") or [])
                          if ct.get("camera_id") == cam_id), None)
            if not entry:
                continue

            # Khớp theo TÊN frame, không lấy template đầu tiên: mỗi camera có nhiều
            # frame ("Frame 3", "Frame 4") và bản ghi fail nói rõ nó thuộc frame
            # nào. Lấy bừa cái đầu là hiện ảnh của một frame khác — vẫn ra ảnh,
            # vẫn trông hợp lý, và sai.
            name = s.get("template_name")
            tpls = entry.get("templates") or []
            tpl = next((x for x in tpls if x.get("name") == name), None)
            if tpl is None:
                continue

            fn = str(tpl.get("image_url") or "").rsplit("/", 1)[-1]
            if not fn:
                continue

            s["template"] = {
                # Serve qua chính agent service: nó đã mount backend/uploads, nên
                # ảnh vẫn hiện khi backend đang restart.
                "url": "/api/uploads/templates/" + fn,
                "name": tpl.get("name"),
                "width": tpl.get("image_width"),
                "height": tpl.get("image_height"),
                "loaded_at": str(load.get("loaded_at") or "")[:19],
                "loaded_by": load.get("loaded_by_full_name"),
                "rois": _rois(tpl, s.get("expected")),
            }
            done += 1

    # Bỏ các khoá nội bộ. `_ts_utc` là `datetime` nên không JSON hoá được, còn
    # `_causes`/`_product` chỉ dùng để chọn mẫu. Dọn ở đây thay vì trông chờ tầng
    # trên lọc — chỗ này là điểm cuối cùng mọi mẫu đều đi qua.
    for s in samples:
        for k in ("_ts_utc", "_causes", "_product"):
            s.pop(k, None)

    return done
