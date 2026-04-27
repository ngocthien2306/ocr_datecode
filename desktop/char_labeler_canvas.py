"""
Image canvas for the char-labeler tool.

Renders one source text image with N character bboxes overlaid. Supports:
  - Selecting a char (click)
  - Moving / resizing the selected bbox
  - Drawing a new bbox in DRAW mode
  - Color-coding by label (ok/ng/unlabeled) + per-char id text overlay
"""
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont
from PyQt5.QtWidgets import QLabel


# Color scheme (BGR-style as Qt RGB)
COLOR_OK         = QColor(40, 220, 80)
COLOR_NG         = QColor(240, 60, 60)
COLOR_UNLABELED  = QColor(160, 160, 160)
COLOR_SELECTED   = QColor(255, 200, 0)


class CharLabelerCanvas(QLabel):
    """
    Canvas that displays one text image + draggable per-char bboxes.

    Char data structure expected from caller (list of dicts):
      [{'bbox': (x, y, w, h), 'char_id': 'L', 'label': 'ok'|'ng'|None,
        'ocr_conf': 0.99}, ...]
    """
    MODE_SELECT = 0
    MODE_DRAW = 1
    HANDLE_SIZE = 6

    selectionChanged = pyqtSignal(int)         # idx (or -1)
    bboxesChanged = pyqtSignal()               # any add/edit/delete

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setStyleSheet("background-color: #1e1e1e;")
        self.setFocusPolicy(Qt.StrongFocus)

        self._pixmap = QPixmap()
        self._scale = 1.0
        self.chars = []                # list of dicts (see docstring)
        self.selected = -1
        self.mode = self.MODE_SELECT

        self._dragging = False
        self._drag_start = QPoint()
        self._drag_offset = QPoint()
        self._resize_corner = None     # 'nw'|'ne'|'sw'|'se'|None

    # -------------------------------------------------------------- public
    def set_image(self, bgr_img):
        if bgr_img is None or bgr_img.size == 0:
            self._pixmap = QPixmap()
            self.setPixmap(self._pixmap)
            return
        h, w = bgr_img.shape[:2]
        if bgr_img.ndim == 2:
            bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)
        buf = np.ascontiguousarray(bgr_img)
        qimg = QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped().copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_display()

    def set_chars(self, chars):
        """Replace chars list (each item: dict with 'bbox', 'char_id', 'label', 'ocr_conf')"""
        self.chars = [dict(c) for c in (chars or [])]
        self.selected = -1
        self.update()
        self.selectionChanged.emit(-1)

    def get_chars(self):
        return [dict(c) for c in self.chars]

    def set_mode(self, mode):
        self.mode = mode
        self._dragging = False
        self.update()

    def select_index(self, idx):
        if idx == self.selected:
            return
        if idx < -1 or idx >= len(self.chars):
            return
        self.selected = idx
        self.update()
        self.selectionChanged.emit(idx)

    def update_selected_label(self, label):
        """label ∈ 'ok' | 'ng' | None"""
        if 0 <= self.selected < len(self.chars):
            self.chars[self.selected]['label'] = label
            self.update()
            self.bboxesChanged.emit()

    def update_selected_char_id(self, char_id):
        if 0 <= self.selected < len(self.chars):
            self.chars[self.selected]['char_id'] = char_id
            self.update()
            self.bboxesChanged.emit()

    def delete_selected(self):
        if 0 <= self.selected < len(self.chars):
            del self.chars[self.selected]
            self.selected = min(self.selected, len(self.chars) - 1)
            self.update()
            self.selectionChanged.emit(self.selected)
            self.bboxesChanged.emit()

    def bulk_set_label(self, label):
        for c in self.chars:
            c['label'] = label
        self.update()
        self.bboxesChanged.emit()

    # -------------------------------------------------------------- helpers

    def _update_display(self):
        if self._pixmap.isNull():
            return
        target_w = max(800, self.width() if self.width() > 100 else 800)
        target_h = max(160, self.height() if self.height() > 100 else 160)
        sw = target_w / self._pixmap.width()
        sh = target_h / self._pixmap.height()
        self._scale = max(0.1, min(sw, sh, 4.0))
        scaled = self._pixmap.scaled(
            int(self._pixmap.width() * self._scale),
            int(self._pixmap.height() * self._scale),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.setFixedSize(scaled.size())
        self.setPixmap(scaled)
        self.update()

    def _to_image(self, p):
        return QPoint(int(p.x() / self._scale), int(p.y() / self._scale))

    def _to_disp(self, bbox):
        x, y, w, h = bbox
        s = self._scale
        return QRect(int(x * s), int(y * s), int(w * s), int(h * s))

    def _hit_corner(self, rect_disp, p):
        hs = self.HANDLE_SIZE
        corners = {
            'nw': QRect(rect_disp.left() - hs // 2, rect_disp.top() - hs // 2, hs, hs),
            'ne': QRect(rect_disp.right() - hs // 2, rect_disp.top() - hs // 2, hs, hs),
            'sw': QRect(rect_disp.left() - hs // 2, rect_disp.bottom() - hs // 2, hs, hs),
            'se': QRect(rect_disp.right() - hs // 2, rect_disp.bottom() - hs // 2, hs, hs),
        }
        for name, c in corners.items():
            if c.contains(p):
                return name
        return None

    def _bbox_contains(self, bbox, img_p):
        x, y, w, h = bbox
        return (x <= img_p.x() <= x + w and y <= img_p.y() <= y + h)

    def _color_for(self, label):
        if label == 'ok':
            return COLOR_OK
        if label == 'ng':
            return COLOR_NG
        return COLOR_UNLABELED

    # -------------------------------------------------------------- mouse

    def mousePressEvent(self, ev):
        if self._pixmap.isNull():
            return
        p = ev.pos()
        img_p = self._to_image(p)

        if self.mode == self.MODE_DRAW:
            self._dragging = True
            self._drag_start = img_p
            self.chars.append({
                'bbox': (img_p.x(), img_p.y(), 1, 1),
                'char_id': '?',
                'label': None,
                'ocr_conf': None,
            })
            self.selected = len(self.chars) - 1
            self.selectionChanged.emit(self.selected)
            self.update()
            return

        # SELECT mode: try resize handle on selected first
        if 0 <= self.selected < len(self.chars):
            sel_disp = self._to_disp(self.chars[self.selected]['bbox'])
            corner = self._hit_corner(sel_disp, p)
            if corner:
                self._dragging = True
                self._resize_corner = corner
                return

        clicked = -1
        for i in range(len(self.chars) - 1, -1, -1):
            if self._bbox_contains(self.chars[i]['bbox'], img_p):
                clicked = i
                break

        self.selected = clicked
        self.selectionChanged.emit(clicked)
        if clicked >= 0:
            self._dragging = True
            x, y, _, _ = self.chars[clicked]['bbox']
            self._drag_offset = QPoint(img_p.x() - x, img_p.y() - y)
            self._resize_corner = None
        self.update()

    def mouseMoveEvent(self, ev):
        if not self._dragging:
            return
        img_p = self._to_image(ev.pos())

        if self.mode == self.MODE_DRAW and 0 <= self.selected < len(self.chars):
            x0, y0 = self._drag_start.x(), self._drag_start.y()
            x1, y1 = img_p.x(), img_p.y()
            x = min(x0, x1); y = min(y0, y1)
            w = abs(x1 - x0); h = abs(y1 - y0)
            self.chars[self.selected]['bbox'] = (x, y, w, h)
            self.update()
            return

        if 0 <= self.selected < len(self.chars):
            x, y, w, h = self.chars[self.selected]['bbox']
            if self._resize_corner:
                if self._resize_corner == 'nw':
                    nx, ny = img_p.x(), img_p.y()
                    self.chars[self.selected]['bbox'] = (
                        nx, ny, max(2, x + w - nx), max(2, y + h - ny))
                elif self._resize_corner == 'ne':
                    self.chars[self.selected]['bbox'] = (
                        x, img_p.y(), max(2, img_p.x() - x), max(2, y + h - img_p.y()))
                elif self._resize_corner == 'sw':
                    self.chars[self.selected]['bbox'] = (
                        img_p.x(), y, max(2, x + w - img_p.x()), max(2, img_p.y() - y))
                elif self._resize_corner == 'se':
                    self.chars[self.selected]['bbox'] = (
                        x, y, max(2, img_p.x() - x), max(2, img_p.y() - y))
            else:
                nx = img_p.x() - self._drag_offset.x()
                ny = img_p.y() - self._drag_offset.y()
                self.chars[self.selected]['bbox'] = (nx, ny, w, h)
            self.update()

    def mouseReleaseEvent(self, _ev):
        self._dragging = False
        self._resize_corner = None
        # Drop tiny rects from DRAW mode
        if self.mode == self.MODE_DRAW and 0 <= self.selected < len(self.chars):
            x, y, w, h = self.chars[self.selected]['bbox']
            if w < 4 or h < 4:
                del self.chars[self.selected]
                self.selected = -1
                self.selectionChanged.emit(-1)
        self.bboxesChanged.emit()
        self.update()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        elif ev.key() == Qt.Key_Left:
            self.select_index(max(0, self.selected - 1) if self.selected > 0 else
                              max(0, len(self.chars) - 1))
        elif ev.key() == Qt.Key_Right:
            if self.chars:
                self.select_index((self.selected + 1) % len(self.chars))
        else:
            super().keyPressEvent(ev)

    # -------------------------------------------------------------- paint

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        for i, c in enumerate(self.chars):
            disp = self._to_disp(c['bbox'])
            base_color = self._color_for(c.get('label'))
            is_sel = (i == self.selected)
            border = COLOR_SELECTED if is_sel else base_color
            pen = QPen(border, 3 if is_sel else 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(disp)

            # Char id label above bbox
            cid = str(c.get('char_id') or '?')
            font = QFont('Helvetica', 10, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QPen(border, 1))
            painter.drawText(disp.left() + 1, max(12, disp.top() - 2), cid)

            # Index label below
            painter.setFont(QFont('Helvetica', 8))
            painter.drawText(disp.left() + 1, disp.bottom() + 11, f'#{i}')

            # Resize handles for selected
            if is_sel:
                painter.setBrush(QBrush(border))
                hs = self.HANDLE_SIZE
                for cx, cy in [(disp.left(), disp.top()),
                               (disp.right(), disp.top()),
                               (disp.left(), disp.bottom()),
                               (disp.right(), disp.bottom())]:
                    painter.drawRect(cx - hs // 2, cy - hs // 2, hs, hs)
