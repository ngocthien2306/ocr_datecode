"""
Char editor dialog — annotate per-character bboxes inside a text/datecode
region. Bboxes can be auto-proposed via segment_characters and then edited
manually. Result is stored back in the parent region's `chars` field.
"""
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QMessageBox, QWidget, QScrollArea, QSizePolicy, QComboBox,
)


class _CharCanvas(QLabel):
    """
    Image-display label with editable per-char rectangles drawn on top.
    Modes:
      SELECT: click rect to select; drag to move; drag corner to resize; Delete to remove
      DRAW:   click+drag to add new rect
    """
    MODE_SELECT = 0
    MODE_DRAW = 1
    HANDLE_SIZE = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setStyleSheet("background-color: #1e1e1e;")
        self._pixmap = QPixmap()
        self._scale = 1.0           # display scale relative to source image
        self.rects = []             # list[QRect] in IMAGE coords
        self.selected = -1
        self.mode = self.MODE_SELECT

        self._dragging = False
        self._drag_start = QPoint()
        self._drag_offset = QPoint()
        self._resize_corner = None  # one of 'nw','ne','sw','se' or None

    def set_image(self, bgr_img):
        """Set the image to display (numpy BGR). Resets rects."""
        if bgr_img is None or bgr_img.size == 0:
            self._pixmap = QPixmap()
            self.setPixmap(self._pixmap)
            return
        h, w, ch = bgr_img.shape if bgr_img.ndim == 3 else (*bgr_img.shape, 1)
        if ch == 1:
            bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)
            ch = 3
        buf = np.ascontiguousarray(bgr_img)
        qimg = QImage(buf.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped().copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_display()

    def _update_display(self):
        if self._pixmap.isNull():
            return
        # Fit to a sensible display size while keeping nice integer scale
        target_w = max(800, self.width() if self.width() > 0 else 800)
        target_h = max(200, self.height() if self.height() > 0 else 200)
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

    def set_rects(self, rects):
        self.rects = [QRect(r) for r in rects]
        self.selected = -1
        self.update()

    def get_rects(self):
        return [QRect(r) for r in self.rects]

    def set_mode(self, mode):
        self.mode = mode
        self._dragging = False
        self.update()

    def delete_selected(self):
        if 0 <= self.selected < len(self.rects):
            del self.rects[self.selected]
            self.selected = -1
            self.update()

    # --------- coordinate helpers (display ↔ image)
    def _to_image(self, p):
        return QPoint(int(p.x() / self._scale), int(p.y() / self._scale))

    def _to_display_rect(self, r):
        s = self._scale
        return QRect(int(r.x() * s), int(r.y() * s),
                      int(r.width() * s), int(r.height() * s))

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

    # --------- mouse events
    def mousePressEvent(self, ev):
        p = ev.pos()
        img_p = self._to_image(p)

        if self.mode == self.MODE_DRAW:
            self._dragging = True
            self._drag_start = img_p
            self.rects.append(QRect(img_p, img_p))
            self.selected = len(self.rects) - 1
            return

        # SELECT mode
        # Check resize handle on currently selected first
        if 0 <= self.selected < len(self.rects):
            sel_disp = self._to_display_rect(self.rects[self.selected])
            corner = self._hit_corner(sel_disp, p)
            if corner:
                self._dragging = True
                self._resize_corner = corner
                return

        # Try selecting a rect under cursor (top to bottom)
        clicked_idx = -1
        for i in range(len(self.rects) - 1, -1, -1):
            if self.rects[i].contains(img_p):
                clicked_idx = i
                break
        self.selected = clicked_idx
        if clicked_idx >= 0:
            self._dragging = True
            self._drag_offset = img_p - self.rects[clicked_idx].topLeft()
            self._resize_corner = None
        self.update()

    def mouseMoveEvent(self, ev):
        if not self._dragging:
            return
        img_p = self._to_image(ev.pos())

        if self.mode == self.MODE_DRAW:
            r = QRect(self._drag_start, img_p).normalized()
            self.rects[self.selected] = r
            self.update()
            return

        if 0 <= self.selected < len(self.rects):
            r = self.rects[self.selected]
            if self._resize_corner:
                # Resize from corner
                if self._resize_corner == 'nw':
                    r.setTopLeft(img_p)
                elif self._resize_corner == 'ne':
                    r.setTopRight(img_p)
                elif self._resize_corner == 'sw':
                    r.setBottomLeft(img_p)
                elif self._resize_corner == 'se':
                    r.setBottomRight(img_p)
                self.rects[self.selected] = r.normalized()
            else:
                # Translate
                new_tl = img_p - self._drag_offset
                self.rects[self.selected] = QRect(new_tl, r.size())
            self.update()

    def mouseReleaseEvent(self, _ev):
        self._dragging = False
        self._resize_corner = None
        # Drop zero-area rects
        if self.mode == self.MODE_DRAW and 0 <= self.selected < len(self.rects):
            r = self.rects[self.selected]
            if r.width() < 3 or r.height() < 3:
                del self.rects[self.selected]
                self.selected = -1
        self.update()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(ev)

    # --------- painting
    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        for i, r in enumerate(self.rects):
            disp = self._to_display_rect(r)
            color = QColor(255, 200, 0) if i == self.selected else QColor(0, 220, 0)
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(disp)
            # Index label
            painter.setPen(QPen(color, 1))
            painter.drawText(disp.topLeft() + QPoint(2, -2), str(i))
            # Resize handles for selected
            if i == self.selected:
                painter.setBrush(QBrush(color))
                hs = self.HANDLE_SIZE
                for cx, cy in [(disp.left(), disp.top()),
                               (disp.right(), disp.top()),
                               (disp.left(), disp.bottom()),
                               (disp.right(), disp.bottom())]:
                    painter.drawRect(cx - hs // 2, cy - hs // 2, hs, hs)


class CharEditorDialog(QDialog):
    """
    Open from the Annotation tab to mark per-character bboxes inside a text
    region. Stores result back via `get_chars()` as a list of dicts in JSON
    format compatible with BoundingBox.chars.
    """

    def __init__(self, region_image_bgr, existing_chars=None,
                 region_label="text", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit chars — {region_label}")
        self.resize(1100, 600)

        self.region_image = region_image_bgr
        self._build_ui()

        # Convert existing chars (image coords, may be polygon or rectangle) to QRects
        # for editing. We display in axis-aligned rectangles; polygons get reduced
        # to their bounding box during edit and saved back as rectangles.
        rects = []
        for c in (existing_chars or []):
            if c.get('shape') == 'polygon' and 'points' in c:
                pts = c['points']
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                rects.append(QRect(int(min(xs)), int(min(ys)),
                                    int(max(xs) - min(xs)),
                                    int(max(ys) - min(ys))))
            else:
                rects.append(QRect(int(c.get('x', 0)), int(c.get('y', 0)),
                                    int(c.get('width', 0)), int(c.get('height', 0))))
        self.canvas.set_image(self.region_image)
        self.canvas.set_rects(rects)
        self._refresh_count()

    def _build_ui(self):
        outer = QVBoxLayout(self)

        # Toolbar
        bar = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Select / Move / Resize", _CharCanvas.MODE_SELECT)
        self.mode_combo.addItem("Draw new char", _CharCanvas.MODE_DRAW)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        bar.addWidget(QLabel("Mode:"))
        bar.addWidget(self.mode_combo)

        bar.addSpacing(16)
        propose_btn = QPushButton("✨ Auto-propose")
        propose_btn.clicked.connect(self._auto_propose)
        propose_btn.setToolTip("Run segment_characters on the region image and "
                                "use detected char bboxes as proposals.")
        bar.addWidget(propose_btn)

        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self._clear_all)
        bar.addWidget(clear_btn)

        del_btn = QPushButton("Delete selected")
        del_btn.clicked.connect(self.canvas_delete_selected)
        bar.addWidget(del_btn)

        bar.addStretch(1)
        self.count_label = QLabel("0 chars")
        bar.addWidget(self.count_label)
        outer.addLayout(bar)

        # Canvas inside scroll area
        self.canvas = _CharCanvas()
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setStyleSheet("background-color: #2b2b2b;")
        outer.addWidget(scroll, stretch=1)

        # Bottom buttons
        btm = QHBoxLayout()
        btm.addStretch(1)
        ok_btn = QPushButton("Save")
        ok_btn.clicked.connect(self._on_save)
        ok_btn.setDefault(True)
        btm.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btm.addWidget(cancel_btn)
        outer.addLayout(btm)

    def _on_mode_changed(self, _idx):
        self.canvas.set_mode(self.mode_combo.currentData())

    def canvas_delete_selected(self):
        self.canvas.delete_selected()
        self._refresh_count()

    def _clear_all(self):
        self.canvas.set_rects([])
        self._refresh_count()

    def _refresh_count(self):
        self.count_label.setText(f"{len(self.canvas.rects)} chars")

    def _auto_propose(self):
        try:
            from char_segmenter import segment_characters
        except ImportError as e:
            QMessageBox.critical(self, "Auto-propose failed",
                                  f"char_segmenter unavailable: {e}")
            return
        boxes, _, _, _, _ = segment_characters(self.region_image)
        if not boxes:
            QMessageBox.information(
                self, "Auto-propose",
                "No chars detected. Try drawing manually."
            )
            return
        rects = [QRect(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]
        # Sort left-to-right
        rects.sort(key=lambda r: r.x())
        self.canvas.set_rects(rects)
        self._refresh_count()

    def _on_save(self):
        rects = self.canvas.get_rects()
        # Sort L→R so per-char comparison order is deterministic
        rects.sort(key=lambda r: r.x())
        self._final_rects = rects
        self.accept()

    def get_chars(self):
        """Return chars in JSON-serialisable format (image coords, rectangles)."""
        rects = getattr(self, '_final_rects', self.canvas.get_rects())
        return [
            {
                'shape': 'rectangle',
                'x': int(r.x()),
                'y': int(r.y()),
                'width': int(r.width()),
                'height': int(r.height()),
            }
            for r in rects
        ]

    # paint mouse events on canvas — manual delete connection
    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.canvas_delete_selected()
        else:
            super().keyPressEvent(ev)
