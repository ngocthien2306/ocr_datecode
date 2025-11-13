"""
Type Dialog
Dialog để chọn type cho bounding box
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QButtonGroup, QRadioButton)
from PyQt5.QtCore import Qt


class TypeDialog(QDialog):
    """Dialog để chọn type cho bbox"""

    BBOX_TYPES = [
        ('template', 'Template', 'Template matching region'),
        ('text', 'Text', 'Text/OCR region'),
        ('datecode', 'Datecode', 'Date code region'),
        ('barcode', 'Barcode', 'Barcode/QR code region'),
        ('crop_area', 'Crop Area', 'ROI for faster matching (optional)')
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select BBox Type")
        self.setModal(True)
        self.setMinimumWidth(300)

        self.selected_type = 'template'  # Default
        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Select the type for this bounding box:")
        title.setStyleSheet("QLabel { font-weight: bold; margin-bottom: 10px; }")
        layout.addWidget(title)

        # Radio buttons
        self.button_group = QButtonGroup(self)
        for i, (type_id, type_name, description) in enumerate(self.BBOX_TYPES):
            radio = QRadioButton(type_name)
            radio.setProperty('type_id', type_id)

            # Set default selection
            if type_id == 'template':
                radio.setChecked(True)

            # Add description as tooltip
            radio.setToolTip(description)

            self.button_group.addButton(radio, i)
            layout.addWidget(radio)

        layout.addSpacing(10)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def accept(self):
        """Xử lý khi nhấn OK"""
        checked_button = self.button_group.checkedButton()
        if checked_button:
            self.selected_type = checked_button.property('type_id')
        super().accept()

    def get_selected_type(self):
        """Lấy type đã chọn"""
        return self.selected_type

    def keyPressEvent(self, event):
        """Xử lý phím tắt"""
        # Phím 1-5 để chọn nhanh
        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_5:
            index = key - Qt.Key_1
            if index < len(self.BBOX_TYPES):
                button = self.button_group.button(index)
                if button:
                    button.setChecked(True)
        # Enter để OK
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self.accept()
        # Esc để Cancel
        elif key == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)
