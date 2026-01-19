"""
Styles and Theme for the application
"""

# Modern dark theme stylesheet
APP_STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

/* Buttons */
QPushButton {
    background-color: #2d2d30;
    color: #e0e0e0;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 8px 16px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #3e3e42;
    border-color: #007acc;
}

QPushButton:pressed {
    background-color: #007acc;
}

QPushButton:checked {
    background-color: #0e639c;
    border-color: #1177bb;
}

QPushButton:disabled {
    background-color: #2d2d30;
    color: #656565;
    border-color: #3f3f46;
}

QPushButton#primaryButton {
    background-color: #0e639c;
    border-color: #1177bb;
}

QPushButton#primaryButton:hover {
    background-color: #1177bb;
}

QPushButton#dangerButton {
    background-color: #c72e2e;
    border-color: #e51400;
}

QPushButton#dangerButton:hover {
    background-color: #e51400;
}

/* List Widget */
QListWidget {
    background-color: #252526;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 4px;
    outline: none;
}

QListWidget::item {
    padding: 8px;
    border-radius: 3px;
    margin: 2px;
}

QListWidget::item:selected {
    background-color: #094771;
    color: #ffffff;
}

QListWidget::item:hover {
    background-color: #2a2d2e;
}

/* Labels */
QLabel {
    color: #e0e0e0;
    background-color: transparent;
}

QLabel#headerLabel {
    font-size: 14px;
    font-weight: bold;
    color: #ffffff;
    padding: 4px 0;
}

QLabel#infoLabel {
    color: #999999;
    font-size: 11px;
    padding: 4px;
}

QLabel#countLabel {
    color: #4ec9b0;
    font-weight: bold;
}

/* Scroll bars */
QScrollBar:vertical {
    background-color: #1e1e1e;
    width: 12px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #424242;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #4e4e4e;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #1e1e1e;
    height: 12px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #424242;
    border-radius: 6px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #4e4e4e;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Toolbar */
QWidget#toolbar {
    background-color: #2d2d30;
    border-radius: 4px;
    padding: 8px;
}

/* Image viewer */
QLabel#imageViewer {
    background-color: #1a1a1a;
    border: 1px solid #3f3f46;
    border-radius: 4px;
}

/* Panels */
QWidget#leftPanel {
    background-color: #252526;
    border-right: 1px solid #3f3f46;
}

QWidget#bboxPanel {
    background-color: #252526;
    border-left: 1px solid #3f3f46;
}

/* Dialog */
QDialog {
    background-color: #2d2d30;
}

QRadioButton {
    color: #e0e0e0;
    spacing: 8px;
    padding: 4px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 2px solid #3f3f46;
    background-color: #1e1e1e;
}

QRadioButton::indicator:checked {
    background-color: #007acc;
    border-color: #007acc;
}

QRadioButton::indicator:hover {
    border-color: #007acc;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #3f3f46;
    background-color: #1e1e1e;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #2d2d30;
    color: #e0e0e0;
    border: 1px solid #3f3f46;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 10px 20px;
    margin-right: 2px;
    min-width: 100px;
}

QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #ffffff;
    border-color: #007acc;
    border-bottom: 2px solid #007acc;
}

QTabBar::tab:hover:!selected {
    background-color: #3e3e42;
}

/* Text Edit */
QTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 8px;
    color: #e0e0e0;
}

/* Progress Bar */
QProgressBar {
    background-color: #2d2d30;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    text-align: center;
    color: #e0e0e0;
}

QProgressBar::chunk {
    background-color: #007acc;
    border-radius: 3px;
}

/* Right Panel */
QWidget#rightPanel {
    background-color: #252526;
    border-left: 1px solid #3f3f46;
}

/* ComboBox */
QComboBox {
    background-color: #2d2d30;
    color: #e0e0e0;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 6px 10px;
}

QComboBox:hover {
    border-color: #007acc;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #2d2d30;
    color: #e0e0e0;
    border: 1px solid #3f3f46;
    selection-background-color: #094771;
}
"""

