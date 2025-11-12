#!/usr/bin/env python3
"""
Image Annotation Tool
Entry point của ứng dụng
"""
import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Image Annotation Tool")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
