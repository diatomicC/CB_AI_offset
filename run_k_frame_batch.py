#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch K-frame batch detection UI (multiple images, grid results)."""

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from gui.k_frame_batch_ui import KFrameBatchWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Arial", 11))
    win = KFrameBatchWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
