"""Standalone launcher for Focus Reset Game."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from .ui import FocusResetDialog


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    dialog = FocusResetDialog()
    dialog.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
