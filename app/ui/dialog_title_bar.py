"""Shared frameless dialog title bar with a single macOS-style close dot."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QWidget


class DialogTitleBar(QFrame):
    """Reusable dialog header with drag support and one red close button."""

    def __init__(self, title: str = "", show_title: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("topHeaderBar")
        self.setFixedHeight(40)

        self._drag_start_pos: Optional[QPoint] = None
        self._drag_start_window_pos: Optional[QPoint] = None

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 6, 12, 6)
        root.setSpacing(10)

        self.title_label = QLabel("")
        self.title_label.setObjectName("topHeaderTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.title_label.setVisible(bool(show_title))
        if show_title:
            self.set_title(title)
            root.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        root.addStretch(1)

        self.controls_host = QWidget(self)
        self.controls_host.setObjectName("titleBarDotsHost")
        controls = QHBoxLayout(self.controls_host)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(7)

        self.btn_close = QToolButton(self)
        self.btn_close.setObjectName("titleBarCloseDot")
        self.btn_close.setToolTip("Đóng")
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_close.setText("")
        self.btn_close.setFixedSize(12, 12)
        self.btn_close.setAutoRaise(True)
        self.btn_close.clicked.connect(self._close_window)
        controls.addWidget(self.btn_close)

        root.addWidget(self.controls_host, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

    def _window(self) -> Optional[QWidget]:
        window = self.window()
        return window if isinstance(window, QWidget) else None

    def set_title(self, title: str) -> None:
        self.title_label.setText(str(title or "").strip())

    def _is_over_control(self, pos: QPointF) -> bool:
        point = pos.toPoint()
        if isinstance(self.childAt(point), QToolButton):
            return True

        local = self.controls_host.mapFrom(self, point)
        return self.controls_host.rect().contains(local)

    def _start_system_move(self) -> bool:
        window = self._window()
        if window is None:
            return False

        handle = window.windowHandle()
        if handle is None or not hasattr(handle, "startSystemMove"):
            return False

        try:
            return bool(handle.startSystemMove())
        except RuntimeError:
            return False

    def _close_window(self) -> None:
        window = self._window()
        if window is not None:
            window.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self._is_over_control(event.position()):
            if self._start_system_move():
                event.accept()
                return

            window = self._window()
            if window is not None:
                self._drag_start_pos = event.globalPosition().toPoint()
                self._drag_start_window_pos = window.pos()
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and self._drag_start_pos is not None
            and self._drag_start_window_pos is not None
        ):
            window = self._window()
            if window is not None:
                delta = event.globalPosition().toPoint() - self._drag_start_pos
                window.move(self._drag_start_window_pos + delta)
                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None
        self._drag_start_window_pos = None
        super().mouseReleaseEvent(event)
