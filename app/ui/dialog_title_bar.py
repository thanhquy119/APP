"""Shared frameless dialog title bar with macOS-style window dots."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, QPointF, Qt, QTimer
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QWidget


class DialogTitleBar(QFrame):
    """Reusable dialog header with drag, title, and macOS-style window dots.

    Provides minimize, maximize/restore, and close buttons matching the
    main window title bar style.  Designed to be embedded inside a
    frameless ``QDialog`` container.
    """

    def __init__(self, title: str = "", *, is_dark: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("reportTitleBar")
        self.setFixedHeight(44)
        self._is_dark = is_dark
        self._drag_start_pos: Optional[QPoint] = None
        self._drag_start_window_pos: Optional[QPoint] = None
        self._max_toggle_guard = False

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 8, 12, 8)
        root.setSpacing(10)

        # ── Title ─────────────────────────────────────────────────────────
        self.title_label = QLabel(title)
        self.title_label.setObjectName("reportTitleText")
        self.title_label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        self.title_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        root.addWidget(
            self.title_label,
            0,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
        )

        root.addStretch(1)

        # ── Window control dots ───────────────────────────────────────────
        self.controls_host = QWidget(self)
        self.controls_host.setObjectName("titleBarDotsHost")
        controls = QHBoxLayout(self.controls_host)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(7)

        self.btn_min = self._create_dot("titleBarMinDot", "Thu nhỏ")
        self.btn_max = self._create_dot("titleBarMaxDot", "Phóng to")
        self.btn_close = self._create_dot("titleBarCloseDot", "Đóng")

        self.btn_min.clicked.connect(self._minimize_window)
        self.btn_max.clicked.connect(self._toggle_max_restore)
        self.btn_close.clicked.connect(self._close_window)

        controls.addWidget(self.btn_min)
        controls.addWidget(self.btn_max)
        controls.addWidget(self.btn_close)

        root.addWidget(
            self.controls_host,
            0,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
        )

        self.sync_window_state()

    # ── Dot factory ───────────────────────────────────────────────────────

    @staticmethod
    def _create_dot(object_name: str, tooltip: str) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName(object_name)
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setText("")
        btn.setFixedSize(12, 12)
        btn.setAutoRaise(True)
        return btn

    # ── Window helpers ────────────────────────────────────────────────────

    def _window(self) -> Optional[QWidget]:
        window = self.window()
        return window if isinstance(window, QWidget) else None

    def set_title(self, title: str) -> None:
        self.title_label.setText(str(title or "").strip())

    def _is_window_maximized(self) -> bool:
        window = self._window()
        if window is None:
            return False
        if window.isMaximized() or (
            window.windowState() & Qt.WindowState.WindowMaximized
        ):
            return True
        # Frameless windows can miss the maximised bit – geometry fallback.
        handle = window.windowHandle()
        screen = handle.screen() if handle is not None else window.screen()
        if screen is None:
            return False
        available = screen.availableGeometry()
        frame = window.frameGeometry()
        tol = 8
        fills_h = (
            abs(frame.left() - available.left()) <= tol
            and abs(frame.right() - available.right()) <= tol
        )
        fills_v = (
            abs(frame.top() - available.top()) <= tol
            and abs(frame.bottom() - available.bottom()) <= tol
        )
        return fills_h and fills_v

    def sync_window_state(self) -> None:
        is_maximized = self._is_window_maximized()
        self.setProperty("maximized", is_maximized)
        self.btn_max.setProperty("windowMaximized", is_maximized)
        self.style().unpolish(self)
        self.style().polish(self)
        self.btn_max.style().unpolish(self.btn_max)
        self.btn_max.style().polish(self.btn_max)
        self.btn_max.setToolTip("Khôi phục" if is_maximized else "Phóng to")

    # ── Control hit-testing ───────────────────────────────────────────────

    def _is_over_control(self, pos: QPointF) -> bool:
        point = pos.toPoint()
        if isinstance(self.childAt(point), QToolButton):
            return True
        local = self.controls_host.mapFrom(self, point)
        return self.controls_host.rect().contains(local)

    # ── System move ───────────────────────────────────────────────────────

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

    # ── Window actions ────────────────────────────────────────────────────

    def _minimize_window(self) -> None:
        window = self._window()
        if window is not None:
            window.showMinimized()

    def _clear_max_toggle_guard(self) -> None:
        self._max_toggle_guard = False

    def _toggle_max_restore(self) -> None:
        window = self._window()
        if window is None or self._max_toggle_guard:
            return
        self._max_toggle_guard = True
        QTimer.singleShot(0, self._clear_max_toggle_guard)

        ui_maximized = bool(self.btn_max.property("windowMaximized"))
        window_maximized = self._is_window_maximized()
        is_maximized = ui_maximized or window_maximized

        if is_maximized:
            window.showNormal()
        else:
            window.showMaximized()

        QTimer.singleShot(0, self.sync_window_state)
        QTimer.singleShot(120, self.sync_window_state)

    def _close_window(self) -> None:
        window = self._window()
        if window is not None:
            window.close()

    # ── Mouse events ──────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._is_over_control(event.position())
        ):
            if self._start_system_move():
                event.accept()
                return

            window = self._window()
            if window is not None and not window.isMaximized():
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
            if window is not None and not window.isMaximized():
                delta = event.globalPosition().toPoint() - self._drag_start_pos
                window.move(self._drag_start_window_pos + delta)
                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None
        self._drag_start_window_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._is_over_control(event.position())
        ):
            controls_left = (
                self.controls_host.x()
                if hasattr(self, "controls_host")
                else self.width()
            )
            if event.position().x() < (controls_left - 8):
                self._toggle_max_restore()
                event.accept()
                return

        super().mouseDoubleClickEvent(event)
