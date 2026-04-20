"""Authentication dialog for mandatory login/register flow."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt, pyqtProperty, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..logic.auth_manager import AuthManager
from .notice_dialog import NoticeDialog
from .theme import get_stylesheet


class AnimatedAuthTabBar(QTabBar):
    """Segmented tab bar with a smooth sliding active pill."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDrawBase(False)
        self.setExpanding(False)
        self.setUsesScrollButtons(False)
        self.setMovable(False)
        self.setElideMode(Qt.TextElideMode.ElideNone)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(46)
        self._segment_gap = 8.0
        self._frame_inset = 2.0

        self._track_bg = QColor("#152438")
        self._track_border = QColor("#34506a")
        self._text_color = QColor("#b8c9dc")
        self._active_bg = QColor("#4e8f8a")
        self._active_border = QColor("#67c8ba")
        self._active_text = QColor("#eaf7ff")

        self._indicator_x = 0.0
        self._indicator_w = 0.0

        self._anim_x = QPropertyAnimation(self, b"indicatorX", self)
        self._anim_x.setDuration(190)
        self._anim_x.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._anim_w = QPropertyAnimation(self, b"indicatorW", self)
        self._anim_w.setDuration(190)
        self._anim_w.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.currentChanged.connect(self._on_current_changed)

    @pyqtProperty(float)
    def indicatorX(self) -> float:
        return self._indicator_x

    @indicatorX.setter
    def indicatorX(self, value: float) -> None:
        self._indicator_x = float(value)
        self.update()

    @pyqtProperty(float)
    def indicatorW(self) -> float:
        return self._indicator_w

    @indicatorW.setter
    def indicatorW(self, value: float) -> None:
        self._indicator_w = float(value)
        self.update()

    def apply_theme(
        self,
        *,
        track_bg: str,
        track_border: str,
        text_color: str,
        active_bg: str,
        active_border: str,
        active_text: str,
    ) -> None:
        self._track_bg = QColor(track_bg)
        self._track_border = QColor(track_border)
        self._text_color = QColor(text_color)
        self._active_bg = QColor(active_bg)
        self._active_border = QColor(active_border)
        self._active_text = QColor(active_text)
        self._sync_indicator(animated=False)
        self.update()

    def _target_indicator_rect(self, index: int) -> QRectF:
        if index < 0 or index >= self.count():
            return QRectF()

        frame = QRectF(self.rect()).adjusted(
            self._frame_inset,
            self._frame_inset,
            -self._frame_inset,
            -self._frame_inset,
        )
        if frame.width() <= 0.0 or frame.height() <= 0.0:
            return QRectF()

        count = max(1, self.count())
        gap = self._segment_gap
        total_gap = gap * (count - 1)
        segment_width = max(1.0, (frame.width() - total_gap) / count)
        x = frame.x() + (float(index) * (segment_width + gap))
        return QRectF(x, frame.y(), segment_width, frame.height())

    def tabSizeHint(self, index: int) -> QSize:
        base = super().tabSizeHint(index)
        count = max(1, self.count())
        usable_width = max(
            240,
            self.width() - int(self._frame_inset * 2) - int((count - 1) * self._segment_gap),
        )
        seg_width = max(120, usable_width // count)
        return QSize(seg_width, max(38, base.height()))

    def _sync_indicator(self, *, animated: bool) -> None:
        target = self._target_indicator_rect(self.currentIndex())
        if target.isNull():
            return

        if (not animated) or self._indicator_w <= 0.0:
            self.indicatorX = float(target.x())
            self.indicatorW = float(target.width())
            return

        self._anim_x.stop()
        self._anim_w.stop()

        self._anim_x.setStartValue(self._indicator_x)
        self._anim_x.setEndValue(float(target.x()))
        self._anim_x.start()

        self._anim_w.setStartValue(self._indicator_w)
        self._anim_w.setEndValue(float(target.width()))
        self._anim_w.start()

    @pyqtSlot(int)
    def _on_current_changed(self, _index: int) -> None:
        self._sync_indicator(animated=True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_indicator(animated=False)

    def tabInserted(self, index: int) -> None:
        super().tabInserted(index)
        self._sync_indicator(animated=False)

    def tabRemoved(self, index: int) -> None:
        super().tabRemoved(index)
        self._sync_indicator(animated=False)

    def paintEvent(self, event):
        del event

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        frame = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(self._track_border, 1.0))
        painter.setBrush(self._track_bg)
        painter.drawRoundedRect(frame, 11.0, 11.0)

        if self.count() > 0 and self._indicator_w > 0.0:
            indicator = QRectF(
                self._indicator_x,
                self._frame_inset,
                self._indicator_w,
                max(0.0, self.height() - (self._frame_inset * 2)),
            )
            painter.setPen(QPen(self._active_border, 1.0))
            painter.setBrush(self._active_bg)
            painter.drawRoundedRect(indicator, 9.0, 9.0)

        font = painter.font()
        font.setPointSizeF(11.0)
        font.setWeight(600)
        painter.setFont(font)

        for i in range(self.count()):
            text_rect = self._target_indicator_rect(i).toRect()
            painter.setPen(self._active_text if i == self.currentIndex() else self._text_color)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.tabText(i))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            for index in range(self.count()):
                if self._target_indicator_rect(index).contains(pos):
                    if index != self.currentIndex():
                        self.setCurrentIndex(index)
                    event.accept()
                    return

        super().mousePressEvent(event)


class AuthDialog(QDialog):
    """Modal login/register dialog that blocks app access until authenticated."""

    def __init__(
        self,
        config: Optional[dict] = None,
        parent=None,
        auth_manager: Optional[AuthManager] = None,
    ):
        super().__init__(parent)
        self.config = dict(config or {})
        self._exit_requested = False
        self._theme_mode = str(self.config.get("theme_mode", "light")).strip().lower()
        if self._theme_mode not in {"dark", "light"}:
            self._theme_mode = "light"

        # Reuse the same root gradient token as MainWindow.
        self.setObjectName("appRoot")

        self._auth_manager = auth_manager or AuthManager(self.config)
        self._auth_manager.configure(self.config)

        self.setWindowTitle("FocusGuardian")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setMinimumWidth(460)
        self.setMinimumHeight(380)
        self._init_ui()
        self._apply_theme(self._theme_mode == "dark")

    @property
    def exit_requested(self) -> bool:
        return bool(self._exit_requested)

    def _confirm_exit(self) -> bool:
        return bool(
            NoticeDialog.confirm(
                self,
                "Thoát ứng dụng",
                "Bạn chưa đăng nhập. Bạn có muốn thoát ứng dụng không?",
                config=self.config,
                confirm_text="Thoát ứng dụng",
                cancel_text="Ở lại",
            )
        )

    def _apply_theme(self, is_dark: bool) -> None:
        self._theme_mode = "dark" if is_dark else "light"
        base = get_stylesheet(is_dark)

        if is_dark:
            muted = "#b5c6db"
            ok = "#8ff5dd"
            bad = "#ffb4a8"
            form_bg = "rgba(17, 29, 44, 0.62)"
            form_border = "rgba(138, 168, 203, 0.20)"
            pane_bg = "transparent"
            title_bg = "#0f1b2a"
            text_select_bg = "#3f6fb5"
            text_select_fg = "#edf4fd"
            # Match legacy calm-tech palette used across the old UI.
            switch_bg = "#152437"
            switch_border = "#2f455d"
            tab_text = "#b7c8db"
            tab_active_bg = "#2b4561"
            tab_active_border = "#4f6f92"
            tab_active_text = "#edf4fd"
        else:
            muted = "#4f6074"
            ok = "#0e8169"
            bad = "#b83232"
            form_bg = "rgba(236, 244, 252, 0.70)"
            form_border = "rgba(95, 125, 165, 0.26)"
            pane_bg = "transparent"
            title_bg = "#eef5fc"
            text_select_bg = "#b9d2ec"
            text_select_fg = "#16324a"
            switch_bg = "#edf5fd"
            switch_border = "#bfd1e4"
            tab_text = "#4b647d"
            tab_active_bg = "#d6e5f6"
            tab_active_border = "#9db9d7"
            tab_active_text = "#1f3a53"

        self.setStyleSheet(
            base
            + f"""
            QTabWidget#authTabs {{
                background: transparent;
            }}

            QTabWidget#authTabs::tab-bar {{
                height: 0px;
                width: 0px;
            }}

            QTabWidget#authTabs::pane {{
                border: none;
                border-radius: 0px;
                background: {pane_bg};
                margin-top: 8px;
                padding: 0px;
            }}

            QGroupBox {{
                border-radius: 10px;
                border: 1px solid {form_border};
                background: {form_bg};
                margin-top: 14px;
                padding: 18px 16px 16px 16px;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                top: -2px;
                padding: 0 8px;
                color: {muted};
                background: {title_bg};
                font-size: 13px;
                font-weight: 700;
            }}

            QLineEdit {{
                selection-background-color: {text_select_bg};
                selection-color: {text_select_fg};
            }}

            QLabel#authStatus {{
                border-radius: 6px;
                padding: 8px 10px;
                background: transparent;
                border: 1px solid {form_border};
                color: {muted};
                font-size: 12px;
            }}

            QLabel#authStatus[status="ok"] {{
                color: {ok};
            }}

            QLabel#authStatus[status="error"] {{
                color: {bad};
            }}
            """
        )

        if hasattr(self, "auth_tab_bar"):
            self.auth_tab_bar.apply_theme(
                track_bg=switch_bg,
                track_border=switch_border,
                text_color=tab_text,
                active_bg=tab_active_bg,
                active_border=tab_active_border,
                active_text=tab_active_text,
            )

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        switcher_row = QHBoxLayout()
        switcher_row.setContentsMargins(0, 0, 0, 0)
        switcher_row.setSpacing(0)
        switcher_row.addStretch(1)

        self.auth_tab_bar = AnimatedAuthTabBar(self)
        self.auth_tab_bar.setObjectName("authTabsBar")
        self.auth_tab_bar.setFixedWidth(380)
        self.auth_tab_bar.addTab("Đăng nhập")
        self.auth_tab_bar.addTab("Đăng ký")
        switcher_row.addWidget(self.auth_tab_bar, 0, Qt.AlignmentFlag.AlignCenter)
        switcher_row.addStretch(1)
        root.addLayout(switcher_row)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("authTabs")
        self.tabs.tabBar().setVisible(False)
        self.tabs.setDocumentMode(False)
        self.tabs.addTab(self._build_login_tab(), "Đăng nhập")
        self.tabs.addTab(self._build_register_tab(), "Đăng ký")
        root.addWidget(self.tabs, 1)

        self.form_status = QLabel("")
        self.form_status.setObjectName("authStatus")
        self.form_status.setWordWrap(True)
        self.form_status.setProperty("status", "info")
        self.form_status.hide()
        root.addWidget(self.form_status)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(10)

        self.btn_close = QPushButton("Thoát ứng dụng")
        self.btn_close.setObjectName("ghostButton")
        self.btn_close.setMinimumHeight(38)
        self.btn_close.clicked.connect(self.reject)
        action_row.addWidget(self.btn_close)

        action_row.addStretch(1)

        self.btn_submit = QPushButton("Đăng nhập")
        self.btn_submit.setObjectName("primaryButton")
        self.btn_submit.setMinimumHeight(38)
        self.btn_submit.setMinimumWidth(140)
        self.btn_submit.clicked.connect(self._submit_current_tab)
        action_row.addWidget(self.btn_submit)

        root.addLayout(action_row)

        # Keep external switcher and stacked tabs in sync.
        self.auth_tab_bar.currentChanged.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self.auth_tab_bar.setCurrentIndex)

        # Connect after submit button exists because currentIndex updates can emit currentChanged.
        self.tabs.currentChanged.connect(self._update_submit_text)
        self._update_submit_text(self.tabs.currentIndex())

    def _build_login_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 10, 0, 0)

        group = QGroupBox("Đăng nhập bằng username")
        form = QFormLayout(group)

        self.login_username = QLineEdit()
        self.login_username.setPlaceholderText("Tên đăng nhập đã đăng ký")
        form.addRow("Username:", self.login_username)

        self.login_password = QLineEdit()
        self.login_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.login_password.setPlaceholderText("Mật khẩu")
        self.login_password.returnPressed.connect(self._login)
        form.addRow("Mật khẩu:", self.login_password)

        layout.addWidget(group)
        layout.addStretch(1)
        return widget

    def _build_register_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 10, 0, 0)

        group = QGroupBox("Tạo tài khoản mới")
        form = QFormLayout(group)

        self.register_username = QLineEdit()
        self.register_username.setPlaceholderText("3-32 ký tự")
        form.addRow("Username:", self.register_username)

        self.register_password = QLineEdit()
        self.register_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.register_password.setPlaceholderText("Tối thiểu 8 ký tự")
        form.addRow("Mật khẩu:", self.register_password)

        self.register_confirm_password = QLineEdit()
        self.register_confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.register_confirm_password.setPlaceholderText("Nhập lại mật khẩu")
        self.register_confirm_password.returnPressed.connect(self._register)
        form.addRow("Xác nhận:", self.register_confirm_password)

        layout.addWidget(group)
        layout.addStretch(1)
        return widget

    def _set_form_status(self, text: str, status: str) -> None:
        self.form_status.setText(text)
        self.form_status.setProperty("status", status)
        self.form_status.style().unpolish(self.form_status)
        self.form_status.style().polish(self.form_status)
        self.form_status.show()

    def _is_auth_ready(self) -> bool:
        if not bool(self.config.get("enable_google_sheets_sync", False)):
            return False
        if not str(self.config.get("google_sheets_id", "") or "").strip():
            return False
        return True

    @pyqtSlot()
    def _submit_current_tab(self) -> None:
        if self.tabs.currentIndex() == 0:
            self._login()
        else:
            self._register()

    @pyqtSlot(int)
    def _update_submit_text(self, index: int) -> None:
        self.btn_submit.setText("Đăng nhập" if index == 0 else "Tạo tài khoản")

    @pyqtSlot()
    def _login(self) -> None:
        self._auth_manager.configure(self.config)
        result = self._auth_manager.login(
            username=self.login_username.text().strip(),
            password=self.login_password.text(),
        )

        if not result.success:
            self._set_form_status(result.message, "error")
            return

        self._set_form_status(result.message, "ok")
        self.accept()

    @pyqtSlot()
    def _register(self) -> None:
        self._auth_manager.configure(self.config)
        result = self._auth_manager.register(
            username=self.register_username.text().strip(),
            password=self.register_password.text(),
            confirm_password=self.register_confirm_password.text(),
        )

        if not result.success:
            self._set_form_status(result.message, "error")
            return

        NoticeDialog.info(
            self,
            "Đăng ký",
            "Đăng ký thành công. Bạn đã được đăng nhập.",
            config=self.config,
            button_text="Tiếp tục",
        )
        self._set_form_status("Đăng ký thành công", "ok")
        self.accept()

    def closeEvent(self, event):
        if self._auth_manager.is_authenticated():
            self.setResult(int(QDialog.DialogCode.Accepted))
            event.accept()
            return

        if self._exit_requested:
            event.accept()
            return

        # Keep the gate explicit: only close when user confirms exit.
        if self._confirm_exit():
            self._exit_requested = True
            event.accept()
        else:
            event.ignore()

    def reject(self) -> None:
        if self._auth_manager.is_authenticated():
            self.accept()
            return

        if self._confirm_exit():
            self._exit_requested = True
            super().reject()
