"""Reusable themed notification dialog for calm, consistent app messaging."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _is_dark_mode(parent: Optional[QWidget], config: Optional[dict] = None) -> bool:
    """Infer dark/light mode from explicit config first, then parent state."""
    if isinstance(config, dict):
        mode = str(config.get("theme_mode", "dark")).strip().lower()
        if mode in {"dark", "light"}:
            return mode != "light"

    if parent is not None:
        parent_config = getattr(parent, "config", None)
        if isinstance(parent_config, dict):
            mode = str(parent_config.get("theme_mode", "dark")).strip().lower()
            if mode in {"dark", "light"}:
                return mode != "light"

        parent_mode = str(getattr(parent, "_theme_mode", "dark") or "dark").strip().lower()
        if parent_mode in {"dark", "light"}:
            return parent_mode != "light"

    return True


class NoticeDialog(QDialog):
    """Custom modal dialog used for info/warning/error/confirm notifications."""

    def __init__(
        self,
        title: str,
        message: str,
        *,
        kind: str = "info",
        is_dark: bool = True,
        confirm_text: str = "Đã hiểu",
        cancel_text: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._kind = kind
        self._title = str(title or "Thông báo")
        self._message = str(message or "")

        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("noticeDialog")

        self._build_ui(confirm_text=confirm_text, cancel_text=cancel_text)
        self._apply_style(is_dark=is_dark)
        self._center_in_parent()

    def _build_ui(self, *, confirm_text: str, cancel_text: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(0)

        self.card = QFrame()
        self.card.setObjectName("noticeCard")
        root.addWidget(self.card)

        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(26)
        shadow.setColor(QColor(0, 0, 0, 95))
        shadow.setOffset(0, 8)
        self.card.setGraphicsEffect(shadow)

        content = QVBoxLayout(self.card)
        content.setContentsMargins(20, 18, 20, 16)
        content.setSpacing(14)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)
        content.addLayout(header)

        icon_badge = QFrame()
        icon_badge.setObjectName("noticeIconBadge")
        icon_badge.setFixedSize(30, 30)
        icon_layout = QVBoxLayout(icon_badge)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_layout.setSpacing(0)

        icon_label = QLabel(self._icon_for_kind(self._kind))
        icon_label.setObjectName("noticeIcon")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_layout.addWidget(icon_label)
        header.addWidget(icon_badge)

        title_label = QLabel(self._title)
        title_label.setObjectName("noticeTitle")
        title_label.setWordWrap(True)
        header.addWidget(title_label, 1)

        message_label = QLabel(self._message)
        message_label.setObjectName("noticeMessage")
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        content.addWidget(message_label)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 2, 0, 0)
        button_row.setSpacing(8)
        button_row.addStretch(1)

        if cancel_text:
            cancel_btn = QPushButton(cancel_text)
            cancel_btn.setObjectName("noticeGhostButton")
            cancel_btn.setMinimumHeight(34)
            cancel_btn.clicked.connect(self.reject)
            button_row.addWidget(cancel_btn)

        confirm_btn = QPushButton(confirm_text)
        confirm_btn.setObjectName("noticePrimaryButton")
        confirm_btn.setMinimumHeight(34)
        confirm_btn.clicked.connect(self.accept)
        confirm_btn.setDefault(True)
        button_row.addWidget(confirm_btn)

        content.addLayout(button_row)
        self.setMinimumWidth(420)
        self.setMaximumWidth(560)

    @staticmethod
    def _icon_for_kind(kind: str) -> str:
        mapping = {
            "info": "i",
            "warning": "!",
            "error": "x",
            "confirm": "?",
        }
        return mapping.get(kind, "i")

    def _apply_style(self, *, is_dark: bool) -> None:
        if is_dark:
            card_bg = "#162537"
            card_border = "#3a5169"
            title_color = "#edf5ff"
            body_color = "#b5c8de"
            ghost_bg = "rgba(123, 152, 186, 0.15)"
            ghost_hover = "rgba(123, 152, 186, 0.24)"
            ghost_text = "#d6e6f7"
            ghost_border = "rgba(123, 152, 186, 0.36)"

            kind_palette = {
                "info": ("rgba(125, 162, 203, 0.14)", "rgba(125, 162, 203, 0.36)", "#d7e8fb", "#59d5c0", "#4ec8b3", "#07251f"),
                "warning": ("rgba(239, 189, 120, 0.18)", "rgba(239, 189, 120, 0.40)", "#ffe6bc", "#efbd78", "#e8b264", "#2f1f08"),
                "error": ("rgba(239, 157, 149, 0.18)", "rgba(239, 157, 149, 0.40)", "#ffd9d4", "#ef9d95", "#e58f86", "#2f1010"),
                "confirm": ("rgba(134, 169, 255, 0.18)", "rgba(134, 169, 255, 0.38)", "#dce8ff", "#86a9ff", "#759af7", "#071d42"),
            }
        else:
            card_bg = "#ffffff"
            card_border = "#c7d9ea"
            title_color = "#17314a"
            body_color = "#4a637c"
            ghost_bg = "rgba(86, 119, 156, 0.09)"
            ghost_hover = "rgba(86, 119, 156, 0.16)"
            ghost_text = "#2e4f70"
            ghost_border = "rgba(86, 119, 156, 0.30)"

            kind_palette = {
                "info": ("rgba(92, 128, 168, 0.12)", "rgba(92, 128, 168, 0.34)", "#315678", "#2f9f90", "#258f81", "#ffffff"),
                "warning": ("rgba(185, 121, 47, 0.14)", "rgba(185, 121, 47, 0.32)", "#7a5123", "#b9792f", "#a96e2a", "#ffffff"),
                "error": ("rgba(185, 82, 77, 0.14)", "rgba(185, 82, 77, 0.32)", "#7f2f2c", "#b9524d", "#aa4743", "#ffffff"),
                "confirm": ("rgba(63, 111, 181, 0.14)", "rgba(63, 111, 181, 0.30)", "#2f5681", "#3f6fb5", "#3768ad", "#ffffff"),
            }

        icon_bg, icon_border, icon_text, primary_bg, primary_hover, primary_text = kind_palette.get(
            self._kind,
            kind_palette["info"],
        )

        self.setStyleSheet(
            f"""
            QDialog#noticeDialog {{
                background: transparent;
            }}
            QFrame#noticeCard {{
                background: {card_bg};
                border: 1px solid {card_border};
                border-radius: 16px;
            }}
            QFrame#noticeIconBadge {{
                background: {icon_bg};
                border: 1px solid {icon_border};
                border-radius: 15px;
            }}
            QLabel#noticeIcon {{
                color: {icon_text};
                font-size: 15px;
                font-weight: 800;
            }}
            QLabel#noticeTitle {{
                color: {title_color};
                font-size: 16px;
                font-weight: 700;
            }}
            QLabel#noticeMessage {{
                color: {body_color};
                font-size: 13px;
                line-height: 1.35;
                border: none;
            }}
            QPushButton#noticePrimaryButton {{
                background: {primary_bg};
                color: {primary_text};
                border: none;
                border-radius: 10px;
                padding: 7px 14px;
                font-weight: 700;
            }}
            QPushButton#noticePrimaryButton:hover {{
                background: {primary_hover};
            }}
            QPushButton#noticeGhostButton {{
                background: {ghost_bg};
                color: {ghost_text};
                border: 1px solid {ghost_border};
                border-radius: 10px;
                padding: 7px 14px;
                font-weight: 600;
            }}
            QPushButton#noticeGhostButton:hover {{
                background: {ghost_hover};
            }}
            """
        )

    def _center_in_parent(self) -> None:
        parent = self.parentWidget()
        center = None

        if parent is not None:
            center = parent.frameGeometry().center()
        else:
            app = QApplication.instance()
            active_screen = None
            if app is not None and app.activeWindow() is not None:
                active_screen = app.activeWindow().screen()
            if active_screen is None:
                active_screen = QGuiApplication.primaryScreen()
            if active_screen is not None:
                center = active_screen.availableGeometry().center()

        if center is None:
            return

        self.adjustSize()
        self.move(center.x() - (self.width() // 2), center.y() - (self.height() // 2))

    @classmethod
    def info(
        cls,
        parent: Optional[QWidget],
        title: str,
        message: str,
        *,
        config: Optional[dict] = None,
        button_text: str = "Đã hiểu",
    ) -> None:
        dialog = cls(
            title,
            message,
            kind="info",
            is_dark=_is_dark_mode(parent, config),
            confirm_text=button_text,
            parent=parent,
        )
        dialog.exec()

    @classmethod
    def warning(
        cls,
        parent: Optional[QWidget],
        title: str,
        message: str,
        *,
        config: Optional[dict] = None,
        button_text: str = "Đã hiểu",
    ) -> None:
        dialog = cls(
            title,
            message,
            kind="warning",
            is_dark=_is_dark_mode(parent, config),
            confirm_text=button_text,
            parent=parent,
        )
        dialog.exec()

    @classmethod
    def error(
        cls,
        parent: Optional[QWidget],
        title: str,
        message: str,
        *,
        config: Optional[dict] = None,
        button_text: str = "Đã hiểu",
    ) -> None:
        dialog = cls(
            title,
            message,
            kind="error",
            is_dark=_is_dark_mode(parent, config),
            confirm_text=button_text,
            parent=parent,
        )
        dialog.exec()

    @classmethod
    def confirm(
        cls,
        parent: Optional[QWidget],
        title: str,
        message: str,
        *,
        config: Optional[dict] = None,
        confirm_text: str = "Xác nhận",
        cancel_text: str = "Hủy",
    ) -> bool:
        dialog = cls(
            title,
            message,
            kind="confirm",
            is_dark=_is_dark_mode(parent, config),
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            parent=parent,
        )
        return dialog.exec() == int(QDialog.DialogCode.Accepted)
