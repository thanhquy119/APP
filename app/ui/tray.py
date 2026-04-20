"""
System Tray - Tray icon and menu for FocusGuardian.

Provides:
- Tray icon with status indication
- Quick access menu
- Notifications
- Minimize to tray functionality
"""

import logging
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QApplication
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QObject
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QAction

from ..logic.focus_engine import FocusState

logger = logging.getLogger(__name__)


# State colors for tray icon
STATE_COLORS = {
    FocusState.ON_SCREEN_READING: "#47b79d",   # Mint green
    FocusState.OFFSCREEN_WRITING: "#5d8ee0",   # Calm blue
    FocusState.PHONE_DISTRACTION: "#d7746b",   # Soft red
    FocusState.DROWSY_FATIGUE: "#d39a53",      # Warm amber
    FocusState.AWAY: "#94a7b9",                # Muted slate
    FocusState.UNCERTAIN: "#7a92aa",           # Blue-gray
}

STATE_TOOLTIPS = {
    FocusState.ON_SCREEN_READING: "FocusGuardian - Đang tập trung",
    FocusState.OFFSCREEN_WRITING: "FocusGuardian - Đang ghi chép",
    FocusState.PHONE_DISTRACTION: "FocusGuardian - ⚠️ Mất tập trung!",
    FocusState.DROWSY_FATIGUE: "FocusGuardian - 😴 Có vẻ mệt mỏi",
    FocusState.AWAY: "FocusGuardian - Không có mặt",
    FocusState.UNCERTAIN: "FocusGuardian - Đang theo dõi...",
}


def create_colored_icon(color: str, size: int = 64) -> QIcon:
    """Create a simple colored circle icon."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Draw circle
    painter.setBrush(QColor(color))
    painter.setPen(Qt.PenStyle.NoPen)
    margin = 4
    painter.drawEllipse(margin, margin, size - 2 * margin, size - 2 * margin)

    # Draw inner highlight
    highlight = QColor(255, 255, 255, 60)
    painter.setBrush(highlight)
    painter.drawEllipse(margin + 4, margin + 4, size // 3, size // 3)

    painter.end()

    return QIcon(pixmap)


class SystemTray(QObject):
    """System tray icon and menu manager."""

    # Signals
    show_window_requested = pyqtSignal()
    logout_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    start_tracking_requested = pyqtSignal()
    stop_tracking_requested = pyqtSignal()
    take_break_requested = pyqtSignal()
    open_settings_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_state = FocusState.UNCERTAIN
        self.is_tracking = False
        self.theme_mode = "dark"

        self._init_tray()
        self._init_menu()

    def _init_tray(self):
        """Initialize system tray icon."""
        self.tray = QSystemTrayIcon(self.parent())

        # Set initial icon
        self._update_icon()

        # Connect signals
        self.tray.activated.connect(self._on_activated)

        # Show tray
        self.tray.show()

    def _init_menu(self):
        """Initialize context menu."""
        self.menu = QMenu()
        self._apply_menu_theme()

        # Show window
        self.action_show = QAction("Mở cửa sổ chính", self.menu)
        self.action_show.triggered.connect(self.show_window_requested.emit)
        self.menu.addAction(self.action_show)

        self.menu.addSeparator()

        # Start/Stop tracking
        self.action_toggle = QAction("Bắt đầu theo dõi", self.menu)
        self.action_toggle.triggered.connect(self._toggle_tracking)
        self.menu.addAction(self.action_toggle)

        # Take break
        self.action_break = QAction("Nghỉ giải lao", self.menu)
        self.action_break.triggered.connect(self.take_break_requested.emit)
        self.menu.addAction(self.action_break)

        self.menu.addSeparator()

        # Settings
        self.action_settings = QAction("Cài đặt", self.menu)
        self.action_settings.triggered.connect(self.open_settings_requested.emit)
        self.menu.addAction(self.action_settings)

        # Logout
        self.action_logout = QAction("Đăng xuất", self.menu)
        self.action_logout.triggered.connect(self.logout_requested.emit)
        self.menu.addAction(self.action_logout)

        self.menu.addSeparator()

        # Quit
        self.action_quit = QAction("Thoát", self.menu)
        self.action_quit.triggered.connect(self.quit_requested.emit)
        self.menu.addAction(self.action_quit)

        self.tray.setContextMenu(self.menu)

    def _apply_menu_theme(self) -> None:
        """Apply menu style according to selected dark/light theme."""
        is_dark = str(self.theme_mode or "dark").strip().lower() != "light"
        if is_dark:
            self.menu.setStyleSheet(
                """
                QMenu {
                    background-color: #152334;
                    border: 1px solid #2d4158;
                    border-radius: 8px;
                    padding: 6px;
                }
                QMenu::item {
                    color: #e7f1ff;
                    padding: 8px 22px;
                    border-radius: 6px;
                }
                QMenu::item:selected {
                    background-color: #5ac9b8;
                    color: #062922;
                }
                QMenu::separator {
                    height: 1px;
                    background: #2d4158;
                    margin: 4px 8px;
                }
                """
            )
            return

        self.menu.setStyleSheet(
            """
            QMenu {
                background-color: #f8fbff;
                border: 1px solid #c8d7e7;
                border-radius: 10px;
                padding: 6px;
            }
            QMenu::item {
                color: #223246;
                padding: 8px 22px;
                border-radius: 6px;
            }
            QMenu::item:selected {
                background-color: #53bca9;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background: #d4e0ec;
                margin: 4px 8px;
            }
            """
        )

    def set_theme_mode(self, theme_mode: str) -> None:
        """Update tray menu styling when app theme changes."""
        normalized = str(theme_mode or "dark").strip().lower()
        self.theme_mode = "light" if normalized == "light" else "dark"
        if hasattr(self, "menu") and self.menu is not None:
            self._apply_menu_theme()

    def _update_icon(self):
        """Update tray icon based on current state."""
        color = STATE_COLORS.get(self.current_state, "#607D8B")
        icon = create_colored_icon(color)
        self.tray.setIcon(icon)

        # Update tooltip
        tooltip = STATE_TOOLTIPS.get(self.current_state, "FocusGuardian")
        self.tray.setToolTip(tooltip)

    @pyqtSlot(QSystemTrayIcon.ActivationReason)
    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason):
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_window_requested.emit()
        elif reason == QSystemTrayIcon.ActivationReason.Trigger:
            # Single click - show menu on Windows
            pass

    @pyqtSlot()
    def _toggle_tracking(self):
        """Toggle tracking state."""
        if self.is_tracking:
            self.stop_tracking_requested.emit()
        else:
            self.start_tracking_requested.emit()

    def set_state(self, state: FocusState):
        """Update the displayed state."""
        self.current_state = state
        self._update_icon()

    def set_tracking(self, is_tracking: bool):
        """Update tracking state."""
        self.is_tracking = is_tracking

        if is_tracking:
            self.action_toggle.setText("Dừng theo dõi")
        else:
            self.action_toggle.setText("Bắt đầu theo dõi")

    def show_notification(
        self,
        title: str,
        message: str,
        icon: QSystemTrayIcon.MessageIcon = QSystemTrayIcon.MessageIcon.Information,
        duration_ms: int = 5000
    ):
        """Show a system notification."""
        self.tray.showMessage(title, message, icon, duration_ms)

    def show_focus_notification(self, state: FocusState):
        """Show notification for state change."""
        notifications = {
            FocusState.PHONE_DISTRACTION: (
                "📱 Mất tập trung!",
                "Có vẻ bạn đang sử dụng điện thoại. Hãy quay lại học tập!",
                QSystemTrayIcon.MessageIcon.Warning
            ),
            FocusState.DROWSY_FATIGUE: (
                "😴 Có vẻ bạn mệt mỏi",
                "Hãy nghỉ ngơi một chút để phục hồi năng lượng.",
                QSystemTrayIcon.MessageIcon.Information
            ),
            FocusState.AWAY: (
                "🚶 Bạn đã rời đi",
                "Phiên tập trung đã tạm dừng.",
                QSystemTrayIcon.MessageIcon.Information
            ),
        }

        if state in notifications:
            title, message, icon = notifications[state]
            self.show_notification(title, message, icon)

    def show_break_reminder(self, break_minutes: int = 5):
        """Show break reminder notification."""
        safe_minutes = max(1, int(break_minutes or 5))
        self.show_notification(
            "Đến giờ nghỉ giải lao",
            f"Bạn đã tập trung khá lâu. Hãy nghỉ {safe_minutes} phút để phục hồi.",
            QSystemTrayIcon.MessageIcon.Information
        )

    def hide(self):
        """Hide the tray icon."""
        self.tray.hide()

    def show(self):
        """Show the tray icon."""
        self.tray.show()

    def is_available(self) -> bool:
        """Check if system tray is available."""
        return QSystemTrayIcon.isSystemTrayAvailable()
