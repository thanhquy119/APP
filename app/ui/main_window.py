"""
Main Window - Primary application window for FocusGuardian.

Displays:
- Live camera preview with face mesh overlay
- Current focus state and score
- Session statistics
- Focus timeline graph
"""

import time
import logging
import math
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Dict, Any

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize, QVariantAnimation, QPointF, QPoint, QEvent, QRectF
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QConicalGradient, QRadialGradient, QLinearGradient, QBrush, QIcon, QPainterPath
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QProgressBar,
    QSizePolicy, QScrollArea,
    QGraphicsDropShadowEffect, QDialog, QGraphicsOpacityEffect, QToolButton
)

import cv2
import numpy as np

from ..logic.focus_engine import FocusState, FocusEngine, FrameFeatures
from ..logic.cloud_payloads import PROFILE_SCOPED_CONFIG_KEYS, PROFILE_SCOPED_DEFAULT_SETTINGS
from ..logic.session_analytics import SessionAnalyticsStore
from ..logic.scientific_validation import ValidationDataStore
from ..logic.zalo_alerts import ZaloAlertManager
from ..logic.auth_manager import AuthManager
from ..logic.focus_audio import FocusAudioManager
from ..logic.task_context import (
    TaskContextMonitor,
    TaskContextClassifier,
    TaskContextStats,
)
from ..utils.win_idle import get_idle_seconds
from .notice_dialog import NoticeDialog
from .theme import get_stylesheet
from .context_dialogs import (
    SessionContextDialog,
    SessionBoardingPassDialog,
    SessionHabitReportDialog,
    ContextCheckInDialog,
    SessionExitDialog,
)
from .journey_pip import FocusJourneyPiPWindow

LOCAL_PROFILE_SETTINGS_CACHE = Path(__file__).resolve().parents[2] / "analytics" / "profile_settings_cache.json"

# Type hints for vision modules
if TYPE_CHECKING:
    from ..vision import VisionPipeline, VisionResult, CameraCapture

logger = logging.getLogger(__name__)


# Color scheme for focus states
STATE_COLORS = {
    FocusState.ON_SCREEN_READING: "#59d5c0",   # Mint
    FocusState.OFFSCREEN_WRITING: "#7ea9ff",   # Soft blue
    FocusState.PHONE_DISTRACTION: "#f09d95",   # Soft red
    FocusState.DROWSY_FATIGUE: "#efbd78",      # Warm amber
    FocusState.AWAY: "#8ea1b5",                # Muted steel
    FocusState.UNCERTAIN: "#7f93aa",           # Calm blue-gray
}

STATE_NAMES = {
    FocusState.ON_SCREEN_READING: "Tín hiệu làm việc ổn định",
    FocusState.OFFSCREEN_WRITING: "Làm việc ổn định",
    FocusState.PHONE_DISTRACTION: "Lệch khỏi nhiệm vụ",
    FocusState.DROWSY_FATIGUE: "Có dấu hiệu mệt",
    FocusState.AWAY: "Ngoài khung camera",
    FocusState.UNCERTAIN: "Chưa đủ tin cậy",
}

# OpenCV text rendering does not support Vietnamese diacritics reliably.
OVERLAY_STATE_NAMES = {
    FocusState.ON_SCREEN_READING: "Tin hieu lam viec on dinh",
    FocusState.OFFSCREEN_WRITING: "Lam viec on dinh",
    FocusState.PHONE_DISTRACTION: "Lech khoi nhiem vu",
    FocusState.DROWSY_FATIGUE: "Co dau hieu met",
    FocusState.AWAY: "Ngoai khung camera",
    FocusState.UNCERTAIN: "Chua du tin cay",
}

STATE_ICONS = {
    FocusState.ON_SCREEN_READING: "•",
    FocusState.OFFSCREEN_WRITING: "•",
    FocusState.PHONE_DISTRACTION: "•",
    FocusState.DROWSY_FATIGUE: "•",
    FocusState.AWAY: "•",
    FocusState.UNCERTAIN: "•",
}


class TitleBarWidget(QFrame):
    """Custom calm-tech title bar with macOS-style window dots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topHeaderBar")
        self.setFixedHeight(40)
        self._drag_start_pos: Optional[QPoint] = None
        self._drag_start_window_pos: Optional[QPoint] = None
        self._max_toggle_guard = False

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 6, 12, 6)
        root.setSpacing(10)

        self.controls_host = QWidget()
        self.controls_host.setObjectName("titleBarDotsHost")
        controls = QHBoxLayout(self.controls_host)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(7)

        self.btn_close = self._create_control_button("titleBarCloseDot", "Đóng")
        self.btn_min = self._create_control_button("titleBarMinDot", "Thu nhỏ")
        self.btn_max = self._create_control_button("titleBarMaxDot", "Phóng to")

        self.btn_min.clicked.connect(self._minimize_window)
        self.btn_max.clicked.connect(self._toggle_max_restore)
        self.btn_close.clicked.connect(self._close_window)

        controls.addWidget(self.btn_min)
        controls.addWidget(self.btn_max)
        controls.addWidget(self.btn_close)

        root.addStretch(1)
        root.addWidget(self.controls_host, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

        self.sync_window_state()

    def _create_control_button(self, object_name: str, tooltip: str) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName(object_name)
        button.setToolTip(tooltip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setText("")
        button.setFixedSize(12, 12)
        button.setAutoRaise(True)
        return button

    def _window(self) -> Optional[QWidget]:
        window = self.window()
        return window if isinstance(window, QWidget) else None

    def set_title(self, title: str) -> None:
        _ = title

    def _is_window_maximized(self) -> bool:
        window = self._window()
        if window is None:
            return False

        if window.isMaximized() or (window.windowState() & Qt.WindowState.WindowMaximized):
            return True

        # Frameless windows on Windows can occasionally miss the maximized bit;
        # fallback to geometry check against screen available area.
        handle = window.windowHandle()
        screen = handle.screen() if handle is not None else window.screen()
        if screen is None:
            return False

        available = screen.availableGeometry()
        frame = window.frameGeometry()
        tol = 8

        fills_horizontally = (
            abs(frame.left() - available.left()) <= tol
            and abs(frame.right() - available.right()) <= tol
        )
        fills_vertically = (
            abs(frame.top() - available.top()) <= tol
            and abs(frame.bottom() - available.bottom()) <= tol
        )
        return fills_horizontally and fills_vertically

    def sync_window_state(self) -> None:
        is_maximized = self._is_window_maximized()

        self.setProperty("maximized", is_maximized)
        self.btn_max.setProperty("windowMaximized", is_maximized)
        self.style().unpolish(self)
        self.style().polish(self)
        self.btn_max.style().unpolish(self.btn_max)
        self.btn_max.style().polish(self.btn_max)

        if is_maximized:
            self.btn_max.setToolTip("Khôi phục")
        else:
            self.btn_max.setToolTip("Phóng to")

    def _is_over_control(self, pos: QPointF) -> bool:
        point = pos.toPoint()
        if isinstance(self.childAt(point), QToolButton):
            return True

        if hasattr(self, "controls_host"):
            local = self.controls_host.mapFrom(self, point)
            if self.controls_host.rect().contains(local):
                return True

        return False

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

    def _minimize_window(self) -> None:
        window = self._window()
        if window is not None:
            window.showMinimized()

    def _clear_max_toggle_guard(self) -> None:
        self._max_toggle_guard = False

    def _toggle_max_restore(self) -> None:
        window = self._window()
        if window is None:
            return

        if self._max_toggle_guard:
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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self._is_over_control(event.position()):
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
        if event.button() == Qt.MouseButton.LeftButton and not self._is_over_control(event.position()):
            controls_left = self.controls_host.x() if hasattr(self, "controls_host") else self.width()
            if event.position().x() < (controls_left - 8):
                self._toggle_max_restore()
                event.accept()
                return

        super().mouseDoubleClickEvent(event)


class CameraWidget(QFrame):
    """Large monitoring panel with polished empty state."""

    retry_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cameraFrame")
        self.setMinimumSize(340, 220)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(22)
        shadow.setColor(QColor(0, 0, 0, 70))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self.image_label)

        self.empty_state = QWidget()
        self.empty_state.setObjectName("cameraEmptyState")
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setContentsMargins(36, 24, 36, 24)
        empty_layout.setSpacing(10)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_ring = QFrame()
        icon_ring.setObjectName("cameraEmptyIconRing")
        icon_ring_layout = QVBoxLayout(icon_ring)
        icon_ring_layout.setContentsMargins(0, 0, 0, 0)
        icon_ring_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        empty_icon = QLabel()
        empty_icon.setObjectName("cameraEmptyIcon")
        empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_icon.setFixedSize(30, 30)
        empty_icon.setPixmap(self._build_camera_pixmap(22, QColor("#9bbcff")))
        icon_ring_layout.addWidget(empty_icon, 0, Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(icon_ring, 0, Qt.AlignmentFlag.AlignCenter)

        empty_title = QLabel("Camera chưa kết nối")
        empty_title.setObjectName("cameraEmptyTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_title)

        empty_subtitle = QLabel("Kiểm tra webcam hoặc nhấn Bắt đầu để thử lại")
        empty_subtitle.setObjectName("cameraEmptySubtitle")
        empty_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_subtitle.setWordWrap(True)
        empty_subtitle.setMaximumWidth(320)
        empty_layout.addWidget(empty_subtitle)

        self.retry_button = QPushButton("Thử lại camera")
        self.retry_button.setObjectName("cameraRetryButton")
        self.retry_button.setFixedHeight(34)
        self.retry_button.setIcon(QIcon(self._build_camera_pixmap(16, QColor("#cfe1ff"))))
        self.retry_button.setIconSize(QSize(14, 14))
        self.retry_button.clicked.connect(self.retry_requested.emit)
        empty_layout.addWidget(self.retry_button)

        root.addWidget(self.empty_state)

        self._last_rgb_frame: Optional[np.ndarray] = None
        self._show_placeholder()

    @staticmethod
    def _build_camera_pixmap(size: int, color: QColor) -> QPixmap:
        """Create a small centered camera glyph to avoid emoji alignment drift."""
        icon_size = max(12, int(size))
        pixmap = QPixmap(icon_size, icon_size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        stroke = max(1.2, icon_size * 0.08)
        pen = QPen(color)
        pen.setWidthF(stroke)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        body_x = icon_size * 0.14
        body_y = icon_size * 0.34
        body_w = icon_size * 0.72
        body_h = icon_size * 0.44
        corner = icon_size * 0.11
        painter.drawRoundedRect(QRectF(body_x, body_y, body_w, body_h), corner, corner)

        top_x = icon_size * 0.39
        top_y = icon_size * 0.22
        top_w = icon_size * 0.22
        top_h = icon_size * 0.12
        painter.drawRoundedRect(QRectF(top_x, top_y, top_w, top_h), corner * 0.6, corner * 0.6)

        lens_center = QPointF(icon_size * 0.50, icon_size * 0.56)
        lens_radius = icon_size * 0.14
        painter.drawEllipse(lens_center, lens_radius, lens_radius)

        painter.end()
        return pixmap

    def _show_placeholder(self):
        """Show empty state when no camera feed is available."""
        self._last_rgb_frame = None
        self.image_label.clear()
        self.image_label.hide()
        self.empty_state.show()

    def _show_frame(self, rgb: np.ndarray) -> None:
        """Render an RGB frame with smooth scaling."""
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def update_frame(self, frame: np.ndarray):
        """Update panel with a new frame or fallback to empty state."""
        if frame is None:
            self._show_placeholder()
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_rgb_frame = rgb
        self.empty_state.hide()
        self.image_label.show()
        self._show_frame(rgb)

    def resizeEvent(self, event):
        """Keep the camera feed sharp when panel size changes."""
        super().resizeEvent(event)
        if self._last_rgb_frame is not None and self.image_label.isVisible():
            self._show_frame(self._last_rgb_frame)


class LiveStatusStrip(QFrame):
    """Compact strip for camera runtime statuses."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statusStrip")
        self.is_dark = True
        self._last_stream = "Disconnected"
        self._last_face = "No face"
        self._last_lighting = "Unknown"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        self.values: dict[str, QLabel] = {}
        self._add_chip(layout, "stream", "Luồng", "Disconnected")
        self._add_chip(layout, "face", "Khuôn mặt", "No face")
        self._add_chip(layout, "lighting", "Ánh sáng", "Unknown")

    def _add_chip(self, parent_layout: QHBoxLayout, key: str, caption: str, initial_value: str) -> None:
        chip = QFrame()
        chip.setObjectName("statusChip")
        chip_layout = QVBoxLayout(chip)
        chip_layout.setContentsMargins(10, 6, 10, 6)
        chip_layout.setSpacing(1)

        cap = QLabel(caption)
        cap.setObjectName("statusLabel")
        val = QLabel(initial_value)
        val.setObjectName("statusValue")

        chip_layout.addWidget(cap)
        chip_layout.addWidget(val)

        parent_layout.addWidget(chip, 1)
        self.values[key] = val

    def set_status(self, stream: str, face: str, lighting: str) -> None:
        """Refresh runtime statuses shown in the strip."""
        self._last_stream = stream
        self._last_face = face
        self._last_lighting = lighting

        self.values["stream"].setText(stream)
        self.values["face"].setText(face)
        self.values["lighting"].setText(lighting)

        stream_lower = stream.lower()
        if stream_lower == "live":
            stream_color = "#7ef4d4" if self.is_dark else "#0f7c68"
        elif stream_lower == "paused":
            stream_color = "#ffe1a0" if self.is_dark else "#8b6125"
        else:
            stream_color = "#f7b3b3" if self.is_dark else "#9f3e39"

        self.values["stream"].setStyleSheet(f"color: {stream_color}; font-weight: 700;")

        lighting_lower = lighting.lower()
        if any(token in lighting_lower for token in ("tot", "good", "tốt")):
            lighting_color = "#7ef4d4" if self.is_dark else "#0f7c68"
        elif any(token in lighting_lower for token in ("yeu", "low", "gat", "strong", "yếu", "gắt")):
            lighting_color = "#efbd78" if self.is_dark else "#9a641e"
        else:
            lighting_color = "#d8e6f7" if self.is_dark else "#1f3a55"
        self.values["lighting"].setStyleSheet(f"color: {lighting_color}; font-weight: 700;")

    def update_theme(self, is_dark: bool) -> None:
        """Apply theme-aware text accents for the status strip."""
        self.is_dark = bool(is_dark)
        self.set_status(
            stream=self._last_stream,
            face=self._last_face,
            lighting=self._last_lighting,
        )

class FocusScoreWidget(QFrame):
    """Circular widget showing work-readiness score."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_dark = True
        self.score = 100.0
        self._target_score = 100.0
        self.state = FocusState.UNCERTAIN
        self.setMinimumSize(160, 160)
        self.setMaximumSize(210, 210)
        self.setStyleSheet("background: transparent;")

        # Setup animation
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(600)  # 600ms for smooth transition
        self._animation.valueChanged.connect(self._animate_score)

        # Add drop shadow
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(20)
        self._shadow.setColor(QColor(0, 0, 0, 85))
        self._shadow.setOffset(0, 3)
        self.setGraphicsEffect(self._shadow)

    def _animate_score(self, value):
        self.score = self._sanitize_score(value, self.score)
        self.update()

    @staticmethod
    def _sanitize_score(value, fallback: float = 0.0) -> float:
        """Convert animation value to a safe numeric score in [0, 100]."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float(fallback if fallback is not None else 0.0)
        return max(0.0, min(100.0, numeric))

    def set_score(self, score: float, state: FocusState):
        """Update the displayed score and state smoothly."""
        self.state = state
        self._target_score = self._sanitize_score(score, self._target_score)
        current_score = self._sanitize_score(self.score, self._target_score)
        self.score = current_score
        self._update_glow()

        self._animation.stop()
        self._animation.setStartValue(current_score)
        self._animation.setEndValue(self._target_score)
        self._animation.start()

    def _update_glow(self) -> None:
        """Keep the ring glow subtle and contextual."""
        if self.state in (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING) and self._target_score >= 78:
            self._shadow.setBlurRadius(22)
            self._shadow.setColor(QColor(89, 213, 192, 72) if self.is_dark else QColor(62, 169, 154, 98))
            self._shadow.setOffset(0, 2)
        elif self.state in (FocusState.PHONE_DISTRACTION, FocusState.DROWSY_FATIGUE) or self._target_score < 58:
            self._shadow.setBlurRadius(18)
            self._shadow.setColor(QColor(239, 157, 149, 58) if self.is_dark else QColor(193, 96, 90, 86))
            self._shadow.setOffset(0, 2)
        else:
            self._shadow.setBlurRadius(14)
            self._shadow.setColor(QColor(10, 20, 34, 72) if self.is_dark else QColor(98, 121, 149, 64))
            self._shadow.setOffset(0, 2)

    def paintEvent(self, event):
        """Custom paint for circular score display."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get dimensions
        rect = self.rect()
        size = min(rect.width(), rect.height()) - 20
        x = (rect.width() - size) // 2
        y = (rect.height() - size) // 2

        if not self.is_dark:
            halo = QRadialGradient(QPointF(x + (size / 2), y + (size / 2)), (size - 8) / 2)
            halo.setColorAt(0.0, QColor(255, 255, 255, 180))
            halo.setColorAt(0.72, QColor(216, 230, 245, 88))
            halo.setColorAt(1.0, QColor(194, 211, 231, 24))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(halo))
            painter.drawEllipse(x + 2, y + 2, size - 4, size - 4)

        # Draw inner base for a refined ring look.
        inner_color = QColor("#102031") if self.is_dark else QColor("#f7fbff")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(inner_color)
        painter.drawEllipse(x + 22, y + 22, size - 44, size - 44)

        # Draw track (background arc)
        track_color = "#2a3a4c" if self.is_dark else "#c6d8ea"
        track_pen = QPen(QColor(track_color))
        track_pen.setWidth(10)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(x + 6, y + 6, size - 12, size - 12, 0, 360 * 16)

        # Draw progress arc
        color = QColor(STATE_COLORS.get(self.state, "#607D8B"))
        gradient = QConicalGradient(QPointF(x + (size / 2), y + (size / 2)), -90)
        gradient.setColorAt(0.0, color.lighter(118))
        gradient.setColorAt(0.55, color)
        gradient.setColorAt(1.0, color.darker(122))

        pen = QPen(QBrush(gradient), 10)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        start_angle = 90 * 16
        safe_score = self._sanitize_score(self.score, self._target_score)
        span_angle = -int(safe_score * 3.6 * 16)
        painter.drawArc(x + 6, y + 6, size - 12, size - 12, start_angle, span_angle)

        # Draw a small endpoint marker for modern score-ring finishing.
        radius = (size - 12) / 2
        center_x = x + (size / 2)
        center_y = y + (size / 2)
        end_deg = 90.0 - (safe_score * 3.6)
        end_rad = math.radians(end_deg)
        tip_x = center_x + (radius * math.cos(end_rad))
        tip_y = center_y - (radius * math.sin(end_rad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(tip_x, tip_y), 3.5, 3.5)

        # Draw score text
        painter.setPen(self.text_color if hasattr(self, 'text_color') else QColor("#fafafa"))
        font = QFont("Segoe UI Variable Display", 38, QFont.Weight.DemiBold)
        painter.setFont(font)
        # Center in the circle
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{int(safe_score)}")

    def update_theme(self, is_dark: bool):
        self.is_dark = is_dark
        self.text_color = QColor("#fafafa") if is_dark else QColor("#173247")
        self._update_glow()
        self.update()


class BreathingCircleWidget(QWidget):
    """Animated breathing circle used in break overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self._phase = 0.0

    def set_phase(self, phase: float) -> None:
        self._phase = max(0.0, min(1.0, float(phase)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(8, 8, -8, -8)
        center = QPointF(rect.center())
        max_radius = float(min(rect.width(), rect.height()) * 0.42)
        radius = float((max_radius * 0.55) + (max_radius * 0.35 * self._phase))

        painter.setPen(QPen(QColor(130, 176, 255, 90), 2))
        painter.setBrush(QColor(48, 96, 170, 35))
        painter.drawEllipse(center, max_radius, max_radius)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(95, 184, 255, 150))
        painter.drawEllipse(center, radius, radius)


class BreakModeDialog(QDialog):
    """Calm break modal with breathing animation and short countdown."""

    def __init__(self, duration_seconds: int = 12, parent=None):
        super().__init__(parent)
        self.duration_seconds = max(8, min(90, int(duration_seconds)))
        self.remaining_seconds = self.duration_seconds
        self._is_closing = False

        self.setModal(True)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setObjectName("breakOverlay")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        dimmer = QFrame()
        dimmer.setObjectName("breakOverlayDim")
        dim_layout = QVBoxLayout(dimmer)
        dim_layout.setContentsMargins(24, 24, 24, 24)
        dim_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_opacity = QGraphicsOpacityEffect(dimmer)
        self._overlay_opacity.setOpacity(0.0)
        dimmer.setGraphicsEffect(self._overlay_opacity)

        card = QFrame()
        card.setObjectName("breakOverlayCard")
        card.setMinimumWidth(420)
        card.setMaximumWidth(520)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(26, 24, 26, 24)
        card_layout.setSpacing(10)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Phục hồi ngắn")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)

        self.breathing_circle = BreathingCircleWidget()
        card_layout.addWidget(self.breathing_circle, 0, Qt.AlignmentFlag.AlignCenter)

        self.phase_label = QLabel("Hít vào")
        self.phase_label.setObjectName("breakPhaseText")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.phase_label)

        self.countdown_label = QLabel("00:00")
        self.countdown_label.setObjectName("breakCountdownText")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.countdown_label)

        self.message_label = QLabel("Thả lỏng vai và mắt trong vài nhịp thở.")
        self.message_label.setObjectName("mutedLabel")
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.message_label)

        skip_button = QPushButton("Bỏ qua")
        skip_button.setObjectName("ghostButton")
        skip_button.setFixedHeight(34)
        skip_button.clicked.connect(self.accept)
        card_layout.addWidget(skip_button)

        dim_layout.addWidget(card)
        root.addWidget(dimmer)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(4200)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.setLoopCount(-1)
        self._animation.valueChanged.connect(self._on_breath_progress)
        self._fade_animation = QVariantAnimation(self)
        self._fade_animation.setDuration(280)
        self._fade_animation.valueChanged.connect(self._on_fade_value)
        self._fade_animation.finished.connect(self._on_fade_finished)
        self._fading_out = False
        self._update_countdown_text()

    def showEvent(self, event):
        super().showEvent(event)
        if self.parent() is not None:
            parent_geom = self.parent().geometry()
            self.setGeometry(parent_geom)

        self._fading_out = False
        self._fade_animation.stop()
        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.start()
        self._animation.start()
        self._timer.start(1000)

    def closeEvent(self, event):
        self._timer.stop()
        self._animation.stop()
        super().closeEvent(event)

    def accept(self):
        if self._is_closing:
            return
        self._is_closing = True
        self._timer.stop()
        self._animation.stop()
        self._fading_out = True
        self._fade_animation.stop()
        self._fade_animation.setStartValue(float(self._overlay_opacity.opacity()))
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.start()

    def reject(self):
        self.accept()

    def _on_breath_progress(self, value):
        try:
            progress = float(value)
        except (TypeError, ValueError):
            progress = 0.0

        if progress <= 0.5:
            phase = progress * 2.0
            self.phase_label.setText("Hít vào")
        else:
            phase = (1.0 - progress) * 2.0
            self.phase_label.setText("Thở ra")

        self.breathing_circle.set_phase(phase)

    def _tick(self):
        self.remaining_seconds = max(0, self.remaining_seconds - 1)
        self._update_countdown_text()
        if self.remaining_seconds <= 0:
            self.accept()

    def _update_countdown_text(self) -> None:
        minutes, seconds = divmod(self.remaining_seconds, 60)
        self.countdown_label.setText(f"{minutes:02d}:{seconds:02d}")

    def _on_fade_value(self, value) -> None:
        try:
            opacity = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            opacity = 1.0
        self._overlay_opacity.setOpacity(opacity)

    def _on_fade_finished(self) -> None:
        if self._fading_out:
            super().accept()
            self._is_closing = False


class TrendSparkline(QFrame):
    """Compact sparkline to visualize focus trend over time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trendSparkline")
        self.setMinimumHeight(96)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._values: list[float] = []
        self.is_dark = True

    def update_theme(self, is_dark: bool) -> None:
        self.is_dark = bool(is_dark)
        self.update()

    def set_values(self, values: list[float]):
        """Store normalized values and trigger redraw."""
        normalized: list[float] = []
        for value in values[-80:]:
            try:
                normalized.append(max(0.0, min(100.0, float(value))))
            except (TypeError, ValueError):
                continue
        self._values = normalized
        self.update()

    def paintEvent(self, event):
        """Draw a minimal trend chart that adapts to dark/light themes."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        chart_rect = self.rect().adjusted(8, 8, -8, -8)
        painter.setPen(Qt.PenStyle.NoPen)
        chart_bg = QColor("#111f2f") if self.is_dark else QColor("#edf3fb")
        painter.setBrush(chart_bg)
        painter.drawRoundedRect(chart_rect, 10, 10)

        if len(self._values) < 2:
            painter.setPen(QColor("#7b8aa0") if self.is_dark else QColor("#607488"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(chart_rect, Qt.AlignmentFlag.AlignCenter, "Đang thu thập dữ liệu xu hướng...")
            return

        low = min(self._values)
        high = max(self._values)
        span = max(1.0, high - low)

        left = chart_rect.left() + 8
        right = chart_rect.right() - 8
        top = chart_rect.top() + 8
        bottom = chart_rect.bottom() - 8
        width = max(1.0, float(right - left))
        height = max(1.0, float(bottom - top))

        mid_pen = QPen(QColor("#2c3f56") if self.is_dark else QColor("#b8cade"))
        mid_pen.setWidth(1)
        painter.setPen(mid_pen)
        mid_y = int(top + (height * 0.5))
        painter.drawLine(int(left), mid_y, int(right), mid_y)

        total = len(self._values)
        points: list[QPointF] = []
        for idx, value in enumerate(self._values):
            x = left + (idx * width / max(1, total - 1))
            ratio = (value - low) / span
            y = bottom - (ratio * height)
            points.append(QPointF(x, y))

        slope = self._values[-1] - self._values[0]
        if slope <= -6:
            line_color = QColor("#e9b16e") if self.is_dark else QColor("#b87832")
        elif slope >= 6:
            line_color = QColor("#74d8c6") if self.is_dark else QColor("#218f7b")
        else:
            line_color = QColor("#8db1ff") if self.is_dark else QColor("#4f75bf")

        line_pen = QPen(line_color)
        line_pen.setWidth(2)
        line_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        line_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(line_pen)

        for i in range(total - 1):
            painter.drawLine(points[i], points[i + 1])

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(line_color)
        painter.drawEllipse(points[-1], 3.2, 3.2)


class FocusGuidanceWidget(QFrame):
    """Calm-tech card that answers continue vs short break."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("guidanceCard")
        self.setProperty("summaryCard", True)
        self.is_dark = True

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(12, 20, 34, 40))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Trạng thái hiện tại")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.state_context = QLabel("Trạng thái: Không xác định")
        self.state_context.setObjectName("mutedLabel")
        layout.addWidget(self.state_context)

        self.decision_badge = QLabel("Sẵn sàng theo dõi")
        self.decision_badge.setObjectName("coachBadge")
        self.decision_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.decision_badge)

        self.detail_label = QLabel(
            "Bật theo dõi để hệ thống đưa gợi ý dựa trên tín hiệu hành vi hiện tại."
        )
        self.detail_label.setObjectName("mutedLabel")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

        self._detail_opacity = QGraphicsOpacityEffect(self.detail_label)
        self._detail_opacity.setOpacity(1.0)
        self.detail_label.setGraphicsEffect(self._detail_opacity)
        self._detail_fade = QVariantAnimation(self)
        self._detail_fade.setDuration(220)
        self._detail_fade.valueChanged.connect(self._fade_detail)

    def set_guidance(
        self,
        mode: str,
        decision: str,
        detail: str,
        state_text: str,
    ) -> None:
        """Refresh recommendation card state."""
        self.decision_badge.setText(decision)
        self.detail_label.setText(detail)
        self.state_context.setText(f"Trạng thái: {state_text}")

        if self.is_dark:
            badge_styles = {
                "good": ("#17372f", "#8ff5dd", "#285a4e"),
                "watch": ("#3a2d14", "#ffd38a", "#6f5328"),
                "break": ("#462218", "#ffbea7", "#7b3b2d"),
            }
        else:
            badge_styles = {
                "good": ("#e3f6f1", "#1f6e62", "#9bd9cb"),
                "watch": ("#fff2dd", "#7c5820", "#e9cf9c"),
                "break": ("#fde9e4", "#8a3f35", "#e6b4aa"),
            }

        bg, fg, border = badge_styles.get(mode, badge_styles["good"])
        self.decision_badge.setStyleSheet(
            "border-radius: 999px; padding: 8px 12px;"
            "font-weight: 700;"
            f"background-color: {bg}; color: {fg}; border: 1px solid {border};"
        )

        self._detail_fade.stop()
        self._detail_fade.setStartValue(0.62)
        self._detail_fade.setEndValue(1.0)
        self._detail_fade.start()

    def _fade_detail(self, value) -> None:
        try:
            opacity = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            opacity = 1.0
        self._detail_opacity.setOpacity(opacity)

    def update_theme(self, is_dark: bool) -> None:
        """Store current theme mode for dynamic badge styling."""
        self.is_dark = bool(is_dark)


class TrendInsightWidget(QFrame):
    """Mini trend card showing slope and work-cycle load."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trendCard")
        self.setProperty("summaryCard", True)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(12, 20, 34, 40))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        title = QLabel("Insight")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        trend_header = QHBoxLayout()
        trend_header.setContentsMargins(0, 0, 0, 0)
        trend_header.setSpacing(8)
        trend_label = QLabel("Xu hướng làm việc")
        trend_label.setObjectName("mutedLabel")
        self.trend_value = QLabel("Đang chờ dữ liệu")
        self.trend_value.setObjectName("trendValue")
        trend_header.addWidget(trend_label)
        trend_header.addStretch(1)
        trend_header.addWidget(self.trend_value)
        layout.addLayout(trend_header)

        self.sparkline = TrendSparkline()
        self.sparkline.setMinimumHeight(78)
        layout.addWidget(self.sparkline)

        cycle_header = QHBoxLayout()
        cycle_header.setContentsMargins(0, 0, 0, 0)
        cycle_header.setSpacing(8)
        cycle_label = QLabel("Tải chu kỳ")
        cycle_label.setObjectName("mutedLabel")
        self.cycle_value = QLabel("0%")
        self.cycle_value.setObjectName("trendValue")
        cycle_header.addWidget(cycle_label)
        cycle_header.addStretch(1)
        cycle_header.addWidget(self.cycle_value)
        layout.addLayout(cycle_header)

        self.cycle_progress = QProgressBar()
        self.cycle_progress.setObjectName("cycleProgress")
        self.cycle_progress.setRange(0, 100)
        self.cycle_progress.setValue(0)
        self.cycle_progress.setTextVisible(False)
        self.cycle_progress.setFixedHeight(8)
        layout.addWidget(self.cycle_progress)

        self.insight_note = QLabel("Hệ thống đang xây dựng baseline xu hướng làm việc.")
        self.insight_note.setObjectName("mutedLabel")
        self.insight_note.setWordWrap(True)
        layout.addWidget(self.insight_note)

        self._note_opacity = QGraphicsOpacityEffect(self.insight_note)
        self._note_opacity.setOpacity(1.0)
        self.insight_note.setGraphicsEffect(self._note_opacity)
        self._note_fade = QVariantAnimation(self)
        self._note_fade.setDuration(220)
        self._note_fade.valueChanged.connect(self._fade_note)

    def set_insight(
        self,
        trend_text: str,
        trend_color: str,
        cycle_percent: int,
        trend_values: list[float],
    ) -> None:
        """Refresh trend and cycle load information."""
        safe_percent = max(0, min(100, int(cycle_percent)))
        self.trend_value.setText(trend_text)
        self.trend_value.setStyleSheet(f"color: {trend_color}; font-weight: 700;")
        self.cycle_progress.setValue(safe_percent)
        self.cycle_value.setText(f"{safe_percent}%")
        self.sparkline.set_values(trend_values)

        if safe_percent >= 85 and "giảm" in trend_text.lower():
            note = "Sắp chạm ngưỡng mệt. Kết thúc phần hiện tại rồi nghỉ ngắn 3-5 phút."
        elif safe_percent < 45 and "phục hồi" in trend_text.lower():
            note = "Nhịp làm việc đang đi lên. Đây là thời điểm tốt để xử lý tác vụ quan trọng."
        else:
            note = "Theo dõi xu hướng để nghỉ đúng nhịp, giữ nhịp làm việc ổn định suốt phiên."

        self.insight_note.setText(note)
        self._note_fade.stop()
        self._note_fade.setStartValue(0.65)
        self._note_fade.setEndValue(1.0)
        self._note_fade.start()

    def _fade_note(self, value) -> None:
        try:
            opacity = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            opacity = 1.0
        self._note_opacity.setOpacity(opacity)


class StatsWidget(QFrame):
    """Compact work-session metrics designed for calm-tech UI."""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statsCard")
        self.setProperty("summaryCard", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Tổng nhịp làm việc trong hôm nay, gồm các phiên đã lưu và phiên đang chạy")

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(12, 20, 34, 42))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        title = QLabel("Nhịp làm việc hôm nay")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        subtitle = QLabel("Cộng dồn trong ngày, gồm phiên đang chạy")
        subtitle.setObjectName("metricRowLabel")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        rows = QVBoxLayout()
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(6)

        self.labels = {}
        stats = [
            ("session_time", "Thời gian hôm nay", "00:00:00", "◷"),
            ("focus_time", "Thời gian làm việc ổn định", "00:00:00", "◎"),
            ("distraction_count", "Số lần lệch nhịp", "0", "!"),
            ("break_count", "Số lần nghỉ", "0", "◌"),
            ("avg_score", "Mức sẵn sàng TB", "0", "◉"),
        ]

        for index, (key, label, default, icon_text) in enumerate(stats):
            row = QFrame()
            row.setObjectName("metricRow")
            row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 8, 10, 8)
            row_layout.setSpacing(10)

            icon = QLabel(icon_text)
            icon.setObjectName("metricRowIcon")
            icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon.setFixedWidth(16)

            caption = QLabel(label)
            caption.setObjectName("metricRowLabel")
            caption.setWordWrap(True)
            caption.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

            value_label = QLabel(default)
            value_label.setObjectName("metricRowValue")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
            value_label.setMinimumWidth(88)

            row_layout.addWidget(icon)
            row_layout.addWidget(caption, 1)
            row_layout.addWidget(value_label)

            rows.addWidget(row)
            self.labels[key] = value_label

            if index < len(stats) - 1:
                divider = QFrame()
                divider.setObjectName("metricDivider")
                divider.setFrameShape(QFrame.Shape.HLine)
                divider.setFrameShadow(QFrame.Shadow.Plain)
                rows.addWidget(divider)

        root.addLayout(rows)
        for child in self.findChildren(QWidget):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def apply_theme(self, is_dark: bool):
        """Kept for backward compatibility with legacy calls."""
        _ = is_dark

    def update_stats(self, stats: dict):
        """Update displayed statistics."""
        for key, value in stats.items():
            if key in self.labels:
                self.labels[key].setText(str(value))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


FOCUS_AIRPORT_VISUALS: Dict[str, Dict[str, Any]] = {
    "DAD": {"name": "Da Nang", "point": (0.22, 0.56)},
    "SGN": {"name": "Saigon", "point": (0.62, 0.76)},
    "HAN": {"name": "Ha Noi", "point": (0.62, 0.24)},
    "HUI": {"name": "Hue", "point": (0.36, 0.45)},
    "DLI": {"name": "Da Lat", "point": (0.54, 0.68)},
    "CXR": {"name": "Cam Ranh", "point": (0.72, 0.62)},
    "VCA": {"name": "Can Tho", "point": (0.46, 0.82)},
    "PQC": {"name": "Phu Quoc", "point": (0.30, 0.78)},
    "BMV": {"name": "Buon Ma Thuot", "point": (0.50, 0.58)},
    "VII": {"name": "Vinh", "point": (0.45, 0.33)},
    "VCL": {"name": "Chu Lai", "point": (0.30, 0.52)},
    "BKK": {"name": "Bangkok", "point": (0.22, 0.66)},
    "SIN": {"name": "Singapore", "point": (0.46, 0.92)},
    "KUL": {"name": "Kuala Lumpur", "point": (0.36, 0.88)},
    "PNH": {"name": "Phnom Penh", "point": (0.30, 0.76)},
    "VTE": {"name": "Vientiane", "point": (0.28, 0.38)},
    "REP": {"name": "Siem Reap", "point": (0.24, 0.68)},
}

FOCUS_ROUTE_VISUALS: Dict[str, Dict[str, Any]] = {
    "DAD-SGN": {"distance_km": 605, "curve": -0.18},
    "DAD-HAN": {"distance_km": 628, "curve": 0.16},
    "HAN-HUI": {"distance_km": 572, "curve": -0.22},
    "SGN-DLI": {"distance_km": 214, "curve": 0.20},
    "DAD-CXR": {"distance_km": 438, "curve": -0.14},
    "SGN-CXR": {"distance_km": 318, "curve": -0.18},
    "HAN-DAD": {"distance_km": 628, "curve": -0.16},
    "HUI-SGN": {"distance_km": 631, "curve": 0.18},
    "DAD-DLI": {"distance_km": 478, "curve": 0.22},
    "SGN-PQC": {"distance_km": 300, "curve": 0.14},
    "DAD-VII": {"distance_km": 402, "curve": -0.20},
    "SGN-VCA": {"distance_km": 133, "curve": -0.14},
    "HAN-BMV": {"distance_km": 982, "curve": 0.22},
    "VCL-DLI": {"distance_km": 408, "curve": 0.18},
    "DAD-BMV": {"distance_km": 375, "curve": 0.16},
    "HUI-DLI": {"distance_km": 522, "curve": 0.20},
}


class FocusRouteMapWidget(QFrame):
    """Symbolic 2D focus journey map drawn with Qt only."""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("journeyMapCard")
        self.setProperty("summaryCard", True)
        self.setMinimumHeight(330)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.is_dark = True
        self.route: Dict[str, Any] = {}
        self.journey_data: Dict[str, Any] = self._build_route_visual_model({})
        self.progress = 0.0
        self.remaining_seconds = 0
        self.phase = "Boarding"
        self.status = ""
        self._route_signature: Optional[tuple] = None
        self._progress_signature: Optional[tuple] = None
        self._label_text = ""
        self._canvas_base_cache: Optional[QPixmap] = None
        self._canvas_base_cache_key: Optional[tuple] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Focus Journey")
        title.setObjectName("sectionTitle")
        header.addWidget(title, 1)
        layout.addLayout(header)

        self.canvas = QWidget()
        self.canvas.setMinimumHeight(242)
        self.canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        self.canvas.paintEvent = self._paint_canvas
        self.canvas.mousePressEvent = self._handle_canvas_mouse_press
        layout.addWidget(self.canvas)

        self.info_label = QLabel("Focus journey ready")
        self.info_label.setObjectName("mutedLabel")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def _handle_canvas_mouse_press(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        event.ignore()

    def set_theme(self, is_dark: bool) -> None:
        new_value = bool(is_dark)
        if self.is_dark == new_value:
            return
        self.is_dark = new_value
        self._invalidate_canvas_cache()
        self.update()
        self.canvas.update()

    def set_journey_data(self, journey_data: Dict[str, Any]) -> None:
        route = dict(journey_data or {})
        signature = self._route_signature_for_payload(route)
        if signature == self._route_signature:
            self.route = route
            return
        self._route_signature = signature
        self.route = route
        self.journey_data = self._build_route_visual_model(self.route)
        self._update_journey_labels()
        self.canvas.update()

    def set_progress(self, progress: float, remaining_seconds: int = 0, phase: str = "", status: str = "") -> None:
        next_progress = max(0.0, min(1.0, float(progress or 0.0)))
        next_remaining = max(0, int(remaining_seconds or 0))
        next_phase = str(phase or self.phase or "Boarding")
        next_status = str(status or "")
        signature = (round(next_progress, 4), next_remaining, next_phase, next_status)
        if signature == self._progress_signature:
            return
        self._progress_signature = signature
        self.progress = next_progress
        self.remaining_seconds = next_remaining
        if phase:
            self.phase = next_phase
        self.status = next_status
        self._update_journey_labels()
        self.canvas.update()

    def _legacy_update_route(
        self,
        route_payload: Dict[str, Any],
        progress: float,
        remaining_seconds: int,
        phase: str,
        status: str = "",
    ) -> None:
        self.route = dict(route_payload or {})
        self.progress = max(0.0, min(1.0, float(progress or 0.0)))
        self.remaining_seconds = max(0, int(remaining_seconds or 0))
        self.phase = str(phase or "Boarding")
        self.status = str(status or "")
        route_code = self._route_code()
        pct = int(round(self.progress * 100))
        mins, secs = divmod(self.remaining_seconds, 60)
        remain = f"{mins}p {secs:02d}s" if self.remaining_seconds else "0p 00s"
        suffix = f"  |  {self.status}" if self.status else ""
        self.info_label.setText(f"{route_code}  |  {pct}%  |  Còn {remain}  |  {self.phase}{suffix}")
        self.canvas.update()

    def _legacy_route_code(self) -> str:
        a = str(self.route.get("route_from_code") or self.route.get("from_code") or "DAD")
        b = str(self.route.get("route_to_code") or self.route.get("to_code") or "SGN")
        return f"{a} → {b}"

    def update_route(
        self,
        route_payload: Dict[str, Any],
        progress: float,
        remaining_seconds: int,
        phase: str,
        status: str = "",
    ) -> None:
        self.phase = str(phase or "Boarding")
        self.set_journey_data(route_payload or {})
        self.set_progress(progress, remaining_seconds, self.phase, status)

    def _update_journey_labels(self) -> None:
        route_code = self._route_code()
        pct = int(round(self.progress * 100))
        metrics = self._compute_remaining_metrics()
        suffix = f" • {self.status}" if self.status else ""
        text = f"{self.phase}{suffix} • {route_code} • còn {metrics['remaining_minutes']} phút • {pct}%"
        if text != self._label_text:
            self._label_text = text
            self.info_label.setText(text)

    @staticmethod
    def _route_signature_for_payload(payload: Dict[str, Any]) -> tuple:
        data = dict(payload or {})
        def as_int(value: Any) -> int:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return 0

        return (
            str(data.get("route_from_code") or data.get("from_code") or "").strip().upper(),
            str(data.get("route_to_code") or data.get("to_code") or "").strip().upper(),
            str(data.get("route_from_name") or data.get("from_name") or ""),
            str(data.get("route_to_name") or data.get("to_name") or ""),
            as_int(data.get("route_distance_km") or data.get("distance_km")),
            as_int(data.get("route_duration_minutes") or data.get("duration_minutes") or data.get("planned_minutes")),
            str(data.get("curve") or ""),
        )

    def _route_code(self) -> str:
        a = str(self.journey_data.get("from_code") or "DAD")
        b = str(self.journey_data.get("to_code") or "SGN")
        return f"{a} → {b}"

    def _build_route_visual_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(payload or {})
        from_code = str(data.get("route_from_code") or data.get("from_code") or "").strip().upper()
        to_code = str(data.get("route_to_code") or data.get("to_code") or "").strip().upper()
        if not from_code or not to_code:
            from_code, to_code = "DAD", "SGN"

        pair_key = f"{from_code}-{to_code}"
        reverse_key = f"{to_code}-{from_code}"
        route_visual = dict(FOCUS_ROUTE_VISUALS.get(pair_key) or {})
        if not route_visual and reverse_key in FOCUS_ROUTE_VISUALS:
            reverse = FOCUS_ROUTE_VISUALS[reverse_key]
            route_visual = {
                "distance_km": reverse.get("distance_km", 405),
                "curve": -float(reverse.get("curve", 0.16) or 0.16),
            }

        start = FOCUS_AIRPORT_VISUALS.get(from_code, self._generated_airport_visual(from_code))
        end = FOCUS_AIRPORT_VISUALS.get(to_code, self._generated_airport_visual(to_code))
        distance = int(
            data.get("route_distance_km")
            or data.get("distance_km")
            or route_visual.get("distance_km")
            or self._estimate_route_distance(start["point"], end["point"])
        )
        duration = int(
            data.get("route_duration_minutes")
            or data.get("duration_minutes")
            or data.get("planned_minutes")
            or data.get("deadline_minutes")
            or (int(getattr(self, "remaining_seconds", 0) or 0) // 60 if int(getattr(self, "remaining_seconds", 0) or 0) > 0 else 25)
            or 25
        )
        curve = float(data.get("curve") or route_visual.get("curve") or self._generated_curve_bias(pair_key))
        return {
            "from_code": from_code,
            "to_code": to_code,
            "from_name": str(data.get("from_name") or data.get("route_from_name") or start.get("name") or from_code),
            "to_name": str(data.get("to_name") or data.get("route_to_name") or end.get("name") or to_code),
            "duration_minutes": max(1, duration),
            "distance_km": max(1, distance),
            "start_point": tuple(start["point"]),
            "end_point": tuple(end["point"]),
            "curve": max(-0.35, min(0.35, curve)),
        }

    @staticmethod
    def _generated_airport_visual(code: str) -> Dict[str, Any]:
        seed = sum((index + 1) * ord(ch) for index, ch in enumerate(code or "FG"))
        x = 0.18 + ((seed * 37) % 64) / 100.0
        y = 0.20 + ((seed * 53) % 58) / 100.0
        return {"name": code or "Focus", "point": (max(0.12, min(0.88, x)), max(0.16, min(0.86, y)))}

    @staticmethod
    def _estimate_route_distance(start: tuple[float, float], end: tuple[float, float]) -> int:
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        return int(max(120, min(900, round(math.sqrt(dx * dx + dy * dy) * 900))))

    @staticmethod
    def _generated_curve_bias(pair_key: str) -> float:
        seed = sum(ord(ch) for ch in pair_key or "FG")
        return (0.12 + (seed % 12) / 100.0) * (-1 if seed % 2 else 1)

    def _compute_remaining_metrics(self) -> Dict[str, int]:
        total_minutes = int(self.journey_data.get("duration_minutes", 25) or 25)
        total_distance = int(self.journey_data.get("distance_km", 405) or 405)
        remaining_minutes = int(max(0, round(total_minutes * (1.0 - self.progress))))
        return {
            "total_minutes": total_minutes,
            "total_distance_km": total_distance,
            "remaining_minutes": remaining_minutes,
            "distance_left_km": int(max(0, round(total_distance * (1.0 - self.progress)))),
        }

    def _paint_canvas(self, event) -> None:
        _ = event
        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.canvas.rect().adjusted(2, 2, -2, -2)
        if rect.width() <= 4 or rect.height() <= 4:
            return
        local_rect = QRectF(0, 0, rect.width(), rect.height())
        painter.drawPixmap(rect.topLeft(), self._canvas_base_pixmap(local_rect))
        painter.save()
        painter.translate(QPointF(rect.topLeft()))
        self._paint_2d(painter, local_rect)
        self._paint_canvas_metrics(painter, local_rect)
        painter.restore()

    def _invalidate_canvas_cache(self) -> None:
        self._canvas_base_cache = None
        self._canvas_base_cache_key = None

    def _canvas_base_pixmap(self, rect: QRectF) -> QPixmap:
        width = max(1, int(rect.width()))
        height = max(1, int(rect.height()))
        key = (width, height, bool(self.is_dark))
        if self._canvas_base_cache is not None and self._canvas_base_cache_key == key:
            return self._canvas_base_cache

        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        local_rect = QRectF(0, 0, width, height)

        bg_top = QColor("#101b2a" if self.is_dark else "#edf5fd")
        bg_bottom = QColor("#0b131d" if self.is_dark else "#f8fcff")
        grad = QLinearGradient(local_rect.topLeft(), local_rect.bottomRight())
        grad.setColorAt(0.0, bg_top)
        grad.setColorAt(1.0, bg_bottom)
        painter.setPen(QPen(QColor("#2a394b" if self.is_dark else "#c5d6e8"), 1))
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(local_rect, 12, 12)
        self._paint_map_base(painter, local_rect)
        painter.end()

        self._canvas_base_cache = pixmap
        self._canvas_base_cache_key = key
        return pixmap

    def _paint_map_base(self, painter: QPainter, rect) -> None:
        painter.save()
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(rect), 12, 12)
        painter.setClipPath(clip)

        inner = QRectF(rect).adjusted(10, 10, -10, -10)
        surface = QLinearGradient(inner.topLeft(), inner.bottomRight())
        if self.is_dark:
            surface.setColorAt(0.0, QColor("#0c1725"))
            surface.setColorAt(0.55, QColor("#102235"))
            surface.setColorAt(1.0, QColor("#0a1320"))
        else:
            surface.setColorAt(0.0, QColor("#edf8ff"))
            surface.setColorAt(0.55, QColor("#f7fcff"))
            surface.setColorAt(1.0, QColor("#e8f4fb"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(surface))
        painter.drawRoundedRect(inner, 10, 10)

        self._draw_soft_grid(painter, inner)
        self._draw_depth_contours(painter, inner)
        painter.restore()

    def _draw_soft_grid(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        grid_color = QColor(132, 165, 190, 24 if self.is_dark else 34)
        axis_color = QColor(132, 165, 190, 34 if self.is_dark else 48)
        spacing_x = max(54.0, rect.width() / 4.0)
        spacing_y = max(46.0, rect.height() / 3.5)
        painter.setPen(QPen(grid_color, 1))

        x = rect.left() + spacing_x
        while x < rect.right() - 6:
            painter.drawLine(QPointF(x, rect.top() + 22), QPointF(x, rect.bottom() - 52))
            x += spacing_x

        y = rect.top() + spacing_y
        while y < rect.bottom() - 42:
            painter.drawLine(QPointF(rect.left() + 20, y), QPointF(rect.right() - 20, y))
            y += spacing_y

        painter.setPen(QPen(axis_color, 1))
        painter.drawLine(
            QPointF(rect.left() + rect.width() * 0.18, rect.bottom() - 56),
            QPointF(rect.right() - rect.width() * 0.18, rect.top() + 34),
        )
        painter.restore()

    def _draw_depth_contours(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        colors = [
            QColor(95, 221, 210, 16 if self.is_dark else 22),
            QColor(126, 169, 255, 12 if self.is_dark else 18),
            QColor(95, 221, 210, 10 if self.is_dark else 16),
        ]
        paths = []
        path = QPainterPath(QPointF(rect.left() + rect.width() * 0.10, rect.top() + rect.height() * 0.72))
        path.cubicTo(
            QPointF(rect.left() + rect.width() * 0.22, rect.top() + rect.height() * 0.55),
            QPointF(rect.left() + rect.width() * 0.36, rect.top() + rect.height() * 0.58),
            QPointF(rect.left() + rect.width() * 0.46, rect.top() + rect.height() * 0.44),
        )
        paths.append(path)

        path = QPainterPath(QPointF(rect.left() + rect.width() * 0.55, rect.top() + rect.height() * 0.36))
        path.cubicTo(
            QPointF(rect.left() + rect.width() * 0.68, rect.top() + rect.height() * 0.25),
            QPointF(rect.left() + rect.width() * 0.78, rect.top() + rect.height() * 0.48),
            QPointF(rect.left() + rect.width() * 0.92, rect.top() + rect.height() * 0.36),
        )
        paths.append(path)

        path = QPainterPath(QPointF(rect.left() + rect.width() * 0.18, rect.top() + rect.height() * 0.28))
        path.cubicTo(
            QPointF(rect.left() + rect.width() * 0.34, rect.top() + rect.height() * 0.20),
            QPointF(rect.left() + rect.width() * 0.48, rect.top() + rect.height() * 0.30),
            QPointF(rect.left() + rect.width() * 0.66, rect.top() + rect.height() * 0.16),
        )
        paths.append(path)

        for color, path in zip(colors, paths):
            painter.setPen(QPen(color, 1.4, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(path)
        painter.restore()

    def _paint_canvas_metrics(self, painter: QPainter, rect) -> None:
        painter.save()
        route_code = self._route_code()
        pct = int(round(self.progress * 100))
        metrics = self._compute_remaining_metrics()
        mins = int(metrics["remaining_minutes"])
        distance_left = int(metrics["distance_left_km"])
        text_color = QColor("#f6fbff" if self.is_dark else "#142638")
        muted = QColor(255, 255, 255, 170) if self.is_dark else QColor(20, 38, 56, 170)
        panel_bg = QColor(8, 17, 28, 120) if self.is_dark else QColor(255, 255, 255, 150)
        planned_min = int(metrics["total_minutes"])
        progress_text = f"{pct}% / {planned_min} phút" if planned_min > 0 else f"{pct}%"

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(muted)
        painter.drawText(QPointF(rect.left() + 18, rect.top() + 25), route_code)
        painter.drawText(
            QRectF(rect.right() - 130, rect.top() + 10, 112, 22),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            progress_text,
        )

        panel_w = min(166.0, max(132.0, rect.width() * 0.36))
        left_panel = QRectF(rect.left() + 14, rect.bottom() - 62, panel_w, 48)
        right_panel = QRectF(rect.right() - panel_w - 14, rect.bottom() - 62, panel_w, 48)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(panel_bg)
        painter.drawRoundedRect(left_panel, 8, 8)
        painter.drawRoundedRect(right_panel, 8, 8)

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(muted)
        painter.drawText(left_panel.adjusted(10, 6, -10, -26), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "Thời gian còn lại")
        painter.drawText(right_panel.adjusted(10, 6, -10, -26), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "Quãng đường còn lại")
        painter.setFont(QFont("Segoe UI", 15, QFont.Weight.DemiBold))
        painter.setPen(text_color)
        painter.drawText(left_panel.adjusted(10, 22, -10, -4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{mins} min")
        painter.drawText(right_panel.adjusted(10, 22, -10, -4), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{distance_left} km")
        painter.restore()

    def _route_points(self, rect) -> tuple[QPointF, QPointF, QPointF, QPointF]:
        bounds = QRectF(rect).adjusted(48, 54, -48, -80)
        start_norm = self.journey_data.get("start_point", (0.22, 0.56))
        end_norm = self.journey_data.get("end_point", (0.62, 0.76))
        start = QPointF(
            bounds.left() + bounds.width() * float(start_norm[0]),
            bounds.top() + bounds.height() * float(start_norm[1]),
        )
        end = QPointF(
            bounds.left() + bounds.width() * float(end_norm[0]),
            bounds.top() + bounds.height() * float(end_norm[1]),
        )
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = max(1.0, math.sqrt(dx * dx + dy * dy))
        normal = QPointF(-dy / length, dx / length)
        curve = float(self.journey_data.get("curve", 0.16) or 0.16)
        lift = min(bounds.width(), bounds.height()) * curve
        c1 = QPointF(start.x() + dx * 0.34 + normal.x() * lift, start.y() + dy * 0.34 + normal.y() * lift)
        c2 = QPointF(start.x() + dx * 0.66 + normal.x() * lift, start.y() + dy * 0.66 + normal.y() * lift)
        return start, c1, c2, end

    def _bezier_point(self, p0: QPointF, c1: QPointF, c2: QPointF, p3: QPointF, t: float) -> QPointF:
        u = 1.0 - t
        return QPointF(
            (u ** 3 * p0.x()) + (3 * u * u * t * c1.x()) + (3 * u * t * t * c2.x()) + (t ** 3 * p3.x()),
            (u ** 3 * p0.y()) + (3 * u * u * t * c1.y()) + (3 * u * t * t * c2.y()) + (t ** 3 * p3.y()),
        )

    def _bezier_tangent(self, p0: QPointF, c1: QPointF, c2: QPointF, p3: QPointF, t: float) -> QPointF:
        u = 1.0 - t
        return QPointF(
            (3 * u * u * (c1.x() - p0.x())) + (6 * u * t * (c2.x() - c1.x())) + (3 * t * t * (p3.x() - c2.x())),
            (3 * u * u * (c1.y() - p0.y())) + (6 * u * t * (c2.y() - c1.y())) + (3 * t * t * (p3.y() - c2.y())),
        )

    def _paint_2d(self, painter: QPainter, rect) -> None:
        start, c1, c2, end = self._route_points(rect)
        self._draw_route(painter, start, c1, c2, end)
        self._draw_airport_marker(
            painter,
            start,
            str(self.route.get("route_from_code") or self.route.get("from_code") or "DAD"),
            rect,
            is_destination=False,
        )
        self._draw_airport_marker(
            painter,
            end,
            str(self.route.get("route_to_code") or self.route.get("to_code") or "SGN"),
            rect,
            is_destination=True,
        )
        plane_pos = self._bezier_point(start, c1, c2, end, self.progress)
        tangent = self._bezier_tangent(start, c1, c2, end, self.progress)
        self._draw_plane_marker(painter, plane_pos, tangent)

    def _draw_route(self, painter: QPainter, p0: QPointF, c1: QPointF, c2: QPointF, p3: QPointF) -> None:
        path = QPainterPath(p0)
        path.cubicTo(c1, c2, p3)
        base_color = QColor(139, 185, 207, 86 if self.is_dark else 120)
        active_color = QColor("#63e6d8" if self.is_dark else "#248f86")

        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0, 34 if self.is_dark else 18), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPath(path.translated(0, 1.5))
        painter.setPen(QPen(base_color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPath(path)

        progress_path = QPainterPath(p0)
        samples = max(2, int(48 * self.progress))
        for i in range(1, samples + 1):
            t = min(self.progress, self.progress * i / samples)
            progress_path.lineTo(self._bezier_point(p0, c1, c2, p3, t))

        if self.progress > 0.0:
            painter.setPen(QPen(QColor(active_color.red(), active_color.green(), active_color.blue(), 42), 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(progress_path)
            painter.setPen(QPen(active_color, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(progress_path)
        painter.restore()

    def _draw_airport_marker(
        self,
        painter: QPainter,
        point: QPointF,
        label: str,
        bounds,
        *,
        is_destination: bool,
    ) -> None:
        painter.save()
        fill = QColor("#13283a" if self.is_dark else "#ffffff")
        ring = QColor("#63e6d8" if is_destination else "#8aa6ba")
        glow = QColor(ring.red(), ring.green(), ring.blue(), 32 if self.is_dark else 26)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(point, 12, 12)
        painter.setPen(QPen(ring, 2))
        painter.setBrush(QBrush(fill))
        painter.drawEllipse(point, 6, 6)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(QColor("#d8eafa" if self.is_dark else "#24465c"))
        label_w = 54
        x = max(bounds.left() + 12, min(point.x() - label_w / 2, bounds.right() - label_w - 12))
        y = min(point.y() + 14, bounds.bottom() - 28)
        painter.drawText(QRectF(x, y, label_w, 18), Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()

    def _draw_plane_marker(self, painter: QPainter, point: QPointF, tangent: QPointF) -> None:
        angle = math.degrees(math.atan2(tangent.y(), tangent.x())) + 90.0
        accent = QColor("#63e6d8" if self.is_dark else "#248f86")
        body = QColor("#f7feff" if self.is_dark else "#113143")
        outline = QColor("#0b1a24" if self.is_dark else "#ffffff")

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 34))
        painter.drawEllipse(point, 18, 18)
        painter.translate(point)
        painter.rotate(angle)

        plane = QPainterPath()
        plane.moveTo(0, -12)
        plane.lineTo(4, 3)
        plane.lineTo(11, 7)
        plane.lineTo(3, 8)
        plane.lineTo(0, 14)
        plane.lineTo(-3, 8)
        plane.lineTo(-11, 7)
        plane.lineTo(-4, 3)
        plane.closeSubpath()

        painter.setPen(QPen(outline, 1.3))
        painter.setBrush(QBrush(body))
        painter.drawPath(plane)
        painter.setPen(QPen(accent, 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(0, -7, 0, 9)
        painter.restore()


class JourneyProgressWidget(QFrame):
    """Card hiển thị tiến trình hành trình làm việc trong phiên."""

    PHASES = {
        "warmup":    ("Khởi động",       "#9fd6ff"),
        "focusing":  ("Tín hiệu ổn định", "#59d5c0"),
        "declining": ("Cần nghỉ",        "#efbd78"),
        "landing":   ("Hạ cánh",         "#f09d95"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("journeyCard")
        self.setProperty("summaryCard", True)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(12, 20, 34, 40))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        self._title_lbl = QLabel("Hành trình làm việc")
        self._title_lbl.setObjectName("sectionTitle")
        title_row.addWidget(self._title_lbl, 1)
        self._mode_badge = QLabel("")
        self._mode_badge.setObjectName("coachBadge")
        self._mode_badge.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        title_row.addWidget(self._mode_badge)
        layout.addLayout(title_row)

        self._goal_lbl = QLabel("")
        self._goal_lbl.setObjectName("mutedLabel")
        self._goal_lbl.setWordWrap(True)
        layout.addWidget(self._goal_lbl)

        self._phase_badge = QLabel("Khởi động")
        self._phase_badge.setObjectName("coachBadge")
        self._phase_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._phase_badge, 0, Qt.AlignmentFlag.AlignLeft)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("cycleProgress")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(8)
        layout.addWidget(self._progress_bar)

        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.setSpacing(8)
        self._pct_lbl = QLabel("0%")
        self._pct_lbl.setObjectName("trendValue")
        self._remaining_lbl = QLabel("")
        self._remaining_lbl.setObjectName("mutedLabel")
        self._remaining_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        info_row.addWidget(self._pct_lbl)
        info_row.addStretch(1)
        info_row.addWidget(self._remaining_lbl)
        layout.addLayout(info_row)

    def update_journey(
        self,
        phase: str,
        percent: int,
        remaining_seconds: int,
        goal: str,
        session_mode: str,
        is_dark: bool = True,
    ) -> None:
        phase_label, phase_color = self.PHASES.get(phase, ("Đang theo dõi", "#9fd6ff"))
        safe_pct = max(0, min(100, int(percent)))

        self._progress_bar.setValue(safe_pct)
        self._pct_lbl.setText(f"{safe_pct}%")

        if goal:
            self._goal_lbl.setText(f"Mục tiêu: {goal}")
            self._goal_lbl.show()
        else:
            self._goal_lbl.hide()

        self._phase_badge.setText(phase_label)
        color_fg = phase_color
        color_bg = f"rgba({int(phase_color[1:3],16)},{int(phase_color[3:5],16)},{int(phase_color[5:7],16)},0.18)"
        self._phase_badge.setStyleSheet(
            f"border-radius: 999px; padding: 4px 10px; font-weight: 700;"
            f"background-color: {color_bg}; color: {color_fg}; border: 1px solid {color_fg};"
        )

        mode_labels = {"normal": "Bình thường", "deep": "Deep Focus", "deadline": "Deadline"}
        mode_text = mode_labels.get(session_mode, "")
        self._mode_badge.setText(mode_text)
        self._mode_badge.setVisible(bool(mode_text))

        if remaining_seconds > 0:
            m, s = divmod(remaining_seconds, 60)
            self._remaining_lbl.setText(f"Còn {m}p {s:02d}s")
        else:
            self._remaining_lbl.setText("")


class MainWindow(QMainWindow):
    """Main application window for FocusGuardian."""

    # Signals
    state_changed = pyqtSignal(FocusState)
    score_changed = pyqtSignal(float)
    break_suggested = pyqtSignal()
    config_changed = pyqtSignal(dict)
    logout_requested = pyqtSignal()
    context_alert_requested = pyqtSignal(str, str)

    def __init__(self, config: Optional[dict] = None, auth_manager: Optional[AuthManager] = None):
        super().__init__()
        self.config = config or {}
        self.config.setdefault("theme_mode", "dark")
        self.config.setdefault("enable_focus_audio", False)
        self.config.setdefault("focus_audio_track", "rain_light")
        self.config.setdefault("focus_audio_volume", 30)
        self.config.setdefault("vision_target_fps", 8)
        self.config.setdefault("camera_preview_fps", 12)
        self.config.setdefault("enable_performance_logging", False)
        self.config.setdefault("enable_journey_pip", True)
        self.config.setdefault("enable_validation_logging", True)
        self.config["show_overlay"] = False
        self.config["enable_personalization"] = True
        self.config["auto_apply_personalization"] = True
        self._vision_init_error = ""

        self.auth_manager = auth_manager or AuthManager(self.config)
        self.auth_manager.configure(self.config)

        # Session analytics and personalization
        self.analytics_store = SessionAnalyticsStore(cloud_config=self.config)
        self.validation_store = ValidationDataStore()
        self.zalo_alert_manager = ZaloAlertManager(self.config)
        self.focus_audio_manager = FocusAudioManager(config=self.config, parent=self)
        self.profile_name = self._get_profile_name()
        self._reset_profile_scoped_settings_to_defaults()
        self._load_profile_scoped_settings_from_supabase(seed_if_missing=True)
        self.session_started_at: Optional[float] = None
        self.state_time_by_state: Dict[str, float] = {
            state.name: 0.0 for state in FocusState
        }
        self.raw_state_time_by_state: Dict[str, float] = {
            state.name: 0.0 for state in FocusState
        }
        self.display_state: FocusState = FocusState.UNCERTAIN
        self._display_focused_state: Optional[FocusState] = None
        self._display_hold_until: float = 0.0
        self._last_recommendation: Dict[str, Any] = {}
        self.focus_trend_samples: list[float] = []

        # Digital work-context tracking.
        self.task_context_monitor = TaskContextMonitor()
        self.task_context_classifier = TaskContextClassifier()
        self.task_context_classifier.update_from_app_config(self.config)
        self.task_context_stats = TaskContextStats()
        self._session_context_payload: Dict[str, Any] = {}
        self._session_exit_payload: Dict[str, Any] = {}
        self._session_checkins: list[Dict[str, Any]] = []
        self._last_checkin_at = 0.0
        self._checkin_timestamps: list[float] = []
        self._last_behavior_summary: Dict[str, Any] = {}
        self._behavior_checkin_modifier_since: Dict[str, float] = {}
        self._focus_confirmation_until = 0.0
        self._last_vision_process_at = 0.0
        self._vision_processing_ema_ms = 0.0
        self._vision_effective_frames = 0
        self._vision_skipped_frames = 0
        self._last_perf_log_at = 0.0
        self._last_validation_prediction_at = 0.0
        self._last_boarding_preview_at = 0.0
        self._context_alert_last_signature = ""
        self._context_alert_last_at = 0.0

        # Initialize components
        self._init_vision()
        self._init_engine()
        self._init_ui()
        self._init_timers()

        # Session tracking
        self.session_time_seconds = 0
        self._session_paused = False
        self._pause_started_at = 0.0
        self._paused_total_seconds = 0.0
        self.focus_time = 0.0
        self.distraction_count = 0
        self.break_count = 0
        self.score_samples = []

        # State for break suggestions
        self.last_break_time = time.time()
        self.continuous_focus_time = 0.0
        self._last_distraction_break_time = 0.0
        self._auto_break_pending = False
        self._break_dialog_open = False

        # Journey / Deep Focus state
        self._session_mode: str = "normal"          # "normal" | "deep" | "deadline"
        self._session_goal: str = ""
        self._session_planned_minutes: int = 0
        self._session_route_payload: Dict[str, Any] = {}
        self._session_journey_enabled: bool = False
        self._journey_phase_end: str = "Boarding"
        self._journey_completion_ratio: float = 0.0
        self._journey_map_dialog: Optional[QDialog] = None
        self._journey_map_dialog_route_key = ()
        self._journey_pip_window: Optional[FocusJourneyPiPWindow] = None
        self._journey_pip_hidden_for_session: bool = False
        self._journey_pip_closed_until_restore: bool = False
        self._journey_pip_progress_key = ()
        self._journey_waiting_for_boarding: bool = False
        self._journey_calibration_reset_done: bool = True
        self._journey_session_id: int = 0
        self._validation_session_id: str = ""
        self._deep_focus_active: bool = False
        self._before_break_snapshot: dict = {}      # snapshot before each break for recovery validation
        self._break_snapshots: list = []            # list of {before, after, transfer_score}

        # Startup calibration and score smoothing to avoid early score drops.
        self._analysis_warmup_seconds = max(3.0, float(self.config.get("analysis_warmup_seconds", 12.0)))
        self._analysis_started_at = 0.0
        self._initial_baseline_samples: list[Dict[str, Any]] = []
        self._initial_session_baseline: Dict[str, Any] = {}
        self._initial_baseline_finalized = False
        self._display_score = 100.0
        self._score_drop_speed_per_sec = max(1.0, float(self.config.get("score_drop_speed_per_sec", 2.6)))
        self._score_rise_speed_per_sec = max(
            self._score_drop_speed_per_sec,
            float(self.config.get("score_rise_speed_per_sec", 10.0)),
        )
        self._display_uncertain_hold_seconds = max(
            0.8,
            float(self.config.get("display_uncertain_hold_seconds", 2.0)),
        )
        self._last_state_frame_timestamp: Optional[float] = None

        # Frameless-window edge resize support.
        self._closing = False
        self._resize_border_px = 6
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        QTimer.singleShot(0, self._refresh_today_stats_card)

    def _init_vision(self):
        """Initialize vision components using MediaPipe Tasks API."""
        try:
            from ..vision import VisionPipeline, CameraCapture, CameraConfig, ensure_models
            from ..vision.phone_detector import PhoneDetector, PhoneDetectorConfig

            # Ensure model files are downloaded
            logger.info("Checking vision models...")
            if not ensure_models():
                logger.warning("Some models failed to download, vision may not work properly")

            camera_id = int(self.config.get("camera_id", 0))
            width, height = self._parse_resolution(self.config.get("resolution", "640x480"))
            fps = int(self.config.get("fps", 15))
            camera_config = CameraConfig(
                camera_index=camera_id,
                width=width,
                height=height,
                fps=fps,
                process_width=min(width, 480),
                process_height=min(height, 360),
            )
            self.camera = CameraCapture(config=camera_config)
            self.vision_pipeline = VisionPipeline(use_live_stream=False)
            self._apply_profile_vision_calibration()

            phone_enabled = bool(self.config.get("enable_phone_detection", True))
            phone_mode = str(self.config.get("phone_detection_mode", "heuristic"))
            phone_conf_threshold = float(self.config.get("phone_confidence_threshold", 0.55))
            phone_interval_frames = max(1, int(self.config.get("phone_detection_interval_frames", 4) or 4))
            phone_confirm_window_seconds = max(
                0.8,
                float(self.config.get("phone_confirmation_window_seconds", 2.5) or 2.5),
            )
            phone_confirm_hits = max(1, int(self.config.get("phone_confirmation_min_hits", 3) or 3))

            self.phone_detector = PhoneDetector(
                PhoneDetectorConfig(
                    enabled=phone_enabled,
                    model_type=phone_mode,
                    confidence_threshold=phone_conf_threshold,
                    run_interval_frames=phone_interval_frames,
                    confirmation_window_seconds=phone_confirm_window_seconds,
                    confirmation_min_hits=phone_confirm_hits,
                )
            )

            if not self.phone_detector.initialize() and phone_enabled:
                logger.warning(
                    "Requested phone detector mode '%s' unavailable; fallback to heuristic",
                    phone_mode,
                )
                self.phone_detector = PhoneDetector(
                    PhoneDetectorConfig(
                        enabled=True,
                        model_type="heuristic",
                        confidence_threshold=phone_conf_threshold,
                        run_interval_frames=phone_interval_frames,
                        confirmation_window_seconds=phone_confirm_window_seconds,
                        confirmation_min_hits=phone_confirm_hits,
                    )
                )
                self.phone_detector.initialize()
                self.config["phone_detection_mode"] = "heuristic"
                self.config["phone_detection_mode"] = "heuristic"

            self.vision_available = True
            self._vision_init_error = ""
            logger.info("Vision modules initialized successfully")

        except Exception as e:
            logger.warning(f"Vision modules not available: {e}")
            self.camera = None
            self.vision_pipeline = None
            self.phone_detector = None
            self.vision_available = False
            self._vision_init_error = f"{type(e).__name__}: {e}"

        self.camera_running = False

    def _vision_calibration_store(self) -> Dict[str, Any]:
        payload = self.config.get("vision_calibration")
        if isinstance(payload, dict):
            return payload
        self.config["vision_calibration"] = {}
        return self.config["vision_calibration"]

    def _apply_profile_vision_calibration(self) -> None:
        if not hasattr(self, "vision_pipeline") or self.vision_pipeline is None:
            return

        profile = self._get_profile_name()
        store = self._vision_calibration_store()
        payload = store.get(profile)
        if isinstance(payload, dict):
            self.vision_pipeline.apply_calibration(payload)
        else:
            self.vision_pipeline.apply_calibration(None)

    def _persist_profile_vision_calibration(self, calibration_payload: Dict[str, Any]) -> None:
        if not calibration_payload:
            return

        profile = self._get_profile_name()
        store = self._vision_calibration_store()
        store[profile] = dict(calibration_payload)
        self.config["vision_calibration"] = store
        self.config_changed.emit(self.config.copy())

    def _init_engine(self):
        """Initialize focus engine."""
        self.engine = FocusEngine(profile_name=self.profile_name)
        self._apply_focus_engine_config()
        self.current_state = FocusState.UNCERTAIN
        self.current_score = 100.0

    def _apply_focus_engine_config(self) -> None:
        """Apply UI/config threshold values to FocusEngine runtime config."""
        self.engine.clear_personalization()
        cfg = self.engine.config
        cfg.head_down_pitch_threshold = float(self.config.get("head_down_threshold", cfg.head_down_pitch_threshold))
        cfg.head_away_yaw_threshold = float(self.config.get("look_away_threshold", cfg.head_away_yaw_threshold))
        cfg.write_score_threshold = float(self.config.get("write_score_threshold", cfg.write_score_threshold))
        cfg.drowsy_closure_ratio = float(self.config.get("eye_closure_threshold", cfg.drowsy_closure_ratio))
        cfg.drowsy_ear_threshold = float(self.config.get("ear_threshold", cfg.drowsy_ear_threshold))
        cfg.perclos_threshold = float(self.config.get("perclos_threshold", cfg.perclos_threshold))

        # Eye-gaze and head-down disambiguation controls.
        cfg.eye_look_down_threshold = float(self.config.get("eye_look_down_threshold", cfg.eye_look_down_threshold))
        cfg.eye_look_up_threshold = float(self.config.get("eye_look_up_threshold", cfg.eye_look_up_threshold))
        cfg.phone_eye_down_min_duration = float(self.config.get("phone_eye_down_min_duration", cfg.phone_eye_down_min_duration))
        cfg.phone_confidence_min = max(
            0.2,
            min(0.95, float(self.config.get("phone_confidence_min", cfg.phone_confidence_min))),
        )
        cfg.blink_rate_low_screen_max = float(self.config.get("blink_rate_low_screen_max", cfg.blink_rate_low_screen_max))
        cfg.blink_rate_high_fatigue_min = float(self.config.get("blink_rate_high_fatigue_min", cfg.blink_rate_high_fatigue_min))
        cfg.deep_head_down_pitch_threshold = float(
            self.config.get("deep_head_down_threshold", cfg.deep_head_down_pitch_threshold)
        )
        cfg.deep_head_down_min_duration = float(
            self.config.get("deep_head_down_min_duration", cfg.deep_head_down_min_duration)
        )
        cfg.deep_head_down_eye_missing_ear_threshold = float(
            self.config.get("deep_head_down_eye_missing_ear_threshold", cfg.deep_head_down_eye_missing_ear_threshold)
        )
        cfg.deep_head_down_eye_closure_ratio_min = float(
            self.config.get("deep_head_down_eye_closure_ratio_min", cfg.deep_head_down_eye_closure_ratio_min)
        )

        cfg.hysteresis_enter = max(0.15, float(self.config.get("hysteresis_enter", cfg.hysteresis_enter)))
        cfg.focused_state_hold_seconds = max(
            0.6,
            float(self.config.get("focused_state_hold_seconds", cfg.focused_state_hold_seconds)),
        )
        cfg.uncertain_short_soft_seconds = max(
            0.6,
            float(self.config.get("uncertain_short_soft_seconds", cfg.uncertain_short_soft_seconds)),
        )
        cfg.uncertain_behavior_window_seconds = max(
            cfg.uncertain_short_soft_seconds,
            float(self.config.get("uncertain_behavior_window_seconds", cfg.uncertain_behavior_window_seconds)),
        )
        cfg.score_recover_rate = max(1.0, float(self.config.get("score_recover_rate", cfg.score_recover_rate)))
        cfg.score_drop_rate = max(1.0, float(self.config.get("score_drop_rate", cfg.score_drop_rate)))
        cfg.score_noise_softening_seconds = max(
            0.4,
            float(self.config.get("score_noise_softening_seconds", cfg.score_noise_softening_seconds)),
        )
        cfg.score_confidence_floor_focused = max(
            0.2,
            min(0.95, float(self.config.get("score_confidence_floor_focused", cfg.score_confidence_floor_focused))),
        )
        cfg.score_confidence_floor_uncertain = max(
            0.05,
            min(0.9, float(self.config.get("score_confidence_floor_uncertain", cfg.score_confidence_floor_uncertain))),
        )
        cfg.score_recover_rate_focused_stable = max(
            0.5,
            float(self.config.get("score_recover_rate_focused_stable", cfg.score_recover_rate_focused_stable)),
        )
        cfg.score_recover_rate_focused_unstable = max(
            0.2,
            float(self.config.get("score_recover_rate_focused_unstable", cfg.score_recover_rate_focused_unstable)),
        )
        cfg.score_drop_rate_distraction_strong = max(
            0.5,
            float(self.config.get("score_drop_rate_distraction_strong", cfg.score_drop_rate_distraction_strong)),
        )
        cfg.score_drop_rate_distraction_soft = max(
            0.2,
            float(self.config.get("score_drop_rate_distraction_soft", cfg.score_drop_rate_distraction_soft)),
        )
        cfg.score_drop_rate_drowsy_strong = max(
            0.5,
            float(self.config.get("score_drop_rate_drowsy_strong", cfg.score_drop_rate_drowsy_strong)),
        )
        cfg.score_uncertain_soft_penalty = max(
            0.0,
            float(self.config.get("score_uncertain_soft_penalty", cfg.score_uncertain_soft_penalty)),
        )
        cfg.time_on_task_drift_start_minutes = max(
            5.0,
            float(self.config.get("time_on_task_drift_start_minutes", cfg.time_on_task_drift_start_minutes)),
        )
        cfg.time_on_task_drift_per_minute = max(
            0.0,
            float(self.config.get("time_on_task_drift_per_minute", cfg.time_on_task_drift_per_minute)),
        )
        cfg.break_recovery_boost_window_seconds = max(
            0.0,
            float(self.config.get("break_recovery_boost_window_seconds", cfg.break_recovery_boost_window_seconds)),
        )
        cfg.refocus_validation_seconds = max(
            0.5,
            float(self.config.get("refocus_validation_seconds", cfg.refocus_validation_seconds)),
        )
        cfg.refocus_confidence_min = max(
            0.2,
            min(0.95, float(self.config.get("refocus_confidence_min", cfg.refocus_confidence_min))),
        )
        cfg.refocus_face_ratio_min = max(
            0.2,
            min(0.95, float(self.config.get("refocus_face_ratio_min", cfg.refocus_face_ratio_min))),
        )
        cfg.refocus_recover_rate_locked = max(
            0.05,
            float(self.config.get("refocus_recover_rate_locked", cfg.refocus_recover_rate_locked)),
        )
        cfg.vision_confidence_uncertain_threshold = max(
            0.05,
            min(
                0.95,
                float(
                    self.config.get(
                        "vision_confidence_uncertain_threshold",
                        cfg.vision_confidence_uncertain_threshold,
                    )
                ),
            ),
        )
        cfg.vision_confidence_hard_floor = max(
            0.0,
            min(
                cfg.vision_confidence_uncertain_threshold,
                float(self.config.get("vision_confidence_hard_floor", cfg.vision_confidence_hard_floor)),
            ),
        )
        cfg.refocus_recover_ramp_seconds = max(
            0.2,
            float(self.config.get("refocus_recover_ramp_seconds", cfg.refocus_recover_ramp_seconds)),
        )
        cfg.fatigued_working_engagement_min = max(
            0.35,
            min(0.9, float(self.config.get("fatigued_working_engagement_min", cfg.fatigued_working_engagement_min))),
        )
        cfg.fatigued_working_distraction_max = max(
            0.1,
            min(0.8, float(self.config.get("fatigued_working_distraction_max", cfg.fatigued_working_distraction_max))),
        )
        cfg.passive_attention_idle_seconds = max(
            20.0,
            float(self.config.get("passive_attention_idle_seconds", cfg.passive_attention_idle_seconds)),
        )
        cfg.low_confidence_modifier_threshold = max(
            0.1,
            min(0.7, float(self.config.get("low_confidence_modifier_threshold", cfg.low_confidence_modifier_threshold))),
        )

        self.engine.capture_base_config()

    def _focus_engine_defaults(self) -> Dict[str, Any]:
        """Export current global engine defaults before user personalization is applied."""
        cfg = self.engine.config
        estimated_ear_threshold = max(0.16, min(0.32, cfg.drowsy_ear_threshold / 0.82))
        return {
            "ear_threshold": estimated_ear_threshold,
            "drowsy_ear_threshold": float(cfg.drowsy_ear_threshold),
            "drowsy_closure_ratio": float(cfg.drowsy_closure_ratio),
            "perclos_threshold": float(cfg.perclos_threshold),
            "blink_rate_low_screen_max": float(cfg.blink_rate_low_screen_max),
            "blink_rate_high_fatigue_min": float(cfg.blink_rate_high_fatigue_min),
            "fatigue_head_down_min_duration": float(cfg.fatigue_head_down_min_duration),
            "phone_eye_down_min_duration": float(cfg.phone_eye_down_min_duration),
            "phone_confidence_min": float(cfg.phone_confidence_min),
            "score_drop_rate": float(cfg.score_drop_rate),
            "score_recover_rate": float(cfg.score_recover_rate),
            "score_noise_softening_seconds": float(cfg.score_noise_softening_seconds),
            "score_confidence_floor_focused": float(cfg.score_confidence_floor_focused),
            "score_confidence_floor_uncertain": float(cfg.score_confidence_floor_uncertain),
            "score_recover_rate_focused_stable": float(cfg.score_recover_rate_focused_stable),
            "score_recover_rate_focused_unstable": float(cfg.score_recover_rate_focused_unstable),
            "score_drop_rate_distraction_strong": float(cfg.score_drop_rate_distraction_strong),
            "score_drop_rate_distraction_soft": float(cfg.score_drop_rate_distraction_soft),
            "score_drop_rate_drowsy_strong": float(cfg.score_drop_rate_drowsy_strong),
            "score_uncertain_soft_penalty": float(cfg.score_uncertain_soft_penalty),
            "time_on_task_drift_start_minutes": float(cfg.time_on_task_drift_start_minutes),
            "time_on_task_drift_per_minute": float(cfg.time_on_task_drift_per_minute),
            "break_recovery_boost_window_seconds": float(cfg.break_recovery_boost_window_seconds),
            "score_target_uncertain": float(cfg.score_target_uncertain),
            "refocus_validation_seconds": float(cfg.refocus_validation_seconds),
            "vision_confidence_uncertain_threshold": float(cfg.vision_confidence_uncertain_threshold),
            "vision_confidence_hard_floor": float(cfg.vision_confidence_hard_floor),
            "focused_state_hold_seconds": float(cfg.focused_state_hold_seconds),
            "uncertain_short_soft_seconds": float(cfg.uncertain_short_soft_seconds),
            "uncertain_behavior_window_seconds": float(cfg.uncertain_behavior_window_seconds),
        }

    def _apply_personalized_vision_thresholds(self, threshold_payload: Dict[str, Any]) -> None:
        """Apply blink EAR threshold to vision pipeline when personalization is available."""
        if not isinstance(threshold_payload, dict):
            return

        if self.vision_pipeline is None:
            return

        ear_threshold = threshold_payload.get("ear_threshold")
        if ear_threshold is None:
            return

        try:
            threshold_value = max(0.12, min(0.35, float(ear_threshold)))
        except (TypeError, ValueError):
            return

        if hasattr(self.vision_pipeline, "set_blink_threshold"):
            self.vision_pipeline.set_blink_threshold(threshold_value)
        else:
            self.vision_pipeline._blink_threshold = threshold_value

    def _init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("FocusGuardian")
        self.setMinimumSize(1060, 680)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)

        # Central widget
        central = QWidget()
        central.setObjectName("appRoot")
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(10)
        root_layout.setContentsMargins(10, 10, 10, 10)

        self.title_bar = TitleBarWidget()
        self.title_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        root_layout.addWidget(self.title_bar, 0, Qt.AlignmentFlag.AlignTop)

        content_host = QWidget()
        content_host.setObjectName("mainContentHost")

        # Main content layout
        main_layout = QHBoxLayout(content_host)
        main_layout.setSpacing(18)
        main_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(content_host, 1)

        self._root_layout = root_layout

        # Left column (65%): header, camera panel, live strip, actions
        left_column = QWidget()
        left_column.setObjectName("leftColumn")
        left_column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_column.setMinimumWidth(560)

        left_panel = QVBoxLayout(left_column)
        left_panel.setSpacing(16)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(14)
        header_row.addStretch(1)

        self.btn_logout = QPushButton("Đăng xuất")
        self.btn_logout.setObjectName("logoutButton")
        self.btn_logout.setToolTip("Đăng xuất tài khoản hiện tại")
        self.btn_logout.setFixedHeight(38)
        self.btn_logout.setMinimumWidth(104)
        self.btn_logout.clicked.connect(self._request_logout)
        header_row.addWidget(self.btn_logout, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        self.btn_settings = QPushButton()
        self.btn_settings.setObjectName("iconButton")
        self.btn_settings.setToolTip("Cài đặt")
        self.btn_settings.setFixedSize(38, 38)
        self._set_settings_button_icon()
        self.btn_settings.setIconSize(QSize(18, 18))
        self.btn_settings.clicked.connect(self._open_settings)
        header_row.addWidget(self.btn_settings, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        left_panel.addLayout(header_row)

        camera_card = QFrame()
        camera_card.setObjectName("cameraCard")
        camera_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        camera_card_layout = QVBoxLayout(camera_card)
        camera_card_layout.setContentsMargins(14, 14, 14, 14)
        camera_card_layout.setSpacing(12)

        # Camera view
        self.camera_widget = CameraWidget()
        self.camera_widget.retry_requested.connect(self._retry_camera_start)
        self.camera_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        camera_card_layout.addWidget(self.camera_widget, 1)

        self.live_status_strip = LiveStatusStrip()
        self.live_status_strip.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        camera_card_layout.addWidget(self.live_status_strip)

        left_panel.addWidget(camera_card, 1)

        # Control buttons
        controls = QHBoxLayout()
        controls.setSpacing(12)

        self.btn_start = QPushButton("Bắt đầu")
        self.btn_start.setObjectName("primaryButton")
        self.btn_start.setCheckable(True)
        self.btn_start.setMinimumHeight(46)

        self.btn_start.clicked.connect(self._toggle_tracking)
        controls.addWidget(self.btn_start)

        self.btn_break = QPushButton("Nghỉ ngay")
        self.btn_break.setObjectName("secondaryButton")
        self.btn_break.setMinimumHeight(46)

        self.btn_break.clicked.connect(self._take_break)
        controls.addWidget(self.btn_break)

        controls.addStretch(1)

        left_panel.addLayout(controls)

        main_layout.addWidget(left_column, 64)

        # Right column (35%): coherent vertical summary cards
        self.right_column_scroll = QScrollArea()
        self.right_column_scroll.setObjectName("rightColumnScroll")
        self.right_column_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.right_column_scroll.setWidgetResizable(True)
        self.right_column_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.right_column_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.right_column_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.right_column_scroll.setMinimumWidth(350)

        right_host = QWidget()
        right_host.setObjectName("rightColumnHost")
        right_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        right_panel = QVBoxLayout(right_host)
        right_panel.setSpacing(12)

        # Work readiness score card
        score_container = QFrame()
        score_container.setObjectName("scoreCard")
        score_container.setProperty("summaryCard", True)
        score_container.setToolTip(
            "Mức sẵn sàng làm việc là chỉ số 0-100 từ tín hiệu hiện tại: "
            "hành vi theo nhiệm vụ, mệt mỏi, rủi ro phân tâm và độ tin cậy camera. "
            "98 nghĩa là tín hiệu đang rất ổn định, không phải kết luận tuyệt đối."
        )
        score_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        score_shadow = QGraphicsDropShadowEffect(score_container)
        score_shadow.setBlurRadius(12)
        score_shadow.setColor(QColor(12, 20, 34, 40))
        score_shadow.setOffset(0, 2)
        score_container.setGraphicsEffect(score_shadow)
        score_layout = QVBoxLayout(score_container)
        score_layout.setContentsMargins(18, 18, 18, 18)
        score_layout.setSpacing(12)

        score_header = QHBoxLayout()
        score_header.setContentsMargins(0, 0, 0, 0)
        score_header.setSpacing(10)

        score_title = QLabel("Mức sẵn sàng làm việc")
        score_title.setObjectName("sectionTitle")
        score_title.setToolTip(score_container.toolTip())
        score_header.addWidget(score_title, 1)

        self.state_badge = QLabel("Chưa đủ tin cậy")
        self.state_badge.setObjectName("stateBadge")
        self.state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_header.addWidget(self.state_badge, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        score_layout.addLayout(score_header)

        self.score_widget = FocusScoreWidget()
        self.score_widget.setToolTip(score_container.toolTip())
        score_layout.addWidget(self.score_widget, 0, Qt.AlignmentFlag.AlignCenter)

        self.score_breakdown_panel = self._create_score_breakdown_panel()
        score_layout.addWidget(self.score_breakdown_panel)

        right_panel.addWidget(score_container)

        # Journey progress card — shown when a session is active
        self.route_map_widget = FocusRouteMapWidget()
        right_panel.addWidget(self.route_map_widget)
        self.route_map_widget.update_route({}, 0.0, 0, "Boarding", "ready")
        self.route_map_widget.clicked.connect(self._open_journey_map_dialog)
        self.route_map_widget.hide()

        self.journey_widget = JourneyProgressWidget()
        self.journey_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_panel.addWidget(self.journey_widget)
        self.journey_widget.hide()

        # Session statistics card
        self.stats_widget = StatsWidget()
        self.stats_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.stats_widget.clicked.connect(self._open_work_rhythm_report)
        right_panel.addWidget(self.stats_widget)

        # Task context card — hidden from UI; monitoring continues in background
        self.task_context_card = self._create_task_context_card()
        self.task_context_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_panel.addWidget(self.task_context_card)
        self.task_context_card.hide()

        right_panel.addStretch(1)

        self.right_column_scroll.setWidget(right_host)

        main_layout.addWidget(self.right_column_scroll, 36)
        self._main_layout = main_layout

        self._apply_theme()
        self._update_state_badge(FocusState.UNCERTAIN, 0.0, "Sẵn sàng bắt đầu phiên mới. Chỉ số này dùng để hỗ trợ nhịp làm việc, không đo trực tiếp suy nghĩ.")
        self._update_live_status(face_detected=None, lighting="Unknown")
        self._refresh_focus_guidance()
        self._sync_responsive_layout()
        self._sync_title_bar_state()


    def _create_score_breakdown_panel(self) -> QFrame:
        """Create a compact composite-score breakdown panel.

        The number shown above is not a direct mind-reading score. It is a
        behavioral readiness indicator built from engagement, fatigue, and
        distraction-risk channels.
        """
        panel = QFrame()
        panel.setObjectName("scoreBreakdownPanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header = QLabel("Thành phần chỉ số")
        header.setObjectName("metricRowLabel")
        layout.addWidget(header)

        self.score_breakdown_labels: Dict[str, QLabel] = {}
        self.score_breakdown_bars: Dict[str, QProgressBar] = {}

        rows = [
            ("engagement", "Dấu hiệu theo nhiệm vụ", "0%"),
            ("fatigue", "Mệt mỏi", "0%"),
            ("distraction", "Rủi ro phân tâm", "0%"),
        ]

        for key, label_text, default_text in rows:
            row = QFrame()
            row.setObjectName("scoreBreakdownRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            label = QLabel(label_text)
            label.setObjectName("mutedLabel")
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

            value = QLabel(default_text)
            value.setObjectName("trendValue")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setMinimumWidth(42)

            row_layout.addWidget(label)
            row_layout.addWidget(value)
            layout.addWidget(row)

            bar = QProgressBar()
            bar.setObjectName("readinessBreakdownBar")
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            layout.addWidget(bar)

            self.score_breakdown_labels[key] = value
            self.score_breakdown_bars[key] = bar

        return panel

    def _update_score_breakdown(self) -> None:
        """Update composite score channels if the engine exposes them."""
        if not hasattr(self, "score_breakdown_labels"):
            return

        breakdown: Dict[str, float] = {}
        try:
            if self.engine is not None and hasattr(self.engine, "get_score_breakdown"):
                raw = self.engine.get_score_breakdown()
                if isinstance(raw, dict):
                    breakdown = {
                        "engagement": float(raw.get("engagement", 0.0) or 0.0),
                        "fatigue": float(raw.get("fatigue", 0.0) or 0.0),
                        "distraction": float(raw.get("distraction", 0.0) or 0.0),
                    }
        except Exception as exc:
            logger.debug("Unable to update score breakdown: %s", exc)
            breakdown = {}

        if not breakdown:
            breakdown = {"engagement": 0.0, "fatigue": 0.0, "distraction": 0.0}

        try:
            digital_risk = float(getattr(getattr(self, "task_context_stats", None), "risk_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            digital_risk = 0.0
        visual_risk = max(0.0, min(1.0, float(breakdown.get("distraction", 0.0) or 0.0)))
        digital_risk = max(0.0, min(1.0, digital_risk))

        state = getattr(self, "display_state", getattr(self, "current_state", FocusState.UNCERTAIN))
        score = float(getattr(self, "current_score", 100.0) or 100.0)
        task_stats = getattr(self, "task_context_stats", None)
        current_context = str(getattr(task_stats, "current_category", "") or "").strip().lower()
        task_alignment = max(0.0, min(1.0, float(getattr(task_stats, "task_alignment_ratio", 0.0) or 0.0)))
        strong_digital_context = digital_risk >= 0.45
        explicit_visual_distraction = state == FocusState.PHONE_DISTRACTION
        task_context_active = current_context == "task_related"

        visible_engagement = max(0.0, min(1.0, float(breakdown.get("engagement", 0.0) or 0.0)))
        if task_context_active and not strong_digital_context:
            visible_engagement = max(visible_engagement, 0.70 + 0.12 * task_alignment)
        elif (
            score >= 85.0
            and not strong_digital_context
            and state in (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING)
        ):
            visible_engagement = max(visible_engagement, 0.62)
        breakdown["engagement"] = visible_engagement

        # Keep the visible risk channel interpretable: uncertain camera/posture
        # lowers task-evidence, but should not look like strong distraction
        # unless phone/context evidence is actually present.
        if not strong_digital_context and not explicit_visual_distraction:
            if task_context_active:
                visual_risk = min(visual_risk, 0.12)
            elif score >= 88.0:
                visual_risk = min(visual_risk, 0.14)
            elif state in (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING):
                visual_risk = min(visual_risk, 0.18)
            elif state in (FocusState.UNCERTAIN, FocusState.AWAY):
                visual_risk = min(visual_risk, 0.22)

        breakdown["distraction"] = max(visual_risk, digital_risk)

        for key, value in breakdown.items():
            label = self.score_breakdown_labels.get(key)
            if label is None:
                continue
            pct = int(round(max(0.0, min(1.0, float(value))) * 100.0))
            label.setText(f"{pct}%")
            bar = getattr(self, "score_breakdown_bars", {}).get(key)
            if bar is not None:
                bar.setValue(pct)

    def _create_task_context_card(self) -> QFrame:
        """Create a compact card for digital work-context awareness."""
        card = QFrame()
        card.setObjectName("taskContextCard")
        card.setProperty("summaryCard", True)

        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(12, 20, 34, 40))
        shadow.setOffset(0, 2)
        card.setGraphicsEffect(shadow)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        title = QLabel("Bối cảnh nhiệm vụ")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.task_context_app_label = QLabel("Ứng dụng: Chưa có dữ liệu")
        self.task_context_app_label.setObjectName("mutedLabel")
        self.task_context_app_label.setWordWrap(True)
        layout.addWidget(self.task_context_app_label)

        self.task_context_category_label = QLabel("Loại: Không rõ")
        self.task_context_category_label.setObjectName("mutedLabel")
        self.task_context_category_label.setWordWrap(True)
        layout.addWidget(self.task_context_category_label)

        self.task_context_alignment_label = QLabel("Task Alignment: --")
        self.task_context_alignment_label.setObjectName("trendValue")
        layout.addWidget(self.task_context_alignment_label)

        self.task_context_switch_label = QLabel("Chuyển app: 0")
        self.task_context_switch_label.setObjectName("mutedLabel")
        layout.addWidget(self.task_context_switch_label)

        self.task_context_hint_label = QLabel(
            "Chỉ dùng metadata ứng dụng/cửa sổ để hiểu bối cảnh, không chụp màn hình."
        )
        self.task_context_hint_label.setObjectName("mutedLabel")
        self.task_context_hint_label.setWordWrap(True)
        layout.addWidget(self.task_context_hint_label)

        return card

    def _apply_theme(self):
        theme_mode = str(self.config.get("theme_mode", "dark")).strip().lower()
        is_dark = theme_mode != "light"
        self.setStyleSheet(get_stylesheet(is_dark))
        if hasattr(self, "title_bar"):
            self.title_bar.set_title(self.windowTitle())
            self.title_bar.sync_window_state()
        self._set_settings_button_icon()
        if hasattr(self, "live_status_strip"):
            self.live_status_strip.update_theme(is_dark)
        if hasattr(self, "guidance_widget"):
            self.guidance_widget.update_theme(is_dark)
        if hasattr(self, "route_map_widget"):
            self.route_map_widget.set_theme(is_dark)
        if hasattr(self, "_journey_pip_window") and self._journey_pip_window is not None:
            self._journey_pip_window.update_theme(theme_mode)
        self.score_widget.update_theme(is_dark)
        self.stats_widget.apply_theme(is_dark)
        if hasattr(self, "trend_widget") and hasattr(self.trend_widget, "sparkline"):
            self.trend_widget.sparkline.update_theme(is_dark)
        if hasattr(self, "state_badge"):
            self._update_state_badge(self.current_state, 0.0, "")
        if hasattr(self, "guidance_widget"):
            self._refresh_focus_guidance()
        self._sync_title_bar_state()

    def _set_settings_button_icon(self) -> None:
        """Set a clear settings icon with a cross-platform fallback."""
        if not hasattr(self, "btn_settings"):
            return

        icon = QIcon.fromTheme("preferences-system")
        if icon.isNull():
            # Reliable fallback for environments without themed/system icons.
            self.btn_settings.setIcon(QIcon())
            self.btn_settings.setText("⚙")
            return

        self.btn_settings.setText("")
        self.btn_settings.setIcon(icon)

    def _init_timers(self):
        """Initialize update timers."""
        # Keep preview smoother than vision processing without forcing 30 FPS on weaker CPUs.
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._process_frame)
        try:
            preview_fps = float(self.config.get("camera_preview_fps", 0) or 0)
        except (TypeError, ValueError):
            preview_fps = 0.0
        if preview_fps <= 0.0:
            try:
                target_fps = float(self.config.get("vision_target_fps", 8) or 8)
            except (TypeError, ValueError):
                target_fps = 8.0
            preview_fps = max(6.0, min(16.0, target_fps * 1.5))
        self.frame_interval = int(round(1000.0 / max(6.0, min(18.0, preview_fps))))

        # Stats update timer (1 second)
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_stats)

        # Task-context timer. This samples only foreground app metadata, not screen content.
        self.task_context_timer = QTimer()
        self.task_context_timer.timeout.connect(self._sample_task_context)
        try:
            interval_ms = int(float(self.config.get("task_context_sample_interval_seconds", 5.0) or 5.0) * 1000)
        except (TypeError, ValueError):
            interval_ms = 5000
        self.task_context_interval_ms = max(2000, min(30000, interval_ms))

    def _get_profile_name(self) -> str:
        """Return active profile name from config."""
        fallback = str(self.config.get("profile_name", "default")).strip() or "default"
        return self.auth_manager.get_effective_profile_name(fallback)

    def _profile_scoped_settings_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key in PROFILE_SCOPED_CONFIG_KEYS:
            if key in self.config:
                payload[key] = self.config.get(key)
        return payload

    @staticmethod
    def _normalize_profile_cache_name(profile_name: str) -> str:
        return (profile_name or "default").strip().lower() or "default"

    def _load_local_profile_settings_cache(self) -> Dict[str, Any]:
        try:
            if not LOCAL_PROFILE_SETTINGS_CACHE.exists():
                return {}
            with open(LOCAL_PROFILE_SETTINGS_CACHE, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.debug("Failed to load local profile settings cache: %s", exc)
            return {}

    def _save_local_profile_settings_cache(self, data: Dict[str, Any]) -> None:
        try:
            LOCAL_PROFILE_SETTINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(LOCAL_PROFILE_SETTINGS_CACHE, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.debug("Failed to save local profile settings cache: %s", exc)

    def _load_profile_scoped_settings_from_local_cache(self, profile_name: str) -> bool:
        cache = self._load_local_profile_settings_cache()
        key = self._normalize_profile_cache_name(profile_name)
        settings = cache.get(key)
        if not isinstance(settings, dict):
            return False

        restored: Dict[str, Any] = {}
        for setting_key in PROFILE_SCOPED_CONFIG_KEYS:
            if setting_key in settings:
                restored[setting_key] = settings.get(setting_key)
        if not restored:
            return False

        self.config.update(restored)
        logger.info(
            "Loaded %s profile-scoped settings from local cache for profile '%s'",
            len(restored),
            profile_name,
        )
        return True

    def _save_profile_scoped_settings_to_local_cache(self, profile_name: str) -> None:
        normalized_profile = str(profile_name or self._get_profile_name()).strip() or "default"
        cache = self._load_local_profile_settings_cache()
        cache[self._normalize_profile_cache_name(normalized_profile)] = self._profile_scoped_settings_payload()
        self._save_local_profile_settings_cache(cache)

    def _reset_profile_scoped_settings_to_defaults(self) -> None:
        if not bool(self.config.get("enable_supabase_sync", False)):
            return

        for key in PROFILE_SCOPED_CONFIG_KEYS:
            if key in PROFILE_SCOPED_DEFAULT_SETTINGS:
                self.config[key] = PROFILE_SCOPED_DEFAULT_SETTINGS[key]

    def _load_profile_scoped_settings_from_supabase(self, *, seed_if_missing: bool = False) -> None:
        profile_name = str(self.profile_name or self._get_profile_name()).strip() or "default"
        if not bool(self.config.get("enable_supabase_sync", False)):
            self._load_profile_scoped_settings_from_local_cache(profile_name)
            return

        loaded = self.analytics_store.supabase_sync.load_profile_settings(profile_name)
        if loaded is None:
            self._load_profile_scoped_settings_from_local_cache(profile_name)
            return

        if loaded:
            self.config.update(loaded)
            self._save_profile_scoped_settings_to_local_cache(profile_name)
            logger.info(
                "Loaded %s profile-scoped settings from Supabase for profile '%s'",
                len(loaded),
                profile_name,
            )
            return

        if seed_if_missing:
            payload = self._profile_scoped_settings_payload()
            seeded = self.analytics_store.supabase_sync.upsert_profile_settings(profile_name, payload)
            if seeded:
                logger.info("Seeded profile-scoped settings to Supabase for profile '%s'", profile_name)

    def _sync_profile_scoped_settings_to_supabase(self) -> None:
        if not bool(self.config.get("enable_supabase_sync", False)):
            self._save_profile_scoped_settings_to_local_cache(self.profile_name or self._get_profile_name())
            return

        profile_name = str(self.profile_name or self._get_profile_name()).strip() or "default"
        self.config["profile_name"] = profile_name
        payload = self._profile_scoped_settings_payload()
        self._save_profile_scoped_settings_to_local_cache(profile_name)
        synced = self.analytics_store.supabase_sync.upsert_profile_settings(profile_name, payload)
        if not synced:
            logger.debug(
                "Skipped syncing profile-scoped settings for profile '%s' (Supabase unavailable)",
                profile_name,
            )

    def _reset_session_tracking(self) -> None:
        """Reset counters at the beginning of a tracking session."""
        self.session_time_seconds = 0
        self._session_paused = False
        self._pause_started_at = 0.0
        self._paused_total_seconds = 0.0
        self.focus_time = 0.0
        self.raw_focus_time = 0.0
        self.distraction_count = 0
        self.break_count = 0
        self.score_samples = []
        self.raw_score_samples = []
        self.focus_trend_samples = []
        self.current_state = FocusState.UNCERTAIN
        self.display_state = FocusState.UNCERTAIN
        self._display_focused_state = None
        self._display_hold_until = 0.0
        self.current_score = 100.0
        self._display_score = 100.0
        self.continuous_focus_time = 0.0
        self.state_time_by_state = {state.name: 0.0 for state in FocusState}
        self.raw_state_time_by_state = {state.name: 0.0 for state in FocusState}

        self._session_eye_metric_frames = 0
        self._session_blink_count = 0
        self._session_eye_closed_frames = 0
        self._session_perclos_frames = 0
        self._session_ear_sum = 0.0
        self._session_ear_samples = 0
        self._session_focus_score_start: Optional[float] = None
        self._session_focus_score_end: Optional[float] = None
        self._session_fatigue_onset_seconds: Optional[float] = None
        self._session_total_frames = 0
        self._session_face_detected_frames = 0
        self._session_uncertain_noise_seconds = 0.0
        self._session_uncertain_behavioral_seconds = 0.0
        self._session_uncertain_clean_candidate_seconds = 0.0
        self._session_state_segments: list[Dict[str, Any]] = []
        self._session_eye_metric_seconds = 0.0
        self._last_state_frame_timestamp = None
        self._initial_baseline_samples = []
        self._initial_session_baseline = {}
        self._initial_baseline_finalized = False

        self._session_context_payload = {}
        self._session_exit_payload = {}
        self._session_route_payload = {}
        self._session_journey_enabled = False
        self._journey_phase_end = "Boarding"
        self._journey_completion_ratio = 0.0
        self._journey_waiting_for_boarding = False
        self._journey_calibration_reset_done = True
        self._journey_session_id = 0
        self._validation_session_id = ""
        self._session_checkins = []
        self._checkin_timestamps = []
        self._last_checkin_at = 0.0
        self._last_behavior_summary = {}
        self._behavior_checkin_modifier_since = {}
        self._focus_confirmation_until = 0.0
        self._last_vision_process_at = 0.0
        self._vision_processing_ema_ms = 0.0
        self._vision_effective_frames = 0
        self._vision_skipped_frames = 0
        self._last_perf_log_at = 0.0
        self._last_boarding_preview_at = 0.0
        self._context_alert_last_signature = ""
        self._context_alert_last_at = 0.0
        try:
            self.task_context_classifier.update_from_app_config(self.config)
            self.task_context_classifier.clear_samples()
            self.task_context_stats = TaskContextStats()
        except Exception as exc:
            logger.debug("Failed to reset task context classifier: %s", exc)

        self.zalo_alert_manager.reset_session()

    def _compute_frame_elapsed_seconds(self, frame_timestamp: float) -> float:
        """Return elapsed real-time seconds between processed frames."""
        fallback_dt = max(1.0 / 120.0, float(self.frame_interval) / 1000.0)
        last_ts = self._last_state_frame_timestamp
        self._last_state_frame_timestamp = frame_timestamp

        if last_ts is None:
            return fallback_dt

        dt = frame_timestamp - last_ts
        if dt <= 0.0:
            return fallback_dt

        # Clamp spikes from occasional pipeline stalls.
        return max(1.0 / 120.0, min(0.5, dt))

    def _track_session_eye_metrics(self, features: FrameFeatures, elapsed_seconds: float) -> None:
        """Accumulate blink/closure/EAR metrics for per-session personalization."""
        if self._is_initial_analysis_phase():
            return

        if not features.face_detected:
            return

        if features.ear_avg is None and features.eye_closure_level is None:
            return

        self._session_eye_metric_frames += 1
        self._session_eye_metric_seconds += max(0.0, float(elapsed_seconds))

        if features.blink_detected:
            self._session_blink_count += 1
        if features.is_eye_closed:
            self._session_eye_closed_frames += 1

        if features.ear_avg is not None:
            ear_value = float(features.ear_avg)
            self._session_ear_sum += ear_value
            self._session_ear_samples += 1
            if ear_value <= self.engine.config.drowsy_ear_threshold:
                self._session_perclos_frames += 1
        elif features.eye_closure_level is not None and float(features.eye_closure_level) >= 0.8:
            self._session_perclos_frames += 1

    @staticmethod
    def _median_float(values: list[Any]) -> Optional[float]:
        clean: list[float] = []
        for value in values:
            try:
                if value is None or value == "":
                    continue
                clean.append(float(value))
            except (TypeError, ValueError):
                continue
        if not clean:
            return None
        clean.sort()
        mid = len(clean) // 2
        if len(clean) % 2:
            return float(clean[mid])
        return float((clean[mid - 1] + clean[mid]) / 2.0)

    @staticmethod
    def _mean_float(values: list[Any]) -> Optional[float]:
        clean: list[float] = []
        for value in values:
            try:
                if value is None or value == "":
                    continue
                clean.append(float(value))
            except (TypeError, ValueError):
                continue
        if not clean:
            return None
        return float(sum(clean) / len(clean))

    @staticmethod
    def _clamped_ratio(value: float, low: float = 0.0, high: float = 1.0) -> float:
        try:
            return max(low, min(high, float(value)))
        except (TypeError, ValueError):
            return low

    def _record_initial_session_baseline_sample(
        self,
        *,
        features: FrameFeatures,
        score: float,
        state: FocusState,
        state_info: Dict[str, Any],
        lighting: str,
        frame_timestamp: float,
    ) -> None:
        """Collect the first stable seconds as a within-session reference point."""
        if (
            not self.camera_running
            or not self._is_initial_analysis_phase()
            or bool(getattr(self, "_journey_waiting_for_boarding", False))
            or bool(getattr(self, "_initial_baseline_finalized", False))
        ):
            return

        summary = state_info.get("behavior_summary", {}) if isinstance(state_info, dict) else {}
        if not isinstance(summary, dict):
            summary = {}
        sample = {
            "timestamp": float(frame_timestamp),
            "work_readiness": float(max(0.0, min(100.0, float(score or 0.0)))),
            "raw_state": state.name,
            "confidence": float(state_info.get("confidence", 0.0) or 0.0),
            "fatigue_index": float(summary.get("fatigue_index", 0.0) or 0.0),
            "distraction_risk": float(summary.get("distraction_risk", 0.0) or 0.0),
            "engagement_index": float(summary.get("engagement_index", 0.0) or 0.0),
            "confidence_index": float(summary.get("confidence_index", 0.0) or 0.0),
            "face_present": bool(features.face_detected),
            "vision_confidence": float(features.vision_confidence or 0.0),
            "camera_quality": str(lighting or ""),
            "ear_avg": features.ear_avg,
            "eye_closure_level": features.eye_closure_level,
            "perclos_ratio": features.perclos_ratio,
            "head_pitch": features.head_pitch,
            "head_yaw": features.head_yaw,
            "head_roll": features.head_roll,
            "hand_write_score": float(features.hand_write_score or 0.0),
            "phone_present": bool(features.phone_present),
        }
        self._initial_baseline_samples.append(sample)
        if len(self._initial_baseline_samples) > 240:
            self._initial_baseline_samples = self._initial_baseline_samples[-240:]

    def _finalize_initial_session_baseline(self, finished_at: Optional[float] = None) -> Dict[str, Any]:
        """Summarize warmup samples into a start-of-session behavioral baseline."""
        if bool(getattr(self, "_initial_baseline_finalized", False)):
            return dict(getattr(self, "_initial_session_baseline", {}) or {})

        self._initial_baseline_finalized = True
        samples = list(getattr(self, "_initial_baseline_samples", []) or [])
        if not samples:
            self._initial_session_baseline = {
                "initial_baseline_available": False,
                "initial_baseline_quality": 0.0,
                "initial_baseline_samples": 0,
            }
            return dict(self._initial_session_baseline)

        started_at = float(samples[0].get("timestamp", time.time()) or time.time())
        ended_at = float(finished_at if finished_at is not None else samples[-1].get("timestamp", time.time()))
        duration = max(0.0, ended_at - started_at)
        face_ratio = sum(1 for item in samples if bool(item.get("face_present"))) / max(1, len(samples))
        phone_ratio = sum(1 for item in samples if bool(item.get("phone_present"))) / max(1, len(samples))
        vision_conf = self._median_float([item.get("vision_confidence") for item in samples]) or 0.0
        duration_ratio = self._clamped_ratio(duration / max(1.0, float(self._analysis_warmup_seconds)))
        sample_ratio = self._clamped_ratio(len(samples) / 10.0)
        quality = self._clamped_ratio(
            (face_ratio * 0.42) + (vision_conf * 0.34) + (duration_ratio * 0.16) + (sample_ratio * 0.08)
        )

        state_counts: Dict[str, int] = {}
        for item in samples:
            state_name = str(item.get("raw_state", "") or "")
            if state_name:
                state_counts[state_name] = state_counts.get(state_name, 0) + 1
        dominant_state = max(state_counts.items(), key=lambda pair: pair[1])[0] if state_counts else ""

        self._initial_session_baseline = {
            "initial_baseline_available": True,
            "initial_baseline_quality": round(float(quality), 4),
            "initial_baseline_samples": int(len(samples)),
            "initial_baseline_duration_seconds": round(float(duration), 2),
            "initial_baseline_started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
            "initial_baseline_finished_at": datetime.fromtimestamp(ended_at).isoformat(timespec="seconds"),
            "initial_work_readiness": self._median_float([item.get("work_readiness") for item in samples]),
            "initial_fatigue_index": self._median_float([item.get("fatigue_index") for item in samples]),
            "initial_distraction_risk": self._median_float([item.get("distraction_risk") for item in samples]),
            "initial_engagement_index": self._median_float([item.get("engagement_index") for item in samples]),
            "initial_confidence_index": self._median_float([item.get("confidence_index") for item in samples]),
            "initial_camera_confidence": round(float(vision_conf), 4),
            "initial_face_presence_ratio": round(float(face_ratio), 4),
            "initial_phone_presence_ratio": round(float(phone_ratio), 4),
            "initial_eye_open_baseline": self._median_float([
                item.get("ear_avg") for item in samples
                if item.get("ear_avg") not in (None, "") and not bool(item.get("phone_present"))
            ]),
            "initial_eye_closure_baseline": self._median_float([item.get("eye_closure_level") for item in samples]),
            "initial_perclos_ratio": self._median_float([item.get("perclos_ratio") for item in samples]),
            "initial_head_pitch": self._median_float([item.get("head_pitch") for item in samples]),
            "initial_head_yaw": self._median_float([item.get("head_yaw") for item in samples]),
            "initial_head_roll": self._median_float([item.get("head_roll") for item in samples]),
            "initial_hand_write_score": self._median_float([item.get("hand_write_score") for item in samples]),
            "initial_dominant_raw_state": dominant_state,
        }
        logger.info(
            "Initial session baseline captured: samples=%s quality=%.2f readiness=%s",
            len(samples),
            quality,
            self._initial_session_baseline.get("initial_work_readiness"),
        )
        return dict(self._initial_session_baseline)

    def _initial_baseline_delta_fields(
        self,
        *,
        work_readiness: Optional[float] = None,
        fatigue_index: Optional[float] = None,
        distraction_risk: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return flat fields comparing current signals to the session-start baseline."""
        baseline = dict(getattr(self, "_initial_session_baseline", {}) or {})
        if not baseline.get("initial_baseline_available"):
            return {}

        fields: Dict[str, Any] = {
            "initial_work_readiness": baseline.get("initial_work_readiness"),
            "initial_fatigue_index": baseline.get("initial_fatigue_index"),
            "initial_distraction_risk": baseline.get("initial_distraction_risk"),
            "initial_baseline_quality": baseline.get("initial_baseline_quality"),
        }

        initial_wr = baseline.get("initial_work_readiness")
        if initial_wr not in (None, "") and work_readiness is not None:
            current_wr = float(work_readiness)
            initial_wr_float = float(initial_wr)
            fields["readiness_delta_from_start"] = round(current_wr - initial_wr_float, 3)
            if initial_wr_float > 1e-6:
                fields["recovery_to_initial_ratio"] = round(self._clamped_ratio(current_wr / initial_wr_float, 0.0, 1.5), 4)

        initial_fatigue = baseline.get("initial_fatigue_index")
        if initial_fatigue not in (None, "") and fatigue_index is not None:
            fields["fatigue_delta_from_start"] = round(float(fatigue_index) - float(initial_fatigue), 4)

        initial_distraction = baseline.get("initial_distraction_risk")
        if initial_distraction not in (None, "") and distraction_risk is not None:
            fields["distraction_delta_from_start"] = round(float(distraction_risk) - float(initial_distraction), 4)

        return fields

    def _is_initial_analysis_phase(self) -> bool:
        """Return True while the startup calibration window is still active."""
        if not self.camera_running:
            return False
        if bool(getattr(self, "_journey_waiting_for_boarding", False)):
            return True
        session_start = getattr(self, "session_started_at", None)
        if session_start is not None and time.time() < float(session_start):
            return True
        if self._analysis_started_at <= 0.0:
            return False
        return (time.time() - self._analysis_started_at) < self._analysis_warmup_seconds

    def _analysis_seconds_left(self) -> int:
        """Return rounded-up remaining warmup seconds."""
        session_start = getattr(self, "session_started_at", None)
        if session_start is not None and time.time() < float(session_start):
            return int(math.ceil(max(0.0, float(session_start) - time.time())))
        if self._analysis_started_at <= 0.0:
            return 0
        elapsed = time.time() - self._analysis_started_at
        remaining = max(0.0, self._analysis_warmup_seconds - elapsed)
        return int(math.ceil(remaining))

    def _compute_display_score(self, raw_score: float) -> float:
        """Apply startup hold and asymmetric smoothing to score transitions."""
        if self._is_initial_analysis_phase():
            self._display_score = 100.0
            return self._display_score

        try:
            target_score = max(0.0, min(100.0, float(raw_score)))
        except (TypeError, ValueError):
            target_score = self._display_score

        frame_seconds = max(0.016, float(self.frame_interval) / 1000.0)
        blended_target = (self._display_score * 0.82) + (target_score * 0.18)

        if blended_target < self._display_score:
            max_step = self._score_drop_speed_per_sec * frame_seconds
            self._display_score = max(blended_target, self._display_score - max_step)
        else:
            max_step = self._score_rise_speed_per_sec * frame_seconds
            self._display_score = min(blended_target, self._display_score + max_step)

        return self._display_score

    def _apply_personalized_schedule(self) -> None:
        """Load and optionally apply personalized work/break timing for the active profile."""
        self.profile_name = self._get_profile_name()
        self.engine.set_profile(self.profile_name)
        self.config["enable_personalization"] = True
        self.config["auto_apply_personalization"] = True
        self._apply_profile_vision_calibration()

        try:
            minutes_since_last_break = max(0.0, (time.time() - self.last_break_time) / 60.0)
            bundle = self.analytics_store.get_personalization_bundle(
                self.profile_name,
                default_work=int(self.config.get("break_interval_minutes", 25)),
                default_break=int(self.config.get("break_duration_minutes", 5)),
                minutes_since_last_break=minutes_since_last_break,
                focus_engine_defaults=self._focus_engine_defaults(),
            )
        except Exception as exc:
            logger.warning("Failed to load personalized schedule: %s", exc)
            return

        recommendation = bundle.get("recommendation", {})
        baseline_payload = bundle.get("baseline", {}) or {}
        threshold_payload = bundle.get("thresholds", {}) or {}

        self._last_recommendation = recommendation
        self.config["break_interval_minutes"] = int(recommendation.get("work_minutes", 25))
        self.config["break_duration_minutes"] = int(recommendation.get("break_minutes", 5))
        self.config_changed.emit(self.config.copy())

        self.engine.set_personalized_thresholds(
            personalized_thresholds=threshold_payload,
            profile_name=self.profile_name,
            user_baseline=baseline_payload,
            session_context={
                "minutes_since_last_break": minutes_since_last_break,
                "is_tracking": bool(self.camera_running),
            },
        )
        self._apply_personalized_vision_thresholds(threshold_payload)

        logger.info(
            "Personalized plan for profile '%s': work=%s min, break=%s min (%s, stage=%s)",
            self.profile_name,
            recommendation.get("work_minutes", 25),
            recommendation.get("break_minutes", 5),
            recommendation.get("reason", "n/a"),
            recommendation.get("adaptation_stage", "cold_start"),
        )

    def _current_schedule_minutes(self) -> tuple[int, int]:
        """Return current active work/break minutes with recommendation priority."""
        default_work = int(self.config.get("break_interval_minutes", 25))
        default_break = int(self.config.get("break_duration_minutes", 5))

        work_raw = self._last_recommendation.get("work_minutes", default_work)
        break_raw = self._last_recommendation.get("break_minutes", default_break)

        try:
            work_minutes = int(float(work_raw))
        except (TypeError, ValueError):
            work_minutes = int(default_work)

        try:
            break_minutes = int(float(break_raw))
        except (TypeError, ValueError):
            break_minutes = int(default_break)

        work_minutes = max(15, min(60, work_minutes))
        break_minutes = max(3, min(20, break_minutes))

        changed = False
        if int(self.config.get("break_interval_minutes", default_work)) != work_minutes:
            self.config["break_interval_minutes"] = work_minutes
            changed = True
        if int(self.config.get("break_duration_minutes", default_break)) != break_minutes:
            self.config["break_duration_minutes"] = break_minutes
            changed = True

        if changed:
            self.config_changed.emit(self.config.copy())

        return work_minutes, break_minutes

    def _task_context_enabled(self) -> bool:
        """Whether foreground app/window context sampling is enabled."""
        return bool(self.config.get("enable_task_context_monitoring", True))

    def _sample_task_context(self) -> None:
        """Sample active app/window metadata and update context-aware UI."""
        if not self._task_context_enabled():
            return

        try:
            now = time.time()
            sample = self.task_context_monitor.get_active_context(timestamp=now)
            sample = self.task_context_classifier.annotate(sample)
            self.task_context_stats = self.task_context_classifier.compute_stats(now=now)
            self._update_task_context_card(sample, self.task_context_stats)
            self._update_score_breakdown()
            self._maybe_show_context_alert(sample, self.task_context_stats)
            self._maybe_show_context_checkin(sample, self.task_context_stats)
        except Exception as exc:
            logger.debug("Task context sampling failed: %s", exc)

    @staticmethod
    def _context_csv_tokens(value: Any, fallback: Any = "") -> tuple[str, ...]:
        tokens = TaskContextClassifier._normalize_tokens(value)
        if tokens:
            return tokens
        return TaskContextClassifier._normalize_tokens(fallback)

    def _context_alert_match_reason(self, sample) -> str:
        default_keywords = PROFILE_SCOPED_DEFAULT_SETTINGS.get("task_context_distracting_keywords", "")
        default_apps = PROFILE_SCOPED_DEFAULT_SETTINGS.get("task_context_distracting_apps", "")
        keywords = self._context_csv_tokens(
            self.config.get("task_context_distracting_keywords", default_keywords),
            default_keywords,
        )
        apps = self._context_csv_tokens(
            self.config.get("task_context_distracting_apps", default_apps),
            default_apps,
        )

        process_name = str(getattr(sample, "process_name", "") or "").strip().lower()
        app_id = str(getattr(sample, "app_id", "") or "").strip().lower()
        title = str(getattr(sample, "window_title", "") or "").strip().lower()
        context_text = str(getattr(sample, "context_text", "") or "").strip().lower()
        app_text = f"{process_name} {app_id}".strip()
        title_text = f"{title} {context_text}".strip()

        for app_rule in apps:
            token = str(app_rule or "").strip().lower()
            if token and (token == process_name or token == app_id or token in app_text):
                return f"app:{token}"

        for keyword in keywords:
            token = str(keyword or "").strip().lower()
            if token and (token in title_text or token in app_text):
                return f"keyword:{token}"

        return ""

    def _maybe_show_context_alert(self, sample, stats: TaskContextStats) -> None:
        """Notify only when the active app/tab looks likely to pull attention away."""
        if not bool(getattr(self, "camera_running", False)):
            return
        if bool(getattr(self, "_session_paused", False)) or bool(getattr(self, "_break_dialog_open", False)):
            return
        if not bool(self.config.get("task_context_alert_enabled", True)):
            return

        category = str(getattr(sample, "category", "unknown") or "unknown").strip().lower()
        if category != "distracting":
            return

        try:
            risk = float(getattr(stats, "risk_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            risk = 0.0
        try:
            threshold = float(self.config.get("task_context_alert_threshold", 0.68) or 0.68)
        except (TypeError, ValueError):
            threshold = 0.68
        if risk < max(0.2, min(0.98, threshold)):
            return

        match_reason = self._context_alert_match_reason(sample)
        if not match_reason:
            return

        now = time.time()
        try:
            cooldown = float(self.config.get("task_context_alert_cooldown_seconds", 120) or 120)
        except (TypeError, ValueError):
            cooldown = 120.0
        cooldown = max(30.0, min(900.0, cooldown))

        signature = f"{str(getattr(sample, 'app_id', '') or '')}|{match_reason}"
        if signature == self._context_alert_last_signature and now - self._context_alert_last_at < cooldown:
            return

        self._context_alert_last_signature = signature
        self._context_alert_last_at = now

        payload = {
            "timestamp": int(now),
            "checkin_type": "context_alert",
            "active_app": str(getattr(sample, "process_name", "") or ""),
            "active_category": category,
            "alert_reason": match_reason,
            "action": "notification_only",
            "task_alignment_ratio": float(getattr(stats, "task_alignment_ratio", 0.0) or 0.0),
            "digital_distraction_risk": float(getattr(stats, "risk_score", 0.0) or 0.0),
            "focus_score": float(getattr(self, "current_score", 0.0) or 0.0),
        }
        self._session_checkins.append(payload)

        app_name = payload["active_app"] or "ứng dụng hiện tại"
        title = "Nhắc nhẹ Deep Focus" if bool(getattr(self, "_deep_focus_active", False)) else "Nhắc nhẹ nhịp làm việc"
        message = f"{app_name} có dấu hiệu dễ kéo bạn khỏi nhiệm vụ. Quay lại việc chính nhé."
        self.context_alert_requested.emit(title, message)
        logger.info("Task context alert: app=%s reason=%s risk=%.2f", app_name, match_reason, risk)

    @staticmethod
    def _category_label(category: str) -> str:
        labels = {
            "task_related": "Liên quan nhiệm vụ",
            "neutral": "Trung tính",
            "distracting": "Có nguy cơ gây xao nhãng",
            "unknown": "Không rõ",
            "excluded": "Đã loại trừ",
        }
        return labels.get(str(category or "unknown").strip().lower(), "Không rõ")

    def _update_task_context_card(self, sample, stats: TaskContextStats) -> None:
        """Refresh the small task-context card."""
        if not hasattr(self, "task_context_app_label"):
            return

        app_name = str(getattr(sample, "process_name", "") or "").strip()
        if not app_name:
            app_name = str(getattr(sample, "app_id", "") or "Không rõ")

        category = str(getattr(sample, "category", "") or stats.current_category or "unknown")
        alignment_percent = int(max(0.0, min(1.0, float(stats.task_alignment_ratio))) * 100)

        self.task_context_app_label.setText(f"Ứng dụng: {app_name}")
        self.task_context_category_label.setText(f"Loại: {self._category_label(category)}")
        self.task_context_alignment_label.setText(f"Task Alignment: {alignment_percent}%")
        self.task_context_switch_label.setText(f"Chuyển app: {int(stats.context_switch_count)}")

        if category == "distracting":
            self.task_context_hint_label.setText(
                "Bạn đang ở app có nguy cơ gây xao nhãng. Nếu đây là app phục vụ công việc, hãy thêm vào nhóm liên quan nhiệm vụ trong Cài đặt."
            )
        elif category == "task_related":
            self.task_context_hint_label.setText("Bối cảnh số hiện tại đang phù hợp với nhiệm vụ.")
        elif category == "excluded":
            self.task_context_hint_label.setText("Ứng dụng này đã được loại trừ khỏi phân tích bối cảnh.")
        else:
            self.task_context_hint_label.setText(
                "Chỉ dùng metadata ứng dụng/cửa sổ để hiểu bối cảnh, không chụp màn hình."
            )

    def _maybe_show_context_checkin(self, sample, stats: TaskContextStats) -> None:
        """Ask a short human-in-the-loop check-in when context risk is high."""
        if not bool(getattr(self, "camera_running", False)):
            return
        if bool(getattr(self, "_break_dialog_open", False)):
            return
        if bool(getattr(self, "_deep_focus_active", False)):
            return
        if bool(self.config.get("task_context_alert_enabled", True)):
            return
        if not bool(self.config.get("task_context_checkin_enabled", True)):
            return

        try:
            risk = float(getattr(stats, "risk_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            risk = 0.0

        try:
            risk_threshold = float(self.config.get("task_context_checkin_risk_threshold", 0.72) or 0.72)
        except (TypeError, ValueError):
            risk_threshold = 0.72

        if risk < risk_threshold:
            return

        now = time.time()
        try:
            cooldown_minutes = float(self.config.get("task_context_checkin_cooldown_minutes", 8) or 8)
        except (TypeError, ValueError):
            cooldown_minutes = 8.0

        if now - self._last_checkin_at < cooldown_minutes * 60.0:
            return

        try:
            max_per_hour = int(float(self.config.get("task_context_checkin_max_per_hour", 3) or 3))
        except (TypeError, ValueError):
            max_per_hour = 3

        self._checkin_timestamps = [t for t in self._checkin_timestamps if now - float(t) <= 3600.0]
        if len(self._checkin_timestamps) >= max_per_hour:
            return

        self._last_checkin_at = now
        self._checkin_timestamps.append(now)

        dialog = ContextCheckInDialog(
            risk_score=risk,
            state_name=STATE_NAMES.get(self.current_state, self.current_state.name),
            config=self.config,
            parent=self,
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            payload = dialog.get_payload()
        else:
            payload = {
                "answer": "skipped",
                "answer_label": "Bỏ qua",
                "note": "",
                "risk_score": risk,
                "state_name": STATE_NAMES.get(self.current_state, self.current_state.name),
            }

        payload.update(
            {
                "timestamp": int(now),
                "active_app": str(getattr(sample, "process_name", "") or ""),
                "active_category": str(getattr(sample, "category", "unknown") or "unknown"),
                "task_alignment_ratio": float(getattr(stats, "task_alignment_ratio", 0.0) or 0.0),
                "digital_distraction_risk": float(getattr(stats, "risk_score", 0.0) or 0.0),
                "focus_score": float(getattr(self, "current_score", 0.0) or 0.0),
            }
        )
        self._session_checkins.append(payload)

    def _maybe_show_behavior_checkin(self, summary: Dict[str, Any], now: Optional[float] = None) -> None:
        """Ask for a light self-report when behavior channels conflict for a while."""
        if not isinstance(summary, dict):
            return
        if not bool(getattr(self, "camera_running", False)):
            return
        if bool(getattr(self, "_break_dialog_open", False)):
            return
        if not bool(self.config.get("behavior_conflict_checkin_enabled", True)):
            return

        modifier = str(summary.get("status_modifier", "") or "")
        if modifier not in {"fatigued_but_working", "possible_passive_attention"}:
            self._behavior_checkin_modifier_since.pop("fatigued_but_working", None)
            self._behavior_checkin_modifier_since.pop("possible_passive_attention", None)
            return

        now_ts = float(now if now is not None else time.time())
        if bool(getattr(self, "_deep_focus_active", False)):
            try:
                risk = float(summary.get("distraction_risk", 0.0) or 0.0)
            except (TypeError, ValueError):
                risk = 0.0
            if risk < 0.78:
                return

        since = self._behavior_checkin_modifier_since.get(modifier)
        if since is None:
            self._behavior_checkin_modifier_since[modifier] = now_ts
            return

        try:
            delay_seconds = float(self.config.get("behavior_conflict_checkin_delay_seconds", 90) or 90)
        except (TypeError, ValueError):
            delay_seconds = 90.0
        if now_ts - since < max(30.0, delay_seconds):
            return

        try:
            cooldown_minutes = float(self.config.get("task_context_checkin_cooldown_minutes", 8) or 8)
        except (TypeError, ValueError):
            cooldown_minutes = 8.0
        if now_ts - self._last_checkin_at < cooldown_minutes * 60.0:
            return

        try:
            max_per_hour = int(float(self.config.get("task_context_checkin_max_per_hour", 3) or 3))
        except (TypeError, ValueError):
            max_per_hour = 3
        self._checkin_timestamps = [t for t in self._checkin_timestamps if now_ts - float(t) <= 3600.0]
        if len(self._checkin_timestamps) >= max_per_hour:
            return

        self._last_checkin_at = now_ts
        self._checkin_timestamps.append(now_ts)
        self._behavior_checkin_modifier_since[modifier] = now_ts

        risk_score = max(
            float(summary.get("fatigue_index", 0.0) or 0.0),
            float(summary.get("distraction_risk", 0.0) or 0.0),
            0.42,
        )
        state_name = (
            "Đang làm việc nhưng có dấu hiệu mệt"
            if modifier == "fatigued_but_working"
            else "Có thể đang đọc thụ động"
        )
        dialog = ContextCheckInDialog(
            risk_score=risk_score,
            state_name=state_name,
            config=self.config,
            parent=self,
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            payload = dialog.get_payload()
            answer = str(payload.get("answer", "") or "")
            if answer in {"on_task", "slight_drift"}:
                self._focus_confirmation_until = now_ts + 300.0
                if hasattr(self.engine, "apply_behavior_feedback"):
                    self.engine.apply_behavior_feedback(answer, hold_seconds=300.0)
            elif answer == "need_break":
                if hasattr(self.engine, "apply_behavior_feedback"):
                    self.engine.apply_behavior_feedback(answer, hold_seconds=0.0)
        else:
            payload = {
                "answer": "skipped",
                "answer_label": "Bỏ qua",
                "note": "",
                "risk_score": risk_score,
                "state_name": state_name,
            }

        payload.update(
            {
                "timestamp": int(now_ts),
                "checkin_type": "behavior_conflict",
                "status_modifier": modifier,
                "engagement_index": float(summary.get("engagement_index", 0.0) or 0.0),
                "fatigue_index": float(summary.get("fatigue_index", 0.0) or 0.0),
                "distraction_risk": float(summary.get("distraction_risk", 0.0) or 0.0),
                "confidence_index": float(summary.get("confidence_index", 0.0) or 0.0),
                "focus_score": float(getattr(self, "current_score", 0.0) or 0.0),
            }
        )
        self._session_checkins.append(payload)

    def _update_focus_journey_origin_after_session(self, session_seconds: int) -> None:
        """Advance the next journey origin only after the current flight reaches destination."""
        if not bool(getattr(self, "_session_journey_enabled", False)):
            return
        payload = dict(getattr(self, "_session_context_payload", {}) or {})
        payload.update(dict(getattr(self, "_session_route_payload", {}) or {}))
        to_code = str(payload.get("route_to_code", "") or "").strip().upper()
        if not to_code:
            return

        planned_minutes = int(
            getattr(self, "_session_planned_minutes", 0)
            or payload.get("planned_minutes", 0)
            or payload.get("route_duration_minutes", 0)
            or 0
        )
        completion_ratio = float(getattr(self, "_journey_completion_ratio", 0.0) or 0.0)
        if planned_minutes > 0:
            completion_ratio = max(
                completion_ratio,
                min(1.0, float(session_seconds) / max(1.0, float(planned_minutes * 60))),
            )

        if completion_ratio < 0.985:
            return

        to_name = str(payload.get("route_to_name", "") or to_code)
        self.config["focus_journey_current_airport"] = to_code
        self.config["focus_journey_current_airport_name"] = to_name
        self.config["last_journey_to_code"] = to_code
        self.config["last_journey_to_name"] = to_name
        self.config["last_journey_arrived_at"] = int(time.time())

    def _persist_session_analytics(self) -> None:
        """Persist current session data for later analysis and personalization."""
        if self.session_started_at is None:
            return

        now = time.time()
        session_seconds = max(
            int(now - self.session_started_at - self._effective_paused_total_seconds(now)),
            int(self.session_time_seconds),
        )
        self.session_started_at = None

        # Ignore too-short runs to avoid noisy personalization.
        if session_seconds < 30:
            return

        avg_score_raw = (
            float(sum(self.raw_score_samples) / len(self.raw_score_samples))
            if self.raw_score_samples
            else float(self.current_score)
        )
        avg_score_display = (
            float(sum(self.score_samples) / len(self.score_samples))
            if self.score_samples
            else float(self.current_score)
        )

        eye_metric_frames = max(0, int(getattr(self, "_session_eye_metric_frames", 0)))
        eye_metric_seconds = max(0.0, float(getattr(self, "_session_eye_metric_seconds", 0.0) or 0.0))
        if eye_metric_seconds <= 0.0 and eye_metric_frames > 0:
            eye_metric_seconds = eye_metric_frames * max(0.001, float(self.frame_interval) / 1000.0)
        blink_rate_per_min = (
            (float(getattr(self, "_session_blink_count", 0)) * 60.0) / max(eye_metric_seconds, 1e-6)
            if eye_metric_frames > 0
            else 0.0
        )
        eye_closure_ratio = (
            float(getattr(self, "_session_eye_closed_frames", 0)) / eye_metric_frames
            if eye_metric_frames > 0
            else 0.0
        )
        perclos = (
            float(getattr(self, "_session_perclos_frames", 0)) / eye_metric_frames
            if eye_metric_frames > 0
            else 0.0
        )
        avg_ear = (
            float(getattr(self, "_session_ear_sum", 0.0)) / max(1, int(getattr(self, "_session_ear_samples", 0)))
            if int(getattr(self, "_session_ear_samples", 0)) > 0
            else 0.0
        )

        focus_score_start = self._session_focus_score_start if self._session_focus_score_start is not None else avg_score_raw
        focus_score_end = self._session_focus_score_end if self._session_focus_score_end is not None else avg_score_raw
        score_drop_per_hour = (
            ((float(focus_score_start) - float(focus_score_end)) / max(session_seconds / 3600.0, 1e-6))
            if session_seconds > 0
            else 0.0
        )

        minutes_since_last_break = max(0.0, (time.time() - self.last_break_time) / 60.0)
        fatigue_onset_minutes = (
            (self._session_fatigue_onset_seconds / 60.0)
            if self._session_fatigue_onset_seconds is not None
            else None
        )

        face_presence_ratio = (
            float(getattr(self, "_session_face_detected_frames", 0))
            / max(1.0, float(getattr(self, "_session_total_frames", 0)))
        )

        raw_state_seconds = self.raw_state_time_by_state.copy()
        display_state_seconds = self.state_time_by_state.copy()
        active_work_minutes, active_break_minutes = self._current_schedule_minutes()
        if self._session_planned_minutes > 0:
            self._journey_completion_ratio = max(
                0.0,
                min(1.0, float(session_seconds) / max(1.0, float(self._session_planned_minutes * 60))),
            )
            self._journey_phase_end = self._journey_phase_from_progress(int(self._journey_completion_ratio * 100))

        task_context_summary: Dict[str, Any] = {}
        try:
            self.task_context_stats = self.task_context_classifier.compute_stats(now=time.time())
            task_context_summary = self.task_context_classifier.summarize_for_report(self.task_context_stats)
        except Exception as exc:
            logger.debug("Failed to build task context summary: %s", exc)

        initial_baseline = dict(getattr(self, "_initial_session_baseline", {}) or {})
        baseline_deltas = self._initial_baseline_delta_fields(
            work_readiness=avg_score_raw,
            fatigue_index=None,
            distraction_risk=None,
        )

        session_record = {
            "timestamp": int(time.time()),
            "profile_name": self.profile_name,
            "session_id": str(getattr(self, "_validation_session_id", "") or getattr(self, "_journey_session_id", "")),
            "session_seconds": session_seconds,
            "focus_seconds": float(self.raw_focus_time),
            "focus_seconds_display": float(self.focus_time),
            "distraction_count": int(self.distraction_count),
            "break_count": int(self.break_count),
            "avg_score": avg_score_raw,
            "avg_score_display": avg_score_display,
            "min_score": float(min(self.raw_score_samples)) if self.raw_score_samples else float(self.current_score),
            "max_score": float(max(self.raw_score_samples)) if self.raw_score_samples else float(self.current_score),
            "focus_score_start": float(focus_score_start),
            "focus_score_end": float(focus_score_end),
            "score_drop_per_hour": float(score_drop_per_hour),
            "blink_rate_per_min": float(blink_rate_per_min),
            "avg_ear": float(avg_ear),
            "eye_closure_ratio": float(eye_closure_ratio),
            "perclos": float(perclos),
            "fatigue_onset_minutes": float(fatigue_onset_minutes) if fatigue_onset_minutes is not None else None,
            "minutes_since_last_break": float(minutes_since_last_break),
            "state_seconds": raw_state_seconds,
            "state_seconds_display": display_state_seconds,
            "state_segments": list(getattr(self, "_session_state_segments", [])),
            "uncertain_measurement_noise_seconds": float(getattr(self, "_session_uncertain_noise_seconds", 0.0)),
            "uncertain_behavioral_seconds": float(getattr(self, "_session_uncertain_behavioral_seconds", 0.0)),
            "uncertain_clean_candidate_seconds": float(getattr(self, "_session_uncertain_clean_candidate_seconds", 0.0)),
            "face_presence_ratio": float(face_presence_ratio),
            "work_interval_minutes_used": int(active_work_minutes),
            "break_duration_minutes_used": int(active_break_minutes),
            "session_context": dict(self._session_context_payload or {}),
            "session_mode": str(self._session_mode or "normal"),
            "duration_source": str((self._session_context_payload or {}).get("duration_source", "")),
            "selected_route_id": str((self._session_context_payload or {}).get("selected_route_id", "")),
            "selected_route_label": str((self._session_context_payload or {}).get("selected_route_label", "")),
            "route_from_code": str((self._session_context_payload or {}).get("route_from_code", "")),
            "route_to_code": str((self._session_context_payload or {}).get("route_to_code", "")),
            "route_theme": str((self._session_context_payload or {}).get("route_theme", "")),
            "planned_minutes": int(self._session_planned_minutes or 0),
            "journey_completion_ratio": float(getattr(self, "_journey_completion_ratio", 0.0) or 0.0),
            "journey_phase_end": str(getattr(self, "_journey_phase_end", "") or ""),
            "session_exit": dict(self._session_exit_payload or {}),
            "checkins": list(self._session_checkins or []),
            "initial_session_baseline": dict(initial_baseline),
            "initial_work_readiness": baseline_deltas.get("initial_work_readiness"),
            "readiness_delta_from_start": baseline_deltas.get("readiness_delta_from_start"),
            "initial_fatigue_index": baseline_deltas.get("initial_fatigue_index"),
            "initial_distraction_risk": baseline_deltas.get("initial_distraction_risk"),
            "initial_baseline_quality": baseline_deltas.get("initial_baseline_quality"),
            "task_context_summary": dict(task_context_summary or {}),
            "task_alignment_avg": float(task_context_summary.get("task_alignment_ratio", 0.0) or 0.0),
            "digital_distraction_risk_avg": float(task_context_summary.get("risk_score", 0.0) or 0.0),
            "context_switch_count": int(task_context_summary.get("context_switch_count", 0) or 0),
            "active_context_category": str(task_context_summary.get("current_category", "unknown") or "unknown"),
            "active_context_app": str(task_context_summary.get("current_app_id", "") or ""),
        }

        try:
            recommendation = self.analytics_store.record_session(
                self.profile_name,
                session_record,
                default_work=int(active_work_minutes),
                default_break=int(active_break_minutes),
            )
        except Exception as exc:
            logger.warning("Failed to persist session analytics: %s", exc)
            return

        self._last_recommendation = recommendation
        self.config["break_interval_minutes"] = int(recommendation.get("work_minutes", 25))
        self.config["break_duration_minutes"] = int(recommendation.get("break_minutes", 5))
        self._update_focus_journey_origin_after_session(session_seconds)
        self._refresh_today_stats_card()
        self.config_changed.emit(self.config.copy())

        logger.info(
            "Saved session analytics for '%s': duration=%ss, avg_score=%.1f, rec=%s/%s min",
            self.profile_name,
            session_seconds,
            avg_score_raw,
            recommendation.get("work_minutes", 25),
            recommendation.get("break_minutes", 5),
        )

    @pyqtSlot()
    def _toggle_tracking(self):
        """Toggle camera tracking on/off."""
        if self.btn_start.isChecked():
            self._start_tracking()
        else:
            self._stop_tracking()

    def _start_tracking(self):
        """Start camera and focus tracking."""
        if not self.vision_available:
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            py_exec = sys.executable
            project_python = Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe"
            run_hint = (
                f"\nGợi ý chạy đúng môi trường:\n{project_python} main.py"
                if project_python.exists()
                else ""
            )
            reason_text = f"\nChi tiết lỗi: {self._vision_init_error}" if self._vision_init_error else ""
            NoticeDialog.warning(
                self,
                "Vision không khả dụng",
                "Không thể khởi tạo vision pipeline trong môi trường hiện tại.\n"
                f"Python: {py_ver}\n"
                f"Interpreter: {py_exec}"
                f"{reason_text}"
                f"{run_hint}",
                config=self.config,
            )
            self.btn_start.setChecked(False)
            return

        try:
            if bool(self.config.get("session_goal_prompt_enabled", True)):
                context_dialog = SessionContextDialog(config=self.config, parent=self)
                if context_dialog.exec() == QDialog.DialogCode.Accepted:
                    self._session_context_payload = context_dialog.get_payload()
                    session_mode = str(self._session_context_payload.get("session_mode", "normal") or "normal")
                    self.config["deadline_mode_enabled"] = session_mode == "deadline"
                    if self._session_context_payload.get("deadline_minutes"):
                        self.config["deadline_focus_minutes"] = int(
                            self._session_context_payload.get("deadline_minutes")
                        )

                    # Show boarding pass — user confirms before tracking starts
                    self._session_mode = session_mode
                    self._session_goal = str(self._session_context_payload.get("goal", "") or "")
                    self._session_planned_minutes = int(self._session_context_payload.get("planned_minutes", 0) or 0)
                    self._session_journey_enabled = bool(self._session_context_payload.get("journey_enabled", False))
                    if self._session_journey_enabled:
                        boarding = SessionBoardingPassDialog(
                            context_payload=self._session_context_payload,
                            config=self.config,
                            parent=None,
                        )
                        QTimer.singleShot(0, self.hide)
                        if boarding.exec() != QDialog.DialogCode.Accepted:
                            self.showNormal()
                            self.raise_()
                            self.activateWindow()
                            self.btn_start.setChecked(False)
                            return
                        self._session_route_payload = dict(self._session_context_payload or {})
                    else:
                        self._session_route_payload = {}
                    self.config["session_mode"] = self._session_mode
                else:
                    self._session_context_payload = {
                        "goal": "",
                        "task_type": "unspecified",
                        "session_mode": "normal",
                        "planned_minutes": 0,
                        "deadline_mode": bool(self.config.get("deadline_mode_enabled", False)),
                        "deadline_minutes": int(self.config.get("deadline_focus_minutes", 45) or 45),
                        "note": "",
                        "skipped": True,
                    }
                    self._session_mode = "normal"
                    self._session_goal = ""
                    self._session_planned_minutes = 0
                    self._session_route_payload = {}
                    self._session_journey_enabled = False

            if not self.camera.start():
                NoticeDialog.warning(
                    self,
                    "Lỗi Camera",
                    "Không thể khởi động camera. Vui lòng kiểm tra kết nối.",
                    config=self.config,
                )
                if bool(getattr(self, "_session_journey_enabled", False)):
                    self.showNormal()
                    self.raise_()
                    self.activateWindow()
                self.btn_start.setChecked(False)
                return

            self._apply_personalized_schedule()
            pending_context_payload = dict(self._session_context_payload or {})
            pending_route_payload = dict(self._session_route_payload or {})
            pending_session_mode = str(self._session_mode or "normal")
            pending_session_goal = str(self._session_goal or "")
            pending_planned_minutes = int(self._session_planned_minutes or 0)
            pending_journey_enabled = bool(self._session_journey_enabled)
            self._reset_session_tracking()
            self._session_context_payload = pending_context_payload
            self._session_route_payload = pending_route_payload
            self._session_mode = pending_session_mode
            self._session_goal = pending_session_goal
            self._session_planned_minutes = pending_planned_minutes
            self._session_journey_enabled = pending_journey_enabled
            self._break_snapshots = []
            self._before_break_snapshot = {}
            self._journey_pip_hidden_for_session = False
            self._journey_pip_closed_until_restore = False
            self._journey_pip_progress_key = ()
            self._journey_waiting_for_boarding = self._session_journey_enabled
            self._journey_calibration_reset_done = True
            self._journey_session_id = int(time.time())
            self._validation_session_id = f"{self.profile_name}:{self._journey_session_id}"
            self.session_started_at = None
            self._analysis_started_at = 0.0
            self._session_paused = False
            self._pause_started_at = 0.0
            self._paused_total_seconds = 0.0
            self.last_break_time = time.time()

            # Activate deep focus mode if requested
            self._deep_focus_active = (self._session_mode == "deep")
            self._apply_deep_focus_ui(self._deep_focus_active)

            self.camera_running = True
            self.frame_timer.start(self.frame_interval)
            self.stats_timer.start(1000)
            if self._task_context_enabled():
                self.task_context_classifier.update_from_app_config(self.config)
                self.task_context_timer.start(self.task_context_interval_ms)
            self.btn_start.setText("Dừng")
            self.engine.reset()
            self.display_state = FocusState.UNCERTAIN
            self.score_widget.set_score(100.0, FocusState.UNCERTAIN)
            self._update_score_breakdown()
            self._update_state_badge(FocusState.UNCERTAIN, 0.0, "Đang theo dõi phiên làm việc...")
            self._update_live_status(face_detected=None, lighting="Calibrating")
            self._refresh_focus_guidance()
            if self._session_journey_enabled:
                self._update_journey_widget()
                QTimer.singleShot(650, self._open_journey_map_dialog)
            else:
                self._begin_journey_measurement_after_boarding()

            logger.info("Focus tracking started (mode=%s)", self._session_mode)

        except Exception as e:
            logger.error(f"Failed to start tracking: {e}")
            if bool(getattr(self, "_session_journey_enabled", False)):
                self.showNormal()
                self.raise_()
                self.activateWindow()
            NoticeDialog.error(
                self,
                "Lỗi",
                f"Không thể bắt đầu: {e}",
                config=self.config,
            )
            self.btn_start.setChecked(False)
            self._update_live_status(face_detected=False, lighting="Unknown")

    def _stop_tracking(self):
        """Stop camera and focus tracking."""
        was_running = self.camera_running
        if bool(getattr(self, "_session_paused", False)) and float(getattr(self, "_pause_started_at", 0.0) or 0.0) > 0.0:
            self._paused_total_seconds += max(0.0, time.time() - float(self._pause_started_at))
        self._session_paused = False
        self._pause_started_at = 0.0
        self.frame_timer.stop()
        self.stats_timer.stop()
        if hasattr(self, "task_context_timer"):
            self.task_context_timer.stop()
        if self.camera is not None:
            self.camera.stop()
        self.camera_running = False
        self._hide_journey_pip()
        self._analysis_started_at = 0.0
        self._display_score = 100.0
        self.display_state = FocusState.UNCERTAIN
        self.btn_start.setText("Bắt đầu")
        self.camera_widget.update_frame(None)
        self._update_state_badge(FocusState.UNCERTAIN, 0.0, "Đã dừng theo dõi.\nNhấn Bắt đầu để chạy lại.")
        self._update_live_status(face_detected=False, lighting="Unknown")
        self._refresh_focus_guidance()

        # Deactivate deep focus / journey UI
        self._apply_deep_focus_ui(False)
        if hasattr(self, "route_map_widget"):
            self.route_map_widget.setVisible(bool(getattr(self, "_session_journey_enabled", False)))
            self.route_map_widget.update_route(self._session_route_payload, 0.0, 0, "Boarding", "ready")
        self._refresh_journey_map_dialog()
        if hasattr(self, "journey_widget"):
            self.journey_widget.hide()

        if was_running:
            # Build a quick session summary for the exit dialog
            session_summary: dict = {}
            try:
                session_seconds = max(
                    int(time.time() - (self.session_started_at or time.time())),
                    int(self.session_time_seconds),
                )
                avg_score = (
                    float(sum(self.raw_score_samples) / len(self.raw_score_samples))
                    if self.raw_score_samples else float(self.current_score)
                )
                session_summary = {
                    "session_seconds": session_seconds,
                    "focus_seconds": float(self.raw_focus_time),
                    "avg_score": avg_score,
                    "distraction_count": int(self.distraction_count),
                }
                # Add next-session suggestion from analytics
                try:
                    rec = self.analytics_store.get_recommendation(
                        self.profile_name,
                        default_work=int(self.config.get("break_interval_minutes", 25)),
                        default_break=int(self.config.get("break_duration_minutes", 5)),
                    )
                    w = int(rec.get("work_minutes", 25))
                    b = int(rec.get("break_minutes", 5))
                    session_summary["next_session_suggestion"] = f"Gợi ý phiên sau: làm việc {w}p, nghỉ {b}p."
                except Exception:
                    pass
            except Exception as exc:
                logger.debug("Failed to build session summary for exit dialog: %s", exc)

            if bool(self.config.get("session_exit_feedback_enabled", True)):
                try:
                    exit_dialog = SessionExitDialog(
                        config=self.config,
                        session_summary=session_summary,
                        parent=self,
                    )
                    if exit_dialog.exec() == QDialog.DialogCode.Accepted:
                        self._session_exit_payload = exit_dialog.get_payload()
                    else:
                        self._session_exit_payload = {
                            "reason": "skipped",
                            "reason_label": "Bỏ qua",
                            "focus_rating": None,
                            "note": "",
                        }
                except Exception as exc:
                    logger.debug("Session exit dialog failed: %s", exc)
                    self._session_exit_payload = {"reason": "dialog_error", "error": str(exc)}

            self._persist_session_analytics()

            # Show habit report after persisting
            self._show_habit_report_if_ready()

        logger.info("Focus tracking stopped")

    @pyqtSlot()
    def _process_frame(self):
        """Process a single camera frame."""
        if not self.camera_running:
            return

        frame = self.camera.get_frame()
        if frame is None:
            self._update_live_status(face_detected=False, lighting="Unknown")
            return

        timestamp = time.time()
        if bool(getattr(self, "_journey_waiting_for_boarding", False)):
            if timestamp - float(getattr(self, "_last_boarding_preview_at", 0.0) or 0.0) >= 0.125:
                self._last_boarding_preview_at = timestamp
                self.camera_widget.update_frame(frame)
            return

        timestamp_ms = int(timestamp * 1000)
        try:
            target_fps = float(self.config.get("vision_target_fps", 8) or 8)
        except (TypeError, ValueError):
            target_fps = 8.0
        target_fps = max(4.0, min(12.0, target_fps))
        target_interval = 1.0 / target_fps

        if (
            self._last_vision_process_at > 0.0
            and timestamp - self._last_vision_process_at < target_interval
        ):
            self._vision_skipped_frames += 1
            if not bool(self.config.get("show_overlay", False)):
                self.camera_widget.update_frame(frame)
            return

        self._last_vision_process_at = timestamp
        elapsed_seconds = self._compute_frame_elapsed_seconds(timestamp)
        perf_start = time.perf_counter()

        try:
            # Process through unified vision pipeline
            vision_result = self.vision_pipeline.process(frame, timestamp_ms)
            quality = vision_result.quality
            quality_summary = self.vision_pipeline.get_quality_summary(quality)
            vision_confidence = float(quality.overall_confidence)

            calibration_result = self.vision_pipeline.consume_latest_calibration_result()
            if calibration_result is not None:
                if calibration_result.success:
                    self._persist_profile_vision_calibration(calibration_result.calibration.to_dict())
                    logger.info(
                        "Vision calibration updated for profile '%s' with %s samples",
                        self._get_profile_name(),
                        calibration_result.sample_count,
                    )
                else:
                    logger.info("Vision calibration skipped: %s", calibration_result.message)

            # Extract features from vision result
            face_detected = vision_result.face_detected
            self._session_total_frames += 1
            if face_detected:
                self._session_face_detected_frames += 1

            head_pitch, head_yaw, head_roll = None, None, None
            ear_avg, is_eye_closed, blink_detected = None, False, False
            eye_look_down, eye_look_up = None, None
            eye_closure_level = None
            perclos_ratio = None
            hand_present, hand_write_score, hand_region = False, 0.0, "none"
            hand_writing_confidence = 0.0

            if vision_result.head_pose:
                head_pitch = vision_result.head_pose.pitch
                head_yaw = vision_result.head_pose.yaw
                head_roll = vision_result.head_pose.roll

            if vision_result.eye_metrics:
                ear_avg = vision_result.eye_metrics.avg_ear
                is_eye_closed = vision_result.eye_metrics.is_closed
                blink_detected = vision_result.eye_metrics.blink_detected
                eye_look_down = vision_result.eye_metrics.look_down
                eye_look_up = vision_result.eye_metrics.look_up
                eye_closure_level = (
                    (float(vision_result.eye_metrics.left_closure) + float(vision_result.eye_metrics.right_closure))
                    / 2.0
                )
                perclos_ratio = float(vision_result.eye_metrics.perclos_ratio)

            if vision_result.hand_metrics:
                hand_present = vision_result.hand_metrics.detected
                hand_write_score = vision_result.hand_metrics.write_score
                hand_region = vision_result.hand_metrics.region
                hand_writing_confidence = float(vision_result.hand_metrics.writing_confidence)

            phone_present = False
            phone_confidence = None
            if self.phone_detector is not None:
                phone_state = self.phone_detector.process(frame, timestamp_ms=timestamp_ms)
                phone_present = bool(phone_state.phone_present)
                phone_confidence = float(phone_state.phone_confidence)

            lighting_quality = quality.lighting_quality or self._estimate_lighting_quality(frame)

            # System idle
            idle_seconds = get_idle_seconds()

            # Extra confidence fields for FocusEngine gating
            head_pose_confidence = None
            eye_confidence_val = None
            face_tracking_conf = None
            if vision_result.head_pose is not None:
                head_pose_confidence = float(vision_result.head_pose.confidence)
            if vision_result.eye_metrics is not None:
                eye_confidence_val = float(vision_result.eye_metrics.eye_confidence)
                hand_writing_confidence = float(vision_result.hand_metrics.writing_confidence) if vision_result.hand_metrics else hand_writing_confidence
            face_tracking_conf = float(quality.face_tracking_confidence)
            quality_warnings_tuple = tuple(quality.quality_warnings)

            # Create frame features
            features = FrameFeatures(
                timestamp=timestamp,
                face_detected=face_detected,
                head_pitch=head_pitch,
                head_yaw=head_yaw,
                head_roll=head_roll,
                ear_avg=ear_avg,
                is_eye_closed=is_eye_closed,
                blink_detected=blink_detected,
                hand_present=hand_present,
                hand_write_score=hand_write_score,
                hand_region=hand_region,
                phone_present=phone_present,
                idle_seconds=idle_seconds,
                eye_look_down=eye_look_down,
                eye_look_up=eye_look_up,
                eye_closure_level=eye_closure_level,
                perclos_ratio=perclos_ratio,
                phone_confidence=phone_confidence,
                vision_confidence=vision_confidence,
                hand_writing_confidence=hand_writing_confidence,
                head_pose_confidence=head_pose_confidence,
                eye_confidence=eye_confidence_val,
                face_tracking_confidence=face_tracking_conf,
                quality_warnings=quality_warnings_tuple,
            )

            self._track_session_eye_metrics(features, elapsed_seconds)

            # Process through engine
            state = self.engine.process_frame(features)
            score = self.engine.focus_score
            state_info = self.engine.get_state_info()
            self._record_initial_session_baseline_sample(
                features=features,
                score=score,
                state=state,
                state_info=state_info,
                lighting=lighting_quality,
                frame_timestamp=timestamp,
            )
            # Update UI
            display_state, display_confidence, display_reason = self._update_state(
                state,
                score,
                state_info,
                frame_timestamp=timestamp,
                elapsed_seconds=elapsed_seconds,
            )
            self._record_validation_state_prediction(
                raw_state=state,
                display_state=display_state,
                score=score,
                confidence=display_confidence,
                reason=display_reason,
                features=features,
                lighting=lighting_quality,
                state_info=state_info,
                timestamp=timestamp,
            )

            # Draw overlays on frame
            display_frame = self._draw_overlays(
                frame,
                features,
                display_state,
                display_confidence,
                display_reason,
            )
            self.camera_widget.update_frame(display_frame)
            self._update_live_status(
                face_detected=face_detected,
                lighting=lighting_quality,
                quality_summary=quality_summary,
            )

            # Check for break suggestion
            self._check_break_suggestion(display_state)
            self._record_vision_performance((time.perf_counter() - perf_start) * 1000.0)

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self._update_live_status(face_detected=False, lighting="Unknown")

    def _record_vision_performance(self, processing_ms: float) -> None:
        """Track light performance telemetry without spamming the log."""
        try:
            value = max(0.0, float(processing_ms))
        except (TypeError, ValueError):
            return

        if self._vision_processing_ema_ms <= 0.0:
            self._vision_processing_ema_ms = value
        else:
            self._vision_processing_ema_ms = (0.88 * self._vision_processing_ema_ms) + (0.12 * value)
        self._vision_effective_frames += 1

        if not bool(self.config.get("enable_performance_logging", False)):
            return

        now = time.time()
        if now - self._last_perf_log_at < 20.0:
            return

        self._last_perf_log_at = now
        logger.info(
            "Vision perf: ema=%.1fms processed=%s skipped=%s target_fps=%s",
            self._vision_processing_ema_ms,
            self._vision_effective_frames,
            self._vision_skipped_frames,
            self.config.get("vision_target_fps", 8),
        )

    def _draw_overlays(self, frame: np.ndarray,
                       features: FrameFeatures,
                       state: FocusState,
                       state_confidence: float = 0.0,
                       state_reason: str = "") -> np.ndarray:
        """Draw a compact, non-technical live guidance overlay."""
        # User preference: keep camera feed clean without top-left text panel.
        if not bool(self.config.get("show_overlay", False)):
            return frame

        display = frame.copy()

        if self._is_initial_analysis_phase():
            state = FocusState.UNCERTAIN
            state_confidence = 0.0
            state_reason = "Đang lấy mốc đầu phiên"

        # Draw subtle state-colored frame
        color = STATE_COLORS.get(state, "#607D8B")
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        cv2.rectangle(display, (0, 0), (display.shape[1]-1, display.shape[0]-1),
                      (b, g, r), 2)

        work_minutes, _ = self._current_schedule_minutes()
        break_interval_seconds = max(60, int(float(work_minutes) * 60))
        cycle_percent = int(min(100.0, (self.continuous_focus_time / break_interval_seconds) * 100.0))

        if self._is_distraction_state(state):
            recommendation = "Goi y: Nghi ngan 3-5 phut"
        elif cycle_percent >= 75:
            recommendation = "Goi y: Chuan bi nghi sau phan hien tai"
        else:
            recommendation = "Goi y: Co the tiep tuc"

        confidence_text = f"Do tin cay: {state_confidence:.0%}" if state_confidence > 0 else "Do tin cay: Dang cap nhat"
        info_lines = [
            f"Trang thai: {OVERLAY_STATE_NAMES.get(state, state.name)}",
            confidence_text,
            recommendation,
        ]

        panel_x = 8
        panel_y = 8
        line_height = 24
        panel_height = 12 + line_height * (len(info_lines) + (1 if state_reason else 0))
        panel_width = min(display.shape[1] - 16, 430)

        # Soft dark panel for readable calm-tech labels.
        overlay = display.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (17, 24, 36),
            -1,
        )
        cv2.addWeighted(overlay, 0.82, display, 0.18, 0, display)

        y = panel_y + 24

        for line in info_lines:
            cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (234, 243, 255), 1, cv2.LINE_AA)
            y += 22

        if state_reason:
            reason_text = state_reason[:70]
            cv2.putText(display, f"Ghi chu: {reason_text}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (180, 196, 218), 1, cv2.LINE_AA)

        return display

    @staticmethod
    def _badge_text_color(hex_color: str) -> str:
        """Choose a readable text color for state badge backgrounds."""
        try:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
        except (TypeError, ValueError):
            return "#ffffff"

        luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
        return "#0b1120" if luminance > 165 else "#ffffff"

    def _update_state_badge(self, state: FocusState, confidence: float, reason: str):
        """Update score status chip and short insight text."""
        trend_delta = self._compute_focus_trend_delta()
        score_now = float(self.current_score)
        summary = getattr(self, "_last_behavior_summary", {}) or {}
        status_modifier = str(summary.get("status_modifier", "") or "")
        is_dark = str(self.config.get("theme_mode", "dark")).strip().lower() != "light"

        if is_dark:
            palette = {
                "warmup": ("rgba(127, 147, 170, 0.16)", "#d9e5f5", "rgba(127, 147, 170, 0.28)"),
                "hold": ("rgba(158, 209, 255, 0.16)", "#d7edff", "rgba(158, 209, 255, 0.30)"),
                "idle": ("rgba(127, 147, 170, 0.16)", "#d9e5f5", "rgba(127, 147, 170, 0.28)"),
                "break": ("rgba(239, 157, 149, 0.18)", "#ffd6d0", "rgba(239, 157, 149, 0.30)"),
                "watch": ("rgba(239, 189, 120, 0.18)", "#ffe3b5", "rgba(239, 189, 120, 0.30)"),
                "good": ("rgba(89, 213, 192, 0.18)", "#c6f8ee", "rgba(89, 213, 192, 0.30)"),
            }
        else:
            palette = {
                "warmup": ("rgba(103, 127, 154, 0.14)", "#2f4a64", "rgba(103, 127, 154, 0.30)"),
                "hold": ("rgba(98, 161, 214, 0.14)", "#1e5b84", "rgba(98, 161, 214, 0.30)"),
                "idle": ("rgba(103, 127, 154, 0.14)", "#2f4a64", "rgba(103, 127, 154, 0.30)"),
                "break": ("rgba(214, 103, 91, 0.16)", "#8f3b32", "rgba(214, 103, 91, 0.32)"),
                "watch": ("rgba(201, 144, 63, 0.16)", "#7a5520", "rgba(201, 144, 63, 0.32)"),
                "good": ("rgba(41, 151, 136, 0.16)", "#1d6c60", "rgba(41, 151, 136, 0.32)"),
            }

        if self._is_initial_analysis_phase():
            chip_bg, chip_fg, chip_border = palette["warmup"]
            if bool(getattr(self, "_journey_waiting_for_boarding", False)):
                chip_text = "Boarding"
                hint_text = "Xé vé trong Focus Journey để bắt đầu hiệu chỉnh."
            else:
                chip_text = "Lấy mốc đầu phiên"
                seconds_left = self._analysis_seconds_left()
                hint_text = f"Giữ tư thế làm việc tự nhiên trong {seconds_left}s để hệ thống lấy mốc đầu phiên."
        elif (
            "đang giữ trạng thái ổn định" in reason.lower()
            or "tín hiệu tạm thời chưa rõ" in reason.lower()
        ):
            chip_text = "Giữ ổn định"
            chip_bg, chip_fg, chip_border = palette["hold"]
            hint_text = "Tín hiệu tạm thời chưa rõ, hệ thống đang giữ trạng thái ổn định để tránh nhảy sai."
        elif not self.camera_running or (state == FocusState.UNCERTAIN and len(self.focus_trend_samples) < 10):
            chip_text = "Chưa phân tích"
            chip_bg, chip_fg, chip_border = palette["idle"]
            hint_text = "Đang thu thập dữ liệu để đánh giá xu hướng làm việc."
        elif state == FocusState.UNCERTAIN and self.engine is not None and (
            "low_vision_confidence" in getattr(self.engine, "_last_uncertain_reason", "")
            or "calibration_missing" in getattr(self.engine, "_last_uncertain_reason", "")
        ):
            chip_text = "Camera chưa rõ"
            chip_bg, chip_fg, chip_border = palette["watch"]
            if "calibration_missing" in getattr(self.engine, "_last_uncertain_reason", ""):
                hint_text = "Dữ liệu camera chưa đủ tin cậy. Thử hiệu chỉnh camera (Settings → Vision)."
            else:
                hint_text = "Dữ liệu camera chưa đủ tin cậy. Kiểm tra ánh sáng hoặc vị trí camera."
        elif status_modifier == "fatigued_but_working":
            chip_text = "Làm việc • hơi mệt"
            chip_bg, chip_fg, chip_border = palette["watch"]
            hint_text = "Đang làm việc nhưng có dấu hiệu mệt. Nên nghỉ mắt 1-2 phút hoặc thở ngắn nếu kéo dài."
        elif status_modifier == "possible_passive_attention":
            chip_text = "Đọc thụ động?"
            chip_bg, chip_fg, chip_border = palette["watch"]
            hint_text = "Có thể đang đọc thụ động. Hãy đặt một mục tiêu nhỏ 5 phút để xác nhận nhịp làm việc."
        elif status_modifier == "low_confidence":
            chip_text = "Camera chưa rõ"
            chip_bg, chip_fg, chip_border = palette["watch"]
            hint_text = "Dữ liệu camera chưa đủ tin cậy. Điều chỉnh ánh sáng hoặc góc camera trước khi kết luận."
        elif state == FocusState.PHONE_DISTRACTION:
            chip_text = "Lệch khỏi nhiệm vụ"
            chip_bg, chip_fg, chip_border = palette["break"]
            hint_text = "Hệ thống ghi nhận dấu hiệu có thể lệch khỏi tác vụ."
        elif state == FocusState.DROWSY_FATIGUE:
            chip_text = "Có dấu hiệu mệt"
            chip_bg, chip_fg, chip_border = palette["watch"]
            hint_text = "Có thể cân nhắc nghỉ mắt ngắn hoặc tạm nghỉ phục hồi."
        elif state == FocusState.UNCERTAIN:
            chip_text = "Chưa đủ tin cậy"
            chip_bg, chip_fg, chip_border = palette["watch"]
            hint_text = "Cần thêm dữ liệu hoặc điều chỉnh ánh sáng/góc camera."
        elif score_now < 58:
            chip_text = "Có dấu hiệu mệt"
            chip_bg, chip_fg, chip_border = palette["break"]
            hint_text = "Có thể cân nhắc nghỉ mắt ngắn hoặc tạm nghỉ phục hồi."
        elif trend_delta <= -3.5 or score_now < 76:
            chip_text = "Lệch nhịp nhẹ"
            chip_bg, chip_fg, chip_border = palette["watch"]
            delta_points = max(1, int(abs(trend_delta)))
            hint_text = f"Giảm {delta_points} điểm so với xu hướng gần nhất."
        elif state == FocusState.ON_SCREEN_READING:
            chip_text = "Tín hiệu làm việc ổn định"
            chip_bg, chip_fg, chip_border = palette["good"]
            hint_text = "Dựa trên tín hiệu hành vi hiện tại, chưa thấy dấu hiệu rõ của mệt mỏi hoặc lệch nhiệm vụ."
        elif state == FocusState.OFFSCREEN_WRITING:
            chip_text = "Làm việc ổn định"
            chip_bg, chip_fg, chip_border = palette["good"]
            hint_text = "Dựa trên tín hiệu hành vi hiện tại, hoạt động ghi chép/làm việc ngoài màn hình đang ổn định."
        else:
            chip_text = "Làm việc ổn định"
            chip_bg, chip_fg, chip_border = palette["good"]
            stable_minutes = max(1, int(self.continuous_focus_time // 60))
            hint_text = f"Tín hiệu ổn định trong {stable_minutes} phút gần đây."

        self.state_badge.setText(chip_text)
        self.state_badge.setStyleSheet(
            "border-radius: 999px; padding: 5px 12px; font-weight: 650;"
            f"background:{chip_bg}; color:{chip_fg}; border:1px solid {chip_border};"
        )

        state_name = STATE_NAMES.get(state, state.name)
        self.state_badge.setToolTip(hint_text or reason or state_name)

    def _append_session_state_segment(
        self,
        state: FocusState,
        seconds: float,
        uncertain_reason_type: str = "",
    ) -> None:
        """Append compact contiguous state segments for analytics cleaning."""
        if seconds <= 0.0:
            return

        reason_type = uncertain_reason_type.strip().lower() if state == FocusState.UNCERTAIN else ""
        if (
            self._session_state_segments
            and self._session_state_segments[-1].get("state") == state.name
            and str(self._session_state_segments[-1].get("uncertain_reason_type", "")).strip().lower() == reason_type
        ):
            self._session_state_segments[-1]["seconds"] = (
                float(self._session_state_segments[-1].get("seconds", 0.0) or 0.0) + seconds
            )
            return

        self._session_state_segments.append(
            {
                "state": state.name,
                "seconds": float(seconds),
                "uncertain_reason_type": reason_type,
            }
        )

    def _update_state(
        self,
        state: FocusState,
        score: float,
        state_info: Dict[str, Any],
        frame_timestamp: float,
        elapsed_seconds: float,
    ) -> tuple[FocusState, float, str]:
        """Update UI with display-stabilized state while preserving raw analytics signals."""
        raw_confidence = float(state_info.get("confidence", 0.0) or 0.0)
        raw_reason = str(state_info.get("reason", "") or "")
        uncertain_reason_type = str(state_info.get("uncertain_reason_type", "") or "").strip().lower()
        focused_hold_active = bool(state_info.get("focused_hold_active", False))
        uncertain_clean_candidate = bool(state_info.get("uncertain_clean_candidate", False))
        raw_summary = state_info.get("behavior_summary", {})
        behavior_summary = raw_summary if isinstance(raw_summary, dict) else {}
        self._last_behavior_summary = dict(behavior_summary)
        status_modifier = str(behavior_summary.get("status_modifier", "") or "")
        try:
            distraction_risk = float(behavior_summary.get("distraction_risk", 0.0) or 0.0)
        except (TypeError, ValueError):
            distraction_risk = 0.0
        try:
            uncertain_grace_remaining = max(0.0, float(state_info.get("uncertain_grace_remaining", 0.0) or 0.0))
        except (TypeError, ValueError):
            uncertain_grace_remaining = 0.0

        in_warmup = self._is_initial_analysis_phase()
        now_ts = frame_timestamp
        if not in_warmup and not bool(getattr(self, "_journey_calibration_reset_done", True)):
            self._finalize_initial_session_baseline(frame_timestamp)
            self._journey_calibration_reset_done = True
            self._session_total_frames = 0
            self._session_face_detected_frames = 0
            self._last_state_frame_timestamp = frame_timestamp
            elapsed_seconds = 0.0

        effective_state = FocusState.UNCERTAIN if in_warmup else state
        effective_score = self._compute_display_score(score)
        effective_confidence = 0.0 if in_warmup else raw_confidence
        effective_reason = "Đang lấy mốc đầu phiên" if in_warmup else raw_reason

        if not in_warmup:
            if state in (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING):
                self._display_focused_state = state
                self._display_hold_until = now_ts + self._display_uncertain_hold_seconds
            elif (
                state in (FocusState.PHONE_DISTRACTION, FocusState.DROWSY_FATIGUE, FocusState.AWAY)
                and status_modifier != "fatigued_but_working"
            ):
                self._display_focused_state = None
                self._display_hold_until = now_ts

            if state == FocusState.UNCERTAIN:
                keep_focused_display = (
                    self._display_focused_state is not None
                    and (
                        uncertain_reason_type == "measurement_noise"
                        or uncertain_clean_candidate
                        or focused_hold_active
                    )
                    and (
                        now_ts <= self._display_hold_until
                        or uncertain_grace_remaining > 0.0
                        or focused_hold_active
                    )
                )

                if keep_focused_display:
                    effective_state = self._display_focused_state
                    effective_confidence = max(0.34, min(0.68, raw_confidence))
                    effective_reason = "Tín hiệu tạm thời chưa rõ, đang giữ trạng thái ổn định"

            strong_phone_evidence = state == FocusState.PHONE_DISTRACTION and distraction_risk >= 0.66
            if status_modifier == "fatigued_but_working" and not strong_phone_evidence:
                effective_state = self._display_focused_state or FocusState.ON_SCREEN_READING
                effective_score = max(effective_score, 66.0)
                effective_confidence = max(effective_confidence, min(0.82, raw_confidence + 0.08))
                effective_reason = "Đang làm việc nhưng có dấu hiệu mệt"
            elif status_modifier == "possible_passive_attention" and not strong_phone_evidence:
                effective_state = FocusState.UNCERTAIN if state == FocusState.ON_SCREEN_READING else effective_state
                effective_confidence = min(effective_confidence, 0.62)
                effective_reason = "Có thể đang đọc thụ động, cần thêm bằng chứng làm việc"
            elif status_modifier == "low_confidence" and not strong_phone_evidence:
                effective_state = FocusState.UNCERTAIN
                effective_confidence = min(effective_confidence, 0.42)
                effective_reason = "Dữ liệu camera chưa đủ tin cậy"

        self.display_state = effective_state

        # Track state changes
        state_changed = effective_state != self.current_state
        if state_changed:
            previous_state = self.current_state

            if effective_state == FocusState.PHONE_DISTRACTION:
                self.distraction_count += 1
            self.state_changed.emit(effective_state)

            if (
                not in_warmup
                and
                self.config.get("auto_break_on_distraction", True)
                and self._is_focused_state(previous_state)
                and self._is_distraction_state(effective_state)
            ):
                self._schedule_distraction_break(effective_state)

            if (
                not in_warmup
                and effective_state == FocusState.DROWSY_FATIGUE
                and self._session_fatigue_onset_seconds is None
                and self.session_started_at is not None
            ):
                self._session_fatigue_onset_seconds = max(0.0, time.time() - self.session_started_at)

        self.current_state = effective_state
        self.current_score = effective_score
        self._update_state_badge(effective_state, effective_confidence, effective_reason)

        # Update widgets
        self.score_widget.set_score(effective_score, effective_state)
        self._update_score_breakdown()
        if not in_warmup:
            self.score_samples.append(effective_score)

        frame_seconds = max(0.0, float(elapsed_seconds))
        if not in_warmup:
            raw_score_value = max(0.0, min(100.0, float(score)))
            self.raw_score_samples.append(raw_score_value)
            if self._session_focus_score_start is None:
                self._session_focus_score_start = raw_score_value
            self._session_focus_score_end = raw_score_value

            self.raw_state_time_by_state[state.name] = (
                self.raw_state_time_by_state.get(state.name, 0.0) + frame_seconds
            )
            if state in (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING):
                self.raw_focus_time += frame_seconds

            if state == FocusState.UNCERTAIN:
                if uncertain_reason_type == "measurement_noise":
                    self._session_uncertain_noise_seconds += frame_seconds
                else:
                    self._session_uncertain_behavioral_seconds += frame_seconds
                if uncertain_clean_candidate:
                    self._session_uncertain_clean_candidate_seconds += frame_seconds

            self._append_session_state_segment(
                state,
                frame_seconds,
                uncertain_reason_type=uncertain_reason_type,
            )

            # Track focus time only after initial calibration.
            self.state_time_by_state[effective_state.name] = (
                self.state_time_by_state.get(effective_state.name, 0.0) + frame_seconds
            )

            if effective_state in (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING):
                self.focus_time += frame_seconds
                self.continuous_focus_time += frame_seconds
            else:
                self.continuous_focus_time = 0
        else:
            self.continuous_focus_time = 0

        if state_changed:
            self._refresh_focus_guidance()

        if not in_warmup:
            self._maybe_show_behavior_checkin(behavior_summary, now_ts)

        if not in_warmup and self.camera_running:
            alert_state = effective_state
            alert_score = effective_score
            alert_confidence = effective_confidence
            alert_reason = effective_reason

            # Zalo alerts should reflect the engine's stable risk signal even
            # when the UI display layer softens a noisy/low-confidence frame to
            # UNCERTAIN. This keeps realtime alerts aligned with FocusEngine
            # state-transition logs such as AWAY and DROWSY_FATIGUE.
            if (
                state in (FocusState.PHONE_DISTRACTION, FocusState.DROWSY_FATIGUE, FocusState.AWAY)
                and status_modifier != "fatigued_but_working"
            ):
                alert_state = state
                alert_score = max(0.0, min(100.0, float(score)))
                alert_confidence = raw_confidence
                alert_reason = raw_reason

                if alert_state != effective_state:
                    logger.debug(
                        "Zalo alert using raw state: raw=%s effective=%s reason=%s",
                        alert_state.name,
                        effective_state.name,
                        alert_reason[:80],
                    )

            alert_event = self.zalo_alert_manager.handle_state_update(
                alert_state,
                score=alert_score,
                confidence=alert_confidence,
                reason=alert_reason,
                timestamp=now_ts,
                recommendation=self._last_recommendation,
                in_warmup=in_warmup,
            )
            if alert_event is not None:
                if alert_event.success:
                    logger.debug("Sent Zalo state alert: %s", alert_event.alert_key)
                else:
                    logger.warning("Zalo state alert skipped/failed: %s", alert_event.detail)

        self.score_changed.emit(effective_score)
        return effective_state, effective_confidence, effective_reason

    def _record_validation_state_prediction(
        self,
        *,
        raw_state: FocusState,
        display_state: FocusState,
        score: float,
        confidence: float,
        reason: str,
        features: FrameFeatures,
        lighting: str,
        state_info: Dict[str, Any],
        timestamp: float,
    ) -> None:
        """Persist low-frequency app-state samples for observer-label evaluation."""
        if not bool(self.config.get("enable_validation_logging", True)):
            return
        if self._is_initial_analysis_phase():
            return
        if timestamp - float(getattr(self, "_last_validation_prediction_at", 0.0) or 0.0) < 1.0:
            return

        self._last_validation_prediction_at = timestamp
        summary = state_info.get("behavior_summary", {}) if isinstance(state_info, dict) else {}
        if not isinstance(summary, dict):
            summary = {}
        fatigue_index = float(summary.get("fatigue_index", 0.0) or 0.0)
        distraction_risk = float(summary.get("distraction_risk", 0.0) or 0.0)
        baseline_fields = self._initial_baseline_delta_fields(
            work_readiness=float(self.current_score),
            fatigue_index=fatigue_index,
            distraction_risk=distraction_risk,
        )
        try:
            record = {
                    "timestamp": timestamp,
                    "session_id": str(getattr(self, "_validation_session_id", "") or getattr(self, "_journey_session_id", "")),
                    "profile_name": str(self.profile_name or self._get_profile_name()),
                    "app_state": display_state.name,
                    "raw_state": raw_state.name,
                    "display_state": display_state.name,
                    "confidence": round(float(confidence or 0.0), 4),
                    "work_readiness": round(float(self.current_score), 3),
                    "raw_work_readiness": round(float(score or 0.0), 3),
                    "face_present": bool(features.face_detected),
                    "camera_quality": str(lighting or ""),
                    "status_modifier": str(summary.get("status_modifier", "") or ""),
                    "reason": str(reason or "")[:240],
                    "elapsed_session_seconds": int(self.session_time_seconds),
                }
            record.update(baseline_fields)
            self.validation_store.append_state_prediction(record)
        except Exception as exc:
            logger.debug("Failed to record validation state prediction: %s", exc)

    @staticmethod
    def _is_focused_state(state: FocusState) -> bool:
        return state in (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING)

    @staticmethod
    def _is_distraction_state(state: FocusState) -> bool:
        return state == FocusState.PHONE_DISTRACTION

    @pyqtSlot()
    def _retry_camera_start(self) -> None:
        """Handle retry action from camera empty state."""
        if self.camera_running:
            return

        self.btn_start.setChecked(True)
        self._start_tracking()
        if not self.camera_running:
            self.btn_start.setChecked(False)

    @staticmethod
    def _estimate_lighting_quality(frame: np.ndarray) -> str:
        """Estimate lighting quality using grayscale brightness."""
        if frame is None or frame.size == 0:
            return "Unknown"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        if brightness < 60:
            return "Low"
        if brightness > 185:
            return "Strong"
        return "Good"

    @staticmethod
    def _format_lighting_status(lighting: str) -> str:
        """Keep the lighting chip focused only on illumination quality."""
        normalized = str(lighting or "Unknown").strip().lower()
        if any(token in normalized for token in ("low", "yeu", "yếu")):
            return "Ánh sáng yếu"
        if any(token in normalized for token in ("strong", "gat", "gắt")):
            return "Ánh sáng gắt"
        if any(token in normalized for token in ("good", "tot", "tốt")):
            return "Ánh sáng tốt"
        return "Chưa rõ"

    def _update_live_status(
        self,
        face_detected: Optional[bool],
        lighting: str,
        quality_summary: Optional[str] = None,
    ) -> None:
        """Refresh the live status strip under the camera panel."""
        if not hasattr(self, "live_status_strip"):
            return

        if self.camera_running:
            stream_status = "Live"
        elif self.vision_available:
            stream_status = "Paused"
        else:
            stream_status = "Disconnected"

        if face_detected is True:
            face_status = "Face detected"
        elif face_detected is False:
            face_status = "No face"
        else:
            face_status = "Waiting"

        lighting_text = self._format_lighting_status(lighting)
        self.live_status_strip.set_status(
            stream=stream_status,
            face=face_status,
            lighting=lighting_text,
        )

    def _can_trigger_distraction_break(self) -> bool:
        if not self.config.get("enable_break_reminders", True):
            return False

        if self._break_dialog_open:
            return False

        cooldown_minutes = int(self.config.get("distraction_break_cooldown_minutes", 15))
        cooldown_seconds = max(0, cooldown_minutes) * 60
        elapsed = time.time() - self._last_distraction_break_time
        return elapsed >= cooldown_seconds

    def _schedule_distraction_break(self, state: FocusState) -> None:
        if self._auto_break_pending:
            return

        self._auto_break_pending = True
        QTimer.singleShot(0, lambda s=state: self._trigger_distraction_break(s))

    def _trigger_distraction_break(self, state: FocusState) -> None:
        self._auto_break_pending = False

        if not self.camera_running:
            return

        if not self._can_trigger_distraction_break():
            return

        self._last_distraction_break_time = time.time()
        self.break_suggested.emit()

        _, break_minutes = self._current_schedule_minutes()
        state_name = STATE_NAMES.get(state, state.name)
        NoticeDialog.info(
            self,
            "Nhắc nghỉ phục hồi",
            (
                f"Dựa trên tín hiệu hành vi hiện tại, có dấu hiệu lệch khỏi nhiệm vụ ({state_name}).\n"
                f"Hãy nghỉ {break_minutes} phút và thực hiện bài nghỉ ngắn."
            ),
            config=self.config,
            button_text="Bắt đầu nghỉ",
        )

        self._take_break(auto_triggered=True)

    def _record_focus_sample(self, score: float) -> None:
        """Store low-frequency score samples for trend guidance UI."""
        if self._is_initial_analysis_phase():
            return

        try:
            numeric = max(0.0, min(100.0, float(score)))
        except (TypeError, ValueError):
            return

        self.focus_trend_samples.append(numeric)
        if len(self.focus_trend_samples) > 480:
            del self.focus_trend_samples[:-480]

    def _compute_focus_trend_delta(self) -> float:
        """Return trend delta between the recent and previous score windows."""
        recent = self.focus_trend_samples[-90:]
        if len(recent) < 10:
            return 0.0

        pivot = len(recent) // 2
        first = recent[:pivot]
        second = recent[pivot:]
        if not first or not second:
            return 0.0

        first_avg = sum(first) / len(first)
        second_avg = sum(second) / len(second)
        return second_avg - first_avg

    def _refresh_focus_guidance(self) -> None:
        """Update recommendation and insight cards."""
        if not hasattr(self, "guidance_widget"):
            return

        # In deep focus mode, guidance widget is hidden — only update trend widget
        deep_focus = bool(getattr(self, "_deep_focus_active", False))
        is_dark = str(self.config.get("theme_mode", "dark")).strip().lower() != "light"

        if self.camera_running and self._is_initial_analysis_phase():
            seconds_left = self._analysis_seconds_left()
            if not deep_focus:
                self.guidance_widget.set_guidance(
                    mode="good",
                    decision="Đang lấy mốc đầu phiên",
                    detail=f"Dựa trên tín hiệu hành vi hiện tại, hệ thống đang lấy mốc đầu phiên trong {seconds_left}s và chưa dùng dữ liệu này để kết luận.",
                    state_text="Đang lấy mốc đầu phiên",
                )
            if hasattr(self, "trend_widget") and not deep_focus:
                self.trend_widget.set_insight(
                    trend_text="Mốc đầu phiên",
                    trend_color="#9fd6ff" if is_dark else "#2f587f",
                    cycle_percent=0,
                    trend_values=[],
                )
            return

        trend_delta = self._compute_focus_trend_delta()
        if trend_delta <= -7:
            trend_text = "Đang giảm rõ"
            trend_color = "#f6c177" if is_dark else "#8a5d24"
        elif trend_delta <= -3:
            trend_text = "Giảm nhẹ"
            trend_color = "#ffde95" if is_dark else "#9b6f2d"
        elif trend_delta >= 5:
            trend_text = "Đang phục hồi"
            trend_color = "#8ff5dd" if is_dark else "#0f7466"
        else:
            trend_text = "Ổn định"
            trend_color = "#9fd6ff" if is_dark else "#2f587f"

        work_minutes, _ = self._current_schedule_minutes()
        break_interval_seconds = max(60, int(float(work_minutes) * 60))
        elapsed_since_break = 0.0
        if self.camera_running:
            elapsed_since_break = max(0.0, time.time() - self.last_break_time)

        cycle_percent = int(min(100.0, (elapsed_since_break / break_interval_seconds) * 100.0))
        score_now = float(self.current_score)
        guidance_summary = getattr(self, "_last_behavior_summary", {}) or {}
        guidance_modifier = str(guidance_summary.get("status_modifier", "") or "")

        if not self.camera_running:
            mode = "good"
            decision = "Đang chờ tín hiệu"
            detail = "Chưa có dữ liệu để đánh giá dựa trên tín hiệu hành vi hiện tại. Nhấn Bắt đầu để hệ thống quan sát nhịp làm việc."
        elif self.current_state == FocusState.UNCERTAIN or guidance_modifier == "low_confidence":
            mode = "watch"
            decision = "Chưa đủ tin cậy"
            detail = "Cần thêm dữ liệu hoặc điều chỉnh ánh sáng/góc camera."
        elif self.current_state == FocusState.PHONE_DISTRACTION:
            mode = "break"
            decision = "Lệch khỏi nhiệm vụ"
            detail = "Hệ thống ghi nhận dấu hiệu có thể lệch khỏi tác vụ."
        elif self.current_state == FocusState.DROWSY_FATIGUE:
            mode = "watch"
            decision = "Có dấu hiệu mệt"
            detail = "Có thể cân nhắc nghỉ mắt ngắn hoặc tạm nghỉ phục hồi."
        elif guidance_modifier == "fatigued_but_working":
            mode = "watch"
            decision = "Đang làm việc nhưng có dấu hiệu mệt"
            detail = "Dựa trên tín hiệu hành vi hiện tại, đây không được xem là lệch khỏi nhiệm vụ. Nên nghỉ mắt 1-2 phút nếu cảm giác mệt kéo dài."
        elif guidance_modifier == "possible_passive_attention":
            mode = "watch"
            decision = "Có thể đang đọc thụ động"
            detail = "Dựa trên tín hiệu hành vi hiện tại, mức chủ động chưa rõ. Hãy đặt một mục tiêu nhỏ trong 5 phút để biến việc nhìn màn hình thành hành động rõ ràng hơn."
        elif cycle_percent >= 100 or (trend_delta <= -8 and score_now < 70):
            mode = "break"
            decision = "Nên nghỉ ngắn 2-3 phút"
            detail = "Dựa trên tín hiệu hành vi hiện tại, phiên làm việc đã kéo dài tương đối lâu. Nghỉ ngắn lúc này có thể giúp duy trì nhịp làm việc."
        elif cycle_percent >= 75 or trend_delta <= -4 or score_now < 70:
            mode = "watch"
            decision = "Có dấu hiệu giảm chú ý"
            detail = "Dựa trên tín hiệu hành vi hiện tại, mức sẵn sàng làm việc có dấu hiệu giảm. Có thể hoàn tất phần đang làm rồi nghỉ ngắn."
        else:
            mode = "good"
            decision = "Tín hiệu làm việc ổn định"
            detail = "Dựa trên tín hiệu hành vi hiện tại, chưa thấy dấu hiệu rõ của mệt mỏi hoặc lệch nhiệm vụ."

        # In deep focus mode: only show guidance widget when risk is high
        if deep_focus:
            high_risk = mode == "break" or (mode == "watch" and score_now < 45)
            self.guidance_widget.setVisible(high_risk)
            if high_risk:
                self.guidance_widget.set_guidance(
                    mode=mode,
                    decision=decision,
                    detail=detail,
                    state_text=STATE_NAMES.get(self.current_state, self.current_state.name),
                )
        else:
            self.guidance_widget.setVisible(True)
            self.guidance_widget.set_guidance(
                mode=mode,
                decision=decision,
                detail=detail,
                state_text=STATE_NAMES.get(self.current_state, self.current_state.name),
            )

        if hasattr(self, "trend_widget") and not deep_focus:
            self.trend_widget.set_insight(
                trend_text=trend_text,
                trend_color=trend_color,
                cycle_percent=cycle_percent,
                trend_values=self.focus_trend_samples[-90:],
            )

    def _check_break_suggestion(self, state: FocusState):
        """Check if a break should be suggested."""
        if not self.config.get("enable_break_reminders", True):
            return

        work_minutes, _ = self._current_schedule_minutes()
        break_interval = int(work_minutes) * 60
        _ = state

        if self.continuous_focus_time >= break_interval:
            alert_event = self.zalo_alert_manager.handle_break_reminder(
                focus_cycle_seconds=self.continuous_focus_time,
                break_interval_seconds=float(break_interval),
                recommendation=self._last_recommendation,
                timestamp=time.time(),
            )
            if alert_event is not None:
                if alert_event.success:
                    logger.debug("Sent Zalo break reminder alert")
                else:
                    logger.warning("Zalo break reminder skipped/failed: %s", alert_event.detail)

            self.break_suggested.emit()
            self.continuous_focus_time = 0
            self.last_break_time = time.time()

    def _effective_paused_total_seconds(self, now: Optional[float] = None) -> float:
        """Return persisted pause time plus the currently active pause, if any."""
        current = time.time() if now is None else float(now)
        paused_total = max(0.0, float(getattr(self, "_paused_total_seconds", 0.0) or 0.0))
        pause_started = float(getattr(self, "_pause_started_at", 0.0) or 0.0)
        if bool(getattr(self, "_session_paused", False)) and pause_started > 0.0:
            paused_total += max(0.0, current - pause_started)
        return paused_total

    def _today_stats_payload(self, live_session: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Build the main stats card from today's saved sessions plus live data."""
        try:
            summary = self.analytics_store.build_work_rhythm_summary(
                self.profile_name,
                live_session=live_session,
            )
            day = dict((summary.get("periods", {}) or {}).get("day", {}) or {})
            total_seconds = float(day.get("total_seconds", 0.0) or 0.0)
            focus_seconds = float(day.get("focus_seconds", 0.0) or 0.0)
            avg_score = float(day.get("avg_score", 0.0) or 0.0)
            return {
                "session_time": self._format_time(total_seconds),
                "focus_time": self._format_time(focus_seconds),
                "distraction_count": str(int(day.get("distraction_count", 0) or 0)),
                "break_count": str(int(day.get("break_count", 0) or 0)),
                "avg_score": f"{avg_score:.0f}" if total_seconds > 0 else "0",
            }
        except Exception as exc:
            logger.debug("Failed to build today's stats card: %s", exc)

        avg_score_text = "0"
        if self.score_samples:
            avg_score_text = f"{sum(self.score_samples) / len(self.score_samples):.0f}"
        elif self.camera_running and self._is_initial_analysis_phase():
            avg_score_text = "100"
        return {
            "session_time": self._format_time(self.session_time_seconds),
            "focus_time": self._format_time(self.focus_time),
            "distraction_count": str(self.distraction_count),
            "break_count": str(self.break_count),
            "avg_score": avg_score_text,
        }

    def _refresh_today_stats_card(self) -> None:
        if not hasattr(self, "stats_widget"):
            return
        self.stats_widget.update_stats(
            self._today_stats_payload(self._current_work_rhythm_live_session())
        )

    @pyqtSlot()
    def _update_stats(self):
        """Update statistics display."""
        if self.camera_running:
            now = time.time()
            if bool(getattr(self, "_journey_waiting_for_boarding", False)):
                self.session_time_seconds = 0
            elif self.session_started_at is not None:
                self.session_time_seconds = max(
                    0,
                    int(now - self.session_started_at - self._effective_paused_total_seconds(now)),
                )
            else:
                self.session_time_seconds = 0
            if (
                not getattr(self, "_session_paused", False)
                and (self.session_time_seconds > 0 or not self._is_initial_analysis_phase())
            ):
                self._record_focus_sample(self.current_score)

        self._refresh_today_stats_card()
        self._refresh_focus_guidance()
        self._update_journey_widget()

    def _current_work_rhythm_live_session(self) -> Optional[Dict[str, Any]]:
        """Return a lightweight unsaved session snapshot for the results dialog."""
        if not bool(getattr(self, "camera_running", False)):
            return None

        if self.session_started_at is not None and not bool(getattr(self, "_journey_waiting_for_boarding", False)):
            now = time.time()
            session_seconds = max(
                0,
                int(now - self.session_started_at - self._effective_paused_total_seconds(now)),
            )
        else:
            session_seconds = int(getattr(self, "session_time_seconds", 0) or 0)

        if session_seconds <= 0:
            return None

        raw_samples = list(getattr(self, "raw_score_samples", []) or [])
        display_samples = list(getattr(self, "score_samples", []) or [])
        if raw_samples:
            avg_score = float(sum(raw_samples) / len(raw_samples))
        elif display_samples:
            avg_score = float(sum(display_samples) / len(display_samples))
        else:
            avg_score = float(getattr(self, "current_score", 0.0) or 0.0)

        fatigue_onset = None
        if getattr(self, "_session_fatigue_onset_seconds", None) is not None:
            fatigue_onset = float(self._session_fatigue_onset_seconds) / 60.0

        return {
            "timestamp": int(time.time()),
            "profile_name": self.profile_name,
            "session_seconds": int(session_seconds),
            "session_seconds_cleaned": int(session_seconds),
            "focus_seconds": float(getattr(self, "raw_focus_time", getattr(self, "focus_time", 0.0)) or 0.0),
            "focus_seconds_cleaned": float(getattr(self, "raw_focus_time", getattr(self, "focus_time", 0.0)) or 0.0),
            "distraction_count": int(getattr(self, "distraction_count", 0) or 0),
            "distraction_count_cleaned": float(getattr(self, "distraction_count", 0) or 0),
            "break_count": int(getattr(self, "break_count", 0) or 0),
            "avg_score": avg_score,
            "avg_score_cleaned": avg_score,
            "fatigue_onset_minutes": fatigue_onset,
            "state_seconds": dict(getattr(self, "raw_state_time_by_state", {}) or {}),
            "state_seconds_display": dict(getattr(self, "state_time_by_state", {}) or {}),
        }

    @pyqtSlot()
    def _open_work_rhythm_report(self) -> None:
        """Open day/week/month work-rhythm results from the statistics card."""
        try:
            from .work_rhythm_dialog import WorkRhythmReportDialog

            summary = self.analytics_store.build_work_rhythm_summary(
                self.profile_name,
                live_session=self._current_work_rhythm_live_session(),
            )

            if not hasattr(self, "_work_rhythm_dialog") or self._work_rhythm_dialog is None:
                self._work_rhythm_dialog = WorkRhythmReportDialog(
                    summary=summary,
                    config=self.config,
                    parent=None,
                )
                self._work_rhythm_dialog.dismissed.connect(self._handle_work_rhythm_dismissed)

            # Hide main window when showing the dialog
            self.hide()
            self._work_rhythm_dialog.show()
            self._work_rhythm_dialog.raise_()
            self._work_rhythm_dialog.activateWindow()

        except Exception as exc:
            logger.exception("Failed to open work rhythm report: %s", exc)
            NoticeDialog.warning(
                self,
                "Nhịp làm việc",
                "Chưa thể mở báo cáo nhịp làm việc lúc này.",
                config=self.config,
            )

    @pyqtSlot()
    def _handle_work_rhythm_dismissed(self) -> None:
        """Called when the work rhythm report is closed."""
        self._work_rhythm_dialog = None
        self.show()
        self.raise_()
        self.activateWindow()

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @pyqtSlot()
    def _take_break(self, auto_triggered: bool = False):
        """Handle break button click with before/after recovery validation."""
        # Snapshot state before break for recovery validation
        self._before_break_snapshot = {
            "timestamp": time.time(),
            "work_readiness": float(self.current_score),
            "state": self.current_state.name,
            "session_seconds": int(self.session_time_seconds),
        }
        try:
            breakdown = self.engine.get_score_breakdown() if hasattr(self.engine, "get_score_breakdown") else {}
            self._before_break_snapshot["fatigue_index"] = float(breakdown.get("fatigue", 0.0) or 0.0)
            self._before_break_snapshot["distraction_risk"] = float(breakdown.get("distraction", 0.0) or 0.0)
        except Exception:
            pass
        self._before_break_snapshot.update(
            self._initial_baseline_delta_fields(
                work_readiness=float(self._before_break_snapshot.get("work_readiness", self.current_score) or self.current_score),
                fatigue_index=self._before_break_snapshot.get("fatigue_index"),
                distraction_risk=self._before_break_snapshot.get("distraction_risk"),
            )
        )

        self.break_count += 1
        self.continuous_focus_time = 0
        self.zalo_alert_manager.mark_recovered()

        should_resume = bool(self.config.get("auto_resume_after_break", True))
        was_tracking = self.camera_running
        break_pause_started = 0.0

        if was_tracking:
            break_pause_started = time.time()
            self._session_paused = True
            self._pause_started_at = break_pause_started
            self.frame_timer.stop()
            self.stats_timer.stop()
            if hasattr(self, "task_context_timer"):
                self.task_context_timer.stop()
            if self.camera is not None:
                self.camera.stop()
            self._update_state_badge(self.display_state, float(self.current_score), "Đang nghỉ ngắn. Phiên sẽ tiếp tục sau khi quay lại.")
            self._update_journey_pip_data(force=True)
            self._sync_journey_pip_visibility()

        self._break_dialog_open = True
        game_result: dict = {}

        if not auto_triggered:
            overlay_seconds = int(self.config.get("break_overlay_seconds", 12))
            break_overlay = BreakModeDialog(duration_seconds=overlay_seconds, parent=self)
            break_overlay.exec()

        self.hide()
        try:
            # Open the focused recovery workflow and capture attention probe result
            game_result = self._open_games_with_result()
        finally:
            self._break_dialog_open = False
            self.showNormal()
            self.raise_()
            self.activateWindow()

        # Record break snapshot with game result for recovery validation
        if self._before_break_snapshot:
            snap = dict(self._before_break_snapshot)
            snap["game_result"] = game_result
            snap["break_type"] = "mini_game" if game_result else "breathing"
            snap["snapshot_id"] = f"{int(float(snap.get('timestamp', time.time())) * 1000)}-{len(self._break_snapshots)}"
            self._break_snapshots.append(snap)
            self._schedule_break_recovery_validation(snap)

        if was_tracking and should_resume:
            paused_from = float(getattr(self, "_pause_started_at", 0.0) or break_pause_started or time.time())
            self._paused_total_seconds += max(0.0, time.time() - paused_from)
            self._pause_started_at = 0.0
            self._session_paused = False
            self.last_break_time = time.time()

            restarted = True
            if self.camera is not None:
                restarted = bool(self.camera.start())
            if not restarted:
                self.camera_running = False
                self.btn_start.setChecked(False)
                self.btn_start.setText("Bắt đầu")
                NoticeDialog.warning(
                    self,
                    "Camera",
                    "Không thể mở lại camera sau khi nghỉ. Phiên đã tạm dừng, bạn có thể bấm Bắt đầu để chạy lại.",
                    config=self.config,
                )
                return

            self.camera_running = True
            self.frame_timer.start(self.frame_interval)
            self.stats_timer.start(1000)
            if self._task_context_enabled():
                self.task_context_classifier.update_from_app_config(self.config)
                self.task_context_timer.start(self.task_context_interval_ms)
            self.btn_start.setChecked(True)
            self.btn_start.setText("Dừng")
            self._update_stats()
            self._refresh_journey_map_dialog()
            self._update_journey_pip_data(force=True)
            self._sync_journey_pip_visibility()
        elif was_tracking:
            self._session_paused = False
            self._pause_started_at = 0.0
            self.btn_start.setChecked(False)
            self._stop_tracking()

    # ── Deep Focus / Journey helpers ─────────────────────────────────────────

    def _apply_deep_focus_ui(self, active: bool) -> None:
        """Show/hide widgets based on deep focus mode."""
        if hasattr(self, "trend_widget"):
            self.trend_widget.setVisible(not active)
        if hasattr(self, "guidance_widget"):
            self.guidance_widget.setVisible(not active)
        if hasattr(self, "route_map_widget"):
            self.route_map_widget.setVisible(bool(getattr(self, "_session_journey_enabled", False)))
        if hasattr(self, "journey_widget"):
            self.journey_widget.setVisible(
                bool(getattr(self, "_session_journey_enabled", False)) and (active or self.camera_running)
            )
        if hasattr(self, "task_context_card"):
            self.task_context_card.hide()
        # In deep focus, hide secondary buttons
        if hasattr(self, "btn_break"):
            self.btn_break.setVisible(not active)

    def _toggle_journey_pause(self) -> None:
        if not bool(getattr(self, "camera_running", False)):
            return
        if bool(getattr(self, "_session_paused", False)):
            self._resume_journey_session()
        else:
            self._pause_journey_session()

    def _pause_journey_session(self) -> None:
        if not bool(getattr(self, "camera_running", False)) or bool(getattr(self, "_session_paused", False)):
            return
        self._session_paused = True
        self._pause_started_at = time.time()
        self.frame_timer.stop()
        self.stats_timer.stop()
        if hasattr(self, "task_context_timer"):
            self.task_context_timer.stop()
        dialog = getattr(self, "_journey_map_dialog", None)
        if dialog is not None and hasattr(dialog, "set_paused"):
            dialog.set_paused(True)
        self._update_state_badge(self.display_state, float(self.current_score), "Tam dung phien tap trung.")
        self._refresh_journey_map_dialog()
        self._update_journey_pip_data(force=True)
        self._sync_journey_pip_visibility()

    def _resume_journey_session(self) -> None:
        if not bool(getattr(self, "camera_running", False)) or not bool(getattr(self, "_session_paused", False)):
            return
        pause_started = float(getattr(self, "_pause_started_at", 0.0) or 0.0)
        if pause_started > 0:
            self._paused_total_seconds += max(0.0, time.time() - pause_started)
        self._pause_started_at = 0.0
        self._session_paused = False
        self.frame_timer.start(self.frame_interval)
        self.stats_timer.start(1000)
        if self._task_context_enabled():
            self.task_context_classifier.update_from_app_config(self.config)
            self.task_context_timer.start(self.task_context_interval_ms)
        dialog = getattr(self, "_journey_map_dialog", None)
        if dialog is not None and hasattr(dialog, "set_paused"):
            dialog.set_paused(False)
        self._update_stats()

    def _current_focus_journey_payload(self) -> Dict[str, Any]:
        """Return the current selected focus journey as a single UI payload."""
        payload = dict(getattr(self, "_session_context_payload", {}) or {})
        payload.update(dict(getattr(self, "_session_route_payload", {}) or {}))

        if hasattr(self, "route_map_widget"):
            visual = dict(getattr(self.route_map_widget, "journey_data", {}) or {})
            if visual:
                payload.setdefault("route_from_code", visual.get("from_code"))
                payload.setdefault("route_to_code", visual.get("to_code"))
                payload.setdefault("route_from_name", visual.get("from_name"))
                payload.setdefault("route_to_name", visual.get("to_name"))
                payload.setdefault("route_distance_km", visual.get("distance_km"))

        planned_minutes = int(
            payload.get("planned_minutes")
            or getattr(self, "_session_planned_minutes", 0)
            or payload.get("route_duration_minutes")
            or 25
        )
        payload["planned_minutes"] = planned_minutes
        if planned_minutes > 0:
            payload["route_duration_minutes"] = planned_minutes
        payload["journey_session_id"] = int(
            getattr(self, "_journey_session_id", 0)
            or float(self.session_started_at or 0.0)
        )
        return payload

    def _current_focus_journey_metrics(self) -> tuple[Dict[str, Any], float, int, int, str]:
        from .journey_map_dialog import build_journey_model

        payload = self._current_focus_journey_payload()
        model = build_journey_model(payload)

        planned_seconds = int(model.get("duration_minutes", 25) or 25) * 60
        if self.camera_running and planned_seconds > 0:
            progress = max(0.0, min(1.0, float(self.session_time_seconds) / float(planned_seconds)))
        else:
            progress = max(0.0, min(1.0, float(getattr(self, "_journey_completion_ratio", 0.0) or 0.0)))

        remaining_seconds = int(max(0, round(planned_seconds * (1.0 - progress))))
        total_distance = int(model.get("distance_km", 0) or 0)
        distance_left = int(max(0, round(total_distance * (1.0 - progress))))
        phase = str(getattr(self, "_journey_phase_end", "") or self._journey_phase_from_progress(int(progress * 100)))
        return payload, progress, remaining_seconds, distance_left, phase

    def _journey_pip_enabled(self) -> bool:
        """Return whether the lightweight Journey PiP is allowed by config."""
        return bool(self.config.get("enable_journey_pip", True)) and bool(
            getattr(self, "_session_journey_enabled", False)
        )

    def _ensure_journey_pip(self) -> FocusJourneyPiPWindow:
        """Create the Journey PiP window lazily."""
        if self._journey_pip_window is None:
            self._journey_pip_window = FocusJourneyPiPWindow(
                theme_mode=str(self.config.get("theme_mode", "dark"))
            )
            self._journey_pip_window.openRequested.connect(self._restore_from_journey_pip)
            self._journey_pip_window.closeRequested.connect(self._close_journey_pip_from_button)
        return self._journey_pip_window

    def _show_journey_pip(self) -> None:
        """Show the compact Journey PiP without creating extra timers."""
        if (
            not self._journey_pip_enabled()
            or not bool(getattr(self, "camera_running", False))
            or bool(getattr(self, "_journey_pip_hidden_for_session", False))
        ):
            return

        pip = self._ensure_journey_pip()
        pip.update_theme(str(self.config.get("theme_mode", "dark")))
        self._update_journey_pip_data(force=True)
        pip.place_near_parent(self)
        if not pip.isVisible():
            pip.show()
        pip.raise_()

    def _hide_journey_pip(self) -> None:
        """Hide PiP without changing the current session preference."""
        pip = getattr(self, "_journey_pip_window", None)
        if pip is not None and pip.isVisible():
            pip.hide()

    def _close_journey_pip_from_button(self) -> None:
        """Close PiP for the current minimized/hidden window cycle only."""
        self._journey_pip_closed_until_restore = True
        self._hide_journey_pip()

    def _hide_journey_pip_for_session(self) -> None:
        """User chose to hide PiP for this active session only."""
        self._journey_pip_hidden_for_session = True
        self._hide_journey_pip()

    def _restore_from_journey_pip(self) -> None:
        """Restore the full Journey map from PiP and hide the floating monitor."""
        self._journey_pip_closed_until_restore = False
        self._hide_journey_pip()
        if (
            bool(getattr(self, "_session_journey_enabled", False))
            and bool(getattr(self, "camera_running", False))
        ):
            dialog = getattr(self, "_journey_map_dialog", None)
            if dialog is None:
                self._open_journey_map_dialog()
                return

            self.hide()
            dialog.showNormal()
            dialog.show()
            self._refresh_journey_map_dialog(force_route=True)
            dialog.raise_()
            dialog.activateWindow()
            return

        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _sync_journey_pip_visibility(self) -> None:
        """Show PiP only while tracking is active and the main window is minimized/hidden."""
        if bool(getattr(self, "_closing", False)) or not self._journey_pip_enabled():
            self._hide_journey_pip()
            return

        if not bool(getattr(self, "camera_running", False)):
            self._hide_journey_pip()
            return

        dialog = getattr(self, "_journey_map_dialog", None)
        dialog_minimized = bool(
            dialog is not None
            and (dialog.isMinimized() or bool(dialog.windowState() & Qt.WindowState.WindowMinimized))
        )
        if dialog is not None and dialog.isVisible() and not dialog_minimized:
            self._hide_journey_pip()
            return

        if bool(getattr(self, "_journey_pip_hidden_for_session", False)):
            self._hide_journey_pip()
            return

        if bool(getattr(self, "_journey_pip_closed_until_restore", False)):
            self._hide_journey_pip()
            return

        minimized = bool(self.windowState() & Qt.WindowState.WindowMinimized) or self.isMinimized()
        hidden_to_tray = not self.isVisible()
        if minimized or hidden_to_tray:
            self._show_journey_pip()
        else:
            self._hide_journey_pip()

    def _update_journey_pip_data(
        self,
        *,
        payload: Optional[Dict[str, Any]] = None,
        progress: Optional[float] = None,
        remaining_seconds: Optional[int] = None,
        phase: Optional[str] = None,
        status_text: str = "",
        force: bool = False,
    ) -> None:
        """Push current Journey values into PiP if the window exists."""
        pip = getattr(self, "_journey_pip_window", None)
        if pip is None:
            return

        if payload is None or progress is None or remaining_seconds is None or phase is None:
            payload, progress, remaining_seconds, _distance_left, phase = self._current_focus_journey_metrics()

        status = "Tạm dừng" if bool(getattr(self, "_session_paused", False)) else str(status_text or "")
        route_from = str(
            (payload or {}).get("route_from_code")
            or (payload or {}).get("from_code")
            or ""
        )
        route_to = str(
            (payload or {}).get("route_to_code")
            or (payload or {}).get("to_code")
            or ""
        )
        progress_key = (
            route_from,
            route_to,
            int(round(float(progress or 0.0) * 1000)),
            int(remaining_seconds or 0),
            str(phase or ""),
            status,
            getattr(self.current_state, "name", str(self.current_state)),
        )
        if not force and progress_key == getattr(self, "_journey_pip_progress_key", ()):
            return

        pip.update_data(
            route_from_code=route_from,
            route_to_code=route_to,
            progress=float(progress or 0.0),
            remaining_seconds=int(remaining_seconds or 0),
            phase=str(phase or "Boarding"),
            status_text=status,
            state=self.current_state,
            payload=dict(payload or {}),
        )
        self._journey_pip_progress_key = progress_key

    def _begin_journey_measurement_after_boarding(self) -> None:
        """Start calibration and the session clock only after the ticket is torn."""
        if not bool(getattr(self, "camera_running", False)):
            return

        now = time.time()
        warmup_seconds = max(0.0, float(getattr(self, "_analysis_warmup_seconds", 0.0) or 0.0))
        self._journey_waiting_for_boarding = False
        self._journey_calibration_reset_done = False
        self._last_boarding_preview_at = 0.0
        self._analysis_started_at = now
        self._initial_baseline_samples = []
        self._initial_session_baseline = {}
        self._initial_baseline_finalized = False
        self.session_started_at = now + warmup_seconds
        self.session_time_seconds = 0
        self._paused_total_seconds = 0.0
        self._pause_started_at = 0.0
        self.last_break_time = self.session_started_at
        self._last_state_frame_timestamp = None
        self._display_score = 100.0
        self.current_score = 100.0
        self.current_state = FocusState.UNCERTAIN
        self.display_state = FocusState.UNCERTAIN
        self.continuous_focus_time = 0.0
        self.focus_time = 0.0
        self.raw_focus_time = 0.0
        self.score_samples = []
        self.raw_score_samples = []
        self.focus_trend_samples = []
        self.raw_state_time_by_state = {state.name: 0.0 for state in FocusState}
        self.state_time_by_state = {state.name: 0.0 for state in FocusState}
        self._session_state_segments = []
        self._session_focus_score_start = None
        self._session_focus_score_end = None
        self._session_fatigue_onset_seconds = None
        if self.engine is not None:
            self.engine.reset()

        self.score_widget.set_score(100.0, FocusState.UNCERTAIN)
        self._update_score_breakdown()
        self._update_state_badge(FocusState.UNCERTAIN, 0.0, "Đang lấy mốc đầu phiên sau check-in.")
        self._update_journey_widget()
        self._refresh_journey_map_dialog(force_route=True)
        self._update_journey_pip_data(force=True)

    def _open_journey_map_dialog(self) -> None:
        """Open the large flight-focus map for the selected journey."""
        if not bool(getattr(self, "_session_journey_enabled", False)):
            return
        try:
            from .journey_map_dialog import FocusJourneyMapDialog
        except Exception as exc:
            logger.exception("Unable to open Focus Journey map dialog: %s", exc)
            NoticeDialog.warning(
                self,
                "Focus Journey",
                "Khong the mo ban do hanh trinh luc nay. FocusGuardian se tiep tuc hien thi ban do nho.",
                config=self.config,
            )
            return

        if self._journey_map_dialog is None:
            self._journey_map_dialog = FocusJourneyMapDialog(
                config=self.config,
                audio_manager=getattr(self, "focus_audio_manager", None),
                parent=None,
            )
            self._journey_map_dialog.pauseRequested.connect(self._toggle_journey_pause)
            if hasattr(self._journey_map_dialog, "boardingCompleted"):
                self._journey_map_dialog.boardingCompleted.connect(self._begin_journey_measurement_after_boarding)
            if hasattr(self._journey_map_dialog, "dismissed"):
                self._journey_map_dialog.dismissed.connect(self._handle_journey_map_dismissed)
            if hasattr(self._journey_map_dialog, "minimizedRequested"):
                self._journey_map_dialog.minimizedRequested.connect(self._handle_journey_map_minimized)
            self._journey_map_dialog_route_key = ()
            self._journey_map_dialog_progress_key = ()

        self.hide()
        self._journey_map_dialog.show()
        if hasattr(self._journey_map_dialog, "set_paused"):
            self._journey_map_dialog.set_paused(bool(getattr(self, "_session_paused", False)))
        self._refresh_journey_map_dialog(force_route=True)
        self._journey_map_dialog.raise_()
        self._journey_map_dialog.activateWindow()

    def _refresh_journey_map_dialog(self, *, force_route: bool = False) -> None:
        if not bool(getattr(self, "_session_journey_enabled", False)):
            return
        dialog = getattr(self, "_journey_map_dialog", None)
        if dialog is None or not dialog.isVisible():
            return

        payload, progress, remaining_seconds, distance_left, phase = self._current_focus_journey_metrics()
        route_key = (
            str(payload.get("route_from_code") or payload.get("from_code") or ""),
            str(payload.get("route_to_code") or payload.get("to_code") or ""),
            int(payload.get("planned_minutes") or payload.get("route_duration_minutes") or 0),
            int(payload.get("route_distance_km") or payload.get("distance_km") or 0),
        )
        if force_route or route_key != getattr(self, "_journey_map_dialog_route_key", ()):
            dialog.set_journey_data(payload)
            self._journey_map_dialog_route_key = route_key
            self._journey_map_dialog_progress_key = ()
        progress_key = (
            int(round(progress * 10000)),
            int(remaining_seconds),
            int(distance_left),
            str(phase or ""),
        )
        if progress_key != getattr(self, "_journey_map_dialog_progress_key", ()):
            dialog.update_progress(progress, remaining_seconds, distance_left, phase)
            self._journey_map_dialog_progress_key = progress_key

    def _handle_journey_map_dismissed(self, ticket_checked: bool = False) -> None:
        """Do not leave tracking stuck in boarding when the flight map is closed."""
        self._journey_map_dialog_progress_key = ()
        if bool(getattr(self, "_closing", False)):
            return
        self._journey_pip_closed_until_restore = False
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self._hide_journey_pip()
        if not bool(getattr(self, "camera_running", False)):
            return
        if bool(getattr(self, "_journey_waiting_for_boarding", False)):
            logger.info(
                "Focus Journey map dismissed during boarding; starting measurement (ticket_checked=%s)",
                bool(ticket_checked),
            )
            dialog = getattr(self, "_journey_map_dialog", None)
            if dialog is not None and hasattr(dialog, "mark_boarding_complete"):
                dialog.mark_boarding_complete()
            QTimer.singleShot(0, self._begin_journey_measurement_after_boarding)

    def _handle_journey_map_minimized(self) -> None:
        """Show PiP when the standalone Journey map is minimized."""
        if bool(getattr(self, "_closing", False)):
            return
        if not bool(getattr(self, "camera_running", False)):
            return
        if not bool(getattr(self, "_session_journey_enabled", False)):
            return
        self._journey_pip_closed_until_restore = False
        self.hide()
        self._update_journey_pip_data(force=True)
        self._show_journey_pip()

    def _update_journey_widget(self) -> None:
        """Refresh the journey progress card."""
        if not hasattr(self, "journey_widget"):
            return
        if not bool(getattr(self, "_session_journey_enabled", False)):
            self.journey_widget.hide()
            if hasattr(self, "route_map_widget"):
                self.route_map_widget.hide()
            self._hide_journey_pip()
            return
        if not self.camera_running:
            self.journey_widget.hide()
            if hasattr(self, "route_map_widget"):
                self.route_map_widget.show()
                self.route_map_widget.update_route(self._session_route_payload, 0.0, 0, "Boarding", "ready")
            self._refresh_journey_map_dialog()
            self._hide_journey_pip()
            return

        self.journey_widget.show()
        if hasattr(self, "route_map_widget"):
            self.route_map_widget.show()

        planned_s = self._session_planned_minutes * 60
        elapsed_s = int(self.session_time_seconds)
        is_dark = str(self.config.get("theme_mode", "dark")).strip().lower() != "light"

        if planned_s > 0:
            pct = int(min(100, elapsed_s * 100 // planned_s))
            remaining = max(0, planned_s - elapsed_s)
        else:
            work_min, _ = self._current_schedule_minutes()
            work_s = work_min * 60
            elapsed_since_break = max(0.0, time.time() - self.last_break_time)
            pct = int(min(100, elapsed_since_break * 100 // max(1, work_s)))
            remaining = max(0, work_s - int(elapsed_since_break))

        # Determine phase
        score = float(self.current_score)
        trend = self._compute_focus_trend_delta()
        if self._is_initial_analysis_phase():
            phase = "warmup"
        elif pct >= 90 or (planned_s > 0 and remaining < 120):
            phase = "landing"
        elif score < 55 or trend <= -8 or self._is_distraction_state(self.current_state):
            phase = "declining"
        else:
            phase = "focusing"

        route_phase = self._journey_phase_from_progress(pct)
        route_status = ""
        if self.current_state in (FocusState.DROWSY_FATIGUE, FocusState.PHONE_DISTRACTION):
            route_phase = "Turbulence"
            route_status = "nhẹ"
        self._journey_phase_end = route_phase
        self._journey_completion_ratio = max(0.0, min(1.0, pct / 100.0))

        if hasattr(self, "route_map_widget"):
            self.route_map_widget.update_route(
                route_payload=self._session_route_payload,
                progress=self._journey_completion_ratio,
                remaining_seconds=remaining,
                phase=route_phase,
                status=route_status,
            )
        self._update_journey_pip_data(
            payload=self._current_focus_journey_payload(),
            progress=self._journey_completion_ratio,
            remaining_seconds=remaining,
            phase=route_phase,
            status_text=route_status,
        )
        self._sync_journey_pip_visibility()
        self._refresh_journey_map_dialog()

        self.journey_widget.update_journey(
            phase=phase,
            percent=pct,
            remaining_seconds=remaining,
            goal=self._session_goal,
            session_mode=self._session_mode,
            is_dark=is_dark,
        )

    @staticmethod
    def _journey_phase_from_progress(percent: int) -> str:
        pct = max(0, min(100, int(percent)))
        if pct < 5:
            return "Boarding"
        if pct < 15:
            return "Takeoff"
        if pct < 85:
            return "Cruise"
        return "Landing"

    def _open_games_with_result(self) -> dict:
        """Open the recovery game and return attention probe result dict."""
        try:
            from ..focus_reset_game.ui_v2 import FocusResetDialog
            from ..focus_reset_game.config import load_focus_reset_config
            from ..focus_reset_game.storage import SessionStorage

            cfg = load_focus_reset_config()
            storage = SessionStorage(cfg.history_path)
            theme_mode = str(self.config.get("theme_mode", "dark"))
            sound_enabled = bool(self.config.get("enable_focus_audio", False))
            volume = int(self.config.get("focus_audio_volume", 70))
            _work_minutes, break_minutes = self._current_schedule_minutes()
            before = dict(getattr(self, "_before_break_snapshot", {}) or {})
            break_context = {
                "profile_name": str(getattr(self, "profile_name", "default") or "default"),
                "planned_break_minutes": int(break_minutes),
                "break_duration_minutes": int(break_minutes),
                "pre_work_readiness": float(before.get("work_readiness", self.current_score) or self.current_score),
                "state": str(before.get("state", getattr(self.current_state, "name", "")) or ""),
                "session_seconds": int(before.get("session_seconds", self.session_time_seconds) or 0),
                "fatigue_index": float(before.get("fatigue_index", 0.0) or 0.0),
                "distraction_risk": float(before.get("distraction_risk", 0.0) or 0.0),
                "initial_work_readiness": before.get("initial_work_readiness"),
                "readiness_delta_from_start": before.get("readiness_delta_from_start"),
                "initial_baseline_quality": before.get("initial_baseline_quality"),
            }

            dialog = FocusResetDialog(
                parent=self,
                config=cfg,
                storage=storage,
                theme_mode=theme_mode,
                app_sound_enabled=sound_enabled,
                app_volume=volume,
                break_context=break_context,
            )
            dialog.exec()

            # Retrieve attention probe result if available
            if hasattr(dialog, "get_attention_probe_result"):
                return dialog.get_attention_probe_result()
        except Exception as exc:
            logger.debug("Focus reset game failed: %s", exc)
        return {}

    def _schedule_break_recovery_validation(self, snap: dict) -> None:
        """Capture work-readiness transfer after the user has returned to work."""
        game = snap.get("game_result", {}) or {}
        if not bool(game.get("probe_completed", False)):
            return

        delay_minutes = float(self.config.get("break_recovery_validation_minutes", 5.0) or 5.0)
        delay_minutes = max(5.0, min(10.0, delay_minutes))
        snap["validation_due_at"] = time.time() + (delay_minutes * 60.0)
        snap["validation_delay_target_minutes"] = delay_minutes

        snapshot_id = str(snap.get("snapshot_id", ""))
        if not snapshot_id:
            return
        validation_snapshot = dict(snap)
        QTimer.singleShot(
            int(delay_minutes * 60_000),
            lambda sid=snapshot_id, payload=validation_snapshot: self._capture_break_recovery_validation(sid, payload),
        )

    def _capture_break_recovery_validation(self, snapshot_id: str, fallback_snap: dict | None = None) -> None:
        snap = next(
            (item for item in reversed(self._break_snapshots) if str(item.get("snapshot_id", "")) == snapshot_id),
            None,
        )
        if snap is None and fallback_snap:
            snap = dict(fallback_snap)
        if not snap or snap.get("validation_status") == "validated":
            return

        if not bool(getattr(self, "camera_running", False)) or bool(getattr(self, "_session_paused", False)):
            snap["validation_status"] = "skipped_not_tracking"
            return

        post_score = float(getattr(self, "current_score", 50.0) or 50.0)
        now = time.time()
        snap["post_work_readiness"] = post_score
        snap["validated_at"] = now
        snap["validation_delay_minutes"] = max(0.0, (now - float(snap.get("timestamp", now) or now)) / 60.0)
        initial_wr = snap.get("initial_work_readiness")
        if initial_wr not in (None, ""):
            initial_wr_float = float(initial_wr)
            snap["post_readiness_delta_from_start"] = round(post_score - initial_wr_float, 3)
            if initial_wr_float > 1e-6:
                snap["recovery_to_initial_ratio"] = round(
                    self._clamped_ratio(post_score / initial_wr_float, 0.0, 1.5),
                    4,
                )
        return_state_seconds = dict(getattr(self, "state_time_by_state", {}) or {})
        return_total_seconds = max(1.0, sum(float(v or 0.0) for v in return_state_seconds.values()))
        return_stable_seconds = (
            float(return_state_seconds.get("ON_SCREEN_READING", 0.0) or 0.0)
            + float(return_state_seconds.get("OFFSCREEN_WRITING", 0.0) or 0.0)
        )
        snap["return_state_seconds"] = return_state_seconds
        snap["return_work_stable_ratio"] = max(0.0, min(1.0, return_stable_seconds / return_total_seconds))
        snap["return_distraction_count"] = int(getattr(self, "distraction_count", 0) or 0)
        snap["return_drowsy_seconds"] = float(return_state_seconds.get("DROWSY_FATIGUE", 0.0) or 0.0)
        snap["return_away_seconds"] = float(return_state_seconds.get("AWAY", 0.0) or 0.0)
        snap["transfer_score"] = self._compute_recovery_validation(snap, post_score)
        snap["recovery_success"] = bool(
            float(snap.get("return_work_stable_ratio", 0.0) or 0.0) >= 0.70
            and post_score >= 60.0
            and float(snap.get("return_drowsy_seconds", 0.0) or 0.0) <= 20.0
            and float(snap.get("return_away_seconds", 0.0) or 0.0) <= 20.0
        )
        snap["validation_status"] = "validated"
        self._persist_focus_reset_recovery_validation(snap)

    def _persist_focus_reset_recovery_validation(self, snap: dict) -> None:
        try:
            from ..focus_reset_game.config import load_focus_reset_config
            from ..focus_reset_game.storage import SessionStorage

            cfg = load_focus_reset_config()
            storage = SessionStorage(cfg.history_path)
            game = snap.get("game_result", {}) or {}
            before_wr = float(snap.get("work_readiness", 50.0) or 50.0)
            post_wr = float(snap.get("post_work_readiness", before_wr) or before_wr)
            timestamp = float(snap.get("timestamp", time.time()) or time.time())
            validated_at = float(snap.get("validated_at", time.time()) or time.time())

            recovery_record = {
                "timestamp": datetime.fromtimestamp(timestamp).isoformat(timespec="seconds"),
                "validated_at": datetime.fromtimestamp(validated_at).isoformat(timespec="seconds"),
                "session_id": str(getattr(self, "_validation_session_id", "") or getattr(self, "_journey_session_id", "")),
                "profile_name": str(self.profile_name or self._get_profile_name()),
                "validation_delay_minutes": round(float(snap.get("validation_delay_minutes", 0.0) or 0.0), 2),
                "pre_work_readiness": round(before_wr, 2),
                "post_work_readiness": round(post_wr, 2),
                "readiness_delta": round(post_wr - before_wr, 2),
                "initial_work_readiness": snap.get("initial_work_readiness"),
                "readiness_delta_from_start": snap.get("readiness_delta_from_start"),
                "post_readiness_delta_from_start": snap.get("post_readiness_delta_from_start"),
                "recovery_to_initial_ratio": snap.get("recovery_to_initial_ratio"),
                "initial_fatigue_index": snap.get("initial_fatigue_index"),
                "fatigue_delta_from_start": snap.get("fatigue_delta_from_start"),
                "initial_distraction_risk": snap.get("initial_distraction_risk"),
                "distraction_delta_from_start": snap.get("distraction_delta_from_start"),
                "initial_baseline_quality": snap.get("initial_baseline_quality"),
                "fatigue_index": round(float(snap.get("fatigue_index", 0.0) or 0.0), 4),
                "distraction_risk": round(float(snap.get("distraction_risk", 0.0) or 0.0), 4),
                "transfer_score": round(float(snap.get("transfer_score", 0.0) or 0.0), 4),
                "recovery_success": bool(snap.get("recovery_success", False)),
                "break_type": str(snap.get("break_type", "")),
                "probe_completed": bool(game.get("probe_completed", False)),
                "game_attention_score": game.get("game_attention_score"),
                "attention_stability": game.get("attention_stability"),
                "accuracy": game.get("accuracy"),
                "avg_reaction_time_ms": game.get("avg_reaction_time_ms"),
                "reaction_variability_ms": game.get("reaction_variability_ms"),
                "omission_errors": game.get("omission_errors"),
                "commission_errors": game.get("commission_errors"),
                "self_report_ready": game.get("self_report_ready"),
                "best_game": game.get("best_game"),
                "weakest_game": game.get("weakest_game"),
                "selected_games": list(game.get("selected_games", []) or []),
                "game_scores": dict(game.get("game_scores", {}) or {}),
                "return_work_stable_ratio": round(float(snap.get("return_work_stable_ratio", 0.0) or 0.0), 4),
                "return_distraction_count": int(snap.get("return_distraction_count", 0) or 0),
                "return_drowsy_seconds": round(float(snap.get("return_drowsy_seconds", 0.0) or 0.0), 2),
                "return_away_seconds": round(float(snap.get("return_away_seconds", 0.0) or 0.0), 2),
            }
            storage.append_recovery_validation(recovery_record)
            self.validation_store.append_scientific_event({
                **recovery_record,
                "timestamp": timestamp,
                "timestamp_iso": recovery_record["timestamp"],
                "event_type": "break_recovery",
            })
        except Exception as exc:
            logger.debug("Failed to persist recovery validation: %s", exc)

    def _compute_recovery_validation(self, snap: dict, post_score: float) -> float:
        """
        Compute transfer_score for a break.

        Rules:
        - game_attention_score good + post_score good → high transfer
        - game good + post_score poor → low transfer (task motivation issue)
        - game poor + post_score poor → fatigue likely
        - game poor + post_score good → game not suitable for this user
        """
        before_wr = float(snap.get("work_readiness", 50.0) or 50.0)
        game = snap.get("game_result", {}) or {}
        game_score = float(game.get("game_attention_score", -1.0) or -1.0)
        accuracy = float(game.get("accuracy", -1.0) or -1.0)

        # Score improvement ratio
        improvement = (post_score - before_wr) / max(1.0, 100.0 - before_wr)
        improvement = max(-1.0, min(1.0, improvement))

        if game_score < 0:
            # No game data — use score improvement only
            return max(0.0, min(1.0, 0.5 + improvement * 0.5))

        game_good = game_score >= 0.65 and (accuracy < 0 or accuracy >= 0.70)
        post_good = post_score >= 60

        if game_good and post_good:
            transfer = 0.75 + improvement * 0.25
        elif game_good and not post_good:
            transfer = 0.35  # task motivation / difficulty issue
        elif not game_good and not post_good:
            transfer = 0.15  # likely fatigue
        else:
            transfer = 0.55  # game not suitable but user recovered

        return max(0.0, min(1.0, float(transfer)))

    def _show_habit_report_if_ready(self) -> None:
        """Show SessionHabitReportDialog after a session if enough data exists."""
        if self.session_time_seconds < 120:
            return
        try:
            # Build a minimal session record for the report
            avg_score = (
                float(sum(self.raw_score_samples) / len(self.raw_score_samples))
                if self.raw_score_samples else float(self.current_score)
            )
            session_seconds = int(self.session_time_seconds)
            focus_seconds = float(getattr(self, "raw_focus_time", 0.0))

            # Attach break effectiveness from snapshots
            break_effectiveness = []
            for snap in self._break_snapshots:
                post_score = float(snap.get("post_work_readiness", avg_score))
                transfer = float(snap.get("transfer_score", self._compute_recovery_validation(snap, post_score)) or 0.0)
                break_effectiveness.append({
                    "break_type": str(snap.get("break_type", "nghỉ")),
                    "transfer_score": transfer,
                })

            session_record = {
                "session_seconds": session_seconds,
                "focus_seconds": float(focus_seconds),
                "avg_score": avg_score,
                "distraction_count": int(self.distraction_count),
                "score_drop_per_hour": float(getattr(self, "_session_focus_score_start", avg_score) or avg_score)
                    - float(getattr(self, "_session_focus_score_end", avg_score) or avg_score),
                "fatigue_onset_minutes": (
                    self._session_fatigue_onset_seconds / 60.0
                    if getattr(self, "_session_fatigue_onset_seconds", None) is not None
                    else None
                ),
                "perclos": float(getattr(self, "_session_perclos_frames", 0))
                    / max(1, int(getattr(self, "_session_eye_metric_frames", 1))),
                "blink_rate_per_min": 0.0,
                "eye_closure_ratio": 0.0,
                "session_context": dict(self._session_context_payload or {}),
                "checkins": list(self._session_checkins or []),
                "work_interval_minutes_used": int(self.config.get("break_interval_minutes", 25)),
                "break_duration_minutes_used": int(self.config.get("break_duration_minutes", 5)),
            }

            habit_report = self.analytics_store.build_session_habit_report(
                session_record, profile_name=self.profile_name
            )
            habit_report["break_effectiveness"] = break_effectiveness

            report_dialog = SessionHabitReportDialog(
                habit_report=habit_report,
                config=self.config,
                parent=self,
            )
            report_dialog.exec()
        except Exception as exc:
            logger.debug("Failed to show habit report: %s", exc)

    @pyqtSlot()
    def _open_settings(self):
        """Open settings dialog."""
        from .settings_dialog import SettingsDialog

        dialog = SettingsDialog(
            self.config,
            self,
            focus_audio_manager=self.focus_audio_manager,
            personalization_status=self._personalization_status_payload(),
        )
        dialog.config_applied.connect(self._on_settings_applied)
        dialog.baseline_reset_requested.connect(lambda: self._reset_current_profile_baseline(dialog))
        dialog.exec()

    def _personalization_status_payload(self) -> Dict[str, Any]:
        try:
            return self.analytics_store.get_personalization_status(self.profile_name)
        except Exception as exc:
            logger.debug("Failed to build personalization status: %s", exc)
            return {
                "label": "Chưa đủ dữ liệu",
                "eligible_sessions": 0,
                "confidence": 0.0,
            }

    def _reset_current_profile_baseline(self, dialog: Optional[QDialog] = None) -> None:
        """Reset personalization baseline for the active profile only."""
        try:
            default_work = int(self.config.get("break_interval_minutes", 25) or 25)
            default_break = int(self.config.get("break_duration_minutes", 5) or 5)
            status = self.analytics_store.reset_profile_baseline(
                self.profile_name,
                default_work=default_work,
                default_break=default_break,
            )
            self._last_recommendation = {
                "work_minutes": default_work,
                "break_minutes": default_break,
                "confidence": 0.0,
                "based_on_sessions": 0,
                "adaptation_stage": "cold_start",
            }
            if self.engine is not None:
                self.engine.clear_personalization()
            if dialog is not None and hasattr(dialog, "set_personalization_status"):
                dialog.set_personalization_status(status)
            NoticeDialog.info(
                dialog or self,
                "Đã đặt lại baseline",
                "App sẽ học lại baseline từ các phiên mới của hồ sơ hiện tại.",
                config=self.config,
            )
        except Exception as exc:
            logger.exception("Failed to reset personalization baseline: %s", exc)
            NoticeDialog.warning(
                dialog or self,
                "Không thể đặt lại baseline",
                "Có lỗi khi đặt lại baseline cá nhân hóa.",
                config=self.config,
            )

    @pyqtSlot(dict)
    def _on_settings_applied(self, updates: dict) -> None:
        """Apply settings updates immediately without closing the dialog."""
        updates = dict(updates or {})
        if updates.get("enable_zalo_alerts") and str(updates.get("zalo_chat_id", "") or "").strip():
            self.config["_zalo_alerts_user_enabled_once"] = True
        self.config.update(updates)
        self._apply_config()
        self._sync_profile_scoped_settings_to_supabase()
        self.config_changed.emit(self.config.copy())

    @pyqtSlot()
    def _request_logout(self) -> None:
        """Request application-level logout and auth-gate return."""
        confirm = NoticeDialog.confirm(
            self,
            "Đăng xuất khỏi FocusGuardian",
            "Phiên hiện tại sẽ đóng và bạn sẽ quay lại màn đăng nhập. Dữ liệu đã lưu vẫn được giữ nguyên.",
            config=self.config,
            confirm_text="Đăng xuất",
            cancel_text="Ở lại",
        )
        if not confirm:
            return
        self.logout_requested.emit()

    def refresh_authenticated_profile(self) -> None:
        """Refresh profile-dependent runtime state after login/logout changes."""
        self.profile_name = self._get_profile_name()
        self.config["profile_name"] = self.profile_name
        self._reset_profile_scoped_settings_to_defaults()
        self._load_profile_scoped_settings_from_supabase(seed_if_missing=True)
        self._apply_config()
        self._refresh_today_stats_card()
        self.config_changed.emit(self.config.copy())

    def _apply_config(self):
        """Apply configuration changes."""
        self.config["enable_personalization"] = True
        self.config["auto_apply_personalization"] = True
        self.auth_manager.configure(self.config)
        self._apply_theme()
        self._display_uncertain_hold_seconds = max(
            0.8,
            float(self.config.get("display_uncertain_hold_seconds", self._display_uncertain_hold_seconds)),
        )
        self.profile_name = self._get_profile_name()
        self.config["profile_name"] = self.profile_name
        self.analytics_store.configure_supabase(self.config)
        self._normalize_zalo_runtime_config()
        self.zalo_alert_manager.configure(self.config)
        self.focus_audio_manager.load_from_config(self.config)
        self._apply_focus_engine_config()
        self._apply_personalized_schedule()

        # Update camera if needed
        if self.vision_available and self.camera:
            from ..vision import CameraCapture, CameraConfig

            new_camera_id = int(self.config.get("camera_id", 0))
            width, height = self._parse_resolution(self.config.get("resolution", "640x480"))
            fps = int(self.config.get("fps", 15))

            current = self.camera.config
            camera_changed = (
                current.camera_index != new_camera_id
                or current.width != width
                or current.height != height
                or current.fps != fps
            )

            if camera_changed:
                if self.camera_running:
                    self._stop_tracking()
                    self.btn_start.setChecked(False)

                self.camera = CameraCapture(
                    config=CameraConfig(
                        camera_index=new_camera_id,
                        width=width,
                        height=height,
                        fps=fps,
                        process_width=min(width, 480),
                        process_height=min(height, 360),
                    )
                )

            # Reconfigure phone detector from updated settings
            from ..vision.phone_detector import PhoneDetector, PhoneDetectorConfig

            phone_enabled = bool(self.config.get("enable_phone_detection", True))
            phone_mode = str(self.config.get("phone_detection_mode", "heuristic"))
            phone_conf_threshold = float(self.config.get("phone_confidence_threshold", 0.55))
            phone_interval_frames = max(1, int(self.config.get("phone_detection_interval_frames", 4) or 4))
            phone_confirm_window_seconds = max(
                0.8,
                float(self.config.get("phone_confirmation_window_seconds", 2.5) or 2.5),
            )
            phone_confirm_hits = max(1, int(self.config.get("phone_confirmation_min_hits", 3) or 3))

            if hasattr(self, "phone_detector") and self.phone_detector is not None:
                self.phone_detector.release()

            self.phone_detector = PhoneDetector(
                PhoneDetectorConfig(
                    enabled=phone_enabled,
                    model_type=phone_mode,
                    confidence_threshold=phone_conf_threshold,
                    run_interval_frames=phone_interval_frames,
                    confirmation_window_seconds=phone_confirm_window_seconds,
                    confirmation_min_hits=phone_confirm_hits,
                )
            )

            if not self.phone_detector.initialize() and phone_enabled:
                logger.warning(
                    "Requested phone detector mode '%s' unavailable; fallback to heuristic",
                    phone_mode,
                )
                self.phone_detector = PhoneDetector(
                    PhoneDetectorConfig(
                        enabled=True,
                        model_type="heuristic",
                        confidence_threshold=phone_conf_threshold,
                        run_interval_frames=phone_interval_frames,
                        confirmation_window_seconds=phone_confirm_window_seconds,
                        confirmation_min_hits=phone_confirm_hits,
                    )
                )
                self.phone_detector.initialize()

    def _normalize_zalo_runtime_config(self) -> None:
        """Keep Zalo alert runtime state coherent across profile/cloud reloads."""
        chat_id = str(self.config.get("zalo_chat_id", "") or "").strip()
        user_enabled_once = bool(self.config.get("_zalo_alerts_user_enabled_once", False))
        if chat_id and bool(self.config.get("enable_zalo_alerts", False)):
            self.config["_zalo_alerts_user_enabled_once"] = True
            user_enabled_once = True
        if chat_id and user_enabled_once and not bool(self.config.get("enable_zalo_alerts", False)):
            logger.debug("Restoring Zalo alerts enabled state from connected runtime chat_id")
            self.config["enable_zalo_alerts"] = True

        try:
            legacy_threshold = float(self.config.get("zalo_alert_threshold_seconds", 5) or 5)
        except (TypeError, ValueError):
            legacy_threshold = 5.0
        if "zalo_distraction_confirm_seconds" not in self.config:
            self.config["zalo_distraction_confirm_seconds"] = 5 if legacy_threshold >= 30 else int(max(1, legacy_threshold))

        if "zalo_state_cooldown_seconds" not in self.config:
            try:
                cooldown_minutes = float(self.config.get("zalo_alert_cooldown_minutes", 10) or 0)
            except (TypeError, ValueError):
                cooldown_minutes = 10.0
            self.config["zalo_state_cooldown_seconds"] = int(max(0.0, cooldown_minutes) * 60)

        logger.debug(
            "MainWindow Zalo config before manager: enabled=%s chat_id=%s confirm=%ss cooldown=%ss",
            bool(self.config.get("enable_zalo_alerts", False)),
            "set" if chat_id else "missing",
            self.config.get("zalo_distraction_confirm_seconds"),
            self.config.get("zalo_state_cooldown_seconds"),
        )

    @staticmethod
    def _parse_resolution(resolution: str) -> tuple[int, int]:
        """Parse resolution string like 640x480 to numeric width and height."""
        try:
            width_str, height_str = resolution.lower().split("x", 1)
            width = int(width_str.strip())
            height = int(height_str.strip())
            if width > 0 and height > 0:
                return width, height
        except (AttributeError, ValueError):
            pass
        return 640, 480

    @pyqtSlot()
    def _open_games(self):
        """Open the Focus Reset recovery dialog directly."""
        self._open_games_with_result()

    def stop_focus_audio(self) -> None:
        """Stop focus background audio immediately."""
        if hasattr(self, "focus_audio_manager") and self.focus_audio_manager is not None:
            self.focus_audio_manager.stop()

    def restore_focus_audio_from_config(self) -> None:
        """Restore focus audio state from current runtime config."""
        if hasattr(self, "focus_audio_manager") and self.focus_audio_manager is not None:
            self.focus_audio_manager.load_from_config(self.config)

    def _sync_responsive_layout(self) -> None:
        """Adjust column balance for smaller widths without breaking the design."""
        if not hasattr(self, "_main_layout"):
            return

        width = max(1, self.width())
        if width < 1180:
            self._main_layout.setStretch(0, 58)
            self._main_layout.setStretch(1, 42)
            if hasattr(self, "right_column_scroll"):
                self.right_column_scroll.setMinimumWidth(330)
        elif width < 1320:
            self._main_layout.setStretch(0, 61)
            self._main_layout.setStretch(1, 39)
            if hasattr(self, "right_column_scroll"):
                self.right_column_scroll.setMinimumWidth(340)
        else:
            self._main_layout.setStretch(0, 64)
            self._main_layout.setStretch(1, 36)
            if hasattr(self, "right_column_scroll"):
                self.right_column_scroll.setMinimumWidth(350)

    def _resize_edges_from_local_pos(self, local_pos: QPoint) -> Qt.Edge:
        """Compute frameless resize edges from local coordinates."""
        if self.isMaximized() or self.isFullScreen():
            return Qt.Edge(0)

        x = int(local_pos.x())
        y = int(local_pos.y())
        w = max(1, self.width())
        h = max(1, self.height())
        m = max(4, int(self._resize_border_px))

        on_left = 0 <= x <= m
        on_right = (w - m - 1) <= x <= (w - 1)
        on_top = 0 <= y <= m
        on_bottom = (h - m - 1) <= y <= (h - 1)

        edges = Qt.Edge(0)
        if on_left:
            edges |= Qt.Edge.LeftEdge
        elif on_right:
            edges |= Qt.Edge.RightEdge

        if on_top:
            edges |= Qt.Edge.TopEdge
        elif on_bottom:
            edges |= Qt.Edge.BottomEdge

        return edges

    @staticmethod
    def _cursor_for_resize_edges(edges: Qt.Edge) -> Qt.CursorShape:
        """Map resize edges to expected desktop resize cursors."""
        has_left = bool(edges & Qt.Edge.LeftEdge)
        has_right = bool(edges & Qt.Edge.RightEdge)
        has_top = bool(edges & Qt.Edge.TopEdge)
        has_bottom = bool(edges & Qt.Edge.BottomEdge)

        if (has_top and has_left) or (has_bottom and has_right):
            return Qt.CursorShape.SizeFDiagCursor
        if (has_top and has_right) or (has_bottom and has_left):
            return Qt.CursorShape.SizeBDiagCursor
        if has_left or has_right:
            return Qt.CursorShape.SizeHorCursor
        if has_top or has_bottom:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def _start_system_resize(self, edges: Qt.Edge) -> bool:
        """Begin native frameless resize when the pointer is on a window edge."""
        if edges == Qt.Edge(0):
            return False
        if self.isMaximized() or self.isFullScreen():
            return False

        handle = self.windowHandle()
        if handle is None or not hasattr(handle, "startSystemResize"):
            return False

        try:
            return bool(handle.startSystemResize(edges))
        except RuntimeError:
            return False

    def _update_resize_cursor(self, global_pos: QPoint) -> None:
        """Update cursor to communicate available edge/corner resize."""
        local = self.mapFromGlobal(global_pos)
        edges = self._resize_edges_from_local_pos(local)
        shape = self._cursor_for_resize_edges(edges)

        if shape == Qt.CursorShape.ArrowCursor:
            self.unsetCursor()
        else:
            self.setCursor(shape)

    def eventFilter(self, obj, event):
        """Handle frameless resize from child widgets and the main surface."""
        if isinstance(obj, QWidget) and obj.window() is self:
            event_type = event.type()

            if event_type == QEvent.Type.MouseMove and hasattr(event, "globalPosition"):
                if not (event.buttons() & Qt.MouseButton.LeftButton):
                    self._update_resize_cursor(event.globalPosition().toPoint())

            elif event_type == QEvent.Type.MouseButtonPress and hasattr(event, "globalPosition"):
                if event.button() == Qt.MouseButton.LeftButton:
                    local = self.mapFromGlobal(event.globalPosition().toPoint())
                    edges = self._resize_edges_from_local_pos(local)
                    if edges != Qt.Edge(0) and self._start_system_resize(edges):
                        return True

            elif event_type == QEvent.Type.Leave:
                self.unsetCursor()

        return super().eventFilter(obj, event)

    def _sync_title_bar_state(self) -> None:
        """Keep frameless title bar controls and outer shell spacing in sync."""
        if hasattr(self, "title_bar"):
            self.title_bar.sync_window_state()

        if not hasattr(self, "_root_layout"):
            return

        if self.isMaximized():
            self._root_layout.setContentsMargins(0, 0, 0, 0)
            self._root_layout.setSpacing(0)
        else:
            self._root_layout.setContentsMargins(10, 10, 10, 10)
            self._root_layout.setSpacing(10)

    def resizeEvent(self, event):
        """Keep layout stable and readable as the window size changes."""
        super().resizeEvent(event)
        self._sync_responsive_layout()

    def closeEvent(self, event):
        """Handle window close."""
        self._closing = True
        self._hide_journey_pip()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)

        self._stop_tracking()
        self.stop_focus_audio()
        # Close vision pipeline
        if hasattr(self, 'vision_pipeline') and self.vision_pipeline:
            self.vision_pipeline.close()
        if hasattr(self, 'phone_detector') and self.phone_detector:
            self.phone_detector.release()
        event.accept()

    def changeEvent(self, event):
        """Handle minimize-to-tray behavior and preserve stable window state transitions."""
        super().changeEvent(event)
        if event.type() != QEvent.Type.WindowStateChange:
            return

        self._sync_title_bar_state()
        QTimer.singleShot(0, self._sync_journey_pip_visibility)

    def hideEvent(self, event):
        """Show Journey PiP when the main window is hidden to tray during a session."""
        super().hideEvent(event)
        QTimer.singleShot(0, self._sync_journey_pip_visibility)

    def showEvent(self, event):
        """Hide Journey PiP when the main window is visible again."""
        super().showEvent(event)
        self._journey_pip_closed_until_restore = False
        QTimer.singleShot(0, self._sync_journey_pip_visibility)
