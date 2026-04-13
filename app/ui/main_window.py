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
from typing import Optional, TYPE_CHECKING, Dict, Any

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize, QVariantAnimation, QPointF, QEvent, QRectF
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QConicalGradient, QBrush, QIcon
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QProgressBar,
    QSizePolicy, QScrollArea,
    QMessageBox, QGraphicsDropShadowEffect, QDialog, QStyle, QGraphicsOpacityEffect
)

import cv2
import numpy as np

from ..logic.focus_engine import FocusState, FocusEngine, FrameFeatures
from ..logic.session_analytics import SessionAnalyticsStore
from ..utils.win_idle import get_idle_seconds
from .theme import get_stylesheet

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
    FocusState.ON_SCREEN_READING: "Đang tập trung",
    FocusState.OFFSCREEN_WRITING: "Đang ghi chép",
    FocusState.PHONE_DISTRACTION: "Mất tập trung",
    FocusState.DROWSY_FATIGUE: "Mệt mỏi / Buồn ngủ",
    FocusState.AWAY: "Không có mặt",
    FocusState.UNCERTAIN: "Không xác định",
}

# OpenCV text rendering does not support Vietnamese diacritics reliably.
OVERLAY_STATE_NAMES = {
    FocusState.ON_SCREEN_READING: "Dang tap trung",
    FocusState.OFFSCREEN_WRITING: "Dang ghi chep",
    FocusState.PHONE_DISTRACTION: "Mat tap trung",
    FocusState.DROWSY_FATIGUE: "Met moi / Buon ngu",
    FocusState.AWAY: "Khong co mat",
    FocusState.UNCERTAIN: "Khong xac dinh",
}

STATE_ICONS = {
    FocusState.ON_SCREEN_READING: "•",
    FocusState.OFFSCREEN_WRITING: "•",
    FocusState.PHONE_DISTRACTION: "•",
    FocusState.DROWSY_FATIGUE: "•",
    FocusState.AWAY: "•",
    FocusState.UNCERTAIN: "•",
}


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
    """Compact strip for stream and model runtime statuses."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statusStrip")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        self.values: dict[str, QLabel] = {}
        self._add_chip(layout, "stream", "Luồng", "Disconnected")
        self._add_chip(layout, "face", "Khuôn mặt", "No face")
        self._add_chip(layout, "lighting", "Ánh sáng", "Unknown")
        self._add_chip(layout, "model", "Mô hình", "Initializing")

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

    def set_status(self, stream: str, face: str, lighting: str, model: str) -> None:
        """Refresh runtime statuses shown in the strip."""
        self.values["stream"].setText(stream)
        self.values["face"].setText(face)
        self.values["lighting"].setText(lighting)
        self.values["model"].setText(model)

        stream_lower = stream.lower()
        if stream_lower == "live":
            stream_color = "#7ef4d4"
        elif stream_lower == "paused":
            stream_color = "#ffe1a0"
        else:
            stream_color = "#f7b3b3"

        self.values["stream"].setStyleSheet(f"color: {stream_color}; font-weight: 700;")

class FocusScoreWidget(QFrame):
    """Circular widget showing focus score."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_dark = True
        self.score = 100.0
        self._target_score = 100.0
        self.state = FocusState.UNCERTAIN
        self.setMinimumSize(180, 180)
        self.setMaximumSize(240, 240)
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
            self._shadow.setColor(QColor(89, 213, 192, 72))
            self._shadow.setOffset(0, 2)
        elif self.state in (FocusState.PHONE_DISTRACTION, FocusState.DROWSY_FATIGUE) or self._target_score < 58:
            self._shadow.setBlurRadius(18)
            self._shadow.setColor(QColor(239, 157, 149, 58))
            self._shadow.setOffset(0, 2)
        else:
            self._shadow.setBlurRadius(14)
            self._shadow.setColor(QColor(10, 20, 34, 72))
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

        # Draw inner base for a refined ring look.
        inner_color = QColor("#102031") if self.is_dark else QColor("#eef3f8")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(inner_color)
        painter.drawEllipse(x + 22, y + 22, size - 44, size - 44)

        # Draw track (background arc)
        track_color = "#2a3a4c" if self.is_dark else "#d4d4d8"
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
        self.text_color = QColor("#fafafa") if is_dark else QColor("#18181b")
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
        center = rect.center()
        max_radius = min(rect.width(), rect.height()) * 0.42
        radius = (max_radius * 0.55) + (max_radius * 0.35 * self._phase)

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
        """Draw a minimal trend chart suitable for dense dark-mode cards."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        chart_rect = self.rect().adjusted(8, 8, -8, -8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#111f2f"))
        painter.drawRoundedRect(chart_rect, 10, 10)

        if len(self._values) < 2:
            painter.setPen(QColor("#7b8aa0"))
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

        mid_pen = QPen(QColor("#2c3f56"))
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
            line_color = QColor("#e9b16e")
        elif slope >= 6:
            line_color = QColor("#74d8c6")
        else:
            line_color = QColor("#8db1ff")

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

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(12, 20, 34, 40))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Khuyến nghị hiện tại")
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
            "Bật theo dõi để hệ thống nhận diện nhịp tập trung và gợi ý nghỉ đúng lúc."
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

        badge_styles = {
            "good": ("#17372f", "#8ff5dd", "#285a4e"),
            "watch": ("#3a2d14", "#ffd38a", "#6f5328"),
            "break": ("#462218", "#ffbea7", "#7b3b2d"),
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

        title = QLabel("Insight nhanh")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        trend_header = QHBoxLayout()
        trend_header.setContentsMargins(0, 0, 0, 0)
        trend_header.setSpacing(8)
        trend_label = QLabel("Xu hướng tập trung")
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

        self.insight_note = QLabel("Hệ thống đang xây dựng baseline xu hướng tập trung.")
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
            note = "Nhịp tập trung đang đi lên. Đây là thời điểm tốt để xử lý tác vụ quan trọng."
        else:
            note = "Theo dõi xu hướng để nghỉ đúng nhịp, giữ hiệu quả ổn định suốt phiên làm việc."

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statsCard")
        self.setProperty("summaryCard", True)

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

        rows = QVBoxLayout()
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(6)

        self.labels = {}
        stats = [
            ("session_time", "Thời gian phiên", "00:00:00", "◷"),
            ("focus_time", "Thời gian tập trung", "00:00:00", "◎"),
            ("distraction_count", "Số lần mất tập trung", "0", "!"),
            ("break_count", "Số lần nghỉ", "0", "◌"),
            ("avg_score", "Điểm trung bình", "0", "◉"),
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

    def apply_theme(self, is_dark: bool):
        """Kept for backward compatibility with legacy calls."""
        _ = is_dark

    def update_stats(self, stats: dict):
        """Update displayed statistics."""
        for key, value in stats.items():
            if key in self.labels:
                self.labels[key].setText(str(value))

class MainWindow(QMainWindow):
    """Main application window for FocusGuardian."""

    # Signals
    state_changed = pyqtSignal(FocusState)
    score_changed = pyqtSignal(float)
    break_suggested = pyqtSignal()
    config_changed = pyqtSignal(dict)

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.config = config or {}
        self.config["theme_mode"] = "dark"
        self.config["show_overlay"] = False

        # Session analytics and personalization
        self.analytics_store = SessionAnalyticsStore()
        self.profile_name = self._get_profile_name()
        self.session_started_at: Optional[float] = None
        self.state_time_by_state: Dict[str, float] = {
            state.name: 0.0 for state in FocusState
        }
        self._last_recommendation: Dict[str, Any] = {}
        self.focus_trend_samples: list[float] = []

        # Initialize components
        self._init_vision()
        self._init_engine()
        self._init_ui()
        self._init_timers()

        # Session tracking
        self.session_time_seconds = 0
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

        # Startup calibration and score smoothing to avoid early score drops.
        self._analysis_warmup_seconds = max(3.0, float(self.config.get("analysis_warmup_seconds", 12.0)))
        self._analysis_started_at = 0.0
        self._display_score = 100.0
        self._score_drop_speed_per_sec = max(1.0, float(self.config.get("score_drop_speed_per_sec", 2.6)))
        self._score_rise_speed_per_sec = max(
            self._score_drop_speed_per_sec,
            float(self.config.get("score_rise_speed_per_sec", 10.0)),
        )

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
            fps = int(self.config.get("fps", 30))
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

            phone_enabled = bool(self.config.get("enable_phone_detection", True))
            phone_mode = str(self.config.get("phone_detection_mode", "heuristic"))
            phone_conf_threshold = float(self.config.get("phone_confidence_threshold", 0.55))

            self.phone_detector = PhoneDetector(
                PhoneDetectorConfig(
                    enabled=phone_enabled,
                    model_type=phone_mode,
                    confidence_threshold=phone_conf_threshold,
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
                    )
                )
                self.phone_detector.initialize()

            self.vision_available = True
            logger.info("Vision modules initialized successfully")

        except Exception as e:
            logger.warning(f"Vision modules not available: {e}")
            self.camera = None
            self.vision_pipeline = None
            self.phone_detector = None
            self.vision_available = False

        self.camera_running = False

    def _init_engine(self):
        """Initialize focus engine."""
        self.engine = FocusEngine()
        self._apply_focus_engine_config()
        self.current_state = FocusState.UNCERTAIN
        self.current_score = 100.0

    def _apply_focus_engine_config(self) -> None:
        """Apply UI/config threshold values to FocusEngine runtime config."""
        cfg = self.engine.config
        cfg.head_down_pitch_threshold = float(self.config.get("head_down_threshold", cfg.head_down_pitch_threshold))
        cfg.head_away_yaw_threshold = float(self.config.get("look_away_threshold", cfg.head_away_yaw_threshold))
        cfg.write_score_threshold = float(self.config.get("write_score_threshold", cfg.write_score_threshold))
        cfg.drowsy_closure_ratio = float(self.config.get("eye_closure_threshold", cfg.drowsy_closure_ratio))
        cfg.drowsy_ear_threshold = float(self.config.get("ear_threshold", cfg.drowsy_ear_threshold))

        # Eye-gaze and head-down disambiguation controls.
        cfg.eye_look_down_threshold = float(self.config.get("eye_look_down_threshold", cfg.eye_look_down_threshold))
        cfg.eye_look_up_threshold = float(self.config.get("eye_look_up_threshold", cfg.eye_look_up_threshold))
        cfg.phone_eye_down_min_duration = float(self.config.get("phone_eye_down_min_duration", cfg.phone_eye_down_min_duration))
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
        cfg.score_recover_rate = max(1.0, float(self.config.get("score_recover_rate", cfg.score_recover_rate)))
        cfg.score_drop_rate = max(1.0, float(self.config.get("score_drop_rate", cfg.score_drop_rate)))

    def _init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("FocusGuardian")
        self.setMinimumSize(1060, 680)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(18)
        main_layout.setContentsMargins(18, 18, 18, 18)

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

        title_block = QVBoxLayout()
        title_block.setContentsMargins(0, 0, 0, 0)
        title_block.setSpacing(4)

        hero_title = QLabel("Theo dõi tập trung")
        hero_title.setObjectName("heroTitle")
        title_block.addWidget(hero_title)

        hero_subtitle = QLabel(
            "Giám sát trạng thái chú ý theo thời gian thực và đề xuất nghỉ hợp lý"
        )
        hero_subtitle.setObjectName("heroSubtitle")
        hero_subtitle.setWordWrap(True)
        title_block.addWidget(hero_subtitle)

        header_row.addLayout(title_block, 1)

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
        right_panel.addStretch(1)

        # Focus score card
        score_container = QFrame()
        score_container.setObjectName("scoreCard")
        score_container.setProperty("summaryCard", True)
        score_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        score_shadow = QGraphicsDropShadowEffect(score_container)
        score_shadow.setBlurRadius(12)
        score_shadow.setColor(QColor(12, 20, 34, 40))
        score_shadow.setOffset(0, 2)
        score_container.setGraphicsEffect(score_shadow)
        score_layout = QVBoxLayout(score_container)
        score_layout.setContentsMargins(16, 16, 16, 16)
        score_layout.setSpacing(10)

        score_title = QLabel("Điểm tập trung")
        score_title.setObjectName("sectionTitle")
        score_layout.addWidget(score_title)
        self.score_widget = FocusScoreWidget()
        score_layout.addWidget(self.score_widget, 0, Qt.AlignmentFlag.AlignCenter)

        self.state_badge = QLabel("Không xác định")
        self.state_badge.setObjectName("stateBadge")
        self.state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_layout.addWidget(self.state_badge, 0, Qt.AlignmentFlag.AlignCenter)

        self.state_hint = QLabel("Sẵn sàng bắt đầu phiên mới.")
        self.state_hint.setObjectName("mutedLabel")
        self.state_hint.setWordWrap(True)
        self.state_hint.setMinimumHeight(44)
        self.state_hint.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.state_hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._state_hint_opacity = QGraphicsOpacityEffect(self.state_hint)
        self._state_hint_opacity.setOpacity(1.0)
        self.state_hint.setGraphicsEffect(self._state_hint_opacity)
        self._state_hint_fade = QVariantAnimation(self)
        self._state_hint_fade.setDuration(220)
        self._state_hint_fade.valueChanged.connect(self._fade_state_hint)
        score_layout.addWidget(self.state_hint)

        right_panel.addWidget(score_container)

        # Current state / recommendation card
        self.guidance_widget = FocusGuidanceWidget()
        self.guidance_widget.setProperty("summaryCard", True)
        self.guidance_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_panel.addWidget(self.guidance_widget)

        # Session statistics card
        self.stats_widget = StatsWidget()
        self.stats_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_panel.addWidget(self.stats_widget)

        # Mini trend / insight card
        self.trend_widget = TrendInsightWidget()
        self.trend_widget.setProperty("summaryCard", True)
        self.trend_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_panel.addWidget(self.trend_widget)

        right_panel.addStretch(1)

        self.right_column_scroll.setWidget(right_host)

        main_layout.addWidget(self.right_column_scroll, 36)
        self._main_layout = main_layout

        self._apply_theme()
        self._update_state_badge(FocusState.UNCERTAIN, 0.0, "Sẵn sàng bắt đầu phiên mới.")
        self._update_live_status(face_detected=None, lighting="Unknown")
        self._refresh_focus_guidance()
        self._sync_responsive_layout()

    def _apply_theme(self):
        self.config["theme_mode"] = "dark"
        self.setStyleSheet(get_stylesheet(True))
        self._set_settings_button_icon()
        self.score_widget.update_theme(True)
        self.stats_widget.apply_theme(True)
        if hasattr(self, "state_badge") and hasattr(self, "state_hint"):
            self._update_state_badge(self.current_state, 0.0, self.state_hint.text())
        if hasattr(self, "guidance_widget"):
            self._refresh_focus_guidance()

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
        # Frame processing timer (30 FPS)
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._process_frame)
        self.frame_interval = 33  # ~30 FPS

        # Stats update timer (1 second)
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_stats)

    def _get_profile_name(self) -> str:
        """Return active profile name from config."""
        profile_name = str(self.config.get("profile_name", "default")).strip()
        return profile_name or "default"

    def _reset_session_tracking(self) -> None:
        """Reset counters at the beginning of a tracking session."""
        self.session_time_seconds = 0
        self.focus_time = 0.0
        self.distraction_count = 0
        self.break_count = 0
        self.score_samples = []
        self.focus_trend_samples = []
        self.current_state = FocusState.UNCERTAIN
        self.current_score = 100.0
        self._display_score = 100.0
        self.continuous_focus_time = 0.0
        self.state_time_by_state = {state.name: 0.0 for state in FocusState}

    def _is_initial_analysis_phase(self) -> bool:
        """Return True while the startup calibration window is still active."""
        if not self.camera_running or self._analysis_started_at <= 0.0:
            return False
        return (time.time() - self._analysis_started_at) < self._analysis_warmup_seconds

    def _analysis_seconds_left(self) -> int:
        """Return rounded-up remaining warmup seconds."""
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

        if not self.config.get("enable_personalization", True):
            return

        try:
            recommendation = self.analytics_store.get_recommendation(
                self.profile_name,
                default_work=int(self.config.get("break_interval_minutes", 25)),
                default_break=int(self.config.get("break_duration_minutes", 5)),
            )
        except Exception as exc:
            logger.warning("Failed to load personalized schedule: %s", exc)
            return

        self._last_recommendation = recommendation

        if self.config.get("auto_apply_personalization", True):
            self.config["break_interval_minutes"] = int(recommendation.get("work_minutes", 25))
            self.config["break_duration_minutes"] = int(recommendation.get("break_minutes", 5))
            self.config_changed.emit(self.config.copy())

        logger.info(
            "Personalized plan for profile '%s': work=%s min, break=%s min (%s)",
            self.profile_name,
            recommendation.get("work_minutes", 25),
            recommendation.get("break_minutes", 5),
            recommendation.get("reason", "n/a"),
        )

    def _persist_session_analytics(self) -> None:
        """Persist current session data for later analysis and personalization."""
        if self.session_started_at is None:
            return

        session_seconds = max(
            int(time.time() - self.session_started_at),
            int(self.session_time_seconds),
        )
        self.session_started_at = None

        # Ignore too-short runs to avoid noisy personalization.
        if session_seconds < 30:
            return

        avg_score = float(sum(self.score_samples) / len(self.score_samples)) if self.score_samples else float(self.current_score)
        session_record = {
            "timestamp": int(time.time()),
            "profile_name": self.profile_name,
            "session_seconds": session_seconds,
            "focus_seconds": float(self.focus_time),
            "distraction_count": int(self.distraction_count),
            "break_count": int(self.break_count),
            "avg_score": avg_score,
            "min_score": float(min(self.score_samples)) if self.score_samples else float(self.current_score),
            "max_score": float(max(self.score_samples)) if self.score_samples else float(self.current_score),
            "state_seconds": self.state_time_by_state.copy(),
            "work_interval_minutes_used": int(self.config.get("break_interval_minutes", 25)),
            "break_duration_minutes_used": int(self.config.get("break_duration_minutes", 5)),
        }

        try:
            recommendation = self.analytics_store.record_session(
                self.profile_name,
                session_record,
                default_work=int(self.config.get("break_interval_minutes", 25)),
                default_break=int(self.config.get("break_duration_minutes", 5)),
            )
        except Exception as exc:
            logger.warning("Failed to persist session analytics: %s", exc)
            return

        self._last_recommendation = recommendation

        if self.config.get("enable_personalization", True) and self.config.get("auto_apply_personalization", True):
            self.config["break_interval_minutes"] = int(recommendation.get("work_minutes", 25))
            self.config["break_duration_minutes"] = int(recommendation.get("break_minutes", 5))
            self.config_changed.emit(self.config.copy())

        logger.info(
            "Saved session analytics for '%s': duration=%ss, avg_score=%.1f, rec=%s/%s min",
            self.profile_name,
            session_seconds,
            avg_score,
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
            QMessageBox.warning(
                self,
                "Vision không khả dụng",
                "Các module vision không được cài đặt đúng.\n"
                "Vui lòng sử dụng Python 3.10 hoặc 3.11 để chạy đầy đủ tính năng."
            )
            self.btn_start.setChecked(False)
            return

        try:
            if not self.camera.start():
                QMessageBox.warning(
                    self,
                    "Lỗi Camera",
                    "Không thể khởi động camera. Vui lòng kiểm tra kết nối."
                )
                self.btn_start.setChecked(False)
                return

            self._apply_personalized_schedule()
            self._reset_session_tracking()
            self.session_started_at = time.time()
            self._analysis_started_at = self.session_started_at
            self.last_break_time = self.session_started_at

            self.camera_running = True
            self.frame_timer.start(self.frame_interval)
            self.stats_timer.start(1000)  # Restart stats timer
            self.btn_start.setText("Dừng")
            self.engine.reset()
            self.score_widget.set_score(100.0, FocusState.UNCERTAIN)
            self._update_state_badge(FocusState.UNCERTAIN, 0.0, "Đang theo dõi phiên học...")
            self._update_live_status(face_detected=None, lighting="Calibrating")
            self._refresh_focus_guidance()

            logger.info("Focus tracking started")

        except Exception as e:
            logger.error(f"Failed to start tracking: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể bắt đầu: {e}")
            self.btn_start.setChecked(False)
            self._update_live_status(face_detected=False, lighting="Unknown")

    def _stop_tracking(self):
        """Stop camera and focus tracking."""
        was_running = self.camera_running
        self.frame_timer.stop()
        self.stats_timer.stop()  # Stop stats update timer
        if self.camera is not None:
            self.camera.stop()
        self.camera_running = False
        self._analysis_started_at = 0.0
        self._display_score = 100.0
        self.btn_start.setText("Bắt đầu")
        self.camera_widget.update_frame(None)
        self._update_state_badge(FocusState.UNCERTAIN, 0.0, "Đã dừng theo dõi.\nNhấn Bắt đầu để chạy lại.")
        self._update_live_status(face_detected=False, lighting="Unknown")
        self._refresh_focus_guidance()

        if was_running:
            self._persist_session_analytics()

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
        timestamp_ms = int(timestamp * 1000)

        try:
            # Process through unified vision pipeline
            vision_result = self.vision_pipeline.process(frame, timestamp_ms)

            # Extract features from vision result
            face_detected = vision_result.face_detected

            head_pitch, head_yaw, head_roll = None, None, None
            ear_avg, is_eye_closed, blink_detected = None, False, False
            eye_look_down, eye_look_up = None, None
            hand_present, hand_write_score, hand_region = False, 0.0, "none"

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

            if vision_result.hand_metrics:
                hand_present = vision_result.hand_metrics.detected
                hand_write_score = vision_result.hand_metrics.write_score
                hand_region = vision_result.hand_metrics.region

            phone_present = False
            if self.phone_detector is not None:
                phone_state = self.phone_detector.process(frame)
                phone_present = phone_state.phone_present

            lighting_quality = self._estimate_lighting_quality(frame)

            # System idle
            idle_seconds = get_idle_seconds()

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
            )

            # Process through engine
            state = self.engine.process_frame(features)
            score = self.engine.focus_score
            state_info = self.engine.get_state_info()
            state_confidence = float(state_info.get("confidence", 0.0))
            state_reason = str(state_info.get("reason", ""))

            # Update UI
            self._update_state(state, score, state_confidence, state_reason)

            # Draw overlays on frame
            display_frame = self._draw_overlays(
                frame,
                features,
                state,
                state_confidence,
                state_reason,
            )
            self.camera_widget.update_frame(display_frame)
            self._update_live_status(face_detected=face_detected, lighting=lighting_quality)

            # Check for break suggestion
            self._check_break_suggestion(state)

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self._update_live_status(face_detected=False, lighting="Unknown")

    def _draw_overlays(self, frame: np.ndarray,
                       features: FrameFeatures,
                       state: FocusState,
                       state_confidence: float = 0.0,
                       state_reason: str = "") -> np.ndarray:
        """Draw a compact, non-technical live guidance overlay."""
        display = frame.copy()

        # User preference: keep camera feed clean without top-left text panel.
        if not bool(self.config.get("show_overlay", False)):
            return display

        if self._is_initial_analysis_phase():
            state = FocusState.UNCERTAIN
            state_confidence = 0.0
            state_reason = "Đang hiệu chỉnh dữ liệu ban đầu"

        # Draw subtle state-colored frame
        color = STATE_COLORS.get(state, "#607D8B")
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        cv2.rectangle(display, (0, 0), (display.shape[1]-1, display.shape[0]-1),
                      (b, g, r), 2)

        break_interval_seconds = max(60, int(float(self.config.get("break_interval_minutes", 25)) * 60))
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

        if self._is_initial_analysis_phase():
            chip_text = "Đang hiệu chỉnh"
            chip_style = "background:rgba(127, 147, 170, 0.16); color:#d9e5f5; border:1px solid rgba(127, 147, 170, 0.28);"
            seconds_left = self._analysis_seconds_left()
            hint_text = f"Giữ điểm ổn định trong {seconds_left}s để hệ thống lấy baseline ban đầu."
        elif not self.camera_running or (state == FocusState.UNCERTAIN and len(self.focus_trend_samples) < 10):
            chip_text = "Chưa phân tích"
            chip_style = "background:rgba(127, 147, 170, 0.16); color:#d9e5f5; border:1px solid rgba(127, 147, 170, 0.28);"
            hint_text = "Đang thu thập dữ liệu để đánh giá xu hướng tập trung."
        elif self._is_distraction_state(state) or score_now < 58:
            chip_text = "Cần nghỉ"
            chip_style = "background:rgba(239, 157, 149, 0.18); color:#ffd6d0; border:1px solid rgba(239, 157, 149, 0.30);"
            hint_text = "Dấu hiệu quá tải chú ý. Nên nghỉ ngắn 2-3 phút để hồi phục."
        elif trend_delta <= -3.5 or score_now < 76:
            chip_text = "Mất tập trung nhẹ"
            chip_style = "background:rgba(239, 189, 120, 0.18); color:#ffe3b5; border:1px solid rgba(239, 189, 120, 0.30);"
            delta_points = max(1, int(abs(trend_delta)))
            hint_text = f"Giảm {delta_points} điểm so với xu hướng gần nhất."
        else:
            chip_text = "Ổn định"
            chip_style = "background:rgba(89, 213, 192, 0.18); color:#c6f8ee; border:1px solid rgba(89, 213, 192, 0.30);"
            stable_minutes = max(1, int(self.continuous_focus_time // 60))
            hint_text = f"Ổn định trong {stable_minutes} phút gần đây."

        self.state_badge.setText(chip_text)
        self.state_badge.setStyleSheet(
            "border-radius: 999px; padding: 5px 12px; font-weight: 650;"
            + chip_style
        )

        state_name = STATE_NAMES.get(state, state.name)
        self.state_hint.setText(hint_text)
        self.state_hint.setToolTip(reason or state_name)
        if hasattr(self, "_state_hint_fade"):
            self._state_hint_fade.stop()
            self._state_hint_fade.setStartValue(0.65)
            self._state_hint_fade.setEndValue(1.0)
            self._state_hint_fade.start()

    def _fade_state_hint(self, value) -> None:
        """Subtle fade for score insight text transitions."""
        if not hasattr(self, "_state_hint_opacity"):
            return
        try:
            opacity = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            opacity = 1.0
        self._state_hint_opacity.setOpacity(opacity)

    def _update_state(self, state: FocusState, score: float,
                      state_confidence: float = 0.0,
                      state_reason: str = ""):
        """Update UI with new state and score."""
        in_warmup = self._is_initial_analysis_phase()
        effective_state = FocusState.UNCERTAIN if in_warmup else state
        effective_score = self._compute_display_score(score)
        effective_confidence = 0.0 if in_warmup else state_confidence
        effective_reason = "Đang hiệu chỉnh dữ liệu ban đầu" if in_warmup else state_reason

        # Track state changes
        state_changed = effective_state != self.current_state
        if state_changed:
            previous_state = self.current_state

            if effective_state in (FocusState.PHONE_DISTRACTION, FocusState.DROWSY_FATIGUE):
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

        self.current_state = effective_state
        self.current_score = effective_score
        self._update_state_badge(effective_state, effective_confidence, effective_reason)

        # Update widgets
        self.score_widget.set_score(effective_score, effective_state)
        if not in_warmup:
            self.score_samples.append(effective_score)

        frame_seconds = self.frame_interval / 1000
        if not in_warmup:
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

        self.score_changed.emit(effective_score)

    @staticmethod
    def _is_focused_state(state: FocusState) -> bool:
        return state in (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING)

    @staticmethod
    def _is_distraction_state(state: FocusState) -> bool:
        return state in (FocusState.PHONE_DISTRACTION, FocusState.DROWSY_FATIGUE)

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

    def _update_live_status(self, face_detected: Optional[bool], lighting: str) -> None:
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

        model_status = "Ready" if self.vision_available and self.vision_pipeline is not None else "Limited"
        self.live_status_strip.set_status(
            stream=stream_status,
            face=face_status,
            lighting=lighting,
            model=model_status,
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

        break_minutes = int(self.config.get("break_duration_minutes", 5))
        state_name = STATE_NAMES.get(state, state.name)
        QMessageBox.information(
            self,
            "Nhắc nghỉ phục hồi tập trung",
            (
                f"Phát hiện bạn bắt đầu mất tập trung ({state_name}).\n"
                f"Hãy nghỉ {break_minutes} phút và thực hiện bài phục hồi tập trung."
            ),
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

        if self.camera_running and self._is_initial_analysis_phase():
            seconds_left = self._analysis_seconds_left()
            self.guidance_widget.set_guidance(
                mode="good",
                decision="Đang hiệu chỉnh ban đầu",
                detail=f"Giữ điểm 100 trong {seconds_left}s đầu để ổn định rồi mới đánh giá chính xác.",
                state_text="Đang thu thập baseline",
            )
            if hasattr(self, "trend_widget"):
                self.trend_widget.set_insight(
                    trend_text="Đang hiệu chỉnh",
                    trend_color="#9fd6ff",
                    cycle_percent=0,
                    trend_values=[],
                )
            return

        trend_delta = self._compute_focus_trend_delta()
        if trend_delta <= -7:
            trend_text = "Đang giảm rõ"
            trend_color = "#f6c177"
        elif trend_delta <= -3:
            trend_text = "Giảm nhẹ"
            trend_color = "#ffde95"
        elif trend_delta >= 5:
            trend_text = "Đang phục hồi"
            trend_color = "#8ff5dd"
        else:
            trend_text = "Ổn định"
            trend_color = "#9fd6ff"

        break_interval_seconds = max(
            60,
            int(float(self.config.get("break_interval_minutes", 25)) * 60),
        )
        elapsed_since_break = 0.0
        if self.camera_running:
            elapsed_since_break = max(0.0, time.time() - self.last_break_time)

        cycle_percent = int(min(100.0, (elapsed_since_break / break_interval_seconds) * 100.0))
        score_now = float(self.current_score)

        if not self.camera_running:
            mode = "good"
            decision = "Bạn đang ở trạng thái chờ"
            detail = "Nhấn Bắt đầu để hệ thống theo dõi và đưa gợi ý phục hồi theo thời gian thực."
        elif self._is_distraction_state(self.current_state):
            mode = "break"
            decision = "Nên nghỉ ngắn 2-3 phút"
            detail = "Có dấu hiệu giảm chú ý rõ. Nghỉ một nhịp ngắn để lấy lại trạng thái tập trung."
        elif cycle_percent >= 100 or (trend_delta <= -8 and score_now < 70):
            mode = "break"
            decision = "Nên nghỉ ngắn 2-3 phút"
            detail = "Bạn đã làm việc liên tục tương đối lâu. Nghỉ ngắn lúc này sẽ giúp duy trì hiệu suất tốt hơn."
        elif cycle_percent >= 75 or trend_delta <= -4 or score_now < 70:
            mode = "watch"
            decision = "Có dấu hiệu giảm chú ý"
            detail = "Bạn có thể hoàn tất phần đang làm, sau đó nghỉ ngắn để tránh hụt năng lượng tập trung."
        else:
            mode = "good"
            decision = "Bạn đang tập trung tốt"
            detail = "Trạng thái hiện tại ổn định. Hãy giữ nhịp làm việc hiện tại để đạt hiệu quả cao."

        self.guidance_widget.set_guidance(
            mode=mode,
            decision=decision,
            detail=detail,
            state_text=STATE_NAMES.get(self.current_state, self.current_state.name),
        )

        if hasattr(self, "trend_widget"):
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

        break_interval = self.config.get("break_interval_minutes", 25) * 60

        if self.continuous_focus_time >= break_interval:
            self.break_suggested.emit()
            self.continuous_focus_time = 0
            self.last_break_time = time.time()

    @pyqtSlot()
    def _update_stats(self):
        """Update statistics display."""
        if self.camera_running:
            self.session_time_seconds += 1
            self._record_focus_sample(self.current_score)

        avg_score_text = "0"
        if self.score_samples:
            avg_score_text = f"{sum(self.score_samples) / len(self.score_samples):.0f}"
        elif self.camera_running and self._is_initial_analysis_phase():
            avg_score_text = "100"

        stats = {
            "session_time": self._format_time(self.session_time_seconds),
            "focus_time": self._format_time(self.focus_time),
            "distraction_count": str(self.distraction_count),
            "break_count": str(self.break_count),
            "avg_score": avg_score_text,
        }

        self.stats_widget.update_stats(stats)
        self._refresh_focus_guidance()

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @pyqtSlot()
    def _take_break(self, auto_triggered: bool = False):
        """Handle break button click."""
        self.break_count += 1
        self.last_break_time = time.time()
        self.continuous_focus_time = 0

        should_resume = bool(self.config.get("auto_resume_after_break", True))
        was_tracking = self.camera_running

        if was_tracking:
            self.btn_start.setChecked(False)
            self._stop_tracking()

        self._break_dialog_open = True
        try:
            if not auto_triggered:
                overlay_seconds = int(self.config.get("break_overlay_seconds", 12))
                break_overlay = BreakModeDialog(duration_seconds=overlay_seconds, parent=self)
                break_overlay.exec()

            # Open the focused recovery workflow directly.
            self._open_games()
        finally:
            self._break_dialog_open = False

        if auto_triggered and was_tracking and should_resume:
            self.btn_start.setChecked(True)
            self._start_tracking()

    @pyqtSlot()
    def _open_settings(self):
        """Open settings dialog."""
        from .settings_dialog import SettingsDialog
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.config.update(dialog.get_config())
            self._apply_config()
            self.config_changed.emit(self.config.copy())

    def _apply_config(self):
        """Apply configuration changes."""
        self._apply_theme()
        self.profile_name = self._get_profile_name()
        self._apply_focus_engine_config()

        # Update camera if needed
        if self.vision_available and self.camera:
            from ..vision import CameraCapture, CameraConfig

            new_camera_id = int(self.config.get("camera_id", 0))
            width, height = self._parse_resolution(self.config.get("resolution", "640x480"))
            fps = int(self.config.get("fps", 30))

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

            if hasattr(self, "phone_detector") and self.phone_detector is not None:
                self.phone_detector.release()

            self.phone_detector = PhoneDetector(
                PhoneDetectorConfig(
                    enabled=phone_enabled,
                    model_type=phone_mode,
                    confidence_threshold=phone_conf_threshold,
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
                    )
                )
                self.phone_detector.initialize()

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
        from ..focus_reset_game.ui import FocusResetDialog

        recovery_dialog = FocusResetDialog(self)
        recovery_dialog.exec()  # Modal dialog

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

    def resizeEvent(self, event):
        """Keep layout stable and readable as the window size changes."""
        super().resizeEvent(event)
        self._sync_responsive_layout()

    def closeEvent(self, event):
        """Handle window close."""
        self._stop_tracking()
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

        if self.isMinimized() and bool(self.config.get("minimize_to_tray", True)):
            # Hide on the next event loop tick to avoid flicker/glitches on Windows.
            QTimer.singleShot(0, self.hide)
