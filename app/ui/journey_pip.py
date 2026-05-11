"""Small always-on-top Focus Journey picture-in-picture window."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBitmap, QColor, QBrush, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .journey_map_dialog import FallbackJourneyMapWidget, build_journey_model


class MiniJourneyMapCanvas(QWidget):
    """Tiny native map-style route preview for the PiP window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pipMapCanvas")
        self.setMinimumHeight(92)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._model: Dict[str, Any] = build_journey_model({})
        self._progress = 0.0
        self._phase = "Boarding"
        self._state_color = QColor("#7f93aa")
        self._is_dark = True

    def set_theme(self, is_dark: bool) -> None:
        next_value = bool(is_dark)
        if self._is_dark == next_value:
            return
        self._is_dark = next_value
        self.update()

    def set_data(
        self,
        payload: Dict[str, Any] | None,
        progress: float,
        phase: str,
        state_color: str,
        hold_motion: bool = False,
    ) -> None:
        self._model = build_journey_model(payload or {})
        self._progress = max(0.0, min(1.0, float(progress or 0.0)))
        self._phase = str(phase or "Boarding")
        self._state_color = QColor(str(state_color or "#7f93aa"))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        if rect.width() <= 4 or rect.height() <= 4:
            return

        self._draw_background(painter, rect)
        points = self._project_route(rect.adjusted(12, 10, -12, -14))
        if len(points) >= 2:
            self._draw_route(painter, points)
            self._draw_airport(painter, points[0], str(self._model.get("from_code") or "---"), start=True)
            self._draw_airport(painter, points[-1], str(self._model.get("to_code") or "---"), start=False)
            plane = self._point_at_progress(points, self._progress)
            next_plane = self._point_at_progress(points, min(1.0, self._progress + 0.018))
            self._draw_plane(painter, plane, next_plane)

        painter.setPen(QPen(QColor(236, 246, 255, 205) if self._is_dark else QColor(20, 46, 67, 205)))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.drawText(
            rect.adjusted(10, rect.height() - 24, -10, -4),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            self._phase,
        )

    def _draw_background(self, painter: QPainter, rect: QRectF) -> None:
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        if self._is_dark:
            gradient.setColorAt(0.0, QColor("#152538"))
            gradient.setColorAt(0.56, QColor("#0e1a29"))
            gradient.setColorAt(1.0, QColor("#0a1421"))
            grid = QColor(126, 165, 196, 28)
            land = QColor(78, 113, 135, 36)
            border = QColor(125, 164, 199, 56)
        else:
            gradient.setColorAt(0.0, QColor("#eaf5fd"))
            gradient.setColorAt(0.56, QColor("#d9ebf7"))
            gradient.setColorAt(1.0, QColor("#cde1f0"))
            grid = QColor(67, 102, 132, 28)
            land = QColor(100, 149, 148, 34)
            border = QColor(75, 109, 138, 54)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(rect, 12, 12)

        painter.setPen(QPen(grid, 1))
        step = 28
        x = rect.left() + 16
        while x < rect.right() - 8:
            painter.drawLine(QPointF(x, rect.top() + 8), QPointF(x, rect.bottom() - 8))
            x += step
        y = rect.top() + 16
        while y < rect.bottom() - 8:
            painter.drawLine(QPointF(rect.left() + 8, y), QPointF(rect.right() - 8, y))
            y += step

        painter.setPen(QPen(land, 1.2))
        for idx, (cx, cy, w, h) in enumerate(
            (
                (0.24, 0.42, 0.34, 0.22),
                (0.62, 0.36, 0.42, 0.20),
                (0.50, 0.72, 0.50, 0.18),
            )
        ):
            blob = QRectF(
                rect.left() + rect.width() * cx - rect.width() * w / 2,
                rect.top() + rect.height() * cy - rect.height() * h / 2,
                rect.width() * w,
                rect.height() * h,
            )
            painter.drawEllipse(blob)
            if idx == 1:
                painter.drawEllipse(blob.adjusted(14, 8, -18, -10))

        painter.setPen(QPen(border, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 12, 12)

    def _project_route(self, rect: QRectF) -> List[QPointF]:
        raw = list(self._model.get("curve_points") or [])
        if len(raw) < 2:
            return []

        lats = [float(point[0]) for point in raw]
        lngs = [float(point[1]) for point in raw]
        min_lat, max_lat = min(lats), max(lats)
        min_lng, max_lng = min(lngs), max(lngs)
        lat_span = max(0.001, max_lat - min_lat)
        lng_span = max(0.001, max_lng - min_lng)

        # Keep short routes visually readable.
        if lat_span < lng_span * 0.35:
            center = (min_lat + max_lat) / 2.0
            lat_span = lng_span * 0.35
            min_lat = center - lat_span / 2.0
            max_lat = center + lat_span / 2.0
        if lng_span < lat_span * 0.45:
            center = (min_lng + max_lng) / 2.0
            lng_span = lat_span * 0.45
            min_lng = center - lng_span / 2.0
            max_lng = center + lng_span / 2.0

        points: List[QPointF] = []
        for lat, lng in zip(lats, lngs):
            x = rect.left() + ((lng - min_lng) / max(0.001, max_lng - min_lng)) * rect.width()
            y = rect.bottom() - ((lat - min_lat) / max(0.001, max_lat - min_lat)) * rect.height()
            points.append(QPointF(x, y))
        return points

    def _draw_route(self, painter: QPainter, points: List[QPointF]) -> None:
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)

        shadow = QColor(4, 13, 24, 150) if self._is_dark else QColor(255, 255, 255, 180)
        painter.setPen(QPen(shadow, 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(path)
        painter.setPen(QPen(QColor(142, 174, 204, 120), 2.2, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
        painter.drawPath(path)

        count = len(points)
        exact = max(0.0, min(1.0, self._progress)) * (count - 1)
        active_index = max(0, min(count - 2, int(math.floor(exact))))
        active = QPainterPath(points[0])
        for point in points[1 : active_index + 1]:
            active.lineTo(point)
        active.lineTo(self._point_at_progress(points, self._progress))
        painter.setPen(QPen(QColor("#59d5c0"), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPath(active)

    def _draw_airport(self, painter: QPainter, point: QPointF, code: str, *, start: bool) -> None:
        color = QColor("#9fb6ce") if start else QColor("#ffd15f")
        fill = QColor(14, 26, 40, 210) if self._is_dark else QColor(255, 255, 255, 225)
        painter.setPen(QPen(color, 1.2))
        painter.setBrush(fill)
        box = QRectF(point.x() - 18, point.y() - 11, 36, 22)
        painter.drawRoundedRect(box, 7, 7)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(color))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, code[:4])

    def _draw_plane(self, painter: QPainter, point: QPointF, next_point: QPointF) -> None:
        angle = math.degrees(math.atan2(next_point.y() - point.y(), next_point.x() - point.x()))
        halo = QColor(self._state_color)
        halo.setAlpha(58)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(halo)
        painter.drawEllipse(point, 18, 18)
        painter.save()
        painter.translate(point)
        painter.rotate(angle)
        painter.setBrush(QColor("#edf4fd") if self._is_dark else QColor("#123047"))
        painter.setPen(QPen(QColor("#0b1422") if self._is_dark else QColor("#ffffff"), 0.8))
        plane = QPainterPath()
        plane.moveTo(13, 0)
        plane.lineTo(-9, -6)
        plane.lineTo(-4, 0)
        plane.lineTo(-9, 6)
        plane.closeSubpath()
        painter.drawPath(plane)
        painter.restore()

    @staticmethod
    def _point_at_progress(points: List[QPointF], progress: float) -> QPointF:
        if not points:
            return QPointF()
        if len(points) == 1:
            return points[0]
        p = max(0.0, min(1.0, float(progress or 0.0)))
        raw = p * (len(points) - 1)
        index = max(0, min(len(points) - 2, int(math.floor(raw))))
        frac = raw - index
        a = points[index]
        b = points[index + 1]
        return QPointF(a.x() + (b.x() - a.x()) * frac, a.y() + (b.y() - a.y()) * frac)


class SatelliteJourneyPiPMapCanvas(FallbackJourneyMapWidget):
    """Compact satellite Journey renderer for PiP, without text overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pipSatelliteMap")
        self.setMinimumSize(1, 220)
        self.setMaximumHeight(230)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._last_payload_key: tuple = ()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_rounded_mask()

    def _update_rounded_mask(self) -> None:
        if self.width() <= 0 or self.height() <= 0:
            return
        mask = QBitmap(self.size())
        mask.fill(Qt.GlobalColor.color0)
        painter = QPainter(mask)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.GlobalColor.color1)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(self.rect()), 10, 10)
        painter.end()
        self.setMask(mask)

    def set_theme(self, is_dark: bool) -> None:
        _ = is_dark

    def set_data(
        self,
        payload: Dict[str, Any] | None,
        progress: float,
        phase: str,
        state_color: str,
        hold_motion: bool = False,
    ) -> None:
        data = dict(payload or {})
        key = (
            str(data.get("route_from_code") or data.get("from_code") or ""),
            str(data.get("route_to_code") or data.get("to_code") or ""),
            int(data.get("planned_minutes") or data.get("route_duration_minutes") or 0),
            int(data.get("route_distance_km") or data.get("distance_km") or 0),
        )
        if key != self._last_payload_key:
            self.set_journey_data(data)
            self._last_payload_key = key

        model = build_journey_model(data)
        safe_progress = max(0.0, min(1.0, float(progress or 0.0)))
        remaining = int(round(int(model.get("duration_minutes", 25) or 25) * 60 * (1.0 - safe_progress)))
        distance = int(round(int(model.get("distance_km", 0) or 0) * (1.0 - safe_progress)))
        self.update_progress(
            safe_progress,
            max(0, remaining),
            max(0, distance),
            str(phase or "Boarding"),
        )
        self.set_motion_paused(bool(hold_motion) or safe_progress <= 0.0001)
        self._update_rounded_mask()

    def _draw_overlays(self, painter: QPainter, rect: QRectF) -> None:
        _ = painter
        _ = rect

    def _draw_vignette(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        shade = QLinearGradient(rect.topLeft(), rect.bottomRight())
        shade.setColorAt(0.0, QColor(4, 12, 20, 18))
        shade.setColorAt(0.52, QColor(4, 12, 20, 0))
        shade.setColorAt(1.0, QColor(4, 12, 20, 88))
        painter.fillRect(rect, QBrush(shade))
        painter.restore()

    def _draw_airport(self, painter: QPainter, point: QPointF, code: str, color: QColor) -> None:
        painter.save()
        painter.setPen(QPen(color, 1.4))
        painter.setBrush(QColor(5, 13, 23, 210))
        box = QRectF(point.x() - 17, point.y() - 12, 34, 24)
        painter.drawRoundedRect(box, 8, 8)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(color)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, code[:4])
        painter.restore()

    def _draw_plane(self, painter: QPainter, point: QPointF, next_point: QPointF) -> None:
        angle = math.degrees(math.atan2(next_point.y() - point.y(), next_point.x() - point.x()))
        painter.save()
        painter.translate(point)
        painter.rotate(angle)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 46))
        painter.drawEllipse(QPointF(0, 0), 20, 20)
        painter.setBrush(QColor(99, 230, 216, 38))
        painter.drawEllipse(QPointF(0, 0), 14, 14)

        plane = QPainterPath()
        plane.moveTo(17, 0)
        plane.cubicTo(12, -2.4, 6, -3.4, 0, -3.2)
        plane.lineTo(-10, -12)
        plane.cubicTo(-12, -13.5, -14, -12.8, -13.2, -9.8)
        plane.lineTo(-8, -2.2)
        plane.lineTo(-17, -0.8)
        plane.cubicTo(-19.2, -0.5, -19.2, 0.5, -17, 0.8)
        plane.lineTo(-8, 2.2)
        plane.lineTo(-13.2, 9.8)
        plane.cubicTo(-14, 12.8, -12, 13.5, -10, 12)
        plane.lineTo(0, 3.2)
        plane.cubicTo(6, 3.4, 12, 2.4, 17, 0)
        plane.closeSubpath()
        painter.setPen(QPen(QColor(5, 13, 22, 210), 1.1))
        painter.setBrush(QColor("#f8ffff"))
        painter.drawPath(plane)
        painter.restore()


class FocusJourneyPiPWindow(QWidget):
    """Compact frameless Journey monitor shown while the main window is minimized."""

    openRequested = pyqtSignal()
    closeRequested = pyqtSignal()

    STATE_COLORS = {
        "ON_SCREEN_READING": "#59d5c0",
        "OFFSCREEN_WRITING": "#7ea9ff",
        "PHONE_DISTRACTION": "#f09d95",
        "DROWSY_FATIGUE": "#efbd78",
        "AWAY": "#8ea1b5",
        "UNCERTAIN": "#7f93aa",
    }

    STATE_LABELS = {
        "ON_SCREEN_READING": "Ổn định",
        "OFFSCREEN_WRITING": "Ghi chép",
        "PHONE_DISTRACTION": "Lệch nhiệm vụ",
        "DROWSY_FATIGUE": "Mệt nhẹ",
        "AWAY": "Vắng mặt",
        "UNCERTAIN": "Chưa rõ",
    }

    def __init__(self, theme_mode: str = "dark", parent=None):
        super().__init__(parent)
        self.setObjectName("focusJourneyPiP")
        self.setWindowTitle("Focus Journey")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowOpacity(0.96)
        self.setFixedSize(350, 350)

        self._drag_offset: QPoint | None = None
        self._user_moved = False
        self._theme_mode = str(theme_mode or "dark").strip().lower()

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(0)

        self.card = QFrame()
        self.card.setObjectName("pipCard")
        root_layout.addWidget(self.card)

        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(4, 10, 18, 90))
        self.card.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(12, 8, 12, 8)
        card_layout.setSpacing(5)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self.status_dot = QLabel()
        self.status_dot.setObjectName("pipStatusDot")
        self.status_dot.setFixedSize(10, 10)
        header.addWidget(self.status_dot, 0, Qt.AlignmentFlag.AlignVCenter)

        self.route_label = QLabel("--- → ---")
        self.route_label.setObjectName("pipRoute")
        self.route_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        header.addWidget(self.route_label, 1)

        self.hide_button = QToolButton()
        self.hide_button.setObjectName("pipIconButton")
        self.hide_button.setText("×")
        self.hide_button.setToolTip("Ẩn PiP trong phiên này")
        self.hide_button.setFixedSize(24, 24)
        self.hide_button.setToolTip("Tắt")
        self.hide_button.clicked.connect(lambda _checked=False: self.closeRequested.emit())
        header.addWidget(self.hide_button)
        card_layout.addLayout(header)

        self.map_canvas = SatelliteJourneyPiPMapCanvas()
        card_layout.addWidget(self.map_canvas, 1)

        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(10)

        self.progress_label = QLabel("0%")
        self.progress_label.setObjectName("pipProgressText")
        self.progress_label.setFixedWidth(72)
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        middle.addWidget(self.progress_label, 0, Qt.AlignmentFlag.AlignBottom)

        detail_col = QVBoxLayout()
        detail_col.setContentsMargins(0, 0, 0, 0)
        detail_col.setSpacing(0)
        detail_col.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self.phase_label = QLabel("Boarding")
        self.phase_label.setObjectName("pipPhase")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.remaining_label = QLabel("00:00 còn lại")
        self.remaining_label.setObjectName("pipRemainingLarge")
        self.remaining_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self.status_label = QLabel("Chưa rõ")
        self.status_label.setObjectName("pipMuted")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.phase_label.hide()
        self.status_label.hide()
        detail_col.addStretch(1)
        detail_col.addWidget(self.remaining_label)
        detail_col.addWidget(self.status_label)
        middle.addStretch(1)
        middle.addLayout(detail_col, 1)

        card_layout.addLayout(middle)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("pipProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(7)
        card_layout.addWidget(self.progress_bar)

        self.update_theme(self._theme_mode)
        self.update_data()

    def update_theme(self, theme_mode: str | bool = "dark") -> None:
        """Apply dark/light calm-tech styling."""
        if isinstance(theme_mode, bool):
            is_dark = bool(theme_mode)
        else:
            is_dark = str(theme_mode or "dark").strip().lower() != "light"
        self._theme_mode = "dark" if is_dark else "light"
        if hasattr(self, "map_canvas"):
            self.map_canvas.set_theme(is_dark)

        if is_dark:
            colors = {
                "bg": "rgba(15, 27, 41, 238)",
                "surface": "rgba(34, 54, 76, 185)",
                "surface_hover": "rgba(48, 76, 105, 210)",
                "border": "rgba(132, 165, 199, 0.28)",
                "text": "#edf4fd",
                "muted": "#a8bad0",
                "accent": "#59d5c0",
                "track": "rgba(120, 148, 177, 0.22)",
            }
        else:
            colors = {
                "bg": "rgba(255, 255, 255, 238)",
                "surface": "rgba(229, 240, 250, 205)",
                "surface_hover": "rgba(213, 231, 244, 230)",
                "border": "rgba(93, 124, 154, 0.26)",
                "text": "#182c41",
                "muted": "#435d76",
                "accent": "#2f9f90",
                "track": "rgba(111, 143, 172, 0.22)",
            }

        self.setStyleSheet(
            f"""
            QFrame#pipCard {{
                background-color: {colors['bg']};
                border: 1px solid {colors['border']};
                border-radius: 16px;
            }}
            QLabel#pipRoute {{
                color: {colors['text']};
                font-size: 15px;
                font-weight: 800;
            }}
            QLabel#pipProgressText {{
                color: {colors['text']};
                font-size: 28px;
                font-weight: 850;
            }}
            QLabel#pipPhase {{
                color: {colors['text']};
                font-size: 13px;
                font-weight: 750;
            }}
            QLabel#pipMuted {{
                color: {colors['muted']};
                font-size: 11px;
                font-weight: 600;
            }}
            QLabel#pipRemainingLarge {{
                color: {colors['text']};
                font-size: 24px;
                font-weight: 850;
            }}
            QToolButton#pipIconButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 12px;
                color: {colors['muted']};
                font-size: 16px;
                font-weight: 800;
            }}
            QToolButton#pipIconButton:hover {{
                background-color: {colors['surface']};
                border-color: {colors['border']};
                color: {colors['text']};
            }}
            QProgressBar#pipProgressBar {{
                background-color: {colors['track']};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar#pipProgressBar::chunk {{
                background-color: {colors['accent']};
                border-radius: 4px;
            }}
            """
        )

    def update_data(
        self,
        *,
        route_from_code: str = "",
        route_to_code: str = "",
        progress: float = 0.0,
        remaining_seconds: int = 0,
        phase: str = "Boarding",
        status_text: str = "",
        state: Any = None,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        """Refresh PiP labels from the current Focus Journey session."""
        from_code = str(route_from_code or "---").strip().upper()[:4]
        to_code = str(route_to_code or "---").strip().upper()[:4]

        try:
            progress_value = float(progress or 0.0)
        except (TypeError, ValueError):
            progress_value = 0.0
        if progress_value > 1.0:
            progress_value = progress_value / 100.0
        progress_value = max(0.0, min(1.0, progress_value))
        percent = int(round(progress_value * 100))

        phase_text = str(phase or "Boarding").strip() or "Boarding"
        status = str(status_text or "").strip()
        state_name = self._state_name(state)

        self.route_label.setText(f"{from_code} → {to_code}")
        self.progress_label.setText(f"{percent}%")
        self.progress_bar.setValue(percent)
        self.phase_label.setText("")
        self.phase_label.hide()
        self.remaining_label.setText(f"{self._format_remaining(remaining_seconds)} còn lại")
        self.status_label.setText("")
        self.status_label.hide()
        state_color = self.STATE_COLORS.get(state_name, "#7f93aa")
        self._set_status_dot(state_color)

        map_payload = dict(payload or {})
        map_payload.setdefault("route_from_code", from_code if from_code != "---" else "")
        map_payload.setdefault("route_to_code", to_code if to_code != "---" else "")
        hold_motion = progress_value <= 0.0001 or status == "Tạm dừng"
        self.map_canvas.set_data(map_payload, progress_value, phase_text, state_color, hold_motion=hold_motion)

    def place_near_parent(self, parent: QWidget | None = None) -> None:
        """Move near the bottom-right of the current screen unless the user dragged it."""
        if self._user_moved:
            return

        screen = parent.screen() if parent is not None and parent.screen() is not None else QApplication.primaryScreen()
        if screen is None:
            return

        rect = screen.availableGeometry()
        margin = 22
        self.move(rect.right() - self.width() - margin, rect.bottom() - self.height() - margin)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            self._user_moved = True
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.openRequested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if hasattr(self, "map_canvas"):
            self.map_canvas._update_rounded_mask()

    def _hide_for_session(self) -> None:
        self.closeRequested.emit()
        self.hide()

    def _set_status_dot(self, color: str) -> None:
        self.status_dot.setStyleSheet(
            f"background-color: {color}; border-radius: 5px; border: 1px solid rgba(255, 255, 255, 0.35);"
        )

    @staticmethod
    def _state_name(state: Any) -> str:
        if state is None:
            return "UNCERTAIN"
        name = getattr(state, "name", None)
        if name:
            return str(name)
        text = str(state or "").strip()
        return text.upper() if text else "UNCERTAIN"

    @staticmethod
    def _format_remaining(seconds: int | float) -> str:
        try:
            total = max(0, int(seconds or 0))
        except (TypeError, ValueError):
            total = 0
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
