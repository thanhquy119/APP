"""Work-rhythm results dashboard for day, week, and month views."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .theme import get_stylesheet, _theme_tokens
from .dialog_title_bar import DialogTitleBar


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds or 0.0))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}g {minutes:02d}p"
    if minutes:
        return f"{minutes}p {secs:02d}s"
    return f"{secs}s"


def _format_percent(value: float) -> str:
    return f"{max(0.0, min(1.0, float(value or 0.0))) * 100:.0f}%"


class RhythmMetricCard(QFrame):
    """Small KPI tile used in the work-rhythm dashboard."""

    def __init__(self, title: str, value: str, detail: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("rhythmMetricCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(94)

        from PyQt6.QtWidgets import QGraphicsDropShadowEffect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(12, 20, 34, 40))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(5)

        title_label = QLabel(title)
        title_label.setObjectName("rhythmMetricTitle")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setObjectName("rhythmMetricValue")
        value_label.setWordWrap(True)
        layout.addWidget(value_label)

        detail_label = QLabel(detail)
        detail_label.setObjectName("rhythmMetricDetail")
        detail_label.setWordWrap(True)
        layout.addWidget(detail_label)


class RhythmTrendChart(QFrame):
    """Bar + line chart: effective work minutes and average readiness."""

    def __init__(self, *, is_dark: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("rhythmChartCard")
        self.setMinimumHeight(284)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._is_dark = bool(is_dark)
        self._points: List[Dict[str, Any]] = []

        from PyQt6.QtWidgets import QGraphicsDropShadowEffect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(12, 20, 34, 40))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

    def set_points(self, points: List[Dict[str, Any]]) -> None:
        self._points = list(points or [])
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(20, 20, -20, -20)

        text = QColor("#edf4fd" if self._is_dark else "#182c41")
        muted = QColor("#9baec5" if self._is_dark else "#58718b")
        accent = QColor("#59d5c0" if self._is_dark else "#2f9f90")
        line_color = QColor("#86a9ff" if self._is_dark else "#3f6fb5")
        grid = QColor(139, 163, 190, 32 if self._is_dark else 46)

        title_rect = QRectF(rect.left(), rect.top(), rect.width(), 24)
        painter.setPen(text)
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "Thống kê thời gian")

        active = [point for point in self._points if int(point.get("session_count", 0) or 0) > 0]
        if not active:
            painter.setPen(muted)
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Chưa có dữ liệu trong kỳ này")
            return

        chart = rect.adjusted(16, 52, -16, -40)
        painter.setPen(QPen(grid, 1))
        for i in range(4):
            y = chart.top() + chart.height() * i / 3.0
            painter.drawLine(QPointF(chart.left(), y), QPointF(chart.right(), y))

        max_focus = max(10.0, max(float(point.get("focus_minutes", 0.0) or 0.0) for point in self._points))
        count = max(1, len(self._points))
        slot = chart.width() / float(count)
        bar_width = max(4.0, min(24.0, slot * 0.52))
        line_points: List[QPointF] = []

        painter.setPen(Qt.PenStyle.NoPen)
        for idx, point in enumerate(self._points):
            focus_minutes = max(0.0, float(point.get("focus_minutes", 0.0) or 0.0))
            x = chart.left() + idx * slot + (slot - bar_width) / 2.0
            h = (focus_minutes / max_focus) * chart.height()
            bar = QRectF(x, chart.bottom() - h, bar_width, h)
            alpha = 210 if int(point.get("session_count", 0) or 0) else 42
            painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(), alpha))
            painter.drawRoundedRect(bar, 4, 4)

            avg_score = point.get("avg_score")
            if avg_score is not None:
                try:
                    score = max(0.0, min(100.0, float(avg_score)))
                    px = chart.left() + idx * slot + slot / 2.0
                    py = chart.bottom() - (score / 100.0) * chart.height()
                    line_points.append(QPointF(px, py))
                except (TypeError, ValueError):
                    pass

        if len(line_points) >= 2:
            pen = QPen(line_color, 2.2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            for idx in range(len(line_points) - 1):
                painter.drawLine(line_points[idx], line_points[idx + 1])

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(line_color)
        for point in line_points:
            painter.drawEllipse(point, 3.4, 3.4)

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(muted)
        label_step = max(1, int(round(count / 6.0)))
        for idx, point in enumerate(self._points):
            if idx % label_step != 0 and idx != count - 1:
                continue
            label = str(point.get("label", "") or "")
            x = chart.left() + idx * slot
            label_rect = QRectF(x - slot * 0.5, chart.bottom() + 8, slot * 2.0, 20)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

        legend_y = chart.bottom() + 32
        painter.setPen(muted)
        painter.setFont(QFont("Segoe UI", 8))
        painter.setBrush(accent)
        painter.drawRoundedRect(QRectF(rect.left() + 14, legend_y + 3, 18, 7), 3, 3)
        painter.drawText(QPointF(rect.left() + 38, legend_y + 10), "phút ổn định")
        painter.setPen(QPen(line_color, 2))
        painter.drawLine(QPointF(rect.left() + 138, legend_y + 7), QPointF(rect.left() + 158, legend_y + 7))
        painter.setPen(muted)
        painter.drawText(QPointF(rect.left() + 166, legend_y + 10), "sẵn sàng TB")


class RhythmStateStackChart(QFrame):
    """State-composition chart for the selected period."""

    LABELS = {
        "focused": "Làm việc ổn định",
        "distraction": "Lệch nhịp",
        "fatigue": "Mệt",
        "away": "Vắng",
        "uncertain": "Chưa rõ",
    }

    COLORS = {
        "focused": "#59d5c0",
        "distraction": "#f09d95",
        "fatigue": "#efbd78",
        "away": "#8ea1b5",
        "uncertain": "#86a9ff",
    }

    def __init__(self, *, is_dark: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("rhythmChartCard")
        self.setMinimumHeight(284)
        self.setMinimumWidth(260)
        self._is_dark = bool(is_dark)
        self._distribution: Dict[str, Dict[str, float]] = {}

        from PyQt6.QtWidgets import QGraphicsDropShadowEffect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(12, 20, 34, 40))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

    def set_distribution(self, distribution: Dict[str, Dict[str, float]]) -> None:
        self._distribution = dict(distribution or {})
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(20, 20, -20, -20)

        text = QColor("#edf4fd" if self._is_dark else "#182c41")
        muted = QColor("#9baec5" if self._is_dark else "#58718b")

        painter.setPen(text)
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(rect.left(), rect.top(), rect.width(), 24),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "Cơ cấu trạng thái",
        )

        if not self._distribution:
            painter.setPen(muted)
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Chưa có dữ liệu")
            return

        bar_rect = QRectF(rect.left(), rect.top() + 48, rect.width(), 22)
        x = bar_rect.left()
        for key in ("focused", "distraction", "fatigue", "away", "uncertain"):
            ratio = max(0.0, min(1.0, float(self._distribution.get(key, {}).get("ratio", 0.0) or 0.0)))
            if ratio <= 0.0:
                continue
            w = max(2.0, bar_rect.width() * ratio)
            segment = QRectF(x, bar_rect.top(), min(w, bar_rect.right() - x), bar_rect.height())
            color = QColor(self.COLORS[key])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(segment, 6, 6)
            x += w
            if x >= bar_rect.right():
                break

        y = bar_rect.bottom() + 28
        painter.setFont(QFont("Segoe UI", 9))
        for key in ("focused", "distraction", "fatigue", "away", "uncertain"):
            payload = self._distribution.get(key, {}) or {}
            seconds = float(payload.get("seconds", 0.0) or 0.0)
            ratio = float(payload.get("ratio", 0.0) or 0.0)
            if seconds <= 0.0 and ratio <= 0.0:
                continue
            color = QColor(self.COLORS[key])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(rect.left(), y - 10, 10, 10), 3, 3)
            painter.setPen(text)
            painter.drawText(QPointF(rect.left() + 18, y), self.LABELS[key])
            painter.setPen(muted)
            value = f"{ratio * 100:.0f}%  ·  {_format_duration(seconds)}"
            painter.drawText(
                QRectF(rect.left() + 80, y - 15, rect.width() - 80, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                value,
            )
            y += 28


class WorkRhythmReportDialog(QDialog):
    """Show meaningful work-rhythm results for day, week, and month."""

    PERIOD_ORDER = (("day", "Ngày"), ("week", "Tuần"), ("month", "Tháng"))
    dismissed = pyqtSignal()

    def __init__(
        self,
        *,
        summary: Dict[str, Any],
        config: Optional[dict] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._summary = dict(summary or {})
        self._config = dict(config or {})
        self._is_dark = str(self._config.get("theme_mode", "dark")).strip().lower() != "light"
        self.setObjectName("workRhythmReportDialog")
        self.setWindowTitle("Kết quả nhịp làm việc")
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(900, 660)
        self.resize(980, 720)
        self.setStyleSheet(get_stylesheet(self._is_dark) + self._local_stylesheet())
        self._build_ui()

    def closeEvent(self, event):
        self.dismissed.emit()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        # ── Drop shadow wrapper ────────────────────────────────────────────
        from PyQt6.QtWidgets import QGraphicsDropShadowEffect

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 80 if self._is_dark else 50))
        self.setGraphicsEffect(shadow)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)  # shadow breathing room
        outer.setSpacing(0)

        container = QFrame()
        container.setObjectName("rhythmContainer")
        outer.addWidget(container)

        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # ── Custom title bar with 3 dots ──────────────────────────────────
        self._title_bar = DialogTitleBar(
            "Kết quả nhịp làm việc",
            is_dark=self._is_dark,
            parent=container,
        )
        container_layout.addWidget(self._title_bar)

        # ── Content area ─────────────────────────────────────────────────
        content = QVBoxLayout()
        content.setContentsMargins(18, 4, 18, 18)
        content.setSpacing(14)

        tabs = QTabWidget()
        tabs.setObjectName("rhythmTabs")
        periods = dict(self._summary.get("periods", {}) or {})
        for key, label in self.PERIOD_ORDER:
            tabs.addTab(self._build_period_page(periods.get(key, {})), label)
        content.addWidget(tabs, 1)

        container_layout.addLayout(content, 1)

    def _build_period_page(self, period: Dict[str, Any]) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName("rhythmScroll")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        host = QWidget()
        host.setObjectName("rhythmPage")
        layout = QVBoxLayout(host)
        layout.setContentsMargins(2, 14, 2, 2)
        layout.setSpacing(14)

        metric_grid = QGridLayout()
        metric_grid.setContentsMargins(0, 0, 0, 0)
        metric_grid.setHorizontalSpacing(10)
        metric_grid.setVerticalSpacing(10)

        metric_cards = self._metric_cards_for_period(period)
        for index, card in enumerate(metric_cards):
            metric_grid.addWidget(card, index // 4, index % 4)
        layout.addLayout(metric_grid)

        charts = QHBoxLayout()
        charts.setContentsMargins(0, 0, 0, 0)
        charts.setSpacing(12)

        trend_chart = RhythmTrendChart(is_dark=self._is_dark)
        trend_chart.set_points(list(period.get("points", []) or []))
        charts.addWidget(trend_chart, 3)

        state_chart = RhythmStateStackChart(is_dark=self._is_dark)
        state_chart.set_distribution(dict(period.get("state_distribution", {}) or {}))
        charts.addWidget(state_chart, 2)
        layout.addLayout(charts)

        scroll.setWidget(host)
        return scroll

    def _metric_cards_for_period(self, period: Dict[str, Any]) -> List[RhythmMetricCard]:
        session_count = int(period.get("session_count", 0) or 0)
        total_seconds = float(period.get("total_seconds", 0.0) or 0.0)
        focus_seconds = float(period.get("focus_seconds", 0.0) or 0.0)
        focus_ratio = float(period.get("focus_ratio", 0.0) or 0.0)
        avg_score = float(period.get("avg_score", 0.0) or 0.0)
        distraction_rate = float(period.get("distractions_per_hour", 0.0) or 0.0)
        best_bucket = str(period.get("best_bucket_label", "") or "")
        live_note = "Có cộng phiên đang chạy" if bool(period.get("live_session_included", False)) else f"{session_count} phiên"

        return [
            RhythmMetricCard(
                "Làm việc ổn định",
                _format_duration(focus_seconds),
                f"Tổng theo dõi {_format_duration(total_seconds)}",
            ),
            RhythmMetricCard(
                "Độ ổn định",
                _format_percent(focus_ratio),
                live_note,
            ),
            RhythmMetricCard(
                "Sẵn sàng trung bình",
                f"{avg_score:.0f}",
                f"Tốt nhất: {best_bucket}" if best_bucket else "Đang tích lũy dữ liệu",
            ),
            RhythmMetricCard(
                "Lệch nhịp mỗi giờ",
                f"{distraction_rate:.1f}",
                f"{int(period.get('distraction_count', 0) or 0)} lần lệch nhịp",
            ),
        ]

    def _local_stylesheet(self) -> str:
        if self._is_dark:
            surface = "#101b29"
            surface_alt = "#132234"
            border = "rgba(128, 155, 183, 0.26)"
            text = "#edf4fd"
            muted = "#9baec5"
            accent = "#59d5c0"
        else:
            surface = "#ffffff"
            surface_alt = "#f3f8fe"
            border = "rgba(104, 136, 170, 0.26)"
            text = "#182c41"
            muted = "#58718b"
            accent = "#2f9f90"

        t = _theme_tokens(self._is_dark)

        return f"""
            QDialog#workRhythmReportDialog {{
                background: transparent;
                border: none;
            }}
            QFrame#rhythmContainer {{
                background-color: {surface_alt};
                border: 1px solid {border};
                border-radius: 14px;
            }}

            /* ── Title bar ── */
            QFrame#reportTitleBar {{
                background-color: transparent;
                border: none;
                border-top-left-radius: 13px;
                border-top-right-radius: 13px;
            }}
            QLabel#reportTitleText {{
                color: {text};
                font-size: 14px;
                font-weight: 700;
                letter-spacing: 0.2px;
            }}
            QToolButton#titleBarCloseDot,
            QToolButton#titleBarMinDot,
            QToolButton#titleBarMaxDot {{
                min-width: 12px;
                max-width: 12px;
                min-height: 12px;
                max-height: 12px;
                border-radius: 6px;
                border: none;
                padding: 0;
                background-color: transparent;
            }}
            QToolButton#titleBarCloseDot {{
                background-color: {t.get('titlebar_dot_close', '#ff5f57')};
            }}
            QToolButton#titleBarCloseDot:hover {{
                background-color: {t.get('titlebar_dot_close_hover', '#ff736d')};
            }}
            QToolButton#titleBarCloseDot:pressed {{
                background-color: {t.get('titlebar_dot_close_pressed', '#e14f49')};
            }}
            QToolButton#titleBarMinDot {{
                background-color: {t.get('titlebar_dot_min', '#febc2e')};
            }}
            QToolButton#titleBarMinDot:hover {{
                background-color: {t.get('titlebar_dot_min_hover', '#ffca4c')};
            }}
            QToolButton#titleBarMinDot:pressed {{
                background-color: {t.get('titlebar_dot_min_pressed', '#dea225')};
            }}
            QToolButton#titleBarMaxDot {{
                background-color: {t.get('titlebar_dot_max', '#28c840')};
            }}
            QToolButton#titleBarMaxDot:hover {{
                background-color: {t.get('titlebar_dot_max_hover', '#42d95a')};
            }}
            QToolButton#titleBarMaxDot:pressed {{
                background-color: {t.get('titlebar_dot_max_pressed', '#1faa36')};
            }}

            QLabel#rhythmDialogSubtitle {{
                color: {muted};
                font-size: 12px;
                font-weight: 500;
            }}
            QScrollArea#rhythmScroll {{
                background: transparent;
                border: none;
            }}
            QWidget#rhythmPage {{
                background: transparent;
            }}
            QFrame#rhythmMetricCard,
            QFrame#rhythmInsightCard,
            QFrame#rhythmChartCard {{
                background-color: {surface};
                border: 1px solid {border};
                border-radius: 16px;
            }}
            QLabel#rhythmMetricTitle {{
                color: {muted};
                font-size: 11px;
                font-weight: 650;
            }}
            QLabel#rhythmMetricValue {{
                color: {text};
                font-size: 23px;
                font-weight: 760;
            }}
            QLabel#rhythmMetricDetail,
            QLabel#rhythmInsightText {{
                color: {muted};
                font-size: 12px;
                line-height: 1.45;
            }}
            QTabWidget#rhythmTabs::pane {{
                border: none;
                background: transparent;
            }}
            QTabWidget#rhythmTabs QTabBar::tab {{
                background-color: transparent;
                border-bottom: 2px solid transparent;
                color: {muted};
                min-width: 88px;
                padding: 9px 18px;
                font-weight: 600;
                font-size: 13px;
            }}
            QTabWidget#rhythmTabs QTabBar::tab:selected {{
                color: {text};
                border-bottom: 2px solid {accent};
            }}
            QTabWidget#rhythmTabs QTabBar::tab:hover:!selected {{
                color: {text};
                border-bottom: 2px solid rgba(128, 155, 183, 0.4);
            }}
        """
