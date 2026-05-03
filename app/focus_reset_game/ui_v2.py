"""Giao diện Focus Reset (phiên bản tiếng Việt)."""

from __future__ import annotations

from datetime import datetime
import random
import threading
import time
from pathlib import Path

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config import FocusResetConfig, Theme, load_focus_reset_config, save_focus_reset_config
from .game_gonogo import build_gonogo_trials, summarize_gonogo
from .game_logic import active_trial_at, evaluate_trials
from .game_sequence import build_round_lengths, build_sequence, evaluate_sequence
from .game_visual_search import build_visual_specs, evaluate_visual
from .metrics import build_session_summary
from .models import (
    MetricSummary,
    SequenceRoundResult,
    SequenceSummary,
    SessionSummary,
    TrialSpec,
    VisualRoundResult,
    VisualRoundSpec,
    VisualSummary,
)
from .storage import SessionStorage, build_session_record


class AttentionProbeTitleBar(QFrame):
    """Frameless dialog title bar with calm macOS-style window dots."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("topHeaderBar")
        self.setFixedHeight(48)

        self._max_toggle_guard = False

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 8, 14, 8)
        root.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("topHeaderTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        root.addWidget(self.title_label, 1)

        self.controls_host = QWidget(self)
        self.controls_host.setObjectName("titleBarDotsHost")
        controls = QHBoxLayout(self.controls_host)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        self.btn_close = self._create_dot("titleBarCloseDot", "Đóng")
        self.btn_min = self._create_dot("titleBarMinDot", "Thu nhỏ")
        self.btn_max = self._create_dot("titleBarMaxDot", "Căn lại cửa sổ")

        self.btn_close.clicked.connect(self._close_window)
        self.btn_min.clicked.connect(self._minimize_window)
        self.btn_max.clicked.connect(self._toggle_max_restore)

        controls.addWidget(self.btn_min)
        controls.addWidget(self.btn_max)
        controls.addWidget(self.btn_close)
        root.addWidget(self.controls_host, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.sync_window_state()

    def _create_dot(self, object_name: str, tooltip: str) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName(object_name)
        button.setToolTip(tooltip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setText("")
        button.setFixedSize(12, 12)
        button.setAutoRaise(True)
        return button

    def _window(self) -> QWidget | None:
        window = self.window()
        return window if isinstance(window, QWidget) else None

    def _is_window_maximized(self) -> bool:
        window = self._window()
        if window is None:
            return False
        if window.isMaximized() or (window.windowState() & Qt.WindowState.WindowMaximized):
            return True

        handle = window.windowHandle()
        screen = handle.screen() if handle is not None else window.screen()
        if screen is None:
            return False

        available = screen.availableGeometry()
        frame = window.frameGeometry()
        tolerance = 8
        return (
            abs(frame.left() - available.left()) <= tolerance
            and abs(frame.right() - available.right()) <= tolerance
            and abs(frame.top() - available.top()) <= tolerance
            and abs(frame.bottom() - available.bottom()) <= tolerance
        )

    def sync_window_state(self) -> None:
        is_maximized = self._is_window_maximized()
        self.setProperty("maximized", is_maximized)
        self.btn_max.setProperty("windowMaximized", is_maximized)
        self.style().unpolish(self)
        self.style().polish(self)
        self.btn_max.style().unpolish(self.btn_max)
        self.btn_max.style().polish(self.btn_max)
        self.btn_max.setToolTip("Khôi phục kích thước" if is_maximized else "Căn lại cửa sổ")

    def _is_over_control(self, pos: QPointF) -> bool:
        point = pos.toPoint()
        if isinstance(self.childAt(point), QToolButton):
            return True
        local = self.controls_host.mapFrom(self, point)
        return self.controls_host.rect().contains(local)

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

        window.showNormal()
        reset_geometry = getattr(window, "_apply_window_geometry", None)
        if callable(reset_geometry):
            QTimer.singleShot(0, reset_geometry)
        QTimer.singleShot(0, self.sync_window_state)
        QTimer.singleShot(120, self.sync_window_state)

    def _close_window(self) -> None:
        window = self._window()
        if window is not None:
            window.close()

    def mousePressEvent(self, event):
        if not self._is_over_control(event.position()):
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._is_over_control(event.position()):
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._is_over_control(event.position()):
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self._is_over_control(event.position()):
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


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


class AttentionTrendWidget(QWidget):
    """Compact chart for post-probe recovery validation history."""

    def __init__(self, theme: Theme, parent=None):
        super().__init__(parent)
        self.theme = theme
        self._records: list[dict] = []
        self.setObjectName("trendChart")
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_records(self, records: list[dict]) -> None:
        self._records = list(records or [])[-10:]
        self.update()

    @staticmethod
    def _as_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def score_from_record(record: dict) -> float | None:
        if "transfer_score" in record:
            transfer = AttentionTrendWidget._as_float(record.get("transfer_score"), -1.0)
            if transfer >= 0.0:
                score = transfer * 100.0 if transfer <= 1.0 else transfer
                return max(0.0, min(100.0, score))

        if "post_work_readiness" in record:
            score = AttentionTrendWidget._as_float(record.get("post_work_readiness"), -1.0)
            if score >= 0.0:
                return max(0.0, min(100.0, score))

        return None

    def scores(self) -> list[float]:
        scores: list[float] = []
        for record in self._records:
            score = self.score_from_record(record)
            if score is not None:
                scores.append(score)
        return scores

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(self.theme.border), 1))
        painter.setBrush(QColor(self.theme.panel))
        painter.drawRoundedRect(outer, 8, 8)

        chart = QRectF(self.rect()).adjusted(18, 18, -18, -20)
        grid_color = QColor(self.theme.border)
        grid_color.setAlpha(105)
        painter.setPen(QPen(grid_color, 1))
        for ratio in (0.25, 0.50, 0.75):
            y = chart.top() + chart.height() * ratio
            painter.drawLine(QPointF(chart.left(), y), QPointF(chart.right(), y))

        scores = self.scores()
        if not scores:
            painter.setPen(QColor(self.theme.text_muted))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Chưa có dữ liệu sau 5-10p")
            return

        count = len(scores)
        if count == 1:
            xs = [chart.center().x()]
        else:
            step = chart.width() / float(count - 1)
            xs = [chart.left() + (idx * step) for idx in range(count)]

        base_y = chart.bottom()
        bar_width = max(10.0, min(28.0, chart.width() / max(8.0, count * 2.2)))
        accent = QColor(self.theme.accent)
        muted_accent = QColor(self.theme.accent)
        muted_accent.setAlpha(70)

        points: list[QPointF] = []
        for x, score in zip(xs, scores):
            y = chart.bottom() - (score / 100.0) * chart.height()
            points.append(QPointF(x, y))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(muted_accent)
            painter.drawRoundedRect(QRectF(x - bar_width / 2, y, bar_width, base_y - y), 5, 5)

        if len(points) >= 2:
            painter.setPen(QPen(accent, 3))
            for start, end in zip(points, points[1:]):
                painter.drawLine(start, end)

        dot_border = QColor(self.theme.text_primary)
        for idx, point in enumerate(points):
            painter.setPen(QPen(dot_border if idx == len(points) - 1 else accent, 2))
            painter.setBrush(QColor(self.theme.panel_soft) if idx < len(points) - 1 else accent)
            radius = 4.5 if idx < len(points) - 1 else 6.0
            painter.drawEllipse(point, radius, radius)


class FocusResetDialog(QDialog):
    """Attention Probe dialog with short post-break attention tasks."""

    GAME_ORDER = ["gonogo", "stroop", "flanker", "sequence", "visual"]
    GAME_TITLES = {
        "gonogo": "Phản xạ Go/No-Go",
        "stroop": "Stroop màu",
        "flanker": "Mũi tên Flanker",
        "sequence": "Ghi nhớ chuỗi",
        "visual": "Tìm kiếm thị giác",
    }
    ATTENTION_PHASES = {"baseline", "gonogo", "stroop", "flanker"}

    def __init__(
        self,
        parent=None,
        config: FocusResetConfig | None = None,
        storage: SessionStorage | None = None,
        theme_mode: str = "dark",
        app_sound_enabled: bool = True,
        app_volume: int = 70,
        break_context: dict | None = None,
    ):
        super().__init__(parent)

        self.cfg = config or load_focus_reset_config()
        if tuple(str(symbol).upper() for symbol in self.cfg.sequence.symbols[:4]) == ("A", "S", "D", "F"):
            self.cfg.sequence.symbols = ("1", "2", "3", "4", "5")
        elif tuple(str(symbol) for symbol in self.cfg.sequence.symbols) == ("1", "2", "3", "4"):
            self.cfg.sequence.symbols = ("1", "2", "3", "4", "5")
        self.theme_mode = "light" if str(theme_mode or "dark").strip().lower() == "light" else "dark"
        self.theme = Theme.for_mode(self.theme_mode)
        self.storage = storage or SessionStorage(self.cfg.history_path)
        self._app_sound_enabled = bool(app_sound_enabled)
        self._app_volume = max(0, min(100, int(app_volume)))
        self._break_context = dict(break_context or {})

        self._last_sound_at = 0.0
        self._resize_margin = 8
        self._resize_edges: set[str] = set()
        self._resize_start_pos: QPoint | None = None
        self._resize_start_geometry = None

        self._phase_timer = QTimer(self)
        self._phase_timer.timeout.connect(self._on_phase_tick)

        self._sequence_timeout = QTimer(self)
        self._sequence_timeout.setSingleShot(True)
        self._sequence_timeout.timeout.connect(self._on_sequence_timeout)

        self._sequence_token = 0

        self._reset_runtime_state()

        self.setObjectName("attentionProbeDialog")
        self.setWindowTitle("Kiểm tra chú ý ngắn")
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
        )
        self.setSizeGripEnabled(False)
        self.setModal(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet(self._build_stylesheet())

        self.stack = QStackedWidget(self)
        self.stack.setObjectName("probeStack")
        self.stack.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 12)
        root.setSpacing(6)

        self.title_bar = AttentionProbeTitleBar("Kiểm tra chú ý ngắn", self)
        root.addWidget(self.title_bar, 0)
        root.addWidget(self.stack, 1)

        self.page_menu = self._build_menu_page()
        self.page_instructions = self._build_instructions_page()
        self.page_select = self._build_game_select_page()
        self.page_settings = self._build_settings_page()
        self.page_recovery_results = self._build_recovery_results_page()
        self.page_history = self._build_history_page()
        self.page_session = self._build_session_page()
        self.page_results = self._build_results_page()

        self.stack.addWidget(self.page_menu)
        self.stack.addWidget(self.page_instructions)
        self.stack.addWidget(self.page_select)
        self.stack.addWidget(self.page_settings)
        self.stack.addWidget(self.page_recovery_results)
        self.stack.addWidget(self.page_history)
        self.stack.addWidget(self.page_session)
        self.stack.addWidget(self.page_results)
        self.stack.currentChanged.connect(self._sync_stack_visibility)
        self.stack.currentChanged.connect(self._fit_to_current_page)

        self._load_settings_to_controls()
        self._rebuild_sequence_symbol_buttons()
        self._show_menu()
        self._apply_window_geometry()
        QTimer.singleShot(0, self._fit_to_current_page)

    def _resolve_screen(self):
        parent = self.parentWidget()
        if parent is not None and parent.windowHandle() is not None and parent.windowHandle().screen() is not None:
            return parent.windowHandle().screen()
        if parent is not None and parent.screen() is not None:
            return parent.screen()
        return self.screen() or QGuiApplication.primaryScreen()

    def _apply_window_geometry(self) -> None:
        screen = self._resolve_screen()
        if screen is None:
            self.setMinimumSize(760, 520)
            self.resize(900, 620)
            return

        available = screen.availableGeometry()
        margin = 24

        max_w = max(700, available.width() - (margin * 2))
        max_h = max(500, available.height() - (margin * 2))

        desired_w = 900
        desired_h = 620

        target_w = min(desired_w, max_w)
        target_h = min(desired_h, max_h)

        min_w = min(680, max_w)
        min_h = min(460, max_h)
        self.setMinimumSize(min_w, min_h)
        self.setMaximumSize(16777215, 16777215)
        self.resize(target_w, target_h)

        x = available.x() + (available.width() - target_w) // 2
        y = available.y() + (available.height() - target_h) // 2
        self.move(max(available.x(), x), max(available.y(), y))

    def showEvent(self, event):
        super().showEvent(event)
        self._clamp_to_screen()
        self.title_bar.sync_window_state()

    def changeEvent(self, event):
        super().changeEvent(event)
        if hasattr(self, "title_bar"):
            QTimer.singleShot(0, self.title_bar.sync_window_state)

    def _resize_edges_at(self, pos: QPoint) -> set[str]:
        return set()

    def _cursor_for_edges(self, edges: set[str]):
        if {"left", "top"} <= edges or {"right", "bottom"} <= edges:
            return Qt.CursorShape.SizeFDiagCursor
        if {"right", "top"} <= edges or {"left", "bottom"} <= edges:
            return Qt.CursorShape.SizeBDiagCursor
        if "left" in edges or "right" in edges:
            return Qt.CursorShape.SizeHorCursor
        if "top" in edges or "bottom" in edges:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def _resize_to_global_pos(self, global_pos: QPoint) -> None:
        if not self._resize_edges or self._resize_start_pos is None or self._resize_start_geometry is None:
            return

        delta = global_pos - self._resize_start_pos
        geom = self._resize_start_geometry
        x = geom.x()
        y = geom.y()
        w = geom.width()
        h = geom.height()
        min_w = self.minimumWidth()
        min_h = self.minimumHeight()

        if "left" in self._resize_edges:
            new_w = max(min_w, w - delta.x())
            x = geom.right() - new_w + 1
            w = new_w
        elif "right" in self._resize_edges:
            w = max(min_w, w + delta.x())

        if "top" in self._resize_edges:
            new_h = max(min_h, h - delta.y())
            y = geom.bottom() - new_h + 1
            h = new_h
        elif "bottom" in self._resize_edges:
            h = max(min_h, h + delta.y())

        self.setGeometry(x, y, w, h)

    def _is_in_titlebar_area(self, pos: QPoint) -> bool:
        if not hasattr(self, "title_bar"):
            return False
        return self.title_bar.geometry().contains(pos)

    def mousePressEvent(self, event):
        if self._is_in_titlebar_area(event.position().toPoint()):
            event.accept()
            return

        edges = self._resize_edges_at(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and edges:
            self._resize_edges = edges
            self._resize_start_pos = event.globalPosition().toPoint()
            self._resize_start_geometry = self.geometry()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_in_titlebar_area(event.position().toPoint()):
            self.unsetCursor()
            event.accept()
            return

        if self._resize_edges and self._resize_start_pos is not None:
            self._resize_to_global_pos(event.globalPosition().toPoint())
            event.accept()
            return

        edges = self._resize_edges_at(event.position().toPoint())
        self.setCursor(self._cursor_for_edges(edges))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_in_titlebar_area(event.position().toPoint()):
            self._resize_edges = set()
            self._resize_start_pos = None
            self._resize_start_geometry = None
            self.unsetCursor()
            event.accept()
            return

        self._resize_edges = set()
        self._resize_start_pos = None
        self._resize_start_geometry = None
        self.unsetCursor()
        super().mouseReleaseEvent(event)

    def _clamp_to_screen(self) -> None:
        screen = self._resolve_screen()
        if screen is None:
            return

        available = screen.availableGeometry()
        max_w = max(640, available.width() - 12)
        max_h = max(480, available.height() - 12)

        if self.width() > max_w or self.height() > max_h:
            self.resize(min(self.width(), max_w), min(self.height(), max_h))

        max_x = available.right() - self.width() + 1
        max_y = available.bottom() - self.height() + 1
        clamped_x = min(max(self.x(), available.x()), max_x)
        clamped_y = min(max(self.y(), available.y()), max_y)
        self.move(clamped_x, clamped_y)

    def _fit_to_current_page(self, *_args) -> None:
        self._sync_stack_visibility()
        current = self.stack.currentWidget()
        if current is None:
            return

        if not (self.isMaximized() or (self.windowState() & Qt.WindowState.WindowMaximized)):
            self._clamp_to_screen()

    def _sync_stack_visibility(self, *_args) -> None:
        current = self.stack.currentWidget()
        for idx in range(self.stack.count()):
            widget = self.stack.widget(idx)
            widget.setVisible(widget is current)

    def _set_current_page(self, page: QWidget) -> None:
        self.stack.setCurrentWidget(page)
        self._sync_stack_visibility()
        self._fit_to_current_page()

    def _build_stylesheet(self) -> str:
        return f"""
            QDialog#attentionProbeDialog {{
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 {self.theme.hero_bg},
                    stop: 0.45 {self.theme.background},
                    stop: 1 {self.theme.panel_soft}
                );
                color: {self.theme.text_primary};
                font-family: 'Bahnschrift', 'Segoe UI', sans-serif;
                font-size: 14px;
            }}
            QStackedWidget#probeStack {{
                background-color: transparent;
                border: none;
            }}
            QFrame#topHeaderBar {{
                background-color: transparent;
                border: none;
                border-radius: 0px;
            }}
            QLabel#topHeaderTitle {{
                color: {self.theme.text_primary};
                font-size: 15px;
                font-weight: 700;
            }}
            QWidget#titleBarDotsHost {{
                background: transparent;
            }}
            QFrame#menuPanel {{
                background-color: {self.theme.panel};
                border: 1px solid {self.theme.border};
                border-radius: 8px;
            }}
            QFrame#menuHero {{
                background-color: transparent;
                border: none;
            }}
            QFrame#trendPanel {{
                background-color: {self.theme.panel_soft};
                border: 1px solid {self.theme.border};
                border-radius: 8px;
            }}
            QFrame#trendMetric {{
                background-color: {self.theme.panel};
                border: 1px solid {self.theme.interactive_border};
                border-radius: 8px;
            }}
            QLabel#kicker {{
                color: {self.theme.accent};
                font-size: 12px;
                font-weight: 800;
            }}
            QLabel#menuTitle {{
                color: {self.theme.text_primary};
                font-size: 30px;
                font-weight: 800;
            }}
            QLabel#menuSubtitle {{
                color: {self.theme.text_muted};
                font-size: 16px;
                font-weight: 500;
            }}
            QLabel#trendTitle {{
                color: {self.theme.text_primary};
                font-size: 18px;
                font-weight: 800;
            }}
            QLabel#trendValue {{
                color: {self.theme.text_primary};
                font-size: 18px;
                font-weight: 800;
            }}
            QLabel#trendLabel {{
                color: {self.theme.text_muted};
                font-size: 12px;
                font-weight: 600;
            }}
            QToolButton#titleBarCloseDot,
            QToolButton#titleBarMinDot,
            QToolButton#titleBarMaxDot {{
                min-width: 12px;
                max-width: 12px;
                min-height: 12px;
                max-height: 12px;
                border: none;
                border-radius: 6px;
                padding: 0px;
                margin: 0px;
            }}
            QToolButton#titleBarCloseDot {{
                background-color: {self.theme.titlebar_dot_close};
            }}
            QToolButton#titleBarCloseDot:hover {{
                background-color: {self.theme.titlebar_dot_close_hover};
            }}
            QToolButton#titleBarCloseDot:pressed {{
                background-color: {self.theme.titlebar_dot_close_pressed};
            }}
            QToolButton#titleBarMinDot {{
                background-color: {self.theme.titlebar_dot_min};
            }}
            QToolButton#titleBarMinDot:hover {{
                background-color: {self.theme.titlebar_dot_min_hover};
            }}
            QToolButton#titleBarMinDot:pressed {{
                background-color: {self.theme.titlebar_dot_min_pressed};
            }}
            QToolButton#titleBarMaxDot {{
                background-color: {self.theme.titlebar_dot_max};
            }}
            QToolButton#titleBarMaxDot:hover {{
                background-color: {self.theme.titlebar_dot_max_hover};
            }}
            QToolButton#titleBarMaxDot:pressed {{
                background-color: {self.theme.titlebar_dot_max_pressed};
            }}
            QFrame#panel {{
                background-color: {self.theme.panel};
                border: 1px solid {self.theme.border};
                border-radius: 8px;
            }}
            QFrame#probeOption {{
                background-color: {self.theme.panel_soft};
                border: 1px solid {self.theme.border};
                border-radius: 8px;
            }}
            QFrame#metricRow {{
                background-color: {self.theme.panel_soft};
                border: 1px solid {self.theme.interactive_border};
                border-radius: 8px;
                min-height: 40px;
                max-height: 42px;
            }}
            QLabel#metricRowIcon {{
                color: {self.theme.text_muted};
                font-size: 12px;
                font-weight: 800;
            }}
            QLabel#metricRowLabel {{
                color: {self.theme.text_muted};
                font-size: 12px;
                font-weight: 500;
            }}
            QLabel#metricRowValue {{
                color: {self.theme.text_primary};
                font-size: 15px;
                font-weight: 700;
                min-width: 132px;
            }}
            QFrame#hero {{
                background-color: {self.theme.hero_bg};
                border: 1px solid {self.theme.border};
                border-radius: 8px;
            }}
            QLabel#title {{
                font-size: 24px;
                font-weight: 800;
                color: {self.theme.text_primary};
            }}
            QLabel#subtitle {{
                font-size: 14px;
                color: {self.theme.text_muted};
            }}
            QLabel#muted {{
                color: {self.theme.text_muted};
            }}
            QLabel#finePrint {{
                color: {self.theme.text_muted};
                font-size: 12px;
                line-height: 1.35;
            }}
            QLabel#value {{
                font-size: 16px;
                font-weight: 700;
            }}
            QPushButton {{
                background-color: {self.theme.interactive_bg};
                color: {self.theme.text_primary};
                border: 1px solid {self.theme.interactive_border};
                border-radius: 8px;
                padding: 7px 14px;
                font-weight: 600;
                min-height: 32px;
            }}
            QPushButton:hover {{
                background-color: {self.theme.interactive_hover};
                border-color: {self.theme.accent_border};
            }}
            QPushButton:pressed {{
                background-color: {self.theme.panel_soft};
            }}
            QPushButton#primary {{
                background-color: {self.theme.accent};
                color: {self.theme.accent_text};
                border: 1px solid {self.theme.accent_border};
                font-weight: 700;
            }}
            QPushButton#primary:hover {{
                background-color: {self.theme.accent_hover};
                border-color: {self.theme.accent_border};
            }}
            QGroupBox {{
                border: 1px solid {self.theme.border};
                border-radius: 8px;
                margin-top: 10px;
                padding: 8px;
                background-color: {self.theme.panel_soft};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {self.theme.text_primary};
                font-weight: 700;
            }}
            QProgressBar {{
                background-color: {self.theme.progress_bg};
                border: 1px solid {self.theme.border};
                border-radius: 7px;
                text-align: center;
                color: {self.theme.text_primary};
                min-height: 18px;
                font-weight: 700;
            }}
            QProgressBar::chunk {{
                background-color: {self.theme.accent};
                border-radius: 7px;
            }}
            QTableWidget {{
                background-color: {self.theme.table_bg};
                border: 1px solid {self.theme.border};
                border-radius: 8px;
                gridline-color: {self.theme.table_grid};
                color: {self.theme.text_primary};
                selection-background-color: {self.theme.selection_bg};
            }}
            QHeaderView::section {{
                background-color: {self.theme.table_header_bg};
                color: {self.theme.text_primary};
                border: none;
                border-right: 1px solid {self.theme.table_grid};
                padding: 8px;
                font-weight: 700;
            }}
            QPushButton#readyOption {{
                padding: 8px 6px;
                min-height: 62px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton#readyOption:checked {{
                background-color: {self.theme.accent};
                color: {self.theme.accent_text};
                border-color: {self.theme.accent_border};
            }}
        """

    def _interactive_button_style(self) -> str:
        return (
            "QPushButton {"
            f"background-color: {self.theme.interactive_bg};"
            f"color: {self.theme.text_primary};"
            f"border: 1px solid {self.theme.interactive_border};"
            "border-radius: 8px;"
            "font-size: 18px;"
            "font-weight: 700;"
            "padding: 4px;"
            "min-height: 30px;"
            "}"
            f"QPushButton:hover {{ background-color: {self.theme.interactive_hover}; }}"
        )

    def _rgba(self, color: str, alpha: int) -> str:
        value = QColor(color)
        if not value.isValid():
            value = QColor(self.theme.accent)
        return f"rgba({value.red()}, {value.green()}, {value.blue()}, {max(0, min(255, int(alpha)))})"

    def _set_feedback_banner(self, label: QLabel, kind: str, text: str) -> None:
        kind_name = str(kind or "info").strip().lower()
        if kind_name == "success":
            fg = self.theme.success_text
            bg = self._rgba(self.theme.success_text, 36)
            border = self._rgba(self.theme.success_text, 135)
        elif kind_name == "error":
            fg = self.theme.error_text
            bg = self._rgba(self.theme.error_text, 34)
            border = self._rgba(self.theme.error_text, 132)
        else:
            fg = self.theme.info_text
            bg = self._rgba(self.theme.info_text, 28)
            border = self._rgba(self.theme.info_text, 110)

        label.setText(text or "")
        label.setStyleSheet(
            "QLabel {"
            f"color: {fg};"
            f"background-color: {bg if text else 'transparent'};"
            f"border: 1px solid {border if text else 'transparent'};"
            "border-radius: 8px;"
            "padding: 10px 14px;"
            "font-size: 17px;"
            "font-weight: 800;"
            "min-height: 42px;"
            "}"
        )

    def _clear_feedback_banner(self, label: QLabel) -> None:
        self._set_feedback_banner(label, "info", "")

    def _play_feedback_sound(self, kind: str) -> None:
        """Play subtle sound cues when enabled in both app and game settings."""
        if not self._app_sound_enabled or not bool(self.cfg.sound_enabled):
            return
        if self._app_volume <= 0:
            return

        now = time.monotonic()
        if now - self._last_sound_at < 0.14:
            return
        self._last_sound_at = now

        kind_name = str(kind or "info").strip().lower()
        if kind_name not in {"success", "error", "info"}:
            kind_name = "info"

        try:
            import winsound

            tone_map = {
                "success": ((760, 28), (980, 38)),
                "error": ((440, 46),),
                "info": ((680, 32),),
            }

            volume_scale = 0.75 + (self._app_volume / 100.0) * 0.35

            def _beep_pattern(pattern: tuple[tuple[int, int], ...]) -> None:
                for idx, (frequency, duration_ms) in enumerate(pattern):
                    winsound.Beep(int(frequency), max(22, min(60, int(duration_ms * volume_scale))))
                    if idx < len(pattern) - 1:
                        time.sleep(0.018)

            threading.Thread(
                target=_beep_pattern,
                args=(tone_map[kind_name],),
                daemon=True,
            ).start()
            return
        except Exception:
            pass

        QApplication.beep()

    def _build_menu_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(0)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        panel = QFrame()
        panel.setObjectName("menuPanel")
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        panel.setMaximumWidth(760)
        panel.setFixedHeight(330)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(28, 24, 28, 24)
        panel_layout.setSpacing(10)

        hero = QFrame()
        hero.setObjectName("menuHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(6, 6, 6, 6)
        hero_layout.setSpacing(10)

        kicker = QLabel("ATTENTION PROBE")
        kicker.setObjectName("kicker")
        hero_layout.addWidget(kicker)

        title = QLabel("Kiểm tra chú ý ngắn")
        title.setObjectName("menuTitle")
        title.setWordWrap(True)
        hero_layout.addWidget(title)

        subtitle = QLabel("Bài kiểm tra phản ứng sau nghỉ")
        subtitle.setObjectName("menuSubtitle")
        subtitle.setWordWrap(True)
        hero_layout.addWidget(subtitle)

        hero_layout.addStretch()

        start_btn = QPushButton("Bắt đầu")
        start_btn.setObjectName("primary")
        start_btn.clicked.connect(self._start_auto_probe)
        hero_layout.addWidget(start_btn)

        result_btn = QPushButton("Kết quả")
        result_btn.clicked.connect(self._show_recovery_results)
        hero_layout.addWidget(result_btn)

        panel_layout.addWidget(hero)

        layout.addWidget(panel, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        return page

    def _build_recovery_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(0)
        layout.setContentsMargins(24, 18, 24, 18)

        panel = QFrame()
        panel.setObjectName("trendPanel")
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        panel.setMaximumWidth(760)
        panel.setFixedHeight(410)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(18, 16, 18, 16)
        panel_layout.setSpacing(10)

        trend_header = QHBoxLayout()
        trend_title = QLabel("Kết quả sau 5-10 phút")
        trend_title.setObjectName("trendTitle")
        trend_header.addWidget(trend_title)
        trend_header.addStretch()

        history_btn = QPushButton("Xem lịch sử")
        history_btn.clicked.connect(self._show_history)
        trend_header.addWidget(history_btn)
        panel_layout.addLayout(trend_header)

        self.menu_trend_chart = AttentionTrendWidget(self.theme)
        panel_layout.addWidget(self.menu_trend_chart)

        metric_row = QHBoxLayout()
        metric_row.setSpacing(8)
        self.trend_sessions_value = QLabel("0")
        self.trend_score_value = QLabel("-")
        self.trend_delta_value = QLabel("-")
        metric_row.addWidget(self._build_trend_metric("Xác nhận", self.trend_sessions_value))
        metric_row.addWidget(self._build_trend_metric("Sẵn sàng", self.trend_score_value))
        metric_row.addWidget(self._build_trend_metric("Hiệu quả", self.trend_delta_value))
        panel_layout.addLayout(metric_row)

        back_row = QHBoxLayout()
        back_row.addStretch()
        back_btn = QPushButton("Quay lại")
        back_btn.clicked.connect(self._show_menu)
        back_row.addWidget(back_btn)
        panel_layout.addLayout(back_row)

        layout.addWidget(panel, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        return page

    def _build_trend_metric(self, label: str, value_widget: QLabel) -> QFrame:
        card = QFrame()
        card.setObjectName("trendMetric")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        value_widget.setObjectName("trendValue")
        value_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_widget)

        label_widget = QLabel(label)
        label_widget.setObjectName("trendLabel")
        label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_widget)
        return card

    def _build_instructions_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 12, 18, 12)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(22, 20, 22, 20)
        panel_layout.setSpacing(12)

        title = QLabel("Attention Probe")
        title.setStyleSheet("font-size: 22px; font-weight: 800;")
        panel_layout.addWidget(title)

        body = QLabel(
            "1. Baseline ngắn: lấy mốc phản ứng trong phiên.\n"
            "2. Go/No-Go: bấm Space khi thấy tín hiệu xanh, không bấm khi thấy đỏ.\n"
            "3. Ghi nhớ chuỗi: nhớ ký tự và nhập lại đúng thứ tự.\n"
            "4. Tìm kiếm thị giác: tìm ký tự mục tiêu trong lưới sạch.\n"
            "5. Kết quả: chỉ số chú ý ngắn hạn, không phải kết luận chắc chắn cho phiên làm việc tiếp theo."
        )
        body.setWordWrap(True)
        body.setObjectName("muted")
        panel_layout.addWidget(body)

        key_hint = QLabel(f"Phím thao tác chính: {self.cfg.response_key_name}")
        key_hint.setObjectName("muted")
        panel_layout.addWidget(key_hint)

        row = QHBoxLayout()
        row.addStretch()

        back_btn = QPushButton("Quay lại")
        back_btn.clicked.connect(self._show_menu)
        row.addWidget(back_btn)

        panel_layout.addLayout(row)
        layout.addWidget(panel)
        return page

    def _build_game_select_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 16, 22, 16)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        panel = QFrame()
        panel.setObjectName("panel")
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        panel.setMaximumWidth(980)
        panel.setFixedHeight(488)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(22, 16, 22, 16)
        panel_layout.setSpacing(9)

        title = QLabel("Chọn bài kiểm tra")
        title.setStyleSheet("font-size: 20px; font-weight: 800;")
        panel_layout.addWidget(title)

        subtitle = QLabel("Có thể chạy một bài riêng lẻ hoặc chọn nhiều bài để có summary ổn định hơn.")
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        panel_layout.addWidget(subtitle)

        self.chk_gonogo = QCheckBox("Chọn Go/No-Go")
        self.chk_gonogo.setChecked(True)
        panel_layout.addWidget(
            self._build_game_card(
                "Phản xạ Go/No-Go",
                "Đo readiness phản ứng và khả năng ức chế khi gặp tín hiệu No-Go.",
                "Chỉ số: reaction time, omission, commission, attention stability.",
                self.chk_gonogo,
            )
        )

        self.chk_sequence = QCheckBox("Chọn ghi nhớ chuỗi")
        self.chk_sequence.setChecked(True)
        panel_layout.addWidget(
            self._build_game_card(
                "Ghi nhớ chuỗi",
                "Đo working memory ngắn hạn và tính nhất quán khi nhập lại chuỗi.",
                "Chỉ số: độ chính xác, độ dài tối đa, consistency.",
                self.chk_sequence,
            )
        )

        self.chk_visual = QCheckBox("Chọn tìm kiếm thị giác")
        self.chk_visual.setChecked(True)
        panel_layout.addWidget(
            self._build_game_card(
                "Tìm kiếm thị giác",
                "Đo selective attention và visual scanning trong lưới ký tự rõ ràng.",
                "Chỉ số: độ chính xác, tốc độ hoàn thành, số lần bấm sai.",
                self.chk_visual,
            )
        )

        self.select_status = QLabel("")
        self.select_status.setObjectName("muted")
        panel_layout.addWidget(self.select_status)

        row = QHBoxLayout()

        start_btn = QPushButton("Bắt đầu bài đã chọn")
        start_btn.setObjectName("primary")
        start_btn.clicked.connect(self._start_selected_games_only)
        row.addWidget(start_btn)

        row.addStretch()

        back_btn = QPushButton("Quay lại")
        back_btn.clicked.connect(self._show_menu)
        row.addWidget(back_btn)

        panel_layout.addLayout(row)
        layout.addWidget(panel, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        return page

    def _build_game_card(self, title: str, desc: str, metric_line: str, checkbox: QCheckBox) -> QFrame:
        card = QFrame()
        card.setObjectName("probeOption")

        row = QHBoxLayout(card)
        row.setContentsMargins(14, 12, 14, 12)
        row.setSpacing(10)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        text_col.addWidget(title_label)

        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setObjectName("muted")
        text_col.addWidget(desc_label)

        metrics = QLabel(metric_line)
        metrics.setWordWrap(True)
        metrics.setStyleSheet(f"color: {self.theme.info_text};")
        text_col.addWidget(metrics)

        row.addLayout(text_col, 1)
        row.addWidget(checkbox)

        return card

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(12)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(18, 18, 18, 18)
        panel_layout.setSpacing(10)

        title = QLabel("Cài đặt")
        title.setStyleSheet("font-size: 22px; font-weight: 800;")
        panel_layout.addWidget(title)

        basic_group = QGroupBox("Phiên")
        basic_form = QFormLayout(basic_group)
        basic_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.spin_baseline = QSpinBox()
        self.spin_baseline.setRange(10, 60)
        basic_form.addRow("Mốc ban đầu (s)", self.spin_baseline)

        self.spin_micro_break = QSpinBox()
        self.spin_micro_break.setRange(5, 30)
        basic_form.addRow("Micro-break (s)", self.spin_micro_break)

        self.spin_final_break = QSpinBox()
        self.spin_final_break.setRange(20, 90)
        basic_form.addRow("Nhịp thở cuối (s)", self.spin_final_break)

        self.spin_inhale = QDoubleSpinBox()
        self.spin_inhale.setRange(2.0, 8.0)
        self.spin_inhale.setSingleStep(0.5)
        basic_form.addRow("Hít vào (s)", self.spin_inhale)

        self.spin_exhale = QDoubleSpinBox()
        self.spin_exhale.setRange(3.0, 10.0)
        self.spin_exhale.setSingleStep(0.5)
        basic_form.addRow("Thở ra (s)", self.spin_exhale)

        self.chk_sound = QCheckBox("Bật âm thanh nhắc")
        basic_form.addRow("Âm thanh", self.chk_sound)

        panel_layout.addWidget(basic_group)

        gonogo_group = QGroupBox("Go/No-Go")
        gonogo_form = QFormLayout(gonogo_group)
        gonogo_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.spin_gonogo_duration = QSpinBox()
        self.spin_gonogo_duration.setRange(30, 120)
        gonogo_form.addRow("Thời lượng game (s)", self.spin_gonogo_duration)

        self.spin_gonogo_target_prob = QDoubleSpinBox()
        self.spin_gonogo_target_prob.setRange(0.60, 0.85)
        self.spin_gonogo_target_prob.setDecimals(2)
        self.spin_gonogo_target_prob.setSingleStep(0.01)
        gonogo_form.addRow("Tỉ lệ mục tiêu", self.spin_gonogo_target_prob)

        self.spin_gonogo_stim_ms = QSpinBox()
        self.spin_gonogo_stim_ms.setRange(500, 1200)
        gonogo_form.addRow("Thời gian hiện mục tiêu (ms)", self.spin_gonogo_stim_ms)

        self.spin_gonogo_gap_ms = QSpinBox()
        self.spin_gonogo_gap_ms.setRange(300, 1000)
        gonogo_form.addRow("Khoảng nghỉ giữa 2 mục tiêu (ms)", self.spin_gonogo_gap_ms)

        panel_layout.addWidget(gonogo_group)

        seq_group = QGroupBox("Ghi nhớ chuỗi")
        seq_form = QFormLayout(seq_group)
        seq_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.spin_seq_rounds = QSpinBox()
        self.spin_seq_rounds.setRange(3, 12)
        seq_form.addRow("Số vòng", self.spin_seq_rounds)

        self.spin_seq_start = QSpinBox()
        self.spin_seq_start.setRange(2, 5)
        seq_form.addRow("Độ dài bắt đầu", self.spin_seq_start)

        self.spin_seq_max = QSpinBox()
        self.spin_seq_max.setRange(4, 8)
        seq_form.addRow("Độ dài tối đa", self.spin_seq_max)

        self.spin_seq_show_ms = QSpinBox()
        self.spin_seq_show_ms.setRange(350, 1200)
        seq_form.addRow("Thời gian hiện ký tự (ms)", self.spin_seq_show_ms)

        self.spin_seq_gap_ms = QSpinBox()
        self.spin_seq_gap_ms.setRange(120, 600)
        seq_form.addRow("Khoảng cách giữa ký tự (ms)", self.spin_seq_gap_ms)

        self.spin_seq_timeout = QSpinBox()
        self.spin_seq_timeout.setRange(6, 25)
        seq_form.addRow("Thời gian chờ nhập (s)", self.spin_seq_timeout)

        panel_layout.addWidget(seq_group)

        vis_group = QGroupBox("Tìm kiếm thị giác")
        vis_form = QFormLayout(vis_group)
        vis_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.spin_vis_rounds = QSpinBox()
        self.spin_vis_rounds.setRange(4, 15)
        vis_form.addRow("Số vòng", self.spin_vis_rounds)

        self.spin_vis_grid_start = QSpinBox()
        self.spin_vis_grid_start.setRange(3, 6)
        vis_form.addRow("Kích thước lưới bắt đầu", self.spin_vis_grid_start)

        self.spin_vis_grid_max = QSpinBox()
        self.spin_vis_grid_max.setRange(4, 8)
        vis_form.addRow("Kích thước lưới tối đa", self.spin_vis_grid_max)

        self.spin_vis_timeout = QSpinBox()
        self.spin_vis_timeout.setRange(6, 25)
        vis_form.addRow("Giới hạn mỗi vòng (s)", self.spin_vis_timeout)

        panel_layout.addWidget(vis_group)

        self.settings_status = QLabel("")
        self.settings_status.setObjectName("muted")
        panel_layout.addWidget(self.settings_status)

        button_row = QHBoxLayout()

        save_btn = QPushButton("Lưu")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._save_settings)
        button_row.addWidget(save_btn)

        reset_btn = QPushButton("Đặt lại mặc định")
        reset_btn.clicked.connect(self._reset_settings_defaults)
        button_row.addWidget(reset_btn)

        button_row.addStretch()

        back_btn = QPushButton("Quay lại")
        back_btn.clicked.connect(self._show_menu)
        button_row.addWidget(back_btn)

        panel_layout.addLayout(button_row)

        layout.addWidget(panel)
        return page

    def _build_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 12, 18, 12)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(18, 18, 18, 18)
        panel_layout.setSpacing(10)

        title = QLabel("Lịch sử")
        title.setStyleSheet("font-size: 22px; font-weight: 800;")
        panel_layout.addWidget(title)

        self.history_table = QTableWidget(0, 11)
        self.history_table.setHorizontalHeaderLabels(
            [
                "Thời gian",
                "RT mốc đầu",
                "RT Go/No-Go",
                "Chính xác",
                "Ổn định chú ý",
                "Bài mạnh nhất",
                "Bài yếu nhất",
                "Score Go/No-Go",
                "Score Chuỗi",
                "Score Thị giác",
                "So sánh",
            ]
        )
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        panel_layout.addWidget(self.history_table, 1)

        self.history_status = QLabel("")
        self.history_status.setObjectName("muted")
        panel_layout.addWidget(self.history_status)

        buttons = QHBoxLayout()

        refresh_btn = QPushButton("Làm mới")
        refresh_btn.clicked.connect(self._load_history_table)
        buttons.addWidget(refresh_btn)

        export_btn = QPushButton("Xuất CSV")
        export_btn.clicked.connect(self._export_history_csv)
        buttons.addWidget(export_btn)

        buttons.addStretch()

        back_btn = QPushButton("Quay lại")
        back_btn.clicked.connect(self._show_menu)
        buttons.addWidget(back_btn)

        panel_layout.addLayout(buttons)
        layout.addWidget(panel)
        return page

    def _build_session_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 12, 18, 12)

        top_panel = QFrame()
        top_panel.setObjectName("panel")
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(16, 14, 16, 14)
        top_layout.setSpacing(8)

        row = QHBoxLayout()

        self.phase_title = QLabel("Sẵn sàng")
        self.phase_title.setStyleSheet("font-size: 20px; font-weight: 800;")
        row.addWidget(self.phase_title)

        row.addStretch()

        self.round_label = QLabel("-")
        self.round_label.setObjectName("muted")
        row.addWidget(self.round_label)

        self.remaining_label = QLabel("00.0s")
        self.remaining_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        row.addWidget(self.remaining_label)

        top_layout.addLayout(row)

        self.phase_progress = QProgressBar()
        self.phase_progress.setRange(0, 100)
        self.phase_progress.setValue(0)
        top_layout.addWidget(self.phase_progress)

        layout.addWidget(top_panel)

        stage = QFrame()
        stage.setObjectName("panel")
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(20, 16, 20, 16)
        stage_layout.setSpacing(10)

        self.stage_stack = QStackedWidget()
        stage_layout.addWidget(self.stage_stack, 1)

        layout.addWidget(stage, 1)

        footer = QHBoxLayout()
        footer.addStretch()

        abort_btn = QPushButton("Dừng bài kiểm tra")
        abort_btn.clicked.connect(self._abort_session)
        footer.addWidget(abort_btn)

        layout.addLayout(footer)

        self.stage_attention = self._build_attention_stage()
        self.stage_sequence = self._build_sequence_stage()
        self.stage_visual = self._build_visual_stage()
        self.stage_break = self._build_break_stage()

        self.stage_stack.addWidget(self.stage_attention)
        self.stage_stack.addWidget(self.stage_sequence)
        self.stage_stack.addWidget(self.stage_visual)
        self.stage_stack.addWidget(self.stage_break)

        return page

    def _build_attention_stage(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

        self.stimulus_label = QLabel("")
        self.stimulus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stimulus_label.setFont(QFont("Bahnschrift", 120, QFont.Weight.Bold))
        self.stimulus_label.setMinimumHeight(230)
        self.stimulus_label.setStyleSheet("border: none;")
        layout.addWidget(self.stimulus_label)

        self.attention_hint = QLabel("Bấm Space khi thấy tín hiệu xanh")
        self.attention_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.attention_hint.setObjectName("muted")
        layout.addWidget(self.attention_hint)

        self.attention_feedback = QLabel("")
        self.attention_feedback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.attention_feedback.setWordWrap(True)
        self.attention_feedback.setMinimumHeight(48)
        self._clear_feedback_banner(self.attention_feedback)
        layout.addWidget(self.attention_feedback)

        return w

    def _build_sequence_stage(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)

        self.sequence_mode_label = QLabel("Ghi nhớ")
        self.sequence_mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sequence_mode_label.setStyleSheet("font-size: 18px; font-weight: 800;")
        layout.addWidget(self.sequence_mode_label)

        self.sequence_show_label = QLabel("-")
        self.sequence_show_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sequence_show_label.setFont(QFont("Bahnschrift", 88, QFont.Weight.Bold))
        self.sequence_show_label.setMinimumHeight(200)
        layout.addWidget(self.sequence_show_label)

        self.sequence_input_label = QLabel("")
        self.sequence_input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sequence_input_label.setObjectName("muted")
        self.sequence_input_label.setVisible(False)
        layout.addWidget(self.sequence_input_label)

        self.sequence_feedback = QLabel("")
        self.sequence_feedback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sequence_feedback.setWordWrap(True)
        self.sequence_feedback.setMinimumHeight(48)
        self._clear_feedback_banner(self.sequence_feedback)
        layout.addWidget(self.sequence_feedback)

        self.sequence_buttons_row = QHBoxLayout()
        self.sequence_buttons_row.setSpacing(8)
        layout.addLayout(self.sequence_buttons_row)

        self.sequence_symbol_buttons: list[QPushButton] = []
        return w

    def _build_visual_stage(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        self.visual_instruction = QLabel("Tìm và bấm vào ký tự mục tiêu")
        self.visual_instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visual_instruction.setStyleSheet("font-size: 17px; font-weight: 700;")
        layout.addWidget(self.visual_instruction)

        self.visual_status = QLabel("")
        self.visual_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visual_status.setWordWrap(True)
        self.visual_status.setMinimumHeight(48)
        self._clear_feedback_banner(self.visual_status)
        layout.addWidget(self.visual_status)

        self.visual_grid_widget = QWidget()
        self.visual_grid_layout = QGridLayout(self.visual_grid_widget)
        self.visual_grid_layout.setSpacing(6)
        layout.addWidget(self.visual_grid_widget, 1)

        self.visual_buttons: list[QPushButton] = []
        return w

    def _build_break_stage(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

        self.breath_widget = BreathingCircleWidget()
        layout.addWidget(self.breath_widget, 0, Qt.AlignmentFlag.AlignCenter)

        self.break_phase_label = QLabel("Hít vào")
        self.break_phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.break_phase_label.setFont(QFont("Bahnschrift", 30, QFont.Weight.Bold))
        layout.addWidget(self.break_phase_label)

        self.break_countdown_label = QLabel("")
        self.break_countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.break_countdown_label.setObjectName("muted")
        layout.addWidget(self.break_countdown_label)

        return w

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        layout.setContentsMargins(18, 12, 18, 12)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(20, 16, 20, 16)
        panel_layout.setSpacing(8)

        # Result metric labels (created for API compatibility, not displayed in layout)
        self.result_probe_score = QLabel("-")
        self.result_attention_stability = QLabel("-")
        self.result_accuracy = QLabel("-")
        self.result_avg_rt = QLabel("-")
        self.result_rt_var = QLabel("-")
        self.result_omissions = QLabel("-")
        self.result_commissions = QLabel("-")
        self.result_best = QLabel("-")
        self.result_weakest = QLabel("-")
        self.result_gonogo = QLabel("-")
        self.result_stroop = QLabel("-")
        self.result_flanker = QLabel("-")
        self.result_sequence = QLabel("-")
        self.result_visual = QLabel("-")

        self.result_feedback = QLabel("")
        self.result_feedback.setWordWrap(True)
        self.result_feedback.setStyleSheet(
            f"background-color: {self.theme.panel_soft}; border: 1px solid {self.theme.border}; "
            f"border-radius: 8px; color: {self.theme.text_primary}; padding: 12px; font-size: 15px;"
        )
        panel_layout.addWidget(self.result_feedback)

        self.result_disclaimer = QLabel(
            "Kết quả này chỉ phản ánh khả năng phản ứng trong bài kiểm tra ngắn. "
            "Hiệu quả phục hồi sẽ được xác nhận thêm khi bạn quay lại làm việc trong vài phút tiếp theo."
        )
        self.result_disclaimer.setObjectName("finePrint")
        self.result_disclaimer.setWordWrap(True)
        panel_layout.addWidget(self.result_disclaimer)

        ready_title = QLabel("Bạn đã sẵn sàng quay lại làm việc chưa?")
        ready_title.setStyleSheet("font-size: 14px; font-weight: 700;")
        panel_layout.addWidget(ready_title)

        self.ready_group = QButtonGroup(self)
        self.ready_group.setExclusive(True)
        self.ready_buttons: list[QPushButton] = []
        ready_row = QHBoxLayout()
        ready_row.setSpacing(6)
        ready_options = [
            (1, "1\nChưa sẵn sàng", "Chưa sẵn sàng"),
            (2, "2\nHơi mệt", "Hơi mệt"),
            (3, "3\nBình thường", "Bình thường"),
            (4, "4\nKhá sẵn sàng", "Khá sẵn sàng"),
            (5, "5\nRất sẵn sàng", "Rất sẵn sàng"),
        ]
        for value, label, tooltip in ready_options:
            btn = QPushButton(label)
            btn.setObjectName("readyOption")
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setMinimumHeight(64)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _checked=False, v=value: self._set_self_report_ready(v))
            self.ready_group.addButton(btn, value)
            self.ready_buttons.append(btn)
            ready_row.addWidget(btn)
        panel_layout.addLayout(ready_row)

        row = QHBoxLayout()

        replay_btn = QPushButton("Kiểm tra lại")
        replay_btn.clicked.connect(self._start_auto_probe)
        row.addWidget(replay_btn)

        row.addStretch()

        work_btn = QPushButton("Quay lại làm việc")
        work_btn.setObjectName("primary")
        work_btn.clicked.connect(self.accept)
        row.addWidget(work_btn)

        panel_layout.addLayout(row)
        layout.addWidget(panel)
        return page

    def _add_result_row(self, grid: QGridLayout, row: int, label: str, value_widget: QLabel) -> None:
        item = QFrame()
        item.setObjectName("metricRow")
        item.setFixedHeight(42)
        item.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        item_layout = QHBoxLayout(item)
        item_layout.setContentsMargins(10, 5, 10, 5)
        item_layout.setSpacing(8)

        icon_text = {
            0: "A",
            1: "%",
            2: "RT",
            3: "~",
            4: "M",
            5: "W",
            6: "S",
            7: "+",
            8: "!",
            9: "G",
            10: "C",
            11: "F",
            12: "Q",
            13: "V",
        }.get(row, "*")

        icon = QLabel(icon_text)
        icon.setObjectName("metricRowIcon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setFixedWidth(18)

        left = QLabel(label)
        left.setObjectName("metricRowLabel")
        left.setWordWrap(False)

        value_widget.setObjectName("metricRowValue")
        value_widget.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value_widget.setMinimumWidth(132)
        value_widget.setWordWrap(False)

        item_layout.addWidget(icon)
        item_layout.addWidget(left, 1)
        item_layout.addWidget(value_widget)
        grid_row = row // 2
        grid.addWidget(item, grid_row, row % 2)
        grid.setRowMinimumHeight(grid_row, 42)

    def _set_self_report_ready(self, value: int) -> None:
        self._self_report_ready = max(1, min(5, int(value)))
        for btn in getattr(self, "ready_buttons", []):
            btn.setChecked(self.ready_group.id(btn) == self._self_report_ready)

    def _reset_ready_selection(self) -> None:
        self._self_report_ready = None
        if not hasattr(self, "ready_group"):
            return
        self.ready_group.setExclusive(False)
        for btn in self.ready_buttons:
            btn.setChecked(False)
        self.ready_group.setExclusive(True)

    def _history_attention_score(self, record: dict) -> float | None:
        return AttentionTrendWidget.score_from_record(record)

    def _refresh_menu_history_summary(self) -> None:
        if not hasattr(self, "menu_trend_chart"):
            return

        records = self.storage.load_recovery_validations()
        recent = records[-10:]
        self.menu_trend_chart.set_records(recent)

        if not recent:
            self.trend_sessions_value.setText("0")
            self.trend_score_value.setText("-")
            self.trend_delta_value.setText("-")
            self.trend_delta_value.setStyleSheet("")
            return

        scores = [score for item in recent if (score := self._history_attention_score(item)) is not None]
        if not scores:
            self.trend_sessions_value.setText(str(len(records)))
            self.trend_score_value.setText("-")
            self.trend_delta_value.setText("-")
            self.trend_delta_value.setStyleSheet("")
            return

        self.trend_sessions_value.setText(str(len(records)))
        latest_record = recent[-1]
        post_ready = AttentionTrendWidget._as_float(latest_record.get("post_work_readiness"), -1.0)
        latest_transfer = scores[-1]
        self.trend_score_value.setText(f"{post_ready:.0f}" if post_ready >= 0.0 else "-")
        self.trend_delta_value.setText(f"{latest_transfer:.0f}")
        if latest_transfer >= 65.0:
            self.trend_delta_value.setStyleSheet(f"color: {self.theme.success_text};")
        elif latest_transfer < 40.0:
            self.trend_delta_value.setStyleSheet(f"color: {self.theme.error_text};")
        else:
            self.trend_delta_value.setStyleSheet("")

    def _show_menu(self) -> None:
        self._stop_all_timers()
        self._refresh_menu_history_summary()
        self._set_current_page(self.page_menu)

    def _show_recovery_results(self) -> None:
        self._refresh_menu_history_summary()
        self._set_current_page(self.page_recovery_results)

    def _show_instructions(self) -> None:
        self._set_current_page(self.page_instructions)

    def _show_game_select(self) -> None:
        self.select_status.setText("")
        self._set_current_page(self.page_select)

    def _show_settings(self) -> None:
        self._load_settings_to_controls()
        self.settings_status.setText("")
        self._set_current_page(self.page_settings)

    def _show_history(self) -> None:
        self._load_history_table()
        self._set_current_page(self.page_history)

    def _load_settings_to_controls(self) -> None:
        self.spin_baseline.setValue(int(self.cfg.baseline_duration_s))
        self.spin_micro_break.setValue(int(self.cfg.micro_break_s))
        self.spin_final_break.setValue(int(self.cfg.final_breathing_break_s))
        self.spin_inhale.setValue(float(self.cfg.inhale_seconds))
        self.spin_exhale.setValue(float(self.cfg.exhale_seconds))
        self.chk_sound.setChecked(bool(self.cfg.sound_enabled))
        self.chk_sound.setEnabled(self._app_sound_enabled)
        if not self._app_sound_enabled:
            self.chk_sound.setToolTip("Bat am thanh chung trong app de su dung feedback sound.")
        else:
            self.chk_sound.setToolTip("")

        self.spin_gonogo_duration.setValue(int(self.cfg.gonogo.round_duration_s))
        self.spin_gonogo_target_prob.setValue(float(self.cfg.gonogo.target_probability))
        self.spin_gonogo_stim_ms.setValue(int(self.cfg.gonogo.stimulus_duration_ms))
        self.spin_gonogo_gap_ms.setValue(int(self.cfg.gonogo.inter_stimulus_ms))

        self.spin_seq_rounds.setValue(int(self.cfg.sequence.rounds))
        self.spin_seq_start.setValue(int(self.cfg.sequence.start_length))
        self.spin_seq_max.setValue(int(self.cfg.sequence.max_length))
        self.spin_seq_show_ms.setValue(int(self.cfg.sequence.show_item_ms))
        self.spin_seq_gap_ms.setValue(int(self.cfg.sequence.gap_ms))
        self.spin_seq_timeout.setValue(int(self.cfg.sequence.input_timeout_s))

        self.spin_vis_rounds.setValue(int(self.cfg.visual.rounds))
        self.spin_vis_grid_start.setValue(int(self.cfg.visual.grid_start))
        self.spin_vis_grid_max.setValue(int(self.cfg.visual.grid_max))
        self.spin_vis_timeout.setValue(int(self.cfg.visual.round_timeout_s))

    def _save_settings(self) -> None:
        if self.spin_seq_max.value() < self.spin_seq_start.value():
            self.spin_seq_max.setValue(self.spin_seq_start.value())

        if self.spin_vis_grid_max.value() < self.spin_vis_grid_start.value():
            self.spin_vis_grid_max.setValue(self.spin_vis_grid_start.value())

        self.cfg.baseline_duration_s = int(self.spin_baseline.value())
        self.cfg.micro_break_s = int(self.spin_micro_break.value())
        self.cfg.final_breathing_break_s = int(self.spin_final_break.value())
        self.cfg.inhale_seconds = float(self.spin_inhale.value())
        self.cfg.exhale_seconds = float(self.spin_exhale.value())
        self.cfg.sound_enabled = bool(self.chk_sound.isChecked() and self._app_sound_enabled)

        self.cfg.gonogo.round_duration_s = int(self.spin_gonogo_duration.value())
        self.cfg.gonogo.target_probability = float(self.spin_gonogo_target_prob.value())
        self.cfg.gonogo.stimulus_duration_ms = int(self.spin_gonogo_stim_ms.value())
        self.cfg.gonogo.inter_stimulus_ms = int(self.spin_gonogo_gap_ms.value())

        self.cfg.sequence.rounds = int(self.spin_seq_rounds.value())
        self.cfg.sequence.start_length = int(self.spin_seq_start.value())
        self.cfg.sequence.max_length = int(self.spin_seq_max.value())
        self.cfg.sequence.show_item_ms = int(self.spin_seq_show_ms.value())
        self.cfg.sequence.gap_ms = int(self.spin_seq_gap_ms.value())
        self.cfg.sequence.input_timeout_s = int(self.spin_seq_timeout.value())

        self.cfg.visual.rounds = int(self.spin_vis_rounds.value())
        self.cfg.visual.grid_start = int(self.spin_vis_grid_start.value())
        self.cfg.visual.grid_max = int(self.spin_vis_grid_max.value())
        self.cfg.visual.round_timeout_s = int(self.spin_vis_timeout.value())

        save_focus_reset_config(self.cfg)
        self.storage = SessionStorage(self.cfg.history_path)

        self._rebuild_sequence_symbol_buttons()
        self.settings_status.setText(f"Đã lưu cài đặt vào {self.cfg.settings_path}")

    def _reset_settings_defaults(self) -> None:
        self.cfg = FocusResetConfig()
        self._load_settings_to_controls()
        self._rebuild_sequence_symbol_buttons()
        self.settings_status.setText("Đã tải giá trị mặc định. Bấm Lưu để áp dụng.")

    def _collect_selected_games(self) -> list[str]:
        selected: list[str] = []
        if hasattr(self, "chk_gonogo") and self.chk_gonogo.isChecked():
            selected.append("gonogo")
        if hasattr(self, "chk_stroop") and self.chk_stroop.isChecked():
            selected.append("stroop")
        if hasattr(self, "chk_flanker") and self.chk_flanker.isChecked():
            selected.append("flanker")
        if hasattr(self, "chk_sequence") and self.chk_sequence.isChecked():
            selected.append("sequence")
        if hasattr(self, "chk_visual") and self.chk_visual.isChecked():
            selected.append("visual")
        return selected

    def _context_float(self, *keys: str, default: float = 0.0) -> float:
        for key in keys:
            if key not in self._break_context:
                continue
            try:
                return float(self._break_context.get(key))
            except (TypeError, ValueError):
                continue
        return default

    def _choose_auto_probe_games(self) -> list[str]:
        """
        Build a personalized task bundle instead of random fixed selection.

        Signals used: planned break length, pre-break work readiness,
        fatigue/distraction channels, recent task exposure, and post-return
        transfer validation from prior breaks.
        """
        all_games = list(self.GAME_ORDER)
        recent_rows = self.storage.load()[-10:]
        validations = self.storage.load_recovery_validations()[-20:]

        recent_ids: list[str] = []
        for row in recent_rows:
            stored = row.get("selected_games")
            if isinstance(stored, list):
                recent_ids.extend(str(item) for item in stored if str(item) in all_games)
                continue
            for game_id, score_key in (
                ("gonogo", "score_gonogo"),
                ("stroop", "score_stroop"),
                ("flanker", "score_flanker"),
                ("sequence", "score_sequence"),
                ("visual", "score_visual"),
            ):
                try:
                    if float(row.get(score_key, 0.0) or 0.0) > 0:
                        recent_ids.append(game_id)
                except (TypeError, ValueError):
                    pass

        transfer_by_game: dict[str, list[float]] = {game_id: [] for game_id in all_games}
        for row in validations:
            transfer = AttentionTrendWidget.score_from_record(row)
            if transfer is None:
                continue
            selected = row.get("selected_games")
            if not isinstance(selected, list):
                selected = []
            for game_id in selected:
                if game_id in transfer_by_game:
                    transfer_by_game[game_id].append(float(transfer))

        break_minutes = self._context_float(
            "break_duration_minutes",
            "planned_break_minutes",
            "break_minutes",
            default=float(self.cfg.micro_break_s) / 60.0,
        )
        pre_ready = self._context_float("pre_work_readiness", "work_readiness", default=55.0)
        fatigue = self._context_float("fatigue_index", "fatigue", default=0.0)
        distraction = self._context_float("distraction_risk", "distraction", default=0.0)
        profile_name = str(self._break_context.get("profile_name", "default") or "default")

        recent_transfer = AttentionTrendWidget.score_from_record(validations[-1]) if validations else None
        if break_minutes < 4.0 and pre_ready >= 55.0:
            target_count = 1
        elif break_minutes >= 8.0 and fatigue < 0.70 and pre_ready >= 45.0:
            target_count = 3
        else:
            target_count = 2
        if recent_transfer is not None and recent_transfer < 45.0:
            target_count = min(target_count, 2)

        weights = {
            "gonogo": 1.05,
            "stroop": 1.00,
            "flanker": 1.00,
            "sequence": 0.95,
            "visual": 0.95,
        }

        if fatigue >= 0.65 or pre_ready < 45.0:
            weights["visual"] += 0.75
            weights["gonogo"] += 0.25
            weights["sequence"] -= 0.25
            weights["stroop"] -= 0.20
            weights["flanker"] -= 0.15
        elif distraction >= 0.55:
            weights["gonogo"] += 0.75
            weights["flanker"] += 0.45
            weights["stroop"] += 0.30
        elif pre_ready >= 70.0:
            weights["stroop"] += 0.45
            weights["flanker"] += 0.40
            weights["sequence"] += 0.25

        if break_minutes >= 8.0:
            weights["sequence"] += 0.25
            weights["visual"] += 0.20
        elif break_minutes < 4.0:
            weights["sequence"] -= 0.20

        for game_id, transfers in transfer_by_game.items():
            if not transfers:
                continue
            avg_transfer = sum(transfers[-4:]) / len(transfers[-4:])
            if avg_transfer >= 65.0:
                weights[game_id] += 0.35
            elif avg_transfer < 40.0:
                weights[game_id] -= 0.35

        for game_id in all_games:
            weights[game_id] -= min(0.55, recent_ids.count(game_id) * 0.16)
            weights[game_id] = max(0.05, weights[game_id])

        seed = f"{profile_name}:{datetime.now().date().isoformat()}:{len(recent_rows)}:{len(validations)}:{int(break_minutes * 10)}"
        rng = random.Random(seed)
        ranked = sorted(
            all_games,
            key=lambda game_id: weights[game_id] + rng.uniform(0.0, 0.08),
            reverse=True,
        )

        selected: list[str] = []
        inhibition = [game for game in ranked if game in {"gonogo", "stroop", "flanker"}]
        if inhibition and target_count >= 2:
            selected.append(inhibition[0])

        for game_id in ranked:
            if game_id not in selected:
                selected.append(game_id)
            if len(selected) >= target_count:
                break

        return selected[:target_count] or ["gonogo"]

    def _start_auto_probe(self) -> None:
        self._start_session_for_games(self._choose_auto_probe_games(), include_baseline=True)

    def _start_recovery_session(self) -> None:
        """Backward-compatible entrypoint for older callers."""
        self._start_auto_probe()

    def _start_session_for_games(self, games: list[str], include_baseline: bool) -> None:
        valid = [game for game in games if game in self.GAME_ORDER]
        if not valid:
            valid = self._choose_auto_probe_games()

        for attr, game_id in (
            ("chk_gonogo", "gonogo"),
            ("chk_stroop", "stroop"),
            ("chk_flanker", "flanker"),
            ("chk_sequence", "sequence"),
            ("chk_visual", "visual"),
        ):
            checkbox = getattr(self, attr, None)
            if checkbox is not None:
                checkbox.setChecked(game_id in valid)
        self._start_session(include_baseline=include_baseline, selected_games=valid)

    def _start_selected_games_only(self) -> None:
        self._start_session(include_baseline=True)

    def _start_session(self, include_baseline: bool, selected_games: list[str] | None = None) -> None:
        selected = [game for game in (selected_games or self._collect_selected_games()) if game in self.GAME_ORDER]
        if not selected:
            self.select_status.setText("Vui lòng chọn ít nhất một bài kiểm tra.")
            self._set_current_page(self.page_select)
            return

        self._reset_runtime_state()
        self._selected_games = selected

        self._session_steps = []
        if include_baseline:
            self._session_steps.append("baseline")

        for idx, game in enumerate(self._selected_games):
            self._session_steps.append(game)
            if idx < len(self._selected_games) - 1:
                self._session_steps.append("break")

        self._session_steps.append("final_break")

        self._set_current_page(self.page_session)
        self._step_index = -1
        self._advance_step()

    def _reset_runtime_state(self) -> None:
        self._selected_games = list(self.GAME_ORDER)
        self._session_steps: list[str] = []
        self._step_index = -1

        self._phase_mode = "idle"
        self._phase_started_at = 0.0
        self._phase_duration_ms = 0

        self._phase_trials: list[TrialSpec] = []
        self._phase_responses: dict[int, int] = {}
        self._phase_extra_commissions = 0

        self._baseline_summary: MetricSummary | None = None
        self._gonogo_summary: MetricSummary | None = None
        self._stroop_summary: MetricSummary | None = None
        self._flanker_summary: MetricSummary | None = None
        self._sequence_summary: SequenceSummary | None = None
        self._visual_summary: VisualSummary | None = None
        self._final_summary: SessionSummary | None = None
        self._attention_trial_payloads: dict[int, dict] = {}

        self._sequence_round_sequences: list[list[str]] = []
        self._sequence_round_lengths: list[int] = []
        self._sequence_round_index = 0
        self._sequence_expected: list[str] = []
        self._sequence_input: list[str] = []
        self._sequence_input_started_at = 0.0
        self._sequence_results: list[SequenceRoundResult] = []

        self._visual_specs: list[VisualRoundSpec] = []
        self._visual_round_index = 0
        self._visual_round_started_at = 0.0
        self._visual_round_miss_clicks = 0
        self._visual_target_index = -1
        self._visual_round_resolved = False
        self._visual_results: list[VisualRoundResult] = []

        self._self_report_ready: int | None = None
        self._result_created_at = ""
        self._last_attention_trial_index: int | None = None
        self._last_attention_is_target: bool | None = None
        if hasattr(self, "ready_group"):
            self._reset_ready_selection()

    def _advance_step(self) -> None:
        self._stop_all_timers()
        self._step_index += 1
        if self._step_index >= len(self._session_steps):
            self._finish_recovery_session()
            return

        step = self._session_steps[self._step_index]
        if step == "baseline":
            self._start_baseline_phase()
        elif step == "gonogo":
            self._start_gonogo_phase()
        elif step == "stroop":
            self._start_stroop_phase()
        elif step == "flanker":
            self._start_flanker_phase()
        elif step == "sequence":
            self._start_sequence_phase()
        elif step == "visual":
            self._start_visual_phase()
        elif step == "break":
            self._start_break_phase(final=False)
        elif step == "final_break":
            self._start_break_phase(final=True)

    def _start_baseline_phase(self) -> None:
        self._start_attention_phase(
            mode="baseline",
            duration_s=int(self.cfg.baseline_duration_s),
            title="Baseline chú ý ngắn",
            subtitle="Mốc phản ứng trong phiên",
            hint="Bấm Space khi thấy tín hiệu xanh. Không bấm khi thấy đỏ.",
        )

    def _start_gonogo_phase(self) -> None:
        self._start_attention_phase(
            mode="gonogo",
            duration_s=int(self.cfg.gonogo.round_duration_s),
            title="Phản xạ Go/No-Go",
            subtitle=self._game_position_label("gonogo"),
            hint="Bấm khi thấy xanh, giữ tay khi thấy đỏ.",
        )

    def _start_stroop_phase(self) -> None:
        self._start_attention_phase(
            mode="stroop",
            duration_s=max(24, min(40, int(self.cfg.gonogo.round_duration_s))),
            title="Stroop màu",
            subtitle=self._game_position_label("stroop"),
            hint="Bấm Space khi chữ và màu trùng nhau. Không bấm khi khác nhau.",
        )

    def _start_flanker_phase(self) -> None:
        self._start_attention_phase(
            mode="flanker",
            duration_s=max(24, min(40, int(self.cfg.gonogo.round_duration_s))),
            title="Mũi tên Flanker",
            subtitle=self._game_position_label("flanker"),
            hint="Bấm Space khi mũi tên giữa hướng sang phải.",
        )

    def _start_attention_phase(
        self,
        mode: str,
        duration_s: int,
        title: str,
        subtitle: str,
        hint: str,
    ) -> None:
        self._phase_mode = mode
        self._phase_duration_ms = max(1, int(duration_s * 1000))
        self._phase_started_at = time.perf_counter()
        self._phase_responses = {}
        self._phase_extra_commissions = 0
        self._phase_trials = build_gonogo_trials(self.cfg.gonogo, duration_s=duration_s)
        self._attention_trial_payloads = self._build_attention_trial_payloads(mode, self._phase_trials)

        self.phase_title.setText(title)
        self.round_label.setText(subtitle)
        self.remaining_label.setText(f"{duration_s:04.1f}s")
        self.phase_progress.setValue(0)

        self.stage_stack.setCurrentWidget(self.stage_attention)
        self.attention_hint.setText(hint)
        self._clear_feedback_banner(self.attention_feedback)
        self.stimulus_label.setText("")
        self.stimulus_label.setStyleSheet("border: none;")
        self._last_attention_trial_index = None
        self._last_attention_is_target = None

        self._phase_timer.start(16)
        self.setFocus()

    def _build_attention_trial_payloads(self, mode: str, trials: list[TrialSpec]) -> dict[int, dict]:
        rng = random.Random(time.time_ns())
        payloads: dict[int, dict] = {}

        if mode == "stroop":
            palette = [
                ("ĐỎ", self.theme.nogo_color),
                ("XANH", self.theme.target_color),
                ("VÀNG", self.theme.titlebar_dot_min),
                ("LAM", self.theme.info_text),
            ]
            for trial in trials:
                color_name, color_value = rng.choice(palette)
                if trial.is_target:
                    word = color_name
                else:
                    word = rng.choice([name for name, _color in palette if name != color_name])
                payloads[trial.index] = {
                    "text": word,
                    "color": color_value,
                    "font_size": 64,
                }

        elif mode == "flanker":
            for trial in trials:
                center = ">" if trial.is_target else "<"
                flank = center if rng.random() < 0.55 else (">" if center == "<" else "<")
                payloads[trial.index] = {
                    "text": f"{flank}{flank}{center}{flank}{flank}",
                    "color": self.theme.text_primary,
                    "font_size": 78,
                }

        return payloads

    def _start_sequence_phase(self) -> None:
        self.phase_title.setText("Ghi nhớ chuỗi")
        self.round_label.setText(self._game_position_label("sequence"))
        self.phase_progress.setValue(0)
        self.remaining_label.setText(f"{self.cfg.sequence.input_timeout_s:04.1f}s")

        self.stage_stack.setCurrentWidget(self.stage_sequence)
        self._clear_feedback_banner(self.sequence_feedback)

        self._sequence_results = []
        self._sequence_round_index = 0
        self._sequence_round_lengths = build_round_lengths(self.cfg.sequence)
        self._sequence_round_sequences = [
            build_sequence(self.cfg.sequence.symbols, length)
            for length in self._sequence_round_lengths
        ]

        self._start_sequence_round()

    def _start_sequence_round(self) -> None:
        if self._sequence_round_index >= len(self._sequence_round_sequences):
            self._finish_sequence_phase()
            return

        self._stop_all_timers(keep_sequence_token=False)

        self._phase_mode = "sequence_show"
        self._sequence_expected = list(self._sequence_round_sequences[self._sequence_round_index])
        self._sequence_input = []

        self.sequence_mode_label.setText(
            f"Ghi nhớ chuỗi {self._sequence_round_index + 1}/{len(self._sequence_round_sequences)}"
        )
        self.sequence_input_label.setText("")
        self._clear_feedback_banner(self.sequence_feedback)

        for btn in self.sequence_symbol_buttons:
            btn.setEnabled(False)

        self._sequence_token += 1
        token = self._sequence_token
        self._play_sequence_symbol(token=token, index=0)

    def _play_sequence_symbol(self, token: int, index: int) -> None:
        if token != self._sequence_token or self._phase_mode != "sequence_show":
            return

        if index >= len(self._sequence_expected):
            self.sequence_show_label.setText("...")
            QTimer.singleShot(200, lambda t=token: self._enter_sequence_input(t))
            return

        self.sequence_show_label.setText(self._sequence_expected[index])
        QTimer.singleShot(
            int(self.cfg.sequence.show_item_ms),
            lambda t=token, i=index: self._clear_sequence_symbol(t, i),
        )

    def _clear_sequence_symbol(self, token: int, index: int) -> None:
        if token != self._sequence_token or self._phase_mode != "sequence_show":
            return

        self.sequence_show_label.setText("•")
        QTimer.singleShot(
            int(self.cfg.sequence.gap_ms),
            lambda t=token, i=index + 1: self._play_sequence_symbol(t, i),
        )

    def _render_sequence_input_progress(self) -> None:
        expected_len = len(self._sequence_expected)
        slots = [
            self._sequence_input[idx] if idx < len(self._sequence_input) else "_"
            for idx in range(expected_len)
        ]
        self.sequence_show_label.setText(" ".join(slots))

    def _enter_sequence_input(self, token: int) -> None:
        if token != self._sequence_token or self._phase_mode != "sequence_show":
            return

        self._phase_mode = "sequence_input"
        self._phase_duration_ms = int(self.cfg.sequence.input_timeout_s * 1000)
        self._phase_started_at = time.perf_counter()

        self.sequence_mode_label.setText("Nhập lại chuỗi")
        self._render_sequence_input_progress()
        self.sequence_input_label.setText("")

        for btn in self.sequence_symbol_buttons:
            btn.setEnabled(True)

        self._sequence_input_started_at = time.perf_counter()
        self._phase_timer.start(40)
        self._sequence_timeout.start(self._phase_duration_ms)
        self.setFocus()

    def _on_sequence_symbol(self, symbol: str) -> None:
        if self._phase_mode != "sequence_input":
            return

        self._sequence_input.append(symbol)
        self._render_sequence_input_progress()

        if len(self._sequence_input) >= len(self._sequence_expected):
            self._finalize_sequence_round(timeout=False)

    def _on_sequence_backspace(self) -> None:
        if self._phase_mode != "sequence_input" or not self._sequence_input:
            return

        self._sequence_input.pop()
        self._render_sequence_input_progress()
        self._clear_feedback_banner(self.sequence_feedback)

    def _sequence_mistake_count(self, timeout: bool) -> int:
        compared = zip(self._sequence_input, self._sequence_expected)
        mismatches = sum(1 for entered, expected in compared if entered != expected)
        missing = max(0, len(self._sequence_expected) - len(self._sequence_input))
        extra = max(0, len(self._sequence_input) - len(self._sequence_expected))
        timeout_penalty = 1 if timeout and missing == 0 else 0
        return int(mismatches + missing + extra + timeout_penalty)

    @pyqtSlot()
    def _on_sequence_timeout(self) -> None:
        if self._phase_mode != "sequence_input":
            return
        self._finalize_sequence_round(timeout=True)

    def _finalize_sequence_round(self, timeout: bool) -> None:
        if self._phase_mode != "sequence_input":
            return

        self._stop_all_timers(keep_sequence_token=True)

        for btn in self.sequence_symbol_buttons:
            btn.setEnabled(False)

        elapsed_ms = (time.perf_counter() - self._sequence_input_started_at) * 1000.0
        if timeout:
            elapsed_ms = float(self.cfg.sequence.input_timeout_s * 1000)

        mistake_count = self._sequence_mistake_count(timeout=timeout)
        correct = (self._sequence_input == self._sequence_expected) and not timeout

        self._sequence_results.append(
            SequenceRoundResult(
                round_index=self._sequence_round_index,
                sequence_length=len(self._sequence_expected),
                correct=correct,
                response_time_ms=max(0.0, elapsed_ms),
                mistakes=mistake_count,
            )
        )

        if correct:
            self._set_feedback_banner(self.sequence_feedback, "success", "Đúng")
            self._play_feedback_sound("success")
        elif timeout:
            self._set_feedback_banner(self.sequence_feedback, "error", "Hết giờ")
            self._play_feedback_sound("error")
        else:
            self._set_feedback_banner(self.sequence_feedback, "error", "Chưa đúng thứ tự")
            self._play_feedback_sound("error")

        self._sequence_round_index += 1
        progress = int((self._sequence_round_index / max(1, len(self._sequence_round_sequences))) * 100)
        self.phase_progress.setValue(max(0, min(100, progress)))
        self.remaining_label.setText("00.0s")

        QTimer.singleShot(500, self._start_sequence_round)

    def _finish_sequence_phase(self) -> None:
        self._sequence_summary = evaluate_sequence(self._sequence_results)
        self._set_feedback_banner(
            self.sequence_feedback,
            "info",
            f"Hoàn thành - Chính xác: {self._sequence_summary.accuracy:.1f}% | Chuỗi dài nhất: {self._sequence_summary.max_sequence_length}"
        )
        self._play_feedback_sound("info")
        QTimer.singleShot(550, self._advance_step)

    def _start_visual_phase(self) -> None:
        self.phase_title.setText("Tìm kiếm thị giác")
        self.round_label.setText(self._game_position_label("visual"))
        self.stage_stack.setCurrentWidget(self.stage_visual)

        self._visual_specs = build_visual_specs(self.cfg.visual)
        self._visual_results = []
        self._visual_round_index = 0
        self._start_visual_round()

    def _start_visual_round(self) -> None:
        if self._visual_round_index >= len(self._visual_specs):
            self._finish_visual_phase()
            return

        spec = self._visual_specs[self._visual_round_index]
        self._visual_target_index = int(spec.target_index)
        self._visual_round_miss_clicks = 0
        self._visual_round_resolved = False

        self._build_visual_grid(spec)

        self.visual_instruction.setText(
            f"Vòng {self._visual_round_index + 1}/{len(self._visual_specs)} - Tìm '{spec.target_symbol}'"
        )
        self._clear_feedback_banner(self.visual_status)

        self._phase_mode = "visual"
        self._phase_duration_ms = int(self.cfg.visual.round_timeout_s * 1000)
        self._phase_started_at = time.perf_counter()
        self._visual_round_started_at = self._phase_started_at
        self.phase_progress.setValue(0)
        self.remaining_label.setText(f"{self.cfg.visual.round_timeout_s:04.1f}s")

        self._phase_timer.start(40)

    def _build_visual_grid(self, spec: VisualRoundSpec) -> None:
        self._clear_layout(self.visual_grid_layout)
        self.visual_buttons = []

        style = self._interactive_button_style()
        for idx in range(spec.rows * spec.cols):
            text = spec.target_symbol if idx == spec.target_index else spec.distractor_symbol
            btn = QPushButton(text)
            btn.setStyleSheet(style)
            btn.clicked.connect(lambda _checked=False, i=idx: self._on_visual_cell_clicked(i))
            self.visual_grid_layout.addWidget(btn, idx // spec.cols, idx % spec.cols)
            self.visual_buttons.append(btn)

    def _on_visual_cell_clicked(self, index: int) -> None:
        if self._phase_mode != "visual" or self._visual_round_resolved:
            return

        if index == self._visual_target_index:
            self._set_feedback_banner(self.visual_status, "success", "Đã tìm thấy mục tiêu")
            self._play_feedback_sound("success")
            self._finish_visual_round(correct=True, timeout=False)
            return

        self._visual_round_miss_clicks += 1
        self._set_feedback_banner(self.visual_status, "error", f"Bấm sai: {self._visual_round_miss_clicks}")
        self._play_feedback_sound("error")

    def _finish_visual_round(self, correct: bool, timeout: bool) -> None:
        if self._visual_round_resolved:
            return

        self._visual_round_resolved = True
        self._stop_all_timers(keep_sequence_token=True)

        elapsed_ms = (time.perf_counter() - self._visual_round_started_at) * 1000.0
        if timeout:
            elapsed_ms = float(self._phase_duration_ms)

        self._visual_results.append(
            VisualRoundResult(
                round_index=self._visual_round_index,
                correct=correct,
                search_time_ms=max(0.0, elapsed_ms),
                miss_clicks=self._visual_round_miss_clicks,
                timeout=timeout,
            )
        )

        self._visual_round_index += 1
        progress = int((self._visual_round_index / max(1, len(self._visual_specs))) * 100)
        self.phase_progress.setValue(max(0, min(100, progress)))

        if timeout and not correct:
            self._play_feedback_sound("error")

        QTimer.singleShot(450, self._start_visual_round)

    def _finish_visual_phase(self) -> None:
        self._visual_summary = evaluate_visual(self._visual_results)
        self._set_feedback_banner(
            self.visual_status,
            "info",
            f"Hoàn thành - Chính xác: {self._visual_summary.accuracy:.1f}% | Bấm sai: {self._visual_summary.miss_click_count}"
        )
        self._play_feedback_sound("info")
        QTimer.singleShot(550, self._advance_step)

    def _start_break_phase(self, final: bool) -> None:
        self.stage_stack.setCurrentWidget(self.stage_break)

        self._phase_mode = "final_break" if final else "break"
        duration_s = int(self.cfg.final_breathing_break_s if final else self.cfg.micro_break_s)
        self._phase_duration_ms = max(1, duration_s * 1000)
        self._phase_started_at = time.perf_counter()

        self.phase_title.setText("Nhịp thở cuối" if final else "Nghỉ ngắn")
        self.round_label.setText("Ổn định nhịp thở")
        self.remaining_label.setText(f"{duration_s:04.1f}s")
        self.phase_progress.setValue(0)
        self.break_phase_label.setText("Hít vào")
        self.break_countdown_label.setText("Làm theo nhịp thở")

        self._phase_timer.start(33)

    @pyqtSlot()
    def _on_phase_tick(self) -> None:
        elapsed_ms = int((time.perf_counter() - self._phase_started_at) * 1000)
        remain_ms = max(0, self._phase_duration_ms - elapsed_ms)

        progress = int((elapsed_ms / max(1, self._phase_duration_ms)) * 100)
        self.phase_progress.setValue(max(0, min(100, progress)))
        self.remaining_label.setText(f"{remain_ms / 1000.0:04.1f}s")

        if self._phase_mode in self.ATTENTION_PHASES:
            self._tick_attention_phase(elapsed_ms)
            if elapsed_ms >= self._phase_duration_ms:
                self._finish_attention_phase()
            return

        if self._phase_mode in {"break", "final_break"}:
            self._tick_break_phase(elapsed_ms, remain_ms)
            if elapsed_ms >= self._phase_duration_ms:
                self._finish_break_phase()
            return

        if self._phase_mode == "sequence_input":
            if elapsed_ms >= self._phase_duration_ms:
                self._on_sequence_timeout()
            return

        if self._phase_mode == "visual":
            if elapsed_ms >= self._phase_duration_ms and not self._visual_round_resolved:
                self._set_feedback_banner(self.visual_status, "error", "Hết giờ")
                self._finish_visual_round(correct=False, timeout=True)

    def _tick_attention_phase(self, elapsed_ms: int) -> None:
        slot_ms = int(self.cfg.gonogo.stimulus_duration_ms + self.cfg.gonogo.inter_stimulus_ms)
        idx, _ = active_trial_at(
            elapsed_ms=elapsed_ms,
            trials=self._phase_trials,
            stimulus_duration_ms=int(self.cfg.gonogo.stimulus_duration_ms),
            trial_slot_ms=slot_ms,
        )

        if idx is None:
            if self._last_attention_trial_index is not None or self.stimulus_label.text():
                self.stimulus_label.setText("")
                self.stimulus_label.setStyleSheet("border: none;")
            if self.attention_feedback.text().strip():
                self._clear_feedback_banner(self.attention_feedback)
            self._last_attention_trial_index = None
            self._last_attention_is_target = None
            return

        trial = self._phase_trials[idx]
        if idx == self._last_attention_trial_index and trial.is_target == self._last_attention_is_target:
            return

        if self.attention_feedback.text().strip():
            self._clear_feedback_banner(self.attention_feedback)

        self._last_attention_trial_index = idx
        self._last_attention_is_target = trial.is_target
        payload = self._attention_trial_payloads.get(trial.index)
        if payload:
            color = str(payload.get("color", self.theme.text_primary))
            font_size = int(payload.get("font_size", 72))
            self.stimulus_label.setText(str(payload.get("text", "")))
            self.stimulus_label.setStyleSheet(
                f"color: {color}; border: none; font-size: {font_size}px; font-weight: 800;"
            )
            return

        color = self.theme.target_color if trial.is_target else self.theme.nogo_color
        self.stimulus_label.setText("●")
        self.stimulus_label.setStyleSheet(f"color: {color}; border: none;")

    def _finish_attention_phase(self) -> None:
        mode = self._phase_mode
        self._stop_all_timers(keep_sequence_token=True)

        results = evaluate_trials(self._phase_trials, self._phase_responses)
        baseline_rt = self._baseline_summary.average_reaction_ms if self._baseline_summary else None
        summary = summarize_gonogo(
            results,
            extra_commissions=self._phase_extra_commissions,
            baseline_avg_rt_ms=baseline_rt if mode in {"gonogo", "stroop", "flanker"} else None,
        )

        if mode == "baseline":
            self._baseline_summary = summary
            self._set_feedback_banner(self.attention_feedback, "info", "Đã ghi nhận baseline của phiên")
        else:
            if mode == "stroop":
                self._stroop_summary = summary
                phase_name = "Stroop màu"
            elif mode == "flanker":
                self._flanker_summary = summary
                phase_name = "Flanker"
            else:
                self._gonogo_summary = summary
                phase_name = "Go/No-Go"
            self._set_feedback_banner(
                self.attention_feedback,
                "info",
                f"Hoàn thành {phase_name} - Chính xác: {summary.accuracy:.1f}% | Ổn định chú ý: {summary.focus_stability:.1f}"
            )

        self._play_feedback_sound("info")

        QTimer.singleShot(500, self._advance_step)

    def _tick_break_phase(self, elapsed_ms: int, remain_ms: int) -> None:
        inhale = float(self.cfg.inhale_seconds)
        exhale = float(self.cfg.exhale_seconds)
        cycle = max(0.1, inhale + exhale)
        t = (elapsed_ms / 1000.0) % cycle

        if t < inhale:
            phase = t / max(inhale, 1e-6)
            self.break_phase_label.setText("Hít vào")
        else:
            phase = 1.0 - ((t - inhale) / max(exhale, 1e-6))
            self.break_phase_label.setText("Thở ra")

        self.breath_widget.set_phase(phase)
        self.break_countdown_label.setText(f"Còn lại: {remain_ms / 1000.0:0.1f}s")

    def _finish_break_phase(self) -> None:
        self._stop_all_timers(keep_sequence_token=True)
        QTimer.singleShot(400, self._advance_step)

    def _finish_recovery_session(self) -> None:
        self._stop_all_timers()

        self._final_summary = build_session_summary(
            baseline=self._baseline_summary,
            gonogo=self._gonogo_summary,
            sequence=self._sequence_summary,
            visual=self._visual_summary,
            additional_metrics={
                "Stroop Match": self._stroop_summary,
                "Flanker Arrows": self._flanker_summary,
            },
        )
        self._result_created_at = datetime.now().isoformat(timespec="seconds")
        self._render_results(self._final_summary)

        record = build_session_record(
            session_summary=self._final_summary,
            baseline_summary=self._baseline_summary,
            gonogo_summary=self._gonogo_summary,
            sequence_summary=self._sequence_summary,
            visual_summary=self._visual_summary,
        )
        record["selected_games"] = list(self._selected_games)
        record["selected_game_titles"] = [self.GAME_TITLES.get(game, game) for game in self._selected_games]
        if self._stroop_summary is not None:
            record["score_stroop"] = round(float(self._final_summary.game_scores.get("Stroop Match", 0.0)), 2)
        if self._flanker_summary is not None:
            record["score_flanker"] = round(float(self._final_summary.game_scores.get("Flanker Arrows", 0.0)), 2)
        self.storage.append(record)

        self._set_current_page(self.page_results)
        self._play_feedback_sound("info")

    def _render_results(self, summary: SessionSummary) -> None:
        probe_score = self._compute_attention_probe_score(summary) * 100.0
        self.result_probe_score.setText(f"{probe_score:.1f} / 100")
        self.result_attention_stability.setText(f"{summary.focus_stability:.1f} / 100")
        self.result_accuracy.setText(f"{summary.accuracy:.1f}%")
        self.result_avg_rt.setText(f"{summary.average_reaction_ms:.0f} ms")
        self.result_rt_var.setText(f"{summary.reaction_variability_ms:.0f} ms")
        self.result_omissions.setText(str(int(summary.omission_errors)))
        self.result_commissions.setText(str(int(summary.commission_errors)))
        self.result_best.setText(self._game_name_to_vn(summary.best_game))
        self.result_weakest.setText(self._game_name_to_vn(summary.weakest_game))

        scores = summary.game_scores or {}
        self.result_gonogo.setText(self._format_optional_score(scores.get("Go/No-Go")))
        self.result_stroop.setText(self._format_optional_score(scores.get("Stroop Match")))
        self.result_flanker.setText(self._format_optional_score(scores.get("Flanker Arrows")))
        self.result_sequence.setText(self._format_optional_score(scores.get("Sequence Memory")))
        self.result_visual.setText(self._format_optional_score(scores.get("Visual Search")))

        self.result_feedback.setText(summary.feedback)
        self._reset_ready_selection()

    def _format_optional_score(self, value: float | None) -> str:
        if value is None:
            return "-"
        return f"{float(value):.1f}"

    def _compute_attention_probe_score(self, summary: SessionSummary) -> float:
        accuracy_norm = max(0.0, min(1.0, float(summary.accuracy) / 100.0))
        stability_norm = max(0.0, min(1.0, float(summary.focus_stability) / 100.0))
        rt_ms = float(summary.average_reaction_ms) if summary.average_reaction_ms > 0 else 400.0
        rt_norm = max(0.0, min(1.0, 1.0 - (rt_ms - 200.0) / 600.0))
        return (accuracy_norm * 0.45) + (stability_norm * 0.35) + (rt_norm * 0.20)

    def _load_history_table(self) -> None:
        rows = list(reversed(self.storage.load()))
        self.history_table.setRowCount(0)

        for row_idx, item in enumerate(rows):
            self.history_table.insertRow(row_idx)

            gonogo_rt = item.get("gonogo_rt_ms", item.get("session_rt_ms", 0.0))

            values = [
                str(item.get("timestamp", "")),
                f"{float(item.get('baseline_rt_ms', 0.0)):.1f}",
                f"{float(gonogo_rt):.1f}",
                f"{float(item.get('accuracy', 0.0)):.1f}%",
                f"{float(item.get('focus_stability', 0.0)):.1f}",
                self._game_name_to_vn(str(item.get("best_game", "-"))),
                self._game_name_to_vn(str(item.get("weakest_game", "-"))),
                f"{float(item.get('score_gonogo', 0.0)):.1f}",
                f"{float(item.get('score_sequence', 0.0)):.1f}",
                f"{float(item.get('score_visual', 0.0)):.1f}",
                self._comparison_to_vn(str(item.get("comparison", ""))),
            ]

            for col_idx, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.history_table.setItem(row_idx, col_idx, table_item)

        self.history_status.setText(f"Đã tải {len(rows)} phiên")

    def _export_history_csv(self) -> None:
        suggested = str(self.cfg.history_path.with_suffix(".csv"))
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất lịch sử Attention Probe",
            suggested,
            "Tệp CSV (*.csv)",
        )

        if not target:
            return

        try:
            out_path = self.storage.export_csv(Path(target))
            self.history_status.setText(f"Đã xuất: {out_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Xuất thất bại", str(exc))

    def _register_attention_response(self) -> None:
        elapsed_ms = int((time.perf_counter() - self._phase_started_at) * 1000)
        slot_ms = int(self.cfg.gonogo.stimulus_duration_ms + self.cfg.gonogo.inter_stimulus_ms)

        idx, rt_ms = active_trial_at(
            elapsed_ms=elapsed_ms,
            trials=self._phase_trials,
            stimulus_duration_ms=int(self.cfg.gonogo.stimulus_duration_ms),
            trial_slot_ms=slot_ms,
        )

        if idx is None:
            self._phase_extra_commissions += 1
            self._set_feedback_banner(self.attention_feedback, "error", "Bấm quá sớm")
            self._play_feedback_sound("error")
            return

        if idx in self._phase_responses:
            return

        self._phase_responses[idx] = int(rt_ms or 0)
        trial = self._phase_trials[idx]

        if trial.is_target:
            self._set_feedback_banner(self.attention_feedback, "success", "Đúng")
            self._play_feedback_sound("success")
        else:
            self._set_feedback_banner(self.attention_feedback, "error", "Sai")
            self._play_feedback_sound("error")

    def _abort_session(self) -> None:
        self._stop_all_timers()
        self._show_menu()

    def _stop_all_timers(self, keep_sequence_token: bool = False) -> None:
        self._phase_timer.stop()
        self._sequence_timeout.stop()
        if not keep_sequence_token:
            self._sequence_token += 1

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key.Key_Escape and self._phase_mode in {
            "baseline",
            "gonogo",
            "stroop",
            "flanker",
            "sequence_show",
            "sequence_input",
            "visual",
            "break",
            "final_break",
        }:
            self._abort_session()
            event.accept()
            return

        if key == Qt.Key.Key_Space and self._phase_mode in self.ATTENTION_PHASES:
            self._register_attention_response()
            event.accept()
            return

        if self._phase_mode == "sequence_input":
            if key in {Qt.Key.Key_Backspace, Qt.Key.Key_Delete}:
                self._on_sequence_backspace()
                event.accept()
                return

            text = event.text().strip().upper()
            if text:
                for symbol in self.cfg.sequence.symbols:
                    if text == symbol[:1].upper():
                        self._on_sequence_symbol(symbol)
                        event.accept()
                        return

        super().keyPressEvent(event)

    def closeEvent(self, event):
        self._stop_all_timers()
        super().closeEvent(event)

    def _rebuild_sequence_symbol_buttons(self) -> None:
        self._clear_layout(self.sequence_buttons_row)
        self.sequence_symbol_buttons = []

        style = self._interactive_button_style().replace("border-radius: 8px;", "border-radius: 10px;")
        for symbol in self.cfg.sequence.symbols:
            btn = QPushButton(symbol)
            btn.setEnabled(False)
            btn.setStyleSheet(style + "QPushButton { min-width: 66px; padding: 8px; }")
            btn.clicked.connect(lambda _checked=False, s=symbol: self._on_sequence_symbol(s))
            self.sequence_buttons_row.addWidget(btn)
            self.sequence_symbol_buttons.append(btn)

        backspace_btn = QPushButton("Xóa")
        backspace_btn.setEnabled(False)
        backspace_btn.setStyleSheet(style + "QPushButton { min-width: 74px; padding: 8px; }")
        backspace_btn.clicked.connect(self._on_sequence_backspace)
        self.sequence_buttons_row.addWidget(backspace_btn)
        self.sequence_symbol_buttons.append(backspace_btn)

    def _game_position_label(self, game_id: str) -> str:
        total = max(1, len(self._selected_games))
        try:
            index = self._selected_games.index(game_id) + 1
        except ValueError:
            index = 1
        return f"Bài {index}/{total}"

    def get_attention_probe_result(self) -> dict:
        """
        Return structured Attention Probe metrics for MainWindow/session analytics.

        The composite score remains 0-1 for compatibility with recovery
        validation, while UI-facing per-game scores remain 0-100.
        """
        if self._final_summary is None:
            return {"probe_completed": False}

        s = self._final_summary
        game_attention_score = self._compute_attention_probe_score(s)
        accuracy_norm = max(0.0, min(1.0, float(s.accuracy) / 100.0))
        rt_ms = float(s.average_reaction_ms) if s.average_reaction_ms > 0 else 0.0

        visual_misses = int(self._visual_summary.miss_click_count if self._visual_summary else 0)
        visual_timeouts = sum(1 for result in self._visual_results if result.timeout)
        sequence_mistakes = sum(max(0, int(result.mistakes)) for result in self._sequence_results)
        miss_count = int(s.omission_errors) + int(visual_timeouts)
        wrong_count = int(s.commission_errors) + visual_misses + sequence_mistakes

        result = {
            "probe_completed": True,
            "game_attention_score": round(game_attention_score, 4),
            "attention_stability": round(float(s.focus_stability), 1),
            "accuracy": round(accuracy_norm, 4),
            "avg_reaction_time_ms": round(rt_ms, 1),
            "reaction_variability_ms": round(float(s.reaction_variability_ms), 1),
            "commission_errors": int(s.commission_errors),
            "omission_errors": int(s.omission_errors),
            "miss_count": int(miss_count),
            "wrong_count": int(wrong_count),
            "best_game": str(s.best_game or ""),
            "weakest_game": str(s.weakest_game or ""),
            "game_scores": dict(s.game_scores or {}),
            "selected_games": list(self._selected_games),
            "self_report_ready": self._self_report_ready,
            "created_at": self._result_created_at or datetime.now().isoformat(timespec="seconds"),
            # Compatibility keys used by the current MainWindow recovery logic.
            "avg_reaction_ms": round(rt_ms, 1),
            "focus_stability": round(float(s.focus_stability), 1),
            "completed": True,
        }
        return result

    def _comparison_to_vn(self, value: str) -> str:
        mapping = {
            "Better": "Tốt hơn",
            "Similar": "Tương đương",
            "Worse": "Kém hơn",
        }
        return mapping.get(value, value)

    def _game_name_to_vn(self, value: str) -> str:
        mapping = {
            "Go/No-Go": "Go/No-Go",
            "Stroop Match": "Stroop màu",
            "Flanker Arrows": "Mũi tên Flanker",
            "Sequence Memory": "Ghi nhớ chuỗi",
            "Visual Search": "Tìm kiếm thị giác",
            "-": "-",
        }
        return mapping.get(value, value)

    @staticmethod
    def _clear_layout(layout: QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                continue

            child_layout = item.layout()
            if child_layout is not None:
                FocusResetDialog._clear_layout(child_layout)
