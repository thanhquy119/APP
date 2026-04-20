"""Giao diện Focus Reset (phiên bản tiếng Việt)."""

from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
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
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
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


class BreathingCircleWidget(QWidget):
    """Vòng tròn mô phỏng nhịp thở cho các pha nghỉ ngắn."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scale = 0.55
        self.setMinimumSize(220, 220)

    def set_scale(self, scale: float) -> None:
        self._scale = max(0.3, min(1.0, float(scale)))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        cx = width // 2
        cy = height // 2

        max_radius = min(width, height) * 0.38
        radius = int(max_radius * self._scale)
        glow_radius = radius + 20

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(56, 189, 248, 40))
        painter.drawEllipse(cx - glow_radius, cy - glow_radius, glow_radius * 2, glow_radius * 2)

        painter.setBrush(QColor(56, 189, 248, 180))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        pen = QPen(QColor(125, 211, 252, 220))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)


class FocusResetDialog(QDialog):
    """Hộp thoại phục hồi tập trung với 3 mini-game và lịch sử."""

    GAME_ORDER = ["gonogo", "sequence", "visual"]
    GAME_TITLES = {
        "gonogo": "Phản xạ Go/No-Go",
        "sequence": "Ghi nhớ chuỗi",
        "visual": "Tìm kiếm thị giác",
    }

    def __init__(
        self,
        parent=None,
        config: FocusResetConfig | None = None,
        storage: SessionStorage | None = None,
        theme_mode: str = "dark",
        app_sound_enabled: bool = True,
        app_volume: int = 70,
    ):
        super().__init__(parent)

        self.cfg = config or load_focus_reset_config()
        self.theme_mode = "light" if str(theme_mode or "dark").strip().lower() == "light" else "dark"
        self.theme = Theme.for_mode(self.theme_mode)
        self.storage = storage or SessionStorage(self.cfg.history_path)
        self._app_sound_enabled = bool(app_sound_enabled)
        self._app_volume = max(0, min(100, int(app_volume)))

        self._phase_timer = QTimer(self)
        self._phase_timer.timeout.connect(self._on_phase_tick)

        self._sequence_timeout = QTimer(self)
        self._sequence_timeout.setSingleShot(True)
        self._sequence_timeout.timeout.connect(self._on_sequence_timeout)

        self._sequence_token = 0

        self._reset_runtime_state()

        self.setWindowTitle("Phục hồi tập trung")
        self.setModal(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet(self._build_stylesheet())

        self.stack = QStackedWidget(self)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 18)
        root.addWidget(self.stack)

        self.page_menu = self._build_menu_page()
        self.page_instructions = self._build_instructions_page()
        self.page_select = self._build_game_select_page()
        self.page_settings = self._build_settings_page()
        self.page_history = self._build_history_page()
        self.page_session = self._build_session_page()
        self.page_results = self._build_results_page()

        self.stack.addWidget(self.page_menu)
        self.stack.addWidget(self.page_instructions)
        self.stack.addWidget(self.page_select)
        self.stack.addWidget(self.page_settings)
        self.stack.addWidget(self.page_history)
        self.stack.addWidget(self.page_session)
        self.stack.addWidget(self.page_results)
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
            self.resize(980, 740)
            return

        available = screen.availableGeometry()
        margin = 24

        max_w = max(700, available.width() - (margin * 2))
        max_h = max(500, available.height() - (margin * 2))

        desired_w = 980
        desired_h = 740

        target_w = min(desired_w, max_w)
        target_h = min(desired_h, max_h)

        min_w = min(760, max_w)
        min_h = min(520, max_h)
        self.setMinimumSize(min_w, min_h)
        self.resize(target_w, target_h)

        x = available.x() + (available.width() - target_w) // 2
        y = available.y() + (available.height() - target_h) // 2
        self.move(max(available.x(), x), max(available.y(), y))

    def showEvent(self, event):
        super().showEvent(event)
        self._clamp_to_screen()

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
        current = self.stack.currentWidget()
        if current is None:
            return

        screen = self._resolve_screen()
        max_h = 900
        if screen is not None:
            max_h = max(500, screen.availableGeometry().height() - 12)

        margins = self.layout().contentsMargins()
        frame_padding = margins.top() + margins.bottom() + 40
        desired_h = current.sizeHint().height() + frame_padding

        compact_pages = {
            self.page_menu,
            self.page_select,
            self.page_instructions,
        }
        if current in compact_pages:
            desired_h = min(desired_h, 640)

        target_h = max(self.minimumHeight(), min(desired_h, max_h))
        if abs(self.height() - target_h) > 6:
            self.resize(self.width(), target_h)

        self._clamp_to_screen()

    def _build_stylesheet(self) -> str:
        return f"""
            QDialog {{
                background-color: {self.theme.background};
                color: {self.theme.text_primary};
                font-family: 'Bahnschrift', 'Segoe UI', sans-serif;
                font-size: 14px;
            }}
            QFrame#panel {{
                background-color: {self.theme.panel};
                border: 1px solid {self.theme.border};
                border-radius: 16px;
            }}
            QFrame#hero {{
                background-color: {self.theme.hero_bg};
                border: 1px solid {self.theme.border};
                border-radius: 16px;
            }}
            QLabel#title {{
                font-size: 34px;
                font-weight: 800;
                color: {self.theme.text_primary};
            }}
            QLabel#subtitle {{
                font-size: 15px;
                color: {self.theme.text_muted};
            }}
            QLabel#muted {{
                color: {self.theme.text_muted};
            }}
            QLabel#value {{
                font-size: 16px;
                font-weight: 700;
            }}
            QPushButton {{
                background-color: {self.theme.panel_alt};
                color: {self.theme.text_primary};
                border: 1px solid {self.theme.border};
                border-radius: 12px;
                padding: 10px 16px;
                font-weight: 600;
                min-height: 38px;
            }}
            QPushButton:hover {{
                background-color: {self.theme.interactive_hover};
                border-color: {self.theme.accent_border};
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
                border-radius: 12px;
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
                border-radius: 8px;
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
                border-radius: 12px;
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
        """

    def _interactive_button_style(self) -> str:
        return (
            "QPushButton {"
            f"background-color: {self.theme.interactive_bg};"
            f"border: 1px solid {self.theme.interactive_border};"
            "border-radius: 8px;"
            "font-size: 18px;"
            "font-weight: 700;"
            "padding: 4px;"
            "min-height: 30px;"
            "}"
            f"QPushButton:hover {{ background-color: {self.theme.interactive_hover}; }}"
        )

    def _play_feedback_sound(self, kind: str) -> None:
        """Play subtle sound cues when enabled in both app and game settings."""
        if not self._app_sound_enabled or not bool(self.cfg.sound_enabled):
            return
        if self._app_volume <= 0:
            return

        kind_name = str(kind or "info").strip().lower()
        if kind_name not in {"success", "error", "info"}:
            kind_name = "info"

        try:
            import winsound

            tone_map = {
                "success": 860,
                "error": 470,
                "info": 700,
            }
            duration = max(25, min(95, int(25 + (self._app_volume * 0.7))))
            winsound.Beep(int(tone_map[kind_name]), int(duration))
            return
        except Exception:
            pass

        QApplication.beep()

    def _build_menu_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 12, 18, 12)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 20, 24, 20)
        panel_layout.setSpacing(12)

        title = QLabel("Phục hồi tập trung")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(title)

        subtitle = QLabel("Bộ mini-game ngắn giúp bạn lấy lại tập trung")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(subtitle)

        start_btn = QPushButton("Bắt đầu phiên phục hồi")
        start_btn.setObjectName("primary")
        start_btn.clicked.connect(self._start_recovery_session)
        panel_layout.addWidget(start_btn)

        select_btn = QPushButton("Chọn mini-game")
        select_btn.clicked.connect(self._show_game_select)
        panel_layout.addWidget(select_btn)

        instructions_btn = QPushButton("Hướng dẫn")
        instructions_btn.clicked.connect(self._show_instructions)
        panel_layout.addWidget(instructions_btn)

        settings_btn = QPushButton("Cài đặt mini-game")
        settings_btn.clicked.connect(self._show_settings)
        panel_layout.addWidget(settings_btn)

        history_btn = QPushButton("Lịch sử")
        history_btn.clicked.connect(self._show_history)
        panel_layout.addWidget(history_btn)

        exit_btn = QPushButton("Đóng")
        exit_btn.clicked.connect(self.close)
        panel_layout.addWidget(exit_btn)

        layout.addWidget(panel)
        return page

    def _build_instructions_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 12, 18, 12)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(22, 20, 22, 20)
        panel_layout.setSpacing(12)

        title = QLabel("Hướng dẫn")
        title.setStyleSheet("font-size: 22px; font-weight: 800;")
        panel_layout.addWidget(title)

        body = QLabel(
            "1. Mốc ban đầu: đo nhanh để lấy baseline.\n"
            "2. Go/No-Go: nhấn Space khi thấy mục tiêu xanh.\n"
            "3. Ghi nhớ chuỗi: nhớ ký tự và nhập lại đúng thứ tự.\n"
            "4. Tìm kiếm thị giác: tìm ký tự mục tiêu trong lưới.\n"
            "5. Giữa các game có nghỉ nhịp thở ngắn + 1 lần nghỉ cuối.\n"
            "6. Kết quả: độ ổn định tập trung, so sánh với baseline, game mạnh/yếu nhất."
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
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(12)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(20, 18, 20, 18)
        panel_layout.setSpacing(12)

        title = QLabel("Chọn mini-game")
        title.setStyleSheet("font-size: 22px; font-weight: 800;")
        panel_layout.addWidget(title)

        subtitle = QLabel("Bật các game bạn muốn cho lượt phục hồi này. Cần ít nhất 1 game.")
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        panel_layout.addWidget(subtitle)

        self.chk_gonogo = QCheckBox("Bật Go/No-Go")
        self.chk_gonogo.setChecked(True)
        panel_layout.addWidget(
            self._build_game_card(
                "Phản xạ Go/No-Go",
                "Rèn phản xạ và khả năng ức chế hành động sai.",
                "Chỉ số: thời gian phản ứng, lỗi commission/omission.",
                self.chk_gonogo,
            )
        )

        self.chk_sequence = QCheckBox("Bật game ghi nhớ chuỗi")
        self.chk_sequence.setChecked(True)
        panel_layout.addWidget(
            self._build_game_card(
                "Ghi nhớ chuỗi",
                "Luyện trí nhớ ngắn hạn và độ chính xác theo thứ tự.",
                "Chỉ số: độ chính xác, độ dài tối đa, độ ổn định phản hồi.",
                self.chk_sequence,
            )
        )

        self.chk_visual = QCheckBox("Bật game tìm kiếm thị giác")
        self.chk_visual.setChecked(True)
        panel_layout.addWidget(
            self._build_game_card(
                "Tìm kiếm thị giác",
                "Luyện tập trung thị giác trong lưới ký tự tăng dần độ khó.",
                "Chỉ số: độ chính xác, tốc độ hoàn thành, số lần bấm sai.",
                self.chk_visual,
            )
        )

        self.select_status = QLabel("")
        self.select_status.setObjectName("muted")
        panel_layout.addWidget(self.select_status)

        row = QHBoxLayout()

        start_btn = QPushButton("Chơi ngay game đã chọn")
        start_btn.setObjectName("primary")
        start_btn.clicked.connect(self._start_selected_games_only)
        row.addWidget(start_btn)

        row.addStretch()

        back_btn = QPushButton("Quay lại")
        back_btn.clicked.connect(self._show_menu)
        row.addWidget(back_btn)

        panel_layout.addLayout(row)
        layout.addWidget(panel)
        return page

    def _build_game_card(self, title: str, desc: str, metric_line: str, checkbox: QCheckBox) -> QFrame:
        card = QFrame()
        card.setObjectName("panel")

        row = QHBoxLayout(card)
        row.setContentsMargins(14, 12, 14, 12)
        row.setSpacing(10)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 18px; font-weight: 700;")
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
                "Ổn định",
                "Trò chơi tốt nhất",
                "Trò chơi yếu nhất",
                "Điểm Go/No-Go",
                "Điểm Chuỗi",
                "Điểm Thị giác",
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

        abort_btn = QPushButton("Dừng phiên")
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

        self.attention_hint = QLabel("Nhấn Space khi thấy mục tiêu xanh")
        self.attention_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.attention_hint.setObjectName("muted")
        layout.addWidget(self.attention_hint)

        self.attention_feedback = QLabel("")
        self.attention_feedback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.attention_feedback.setObjectName("muted")
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
        layout.addWidget(self.sequence_input_label)

        self.sequence_feedback = QLabel("")
        self.sequence_feedback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sequence_feedback.setObjectName("muted")
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
        self.visual_status.setObjectName("muted")
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
        layout.setSpacing(12)
        layout.setContentsMargins(18, 12, 18, 12)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(20, 18, 20, 18)
        panel_layout.setSpacing(12)

        title = QLabel("Kết quả phiên")
        title.setStyleSheet("font-size: 24px; font-weight: 800;")
        panel_layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)

        self.result_focus = QLabel("-")
        self.result_accuracy = QLabel("-")
        self.result_comparison = QLabel("-")
        self.result_best = QLabel("-")
        self.result_weakest = QLabel("-")
        self.result_gonogo = QLabel("-")
        self.result_sequence = QLabel("-")
        self.result_visual = QLabel("-")

        self._add_result_row(grid, 0, "Độ ổn định tập trung", self.result_focus)
        self._add_result_row(grid, 1, "Độ chính xác tổng", self.result_accuracy)
        self._add_result_row(grid, 2, "So với baseline", self.result_comparison)
        self._add_result_row(grid, 3, "Trò chơi tốt nhất", self.result_best)
        self._add_result_row(grid, 4, "Trò chơi cần cải thiện", self.result_weakest)
        self._add_result_row(grid, 5, "Điểm Go/No-Go", self.result_gonogo)
        self._add_result_row(grid, 6, "Điểm Ghi nhớ chuỗi", self.result_sequence)
        self._add_result_row(grid, 7, "Điểm Tìm kiếm thị giác", self.result_visual)

        panel_layout.addLayout(grid)

        self.result_feedback = QLabel("")
        self.result_feedback.setWordWrap(True)
        self.result_feedback.setStyleSheet(
            f"background-color: {self.theme.panel_soft}; border: 1px solid {self.theme.border}; "
            f"border-radius: 12px; color: {self.theme.text_primary}; padding: 12px; font-size: 15px;"
        )
        panel_layout.addWidget(self.result_feedback)

        row = QHBoxLayout()

        replay_btn = QPushButton("Bắt đầu phiên mới")
        replay_btn.setObjectName("primary")
        replay_btn.clicked.connect(self._start_recovery_session)
        row.addWidget(replay_btn)

        row.addStretch()

        history_btn = QPushButton("Lịch sử")
        history_btn.clicked.connect(self._show_history)
        row.addWidget(history_btn)

        menu_btn = QPushButton("Màn hình chính")
        menu_btn.clicked.connect(self._show_menu)
        row.addWidget(menu_btn)

        panel_layout.addLayout(row)
        layout.addWidget(panel)
        return page

    def _add_result_row(self, grid: QGridLayout, row: int, label: str, value_widget: QLabel) -> None:
        left = QLabel(label)
        left.setObjectName("muted")
        grid.addWidget(left, row, 0)

        value_widget.setObjectName("value")
        value_widget.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(value_widget, row, 1)

    def _show_menu(self) -> None:
        self._stop_all_timers()
        self.stack.setCurrentWidget(self.page_menu)

    def _show_instructions(self) -> None:
        self.stack.setCurrentWidget(self.page_instructions)

    def _show_game_select(self) -> None:
        self.select_status.setText("")
        self.stack.setCurrentWidget(self.page_select)

    def _show_settings(self) -> None:
        self._load_settings_to_controls()
        self.settings_status.setText("")
        self.stack.setCurrentWidget(self.page_settings)

    def _show_history(self) -> None:
        self._load_history_table()
        self.stack.setCurrentWidget(self.page_history)

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
        if self.chk_gonogo.isChecked():
            selected.append("gonogo")
        if self.chk_sequence.isChecked():
            selected.append("sequence")
        if self.chk_visual.isChecked():
            selected.append("visual")
        return selected

    def _start_recovery_session(self) -> None:
        self._start_session(include_baseline=True)

    def _start_selected_games_only(self) -> None:
        self._start_session(include_baseline=False)

    def _start_session(self, include_baseline: bool) -> None:
        selected = self._collect_selected_games()
        if not selected:
            self.select_status.setText("Vui lòng chọn ít nhất một mini-game.")
            self.stack.setCurrentWidget(self.page_select)
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

        self.stack.setCurrentWidget(self.page_session)
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
        self._sequence_summary: SequenceSummary | None = None
        self._visual_summary: VisualSummary | None = None
        self._final_summary: SessionSummary | None = None

        self._sequence_round_sequences: list[list[str]] = []
        self._sequence_round_lengths: list[int] = []
        self._sequence_round_index = 0
        self._sequence_expected: list[str] = []
        self._sequence_input: list[str] = []
        self._sequence_mistakes = 0
        self._sequence_input_started_at = 0.0
        self._sequence_results: list[SequenceRoundResult] = []

        self._visual_specs: list[VisualRoundSpec] = []
        self._visual_round_index = 0
        self._visual_round_started_at = 0.0
        self._visual_round_miss_clicks = 0
        self._visual_target_index = -1
        self._visual_round_resolved = False
        self._visual_results: list[VisualRoundResult] = []

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
            title="Mốc ban đầu",
            subtitle="Đo mốc ban đầu",
            hint="Nhấn Space khi thấy mục tiêu xanh",
        )

    def _start_gonogo_phase(self) -> None:
        self._start_attention_phase(
            mode="gonogo",
            duration_s=int(self.cfg.gonogo.round_duration_s),
            title="Phản xạ Go/No-Go",
            subtitle=self._game_position_label("gonogo"),
            hint="Nhấn nhanh và đúng, tránh bấm vào mục đỏ",
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

        self.phase_title.setText(title)
        self.round_label.setText(subtitle)
        self.remaining_label.setText(f"{duration_s:04.1f}s")
        self.phase_progress.setValue(0)

        self.stage_stack.setCurrentWidget(self.stage_attention)
        self.attention_hint.setText(hint)
        self.attention_feedback.setText("")
        self.stimulus_label.setText("")

        self._phase_timer.start(16)
        self.setFocus()

    def _start_sequence_phase(self) -> None:
        self.phase_title.setText("Ghi nhớ chuỗi")
        self.round_label.setText(self._game_position_label("sequence"))
        self.phase_progress.setValue(0)
        self.remaining_label.setText(f"{self.cfg.sequence.input_timeout_s:04.1f}s")

        self.stage_stack.setCurrentWidget(self.stage_sequence)
        self.sequence_feedback.setText("")

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
        self._sequence_mistakes = 0

        self.sequence_mode_label.setText(
            f"Ghi nhớ chuỗi {self._sequence_round_index + 1}/{len(self._sequence_round_sequences)}"
        )
        self.sequence_input_label.setText(" ")
        self.sequence_feedback.setText("")

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

    def _enter_sequence_input(self, token: int) -> None:
        if token != self._sequence_token or self._phase_mode != "sequence_show":
            return

        self._phase_mode = "sequence_input"
        self._phase_duration_ms = int(self.cfg.sequence.input_timeout_s * 1000)
        self._phase_started_at = time.perf_counter()

        self.sequence_mode_label.setText("Nhập lại chuỗi")
        self.sequence_show_label.setText(" ".join(self._sequence_expected))
        self.sequence_input_label.setText("Nhập: ")

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
        idx = len(self._sequence_input) - 1
        if idx < len(self._sequence_expected) and symbol != self._sequence_expected[idx]:
            self._sequence_mistakes += 1

        rendered = " ".join(self._sequence_input)
        self.sequence_input_label.setText(f"Nhập: {rendered}")

        if len(self._sequence_input) >= len(self._sequence_expected):
            self._finalize_sequence_round(timeout=False)

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

        correct = (self._sequence_input == self._sequence_expected) and not timeout and self._sequence_mistakes == 0

        self._sequence_results.append(
            SequenceRoundResult(
                round_index=self._sequence_round_index,
                sequence_length=len(self._sequence_expected),
                correct=correct,
                response_time_ms=max(0.0, elapsed_ms),
                mistakes=self._sequence_mistakes + (1 if timeout else 0),
            )
        )

        if correct:
            self.sequence_feedback.setStyleSheet(f"color: {self.theme.success_text};")
            self.sequence_feedback.setText("Đúng")
            self._play_feedback_sound("success")
        elif timeout:
            self.sequence_feedback.setStyleSheet(f"color: {self.theme.error_text};")
            self.sequence_feedback.setText("Hết giờ")
            self._play_feedback_sound("error")
        else:
            self.sequence_feedback.setStyleSheet(f"color: {self.theme.error_text};")
            self.sequence_feedback.setText("Sai thứ tự")
            self._play_feedback_sound("error")

        self._sequence_round_index += 1
        progress = int((self._sequence_round_index / max(1, len(self._sequence_round_sequences))) * 100)
        self.phase_progress.setValue(max(0, min(100, progress)))
        self.remaining_label.setText("00.0s")

        QTimer.singleShot(500, self._start_sequence_round)

    def _finish_sequence_phase(self) -> None:
        self._sequence_summary = evaluate_sequence(self._sequence_results)
        self.sequence_feedback.setStyleSheet(f"color: {self.theme.info_text};")
        self.sequence_feedback.setText(
            f"Hoàn thành - Chính xác: {self._sequence_summary.accuracy:.1f}% | Độ dài tối đa: {self._sequence_summary.max_sequence_length}"
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
        self.visual_status.setText("Bỏ qua ký tự gây nhiễu và bấm nhanh vào mục tiêu.")

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
            self.visual_status.setStyleSheet(f"color: {self.theme.success_text};")
            self.visual_status.setText("Đã tìm thấy mục tiêu")
            self._play_feedback_sound("success")
            self._finish_visual_round(correct=True, timeout=False)
            return

        self._visual_round_miss_clicks += 1
        self.visual_status.setStyleSheet(f"color: {self.theme.error_text};")
        self.visual_status.setText(f"Số lần bấm sai: {self._visual_round_miss_clicks}")

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
        self.visual_status.setStyleSheet(f"color: {self.theme.info_text};")
        self.visual_status.setText(
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
        self.round_label.setText("Phục hồi")
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

        if self._phase_mode in {"baseline", "gonogo"}:
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
                self.visual_status.setStyleSheet(f"color: {self.theme.error_text};")
                self.visual_status.setText("Hết giờ")
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
            self.stimulus_label.setText("")
            return

        trial = self._phase_trials[idx]
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
            baseline_avg_rt_ms=baseline_rt if mode == "gonogo" else None,
        )

        if mode == "baseline":
            self._baseline_summary = summary
            self.attention_feedback.setStyleSheet(f"color: {self.theme.info_text};")
            self.attention_feedback.setText("Đã xong baseline")
        else:
            self._gonogo_summary = summary
            self.attention_feedback.setStyleSheet(f"color: {self.theme.info_text};")
            self.attention_feedback.setText(
                f"Hoàn thành Go/No-Go - Chính xác: {summary.accuracy:.1f}% | Ổn định: {summary.focus_stability:.1f}"
            )

        self._play_feedback_sound("info")

        QTimer.singleShot(500, self._advance_step)

    def _tick_break_phase(self, elapsed_ms: int, remain_ms: int) -> None:
        inhale = float(self.cfg.inhale_seconds)
        exhale = float(self.cfg.exhale_seconds)
        cycle = max(0.1, inhale + exhale)
        t = (elapsed_ms / 1000.0) % cycle

        if t < inhale:
            ratio = t / max(inhale, 1e-6)
            self.break_phase_label.setText("Hít vào")
            scale = 0.45 + 0.45 * ratio
        else:
            ratio = (t - inhale) / max(exhale, 1e-6)
            self.break_phase_label.setText("Thở ra")
            scale = 0.90 - 0.45 * ratio

        self.breath_widget.set_scale(scale)
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
        )
        self._render_results(self._final_summary)

        record = build_session_record(
            session_summary=self._final_summary,
            baseline_summary=self._baseline_summary,
            gonogo_summary=self._gonogo_summary,
            sequence_summary=self._sequence_summary,
            visual_summary=self._visual_summary,
        )
        self.storage.append(record)

        self.stack.setCurrentWidget(self.page_results)
        self._play_feedback_sound("info")

    def _render_results(self, summary: SessionSummary) -> None:
        self.result_focus.setText(f"{summary.focus_stability:.1f} / 100")
        self.result_accuracy.setText(f"{summary.accuracy:.1f}%")
        self.result_comparison.setText(self._comparison_to_vn(summary.comparison))
        self.result_best.setText(self._game_name_to_vn(summary.best_game))
        self.result_weakest.setText(self._game_name_to_vn(summary.weakest_game))

        scores = summary.game_scores or {}
        self.result_gonogo.setText(f"{float(scores.get('Go/No-Go', 0.0)):.1f}")
        self.result_sequence.setText(f"{float(scores.get('Sequence Memory', 0.0)):.1f}")
        self.result_visual.setText(f"{float(scores.get('Visual Search', 0.0)):.1f}")

        self.result_feedback.setText(summary.feedback)

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
            "Xuất lịch sử phục hồi",
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
            self.attention_feedback.setStyleSheet(f"color: {self.theme.error_text};")
            self.attention_feedback.setText("Bấm quá sớm / chưa có mục tiêu")
            self._play_feedback_sound("error")
            return

        if idx in self._phase_responses:
            return

        self._phase_responses[idx] = int(rt_ms or 0)
        trial = self._phase_trials[idx]

        if trial.is_target:
            self.attention_feedback.setStyleSheet(f"color: {self.theme.success_text};")
            self.attention_feedback.setText(f"Trúng: {int(rt_ms or 0)} ms")
            self._play_feedback_sound("success")
        else:
            self.attention_feedback.setStyleSheet(f"color: {self.theme.error_text};")
            self.attention_feedback.setText("Lỗi bấm sai (No-Go)")
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
            "sequence_show",
            "sequence_input",
            "visual",
            "break",
            "final_break",
        }:
            self._abort_session()
            event.accept()
            return

        if key == Qt.Key.Key_Space and self._phase_mode in {"baseline", "gonogo"}:
            self._register_attention_response()
            event.accept()
            return

        if self._phase_mode == "sequence_input":
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

    def _game_position_label(self, game_id: str) -> str:
        total = max(1, len(self._selected_games))
        try:
            index = self._selected_games.index(game_id) + 1
        except ValueError:
            index = 1
        return f"Màn {index}/{total}"

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
