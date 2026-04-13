"""
Mini Games - Research-backed break activities for FocusGuardian.

This module offers short activities designed to help users recover attention
before returning to work/study:
- Paced breathing (4-2-6) to downshift stress arousal.
- Visual reset (20-20-20 inspired) to reduce eye strain.
- Stroop color task to reactivate executive attention control.
"""

from __future__ import annotations

import random
import time
import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QFrame,
    QProgressBar,
    QDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QGuiApplication, QPainter, QColor, QBrush

logger = logging.getLogger(__name__)


ACCENT_COLOR = "#38bdf8"
ACCENT_COLOR_HOVER = "#0ea5e9"
SURFACE_BG = "#070f1d"
SURFACE_PANEL = "#0f172a"
SURFACE_ELEVATED = "#111c2f"
BORDER_COLOR = "#243244"
TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED = "#94a3b8"


def _activity_panel_style() -> str:
    return f"""
        QFrame#activityPanel {{
            background-color: {SURFACE_PANEL};
            border: 1px solid {BORDER_COLOR};
            border-radius: 16px;
        }}
        QLabel {{
            color: {TEXT_PRIMARY};
            background: transparent;
            border: none;
        }}
        QLabel#hint {{
            color: {TEXT_MUTED};
        }}
    """


def _progress_style(chunk_color: str = ACCENT_COLOR) -> str:
    return f"""
        QProgressBar {{
            background-color: {SURFACE_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
            text-align: center;
            color: #020617;
            font-weight: 700;
            min-height: 18px;
        }}
        QProgressBar::chunk {{
            background-color: {chunk_color};
            border-radius: 7px;
        }}
    """


def _primary_button_style() -> str:
    return f"""
        QPushButton {{
            background-color: {ACCENT_COLOR};
            color: #062033;
            border: 1px solid #7dd3fc;
            border-radius: 12px;
            padding: 10px 18px;
            font-weight: 700;
        }}
        QPushButton:hover {{
            background-color: {ACCENT_COLOR_HOVER};
            border-color: #bae6fd;
        }}
        QPushButton:pressed {{
            background-color: #0284c7;
        }}
        QPushButton:disabled {{
            background-color: #334155;
            color: #c4c4c5;
            border-color: #475569;
        }}
    """


def _secondary_button_style() -> str:
    return f"""
        QPushButton {{
            background-color: {SURFACE_ELEVATED};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
            padding: 10px 18px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: #16233a;
            border-color: #64748b;
        }}
        QPushButton:pressed {{
            background-color: #0f172a;
        }}
        QPushButton:disabled {{
            background-color: #1f2937;
            color: #94a3b8;
            border-color: #334155;
        }}
    """


class BreathResetActivity(QFrame):
    """Guided 4-2-6 breathing activity (short autonomic reset)."""

    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._phases: list[tuple[str, float]] = [
            ("Hít vào", 4.0),
            ("Giữ", 2.0),
            ("Thở ra", 6.0),
        ]
        self._target_cycles = 4

        self._running = False
        self._cycle = 0
        self._phase_index = 0
        self._phase_elapsed = 0.0

        self._min_circle = 64
        self._max_circle = 172
        self._circle_size = self._min_circle

        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick)

        self._init_ui()

    def _init_ui(self):
        self.setObjectName("activityPanel")
        self.setStyleSheet(_activity_panel_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)

        title = QLabel("Nhịp thở 4-2-6")
        title.setFont(QFont("Bahnschrift", 19, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        note = QLabel(
            "Bằng chứng ứng dụng: nhịp thở chậm giúp giảm kích hoạt stress và ổn định lại chú ý trong vài phút."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(note)

        self._circle_area = QWidget()
        self._circle_area.setFixedSize(240, 240)
        layout.addWidget(self._circle_area, 0, Qt.AlignmentFlag.AlignCenter)

        self._phase_label = QLabel("Nhấn Bắt đầu")
        self._phase_label.setFont(QFont("Bahnschrift", 17, QFont.Weight.DemiBold))
        self._phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._phase_label)

        self._timer_label = QLabel("")
        self._timer_label.setObjectName("hint")
        self._timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._timer_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setStyleSheet(_progress_style())
        layout.addWidget(self._progress)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)

        self._start_btn = QPushButton("Bắt đầu")
        self._start_btn.clicked.connect(self.start)
        self._start_btn.setStyleSheet(_primary_button_style())
        buttons.addWidget(self._start_btn)

        self._back_btn = QPushButton("Quay lại")
        self._back_btn.clicked.connect(self._emit_finished)
        self._back_btn.setStyleSheet(_secondary_button_style())
        buttons.addWidget(self._back_btn)

        layout.addLayout(buttons)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        area = self._circle_area.geometry()
        center_x = area.x() + area.width() // 2
        center_y = area.y() + area.height() // 2

        phase_name = self._phases[self._phase_index][0] if self._running else "idle"
        color_map = {
            "idle": QColor("#64748b"),
            "Hít vào": QColor("#38bdf8"),
            "Giữ": QColor("#0ea5e9"),
            "Thở ra": QColor("#0284c7"),
        }
        base = color_map.get(phase_name, QColor("#64748b"))

        glow = QColor(base)
        glow.setAlpha(55)
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            center_x - (self._circle_size + 22) // 2,
            center_y - (self._circle_size + 22) // 2,
            self._circle_size + 22,
            self._circle_size + 22,
        )

        painter.setBrush(QBrush(base))
        painter.drawEllipse(
            center_x - self._circle_size // 2,
            center_y - self._circle_size // 2,
            self._circle_size,
            self._circle_size,
        )

    @pyqtSlot()
    def start(self):
        self._running = True
        self._cycle = 0
        self._phase_index = 0
        self._phase_elapsed = 0.0
        self._circle_size = self._min_circle

        self._start_btn.setEnabled(False)
        self._start_btn.setText("Đang chạy...")
        self._update_phase_labels()
        self._update_progress()

        self._tick_timer.start(50)

    def _update_phase_labels(self):
        phase_name, phase_duration = self._phases[self._phase_index]
        remain = max(0.0, phase_duration - self._phase_elapsed)
        self._phase_label.setText(phase_name)
        self._timer_label.setText(f"Còn {remain:0.1f}s • Vòng {self._cycle + 1}/{self._target_cycles}")

    def _update_progress(self):
        total_phase_count = self._target_cycles * len(self._phases)
        completed_phase_count = self._cycle * len(self._phases) + self._phase_index

        phase_duration = self._phases[self._phase_index][1]
        phase_progress = min(1.0, self._phase_elapsed / max(phase_duration, 1e-6))
        done = completed_phase_count + phase_progress
        percent = int((done / total_phase_count) * 100)
        self._progress.setValue(max(0, min(100, percent)))

    def _update_circle_size(self):
        phase_name, phase_duration = self._phases[self._phase_index]
        ratio = min(1.0, self._phase_elapsed / max(phase_duration, 1e-6))

        if phase_name == "Hít vào":
            self._circle_size = int(self._min_circle + (self._max_circle - self._min_circle) * ratio)
        elif phase_name == "Giữ":
            self._circle_size = self._max_circle
        else:  # Thở ra
            self._circle_size = int(self._max_circle - (self._max_circle - self._min_circle) * ratio)

    def _advance_phase(self):
        self._phase_elapsed = 0.0
        self._phase_index += 1

        if self._phase_index >= len(self._phases):
            self._phase_index = 0
            self._cycle += 1

        if self._cycle >= self._target_cycles:
            self._finish()

    def _tick(self):
        if not self._running:
            return

        self._phase_elapsed += 0.05
        phase_duration = self._phases[self._phase_index][1]

        if self._phase_elapsed >= phase_duration:
            self._advance_phase()
            if not self._running:
                return

        self._update_circle_size()
        self._update_phase_labels()
        self._update_progress()
        self.update()

    def _finish(self):
        self._running = False
        self._tick_timer.stop()
        self._circle_size = self._min_circle
        self._progress.setValue(100)

        self._phase_label.setText("Hoàn thành")
        self._timer_label.setText("Nhịp thở đã ổn định hơn. Chuẩn bị quay lại học.")
        self._start_btn.setEnabled(True)
        self._start_btn.setText("Làm lại")
        self.update()

        QTimer.singleShot(1200, self._emit_finished)

    def _emit_finished(self):
        self.finished.emit()


class VisualResetActivity(QFrame):
    """Guided visual reset based on 20-20-20 inspired micro-routine."""

    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._steps: list[tuple[str, str, int]] = [
            (
                "Nhìn xa khoảng 6m",
                "Mốc 20-20-20: tạm rời màn hình để giảm mỏi điều tiết.",
                20,
            ),
            (
                "Chớp mắt chậm và đều",
                "Giữ nhịp chớp mắt giúp hạn chế khô mắt khi làm việc màn hình lâu.",
                15,
            ),
            (
                "Luân phiên nhìn gần - xa",
                "Đổi tiêu cự vài lần để thư giãn cơ mắt và tái lấy nét.",
                20,
            ),
        ]

        self._running = False
        self._step_index = 0
        self._remaining = 0
        self._total_seconds = sum(step[2] for step in self._steps)
        self._elapsed = 0

        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick)

        self._init_ui()

    def _init_ui(self):
        self.setObjectName("activityPanel")
        self.setStyleSheet(_activity_panel_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)

        title = QLabel("Reset mắt 20-20-20")
        title.setFont(QFont("Bahnschrift", 19, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self._step_title = QLabel("Nhấn Bắt đầu")
        self._step_title.setFont(QFont("Bahnschrift", 17, QFont.Weight.DemiBold))
        self._step_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._step_title)

        self._step_hint = QLabel("Bài tập ngắn 55 giây giúp giảm mỏi mắt khi học liên tục.")
        self._step_hint.setObjectName("hint")
        self._step_hint.setWordWrap(True)
        self._step_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._step_hint)

        self._countdown = QLabel("")
        self._countdown.setFont(QFont("Segoe UI", 30, QFont.Weight.Bold))
        self._countdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._countdown.setStyleSheet("color: #38bdf8; border: none;")
        layout.addWidget(self._countdown)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setStyleSheet(_progress_style())
        layout.addWidget(self._progress)

        self._step_progress_label = QLabel("")
        self._step_progress_label.setObjectName("hint")
        self._step_progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._step_progress_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)

        self._start_btn = QPushButton("Bắt đầu")
        self._start_btn.clicked.connect(self.start)
        self._start_btn.setStyleSheet(_primary_button_style())
        buttons.addWidget(self._start_btn)

        back_btn = QPushButton("Quay lại")
        back_btn.clicked.connect(self.finished.emit)
        back_btn.setStyleSheet(_secondary_button_style())
        buttons.addWidget(back_btn)

        layout.addLayout(buttons)

    @pyqtSlot()
    def start(self):
        self._running = True
        self._step_index = 0
        self._elapsed = 0
        self._enter_step(0)

        self._start_btn.setEnabled(False)
        self._start_btn.setText("Đang chạy...")
        self._tick_timer.start(1000)

    def _enter_step(self, index: int):
        self._step_index = index
        title, hint, duration = self._steps[index]
        self._remaining = duration

        self._step_title.setText(title)
        self._step_hint.setText(hint)
        self._countdown.setText(str(self._remaining))
        self._step_progress_label.setText(f"Bước {index + 1}/{len(self._steps)}")

    def _update_progress(self):
        percent = int((self._elapsed / max(self._total_seconds, 1)) * 100)
        self._progress.setValue(max(0, min(100, percent)))

    def _tick(self):
        if not self._running:
            return

        self._remaining -= 1
        self._elapsed += 1
        self._countdown.setText(str(max(0, self._remaining)))
        self._update_progress()

        if self._remaining <= 0:
            next_step = self._step_index + 1
            if next_step >= len(self._steps):
                self._finish()
                return
            self._enter_step(next_step)

    def _finish(self):
        self._running = False
        self._tick_timer.stop()
        self._progress.setValue(100)
        self._step_title.setText("Hoàn thành")
        self._step_hint.setText("Mắt đã được thả lỏng hơn. Sẵn sàng quay lại công việc.")
        self._countdown.setText("OK")
        self._step_progress_label.setText("")

        self._start_btn.setEnabled(True)
        self._start_btn.setText("Làm lại")

        QTimer.singleShot(1000, self.finished.emit)


class StroopFocusActivity(QFrame):
    """Short Stroop task for executive-control and attention reactivation."""

    finished = pyqtSignal()

    _COLORS: list[tuple[str, str]] = [
        ("ĐỎ", "#ef4444"),
        ("XANH", "#22c55e"),
        ("DƯƠNG", "#3b82f6"),
        ("VÀNG", "#f59e0b"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        self._active = False
        self._round_total = 14
        self._round_index = 0
        self._correct = 0
        self._reaction_times: list[float] = []

        self._target_color_name = ""
        self._round_started_at = 0.0

        self._init_ui()

    def _init_ui(self):
        self.setObjectName("activityPanel")
        self.setStyleSheet(_activity_panel_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)

        title = QLabel("Stroop 60s")
        title.setFont(QFont("Bahnschrift", 19, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        note = QLabel(
            "Nhiệm vụ Stroop ngắn giúp kích hoạt kiểm soát chú ý và ức chế phản xạ đọc tự động."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(note)

        self._progress_label = QLabel("0/14")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setObjectName("hint")
        layout.addWidget(self._progress_label)

        self._stimulus = QLabel("Nhấn Bắt đầu")
        self._stimulus.setFont(QFont("Bahnschrift", 42, QFont.Weight.Bold))
        self._stimulus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stimulus.setMinimumHeight(120)
        layout.addWidget(self._stimulus)

        self._instruction = QLabel("Chọn MÀU CHỮ, không chọn theo nghĩa của từ.")
        self._instruction.setObjectName("hint")
        self._instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._instruction)

        color_row_1 = QHBoxLayout()
        color_row_1.setSpacing(8)
        color_row_2 = QHBoxLayout()
        color_row_2.setSpacing(8)

        self._answer_buttons: list[QPushButton] = []

        for idx, (name, hex_color) in enumerate(self._COLORS):
            btn = QPushButton(name)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _checked=False, n=name: self._submit_answer(n))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #1f2937;
                    color: #e2e8f0;
                    border: 1px solid #334155;
                    border-left: 4px solid {hex_color};
                    border-radius: 10px;
                    padding: 10px 12px;
                    font-weight: 600;
                    min-height: 40px;
                }}
                QPushButton:hover {{
                    background-color: #334155;
                }}
                QPushButton:disabled {{
                    color: #94a3b8;
                    border-color: #334155;
                    border-left: 4px solid #475569;
                    background-color: #1f2937;
                }}
            """)

            if idx < 2:
                color_row_1.addWidget(btn)
            else:
                color_row_2.addWidget(btn)
            self._answer_buttons.append(btn)

        layout.addLayout(color_row_1)
        layout.addLayout(color_row_2)

        self._stats_label = QLabel("")
        self._stats_label.setObjectName("hint")
        self._stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._stats_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)

        self._start_btn = QPushButton("Bắt đầu")
        self._start_btn.clicked.connect(self.start)
        self._start_btn.setStyleSheet(_primary_button_style())
        buttons.addWidget(self._start_btn)

        back_btn = QPushButton("Quay lại")
        back_btn.clicked.connect(self.finished.emit)
        back_btn.setStyleSheet(_secondary_button_style())
        buttons.addWidget(back_btn)

        layout.addLayout(buttons)

    @pyqtSlot()
    def start(self):
        self._active = True
        self._round_index = 0
        self._correct = 0
        self._reaction_times.clear()

        for btn in self._answer_buttons:
            btn.setEnabled(True)

        self._start_btn.setEnabled(False)
        self._start_btn.setText("Đang chạy...")
        self._stats_label.setText("Ưu tiên chính xác trước, rồi mới nhanh.")

        self._next_round()

    def _next_round(self):
        if self._round_index >= self._round_total:
            self._finish()
            return

        word_name, _ = random.choice(self._COLORS)

        if random.random() < 0.75:
            candidates = [item for item in self._COLORS if item[0] != word_name]
            color_name, color_hex = random.choice(candidates)
        else:
            color_name, color_hex = next(item for item in self._COLORS if item[0] == word_name)

        self._target_color_name = color_name
        self._stimulus.setText(word_name)
        self._stimulus.setStyleSheet(f"color: {color_hex};")
        self._progress_label.setText(f"{self._round_index + 1}/{self._round_total}")
        self._round_started_at = time.time()

    def _submit_answer(self, answer_name: str):
        if not self._active:
            return

        rt = max(0.0, time.time() - self._round_started_at)
        self._reaction_times.append(rt)

        if answer_name == self._target_color_name:
            self._correct += 1
            self._stats_label.setText("Đúng")
        else:
            self._stats_label.setText(f"Sai • Đáp án: {self._target_color_name}")

        self._round_index += 1
        QTimer.singleShot(220, self._next_round)

    def _finish(self):
        self._active = False

        for btn in self._answer_buttons:
            btn.setEnabled(False)

        accuracy = (self._correct / max(1, self._round_total)) * 100.0
        avg_rt = (sum(self._reaction_times) / len(self._reaction_times)) if self._reaction_times else 0.0

        self._stimulus.setText("Hoàn thành")
        self._stimulus.setStyleSheet("color: #22c55e;")
        self._instruction.setText("Bạn vừa kích hoạt lại chú ý điều hành.")
        self._stats_label.setText(f"Độ chính xác: {accuracy:.0f}% • Phản hồi TB: {avg_rt:.2f}s")
        self._progress_label.setText(f"{self._round_total}/{self._round_total}")

        self._start_btn.setEnabled(True)
        self._start_btn.setText("Làm lại")

        QTimer.singleShot(1200, self.finished.emit)


class MiniGamesWidget(QDialog):
    """Modal dialog for structured break activities."""

    def __init__(self, parent=None, break_duration_seconds: Optional[int] = None):
        super().__init__(parent)

        self.break_duration_seconds = max(0, int(break_duration_seconds or 0))
        self.remaining_break_seconds = self.break_duration_seconds
        self._enforce_break_duration = self.break_duration_seconds > 0

        self._break_timer = QTimer(self)
        self._break_timer.timeout.connect(self._tick_break_timer)

        self.setWindowTitle("Nghỉ phục hồi tập trung")
        self.setModal(True)

        self.setStyleSheet("""
            QDialog {
                background-color: #050b18;
                color: #e2e8f0;
                font-family: 'Bahnschrift', 'Segoe UI', sans-serif;
            }
            QLabel#subtle {
                color: #cbd5e1;
            }
            QFrame#heroBanner {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 16px;
            }
            QFrame#card {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 14px;
            }
            QLabel#cardTitle {
                color: #f8fafc;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#cardDesc {
                color: #e2e8f0;
                font-size: 12px;
            }
            QLabel#evidence {
                color: #93c5fd;
                font-size: 12px;
            }
            QProgressBar {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 8px;
                text-align: center;
                color: #020617;
                font-weight: 700;
                min-height: 18px;
            }
            QProgressBar::chunk {
                background-color: #38bdf8;
                border-radius: 7px;
            }
        """)

        self._init_ui()
        self._apply_window_geometry()
        self._setup_break_timer()

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
            self.setMinimumSize(620, 500)
            self.resize(740, 640)
            return

        available = screen.availableGeometry()
        margin = 20

        max_w = max(620, available.width() - (margin * 2))
        max_h = max(500, available.height() - (margin * 2))

        desired_w = 740
        desired_h = 640

        target_w = min(desired_w, max_w)
        target_h = min(desired_h, max_h)

        min_w = min(620, max_w)
        min_h = min(500, max_h)
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
        max_w = max(580, available.width() - 10)
        max_h = max(460, available.height() - 10)

        if self.width() > max_w or self.height() > max_h:
            self.resize(min(self.width(), max_w), min(self.height(), max_h))

        max_x = available.right() - self.width() + 1
        max_y = available.bottom() - self.height() + 1
        clamped_x = min(max(self.x(), available.x()), max_x)
        clamped_y = min(max(self.y(), available.y()), max_y)
        self.move(clamped_x, clamped_y)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 22, 24, 24)

        hero = QFrame()
        hero.setObjectName("heroBanner")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(18, 16, 18, 16)
        hero_layout.setSpacing(6)

        title = QLabel("Nghỉ phục hồi có hướng dẫn")
        title.setFont(QFont("Bahnschrift", 22, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #f8fafc; background: transparent; border: none;")
        hero_layout.addWidget(title)

        subtitle = QLabel("Chọn một hoạt động 2-3 phút để giảm mỏi và phục hồi tập trung trước khi quay lại học")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #cbd5e1; background: transparent; border: none; font-size: 12px;")
        hero_layout.addWidget(subtitle)

        layout.addWidget(hero)

        self.break_timer_label = QLabel("Chuẩn bị bắt đầu")
        self.break_timer_label.setObjectName("subtle")
        self.break_timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.break_timer_label)

        self.break_progress = QProgressBar()
        self.break_progress.setRange(0, 100)
        self.break_progress.setValue(0)
        layout.addWidget(self.break_progress)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)

        selection = QWidget()
        sel_layout = QVBoxLayout(selection)
        sel_layout.setSpacing(12)
        sel_layout.setContentsMargins(0, 0, 0, 0)

        self._add_activity_card(
            sel_layout,
            title="Focus Reset Game",
            desc="Chuoi phuc hoi tap trung day du: baseline, go/no-go, sequence memory, visual search, micro-break.",
            evidence="Theo doi reaction time, errors va focus stability de danh gia do phuc hoi tap trung.",
            accent=ACCENT_COLOR,
            callback=self._launch_focus_reset,
        )

        self._add_activity_card(
            sel_layout,
            title="Reset mat 20-20-20",
            desc="Nghi ngan giup giam moi mat va tai can bang thi giac truoc khi quay lai lam viec.",
            evidence="Bai tap nhin xa, chop mat va doi tieu cu ngan han de giam met moi man hinh.",
            accent=ACCENT_COLOR,
            callback=self._show_visual,
        )

        controls = QHBoxLayout()
        controls.setSpacing(10)

        quick_btn = QPushButton("Bắt đầu nhanh: Focus Reset Game")
        quick_btn.clicked.connect(self._launch_focus_reset)
        quick_btn.setStyleSheet(_primary_button_style())
        controls.addWidget(quick_btn, 1)

        self.end_break_btn = QPushButton("Kết thúc nghỉ")
        self.end_break_btn.clicked.connect(self._on_end_break_requested)
        self.end_break_btn.setStyleSheet(_secondary_button_style())
        controls.addWidget(self.end_break_btn)

        sel_layout.addLayout(controls)

        self.stack.addWidget(selection)

        self.visual = VisualResetActivity()
        self.visual.finished.connect(self._show_selection)
        self.stack.addWidget(self.visual)

    def _add_activity_card(
        self,
        container_layout: QVBoxLayout,
        title: str,
        desc: str,
        evidence: str,
        accent: str,
        callback,
    ):
        frame = QFrame()
        frame.setObjectName("card")
        frame.setCursor(Qt.CursorShape.PointingHandCursor)
        frame.setStyleSheet(
            frame.styleSheet() +
            f"QFrame#card:hover {{ border-color: {accent}; background-color: #132138; }}"
        )

        row = QHBoxLayout(frame)
        row.setContentsMargins(14, 12, 14, 12)
        row.setSpacing(10)

        accent_strip = QFrame()
        accent_strip.setFixedWidth(8)
        accent_strip.setStyleSheet(
            f"background-color: {accent};"
            "border-radius: 4px;"
        )
        row.addWidget(accent_strip)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        text_col.addWidget(title_label)

        desc_label = QLabel(desc)
        desc_label.setObjectName("cardDesc")
        desc_label.setWordWrap(True)
        text_col.addWidget(desc_label)

        evidence_label = QLabel(evidence)
        evidence_label.setObjectName("evidence")
        evidence_label.setWordWrap(True)
        text_col.addWidget(evidence_label)

        row.addLayout(text_col, 1)

        go_btn = QPushButton("Bắt đầu")
        go_btn.clicked.connect(callback)
        go_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {accent};
                color: #052033;
                border: 1px solid #7dd3fc;
                border-radius: 10px;
                padding: 9px 13px;
                font-weight: 700;
                min-width: 92px;
            }}
            QPushButton:hover {{
                background-color: #0ea5e9;
            }}
        """)
        row.addWidget(go_btn)

        frame.mousePressEvent = lambda _event, cb=callback: cb()

        container_layout.addWidget(frame)

    def _setup_break_timer(self):
        if not self._enforce_break_duration:
            self.break_progress.hide()
            self.break_timer_label.setText("Gợi ý: chọn 1 hoạt động rồi quay lại học trong 2-5 phút")
            return

        self.end_break_btn.setEnabled(False)
        self._update_break_timer_ui()
        self._break_timer.start(1000)

    def _update_break_timer_ui(self):
        if not self._enforce_break_duration:
            return

        total = max(1, self.break_duration_seconds)
        elapsed = total - self.remaining_break_seconds
        progress = int((elapsed / total) * 100)
        self.break_progress.setValue(max(0, min(100, progress)))
        self.break_timer_label.setText(
            f"Nghỉ phục hồi còn: {self._format_duration(self.remaining_break_seconds)}"
        )

    @staticmethod
    def _format_duration(seconds: int) -> str:
        minutes = seconds // 60
        remain = seconds % 60
        return f"{minutes:02d}:{remain:02d}"

    def _tick_break_timer(self):
        if self.remaining_break_seconds <= 0:
            self._break_timer.stop()
            return

        self.remaining_break_seconds -= 1
        self._update_break_timer_ui()

        if self.remaining_break_seconds <= 0:
            self._break_timer.stop()
            self.end_break_btn.setEnabled(True)
            self.end_break_btn.setText("Quay lại làm việc")
            self.break_timer_label.setText("Đã đủ thời gian nghỉ. Bạn có thể quay lại phiên học.")
            QTimer.singleShot(500, self.accept)

    @pyqtSlot()
    def _show_selection(self):
        self.stack.setCurrentIndex(0)

    @pyqtSlot()
    def _show_breath(self):
        self._launch_focus_reset()

    @pyqtSlot()
    def _show_visual(self):
        self.stack.setCurrentWidget(self.visual)

    @pyqtSlot()
    def _launch_focus_reset(self):
        """Open the full Focus Reset Game workflow as a nested modal dialog."""
        try:
            from ..focus_reset_game.ui import FocusResetDialog

            dialog = FocusResetDialog(self)
            dialog.exec()
        except Exception as exc:
            logger.exception("Failed to open Focus Reset Game dialog")
            QMessageBox.warning(
                self,
                "Focus Reset Game",
                f"Khong the mo Focus Reset Game:\n{exc}",
            )
        finally:
            self._show_selection()

    @pyqtSlot()
    def _show_stroop(self):
        self._launch_focus_reset()

    @pyqtSlot()
    def _on_end_break_requested(self):
        if self._enforce_break_duration and self.remaining_break_seconds > 0:
            self.stack.setCurrentIndex(0)
            self.break_timer_label.setText(
                f"Vui lòng hoàn tất tối thiểu {self._format_duration(self.remaining_break_seconds)} để kết thúc nghỉ"
            )
            return

        self.accept()

    def closeEvent(self, event):
        if self._enforce_break_duration and self.remaining_break_seconds > 0:
            event.ignore()
            return

        if self._break_timer.isActive():
            self._break_timer.stop()

        super().closeEvent(event)


# Backward-compatible names kept for legacy imports/tests.
BreatherGame = BreathResetActivity
MemoryGame = StroopFocusActivity
TypingGame = VisualResetActivity
