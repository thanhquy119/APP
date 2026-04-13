"""
Settings Dialog - Configuration UI for FocusGuardian.

Allows users to configure:
- Camera selection
- Focus thresholds
- Break reminders
- Notification preferences
- Sound settings
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QWidget, QFormLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QSlider, QPushButton, QGroupBox,
    QDialogButtonBox, QFileDialog, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont

import cv2
from .theme import get_stylesheet

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Settings configuration dialog."""

    def __init__(self, config: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.config = config.copy() if config else {}
        self._theme_mode = "dark"
        self.config["theme_mode"] = "dark"

        self.setWindowTitle("Cài đặt - FocusGuardian")
        self.setMinimumSize(500, 450)
        self._apply_theme(True)

        self._init_ui()
        self._load_config()
        self._connect_interactions()
        self._sync_control_states()
        self._on_volume_changed(self.volume_slider.value())

    def _apply_theme(self, is_dark: bool):
        """Apply stylesheet for settings dialog (dark mode only)."""
        _ = is_dark
        base = get_stylesheet(True)
        surface = "#18181b"
        muted = "#cbd5e1"
        border = "#27272a"

        self.setStyleSheet(base + f"""
            QDialog {{
                background-color: {surface};
            }}
            QLabel#sectionHint {{
                color: {muted};
                font-size: 12px;
            }}
            QLabel#volumeValue {{
                color: {muted};
                font-weight: 600;
                min-width: 40px;
            }}
            QDialogButtonBox {{
                border-top: 1px solid {border};
                padding-top: 10px;
            }}
            QDialogButtonBox QPushButton {{
                min-width: 104px;
            }}
        """)

    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        header = QLabel("Tùy chỉnh camera, thuật toán và nhắc nghỉ theo thói quen học của bạn.")
        header.setObjectName("sectionHint")
        header.setWordWrap(True)
        layout.addWidget(header)

        # Tab widget
        self.tabs = QTabWidget()

        # Camera tab
        self.tabs.addTab(self._create_camera_tab(), "Camera")

        # Focus tab
        self.tabs.addTab(self._create_focus_tab(), "Tập trung")

        # Breaks tab
        self.tabs.addTab(self._create_breaks_tab(), "Nghỉ giải lao")

        # Appearance tab
        self.tabs.addTab(self._create_appearance_tab(), "Giao diện")

        # Notifications tab
        self.tabs.addTab(self._create_notifications_tab(), "Thông báo")

        layout.addWidget(self.tabs)

        # Dialog buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.RestoreDefaults
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self._restore_defaults)

        ok_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_btn = self.buttons.button(QDialogButtonBox.StandardButton.Cancel)
        restore_btn = self.buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        if ok_btn is not None:
            ok_btn.setText("Lưu")
            ok_btn.setObjectName("primaryButton")
        if cancel_btn is not None:
            cancel_btn.setText("Hủy")
        if restore_btn is not None:
            restore_btn.setText("Mặc định")
            restore_btn.setObjectName("ghostButton")

        layout.addWidget(self.buttons)

    def _create_camera_tab(self) -> QWidget:
        """Create camera settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Camera selection
        group = QGroupBox("Thiết bị Camera")
        form = QFormLayout(group)

        self.camera_combo = QComboBox()
        self._populate_cameras()
        form.addRow("Camera:", self.camera_combo)

        self.camera_resolution = QComboBox()
        self.camera_resolution.addItems(["640x480", "1280x720", "1920x1080"])
        form.addRow("Độ phân giải:", self.camera_resolution)

        self.camera_fps = QSpinBox()
        self.camera_fps.setRange(10, 60)
        self.camera_fps.setValue(30)
        self.camera_fps.setSuffix(" FPS")
        form.addRow("Tốc độ khung hình:", self.camera_fps)

        # Test camera button
        test_btn = QPushButton("Kiểm tra Camera")
        test_btn.clicked.connect(self._test_camera)
        form.addRow("", test_btn)

        layout.addWidget(group)

        # Preview settings
        preview_group = QGroupBox("Hiển thị")
        preview_form = QFormLayout(preview_group)

        self.show_overlay = QCheckBox("Hiển thị thông tin overlay")
        self.show_overlay.setChecked(True)
        preview_form.addRow(self.show_overlay)

        self.show_face_mesh = QCheckBox("Hiển thị Face Mesh")
        self.show_face_mesh.setChecked(False)
        preview_form.addRow(self.show_face_mesh)

        layout.addWidget(preview_group)
        layout.addStretch()

        return widget

    def _create_focus_tab(self) -> QWidget:
        """Create focus settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Head pose thresholds
        head_group = QGroupBox("Ngưỡng tư thế đầu")
        head_form = QFormLayout(head_group)

        self.head_down_threshold = QDoubleSpinBox()
        self.head_down_threshold.setRange(-45, 0)
        self.head_down_threshold.setValue(-22)
        self.head_down_threshold.setSuffix("°")
        head_form.addRow("Cúi đầu (pitch):", self.head_down_threshold)

        self.look_away_threshold = QDoubleSpinBox()
        self.look_away_threshold.setRange(0, 90)
        self.look_away_threshold.setValue(30)
        self.look_away_threshold.setSuffix("°")
        head_form.addRow("Nhìn ngang (yaw):", self.look_away_threshold)

        layout.addWidget(head_group)

        eye_gaze_group = QGroupBox("Phân biệt cúi đầu và dùng điện thoại")
        eye_gaze_form = QFormLayout(eye_gaze_group)

        self.eye_look_down_threshold = QDoubleSpinBox()
        self.eye_look_down_threshold.setRange(0.1, 0.95)
        self.eye_look_down_threshold.setValue(0.35)
        self.eye_look_down_threshold.setSingleStep(0.05)
        eye_gaze_form.addRow("Ngưỡng mắt nhìn xuống:", self.eye_look_down_threshold)

        self.eye_look_up_threshold = QDoubleSpinBox()
        self.eye_look_up_threshold.setRange(0.1, 0.95)
        self.eye_look_up_threshold.setValue(0.30)
        self.eye_look_up_threshold.setSingleStep(0.05)
        eye_gaze_form.addRow("Ngưỡng mắt nhìn lên màn hình:", self.eye_look_up_threshold)

        self.phone_eye_down_duration = QSpinBox()
        self.phone_eye_down_duration.setRange(10, 180)
        self.phone_eye_down_duration.setValue(45)
        self.phone_eye_down_duration.setSuffix(" giây")
        eye_gaze_form.addRow("Cúi + mắt xuống để tính phone:", self.phone_eye_down_duration)

        self.enable_phone_detection = QCheckBox("Bật nhận diện điện thoại trực tiếp")
        self.enable_phone_detection.setChecked(True)
        eye_gaze_form.addRow(self.enable_phone_detection)

        self.phone_detection_mode = QComboBox()
        self.phone_detection_mode.addItem("Heuristic (nhẹ, không cần model)", "heuristic")
        self.phone_detection_mode.addItem("YOLOv8 (chính xác hơn)", "yolov8")
        self.phone_detection_mode.addItem("Tắt (stub)", "stub")
        eye_gaze_form.addRow("Kiểu nhận diện phone:", self.phone_detection_mode)

        self.phone_confidence_threshold = QDoubleSpinBox()
        self.phone_confidence_threshold.setRange(0.30, 0.95)
        self.phone_confidence_threshold.setSingleStep(0.05)
        self.phone_confidence_threshold.setValue(0.55)
        eye_gaze_form.addRow("Ngưỡng xác nhận phone:", self.phone_confidence_threshold)

        self.low_blink_screen_max = QDoubleSpinBox()
        self.low_blink_screen_max.setRange(3.0, 20.0)
        self.low_blink_screen_max.setValue(10.0)
        self.low_blink_screen_max.setSingleStep(0.5)
        self.low_blink_screen_max.setSuffix(" /phút")
        eye_gaze_form.addRow("Chớp thấp (vẫn tập trung):", self.low_blink_screen_max)

        self.high_blink_fatigue_min = QDoubleSpinBox()
        self.high_blink_fatigue_min.setRange(10.0, 45.0)
        self.high_blink_fatigue_min.setValue(22.0)
        self.high_blink_fatigue_min.setSingleStep(0.5)
        self.high_blink_fatigue_min.setSuffix(" /phút")
        eye_gaze_form.addRow("Chớp cao (mệt mỏi):", self.high_blink_fatigue_min)

        layout.addWidget(eye_gaze_group)

        # Writing detection
        write_group = QGroupBox("Phát hiện ghi chép")
        write_form = QFormLayout(write_group)

        self.write_score_threshold = QDoubleSpinBox()
        self.write_score_threshold.setRange(0.1, 1.0)
        self.write_score_threshold.setValue(0.4)
        self.write_score_threshold.setSingleStep(0.1)
        write_form.addRow("Ngưỡng viết:", self.write_score_threshold)

        layout.addWidget(write_group)

        # Drowsy detection
        drowsy_group = QGroupBox("Phát hiện mệt mỏi")
        drowsy_form = QFormLayout(drowsy_group)

        self.eye_closure_threshold = QDoubleSpinBox()
        self.eye_closure_threshold.setRange(0.1, 0.5)
        self.eye_closure_threshold.setValue(0.3)
        self.eye_closure_threshold.setSingleStep(0.05)
        drowsy_form.addRow("Tỷ lệ nhắm mắt:", self.eye_closure_threshold)

        self.ear_threshold = QDoubleSpinBox()
        self.ear_threshold.setRange(0.1, 0.3)
        self.ear_threshold.setValue(0.18)
        self.ear_threshold.setSingleStep(0.02)
        drowsy_form.addRow("EAR tối thiểu:", self.ear_threshold)

        layout.addWidget(drowsy_group)
        layout.addStretch()

        return widget

    def _create_breaks_tab(self) -> QWidget:
        """Create break settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Break reminders
        reminder_group = QGroupBox("Nhắc nghỉ giải lao")
        reminder_form = QFormLayout(reminder_group)

        self.enable_break_reminders = QCheckBox("Bật nhắc nhở nghỉ giải lao")
        self.enable_break_reminders.setChecked(True)
        reminder_form.addRow(self.enable_break_reminders)

        self.break_interval = QSpinBox()
        self.break_interval.setRange(5, 120)
        self.break_interval.setValue(25)
        self.break_interval.setSuffix(" phút")
        reminder_form.addRow("Thời gian làm việc:", self.break_interval)

        self.break_duration = QSpinBox()
        self.break_duration.setRange(1, 30)
        self.break_duration.setValue(5)
        self.break_duration.setSuffix(" phút")
        reminder_form.addRow("Thời gian nghỉ:", self.break_duration)

        layout.addWidget(reminder_group)

        auto_break_group = QGroupBox("Nghỉ tự động khi mất tập trung")
        auto_break_form = QFormLayout(auto_break_group)

        self.auto_break_on_distraction = QCheckBox(
            "Tự động nhắc nghỉ khi vừa bắt đầu mất tập trung"
        )
        self.auto_break_on_distraction.setChecked(True)
        auto_break_form.addRow(self.auto_break_on_distraction)

        self.distraction_break_cooldown = QSpinBox()
        self.distraction_break_cooldown.setRange(1, 120)
        self.distraction_break_cooldown.setValue(15)
        self.distraction_break_cooldown.setSuffix(" phút")
        auto_break_form.addRow("Khoảng cách giữa 2 lần nhắc:", self.distraction_break_cooldown)

        self.auto_resume_after_break = QCheckBox("Tự quay lại theo dõi sau khi nghỉ")
        self.auto_resume_after_break.setChecked(True)
        auto_break_form.addRow(self.auto_resume_after_break)

        layout.addWidget(auto_break_group)

        # Pomodoro
        pomodoro_group = QGroupBox("Phương pháp Pomodoro")
        pomodoro_form = QFormLayout(pomodoro_group)

        self.enable_pomodoro = QCheckBox("Sử dụng phương pháp Pomodoro")
        self.enable_pomodoro.setChecked(False)
        pomodoro_form.addRow(self.enable_pomodoro)

        self.pomodoro_work = QSpinBox()
        self.pomodoro_work.setRange(15, 60)
        self.pomodoro_work.setValue(25)
        self.pomodoro_work.setSuffix(" phút")
        pomodoro_form.addRow("Thời gian làm việc:", self.pomodoro_work)

        self.pomodoro_short_break = QSpinBox()
        self.pomodoro_short_break.setRange(1, 15)
        self.pomodoro_short_break.setValue(5)
        self.pomodoro_short_break.setSuffix(" phút")
        pomodoro_form.addRow("Nghỉ ngắn:", self.pomodoro_short_break)

        self.pomodoro_long_break = QSpinBox()
        self.pomodoro_long_break.setRange(10, 60)
        self.pomodoro_long_break.setValue(15)
        self.pomodoro_long_break.setSuffix(" phút")
        pomodoro_form.addRow("Nghỉ dài (mỗi 4 vòng):", self.pomodoro_long_break)

        layout.addWidget(pomodoro_group)

        personalization_group = QGroupBox("Cá nhân hóa theo từng người")
        personalization_form = QFormLayout(personalization_group)

        self.profile_name = QLineEdit()
        self.profile_name.setPlaceholderText("Ví dụ: minh, lan, user_01")
        personalization_form.addRow("Hồ sơ:", self.profile_name)

        self.enable_personalization = QCheckBox(
            "Phân tích lịch sử để gợi ý thời gian học/nghỉ"
        )
        self.enable_personalization.setChecked(True)
        personalization_form.addRow(self.enable_personalization)

        self.auto_apply_personalization = QCheckBox(
            "Tự áp dụng khuyến nghị ở phiên tiếp theo"
        )
        self.auto_apply_personalization.setChecked(True)
        personalization_form.addRow(self.auto_apply_personalization)

        layout.addWidget(personalization_group)
        layout.addStretch()

        return widget

    def _create_appearance_tab(self) -> QWidget:
        """Create appearance settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        appearance_group = QGroupBox("Giao diện")
        appearance_form = QFormLayout(appearance_group)

        self.theme_mode_combo = QComboBox()
        self.theme_mode_combo.addItem("Night Mode (Dark)", "dark")
        self.theme_mode_combo.setCurrentIndex(0)
        self.theme_mode_combo.setEnabled(False)
        self.theme_mode_combo.currentIndexChanged.connect(self._on_theme_mode_changed)
        appearance_form.addRow("Chế độ màu:", self.theme_mode_combo)

        note = QLabel("Dark mode đang được cố định để tối ưu trải nghiệm sử dụng.")
        note.setObjectName("sectionHint")
        note.setWordWrap(True)
        appearance_form.addRow("", note)

        layout.addWidget(appearance_group)
        layout.addStretch()

        return widget

    def _create_notifications_tab(self) -> QWidget:
        """Create notifications settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Notification settings
        notif_group = QGroupBox("Thông báo")
        notif_form = QFormLayout(notif_group)

        self.enable_notifications = QCheckBox("Bật thông báo")
        self.enable_notifications.setChecked(True)
        notif_form.addRow(self.enable_notifications)

        self.notify_distraction = QCheckBox("Thông báo khi mất tập trung")
        self.notify_distraction.setChecked(True)
        notif_form.addRow(self.notify_distraction)

        self.notify_break = QCheckBox("Thông báo đến giờ nghỉ")
        self.notify_break.setChecked(True)
        notif_form.addRow(self.notify_break)

        self.notify_drowsy = QCheckBox("Thông báo khi mệt mỏi")
        self.notify_drowsy.setChecked(True)
        notif_form.addRow(self.notify_drowsy)

        layout.addWidget(notif_group)

        # Sound settings
        sound_group = QGroupBox("Âm thanh")
        sound_form = QFormLayout(sound_group)

        self.enable_sounds = QCheckBox("Bật âm thanh")
        self.enable_sounds.setChecked(True)
        sound_form.addRow(self.enable_sounds)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(self.volume_slider)
        self.volume_value = QLabel("70%")
        self.volume_value.setObjectName("volumeValue")
        self.volume_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        volume_layout.addWidget(self.volume_value)
        sound_form.addRow("Âm lượng:", volume_layout)

        # Sound file selection
        sound_layout = QHBoxLayout()
        self.sound_file = QLineEdit()
        self.sound_file.setPlaceholderText("Chọn file âm thanh...")
        sound_layout.addWidget(self.sound_file)

        self.browse_sound_btn = QPushButton("Chọn")
        self.browse_sound_btn.clicked.connect(self._browse_sound)
        sound_layout.addWidget(self.browse_sound_btn)

        sound_form.addRow("File âm thanh:", sound_layout)

        layout.addWidget(sound_group)
        layout.addStretch()

        return widget

    @pyqtSlot(int)
    def _on_theme_mode_changed(self, _index: int):
        """Theme mode is fixed to dark."""
        self._theme_mode = "dark"
        self._apply_theme(True)

    def _connect_interactions(self):
        """Connect dynamic control interactions."""
        self.enable_break_reminders.toggled.connect(self._sync_control_states)
        self.auto_break_on_distraction.toggled.connect(self._sync_control_states)
        self.enable_pomodoro.toggled.connect(self._sync_control_states)
        self.enable_personalization.toggled.connect(self._sync_control_states)
        self.enable_notifications.toggled.connect(self._sync_control_states)
        self.enable_sounds.toggled.connect(self._sync_control_states)
        self.enable_phone_detection.toggled.connect(self._sync_control_states)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)

    @pyqtSlot(int)
    def _on_volume_changed(self, value: int):
        self.volume_value.setText(f"{value}%")

    @pyqtSlot()
    def _sync_control_states(self):
        """Enable/disable dependent controls for a clearer settings UX."""
        breaks_enabled = self.enable_break_reminders.isChecked()
        self.break_interval.setEnabled(breaks_enabled)
        self.break_duration.setEnabled(breaks_enabled)

        auto_break_enabled = breaks_enabled and self.auto_break_on_distraction.isChecked()
        self.distraction_break_cooldown.setEnabled(auto_break_enabled)
        self.auto_resume_after_break.setEnabled(auto_break_enabled)

        pomodoro_enabled = breaks_enabled and self.enable_pomodoro.isChecked()
        self.pomodoro_work.setEnabled(pomodoro_enabled)
        self.pomodoro_short_break.setEnabled(pomodoro_enabled)
        self.pomodoro_long_break.setEnabled(pomodoro_enabled)

        phone_detection_enabled = self.enable_phone_detection.isChecked()
        self.phone_detection_mode.setEnabled(phone_detection_enabled)
        self.phone_confidence_threshold.setEnabled(phone_detection_enabled)

        personalization_enabled = self.enable_personalization.isChecked()
        self.profile_name.setEnabled(personalization_enabled)
        self.auto_apply_personalization.setEnabled(personalization_enabled)

        notifications_enabled = self.enable_notifications.isChecked()
        self.notify_distraction.setEnabled(notifications_enabled)
        self.notify_break.setEnabled(notifications_enabled)
        self.notify_drowsy.setEnabled(notifications_enabled)
        self.enable_sounds.setEnabled(notifications_enabled)

        sounds_enabled = notifications_enabled and self.enable_sounds.isChecked()
        self.volume_slider.setEnabled(sounds_enabled)
        self.sound_file.setEnabled(sounds_enabled)
        self.browse_sound_btn.setEnabled(sounds_enabled)

    def _populate_cameras(self):
        """Populate camera dropdown with available cameras fast."""
        self.camera_combo.clear()

        # Add basic list directly to avoid blocking UI freeze on Windows
        for i in range(4):
            self.camera_combo.addItem(f"Camera {i}", i)

    @pyqtSlot()
    def _test_camera(self):
        """Test selected camera."""
        from PyQt6.QtWidgets import QMessageBox

        camera_id = self.camera_combo.currentData()
        if camera_id is None or camera_id < 0:
            QMessageBox.warning(self, "Lỗi", "Không có camera để kiểm tra.")
            return

        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                QMessageBox.information(
                    self, "Thành công",
                    f"Camera {camera_id} hoạt động bình thường!\n"
                    f"Độ phân giải: {frame.shape[1]}x{frame.shape[0]}"
                )
            else:
                QMessageBox.warning(self, "Lỗi", "Không thể đọc frame từ camera.")
        else:
            QMessageBox.warning(self, "Lỗi", "Không thể mở camera.")

    @pyqtSlot()
    def _browse_sound(self):
        """Browse for sound file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file âm thanh",
            "",
            "Audio Files (*.wav *.mp3 *.ogg)"
        )
        if file_path:
            self.sound_file.setText(file_path)

    def _load_config(self):
        """Load configuration into UI widgets."""
        # Camera
        camera_id = self.config.get("camera_id", 0)
        index = self.camera_combo.findData(camera_id)
        if index >= 0:
            self.camera_combo.setCurrentIndex(index)

        resolution = self.config.get("resolution", "640x480")
        self.camera_resolution.setCurrentText(resolution)

        self.camera_fps.setValue(self.config.get("fps", 30))
        self.show_overlay.setChecked(self.config.get("show_overlay", True))
        self.show_face_mesh.setChecked(self.config.get("show_face_mesh", False))

        # Focus
        self.head_down_threshold.setValue(self.config.get("head_down_threshold", -22))
        self.look_away_threshold.setValue(self.config.get("look_away_threshold", 30))
        self.eye_look_down_threshold.setValue(self.config.get("eye_look_down_threshold", 0.35))
        self.eye_look_up_threshold.setValue(self.config.get("eye_look_up_threshold", 0.30))
        self.phone_eye_down_duration.setValue(self.config.get("phone_eye_down_min_duration", 45))
        self.enable_phone_detection.setChecked(self.config.get("enable_phone_detection", True))
        phone_mode = str(self.config.get("phone_detection_mode", "heuristic"))
        phone_mode_index = self.phone_detection_mode.findData(phone_mode)
        self.phone_detection_mode.setCurrentIndex(phone_mode_index if phone_mode_index >= 0 else 0)
        self.phone_confidence_threshold.setValue(self.config.get("phone_confidence_threshold", 0.55))
        self.low_blink_screen_max.setValue(self.config.get("blink_rate_low_screen_max", 10.0))
        self.high_blink_fatigue_min.setValue(self.config.get("blink_rate_high_fatigue_min", 22.0))
        self.write_score_threshold.setValue(self.config.get("write_score_threshold", 0.4))
        self.eye_closure_threshold.setValue(self.config.get("eye_closure_threshold", 0.3))
        self.ear_threshold.setValue(self.config.get("ear_threshold", 0.18))

        # Breaks
        self.enable_break_reminders.setChecked(self.config.get("enable_break_reminders", True))
        self.break_interval.setValue(self.config.get("break_interval_minutes", 25))
        self.break_duration.setValue(self.config.get("break_duration_minutes", 5))
        self.profile_name.setText(str(self.config.get("profile_name", "default")))
        self.enable_personalization.setChecked(self.config.get("enable_personalization", True))
        self.auto_apply_personalization.setChecked(self.config.get("auto_apply_personalization", True))
        self.auto_break_on_distraction.setChecked(self.config.get("auto_break_on_distraction", True))
        self.distraction_break_cooldown.setValue(self.config.get("distraction_break_cooldown_minutes", 15))
        self.auto_resume_after_break.setChecked(self.config.get("auto_resume_after_break", True))
        self.enable_pomodoro.setChecked(self.config.get("enable_pomodoro", False))
        self.pomodoro_work.setValue(self.config.get("pomodoro_work", 25))
        self.pomodoro_short_break.setValue(self.config.get("pomodoro_short_break", 5))
        self.pomodoro_long_break.setValue(self.config.get("pomodoro_long_break", 15))

        # Appearance
        self.theme_mode_combo.setCurrentIndex(0)

        # Notifications
        self.enable_notifications.setChecked(self.config.get("enable_notifications", True))
        self.notify_distraction.setChecked(self.config.get("notify_distraction", True))
        self.notify_break.setChecked(self.config.get("notify_break", True))
        self.notify_drowsy.setChecked(self.config.get("notify_drowsy", True))
        self.enable_sounds.setChecked(self.config.get("enable_sounds", True))
        self.volume_slider.setValue(self.config.get("volume", 70))
        self.sound_file.setText(self.config.get("sound_file", ""))
        self._sync_control_states()
        self._on_volume_changed(self.volume_slider.value())

    def get_config(self) -> dict:
        """Get configuration from UI widgets."""
        return {
            # Camera
            "camera_id": self.camera_combo.currentData() or 0,
            "resolution": self.camera_resolution.currentText(),
            "fps": self.camera_fps.value(),
            "show_overlay": self.show_overlay.isChecked(),
            "show_face_mesh": self.show_face_mesh.isChecked(),

            # Focus
            "head_down_threshold": self.head_down_threshold.value(),
            "look_away_threshold": self.look_away_threshold.value(),
            "eye_look_down_threshold": self.eye_look_down_threshold.value(),
            "eye_look_up_threshold": self.eye_look_up_threshold.value(),
            "phone_eye_down_min_duration": self.phone_eye_down_duration.value(),
            "enable_phone_detection": self.enable_phone_detection.isChecked(),
            "phone_detection_mode": self.phone_detection_mode.currentData() or "heuristic",
            "phone_confidence_threshold": self.phone_confidence_threshold.value(),
            "blink_rate_low_screen_max": self.low_blink_screen_max.value(),
            "blink_rate_high_fatigue_min": self.high_blink_fatigue_min.value(),
            "write_score_threshold": self.write_score_threshold.value(),
            "eye_closure_threshold": self.eye_closure_threshold.value(),
            "ear_threshold": self.ear_threshold.value(),

            # Breaks
            "enable_break_reminders": self.enable_break_reminders.isChecked(),
            "break_interval_minutes": self.break_interval.value(),
            "break_duration_minutes": self.break_duration.value(),
            "profile_name": self.profile_name.text().strip() or "default",
            "enable_personalization": self.enable_personalization.isChecked(),
            "auto_apply_personalization": self.auto_apply_personalization.isChecked(),
            "auto_break_on_distraction": self.auto_break_on_distraction.isChecked(),
            "distraction_break_cooldown_minutes": self.distraction_break_cooldown.value(),
            "auto_resume_after_break": self.auto_resume_after_break.isChecked(),
            "enable_pomodoro": self.enable_pomodoro.isChecked(),
            "pomodoro_work": self.pomodoro_work.value(),
            "pomodoro_short_break": self.pomodoro_short_break.value(),
            "pomodoro_long_break": self.pomodoro_long_break.value(),

            # Appearance
            "theme_mode": "dark",

            # Notifications
            "enable_notifications": self.enable_notifications.isChecked(),
            "notify_distraction": self.notify_distraction.isChecked(),
            "notify_break": self.notify_break.isChecked(),
            "notify_drowsy": self.notify_drowsy.isChecked(),
            "enable_sounds": self.enable_sounds.isChecked(),
            "volume": self.volume_slider.value(),
            "sound_file": self.sound_file.text(),
        }

    @pyqtSlot()
    def _restore_defaults(self):
        """Restore default settings."""
        self.config = {}
        self._load_config()
        self._sync_control_states()
