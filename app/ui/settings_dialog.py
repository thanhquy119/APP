"""Settings dialog with Appearance, Focus Audio, and Zalo Alerts settings."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt6.QtCore import QObject, QPointF, QSignalBlocker, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .notice_dialog import NoticeDialog
from .theme import get_stylesheet
from ..logic.focus_audio import (
    DEFAULT_FOCUS_AUDIO_TRACK,
    DEFAULT_FOCUS_AUDIO_VOLUME,
    FOCUS_AUDIO_TRACKS,
    FocusAudioManager,
)
from ..logic.zalo_bot import (
    FIXED_ZALO_BOT_TOKEN,
    ZaloBotClient,
    ZaloBotConfig,
)

logger = logging.getLogger(__name__)
ZALO_CONNECT_SUCCESS_TEXT = (
    "FocusGuardian: Kết nối Zalo Bot thành công. "
    "Bạn đã có thể nhận cảnh báo realtime từ ứng dụng."
)
ZALO_QR_EXACT_FILE = Path(__file__).resolve().parents[2] / "assets" / "zalo" / "oa_qr_exact.png"


class _ZaloChatIdFetchWorker(QObject):
    """Background worker for fetching latest chat_id via getUpdates."""

    finished = pyqtSignal(bool, str, str)

    def __init__(
        self,
        app_config: Dict[str, Any],
        max_wait_seconds: int = 75,
        poll_interval_seconds: float = 4.0,
    ):
        super().__init__()
        self._config = dict(app_config or {})
        self._max_wait_seconds = max(10, int(max_wait_seconds))
        self._poll_interval_seconds = max(1.5, float(poll_interval_seconds))

    @staticmethod
    def _can_retry(message: str) -> bool:
        normalized = str(message or "").strip().lower()
        if not normalized:
            return True
        if "webhook" in normalized:
            return False
        if "token" in normalized and "thiếu" in normalized:
            return False
        retry_hints = (
            "không có tin nhắn mới",
            "khong co tin nhan moi",
            "message.chat.id",
            "json",
            "timeout",
            "kết nối",
            "ket noi",
            "mạng",
            "mang",
        )
        return any(hint in normalized for hint in retry_hints)

    @pyqtSlot()
    def run(self) -> None:
        try:
            client = ZaloBotClient(ZaloBotConfig.from_app_config(self._config))
            deadline = time.monotonic() + float(self._max_wait_seconds)
            last_message = "Đang chờ tin nhắn 'Hello' từ Zalo để lấy chat_id..."

            while time.monotonic() <= deadline:
                success, message, chat_id, _ = client.fetch_latest_chat_id()
                if success and chat_id:
                    self.finished.emit(True, str(message), str(chat_id or ""))
                    return

                last_message = str(message or last_message)
                if not self._can_retry(last_message):
                    self.finished.emit(False, last_message, "")
                    return

                time.sleep(self._poll_interval_seconds)

            self.finished.emit(
                False,
                "Hết thời gian chờ. Hãy quét QR, quan tâm bot và nhắn 'Hello' rồi thử lại.",
                "",
            )
        except Exception as exc:
            logger.exception("Unexpected error while fetching Zalo chat_id: %s", exc)
            self.finished.emit(
                False,
                "Không thể lấy chat_id do lỗi nội bộ. Vui lòng thử lại sau.",
                "",
            )


class _ZaloQrGuideDialog(QDialog):
    """Guide dialog that shows QR to connect with Zalo OA before fetching chat_id."""

    def __init__(self, parent=None, is_dark: bool = True):
        super().__init__(parent)
        self._is_dark = bool(is_dark)
        self.setObjectName("zaloQrGuideDialog")
        self.setWindowTitle("Kết nối với Zalo Bot")
        self.setMinimumWidth(460)
        self._apply_theme()
        self._init_ui()

    def _apply_theme(self) -> None:
        base = get_stylesheet(self._is_dark)
        if self._is_dark:
            panel_bg = "rgba(16, 27, 40, 0.97)"
            panel_border = "#35506b"
            title_color = "#eaf3ff"
            hint_color = "#cbd5e1"
            frame_bg = "rgba(10, 18, 28, 0.45)"
            frame_border = "rgba(130, 152, 176, 0.35)"
        else:
            panel_bg = "rgba(255, 255, 255, 0.98)"
            panel_border = "#bfd1e4"
            title_color = "#1f3952"
            hint_color = "#4d6074"
            frame_bg = "rgba(247, 251, 255, 0.95)"
            frame_border = "#cfdaea"

        self.setStyleSheet(
            base
            + f"""
            QDialog#zaloQrGuideDialog {{
                background: {panel_bg};
                border: 1px solid {panel_border};
                border-radius: 18px;
            }}
            QLabel#zaloQrTitle {{
                color: {title_color};
                font-size: 17px;
                font-weight: 700;
            }}
            QLabel#zaloQrHint {{
                color: {hint_color};
                font-size: 12px;
            }}
            QLabel#zaloQrFrame {{
                border: 1px solid {frame_border};
                border-radius: 14px;
                background: {frame_bg};
                padding: 10px;
            }}
            """
        )

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        title = QLabel("Bước 1: Quét mã QR để mở OA trên Zalo")
        title.setObjectName("zaloQrTitle")
        title.setWordWrap(True)
        root.addWidget(title)

        qr_label = QLabel()
        qr_label.setObjectName("zaloQrFrame")
        qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        qr_label.setMinimumHeight(310)

        pixmap = self._load_qr_pixmap()
        if pixmap is not None and not pixmap.isNull():
            qr_label.setPixmap(
                pixmap.scaled(
                    320,
                    320,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            qr_label.setText(
                "Không tải được mã QR.\n"
                "Vui lòng kiểm tra lại file ảnh QR trong assets."
            )
        root.addWidget(qr_label)

        guide = QLabel(
            "Bước 2: Sau khi quét QR và quan tâm OA, hãy nhắn đúng từ 'Hello' cho bot.\n"
            "Ứng dụng đang tự động lấy chat_id ở nền và sẽ tự gửi tin nhắn xác nhận khi kết nối thành công."
        )
        guide.setWordWrap(True)
        guide.setObjectName("zaloQrHint")
        root.addWidget(guide)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 2, 0, 0)
        action_row.setSpacing(8)

        action_row.addStretch(1)

        close_btn = QPushButton("Đóng")
        close_btn.setObjectName("primaryButton")
        close_btn.clicked.connect(self.accept)
        action_row.addWidget(close_btn)

        root.addLayout(action_row)

    @staticmethod
    def _load_qr_pixmap() -> Optional[QPixmap]:
        if not ZALO_QR_EXACT_FILE.exists():
            logger.warning("Exact Zalo QR image not found: %s", ZALO_QR_EXACT_FILE)
            return None

        pixmap = QPixmap(str(ZALO_QR_EXACT_FILE))
        if pixmap.isNull():
            logger.warning("Exact Zalo QR image exists but could not be loaded: %s", ZALO_QR_EXACT_FILE)
            return None
        return pixmap

class CalmComboBox(QComboBox):
    """Theme-aware combobox with a lightweight custom chevron arrow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_dark = True
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(34)

    def set_theme_mode(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        arrow_color = QColor("#b7d1eb" if self._is_dark else "#466885")
        if not self.isEnabled():
            arrow_color = QColor("#6f8195" if self._is_dark else "#9db1c6")

        pen = QPen(
            arrow_color,
            1.8,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(pen)

        center_x = self.width() - 16
        center_y = (self.height() / 2.0) + 0.5
        painter.drawLine(
            QPointF(center_x - 4.0, center_y - 2.2),
            QPointF(center_x, center_y + 2.0),
        )
        painter.drawLine(
            QPointF(center_x, center_y + 2.0),
            QPointF(center_x + 4.0, center_y - 2.2),
        )


class SettingsDialog(QDialog):
    """Refactored settings dialog with appearance/audio and Zalo tabs."""

    config_applied = pyqtSignal(dict)

    def __init__(
        self,
        config: Optional[dict] = None,
        parent=None,
        focus_audio_manager: Optional[FocusAudioManager] = None,
    ):
        super().__init__(parent)
        self.config = dict(config or {})
        self.focus_audio_manager = focus_audio_manager
        self._theme_mode = str(self.config.get("theme_mode", "light")).strip().lower()
        if self._theme_mode not in {"dark", "light"}:
            self._theme_mode = "light"

        # Reuse the same root gradient token as MainWindow.
        self.setObjectName("appRoot")
        self._chat_id_fetch_thread: Optional[QThread] = None
        self._chat_id_fetch_worker: Optional[_ZaloChatIdFetchWorker] = None
        self._zalo_guide_dialog: Optional[_ZaloQrGuideDialog] = None
        self._is_fetching_chat_id = False
        try:
            committed_volume = int(float(self.config.get("focus_audio_volume", DEFAULT_FOCUS_AUDIO_VOLUME)))
        except (TypeError, ValueError):
            committed_volume = DEFAULT_FOCUS_AUDIO_VOLUME
        self._audio_committed_config: Dict[str, Any] = {
            "enable_focus_audio": bool(self.config.get("enable_focus_audio", False)),
            "focus_audio_track": str(self.config.get("focus_audio_track", DEFAULT_FOCUS_AUDIO_TRACK) or DEFAULT_FOCUS_AUDIO_TRACK),
            "focus_audio_volume": max(0, min(100, committed_volume)),
        }

        self.setWindowTitle("Cài đặt")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setMinimumSize(560, 460)

        self._apply_theme(self._theme_mode == "dark")
        self._init_ui()
        self._load_config()
        self._connect_interactions()
        self._sync_control_states()

    def _apply_theme(self, is_dark: bool) -> None:
        self._theme_mode = "dark" if is_dark else "light"
        base = get_stylesheet(is_dark)

        if is_dark:
            muted = "#cbd5e1"
            border = "#2b3644"
            info = "#9fd6ff"
            ok = "#8ff5dd"
            bad = "#ffb4a8"
            pane_bg = "rgba(14, 24, 38, 0.30)"
            combo_bg = "rgba(18, 31, 46, 0.86)"
            combo_border = "#35506b"
            combo_hover = "#466988"
            combo_focus = "#62b6c7"
            combo_text = "#eaf3ff"
            combo_drop_bg = "#132131"
            combo_drop_border = "#35506b"
            combo_item_hover = "rgba(130, 167, 208, 0.20)"
            combo_item_selected = "rgba(89, 213, 192, 0.28)"
            combo_arrow_bg = "rgba(112, 151, 194, 0.16)"
            combo_arrow_sep = "rgba(121, 158, 199, 0.38)"
            volume_input_bg = "rgba(14, 24, 37, 0.92)"
            volume_input_border = "#2e4964"
            volume_input_focus = "#59d5c0"
            volume_text = "#eaf3ff"
            volume_slider_groove = "#2b3f55"
            volume_slider_sub = "#3f8fc9"
            volume_slider_handle = "#59d5c0"
        else:
            muted = "#4d6074"
            border = "#d6deea"
            info = "#296f9d"
            ok = "#0d7a65"
            bad = "#b83232"
            pane_bg = "rgba(234, 243, 252, 0.42)"
            combo_bg = "rgba(255, 255, 255, 0.95)"
            combo_border = "#b8ccdf"
            combo_hover = "#9cb7d2"
            combo_focus = "#4e84b5"
            combo_text = "#1f3952"
            combo_drop_bg = "#f4f9ff"
            combo_drop_border = "#bfd1e4"
            combo_item_hover = "rgba(106, 141, 177, 0.18)"
            combo_item_selected = "rgba(47, 159, 144, 0.24)"
            combo_arrow_bg = "rgba(123, 160, 196, 0.12)"
            combo_arrow_sep = "rgba(120, 156, 193, 0.34)"
            volume_input_bg = "#ffffff"
            volume_input_border = "#b8ccdf"
            volume_input_focus = "#2f9f90"
            volume_text = "#1f3952"
            volume_slider_groove = "#c6d7ea"
            volume_slider_sub = "#70a6d2"
            volume_slider_handle = "#2f9f90"

        self.setStyleSheet(
            base
            + f"""
            QDialog QTabWidget::pane {{
                border: 1px solid {border};
                border-radius: 0px;
                background: {pane_bg};
            }}
            QLabel#sectionHint {{
                color: {muted};
                font-size: 12px;
            }}
            QLabel#statusInfo {{ color: {info}; font-size: 12px; }}
            QLabel#statusOk {{ color: {ok}; font-size: 12px; font-weight: 600; }}
            QLabel#statusError {{ color: {bad}; font-size: 12px; font-weight: 600; }}
            QDialog QGroupBox {{
                border: 1px solid {border};
                border-radius: 20px;
                margin-top: 10px;
                padding: 10px;
                background: transparent;
            }}
            QDialog QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: {muted};
                background: transparent;
            }}
            QComboBox#themeModeCombo,
            QComboBox#focusAudioTrackCombo {{
                background: {combo_bg};
                border: 1px solid {combo_border};
                border-radius: 10px;
                color: {combo_text};
                padding: 7px 36px 7px 12px;
                min-height: 20px;
                font-size: 13px;
                font-weight: 600;
            }}
            QComboBox#themeModeCombo:hover,
            QComboBox#focusAudioTrackCombo:hover {{
                border-color: {combo_hover};
            }}
            QComboBox#themeModeCombo:focus,
            QComboBox#focusAudioTrackCombo:focus {{
                border-color: {combo_focus};
            }}
            QComboBox#themeModeCombo::drop-down,
            QComboBox#focusAudioTrackCombo::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left: 1px solid {combo_arrow_sep};
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
                background: {combo_arrow_bg};
            }}
            QComboBox#themeModeCombo::down-arrow,
            QComboBox#focusAudioTrackCombo::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
            }}
            QComboBoxPrivateContainer {{
                background: {combo_drop_bg};
                border: 1px solid {combo_drop_border};
                border-radius: 10px;
                padding: 0px;
                margin: 0px;
            }}
            QComboBoxPrivateContainer QAbstractItemView {{
                background: {combo_drop_bg};
                border: none;
                margin: 0px;
                padding: 0px;
                outline: 0px;
            }}
            QListView#themeModeComboView,
            QListView#focusAudioTrackComboView {{
                background: {combo_drop_bg};
                color: {combo_text};
                border: none;
                border-radius: 9px;
                padding: 0px;
                margin: 0px;
                outline: 0px;
                selection-background-color: {combo_item_selected};
                selection-color: {combo_text};
            }}
            QListView#themeModeComboView::item,
            QListView#focusAudioTrackComboView::item {{
                min-height: 30px;
                padding: 6px 10px;
                border-radius: 7px;
                margin: 0px;
            }}
            QListView#themeModeComboView::item:hover,
            QListView#focusAudioTrackComboView::item:hover {{
                background: {combo_item_hover};
            }}
            QListView#themeModeComboView::item:selected,
            QListView#focusAudioTrackComboView::item:selected {{
                background: {combo_item_selected};
            }}
            QSpinBox#focusAudioVolumeSpin {{
                background: {volume_input_bg};
                border: 1px solid {volume_input_border};
                border-radius: 20px;
                color: {volume_text};
                min-height: 28px;
                padding: 6px 10px;
                font-size: 13px;
                font-weight: 700;
                selection-background-color: {volume_input_focus};
            }}
            QSpinBox#focusAudioVolumeSpin:focus {{
                border-color: {volume_input_focus};
            }}
            QSpinBox#focusAudioVolumeSpin::up-button,
            QSpinBox#focusAudioVolumeSpin::down-button,
            QSpinBox#focusAudioVolumeSpin::up-arrow,
            QSpinBox#focusAudioVolumeSpin::down-arrow {{
                width: 0px;
                height: 0px;
                border: none;
                background: transparent;
            }}
            QSlider#focusAudioVolumeSlider::groove:horizontal {{
                height: 7px;
                border-radius: 4px;
                background: {volume_slider_groove};
            }}
            QSlider#focusAudioVolumeSlider::sub-page:horizontal {{
                border-radius: 4px;
                background: {volume_slider_sub};
            }}
            QSlider#focusAudioVolumeSlider::add-page:horizontal {{
                border-radius: 4px;
                background: {volume_slider_groove};
            }}
            QSlider#focusAudioVolumeSlider::handle:horizontal {{
                width: 18px;
                height: 18px;
                border-radius: 9px;
                margin: -6px 0;
                background: {volume_slider_handle};
                border: 1px solid {volume_slider_handle};
            }}
            """
        )

        if hasattr(self, "theme_mode_combo"):
            self.theme_mode_combo.set_theme_mode(is_dark)
        if hasattr(self, "focus_audio_track_combo"):
            self.focus_audio_track_combo.set_theme_mode(is_dark)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_appearance_tab(), "Chung")
        self.tabs.addTab(self._create_zalo_tab(), "Thông báo")
        layout.addWidget(self.tabs)

        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 4, 0, 0)
        footer_layout.setSpacing(8)
        footer_layout.addStretch(1)

        self.btn_close = QPushButton("Đóng")
        self.btn_close.setObjectName("ghostButton")
        self.btn_close.clicked.connect(self.reject)
        footer_layout.addWidget(self.btn_close)

        self.btn_save = QPushButton("Lưu")
        self.btn_save.setObjectName("primaryButton")
        self.btn_save.clicked.connect(self._apply_changes)
        footer_layout.addWidget(self.btn_save)

        layout.addLayout(footer_layout)

    def _create_appearance_tab(self) -> QWidget:
        widget = QWidget()
        root = QVBoxLayout(widget)

        group = QGroupBox("Giao diện")
        form = QFormLayout(group)

        self.theme_mode_combo = CalmComboBox()
        self.theme_mode_combo.setObjectName("themeModeCombo")
        theme_view = QListView(self.theme_mode_combo)
        theme_view.setObjectName("themeModeComboView")
        theme_view.setFrameShape(QFrame.Shape.NoFrame)
        theme_view.setContentsMargins(0, 0, 0, 0)
        theme_view.setViewportMargins(0, 0, 0, 0)
        theme_view.setSpacing(0)
        theme_view.setUniformItemSizes(True)
        self.theme_mode_combo.setView(theme_view)
        self.theme_mode_combo.addItem("Sáng", "light")
        self.theme_mode_combo.addItem("Tối", "dark")
        form.addRow("Chế độ màu:", self.theme_mode_combo)

        hint = QLabel("Thay đổi theme chỉ được áp dụng sau khi bấm Lưu.")
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        form.addRow("", hint)

        audio_group = QGroupBox("Âm thanh hỗ trợ tập trung")
        audio_form = QFormLayout(audio_group)

        self.enable_focus_audio = QCheckBox("Bật âm thanh hỗ trợ tập trung")
        audio_form.addRow(self.enable_focus_audio)

        self.focus_audio_track_combo = CalmComboBox()
        self.focus_audio_track_combo.setObjectName("focusAudioTrackCombo")
        focus_audio_view = QListView(self.focus_audio_track_combo)
        focus_audio_view.setObjectName("focusAudioTrackComboView")
        focus_audio_view.setFrameShape(QFrame.Shape.NoFrame)
        focus_audio_view.setContentsMargins(0, 0, 0, 0)
        focus_audio_view.setViewportMargins(0, 0, 0, 0)
        focus_audio_view.setSpacing(0)
        focus_audio_view.setUniformItemSizes(True)
        self.focus_audio_track_combo.setView(focus_audio_view)
        for track in FOCUS_AUDIO_TRACKS:
            self.focus_audio_track_combo.addItem(track.label, track.key)
        audio_form.addRow("Chọn loại âm thanh:", self.focus_audio_track_combo)

        volume_row = QHBoxLayout()
        volume_row.setContentsMargins(0, 0, 0, 0)
        volume_row.setSpacing(8)

        self.focus_audio_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.focus_audio_volume_slider.setObjectName("focusAudioVolumeSlider")
        self.focus_audio_volume_slider.setRange(0, 100)
        self.focus_audio_volume_slider.setSingleStep(1)
        self.focus_audio_volume_slider.setPageStep(5)

        self.focus_audio_volume_spin = QSpinBox()
        self.focus_audio_volume_spin.setObjectName("focusAudioVolumeSpin")
        self.focus_audio_volume_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.focus_audio_volume_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.focus_audio_volume_spin.setRange(0, 100)
        self.focus_audio_volume_spin.setSuffix("%")
        self.focus_audio_volume_spin.setSingleStep(1)
        self.focus_audio_volume_spin.setFixedWidth(84)

        volume_row.addWidget(self.focus_audio_volume_slider, 1)
        volume_row.addWidget(self.focus_audio_volume_spin)
        audio_form.addRow("Âm lượng:", volume_row)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)

        self.focus_audio_preview_btn = QPushButton("Nghe thử")
        self.focus_audio_preview_btn.setObjectName("ghostButton")
        action_row.addWidget(self.focus_audio_preview_btn)

        self.focus_audio_stop_btn = QPushButton("Dừng")
        self.focus_audio_stop_btn.setObjectName("ghostButton")
        action_row.addWidget(self.focus_audio_stop_btn)

        action_row.addStretch(1)
        audio_form.addRow("", action_row)

        self.focus_audio_status_label = QLabel("Âm thanh nền đang tắt.")
        self.focus_audio_status_label.setObjectName("statusInfo")
        self.focus_audio_status_label.setWordWrap(True)
        audio_form.addRow("", self.focus_audio_status_label)


        root.addWidget(group)
        root.addWidget(audio_group)
        root.addStretch(1)
        return widget

    def _create_zalo_tab(self) -> QWidget:
        widget = QWidget()
        root = QVBoxLayout(widget)

        group = QGroupBox("Thiết lập Zalo Alerts")
        form = QFormLayout(group)

        self.enable_zalo_alerts = QCheckBox("Bật cảnh báo realtime qua Zalo Bot")
        form.addRow(self.enable_zalo_alerts)

        action_row = QHBoxLayout()
        self.zalo_connect_bot_btn = QPushButton("Kết nối với Zalo Bot")
        self.zalo_connect_bot_btn.setObjectName("primaryButton")
        self.zalo_connect_bot_btn.clicked.connect(self._start_zalo_connect_flow)
        action_row.addWidget(self.zalo_connect_bot_btn)

        self.zalo_test_message_btn = QPushButton("Gửi thử tin nhắn")
        self.zalo_test_message_btn.setObjectName("ghostButton")
        self.zalo_test_message_btn.clicked.connect(self._send_test_zalo_message)
        action_row.addWidget(self.zalo_test_message_btn)

        form.addRow("", action_row)

        self.zalo_status_label = QLabel(
            "Nhấn 'Kết nối với Zalo Bot' để mở QR và tự động lấy chat_id. Sau khi quét, hãy nhắn 'Hello' cho bot."
        )
        self.zalo_status_label.setObjectName("statusInfo")
        self.zalo_status_label.setWordWrap(True)
        form.addRow("", self.zalo_status_label)

        root.addWidget(group)
        root.addStretch(1)
        return widget

    def _connect_interactions(self) -> None:
        self.enable_focus_audio.toggled.connect(self._on_focus_audio_enabled_toggled)
        self.focus_audio_track_combo.currentIndexChanged.connect(self._on_focus_audio_track_changed)
        self.focus_audio_volume_slider.valueChanged.connect(self._on_focus_audio_volume_slider_changed)
        self.focus_audio_volume_spin.valueChanged.connect(self._on_focus_audio_volume_spin_changed)
        self.focus_audio_preview_btn.clicked.connect(self._preview_focus_audio)
        self.focus_audio_stop_btn.clicked.connect(self._stop_focus_audio)

        self.enable_zalo_alerts.toggled.connect(self._sync_control_states)

    def _set_audio_status(self, text: str, level: str = "info") -> None:
        if level == "ok":
            self.focus_audio_status_label.setObjectName("statusOk")
        elif level == "error":
            self.focus_audio_status_label.setObjectName("statusError")
        else:
            self.focus_audio_status_label.setObjectName("statusInfo")

        self.focus_audio_status_label.setText(text)
        self.focus_audio_status_label.style().unpolish(self.focus_audio_status_label)
        self.focus_audio_status_label.style().polish(self.focus_audio_status_label)

    def _audio_manager_ready(self) -> bool:
        return self.focus_audio_manager is not None and self.focus_audio_manager.is_available()

    def _extract_audio_form_config(self) -> Dict[str, Any]:
        track_key = str(self.focus_audio_track_combo.currentData() or DEFAULT_FOCUS_AUDIO_TRACK)
        return {
            "enable_focus_audio": self.enable_focus_audio.isChecked(),
            "focus_audio_track": track_key,
            "focus_audio_volume": int(self.focus_audio_volume_spin.value()),
        }

    def _restore_audio_runtime_to_committed(self) -> None:
        if not self._audio_manager_ready():
            return

        try:
            self.focus_audio_manager.load_from_config(self._audio_committed_config)
        except Exception as exc:
            logger.exception("Failed to restore committed focus audio state: %s", exc)

    @pyqtSlot(bool)
    def _on_focus_audio_enabled_toggled(self, enabled: bool) -> None:
        if not self._audio_manager_ready():
            self._set_audio_status("Backend audio không khả dụng trong môi trường hiện tại.", level="error")
            return

        track_key = str(self.focus_audio_track_combo.currentData() or DEFAULT_FOCUS_AUDIO_TRACK)
        volume = int(self.focus_audio_volume_spin.value())
        self.focus_audio_manager.set_volume(volume)

        if enabled:
            ok, message = self.focus_audio_manager.play(track_key)
            if ok:
                self._set_audio_status("Đã bật âm thanh nền hỗ trợ tập trung.", level="ok")
            else:
                blocker = QSignalBlocker(self.enable_focus_audio)
                self.enable_focus_audio.setChecked(False)
                del blocker
                self._set_audio_status(message or "Không thể phát audio đã chọn.", level="error")
        else:
            self.focus_audio_manager.set_enabled(False)
            self._set_audio_status("Đã tắt âm thanh nền.", level="info")

    @pyqtSlot(int)
    def _on_focus_audio_track_changed(self, index: int) -> None:
        _ = index
        if not self._audio_manager_ready():
            return

        track_key = str(self.focus_audio_track_combo.currentData() or DEFAULT_FOCUS_AUDIO_TRACK)
        self.focus_audio_manager.set_track(track_key)

        if self.enable_focus_audio.isChecked():
            ok, message = self.focus_audio_manager.play(track_key)
            if not ok:
                self._set_audio_status(message or "Không thể chuyển sang âm thanh mới.", level="error")
                return
            self._set_audio_status("Đã chuyển sang âm thanh nền mới.", level="ok")
            return

        if self.focus_audio_manager.is_previewing():
            ok, message = self.focus_audio_manager.preview(track_key)
            if not ok:
                self._set_audio_status(message or "Không thể nghe thử âm thanh mới.", level="error")

    @pyqtSlot(int)
    def _on_focus_audio_volume_slider_changed(self, value: int) -> None:
        clamped = max(0, min(100, int(value)))
        blocker = QSignalBlocker(self.focus_audio_volume_spin)
        self.focus_audio_volume_spin.setValue(clamped)
        del blocker
        self._apply_focus_audio_volume(clamped)

    @pyqtSlot(int)
    def _on_focus_audio_volume_spin_changed(self, value: int) -> None:
        clamped = max(0, min(100, int(value)))
        blocker = QSignalBlocker(self.focus_audio_volume_slider)
        self.focus_audio_volume_slider.setValue(clamped)
        del blocker
        self._apply_focus_audio_volume(clamped)

    def _apply_focus_audio_volume(self, value: int) -> None:
        if self._audio_manager_ready():
            self.focus_audio_manager.set_volume(value)

    @pyqtSlot()
    def _preview_focus_audio(self) -> None:
        if not self._audio_manager_ready():
            self._set_audio_status("Backend audio không khả dụng trong môi trường hiện tại.", level="error")
            return

        track_key = str(self.focus_audio_track_combo.currentData() or DEFAULT_FOCUS_AUDIO_TRACK)
        volume = int(self.focus_audio_volume_spin.value())
        self.focus_audio_manager.set_volume(volume)
        ok, message = self.focus_audio_manager.preview(track_key)
        if ok:
            self._set_audio_status("Đang nghe thử âm thanh đã chọn.", level="ok")
        else:
            self._set_audio_status(message or "Không thể nghe thử âm thanh đã chọn.", level="error")

    @pyqtSlot()
    def _stop_focus_audio(self) -> None:
        if not self._audio_manager_ready():
            return

        self.focus_audio_manager.stop()
        if self.enable_focus_audio.isChecked():
            self._set_audio_status("Đã dừng tạm thời. Bấm Nghe thử hoặc bật lại để phát.", level="info")
        else:
            self._set_audio_status("Đã dừng âm thanh.", level="info")

    @pyqtSlot()
    def _sync_control_states(self) -> None:
        audio_ready = self._audio_manager_ready()
        self.focus_audio_preview_btn.setEnabled(audio_ready)
        self.focus_audio_stop_btn.setEnabled(audio_ready)
        if not audio_ready:
            self._set_audio_status("Backend audio không khả dụng trong môi trường hiện tại.", level="error")

        zalo_enabled = self.enable_zalo_alerts.isChecked()
        has_chat_id = bool(str(self.config.get("zalo_chat_id", "") or "").strip())
        can_connect = zalo_enabled and not self._is_fetching_chat_id
        can_test = zalo_enabled and has_chat_id and not self._is_fetching_chat_id

        self.zalo_connect_bot_btn.setEnabled(can_connect)
        self.zalo_test_message_btn.setEnabled(can_test)

        self.btn_save.setEnabled(not self._is_fetching_chat_id)
        self.btn_close.setEnabled(not self._is_fetching_chat_id)

    def _set_zalo_status(self, text: str, level: str = "info") -> None:
        if level == "ok":
            self.zalo_status_label.setObjectName("statusOk")
        elif level == "error":
            self.zalo_status_label.setObjectName("statusError")
        else:
            self.zalo_status_label.setObjectName("statusInfo")

        self.zalo_status_label.setText(text)
        self.zalo_status_label.style().unpolish(self.zalo_status_label)
        self.zalo_status_label.style().polish(self.zalo_status_label)

    @staticmethod
    def _friendly_chat_id_fetch_error(message: str) -> str:
        raw = str(message or "").strip()
        if not raw:
            return "Không lấy được chat_id. Hãy quét QR, nhắn 'Hello' rồi thử lại."

        lowered = raw.lower()
        if "timeout" in lowered:
            return "API Zalo phản hồi chậm. Hãy đảm bảo đã nhắn 'Hello' cho bot rồi thử lại."
        if "webhook" in lowered:
            return "Bot đang bật webhook nên getUpdates không hoạt động."
        if "khong co tin nhan moi" in lowered:
            return "Chưa có tin nhắn mới. Hãy quét QR, quan tâm bot và nhắn 'Hello' rồi thử lại."
        if "khong tim thay message.chat.id" in lowered:
            return "Đã có updates nhưng chưa thấy message.chat.id."
        if "khong chua message.chat.id" in lowered:
            return "Dữ liệu trả về chưa có message.chat.id."
        if "loi mang" in lowered or "ket noi" in lowered:
            return "Không kết nối được tới API Zalo. Hãy kiểm tra mạng."
        if "json" in lowered:
            return "Dữ liệu phản hồi từ Zalo chưa hợp lệ."
        if "traceback" in lowered or "httpsconnectionpool" in lowered:
            return "Không thể lấy chat_id do lỗi kết nối nội bộ."
        return raw

    def _collect_zalo_form_config(self) -> Dict[str, Any]:
        resolved_chat_id = str(self.config.get("zalo_chat_id", "") or "").strip()
        return {
            "enable_zalo_alerts": self.enable_zalo_alerts.isChecked(),
            "zalo_bot_token": FIXED_ZALO_BOT_TOKEN,
            "zalo_chat_id": resolved_chat_id,
            "zalo_webhook_secret": str(self.config.get("zalo_webhook_secret", "") or ""),
            "zalo_alert_on_distraction": bool(self.config.get("zalo_alert_on_distraction", True)),
            "zalo_alert_on_drowsy": bool(self.config.get("zalo_alert_on_drowsy", True)),
            "zalo_alert_on_phone": bool(self.config.get("zalo_alert_on_phone", True)),
            "zalo_alert_on_away": bool(self.config.get("zalo_alert_on_away", True)),
            "zalo_alert_on_break_reminder": bool(self.config.get("zalo_alert_on_break_reminder", True)),
        }

    @pyqtSlot()
    def _send_test_zalo_message(self) -> None:
        if self._is_fetching_chat_id:
            return

        config = self._collect_zalo_form_config()
        if not str(config.get("zalo_chat_id", "") or "").strip():
            NoticeDialog.info(
                self,
                "Zalo Alerts",
                "Bạn chưa kết nối bot. Hãy bấm 'Kết nối với Zalo Bot' trước.",
                config=self.config,
            )
            self._set_zalo_status("Chưa có chat_id. Vui lòng kết nối bot trước khi gửi thử.", level="error")
            return

        client = ZaloBotClient(ZaloBotConfig.from_app_config(config))

        timestamp = time.strftime("%H:%M:%S %d/%m/%Y")
        test_text = (
            "FocusGuardian: Đây là tin nhắn test cảnh báo realtime qua Zalo.\n"
            "Hồ sơ: phiên hiện tại\n"
            f"Thời điểm: {timestamp}"
        )
        success, detail, _ = client.send_message(config.get("zalo_chat_id"), test_text)

        if success:
            NoticeDialog.info(
                self,
                "Zalo Alerts",
                "Đã gửi tin nhắn test thành công.",
                config=self.config,
            )
            self._set_zalo_status("Đã gửi tin nhắn test thành công.", level="ok")
        else:
            NoticeDialog.warning(
                self,
                "Zalo Alerts",
                f"Không gửi được tin nhắn test:\n{detail}",
                config=self.config,
            )
            self._set_zalo_status(f"Lỗi gửi test: {detail}", level="error")

    @pyqtSlot()
    def _start_zalo_connect_flow(self) -> None:
        if self._is_fetching_chat_id:
            return

        if not self.enable_zalo_alerts.isChecked():
            self._set_zalo_status("Hãy bật Zalo Alerts trước khi kết nối bot.", level="error")
            return

        self._fetch_chat_id_automatically()

        guide_dialog = _ZaloQrGuideDialog(self, is_dark=(self._theme_mode == "dark"))
        self._zalo_guide_dialog = guide_dialog
        try:
            guide_dialog.exec()
        finally:
            if self._zalo_guide_dialog is guide_dialog:
                self._zalo_guide_dialog = None

    def _dismiss_zalo_guide_dialog(self) -> None:
        dialog = self._zalo_guide_dialog
        if dialog is None:
            return

        self._zalo_guide_dialog = None
        if dialog.isVisible():
            dialog.accept()

    @pyqtSlot()
    def _fetch_chat_id_automatically(self) -> None:
        if self._is_fetching_chat_id:
            return

        config = self._collect_zalo_form_config()

        self._is_fetching_chat_id = True
        self._set_zalo_status(
            "Đang kết nối... Nếu chưa gửi, hãy nhắn 'Hello' cho bot. Hệ thống đang tự động lấy chat_id.",
            level="info",
        )
        self._sync_control_states()

        self._chat_id_fetch_thread = QThread(self)
        self._chat_id_fetch_worker = _ZaloChatIdFetchWorker(
            config,
            max_wait_seconds=75,
            poll_interval_seconds=4.0,
        )
        self._chat_id_fetch_worker.moveToThread(self._chat_id_fetch_thread)

        self._chat_id_fetch_thread.started.connect(self._chat_id_fetch_worker.run)
        self._chat_id_fetch_worker.finished.connect(self._on_chat_id_fetch_finished)
        self._chat_id_fetch_worker.finished.connect(self._chat_id_fetch_thread.quit)
        self._chat_id_fetch_worker.finished.connect(self._chat_id_fetch_worker.deleteLater)
        self._chat_id_fetch_thread.finished.connect(self._on_chat_id_fetch_thread_finished)
        self._chat_id_fetch_thread.finished.connect(self._chat_id_fetch_thread.deleteLater)

        self._chat_id_fetch_thread.start()

    @pyqtSlot(bool, str, str)
    def _on_chat_id_fetch_finished(self, success: bool, message: str, chat_id: str) -> None:
        self._is_fetching_chat_id = False

        if success and chat_id:
            self._dismiss_zalo_guide_dialog()
            self.config["zalo_chat_id"] = str(chat_id).strip()
            self.config["zalo_bot_token"] = FIXED_ZALO_BOT_TOKEN

            client_config = self._collect_zalo_form_config()
            client = ZaloBotClient(ZaloBotConfig.from_app_config(client_config))
            sent_ok, sent_detail, _ = client.send_message(self.config.get("zalo_chat_id"), ZALO_CONNECT_SUCCESS_TEXT)

            self.config_applied.emit(
                {
                    "zalo_bot_token": FIXED_ZALO_BOT_TOKEN,
                    "zalo_chat_id": self.config.get("zalo_chat_id", ""),
                }
            )

            if sent_ok:
                self._set_zalo_status(
                    "Kết nối thành công. Bot đã gửi tin nhắn xác nhận qua Zalo.",
                    level="ok",
                )
            else:
                logger.warning("Connected chat_id but failed to send connect confirmation: %s", sent_detail)
                self._set_zalo_status(
                    "Đã kết nối bot, nhưng chưa gửi được tin xác nhận. Bạn vẫn có thể nhận cảnh báo.",
                    level="ok",
                )
        else:
            self._set_zalo_status(self._friendly_chat_id_fetch_error(message), level="error")

        self._sync_control_states()

    @pyqtSlot()
    def _on_chat_id_fetch_thread_finished(self) -> None:
        self._chat_id_fetch_worker = None
        self._chat_id_fetch_thread = None

    def _load_config(self) -> None:
        theme_mode = str(self.config.get("theme_mode", "light")).strip().lower()
        idx = self.theme_mode_combo.findData(theme_mode)
        self.theme_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.enable_focus_audio.setChecked(bool(self.config.get("enable_focus_audio", False)))

        track_key = str(self.config.get("focus_audio_track", DEFAULT_FOCUS_AUDIO_TRACK) or DEFAULT_FOCUS_AUDIO_TRACK)
        track_idx = self.focus_audio_track_combo.findData(track_key)
        self.focus_audio_track_combo.setCurrentIndex(track_idx if track_idx >= 0 else 0)

        try:
            focus_volume = int(float(self.config.get("focus_audio_volume", DEFAULT_FOCUS_AUDIO_VOLUME)))
        except (TypeError, ValueError):
            focus_volume = DEFAULT_FOCUS_AUDIO_VOLUME
        focus_volume = max(0, min(100, focus_volume))
        self.focus_audio_volume_slider.setValue(focus_volume)
        self.focus_audio_volume_spin.setValue(focus_volume)

        if self.enable_focus_audio.isChecked():
            self._set_audio_status("Âm thanh nền đang bật.", level="ok")
        else:
            self._set_audio_status("Âm thanh nền đang tắt.", level="info")

        self.enable_zalo_alerts.setChecked(bool(self.config.get("enable_zalo_alerts", False)))
        self.config["zalo_bot_token"] = FIXED_ZALO_BOT_TOKEN
        self.config["zalo_chat_id"] = str(self.config.get("zalo_chat_id", "") or "").strip()

        if self.config.get("zalo_chat_id"):
            self._set_zalo_status("Đã liên kết Zalo Bot. Bạn có thể gửi thử tin nhắn.", level="ok")
        else:
            self._set_zalo_status(
                "Nhấn 'Kết nối với Zalo Bot' để mở QR, sau đó nhắn 'Hello' cho bot.",
                level="info",
            )

        self._audio_committed_config = self._extract_audio_form_config()

    def get_config(self) -> dict:
        theme_mode = str(self.theme_mode_combo.currentData() or self._theme_mode)
        return {
            "theme_mode": theme_mode,
            "enable_focus_audio": self.enable_focus_audio.isChecked(),
            "focus_audio_track": str(self.focus_audio_track_combo.currentData() or DEFAULT_FOCUS_AUDIO_TRACK),
            "focus_audio_volume": int(self.focus_audio_volume_spin.value()),
            "enable_zalo_alerts": self.enable_zalo_alerts.isChecked(),
            "zalo_bot_token": FIXED_ZALO_BOT_TOKEN,
            "zalo_chat_id": str(self.config.get("zalo_chat_id", "") or "").strip(),
            "zalo_webhook_secret": str(self.config.get("zalo_webhook_secret", "") or ""),
            "zalo_alert_on_distraction": bool(self.config.get("zalo_alert_on_distraction", True)),
            "zalo_alert_on_drowsy": bool(self.config.get("zalo_alert_on_drowsy", True)),
            "zalo_alert_on_phone": bool(self.config.get("zalo_alert_on_phone", True)),
            "zalo_alert_on_away": bool(self.config.get("zalo_alert_on_away", True)),
            "zalo_alert_on_break_reminder": bool(self.config.get("zalo_alert_on_break_reminder", True)),
        }

    @pyqtSlot()
    def _apply_changes(self) -> None:
        updates = self.get_config()
        self.config.update(updates)
        self._audio_committed_config = {
            "enable_focus_audio": bool(updates.get("enable_focus_audio", False)),
            "focus_audio_track": str(updates.get("focus_audio_track", DEFAULT_FOCUS_AUDIO_TRACK) or DEFAULT_FOCUS_AUDIO_TRACK),
            "focus_audio_volume": int(updates.get("focus_audio_volume", DEFAULT_FOCUS_AUDIO_VOLUME) or DEFAULT_FOCUS_AUDIO_VOLUME),
        }

        if self._audio_manager_ready():
            self.focus_audio_manager.load_from_config(self._audio_committed_config)

        self.config_applied.emit(dict(updates))
        # Apply dialog theme only after Save; no preview while selecting.
        mode = str(updates.get("theme_mode", self._theme_mode)).strip().lower()
        self._apply_theme(mode != "light")
        self._set_zalo_status("Đã lưu cài đặt.", level="ok")

    def closeEvent(self, event) -> None:
        if self._is_fetching_chat_id:
            NoticeDialog.info(
                self,
                "Zalo Alerts",
                "Đang lấy chat_id. Vui lòng chờ hoàn tất.",
                config=self.config,
            )
            event.ignore()
            return
        self._restore_audio_runtime_to_committed()
        super().closeEvent(event)

    def reject(self) -> None:
        if self._is_fetching_chat_id:
            NoticeDialog.info(
                self,
                "Zalo Alerts",
                "Đang lấy chat_id. Vui lòng chờ hoàn tất.",
                config=self.config,
            )
            return
        self._restore_audio_runtime_to_committed()
        super().reject()
