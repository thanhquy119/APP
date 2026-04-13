#!/usr/bin/env python3
"""
FocusGuardian - Focus Tracking Desktop Application

Main entry point for the application.
"""

import sys
import logging
import json
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("focusguardian.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# Configuration file path
CONFIG_FILE = Path("config.json")
DEFAULT_CONFIG = {
    "camera_id": 0,
    "resolution": "640x480",
    "fps": 30,
    "show_overlay": True,
    "show_face_mesh": False,
    "head_down_threshold": -22,
    "look_away_threshold": 30,
    "eye_look_down_threshold": 0.35,
    "eye_look_up_threshold": 0.30,
    "enable_phone_detection": True,
    "phone_detection_mode": "heuristic",
    "phone_confidence_threshold": 0.55,
    "phone_eye_down_min_duration": 45,
    "blink_rate_low_screen_max": 10.0,
    "blink_rate_high_fatigue_min": 22.0,
    "write_score_threshold": 0.4,
    "eye_closure_threshold": 0.3,
    "ear_threshold": 0.18,
    "enable_break_reminders": True,
    "break_interval_minutes": 25,
    "break_duration_minutes": 5,
    "profile_name": "default",
    "enable_personalization": True,
    "auto_apply_personalization": True,
    "auto_break_on_distraction": True,
    "distraction_break_cooldown_minutes": 15,
    "auto_resume_after_break": True,
    "enable_pomodoro": False,
    "pomodoro_work": 25,
    "pomodoro_short_break": 5,
    "pomodoro_long_break": 15,
    "enable_notifications": True,
    "notify_distraction": True,
    "notify_break": True,
    "notify_drowsy": True,
    "enable_sounds": True,
    "volume": 70,
    "sound_file": "",
    "theme_mode": "dark",
    "minimize_to_tray": True,
    "start_minimized": False,
}


def load_config() -> dict:
    """Load configuration from file."""
    config = DEFAULT_CONFIG.copy()

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                config.update(saved)
            logger.info("Configuration loaded from %s", CONFIG_FILE)
        except Exception as e:
            logger.warning("Failed to load config: %s", e)

    # Keep threshold in a sane range while preserving user preference.
    try:
        head_down = float(config.get("head_down_threshold", -22))
    except (TypeError, ValueError):
        head_down = -22.0
    config["head_down_threshold"] = max(-45.0, min(0.0, head_down))

    # Enforce single dark mode across the app.
    config["theme_mode"] = "dark"
    return config


def save_config(config: dict):
    """Save configuration to file."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info("Configuration saved to %s", CONFIG_FILE)
    except Exception as e:
        logger.error("Failed to save config: %s", e)


class FocusGuardianApp:
    """Main application class."""

    def __init__(self):
        # Create Qt application
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("FocusGuardian")
        self.app.setApplicationDisplayName("🎯 FocusGuardian")
        self.app.setOrganizationName("FocusGuardian")

        # Set default font
        font = QFont("Segoe UI", 10)
        self.app.setFont(font)

        # Load configuration
        self.config = load_config()

        # Apply configured theme palette
        self._apply_theme_palette(self.config.get("theme_mode", "dark"))

        # Initialize components
        self.main_window = None
        self.tray = None

        # Setup
        self._init_main_window()
        self._init_tray()
        self._connect_signals()

    def _apply_theme_palette(self, theme_mode: str):
        """Apply application palette (dark mode only)."""
        self.app.setStyle("Fusion")

        from PyQt6.QtGui import QPalette, QColor
        palette = QPalette()
        _ = theme_mode

        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(238, 238, 238))
        palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(45, 45, 45))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(238, 238, 238))
        palette.setColor(QPalette.ColorRole.Text, QColor(238, 238, 238))
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(238, 238, 238))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(76, 175, 80))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 175, 80))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))

        self.app.setPalette(palette)

    def _init_main_window(self):
        """Initialize main window."""
        from app.ui.main_window import MainWindow
        self.main_window = MainWindow(config=self.config)

        # Connect config save on close
        self.main_window.destroyed.connect(self._on_window_closed)

    def _init_tray(self):
        """Initialize system tray."""
        from app.ui.tray import SystemTray
        self.tray = SystemTray(parent=self.main_window)

        if not self.tray.is_available():
            logger.warning("System tray not available")

    def _connect_signals(self):
        """Connect signals between components."""
        # Main window signals
        self.main_window.state_changed.connect(self._on_state_changed)
        self.main_window.break_suggested.connect(self._on_break_suggested)
        self.main_window.config_changed.connect(self._on_config_changed)

        # Tray signals
        self.tray.show_window_requested.connect(self._show_window)
        self.tray.quit_requested.connect(self._quit)
        self.tray.start_tracking_requested.connect(self._start_tracking)
        self.tray.stop_tracking_requested.connect(self._stop_tracking)
        self.tray.take_break_requested.connect(self._take_break)
        self.tray.open_settings_requested.connect(self._open_settings)

    def _on_config_changed(self, config: dict):
        """Handle config updates emitted by the main window."""
        self.config.update(config)
        self.config["theme_mode"] = "dark"
        self._apply_theme_palette(self.config.get("theme_mode", "dark"))

    def _on_state_changed(self, state):
        """Handle focus state change."""
        from app.logic.focus_engine import FocusState

        self.tray.set_state(state)

        # Show notification for certain states
        if self.config.get("enable_notifications", True):
            if state == FocusState.PHONE_DISTRACTION and self.config.get("notify_distraction", True):
                self.tray.show_focus_notification(state)
            elif state == FocusState.DROWSY_FATIGUE and self.config.get("notify_drowsy", True):
                self.tray.show_focus_notification(state)

    def _on_break_suggested(self):
        """Handle break suggestion."""
        if self.config.get("enable_notifications", True) and self.config.get("notify_break", True):
            self.tray.show_break_reminder()

    def _show_window(self):
        """Show main window."""
        if self.main_window.isMinimized():
            self.main_window.setWindowState(
                self.main_window.windowState() & ~Qt.WindowState.WindowMinimized
            )
        self.main_window.show()
        self.main_window.activateWindow()
        self.main_window.raise_()

    def _start_tracking(self):
        """Start tracking from tray."""
        self.main_window.btn_start.setChecked(True)
        self.main_window._toggle_tracking()
        self.tray.set_tracking(True)

    def _stop_tracking(self):
        """Stop tracking from tray."""
        self.main_window.btn_start.setChecked(False)
        self.main_window._toggle_tracking()
        self.tray.set_tracking(False)

    def _take_break(self):
        """Take break from tray."""
        self.main_window._take_break()

    def _open_settings(self):
        """Open settings from tray."""
        self._show_window()
        self.main_window._open_settings()

    def _on_window_closed(self):
        """Handle window close."""
        # Save latest config from main window
        self.config.update(self.main_window.config)
        save_config(self.config)

    def _quit(self):
        """Quit application."""
        # Stop tracking if running
        if self.main_window.camera_running:
            self.main_window._stop_tracking()

        # Save latest config from main window
        self.config.update(self.main_window.config)
        save_config(self.config)

        # Hide tray
        self.tray.hide()

        # Quit app
        self.app.quit()

    def run(self) -> int:
        """Run the application."""
        logger.info("Starting FocusGuardian")

        try:
            # Show main window
            if not self.config.get("start_minimized", False):
                self.main_window.show()

            # Run event loop
            return self.app.exec()

        except Exception as e:
            logger.exception("Application error: %s", e)
            QMessageBox.critical(
                None,
                "Lỗi",
                f"Đã xảy ra lỗi không mong muốn:\n{e}"
            )
            return 1

        finally:
            logger.info("FocusGuardian stopped")


def main():
    """Main entry point."""
    # Check Python version
    if sys.version_info < (3, 10):
        print("FocusGuardian requires Python 3.10 or higher")
        sys.exit(1)

    # Create and run app
    app = FocusGuardianApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
