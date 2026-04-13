#!/usr/bin/env python3
"""
UI Test - Test UI components without vision modules.

Run this to verify PyQt6 UI works correctly.
"""

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtCore import QTimer


def test_mini_games():
    """Test mini games widget."""
    from app.ui.mini_games import MiniGamesWidget, BreatherGame, MemoryGame, TypingGame

    print("Testing MiniGamesWidget...")
    games = MiniGamesWidget()
    games.show()
    return games


def test_settings_dialog():
    """Test settings dialog."""
    from app.ui.settings_dialog import SettingsDialog

    print("Testing SettingsDialog...")
    dialog = SettingsDialog()
    dialog.show()
    return dialog


def main():
    app = QApplication(sys.argv)

    # Apply dark theme
    app.setStyle("Fusion")

    print("="*50)
    print("FocusGuardian UI Test")
    print("="*50)
    print()
    print("This test runs UI components without vision modules.")
    print("MediaPipe is not compatible with Python 3.13.")
    print("Use Python 3.10 or 3.11 for full functionality.")
    print()

    # Test mini games
    try:
        games_window = test_mini_games()
        print("✅ MiniGamesWidget loaded successfully!")
    except Exception as e:
        print(f"❌ MiniGamesWidget failed: {e}")
        games_window = None

    # Test settings (opens in a separate window)
    try:
        from app.ui.settings_dialog import SettingsDialog
        # Just test import, don't show yet
        print("✅ SettingsDialog module loaded successfully!")
    except Exception as e:
        print(f"❌ SettingsDialog failed: {e}")

    # Test tray
    try:
        from app.ui.tray import SystemTray, create_colored_icon
        icon = create_colored_icon("#4CAF50")
        print("✅ SystemTray module loaded successfully!")
    except Exception as e:
        print(f"❌ SystemTray failed: {e}")

    print()
    print("="*50)
    print("Close the Mini Games window to exit.")
    print("="*50)

    if games_window:
        sys.exit(app.exec())
    else:
        print("No UI to display.")
        sys.exit(1)


if __name__ == "__main__":
    main()
