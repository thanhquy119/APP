"""Shared Qt stylesheet for FocusGuardian UI."""

from __future__ import annotations


def _theme_tokens(is_dark: bool) -> dict[str, str]:
    if is_dark:
        return {
            "bg_g1": "#121d2c",
            "bg_g2": "#0d1623",
            "bg_g3": "#0a111a",
            "bg": "#0b131d",
            "surface": "#131f2d",
            "surface_alt": "#101a27",
            "surface_soft": "#182637",
            "border": "#2a394b",
            "text": "#edf4fd",
            "muted": "#9baec5",
            "section_title": "#d9e5f5",
            "header_bg_g1": "#152437",
            "header_bg_g2": "#112133",
            "header_border": "rgba(126, 154, 184, 0.24)",
            "header_title": "#eaf3ff",
            "header_subtitle": "#9eb4cb",
            "titlebar_dot_close": "#ff5f57",
            "titlebar_dot_close_hover": "#ff736d",
            "titlebar_dot_close_pressed": "#e14f49",
            "titlebar_dot_min": "#febc2e",
            "titlebar_dot_min_hover": "#ffca4c",
            "titlebar_dot_min_pressed": "#dea225",
            "titlebar_dot_max": "#28c840",
            "titlebar_dot_max_hover": "#42d95a",
            "titlebar_dot_max_pressed": "#1faa36",
            "accent": "#59d5c0",
            "accent_hover": "#4abdaa",
            "accent_secondary": "#86a9ff",
            "warning": "#efbd78",
            "danger": "#ef9d95",
            "accent_text": "#07251f",
            "input": "#0f1927",
            "state_badge_text": "#dbe7f7",
            "state_badge_border": "rgba(136, 158, 183, 0.30)",
            "state_badge_bg": "rgba(108, 130, 154, 0.16)",
            "summary_bg": "#162334",
            "camera_bg": "#101c2b",
            "camera_empty_g1": "#182738",
            "camera_empty_g2": "#121e2d",
            "camera_icon_ring_bg": "rgba(134, 169, 255, 0.12)",
            "camera_icon_ring_border": "rgba(134, 169, 255, 0.30)",
            "camera_empty_title": "#e4edf9",
            "camera_empty_hint": "#8ea2ba",
            "retry_btn_bg": "rgba(134, 169, 255, 0.16)",
            "retry_btn_border": "rgba(134, 169, 255, 0.32)",
            "retry_btn_text": "#dce8fb",
            "status_strip_bg": "#142436",
            "status_chip_bg": "rgba(127, 150, 176, 0.09)",
            "status_label": "#88a0ba",
            "status_value": "#d8e6f7",
            "metric_row_bg": "rgba(127, 150, 176, 0.08)",
            "metric_row_border": "rgba(127, 150, 176, 0.15)",
            "metric_divider": "rgba(127, 150, 176, 0.20)",
            "metric_row_icon": "#98aec8",
            "metric_row_value": "#f4f8ff",
            "coach_badge_bg": "rgba(89, 213, 192, 0.18)",
            "coach_badge_border": "rgba(89, 213, 192, 0.30)",
            "coach_badge_text": "#caf8ef",
            "sparkline_bg": "#121f30",
            "button_bg": "#223449",
            "button_hover": "#293f58",
            "button_pressed": "#1b2d42",
            "ghost_text": "#c8d8ed",
            "icon_btn_bg": "#25384d",
            "icon_btn_hover": "#2b4561",
            "icon_btn_text": "#d4e4f8",
            "secondary_btn_bg": "#2a3f58",
            "secondary_btn_hover": "#334c68",
            "group_border": "#2a3b4f",
            "progress_bg": "#162334",
            "progress_border": "#2f465e",
            "cycle_bg": "#1a2a3f",
            "cycle_border": "#34506f",
            "break_dim": "rgba(4, 8, 14, 185)",
            "break_card": "#182536",
            "break_card_border": "#3a516a",
            "break_phase_text": "#cde6ff",
        }

    return {
        "bg_g1": "#f3f9ff",
        "bg_g2": "#edf5fd",
        "bg_g3": "#f8fcff",
        "bg": "#f2f8fe",
        "surface": "#ffffff",
        "surface_alt": "#f5f9ff",
        "surface_soft": "#edf4fc",
        "border": "#c5d6e8",
        "text": "#182c41",
        "muted": "#435d76",
        "section_title": "#1f3a53",
        "header_bg_g1": "#f7fbff",
        "header_bg_g2": "#edf5fd",
        "header_border": "rgba(118, 149, 183, 0.30)",
        "header_title": "#1e3a53",
        "header_subtitle": "#5d758e",
        "titlebar_dot_close": "#ff5f57",
        "titlebar_dot_close_hover": "#ff736d",
        "titlebar_dot_close_pressed": "#e14f49",
        "titlebar_dot_min": "#febc2e",
        "titlebar_dot_min_hover": "#ffd159",
        "titlebar_dot_min_pressed": "#dea225",
        "titlebar_dot_max": "#2fca46",
        "titlebar_dot_max_hover": "#4ddc63",
        "titlebar_dot_max_pressed": "#24ab39",
        "accent": "#2f9f90",
        "accent_hover": "#238f81",
        "accent_secondary": "#3f6fb5",
        "warning": "#b9792f",
        "danger": "#b9524d",
        "accent_text": "#ffffff",
        "input": "#ffffff",
        "state_badge_text": "#284663",
        "state_badge_border": "rgba(93, 125, 161, 0.30)",
        "state_badge_bg": "rgba(122, 156, 194, 0.13)",
        "summary_bg": "#f6faff",
        "camera_bg": "#f6fbff",
        "camera_empty_g1": "#f0f6fd",
        "camera_empty_g2": "#e7f0fb",
        "camera_icon_ring_bg": "rgba(63, 111, 181, 0.12)",
        "camera_icon_ring_border": "rgba(63, 111, 181, 0.28)",
        "camera_empty_title": "#2a435e",
        "camera_empty_hint": "#68819a",
        "retry_btn_bg": "rgba(63, 111, 181, 0.12)",
        "retry_btn_border": "rgba(63, 111, 181, 0.28)",
        "retry_btn_text": "#2e567f",
        "status_strip_bg": "#f2f8ff",
        "status_chip_bg": "rgba(106, 137, 170, 0.12)",
        "status_label": "#4f6780",
        "status_value": "#1f3a55",
        "metric_row_bg": "rgba(104, 136, 170, 0.10)",
        "metric_row_border": "rgba(104, 136, 170, 0.22)",
        "metric_divider": "rgba(104, 136, 170, 0.24)",
        "metric_row_icon": "#4f6a85",
        "metric_row_value": "#182f45",
        "coach_badge_bg": "rgba(47, 159, 144, 0.15)",
        "coach_badge_border": "rgba(47, 159, 144, 0.32)",
        "coach_badge_text": "#1c6d62",
        "sparkline_bg": "#edf5fe",
        "button_bg": "#e6f0fb",
        "button_hover": "#d8e7f8",
        "button_pressed": "#cddff3",
        "ghost_text": "#2f4d68",
        "icon_btn_bg": "#e8f1fb",
        "icon_btn_hover": "#d9e8f8",
        "icon_btn_text": "#294a67",
        "secondary_btn_bg": "#dfebf8",
        "secondary_btn_hover": "#d1e3f6",
        "group_border": "#c9d8e8",
        "progress_bg": "#e7f1fb",
        "progress_border": "#c3d5e9",
        "cycle_bg": "#e7f1fc",
        "cycle_border": "#b9cee5",
        "break_dim": "rgba(133, 151, 174, 126)",
        "break_card": "#ffffff",
        "break_card_border": "#bfd0e2",
        "break_phase_text": "#2d4c69",
    }


def get_stylesheet(is_dark: bool) -> str:
    """Return shared stylesheet for main windows/dialogs."""
    colors = _theme_tokens(bool(is_dark))

    return f"""
        QMainWindow, QDialog {{
            background-color: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 {colors['bg_g1']},
                stop: 0.45 {colors['bg_g2']},
                stop: 1 {colors['bg_g3']}
            );
        }}

        QWidget#appRoot {{
            background-color: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 {colors['bg_g1']},
                stop: 0.45 {colors['bg_g2']},
                stop: 1 {colors['bg_g3']}
            );
        }}

        QWidget {{
            color: {colors['text']};
            background-color: transparent;
            font-family: 'Segoe UI Variable Text', 'Segoe UI', sans-serif;
        }}

        QLabel {{
            color: {colors['text']};
            background: transparent;
        }}

        QLabel#heroTitle {{
            font-size: 26px;
            font-weight: 700;
            letter-spacing: 0.4px;
        }}

        QLabel#heroSubtitle {{
            color: {colors['muted']};
            font-size: 13px;
            font-weight: 500;
            line-height: 1.45;
        }}

        QFrame#topHeaderBar {{
            background-color: transparent;
            border: none;
            border-radius: 12px;
        }}

        QFrame#topHeaderBar[maximized="true"] {{
            border-radius: 0px;
        }}

        QLabel#topHeaderTitle {{
            font-size: 15px;
            font-weight: 680;
            color: {colors['header_title']};
            letter-spacing: 0.2px;
        }}

        QWidget#titleBarDotsHost {{
            background: transparent;
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
            background-color: {colors['titlebar_dot_close']};
        }}

        QToolButton#titleBarCloseDot:hover {{
            background-color: {colors['titlebar_dot_close_hover']};
        }}

        QToolButton#titleBarCloseDot:pressed {{
            background-color: {colors['titlebar_dot_close_pressed']};
        }}

        QToolButton#titleBarMinDot {{
            background-color: {colors['titlebar_dot_min']};
        }}

        QToolButton#titleBarMinDot:hover {{
            background-color: {colors['titlebar_dot_min_hover']};
        }}

        QToolButton#titleBarMinDot:pressed {{
            background-color: {colors['titlebar_dot_min_pressed']};
        }}

        QToolButton#titleBarMaxDot {{
            background-color: {colors['titlebar_dot_max']};
        }}

        QToolButton#titleBarMaxDot:hover {{
            background-color: {colors['titlebar_dot_max_hover']};
        }}

        QToolButton#titleBarMaxDot:pressed {{
            background-color: {colors['titlebar_dot_max_pressed']};
        }}

        QLabel#topHeaderSubtitle {{
            font-size: 12px;
            font-weight: 500;
            color: {colors['header_subtitle']};
        }}

        QLabel#sectionTitle {{
            font-size: 14px;
            font-weight: 650;
            color: {colors['section_title']};
            letter-spacing: 0.3px;
        }}

        QLabel#mutedLabel {{
            color: {colors['muted']};
            font-size: 12px;
            line-height: 1.45;
        }}

        QLabel#stateBadge {{
            color: {colors['state_badge_text']};
            border: 1px solid {colors['state_badge_border']};
            border-radius: 999px;
            background-color: {colors['state_badge_bg']};
            padding: 5px 12px;
            font-weight: 650;
        }}

        QFrame#panel,
        QFrame#cameraCard {{
            background-color: {colors['surface']};
            border-radius: 16px;
            border: 1px solid rgba(128, 150, 176, 0.20);
        }}

        QFrame[summaryCard="true"] {{
            background-color: {colors['summary_bg']};
            border-radius: 16px;
            border: 1px solid rgba(128, 150, 176, 0.18);
        }}

        QScrollArea#rightColumnScroll {{
            background: transparent;
            border: none;
        }}

        QScrollArea#rightColumnScroll > QWidget > QWidget {{
            background: transparent;
        }}

        QScrollArea#rightColumnScroll QWidget#qt_scrollarea_viewport {{
            background: transparent;
        }}

        QScrollBar:vertical {{
            background: transparent;
            width: 8px;
            margin: 2px;
        }}

        QScrollBar::handle:vertical {{
            background: rgba(134, 159, 187, 0.34);
            border-radius: 4px;
            min-height: 22px;
        }}

        QScrollBar::handle:vertical:hover {{
            background: rgba(150, 178, 209, 0.46);
        }}

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical,
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {{
            background: transparent;
            border: none;
            height: 0px;
        }}

        QFrame#cameraFrame {{
            background-color: {colors['camera_bg']};
            border-radius: 14px;
            border: 1px solid rgba(120, 142, 168, 0.20);
        }}

        QWidget#cameraEmptyState {{
            background-color: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 {colors['camera_empty_g1']},
                stop: 1 {colors['camera_empty_g2']}
            );
            border-radius: 14px;
            border: 1px solid rgba(130, 156, 184, 0.22);
        }}

        QFrame#cameraEmptyIconRing {{
            background-color: {colors['camera_icon_ring_bg']};
            border: 1px solid {colors['camera_icon_ring_border']};
            border-radius: 28px;
            min-width: 56px;
            min-height: 56px;
            max-width: 56px;
            max-height: 56px;
        }}

        QLabel#cameraEmptyIcon {{
            font-size: 30px;
            color: {colors['accent_secondary']};
        }}

        QLabel#cameraEmptyTitle {{
            font-size: 16px;
            font-weight: 700;
            color: {colors['camera_empty_title']};
        }}

        QLabel#cameraEmptySubtitle {{
            font-size: 12px;
            color: {colors['muted']};
        }}

        QLabel#cameraEmptyHint {{
            font-size: 11px;
            color: {colors['camera_empty_hint']};
        }}

        QPushButton#cameraRetryButton {{
            background-color: {colors['retry_btn_bg']};
            border: 1px solid {colors['retry_btn_border']};
            color: {colors['retry_btn_text']};
            border-radius: 11px;
            padding: 7px 14px;
            font-size: 12px;
        }}

        QPushButton#cameraRetryButton:hover {{
            background-color: rgba(134, 169, 255, 0.22);
            border-color: rgba(134, 169, 255, 0.42);
        }}

        QFrame#statusStrip {{
            background-color: {colors['status_strip_bg']};
            border: 1px solid rgba(127, 150, 176, 0.20);
            border-radius: 12px;
        }}

        QFrame#statusChip {{
            background-color: {colors['status_chip_bg']};
            border: 1px solid rgba(127, 150, 176, 0.18);
            border-radius: 10px;
        }}

        QLabel#statusLabel {{
            color: {colors['status_label']};
            font-size: 10px;
            font-weight: 600;
        }}

        QLabel#statusValue {{
            color: {colors['status_value']};
            font-size: 11px;
            font-weight: 700;
        }}

        QFrame#metricRow {{
            background-color: {colors['metric_row_bg']};
            border: 1px solid {colors['metric_row_border']};
            border-radius: 10px;
            min-height: 38px;
        }}

        QFrame#metricDivider {{
            background-color: {colors['metric_divider']};
            border: none;
            min-height: 1px;
            max-height: 1px;
        }}

        QLabel#metricRowIcon {{
            font-size: 11px;
            color: {colors['metric_row_icon']};
            font-weight: 700;
        }}

        QLabel#metricRowLabel {{
            font-size: 11px;
            color: {colors['muted']};
            font-weight: 500;
        }}

        QLabel#metricRowValue {{
            font-size: 16px;
            font-weight: 650;
            color: {colors['metric_row_value']};
            min-width: 88px;
        }}

        QLabel#trendValue {{
            font-size: 12px;
            font-weight: 700;
            color: {colors['accent_secondary']};
        }}

        QLabel#coachBadge {{
            border-radius: 999px;
            padding: 8px 12px;
            font-weight: 650;
            background-color: {colors['coach_badge_bg']};
            color: {colors['coach_badge_text']};
            border: 1px solid {colors['coach_badge_border']};
        }}

        QFrame#trendSparkline {{
            background-color: {colors['sparkline_bg']};
            border: 1px solid rgba(127, 150, 176, 0.20);
            border-radius: 12px;
        }}

        QPushButton {{
            background-color: {colors['button_bg']};
            color: {colors['text']};
            border: 1px solid rgba(123, 147, 174, 0.30);
            border-radius: 11px;
            padding: 10px 18px;
            font-weight: 650;
            font-size: 13px;
        }}

        QPushButton:hover {{
            background-color: {colors['button_hover']};
            border-color: rgba(140, 170, 204, 0.44);
        }}

        QPushButton:pressed {{
            background-color: {colors['button_pressed']};
        }}

        QPushButton:disabled {{
            color: {colors['muted']};
            border-color: {colors['border']};
            background-color: {colors['surface_alt']};
        }}

        QPushButton#primaryButton {{
            background-color: {colors['accent']};
            border-color: rgba(89, 213, 192, 0.92);
            color: {colors['accent_text']};
            font-weight: 700;
        }}

        QPushButton#primaryButton:hover {{
            background-color: {colors['accent_hover']};
            border-color: rgba(74, 189, 170, 0.92);
        }}

        QPushButton#ghostButton {{
            background-color: transparent;
            border-color: rgba(123, 147, 174, 0.26);
            color: {colors['ghost_text']};
        }}

        QPushButton#iconButton {{
            background-color: {colors['icon_btn_bg']};
            border: 1px solid rgba(135, 163, 193, 0.30);
            border-radius: 11px;
            padding: 0px;
            font-size: 12px;
            font-weight: 600;
            color: {colors['icon_btn_text']};
        }}

        QPushButton#iconButton:hover {{
            background-color: {colors['icon_btn_hover']};
            border-color: rgba(151, 181, 214, 0.42);
        }}

        QPushButton#secondaryButton {{
            background-color: {colors['secondary_btn_bg']};
            border-color: rgba(140, 167, 198, 0.36);
            color: {colors['text']};
        }}

        QPushButton#secondaryButton:hover {{
            background-color: {colors['secondary_btn_hover']};
            border-color: rgba(159, 189, 224, 0.45);
        }}

        QPushButton:checked {{
            background-color: {colors['text']};
            color: {colors['bg']};
            border-color: {colors['text']};
        }}

        QGroupBox {{
            border: 1px solid {colors['group_border']};
            border-radius: 8px;
            margin-top: 14px;
            padding: 14px;
            font-weight: 600;
            background-color: {colors['surface']};
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 2px 6px;
            color: {colors['muted']};
            background-color: {colors['surface']};
        }}

        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {colors['input']};
            border: 1px solid {colors['border']};
            border-radius: 8px;
            padding: 6px 10px;
            min-height: 18px;
        }}

        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border-color: {colors['accent']};
        }}

        QTabWidget::pane {{
            border: 1px solid {colors['border']};
            border-radius: 10px;
            background-color: {colors['surface']};
        }}

        QTabBar::tab {{
            background-color: {colors['surface_alt']};
            color: {colors['muted']};
            border: 1px solid {colors['border']};
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 8px 14px;
            margin-right: 3px;
            font-weight: 600;
        }}

        QTabBar::tab:selected {{
            color: {colors['text']};
            background-color: {colors['surface']};
        }}

        QCheckBox {{
            spacing: 8px;
        }}

        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid {colors['border']};
            background: {colors['input']};
        }}

        QCheckBox::indicator:checked {{
            background: {colors['accent']};
            border-color: {colors['accent']};
        }}

        QSlider::groove:horizontal {{
            height: 6px;
            background: {colors['border']};
            border-radius: 3px;
        }}

        QSlider::handle:horizontal {{
            width: 16px;
            height: 16px;
            border-radius: 8px;
            margin: -5px 0;
            background: {colors['accent']};
            border: 1px solid {colors['accent']};
        }}

        QProgressBar {{
            background-color: {colors['progress_bg']};
            border: 1px solid {colors['progress_border']};
            border-radius: 8px;
            text-align: center;
            color: {colors['text']};
        }}

        QProgressBar::chunk {{
            background-color: {colors['accent']};
            border-radius: 7px;
        }}

        QProgressBar#cycleProgress {{
            background-color: {colors['cycle_bg']};
            border: 1px solid {colors['cycle_border']};
            border-radius: 6px;
        }}

        QProgressBar#cycleProgress::chunk {{
            background-color: {colors['accent']};
            border-radius: 5px;
        }}

        QDialog#breakOverlay {{
            background: transparent;
        }}

        QFrame#breakOverlayDim {{
            background-color: {colors['break_dim']};
        }}

        QFrame#breakOverlayCard {{
            background-color: {colors['break_card']};
            border: 1px solid {colors['break_card_border']};
            border-radius: 16px;
        }}

        QLabel#breakPhaseText {{
            font-size: 18px;
            font-weight: 700;
            color: {colors['break_phase_text']};
        }}

        QLabel#breakCountdownText {{
            font-size: 28px;
            font-weight: 800;
            color: {colors['accent_secondary']};
            letter-spacing: 1px;
        }}
    """
