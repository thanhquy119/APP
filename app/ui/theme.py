def get_stylesheet(is_dark: bool) -> str:
    """Return shared stylesheet for main windows/dialogs."""
    _ = is_dark  # App now enforces a single dark theme.
    colors = {
        "bg": "#0b131d",
        "surface": "#131f2d",
        "surface_alt": "#101a27",
        "surface_soft": "#182637",
        "border": "#2a394b",
        "text": "#edf4fd",
        "muted": "#9baec5",
        "accent": "#59d5c0",
        "accent_hover": "#4abdaa",
        "accent_secondary": "#86a9ff",
        "warning": "#efbd78",
        "danger": "#ef9d95",
        "accent_text": "#07251f",
        "input": "#0f1927",
    }

    return f"""
        QMainWindow, QDialog {{
            background-color: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 #121d2c,
                stop: 0.45 #0d1623,
                stop: 1 #0a111a
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

        QLabel#sectionTitle {{
            font-size: 14px;
            font-weight: 650;
            color: #d9e5f5;
            letter-spacing: 0.3px;
        }}

        QLabel#mutedLabel {{
            color: {colors['muted']};
            font-size: 12px;
            line-height: 1.45;
        }}

        QLabel#stateBadge {{
            color: #dbe7f7;
            border: 1px solid rgba(136, 158, 183, 0.30);
            border-radius: 999px;
            background-color: rgba(108, 130, 154, 0.16);
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
            background-color: #162334;
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
            background-color: #101c2b;
            border-radius: 14px;
            border: 1px solid rgba(120, 142, 168, 0.20);
        }}

        QWidget#cameraEmptyState {{
            background-color: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 #182738,
                stop: 1 #121e2d
            );
            border-radius: 14px;
            border: 1px solid rgba(130, 156, 184, 0.22);
        }}

        QFrame#cameraEmptyIconRing {{
            background-color: rgba(134, 169, 255, 0.12);
            border: 1px solid rgba(134, 169, 255, 0.30);
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
            color: #e4edf9;
        }}

        QLabel#cameraEmptySubtitle {{
            font-size: 12px;
            color: {colors['muted']};
        }}

        QLabel#cameraEmptyHint {{
            font-size: 11px;
            color: #8ea2ba;
        }}

        QPushButton#cameraRetryButton {{
            background-color: rgba(134, 169, 255, 0.16);
            border: 1px solid rgba(134, 169, 255, 0.32);
            color: #dce8fb;
            border-radius: 11px;
            padding: 7px 14px;
            font-size: 12px;
        }}

        QPushButton#cameraRetryButton:hover {{
            background-color: rgba(134, 169, 255, 0.22);
            border-color: rgba(134, 169, 255, 0.42);
        }}

        QFrame#statusStrip {{
            background-color: #142436;
            border: 1px solid rgba(127, 150, 176, 0.20);
            border-radius: 12px;
        }}

        QFrame#statusChip {{
            background-color: rgba(127, 150, 176, 0.09);
            border: 1px solid rgba(127, 150, 176, 0.18);
            border-radius: 10px;
        }}

        QLabel#statusLabel {{
            color: #88a0ba;
            font-size: 10px;
            font-weight: 600;
        }}

        QLabel#statusValue {{
            color: #d8e6f7;
            font-size: 11px;
            font-weight: 700;
        }}

        QFrame#metricRow {{
            background-color: rgba(127, 150, 176, 0.08);
            border: 1px solid rgba(127, 150, 176, 0.15);
            border-radius: 10px;
            min-height: 38px;
        }}

        QFrame#metricDivider {{
            background-color: rgba(127, 150, 176, 0.20);
            border: none;
            min-height: 1px;
            max-height: 1px;
        }}

        QLabel#metricRowIcon {{
            font-size: 11px;
            color: #98aec8;
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
            color: #f4f8ff;
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
            background-color: rgba(89, 213, 192, 0.18);
            color: #caf8ef;
            border: 1px solid rgba(89, 213, 192, 0.30);
        }}

        QFrame#trendSparkline {{
            background-color: #121f30;
            border: 1px solid rgba(127, 150, 176, 0.20);
            border-radius: 12px;
        }}

        QPushButton {{
            background-color: #223449;
            color: {colors['text']};
            border: 1px solid rgba(123, 147, 174, 0.30);
            border-radius: 11px;
            padding: 10px 18px;
            font-weight: 650;
            font-size: 13px;
        }}

        QPushButton:hover {{
            background-color: #293f58;
            border-color: rgba(140, 170, 204, 0.44);
        }}

        QPushButton:pressed {{
            background-color: #1b2d42;
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
            color: #c8d8ed;
        }}

        QPushButton#iconButton {{
            background-color: #25384d;
            border: 1px solid rgba(135, 163, 193, 0.30);
            border-radius: 11px;
            padding: 0px;
            font-size: 12px;
            font-weight: 600;
            color: #d4e4f8;
        }}

        QPushButton#iconButton:hover {{
            background-color: #2b4561;
            border-color: rgba(151, 181, 214, 0.42);
        }}

        QPushButton#secondaryButton {{
            background-color: #2a3f58;
            border-color: rgba(140, 167, 198, 0.36);
            color: #d8e7fa;
        }}

        QPushButton#secondaryButton:hover {{
            background-color: #334c68;
            border-color: rgba(159, 189, 224, 0.45);
        }}

        QPushButton:checked {{
            background-color: {colors['text']};
            color: {colors['bg']};
            border-color: {colors['text']};
        }}

        QGroupBox {{
            border: 1px solid #2a3b4f;
            border-radius: 10px;
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
            background-color: #162334;
            border: 1px solid #2f465e;
            border-radius: 8px;
            text-align: center;
            color: {colors['text']};
        }}

        QProgressBar::chunk {{
            background-color: {colors['accent']};
            border-radius: 7px;
        }}

        QProgressBar#cycleProgress {{
            background-color: #1a2a3f;
            border: 1px solid #34506f;
            border-radius: 6px;
        }}

        QProgressBar#cycleProgress::chunk {{
            background-color: #64d8bf;
            border-radius: 5px;
        }}

        QDialog#breakOverlay {{
            background: transparent;
        }}

        QFrame#breakOverlayDim {{
            background-color: rgba(4, 8, 14, 185);
        }}

        QFrame#breakOverlayCard {{
            background-color: #182536;
            border: 1px solid #3a516a;
            border-radius: 16px;
        }}

        QLabel#breakPhaseText {{
            font-size: 18px;
            font-weight: 700;
            color: #cde6ff;
        }}

        QLabel#breakCountdownText {{
            font-size: 28px;
            font-weight: 800;
            color: {colors['accent_secondary']};
            letter-spacing: 1px;
        }}
    """
