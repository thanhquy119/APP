"""Context-aware session dialogs for start/check-in/exit flows."""

from __future__ import annotations

import time
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont, QPainter, QPainterPath, QPen, QLinearGradient
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QLayout,
)

from .theme import get_stylesheet, _theme_tokens
from .dialog_title_bar import DialogTitleBar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_dark_mode(config: Optional[dict] = None) -> bool:
    mode = str((config or {}).get("theme_mode", "light") or "light").strip().lower()
    return mode != "light"


def _make_dialog_stylesheet(is_dark: bool) -> str:
    t = _theme_tokens(is_dark)

    if is_dark:
        dialog_bg        = "#111c2b"
        dialog_border    = "#283a50"
        header_bg        = "#152236"
        header_border    = "rgba(126,154,184,0.18)"
        title_color      = "#eaf3ff"
        subtitle_color   = "#8daac4"
        card_bg          = "rgba(14,24,38,0.55)"
        card_border      = "rgba(60,90,120,0.30)"
        row_label_color  = "#8daac4"
        input_bg         = "rgba(10,18,30,0.80)"
        input_border     = "#2e4460"
        input_focus      = "#59d5c0"
        input_text       = "#eaf3ff"
        combo_drop_bg    = "#0f1e2f"
        combo_drop_bdr   = "#2e4460"
        item_hover       = "rgba(89,213,192,0.12)"
        item_sel         = "rgba(89,213,192,0.22)"
        check_indicator  = "#0f1927"
        hint_color       = "#607a94"
        sep_color        = "rgba(80,112,148,0.22)"
    else:
        dialog_bg        = "#f4f9ff"
        dialog_border    = "#c4d5e7"
        header_bg        = "#edf5fd"
        header_border    = "rgba(118,149,183,0.22)"
        title_color      = "#1a3349"
        subtitle_color   = "#4d6880"
        card_bg          = "rgba(255,255,255,0.85)"
        card_border      = "rgba(140,175,210,0.28)"
        row_label_color  = "#4d6880"
        input_bg         = "#ffffff"
        input_border     = "#b8cfe3"
        input_focus      = "#2f9f90"
        input_text       = "#182c41"
        combo_drop_bg    = "#f8fcff"
        combo_drop_bdr   = "#b8cfe3"
        item_hover       = "rgba(47,159,144,0.10)"
        item_sel         = "rgba(47,159,144,0.18)"
        check_indicator  = "#ffffff"
        hint_color       = "#748fa8"
        sep_color        = "rgba(140,175,210,0.25)"

    accent        = t["accent"]
    accent_hover  = t["accent_hover"]
    accent_text   = t["accent_text"]
    ghost_text    = t["ghost_text"]

    return f"""
        QDialog {{
            background: transparent;
            border: none;
        }}

        QFrame#dialogContainer {{
            background-color: {dialog_bg};
            border: 1px solid {dialog_border};
            border-radius: 14px;
        }}

        QWidget {{ background: transparent; color: {input_text}; }}
        QLabel  {{ background: transparent; color: {input_text}; }}

        /* ── Header ── */
        QFrame#dialogHeader {{
            background-color: {header_bg};
            border-bottom: 1px solid {header_border};
            border-top-left-radius: 13px;
            border-top-right-radius: 13px;
        }}
        QLabel#dialogTitle {{
            color: {title_color};
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0.2px;
        }}
        QLabel#dialogSubtitle {{
            color: {subtitle_color};
            font-size: 11px;
            font-weight: 400;
            line-height: 1.4;
        }}

        /* ── Form card ── */
        QFrame#formCard {{
            background-color: {card_bg};
            border: 1px solid {card_border};
            border-radius: 10px;
        }}
        QLabel#rowLabel {{
            color: {row_label_color};
            font-size: 12px;
            font-weight: 550;
            min-width: 148px;
        }}
        /* ── Inputs ── */
        QLineEdit, QSpinBox, QComboBox {{
            background-color: {input_bg};
            border: 1px solid {input_border};
            border-radius: 8px;
            color: {input_text};
            font-size: 13px;
            padding: 0px 10px;
            min-height: 36px;
        }}
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
            border-color: {input_focus};
        }}
        QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{
            color: {hint_color};
            background-color: {card_bg};
            border-color: {sep_color};
        }}

        /* hide spinbox arrows */
        QSpinBox::up-button, QSpinBox::down-button,
        QSpinBox::up-arrow, QSpinBox::down-arrow {{
            width: 0; height: 0; border: none; background: transparent;
        }}

        /* ── ComboBox dropdown ── */
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: center right;
            width: 28px;
            border: none;
            background: transparent;
        }}
        QComboBox::down-arrow {{ image: none; width: 0; height: 0; }}
        QComboBox QAbstractItemView {{
            background: {combo_drop_bg};
            border: 1px solid {combo_drop_bdr};
            border-radius: 8px;
            color: {input_text};
            selection-background-color: {item_sel};
            outline: 0;
        }}
        QComboBox QAbstractItemView::item {{
            min-height: 30px;
            padding: 4px 10px;
            border-radius: 6px;
        }}
        QComboBox QAbstractItemView::item:hover {{ background: {item_hover}; }}
        QComboBox QAbstractItemView::item:selected {{ background: {item_sel}; }}

        /* ── Checkbox ── */
        QCheckBox {{ spacing: 8px; font-size: 13px; color: {input_text}; }}
        QCheckBox::indicator {{
            width: 17px; height: 17px;
            border-radius: 5px;
            border: 1px solid {input_border};
            background: {check_indicator};
        }}
        QCheckBox::indicator:checked {{
            background: {accent};
            border-color: {accent};
        }}
        QCheckBox::indicator:disabled {{
            background: {card_bg};
            border-color: {sep_color};
        }}

        /* ── Buttons ── */
        QPushButton#primaryButton {{
            background-color: {accent};
            border: 1px solid {accent};
            border-radius: 10px;
            color: {accent_text};
            font-size: 13px;
            font-weight: 700;
            padding: 0px 22px;
            min-height: 40px;
        }}
        QPushButton#primaryButton:hover {{
            background-color: {accent_hover};
            border-color: {accent_hover};
        }}
        QPushButton#primaryButton:pressed {{
            background-color: {accent_hover};
        }}
        QPushButton#ghostButton {{
            background-color: transparent;
            border: 1px solid {card_border};
            border-radius: 10px;
            color: {ghost_text};
            font-size: 13px;
            font-weight: 600;
            padding: 0px 18px;
            min-height: 40px;
        }}
        QPushButton#ghostButton:hover {{
            border-color: {input_border};
            background-color: {item_hover};
        }}
        QPushButton#ghostButton:pressed {{
            background-color: {item_sel};
        }}
        QPushButton#ghostButton:checked {{
            border: 2px solid {input_focus};
            background-color: {item_sel};
            color: {input_text};
        }}

        /* ── Close X button ── */
        QPushButton#optionPill {{
            text-align: left;
            background-color: {input_bg};
            border: 1px solid {input_border};
            border-radius: 9px;
            color: {input_text};
            font-size: 13px;
            font-weight: 650;
            padding: 0px 12px;
            min-height: 38px;
        }}
        QPushButton#optionPill:hover {{
            background-color: {item_hover};
            border-color: {input_focus};
        }}
        QPushButton#optionPill:checked {{
            background-color: {item_sel};
            border: 2px solid {input_focus};
            color: {input_text};
        }}
        QPushButton#ratingPill {{
            text-align: center;
            background-color: {input_bg};
            border: 1px solid {input_border};
            border-radius: 9px;
            color: {input_text};
            font-size: 13px;
            font-weight: 700;
            padding: 6px 8px;
            min-height: 48px;
        }}
        QPushButton#ratingPill:hover {{
            background-color: {item_hover};
            border-color: {input_focus};
        }}
        QPushButton#ratingPill:checked {{
            background-color: {item_sel};
            border: 2px solid {input_focus};
            color: {input_text};
        }}

        QPushButton#routeCard {{
            text-align: center;
            background-color: transparent;
            border: none;
            border-radius: 14px;
            padding: 8px 8px;
            min-height: 118px;
            max-height: 132px;
            color: {input_text};
            font-size: 12px;
            font-weight: 600;
            outline: none;
        }}
        QPushButton#routeCard:hover {{
            background-color: transparent;
        }}
        QPushButton#routeCard:checked {{
            background-color: transparent;
        }}
        QPushButton#routeCard:focus {{
            outline: none;
        }}

        QPushButton#closeXButton {{
            background: transparent;
            border: none;
            color: {hint_color};
            font-size: 16px;
            font-weight: 600;
            min-width: 28px;
            max-width: 28px;
            min-height: 28px;
            max-height: 28px;
            border-radius: 6px;
            padding: 0;
        }}
        QPushButton#closeXButton:hover {{
            background-color: {item_hover};
            color: {input_text};
        }}
        QPushButton#checkInButton {{
            background-color: #ffffff;
            color: #08111c;
            border: 1px solid rgba(130,160,190,0.28);
            border-radius: 22px;
            min-height: 46px;
            font-size: 15px;
            font-weight: 700;
            padding: 0px 22px;
        }}
        QPushButton#checkInButton:hover {{
            background-color: #f1fbff;
            border-color: {input_focus};
        }}
        QPushButton#checkInButton:pressed {{
            background-color: #dff7ff;
        }}
        QPushButton#checkInButton:disabled {{
            background-color: rgba(255,255,255,0.45);
            color: rgba(8,17,28,0.55);
        }}

        /* ── Report title bar (custom chrome) ── */
        QFrame#reportTitleBar {{
            background-color: {header_bg};
            border-bottom: 1px solid {header_border};
            border-top-left-radius: 13px;
            border-top-right-radius: 13px;
        }}
        QLabel#reportTitleText {{
            color: {title_color};
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
    """


FOCUS_AIRPORT_DATA: Dict[str, Dict[str, Any]] = {
    "DAD": {"name": "Da Nang", "lat": 16.0439, "lon": 108.1994},
    "SGN": {"name": "Ho Chi Minh City", "lat": 10.8188, "lon": 106.6519},
    "HAN": {"name": "Ha Noi", "lat": 21.2212, "lon": 105.8072},
    "HUI": {"name": "Hue", "lat": 16.4015, "lon": 107.7031},
    "DLI": {"name": "Da Lat", "lat": 11.7500, "lon": 108.3670},
    "CXR": {"name": "Cam Ranh", "lat": 12.2275, "lon": 109.1922},
    "VCA": {"name": "Can Tho", "lat": 10.0851, "lon": 105.7119},
    "PQC": {"name": "Phu Quoc", "lat": 10.1698, "lon": 103.9931},
    "BMV": {"name": "Buon Ma Thuot", "lat": 12.6683, "lon": 108.1203},
    "VII": {"name": "Vinh", "lat": 18.7376, "lon": 105.6708},
    "VCL": {"name": "Chu Lai", "lat": 15.4033, "lon": 108.7060},
    "BKK": {"name": "Bangkok", "lat": 13.6900, "lon": 100.7501},
    "SIN": {"name": "Singapore", "lat": 1.3644, "lon": 103.9915},
    "KUL": {"name": "Kuala Lumpur", "lat": 2.7456, "lon": 101.7072},
    "PNH": {"name": "Phnom Penh", "lat": 11.5466, "lon": 104.8441},
    "VTE": {"name": "Vientiane", "lat": 17.9883, "lon": 102.5633},
    "REP": {"name": "Siem Reap", "lat": 13.4107, "lon": 103.8128},
}


def _airport_distance_km(from_code: str, to_code: str) -> int:
    a = FOCUS_AIRPORT_DATA.get(str(from_code).upper())
    b = FOCUS_AIRPORT_DATA.get(str(to_code).upper())
    if not a or not b:
        return 0
    lat1, lon1 = math.radians(float(a["lat"])), math.radians(float(a["lon"]))
    lat2, lon2 = math.radians(float(b["lat"])), math.radians(float(b["lon"]))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return int(round(6371.0 * 2.0 * math.asin(math.sqrt(h))))


def _round_minutes_to_five(value: int, *, minimum: int = 30, maximum: int = 120) -> int:
    """Round displayed flight duration to the nearest 5-minute mark."""
    try:
        raw = int(float(value))
    except (TypeError, ValueError):
        raw = minimum
    raw = max(minimum, min(maximum, raw))
    rounded = int(math.floor((raw + 2.5) / 5.0) * 5)
    return max(minimum, min(maximum, rounded))


def _focus_route(
    route_id: str,
    from_code: str,
    to_code: str,
    duration_minutes: int,
    task_type: str,
    mode: str,
    route_theme: str,
    difficulty: str,
) -> Dict[str, Any]:
    from_data = FOCUS_AIRPORT_DATA[from_code]
    to_data = FOCUS_AIRPORT_DATA[to_code]
    rounded_minutes = _round_minutes_to_five(duration_minutes)
    return {
        "route_id": route_id,
        "from_code": from_code,
        "to_code": to_code,
        "from_name": str(from_data["name"]),
        "to_name": str(to_data["name"]),
        "duration_minutes": rounded_minutes,
        "raw_duration_minutes": int(duration_minutes),
        "route_distance_km": _airport_distance_km(from_code, to_code),
        "task_type": task_type,
        "mode": mode,
        "route_theme": route_theme,
        "difficulty": difficulty,
        "short_label": f"{from_code} -> {to_code}",
    }


FOCUS_ROUTE_PRESETS: List[Dict[str, Any]] = [
    _focus_route("dad-hui-32", "DAD", "HUI", 32, "reading", "normal", "Short Reading Hop", "short"),
    _focus_route("hui-dad-32", "HUI", "DAD", 32, "reading", "normal", "Return Reading Hop", "short"),
    _focus_route("sgn-pqc-34", "SGN", "PQC", 34, "study", "normal", "Island Sprint", "medium"),
    _focus_route("pqc-sgn-34", "PQC", "SGN", 34, "study", "normal", "Island Return", "medium"),
    _focus_route("sgn-cxr-36", "SGN", "CXR", 36, "deep_work", "normal", "Coastal Focus", "medium"),
    _focus_route("cxr-sgn-36", "CXR", "SGN", 36, "deep_work", "normal", "Coastal Return", "medium"),
    _focus_route("sin-kul-38", "SIN", "KUL", 38, "review", "normal", "Regional Review", "medium"),
    _focus_route("kul-sin-38", "KUL", "SIN", 38, "review", "normal", "Regional Hop", "medium"),
    _focus_route("dli-sgn-40", "DLI", "SGN", 40, "creative", "normal", "Highland Landing", "medium"),
    _focus_route("bkk-rep-42", "BKK", "REP", 42, "reading", "normal", "Temple Reading", "medium"),
    _focus_route("rep-bkk-42", "REP", "BKK", 42, "reading", "normal", "Regional Reading", "medium"),
    _focus_route("dad-bmv-43", "DAD", "BMV", 43, "creative", "normal", "Highland Draft", "medium"),
    _focus_route("bmv-dad-43", "BMV", "DAD", 43, "creative", "normal", "Highland Return", "medium"),
    _focus_route("vii-han-44", "VII", "HAN", 44, "reading", "normal", "Northern Reading", "medium"),
    _focus_route("vcl-dli-46", "VCL", "DLI", 46, "creative", "normal", "Highland Route", "medium"),
    _focus_route("dad-vii-46", "DAD", "VII", 46, "reading", "normal", "Reading Hop", "medium"),
    _focus_route("vii-dad-46", "VII", "DAD", 46, "reading", "normal", "Coastal Reading", "medium"),
    _focus_route("pnh-rep-46", "PNH", "REP", 46, "study", "normal", "Study Hop", "medium"),
    _focus_route("rep-pnh-46", "REP", "PNH", 46, "study", "normal", "Mekong Study", "medium"),
    _focus_route("bmv-hui-48", "BMV", "HUI", 48, "review", "normal", "Review Route", "medium"),
    _focus_route("pqc-cxr-48", "PQC", "CXR", 48, "creative", "normal", "Coastal Creative", "medium"),
    _focus_route("dad-cxr-50", "DAD", "CXR", 50, "deep_work", "normal", "Coastal Deep Work", "medium"),
    _focus_route("cxr-dad-50", "CXR", "DAD", 50, "deep_work", "normal", "Coastal Deep Return", "medium"),
    _focus_route("dad-dli-54", "DAD", "DLI", 54, "creative", "deep", "Creative Cruise", "long"),
    _focus_route("dli-dad-54", "DLI", "DAD", 54, "creative", "deep", "Creative Return", "long"),
    _focus_route("bmv-sgn-58", "BMV", "SGN", 58, "creative", "normal", "Highland Focus", "long"),
    _focus_route("hui-dli-59", "HUI", "DLI", 59, "creative", "normal", "Cloud Route", "long"),
    _focus_route("dli-hui-59", "DLI", "HUI", 59, "creative", "normal", "Cloud Return", "long"),
    _focus_route("sgn-pnh-62", "SGN", "PNH", 62, "study", "normal", "Mekong Focus", "long"),
    _focus_route("pnh-sgn-62", "PNH", "SGN", 62, "study", "normal", "Mekong Return", "long"),
    _focus_route("han-hui-65", "HAN", "HUI", 65, "reading", "normal", "Reading Route", "long"),
    _focus_route("pnh-bkk-66", "PNH", "BKK", 66, "deep_work", "normal", "Regional Deep Work", "long"),
    _focus_route("dad-sgn-69", "DAD", "SGN", 69, "study", "normal", "Study Cruise", "long"),
    _focus_route("bkk-pnh-70", "BKK", "PNH", 70, "deep_work", "normal", "Mekong Deep Work", "long"),
    _focus_route("dad-han-71", "DAD", "HAN", 71, "deep_work", "deep", "Deep Work Route", "long"),
    _focus_route("hui-sgn-72", "HUI", "SGN", 72, "reading", "normal", "Reading Cruise", "long"),
    _focus_route("pqc-pnh-72", "PQC", "PNH", 72, "study", "normal", "Island Study Cruise", "long"),
    _focus_route("rep-sgn-75", "REP", "SGN", 75, "deep_work", "normal", "Borderless Focus", "long"),
    _focus_route("han-vte-78", "HAN", "VTE", 78, "reading", "normal", "Borderless Reading", "long"),
    _focus_route("vte-han-78", "VTE", "HAN", 78, "reading", "normal", "Northern Return", "long"),
    _focus_route("vii-vte-82", "VII", "VTE", 82, "reading", "normal", "Border Reading", "long"),
    _focus_route("vte-vii-82", "VTE", "VII", 82, "reading", "normal", "Border Return", "long"),
    _focus_route("vte-bkk-86", "VTE", "BKK", 86, "deep_work", "deep", "Regional Deep Route", "long"),
    _focus_route("dad-bkk-88", "DAD", "BKK", 88, "deep_work", "deep", "Regional Deep Work", "long"),
    _focus_route("bkk-dad-88", "BKK", "DAD", 88, "deep_work", "deep", "Regional Return", "long"),
    _focus_route("cxr-sin-96", "CXR", "SIN", 96, "creative", "normal", "Island Creative", "long"),
    _focus_route("sin-cxr-96", "SIN", "CXR", 96, "creative", "normal", "Creative Return", "long"),
    _focus_route("sgn-sin-104", "SGN", "SIN", 104, "deep_work", "deadline", "International Focus", "long"),
    _focus_route("sin-sgn-104", "SIN", "SGN", 104, "deep_work", "deadline", "International Return", "long"),
    _focus_route("kul-sgn-106", "KUL", "SGN", 106, "deep_work", "deadline", "Regional Deadline", "long"),
    _focus_route("dad-kul-116", "DAD", "KUL", 116, "deep_work", "deadline", "Long Haul Focus", "long"),
    _focus_route("kul-dad-116", "KUL", "DAD", 116, "deep_work", "deadline", "Long Haul Return", "long"),
    _focus_route("han-bmv-112", "HAN", "BMV", 112, "deep_work", "deadline", "Long Focus", "long"),
]


FOCUS_ROUTE_SLOT_MINUTES = tuple(range(30, 121, 5))


def _route_difficulty_for_minutes(minutes: int) -> str:
    if minutes <= 35:
        return "short"
    if minutes <= 60:
        return "medium"
    return "long"


def _route_task_type_for_minutes(minutes: int) -> str:
    if minutes <= 40:
        return "reading"
    if minutes <= 60:
        return "creative"
    if minutes <= 85:
        return "study"
    return "deep_work"


def _route_mode_for_minutes(minutes: int) -> str:
    if minutes >= 100:
        return "deadline"
    if minutes >= 80:
        return "deep"
    return "normal"


def _estimated_focus_minutes(from_code: str, to_code: str) -> int:
    distance = _airport_distance_km(from_code, to_code)
    if distance <= 0:
        return 30
    # Symbolic focus-flight time: grounded in route distance, but capped to the
    # app's sub-2-hour session range.
    return _round_minutes_to_five(int(round(26.0 + (distance / 13.0))))


def _generated_route_candidate(from_code: str, to_code: str) -> Dict[str, Any]:
    minutes = _estimated_focus_minutes(from_code, to_code)
    return _focus_route(
        f"auto-{from_code.lower()}-{to_code.lower()}-{minutes:03d}",
        from_code,
        to_code,
        minutes,
        _route_task_type_for_minutes(minutes),
        _route_mode_for_minutes(minutes),
        "Scheduled Focus Flight",
        _route_difficulty_for_minutes(minutes),
    )


def _candidate_routes_for_origin(origin: str) -> List[Dict[str, Any]]:
    origin = str(origin or "").strip().upper()
    candidates = [dict(route) for route in FOCUS_ROUTE_PRESETS if str(route.get("from_code", "")).upper() == origin]
    for to_code in sorted(FOCUS_AIRPORT_DATA):
        if to_code == origin:
            continue
        candidates.append(_generated_route_candidate(origin, to_code))

    deduped: Dict[tuple[str, int], Dict[str, Any]] = {}
    for route in candidates:
        key = (
            str(route.get("to_code", "")).upper(),
            int(route.get("raw_duration_minutes", route.get("duration_minutes", 0)) or 0),
        )
        if key not in deduped:
            deduped[key] = route
    return list(deduped.values())


def _build_focus_route_schedule() -> List[Dict[str, Any]]:
    routes: List[Dict[str, Any]] = []
    for origin in sorted(FOCUS_AIRPORT_DATA):
        candidates = _candidate_routes_for_origin(origin)
        if not candidates:
            continue
        for slot_minutes in FOCUS_ROUTE_SLOT_MINUTES:
            source = min(
                candidates,
                key=lambda route: (
                    abs(int(route.get("raw_duration_minutes", route.get("duration_minutes", 0)) or 0) - slot_minutes),
                    abs(int(route.get("duration_minutes", 0) or 0) - slot_minutes),
                    int(route.get("route_distance_km", 0) or 0),
                    str(route.get("to_code", "")),
                ),
            )
            route = dict(source)
            route["source_route_id"] = str(source.get("route_id", ""))
            route["route_id"] = f"slot-{origin.lower()}-{str(route.get('to_code', '')).lower()}-{slot_minutes:03d}"
            route["duration_minutes"] = int(slot_minutes)
            route["schedule_slot_minutes"] = int(slot_minutes)
            route["task_type"] = _route_task_type_for_minutes(slot_minutes)
            route["mode"] = _route_mode_for_minutes(slot_minutes)
            route["difficulty"] = _route_difficulty_for_minutes(slot_minutes)
            route["route_theme"] = str(source.get("route_theme") or "Scheduled Focus Flight")
            route["short_label"] = f"{origin} -> {route.get('to_code', '')}"
            routes.append(route)
    return routes


FOCUS_ROUTE_SCHEDULE: List[Dict[str, Any]] = _build_focus_route_schedule()


def _route_difficulty_label(route: Dict[str, Any]) -> str:
    difficulty = str(route.get("difficulty", "medium") or "medium")
    return {"short": "Ngắn", "medium": "Vừa", "long": "Dài"}.get(difficulty, "Vừa")


def _configured_focus_origin(config: Optional[dict]) -> str:
    data = dict(config or {})
    for key in (
        "focus_journey_current_airport",
        "journey_current_airport",
        "last_journey_to_code",
        "focus_journey_home_airport",
    ):
        code = str(data.get(key, "") or "").strip().upper()
        if code in FOCUS_AIRPORT_DATA:
            return code
    return "DAD"


def _nearest_focus_routes(minutes: int, limit: int = 3, from_code: str = "") -> List[Dict[str, Any]]:
    target = _round_minutes_to_five(minutes)
    origin = str(from_code or "").strip().upper()
    candidate_pool = FOCUS_ROUTE_SCHEDULE or FOCUS_ROUTE_PRESETS
    if origin in FOCUS_AIRPORT_DATA:
        origin_routes = [route for route in candidate_pool if str(route.get("from_code", "")).upper() == origin]
        if origin_routes:
            candidate_pool = origin_routes
    scored = sorted(
        candidate_pool,
        key=lambda route: (
            abs(int(route.get("duration_minutes", 0) or 0) - target),
            int(route.get("duration_minutes", 0) or 0),
            str(route.get("to_code", "")),
        ),
    )
    nearby = scored[:limit]
    return sorted(
        nearby,
        key=lambda route: (
            int(route.get("duration_minutes", 0) or 0),
            int(route.get("route_distance_km", 0) or 0),
            str(route.get("to_code", "")),
        ),
    )


class DurationTimelineSlider(QWidget):
    """Flight-board style duration selector with ruler ticks."""

    valueChanged = pyqtSignal(int)

    def __init__(self, *, is_dark: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._is_dark = bool(is_dark)
        self._minimum = 30
        self._maximum = 120
        self._step = 5
        self._value = 35
        self._dragging = False
        self.setMinimumHeight(74)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def setRange(self, minimum: int, maximum: int) -> None:
        self._minimum = int(minimum)
        self._maximum = max(self._minimum, int(maximum))
        self.setValue(self._value)
        self.update()

    def setMaximum(self, maximum: int) -> None:
        self.setRange(self._minimum, maximum)

    def maximum(self) -> int:
        return self._maximum

    def setSingleStep(self, step: int) -> None:
        self._step = max(1, int(step or 1))

    def setPageStep(self, step: int) -> None:
        _ = step

    def setTickInterval(self, step: int) -> None:
        _ = step

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:
        value = self._clamp_to_step(value)
        if value == self._value:
            self.update()
            return
        self._value = value
        self.update()
        if not self.signalsBlocked():
            self.valueChanged.emit(self._value)

    def stepBy(self, steps: int) -> None:
        self.setValue(self._value + int(steps) * self._step)

    def _clamp_to_step(self, value: int) -> int:
        value = max(self._minimum, min(self._maximum, int(value)))
        snapped = round((value - self._minimum) / self._step) * self._step + self._minimum
        return max(self._minimum, min(self._maximum, int(snapped)))

    def _track_rect(self) -> QRectF:
        return QRectF(18, 24, max(1, self.width() - 36), 8)

    def _x_for_value(self, value: int) -> float:
        track = self._track_rect()
        span = max(1, self._maximum - self._minimum)
        return track.left() + ((value - self._minimum) / span) * track.width()

    def _value_for_x(self, x: float) -> int:
        track = self._track_rect()
        ratio = (float(x) - track.left()) / max(1.0, track.width())
        return self._clamp_to_step(int(round(self._minimum + ratio * (self._maximum - self._minimum))))

    @staticmethod
    def _fmt_minutes(value: int) -> str:
        if value < 60:
            return f"{value}m"
        hours, minutes = divmod(value, 60)
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"

    def mousePressEvent(self, event) -> None:
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.grabMouse()
            self.setValue(self._value_for_x(event.position().x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.isEnabled() and (self._dragging or event.buttons() & Qt.MouseButton.LeftButton):
            self.setValue(self._value_for_x(event.position().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.releaseMouse()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        if not self.isEnabled():
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta:
            self.stepBy(1 if delta > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event) -> None:
        if not self.isEnabled():
            super().keyPressEvent(event)
            return
        key = event.key()
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            self.stepBy(1)
            event.accept()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            self.stepBy(-1)
            event.accept()
            return
        if key == Qt.Key.Key_PageUp:
            self.stepBy(2)
            event.accept()
            return
        if key == Qt.Key.Key_PageDown:
            self.stepBy(-2)
            event.accept()
            return
        if key == Qt.Key.Key_Home:
            self.setValue(self._minimum)
            event.accept()
            return
        if key == Qt.Key.Key_End:
            self.setValue(self._maximum)
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        enabled = self.isEnabled()
        track = self._track_rect()
        bg = QColor("#0d1a29" if self._is_dark else "#f8fdff")
        border = QColor("#38506b" if self._is_dark else "#8bb5d1")
        accent = QColor("#5fe4d4" if self._is_dark else "#2fa79a")
        tick = QColor(220, 238, 250, 210 if self._is_dark else 235)
        label = QColor("#d8eafa" if self._is_dark else "#2a526d")
        if not enabled:
            accent = QColor("#6f8798" if self._is_dark else "#a5b7c4")
            tick = QColor(160, 174, 188, 135)
            label = QColor("#7b93a8" if self._is_dark else "#7d95a7")

        painter.setPen(QPen(border, 1.2))
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(track, 3, 3)

        selected = QRectF(track.left(), track.top(), max(0.0, self._x_for_value(self._value) - track.left()), track.height())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(accent))
        painter.drawRoundedRect(selected, 3, 3)

        painter.setPen(QPen(tick, 2))
        for value in range(self._minimum, self._maximum + 1, self._step):
            x = self._x_for_value(value)
            major = (value % 10 == 0) or value in (self._minimum, self._maximum)
            top = 38 if major else 44
            bottom = 56 if major else 52
            painter.drawLine(int(x), top, int(x), bottom)

        handle_x = self._x_for_value(self._value)
        triangle = QPainterPath()
        triangle.moveTo(handle_x, 4)
        triangle.lineTo(handle_x - 8, 18)
        triangle.lineTo(handle_x + 8, 18)
        triangle.closeSubpath()
        painter.setBrush(QBrush(accent))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(triangle)
        painter.setPen(QPen(bg, 3))
        painter.setBrush(QBrush(accent))
        painter.drawEllipse(QPointF(handle_x, track.center().y()), 13, 13)

    def _label_values(self) -> List[int]:
        preferred = [30, 40, 50, 60, 70, 90, 120]
        values = [v for v in preferred if self._minimum <= v <= self._maximum]
        if self._minimum not in values and self._minimum < 30:
            values.insert(0, self._minimum)
        if self._maximum not in values:
            values.append(self._maximum)
        return values[:6]


class FocusRouteCardButton(QPushButton):
    """Painted route card with flight-code badge."""

    def __init__(self, *, is_dark: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._is_dark = bool(is_dark)
        self._route: Dict[str, Any] = {}
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(128)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_route(self, route: Dict[str, Any]) -> None:
        self._route = dict(route or {})
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        checked = self.isChecked()

        bg = QColor("#07111f" if self._is_dark else "#f4fcff")
        bg_checked = QColor("#dffaff" if self._is_dark else "#ddfbff")
        border = QColor("#243a50" if self._is_dark else "#c9e3ef")
        border_checked = QColor("#f4cf2f" if checked else "#34546b")
        text = QColor("#f8fcff" if self._is_dark and not checked else "#07111f")
        muted = QColor("#9cadbc" if self._is_dark and not checked else "#607686")
        badge_bg = QColor("#ffd52f")

        painter.setPen(QPen(border_checked if checked else border, 1.2))
        painter.setBrush(QBrush(bg_checked if checked else bg))
        painter.drawRoundedRect(rect, 16, 16)

        code = str(self._route.get("to_code") or "DAD")
        badge = QRectF(rect.left() + 14, rect.top() + 14, 76, 32)
        painter.setPen(QPen(QColor("#000814"), 2))
        painter.setBrush(QBrush(badge_bg))
        painter.drawRoundedRect(badge, 7, 7)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#111111"))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, f"✈ {code}")

        name = str(self._route.get("to_name") or code)
        minutes = int(self._route.get("duration_minutes", 0) or 0)
        distance = int(self._route.get("route_distance_km", 0) or 0)
        name_rect = QRectF(rect.left() + 14, rect.top() + 58, rect.width() - 28, 24)
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        painter.setPen(text)
        display_name = painter.fontMetrics().elidedText(
            name,
            Qt.TextElideMode.ElideRight,
            max(1, int(name_rect.width())),
        )
        painter.drawText(
            name_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            display_name,
        )
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        painter.setPen(muted)
        meta = f"{minutes}m"
        if distance > 0:
            meta = f"{meta} · {distance} km"
        painter.drawText(
            QRectF(rect.left() + 14, rect.top() + 88, rect.width() - 28, 22),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            meta,
        )


class BoardingPassTicketWidget(QWidget):
    """Interactive check-in ticket; swipe the perforation after check-in."""

    torn = pyqtSignal()

    def __init__(self, *, context_payload: Dict[str, Any], is_dark: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._ctx = dict(context_payload or {})
        self._is_dark = bool(is_dark)
        self._checked_in = False
        self._dragging = False
        self._tear_progress = 0.0
        self._blink_on = False
        self.setMinimumHeight(438)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(420)
        self._blink_timer.timeout.connect(self._pulse_blink)
        self._blink_timer.start()

    def set_checked_in(self, checked: bool) -> None:
        self._checked_in = bool(checked)
        self._tear_progress = 0.0
        self.setCursor(Qt.CursorShape.PointingHandCursor if self._checked_in else Qt.CursorShape.ArrowCursor)
        self.update()

    def _pulse_blink(self) -> None:
        if self._checked_in:
            self._blink_on = not self._blink_on
            self.update()

    def _route_data(self) -> Dict[str, Any]:
        from_code = str(self._ctx.get("route_from_code") or "DAD").upper()
        to_code = str(self._ctx.get("route_to_code") or "SGN").upper()
        from_name = str(self._ctx.get("route_from_name") or FOCUS_AIRPORT_DATA.get(from_code, {}).get("name") or from_code)
        to_name = str(self._ctx.get("route_to_name") or FOCUS_AIRPORT_DATA.get(to_code, {}).get("name") or to_code)
        distance = int(self._ctx.get("route_distance_km") or _airport_distance_km(from_code, to_code) or 0)
        minutes = int(
            self._ctx.get("route_duration_minutes")
            or self._ctx.get("duration_minutes")
            or self._ctx.get("planned_minutes")
            or 0
        )
        return {
            "from_code": from_code,
            "to_code": to_code,
            "from_name": from_name,
            "to_name": to_name,
            "distance_km": distance,
            "minutes": minutes,
        }

    def _ticket_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(6, 6, -6, -6)

    def _tear_y(self, ticket: QRectF) -> float:
        return ticket.bottom() - 132

    def _barcode_seed(self) -> int:
        route = self._route_data()
        text = f"{route['from_code']}{route['to_code']}{route['minutes']}{route['distance_km']}"
        return sum((i + 1) * ord(ch) for i, ch in enumerate(text))

    def mousePressEvent(self, event) -> None:
        ticket = self._ticket_rect()
        tear_y = self._tear_y(ticket)
        if self._checked_in and event.button() == Qt.MouseButton.LeftButton and abs(event.position().y() - tear_y) <= 34:
            self._dragging = True
            self.grabMouse()
            self._update_tear_progress(event.position().x(), ticket)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._update_tear_progress(event.position().x(), self._ticket_rect())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.releaseMouse()
            if self._tear_progress >= 0.92:
                self._tear_progress = 1.0
                self.update()
                self.torn.emit()
            else:
                self._tear_progress = 0.0
                self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _update_tear_progress(self, x: float, ticket: QRectF) -> None:
        left = ticket.left() + 38
        right = ticket.right() - 38
        self._tear_progress = max(0.0, min(1.0, (float(x) - left) / max(1.0, right - left)))
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        ticket = self._ticket_rect()
        tear_y = self._tear_y(ticket)
        route = self._route_data()

        bg = QColor("#1d1f20" if self._is_dark else "#202324")
        bg_bottom = QColor("#111315")
        grad = QLinearGradient(ticket.topLeft(), ticket.bottomRight())
        grad.setColorAt(0.0, bg)
        grad.setColorAt(1.0, bg_bottom)
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(ticket, 14, 14)

        painter.save()
        painter.setClipRect(ticket.adjusted(0, 0, 0, -126))
        self._draw_dotted_world(painter, ticket)
        painter.restore()

        painter.setFont(QFont("Segoe UI", 34, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(ticket.left() + 34, ticket.top() + 38, 150, 54), Qt.AlignmentFlag.AlignLeft, route["from_code"])
        painter.drawText(QRectF(ticket.right() - 184, ticket.top() + 38, 150, 54), Qt.AlignmentFlag.AlignRight, route["to_code"])

        painter.setFont(QFont("Segoe UI", 12))
        painter.setPen(QColor(210, 214, 218, 180))
        painter.drawText(QRectF(ticket.left() + 36, ticket.top() + 96, 160, 24), Qt.AlignmentFlag.AlignLeft, route["from_name"])
        painter.drawText(QRectF(ticket.right() - 196, ticket.top() + 96, 160, 24), Qt.AlignmentFlag.AlignRight, route["to_name"])

        painter.setFont(QFont("Segoe UI", 12))
        painter.setPen(QColor(210, 214, 218, 165))
        painter.drawText(QRectF(ticket.center().x() - 50, ticket.top() + 104, 100, 22), Qt.AlignmentFlag.AlignCenter, f"{route['minutes']}m")
        self._draw_small_plane(painter, QPointF(ticket.center().x(), ticket.top() + 82), QColor(160, 166, 170, 90))

        label = QColor(210, 214, 218, 150)
        value = QColor("#ffffff")
        painter.setFont(QFont("Segoe UI", 11))
        painter.setPen(label)
        painter.drawText(QRectF(ticket.left() + 36, ticket.top() + 146, 120, 22), "Seat")
        painter.drawText(QRectF(ticket.left() + 36, ticket.top() + 216, 120, 22), "Boarding")
        painter.drawText(QRectF(ticket.right() - 176, ticket.top() + 146, 140, 22), Qt.AlignmentFlag.AlignRight, "Distance")
        painter.drawText(QRectF(ticket.right() - 176, ticket.top() + 216, 140, 22), Qt.AlignmentFlag.AlignRight, "Date")

        painter.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        painter.setPen(value)
        painter.drawText(QRectF(ticket.left() + 36, ticket.top() + 174, 120, 24), "02A")
        painter.drawText(QRectF(ticket.left() + 36, ticket.top() + 244, 120, 24), "Now")
        painter.drawText(QRectF(ticket.right() - 176, ticket.top() + 174, 140, 24), Qt.AlignmentFlag.AlignRight, f"{route['distance_km']} km")
        painter.drawText(QRectF(ticket.right() - 176, ticket.top() + 244, 140, 24), Qt.AlignmentFlag.AlignRight, datetime.now().strftime("%Y/%m/%d"))

        self._draw_perforation(painter, ticket, tear_y)
        self._draw_barcode(painter, ticket.adjusted(38, tear_y - ticket.top() + 28, -38, -22))

    def _draw_perforation(self, painter: QPainter, ticket: QRectF, y: float) -> None:
        cut_color = QColor("#2e6c86" if self._is_dark else "#4d8ba4")
        painter.setPen(QPen(cut_color, 1.4, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(ticket.left() + 38, y), QPointF(ticket.right() - 38, y))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#244c62" if self._is_dark else "#d8f2f7"))
        painter.drawEllipse(QPointF(ticket.left(), y), 18, 18)
        painter.drawEllipse(QPointF(ticket.right(), y), 18, 18)

        if self._checked_in:
            alpha = 175 if self._blink_on else 70
            glow_x = ticket.left() + 38 + (ticket.width() - 76) * self._tear_progress
            painter.setPen(QPen(QColor(95, 228, 212, alpha), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(ticket.left() + 38, y), QPointF(glow_x, y))
            painter.setBrush(QColor(95, 228, 212, alpha))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(glow_x, y), 7, 7)

    def _draw_barcode(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ffffff"))
        painter.drawRect(rect)
        seed = self._barcode_seed()
        x = rect.left() + 8
        max_x = rect.right() - 8
        while x < max_x:
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            width = 1 + (seed % 5)
            gap = 1 + ((seed >> 3) % 3)
            painter.setBrush(QColor("#111111"))
            painter.drawRect(QRectF(x, rect.top() + 6, width, rect.height() - 12))
            x += width + gap
        painter.restore()

    def _draw_dotted_world(self, painter: QPainter, ticket: QRectF) -> None:
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 22))
        clusters = [
            QRectF(ticket.left() + 120, ticket.top() + 22, 120, 115),
            QRectF(ticket.left() + 165, ticket.top() + 132, 115, 150),
            QRectF(ticket.center().x() - 30, ticket.top() + 58, 145, 122),
            QRectF(ticket.right() - 190, ticket.top() + 48, 128, 150),
        ]
        for cluster in clusters:
            step = 6
            for yy in range(int(cluster.top()), int(cluster.bottom()), step):
                for xx in range(int(cluster.left()), int(cluster.right()), step):
                    cx = (xx - cluster.center().x()) / max(1.0, cluster.width() / 2)
                    cy = (yy - cluster.center().y()) / max(1.0, cluster.height() / 2)
                    if cx * cx + cy * cy < 0.82:
                        painter.drawEllipse(QPointF(xx, yy), 1.2, 1.2)
        painter.restore()

    def _draw_small_plane(self, painter: QPainter, point: QPointF, color: QColor) -> None:
        painter.save()
        painter.translate(point)
        painter.rotate(90)
        plane = QPainterPath()
        plane.moveTo(0, -10)
        plane.lineTo(4, 3)
        plane.lineTo(10, 6)
        plane.lineTo(2, 7)
        plane.lineTo(0, 12)
        plane.lineTo(-2, 7)
        plane.lineTo(-10, 6)
        plane.lineTo(-4, 3)
        plane.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPath(plane)
        painter.restore()


class PlaneSeatSelectionWidget(QWidget):
    """Animated cabin seat selector before entering the live journey map."""

    seatSelected = pyqtSignal(str)
    taskTypeSelected = pyqtSignal(str)
    takeoffFinished = pyqtSignal()

    TASK_OPTIONS: tuple[tuple[str, str, str], ...] = (
        ("Làm sâu", "deep_work", "#d7bd70"),
        ("Lập trình", "coding", "#9d9bed"),
        ("Tài liệu", "reading", "#98d78d"),
        ("Sáng tạo", "creative", "#82d8dd"),
        ("Rà soát", "review", "#dc93c9"),
        ("Học tập", "study", "#9aa4b4"),
    )

    def __init__(self, *, is_dark: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._is_dark = bool(is_dark)
        self._entry_progress = 0.0
        self._scroll_progress = 0.0
        self._manual_scroll = 0.0
        self._last_tick_at = time.monotonic()
        self._taking_off = False
        self._selected_seat = ""
        self._selected_task_type = ""
        self.setMinimumHeight(720)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        now = time.monotonic()
        step = max(0.25, min(2.5, (now - self._last_tick_at) / (1.0 / 60.0)))
        self._last_tick_at = now
        if self._entry_progress < 1.0:
            self._entry_progress = min(1.0, self._entry_progress + 0.032 * step)
            self.update()
            return
        if self._taking_off:
            self._scroll_progress = min(1.0, self._scroll_progress + 0.010 * step)
            self.update()
            if self._scroll_progress >= 1.0:
                self._timer.stop()
                self.takeoffFinished.emit()

    def _max_manual_scroll(self) -> float:
        return max(0.0, self._plane_base_height() - self.height() + 150.0)

    def _plane_offset(self) -> float:
        base_height = self._plane_base_height()
        scroll_distance = max(0.0, base_height - self.height() + 118.0)
        entry_offset = (1.0 - self._entry_progress) * self.height() * 0.70
        return entry_offset - self._manual_scroll - (self._scroll_progress * scroll_distance)

    def _plane_base_height(self) -> float:
        return max(2600.0, self.height() * 3.05)

    def _plane_rect(self) -> QRectF:
        width = min(680.0, self.width() * 0.66)
        height = self._plane_base_height()
        return QRectF((self.width() - width) / 2.0, 8 + self._plane_offset(), width, height)

    def _all_seat_rects(self) -> List[tuple[str, QRectF]]:
        body = self._plane_rect()
        rows = 22
        start_y = body.top() + body.height() * 0.205
        tail_top = body.bottom() - body.height() * 0.075
        row_gap = 13.0
        available_h = max(1.0, tail_top - start_y)
        seat_size = min(54.0, max(42.0, (available_h - (rows - 1) * row_gap) / rows))
        gap_x = max(7.0, seat_size * 0.16)
        aisle_gap = max(88.0, body.width() * 0.155)
        center_x = body.center().x()

        col_x = {
            "A": center_x - aisle_gap / 2.0 - seat_size * 2.0 - gap_x,
            "C": center_x - aisle_gap / 2.0 - seat_size,
            "D": center_x + aisle_gap / 2.0,
            "F": center_x + aisle_gap / 2.0 + seat_size + gap_x,
        }

        seats: List[tuple[str, QRectF]] = []
        for row in range(1, rows + 1):
            y = start_y + (row - 1) * (seat_size + row_gap)
            for column in ("A", "C", "D", "F"):
                seats.append((f"{row:02d}{column}", QRectF(col_x[column], y, seat_size, seat_size)))
        return seats

    def _seat_rects(self) -> List[tuple[str, QRectF]]:
        return self._all_seat_rects()

    def _task_panel_rect(self) -> QRectF:
        body = self._plane_rect()
        width = min(560.0, self.width() * 0.88)
        height = 190.0
        selected_rect = next((rect for seat, rect in self._all_seat_rects() if seat == self._selected_seat), None)
        if selected_rect is not None:
            y = selected_rect.top() - height - 20.0
            if y < 54.0:
                y = selected_rect.bottom() + 20.0
        else:
            y = body.top() + body.height() * 0.23
        y = max(54.0, min(y, self.height() - height - 28.0))
        return QRectF((self.width() - width) / 2.0, y, width, height)

    def _task_chip_rects(self) -> List[tuple[str, str, str, QRectF]]:
        panel = self._task_panel_rect()
        chip_h = 38.0
        x = panel.left() + 22.0
        y = panel.top() + 96.0
        rows: List[tuple[str, str, str, QRectF]] = []
        for index, (label, value, color) in enumerate(self.TASK_OPTIONS):
            chip_w = {
                "deep_work": 112.0,
                "coding": 122.0,
                "reading": 112.0,
                "creative": 122.0,
                "review": 112.0,
                "study": 108.0,
            }.get(value, 108.0)
            if index == 3:
                x = panel.left() + 22.0
                y += chip_h + 12.0
            rows.append((label, value, color, QRectF(x, y, chip_w, chip_h)))
            x += chip_w + 12.0
        return rows

    def mousePressEvent(self, event) -> None:
        if self._taking_off or self._entry_progress < 0.92 or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if self._selected_seat:
            for _label, value, _color, rect in self._task_chip_rects():
                if rect.contains(event.position()):
                    self._selected_task_type = value
                    self.taskTypeSelected.emit(value)
                    self._taking_off = True
                    if not self._timer.isActive():
                        self._timer.start()
                    self.update()
                    event.accept()
                    return
        for seat, rect in self._seat_rects():
            if rect.contains(event.position()):
                self._selected_seat = seat
                self.seatSelected.emit(seat)
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:
        if self._taking_off:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        self._manual_scroll = max(0.0, min(self._max_manual_scroll(), self._manual_scroll - delta * 0.48))
        self.update()
        event.accept()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        bg = QLinearGradient(rect.topLeft(), rect.bottomRight())
        bg.setColorAt(0.0, QColor("#111419"))
        bg.setColorAt(0.6, QColor("#07131d"))
        bg.setColorAt(1.0, QColor("#111111"))
        painter.fillRect(rect, QBrush(bg))
        painter.fillRect(rect, QColor(0, 0, 0, 92))

        body = self._plane_rect()
        painter.save()
        fuselage = QPainterPath()
        center_x = body.center().x()
        nose_y = body.top()
        shoulder_y = body.top() + body.height() * 0.142
        lower_y = body.bottom() - body.height() * 0.070
        tail_tip_y = body.bottom() - body.height() * 0.010
        half_body = body.width() * 0.385
        half_tail = body.width() * 0.365
        fuselage.moveTo(center_x, nose_y)
        fuselage.cubicTo(
            center_x + body.width() * 0.170, nose_y + body.height() * 0.002,
            center_x + body.width() * 0.315, nose_y + body.height() * 0.052,
            center_x + half_body, shoulder_y,
        )
        fuselage.cubicTo(
            center_x + body.width() * 0.405, body.top() + body.height() * 0.285,
            center_x + body.width() * 0.392, body.bottom() - body.height() * 0.155,
            center_x + half_tail, lower_y,
        )
        fuselage.cubicTo(
            center_x + body.width() * 0.270, tail_tip_y,
            center_x + body.width() * 0.105, body.bottom(),
            center_x, body.bottom() - body.height() * 0.004,
        )
        fuselage.cubicTo(
            center_x - body.width() * 0.105, body.bottom(),
            center_x - body.width() * 0.270, tail_tip_y,
            center_x - half_tail, lower_y,
        )
        fuselage.cubicTo(
            center_x - body.width() * 0.392, body.bottom() - body.height() * 0.155,
            center_x - body.width() * 0.405, body.top() + body.height() * 0.285,
            center_x - half_body, shoulder_y,
        )
        fuselage.cubicTo(
            center_x - body.width() * 0.315, nose_y + body.height() * 0.052,
            center_x - body.width() * 0.170, nose_y + body.height() * 0.002,
            center_x, nose_y,
        )
        fuselage.closeSubpath()

        wing_root_y = body.top() + body.height() * 0.400
        wing_tip_y = body.top() + body.height() * 0.550
        wing_trail_y = body.top() + body.height() * 0.600
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 18))
        painter.drawPolygon(
            QPointF(center_x - body.width() * 0.350, wing_root_y),
            QPointF(body.left() - body.width() * 0.950, wing_tip_y),
            QPointF(body.left() - body.width() * 0.18, wing_trail_y),
            QPointF(center_x - body.width() * 0.315, wing_trail_y - body.height() * 0.022),
        )
        painter.drawPolygon(
            QPointF(center_x + body.width() * 0.350, wing_root_y),
            QPointF(body.right() + body.width() * 0.950, wing_tip_y),
            QPointF(body.right() + body.width() * 0.18, wing_trail_y),
            QPointF(center_x + body.width() * 0.315, wing_trail_y - body.height() * 0.022),
        )
        plane_grad = QLinearGradient(QPointF(body.center().x(), body.top()), QPointF(body.center().x(), body.bottom()))
        plane_grad.setColorAt(0.0, QColor("#2b2d31"))
        plane_grad.setColorAt(0.46, QColor("#17191c"))
        plane_grad.setColorAt(1.0, QColor("#111215"))
        painter.setPen(QPen(QColor(255, 255, 255, 22), 1))
        painter.setBrush(QBrush(plane_grad))
        painter.drawPath(fuselage)

        # ---- Cockpit windows ----

        window_top = body.top() + body.height() * 0.045
        window_h = body.height() * 0.095
        center_gap = body.width() * 0.034

        inner_top_y = window_top
        inner_bottom_y = window_top + window_h * 0.48
        outer_top_y = window_top + window_h * 0.10
        outer_mid_y = window_top + window_h * 0.34
        outer_bottom_y = window_top + window_h * 0.60

        inner_left_x = center_x - center_gap
        inner_right_x = center_x + center_gap

        outer_left_x = center_x - body.width() * 0.270
        outer_right_x = center_x + body.width() * 0.270

        ctrl_left_top_1 = QPointF(center_x - body.width() * 0.070, window_top - window_h * 0.22)
        ctrl_left_top_2 = QPointF(center_x - body.width() * 0.170, window_top - window_h * 0.02)

        ctrl_left_bottom_1 = QPointF(center_x - body.width() * 0.175, window_top + window_h * 0.46)
        ctrl_left_bottom_2 = QPointF(center_x - body.width() * 0.080, window_top + window_h * 0.40)

        ctrl_right_top_1 = QPointF(center_x + body.width() * 0.070, window_top - window_h * 0.22)
        ctrl_right_top_2 = QPointF(center_x + body.width() * 0.170, window_top - window_h * 0.02)

        ctrl_right_bottom_1 = QPointF(center_x + body.width() * 0.175, window_top + window_h * 0.46)
        ctrl_right_bottom_2 = QPointF(center_x + body.width() * 0.080, window_top + window_h * 0.40)

        glass_grad = QLinearGradient(
            QPointF(center_x, window_top),
            QPointF(center_x, window_top + window_h)
        )
        glass_grad.setColorAt(0.0, QColor(255, 255, 255, 78))
        glass_grad.setColorAt(0.45, QColor(230, 232, 238, 62))
        glass_grad.setColorAt(1.0, QColor(180, 185, 195, 34))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glass_grad))

        # Left window
        left_window = QPainterPath()
        left_window.moveTo(inner_left_x, inner_top_y)
        left_window.cubicTo(
            ctrl_left_top_1,
            ctrl_left_top_2,
            QPointF(outer_left_x, outer_mid_y),
        )
        left_window.lineTo(outer_left_x, outer_bottom_y)
        left_window.cubicTo(
            ctrl_left_bottom_1,
            ctrl_left_bottom_2,
            QPointF(inner_left_x, inner_bottom_y),
        )
        left_window.lineTo(inner_left_x, inner_top_y)
        left_window.closeSubpath()

        # Right window
        right_window = QPainterPath()
        right_window.moveTo(inner_right_x, inner_top_y)
        right_window.cubicTo(
            ctrl_right_top_1,
            ctrl_right_top_2,
            QPointF(outer_right_x, outer_mid_y),
        )
        right_window.lineTo(outer_right_x, outer_bottom_y)
        right_window.cubicTo(
            ctrl_right_bottom_1,
            ctrl_right_bottom_2,
            QPointF(inner_right_x, inner_bottom_y),
        )
        right_window.lineTo(inner_right_x, inner_top_y)
        right_window.closeSubpath()

        painter.drawPath(left_window)
        painter.drawPath(right_window)

        # viền kính nhẹ cho rõ form hơn
        painter.setPen(QPen(QColor(255, 255, 255, 22), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(left_window)
        painter.drawPath(right_window)

        painter.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        painter.setPen(QColor(220, 224, 230, 170))
        row_one_rects = [(seat, rect) for seat, rect in self._all_seat_rects() if seat.startswith("01")]
        row_one_centers = {seat[-1]: rect.center().x() for seat, rect in row_one_rects}
        label_y = min((rect.top() for _seat, rect in row_one_rects), default=body.top() + body.height() * 0.25) - 44.0
        for label in ("A", "C", "D", "F"):
            x = row_one_centers.get(label, body.center().x())
            painter.drawText(QRectF(x - 18, label_y, 36, 24), Qt.AlignmentFlag.AlignCenter, label)
        seat_rects = self._all_seat_rects()
        row_centers: Dict[int, float] = {}
        for seat, seat_rect in seat_rects:
            row_centers.setdefault(int(seat[:2]), seat_rect.center().y())
        painter.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        for row, center_y in sorted(row_centers.items()):
            painter.drawText(
                QRectF(body.center().x() - 24, center_y - 12, 48, 24),
                Qt.AlignmentFlag.AlignCenter,
                f"{row:02d}",
            )

        for seat, seat_rect in seat_rects:
            selected = seat == self._selected_seat
            painter.setPen(QPen(QColor("#83e7d8" if selected else "#3a3a3a"), 1.4))
            painter.setBrush(QColor("#5fe4d4" if selected else "#3c3c3c"))
            painter.drawRoundedRect(seat_rect, 9, 9)

        if self._selected_seat and not self._taking_off:
            self._draw_task_panel(painter)
        painter.restore()

    def _draw_task_panel(self, painter: QPainter) -> None:
        panel = self._task_panel_rect()
        painter.save()
        painter.setPen(QPen(QColor(255, 255, 255, 48), 1))
        painter.setBrush(QColor(80, 80, 80, 220))
        painter.drawRoundedRect(panel, 18, 18)

        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        painter.setPen(QColor(225, 228, 232, 170))
        painter.drawText(QRectF(panel.left() + 22, panel.top() + 16, panel.width() - 44, 24), f"Vị trí tập trung: {self._selected_seat}")
        painter.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(panel.left() + 22, panel.top() + 42, panel.width() - 44, 24), "Chọn nhóm nhiệm vụ cho phiên")
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        painter.setPen(QColor(225, 228, 232, 165))
        painter.drawText(
            QRectF(panel.left() + 22, panel.top() + 66, panel.width() - 44, 22),
            "Ghế là mốc cá nhân hoá hành trình, không ảnh hưởng điểm.",
        )

        for label, value, color, rect in self._task_chip_rects():
            selected = value == self._selected_task_type
            accent = QColor(color)
            bg = QColor(accent)
            bg.setAlpha(70 if not selected else 105)
            painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 120), 1.2))
            painter.setBrush(bg)
            painter.drawRoundedRect(rect, 18, 18)
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
            painter.setPen(QColor(accent.red(), accent.green(), accent.blue(), 235))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()


class _BaseContextDialog(QDialog):
    """Frameless, draggable dialog shell shared by all context dialogs."""

    def __init__(
        self,
        title: str,
        subtitle: str,
        *,
        config: Optional[dict] = None,
        min_width: int = 520,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._is_dark = _is_dark_mode(config)
        self._drag_pos: Optional[QPoint] = None

        self.setModal(True)
        self.setWindowTitle(title)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumWidth(min_width)

        # Outer drop shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 80 if self._is_dark else 50))
        self.setGraphicsEffect(shadow)

        # Root layout wraps everything in a styled container
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)  # shadow breathing room
        outer.setSpacing(0)

        self._container = QFrame()
        self._container.setObjectName("dialogContainer")
        outer.addWidget(self._container)

        self._root = QVBoxLayout(self._container)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(0)

        self._build_header(title, subtitle)
        self.setStyleSheet(get_stylesheet(self._is_dark) + _make_dialog_stylesheet(self._is_dark))

    # ── Header ──────────────────────────────────────────────────────────────

    def _build_header(self, title: str, subtitle: str) -> None:
        header = QFrame()
        header.setObjectName("dialogHeader")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(20, 14, 12, 14)
        h_layout.setSpacing(8)

        text_col = QVBoxLayout()
        text_col.setSpacing(3)

        lbl_title = QLabel(title)
        lbl_title.setObjectName("dialogTitle")
        lbl_title.setWordWrap(True)
        text_col.addWidget(lbl_title)

        # Keep the header minimal like Settings: title only.
        _ = subtitle

        h_layout.addLayout(text_col, 1)

        close_btn = QPushButton("✕")
        close_btn.setObjectName("closeXButton")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.reject)
        h_layout.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignTop)

        self._root.addWidget(header)

    # ── Form card factory ────────────────────────────────────────────────────

    def _make_form_card(self) -> tuple[QFrame, QVBoxLayout]:
        """Return (card_frame, card_layout) for placing form rows inside."""
        card = QFrame()
        card.setObjectName("formCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(0)
        return card, layout

    def _add_row(
        self,
        card_layout: QVBoxLayout,
        label_text: str,
        widget: QWidget,
        *,
        first: bool = False,
    ) -> None:
        """Append a clean label+widget row without separator lines."""
        _ = first

        row = QHBoxLayout()
        row.setContentsMargins(0, 8, 0, 8)
        row.setSpacing(12)

        lbl = QLabel(label_text)
        lbl.setObjectName("rowLabel")
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(lbl, 0)
        row.addWidget(widget, 1)

        card_layout.addLayout(row)

    def _add_full_row(
        self,
        card_layout: QVBoxLayout,
        widget: QWidget,
        *,
        first: bool = False,
        top_margin: int = 8,
        bottom_margin: int = 8,
    ) -> None:
        """Add a full-width widget without separator lines."""
        _ = first

        row = QHBoxLayout()
        row.setContentsMargins(0, top_margin, 0, bottom_margin)
        row.addWidget(widget)
        card_layout.addLayout(row)

    # ── Footer ───────────────────────────────────────────────────────────────

    def _make_footer(self, cancel_text: str, confirm_text: str) -> tuple[QPushButton, QPushButton]:
        """Build footer row; returns (cancel_btn, confirm_btn)."""
        footer = QHBoxLayout()
        footer.setContentsMargins(20, 12, 20, 16)
        footer.setSpacing(8)
        footer.addStretch(1)

        cancel_btn = QPushButton(cancel_text)
        cancel_btn.setObjectName("ghostButton")
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        footer.addWidget(cancel_btn)

        confirm_btn = QPushButton(confirm_text)
        confirm_btn.setObjectName("primaryButton")
        confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        confirm_btn.clicked.connect(self.accept)
        confirm_btn.setDefault(True)
        footer.addWidget(confirm_btn)

        self._root.addLayout(footer)
        return cancel_btn, confirm_btn

    # ── Drag support ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    # ── Convenience: styled combobox with custom arrow painting ──────────────

    @staticmethod
    def _make_combo(items: list[tuple[str, str]]) -> QComboBox:
        """items = [(display_label, value), ...]"""
        combo = QComboBox()
        combo.setCursor(Qt.CursorShape.PointingHandCursor)
        for label, value in items:
            combo.addItem(label, value)
        return combo

    @staticmethod
    def _make_spinbox(lo: int, hi: int, suffix: str, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setRange(lo, hi)
        spin.setSuffix(suffix)
        spin.setValue(value)
        spin.setAlignment(Qt.AlignmentFlag.AlignLeft)
        return spin

    @staticmethod
    def _make_hint(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("hintLabel")
        lbl.setWordWrap(True)
        return lbl

    # ── Body padding ─────────────────────────────────────────────────────────

    def _add_body_padding(self, top: int = 14, bottom: int = 4) -> None:
        self._root.setContentsMargins(0, 0, 0, 0)
        body_wrap = QVBoxLayout()
        body_wrap.setContentsMargins(16, top, 16, bottom)
        body_wrap.setSpacing(10)
        self._body = body_wrap
        self._root.addLayout(body_wrap)

    def _body_add(self, widget: QWidget) -> None:
        self._body.addWidget(widget)


# ---------------------------------------------------------------------------
# SessionContextDialog
# ---------------------------------------------------------------------------

class LegacySessionContextDialog(_BaseContextDialog):
    """Legacy context dialog kept for reference while Focus Journey is active."""

    TASK_TYPES = (
        ("Học tập",           "study"),
        ("Lập trình",         "coding"),
        ("Viết lách",         "writing"),
        ("Đọc tài liệu",      "reading"),
        ("Sáng tạo",          "creative"),
        ("Làm việc sâu",      "deep_work"),
        ("Ôn tập / xem lại", "review"),
        ("Việc hành chính",  "admin"),
        ("Khác",              "other"),
    )

    SESSION_MODES = (
        ("Bình thường",   "normal"),
        ("Deep Focus",    "deep"),
        ("Deadline",      "deadline"),
    )

    def __init__(self, *, config: Optional[dict] = None, parent: Optional[QWidget] = None):
        super().__init__(
            "Thiết lập phiên làm việc",
            "",
            config=config,
            min_width=560,
            parent=parent,
        )
        self._config = dict(config or {})
        self._build_ui()

    def _build_ui(self) -> None:
        self._add_body_padding(top=14, bottom=4)

        card, c_layout = self._make_form_card()

        # Mục tiêu phiên
        self.goal_input = QLineEdit()
        self.goal_input.setPlaceholderText("Ví dụ: hoàn thành module báo cáo tuần")
        self._add_row(c_layout, "Mục tiêu phiên", self.goal_input, first=True)

        # Loại công việc
        self.task_type_combo = self._make_combo(list(self.TASK_TYPES))
        self._add_row(c_layout, "Loại công việc", self.task_type_combo)

        # Chế độ phiên
        self.session_mode_combo = self._make_combo(list(self.SESSION_MODES))
        self.session_mode_combo.currentIndexChanged.connect(self._sync_mode_state)
        self._add_row(c_layout, "Chế độ", self.session_mode_combo)

        # Thời lượng dự kiến
        default_planned = int(self._config.get("deadline_focus_minutes", 45) or 45)
        self.planned_minutes_spin = self._make_spinbox(10, 240, " phút", default_planned)
        self._add_row(c_layout, "Thời lượng dự kiến", self.planned_minutes_spin)

        # Khung deadline (chỉ hiện khi mode = deadline)
        default_deadline = int(self._config.get("deadline_focus_minutes", 45) or 45)
        self.deadline_minutes_spin = self._make_spinbox(10, 180, " phút", default_deadline)
        self._deadline_row_label = QLabel("Khung deadline")
        self._deadline_row_label.setObjectName("rowLabel")
        self._deadline_row_widget = self.deadline_minutes_spin
        self._add_row(c_layout, "Khung deadline", self.deadline_minutes_spin)

        # Ghi chú
        self.note_input = QLineEdit()
        self.note_input.setPlaceholderText("Ghi chú ngắn (tuỳ chọn)")
        self._add_row(c_layout, "Ghi chú", self.note_input)

        # Mode hint
        self._mode_hint = self._make_hint("")
        c_layout.addWidget(self._mode_hint)

        self._body.addWidget(card)

        self._make_footer("Bỏ qua", "Tiếp theo")
        self._sync_mode_state()

        # Pre-select mode from config
        saved_mode = str(self._config.get("session_mode", "normal"))
        for i in range(self.session_mode_combo.count()):
            if self.session_mode_combo.itemData(i) == saved_mode:
                self.session_mode_combo.setCurrentIndex(i)
                break

        if bool(self._config.get("deadline_mode_enabled", False)):
            for i in range(self.session_mode_combo.count()):
                if self.session_mode_combo.itemData(i) == "deadline":
                    self.session_mode_combo.setCurrentIndex(i)
                    break

    def _sync_mode_state(self) -> None:
        mode = str(self.session_mode_combo.currentData() or "normal")
        is_deadline = mode == "deadline"
        self.deadline_minutes_spin.setEnabled(is_deadline)
        self.deadline_minutes_spin.setVisible(is_deadline)

        hints = {
            "normal": "Theo dõi nhịp làm việc bình thường, nghỉ theo khuyến nghị cá nhân hóa.",
            "deep": "UI tối giản, ít cảnh báo hơn. Chỉ thông báo khi rủi ro cao.",
            "deadline": "Theo dõi sát hơn, cảnh báo sớm hơn khi gần hết thời gian.",
        }
        if self._mode_hint is not None:
            self._mode_hint.setText(hints.get(mode, ""))

    def get_payload(self) -> Dict[str, Any]:
        mode = str(self.session_mode_combo.currentData() or "normal")
        return {
            "goal": str(self.goal_input.text() or "").strip(),
            "task_type": str(self.task_type_combo.currentData() or "deep_work"),
            "session_mode": mode,
            "planned_minutes": int(self.planned_minutes_spin.value()),
            "deadline_mode": mode == "deadline",
            "deadline_minutes": int(self.deadline_minutes_spin.value()) if mode == "deadline" else 0,
            "note": str(self.note_input.text() or "").strip(),
        }


# ---------------------------------------------------------------------------
# Focus Journey Route Selector
# ---------------------------------------------------------------------------

class SessionContextDialog(_BaseContextDialog):
    """Collect session context, duration source, and symbolic focus route."""

    TASK_TYPES = (
        ("Học tập", "study"),
        ("Lập trình", "coding"),
        ("Viết lách", "writing"),
        ("Đọc tài liệu", "reading"),
        ("Sáng tạo", "creative"),
        ("Làm việc sâu", "deep_work"),
        ("Ôn tập / xem lại", "review"),
        ("Việc hành chính", "admin"),
        ("Khác", "other"),
    )

    SESSION_MODES = (
        ("Bình thường", "normal"),
        ("Deep Focus", "deep"),
        ("Deadline", "deadline"),
    )

    def __init__(self, *, config: Optional[dict] = None, parent: Optional[QWidget] = None):
        super().__init__("Thiết lập phiên làm việc", "", config=config, min_width=760, parent=parent)
        self._config = dict(config or {})
        self.setGraphicsEffect(None)
        self.duration_source = "personalized"
        self.recommended_minutes = self._recommended_work_minutes()
        self.default_task_type = str(self._config.get("task_type", "deep_work") or "deep_work")
        self.current_origin_code = _configured_focus_origin(self._config)
        self.selected_route: Dict[str, Any] = {}
        self._route_buttons: List[QPushButton] = []
        self._displayed_routes: List[Dict[str, Any]] = []

        if self.layout() is not None:
            self.layout().setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        self._build_ui()

    def _recommended_work_minutes(self) -> int:
        for key in ("recommended_work_minutes", "work_minutes", "break_interval_minutes", "deadline_focus_minutes"):
            try:
                value = int(float(self._config.get(key, 0) or 0))
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                return max(30, min(90, value))
        return 35

    def _build_ui(self) -> None:
        self._add_body_padding(top=14, bottom=4)
        card, c_layout = self._make_form_card()

        self.goal_input = QLineEdit()
        self.goal_input.setPlaceholderText("Ví dụ: hoàn thành module báo cáo tuần")
        self._add_row(c_layout, "Mục tiêu phiên", self.goal_input, first=True)

        self.session_mode_combo = self._make_combo(list(self.SESSION_MODES))
        self.session_mode_combo.currentIndexChanged.connect(self._sync_mode_state)
        self._add_row(c_layout, "Chế độ", self.session_mode_combo)

        duration_box = QWidget()
        duration_layout = QHBoxLayout(duration_box)
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.setSpacing(8)
        self.personalized_btn = QPushButton("Cá nhân hóa")
        self.custom_btn = QPushButton("Tự chọn")
        for btn in (self.personalized_btn, self.custom_btn):
            btn.setObjectName("ghostButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            duration_layout.addWidget(btn)
        self.personalized_btn.clicked.connect(lambda: self._set_duration_source("personalized"))
        self.custom_btn.clicked.connect(lambda: self._set_duration_source("custom"))
        self._add_row(c_layout, "Chọn thời lượng", duration_box)

        self.custom_duration_panel = QWidget()
        custom_layout = QVBoxLayout(self.custom_duration_panel)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.setSpacing(8)

        self.recommendation_label = QLabel("")
        self.recommendation_label.setObjectName("mutedLabel")
        custom_layout.addWidget(self.recommendation_label)

        slider_box = QWidget()
        slider_layout = QHBoxLayout(slider_box)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(10)
        self.duration_slider = DurationTimelineSlider(is_dark=self._is_dark)
        self.duration_slider.setRange(30, 120)
        self.duration_slider.setSingleStep(5)
        self.duration_slider.setPageStep(5)
        self.duration_slider.setTickInterval(5)
        self.planned_minutes_spin = self._make_spinbox(30, 120, " phút", self.recommended_minutes)
        self.planned_minutes_spin.setSingleStep(5)
        self.planned_minutes_spin.setFixedWidth(96)
        self.planned_minutes_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_layout.addWidget(self.duration_slider, 1)
        slider_layout.addWidget(self.planned_minutes_spin)
        custom_layout.addWidget(slider_box)
        self.duration_slider.valueChanged.connect(self._on_duration_slider_changed)
        self.planned_minutes_spin.valueChanged.connect(self._on_duration_spin_changed)

        self.deadline_minutes_spin = self._make_spinbox(
            30, 120, " phút", max(30, int(self._config.get("deadline_focus_minutes", 45) or 45))
        )
        self.deadline_minutes_spin.setSingleStep(5)
        self.deadline_row = QWidget()
        deadline_layout = QHBoxLayout(self.deadline_row)
        deadline_layout.setContentsMargins(0, 8, 0, 8)
        deadline_layout.setSpacing(12)
        deadline_label = QLabel("Khung deadline")
        deadline_label.setObjectName("rowLabel")
        deadline_layout.addWidget(deadline_label, 0)
        deadline_layout.addWidget(self.deadline_minutes_spin, 1)
        c_layout.addWidget(self.deadline_row)

        origin_name = FOCUS_AIRPORT_DATA.get(self.current_origin_code, {}).get("name", self.current_origin_code)
        route_title = QLabel(f"Chọn chuyến làm việc từ {self.current_origin_code} - {origin_name}")
        self.route_title = route_title
        self.route_title.setObjectName("rowLabel")
        self.route_title.setContentsMargins(0, 4, 0, 2)
        custom_layout.addWidget(self.route_title)
        self.route_list = QWidget()
        self.route_list.setMinimumHeight(150)
        self.route_list.setMaximumHeight(156)
        self.route_layout = QHBoxLayout(self.route_list)
        self.route_layout.setContentsMargins(0, 2, 0, 0)
        self.route_layout.setSpacing(12)
        custom_layout.addWidget(self.route_list)
        for index in range(3):
            btn = FocusRouteCardButton(is_dark=self._is_dark)
            btn.setObjectName("routeCard")
            btn.clicked.connect(lambda checked=False, i=index: self._select_route_by_index(i))
            self.route_layout.addWidget(btn)
            self._route_buttons.append(btn)
        c_layout.addWidget(self.custom_duration_panel)

        self.note_input = QLineEdit()
        self._mode_hint = self._make_hint("")
        self._mode_hint.hide()
        self._body.addWidget(card)
        self._make_footer("Bỏ qua", "Tiếp theo")

        saved_mode = str(self._config.get("session_mode", "normal"))
        for i in range(self.session_mode_combo.count()):
            if self.session_mode_combo.itemData(i) == saved_mode:
                self.session_mode_combo.setCurrentIndex(i)
                break
        if bool(self._config.get("deadline_mode_enabled", False)):
            for i in range(self.session_mode_combo.count()):
                if self.session_mode_combo.itemData(i) == "deadline":
                    self.session_mode_combo.setCurrentIndex(i)
                    break

        self._sync_mode_state()
        self._set_duration_source(self.duration_source)
        self._set_duration_value(self.recommended_minutes)

    def _sync_mode_state(self) -> None:
        mode = str(self.session_mode_combo.currentData() or "normal")
        is_deadline = mode == "deadline"
        self.deadline_minutes_spin.setEnabled(is_deadline)
        self.deadline_row.setVisible(is_deadline)

        max_minutes = 120
        self.duration_slider.setMaximum(max_minutes)
        self.planned_minutes_spin.setMaximum(max_minutes)
        if self.planned_minutes_spin.value() > max_minutes:
            self._set_duration_value(max_minutes)
        self._mode_hint.setText({
            "normal": "Theo dõi nhịp làm việc bình thường, nghỉ theo khuyến nghị cá nhân hóa.",
            "deep": "UI tối giản, ít cảnh báo hơn. Chỉ thông báo khi rủi ro cao.",
            "deadline": "Theo dõi sát hơn, cảnh báo sớm hơn khi gần hết thời gian.",
        }.get(mode, ""))
        if is_deadline:
            self.deadline_minutes_spin.setValue(int(self.planned_minutes_spin.value()))
        self._refresh_routes()

    def _round_duration(self, value: int) -> int:
        return _round_minutes_to_five(value, maximum=self.duration_slider.maximum())

    def _set_duration_source(self, source: str) -> None:
        self.duration_source = "custom" if source == "custom" else "personalized"
        self.personalized_btn.setChecked(self.duration_source == "personalized")
        self.custom_btn.setChecked(self.duration_source == "custom")
        self.duration_slider.setEnabled(self.duration_source == "custom")
        self.planned_minutes_spin.setEnabled(self.duration_source == "custom")
        if self.duration_source == "personalized":
            self._set_duration_value(self.recommended_minutes)
        if hasattr(self, "custom_duration_panel"):
            self.custom_duration_panel.setVisible(self.duration_source == "custom")
        self.recommendation_label.setText(f"Gợi ý cho bạn: {int(self.recommended_minutes)} phút")
        self._refresh_routes()

    def _set_duration_value(self, minutes: int) -> None:
        value = self._round_duration(minutes)
        self.duration_slider.blockSignals(True)
        self.planned_minutes_spin.blockSignals(True)
        self.duration_slider.setValue(value)
        self.planned_minutes_spin.setValue(value)
        self.duration_slider.blockSignals(False)
        self.planned_minutes_spin.blockSignals(False)
        if str(self.session_mode_combo.currentData() or "normal") == "deadline":
            self.deadline_minutes_spin.setValue(value)
        self._refresh_routes()

    def _on_duration_slider_changed(self, value: int) -> None:
        self._set_duration_value(value)

    def _on_duration_spin_changed(self, value: int) -> None:
        self._set_duration_value(value)

    def _refresh_routes(self) -> None:
        if not hasattr(self, "_route_buttons"):
            return

        routes = _nearest_focus_routes(
            int(self.planned_minutes_spin.value()),
            limit=3,
            from_code=self.current_origin_code,
        )
        self._displayed_routes = [dict(route) for route in routes]
        if not routes:
            self.selected_route = {}
            for btn in self._route_buttons:
                btn.hide()
            return
        if not self.selected_route or self.selected_route.get("route_id") not in {r["route_id"] for r in routes}:
            target_minutes = int(self.planned_minutes_spin.value())
            self.selected_route = dict(min(
                routes,
                key=lambda route: (
                    abs(int(route.get("duration_minutes", 0) or 0) - target_minutes),
                    int(route.get("duration_minutes", 0) or 0),
                    str(route.get("to_code", "")),
                ),
            ))

        for index, btn in enumerate(self._route_buttons):
            if index >= len(routes):
                btn.hide()
                continue
            route = routes[index]
            if isinstance(btn, FocusRouteCardButton):
                btn.set_route(route)
            btn.setChecked(route.get("route_id") == self.selected_route.get("route_id"))
            btn.show()

    def _select_route_by_index(self, index: int) -> None:
        if 0 <= index < len(self._displayed_routes):
            self._select_route(self._displayed_routes[index])

    def _select_route(self, route: Dict[str, Any]) -> None:
        self.selected_route = dict(route)
        selected_id = self.selected_route.get("route_id")
        for index, btn in enumerate(self._route_buttons):
            route_id = ""
            if index < len(self._displayed_routes):
                route_id = self._displayed_routes[index].get("route_id", "")
            btn.setChecked(route_id == selected_id)

    def get_payload(self) -> Dict[str, Any]:
        mode = str(self.session_mode_combo.currentData() or "normal")
        journey_enabled = self.duration_source == "custom"
        route = dict(self.selected_route or {}) if journey_enabled else {}
        return {
            "goal": str(self.goal_input.text() or "").strip(),
            "task_type": self.default_task_type,
            "session_mode": mode,
            "planned_minutes": int(self.planned_minutes_spin.value()),
            "deadline_mode": mode == "deadline",
            "deadline_minutes": int(self.deadline_minutes_spin.value()) if mode == "deadline" else 0,
            "note": "",
            "duration_source": self.duration_source,
            "journey_enabled": journey_enabled,
            "journey_origin_code": self.current_origin_code,
            "selected_route_id": str(route.get("route_id", "")),
            "selected_route_label": str(route.get("short_label", "")),
            "route_from_code": str(route.get("from_code", "")),
            "route_to_code": str(route.get("to_code", "")),
            "route_from_name": str(route.get("from_name", "")),
            "route_to_name": str(route.get("to_name", "")),
            "route_duration_minutes": int(route.get("duration_minutes", 0) or 0),
            "route_distance_km": int(route.get("route_distance_km", 0) or 0),
            "route_theme": str(route.get("route_theme", "")),
        }


# ---------------------------------------------------------------------------
# ContextCheckInDialog  (unchanged logic, minor polish only)
# ---------------------------------------------------------------------------

class ContextCheckInDialog(_BaseContextDialog):
    """Quick self-report for real-time recovery validation."""

    CHECKIN_OPTIONS = (
        ("Đang đúng task",             "on_task"),
        ("Lệch nhẹ, có thể quay lại", "slight_drift"),
        ("Lệch khỏi nhiệm vụ",        "off_task"),
        ("Mệt, cần nghỉ ngắn",        "need_break"),
    )

    def __init__(
        self,
        *,
        risk_score: float,
        state_name: str,
        config: Optional[dict] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            "Check-in nhanh",
            "Hệ thống ghi nhận rủi ro lệch khỏi nhiệm vụ — bạn đang ở trạng thái nào?",
            config=config,
            min_width=460,
            parent=parent,
        )
        self._risk_score = max(0.0, min(1.0, float(risk_score or 0.0)))
        self._state_name = str(state_name or "unknown")
        self._build_ui()

    def _build_ui(self) -> None:
        self._add_body_padding(top=12, bottom=4)

        card, c_layout = self._make_form_card()

        self.answer_combo = self._make_combo(list(self.CHECKIN_OPTIONS))
        self._add_row(c_layout, "Tình trạng hiện tại", self.answer_combo, first=True)

        self.note_input = QLineEdit()
        self.note_input.setPlaceholderText("Ghi chú ngắn (tuỳ chọn)")
        self._add_row(c_layout, "Ghi chú", self.note_input)

        self._body_add(card)

        self._make_footer("Để sau", "Xác nhận")

    def get_payload(self) -> Dict[str, Any]:
        return {
            "answer": str(self.answer_combo.currentData() or "on_task"),
            "answer_label": str(self.answer_combo.currentText() or ""),
            "note": str(self.note_input.text() or "").strip(),
            "risk_score": float(self._risk_score),
            "state_name": str(self._state_name),
        }


# ---------------------------------------------------------------------------
# SessionExitDialog
# ---------------------------------------------------------------------------

class SessionExitDialog(_BaseContextDialog):
    """Collect end-of-session reason, quick rating, and show a brief session summary."""

    EXIT_REASONS = (
        ("Hoàn thành mục tiêu",   "goal_completed"),
        ("Tạm dừng thủ công",     "manual_stop"),
        ("Chuyển task gấp",       "urgent_switch"),
        ("Mệt / giảm năng lượng", "fatigue"),
        ("Lý do kỹ thuật",        "technical"),
    )

    RATINGS = (
        ("1\nRất thấp",  1),
        ("2\nThấp",      2),
        ("3\nTrung bình", 3),
        ("4\nTốt",       4),
        ("5\nRất tốt",   5),
    )

    def __init__(
        self,
        *,
        config: Optional[dict] = None,
        session_summary: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            "Kết thúc phiên",
            "",
            config=config,
            min_width=600,
            parent=parent,
        )
        self._session_summary = session_summary or {}
        self._reason_by_id: Dict[int, tuple[str, str]] = {}
        self._rating_by_id: Dict[int, tuple[str, int]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        self._add_body_padding(top=14, bottom=4)

        # Session summary card (shown when data is available)
        if self._session_summary:
            summary_card, s_layout = self._make_form_card()
            self._build_summary_rows(s_layout)
            self._body_add(summary_card)

        card, c_layout = self._make_form_card()

        reason_title = QLabel("Lý do kết thúc")
        reason_title.setObjectName("rowLabel")
        c_layout.addWidget(reason_title)

        self.reason_group = QButtonGroup(self)
        self.reason_group.setExclusive(True)
        reason_grid = QGridLayout()
        reason_grid.setContentsMargins(0, 8, 0, 10)
        reason_grid.setHorizontalSpacing(8)
        reason_grid.setVerticalSpacing(8)
        for idx, (label, value) in enumerate(self.EXIT_REASONS):
            btn = QPushButton(label)
            btn.setObjectName("optionPill")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.reason_group.addButton(btn, idx)
            self._reason_by_id[idx] = (label, value)
            reason_grid.addWidget(btn, idx // 2, idx % 2)
        first_reason = self.reason_group.button(0)
        if first_reason is not None:
            first_reason.setChecked(True)
        c_layout.addLayout(reason_grid)

        rating_title = QLabel("Mức sẵn sàng sau phiên")
        rating_title.setObjectName("rowLabel")
        c_layout.addWidget(rating_title)

        self.rating_group = QButtonGroup(self)
        self.rating_group.setExclusive(True)
        rating_row = QHBoxLayout()
        rating_row.setContentsMargins(0, 8, 0, 10)
        rating_row.setSpacing(8)
        for idx, (label, value) in enumerate(self.RATINGS):
            btn = QPushButton(label)
            btn.setObjectName("ratingPill")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.rating_group.addButton(btn, idx)
            self._rating_by_id[idx] = (label, value)
            rating_row.addWidget(btn)
        default_rating = self.rating_group.button(2)
        if default_rating is not None:
            default_rating.setChecked(True)
        c_layout.addLayout(rating_row)

        self.note_input = QLineEdit()
        self.note_input.setPlaceholderText("Ghi chú ngắn (tuỳ chọn)")
        self._add_row(c_layout, "Ghi chú", self.note_input)

        self._body_add(card)
        self._make_footer("Bỏ qua", "Lưu")

    def _build_summary_rows(self, layout: QVBoxLayout) -> None:
        s = self._session_summary
        is_dark = self._is_dark

        header = QLabel("Tóm tắt phiên")
        header.setObjectName("sectionTitle")
        layout.addWidget(header)

        def _row(label: str, value: str) -> None:
            row = QHBoxLayout()
            row.setContentsMargins(0, 4, 0, 4)
            lbl = QLabel(label)
            lbl.setObjectName("rowLabel")
            val = QLabel(value)
            val.setObjectName("trendValue")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lbl, 1)
            row.addWidget(val)
            layout.addLayout(row)

        duration_s = int(s.get("session_seconds", 0) or 0)
        m, sec = divmod(duration_s, 60)
        h, m = divmod(m, 60)
        if h:
            dur_str = f"{h}g {m}p {sec}s"
        else:
            dur_str = f"{m}p {sec}s"
        _row("Thời gian phiên", dur_str)

        focus_s = int(s.get("focus_seconds", 0) or 0)
        fm, fs = divmod(focus_s, 60)
        _row("Thời gian làm việc ổn định", f"{fm}p {fs}s")

        avg_score = float(s.get("avg_score", 0.0) or 0.0)
        _row("Mức sẵn sàng TB", f"{avg_score:.0f}")

        distraction = int(s.get("distraction_count", 0) or 0)
        _row("Số lần lệch nhịp", str(distraction))

        suggestion = str(s.get("next_session_suggestion", "") or "")
        if suggestion:
            hint = QLabel(suggestion)
            hint.setObjectName("hintLabel")
            hint.setWordWrap(True)
            layout.addWidget(hint)

    def get_payload(self) -> Dict[str, Any]:
        reason_label, reason_value = self._reason_by_id.get(
            self.reason_group.checkedId(),
            self.EXIT_REASONS[1],
        )
        rating_label, rating_value = self._rating_by_id.get(
            self.rating_group.checkedId(),
            self.RATINGS[2],
        )
        return {
            "reason": str(reason_value or "manual_stop"),
            "reason_label": str(reason_label or ""),
            "focus_rating": int(rating_value or 3),
            "focus_rating_label": str(rating_label or ""),
            "note": str(self.note_input.text() or "").strip(),
        }


# ---------------------------------------------------------------------------
# SessionBoardingPassDialog — "thẻ phiên" hiển thị sau khi xác nhận setup
# ---------------------------------------------------------------------------

class LegacySessionBoardingPassDialog(_BaseContextDialog):
    """Legacy boarding pass kept for reference while Focus Journey is active."""

    _TASK_TYPE_LABELS: Dict[str, str] = {
        "study":     "Học tập",
        "coding":    "Lập trình",
        "writing":   "Viết lách",
        "reading":   "Đọc tài liệu",
        "creative":  "Sáng tạo",
        "deep_work": "Làm việc sâu",
        "review":    "Ôn tập",
        "admin":     "Việc hành chính",
        "other":     "Khác",
    }

    _MODE_LABELS: Dict[str, str] = {
        "normal":   "Bình thường",
        "deep":     "Deep Focus",
        "deadline": "Deadline",
    }

    def __init__(
        self,
        *,
        context_payload: Dict[str, Any],
        config: Optional[dict] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            "Hành trình làm việc",
            "",
            config=config,
            min_width=480,
            parent=parent,
        )
        self._ctx = context_payload
        self._build_ui()

    def _build_ui(self) -> None:
        self._add_body_padding(top=16, bottom=8)

        card, c_layout = self._make_form_card()

        now = datetime.now()
        planned_min = int(self._ctx.get("planned_minutes", 0) or 0)
        if planned_min > 0:
            end_ts = now.timestamp() + planned_min * 60
            end_str = datetime.fromtimestamp(end_ts).strftime("%H:%M")
        else:
            end_str = "—"

        goal = str(self._ctx.get("goal", "") or "").strip() or "Không đặt mục tiêu"
        task_type = self._TASK_TYPE_LABELS.get(
            str(self._ctx.get("task_type", "") or ""), "Không rõ"
        )
        mode = self._MODE_LABELS.get(
            str(self._ctx.get("session_mode", "normal") or "normal"), "Bình thường"
        )

        def _row(label: str, value: str, bold_value: bool = False) -> None:
            row = QHBoxLayout()
            row.setContentsMargins(0, 6, 0, 6)
            lbl = QLabel(label)
            lbl.setObjectName("rowLabel")
            val = QLabel(value)
            val.setObjectName("trendValue" if bold_value else "mutedLabel")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val.setWordWrap(True)
            row.addWidget(lbl, 1)
            row.addWidget(val, 1)
            c_layout.addLayout(row)

        _row("Mục tiêu", goal, bold_value=True)
        _row("Loại công việc", task_type)
        _row("Chế độ", mode)
        if planned_min > 0:
            _row("Thời lượng dự kiến", f"{planned_min} phút")
        _row("Bắt đầu", now.strftime("%H:%M"))
        if planned_min > 0:
            _row("Dự kiến kết thúc", end_str)

        mode_val = str(self._ctx.get("session_mode", "normal") or "normal")
        if mode_val == "deep":
            hint_text = "Deep Focus: UI tối giản, ít cảnh báo. Chỉ thông báo khi rủi ro cao."
        elif mode_val == "deadline":
            hint_text = "Deadline mode: theo dõi sát, cảnh báo sớm khi gần hết thời gian."
        else:
            hint_text = "Theo dõi nhịp làm việc và nghỉ theo khuyến nghị cá nhân hóa."

        hint = self._make_hint(hint_text)
        c_layout.addWidget(hint)

        self._body_add(card)
        self._make_footer("Huỷ", "Bắt đầu hành trình")


# ---------------------------------------------------------------------------
# Focus Journey Boarding Pass
# ---------------------------------------------------------------------------

class SessionBoardingPassDialog(_BaseContextDialog):
    """Show a compact symbolic boarding pass before tracking starts."""

    _TASK_TYPE_LABELS: Dict[str, str] = {
        "study": "Học tập",
        "coding": "Lập trình",
        "writing": "Viết lách",
        "reading": "Đọc tài liệu",
        "creative": "Sáng tạo",
        "deep_work": "Làm việc sâu",
        "review": "Ôn tập",
        "admin": "Việc hành chính",
        "other": "Khác",
    }

    _MODE_LABELS: Dict[str, str] = {
        "normal": "Bình thường",
        "deep": "Deep Focus",
        "deadline": "Deadline",
    }

    def __init__(
        self,
        *,
        context_payload: Dict[str, Any],
        config: Optional[dict] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__("Hành trình làm việc", "", config=config, min_width=620, parent=parent)
        self._ctx = context_payload
        header = self._root.itemAt(0).widget() if self._root.count() else None
        if header is not None:
            header.hide()
        self.setGraphicsEffect(None)
        self._container.setStyleSheet("QFrame#dialogContainer { background: transparent; border: none; }")
        outer = self.layout()
        if outer is not None:
            outer.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(900, 760)
        self.resize(980, 820)
        self._build_ui()

    def _build_ui(self) -> None:
        self._add_body_padding(top=0, bottom=0)
        self.seat_widget = PlaneSeatSelectionWidget(is_dark=self._is_dark, parent=self)
        self.seat_widget.seatSelected.connect(self._store_selected_seat)
        self.seat_widget.taskTypeSelected.connect(self._store_selected_task_type)
        self.seat_widget.takeoffFinished.connect(self.accept)
        self._body_add(self.seat_widget)
        self.seat_widget.setFocus()

    def _store_selected_seat(self, seat: str) -> None:
        self._ctx["selected_seat"] = seat

    def _store_selected_task_type(self, task_type: str) -> None:
        self._ctx["task_type"] = task_type


# ---------------------------------------------------------------------------
# SessionHabitReportDialog — báo cáo chi tiết sau phiên
# ---------------------------------------------------------------------------

class SessionHabitReportDialog(_BaseContextDialog):
    """Show a detailed post-session habit report with trends and suggestions.

    Uses a custom title bar with macOS-style window dots (close / minimize /
    maximize-restore) instead of the native dialog header.
    """

    def __init__(
        self,
        *,
        habit_report: Dict[str, Any],
        config: Optional[dict] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            "Kết quả nhịp làm việc",
            "",
            config=config,
            min_width=540,
            parent=parent,
        )
        self._report = habit_report
        self._build_ui()

    # ── Override header with custom title bar ──────────────────────────────

    def _build_header(self, title: str, subtitle: str) -> None:
        """Replace the default dialog header with a custom 3-dot title bar."""
        _ = subtitle  # unused
        self._title_bar = DialogTitleBar(
            title,
            is_dark=self._is_dark,
            parent=self._container,
        )
        self._root.addWidget(self._title_bar)

    def _build_ui(self) -> None:
        self._add_body_padding(top=14, bottom=8)

        r = self._report

        # ── Tổng quan ──
        overview_card, ov_layout = self._make_form_card()
        ov_header = QLabel("Tổng quan phiên")
        ov_header.setObjectName("sectionTitle")
        ov_layout.addWidget(ov_header)

        def _row(layout: QVBoxLayout, label: str, value: str) -> None:
            row = QHBoxLayout()
            row.setContentsMargins(0, 5, 0, 5)
            lbl = QLabel(label)
            lbl.setObjectName("rowLabel")
            val = QLabel(value)
            val.setObjectName("trendValue")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lbl, 1)
            row.addWidget(val)
            layout.addLayout(row)

        duration_s = int(r.get("session_seconds", 0) or 0)
        m, sec = divmod(duration_s, 60)
        h, m = divmod(m, 60)
        dur_str = f"{h}g {m}p" if h else f"{m}p {sec}s"
        _row(ov_layout, "Tổng thời gian", dur_str)

        eff_ratio = float(r.get("effective_work_ratio", 0.0) or 0.0)
        _row(ov_layout, "Làm việc ổn định", f"{eff_ratio:.0%}")

        avg_wr = float(r.get("avg_work_readiness", 0.0) or 0.0)
        _row(ov_layout, "Mức sẵn sàng TB", f"{avg_wr:.0f}")

        decline_min = r.get("decline_start_minutes")
        if decline_min is not None:
            _row(ov_layout, "Bắt đầu giảm ổn định", f"sau {decline_min:.0f} phút")

        self._body_add(overview_card)

        # ── Xu hướng ──
        trend_card, tr_layout = self._make_form_card()
        tr_header = QLabel("Xu hướng")
        tr_header.setObjectName("sectionTitle")
        tr_layout.addWidget(tr_header)

        fatigue_trend = str(r.get("fatigue_trend", "") or "")
        if fatigue_trend:
            _row(tr_layout, "Mệt mỏi", fatigue_trend)

        distraction_trend = str(r.get("distraction_trend", "") or "")
        if distraction_trend:
            _row(tr_layout, "Phân tâm", distraction_trend)

        self._body_add(trend_card)

        # ── Phục hồi sau nghỉ ──
        breaks: List[Dict[str, Any]] = r.get("break_effectiveness", []) or []
        if breaks:
            break_card, br_layout = self._make_form_card()
            br_header = QLabel("Phục hồi sau nghỉ")
            br_header.setObjectName("sectionTitle")
            br_layout.addWidget(br_header)
            for i, b in enumerate(breaks[:4], 1):
                transfer = float(b.get("transfer_score", 0.0) or 0.0)
                break_type = str(b.get("break_type", "nghỉ") or "nghỉ")
                label = f"Lần {i} ({break_type})"
                if transfer >= 0.7:
                    verdict = f"Phục hồi tốt ({transfer:.0%})"
                elif transfer >= 0.4:
                    verdict = f"Trung bình ({transfer:.0%})"
                else:
                    verdict = f"Chưa phục hồi ({transfer:.0%})"
                _row(br_layout, label, verdict)
            self._body_add(break_card)

        # ── Gợi ý phiên sau ──
        suggestion = str(r.get("next_session_suggestion", "") or "")
        work_next = r.get("next_work_minutes")
        break_next = r.get("next_break_minutes")
        if suggestion or work_next:
            sug_card, sg_layout = self._make_form_card()
            sg_header = QLabel("Gợi ý phiên sau")
            sg_header.setObjectName("sectionTitle")
            sg_layout.addWidget(sg_header)
            if work_next and break_next:
                _row(sg_layout, "Thời lượng làm việc", f"{work_next} phút")
                _row(sg_layout, "Thời lượng nghỉ", f"{break_next} phút")
            if suggestion:
                hint = self._make_hint(suggestion)
                sg_layout.addWidget(hint)
            self._body_add(sug_card)

        # ── Footer ──
        footer = QHBoxLayout()
        footer.setContentsMargins(20, 12, 20, 16)
        footer.addStretch(1)
        close_btn = QPushButton("Đóng")
        close_btn.setObjectName("primaryButton")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        self._root.addLayout(footer)
