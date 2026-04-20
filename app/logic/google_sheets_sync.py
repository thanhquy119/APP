"""
Google Sheets synchronization for session analytics and per-user baselines.

This module is optional. If credentials or dependencies are missing,
the app continues to work with local JSON analytics.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


PROFILE_SCOPED_CONFIG_KEYS: tuple[str, ...] = (
    "theme_mode",
    "enable_notifications",
    "notify_distraction",
    "notify_break",
    "notify_drowsy",
    "enable_sounds",
    "volume",
    "enable_focus_audio",
    "focus_audio_track",
    "focus_audio_volume",
    "enable_break_reminders",
    "break_interval_minutes",
    "break_duration_minutes",
    "auto_break_on_distraction",
    "distraction_break_cooldown_minutes",
    "auto_resume_after_break",
    "enable_pomodoro",
    "pomodoro_work",
    "pomodoro_short_break",
    "pomodoro_long_break",
    "enable_zalo_alerts",
    "zalo_chat_id",
    "zalo_webhook_secret",
    "zalo_api_timeout_seconds",
    "zalo_alert_cooldown_minutes",
    "zalo_alert_threshold_seconds",
    "zalo_alert_on_distraction",
    "zalo_alert_on_drowsy",
    "zalo_alert_on_phone",
    "zalo_alert_on_away",
    "zalo_alert_on_break_reminder",
)

PROFILE_SCOPED_DEFAULT_SETTINGS: Dict[str, Any] = {
    "theme_mode": "light",
    "enable_notifications": True,
    "notify_distraction": True,
    "notify_break": True,
    "notify_drowsy": True,
    "enable_sounds": True,
    "volume": 70,
    "enable_focus_audio": False,
    "focus_audio_track": "rain_light",
    "focus_audio_volume": 30,
    "enable_break_reminders": True,
    "break_interval_minutes": 25,
    "break_duration_minutes": 5,
    "auto_break_on_distraction": True,
    "distraction_break_cooldown_minutes": 15,
    "auto_resume_after_break": True,
    "enable_pomodoro": False,
    "pomodoro_work": 25,
    "pomodoro_short_break": 5,
    "pomodoro_long_break": 15,
    "enable_zalo_alerts": False,
    "zalo_chat_id": "",
    "zalo_webhook_secret": "",
    "zalo_api_timeout_seconds": 8.0,
    "zalo_alert_cooldown_minutes": 10,
    "zalo_alert_threshold_seconds": 45,
    "zalo_alert_on_distraction": True,
    "zalo_alert_on_drowsy": True,
    "zalo_alert_on_phone": True,
    "zalo_alert_on_away": True,
    "zalo_alert_on_break_reminder": True,
}

_PROFILE_BOOL_KEYS: set[str] = {
    "enable_notifications",
    "notify_distraction",
    "notify_break",
    "notify_drowsy",
    "enable_sounds",
    "enable_focus_audio",
    "enable_break_reminders",
    "auto_break_on_distraction",
    "auto_resume_after_break",
    "enable_pomodoro",
    "enable_zalo_alerts",
    "zalo_alert_on_distraction",
    "zalo_alert_on_drowsy",
    "zalo_alert_on_phone",
    "zalo_alert_on_away",
    "zalo_alert_on_break_reminder",
}

_PROFILE_INT_KEYS: set[str] = {
    "volume",
    "focus_audio_volume",
    "break_interval_minutes",
    "break_duration_minutes",
    "distraction_break_cooldown_minutes",
    "pomodoro_work",
    "pomodoro_short_break",
    "pomodoro_long_break",
    "zalo_alert_cooldown_minutes",
    "zalo_alert_threshold_seconds",
}

_PROFILE_FLOAT_KEYS: set[str] = {
    "zalo_api_timeout_seconds",
}


@dataclass
class GoogleSheetsConfig:
    """Configuration for Google Sheets sync."""

    enabled: bool = False
    spreadsheet_id: str = ""
    session_worksheet_name: str = "focusguardian_sessions"
    baseline_worksheet_name: str = "focusguardian_user_baselines"
    events_worksheet_name: str = "focusguardian_focus_events"
    users_worksheet_name: str = "focusguardian_users"
    profile_settings_worksheet_name: str = "focusguardian_profile_settings"
    service_account_path: str = "google_service_account.json"


class GoogleSheetsSessionSync:
    """Sync sessions, user baselines, and optional focus events to Google Sheets."""

    def __init__(self, config: Optional[GoogleSheetsConfig] = None):
        self.config = config or GoogleSheetsConfig()
        self._spreadsheet = None
        self._worksheets: Dict[str, Any] = {}

    def configure_from_app_config(self, app_config: Dict[str, Any]) -> None:
        """Load sync settings from app config dictionary."""
        session_ws = (
            str(
                app_config.get(
                    "google_sheets_sessions_worksheet",
                    app_config.get("google_sheets_worksheet", "focusguardian_sessions"),
                )
            ).strip()
            or "focusguardian_sessions"
        )
        baseline_ws = (
            str(app_config.get("google_sheets_baseline_worksheet", "focusguardian_user_baselines")).strip()
            or "focusguardian_user_baselines"
        )
        events_ws = (
            str(app_config.get("google_sheets_events_worksheet", "focusguardian_focus_events")).strip()
            or "focusguardian_focus_events"
        )
        users_ws = (
            str(app_config.get("google_sheets_users_worksheet", "focusguardian_users")).strip()
            or "focusguardian_users"
        )
        profile_settings_ws = (
            str(
                app_config.get(
                    "google_sheets_profile_settings_worksheet",
                    "focusguardian_profile_settings",
                )
            ).strip()
            or "focusguardian_profile_settings"
        )

        new_config = GoogleSheetsConfig(
            enabled=bool(app_config.get("enable_google_sheets_sync", False)),
            spreadsheet_id=str(app_config.get("google_sheets_id", "")).strip(),
            session_worksheet_name=session_ws,
            baseline_worksheet_name=baseline_ws,
            events_worksheet_name=events_ws,
            users_worksheet_name=users_ws,
            profile_settings_worksheet_name=profile_settings_ws,
            service_account_path=(
                str(app_config.get("google_service_account_path", "google_service_account.json")).strip()
                or "google_service_account.json"
            ),
        )

        if new_config != self.config:
            self._spreadsheet = None
            self._worksheets = {}

        self.config = new_config

    def append_session(self, session_record: Dict[str, Any]) -> bool:
        """Append one session row to Google Sheets if enabled."""
        worksheet = self._get_or_create_worksheet("sessions")
        if worksheet is None:
            return False

        try:
            worksheet.append_row(
                self._build_session_row(session_record),
                value_input_option="USER_ENTERED",
            )
            return True
        except Exception as exc:
            logger.warning("Failed to append session to Google Sheets: %s", exc)
            return False

    def upsert_user_baseline(self, baseline_record: Dict[str, Any]) -> bool:
        """Insert or update one profile baseline in the baseline worksheet."""
        worksheet = self._get_or_create_worksheet("baselines")
        if worksheet is None:
            return False

        profile_name = str(baseline_record.get("profile_name", "")).strip()
        if not profile_name:
            return False

        row = self._build_baseline_row(baseline_record)

        try:
            row_index = self._find_row_index_by_profile_name(worksheet, profile_name)
            if row_index is not None:
                worksheet.update(f"A{row_index}", [row], value_input_option="USER_ENTERED")
            else:
                worksheet.append_row(row, value_input_option="USER_ENTERED")
            return True
        except Exception as exc:
            logger.warning("Failed to upsert user baseline to Google Sheets: %s", exc)
            return False

    def load_user_baseline(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Load one profile baseline from Google Sheets."""
        worksheet = self._get_or_create_worksheet("baselines")
        if worksheet is None:
            return None

        key = (profile_name or "").strip().lower()
        if not key:
            return None

        try:
            records = worksheet.get_all_records()
        except Exception as exc:
            logger.warning("Failed to read baseline worksheet: %s", exc)
            return None

        for record in records:
            if str(record.get("profile_name", "")).strip().lower() == key:
                return self._parse_baseline_record(record)

        return None

    def load_all_baselines(self) -> Dict[str, Dict[str, Any]]:
        """Load all profile baselines indexed by profile_name."""
        worksheet = self._get_or_create_worksheet("baselines")
        if worksheet is None:
            return {}

        try:
            records = worksheet.get_all_records()
        except Exception as exc:
            logger.warning("Failed to read baseline worksheet: %s", exc)
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        for record in records:
            parsed = self._parse_baseline_record(record)
            key = str(parsed.get("profile_name", "")).strip()
            if key:
                result[key] = parsed
        return result

    def upsert_profile_settings(self, profile_name: str, settings: Dict[str, Any]) -> bool:
        """Insert or update profile-scoped settings in Google Sheets."""
        worksheet = self._get_or_create_worksheet("profile_settings")
        if worksheet is None:
            return False

        normalized_profile = str(profile_name or "").strip()
        if not normalized_profile:
            return False

        payload = self._extract_profile_settings_payload(settings)
        row = self._build_profile_settings_row(normalized_profile, payload)

        try:
            row_index = self._find_row_index_by_profile_name(worksheet, normalized_profile)
            if row_index is not None:
                worksheet.update(f"A{row_index}", [row], value_input_option="USER_ENTERED")
            else:
                worksheet.append_row(row, value_input_option="USER_ENTERED")
            return True
        except Exception as exc:
            logger.warning("Failed to upsert profile settings to Google Sheets: %s", exc)
            return False

    def load_profile_settings(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """
        Load profile-scoped settings from Google Sheets.

        Returns:
        - dict: settings payload (possibly empty) when worksheet is reachable
        - None: worksheet unavailable / read error
        """
        worksheet = self._get_or_create_worksheet("profile_settings")
        if worksheet is None:
            return None

        key = str(profile_name or "").strip().lower()
        if not key:
            return None

        try:
            records = worksheet.get_all_records()
        except Exception as exc:
            logger.warning("Failed to read profile settings worksheet: %s", exc)
            return None

        for record in records:
            if str(record.get("profile_name", "")).strip().lower() == key:
                parsed = self._parse_profile_settings_record(record)
                return dict(parsed.get("settings", {}) or {})

        return {}

    def append_focus_event_summary(self, summary_record: Dict[str, Any]) -> bool:
        """Append optional focus-events summary rows."""
        worksheet = self._get_or_create_worksheet("events")
        if worksheet is None:
            return False

        try:
            worksheet.append_row(
                self._build_event_row(summary_record),
                value_input_option="USER_ENTERED",
            )
            return True
        except Exception as exc:
            logger.warning("Failed to append focus-event summary: %s", exc)
            return False

    def _get_or_create_worksheet(self, kind: str):
        if not self.config.enabled:
            return None

        if not self.config.spreadsheet_id:
            logger.warning("Google Sheets sync enabled but spreadsheet_id is empty")
            return None

        if kind in self._worksheets:
            return self._worksheets[kind]

        spreadsheet = self._open_spreadsheet()
        if spreadsheet is None:
            return None

        title = self._worksheet_title(kind)
        header = self._header(kind)

        try:
            worksheet = spreadsheet.worksheet(title)
        except Exception:
            try:
                worksheet = spreadsheet.add_worksheet(
                    title=title,
                    rows="3000",
                    cols=str(len(header) + 4),
                )
            except Exception as exc:
                logger.warning("Failed to create worksheet '%s': %s", title, exc)
                return None

        self._ensure_header(worksheet, header)
        self._worksheets[kind] = worksheet
        return worksheet

    def _open_spreadsheet(self):
        if self._spreadsheet is not None:
            return self._spreadsheet

        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except Exception as exc:
            logger.warning("Google Sheets sync unavailable (missing dependency): %s", exc)
            return None

        credentials_path = self._resolve_credentials_path()
        if not credentials_path.exists():
            logger.warning("Google Sheets credentials file not found: %s", credentials_path)
            return None

        try:
            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = Credentials.from_service_account_file(str(credentials_path), scopes=scopes)
            client = gspread.authorize(creds)
            self._spreadsheet = client.open_by_key(self.config.spreadsheet_id)
            return self._spreadsheet
        except Exception as exc:
            logger.warning("Failed to initialize Google Sheets client: %s", exc)
            return None

    def _resolve_credentials_path(self) -> Path:
        raw_path = Path(self.config.service_account_path)
        if raw_path.is_absolute():
            return raw_path

        candidates: List[Path] = []
        seen: set[Path] = set()

        def add_candidate(path: Path) -> None:
            key = path
            try:
                key = path.resolve()
            except Exception:
                pass
            if key in seen:
                return
            seen.add(key)
            candidates.append(path)

        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            add_candidate(exe_dir / raw_path)
            add_candidate(exe_dir / "_internal" / raw_path)

            meipass = getattr(sys, "_MEIPASS", "")
            if meipass:
                add_candidate(Path(meipass) / raw_path)

        add_candidate(Path.cwd() / raw_path)
        add_candidate(Path(__file__).resolve().parents[2] / raw_path)

        for path in candidates:
            if path.exists():
                return path

        return candidates[0] if candidates else raw_path

    def _ensure_header(self, worksheet, header: List[str]) -> None:
        try:
            first_row = worksheet.row_values(1)
        except Exception:
            first_row = []

        if first_row != header:
            worksheet.update("A1", [header])

    def _header(self, kind: str) -> List[str]:
        if kind == "sessions":
            return self._session_header()
        if kind == "baselines":
            return self._baseline_header()
        if kind == "events":
            return self._event_header()
        if kind == "profile_settings":
            return self._profile_settings_header()
        return self._session_header()

    def _worksheet_name_from_kind(self, kind: str) -> str:
        if kind == "sessions":
            return self.config.session_worksheet_name
        if kind == "baselines":
            return self.config.baseline_worksheet_name
        if kind == "events":
            return self.config.events_worksheet_name
        if kind == "profile_settings":
            return self.config.profile_settings_worksheet_name
        return self.config.session_worksheet_name

    def _worksheet_title(self, kind: str) -> str:
        return self._worksheet_name_from_kind(kind)

    @staticmethod
    def _session_header() -> List[str]:
        return [
            "timestamp",
            "timestamp_iso",
            "profile_name",
            "session_seconds",
            "focus_seconds",
            "focus_seconds_raw",
            "focus_seconds_cleaned",
            "distraction_count",
            "break_count",
            "avg_score",
            "avg_score_raw",
            "avg_score_cleaned",
            "min_score",
            "max_score",
            "blink_rate_per_min",
            "avg_ear",
            "eye_closure_ratio",
            "perclos",
            "fatigue_onset_minutes",
            "score_drop_per_hour",
            "score_drop_per_hour_raw",
            "score_drop_per_hour_cleaned",
            "uncertain_seconds_raw",
            "uncertain_seconds_cleaned",
            "uncertain_measurement_noise_seconds",
            "uncertain_behavioral_seconds",
            "analytics_quality_score",
            "session_quality_weight",
            "face_presence_ratio",
            "minutes_since_last_break",
            "work_interval_minutes_used",
            "break_duration_minutes_used",
            "state_on_screen",
            "state_writing",
            "state_phone",
            "state_drowsy",
            "state_away",
            "state_uncertain",
        ]

    @staticmethod
    def _baseline_header() -> List[str]:
        return [
            "profile_name",
            "updated_at",
            "updated_at_iso",
            "session_count",
            "personalization_weight",
            "adaptation_stage",
            "blink_rate_baseline",
            "avg_ear_baseline",
            "eye_closure_ratio_baseline",
            "perclos_baseline",
            "average_focus_score_baseline",
            "average_distraction_density",
            "average_fatigue_onset_minutes",
            "focus_score_decay_per_hour",
            "recommended_work_minutes",
            "recommended_break_minutes",
            "last_quality_score",
        ]

    @staticmethod
    def _event_header() -> List[str]:
        return [
            "timestamp",
            "timestamp_iso",
            "profile_name",
            "session_id",
            "event_type",
            "event_count",
            "event_seconds",
            "avg_confidence",
            "metadata",
        ]

    @staticmethod
    def _profile_settings_header() -> List[str]:
        return [
            "profile_name",
            "updated_at",
            "updated_at_iso",
            "settings_json",
        ]

    @staticmethod
    def _build_session_row(session_record: Dict[str, Any]) -> List[Any]:
        timestamp = int(session_record.get("timestamp", 0) or 0)
        ts_iso = (
            datetime.fromtimestamp(timestamp).isoformat(sep=" ", timespec="seconds")
            if timestamp > 0
            else ""
        )

        states = session_record.get("state_seconds", {}) or {}
        return [
            timestamp,
            ts_iso,
            session_record.get("profile_name", ""),
            session_record.get("session_seconds", 0),
            session_record.get("focus_seconds", 0),
            session_record.get("focus_seconds_raw", session_record.get("focus_seconds", 0)),
            session_record.get("focus_seconds_cleaned", session_record.get("focus_seconds", 0)),
            session_record.get("distraction_count", 0),
            session_record.get("break_count", 0),
            session_record.get("avg_score", 0),
            session_record.get("avg_score_raw", session_record.get("avg_score", 0)),
            session_record.get("avg_score_cleaned", session_record.get("avg_score", 0)),
            session_record.get("min_score", 0),
            session_record.get("max_score", 0),
            session_record.get("blink_rate_per_min", 0),
            session_record.get("avg_ear", 0),
            session_record.get("eye_closure_ratio", 0),
            session_record.get("perclos", 0),
            session_record.get("fatigue_onset_minutes", ""),
            session_record.get("score_drop_per_hour", 0),
            session_record.get("score_drop_per_hour_raw", session_record.get("score_drop_per_hour", 0)),
            session_record.get("score_drop_per_hour_cleaned", session_record.get("score_drop_per_hour", 0)),
            session_record.get("uncertain_seconds_raw", ""),
            session_record.get("uncertain_seconds_cleaned", ""),
            session_record.get("uncertain_measurement_noise_seconds", ""),
            session_record.get("uncertain_behavioral_seconds", ""),
            session_record.get("analytics_quality_score", ""),
            session_record.get("session_quality_weight", ""),
            session_record.get("face_presence_ratio", ""),
            session_record.get("minutes_since_last_break", ""),
            session_record.get("work_interval_minutes_used", 0),
            session_record.get("break_duration_minutes_used", 0),
            float(states.get("ON_SCREEN_READING", 0.0) or 0.0),
            float(states.get("OFFSCREEN_WRITING", 0.0) or 0.0),
            float(states.get("PHONE_DISTRACTION", 0.0) or 0.0),
            float(states.get("DROWSY_FATIGUE", 0.0) or 0.0),
            float(states.get("AWAY", 0.0) or 0.0),
            float(states.get("UNCERTAIN", 0.0) or 0.0),
        ]

    @staticmethod
    def _build_baseline_row(baseline_record: Dict[str, Any]) -> List[Any]:
        updated_at = int(baseline_record.get("updated_at", 0) or 0)
        updated_at_iso = (
            datetime.fromtimestamp(updated_at).isoformat(sep=" ", timespec="seconds")
            if updated_at > 0
            else ""
        )
        session_count = int(baseline_record.get("session_count", 0) or 0)
        adaptation_stage = str(
            baseline_record.get("adaptation_stage")
            or ("cold_start" if session_count < 3 else ("hybrid" if session_count <= 7 else "personalized"))
        )

        return [
            baseline_record.get("profile_name", ""),
            updated_at,
            updated_at_iso,
            session_count,
            float(baseline_record.get("personalization_weight", 0.0) or 0.0),
            adaptation_stage,
            float(baseline_record.get("blink_rate_baseline", 0.0) or 0.0),
            float(baseline_record.get("avg_ear_baseline", 0.0) or 0.0),
            float(baseline_record.get("eye_closure_ratio_baseline", 0.0) or 0.0),
            float(baseline_record.get("perclos_baseline", 0.0) or 0.0),
            float(baseline_record.get("average_focus_score_baseline", 0.0) or 0.0),
            float(baseline_record.get("average_distraction_density", 0.0) or 0.0),
            float(baseline_record.get("average_fatigue_onset_minutes", 0.0) or 0.0),
            float(baseline_record.get("focus_score_decay_per_hour", 0.0) or 0.0),
            int(baseline_record.get("recommended_work_minutes", 25) or 25),
            int(baseline_record.get("recommended_break_minutes", 5) or 5),
            float(baseline_record.get("last_quality_score", 0.0) or 0.0),
        ]

    @staticmethod
    def _build_event_row(summary_record: Dict[str, Any]) -> List[Any]:
        timestamp = int(summary_record.get("timestamp", 0) or 0)
        ts_iso = (
            datetime.fromtimestamp(timestamp).isoformat(sep=" ", timespec="seconds")
            if timestamp > 0
            else ""
        )
        return [
            timestamp,
            ts_iso,
            summary_record.get("profile_name", ""),
            summary_record.get("session_id", ""),
            summary_record.get("event_type", ""),
            summary_record.get("event_count", 0),
            summary_record.get("event_seconds", 0),
            summary_record.get("avg_confidence", 0),
            summary_record.get("metadata", ""),
        ]

    @classmethod
    def _build_profile_settings_row(cls, profile_name: str, settings: Dict[str, Any]) -> List[Any]:
        updated_at = int(time.time())
        updated_at_iso = datetime.fromtimestamp(updated_at).isoformat(sep=" ", timespec="seconds")
        payload = cls._normalize_profile_settings_payload(settings)
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return [
            str(profile_name or "").strip(),
            updated_at,
            updated_at_iso,
            serialized,
        ]

    @staticmethod
    def _parse_baseline_record(record: Dict[str, Any]) -> Dict[str, Any]:
        def to_float(key: str, default: float = 0.0) -> float:
            try:
                return float(record.get(key, default) or default)
            except (TypeError, ValueError):
                return float(default)

        def to_int(key: str, default: int = 0) -> int:
            try:
                return int(float(record.get(key, default) or default))
            except (TypeError, ValueError):
                return int(default)

        session_count = to_int("session_count", 0)
        adaptation_stage = str(record.get("adaptation_stage", "") or "").strip()
        if not adaptation_stage:
            adaptation_stage = "cold_start" if session_count < 3 else ("hybrid" if session_count <= 7 else "personalized")

        return {
            "profile_name": str(record.get("profile_name", "")).strip(),
            "updated_at": to_int("updated_at", 0),
            "session_count": session_count,
            "personalization_weight": to_float("personalization_weight", 0.0),
            "adaptation_stage": adaptation_stage,
            "blink_rate_baseline": to_float("blink_rate_baseline", 12.0),
            "avg_ear_baseline": to_float("avg_ear_baseline", 0.25),
            "eye_closure_ratio_baseline": to_float("eye_closure_ratio_baseline", 0.12),
            "perclos_baseline": to_float("perclos_baseline", 0.08),
            "average_focus_score_baseline": to_float("average_focus_score_baseline", 75.0),
            "average_distraction_density": to_float("average_distraction_density", 3.0),
            "average_fatigue_onset_minutes": to_float("average_fatigue_onset_minutes", 35.0),
            "focus_score_decay_per_hour": to_float("focus_score_decay_per_hour", 0.0),
            "recommended_work_minutes": to_int("recommended_work_minutes", 25),
            "recommended_break_minutes": to_int("recommended_break_minutes", 5),
            "last_quality_score": to_float("last_quality_score", 0.0),
        }

    @classmethod
    def _parse_profile_settings_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        parsed_settings: Dict[str, Any] = {}

        raw_payload = record.get("settings_json")
        if isinstance(raw_payload, str) and raw_payload.strip():
            try:
                payload_obj = json.loads(raw_payload)
                if isinstance(payload_obj, dict):
                    parsed_settings.update(payload_obj)
            except Exception as exc:
                logger.warning("Failed to parse settings_json for profile '%s': %s", record.get("profile_name"), exc)

        # Fallback for legacy worksheet schemas that store plain columns.
        if not parsed_settings:
            for key in PROFILE_SCOPED_CONFIG_KEYS:
                if key in record:
                    parsed_settings[key] = record.get(key)

        return {
            "profile_name": str(record.get("profile_name", "")).strip(),
            "updated_at": cls._coerce_int(record.get("updated_at"), 0),
            "settings": cls._normalize_profile_settings_payload(parsed_settings),
        }

    @staticmethod
    def _extract_profile_settings_payload(settings: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        source = dict(settings or {})

        for key in PROFILE_SCOPED_CONFIG_KEYS:
            if key in source:
                payload[key] = source.get(key)

        return GoogleSheetsSessionSync._normalize_profile_settings_payload(payload)

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)

        text = str(value or "").strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
        return bool(default)

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _normalize_profile_settings_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}

        for key in PROFILE_SCOPED_CONFIG_KEYS:
            if key not in payload:
                continue

            value = payload.get(key)
            if key in _PROFILE_BOOL_KEYS:
                normalized[key] = cls._coerce_bool(value)
            elif key in _PROFILE_INT_KEYS:
                normalized[key] = cls._coerce_int(value)
            elif key in _PROFILE_FLOAT_KEYS:
                normalized[key] = cls._coerce_float(value)
            else:
                normalized[key] = "" if value is None else value

        return normalized

    @staticmethod
    def _find_row_index_by_profile_name(worksheet, profile_name: str) -> Optional[int]:
        key = (profile_name or "").strip().lower()
        if not key:
            return None

        try:
            col = worksheet.col_values(1)
        except Exception:
            return None

        for idx, value in enumerate(col[1:], start=2):
            if str(value).strip().lower() == key:
                return idx
        return None
