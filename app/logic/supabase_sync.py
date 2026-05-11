"""Supabase-backed cloud sync for FocusGuardian analytics."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .cloud_payloads import (
    baseline_header,
    build_baseline_row,
    build_event_row,
    build_profile_settings_row,
    build_session_row,
    event_header,
    extract_profile_settings_payload,
    normalize_profile_settings_payload,
    parse_baseline_record,
    parse_profile_settings_record,
    profile_settings_header,
    session_header,
)

logger = logging.getLogger(__name__)


SESSION_NUMERIC_COLUMNS: set[str] = {
    "timestamp",
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
    "session_exit_focus_rating",
}

BASELINE_NUMERIC_COLUMNS: set[str] = {
    "updated_at",
    "session_count",
    "personalization_weight",
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
}

EVENT_NUMERIC_COLUMNS: set[str] = {
    "timestamp",
    "event_count",
    "event_seconds",
    "avg_confidence",
}

PROFILE_SETTINGS_NUMERIC_COLUMNS: set[str] = {"updated_at"}


@dataclass(frozen=True)
class SupabaseConfig:
    """Configuration for Supabase sync."""

    enabled: bool = False
    url: str = ""
    api_key: str = ""
    sessions_table_name: str = "focusguardian_sessions"
    baseline_table_name: str = "focusguardian_user_baselines"
    events_table_name: str = "focusguardian_focus_events"
    users_table_name: str = "focusguardian_users"
    profile_settings_table_name: str = "focusguardian_profile_settings"
    timeout_seconds: float = 10.0


def supabase_key_from_app_config(app_config: Dict[str, Any]) -> str:
    """Return the first configured Supabase API key."""
    env_key = (
        os.environ.get("SUPABASE_API_KEY")
        or os.environ.get("SUPABASE_PUBLISHABLE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or ""
    )
    if env_key.strip():
        return env_key.strip()

    for key in (
        "supabase_api_key",
        "supabase_key",
        "supabase_publishable_key",
        "supabase_anon_key",
        "supabase_service_role_key",
    ):
        value = str(app_config.get(key, "") or "").strip()
        if value:
            return value
    return ""


def supabase_url_from_app_config(app_config: Dict[str, Any]) -> str:
    env_url = os.environ.get("SUPABASE_URL") or ""
    return (env_url or str(app_config.get("supabase_url", "") or "")).strip().rstrip("/")


def supabase_missing_config_message(config: SupabaseConfig) -> str:
    """Return a user-facing message for incomplete Supabase config."""
    if not config.enabled:
        return "Supabase sync dang tat"
    if not config.url:
        return "Thieu Supabase URL"
    if not config.api_key:
        return "Thieu Supabase API key. Dien supabase_publishable_key hoac supabase_anon_key trong config.json"
    return ""


def supabase_config_from_app_config(app_config: Dict[str, Any]) -> SupabaseConfig:
    try:
        timeout_seconds = float(app_config.get("supabase_timeout_seconds", 10.0) or 10.0)
    except (TypeError, ValueError):
        timeout_seconds = 10.0

    return SupabaseConfig(
        enabled=bool(app_config.get("enable_supabase_sync", False)),
        url=supabase_url_from_app_config(app_config),
        api_key=supabase_key_from_app_config(app_config),
        sessions_table_name=(
            str(app_config.get("supabase_sessions_table", "focusguardian_sessions") or "").strip()
            or "focusguardian_sessions"
        ),
        baseline_table_name=(
            str(app_config.get("supabase_baseline_table", "focusguardian_user_baselines") or "").strip()
            or "focusguardian_user_baselines"
        ),
        events_table_name=(
            str(app_config.get("supabase_events_table", "focusguardian_focus_events") or "").strip()
            or "focusguardian_focus_events"
        ),
        users_table_name=(
            str(app_config.get("supabase_users_table", "focusguardian_users") or "").strip()
            or "focusguardian_users"
        ),
        profile_settings_table_name=(
            str(app_config.get("supabase_profile_settings_table", "focusguardian_profile_settings") or "").strip()
            or "focusguardian_profile_settings"
        ),
        timeout_seconds=max(2.0, min(30.0, timeout_seconds)),
    )


def _is_legacy_jwt_key(api_key: str) -> bool:
    return str(api_key or "").strip().startswith("eyJ")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _clean_column_value(column: str, value: Any, numeric_columns: set[str]) -> Any:
    if value == "":
        return None if column in numeric_columns else ""
    cleaned = _jsonable(value)
    if column in numeric_columns:
        if cleaned in ("", None):
            return None
        try:
            number = float(cleaned)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        if column in {
            "timestamp",
            "updated_at",
            "session_count",
            "distraction_count",
            "break_count",
            "work_interval_minutes_used",
            "break_duration_minutes_used",
            "recommended_work_minutes",
            "recommended_break_minutes",
            "event_count",
        }:
            return int(number)
        return number
    return cleaned


def _record_from_row(headers: List[str], row: List[Any], numeric_columns: set[str]) -> Dict[str, Any]:
    return {
        column: _clean_column_value(column, value, numeric_columns)
        for column, value in zip(headers, row)
    }


class SupabaseRestClient:
    """Small REST client for Supabase PostgREST endpoints."""

    def __init__(self, config: SupabaseConfig):
        self.config = config
        self._session = None

    @property
    def is_configured(self) -> bool:
        return bool(self.config.enabled and self.config.url and self.config.api_key)

    def _headers(self, *, prefer: str = "") -> Dict[str, str]:
        headers = {
            "apikey": self.config.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if _is_legacy_jwt_key(self.config.api_key):
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def _request(
        self,
        method: str,
        table_name: str,
        *,
        params: Optional[Dict[str, str]] = None,
        json_payload: Any = None,
        prefer: str = "",
    ) -> Optional[Any]:
        if not self.is_configured:
            return None

        try:
            import requests
        except Exception as exc:
            logger.warning("Supabase sync unavailable (missing requests): %s", exc)
            return None

        if self._session is None:
            self._session = requests.Session()

        url = f"{self.config.url}/rest/v1/{quote(str(table_name), safe='')}"
        try:
            response = self._session.request(
                method.upper(),
                url,
                params=params or {},
                headers=self._headers(prefer=prefer),
                json=_jsonable(json_payload) if json_payload is not None else None,
                timeout=self.config.timeout_seconds,
            )
        except Exception as exc:
            logger.warning("Supabase request failed for table '%s': %s", table_name, exc)
            return None

        if response.status_code not in {200, 201, 204}:
            body = (response.text or "").strip()
            logger.warning(
                "Supabase request failed for table '%s' (%s): %s",
                table_name,
                response.status_code,
                body[:400],
            )
            return None

        if response.status_code == 204 or not (response.text or "").strip():
            return True

        try:
            return response.json()
        except ValueError:
            return True

    def health_check(self, table_name: str) -> bool:
        result = self._request(
            "GET",
            table_name,
            params={"select": "user_id", "limit": "1"},
        )
        return result is not None

    def select(
        self,
        table_name: str,
        *,
        select: str = "*",
        filters: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        params: Dict[str, str] = {"select": select}
        if filters:
            params.update(filters)
        if limit is not None:
            params["limit"] = str(int(limit))

        result = self._request("GET", table_name, params=params)
        if isinstance(result, list):
            return [row for row in result if isinstance(row, dict)]
        return None

    def insert(self, table_name: str, payload: Dict[str, Any]) -> bool:
        return self._request(
            "POST",
            table_name,
            json_payload=payload,
            prefer="return=minimal",
        ) is not None

    def upsert(self, table_name: str, payload: Dict[str, Any], *, on_conflict: str) -> bool:
        return self._request(
            "POST",
            table_name,
            params={"on_conflict": on_conflict},
            json_payload=payload,
            prefer="resolution=merge-duplicates,return=minimal",
        ) is not None

    def update(self, table_name: str, payload: Dict[str, Any], *, filters: Dict[str, str]) -> bool:
        return self._request(
            "PATCH",
            table_name,
            params=dict(filters),
            json_payload=payload,
            prefer="return=minimal",
        ) is not None


class SupabaseSessionSync:
    """Sync sessions, baselines, events and profile settings to Supabase."""

    def __init__(self, config: Optional[SupabaseConfig] = None):
        self.config = config or SupabaseConfig()
        self._client: Optional[SupabaseRestClient] = None

    def configure_from_app_config(self, app_config: Dict[str, Any]) -> None:
        new_config = supabase_config_from_app_config(app_config)
        if new_config != self.config:
            self._client = None
        self.config = new_config

    def _get_client(self) -> Optional[SupabaseRestClient]:
        missing = supabase_missing_config_message(self.config)
        if missing:
            logger.debug(missing)
            return None
        if self._client is None:
            self._client = SupabaseRestClient(self.config)
        return self._client

    def append_session(self, session_record: Dict[str, Any]) -> bool:
        client = self._get_client()
        if client is None:
            return False

        payload = _record_from_row(
            session_header(),
            build_session_row(session_record),
            SESSION_NUMERIC_COLUMNS,
        )
        payload["raw_payload"] = _jsonable(session_record)
        ok = client.insert(self.config.sessions_table_name, payload)
        if not ok:
            logger.debug("Skipped syncing session to Supabase")
        return ok

    def upsert_user_baseline(self, baseline_record: Dict[str, Any]) -> bool:
        client = self._get_client()
        if client is None:
            return False

        profile_name = str(baseline_record.get("profile_name", "") or "").strip()
        if not profile_name:
            return False

        payload = _record_from_row(
            baseline_header(),
            build_baseline_row(baseline_record),
            BASELINE_NUMERIC_COLUMNS,
        )
        payload["raw_payload"] = _jsonable(baseline_record)
        return client.upsert(
            self.config.baseline_table_name,
            payload,
            on_conflict="profile_name",
        )

    def load_user_baseline(self, profile_name: str) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        if client is None:
            return None

        key = str(profile_name or "").strip()
        if not key:
            return None

        rows = client.select(
            self.config.baseline_table_name,
            filters={"profile_name": f"eq.{key}"},
            limit=1,
        )
        if rows is None:
            return None
        if not rows:
            return None
        return parse_baseline_record(rows[0])

    def load_all_baselines(self) -> Dict[str, Dict[str, Any]]:
        client = self._get_client()
        if client is None:
            return {}

        rows = client.select(self.config.baseline_table_name) or []
        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            parsed = parse_baseline_record(row)
            key = str(parsed.get("profile_name", "") or "").strip()
            if key:
                result[key] = parsed
        return result

    def upsert_profile_settings(self, profile_name: str, settings: Dict[str, Any]) -> bool:
        client = self._get_client()
        if client is None:
            return False

        normalized_profile = str(profile_name or "").strip()
        if not normalized_profile:
            return False

        payload_settings = extract_profile_settings_payload(settings)
        payload = _record_from_row(
            profile_settings_header(),
            build_profile_settings_row(normalized_profile, payload_settings),
            PROFILE_SETTINGS_NUMERIC_COLUMNS,
        )
        payload["settings"] = _jsonable(payload_settings)
        return client.upsert(
            self.config.profile_settings_table_name,
            payload,
            on_conflict="profile_name",
        )

    def load_profile_settings(self, profile_name: str) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        if client is None:
            return None

        key = str(profile_name or "").strip()
        if not key:
            return None

        rows = client.select(
            self.config.profile_settings_table_name,
            filters={"profile_name": f"eq.{key}"},
            limit=1,
        )
        if rows is None:
            return None
        if not rows:
            return {}

        row = rows[0]
        raw_settings = row.get("settings")
        if isinstance(raw_settings, dict):
            return normalize_profile_settings_payload(raw_settings)

        parsed = parse_profile_settings_record(row)
        return dict(parsed.get("settings", {}) or {})

    def append_focus_event_summary(self, summary_record: Dict[str, Any]) -> bool:
        client = self._get_client()
        if client is None:
            return False

        payload = _record_from_row(
            event_header(),
            build_event_row(summary_record),
            EVENT_NUMERIC_COLUMNS,
        )
        if payload.get("metadata") == "":
            payload["metadata"] = None
        payload["raw_payload"] = _jsonable(summary_record)
        return client.insert(self.config.events_table_name, payload)
