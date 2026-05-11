"""Shared cloud payload helpers for FocusGuardian.

This module is backend-neutral. Supabase sync uses these helpers to keep a
stable schema for sessions, baselines, focus events, and profile settings.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List

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
    "enable_task_context_monitoring",
    "task_context_sample_interval_seconds",
    "task_context_lookback_minutes",
    "task_context_max_samples",
    "task_context_task_keywords",
    "task_context_distracting_keywords",
    "task_context_neutral_keywords",
    "task_context_task_apps",
    "task_context_distracting_apps",
    "task_context_excluded_keywords",
    "task_context_excluded_apps",
    "task_context_checkin_enabled",
    "task_context_checkin_interval_minutes",
    "task_context_checkin_cooldown_minutes",
    "task_context_checkin_risk_threshold",
    "task_context_checkin_max_per_hour",
    "task_context_alert_enabled",
    "task_context_alert_cooldown_seconds",
    "task_context_alert_threshold",
    "session_goal_prompt_enabled",
    "session_exit_feedback_enabled",
    "deadline_mode_enabled",
    "deadline_focus_minutes",
    "recovery_validation_delay_seconds",
    "recovery_focus_delta_min",
    "session_report_show_on_stop",
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
    "zalo_distraction_confirm_seconds",
    "zalo_state_cooldown_seconds",
    "zalo_alert_on_distraction",
    "zalo_alert_on_drowsy",
    "zalo_alert_on_phone",
    "zalo_alert_on_away",
    "zalo_alert_on_break_reminder",
)

PROFILE_SCOPED_DEFAULT_SETTINGS: Dict[str, Any] = {
    "theme_mode": "dark",
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
    "enable_task_context_monitoring": True,
    "task_context_sample_interval_seconds": 8.0,
    "task_context_lookback_minutes": 5.0,
    "task_context_max_samples": 2400,
    "task_context_task_keywords": (
        "code,coding,visual studio code,visual studio,pycharm,cursor,terminal,powershell,"
        "command prompt,notion,obsidian,word,excel,powerpoint,google docs,google sheets,"
        "google slides,google drive,drive,docs,sheets,slides,research,study,learning,"
        "classroom,coursera,udemy,edx,khan academy,jira,trello,linear,asana,figma,canva,"
        "slack,teams,zoom,meet,google meet,outlook,mail,github,gitlab,stackoverflow,stack overflow"
    ),
    "task_context_distracting_keywords": (
        "youtube,youtube shorts,shorts,facebook,facebook watch,fb watch,instagram,reels,"
        "threads,tiktok,netflix,prime video,disney+,disney plus,hbo,max.com,game,steam,"
        "discord,reddit,x.com,twitter,snapchat,pinterest,news,shopping,shopee,lazada,tiki,"
        "sendo,twitch,kick.com,roblox,valorant,league of legends,liên minh,lien minh,lol,"
        "epic games,garena,minecraft,fortnite,free fire,pubg,genshin,honkai,zenless,dota,"
        "dota 2,counter-strike,counter strike,cs2,overwatch,battle.net"
    ),
    "task_context_neutral_keywords": "explorer,settings,file",
    "task_context_task_apps": (
        "code.exe,cursor.exe,pycharm64.exe,devenv.exe,windowsterminal.exe,powershell.exe,"
        "cmd.exe,notion.exe,obsidian.exe,figma.exe,slack.exe,teams.exe,ms-teams.exe,zoom.exe,"
        "winword.exe,excel.exe,powerpnt.exe,outlook.exe"
    ),
    "task_context_distracting_apps": (
        "steam.exe,steamwebhelper.exe,epicgameslauncher.exe,epicgameslauncher,epicwebhelper.exe,"
        "riotclientservices.exe,riotclientux.exe,leagueclient.exe,leagueclientux.exe,"
        "league of legends.exe,valorant.exe,valorant-win64-shipping.exe,robloxplayerbeta.exe,"
        "minecraftlauncher.exe,fortniteclient-win64-shipping.exe,garena.exe,freefire.exe,"
        "pubg.exe,tslgame.exe,dota2.exe,cs2.exe,overwatch.exe,battle.net.exe,battlenet.exe,"
        "eaapp.exe,ubisoftconnect.exe,genshinimpact.exe,hoyoplay.exe,discord.exe,tiktok.exe"
    ),
    "task_context_excluded_keywords": "focusguardian,notification",
    "task_context_excluded_apps": "",
    "task_context_checkin_enabled": True,
    "task_context_checkin_interval_minutes": 12,
    "task_context_checkin_cooldown_minutes": 8,
    "task_context_checkin_risk_threshold": 0.72,
    "task_context_checkin_max_per_hour": 3,
    "task_context_alert_enabled": True,
    "task_context_alert_cooldown_seconds": 120,
    "task_context_alert_threshold": 0.68,
    "session_goal_prompt_enabled": True,
    "session_exit_feedback_enabled": True,
    "deadline_mode_enabled": False,
    "deadline_focus_minutes": 45,
    "recovery_validation_delay_seconds": 90,
    "recovery_focus_delta_min": 6.0,
    "session_report_show_on_stop": True,
    "enable_pomodoro": False,
    "pomodoro_work": 25,
    "pomodoro_short_break": 5,
    "pomodoro_long_break": 15,
    "enable_zalo_alerts": False,
    "zalo_chat_id": "",
    "zalo_webhook_secret": "",
    "zalo_api_timeout_seconds": 8.0,
    "zalo_alert_cooldown_minutes": 2,
    "zalo_alert_threshold_seconds": 5,
    "zalo_distraction_confirm_seconds": 5,
    "zalo_state_cooldown_seconds": 120,
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
    "enable_task_context_monitoring",
    "task_context_checkin_enabled",
    "task_context_alert_enabled",
    "session_goal_prompt_enabled",
    "session_exit_feedback_enabled",
    "deadline_mode_enabled",
    "session_report_show_on_stop",
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
    "task_context_max_samples",
    "task_context_checkin_interval_minutes",
    "task_context_checkin_cooldown_minutes",
    "task_context_checkin_max_per_hour",
    "task_context_alert_cooldown_seconds",
    "deadline_focus_minutes",
    "recovery_validation_delay_seconds",
    "pomodoro_work",
    "pomodoro_short_break",
    "pomodoro_long_break",
    "zalo_alert_cooldown_minutes",
    "zalo_alert_threshold_seconds",
    "zalo_distraction_confirm_seconds",
    "zalo_state_cooldown_seconds",
}

_PROFILE_FLOAT_KEYS: set[str] = {
    "task_context_sample_interval_seconds",
    "task_context_lookback_minutes",
    "task_context_checkin_risk_threshold",
    "task_context_alert_threshold",
    "recovery_focus_delta_min",
    "zalo_api_timeout_seconds",
}


def session_header() -> List[str]:
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
        "session_exit_reason",
        "session_exit_reason_label",
        "session_exit_focus_rating",
        "session_exit_focus_rating_label",
        "session_exit_note",
    ]


def baseline_header() -> List[str]:
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


def event_header() -> List[str]:
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


def profile_settings_header() -> List[str]:
    return [
        "profile_name",
        "updated_at",
        "updated_at_iso",
        "settings_json",
    ]


def build_session_row(session_record: Dict[str, Any]) -> List[Any]:
    timestamp = int(session_record.get("timestamp", 0) or 0)
    ts_iso = (
        datetime.fromtimestamp(timestamp).isoformat(sep=" ", timespec="seconds")
        if timestamp > 0
        else ""
    )
    states = session_record.get("state_seconds", {}) or {}
    session_exit = session_record.get("session_exit", {}) or {}
    if not isinstance(session_exit, dict):
        session_exit = {}

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
        session_exit.get("reason", ""),
        session_exit.get("reason_label", ""),
        session_exit.get("focus_rating", ""),
        session_exit.get("focus_rating_label", ""),
        session_exit.get("note", ""),
    ]


def build_baseline_row(baseline_record: Dict[str, Any]) -> List[Any]:
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


def build_event_row(summary_record: Dict[str, Any]) -> List[Any]:
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


def build_profile_settings_row(profile_name: str, settings: Dict[str, Any]) -> List[Any]:
    updated_at = int(time.time())
    updated_at_iso = datetime.fromtimestamp(updated_at).isoformat(sep=" ", timespec="seconds")
    payload = normalize_profile_settings_payload(settings)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return [
        str(profile_name or "").strip(),
        updated_at,
        updated_at_iso,
        serialized,
    ]


def parse_baseline_record(record: Dict[str, Any]) -> Dict[str, Any]:
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


def parse_profile_settings_record(record: Dict[str, Any]) -> Dict[str, Any]:
    parsed_settings: Dict[str, Any] = {}
    raw_payload = record.get("settings_json")
    if isinstance(raw_payload, str) and raw_payload.strip():
        try:
            payload_obj = json.loads(raw_payload)
            if isinstance(payload_obj, dict):
                parsed_settings.update(payload_obj)
        except Exception as exc:
            logger.warning("Failed to parse settings_json for profile '%s': %s", record.get("profile_name"), exc)

    if not parsed_settings:
        for key in PROFILE_SCOPED_CONFIG_KEYS:
            if key in record:
                parsed_settings[key] = record.get(key)

    return {
        "profile_name": str(record.get("profile_name", "")).strip(),
        "updated_at": coerce_int(record.get("updated_at"), 0),
        "settings": normalize_profile_settings_payload(parsed_settings),
    }


def extract_profile_settings_payload(settings: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    source = dict(settings or {})
    for key in PROFILE_SCOPED_CONFIG_KEYS:
        if key in source:
            payload[key] = source.get(key)
    return normalize_profile_settings_payload(payload)


def coerce_bool(value: Any, default: bool = False) -> bool:
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


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_profile_settings_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key in PROFILE_SCOPED_CONFIG_KEYS:
        if key not in payload:
            continue
        value = payload.get(key)
        if key in _PROFILE_BOOL_KEYS:
            normalized[key] = coerce_bool(value)
        elif key in _PROFILE_INT_KEYS:
            normalized[key] = coerce_int(value)
        elif key in _PROFILE_FLOAT_KEYS:
            normalized[key] = coerce_float(value)
        else:
            normalized[key] = "" if value is None else value
    return normalized
