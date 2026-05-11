#!/usr/bin/env python3
"""
FocusGuardian - Focus Tracking Desktop Application

Main entry point for the application.
"""

import sys
import logging
import json
import os
from pathlib import Path
from typing import Optional


def _relaunch_with_project_venv_if_needed() -> None:
    """Prefer the project virtualenv when launching from source."""
    if getattr(sys, "frozen", False):
        return
    if os.environ.get("FOCUSGUARDIAN_NO_VENV_REEXEC") == "1":
        return

    app_root = Path(__file__).resolve().parent
    venv_python = app_root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not venv_python.exists():
        return

    try:
        current_python = Path(sys.executable).resolve()
        target_python = venv_python.resolve()
    except Exception:
        current_python = Path(sys.executable)
        target_python = venv_python

    if current_python == target_python:
        return

    if os.environ.get("FOCUSGUARDIAN_VENV_REEXEC") == "1":
        print(
            "FocusGuardian: dang chay sai Python runtime. Hay dung lenh:\n"
            r"  .\.venv\Scripts\python.exe main.py",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    os.environ["FOCUSGUARDIAN_VENV_REEXEC"] = "1"
    print(f"FocusGuardian: relaunching with project Python: {target_python}", flush=True)
    try:
        os.execv(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    except OSError as exc:
        print(
            "FocusGuardian: khong the tu chuyen sang .venv.\n"
            f"Ly do: {exc}\n"
            "Hay chay bang lenh:\n"
            r"  .\.venv\Scripts\python.exe main.py",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc


_relaunch_with_project_venv_if_needed()

from PyQt6.QtWidgets import QApplication, QDialog
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from app.logic.cloud_payloads import PROFILE_SCOPED_CONFIG_KEYS
from app.logic.zalo_bot import FIXED_ZALO_BOT_TOKEN


def _runtime_root_dir() -> Path:
    """Return stable runtime root for user-writable files next to the launcher."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _bundled_root_dir() -> Path:
    """Return PyInstaller's bundled data root, or the source root in dev."""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            return Path(meipass)
        return _runtime_root_dir() / "_internal"
    return Path(__file__).resolve().parent


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            key = path.resolve()
        except Exception:
            key = path
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_runtime_root_dir() / "focusguardian.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# Configuration file path
CONFIG_FILE = _runtime_root_dir() / "config.json"
BUNDLED_CONFIG_FILE = _bundled_root_dir() / "config.json"
DEFAULT_CONFIG = {
    "camera_id": 0,
    "resolution": "640x480",
    "fps": 15,
    "vision_target_fps": 8,
    "camera_preview_fps": 12,
    "show_overlay": True,
    "show_face_mesh": False,
    "head_down_threshold": -22,
    "look_away_threshold": 30,
    "eye_look_down_threshold": 0.35,
    "eye_look_up_threshold": 0.30,
    "enable_phone_detection": True,
    "phone_detection_mode": "heuristic",
    "phone_confidence_threshold": 0.55,
    "phone_detection_interval_frames": 4,
    "phone_confirmation_window_seconds": 2.5,
    "phone_confirmation_min_hits": 3,
    "phone_confidence_min": 0.58,
    "phone_eye_down_min_duration": 45,
    "blink_rate_low_screen_max": 10.0,
    "blink_rate_high_fatigue_min": 22.0,
    "write_score_threshold": 0.4,
    "eye_closure_threshold": 0.3,
    "ear_threshold": 0.18,
    "perclos_threshold": 0.15,
    "focused_state_hold_seconds": 2.2,
    "uncertain_short_soft_seconds": 2.0,
    "uncertain_behavior_window_seconds": 3.5,
    "score_noise_softening_seconds": 2.6,
    "score_confidence_floor_focused": 0.58,
    "score_confidence_floor_uncertain": 0.33,
    "vision_confidence_uncertain_threshold": 0.42,
    "vision_confidence_hard_floor": 0.28,
    "score_recover_rate": 5.5,
    "score_drop_rate": 4.0,
    "score_recover_rate_focused_stable": 4.4,
    "score_recover_rate_focused_unstable": 1.8,
    "score_drop_rate_distraction_strong": 8.2,
    "score_drop_rate_distraction_soft": 3.6,
    "score_drop_rate_drowsy_strong": 6.8,
    "score_uncertain_soft_penalty": 0.24,
    "time_on_task_drift_start_minutes": 22.0,
    "time_on_task_drift_per_minute": 0.08,
    "break_recovery_boost_window_seconds": 45.0,
    "display_uncertain_hold_seconds": 2.0,
    "enable_break_reminders": True,
    "break_interval_minutes": 25,
    "break_duration_minutes": 5,
    "profile_name": "default",
    "vision_calibration": {},
    "enable_personalization": True,
    "auto_apply_personalization": True,
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
    "enable_supabase_sync": True,
    "supabase_url": "https://jprbgccjztseypgzcmea.supabase.co",
    "supabase_api_key": "",
    "supabase_publishable_key": "",
    "supabase_anon_key": "",
    "supabase_service_role_key": "",
    "supabase_sessions_table": "focusguardian_sessions",
    "supabase_baseline_table": "focusguardian_user_baselines",
    "supabase_events_table": "focusguardian_focus_events",
    "supabase_users_table": "focusguardian_users",
    "supabase_profile_settings_table": "focusguardian_profile_settings",
    "supabase_timeout_seconds": 10.0,
    "allow_offline_login": False,
    "enable_zalo_alerts": False,
    "zalo_bot_token": FIXED_ZALO_BOT_TOKEN,
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
    "enable_notifications": True,
    "notify_distraction": True,
    "notify_break": True,
    "notify_drowsy": True,
    "enable_sounds": True,
    "volume": 70,
    "sound_file": "",
    "enable_focus_audio": False,
    "focus_audio_track": "rain_light",
    "focus_audio_volume": 30,
    "theme_mode": "dark",
    "auth_last_username": "",
    "auth_last_profile_name": "",
    "auth_last_login_at_iso": "",
    "auth_session_user_id": "",
    "auth_session_username": "",
    "auth_session_profile_name": "",
    "auth_session_login_at": 0,
    "auth_session_login_at_iso": "",
    "auth_persist_session": True,
    "minimize_to_tray": False,
    "start_minimized": False,
}


def _config_source_files() -> list[Path]:
    """Load packaged defaults first, then user-writable config overrides."""
    return [
        path
        for path in _dedupe_paths([BUNDLED_CONFIG_FILE, CONFIG_FILE])
        if path.exists()
    ]


def load_config() -> dict:
    """Load configuration from file."""
    config = DEFAULT_CONFIG.copy()

    for config_file in _config_source_files():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                config.update(saved)
            logger.info("Configuration loaded from %s", config_file)
        except Exception as e:
            logger.warning("Failed to load config from %s: %s", config_file, e)

    # Keep threshold in a sane range while preserving user preference.
    try:
        head_down = float(config.get("head_down_threshold", -22))
    except (TypeError, ValueError):
        head_down = -22.0
    config["head_down_threshold"] = max(-45.0, min(0.0, head_down))

    theme_mode = str(config.get("theme_mode", "dark")).strip().lower()
    if theme_mode not in {"dark", "light"}:
        config["theme_mode"] = "dark"
    else:
        config["theme_mode"] = theme_mode

    # Zalo bot token is fixed by product decision.
    config["zalo_bot_token"] = FIXED_ZALO_BOT_TOKEN

    config["enable_focus_audio"] = bool(config.get("enable_focus_audio", False))
    config["focus_audio_track"] = str(config.get("focus_audio_track", "rain_light") or "rain_light").strip().lower()
    try:
        focus_audio_volume = int(float(config.get("focus_audio_volume", 30)))
    except (TypeError, ValueError):
        focus_audio_volume = 30
    config["focus_audio_volume"] = max(0, min(100, focus_audio_volume))

    try:
        config["auth_session_login_at"] = int(float(config.get("auth_session_login_at", 0) or 0))
    except (TypeError, ValueError):
        config["auth_session_login_at"] = 0
    config["auth_session_user_id"] = str(config.get("auth_session_user_id", "") or "").strip()
    config["auth_session_username"] = str(config.get("auth_session_username", "") or "").strip()
    config["auth_session_profile_name"] = str(config.get("auth_session_profile_name", "") or "").strip()
    config["auth_session_login_at_iso"] = str(config.get("auth_session_login_at_iso", "") or "").strip()
    config["auth_persist_session"] = bool(config.get("auth_persist_session", True))

    def _normalize_keyword_csv(key: str, default_text: str) -> None:
        raw = config.get(key, default_text)
        tokens: list[str] = []

        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                token = str(item or "").strip().lower()
                if token:
                    tokens.append(token)
        else:
            text = str(raw or "").replace("\n", ",").replace(";", ",")
            for item in text.split(","):
                token = item.strip().lower()
                if token:
                    tokens.append(token)

        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)

        config[key] = ", ".join(deduped)

    config["enable_task_context_monitoring"] = bool(config.get("enable_task_context_monitoring", True))
    config["task_context_checkin_enabled"] = bool(config.get("task_context_checkin_enabled", True))
    config["task_context_alert_enabled"] = bool(config.get("task_context_alert_enabled", True))
    config["session_goal_prompt_enabled"] = bool(config.get("session_goal_prompt_enabled", True))
    config["session_exit_feedback_enabled"] = bool(config.get("session_exit_feedback_enabled", True))
    config["deadline_mode_enabled"] = bool(config.get("deadline_mode_enabled", False))
    config["session_report_show_on_stop"] = bool(config.get("session_report_show_on_stop", True))

    try:
        config["task_context_sample_interval_seconds"] = max(
            2.0,
            min(30.0, float(config.get("task_context_sample_interval_seconds", 5.0) or 5.0)),
        )
    except (TypeError, ValueError):
        config["task_context_sample_interval_seconds"] = 5.0

    try:
        config["task_context_lookback_minutes"] = max(
            1.0,
            min(60.0, float(config.get("task_context_lookback_minutes", 5.0) or 5.0)),
        )
    except (TypeError, ValueError):
        config["task_context_lookback_minutes"] = 5.0

    try:
        config["task_context_max_samples"] = max(
            120,
            min(12000, int(float(config.get("task_context_max_samples", 2400) or 2400))),
        )
    except (TypeError, ValueError):
        config["task_context_max_samples"] = 2400

    try:
        config["task_context_checkin_interval_minutes"] = max(
            2,
            min(60, int(float(config.get("task_context_checkin_interval_minutes", 12) or 12))),
        )
    except (TypeError, ValueError):
        config["task_context_checkin_interval_minutes"] = 12

    try:
        config["task_context_checkin_cooldown_minutes"] = max(
            1,
            min(60, int(float(config.get("task_context_checkin_cooldown_minutes", 8) or 8))),
        )
    except (TypeError, ValueError):
        config["task_context_checkin_cooldown_minutes"] = 8

    try:
        checkin_risk = float(config.get("task_context_checkin_risk_threshold", 0.72) or 0.72)
    except (TypeError, ValueError):
        checkin_risk = 0.72
    config["task_context_checkin_risk_threshold"] = max(0.2, min(0.98, checkin_risk))

    try:
        config["task_context_checkin_max_per_hour"] = max(
            1,
            min(12, int(float(config.get("task_context_checkin_max_per_hour", 3) or 3))),
        )
    except (TypeError, ValueError):
        config["task_context_checkin_max_per_hour"] = 3

    try:
        config["task_context_alert_cooldown_seconds"] = max(
            30,
            min(900, int(float(config.get("task_context_alert_cooldown_seconds", 120) or 120))),
        )
    except (TypeError, ValueError):
        config["task_context_alert_cooldown_seconds"] = 120

    try:
        context_alert_threshold = float(config.get("task_context_alert_threshold", 0.68) or 0.68)
    except (TypeError, ValueError):
        context_alert_threshold = 0.68
    config["task_context_alert_threshold"] = max(0.2, min(0.98, context_alert_threshold))

    try:
        config["deadline_focus_minutes"] = max(
            10,
            min(180, int(float(config.get("deadline_focus_minutes", 45) or 45))),
        )
    except (TypeError, ValueError):
        config["deadline_focus_minutes"] = 45

    try:
        config["recovery_validation_delay_seconds"] = max(
            30,
            min(600, int(float(config.get("recovery_validation_delay_seconds", 90) or 90))),
        )
    except (TypeError, ValueError):
        config["recovery_validation_delay_seconds"] = 90

    try:
        config["recovery_focus_delta_min"] = max(
            0.0,
            min(30.0, float(config.get("recovery_focus_delta_min", 6.0) or 6.0)),
        )
    except (TypeError, ValueError):
        config["recovery_focus_delta_min"] = 6.0

    _normalize_keyword_csv(
        "task_context_task_keywords",
        "code,coding,visual studio code,visual studio,pycharm,cursor,terminal,powershell,command prompt,notion,obsidian,word,excel,powerpoint,google docs,google sheets,google slides,google drive,drive,docs,sheets,slides,research,study,learning,classroom,coursera,udemy,edx,khan academy,jira,trello,linear,asana,figma,canva,slack,teams,zoom,meet,google meet,outlook,mail,github,gitlab,stackoverflow,stack overflow",
    )
    _normalize_keyword_csv(
        "task_context_distracting_keywords",
        "youtube,youtube shorts,shorts,facebook,facebook watch,fb watch,instagram,reels,threads,tiktok,netflix,prime video,disney+,disney plus,hbo,max.com,game,steam,discord,reddit,x.com,twitter,snapchat,pinterest,news,shopping,shopee,lazada,tiki,sendo,twitch,kick.com,roblox,valorant,league of legends,liên minh,lien minh,lol,epic games,garena,minecraft,fortnite,free fire,pubg,genshin,honkai,zenless,dota,dota 2,counter-strike,counter strike,cs2,overwatch,battle.net",
    )
    _normalize_keyword_csv(
        "task_context_neutral_keywords",
        "explorer,settings,file",
    )
    _normalize_keyword_csv(
        "task_context_task_apps",
        "code.exe,cursor.exe,pycharm64.exe,devenv.exe,windowsterminal.exe,powershell.exe,cmd.exe,notion.exe,obsidian.exe,figma.exe,slack.exe,teams.exe,ms-teams.exe,zoom.exe,winword.exe,excel.exe,powerpnt.exe,outlook.exe",
    )
    _normalize_keyword_csv(
        "task_context_distracting_apps",
        "steam.exe,steamwebhelper.exe,epicgameslauncher.exe,epicgameslauncher,epicwebhelper.exe,riotclientservices.exe,riotclientux.exe,leagueclient.exe,leagueclientux.exe,league of legends.exe,valorant.exe,valorant-win64-shipping.exe,robloxplayerbeta.exe,minecraftlauncher.exe,fortniteclient-win64-shipping.exe,garena.exe,freefire.exe,pubg.exe,tslgame.exe,dota2.exe,cs2.exe,overwatch.exe,battle.net.exe,battlenet.exe,eaapp.exe,ubisoftconnect.exe,genshinimpact.exe,hoyoplay.exe,discord.exe,tiktok.exe",
    )
    _normalize_keyword_csv(
        "task_context_excluded_keywords",
        "focusguardian,notification",
    )
    _normalize_keyword_csv(
        "task_context_excluded_apps",
        "",
    )

    try:
        config["phone_detection_interval_frames"] = max(
            1,
            int(float(config.get("phone_detection_interval_frames", 4) or 4)),
        )
    except (TypeError, ValueError):
        config["phone_detection_interval_frames"] = 4

    try:
        config["phone_confirmation_window_seconds"] = max(
            0.8,
            float(config.get("phone_confirmation_window_seconds", 2.5) or 2.5),
        )
    except (TypeError, ValueError):
        config["phone_confirmation_window_seconds"] = 2.5

    try:
        config["phone_confirmation_min_hits"] = max(
            1,
            int(float(config.get("phone_confirmation_min_hits", 3) or 3)),
        )
    except (TypeError, ValueError):
        config["phone_confirmation_min_hits"] = 3

    try:
        phone_confidence_min = float(config.get("phone_confidence_min", 0.58) or 0.58)
    except (TypeError, ValueError):
        phone_confidence_min = 0.58
    config["phone_confidence_min"] = max(0.2, min(0.95, phone_confidence_min))

    try:
        vision_uncertain = float(config.get("vision_confidence_uncertain_threshold", 0.42) or 0.42)
    except (TypeError, ValueError):
        vision_uncertain = 0.42
    vision_uncertain = max(0.05, min(0.95, vision_uncertain))
    config["vision_confidence_uncertain_threshold"] = vision_uncertain

    try:
        vision_hard_floor = float(config.get("vision_confidence_hard_floor", 0.28) or 0.28)
    except (TypeError, ValueError):
        vision_hard_floor = 0.28
    config["vision_confidence_hard_floor"] = max(0.0, min(vision_uncertain, vision_hard_floor))

    if not isinstance(config.get("vision_calibration"), dict):
        config["vision_calibration"] = {}

    # Keep minimize button behavior consistent with taskbar apps.
    config["minimize_to_tray"] = bool(config.get("minimize_to_tray", False))

    return config


def save_config(config: dict):
    """Save configuration to file."""
    persistable = dict(config or {})
    for key in list(persistable.keys()):
        if str(key).startswith("_"):
            persistable.pop(key, None)

    if bool(persistable.get("enable_supabase_sync", False)):
        for key in PROFILE_SCOPED_CONFIG_KEYS:
            # Keep theme locally so startup/auth palette follows the last selected mode
            # even when cloud sync is temporarily unavailable.
            if key == "theme_mode":
                continue
            persistable.pop(key, None)

    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(persistable, f, indent=2, ensure_ascii=False)
        logger.info("Configuration saved to %s", CONFIG_FILE)
    except Exception as e:
        logger.error("Failed to save config: %s", e)


class FocusGuardianApp:
    """Main application class."""

    def __init__(self):
        # Create Qt application
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("FocusGuardian")
        self.app.setApplicationDisplayName("FocusGuardian")
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

        from app.logic.auth_manager import AuthManager

        self.auth_manager = AuthManager(self.config)
        self.auth_manager.configure(self.config)
        self._startup_cancelled = False
        self._auth_gate_debug_state = ""

        restored = self._restore_persisted_auth_session()
        if not restored and not self._run_auth_gate():
            self._startup_cancelled = True
            return

        self._apply_authenticated_session_to_config()
        save_config(self.config)

        # Setup
        self._init_main_window()
        # MainWindow may load profile-scoped theme from Supabase during init.
        # Re-apply app palette to keep popup/frame colors consistent at startup.
        self._apply_theme_palette(self.config.get("theme_mode", "dark"))
        self._init_tray()
        self._connect_signals()

    def _run_auth_gate(self) -> bool:
        """Show mandatory auth dialog before allowing access to the app."""
        from app.ui.auth_dialog import AuthDialog

        self.auth_manager.configure(self.config)
        dialog_parent = self.main_window if self.main_window is not None else None

        while True:
            dialog = AuthDialog(config=self.config, parent=dialog_parent, auth_manager=self.auth_manager)
            result = dialog.exec()

            is_authenticated = self.auth_manager.is_authenticated()
            accepted = result == int(QDialog.DialogCode.Accepted)
            exit_requested = bool(getattr(dialog, "exit_requested", False))

            self._auth_gate_debug_state = (
                f"dialog_result={result}, accepted={accepted}, authenticated={is_authenticated}"
            )

            if is_authenticated:
                if not accepted:
                    logger.warning(
                        "Auth dialog closed with Rejected code but session is authenticated; continuing startup"
                    )
                return True

            if accepted and not is_authenticated:
                logger.warning(
                    "Auth dialog returned Accepted but no authenticated session was found"
                )

            if exit_requested:
                return False

            logger.warning("Auth dialog was dismissed without authentication; reopening auth gate")

    def _apply_authenticated_session_to_config(self) -> None:
        """Persist authenticated identity fields into runtime config."""
        session = self.auth_manager.get_current_session()
        if session is None:
            self.config["auth_last_username"] = ""
            self.config["auth_last_profile_name"] = "default"
            self.config["auth_last_login_at_iso"] = ""
            self.config["profile_name"] = "default"
            self.config["auth_session_user_id"] = ""
            self.config["auth_session_username"] = ""
            self.config["auth_session_profile_name"] = ""
            self.config["auth_session_login_at"] = 0
            self.config["auth_session_login_at_iso"] = ""
            return

        self.config["auth_last_username"] = str(session.user.username or "")
        self.config["auth_last_profile_name"] = str(session.user.profile_name or "default")
        self.config["auth_last_login_at_iso"] = str(session.login_at_iso or "")
        self.config["profile_name"] = str(session.user.profile_name or "default")
        self.config["enable_personalization"] = True
        self.config["auto_apply_personalization"] = True

        if bool(self.config.get("auth_persist_session", True)):
            self.config["auth_session_user_id"] = str(session.user.user_id or "")
            self.config["auth_session_username"] = str(session.user.username or "")
            self.config["auth_session_profile_name"] = str(session.user.profile_name or "default")
            self.config["auth_session_login_at"] = int(session.login_at or 0)
            self.config["auth_session_login_at_iso"] = str(session.login_at_iso or "")

    def _restore_persisted_auth_session(self) -> bool:
        """Try restoring last authenticated session from local config."""
        if not bool(self.config.get("auth_persist_session", True)):
            return False

        cached_user_id = str(self.config.get("auth_session_user_id", "") or "").strip()
        cached_username = str(self.config.get("auth_session_username", "") or "").strip()
        cached_profile = str(self.config.get("auth_session_profile_name", "") or "").strip()
        cached_login_iso = str(self.config.get("auth_session_login_at_iso", "") or "").strip()
        try:
            cached_login_at = int(float(self.config.get("auth_session_login_at", 0) or 0))
        except (TypeError, ValueError):
            cached_login_at = 0

        if not cached_user_id and not cached_username:
            return False

        result = self.auth_manager.restore_cached_session(
            user_id=cached_user_id,
            username=cached_username,
            profile_name=cached_profile,
            login_at=cached_login_at,
            login_at_iso=cached_login_iso,
            verify_backend=True,
        )
        if not result.success:
            logger.info("Skip restoring cached session: %s", result.message)
            self.config["auth_session_user_id"] = ""
            self.config["auth_session_username"] = ""
            self.config["auth_session_profile_name"] = ""
            self.config["auth_session_login_at"] = 0
            self.config["auth_session_login_at_iso"] = ""
            return False

        session = result.session
        username = session.user.username if session is not None else cached_username
        logger.info("Restored cached login session for '%s'", username)
        return True

    def _apply_theme_palette(self, theme_mode: str):
        """Apply application palette according to dark/light theme."""
        self.app.setStyle("Fusion")

        from PyQt6.QtGui import QPalette, QColor

        is_dark = str(theme_mode or "dark").strip().lower() != "light"
        palette = QPalette()

        if is_dark:
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
        else:
            palette.setColor(QPalette.ColorRole.Window, QColor(246, 248, 252))
            palette.setColor(QPalette.ColorRole.WindowText, QColor(33, 42, 53))
            palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor(239, 244, 251))
            palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
            palette.setColor(QPalette.ColorRole.ToolTipText, QColor(33, 42, 53))
            palette.setColor(QPalette.ColorRole.Text, QColor(33, 42, 53))
            palette.setColor(QPalette.ColorRole.Button, QColor(233, 240, 248))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor(33, 42, 53))
            palette.setColor(QPalette.ColorRole.BrightText, QColor(176, 39, 39))
            palette.setColor(QPalette.ColorRole.Link, QColor(15, 122, 101))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(80, 164, 146))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

        self.app.setPalette(palette)

    def _init_main_window(self):
        """Initialize main window."""
        from app.ui.main_window import MainWindow
        self.main_window = MainWindow(config=self.config, auth_manager=self.auth_manager)

        # Connect config save on close
        self.main_window.destroyed.connect(self._on_window_closed)

    def _init_tray(self):
        """Initialize system tray."""
        from app.ui.tray import SystemTray
        self.tray = SystemTray(parent=self.main_window)
        self.tray.set_theme_mode(str(self.config.get("theme_mode", "dark")))

        if not self.tray.is_available():
            logger.warning("System tray not available")

    def _connect_signals(self):
        """Connect signals between components."""
        # Main window signals
        self.main_window.state_changed.connect(self._on_state_changed)
        self.main_window.break_suggested.connect(self._on_break_suggested)
        self.main_window.config_changed.connect(self._on_config_changed)
        self.main_window.logout_requested.connect(self._handle_logout_request)
        self.main_window.context_alert_requested.connect(self._on_context_alert_requested)

        # Tray signals
        self.tray.show_window_requested.connect(self._show_window)
        self.tray.quit_requested.connect(self._quit)
        self.tray.logout_requested.connect(self._handle_logout_request)
        self.tray.start_tracking_requested.connect(self._start_tracking)
        self.tray.stop_tracking_requested.connect(self._stop_tracking)
        self.tray.take_break_requested.connect(self._take_break)
        self.tray.open_settings_requested.connect(self._open_settings)

    def _handle_logout_request(self):
        """Logout current user and force re-auth before continuing."""
        if self.main_window and self.main_window.camera_running:
            self.main_window._stop_tracking()
        if self.main_window is not None:
            self.main_window.stop_focus_audio()

        self.auth_manager.logout()
        self._apply_authenticated_session_to_config()
        save_config(self.config)

        if self.main_window is not None:
            self.main_window.hide()
        if self.tray is not None:
            self.tray.set_tracking(False)

        if not self._run_auth_gate():
            self._quit()
            return

        self._apply_authenticated_session_to_config()
        save_config(self.config)

        if self.main_window is not None:
            self.main_window.refresh_authenticated_profile()
            self.main_window.restore_focus_audio_from_config()
            self._show_window()

    def _on_config_changed(self, config: dict):
        """Handle config updates emitted by the main window."""
        self.config.update(config)
        self._apply_theme_palette(self.config.get("theme_mode", "dark"))
        if self.tray is not None:
            self.tray.set_theme_mode(str(self.config.get("theme_mode", "dark")))
        save_config(self.config)

    def _on_state_changed(self, state):
        """Handle focus state change."""
        from app.logic.focus_engine import FocusState

        if self.tray is None:
            return

        self.tray.set_state(state)

        # Show notification for certain states
        if self.config.get("enable_notifications", True):
            if state == FocusState.PHONE_DISTRACTION and self.config.get("notify_distraction", True):
                self.tray.show_focus_notification(state)
            elif state == FocusState.DROWSY_FATIGUE and self.config.get("notify_drowsy", True):
                self.tray.show_focus_notification(state)

    def _on_break_suggested(self):
        """Handle break suggestion."""
        if self.main_window is None or self.tray is None:
            return
        if self.config.get("enable_notifications", True) and self.config.get("notify_break", True):
            break_minutes = int(self.main_window.config.get("break_duration_minutes", 5) or 5)
            self.tray.show_break_reminder(break_minutes=break_minutes)

    def _on_context_alert_requested(self, title: str, message: str):
        """Handle digital context nudges from active app/tab monitoring."""
        if self.tray is None:
            return
        if not self.config.get("enable_notifications", True):
            return
        if not self.config.get("notify_distraction", True):
            return
        try:
            from PyQt6.QtWidgets import QSystemTrayIcon

            icon = QSystemTrayIcon.MessageIcon.Warning
        except Exception:
            icon = None
        if icon is None:
            self.tray.show_notification(str(title or "Nhắc nhẹ"), str(message or "Quay lại nhiệm vụ chính nhé."))
        else:
            self.tray.show_notification(
                str(title or "Nhắc nhẹ"),
                str(message or "Quay lại nhiệm vụ chính nhé."),
                icon,
                5000,
            )

    def _show_window(self):
        """Show main window."""
        if self.main_window is None:
            return
        if self.main_window.isMinimized():
            self.main_window.setWindowState(
                self.main_window.windowState() & ~Qt.WindowState.WindowMinimized
            )
        self.main_window.show()
        self.main_window.activateWindow()
        self.main_window.raise_()

    def _ensure_startup_window_visible(self) -> None:
        """Force the main window visible after login unless start-minimized is enabled."""
        if self.main_window is None:
            return
        if bool(self.config.get("start_minimized", False)):
            return
        if self.main_window.isVisible():
            return
        self._show_window()

    def _start_tracking(self):
        """Start tracking from tray."""
        if self.main_window is None or self.tray is None:
            return
        self.main_window.btn_start.setChecked(True)
        self.main_window._toggle_tracking()
        self.tray.set_tracking(True)

    def _stop_tracking(self):
        """Stop tracking from tray."""
        if self.main_window is None or self.tray is None:
            return
        self.main_window.btn_start.setChecked(False)
        self.main_window._toggle_tracking()
        self.tray.set_tracking(False)

    def _take_break(self):
        """Take break from tray."""
        if self.main_window is None:
            return
        self.main_window._take_break()

    def _open_settings(self):
        """Open settings from tray."""
        if self.main_window is None:
            return
        self._show_window()
        self.main_window._open_settings()

    def _on_window_closed(self):
        """Handle window close."""
        # Save latest config from main window
        if self.main_window is not None:
            self.config.update(self.main_window.config)
        save_config(self.config)

    def _quit(self):
        """Quit application."""
        # Stop tracking if running
        if self.main_window is not None and self.main_window.camera_running:
            self.main_window._stop_tracking()
        if self.main_window is not None:
            self.main_window.stop_focus_audio()

        # Save latest config from main window
        if self.main_window is not None:
            self.config.update(self.main_window.config)
        save_config(self.config)

        # Hide tray
        if self.tray is not None:
            self.tray.hide()

        # Quit app
        self.app.quit()

    def run(self) -> int:
        """Run the application."""
        if self._startup_cancelled:
            logger.info(
                "Startup cancelled before authentication (%s)",
                self._auth_gate_debug_state or "auth gate returned false",
            )
            return 0

        logger.info("Starting FocusGuardian")

        try:
            # Show main window
            if self.main_window is not None:
                if bool(self.config.get("start_minimized", False)):
                    self.main_window.hide()
                else:
                    self._show_window()
                    QTimer.singleShot(220, self._ensure_startup_window_visible)

            # Run event loop
            exit_code = self.app.exec()
            logger.info("Qt event loop exited with code %s", exit_code)
            return exit_code

        except Exception as e:
            logger.exception("Application error: %s", e)
            from app.ui.notice_dialog import NoticeDialog

            NoticeDialog.error(
                None,
                "Lỗi",
                f"Đã xảy ra lỗi không mong muốn:\n{e}",
                config=self.config,
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
