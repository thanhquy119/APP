"""
Realtime Zalo alert manager for FocusGuardian.

Responsibilities:
- Receive stable focus state updates from UI/runtime
- Decide when to send alerts (short internal confirmation + episode lock)
- Format concise user-facing messages
- Prevent duplicate/spam notifications
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from .focus_engine import FocusState
from .zalo_bot import ZaloBotClient, ZaloBotConfig

logger = logging.getLogger(__name__)


@dataclass
class ZaloAlertEvent:
    """Represents one attempted outbound alert."""

    alert_key: str
    state_name: str
    message: str
    success: bool
    detail: str
    timestamp: float


class ZaloAlertManager:
    """Manage anti-spam Zalo outbound alerts from stable focus states."""

    FOCUSED_STATES = {
        FocusState.ON_SCREEN_READING,
        FocusState.OFFSCREEN_WRITING,
    }
    BAD_STATES = {
        FocusState.PHONE_DISTRACTION,
        FocusState.DROWSY_FATIGUE,
        FocusState.AWAY,
    }

    def __init__(
        self,
        app_config: Optional[Dict[str, Any]] = None,
        client: Optional[ZaloBotClient] = None,
    ):
        app_config = app_config or {}
        self._config: Dict[str, Any] = {}
        self._profile_name: str = "default"
        self._enabled: bool = False

        # Internal smoothing only; not exposed as a visible UI input.
        self._state_cooldown_seconds: float = 0.0
        self._break_cooldown_seconds: float = 120.0
        self._threshold_seconds: float = 1.2

        self._alert_on_distraction = True
        self._alert_on_drowsy = True
        self._alert_on_phone = True
        self._alert_on_away = True
        self._alert_on_break_reminder = True

        self._active_state: Optional[FocusState] = None
        self._active_state_started_at: float = 0.0
        self._sent_keys_in_active_streak: Set[str] = set()

        self._last_global_alert_at: float = 0.0
        self._last_alert_at_by_key: Dict[str, float] = {}
        self._recovered_since_last_alert: bool = True

        bot_config = ZaloBotConfig.from_app_config(app_config)
        self.client = client or ZaloBotClient(bot_config)
        self.configure(app_config)

    def configure(self, app_config: Dict[str, Any]) -> None:
        """Refresh manager and bot settings from app config."""
        self._config = dict(app_config or {})
        self._profile_name = str(self._config.get("profile_name", "default") or "").strip() or "default"

        self._enabled = bool(self._config.get("enable_zalo_alerts", False))
        legacy_threshold = float(self._config.get("zalo_alert_threshold_seconds", 1.2) or 1.2)
        self._threshold_seconds = max(
            0.6,
            float(self._config.get("zalo_distraction_confirm_seconds", legacy_threshold) or legacy_threshold),
        )
        legacy_cooldown_minutes = float(self._config.get("zalo_alert_cooldown_minutes", 0.0) or 0.0)
        legacy_cooldown_seconds = legacy_cooldown_minutes * 60.0 if legacy_cooldown_minutes > 0 else 120.0
        self._state_cooldown_seconds = max(
            0.0,
            float(self._config.get("zalo_state_cooldown_seconds", 0.0) or 0.0),
        )
        self._break_cooldown_seconds = max(
            0.0,
            float(self._config.get("zalo_break_cooldown_seconds", legacy_cooldown_seconds) or legacy_cooldown_seconds),
        )

        self._alert_on_distraction = bool(self._config.get("zalo_alert_on_distraction", True))
        self._alert_on_drowsy = bool(self._config.get("zalo_alert_on_drowsy", True))
        self._alert_on_phone = bool(self._config.get("zalo_alert_on_phone", True))
        self._alert_on_away = bool(self._config.get("zalo_alert_on_away", True))
        self._alert_on_break_reminder = bool(self._config.get("zalo_alert_on_break_reminder", True))

        self.client.update_config(ZaloBotConfig.from_app_config(self._config))

    def reset_session(self) -> None:
        """Reset streak/debounce state at the beginning of a tracking session."""
        self._active_state = None
        self._active_state_started_at = 0.0
        self._sent_keys_in_active_streak.clear()
        self._recovered_since_last_alert = True

    def mark_recovered(self) -> None:
        """Mark that user returned to a focused state."""
        self._active_state = None
        self._active_state_started_at = 0.0
        self._sent_keys_in_active_streak.clear()
        self._recovered_since_last_alert = True

    def handle_state_update(
        self,
        state: FocusState,
        *,
        score: float,
        confidence: float,
        reason: str,
        timestamp: Optional[float] = None,
        recommendation: Optional[Dict[str, Any]] = None,
        in_warmup: bool = False,
    ) -> Optional[ZaloAlertEvent]:
        """
        Process one stable state update and emit at most one alert event.

        This method should be fed with display/stabilized state from UI,
        not raw/noisy frame-level state.
        """
        now = float(timestamp if timestamp is not None else time.time())

        if not self._enabled or in_warmup:
            return None

        if state in self.FOCUSED_STATES:
            self.mark_recovered()
            return None

        if state == FocusState.UNCERTAIN:
            # Keep current episode; noisy UNCERTAIN blips should not reset anti-spam lock.
            return None

        if state not in self.BAD_STATES:
            return None

        # Start a new non-focused episode only when we previously recovered.
        if self._active_state is None or self._active_state in self.FOCUSED_STATES:
            self._active_state = state
            self._active_state_started_at = now
            self._sent_keys_in_active_streak.clear()

        active_duration = max(0.0, now - self._active_state_started_at)
        if active_duration < self._threshold_seconds:
            return None

        alert_key = self._resolve_alert_key_for_state(state)
        if not alert_key:
            return None

        # Exactly one alert per non-focused episode; send again only after focused recovery.
        if self._sent_keys_in_active_streak and not self._recovered_since_last_alert:
            return None

        message = self._format_alert_message(
            alert_key=alert_key,
            active_duration=active_duration,
            score=score,
            confidence=confidence,
            reason=reason,
            recommendation=recommendation or {},
            timestamp=now,
        )

        success, detail, _ = self.client.send_message(None, message)

        # Treat send attempt as consumed so failed network does not spam every frame.
        self._mark_alert_attempt(alert_key, now)

        if not success:
            logger.warning("Zalo state alert failed (%s): %s", alert_key, detail)

        return ZaloAlertEvent(
            alert_key=alert_key,
            state_name=state.name,
            message=message,
            success=success,
            detail=detail,
            timestamp=now,
        )

    def handle_break_reminder(
        self,
        *,
        focus_cycle_seconds: float,
        break_interval_seconds: float,
        recommendation: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[ZaloAlertEvent]:
        """Emit one break-reminder alert when break interval is reached."""
        now = float(timestamp if timestamp is not None else time.time())

        if not self._enabled or not self._alert_on_break_reminder:
            return None

        if break_interval_seconds <= 0:
            return None

        if float(focus_cycle_seconds) < float(break_interval_seconds):
            return None

        alert_key = "break_reminder"
        if not self._is_cooldown_ready(alert_key, now):
            return None

        message = self._format_break_message(
            focus_cycle_seconds=float(focus_cycle_seconds),
            recommendation=recommendation or {},
            timestamp=now,
        )

        success, detail, _ = self.client.send_message(None, message)
        self._mark_alert_attempt(alert_key, now)

        if not success:
            logger.warning("Zalo break reminder failed: %s", detail)

        return ZaloAlertEvent(
            alert_key=alert_key,
            state_name="BREAK_REMINDER",
            message=message,
            success=success,
            detail=detail,
            timestamp=now,
        )

    def _resolve_alert_key_for_state(self, state: FocusState) -> Optional[str]:
        if state == FocusState.PHONE_DISTRACTION:
            if self._alert_on_phone:
                return "phone"
            if self._alert_on_distraction:
                return "distraction"
            return None

        if state == FocusState.DROWSY_FATIGUE:
            if self._alert_on_drowsy:
                return "drowsy"
            if self._alert_on_distraction:
                return "distraction"
            return None

        if state == FocusState.AWAY:
            if self._alert_on_away:
                return "away"
            if self._alert_on_distraction:
                return "distraction"
            return None

        return None

    def _is_cooldown_ready(self, alert_key: str, now: float) -> bool:
        cooldown_seconds = (
            self._break_cooldown_seconds
            if alert_key == "break_reminder"
            else self._state_cooldown_seconds
        )

        if cooldown_seconds <= 0:
            return True

        if (now - self._last_global_alert_at) < cooldown_seconds:
            return False

        last_key_at = self._last_alert_at_by_key.get(alert_key, 0.0)
        return (now - last_key_at) >= cooldown_seconds

    def _mark_alert_attempt(self, alert_key: str, timestamp: float) -> None:
        self._sent_keys_in_active_streak.add(alert_key)
        self._last_global_alert_at = float(timestamp)
        self._last_alert_at_by_key[alert_key] = float(timestamp)
        self._recovered_since_last_alert = False

    def _format_alert_message(
        self,
        *,
        alert_key: str,
        active_duration: float,
        score: float,
        confidence: float,
        reason: str,
        recommendation: Dict[str, Any],
        timestamp: float,
    ) -> str:
        profile = self._profile_name
        time_text = time.strftime("%H:%M:%S %d/%m/%Y", time.localtime(timestamp))
        duration_text = self._format_duration(active_duration)
        safe_score = max(0.0, min(100.0, float(score or 0.0)))
        safe_conf = max(0.0, min(1.0, float(confidence or 0.0)))

        base_map = {
            "distraction": "FocusGuardian: Phát hiện bạn đang mất tập trung.",
            "drowsy": "FocusGuardian: Phát hiện dấu hiệu buồn ngủ. Nên nghỉ ngắn 3-5 phút.",
            "phone": "FocusGuardian: Có dấu hiệu đang dùng điện thoại trong giờ làm việc.",
            "away": "FocusGuardian: Bạn đã rời khỏi bàn quá lâu.",
        }

        headline = base_map.get(alert_key, "FocusGuardian: Phát hiện trạng thái cần chú ý.")

        recommendation_text = self._short_recommendation(recommendation)
        reason_text = str(reason or "").strip()

        lines = [
            headline,
            f"Hồ sơ: {profile}",
            f"Thời gian trạng thái: {duration_text}",
            f"Điểm hiện tại: {safe_score:.0f} | Độ tin cậy: {safe_conf:.0%}",
            f"Thời điểm: {time_text}",
        ]

        if recommendation_text:
            lines.append(f"Gợi ý: {recommendation_text}")

        if reason_text:
            lines.append(f"Chi tiết: {reason_text[:120]}")

        return "\n".join(lines)

    def _format_break_message(
        self,
        *,
        focus_cycle_seconds: float,
        recommendation: Dict[str, Any],
        timestamp: float,
    ) -> str:
        profile = self._profile_name
        time_text = time.strftime("%H:%M:%S %d/%m/%Y", time.localtime(timestamp))
        duration_text = self._format_duration(focus_cycle_seconds)
        recommendation_text = self._short_recommendation(recommendation)

        lines = [
            "FocusGuardian: Đã đến lúc nghỉ giải lao theo nhịp cá nhân hóa.",
            f"Hồ sơ: {profile}",
            f"Đã tập trung liên tục: {duration_text}",
            f"Thời điểm: {time_text}",
        ]

        if recommendation_text:
            lines.append(f"Gợi ý: {recommendation_text}")

        return "\n".join(lines)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        safe_seconds = max(0.0, float(seconds or 0.0))
        if safe_seconds >= 120:
            mins = int(round(safe_seconds / 60.0))
            return f"{mins} phút"
        return f"{int(round(safe_seconds))} giây"

    @staticmethod
    def _short_recommendation(recommendation: Dict[str, Any]) -> str:
        if not isinstance(recommendation, dict):
            return ""

        reason = str(recommendation.get("reason", "") or "").strip()
        if reason:
            return reason[:120]

        work = recommendation.get("work_minutes")
        rest = recommendation.get("break_minutes")
        try:
            if work is not None and rest is not None:
                return f"Làm việc {int(work)} phút, nghỉ {int(rest)} phút."
        except (TypeError, ValueError):
            pass

        return ""
