"""Tests for Zalo realtime alert manager anti-spam behavior."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from app.logic.focus_engine import FocusState
from app.logic.zalo_alerts import ZaloAlertManager


class FakeZaloClient:
    """Simple fake bot client for deterministic alert-manager tests."""

    def __init__(self):
        self.messages: list[tuple[Optional[str], str]] = []
        self.last_config: Dict[str, Any] = {}

    def update_config(self, config) -> None:
        self.last_config = {
            "enabled": getattr(config, "enabled", False),
            "bot_token": getattr(config, "bot_token", ""),
            "chat_id": getattr(config, "chat_id", ""),
        }

    def send_message(self, chat_id: Optional[str], text: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        self.messages.append((chat_id, text))
        return True, "ok", {"ok": True}


def _base_config() -> Dict[str, Any]:
    return {
        "enable_zalo_alerts": True,
        "zalo_bot_token": "token:abc",
        "zalo_chat_id": "chat-1",
        "zalo_distraction_confirm_seconds": 10,
        "zalo_break_cooldown_seconds": 60,
        "zalo_alert_on_distraction": True,
        "zalo_alert_on_drowsy": True,
        "zalo_alert_on_phone": True,
        "zalo_alert_on_away": True,
        "zalo_alert_on_break_reminder": True,
        "profile_name": "tester",
    }


def test_state_alert_waits_for_threshold_and_sends_once_per_streak():
    client = FakeZaloClient()
    manager = ZaloAlertManager(_base_config(), client=client)

    assert manager.handle_state_update(
        FocusState.PHONE_DISTRACTION,
        score=60.0,
        confidence=0.9,
        reason="phone",
        timestamp=100.0,
        recommendation={},
        in_warmup=False,
    ) is None

    first = manager.handle_state_update(
        FocusState.PHONE_DISTRACTION,
        score=58.0,
        confidence=0.91,
        reason="phone",
        timestamp=112.5,
        recommendation={},
        in_warmup=False,
    )
    assert first is not None
    assert first.success is True
    assert first.alert_key == "phone"

    repeat_same_streak = manager.handle_state_update(
        FocusState.PHONE_DISTRACTION,
        score=55.0,
        confidence=0.95,
        reason="phone",
        timestamp=130.0,
        recommendation={},
        in_warmup=False,
    )
    assert repeat_same_streak is None
    assert len(client.messages) == 1


def test_state_alert_requires_recovery_for_resend():
    client = FakeZaloClient()
    manager = ZaloAlertManager(_base_config(), client=client)

    manager.handle_state_update(
        FocusState.DROWSY_FATIGUE,
        score=49.0,
        confidence=0.88,
        reason="drowsy",
        timestamp=200.0,
        recommendation={},
        in_warmup=False,
    )
    first = manager.handle_state_update(
        FocusState.DROWSY_FATIGUE,
        score=47.0,
        confidence=0.9,
        reason="drowsy",
        timestamp=211.0,
        recommendation={},
        in_warmup=False,
    )
    assert first is not None

    # Recovery starts a new episode and allows a new send after short confirmation.
    manager.handle_state_update(
        FocusState.ON_SCREEN_READING,
        score=80.0,
        confidence=0.8,
        reason="focused",
        timestamp=220.0,
        recommendation={},
        in_warmup=False,
    )
    manager.handle_state_update(
        FocusState.DROWSY_FATIGUE,
        score=50.0,
        confidence=0.88,
        reason="drowsy",
        timestamp=225.0,
        recommendation={},
        in_warmup=False,
    )
    second_episode = manager.handle_state_update(
        FocusState.DROWSY_FATIGUE,
        score=48.0,
        confidence=0.9,
        reason="drowsy",
        timestamp=236.0,
        recommendation={},
        in_warmup=False,
    )
    assert second_episode is not None
    assert second_episode.alert_key == "drowsy"

    # Within the same non-focused episode, it should not resend repeatedly.
    repeated_same_episode = manager.handle_state_update(
        FocusState.DROWSY_FATIGUE,
        score=45.0,
        confidence=0.92,
        reason="drowsy",
        timestamp=275.0,
        recommendation={},
        in_warmup=False,
    )
    assert repeated_same_episode is None
    assert len(client.messages) == 2


def test_uncertain_state_never_sends_alert():
    client = FakeZaloClient()
    manager = ZaloAlertManager(_base_config(), client=client)

    event = manager.handle_state_update(
        FocusState.UNCERTAIN,
        score=70.0,
        confidence=0.3,
        reason="noise",
        timestamp=500.0,
        recommendation={},
        in_warmup=False,
    )

    assert event is None
    assert client.messages == []


def test_break_alert_obeys_cooldown():
    client = FakeZaloClient()
    manager = ZaloAlertManager(_base_config(), client=client)

    first = manager.handle_break_reminder(
        focus_cycle_seconds=1800.0,
        break_interval_seconds=1500.0,
        recommendation={"work_minutes": 25, "break_minutes": 5},
        timestamp=1000.0,
    )
    assert first is not None
    assert first.alert_key == "break_reminder"

    blocked = manager.handle_break_reminder(
        focus_cycle_seconds=1850.0,
        break_interval_seconds=1500.0,
        recommendation={},
        timestamp=1040.0,
    )
    assert blocked is None

    resent = manager.handle_break_reminder(
        focus_cycle_seconds=1920.0,
        break_interval_seconds=1500.0,
        recommendation={},
        timestamp=1075.0,
    )
    assert resent is not None
    assert len(client.messages) == 2
