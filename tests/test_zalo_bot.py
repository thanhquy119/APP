"""Tests for Zalo bot client helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.logic.zalo_bot import ZaloBotClient, ZaloBotConfig


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: Optional[Dict[str, Any]] = None,
        text: str = "",
        json_error: Optional[Exception] = None,
    ):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls = []

    def get(self, *args, **kwargs):
        self.calls.append(("get", args, kwargs))
        return self.response

    def post(self, *args, **kwargs):
        self.calls.append(("post", args, kwargs))
        return self.response


def _config(token: str = "token:abc") -> ZaloBotConfig:
    return ZaloBotConfig(enabled=True, bot_token=token, timeout_seconds=3.0)


def test_fetch_latest_chat_id_requires_token():
    client = ZaloBotClient(_config(token=""), session=FakeSession(FakeResponse()))
    success, message, chat_id, payload = client.fetch_latest_chat_id()

    assert success is False
    assert "Thiếu Bot Token" in message
    assert chat_id is None
    assert payload is None


def test_fetch_latest_chat_id_success_from_updates():
    response = FakeResponse(
        status_code=200,
        payload={
            "ok": True,
            "result": [
                {"message": {"chat": {"id": "12345"}}},
                {"message": {"chat": {"id": 67890}}},
            ],
        },
    )
    session = FakeSession(response)
    client = ZaloBotClient(_config(), session=session)

    success, message, chat_id, payload = client.fetch_latest_chat_id(limit=5)

    assert success is True
    assert chat_id == "67890"
    assert "thành công" in message.lower()
    assert isinstance(payload, dict)
    assert len(session.calls) == 1
    assert session.calls[0][0] == "post"


def test_fetch_latest_chat_id_success_from_result_dict_shape():
    response = FakeResponse(
        status_code=200,
        payload={
            "ok": True,
            "result": {
                "message": {
                    "chat": {
                        "id": "dict-chat-id",
                    }
                }
            },
        },
    )
    client = ZaloBotClient(_config(), session=FakeSession(response))

    success, message, chat_id, payload = client.fetch_latest_chat_id()

    assert success is True
    assert chat_id == "dict-chat-id"
    assert "thành công" in message.lower()
    assert isinstance(payload, dict)


def test_fetch_latest_chat_id_handles_webhook_conflict():
    response = FakeResponse(
        status_code=409,
        payload={"ok": False, "message": "Webhook is active"},
        text="Webhook is active",
    )
    client = ZaloBotClient(_config(), session=FakeSession(response))

    success, message, chat_id, _ = client.fetch_latest_chat_id()

    assert success is False
    assert chat_id is None
    assert "webhook" in message.lower()


def test_fetch_latest_chat_id_handles_invalid_json():
    response = FakeResponse(
        status_code=200,
        payload=None,
        json_error=ValueError("invalid json"),
    )
    client = ZaloBotClient(_config(), session=FakeSession(response))

    success, message, chat_id, _ = client.fetch_latest_chat_id()

    assert success is False
    assert chat_id is None
    assert "json" in message.lower()


def test_fetch_latest_chat_id_handles_empty_updates():
    response = FakeResponse(
        status_code=200,
        payload={"ok": True, "result": []},
    )
    client = ZaloBotClient(_config(), session=FakeSession(response))

    success, message, chat_id, _ = client.fetch_latest_chat_id()

    assert success is False
    assert chat_id is None
    assert "không có tin nhắn mới" in message.lower()
