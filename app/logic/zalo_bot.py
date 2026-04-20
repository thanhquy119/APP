"""
Zalo Bot Platform integration helpers.

This module provides a thin outbound client that follows the Zalo Bot API style:
https://bot-api.zaloplatforms.com/bot${BOT_TOKEN}/functionName
"""

from __future__ import annotations

import hmac
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import requests
    from requests.exceptions import RequestException, Timeout
except Exception:  # pragma: no cover - defensive fallback when requests is missing.
    requests = None

    class RequestException(Exception):
        """Fallback request exception type when requests is unavailable."""

    class Timeout(RequestException):
        """Fallback timeout exception type when requests is unavailable."""


logger = logging.getLogger(__name__)

ZALO_BOT_API_BASE = "https://bot-api.zaloplatforms.com"
FIXED_ZALO_BOT_TOKEN = "4093410856336467151:JwrzJqRefYCNOvhvpzXcfBPRzhwnofFIiiDcxYAncRQQNsSsAyHfILNJCMHzFvQQ"
ZALO_BOT_OA_ID = "4093410856336467151"
ZALO_BOT_OA_LINK = f"https://zalo.me/{ZALO_BOT_OA_ID}"


@dataclass
class ZaloBotConfig:
    """Runtime configuration for Zalo Bot outbound messaging."""

    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    webhook_secret_token: str = ""
    timeout_seconds: float = 25.0

    @classmethod
    def from_app_config(cls, app_config: Optional[Dict[str, Any]]) -> "ZaloBotConfig":
        app_config = app_config or {}

        timeout_raw = app_config.get("zalo_api_timeout_seconds", 25.0)
        try:
            timeout_seconds = float(timeout_raw)
        except (TypeError, ValueError):
            timeout_seconds = 25.0

        configured_token = str(app_config.get("zalo_bot_token", "") or "").strip()
        if configured_token and configured_token != FIXED_ZALO_BOT_TOKEN:
            logger.warning("Configured Zalo token differs from fixed app token; using fixed token")

        return cls(
            enabled=bool(app_config.get("enable_zalo_alerts", False)),
            bot_token=FIXED_ZALO_BOT_TOKEN,
            chat_id=str(app_config.get("zalo_chat_id", "") or "").strip(),
            webhook_secret_token=str(app_config.get("zalo_webhook_secret", "") or "").strip(),
            timeout_seconds=max(2.0, min(30.0, timeout_seconds)),
        )


class ZaloBotClient:
    """HTTP client for Zalo Bot Platform outbound actions."""

    def __init__(self, config: Optional[ZaloBotConfig] = None, session: Optional[Any] = None):
        self.config = config or ZaloBotConfig()
        self._session = session

    def update_config(self, config: ZaloBotConfig) -> None:
        """Update runtime bot configuration."""
        self.config = config

    def validate_config(self, require_chat_id: bool = True) -> Tuple[bool, str]:
        """Validate local configuration before calling Zalo API."""
        if requests is None:
            return False, "Thiếu thư viện requests để gọi Zalo API"

        token = str(self.config.bot_token or "").strip()
        if not token:
            return False, "Thiếu Bot Token"

        if require_chat_id:
            chat_id = str(self.config.chat_id or "").strip()
            if not chat_id:
                return False, "Thiếu chat_id nhận cảnh báo"

        return True, "Cấu hình hợp lệ"

    def send_message(
        self,
        chat_id: Optional[str],
        text: str,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Send message with Zalo API sendMessage endpoint.

        Endpoint format:
        POST https://bot-api.zaloplatforms.com/bot${BOT_TOKEN}/sendMessage
        payload: {"chat_id": "...", "text": "..."}
        """
        resolved_chat_id = str(chat_id or self.config.chat_id or "").strip()
        message_text = str(text or "").strip()

        if not message_text:
            return False, "Nội dung tin nhắn rỗng", None

        ok, message = self.validate_config(require_chat_id=False)
        if not ok:
            return False, message, None

        if not resolved_chat_id:
            return False, "Thiếu chat_id để gửi tin nhắn", None

        url = self._build_endpoint("sendMessage")
        payload = {
            "chat_id": resolved_chat_id,
            "text": message_text,
        }

        try:
            response = self._get_session().post(
                url,
                json=payload,
                timeout=float(self.config.timeout_seconds),
            )
        except RequestException as exc:
            logger.warning("Zalo sendMessage network error: %s", exc)
            return False, f"Lỗi mạng khi gọi Zalo API: {exc}", None
        except Exception as exc:  # pragma: no cover - defensive fallback.
            logger.warning("Zalo sendMessage unexpected error: %s", exc)
            return False, f"Lỗi không xác định khi gọi Zalo API: {exc}", None

        return self._parse_send_message_response(response)

    def test_connection(self, chat_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate configuration and send a short health-check message.

        This method intentionally uses sendMessage because Zalo Bot API
        integration docs guarantee this endpoint for outbound messaging.
        """
        resolved_chat_id = str(chat_id or self.config.chat_id or "").strip()

        ok, message = self.validate_config(require_chat_id=False)
        if not ok:
            return False, message

        if not resolved_chat_id:
            return False, "Thiếu chat_id để kiểm tra kết nối"

        success, detail, _ = self.send_message(
            resolved_chat_id,
            "FocusGuardian: Kết nối Zalo bot thành công.",
        )
        if success:
            return True, "Kết nối Zalo bot thành công"
        return False, detail

    def fetch_latest_chat_id(
        self,
        limit: int = 20,
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[str], Optional[Dict[str, Any]]]:
        """
        Fetch the latest chat_id from getUpdates endpoint.

        This helper is intended for local/dev flows where users:
        1) open Zalo OA via QR/link
        2) send any message to the bot (e.g. 'Hello')
        3) app polls getUpdates to resolve chat_id automatically

        Note: getUpdates may fail when webhook mode is active.
        """
        ok, message = self.validate_config(require_chat_id=False)
        if not ok:
            return False, message, None, None

        url = self._build_endpoint("getUpdates")
        safe_limit = max(1, min(100, int(limit)))
        request_timeout = float(
            timeout_seconds if timeout_seconds is not None else self.config.timeout_seconds
        )
        max_attempts = 3
        retry_delay_seconds = 1.2
        response = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._get_session().post(
                    url,
                    json={"limit": safe_limit},
                    timeout=request_timeout,
                )
                break
            except Timeout as exc:
                logger.warning(
                    "Zalo getUpdates timeout on attempt %s/%s: %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds)
                    continue
                return False, self._build_timeout_guidance_message(), None, None
            except RequestException as exc:
                logger.warning(
                    "Zalo getUpdates network error on attempt %s/%s: %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds)
                    continue
                return False, self._build_network_guidance_message(), None, None
            except Exception as exc:  # pragma: no cover - defensive fallback.
                logger.warning("Zalo getUpdates unexpected error: %s", exc)
                return (
                    False,
                    "Không thể lấy chat_id do lỗi không xác định. Vui lòng thử lại sau.",
                    None,
                    None,
                )

        if response is None:
            return False, self._build_network_guidance_message(), None, None

        body = self._safe_json(response)

        if response.status_code >= 400:
            detail = self._extract_error_detail(response, body)
            if self._is_webhook_conflict_error(response.status_code, detail):
                return (
                    False,
                    "Không thể dùng getUpdates vì bot đang bật webhook. Hãy tắt webhook hoặc dùng webhook payload để lấy chat_id.",
                    None,
                    body,
                )
            if response.status_code in (401, 403):
                return (
                    False,
                    "Bot Token không hợp lệ hoặc bot không có quyền gọi getUpdates. Vui lòng kiểm tra lại token.",
                    None,
                    body,
                )
            return (
                False,
                f"Zalo API trả về lỗi {response.status_code}. Vui lòng thử lại sau hoặc kiểm tra cấu hình bot.",
                None,
                body,
            )

        if body is None:
            return False, "Zalo API trả về dữ liệu không phải JSON hợp lệ", None, None

        ok_field = body.get("ok")
        err_field = body.get("error")
        if ok_field is False or (
            isinstance(err_field, (int, str))
            and str(err_field) not in ("", "0", "false", "False")
        ):
            detail = str(body.get("message") or err_field or "Unknown error").strip()
            if self._is_webhook_conflict_error(response.status_code, detail):
                return (
                    False,
                    "Không thể dùng getUpdates vì bot đang bật webhook. Hãy tắt webhook hoặc dùng webhook payload để lấy chat_id.",
                    None,
                    body,
                )
            lowered_detail = detail.lower()
            if "timeout" in lowered_detail:
                return False, self._build_timeout_guidance_message(), None, body
            if "network" in lowered_detail or "connection" in lowered_detail:
                return False, self._build_network_guidance_message(), None, body
            return (
                False,
                f"Zalo API phản hồi không thành công: {detail}",
                None,
                body,
            )

        result = body.get("result")
        if isinstance(result, list) and not result:
            return (
                False,
                "Không có tin nhắn mới. Hãy nhắn cho bot một tin bất kỳ (ví dụ: hello) rồi thử lại.",
                None,
                body,
            )

        chat_id = self._extract_chat_id_from_update(body)
        if chat_id:
            return True, f"Đã lấy chat_id thành công: {chat_id}", chat_id, body

        logger.warning(
            "Zalo getUpdates JSON parsed but chat_id was not found. status=%s body=%s",
            response.status_code,
            self._truncate_for_log(body),
        )

        return (
            False,
            "Zalo API trả về JSON nhưng không chứa message.chat.id. Hãy nhắn trực tiếp cho bot rồi thử lại.",
            None,
            body,
        )

    @staticmethod
    def _build_timeout_guidance_message() -> str:
        return (
            "API Zalo phản hồi chậm hoặc bị timeout. "
            "Hãy thử lại sau, tăng timeout, hoặc kiểm tra webhook. "
            "Bạn cũng nên nhắn bot một tin mới rồi thử lại."
        )

    @staticmethod
    def _build_network_guidance_message() -> str:
        return (
            "Không thể kết nối tới API Zalo do lỗi mạng. "
            "Hãy kiểm tra kết nối Internet/VPN/firewall, nhắn bot một tin mới rồi thử lại, "
            "và kiểm tra bot có đang bật webhook hay không."
        )

    def verify_webhook_secret(self, received_header: Optional[str]) -> bool:
        """
        Verify webhook header token for optional inbound event handling.

        Expected header key from Zalo docs:
        X-Bot-Api-Secret-Token
        """
        expected = str(self.config.webhook_secret_token or "").strip()
        if not expected:
            return True

        incoming = str(received_header or "").strip()
        if not incoming:
            return False

        return hmac.compare_digest(expected, incoming)

    def validate_webhook_event(
        self,
        headers: Dict[str, Any],
        payload: Any,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Minimal webhook validation helper.

        This method is intentionally lightweight because FocusGuardian currently
        focuses on outbound alerting, not running a full webhook server.
        """
        header_token = None
        if isinstance(headers, dict):
            header_token = headers.get("X-Bot-Api-Secret-Token")

        if not self.verify_webhook_secret(header_token):
            return False, "Webhook secret token không hợp lệ", None

        if payload is None:
            return False, "Webhook payload rỗng", None

        if isinstance(payload, dict):
            return True, "Webhook event hợp lệ", payload

        return False, "Webhook payload không phải JSON object", None

    def _build_endpoint(self, function_name: str) -> str:
        token = str(self.config.bot_token or "").strip()
        func = str(function_name or "").strip()
        return f"{ZALO_BOT_API_BASE}/bot{token}/{func}"

    def _get_session(self) -> Any:
        if self._session is None:
            if requests is None:
                raise RuntimeError("requests is not installed")
            self._session = requests.Session()
        return self._session

    @staticmethod
    def _safe_json(response: Any) -> Optional[Dict[str, Any]]:
        try:
            parsed = response.json()
        except Exception:
            return None

        if isinstance(parsed, dict):
            return parsed
        return None

    def _parse_send_message_response(
        self,
        response: Any,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        body = self._safe_json(response)

        if response.status_code >= 400:
            detail = self._extract_error_detail(response, body)
            message = f"Zalo API trả về lỗi {response.status_code}: {detail}"
            logger.warning(message)
            return False, message, body

        if body is not None:
            # Some APIs expose 'error' or 'ok'. Handle both patterns safely.
            ok_field = body.get("ok")
            err_field = body.get("error")
            if ok_field is False or (isinstance(err_field, (int, str)) and str(err_field) not in ("", "0", "false", "False")):
                detail = str(body.get("message") or err_field or "Unknown error").strip()
                message = f"Zalo API phản hồi không thành công: {detail}"
                logger.warning(message)
                return False, message, body

        return True, "Gửi tin nhắn Zalo thành công", body

    @classmethod
    def _extract_chat_id_from_update(cls, update: Any, depth: int = 0) -> Optional[str]:
        if depth > 8:
            return None

        known_paths = (
            ("message", "chat", "id"),
            ("result", "message", "chat", "id"),
            ("result", "chat", "id"),
            ("chat", "id"),
            ("data", "message", "chat", "id"),
            ("data", "result", "message", "chat", "id"),
        )

        for path in known_paths:
            chat_id = cls._normalize_chat_id(cls._get_nested_value(update, path))
            if chat_id:
                return chat_id

        if isinstance(update, list):
            for item in reversed(update):
                chat_id = cls._extract_chat_id_from_update(item, depth + 1)
                if chat_id:
                    return chat_id
            return None

        if not isinstance(update, dict):
            return None

        for key in ("result", "message", "data", "payload", "update", "updates"):
            if key in update:
                chat_id = cls._extract_chat_id_from_update(update.get(key), depth + 1)
                if chat_id:
                    return chat_id

        for value in update.values():
            chat_id = cls._extract_chat_id_from_update(value, depth + 1)
            if chat_id:
                return chat_id

        return None

    @staticmethod
    def _get_nested_value(node: Any, path: Tuple[str, ...]) -> Any:
        current = node
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if current is None:
                return None
        return current

    @staticmethod
    def _normalize_chat_id(chat_id: Any) -> Optional[str]:
        if chat_id is None:
            return None
        try:
            value = str(chat_id).strip()
        except Exception:
            return None
        return value or None

    @staticmethod
    def _truncate_for_log(value: Any, max_chars: int = 1600) -> str:
        try:
            text = str(value)
        except Exception:
            text = repr(value)
        if len(text) > max_chars:
            return f"{text[:max_chars]}...(truncated)"
        return text

    @staticmethod
    def _extract_error_detail(response: Any, body: Optional[Dict[str, Any]]) -> str:
        detail = ""
        if body:
            detail = str(
                body.get("description")
                or body.get("message")
                or body.get("error")
                or ""
            ).strip()
        if not detail:
            detail = response.text.strip()[:250] if getattr(response, "text", "") else "HTTP error"
        return detail

    @staticmethod
    def _is_webhook_conflict_error(status_code: int, detail: str) -> bool:
        if int(status_code) == 409:
            return True

        lowered = str(detail or "").lower()
        return "webhook" in lowered and (
            "conflict" in lowered
            or "set webhook" in lowered
            or "setwebhook" in lowered
            or "active" in lowered
        )
