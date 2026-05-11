"""Supabase-backed user account storage."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from .auth import UserAccount, now_ts, timestamp_to_iso
from .supabase_sync import (
    SupabaseConfig,
    SupabaseRestClient,
    supabase_config_from_app_config,
    supabase_missing_config_message,
)

logger = logging.getLogger(__name__)


class SupabaseUserStore:
    """Store user accounts in the configured Supabase users table."""

    USERS_HEADER = [
        "user_id",
        "username",
        "password_hash",
        "created_at",
        "created_at_iso",
        "last_login_at",
        "last_login_at_iso",
        "is_active",
        "profile_name",
    ]

    def __init__(self, app_config: Optional[dict[str, Any]] = None):
        self.config = SupabaseConfig()
        self._client: Optional[SupabaseRestClient] = None
        self._availability_error = ""

        if app_config is not None:
            self.configure_from_app_config(app_config)

    def configure_from_app_config(self, app_config: dict[str, Any]) -> None:
        new_config = supabase_config_from_app_config(app_config or {})
        if new_config != self.config:
            self._client = None
            self._availability_error = ""
        self.config = new_config

    @property
    def availability_error(self) -> str:
        return self._availability_error

    def ensure_available(self) -> bool:
        missing_config = supabase_missing_config_message(self.config)
        if missing_config:
            self._availability_error = missing_config
            return False

        client = self._get_client()
        if client is None:
            self._availability_error = "Khong khoi tao duoc Supabase client"
            return False

        if not client.health_check(self.config.users_table_name):
            self._availability_error = "Khong ket noi duoc Supabase"
            return False

        self._availability_error = ""
        return True

    def find_by_identity(self, identity: str) -> Optional[UserAccount]:
        key = str(identity or "").strip().lower()
        if not key:
            return None

        for user in self._load_all_users():
            if user.username.lower() == key:
                return user
        return None

    def find_by_username(self, username: str) -> Optional[UserAccount]:
        return self.find_by_identity(username)

    def find_by_user_id(self, user_id: str) -> Optional[UserAccount]:
        key = str(user_id or "").strip().lower()
        if not key:
            return None

        client = self._get_client()
        if client is None:
            return None

        rows = client.select(
            self.config.users_table_name,
            filters={"user_id": f"eq.{key}"},
            limit=1,
        )
        if not rows:
            return None

        try:
            return UserAccount.from_record(rows[0])
        except Exception as exc:
            logger.debug("Skip malformed Supabase user row: %s", exc)
            return None

    def create_user(self, user: UserAccount) -> tuple[bool, str]:
        if not self.ensure_available():
            return False, self._availability_error or "Supabase chua san sang"

        if self.find_by_username(user.username) is not None:
            return False, "Username da ton tai"

        client = self._get_client()
        if client is None:
            return False, "Khong khoi tao duoc Supabase client"

        record = user.to_record()
        record["raw_payload"] = user.to_record()
        ok = client.insert(self.config.users_table_name, record)
        if not ok:
            self._availability_error = "Khong the ghi du lieu user len Supabase"
            return False, self._availability_error
        return True, "Dang ky thanh cong"

    def update_last_login(self, user_id: str, timestamp: Optional[int] = None) -> bool:
        client = self._get_client()
        if client is None:
            return False

        key = str(user_id or "").strip()
        if not key:
            return False

        ts = int(timestamp or now_ts())
        ok = client.update(
            self.config.users_table_name,
            {
                "last_login_at": ts,
                "last_login_at_iso": timestamp_to_iso(ts),
            },
            filters={"user_id": f"eq.{key}"},
        )
        if not ok:
            logger.debug("Failed to update Supabase last_login_at for user '%s'", key)
        return ok

    def create_user_account(
        self,
        *,
        username: str,
        password_hash: str,
        profile_name: str,
    ) -> UserAccount:
        created_at = now_ts()
        created_at_iso = timestamp_to_iso(created_at)
        user_id = uuid.uuid4().hex
        return UserAccount(
            user_id=user_id,
            username=username,
            password_hash=password_hash,
            created_at=created_at,
            created_at_iso=created_at_iso,
            last_login_at=created_at,
            last_login_at_iso=created_at_iso,
            is_active=True,
            profile_name=profile_name,
        )

    def _load_all_users(self) -> list[UserAccount]:
        client = self._get_client()
        if client is None:
            return []

        rows = client.select(self.config.users_table_name) or []
        users: list[UserAccount] = []
        for row in rows:
            try:
                users.append(UserAccount.from_record(row))
            except Exception as exc:
                logger.debug("Skip malformed Supabase user row: %s", exc)
        return users

    def _get_client(self) -> Optional[SupabaseRestClient]:
        if not self.config.enabled or not self.config.url or not self.config.api_key:
            return None
        if self._client is None:
            self._client = SupabaseRestClient(self.config)
        return self._client
