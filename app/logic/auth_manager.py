"""Authentication manager coordinating validation, user store and session state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .auth import (
    CurrentUserSession,
    hash_password,
    is_valid_username,
    normalize_profile_name,
    normalize_username,
    now_ts,
    timestamp_to_iso,
    verify_password,
)
from .user_store import GoogleSheetsUserStore


@dataclass(slots=True)
class AuthResult:
    success: bool
    message: str
    session: Optional[CurrentUserSession] = None


class AuthManager:
    """High-level login/register/logout API for UI consumers."""

    def __init__(self, app_config: Optional[dict] = None, store: Optional[GoogleSheetsUserStore] = None):
        self._store = store or GoogleSheetsUserStore(app_config or {})
        self._current_session: Optional[CurrentUserSession] = None

    def configure(self, app_config: dict) -> None:
        self._store.configure_from_app_config(app_config or {})

    @property
    def availability_error(self) -> str:
        return self._store.availability_error

    def is_authenticated(self) -> bool:
        return self._current_session is not None

    def get_current_session(self) -> Optional[CurrentUserSession]:
        return self._current_session

    def restore_cached_session(
        self,
        *,
        user_id: str = "",
        username: str = "",
        profile_name: str = "",
        login_at: Optional[int] = None,
        login_at_iso: str = "",
    ) -> AuthResult:
        """Restore session from locally cached identity data.

        It prefers verifying with Google Sheets backend when available.
        If backend is unavailable, falls back to a local in-memory session.
        """
        key_user_id = str(user_id or "").strip().lower()
        key_username = normalize_username(username)
        if not key_user_id and not key_username:
            return AuthResult(False, "Thiếu dữ liệu phiên đăng nhập đã lưu")

        user = None
        backend_ok = self._store.ensure_available()
        if backend_ok:
            if key_user_id:
                user = self._store.find_by_user_id(key_user_id)
            if user is None and key_username:
                user = self._store.find_by_identity(key_username)
            if user is None:
                return AuthResult(False, "Không tìm thấy tài khoản đã lưu")
        else:
            effective_profile = normalize_profile_name(profile_name or key_username)
            ts = int(login_at or now_ts())
            ts_iso = str(login_at_iso or timestamp_to_iso(ts))
            user = self._store.create_user_account(
                username=key_username or "cached_user",
                password_hash="",
                profile_name=effective_profile,
            )
            user.user_id = key_user_id or user.user_id
            user.last_login_at = ts
            user.last_login_at_iso = ts_iso

        if user is None:
            return AuthResult(False, "Không thể khôi phục phiên đã lưu")
        if not user.is_active:
            return AuthResult(False, "Tài khoản đã bị vô hiệu hoá")

        ts = int(login_at or now_ts())
        ts_iso = str(login_at_iso or timestamp_to_iso(ts))
        session = CurrentUserSession(user=user, login_at=ts, login_at_iso=ts_iso)
        self._current_session = session
        return AuthResult(True, "Khôi phục phiên đăng nhập thành công", session=session)

    def get_effective_profile_name(self, fallback: str = "default") -> str:
        if self._current_session is None:
            return normalize_profile_name(fallback)
        return normalize_profile_name(self._current_session.user.profile_name or fallback)

    def check_backend(self) -> tuple[bool, str]:
        """Check whether Google Sheets user storage is reachable."""
        ok = self._store.ensure_available()
        if ok:
            return True, "OK"
        return False, self._store.availability_error or "Google Sheets backend unavailable"

    def register(
        self,
        *,
        username: str,
        password: str,
        confirm_password: str,
        profile_name: str = "",
    ) -> AuthResult:
        username_norm = normalize_username(username)

        if not self._store.ensure_available():
            return AuthResult(False, self._store.availability_error or "Google Sheets chưa sẵn sàng")

        if not is_valid_username(username_norm):
            return AuthResult(False, "Username phải dài 3-32 ký tự, chỉ gồm chữ/số/._-")
        if len(password or "") < 8:
            return AuthResult(False, "Mật khẩu cần tối thiểu 8 ký tự")
        if password != confirm_password:
            return AuthResult(False, "Mật khẩu xác nhận không khớp")

        if self._store.find_by_username(username_norm) is not None:
            return AuthResult(False, "Username đã tồn tại")

        hash_value = hash_password(password)
        effective_profile = normalize_profile_name(profile_name or username_norm)
        user = self._store.create_user_account(
            username=username_norm,
            password_hash=hash_value,
            profile_name=effective_profile,
        )

        ok, message = self._store.create_user(user)
        if not ok:
            return AuthResult(False, message)

        self._store.update_last_login(user.user_id, user.last_login_at)
        session = CurrentUserSession(user=user, login_at=user.last_login_at, login_at_iso=user.last_login_at_iso)
        self._current_session = session
        return AuthResult(True, "Đăng ký thành công", session=session)

    def login(self, *, username: str, password: str) -> AuthResult:
        key = normalize_username(username)
        if not key:
            return AuthResult(False, "Vui lòng nhập username")
        if not password:
            return AuthResult(False, "Vui lòng nhập mật khẩu")

        if not self._store.ensure_available():
            return AuthResult(False, self._store.availability_error or "Google Sheets chưa sẵn sàng")

        user = self._store.find_by_identity(key)
        if user is None:
            return AuthResult(False, "Sai username hoặc mật khẩu")
        if not user.is_active:
            return AuthResult(False, "Tài khoản đã bị vô hiệu hoá")
        if not verify_password(password, user.password_hash):
            return AuthResult(False, "Sai username hoặc mật khẩu")

        login_ts = now_ts()
        login_iso = timestamp_to_iso(login_ts)
        self._store.update_last_login(user.user_id, login_ts)
        user.last_login_at = login_ts
        user.last_login_at_iso = login_iso

        session = CurrentUserSession(user=user, login_at=login_ts, login_at_iso=login_iso)
        self._current_session = session
        return AuthResult(True, "Đăng nhập thành công", session=session)

    def logout(self) -> AuthResult:
        if self._current_session is None:
            return AuthResult(True, "Đã đăng xuất")

        self._current_session = None
        return AuthResult(True, "Đăng xuất thành công")
